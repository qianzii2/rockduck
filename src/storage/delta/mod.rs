//! Delta store module - three-layer incremental storage for HTAP workloads.
//!
//! # Architecture
//!
//! Write path: DeltaCell -> L1 DeltaMemStore (ping-pong) -> flush -> L2 DeltaL2Disk
//!              compact -> L3 DeltaL3Frozen
//!
//! Read path: query(snapshot_txn) -> K-Way Merge(L1, L2, L3) -> DeltaCell
//!
//! Layers:
//! - **L1**: In-memory ping-pong BTreeMap with ArcSwap, flushes to disk when threshold reached
//! - **L2**: Guard-indexed column files on disk, ZoneMap-backed patch index
//! - **L3**: Compacted frozen patches, merged into base column files
//!
//! References:
//! - ClickHouse Lightweight UPDATE (patch parts): <https://clickhouse.com/blog/updates-in-clickhouse-2-sql-style-updates>
//! - Delta Lake Deletion Vectors: <https://delta.io/blog/2023-07-05-deletion-vectors/>
//! - Iceberg V3 Deletion Vectors: <https://iceberglakehouse.com/iceberg/iceberg-spec-v3/>
//! - SILK (ATC 2019): preventing background compaction from hurting foreground queries
//! - EcoTune (SIGMOD 2025): rethinking compaction policies in LSM-trees

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;
use parking_lot::RwLock;

use crate::error::Result;

pub mod disk_store;
pub mod flush_engine;
pub mod frozen_store;
pub mod mem_store;
pub mod merge;
pub mod sparsity;
pub mod types;

// Re-export types for convenience — explicitly list all public items from types.rs
pub use disk_store::{
    patch_row_positions, ColumnMergePlan, DeltaL2Disk, Guard, GuardKey, MergePlan,
};
pub use flush_engine::{
    CompactionAction, CompactionDecision, CompactionPolicy, CompactionPriority, EcoTuneSelector,
    FlushEngine, ForegroundMonitor, WorkloadProfile,
};
pub use frozen_store::DeltaL3Frozen;
pub use mem_store::{ColumnClass, ColumnHeatTracker, DeltaMemStore, FlushPolicy};
pub use merge::{apply_deltas_to_batch, apply_sparse_patch, k_way_merge, DeltaMerger};
pub use sparsity::SparsitySelector;
pub use types::{DeltaCell, DeltaCheckpointState, DeltaPatch, DeltaPatchFormat, ZoneMap};

// Re-export WalOp from write module for L1 recovery
pub use crate::write::wal_recovery::WalOp;

/// Configuration for the delta store layer.
#[derive(Debug, Clone)]
pub struct DeltaConfig {
    /// L1 flush threshold in bytes. Default: 64MB.
    pub l1_flush_threshold: usize,
    /// L2 patch count threshold to trigger compaction. Default: 64.
    pub l2_compaction_threshold: usize,
    /// Sync level for L1 writes. Default: Batch(100ms).
    pub sync_level: SyncLevel,
    /// Data directory for delta files.
    pub data_dir: std::path::PathBuf,
}

impl Default for DeltaConfig {
    fn default() -> Self {
        Self {
            l1_flush_threshold: 64 * 1024 * 1024,
            l2_compaction_threshold: 64,
            sync_level: SyncLevel::Batch {
                ms: 100,
                max_pending: 1000,
            },
            data_dir: std::path::PathBuf::from("."),
        }
    }
}

/// Synchronization level for L1 writes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SyncLevel {
    /// Every write is fsync'd immediately. Zero durability risk, lower throughput.
    Immediate,
    /// Group commit: flush after N ms or N pending entries, whichever first.
    Batch { ms: u64, max_pending: usize },
    /// Fully async: write to memory only, rely on background sync.
    Async,
}

/// Trait for querying deltas — implemented by DeltaLayerStack and compatible wrappers.
/// This is the plug point for scan.rs and point_get.rs.
pub trait DeltaQueryLayer: Send + Sync {
    /// Query all deltas visible for a segment at a given snapshot.
    fn query(
        &self,
        seg_id: &str,
        snapshot_txn: u64,
        commit_ts_by_txn: &HashMap<u64, u64>,
        active_txns: &HashSet<u64>,
    ) -> Result<Vec<DeltaCell>>;

    /// Query a single cell delta (for point-get optimization).
    ///
    /// Errors from L2/L3 disk reads are propagated. NotFound is returned as `Ok(None)`.
    fn query_cell(
        &self,
        seg_id: &str,
        row_offset: u32,
        column: &str,
        snapshot_txn: u64,
        commit_ts_by_txn: &HashMap<u64, u64>,
        active_txns: &HashSet<u64>,
    ) -> Result<Option<DeltaCell>>;

    /// Get all segment IDs across all layers (L1 + L2 + L3).
    fn get_all_segment_ids(&self) -> Vec<String>;

    /// Query all layers with guard-level deduplication.
    /// For each (seg_id, col, row_offset), returns only the most recent delta
    /// (highest txn_id) across all layers.
    fn query_all_layers(&self, seg_id: &str, committed: u64, since: u64) -> Vec<DeltaCell>;

    /// Batch query across multiple segments in one shot.
    ///
    /// cdc005 fix: replaces N individual `query()` calls with a single
    /// multi-segment scan. Implementations can optimize by amortizing I/O
    /// (e.g., prefetching L2/L3 column files for all segments at once).
    fn query_batch(
        &self,
        seg_ids: &[String],
        snapshot_txn: u64,
        commit_ts_by_txn: &HashMap<u64, u64>,
        active_txns: &HashSet<u64>,
    ) -> Result<HashMap<String, Vec<DeltaCell>>>;

    /// Batch query across multiple segments with true batch I/O.
    ///
    /// Implementations can optimize by:
    /// - Parallel file opens across segments (vs. sequential in query_batch)
    /// - LRU cache reuse across multiple segments
    /// - Reduced syscall overhead from batching
    ///
    /// Returns per-segment delta results. Callers should merge L1/L3 layers separately.
    fn get_visible_batch(
        &self,
        seg_ids: &[String],
        snapshot_txn: u64,
    ) -> Result<HashMap<String, Vec<DeltaCell>>>;
}

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering as AtomicOrdering};

/// The three-layer delta storage engine.
pub struct DeltaLayerStack {
    pub l1: Arc<DeltaMemStore>,
    pub l2: Arc<DeltaL2Disk>,
    pub l3: Arc<DeltaL3Frozen>,
    pub config: DeltaConfig,
    /// Shared L2 patch count (shared with FlushEngine via Arc).
    l2_patch_count: Arc<AtomicUsize>,
    /// Shared L3 patch count (incremented on L2→L3 compaction).
    l3_patch_count: Arc<AtomicUsize>,
    /// Cache for recently flushed deltas to prevent race window loss in query_all_layers.
    /// When flush_l1_to_l2 is in progress, deltas are temporarily stored here.
    /// Shared with FlushEngine via Arc clone.
    #[allow(clippy::type_complexity)]
    pub recent_flush: Arc<RwLock<BTreeMap<(String, String, u64), DeltaCell>>>,
    /// D5 fix: flush epoch counter. Incremented atomically after each L1→L2 flush.
    /// Allows read_changes to detect concurrent flush and retry segment enumeration.
    pub flush_epoch: Arc<AtomicU64>,
}

impl DeltaLayerStack {
    /// Create a new delta layer stack with shared counters.
    pub fn new(
        config: DeltaConfig,
        l2_patch_count: Arc<AtomicUsize>,
        l3_patch_count: Arc<AtomicUsize>,
    ) -> Self {
        let data_dir = config.data_dir.clone();
        Self {
            l1: Arc::new(DeltaMemStore::new(
                config.l1_flush_threshold,
                config.sync_level.clone(),
            )),
            l2: Arc::new(DeltaL2Disk::new(data_dir.join("l2"))),
            l3: Arc::new(DeltaL3Frozen::new(data_dir.join("l3"))),
            config,
            l2_patch_count,
            l3_patch_count,
            recent_flush: Arc::new(RwLock::new(BTreeMap::new())),
            flush_epoch: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Create a new delta layer stack with independent counters (convenience for tests).
    pub fn new_with_default_counters(config: DeltaConfig) -> Self {
        Self::new(
            config,
            Arc::new(AtomicUsize::new(0)),
            Arc::new(AtomicUsize::new(0)),
        )
    }

    /// Get all segment IDs across all layers (L1 + L2 + L3).
    /// Deduplicated to avoid duplicates from segments that exist in multiple layers.
    pub fn get_all_segment_ids(&self) -> Vec<String> {
        let mut seg_ids = std::collections::HashSet::new();
        seg_ids.extend(self.l1.get_segment_ids());
        seg_ids.extend(self.l2.get_segment_ids());
        seg_ids.extend(self.l3.get_segment_ids());
        seg_ids.into_iter().collect()
    }

    /// Get all segment IDs across all layers (L1 + L2 + L3) for a specific table.
    /// Callers pass the table's known segment IDs so cross-table delta pressure cannot leak.
    pub fn get_segment_ids_for_table(
        &self,
        table_segment_ids: &std::collections::HashSet<String>,
    ) -> Vec<String> {
        self.get_all_segment_ids()
            .into_iter()
            .filter(|seg_id| table_segment_ids.contains(seg_id))
            .collect()
    }

    /// d028: Replay committed WAL ops into L1 memstore after crash recovery.
    ///
    /// After WAL replay applies committed ops to KV data files, this method rebuilds
    /// the in-memory L1 state so `get_visible` returns all recovered entries.
    /// Only called for `SyncLevel::Async` mode where WAL entries are not synced
    /// during normal operation and need explicit replay into L1.
    pub fn recover_from_wal(&self, ops: Vec<WalOp>) {
        use crate::write::OpPayload;

        let ops_count = ops.len();
        for op in ops {
            let delta = DeltaCell {
                seg_id: match &op.payload {
                    OpPayload::Insert { seg_id, .. } => seg_id.clone(),
                    OpPayload::Update { new_seg_id, .. } => new_seg_id.clone(),
                    OpPayload::Delete { seg_id, .. } => seg_id.clone(),
                    _ => continue,
                },
                row_offset: match &op.payload {
                    OpPayload::Insert { offset, .. } => *offset,
                    OpPayload::Update { offset, .. } => *offset,
                    OpPayload::Delete { offset, .. } => *offset,
                    _ => continue,
                },
                column: match &op.payload {
                    OpPayload::Insert { columns, .. } => {
                        // TBD: column files may have fewer rows than visibility — add boundary check
                        columns.first().cloned().unwrap_or_default()
                    }
                    OpPayload::Update { columns, .. } => {
                        // TBD: column files may have fewer rows than visibility — add boundary check
                        columns.first().cloned().unwrap_or_default()
                    }
                    OpPayload::Delete { before_row, .. } => {
                        before_row.first().map(|(col, _)| col.clone()).unwrap_or_default()
                    }
                    _ => continue,
                },
                txn_id: op.txn_id,
                before: None,
                after: None,
                committed: true,
                ts: 0,
            };
            self.l1.replay_into(delta);
        }
        tracing::info!("DeltaLayerStack: replayed {} ops into L1", ops_count);
    }

    /// Get all visible deltas for a segment across all layers (L1 + L2 + L3).
    ///
    /// ## Concurrency Model
    ///
    /// This function reads from L1, L2, and L3 in sequence. To prevent race conditions
    /// where a flush (L1→L2) happens between reading L1 and L2 (causing data loss),
    /// recently flushed deltas are cached in `recent_flush` during flush operations.
    ///
    /// ## Race Window Prevention
    ///
    /// - Before flush: `recent_flush` is cleared and populated with drained L1 data
    /// - During flush: query reads `recent_flush` in addition to L1/L2/L3
    /// - After flush: `recent_flush` is cleared
    ///
    /// This ensures that if a flush occurs between L1 and L2 reads, the flushed data
    /// is still visible in `recent_flush`.
    pub fn query_all_layers(&self, seg_id: &str, committed: u64, since: u64) -> Vec<DeltaCell> {
        let mut all_deltas = Vec::new();

        // L1: in-memory, cannot fail
        let l1 = self.l1.get_visible(seg_id, committed).as_ref().clone();
        all_deltas.extend(
            l1.into_iter()
                .filter(|d| d.txn_id > since && d.txn_id <= committed && d.committed),
        );

        // Recent flush cache: prevents data loss if flush happens between L1 and L2 reads
        {
            let recent = self.recent_flush.read();
            all_deltas.extend(
                recent
                    .values()
                    .filter(|d| {
                        d.seg_id == seg_id
                            && d.txn_id > since
                            && d.txn_id <= committed
                            && d.committed
                    })
                    .cloned(),
            );
        }

        // L2: disk read
        if let Ok(l2) = self.l2.get_visible(seg_id, committed) {
            all_deltas.extend(
                l2.into_iter()
                    .filter(|d| d.txn_id > since && d.txn_id <= committed && d.committed),
            );
        }

        // L3: frozen read
        if let Ok(l3) = self.l3.get_visible(seg_id, committed) {
            all_deltas.extend(
                l3.into_iter()
                    .filter(|d| d.txn_id > since && d.txn_id <= committed && d.committed),
            );
        }

        // Guard-level deduplication: for each (seg_id, col, row_offset),
        // keep only the delta with the highest txn_id (most recent).
        // This handles cases where the same cell was updated multiple times
        // across layers (L1 update overwrites L2, L2 flush overwrites L3, etc.).
        let mut deduped: BTreeMap<(String, String, u64), DeltaCell> = BTreeMap::new();
        for delta in all_deltas {
            let key = (delta.seg_id.clone(), delta.column.clone(), delta.row_offset);
            if let Some(existing) = deduped.get(&key) {
                if delta.txn_id > existing.txn_id {
                    deduped.insert(key, delta);
                }
            } else {
                deduped.insert(key, delta);
            }
        }

        deduped.into_values().collect()
    }

    /// Cache deltas from a recent flush to prevent race window loss.
    /// Called by FlushEngine during flush_l1_to_l2.
    pub fn cache_recent_flush(&self, deltas: Vec<DeltaCell>) {
        let mut recent = self.recent_flush.write();
        recent.clear();
        for delta in deltas {
            let key = (delta.seg_id.clone(), delta.column.clone(), delta.row_offset);
            recent.insert(key, delta);
        }
    }

    /// Clear the recent flush cache. Called after flush completes.
    /// Also increments flush_epoch so concurrent readers can detect the flush.
    pub fn clear_recent_flush(&self) {
        let mut recent = self.recent_flush.write();
        recent.clear();
        drop(recent);
        // D5 fix: increment flush_epoch after flush completes
        self.flush_epoch.fetch_add(1, AtomicOrdering::SeqCst);
    }

    /// Put a delta cell into L1. Returns error if WAL sync fails (P8-41).
    pub fn put(&self, delta: DeltaCell) -> Result<()> {
        self.l1.put(delta)
    }

    /// Record that a patch was appended to L2 (increments L2 patch counter).
    pub fn record_l2_patch(&self) {
        self.l2_patch_count.fetch_add(1, AtomicOrdering::Relaxed);
    }

    /// Record that patches were compacted from L2 to L3.
    /// Decrements L2 count and increments L3 count.
    pub fn record_l2_to_l3_compaction(&self, l2_patches_removed: usize) {
        // Use saturating_sub to prevent underflow if patches_removed > actual count.
        let current = self.l2_patch_count.load(AtomicOrdering::Relaxed);
        let new = current.saturating_sub(l2_patches_removed);
        self.l2_patch_count.store(new, AtomicOrdering::Relaxed);
        self.l3_patch_count
            .fetch_add(l2_patches_removed, AtomicOrdering::Relaxed);
    }

    /// Get current patch counts for checkpointing.
    pub fn patch_counts(&self) -> (usize, usize) {
        (
            self.l2_patch_count.load(AtomicOrdering::Relaxed),
            self.l3_patch_count.load(AtomicOrdering::Relaxed),
        )
    }

    /// Query all visible deltas across all layers.
    fn query_all(
        &self,
        seg_id: &str,
        snapshot_txn: u64,
        commit_ts_by_txn: &HashMap<u64, u64>,
        active_txns: &HashSet<u64>,
    ) -> Result<Vec<DeltaCell>> {
        let l1_deltas = self.l1.get_visible(seg_id, snapshot_txn).as_ref().clone();
        let l2_deltas = self.l2.get_visible(seg_id, snapshot_txn)?;
        let l3_deltas = self.l3.get_visible(seg_id, snapshot_txn)?;

        let layers = vec![l1_deltas, l2_deltas, l3_deltas];
        Ok(k_way_merge(
            layers,
            snapshot_txn,
            commit_ts_by_txn,
            active_txns,
        ))
    }
}

impl DeltaQueryLayer for DeltaLayerStack {
    fn query(
        &self,
        seg_id: &str,
        snapshot_txn: u64,
        commit_ts_by_txn: &HashMap<u64, u64>,
        active_txns: &HashSet<u64>,
    ) -> Result<Vec<DeltaCell>> {
        self.query_all(seg_id, snapshot_txn, commit_ts_by_txn, active_txns)
    }

    fn query_cell(
        &self,
        seg_id: &str,
        row_offset: u32,
        column: &str,
        snapshot_txn: u64,
        commit_ts_by_txn: &HashMap<u64, u64>,
        active_txns: &HashSet<u64>,
    ) -> Result<Option<DeltaCell>> {
        let candidates = self.query_all(seg_id, snapshot_txn, commit_ts_by_txn, active_txns)?;
        Ok(candidates
            .into_iter()
            .filter(|d| d.row_offset == row_offset as u64 && d.column == column)
            .max_by_key(|d| d.txn_id))
    }

    fn get_all_segment_ids(&self) -> Vec<String> {
        self.get_all_segment_ids()
    }

    fn query_all_layers(&self, seg_id: &str, committed: u64, since: u64) -> Vec<DeltaCell> {
        DeltaLayerStack::query_all_layers(self, seg_id, committed, since)
    }

    /// D20 fix: True batch query across multiple segments.
    ///
    /// Merges L1 (in-memory) → recent_flush → L2 (batch disk) → L3 (frozen).
    /// Uses L2's `get_visible_batch` for parallel I/O optimization.
    fn get_visible_batch(
        &self,
        seg_ids: &[String],
        snapshot_txn: u64,
    ) -> Result<HashMap<String, Vec<DeltaCell>>> {
        // L2 batch query (the main optimization)
        let l2_results = self.l2.get_visible_batch(seg_ids, snapshot_txn)?;

        let mut results: HashMap<String, Vec<DeltaCell>> =
            seg_ids.iter().map(|s| (s.clone(), Vec::new())).collect();

        for seg_id in seg_ids {
            let mut all_deltas = Vec::new();

            // L1: in-memory
            let l1 = self.l1.get_visible(seg_id, snapshot_txn).as_ref().clone();
            all_deltas.extend(
                l1.into_iter()
                    .filter(|d| d.txn_id <= snapshot_txn && d.committed)
            );

            // Recent flush cache
            {
                let recent = self.recent_flush.read();
                all_deltas.extend(
                    recent
                        .values()
                        .filter(|d| {
                            d.seg_id == *seg_id
                                && d.txn_id <= snapshot_txn
                                && d.committed
                        })
                        .cloned()
                );
            }

            // L2 batch results
            if let Some(l2) = l2_results.get(seg_id) {
                all_deltas.extend(l2.iter().cloned());
            }

            // L3: frozen
            if let Ok(l3) = self.l3.get_visible(seg_id, snapshot_txn) {
                all_deltas.extend(l3);
            }

            // Deduplicate: latest txn wins per (seg_id, column, row_offset)
            let mut deduped: BTreeMap<(String, String, u64), DeltaCell> = BTreeMap::new();
            for delta in all_deltas {
                let key = (delta.seg_id.clone(), delta.column.clone(), delta.row_offset);
                if let Some(existing) = deduped.get(&key) {
                    if delta.txn_id > existing.txn_id {
                        deduped.insert(key, delta);
                    }
                } else {
                    deduped.insert(key, delta);
                }
            }

            results.insert(seg_id.clone(), deduped.into_values().collect());
        }

        Ok(results)
    }

    fn query_batch(
        &self,
        seg_ids: &[String],
        snapshot_txn: u64,
        commit_ts_by_txn: &HashMap<u64, u64>,
        active_txns: &HashSet<u64>,
    ) -> Result<HashMap<String, Vec<DeltaCell>>> {
        // cdc005+d20 fix: delegate to get_visible_batch for true parallel I/O.
        // The legacy commit_ts_by_txn/active_txns filtering is applied after
        // the batch merge, matching the same semantics as query_all().
        let batch = DeltaLayerStack::get_visible_batch(self, seg_ids, snapshot_txn)?;

        // Apply legacy commit_ts_by_txn filtering (txn visibility check)
        let mut results = HashMap::new();
        for (seg_id, deltas) in batch {
            let filtered: Vec<DeltaCell> = deltas
                .into_iter()
                .filter(|d| {
                    // committed == true means it's in commit history
                    if !d.committed {
                        return false;
                    }
                    // check active_txns: in-flight txns are not visible
                    if active_txns.contains(&d.txn_id) {
                        return false;
                    }
                    // commit_ts_by_txn: verify this txn has a commit_ts
                    if commit_ts_by_txn.is_empty() {
                        // no commit history means this delta is invisible
                        // (backward compat: legacy cells without commit history are filtered)
                        return false;
                    }
                    // txn must be in commit history
                    if !commit_ts_by_txn.contains_key(&d.txn_id) {
                        return false;
                    }
                    // commit_ts of this txn must be <= snapshot_txn
                    let commit_ts = commit_ts_by_txn.get(&d.txn_id).unwrap();
                    *commit_ts <= snapshot_txn
                })
                .collect();
            results.insert(seg_id, filtered);
        }
        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn query_filters_out_uncommitted_legacy_cells_without_commit_history() {
        let delta_layer = DeltaLayerStack::new_with_default_counters(DeltaConfig::default());
        delta_layer
            .put(DeltaCell {
                seg_id: "seg-a".to_string(),
                row_offset: 3,
                column: "value".to_string(),
                txn_id: 41,
                before: None,
                after: Some(Arc::new(vec![9])),
                committed: true,
                ts: 0,
            })
            .expect("put delta");

        let active_txns = HashSet::new();
        let visible = delta_layer
            .query("seg-a", 100, &HashMap::new(), &active_txns)
            .expect("query succeeds");
        assert!(
            visible.is_empty(),
            "legacy committed flag must not bypass commit history"
        );

        let mut commit_ts_by_txn = HashMap::new();
        commit_ts_by_txn.insert(41, 40);
        let visible = delta_layer
            .query("seg-a", 100, &commit_ts_by_txn, &active_txns)
            .expect("query succeeds");
        assert_eq!(visible.len(), 1);
        assert_eq!(visible[0].txn_id, 41);
    }
}
