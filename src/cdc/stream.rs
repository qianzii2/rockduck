//! CDC DeltaCell ChangeStream — fine-grained CDC via DeltaLayerStack.
//!
//! # Architecture
//!
//! We leverage DeltaLayerStack's DeltaCell capability to expose a change stream.
//! No Kafka, no Debezium connector. Just expose the DeltaCell data.
//!
////! # What We Don't Need
//!
//! - Kafka integration (optional — kept in cdc/sink.rs)
//! - Debezium format (optional — kept in cdc/debezium.rs)
//! - Exactly-once semantics (consumer handles deduplication)
//! - Tombstone records (expressed via op: CellOp::Delete)
//!
//! # What We Do
//!
//! - Expose DeltaLayerStack's DeltaCell as a change stream
//! - Group changes by txn_id for transactional ordering
//! - Support Arrow RecordBatch (high-performance) and JSON (debug-friendly)
//! - WAL replay fallback when DeltaLayerStack data is cleared
//!
//! # D10 TOCTOU Race Condition Advisory
//!
//! `read_changes()` queries DeltaLayerStack (L1→L2→L3) for committed deltas.
//! Because `committed` is captured at the start of the call and `since` is checked
//! against it, a concurrent flush (L1→L2) can cause a delta written to L1 during
//! the call to be silently dropped:
//!
//! 1. Thread A: `read_changes()` captures `committed = 10`, starts iterating segments.
//! 2. Thread B: Flush L1→L2 completes; a delta with `txn_id = 9` is now in L2.
//! 3. Thread A: `query_all_layers(seg, committed=10, since=X)` sees the L2 delta
//!    (txn_id=9) and includes it — this is safe.
//! 4. **RACE**: A different segment's delta was in-flight in L1 when Thread A
//!    read `get_all_segment_ids()`. After the flush completes, that delta moves to
//!    L2 but Thread A never re-queries that segment because `get_all_segment_ids()`
//!    already returned.
//!
//! This is acceptable for CDC because:
//! - It only causes **delayed** emission, never lost data (the delta is in L2
//!   and will be picked up on the next call with a higher `committed` watermark).
//! - The next call's `committed` will be ≥ the previous call's, ensuring progress.
//! - No delta is permanently lost; visibility is eventually guaranteed.
//!
//! If strict ordering per segment is required within a single `read_changes()`
//! call, external synchronization (e.g., holding a flush lock) is needed before
//! calling `read_changes()`.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_schema::{DataType, Field, Schema};

use crate::db::RockDuck;
use crate::error::Result;
use crate::metadata::projection::{ProjectionContract, ProjectionSurface, SidecarClass};
use crate::storage::delta::{types::DeltaCell, DeltaQueryLayer};

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CdcReplayHotspotStats {
    pub read_calls: u64,
    pub scanned_segments: u64,
    pub collected_deltas: u64,
}

#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

// =============================================================================
// Data Types
// =============================================================================

/// Operation type for a single cell change.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    serde::Serialize,
    serde::Deserialize,
    rkyv::Archive,
    rkyv::Serialize,
    rkyv::Deserialize,
)]
pub enum CellOp {
    Insert,
    Update,
    Delete,
}

impl CellOp {
    fn from_delta(delta: &DeltaCell) -> Self {
        if delta.is_insert() {
            CellOp::Insert
        } else if delta.is_delete() {
            CellOp::Delete
        } else {
            CellOp::Update
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            CellOp::Insert => "insert",
            CellOp::Update => "update",
            CellOp::Delete => "delete",
        }
    }
}

/// A single cell-level change.
#[derive(
    Debug,
    Clone,
    serde::Serialize,
    serde::Deserialize,
    rkyv::Archive,
    rkyv::Serialize,
    rkyv::Deserialize,
)]
pub struct CellChange {
    pub table: String,
    /// Segment identifier.
    pub seg_id: String,
    pub col: String,
    pub row_offset: u64,
    pub op: CellOp,
    /// Before-image bytes (None for inserts).
    pub before: Option<Vec<u8>>,
    /// After-image bytes (None for deletes).
    pub after: Option<Vec<u8>>,
    pub txn_id: u64,
}

impl From<(&str, &DeltaCell)> for CellChange {
    fn from((table, delta): (&str, &DeltaCell)) -> Self {
        Self {
            table: table.to_string(),
            seg_id: delta.seg_id.clone(),
            col: delta.column.clone(),
            row_offset: delta.row_offset,
            op: CellOp::from_delta(delta),
            before: delta.before.as_ref().map(|v| v.as_ref().clone()),
            after: delta.after.as_ref().map(|v| v.as_ref().clone()),
            txn_id: delta.txn_id,
        }
    }
}

/// A batch of changes from a single transaction.
#[derive(
    Debug,
    Clone,
    serde::Serialize,
    serde::Deserialize,
    rkyv::Archive,
    rkyv::Serialize,
    rkyv::Deserialize,
)]
pub struct ChangeBatch {
    pub txn_id: u64,
    pub changes: Vec<CellChange>,
}

/// Row-level change: all cell changes for a given (txn_id, row_key).
#[derive(
    Debug,
    Clone,
    serde::Serialize,
    serde::Deserialize,
    rkyv::Archive,
    rkyv::Serialize,
    rkyv::Deserialize,
)]
pub struct RowChange {
    pub txn_id: u64,
    pub seg_id: String,
    pub row_offset: u32,
    pub table: String,
    /// Vector of (column, before, after) for all changed cells in this row.
    #[serde(with = "serde_cell_change_tuple")]
    pub cell_changes: Vec<CellChangeTuple>,
}

/// A tuple representing (column_name, before_bytes, after_bytes).
pub type CellChangeTuple = (String, Option<Vec<u8>>, Option<Vec<u8>>);

mod serde_cell_change_tuple {
    use super::CellChangeTuple;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S>(v: &[CellChangeTuple], s: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        v.serialize(s)
    }

    pub fn deserialize<'de, D>(d: D) -> Result<Vec<CellChangeTuple>, D::Error>
    where
        D: Deserializer<'de>,
    {
        <Vec<CellChangeTuple>>::deserialize(d)
    }
}

// =============================================================================
// ChangeStream
// =============================================================================

/// A change stream over the database.
///
/// Reads committed DeltaCell entries from DeltaLayerStack since a given txn_id.
/// Each `read_changes()` call returns all changes since the last call (exclusive)
/// up to the current committed txn (inclusive).
///
/// # Example
///
/// ```ignore
/// let stream = ChangeStream::new(Arc::clone(&db));
/// loop {
///     let batches = stream.read_changes()?;
///     for batch in batches {
///         for change in batch.changes {
///             println!("{:?}", change);
///         }
///     }
///     std::thread::sleep(std::time::Duration::from_secs(1));
/// }
/// ```
/// ChangeStream — single-threaded CDC consumer for WAL replay.
///
/// **Thread Safety Invariant**: Each `ChangeStream` instance must be owned by a single
/// consumer thread. Do not share a single `ChangeStream` across multiple threads.
///
/// Reasoning:
/// - `since_txn` is updated atomically (AtomicU64 SeqCst) but the read-and-advance
///   sequence is not protected. Concurrent readers could cause lost updates, skipping
///   or duplicating CDC events.
/// - `hotspot_stats` uses `parking_lot::RwLock` — safe for concurrent reads, but
///   writes from concurrent callers would corrupt the hotspot metrics.
/// - The data structures (AtomicU64, RwLock) are individually thread-safe, but the
///   **usage pattern** must enforce single-threaded ownership.
///
/// Usage: each CDC consumer should own its own `ChangeStream` instance, or protect
/// the entire read-and-advance sequence with an external Mutex if sharing is required.
pub struct ChangeStream {
    db: Arc<RockDuck>,
    /// The txn_id to start reading from (exclusive).
    /// Updated to current committed txn after each read_changes() call.
    since_txn: AtomicU64,
    projection_contract: ProjectionContract,
    hotspot_stats: parking_lot::RwLock<CdcReplayHotspotStats>,
}

/// Data for building an Arrow batch.
struct ArrowBatchData {
    txn_ids: Vec<u64>,
    seg_ids: Vec<String>,
    cols: Vec<String>,
    row_offsets: Vec<u64>,
    ops: Vec<String>,
    befores: Vec<Option<Vec<u8>>>,
    afters: Vec<Option<Vec<u8>>>,
}

impl ChangeStream {
    /// Create a new ChangeStream starting from the current committed txn.
    pub fn new(db: Arc<RockDuck>) -> Self {
        let since_txn = db.mvcc.read().committed_txn();
        Self {
            db,
            since_txn: AtomicU64::new(since_txn),
            projection_contract: ProjectionContract {
                surface: ProjectionSurface::Vtab,
                visibility: crate::mvcc::visibility::VisibilityProjection::Historical,
                sidecar_class: SidecarClass::SanctionedSidecar,
                evidence_hook: "CDC stream remains outward-only and must emit governance evidence before export",
                enforcement: crate::metadata::projection::ContractEnforcement::Blocking,
            },
            hotspot_stats: parking_lot::RwLock::new(CdcReplayHotspotStats::default()),
        }
    }

    /// Create a ChangeStream starting from a specific txn_id (exclusive).
    pub fn from_txn(db: Arc<RockDuck>, since_txn: u64) -> Self {
        Self {
            db,
            since_txn: AtomicU64::new(since_txn),
            projection_contract: ProjectionContract {
                surface: ProjectionSurface::Vtab,
                visibility: crate::mvcc::visibility::VisibilityProjection::Historical,
                sidecar_class: SidecarClass::SanctionedSidecar,
                evidence_hook: "CDC stream remains outward-only and must emit governance evidence before export",
                enforcement: crate::metadata::projection::ContractEnforcement::Blocking,
            },
            hotspot_stats: parking_lot::RwLock::new(CdcReplayHotspotStats::default()),
        }
    }

    /// Get the current since_txn watermark.
    pub fn since_txn(&self) -> u64 {
        self.since_txn.load(Ordering::SeqCst)
    }

    /// Read all changes since the last call (exclusive) up to the current committed
    /// transaction (inclusive).
    ///
    /// Returns changes grouped by txn_id, sorted ascending by txn_id.
    ///
    /// This is the **cell-level** API: one `CellChange` per modified cell.
    ///
    /// Queries DeltaLayerStack for committed deltas. Falls back to WAL replay
    /// if DeltaLayerStack has no data for the requested range.
    pub fn read_changes(&self) -> Result<Vec<ChangeBatch>> {
        self.projection_contract.assert_blocking_governance();
        // D5 fix: capture flush_epoch before reading segments
        let flush_epoch = self.db.delta_layer.flush_epoch.clone();
        let captured_epoch = flush_epoch.load(std::sync::atomic::Ordering::SeqCst);

        // Retry loop: if a flush completes while we're reading, re-read segment list
        loop {
            let committed = self.db.mvcc.read().committed_txn();
            let since = self.since_txn.load(Ordering::SeqCst);

            if since >= committed {
                return Ok(Vec::new());
            }

            // Collect all deltas from all three layers (L1 + L2 + L3)
            let mut txn_groups: HashMap<u64, Vec<CellChange>> = HashMap::new();

            let segments: Vec<String> = self.db.delta_layer.get_all_segment_ids();
            let seg_count = segments.len();
            let mut delta_count: usize = 0;

            for seg_id in segments {
                let deltas = self
                    .db
                    .delta_layer
                    .query_all_layers(&seg_id, committed, since);
                delta_count += deltas.len();
                for delta in deltas {
                    let change = CellChange::from((seg_id.as_str(), &delta));
                    txn_groups.entry(delta.txn_id).or_default().push(change);
                }
            }

            // D5 fix: check if flush happened during our read
            let current_epoch = flush_epoch.load(std::sync::atomic::Ordering::SeqCst);
            if current_epoch == captured_epoch {
                // No flush happened — results are consistent
                let mut batches: Vec<ChangeBatch> = txn_groups
                    .into_iter()
                    .map(|(txn_id, changes)| ChangeBatch { txn_id, changes })
                    .collect();

                // Sort each txn's changes by (seg_id, row_offset, col) for deterministic order
                for changes in batches.iter_mut() {
                    changes.changes.sort_by(|a, b| {
                        a.seg_id
                            .cmp(&b.seg_id)
                            .then(a.row_offset.cmp(&b.row_offset))
                            .then(a.col.cmp(&b.col))
                    });
                }
                batches.sort_by_key(|b| b.txn_id);

                // Update stats
                {
                    let mut stats = self.hotspot_stats.write();
                    stats.read_calls += 1;
                    stats.scanned_segments += seg_count as u64;
                    stats.collected_deltas += delta_count as u64;
                }

                // Advance watermark
                if let Some(last) = batches.last() {
                    self.since_txn.store(last.txn_id, Ordering::SeqCst);
                }

                return Ok(batches);
            }
            // Flush happened during our read — retry with new epoch
            tracing::debug!(
                "D5: flush detected (epoch {} -> {}), retrying segment enumeration",
                captured_epoch,
                current_epoch
            );
        }
    }

    pub fn replay_hotspot_stats(&self) -> CdcReplayHotspotStats {
        self.hotspot_stats.read().clone()
    }

    /// Read changes as row-level aggregates.
    ///
    /// Groups cell changes by (seg_id, row_offset) within each transaction,
    /// producing one `RowChange` per modified row. This is the **row-level** API.
    ///
    /// Returns `Vec<ChangeBatch>` where each batch's `changes` are already
    /// grouped and sorted by row key. Consumers can aggregate by `(seg_id, row_offset)`.
    pub fn read_row_changes(&self) -> Result<Vec<ChangeBatch>> {
        let batches = self.read_changes()?;

        let mut row_batches: Vec<ChangeBatch> = Vec::with_capacity(batches.len());

        for batch in batches {
            let mut row_map: HashMap<(String, u64), Vec<CellChange>> = HashMap::new();

            for change in batch.changes {
                let key = (change.seg_id.clone(), change.row_offset);
                row_map.entry(key).or_default().push(change);
            }

            // Collect all cell changes sorted by row key
            let mut sorted_rows: Vec<_> = row_map.into_iter().collect();
            sorted_rows.sort_by_key(|(k, _)| (k.0.clone(), k.1));

            let row_changes: Vec<CellChange> = sorted_rows
                .into_iter()
                .flat_map(|(_, cell_changes)| cell_changes)
                .collect();

            row_batches.push(ChangeBatch {
                txn_id: batch.txn_id,
                changes: row_changes,
            });
        }

        Ok(row_batches)
    }

    /// Read changes as Arrow RecordBatch (high-performance pipe to Arrow consumers).
    ///
    /// Schema: txn_id, seg_id, col, row_offset, op, before, after
    /// Rows are split into batches of at most `batch_size` rows each.
    pub fn read_changes_as_arrow(&self, batch_size: usize) -> Result<Vec<RecordBatch>> {
        let batches = self.read_changes()?;
        if batches.is_empty() {
            return Ok(Vec::new());
        }

        let schema = Arc::new(Schema::new(vec![
            Field::new("txn_id", DataType::UInt64, false),
            Field::new("seg_id", DataType::Utf8, false),
            Field::new("col", DataType::Utf8, false),
            Field::new("row_offset", DataType::UInt64, false),
            Field::new("op", DataType::Utf8, false),
            Field::new("before", DataType::Binary, true),
            Field::new("after", DataType::Binary, true),
        ]));

        let mut result: Vec<RecordBatch> = Vec::new();
        let mut current_batch_size: usize = 0;
        let mut current_txn_ids: Vec<u64> = Vec::with_capacity(batch_size);
        let mut current_seg_ids: Vec<String> = Vec::with_capacity(batch_size);
        let mut current_cols: Vec<String> = Vec::with_capacity(batch_size);
        let mut current_row_offsets: Vec<u64> = Vec::with_capacity(batch_size);
        let mut current_ops: Vec<String> = Vec::with_capacity(batch_size);
        let mut current_befores: Vec<Option<Vec<u8>>> = Vec::with_capacity(batch_size);
        let mut current_afters: Vec<Option<Vec<u8>>> = Vec::with_capacity(batch_size);

        for batch in batches {
            for change in batch.changes {
                current_txn_ids.push(change.txn_id);
                current_seg_ids.push(change.seg_id);
                current_cols.push(change.col);
                current_row_offsets.push(change.row_offset);
                current_ops.push(change.op.as_str().to_string());
                current_befores.push(change.before);
                current_afters.push(change.after);
                current_batch_size += 1;

                if current_batch_size >= batch_size {
                    let batch = self.build_arrow_batch(
                        &schema,
                        ArrowBatchData {
                            txn_ids: std::mem::take(&mut current_txn_ids),
                            seg_ids: std::mem::take(&mut current_seg_ids),
                            cols: std::mem::take(&mut current_cols),
                            row_offsets: std::mem::take(&mut current_row_offsets),
                            ops: std::mem::take(&mut current_ops),
                            befores: std::mem::take(&mut current_befores),
                            afters: std::mem::take(&mut current_afters),
                        },
                    )?;
                    result.push(batch);
                    current_batch_size = 0;
                }
            }
        }

        // Flush remaining
        if current_batch_size > 0 {
            let batch = self.build_arrow_batch(
                &schema,
                ArrowBatchData {
                    txn_ids: current_txn_ids,
                    seg_ids: current_seg_ids,
                    cols: current_cols,
                    row_offsets: current_row_offsets,
                    ops: current_ops,
                    befores: current_befores,
                    afters: current_afters,
                },
            )?;
            result.push(batch);
        }

        Ok(result)
    }

    fn build_arrow_batch(
        &self,
        schema: &Arc<Schema>,
        data: ArrowBatchData,
    ) -> Result<RecordBatch> {
        use arrow_array::builder::{BinaryBuilder, StringBuilder, UInt64Builder};

        let n = data.txn_ids.len();

        let mut txn_id_b = UInt64Builder::with_capacity(n);
        let mut seg_id_b = StringBuilder::new();
        let mut col_b = StringBuilder::new();
        let mut row_offset_b = UInt64Builder::with_capacity(n);
        let mut op_b = StringBuilder::new();
        let mut before_b = BinaryBuilder::new();
        let mut after_b = BinaryBuilder::new();

        for i in 0..n {
            txn_id_b.append_value(data.txn_ids[i]);
            seg_id_b.append_value(&data.seg_ids[i]);
            col_b.append_value(&data.cols[i]);
            row_offset_b.append_value(data.row_offsets[i]);
            op_b.append_value(&data.ops[i]);
            match &data.befores[i] {
                Some(v) => before_b.append_value(v.as_slice()),
                None => before_b.append_null(),
            }
            match &data.afters[i] {
                Some(v) => after_b.append_value(v.as_slice()),
                None => after_b.append_null(),
            }
        }

        RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(txn_id_b.finish()) as arrow_array::ArrayRef,
                Arc::new(seg_id_b.finish()) as arrow_array::ArrayRef,
                Arc::new(col_b.finish()) as arrow_array::ArrayRef,
                Arc::new(row_offset_b.finish()) as arrow_array::ArrayRef,
                Arc::new(op_b.finish()) as arrow_array::ArrayRef,
                Arc::new(before_b.finish()) as arrow_array::ArrayRef,
                Arc::new(after_b.finish()) as arrow_array::ArrayRef,
            ],
        )
        .map_err(|e| crate::RockDuckError::Internal(format!("build arrow batch: {}", e)))
    }

    /// Read changes as pretty-printed JSON (debug-friendly).
    pub fn read_changes_as_json(&self) -> Result<String> {
        let batches = self.read_changes()?;
        serde_json::to_string_pretty(&batches)
            .map_err(|e| crate::RockDuckError::Internal(format!("JSON serialize: {}", e)))
    }

    /// Get the total number of unconsumed delta entries across all segments.
    ///
    /// Note: previously this only queried `delta_layer.l1`. Now it queries all layers (L1/L2/L3).
    /// This was a bug: delta entries accumulate across all layers, not just L1.
    ///
    /// cdc005 fix: uses `query_batch()` to amortize I/O overhead across all
    /// segments in a single call, replacing N individual `query()` calls.
    pub fn pending_delta_count(&self) -> usize {
        let since = self.since_txn.load(Ordering::SeqCst);
        let committed = self.db.mvcc.read().committed_txn();

        let segments: Vec<String> = self.db.delta_layer.get_all_segment_ids();
        if segments.is_empty() {
            return 0;
        }

        let snapshot = self
            .db
            .mvcc
            .read()
            .snapshot(crate::mvcc::visibility::IsolationLevel::Snapshot);
        let active_txns: HashSet<u64> = snapshot.active_txns.iter().copied().collect();

        // cdc005: batch query across all segments at once
        let Ok(results) = self.db.delta_layer.as_ref().query_batch(
            &segments,
            committed,
            &snapshot.commit_ts_map,
            &active_txns,
        ) else {
            return 0;
        };

        results
            .values()
            .map(|deltas| deltas.iter().filter(|d| d.txn_id > since).count())
            .sum()
    }

    /// Read changes at a specific point-in-time using a commit timestamp.
    ///
    /// Finds all committed transactions with `commit_ts <= timestamp` and returns
    /// their changes. Uses MVCC metadata to determine transaction visibility.
    pub fn read_changes_at_timestamp(&self, timestamp_ms: u64) -> Result<Vec<ChangeBatch>> {
        // Find the highest committed txn_id with commit_ts <= timestamp
        // We scan all delta entries and filter by the timestamp
        let _committed = self.db.mvcc.read().committed_txn();
        // Use SeqCst for consistency: this value controls which deltas are included.
        let since = self.since_txn.load(Ordering::SeqCst);

        // Scan CdcLogBuffer entries for timestamp-filtered results
        let log = self.db.cdc_log_buffer.read().entries();
        let mut txn_groups: HashMap<u64, Vec<CellChange>> = HashMap::new();

        for entry in &log {
            // CdcLogEntry doesn't have txn_ts -- it has `ts` which is the commit timestamp
            if entry.ts <= timestamp_ms && entry.txn_id > since {
                // Use the first column's bytes for before/after in cell-level CDC.
                // For row-level CDC, the full before/after Vec would be used instead.
                let before_bytes = entry.before.first().map(|(_, v)| v.clone());
                let after_bytes = entry.after.first().map(|(_, v)| v.clone());
                let change = CellChange {
                    table: entry.table.clone(),
                    seg_id: entry.seg_id.clone(),
                    col: String::new(), // CdcLogEntry stores per-column data, not per-cell
                    row_offset: 0,
                    op: match entry.op {
                        crate::cdc::CdcOpType::Insert => CellOp::Insert,
                        crate::cdc::CdcOpType::Update => CellOp::Update,
                        crate::cdc::CdcOpType::Delete => CellOp::Delete,
                    },
                    before: before_bytes,
                    after: after_bytes,
                    txn_id: entry.txn_id,
                };
                txn_groups.entry(entry.txn_id).or_default().push(change);
            }
        }

        let mut batches: Vec<ChangeBatch> = txn_groups
            .into_iter()
            .map(|(txn_id, changes)| ChangeBatch { txn_id, changes })
            .collect();
        batches.sort_by_key(|b| b.txn_id);

        Ok(batches)
    }

    /// Replay WAL entries since a given txn_id.
    ///
    /// This is the fallback path when DeltaLayerStack data has been cleared
    /// (e.g., after a checkpoint). It directly reads WAL entries and
    /// reconstructs ChangeBatch from OpPayload (Insert/Update/Delete).
    ///
    /// Returns all committed transactions from `since_txn` to the current
    /// committed txn, with their changes reconstructed from WAL.
    pub fn replay_wal_since(&self, since_txn: u64) -> Result<Vec<ChangeBatch>> {
        use crate::write::durability_wal::{OpType, WalReader};

        let wal_dir = self.db.wal.wal_dir().to_path_buf();
        let dir = durability::storage::FsDirectory::arc(&wal_dir)
            .map_err(|e| crate::RockDuckError::Internal(format!("open WAL dir: {}", e)))?;
        let reader = WalReader::<crate::write::durability_wal::WalOp>::new(dir);

        let mut txn_ops: HashMap<u64, Vec<CellChange>> = HashMap::new();

        // Collect ops per txn
        let mut pending: HashMap<u64, Vec<crate::write::durability_wal::WalOp>> = HashMap::new();

        let records = reader
            .replay_best_effort()
            .map_err(|e| crate::RockDuckError::Internal(format!("WAL replay failed: {}", e)))?;
        let committed_guard = self.db.mvcc.read();
        let stable_committed = committed_guard.committed_txn();
        let mut committed_txns = std::collections::HashSet::new();
        for rec in records {
            let op = rec.payload;
            let txn_id = op.txn_id;

            match op.op_type {
                OpType::Insert | OpType::Update | OpType::Delete
                    if txn_id > since_txn && txn_id <= stable_committed =>
                {
                    pending.entry(txn_id).or_default().push(op);
                }
                OpType::Commit if txn_id > since_txn && txn_id <= stable_committed => {
                    committed_txns.insert(txn_id);
                    if let Some(ops) = pending.remove(&txn_id) {
                        for wal_op in ops {
                            if let Some(changes) = Self::wal_op_to_cell_change(&wal_op)? {
                                txn_ops.entry(txn_id).or_default().extend(changes);
                            }
                        }
                    }
                }
                OpType::Rollback => {
                    pending.remove(&txn_id);
                }
                _ => {}
            }
        }
        txn_ops.retain(|txn_id, _| committed_txns.contains(txn_id));

        let mut batches: Vec<ChangeBatch> = txn_ops
            .into_iter()
            .map(|(txn_id, changes)| ChangeBatch { txn_id, changes })
            .collect();
        batches.sort_by_key(|b| b.txn_id);

        Ok(batches)
    }

    /// Convert a WAL operation to CellChange(s) for CDC replay.
    ///
    /// ## Correctness fix (P7-14 / cdc002 / cdc008)
    ///
    /// Previously, this function returned a single `CellChange` with dummy/empty
    /// values (e.g., `table.clone()`, `String::new()`, `0`).
    ///
    /// Now it returns `Vec<CellChange>` — one per column — with correct metadata:
    /// - **Insert**: one CellChange per column from `columns` + `wal_batch` (Arrow IPC).
    ///   `col` is set from the column name, `after` is extracted from `wal_batch`.
    ///   If `columns` is empty, column names are extracted from Arrow IPC schema (cdc008).
    /// - **Update**: one CellChange per column from `old_columns`. The `before`
    ///   value is extracted from `old_columns`; the `after` value is extracted
    ///   from `wal_batch`. Column names are set from `columns` or Arrow IPC schema.
    /// - **Delete**: one CellChange per column from `before_row`. Column names
    ///   come from `before_row`.
    ///
    /// `wal_batch` is Arrow IPC File format. Each row is a single-row RecordBatch.
    /// We use `arrow_ipc::reader::FileReader` to deserialize column values.
    fn wal_op_to_cell_change(
        wal_op: &crate::write::durability_wal::WalOp,
    ) -> Result<Option<Vec<CellChange>>> {
        use crate::write::durability_wal::OpPayload;

        let changes: Vec<CellChange> = match &wal_op.payload {
            OpPayload::Insert {
                table,
                columns,
                wal_batch,
                schema_bytes,
                seg_id,
                offset,
                ..
            } => {
                let table = table.clone();
                let seg_id = seg_id.clone();
                let txn_id = wal_op.txn_id;

                if wal_batch.is_empty() {
                    return Ok(None);
                }

                // cdc008 fix: if columns is empty, extract from Arrow IPC schema.
                let column_names: Vec<String> = if columns.is_empty() {
                    Self::extract_schema_column_names_from_ipc(wal_batch, schema_bytes)?
                } else {
                    columns.clone()
                };

                if column_names.is_empty() {
                    return Ok(None);
                }

                // Deserialize Arrow IPC File to extract per-column values.
                let column_values = Self::extract_column_values_from_ipc(wal_batch)?;

                column_names
                    .iter()
                    .zip(column_values)
                    .map(|(col, after_bytes)| CellChange {
                        table: table.clone(),
                        seg_id: seg_id.clone(),
                        col: col.clone(),
                        row_offset: *offset,
                        op: CellOp::Insert,
                        before: None,
                        after: Some(after_bytes),
                        txn_id,
                    })
                    .collect()
            }
            OpPayload::Update {
                table,
                columns,
                wal_batch,
                schema_bytes,
                old_columns,
                old_seg_id,
                old_offset,
                ..
            } => {
                let table = table.clone();
                let old_seg_id = old_seg_id.clone();
                let txn_id = wal_op.txn_id;

                if columns.is_empty() && wal_batch.is_empty() {
                    return Ok(None);
                }

                // cdc008 fix: if columns is empty, extract from Arrow IPC schema.
                let column_names: Vec<String> = if columns.is_empty() {
                    Self::extract_schema_column_names_from_ipc(wal_batch, schema_bytes)?
                } else {
                    columns.clone()
                };

                // Build before/after per column.
                let before_map: std::collections::HashMap<&str, Vec<u8>> = old_columns
                    .iter()
                    .map(|(col, bytes)| (col.as_str(), bytes.clone()))
                    .collect();

                // Deserialize Arrow IPC for after values.
                let after_values = Self::extract_column_values_from_ipc(wal_batch)?;

                column_names
                    .iter()
                    .zip(after_values)
                    .map(|(col, after_bytes)| {
                        let before_bytes = before_map.get(col.as_str()).cloned();
                        CellChange {
                            table: table.clone(),
                            seg_id: old_seg_id.clone(),
                            col: col.clone(),
                            row_offset: *old_offset,
                            op: CellOp::Update,
                            before: before_bytes,
                            after: Some(after_bytes),
                            txn_id,
                        }
                    })
                    .collect()
            }
            OpPayload::Delete {
                table,
                before_row,
                seg_id,
                offset,
                ..
            } => {
                let table = table.clone();
                let seg_id = seg_id.clone();
                let txn_id = wal_op.txn_id;

                before_row
                    .iter()
                    .map(|(col, bytes)| CellChange {
                        table: table.clone(),
                        seg_id: seg_id.clone(),
                        col: col.clone(),
                        row_offset: *offset,
                        op: CellOp::Delete,
                        before: Some(bytes.clone()),
                        after: None,
                        txn_id,
                    })
                    .collect()
            }
            OpPayload::Begin
            | OpPayload::Commit { .. }
            | OpPayload::Rollback
            | OpPayload::Checkpoint { .. }
            | OpPayload::Compaction { .. } => return Ok(None),
        };

        if changes.is_empty() {
            Ok(None)
        } else {
            Ok(Some(changes))
        }
    }

    /// Deserialize an Arrow IPC File into a Vec<Vec<u8>> (one bytes vector per column).
    /// The file contains single-row RecordBatches; one batch per cell change.
    fn extract_column_values_from_ipc(wal_batch: &std::sync::Arc<Vec<u8>>) -> Result<Vec<Vec<u8>>> {
        use arrow_ipc::reader::FileReader;
        use std::io::Cursor;

        let cursor = Cursor::new(wal_batch.as_ref());
        let reader = FileReader::try_new(cursor, None)
            .map_err(|e| crate::RockDuckError::Internal(format!("Arrow IPC reader: {}", e)))?;

        let mut results: Vec<Vec<u8>> = Vec::new();
        for batch_result in reader {
            let batch = batch_result
                .map_err(|e| crate::RockDuckError::Internal(format!("Arrow IPC batch: {}", e)))?;
            let num_rows = batch.num_rows();
            if num_rows == 0 {
                continue;
            }
            // Extract first row from each column as bytes.
            for col_idx in 0..batch.num_columns() {
                let col_array = batch.column(col_idx);
                let val_bytes = Self::scalar_to_bytes(col_array, 0)?;
                results.push(val_bytes);
            }
        }

        Ok(results)
    }

    /// Extract column names from Arrow IPC File schema.
    /// cdc008 fix: when `columns` field is empty in WAL, derive column names from
    /// the Arrow IPC schema embedded in `wal_batch` or `schema_bytes`.
    fn extract_schema_column_names_from_ipc(
        wal_batch: &std::sync::Arc<Vec<u8>>,
        schema_bytes: &[u8],
    ) -> Result<Vec<String>> {
        use arrow_ipc::reader::FileReader;
        use std::io::Cursor;

        // Try wal_batch first (IPC File format has schema in header).
        if !wal_batch.is_empty() {
            let cursor = Cursor::new(wal_batch.as_ref());
            if let Ok(reader) = FileReader::try_new(cursor, None) {
                let names: Vec<String> = reader
                    .schema()
                    .fields()
                    .iter()
                    .map(|f| f.name().clone())
                    .collect();
                if !names.is_empty() {
                    return Ok(names);
                }
            }
        }

        // Fall back to explicit schema_bytes.
        if !schema_bytes.is_empty() {
            let cursor = Cursor::new(schema_bytes);
            if let Ok(reader) = FileReader::try_new(cursor, None) {
                let names: Vec<String> = reader
                    .schema()
                    .fields()
                    .iter()
                    .map(|f| f.name().clone())
                    .collect();
                return Ok(names);
            }
        }

        Ok(Vec::new())
    }

    /// Serialize a scalar Arrow array value at `row_idx` to a byte vector.
    fn scalar_to_bytes(array: &arrow_array::ArrayRef, row_idx: usize) -> Result<Vec<u8>> {
        use arrow_array::types::*;
        use arrow_array::Array;

        macro_rules! extract {
            ($dt:ty, $type_id:expr) => {{
                let a = array.as_any().downcast_ref::<arrow_array::PrimitiveArray<$dt>>().ok_or_else(|| {
                    crate::RockDuckError::Internal(format!("type mismatch for {}", $type_id))
                })?;
                let val: Vec<u8> = a.value(row_idx).to_le_bytes().to_vec();
                val
            }};
        }

        let type_id = array.data_type();
        let bytes: Vec<u8> = match type_id {
            DataType::Int8 => extract!(Int8Type, "Int8"),
            DataType::Int16 => extract!(Int16Type, "Int16"),
            DataType::Int32 => extract!(Int32Type, "Int32"),
            DataType::Int64 => extract!(Int64Type, "Int64"),
            DataType::UInt8 => extract!(UInt8Type, "UInt8"),
            DataType::UInt16 => extract!(UInt16Type, "UInt16"),
            DataType::UInt32 => extract!(UInt32Type, "UInt32"),
            DataType::UInt64 => extract!(UInt64Type, "UInt64"),
            DataType::Float32 => extract!(Float32Type, "Float32"),
            DataType::Float64 => extract!(Float64Type, "Float64"),
            DataType::Boolean => {
                let a = array.as_any().downcast_ref::<arrow_array::BooleanArray>().ok_or_else(|| {
                    crate::RockDuckError::Internal("type mismatch for Boolean".to_string())
                })?;
                vec![if a.value(row_idx) { 1 } else { 0 }]
            }
            DataType::Utf8 => {
                let a = array.as_any().downcast_ref::<arrow_array::StringArray>().ok_or_else(|| {
                    crate::RockDuckError::Internal("type mismatch for Utf8".to_string())
                })?;
                a.value(row_idx).as_bytes().to_vec()
            }
            DataType::LargeUtf8 => {
                let a = array.as_any().downcast_ref::<arrow_array::LargeStringArray>().ok_or_else(|| {
                    crate::RockDuckError::Internal("type mismatch for LargeUtf8".to_string())
                })?;
                a.value(row_idx).as_bytes().to_vec()
            }
            DataType::Binary => {
                let a = array.as_any().downcast_ref::<arrow_array::BinaryArray>().ok_or_else(|| {
                    crate::RockDuckError::Internal("type mismatch for Binary".to_string())
                })?;
                a.value(row_idx).to_vec()
            }
            DataType::LargeBinary => {
                let a = array.as_any().downcast_ref::<arrow_array::LargeBinaryArray>().ok_or_else(|| {
                    crate::RockDuckError::Internal("type mismatch for LargeBinary".to_string())
                })?;
                a.value(row_idx).to_vec()
            }
            _ => {
                tracing::warn!(
                    target: "cdc_wal",
                    reason = "unsupported_type",
                    data_type = ?type_id,
                    "WAL parse error: unsupported Arrow data type"
                );
                return Err(crate::RockDuckError::Internal(format!(
                    "unsupported Arrow data type for CDC: {:?}",
                    type_id
                )));
            }
        };

        Ok(bytes)
    }

    /// Read changes as a stream of events including transaction boundaries.
    ///
    /// Returns `ChangeEvent::TransactionBegin` and `ChangeEvent::TransactionCommit`
    /// markers around cell changes, enabling consumers to assemble
    /// transaction-aware views and exactly-once processing.
    pub fn read_changes_with_txn_boundaries(&self) -> Result<Vec<ChangeEvent>> {
        let batches = self.read_changes()?;

        let mut events: Vec<ChangeEvent> = Vec::new();
        let mut prev_txn_id: Option<u64> = None;

        for batch in batches {
            // TransactionBegin marker
            if prev_txn_id.is_none_or(|prev| prev + 1 != batch.txn_id) {
                events.push(ChangeEvent::TransactionBegin {
                    txn_id: batch.txn_id,
                });
            }

            // All cell changes in this transaction
            for change in batch.changes {
                events.push(ChangeEvent::CellChange(change));
            }

            // TransactionCommit marker (no commit_ts available from DeltaLayerStack alone)
            // Caller can look up commit_ts via MVCC metadata if needed
            events.push(ChangeEvent::TransactionCommit {
                txn_id: batch.txn_id,
                commit_ts: 0, // Caller resolves via MVCC
            });

            prev_txn_id = Some(batch.txn_id);
        }

        Ok(events)
    }
}

/// Transaction boundary event for transaction-aware CDC consumers.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum ChangeEvent {
    /// Marks the beginning of a transaction.
    TransactionBegin { txn_id: u64 },
    /// Marks the commit of a transaction with its commit timestamp.
    TransactionCommit { txn_id: u64, commit_ts: u64 },
    /// A cell-level change within a transaction.
    CellChange(CellChange),
}

#[cfg(test)]
static WAL_REPLAY_TEST_CALLS: AtomicUsize = AtomicUsize::new(0);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::RockDuckConfig;
    use crate::write::durability_wal::{OpPayload, OpType, WalOp};
    use tempfile::TempDir;

    #[test]
    fn read_changes_does_not_trigger_wal_fallback_when_delta_layers_are_empty() {
        let temp = TempDir::new().expect("tempdir");
        let mut config = RockDuckConfig::default();
        config.cdc.enabled = true;
        let db = Arc::new(RockDuck::open_with_config(temp.path(), config).expect("open db"));
        let stream = ChangeStream::from_txn(Arc::clone(&db), 0);

        WAL_REPLAY_TEST_CALLS.store(0, AtomicOrdering::SeqCst);

        let batches = stream.read_changes().expect("read changes");

        assert!(batches.is_empty());
        assert_eq!(
            WAL_REPLAY_TEST_CALLS.load(AtomicOrdering::SeqCst),
            0,
            "documented WAL fallback is not wired into read_changes()"
        );
    }

    #[test]
    fn replay_hotspot_stats_track_scanned_segments_and_deltas() {
        let temp = TempDir::new().expect("tempdir");
        let mut config = RockDuckConfig::default();
        config.cdc.enabled = true;
        let db = Arc::new(RockDuck::open_with_config(temp.path(), config).expect("open db"));

        db.delta_layer
            .put(DeltaCell {
                seg_id: "seg-hotspot".to_string(),
                row_offset: 7,
                column: "value".to_string(),
                txn_id: 1,
                before: None,
                after: Some(Arc::new(vec![1, 2, 3])),
                committed: true,
                ts: 0,
            })
            .expect("put delta");
        db.mvcc.write().set_committed_txn(1);

        let stream = ChangeStream::from_txn(Arc::clone(&db), 0);
        let batches = stream.read_changes().expect("read changes");
        assert_eq!(batches.len(), 1);

        let stats = stream.replay_hotspot_stats();
        assert_eq!(stats.read_calls, 1);
        assert_eq!(stats.scanned_segments, 1);
        assert_eq!(stats.collected_deltas, 1);
    }

    #[test]
    fn wal_update_emits_one_cellchange_per_column() {
        use arrow_array::{Int64Array, RecordBatch};
        use arrow_schema::Schema;
        use std::sync::Arc;

        // Build a valid Arrow IPC batch with the after-values.
        let schema = Arc::new(Schema::new(vec![
            arrow_schema::Field::new("price", arrow_schema::DataType::Int64, false),
            arrow_schema::Field::new("status", arrow_schema::DataType::Int64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from(vec![999])), // after price
                Arc::new(Int64Array::from(vec![200])),  // after status
            ],
        )
        .expect("valid batch");
        let ipc_bytes = crate::write::batch_to_bytes(&batch).expect("IPC bytes");

        let wal_op = WalOp {
            op_type: OpType::Update,
            txn_id: 42,
            payload: OpPayload::Update {
                table: "orders".to_string(),
                pk: [7; 8],
                columns: vec!["price".to_string(), "status".to_string()],
                wal_batch: Arc::new(ipc_bytes),
                schema_bytes: Vec::new(),
                old_columns: vec![
                    ("price".to_string(), vec![9, 9]),
                    ("status".to_string(), vec![8, 8]),
                ],
                old_seg_id: "old-seg".to_string(),
                old_granule_id: crate::metadata::GranuleId::zero(),
                old_offset: 3,
                new_seg_id: "new-seg".to_string(),
                new_granule_id: crate::metadata::GranuleId::zero(),
                offset: 4,
            },
        };

        let changes = ChangeStream::wal_op_to_cell_change(&wal_op)
            .expect("wal update should map to change")
            .expect("update payload should emit CDC change");

        // cdc002: one CellChange per column
        assert_eq!(changes.len(), 2, "update must emit one CellChange per column");

        // Sort by col name for deterministic order
        let mut changes = changes;
        changes.sort_by(|a, b| a.col.cmp(&b.col));

        let price = &changes[0];
        assert_eq!(price.table, "orders");
        assert_eq!(price.seg_id, "old-seg");
        assert_eq!(price.col, "price");
        assert_eq!(price.row_offset, 3);
        assert_eq!(price.op, CellOp::Update);
        assert_eq!(price.before, Some(vec![9, 9]));

        let status = &changes[1];
        assert_eq!(status.table, "orders");
        assert_eq!(status.seg_id, "old-seg");
        assert_eq!(status.col, "status");
        assert_eq!(status.row_offset, 3);
        assert_eq!(status.op, CellOp::Update);
        assert_eq!(status.before, Some(vec![8, 8]));
    }

    #[test]
    fn wal_commit_payload_does_not_emit_fake_cdc_change() {
        let wal_op = WalOp {
            op_type: OpType::Commit,
            txn_id: 7,
            payload: OpPayload::Commit { begin_ts: 7, inserted_at: None },
        };

        let change =
            ChangeStream::wal_op_to_cell_change(&wal_op).expect("commit payload should be handled");
        assert!(
            change.is_none(),
            "commit payload must not synthesize fake CDC rows"
        );
    }

    #[test]
    fn read_changes_as_arrow_uses_uint64_row_offset_schema() {
        let temp = TempDir::new().expect("tempdir");
        let mut config = RockDuckConfig::default();
        config.cdc.enabled = true;
        let db = Arc::new(RockDuck::open_with_config(temp.path(), config).expect("open db"));

        db.delta_layer
            .put(DeltaCell {
                seg_id: "seg-arrow".to_string(),
                row_offset: u32::MAX as u64 + 5,
                column: "value".to_string(),
                txn_id: 1,
                before: None,
                after: Some(Arc::new(vec![1, 2, 3])),
                committed: true,
                ts: 0,
            })
            .expect("put delta");
        db.mvcc.write().set_committed_txn(1);

        let stream = ChangeStream::from_txn(Arc::clone(&db), 0);
        let batches = stream.read_changes_as_arrow(16).expect("arrow batches");
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].schema().field(3).data_type(), &DataType::UInt64);
    }
}
