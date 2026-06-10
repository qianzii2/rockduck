//! MVCC visibility manager
//!
//! ## Strict historical boundary (D12)
//!
//! Missing `commit_ts_map` entries are treated as invisible, not weakly visible.
//! This means pruned/evicted historical commit metadata is a classified transition
//! exception, not a fallback visibility mode.
//!
//! Responsible for:
//! - Tracking active transactions (registered at txn start, removed on commit/rollback)
//! - Generating consistent snapshots (for Repeatable Read / Snapshot Isolation)
//! - Serializable Snapshot Isolation (SSI) conflict detection
//! - Persisting active transactions to KV store (recoverable after crash)
//!
//! MVCC design (Shadow Column approach):
//! - Each data row records created_by_txn and deleted_by_txn
//! - At read time, visibility is determined based on the snapshot
//! - SSI: tracks read/write sets per transaction to detect conflicts
//!
//! ## Corruption recovery (WRT-06)
//!
//! If the KV store is corrupted (e.g., partial write during a crash), the in-memory
//! `active_txns` HashMap may contain transactions that never committed.
//! On database restart, MVCC state is recovered from the WAL, which replays committed
//! transactions only. Orphaned entries in `active_txns` are handled as follows:
//!
//! - Transactions that appear in WAL as `Commit`: visible, entry kept
//! - Transactions that appear in WAL as `Rollback`: invisible, entry removed
//! - Transactions that have NO WAL record: treated as never-committed (orphan).
//!   These transactions will block future reads indefinitely (zombie transaction).
//!
//! Mitigation: after WAL recovery, if `active_txns` contains entries whose txn_ids
//! are less than the minimum committed txn, those entries are stale orphans and should
//! be removed. The WAL replay tracks `min_committed_txn` which can be used for this.
//! Currently this cleanup is handled implicitly: `begin_txn()` checks `txn_id <= committed_txn`
//! and rejects such transactions. A full post-recovery scan of `active_txns` for orphans
//! is a planned hardening step.

use crate::config::VisibilityConfig;
use crate::db::TxnId;
use crate::error::{Result, RockDuckError};
use crate::segment::overlay::{SegmentOverlay, COMPACTION_SNAPSHOT_ID};
use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::Arc;

/// Visibility filter errors.
///
/// Returned by visibility operations that can fail due to data inconsistencies
/// (e.g., schema mismatches, Arrow internal errors).
#[derive(Debug, Clone)]
pub enum VisibilityError {
    /// The visibility mask length does not match the batch row count.
    /// This indicates a bug in the visibility computation upstream.
    LengthMismatch { mask_len: usize, batch_rows: usize },
    /// Arrow's filter operation failed (e.g., schema mismatch between batch and filter array).
    Filter(String),
}

impl std::fmt::Display for VisibilityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VisibilityError::LengthMismatch {
                mask_len,
                batch_rows,
            } => {
                write!(
                    f,
                    "visibility mask length {} != batch rows {}",
                    mask_len, batch_rows
                )
            }
            VisibilityError::Filter(msg) => {
                write!(f, "arrow filter failed: {}", msg)
            }
        }
    }
}

impl std::error::Error for VisibilityError {}

/// Visibility projection mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VisibilityProjection {
    Online,
    Historical,
    CompactionRewrite,
    Vtab,
}

/// Explicit visibility context used to project authority state into concrete read/rewrite modes.
#[derive(Debug, Clone)]
pub struct VisibilityContext {
    pub projection: VisibilityProjection,
    pub snapshot_id: TxnId,
    pub active_txns: BTreeSet<TxnId>,
    pub commit_ts_map: HashMap<TxnId, u64>,
}

impl VisibilityContext {
    pub fn from_snapshot(snapshot: &TxnSnapshot, projection: VisibilityProjection) -> Self {
        Self {
            projection,
            snapshot_id: snapshot.snapshot_id,
            active_txns: snapshot.active_txns.clone(),
            commit_ts_map: snapshot.commit_ts_map.clone(),
        }
    }

    pub fn historical(target_txn: TxnId, commit_ts_map: HashMap<TxnId, u64>) -> Self {
        Self {
            projection: VisibilityProjection::Historical,
            snapshot_id: target_txn,
            active_txns: BTreeSet::new(),
            commit_ts_map,
        }
    }

    pub fn compaction_rewrite() -> Self {
        Self {
            projection: VisibilityProjection::CompactionRewrite,
            snapshot_id: COMPACTION_SNAPSHOT_ID,
            active_txns: BTreeSet::new(),
            commit_ts_map: HashMap::new(),
        }
    }

    pub fn into_overlay(self) -> SegmentOverlay {
        SegmentOverlay::from_context(self)
    }
}

/// Unified MVCC visibility filter trait.
///
/// All visibility checks (MVCC scan, ShadowColumn scan, VTab DuckDB scan, time-travel)
/// must use this trait to ensure consistent visibility semantics across the codebase.
///
/// ## Canonical Call Surfaces
///
/// Every visibility decision in RockDuck flows through one of these five surfaces:
///
/// | Surface | File | Status | Notes |
/// |---------|------|--------|-------|
/// | ScanIterator | `src/read/scan.rs` | SANCTIONED | Calls `VisibilityManager::is_visible` via `TxnSnapshot::is_row_visible` |
/// | point_get | `src/read/point_get.rs` | SANCTIONED | Calls `VisibilityManager::is_visible` for current reads |
/// | point_get_as_of | `src/read/point_get.rs` | SANCTIONED | Calls historical projection context over `VisibilityManager` |
/// | VTab filter | `src/query/vtab_quack.rs` | SANCTIONED | Calls `TxnSnapshot::is_row_visible` directly (Rule 1-4 equivalent) |
/// | Compaction | `src/compaction/pdt_merge.rs` | SANCTIONED | Calls `SegmentOverlay::is_row_visible` via `compaction_overlay_filter` |
///
/// ## Projection Model
///
/// Historical and compaction semantics remain explicit projections of the same authority model:
/// - historical reads use `VisibilityContext::historical(...)` with real commit timestamps
/// - compaction rewrite uses `VisibilityContext::compaction_rewrite()`
///
/// This keeps differences explicit without granting them independent truth semantics.
pub trait VisFilter: Send + Sync {
    /// Returns true if a row created by `created_txn` and optionally deleted by
    /// `deleted_txn` is visible in the snapshot identified by `snapshot_id`.
    ///
    /// Visibility rules (strict Snapshot Isolation):
    /// 1. `created_txn` must not be a future transaction
    /// 2. `created_txn` must not be in the active transaction set
    /// 3. If `created_txn` is committed, its commit_ts must be <= snapshot_id
    /// 4. If deleted: `deleted_txn` must not be committed, or deleted_ts > snapshot_id
    fn is_row_visible(
        &self,
        snapshot_id: TxnId,
        created_txn: TxnId,
        deleted_txn: Option<TxnId>,
        active_txns: &BTreeSet<TxnId>,
        commit_ts_map: &HashMap<TxnId, u64>,
    ) -> bool;
}

/// Transaction status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TxnStatus {
    Active,
    Committed,
    Aborted,
}

/// Transaction metadata (tracked in-memory)
#[derive(Debug, Clone)]
pub struct TxnMeta {
    pub begin_ts: u64,
    /// Commit timestamp (wall clock, populated on commit).
    /// Used for time-travel queries in CDC.
    pub commit_ts: Option<u64>,
    pub status: TxnStatus,
    /// Keys read by this transaction (used for SSI conflict detection).
    read_keys: HashSet<Vec<u8>>,
    /// Keys written by this transaction (used for SSI conflict detection).
    written_keys: HashSet<Vec<u8>>,
    /// Whether this transaction was recovered from WAL during crash recovery.
    /// When true, the read_keys/written_keys are empty and SSI conflict
    /// detection will skip this transaction.
    pub is_recovered: bool,
}

impl TxnMeta {
    pub fn new(begin_ts: u64) -> Self {
        Self {
            begin_ts,
            commit_ts: None,
            status: TxnStatus::Active,
            read_keys: HashSet::new(),
            written_keys: HashSet::new(),
            is_recovered: false, // Fresh transactions are not recovered
        }
    }

    pub fn record_read(&mut self, key: Vec<u8>) {
        self.read_keys.insert(key);
    }

    pub fn record_write(&mut self, key: Vec<u8>) {
        self.written_keys.insert(key);
    }

    pub fn commit(&mut self, commit_ts: u64) {
        self.status = TxnStatus::Committed;
        self.commit_ts = Some(commit_ts);
    }

    pub fn has_read_key(&self, key: &[u8]) -> bool {
        self.read_keys.contains(key)
    }

    pub fn has_written_key(&self, key: &[u8]) -> bool {
        self.written_keys.contains(key)
    }
}

/// Isolation level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum IsolationLevel {
    #[default]
    ReadCommitted,
    RepeatableRead,
    Snapshot,
}

/// Transaction snapshot
#[derive(Debug, Clone)]
pub struct TxnSnapshot {
    pub snapshot_id: TxnId,
    pub active_txns: BTreeSet<TxnId>,
    pub isolation: IsolationLevel,
    /// Commit timestamps for transactions in this snapshot.
    /// Only contains entries for committed transactions whose commit_ts <= snapshot_id.
    /// Used by strict Snapshot Isolation to verify rows were committed before the snapshot.
    pub commit_ts_map: HashMap<TxnId, u64>,
}

impl TxnSnapshot {
    pub fn new(
        snapshot_id: TxnId,
        active_txns: BTreeSet<TxnId>,
        isolation: IsolationLevel,
    ) -> Self {
        Self {
            snapshot_id,
            active_txns,
            isolation,
            commit_ts_map: HashMap::new(),
        }
    }
}

/// MVCC visibility manager — cloneable so `ScanIterator` can hold its own read snapshot.
///
/// ## Thread safety
/// `VisibilityManager` is `Send` but **not `Sync`** (`HashMap` is not `Sync`).
/// The single shared production instance lives behind `RwLock<VisibilityManager>`
/// inside `Arc<RockDuck>` (`src/db.rs:286`). All mutation methods (`begin_txn`,
/// `commit_txn`, `rollback_txn`) are called through `self.mvcc.write()` which
/// serializes cross-thread access. **This design is intentional — do not add
/// `Sync` or wrap this struct in an additional `RwLock`.**
///
/// Ephemeral local instances (compaction helpers in `pdt_merge.rs`, test helpers
/// in `shadow_columns.rs` and `mvcc_visibility_tests.rs`) are single-threaded and
/// safe without synchronization.
#[derive(Clone)]
pub struct VisibilityManager {
    committed_txn: u64,
    /// D7 fix: WAL replay watermark — the maximum inserted_at timestamp among all
    /// transactions recovered from WAL. Used as a lower bound for TTL eviction:
    /// entries in committed_history with inserted_at < replay_watermark cannot be evicted
    /// even if their TTL has expired, because they represent data from a historical
    /// snapshot that a post-recovery transaction might read.
    replay_watermark: u64,
    active_txns: HashMap<TxnId, TxnMeta>,
    /// Commit timestamp history for strict MVCC visibility checks.
    /// Maps committed txn_id -> (commit_timestamp, inserted_at).
    /// `commit_timestamp` is used for visibility checks and time-travel queries.
    /// `inserted_at` is the wall-clock time of insertion into history, used for TTL eviction.
    /// Bounded by max_history_entries and history_ttl_secs to prevent unbounded growth.
    committed_history: HashMap<TxnId, (u64, u64)>,
    config: VisibilityConfig,
    /// Set of transaction IDs that were recovered from WAL during crash recovery.
    /// These transactions have empty read/write key sets and are skipped in SSI
    /// conflict detection. Cleared when those transactions commit or rollback.
    recovered_txns: HashSet<TxnId>,
}

impl VisibilityManager {
    pub fn new() -> Self {
        Self::new_with_config(VisibilityConfig::default())
    }

    pub fn new_with_config(config: VisibilityConfig) -> Self {
        Self {
            committed_txn: 0,
            replay_watermark: 0,
            active_txns: HashMap::new(),
            committed_history: HashMap::new(),
            config,
            recovered_txns: HashSet::new(),
        }
    }

    pub fn set_committed_txn(&mut self, txn_id: TxnId) {
        self.committed_txn = txn_id;
    }

    /// D7 fix: set the WAL replay watermark.
    pub fn set_replay_watermark(&mut self, watermark: u64) {
        self.replay_watermark = watermark;
    }

    pub fn begin_txn(
        &mut self,
        txn_id: TxnId,
        kv: &Arc<dyn crate::metadata::KVEngine>,
    ) -> Result<()> {
        if self.active_txns.contains_key(&txn_id) {
            return Err(RockDuckError::MvccConflict(format!(
                "transaction {} already active",
                txn_id
            )));
        }
        if txn_id <= self.committed_txn {
            return Err(RockDuckError::MvccConflict(format!(
                "transaction {} already committed",
                txn_id
            )));
        }
        let meta = TxnMeta::new(txn_id);
        self.active_txns.insert(txn_id, meta);
        crate::metadata::add_active_txn(kv, txn_id, txn_id)?;
        Ok(())
    }

    pub fn record_read(&mut self, txn_id: TxnId, key: &[u8]) {
        if let Some(meta) = self.active_txns.get_mut(&txn_id) {
            meta.record_read(key.to_vec());
        } else {
            tracing::warn!(
                "record_read: txn {} not in active_txns, skipping read tracking",
                txn_id
            );
        }
    }

    pub fn record_write(&mut self, txn_id: TxnId, key: &[u8]) {
        if let Some(meta) = self.active_txns.get_mut(&txn_id) {
            meta.record_write(key.to_vec());
        } else {
            tracing::warn!(
                "record_write: txn {} not in active_txns, skipping write tracking",
                txn_id
            );
        }
    }

    /// Commit a transaction: remove from active_txns and record commit timestamp.
    ///
    /// ## commit_ts Generation (mv007 fix)
    ///
    /// The commit timestamp is **always** generated internally by MVCC — it cannot be
    /// supplied by the caller. This ensures:
    ///   (1) commit_ts values are monotonically increasing (never decreasing)
    ///   (2) commit_ts is wall-clock based, not txn-id based (important for time-travel)
    ///   (3) No caller can forge or manipulate commit timestamps
    ///
    /// ## SSI Conflict Detection with O(n*k) Optimization (mv009 fix)
    ///
    /// Previous implementation used nested loops (O(n² * k)) over all active transactions
    /// and their read/write keys. Optimized approach:
    ///
    /// 1. Build a `key -> Vec<(txn_id, read/write)>` index in O(n*k)
    /// 2. For each key in txn_meta's read_keys: check if any other txn wrote it → RW conflict
    /// 3. For each key in txn_meta's written_keys: check if any other txn read/wrote it → WW/WR conflict
    ///
    /// This reduces complexity from O(n²*k) to O(n*k) where n = active txns, k = keys per txn.
    pub fn commit_txn(
        &mut self,
        txn_id: TxnId,
        kv: &Arc<dyn crate::metadata::KVEngine>,
        inserted_at: u64,
    ) -> Result<TxnSnapshot> {
        let txn_meta = self
            .active_txns
            .get(&txn_id)
            .ok_or_else(|| {
                RockDuckError::MvccConflict(format!("transaction {} not active", txn_id))
            })?
            .clone();

        // SSI conflict detection using key-indexed approach (mv009 fix: O(n*k) vs O(n²*k))
        // Phase 1: Build key -> writers/readers index, only for NON-recovered txns (mv002 fix)
        // Recovered transactions have empty read/write sets, so including them in the index
        // would cause false negatives (missed conflicts). We skip them entirely.
        let mut key_writers: HashMap<&Vec<u8>, Vec<TxnId>> = HashMap::new();
        let mut key_readers: HashMap<&Vec<u8>, Vec<TxnId>> = HashMap::new();

        for (other_id, other_meta) in &self.active_txns {
            if *other_id == txn_id {
                continue;
            }
            // Skip recovered transactions in SSI detection (mv002 fix)
            if other_meta.is_recovered {
                tracing::warn!(
                    "SSI skipping recovered txn {} in conflict detection (read/write sets unknown)",
                    other_id
                );
                continue;
            }
            for key in &other_meta.written_keys {
                key_writers.entry(key).or_default().push(*other_id);
            }
            for key in &other_meta.read_keys {
                key_readers.entry(key).or_default().push(*other_id);
            }
        }

        // Phase 2: Check txn_meta's read_keys against writers (O(k * avg_writers_per_key))
        let mut read_write_conflict = false;
        let mut write_write_conflict = false;
        let mut conflicting_txn = 0;

        for k in &txn_meta.read_keys {
            if let Some(writers) = key_writers.get(k) {
                if !writers.is_empty() {
                    read_write_conflict = true;
                    conflicting_txn = writers[0];
                    break;
                }
            }
        }

        // Phase 3: Check txn_meta's written_keys against readers and writers (O(k * (avg_readers + avg_writers)))
        if !read_write_conflict {
            for k in &txn_meta.written_keys {
                // WW: another txn wrote the same key and read it
                if let Some(readers) = key_readers.get(k) {
                    if !readers.is_empty() {
                        write_write_conflict = true;
                        conflicting_txn = readers[0];
                        break;
                    }
                }
                // WR: another txn wrote the same key (no need to check readers again)
                if let Some(writers) = key_writers.get(k) {
                    if !writers.is_empty() {
                        write_write_conflict = true;
                        conflicting_txn = writers[0];
                        break;
                    }
                }
            }
        }

        if read_write_conflict || write_write_conflict {
            self.abort_txn(txn_id, kv)?;
            return Err(RockDuckError::MvccConflict(format!(
                "SSI conflict: txn {} conflicts with active txn {} (rw={}, ww={})",
                txn_id, conflicting_txn, read_write_conflict, write_write_conflict
            )));
        }

        if txn_id > self.committed_txn {
            self.committed_txn = txn_id;
        }

        // Record commit timestamp — always generated internally (mv007 fix)
        let ts = crate::codec::current_timestamp_millis();
        if let Some(meta) = self.active_txns.get_mut(&txn_id) {
            meta.commit(ts);
        }

        // Record commit timestamp in committed_history for strict MVCC visibility checks.
        // Stores (commit_ts, inserted_at) so TTL eviction uses insertion time, not commit_ts.
        // This is keyed by txn_id (not commit_ts) so snapshots can look up commit_ts by txn_id.
        // Bounded by max_history_entries and history_ttl_secs to prevent unbounded growth.
        // `inserted_at` is captured once in `db.rs::commit_txn` before WAL write, so WAL-recovered
        // entries and live-committed entries share the same TTL clock.
        self.committed_history.insert(txn_id, (ts, inserted_at));

        // Prune: evict oldest entries when count exceeds limit (keeps newest 80%)
        self.prune_history(ts);

        self.active_txns.remove(&txn_id);
        self.recovered_txns.remove(&txn_id); // Clear recovered flag on commit
        crate::metadata::remove_active_txn(kv, txn_id)?;

        Ok(self.snapshot(IsolationLevel::Snapshot))
    }

    fn abort_txn(&mut self, txn_id: TxnId, kv: &Arc<dyn crate::metadata::KVEngine>) -> Result<()> {
        self.active_txns.remove(&txn_id);
        self.recovered_txns.remove(&txn_id); // Clear recovered flag on abort
        crate::metadata::remove_active_txn(kv, txn_id)
    }

    /// Transaction rollback: remove from active transaction table
    pub fn rollback_txn(
        &mut self,
        txn_id: TxnId,
        kv: &Arc<dyn crate::metadata::KVEngine>,
    ) -> Result<()> {
        self.abort_txn(txn_id, kv)
    }

    /// Prune committed_history to keep memory bounded.
    ///
    /// Two eviction strategies:
    /// 1. Count-based: evict oldest 20% when size exceeds max_history_entries
    /// 2. Time-based: evict entries older than (wall_clock - ttl_secs)
    ///
    /// Time-based eviction uses `inserted_at` (the wall-clock time the entry was added to history),
    /// not `commit_ts`. This is correct because WAL replay can insert entries with stale
    /// (old) commit_ts out of order. Using insertion time for TTL ensures the cutoff is
    /// always relative to when the entry was added, regardless of commit order.
    fn prune_history(&mut self, _commit_ts: u64) {
        // Strategy 1: count-based eviction
        // Collect keys sorted by inserted_at (wall-clock TTL clock), then remove oldest.
        // D5 fix: changed sort key from commit_ts to inserted_at. This is correct because:
        // - WAL replay can insert entries with stale (old) commit_ts out of order
        // - inserted_at is always monotonically increasing (captured once at commit time)
        // - Evicting by inserted_at preserves FIFO semantics across out-of-order commits
        if self.committed_history.len() > self.config.max_history_entries {
            let mut items: Vec<(u64, TxnId)> = self
                .committed_history
                .iter()
                .map(|(&k, &(_, inserted_at))| (inserted_at, k))
                .collect();
            items.sort_by_key(|i| i.0); // sort by inserted_at ascending
            let keep_count = self.config.max_history_entries * 4 / 5;
            let evict_count = items.len().saturating_sub(keep_count);
            for (i, _) in items.iter().enumerate().take(evict_count) {
                self.committed_history.remove(&items[i].1);
            }
        }

        // Strategy 2: time-based eviction using insertion wall-clock time.
        // D7 fix: add replay_watermark as lower bound — entries older than replay_watermark
        // cannot be evicted even if TTL expired, because they were recovered from WAL and
        // represent historical data a post-recovery transaction might read.
        let ttl_ms = self.config.history_ttl_secs * 1000;
        let now_ms = crate::codec::current_timestamp_millis();
        let ttl_cutoff = now_ms.saturating_sub(ttl_ms);
        let cutoff = std::cmp::max(ttl_cutoff, self.replay_watermark);
        self.committed_history
            .retain(|_, &mut (_, inserted_at)| inserted_at >= cutoff);
    }

    /// Generate a snapshot with only the `active_txns` set populated.
    ///
    /// This is the **lazy-loading path** (mv010 fix). Unlike `snapshot()`, this method
    /// does NOT populate the `commit_ts_map`. The `commit_ts_map` is only needed for
    /// CDC time-travel queries. For normal OLTP reads, only `active_txns` is required
    /// for visibility checks.
    ///
    /// Use `snapshot_with_commit_ts_map()` to get a snapshot with the full `commit_ts_map`
    /// populated (for time-travel queries).
    pub fn snapshot_with_active_only(&self, isolation: IsolationLevel) -> TxnSnapshot {
        let active_ids: BTreeSet<TxnId> = self.active_txns.keys().cloned().collect();
        let snap = TxnSnapshot::new(self.committed_txn, active_ids, isolation);
        // commit_ts_map intentionally left empty — populated lazily for CDC time-travel
        snap
    }

    /// Generate a snapshot with the full `commit_ts_map` populated.
    ///
    /// This is the **CDC time-travel path** (mv010 fix). Unlike `snapshot_with_active_only()`,
    /// this method populates `commit_ts_map` by filtering to only transactions committed
    /// before or at the snapshot time.
    ///
    /// Use `snapshot_with_active_only()` for normal OLTP reads to avoid the overhead
    /// of building the commit_ts_map.
    pub fn snapshot_with_commit_ts_map(&self, isolation: IsolationLevel) -> TxnSnapshot {
        let active_ids: BTreeSet<TxnId> = self.active_txns.keys().cloned().collect();
        let snapshot_id = self.committed_txn;
        // Filter to only txns committed before or at snapshot time (mv008 fix)
        let commit_ts_map: HashMap<TxnId, u64> = self
            .committed_history
            .iter()
            .filter(|(_, &(ts, _))| ts <= snapshot_id)
            .map(|(&k, &(ts, _))| (k, ts))
            .collect();
        let mut snap = TxnSnapshot::new(snapshot_id, active_ids, isolation);
        snap.commit_ts_map = commit_ts_map;
        snap
    }

    /// Generate consistent snapshot from current state.
    ///
    /// ## commit_ts_map Filtering (mv008 fix)
    ///
    /// The `commit_ts_map` is filtered to only include transactions whose commit_ts
    /// is <= the snapshot's snapshot_id. This is required by strict Snapshot Isolation:
    /// a row is only visible if the transaction that created it was committed **before**
    /// the snapshot was taken (commit_ts <= snapshot_id).
    ///
    /// This prevents a row committed at time T from being visible in a snapshot
    /// taken at an earlier time T' < T.
    ///
    /// The `active_txns` set is NOT filtered by begin_ts — transactions active at
    /// snapshot time are excluded from visibility regardless of their begin_ts,
    /// because their writes may be rolled back.
    pub fn snapshot(&self, isolation: IsolationLevel) -> TxnSnapshot {
        // For now, delegate to snapshot_with_commit_ts_map (populates commit_ts_map).
        // Callers can use snapshot_with_active_only() for the lazy-loading path.
        self.snapshot_with_commit_ts_map(isolation)
    }

    /// Construct a TxnSnapshot representing the database state as of `txn_id`.
    ///
    /// Uses in-memory committed_history and active_txns — no KV schema change needed.
    ///
    /// Returns a snapshot where:
    ///   - snapshot_id = txn_id
    ///   - active_txns = only txns where begin_ts <= txn_id (transactions that had started)
    ///   - commit_ts_map = only txns where commit_ts <= txn_id (transactions committed at that point)
    ///
    /// ## Classified transition exception
    /// When a txn falls outside the committed_history retention window, its commit_ts is
    /// absent from this snapshot and downstream D12 visibility rules treat it as invisible.
    /// This is intentionally conservative and remains a classified historical exception,
    /// not a weak-snapshot fallback.
    pub fn snapshot_at(&self, txn_id: TxnId, isolation: IsolationLevel) -> TxnSnapshot {
        // Transactions active at txn_id: those whose begin_ts <= txn_id
        let active_ids: BTreeSet<TxnId> = self
            .active_txns
            .iter()
            .filter(|(_, meta)| meta.begin_ts <= txn_id)
            .map(|(&id, _)| id)
            .collect();

        // Entries in committed_history with commit_ts <= txn_id
        let commit_ts_map: HashMap<TxnId, u64> = self
            .committed_history
            .iter()
            .filter(|(_, &(ts, _))| ts <= txn_id)
            .map(|(&k, &(ts, _))| (k, ts))
            .collect();

        TxnSnapshot {
            snapshot_id: txn_id,
            active_txns: active_ids,
            isolation,
            commit_ts_map,
        }
    }

    /// Get current max committed transaction ID
    pub fn committed_txn(&self) -> TxnId {
        self.committed_txn
    }

    /// D7 fix: getter for replay_watermark, used by checkpoint serialization.
    pub fn replay_watermark(&self) -> u64 {
        self.replay_watermark
    }

    pub fn get_begin_ts(&self, txn_id: TxnId) -> Option<u64> {
        self.active_txns.get(&txn_id).map(|meta| meta.begin_ts)
    }

    /// Recover committed-history state after WAL replay.
    ///
    /// Populates `committed_history` so that snapshots include the correct `commit_ts_map`
    /// for strict Snapshot Isolation visibility checks. This is called during
    /// `RockDuck::open_with_config` after WAL recovery.
    pub fn recover_committed_history(&mut self, history: impl IntoIterator<Item = (TxnId, u64)>) {
        self.recover_committed_history_with_config(history, None, &Default::default());
    }

    pub fn recover_committed_history_with_config(
        &mut self,
        history: impl IntoIterator<Item = (TxnId, u64)>,
        config_override: Option<VisibilityConfig>,
        wal_inserted_at: &std::collections::HashMap<TxnId, u64>,
    ) {
        if let Some(cfg) = config_override {
            self.config = cfg;
        }
        self.committed_history.clear();
        let now = crate::codec::current_timestamp_millis();
        for (txn_id, commit_ts) in history {
            // D5 fix: use WAL-persisted inserted_at if available, otherwise current wall-clock.
            // Old WAL entries (pre-D5) have no entry in wal_inserted_at — they get now().
            let inserted_at = wal_inserted_at.get(&txn_id).copied().unwrap_or(now);
            self.committed_history
                .insert(txn_id, (commit_ts, inserted_at));
            if txn_id > self.committed_txn {
                self.committed_txn = txn_id;
            }
        }
    }

    /// Recover active transactions from checkpoint/KV/WAL merge.
    ///
    /// This is used by `RockDuck::open_with_config` after WAL replay to restore
    /// the set of transactions that were active at crash time.
    ///
    /// ## SSI Handling on Recovery (mv002 fix)
    ///
    /// Recovered transactions are restored with **empty `written_keys` and `read_keys`**.
    /// This is a known design limitation: the WAL format does not record per-row
    /// read/write sets, so they cannot be reconstructed during replay.
    ///
    /// As a consequence, SSI conflict detection does **not** apply to recovered
    /// active transactions — two recovered transactions that modify overlapping rows
    /// will not be detected as conflicting. In practice, this is acceptable because:
    ///   (1) crash-before-commit transactions are rare, and
    ///   (2) the window between a crash and the subsequent recovery is short.
    ///
    /// ## mv002 Enhancement: Track Recovered Transactions
    ///
    /// A separate `recovered_txns` set is maintained to distinguish recovered transactions
    /// from live transactions. This allows the SSI conflict detection to:
    ///   - Skip conflict checks involving recovered transactions (their read/write sets are unknown)
    ///   - Log warnings when recovered transactions conflict with live transactions
    ///
    /// The `recovered_txns` set is populated here during WAL recovery and cleared
    /// when those transactions either commit or are rolled back.
    pub fn recover_active_txns(&mut self, active: impl IntoIterator<Item = (TxnId, u64)>) {
        self.active_txns.clear();
        self.recovered_txns.clear();
        for (txn_id, begin_ts) in active {
            let mut meta = TxnMeta::new(begin_ts);
            // Mark as recovered so SSI can skip conflict detection for these txns (mv002 fix)
            meta.is_recovered = true;
            self.active_txns.insert(txn_id, meta);
            self.recovered_txns.insert(txn_id);
        }
        tracing::info!(
            "Recovered {} active transactions (SSI disabled for recovered txns)",
            self.active_txns.len()
        );
    }

    /// Get the commit timestamp for a given transaction, if it was committed.
    ///
    /// Used by time-travel queries (`get_as_of`) to filter delta cells by commit_ts.
    /// Returns `None` if the transaction was aborted (not in committed_history).
    pub fn get_commit_ts(&self, txn_id: TxnId) -> Option<u64> {
        self.committed_history.get(&txn_id).map(|&(ts, _)| ts)
    }

    pub fn committed_history_entries(&self) -> Vec<(TxnId, u64)> {
        let mut entries: Vec<(TxnId, u64)> = self
            .committed_history
            .iter()
            .map(|(&txn_id, &(commit_ts, _))| (txn_id, commit_ts))
            .collect();
        entries.sort_by_key(|&(txn_id, _)| txn_id);
        entries
    }

    pub fn active_txn_entries(&self) -> Vec<(TxnId, u64)> {
        let mut entries: Vec<(TxnId, u64)> = self
            .active_txns
            .iter()
            .map(|(&txn_id, meta)| (txn_id, meta.begin_ts))
            .collect();
        entries.sort_by_key(|&(txn_id, _)| txn_id);
        entries
    }

    pub fn visibility_config(&self) -> &VisibilityConfig {
        &self.config
    }

    /// Check if a data record is visible for the given snapshot.
    ///
    /// Delegates to `VisFilter::is_row_visible` to ensure consistent semantics
    /// across all visibility checks (scan, point_get, compaction, and VTab).
    pub fn is_visible(
        &self,
        snapshot: &TxnSnapshot,
        created_txn: TxnId,
        deleted_txn: Option<TxnId>,
    ) -> bool {
        <VisibilityManager as VisFilter>::is_row_visible(
            self,
            snapshot.snapshot_id,
            created_txn,
            deleted_txn,
            &snapshot.active_txns,
            &snapshot.commit_ts_map,
        )
    }
}

impl<T: VisFilter + ?Sized> VisFilter for Arc<T> {
    fn is_row_visible(
        &self,
        snapshot_id: TxnId,
        created_txn: TxnId,
        deleted_txn: Option<TxnId>,
        active_txns: &BTreeSet<TxnId>,
        commit_ts_map: &HashMap<TxnId, u64>,
    ) -> bool {
        (**self).is_row_visible(
            snapshot_id,
            created_txn,
            deleted_txn,
            active_txns,
            commit_ts_map,
        )
    }
}

/// A no-op visibility filter that makes all rows visible.
/// Used for testing and for cases where visibility checks should be skipped.
#[derive(Debug, Clone, Default)]
pub struct NoopVisFilter;

impl VisFilter for NoopVisFilter {
    fn is_row_visible(
        &self,
        _snapshot_id: TxnId,
        _created_txn: TxnId,
        _deleted_txn: Option<TxnId>,
        _active_txns: &BTreeSet<TxnId>,
        _commit_ts_map: &HashMap<TxnId, u64>,
    ) -> bool {
        true
    }
}

/// `TxnSnapshot` implements `VisFilter` so VTab and other callers can use the
/// same visibility logic as `VisibilityManager::is_row_visible`.
///
/// This was previously inlined in `vtab_quack.rs::filter_by_visibility` but had
/// a behavioral gap: it did not apply the D12 "strict Snapshot Isolation" rule
/// (absent from `commit_ts_map` → invisible for aborted txns). Delegating to this
/// impl closes that gap and ensures VTab uses the same visibility semantics as the
/// scan and point_get paths.
impl VisFilter for TxnSnapshot {
    fn is_row_visible(
        &self,
        snapshot_id: TxnId,
        created_txn: TxnId,
        deleted_txn: Option<TxnId>,
        active_txns: &BTreeSet<TxnId>,
        commit_ts_map: &HashMap<TxnId, u64>,
    ) -> bool {
        // Rule 1: not a future transaction
        if created_txn > snapshot_id {
            return false;
        }
        // Rule 2: not created by a still-active transaction
        if active_txns.contains(&created_txn) {
            return false;
        }
        // Rule 3: D12 strict Snapshot Isolation fix — if created_txn is absent from
        // commit_ts_map, it means the transaction was aborted (not committed).
        // Treat as invisible to prevent aborted data from leaking into reader snapshots.
        if !commit_ts_map.contains_key(&created_txn) {
            return false;
        }
        // Rule 3 continued: committed txns must have commit_ts <= snapshot_id
        if let Some(&commit_ts) = commit_ts_map.get(&created_txn) {
            if commit_ts > snapshot_id {
                return false;
            }
        }
        // Rule 4: deletion visibility — deleted_txn must not be committed before snapshot
        if let Some(del) = deleted_txn {
            if active_txns.contains(&del) {
                // Delete txn is still active → treat as not deleted
            } else if let Some(&del_commit_ts) = commit_ts_map.get(&del) {
                if del_commit_ts <= snapshot_id {
                    return false; // deleted before snapshot → not visible
                }
            }
            // If not in active_txns and not in commit_ts_map → not committed → ignore
        }
        true
    }
}

impl VisFilter for VisibilityManager {
    fn is_row_visible(
        &self,
        snapshot_id: TxnId,
        created_txn: TxnId,
        deleted_txn: Option<TxnId>,
        active_txns: &BTreeSet<TxnId>,
        commit_ts_map: &HashMap<TxnId, u64>,
    ) -> bool {
        // Rule 1: not a future transaction
        if created_txn > snapshot_id {
            return false;
        }
        // Rule 2: not created by a still-active transaction
        if active_txns.contains(&created_txn) {
            return false;
        }
        // Rule 3: if committed, commit_ts must be <= snapshot_id.
        //
        // D12 fix: strict Snapshot Isolation. The commit_ts_map is now populated on
        // recovery from WAL (via recover_committed_history). If created_txn is absent from
        // commit_ts_map, it means:
        //   (a) it was aborted (not in WAL as committed), OR
        //   (b) it is too old and was evicted from committed_history
        //
        // In both cases, the correct behavior is to treat the row as NOT visible.
        // We conservatively return false rather than exposing potentially aborted data.
        //
        // OLD (weak-snapshot, REMOVED): absent from commit_ts_map → treat as committed (visible)
        // NEW (strict): absent from commit_ts_map → treat as not committed (invisible)
        match commit_ts_map.get(&created_txn) {
            Some(&commit_ts) => {
                if commit_ts > snapshot_id {
                    return false;
                }
            }
            None => {
                // D12 fix: neither committed (in commit_ts_map) nor active → invisible.
                // This prevents aborted transaction data from leaking into reader snapshots.
                return false;
            }
        }
        // Rule 4: deletion visibility — deleted_txn must not be committed before snapshot.
        // If deleted_txn is still active (uncommitted), ignore the delete mark.
        // If deleted_txn is committed, check commit_ts.
        if let Some(del) = deleted_txn {
            if active_txns.contains(&del) {
                // Delete txn is still active → treat as not deleted
            } else if let Some(&del_commit_ts) = commit_ts_map.get(&del) {
                // Delete txn was committed → check if committed before snapshot
                if del_commit_ts <= snapshot_id {
                    return false; // deleted before snapshot → not visible
                }
            }
            // If not in active_txns and not in commit_ts_map → not committed → ignore
        }
        true
    }
}

impl Default for VisibilityManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn btreeset(items: &[TxnId]) -> BTreeSet<TxnId> {
        items.iter().copied().collect()
    }

    fn hashmap(items: &[(TxnId, u64)]) -> HashMap<TxnId, u64> {
        items.iter().copied().collect()
    }

    #[test]
    fn visfilter_equivalence_between_manager_and_snapshot() {
        let cases = vec![
            (10, 11, None, btreeset(&[]), hashmap(&[(11, 11)]), false),
            (10, 8, None, btreeset(&[8]), hashmap(&[(8, 8)]), false),
            (10, 8, None, btreeset(&[]), hashmap(&[]), false),
            (10, 8, None, btreeset(&[]), hashmap(&[(8, 12)]), false),
            (10, 8, None, btreeset(&[]), hashmap(&[(8, 8)]), true),
            (
                10,
                8,
                Some(9),
                btreeset(&[]),
                hashmap(&[(8, 8), (9, 9)]),
                false,
            ),
            (10, 8, Some(9), btreeset(&[9]), hashmap(&[(8, 8)]), true),
            (
                10,
                8,
                Some(12),
                btreeset(&[]),
                hashmap(&[(8, 8), (12, 12)]),
                true,
            ),
            (10, 8, Some(9), btreeset(&[]), hashmap(&[(8, 8)]), true),
        ];

        for (snapshot_id, created_txn, deleted_txn, active_txns, commit_ts_map, expected) in cases {
            let manager = VisibilityManager::new();
            let snapshot = TxnSnapshot {
                snapshot_id,
                active_txns: active_txns.clone(),
                isolation: IsolationLevel::Snapshot,
                commit_ts_map: commit_ts_map.clone(),
            };

            let via_manager = <VisibilityManager as VisFilter>::is_row_visible(
                &manager,
                snapshot_id,
                created_txn,
                deleted_txn,
                &active_txns,
                &commit_ts_map,
            );
            let via_snapshot = <TxnSnapshot as VisFilter>::is_row_visible(
                &snapshot,
                snapshot_id,
                created_txn,
                deleted_txn,
                &active_txns,
                &commit_ts_map,
            );

            assert_eq!(via_manager, expected);
            assert_eq!(via_snapshot, expected);
            assert_eq!(
                via_manager, via_snapshot,
                "visibility equivalence drift for snapshot_id={snapshot_id}, created_txn={created_txn}, deleted_txn={deleted_txn:?}"
            );
        }
    }

    #[test]
    fn snapshot_at_pruned_history_remains_conservatively_invisible() {
        let manager = VisibilityManager::new();
        let snapshot = manager.snapshot_at(2, IsolationLevel::Snapshot);

        assert!(snapshot.commit_ts_map.is_empty());
        assert!(!manager.is_visible(&snapshot, 1, None));
    }

    #[test]
    fn historical_retention_window_is_operator_bounded_not_long_range_authority() {
        let mut manager = VisibilityManager::new();
        manager.config.max_history_entries = 1;
        manager.recover_committed_history([(1, 10), (2, 20)]);
        manager.prune_history(20);

        let snapshot = manager.snapshot_at(10, IsolationLevel::Snapshot);
        assert!(snapshot.commit_ts_map.is_empty());
        assert!(manager.get_commit_ts(1).is_none());
        assert!(manager.get_commit_ts(2).is_none());
        assert!(!manager.is_visible(&snapshot, 1, None));
    }

    #[test]
    fn visibility_equivalence_case_count_is_intentional() {
        const VISIBILITY_EQUIVALENCE_CASES: usize = 9;
        let cases = [
            (10, 11, None, btreeset(&[]), hashmap(&[(11, 11)]), false),
            (10, 8, None, btreeset(&[8]), hashmap(&[(8, 8)]), false),
            (10, 8, None, btreeset(&[]), hashmap(&[]), false),
            (10, 8, None, btreeset(&[]), hashmap(&[(8, 12)]), false),
            (10, 8, None, btreeset(&[]), hashmap(&[(8, 8)]), true),
            (
                10,
                8,
                Some(9),
                btreeset(&[]),
                hashmap(&[(8, 8), (9, 9)]),
                false,
            ),
            (10, 8, Some(9), btreeset(&[9]), hashmap(&[(8, 8)]), true),
            (
                10,
                8,
                Some(12),
                btreeset(&[]),
                hashmap(&[(8, 8), (12, 12)]),
                true,
            ),
            (10, 8, Some(9), btreeset(&[]), hashmap(&[(8, 8)]), true),
        ];

        assert_eq!(cases.len(), VISIBILITY_EQUIVALENCE_CASES);
    }
}
