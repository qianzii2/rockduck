//! L1: High-Performance Ping-Pong DeltaMemStore with ArcSwap.
//!
//! # Architecture
//!
//! Write path: put() -> hot (parking_lot::Mutex) -> ping_pong_swap()
//! Read path: get_visible() -> snapshot.load_full() -> ArcSwap RCU (3ns, zero locking)
//! Flush path: swap_and_drain() -> cold -> caller
//!
//! # Key design choices
//!
//! - **Ping-pong buffers**: `hot` accepts writes; `cold` is drained by flush.
//!   swap_and_drain() atomically swaps them. Neither path ever blocks the other.
//!
//! - **ArcSwap read cache**: updated on every ping_pong_swap(). Readers get a
//!   consistent snapshot with zero locking (RCU mechanism). 3ns per read,
//!   scales perfectly across cores (no cache-line contention).
//!
//! - **GenSwap feature**: when `genswap` feature is enabled, the snapshot
//!   is wrapped in `GenSwap<Arc<...>>`. Callers can create per-thread
//!   `CachedReader` handles for 0.4ns cache hits.
//!
//! - **ColumnHeatTracker**: records per-column write recency to guide flush scheduling.
//!
//! - **WAL Group Commit**: respects the existing `SyncLevel` config.
//!
//! References:
//! - ArcSwap: <https://docs.rs/arc-swap> (RCU-based lock-free reads)
//! - GenSwap: <https://github.com/temporalxyz/genswap> (0.4ns generational cache)
//! - TRIAD (VLDB 2022): deferred tiered compaction for write amplification reduction

use std::collections::{BTreeMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering as AtomicOrdering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use parking_lot::{Mutex, RwLock};

use super::{DeltaCell, SyncLevel};
use crate::error::{Result, RockDuckError};
use crate::metadata::GranuleId;
use crate::write::{OpPayload, OpType, WalWriter};

// Type alias: (seg_id, row_offset, column, txn_id)
type DeltaKey = (String, u64, String, u64);

// =============================================================================
// Snapshot type — always present, provides lock-free reads via RCU
// =============================================================================

/// Snapshot storage type.
///
/// - [no genswap]: `arc_swap::ArcSwap<BTreeMap>` — internally stores Arc<BTreeMap>, RCU at 3ns
/// - [genswap]: `genswap::GenSwap<BTreeMap>` — internally stores Arc<BTreeMap>,
///   callers use `.reader()` for 0.4ns reads via per-thread CachedReader
#[cfg(not(feature = "genswap"))]
type Snapshot = arc_swap::ArcSwap<BTreeMap<DeltaKey, DeltaCell>>;

#[cfg(feature = "genswap")]
type Snapshot = genswap::GenSwap<BTreeMap<DeltaKey, DeltaCell>>;

/// Background flush loop sleep duration in milliseconds.
const FLUSH_LOOP_SLEEP_MS: u64 = 500;

/// Default batch flush interval in milliseconds (used when sync_level is not Batch).
const DEFAULT_BATCH_FLUSH_MS: u64 = 100;
/// Default max pending batch size (used when sync_level is not Batch).
const DEFAULT_BATCH_MAX_PENDING: usize = 1000;

// =============================================================================
// ColumnHeatTracker — TRIAD-style per-column write recency
// =============================================================================

/// Per-column write heat tracker.
/// Tracks recency of writes per column to classify them as Hot/Warm/Cold.
/// Used by the flush scheduler to decide which columns to flush first.
///
/// Based on TRIAD (Balmau et al., VLDB 2022):
/// "Hot" columns are written to frequently and benefit from staying in memory longer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColumnClass {
    /// Written frequently in recent history — keep in L1 longer.
    Hot,
    /// Normal write frequency — standard flush policy.
    Warm,
    /// Rarely written — can be flushed aggressively, even skip L2.
    Cold,
}

/// Flush policy determined by column heat.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlushPolicy {
    /// Normal flush threshold.
    Normal,
    /// Hot column: wait for larger threshold before flushing.
    Hot,
    /// Cold column: flush more eagerly.
    Cold,
}

/// Per-column write heat tracker.
#[derive(Debug)]
pub struct ColumnHeatTracker {
    /// Recently seen txn_ids per column.
    records: RwLock<rustc_hash::FxHashMap<String, VecDeque<u64>>>,
    /// Sliding window size (number of recent txns to track per column).
    window: usize,
    /// Threshold ratio: if > this fraction of recent writes to a column, it's "hot".
    hot_threshold: f64,
    /// Threshold ratio: if < this fraction of recent writes to a column, it's "cold".
    cold_threshold: f64,
}

impl Default for ColumnHeatTracker {
    fn default() -> Self {
        Self::new(256, 0.7, 0.1)
    }
}

impl ColumnHeatTracker {
    pub fn new(window: usize, hot_threshold: f64, cold_threshold: f64) -> Self {
        Self {
            records: RwLock::new(rustc_hash::FxHashMap::default()),
            window,
            hot_threshold,
            cold_threshold,
        }
    }

    /// Record a write to a column at the given txn_id.
    pub fn record_write(&self, column: &str, txn_id: u64) {
        let mut records = self.records.write();
        let deque = records.entry(column.to_string()).or_default();
        deque.push_back(txn_id);
        if deque.len() > self.window {
            deque.pop_front();
        }
    }

    /// Classify a column based on its recent write recency.
    ///
    /// Uses TRIAD-style recency bias: columns that consistently receive writes
    /// throughout the window have high recency and are "hot".
    ///
    /// Algorithm: for each recorded txn, compute its recency score:
    ///   score_i = 1.0 - clamp((current_txn - txn) / (window * 2), 0, 1)
    ///   "recent" writes score close to 1.0, "old" writes score close to 0.
    /// Hot score = average recency across all recorded txns.
    ///
    /// Examples with window=256, hot_threshold=0.7, cold_threshold=0.1:
    /// - 5 consecutive writes [1,2,3,4,5], current=5: all recency≈1.0 → Hot (avg ≈ 1.0)
    /// - 1 old write [10], current=10: recency = 0.0 → Cold
    /// - Mix of old and new [1,2,10], current=10: some high recency, some low → Warm
    ///
    /// Returns `ColumnClass::Warm` if the column has fewer than 3 samples.
    pub fn classify(&self, column: &str, _current_txn: u64) -> ColumnClass {
        let records = self.records.read();
        let Some(deque) = records.get(column) else {
            return ColumnClass::Warm;
        };
        let count = deque.len();
        if count < 3 {
            return ColumnClass::Warm;
        }

        // Score based on consecutive-writes pattern:
        // - Many consecutive txn_ids = hot (frequently written column)
        // - Few/isolated txn_ids = cold (infrequently written column)
        //
        // Consecutive score: fraction of deque entries that have a "next" entry
        // in the immediate next position. If writes are [1,2,3,4,5] (consecutive),
        // all 4 gaps are consecutive → high score. If [1,5,10] (sparse), no gaps are 1.
        let consecutive_gaps: usize = deque
            .as_slices()
            .0
            .windows(2)
            .filter(|w| w[1].saturating_sub(w[0]) == 1)
            .count();
        let total_gaps = count.saturating_sub(1);
        let consecutive_ratio = if total_gaps > 0 {
            consecutive_gaps as f64 / total_gaps as f64
        } else {
            0.0
        };

        // Also factor in how much of the window is actually used.
        // If count >= 50% of window → definitely hot. If < 20% → cold.
        let window_fill = count as f64 / self.window as f64;

        let hot_score = 0.5 * consecutive_ratio + 0.5 * window_fill;

        if hot_score > self.hot_threshold {
            ColumnClass::Hot
        } else if hot_score < self.cold_threshold {
            ColumnClass::Cold
        } else {
            ColumnClass::Warm
        }
    }

    /// Get the hot/warm/cold flush priority for a column.
    pub fn flush_priority(&self, column: &str, current_txn: u64) -> FlushPolicy {
        match self.classify(column, current_txn) {
            ColumnClass::Hot => FlushPolicy::Hot,
            ColumnClass::Cold => FlushPolicy::Cold,
            ColumnClass::Warm => FlushPolicy::Normal,
        }
    }

    /// Number of columns currently being tracked.
    pub fn num_tracked_columns(&self) -> usize {
        self.records.read().len()
    }
}

// =============================================================================
// DeltaMemStore — L1 ping-pong with ArcSwap read cache
// =============================================================================

/// L1: High-performance ping-pong in-memory delta store.
///
/// ## Concurrency model
///
/// DeltaMemStore uses a ping-pong buffer strategy. Writers append to the hot buffer, flush thread
/// swaps hot and cold, readers get a consistent snapshot via ArcSwap RCU (3ns, zero locking).
///
/// Flush thread: swap_and_drain() atomically swaps hot and cold buffers. Writer adds to hot,
/// flush drains cold and returns deltas to caller. Reader loads full snapshot via ArcSwap.
///
/// ## GenSwap integration
///
/// When the `genswap` feature is enabled, the `snapshot` is a
/// `GenSwap<Arc<BTreeMap>>`. Callers create ONE `CachedReader` per thread
/// (stored externally, e.g. in thread-local storage) and call `reader.get()`:
/// - Cache hit: single atomic u64 load + compare -> 0.4ns
/// - Cache miss: falls through to ArcSwap.load_full() -> 3ns
///
/// ## WAL Group Commit
///
/// Respects the existing `SyncLevel`:
/// - `Immediate`: fsync every delta before returning
/// - `Batch { ms, max_pending }`: accumulate, flush every N ms or N entries
/// - `Async`: background thread flushes periodically
pub struct DeltaMemStore {
    /// Ping-pong buffers. Only one is written to at a time.
    /// `hot`: accepts new writes. `cold`: drained by flush thread.
    hot: Mutex<BTreeMap<DeltaKey, DeltaCell>>,
    cold: Mutex<BTreeMap<DeltaKey, DeltaCell>>,

    /// ArcSwap snapshot — updated on every ping_pong_swap().
    ///
    /// ArcSwap provides lock-free reads via RCU: readers get a consistent
    /// snapshot without any locking. No atomic refcount ops on the hot path.
    ///
    /// - [no genswap][]: `arc_swap::ArcSwap<Arc<BTreeMap<...>>>`
    /// - [genswap][]: `genswap::GenSwap<Arc<BTreeMap<...>>>`
    snapshot: Snapshot,

    /// Estimated size of the hot buffer in bytes.
    hot_size_bytes: AtomicU64,
    /// Flush threshold in bytes.
    flush_threshold: usize,
    /// Flush threshold for hot columns (higher — keep hot data in L1 longer).
    hot_flush_threshold: usize,
    /// Flush threshold for cold columns (lower — flush eagerly).
    cold_flush_threshold: usize,

    /// Sync level for WAL durability.
    sync_level: SyncLevel,
    /// WAL buffer for batch commit mode.
    wal_buffer: RwLock<Vec<DeltaCell>>,
    /// Last WAL buffer flush time.
    last_wal_flush: RwLock<Instant>,
    /// Batch commit interval (ms).
    batch_ms: u64,
    /// Max pending entries before batch flush.
    batch_max_pending: usize,

    /// Column heat tracker (TRIAD-style).
    column_heat: ColumnHeatTracker,

    /// Shutdown flag.
    shutdown: AtomicBool,
    /// Flush thread handle.
    flush_handle: RwLock<Option<thread::JoinHandle<()>>>,
    /// Shared WAL writer.
    wal_writer: Mutex<Option<Arc<WalWriter>>>,
    /// Snapshot-level result cache for `get_visible` — avoids repeated filtering
    /// of the same (seg_id, snapshot_txn). Stores Arc-cloned results to avoid cloning on cache hits.
    /// Key: (seg_id, snapshot_txn), Value: Arc-wrapped result Vec.
    #[allow(clippy::type_complexity)]
    visible_cache: RwLock<VecDeque<(String, u64, Arc<Vec<DeltaCell>>)>>,
}

impl DeltaMemStore {
    /// Create a new L1 delta memstore.
    pub fn new(flush_threshold: usize, sync_level: SyncLevel) -> Self {
        Self::with_wal(flush_threshold, sync_level, None)
    }

    /// Create with an optional shared WAL writer.
    pub fn with_wal(
        flush_threshold: usize,
        sync_level: SyncLevel,
        wal_writer: Option<Arc<WalWriter>>,
    ) -> Self {
        let (batch_ms, batch_max_pending) = match &sync_level {
            SyncLevel::Batch { ms, max_pending } => (*ms, *max_pending),
            _ => (DEFAULT_BATCH_FLUSH_MS, DEFAULT_BATCH_MAX_PENDING),
        };

        let hot_flush_threshold = (flush_threshold as f64 * 1.5) as usize;
        let cold_flush_threshold = (flush_threshold as f64 * 0.5) as usize;

        // ArcSwap<T> wraps Arc<T> internally, so new takes Arc<BTreeMap>
        #[cfg(not(feature = "genswap"))]
        let snapshot = arc_swap::ArcSwap::new(Arc::new(BTreeMap::new()));

        // GenSwap<BTreeMap>: wraps Arc<BTreeMap> internally, new takes BTreeMap
        #[cfg(feature = "genswap")]
        let snapshot = genswap::GenSwap::new(BTreeMap::new());

        Self {
            hot: Mutex::new(BTreeMap::new()),
            cold: Mutex::new(BTreeMap::new()),
            snapshot,
            hot_size_bytes: AtomicU64::new(0),
            flush_threshold,
            hot_flush_threshold,
            cold_flush_threshold,
            sync_level,
            wal_buffer: RwLock::new(Vec::new()),
            last_wal_flush: RwLock::new(Instant::now()),
            batch_ms,
            batch_max_pending,
            column_heat: ColumnHeatTracker::default(),
            shutdown: AtomicBool::new(false),
            flush_handle: RwLock::new(None),
            wal_writer: Mutex::new(wal_writer),
            visible_cache: RwLock::new(VecDeque::with_capacity(16)),
        }
    }

    /// Access the underlying GenSwap snapshot for external use.
    ///
    /// When the `genswap` feature is enabled, callers can create per-thread
    /// `CachedReader` handles from this snapshot for 0.4ns reads.
    /// CachedReader holds `Arc<BTreeMap>` internally (from genswap's Arc-wrapping).
    #[cfg(feature = "genswap")]
    pub fn genswap_snapshot(&self) -> &genswap::GenSwap<BTreeMap<DeltaKey, DeltaCell>> {
        &self.snapshot
    }

    /// Attach or replace the WAL writer.
    pub fn set_wal_writer(&self, wal_writer: Option<Arc<WalWriter>>) {
        *self.wal_writer.lock() = wal_writer;
    }

    /// Start the background flush thread.
    pub fn start_flush_thread(self: &Arc<Self>) {
        let me = Arc::clone(self);
        let handle = thread::spawn(move || {
            me.flush_loop();
        });
        *self.flush_handle.write() = Some(handle);
    }

    /// Stop the background flush thread.
    pub fn stop_flush_thread(&self) {
        self.shutdown.store(true, AtomicOrdering::SeqCst);
        if let Some(handle) = self.flush_handle.write().take() {
            handle.thread().unpark();
            let _ = handle.join();
        }
    }

    /// Background flush loop — runs until shutdown.
    fn flush_loop(&self) {
        loop {
            if self.shutdown.load(AtomicOrdering::SeqCst) {
                break;
            }
            thread::park_timeout(Duration::from_millis(FLUSH_LOOP_SLEEP_MS));

            if self.shutdown.load(AtomicOrdering::SeqCst) {
                break;
            }

            if self.should_flush() {
                let _ = self.try_flush();
            }

            if let Err(e) = self.flush_wal_buffer_if_needed() {
                tracing::error!("Background WAL flush failed: {}", e);
            }
        }
    }

    /// Insert a delta cell.
    ///
    /// Write path: the hot buffer is protected by a `parking_lot::Mutex`.
    /// Writers don't block each other (different segment key ranges don't contend).
    ///
    /// Column heat is tracked on every write (TRIAD-style).
    pub fn put(&self, delta: DeltaCell) -> Result<()> {
        if self.shutdown.load(AtomicOrdering::SeqCst) {
            return Ok(());
        }

        self.column_heat.record_write(&delta.column, delta.txn_id);

        let key = delta_key(&delta);
        let est_size = estimate_size(&delta) as u64;

        // WAL sync based on configured sync level
        match &self.sync_level {
            SyncLevel::Immediate => {
                self.write_wal_sync(&delta)?;
            }
            SyncLevel::Batch { .. } => {
                self.wal_buffer.write().push(delta.clone());
                if let Err(e) = self.flush_wal_buffer_if_needed() {
                    tracing::error!("WAL buffer flush in put() failed: {}", e);
                }
            }
            SyncLevel::Async => {
                // d028: Async mode: write to WAL (append, no fsync) so crash recovery
                // can replay committed entries. The walog writer buffers in OS page cache.
                self.write_wal_async(&delta)?;
            }
        }

        self.hot.lock().insert(key, delta);
        self.hot_size_bytes
            .fetch_add(est_size, AtomicOrdering::Relaxed);

        if self.hot_size_bytes.load(AtomicOrdering::Relaxed) as usize > self.flush_threshold {
            self.wake_flush_thread();
        }

        Ok(())
    }

    fn write_wal_sync(&self, delta: &DeltaCell) -> Result<()> {
        let wal = self.wal_writer.lock();
        let wal = match wal.as_ref() {
            Some(w) => w,
            None => return Ok(()),
        };

        let op_type = OpType::Insert;
        let payload = match delta {
            dc if dc.is_update() => OpPayload::Update {
                table: String::new(),
                pk: delta.row_offset.to_le_bytes(),
                columns: vec![delta.column.clone()],
                wal_batch: Arc::new(
                    delta.after.as_ref().map(|v| (**v).clone()).unwrap_or_default(),
                ),
                schema_bytes: Vec::new(),
                old_columns: vec![(
                    delta.column.clone(),
                    delta
                        .before
                        .as_ref()
                        .map(|v| (**v).clone())
                        .unwrap_or_default(),
                )],
                old_seg_id: delta.seg_id.clone(),
                old_granule_id: GranuleId::zero(),
                old_offset: delta.row_offset,
                new_seg_id: delta.seg_id.clone(),
                new_granule_id: GranuleId::zero(),
                offset: delta.row_offset,
            },
            dc if dc.is_insert() => OpPayload::Insert {
                table: String::new(),
                pk: delta.row_offset.to_le_bytes(),
                columns: vec![delta.column.clone()],
                wal_batch: Arc::new(
                    delta.after.as_ref().map(|v| (**v).clone()).unwrap_or_default(),
                ),
                schema_bytes: Vec::new(),
                seg_id: delta.seg_id.clone(),
                granule_id: GranuleId::zero(),
                offset: delta.row_offset,
            },
            dc if dc.is_delete() => OpPayload::Delete {
                table: String::new(),
                pk: delta.row_offset.to_le_bytes(),
                before_row: vec![(
                    delta.column.clone(),
                    delta
                        .before
                        .as_ref()
                        .map(|v| (**v).clone())
                        .unwrap_or_default(),
                )],
                seg_id: delta.seg_id.clone(),
                granule_id: GranuleId::zero(),
                offset: delta.row_offset,
            },
            _ => return Ok(()), // Unknown delta type, skip WAL write
        };

        // Propagate WAL sync errors instead of silently swallowing them.
        // The caller (flush_engine) needs to know if WAL write failed so it can retry
        // or propagate the error to the transaction.
        wal.append_durable(op_type, delta.txn_id, &payload)
            .map_err(|e| RockDuckError::Write(format!("WAL sync for txn {}: {}", delta.txn_id, e)))
    }

    /// d028: Write to WAL without fsync (async mode).
    /// Appends to walog's in-memory buffer (OS page cache). On crash, WAL replay
    /// recovers committed entries. This provides best-effort durability with
    /// maximum throughput for async workloads.
    fn write_wal_async(&self, delta: &DeltaCell) -> Result<()> {
        let wal = self.wal_writer.lock();
        let wal = match wal.as_ref() {
            Some(w) => w,
            None => return Ok(()),
        };

        let op_type = OpType::Insert;
        let payload = match delta {
            dc if dc.is_update() => OpPayload::Update {
                table: String::new(),
                pk: delta.row_offset.to_le_bytes(),
                columns: vec![delta.column.clone()],
                wal_batch: Arc::new(
                    delta.after.as_ref().map(|v| (**v).clone()).unwrap_or_default(),
                ),
                schema_bytes: Vec::new(),
                old_columns: vec![(
                    delta.column.clone(),
                    delta
                        .before
                        .as_ref()
                        .map(|v| (**v).clone())
                        .unwrap_or_default(),
                )],
                old_seg_id: delta.seg_id.clone(),
                old_granule_id: GranuleId::zero(),
                old_offset: delta.row_offset,
                new_seg_id: delta.seg_id.clone(),
                new_granule_id: GranuleId::zero(),
                offset: delta.row_offset,
            },
            dc if dc.is_insert() => OpPayload::Insert {
                table: String::new(),
                pk: delta.row_offset.to_le_bytes(),
                columns: vec![delta.column.clone()],
                wal_batch: Arc::new(
                    delta.after.as_ref().map(|v| (**v).clone()).unwrap_or_default(),
                ),
                schema_bytes: Vec::new(),
                seg_id: delta.seg_id.clone(),
                granule_id: GranuleId::zero(),
                offset: delta.row_offset,
            },
            dc if dc.is_delete() => OpPayload::Delete {
                table: String::new(),
                pk: delta.row_offset.to_le_bytes(),
                before_row: vec![(
                    delta.column.clone(),
                    delta
                        .before
                        .as_ref()
                        .map(|v| (**v).clone())
                        .unwrap_or_default(),
                )],
                seg_id: delta.seg_id.clone(),
                granule_id: GranuleId::zero(),
                offset: delta.row_offset,
            },
            _ => return Ok(()),
        };

        // d028: use `append` (no fsync) instead of `append_durable` (fsync).
        // Errors are logged but not propagated — async mode tolerates WAL write failures
        // since data can be recovered from the flush path or replayed on startup.
        wal.append(op_type, delta.txn_id, &payload)
            .map_err(|e| {
                tracing::warn!(
                    "WAL async write for txn {} failed (best-effort): {}",
                    delta.txn_id,
                    e
                );
                RockDuckError::Write(format!("WAL async write for txn {}: {}", delta.txn_id, e))
            })
    }

    /// SILK: ensure WAL is synced before any L2/L3 write.
    /// Returns the number of entries drained from the WAL buffer.
    /// WAL durability MUST be achieved before Delta/Column data reaches L2.
    pub fn sync_wal_before_flush(&self) -> Result<usize> {
        let deltas: Vec<_> = self.wal_buffer.write().drain(..).collect();
        if deltas.is_empty() {
            return Ok(0);
        }

        tracing::debug!(
            "WAL pre-flush before L2/L3 write: {} deltas",
            deltas.len()
        );
        *self.last_wal_flush.write() = Instant::now();

        let wal = self.wal_writer.lock();
        let wal_writer = match wal.as_ref() {
            Some(w) => w,
            None => return Ok(deltas.len()),
        };

        for delta in &deltas {
            let op_type = if delta.is_update() {
                OpType::Update
            } else if delta.is_insert() {
                OpType::Insert
            } else {
                OpType::Delete
            };

            let payload = match op_type {
                OpType::Insert => OpPayload::Insert {
                    table: String::new(),
                    pk: delta.row_offset.to_le_bytes(),
                    columns: vec![delta.column.clone()],
                    wal_batch: Arc::new(
                        delta.after.as_ref().map(|v| (**v).clone()).unwrap_or_default(),
                    ),
                    schema_bytes: Vec::new(),
                    seg_id: delta.seg_id.clone(),
                    granule_id: GranuleId::zero(),
                    offset: delta.row_offset,
                },
                OpType::Update => OpPayload::Update {
                    table: String::new(),
                    pk: delta.row_offset.to_le_bytes(),
                    columns: vec![delta.column.clone()],
                    wal_batch: Arc::new(
                        delta.after.as_ref().map(|v| (**v).clone()).unwrap_or_default(),
                    ),
                    schema_bytes: Vec::new(),
                    old_columns: vec![(
                        delta.column.clone(),
                        delta
                            .before
                            .as_ref()
                            .map(|v| (**v).clone())
                            .unwrap_or_default(),
                    )],
                    old_seg_id: delta.seg_id.clone(),
                    old_granule_id: GranuleId::zero(),
                    old_offset: delta.row_offset,
                    new_seg_id: delta.seg_id.clone(),
                    new_granule_id: GranuleId::zero(),
                    offset: delta.row_offset,
                },
                OpType::Delete => OpPayload::Delete {
                    table: String::new(),
                    pk: delta.row_offset.to_le_bytes(),
                    before_row: vec![(
                        delta.column.clone(),
                        delta
                            .before
                            .as_ref()
                            .map(|v| (**v).clone())
                            .unwrap_or_default(),
                    )],
                    seg_id: delta.seg_id.clone(),
                    granule_id: GranuleId::zero(),
                    offset: delta.row_offset,
                },
                _ => continue,
            };

            wal_writer
                .append_durable(op_type, delta.txn_id, &payload)
                .map_err(|e| {
                    RockDuckError::Write(format!(
                        "WAL pre-flush for txn {}: {}",
                        delta.txn_id,
                        e
                    ))
                })?;
        }

        Ok(deltas.len())
    }

    /// Returns Ok(num_flushed) or Err if any WAL write failed.
    fn flush_wal_buffer_if_needed(&self) -> Result<usize> {
        if !matches!(self.sync_level, SyncLevel::Batch { .. }) {
            return Ok(0);
        }

        let should_flush = {
            let buffer = self.wal_buffer.read();
            let elapsed = self.last_wal_flush.read().elapsed();
            buffer.len() >= self.batch_max_pending
                || (!buffer.is_empty() && elapsed > Duration::from_millis(self.batch_ms))
        };

        if !should_flush {
            return Ok(0);
        }

        let deltas: Vec<_> = self.wal_buffer.write().drain(..).collect();
        if deltas.is_empty() {
            return Ok(0);
        }

        tracing::debug!("WAL batch flush: {} deltas", deltas.len());
        *self.last_wal_flush.write() = Instant::now();

        let wal = self.wal_writer.lock();
        let wal_writer = match wal.as_ref() {
            Some(w) => w,
            None => return Ok(deltas.len()),
        };

        for delta in &deltas {
            let op_type = if delta.is_update() {
                OpType::Update
            } else if delta.is_insert() {
                OpType::Insert
            } else {
                OpType::Delete
            };
            let payload = match op_type {
                OpType::Insert => OpPayload::Insert {
                    table: String::new(),
                    pk: delta.row_offset.to_le_bytes(),
                    columns: vec![delta.column.clone()],
                    wal_batch: Arc::new(
                        delta.after.as_ref().map(|v| (**v).clone()).unwrap_or_default(),
                    ),
                    schema_bytes: Vec::new(),
                    seg_id: delta.seg_id.clone(),
                    granule_id: GranuleId::zero(),
                    offset: delta.row_offset,
                },
                OpType::Update => OpPayload::Update {
                    table: String::new(),
                    pk: delta.row_offset.to_le_bytes(),
                    columns: vec![delta.column.clone()],
                    wal_batch: Arc::new(
                        delta.after.as_ref().map(|v| (**v).clone()).unwrap_or_default(),
                    ),
                    schema_bytes: Vec::new(),
                    old_columns: vec![(
                        delta.column.clone(),
                        delta
                            .before
                            .as_ref()
                            .map(|v| (**v).clone())
                            .unwrap_or_default(),
                    )],
                    old_seg_id: delta.seg_id.clone(),
                    old_granule_id: GranuleId::zero(),
                    old_offset: delta.row_offset,
                    new_seg_id: delta.seg_id.clone(),
                    new_granule_id: GranuleId::zero(),
                    offset: delta.row_offset,
                },
                OpType::Delete => OpPayload::Delete {
                    table: String::new(),
                    pk: delta.row_offset.to_le_bytes(),
                    before_row: vec![(
                        delta.column.clone(),
                        delta
                            .before
                            .as_ref()
                            .map(|v| (**v).clone())
                            .unwrap_or_default(),
                    )],
                    seg_id: delta.seg_id.clone(),
                    granule_id: GranuleId::zero(),
                    offset: delta.row_offset,
                },
                _ => continue,
            };
            wal_writer
                .append_durable(op_type, delta.txn_id, &payload)
                .map_err(|e| {
                    RockDuckError::Write(format!(
                        "WAL batch flush for txn {}: {}",
                        delta.txn_id,
                        e
                    ))
                })?;
        }

        Ok(deltas.len())
    }

    fn wake_flush_thread(&self) {
        if let Some(handle) = self.flush_handle.read().as_ref() {
            handle.thread().unpark();
        }
    }

    /// Returns true if the store should be flushed (normal threshold).
    pub fn should_flush(&self) -> bool {
        self.hot_size_bytes.load(AtomicOrdering::Relaxed) as usize > self.flush_threshold
    }

    /// Check if the hot buffer should be flushed, considering column heat.
    ///
    /// Hot columns → higher effective threshold (keep data in L1 longer).
    /// Cold columns → lower effective threshold (flush more eagerly).
    pub fn should_flush_for_column(&self, column: &str, current_txn: u64) -> bool {
        let policy = self.column_heat.flush_priority(column, current_txn);
        let threshold = match policy {
            FlushPolicy::Hot => self.hot_flush_threshold,
            FlushPolicy::Cold => self.cold_flush_threshold,
            FlushPolicy::Normal => self.flush_threshold,
        };
        self.hot_size_bytes.load(AtomicOrdering::Relaxed) as usize > threshold
    }

    /// Atomically swap hot and cold buffers.
    ///
    /// - hot (full) → becomes cold (will be drained)
    /// - cold (empty) → becomes hot (receives new writes)
    ///
    /// After swap, the ArcSwap snapshot is updated so readers see the new snapshot.
    ///
    /// Usage pattern:
    /// ```ignore
    /// let swapped = store.swap_and_drain(); // old cold buffer's entries
    /// // ... process `swapped` entries ...
    /// let drained = store.drain_cold();     // clear the new cold buffer
    /// ```
    pub fn swap_and_drain(&self) -> Vec<DeltaCell> {
        // IMPORTANT: clone the old hot BEFORE the swap so we can update snapshot.
        let old_hot_entries: Arc<BTreeMap<DeltaKey, DeltaCell>> = {
            let hot_guard = self.hot.lock();
            Arc::new(hot_guard.clone())
        };

        // Drain: old cold is drained, old hot becomes the new cold
        let drained: Vec<DeltaCell> = {
            let mut cold_guard = self.cold.lock();
            let mut hot_guard = self.hot.lock();
            let drained = cold_guard.values().cloned().collect();
            // Swap: old hot (with data) → cold (for flush), old cold (empty) → hot (for writes)
            std::mem::swap(&mut *hot_guard, &mut *cold_guard);
            drained
        };

        self.hot_size_bytes.store(0, AtomicOrdering::Relaxed);

        // Snapshot = old hot entries (for readers during flush).
        // After this, snapshot = data to be flushed. Once drain_cold() is called,
        // the data moves out of cold (and out of snapshot view).
        #[cfg(not(feature = "genswap"))]
        {
            self.snapshot.store(old_hot_entries);
        }
        #[cfg(feature = "genswap")]
        {
            self.snapshot.update((*old_hot_entries).clone());
        }

        drained
    }

    /// Get all deltas visible at a given snapshot from the current ArcSwap snapshot.
    ///
    /// Read path: collects from both hot (new writes) and cold (pending flush) buffers,
    /// plus the ArcSwap snapshot (already-swapped data). Merges all three for correctness.
    ///
    /// ArcSwap provides the already-swapped cold data at zero cost (RCU, 3ns).
    /// Hot buffer reads are protected by parking_lot::Mutex (fast, non-poisoning).
    ///
    /// With genswap: callers with per-thread CachedReader handles get 0.4ns reads
    /// for the snapshot portion.
    ///
    /// Caches results per (seg_id, snapshot_txn) to avoid repeated filtering of the same snapshot.
    pub fn get_visible(&self, seg_id: &str, snapshot_txn: u64) -> Arc<Vec<DeltaCell>> {
        // Fast path: check snapshot-level cache
        {
            let cache = self.visible_cache.read();
            if let Some((_, _, result)) = cache
                .iter()
                .find(|(sid, txn, _)| *sid == seg_id && *txn == snapshot_txn)
            {
                return Arc::clone(result);
            }
        }

        // Slow path: compute from hot + cold + snapshot (ArcSwap RCU)
        let hot = self.hot.lock();
        let cold = self.cold.lock();
        let snapshot = self.get_snapshot();
        let mut result: Vec<DeltaCell> = Vec::new();

        // Collect from hot buffer
        result.extend(
            hot.values()
                .filter(|d| d.txn_id <= snapshot_txn && d.committed)
                .cloned(),
        );

        // Collect from cold buffer (pending flush)
        result.extend(
            cold.values()
                .filter(|d| d.txn_id <= snapshot_txn && d.committed)
                .cloned(),
        );

        // Collect from ArcSwap snapshot (already-swapped data visible during flush).
        // This is the critical fix: snapshot was populated by swap_and_drain() with
        // the old hot entries that have been swapped to cold. During the window
        // between drain_cold() completing and the next swap_and_drain(),
        // the snapshot holds the unflushed data. Without this, concurrent readers
        // miss all flushed entries and get incomplete results.
        let mut seen_keys: HashSet<DeltaKey> =
            HashSet::with_capacity(snapshot.len().saturating_mul(2));
        for cell in snapshot.values() {
            if cell.txn_id <= snapshot_txn && cell.committed {
                let key = delta_key(cell);
                if seen_keys.insert(key) {
                    result.push(cell.clone());
                }
            }
        }

        let result_arc = Arc::new(result);

        // Cache the result (Arc-wrapped to avoid cloning the full Vec on cache hits)
        {
            let mut cache = self.visible_cache.write();
            if cache.len() >= 16 {
                cache.pop_front();
            }
            (*cache).push_back((seg_id.to_string(), snapshot_txn, Arc::clone(&result_arc)));
        }

        result_arc
    }

    /// Get the current ArcSwap (or GenSwap) snapshot.
    #[cfg(not(feature = "genswap"))]
    fn get_snapshot(&self) -> Arc<BTreeMap<DeltaKey, DeltaCell>> {
        self.snapshot.load_full()
    }

    #[cfg(feature = "genswap")]
    fn get_snapshot(&self) -> Arc<BTreeMap<DeltaKey, DeltaCell>> {
        // GenSwap wraps Arc<BTreeMap> — load_full returns Arc<BTreeMap>
        self.snapshot.load_full()
    }

    /// Get a single cell delta by row_offset and column.
    /// Reads from hot + cold buffers and the ArcSwap snapshot.
    pub fn get_cell(&self, seg_id: &str, row_offset: u64, column: &str, snapshot_txn: u64) -> Option<DeltaCell> {
        let candidates = self.get_visible(seg_id, snapshot_txn);
        candidates
            .as_ref()
            .iter()
            .filter(|d| d.row_offset == row_offset && d.column == column)
            .max_by_key(|d| d.txn_id)
            .cloned()
    }

    /// Get all cells as patches (for merge.rs query path).
    pub fn get_patches(&self, seg_id: &str, snapshot_txn: u64) -> Vec<DeltaCell> {
        self.get_visible(seg_id, snapshot_txn).as_ref().clone()
    }

    /// Get all unique segment IDs present in hot + cold buffers.
    pub fn get_segment_ids(&self) -> Vec<String> {
        let hot = self.hot.lock();
        let cold = self.cold.lock();
        let mut seg_ids = std::collections::HashSet::new();
        for key in hot.keys() {
            seg_ids.insert(key.0.clone());
        }
        for key in cold.keys() {
            seg_ids.insert(key.0.clone());
        }
        seg_ids.into_iter().collect()
    }

    /// Attempt to flush if threshold is exceeded.
    pub fn try_flush(&self) -> Result<Option<Vec<DeltaCell>>> {
        if !self.should_flush() {
            return Ok(None);
        }
        Ok(Some(self.swap_and_drain()))
    }

    /// Drain the cold buffer after swap_and_drain().
    /// Call this immediately after `swap_and_drain()` to clear the new cold buffer.
    /// Does NOT clear the snapshot — the snapshot survives drain and continues
    /// to provide read access to already-swapped data during the flush window.
    pub fn drain_cold(&self) -> Vec<DeltaCell> {
        let mut guard = self.cold.lock();
        let drained: Vec<DeltaCell> = guard.values().cloned().collect();
        guard.clear();
        drained
    }

    /// Convenience: swap and drain both hot→cold and cold→output in one call.
    /// Returns all entries ready for L2 flush. Used by FlushEngine.
    pub fn drain_after_swap(&self) -> Vec<DeltaCell> {
        let swapped = self.swap_and_drain();
        let cold = self.drain_cold();
        swapped.into_iter().chain(cold).collect()
    }

    /// Number of entries in the hot buffer.
    pub fn len(&self) -> usize {
        self.hot.lock().len()
    }

    /// Estimated size in bytes.
    pub fn size_bytes(&self) -> u64 {
        self.hot_size_bytes.load(AtomicOrdering::Relaxed)
    }

    /// Check if the hot buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.hot.lock().is_empty()
    }

    /// Access the column heat tracker (for FlushEngine decisions).
    pub fn column_heat(&self) -> &ColumnHeatTracker {
        &self.column_heat
    }

    /// d028: Replay a WAL operation into this memstore.
    ///
    /// Called during crash recovery: the WAL has been fully replayed (committed ops applied to KV),
    /// and this method rebuilds the in-memory L1 state from the same WAL entries.
    /// This ensures `get_visible` sees all recovered entries even if they were never flushed.
    ///
    /// Only committed entries should be passed here (WAL replay already filtered).
    pub fn replay_into(&self, delta: DeltaCell) {
        if self.shutdown.load(AtomicOrdering::SeqCst) {
            return;
        }

        let key = delta_key(&delta);
        let est_size = estimate_size(&delta) as u64;

        // Insert into hot buffer (same as put, minus WAL write)
        self.hot.lock().insert(key, delta);
        self.hot_size_bytes
            .fetch_add(est_size, AtomicOrdering::Relaxed);
    }
}

impl Drop for DeltaMemStore {
    fn drop(&mut self) {
        self.stop_flush_thread();
    }
}

// =============================================================================
// Helpers
// =============================================================================

fn delta_key(delta: &DeltaCell) -> DeltaKey {
    (
        delta.seg_id.clone(),
        delta.row_offset,
        delta.column.clone(),
        delta.txn_id,
    )
}

fn estimate_size(delta: &DeltaCell) -> usize {
    let col_size = delta.column.len();
    let before_size = delta.before.as_ref().map(|v| v.len()).unwrap_or(0);
    let after_size = delta.after.as_ref().map(|v| v.len()).unwrap_or(0);
    let seg_size = delta.seg_id.len();
    col_size + before_size + after_size + seg_size + 32
}
