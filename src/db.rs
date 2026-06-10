//! RockDuck main database struct and entry point

use fastbloom::BloomFilter;
use parking_lot::RwLock;
use rustc_hash::FxHashMap;
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

use crate::cdc::{CdcGranularity, CdcLogBuffer, CdcLogEntry, CdcOp, CdcWalWriter};
use crate::compaction::access_tracker::AccessTrackerHandle;
use crate::compaction::adaptive::AdaptiveCompactionScheduler;
use crate::compaction::io_scheduler::{IOSchedulerConfig, IOSchedulerHandle};
use crate::compaction::nonblocking::{NonBlockingCompactor, NonBlockingConfig};
use crate::config::RockDuckConfig;
use crate::error::{Result, RockDuckError};
use crate::metadata;
use crate::metadata::kv_engine::KVEngine;
use crate::metadata::seg_meta::SegmentMetaCache;
use crate::metadata::GranuleId;
use crate::mvcc::visibility::{TxnSnapshot, VisibilityManager};
use crate::read::scan::ScanIterator;
use crate::storage::delta::{
    types::DeltaCheckpointState, DeltaConfig, DeltaLayerStack, FlushEngine, SyncLevel,
};
use crate::write::checkpoint::{CheckpointManager, CheckpointMvccState, CheckpointState};
use crate::write::durability_wal::{GroupCommitConfig, OpPayload, OpType, WalConfig, WalWriter};
use crate::write::heat_tracker::WriteHeatTrackerHandle;
use crate::write::wal_recovery::{
    replay_committed_ops, RecoveryResult, ReplayError, ReplayErrorKind, WalOp,
};

/// Transaction ID type
pub type TxnId = u64;

/// LRU-bounded bloom filter cache with capacity limit.
/// Uses `RwLock<HashMap>` instead of `DashMap` to avoid concurrent eviction races.
/// LRU position tracked via HashMap<String, usize> for O(1) refresh (db007 fix).
pub struct BoundedBloomFilterCache {
    inner: RwLock<FxHashMap<String, Arc<parking_lot::Mutex<BloomFilter>>>>,
    /// LRU order: VecDeque of keys, oldest at front.
    insertion_order: parking_lot::Mutex<VecDeque<String>>,
    /// O(1) position lookup for LRU refresh: key -> index in VecDeque (db007 fix).
    position_index: RwLock<FxHashMap<String, usize>>,
    capacity: usize,
}

impl BoundedBloomFilterCache {
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: RwLock::new(FxHashMap::with_capacity_and_hasher(
                capacity,
                Default::default(),
            )),
            insertion_order: parking_lot::Mutex::new(VecDeque::with_capacity(capacity)),
            position_index: RwLock::new(FxHashMap::with_capacity_and_hasher(
                capacity,
                Default::default(),
            )),
            capacity,
        }
    }

    fn evict_one(&self) {
        let mut order = self.insertion_order.lock();
        if let Some(oldest) = order.pop_front() {
            self.inner.write().remove(&oldest);
            self.position_index.write().remove(&oldest);
            // Rebuild index with decremented positions
            let mut idx = self.position_index.write();
            for (i, k) in order.iter().enumerate() {
                idx.insert(k.clone(), i);
            }
        }
    }

    pub fn insert(&self, key: String, bf: BloomFilter) {
        self.evict_one();
        self.push_back_lru(&key);
        self.inner
            .write()
            .insert(key, Arc::new(parking_lot::Mutex::new(bf)));
    }

    /// Get an Arc<Mutex<BloomFilter>> for the given key, creating it with `f` if absent.
    ///
    /// ## Atomic Get-or-Insert (db011 fix)
    ///
    /// Holds the write lock through the entire lookup+insert+LRU-update sequence,
    /// eliminating the TOCTOU race between the old fast-path read-lock release
    /// and the subsequent insertion_order lock acquisition.
    pub fn get_or_insert_with<F: FnOnce() -> BloomFilter>(
        &self,
        key: &str,
        f: F,
    ) -> Arc<parking_lot::Mutex<BloomFilter>> {
        // db011 fix: Hold the write lock for the entire operation.
        // LRU updates are done atomically under the same lock.
        let mut map = self.inner.write();
        let key_owned = key.to_string();

        if let Some(entry) = map.get(&key_owned) {
            // Key found: refresh LRU position atomically, then return.
            let result = Arc::clone(entry);
            // db011 fix: O(1) LRU refresh under the same write lock - no TOCTOU window.
            self.refresh_lru_position_under_lock(&key_owned, &mut map);
            return result;
        }

        // Key absent: evict if at capacity, then insert atomically.
        if map.len() >= self.capacity {
            // db011 fix: evict_under_lock holds the write lock throughout.
            self.evict_one_under_lock(&mut map);
        }

        let entry = Arc::new(parking_lot::Mutex::new(f()));
        // db011 fix: O(1) LRU push under the same write lock - no TOCTOU window.
        self.push_back_lru_under_lock(&key_owned, &mut map);
        map.insert(key_owned, Arc::clone(&entry));
        entry
    }

    /// O(1) LRU refresh under write lock (db011 fix).
    /// Must be called while `map` write guard is held.
    fn refresh_lru_position_under_lock(
        &self,
        key: &str,
        _map: &mut FxHashMap<String, Arc<parking_lot::Mutex<BloomFilter>>>,
    ) {
        let pos = {
            let idx = self.position_index.read();
            *idx.get(key).unwrap_or(&usize::MAX)
        };
        if pos == usize::MAX {
            return;
        }
        let mut order = self.insertion_order.lock();
        let new_pos = order.len().saturating_sub(1);
        if pos < new_pos {
            order.remove(pos);
            order.push_back(key.to_string());
            // O(1) targeted update for the moved key and affected entries.
            let mut idx = self.position_index.write();
            idx.insert(key.to_string(), new_pos);
            for k in order.iter().take(pos) {
                if let Some(p) = idx.get_mut(k) {
                    *p -= 1;
                }
            }
        }
    }

    /// O(1) LRU push under write lock (db011 fix).
    /// Must be called while `map` write guard is held.
    fn push_back_lru_under_lock(
        &self,
        key: &str,
        _map: &mut FxHashMap<String, Arc<parking_lot::Mutex<BloomFilter>>>,
    ) {
        let new_pos = {
            let mut order = self.insertion_order.lock();
            order.push_back(key.to_string());
            order.len() - 1
        };
        self.position_index.write().insert(key.to_string(), new_pos);
    }

    /// Evict one entry under the write lock (db011 fix).
    fn evict_one_under_lock(
        &self,
        _map: &mut FxHashMap<String, Arc<parking_lot::Mutex<BloomFilter>>>,
    ) {
        let mut order = self.insertion_order.lock();
        if let Some(oldest) = order.pop_front() {
            drop(order);
            self.inner.write().remove(&oldest);
            let mut idx = self.position_index.write();
            idx.remove(&oldest);
            // O(1) targeted update: decrement positions for remaining entries.
            let order_guard = self.insertion_order.lock();
            let new_len = order_guard.len();
            for (i, k) in order_guard.iter().enumerate() {
                if i >= new_len {
                    break;
                }
                idx.insert(k.clone(), i);
            }
        }
    }

    /// Move an existing key to the back of the LRU queue (O(1) via position index).
    /// db007 fix: O(1) targeted update instead of rebuilding entire index.
    /// Also used by db011 fix: called from get_or_insert_with under lock.
    #[allow(dead_code)]
    fn refresh_lru_position(&self, key: &str) {
        let pos = {
            let idx = self.position_index.read();
            *idx.get(key).unwrap_or(&usize::MAX)
        };
        if pos == usize::MAX {
            return;
        }
        let mut order = self.insertion_order.lock();
        let new_pos = order.len().saturating_sub(1);
        if pos < new_pos {
            // db007 fix: O(1) targeted update - only update the moved key and affected entries.
            order.remove(pos);
            order.push_back(key.to_string());
            let mut idx = self.position_index.write();
            idx.insert(key.to_string(), new_pos);
            for k in order.iter().take(pos) {
                if let Some(p) = idx.get_mut(k) {
                    *p -= 1;
                }
            }
        }
    }

    /// Append a new key to the back of the LRU queue.
    /// db007 fix: O(1) targeted update instead of rebuilding entire index.
    /// Also used by db011 fix: called from get_or_insert_with under lock.
    fn push_back_lru(&self, key: &str) {
        let new_pos = {
            let mut order = self.insertion_order.lock();
            order.push_back(key.to_string());
            order.len() - 1
        };
        // db007 fix: O(1) targeted update - just insert the new key's position.
        self.position_index.write().insert(key.to_string(), new_pos);
    }

    /// Rebuild the position index from the current VecDeque state.
    /// db007 fix: kept for remove() which needs full rebuild after VecDeque remove,
    /// but all other operations use O(1) targeted updates.
    #[allow(dead_code)]
    fn rebuild_position_index(&self) {
        let order = self.insertion_order.lock();
        let mut idx = self.position_index.write();
        idx.clear();
        for (i, k) in order.iter().enumerate() {
            idx.insert(k.clone(), i);
        }
    }

    pub fn remove(&self, key: &str) {
        // Acquire all locks in consistent order to prevent deadlock
        let mut inner = self.inner.write();
        let mut order = self.insertion_order.lock();
        let mut idx = self.position_index.write();

        inner.remove(key);
        if let Some(&pos) = idx.get(key) {
            order.remove(pos);
            idx.remove(key);
            // Rebuild index with decremented positions after the removed slot
            idx.clear();
            for (i, k) in order.iter().enumerate() {
                idx.insert(k.clone(), i);
            }
        }
    }

    pub fn len(&self) -> usize {
        self.inner.read().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Iterate over (segment_id, Arc<Mutex<BloomFilter>>) pairs.
    /// db008 fix: Returns a new Arc wrapping the HashMap for O(1) cloning.
    /// Callers get an Arc they can hold while iterating.
    pub fn iter(&self) -> Arc<FxHashMap<String, Arc<parking_lot::Mutex<BloomFilter>>>> {
        Arc::new((*self.inner.read()).clone())
    }
}

/// RockDuck — HTAP embedded database instance
pub struct RockDuck {
    pub kv: Arc<dyn KVEngine>,

    pub data_dir: PathBuf,

    pub config: RockDuckConfig,

    pub wal: Arc<WalWriter>,

    /// Primary MVCC manager for mutation (begin/commit/rollback).
    pub mvcc: RwLock<VisibilityManager>,

    pub txn_counter: AtomicU64,

    pub delta_layer: Arc<DeltaLayerStack>,

    pub segment_bloom_filters: BoundedBloomFilterCache,

    pub seg_meta_cache: RwLock<SegmentMetaCache>,

    pub compaction_scheduler: RwLock<AdaptiveCompactionScheduler>,

    pub io_scheduler: Option<IOSchedulerHandle>,

    pub access_tracker: AccessTrackerHandle,

    pub write_heat_tracker: WriteHeatTrackerHandle,

    pub router: Option<Arc<crate::query::routing::QueryRouter>>,

    pub bloom_filter_dir: PathBuf,

    pub cdc_log_buffer: RwLock<CdcLogBuffer>,

    /// CDC WAL writer — `None` when CDC is disabled.
    /// Separate WAL from the data WAL; provides durable CDC event persistence.
    pub cdc_wal: Option<Arc<CdcWalWriter>>,

    /// Per-transaction pending CDC entries accumulated during the txn.
    /// Flushed to `cdc_wal` and `cdc_log_buffer` at `commit_txn`.
    pending_cdc_entries: RwLock<std::collections::HashMap<TxnId, Vec<CdcLogEntry>>>,

    pub cdc_granularity: CdcGranularity,

    pub txn_delta_collectors: RwLock<std::collections::HashMap<TxnId, Arc<DeltaLayerStack>>>,

    pub checkpoint_manager: CheckpointManager,

    pub last_checkpoint_txn: AtomicU64,

    pub flush_engine: Option<Arc<FlushEngine>>,

    pub outstanding_scan_bytes: Arc<AtomicU64>,

    /// Cached MVCC snapshot for lazy-loading optimization (mv010 fix).
    ///
    /// ## Lazy-loading Strategy
    ///
    /// The `commit_ts_map` in a snapshot is only needed for CDC time-travel queries,
    /// which filter delta cells by commit_ts. For normal OLTP reads, the `active_txns`
    /// set is sufficient for visibility checks (O(log n) lookup).
    ///
    /// This cache stores a lazily-populated `Arc<TxnSnapshot>` that:
    ///   - Is invalidated on any MVCC mutation (begin_txn, commit_txn, rollback_txn)
    ///   - Is only fully populated with `commit_ts_map` when needed for time-travel
    ///   - For normal reads, uses `snapshot_with_active_only()` to avoid populating commit_ts_map
    ///
    /// ## Cache Invalidation
    ///
    /// The cache must be invalidated whenever `VisibilityManager` state changes.
    /// Since RockDuck holds both the cache and the mvcc RwLock, invalidation
    /// is done by resetting `cached_snapshot` to None before releasing the write lock.
    cached_snapshot: RwLock<Option<Arc<TxnSnapshot>>>,
}

#[derive(Debug, Clone)]
pub struct CurrentRealityBaseline {
    pub truth_chain: &'static str,
    pub visibility_surfaces: Vec<&'static str>,
    pub routing_contract: &'static str,
    pub maintenance_contract: &'static str,
    pub governance_posture: &'static str,
}

#[derive(Debug, Clone)]
pub struct AscensionTargetBaseline {
    pub truth_chain: &'static str,
    pub visibility_target: &'static str,
    pub routing_target: &'static str,
    pub maintenance_target: &'static str,
    pub governance_target: &'static str,
}

#[derive(Debug, Clone)]
pub struct BaselineResetPackage {
    pub current_reality: CurrentRealityBaseline,
    pub ascension_target: AscensionTargetBaseline,
}

impl RockDuck {
    pub fn baseline_reset_package() -> BaselineResetPackage {
        BaselineResetPackage {
            current_reality: CurrentRealityBaseline {
                truth_chain: "WAL -> checkpoint/KV baseline -> WAL replay overlay -> VisibilityManager",
                visibility_surfaces: vec![
                    "scan via TxnSnapshot/VisibilityManager",
                    "point_get current reads via VisibilityManager",
                    "time-travel via HistoricalVisibility transition exception",
                    "VTab via TxnSnapshot::is_row_visible",
                    "compaction via SegmentOverlay filter",
                ],
                routing_contract: "RouteDecision selects the main scan execution template; several sanctioned bypasses still skip that authority",
                maintenance_contract: "PDT merge is the only mature physical rewrite; flush/checkpoint are adjacent but not unified under one task model",
                governance_posture: "verification cards + tracing debug form a hint layer, not layered enforcement",
            },
            ascension_target: AscensionTargetBaseline {
                truth_chain: "truth authority remains singular while projections become explicit and auditable",
                visibility_target: "visibility differences are represented as controlled projection modes under one authority contract",
                routing_target: "RouteDecision chooses real execution templates and feeds back real execution outcomes",
                maintenance_target: "rewrite/flush/metadata share maintenance language and budget while checkpoint keeps truth-plane privilege",
                governance_target: "tests/CI/static gates enforce classified boundaries, with minimal runtime invariants for correctness-critical seams",
            },
        }
    }

    pub fn open(path: impl Into<PathBuf>) -> Result<Self> {
        Self::open_with_config(path, RockDuckConfig::default())
    }

    pub fn open_with_config(path: impl Into<PathBuf>, mut config: RockDuckConfig) -> Result<Self> {
        let data_dir = path.into();

        let kv: Arc<dyn KVEngine> = {
            let engine = metadata::mace_adapter::MaceKVEngine::open(&data_dir)?;
            Arc::new(engine) as Arc<dyn KVEngine>
        };

        let wal_config = config.wal_config.take().unwrap_or_else(|| WalConfig {
            wal_dir: data_dir.join("wal"),
            max_file_size: 128 * 1024 * 1024,
            enabled: true,
            group_commit: Some(GroupCommitConfig::default()),
        });
        let wal = Arc::new(WalWriter::open(&data_dir, wal_config.clone())?);
        let wal_checkpoint = Arc::clone(&wal);
        let checkpoint_manager = CheckpointManager::new(&data_dir, wal_checkpoint.clone())?;
        let latest_checkpoint = checkpoint_manager.load_latest()?;

        // E1: WAL recovery — replay all committed ops from the WAL into data files.
        // This ensures data files (Vortex columns, PK index, visibility) are fully
        // reconstructed after a crash. WAL replay uses covering writes for data columns
        // and append mode for vis columns, so replaying the same op twice is safe.
        // On a clean shutdown (empty WAL), this is a no-op.
        let wal_dir = wal_config.wal_dir.clone();
        let kv_for_recovery = Arc::clone(&kv);
        let data_dir_for_recovery = data_dir.clone();

        // Persist committed_txn from KV before recovery overwrites it.
        let committed_txn_kv = metadata::get_committed_txn(&kv).unwrap_or(0);
        let checkpoint_state = latest_checkpoint.as_ref().map(|(_, state)| state.clone());
        let truth_verification = CheckpointManager::truth_package_verification();
        tracing::debug!(
            truth_main = truth_verification.truth_boundary.main_path,
            truth_bypasses = ?truth_verification.truth_boundary.bypass_paths,
            truth_landing = ?truth_verification.truth_boundary.landing_files,
            recovery_main = truth_verification.recovery_boundary.main_path,
            recovery_bypasses = ?truth_verification.recovery_boundary.bypass_paths,
            recovery_landing = ?truth_verification.recovery_boundary.landing_files,
            "truth package verification prepared"
        );
        let recovery_verification = CheckpointManager::recovery_verification_card();
        tracing::debug!(
            target = recovery_verification.main_path,
            bypasses = ?recovery_verification.bypass_paths,
            landing = ?recovery_verification.landing_files,
            recovery_order = ?recovery_verification.recovery_order,
            "recovery triple-check verified"
        );

        let wal_recovery_result =
            Self::replay_wal_ops(&wal_dir, &data_dir_for_recovery, &kv_for_recovery).map_err(
                |e| {
                    match e.kind {
                        ReplayErrorKind::Corruption(msg) => {
                            // Classified as Evidence-stale degrade (not fail-stop).
                            // Single txn's committed writes are skipped; all other committed
                            // txns replay normally. DB continues to open with potentially stale
                            // evidence for the skipped txn only. replay_failure file records
                            // the skipped txn for manual reconciliation.
                            tracing::error!(
                                "WAL recovery: evidence-stale: data corruption detected ({}) \
                         — skipping affected txn, replay_failure recorded at {}. \
                         DB will open with degraded evidence for this txn.",
                                msg,
                                wal_dir.display()
                            );
                            RockDuckError::Internal(format!(
                                "WAL evidence-stale during recovery: {} (see wal/replay_failure)",
                                msg
                            ))
                        }
                        ReplayErrorKind::Fatal(msg) => {
                            tracing::error!("WAL recovery fatal error: {}. DB cannot open.", msg);
                            RockDuckError::Write(format!("WAL recovery fatal error: {}", msg))
                        }
                        ReplayErrorKind::Recoverable(msg) | ReplayErrorKind::Ambiguous(msg) => {
                            // These should not propagate from replay_committed_ops,
                            // but handle them defensively.
                            tracing::error!(
                                "WAL recovery unexpected error kind: {}. Treating as fatal.",
                                msg
                            );
                            RockDuckError::Write(format!("WAL recovery unexpected: {}", msg))
                        }
                    }
                },
            )?;

        let mut mvcc = VisibilityManager::new();

        // Use the max of the KV-stored committed_txn and the highest committed txn seen in WAL.
        // If replay degraded for a committed txn, we must still preserve the monotonic truth boundary
        // in KV/MVCC rather than silently rewinding committed authority to the last fully replayed txn.
        let committed_txn = committed_txn_kv.max(wal_recovery_result.max_seen_committed_txn);
        debug_assert!(
            committed_txn >= committed_txn_kv,
            "recovery authority regression: committed_txn must preserve KV baseline"
        );
        debug_assert!(
            committed_txn >= wal_recovery_result.max_seen_committed_txn,
            "recovery authority regression: committed_txn must preserve WAL authority"
        );
        mvcc.set_committed_txn(committed_txn);
        // D7 fix: set WAL replay watermark so TTL eviction has correct lower bound
        mvcc.set_replay_watermark(wal_recovery_result.replay_watermark);
        let next_txn = committed_txn.saturating_add(1);

        // D8extra fix: if WAL recovery found a higher committed_txn than stored in KV,
        // write it back so future recovery starts from the correct point.
        if wal_recovery_result.max_seen_committed_txn > committed_txn_kv {
            if let Err(e) =
                metadata::put_committed_txn(&kv, wal_recovery_result.max_seen_committed_txn)
            {
                tracing::warn!(
                    "Failed to persist committed_txn {} to KV: {}",
                    wal_recovery_result.max_seen_committed_txn,
                    e
                );
            } else {
                tracing::debug!(
                    "Persisted committed_txn {} to KV after WAL recovery",
                    wal_recovery_result.max_seen_committed_txn
                );
            }
        }

        // Strict MVCC fix: recover commit_ts_map from checkpoint/KV first, then WAL.
        // WAL remains the source of truth for overlapping txn_ids.
        let mut kv_commit_ts = checkpoint_state
            .as_ref()
            .and_then(|state| state.mvcc.as_ref())
            .map(|mvcc| mvcc.committed_history.clone().into_iter().collect())
            .unwrap_or_else(|| metadata::get_committed_txns(&kv).unwrap_or_default());
        for (txn_id, commit_ts) in &wal_recovery_result.commit_ts_map {
            kv_commit_ts.insert(*txn_id, *commit_ts);
        }
        debug_assert!(
            wal_recovery_result
                .commit_ts_map
                .iter()
                .all(|(txn_id, commit_ts)| kv_commit_ts.get(txn_id) == Some(commit_ts)),
            "recovery authority regression: WAL commit_ts overlay must win for overlapping txn_ids"
        );
        let recovered_config = checkpoint_state
            .as_ref()
            .and_then(|state| state.mvcc.as_ref())
            .map(|mvcc| mvcc.visibility_config.clone());
        mvcc.recover_committed_history_with_config(
            kv_commit_ts,
            recovered_config,
            &wal_recovery_result.inserted_at_map,
        );

        // P1-A fix: recover active transactions from checkpoint/KV plus WAL.
        // WAL-recovered txns take precedence (WAL is the source of truth for committed state).
        let checkpoint_active = checkpoint_state
            .as_ref()
            .and_then(|state| state.mvcc.as_ref())
            .map(|mvcc| mvcc.active_txns.clone())
            .unwrap_or_else(|| metadata::get_active_txns(&kv).unwrap_or_default());
        let wal_active = wal_recovery_result.active_txns;
        let wal_active_count = wal_active.len();
        let wal_active_overlay = wal_active.clone();
        // Deduplicate: for each txn_id in WAL, prefer the WAL version.
        let mut merged: Vec<(TxnId, u64)> = checkpoint_active
            .into_iter()
            .filter(|(id, _)| !wal_active.iter().any(|(wid, _)| wid == id))
            .collect();
        merged.extend(wal_active);
        debug_assert!(
            merged.iter().all(|(txn_id, begin_ts)| {
                let wal_match = wal_active_overlay
                    .iter()
                    .find(|(wal_txn_id, _)| wal_txn_id == txn_id);
                wal_match.map(|(_, wal_begin_ts)| wal_begin_ts == begin_ts).unwrap_or(true)
            }),
            "recovery authority regression: WAL active_txn overlay must win for overlapping txn_ids"
        );
        if !merged.is_empty() {
            tracing::debug!(
                "Recovering {} active transactions ({} from KV, {} from WAL)",
                merged.len(),
                merged.len() - wal_active_count,
                wal_active_count
            );
        }
        mvcc.recover_active_txns(merged);

        let mvcc = RwLock::new(mvcc);

        let bloom_filter_dir = data_dir.join("bloom_filters");
        let outstanding_scan_bytes = Arc::new(AtomicU64::new(0));
        let l2_patch_count = Arc::new(AtomicUsize::new(0));
        let l3_patch_count = Arc::new(AtomicUsize::new(0));

        let delta_layer = Arc::new(DeltaLayerStack::new(
            DeltaConfig {
                data_dir: data_dir.clone(),
                ..Default::default()
            },
            Arc::clone(&l2_patch_count),
            Arc::clone(&l3_patch_count),
        ));

        // D8 fix: Share the recent_flush cache between DeltaLayerStack and FlushEngine
        // to prevent race window loss during L1→L2 flush.
        let recent_flush_cache = delta_layer.recent_flush.clone();
        let _delta_layer_for_engine = delta_layer.clone();

        // D11 fix: Recover orphan segment files on startup.
        // Clean up .tmp files and orphaned segment files that may have been left
        // behind by interrupted compaction or flush operations.
        if let Err(e) = delta_layer.l2.recover() {
            tracing::warn!("Delta L2 recovery failed (non-fatal): {}", e);
        }

        // d028: Rebuild L1 in-memory state from WAL entries replayed during crash recovery.
        // Only needed for Async sync level where WAL entries are not synced during normal operation.
        if matches!(delta_layer.config.sync_level, SyncLevel::Async) {
            delta_layer.recover_from_wal(wal_recovery_result.committed_ops);
        }

        // D8 fix: Create FlushEngine with recent_flush_cache for race window prevention.
        // The cache is passed via new_with_cache() to enable query_all_layers to see
        // recently flushed deltas during the flush window.
        let flush_engine = if config.compaction.background_enabled {
            let fe = Arc::new(FlushEngine::new_with_cache(
                delta_layer.l1.clone(),
                delta_layer.l2.clone(),
                delta_layer.l3.clone(),
                Arc::clone(&l2_patch_count),
                Arc::clone(&l3_patch_count),
                Arc::clone(&outstanding_scan_bytes),
                config.compaction.memstore_threshold,
                recent_flush_cache.clone(),
                delta_layer.flush_epoch.clone(),
            ));
            fe.clone().start();
            Some(fe)
        } else {
            None
        };

        let adaptive_scheduler = AdaptiveCompactionScheduler::new();
        let access_tracker = AccessTrackerHandle::new();
        let write_heat_tracker = WriteHeatTrackerHandle::new();
        let compaction_scheduler = RwLock::new(adaptive_scheduler.clone());
        let io_scheduler = None;

        let baseline_reset = Self::baseline_reset_package();
        let verification_templates = crate::query::routing::QueryRouter::verification_templates();
        tracing::debug!(
            current_truth = baseline_reset.current_reality.truth_chain,
            current_visibility = ?baseline_reset.current_reality.visibility_surfaces,
            current_routing = baseline_reset.current_reality.routing_contract,
            current_maintenance = baseline_reset.current_reality.maintenance_contract,
            current_governance = baseline_reset.current_reality.governance_posture,
            target_truth = baseline_reset.ascension_target.truth_chain,
            target_visibility = baseline_reset.ascension_target.visibility_target,
            target_routing = baseline_reset.ascension_target.routing_target,
            target_maintenance = baseline_reset.ascension_target.maintenance_target,
            target_governance = baseline_reset.ascension_target.governance_target,
            "baseline reset package prepared"
        );
        let evidence_verification = crate::read::scan::evidence_package_verification();
        let maintenance_verification = crate::compaction::adaptive::AdaptiveCompactionScheduler::maintenance_package_verification();
        let governance_verification =
            crate::compaction::adaptive::AdaptiveCompactionScheduler::governance_verification_card(
            );
        crate::compaction::adaptive::AdaptiveCompactionScheduler::assert_layered_governance();
        tracing::debug!(
            call_boundary = verification_templates.call_boundary.goal,
            authority_state = verification_templates.authority_state.goal,
            exception_path = verification_templates.exception_path.goal,
            call_outputs = ?verification_templates.call_boundary.required_output,
            authority_outputs = ?verification_templates.authority_state.required_output,
            exception_outputs = ?verification_templates.exception_path.required_output,
            evidence_router_main = evidence_verification.router_boundary.main_path,
            evidence_metadata_main = evidence_verification.metadata_boundary.main_path,
            maintenance_main = maintenance_verification.debt_signal_boundary.main_path,
            governance_main = governance_verification.main_path,
            "subagent verification templates prepared"
        );
        tracing::debug!(
            evidence_router_bypasses = ?evidence_verification.router_boundary.bypass_paths,
            evidence_metadata_bypasses = ?evidence_verification.metadata_boundary.bypass_paths,
            maintenance_bypasses = ?maintenance_verification.debt_signal_boundary.bypass_paths,
            governance_bypasses = ?governance_verification.bypass_paths,
            evidence_router_landing = ?evidence_verification.router_boundary.landing_files,
            evidence_metadata_landing = ?evidence_verification.metadata_boundary.landing_files,
            maintenance_landing = ?maintenance_verification.debt_signal_boundary.landing_files,
            governance_landing = ?governance_verification.landing_files,
            "package verification cards prepared"
        );
        let router = match crate::query::routing::QueryRouter::open(&kv, config.router.clone()) {
            Ok(r) => Some(Arc::new(r)),
            Err(e) => {
                tracing::warn!("failed to open query router, disabling: {}", e);
                None
            }
        };

        let last_checkpoint_txn = AtomicU64::new(committed_txn);

        let cdc_granularity = config.cdc.granularity;
        let cdc_log_buffer = RwLock::new(CdcLogBuffer::new(config.cdc.log_buffer_size));
        let pending_cdc_entries = RwLock::new(std::collections::HashMap::new());

        // CDC WAL: open only when CDC is enabled.
        // Separate WAL directory from the data WAL, with its own segment files.
        let cdc_wal = if config.cdc.enabled {
            let cdc_wal_dir = data_dir.join("cdc_wal");
            match CdcWalWriter::open(cdc_wal_dir) {
                Ok(writer) => {
                    tracing::info!("CDC WAL opened successfully");
                    Some(Arc::new(writer))
                }
                Err(e) => {
                    tracing::warn!("failed to open CDC WAL, CDC will be in-memory only: {}", e);
                    None
                }
            }
        } else {
            None
        };

        let segment_bloom_filters = BoundedBloomFilterCache::new(10000);
        if bloom_filter_dir.exists() {
            // db004 fix: iterate entries directly and propagate IO errors instead of silently ignoring them.
            let entries = std::fs::read_dir(&bloom_filter_dir)
                .map_err(RockDuckError::Io)?;
            for entry in entries {
                let entry = entry.map_err(RockDuckError::Io)?;
                let path = entry.path();
                if path.extension().and_then(|s| s.to_str()) == Some("bf") {
                    let seg_id = path.file_stem().and_then(|s| s.to_str()).map(String::from);
                    if let Some(seg_id) = seg_id {
                        // db004 fix: propagate read errors instead of ignoring
                        let data = std::fs::read(&path)
                            .map_err(RockDuckError::Io)?;
                        let bf = postcard::from_bytes::<BloomFilter>(&data)
                            .map_err(|e| RockDuckError::Codec(format!("deserialize bloom filter {}: {}", seg_id, e)))?;
                        segment_bloom_filters.insert(seg_id, bf);
                    }
                }
            }
        }

        let mut db = Arc::new(Self {
            kv,
            data_dir,
            config,
            wal,
            mvcc,
            txn_counter: AtomicU64::new(next_txn),
            delta_layer,
            segment_bloom_filters,
            seg_meta_cache: RwLock::new(SegmentMetaCache::default()),
            compaction_scheduler,
            access_tracker,
            write_heat_tracker,
            router,
            io_scheduler,
            bloom_filter_dir,
            cdc_log_buffer,
            cdc_wal,
            pending_cdc_entries,
            cdc_granularity,
            txn_delta_collectors: RwLock::new(std::collections::HashMap::new()),
            checkpoint_manager,
            last_checkpoint_txn,
            flush_engine,
            outstanding_scan_bytes,
            cached_snapshot: RwLock::new(None),
        });

        if db.config.compaction.background_enabled {
            let scheduler_arc = Arc::new(parking_lot::Mutex::new(
                db.compaction_scheduler.read().clone(),
            ));
            let io_config = IOSchedulerConfig::default();
            let compactor: Arc<dyn crate::compaction::io_scheduler::CompactionExecutor> =
                Arc::new(NonBlockingCompactor::with_access_tracker(
                    Arc::clone(&db),
                    NonBlockingConfig {
                        num_threads: db.config.compaction.max_concurrent_tasks.max(1),
                        del_ratio_threshold: db.config.compaction.del_ratio_threshold,
                        ..Default::default()
                    },
                    db.access_tracker.clone(),
                ));
            let handle = IOSchedulerHandle::with_compactor(io_config, scheduler_arc, compactor);
            Arc::get_mut(&mut db)
                .expect("db arc must be unique during runtime assembly")
                .io_scheduler = Some(handle);
            if let Some(handle) = db.io_scheduler.as_ref() {
                handle.start();
            }
        }

        Arc::try_unwrap(db).map_err(|_| {
            RockDuckError::Internal("runtime assembly leaked shared RockDuck reference".into())
        })
    }

    /// Allocate the next transaction ID, with overflow detection.
    ///
    /// ## Overflow Safety (mv001 fix)
    ///
    /// Before incrementing, checks if `txn_counter` has already reached `u64::MAX`.
    /// If so, returns an error instead of wrapping around, which would cause
    /// transaction ID collisions and violate MVCC correctness guarantees.
    ///
    /// ## Medium-term: Persistent txn_counter
    ///
    /// For crash-recovery correctness, the txn_counter should be persisted to KV
    /// so that on restart, the next txn_id does not collide with previously used IDs.
    /// Currently, txn_counter is in-memory only; a restart will allocate new IDs
    /// starting from `max(committed_txn, WAL_max) + 1`, which is safe because
    /// committed_txn provides a lower bound on the highest txn_id ever used.
    pub fn next_txn_id(&self) -> Result<TxnId> {
        let prev = self.txn_counter.load(Ordering::SeqCst);
        if prev == u64::MAX {
            return Err(RockDuckError::Internal(
                "transaction ID counter overflow: u64::MAX reached, cannot allocate new transaction"
                    .into(),
            ));
        }
        Ok(self.txn_counter.fetch_add(1, Ordering::SeqCst))
    }

    /// Begin a new transaction.
    ///
    /// Allocates the next transaction ID, registers the transaction in MVCC's active_txns
    /// table (both in-memory and persisted to KV), and returns the txn_id to the caller.
    /// All subsequent writes on this transaction must use the returned txn_id.
    pub fn begin_txn(&self) -> Result<TxnId> {
        let txn_id = self.next_txn_id()?;
        {
            let mut guard = self.cached_snapshot.write();
            *guard = None; // Invalidate cache on any MVCC mutation (mv010 fix)
        }
        self.mvcc
            .write()
            .begin_txn(txn_id, &self.kv)
            .map_err(|e| RockDuckError::Internal(format!("begin_txn failed: {}", e)))?;
        Ok(txn_id)
    }

    /// Commit a transaction: persist all writes and remove from active_txns.
    ///
    /// Write order: WAL flush (durability boundary) → MVCC commit (remove from active_txns) → persist committed_txn.
    /// On WAL failure: returns error without modifying MVCC state (txn remains in active_txns, can retry).
    /// On MVCC failure: WAL is already durable — txn appears committed in WAL but won't be visible
    /// until the process restarts and MVCC state is recovered.
    pub fn commit_txn(&self, txn_id: TxnId) -> Result<()> {
        // Invalidate snapshot cache on any MVCC mutation (mv010 fix)
        {
            let mut guard = self.cached_snapshot.write();
            *guard = None;
        }

        // Extract begin_ts from active txn metadata before committing.
        // begin_ts is persisted in the WAL so recovery can restore active_txns state.
        let begin_ts = self.mvcc.read().get_begin_ts(txn_id).ok_or_else(|| {
            RockDuckError::MvccConflict(format!(
                "commit_txn: txn {} missing begin_ts in active_txns",
                txn_id
            ))
        })?;

        // Capture inserted_at once for both WAL persistence and visibility recording.
        // Using the same value in WAL and committed_history means WAL-recovered entries
        // have the same TTL clock as live-committed entries.
        let inserted_at = crate::codec::current_timestamp_millis();

        self.wal.append_durable(
            OpType::Commit,
            txn_id,
            &OpPayload::Commit {
                begin_ts,
                inserted_at: Some(inserted_at),
            },
        )?;

        self.mvcc.write().commit_txn(txn_id, &self.kv, inserted_at)?;

        metadata::put_committed_txn(&self.kv, txn_id)?;
        // Persist commit_ts to KV for strict MVCC visibility.
        // Even if committed_history is pruned from memory, the commit_ts survives in KV.
        // Recovery loads from KV first, then WAL entries take precedence.
        metadata::commit_txn_record(&self.kv, txn_id, txn_id)?;

        // WAL+MVCC Commit Metrics (WAL-1 monitoring)
        // Track commit success to detect WAL success + KV failure patterns
        tracing::info!(
            target: "mvcc_metrics",
            txn_id,
            action = "commit_success",
            "Transaction committed successfully"
        );

        // Flush pending CDC entries for this transaction to both WAL and in-memory buffer.
        // CDC buffer full is now a hard error: transactions cannot commit if CDC buffer overflows.
        // This ensures at-least-once delivery is maintained.
        let pending: Vec<CdcLogEntry> = self
            .pending_cdc_entries
            .write()
            .remove(&txn_id)
            .unwrap_or_default();

        if !pending.is_empty() {
            // Write to CDC WAL (durable path)
            if let Some(ref cdc_wal) = self.cdc_wal {
                for entry in &pending {
                    let op = CdcOp::from_entry(entry);
                    if let Err(e) = cdc_wal.append_and_flush(op) {
                        tracing::warn!("CDC WAL write failed for txn {}: {}", txn_id, e);
                        // Continue: CDC WAL failure is non-fatal; events reconstructible from main WAL
                    }
                }
            }

            // Write to in-memory buffer (for time-travel queries).
            // BUFFER FULL IS A HARD ERROR: block transaction commit to prevent CDC event loss.
            {
                for entry in pending {
                    self.cdc_log_buffer.write()
                        .push(entry)
                        .map_err(|e| RockDuckError::Internal(format!(
                            "CDC log buffer full during commit: {} (consider increasing cdc.log_buffer_size)", e
                        )))?;
                }
            }
        }

        Ok(())
    }

    pub fn rollback_txn(&self, txn_id: TxnId) -> Result<()> {
        // Invalidate snapshot cache on any MVCC mutation (mv010 fix)
        {
            let mut guard = self.cached_snapshot.write();
            *guard = None;
        }

        // MVCC rollback must happen FIRST: if WAL write fails, the txn is still
        // cleaned from active_txns and won't block readers forever.
        // WAL rollback record is idempotent — if it fails, the txn is already gone
        // from MVCC state, and recovery will see no Rollback record (treated as
        // already-aborted on next restart, which is correct).

        // WAL+MVCC Rollback Metrics (WAL-1 monitoring)
        tracing::info!(
            target: "mvcc_metrics",
            txn_id,
            action = "rollback_start",
            "Transaction rollback initiated"
        );

        self.mvcc.write().rollback_txn(txn_id, &self.kv)?;
        if let Err(e) = self
            .wal
            .append(OpType::Rollback, txn_id, &OpPayload::Rollback)
        {
            // WAL write failed but MVCC is already cleaned — log and continue.
            // The transaction will appear as never-rolled-back in WAL replay,
            // which is fine since the txn_id is already gone from active_txns.
            tracing::error!(
                "WAL append Rollback failed for txn_id={} (MVCC state already cleaned): {}",
                txn_id,
                e
            );
            // CDC state cleanup still applies.
            self.pending_cdc_entries.write().remove(&txn_id);
            return Err(RockDuckError::Write(format!(
                "WAL rollback log failed: {e}"
            )));
        }
        self.pending_cdc_entries.write().remove(&txn_id);
        Ok(())
    }

    /// Push a CDC entry for the given transaction.
    /// Entries are accumulated in `pending_cdc_entries` and flushed at `commit_txn`.
    /// Staging never fails; actual flush errors (e.g. buffer full) are handled in `commit_txn`.
    pub fn push_cdc_entry(&self, entry: CdcLogEntry) {
        self.pending_cdc_entries
            .write()
            .entry(entry.txn_id)
            .or_default()
            .push(entry);
    }

    pub fn snapshot(&self) -> TxnSnapshot {
        // Check cache first (mv010 lazy-loading fix)
        if let Some(cached) = self.cached_snapshot.read().as_ref() {
            // Return cached snapshot for read-heavy workloads
            return cached.as_ref().clone();
        }

        // Cache miss: generate new snapshot
        let snap = self
            .mvcc
            .read()
            .snapshot(crate::mvcc::visibility::IsolationLevel::Snapshot);

        // Cache the snapshot as an Arc for cheap cloning
        let cached = Arc::new(snap.clone());
        *self.cached_snapshot.write() = Some(cached);

        snap
    }

    /// Get a snapshot with only active_txns populated (lazy-loading path).
    ///
    /// This is faster than `snapshot()` for normal OLTP reads because it avoids
    /// building the `commit_ts_map`. The `commit_ts_map` is only needed for
    /// CDC time-travel queries.
    pub fn snapshot_active_only(&self) -> TxnSnapshot {
        // Check cache first
        if let Some(cached) = self.cached_snapshot.read().as_ref() {
            return cached.as_ref().clone();
        }

        // Cache miss: generate new snapshot with active-only (no commit_ts_map)
        let snap = self
            .mvcc
            .read()
            .snapshot_with_active_only(crate::mvcc::visibility::IsolationLevel::Snapshot);

        let cached = Arc::new(snap.clone());
        *self.cached_snapshot.write() = Some(cached);

        snap
    }

    /// Get a snapshot with the full commit_ts_map populated.
    ///
    /// This is the CDC time-travel path. Use this only when you need to filter
    /// rows by their commit timestamps.
    pub fn snapshot_with_commit_ts_map(&self) -> TxnSnapshot {
        self.mvcc
            .read()
            .snapshot_with_commit_ts_map(crate::mvcc::visibility::IsolationLevel::Snapshot)
    }

    /// Scan a table, returning a `ScanIterator` over all visible RecordBatches.
    pub fn scan(&self, opts: crate::read::ScanOptions) -> ScanIterator<'_> {
        ScanIterator::new(self, opts, None)
    }

    /// Batch-oriented scan, identical to `scan` but named for batch use cases.
    pub fn scan_batches(&self, opts: crate::read::ScanOptions) -> ScanIterator<'_> {
        ScanIterator::new(self, opts, None)
    }

    pub fn persist_bloom_filters(&self) -> Result<()> {
        std::fs::create_dir_all(&self.bloom_filter_dir)
            .map_err(|e| RockDuckError::Write(format!("create bloom dir: {}", e)))?;

        let map = self.segment_bloom_filters.iter();
        for (seg_id, bf_lock) in map.iter() {
            let bf = bf_lock.lock();
            let path = self.bloom_filter_dir.join(format!("{}.bf", seg_id));
            let tmp = path.with_extension("bf.tmp");
            let data = postcard::to_allocvec(&*bf)
                .map_err(|e| RockDuckError::Write(format!("serialize bloom: {}", e)))?;
            std::fs::write(&tmp, &data)
                .map_err(|e| RockDuckError::Write(format!("write bloom tmp: {}", e)))?;
            std::fs::rename(&tmp, &path)
                .map_err(|e| RockDuckError::Write(format!("rename bloom: {}", e)))?;
        }
        Ok(())
    }

    pub fn persist_delta_checkpoint(&self, ckpt_id: u64, committed_txn: TxnId) -> Result<()> {
        let ckpt_dir = self.data_dir.join("checkpoints");
        std::fs::create_dir_all(&ckpt_dir)
            .map_err(|e| RockDuckError::Write(format!("create checkpoint dir: {}", e)))?;
        let (l2_patches, l3_patches) = self.delta_layer.patch_counts();
        let delta_ckpt = DeltaCheckpointState {
            committed_txn,
            l1_entry_count: self.delta_layer.l1.len(),
            l1_size_bytes: self.delta_layer.l1.size_bytes(),
            l2_patch_counts: l2_patches as u64,
            l3_patch_counts: l3_patches as u64,
        };
        let bytes = postcard::to_allocvec(&delta_ckpt)
            .map_err(|e| RockDuckError::Codec(format!("serialize delta checkpoint: {}", e)))?;
        let path = ckpt_dir.join(format!("delta_ckpt_{}.bin", ckpt_id));
        std::fs::write(&path, &bytes)
            .map_err(|e| RockDuckError::Write(format!("write delta checkpoint {}: {}", ckpt_id, e)))?;
        Ok(())
    }

    pub fn maybe_checkpoint(&self, txn_id: TxnId) -> Result<()> {
        let last = self.last_checkpoint_txn.load(Ordering::SeqCst);
        let txn_interval = self.config.checkpoint.txn_count_threshold;
        let wal_size_threshold = self.config.checkpoint.wal_size_threshold;

        let since_last = txn_id.saturating_sub(last);
        let should_checkpoint_by_txn = since_last >= txn_interval;

        let should_checkpoint_by_size = wal_size_threshold > 0
            && self.wal.size_bytes() >= wal_size_threshold;

        if !should_checkpoint_by_txn && !should_checkpoint_by_size {
            return Ok(());
        }

        tracing::info!(
            "triggering checkpoint at txn {} (since_last={}, wal_size={})",
            txn_id,
            since_last,
            self.wal.size_bytes()
        );

        // Flush WAL
        {
            let mut w = self.wal.get_mut_writer();
            w.flush_and_sync()
                .map_err(|e| RockDuckError::Write(format!("WAL flush: {e}")))?;
        }

        // Persist bloom filters (db005 fix: propagate errors)
        self.persist_bloom_filters()?;

        // Persist delta checkpoint (propagate errors)
        self.persist_delta_checkpoint(txn_id, txn_id)?;

        // WAL fuzzy checkpoint (non-fatal)
        let ts_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        if let Err(e) = self.checkpoint_manager.checkpoint(CheckpointState {
            checkpoint_id: txn_id,
            committed_txn: txn_id,
            timestamp_ms: ts_ms,
            replay_watermark: self.mvcc.read().replay_watermark(),
            mvcc: Some(CheckpointMvccState {
                active_txns: self.mvcc.read().active_txn_entries(),
                committed_history: self.mvcc.read().committed_history_entries(),
                visibility_config: self.mvcc.read().visibility_config().clone(),
            }),
        }) {
            tracing::warn!("fuzzy_checkpoint failed (non-fatal): {}", e);
        }

        self.last_checkpoint_txn.store(txn_id, Ordering::SeqCst);
        Ok(())
    }

    pub fn flush_compaction(&self) {
        if let Some(ref fe) = self.flush_engine {
            if let Err(e) = fe.flush_l1_to_l2() {
                tracing::warn!("flush_l1_to_l2 failed: {}", e);
            }
        }
    }

    /// Replay all committed WAL operations to reconstruct data files after a crash.
    ///
    /// This is called during `open_with_config`.  The replay is idempotent — append-mode
    /// writes and PK index updates can safely run multiple times.  On a clean shutdown
    /// (WAL was flushed and truncated), this is a no-op.
    fn replay_wal_ops(
        wal_dir: &std::path::Path,
        data_dir: &std::path::Path,
        kv: &Arc<dyn KVEngine>,
    ) -> std::result::Result<RecoveryResult, ReplayError> {
        let kv_ref = Arc::clone(kv);

        let apply = move |op: &WalOp| -> Result<()> {
            match &op.payload {
                OpPayload::Insert {
                    table,
                    pk,
                    columns,
                    wal_batch,
                    seg_id,
                    offset,
                    ..
                } => {
                    let layout = crate::segment::layout::SegmentLayout::new(data_dir, seg_id);
                    let batch = crate::write::wal_utils::ipc_stream_to_batch(wal_batch.as_ref())?;
                    for (col_idx, col_name) in columns.iter().enumerate() {
                        let arr = batch.column(col_idx);
                        crate::write::insert::write_column_data_final(
                            &layout,
                            col_name,
                            std::slice::from_ref(arr),
                        )?;
                    }
                    let vis_arrays = batch.columns()[columns.len()..].to_vec();
                    if !vis_arrays.is_empty() {
                        crate::write::insert::write_vis_column_replay(&layout, &vis_arrays)?;
                    }
                    crate::metadata::pk_skiplist::put_pk_index_double(
                        &kv_ref,
                        table,
                        pk,
                        seg_id,
                        GranuleId::zero(),
                        *offset as u32,
                    )?;
                }
                OpPayload::Update {
                    table,
                    pk,
                    columns,
                    wal_batch,
                    old_seg_id,
                    old_offset,
                    new_seg_id,
                    offset,
                    ..
                } => {
                    let old_layout =
                        crate::segment::layout::SegmentLayout::new(data_dir, old_seg_id);
                    crate::write::insert::mark_visibility_deleted_replay(
                        old_layout.vis_path(),
                        *old_offset as u32,
                        op.txn_id,
                    )?;

                    let new_layout =
                        crate::segment::layout::SegmentLayout::new(data_dir, new_seg_id);
                    let batch = crate::write::wal_utils::ipc_stream_to_batch(wal_batch.as_ref())?;
                    for (col_idx, col_name) in columns.iter().enumerate() {
                        let arr = batch.column(col_idx);
                        crate::write::insert::write_column_data_final(
                            &new_layout,
                            col_name,
                            std::slice::from_ref(arr),
                        )?;
                    }
                    let vis_arrays = batch.columns()[columns.len()..].to_vec();
                    if !vis_arrays.is_empty() {
                        crate::write::insert::write_vis_column_replay(&new_layout, &vis_arrays)?;
                    }
                    crate::metadata::pk_skiplist::put_pk_index_double(
                        &kv_ref,
                        table,
                        pk,
                        new_seg_id,
                        GranuleId::zero(),
                        *offset as u32,
                    )?;
                }
                OpPayload::Delete {
                    table,
                    pk,
                    seg_id,
                    granule_id,
                    offset,
                    ..
                } => {
                    let old_layout = crate::segment::layout::SegmentLayout::new(data_dir, seg_id);
                    crate::write::insert::mark_visibility_deleted_replay(
                        old_layout.vis_path(),
                        *offset as u32,
                        op.txn_id,
                    )?;
                    crate::metadata::pk_skiplist::delete_pk_index_double(
                        &kv_ref,
                        table,
                        pk,
                        seg_id,
                        *granule_id,
                    )?;
                }
                OpPayload::Commit { .. }
                | OpPayload::Rollback
                | OpPayload::Begin
                | OpPayload::Checkpoint { .. } => {
                    // No-op for data files.
                }
                OpPayload::Compaction {
                    old_seg_id,
                    new_seg_id,
                    pk_entries,
                    commit,
                } => {
                    if *commit {
                        // Replay compaction: deserialize pk_entries and apply the same
                        // pk_lookup + alias updates that run_compaction applied.
                        // Payload stores Vec<(pk, granule_id, row_offset)>; seg_id is at record level.
                        let entries: Vec<(Vec<u8>, u32, u32)> = crate::codec::decode(pk_entries)
                            .map_err(|e| {
                                crate::RockDuckError::Codec(format!(
                                    "compaction pk_entries decode failed: {}",
                                    e
                                ))
                            })?;

                        let mut pk_ops = Vec::with_capacity(entries.len() * 2);
                        for (pk, granule_id_raw, row_offset) in &entries {
                            let granule_id = crate::metadata::GranuleId::new(*granule_id_raw);
                            let new_value = crate::codec::encode(&(
                                new_seg_id.clone(),
                                *granule_id_raw,
                                *row_offset,
                            ))?;
                            let lookup_key = crate::metadata::pk_skiplist::pk_lookup_key(pk);
                            let old_idx_key = crate::metadata::pk_skiplist::pk_index_key(
                                old_seg_id, granule_id, pk,
                            );
                            pk_ops.push(crate::metadata::kv_engine::KVOp::Put {
                                key: lookup_key,
                                value: new_value,
                            });
                            pk_ops.push(crate::metadata::kv_engine::KVOp::Delete {
                                key: old_idx_key,
                            });
                        }
                        if !pk_ops.is_empty() {
                            kv_ref.write_batch(crate::metadata::kv_engine::CF_PK_IDX, &pk_ops)?;
                        }
                        crate::metadata::seg_alias::write_alias(&kv_ref, old_seg_id, new_seg_id)?;
                        tracing::info!(
                            "Compaction WAL replay: {} -> {} (pk_entries={})",
                            old_seg_id,
                            new_seg_id,
                            entries.len()
                        );
                    }
                    // If !commit: rollback path (write_abort WAL record to undo changes).
                    // Currently compaction always commits; rollback path reserved for future use.
                }
            }
            Ok(())
        };

        replay_committed_ops(wal_dir, &apply)
    }
}
