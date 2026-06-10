//! Flush Engine - SILK + EcoTune Compaction Scheduler.
//!
//! # Architecture
//!
//! FlushEngine has these components: ForegroundMonitor (SILK), EcoTunePolicy (SIGMOD 2025),
//! L1 (DeltaMemStore), L2 (DeltaL2Disk), L3 (DeltaL3Frozen).
//!
//! Scheduler loop (500ms): 1) SILK: check foreground load, 2) P0: L1 flush (if threshold met
//! AND fg_load < 70%), 3) P1: Guard merge (if EcoTune policy says Tiering/Lazy AND fg spare > 30%),
//! 4) P2: L2->L3 compaction (if EcoTune policy says Leveling AND fg spare > 50%),
//! 5) EcoTune: update workload profile and recompute policy.
//!
//! # SILK (ATC 2019)
//!
//! Key insight: background compaction competes with foreground queries for I/O bandwidth.
//! SILK reserves I/O bandwidth for foreground: background I/O is only allowed when
//! foreground load < reserved_threshold (default 30%).
//!
//! # EcoTune (SIGMOD 2025)
//!
//! Key insight: compaction is an investment. Write amplification is the cost; query
//! acceleration is the return. Earlier compactions give higher cumulative returns.
//! EcoTune uses DP to find the optimal compaction policy between global compactions.
//!
//! # Configurable async/sync
//!
//! - `async_enabled = true`: background scheduler thread runs continuously
//! - `async_enabled = false`: caller uses `try_flush()` / `try_compact()` explicitly
//!
//! References:
//! - SILK (ATC 2019): preventing background compaction from hurting foreground queries
//! - EcoTune (SIGMOD 2025): rethinking compaction policies in LSM-trees

use crate::compaction::scheduler::{MaintenanceTask, RewriteAction};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering as AtomicOrdering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use parking_lot::{Mutex, RwLock};

use super::disk_store::{DeltaL2Disk, GuardKey};
use super::frozen_store::DeltaL3Frozen;
use super::mem_store::DeltaMemStore;
use super::sparsity::SparsitySelector;
use super::types::{DeltaCell, DeltaPatch, ZoneMap};
use crate::error::{Result, RockDuckError};
use std::sync::atomic::AtomicUsize;

// =============================================================================
// ForegroundMonitor — SILK-style I/O load tracking
// =============================================================================

/// Tracks foreground I/O load to coordinate background compaction.
/// Based on SILK (ATC 2019): background I/O only runs when foreground
/// has spare bandwidth.
#[derive(Debug)]
pub struct ForegroundMonitor {
    /// Recent foreground I/O throughput samples (MB/s), newest last.
    /// Uses VecDeque as a ring buffer for O(1) push and eviction.
    recent_mbps: Mutex<VecDeque<f64>>,
    /// Last sample timestamp.
    last_sample: Mutex<Instant>,
    /// Sampling interval (ms).
    #[allow(dead_code)]
    interval_ms: u64,
    /// Reserved foreground ratio (SILK: 0.3 = 30% reserved for foreground).
    /// Background I/O is only allowed when (1 - fg_load) > reserved_ratio.
    reserved_ratio: f64,
    /// Current outstanding foreground I/O (estimated bytes/s).
    outstanding_io: AtomicU64,
}

impl Default for ForegroundMonitor {
    fn default() -> Self {
        Self::new(500, 0.3)
    }
}

impl ForegroundMonitor {
    pub fn new(interval_ms: u64, reserved_ratio: f64) -> Self {
        Self {
            recent_mbps: Mutex::new(VecDeque::with_capacity(64)),
            last_sample: Mutex::new(Instant::now()),
            interval_ms,
            reserved_ratio,
            outstanding_io: AtomicU64::new(0),
        }
    }

    /// Record a foreground I/O sample (MB/s).
    pub fn record_sample(&self, mbps: f64) {
        let mut recent = self.recent_mbps.lock();
        recent.push_back(mbps);
        if recent.len() > 64 {
            recent.pop_front();
        }
        *self.last_sample.lock() = Instant::now();
    }

    /// Get the current foreground load as a fraction [0.0, 1.0].
    /// 0.0 = idle, 1.0 = saturated.
    ///
    /// Uses the MAXIMUM of recent samples — SILK's key principle is that
    /// background I/O must not hurt foreground queries, so we must account
    /// for the peak foreground load (not average). Even brief saturation
    /// hurts query latency.
    pub fn current_load(&self) -> f64 {
        let recent = self.recent_mbps.lock();
        if recent.is_empty() {
            return 0.0;
        }
        // Use MAX: we care about peak foreground load
        let max_mbps = recent.iter().copied().fold(0.0f64, f64::max);
        (max_mbps / 100.0).min(1.0) // Normalize: 100 MB/s = 1.0
    }

    /// SILK principle: can background I/O run?
    ///
    /// Returns true when foreground has spare bandwidth:
    /// `(1 - current_load) > reserved_ratio`
    ///
    /// Example: reserved_ratio = 0.3 (30% reserved)
    /// - fg_load = 0.5 (50%) → spare = 0.5 > 0.3 → true
    /// - fg_load = 0.8 (80%) → spare = 0.2 < 0.3 → false
    pub fn can_run_background(&self) -> bool {
        (1.0 - self.current_load()) > self.reserved_ratio
    }

    /// SILK: set outstanding foreground I/O estimate.
    pub fn set_outstanding_io(&self, bytes_per_sec: u64) {
        self.outstanding_io
            .store(bytes_per_sec, AtomicOrdering::Relaxed);
    }

    /// SILK: get outstanding foreground I/O estimate.
    pub fn outstanding_io(&self) -> u64 {
        self.outstanding_io.load(AtomicOrdering::Relaxed)
    }
}

// =============================================================================
// EcoTune Policy Selector — SIGMOD 2025
// =============================================================================

/// Compaction policy, selected by EcoTune based on workload profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CompactionPolicy {
    /// All levels use leveling: low write amp, higher read amp.
    /// Best for read-heavy workloads.
    Leveling,
    /// Upper levels tiered, lower levels leveled.
    /// Lower write amp than Leveling.
    Tiering,
    /// Defer small merges until overlap is large enough.
    /// Best for write-heavy workloads.
    LazyLeveling,
    /// Hot data uses leveling, cold data uses tiering.
    #[default]
    HotCold,
}

/// Workload profile for EcoTune decision.
#[derive(Debug, Clone, Default)]
pub struct WorkloadProfile {
    pub read_amp: f64,
    pub write_amp: f64,
    pub p99_latency_ms: f64,
    pub queries_per_second: f64,
    pub fg_io_mbps: f64,
}

impl WorkloadProfile {
    pub fn new() -> Self {
        Self::default()
    }
}

/// EcoTune-style policy selector.
/// Uses simple heuristic rules derived from EcoTune's DP analysis:
/// - High write amp + low read amp → Tiering
/// - High read amp + high latency → Leveling
/// - High QPS + moderate amp → LazyLeveling
/// - Default → HotCold
#[derive(Debug)]
pub struct EcoTuneSelector {
    profile: RwLock<WorkloadProfile>,
    policy: RwLock<CompactionPolicy>,
}

impl Default for EcoTuneSelector {
    fn default() -> Self {
        Self {
            profile: RwLock::new(WorkloadProfile::default()),
            policy: RwLock::new(CompactionPolicy::HotCold),
        }
    }
}

impl EcoTuneSelector {
    pub fn new() -> Self {
        Self::default()
    }

    /// Update the workload profile (called periodically).
    pub fn update_profile(&self, profile: WorkloadProfile) {
        *self.profile.write() = profile.clone();
        *self.policy.write() = Self::select_policy_internal(&profile);
    }

    /// Get the current selected policy.
    pub fn policy(&self) -> CompactionPolicy {
        *self.policy.read()
    }

    /// Get the current workload profile.
    pub fn profile(&self) -> WorkloadProfile {
        self.profile.read().clone()
    }

    /// Select policy based on workload profile.
    ///
    /// Based on EcoTune (SIGMOD 2025):
    /// > "The earlier a compaction is conducted, the greater the cumulative future returns."
    /// > "When the LSM-tree is far from a global compaction, compacting multiple sorted runs
    /// > into one improves query speed for a longer period."
    ///
    /// Rule matrix (simplified from EcoTune DP):
    /// - write_amp > 10 && read_amp < 2 → Tiering (minimize write amp)
    /// - read_amp > 5 && p99 > 50ms → Leveling (minimize read amp)
    /// - queries_per_second > 1000 → LazyLeveling (defer small merges)
    /// - default → HotCold
    fn select_policy_internal(profile: &WorkloadProfile) -> CompactionPolicy {
        // Tiering: write-heavy, read-light
        if profile.write_amp > 10.0 && profile.read_amp < 2.0 {
            return CompactionPolicy::Tiering;
        }
        // Leveling: read-heavy, high latency
        if profile.read_amp > 5.0 && profile.p99_latency_ms > 50.0 {
            return CompactionPolicy::Leveling;
        }
        // LazyLeveling: very high QPS
        if profile.queries_per_second > 1000.0 {
            return CompactionPolicy::LazyLeveling;
        }
        // Default: hot-cold separation
        CompactionPolicy::HotCold
    }

    /// Estimate expected return from a compaction task.
    /// EcoTune: return = read_amp_reduction * discount - cost
    pub fn expected_return(&self, cost_mb: f64, rounds_from_global: usize) -> f64 {
        let _profile = self.profile.read().clone();
        // Simplified: assume each compaction reduces read_amp by 0.5
        let read_amp_reduction = 0.5;
        // Discount: earlier compactions give higher cumulative returns
        let discount = 1.0 / (1.0 + rounds_from_global as f64 * 0.2);
        read_amp_reduction * discount - cost_mb * 0.1
    }
}

// =============================================================================
// CompactionDecision — scheduler decision
// =============================================================================

/// Priority levels for compaction actions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum CompactionPriority {
    /// P0: L1 flush — must not be delayed
    Flush = 0,
    /// P1: L1→L2 minor compaction — affects write latency
    Minor = 1,
    /// P2: Guard merge / L2→L3 major compaction — affects read performance
    Major = 2,
}

/// A compaction action selected by the scheduler.
#[derive(Debug, Clone)]
pub enum CompactionAction {
    /// L1 ping-pong swap and drain to L2.
    L1Flush,
    /// Minor compaction: L1 → L2.
    L1ToL2 { column: String },
    /// Guard merge: compact patches within a guard.
    GuardMerge { guard_key: GuardKey },
    /// Major compaction: L2 → L3.
    L2ToL3 { seg_id: String },
}

impl CompactionAction {
    pub fn priority(&self) -> CompactionPriority {
        match self {
            Self::L1Flush => CompactionPriority::Flush,
            Self::L1ToL2 { .. } => CompactionPriority::Minor,
            Self::GuardMerge { .. } => CompactionPriority::Major,
            Self::L2ToL3 { .. } => CompactionPriority::Major,
        }
    }

    pub fn rewrite_action(&self) -> RewriteAction {
        match self {
            Self::L1Flush => RewriteAction::FlushL1ToL2,
            Self::L1ToL2 { .. } => RewriteAction::FlushL1ToL2,
            Self::GuardMerge { .. } => RewriteAction::GuardMerge,
            Self::L2ToL3 { .. } => RewriteAction::CompactL2ToL3,
        }
    }

    pub fn maintenance_task(&self) -> MaintenanceTask {
        let seg_id = match self {
            Self::L1Flush => None,
            Self::L1ToL2 { column } => Some(column.clone()),
            Self::GuardMerge { guard_key } => Some(guard_key.seg_id.clone()),
            Self::L2ToL3 { seg_id } => Some(seg_id.clone()),
        };
        MaintenanceTask {
            action: self.rewrite_action(),
            seg_id,
            size_bytes: 0,
        }
    }
}

/// A compaction decision produced by the scheduler.
#[derive(Debug, Clone)]
pub struct CompactionDecision {
    pub action: CompactionAction,
    pub maintenance: MaintenanceTask,
    pub priority: CompactionPriority,
    pub expected_benefit: f64,
    pub concurrency: usize,
}

impl CompactionDecision {
    pub fn new(action: CompactionAction, expected_benefit: f64, concurrency: usize) -> Self {
        let maintenance = action.maintenance_task();
        Self {
            priority: action.priority(),
            maintenance,
            action,
            expected_benefit,
            concurrency,
        }
    }
}

// =============================================================================
// FlushEngine — SILK + EcoTune Compaction Scheduler
// =============================================================================

/// Flush Engine — coordinates L1→L2 and L2→L3 data movement.
///
/// Coordinates three responsibilities:
/// 1. **L1 flush**: when hot buffer exceeds threshold AND foreground load < 70%
/// 2. **Guard merge**: when L2 guard has too many patches (EcoTune policy-aware)
/// 3. **L2→L3 compaction**: when EcoTune says Leveling AND foreground has spare
pub struct FlushEngine {
    /// Reference to L1 memstore.
    l1: Arc<DeltaMemStore>,
    /// Reference to L2 disk store.
    l2: Arc<DeltaL2Disk>,
    /// Reference to L3 frozen store.
    l3: Arc<DeltaL3Frozen>,
    /// L2 patch count (shared via Arc, updated by FlushEngine, read by checkpoint).
    l2_patch_count: Arc<AtomicUsize>,
    /// L3 patch count (incremented on L2→L3 compaction).
    l3_patch_count: Arc<AtomicUsize>,

    /// SILK foreground monitor.
    fg_monitor: ForegroundMonitor,
    /// EcoTune policy selector.
    ecotune: EcoTuneSelector,

    /// L2 patch count threshold to trigger compaction.
    /// Thread-safe: updated atomically via `set_compaction_threshold`.
    compaction_threshold: AtomicU32,

    /// Background scheduler handle.
    handle: RwLock<Option<thread::JoinHandle<()>>>,
    /// Shutdown flag.
    shutdown: AtomicBool,
    /// Sparsity selector for patch building.
    sparsity_selector: SparsitySelector,

    /// Configurable: async or sync mode.
    async_enabled: bool,
    /// Scheduler tick interval (ms).
    tick_interval_ms: u64,
    /// Minimum interval between flushes (ms).
    min_flush_interval_ms: u64,
    /// Last flush timestamp.
    last_flush: RwLock<Instant>,
    /// Shared counter of outstanding foreground scan I/O bytes (from scan.rs).
    outstanding_scan_bytes: Arc<AtomicU64>,
    /// Recent flush cache for race window prevention in query_all_layers.
    #[allow(clippy::type_complexity)]
    recent_flush_cache:
        Option<Arc<RwLock<BTreeMap<(String, String, u64), DeltaCell>>>>,
    /// D5 fix: flush epoch counter. Incremented after each L1→L2 flush.
    /// Enables read_changes to detect concurrent flush and retry.
    flush_epoch: Arc<AtomicU64>,
}

impl FlushEngine {
    /// Create a new flush engine.
    pub fn new(
        l1: Arc<DeltaMemStore>,
        l2: Arc<DeltaL2Disk>,
        l3: Arc<DeltaL3Frozen>,
        l2_patch_count: Arc<AtomicUsize>,
        l3_patch_count: Arc<AtomicUsize>,
        outstanding_scan_bytes: Arc<AtomicU64>,
        compaction_threshold: usize,
    ) -> Self {
        Self {
            l1,
            l2,
            l3,
            l2_patch_count,
            l3_patch_count,
            outstanding_scan_bytes,
            fg_monitor: ForegroundMonitor::default(),
            ecotune: EcoTuneSelector::new(),
            compaction_threshold: AtomicU32::new(compaction_threshold as u32),
            handle: RwLock::new(None),
            shutdown: AtomicBool::new(false),
            sparsity_selector: SparsitySelector::new(),
            async_enabled: false,
            tick_interval_ms: 500,
            min_flush_interval_ms: 100,
            last_flush: RwLock::new(Instant::now()),
            recent_flush_cache: None,
            flush_epoch: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Create a new flush engine with a pre-configured recent flush cache.
    /// This is the recommended constructor for production use with D8 race condition fix.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub fn new_with_cache(
        l1: Arc<DeltaMemStore>,
        l2: Arc<DeltaL2Disk>,
        l3: Arc<DeltaL3Frozen>,
        l2_patch_count: Arc<AtomicUsize>,
        l3_patch_count: Arc<AtomicUsize>,
        outstanding_scan_bytes: Arc<AtomicU64>,
        compaction_threshold: usize,
        recent_flush_cache: Arc<RwLock<BTreeMap<(String, String, u64), DeltaCell>>>,
        // D5 fix: flush_epoch to increment after each flush completes
        flush_epoch: Arc<AtomicU64>,
    ) -> Self {
        Self {
            l1,
            l2,
            l3,
            l2_patch_count,
            l3_patch_count,
            outstanding_scan_bytes,
            fg_monitor: ForegroundMonitor::default(),
            ecotune: EcoTuneSelector::new(),
            compaction_threshold: AtomicU32::new(compaction_threshold as u32),
            handle: RwLock::new(None),
            shutdown: AtomicBool::new(false),
            sparsity_selector: SparsitySelector::new(),
            async_enabled: false,
            tick_interval_ms: 500,
            min_flush_interval_ms: 100,
            last_flush: RwLock::new(Instant::now()),
            recent_flush_cache: Some(recent_flush_cache),
            flush_epoch,
        }
    }

    /// Configure async mode.
    pub fn with_async(mut self, async_enabled: bool, tick_interval_ms: u64) -> Self {
        self.async_enabled = async_enabled;
        self.tick_interval_ms = tick_interval_ms;
        self
    }

    /// Configure foreground monitor.
    pub fn with_fg_monitor(mut self, interval_ms: u64, reserved_ratio: f64) -> Self {
        self.fg_monitor = ForegroundMonitor::new(interval_ms, reserved_ratio);
        self
    }

    /// Set the recent flush cache for race window prevention.
    /// This enables query_all_layers to see recently flushed deltas during the flush window.
    /// Must be called before starting the flush engine.
    #[allow(clippy::type_complexity)]
    pub fn set_recent_flush_cache(
        &mut self,
        recent_flush: Arc<RwLock<BTreeMap<(String, String, u64), DeltaCell>>>,
    ) {
        self.recent_flush_cache = Some(recent_flush);
    }

    /// Set the L2 patch count threshold that triggers compaction.
    /// Thread-safe: can be called at runtime without locking.
    pub fn set_compaction_threshold(&self, threshold: u32) {
        self.compaction_threshold
            .store(threshold, AtomicOrdering::Relaxed);
    }

    /// Read the current compaction threshold.
    #[allow(dead_code)]
    fn compaction_threshold(&self) -> u32 {
        self.compaction_threshold.load(AtomicOrdering::Relaxed)
    }

    /// Start the background scheduler thread.
    pub fn start(self: &Arc<Self>) {
        if !self.async_enabled {
            return;
        }
        let me = Arc::clone(self);
        let handle = thread::spawn(move || {
            me.run_loop();
        });
        *self.handle.write() = Some(handle);
    }

    /// Stop the scheduler thread.
    pub fn stop(&self) {
        self.shutdown.store(true, AtomicOrdering::SeqCst);
        if let Some(h) = self.handle.write().take() {
            let _ = h.join();
        }
    }

    /// SILK: poll outstanding scan I/O bytes and feed them to the foreground monitor.
    fn poll_outstanding_io(&self) {
        let bytes = self.outstanding_scan_bytes.swap(0, AtomicOrdering::Relaxed);
        if bytes > 0 {
            // bytes per 500ms tick → bytes per second
            let bytes_per_sec = bytes * 2;
            self.fg_monitor.set_outstanding_io(bytes_per_sec);
        }
    }

    /// Main scheduler loop (only runs when async_enabled).
    fn run_loop(&self) {
        loop {
            if self.shutdown.load(AtomicOrdering::SeqCst) {
                break;
            }
            thread::sleep(Duration::from_millis(self.tick_interval_ms));

            if self.shutdown.load(AtomicOrdering::SeqCst) {
                break;
            }

            // SILK: poll outstanding scan I/O and update foreground monitor
            self.poll_outstanding_io();

            // Update workload profile for EcoTune
            self.update_profile();

            // P0: L1 flush (SILK: check fg load first)
            if let Some(decision) = self.check_p0_flush() {
                if let Err(e) = self.execute(decision) {
                    tracing::error!("P0 flush failed: {}", e);
                }
            }

            // P1/P2: Compaction (SILK: only if spare bandwidth)
            if self.fg_monitor.can_run_background() {
                if let Some(decision) = self.select_compaction() {
                    if let Err(e) = self.execute(decision) {
                        tracing::error!("Compaction failed: {}", e);
                    }
                }
            }
        }
    }

    /// SILK P0: Check if L1 flush should run.
    fn check_p0_flush(&self) -> Option<CompactionDecision> {
        let fg_load = self.fg_monitor.current_load();

        // SILK: don't flush if foreground load > 70%
        if fg_load > 0.7 {
            tracing::trace!("P0 deferred: fg_load={:.2} > 0.7", fg_load);
            return None;
        }

        if !self.l1.should_flush() {
            return None;
        }

        // Check min interval
        let elapsed = self.last_flush.read().elapsed();
        if elapsed < Duration::from_millis(self.min_flush_interval_ms) {
            return None;
        }

        Some(CompactionDecision::new(
            CompactionAction::L1Flush,
            self.l1.size_bytes() as f64 * 0.5,
            1,
        ))
    }

    /// SILK + EcoTune P1/P2: Select next compaction task.
    fn select_compaction(&self) -> Option<CompactionDecision> {
        let policy = self.ecotune.policy();

        match policy {
            CompactionPolicy::Tiering | CompactionPolicy::LazyLeveling => {
                // Tiering: only major compaction (L2→L3)
                self.select_l2_to_l3_compaction()
            }
            CompactionPolicy::Leveling => {
                // Leveling: minor (Guard merge) first
                self.select_guard_merge()
                    .or_else(|| self.select_l2_to_l3_compaction())
            }
            CompactionPolicy::HotCold => {
                // HotCold: hybrid — minor if guards need it, major if beneficial
                self.select_guard_merge()
                    .or_else(|| self.select_l2_to_l3_compaction())
            }
        }
    }

    fn select_guard_merge(&self) -> Option<CompactionDecision> {
        let guard_key = self.l2.find_overloaded_guard()?;
        let total_entries = self.l2.num_entries();
        let benefit = total_entries as f64 * 0.1;
        Some(CompactionDecision::new(
            CompactionAction::GuardMerge { guard_key },
            benefit,
            1,
        ))
    }

    fn select_l2_to_l3_compaction(&self) -> Option<CompactionDecision> {
        let _profile = self.ecotune.profile();
        let benefit = self.ecotune.expected_return(10.0, 1);
        let candidate_seg_id = self
            .l2
            .find_overloaded_guard()
            .map(|guard_key| guard_key.seg_id)?;

        if benefit > 0.0 {
            Some(CompactionDecision::new(
                CompactionAction::L2ToL3 {
                    seg_id: candidate_seg_id,
                },
                benefit,
                2,
            ))
        } else {
            None
        }
    }

    /// Execute a compaction decision.
    fn execute(&self, decision: CompactionDecision) -> Result<()> {
        match decision.maintenance.action {
            RewriteAction::FlushL1ToL2 => {
                self.flush_l1_to_l2()?;
                *self.last_flush.write() = Instant::now();
            }
            RewriteAction::GuardMerge => {
                if let CompactionAction::GuardMerge { guard_key } = decision.action {
                    tracing::debug!("Guard merge: {:?}", guard_key);
                    self.l2.schedule_guard_merge(&guard_key)?;
                }
            }
            RewriteAction::CompactL2ToL3 => {
                if let CompactionAction::L2ToL3 { seg_id } = decision.action {
                    self.compact_l2_to_l3(&seg_id)?;
                }
            }
            RewriteAction::MetadataEvidenceRefresh => {
                tracing::debug!(
                    "Metadata evidence refresh is modeled but not executed in FlushEngine yet"
                );
            }
            RewriteAction::CheckpointPrivilege => {
                tracing::debug!(
                    "Checkpoint privilege remains outside generic FlushEngine execution"
                );
            }
            RewriteAction::PdtMerge
            | RewriteAction::SmallFileMerge
            | RewriteAction::QueryDriven => {
                tracing::debug!(
                    "FlushEngine received non-delta rewrite action {:?}; execution remains elsewhere",
                    decision.maintenance.action
                );
            }
        }
        Ok(())
    }

    /// Update EcoTune workload profile from recent observations.
    fn update_profile(&self) {
        let fg_load = self.fg_monitor.current_load();
        let _fg_io = self.fg_monitor.outstanding_io() as f64 / 1_048_576.0; // bytes -> MB

        // Simplified profile estimation
        let mut profile = WorkloadProfile::new();
        profile.fg_io_mbps = fg_load * 100.0; // rough normalization

        // Read/write amp estimation (simplified)
        // In production: track actual read/write amplification
        profile.read_amp = 2.0 + fg_load * 3.0;
        profile.write_amp = 5.0;

        self.ecotune.update_profile(profile);
    }

    /// SILK: record a foreground I/O sample.
    pub fn record_fg_load(&self, mbps: f64) {
        self.fg_monitor.record_sample(mbps);
    }

    /// Flush L1 to L2 — drain the cold buffer and write patches.
    ///
    /// Called by the scheduler (async) or directly (sync).
    /// Groups deltas by (seg_id, column), builds patches, appends to L2.
    ///
    /// # WAL/Delta Ordering Invariant (D4 fix)
    ///
    /// WAL durability MUST be achieved BEFORE Delta data reaches L2.
    /// This ensures that if a crash occurs after L2 write but before WAL sync,
    /// the WAL can replay the transaction and L2 state is recoverable.
    pub fn flush_l1_to_l2(&self) -> Result<usize> {
        let all = self.l1.drain_after_swap();

        // D8 fix: cache recently flushed deltas to prevent race window loss in query_all_layers.
        // This ensures that if a flush happens between L1 and L2 reads, the flushed data
        // is still visible via the recent_flush cache.
        if let Some(ref cache) = self.recent_flush_cache {
            let mut recent = cache.write();
            recent.clear();
            for delta in &all {
                let key = (delta.seg_id.clone(), delta.column.clone(), delta.row_offset);
                recent.insert(key, delta.clone());
            }
        }

        // D4 fix: SILK ordering guarantee — WAL must be durable before L2 write.
        // Sync any buffered WAL entries BEFORE appending to L2.
        // This prevents data loss if a crash occurs between L2 append and WAL sync.
        self.l1.sync_wal_before_flush()?;

        // Group by (seg_id, column)
        let mut groups: HashMap<(String, String), Vec<DeltaCell>> = HashMap::new();
        for delta in all {
            groups
                .entry((delta.seg_id.clone(), delta.column.clone()))
                .or_default()
                .push(delta);
        }

        let num_groups = groups.len();

        // Write each group to L2
        for ((seg_id, col), cells) in groups {
            let patch = build_patch_from_cells(&cells, &self.sparsity_selector)?;
            self.l2.append_patch(&seg_id, &col, &patch)?;
            // Increment count AFTER append succeeds. If append_patch returns Err,
            // the ? above propagates and this line is never reached, so the counter
            // stays consistent with actual persisted state.
            self.l2_patch_count
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }

        // D8 fix: clear the recent flush cache after flush completes.
        // This prevents stale data from being returned in subsequent queries.
        // D5 fix: also increment flush_epoch so concurrent readers can detect the flush.
        if let Some(ref cache) = self.recent_flush_cache {
            let mut recent = cache.write();
            recent.clear();
        }
        self.flush_epoch.fetch_add(1, std::sync::atomic::Ordering::SeqCst);

        tracing::debug!("Flushed L1→L2: {} column groups", num_groups);
        Ok(num_groups)
    }

    /// Flush L1 to L2 (sync, for use by external callers).
    pub fn try_flush(&self) -> Result<Option<usize>> {
        if self.l1.should_flush() {
            Ok(Some(self.flush_l1_to_l2()?))
        } else {
            Ok(None)
        }
    }

    /// Compact L2 to L3.
    ///
    /// Minimal implementation: selects guards with the oldest txn ranges,
    /// reads their patches, and writes them to L3 as frozen patches.
    /// Full implementation would merge patches with base column files
    /// (vectorized Arrow kernels) to produce compacted data.
    pub fn compact_l2_to_l3(&self, seg_id: &str) -> Result<()> {
        let guards_to_compact: Vec<_> = {
            let mut candidates = self.l2.guards_by_seg(seg_id);
            candidates.sort_by_key(|&(_, min_txn)| min_txn);
            candidates.into_iter().take(4).collect()
        };

        if guards_to_compact.is_empty() {
            tracing::debug!("L2→L3: no guards to compact for seg {}", seg_id);
            return Ok(());
        }

        let mut compacted_count = 0usize;
        for (guard_key, min_txn) in guards_to_compact {
            // Read all patches from this guard
            let patches = self.l2.read_all_patches_from_guard(&guard_key)?;
            if patches.is_empty() {
                continue;
            }

            let output_count = patches.len();

            // Write each patch to L3 as a frozen patch.
            // Fix #5: Each patch needs a unique patch_id to avoid overwriting.
            // Using min_txn for all patches in the same guard causes data loss
            // when multiple patches write to the same L3 file path.
            // Use a counter to generate unique patch_ids.
            for (patch_idx, patch) in patches.iter().enumerate() {
                let unique_patch_id = min_txn * 1000 + patch_idx as u64;
                self.l3.write_compacted(
                    seg_id,
                    &guard_key.col,
                    unique_patch_id,
                    patch.format.clone(),
                    ZoneMap {
                        min_txn: patch.txn_range.0,
                        max_txn: patch.txn_range.1,
                        affected_rows: 0,
                        min_value: None,
                        max_value: None,
                    },
                )?;
            }

            // Remove compacted patches from L2 (delete the guard file)
            if let Err(e) = self.l2.delete_guard_file(&guard_key) {
                tracing::warn!("L2→L3: failed to delete guard file: {}", e);
            }

            compacted_count += output_count;
            tracing::info!(
                "L2→L3: compacted {} patches from guard {:?} (seg={}, col={})",
                output_count,
                guard_key,
                seg_id,
                guard_key.col
            );
        }

        tracing::info!(
            "L2→L3 compaction complete for seg {}: {} patches compacted",
            seg_id,
            compacted_count
        );

        // Update patch counters: L2 patches removed, L3 patches added.
        // Fix #4: Use saturating_sub to prevent underflow if compacted_count > actual count.
        // This can happen if delete_guard_file fails and guard is re-selected.
        self.l3_patch_count
            .fetch_add(compacted_count, std::sync::atomic::Ordering::Relaxed);
        let _ = self.l2_patch_count
            .fetch_update(std::sync::atomic::Ordering::Relaxed, std::sync::atomic::Ordering::Relaxed, |v| {
                Some(v.saturating_sub(compacted_count))
            });

        Ok(())
    }

    /// Try to run compaction (sync, for use by external callers).
    ///
    /// Returns the compaction decision if one was selected, and executes it immediately.
    /// Returns None if foreground load is too high to run background compaction.
    ///
    /// NOTE: This method executes the compaction inline. For async scheduling, use
    /// `select_compaction()` and submit the decision to the scheduler separately.
    pub fn try_compact(&self) -> Result<Option<CompactionDecision>> {
        match self.select_compaction() {
            Some(decision) if self.fg_monitor.can_run_background() => {
                self.execute(decision.clone())?;
                Ok(Some(decision))
            }
            other => Ok(other),
        }
    }

    /// Access the foreground monitor (for testing).
    pub fn fg_monitor(&self) -> &ForegroundMonitor {
        &self.fg_monitor
    }

    /// Access the EcoTune selector (for testing).
    pub fn ecotune(&self) -> &EcoTuneSelector {
        &self.ecotune
    }
}

impl Drop for FlushEngine {
    fn drop(&mut self) {
        self.stop();
    }
}

// =============================================================================
// Patch building utilities
// =============================================================================

/// Build a DeltaPatch from a list of DeltaCells for the same (seg_id, col).
pub fn build_patch_from_cells(
    cells: &[DeltaCell],
    selector: &SparsitySelector,
) -> Result<DeltaPatch> {
    if cells.is_empty() {
        return Err(RockDuckError::Internal(
            "Cannot build patch from empty cells".into(),
        ));
    }

    let mut cells = cells.to_vec();
    cells.sort_by_key(|c| c.txn_id);

    let seg_id = cells[0].seg_id.clone();
    let col = cells[0].column.clone();
    let txn_range = (
        cells.first().map(|c| c.txn_id).unwrap_or(0),
        cells.last().map(|c| c.txn_id).unwrap_or(0),
    );

    // Build index: row_offset -> latest DeltaCell (highest txn wins for dedup)
    let mut index: rustc_hash::FxHashMap<u64, &DeltaCell> = rustc_hash::FxHashMap::default();
    for cell in &cells {
        if let Some(existing) = index.get(&cell.row_offset) {
            if cell.txn_id > existing.txn_id {
                index.insert(cell.row_offset, cell);
            }
        } else {
            index.insert(cell.row_offset, cell);
        }
    }

    // Extract sorted positions and corresponding values in one pass — O(N)
    let mut positions: Vec<u64> = index.keys().copied().collect();
    positions.sort_unstable();

    let mut values_bytes: Vec<Vec<u8>> = Vec::with_capacity(positions.len());
    for pos in &positions {
        if let Some(cell) = index.get(pos) {
            let val = cell
                .after
                .as_ref()
                .map(|v| (**v).clone())
                .unwrap_or_default();
            values_bytes.push(val);
        } else {
            values_bytes.push(Vec::new());
        }
    }

    let total_rows = positions.iter().max().copied().unwrap_or(0) as u64 + 1;

    use arrow_array::BinaryArray;
    let values_binary = BinaryArray::from_vec(values_bytes.iter().map(|v| v.as_slice()).collect());

    let format = selector.build_format(
        &positions.iter().map(|p| *p as u32).collect::<Vec<u32>>(),
        &values_binary,
        total_rows,
        txn_range.1,
    );
    let affected = positions.len() as u64;
    let patch_id = txn_range.1;

    let zone_map = ZoneMap {
        min_txn: txn_range.0,
        max_txn: txn_range.1,
        affected_rows: affected,
        min_value: None,
        max_value: None,
    };

    Ok(DeltaPatch {
        seg_id,
        column: col,
        patch_id,
        txn_range,
        format,
        zone_map,
    })
}

// =============================================================================
