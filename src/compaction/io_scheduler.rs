//! CompactionIOScheduler — SILK (ATC 2019) I/O scheduling for compaction.
//!
//! Coordinates foreground (client) and background (compaction) I/O to minimize tail latency.
//!
//! # SILK Three Principles
//!
//! 1. **Opportunistic bandwidth**: when foreground load is low, allocate spare I/O bandwidth
//!    to internal compaction operations.
//! 2. **Priority scheduling**: flush > L0→L1 (low level) > L2+ (high level).
//!    Lower levels affect client latency more immediately.
//! 3. **Preemption**: low-level compaction tasks can preempt high-level ones.
//!
//! Current default-runtime note: the scheduler worker may be started in the default runtime,
//! but query-driven and typed-debt dispatch still remain non-authoritative unless their
//! explicit producer/admission gates are satisfied.
//!
//! # Architecture
//!
//! A background worker thread pulls from three priority queues (P0/P1/P2).
//! A token bucket rate-limits compaction I/O. A system load monitor tracks
//! foreground bandwidth utilization every `monitor_interval_ms` milliseconds.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering as AtomicOrdering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::{Condvar, Mutex};
use tracing::{info, warn};

use super::ecotune::WorkloadProfile;

// We reuse CompactionTask and the shared maintenance action surface from the scheduler module.
use crate::compaction::scheduler::{CompactionStrategy, CompactionTask, RewriteAction};
use crate::RockDuckError;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Configuration for the I/O scheduler.
#[derive(Debug, Clone)]
pub struct IOSchedulerConfig {
    /// Total I/O bandwidth available to the LSM store (MB/s).
    pub total_bandwidth_mbps: u64,
    /// Fraction of bandwidth reserved for foreground (client) ops (0.0–1.0).
    /// SILK default: 0.3 (30% reserved).
    pub foreground_reserved_ratio: f64,
    /// Minimum bandwidth guaranteed for low-level compaction (MB/s).
    pub min_lowlevel_bandwidth_mbps: u64,
    /// Token bucket capacity (MB).
    pub token_bucket_capacity_mb: u64,
    /// Token bucket refill rate (MB/ms).
    pub refill_rate_mb_per_ms: u64,
    /// System load monitoring interval (ms).
    pub monitor_interval_ms: u64,
    /// Low-load watermark — if load < this, accelerate compaction (0.0–1.0).
    pub low_load_watermark: f64,
    /// Background worker poll interval (ms).
    pub worker_poll_interval_ms: u64,
}

impl Default for IOSchedulerConfig {
    fn default() -> Self {
        Self {
            total_bandwidth_mbps: 1000, // 1 GB/s
            foreground_reserved_ratio: 0.3,
            min_lowlevel_bandwidth_mbps: 50,
            token_bucket_capacity_mb: 256,
            refill_rate_mb_per_ms: 1, // 1 GB/s
            monitor_interval_ms: 10,
            low_load_watermark: 0.3,
            worker_poll_interval_ms: 50,
        }
    }
}

// ---------------------------------------------------------------------------
// CompactionPriority (SILK P0/P1/P2)
// ---------------------------------------------------------------------------

/// SILK compaction priority levels.
///
/// - **Flush (P0)**: memtable flush — must be fast to avoid blocking writes.
/// - **LowLevel (P1)**: L0→L1 compaction — must run to prevent L0 overflow.
/// - **HighLevel (P2)**: L2+ compaction — can be deferred without immediate client impact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum CompactionPriority {
    /// P0: Flush — highest priority. Affects write latency directly.
    Flush = 0,
    /// P1: Low-level compaction (L0→L1 style). Prevents L0 overflow.
    LowLevel = 1,
    /// P2: High-level compaction (L2+). Can be preempted.
    HighLevel = 2,
}

impl CompactionPriority {
    /// Determine priority from segment age (in transactions).
    /// Newer segments → higher priority (affects write latency). Older → lower priority.
    pub fn from_segment_age(created_txn: u64, now_txn: u64) -> Self {
        let age = now_txn.saturating_sub(created_txn);
        if age < 10 {
            CompactionPriority::Flush
        } else if age < 10000 {
            CompactionPriority::LowLevel
        } else {
            CompactionPriority::HighLevel
        }
    }

    /// Estimate the I/O cost per unit of compaction work at this priority level.
    pub fn io_cost_factor(&self) -> f64 {
        match self {
            CompactionPriority::Flush => 0.1,
            CompactionPriority::LowLevel => 0.5,
            CompactionPriority::HighLevel => 1.0,
        }
    }
}

/// Extend CompactionTask with SILK priority.
#[derive(Debug, Clone)]
pub struct SchedulableTask {
    pub task: CompactionTask,
    pub priority: CompactionPriority,
}

impl SchedulableTask {
    pub fn new(task: CompactionTask, priority: CompactionPriority) -> Self {
        Self { task, priority }
    }
}

/// Task ordering for BinaryHeap: highest priority (most urgent) pops first.
impl Ord for SchedulableTask {
    fn cmp(&self, other: &Self) -> Ordering {
        // BinaryHeap is a max-heap: highest Ord value wins.
        //
        // For SILK priority: Flush (P0) is most urgent, then LowLevel (P1), then HighLevel (P2).
        // Since Flush(0) < LowLevel(1) < HighLevel(2) in enum ordinals,
        // we INVERT so that the most urgent (smallest enum) is "greatest" in max-heap.
        //
        // For equal SILK priority: higher float task.priority wins (more urgent task first).
        match self.priority.cmp(&other.priority) {
            Ordering::Equal => {
                // Equal SILK priority → compare float task priority.
                // Higher float = more urgent = should win in max-heap.
                self.task.priority.total_cmp(&other.task.priority)
            }
            // INVERT: smaller enum value = more urgent = should be "greater" to win max-heap.
            ord => ord.reverse(),
        }
    }
}

impl PartialOrd for SchedulableTask {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Eq for SchedulableTask {}
impl PartialEq for SchedulableTask {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority && self.task.seg_id == other.task.seg_id
    }
}

// ---------------------------------------------------------------------------
// SystemLoadMonitor
// ---------------------------------------------------------------------------

/// Monitors foreground (client) I/O load and determines whether the system
/// is in a low-load period where compaction can use spare bandwidth.
pub struct SystemLoadMonitor {
    config: IOSchedulerConfig,
    /// Recent foreground bandwidth samples (MB/s), newest at back.
    recent_foreground_mbps: Mutex<VecDeque<f64>>,
    /// Timestamp of last sample.
    last_sample: Mutex<Instant>,
}

impl SystemLoadMonitor {
    pub fn new(config: IOSchedulerConfig) -> Self {
        Self {
            config,
            recent_foreground_mbps: Mutex::new(VecDeque::with_capacity(100)),
            last_sample: Mutex::new(Instant::now()),
        }
    }

    /// Record a foreground (client) I/O bandwidth sample (MB/s).
    pub fn record_foreground_load(&self, mbps: f64) {
        let mut samples = self.recent_foreground_mbps.lock();
        samples.push_back(mbps);
        if samples.len() > 100 {
            samples.pop_front();
        }
        *self.last_sample.lock() = Instant::now();
    }

    /// Returns the current foreground load as a fraction of total bandwidth (0.0–1.0).
    pub fn current_load(&self) -> f64 {
        let samples = self.recent_foreground_mbps.lock();
        if samples.is_empty() {
            return 0.0;
        }
        // Use the most recent sample (SILK uses 10ms monitoring granularity).
        let latest = samples[samples.len() - 1];
        let total = self.config.total_bandwidth_mbps as f64;
        (latest / total).min(1.0)
    }

    /// Returns the average foreground load over recent samples.
    pub fn average_load(&self) -> f64 {
        let samples = self.recent_foreground_mbps.lock();
        if samples.is_empty() {
            return 0.0;
        }
        let sum: f64 = samples.iter().sum();
        let total = self.config.total_bandwidth_mbps as f64;
        (sum / samples.len() as f64 / total).min(1.0)
    }

    /// Returns true if the system is in a low-load period where compaction
    /// can use spare I/O bandwidth (SILK principle 1).
    pub fn is_low_load(&self) -> bool {
        self.current_load() < self.config.low_load_watermark
    }

    /// Returns the available bandwidth for internal (compaction) operations (MB/s).
    pub fn available_internal_bandwidth(&self) -> u64 {
        let foreground_used = {
            let samples = self.recent_foreground_mbps.lock();
            samples.back().copied().unwrap_or(0.0)
        };
        let reserved =
            self.config.total_bandwidth_mbps as f64 * self.config.foreground_reserved_ratio;
        let available = self.config.total_bandwidth_mbps as f64 - foreground_used - reserved;
        available.max(self.config.min_lowlevel_bandwidth_mbps as f64) as u64
    }
}

// ---------------------------------------------------------------------------
// TokenBucket
// ---------------------------------------------------------------------------

/// Token bucket rate limiter for compaction I/O bandwidth.
///
/// # D3 Advisory: `acquire` vs `try_acquire`
///
/// `acquire(bytes, timeout)` takes an **absolute deadline** (`Instant`), not a
/// relative duration. It blocks, polling every 5ms, until either:
/// - Tokens become available → returns `true`.
/// - `Instant::now() >= deadline` → returns `false`.
///
/// `try_acquire(bytes)` is non-blocking: it refills once and immediately returns
/// `true`/`false`.
///
/// **Current usage**: Only `try_acquire` is used in the compaction scheduler.
/// `acquire` exists for callers that need bounded blocking (e.g., foreground
/// flush requests that can wait for I/O budget). Prefer `try_acquire` for
/// background compaction loops to avoid thread parking overhead. Use `acquire`
/// when you intentionally want to wait up to `timeout` before giving up, noting
/// that the timeout is absolute — pass `Instant::now() + duration` to convert.
pub struct TokenBucket {
    state: Mutex<TokenState>,
    refill_rate_bytes_per_ms: u64,
}

struct TokenState {
    tokens_bytes: u64,
    capacity_bytes: u64,
    last_refill: Instant,
}

impl TokenBucket {
    pub fn new(capacity_mb: u64, refill_rate_mb_per_ms: u64) -> Self {
        let capacity_bytes = capacity_mb * 1024 * 1024;
        Self {
            state: Mutex::new(TokenState {
                tokens_bytes: capacity_bytes,
                capacity_bytes,
                last_refill: Instant::now(),
            }),
            refill_rate_bytes_per_ms: refill_rate_mb_per_ms * 1024 * 1024,
        }
    }

    /// Refill and try to acquire `bytes` tokens in a single lock acquisition.
    /// Returns true if successful.
    pub fn try_acquire(&self, bytes: u64) -> bool {
        let mut state = self.state.lock();
        let elapsed = state.last_refill.elapsed().as_millis() as u64;
        if elapsed > 0 {
            let refill = elapsed.saturating_mul(self.refill_rate_bytes_per_ms);
            state.tokens_bytes = state.tokens_bytes.saturating_add(refill).min(state.capacity_bytes);
            state.last_refill = Instant::now();
        }
        if state.tokens_bytes >= bytes {
            state.tokens_bytes -= bytes;
            true
        } else {
            false
        }
    }

    /// Try to acquire bytes, blocking until available or timeout.
    /// Returns true if acquired, false on timeout.
    pub fn acquire(&self, bytes: u64, timeout: Duration) -> bool {
        let deadline = Instant::now() + timeout;
        loop {
            if self.try_acquire(bytes) {
                return true;
            }
            if Instant::now() >= deadline {
                return false;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    /// Returns the current token balance (bytes).
    pub fn available(&self) -> u64 {
        let state = self.state.lock();
        let elapsed = state.last_refill.elapsed().as_millis() as u64;
        if elapsed > 0 {
            let refill = elapsed.saturating_mul(self.refill_rate_bytes_per_ms);
            state.tokens_bytes.saturating_add(refill).min(state.capacity_bytes)
        } else {
            state.tokens_bytes
        }
    }

    /// Set a new refill rate (SILK: increase when low-load, decrease when high-load).
    #[allow(dead_code)]
    pub fn set_refill_rate(&self, _mb_per_ms: u64) {
        tracing::debug!(
            "set_refill_rate called — SILK dynamic refill rate adjustment is not yet implemented. \
             Rate is fixed at construction ({} bytes/ms).",
            self.refill_rate_bytes_per_ms
        );
    }
}

// ---------------------------------------------------------------------------
// CompactionIOScheduler
// ---------------------------------------------------------------------------

/// Statistics for the I/O scheduler.
#[derive(Debug, Default)]
pub struct IOSchedulerStats {
    pub tasks_enqueued: AtomicU64,
    pub tasks_completed: AtomicU64,
    pub tasks_preempted: AtomicU64,
    pub token_wait_time_ms: AtomicU64,
    pub current_load: AtomicU64, // 0-100 as percentage
}

/// Trait for executing compaction tasks — injected into the scheduler to decouple
/// the I/O scheduling logic from the actual compaction implementation.
pub trait CompactionExecutor: Send + Sync {
    /// Execute a PDT merge compaction for the given segment.
    fn execute_pdt_merge(
        &self,
        seg_id: &str,
    ) -> crate::error::Result<crate::compaction::pdt_merge::MergeStats>;
    /// Execute a small file merge for the given segments.
    fn execute_small_file_merge(
        &self,
        seg_ids: &[String],
    ) -> crate::error::Result<crate::compaction::pdt_merge::MergeStats>;
    /// Execute query-driven compaction for the given segment.
    /// Returns `Ok(true)` only if rewrite work actually executed.
    fn execute_query_driven(&self, seg_id: &str) -> crate::error::Result<bool>;
    /// Set a callback invoked after each successful compaction.
    ///
    /// Uses interior RwLock mutability so this can be called via `Arc<dyn CompactionExecutor>`.
    /// The callback receives the new segment ID and a typed debt-resolution outcome.
    /// Proxy bootstrap outcomes remain bounded maintenance signals; measured delete-resolution
    /// outcomes may inform maintenance governance but still do not bypass query feedback authority.
    fn set_compaction_callback(
        &self,
        cb: Option<
            Box<dyn Fn(String, crate::compaction::adaptive::DebtResolutionOutcome) + Send + Sync>,
        >,
    );
}

/// The SILK I/O scheduler — coordinates compaction with foreground I/O.
pub struct CompactionIOScheduler {
    config: IOSchedulerConfig,
    monitor: SystemLoadMonitor,
    token_bucket: TokenBucket,

    /// Three priority queues (SILK P0/P1/P2).
    p0_queue: Mutex<BinaryHeap<SchedulableTask>>, // Flush
    p1_queue: Mutex<BinaryHeap<SchedulableTask>>, // LowLevel
    p2_queue: Mutex<BinaryHeap<SchedulableTask>>, // HighLevel

    /// Currently executing task (if any).
    current_task: Mutex<Option<SchedulableTask>>,

    /// Background worker state.
    running: Arc<AtomicBool>,

    /// Compaction executor — performs actual compaction work.
    /// Set via `with_compactor()` at construction time.
    compactor: Option<Arc<dyn CompactionExecutor>>,

    /// Adaptive scheduler — receives compaction result feedback for hill climbing.
    /// Set via `with_scheduler()` or `with_scheduler_and_compactor()` at construction time.
    scheduler: Option<Arc<Mutex<crate::compaction::adaptive::AdaptiveCompactionScheduler>>>,

    /// EcoTune policy selector for live workload observation on the producer path.
    ecotune: Option<crate::compaction::ecotune::EcoTuneHandle>,

    /// Worker thread handle — stored in a shared Arc so `start(self: Arc<Self>)` can save it
    /// and `stop(&self)` can join it. `let _` on the JoinHandle silently drops it and makes
    /// thread-join impossible on shutdown.
    worker_handle: Arc<Mutex<Option<std::thread::JoinHandle<()>>>>,

    /// Statistics.
    stats: IOSchedulerStats,

    /// Delay-retry queue: tasks that could not acquire tokens are re-enqueued here
    /// instead of being silently dropped. FIFO ordering ensures fairness within this queue.
    delay_retry_queue: Mutex<VecDeque<SchedulableTask>>,

    /// Condvar for efficient worker wake-up instead of fixed 50ms polling.
    wake_notify: Condvar,
}

impl CompactionIOScheduler {
    /// Create a new I/O scheduler with the given config.
    pub fn new(config: IOSchedulerConfig) -> Self {
        Self::with_scheduler_and_compactor_internal(config, None, None, None)
    }

    /// Create a new I/O scheduler with a compaction executor and adaptive scheduler.
    pub fn with_compactor(
        config: IOSchedulerConfig,
        scheduler: Arc<Mutex<crate::compaction::adaptive::AdaptiveCompactionScheduler>>,
        compactor: Arc<dyn CompactionExecutor>,
    ) -> Self {
        Self::with_scheduler_and_compactor_internal(
            config,
            Some(scheduler),
            Some(compactor),
            Some(crate::compaction::ecotune::EcoTuneHandle::default()),
        )
    }

    /// Create a new I/O scheduler with an adaptive scheduler and compaction executor.
    ///
    /// The adaptive scheduler receives compaction result callbacks via `record_compaction_result`
    /// after each successful PDT merge, enabling hill climbing weight optimization.
    pub fn with_scheduler_and_compactor(
        config: IOSchedulerConfig,
        scheduler: Arc<Mutex<crate::compaction::adaptive::AdaptiveCompactionScheduler>>,
        compactor: Arc<dyn CompactionExecutor>,
    ) -> Self {
        Self::with_scheduler_and_compactor_internal(
            config,
            Some(scheduler),
            Some(compactor),
            Some(crate::compaction::ecotune::EcoTuneHandle::default()),
        )
    }

    fn with_scheduler_and_compactor_internal(
        config: IOSchedulerConfig,
        scheduler: Option<Arc<Mutex<crate::compaction::adaptive::AdaptiveCompactionScheduler>>>,
        compactor: Option<Arc<dyn CompactionExecutor>>,
        ecotune: Option<crate::compaction::ecotune::EcoTuneHandle>,
    ) -> Self {
        let refill_mb_per_ms = config.refill_rate_mb_per_ms;
        // Clone the scheduler Arc before constructing s so the closure below
        // captures an owned clone rather than a borrowed reference into s.
        let sched_for_callback = scheduler.clone();
        let s = Self {
            config: config.clone(),
            monitor: SystemLoadMonitor::new(config.clone()),
            token_bucket: TokenBucket::new(config.token_bucket_capacity_mb, refill_mb_per_ms),
            p0_queue: Mutex::new(BinaryHeap::new()),
            p1_queue: Mutex::new(BinaryHeap::new()),
            p2_queue: Mutex::new(BinaryHeap::new()),
            current_task: Mutex::new(None),
            running: Arc::new(AtomicBool::new(false)),
            compactor,
            scheduler: scheduler.clone(),
            ecotune,
            stats: IOSchedulerStats::default(),
            worker_handle: Arc::new(Mutex::new(None)),
            delay_retry_queue: Mutex::new(VecDeque::new()),
            wake_notify: Condvar::new(),
        };
        // Wire compaction callback to adaptive scheduler (Phase F).
        if let (Some(ref sched), Some(ref comp)) = (&sched_for_callback, &s.compactor) {
            // Clone compactor Arc so we can call &mut method (set_compaction_callback takes &self
            // but uses RwLock interior mutability — clone gives us independent ownership).
            let comp_clone: Arc<dyn CompactionExecutor> = Arc::clone(comp);
            let sched_clone = Arc::clone(sched);
            comp_clone.set_compaction_callback(Some(Box::new(move |new_seg_id, outcome| {
                let timestamp = crate::codec::current_timestamp_millis();
                let mut scheduler = sched_clone.lock();
                if let Some(old_seg_id) = outcome.source_seg_id.as_deref() {
                    scheduler.observe_maintenance_feedback_seed(old_seg_id, &new_seg_id, timestamp);
                } else {
                    scheduler.feedback().record_hit(&new_seg_id, timestamp);
                }
                scheduler.record_compaction_outcome(&new_seg_id, outcome);
            })));
        }
        s
    }

    // -------------------------------------------------------------------------
    // Public API
    // -------------------------------------------------------------------------

    /// Enqueue a compaction task with the given SILK priority.
    pub fn enqueue(&self, task: CompactionTask, priority: CompactionPriority) {
        if task.maintenance.action.budget()
            == crate::compaction::scheduler::RewriteBudget::EvidenceDriven
            && task.size_bytes == 0
        {
            tracing::debug!(
                seg_id = %task.seg_id,
                action = ?task.maintenance.action,
                "SILK rejected stub-sized evidence-driven task before queue admission"
            );
            return;
        }

        let schedulable = SchedulableTask::new(task, priority);
        self.stats
            .tasks_enqueued
            .fetch_add(1, AtomicOrdering::Relaxed);

        match priority {
            CompactionPriority::Flush => self.p0_queue.lock().push(schedulable),
            CompactionPriority::LowLevel => self.p1_queue.lock().push(schedulable),
            CompactionPriority::HighLevel => self.p2_queue.lock().push(schedulable),
        }
        // Wake the worker if it's sleeping on the Condvar.
        self.wake_notify.notify_one();
    }

    /// Enqueue a task, inferring priority from segment age.
    pub fn enqueue_with_age(&self, task: CompactionTask, segment_created_txn: u64, now_txn: u64) {
        let priority = CompactionPriority::from_segment_age(segment_created_txn, now_txn);
        self.enqueue(task, priority);
    }

    /// Enqueue a task with default (medium) priority.
    pub fn enqueue_default(&self, task: CompactionTask) {
        self.enqueue(task, CompactionPriority::LowLevel);
    }

    /// Record a foreground I/O load sample (called from client write/read paths).
    pub fn record_foreground_load(&self, mbps: f64) {
        self.monitor.record_foreground_load(mbps);
        // Update load stat.
        let load_pct = (self.monitor.current_load() * 100.0) as u64;
        self.stats
            .current_load
            .store(load_pct, AtomicOrdering::Relaxed);
    }

    /// Start the background compaction worker thread.
    ///
    /// The worker thread is spawned via `Arc::clone(&self)` and runs until `stop()` is
    /// called or the owner `Arc` is dropped.
    /// The `JoinHandle` is stored in `self.worker_handle` so `stop()` can join the thread.
    pub fn start(self: Arc<Self>) -> crate::error::Result<()> {
        if self.running.load(AtomicOrdering::Acquire) {
            warn!("IOScheduler already running");
            return Ok(());
        }

        self.running.store(true, AtomicOrdering::Release);
        let running = Arc::clone(&self.running);

        let handle = std::thread::Builder::new()
            .name("silk-compaction-worker".into())
            .spawn({
                let s = Arc::clone(&self);
                move || {
                    // Note: the Condvar timeout (100ms) ensures the loop periodically checks
                    // `running`, so `stop()` will be detected even when queues are empty.
                    info!("SILK compaction worker started");
                    while running.load(AtomicOrdering::Acquire) {
                        // 1. Check delay-retry queue.
                        let task = s.delay_retry_queue.lock().pop_front();

                        let task = if let Some(t) = task {
                            t
                        } else {
                            // 2. Try priority queues (fast path — returns None if all empty).
                            match s.select_next() {
                                Some(t) => t,
                                None => {
                                    // 3. All queues empty: wait on Condvar for up to 100ms,
                                    // then loop back to check `running` flag.
                                    let mut queue = s.delay_retry_queue.lock();
                                    s.wake_notify.wait_for(&mut queue, Duration::from_millis(100));
                                    continue;
                                }
                            }
                        };

                        // 4. Try to acquire I/O budget.
                        if s.token_bucket.try_acquire(task.task.estimated_io_bytes()) {
                            s.execute_task(&task);
                        } else {
                            s.delay_retry_queue.lock().push_back(task);
                        }
                    }
                    info!("SILK compaction worker stopped");
                }
            })
            .map_err(|e| {
                RockDuckError::Internal(format!("failed to spawn compaction worker thread: {}", e))
            })?;
        *self.worker_handle.lock() = Some(handle);
        Ok(())
    }

    /// Stop the background worker and join the thread.
    pub fn stop(&self) {
        self.running.store(false, AtomicOrdering::Release);
        // Wake the worker immediately so it exits the Condvar wait promptly.
        self.wake_notify.notify_one();
        if let Some(handle) = self.worker_handle.lock().take() {
            let _ = handle.join();
        }
    }

    /// Observe the current workload on the real producer path and ask EcoTune for a live policy.
    /// Returns the workload profile if successfully collected, None otherwise.
    fn observe_ecotune_policy(&self) -> Option<WorkloadProfile> {
        let (Some(ecotune), Some(scheduler)) = (&self.ecotune, &self.scheduler) else {
            return None;
        };
        let profile = scheduler.lock().collect_workload_profile();
        let policy = ecotune.select_policy(&profile);
        let tiering_ratio = ecotune.main_level_tiering_ratio(&profile);
        let prefer_read_optimized = ecotune.prefer_read_optimized(&profile);
        let prefer_write_optimized = ecotune.prefer_write_optimized(&profile);
        let dynamic_refill = if prefer_write_optimized {
            self.config.refill_rate_mb_per_ms.saturating_mul(2).max(1)
        } else if prefer_read_optimized {
            self.config.refill_rate_mb_per_ms.max(1) / 2
        } else {
            self.config.refill_rate_mb_per_ms.max(1)
        };
        self.token_bucket.set_refill_rate(dynamic_refill.max(1));
        tracing::debug!(
            policy = ?policy,
            rq_ratio = profile.rq_ratio,
            write_speed_mbps = profile.write_speed_mbps,
            avg_selectivity = profile.avg_selectivity,
            point_query_ratio = profile.point_query_ratio,
            tiering_ratio,
            prefer_read_optimized,
            prefer_write_optimized,
            dynamic_refill,
            "EcoTune observed live producer-path workload"
        );
        Some(profile)
    }

    /// Select the next task to execute (SILK priority ordering).
    /// Returns None if all queues are empty.
    pub fn select_next(&self) -> Option<SchedulableTask> {
        // Fast path: check if any queue has tasks before doing expensive EcoTune observation.
        if self.p0_queue.lock().is_empty()
            && self.p1_queue.lock().is_empty()
            && self.p2_queue.lock().is_empty()
        {
            return None;
        }

        // Slow path: observe workload and reuse the profile to avoid duplicate lock acquisition.
        let prefer_read_optimized = self
            .ecotune
            .as_ref()
            .and_then(|ecotune| {
                self.observe_ecotune_policy()
                    .map(|profile| ecotune.prefer_read_optimized(&profile))
            })
            .unwrap_or(false);
        // SILK: P0 > P1 > P2, but allow read-optimized workloads to drain high-level
        // maintenance before low-level heuristic work when no flush is pending.
        if let Some(task) = self.p0_queue.lock().pop() {
            return Some(task);
        }
        if prefer_read_optimized {
            if let Some(task) = self.p2_queue.lock().pop() {
                return Some(task);
            }
            return self.p1_queue.lock().pop();
        }
        if let Some(task) = self.p1_queue.lock().pop() {
            return Some(task);
        }
        self.p2_queue.lock().pop()
    }

    /// Returns the number of pending tasks across all queues.
    pub fn pending_count(&self) -> usize {
        let p0 = self.p0_queue.lock().len();
        let p1 = self.p1_queue.lock().len();
        let p2 = self.p2_queue.lock().len();
        p0 + p1 + p2
    }

    /// Returns true if there are no pending tasks.
    pub fn is_empty(&self) -> bool {
        self.pending_count() == 0
    }

    /// Check if a higher-priority task can preempt the currently executing one.
    pub fn can_preempt_current(&self, incoming: &SchedulableTask) -> bool {
        if let Some(current) = self.current_task.lock().as_ref() {
            // SILK principle 3: P2 tasks can be preempted by P0/P1.
            current.priority > incoming.priority
        } else {
            false
        }
    }

    /// Get a snapshot of current statistics.
    pub fn stats(&self) -> IOSchedulerStatsSnapshot {
        IOSchedulerStatsSnapshot {
            tasks_enqueued: self.stats.tasks_enqueued.load(AtomicOrdering::Relaxed),
            tasks_completed: self.stats.tasks_completed.load(AtomicOrdering::Relaxed),
            tasks_preempted: self.stats.tasks_preempted.load(AtomicOrdering::Relaxed),
            pending_tasks: self.pending_count(),
            current_load_pct: self.stats.current_load.load(AtomicOrdering::Relaxed),
            available_bandwidth_mbps: self.monitor.available_internal_bandwidth(),
            is_low_load: self.monitor.is_low_load(),
        }
    }

    /// Returns the current system load (0.0–1.0).
    pub fn current_load(&self) -> f64 {
        self.monitor.current_load()
    }

    /// Returns true if the system is in a low-load period.
    pub fn is_low_load(&self) -> bool {
        self.monitor.is_low_load()
    }

    /// Execute a compaction task using the injected compactor (if any).
    ///
    /// If no compactor is set, logs a warning and skips execution.
    /// Success/failure is recorded in the scheduler statistics.
    fn execute_task(&self, task: &SchedulableTask) {
        let compactor = match &self.compactor {
            Some(c) => Arc::clone(c),
            None => {
                tracing::warn!(
                    "SILK: no compactor configured, skipping task for segment: {}",
                    task.task.seg_id
                );
                return;
            }
        };

        // B3-12 fix: warn when tasks are being skipped due to I/O budget exhaustion.
        // This indicates sustained foreground load preventing background compaction.
        if self.token_bucket.available() < task.task.estimated_io_bytes() {
            tracing::warn!(
                "SILK: I/O budget exhausted, task delayed for segment: {} (priority: {:?})",
                task.task.seg_id,
                task.priority
            );
        }

        let seg_id = &task.task.seg_id;
        let priority_name = format!("{:?}", task.priority);
        let io_bytes = task.task.estimated_io_bytes();
        let mut executed = false;

        match task.task.maintenance.action {
            RewriteAction::PdtMerge => match compactor.execute_pdt_merge(seg_id) {
                Ok(stats) => {
                    executed = true;
                    tracing::info!(
                            "SILK executed PDT merge: seg={} priority={} bytes={} rows_r={} rows_w={} rows_d= {}",
                            seg_id,
                            priority_name,
                            io_bytes,
                            stats.rows_read,
                            stats.rows_written,
                            stats.rows_dropped
                        );
                }
                Err(e) => {
                    tracing::error!("SILK PDT merge failed for seg={}: {}", seg_id, e);
                }
            },
            RewriteAction::SmallFileMerge => {
                match compactor.execute_small_file_merge(std::slice::from_ref(seg_id)) {
                    Ok(stats) => {
                        executed = true;
                        tracing::info!(
                            "SILK executed small file merge: seg={} rows_r={} rows_w={} rows_d={}",
                            seg_id,
                            stats.rows_read,
                            stats.rows_written,
                            stats.rows_dropped
                        );
                    }
                    Err(e) => {
                        tracing::error!("SILK small file merge failed for seg={}: {}", seg_id, e);
                    }
                }
            }
            RewriteAction::QueryDriven => {
                if task.task.size_bytes == 0 {
                    tracing::debug!(
                        "SILK rejected stub-sized QueryDriven task without runtime rewrite sizing: seg={}",
                        seg_id
                    );
                    return;
                }
                match compactor.execute_query_driven(seg_id) {
                    Ok(did_execute) => {
                        executed = did_execute;
                        if did_execute {
                            tracing::debug!(
                                "SILK executed query-driven compaction: seg={}",
                                seg_id
                            );
                        } else {
                            tracing::debug!(
                                "SILK query-driven observation-only pass: seg={} (no rewrite executed)",
                                seg_id
                            );
                        }
                    }
                    Err(e) => {
                        tracing::error!(
                            "SILK query-driven compaction failed for seg={}: {}",
                            seg_id,
                            e
                        );
                    }
                }
            }
            RewriteAction::FlushL1ToL2
            | RewriteAction::CompactL2ToL3
            | RewriteAction::GuardMerge
            | RewriteAction::MetadataEvidenceRefresh
            | RewriteAction::CheckpointPrivilege => {
                debug_assert!(
                    false,
                    "non-executable maintenance action reached SILK dispatch: {:?}",
                    task.task.maintenance.action
                );
                tracing::warn!(
                    "SILK maintenance action not executed by compactor: seg={} action={:?}",
                    seg_id,
                    task.task.maintenance.action
                );
            }
        }

        if executed {
            self.stats
                .tasks_completed
                .fetch_add(1, AtomicOrdering::Relaxed);
        }
    }
}

impl Drop for CompactionIOScheduler {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Snapshot of scheduler statistics.
#[derive(Debug, Clone)]
pub struct IOSchedulerStatsSnapshot {
    pub tasks_enqueued: u64,
    pub tasks_completed: u64,
    pub tasks_preempted: u64,
    pub pending_tasks: usize,
    pub current_load_pct: u64,
    pub available_bandwidth_mbps: u64,
    pub is_low_load: bool,
}

// ---------------------------------------------------------------------------
// Shared handle
// ---------------------------------------------------------------------------

/// Thread-safe handle to the I/O scheduler.
///
/// Uses `Arc<CompactionIOScheduler>` — all mutable state inside the scheduler
/// uses interior Mutex/Atomic types, so shared read access through Arc is safe.
/// `start()` requires taking ownership via `Arc::clone` (see `start(self: Arc<Self>)`).
#[derive(Clone)]
pub struct IOSchedulerHandle {
    inner: Arc<CompactionIOScheduler>,
}

#[cfg(test)]
#[allow(clippy::items_after_test_module)]
mod tests {
    use super::*;
    use crate::compaction::pdt_merge::MergeStats;

    struct NoopExecutor;

    struct RecordingExecutor {
        query_driven_calls: Arc<std::sync::atomic::AtomicU64>,
    }

    impl CompactionExecutor for NoopExecutor {
        fn execute_pdt_merge(&self, _seg_id: &str) -> crate::error::Result<MergeStats> {
            Ok(MergeStats::default())
        }

        fn execute_small_file_merge(
            &self,
            _seg_ids: &[String],
        ) -> crate::error::Result<MergeStats> {
            Ok(MergeStats::default())
        }

        fn execute_query_driven(&self, _seg_id: &str) -> crate::error::Result<bool> {
            Ok(false)
        }

        fn set_compaction_callback(
            &self,
            _cb: Option<
                Box<
                    dyn Fn(String, crate::compaction::adaptive::DebtResolutionOutcome)
                        + Send
                        + Sync,
                >,
            >,
        ) {
        }
    }

    impl CompactionExecutor for RecordingExecutor {
        fn execute_pdt_merge(&self, _seg_id: &str) -> crate::error::Result<MergeStats> {
            Ok(MergeStats::default())
        }

        fn execute_small_file_merge(
            &self,
            _seg_ids: &[String],
        ) -> crate::error::Result<MergeStats> {
            Ok(MergeStats::default())
        }

        fn execute_query_driven(&self, _seg_id: &str) -> crate::error::Result<bool> {
            self.query_driven_calls
                .fetch_add(1, AtomicOrdering::Relaxed);
            Ok(false)
        }

        fn set_compaction_callback(
            &self,
            _cb: Option<
                Box<
                    dyn Fn(String, crate::compaction::adaptive::DebtResolutionOutcome)
                        + Send
                        + Sync,
                >,
            >,
        ) {
        }
    }

    #[test]
    #[cfg(not(debug_assertions))]
    fn unsupported_maintenance_actions_do_not_increment_completed_counter() {
        let scheduler = CompactionIOScheduler::with_compactor(
            IOSchedulerConfig::default(),
            Arc::new(Mutex::new(
                crate::compaction::adaptive::AdaptiveCompactionScheduler::new(),
            )),
            Arc::new(NoopExecutor),
        );

        let task = CompactionTask {
            seg_id: "seg-unsupported".to_string(),
            priority: ordered_float::NotNan::new(1.0).expect("priority"),
            strategy: CompactionStrategy::PdtMerge,
            maintenance: crate::compaction::scheduler::MaintenanceTask {
                action: RewriteAction::MetadataEvidenceRefresh,
                seg_id: Some("seg-unsupported".to_string()),
                size_bytes: 1024,
            },
            created_at: crate::codec::current_timestamp_millis(),
            size_bytes: 1024,
        };

        scheduler.execute_task(&SchedulableTask::new(task, CompactionPriority::LowLevel));
        let stats = scheduler.stats();
        assert_eq!(
            stats.tasks_completed, 0,
            "unsupported actions must not masquerade as completed work"
        );
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "non-executable maintenance action reached SILK dispatch")]
    fn unsupported_maintenance_actions_fail_loudly_in_debug_builds() {
        let scheduler = CompactionIOScheduler::with_compactor(
            IOSchedulerConfig::default(),
            Arc::new(Mutex::new(
                crate::compaction::adaptive::AdaptiveCompactionScheduler::new(),
            )),
            Arc::new(NoopExecutor),
        );

        let task = CompactionTask {
            seg_id: "seg-unsupported".to_string(),
            priority: ordered_float::NotNan::new(1.0).expect("priority"),
            strategy: CompactionStrategy::PdtMerge,
            maintenance: crate::compaction::scheduler::MaintenanceTask {
                action: RewriteAction::MetadataEvidenceRefresh,
                seg_id: Some("seg-unsupported".to_string()),
                size_bytes: 1024,
            },
            created_at: crate::codec::current_timestamp_millis(),
            size_bytes: 1024,
        };

        scheduler.execute_task(&SchedulableTask::new(task, CompactionPriority::LowLevel));
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "non-executable maintenance action reached SILK dispatch")]
    fn non_executable_maintenance_actions_are_tamper_evident_in_debug() {
        let scheduler = CompactionIOScheduler::with_compactor(
            IOSchedulerConfig::default(),
            Arc::new(Mutex::new(
                crate::compaction::adaptive::AdaptiveCompactionScheduler::new(),
            )),
            Arc::new(NoopExecutor),
        );

        let task = CompactionTask {
            seg_id: "seg-debug-assert".to_string(),
            priority: ordered_float::NotNan::new(1.0).expect("priority"),
            strategy: CompactionStrategy::PdtMerge,
            maintenance: crate::compaction::scheduler::MaintenanceTask {
                action: RewriteAction::MetadataEvidenceRefresh,
                seg_id: Some("seg-debug-assert".to_string()),
                size_bytes: 1024,
            },
            created_at: crate::codec::current_timestamp_millis(),
            size_bytes: 1024,
        };

        scheduler.execute_task(&SchedulableTask::new(task, CompactionPriority::LowLevel));
    }

    #[test]
    fn stub_sized_query_driven_tasks_are_rejected_before_dispatch() {
        let scheduler = CompactionIOScheduler::with_compactor(
            IOSchedulerConfig::default(),
            Arc::new(Mutex::new(
                crate::compaction::adaptive::AdaptiveCompactionScheduler::new(),
            )),
            Arc::new(NoopExecutor),
        );

        let task = CompactionTask::new(
            "seg-query".to_string(),
            1.0,
            CompactionStrategy::QueryDriven,
            0,
        );
        scheduler.execute_task(&SchedulableTask::new(task, CompactionPriority::LowLevel));

        let stats = scheduler.stats();
        assert_eq!(
            stats.tasks_completed, 0,
            "stub-sized query-driven work must not count as completed rewrite"
        );
    }

    #[test]
    fn sized_query_driven_tasks_reach_runtime_executor_without_counting_completed_work() {
        let query_driven_calls = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let scheduler = CompactionIOScheduler::with_compactor(
            IOSchedulerConfig::default(),
            Arc::new(Mutex::new(
                crate::compaction::adaptive::AdaptiveCompactionScheduler::new(),
            )),
            Arc::new(RecordingExecutor {
                query_driven_calls: Arc::clone(&query_driven_calls),
            }),
        );

        let task = CompactionTask::new(
            "seg-query-sized".to_string(),
            1.0,
            CompactionStrategy::QueryDriven,
            1024,
        );
        scheduler.execute_task(&SchedulableTask::new(task, CompactionPriority::LowLevel));

        let stats = scheduler.stats();
        assert_eq!(
            query_driven_calls.load(AtomicOrdering::Relaxed),
            1,
            "sized query-driven work should reach the runtime executor"
        );
        assert_eq!(
            stats.tasks_completed, 0,
            "runtime-owned query-driven work may reach executor before physical rewrite exists"
        );
    }
    #[test]
    fn enqueue_with_maintenance_respects_resolved_rewrite_budget() {
        let handle = IOSchedulerHandle::with_compactor(
            IOSchedulerConfig::default(),
            Arc::new(Mutex::new(
                crate::compaction::adaptive::AdaptiveCompactionScheduler::new(),
            )),
            Arc::new(NoopExecutor),
        );

        let delete_heavy = crate::segment::meta::SegmentMeta {
            seg_id: "seg-delete".to_string(),
            table_id: "orders".to_string(),
            status: crate::segment::meta::SegmentStatus::Active,
            seg_type: crate::segment::meta::SegmentType::Vortex,
            columns: Vec::new(),
            min_key: Vec::new(),
            max_key: Vec::new(),
            row_count: 100,
            alive_row_count: 40,
            del_ratio: 0.6,
            size_bytes: 512 * 1024,
            created_txn: 1,
            updated_txn: 1,
            updated_at: 0,
            file_paths: Vec::new(),
            granules: Vec::new(),
            has_visibility_columns: true,
            delta_file_id: None,
            delta_row_count: 0,
            delta_l1_bytes: 0,
        };

        let admitted =
            handle.enqueue_with_maintenance(&delete_heavy, 1.0, CompactionPriority::LowLevel);
        assert!(admitted, "hard-gated delete debt should admit PDT merge");
        let next = handle
            .select_next()
            .expect("expected admitted maintenance task");
        assert_eq!(next.task.maintenance.action, RewriteAction::PdtMerge);
    }

    #[test]
    fn enqueue_with_maintenance_keeps_query_driven_evidence_demoted_until_runtime_rewrite_exists() {
        let handle = IOSchedulerHandle::with_compactor(
            IOSchedulerConfig::default(),
            Arc::new(Mutex::new(
                crate::compaction::adaptive::AdaptiveCompactionScheduler::new(),
            )),
            Arc::new(NoopExecutor),
        );

        let evidence_only = crate::segment::meta::SegmentMeta {
            seg_id: "seg-evidence-only".to_string(),
            table_id: "orders".to_string(),
            status: crate::segment::meta::SegmentStatus::Active,
            seg_type: crate::segment::meta::SegmentType::Vortex,
            columns: Vec::new(),
            min_key: Vec::new(),
            max_key: Vec::new(),
            row_count: 100,
            alive_row_count: 100,
            del_ratio: 0.0,
            size_bytes: 2 * 1024 * 1024,
            created_txn: 1,
            updated_txn: 1,
            updated_at: 0,
            file_paths: Vec::new(),
            granules: Vec::new(),
            has_visibility_columns: true,
            delta_file_id: None,
            delta_row_count: 0,
            delta_l1_bytes: 0,
        };
        handle
            .inner
            .scheduler
            .as_ref()
            .expect("adaptive scheduler")
            .lock()
            .feedback()
            .record_miss("seg-evidence-only", 1);

        let admitted =
            handle.enqueue_with_maintenance(&evidence_only, 1.0, CompactionPriority::LowLevel);
        assert!(
            !admitted,
            "prune-miss evidence should stay demoted until query-driven rewrite becomes executor-authoritative"
        );
        let stats = handle.stats();
        assert_eq!(
            stats.tasks_enqueued, 0,
            "demoted query-driven maintenance work must not count as queued authority"
        );
        assert!(
            handle.select_next().is_none(),
            "demoted query-driven evidence must not land in SILK queues"
        );
    }

    #[test]
    fn direct_enqueue_rejects_stub_sized_evidence_driven_tasks_before_queue_admission() {
        let handle = IOSchedulerHandle::with_compactor(
            IOSchedulerConfig::default(),
            Arc::new(Mutex::new(
                crate::compaction::adaptive::AdaptiveCompactionScheduler::new(),
            )),
            Arc::new(NoopExecutor),
        );

        let task = CompactionTask::new(
            "seg-direct-stub".to_string(),
            1.0,
            CompactionStrategy::QueryDriven,
            0,
        );
        handle.enqueue(task, CompactionPriority::LowLevel);

        let stats = handle.stats();
        assert_eq!(
            stats.tasks_enqueued, 0,
            "stub-sized evidence-driven work must be rejected before queue admission"
        );
        assert!(
            handle.select_next().is_none(),
            "rejected direct enqueue must not land in SILK queues"
        );
    }

    #[test]
    fn direct_enqueue_allows_sized_query_driven_tasks_to_reach_runtime_queue() {
        let handle = IOSchedulerHandle::with_compactor(
            IOSchedulerConfig::default(),
            Arc::new(Mutex::new(
                crate::compaction::adaptive::AdaptiveCompactionScheduler::new(),
            )),
            Arc::new(NoopExecutor),
        );

        let task = CompactionTask::new(
            "seg-query-admitted".to_string(),
            1.0,
            CompactionStrategy::QueryDriven,
            2 * 1024 * 1024,
        );
        handle.enqueue(task, CompactionPriority::LowLevel);

        let stats = handle.stats();
        assert_eq!(
            stats.tasks_enqueued, 1,
            "sized query-driven work should count once when admitted to the runtime queue"
        );
        let next = handle
            .select_next()
            .expect("expected admitted query-driven task");
        assert_eq!(next.task.maintenance.action, RewriteAction::QueryDriven);
        assert_eq!(next.task.size_bytes, 2 * 1024 * 1024);
    }

    #[test]
    fn maintenance_ledger_is_exposed_through_scheduler_handle() {
        let handle = IOSchedulerHandle::with_compactor(
            IOSchedulerConfig::default(),
            Arc::new(Mutex::new(
                crate::compaction::adaptive::AdaptiveCompactionScheduler::new(),
            )),
            Arc::new(NoopExecutor),
        );
        let metas = vec![crate::segment::meta::SegmentMeta {
            seg_id: "seg-ledger".to_string(),
            table_id: "orders".to_string(),
            status: crate::segment::meta::SegmentStatus::Active,
            seg_type: crate::segment::meta::SegmentType::Vortex,
            columns: Vec::new(),
            min_key: Vec::new(),
            max_key: Vec::new(),
            row_count: 100,
            alive_row_count: 40,
            del_ratio: 0.6,
            size_bytes: 512 * 1024,
            created_txn: 1,
            updated_txn: 1,
            updated_at: 0,
            file_paths: Vec::new(),
            granules: Vec::new(),
            has_visibility_columns: true,
            delta_file_id: None,
            delta_row_count: 0,
            delta_l1_bytes: 0,
        }];

        let ledger = handle.maintenance_ledger(&metas);
        assert_eq!(ledger.rows.len(), 1);
        assert_eq!(ledger.rows[0].seg_id, "seg-ledger");
        assert!(ledger.rows[0]
            .debt_flags
            .contains(crate::compaction::adaptive::DebtFlags::HIGH_DELETE_RATIO));
    }

    #[test]
    fn maintenance_governance_snapshot_is_exposed_through_scheduler_handle() {
        let handle = IOSchedulerHandle::with_compactor(
            IOSchedulerConfig::default(),
            Arc::new(Mutex::new(
                crate::compaction::adaptive::AdaptiveCompactionScheduler::new(),
            )),
            Arc::new(NoopExecutor),
        );
        let metas = vec![crate::segment::meta::SegmentMeta {
            seg_id: "seg-governance".to_string(),
            table_id: "orders".to_string(),
            status: crate::segment::meta::SegmentStatus::Active,
            seg_type: crate::segment::meta::SegmentType::Vortex,
            columns: Vec::new(),
            min_key: Vec::new(),
            max_key: Vec::new(),
            row_count: 100,
            alive_row_count: 100,
            del_ratio: 0.0,
            size_bytes: 2 * 1024 * 1024,
            created_txn: 1,
            updated_txn: 1,
            updated_at: 0,
            file_paths: Vec::new(),
            granules: Vec::new(),
            has_visibility_columns: true,
            delta_file_id: None,
            delta_row_count: 0,
            delta_l1_bytes: 0,
        }];
        handle
            .inner
            .scheduler
            .as_ref()
            .expect("adaptive scheduler")
            .lock()
            .feedback()
            .record_miss("seg-governance", 1);

        let snapshot = handle.maintenance_governance_snapshot(&metas);
        assert_eq!(snapshot.rows.len(), 1);
        assert_eq!(snapshot.rows[0].seg_id, "seg-governance");
        assert_eq!(
            snapshot.rows[0].action,
            Some(crate::compaction::scheduler::RewriteAction::QueryDriven)
        );
        assert!(!snapshot.rows[0].executor_authoritative);
    }
}

impl IOSchedulerHandle {
    pub fn new(config: IOSchedulerConfig) -> Self {
        Self {
            inner: Arc::new(CompactionIOScheduler::new(config)),
        }
    }

    /// Create a scheduler with a compaction executor and adaptive scheduler injected.
    /// The executor is used by SILK's execute_task() to perform actual compaction,
    /// and the adaptive scheduler receives compaction result callbacks.
    pub fn with_compactor(
        config: IOSchedulerConfig,
        scheduler: Arc<Mutex<crate::compaction::adaptive::AdaptiveCompactionScheduler>>,
        compactor: Arc<dyn CompactionExecutor>,
    ) -> Self {
        Self {
            inner: Arc::new(CompactionIOScheduler::with_compactor(
                config, scheduler, compactor,
            )),
        }
    }

    pub fn enqueue(&self, task: CompactionTask, priority: CompactionPriority) {
        self.inner.enqueue(task, priority);
    }

    pub fn enqueue_default(&self, task: CompactionTask) {
        self.inner.enqueue_default(task);
    }

    pub fn enqueue_with_age(&self, task: CompactionTask, seg_created: u64, now_txn: u64) {
        self.inner.enqueue_with_age(task, seg_created, now_txn);
    }

    pub fn enqueue_with_maintenance(
        &self,
        seg_meta: &crate::segment::meta::SegmentMeta,
        task_priority: f64,
        priority: CompactionPriority,
    ) -> bool {
        let Some(scheduler) = self.inner.scheduler.as_ref() else {
            return false;
        };
        let scheduler = scheduler.lock();
        let Some(action) = scheduler.maintenance_action_for_segment(seg_meta) else {
            return false;
        };
        if !action.is_executor_authoritative() {
            return false;
        }
        let budget = scheduler.rewrite_budget_for_segment(seg_meta, action);

        let strategy = match (action, budget) {
            (RewriteAction::PdtMerge, crate::compaction::scheduler::RewriteBudget::HardGate) => {
                CompactionStrategy::PdtMerge
            }
            (
                RewriteAction::SmallFileMerge,
                crate::compaction::scheduler::RewriteBudget::HeuristicPriority,
            ) => CompactionStrategy::SmallFileMerge,
            (
                RewriteAction::QueryDriven,
                crate::compaction::scheduler::RewriteBudget::EvidenceDriven,
            ) if seg_meta.size_bytes > 0 => CompactionStrategy::QueryDriven,
            _ => return false,
        };

        let task = CompactionTask::new(
            seg_meta.seg_id.clone(),
            task_priority,
            strategy,
            seg_meta.size_bytes,
        );
        self.inner.enqueue(task, priority);
        true
    }

    pub fn maintenance_ledger(
        &self,
        metas: &[crate::segment::meta::SegmentMeta],
    ) -> crate::compaction::adaptive::MaintenanceLedger {
        self.inner
            .scheduler
            .as_ref()
            .map(|scheduler| scheduler.lock().maintenance_ledger(metas))
            .unwrap_or_default()
    }

    pub fn maintenance_governance_snapshot(
        &self,
        metas: &[crate::segment::meta::SegmentMeta],
    ) -> crate::compaction::adaptive::MaintenanceGovernanceSnapshot {
        self.inner
            .scheduler
            .as_ref()
            .map(|scheduler| scheduler.lock().maintenance_governance_snapshot(metas))
            .unwrap_or_default()
    }

    pub fn select_next(&self) -> Option<SchedulableTask> {
        self.inner.select_next()
    }

    pub fn pending_count(&self) -> usize {
        self.inner.pending_count()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    pub fn can_preempt_current(&self, incoming: &SchedulableTask) -> bool {
        self.inner.can_preempt_current(incoming)
    }

    pub fn stats(&self) -> IOSchedulerStatsSnapshot {
        self.inner.stats()
    }

    pub fn current_load(&self) -> f64 {
        self.inner.current_load()
    }

    pub fn is_low_load(&self) -> bool {
        self.inner.is_low_load()
    }

    /// Start the SILK compaction background worker.
    /// Calls `start(self: Arc<Self>)` by cloning the inner Arc.
    pub fn start(&self) {
        let _ = Arc::clone(&self.inner).start();
    }

    pub fn stop(&self) {
        self.inner.stop()
    }
}
