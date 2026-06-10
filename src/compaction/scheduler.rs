//! Compaction task scheduler — priority queue driven by adaptive feedback and RangeReduce.
//!
//! Manages a BinaryHeap of compaction tasks. Priority is computed by
//! [`AdaptiveCompactionScheduler`](super::adaptive::AdaptiveCompactionScheduler) using
//! query feedback and Zone Map metrics.
//!
//! # Thread Safety
//!
//! This module is **NOT thread-safe**. The scheduler must be owned by a single
//! compaction background thread. All public methods (`push`, `pop`, `is_empty`, etc.)
//! are designed for single-threaded access. Concurrent access from multiple threads
//! without external synchronization will cause data races on the internal `tasks` heap.
//!
//! Callers must ensure that all access goes through the same dedicated thread (e.g.,
//! via channel-based task submission or an `Arc<Mutex<CompactionScheduler>>` wrapper).
//!
//! # Rewrite Action Classification
//!
//! All rewrite actions fall into one of three categories:
//!
//! | Action | Implementation Status | Physical Effect | Constraints |
//! |---------|--------------------|----------------|-------------|
//! | `PdtMerge` | **Implemented** | Filter deleted rows, rewrite as new segment | min_del_ratio threshold |
//! | `SmallFileMerge` | **Implemented** | Sequential PDT merges | Delegates to execute_pdt_merge per segment |
//! | `QueryDriven` | **Stub** | AccessTracker-based sort | `execute_query_driven` logs and skips (not wired); tasks use `size_bytes = 0` and carry no authoritative rewrite sizing |
//!
//! ## Design Notes
//!
//! - `SmallFileMerge` is not a separate physical rewrite — it loops over `execute_pdt_merge`.
//!   This is acceptable as a **proof-of-concept** for multi-way merge but should not be
//!   relied upon for production multi-segment layout optimization.
//! - Current limitation: the I/O scheduler always passes a single-element slice to
//!   `execute_small_file_merge` (`io_scheduler.rs`). There is currently no producer that
//!   batches multiple segment IDs into one `SmallFileMerge` task. Today this is effectively
//!   a single-segment PDT-merge wrapper, not a mature multi-way merge framework.
//! - This is not a bug — it is a bounded proof-of-concept implementation gap.
//!   A future multi-way merge implementation should first add a scheduler producer that
//!   groups several small segments into one task before changing the executor.
//! - `QueryDriven` requires `AccessTracker` to be plumbed through to the compactor.
//!   Today the producer signal can be inspected, but executor behavior remains observation-only:
//!   `execute_query_driven` verifies preconditions and logs without delegating rewrite authority.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::compaction::access_tracker::AccessTrackerHandle;
use ordered_float::NotNan;

/// A compaction task with priority and strategy.
#[derive(Debug, Clone)]
// Note: we implement Ord manually using NotNan<f64> for total_cmp-equivalent behavior,
// so skip the derived Eq (which would conflict with our custom PartialEq).
pub struct CompactionTask {
    /// Segment to compact.
    pub seg_id: String,
    /// Priority score (higher = more urgent). NotNaN guarantees Ord is total.
    pub priority: NotNan<f64>,
    /// Compaction strategy.
    pub strategy: CompactionStrategy,
    /// Shared maintenance task identity.
    pub maintenance: MaintenanceTask,
    /// When this task was created (milliseconds).
    pub created_at: u64,
    /// Estimated on-disk size of this segment for I/O rate limiting.
    pub size_bytes: u64,
}

/// Compaction strategy — determines how a task is executed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompactionStrategy {
    /// Standard PDT merge — filter deleted rows, rewrite as new segment.
    PdtMerge,
    /// Query-driven compaction (RangeReduce) — write scan results back as a sorted run.
    QueryDriven,
    /// Merge several small segments into a larger one.
    SmallFileMerge,
}

/// Shared maintenance budget classification used by rewrite and flush coordination.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RewriteBudget {
    HardGate,
    HeuristicPriority,
    EvidenceDriven,
}

/// Shared maintenance action language across rewrite/flush/metadata coordination.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RewriteAction {
    PdtMerge,
    SmallFileMerge,
    QueryDriven,
    FlushL1ToL2,
    CompactL2ToL3,
    GuardMerge,
    MetadataEvidenceRefresh,
    CheckpointPrivilege,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActionAuthority {
    Executable,
    DemotedScaffold,
    TruthPlanePrivileged,
}

impl RewriteAction {
    pub fn budget(self) -> RewriteBudget {
        match self {
            RewriteAction::PdtMerge => RewriteBudget::HardGate,
            RewriteAction::SmallFileMerge => RewriteBudget::HeuristicPriority,
            RewriteAction::QueryDriven => RewriteBudget::EvidenceDriven,
            RewriteAction::FlushL1ToL2 => RewriteBudget::HardGate,
            RewriteAction::CompactL2ToL3 => RewriteBudget::HeuristicPriority,
            RewriteAction::GuardMerge => RewriteBudget::HeuristicPriority,
            RewriteAction::MetadataEvidenceRefresh => RewriteBudget::EvidenceDriven,
            RewriteAction::CheckpointPrivilege => RewriteBudget::HardGate,
        }
    }

    pub fn authority(self) -> ActionAuthority {
        match self {
            RewriteAction::PdtMerge | RewriteAction::SmallFileMerge => ActionAuthority::Executable,
            RewriteAction::QueryDriven
            | RewriteAction::FlushL1ToL2
            | RewriteAction::CompactL2ToL3
            | RewriteAction::GuardMerge
            | RewriteAction::MetadataEvidenceRefresh => ActionAuthority::DemotedScaffold,
            RewriteAction::CheckpointPrivilege => ActionAuthority::TruthPlanePrivileged,
        }
    }

    pub fn is_executor_authoritative(self) -> bool {
        matches!(self.authority(), ActionAuthority::Executable)
    }

    pub fn truth_plane_privileged(self) -> bool {
        matches!(self, RewriteAction::CheckpointPrivilege)
    }
}

impl From<CompactionStrategy> for RewriteAction {
    fn from(value: CompactionStrategy) -> Self {
        match value {
            CompactionStrategy::PdtMerge => RewriteAction::PdtMerge,
            CompactionStrategy::QueryDriven => RewriteAction::QueryDriven,
            CompactionStrategy::SmallFileMerge => RewriteAction::SmallFileMerge,
        }
    }
}

/// Unified maintenance task surface. Checkpoint remains representable but privileged.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MaintenanceTask {
    pub action: RewriteAction,
    pub seg_id: Option<String>,
    pub size_bytes: u64,
}

impl MaintenanceTask {
    pub fn checkpoint_privilege() -> Self {
        Self {
            action: RewriteAction::CheckpointPrivilege,
            seg_id: None,
            size_bytes: 0,
        }
    }

    pub fn is_truth_plane_privileged(&self) -> bool {
        self.action.truth_plane_privileged()
    }
}

impl CompactionTask {
    /// Create a new compaction task.
    pub fn new(
        seg_id: String,
        priority: f64,
        strategy: CompactionStrategy,
        size_bytes: u64,
    ) -> Self {
        let action = RewriteAction::from(strategy);
        Self {
            maintenance: MaintenanceTask {
                action,
                seg_id: Some(seg_id.clone()),
                size_bytes,
            },
            seg_id,
            priority: NotNan::new(priority).expect("priority must not be NaN"),
            strategy,
            created_at: crate::codec::current_timestamp_millis(),
            size_bytes,
        }
    }

    /// Create a RangeReduce (query-driven) compaction task.
    pub fn query_driven(seg_id: String, priority: f64, size_bytes: u64) -> Self {
        debug_assert!(
            size_bytes == 0,
            "QueryDriven tasks remain stub-sized until a runtime consumer owns authoritative rewrite sizing"
        );
        Self::new(
            seg_id,
            priority,
            CompactionStrategy::QueryDriven,
            size_bytes,
        )
    }

    /// Estimated I/O bytes for token bucket scheduling.
    /// Used by the SILK I/O scheduler to rate-limit compaction.
    pub fn estimated_io_bytes(&self) -> u64 {
        self.size_bytes.min(64 * 1024 * 1024)
    }
}

/// Task ordering for BinaryHeap: highest priority first.
/// NotNan<f64> implements Ord via total_cmp, so delegation is trivial.
impl Ord for CompactionTask {
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority.cmp(&other.priority)
    }
}

impl PartialOrd for CompactionTask {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for CompactionTask {
    fn eq(&self, other: &Self) -> bool {
        // Compare by identity: same seg_id + created_at timestamp.
        // Priority is excluded because it can be NaN and f64 doesn't implement Ord/Eq.
        self.seg_id == other.seg_id && self.created_at == other.created_at
    }
}

impl Eq for CompactionTask {} // ---------------------------------------------------------------------------
                              // CompactionScheduler
                              // ---------------------------------------------------------------------------

/// Priority queue of compaction tasks driven by adaptive feedback.
///
/// Uses a [`BinaryHeap`] ordered by priority. The heap holds tasks across
/// all strategy types; the scheduler is responsible for selecting which task
/// to execute next based on system state.
    pub struct CompactionScheduler {
    heap: BinaryHeap<CompactionTask>,
    /// Access tracker for RangeReduce decisions.
    access_tracker: Option<AccessTrackerHandle>,
}

impl Default for CompactionScheduler {
    #[allow(dead_code)]
    fn default() -> Self {
        Self {
            heap: BinaryHeap::new(),
            access_tracker: None,
        }
    }
}

impl CompactionScheduler {
    /// Create a new empty scheduler.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a scheduler with an access tracker for RangeReduce support.
    pub fn with_access_tracker(access_tracker: AccessTrackerHandle) -> Self {
        Self {
            heap: BinaryHeap::new(),
            access_tracker: Some(access_tracker),
        }
    }

    /// Enqueue a new compaction task.
    /// Stub-sized tasks (size_bytes == 0) are silently dropped to avoid O(n) dequeue overhead.
    pub fn enqueue(&mut self, task: CompactionTask) {
        if task.size_bytes == 0 {
            return;
        }
        self.heap.push(task);
    }

    /// Dequeue the highest-priority task, if any.
    pub fn dequeue(&mut self) -> Option<CompactionTask> {
        while let Some(task) = self.heap.pop() {
            if task.maintenance.is_truth_plane_privileged()
                || task.maintenance.action.budget() != RewriteBudget::EvidenceDriven
                || task.size_bytes > 0
            {
                return Some(task);
            }
        }
        None
    }

    /// Peek at the highest-priority task without removing it.
    pub fn peek(&self) -> Option<&CompactionTask> {
        self.heap.peek()
    }

    /// Returns the number of pending tasks.
    pub fn len(&self) -> usize {
        self.heap.len()
    }

    /// Returns true if there are no pending tasks.
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    /// Clear all pending tasks.
    pub fn clear(&mut self) {
        self.heap.clear();
    }

    // -------------------------------------------------------------------------
    // RangeReduce support
    // -------------------------------------------------------------------------

    /// Select the top-N segments that are candidates for RangeReduce compaction,
    /// based on access tracking data.
    ///
    /// Returns tasks with [`CompactionStrategy::QueryDriven`].
    pub fn select_targets(&self, max_count: usize) -> Vec<CompactionTask> {
        let Some(ref tracker) = self.access_tracker else {
            return Vec::new();
        };

        let candidates = tracker.select_targets(max_count);
        candidates
            .into_iter()
            .map(|(seg_id, score)| {
                // QueryDriven remains a stub producer surface until a real runtime consumer
                // can supply authoritative segment sizing, admission control, and execution semantics.
                // `size_bytes = 0` is intentional blocker evidence: the producer does not yet own
                // truthful rewrite sizing, so these tasks must not be treated as active authority.
                CompactionTask::query_driven(seg_id, score, 0)
            })
            .collect()
    }

    /// Record a scan access on a segment (triggers RangeReduce tracking).
    ///
    /// Call this from scan paths after loading segment batches.
    pub fn record_access(&self, seg_id: &str, start_row: u32, row_count: u32) {
        if let Some(ref tracker) = self.access_tracker {
            tracker.mark_access(seg_id, start_row, row_count);
        }
    }

    /// Record a scan key range for overlap detection.
    pub fn record_scan_range(&self, seg_id: &str, start_key: Vec<u8>, end_key: Vec<u8>) {
        if let Some(ref tracker) = self.access_tracker {
            tracker.set_scan_range(seg_id, start_key, end_key);
        }
    }

    /// Returns the hot score for a segment (0.0–1.0).
    pub fn hot_score(&self, seg_id: &str) -> f64 {
        self.access_tracker
            .as_ref()
            .map(|t| t.hot_score(seg_id))
            .unwrap_or(0.0)
    }

    /// Returns the overlap score for a segment (0.0–1.0).
    pub fn overlap_score(&self, seg_id: &str) -> f64 {
        self.access_tracker
            .as_ref()
            .map(|t| t.overlap_score(seg_id))
            .unwrap_or(0.0)
    }

    /// Check if a segment should be RangeReduce-compacted.
    pub fn should_range_reduce(&self, seg_id: &str) -> bool {
        self.access_tracker
            .as_ref()
            .map(|t| t.should_range_reduce(seg_id))
            .unwrap_or(false)
    }
}

#[cfg(test)]
mod query_driven_stub_tests {
    use super::*;

    #[test]
    fn query_driven_tasks_carry_stub_sizing_until_runtime_consumer_exists() {
        let tracker = AccessTrackerHandle::with_config(crate::compaction::access_tracker::Config {
            hot_score_threshold: 0.0,
            overlap_threshold: 0.0,
            ..Default::default()
        });
        let scheduler = CompactionScheduler::with_access_tracker(tracker.clone());
        tracker.mark_access("seg-hot", 0, 128);
        tracker.set_scan_range("seg-hot", vec![0], vec![255]);
        tracker.set_scan_range("seg-hot", vec![32], vec![200]);

        let tasks = scheduler.select_targets(1);
        let task = tasks
            .iter()
            .find(|task| task.strategy == CompactionStrategy::QueryDriven)
            .expect("expected query-driven stub task");

        assert_eq!(task.size_bytes, 0);
        assert_eq!(task.maintenance.action, RewriteAction::QueryDriven);
        assert_eq!(
            task.maintenance.action.budget(),
            RewriteBudget::EvidenceDriven
        );
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(
        expected = "QueryDriven tasks remain stub-sized until a runtime consumer owns authoritative rewrite sizing"
    )]
    fn query_driven_constructor_rejects_sized_tasks_in_debug_builds() {
        let _ = CompactionTask::query_driven("seg-sized".to_string(), 10.0, 2048);
    }

    #[test]
    fn dequeue_allows_evidence_driven_tasks_with_runtime_sizing() {
        let mut scheduler = CompactionScheduler::new();
        scheduler.enqueue(CompactionTask::new(
            "seg-sized".to_string(),
            10.0,
            CompactionStrategy::QueryDriven,
            2048,
        ));

        let task = scheduler
            .dequeue()
            .expect("expected sized evidence-driven task");
        assert_eq!(task.seg_id, "seg-sized");
        assert_eq!(task.maintenance.action, RewriteAction::QueryDriven);
        assert_eq!(
            task.maintenance.action.budget(),
            RewriteBudget::EvidenceDriven
        );
        assert_eq!(task.size_bytes, 2048);
    }

    #[test]
    fn dequeue_skips_evidence_driven_stub_tasks_without_runtime_sizing() {
        let mut scheduler = CompactionScheduler::new();
        scheduler.enqueue(CompactionTask::query_driven(
            "seg-stub".to_string(),
            10.0,
            0,
        ));
        scheduler.enqueue(CompactionTask::new(
            "seg-real".to_string(),
            5.0,
            CompactionStrategy::PdtMerge,
            1024,
        ));

        let task = scheduler.dequeue().expect("expected runnable task");
        assert_eq!(task.seg_id, "seg-real");
        assert_eq!(task.maintenance.action, RewriteAction::PdtMerge);
        assert!(scheduler.dequeue().is_none());
    }
}
