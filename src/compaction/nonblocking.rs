//! Non-Blocking Compaction implementation
//!
//! Core idea: writes always append to new granules, compaction runs in background threads.
//! Compaction background thread reads old granules -> merges -> writes new granules.
//! Reads check both old and new paths simultaneously and merge results.
//! Writes are not blocked by compaction.

use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tracing::debug;

use crate::compaction::io_scheduler::{CompactionExecutor, CompactionPriority, IOSchedulerHandle};
use crate::compaction::pdt_merge::MergeStats;
use crate::compaction::pdt_merge::{compact_segment, PdtMergeConfig};
use crate::db::RockDuck;
use crate::error::Result;
use crate::error::RockDuckError;
use crate::metadata;
use crate::metadata::kv_engine::{KVOp, CF_PK_IDX};
use crate::metadata::seg_alias;
use crate::metadata::GranuleId;
use crate::query::time_travel_impl::build_segment_version_index;
use crate::write::durability_wal::{OpPayload, OpType};

/// Compaction status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CompactionStatus {
    /// Idle
    #[default]
    Idle,
    /// Running
    Running,
    /// Paused
    Paused,
    /// Error
    Error,
}

/// Single segment merge state
#[derive(Debug, Clone)]
pub struct SegmentMergeState {
    pub old_seg_id: String,
    pub new_seg_id: String,
    pub started_at: u64,
    pub progress: f32,
    pub status: CompactionStatus,
}

impl SegmentMergeState {
    pub fn new(old_seg_id: String, new_seg_id: String) -> Self {
        Self {
            old_seg_id,
            new_seg_id,
            started_at: crate::codec::current_timestamp_secs(),
            progress: 0.0,
            status: CompactionStatus::Running,
        }
    }
}

/// Non-Blocking Compactor configuration
#[derive(Debug, Clone)]
pub struct NonBlockingConfig {
    /// Number of compaction threads
    pub num_threads: usize,
    /// Compaction trigger threshold (delete ratio)
    pub del_ratio_threshold: f64,
    /// Compaction check interval
    pub check_interval: Duration,
    /// Maximum concurrent merges
    pub max_concurrent_merges: usize,
}

impl Default for NonBlockingConfig {
    fn default() -> Self {
        Self {
            num_threads: 2,
            del_ratio_threshold: 0.1,
            check_interval: Duration::from_secs(60),
            max_concurrent_merges: 4,
        }
    }
}

/// Non-Blocking Compactor — SILK-coordinated background compaction.
pub struct NonBlockingCompactor {
    docdb: Arc<RockDuck>,
    config: NonBlockingConfig,
    /// Segments being merged
    merging: RwLock<HashMap<String, SegmentMergeState>>,
    /// SILK I/O scheduler for coordination with foreground I/O.
    io_scheduler: Option<IOSchedulerHandle>,
    /// Query-driven access tracker handle.
    access_tracker: Option<crate::compaction::access_tracker::AccessTrackerHandle>,
    /// Callback invoked after each successful compaction.
    /// Receives the rewritten segment ID and a typed debt-resolution outcome.
    /// This remains bounded maintenance feedback, not direct query-routing authority.
    #[allow(clippy::type_complexity)]
    compaction_callback: RwLock<
        Option<
            Box<dyn Fn(String, crate::compaction::adaptive::DebtResolutionOutcome) + Send + Sync>,
        >,
    >,
}

/// Compaction result event
#[derive(Debug, Clone)]
pub struct CompactionEvent {
    pub old_seg_id: String,
    pub new_seg_id: String,
    pub success: bool,
    pub stats: Option<MergeStats>,
}

impl NonBlockingCompactor {
    /// Create a new Non-Blocking Compactor without I/O scheduler.
    pub fn new(docdb: Arc<RockDuck>, config: NonBlockingConfig) -> Self {
        Self {
            docdb,
            config,
            merging: RwLock::new(HashMap::new()),
            io_scheduler: None,
            access_tracker: None,
            compaction_callback: RwLock::new(None),
        }
    }

    /// Create a new Non-Blocking Compactor with SILK I/O scheduler.
    pub fn with_scheduler(
        docdb: Arc<RockDuck>,
        config: NonBlockingConfig,
        io_scheduler: IOSchedulerHandle,
    ) -> Self {
        Self {
            docdb,
            config,
            merging: RwLock::new(HashMap::new()),
            io_scheduler: Some(io_scheduler),
            access_tracker: None,
            compaction_callback: RwLock::new(None),
        }
    }

    /// Create a new Non-Blocking Compactor with an AccessTracker handle.
    pub fn with_access_tracker(
        docdb: Arc<RockDuck>,
        config: NonBlockingConfig,
        access_tracker: crate::compaction::access_tracker::AccessTrackerHandle,
    ) -> Self {
        Self {
            docdb,
            config,
            merging: RwLock::new(HashMap::new()),
            io_scheduler: None,
            access_tracker: Some(access_tracker),
            compaction_callback: RwLock::new(None),
        }
    }

    /// Create a new Non-Blocking Compactor with SILK I/O scheduler and AccessTracker handle.
    pub fn with_scheduler_and_access_tracker(
        docdb: Arc<RockDuck>,
        config: NonBlockingConfig,
        io_scheduler: IOSchedulerHandle,
        access_tracker: crate::compaction::access_tracker::AccessTrackerHandle,
    ) -> Self {
        Self {
            docdb,
            config,
            merging: RwLock::new(HashMap::new()),
            io_scheduler: Some(io_scheduler),
            access_tracker: Some(access_tracker),
            compaction_callback: RwLock::new(None),
        }
    }

    /// Set a callback invoked after each successful compaction.
    ///
    /// Used by the adaptive scheduler to wire typed maintenance feedback.
    /// Outcomes explicitly distinguish proxy bootstrap signals from measured delete-resolution evidence.
    pub fn set_compaction_callback(
        &self,
        cb: Option<
            Box<dyn Fn(String, crate::compaction::adaptive::DebtResolutionOutcome) + Send + Sync>,
        >,
    ) {
        *self.compaction_callback.write() = cb;
    }

    /// Returns true if a segment is currently being compacted.
    ///
    /// SAFETY: Concurrent queries during compaction are safe by design. Nonblocking
    /// compaction never deletes old segments — it appends new segments and leaves old ones
    /// untouched. The old segment remains visible to queries throughout compaction (reads
    /// pre-compaction data, which is correct for Snapshot Isolation). Only `phase.rs`
    /// compactor handles segment deletion (GC). This means queries never see partially-
    /// compacted data or miss deleted rows due to compaction timing.
    pub fn is_compacting(&self, seg_id: &str) -> bool {
        self.merging.read().contains_key(seg_id)
    }

    /// Trigger compaction (async, non-blocking for writes)
    pub fn trigger(&self, seg_id: &str) -> Result<()> {
        // Check if already merging
        {
            let merging = self.merging.read();
            if merging.contains_key(seg_id) {
                debug!("Segment {} already merging", seg_id);
                return Ok(());
            }
        }

        // Check if merge threshold is met
        if let Some(meta) = metadata::get_segment_meta(&self.docdb.kv, seg_id)? {
            if meta.del_ratio < self.config.del_ratio_threshold {
                debug!(
                    "Segment {} del_ratio {} below threshold",
                    seg_id, meta.del_ratio
                );
                return Ok(());
            }
        }

        // Record merge state
        let new_seg_id = generate_seg_id();
        {
            let mut merging = self.merging.write();
            merging.insert(
                seg_id.to_string(),
                SegmentMergeState::new(seg_id.to_string(), new_seg_id.clone()),
            );
        }

        // SILK I/O scheduler: route through the shared maintenance admission gate
        // before any task becomes executable background work.
        if let Some(ref scheduler) = self.io_scheduler {
            let seg_meta = metadata::get_segment_meta(&self.docdb.kv, seg_id)
                .ok()
                .flatten();
            let priority = match (seg_meta.as_ref(), self.docdb.next_txn_id()) {
                (Some(m), Ok(next_txn)) => {
                    CompactionPriority::from_segment_age(m.created_txn, next_txn)
                }
                _ => CompactionPriority::LowLevel,
            };

            if let Some(meta) = seg_meta.as_ref() {
                if !scheduler.enqueue_with_maintenance(meta, 1.0, priority) {
                    self.merging.write().remove(seg_id);
                    debug!(
                        seg_id,
                        "Compaction task rejected by typed maintenance action mapping"
                    );
                    return Err(RockDuckError::Compaction(format!(
                        "segment {} rejected by typed maintenance admission",
                        seg_id
                    )));
                }
            } else {
                self.merging.write().remove(seg_id);
                debug!(seg_id, "Segment metadata unavailable, compaction rejected before typed maintenance admission");
                return Err(RockDuckError::Compaction(format!(
                    "segment {} metadata unavailable before typed maintenance admission",
                    seg_id
                )));
            }
        } else {
            self.merging.write().remove(seg_id);
            debug!(
                seg_id,
                "Compaction rejected because no I/O scheduler is configured"
            );
            return Err(RockDuckError::Compaction(format!(
                "segment {} cannot be dispatched: no I/O scheduler is configured",
                seg_id
            )));
        }

        Ok(())
    }

    /// Batch trigger compaction
    pub fn trigger_batch(&self, seg_ids: &[String]) -> Result<()> {
        for seg_id in seg_ids {
            self.trigger(seg_id)?;
        }
        Ok(())
    }

    /// Check if a segment is being merged
    pub fn is_merging(&self, seg_id: &str) -> bool {
        let merging = self.merging.read();
        merging.contains_key(seg_id)
    }

    /// Get merge state
    pub fn get_state(&self, seg_id: &str) -> Option<SegmentMergeState> {
        let merging = self.merging.read();
        merging.get(seg_id).cloned()
    }

    /// List all segments being merged
    pub fn list_merging(&self) -> Vec<String> {
        let merging = self.merging.read();
        merging.keys().cloned().collect()
    }
}

/// Generate a new segment ID
pub fn generate_seg_id() -> String {
    uuid::Uuid::now_v7().to_string()
}

/// Run actual compaction for a single segment.
///
/// WAL-protected compaction sequence:
/// 1. Compact segment data → new segment files (no metadata updates yet)
/// 2. Serialize pk_entries for WAL payload
/// 3. Write WAL Compaction record (durability boundary) — must succeed before KV changes
/// 4. Update pk_lookup entries (write_batch)
/// 5. Write alias
/// 6. Update segment metadata
/// 7. Build version index
///
/// If WAL write fails at step 3, the compaction is aborted (new segment orphaned, no harm).
/// If crash occurs after step 3 but before step 7, WAL recovery replays the Compaction record
/// to finish or rollback the in-flight compaction.
pub fn run_compaction(
    docdb: &Arc<RockDuck>,
    seg_id: &str,
    config: &NonBlockingConfig,
) -> Result<(crate::segment::meta::SegmentMeta, MergeStats)> {
    let meta = metadata::get_segment_meta(&docdb.kv, seg_id)?
        .ok_or_else(|| RockDuckError::Compaction(format!("Segment {} not found in KV", seg_id)))?;

    if meta.row_count == 0 {
        return Ok((meta, MergeStats::default()));
    }

    let dead_rows = meta.row_count.saturating_sub(meta.alive_row_count);
    let del_ratio = dead_rows as f64 / meta.row_count as f64;
    if del_ratio < config.del_ratio_threshold {
        debug!(
            "Segment {} del_ratio {} below threshold {}, skipping",
            seg_id, del_ratio, config.del_ratio_threshold
        );
        return Ok((meta, MergeStats::default()));
    }

    let pdt_config = PdtMergeConfig {
        min_del_ratio: config.del_ratio_threshold,
        target_granule_rows: 1024 * 1024,
        parallel_io: true,
        parallelism: config.num_threads,
    };

    let (new_meta, stats) = compact_segment(&docdb.data_dir, seg_id, &meta, &pdt_config, None)?;

    // Collect pk_entries for WAL payload before any KV mutations.
    // WAL payload stores: Vec<(pk, granule_id, row_offset)> — seg_id is at WAL record level.
    let old_entries = metadata::pk_skiplist::list_skiplist_entries(&docdb.kv, seg_id)?;
    let pk_entries: Vec<(Vec<u8>, u32, u32)> = old_entries
        .into_iter()
        .map(|entry| (entry.pk, entry.granule_id.get(), entry.row_offset))
        .collect();

    // WAL durability boundary — must succeed before any KV mutation.
    // If this fails, compaction is aborted (new segment stays orphaned, harmless).
    let pk_entries_bytes = crate::codec::encode(&pk_entries)?;
    let wal_payload = OpPayload::Compaction {
        old_seg_id: seg_id.to_string(),
        new_seg_id: new_meta.seg_id.clone(),
        pk_entries: pk_entries_bytes,
        commit: true,
    };
    docdb
        .wal
        .append_durable(OpType::Compaction, 0, &wal_payload)?;

    // WAL succeeded — now apply KV mutations. These are protected by the WAL record.
    // Build pk_transition_ops from the collected entries.
    let mut pk_transition_ops = Vec::with_capacity(pk_entries.len() * 2);
    for (pk, granule_id, row_offset) in &pk_entries {
        let new_granule_id = GranuleId::new(*granule_id);
        let new_value =
            crate::codec::encode(&(new_meta.seg_id.clone(), new_granule_id.get(), *row_offset))?;
        let lookup_key = metadata::pk_skiplist::pk_lookup_key(pk);
        let old_idx_key = metadata::pk_skiplist::pk_index_key(seg_id, new_granule_id, pk);
        pk_transition_ops.push(KVOp::Put {
            key: lookup_key,
            value: new_value,
        });
        pk_transition_ops.push(KVOp::Delete { key: old_idx_key });
    }
    if !pk_transition_ops.is_empty() {
        docdb.kv.write_batch(CF_PK_IDX, &pk_transition_ops)?;
    }

    seg_alias::write_alias(&docdb.kv, seg_id, &new_meta.seg_id)?;
    metadata::put_segment_meta(&docdb.kv, &new_meta)?;
    build_segment_version_index(
        &docdb.kv,
        &new_meta.table_id,
        &docdb.data_dir,
        &new_meta.seg_id,
        &new_meta
            .columns
            .iter()
            .map(|c| c.name.clone())
            .collect::<Vec<_>>(),
    )?;
    docdb.kv.flush()?;

    // Invalidate caches after all mutations complete.
    docdb.seg_meta_cache.write().invalidate(seg_id);
    docdb.seg_meta_cache.write().invalidate(&new_meta.seg_id);

    tracing::info!(
        "Compaction completed: {} -> {} (rows={}, bytes={})",
        seg_id,
        new_meta.seg_id,
        stats.rows_written,
        stats.bytes_written
    );

    Ok((new_meta, stats))
}

// =============================================================================
// Drop — safety net to clear the merging HashMap if the compactor is dropped
// while compaction is still in flight (e.g., shutdown, panic propagation).
// This does NOT replace the per-call cleanup in execute_pdt_merge; it only
// catches cases where the compactor itself is destroyed while tasks are pending.
// =============================================================================

impl Drop for NonBlockingCompactor {
    fn drop(&mut self) {
        let seg_ids: Vec<_> = self.merging.write().keys().cloned().collect();
        if !seg_ids.is_empty() {
            tracing::warn!(
                "NonBlockingCompactor dropped with {} unmerged segment(s): {:?}",
                seg_ids.len(),
                seg_ids
            );
        }
        // Safety net: clear all in-flight merging state.
        // On normal shutdown this is harmless (scheduler has already cleaned up).
        // On panic/cancellation this prevents the HashMap from leaking.
        self.merging.write().clear();
    }
}

// =============================================================================
// CompactionExecutor trait — enables injection into SILK I/O scheduler
// =============================================================================

#[cfg(test)]
#[allow(clippy::items_after_test_module)]
mod admission_gate_tests {
    use super::*;
    use crate::compaction::io_scheduler::{IOSchedulerConfig, IOSchedulerHandle};
    use crate::segment::meta::{SegmentMeta, SegmentStatus, SegmentType};

    fn make_meta(seg_id: &str, del_ratio: f64, size_bytes: u64, created_txn: u64) -> SegmentMeta {
        SegmentMeta {
            seg_id: seg_id.to_string(),
            table_id: "orders".to_string(),
            status: SegmentStatus::Active,
            seg_type: SegmentType::Vortex,
            columns: Vec::new(),
            min_key: Vec::new(),
            max_key: Vec::new(),
            row_count: 100,
            alive_row_count: ((1.0 - del_ratio) * 100.0).round() as u64,
            del_ratio,
            size_bytes,
            created_txn,
            updated_txn: created_txn,
            updated_at: 0,
            file_paths: Vec::new(),
            granules: Vec::new(),
            has_visibility_columns: true,
            delta_file_id: None,
            delta_row_count: 0,
            delta_l1_bytes: 0,
        }
    }

    #[test]
    fn trigger_routes_pdt_merge_through_admission_gate_before_enqueue() {
        let dir = tempfile::tempdir().expect("tempdir");
        let docdb = Arc::new(crate::db::RockDuck::open(dir.path()).expect("open db"));
        let seg_id = "seg-admission";
        let meta = make_meta(seg_id, 0.5, 4096, 1);
        metadata::put_segment_meta(&docdb.kv, &meta).expect("put meta");

        let scheduler = Arc::new(parking_lot::Mutex::new(
            crate::compaction::adaptive::AdaptiveCompactionScheduler::new(),
        ));
        let io = IOSchedulerHandle::with_compactor(
            IOSchedulerConfig::default(),
            scheduler,
            Arc::new(NonBlockingCompactor::new(
                docdb.clone(),
                NonBlockingConfig::default(),
            )),
        );
        let compactor =
            NonBlockingCompactor::with_scheduler(docdb, NonBlockingConfig::default(), io.clone());

        compactor.trigger(seg_id).expect("trigger should succeed");

        let stats = io.stats();
        assert_eq!(
            stats.tasks_enqueued, 1,
            "admitted PDT merge should be enqueued once"
        );
        assert_eq!(
            stats.pending_tasks, 1,
            "admitted PDT merge should remain pending in SILK queue"
        );
        assert!(
            compactor.is_compacting(seg_id),
            "segment should remain marked as in-flight after enqueue"
        );
    }

    #[test]
    fn trigger_keeps_query_driven_evidence_demoted_until_runtime_rewrite_exists() {
        let dir = tempfile::tempdir().expect("tempdir");
        let docdb = Arc::new(crate::db::RockDuck::open(dir.path()).expect("open db"));
        let seg_id = "seg-query-driven";
        let meta = make_meta(seg_id, 0.2, 2 * 1024 * 1024, 1);
        metadata::put_segment_meta(&docdb.kv, &meta).expect("put meta");

        let scheduler_arc = Arc::new(parking_lot::Mutex::new(
            crate::compaction::adaptive::AdaptiveCompactionScheduler::new(),
        ));
        scheduler_arc.lock().feedback().record_miss(seg_id, 1);

        let io = IOSchedulerHandle::with_compactor(
            IOSchedulerConfig::default(),
            scheduler_arc,
            Arc::new(NonBlockingCompactor::new(
                docdb.clone(),
                NonBlockingConfig::default(),
            )),
        );
        let compactor =
            NonBlockingCompactor::with_scheduler(docdb, NonBlockingConfig::default(), io.clone());

        let err = compactor.trigger(seg_id).expect_err(
            "trigger should reject demoted query-driven evidence until runtime rewrite exists",
        );
        let msg = format!("{err}");
        assert!(msg.contains("typed maintenance admission"));

        let stats = io.stats();
        assert_eq!(
            stats.tasks_enqueued, 0,
            "demoted query-driven segment must not enqueue runtime authority"
        );
        assert_eq!(
            stats.pending_tasks, 0,
            "demoted query-driven work must not remain pending in SILK queue"
        );
        assert!(
            !compactor.is_compacting(seg_id),
            "demoted query-driven rejection must not leave false in-flight state"
        );
        assert!(
            io.select_next().is_none(),
            "demoted query-driven evidence must not enqueue a task"
        );
    }
    #[test]
    fn trigger_returns_error_without_scheduler_and_does_not_leave_false_inflight_state() {
        let dir = tempfile::tempdir().expect("tempdir");
        let docdb = Arc::new(crate::db::RockDuck::open(dir.path()).expect("open db"));
        let seg_id = "seg-no-scheduler";
        let meta = make_meta(seg_id, 0.5, 4096, 1);
        metadata::put_segment_meta(&docdb.kv, &meta).expect("put meta");

        let compactor = NonBlockingCompactor::new(docdb, NonBlockingConfig::default());

        let err = compactor
            .trigger(seg_id)
            .expect_err("trigger should fail loudly when no scheduler is configured");
        let msg = format!("{err}");
        assert!(msg.contains("no I/O scheduler is configured"));
        assert!(
            !compactor.is_compacting(seg_id),
            "no-scheduler rejection must not leave false in-flight state"
        );
        assert!(
            !compactor.is_merging(seg_id),
            "no-scheduler rejection must not leave false merge tracking"
        );
    }

    #[test]
    fn typed_action_mapping_selects_small_file_merge_for_small_segments() {
        let dir = tempfile::tempdir().expect("tempdir");
        let scheduler_arc = Arc::new(parking_lot::Mutex::new(
            crate::compaction::adaptive::AdaptiveCompactionScheduler::new(),
        ));
        let io = IOSchedulerHandle::with_compactor(
            IOSchedulerConfig::default(),
            scheduler_arc,
            Arc::new(NonBlockingCompactor::new(
                Arc::new(crate::db::RockDuck::open(dir.path()).expect("open db")),
                NonBlockingConfig::default(),
            )),
        );
        let meta = make_meta("seg-small", 0.2, 512 * 1024, 1);

        let admitted = io.enqueue_with_maintenance(&meta, 1.0, CompactionPriority::LowLevel);
        assert!(
            admitted,
            "small file segment should produce executable maintenance work"
        );

        let task = io.select_next().expect("expected queued task");
        assert_eq!(
            task.task.maintenance.action,
            crate::compaction::scheduler::RewriteAction::SmallFileMerge
        );
    }

    #[test]
    fn zero_debt_segments_are_rejected_by_typed_maintenance_admission() {
        let dir = tempfile::tempdir().expect("tempdir");
        let scheduler_arc = Arc::new(parking_lot::Mutex::new(
            crate::compaction::adaptive::AdaptiveCompactionScheduler::new(),
        ));
        let io = IOSchedulerHandle::with_compactor(
            IOSchedulerConfig::default(),
            scheduler_arc,
            Arc::new(NonBlockingCompactor::new(
                Arc::new(crate::db::RockDuck::open(dir.path()).expect("open db")),
                NonBlockingConfig::default(),
            )),
        );
        let meta = make_meta("seg-clean", 0.01, 32 * 1024 * 1024, 1);

        let admitted = io.enqueue_with_maintenance(&meta, 1.0, CompactionPriority::LowLevel);
        assert!(!admitted, "zero-debt segment must not default to PDT merge");
        assert!(
            io.select_next().is_none(),
            "rejected zero-debt segment must not enqueue fallback work"
        );
    }
}

impl CompactionExecutor for NonBlockingCompactor {
    fn execute_pdt_merge(&self, seg_id: &str) -> crate::error::Result<MergeStats> {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let config = NonBlockingConfig {
                num_threads: self.config.num_threads,
                del_ratio_threshold: self.config.del_ratio_threshold,
                check_interval: self.config.check_interval,
                max_concurrent_merges: self.config.max_concurrent_merges,
            };
            run_compaction(&self.docdb, seg_id, &config)
        }));

        match result {
            Ok(Ok((new_meta, stats))) => {
                self.merging.write().remove(seg_id);
                // Phase 4 safety boundary: this callback is intentionally bounded proxy-only.
                // It reports that delete debt was eliminated by the rewrite, but it must not be
                // promoted to measured query-quality authority until the adaptive seam becomes
                // segment-aware and can compare same-segment evidence instead of cross-segment first hits.
                if let Some(ref cb) = *self.compaction_callback.read() {
                    let outcome =
                        crate::compaction::adaptive::DebtResolutionOutcome::from_merge_stats(
                            seg_id, &stats,
                        );
                    cb(new_meta.seg_id.clone(), outcome);
                }
                Ok(stats)
            }
            Ok(Err(e)) => {
                self.merging.write().remove(seg_id);
                tracing::warn!("Compaction failed for seg {}: {}", seg_id, e);
                Err(e)
            }
            Err(panic_info) => {
                self.merging.write().remove(seg_id);
                let msg = if let Some(s) = panic_info.downcast_ref::<&str>() {
                    format!("Compaction panicked for seg {}: {}", seg_id, s)
                } else if let Some(s) = panic_info.downcast_ref::<String>() {
                    format!("Compaction panicked for seg {}: {}", seg_id, s)
                } else {
                    format!("Compaction panicked for seg {} (unknown payload)", seg_id)
                };
                tracing::error!("{}", msg);
                Err(RockDuckError::Compaction(msg))
            }
        }
    }

    fn execute_small_file_merge(&self, seg_ids: &[String]) -> crate::error::Result<MergeStats> {
        // For small file merge, we run PDT merge on each segment and sum the stats.
        // A true multi-way merge would be more efficient, but this provides
        // correctness at minimal implementation cost.
        let mut total_stats = MergeStats::default();
        for seg_id in seg_ids {
            match self.execute_pdt_merge(seg_id) {
                Ok(stats) => {
                    total_stats.rows_read += stats.rows_read;
                    total_stats.rows_written += stats.rows_written;
                    total_stats.rows_dropped += stats.rows_dropped;
                    total_stats.bytes_read += stats.bytes_read;
                    total_stats.bytes_written += stats.bytes_written;
                    total_stats.granules_created += stats.granules_created;
                }
                Err(e) => {
                    tracing::warn!("Small file merge failed for seg={}: {}", seg_id, e);
                }
            }
        }
        Ok(total_stats)
    }

    fn execute_query_driven(&self, seg_id: &str) -> crate::error::Result<bool> {
        let Some(ref tracker) = self.access_tracker else {
            tracing::debug!(
                seg_id,
                "Query-driven compaction remains blocked: AccessTracker not wired"
            );
            return Ok(false);
        };
        if !tracker.should_range_reduce(seg_id) {
            tracing::debug!(
                seg_id,
                "Query-driven compaction skipped: producer preconditions not met"
            );
            return Ok(false);
        }

        let priority = tracker.priority_score(seg_id);
        tracing::debug!(
            seg_id,
            priority,
            "Query-driven compaction remains runtime-owned: task passed sizing/admission gates, but physical rewrite is still not implemented"
        );
        Ok(false)
    }

    fn set_compaction_callback(
        &self,
        cb: Option<
            Box<dyn Fn(String, crate::compaction::adaptive::DebtResolutionOutcome) + Send + Sync>,
        >,
    ) {
        *self.compaction_callback.write() = cb;
    }
}
