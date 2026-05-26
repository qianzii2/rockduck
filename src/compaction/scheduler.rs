//! Compaction 调度器
//!
//! 优先级分数：score = del_ratio^2 * 10 + log(size_mb) * 0.5 + log(age_hours) * 0.3
//! 触发条件：
//! - del_ratio > 0.5（必须合并）
//! - segment_size < 1MB（小文件合并）
//! - 后台 I/O 空闲时

use std::collections::BinaryHeap;
use std::cmp::Ordering;
use tracing::debug;

use crate::db::RockDuck;
use crate::error::Result;
use crate::metadata;
use crate::segment::meta::SegmentStatus;

/// Compaction 任务
#[derive(Debug)]
pub struct CompactionTask {
    pub seg_id: String,
    pub priority: f64,
    pub reason: CompactionReason,
    pub created_at: u64,
}

impl PartialEq for CompactionTask {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority
    }
}

impl Eq for CompactionTask {}

impl CompactionTask {
    pub fn new(seg_id: String, priority: f64, reason: CompactionReason) -> Self {
        Self {
            seg_id,
            priority,
            reason,
            created_at: crate::codec::current_timestamp_secs(),
        }
    }
}

impl Ord for CompactionTask {
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority.partial_cmp(&other.priority).unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for CompactionTask {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.priority.partial_cmp(&other.priority)
    }
}

/// Compaction 原因
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompactionReason {
    HighDeleteRatio,
    SmallFile,
    Periodic,
    /// 单列更新多但删除少：先增量物化，不触发 full compaction（Feature 4）
    IncrementalMaterialize,
}

/// Compaction 调度器
pub struct CompactionScheduler {
    /// 优先级队列
    queue: BinaryHeap<CompactionTask>,
    /// 调度间隔（秒）
    interval_secs: u64,
    /// del_ratio 阈值
    del_ratio_threshold: f64,
    /// 小文件阈值（字节）
    small_file_threshold: usize,
}

impl CompactionScheduler {
    pub fn new() -> Self {
        Self {
            queue: BinaryHeap::new(),
            interval_secs: 60,
            del_ratio_threshold: 0.5,
            small_file_threshold: 1024 * 1024,
        }
    }

    /// 评估所有 segment 并更新优先级
    pub fn evaluate(&mut self, db: &RockDuck) -> Result<()> {
        let seg_ids = metadata::seg_meta::list_table_segments(&db.db, "default")?;
        let now = crate::codec::current_timestamp_secs();

        for seg_id in seg_ids {
            if let Some(meta) = metadata::rocksdb::get_segment_meta(&db.db, &seg_id)? {
                if meta.status != SegmentStatus::Active {
                    continue;
                }

                let priority = self.calculate_priority(&meta, now);
                let reason = self.determine_reason(&meta);

                if priority > 0.0 {
                    self.queue.push(CompactionTask::new(seg_id.clone(), priority, reason));
                    debug!("Compaction candidate: seg_id={}, priority={:.2}, reason={:?}",
                        seg_id, priority, reason);
                }
            }
        }

        Ok(())
    }

    /// 计算优先级分数
    pub(crate) fn calculate_priority(&self, meta: &crate::metadata::SegmentMeta, now: u64) -> f64 {
        let age_hours = if meta.created_at > 0 {
            ((now - meta.created_at) as f64) / 3600.0
        } else {
            1.0
        };

        let size_mb = meta.uncompressed_size as f64 / (1024.0 * 1024.0);

        // 优先级分数公式
        let del_score = meta.del_ratio.powi(2) * 10.0;
        let size_score = size_mb.max(1.0).log2() * 0.5;
        let age_score = age_hours.log2() * 0.3;

        let priority = del_score + size_score + age_score;
        priority
    }

    /// 确定 compaction 原因
    pub(crate) fn determine_reason(&self, meta: &crate::metadata::SegmentMeta) -> CompactionReason {
        if meta.del_ratio > self.del_ratio_threshold {
            CompactionReason::HighDeleteRatio
        } else if meta.uncompressed_size < self.small_file_threshold as u64 {
            CompactionReason::SmallFile
        } else {
            CompactionReason::Periodic
        }
    }

    /// 确定 compaction 原因（Feature 4: 带 Update Mask 的版本）
    #[allow(dead_code)]
    pub(crate) fn determine_reason_with_updates(
        &self,
        meta: &crate::metadata::SegmentMeta,
        upd_ratio: f64,
        del_ratio: f64,
    ) -> CompactionReason {
        // 条件：单列更新多但删除少 → 增量物化更划算
        if upd_ratio > 0.3 && del_ratio < 0.5 {
            CompactionReason::IncrementalMaterialize
        } else {
            self.determine_reason(meta)
        }
    }

    /// 计算优先级分数（Feature 5: Query Feedback 驱动的增强版）
    ///
    /// score = del_ratio^2 * 10
    ///       + log(size_mb) * 0.3
    ///       + log(age_hours) * 0.2
    ///       + feedback.staleness_penalty(seg_id) * 5.0   // Zone Map 失准惩罚
    ///       + (1.0 - feedback.prune_hit_ratio(seg_id)) * 3.0  // 裁剪失效惩罚
    pub fn calculate_priority_with_feedback(
        &self,
        meta: &crate::metadata::SegmentMeta,
        now: u64,
        feedback: &crate::query::feedback::QueryFeedbackCollector,
    ) -> f64 {
        let age_hours = if meta.created_at > 0 {
            ((now - meta.created_at) as f64) / 3600.0
        } else {
            1.0
        };

        let size_mb = meta.uncompressed_size as f64 / (1024.0 * 1024.0);

        let del_score = meta.del_ratio.powi(2) * 10.0;
        let size_score = size_mb.max(1.0).log2() * 0.3;
        let age_score = age_hours.log2() * 0.2;

        // Feature 5: Query feedback penalties
        let stale_penalty = feedback.staleness_penalty(&meta.seg_id) * 5.0;
        let miss_penalty = (1.0 - feedback.prune_hit_ratio(&meta.seg_id)) * 3.0;

        del_score + size_score + age_score + stale_penalty + miss_penalty
    }

    /// 获取下一个 compaction 任务
    pub fn next_task(&mut self) -> Option<CompactionTask> {
        self.queue.pop()
    }

    /// 检查是否有待处理的 compaction
    pub fn has_pending(&self) -> bool {
        !self.queue.is_empty()
    }

    /// 清空队列
    pub fn clear(&mut self) {
        self.queue.clear();
    }

    /// 设置调度间隔
    pub fn set_interval(&mut self, secs: u64) {
        self.interval_secs = secs;
    }

    /// 设置 del_ratio 阈值
    pub fn set_del_ratio_threshold(&mut self, threshold: f64) {
        self.del_ratio_threshold = threshold;
    }

    /// 设置小文件阈值
    pub fn set_small_file_threshold(&mut self, threshold: usize) {
        self.small_file_threshold = threshold;
    }
}

impl Default for CompactionScheduler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ============================================================
    // CompactionScheduler construction
    // ============================================================

    #[test]
    fn test_compaction_scheduler_new() {
        let scheduler = CompactionScheduler::new();
        assert!(!scheduler.has_pending());
    }

    #[test]
    fn test_compaction_scheduler_default() {
        let scheduler: CompactionScheduler = Default::default();
        assert!(!scheduler.has_pending());
    }

    // ============================================================
    // CompactionTask
    // ============================================================

    #[test]
    fn test_compaction_task_new() {
        let task = CompactionTask::new(
            "seg_001".to_string(),
            5.0,
            CompactionReason::HighDeleteRatio,
        );
        assert_eq!(task.seg_id, "seg_001");
        assert!((task.priority - 5.0).abs() < 1e-9);
        assert!(task.created_at > 0);
    }

    #[test]
    fn test_compaction_task_ordering() {
        let task_low = CompactionTask::new("seg_low".to_string(), 1.0, CompactionReason::Periodic);
        let task_high = CompactionTask::new("seg_high".to_string(), 10.0, CompactionReason::HighDeleteRatio);
        assert!(task_high > task_low);
        assert!(task_low < task_high);
    }

    #[test]
    fn test_compaction_task_eq() {
        let task1 = CompactionTask::new("seg_001".to_string(), 5.0, CompactionReason::Periodic);
        let task2 = CompactionTask::new("seg_001".to_string(), 5.0, CompactionReason::Periodic);
        // Same priority tasks are not greater or less than each other
        assert!(!(task1 > task2) && !(task1 < task2));
    }

    // ============================================================
    // CompactionScheduler methods
    // ============================================================

    #[test]
    fn test_compaction_scheduler_clear() {
        let mut scheduler: CompactionScheduler = Default::default();
        assert!(!scheduler.has_pending());
        scheduler.clear();
        assert!(!scheduler.has_pending());
    }

    #[test]
    fn test_compaction_scheduler_next_task_empty() {
        let mut scheduler = CompactionScheduler::new();
        assert!(scheduler.next_task().is_none());
    }

    #[test]
    fn test_compaction_scheduler_set_interval() {
        let mut scheduler = CompactionScheduler::new();
        scheduler.set_interval(120);
    }

    #[test]
    fn test_compaction_scheduler_set_del_ratio_threshold() {
        let mut scheduler = CompactionScheduler::new();
        scheduler.set_del_ratio_threshold(0.3);
    }

    // ============================================================
    // calculate_priority tests
    // ============================================================

    #[test]
    fn test_calculate_priority_formula() {
        let scheduler = CompactionScheduler::new();
        let mut meta = crate::segment::meta::SegmentMeta::new(
            "seg".into(), "t".into(), vec![],
        );
        meta.row_count = 100;
        meta.deleted_rows = 50;
        meta.del_ratio = 0.5;
        meta.uncompressed_size = 10 * 1024 * 1024; // 10 MB
        meta.created_at = 0; // Force age = 1.0
        let now = 3600; // 1 hour after epoch
        let priority = scheduler.calculate_priority(&meta, now);
        let expected = 0.5_f64.powi(2) * 10.0
            + (10.0_f64).log2() * 0.5
            + 1.0_f64.log2() * 0.3;
        assert!(
            (priority - expected).abs() < 1e-6,
            "priority={}, expected={}",
            priority,
            expected
        );
    }

    #[test]
    fn test_calculate_priority_zero_del_ratio() {
        let scheduler = CompactionScheduler::new();
        let mut meta = crate::segment::meta::SegmentMeta::new(
            "seg".into(), "t".into(), vec![],
        );
        meta.del_ratio = 0.0;
        meta.uncompressed_size = 5 * 1024 * 1024; // 5 MB
        meta.created_at = 0;
        let priority = scheduler.calculate_priority(&meta, 3600);
        // del_score = 0, size_score = log2(5)*0.5, age_score = log2(1)*0.3 = 0
        let expected = 5.0_f64.log2() * 0.5;
        assert!((priority - expected).abs() < 1e-6);
    }

    #[test]
    fn test_calculate_priority_full_del_ratio() {
        let scheduler = CompactionScheduler::new();
        let mut meta = crate::segment::meta::SegmentMeta::new(
            "seg".into(), "t".into(), vec![],
        );
        meta.del_ratio = 1.0;
        meta.uncompressed_size = 1 * 1024 * 1024; // 1 MB (minimum)
        meta.created_at = 0;
        let priority = scheduler.calculate_priority(&meta, 3600);
        // del_score = 1.0^2 * 10 = 10, size_score = log2(1)*0.5 = 0, age_score = log2(1)*0.3 = 0
        assert!((priority - 10.0).abs() < 1e-6);
    }

    // ============================================================
    // determine_reason tests
    // ============================================================

    #[test]
    fn test_determine_reason_high_del_ratio() {
        let mut scheduler = CompactionScheduler::new();
        scheduler.set_del_ratio_threshold(0.5);
        let mut meta = crate::segment::meta::SegmentMeta::new(
            "seg".into(), "t".into(), vec![],
        );
        meta.del_ratio = 0.6;
        meta.uncompressed_size = 10 * 1024 * 1024;
        let reason = scheduler.determine_reason(&meta);
        assert_eq!(reason, CompactionReason::HighDeleteRatio);
    }

    #[test]
    fn test_determine_reason_small_file() {
        let mut scheduler = CompactionScheduler::new();
        scheduler.set_small_file_threshold(1024 * 1024); // 1MB
        let mut meta = crate::segment::meta::SegmentMeta::new(
            "seg".into(), "t".into(), vec![],
        );
        meta.del_ratio = 0.1;
        meta.uncompressed_size = 512 * 1024; // 512KB < 1MB
        let reason = scheduler.determine_reason(&meta);
        assert_eq!(reason, CompactionReason::SmallFile);
    }

    #[test]
    fn test_determine_reason_periodic() {
        let scheduler = CompactionScheduler::new();
        let mut meta = crate::segment::meta::SegmentMeta::new(
            "seg".into(), "t".into(), vec![],
        );
        meta.del_ratio = 0.1;
        meta.uncompressed_size = 10 * 1024 * 1024;
        let reason = scheduler.determine_reason(&meta);
        assert_eq!(reason, CompactionReason::Periodic);
    }

    #[test]
    fn test_determine_reason_at_threshold() {
        let mut scheduler = CompactionScheduler::new();
        scheduler.set_del_ratio_threshold(0.5);
        let mut meta = crate::segment::meta::SegmentMeta::new(
            "seg".into(), "t".into(), vec![],
        );
        meta.del_ratio = 0.5; // Exactly at threshold, not > threshold
        meta.uncompressed_size = 10 * 1024 * 1024;
        let reason = scheduler.determine_reason(&meta);
        assert_eq!(reason, CompactionReason::Periodic);
    }

    // ============================================================
    // CompactionReason Debug
    // ============================================================

    #[test]
    fn test_compaction_reason_debug() {
        for reason in [
            CompactionReason::HighDeleteRatio,
            CompactionReason::SmallFile,
            CompactionReason::Periodic,
            CompactionReason::IncrementalMaterialize,
        ] {
            let debug_str = format!("{:?}", reason);
            assert!(!debug_str.is_empty());
        }
    }

    // ============================================================
    // CompactionTask Debug
    // ============================================================

    #[test]
    fn test_compaction_task_debug() {
        let task = CompactionTask::new("seg_001".to_string(), 5.0, CompactionReason::Periodic);
        let debug_str = format!("{:?}", task);
        assert!(!debug_str.is_empty());
    }

    // ============================================================
    // Feature 4: determine_reason_with_updates
    // ============================================================

    #[test]
    fn test_determine_reason_with_updates_incremental() {
        let scheduler = CompactionScheduler::new();
        let mut meta = crate::segment::meta::SegmentMeta::new(
            "seg".into(), "t".into(), vec![],
        );
        meta.del_ratio = 0.1;
        meta.uncompressed_size = 10 * 1024 * 1024;

        // upd_ratio = 0.4 > 0.3, del_ratio = 0.1 < 0.5 → IncrementalMaterialize
        let reason = scheduler.determine_reason_with_updates(&meta, 0.4, 0.1);
        assert_eq!(reason, CompactionReason::IncrementalMaterialize);
    }

    #[test]
    fn test_determine_reason_with_updates_high_del() {
        let scheduler = CompactionScheduler::new();
        let mut meta = crate::segment::meta::SegmentMeta::new(
            "seg".into(), "t".into(), vec![],
        );
        meta.del_ratio = 0.6;
        meta.uncompressed_size = 10 * 1024 * 1024;

        // del_ratio = 0.6 > 0.5 → HighDeleteRatio (not IncrementalMaterialize)
        let reason = scheduler.determine_reason_with_updates(&meta, 0.4, 0.6);
        assert_eq!(reason, CompactionReason::HighDeleteRatio);
    }

    #[test]
    fn test_determine_reason_with_updates_low_update() {
        let scheduler = CompactionScheduler::new();
        let mut meta = crate::segment::meta::SegmentMeta::new(
            "seg".into(), "t".into(), vec![],
        );
        meta.del_ratio = 0.1;
        meta.uncompressed_size = 10 * 1024 * 1024;

        // upd_ratio = 0.1 < 0.3 → fall through to Periodic
        let reason = scheduler.determine_reason_with_updates(&meta, 0.1, 0.1);
        assert_eq!(reason, CompactionReason::Periodic);
    }

    // ============================================================
    // Feature 5: calculate_priority_with_feedback
    // ============================================================

    #[test]
    fn test_calculate_priority_with_feedback() {
        let scheduler = CompactionScheduler::new();
        let mut meta = crate::segment::meta::SegmentMeta::new(
            "seg_001".into(), "t".into(), vec![],
        );
        meta.del_ratio = 0.5;
        meta.uncompressed_size = 10 * 1024 * 1024;
        meta.created_at = 0;
        let now = 3600; // 1 hour

        let feedback = crate::query::feedback::QueryFeedbackCollector::new();
        feedback.record_query("seg_001", 100, 5); // miss

        let priority = scheduler.calculate_priority_with_feedback(&meta, now, &feedback);

        // Trace: age_hours = (3600-0)/3600 = 1.0 → age_score = log2(1)*0.2 = 0
        //        size_mb = 10 → size_score = log2(10)*0.3 ≈ 0.997
        //        del_score = 0.5^2*10 = 2.5
        //        base = 2.5 + 0.997 + 0 = 3.497
        //        1 miss → staleness_penalty = 0.1 → stale_penalty = 0.1*5 = 0.5
        //        prune_hit_ratio = 0 hits / 1 miss = 0 → miss_penalty = (1-0)*3 = 3.0
        //        total ≈ 3.497 + 0.5 + 3.0 ≈ 7.0
        let base = 2.5 + (10.0_f64).log2() * 0.3;
        let expected = base + 0.5 + 3.0; // stale_penalty + miss_penalty

        assert!(
            (priority - expected).abs() < 0.1,
            "priority={:.2}, expected≈{:.2}",
            priority,
            expected
        );
    }

    #[test]
    fn test_calculate_priority_with_feedback_all_hits() {
        let scheduler = CompactionScheduler::new();
        let mut meta = crate::segment::meta::SegmentMeta::new(
            "seg_001".into(), "t".into(), vec![],
        );
        meta.del_ratio = 0.0;
        meta.uncompressed_size = 10 * 1024 * 1024;
        meta.created_at = 0;
        let now = 3600;

        let feedback = crate::query::feedback::QueryFeedbackCollector::new();
        // 3 hits, 0 misses → hit ratio = 1.0
        feedback.record_query("seg_001", 100, 90);
        feedback.record_query("seg_001", 100, 80);
        feedback.record_query("seg_001", 100, 95);

        let priority = scheduler.calculate_priority_with_feedback(&meta, now, &feedback);

        // Base score: del=0, size≈1.04, age=0 → 1.04
        // Penalties: staleness ≈ 0, miss_ratio = 0 → 0
        let expected = (10.0_f64).log2() * 0.3;
        assert!(
            (priority - expected).abs() < 0.1,
            "priority={:.2}, expected≈{:.2}",
            priority,
            expected
        );
    }
}
