//! Adaptive Compaction Scheduler — Hill Climbing Weight Optimization
//!
//! 替代固定的 compaction 权重系数，用 hill climbing 自动调优。
//!
//! 思路：
//! - 每次 compaction 完成后，检查 Zone Map 命中率是否提升
//! - 如果提升 → 继续往同方向调权重
//! - 如果下降 → 回退并反向
//! - 每 N 轮评估一次，避免抖动
//!
//! 与固定权重模型的对比：
//! - 固定：`staleness_coef = 5.0`（手调）
//! - 自适应：`staleness_coef` 根据实际 Zone Map 命中率自动调整

use crate::query::feedback::QueryFeedbackCollector;
use tracing::debug;

/// 自适应 compaction 权重
#[derive(Debug, Clone)]
pub struct CompactionWeights {
    /// 删除率系数（高删除率优先合并）
    pub del_coef: f64,
    /// 大小系数（小文件合并）
    pub size_coef: f64,
    /// 年龄系数（老数据优先合并）
    pub age_coef: f64,
    /// Zone Map 失准惩罚系数
    pub stale_coef: f64,
    /// 裁剪失效惩罚系数
    pub miss_coef: f64,
}

impl Default for CompactionWeights {
    fn default() -> Self {
        Self {
            del_coef: 10.0,
            size_coef: 0.3,
            age_coef: 0.2,
            stale_coef: 5.0,
            miss_coef: 3.0,
        }
    }
}

impl CompactionWeights {
    /// 使用指定初始值创建权重
    pub fn new(del: f64, size: f64, age: f64, stale: f64, miss: f64) -> Self {
        Self {
            del_coef: del,
            size_coef: size,
            age_coef: age,
            stale_coef: stale,
            miss_coef: miss,
        }
    }
}

/// 自适应 compaction 调度器
pub struct AdaptiveCompactionScheduler {
    /// 当前权重
    weights: CompactionWeights,
    /// 查询反馈收集器
    feedback: QueryFeedbackCollector,
    /// 上次调整后的权重（用于回退）
    prev_weights: CompactionWeights,
    /// 调整步长
    step_size: f64,
    /// 最小查询次数阈值（低于此不做调整）
    min_query_count: u64,
    /// 调整间隔（每 N 次调度评估后调整一次）
    adjustment_interval: usize,
    /// 调度调用计数
    call_count: usize,
    /// 上次 compaction 后 Zone Map 命中率
    prev_hit_ratio: f64,
}

impl AdaptiveCompactionScheduler {
    pub fn new() -> Self {
        Self {
            weights: CompactionWeights::default(),
            feedback: QueryFeedbackCollector::new(),
            prev_weights: CompactionWeights::default(),
            step_size: 0.5,
            min_query_count: 5,
            adjustment_interval: 10,
            call_count: 0,
            prev_hit_ratio: 0.5,
        }
    }

    /// 获取当前权重
    pub fn weights(&self) -> &CompactionWeights {
        &self.weights
    }

    /// 获取查询反馈收集器引用
    pub fn feedback(&self) -> &QueryFeedbackCollector {
        &self.feedback
    }

    /// 获取查询反馈收集器可变引用（用于记录）
    pub fn feedback_mut(&mut self) -> &mut QueryFeedbackCollector {
        &mut self.feedback
    }

    /// 记录一次 compaction 后的性能指标（用于 hill climbing）
    ///
    /// `prune_hit_ratio_after` — compaction 后该 segment 的 Zone Map 命中率
    pub fn record_compaction_result(&mut self, seg_id: &str, prune_hit_ratio_after: f64) {
        let delta = prune_hit_ratio_after - self.prev_hit_ratio;

        if delta > 0.05 {
            // 命中率提升 → 确认当前调整方向，继续
            debug!("Compaction improved hit ratio: {:.3} -> {:.3} (delta={:.3})",
                self.prev_hit_ratio, prune_hit_ratio_after, delta);
        } else if delta < -0.05 {
            // 命中率下降 → 回退到上次权重
            self.weights = self.prev_weights.clone();
            self.step_size = (self.step_size * 0.5).max(0.05);
            debug!("Compaction degraded hit ratio, rolling back (step_size={:.3})", self.step_size);
        }

        self.prev_hit_ratio = prune_hit_ratio_after;
    }

    /// Hill climbing 调整权重（每 adjustment_interval 次调用评估一次）
    pub fn adjust(&mut self) {
        self.call_count += 1;

        if self.call_count % self.adjustment_interval != 0 {
            return;
        }

        let stats = self.feedback.get_all_stats();
        if stats.is_empty() {
            return;
        }

        // 保存当前权重（用于回退）
        self.prev_weights = self.weights.clone();

        for (seg_id, s) in stats {
            if s.query_count < self.min_query_count {
                continue;
            }

            // 策略：如果某个 segment Zone Map 持续失准（penalty > 0.5）→ 提高 stale_coef
            if s.staleness_penalty > 0.5 {
                let new_val = self.weights.stale_coef + self.step_size;
                if new_val <= 20.0 {
                    self.weights.stale_coef = new_val;
                    debug!("HC: seg={} stale_penalty={:.2} -> stale_coef={:.2}",
                        seg_id, s.staleness_penalty, self.weights.stale_coef);
                }
            } else if s.staleness_penalty < 0.1 && s.query_count > 20 {
                // Zone Map 持续准确 → 可以降低 stale_coef（更保守的 compaction）
                let new_val = self.weights.stale_coef - self.step_size * 0.5;
                if new_val >= 1.0 {
                    self.weights.stale_coef = new_val;
                    debug!("HC: seg={} stable -> stale_coef={:.2}",
                        seg_id, self.weights.stale_coef);
                }
            }

            // 策略：如果 prune miss 率高 → 提高 miss_coef
            let miss_ratio = 1.0 - s.prune_hit_ratio();
            if miss_ratio > 0.5 && s.query_count > 10 {
                let new_val = self.weights.miss_coef + self.step_size;
                if new_val <= 10.0 {
                    self.weights.miss_coef = new_val;
                    debug!("HC: seg={} miss_ratio={:.2} -> miss_coef={:.2}",
                        seg_id, miss_ratio, self.weights.miss_coef);
                }
            }
        }
    }

    /// 计算某个 segment 的优先级（使用自适应权重）
    pub fn calculate_priority(
        &self,
        meta: &crate::segment::meta::SegmentMeta,
        now: u64,
    ) -> f64 {
        let age_hours = if meta.created_at > 0 {
            ((now - meta.created_at) as f64) / 3600.0
        } else {
            1.0
        };

        let size_mb = meta.uncompressed_size as f64 / (1024.0 * 1024.0);

        let del_score = meta.del_ratio.powi(2) * self.weights.del_coef;
        let size_score = size_mb.max(1.0).log2() * self.weights.size_coef;
        let age_score = age_hours.log2() * self.weights.age_coef;

        // Zone Map 失准惩罚
        let stale_penalty = self.feedback.staleness_penalty(&meta.seg_id) * self.weights.stale_coef;
        // 裁剪失效惩罚
        let miss_penalty = (1.0 - self.feedback.prune_hit_ratio(&meta.seg_id)) * self.weights.miss_coef;

        del_score + size_score + age_score + stale_penalty + miss_penalty
    }
}

impl Default for AdaptiveCompactionScheduler {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for AdaptiveCompactionScheduler {
    fn clone(&self) -> Self {
        Self {
            weights: self.weights.clone(),
            feedback: self.feedback.clone(),
            prev_weights: self.prev_weights.clone(),
            step_size: self.step_size,
            min_query_count: self.min_query_count,
            adjustment_interval: self.adjustment_interval,
            call_count: self.call_count,
            prev_hit_ratio: self.prev_hit_ratio,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compaction_weights_default() {
        let w = CompactionWeights::default();
        assert_eq!(w.del_coef, 10.0);
        assert_eq!(w.stale_coef, 5.0);
        assert_eq!(w.miss_coef, 3.0);
    }

    #[test]
    fn test_compaction_weights_new() {
        let w = CompactionWeights::new(8.0, 0.5, 0.4, 3.0, 2.0);
        assert_eq!(w.del_coef, 8.0);
        assert_eq!(w.stale_coef, 3.0);
    }

    #[test]
    fn test_adaptive_scheduler_new() {
        let scheduler = AdaptiveCompactionScheduler::new();
        assert_eq!(scheduler.weights.del_coef, 10.0);
        assert_eq!(scheduler.step_size, 0.5);
        assert_eq!(scheduler.adjustment_interval, 10);
    }

    #[test]
    fn test_record_compaction_improves() {
        let mut scheduler = AdaptiveCompactionScheduler::new();
        scheduler.prev_hit_ratio = 0.3;
        scheduler.record_compaction_result("seg_001", 0.6);
        // delta = 0.3 > 0.05, no rollback
        assert_eq!(scheduler.prev_hit_ratio, 0.6);
    }

    #[test]
    fn test_record_compaction_degrades() {
        let mut scheduler = AdaptiveCompactionScheduler::new();
        scheduler.weights.stale_coef = 7.0;
        scheduler.prev_weights.stale_coef = 5.0;
        scheduler.prev_hit_ratio = 0.6;
        scheduler.record_compaction_result("seg_001", 0.3);
        // delta = -0.3 < -0.05, should rollback
        assert_eq!(scheduler.weights.stale_coef, 5.0); // rolled back
        assert!(scheduler.step_size < 0.5); // step_size halved
    }

    #[test]
    fn test_adjust_no_stats() {
        let mut scheduler = AdaptiveCompactionScheduler::new();
        scheduler.adjust();
        // No stats, no change
        assert_eq!(scheduler.weights.stale_coef, 5.0);
    }

    #[test]
    fn test_clone() {
        let scheduler = AdaptiveCompactionScheduler::new();
        let cloned = scheduler.clone();
        assert_eq!(cloned.weights.del_coef, scheduler.weights.del_coef);
    }

    #[test]
    fn test_calculate_priority_del_ratio_high_wins() {
        let scheduler = AdaptiveCompactionScheduler::new();

        // Same size and age, but different del_ratio
        let meta_low = crate::segment::meta::SegmentMeta::new(
            "seg_low".to_string(),
            "t".to_string(),
            vec![],
        );
        let mut meta_low = meta_low;
        meta_low.del_ratio = 0.05;
        meta_low.created_at = 1000;

        let meta_high = crate::segment::meta::SegmentMeta::new(
            "seg_high".to_string(),
            "t".to_string(),
            vec![],
        );
        let mut meta_high = meta_high;
        meta_high.del_ratio = 0.2;
        meta_high.created_at = 1000;

        let now = 2000u64;
        let priority_low = scheduler.calculate_priority(&meta_low, now);
        let priority_high = scheduler.calculate_priority(&meta_high, now);

        // Higher del_ratio should give higher priority
        assert!(
            priority_high > priority_low,
            "priority_high({:.3}) should be > priority_low({:.3})",
            priority_high,
            priority_low
        );
    }

    #[test]
    fn test_calculate_priority_feedback_penalty() {
        let scheduler = AdaptiveCompactionScheduler::new();

        let mut meta = crate::segment::meta::SegmentMeta::new(
            "seg_001".to_string(),
            "t".to_string(),
            vec![],
        );
        meta.created_at = 1000;

        let now = 2000u64;

        // The priority formula adds: staleness_penalty * stale_coef
        // Without feedback, staleness_penalty = 0.0, so penalty contribution = 0
        let priority_baseline = scheduler.calculate_priority(&meta, now);

        // Inject feedback: staleness_penalty(seg_001) = 0.0 (no feedback registered)
        // The penalty component in priority is:
        //   self.feedback.staleness_penalty(&meta.seg_id) * self.weights.stale_coef
        // Since we can't easily inject a specific penalty without modifying the internal stats,
        // we verify the baseline priority does NOT include penalty contributions
        // (i.e., priority is determined solely by del/size/age scores when no feedback exists)
        assert!(
            priority_baseline >= 0.0,
            "baseline priority should be non-negative"
        );

        // Higher penalty increases priority through the stale_penalty term
        // record_miss raises staleness_penalty, which adds to priority
        let mut scheduler2 = AdaptiveCompactionScheduler::new();
        // Inject a prune miss (Zone Map over-pruned): estimated >> actual
        scheduler2.feedback_mut().record_query("seg_001", 100, 0);
        let priority_with_feedback = scheduler2.calculate_priority(&meta, now);

        // With a prune miss recorded, staleness_penalty > 0, so priority should be higher
        assert!(
            priority_with_feedback > priority_baseline,
            "priority_with_feedback({:.3}) should exceed baseline({:.3})",
            priority_with_feedback,
            priority_baseline
        );
    }

    #[test]
    fn test_adjust_weights_clamped() {
        let mut scheduler = AdaptiveCompactionScheduler::new();

        // Force staleness penalty > 0.5 by recording many prune misses
        // record_query(estimated > actual * 2) triggers a miss, which increases staleness
        for _ in 0..100 {
            scheduler.feedback_mut().record_query("seg_hot", 100, 0);
        }
        scheduler.call_count = 9; // Will become 10 after first adjust
        scheduler.step_size = 5.0;

        // Try to push stale_coef above 20.0
        for _ in 0..10 {
            scheduler.adjust();
        }

        // stale_coef should be clamped at 20.0 (upper bound)
        assert!(
            scheduler.weights.stale_coef <= 20.0,
            "stale_coef ({:.2}) should be clamped at 20.0",
            scheduler.weights.stale_coef
        );

        // Try to push stale_coef below 1.0 (lower bound per the code: new_val >= 1.0)
        let mut scheduler2 = AdaptiveCompactionScheduler::new();
        scheduler2.step_size = 0.5;
        // With prune hits (actual >= estimated / 2), staleness decreases
        for _ in 0..100 {
            scheduler2.feedback_mut().record_query("seg_cool", 0, 100);
        }
        scheduler2.call_count = 9;

        for _ in 0..20 {
            scheduler2.adjust();
        }

        // stale_coef should stay at minimum 1.0 (lower bound)
        assert!(
            scheduler2.weights.stale_coef >= 1.0,
            "stale_coef ({:.2}) should not go below 1.0",
            scheduler2.weights.stale_coef
        );
    }

    #[test]
    fn test_record_compaction_result_no_improvement_no_change() {
        let mut scheduler = AdaptiveCompactionScheduler::new();
        scheduler.weights.stale_coef = 7.0;
        scheduler.prev_weights.stale_coef = 7.0;
        scheduler.prev_hit_ratio = 0.5;

        // delta = 0.03, absolute value < 0.05, should not trigger rollback
        scheduler.record_compaction_result("seg_001", 0.53);

        // Weights should remain unchanged
        assert_eq!(scheduler.weights.stale_coef, 7.0);
        assert_eq!(scheduler.prev_hit_ratio, 0.53);

        // Also test negative delta within threshold
        scheduler.prev_hit_ratio = 0.5;
        scheduler.record_compaction_result("seg_001", 0.48);
        // delta = -0.02, absolute value < 0.05, no rollback
        assert_eq!(scheduler.weights.stale_coef, 7.0);
    }

    #[test]
    fn test_compaction_weights_bounds_known() {
        let w = CompactionWeights::new(1.0, 0.1, 0.1, 2.0, 1.0);
        assert_eq!(w.del_coef, 1.0);
        assert_eq!(w.stale_coef, 2.0);
        assert_eq!(w.miss_coef, 1.0);
    }

    #[test]
    fn test_adjust_not_called_on_interval_mismatch() {
        let mut scheduler = AdaptiveCompactionScheduler::new();
        scheduler.step_size = 5.0;
        // Register feedback so adjust() has something to act on
        for _ in 0..100 {
            scheduler.feedback_mut().record_query("seg_hot", 100, 0);
        }

        let original_stale = scheduler.weights.stale_coef;

        // call_count is 0, 0 % 10 != 0, so adjust should do nothing
        scheduler.call_count = 1;
        scheduler.adjust();
        assert_eq!(scheduler.weights.stale_coef, original_stale);

        scheduler.call_count = 5;
        scheduler.adjust();
        assert_eq!(scheduler.weights.stale_coef, original_stale);
    }

    #[test]
    fn test_miss_coef_clamped_at_10() {
        let mut scheduler = AdaptiveCompactionScheduler::new();
        scheduler.step_size = 5.0;

        // Record high miss ratio: prune misses mean (1 - prune_hit_ratio) is high
        // This triggers miss_ratio > 0.5 and query_count > 10 path
        for _ in 0..50 {
            scheduler.feedback_mut().record_query("seg_missy", 100, 0);
        }
        scheduler.call_count = 9;

        for _ in 0..20 {
            scheduler.adjust();
        }

        // miss_coef should be clamped at 10.0
        assert!(
            scheduler.weights.miss_coef <= 10.0,
            "miss_coef ({:.2}) should be clamped at 10.0",
            scheduler.weights.miss_coef
        );
    }
}
