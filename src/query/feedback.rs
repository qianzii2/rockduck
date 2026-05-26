//! Query Feedback Collector — Track query patterns to drive compaction scheduling
//!
//! Accumulates per-segment statistics about:
//!   - How often each segment is queried
//!   - Whether Zone Map pruning helped (prune_hits) or hurt (prune_misses)
//!   - Accumulated staleness penalty for compaction scoring
//!
//!参照:
//!   - WAIR (Snowflake): query-driven incremental reclustering
//!   - Databricks AQE: Adaptive Query Execution with runtime statistics

use std::collections::HashMap;
use tracing::debug;

/// Per-segment query statistics
#[derive(Debug, Clone)]
pub struct SegmentQueryStats {
    /// Number of queries involving this segment
    pub query_count: u64,
    /// Zone Map helped prune granules (estimated >> actual matches)
    pub zone_map_prune_hits: u64,
    /// Zone Map failed to prune (estimated much larger than actual matches)
    pub zone_map_prune_misses: u64,
    /// Accumulated staleness penalty (increases when misses > hits)
    pub staleness_penalty: f64,
}

impl Default for SegmentQueryStats {
    fn default() -> Self {
        Self {
            query_count: 0,
            zone_map_prune_hits: 0,
            zone_map_prune_misses: 0,
            staleness_penalty: 0.0,
        }
    }
}

impl SegmentQueryStats {
    /// Record a prune hit: Zone Map estimated many granules, actual had many matches
    pub fn record_hit(&mut self) {
        self.query_count += 1;
        self.zone_map_prune_hits += 1;
        // Decrease penalty on hit
        self.staleness_penalty = (self.staleness_penalty - 0.05).max(0.0);
    }

    /// Record a prune miss: Zone Map estimated many granules, actual had few matches
    pub fn record_miss(&mut self) {
        self.query_count += 1;
        self.zone_map_prune_misses += 1;
        // Increase penalty on miss (capped at 1.0)
        self.staleness_penalty = (self.staleness_penalty + 0.1).min(1.0);
    }

    /// Record an inconclusive result (too close to call)
    pub fn record_neutral(&mut self) {
        self.query_count += 1;
    }

    /// Prune hit ratio: hits / (hits + misses)
    pub fn prune_hit_ratio(&self) -> f64 {
        let total = self.zone_map_prune_hits + self.zone_map_prune_misses;
        if total == 0 {
            0.5 // neutral if no data yet
        } else {
            self.zone_map_prune_hits as f64 / total as f64
        }
    }

    /// Prune miss ratio
    pub fn prune_miss_ratio(&self) -> f64 {
        1.0 - self.prune_hit_ratio()
    }

    /// Confidence: how many queries have we recorded?
    pub fn confidence(&self) -> u64 {
        self.zone_map_prune_hits + self.zone_map_prune_misses
    }
}

/// Global query feedback collector
#[derive(Debug, Default)]
pub struct QueryFeedbackCollector {
    /// seg_id → stats
    stats: std::sync::RwLock<HashMap<String, SegmentQueryStats>>,
}

impl Clone for QueryFeedbackCollector {
    fn clone(&self) -> Self {
        Self::default()
    }
}

impl QueryFeedbackCollector {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a query result for a segment.
    ///
    /// `estimated_pruned` — how many granules Zone Map said could be skipped
    /// `actual_matching` — how many granules actually contained matching rows
    pub fn record_query(
        &self,
        seg_id: &str,
        estimated_pruned_granules: u64,
        actual_matching_granules: u64,
    ) {
        if estimated_pruned_granules == 0 && actual_matching_granules == 0 {
            // Nothing to record
            return;
        }

        let mut stats = self.stats.write().unwrap();
        let entry = stats.entry(seg_id.to_string()).or_default();

        if estimated_pruned_granules > actual_matching_granules * 2 {
            // Zone Map over-pruned: it estimated many granules could be skipped,
            // but most actually had matching rows → prune miss
            entry.record_miss();
            debug!(
                "QueryFeedback: seg={} miss (est={}, actual={})",
                seg_id, estimated_pruned_granules, actual_matching_granules
            );
        } else if actual_matching_granules >= estimated_pruned_granules / 2 {
            // Zone Map was reasonably accurate → prune hit
            entry.record_hit();
        } else {
            entry.record_neutral();
        }
    }

    /// Get staleness penalty for a segment (0.0 - 1.0)
    pub fn staleness_penalty(&self, seg_id: &str) -> f64 {
        let stats = self.stats.read().unwrap();
        stats.get(seg_id)
            .map(|s| s.staleness_penalty)
            .unwrap_or(0.0)
    }

    /// Get prune hit ratio for a segment
    pub fn prune_hit_ratio(&self, seg_id: &str) -> f64 {
        let stats = self.stats.read().unwrap();
        stats.get(seg_id)
            .map(|s| s.prune_hit_ratio())
            .unwrap_or(0.5)
    }

    /// Get all segment stats (for debugging / inspection)
    pub fn get_stats(&self) -> HashMap<String, SegmentQueryStats> {
        let stats = self.stats.read().unwrap();
        stats.clone()
    }

    /// Alias for get_stats — returns all per-segment query statistics
    pub fn get_all_stats(&self) -> HashMap<String, SegmentQueryStats> {
        self.get_stats()
    }

    /// Get stats for a specific segment
    pub fn get_segment_stats(&self, seg_id: &str) -> Option<SegmentQueryStats> {
        let stats = self.stats.read().unwrap();
        stats.get(seg_id).cloned()
    }

    /// Clear stats for a segment (e.g., after compaction)
    pub fn clear_segment(&self, seg_id: &str) {
        let mut stats = self.stats.write().unwrap();
        stats.remove(seg_id);
    }

    /// Clear all stats
    pub fn clear_all(&self) {
        let mut stats = self.stats.write().unwrap();
        stats.clear();
    }

    /// Get segments sorted by staleness penalty (highest first)
    pub fn top_stale_segments(&self, limit: usize) -> Vec<(String, f64)> {
        let stats = self.stats.read().unwrap();
        let mut entries: Vec<_> = stats.iter()
            .map(|(k, v)| (k.clone(), v.staleness_penalty))
            .collect();
        entries.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        entries.truncate(limit);
        entries
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ============================================================
    // SegmentQueryStats
    // ============================================================

    #[test]
    fn test_stats_default() {
        let s = SegmentQueryStats::default();
        assert_eq!(s.query_count, 0);
        assert_eq!(s.zone_map_prune_hits, 0);
        assert_eq!(s.zone_map_prune_misses, 0);
        assert!((s.staleness_penalty - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_record_hit() {
        let mut s = SegmentQueryStats::default();
        s.record_hit();
        assert_eq!(s.query_count, 1);
        assert_eq!(s.zone_map_prune_hits, 1);
        assert!((s.staleness_penalty - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_record_miss() {
        let mut s = SegmentQueryStats::default();
        s.record_miss();
        assert_eq!(s.query_count, 1);
        assert_eq!(s.zone_map_prune_misses, 1);
        assert!((s.staleness_penalty - 0.1).abs() < 1e-9);
    }

    #[test]
    fn test_stats_penalty_capped() {
        let mut s = SegmentQueryStats::default();
        for _ in 0..15 {
            s.record_miss();
        }
        assert!((s.staleness_penalty - 1.0).abs() < 1e-9, "penalty should cap at 1.0");
    }

    #[test]
    fn test_stats_penalty_floor() {
        let mut s = SegmentQueryStats::default();
        s.staleness_penalty = 0.1;
        s.record_hit();
        assert!((s.staleness_penalty - 0.05).abs() < 1e-9);
        s.record_hit();
        assert!((s.staleness_penalty - 0.0).abs() < 1e-9, "penalty should floor at 0.0");
    }

    #[test]
    fn test_stats_prune_hit_ratio() {
        let mut s = SegmentQueryStats::default();
        assert!((s.prune_hit_ratio() - 0.5).abs() < 1e-9, "no data -> 0.5");

        s.record_miss();
        s.record_miss();
        s.record_miss();
        s.record_hit();
        assert!((s.prune_hit_ratio() - 0.25).abs() < 1e-9);
    }

    #[test]
    fn test_stats_confidence() {
        let mut s = SegmentQueryStats::default();
        s.record_hit();
        s.record_miss();
        s.record_neutral();
        assert_eq!(s.confidence(), 2); // only hit/miss count
    }

    // ============================================================
    // QueryFeedbackCollector
    // ============================================================

    #[test]
    fn test_collector_new() {
        let collector = QueryFeedbackCollector::new();
        assert!(collector.get_stats().is_empty());
    }

    #[test]
    fn test_collector_record_miss() {
        let collector = QueryFeedbackCollector::new();
        collector.record_query("seg_001", 100, 10);

        let stats = collector.get_segment_stats("seg_001").unwrap();
        assert_eq!(stats.zone_map_prune_misses, 1);
        assert!((stats.staleness_penalty - 0.1).abs() < 1e-9);
    }

    #[test]
    fn test_collector_record_hit() {
        let collector = QueryFeedbackCollector::new();
        collector.record_query("seg_001", 100, 80);

        let stats = collector.get_segment_stats("seg_001").unwrap();
        assert_eq!(stats.zone_map_prune_hits, 1);
    }

    #[test]
    fn test_collector_staleness_penalty() {
        let collector = QueryFeedbackCollector::new();
        collector.record_query("seg_001", 100, 10); // miss
        collector.record_query("seg_001", 100, 10); // miss

        let penalty = collector.staleness_penalty("seg_001");
        assert!((penalty - 0.2).abs() < 1e-9);
    }

    #[test]
    fn test_collector_staleness_penalty_missing() {
        let collector = QueryFeedbackCollector::new();
        assert!((collector.staleness_penalty("seg_999") - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_collector_prune_hit_ratio() {
        let collector = QueryFeedbackCollector::new();
        collector.record_query("seg_001", 100, 5);  // miss
        collector.record_query("seg_001", 100, 90);  // hit
        collector.record_query("seg_001", 100, 5);  // miss

        let ratio = collector.prune_hit_ratio("seg_001");
        assert!((ratio - 1.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_collector_clear_segment() {
        let collector = QueryFeedbackCollector::new();
        collector.record_query("seg_001", 100, 10);
        collector.clear_segment("seg_001");
        assert!(collector.get_segment_stats("seg_001").is_none());
    }

    #[test]
    fn test_collector_clear_all() {
        let collector = QueryFeedbackCollector::new();
        collector.record_query("seg_001", 100, 10);
        collector.record_query("seg_002", 100, 10);
        collector.clear_all();
        assert!(collector.get_stats().is_empty());
    }

    #[test]
    fn test_collector_top_stale_segments() {
        let collector = QueryFeedbackCollector::new();
        collector.record_query("seg_a", 100, 5);  // staleness 0.1
        collector.record_query("seg_b", 100, 5);  // staleness 0.1
        collector.record_query("seg_b", 100, 5);  // staleness 0.2
        collector.record_query("seg_c", 100, 80);  // staleness 0.0

        let top = collector.top_stale_segments(3);
        assert_eq!(top[0].0, "seg_b");
        assert!((top[0].1 - 0.2).abs() < 1e-9);
        assert_eq!(top[1].0, "seg_a");
        assert!((top[1].1 - 0.1).abs() < 1e-9);
    }

    #[test]
    fn test_collector_zero_granules_no_record() {
        let collector = QueryFeedbackCollector::new();
        collector.record_query("seg_001", 0, 0); // nothing to record
        assert!(collector.get_segment_stats("seg_001").is_none());
    }

    #[test]
    fn test_collector_get_all_stats() {
        let collector = QueryFeedbackCollector::new();
        collector.record_query("seg_001", 100, 10);
        collector.record_query("seg_002", 100, 90);

        let all = collector.get_stats();
        assert_eq!(all.len(), 2);
        assert!(all.contains_key("seg_001"));
        assert!(all.contains_key("seg_002"));
    }

    // ============================================================
    // Integration: feedback signal must affect compaction priority
    // ============================================================

    #[test]
    fn test_priority_with_feedback_higher_than_without() {
        // Accumulate 5 misses → staleness_penalty approaches 0.5
        let collector = QueryFeedbackCollector::new();
        for _ in 0..5 {
            collector.record_query("seg_001", 100, 5); // big miss
        }
        assert!(collector.staleness_penalty("seg_001") > 0.3,
            "5 misses should produce significant staleness penalty");

        let scheduler = crate::compaction::scheduler::CompactionScheduler::new();
        let mut meta = crate::segment::meta::SegmentMeta::new(
            "seg_001".into(), "t".into(),
            vec![crate::segment::meta::ColumnDef::new("c".into(), crate::segment::meta::DataType::Int64)],
        );
        meta.del_ratio = 0.1;
        meta.uncompressed_size = 10 * 1024 * 1024; // 10MB
        // Use created_at = 0 to get age_hours = 1.0 (avoids log2(0) = -inf)
        meta.created_at = 0;

        let now = crate::codec::current_timestamp_secs();
        let priority_with_feedback = scheduler.calculate_priority_with_feedback(&meta, now, &collector);
        let priority_without_feedback = scheduler.calculate_priority(&meta, now);

        assert!(priority_with_feedback > priority_without_feedback,
            "priority with feedback ({:.2}) should be higher than without ({:.2})",
            priority_with_feedback, priority_without_feedback);
    }

    #[test]
    fn test_feedback_prune_hit_ratio_affects_priority() {
        // Many misses (low prune_hit_ratio) → significant miss penalty in priority score
        let collector = QueryFeedbackCollector::new();
        // est > act*2 → miss. est=100, act=5 → 100 > 10 → miss
        for _ in 0..8 {
            collector.record_query("seg_001", 100, 5); // miss
        }

        let scheduler = crate::compaction::scheduler::CompactionScheduler::new();
        let mut meta = crate::segment::meta::SegmentMeta::new(
            "seg_001".into(), "t".into(),
            vec![crate::segment::meta::ColumnDef::new("c".into(), crate::segment::meta::DataType::Int64)],
        );
        meta.del_ratio = 0.0;
        meta.uncompressed_size = 1024 * 1024;
        meta.created_at = 0; // age_hours = 1.0 → log2(1) = 0

        let now = crate::codec::current_timestamp_secs();
        let priority = scheduler.calculate_priority_with_feedback(&meta, now, &collector);

        // 8 misses → staleness_penalty = min(0.8, 1.0), prune_hit_ratio = 0.0
        // miss_penalty = (1 - 0) * 3.0 = 3.0
        // stale_penalty = 0.8 * 5.0 = 4.0
        assert!(priority > 5.0,
            "low prune_hit_ratio should contribute significant miss penalty, got {:.2}", priority);
    }

    #[test]
    fn test_new_segment_has_no_penalty() {
        // Never-queried segment: default penalty = 0, default prune_hit_ratio = 0.5
        let collector = QueryFeedbackCollector::new();
        let penalty = collector.staleness_penalty("new_seg");
        let ratio = collector.prune_hit_ratio("new_seg");
        assert!((penalty - 0.0).abs() < 1e-9);
        assert!((ratio - 0.5).abs() < 1e-9,
            "no data → neutral prune_hit_ratio of 0.5");
    }

    #[test]
    fn test_feedback_cleared_after_compaction() {
        let collector = QueryFeedbackCollector::new();
        collector.record_query("seg_001", 100, 5);
        assert!(collector.staleness_penalty("seg_001") > 0.0);

        collector.clear_segment("seg_001");
        assert!((collector.staleness_penalty("seg_001") - 0.0).abs() < 1e-9,
            "staleness_penalty must be 0 after clearing segment stats (compaction done)");
    }
}
