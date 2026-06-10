//! Query feedback collector for adaptive compaction
//!
//! Collects query statistics to inform compaction decisions.

use parking_lot::RwLock;
use std::collections::HashMap;

/// Query feedback entry
#[derive(Debug, Clone)]
pub struct QueryFeedback {
    /// Segment ID
    pub seg_id: String,
    /// Hit count (how many times this segment was accessed)
    pub hit_count: u64,
    /// Miss count (accesses that needed to fall back to other segments)
    pub miss_count: u64,
    /// Last access time
    pub last_access: u64,
    /// Query count
    pub query_count: u64,
}

impl QueryFeedback {
    /// Calculate hit ratio.
    ///
    /// # Semantics
    ///
    /// When there is no query history (hit_count + miss_count == 0), returns 1.0.
    /// This is an **optimistic assumption**: new segments with no feedback are
    /// assumed to have perfect hit ratio (0 staleness), so they won't be
    /// prioritized for compaction based on staleness.
    ///
    /// This design choice means:
    /// - New segments won't be penalized for missing feedback data
    /// - Compaction focus remains on segments with known staleness issues
    /// - The trade-off is that truly new/hot data won't be compacted early
    ///
    /// If you need explicit "unknown" handling, consider using `Option<f64>`.
    pub fn hit_ratio(&self) -> f64 {
        let total = self.hit_count + self.miss_count;
        if total == 0 {
            // Optimistic assumption: no history = perfect hit ratio.
            1.0
        } else {
            self.hit_count as f64 / total as f64
        }
    }

    /// Calculate staleness penalty: 1.0 - hit_ratio.
    ///
    /// When hit_ratio() returns 1.0 (no history), staleness_penalty is 0.0.
    /// This means new segments are considered "not stale" until proven otherwise.
    pub fn staleness_penalty(&self) -> f64 {
        1.0 - self.hit_ratio()
    }

    /// Calculate prune hit ratio.
    ///
    /// This is equivalent to hit_ratio() for zone map pruning purposes.
    /// Same optimistic assumption applies when no history is available.
    pub fn prune_hit_ratio(&self) -> f64 {
        self.hit_ratio()
    }
}

/// Query feedback collector
pub struct QueryFeedbackCollector {
    /// Feedback data per segment
    data: RwLock<HashMap<String, QueryFeedback>>,
}

impl QueryFeedbackCollector {
    /// Create a new collector
    pub fn new() -> Self {
        Self {
            data: RwLock::new(HashMap::new()),
        }
    }

    /// Record a segment hit
    pub fn record_hit(&self, seg_id: &str, timestamp: u64) {
        let mut data = self.data.write();
        let feedback = data
            .entry(seg_id.to_string())
            .or_insert_with(|| QueryFeedback {
                seg_id: seg_id.to_string(),
                hit_count: 0,
                miss_count: 0,
                last_access: 0,
                query_count: 0,
            });
        feedback.hit_count += 1;
        feedback.query_count += 1;
        feedback.last_access = timestamp;
    }

    /// Record a segment miss
    pub fn record_miss(&self, seg_id: &str, timestamp: u64) {
        let mut data = self.data.write();
        let feedback = data
            .entry(seg_id.to_string())
            .or_insert_with(|| QueryFeedback {
                seg_id: seg_id.to_string(),
                hit_count: 0,
                miss_count: 0,
                last_access: 0,
                query_count: 0,
            });
        feedback.miss_count += 1;
        feedback.query_count += 1;
        feedback.last_access = timestamp;
    }

    /// Seed feedback for a rewritten segment from its predecessor.
    ///
    /// This keeps post-compaction signals from falling back to the synthetic
    /// no-data defaults on the first scheduling pass after a rewrite.
    ///
    /// # Design Notes
    ///
    /// - `hit_count` and `miss_count` are inherited: these represent the access
    ///   frequency pattern of the old segment, which is valuable signal for the new one.
    /// - `query_count` is reset to 0: this ensures the new segment enters the
    ///   observation period, preventing premature scheduling decisions based on
    ///   inherited query history.
    pub fn alias_feedback(&self, old_seg_id: &str, new_seg_id: &str, timestamp: u64) {
        let mut data = self.data.write();
        let seeded = data
            .get(old_seg_id)
            .cloned()
            .unwrap_or_else(|| QueryFeedback {
                seg_id: old_seg_id.to_string(),
                hit_count: 0,
                miss_count: 0,
                last_access: timestamp,
                query_count: 0,
            });
        data.insert(
            new_seg_id.to_string(),
            QueryFeedback {
                seg_id: new_seg_id.to_string(),
                last_access: timestamp,
                // Inherit access pattern: hot/miss ratio is meaningful for the new segment.
                hit_count: seeded.hit_count,
                miss_count: seeded.miss_count,
                // Reset query count: new segment should accumulate fresh query history.
                query_count: 0,
            },
        );
    }

    /// Get feedback for a segment
    pub fn get_feedback(&self, seg_id: &str) -> Option<QueryFeedback> {
        self.data.read().get(seg_id).cloned()
    }

    /// Get all feedback entries
    pub fn get_all(&self) -> Vec<QueryFeedback> {
        self.data.read().values().cloned().collect()
    }

    /// Get all feedback entries as a map
    pub fn get_all_stats(&self) -> Vec<(String, QueryFeedback)> {
        self.data
            .read()
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }

    /// Get staleness penalty for a segment.
    ///
    /// Returns 0.0 if no feedback exists for this segment.
    /// This is the optimistic assumption: unknown segments are not stale.
    pub fn staleness_penalty(&self, seg_id: &str) -> f64 {
        self.get_feedback(seg_id)
            .map(|f| f.staleness_penalty())
            .unwrap_or(0.0)
    }

    /// Get prune hit ratio for a segment.
    ///
    /// Returns 1.0 if no feedback exists for this segment.
    /// This is the optimistic assumption: unknown segments have perfect pruning.
    pub fn prune_hit_ratio(&self, seg_id: &str) -> f64 {
        self.get_feedback(seg_id)
            .map(|f| f.prune_hit_ratio())
            .unwrap_or(1.0)
    }

    /// Clear all feedback
    pub fn clear(&self) {
        self.data.write().clear();
    }

    /// Prune entries for segments that no longer exist (e.g., after compaction).
    /// Call this after compaction merges away old segments.
    pub fn prune_for_segments(&self, alive_seg_ids: &std::collections::HashSet<String>) {
        let mut data = self.data.write();
        data.retain(|seg_id, _| alive_seg_ids.contains(seg_id));
    }

    /// Prune entries older than `max_age_txn` transactions.
    /// Call this periodically to bound memory growth.
    pub fn prune_stale(&self, now_txn: u64, max_age: u64) {
        let mut data = self.data.write();
        data.retain(|_, fb| now_txn.saturating_sub(fb.last_access) < max_age);
    }

    /// Number of entries currently tracked.
    pub fn len(&self) -> usize {
        self.data.read().len()
    }

    /// Returns true if no entries are tracked.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Average selectivity across all tracked segments.
    /// Returns 0.01 as a sensible default when no data is available.
    pub fn avg_selectivity(&self) -> f64 {
        let data = self.data.read();
        if data.is_empty() {
            return 0.01;
        }
        let total_selectivity: f64 = data
            .values()
            .map(|f| {
                // Infer selectivity from hit ratio: high hit ratio → high selectivity (more rows returned).
                // point_query_ratio estimation: hit_count / query_count as proxy.
                let hit_ratio = if f.query_count > 0 {
                    f.hit_count as f64 / f.query_count as f64
                } else {
                    0.5
                };
                // Rough mapping: hit_ratio ~ selectivity. A query that hits many rows = high selectivity.
                // Conservative estimate: selectivity ≈ min(hit_ratio, 0.5).
                (hit_ratio * 0.5).clamp(0.0001, 1.0)
            })
            .sum();
        total_selectivity / data.len() as f64
    }
}

/// Shared handle for query feedback collector.
/// Uses Arc so multiple components can share the same collector without cloning internal state.
#[derive(Clone)]
pub struct FeedbackHandle {
    inner: std::sync::Arc<QueryFeedbackCollector>,
}

impl FeedbackHandle {
    pub fn new() -> Self {
        Self {
            inner: std::sync::Arc::new(QueryFeedbackCollector::new()),
        }
    }

    pub fn record_hit(&self, seg_id: &str, timestamp: u64) {
        self.inner.record_hit(seg_id, timestamp);
    }

    pub fn record_miss(&self, seg_id: &str, timestamp: u64) {
        self.inner.record_miss(seg_id, timestamp);
    }

    pub fn get_feedback(&self, seg_id: &str) -> Option<QueryFeedback> {
        self.inner.get_feedback(seg_id)
    }

    pub fn alias_feedback(&self, old_seg_id: &str, new_seg_id: &str, timestamp: u64) {
        self.inner.alias_feedback(old_seg_id, new_seg_id, timestamp);
    }

    pub fn get_all(&self) -> Vec<QueryFeedback> {
        self.inner.get_all()
    }

    pub fn staleness_penalty(&self, seg_id: &str) -> f64 {
        self.inner.staleness_penalty(seg_id)
    }

    pub fn prune_hit_ratio(&self, seg_id: &str) -> f64 {
        self.inner.prune_hit_ratio(seg_id)
    }

    pub fn prune_for_segments(&self, alive: &std::collections::HashSet<String>) {
        self.inner.prune_for_segments(alive);
    }

    pub fn prune_stale(&self, now_txn: u64, max_age: u64) {
        self.inner.prune_stale(now_txn, max_age);
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns true if no entries are tracked.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn get_all_stats(&self) -> Vec<(String, QueryFeedback)> {
        self.inner.get_all_stats()
    }

    pub fn avg_selectivity(&self) -> f64 {
        self.inner.avg_selectivity()
    }
}

impl Default for FeedbackHandle {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for QueryFeedbackCollector {
    fn default() -> Self {
        Self::new()
    }
}
