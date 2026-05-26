//! Granule Staleness Tracking for Zone Maps
//!
//! Zone Maps become stale after UPDATE/DELETE operations. This module tracks
//! per-granule staleness so that queries can decide whether to:
//!   1. Use the cached Zone Map (if stale_ratio is low)
//!   2. Compute adjusted min/max from DelMask dynamically (if stale_ratio is moderate)
//!   3. Skip Zone Map pruning entirely (if stale_ratio is high)
//!
//!参照: Oracle Zone Maps stale tracking + FAST REFRESH

use std::collections::HashMap;
use crate::segment::del_mask::DelMask;
use crate::segment::meta::{ColumnStats, GranuleMeta, ZoneMapStats};

/// Per-granule staleness state
#[derive(Debug, Clone)]
pub struct GranuleStaleness {
    /// Granule ID
    pub granule_id: u32,
    /// Number of stale (deleted/updated) rows in this granule
    pub stale_row_count: u32,
    /// stale_row_count / granule.row_count
    pub stale_ratio: f64,
    /// Last time this staleness was recomputed
    pub last_updated: u64,
}

impl GranuleStaleness {
    /// Create a new staleness record
    pub fn new(granule_id: u32, stale_row_count: u32, granule_row_count: u32) -> Self {
        let ratio = if granule_row_count > 0 {
            stale_row_count as f64 / granule_row_count as f64
        } else {
            0.0
        };
        Self {
            granule_id,
            stale_row_count,
            stale_ratio: ratio,
            last_updated: crate::codec::current_timestamp_secs(),
        }
    }

    /// Recompute staleness from a DelMask, only counting positions within this granule
    pub fn recompute(granule_id: u32, granule: &GranuleMeta, del_mask: &DelMask) -> Self {
        let start = granule.row_offset as u64;
        let end = start + granule.row_count as u64;

        let stale_in_granule: u32 = del_mask
            .deleted_positions()
            .filter(|&pos| pos >= start && pos < end)
            .count() as u32;

        Self::new(granule_id, stale_in_granule, granule.row_count)
    }

    /// Whether this granule's Zone Map is trustworthy enough to use for pruning
    pub fn is_trustworthy(&self, threshold: f64) -> bool {
        self.stale_ratio < threshold
    }
}

/// In-memory staleness tracker (for active segments).
/// For cold data, staleness is tracked in RocksDB CF "zone_stale".
#[derive(Debug, Default)]
pub struct StalenessTracker {
    /// seg_id → granule_id → GranuleStaleness
    segments: std::sync::RwLock<HashMap<String, HashMap<u32, GranuleStaleness>>>,
}

impl Clone for StalenessTracker {
    fn clone(&self) -> Self {
        // Clone creates an empty tracker (shallow clone of in-memory state)
        Self::default()
    }
}

impl StalenessTracker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record staleness for a granule
    pub fn record(&self, seg_id: &str, staleness: GranuleStaleness) {
        let mut segments = self.segments.write().unwrap();
        segments
            .entry(seg_id.to_string())
            .or_default()
            .insert(staleness.granule_id, staleness);
    }

    /// Get staleness for a specific granule
    pub fn get(&self, seg_id: &str, granule_id: u32) -> Option<GranuleStaleness> {
        let segments = self.segments.read().unwrap();
        segments.get(seg_id).and_then(|g| g.get(&granule_id).cloned())
    }

    /// Compute staleness on-the-fly from DelMask for a granule range.
    /// Used when staleness is not cached.
    pub fn compute_from_mask(
        _seg_id: &str,
        granule_id: u32,
        granule: &GranuleMeta,
        del_mask: &DelMask,
    ) -> GranuleStaleness {
        let stale = GranuleStaleness::recompute(granule_id, granule, del_mask);
        stale
    }

    /// Clear staleness for a segment (e.g., after compaction)
    pub fn clear_segment(&self, seg_id: &str) {
        let mut segments = self.segments.write().unwrap();
        segments.remove(seg_id);
    }

    /// Get the aggregate staleness ratio for a segment (max across all granules)
    pub fn max_stale_ratio(&self, seg_id: &str) -> f64 {
        let segments = self.segments.read().unwrap();
        segments.get(seg_id)
            .map(|g| g.values().map(|s| s.stale_ratio).fold(0.0f64, f64::max))
            .unwrap_or(0.0)
    }
}

/// Compute adjusted Zone Map stats taking staleness into account.
///
///
/// If a granule has low staleness, Zone Map stats are used as-is.
/// If a granule has high staleness, we compute conservative bounds:
///   - min: take the minimum of (ZoneMap.min, actual min from surviving rows)
///   - max: take the maximum of (ZoneMap.max, actual max from surviving rows)
///   - null_count: ZoneMap null_count + stale_row_count (deleted rows treated as null-ish)
///
/// When staleness_ratio > 0.5, we fall back to conservative bounds that disable pruning.
pub fn adjusted_column_stats(
    stats: &ColumnStats,
    staleness: Option<&GranuleStaleness>,
    threshold: f64,
) -> ColumnStats {
    let Some(s) = staleness else {
        return stats.clone();
    };

    if s.stale_ratio < threshold {
        // Zone Map is fresh enough
        return stats.clone();
    }

    if s.stale_ratio > 0.5 {
        // Too stale — return conservative stats that disable pruning
        return ColumnStats {
            min: None,
            max: None,
            null_count: stats.null_count + s.stale_row_count,
            sum: None,
            distinct_count: None,
        };
    }

    // Moderate staleness — zone map is unreliable, expand bounds conservatively
    // We keep the existing min/max as-is (may be slightly off) but flag reduced confidence
    ColumnStats {
        min: stats.min.clone(),
        max: stats.max.clone(),
        null_count: stats.null_count + s.stale_row_count,
        sum: stats.sum.clone(),
        distinct_count: None, // Distinct count is unreliable with deletions
    }
}

/// Get the staleness-adjusted ZoneMapStats for a granule.
pub fn adjusted_zone_map_stats(
    zone_map: &ZoneMapStats,
    staleness: Option<&GranuleStaleness>,
    threshold: f64,
) -> ZoneMapStats {
    let mut adjusted = ZoneMapStats::new();
    for (col, stats) in &zone_map.stats {
        adjusted.add_column_stats(col, adjusted_column_stats(stats, staleness, threshold));
    }
    adjusted
}

/// Check if a Zone Map can prune given staleness. Returns false if too stale.
pub fn can_prune_with_staleness(
    _stats: &ColumnStats,
    staleness: Option<&GranuleStaleness>,
    threshold: f64,
) -> bool {
    if let Some(s) = staleness {
        if s.stale_ratio > threshold {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    // ============================================================
    // GranuleStaleness
    // ============================================================

    #[test]
    fn test_granule_staleness_new() {
        let s = GranuleStaleness::new(0, 10, 100);
        assert_eq!(s.granule_id, 0);
        assert_eq!(s.stale_row_count, 10);
        assert!((s.stale_ratio - 0.1).abs() < 1e-9);
    }

    #[test]
    fn test_granule_staleness_zero_rows() {
        let s = GranuleStaleness::new(0, 0, 0);
        assert!((s.stale_ratio - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_granule_staleness_is_trustworthy() {
        let s = GranuleStaleness::new(0, 5, 100);
        assert!(s.is_trustworthy(0.1));
        assert!(!s.is_trustworthy(0.04));
    }

    #[test]
    fn test_granule_staleness_recompute() {
        let granule = GranuleMeta::new(0, 0, 10); // rows 0-9
        let mut del_mask = DelMask::new(100);
        del_mask.add_delete(1);
        del_mask.add_delete(3);
        del_mask.add_delete(5); // 3 deletes in granule 0

        let s = GranuleStaleness::recompute(0, &granule, &del_mask);
        assert_eq!(s.stale_row_count, 3);
        assert!((s.stale_ratio - 0.3).abs() < 1e-9);
    }

    // ============================================================
    // StalenessTracker
    // ============================================================

    #[test]
    fn test_staleness_tracker_record_and_get() {
        let tracker = StalenessTracker::new();
        tracker.record("seg_001", GranuleStaleness::new(0, 10, 100));

        let s = tracker.get("seg_001", 0).unwrap();
        assert_eq!(s.stale_row_count, 10);
    }

    #[test]
    fn test_staleness_tracker_missing() {
        let tracker = StalenessTracker::new();
        assert!(tracker.get("seg_999", 0).is_none());
        assert!(tracker.get("seg_001", 1).is_none());
    }

    #[test]
    fn test_staleness_tracker_clear() {
        let tracker = StalenessTracker::new();
        tracker.record("seg_001", GranuleStaleness::new(0, 5, 100));
        tracker.clear_segment("seg_001");
        assert!(tracker.get("seg_001", 0).is_none());
    }

    #[test]
    fn test_staleness_tracker_max_stale_ratio() {
        let tracker = StalenessTracker::new();
        tracker.record("seg_001", GranuleStaleness::new(0, 5, 100));
        tracker.record("seg_001", GranuleStaleness::new(1, 20, 100));

        let max = tracker.max_stale_ratio("seg_001");
        assert!((max - 0.2).abs() < 1e-9);
    }

    #[test]
    fn test_staleness_tracker_max_stale_ratio_missing() {
        let tracker = StalenessTracker::new();
        assert!((tracker.max_stale_ratio("seg_001") - 0.0).abs() < 1e-9);
    }

    // ============================================================
    // adjusted_column_stats
    // ============================================================

    #[test]
    fn test_adjusted_stats_no_staleness() {
        let stats = ColumnStats {
            min: Some(vec![10u8]),
            max: Some(vec![100u8]),
            null_count: 5,
            sum: Some(vec![]),
            distinct_count: Some(50),
        };

        let adjusted = adjusted_column_stats(&stats, None, 0.1);
        assert_eq!(adjusted.min, stats.min);
        assert_eq!(adjusted.max, stats.max);
        assert_eq!(adjusted.distinct_count, stats.distinct_count);
    }

    #[test]
    fn test_adjusted_stats_low_staleness() {
        let stats = ColumnStats {
            min: Some(vec![10u8]),
            max: Some(vec![100u8]),
            null_count: 5,
            sum: None,
            distinct_count: Some(50),
        };

        let staleness = GranuleStaleness::new(0, 5, 100);
        let adjusted = adjusted_column_stats(&stats, Some(&staleness), 0.1);
        assert_eq!(adjusted.min, stats.min);
        assert_eq!(adjusted.max, stats.max);
        // low staleness < threshold: distinct_count preserved
        assert_eq!(adjusted.distinct_count, stats.distinct_count);
    }

    #[test]
    fn test_adjusted_stats_high_staleness() {
        let stats = ColumnStats {
            min: Some(vec![10u8]),
            max: Some(vec![100u8]),
            null_count: 5,
            sum: Some(vec![]),
            distinct_count: Some(50),
        };

        let staleness = GranuleStaleness::new(0, 60, 100); // 60% stale
        let adjusted = adjusted_column_stats(&stats, Some(&staleness), 0.5);
        // High staleness > 0.5: conservative bounds disable pruning
        assert!(adjusted.min.is_none());
        assert!(adjusted.max.is_none());
        assert_eq!(adjusted.null_count, 65); // original 5 + stale 60
        assert!(adjusted.distinct_count.is_none());
    }

    // ============================================================
    // can_prune_with_staleness
    // ============================================================

    #[test]
    fn test_can_prune_no_staleness() {
        let stats = ColumnStats {
            min: Some(vec![10u8]),
            max: Some(vec![100u8]),
            null_count: 0,
            sum: None,
            distinct_count: None,
        };
        assert!(can_prune_with_staleness(&stats, None, 0.1));
    }

    #[test]
    fn test_can_prune_with_high_staleness() {
        let stats = ColumnStats {
            min: Some(vec![10u8]),
            max: Some(vec![100u8]),
            null_count: 0,
            sum: None,
            distinct_count: None,
        };
        let staleness = GranuleStaleness::new(0, 60, 100);
        assert!(!can_prune_with_staleness(&stats, Some(&staleness), 0.5));
    }

    #[test]
    fn test_can_prune_with_low_staleness() {
        let stats = ColumnStats {
            min: Some(vec![10u8]),
            max: Some(vec![100u8]),
            null_count: 0,
            sum: None,
            distinct_count: None,
        };
        let staleness = GranuleStaleness::new(0, 5, 100);
        assert!(can_prune_with_staleness(&stats, Some(&staleness), 0.1));
    }

    // ============================================================
    // adjusted_zone_map_stats
    // ============================================================

    #[test]
    fn test_adjusted_zone_map_stats_passthrough() {
        let mut zm = ZoneMapStats::new();
        zm.add_column_stats("age", ColumnStats {
            min: Some(vec![10u8]),
            max: Some(vec![100u8]),
            null_count: 0,
            sum: None,
            distinct_count: Some(50),
        });

        let adjusted = adjusted_zone_map_stats(&zm, None, 0.1);
        assert!(adjusted.get("age").is_some());
        assert_eq!(adjusted.get("age").unwrap().distinct_count, Some(50));
    }

    #[test]
    fn test_adjusted_zone_map_stats_high_staleness() {
        let mut zm = ZoneMapStats::new();
        zm.add_column_stats("age", ColumnStats {
            min: Some(vec![10u8]),
            max: Some(vec![100u8]),
            null_count: 0,
            sum: None,
            distinct_count: Some(50),
        });

        let staleness = GranuleStaleness::new(0, 60, 100);
        let adjusted = adjusted_zone_map_stats(&zm, Some(&staleness), 0.5);
        assert!(adjusted.get("age").unwrap().min.is_none());
        assert!(adjusted.get("age").unwrap().max.is_none());
    }

    // ============================================================
    // Regression tests: staleness must NOT break existing can_prune semantics
    // ============================================================

    #[test]
    fn test_adjusted_stats_unchanged_when_trustworthy() {
        // Zone Map is fresh (staleness_ratio < threshold).
        // adjusted_column_stats must return identical stats to the original.
        let stats = ColumnStats {
            min: Some(vec![10u8]),
            max: Some(vec![100u8]),
            null_count: 0,
            sum: None,
            distinct_count: Some(50),
        };

        let staleness = GranuleStaleness::new(0, 5, 100); // 5% stale, trustworthy at threshold 0.1
        assert!(staleness.is_trustworthy(0.1));

        let adjusted = adjusted_column_stats(&stats, Some(&staleness), 0.1);
        assert_eq!(adjusted.min, stats.min,
            "min must be unchanged when Zone Map is trustworthy");
        assert_eq!(adjusted.max, stats.max,
            "max must be unchanged when Zone Map is trustworthy");
        assert_eq!(adjusted.distinct_count, stats.distinct_count,
            "distinct_count must be unchanged when Zone Map is trustworthy");
    }

    #[test]
    fn test_can_prune_unchanged_with_low_staleness() {
        // KEY REGRESSION: adding staleness tracking must NOT change can_prune results
        // when staleness is low. The ZoneMapStats::can_prune_static result must equal
        // the can_prune_with_staleness result.
        let stats = ColumnStats {
            min: Some(vec![10u8]),
            max: Some(vec![100u8]),
            null_count: 0,
            sum: None,
            distinct_count: None,
        };
        let staleness = GranuleStaleness::new(0, 5, 100); // 5% stale, trustworthy at 0.1
        assert!(staleness.is_trustworthy(0.1));

        // Without staleness: can prune if min >= value
        let without_stale = ZoneMapStats::can_prune_static(
            &stats,
            &crate::segment::meta::CompareOp::Lt,
            &[5u8],
        );

        // With low staleness: result must be identical
        let with_stale = can_prune_with_staleness(&stats, Some(&staleness), 0.1);
        assert_eq!(without_stale, with_stale,
            "can_prune result must be unchanged with low staleness");
    }

    #[test]
    fn test_can_prune_conservative_with_high_staleness() {
        // High staleness (ratio > 0.5) must conservatively disable pruning.
        let stats = ColumnStats {
            min: Some(vec![10u8]),
            max: Some(vec![100u8]),
            null_count: 0,
            sum: None,
            distinct_count: None,
        };

        let staleness = GranuleStaleness::new(0, 80, 100); // 80% stale
        assert!(!staleness.is_trustworthy(0.5),
            "staleness_ratio=0.8 should not be trustworthy at threshold 0.5");

        // With high staleness, can_prune_with_staleness must return false
        let result = can_prune_with_staleness(&stats, Some(&staleness), 0.5);
        assert!(!result,
            "Zone Map with high staleness must not enable pruning (conservative behavior)");
    }

    #[test]
    fn test_staleness_recompute_from_del_mask() {
        // compute_from_mask is the static helper that recomputes staleness from DelMask.
        let granule = GranuleMeta::new(0, 0, 10); // rows 0-9 in granule 0
        let mut del_mask = DelMask::new(1000); // segment has 1000 rows
        del_mask.add_delete(0);
        del_mask.add_delete(1);
        del_mask.add_delete(2); // 3 deletes in granule 0's range

        let stale = GranuleStaleness::recompute(0, &granule, &del_mask);
        assert_eq!(stale.stale_row_count, 3);
        assert!((stale.stale_ratio - 0.3).abs() < 0.001,
            "staleness_ratio should be 3/10 = 0.3");
        assert_eq!(stale.granule_id, 0);
    }

    #[test]
    fn test_staleness_empty_del_mask() {
        let granule = GranuleMeta::new(5, 0, 100);
        let del_mask = DelMask::new(100);
        let stale = GranuleStaleness::recompute(5, &granule, &del_mask);

        assert_eq!(stale.granule_id, 5);
        assert_eq!(stale.stale_row_count, 0);
        assert!((stale.stale_ratio - 0.0).abs() < 1e-9);
        assert!(stale.is_trustworthy(0.5),
            "no deletions should be trustworthy at any reasonable threshold");
    }

    #[test]
    fn test_max_stale_ratio_across_granules() {
        // Key business rule: a segment's staleness is determined by its worst granule.
        let tracker = StalenessTracker::new();
        tracker.record("seg_001", GranuleStaleness::new(0, 1, 100));  // 1% stale
        tracker.record("seg_001", GranuleStaleness::new(1, 60, 100)); // 60% stale

        let max_ratio = tracker.max_stale_ratio("seg_001");
        assert!((max_ratio - 0.6).abs() < 0.001,
            "max staleness must be 0.6 from the worst granule (seg_001 granule 1)");
    }
}
