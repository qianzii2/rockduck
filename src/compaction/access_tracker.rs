//! AccessTracker — RangeReduce (ICDE 2026) query-driven compaction.
//!
//! Tracks which granules of which segments have been accessed by recent range scans.
//! Uses granule-level (not row-level) tracking to bound memory overhead while
//! providing enough precision to identify hot data regions.
//!
//! # Algorithm
//!
//! 1. [`mark_access`] records that a scan read `row_count` rows starting at `start_row`
//!    within a segment. Accesses are bucketed into granules (default: 8192 rows).
//! 2. Granule access counts are accumulated in a sparse `BTreeMap` — only accessed
//!    granules are stored, so cold granules cost zero memory.
//! 3. Segment-level hot scores are maintained as an EWMA of granule accesses.
//! 4. [`should_range_reduce`] evaluates two conditions:
//!    - **Overlap**: recent scans on the segment have high key-range overlap
//!      (the same data is being re-read repeatedly)
//!    - **Hot score**: the segment's EWMA hot score exceeds the configured threshold
//! 5. [`select_targets`] returns the top-N segments that satisfy both conditions,
//!    sorted by combined priority score.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::Arc;

use parking_lot::RwLock;

/// Default granule size (rows per granule). 8192 matches ClickHouse's default.
/// Made configurable so callers can tune for their workload.
pub const DEFAULT_GRANULE_SIZE: u32 = 8192;

/// Default number of granule accesses before a granule is considered "hot".
pub const DEFAULT_HOT_GRANULE_ACCESS_COUNT: u16 = 3;

/// Default hot score EWMA threshold (0.0–1.0).
pub const DEFAULT_HOT_SCORE_THRESHOLD: f64 = 0.3;

/// Default overlap ratio threshold (fraction of recent scans that overlap).
pub const DEFAULT_OVERLAP_THRESHOLD: f64 = 0.3;

/// Default maximum number of segments to track simultaneously.
pub const DEFAULT_MAX_TRACKED_SEGMENTS: usize = 1000;

/// Default maximum recent scan ranges to retain per segment for overlap detection.
pub const DEFAULT_MAX_SCAN_RANGES_PER_SEG: usize = 64;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Configuration for the AccessTracker.
#[derive(Debug, Clone)]
pub struct Config {
    /// Rows per granule. Smaller = more precise, more memory.
    pub granule_size: u32,
    /// Minimum accesses to a granule before it contributes to the hot score.
    pub hot_granule_access_count: u16,
    /// EWMA decay factor for segment hot scores (0.875 as in TRIAD).
    pub hot_score_decay: f64,
    /// Hot score threshold — segment must exceed this to be considered hot.
    pub hot_score_threshold: f64,
    /// Overlap ratio threshold — recent scans must exceed this to trigger RangeReduce.
    pub overlap_threshold: f64,
    /// Maximum number of segments tracked simultaneously.
    pub max_tracked_segments: usize,
    /// Maximum scan ranges retained per segment for overlap detection.
    pub max_scan_ranges_per_seg: usize,
    /// EWMA decay for scan range overlap score (0.9).
    pub overlap_decay: f64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            granule_size: DEFAULT_GRANULE_SIZE,
            hot_granule_access_count: DEFAULT_HOT_GRANULE_ACCESS_COUNT,
            hot_score_decay: 0.875,
            hot_score_threshold: DEFAULT_HOT_SCORE_THRESHOLD,
            overlap_threshold: DEFAULT_OVERLAP_THRESHOLD,
            max_tracked_segments: DEFAULT_MAX_TRACKED_SEGMENTS,
            max_scan_ranges_per_seg: DEFAULT_MAX_SCAN_RANGES_PER_SEG,
            overlap_decay: 0.9,
        }
    }
}

// ---------------------------------------------------------------------------
// ScanRange
// ---------------------------------------------------------------------------

/// A key range covered by a single scan operation.
#[derive(Debug, Clone)]
pub struct ScanRange {
    /// Start key (inclusive).
    pub start_key: Vec<u8>,
    /// End key (exclusive).
    pub end_key: Vec<u8>,
}

impl ScanRange {
    /// Returns true if this range overlaps with another.
    pub fn overlaps(&self, other: &ScanRange) -> bool {
        // Empty ranges never overlap
        if self.start_key >= self.end_key || other.start_key >= other.end_key {
            return false;
        }
        // Two ranges [a,b) and [c,d) overlap iff a < d && c < b
        self.start_key < other.end_key && other.start_key < self.end_key
    }
}

// ---------------------------------------------------------------------------
// Internal types
// ---------------------------------------------------------------------------

/// Granule index → access count. Sparse storage: only accessed granules are present.
type GranuleMap = BTreeMap<u32, u16>;

/// Segment tracking state.
#[derive(Debug, Clone)]
struct SegmentState {
    /// Granule → access count (sparse).
    granule_map: GranuleMap,
    /// EWMA hot score for this segment (0.0–1.0).
    hot_score: f64,
    /// Recent scan ranges for overlap detection (bounded).
    scan_ranges: VecDeque<ScanRange>,
    /// EWMA overlap score (0.0–1.0).
    overlap_score: f64,
    /// Last time the segment was updated (for LRU eviction).
    last_update_ms: u64,
}

impl Default for SegmentState {
    fn default() -> Self {
        Self {
            granule_map: GranuleMap::new(),
            hot_score: 0.0,
            scan_ranges: VecDeque::new(),
            overlap_score: 0.0,
            last_update_ms: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// AccessTracker
// ---------------------------------------------------------------------------

/// Tracks scan access patterns for RangeReduce (ICDE 2026) query-driven compaction.
///
/// # Memory Model
///
/// - Per-segment: `BTreeMap<granule_idx, u16>` — sparse, zero for cold granules.
/// - At 1000 segments × avg 10 hot granules × 8 bytes ≈ 80 KB base overhead.
/// - Scan ranges bounded to 64 per segment, keys are copied on access.
pub struct AccessTracker {
    config: Config,
    /// seg_id → SegmentState
    segments: RwLock<HashMap<String, SegmentState>>,
    /// Auxiliary LRU index: (last_update_ms, seg_id) → seg_id, sorted by last_update_ms.
    /// Enables O(1) LRU eviction instead of O(n) HashMap scan.
    lru_index: RwLock<BTreeMap<(u64, String), String>>,
    /// Ring buffer of recent global scan ranges (for coarse-grained overlap estimation).
    recent_global_scans: RwLock<VecDeque<ScanRange>>,
    /// Statistics
    stats: RwLock<TrackerStats>,
}

#[derive(Debug, Default, Clone)]
pub struct TrackerStats {
    /// Total calls to mark_access.
    pub total_access_calls: u64,
    /// Total calls to select_targets.
    pub total_select_calls: u64,
    /// Number of segments currently tracked.
    pub segments_tracked: usize,
    /// Number of RangeReduce candidates identified.
    pub candidates_found: usize,
}

impl AccessTracker {
    /// Create a new AccessTracker with default config.
    pub fn new() -> Self {
        Self::with_config(Config::default())
    }

    /// Create a new AccessTracker with custom config.
    pub fn with_config(config: Config) -> Self {
        Self {
            config,
            segments: RwLock::new(HashMap::new()),
            lru_index: RwLock::new(BTreeMap::new()),
            recent_global_scans: RwLock::new(VecDeque::new()),
            stats: RwLock::new(TrackerStats::default()),
        }
    }

    // -------------------------------------------------------------------------
    // Public API
    // -------------------------------------------------------------------------

    /// Record that a scan accessed rows in a segment.
    ///
    /// `start_row` is the row offset within the segment (0-based).
    /// `row_count` is the number of rows read.
    ///
    /// This is the primary method called by scan paths to report access.
    pub fn mark_access(&self, seg_id: &str, start_row: u32, row_count: u32, now_ms: u64) {
        if row_count == 0 || seg_id.is_empty() {
            return;
        }

        let granule_size = self.config.granule_size;
        let start_g = start_row / granule_size;
        let end_g = (start_row + row_count - 1) / granule_size;

        let mut segments = self.segments.write();
        let mut lru_index = self.lru_index.write();

        // LRU eviction: if we're at capacity, drop the least-recently used segment.
        // Use O(1) BTreeMap lookup instead of O(n) HashMap scan.
        if segments.len() >= self.config.max_tracked_segments {
            if let Some((_key, lru_key)) = lru_index.pop_first() {
                segments.remove(&lru_key);
            }
        }

        let state = segments.entry(seg_id.to_string()).or_default();
        let seg_id_owned = seg_id.to_string();
        
        // Update LRU index: remove old entry and insert new one.
        if state.last_update_ms > 0 {
            lru_index.remove(&(state.last_update_ms, seg_id_owned.clone()));
        }
        state.last_update_ms = now_ms;
        lru_index.insert((now_ms, seg_id_owned.clone()), seg_id_owned);

        // Increment granule access counts (saturate at u16::MAX).
        for g in start_g..=end_g {
            state
                .granule_map
                .entry(g)
                .and_modify(|c| *c = c.saturating_add(1))
                .or_insert(1);
        }

        // Update segment hot score via EWMA.
        let granule_count = (end_g - start_g + 1) as f64;
        let hot_increment = (granule_count * 0.1).min(1.0);
        state.hot_score = state.hot_score * self.config.hot_score_decay
            + hot_increment * (1.0 - self.config.hot_score_decay);
        state.hot_score = state.hot_score.min(1.0);

        // Record this scan range for overlap detection.
        // The caller passes key range info via set_scan_range before calling mark_access,
        // but we also accept a zero-length range for callers that don't have key info.
        // Overlap detection is best-effort; missing key info just means no overlap boost.

        let mut stats = self.stats.write();
        stats.total_access_calls += 1;
        stats.segments_tracked = segments.len();
    }

    /// Record the key range of a scan on a segment. Call this before `mark_access`
    /// when you have the actual key range, for more accurate overlap detection.
    ///
    /// If the segment doesn't exist yet (e.g., only scan range known, no data accessed),
    /// this creates it so overlap tracking starts immediately.
    pub fn record_scan_range(&self, seg_id: &str, range: ScanRange, now_ms: u64) {
        let mut segments = self.segments.write();
        let mut lru_index = self.lru_index.write();
        let state = segments.entry(seg_id.to_string()).or_default();
        let seg_id_owned = seg_id.to_string();

        // Update LRU index: remove old entry and insert new one.
        if state.last_update_ms > 0 {
            lru_index.remove(&(state.last_update_ms, seg_id_owned.clone()));
        }
        state.last_update_ms = now_ms;
        lru_index.insert((now_ms, seg_id_owned.clone()), seg_id_owned);

        // Maintain bounded scan range list (FIFO eviction).
        if state.scan_ranges.len() >= self.config.max_scan_ranges_per_seg {
            state.scan_ranges.pop_front();
        }
        state.scan_ranges.push_back(range.clone());

        // Update overlap score: fraction of recent ranges that overlap with the latest.
        if let Some(last) = state.scan_ranges.back() {
            let overlap_count = state
                .scan_ranges
                .iter()
                .filter(|r| r.overlaps(last))
                .count();
            let ratio = overlap_count as f64 / state.scan_ranges.len() as f64;
            state.overlap_score = state.overlap_score * self.config.overlap_decay
                + ratio * (1.0 - self.config.overlap_decay);
        }

        // Also track globally.
        drop(segments);
        drop(lru_index);
        let mut global = self.recent_global_scans.write();
        if global.len() >= 256 {
            global.pop_front();
        }
        global.push_back(range);
    }

    /// Set the key range of the current scan on a segment, to be recorded
    /// alongside the next `mark_access` call.
    /// This is a convenience for scan paths that know the key bounds.
    pub fn set_scan_range(&self, seg_id: &str, start_key: Vec<u8>, end_key: Vec<u8>, now_ms: u64) {
        self.record_scan_range(seg_id, ScanRange { start_key, end_key }, now_ms);
    }

    /// Returns the hot score for a segment (0.0–1.0).
    pub fn hot_score(&self, seg_id: &str) -> f64 {
        self.segments
            .read()
            .get(seg_id)
            .map(|s| s.hot_score)
            .unwrap_or(0.0)
    }

    /// Returns the overlap score for a segment (0.0–1.0).
    pub fn overlap_score(&self, seg_id: &str) -> f64 {
        self.segments
            .read()
            .get(seg_id)
            .map(|s| s.overlap_score)
            .unwrap_or(0.0)
    }

    /// Returns true if a segment should be considered for RangeReduce compaction.
    ///
    /// Both conditions must hold:
    /// - Overlap score > `overlap_threshold` (the segment is being re-read repeatedly)
    /// - Hot score > `hot_score_threshold` (the segment contains frequently accessed data)
    pub fn should_range_reduce(&self, seg_id: &str) -> bool {
        let segments = self.segments.read();
        match segments.get(seg_id) {
            Some(s) => {
                s.overlap_score > self.config.overlap_threshold
                    && s.hot_score > self.config.hot_score_threshold
            }
            None => false,
        }
    }

    /// Returns the combined priority score for a segment.
    /// Used by the scheduler to order RangeReduce tasks.
    pub fn priority_score(&self, seg_id: &str) -> f64 {
        let segments = self.segments.read();
        match segments.get(seg_id) {
            Some(s) => s.hot_score * 0.6 + s.overlap_score * 0.4,
            None => 0.0,
        }
    }

    /// Select the top-N segments that are candidates for RangeReduce compaction,
    /// ordered by combined priority score.
    pub fn select_targets(&self, max_count: usize) -> Vec<(String, f64)> {
        let segments = self.segments.read();

        let mut candidates: Vec<_> = segments
            .iter()
            .filter(|(_seg_id, s)| {
                // Both conditions must hold — checked inline to avoid re-entrant lock.
                s.overlap_score > self.config.overlap_threshold
                    && s.hot_score > self.config.hot_score_threshold
            })
            .map(|(seg_id, s)| {
                let score = s.hot_score * 0.6 + s.overlap_score * 0.4;
                (seg_id.clone(), score)
            })
            .collect();

        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(max_count);

        let mut stats = self.stats.write();
        stats.total_select_calls += 1;
        stats.candidates_found = candidates.len();

        candidates
    }

    /// Returns the number of segments currently being tracked.
    pub fn tracked_count(&self) -> usize {
        self.segments.read().len()
    }

    /// Returns a snapshot of tracker statistics.
    pub fn stats(&self) -> TrackerStats {
        self.stats.read().clone()
    }

    /// Remove tracking state for a segment (call after compaction merges it away).
    pub fn forget_segment(&self, seg_id: &str) {
        let mut segments = self.segments.write();
        let mut lru_index = self.lru_index.write();
        
        // Remove from both HashMap and LRU index.
        if let Some(state) = segments.get(seg_id) {
            lru_index.remove(&(state.last_update_ms, seg_id.to_string()));
        }
        segments.remove(seg_id);
    }

    /// Forget all segments (call when restarting or when stats are stale).
    pub fn clear(&self) {
        self.segments.write().clear();
        self.lru_index.write().clear();
        self.recent_global_scans.write().clear();
    }

    /// Drain all pending state into a Vec (useful for serialization or inspection).
    pub fn drain_state(&self) -> Vec<(String, f64, f64)> {
        self.segments
            .read()
            .iter()
            .map(|(k, s)| (k.clone(), s.hot_score, s.overlap_score))
            .collect()
    }

    /// Returns the granule-level access map for a segment (for testing/debugging).
    #[allow(dead_code)]
    fn granule_map(&self, seg_id: &str) -> BTreeMap<u32, u16> {
        self.segments
            .read()
            .get(seg_id)
            .map(|s| s.granule_map.clone())
            .unwrap_or_default()
    }
}

impl Default for AccessTracker {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Shared handle (Clone-able)
// ---------------------------------------------------------------------------

/// A cloneable handle to an AccessTracker, safe to share across threads.
#[derive(Clone)]
pub struct AccessTrackerHandle {
    inner: Arc<AccessTracker>,
}

impl AccessTrackerHandle {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(AccessTracker::new()),
        }
    }

    pub fn with_config(config: Config) -> Self {
        Self {
            inner: Arc::new(AccessTracker::with_config(config)),
        }
    }

    pub fn mark_access(&self, seg_id: &str, start_row: u32, row_count: u32) {
        let now = crate::codec::current_timestamp_millis();
        self.inner.mark_access(seg_id, start_row, row_count, now);
    }

    pub fn set_scan_range(&self, seg_id: &str, start_key: Vec<u8>, end_key: Vec<u8>) {
        let now = crate::codec::current_timestamp_millis();
        self.inner.set_scan_range(seg_id, start_key, end_key, now);
    }

    pub fn record_scan_range(&self, seg_id: &str, range: ScanRange) {
        let now = crate::codec::current_timestamp_millis();
        self.inner.record_scan_range(seg_id, range, now);
    }

    pub fn hot_score(&self, seg_id: &str) -> f64 {
        self.inner.hot_score(seg_id)
    }

    pub fn overlap_score(&self, seg_id: &str) -> f64 {
        self.inner.overlap_score(seg_id)
    }

    pub fn should_range_reduce(&self, seg_id: &str) -> bool {
        self.inner.should_range_reduce(seg_id)
    }

    pub fn priority_score(&self, seg_id: &str) -> f64 {
        self.inner.priority_score(seg_id)
    }

    pub fn select_targets(&self, max_count: usize) -> Vec<(String, f64)> {
        self.inner.select_targets(max_count)
    }

    pub fn tracked_count(&self) -> usize {
        self.inner.tracked_count()
    }

    pub fn stats(&self) -> TrackerStats {
        self.inner.stats()
    }

    pub fn forget_segment(&self, seg_id: &str) {
        self.inner.forget_segment(seg_id);
    }

    pub fn clear(&self) {
        self.inner.clear();
    }
}

impl Default for AccessTrackerHandle {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
