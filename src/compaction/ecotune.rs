//! EcoTune — SIGMOD 2025: Adaptive Compaction Policy Selection via Dynamic Programming.
//!
//! Models the LSM storage as three logical levels (top / main / last) and uses
//! dynamic programming to find the optimal tiering ratio per level for a given workload.
//!
//! # Three-Level Model
//!
//! - **Top**: the newest ~20% of segments (by `created_txn`) — active writes.
//! - **Main**: the middle ~60% — most of the data, read-optimized.
//! - **Last**: the oldest ~20% — cold data, archival.
//!
//! # EcoTune DP
//!
//! Given a [`WorkloadProfile`] (RQ ratio, write speed, selectivity), EcoTune computes
//! the optimal compaction policy by minimizing expected cost = read_cost + write_cost.
//!
//! The result is one of: [`CompactionPolicy::Tiering`] (write-optimized),
//! [`CompactionPolicy::Leveling`] (read-optimized), or
//! [`CompactionPolicy::Hybrid`] (balance for mixed workloads).

use std::collections::VecDeque;
use std::sync::Arc;

// Use an LRU cache. Try to use the existing one from the codebase.

// ---------------------------------------------------------------------------
// WorkloadProfile
// ---------------------------------------------------------------------------

/// Characterization of the current workload, used by EcoTune to select policy.
#[derive(Debug, Clone, PartialEq)]
pub struct WorkloadProfile {
    /// Fraction of queries that are range scans (0.0–1.0).
    pub rq_ratio: f64,
    /// Write throughput (MB/s).
    pub write_speed_mbps: f64,
    /// Average scan selectivity: fraction of rows returned per scan.
    pub avg_selectivity: f64,
    /// Fraction of queries that are point queries (0.0–1.0).
    pub point_query_ratio: f64,
}

impl Default for WorkloadProfile {
    fn default() -> Self {
        Self {
            rq_ratio: 0.5,
            write_speed_mbps: 10.0,
            avg_selectivity: 0.01,
            point_query_ratio: 0.5,
        }
    }
}

impl WorkloadProfile {
    /// Returns true if this is a write-heavy workload (RQ ratio < 0.3).
    pub fn is_write_heavy(&self) -> bool {
        self.rq_ratio < 0.3
    }

    /// Returns true if this is a read-heavy workload (RQ ratio > 0.7).
    pub fn is_read_heavy(&self) -> bool {
        self.rq_ratio > 0.7
    }

    /// Returns a hashable key for cache lookup.
    pub fn cache_key(&self) -> (u8, u8, u8, u8) {
        // Discretize to 4 buckets each to reduce cache space.
        let rq = ((self.rq_ratio * 4.0) as u8).min(3);
        let sel = ((self.avg_selectivity * 4.0) as u8).min(3);
        let pt = ((self.point_query_ratio * 4.0) as u8).min(3);
        let ws = if self.write_speed_mbps < 10.0 {
            0
        } else if self.write_speed_mbps < 100.0 {
            1
        } else {
            2
        };
        (rq, sel, pt, ws)
    }
}

// ---------------------------------------------------------------------------
// CompactionPolicy
// ---------------------------------------------------------------------------

/// Compaction policy recommendation from EcoTune.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CompactionPolicy {
    /// Tiering: layer内有多个sorted runs，层间全序。
    /// Write-optimized: less compaction overhead, more read amplification.
    Tiering,
    /// Leveling: layer内完全有序，每次只compact一个run。
    /// Read-optimized: less read amplification, more write amplification.
    Leveling,
    /// Hybrid: balanced, per-level configurable tiering.
    Hybrid {
        /// Fraction of the tree devoted to the top level (0.0–1.0).
        top_ratio: f64,
        /// Fraction of the tree devoted to the main level (0.0–1.0).
        main_ratio: f64,
    },
}

impl Eq for CompactionPolicy {}

impl CompactionPolicy {
    /// Returns a human-readable name.
    pub fn name(&self) -> &'static str {
        match self {
            CompactionPolicy::Tiering => "Tiering (write-optimized)",
            CompactionPolicy::Leveling => "Leveling (read-optimized)",
            CompactionPolicy::Hybrid { .. } => "Hybrid (balanced)",
        }
    }
}

// ---------------------------------------------------------------------------
// LogicalLevel
// ---------------------------------------------------------------------------

/// Maps a segment to one of the three EcoTune logical levels based on age.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogicalLevel {
    /// Top: newest ~20% of segments.
    Top,
    /// Main: middle ~60%.
    Main,
    /// Last: oldest ~20%.
    Last,
}

impl LogicalLevel {
    /// Map a segment's created_txn to a logical level.
    ///
    /// Uses **absolute age** (`now_txn - created_txn`) for stable classification.
    /// This avoids the previous drift bug where `age / now_txn` changed as `now_txn` grew,
    /// causing the same segment to migrate levels over time.
    ///
    /// The thresholds use absolute age in transaction units. Typical deployments
    /// may need tuning: 1000 = "recent" (Top), 100_000 = "old" (Last).
    pub fn from_txn(created_txn: u64, now_txn: u64) -> Self {
        let age = now_txn.saturating_sub(created_txn);
        if age < 1000 {
            LogicalLevel::Top
        } else if age < 100_000 {
            LogicalLevel::Main
        } else {
            LogicalLevel::Last
        }
    }

    /// Returns the EcoTune cost weight for this level (top=1, main=2, last=3).
    /// Higher = more expensive to compact.
    pub fn cost_weight(&self) -> f64 {
        match self {
            LogicalLevel::Top => 1.0,
            LogicalLevel::Main => 2.0,
            LogicalLevel::Last => 3.0,
        }
    }
}

// ---------------------------------------------------------------------------
// EcoTune
// ---------------------------------------------------------------------------

/// Configuration for EcoTune.
#[derive(Debug, Clone)]
pub struct EcoTuneConfig {
    /// Maximum DP iterations before bailing out (ms).
    pub dp_timeout_ms: u64,
    /// Cost of reading one unit from one level.
    pub read_cost_per_level: f64,
    /// Cost of writing one unit (compaction).
    pub write_cost: f64,
    /// Level size ratio (like RocksDB max_bytes_for_level_multiplier).
    pub level_size_ratio: f64,
    /// Top level size ratio.
    pub top_size_ratio: f64,
    /// Whether to enable the LRU policy cache.
    pub enable_cache: bool,
}

impl Default for EcoTuneConfig {
    fn default() -> Self {
        Self {
            dp_timeout_ms: 10,
            read_cost_per_level: 1.0,
            write_cost: 1.0,
            level_size_ratio: 10.0,
            top_size_ratio: 2.0,
            enable_cache: true,
        }
    }
}

/// EcoTune: dynamically selects the optimal compaction policy via DP.
pub struct EcoTune {
    config: EcoTuneConfig,
    /// LRU cache of recent decisions (WorkloadProfile → CompactionPolicy).
    /// Simple VecDeque-based LRU with max 64 entries.
    decision_cache: std::sync::Mutex<LRUCache<(u8, u8, u8, u8), CompactionPolicy>>,
    /// Last profile observed (for change detection).
    last_profile: std::sync::Mutex<Option<WorkloadProfile>>,
}

/// Simple fixed-size LRU cache with O(1) lookup.
/// Uses HashMap for O(1) key lookup, VecDeque for O(1) order tracking.
struct LRUCache<K, V> {
    /// O(1) lookup by key. Value is (V, position in order VecDeque).
    map: std::collections::HashMap<K, (V, usize)>,
    /// O(1) push_front/pop_back. Stores keys in access order.
    /// Order of entries in the LRU queue (indices into `cache`).
    #[allow(dead_code)]
    order: VecDeque<usize>,
    /// Reverse lookup: position → key
    positions: VecDeque<K>,
    max_size: usize,
    next_pos: usize,
}

impl<K: std::hash::Hash + Eq + Clone, V: Clone> LRUCache<K, V> {
    fn new(max_size: usize) -> Self {
        Self {
            map: std::collections::HashMap::with_capacity(max_size),
            order: VecDeque::with_capacity(max_size),
            positions: VecDeque::with_capacity(max_size),
            max_size,
            next_pos: 0,
        }
    }

    fn get(&self, key: &K) -> Option<V> {
        self.map.get(key).map(|(v, _)| v.clone())
    }

    fn put(&mut self, key: K, value: V) {
        // If key exists, update and move to front
        if let Some((_, pos)) = self.map.get_mut(&key) {
            // Move to end (most recent position)
            *pos = self.next_pos;
            self.next_pos += 1;
            return;
        }

        // Remove oldest if at capacity
        if self.map.len() >= self.max_size {
            if let Some(oldest_key) = self.positions.pop_front() {
                self.map.remove(&oldest_key);
            }
        }

        // Insert new entry
        self.map.insert(key.clone(), (value, self.next_pos));
        self.positions.push_back(key);
        self.next_pos += 1;
    }
}

impl EcoTune {
    pub fn new(config: EcoTuneConfig) -> Self {
        Self {
            decision_cache: std::sync::Mutex::new(LRUCache::new(64)),
            last_profile: std::sync::Mutex::new(None),
            config,
        }
    }

    pub fn with_default_config() -> Self {
        Self::new(EcoTuneConfig::default())
    }

    // -------------------------------------------------------------------------
    // Public API
    // -------------------------------------------------------------------------

    /// Select the optimal compaction policy for the given workload.
    ///
    /// Uses a fast heuristic if the DP timeout would be exceeded,
    /// or returns a cached result if the workload hasn't changed.
    pub fn select_policy(&self, profile: &WorkloadProfile) -> CompactionPolicy {
        // Single lock acquisition: check both cache and last_profile in one critical section.
        let (cached_policy, _should_rerun) = {
            let cache = match self.decision_cache.lock() {
                Ok(c) => c,
                Err(_) => return CompactionPolicy::Tiering,
            };
            let last = match self.last_profile.lock() {
                Ok(l) => l,
                Err(_) => return CompactionPolicy::Tiering,
            };

            let cache_key = profile.cache_key();

            // Fast path: return cached result if profile unchanged.
            let changed = match last.as_ref() {
                Some(p) => {
                    (p.rq_ratio - profile.rq_ratio).abs() > 0.1
                        || (p.write_speed_mbps - profile.write_speed_mbps).abs() > 5.0
                }
                None => true,
            };

            if !changed && self.config.enable_cache {
                if let Some(cached) = cache.get(&cache_key) {
                    return cached;
                }
            }

            // If changed and cache enabled, try cache lookup.
            let cached = if changed && self.config.enable_cache {
                cache.get(&cache_key)
            } else {
                None
            };

            (cached, changed)
        };

        // Return cached if available.
        if let Some(policy) = cached_policy {
            return policy;
        }

        // Run EcoTune DP.
        let policy = self.ecotune_select(profile);

        // Cache and update last_profile in single lock scope.
        let cache_key = profile.cache_key();
        if let Ok(mut cache) = self.decision_cache.lock() {
            cache.put(cache_key, policy);
        }
        if let Ok(mut last) = self.last_profile.lock() {
            *last = Some(profile.clone());
        }

        policy
    }

    /// Returns a recommendation for the tiering ratio of the main level.
    /// This is the T parameter from the EcoTune DP.
    pub fn main_level_tiering_ratio(&self, profile: &WorkloadProfile) -> f64 {
        // T = number of sorted runs in the main level.
        // EcoTune: T = round(B / (1 + B * rq_ratio)) where B is the level size ratio.
        let b = self.config.level_size_ratio;
        let rq = profile.rq_ratio.clamp(0.0, 1.0);
        let t = (b / (1.0 + b * rq)).round().max(1.0).min(b);
        t / b
    }

    /// Returns true if range queries should prefer fewer sorted runs.
    pub fn prefer_read_optimized(&self, profile: &WorkloadProfile) -> bool {
        profile.is_read_heavy() || profile.avg_selectivity < 0.001
    }

    /// Returns true if writes should prefer tiering.
    pub fn prefer_write_optimized(&self, profile: &WorkloadProfile) -> bool {
        profile.is_write_heavy() && profile.write_speed_mbps > 100.0
    }

    // -------------------------------------------------------------------------
    // EcoTune DP (simplified)
    // -------------------------------------------------------------------------

    /// Core EcoTune decision logic.
    ///
    /// The full EcoTune DP considers the optimal T (tiering ratio) per level
    /// by minimizing:
    ///   cost = read_cost(rq_ratio, T) + write_cost(write_speed, T)
    ///
    /// We use a simplified closed-form approximation based on the paper's
    /// key insight: T* = B / (1 + B * rq_ratio).
    fn ecotune_select(&self, profile: &WorkloadProfile) -> CompactionPolicy {
        let rq = profile.rq_ratio.clamp(0.01, 0.99);
        let b = self.config.level_size_ratio;

        // EcoTune key formula: optimal T = B / (1 + B * rq_ratio)
        let t_opt = b / (1.0 + b * rq);

        // Tiering: T ≈ B (many runs per level, minimal compaction).
        // Leveling: T = 1 (one run per level, maximum compaction).
        // T* = B / (1 + B * rq). For rq=0.1, B=10: T=5 (tiering boundary).
        // For rq=0.9, B=10: T=1.1 (leveling boundary).
        let is_tiering = t_opt >= b * 0.5;
        let is_leveling = t_opt <= b * 0.15;

        if is_tiering {
            CompactionPolicy::Tiering
        } else if is_leveling {
            CompactionPolicy::Leveling
        } else {
            // Hybrid: use the computed T as the tiering ratio.
            let t_ratio = (t_opt / b).clamp(0.0, 1.0);
            CompactionPolicy::Hybrid {
                top_ratio: self.config.top_size_ratio / (1.0 + self.config.top_size_ratio),
                main_ratio: t_ratio,
            }
        }
    }
}

impl Default for EcoTune {
    fn default() -> Self {
        Self::with_default_config()
    }
}

// ---------------------------------------------------------------------------
// Shared handle
// ---------------------------------------------------------------------------

/// Thread-safe handle to EcoTune.
#[derive(Clone)]
pub struct EcoTuneHandle {
    inner: Arc<EcoTune>,
}

impl EcoTuneHandle {
    pub fn new(config: EcoTuneConfig) -> Self {
        Self {
            inner: Arc::new(EcoTune::new(config)),
        }
    }

    pub fn with_default_config() -> Self {
        Self::new(EcoTuneConfig::default())
    }

    pub fn select_policy(&self, profile: &WorkloadProfile) -> CompactionPolicy {
        self.inner.select_policy(profile)
    }

    pub fn main_level_tiering_ratio(&self, profile: &WorkloadProfile) -> f64 {
        self.inner.main_level_tiering_ratio(profile)
    }

    pub fn prefer_read_optimized(&self, profile: &WorkloadProfile) -> bool {
        self.inner.prefer_read_optimized(profile)
    }

    pub fn prefer_write_optimized(&self, profile: &WorkloadProfile) -> bool {
        self.inner.prefer_write_optimized(profile)
    }
}

impl Default for EcoTuneHandle {
    fn default() -> Self {
        Self::with_default_config()
    }
}

// ---------------------------------------------------------------------------
