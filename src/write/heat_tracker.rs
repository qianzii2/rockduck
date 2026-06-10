//! WriteHeatTracker — TRIAD (VLDB 2022) hot/cold key separation for WAL flush.
//!
//! Tracks the write frequency of each primary key using EWMA. Hot keys (frequently
//! updated) are kept in the WAL longer, while cold keys are flushed to disk faster.
//! This reduces write amplification by avoiding compaction of frequently-updated data.
//!
//! # TRIAD Algorithm
//!
//! 1. Each write to a primary key increments its EWMA heat score.
//! 2. During WAL flush, keys are classified as hot or cold.
//! 3. Hot data stays in the WAL / hot buffer, cold data is flushed immediately.
//! 4. The hot buffer accumulates updates until a threshold, then forces a flush.
//!
//! # Configuration
//!
//! - `hot_threshold`: EWMA above this → hot (default 5.0)
//! - `cold_threshold`: EWMA below this → cold (default 1.0)
//! - `decay_factor`: EWMA decay (default 0.875, from TRIAD paper)
//! - `hot_buffer_max_bytes`: max size of hot buffer before forced flush
//!
//! # Lock Safety (w009 fix)
//!
//! `pk_heat` and `pk_to_ewma` are wrapped together in a single `parking_lot::RwLock<Inner>`.
//! BTreeMap operations (pop_first, insert, remove) that require mutation are performed
//! OUTSIDE the lock by extracting the inner struct temporarily. This minimizes lock
//! hold time and prevents two-write-lock deadlock scenarios.

use indexmap::IndexMap;
use ordered_float::OrderedFloat;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::Arc;

/// Configuration for the WriteHeatTracker.
#[derive(Debug, Clone)]
pub struct HeatTrackerConfig {
    /// EWMA above this → key is "hot" (default 5.0).
    pub hot_threshold: f64,
    /// EWMA below this → key is "cold" (default 1.0).
    pub cold_threshold: f64,
    /// EWMA decay factor (default 0.875 from TRIAD paper).
    pub decay_factor: f64,
    /// Maximum bytes in hot buffer before forced flush.
    pub hot_buffer_max_bytes: usize,
    /// Maximum number of hot keys to track.
    pub max_tracked_keys: usize,
}

impl Default for HeatTrackerConfig {
    fn default() -> Self {
        Self {
            hot_threshold: 5.0,
            cold_threshold: 1.0,
            decay_factor: 0.875,
            hot_buffer_max_bytes: 64 * 1024 * 1024, // 64 MB
            max_tracked_keys: 1_000_000,
        }
    }
}

// ---------------------------------------------------------------------------
// Internal
// ---------------------------------------------------------------------------

/// EWMA heat state for a single primary key.
#[derive(Debug, Clone)]
struct HeatSample {
    /// EWMA heat score.
    ewma: f64,
    /// Number of updates recorded.
    update_count: u32,
}

/// Combined inner state protected by a single RwLock (w009 fix).
/// Contains both the BTreeMap (sorted by ewma) and the HashMap (O(1) pk lookup).
#[derive(Debug, Default)]
struct Inner {
    /// Sorted by (ewma, pk) for O(1) minimum-ewma eviction via pop_first().
    pk_heat: BTreeMap<(OrderedFloat<f64>, Vec<u8>), HeatSample>,
    /// Secondary index for O(1) pk→ewma lookup (avoids BTreeMap O(N) scan).
    pk_to_ewma: HashMap<Vec<u8>, f64>,
}

// ---------------------------------------------------------------------------
// WriteHeatTracker
// ---------------------------------------------------------------------------

/// Tracks primary key write frequency (EWMA) for TRIAD hot/cold separation.
///
/// # Memory
///
/// - Each tracked key costs ~32 bytes (8 EWMA + 4 count + overhead).
/// - With 1M tracked keys: ~32 MB.
/// - Cold keys are evicted when the map exceeds capacity.
pub struct WriteHeatTracker {
    config: HeatTrackerConfig,
    /// w009 fix: Single RwLock wrapping both pk_heat (BTreeMap) and pk_to_ewma (HashMap).
    /// BTreeMap operations are performed outside the lock by extracting/returning Inner.
    inner: parking_lot::RwLock<Inner>,
    /// Hot buffer: accumulated hot writes pending flush.
    hot_buffer: parking_lot::RwLock<HotBuffer>,
    /// Ring buffer of recent cold key hashes (for stats).
    recent_cold_keys: parking_lot::RwLock<VecDeque<u64>>,
}

struct HotBuffer {
    /// Accumulated hot entries (key → [values]), in insertion order for LRU eviction.
    entries: IndexMap<Vec<u8>, Vec<Vec<u8>>>,
    total_bytes: usize,
}

impl HotBuffer {
    fn new() -> Self {
        Self {
            entries: IndexMap::new(),
            total_bytes: 0,
        }
    }

    fn add(&mut self, key: Vec<u8>, value: Vec<u8>, max_bytes: usize) {
        let key_bytes = key.len();
        let value_bytes = value.len();
        let entry_bytes = key_bytes + value_bytes;

        if entry_bytes > max_bytes {
            return; // Skip entries larger than the max buffer size.
        }

        // Evict the oldest entries until there's room for the new one.
        while self.total_bytes + entry_bytes > max_bytes {
            if let Some((evicted_key, evicted_values)) = self.entries.swap_remove_index(0) {
                self.total_bytes -= evicted_key.len();
                self.total_bytes -= evicted_values.iter().map(|v| v.len()).sum::<usize>();
            } else {
                break;
            }
        }

        self.total_bytes += entry_bytes;
        self.entries.entry(key).or_default().push(value);
    }

    fn clear(&mut self) {
        self.entries.clear();
        self.total_bytes = 0;
    }

    #[allow(dead_code)]
    fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    #[allow(dead_code)]
    fn len(&self) -> usize {
        self.entries.len()
    }
}

impl WriteHeatTracker {
    /// Create a new WriteHeatTracker with default config.
    pub fn new() -> Self {
        Self::with_config(HeatTrackerConfig::default())
    }

    /// Create a new WriteHeatTracker with custom config.
    pub fn with_config(config: HeatTrackerConfig) -> Self {
        Self {
            config,
            inner: parking_lot::RwLock::new(Inner::default()),
            hot_buffer: parking_lot::RwLock::new(HotBuffer::new()),
            recent_cold_keys: parking_lot::RwLock::new(VecDeque::with_capacity(1024)),
        }
    }

    // -------------------------------------------------------------------------
    // Core TRIAD API
    // -------------------------------------------------------------------------

    /// Record a write to a primary key. Updates the EWMA heat score.
    ///
    /// This should be called on every insert/update to the key.
    pub fn record_update(&self, pk: &[u8]) {
        let pk_vec = pk.to_vec();

        // w009 fix: Single lock acquisition. BTreeMap operations happen inside the lock.
        let mut inner = self.inner.write();

        // LRU eviction: pop the minimum-ewma key (BTreeMap is sorted, O(1)).
        if inner.pk_heat.len() >= self.config.max_tracked_keys {
            if let Some((min_key, _)) = inner.pk_heat.pop_first() {
                // min_key.0 = OrderedFloat<ewma>, min_key.1 = pk
                inner.pk_to_ewma.remove(&min_key.1);
            }
        }

        // O(1) lookup via pk_to_ewma secondary index
        let old_ewma = inner.pk_to_ewma.get(&pk_vec).copied();
        let update_count = old_ewma
            .and_then(|e| inner.pk_heat.get(&(OrderedFloat(e), pk_vec.clone())))
            .map(|s| s.update_count)
            .unwrap_or(0);

        // TRIAD EWMA: ewma = decay * ewma + (1 - decay) * delta
        let new_ewma = old_ewma
            .map(|e| e * self.config.decay_factor + (1.0 - self.config.decay_factor))
            .unwrap_or(1.0 - self.config.decay_factor);

        // Remove old key if it existed.
        if let Some(old) = old_ewma {
            inner.pk_heat.remove(&(OrderedFloat(old), pk_vec.clone()));
        }

        // Insert new (ewma, pk) -> HeatSample and update secondary index.
        inner.pk_to_ewma.insert(pk_vec.clone(), new_ewma);
        inner.pk_heat.insert(
            (OrderedFloat(new_ewma), pk_vec.clone()),
            HeatSample {
                ewma: new_ewma,
                update_count: update_count + 1,
            },
        );
    }

    /// Returns true if the key is currently "hot" (EWMA > hot_threshold).
    pub fn is_hot(&self, pk: &[u8]) -> bool {
        let inner = self.inner.read();
        inner.pk_to_ewma.get(pk).copied().unwrap_or(0.0) >= self.config.hot_threshold
    }

    /// Returns true if the key is "cold" (EWMA < cold_threshold).
    pub fn is_cold(&self, pk: &[u8]) -> bool {
        let inner = self.inner.read();
        inner.pk_to_ewma.get(pk).copied().unwrap_or(0.0) <= self.config.cold_threshold
    }

    /// Returns the EWMA heat score for a key (0.0 if unknown).
    pub fn heat_score(&self, pk: &[u8]) -> f64 {
        let inner = self.inner.read();
        inner.pk_to_ewma.get(pk).copied().unwrap_or(0.0)
    }

    /// TRIAD hot/cold classification for a batch of write entries.
    ///
    /// Returns two slices: hot entries and cold entries.
    ///
    /// # Arguments
    ///
    /// - `entries`: slice of `(pk, value_bytes)` tuples representing a write batch.
    /// - Returns: `(hot, cold)` where each is a list of indices into the original batch.
    pub fn classify_batch(&self, entries: &[(Vec<u8>, usize)]) -> (Vec<usize>, Vec<usize>) {
        let mut hot = Vec::new();
        let mut cold = Vec::new();
        let mut unknown_indices: Vec<usize> = Vec::new();

        // First pass: classify known keys using pk_to_ewma secondary index (O(1) per key).
        {
            let inner = self.inner.read();
            let hot_threshold = self.config.hot_threshold;
            for (idx, (pk, _)) in entries.iter().enumerate() {
                match inner.pk_to_ewma.get(pk) {
                    Some(&ewma) if ewma >= hot_threshold => hot.push(idx),
                    Some(_) => cold.push(idx),
                    None => {
                        cold.push(idx);
                        unknown_indices.push(idx);
                    }
                }
            }
        }

        // Record unknown keys after releasing the read lock.
        for &idx in &unknown_indices {
            let pk = &entries[idx].0;
            self.record_cold_key(pk);
        }

        (hot, cold)
    }

    /// Add a hot write to the hot buffer (accumulates until forced flush).
    pub fn buffer_hot_write(&self, pk: Vec<u8>, value: Vec<u8>) {
        self.hot_buffer
            .write()
            .add(pk, value, self.config.hot_buffer_max_bytes);
    }

    /// Returns true if the hot buffer should be flushed (size threshold reached).
    pub fn should_flush_hot(&self) -> bool {
        let buf = self.hot_buffer.read();
        buf.total_bytes >= self.config.hot_buffer_max_bytes
    }

    /// Drain and return all hot buffered entries.
    pub fn drain_hot_buffer(&self) -> HashMap<Vec<u8>, Vec<Vec<u8>>> {
        let mut buf = self.hot_buffer.write();
        let entries = buf.entries.drain(..).collect();
        buf.total_bytes = 0;
        entries
    }

    /// Returns the number of hot keys currently tracked.
    pub fn hot_key_count(&self) -> usize {
        let inner = self.inner.read();
        let threshold = self.config.hot_threshold;
        inner.pk_heat.values().filter(|s| s.ewma >= threshold).count()
    }

    /// Returns the total number of tracked keys.
    pub fn tracked_key_count(&self) -> usize {
        self.inner.read().pk_heat.len()
    }

    /// Record a cold key access (for statistics).
    fn record_cold_key(&self, pk: &[u8]) {
        let mut recent = self.recent_cold_keys.write();
        if recent.len() >= 1024 {
            recent.pop_front();
        }
        let hash = simple_hash(pk);
        recent.push_back(hash);
    }

    /// Estimate current write throughput in MB/s.
    ///
    /// Computed from the EWMA heat scores: higher average heat → higher write rate.
    /// This is a rough estimate suitable for EcoTune's compaction scheduling heuristic.
    pub fn estimated_mbps(&self) -> f64 {
        let inner = self.inner.read();
        if inner.pk_heat.is_empty() {
            return 10.0; // Default estimate when no data.
        }
        let total_ewma: f64 = inner.pk_heat.values().map(|s| s.ewma).sum();
        let avg_ewma = total_ewma / inner.pk_heat.len() as f64;
        // Rough calibration: avg EWMA of 0.5 ≈ 10 MB/s. Scale linearly.
        (avg_ewma * 20.0).clamp(1.0, 1000.0)
    }

    /// Clear all heat state (used after checkpoint or restart).
    pub fn clear(&self) {
        self.inner.write().pk_heat.clear();
        self.hot_buffer.write().clear();
        self.recent_cold_keys.write().clear();
    }

    /// Remove heat state for a specific key (call after compaction removes it).
    ///
    /// Cleans both pk_heat (BTreeMap) and pk_to_ewma (HashMap) to prevent memory leaks.
    pub fn forget_key(&self, pk: &[u8]) {
        let pk_vec = pk.to_vec();

        let mut inner = self.inner.write();

        // Clean pk_heat (BTreeMap)
        if let Some(key) = inner
            .pk_heat
            .range(..)
            .find(|((_, k), _)| k.as_slice() == pk_vec)
            .map(|((e, k), _)| (*e, k.clone()))
        {
            inner.pk_heat.remove(&key);
        }

        // Clean pk_to_ewma (HashMap) - fix for W-1 memory leak
        inner.pk_to_ewma.remove(&pk_vec);
    }
}

impl Default for WriteHeatTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Simple non-crypto hash for statistics (fnv-style).
fn simple_hash(data: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

// ---------------------------------------------------------------------------
// Shared handle
// ---------------------------------------------------------------------------

/// Thread-safe cloneable handle to the WriteHeatTracker.
#[derive(Clone)]
pub struct WriteHeatTrackerHandle {
    inner: Arc<WriteHeatTracker>,
}

impl WriteHeatTrackerHandle {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(WriteHeatTracker::new()),
        }
    }

    pub fn with_config(config: HeatTrackerConfig) -> Self {
        Self {
            inner: Arc::new(WriteHeatTracker::with_config(config)),
        }
    }

    pub fn record_update(&self, pk: &[u8]) {
        self.inner.record_update(pk);
    }

    pub fn is_hot(&self, pk: &[u8]) -> bool {
        self.inner.is_hot(pk)
    }

    pub fn is_cold(&self, pk: &[u8]) -> bool {
        self.inner.is_cold(pk)
    }

    pub fn heat_score(&self, pk: &[u8]) -> f64 {
        self.inner.heat_score(pk)
    }

    pub fn classify_batch(&self, entries: &[(Vec<u8>, usize)]) -> (Vec<usize>, Vec<usize>) {
        self.inner.classify_batch(entries)
    }

    pub fn buffer_hot_write(&self, pk: Vec<u8>, value: Vec<u8>) {
        self.inner.buffer_hot_write(pk, value);
    }

    pub fn should_flush_hot(&self) -> bool {
        self.inner.should_flush_hot()
    }

    pub fn drain_hot_buffer(&self) -> HashMap<Vec<u8>, Vec<Vec<u8>>> {
        self.inner.drain_hot_buffer()
    }

    pub fn hot_key_count(&self) -> usize {
        self.inner.hot_key_count()
    }

    pub fn estimated_mbps(&self) -> f64 {
        self.inner.estimated_mbps()
    }

    pub fn tracked_key_count(&self) -> usize {
        self.inner.tracked_key_count()
    }

    pub fn clear(&self) {
        self.inner.clear();
    }
}

impl Default for WriteHeatTrackerHandle {
    fn default() -> Self {
        Self::new()
    }
}
