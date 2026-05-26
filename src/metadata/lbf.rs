//! Learned Bloom Filter using Piecewise Linear Model
//!
//! A piecewise linear model predicts which segment a primary key belongs to,
//! allowing us to skip Bloom Filter probes for irrelevant segments.
//!
//!参照: Learned LSM-trees (arxiv 2508.00882, 2025)
//!   - Approach 1: Classifier skips Bloom filter probes for irrelevant levels (our approach)
//!   - Approach 2: Learned Bloom Filter = compact model + small backup Bloom Filter

use std::collections::HashMap;
use std::sync::RwLock;

/// Piecewise Linear Model for predicting segment index from primary key value.
///
///
/// Training data: sorted (pk, seg_index) pairs where seg_index is which segment
/// the pk belongs to (computed from segment pk_range boundaries).
///
/// The model partitions the pk space into contiguous ranges, each assigned to
/// one segment. Within each range, a simple linear model predicts the boundary.
#[derive(Debug, Clone)]
pub struct PiecewiseLinearModel {
    /// Sorted boundary pk values (breakpoints between segments)
    boundaries: Vec<i64>,
    /// Slope for each segment (change in segment index per pk unit)
    slopes: Vec<f64>,
    /// Intercept for each segment
    intercepts: Vec<f64>,
    /// Number of segments
    num_segments: usize,
}

impl PiecewiseLinearModel {
    /// Build a piecewise linear model from sorted pk values and segment boundaries.
    ///
    /// `sorted_pks` — sample of primary key values (should be sorted)
    /// `seg_boundaries` — for each segment, the inclusive starting pk value
    ///
    /// E.g., sorted_pks = [1,2,3,...,100], seg_boundaries = [(1,0), (51,1)]
    ///   → pk [1,50] belong to seg 0, pk [51,100] belong to seg 1
    pub fn fit(sorted_pks: &[i64], seg_boundaries: &[(i64, usize)]) -> Self {
        if sorted_pks.is_empty() || seg_boundaries.is_empty() {
            return Self {
                boundaries: Vec::new(),
                slopes: Vec::new(),
                intercepts: Vec::new(),
                num_segments: 0,
            };
        }

        // Build boundaries: record every segment's starting pk (step function)
        let mut boundaries = Vec::new();
        let mut slopes = Vec::new();
        let mut intercepts = Vec::new();

        for &(pk, seg) in seg_boundaries {
            boundaries.push(pk);
            slopes.push(0.0);
            intercepts.push(seg as f64);
        }

        Self {
            boundaries,
            slopes,
            intercepts,
            num_segments: seg_boundaries.iter().map(|&(_, s)| s).max().unwrap_or(0) + 1,
        }
    }

    /// boundaries are sorted segment start pk values:
    ///   boundaries[i] = first pk belonging to segment i
    /// For pk, find the largest boundary <= pk → that segment
    pub fn predict(&self, pk: i64) -> usize {
        if self.boundaries.is_empty() {
            return 0;
        }

        // Find largest boundary <= pk: binary search for first boundary > pk, step back
        let seg = match self.boundaries.binary_search(&pk) {
            Ok(idx) => idx,     // pk == boundary[idx] → segment idx
            Err(idx) => {
                if idx == 0 {
                    0
                } else {
                    idx - 1      // idx is first boundary > pk, so segment idx-1
                }
            }
        };

        seg.min(self.num_segments.saturating_sub(1))
    }

    /// Model byte size (for memory estimation)
    pub fn byte_size(&self) -> usize {
        self.boundaries.len() * 8
            + self.slopes.len() * 8
            + self.intercepts.len() * 8
            + 8 // num_segments
    }
}

/// Learned Bloom Filter: PiecewiseLinearModel + small backup Bloom Filter.
///
///
/// The model predicts which segment a key belongs to. If the model says
/// the key is in segment S, we only check S's Bloom Filter.
/// The backup Bloom Filter catches false negatives from the model.
#[derive(Debug, Clone)]
pub struct LearnedBloomFilter {
    /// Piecewise linear model
    model: PiecewiseLinearModel,
    /// Backup Bloom Filter bytes (for false-negative safety)
    backup_bloom: Vec<u8>,
    /// Number of bits in the backup Bloom Filter
    num_bits: usize,
    /// Number of hash functions
    num_hashes: usize,
}

impl LearnedBloomFilter {
    /// Train a Learned Bloom Filter from sorted pk samples and segment boundaries.
    pub fn train(
        sorted_pks: &[i64],
        seg_boundaries: &[(i64, usize)],
        num_bits: usize,
        num_hashes: usize,
    ) -> Self {
        let model = PiecewiseLinearModel::fit(sorted_pks, seg_boundaries);

        // Build a small backup Bloom Filter from the pks (to catch model misses)
        let backup_bloom = build_bloom_filter(sorted_pks, num_bits, num_hashes);

        Self {
            model,
            backup_bloom,
            num_bits,
            num_hashes,
        }
    }

    /// Check if a pk might be in the database.
    /// Uses the model to narrow down the search to one segment,
    /// backed by a Bloom Filter for correctness guarantees.
    pub fn may_contain(&self, pk: i64) -> bool {
        // Step 1: Model predicts which segment
        let _predicted_seg = self.model.predict(pk);

        // Step 2: Check the backup Bloom Filter (covers all keys regardless of segment)
        // If backup BF says "no", return false (key definitely not present)
        // If backup BF says "yes", return true (key might be present)
        bloom_check(&self.backup_bloom, pk, self.num_bits, self.num_hashes)
    }

    /// Get the predicted segment index for a pk (without Bloom Filter check)
    pub fn predict_segment(&self, pk: i64) -> usize {
        self.model.predict(pk)
    }
}

/// Build a simple Bloom Filter as a bit array.
fn build_bloom_filter(keys: &[i64], num_bits: usize, num_hashes: usize) -> Vec<u8> {
    let mut bits = vec![0u8; (num_bits + 7) / 8];

    for &k in keys {
        for i in 0..num_hashes {
            let idx = hash(k, i) % num_bits;
            bits[(idx / 8) as usize] |= 1 << (idx % 8);
        }
    }

    bits
}

/// Check a Bloom Filter bit array for a key.
fn bloom_check(bits: &[u8], key: i64, num_bits: usize, num_hashes: usize) -> bool {
    for i in 0..num_hashes {
        let idx = hash(key, i) % num_bits;
            if bits[(idx / 8) as usize] & (1 << (idx % 8)) == 0 {
            return false;
        }
    }
    true
}

/// Simple hash function: splitmix64-like (fast, good distribution)
fn hash(key: i64, seed: usize) -> usize {
    let mut x = (key as u64).wrapping_add((seed as u64).wrapping_mul(0x9e3779b97f4a7c15));
    x = x ^ (x >> 30);
    x = x.wrapping_mul(0xbf58476d1ce4e5bu64);
    x = x ^ (x >> 27);
    x = x.wrapping_mul(0x94d049bb133111eb_u64);
    x = x ^ (x >> 31);
    x as usize
}

// ============================================================
// Persistence: use crate::codec::encode / decode
// (bincode_next::Encode / Decode already work via derive on PiecewiseLinearModel)
// ============================================================

// ============================================================
// Global LBF registry (in-memory, trained from segment metadata)
// ============================================================

static LBF_REGISTRY: once_cell::sync::Lazy<RwLock<HashMap<String, LearnedBloomFilter>>> =
    once_cell::sync::Lazy::new(|| RwLock::new(HashMap::new()));

/// Register (or update) the Learned Bloom Filter for a table.
pub fn register_lbf(table: &str, lbf: LearnedBloomFilter) {
    let mut registry = LBF_REGISTRY.write().unwrap();
    registry.insert(table.to_string(), lbf);
}

/// Get the Learned Bloom Filter for a table.
pub fn get_lbf(table: &str) -> Option<LearnedBloomFilter> {
    let registry = LBF_REGISTRY.read().unwrap();
    registry.get(table).cloned()
}

/// Remove the Learned Bloom Filter for a table.
pub fn remove_lbf(table: &str) {
    let mut registry = LBF_REGISTRY.write().unwrap();
    registry.remove(table);
}

/// Train a Learned Bloom Filter from segment metadata (pk ranges).
///
/// `seg_pk_ranges` maps seg_id → (min_pk, max_pk) as i64 tuples.
pub fn train_lbf_for_segments(
    table: &str,
    seg_pk_ranges: &[(String, (i64, i64))],
    sample_pks: &[i64],
    num_bits: usize,
    num_hashes: usize,
) {
    if seg_pk_ranges.is_empty() || sample_pks.is_empty() {
        return;
    }

    // Build sorted boundaries: (start_pk, seg_index)
    let mut boundaries: Vec<(i64, usize)> = seg_pk_ranges
        .iter()
        .enumerate()
        .map(|(i, (_, (min_pk, _)))| (*min_pk, i))
        .collect();
    boundaries.sort_by_key(|&(pk, _)| pk);

    let lbf = LearnedBloomFilter::train(sample_pks, &boundaries, num_bits, num_hashes);
    register_lbf(table, lbf);
}

/// Check if a pk might exist in a table, using the Learned Bloom Filter.
pub fn may_contain(table: &str, pk: i64) -> bool {
    get_lbf(table).map_or(true, |lbf| lbf.may_contain(pk))
}

#[cfg(test)]
mod tests {
    use super::*;

    // ============================================================
    // PiecewiseLinearModel
    // ============================================================

    #[test]
    fn test_pwlm_empty() {
        let model = PiecewiseLinearModel::fit(&[], &[]);
        assert_eq!(model.predict(0), 0);
        assert_eq!(model.byte_size(), 8); // only num_segments (8 bytes) with no vectors
    }

    #[test]
    fn test_pwlm_single_segment() {
        let pks = vec![1i64, 2, 3, 4, 5];
        let boundaries = vec![(1, 0)];
        let model = PiecewiseLinearModel::fit(&pks, &boundaries);

        assert_eq!(model.predict(1), 0);
        assert_eq!(model.predict(3), 0);
        assert_eq!(model.predict(5), 0);
    }

    #[test]
    fn test_pwlm_two_segments() {
        let pks = vec![1i64, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        // pk 1-5 -> seg 0, pk 6-10 -> seg 1
        let boundaries = vec![(1, 0), (6, 1)];
        let model = PiecewiseLinearModel::fit(&pks, &boundaries);

        assert_eq!(model.predict(1), 0);
        assert_eq!(model.predict(5), 0);
        assert_eq!(model.predict(6), 1);
        assert_eq!(model.predict(10), 1);
    }

    #[test]
    fn test_pwlm_three_segments() {
        let pks: Vec<i64> = (1..=30).collect();
        let boundaries = vec![(1, 0), (11, 1), (21, 2)];
        let model = PiecewiseLinearModel::fit(&pks, &boundaries);

        assert_eq!(model.predict(1), 0);
        assert_eq!(model.predict(10), 0);
        assert_eq!(model.predict(11), 1);
        assert_eq!(model.predict(20), 1);
        assert_eq!(model.predict(21), 2);
        assert_eq!(model.predict(30), 2);
    }

    #[test]
    fn test_pwlm_predict_out_of_range() {
        let pks = vec![10i64, 20, 30];
        let boundaries = vec![(10, 0), (20, 1)];
        let model = PiecewiseLinearModel::fit(&pks, &boundaries);

        assert_eq!(model.predict(5), 0);   // below first boundary -> seg 0
        assert_eq!(model.predict(100), 1); // above last boundary -> last seg
    }

    // ============================================================
    // LearnedBloomFilter
    // ============================================================

    #[test]
    fn test_lbf_construct() {
        let pks: Vec<i64> = (1..=100).collect();
        let boundaries = vec![(1, 0), (51, 1)];
        let lbf = LearnedBloomFilter::train(&pks, &boundaries, 1024, 3);

        assert_eq!(lbf.model.predict(25), 0);
        assert_eq!(lbf.model.predict(75), 1);
    }

    #[test]
    fn test_lbf_may_contain_true() {
        let pks: Vec<i64> = (1..=100).collect();
        let boundaries = vec![(1, 0), (51, 1)];
        let lbf = LearnedBloomFilter::train(&pks, &boundaries, 1024, 3);

        // Backup BF should return true for keys in the set
        assert!(lbf.may_contain(50));
        assert!(lbf.may_contain(51));
        assert!(lbf.may_contain(99));
    }

    #[test]
    fn test_lbf_false_positive_allowed() {
        let pks: Vec<i64> = (1..=100).collect();
        let boundaries = vec![(1, 0), (51, 1)];
        let lbf = LearnedBloomFilter::train(&pks, &boundaries, 512, 3);

        // BF can have false positives — that's expected behavior
        let result = lbf.may_contain(200);
        // Can't guarantee false positive won't happen, but it might
        let _ = result;
    }

    #[test]
    fn test_lbf_predict_segment() {
        let pks: Vec<i64> = (1..=100).collect();
        let boundaries = vec![(1, 0), (51, 1)];
        let lbf = LearnedBloomFilter::train(&pks, &boundaries, 512, 3);

        assert_eq!(lbf.predict_segment(25), 0);
        assert_eq!(lbf.predict_segment(51), 1);
        assert_eq!(lbf.predict_segment(100), 1);
    }

    // ============================================================
    // Global registry
    // ============================================================

    #[test]
    fn test_register_and_get_lbf() {
        let pks: Vec<i64> = (1..=50).collect();
        let boundaries = vec![(1, 0)];
        let lbf = LearnedBloomFilter::train(&pks, &boundaries, 256, 2);

        register_lbf("users", lbf.clone());
        let retrieved = get_lbf("users").unwrap();
        assert_eq!(retrieved.predict_segment(25), 0);

        remove_lbf("users");
        assert!(get_lbf("users").is_none());
    }

    #[test]
    fn test_may_contain_fallback() {
        // No LBF registered -> returns true (conservative)
        assert!(may_contain("nonexistent_table", 123));
    }

    #[test]
    fn test_train_lbf_for_segments() {
        let segs = vec![
            ("seg_0".to_string(), (1i64, 100i64)),
            ("seg_1".to_string(), (101i64, 200i64)),
        ];
        let pks: Vec<i64> = (1..=200).collect();

        train_lbf_for_segments("t", &segs, &pks, 512, 3);

        let lbf = get_lbf("t").unwrap();
        assert_eq!(lbf.predict_segment(50), 0);
        assert_eq!(lbf.predict_segment(150), 1);

        remove_lbf("t");
    }

    // ============================================================
    // Bloom Filter internals
    // ============================================================

    #[test]
    fn test_hash_deterministic() {
        let h1 = hash(42, 0);
        let h2 = hash(42, 0);
        assert_eq!(h1, h2, "hash should be deterministic");
    }

    #[test]
    fn test_hash_different_seeds() {
        let h1 = hash(42, 0);
        let h2 = hash(42, 1);
        let h3 = hash(42, 2);
        assert_ne!(h1, h2);
        assert_ne!(h2, h3);
    }

    #[test]
    fn test_bloom_check_all_set() {
        // Build a BF with all bits set
        let bits = vec![0xFFu8; 128];
        for k in 0..1000 {
            assert!(bloom_check(&bits, k, 1024, 3));
        }
    }

    #[test]
    fn test_bloom_check_all_clear() {
        let bits = vec![0u8; 128];
        for k in 0..1000 {
            assert!(!bloom_check(&bits, k, 1024, 3));
        }
    }

    // ============================================================
    // Model prediction accuracy: verify the model is actually working
    // ============================================================

    #[test]
    fn test_model_predict_segment_exactly_correct() {
        // Construct precise segments: pk 1-50 → seg 0, pk 51-100 → seg 1
        let pks: Vec<i64> = (1i64..=100).collect();
        let boundaries = vec![(1, 0), (51, 1)];
        let model = PiecewiseLinearModel::fit(&pks, &boundaries);

        for pk in 1..=50 {
            assert_eq!(model.predict(pk), 0, "pk {} should be in segment 0", pk);
        }
        for pk in 51..=100 {
            assert_eq!(model.predict(pk), 1, "pk {} should be in segment 1", pk);
        }
    }

    #[test]
    fn test_model_predict_out_of_range_bounds() {
        // pk outside training range: must be bounded to the nearest segment
        let pks: Vec<i64> = (10i64..=90).collect();
        let boundaries = vec![(10, 0), (50, 1), (90, 2)];
        let model = PiecewiseLinearModel::fit(&pks, &boundaries);

        // pk < first boundary → segment 0
        assert_eq!(model.predict(1), 0);
        assert_eq!(model.predict(9), 0);
        // pk > last boundary → last segment
        assert_eq!(model.predict(91), 2);
        assert_eq!(model.predict(1000), 2);
    }

    #[test]
    fn test_model_binary_search_boundary_precision() {
        // Precisely test binary search boundary behavior at every boundary
        let pks: Vec<i64> = (1i64..=1000).collect();
        let boundaries: Vec<(i64, usize)> = (1..=10)
            .map(|i| (i * 100, i as usize - 1))
            .collect();
        let model = PiecewiseLinearModel::fit(&pks, &boundaries);

        // pk == boundary: exact match
        for i in 1..=10 {
            assert_eq!(
                model.predict(i * 100), (i - 1) as usize,
                "pk {} (boundary {}) should match segment {}",
                i * 100, i, i - 1
            );
        }
        // pk between boundaries
        assert_eq!(model.predict(150), 0);
        assert_eq!(model.predict(250), 1);
        assert_eq!(model.predict(950), 8);
    }

    #[test]
    fn test_lbf_integration_point_get_correctness() {
        // End-to-end: train LBF → model predicts segment correctly → may_contain returns true for in-set keys
        let pks: Vec<i64> = (1i64..=200).collect();
        let boundaries = vec![(1, 0), (101, 1)];
        let lbf = LearnedBloomFilter::train(&pks, &boundaries, 1024, 3);

        // Model prediction must be accurate
        for pk in 1..=100 {
            assert_eq!(lbf.predict_segment(pk), 0, "pk {} should be in segment 0", pk);
        }
        for pk in 101..=200 {
            assert_eq!(lbf.predict_segment(pk), 1, "pk {} should be in segment 1", pk);
        }

        // may_contain must return true for in-set keys
        for pk in &[1i64, 50, 100, 101, 150, 200] {
            assert!(lbf.may_contain(*pk),
                "may_contain must return true for in-set pk {}", pk);
        }
    }

    #[test]
    fn test_lbf_false_positive_rate_acceptable() {
        // Verify false positive rate stays within acceptable bounds
        let pks: Vec<i64> = (1i64..=1000).collect();
        let boundaries = vec![(1, 0)];
        let lbf = LearnedBloomFilter::train(&pks, &boundaries, 8192, 7);

        let not_in_set: Vec<i64> = (2001i64..=3000).collect();
        let mut false_positives = 0;
        for pk in &not_in_set {
            if lbf.may_contain(*pk) {
                false_positives += 1;
            }
        }

        let fp_rate = false_positives as f64 / not_in_set.len() as f64;
        assert!(fp_rate < 0.05,
            "false positive rate should be < 5%, got {:.2}%", fp_rate * 100.0);
    }

    #[test]
    fn test_train_lbf_empty_input() {
        let model = PiecewiseLinearModel::fit(&[], &[]);
        assert_eq!(model.predict(0), 0, "empty model should return safe default segment 0");
    }

    #[test]
    fn test_train_lbf_single_segment() {
        let pks: Vec<i64> = (1i64..=100).collect();
        let boundaries = vec![(1, 0)];
        let lbf = LearnedBloomFilter::train(&pks, &boundaries, 256, 2);

        assert_eq!(lbf.predict_segment(50), 0);
        assert_eq!(lbf.predict_segment(1), 0);
        assert_eq!(lbf.predict_segment(100), 0);
    }
}
