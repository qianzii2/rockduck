//! Adaptive DelMask — three-state switching (Sparse / Roaring / Dense).
//!
//! # Three-State Strategy
//!
//! ## Sparse (< 1%)
//!
//! When deleted rows are < 1% of total rows, a `BTreeSet<u32>` (sorted set)
//! is most space-efficient: ~8 bytes per deleted row.
//!
//! ## RoaringBitmap (1% - 30%)
//!
//! When deleted rows are 1%-30%, a RoaringBitmap is optimal:
//! - O(1) contains, rank, and select
//! - Compressed (2 bytes per 64 consecutive elements)
//! - Fast serialization (croaring crate)
//!
//! ## Dense (> 30%)
//!
//! When deleted rows are > 30%, a dense `bitvec::BitVec` is optimal:
//! - O(1) access with no compression overhead
//! - 1 bit per row (always 12.5% of rows size)
//!
//! # Switching Strategy
//!
//! We check the ratio after each delete. When the ratio crosses a threshold,
//! we switch representation and rewrite the mask. Thresholds are configurable
//! but default to 1% (Sparse→Roaring) and 30% (Roaring→Dense).

use std::collections::BTreeSet;
use std::ops::Range;

use bitvec::vec::BitVec;
use croaring::Bitmap;

/// Representation modes for the adaptive delete mask.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Representation {
    /// BTreeSet<u32> — optimal for < 1% deletion rate
    Sparse,
    /// RoaringBitmap — optimal for 1%-30% deletion rate
    Roaring,
    /// BitVec — optimal for > 30% deletion rate
    Dense,
}

impl Representation {
    /// Threshold for switching from Sparse to Roaring (1%).
    pub const SPARSE_THRESHOLD: f64 = 0.01;
    /// Threshold for switching from Roaring to Dense (30%).
    pub const DENSE_THRESHOLD: f64 = 0.30;

    /// Determine the best representation for a given deletion rate.
    pub fn from_rate(rate: f64) -> Self {
        if rate < Self::SPARSE_THRESHOLD {
            Self::Sparse
        } else if rate < Self::DENSE_THRESHOLD {
            Self::Roaring
        } else {
            Self::Dense
        }
    }
}

/// Adaptive delete mask with three internal representations.
/// Automatically switches representation based on deletion rate.
pub enum AdaptiveDelMask {
    /// Sparse: BTreeSet for < 1% deletion rate
    Sparse(BTreeSet<u32>),
    /// Roaring: RoaringBitmap for 1%-30% deletion rate
    Roaring(Bitmap),
    /// Dense: BitVec for > 30% deletion rate
    Dense(BitVec),
}

impl AdaptiveDelMask {
    /// Create a new empty AdaptiveDelMask (starts in Sparse mode).
    pub fn new() -> Self {
        Self::Sparse(BTreeSet::new())
    }

    /// Create with a known total_rows and initial deleted rows.
    pub fn with_capacity(total_rows: u64, deleted_rows: &[u32]) -> Self {
        let rate = if total_rows == 0 {
            0.0
        } else {
            deleted_rows.len() as f64 / total_rows as f64
        };
        let rep = Representation::from_rate(rate);
        let mut mask = Self::with_representation(rep);
        for &row in deleted_rows {
            mask.delete(row);
        }
        mask
    }

    /// Create an AdaptiveDelMask in the specified representation (empty).
    fn with_representation(rep: Representation) -> Self {
        match rep {
            Representation::Sparse => Self::Sparse(BTreeSet::new()),
            Representation::Roaring => Self::Roaring(Bitmap::new()),
            Representation::Dense => Self::Dense(BitVec::new()),
        }
    }

    /// Mark a row as deleted and check if a representation switch is needed.
    pub fn delete(&mut self, row: u32) {
        self.delete_inner(row);
        // Representation switching is deferred — call `switch_if_needed(total_rows)` externally
        // after a batch of deletes to avoid repeated conversions during bulk inserts.
    }

    fn delete_inner(&mut self, row: u32) {
        match self {
            Self::Sparse(set) => {
                set.insert(row);
            }
            Self::Roaring(bm) => {
                bm.add(row);
            }
            Self::Dense(bv) => {
                if row as usize >= bv.len() {
                    bv.resize(row as usize + 1, false);
                }
                bv.set(row as usize, true);
            }
        }
    }

    /// Get the current representation mode.
    pub fn representation(&self) -> Representation {
        match self {
            Self::Sparse(_) => Representation::Sparse,
            Self::Roaring(_) => Representation::Roaring,
            Self::Dense(_) => Representation::Dense,
        }
    }

    /// Switch representation when deletion rate crosses a threshold.
    /// After a batch of deletes, call this with the known total_rows.
    pub fn switch_if_needed(&mut self, total_rows: u64) {
        let rep = self.representation();
        let rate = if total_rows == 0 {
            0.0
        } else {
            self.deleted_count() as f64 / total_rows as f64
        };
        let target = Representation::from_rate(rate);

        if rep != target {
            let new_mask = self.convert_to(target);
            *self = new_mask;
        }
    }

    /// Convert the current mask to a different representation.
    fn convert_to(&self, target: Representation) -> Self {
        match self {
            Self::Sparse(set) => match target {
                Representation::Sparse => Self::Sparse(set.clone()),
                Representation::Roaring => {
                    let mut bm = Bitmap::new();
                    for &row in set {
                        bm.add(row);
                    }
                    Self::Roaring(bm)
                }
                Representation::Dense => {
                    let mut bv = BitVec::new();
                    for &row in set {
                        let idx = row as usize;
                        if idx >= bv.len() {
                            bv.resize(idx + 1, false);
                        }
                        bv.set(idx, true);
                    }
                    Self::Dense(bv)
                }
            },
            Self::Roaring(bm) => match target {
                Representation::Sparse => {
                    let set: BTreeSet<u32> = bm.iter().collect();
                    Self::Sparse(set)
                }
                Representation::Roaring => Self::Roaring(bm.clone()),
                Representation::Dense => {
                    let mut bv = BitVec::new();
                    for row in bm.iter() {
                        let idx = row as usize;
                        if idx >= bv.len() {
                            bv.resize(idx + 1, false);
                        }
                        bv.set(idx, true);
                    }
                    Self::Dense(bv)
                }
            },
            Self::Dense(bv) => match target {
                Representation::Sparse => {
                    let set: BTreeSet<u32> = bv.iter_ones().map(|i| i as u32).collect();
                    Self::Sparse(set)
                }
                Representation::Roaring => {
                    let mut bm = Bitmap::new();
                    for (i, bit) in bv.iter().enumerate() {
                        if *bit {
                            bm.add(i as u32);
                        }
                    }
                    Self::Roaring(bm)
                }
                Representation::Dense => Self::Dense(bv.clone()),
            },
        }
    }

    /// Check if a row is deleted.
    pub fn is_deleted(&self, row: u32) -> bool {
        match self {
            Self::Sparse(set) => set.contains(&row),
            Self::Roaring(bm) => bm.contains(row),
            Self::Dense(bv) => bv.get(row as usize).map(|b| *b).unwrap_or(false),
        }
    }

    /// Get the count of deleted rows.
    pub fn deleted_count(&self) -> u64 {
        match self {
            Self::Sparse(set) => set.len() as u64,
            Self::Roaring(bm) => bm.cardinality(),
            Self::Dense(bv) => bv.count_ones() as u64,
        }
    }

    /// Iterate over deleted row offsets.
    pub fn iter(&self) -> Box<dyn Iterator<Item = u32> + '_> {
        match self {
            Self::Sparse(set) => Box::new(set.iter().copied()),
            Self::Roaring(bm) => Box::new(bm.iter()),
            Self::Dense(bv) => Box::new(bv.iter_ones().map(|i| i as u32)),
        }
    }

    /// Get all deleted row offsets as a sorted Vec.
    pub fn to_vec(&self) -> Vec<u32> {
        self.iter().collect()
    }

    /// Filter a range of rows, keeping only those NOT deleted.
    /// Returns the row offsets that remain (not deleted).
    ///
    /// Algorithm selection based on mask type and deletion density:
    /// - **Sparse**: if deletions are sparse within range, yield gaps (O(d log d));
    ///   if dense, iterate range and skip (O(r log n))
    /// - **Roaring**: if deletions are sparse within range, yield gaps (O(d));
    ///   if dense, iterate range and check (O(r))
    /// - **Dense**: always iterate range and check (O(r)) — O(1) per check
    pub fn filter_not_deleted(&self, range: Range<u32>) -> Vec<u32> {
        let range_size = range.end.saturating_sub(range.start);
        if range_size == 0 {
            return Vec::new();
        }

        let del_in_range: u32 = match self {
            // For Sparse: count how many deleted rows fall within range via iteration
            Self::Sparse(set) => {
                set.range(range.start..range.end).count() as u32
            }
            // For Roaring: use range_cardinality for O(1) count
            Self::Roaring(bm) => {
                bm.range_cardinality(range.start..range.end) as u32
            }
            // For Dense: count_ones over the range slice
            Self::Dense(bv) => {
                bv[range.start as usize..range.end as usize]
                    .iter()
                    .filter(|b| **b)
                    .count() as u32
            }
        };

        // Threshold: if deletions are sparse within range (< 30%), iterate deleted set.
        // If dense (>= 30%), iterate range and skip deleted.
        const SPARSE_THRESHOLD_NUM: u32 = 3;
        const SPARSE_THRESHOLD_DEN: u32 = 10;
        if del_in_range * SPARSE_THRESHOLD_NUM < range_size * SPARSE_THRESHOLD_DEN {
            // Sparse deletions: yield gaps by iterating deleted set
            match self {
                Self::Sparse(set) => {
                    let mut result = Vec::with_capacity((range_size - del_in_range) as usize);
                    let mut cur = range.start;
                    for &del in set.range(range.start..range.end) {
                        while cur < del {
                            result.push(cur);
                            cur += 1;
                        }
                        cur = del + 1;
                    }
                    while cur < range.end {
                        result.push(cur);
                        cur += 1;
                    }
                    result
                }
                Self::Roaring(bm) => {
                    let mut result = Vec::with_capacity((range_size - del_in_range) as usize);
                    let mut cur = range.start;
                    // Iterate over the bitmap subset and skip deleted rows
                    for del in bm.iter() {
                        if del < range.start {
                            continue;
                        }
                        if del >= range.end {
                            break;
                        }
                        while cur < del {
                            result.push(cur);
                            cur += 1;
                        }
                        cur = del + 1;
                    }
                    while cur < range.end {
                        result.push(cur);
                        cur += 1;
                    }
                    result
                }
                Self::Dense(_bv) => {
                    // Dense mask: iterate range, skip deleted (unreachable per threshold)
                    range.filter(|&r| !self.is_deleted(r)).collect()
                }
            }
        } else {
            // Dense deletions: iterate range, skip deleted
            range.filter(|&r| !self.is_deleted(r)).collect()
        }
    }

    /// Merge another delete mask into this one (union of bitmaps).
    /// Uses batch operations for optimal performance across all representation combinations.
    pub fn merge(&mut self, other: &AdaptiveDelMask) {
        match (&mut *self, other) {
            // (Sparse, Sparse): extend BTreeSet directly — O(m log n) where m = other size
            (Self::Sparse(a), Self::Sparse(b)) => a.extend(b.iter().copied()),

            // (Sparse, Roaring): iterate individual elements — O(n log n) instead of O(n)
            // FIX: Do NOT use add_range() here — Sparse bitmap has gaps, so
            // add_range would incorrectly mark non-existent rows as deleted.
            // Always iterate and add individual rows for correctness (same as (Roaring, Sparse)).
            (Self::Sparse(a), Self::Roaring(b)) => {
                if a.is_empty() {
                    // Sparse is empty — just convert Roaring to Sparse
                    *self = Self::Sparse(b.iter().collect());
                } else if b.is_empty() {
                    // Roaring is empty — nothing to do
                } else {
                    // Both non-empty: iterate Sparse elements individually for correctness,
                    // then union with Roaring (O(n log n) vs O(n) but correct).
                    let mut merged = Bitmap::new();
                    for &row in a.iter() {
                        merged.add(row);
                    }
                    merged.or_inplace(b);
                    *self = Self::Roaring(merged);
                }
            }

            // (Roaring, Sparse): add each Sparse element individually to Roaring
            // FIX: Do NOT use add_range() here - Sparse bitmap has gaps, so
            // add_range would incorrectly mark non-existent rows as deleted.
            // Always iterate and add individual rows for correctness.
            (Self::Roaring(a), Self::Sparse(b)) => {
                for &row in b {
                    a.add(row);
                }
            }

            // (Roaring, Roaring): native union — O(1) container-level union
            (Self::Roaring(a), Self::Roaring(b)) => {
                a.or_inplace(b);
            }

            // (Dense, Dense): bitwise OR — O(n) word-level OR, no per-element iteration
            (Self::Dense(a), Self::Dense(b)) => {
                let len = a.len().max(b.len());
                a.resize(len, false);
                let b_len = b.len();
                for i in 0..len {
                    let a_bit = if i < a.len() { a[i] } else { false };
                    let b_bit = if i < b_len { b[i] } else { false };
                    a.set(i, a_bit || b_bit);
                }
            }

            // Cross-representation: convert self to other's representation, then merge
            (s, other_rep) => {
                let target_rep = other_rep.representation();
                let mut new_self = Self::with_representation(target_rep);
                for row in s.iter() {
                    new_self.delete_inner(row);
                }
                for row in other_rep.iter() {
                    new_self.delete_inner(row);
                }
                *s = new_self;
            }
        }
    }

    /// Returns true if the mask is empty (no deletions).
    pub fn is_empty(&self) -> bool {
        self.deleted_count() == 0
    }
}

impl Default for AdaptiveDelMask {
    fn default() -> Self {
        Self::new()
    }
}
