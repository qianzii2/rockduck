//! Zone map statistics
//!
//! Zone maps store per-segment, per-column min/max statistics.
//! Used during scan to skip segments that provably contain no matching rows.
//!
//! Delegates min/max computation to `block_stats::compute_full_stats`
//! which uses concrete typed arrays for efficiency.

use arrow_array::ArrayRef;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

pub use crate::metadata::block_stats::compute_full_stats;
use crate::metadata::block_zone_map::bytes_lt;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZoneMapStats {
    pub columns: Vec<ColumnStats>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnStats {
    pub column_id: i32,
    pub min_value: Option<Vec<u8>>,
    pub max_value: Option<Vec<u8>>,
    pub null_count: i64,
    pub nan_count: i64,
    /// Data type for this column, used for type-aware zone map comparisons.
    /// Without this, `may_overlap` uses raw byte comparison which is incorrect
    /// for float (NaN, byte-order) and string (lexicographic vs numeric) types.
    pub data_type: arrow_schema::DataType,
}

impl ZoneMapStats {
    /// Create a new empty zone map.
    pub fn new() -> Self {
        Self {
            columns: Vec::new(),
        }
    }

    /// Load zone map from JSON file.
    pub fn load(path: &Path) -> std::io::Result<Self> {
        let data = fs::read(path)?;
        serde_json::from_slice(&data)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }

    /// Save zone map to JSON file.
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        let data = serde_json::to_vec_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        fs::write(path, data)
    }

    /// Update zone map statistics from an Arrow Array for a specific column.
    /// Uses block_stats::compute_full_stats for efficient typed min/max (no row-by-row loops).
    pub fn update_col(&mut self, column_id: i32, arr: &ArrayRef) {
        let dt = arr.data_type().clone();
        let cs = if let Some(cs) = self.columns.iter_mut().find(|c| c.column_id == column_id) {
            cs
        } else {
            self.columns.push(ColumnStats {
                column_id,
                min_value: None,
                max_value: None,
                null_count: 0,
                nan_count: 0,
                data_type: dt.clone(),
            });
            self.columns.last_mut().unwrap()
        };

        // Delegate to block_stats for efficient min/max (LLVM auto-vectorizes this)
        let stats = compute_full_stats(arr.as_ref());
        cs.null_count += stats.null_count as i64;

        if let Some(bytes) = stats.min_bytes {
            cs.min_value = Some(match &cs.min_value {
                Some(existing) => {
                    if existing.as_slice() <= bytes.as_slice() {
                        existing.clone()
                    } else {
                        bytes
                    }
                }
                None => bytes,
            });
        }
        if let Some(bytes) = stats.max_bytes {
            cs.max_value = Some(match &cs.max_value {
                Some(existing) => {
                    if existing.as_slice() >= bytes.as_slice() {
                        existing.clone()
                    } else {
                        bytes
                    }
                }
                None => bytes,
            });
        }
    }

    /// Check if this zone map could contain values within the given predicate range,
    /// using column index (position) instead of column_id.
    ///
    /// Uses type-aware byte comparison via `bytes_lt` to handle float NaN
    /// and string lexicographic ordering correctly.
    pub fn may_overlap_by_col_idx(
        &self,
        col_idx: usize,
        predicate_min: &[u8],
        predicate_max: &[u8],
    ) -> bool {
        let cs = match self.columns.get(col_idx) {
            Some(cs) => cs,
            None => return true,
        };

        if let Some(seg_max) = &cs.max_value {
            if bytes_lt(seg_max.as_slice(), predicate_min, &cs.data_type) {
                return false;
            }
        }
        if let Some(seg_min) = &cs.min_value {
            if bytes_lt(predicate_max, seg_min.as_slice(), &cs.data_type) {
                return false;
            }
        }

        true
    }

    /// Check if this zone map could contain values within the given predicate range.
    ///
    /// Uses type-aware byte comparison via `bytes_lt` to handle float NaN
    /// and string lexicographic ordering correctly.
    pub fn may_overlap(&self, column_id: i32, predicate_min: &[u8], predicate_max: &[u8]) -> bool {
        let cs = match self.columns.iter().find(|c| c.column_id == column_id) {
            Some(cs) => cs,
            None => return true,
        };

        if let Some(seg_max) = &cs.max_value {
            if bytes_lt(seg_max.as_slice(), predicate_min, &cs.data_type) {
                return false;
            }
        }
        if let Some(seg_min) = &cs.min_value {
            if bytes_lt(predicate_max, seg_min.as_slice(), &cs.data_type) {
                return false;
            }
        }

        true
    }
}

impl Default for ZoneMapStats {
    fn default() -> Self {
        Self::new()
    }
}
