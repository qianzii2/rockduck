//! LEA -- Learned Encoding Advisor: feature extraction.
//!
//! Extracts statistical features from column data for encoding selection.
//! All features are computed from a sample (default 1000 rows) for efficiency.

use arrow_array::Array;
use arrow_schema::DataType;

/// Arrow type codes for ML input.
pub const TYPE_NULL: u8 = 0;
pub const TYPE_BOOL: u8 = 1;
pub const TYPE_U8: u8 = 2;
pub const TYPE_U16: u8 = 3;
pub const TYPE_U32: u8 = 4;
pub const TYPE_U64: u8 = 5;
pub const TYPE_I8: u8 = 6;
pub const TYPE_I16: u8 = 7;
pub const TYPE_I32: u8 = 8;
pub const TYPE_I64: u8 = 9;
pub const TYPE_F32: u8 = 10;
pub const TYPE_F64: u8 = 11;
pub const TYPE_UTF8: u8 = 12;
pub const TYPE_BIN: u8 = 13;
pub const TYPE_OTHER: u8 = 255;

/// Statistical features extracted from a column sample.
#[derive(Debug, Clone, Default)]
pub struct ColumnFeatures {
    /// Ratio of unique values to total rows. High cardinality (near 1.0) → Raw.
    pub cardinality_ratio: f64,
    /// Shannon entropy in bits. Low entropy → RLE/Dictionary friendly.
    pub entropy: f64,
    /// Average run length (consecutive equal values).
    pub run_length_avg: f64,
    /// Maximum run length.
    pub run_length_max: usize,
    /// Fraction of null values.
    pub null_ratio: f64,
    /// log2 of value range (max - min + 1). Small range → BitPacking.
    pub range_log2: f64,
    /// Fraction of pairs that are in ascending order. Sorted → Delta/FoR.
    pub sortedness: f64,
    /// Estimated temporal locality score (0 = random, 1 = clustered).
    ///
    /// DESIGN-DECISION: `temporal_score` is hardcoded to 0.5 across all extractors.
    ///
    /// Reason: Computing true temporal locality requires timestamp correlation analysis,
    /// which needs access to write timestamps or arrival order metadata. Without such
    /// metadata in the column data itself, we cannot reliably detect temporal patterns.
    ///
    /// The default of 0.5 represents neutral temporal locality. Future implementations
    /// could derive this from:
    /// - Segment creation timestamps (accessible via SegmentMeta)
    /// - Row position correlation with time-based primary keys
    /// - Explicit write order metadata
    ///
    /// Status: FUNCTIONALITY-NOT-IMPLEMENTED (requires business definition of "temporal score")
    pub temporal_score: f64,
    /// Encoded Arrow type (see TYPE_* constants above).
    pub col_type: u8,
    /// Total number of rows in the sample.
    pub sample_size: usize,
}

impl ColumnFeatures {
    /// Extract features from an Arrow array by sampling `sample_size` rows.
    pub fn from_sample(arr: &dyn Array, sample_size: usize) -> Self {
        let arr_len = arr.len();
        if arr_len == 0 {
            return Self {
                col_type: TYPE_OTHER,
                ..Default::default()
            };
        }

        let col_type = Self::type_code(arr.data_type());
        let n = arr_len.min(sample_size);
        let step = if arr_len <= n { 1 } else { arr_len / n };

        let indices: Vec<usize> = (0..arr_len).step_by(step).take(n).collect();
        let actual_n = indices.len();
        if actual_n == 0 {
            return Self {
                col_type,
                sample_size: 0,
                ..Default::default()
            };
        }

        let null_count = indices.iter().filter(|&&i| arr.is_null(i)).count();
        let _non_null = actual_n - null_count;

        match arr.data_type() {
            DataType::Int32 => {
                let downcast = arr
                    .as_any()
                    .downcast_ref::<arrow_array::Int32Array>()
                    .expect("lea_features: Int32Array downcast failed - internal type mismatch");
                Self::extract_int::<i32>(
                    arr_len,
                    &indices,
                    null_count,
                    |i| downcast.value(i),
                    col_type,
                )
            }
            DataType::Int64 => {
                let downcast = arr
                    .as_any()
                    .downcast_ref::<arrow_array::Int64Array>()
                    .expect("lea_features: Int64Array downcast failed - internal type mismatch");
                Self::extract_int::<i64>(
                    arr_len,
                    &indices,
                    null_count,
                    |i| downcast.value(i),
                    col_type,
                )
            }
            DataType::UInt64 => {
                let downcast = arr
                    .as_any()
                    .downcast_ref::<arrow_array::UInt64Array>()
                    .expect("lea_features: UInt64Array downcast failed - internal type mismatch");
                Self::extract_int::<u64>(
                    arr_len,
                    &indices,
                    null_count,
                    |i| downcast.value(i),
                    col_type,
                )
            }
            DataType::Float32 => {
                let downcast = arr
                    .as_any()
                    .downcast_ref::<arrow_array::Float32Array>()
                    .expect("lea_features: Float32Array downcast failed - internal type mismatch");
                Self::extract_float_f32(&indices, null_count, |i| downcast.value(i))
            }
            DataType::Float64 => {
                let downcast = arr
                    .as_any()
                    .downcast_ref::<arrow_array::Float64Array>()
                    .expect("lea_features: Float64Array downcast failed - internal type mismatch");
                Self::extract_float_f64(&indices, null_count, |i| downcast.value(i))
            }
            DataType::Utf8 => {
                let downcast = arr
                    .as_any()
                    .downcast_ref::<arrow_array::StringArray>()
                    .expect("lea_features: StringArray downcast failed - internal type mismatch");
                Self::extract_str(&indices, null_count, |i| downcast.value(i))
            }
            _ => Self::extract_generic(arr_len, &indices, null_count),
        }
    }

    fn extract_int<T: Copy + PartialOrd + std::hash::Hash + Eq + Into<i128>>(
        _arr_len: usize,
        indices: &[usize],
        null_count: usize,
        get_value: impl Fn(usize) -> T,
        col_type: u8,
    ) -> Self {
        let actual_n = indices.len();
        let non_null = actual_n.saturating_sub(null_count);

        // Use HashMap to count occurrences for entropy calculation
        let mut unique_counts: std::collections::HashMap<T, usize> =
            std::collections::HashMap::new();
        let mut ordered_pairs_asc = 0usize;
        let mut ordered_pairs_desc = 0usize;
        let mut run_length_max = 0usize;
        let mut run_length_sum = 0usize;
        let mut current_run = 1usize;
        let mut prev_val: Option<T> = None;
        let mut min_val: Option<i128> = None;
        let mut max_val: Option<i128> = None;

        for &i in indices {
            let val = get_value(i);
            let val_i128: i128 = val.into();
            *unique_counts.entry(val).or_insert(0) += 1;

            if let Some(ref mut m) = min_val {
                if val_i128 < *m {
                    *m = val_i128;
                }
            } else {
                min_val = Some(val_i128);
            }
            if let Some(ref mut m) = max_val {
                if val_i128 > *m {
                    *m = val_i128;
                }
            } else {
                max_val = Some(val_i128);
            }

            if let Some(prev) = prev_val {
                let prev_i128: i128 = prev.into();
                if val_i128 == prev_i128 {
                    current_run += 1;
                } else {
                    run_length_sum += current_run;
                    run_length_max = run_length_max.max(current_run);
                    current_run = 1;
                    if val_i128 > prev_i128 {
                        ordered_pairs_asc += 1;
                    } else if val_i128 < prev_i128 {
                        ordered_pairs_desc += 1;
                    }
                }
            } else {
                current_run = 1;
            }
            prev_val = Some(val);
        }
        run_length_sum += current_run;
        run_length_max = run_length_max.max(current_run);

        let cardinality_ratio = if non_null > 0 {
            unique_counts.len() as f64 / non_null as f64
        } else {
            1.0
        };

        let run_length_avg = if non_null > 0 {
            run_length_sum as f64 / non_null as f64
        } else {
            1.0
        };

        // PR 7-A fix (CODEC-7): count both ascending and descending ordered pairs.
        // sortedness = max(asc, desc) / (non_null - 1) -- either direction indicates sorted data.
        let max_ordered = ordered_pairs_asc.max(ordered_pairs_desc);
        let sortedness = if non_null > 1 {
            max_ordered as f64 / (non_null - 1) as f64
        } else {
            0.0
        };

        let range_log2 = if let (Some(min), Some(max)) = (min_val, max_val) {
            if min < max {
                ((max - min).unsigned_abs().max(1) as f64).log2().max(0.0)
            } else {
                0.0
            }
        } else {
            0.0
        };

        // Correct Shannon entropy: H = -sum(p_i * log2(p_i)) where p_i = count_i / non_null
        let entropy = if non_null > 1 && unique_counts.len() > 1 {
            unique_counts
                .values()
                .map(|&count| {
                    let p = count as f64 / non_null as f64;
                    if p > 0.0 {
                        -p * p.log2()
                    } else {
                        0.0
                    }
                })
                .sum()
        } else {
            0.0
        };

        Self {
            cardinality_ratio,
            entropy,
            run_length_avg,
            run_length_max,
            null_ratio: null_count as f64 / actual_n as f64,
            range_log2,
            sortedness,
            temporal_score: 0.5,
            col_type, //: use passed col_type instead of hardcoded TYPE_OTHER
            sample_size: actual_n,
        }
    }

    fn extract_float_f64(
        indices: &[usize],
        null_count: usize,
        get_value: impl Fn(usize) -> f64,
    ) -> Self {
        use std::collections::HashSet;

        let actual_n = indices.len();
        let na_count = indices.iter().filter(|&&i| get_value(i).is_nan()).count();
        let non_null = actual_n.saturating_sub(null_count).saturating_sub(na_count);

        // Use u64 bit representation for HashSet (f64 doesn't impl Hash/Eq)
        let mut unique_bits = HashSet::new();
        let mut ordered_pairs = 0usize;
        let mut run_length_max = 0usize;
        let mut run_length_sum = 0usize;
        let mut current_run = 1usize;
        let mut prev_val: Option<f64> = None;
        let mut min_val: Option<f64> = None;
        let mut max_val: Option<f64> = None;

        for &i in indices {
            let val = get_value(i);

            if val.is_nan() {
                // NaN values are excluded from cardinality calculation.
                // NaN != NaN by IEEE-754 semantics, so counting NaN bits would inflate
                // the cardinality estimate without meaningful information.
                continue;
            }

            if let Some(ref mut m) = min_val {
                if val < *m {
                    *m = val;
                }
            } else {
                min_val = Some(val);
            }
            if let Some(ref mut m) = max_val {
                if val > *m {
                    *m = val;
                }
            } else {
                max_val = Some(val);
            }

            unique_bits.insert(val.to_bits());

            if let Some(prev) = prev_val {
                if (val - prev).abs() < f64::EPSILON {
                    current_run += 1;
                } else {
                    run_length_sum += current_run;
                    run_length_max = run_length_max.max(current_run);
                    current_run = 1;
                    if val > prev {
                        ordered_pairs += 1;
                    }
                }
            } else {
                current_run = 1;
            }
            prev_val = Some(val);
        }
        run_length_sum += current_run;
        run_length_max = run_length_max.max(current_run);

        let cardinality_ratio = if non_null > 0 {
            unique_bits.len() as f64 / non_null as f64
        } else {
            1.0
        };

        let run_length_avg = if non_null > 0 {
            run_length_sum as f64 / non_null as f64
        } else {
            1.0
        };

        let sortedness = if non_null > 1 {
            ordered_pairs as f64 / (non_null - 1) as f64
        } else {
            0.0
        };

        let range_log2 = if let (Some(min), Some(max)) = (min_val, max_val) {
            if max.is_finite() && min.is_finite() && max > min {
                ((max - min) + 1.0).log2().max(0.0)
            } else {
                0.0
            }
        } else {
            0.0
        };

        // Shannon entropy approximation for floats.
        // For unique bit patterns (each frequency = 1), entropy = log2(unique_count).
        // This is meaningful: high entropy = high cardinality.
        let entropy = if non_null > 1 && unique_bits.len() > 1 {
            (unique_bits.len() as f64).log2()
        } else {
            0.0
        };

        Self {
            cardinality_ratio,
            entropy,
            run_length_avg,
            run_length_max,
            null_ratio: null_count as f64 / actual_n as f64,
            range_log2,
            sortedness,
            temporal_score: 0.5,
            col_type: TYPE_F64,
            sample_size: actual_n,
        }
    }

    fn extract_float_f32(
        indices: &[usize],
        null_count: usize,
        get_value: impl Fn(usize) -> f32,
    ) -> Self {
        use std::collections::HashSet;

        let actual_n = indices.len();
        let na_count = indices.iter().filter(|&&i| get_value(i).is_nan()).count();
        let non_null = actual_n.saturating_sub(null_count).saturating_sub(na_count);

        // Use u32 bit representation for HashSet (f32 doesn't impl Hash/Eq)
        let mut unique_bits: HashSet<u32> = HashSet::new();
        let mut ordered_pairs = 0usize;
        let mut run_length_max = 0usize;
        let mut run_length_sum = 0usize;
        let mut current_run = 1usize;
        let mut prev_val: Option<f32> = None;
        let mut min_val: Option<f32> = None;
        let mut max_val: Option<f32> = None;

        for &i in indices {
            let val = get_value(i);

            if val.is_nan() {
                // NaN values are excluded from cardinality calculation.
                // NaN != NaN by IEEE-754 semantics, so counting NaN bits would inflate
                // the cardinality estimate without meaningful information.
                continue;
            }

            unique_bits.insert(val.to_bits());

            if let Some(ref mut m) = min_val {
                if val < *m {
                    *m = val;
                }
            } else {
                min_val = Some(val);
            }
            if let Some(ref mut m) = max_val {
                if val > *m {
                    *m = val;
                }
            } else {
                max_val = Some(val);
            }

            // Track ordered pairs for sortedness metric
            if let Some(prev) = prev_val {
                if val >= prev {
                    ordered_pairs += 1;
                } else {
                    run_length_sum += current_run;
                    run_length_max = run_length_max.max(current_run);
                    current_run = 1;
                }
            }
            prev_val = Some(val);
        }
        run_length_sum += current_run;
        run_length_max = run_length_max.max(current_run);

        let cardinality_ratio = if non_null > 0 {
            unique_bits.len() as f64 / non_null as f64
        } else {
            1.0
        };

        let run_length_avg = if non_null > 0 {
            run_length_sum as f64 / non_null as f64
        } else {
            1.0
        };

        let sortedness = if non_null > 1 {
            ordered_pairs as f64 / (non_null - 1) as f64
        } else {
            0.0
        };

        let range_log2 = if let (Some(min), Some(max)) = (min_val, max_val) {
            if max.is_finite() && min.is_finite() && max > min {
                ((max - min) + 1.0f32).log2().max(0.0) as f64
            } else {
                0.0
            }
        } else {
            0.0
        };

        // Shannon entropy approximation for floats.
        // For unique bit patterns (each frequency = 1), entropy = log2(unique_count).
        // This is meaningful: high entropy = high cardinality.
        let entropy = if non_null > 1 && unique_bits.len() > 1 {
            (unique_bits.len() as f64).log2()
        } else {
            0.0
        };

        Self {
            cardinality_ratio,
            entropy,
            run_length_avg,
            run_length_max,
            null_ratio: null_count as f64 / actual_n as f64,
            range_log2,
            sortedness,
            temporal_score: 0.5,
            col_type: TYPE_F32,
            sample_size: actual_n,
        }
    }

    fn extract_str<'a>(
        indices: &[usize],
        null_count: usize,
        get_value: impl Fn(usize) -> &'a str,
    ) -> Self {
        let actual_n = indices.len();
        let non_null = actual_n.saturating_sub(null_count);

        let mut unique_counts: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();
        let mut run_length_max = 0usize;
        let mut run_length_sum = 0usize;
        let mut current_run = 1usize;
        let mut prev_val: Option<&str> = None;

        for &i in indices {
            let val = get_value(i);
            let entry = unique_counts.entry(val.to_string());
            *entry.or_insert(0) += 1;

            if let Some(prev) = prev_val {
                if val == prev {
                    current_run += 1;
                } else {
                    run_length_sum += current_run;
                    run_length_max = run_length_max.max(current_run);
                    current_run = 1;
                }
            } else {
                current_run = 1;
            }
            prev_val = Some(val);
        }
        run_length_sum += current_run;
        run_length_max = run_length_max.max(current_run);

        let cardinality_ratio = if non_null > 0 {
            unique_counts.len() as f64 / non_null as f64
        } else {
            1.0
        };

        // Correct Shannon entropy: H = -sum(p_i * log2(p_i)) where p_i = count_i / non_null
        let entropy = if non_null > 1 && unique_counts.len() > 1 {
            unique_counts
                .values()
                .map(|&count| {
                    let p = count as f64 / non_null as f64;
                    if p > 0.0 {
                        -p * p.log2()
                    } else {
                        0.0
                    }
                })
                .sum()
        } else {
            0.0
        };

        Self {
            cardinality_ratio,
            entropy,
            run_length_avg: if non_null > 0 {
                run_length_sum as f64 / non_null as f64
            } else {
                1.0
            },
            run_length_max,
            null_ratio: null_count as f64 / actual_n as f64,
            range_log2: 0.0,
            sortedness: 0.0,
            temporal_score: 0.5,
            col_type: TYPE_UTF8,
            sample_size: actual_n,
        }
    }

    fn extract_generic(arr_len: usize, indices: &[usize], null_count: usize) -> Self {
        let actual_n = indices.len();
        Self {
            cardinality_ratio: 1.0,
            entropy: (arr_len as f64).log2().max(0.0),
            run_length_avg: 1.0,
            run_length_max: 1,
            null_ratio: null_count as f64 / actual_n as f64,
            range_log2: 0.0,
            sortedness: 0.0,
            temporal_score: 0.5,
            col_type: TYPE_OTHER,
            sample_size: actual_n,
        }
    }

    fn type_code(dt: &DataType) -> u8 {
        use DataType::*;
        match dt {
            Null => TYPE_NULL,
            Boolean => TYPE_BOOL,
            UInt8 => TYPE_U8,
            UInt16 => TYPE_U16,
            UInt32 => TYPE_U32,
            UInt64 => TYPE_U64,
            Int8 => TYPE_I8,
            Int16 => TYPE_I16,
            Int32 => TYPE_I32,
            Int64 => TYPE_I64,
            Float32 => TYPE_F32,
            Float64 => TYPE_F64,
            Utf8 | LargeUtf8 => TYPE_UTF8,
            Binary | LargeBinary => TYPE_BIN,
            _ => TYPE_OTHER,
        }
    }

    /// Convert features to a normalized feature vector for ML model input.
    pub fn to_feature_vec(&self) -> [f64; 9] {
        [
            self.cardinality_ratio.clamp(0.0, 1.0),
            self.entropy.clamp(0.0, 32.0) / 32.0,
            (self.run_length_avg.log2().clamp(0.0, 16.0) / 16.0),
            (self.run_length_max as f64).log2().clamp(0.0, 16.0) / 16.0,
            self.null_ratio.clamp(0.0, 1.0),
            self.range_log2.clamp(0.0, 64.0) / 64.0,
            self.sortedness.clamp(0.0, 1.0),
            self.temporal_score.clamp(0.0, 1.0),
            self.col_type as f64 / 255.0,
        ]
    }
}
