//! Block-level statistics computation for Zone Maps.
//!
//! Computes min/max/null_count per column block slice.
//! Uses concrete typed arrays (Int64Array, Float64Array, etc.) with no generics.

use arrow_array::{
    types::Int32Type, Array, BooleanArray, DictionaryArray, Float32Array, Float64Array, Int16Array,
    Int32Array, Int64Array, Int8Array, LargeStringArray, StringArray, UInt16Array, UInt32Array,
    UInt64Array, UInt8Array,
};

/// Result of computing statistics for a column slice within one block.
#[derive(Debug, Clone)]
pub struct BlockColumnStats {
    pub min_bytes: Option<Vec<u8>>,
    pub max_bytes: Option<Vec<u8>>,
    pub null_count: u32,
}

impl BlockColumnStats {
    pub fn merge_with(&mut self, other: &BlockColumnStats) {
        self.null_count += other.null_count;
        if let Some(ref other_min) = other.min_bytes {
            let dominated = self
                .min_bytes
                .as_ref()
                .is_none_or(|cur| cur.as_slice() > other_min.as_slice());
            if dominated {
                self.min_bytes = Some(other_min.clone());
            }
        }
        if let Some(ref other_max) = other.max_bytes {
            let dominated = self
                .max_bytes
                .as_ref()
                .is_none_or(|cur| cur.as_slice() < other_max.as_slice());
            if dominated {
                self.max_bytes = Some(other_max.clone());
            }
        }
    }

    pub fn merge_all(stats: &[BlockColumnStats]) -> BlockColumnStats {
        let mut result = BlockColumnStats {
            min_bytes: None,
            max_bytes: None,
            null_count: 0,
        };
        for s in stats {
            result.merge_with(s);
        }
        result
    }
}

macro_rules! int_stats_from {
    ($arr:expr, $offset:expr, $len:expr, $non_null:expr) => {{
        let vals = $arr.values();
        let start = $offset;
        let end = ($offset + $len).min(vals.len());
        if start >= end || $non_null == 0 {
            return BlockColumnStats {
                min_bytes: None,
                max_bytes: None,
                null_count: 0,
            };
        }
        // Find first non-null value to initialize min/max
        let mut first_valid: Option<_> = None;
        for i in start..end {
            if !$arr.is_null(i) {
                first_valid = Some(i);
                break;
            }
        }
        let Some(first) = first_valid else {
            return BlockColumnStats {
                min_bytes: None,
                max_bytes: None,
                null_count: 0,
            };
        };
        let mut mn = vals[first];
        let mut mx = vals[first];
        for i in (first + 1)..end {
            if !$arr.is_null(i) {
                let v = vals[i];
                if v < mn {
                    mn = v;
                }
                if v > mx {
                    mx = v;
                }
            }
        }
        BlockColumnStats {
            min_bytes: Some(mn.to_le_bytes().to_vec()),
            max_bytes: Some(mx.to_le_bytes().to_vec()),
            null_count: 0,
        }
    }};
}

macro_rules! float_stats_from {
    ($arr:expr, $offset:expr, $len:expr, $non_null:expr) => {{
        let vals = $arr.values();
        let start = $offset;
        let end = ($offset + $len).min(vals.len());
        if start >= end || $non_null == 0 {
            return BlockColumnStats {
                min_bytes: None,
                max_bytes: None,
                null_count: (($len as u32) - ($non_null as u32)),
            };
        }
        let mut mn: Option<_> = None;
        let mut mx: Option<_> = None;
        for i in start..end {
            if !$arr.is_null(i) {
                let v = vals[i];
                if !v.is_nan() {
                    mn = Some(mn.map_or(v, |m| if v < m { v } else { m }));
                    mx = Some(mx.map_or(v, |m| if v > m { v } else { m }));
                }
            }
        }
        BlockColumnStats {
            min_bytes: mn.map(|v| v.to_le_bytes().to_vec()),
            max_bytes: mx.map(|v| v.to_le_bytes().to_vec()),
            // Fix M-6: now correctly reports null_count instead of always 0
            null_count: ($len as u32).saturating_sub($non_null as u32),
        }
    }};
}

/// Compute min/max/null_count for a column array slice [offset, offset + len).
pub fn compute_block_stats(arr: &dyn Array, offset: usize, len: usize) -> BlockColumnStats {
    if len == 0 {
        return BlockColumnStats {
            min_bytes: None,
            max_bytes: None,
            null_count: 0,
        };
    }

    // Count nulls in the slice range
    let mut null_count = 0u32;
    for i in offset..(offset + len).min(arr.len()) {
        if arr.is_null(i) {
            null_count += 1;
        }
    }
    // Number of non-null values
    let non_null_count = len.saturating_sub(null_count as usize);

    match arr.data_type() {
        // Float types
        arrow_schema::DataType::Float32 => {
            if let Some(a) = arr.as_any().downcast_ref::<Float32Array>() {
                let mut s = float_stats_from!(a, offset, len, non_null_count);
                s.null_count = null_count;
                return s;
            }
            BlockColumnStats {
                min_bytes: None,
                max_bytes: None,
                null_count,
            }
        }
        arrow_schema::DataType::Float64 => {
            if let Some(a) = arr.as_any().downcast_ref::<Float64Array>() {
                let mut s = float_stats_from!(a, offset, len, non_null_count);
                s.null_count = null_count;
                return s;
            }
            BlockColumnStats {
                min_bytes: None,
                max_bytes: None,
                null_count,
            }
        }
        // Signed integers
        arrow_schema::DataType::Int8 => {
            if let Some(a) = arr.as_any().downcast_ref::<Int8Array>() {
                let mut s = int_stats_from!(a, offset, len, non_null_count);
                s.null_count = null_count;
                return s;
            }
            BlockColumnStats {
                min_bytes: None,
                max_bytes: None,
                null_count,
            }
        }
        arrow_schema::DataType::Int16 => {
            if let Some(a) = arr.as_any().downcast_ref::<Int16Array>() {
                let mut s = int_stats_from!(a, offset, len, non_null_count);
                s.null_count = null_count;
                return s;
            }
            BlockColumnStats {
                min_bytes: None,
                max_bytes: None,
                null_count,
            }
        }
        arrow_schema::DataType::Int32 => {
            if let Some(a) = arr.as_any().downcast_ref::<Int32Array>() {
                let mut s = int_stats_from!(a, offset, len, non_null_count);
                s.null_count = null_count;
                return s;
            }
            BlockColumnStats {
                min_bytes: None,
                max_bytes: None,
                null_count,
            }
        }
        arrow_schema::DataType::Int64 => {
            if let Some(a) = arr.as_any().downcast_ref::<Int64Array>() {
                let mut s = int_stats_from!(a, offset, len, non_null_count);
                s.null_count = null_count;
                return s;
            }
            BlockColumnStats {
                min_bytes: None,
                max_bytes: None,
                null_count,
            }
        }
        // Unsigned integers
        arrow_schema::DataType::UInt8 => {
            if let Some(a) = arr.as_any().downcast_ref::<UInt8Array>() {
                let mut s = int_stats_from!(a, offset, len, non_null_count);
                s.null_count = null_count;
                return s;
            }
            BlockColumnStats {
                min_bytes: None,
                max_bytes: None,
                null_count,
            }
        }
        arrow_schema::DataType::UInt16 => {
            if let Some(a) = arr.as_any().downcast_ref::<UInt16Array>() {
                let mut s = int_stats_from!(a, offset, len, non_null_count);
                s.null_count = null_count;
                return s;
            }
            BlockColumnStats {
                min_bytes: None,
                max_bytes: None,
                null_count,
            }
        }
        arrow_schema::DataType::UInt32 => {
            if let Some(a) = arr.as_any().downcast_ref::<UInt32Array>() {
                let mut s = int_stats_from!(a, offset, len, non_null_count);
                s.null_count = null_count;
                return s;
            }
            BlockColumnStats {
                min_bytes: None,
                max_bytes: None,
                null_count,
            }
        }
        arrow_schema::DataType::UInt64 => {
            if let Some(a) = arr.as_any().downcast_ref::<UInt64Array>() {
                let mut s = int_stats_from!(a, offset, len, non_null_count);
                s.null_count = null_count;
                return s;
            }
            BlockColumnStats {
                min_bytes: None,
                max_bytes: None,
                null_count,
            }
        }
        // Dictionary
        arrow_schema::DataType::Dictionary(key_type, _)
            if matches!(key_type.as_ref(), &arrow_schema::DataType::Int32) =>
        {
            if let Some(dict) = arr.as_any().downcast_ref::<DictionaryArray<Int32Type>>() {
                let codes = dict.keys();
                let start = offset;
                let end = (offset + len).min(codes.len());
                if start < end && !codes.is_empty() {
                    if codes.is_null(start) {
                        return BlockColumnStats {
                            min_bytes: None,
                            max_bytes: None,
                            null_count,
                        };
                    }
                    let mut mn = codes.value(start);
                    let mut mx = codes.value(start);
                    for i in (start + 1)..end {
                        if codes.is_null(i) {
                            continue;
                        }
                        let v = codes.value(i);
                        if v < mn {
                            mn = v;
                        }
                        if v > mx {
                            mx = v;
                        }
                    }
                    return BlockColumnStats {
                        min_bytes: Some(mn.to_le_bytes().to_vec()),
                        max_bytes: Some(mx.to_le_bytes().to_vec()),
                        null_count,
                    };
                }
            }
            BlockColumnStats {
                min_bytes: None,
                max_bytes: None,
                null_count,
            }
        }
        // Boolean
        arrow_schema::DataType::Boolean => {
            if let Some(a) = arr.as_any().downcast_ref::<BooleanArray>() {
                let mut has_true = false;
                let mut has_false = false;
                for i in offset..(offset + len).min(a.len()) {
                    if !a.is_null(i) {
                        if a.value(i) {
                            has_true = true;
                        } else {
                            has_false = true;
                        }
                    }
                }
                let min_bytes = if has_false { Some(vec![0u8]) } else { None };
                let max_bytes = if has_true { Some(vec![1u8]) } else { None };
                return BlockColumnStats {
                    min_bytes,
                    max_bytes,
                    null_count,
                };
            }
            BlockColumnStats {
                min_bytes: None,
                max_bytes: None,
                null_count,
            }
        }
        // Strings
        arrow_schema::DataType::Utf8 => {
            if let Some(a) = arr.as_any().downcast_ref::<StringArray>() {
                let mut min_str: Option<&str> = None;
                let mut max_str: Option<&str> = None;
                for i in offset..(offset + len).min(a.len()) {
                    if !a.is_null(i) {
                        let s = a.value(i);
                        min_str = Some(match min_str {
                            Some(m) if s < m => s,
                            Some(m) => m,
                            None => s,
                        });
                        max_str = Some(match max_str {
                            Some(m) if s > m => s,
                            Some(m) => m,
                            None => s,
                        });
                    }
                }
                return BlockColumnStats {
                    min_bytes: min_str.map(|s| s.as_bytes().to_vec()),
                    max_bytes: max_str.map(|s| s.as_bytes().to_vec()),
                    null_count,
                };
            }
            BlockColumnStats {
                min_bytes: None,
                max_bytes: None,
                null_count,
            }
        }
        arrow_schema::DataType::LargeUtf8 => {
            if let Some(a) = arr.as_any().downcast_ref::<LargeStringArray>() {
                let mut min_str: Option<&str> = None;
                let mut max_str: Option<&str> = None;
                for i in offset..(offset + len).min(a.len()) {
                    if !a.is_null(i) {
                        let s = a.value(i);
                        min_str = Some(match min_str {
                            Some(m) if s < m => s,
                            Some(m) => m,
                            None => s,
                        });
                        max_str = Some(match max_str {
                            Some(m) if s > m => s,
                            Some(m) => m,
                            None => s,
                        });
                    }
                }
                return BlockColumnStats {
                    min_bytes: min_str.map(|s| s.as_bytes().to_vec()),
                    max_bytes: max_str.map(|s| s.as_bytes().to_vec()),
                    null_count,
                };
            }
            BlockColumnStats {
                min_bytes: None,
                max_bytes: None,
                null_count,
            }
        }
        // Fallback
        _ => BlockColumnStats {
            min_bytes: None,
            max_bytes: None,
            null_count,
        },
    }
}

/// Convenience: compute stats for an entire array (no slicing).
pub fn compute_full_stats(arr: &dyn Array) -> BlockColumnStats {
    compute_block_stats(arr, 0, arr.len())
}
