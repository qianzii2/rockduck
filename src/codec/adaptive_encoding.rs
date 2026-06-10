//! Adaptive column encoding — static heuristics and encoding selection.
//!
//! # Three-Level Strategy
//!
//! ## Level 1: Static Heuristics (type-based, O(1))
//!
//! Infer encoding from Arrow data type alone — no data scanning required.
//! Used as the initial recommendation; overridden by L2/L3 if needed.
//!
//! ## Level 2: Block-Level Adaptive
//!
//! For each block, compute lightweight statistics (cardinality, sortedness)
//! and select encoding. This is delegated to `vortex-btrblocks` which handles
//! block-level selection automatically.
//!
//! ## Level 3: LEA — Learned Encoding Advisor
//!
//! ML-based cost model trained on data distribution + query workload.
//! Provides confidence-weighted encoding recommendations.

/// Threshold for dictionary encoding of string columns.
/// Strings with fewer than this many distinct values benefit from dictionary encoding.
const DICTIONARY_ENCODING_CARDINALITY_THRESHOLD: usize = 1000;

use arrow_array::Array;
use arrow_schema::DataType;

pub use crate::codec::column_encoding::EncodingScheme;

/// Downcast an ArrayRef to a concrete Arrow array type.
/// Returns an error with the actual type instead of panicking.
fn downcast_array<'a, T: arrow_array::Array + 'static>(
    arr: &'a dyn Array,
    type_name: &str,
) -> crate::error::Result<&'a T> {
    arr.as_any().downcast_ref::<T>().ok_or_else(|| {
        crate::RockDuckError::Codec(format!(
            "Expected {} array, got {:?}",
            type_name,
            arr.data_type()
        ))
    })
}

/// L1: Select encoding scheme based on Arrow data type alone.
/// This is a fast O(1) path — no data scanning required.
pub fn analyze_and_select(arr: &dyn Array) -> crate::error::Result<EncodingScheme> {
    use DataType::*;
    Ok(match arr.data_type() {
        // Float types → ALP (Adaptive Lossless compression for floats)
        Float32 | Float64 => EncodingScheme::Alp,

        // Signed integers → BitPacking (ZigZag pre-cond handled by BtrBlocks)
        Int8 | Int16 | Int32 | Int64 => EncodingScheme::BitPacking,

        // Unsigned integers → BitPacking (handled by BtrBlocks)
        UInt8 | UInt16 | UInt32 | UInt64 => EncodingScheme::BitPacking,

        // Low-cardinality strings → Dictionary (handled by BtrBlocks)
        Utf8 | LargeUtf8 => EncodingScheme::Dictionary,

        // Binary types → default to raw
        Binary | LargeBinary | BinaryView | Utf8View => EncodingScheme::Raw,

        // All other types → Raw
        _ => EncodingScheme::Raw,
    })
}

/// L2: Analyze a sample of values and refine the encoding recommendation.
/// Uses lightweight O(sample_size) analysis to check for sortedness and cardinality.
pub fn refine_with_sample(
    arr: &dyn Array,
    sample_size: usize,
) -> crate::error::Result<EncodingScheme> {
    use DataType::*;
    use EncodingScheme::*;

    let dtype = arr.data_type();

    // Float types: always ALP
    if matches!(dtype, Float32 | Float64) {
        return Ok(Alp);
    }

    // For integers, check if values are sorted
    if matches!(
        dtype,
        Int8 | Int16 | Int32 | Int64 | UInt8 | UInt16 | UInt32 | UInt64
    ) {
        if is_sorted_sample(arr, sample_size) {
            return Ok(Delta);
        }
        return Ok(BitPacking);
    }

    // For strings, estimate cardinality from sample
    if matches!(dtype, Utf8 | LargeUtf8) {
        let card = estimate_cardinality(arr, sample_size)?;
        if card < DICTIONARY_ENCODING_CARDINALITY_THRESHOLD {
            return Ok(EncodingScheme::Dictionary);
        }
        return Ok(EncodingScheme::Raw);
    }

    // Default: use L1 suggestion
    analyze_and_select(arr)
}

/// Estimate cardinality from a sample of the array.
fn estimate_cardinality(arr: &dyn Array, sample_size: usize) -> crate::error::Result<usize> {
    let arr_len = arr.len();
    if arr_len == 0 {
        return Ok(0);
    }

    let n = arr_len.min(sample_size);
    let step = if arr_len <= n { 1 } else { arr_len / n };

    let card = match arr.data_type() {
        DataType::Int32 => {
            let downcast = downcast_array::<arrow_array::Int32Array>(arr, "Int32")?;
            let mut unique = std::collections::HashSet::new();
            for i in (0..arr_len).step_by(step) {
                if !downcast.is_null(i) {
                    unique.insert(downcast.value(i));
                }
            }
            unique.len()
        }
        DataType::Int64 => {
            let downcast = downcast_array::<arrow_array::Int64Array>(arr, "Int64")?;
            let mut unique = std::collections::HashSet::new();
            for i in (0..arr_len).step_by(step) {
                if !downcast.is_null(i) {
                    unique.insert(downcast.value(i));
                }
            }
            unique.len()
        }
        DataType::UInt64 => {
            let downcast = downcast_array::<arrow_array::UInt64Array>(arr, "UInt64")?;
            let mut unique = std::collections::HashSet::new();
            for i in (0..arr_len).step_by(step) {
                if !downcast.is_null(i) {
                    unique.insert(downcast.value(i));
                }
            }
            unique.len()
        }
        DataType::Utf8 => {
            let downcast = downcast_array::<arrow_array::StringArray>(arr, "Utf8")?;
            let mut unique = std::collections::HashSet::new();
            for i in (0..arr_len).step_by(step) {
                if !downcast.is_null(i) {
                    unique.insert(downcast.value(i).to_string());
                }
            }
            unique.len()
        }
        _ => arr_len,
    };

    // Extrapolate cardinality from sample
    let full_est = (card as f64) * (arr_len as f64 / n as f64);
    Ok(full_est.ceil() as usize)
}

/// Check if the first N rows of the array appear sorted.
/// Used for detecting monotonic sequences (Delta encoding candidate).
fn is_sorted_sample(arr: &dyn Array, sample_size: usize) -> bool {
    use DataType::*;

    let n = arr.len().min(sample_size).saturating_sub(1);
    if n == 0 {
        return false;
    }

    // Returns None on type mismatch; caller treats None as "not sorted"
    fn check_sorted<T: arrow_array::Array + 'static, F>(
        arr: &dyn Array,
        n: usize,
        f: F,
    ) -> Option<bool>
    where
        F: Fn(&T, usize) -> bool,
    {
        let downcast = arr.as_any().downcast_ref::<T>()?;
        Some((0..n).all(|i| f(downcast, i)))
    }

    match arr.data_type() {
        Int32 => {
            check_sorted::<arrow_array::Int32Array, _>(arr, n, |a, i| a.value(i) <= a.value(i + 1))
                .unwrap_or(false)
        }
        Int64 => {
            check_sorted::<arrow_array::Int64Array, _>(arr, n, |a, i| a.value(i) <= a.value(i + 1))
                .unwrap_or(false)
        }
        UInt64 => {
            check_sorted::<arrow_array::UInt64Array, _>(arr, n, |a, i| a.value(i) <= a.value(i + 1))
                .unwrap_or(false)
        }
        Float64 => check_sorted::<arrow_array::Float64Array, _>(arr, n, |a, i| {
            let l = a.value(i);
            let r = a.value(i + 1);
            l <= r || l.is_nan() || r.is_nan()
        })
        .unwrap_or(false),
        _ => false,
    }
}

/// Returns true if the array type supports ALP encoding.
pub fn supports_alp(dt: &DataType) -> bool {
    matches!(dt, DataType::Float32 | DataType::Float64)
}

/// Returns true if the array type supports Delta encoding.
pub fn supports_delta(dt: &DataType) -> bool {
    matches!(
        dt,
        DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
    )
}

/// Returns true if the array type supports Dictionary encoding.
pub fn supports_dictionary(dt: &DataType) -> bool {
    matches!(
        dt,
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Binary | DataType::LargeBinary
    )
}

// =============================================================================
// Encoding stubs — implemented by Vortex when needed
// =============================================================================

/// Encode a column array with the specified scheme.
/// Delegates to the Vortex encoding pipeline (ALP, BitPacked, Delta, etc.).
#[allow(dead_code, unused_variables)]
pub fn encode_block(arr: &dyn Array, scheme: EncodingScheme) -> crate::error::Result<Vec<u8>> {
    let _ = (arr, scheme);
    Err(crate::error::RockDuckError::Internal(
        "encode_block: use VortexWriter with BtrBlocks instead".into(),
    ))
}

/// Re-encode an existing column file with a new encoding scheme.
/// Reads the existing Vortex data, re-encodes, writes to a temp file, atomically renames.
#[allow(dead_code, unused_variables)]
pub fn reencode_segment_atomic(
    seg_id: &str,
    col_name: &str,
    new_scheme: EncodingScheme,
) -> crate::error::Result<()> {
    let _ = (seg_id, col_name, new_scheme);
    Err(crate::error::RockDuckError::Internal(
        "reencode_segment_atomic: not yet implemented".into(),
    ))
}
