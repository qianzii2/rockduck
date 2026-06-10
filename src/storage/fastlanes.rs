//! FastLanes encoding wrappers for integer columns.
//!
//! FastLanes (CWI VLDB 2023) provides data-parallel, SIMD-friendly integer encodings.

use arrow_schema::DataType;

pub use crate::codec::column_encoding::EncodingScheme;

/// Encoding method recommendation from heuristic analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncodingMethod {
    BitPacking(u8),
    Delta,
    FoR,
}

/// FastLanes encoder — selects and applies the best integer encoding.
pub struct FastLanesEncoder;

impl FastLanesEncoder {
    /// Encode an Arrow integer array with BitPacking (best by default).
    /// Returns `(encoded_array, EncodingScheme)`.
    pub fn encode(
        arr: &dyn arrow_array::Array,
        ctx: &mut vortex_array::ExecutionCtx,
    ) -> crate::error::Result<(vortex_array::ArrayRef, EncodingScheme)> {
        use vortex_array::arrow::FromArrowArray;
        use vortex_array::IntoArray;
        use vortex_fastlanes::{initialize as fl_init, BitPackedData};

        // Register FastLanes plugins on the session
        fl_init(ctx.session());

        // Arrow → vortex ArrayRef → PrimitiveArray
        let vtx_ref = vortex_array::ArrayRef::from_arrow(arr, true)
            .map_err(|e| crate::error::RockDuckError::Codec(format!("Arrow→Vortex: {}", e)))?;

        match arr.data_type() {
            DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64 => {
                let w = compute_bit_width(arr) as u8;
                let packed = BitPackedData::encode(&vtx_ref, w, ctx)
                    .map_err(|e| crate::error::RockDuckError::Codec(format!("BitPacked: {}", e)))?;
                Ok((packed.into_array(), EncodingScheme::BitPacking))
            }
            _ => Err(crate::error::RockDuckError::Codec(format!(
                "FastLanes only supports integer types, got {:?}",
                arr.data_type()
            ))),
        }
    }

    pub fn encode_delta(
        arr: &dyn arrow_array::Array,
        ctx: &mut vortex_array::ExecutionCtx,
    ) -> crate::error::Result<(vortex_array::ArrayRef, EncodingScheme)> {
        use vortex_array::arrays::PrimitiveArray;
        use vortex_array::arrow::FromArrowArray;
        use vortex_array::IntoArray;
        use vortex_fastlanes::{initialize as fl_init_delta, Delta};

        fl_init_delta(ctx.session());

        let vtx_ref = vortex_array::ArrayRef::from_arrow(arr, true)
            .map_err(|e| crate::error::RockDuckError::Codec(format!("Arrow→Vortex: {}", e)))?;

        let prim_arr: PrimitiveArray = vtx_ref
            .try_downcast()
            .map_err(|_| crate::error::RockDuckError::Codec("expected PrimitiveArray".into()))?;

        let delta_arr = Delta::try_from_primitive_array(&prim_arr, ctx)
            .map_err(|e| crate::error::RockDuckError::Codec(format!("Delta: {}", e)))?;
        Ok((delta_arr.into_array(), EncodingScheme::Delta))
    }

    pub fn encode_for(
        arr: &dyn arrow_array::Array,
        ctx: &mut vortex_array::ExecutionCtx,
    ) -> crate::error::Result<(vortex_array::ArrayRef, EncodingScheme)> {
        use vortex_array::arrays::PrimitiveArray;
        use vortex_array::arrow::FromArrowArray;
        use vortex_array::IntoArray;
        use vortex_fastlanes::{initialize as fl_init_for, FoR};

        fl_init_for(ctx.session());

        let vtx_ref = vortex_array::ArrayRef::from_arrow(arr, true)
            .map_err(|e| crate::error::RockDuckError::Codec(format!("Arrow→Vortex: {}", e)))?;

        let prim_arr: PrimitiveArray = vtx_ref
            .try_downcast()
            .map_err(|_| crate::error::RockDuckError::Codec("expected PrimitiveArray".into()))?;

        let for_arr = FoR::encode(prim_arr)
            .map_err(|e| crate::error::RockDuckError::Codec(format!("FoR: {}", e)))?;
        Ok((for_arr.into_array(), EncodingScheme::FOR))
    }
}

fn compute_bit_width(arr: &dyn arrow_array::Array) -> usize {
    let total = arr.len();
    if total == 0 {
        return 1;
    }

    let sample_size = 10000.min(total);
    let step = total / sample_size;
    if step == 0 {
        return 1;
    }

    let is_signed = matches!(
        arr.data_type(),
        DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64
    );

    let mut min_val = i128::MAX;
    let mut max_val = i128::MIN;

    for i in (0..total).step_by(step) {
        let val = extract_i128(arr, i);
        min_val = min_val.min(val);
        max_val = max_val.max(val);
    }

    let range = max_val - min_val;
    if range == 0 {
        return 1;
    }

    // Use saturating arithmetic to avoid overflow for extreme i128 values
    if is_signed {
        let needed = range.saturating_add(1);
        let bits = (needed.ilog2() + 1).max(1) as usize;
        bits.min(64)
    } else {
        let bits = (range.ilog2() + 1).max(1) as usize;
        bits.min(64)
    }
}

fn extract_i128(arr: &dyn arrow_array::Array, idx: usize) -> i128 {
    match arr.data_type() {
        DataType::Int8 => arr
            .as_any()
            .downcast_ref::<arrow_array::Int8Array>()
            .map(|a| a.value(idx) as i128)
            .unwrap_or(0),
        DataType::Int16 => arr
            .as_any()
            .downcast_ref::<arrow_array::Int16Array>()
            .map(|a| a.value(idx) as i128)
            .unwrap_or(0),
        DataType::Int32 => arr
            .as_any()
            .downcast_ref::<arrow_array::Int32Array>()
            .map(|a| a.value(idx) as i128)
            .unwrap_or(0),
        DataType::Int64 => arr
            .as_any()
            .downcast_ref::<arrow_array::Int64Array>()
            .map(|a| a.value(idx) as i128)
            .unwrap_or(0),
        DataType::UInt8 => arr
            .as_any()
            .downcast_ref::<arrow_array::UInt8Array>()
            .map(|a| a.value(idx) as i128)
            .unwrap_or(0),
        DataType::UInt16 => arr
            .as_any()
            .downcast_ref::<arrow_array::UInt16Array>()
            .map(|a| a.value(idx) as i128)
            .unwrap_or(0),
        DataType::UInt32 => arr
            .as_any()
            .downcast_ref::<arrow_array::UInt32Array>()
            .map(|a| a.value(idx) as i128)
            .unwrap_or(0),
        DataType::UInt64 => arr
            .as_any()
            .downcast_ref::<arrow_array::UInt64Array>()
            .map(|a| a.value(idx) as i128)
            .unwrap_or(0),
        _ => 0,
    }
}
