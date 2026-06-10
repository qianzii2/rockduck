//! Vortex ALP + FastLanes Phase B — Adaptive Lossless compression.
//!
//! 1. **Heuristic** (`alp_is_worth_it`, `choose_integer_encoding`)
//! 2. **Encoding** (`alp_encode_float`, `FastLanesEncoder`)
//!
//! ALP (SIGMOD 2024): decimal floats → integers via shared exponent, 2-10x compression.
//! FastLanes (VLDB 2023): "virtual 1024-bit SIMD" design, pure scalar auto-vectorizing.

use arrow_array::Array;
use arrow_schema::DataType;

pub use crate::codec::column_encoding::{EncodingScheme, TableEncodingConfig};
pub use crate::storage::fastlanes::{EncodingMethod, FastLanesEncoder};
pub use vortex_alp::ALPArray;

// =============================================================================
// Heuristic: ALP worth-it analysis
// =============================================================================

/// Returns estimated compression ratio for ALP on a float column.
/// > 1.0 = ALP will compress. 1.0 = not worth it.
/// > Uses 5% stratified sampling, min 1000 rows, max 10000.
pub fn alp_is_worth_it(arr: &dyn arrow_array::Array) -> f64 {
    match arr.data_type() {
        DataType::Float32 | DataType::Float64 => sample_and_decide(arr),
        _ => 1.0,
    }
}

fn sample_and_decide(arr: &dyn arrow_array::Array) -> f64 {
    let total = arr.len();
    if total == 0 {
        return 1.0;
    }

    let sample_size = (total / 20).clamp(1000, 10000);
    if sample_size == 0 {
        return 1.0;
    }
    let step = total / sample_size;
    if step == 0 {
        return 1.0;
    }

    let mut decimal_like = 0i64;
    let mut total_sampled = 0i64;
    let is_f64 = matches!(arr.data_type(), DataType::Float64);

    for i in (0..total).step_by(step) {
        let val = if is_f64 {
            arr.as_any()
                .downcast_ref::<arrow_array::Float64Array>()
                .and_then(|a| {
                    if a.is_valid(i) {
                        Some(a.value(i))
                    } else {
                        None
                    }
                })
                .unwrap_or(0.0)
        } else {
            arr.as_any()
                .downcast_ref::<arrow_array::Float32Array>()
                .and_then(|a| {
                    if a.is_valid(i) {
                        Some(a.value(i) as f64)
                    } else {
                        None
                    }
                })
                .unwrap_or(0.0)
        };

        let exp = ieee754_exponent(val);
        if (-6..=6).contains(&exp) {
            decimal_like += 1;
        }
        total_sampled += 1;
    }

    if total_sampled == 0 {
        return 1.0;
    }

    let decimal_ratio = decimal_like as f64 / total_sampled as f64;

    if decimal_ratio > 0.6 {
        4.0
    } else if decimal_ratio > 0.3 {
        2.0
    } else {
        1.0
    }
}

/// Extract the unbiased exponent from an IEEE 754 double.
#[inline]
fn ieee754_exponent(val: f64) -> i32 {
    let bits = val.to_bits();
    let exp_raw = ((bits >> 52) & 0x7FF) as i32;
    if exp_raw == 0x7FF || exp_raw == 0 {
        i32::MAX
    } else {
        exp_raw - 1023
    }
}

// =============================================================================
// Heuristic: Integer encoding selection
// =============================================================================

/// Returns the recommended integer encoding method based on data sampling.
pub fn choose_integer_encoding(arr: &dyn arrow_array::Array) -> EncodingMethod {
    let total = arr.len();
    if total == 0 {
        return EncodingMethod::BitPacking(1);
    }

    let sample_size = 10000.min(total);
    let step = total / sample_size;
    if step == 0 {
        return EncodingMethod::BitPacking(1);
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
        return EncodingMethod::BitPacking(1);
    }

    // Use checked arithmetic to avoid overflow (especially for i128::MIN)
    let w = if is_signed {
        // For signed integers, the range maps to [-2^(w-1), 2^(w-1)-1]
        // So we need w bits where 2^w > range + 1
        let needed = range.saturating_add(1); // range + 1 without overflow
        let bits = (needed.ilog2() + 1).max(1) as u8;
        bits.min(64)
    } else {
        // For unsigned, 2^w > range
        let bits = (range.ilog2() + 1).max(1) as u8;
        bits.min(64)
    };

    if w <= 32 {
        EncodingMethod::BitPacking(w)
    } else if w <= 64 {
        EncodingMethod::Delta
    } else {
        EncodingMethod::FoR
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

// =============================================================================
// ALP encode / decode
// =============================================================================

/// Encode a float Arrow array with ALP.
/// Returns an ArrayRef (ALP-encoded) that can be written to a Vortex file.
/// Decoding is automatic via VortexFile (self-describing format).
/// Uses the provided ExecutionCtx; caller should ensure it has the ALP plugin registered.
pub fn alp_encode_float(
    arr: &dyn arrow_array::Array,
    ctx: &mut vortex_array::ExecutionCtx,
) -> crate::error::Result<vortex_array::ArrayRef> {
    match arr.data_type() {
        DataType::Float32 => {
            let f32_arr = arr
                .as_any()
                .downcast_ref::<arrow_array::Float32Array>()
                .ok_or_else(|| {
                    crate::error::RockDuckError::Codec(
                        "expected Float32Array for Float32 encoding".into(),
                    )
                })?;

            // Build a nullable Float64Array, preserving validity from the source.
            let mut f64_builder = arrow_array::builder::Float64Builder::with_capacity(arr.len());
            for i in 0..arr.len() {
                if f32_arr.is_valid(i) {
                    f64_builder.append_value(f32_arr.value(i) as f64);
                } else {
                    f64_builder.append_null();
                }
            }
            let f64_arr = f64_builder.finish();

            encode_float_unchecked(&f64_arr, ctx)
        }
        DataType::Float64 => encode_float_unchecked(arr, ctx),
        _ => Err(crate::error::RockDuckError::Codec(format!(
            "alp_encode_float only supports f32/f64, got {:?}",
            arr.data_type()
        ))),
    }
}

fn encode_float_unchecked(
    arr: &dyn arrow_array::Array,
    ctx: &mut vortex_array::ExecutionCtx,
) -> crate::error::Result<vortex_array::ArrayRef> {
    use vortex_alp::alp_encode;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrow::FromArrowArray;
    use vortex_array::IntoArray;

    let vtx_ref = vortex_array::ArrayRef::from_arrow(arr, true)
        .map_err(|e| crate::error::RockDuckError::Codec(format!("Arrow→Vortex: {}", e)))?;

    let prim_arr: PrimitiveArray = vtx_ref.try_downcast().map_err(|_| {
        crate::error::RockDuckError::Codec("expected PrimitiveArray from Arrow conversion".into())
    })?;

    let alp = alp_encode(prim_arr.as_view(), None, ctx)
        .map_err(|e| crate::error::RockDuckError::Codec(format!("ALP encode: {}", e)))?;

    Ok(alp.into_array())
}

/// Decode an ALPArray back to a Vec<f64>.
/// For production, prefer the VortexFile reader path which auto-decodes.
#[allow(deprecated)]
pub fn alp_decode_float(alp_arr: ALPArray) -> crate::error::Result<Vec<f64>> {
    use vortex_alp::decompress_into_array;
    use vortex_array::arrow::ArrowArrayExecutor;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;

    let mut ctx = VortexSessionExecute::create_execution_ctx(&*vortex_array::LEGACY_SESSION);

    // Decode: ALPArray → Vortex PrimitiveArray
    let prim = decompress_into_array(alp_arr, &mut ctx)
        .map_err(|e| crate::error::RockDuckError::Codec(format!("ALP decode: {}", e)))?;

    // PrimitiveArray → Arrow ArrayRef
    let arrow_arr: arrow_array::ArrayRef = prim
        .into_array()
        .execute_arrow(None, &mut ctx)
        .map_err(|e| crate::error::RockDuckError::Codec(format!("execute_arrow: {}", e)))?;

    let float_arr = arrow_arr
        .as_any()
        .downcast_ref::<arrow_array::Float64Array>()
        .ok_or_else(|| {
            crate::error::RockDuckError::Codec("decoded array is not Float64Array".into())
        })?;

    Ok(float_arr.values().to_vec())
}

// =============================================================================
// AdaptiveEncodingPicker — wires ALP + FastLanes into VortexWriter
// =============================================================================

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;

/// 自适应编码选择器 — 根据数据类型和数据特征选择编码。
/// 优先使用 ALP (float) / FastLanes (int)，回退到 BtrBlocks。
pub struct AdaptiveEncodingPicker {
    #[allow(unused)]
    config: TableEncodingConfig,
}

impl AdaptiveEncodingPicker {
    pub fn new(config: TableEncodingConfig) -> Self {
        Self { config }
    }

    /// Pick the best encoding and apply it.
    /// Caller must ensure `ctx` has ALP and FastLanes plugins registered via
    /// `vortex_alp::initialize(ctx.session())` and `vortex_fastlanes::initialize(ctx.session())`.
    /// Returns (encoded_array_ref, scheme).
    pub fn pick_and_encode(
        &self,
        arr: &dyn arrow_array::Array,
        ctx: &mut ExecutionCtx,
    ) -> crate::error::Result<(ArrayRef, EncodingScheme)> {
        match arr.data_type() {
            DataType::Float32 | DataType::Float64 => {
                let ratio = alp_is_worth_it(arr);
                if ratio > 1.0 {
                    let alp_arr = alp_encode_float(arr, ctx)?;
                    Ok((alp_arr, EncodingScheme::Alp))
                } else {
                    self.encode_btrblocks(arr, ctx)
                }
            }
            DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64 => {
                let method = choose_integer_encoding(arr);
                match method {
                    EncodingMethod::BitPacking(_w) => {
                        let packed = FastLanesEncoder::encode(arr, ctx)?;
                        Ok((packed.0, EncodingScheme::BitPacking))
                    }
                    EncodingMethod::Delta => {
                        let delta = FastLanesEncoder::encode_delta(arr, ctx)?;
                        Ok((delta.0, EncodingScheme::Delta))
                    }
                    EncodingMethod::FoR => {
                        let for_arr = FastLanesEncoder::encode_for(arr, ctx)?;
                        Ok((for_arr.0, EncodingScheme::FOR))
                    }
                }
            }
            _ => self.encode_btrblocks(arr, ctx),
        }
    }

    #[allow(dead_code)]
    fn encode_with_scheme(
        &self,
        arr: &dyn arrow_array::Array,
        ctx: &mut ExecutionCtx,
        scheme: EncodingScheme,
    ) -> crate::error::Result<(ArrayRef, EncodingScheme)> {
        match scheme {
            EncodingScheme::Alp | EncodingScheme::AlpRD => {
                let alp_arr = alp_encode_float(arr, ctx)?;
                Ok((alp_arr, EncodingScheme::Alp))
            }
            EncodingScheme::BitPacking => {
                let packed = FastLanesEncoder::encode(arr, ctx)?;
                Ok((packed.0, EncodingScheme::BitPacking))
            }
            EncodingScheme::Delta => {
                let delta = FastLanesEncoder::encode_delta(arr, ctx)?;
                Ok((delta.0, EncodingScheme::Delta))
            }
            EncodingScheme::FOR => {
                let for_arr = FastLanesEncoder::encode_for(arr, ctx)?;
                Ok((for_arr.0, EncodingScheme::FOR))
            }
            _ => self.encode_btrblocks(arr, ctx),
        }
    }

    fn encode_btrblocks(
        &self,
        arr: &dyn arrow_array::Array,
        ctx: &mut ExecutionCtx,
    ) -> crate::error::Result<(ArrayRef, EncodingScheme)> {
        use vortex_array::arrow::FromArrowArray;

        let vtx_arr = vortex_array::ArrayRef::from_arrow(arr, true)
            .map_err(|e| crate::error::RockDuckError::Codec(format!("Arrow→Vortex: {}", e)))?;

        let compressor = vortex_btrblocks::BtrBlocksCompressor::default();
        let comp = compressor
            .compress(&vtx_arr, ctx)
            .map_err(|e| crate::error::RockDuckError::Codec(format!("BtrBlocks: {}", e)))?;

        Ok((comp, EncodingScheme::BtrBlocks))
    }
}
