//! Binary serialization utilities
//!
//! Provides serialization backends:
//! - **Default (`encode`/`decode`)**: postcard — serde-compatible, no_std-friendly binary format.
//!   postcard is used for WAL payloads and general-purpose serialization.
//! - **Adaptive encoding** (via `adaptive_encoding`): Block-level encoding selection.

pub mod adaptive_encoding;
mod timestamp; // TODO[ENCODING]: Adaptive column encoding (ALP, Delta, RLE, etc.)

pub use timestamp::{current_timestamp_millis, current_timestamp_secs};

// Re-export adapters
pub mod column_encoding;
pub mod corr_detector;
pub mod lea_features;
pub mod postcard_adapter;
pub mod serialize; // Versioned file header with magic+version+checksum

// Re-export EncodingScheme for convenience
pub use column_encoding::EncodingScheme;

use arrow_array::ArrayRef;
use arrow_schema::DataType;
use std::sync::Arc;

use thiserror::Error;

#[derive(Error, Debug)]
pub enum CodecError {
    #[error("Encoding error: {0}")]
    Encode(String),
    #[error("Decoding error: {0}")]
    Decode(String),
}

/// Encode object to binary bytes using postcard.
pub fn encode<T: serde::Serialize>(value: &T) -> Result<Vec<u8>, CodecError> {
    postcard_adapter::encode(value)
}

/// Decode from binary bytes using postcard.
pub fn decode<T: serde::de::DeserializeOwned>(bytes: &[u8]) -> Result<T, CodecError> {
    postcard_adapter::decode(bytes)
}

/// Construct a null array of `len` rows for the given Arrow DataType.
///
/// Used when a column file is missing (e.g., after a partial write) but the schema
/// still expects the column to exist with a NULL value.
///
/// Returns an error for unsupported data types instead of silently returning wrong-typed data.
pub fn make_null_array(dt: &DataType, len: usize) -> crate::error::Result<ArrayRef> {
    use arrow_array::{
        BooleanArray, Date32Array, Date64Array, Float32Array, Float64Array,
        Int16Array, Int32Array, Int64Array, Int8Array, NullArray,
        TimestampMicrosecondArray, TimestampMillisecondArray,
        UInt16Array, UInt32Array, UInt64Array, UInt8Array,
    };

    match dt {
        DataType::Int8 => Ok(Arc::new(Int8Array::from(vec![None; len]))),
        DataType::Int16 => Ok(Arc::new(Int16Array::from(vec![None; len]))),
        DataType::Int32 => Ok(Arc::new(Int32Array::from(vec![None; len]))),
        DataType::Int64 => Ok(Arc::new(Int64Array::from(vec![None; len]))),
        DataType::UInt8 => Ok(Arc::new(UInt8Array::from(vec![None; len]))),
        DataType::UInt16 => Ok(Arc::new(UInt16Array::from(vec![None; len]))),
        DataType::UInt32 => Ok(Arc::new(UInt32Array::from(vec![None; len]))),
        DataType::UInt64 => Ok(Arc::new(UInt64Array::from(vec![None; len]))),
        DataType::Float32 => Ok(Arc::new(Float32Array::from(vec![None; len]))),
        DataType::Float64 => Ok(Arc::new(Float64Array::from(vec![None; len]))),
        DataType::Boolean => Ok(Arc::new(BooleanArray::from(vec![None; len]))),
        DataType::Utf8 => Ok(Arc::new({
            let mut b = arrow_array::builder::StringBuilder::new();
            #[allow(unused_must_use)]
            for _ in 0..len {
                b.append_null();
            }
            b.finish()
        })),
        DataType::LargeUtf8 => Ok(Arc::new({
            let mut b = arrow_array::builder::LargeStringBuilder::new();
            #[allow(unused_must_use)]
            for _ in 0..len {
                b.append_null();
            }
            b.finish()
        })),
        DataType::Binary => Ok(Arc::new({
            let mut b = arrow_array::builder::BinaryBuilder::new();
            #[allow(unused_must_use)]
            for _ in 0..len {
                b.append_null();
            }
            b.finish()
        })),
        DataType::LargeBinary => Ok(Arc::new({
            let mut b = arrow_array::builder::LargeBinaryBuilder::new();
            #[allow(unused_must_use)]
            for _ in 0..len {
                b.append_null();
            }
            b.finish()
        })),
        DataType::Date32 => Ok(Arc::new(Date32Array::from(vec![None; len]))),
        DataType::Date64 => Ok(Arc::new(Date64Array::from(vec![None; len]))),
        DataType::Timestamp(arrow_schema::TimeUnit::Microsecond, _) => {
            Ok(Arc::new(TimestampMicrosecondArray::from(vec![None; len])))
        }
        DataType::Timestamp(arrow_schema::TimeUnit::Millisecond, _) => {
            Ok(Arc::new(TimestampMillisecondArray::from(vec![None; len])))
        }
        // For all other types, return NullArray to avoid schema mismatch.
        // This preserves the row count (correctness) but the column type will be NULL.
        _ => {
            tracing::warn!(
                "make_null_array: unsupported data type {:?}, returning NullArray",
                dt
            );
            Ok(Arc::new(NullArray::new(len)))
        }
    }
}
