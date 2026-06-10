//! Postcard serialization adapter.
//!
//! Postcard is a serde-compatible, no_std-friendly binary serialization format
//! with a documented wire format. Used for WAL payloads where schema stability
//! and compatibility matter more than zero-copy deserialization.
//!
//! Note: bincode is deprecated/unmaintained. All uses have been migrated to postcard.

use crate::codec::CodecError;
use serde::{de::DeserializeOwned, Serialize};

/// Encode an object to binary bytes using postcard.
pub fn encode<T: Serialize>(value: &T) -> Result<Vec<u8>, CodecError> {
    postcard::to_allocvec(value).map_err(|e| CodecError::Encode(e.to_string()))
}

/// Decode from binary bytes using postcard.
pub fn decode<T: DeserializeOwned>(bytes: &[u8]) -> Result<T, CodecError> {
    postcard::from_bytes(bytes).map_err(|e| CodecError::Decode(e.to_string()))
}
