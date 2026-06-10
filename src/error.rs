//! Error types for RockDuck

use thiserror::Error;

#[derive(Error, Debug)]
pub enum RockDuckError {
    #[error("Storage error: {0}")]
    Storage(String),

    #[error("Metadata error: {0}")]
    Metadata(String),

    #[error("Segment not found: {0}")]
    SegmentNotFound(String),

    #[error("Column not found: {0}")]
    ColumnNotFound(String),

    #[error("Codec error: {0}")]
    Codec(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Internal error: {0}")]
    Internal(String),

    #[error("Compaction error: {0}")]
    Compaction(String),

    #[error("Delta error: {0}")]
    Delta(String),

    #[error("Read path error: {0}")]
    ReadPath(String),

    #[error("Write error: {0}")]
    Write(String),

    #[error("Query error: {0}")]
    Query(String),

    #[error("Config error: {0}")]
    Config(String),

    #[error("DuckDB error: {0}")]
    DuckDB(String),

    #[error("Kafka error: {0}")]
    Kafka(String),

    #[error("MVCC conflict: {0}")]
    MvccConflict(String),

    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    #[error("Key not found: {0}")]
    KeyNotFound(String),

    #[error("Deserialization error: {0}")]
    Deserialize(String),

    #[error("Unimplemented: {0}")]
    Unimplemented(String),

    #[error("Security: {0}")]
    Security(String),

    /// Visibility data unavailable for time-travel query.
    /// This distinguishes "data doesn't exist" from "data exists but is inaccessible".
    /// Used to fix TT-1: get_as_of None semantics confusion.
    #[error("Visibility data unavailable for segment {seg_id}: {reason}")]
    VisDataUnavailable {
        seg_id: String,
        reason: &'static str,
    },
}

impl From<crate::codec::CodecError> for RockDuckError {
    fn from(e: crate::codec::CodecError) -> Self {
        RockDuckError::Codec(e.to_string())
    }
}

impl From<arrow::error::ArrowError> for RockDuckError {
    fn from(e: arrow::error::ArrowError) -> Self {
        RockDuckError::Internal(format!("Arrow error: {}", e))
    }
}

pub type Result<T> = std::result::Result<T, RockDuckError>;

/// Alias for backward compatibility with existing code
pub type DocDBError = RockDuckError;
