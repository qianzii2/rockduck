//! Value representation for individual row cells.
//!
//! Distinguishes three states that `Vec<u8>` cannot:
//! - **NULL**: the column value is null / not present
//! - **Empty**: the column value is the empty byte sequence (e.g. `""` string, zero-length binary)
//! - **Value**: a non-empty byte sequence
//!
//! Prior to this type, `extract_row_bytes` returned `Vec<u8>` with an empty vec meaning
//! "null OR unknown type OR empty string". Callers had no way to distinguish these cases.

/// Represents a single row cell value with explicit NULL semantics.
#[derive(Debug, Clone, PartialEq)]
pub enum RowValue {
    /// The cell is NULL (no value stored in the column array at this position).
    Null,
    /// The cell holds an explicitly empty value (e.g. `""`, `b""`).
    Empty,
    /// The cell holds a non-empty byte sequence.
    Value(Vec<u8>),
}

impl RowValue {
    /// Returns the byte content for `Value`, `None` for `Null` or `Empty`.
    pub fn as_bytes(&self) -> Option<&[u8]> {
        match self {
            RowValue::Value(v) => Some(v),
            _ => None,
        }
    }

    /// Returns `true` if this is `Null`.
    pub fn is_null(&self) -> bool {
        matches!(self, RowValue::Null)
    }

    /// Converts to bytes for storage in HashMap/Vec contexts.
    /// NULL and Empty both become `Vec::new()` — callers that need to distinguish
    /// NULL from Empty must use `is_null()` / pattern matching instead.
    pub fn to_storage_bytes(&self) -> Vec<u8> {
        match self {
            RowValue::Null => Vec::new(),
            RowValue::Empty => Vec::new(),
            RowValue::Value(v) => v.clone(),
        }
    }
}
