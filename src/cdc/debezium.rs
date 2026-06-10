//! Debezium CDC connector
//!
//! Implements Debezium format for CDC events.

use serde::Serialize;

/// Debezium envelope
#[derive(Debug, Clone, Serialize)]
pub struct DebeziumEnvelope {
    /// Operation type
    pub op: String,
    /// Timestamp
    pub ts_ms: u64,
    /// Before image
    pub before: Option<serde_json::Value>,
    /// After image
    pub after: Option<serde_json::Value>,
}

impl DebeziumEnvelope {
    /// Create a new envelope with just the operation type.
    pub fn new(op: &str) -> Self {
        // Use ok_or_else to handle SystemTime before UNIX_EPOCH gracefully
        let ts_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        Self {
            op: op.to_string(),
            ts_ms,
            before: None,
            after: None,
        }
    }

    /// Create a new envelope with before/after images.
    pub fn with_images(
        op: &str,
        before: Option<serde_json::Value>,
        after: Option<serde_json::Value>,
    ) -> Self {
        // Use ok_or_else to handle SystemTime before UNIX_EPOCH gracefully
        let ts_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        Self {
            op: op.to_string(),
            ts_ms,
            before,
            after,
        }
    }
}
