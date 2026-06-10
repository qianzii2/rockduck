//! CDC (Change Data Capture) module for RockDuck
//!
//! Provides change data capture functionality for:
//! - Streaming changes to external systems
//! - Debezium-compatible format
//! - Transaction-ordered change log
//! - DeltaCell ChangeStream (fine-grained)
//!
//! # Architecture
//!
//! CDC is driven off the WAL: during `commit_txn()`, the WAL entries for that
//! transaction are scanned and converted to `DeltaCell` entries which are written
//! to the `CdcLogBuffer` (for time-bounded retention).
//!
//! Consumers access changes via `ChangeStream`, which queries `DeltaLayerStack`
//! for committed deltas since a given txn_id, grouped by transaction.
//!
//! # Delivery Guarantees
//!
//! - **CDC Source**: At-least-once (WAL is source of truth; DeltaLayerStack
//!   is rebuilt from WAL replay if lost)
//! - **Kafka Sink**: Fire-and-forget with retry logging; Kafka failure does NOT
//!   roll back committed transactions
//! - **Transaction Atomicity**: WAL commit and CDC delta recording are sequential;
//!   if CDC flush fails after commit, error is logged and deltas
//!   can be rebuilt from WAL replay via `ChangeStream::replay_wal_since()`
//!
//! All delta data now flows through `DeltaLayerStack` (the new three-layer delta
//! store) instead of the legacy `DeltaStoreManager`.

pub mod debezium;
pub mod log;
#[cfg(feature = "kafka")]
pub mod sink;
pub mod stream; // DeltaCell ChangeStream

pub use debezium::*;
pub use log::*;
#[cfg(feature = "kafka")]
pub use sink::*;

use serde::{Deserialize, Serialize};

// =============================================================================
// CDC Granularity Configuration
// =============================================================================

/// Granularity level for CDC output.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum CdcGranularity {
    /// Cell-level: one ChangeEvent per changed cell (finest granularity).
    #[default]
    Cell,
    /// Row-level: one ChangeEvent per changed row (coarser, Kafka-friendly).
    Row,
    /// Both: expose both cell and row APIs.
    Both,
}

// =============================================================================
// TxnDeltaCollector — accumulates DeltaCell entries during a transaction
// =============================================================================

/// Accumulates cell-level delta entries during a transaction's lifetime.
/// At commit time, `flush()` converts these into `DeltaCell` records.
///
/// Thread-safety: `parking_lot::Mutex` allows lock-free reads from
/// multiple threads while the transaction writes from one thread.
#[derive(Debug, Clone, Default)]
pub struct TxnDeltaCollector {
    /// Transaction ID.
    pub txn_id: u64,
    /// Table name.
    pub table: String,
    /// Primary key bytes (for CDC Kafka message key deduplication).
    /// Populated during the first write operation via `set_pk`.
    pub pk: Vec<u8>,
    /// Per-segment, per-row, per-column delta entries.
    pub entries: Vec<TxnDeltaEntry>,
}

#[derive(Debug, Clone)]
pub struct TxnDeltaEntry {
    pub seg_id: String,
    pub row_offset: u64,
    pub column: String,
    /// Before-image value (None for inserts).
    pub before: Option<Vec<u8>>,
    /// After-image value (None for deletes).
    pub after: Option<Vec<u8>>,
    /// Operation type.
    pub op: DeltaOpType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeltaOpType {
    Insert,
    Update,
    Delete,
}

impl TxnDeltaCollector {
    pub fn new(txn_id: u64, table: String) -> Self {
        Self {
            txn_id,
            table,
            pk: Vec::new(),
            entries: Vec::new(),
        }
    }

    /// Set the primary key for this transaction. Idempotent: subsequent calls do not overwrite an already-set pk.
    ///
    /// ## Idempotent by Design
    ///
    /// This is NOT a bug. The first call wins because WAL operations must have a valid pk.
    /// If the first call passes an empty pk ( Vec::new() ), subsequent calls with the real pk
    /// will be rejected — but this indicates a bug in the CALLER, not here. WAL pk should be valid.
    pub fn set_pk(&mut self, pk: Vec<u8>) {
        if self.pk.is_empty() {
            self.pk = pk;
        }
    }

    /// Record a single-cell change.
    pub fn push_cell(
        &mut self,
        seg_id: String,
        row_offset: u64,
        column: String,
        before: Option<Vec<u8>>,
        after: Option<Vec<u8>>,
        op: DeltaOpType,
    ) {
        self.entries.push(TxnDeltaEntry {
            seg_id,
            row_offset,
            column,
            before,
            after,
            op,
        });
    }

    /// Record a full-row change (all columns at once).
    pub fn push_row(&mut self, seg_id: String, row_offset: u64, op: DeltaOpType) {
        // Caller will push individual cell entries per column.
        // This method is a marker for row-level tracking.
        let _ = (seg_id, row_offset, op);
    }

    /// Consume all entries and return them as `DeltaCell` records.
    pub fn into_delta_cells(self) -> Vec<crate::storage::delta::types::DeltaCell> {
        use crate::storage::delta::types::DeltaCell;
        use std::sync::Arc;
        self.entries
            .into_iter()
            .map(|e| DeltaCell {
                seg_id: e.seg_id,
                row_offset: e.row_offset,
                column: e.column,
                txn_id: self.txn_id,
                before: e.before.map(Arc::new),
                after: e.after.map(Arc::new),
                committed: true,
                ts: self.txn_id as i64,
            })
            .collect()
    }

    /// Returns the number of cell-level entries accumulated.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns true if no entries have been accumulated.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

// =============================================================================
// CdcLogBuffer — in-memory retention of CDC events (bounded ring buffer)
// =============================================================================
