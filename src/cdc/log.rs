//! CDC log entry
//!
//! CDC (Change Data Capture) for tracking database changes.
//!
//! ## Architecture
//!
//! CDC uses a **dual-path** approach:
//! - **In-memory buffer** (`CdcLogBuffer`): bounded ring buffer for fast time-travel queries
//! - **CDC WAL** (`CdcWalWriter`): separate `walog::WalWriter` for durable CDC event persistence
//!
//! At `commit_txn()`, WAL entries for the committed transaction are scanned and converted
//! to `CdcOp` records, which are written to both paths. CDC WAL is completely independent
//! from the data WAL — it has its own segment files in `cdc_wal/` subdirectory.
//!
//! ## Thread Safety
//!
//! `CdcLogBuffer` uses internal synchronization to support concurrent access:
//! - `VecDeque` is protected by `std::sync::Mutex` for thread-safe push/iter operations
//! - `dropped_count` uses atomic counters for lock-free monitoring
//!
//! ## Delivery Semantics (two distinct guarantees)
//!
//! CDC events flow through two paths with fundamentally different semantics:
//!
//! ### Hard constraint: `CdcLogBuffer` (in-memory)
//!
//! This is a bounded ring buffer used for time-travel queries. When the buffer is full,
//! `push()` returns `Err(CdcLogError::BufferFull)` and **blocks transaction commit**.
//! This is the correct behaviour: a full buffer means CDC data would be lost if we
//! committed silently. The upper layer (`commit_txn`) must handle this error explicitly.
//!
//! ### Fire-and-forget: CDC WAL + Kafka Sink
//!
//! - **CDC WAL** (`CdcWalWriter`): appends `CdcOp` records to durable WAL. If WAL append
//!   fails, the error is logged and the transaction still commits — WAL is best-effort.
//!   Lost events can be rebuilt via `ChangeStream::replay_wal_since()`.
//! - **Kafka Sink** (`CdcSink`): sends events to Kafka. On send failure, the error is
//!   logged at `warn` level and the transaction still commits. Kafka failures do NOT
//!   roll back committed transactions. Lost events can be rebuilt via WAL replay.
//!
//! The implication: if both the in-memory buffer and CDC WAL fail, the transaction commit
//! is blocked. If only Kafka fails, data is recoverable from WAL replay.

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use parking_lot::Mutex;

pub use durability::walog::SyncWalWriter as WalWriter;
pub use durability::walog::WalReader;

/// CDC operation type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CdcOpType {
    Insert,
    Update,
    Delete,
}

/// CDC log entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CdcLogEntry {
    /// Operation type
    pub op: CdcOpType,
    /// Table name
    pub table: String,
    /// Segment identifier.
    pub seg_id: String,
    /// Primary key
    pub pk: Vec<u8>,
    /// Column values (before image for update/delete)
    pub before: Vec<(String, Vec<u8>)>,
    /// Column values (after image for insert/update)
    pub after: Vec<(String, Vec<u8>)>,
    /// Transaction ID
    pub txn_id: u64,
    /// Timestamp
    pub ts: u64,
}

/// CDC log buffer — thread-safe ring buffer for CDC log entries.
///
/// Uses `Mutex<VecDeque>` for thread-safe concurrent access to the entries.
#[derive(Debug)]
pub struct CdcLogBuffer {
    /// Entries (protected by mutex for thread safety)
    entries: Mutex<VecDeque<CdcLogEntry>>,
    /// Max size
    max_size: usize,
    /// Number of entries dropped due to full buffer
    pub dropped_count: std::sync::atomic::AtomicUsize,
}

impl CdcLogBuffer {
    /// Create a new buffer
    pub fn new(max_size: usize) -> Self {
        Self {
            entries: Mutex::new(VecDeque::new()),
            max_size,
            dropped_count: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Add an entry. Returns Err if the buffer is full (entry is NOT silently dropped).
    /// Thread-safe: uses mutex for concurrent access.
    pub fn push(&self, entry: CdcLogEntry) -> Result<(), CdcLogError> {
        let mut entries = self.entries.lock();
        if entries.len() >= self.max_size {
            self.dropped_count
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return Err(CdcLogError::BufferFull {
                dropped: self
                    .dropped_count
                    .load(std::sync::atomic::Ordering::Relaxed),
            });
        }
        entries.push_back(entry);
        Ok(())
    }

    /// Get the number of entries dropped due to buffer overflow.
    pub fn dropped_count(&self) -> usize {
        self.dropped_count
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Get all entries (cloned for thread-safe access).
    pub fn entries(&self) -> Vec<CdcLogEntry> {
        self.entries.lock().iter().cloned().collect()
    }

    /// Clear the buffer
    ///
    /// ## Note: dropped_count is NOT reset
    ///
    /// This is NOT a bug. `dropped_count` tracks total overflow events across the buffer's
    /// lifetime for monitoring/alerting purposes. Resetting it on clear() would lose useful
    /// telemetry. Monitor `dropped_count()` separately from the buffer contents.
    pub fn clear(&self) {
        self.entries.lock().clear();
    }
}

impl Default for CdcLogBuffer {
    fn default() -> Self {
        Self::new(10000)
    }
}

/// Error returned when CDC log buffer is full.
#[derive(Debug, Clone, thiserror::Error)]
pub enum CdcLogError {
    #[error("CDC log buffer full; {} entries dropped so far", dropped)]
    BufferFull { dropped: usize },
}

// =============================================================================
// CDC WAL — durable CDC event persistence
// =============================================================================

/// CDC WAL operation — the record type written to the CDC WAL.
/// This is separate from the main data WAL (`WalOp`).
///
/// Serialized via `serde` (bincode) and written through `walog::WalWriter`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CdcOp {
    /// CDC operation type.
    pub op: CdcOpType,
    /// Table name.
    pub table: String,
    /// Segment identifier.
    pub seg_id: String,
    /// Primary key bytes.
    pub pk: Vec<u8>,
    /// Before-image: column name → value bytes.
    pub before: Vec<(String, Vec<u8>)>,
    /// After-image: column name → value bytes.
    pub after: Vec<(String, Vec<u8>)>,
    /// Transaction ID.
    pub txn_id: u64,
    /// Commit timestamp (ms since epoch).
    pub ts: u64,
}

impl CdcOp {
    /// Construct a `CdcOp` from a `CdcLogEntry`.
    pub fn from_entry(entry: &CdcLogEntry) -> Self {
        Self {
            op: entry.op,
            table: entry.table.clone(),
            seg_id: entry.seg_id.clone(),
            pk: entry.pk.clone(),
            before: entry.before.clone(),
            after: entry.after.clone(),
            txn_id: entry.txn_id,
            ts: entry.ts,
        }
    }
}

/// CDC WAL writer — a `SyncWalWriter` dedicated to CDC events.
/// Lives in its own `cdc_wal/` subdirectory and is completely independent from
/// the main data WAL.
///
/// `SyncWalWriter` provides thread-safe appends with built-in mutex protection
/// and group-commit semantics (batches appends, flushes in background thread).
pub struct CdcWalWriter {
    inner: WalWriter<CdcOp>,
}

impl CdcWalWriter {
    /// Open (or create) a CDC WAL writer under `cdc_wal_dir`.
    pub fn open(cdc_wal_dir: std::path::PathBuf) -> std::result::Result<Self, CdcWalError> {
        std::fs::create_dir_all(&cdc_wal_dir)
            .map_err(|e| CdcWalError::Io(format!("create CDC WAL dir: {}", e)))?;

        let dir: std::sync::Arc<dyn durability::Directory> =
            durability::storage::FsDirectory::arc(cdc_wal_dir)
                .map_err(|e| CdcWalError::Io(format!("FsDirectory::arc: {}", e)))?;

        WalWriter::open(dir)
            .map(|inner| Self { inner })
            .map_err(|e| CdcWalError::Io(format!("WalWriter::open: {}", e)))
    }

    /// Append a CDC operation to the WAL and flush synchronously.
    pub fn append_and_flush(&self, op: CdcOp) -> std::result::Result<(), CdcWalError> {
        self.inner
            .append_durable(&op)
            .map_err(|e| CdcWalError::Io(format!("append_durable: {}", e)))?;
        Ok(())
    }

    /// Append a CDC operation without immediate durability (batched).
    pub fn append(&self, op: CdcOp) -> std::result::Result<u64, CdcWalError> {
        self.inner
            .append(&op)
            .map_err(|e| CdcWalError::Io(format!("append: {}", e)))
    }
}

/// Error type for CDC WAL operations.
#[derive(Debug, thiserror::Error)]
pub enum CdcWalError {
    #[error("CDC WAL I/O error: {}", 0)]
    Io(String),
}
