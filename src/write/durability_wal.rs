//! Durability WAL adapter for RockDuck
//!
//! Architecture: A single `walog::WalWriter` is shared between the normal commit path and
//! the checkpoint path. A `parking_lot::Mutex` provides interior mutability so both paths
//! can access the same writer. `SyncWalWriter` is NOT used because it wraps the writer in
//! a second layer of mutex, adding unnecessary overhead and preventing concurrent reads of
//! metadata.
//!
//! ## Group Commit
//!
//! When `SyncPolicy::GroupCommitStrict` is configured, commits are batched and flushed together:
//! - Background flusher thread waits up to `max_wait_ms` or until `max_batch_size` commits
//!   accumulate, then does ONE `flush_and_sync()` for all of them
//! - Uses `CasLeaderElection` so only one thread performs the sync; others spin-wait
//! - Non-commit ops (Insert, Delete, Update) bypass batching and go straight to WAL
//! - Shutdown uses an mpsc channel: dropping the Sender signals the flusher to exit.
//!   This is cleaner than AtomicBool + unpark because channel disconnect is unambiguous.
//!
//! ## Shutdown Safety
//!
//! When `WalWriter` is dropped:
//!   1. The `Sender` is dropped (moved into `Drop::drop`)
//!   2. The flusher's `Receiver` gets a disconnect notification
//!   3. `recv_timeout` returns `Err(Disconnected)` → flusher exits cleanly
//!   4. OS reclaims the thread when it finishes
//!
//! The channel is NOT wrapped in Arc because the Sender lives in the struct
//! and the Receiver lives in the spawned thread. Arc is not needed since the
//! thread has exclusive ownership of the Receiver.

use durability::walog;
pub use durability::walog::WalReader;
use durability::Directory;
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use crate::error::{Result, RockDuckError};
use crate::metadata::GranuleId;
use crate::write::group_commit::CasLeaderElection;

/// WAL operation type
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    serde::Serialize,
    serde::Deserialize,
    rkyv::Archive,
    rkyv::Serialize,
    rkyv::Deserialize,
)]
pub enum OpType {
    Begin = 0x01,
    Insert = 0x02,
    Delete = 0x03,
    Update = 0x04,
    Checkpoint = 0x05,
    /// Compaction: written by NonBlockingCompactor to protect pk_lookup + alias updates.
    /// WAL recovery replays this to finish or rollback in-flight compaction after a crash.
    Compaction = 0x06,
    Commit = 0x10,
    Rollback = 0x11,
}

impl OpType {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0x01 => Some(Self::Begin),
            0x02 => Some(Self::Insert),
            0x03 => Some(Self::Delete),
            0x04 => Some(Self::Update),
            0x05 => Some(Self::Checkpoint),
            0x06 => Some(Self::Compaction),
            0x10 => Some(Self::Commit),
            0x11 => Some(Self::Rollback),
            _ => None,
        }
    }
}

/// WAL record payload
#[derive(
    Debug,
    Clone,
    serde::Serialize,
    serde::Deserialize,
    rkyv::Archive,
    rkyv::Serialize,
    rkyv::Deserialize,
)]
pub enum OpPayload {
    Begin,
    Insert {
        table: String,
        pk: [u8; 8],
        /// Column names in WAL batch (order matches columns in wal_batch).
        columns: Vec<String>,
        /// Level 1: Arrow IPC Stream bytes for all columns.
        /// Level 3: raw Arrow buffer bytes.
        /// w013 fix: wrapped in Arc so WAL append_durable can share the data without clone.
        wal_batch: Arc<Vec<u8>>,
        /// Level 3 only: Arrow IPC schema bytes for raw batch deserialization.
        #[serde(default)]
        schema_bytes: Vec<u8>,
        seg_id: String,
        granule_id: GranuleId,
        offset: u64,
    },
    Delete {
        table: String,
        pk: [u8; 8],
        #[serde(default)]
        before_row: Vec<(String, Vec<u8>)>,
        seg_id: String,
        #[serde(default)]
        granule_id: GranuleId,
        offset: u64,
    },
    Update {
        table: String,
        pk: [u8; 8],
        /// Column names in new wal_batch.
        columns: Vec<String>,
        /// Level 1: Arrow IPC Stream bytes for new values.
        /// Level 3: raw Arrow buffer bytes.
        /// w013 fix: wrapped in Arc so WAL append_durable can share the data without clone.
        wal_batch: Arc<Vec<u8>>,
        /// Level 3 only: Arrow IPC schema bytes.
        #[serde(default)]
        schema_bytes: Vec<u8>,
        /// Before-image for CDC -- kept as per-column IPC File (single-row, low overhead).
        #[serde(default)]
        old_columns: Vec<(String, Vec<u8>)>,
        old_seg_id: String,
        #[serde(default)]
        old_granule_id: GranuleId,
        old_offset: u64,
        new_seg_id: String,
        #[serde(default)]
        new_granule_id: GranuleId,
        offset: u64,
    },
    Checkpoint {
        checkpoint_id: u64,
    },
    /// Compaction WAL record: written after all four compaction KV operations complete.
    /// Contains the data needed to either finish or rollback in-flight compaction on recovery.
    Compaction {
        #[serde(default)]
        old_seg_id: String,
        #[serde(default)]
        new_seg_id: String,
        /// Serialized pk_lookup transition entries: Vec<(pk, granule_id, row_offset)>.
        /// The seg_id is at the record level (new_seg_id) to avoid duplication per entry.
        #[serde(default)]
        pk_entries: Vec<u8>,
        #[serde(default)]
        commit: bool,
    },
    Commit {
        begin_ts: u64,
        #[serde(default)]
        inserted_at: Option<u64>,
    },
    Rollback,
}

/// Bundled WAL entry
#[derive(
    Debug,
    Clone,
    serde::Serialize,
    serde::Deserialize,
    rkyv::Archive,
    rkyv::Serialize,
    rkyv::Deserialize,
)]
pub struct WalOp {
    pub op_type: OpType,
    pub txn_id: u64,
    pub payload: OpPayload,
}

/// WAL configuration
#[derive(Debug, Clone)]
pub struct WalConfig {
    pub wal_dir: PathBuf,
    pub max_file_size: u64,
    pub enabled: bool,
    /// Group commit configuration. Controls fsync behavior and durability guarantees.
    /// If None, defaults to GroupCommit (async batch flush).
    pub group_commit: Option<GroupCommitConfig>,
}

impl Default for WalConfig {
    fn default() -> Self {
        Self {
            wal_dir: PathBuf::from("wal"),
            max_file_size: 128 * 1024 * 1024,
            enabled: true,
            group_commit: Some(GroupCommitConfig::default()),
        }
    }
}

/// Group commit / fsync policy.
#[derive(Debug, Clone, Copy, Default)]
pub enum SyncPolicy {
    /// Synchronous: fsync every write before returning. Maximum durability, minimum throughput.
    SyncEach,
    /// Group Commit (Strict): batch writes, wait for flush+fsync confirmation before returning.
    /// Durable with group commit throughput benefits. Formerly `GroupCommit` (async) which
    /// could lose committed data on crash; this variant replaces it as the default.
    #[default]
    GroupCommitStrict,
    /// Flush to OS buffer and fsync. Note: current implementation syncs (not truly flush-only).
    /// Preserved for documentation purposes; SyncEach provides equivalent behavior.
    FlushOnly,
}

/// Group commit configuration.
#[derive(Debug, Clone)]
pub struct GroupCommitConfig {
    pub max_batch_size: usize,
    pub max_wait_ms: u64,
    pub policy: SyncPolicy,
}

impl Default for GroupCommitConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 100,
            max_wait_ms: 50,
            policy: SyncPolicy::GroupCommitStrict,
        }
    }
}

/// GC queue wrapper -- uses VecDeque to support push_front (for preserving order on requeue).
pub type GcQueue = std::sync::Arc<parking_lot::Mutex<VecDeque<WalOp>>>;

/// Durability-backed WAL writer.
/// Single-writer architecture: one `walog::WalWriter` wrapped in `parking_lot::Mutex` inside an `Arc`.
/// Both the normal commit path and the background flusher thread access the same writer through
/// the shared Arc<Mutex>. The mutex is fine-grained enough that it only blocks during actual writes.
pub struct WalWriter {
    /// P2-B: changed to Arc<Mutex> so the background flusher thread can share the writer
    /// with caller threads via Arc::clone.
    writer: std::sync::Arc<parking_lot::Mutex<walog::WalWriter<WalOp>>>,
    wal_dir: PathBuf,
    config: WalConfig,
    committed_txn: AtomicU64,

    // Group commit state
    #[allow(dead_code)]
    group_commit_enabled: bool,
    #[allow(dead_code)]
    group_commit_batch_max: usize,
    #[allow(dead_code)]
    group_commit_max_wait_ms: u64,
    /// Shared GC queue (Arc<Mutex<Vec<WalOp>>) between caller threads and the background flusher thread.
    group_commit_queue: GcQueue,
    group_commit_queue_size: std::sync::Arc<AtomicUsize>,
    group_commit_last_sync: std::sync::Arc<parking_lot::Mutex<Instant>>,
    group_commit_flush_needed: std::sync::Arc<AtomicBool>,
    /// Set to true before fsync, false on failure. Used by Strict mode to wait
    /// for actual durable confirmation, not just queue drain.
    group_commit_flush_success: std::sync::Arc<AtomicBool>,
    group_commit_election: std::sync::Arc<CasLeaderElection>,
    /// Channel sender for signaling the background flusher thread to exit.
    /// `Option` so we can `.take()` in Drop without consuming self.
    /// `Some` while alive, `None` after Drop.
    group_commit_shutdown_tx: Option<mpsc::Sender<()>>,
    /// Tracks consecutive flush failures to prevent infinite retry loops.
    /// Reset to 0 on success. Bounded by MAX_FLUSH_RETRIES.
    group_commit_flush_retries: std::sync::Arc<AtomicUsize>,
    /// Background flusher thread handle. Joined in Drop to ensure the flusher
    /// finishes its final flush iteration (draining the queue and syncing to WAL)
    /// before Drop proceeds. This eliminates the race where Drop's try_lock competes
    /// with the flusher's flush_and_sync for the writer lock.
    worker_handle: Option<std::thread::JoinHandle<()>>,
}

impl WalWriter {
    /// Maximum number of consecutive flush failures before giving up.
    pub const MAX_FLUSH_RETRIES: usize = 3;
    pub fn open(data_dir: &Path, config: WalConfig) -> Result<Self> {
        let wal_dir = if config.wal_dir.is_absolute() {
            config.wal_dir.clone()
        } else {
            data_dir.join(&config.wal_dir)
        };

        std::fs::create_dir_all(&wal_dir).map_err(|e| RockDuckError::Write(e.to_string()))?;

        // RECOVER FIRST: scan existing WAL entries before any cleanup.
        let mut recovered_committed = 0u64;
        if let Ok(reader_dir) = durability::storage::FsDirectory::arc(&wal_dir) {
            let reader = WalReader::<WalOp>::new(reader_dir);
            if let Ok(records) = reader.replay_best_effort() {
                for record in records {
                    if record.payload.op_type == OpType::Commit {
                        recovered_committed = recovered_committed.max(record.payload.txn_id);
                    }
                }
            }
        }

        // Cleanup: only remove leftover temp files, not valid WAL segments.
        Self::cleanup_wal_dir(&wal_dir);

        // Single writer for the entire WAL -- no dual-writer.
        // This avoids the segment coordination problem that two separate writers would have.
        let dir = durability::storage::FsDirectory::arc(&wal_dir)
            .map_err(|e| RockDuckError::Write(format!("FsDirectory::arc: {e}")))?;
        let mut writer: walog::WalWriter<WalOp> = walog::WalWriter::open(dir)
            .map_err(|e| RockDuckError::Write(format!("WalWriter::open: {e}")))?;
        writer.set_segment_size_limit_bytes(config.max_file_size);
        writer.set_preallocate_bytes(1024 * 1024);

        let gc_config = config.group_commit.as_ref();

        // Extract group commit params (from config or defaults)
        let group_commit_batch_max = gc_config.map(|c| c.max_batch_size).unwrap_or(100);
        let group_commit_max_wait_ms = gc_config.map(|c| c.max_wait_ms).unwrap_or(50);

        // Channel-based shutdown: dropping the Sender signals the flusher to exit.
        // This is cleaner than AtomicBool + unpark because channel disconnect is
        // unambiguous. The Sender lives in the struct, Receiver in the thread.
        let (shutdown_tx, shutdown_rx) = mpsc::channel::<()>();

        // GC state shared between caller threads and the background flusher thread.
        let gc_queue = std::sync::Arc::new(parking_lot::Mutex::new(VecDeque::with_capacity(1024)));
        let gc_queue_size = std::sync::Arc::new(AtomicUsize::new(0));
        let gc_last_sync = std::sync::Arc::new(parking_lot::Mutex::new(Instant::now()));
        let gc_flush_needed = std::sync::Arc::new(AtomicBool::new(false));
        let gc_flush_success = std::sync::Arc::new(AtomicBool::new(false));
        let gc_election = std::sync::Arc::new(CasLeaderElection::new());
        let gc_flush_retries = std::sync::Arc::new(AtomicUsize::new(0));

        // Clone Arcs to move into the background thread
        let queue_for_thread = std::sync::Arc::clone(&gc_queue);
        let queue_size_for_thread = std::sync::Arc::clone(&gc_queue_size);
        let last_sync_for_thread = std::sync::Arc::clone(&gc_last_sync);
        let flush_needed_for_thread = std::sync::Arc::clone(&gc_flush_needed);
        let flush_success_for_thread = std::sync::Arc::clone(&gc_flush_success);
        let election_for_thread = std::sync::Arc::clone(&gc_election);

        // writer is shared via Arc between the struct field and the background flusher thread.
        // Arc::clone is O(1) refcount increment -- the underlying mutex and writer are NOT copied.
        let writer_arc = std::sync::Arc::new(parking_lot::Mutex::new(writer));

        // P2-B fix: the background flusher thread needs its own Arc clone so it can
        // access the writer independently from the struct field. Without this clone,
        // the JoinHandle goes out of scope at the end of open() and the thread terminates.
        let writer_arc_for_thread = std::sync::Arc::clone(&writer_arc);

        // P2-B fix: spawn the background flusher thread.
        // Wakes up every max_wait_ms, tries CAS leader, flushes pending commits.
        // This ensures the queue is drained even when no new commits arrive.
        // The JoinHandle is stored in the struct so the thread outlives `open()`.
        let worker_handle = std::thread::Builder::new()
            .name("wal-group-commit-flusher".into())
            .spawn(move || {
                loop {
                    // Wait for either a timeout or a shutdown signal.
                    // recv_timeout returns Err(Disconnected) when the Sender is dropped.
                    match shutdown_rx.recv_timeout(Duration::from_millis(group_commit_max_wait_ms))
                    {
                        Ok(()) | Err(mpsc::RecvTimeoutError::Disconnected) => {
                            // Shutdown signal received -- exit the flusher loop.
                            break;
                        }
                        Err(mpsc::RecvTimeoutError::Timeout) => {
                            // Timeout -- do the normal flush work.
                        }
                    }

                    // Try to become leader and flush
                    election_for_thread.add_waiter();
                    if election_for_thread.try_claim_leader() {
                        // Do the flush using the shared GC state
                        let ops: Vec<WalOp> = {
                            let mut q = queue_for_thread.lock();
                            q.drain(..).collect()
                        };

                        if ops.is_empty() {
                            // Nothing to flush -- just release leadership
                            queue_size_for_thread.store(0, Ordering::SeqCst);
                            flush_success_for_thread.store(true, Ordering::SeqCst);
                            election_for_thread.release_leadership();
                            election_for_thread.remove_waiter();
                            continue;
                        }

                        tracing::debug!("Background flusher flushing {} entries", ops.len());

                        if let Some(mut writer) = writer_arc_for_thread.try_lock() {
                            if writer.append_batch(&ops).is_ok()
                            && writer.flush_and_sync().is_ok()
                        {
                            *last_sync_for_thread.lock() = Instant::now();
                            queue_size_for_thread.store(0, Ordering::SeqCst);
                            flush_success_for_thread.store(true, Ordering::SeqCst);
                            flush_needed_for_thread.store(false, Ordering::SeqCst);
                            election_for_thread.release_leadership();
                            election_for_thread.remove_waiter();
                            continue;
                        }
                            // Flush failed -- requeue ops at HEAD to preserve order (Bug #4 fix),
                            // and keep flush_needed = true so the next wake-up will retry.
                            flush_success_for_thread.store(false, Ordering::SeqCst);
                            flush_needed_for_thread.store(true, Ordering::SeqCst);
                            let mut q = queue_for_thread.lock();
                            // Prepend ops at head so ordering is preserved after retry.
                            // VecDeque::push_front inserts at the front (lowest index).
                            for op in ops.into_iter().rev() {
                                q.push_front(op);
                            }
                            queue_size_for_thread.store(q.len(), Ordering::SeqCst);
                        } else {
                            // try_lock failed — flusher thread holds the writer lock.
                            // Do NOT set queue_size=0 (Bug #2 fix): ops are still in the queue.
                            // Setting queue_size=0 would make waiters incorrectly believe the flush
                            // succeeded when it never happened.
                            // Requeue ops at HEAD to preserve order (Issue #15 fix).
                            // The queue lock was released at line 390's `}`, so this acquisition
                            // is safe — no deadlock because we're not re-entering a held lock.
                            flush_success_for_thread.store(false, Ordering::SeqCst);
                            flush_needed_for_thread.store(true, Ordering::SeqCst);
                            let mut q = queue_for_thread.lock();
                            for op in ops.into_iter().rev() {
                                q.push_front(op);
                            }
                            queue_size_for_thread.store(q.len(), Ordering::SeqCst);
                        }
                        election_for_thread.release_leadership();
                    }
                    election_for_thread.remove_waiter();
                }
            })
            .map_err(|e| RockDuckError::Write(format!("spawn flusher thread: {}", e)))?;

        Ok(Self {
            writer: writer_arc,
            wal_dir,
            config: config.clone(),
            committed_txn: AtomicU64::new(recovered_committed),
            group_commit_enabled: gc_config.is_some(),
            group_commit_batch_max,
            group_commit_max_wait_ms,
            // Point at the shared GC state
            group_commit_queue: gc_queue,
            group_commit_queue_size: gc_queue_size.clone(),
            group_commit_last_sync: gc_last_sync.clone(),
            group_commit_flush_needed: gc_flush_needed.clone(),
            group_commit_flush_success: gc_flush_success.clone(),
            group_commit_election: gc_election.clone(),
            // mpsc Sender signals shutdown when dropped; kept as Option to allow take()
            group_commit_shutdown_tx: Some(shutdown_tx),
            group_commit_flush_retries: gc_flush_retries.clone(),
            // P2-B fix: background flusher thread handle stored in struct so it outlives open()
            worker_handle: Some(worker_handle),
        })
    }

    /// Remove only temp/stale WAL files. Does NOT remove valid segments.
    fn cleanup_wal_dir(wal_dir: &Path) {
        if let Ok(entries) = std::fs::read_dir(wal_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_file() {
                    if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                        if name.starts_with('.') {
                            let _ = std::fs::remove_file(&path);
                        }
                    }
                }
            }
        }
    }

    /// Append a record to the WAL without durability.
    pub fn append(&self, op_type: OpType, txn_id: u64, payload: &OpPayload) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }

        let op = WalOp {
            op_type,
            txn_id,
            payload: payload.clone(),
        };

        let mut writer = self.writer.lock();
        writer
            .append(&op)
            .map_err(|e| RockDuckError::Write(format!("WAL append: {e}")))?;

        if op_type == OpType::Commit {
            self.committed_txn
                .fetch_update(Ordering::SeqCst, Ordering::Relaxed, |current| {
                    Some(current.max(txn_id))
                })
                .ok();
        }

        Ok(())
    }

    /// Append and flush to stable storage.
    ///
    /// For `SyncPolicy::SyncEach`: append + immediate fsync.
    /// For `SyncPolicy::GroupCommit`: append to queue, signal background flusher, return immediately.
    /// For `SyncPolicy::GroupCommitStrict`: append to queue, wait for queue drain AND fsync success.
    /// For `SyncPolicy::FlushOnly`: append + flush (no fsync).
    pub fn append_durable(&self, op_type: OpType, txn_id: u64, payload: &OpPayload) -> Result<()> {
        if !self.config.enabled && op_type != OpType::Commit {
            return Ok(());
        }

        let op = WalOp {
            op_type,
            txn_id,
            payload: payload.clone(),
        };

        match self.config.sync_policy() {
            SyncPolicy::SyncEach => {
                let mut writer = self.writer.lock();
                writer
                    .append(&op)
                    .map_err(|e| RockDuckError::Write(format!("WAL append: {e}")))?;
                writer
                    .flush_and_sync()
                    .map_err(|e| RockDuckError::Write(format!("WAL flush_and_sync: {e}")))?;
            }
            SyncPolicy::GroupCommitStrict => {
                // Queue the commit op for batch flush, then wait for durable confirmation:
                // both queue drain AND flush_and_sync success.
                self.queue_commit_for_batch(op.clone())?;
                self.wait_for_batch_flush_strict()?;
            }
            SyncPolicy::FlushOnly => {
                let mut writer = self.writer.lock();
                writer
                    .append(&op)
                    .map_err(|e| RockDuckError::Write(format!("WAL append: {e}")))?;
                // Call flush_and_sync for true durability instead of flush-only.
                // NOTE: FlushOnly now provides the same durability as SyncEach -- if true
                // flush-only (OS buffer only) behavior is needed, a new policy is needed.
                writer
                    .flush_and_sync()
                    .map_err(|e| RockDuckError::Write(format!("WAL flush_and_sync: {e}")))?;
            }
        }

        if op_type == OpType::Commit {
            self.committed_txn
                .fetch_update(Ordering::SeqCst, Ordering::Relaxed, |current| {
                    Some(current.max(txn_id))
                })
                .ok();
        }

        Ok(())
    }

    /// Queue a commit op for batched flush and wake the flusher.
    fn queue_commit_for_batch(&self, op: WalOp) -> Result<()> {
        {
            let mut queue = self.group_commit_queue.lock();
            queue.push_back(op);
            self.group_commit_queue_size
                .store(queue.len(), Ordering::Relaxed);
        }
        // Signal the background flusher that work is available
        self.group_commit_flush_needed
            .store(true, Ordering::Release);
        // Note: we no longer explicitly wake the flusher thread.
        // The flusher uses recv_timeout on the shutdown channel with a timeout,
        // so it wakes up periodically regardless. No unpark needed.
        Ok(())
    }

    /// Wait for the batch to be flushed by the background flusher (eventual mode).
    /// Uses CAS leader election so only one thread does the actual sync.
    #[allow(dead_code)]
    fn wait_for_batch_flush(&self) {
        self.group_commit_election.add_waiter();

        loop {
            // Check if the queue has been flushed (leader did it)
            if self.group_commit_queue_size.load(Ordering::Relaxed) == 0 {
                break;
            }
            // Try to become the leader and do the flush
            if self.group_commit_election.try_claim_leader() {
                self.flush_batch_with_retry();
                self.group_commit_election.release_leadership();
                break;
            }
            // Not the leader -- spin-wait for the leader to finish
            self.group_commit_election.wait_for_sync();
        }

        self.group_commit_election.remove_waiter();
    }

    /// Wait for the batch to be flushed AND durably synced (strict mode).
    ///
    /// Waits for BOTH queue drain AND flush_success flag.
    /// Previously, wait_for_batch_flush only checked queue_size == 0, which could
    /// return while flush_and_sync was still in progress or had failed.
    fn wait_for_batch_flush_strict(&self) -> Result<()> {
        self.group_commit_election.add_waiter();

        loop {
            // Check if the queue has been flushed AND the flush succeeded
            if self.group_commit_queue_size.load(Ordering::SeqCst) == 0
                && self.group_commit_flush_success.load(Ordering::SeqCst)
            {
                break;
            }

            // Try to become the leader and do the flush
            if self.group_commit_election.try_claim_leader() {
                self.flush_batch_with_retry();
                self.group_commit_election.release_leadership();

                // After leader flush, check if we succeeded
                if self.group_commit_flush_success.load(Ordering::SeqCst) {
                    break;
                }
                // Flush failed -- propagate error
                tracing::error!("GroupCommitStrict: flush failed, returning error");
                self.group_commit_election.remove_waiter();
                return Err(RockDuckError::Write(
                    "GroupCommitStrict: WAL flush failed, data may not be durable".into(),
                ));
            }

            // Not the leader -- spin-wait for the leader to finish
            self.group_commit_election.wait_for_sync();

            // After leader finishes, check success
            if self.group_commit_flush_success.load(Ordering::SeqCst) {
                break;
            }
        }

        self.group_commit_election.remove_waiter();
        Ok(())
    }

    /// Flush all queued commits in one batch with immediate retry on failure.
    ///
    /// Tries flush_batch() up to MAX_FLUSH_RETRIES times with 10ms back-off.
    /// On final failure: logs error, keeps ops in queue, keeps flush_needed = true.
    ///    so the next scheduled flusher wake-up will try again
    fn flush_batch_with_retry(&self) {
        for attempt in 0..Self::MAX_FLUSH_RETRIES {
            if self.group_commit_flush_success.load(Ordering::SeqCst) {
                return; // Another thread already succeeded
            }
            self.flush_batch();
            if self.group_commit_flush_success.load(Ordering::SeqCst) {
                return; // Our attempt succeeded
            }
            if attempt + 1 < Self::MAX_FLUSH_RETRIES {
                tracing::warn!(
                    "flush_batch attempt {}/{} failed, retrying in 10ms",
                    attempt + 1,
                    Self::MAX_FLUSH_RETRIES
                );
                thread::sleep(Duration::from_millis(10));
            }
        }
        // All retries exhausted -- log once, ops remain queued, flush_needed stays true
        tracing::error!(
            "flush_batch failed after {} attempts. \
             {} ops remain in queue, background flusher will retry on next wake-up.",
            Self::MAX_FLUSH_RETRIES,
            self.group_commit_queue_size.load(Ordering::SeqCst)
        );
    }

    /// Flush all queued commits in one batch (called by leader thread only).
    /// Internal helper -- callers should use `flush_batch_with_retry` for retries.
    fn flush_batch(&self) {
        // Signal failure by default -- set to true only after flush_and_sync succeeds
        self.group_commit_flush_success
            .store(false, Ordering::SeqCst);

        let ops: Vec<WalOp> = {
            let mut queue = self.group_commit_queue.lock();
            queue.drain(..).collect()
        };

        if ops.is_empty() {
            self.group_commit_flush_success
                .store(true, Ordering::SeqCst);
            return;
        }

        tracing::debug!(
            "GroupCommit appending {} entries via append_batch",
            ops.len()
        );

        if let Err(e) = (|| {
            let mut writer = self.writer.lock();
            let _entry_ids = writer
                .append_batch(&ops)
                .map_err(|e| RockDuckError::Write(format!("WAL append_batch: {e}")))?;
            writer
                .flush_and_sync()
                .map_err(|e| RockDuckError::Write(format!("WAL flush_and_sync: {e}")))
        })() {
            tracing::error!("GroupCommit batch flush failed: {}", e);
            let retries = self
                .group_commit_flush_retries
                .fetch_add(1, Ordering::SeqCst);
            tracing::warn!(
                "Requeueing {} ops for retry ({}/{})",
                ops.len(),
                retries + 1,
                Self::MAX_FLUSH_RETRIES
            );
            let queue_len = {
                let mut queue = self.group_commit_queue.lock();
                // Prepend ops at head to preserve original ordering (Bug #4 fix).
                for op in ops.into_iter().rev() {
                    queue.push_front(op);
                }
                queue.len()
            };
            self.group_commit_queue_size
                .store(queue_len, Ordering::SeqCst);
            // Keep flush_needed = true so the background flusher
            // knows to retry on its next wake-up. The flag must only be cleared
            // on successful flush (see success path below).
            self.group_commit_flush_success
                .store(false, Ordering::SeqCst);
            self.group_commit_flush_needed.store(true, Ordering::SeqCst);
            if retries + 1 >= Self::MAX_FLUSH_RETRIES {
                tracing::error!(
                    "GroupCommit flush failed after {} consecutive retries ({} ops requeued). \
                     Background flusher will continue retrying on the next wake-up.",
                    retries + 1,
                    queue_len
                );
            }
            return;
        }

        self.group_commit_flush_retries.store(0, Ordering::SeqCst);
        self.group_commit_flush_success
            .store(true, Ordering::SeqCst);
        *self.group_commit_last_sync.lock() = Instant::now();
        self.group_commit_queue_size.store(0, Ordering::SeqCst);
        self.group_commit_flush_needed
            .store(false, Ordering::SeqCst);
    }

    pub fn get_mut_writer(&self) -> parking_lot::MutexGuard<'_, walog::WalWriter<WalOp>> {
        self.writer.lock()
    }

    pub fn try_get_mut_writer(
        &self,
    ) -> Option<parking_lot::MutexGuard<'_, walog::WalWriter<WalOp>>> {
        self.writer.try_lock()
    }

    pub fn get_committed_txn(&self) -> u64 {
        self.committed_txn.load(Ordering::SeqCst)
    }

    pub fn estimated_size(&self) -> u64 {
        let guard = self.writer.lock();
        let seg_id = guard.current_segment_id();
        let seg_bytes = guard.current_segment_bytes();
        // Use saturating arithmetic to prevent overflow on large segment IDs
        seg_id
            .saturating_sub(1)
            .saturating_mul(self.config.max_file_size)
            .saturating_add(seg_bytes)
    }

    pub fn active_files(&self) -> Vec<PathBuf> {
        let dir = match durability::storage::FsDirectory::new(&self.wal_dir) {
            Ok(d) => d,
            Err(_) => return vec![],
        };
        match dir.list_dir("") {
            Ok(entries) => entries
                .into_iter()
                .map(|name| self.wal_dir.join(&name))
                .collect(),
            Err(_) => vec![],
        }
    }

    pub fn wal_dir(&self) -> &Path {
        &self.wal_dir
    }

    /// Returns the total size in bytes of all WAL segment files.
    pub fn size_bytes(&self) -> u64 {
        std::fs::read_dir(&self.wal_dir)
            .map(|entries| {
                entries
                    .flatten()
                    .filter_map(|e| e.metadata().ok())
                    .filter(|m| m.is_file())
                    .map(|m| m.len())
                    .sum()
            })
            .unwrap_or(0)
    }
}

impl WalConfig {
    /// Returns the effective sync policy for this WAL configuration.
    fn sync_policy(&self) -> SyncPolicy {
        self.group_commit
            .as_ref()
            .map(|c| c.policy)
            .unwrap_or(SyncPolicy::GroupCommitStrict)
    }
}

impl Drop for WalWriter {
    fn drop(&mut self) {
        // Signal the flusher thread to exit by dropping the Sender.
        // The channel disconnect causes recv_timeout to return Disconnected,
        // which triggers the flusher to break out of its loop.
        if let Some(tx) = self.group_commit_shutdown_tx.take() {
            drop(tx);
        }

        // Join the flusher thread to ensure it completes its final flush iteration.
        // The flusher drains the queue and syncs to WAL while the queue is still
        // populated. After this join, the queue is guaranteed to be empty (flusher
        // drained it) or the flusher has exited without draining (shouldn't happen
        // -- the flusher always drains before breaking). Joining before draining
        // eliminates the race where Drop's try_lock competes with the flusher's
        // flush_and_sync for the writer lock.
        if let Some(handle) = self.worker_handle.take() {
            // join() blocks until the flusher loop exits. On timeout, the flusher
            // will still see the Disconnected error and exit cleanly.
            let _ = handle.join();
        }

        // Drain any remaining entries (should be empty after flusher joined).
        // This is a safety net -- the flusher should have already drained everything.
        let pending: Vec<WalOp> = {
            let mut q = self.group_commit_queue.lock();
            q.drain(..).collect()
        };

        // Flush any remaining entries. This is now a last-resort safety net;
        // the flusher should have already written everything to WAL.
        if !pending.is_empty() {
            if let Some(mut writer) = self.writer.try_lock() {
                if let Err(e) = writer.append_batch(&pending) {
                    tracing::error!(
                        "WalWriter Drop: failed to append {} pending entries to WAL: {}",
                        pending.len(),
                        e
                    );
                } else if let Err(e) = writer.flush_and_sync() {
                    tracing::error!(
                        "WalWriter Drop: WAL flush_and_sync failed: {}. \
                         {} entries may not be durable.",
                        e,
                        pending.len()
                    );
                }
            }
            // If try_lock failed, the flusher join should have waited for the
            // flusher to release the lock, so this branch should be unreachable.
            // We keep it as a defensive fallback.
            else {
                tracing::warn!(
                    "WalWriter Drop: writer locked after flusher joined, {} pending \
                     entries may not be durable",
                    pending.len()
                );
            }
        }
    }
}
