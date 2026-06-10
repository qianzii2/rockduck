//! WAL Recovery: replays committed transactions to reconstruct database state.
//!
//! Recovery verification protocol for every change in this area:
//! 1. Confirm the main path: checkpoint load -> KV MVCC baseline -> WAL replay override.
//! 2. Confirm bypass/fallback paths: v1 checkpoint fallback, corrupt checkpoint fallback, replay_failure handling.
//! 3. Confirm landing files: checkpoint state, MVCC committed history, active txn reconstruction, db open wiring.
//!
//! Recovery phases:
//! 1. SCAN: durability's WalReader reads all WAL entries
//! 2. BUILD: collect operations by txn_id, filter to committed ones
//! 3. REPLAY: for each committed txn, apply() callback reconstructs the data
//!
//! Recovery is idempotent — safe to run multiple times.

use std::collections::{HashMap, HashSet};
use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::Arc;

use parking_lot::Mutex;

use crate::db::TxnId;
use crate::error::RockDuckError;
use crate::write::durability_wal::OpType;
pub use crate::write::durability_wal::WalOp;

// =============================================================================
// Error classification
// =============================================================================

/// Severity of a WAL replay error, determining what action to take.
#[derive(Debug, Clone)]
pub enum ReplayErrorKind {
    /// IO error that is safe to skip — the affected data was already durable
    /// (e.g., file already exists from a previous partial flush).
    Recoverable(String),
    /// WAL payload does not match the expected type — likely an old format
    /// version. Safe to skip but worth noting.
    Ambiguous(String),
    /// Arrow IPC data in the WAL entry is corrupt or unparseable. This txn
    /// cannot be safely replayed. Written to the replay_failure file.
    Corruption(String),
    /// Fatal I/O error from the WAL reader itself (file truncated, permission
    /// denied, etc.) — cannot continue recovery.
    Fatal(String),
}

impl std::fmt::Display for ReplayErrorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReplayErrorKind::Recoverable(msg) => write!(f, "recoverable: {msg}"),
            ReplayErrorKind::Ambiguous(msg) => write!(f, "ambiguous: {msg}"),
            ReplayErrorKind::Corruption(msg) => write!(f, "corruption: {msg}"),
            ReplayErrorKind::Fatal(msg) => write!(f, "fatal: {msg}"),
        }
    }
}

/// WAL replay error — carries a classified kind and details.
#[derive(Debug)]
pub struct ReplayError {
    pub kind: ReplayErrorKind,
    pub txn_id: TxnId,
}

impl std::fmt::Display for ReplayError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "WAL replay error ({}): {}", self.kind, self.txn_id)
    }
}

impl std::error::Error for ReplayError {}

/// Classify an apply error for WAL replay.
fn classify_apply_error(err: &RockDuckError, txn_id: TxnId) -> ReplayErrorKind {
    fn contains_any(haystack: &str, needles: &[&str]) -> bool {
        needles.iter().any(|needle| haystack.contains(needle))
    }

    let msg = err.to_string();
    match err {
        RockDuckError::Codec(_) | RockDuckError::Deserialize(_) => {
            ReplayErrorKind::Corruption(format!("Arrow IPC parse error for txn {}: {msg}", txn_id))
        }
        RockDuckError::Io(io_err)
            if matches!(
                io_err.kind(),
                std::io::ErrorKind::AlreadyExists | std::io::ErrorKind::PermissionDenied
            ) || contains_any(&msg, &["exists", "already"]) =>
        {
            ReplayErrorKind::Recoverable(format!(
                "IO conflict for txn {} (likely already applied): {msg}",
                txn_id
            ))
        }
        RockDuckError::Internal(inner)
            if contains_any(inner, &["Arrow", "IPC", "decode", "parse", "execute_arrow"]) =>
        {
            ReplayErrorKind::Corruption(format!("Arrow IPC parse error for txn {}: {msg}", txn_id))
        }
        _ => ReplayErrorKind::Ambiguous(format!(
            "apply classification unclear for txn {}: {msg}",
            txn_id
        )),
    }
}

/// Write a corruption record to the replay_failure file.
/// File format: one JSON record per line: {"txn_id": N, "reason": "..."}
fn write_replay_failure(wal_dir: &Path, txn_id: TxnId, reason: &str) -> std::io::Result<()> {
    let failure_path = wal_dir.join("replay_failure");
    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(failure_path)?;
    let mut writer = BufWriter::new(file);
    // Escape double quotes in the reason string for valid JSON.
    let escaped_reason = reason.replace('\\', "\\\\").replace('"', "\\\"");
    writeln!(
        writer,
        r#"{{"txn_id": {txn_id}, "reason": "{escaped_reason}"}}"#
    )?;
    writer.flush()
}

/// WAL recovery result.
#[derive(Debug)]
pub struct RecoveryResult {
    /// Last committed transaction ID that was both committed and durably replayed.
    pub last_committed_txn: u64,
    /// Highest committed transaction ID observed in the WAL, even if replay degraded.
    pub max_seen_committed_txn: u64,
    /// Number of committed transactions replayed.
    pub committed_count: usize,
    /// Number of uncommitted transactions discarded.
    pub discarded_count: usize,
    /// Active transactions recovered from WAL: (txn_id, begin_ts).
    /// These are committed txns whose data is still being replayed (not yet durable
    /// in data files at the time of crash). Recovered so MVCC visibility is correct.
    pub active_txns: Vec<(TxnId, u64)>,
    /// Commit timestamp map recovered from WAL Commit entries: txn_id -> commit_ts.
    /// Populated from OpPayload::Commit entries during replay. Used to populate
    /// the MVCC committed_history so snapshots have correct commit_ts_map after recovery.
    pub commit_ts_map: HashMap<TxnId, u64>,
    /// inserted_at timestamps recovered from WAL Commit entries: txn_id -> inserted_at.
    /// Old WAL entries (pre-D5) will have no entry here — recovery will use current wall-clock.
    pub inserted_at_map: HashMap<TxnId, u64>,
    /// d028: Committed WAL ops replayed — returned so callers can rebuild L1 state.
    /// Only Insert/Update/Delete ops from committed transactions are included.
    pub committed_ops: Vec<WalOp>,
    /// D7 fix: WAL replay watermark — maximum inserted_at among all recovered transactions.
    /// Used as lower bound for TTL eviction in VisibilityManager::prune_history.
    pub replay_watermark: u64,
}

#[allow(dead_code)]
fn committed_history_unavailable_error(source: &str, err: RockDuckError) -> RockDuckError {
    RockDuckError::Internal(format!(
        "historical commit authority unavailable in {source}: {err}"
    ))
}

/// Scan WAL and replay all committed operations.
///
/// Uses `durability::walog::WalReader::replay()` to read entries, groups by txn_id,
/// then calls `apply(op)` for each committed op in order.
///
/// Uncommitted transactions are logged and tracked as aborted.
///
/// Errors from `apply()` are classified into four kinds:
///
/// - **Recoverable**: IO conflicts (e.g., file already exists). Safe to skip.
/// - **Ambiguous**: Payload type mismatch (e.g., old format). Safe to skip.
/// - **Corruption**: Arrow IPC parse error. Written to `replay_failure` file and skipped.
/// - **Fatal**: WAL reader I/O error. Propagated up to abort recovery.
///
/// Returns `Err(ReplayError)` only for fatal errors. Recoverable, ambiguous, and
/// corruption errors are logged and the txn is skipped.
pub fn replay_committed_ops(
    wal_dir: &Path,
    apply: impl FnMut(&WalOp) -> Result<(), RockDuckError> + Send,
) -> Result<RecoveryResult, ReplayError> {
    let dir = durability::storage::FsDirectory::arc(wal_dir).map_err(|e| ReplayError {
        kind: ReplayErrorKind::Fatal(format!("FsDirectory::arc: {e}")),
        txn_id: 0,
    })?;

    let reader = durability::walog::WalReader::<WalOp>::new(dir);

    #[allow(clippy::type_complexity)]
    let apply_box: Box<dyn FnMut(&WalOp) -> Result<(), RockDuckError> + Send> = Box::new(apply);
    #[allow(clippy::type_complexity)]
    let apply_arc: Arc<Mutex<Box<dyn FnMut(&WalOp) -> Result<(), RockDuckError> + Send>>> =
        Arc::new(Mutex::new(apply_box));

    let pending: Arc<Mutex<HashMap<u64, Vec<WalOp>>>> = Arc::new(Mutex::new(
        HashMap::with_capacity_and_hasher(1024, Default::default()),
    ));
    let begin_records: Arc<Mutex<HashMap<TxnId, u64>>> = Arc::new(Mutex::new(HashMap::new()));
    let seen_begins: Arc<Mutex<HashSet<TxnId>>> = Arc::new(Mutex::new(HashSet::new()));
    let last_committed: Arc<Mutex<u64>> = Arc::new(Mutex::new(0));
    let max_seen_committed: Arc<Mutex<u64>> = Arc::new(Mutex::new(0));
    let committed_count: Arc<Mutex<usize>> = Arc::new(Mutex::new(0));
    let active_txns: Arc<Mutex<Vec<(TxnId, u64)>>> = Arc::new(Mutex::new(Vec::new()));
    let commit_ts_map: Arc<Mutex<HashMap<TxnId, u64>>> = Arc::new(Mutex::new(HashMap::new()));
    // D5: WAL-recovered inserted_at timestamps for committed_history TTL clock.
    // Old WAL entries (pre-D5) have no inserted_at, so recovery falls back to wall-clock.
    let inserted_at_map: Arc<Mutex<HashMap<TxnId, u64>>> = Arc::new(Mutex::new(HashMap::new()));
    // D7 fix: tracks the maximum inserted_at among all recovered transactions.
    // Used as replay_watermark to bound TTL eviction from below.
    let max_inserted_at: Arc<Mutex<u64>> = Arc::new(Mutex::new(0));
    // d028: collects all committed ops for L1 recovery
    let committed_ops: Arc<Mutex<Vec<WalOp>>> = Arc::new(Mutex::new(Vec::new()));
    let failed_count: Arc<Mutex<usize>> = Arc::new(Mutex::new(0));
    let corruption_count: Arc<Mutex<usize>> = Arc::new(Mutex::new(0));

    let fatal_error: Arc<Mutex<Option<ReplayError>>> = Arc::new(Mutex::new(None));
    let fatal_error2 = Arc::clone(&fatal_error);

    let pending2 = Arc::clone(&pending);
    let begin_records2 = Arc::clone(&begin_records);
    let seen_begins2 = Arc::clone(&seen_begins);
    let last_committed2 = Arc::clone(&last_committed);
    let max_seen_committed2 = Arc::clone(&max_seen_committed);
    let committed_count2 = Arc::clone(&committed_count);
    let active_txns2 = Arc::clone(&active_txns);
    let commit_ts_map2 = Arc::clone(&commit_ts_map);
    let inserted_at_map2 = Arc::clone(&inserted_at_map);
    let max_inserted_at2 = Arc::clone(&max_inserted_at);
    let committed_ops2 = Arc::clone(&committed_ops);
    let apply_arc2 = Arc::clone(&apply_arc);
    let failed_count2 = Arc::clone(&failed_count);
    let corruption_count2 = Arc::clone(&corruption_count);
    let wal_dir_arc = wal_dir.to_path_buf();

    let _final_lsn = reader
        .replay_each(move |rec| {
        let op = rec.payload;

        match op.op_type {
            OpType::Insert | OpType::Delete | OpType::Update => {
                pending2.lock().entry(op.txn_id).or_default().push(op);
            }
            OpType::Begin => {
                pending2.lock().entry(op.txn_id).or_default();
                seen_begins2.lock().insert(op.txn_id);
                begin_records2.lock().insert(op.txn_id, op.txn_id);
            }
            OpType::Commit => {
                {
                    let mut max_seen = max_seen_committed2.lock();
                    if op.txn_id > *max_seen {
                        *max_seen = op.txn_id;
                    }
                }
                // Fix W-11: Changed unwrap_or_default to proper error handling.
                // Use unwrap_or(Vec::new()) since pending may not exist for duplicate commits.
                // Log warning when this happens to detect anomalies.
                let ops: Vec<WalOp> = pending2.lock().remove(&op.txn_id)
                    .unwrap_or_else(|| {
                        tracing::warn!(
                            "WAL replay: no pending ops found for committed txn {}",
                            op.txn_id
                        );
                        Vec::new()
                    });
                let begin_from_begin = begin_records2.lock().remove(&op.txn_id);
                let saw_begin = seen_begins2.lock().remove(&op.txn_id);
                let (begin_ts, commit_ts) = match &op.payload {
                    crate::write::durability_wal::OpPayload::Commit {
                        begin_ts,
                        inserted_at,
                    } => {
                        let ts = *begin_ts;
                        if let Some(begin_record_ts) = begin_from_begin {
                            if begin_record_ts != ts {
                                tracing::warn!(
                                    txn_id = op.txn_id,
                                    commit_begin_ts = ts,
                                    begin_record_ts,
                                    "WAL replay: Commit begin_ts mismatch with Begin record; using Commit payload as recovery authority"
                                );
                            }
                        } else if !saw_begin {
                            tracing::warn!(
                                txn_id = op.txn_id,
                                commit_begin_ts = ts,
                                "WAL replay: Commit observed without matching Begin record; using Commit payload as recovery authority"
                            );
                        }
                        if let Some(ia) = inserted_at {
                            // D5 fix: extract inserted_at from WAL so WAL-recovered entries
                            // share the same TTL clock as live-committed entries.
                            inserted_at_map2.lock().insert(op.txn_id, *ia);
                            // D7 fix: track max inserted_at for replay_watermark
                            let mut max_ia = max_inserted_at2.lock();
                            if *ia > *max_ia {
                                *max_ia = *ia;
                            }
                        }
                        (ts, ts)
                    }
                    _ => {
                        tracing::warn!(
                            "WAL replay: OpType::Commit with non-Commit payload for txn {}, \
                             using txn_id as fallback begin_ts",
                            op.txn_id
                        );
                        (op.txn_id, op.txn_id)
                    }
                };
                commit_ts_map2.lock().insert(op.txn_id, commit_ts);
                active_txns2.lock().push((op.txn_id, begin_ts));
                if !ops.is_empty() {
                    let mut apply = apply_arc2.lock();
                    let mut txn_failed = false;
                    for committed_op in &ops {
                        match apply(committed_op) {
                            Ok(()) => {}
                            Err(e) => {
                                let kind = classify_apply_error(&e, op.txn_id);
                                match &kind {
                                    ReplayErrorKind::Fatal(msg) => {
                                        tracing::error!(
                                            "WAL replay fatal error for txn {}: {}. Aborting recovery.",
                                            op.txn_id, msg
                                        );
                                        let fatal = ReplayError {
                                            kind,
                                            txn_id: op.txn_id,
                                        };
                                        fatal_error2.lock().replace(fatal);
                                        return Ok(());
                                    }
                                    ReplayErrorKind::Corruption(msg) => {
                                        tracing::error!(
                                            "WAL replay corruption detected for txn {}: {}. \
                                             Writing to replay_failure and skipping.",
                                            op.txn_id, msg
                                        );
                                        if let Err(io_err) = write_replay_failure(
                                            &wal_dir_arc, op.txn_id, msg,
                                        ) {
                                            tracing::warn!(
                                                "Failed to write replay_failure file: {}",
                                                io_err
                                            );
                                        }
                                        *corruption_count2.lock() += 1;
                                    }
                                    ReplayErrorKind::Recoverable(msg) => {
                                        tracing::warn!(
                                            "WAL replay recoverable error for txn {}: {}. Skipping.",
                                            op.txn_id, msg
                                        );
                                    }
                                    ReplayErrorKind::Ambiguous(msg) => {
                                        tracing::warn!(
                                            "WAL replay ambiguous error for txn {}: {}. Skipping.",
                                            op.txn_id, msg
                                        );
                                    }
                                }
                                txn_failed = true;
                                break;
                            }
                        }
                    }
                    drop(apply);
                    if !txn_failed {
                        let mut lc = last_committed2.lock();
                        if op.txn_id > *lc {
                            *lc = op.txn_id;
                        }
                        *committed_count2.lock() += 1;
                        // d028: collect committed ops for L1 recovery
                        for committed_op in &ops {
                            committed_ops2.lock().push(committed_op.clone());
                        }
                    } else {
                        *failed_count2.lock() += 1;
                    }
                }
            }
            OpType::Rollback => {
                pending2.lock().remove(&op.txn_id);
            }
            OpType::Checkpoint | OpType::Compaction => {}
        }
        Ok(())
        })
        .map_err(|e| ReplayError {
            kind: ReplayErrorKind::Fatal(format!("WAL reader error: {e}")),
            txn_id: 0,
        })?;

    if let Some(fatal) = fatal_error.lock().take() {
        return Err(fatal);
    }

    // Read counters and clone state for the recovery summary.
    let last_txn = *last_committed.lock();
    let max_seen_txn = *max_seen_committed.lock();
    let commit_cnt = *committed_count.lock();
    let failed = *failed_count.lock();
    // SAFETY: `pending` is exclusively owned here. All committed and rolled-back
    // txn_ids are removed from `pending2` inside the replay closure (lines 241, 344).
    // `pending` contains entries for txns that crashed before their Commit/Rollback
    // record was written — these represent genuinely aborted transactions.
    // The only other access to `pending` is the read-only `.len()` below (line 363),
    // which is used purely for the diagnostic warning. No external code accesses `pending`.
    let aborted = pending.lock().len();
    let recovered_active = active_txns.lock().clone();
    let recovered_commit_ts = commit_ts_map.lock().clone();
    let recovered_inserted_at = inserted_at_map.lock().clone();
    let replay_watermark = *max_inserted_at.lock();

    // d028: collect all committed ops for L1 recovery
    let committed_ops = committed_ops.lock().clone();

    if aborted > 0 {
        tracing::warn!(
            "WAL recovery: {} uncommitted transactions discarded (crash before commit) — \
             treating these as aborted",
            aborted
        );
    }

    if failed > 0 {
        tracing::warn!(
            "WAL recovery: {} transactions skipped due to apply failure",
            failed
        );
    }

    tracing::info!(
        "WAL recovery: last_replayed_committed_txn={}, max_seen_committed_txn={}, committed={}, failed={}, aborted={}, active_txns={}",
        last_txn, max_seen_txn, commit_cnt, failed, aborted, recovered_active.len()
    );

    Ok(RecoveryResult {
        last_committed_txn: last_txn,
        max_seen_committed_txn: max_seen_txn,
        committed_count: commit_cnt,
        discarded_count: aborted,
        active_txns: recovered_active,
        commit_ts_map: recovered_commit_ts,
        inserted_at_map: recovered_inserted_at,
        committed_ops,
        replay_watermark,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::write::durability_wal::OpPayload;
    use tempfile::TempDir;

    fn write_raw_wal_entries(wal_dir: &std::path::Path, entries: &[WalOp]) {
        let dir = durability::storage::FsDirectory::arc(wal_dir).expect("fs dir");
        let mut writer = durability::walog::WalWriter::open(dir).expect("open raw wal writer");
        for entry in entries {
            writer.append(entry).expect("append wal entry");
        }
        writer.flush().expect("flush raw wal writer");
    }

    #[test]
    fn replay_commit_warns_when_begin_record_is_missing() {
        let temp = TempDir::new().expect("tempdir");
        let wal_dir = temp.path().join("wal");
        std::fs::create_dir_all(&wal_dir).expect("create wal dir");

        let commit = WalOp {
            op_type: OpType::Commit,
            txn_id: 9,
            payload: OpPayload::Commit { begin_ts: 77, inserted_at: None },
        };

        let dir = durability::storage::FsDirectory::arc(&wal_dir).expect("fs dir");
        let mut writer = durability::walog::WalWriter::open(dir).expect("open raw wal writer");
        writer.append(&commit).expect("append commit");
        writer.flush().expect("flush raw wal writer");
        drop(writer);

        let result = replay_committed_ops(&wal_dir, |_op| Ok(())).expect("replay should succeed");

        assert_eq!(result.active_txns, vec![(9, 77)]);
        assert_eq!(result.commit_ts_map.get(&9), Some(&77));
        assert_eq!(result.max_seen_committed_txn, 9);
    }

    #[test]
    fn replay_uncommitted_txn_is_discarded_after_crash() {
        let temp = TempDir::new().expect("tempdir");
        let wal_dir = temp.path().join("wal");
        std::fs::create_dir_all(&wal_dir).expect("create wal dir");

        let begin = WalOp {
            op_type: OpType::Begin,
            txn_id: 5,
            payload: OpPayload::Begin,
        };
        let insert = WalOp {
            op_type: OpType::Insert,
            txn_id: 5,
            payload: OpPayload::Insert {
                table: "t".to_string(),
                pk: [5; 8],
                columns: vec!["value".to_string()],
                wal_batch: Arc::new(vec![0x01, 0x02]),
                schema_bytes: Vec::new(),
                seg_id: "seg-5".to_string(),
                granule_id: crate::metadata::GranuleId::zero(),
                offset: 0,
            },
        };

        let dir = durability::storage::FsDirectory::arc(&wal_dir).expect("fs dir");
        let mut writer = durability::walog::WalWriter::open(dir).expect("open raw wal writer");
        writer.append(&begin).expect("append begin");
        writer.append(&insert).expect("append insert");
        writer.flush().expect("flush raw wal writer");
        drop(writer);

        let result = replay_committed_ops(&wal_dir, |_op| Ok(())).expect("replay should succeed");

        assert_eq!(result.last_committed_txn, 0);
        assert_eq!(result.max_seen_committed_txn, 0);
        assert_eq!(result.committed_count, 0);
        assert_eq!(result.discarded_count, 1);
        assert!(result.active_txns.is_empty());
        assert!(result.commit_ts_map.is_empty());
    }

    #[test]
    fn replay_commit_uses_commit_payload_begin_ts_when_begin_mismatches() {
        let temp = TempDir::new().expect("tempdir");
        let wal_dir = temp.path().join("wal");
        std::fs::create_dir_all(&wal_dir).expect("create wal dir");

        let begin = WalOp {
            op_type: OpType::Begin,
            txn_id: 5,
            payload: OpPayload::Begin,
        };
        let commit = WalOp {
            op_type: OpType::Commit,
            txn_id: 5,
            payload: OpPayload::Commit { begin_ts: 99, inserted_at: None },
        };

        let dir = durability::storage::FsDirectory::arc(&wal_dir).expect("fs dir");
        let mut writer = durability::walog::WalWriter::open(dir).expect("open raw wal writer");
        writer.append(&begin).expect("append begin");
        writer.append(&commit).expect("append commit");
        writer.flush().expect("flush raw wal writer");
        drop(writer);

        let result = replay_committed_ops(&wal_dir, |_op| Ok(())).expect("replay should succeed");

        assert_eq!(result.active_txns, vec![(5, 99)]);
        assert_eq!(result.commit_ts_map.get(&5), Some(&99));
        assert_eq!(result.max_seen_committed_txn, 5);
    }

    #[test]
    fn replay_failure_records_corruption_without_aborting_recovery() {
        let temp = TempDir::new().expect("tempdir");
        let wal_dir = temp.path().join("wal");
        std::fs::create_dir_all(&wal_dir).expect("create wal dir");

        let begin = WalOp {
            op_type: OpType::Begin,
            txn_id: 7,
            payload: OpPayload::Begin,
        };
        let insert = WalOp {
            op_type: OpType::Insert,
            txn_id: 7,
            payload: OpPayload::Insert {
                table: "t".to_string(),
                pk: [7; 8],
                columns: vec!["value".to_string()],
                wal_batch: Arc::new(vec![0xff, 0x00, 0xaa, 0x55]),
                schema_bytes: Vec::new(),
                seg_id: "seg-7".to_string(),
                granule_id: crate::metadata::GranuleId::zero(),
                offset: 0,
            },
        };
        let commit = WalOp {
            op_type: OpType::Commit,
            txn_id: 7,
                payload: OpPayload::Commit { begin_ts: 7, inserted_at: None },
        };

        let dir = durability::storage::FsDirectory::arc(&wal_dir).expect("fs dir");
        let mut writer = durability::walog::WalWriter::open(dir).expect("open raw wal writer");
        writer.append(&begin).expect("append begin");
        writer.append(&insert).expect("append insert");
        writer.append(&commit).expect("append commit");
        writer.flush().expect("flush raw wal writer");
        drop(writer);

        let result = replay_committed_ops(&wal_dir, |op| match op.op_type {
            OpType::Insert => Err(RockDuckError::Codec(
                "synthetic Arrow IPC corruption".into(),
            )),
            _ => Ok(()),
        })
        .expect("corruption should degrade, not abort");

        assert_eq!(result.last_committed_txn, 0);
        assert_eq!(result.max_seen_committed_txn, 7);
        assert_eq!(result.committed_count, 0);
        assert_eq!(result.discarded_count, 0);
        assert_eq!(result.commit_ts_map.get(&7), Some(&7));

        let replay_failure = std::fs::read_to_string(wal_dir.join("replay_failure"))
            .expect("replay_failure should exist");
        assert!(replay_failure.contains("\"txn_id\": 7"));
        assert!(replay_failure.contains("synthetic Arrow IPC corruption"));
    }

    #[test]
    fn replay_tracks_highest_committed_txn_across_mixed_outcomes() {
        let temp = TempDir::new().expect("tempdir");
        let wal_dir = temp.path().join("wal");
        std::fs::create_dir_all(&wal_dir).expect("create wal dir");

        let committed_ok = [
            WalOp {
                op_type: OpType::Begin,
                txn_id: 10,
                payload: OpPayload::Begin,
            },
            WalOp {
                op_type: OpType::Insert,
                txn_id: 10,
                payload: OpPayload::Insert {
                    table: "t".to_string(),
                    pk: [10; 8],
                    columns: vec!["value".to_string()],
                    wal_batch: Arc::new(vec![0x01]),
                    schema_bytes: Vec::new(),
                    seg_id: "seg-10".to_string(),
                    granule_id: crate::metadata::GranuleId::zero(),
                    offset: 0,
                },
            },
            WalOp {
                op_type: OpType::Commit,
                txn_id: 10,
                payload: OpPayload::Commit { begin_ts: 10, inserted_at: None },
            },
        ];
        let committed_corrupt = [
            WalOp {
                op_type: OpType::Begin,
                txn_id: 11,
                payload: OpPayload::Begin,
            },
            WalOp {
                op_type: OpType::Insert,
                txn_id: 11,
                payload: OpPayload::Insert {
                    table: "t".to_string(),
                    pk: [11; 8],
                    columns: vec!["value".to_string()],
                    wal_batch: Arc::new(vec![0x02]),
                    schema_bytes: Vec::new(),
                    seg_id: "seg-11".to_string(),
                    granule_id: crate::metadata::GranuleId::zero(),
                    offset: 0,
                },
            },
            WalOp {
                op_type: OpType::Commit,
                txn_id: 11,
                payload: OpPayload::Commit { begin_ts: 11, inserted_at: None },
            },
        ];
        let uncommitted = [
            WalOp {
                op_type: OpType::Begin,
                txn_id: 12,
                payload: OpPayload::Begin,
            },
            WalOp {
                op_type: OpType::Insert,
                txn_id: 12,
                payload: OpPayload::Insert {
                    table: "t".to_string(),
                    pk: [12; 8],
                    columns: vec!["value".to_string()],
                    wal_batch: Arc::new(vec![0x03]),
                    schema_bytes: Vec::new(),
                    seg_id: "seg-12".to_string(),
                    granule_id: crate::metadata::GranuleId::zero(),
                    offset: 0,
                },
            },
        ];

        let mut entries = Vec::new();
        entries.extend_from_slice(&committed_ok);
        entries.extend_from_slice(&committed_corrupt);
        entries.extend_from_slice(&uncommitted);
        write_raw_wal_entries(&wal_dir, &entries);

        let result = replay_committed_ops(&wal_dir, |op| match op.txn_id {
            11 if matches!(op.op_type, OpType::Insert) => {
                Err(RockDuckError::Codec("corrupt committed txn".into()))
            }
            _ => Ok(()),
        })
        .expect("corruption should degrade, not abort");

        assert_eq!(result.last_committed_txn, 10);
        assert_eq!(result.max_seen_committed_txn, 11);
        assert_eq!(result.committed_count, 1);
        assert_eq!(result.discarded_count, 1);
        assert_eq!(result.active_txns, vec![(10, 10), (11, 11)]);
        assert_eq!(result.commit_ts_map.get(&10), Some(&10));
        assert_eq!(result.commit_ts_map.get(&11), Some(&11));
    }

    #[test]
    fn replay_classifies_already_applied_io_conflicts_as_recoverable() {
        let temp = TempDir::new().expect("tempdir");
        let wal_dir = temp.path().join("wal");
        std::fs::create_dir_all(&wal_dir).expect("create wal dir");

        let entries = [
            WalOp {
                op_type: OpType::Begin,
                txn_id: 21,
                payload: OpPayload::Begin,
            },
            WalOp {
                op_type: OpType::Insert,
                txn_id: 21,
                payload: OpPayload::Insert {
                    table: "t".to_string(),
                    pk: [21; 8],
                    columns: vec!["value".to_string()],
                    wal_batch: Arc::new(vec![0x21]),
                    schema_bytes: Vec::new(),
                    seg_id: "seg-21".to_string(),
                    granule_id: crate::metadata::GranuleId::zero(),
                    offset: 0,
                },
            },
            WalOp {
                op_type: OpType::Commit,
                txn_id: 21,
                payload: OpPayload::Commit { begin_ts: 21, inserted_at: None },
            },
        ];
        write_raw_wal_entries(&wal_dir, &entries);

        let result = replay_committed_ops(&wal_dir, |_op| {
            Err(RockDuckError::Io(std::io::Error::new(
                std::io::ErrorKind::AlreadyExists,
                "segment already exists",
            )))
        })
        .expect("already-applied IO conflicts should degrade, not abort");

        assert_eq!(result.last_committed_txn, 0);
        assert_eq!(result.max_seen_committed_txn, 21);
        assert_eq!(result.committed_count, 0);
        assert_eq!(result.commit_ts_map.get(&21), Some(&21));
    }
}
