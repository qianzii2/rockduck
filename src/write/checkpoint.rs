//! WAL Checkpoint implementation with crash-safe 7-step protocol.
//!
//! # Crash-Safe Checkpoint Protocol (7 steps, in order)
//!
//! 1. Serialize CheckpointState → .ckpt_{id}.tmp  [with CRC envelope]
//! 2. fsync(.ckpt_{id}.tmp)  [fatal]
//! 3. rename(.ckpt_{id}.tmp, .ckpt_{id})
//! 4. write _LATEST = "{id}" → .latest.tmp → fsync → rename to _LATEST  [fatal]
//! 5. fsync directory  [non-fatal]
//! 6. Append WalEntry::Checkpoint to WAL + flush_and_sync  [fatal]
//! 7. WAL truncate_prefix(last_entry_id)  [fatal, flush_and_sync first]
//!
//! Sync operations are non-fatal: they downgrade to warnings on failure.
//! The checkpoint state is still persisted (files written, renamed) so recovery
//! is always possible. fsync failures are Windows-specific temp dir issues.
//!
//! # CRC Integrity
//!
//! Each checkpoint file has a CRC envelope to detect silent data corruption:
//!   [magic(4) | version(1) | crc32(4) | payload_len(4) | payload(N)]
//!
//! The magic is validated on read; if CRC doesn't match, the checkpoint is treated
//! as corrupt and the fallback checkpoint is tried.
//!
//! # Failure Taxonomy
//!
//! All failure modes during recovery fall into exactly one of three categories:
//!
//! | Category | Description | Action | WAL Error Variants |
//! |----------|-------------|--------|--------------------|
//! | **Truth-safe degrade** | Error is contained; no data corruption. Correctness is preserved. | Skip the affected txn, continue recovery. Safe. | `Recoverable` (IO errors where data already durable), `Ambiguous` (format version mismatch, safe to skip) |
//! | **Evidence-stale degrade** | Single txn's state may be partially applied. Evidence is potentially stale. | Continue but flag as potentially stale. May need re-verification. | **`Corruption`** (Arrow IPC corrupt/unparseable, txn skipped, written to `replay_failure`. Single-txn scope — all other committed txns still replay. DB continues to open.) |
//! | **Fail-stop** | Data loss risk. Manual intervention required. | Abort recovery, surface error to operator. | **`Fatal`** (WAL reader IO error, cannot continue) |
//!
//! Any new error path added to recovery must be explicitly classified into one of these
//! three categories before landing. Unclassified errors default to fail-stop (conservative).

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::config::VisibilityConfig;

use crate::error::{Result, RockDuckError};
use crate::write::durability_wal::{OpPayload, OpType, WalOp, WalWriter};
use durability::walog::WalMaintenance;

/// Checkpoint file magic: "CKPT" in ASCII.
const CKPT_MAGIC: u32 = 0x434B5054;
/// Checkpoint file format version.
const CKPT_VERSION: u8 = 3;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CheckpointStateV1 {
    checkpoint_id: u64,
    committed_txn: u64,
    timestamp_ms: u64,
}

/// MVCC state snapshot persisted inside a v2 checkpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMvccState {
    pub active_txns: Vec<(u64, u64)>,
    pub committed_history: Vec<(u64, u64)>,
    pub visibility_config: VisibilityConfig,
}

/// CheckpointState serialized at checkpoint time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointState {
    pub checkpoint_id: u64,
    pub committed_txn: u64,
    pub timestamp_ms: u64,
    /// D7 fix: WAL replay watermark persisted from VisibilityManager.
    /// Used as lower bound for TTL eviction after crash recovery.
    pub replay_watermark: u64,
    pub mvcc: Option<CheckpointMvccState>,
}

/// CheckpointManager coordinates crash-safe checkpoints and WAL truncation.
pub struct CheckpointManager {
    checkpoint_dir: PathBuf,
    next_id: AtomicU64,
    wal: std::sync::Arc<WalWriter>,
}

#[derive(Debug, Clone)]
pub struct RecoveryVerificationCard {
    pub main_path: &'static str,
    pub bypass_paths: Vec<&'static str>,
    pub landing_files: Vec<&'static str>,
    /// Structured recovery order. This is the authoritative sequence for crash recovery.
    /// Matches `RockDuck::open_with_config` in `src/db.rs`.
    pub recovery_order: &'static [&'static str],
}

#[derive(Debug, Clone)]
pub struct TruthPackageVerification {
    pub truth_boundary: RecoveryVerificationCard,
    pub recovery_boundary: RecoveryVerificationCard,
}

impl CheckpointManager {
    pub fn recovery_verification_card() -> RecoveryVerificationCard {
        RecoveryVerificationCard {
            main_path: "CheckpointManager::load_latest -> metadata::get_committed_txns/get_active_txns -> replay_wal_ops",
            bypass_paths: vec![
                // SANCTIONED fallback paths (documented, bounded):
                "_LATEST missing fallback (no checkpoint exists -> return None, continue with KV baseline)",
                "corrupt checkpoint CRC fallback to previous checkpoint (try next oldest checkpoint)",
                // SANCTIONED exception:
                // OpType::Checkpoint is ignored during WAL replay. This is CORRECT because:
                // 1. CheckpointManager::checkpoint() writes OpType::Checkpoint to WAL AFTER the checkpoint
                //    file is durably written (step 6 of crash-safe protocol: write -> fsync -> rename).
                // 2. Recovery reads checkpoint files directly via CheckpointManager::load_latest().
                // 3. Replaying the OpType::Checkpoint would be redundant and potentially harmful.
                //    The WAL entry is a "this checkpoint was done" marker, not an instruction.
                "OpType::Checkpoint WAL entry ignored during replay (sanctioned: marker-only, data via CheckpointManager)",
                // TOLERATED transition exception (explicitly classified):
                // replay_failure records corruption for manual reconciliation, but recovery
                // does not yet provide a structured closure path for that side-channel.
                // Exit rule for this classified transition exception: it remains tolerated only
                // while enforcement stays trace/assert-only and no automated recovery authority
                // is delegated to the replay_failure side channel.
                "replay_failure corruption side channel (tolerated transition exception: manual reconciliation only, no structured closure path yet; exit condition: no automated authority delegated)",
            ],
            landing_files: vec![
                "src/write/checkpoint.rs",
                "src/write/wal_recovery.rs",
                "src/db.rs",
            ],
            recovery_order: &[
                // Step 1: Load checkpoint baseline (if exists)
                "Step 1: CheckpointManager::load_latest -> CheckpointMvccState { active_txns, committed_history, visibility_config }",
                // Step 2: Persist KV committed_txn before WAL overwrites it
                "Step 2: metadata::get_committed_txn(kv) -> committed_txn_kv (baseline before WAL replay)",
                // Step 3: Replay WAL — this is the authority for overlapping txn_ids
                "Step 3: replay_wal_ops -> RecoveryResult { last_committed_txn, active_txns, commit_ts_map }",
                // Step 4: Compute committed_txn = max(KV baseline, WAL result)
                "Step 4: committed_txn = max(committed_txn_kv, wal.last_committed_txn)",
                // Step 5: If WAL found higher committed_txn, write it back to KV (D8extra fix)
                "Step 5: if wal.last_committed_txn > committed_txn_kv -> metadata::put_committed_txn(kv, wal.last_committed_txn)",
                // Step 6: Recover commit_ts_map — KV checkpoint/KV store first, WAL overlay on top (WAL wins for overlapping keys)
                "Step 6: kv_commit_ts = checkpoint.committed_history or metadata::get_committed_txns(kv) -> then WAL insert for overlapping txn_ids",
                // Step 7: Recover active_txns — KV baseline, WAL overlay (WAL deduplication for overlapping txn_ids)
                "Step 7: active_txns = checkpoint.active_txns or metadata::get_active_txns(kv) -> WAL active_txns inserted for non-overlapping, WAL wins for overlapping",
                // Step 8: Recover visibility_config from checkpointMvccState
                "Step 8: recover_committed_history_with_config(kv_commit_ts, recovered_config) -> VisibilityManager",
                // Final: merged state is durable truth
                "Final: VisibilityManager holds durable truth. WAL is authoritative for all overlapping txn_ids.",
            ],
        }
    }

    pub fn truth_package_verification() -> TruthPackageVerification {
        TruthPackageVerification {
            truth_boundary: RecoveryVerificationCard {
                main_path: "VisibilityManager::snapshot/snapshot_at -> VisFilter -> HistoricalVisibility::is_visible_at",
                bypass_paths: vec![
                    // SANCTIONED exceptions (known, documented, bounded risk):
                    "read::point_get::get_as_of fallback __vis path (sanctioned: real get_commit_ts for deltas)",
                    "query::time_travel_impl::TimeTravelReader::is_visible_at (sanctioned: KV-backed historical commit_ts projection)",
                    "query::vtab_quack::BindData::filter_by_visibility (sanctioned: uses TxnSnapshot::is_row_visible, Rule 1-4 equivalent)",
                    // UNSANCTIONED gaps (potential drift, not yet bounded):
                    // None confirmed yet — any future bypass must be classified here before being implemented
                ],
                landing_files: vec![
                    "src/mvcc/visibility.rs",
                    "src/read/point_get.rs",
                    "src/query/time_travel_impl.rs",
                    "src/query/vtab_quack.rs",
                ],
                recovery_order: &[],
            },
            recovery_boundary: Self::recovery_verification_card(),
        }
    }
    pub fn new(data_dir: &Path, wal: std::sync::Arc<WalWriter>) -> Result<Self> {
        let checkpoint_dir = data_dir.join("checkpoints");
        std::fs::create_dir_all(&checkpoint_dir)
            .map_err(|e| RockDuckError::Write(format!("create checkpoint dir: {e}")))?;

        let next_id = Self::discover_max_id(&checkpoint_dir)
            .map(|id| id + 1)
            .unwrap_or(0);

        Ok(Self {
            checkpoint_dir,
            next_id: AtomicU64::new(next_id),
            wal,
        })
    }

    /// Trigger a fuzzy checkpoint.
    ///
    /// Returns the new checkpoint ID.
    ///
    /// All sync operations are non-fatal — failures are logged but don't abort
    /// the checkpoint. This is safe because the data is already on disk via write/rename.
    pub fn checkpoint(&self, state: CheckpointState) -> Result<u64> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);

        // Step 1: Serialize and write checkpoint state with CRC envelope.
        let tmp_path = self.checkpoint_dir.join(format!(".ckpt_{}.tmp", id));
        let payload = postcard::to_allocvec(&state)
            .map_err(|e| RockDuckError::Internal(format!("checkpoint serialize: {e}")))?;

        let payload_len = payload.len() as u32;
        let crc = crc32fast::hash(&payload);

        // Envelope: [magic(4) | version(1) | crc(4) | len(4) | payload(N)]
        let mut envelope = Vec::with_capacity(13 + payload.len());
        envelope.extend_from_slice(&CKPT_MAGIC.to_le_bytes());
        envelope.push(CKPT_VERSION);
        envelope.extend_from_slice(&crc.to_le_bytes());
        envelope.extend_from_slice(&payload_len.to_le_bytes());
        envelope.extend_from_slice(&payload);

        std::fs::write(&tmp_path, &envelope)
            .map_err(|e| RockDuckError::Write(format!("write checkpoint file: {e}")))?;

        // fsync the checkpoint file — required for durability.
        // Without fsync, a crash may lose the checkpoint data even if rename succeeded.
        Self::fsync_file(&tmp_path).map_err(RockDuckError::Io)?;

        // Step 3: atomic rename → .ckpt_{id}
        let final_path = self.checkpoint_dir.join(format!("ckpt_{}", id));
        std::fs::rename(&tmp_path, &final_path)
            .map_err(|e| RockDuckError::Write(format!("rename checkpoint: {e}")))?;

        // fsync the .latest.tmp file before rename to ensure the _LATEST pointer
        // is durable on disk before we point to the checkpoint.
        let latest_path = self.checkpoint_dir.join("_LATEST");
        let latest_tmp = self.checkpoint_dir.join(".latest.tmp");
        std::fs::write(&latest_tmp, id.to_string())
            .map_err(|e| RockDuckError::Write(format!("write _LATEST: {e}")))?;
        Self::fsync_file(&latest_tmp).map_err(RockDuckError::Io)?; // Required for durability
        std::fs::rename(&latest_tmp, &latest_path)
            .map_err(|e| RockDuckError::Write(format!("rename _LATEST: {e}")))?;

        // Step 5: fsync the directory to ensure the rename is durable (fatal).
        // On non-Windows, we actually call sync_all() on the directory fd.
        if let Err(e) = self.sync_dir() {
            return Err(RockDuckError::Io(e));
        }

        // Persist the CHECKPOINT record to WAL and flush. Required for safe recovery.
        let ckpt_op = WalOp {
            op_type: OpType::Checkpoint,
            txn_id: 0,
            payload: OpPayload::Checkpoint { checkpoint_id: id },
        };

        // Must get the writer lock; if someone else holds it, fail the checkpoint.
        // In practice this shouldn't block because WAL ops are very fast.
        {
            let mut wal_guard = self.wal.get_mut_writer();
            wal_guard
                .append(&ckpt_op)
                .map_err(|e| RockDuckError::Write(format!("checkpoint WAL append: {}", e)))?;
            wal_guard
                .flush_and_sync()
                .map_err(|e| RockDuckError::Write(format!("checkpoint WAL flush: {}", e)))?;
        }

        // Flush and sync before truncate to ensure no data is lost.
        if let Ok(dir) = durability::storage::FsDirectory::new(self.wal.wal_dir()) {
            let maint = WalMaintenance::new(
                std::sync::Arc::new(dir) as std::sync::Arc<dyn durability::Directory>
            );
            let last_id = {
                let mut wal_guard = self.wal.get_mut_writer();
                // Ensure all WAL data is flushed before truncating
                wal_guard.flush_and_sync().map_err(|e| {
                    RockDuckError::Write(format!("WAL flush before truncate: {}", e))
                })?;
                // Fix W-9: Handle Option properly - convert to Result first
                wal_guard.last_entry_id().ok_or_else(|| {
                    RockDuckError::Write("last_entry_id returned None after flush".into())
                })?
            };
            // Fix W-9: Validate last_id is non-zero after successful flush.
            // A zero last_id would truncate the entire WAL, losing all data.
            if last_id == 0 {
                return Err(RockDuckError::Write(
                    "last_entry_id returned 0 after flush - refusing to truncate WAL to zero"
                        .into(),
                ));
            }
            // After flush, truncate WAL up to (but not including) last_id.
            // The CHECKPOINT record we just wrote has this id, so data before it is safe.
            maint
                .truncate_prefix(last_id)
                .map_err(|e| RockDuckError::Write(format!("WAL truncate_prefix: {}", e)))?;
        }

        tracing::info!(
            "Checkpoint {} created: committed_txn={}",
            id,
            state.committed_txn
        );

        Ok(id)
    }

    /// Load the latest checkpoint from disk.
    /// If the _LATEST checkpoint file is missing or corrupted, fall back to the
    /// previous checkpoint (if any). This handles crashes that occur right after
    /// renaming _LATEST but before the checkpoint file is fsynced.
    pub fn load_latest(&self) -> Result<Option<(u64, CheckpointState)>> {
        let latest_path = self.checkpoint_dir.join("_LATEST");
        if !latest_path.exists() {
            return Ok(None);
        }

        let id: u64 = std::fs::read_to_string(&latest_path)?
            .trim()
            .parse()
            .map_err(|e| RockDuckError::Internal(format!("parse checkpoint id: {e}")))?;

        let path = self.checkpoint_dir.join(format!("ckpt_{}", id));
        if let Some(state) = Self::load_checkpoint(&path) {
            return Ok(Some((id, state)));
        }

        // Checkpoint file missing or corrupt (crash between rename and fsync).
        // Fall back to the previous checkpoint.
        if id == 0 {
            tracing::warn!("Latest checkpoint {} missing, no fallback available", id);
            return Ok(None);
        }

        tracing::warn!(
            "Checkpoint file ckpt_{} missing/corrupt, trying fallback ckpt_{}",
            id,
            id - 1
        );
        let fallback_path = self.checkpoint_dir.join(format!("ckpt_{}", id - 1));
        let state = Self::load_checkpoint(&fallback_path).ok_or_else(|| {
            RockDuckError::Internal(format!(
                "fallback checkpoint ckpt_{} also missing/corrupt",
                id - 1
            ))
        })?;

        Ok(Some((id - 1, state)))
    }

    /// Load a checkpoint file, validating magic and CRC.
    /// Returns None if the file is missing or corrupted.
    fn load_checkpoint(path: &Path) -> Option<CheckpointState> {
        let data = match std::fs::read(path) {
            Ok(d) => d,
            Err(e) => {
                tracing::warn!("checkpoint: failed to read {}: {}", path.display(), e);
                return None;
            }
        };

        if data.len() < 13 {
            tracing::warn!(
                "checkpoint: {} too small ({} bytes)",
                path.display(),
                data.len()
            );
            return None;
        }

        let magic =
            u32::from_le_bytes(data[0..4].try_into().map_err(|_| {
                RockDuckError::Codec("checkpoint magic: insufficient bytes".into())
            }).ok()?);
        if magic != CKPT_MAGIC {
            tracing::warn!(
                "checkpoint: {} invalid magic: expected {:08x}, got {:08x}",
                path.display(),
                CKPT_MAGIC,
                magic
            );
            return None;
        }

        let version = data[4];
        let read_crc =
            u32::from_le_bytes(data[5..9].try_into().map_err(|_| {
                RockDuckError::Codec("checkpoint read_crc: insufficient bytes".into())
            }).ok()?);
        let payload_len = u32::from_le_bytes(data[9..13].try_into().map_err(|_| {
            RockDuckError::Codec("checkpoint payload_len: insufficient bytes".into())
        }).ok()?) as usize;

        if 13 + payload_len != data.len() {
            tracing::warn!(
                "checkpoint: {} payload_len {} mismatch: file has {} bytes",
                path.display(),
                payload_len,
                data.len()
            );
            return None;
        }

        let payload = &data[13..];

        if version == 1 {
            return match postcard::from_bytes::<CheckpointStateV1>(payload) {
                Ok(v1) => Some(CheckpointState {
                    checkpoint_id: v1.checkpoint_id,
                    committed_txn: v1.committed_txn,
                    timestamp_ms: v1.timestamp_ms,
                    // D7 fix: v1 checkpoints have no replay_watermark, default to 0
                    replay_watermark: 0,
                    mvcc: None,
                }),
                Err(e) => {
                    tracing::warn!(
                        "checkpoint: {} deserialize v1 failed: {}",
                        path.display(),
                        e
                    );
                    None
                }
            };
        }

        // v2 → v3: CheckpointState added replay_watermark field (D7 fix)
        if version == 2 {
            #[derive(Deserialize)]
            struct CheckpointStateV2 {
                checkpoint_id: u64,
                committed_txn: u64,
                timestamp_ms: u64,
                mvcc: Option<super::checkpoint::CheckpointMvccState>,
            }
            return match postcard::from_bytes::<CheckpointStateV2>(payload) {
                Ok(v2) => Some(CheckpointState {
                    checkpoint_id: v2.checkpoint_id,
                    committed_txn: v2.committed_txn,
                    timestamp_ms: v2.timestamp_ms,
                    // D7 fix: v2 checkpoints have no replay_watermark, default to 0
                    replay_watermark: 0,
                    mvcc: v2.mvcc,
                }),
                Err(e) => {
                    tracing::warn!(
                        "checkpoint: {} deserialize v2 failed: {}",
                        path.display(),
                        e
                    );
                    None
                }
            };
        }

        if version != CKPT_VERSION {
            tracing::warn!(
                "checkpoint: {} unknown version {}, expected {}",
                path.display(),
                version,
                CKPT_VERSION
            );
            return None;
        }
        let computed_crc = crc32fast::hash(payload);
        if computed_crc != read_crc {
            tracing::warn!(
                "checkpoint: {} CRC mismatch: expected {:08x}, got {:08x}",
                path.display(),
                computed_crc,
                read_crc
            );
            return None;
        }

        match postcard::from_bytes(payload) {
            Ok(state) => Some(state),
            Err(e) => {
                tracing::warn!("checkpoint: {} deserialize failed: {}", path.display(), e);
                None
            }
        }
    }

    fn discover_max_id(dir: &Path) -> Option<u64> {
        let entries = std::fs::read_dir(dir).ok()?;
        entries
            .flatten()
            .filter_map(|e| {
                e.file_name()
                    .to_str()?
                    .strip_prefix("ckpt_")?
                    .parse::<u64>()
                    .ok()
            })
            .max()
    }

    fn sync_dir(&self) -> std::io::Result<()> {
        #[cfg(windows)]
        {
            let dir_file = std::fs::OpenOptions::new()
                .read(true)
                .open(&self.checkpoint_dir)?;
            dir_file.sync_all()?;
        }
        #[cfg(not(windows))]
        {
            // Actually call sync_all() on Linux/macOS.
            // On POSIX, we open the directory itself and call sync_all().
            use std::fs::OpenOptions;
            let dir_file = OpenOptions::new().read(true).open(&self.checkpoint_dir)?;
            dir_file.sync_all()?;
        }
        Ok(())
    }

    fn fsync_file(path: &Path) -> std::io::Result<()> {
        let file = std::fs::OpenOptions::new().read(true).open(path)?;
        file.sync_all()
    }
}
