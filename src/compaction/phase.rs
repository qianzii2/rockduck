//! Crash-Safe Compaction — Three-Phase State Machine
//!
//! Ensures no data loss after compaction crash:
//!
//! Phase::Idle
//!   ↓ [scheduler selects target segs]
//! Phase::Prepare { new_seg_id, old_seg_ids }
//!   - Lock old segs (write Compacting status to RocksDB)
//!   - Start generating new seg data
//   ↓ [new seg data generation complete]
// Phase::Commit { new_seg_id, old_seg_ids }
//!   - Atomic manifest update (all seg status switches)
//   ↓ [manifest update successful]
// Phase::Cleanup { new_seg_id, old_seg_ids }
//!   - Delete old seg files
//   ↓ [cleanup complete]
// Phase::Idle
//!
//! Recovery (db.rs init):
//!   - Scan all segments in Compacting state
//!   - Read phase status
//!   - Continue or rollback by status
//!
//! # Thread Safety
//!
//! This state machine is **NOT thread-safe**. `CompactionPhaseMachine` must be
//! owned by a single compaction thread. Concurrent calls from multiple threads without
//! external synchronization will cause data races on the internal `state` field.
//!
//! The WAL protection for compaction (P0-A fix) addresses the crash-safety gap;
//! this phase machine documents the intended state machine but relies on the WAL
//! record as the authoritative crash-recovery mechanism.

use parking_lot::Mutex;
use crate::db::TxnId;
use crate::error::Result;

/// Compaction phase
#[derive(
    Debug,
    Clone,
    PartialEq,
    Eq,
    serde::Serialize,
    serde::Deserialize,
    rkyv::Archive,
    rkyv::Serialize,
    rkyv::Deserialize,
)]
pub enum CompactionPhase {
    /// Idle, no compaction in progress
    Idle,
    /// Prepare: generating new segment, old segs are locked
    Prepare {
        new_seg_id: String,
        old_seg_ids: Vec<String>,
    },
    /// Commit: new seg generated, waiting for manifest atomic switch
    Commit {
        new_seg_id: String,
        old_seg_ids: Vec<String>,
    },
    /// Cleanup: manifest updated, cleaning old seg files
    Cleanup {
        new_seg_id: String,
        old_seg_ids: Vec<String>,
    },
}

impl CompactionPhase {
    pub fn is_idle(&self) -> bool {
        matches!(self, Self::Idle)
    }

    pub fn phase_name(&self) -> &'static str {
        match self {
            Self::Idle => "Idle",
            Self::Prepare { .. } => "Prepare",
            Self::Commit { .. } => "Commit",
            Self::Cleanup { .. } => "Cleanup",
        }
    }

    /// Serialize (store to RocksDB)
    pub fn serialize(&self) -> Result<Vec<u8>> {
        crate::codec::encode(self).map_err(|e| crate::RockDuckError::Codec(e.to_string()))
    }

    /// Deserialize
    pub fn deserialize(data: &[u8]) -> Option<Self> {
        crate::codec::decode(data).ok()
    }
}

/// Compaction state record (stored in RocksDB)
#[derive(
    Debug,
    Clone,
    serde::Serialize,
    serde::Deserialize,
    rkyv::Archive,
    rkyv::Serialize,
    rkyv::Deserialize,
)]
pub struct CompactionState {
    /// Transaction ID (for WAL correlation)
    pub txn_id: TxnId,
    /// Current phase
    pub phase: CompactionPhase,
    /// Start time
    pub start_time: u64,
}

impl CompactionState {
    pub fn new(txn_id: TxnId, phase: CompactionPhase) -> Self {
        Self {
            txn_id,
            phase,
            start_time: crate::codec::current_timestamp_millis(),
        }
    }

    pub fn serialize(&self) -> Result<Vec<u8>> {
        crate::codec::encode(self).map_err(|e| crate::RockDuckError::Codec(e.to_string()))
    }

    pub fn deserialize(data: &[u8]) -> Option<Self> {
        crate::codec::decode(data).ok()
    }
}

/// CompactionPhase state machine.
///
/// Thread-safe via internal Mutex. All state transitions are serialized.
pub struct CompactionPhaseMachine {
    state: Mutex<Option<CompactionState>>,
}

impl CompactionPhaseMachine {
    pub fn new() -> Self {
        Self {
            state: Mutex::new(None),
        }
    }

    pub fn is_idle(&self) -> bool {
        self.state.lock().as_ref().is_none_or(|s| s.phase.is_idle())
    }

    pub fn current_phase(&self) -> Option<CompactionPhase> {
        self.state.lock().as_ref().map(|s| s.phase.clone())
    }

    pub fn current_state(&self) -> Option<CompactionState> {
        self.state.lock().as_ref().cloned()
    }

    /// Start compaction: enter Prepare phase.
    pub fn begin_compaction(
        &self,
        txn_id: TxnId,
        new_seg_id: String,
        old_seg_ids: Vec<String>,
    ) -> Result<CompactionPhase> {
        let mut state = self.state.lock();
        if state.as_ref().is_some_and(|s| !s.phase.is_idle()) {
            return Err(crate::RockDuckError::Compaction(
                "Compaction already in progress".to_string(),
            ));
        }

        *state = Some(CompactionState::new(
            txn_id,
            CompactionPhase::Prepare {
                new_seg_id,
                old_seg_ids,
            },
        ));

        Ok(state.as_ref().unwrap().phase.clone())
    }

    /// Enter Commit phase from Prepare phase.
    pub fn commit_phase(&self) -> Result<CompactionPhase> {
        let mut state = self.state.lock();
        let inner = state.as_mut().ok_or_else(|| {
            crate::RockDuckError::Compaction("No compaction in progress".to_string())
        })?;

        match &inner.phase {
            CompactionPhase::Prepare {
                new_seg_id,
                old_seg_ids,
            } => {
                inner.phase = CompactionPhase::Commit {
                    new_seg_id: new_seg_id.clone(),
                    old_seg_ids: old_seg_ids.clone(),
                };
                Ok(inner.phase.clone())
            }
            _ => Err(crate::RockDuckError::Compaction(format!(
                "Cannot commit from phase {:?}",
                inner.phase
            ))),
        }
    }

    /// Enter Cleanup phase from Commit phase.
    pub fn cleanup_phase_with_callback<F>(&self, on_cleanup: F) -> Result<CompactionPhase>
    where
        F: Fn(&str),
    {
        let mut state = self.state.lock();
        let inner = state.as_mut().ok_or_else(|| {
            crate::RockDuckError::Compaction("No compaction in progress".to_string())
        })?;

        match &inner.phase {
            CompactionPhase::Commit {
                new_seg_id,
                old_seg_ids,
            } => {
                for old_seg_id in old_seg_ids {
                    on_cleanup(old_seg_id);
                }
                inner.phase = CompactionPhase::Cleanup {
                    new_seg_id: new_seg_id.clone(),
                    old_seg_ids: old_seg_ids.clone(),
                };
                Ok(inner.phase.clone())
            }
            _ => Err(crate::RockDuckError::Compaction(format!(
                "Cannot cleanup from phase {:?}",
                inner.phase
            ))),
        }
    }

    /// cleanup_phase without callback (backward-compatible).
    pub fn cleanup_phase(&self) -> Result<CompactionPhase> {
        self.cleanup_phase_with_callback(|_| {})
    }

    /// Finish compaction: return to Idle.
    pub fn finish(&self) -> Result<()> {
        *self.state.lock() = None;
        Ok(())
    }

    /// Restore compaction state from RocksDB.
    pub fn restore_from_db(&self, data: &[u8]) {
        let mut state = self.state.lock();
        if let Some(s) = CompactionState::deserialize(data) {
            if !s.phase.is_idle() {
                *state = Some(s);
            }
        }
    }
}

impl Default for CompactionPhaseMachine {
    fn default() -> Self {
        Self::new()
    }
}
