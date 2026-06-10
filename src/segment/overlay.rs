//! Unified overlay / visibility adapter helpers.
//!
//! Phase 1/2 bridge: centralizes how non-scan paths reason about row visibility,
//! especially compaction's special all-visible semantics.

use std::collections::{BTreeSet, HashMap};

use crate::db::TxnId;
use crate::mvcc::visibility::{
    IsolationLevel, TxnSnapshot, VisFilter, VisibilityContext, VisibilityManager,
    VisibilityProjection,
};

/// Canonical snapshot id used by compaction's historical "all committed rows are visible"
/// semantics. This is intentionally different from online scan snapshots.
pub const COMPACTION_SNAPSHOT_ID: TxnId = u64::MAX;

/// Visibility adapter for compaction and other non-query paths.
#[derive(Debug, Clone)]
pub struct SegmentOverlay {
    projection: VisibilityProjection,
    snapshot_id: TxnId,
    active_txns: BTreeSet<TxnId>,
    commit_ts_map: HashMap<TxnId, u64>,
}

impl SegmentOverlay {
    /// Online snapshot-backed overlay.
    pub fn from_snapshot(snapshot: &TxnSnapshot) -> Self {
        Self::from_context(VisibilityContext::from_snapshot(
            snapshot,
            VisibilityProjection::Online,
        ))
    }

    pub fn from_context(context: VisibilityContext) -> Self {
        Self {
            projection: context.projection,
            snapshot_id: context.snapshot_id,
            active_txns: context.active_txns,
            commit_ts_map: context.commit_ts_map,
        }
    }

    /// Compaction overlay that preserves today's all-visible behavior:
    /// empty active set + max snapshot id.
    pub fn for_compaction_all_visible() -> Self {
        Self::from_context(VisibilityContext::compaction_rewrite())
    }

    pub fn projection(&self) -> VisibilityProjection {
        self.projection
    }

    /// Evaluate row visibility through the shared VisFilter seam.
    pub fn is_row_visible<F: VisFilter + ?Sized>(
        &self,
        filter: &F,
        created_txn: TxnId,
        deleted_txn: Option<TxnId>,
    ) -> bool {
        filter.is_row_visible(
            self.snapshot_id,
            created_txn,
            deleted_txn,
            &self.active_txns,
            &self.commit_ts_map,
        )
    }

    /// Materialize a synthetic snapshot for legacy callers that still want TxnSnapshot.
    pub fn as_snapshot(&self) -> TxnSnapshot {
        TxnSnapshot {
            snapshot_id: self.snapshot_id,
            active_txns: self.active_txns.clone(),
            isolation: IsolationLevel::Snapshot,
            commit_ts_map: self.commit_ts_map.clone(),
        }
    }

    pub fn as_context(&self) -> VisibilityContext {
        VisibilityContext {
            projection: self.projection,
            snapshot_id: self.snapshot_id,
            active_txns: self.active_txns.clone(),
            commit_ts_map: self.commit_ts_map.clone(),
        }
    }
}

/// The current compaction filter implementation.
///
/// We intentionally use a fresh `VisibilityManager` because compaction's semantics are not
/// "current online visibility"; they are "treat every committed row as visible under the
/// all-visible synthetic snapshot". Keeping this helper centralized makes future Phase 2
/// semantic changes explicit.
pub fn compaction_overlay_filter() -> (SegmentOverlay, VisibilityManager) {
    (
        SegmentOverlay::for_compaction_all_visible(),
        VisibilityManager::new(),
    )
}
