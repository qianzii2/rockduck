//! Metadata module for RockDuck
//!
//! # Evidence Plane Classification
//!
//! All metadata in RockDuck is classified by evidence semantics and consumer contracts.
//!
//! ## Evidence Tiers
//!
//! | Tier | Name | Description | Freshness Contract | Consumers |
//! |------|------|-------------|-------------------|-----------|
//! | **T1** | **Datasource of Truth** | State that, once durable, is never reverted by a later layer | Committed WAL + Checkpoint | All readers (scan, point_get, VTab, time-travel) |
//! | **T2** | **Inference Evidence** | Derived or heuristic evidence that informs decisions but is not itself authoritative | Updated on events (compaction, insert, access) | Routing, compaction scheduling, EcoTune |
//! | **T3** | **Cache / Shadow** | Materialized projections of T1 state, safe to recompute | Recomputed on demand or invalidated on T1 events | VTab, TimeTravelScanner, AccessTracker |
//!
//! ## Column Family Evidence Map
//!
//! | CF Bucket | Tier | Freshness Contract | Evidence Role |
//! |----------|------|-------------------|--------------|
//! | `CF_MVCC` (committed txns) | T1 | Durable, WAL-replayed | Source of truth for visibility decisions |
//! | `CF_MVCC` (active txns) | T1 | Durable, WAL-replayed | Active transaction set for SI |
//! | `CF_STAT` (table_stats) | T2 | Updated on insert/commit | Routing cost model input; `column_stats` always Default (gap) |
//! | `CF_ZONE` / `block_zone_map` | T2 | Computed on flush | Zone map pruning evidence for routing |
//! | `CF_SEG_META` | T2 | Updated on compact/flush | Compaction priority input |
//! | `CF_PK_IDX` | T1 | Durable, WAL-replayed | PK lookup authority |
//! | `CF_VERSIONS` | T3 | Lazy build / recompute | Time-travel version index (rebuilt from T1) |
//! | `CF_BF` | T3 | Recomputed on demand | Bloom filter cache (safe to rebuild) |
//! | `CF_DELTA` | T3 | Shadow of T1 | Delta patches (safe to recompute from WAL) |
//!
//! ## Known Evidence Gaps
//!
//! - `TableStats::column_stats` is always `Default::default()` — zone map evidence exists
//!   in `block_zone_map.rs` but is NOT fed back into routing cost model.
//!   This is a **partial unsanctioned gap**: evidence exists but is not closed-loop.

use crate::query::routing::feedback::ExecutionEvidenceSnapshot;

/// Evidence shared across routing and maintenance.
#[derive(Debug, Clone)]
pub struct EvidenceSnapshot {
    pub table: String,
    pub query_columns: Vec<String>,
    pub table_stats: crate::query::routing::TableStats,
    pub routed_segment_ids: Vec<String>,
    pub executed_segment_ids: Vec<String>,
    pub total_segment_rows: u64,
    pub delta_segment_count: usize,
    pub has_zone_map_predicates: bool,
    pub has_cross_column_or: bool,
    pub projection_contract: Option<ProjectionContract>,
}

impl EvidenceSnapshot {
    pub fn governance_is_complete(&self) -> bool {
        match self.projection_contract.as_ref() {
            Some(contract) => {
                contract.enforcement.blocks_regressions()
                    && !contract.evidence_hook.is_empty()
                    && matches!(
                        contract.sidecar_class,
                        SidecarClass::CoreRead | SidecarClass::SanctionedSidecar
                    )
            }
            None => false,
        }
    }

    pub fn assert_governance_ready(&self) {
        if let Some(contract) = self.projection_contract.as_ref() {
            contract.assert_blocking_governance()
        }
        assert!(
            self.governance_is_complete(),
            "projection contract must carry classified sidecar evidence before enforcement"
        );
    }

    pub fn into_execution_evidence(self) -> ExecutionEvidenceSnapshot {
        self.assert_governance_ready();
        ExecutionEvidenceSnapshot {
            table: self.table,
            query_columns: self.query_columns,
            total_segment_rows: self.total_segment_rows,
            routed_segment_ids: self.routed_segment_ids,
            executed_segment_ids: self.executed_segment_ids,
            delta_segment_count: self.delta_segment_count,
            has_zone_map_predicates: self.has_zone_map_predicates,
            has_cross_column_or: self.has_cross_column_or,
            table_row_count: self.table_stats.row_count,
            total_bytes: self.table_stats.total_bytes,
            compressed_bytes: self.table_stats.compressed_bytes,
            projection_contract: self.projection_contract,
        }
    }
}

pub mod block_stats;
pub mod block_zone_map;
pub mod granule_id;
pub mod kv_engine;
pub mod kv_store;
pub mod layer;
pub mod mace_adapter;
pub mod pk_skiplist;
pub mod projection;
pub mod seg_alias;
pub mod seg_meta;
pub mod zone_map;

// Specific re-exports to avoid ambiguous glob conflicts
pub use block_stats::*;
pub use block_zone_map::*;
pub use granule_id::GranuleId;
pub use kv_engine::{
    KVEngine, KVIter, KVOp, CF_BF, CF_ICEBERG, CF_LAYER, CF_LBF, CF_MVCC, CF_PK_IDX, CF_SEG_META,
    CF_STAT, CF_SYS, CF_VERSIONS, CF_ZONE,
};
pub use kv_store::{
    add_active_txn, commit_txn_record, get_active_txns, get_committed_txn, get_committed_txns,
    get_or_create_table_stats, get_segment_meta, increment_row_count, list_segment_metas,
    put_committed_txn, put_segment_meta, put_segment_meta_and_invalidate, put_table_stats,
    remove_active_txn, update_segment_status, TableStats,
};
pub use layer::*;
pub use pk_skiplist::{
    delete_pk_index_double, delete_pk_index_double_into_batch, get_pk_index_by_pk,
    list_skiplist_entries, parse_pk_index_key, pk_index_key, pk_lookup_key, pk_skiplist_key,
    put_pk_index_double, resolve_pk_entry_seg_id, SkiplistEntry,
};
pub use projection::*;
pub use seg_meta::*;
pub use zone_map::*;

// Governance phase2 matrix tests are in the compaction module
