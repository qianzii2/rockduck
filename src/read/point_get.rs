//! Point-get operations with Bloom filter acceleration
//!
//! Provides:
//! - Bloom filter insertion for primary key deduplication checks
//! - Primary key lookup (O(1) via PK index)
//! - Point-get: reads a single row by primary key from columnar storage

use std::collections::{BTreeSet, HashMap, HashSet};

use crate::db::RockDuck;
use crate::error::Result;
use crate::metadata;
use crate::metadata::pk_skiplist;
use crate::metadata::projection::ProjectionContract;
use crate::mvcc::shadow_columns as sc;
use crate::segment::layout::SegmentLayout;
use crate::storage::delta::DeltaQueryLayer;
use crate::storage::vortex::VortexReader;
use arrow_array::cast::AsArray;

use fastbloom::BloomFilter;

/// Insert a primary key into a Bloom filter for duplicate detection
pub fn bloom_insert(bf: &mut BloomFilter, pk: &[u8]) {
    bf.insert(&pk.to_vec());
}

/// Check if a primary key might exist in a Bloom filter
pub fn bloom_might_contain(bf: &BloomFilter, pk: &[u8]) -> bool {
    bf.contains(&pk.to_vec())
}

/// Get a single row by primary key.
///
/// Returns the column values as a HashMap of column name → RowValue.
///
/// Returns None if:
/// - The PK does not exist in the index
/// - The row has been deleted (Shadow Column marks it deleted)
pub fn get(
    db: &RockDuck,
    table: &str,
    pk: &[u8],
) -> Result<Option<HashMap<String, super::row_value::RowValue>>> {
    let projection_contract = ProjectionContract::point_get();
    projection_contract.assert_blocking_governance();
    tracing::debug!(
        surface = ?projection_contract.surface,
        visibility = ?projection_contract.visibility,
        sidecar_class = ?projection_contract.sidecar_class,
        evidence_hook = projection_contract.evidence_hook,
        table,
        "point_get projection contract"
    );
    // 1. PK index lookup — O(1) via skiplist
    let entry = match pk_skiplist::get_pk_index_by_pk(&db.kv, table, pk)? {
        Some(e) => e,
        None => return Ok(None),
    };
    let (seg_id, _granule_id, row_offset) = entry;

    let layout = SegmentLayout::new(&db.data_dir, &seg_id);

    // 2. Check deltavis first: if row is in deltavis, it's marked deleted regardless of vis
    let vis_path = layout.vis_path();
    {
        let deltavis_path = layout.deltavis_path();
        if deltavis_path.exists() {
            let vis_writer = crate::write::vis_file::VisFileWriter::new(&vis_path);
            if let Ok(entries) = vis_writer.read_deltavis() {
                if entries.iter().any(|(row, _)| *row == row_offset as u64) {
                    return Ok(None);
                }
            }
        }
    }

    // 3. Check __vis.vortex visibility — if not visible, row is invisible
    if vis_path.exists() {
        if let Ok(vis_reader) = VortexReader::open(&vis_path) {
            if let Some((vis_batch, local_idx)) = vis_reader.read_batch_at(row_offset as u64) {
                if sc::has_visibility_columns(&vis_batch.schema()) {
                    let (c_arr, d_arr) = sc::extract_visibility_columns(&vis_batch);
                    let c = c_arr
                        .as_primitive::<arrow_array::types::Int64Type>()
                        .value(local_idx) as u64;
                    let d_raw = d_arr
                        .as_primitive::<arrow_array::types::Int64Type>()
                        .value(local_idx) as u64;
                    let d = if d_raw == sc::NOT_DELETED {
                        None
                    } else {
                        Some(d_raw)
                    };
                    let snapshot = db.snapshot();
                    if !db.mvcc.read().is_visible(&snapshot, c, d) {
                        return Ok(None);
                    }
                }
            }
        }
    }

    // 3. Read the row at row_offset from all column files
    let seg_meta = match metadata::get_segment_meta(&db.kv, &seg_id)? {
        Some(m) => m,
        None => return Ok(None),
    };

    if seg_meta.columns.is_empty() {
        return Ok(None);
    }

    let mut row_data: HashMap<String, super::row_value::RowValue> = HashMap::new();

    for col_def in &seg_meta.columns {
        let col_path = layout.col_path(&col_def.name);
        if !col_path.exists() {
            continue;
        }

        let reader = match VortexReader::open(&col_path) {
            Ok(r) => r,
            Err(e) => {
                tracing::error!(
                    "point_get: failed to open column vortex {}: {}",
                    col_path.display(),
                    e
                );
                continue;
            }
        };

        if let Some((batch, local_idx)) = reader.read_batch_at(row_offset as u64) {
            let col = batch.column(0);
            let val = extract_row_bytes(col, local_idx)?;
            row_data.insert(col_def.name.clone(), val);
        }
    }

    // Step 3.5: Apply delta layer — apply all delta patches affecting this row
    let snapshot = db.snapshot();
    let active_txns: HashSet<u64> = snapshot.active_txns.iter().copied().collect();
    let deltas = db
        .delta_layer
        .query(
            &seg_id,
            snapshot.snapshot_id,
            &snapshot.commit_ts_map,
            &active_txns,
        )
        .map_err(|e| {
            crate::RockDuckError::Query(format!(
                "Delta layer query failed for seg {}: {}",
                seg_id, e
            ))
        })?;
    for delta in deltas {
        if delta.row_offset != row_offset as u64 {
            continue;
        }
        if let Some(ref after) = delta.after {
            if !delta.column.is_empty() {
                let rv = if after.is_empty() {
                    super::row_value::RowValue::Empty
                } else {
                    super::row_value::RowValue::Value(after.as_ref().clone())
                };
                row_data.insert(delta.column.clone(), rv);
            }
        }
    }

    Ok(Some(row_data))
}

#[cfg(test)]
#[allow(clippy::items_after_test_module)]
mod governance_contract_tests {
    use super::*;
    use crate::metadata::projection::ProjectionSurface;

    #[test]
    fn point_get_projection_contract_is_blocking() {
        let contract = ProjectionContract::point_get();
        contract.assert_blocking_governance();
        assert_eq!(contract.surface, ProjectionSurface::PointGet);
    }

    #[test]
    fn historical_point_get_projection_contract_routes_sidecar_evidence() {
        let contract = ProjectionContract::historical_point_get();
        contract.assert_blocking_governance();

        let metadata = crate::metadata::EvidenceSnapshot {
            table: "orders".to_string(),
            query_columns: Vec::new(),
            table_stats: crate::query::routing::TableStats::default(),
            routed_segment_ids: Vec::new(),
            executed_segment_ids: Vec::new(),
            total_segment_rows: 0,
            delta_segment_count: 0,
            has_zone_map_predicates: false,
            has_cross_column_or: false,
            projection_contract: Some(contract),
        };
        metadata.assert_governance_ready();
        assert_eq!(
            metadata.projection_contract.as_ref().unwrap().surface,
            ProjectionSurface::HistoricalPointGet
        );
    }

    #[test]
    fn historical_point_get_projection_contract_is_blocking() {
        let contract = ProjectionContract::historical_point_get();
        contract.assert_blocking_governance();
        assert_eq!(contract.surface, ProjectionSurface::HistoricalPointGet);
    }
}

/// Get a single row by primary key as of a given transaction.
///
/// Returns the column values as a HashMap of column name → RowValue.
///
/// This is the time travel point-get: it reads the historical state of the row
/// at `target_txn` using the version index.
pub fn get_as_of(
    db: &RockDuck,
    table: &str,
    pk: &[u8],
    target_txn: u64,
) -> Result<Option<HashMap<String, super::row_value::RowValue>>> {
    use crate::query::routing::SidecarEvidenceSnapshot;
    use crate::query::TimeTravelReader;

    let projection_contract = ProjectionContract::historical_point_get();
    projection_contract.assert_blocking_governance();
    tracing::debug!(
        surface = ?projection_contract.surface,
        visibility = ?projection_contract.visibility,
        sidecar_class = ?projection_contract.sidecar_class,
        evidence_hook = projection_contract.evidence_hook,
        table,
        target_txn,
        "historical point_get projection contract"
    );
    if let Some(router) = db.router.as_ref() {
        router.observe_sidecar_evidence(
            db,
            &SidecarEvidenceSnapshot {
                table: table.to_string(),
                routed_segment_ids: Vec::new(),
                executed_segment_ids: Vec::new(),
                contract: projection_contract.clone(),
            },
        );
    }

    let reader = TimeTravelReader::new(std::sync::Arc::clone(&db.kv), db.data_dir.clone())?;

    // Try version index first
    if let Some(value) = reader.as_of_read(table, pk, target_txn)? {
        let mut result = HashMap::new();
        result.insert(
            "value".to_string(),
            super::row_value::RowValue::Value(value),
        );
        return Ok(Some(result));
    }

    // Fall back: look up PK index, then scan __vis.vortex for visibility
    let entry = match pk_skiplist::get_pk_index_by_pk(&db.kv, table, pk)? {
        Some(e) => e,
        None => return Ok(None),
    };
    let (seg_id, _granule_id, row_offset) = entry;

    let seg_meta = match metadata::get_segment_meta(&db.kv, &seg_id)? {
        Some(m) => m,
        None => return Ok(None),
    };

    if seg_meta.columns.is_empty() {
        return Ok(None);
    }

    let layout = SegmentLayout::new(&db.data_dir, &seg_id);
    let vis_path = layout.vis_path();

    // Check deltavis: if row is in deltavis, it's marked deleted (time travel always reads current state for deletions)
    {
        let deltavis_path = layout.deltavis_path();
        if deltavis_path.exists() {
            let vis_writer = crate::write::vis_file::VisFileWriter::new(&vis_path);
            if let Ok(entries) = vis_writer.read_deltavis() {
                if entries.iter().any(|(row, _)| *row == row_offset as u64) {
                    return Ok(None);
                }
            }
        }
    }

    // Check visibility: the row exists in __vis.vortex if its row_offset is in range.
    let mut row_found = false;
    let committed_txns_map = crate::metadata::get_committed_txns(&db.kv)?;
    let historical_commit_ts_map: HashMap<u64, u64> = committed_txns_map
        .iter()
        .filter_map(|(&txn_id, &commit_ts)| {
            (commit_ts <= target_txn).then_some((txn_id, commit_ts))
        })
        .collect();
    let vis_mgr = db.mvcc.read();
    {
        if vis_path.exists() {
            if let Ok(vis_reader) = VortexReader::open(&vis_path) {
                if let Some((vis_batch, local_idx)) = vis_reader.read_batch_at(row_offset as u64) {
                    row_found = true;
                    if sc::has_visibility_columns(&vis_batch.schema()) {
                        let (c_arr, d_arr) = sc::extract_visibility_columns(&vis_batch);
                        let c = c_arr
                            .as_primitive::<arrow_array::types::Int64Type>()
                            .value(local_idx) as u64;
                        let d_raw = d_arr
                            .as_primitive::<arrow_array::types::Int64Type>()
                            .value(local_idx) as u64;
                        let d = if d_raw == sc::NOT_DELETED {
                            None
                        } else {
                            Some(d_raw)
                        };
                        let historical_vis = crate::query::time_travel_impl::HistoricalVisibility {
                            created_txn: c,
                            deleted_txn: d,
                        };
                        // Historical visibility now runs as an explicit projection context over
                        // real commit timestamps, and delta patch filtering below now uses the
                        // same KV-backed authority map instead of in-memory retained history.
                        let historical_audit = crate::query::time_travel_impl::HistoricalVisibilityAudit::truth_package();
                        tracing::debug!(
                            target = historical_audit.main_path,
                            bypasses = ?historical_audit.bypass_paths,
                            landing = ?historical_audit.landing_files,
                            projection = "Historical",
                            constraint = "committed_txns filtered to commit_ts <= target_txn at construction",
                            "truth triple-check verified - historical projection surface"
                        );
                        let committed_txns: BTreeSet<u64> = committed_txns_map
                            .iter()
                            .filter_map(|(&txn_id, &commit_ts)| {
                                (commit_ts <= target_txn).then_some(txn_id)
                            })
                            .collect();
                        let historical_context = crate::query::time_travel_impl::HistoricalVisibility::projection_context(
                            target_txn,
                            &committed_txns,
                            &committed_txns_map,
                        );
                        let visible = historical_vis.is_visible_at(&vis_mgr, &historical_context);
                        if !visible {
                            return Ok(None);
                        }
                    }
                }
            }
        }
    }

    // Read the row data columns.
    let mut row_data: HashMap<String, super::row_value::RowValue> = HashMap::new();
    for col_def in &seg_meta.columns {
        let col_path = layout.col_path(&col_def.name);
        if !col_path.exists() {
            continue;
        }
        let reader = match VortexReader::open(&col_path) {
            Ok(r) => r,
            Err(e) => {
                tracing::error!(
                    "point_get: failed to open column vortex {}: {}",
                    col_path.display(),
                    e
                );
                continue;
            }
        };
        if let Some((batch, local_idx)) = reader.read_batch_at(row_offset as u64) {
            let col = batch.column(0);
            let val = extract_row_bytes(col, local_idx)?;
            row_data.insert(col_def.name.clone(), val);
        }
    }

    // Fix: row_data.is_empty() incorrectly treated ALL-NULL rows as non-existent.
    // A row is "found" if its row_offset falls within __vis.vortex row range.
    // All-NULL rows are still valid rows and should be returned.
    if row_found {
        // Apply delta layer patches that were committed at or before target_txn.
        // Delta cells carry txn_id (not commit_ts), so we filter against the same
        // KV-backed historical commit timestamp map used to build the projection.
        let historical_active_txns = HashSet::new();
        let deltas = db.delta_layer.query(
            &seg_id,
            target_txn,
            &historical_commit_ts_map,
            &historical_active_txns,
        )?;
        for delta in deltas {
            if delta.row_offset != row_offset as u64 {
                continue;
            }
            if let Some(ref after) = delta.after {
                if !delta.column.is_empty() {
                    let rv = if after.is_empty() {
                        super::row_value::RowValue::Empty
                    } else {
                        super::row_value::RowValue::Value(after.as_ref().clone())
                    };
                    row_data.insert(delta.column.clone(), rv);
                }
            }
        }

        Ok(Some(row_data))
    } else {
        Ok(None)
    }
}

/// Extract a single row value from an Arrow column at the given row index.
///
/// Returns `RowValue` to explicitly distinguish NULL (no value stored) from
/// empty sequences (e.g. `""` string, zero-length binary). The prior `Vec<u8>`
/// return type used an empty vec to mean both cases, making them indistinguishable.
pub fn extract_row_bytes(
    col: &arrow_array::ArrayRef,
    row_idx: usize,
) -> Result<super::row_value::RowValue> {
    use arrow_array::*;
    if let Some(arr) = col.as_any().downcast_ref::<Int8Array>() {
        if arr.is_null(row_idx) {
            return Ok(super::row_value::RowValue::Null);
        }
        return Ok(super::row_value::RowValue::Value(
            arr.value(row_idx).to_le_bytes().to_vec(),
        ));
    }
    if let Some(arr) = col.as_any().downcast_ref::<Int16Array>() {
        if arr.is_null(row_idx) {
            return Ok(super::row_value::RowValue::Null);
        }
        return Ok(super::row_value::RowValue::Value(
            arr.value(row_idx).to_le_bytes().to_vec(),
        ));
    }
    if let Some(arr) = col.as_any().downcast_ref::<Int32Array>() {
        if arr.is_null(row_idx) {
            return Ok(super::row_value::RowValue::Null);
        }
        return Ok(super::row_value::RowValue::Value(
            arr.value(row_idx).to_le_bytes().to_vec(),
        ));
    }
    if let Some(arr) = col.as_any().downcast_ref::<Int64Array>() {
        if arr.is_null(row_idx) {
            return Ok(super::row_value::RowValue::Null);
        }
        return Ok(super::row_value::RowValue::Value(
            arr.value(row_idx).to_le_bytes().to_vec(),
        ));
    }
    if let Some(arr) = col.as_any().downcast_ref::<UInt8Array>() {
        if arr.is_null(row_idx) {
            return Ok(super::row_value::RowValue::Null);
        }
        return Ok(super::row_value::RowValue::Value(
            arr.value(row_idx).to_le_bytes().to_vec(),
        ));
    }
    if let Some(arr) = col.as_any().downcast_ref::<UInt16Array>() {
        if arr.is_null(row_idx) {
            return Ok(super::row_value::RowValue::Null);
        }
        return Ok(super::row_value::RowValue::Value(
            arr.value(row_idx).to_le_bytes().to_vec(),
        ));
    }
    if let Some(arr) = col.as_any().downcast_ref::<UInt32Array>() {
        if arr.is_null(row_idx) {
            return Ok(super::row_value::RowValue::Null);
        }
        return Ok(super::row_value::RowValue::Value(
            arr.value(row_idx).to_le_bytes().to_vec(),
        ));
    }
    if let Some(arr) = col.as_any().downcast_ref::<UInt64Array>() {
        if arr.is_null(row_idx) {
            return Ok(super::row_value::RowValue::Null);
        }
        return Ok(super::row_value::RowValue::Value(
            arr.value(row_idx).to_le_bytes().to_vec(),
        ));
    }
    if let Some(arr) = col.as_any().downcast_ref::<Float32Array>() {
        if arr.is_null(row_idx) {
            return Ok(super::row_value::RowValue::Null);
        }
        return Ok(super::row_value::RowValue::Value(
            arr.value(row_idx).to_le_bytes().to_vec(),
        ));
    }
    if let Some(arr) = col.as_any().downcast_ref::<Float64Array>() {
        if arr.is_null(row_idx) {
            return Ok(super::row_value::RowValue::Null);
        }
        return Ok(super::row_value::RowValue::Value(
            arr.value(row_idx).to_le_bytes().to_vec(),
        ));
    }
    if let Some(arr) = col.as_any().downcast_ref::<StringArray>() {
        if arr.is_null(row_idx) {
            return Ok(super::row_value::RowValue::Null);
        }
        let s = arr.value(row_idx);
        if s.is_empty() {
            return Ok(super::row_value::RowValue::Empty);
        }
        return Ok(super::row_value::RowValue::Value(s.as_bytes().to_vec()));
    }
    if let Some(arr) = col.as_any().downcast_ref::<LargeStringArray>() {
        if arr.is_null(row_idx) {
            return Ok(super::row_value::RowValue::Null);
        }
        let s = arr.value(row_idx);
        if s.is_empty() {
            return Ok(super::row_value::RowValue::Empty);
        }
        return Ok(super::row_value::RowValue::Value(s.as_bytes().to_vec()));
    }
    if let Some(arr) = col.as_any().downcast_ref::<BinaryArray>() {
        if arr.is_null(row_idx) {
            return Ok(super::row_value::RowValue::Null);
        }
        let v = arr.value(row_idx);
        if v.is_empty() {
            return Ok(super::row_value::RowValue::Empty);
        }
        return Ok(super::row_value::RowValue::Value(v.to_vec()));
    }
    if let Some(arr) = col.as_any().downcast_ref::<LargeBinaryArray>() {
        if arr.is_null(row_idx) {
            return Ok(super::row_value::RowValue::Null);
        }
        let v = arr.value(row_idx);
        if v.is_empty() {
            return Ok(super::row_value::RowValue::Empty);
        }
        return Ok(super::row_value::RowValue::Value(v.to_vec()));
    }
    if let Some(arr) = col.as_any().downcast_ref::<BooleanArray>() {
        if arr.is_null(row_idx) {
            return Ok(super::row_value::RowValue::Null);
        }
        return Ok(super::row_value::RowValue::Value(if arr.value(row_idx) {
            1u8.to_le_bytes().to_vec()
        } else {
            0u8.to_le_bytes().to_vec()
        }));
    }
    // Unknown / unsupported type — conservatively treat as NULL
    Ok(super::row_value::RowValue::Null)
}
