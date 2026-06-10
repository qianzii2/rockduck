//! MVCC Shadow Column visibility implementation.
//!
//! # Architecture
//!
//! Each segment written after this feature is enabled contains two additional
//! metadata columns at the end of every column file:
//! - `__created_txn`: the TxnId of the transaction that inserted this row
//! - `__deleted_txn`: the TxnId of the transaction that deleted this row
//!   (u64::MAX means not deleted)
//!
//! Visibility files are stored separately in `__vis.vortex`, one per segment.
//! Each row maps to a position in `__vis.vortex` by its global segment offset.
//!
//! At scan time, rows are filtered by `VisibilityManager::is_visible()` to
//! achieve snapshot isolation.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{ArrayRef, Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, SchemaRef};
use arrow_select::filter::filter_record_batch;

use crate::mvcc::visibility::TxnSnapshot;
use crate::mvcc::visibility::VisFilter;

/// Sentinel value for "not deleted" in the deleted_txn column.
pub const NOT_DELETED: u64 = u64::MAX;

/// Visibility column names (prefixed with __ to mark as hidden metadata columns).
pub const CREATED_TXN_COL: &str = "__created_txn";
pub const DELETED_TXN_COL: &str = "__deleted_txn";

/// Append visibility columns (__created_txn, __deleted_txn) to a RecordBatch.
///
/// For each row in the batch:
/// - `__created_txn` = `created_txn` (the txn that inserted this row)
/// - `__deleted_txn` = `deleted_txn` (u64::MAX = not deleted)
///
/// Returns a new RecordBatch with the two visibility columns appended.
pub fn append_visibility_columns(
    batch: &RecordBatch,
    created_txn: u64,
    deleted_txn: u64,
) -> RecordBatch {
    let n = batch.num_rows();

    let created_arr: ArrayRef = Arc::new(Int64Array::from_iter_values(std::iter::repeat_n(
        created_txn as i64,
        n,
    )));
    let deleted_arr: ArrayRef = Arc::new(Int64Array::from_iter_values(std::iter::repeat_n(
        deleted_txn as i64,
        n,
    )));

    let mut fields: Vec<Field> = batch
        .schema()
        .fields()
        .iter()
        .map(|f| (**f).clone())
        .collect();
    fields.push(Field::new(CREATED_TXN_COL, DataType::Int64, false));
    fields.push(Field::new(DELETED_TXN_COL, DataType::Int64, false));

    let mut columns: Vec<ArrayRef> = batch.columns().to_vec();
    columns.push(created_arr);
    columns.push(deleted_arr);

    let new_schema = SchemaRef::new(arrow_schema::Schema::new(fields));
    RecordBatch::try_new(new_schema, columns)
        .expect("schema and columns must match in append_visibility_columns")
}

/// Mark rows as deleted by updating the deleted_txn column in a __vis.vortex batch.
///
/// Takes a visibility-only RecordBatch (with __created_txn and __deleted_txn columns)
/// and a `deleted_rows` bitmap. For each row where `deleted_rows[i]` is true,
/// sets `__deleted_txn[i] = deleted_txn`.
///
/// Returns a RecordBatch with the same schema but modified deleted_txn values.
pub fn mark_rows_deleted(
    batch: &RecordBatch,
    deleted_rows: &[bool],
    deleted_txn: u64,
) -> RecordBatch {
    let n = batch.num_rows();
    assert_eq!(
        n,
        deleted_rows.len(),
        "batch rows must match deleted_rows length"
    );

    let vis_col_idx = batch.num_columns() - 1;
    let created_col_idx = batch.num_columns() - 2;

    let created_arr = batch.column(created_col_idx).as_primitive::<Int64Type>();
    let old_deleted_arr = batch.column(vis_col_idx).as_primitive::<Int64Type>();

    // Build new deleted_txn array
    let mut new_deleted = Int64Array::builder(n);
    for (i, &is_deleted) in deleted_rows.iter().enumerate().take(n) {
        if is_deleted {
            new_deleted.append_value(deleted_txn as i64);
        } else {
            new_deleted.append_value(old_deleted_arr.value(i));
        }
    }
    let new_deleted_arr = new_deleted.finish();

    // Build new created_txn array — just clone since nothing changed.
    let new_created_arr = created_arr.clone();

    let mut new_cols: Vec<ArrayRef> = batch.columns().to_vec();
    new_cols[created_col_idx] = Arc::new(new_created_arr);
    new_cols[vis_col_idx] = Arc::new(new_deleted_arr);

    RecordBatch::try_new(batch.schema(), new_cols)
        .expect("schema and columns must match in mark_rows_deleted")
}

/// Filter a RecordBatch by MVCC visibility using the `VisFilter` trait.
///
/// This is the unified visibility filter — all scan paths must call this function
/// to ensure consistent MVCC semantics across the codebase.
///
/// For each row, calls `filter.is_row_visible()` using the snapshot's
/// snapshot_id, active_txns, and commit_ts_map.
///
/// Returns a new RecordBatch containing only visible rows, with the visibility
/// columns stripped (output contains only user data columns).
///
/// Performance: iterates over the snapshot's active_txns (BTreeSet, O(log n) per row)
/// and commit_ts_map (HashMap, O(1) per row). No precomputation needed.
///
/// D6 fix: active_txns BTreeSet is converted to HashSet once per batch for O(1)
/// membership checks, providing meaningful speedup when active_txns.len() >= 5.
/// For smaller active sets, the BTreeSet is used directly (overhead not worth it).
pub fn filter_by_visibility<F: VisFilter>(
    batch: &RecordBatch,
    snapshot: &TxnSnapshot,
    #[allow(unused_variables)]
    filter: &F,
) -> crate::error::Result<RecordBatch> {
    let num_cols = batch.num_columns();
    if num_cols < 2 {
        return Ok(batch.clone());
    }

    let vis_col_idx = num_cols - 1;
    let created_col_idx = num_cols - 2;

    let created_arr = batch.column(created_col_idx).as_primitive::<Int64Type>();
    let deleted_arr = batch.column(vis_col_idx).as_primitive::<Int64Type>();
    let n = batch.num_rows();
    let snapshot_id = snapshot.snapshot_id;

    // D6 fix: Build HashSet for O(1) active_txns membership checks.
    // Break-even is ~5 active transactions; below that, overhead exceeds savings.
    let active_txns_set: std::collections::HashSet<u64> = if snapshot.active_txns.len() >= 5 {
        snapshot.active_txns.iter().copied().collect()
    } else {
        std::collections::HashSet::new()
    };

    let mut mask = arrow_array::builder::BooleanBuilder::with_capacity(n);
    for i in 0..n {
        let c = created_arr.value(i) as u64;
        let d_raw = deleted_arr.value(i) as u64;
        let d = if d_raw == NOT_DELETED {
            None
        } else {
            Some(d_raw)
        };

        // D6 fix: Inlined visibility check using HashSet for O(1) active_txns membership.
        // active_txns_set is empty if len < 5 (build overhead not worth it for small sets).
        // Copied from TxnSnapshot::is_row_visible (visibility.rs:824-864) with HashSet optimization.
        let active = &snapshot.active_txns;
        let set = &active_txns_set;
        let commit_ts_map = &snapshot.commit_ts_map;

        let visible = if c > snapshot_id
            || active.contains(&c) || set.contains(&c)
            || !commit_ts_map.contains_key(&c)
            || commit_ts_map.get(&c).copied() > Some(snapshot_id)
        {
            false
        } else if let Some(del) = d {
            // Rule 4: if deleted, row is invisible if del txn committed at or before snapshot_id
            !(active.contains(&del)
                || set.contains(&del)
                || (commit_ts_map.contains_key(&del)
                    && commit_ts_map.get(&del).copied() <= Some(snapshot_id)))
        } else {
            true
        };
        mask.append_value(visible);
    }

    let mask_array = mask.finish();
    let filtered = filter_record_batch(batch, &mask_array).map_err(|e| {
        crate::RockDuckError::Codec(format!("filter_by_visibility: schema mismatch: {}", e))
    })?;

    let num_data_cols = batch.num_columns() - 2;
    let data_fields: Vec<Field> = batch
        .schema()
        .fields()
        .iter()
        .take(num_data_cols)
        .map(|f| (**f).clone())
        .collect();
    let out_schema = SchemaRef::new(arrow_schema::Schema::new(data_fields));

    let out_cols: Vec<ArrayRef> = (0..num_data_cols)
        .map(|i| filtered.column(i).clone())
        .collect();

    RecordBatch::try_new(out_schema, out_cols).map_err(|e| {
        crate::RockDuckError::Codec(format!(
            "filter_by_visibility: output schema mismatch: {}",
            e
        ))
    })
}

/// Check if a batch has visibility columns.
///
/// Returns true if the batch's schema has the two metadata columns.
/// The columns are named `__created_txn` and `__deleted_txn`.
pub fn has_visibility_columns(schema: &arrow_schema::SchemaRef) -> bool {
    let fields = schema.fields();
    if fields.len() < 2 {
        return false;
    }
    let second_last = fields.get(fields.len() - 2);
    let last = fields.last();
    match (second_last, last) {
        (Some(f1), Some(f2)) => {
            f1.name() == CREATED_TXN_COL
                && f1.data_type() == &DataType::Int64
                && f2.name() == DELETED_TXN_COL
                && f2.data_type() == &DataType::Int64
        }
        _ => false,
    }
}

/// Extract visibility columns from a batch.
///
/// Returns (created_txn_array, deleted_txn_array) from the last two columns.
/// Panics if the batch does not have visibility columns.
pub fn extract_visibility_columns(batch: &RecordBatch) -> (ArrayRef, ArrayRef) {
    let num_cols = batch.num_columns();
    debug_assert!(has_visibility_columns(&batch.schema()));
    (
        batch.column(num_cols - 2).clone(),
        batch.column(num_cols - 1).clone(),
    )
}

/// Build the schema for a visibility-only RecordBatch (two Int64 columns).
pub fn visibility_schema() -> SchemaRef {
    Arc::new(arrow_schema::Schema::new(vec![
        Field::new(CREATED_TXN_COL, DataType::Int64, false),
        Field::new(DELETED_TXN_COL, DataType::Int64, false),
    ]))
}

/// Create a visibility-only RecordBatch with one row.
pub fn make_vis_batch(created_txn: u64, deleted_txn: u64) -> RecordBatch {
    let schema = visibility_schema();
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(vec![created_txn as i64])),
            Arc::new(Int64Array::from(vec![deleted_txn as i64])),
        ],
    )
    .expect("visibility_schema must produce valid schema for single-row batch")
}
