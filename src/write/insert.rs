//! Write path for RockDuck
//!
//! Insert, update, and delete records.
//!
//! Write flow (WAL-only durability):
//! 1. Build RecordBatch with user data + visibility columns
//! 2. Write data columns to final vortex files
//! 3. Append vis batch to __vis.vortex
//! 4. PK index write
//! 5. WAL append + flush  <- durability boundary
//!
//! WAL is the sole source of truth. Data files are a cache rebuilt on replay.
//! vis.vortex uses append-only writes; replay appends are idempotent.
//!
//! ## Transaction Rollback
//!
//! If WAL flush fails (e.g., disk full), the transaction should be rolled back.
//! This module provides `rollback_insert_kv_ops` to revert KV operations that
//! occurred before the failed WAL flush. Note that data file writes cannot be
//! trivially rolled back (vortex files are append-only) — they are reconstructed
//! on WAL replay after a crash.
//!
//! ## Two-Phase Prepare-Commit (w002)
//!
//! The insert/update paths follow two-phase semantics:
//! - Phase 0 (Prepare): Capture rollback plan in memory before any mutations.
//! - Phase 1 (Write): Write data files, vis append, PK index.
//! - Phase 2 (Commit): WAL flush — durability boundary.
//! - On WAL failure: Execute rollback plan from memory.

use crate::error::Result;
use crate::metadata::pk_skiplist;
use crate::metadata::GranuleId;
use crate::mvcc::shadow_columns as sc;
use crate::read::point_get::bloom_insert;
use crate::segment::layout::{generate_seg_id, SegmentLayout};
use crate::segment::meta::SegmentMeta;
use crate::storage::vortex::VortexWriter;
use crate::write::durability_wal::{OpPayload, OpType};
use crate::write::vis_file::VisFileWriter;
use crate::write::wal_utils::{arrow_to_bytes, batch_to_ipc_stream};
use crate::{RockDuck, RockDuckError};
use arrow_array::{ArrayRef, RecordBatch};
use fastbloom::{BloomFilter, DefaultHasher};
use std::collections::HashMap;
use std::sync::Arc;

// =============================================================================
// Rollback Plan (w002: Two-phase prepare-commit)
// =============================================================================

/// Captures all state needed to roll back a write transaction.
/// Built in Phase 0 (prepare), executed only on WAL flush failure.
/// This allows rollback without accessing KV store at the critical moment.
#[derive(Debug)]
struct InsertRollbackPlan {
    table: String,
    pk: Vec<u8>,
    seg_id: String,
    granule_id: GranuleId,
    row_offset: u32,
    txn_id: u64,
}

// =============================================================================
// Public API
// =============================================================================

/// Insert a single record.
///
/// WAL-first pattern — WAL is the durability boundary:
/// 1. Write data columns to final vortex files
/// 2. Append vis batch to __vis.vortex
/// 3. PK index write + bloom filter
/// 4. WAL append + flush  (durability boundary)
/// 5. Zone map update
///
/// WAL contains the complete record; on crash recovery the data files are
/// rebuilt by replaying WAL entries. Data file writes are not durability-critical.
pub fn insert(
    db: &RockDuck,
    table: &str,
    pk: &[u8],
    columns: &HashMap<String, ArrayRef>,
    explicit_txn: Option<u64>,
) -> Result<u64> {
    let txn_id = match explicit_txn {
        Some(id) => id,
        None => db.next_txn_id().map_err(|e| {
            RockDuckError::Internal(format!("failed to allocate transaction ID: {}", e))
        })?,
    };
    let batch = columns_to_batch(columns)?;

    let result = allocate_position(db, table, txn_id)?;
    ensure_segment_columns(db, &result.seg_id, &batch)?;

    let layout = SegmentLayout::new(&db.data_dir, &result.seg_id);

    let full_batch = sc::append_visibility_columns(&batch, txn_id, sc::NOT_DELETED);
    let vis_col_idx = full_batch.num_columns() - 2;
    let num_data_cols = full_batch.num_columns() - 2;

    // Phase 0 (w002): Build rollback plan BEFORE any mutations.
    // This is stored in memory and only executed if WAL flush fails.
    // We don't modify any state yet — this is purely preparatory.
    let rollback_plan = InsertRollbackPlan {
        table: table.to_string(),
        pk: pk.to_vec(),
        seg_id: result.seg_id.clone(),
        granule_id: result.granule_id,
        row_offset: result.row_offset,
        txn_id,
    };

    // Phase 1a: Write data columns directly to final vortex files.
    // Not durability-critical — WAL has the data.
    for col_idx in 0..num_data_cols {
        let col_name = full_batch.schema().field(col_idx).name().to_string();
        let sliced = full_batch.column(col_idx).slice(0, 1);
        write_column_data_final(&layout, &col_name, &[sliced])?;
    }

    // Phase 1b: Append vis batch to __vis.vortex (append-only, idempotent).
    let vis_path = layout.vis_path();
    let vis_arrays = [
        full_batch.column(vis_col_idx).clone(),
        full_batch.column(vis_col_idx + 1).clone(),
    ];
    VisFileWriter::new(&vis_path).append_batch(&vis_arrays)?;

    // Phase 2: Write PK index and stats
    pk_skiplist::put_pk_index_double(
        &db.kv,
        table,
        pk,
        &result.seg_id,
        result.granule_id,
        result.row_offset,
    )?;
    update_table_stats(db, table, 1, 0, 0, txn_id)?;
    insert_pk_into_bloom_filter(db, &result.seg_id, pk)?;

    // Phase 3: WAL flush — this is the durability boundary.
    // On failure, execute rollback plan from memory.
    let wal_result = {
        let wal = &*db.wal;
        let (columns, wal_batch) = batch_to_wal_bytes(&full_batch)?;
        wal.append_durable(
            OpType::Insert,
            txn_id,
            &OpPayload::Insert {
                table: table.to_string(),
                pk: pk
                    .try_into()
                    .map_err(|_| RockDuckError::Write("pk must be exactly 8 bytes".into()))?,
                columns,
                wal_batch: Arc::new(wal_batch),
                schema_bytes: Vec::new(),
                seg_id: result.seg_id.clone(),
                granule_id: result.granule_id,
                offset: result.row_offset as u64,
            },
        )
    };

    if let Err(e) = wal_result {
        // WAL flush failed — execute rollback plan to maintain consistency.
        // w002 fix: rollback plan was captured in Phase 0, so we can execute it
        // without needing to reconstruct the state at the critical moment.
        tracing::error!(
            "insert: WAL flush failed for txn {}: {}. Rolling back with prepared plan.",
            txn_id,
            e
        );
        rollback_with_plan(&rollback_plan, db);
        return Err(e);
    }

    // Phase 3b: CDC entry capture (Insert — after image only)
    {
        let mut after_cols: Vec<(String, Vec<u8>)> = Vec::new();
        for col_idx in 0..num_data_cols {
            let col_name = full_batch.schema().field(col_idx).name().to_string();
            let arr = full_batch.column(col_idx);
            if let Some(bytes) = arrow_to_bytes(arr) {
                after_cols.push((col_name, bytes));
            }
        }
        let cdc_entry = crate::cdc::CdcLogEntry {
            op: crate::cdc::CdcOpType::Insert,
            table: table.to_string(),
            seg_id: result.seg_id.clone(),
            pk: pk.to_vec(),
            before: Vec::new(),
            after: after_cols,
            txn_id,
            ts: crate::codec::current_timestamp_millis(),
        };
        // CDC staging never fails; actual flush with error handling happens in commit_txn
        db.push_cdc_entry(cdc_entry);
    }

    // Phase 4: Update zone map statistics
    let zm_path = layout.zone_map_path();
    if let Ok(mut zm) = crate::metadata::ZoneMapStats::load(&zm_path) {
        for col_idx in 0..num_data_cols {
            let arr = full_batch.column(col_idx);
            zm.update_col(col_idx as i32, arr);
        }
        if let Err(e) = zm.save(&zm_path) {
            tracing::warn!(
                "insert: failed to save ZoneMap after writing txn {} to seg {}: {}",
                txn_id,
                result.seg_id,
                e
            );
        }
    }

    Ok(txn_id)
}

/// Delete a record by primary key using Shadow Column visibility.
///
/// WAL-first pattern — WAL is the durability boundary:
/// 1. WAL append + flush  (durability boundary)
/// 2. PK index delete  (safe to do after WAL: both this and vis are idempotent on recovery)
/// 3. vis append  (not durability-critical; WAL is the source of truth)
///
/// Moving PK index delete after the WAL flush is safe because:
/// - WAL recovery replays Delete by first marking vis deleted, then removing the PK index entry.
/// - Both operations are idempotent: vis append always appends, PK delete succeeds even if absent.
/// - Recovery has the full OpPayload with (seg_id, granule_id, offset) to delete the PK entry directly.
pub fn delete(db: &RockDuck, table: &str, pk: &[u8], txn_id: Option<u64>) -> Result<()> {
    let txn_id = match txn_id {
        Some(id) => id,
        None => db.next_txn_id().map_err(|e| {
            RockDuckError::Internal(format!("failed to allocate transaction ID: {}", e))
        })?,
    };

    let entry = pk_skiplist::get_pk_index_by_pk(&db.kv, table, pk)?
        .ok_or_else(|| RockDuckError::KeyNotFound(format!("PK not found: {:?}", pk)))?;

    let (seg_id, granule_id, row_offset) = entry;
    let layout = crate::segment::layout::SegmentLayout::new(&db.data_dir, &seg_id);

    let before_row = read_row_before_image(db, &seg_id, row_offset)?;

    update_table_stats(db, table, 0, 1, 0, txn_id)?;

    // Phase 1: WAL flush (durability boundary) — must happen before any state mutation.
    {
        let wal = &*db.wal;
        wal.append_durable(
            OpType::Delete,
            txn_id,
            &OpPayload::Delete {
                table: table.to_string(),
                pk: pk
                    .try_into()
                    .map_err(|_| RockDuckError::Write("pk must be exactly 8 bytes".into()))?,
                before_row: before_row.clone(),
                seg_id: seg_id.clone(),
                granule_id,
                offset: row_offset as u64,
            },
        )?;
    }

    // Phase 2: PK index delete — AFTER WAL flush. Safe because WAL recovery is idempotent
    // and KV delete is idempotent (NotFound is swallowed).
    if let Err(e) = pk_skiplist::delete_pk_index_double(&db.kv, table, pk, &seg_id, granule_id) {
        tracing::error!(
            "Delete: PK index delete failed after WAL flush for pk={:?}: {}",
            pk,
            e
        );
        return Err(e);
    }

    // Phase 3: CDC entry capture (Delete — before image only)
    {
        let cdc_entry = crate::cdc::CdcLogEntry {
            op: crate::cdc::CdcOpType::Delete,
            table: table.to_string(),
            seg_id: seg_id.clone(),
            pk: pk.to_vec(),
            before: before_row,
            after: Vec::new(),
            txn_id,
            ts: crate::codec::current_timestamp_millis(),
        };
        db.push_cdc_entry(cdc_entry);
    }

    // Phase 4: vis append AFTER WAL flush and PK delete.
    // WAL provides durability; vis append is idempotent and not correctness-critical.
    let vis_path = layout.vis_path();
    mark_visibility_deleted(&vis_path, granule_id, row_offset, txn_id)?;

    Ok(())
}

/// Update a record by primary key.
///
/// Append-only semantics: marks old row deleted in __vis.vortex, appends new row.
/// WAL is the durability boundary.
pub fn update(
    db: &RockDuck,
    table: &str,
    pk: &[u8],
    columns: &HashMap<String, ArrayRef>,
    txn_id: Option<u64>,
) -> Result<u64> {
    let txn_id = match txn_id {
        Some(id) => id,
        None => db.next_txn_id().map_err(|e| {
            RockDuckError::Internal(format!("failed to allocate transaction ID: {}", e))
        })?,
    };

    let entry = pk_skiplist::get_pk_index_by_pk(&db.kv, table, pk)?
        .ok_or_else(|| RockDuckError::KeyNotFound(format!("PK not found: {:?}", pk)))?;

    let (old_seg_id, old_granule_id, old_row_offset) = entry;

    let old_columns = read_row_before_image(db, &old_seg_id, old_row_offset)?;
    let old_layout = crate::segment::layout::SegmentLayout::new(&db.data_dir, &old_seg_id);

    // Phase 1a: Mark old row deleted in __vis.vortex (append-only).
    let old_vis_path = old_layout.vis_path();
    mark_visibility_deleted(&old_vis_path, old_granule_id, old_row_offset, txn_id)?;

    // Phase 1b: Write new data to final vortex files.
    let result = allocate_position(db, table, txn_id)?;
    ensure_segment_columns(db, &result.seg_id, &columns_to_batch(columns)?)?;
    let new_layout = SegmentLayout::new(&db.data_dir, &result.seg_id);
    let batch = columns_to_batch(columns)?;

    let full_batch = sc::append_visibility_columns(&batch, txn_id, sc::NOT_DELETED);
    let vis_col_idx = full_batch.num_columns() - 2;
    let num_data_cols = full_batch.num_columns() - 2;

    for col_idx in 0..num_data_cols {
        let col_name = full_batch.schema().field(col_idx).name().to_string();
        let sliced = full_batch.column(col_idx).slice(0, 1);
        write_column_data_final(&new_layout, &col_name, &[sliced])?;
    }

    // Phase 1c: Append vis batch for new row.
    let new_vis_path = new_layout.vis_path();
    let vis_arrays = [
        full_batch.column(vis_col_idx).clone(),
        full_batch.column(vis_col_idx + 1).clone(),
    ];
    VisFileWriter::new(&new_vis_path).append_batch(&vis_arrays)?;

    // Phase 2: WAL flush (durability boundary).
    {
        let wal = &*db.wal;
        let (columns, wal_batch) = batch_to_wal_bytes(&full_batch)?;
        wal.append_durable(
            OpType::Update,
            txn_id,
            &OpPayload::Update {
                table: table.to_string(),
                pk: pk
                    .try_into()
                    .map_err(|_| RockDuckError::Write("pk must be exactly 8 bytes".into()))?,
                columns,
                wal_batch: Arc::new(wal_batch),
                schema_bytes: Vec::new(),
                old_columns: old_columns.clone(),
                old_seg_id: old_seg_id.clone(),
                old_granule_id,
                old_offset: old_row_offset as u64,
                new_seg_id: result.seg_id.clone(),
                new_granule_id: result.granule_id,
                offset: result.row_offset as u64,
            },
        )?;
    }

    // Phase 2b: CDC entry capture (Update — both before and after images)
    {
        let mut after_cols: Vec<(String, Vec<u8>)> = Vec::new();
        for col_idx in 0..num_data_cols {
            let col_name = full_batch.schema().field(col_idx).name().to_string();
            let arr = full_batch.column(col_idx);
            if let Some(bytes) = arrow_to_bytes(arr) {
                after_cols.push((col_name, bytes));
            }
        }
        let cdc_entry = crate::cdc::CdcLogEntry {
            op: crate::cdc::CdcOpType::Update,
            table: table.to_string(),
            seg_id: result.seg_id.clone(),
            pk: pk.to_vec(),
            before: old_columns,
            after: after_cols,
            txn_id,
            ts: crate::codec::current_timestamp_millis(),
        };
        // CDC staging never fails; actual flush with error handling happens in commit_txn
        db.push_cdc_entry(cdc_entry);
    }

    // Phase 3: PK index update — after WAL flush so WAL is the durability boundary.
    if let Err(e) =
        pk_skiplist::delete_pk_index_double(&db.kv, table, pk, &old_seg_id, old_granule_id)
    {
        tracing::error!("Update: PK index delete failed after WAL flush: {}", e);
        return Err(e);
    }
    if let Err(e) = pk_skiplist::put_pk_index_double(
        &db.kv,
        table,
        pk,
        &result.seg_id,
        result.granule_id,
        result.row_offset,
    ) {
        tracing::error!("Update: PK index put failed after WAL flush: {}", e);
        return Err(e);
    }

    Ok(txn_id)
}

// =============================================================================
// Internal helpers
// =============================================================================

fn columns_to_batch(columns: &HashMap<String, ArrayRef>) -> Result<RecordBatch> {
    if columns.is_empty() {
        return Err(RockDuckError::InvalidParameter(
            "No columns provided".to_string(),
        ));
    }

    let mut fields = Vec::new();
    let mut arrays = Vec::new();

    let first_len = columns.values().next().map(|a| a.len()).unwrap_or(0);
    if first_len == 0 {
        return Err(RockDuckError::InvalidParameter(
            "Empty columns not allowed".to_string(),
        ));
    }

    for (name, array) in columns.iter() {
        if array.len() != first_len {
            return Err(RockDuckError::InvalidParameter(format!(
                "Column '{}' has {} rows, expected {}",
                name,
                array.len(),
                first_len
            )));
        }
        let field = arrow_schema::Field::new(name, array.data_type().clone(), true);
        fields.push(field);
        arrays.push(array.clone() as ArrayRef);
    }

    let schema = arrow_schema::SchemaRef::new(arrow_schema::Schema::new(fields));
    RecordBatch::try_new(schema, arrays)
        .map_err(|e| RockDuckError::InvalidParameter(format!("Failed to create batch: {}", e)))
}

fn batch_to_wal_bytes(batch: &RecordBatch) -> Result<(Vec<String>, Vec<u8>)> {
    let column_names: Vec<String> = (0..batch.num_columns())
        .map(|i| batch.schema().field(i).name().clone())
        .collect();
    let ipc_bytes = batch_to_ipc_stream(batch)?;
    Ok((column_names, ipc_bytes))
}

/// Granule size: rows are grouped into granules of 8192 rows each.
pub const GRANULE_SIZE: u32 = 8192;

struct PositionResult {
    seg_id: String,
    granule_id: GranuleId,
    row_offset: u32,
}

fn allocate_position(db: &RockDuck, table: &str, _txn_id: u64) -> Result<PositionResult> {
    let seg_id = get_or_create_active_segment(db, table)?;
    let row_offset = db.kv.atomic_increment(
        crate::metadata::CF_SEG_META,
        &crate::metadata::kv_store::segment_row_count_key(&seg_id),
        1,
    )? - 1;
    let row_offset = row_offset as u32;
    let granule_id_val = row_offset / GRANULE_SIZE;
    let granule_id = GranuleId::new(granule_id_val);
    Ok(PositionResult {
        seg_id,
        granule_id,
        row_offset,
    })
}

fn get_or_create_active_segment(db: &RockDuck, table: &str) -> Result<String> {
    if let Some(seg_id) = find_active_segment_for_table(db, table)? {
        return Ok(seg_id);
    }

    let seg_id = format!("seg_{}", generate_seg_id());

    let layout = SegmentLayout::new(&db.data_dir, &seg_id);
    layout.create_dirs()?;

    let meta = SegmentMeta::new(seg_id.clone(), table.to_string(), Vec::new());
    crate::metadata::put_segment_meta(&db.kv, &meta)?;
    db.seg_meta_cache.write().invalidate(&seg_id);

    let bf = create_initial_bloom_filter(db.config.bloom_filter_fpp);
    db.segment_bloom_filters.insert(seg_id.clone(), bf);

    Ok(seg_id)
}

fn find_active_segment_for_table(db: &RockDuck, table: &str) -> Result<Option<String>> {
    {
        let cache = db.seg_meta_cache.read();
        for seg_id in cache.get_seg_ids_for_table(table) {
            if let Some(meta) = cache.get(&seg_id) {
                if meta.status == crate::segment::meta::SegmentStatus::Active {
                    return Ok(Some(meta.seg_id.clone()));
                }
            }
        }
    }

    let segments = crate::metadata::list_segment_metas(&db.kv)?;
    for meta in segments {
        if meta.table_id == table && meta.status == crate::segment::meta::SegmentStatus::Active {
            let _ = db.seg_meta_cache.read().put(&meta);
            return Ok(Some(meta.seg_id));
        }
    }
    Ok(None)
}

/// Write column data directly to its final vortex file.
/// Used by both the normal insert path and WAL recovery replay.
/// Covering write is safe: replay always writes the correct committed state.
pub fn write_column_data_final(
    layout: &SegmentLayout,
    col_name: &str,
    data: &[ArrayRef],
) -> Result<()> {
    let path = layout.col_path(col_name);
    let arr = data
        .first()
        .ok_or_else(|| RockDuckError::Internal("No data to write".to_string()))?;

    if arr.is_empty() {
        return Ok(());
    }

    let field = arrow_schema::Field::new(col_name, arr.data_type().clone(), true);
    let schema = arrow_schema::SchemaRef::new(arrow_schema::Schema::new(vec![field]));
    let arrays: Vec<_> = vec![arr.clone() as arrow_array::ArrayRef];
    let batch = RecordBatch::try_new(schema, arrays)
        .map_err(|e| RockDuckError::Internal(format!("Failed to create batch: {}", e)))?;

    let picker = crate::storage::vortex_alp_ext::AdaptiveEncodingPicker::new(
        crate::codec::column_encoding::TableEncodingConfig::default(),
    );
    let mut writer = VortexWriter::with_encoding_picker(&path, col_name, picker);
    writer.write(batch)?;
    writer.finish()?;

    Ok(())
}

/// Mark a single row as deleted in __vis.vortex by appending to the deltavis file.
/// vis append is idempotent: replaying the same delete produces the same state.
pub fn mark_visibility_deleted(
    vis_path: &std::path::Path,
    _granule_id: GranuleId,
    row_offset: u32,
    txn_id: u64,
) -> Result<()> {
    VisFileWriter::new(vis_path).mark_deleted(row_offset as u64, txn_id)
}

/// Read a single row's column values from segment vortex files.
/// Used to capture before-image for CDC Update and Delete events.
fn read_row_before_image(
    db: &RockDuck,
    seg_id: &str,
    row_offset: u32,
) -> Result<Vec<(String, Vec<u8>)>> {
    let layout = crate::segment::layout::SegmentLayout::new(&db.data_dir, seg_id);

    let seg_meta = crate::metadata::get_segment_meta(&db.kv, seg_id)?
        .ok_or_else(|| crate::RockDuckError::SegmentNotFound(seg_id.to_string()))?;

    let mut before_row = Vec::new();
    for col_def in &seg_meta.columns {
        let col_name = &col_def.name;
        let col_path = layout.col_path(col_name);
        if !col_path.exists() {
            // col_path does not exist: the column file was not written (e.g., after a partial
            // write). We still need the column in the row for schema completeness.
            // Build a NULL array of length 1 with the correct data type.
            let null_arr = crate::codec::make_null_array(&col_def.data_type.to_arrow(), 1)
                .map_err(|e| {
                    crate::RockDuckError::Internal(format!(
                        "make_null_array for '{}': {}",
                        col_name, e
                    ))
                })?;
            let field = arrow_schema::Field::new(col_name, null_arr.data_type().clone(), true);
            let schema = arrow_schema::SchemaRef::new(arrow_schema::Schema::new(vec![field]));
            let single_batch =
                arrow_array::RecordBatch::try_new(schema, vec![null_arr as arrow_array::ArrayRef])
                    .map_err(|e| {
                        crate::RockDuckError::Internal(format!(
                            "build null-row batch for '{}': {}",
                            col_name, e
                        ))
                    })?;
            let bytes = crate::write::wal_utils::batch_to_bytes(&single_batch)?;
            before_row.push((col_name.clone(), bytes));
            continue;
        }

        let reader = crate::storage::vortex::VortexReader::open(&col_path)?;
        let batches = reader.read_all_batches();
        let mut cumulative = 0u32;
        for batch in batches.iter() {
            let n = batch.num_rows() as u32;
            let end = cumulative + n;
            if row_offset >= cumulative && row_offset < end {
                let batch_row_idx = (row_offset - cumulative) as usize;
                let col_arr = batch.column(0);
                let sliced = col_arr.slice(batch_row_idx, 1);
                let is_nullable = batch.schema().field(0).is_nullable();
                let field =
                    arrow_schema::Field::new(col_name, sliced.data_type().clone(), is_nullable);
                let schema = arrow_schema::SchemaRef::new(arrow_schema::Schema::new(vec![field]));
                let single_batch = arrow_array::RecordBatch::try_new(
                    schema,
                    vec![sliced as arrow_array::ArrayRef],
                )
                .map_err(|e| {
                    crate::RockDuckError::Internal(format!("build single-row batch: {}", e))
                })?;
                let bytes = crate::write::wal_utils::batch_to_bytes(&single_batch)?;
                before_row.push((col_name.clone(), bytes));
                break;
            }
            cumulative += n;
        }
    }
    Ok(before_row)
}

fn update_table_stats(
    db: &RockDuck,
    table: &str,
    rows_added: u64,
    rows_deleted: u64,
    new_segments: u32,
    txn_id: u64,
) -> Result<()> {
    if rows_added > 0 {
        crate::metadata::increment_row_count(&db.kv, table, rows_added as i64)?;
    }
    if rows_deleted > 0 {
        crate::metadata::increment_row_count(&db.kv, table, -(rows_deleted as i64))?;
    }
    let mut stats = crate::metadata::get_or_create_table_stats(&db.kv, table)?;
    stats.segment_count = stats.segment_count.saturating_add(new_segments as u64);
    stats.last_updated_txn = txn_id;
    crate::metadata::put_table_stats(&db.kv, &stats)?;
    Ok(())
}

pub fn update_table_stats_replay(
    db: &RockDuck,
    table: &str,
    rows_added: u64,
    rows_deleted: u64,
    new_segments: u32,
    txn_id: u64,
) -> Result<()> {
    update_table_stats(db, table, rows_added, rows_deleted, new_segments, txn_id)
}

fn create_initial_bloom_filter(fpp: f64) -> BloomFilter {
    create_bloom_filter(1_000_000, fpp)
}

pub fn create_bloom_filter(expected_items: usize, fpp: f64) -> BloomFilter<DefaultHasher> {
    BloomFilter::with_false_pos(fpp).expected_items(expected_items)
}

fn insert_pk_into_bloom_filter(db: &RockDuck, seg_id: &str, pk: &[u8]) -> Result<()> {
    let bf_arc = db.segment_bloom_filters.get_or_insert_with(seg_id, || {
        create_bloom_filter(1000, db.config.bloom_filter_fpp)
    });
    let mut bf = bf_arc.lock();
    bloom_insert(&mut bf, pk);
    Ok(())
}

fn ensure_segment_columns(db: &RockDuck, seg_id: &str, batch: &RecordBatch) -> Result<()> {
    let mut meta = crate::metadata::get_segment_meta(&db.kv, seg_id)?
        .ok_or_else(|| RockDuckError::SegmentNotFound(seg_id.to_string()))?;

    if meta.columns.is_empty() {
        let schema = batch.schema();
        let columns: Vec<crate::segment::meta::ColumnDef> = (0..batch.num_columns())
            .map(|i| {
                let field = schema.field(i);
                let dtype = crate::segment::meta::DataType::from_arrow(field.data_type());
                crate::segment::meta::ColumnDef::new(field.name().clone(), dtype)
            })
            .collect();
        meta.columns = columns;
        meta.has_visibility_columns = true;
        crate::metadata::put_segment_meta(&db.kv, &meta)?;
        db.seg_meta_cache.write().invalidate(seg_id);
    }
    Ok(())
}

// =============================================================================
// WAL Recovery Replay Helpers
// =============================================================================

/// Write visibility columns to __vis.vortex during WAL recovery replay.
pub fn write_vis_column_replay(layout: &SegmentLayout, vis_arrays: &[ArrayRef]) -> Result<()> {
    VisFileWriter::new(&layout.vis_path()).append_batch(vis_arrays)
}

/// Mark a row as deleted in __vis.vortex during WAL recovery replay.
pub fn mark_visibility_deleted_replay(
    vis_final_path: std::path::PathBuf,
    row_offset: u32,
    txn_id: u64,
) -> Result<()> {
    VisFileWriter::new(&vis_final_path).mark_deleted(row_offset as u64, txn_id)
}

// =============================================================================
// Transaction Rollback
// =============================================================================

/// Rollback KV operations from a failed insert using a pre-built rollback plan.
///
/// This function reverts the KV operations that occurred before the WAL flush
/// that failed. The rollback plan was captured BEFORE Phase 1 began, so we don't
/// need to reconstruct the state at the critical moment (w002 fix).
///
/// - PK index entry for the inserted PK
/// - Table row count (decremented)
///
/// Note: Data file writes (vortex files, vis.vortex) cannot be trivially
/// rolled back as they use append-only writes. After a crash and recovery,
/// WAL replay will reconstruct the correct state by skipping this uncommitted
/// transaction's data file modifications.
fn rollback_with_plan(plan: &InsertRollbackPlan, db: &RockDuck) {
    // Revert PK index write
    let pk_index_key = pk_skiplist::pk_index_key(&plan.seg_id, plan.granule_id, &plan.pk);
    if let Err(e) = db.kv.delete(crate::metadata::CF_PK_IDX, &pk_index_key) {
        tracing::error!(
            "rollback_with_plan: failed to delete PK index for txn {} seg {} offset {}: {}",
            plan.txn_id,
            plan.seg_id,
            plan.row_offset,
            e
        );
    }

    // Revert table row count (decrement by 1)
    if let Err(e) = crate::metadata::increment_row_count(&db.kv, &plan.table, -1) {
        tracing::error!(
            "rollback_with_plan: failed to decrement row count for txn {} table {}: {}",
            plan.txn_id,
            plan.table,
            e
        );
    }

    tracing::debug!(
        "rollback_with_plan: completed for txn {} (pk={:?})",
        plan.txn_id,
        &plan.pk[..4]
    );
}

/// Rollback KV operations from a failed insert.
///
/// This function reverts the KV operations that occurred before the WAL flush
/// that failed. Call this after a WAL flush failure to clean up:
///
/// - PK index entry for the inserted PK
/// - Table row count (decremented)
/// - Bloom filter entry
///
/// Note: Data file writes (vortex files, vis.vortex) cannot be trivially
/// rolled back as they use append-only writes. After a crash and recovery,
/// WAL replay will reconstruct the correct state by skipping this uncommitted
/// transaction's data file modifications.
pub fn rollback_insert_kv_ops(
    db: &RockDuck,
    table: &str,
    pk: &[u8],
    seg_id: &str,
    granule_id: GranuleId,
    row_offset: u32,
    txn_id: u64,
) {
    // Revert PK index write
    let pk_index_key = pk_skiplist::pk_index_key(seg_id, granule_id, pk);
    if let Err(e) = db.kv.delete(crate::metadata::CF_PK_IDX, &pk_index_key) {
        tracing::error!(
            "rollback_insert: failed to delete PK index for txn {} seg {} offset {}: {}",
            txn_id,
            seg_id,
            row_offset,
            e
        );
    }

    // Revert table row count (decrement by 1)
    if let Err(e) = crate::metadata::increment_row_count(&db.kv, table, -1) {
        tracing::error!(
            "rollback_insert: failed to decrement row count for txn {} table {}: {}",
            txn_id,
            table,
            e
        );
    }

    // Bloom filter entry is in a separate cache; best effort removal
    // Note: Bloom filters don't support removal, so this is logged for observability
    tracing::debug!(
        "rollback_insert: bloom filter for seg {} txn {} cannot be reverted (bloom filters don't support removal)",
        seg_id, txn_id
    );
}
