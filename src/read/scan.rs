//! 范围扫描
//!
//! 扫描主键范围内的所有记录
//!
//! 流程：
//! 1. 范围查询 RocksDB 主键索引
//! 2. 过滤删除记录
//! 3. 读取 Vortex 数据
//! 4. 返回 RecordBatch 流
//!
//! Secondary Projection 加速（Feature 1）：
//! - 如果查询有 filter 且 filter 列上有可用 projection → 用 projection 快速裁剪
//!
//! Mmap 优化（Feature 2）：
//! - Frozen 状态的 segment 使用 mmap 读取列文件，实现零拷贝

use arrow_array::{RecordBatch, ArrayRef};
use arrow_schema::{Schema, Field};
use arrow::compute::filter_record_batch;
use crate::{RockDuck, RockDuckError};
use crate::error::Result;
use crate::metadata::IndexEntry;
use crate::metadata::pk_skiplist;
use crate::segment::layout::SegmentLayout;
use crate::codec::decode;
use crate::query::filter_expr;
use crate::query::router::{RouterParams, QueryType, ReadPath, choose_read_path, estimate_selectivity};
use crate::segment::delta_store::apply_deltas_overlay;
use crate::db::TxnId;

/// 范围扫描
pub fn scan(
    db: &RockDuck,
    table: &str,
    pk_range: Option<(Vec<u8>, Vec<u8>)>,
    filter: Option<&str>,
) -> Result<Vec<RecordBatch>> {
    // Feature 1: 尝试使用 Secondary Projection 加速
    if let Some(f) = filter {
        if let Some(projection_result) = try_scan_with_projection(db, table, f)? {
            return Ok(projection_result);
        }
    }

    // D-3: HTAP 双存储路由决策
    let delta_mgr = db.delta_store.read();
    let has_updates = !delta_mgr.segments_with_deltas().is_empty();
    let delta_count = delta_mgr.delta_count();

    let selectivity = estimate_selectivity(filter);
    let params = RouterParams {
        query_type: if pk_range.is_some() { QueryType::RangeScan } else { QueryType::FullScan },
        has_filter: filter.is_some(),
        filter_selectivity: selectivity,
        has_updates,
        delta_count,
        row_range_size: None,
    };
    let read_path = choose_read_path(&params);

    match &read_path {
        ReadPath::VortexOnly => {
            tracing::debug!("scan: routing to VortexOnly (no updates)");
        }
        ReadPath::DeltaStoreOnly { .. } | ReadPath::Merge { .. } => {
            tracing::debug!("scan: routing to {:?} (has_updates={}, delta_count={})", read_path, has_updates, delta_count);
        }
    }

    // 1. 扫描主键索引
    let entries = scan_pk_index(db, table, pk_range)?;

    // 2. 过滤已删除的记录
    let alive_entries: Vec<IndexEntry> = entries
        .into_iter()
        .filter(|e| !is_deleted_entry(db, e).unwrap_or(true))
        .collect();

    // 3. 读取数据（根据路由决策）
    let records = read_records(db, &alive_entries)?;

    // D-3: Merge 路径：DeltaStore overlay + Vortex
    let records = match &read_path {
        ReadPath::VortexOnly => records,
        ReadPath::DeltaStoreOnly { .. } | ReadPath::Merge { .. } => {
            let delta_mgr = db.delta_store.read();
            let seg_id = alive_entries.first().map(|e| &e.seg_id).unwrap_or(&String::new()).clone();
            if !delta_mgr.segments_with_deltas().is_empty() {
                let mut all_merged = Vec::new();
                for batch in records {
                    let merged = merge_with_deltas_inner(batch, &delta_mgr, &seg_id)?;
                    all_merged.push(merged);
                }
                all_merged
            } else {
                records
            }
        }
    };

    // 4. 应用过滤器（如果有）
    let records = if let Some(filter_str) = filter {
        let mut filtered = Vec::new();
        for batch in records {
            let expr = filter_expr::parse(filter_str)
                .map_err(|e| RockDuckError::Query(e))?;
            let predicate = filter_expr::evaluate(&expr, &batch)
                .map_err(|e| RockDuckError::Query(format!("Filter eval error: {}", e)))?;
            let filtered_batch = filter_record_batch(&batch, &predicate)
                .map_err(|e| RockDuckError::Arrow(e))?;
            filtered.push(filtered_batch);
        }
        filtered
    } else {
        records
    };

    Ok(records)
}

/// 合并单个 RecordBatch 与 DeltaStore overlay
fn merge_with_deltas_inner(
    batch: RecordBatch,
    delta_mgr: &crate::segment::delta_store::DeltaStoreManager,
    seg_id: &str,
) -> Result<RecordBatch> {
    let deltas = delta_mgr.get(seg_id)
        .map(|store| store.get_all_visible_deltas())
        .unwrap_or_default();

    if deltas.is_empty() || deltas.values().all(|m| m.is_empty()) {
        return Ok(batch);
    }

    let mut result_batch = batch;
    for col_idx in 0..result_batch.num_columns() {
        let col_name = result_batch.schema().field(col_idx).name().clone();
        if let Some(col_deltas) = deltas.get(&col_name) {
            if !col_deltas.is_empty() {
                let col_array = result_batch.column(col_idx);
                let merged_col = apply_deltas_overlay(col_deltas, col_array.as_ref(), &col_name)?;
                let fields: Vec<_> = (0..result_batch.num_columns())
                    .map(|i| result_batch.schema().field(i).clone())
                    .collect();
                let mut new_columns: Vec<_> = (0..result_batch.num_columns())
                    .map(|i| result_batch.column(i).clone())
                    .collect();
                new_columns[col_idx] = merged_col;
                result_batch = RecordBatch::try_new(
                    std::sync::Arc::new(arrow_schema::Schema::new(fields)),
                    new_columns,
                )?;
            }
        }
    }
    Ok(result_batch)
}

/// 尝试用 Secondary Projection 加速扫描。
/// 如果有适用的 projection，返回扫描结果；否则返回 None 走原有路径。
fn try_scan_with_projection(
    db: &RockDuck,
    table: &str,
    filter: &str,
) -> Result<Option<Vec<RecordBatch>>> {
    // 解析 filter 列名
    let filter_col = filter.split_whitespace().next().unwrap_or("");

    // 列出表上的所有 projections
    let projections = crate::metadata::projection::list_table_projections(&db.db, table)
        .unwrap_or_default();

    // 找适用的 projection
    let proj = match crate::segment::projection::find_applicable_projection(table, filter_col, &projections) {
        Some(p) => p,
        None => return Ok(None),
    };

    // 检查 predicate 是否可以用 projection 裁剪
    if !crate::segment::projection::can_projection_prune(proj, filter) {
        return Ok(None);
    }

    // TODO: 用 projection 的 Zone Map 裁剪 segments
    //       提取匹配的 pk 列表
    //       用 RocksDB pk_idx 批量读主表
    // Currently returns None to fall through to normal scan
    // until projection data loading is implemented
    Ok(None)
}

/// 扫描主键索引（使用 skiplist CF 做有序范围查询）
/// 优先使用 skiplist CF 的有序遍历，O(n) 范围扫描而非 O(n) filter。
fn scan_pk_index(
    db: &RockDuck,
    table: &str,
    pk_range: Option<(Vec<u8>, Vec<u8>)>,
) -> Result<Vec<IndexEntry>> {
    // 优先走 skiplist CF（有序范围扫描，无 filter 开销）
    let pk_range_clone = pk_range.clone();
    match pk_skiplist::batch_scan_pk(&db.db, table, pk_range) {
        Ok(entries) => {
            tracing::debug!("scan_pk_index: skiplist CF returned {} entries", entries.len());
            Ok(entries)
        }
        Err(e) => {
            // Fallback: 兼容旧数据（未双写的历史数据仍在 pk_idx CF）
            tracing::warn!("skiplist CF scan failed, falling back to pk_idx CF: {}", e);
            scan_pk_index_fallback(db, table, pk_range_clone)
        }
    }
}

/// Fallback：使用 pk_idx CF 扫描（用于未双写的历史数据）
fn scan_pk_index_fallback(
    db: &RockDuck,
    table: &str,
    pk_range: Option<(Vec<u8>, Vec<u8>)>,
) -> Result<Vec<IndexEntry>> {
    let cf = db.db.cf_handle("pk_idx")
        .ok_or_else(|| RockDuckError::Metadata("pk_idx column family not found".to_string()))?;

    let prefix = format!("pk:{}:", table);
    let prefix_bytes = prefix.as_bytes().to_vec();
    let start_key = match &pk_range {
        Some((start, _)) => {
            let mut key = prefix_bytes.clone();
            key.extend_from_slice(start);
            key
        }
        None => prefix_bytes.clone(),
    };

    let end_key = pk_range.as_ref().map(|(_, end)| {
        let mut key = prefix_bytes.clone();
        key.extend_from_slice(end);
        key
    });

    let mut entries = Vec::new();

    let mut iter = db.db.raw_iterator_cf(&cf);
    iter.seek(&start_key);

    while iter.valid() {
        if let Some(key) = iter.key() {
            if !key.starts_with(prefix.as_bytes()) {
                break;
            }

            if let Some(ref end) = end_key {
                if key >= end.as_slice() {
                    break;
                }
            }

            if let Some(value) = iter.value() {
                if let Ok(entry) = decode::<IndexEntry>(value) {
                    entries.push(entry);
                }
            }
        } else {
            break;
        }

        iter.next();
    }

    Ok(entries)
}

/// 检查索引条目是否已删除
fn is_deleted_entry(db: &RockDuck, entry: &IndexEntry) -> Result<bool> {
    let seg_id = &entry.seg_id;
    let offset = entry.offset;

    let layout = SegmentLayout::new(&db.data_dir, seg_id);
    let del_path = layout.del_mask_path();
    
    if !del_path.exists() {
        return Ok(false);
    }
    
    let del_data = std::fs::read(&del_path)?;
    
    let byte_pos = offset as usize / 8;
    let bit_pos = offset as usize % 8;
    
    if byte_pos < del_data.len() {
        Ok((del_data[byte_pos] >> bit_pos) & 1 == 1)
    } else {
        Ok(false)
    }
}

/// 批量读取记录
fn read_records(db: &RockDuck, entries: &[IndexEntry]) -> Result<Vec<RecordBatch>> {
    if entries.is_empty() {
        return Ok(Vec::new());
    }

    // 按 segment 分组
    let mut seg_entries: std::collections::HashMap<String, Vec<(u32, u32)>> = std::collections::HashMap::new();

    for entry in entries {
        seg_entries
            .entry(entry.seg_id.clone())
            .or_default()
            .push((entry.granule_id, entry.offset));
    }

    let mut batches = Vec::new();

    for (seg_id, offsets) in seg_entries {
        let batch = read_segment_batch(db, &seg_id, &offsets)?;
        if batch.num_rows() > 0 {
            batches.push(batch);
        }
    }

    Ok(batches)
}

/// 读取 segment 的部分数据
fn read_segment_batch(
    db: &RockDuck,
    seg_id: &str,
    offsets: &[(u32, u32)],
) -> Result<RecordBatch> {
    if offsets.is_empty() {
        let schema = Schema::new(Vec::<Field>::new());
        return Ok(RecordBatch::new_empty(std::sync::Arc::new(schema)));
    }

    // 收集并排序行偏移（去重 + 排序以保证确定性）
    let mut unique_offsets: Vec<u32> = offsets.iter().map(|&(_, off)| off).collect();
    unique_offsets.sort();
    unique_offsets.dedup();

    // 获取 segment 元数据
    let meta = crate::metadata::rocksdb::get_segment_meta(&db.db, seg_id)?
        .ok_or_else(|| RockDuckError::SegmentNotFound(seg_id.to_string()))?;

    let mut all_columns: Vec<Vec<ArrayRef>> = Vec::new();

    for col_def in &meta.columns {
        let batch = read_arrow_file_for_segment(&db.data_dir, &seg_id, &col_def.name)?;
        // Each column file stores one column named "value" - slice rows by offset
        if batch.num_columns() == 0 {
            continue;
        }
        let col_array = batch.column(0);
        let mut filtered_rows: Vec<ArrayRef> = Vec::new();

        for &row_offset in &unique_offsets {
            if (row_offset as usize) < col_array.len() {
                filtered_rows.push(col_array.slice(row_offset as usize, 1));
            }
        }

        all_columns.push(filtered_rows);
    }

    // 构建 RecordBatch
    let fields: Vec<Field> = meta.columns.iter().map(|c| {
        Field::new(&c.name, c.dtype.to_arrow(), true)
    }).collect();

    if fields.is_empty() {
        let schema = Schema::new(Vec::<Field>::new());
        return Ok(RecordBatch::new_empty(std::sync::Arc::new(schema)));
    }

    // 每列应该只有一行（每个 offset 对应一行）
    let columns: Vec<ArrayRef> = all_columns
        .into_iter()
        .map(|col_slices| {
            if col_slices.len() == 1 {
                col_slices.into_iter().next().unwrap()
            } else {
                combine_filtered_columns(&col_slices).unwrap_or_else(|_| {
                    col_slices.into_iter().next().unwrap_or_else(|| {
                        std::sync::Arc::new(arrow_array::Int64Array::from(Vec::<i64>::new())) as arrow_array::ArrayRef
                    })
                })
            }
        })
        .collect();

    let schema = Schema::new(fields);
    RecordBatch::try_new(std::sync::Arc::new(schema), columns)
        .map_err(|e| RockDuckError::Storage(format!("Failed to create RecordBatch: {}", e)))
}

/// 合并多个单行数组为一个数组
fn combine_filtered_columns(arrays: &[ArrayRef]) -> Result<ArrayRef> {
    if arrays.is_empty() {
        return Err(RockDuckError::Storage("No arrays to combine".to_string()));
    }
    if arrays.len() == 1 {
        return Ok(arrays[0].clone());
    }

    use arrow::compute::concat;
    let refs: Vec<&dyn arrow_array::Array> = arrays.iter().map(|a| a.as_ref()).collect();
    concat(refs.as_slice())
        .map_err(|e| RockDuckError::Storage(format!("Failed to concat arrays: {}", e)))
}

/// 读取 Arrow IPC 文件，根据 segment 状态自动选择 BufReader 或 mmap
fn read_arrow_file(path: &std::path::Path) -> Result<RecordBatch> {
    let reader = std::fs::File::open(path)?;
    let reader = arrow_ipc::reader::FileReader::try_new(
        std::io::BufReader::new(reader),
        None,
    )?;

    let batches: Vec<RecordBatch> = reader
        .filter_map(|b| b.ok())
        .collect();

    if batches.is_empty() {
        return Err(RockDuckError::Storage("Empty file".to_string()));
    }

    if batches.len() == 1 {
        return Ok(batches.into_iter().next().unwrap());
    }

    // 合并多个 batches 的同一列
    let num_columns = batches[0].num_columns();
    let mut result_columns: Vec<ArrayRef> = Vec::new();

    for col_idx in 0..num_columns {
        let mut combined: Vec<ArrayRef> = batches.iter()
            .map(|b| b.column(col_idx).clone())
            .collect();

        if combined.len() == 1 {
            result_columns.push(combined.remove(0));
        } else {
            let refs: Vec<&dyn arrow_array::Array> = combined.iter().map(|a| a.as_ref()).collect();
            let concat = arrow::compute::concat(refs.as_slice())
                .map_err(|e| RockDuckError::Storage(format!("Failed to concat column {}: {}", col_idx, e)))?;
            result_columns.push(concat);
        }
    }

    let schema = batches[0].schema();
    RecordBatch::try_new(schema, result_columns)
        .map_err(|e| RockDuckError::Storage(format!("Failed to create RecordBatch: {}", e)))
}

/// 通过 mmap 读取 Arrow IPC 文件（Frozen segment 专用，零拷贝）
fn read_arrow_file_mmap(path: &std::path::Path) -> Result<RecordBatch> {
    let file = std::fs::File::open(path)?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };

    let reader = arrow_ipc::reader::FileReader::try_new(
        std::io::Cursor::new(mmap.as_ref()),
        None,
    )?;

    let batches: Vec<RecordBatch> = reader
        .filter_map(|b| b.ok())
        .collect();

    if batches.is_empty() {
        return Err(RockDuckError::Storage("Empty file".to_string()));
    }

    if batches.len() == 1 {
        return Ok(batches.into_iter().next().unwrap());
    }

    let num_columns = batches[0].num_columns();
    let mut result_columns: Vec<ArrayRef> = Vec::new();

    for col_idx in 0..num_columns {
        let mut combined: Vec<ArrayRef> = batches.iter()
            .map(|b| b.column(col_idx).clone())
            .collect();

        if combined.len() == 1 {
            result_columns.push(combined.remove(0));
        } else {
            let refs: Vec<&dyn arrow_array::Array> = combined.iter().map(|a| a.as_ref()).collect();
            let concat = arrow::compute::concat(refs.as_slice())
                .map_err(|e| RockDuckError::Storage(format!("Failed to concat column {}: {}", col_idx, e)))?;
            result_columns.push(concat);
        }
    }

    let schema = batches[0].schema();
    RecordBatch::try_new(schema, result_columns)
        .map_err(|e| RockDuckError::Storage(format!("Failed to create RecordBatch: {}", e)))
}

/// 读取 Arrow IPC 文件（根据 segment 状态选择最优读取方式）
fn read_arrow_file_for_segment(
    data_dir: &std::path::Path,
    seg_id: &str,
    col_name: &str,
) -> Result<RecordBatch> {
    let layout = SegmentLayout::new(data_dir, seg_id);
    let col_path = layout.col_path(col_name);
    let meta_path = layout.meta_path();

    // 检查 segment 状态
    if meta_path.exists() {
        if let Ok(data) = std::fs::read(&meta_path) {
            if let Ok(meta) = decode::<crate::segment::meta::SegmentMeta>(&data) {
                if matches!(meta.status, crate::segment::meta::SegmentStatus::Frozen) {
                    return read_arrow_file_mmap(&col_path);
                }
            }
        }
    }

    read_arrow_file(&col_path)
}

/// 计数查询（使用 Del Mask 优化）
pub fn count(db: &RockDuck, table: &str) -> Result<u64> {
    let stats = db.get_table_stats(table)?;
    Ok(stats.map(|s| s.alive_rows()).unwrap_or(0))
}

/// Time-Travel 范围扫描：在指定事务 ID 的快照下扫描
pub fn scan_as_of(
    db: &RockDuck,
    table: &str,
    txn_id: TxnId,
    pk_range: Option<(Vec<u8>, Vec<u8>)>,
    filter: Option<&str>,
) -> Result<Vec<RecordBatch>> {
    let snapshot = db.snapshot_at(txn_id, crate::mvcc::IsolationLevel::Snapshot)?;

    // Feature 1: 尝试使用 Secondary Projection 加速
    if let Some(f) = filter {
        if let Some(projection_result) = try_scan_with_projection(db, table, f)? {
            return Ok(projection_result);
        }
    }

    // 1. 扫描主键索引
    let entries = scan_pk_index(db, table, pk_range)?;

    // 2. 过滤已删除的记录 + MVCC 可见性过滤
    let alive_entries: Vec<IndexEntry> = entries
        .into_iter()
        .filter(|e| !is_deleted_entry(db, e).unwrap_or(true))
        .filter(|e| db.is_visible(&snapshot, e.txn_id, None))
        .collect();

    // 3. 读取数据
    let records = read_records(db, &alive_entries)?;

    // 4. 应用过滤器（如果有）
    let records = if let Some(filter_str) = filter {
        let mut filtered = Vec::new();
        for batch in records {
            let expr = filter_expr::parse(filter_str)
                .map_err(|e| RockDuckError::Query(e))?;
            let predicate = filter_expr::evaluate(&expr, &batch)
                .map_err(|e| RockDuckError::Query(format!("Filter eval error: {}", e)))?;
            let filtered_batch = filter_record_batch(&batch, &predicate)
                .map_err(|e| RockDuckError::Arrow(e))?;
            filtered.push(filtered_batch);
        }
        filtered
    } else {
        records
    };

    Ok(records)
}

/// 聚合查询
#[allow(dead_code)]
pub fn range_aggregate(
    _db: &RockDuck,
    _table: &str,
    _column: &str,
    agg_fn: &str,
) -> Result<ArrayRef> {
    Err(RockDuckError::Query(format!("Aggregate function {} not implemented", agg_fn)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scan_returns_empty_for_nonexistent_table() {
        let temp_dir = tempfile::tempdir().unwrap();
        let db = crate::RockDuck::open(temp_dir.path()).unwrap();
        let result = scan(&db, "nonexistent_table", None, None).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_range_aggregate_returns_error() {
        let temp_dir = tempfile::tempdir().unwrap();
        let db = crate::RockDuck::open(temp_dir.path()).unwrap();
        let result = range_aggregate(&db, "table", "col", "sum");
        assert!(result.is_err());
    }
}
