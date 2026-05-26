//! 写入路径
//!
//! 插入、更新、删除记录
//!
//! 流程：
//! 1. RocksDB transaction 开始
//! 2. 分配 seg_id + granule_id + offset
//! 3. 追加每列数据到 Vortex
//! 4. 写 _del.vortex 新增位置 = 0
//! 5. RocksDB put pk_idx + 更新 seg_meta
//! 6. 更新 ZoneMap
//! 7. transaction commit

use std::collections::HashMap;
use arrow_array::{RecordBatch, ArrayRef};
use quickbloom::{BloomConfig, BloomFilter, BloomMode};
use rocksdb::{WriteBatch, WriteOptions};
use crate::{RockDuck, RockDuckError};
use crate::error::Result;
use crate::metadata::{IndexEntry, rocksdb::{
    pk_index_key, get_or_create_table_stats, put_table_stats
}};
use crate::metadata::pk_skiplist;
use crate::codec::encode;
use crate::segment::meta::{SegmentMeta, SegmentStatus};
use crate::segment::layout::{SegmentLayout, generate_seg_id};

#[cfg(feature = "async")]
use lazyflow::prelude::*;

use crate::read::point_get::bloom_insert;

/// 插入单条记录
pub fn insert(
    db: &RockDuck,
    table: &str,
    pk: &[u8],
    columns: &HashMap<String, ArrayRef>,
) -> Result<u64> {
    let txn_id = db.next_txn_id();
    let batch = columns_to_batch(columns)?;

    // Allocate position
    let result = allocate_position(db, table, &SegmentLayout::new(&db.data_dir, ""), txn_id)?;
    let new_segments_created = if result.new_segment_created { 1 } else { 0 };
    ensure_segment_columns(db, &result.entry.seg_id, &batch)?;

    // Write all columns for this row using buffered write
    for col_idx in 0..batch.num_columns() {
        let col_name = batch.schema().field(col_idx).name().to_string();
        let sliced = batch.column(col_idx).slice(0, 1);
        let layout = crate::segment::layout::SegmentLayout::new(&db.data_dir, &result.entry.seg_id);
        write_column_data(&layout, &col_name, &[sliced])?;
    }

    // Update row count
    increment_segment_row_count(db, &result.entry.seg_id)?;

    // 双写 RocksDB 索引（pk_idx hash + pk_skiplist 有序）
    pk_skiplist::put_pk_index_double(&db.db, table, pk, &result.entry)?;

    // Update table stats
    update_table_stats(db, table, 1, 0, new_segments_created)?;

    // Insert pk into segment's Bloom Filter
    insert_pk_into_bloom_filter(db, &result.entry.seg_id, pk)?;

    Ok(txn_id)
}

/// 批量插入
pub fn insert_batch(
    db: &RockDuck,
    table: &str,
    pks: &[Vec<u8>],
    columns: &HashMap<String, ArrayRef>,
) -> Result<u64> {
    if pks.is_empty() {
        return Ok(0);
    }
    
    let txn_id = db.next_txn_id();
    
    // 创建 RecordBatch
    let batch = columns_to_batch(columns)?;
    
    // 验证行数匹配
    if batch.num_rows() != pks.len() {
        return Err(RockDuckError::InvalidParameter(format!(
            "Row count mismatch: {} pks vs {} columns",
            pks.len(),
            batch.num_rows()
        )).into());
    }
    
    // 批量写入 - buffer data per segment and write once
    let _layout = SegmentLayout::new(&db.data_dir, "");

    // Group writes by segment: seg_id -> (row_indices, per-column arrays)
    let mut seg_writes: std::collections::HashMap<String, Vec<usize>> = std::collections::HashMap::new();
    // Allocate positions in memory: seg_id -> next_offset
    let mut next_offset: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
    let mut entries: Vec<IndexEntry> = Vec::with_capacity(pks.len());
    let mut new_segments_created = 0u32;

    for (i, _pk) in pks.iter().enumerate() {
        let result = allocate_position_with_tracking(db, table, &mut next_offset, txn_id)?;
        ensure_segment_columns(db, &result.entry.seg_id, &batch)?;
        if result.new_segment_created {
            new_segments_created += 1;
        }
        seg_writes.entry(result.entry.seg_id.clone()).or_default().push(i);
        entries.push(result.entry);
    }

    // Write each segment's data once
    for (seg_id, row_indices) in &seg_writes {
        // Build the RecordBatch for this segment's rows
        let mut col_arrays: Vec<Vec<ArrayRef>> = Vec::new();
        for col_idx in 0..batch.num_columns() {
            let mut arrays: Vec<ArrayRef> = Vec::with_capacity(row_indices.len());
            for &row_idx in row_indices {
                let sliced = batch.column(col_idx).slice(row_idx, 1);
                arrays.push(sliced);
            }
            col_arrays.push(arrays);
        }

        // Write the multi-column RecordBatch to each column file
        write_segment_batch(&db.data_dir, &seg_id, &batch.schema(), &col_arrays)?;

        // Update row count
        increment_segment_row_count_by(db, &seg_id, row_indices.len() as u64)?;
    }

    // 双写 RocksDB 索引（pk_idx hash + pk_skiplist 有序）
    let mut rocksdb_batch = WriteBatch::default();
    let pk_idx_cf = crate::metadata::rocksdb::pk_idx_cf(&db.db)?;
    let pk_skiplist_cf = crate::metadata::rocksdb::pk_skiplist_cf(&db.db)?;
    for (i, pk) in pks.iter().enumerate() {
        let key_hash = pk_index_key(table, pk);
        let key_skip = pk_skiplist::pk_skiplist_key(table, pk);
        let value = encode(&entries[i])?;
        rocksdb_batch.put_cf(pk_idx_cf, &key_hash, &value);
        rocksdb_batch.put_cf(pk_skiplist_cf, &key_skip, &value);
    }

    // 提交事务
    let mut opts = WriteOptions::default();
    opts.set_sync(true);
    db.db.write_opt(rocksdb_batch, &opts)?;

    // 批量插入 pks 到各 segment 的 Bloom Filter
    for (seg_id, row_indices) in &seg_writes {
        for &row_idx in row_indices {
            let pk = &pks[row_idx];
            insert_pk_into_bloom_filter(db, seg_id, pk)?;
        }
    }

    // 更新表统计
    update_table_stats(db, table, pks.len() as u64, 0, new_segments_created)?;
    
    Ok(txn_id)
}

/// 删除记录
pub fn delete(db: &RockDuck, table: &str, pk: &[u8]) -> Result<u64> {
    let txn_id = db.next_txn_id();
    
    // 获取主键索引
    let entry = crate::metadata::rocksdb::get_pk_index(&db.db, table, pk)?
        .ok_or_else(|| RockDuckError::IndexEntryNotFound)?;

    // 更新删除掩码
    let layout = SegmentLayout::new(&db.data_dir, &entry.seg_id);
    let del_path = layout.del_mask_path();

    // 读取现有的删除掩码
    let mut del_data = if del_path.exists() {
        std::fs::read(&del_path)?
    } else {
        vec![0u8; 1024] // 默认 8KB 空间
    };
    
    // 设置删除位
    let byte_pos = entry.offset as usize / 8;
    let bit_pos = entry.offset as usize % 8;
    
    if byte_pos < del_data.len() {
        del_data[byte_pos] |= 1 << bit_pos;
    }
    
    // 写回删除掩码
    std::fs::write(&del_path, &del_data)?;
    
    // 更新 segment 元数据
    if let Some(mut meta) = crate::metadata::rocksdb::get_segment_meta(&db.db, &entry.seg_id)? {
        meta.update_del_stats(1);
        crate::metadata::rocksdb::put_segment_meta(&db.db, &meta)?;
    }
    
    // 更新表统计
    update_table_stats(db, table, 0, 1, 0)?;
    
    Ok(txn_id)
}

/// 更新记录：写入 DeltaStore（cell-level before/after image）
pub fn update(
    db: &RockDuck,
    table: &str,
    pk: &[u8],
    columns: &HashMap<String, ArrayRef>,
) -> Result<u64> {
    let txn_id = db.next_txn_id();

    // 1. 获取主键索引，找到目标位置
    let entry = crate::metadata::rocksdb::get_pk_index(&db.db, table, pk)?
        .ok_or_else(|| RockDuckError::IndexEntryNotFound)?;

    // 2. 获取 segment 元数据（用于列定义）
    let _seg_meta = crate::metadata::rocksdb::get_segment_meta(&db.db, &entry.seg_id)?
        .ok_or_else(|| RockDuckError::SegmentNotFound(entry.seg_id.clone()))?;

    // 3. 获取 DeltaStore（write guard held for entire update）
    let mut delta_mgr = db.delta_store.write();
    let seg_store = delta_mgr.get_or_create(&entry.seg_id);
    seg_store.begin_txn(txn_id);

    // 4. 读取旧值并写入 DeltaStore（before image）
    let layout = crate::segment::layout::SegmentLayout::new(&db.data_dir, &entry.seg_id);

    for (col_name, new_array) in columns {
        // 4a. 读取旧列数据
        let col_path = layout.col_path(col_name);
        if !col_path.exists() {
            continue;
        }

        let old_batch = read_arrow_file_internal(&col_path)?;
        let old_col = old_batch.column(0);

        // 4b. 序列化旧值（before image）
        let old_value = crate::segment::upd_mask::serialize_value(old_col, entry.offset as usize)
            .unwrap_or_default();

        // 4c. 序列化新值（after image）
        let new_value = crate::segment::upd_mask::serialize_value(new_array, 0)
            .unwrap_or_default();

        // 4d. 记录到 DeltaStore
        seg_store.record_update(txn_id, col_name.clone(), entry.offset as u64, old_value, new_value);
    }

    // 5. 持久化 DeltaStore
    drop(delta_mgr);
    let store_guard = db.delta_store.read();
    store_guard.persist(&db.data_dir, &entry.seg_id)?;
    drop(store_guard);

    // 6. 更新 segment 元数据（updated_at）
    if let Some(mut meta) = crate::metadata::rocksdb::get_segment_meta(&db.db, &entry.seg_id)? {
        meta.updated_at = crate::codec::current_timestamp_secs();
        crate::metadata::rocksdb::put_segment_meta(&db.db, &meta)?;
    }

    Ok(txn_id)
}

// ================== 内部辅助函数 ==================

/// 将列数据转换为 RecordBatch
fn columns_to_batch(columns: &HashMap<String, ArrayRef>) -> Result<RecordBatch> {
    if columns.is_empty() {
        return Err(RockDuckError::InvalidParameter("No columns provided".to_string()).into());
    }
    
    let num_rows = columns.values().next()
        .map(|a| a.len())
        .unwrap_or(0);
    
    // 验证所有列长度一致
    for (name, array) in columns {
        if array.len() != num_rows {
            return Err(RockDuckError::InvalidParameter(format!(
                "Column {} has {} rows, expected {}",
                name,
                array.len(),
                num_rows
            )).into());
        }
    }
    
    // 创建 schema
    let fields: Vec<_> = columns.keys()
        .map(|name| {
            let dtype = columns.get(name).unwrap().data_type();
            arrow_schema::Field::new(name, dtype.clone(), true)
        })
        .collect();
    
    let schema = arrow_schema::Schema::new(fields);
    
    // 创建列数组
    let cols: Vec<_> = columns.values().cloned().collect();
    
    RecordBatch::try_new(std::sync::Arc::new(schema), cols)
        .map_err(|e| RockDuckError::Arrow(e).into())
}

/// 分配写入位置
fn allocate_position(
    db: &RockDuck,
    table: &str,
    _layout: &SegmentLayout,
    txn_id: u64,
) -> Result<AllocateResult> {
    // 查找活跃的 segment
    let active_seg = find_active_segment(db, table)?;

    let (seg_id, new_segment_created) = match active_seg {
        Some(seg_id) => (seg_id, false),
        None => {
            let new_seg_id = generate_seg_id();
            create_new_segment(db, table, &new_seg_id, &db.data_dir)?;
            (new_seg_id, true)
        }
    };

    // 获取 segment 元数据
    let meta = crate::metadata::rocksdb::get_segment_meta(&db.db, &seg_id)?
        .ok_or_else(|| RockDuckError::SegmentNotFound(seg_id.clone()))?;

    // 计算新行的位置
    let granule_id = meta.granules.len() as u32;
    let offset = meta.row_count as u32;

    Ok(AllocateResult {
        entry: IndexEntry::new(seg_id, granule_id, offset, txn_id),
        new_segment_created,
    })
}

/// Result of allocating a position, including whether a new segment was created
struct AllocateResult {
    entry: IndexEntry,
    new_segment_created: bool,
}

/// 分配写入位置（带内存跟踪，避免重复读取 RocksDB 元数据）
fn allocate_position_with_tracking(
    db: &RockDuck,
    table: &str,
    next_offset: &mut std::collections::HashMap<String, u32>,
    txn_id: u64,
) -> Result<AllocateResult> {
    // 查找活跃的 segment
    let active_seg = find_active_segment(db, table)?;

    let (seg_id, new_segment_created) = match active_seg {
        Some(seg_id) => (seg_id, false),
        None => {
            let new_seg_id = generate_seg_id();
            create_new_segment(db, table, &new_seg_id, &db.data_dir)?;
            next_offset.insert(new_seg_id.clone(), 0);
            (new_seg_id, true)
        }
    };

    // Use tracked offset instead of reading from RocksDB each time
    let offset = *next_offset.entry(seg_id.clone()).or_insert_with(|| {
        // First time for this seg_id, read the current row_count
        crate::metadata::rocksdb::get_segment_meta(&db.db, &seg_id)
            .ok()
            .flatten()
            .map(|m| m.row_count as u32)
            .unwrap_or(0)
    });
    next_offset.insert(seg_id.clone(), offset + 1);

    // 计算 granule_id
    let meta = crate::metadata::rocksdb::get_segment_meta(&db.db, &seg_id)?;
    let granule_id = meta.map(|m| m.granules.len() as u32).unwrap_or(0);

    Ok(AllocateResult {
        entry: IndexEntry::new(seg_id, granule_id, offset, txn_id),
        new_segment_created,
    })
}

/// 更新 segment 行数
pub fn increment_segment_row_count(db: &RockDuck, seg_id: &str) -> Result<()> {
    increment_segment_row_count_by(db, seg_id, 1)
}

fn increment_segment_row_count_by(db: &RockDuck, seg_id: &str, count: u64) -> Result<()> {
    let mut meta = crate::metadata::rocksdb::get_segment_meta(&db.db, seg_id)?
        .ok_or_else(|| RockDuckError::SegmentNotFound(seg_id.to_string()))?;
    meta.row_count += count;
    meta.updated_at = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    crate::metadata::rocksdb::put_segment_meta(&db.db, &meta)?;
    Ok(())
}

/// 查找活跃的 segment
fn find_active_segment(db: &RockDuck, table: &str) -> Result<Option<String>> {
    let segments = crate::metadata::seg_meta::list_table_segments(&db.db, table)?;
    
    for seg_id in segments.iter().rev() {
        if let Some(meta) = crate::metadata::rocksdb::get_segment_meta(&db.db, seg_id)? {
            if meta.status == SegmentStatus::Active {
                // 检查是否还有空间
                if meta.row_count < 1_000_000 { // 1M 行限制
                    return Ok(Some(seg_id.clone()));
                }
            }
        }
    }
    
    Ok(None)
}

/// 创建新的 segment
fn create_new_segment(
    db: &RockDuck,
    table: &str,
    seg_id: &str,
    data_dir: &std::path::Path,
) -> Result<()> {
    // Use the correct layout with the actual seg_id
    let layout = SegmentLayout::new(data_dir, seg_id);

    // 创建目录
    layout.create_dirs()?;

    // 创建空的 SegmentMeta
    let meta = SegmentMeta::new(
        seg_id.to_string(),
        table.to_string(),
        Vec::new(), // 列定义稍后填充
    );

    // 写入元数据
    crate::metadata::rocksdb::put_segment_meta(&db.db, &meta)?;

    // 初始化删除掩码文件
    let del_path = layout.del_mask_path();
    std::fs::write(&del_path, vec![0u8; 1024])?;

    // 创建初始 BloomFilter（仅存内存，不持久化）
    let bf = create_initial_bloom_filter(db.config.bloom_filter_fpp);

    // 缓存到内存
    let mut bfs = db.segment_bloom_filters.write();
    bfs.insert(seg_id.to_string(), bf);

    Ok(())
}

/// 创建初始 BloomFilter
fn create_initial_bloom_filter(fpp: f64) -> BloomFilter {
    create_bloom_filter(1_000_000, fpp)
}

/// 创建 Bloom Filter（用于写入时注册主键）
pub fn create_bloom_filter(expected_items: usize, fpp: f64) -> BloomFilter {
    let config = BloomConfig::new(expected_items, fpp);
    let (size, hashes) = config.parameters();
    BloomFilter::with_mode(size, hashes, BloomMode::Blocked)
}

/// 将 pk 插入到 segment 的 BloomFilter 中
fn insert_pk_into_bloom_filter(db: &RockDuck, seg_id: &str, pk: &[u8]) -> Result<()> {
    let mut bfs = db.segment_bloom_filters.write();

    let bf = bfs
        .get_mut(seg_id)
        .ok_or_else(|| RockDuckError::Storage(format!("BloomFilter not found for segment {}", seg_id)))?;

    bloom_insert(bf, pk);

    Ok(())
}

/// 确保 segment 的列定义已填充
fn ensure_segment_columns(db: &RockDuck, seg_id: &str, batch: &RecordBatch) -> Result<()> {
    let mut meta = crate::metadata::rocksdb::get_segment_meta(&db.db, seg_id)?
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
        crate::metadata::rocksdb::put_segment_meta(&db.db, &meta)?;
    }
    Ok(())
}

/// 写单行到 Vortex
#[allow(dead_code)]
fn write_row_to_vortex(
    db: &RockDuck,
    seg_id: &str,
    batch: &RecordBatch,
    row_idx: usize,
) -> Result<()> {
    let layout = SegmentLayout::new(&db.data_dir, seg_id);

    for col_idx in 0..batch.num_columns() {
        let col_name = batch.schema().field(col_idx).name().to_string();
        let array = batch.column(col_idx);
        let sliced = array.slice(row_idx, 1);
        append_to_column(&layout, &col_name, sliced)?;
    }

    Ok(())
}

/// Write a multi-column RecordBatch to segment column files.
/// Each column is written as a separate Arrow IPC file containing all rows.
fn write_segment_batch(
    data_dir: &std::path::Path,
    seg_id: &str,
    schema: &arrow_schema::SchemaRef,
    col_arrays: &[Vec<ArrayRef>],
) -> Result<()> {
    let layout = SegmentLayout::new(data_dir, seg_id);

    for (col_idx, arrays) in col_arrays.iter().enumerate() {
        let col_name = schema.field(col_idx).name().clone();
        write_column_data(&layout, &col_name, arrays)?;
    }

    Ok(())
}

/// Write multiple rows to a column file. If the file already exists, reads existing data,
/// concatenates, and writes back (avoids Arrow IPC per-batch file corruption).
fn write_column_data(
    layout: &SegmentLayout,
    col_name: &str,
    arrays: &[ArrayRef],
) -> Result<()> {
    let col_path = layout.col_path(col_name);

    if let Some(parent) = col_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    if arrays.is_empty() {
        return Ok(());
    }
    let refs: Vec<&dyn arrow_array::Array> = arrays.iter().map(|a| a.as_ref()).collect();
    let new_data = arrow::compute::concat(refs.as_slice())
        .map_err(|e| RockDuckError::Storage(format!("Failed to concat column '{}': {}", col_name, e)))?;

    let combined = if col_path.exists() {
        let existing_batch = read_arrow_file_internal(&col_path)?;
        let existing_col = existing_batch.column(0);
        let refs2: Vec<&dyn arrow_array::Array> = vec![existing_col.as_ref(), new_data.as_ref()];
        arrow::compute::concat(refs2.as_slice())
            .map_err(|e| RockDuckError::Storage(format!("Failed to concat with existing data for '{}': {}", col_name, e)))?
    } else {
        new_data
    };

    let schema = arrow_schema::Schema::new(vec![
        arrow_schema::Field::new("value", combined.data_type().clone(), true)
    ]);
    let file = std::fs::File::create(&col_path)?;
    let mut writer = arrow_ipc::writer::FileWriter::try_new(
        std::io::BufWriter::new(file),
        &schema,
    )?;
    let batch = RecordBatch::try_new(
        std::sync::Arc::new(schema),
        vec![combined],
    )?;
    writer.write(&batch)?;
    writer.finish()?;

    Ok(())
}

/// Read Arrow IPC file and return the first RecordBatch (column 0 only)
fn read_arrow_file_internal(path: &std::path::Path) -> Result<RecordBatch> {
    let file = std::fs::File::open(path)?;
    let mut reader = arrow_ipc::reader::FileReader::try_new(
        std::io::BufReader::new(file),
        None,
    )?;
    reader.next()
        .ok_or_else(|| RockDuckError::Storage("Empty file".to_string()))?
        .map_err(|e| RockDuckError::Storage(format!("Failed to read batch: {}", e)))
}

/// 追加数据到列文件
fn append_to_column(
    layout: &SegmentLayout,
    col_name: &str,
    array_ref: ArrayRef,
) -> Result<()> {
    let col_path = layout.col_path(col_name);

    if !col_path.exists() {
        std::fs::create_dir_all(col_path.parent().unwrap())?;
    }

    let file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&col_path)?;

    let schema = arrow_schema::Schema::new(vec![
        arrow_schema::Field::new("value", array_ref.data_type().clone(), true)
    ]);
    let mut writer = arrow_ipc::writer::FileWriter::try_new(
        std::io::BufWriter::new(file),
        &schema,
    )?;

    let batch = RecordBatch::try_new(
        std::sync::Arc::new(schema),
        vec![array_ref],
    )?;

    writer.write(&batch)?;
    writer.finish()?;

    Ok(())
}

/// 更新表统计
fn update_table_stats(db: &RockDuck, table: &str, added_rows: u64, deleted_rows: u64, segment_increment: u32) -> Result<()> {
    let mut stats = get_or_create_table_stats(&db.db, table)?;
    stats.add_rows(added_rows);
    stats.add_deleted(deleted_rows);
    for _ in 0..segment_increment {
        stats.add_segment();
    }
    put_table_stats(&db.db, &stats)?;
    Ok(())
}

/// 异步批量插入（使用 lazyflow 进行流水线处理）
/// 这个函数展示如何使用 lazyflow 进行异步批处理
#[cfg(feature = "async")]
pub async fn insert_batch_async(
    db: &RockDuck,
    table: &str,
    pks: Vec<Vec<u8>>,
    columns: HashMap<String, ArrayRef>,
) -> Result<u64> {
    use std::time::Duration;

    let txn_id = db.next_txn_id();

    if pks.is_empty() {
        return Ok(0);
    }

    // 创建 RecordBatch
    let batch = columns_to_batch(&columns)?;

    // 使用 lazyflow 进行并行处理
    let pks_owned: Vec<Vec<u8>> = pks;
    let layout = SegmentLayout::new(&db.data_dir, "");
    let mut rocksdb_batch = WriteBatch::default();

    // 使用 lazyflow 的管道式并行处理
    pipe(pks_owned.into_iter().enumerate())
        .chunks_timeout(1000, Duration::from_millis(50))
        .par_eval_map(8, |(idx, pk): (usize, Vec<u8>)| async move {
            let entry = allocate_position_sync(db, table, &layout, txn_id)?;
            write_row_to_vortex_sync(db, &entry.seg_id, &batch, idx)?;
            Ok((pk, entry))
        })
        .for_each(|result: Result<(Vec<u8>, IndexEntry), RockDuckError>| async move {
            if let Ok((pk, entry)) = result {
                let key = pk_index_key(table, &pk);
                if let Ok(value) = encode(&entry) {
                    rocksdb_batch.put(&key, &value);
                }
            }
        })
        .await;

    // 提交事务
    let mut opts = WriteOptions::default();
    opts.set_sync(true);
    db.db.write_opt(rocksdb_batch, &opts)?;

    // 更新表统计
    update_table_stats(db, table, pks.len() as u64, 0, 0)?;

    Ok(txn_id)
}

/// 同步版本的 allocate_position
#[allow(dead_code)]
fn allocate_position_sync(db: &RockDuck, table: &str, _layout: &SegmentLayout, txn_id: u64) -> Result<IndexEntry> {
    let active_seg = find_active_segment(db, table)?;

    let seg_id = match active_seg {
        Some(seg_id) => seg_id,
        None => {
            let new_seg_id = generate_seg_id();
            create_new_segment(db, table, &new_seg_id, &db.data_dir)?;
            new_seg_id
        }
    };

    let mut meta = crate::metadata::rocksdb::get_segment_meta(&db.db, &seg_id)?
        .ok_or_else(|| RockDuckError::SegmentNotFound(seg_id.clone()))?;

    let granule_id = meta.granules.len() as u32;
    let offset = meta.row_count as u32;

    meta.row_count += 1;
    meta.updated_at = crate::codec::current_timestamp_secs();

    crate::metadata::rocksdb::put_segment_meta(&db.db, &meta)?;

    Ok(IndexEntry::new(seg_id, granule_id, offset, txn_id))
}

/// 同步版本的 write_row_to_vortex
#[allow(dead_code)]
fn write_row_to_vortex_sync(
    db: &RockDuck,
    seg_id: &str,
    batch: &RecordBatch,
    row_idx: usize,
) -> Result<()> {
    let layout = SegmentLayout::new(&db.data_dir, seg_id);

    for col_idx in 0..batch.num_columns() {
        let col_name = batch.schema().field(col_idx).name().to_string();
        let array = batch.column(col_idx);
        let sliced = array.slice(row_idx, 1);
        append_to_column(&layout, &col_name, sliced)?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- columns_to_batch ----

    #[test]
    fn test_columns_to_batch() {
        let mut columns = HashMap::new();
        columns.insert("id".to_string(), std::sync::Arc::new(arrow_array::Int64Array::from(vec![1i64, 2, 3])) as ArrayRef);
        columns.insert("name".to_string(), std::sync::Arc::new(arrow_array::StringArray::from(vec!["a", "b", "c"])) as ArrayRef);

        let batch = columns_to_batch(&columns).unwrap();
        assert_eq!(batch.num_rows(), 3);
        assert_eq!(batch.num_columns(), 2);
    }

    #[test]
    fn test_columns_to_batch_empty() {
        let columns: std::collections::HashMap<String, ArrayRef> = std::collections::HashMap::new();
        let result = columns_to_batch(&columns);
        assert!(result.is_err());
    }

    #[test]
    fn test_columns_to_batch_inconsistent_lengths() {
        let mut columns = std::collections::HashMap::new();
        columns.insert("a".to_string(), std::sync::Arc::new(arrow_array::Int64Array::from(vec![1i64, 2, 3])) as ArrayRef);
        columns.insert("b".to_string(), std::sync::Arc::new(arrow_array::Int64Array::from(vec![4i64, 5])) as ArrayRef);
        let result = columns_to_batch(&columns);
        assert!(result.is_err());
    }

    // ---- insert_batch edge cases ----

    #[test]
    fn test_insert_batch_empty_pks() {
        use tempfile::tempdir;
        use crate::RockDuck;

        let temp_dir = tempdir().unwrap();
        let db = RockDuck::open(temp_dir.path()).unwrap();

        let pks: Vec<Vec<u8>> = vec![];
        let data: std::collections::HashMap<String, ArrayRef> = std::collections::HashMap::new();

        let txn_id = db.insert_batch("users", &pks, &data).unwrap();
        assert_eq!(txn_id, 0);
    }

    #[test]
    fn test_insert_batch_row_count_mismatch() {
        use tempfile::tempdir;
        use crate::RockDuck;

        let temp_dir = tempdir().unwrap();
        let db = RockDuck::open(temp_dir.path()).unwrap();

        // 3 PKs but only 2 rows of data
        let pks = vec![b"pk1".to_vec(), b"pk2".to_vec(), b"pk3".to_vec()];
        let mut data = std::collections::HashMap::new();
        data.insert("id".to_string(), std::sync::Arc::new(arrow_array::Int64Array::from(vec![1i64, 2])) as ArrayRef);

        let result = db.insert_batch("users", &pks, &data);
        assert!(result.is_err());
    }
}
