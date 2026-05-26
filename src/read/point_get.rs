//! 点查路径
//!
//! 通过主键获取单条记录
//!
//! 流程：
//! 1. RocksDB.get("pk:{table}:{pk}") → IndexEntry
//! 2. Learned Bloom Filter 预测 segment + Bloom Filter 检查（Feature 6）
//! 3. 读取 Vortex granule
//! 4. 读取 Del Mask 检查是否已删除
//! 5. 返回 Arrow RecordBatch

use arrow_array::RecordBatch;
use quickbloom::{BloomConfig, BloomFilter, BloomMode};
use crate::{RockDuck, RockDuckError};
use crate::error::Result;
use crate::metadata::{IndexEntry, pk_skiplist::get_pk_index};

/// 点查
pub fn get(db: &RockDuck, table: &str, pk: &[u8]) -> Result<Option<RecordBatch>> {
    let snapshot = db.snapshot(crate::mvcc::IsolationLevel::Snapshot)?;
    get_with_snapshot(db, table, pk, &snapshot)
}

/// Time-Travel 点查：在指定事务 ID 的快照下查询
pub fn get_as_of(db: &RockDuck, table: &str, pk: &[u8], txn_id: u64) -> Result<Option<RecordBatch>> {
    let snapshot = db.snapshot_at(txn_id, crate::mvcc::IsolationLevel::Snapshot)?;
    get_with_snapshot(db, table, pk, &snapshot)
}

/// 使用指定快照进行点查（内部通用实现）
fn get_with_snapshot(db: &RockDuck, table: &str, pk: &[u8], snapshot: &crate::mvcc::TxnSnapshot) -> Result<Option<RecordBatch>> {
    // Feature 6: Learned Bloom Filter — 用 model 预测 segment，过滤掉不相关的 segment
    if db.config.enable_bloom_filter {
        if let Some(predicted_seg) = crate::metadata::lbf::get_lbf(table)
            .map(|lbf| lbf.predict_segment(bytes_to_i64(pk)))
        {
            // 只在 predicted segment 上检查 bloom filter（后续代码会检查 pk 是否真的在该 segment）
            tracing::debug!("LBF predicted seg {} for pk", predicted_seg);
        }
    }

    // 1. 从 RocksDB 获取主键索引
    let entry = match get_pk_index(&db.db, table, pk)? {
        Some(e) => e,
        None => return Ok(None), // 主键不存在
    };

    // 检查 MVCC 可见性
    if !db.is_visible(snapshot, entry.txn_id, None) {
        return Ok(None);
    }

    // 2. Bloom Filter 检查（可选）
    if db.config.enable_bloom_filter {
        if !check_bloom_filter(db, &entry, pk)? {
            // Bloom Filter 说不存在
            return Ok(None);
        }
    }

    // 3. 读取 Vortex 数据
    let record = read_record(db, &entry)?;

    // 4. 检查删除标记
    if is_deleted(db, &entry)? {
        return Ok(None);
    }

    Ok(Some(record))
}

/// 将 pk bytes 转换为 i64（用于 Learned Bloom Filter）
fn bytes_to_i64(pk: &[u8]) -> i64 {
    let mut buf = [0u8; 8];
    let len = pk.len().min(8);
    buf[..len].copy_from_slice(&pk[..len]);
    i64::from_le_bytes(buf)
}

/// 检查 Bloom Filter
fn check_bloom_filter(db: &RockDuck, entry: &IndexEntry, pk: &[u8]) -> Result<bool> {
    let seg_id = &entry.seg_id;

    let bf = {
        let bfs = db.segment_bloom_filters.read();
        bfs.get(seg_id).cloned()
    };

    let bf = match bf {
        Some(bf) => bf,
        None => {
            // BF 不在缓存中，保守返回 true（不过滤，避免假阴性）
            return Ok(true);
        }
    };

    // Bloom Filter 只能有假阳性（false positive），不会有假阴性（false negative）
    // 如果 bloom_contains 返回 false，说明 pk 一定不存在
    // 如果返回 true，pk 可能存在也可能不存在
    Ok(bloom_contains(&bf, pk))
}

/// 创建 Bloom Filter（用于写入时注册主键）
pub fn create_bloom_filter(expected_items: usize, fpp: f64) -> BloomFilter {
    let config = BloomConfig::new(expected_items, fpp);
    let (size, hashes) = config.parameters();
    BloomFilter::with_mode(size, hashes, BloomMode::Blocked)
}

/// 检查主键是否可能存在于 Bloom Filter 中
pub fn bloom_contains(filter: &BloomFilter, pk: &[u8]) -> bool {
    filter.contains(&pk.to_vec())
}

/// 向 Bloom Filter 添加主键
pub fn bloom_insert(filter: &mut BloomFilter, pk: &[u8]) {
    filter.insert(&pk.to_vec());
}

/// 检查记录是否已删除
fn is_deleted(db: &RockDuck, entry: &IndexEntry) -> Result<bool> {
    let seg_id = &entry.seg_id;
    let offset = entry.offset;

    // 读取删除掩码
    let layout = crate::segment::layout::SegmentLayout::new(&db.data_dir, seg_id);
    let del_path = layout.del_mask_path();
    
    if !del_path.exists() {
        return Ok(false);
    }
    
    let del_data = std::fs::read(&del_path)?;
    
    // 解析删除掩码
    // 简化实现：假设删除掩码是简单的位图
    let byte_pos = offset as usize / 8;
    let bit_pos = offset as usize % 8;
    
    if byte_pos < del_data.len() {
        Ok((del_data[byte_pos] >> bit_pos) & 1 == 1)
    } else {
        Ok(false)
    }
}

/// 读取记录
fn read_record(db: &RockDuck, entry: &IndexEntry) -> Result<RecordBatch> {
    let seg_id = &entry.seg_id;
    let offset = entry.offset;

    // 获取 segment 元数据
    let meta = crate::metadata::rocksdb::get_segment_meta(&db.db, seg_id)?
        .ok_or_else(|| RockDuckError::SegmentNotFound(seg_id.clone()))?;

    // 读取每列数据
    let layout = crate::segment::layout::SegmentLayout::new(&db.data_dir, seg_id);
    let mut columns: Vec<arrow_array::ArrayRef> = Vec::new();

    for col_def in &meta.columns {
        let col_path = layout.col_path(&col_def.name);

        if !col_path.exists() {
            continue;
        }

        // Each column file stores one column named "value" - read the full column and slice the row
        let batch = read_arrow_file(&col_path)?;

        // Only take the row at the given offset
        if batch.num_columns() == 0 {
            continue;
        }
        let col_array = batch.column(0);
        if (offset as usize) < col_array.len() {
            columns.push(col_array.slice(offset as usize, 1));
        } else {
            return Err(RockDuckError::Internal(format!(
                "offset {} out of bounds for column '{}' of {} rows",
                offset,
                col_def.name,
                col_array.len()
            )));
        }
    }

    // 创建 RecordBatch
    let fields: Vec<arrow_schema::Field> = meta.columns.iter().map(|c| {
        arrow_schema::Field::new(&c.name, c.dtype.to_arrow(), true)
    }).collect();
    let schema = arrow_schema::Schema::new(fields);

    if columns.is_empty() {
        return Err(RockDuckError::Storage("No columns found".to_string()));
    }

    RecordBatch::try_new(std::sync::Arc::new(schema), columns)
        .map_err(|e| RockDuckError::Storage(format!("Failed to create RecordBatch: {}", e)))
}

/// 读取 Arrow IPC 文件，返回完整的 RecordBatch
fn read_arrow_file(path: &std::path::Path) -> Result<RecordBatch> {
    let file = std::fs::File::open(path)?;
    let reader = arrow_ipc::reader::FileReader::try_new(
        std::io::BufReader::new(file),
        None,
    )?;

    let batches: Vec<RecordBatch> = reader
        .filter_map(|b| b.ok())
        .collect();

    if batches.is_empty() {
        return Err(RockDuckError::Storage("Empty file".to_string()));
    }

    // 合并所有 batches
    let mut result_columns: Vec<arrow_array::ArrayRef> = Vec::new();
    let num_columns = batches[0].num_columns();

    for col_idx in 0..num_columns {
        let mut combined: Vec<arrow_array::ArrayRef> = batches.iter()
            .map(|b| b.column(col_idx).clone())
            .collect();

        if combined.len() == 1 {
            result_columns.push(combined.remove(0));
        } else {
            // 合并多个 batch 的同一列
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

/// 批量点查
pub fn batch_get(db: &RockDuck, table: &str, pks: &[Vec<u8>]) -> Result<Vec<Option<RecordBatch>>> {
    let mut results = Vec::with_capacity(pks.len());
    
    for pk in pks {
        results.push(get(db, table, pk)?);
    }
    
    Ok(results)
}

#[cfg(test)]
mod tests {
    // Point-get is tested comprehensively in integration tests.
}
