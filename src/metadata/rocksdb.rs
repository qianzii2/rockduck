//! RocksDB 初始化和元数据管理
//! 
//! 实现 7 个 Column Family：
//! 1. pk_idx: 主键索引 pk → {seg_id, granule_id, offset}
//! 2. seg_meta: Segment 元数据
//! 3. stat: 表统计信息
//! 4. zone: Zone Map 数据
//! 5. layer: Immutable layer
//! 6. lbf: Learned Bloom Filter（预留）
//! 7. bf: Per-granule Bloom Filter

use std::ops::Deref;
use std::path::Path;
use std::sync::Arc;
use rocksdb::{DB, ColumnFamily, Options, WriteBatch};
use crate::config::RockDuckConfig;
use crate::error::{RockDuckError, Result};
use crate::metadata::{IndexEntry, TableStats, SegmentMeta};
use crate::codec::{encode, decode};

/// CF 名称常量
pub const CF_PK_IDX: &str = "pk_idx";
pub const CF_SEG_META: &str = "seg_meta";
pub const CF_STAT: &str = "stat";
pub const CF_ZONE: &str = "zone";
pub const CF_LAYER: &str = "layer";
pub const CF_LBF: &str = "lbf";
pub const CF_BF: &str = "bf";
pub const CF_PROJ_META: &str = "proj_meta";
pub const CF_MVCC: &str = "mvcc"; // MVCC active transactions
pub const CF_PK_SKIPLIST: &str = "pk_skiplist"; // Skiplist PK index (sorted range scans)
pub const CF_SYS: &str = "sys"; // System metadata (committed_txn, etc.)
pub const CF_ICEBERG: &str = "iceberg_manifest"; // Iceberg native manifest storage

/// 所有 CF 名称列表
pub fn all_cf_names() -> Vec<&'static str> {
    vec![
        CF_PK_IDX,
        CF_PK_SKIPLIST,
        CF_SEG_META,
        CF_STAT,
        CF_ZONE,
        CF_LAYER,
        CF_LBF,
        CF_BF,
        CF_PROJ_META,
        CF_MVCC,
        CF_SYS,
        CF_ICEBERG,
    ]
}

/// Iceberg manifest column family name.
pub fn cf_iceberg() -> &'static str {
    CF_ICEBERG
}

/// RocksDB 默认配置
fn default_options() -> Options {
    let mut opts = Options::default();
    opts.create_if_missing(true);
    opts.create_missing_column_families(true);
    opts.set_max_background_jobs(4);
    opts.set_write_buffer_size(64 * 1024 * 1024); // 64MB
    opts.set_max_write_buffer_number(3);
    opts.set_min_write_buffer_number_to_merge(2);
    opts
}

/// 初始化 RocksDB（包含所有 Column Families）
pub fn init_rocksdb(data_dir: &Path, _config: &RockDuckConfig) -> Result<DB> {
    let rocksdb_path = data_dir.join("meta");

    if !rocksdb_path.exists() {
        std::fs::create_dir_all(&rocksdb_path)?;
    }

    let mut db_opts = default_options();
    db_opts.set_compression_type(rocksdb::DBCompressionType::Lz4);

    let cf_names = all_cf_names();

    // Try to open with column families first (for new DBs)
    match DB::open_cf(&db_opts, &rocksdb_path, &cf_names) {
        Ok(db) => Ok(db),
        Err(_) => {
            // For existing DBs: try opening without specifying CFs, then create missing ones
            match DB::open(&db_opts, &rocksdb_path) {
                Ok(mut db) => {
                    // Create any missing column families
                    for cf_name in &cf_names {
                        if db.cf_handle(cf_name).is_none() {
                            // Use a default options override for the new CF
                            let cf_opts = default_options();
                            if let Err(e) = db.create_cf(cf_name, &cf_opts) {
                                return Err(RockDuckError::Metadata(format!(
                                    "Failed to create column family '{}': {}", cf_name, e
                                )));
                            }
                            eprintln!("Created missing column family: {}", cf_name);
                        }
                    }
                    Ok(db)
                }
                Err(e) => Err(e.into()),
            }
        }
    }
}

/// 获取主键索引 CF
pub fn pk_idx_cf(db: &DB) -> Result<&ColumnFamily> {
    db.cf_handle(CF_PK_IDX)
        .ok_or_else(|| RockDuckError::Metadata(format!("Column family {} not found", CF_PK_IDX)))
}

/// 获取主键 Skiplist 索引 CF
pub fn pk_skiplist_cf(db: &DB) -> Result<&ColumnFamily> {
    db.cf_handle(CF_PK_SKIPLIST)
        .ok_or_else(|| RockDuckError::Metadata(format!("Column family {} not found", CF_PK_SKIPLIST)))
}

/// 获取 Segment 元数据 CF
pub fn seg_meta_cf(db: &DB) -> Result<&ColumnFamily> {
    db.cf_handle(CF_SEG_META)
        .ok_or_else(|| RockDuckError::Metadata(format!("Column family {} not found", CF_SEG_META)))
}

/// 获取表统计 CF
pub fn stat_cf(db: &DB) -> Result<&ColumnFamily> {
    db.cf_handle(CF_STAT)
        .ok_or_else(|| RockDuckError::Metadata(format!("Column family {} not found", CF_STAT)))
}

/// 获取 Zone Map CF
pub fn zone_cf(db: &DB) -> Result<&ColumnFamily> {
    db.cf_handle(CF_ZONE)
        .ok_or_else(|| RockDuckError::Metadata(format!("Column family {} not found", CF_ZONE)))
}

// ================== 主键索引操作 ==================

/// 构建主键索引 key
pub fn pk_index_key(table: &str, pk: &[u8]) -> Vec<u8> {
    let mut key = Vec::with_capacity(table.len() + pk.len() + 3);
    key.extend_from_slice(b"pk:");
    key.extend_from_slice(table.as_bytes());
    key.push(b':');
    key.extend_from_slice(pk);
    key
}

/// 写入主键索引
pub fn put_pk_index(db: &DB, table: &str, pk: &[u8], entry: &IndexEntry) -> Result<()> {
    let cf = pk_idx_cf(db)?;
    let key = pk_index_key(table, pk);
    let value = encode(entry)?;
    db.put_cf(&cf, &key, &value)?;
    Ok(())
}

/// 读取主键索引
pub fn get_pk_index(db: &DB, table: &str, pk: &[u8]) -> Result<Option<IndexEntry>> {
    let cf = pk_idx_cf(db)?;
    let key = pk_index_key(table, pk);
    
    match db.get_cf(&cf, &key)? {
        Some(value) => {
            let entry = decode::<IndexEntry>(&value)?;
            Ok(Some(entry))
        }
        None => Ok(None),
    }
}

/// 删除主键索引
pub fn delete_pk_index(db: &DB, table: &str, pk: &[u8]) -> Result<()> {
    let cf = pk_idx_cf(db)?;
    let key = pk_index_key(table, pk);
    db.delete_cf(&cf, &key)?;
    Ok(())
}

/// 批量写入主键索引
pub fn batch_put_pk_index(
    db: &DB,
    writes: &[(String, Vec<u8>, IndexEntry)]
) -> Result<()> {
    let cf = pk_idx_cf(db)?;
    let mut batch = WriteBatch::default();

    for (table, pk, entry) in writes {
        let key = pk_index_key(table, pk);
        let value = encode(entry)?;
        batch.put_cf(&cf, &key, &value);
    }

    db.write(batch)?;
    Ok(())
}

// ================== Segment 元数据操作 ==================

/// 构建 segment 元数据 key
pub fn seg_meta_key(seg_id: &str) -> Vec<u8> {
    let mut key = Vec::with_capacity(seg_id.len() + 4);
    key.extend_from_slice(b"seg:");
    key.extend_from_slice(seg_id.as_bytes());
    key
}

/// 写入 segment 元数据
pub fn put_segment_meta(db: &DB, meta: &SegmentMeta) -> Result<()> {
    let cf = seg_meta_cf(db)?;
    let key = seg_meta_key(&meta.seg_id);
    let value = encode(meta)?;
    db.put_cf(&cf, &key, &value)?;
    Ok(())
}

/// 读取 segment 元数据
pub fn get_segment_meta(db: &DB, seg_id: &str) -> Result<Option<SegmentMeta>> {
    let cf = seg_meta_cf(db)?;
    let key = seg_meta_key(seg_id);

    match db.get_cf(&cf, &key)? {
        Some(value) => {
            let meta: SegmentMeta = decode(&value)?;
            Ok(Some(meta))
        }
        None => Ok(None),
    }
}

/// 删除 segment 元数据
pub fn delete_segment_meta(db: &DB, seg_id: &str) -> Result<()> {
    let cf = seg_meta_cf(db)?;
    let key = seg_meta_key(seg_id);
    db.delete_cf(&cf, &key)?;
    Ok(())
}

/// 列出所有 segments
pub fn list_segments(db: &DB) -> Result<Vec<String>> {
    let cf = seg_meta_cf(db)?;
    let mut segments = Vec::new();
    let prefix = b"seg:".to_vec();
    
    let mut iter = db.raw_iterator_cf(&cf);
    iter.seek(&prefix);
    
    while iter.valid() {
        if let Some(key) = iter.key() {
            if key.starts_with(&prefix) {
                if let Ok(seg_id) = std::str::from_utf8(&key[4..]) {
                    segments.push(seg_id.to_string());
                }
            }
        }
        iter.next();
    }
    
    Ok(segments)
}

// ================== 表统计操作 ==================

/// 构建表统计 key
pub fn table_stat_key(table: &str) -> Vec<u8> {
    let mut key = Vec::with_capacity(table.len() + 5);
    key.extend_from_slice(b"stat:");
    key.extend_from_slice(table.as_bytes());
    key
}

/// 写入表统计
pub fn put_table_stats(db: &DB, stats: &TableStats) -> Result<()> {
    let cf = stat_cf(db)?;
    let key = table_stat_key(&stats.table);
    let value = encode(stats)?;
    db.put_cf(&cf, &key, &value)?;
    Ok(())
}

/// 读取表统计
pub fn get_table_stats(db: &DB, table: &str) -> Result<Option<TableStats>> {
    let cf = stat_cf(db)?;
    let key = table_stat_key(table);
    
    match db.get_cf(&cf, &key)? {
        Some(value) => {
            let stats: TableStats = decode(&value)?;
            Ok(Some(stats))
        }
        None => Ok(None),
    }
}

/// 获取或创建表统计
pub fn get_or_create_table_stats(db: &DB, table: &str) -> Result<TableStats> {
    Ok(get_table_stats(db, table)?
        .unwrap_or_else(|| TableStats::new(table.to_string())))
}

// ================== Bloom Filter 操作 ==================

/// 获取 granule bloom filter
pub fn get_granule_bf(db: &DB, seg_id: &str, granule_id: u32) -> Result<Option<Vec<u8>>> {
    let cf = db.cf_handle(CF_BF)
        .ok_or_else(|| RockDuckError::Metadata(format!("Column family {} not found", CF_BF)))?;
    
    let key = format!("{}:{}", seg_id, granule_id);
    match db.get_cf(&cf, key.as_bytes())? {
        Some(value) => Ok(Some(value)),
        None => Ok(None),
    }
}

/// 写入 granule bloom filter
pub fn put_granule_bf(db: &DB, seg_id: &str, granule_id: u32, bf_data: &[u8]) -> Result<()> {
    let cf = db.cf_handle(CF_BF)
        .ok_or_else(|| RockDuckError::Metadata(format!("Column family {} not found", CF_BF)))?;
    
    let key = format!("{}:{}", seg_id, granule_id);
    db.put_cf(&cf, key.as_bytes(), bf_data)?;
    Ok(())
}

// ================== MVCC Active Transactions ==================

/// MVCC 活跃事务 Key前缀
const MVCC_ACTIVE_PREFIX: &[u8] = b"active:";

/// 添加活跃事务（txn_id → begin_timestamp）
pub fn add_active_txn(db: &Arc<DB>, txn_id: u64, begin_ts: u64) -> Result<()> {
    let cf = mvcc_cf(db)?;
    let key = format!("active:{}", txn_id);
    let value = encode(&begin_ts)?;
    db.deref().put_cf(&cf, key.as_bytes(), &value)?;
    Ok(())
}

/// 移除活跃事务
pub fn remove_active_txn(db: &Arc<DB>, txn_id: u64) -> Result<()> {
    let cf = mvcc_cf(db)?;
    let key = format!("active:{}", txn_id);
    db.deref().delete_cf(&cf, key.as_bytes())?;
    Ok(())
}

/// 获取所有活跃事务 ID
pub fn get_active_txns(db: &Arc<DB>) -> Result<Vec<u64>> {
    let cf = mvcc_cf(db)?;
    let prefix = MVCC_ACTIVE_PREFIX.to_vec();
    let mut txns = Vec::new();

    let mut iter = db.deref().raw_iterator_cf(&cf);
    iter.seek(&prefix);

    while iter.valid() {
        if let Some(key) = iter.key() {
            if let Some(_value) = iter.value() {
                if key.starts_with(&prefix) {
                    let txn_id_str = std::str::from_utf8(&key[prefix.len()..]).ok();
                    if let Some(s) = txn_id_str {
                        if let Ok(txn_id) = s.parse::<u64>() {
                            txns.push(txn_id);
                        }
                    }
                } else {
                    break;
                }
            }
        }
        iter.next();
    }

    Ok(txns)
}

/// 获取活跃事务的开始时间戳
pub fn get_active_txn_begin(db: &Arc<DB>, txn_id: u64) -> Result<Option<u64>> {
    let cf = mvcc_cf(db)?;
    let key = format!("active:{}", txn_id);
    match db.deref().get_cf(&cf, key.as_bytes())? {
        Some(value) => {
            let ts: u64 = decode(&value)?;
            Ok(Some(ts))
        }
        None => Ok(None),
    }
}

fn mvcc_cf(db: &Arc<DB>) -> Result<&ColumnFamily> {
    db.cf_handle(CF_MVCC)
        .ok_or_else(|| RockDuckError::Metadata(format!("Column family {} not found", CF_MVCC)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pk_index_key() {
        let key = pk_index_key("users", b"123");
        assert_eq!(key, b"pk:users:123");
    }

    #[test]
    fn test_seg_meta_key() {
        let key = seg_meta_key("seg_001");
        assert_eq!(key, b"seg:seg_001");
    }

    #[test]
    fn test_table_stat_key() {
        let key = table_stat_key("users");
        assert_eq!(key, b"stat:users");
    }

    // ============================================================
    // Key building edge cases
    // ============================================================

    #[test]
    fn test_pk_index_key_empty_pk() {
        let key = pk_index_key("table", b"");
        assert_eq!(key, b"pk:table:");
    }

    #[test]
    fn test_pk_index_key_special_chars_in_pk() {
        let key = pk_index_key("users", b"key:with:colons");
        assert_eq!(key, b"pk:users:key:with:colons");
    }

    #[test]
    fn test_pk_index_key_unicode() {
        let key = pk_index_key("users", "用户名".as_bytes());
        // pk:users:用户名
        // UTF-8: pk: = 0x70 0x6b 0x3a, users = 0x75 0x73 0x65 0x72 0x73, : = 0x3a
        // 用户名 = E7 94 A8 E6 88 B7 E5 90 8D
        assert_eq!(key, b"pk:users:\xe7\x94\xa8\xe6\x88\xb7\xe5\x90\x8d");
    }

    #[test]
    fn test_seg_meta_key_empty() {
        let key = seg_meta_key("");
        assert_eq!(key, b"seg:");
    }

    #[test]
    fn test_table_stat_key_empty() {
        let key = table_stat_key("");
        assert_eq!(key, b"stat:");
    }

    // ============================================================
    // all_cf_names
    // ============================================================

    #[test]
    fn test_all_cf_names_contains_all() {
        let names = all_cf_names();
        assert_eq!(names.len(), 12);
        assert!(names.contains(&CF_PK_IDX));
        assert!(names.contains(&CF_ICEBERG));
    }

    #[test]
    fn test_cf_name_constants() {
        assert_eq!(CF_PK_IDX, "pk_idx");
        assert_eq!(CF_PK_SKIPLIST, "pk_skiplist");
        assert_eq!(CF_SEG_META, "seg_meta");
        assert_eq!(CF_STAT, "stat");
        assert_eq!(CF_ZONE, "zone");
        assert_eq!(CF_LAYER, "layer");
        assert_eq!(CF_LBF, "lbf");
        assert_eq!(CF_BF, "bf");
        assert_eq!(CF_PROJ_META, "proj_meta");
        assert_eq!(CF_MVCC, "mvcc");
    }
}
