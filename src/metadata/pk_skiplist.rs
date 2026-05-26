//! 主键 Skiplist 索引操作（有序范围查询）
//!
//! 与 pk_index.rs 的 Hash 索引互补：
//! - pk_index.rs (pk_idx CF)：O(1) 点查，通过 Hash 定位
//! - pk_skiplist.rs (pk_skiplist CF)：有序遍历，支持范围查询
//!
//! 两者共用同一个 RocksDB 实例，使用不同的 Column Family。

use rocksdb::{DB, WriteBatch};
use crate::error::Result;
use crate::metadata::{IndexEntry, rocksdb::pk_index_key};
use crate::codec::{encode, decode};

/// Skiplist CF 键前缀（与 pk_idx 的 "pk:" 前缀区分）
pub const SKIPLIST_KEY_PREFIX: &[u8] = b"sk:";

/// 构建 Skiplist 索引 key：sk:{table}:{pk}
pub fn pk_skiplist_key(table: &str, pk: &[u8]) -> Vec<u8> {
    let mut key = Vec::with_capacity(table.len() + pk.len() + SKIPLIST_KEY_PREFIX.len() + 1);
    key.extend_from_slice(SKIPLIST_KEY_PREFIX);
    key.extend_from_slice(table.as_bytes());
    key.push(b':');
    key.extend_from_slice(pk);
    key
}

/// 从 skiplist key 中提取 pk 字节
pub fn extract_pk_from_skiplist_key(key: &[u8], table_prefix: &[u8]) -> Option<Vec<u8>> {
    // key = sk:{table}:{pk}
    // table_prefix = sk:{table}:
    if !key.starts_with(table_prefix) {
        return None;
    }
    let pk_start = table_prefix.len();
    Some(key[pk_start..].to_vec())
}

/// 获取 skiplist CF 前缀（用于范围扫描时的前缀匹配）
pub fn skiplist_prefix(table: &str) -> Vec<u8> {
    let mut prefix = Vec::with_capacity(SKIPLIST_KEY_PREFIX.len() + table.len() + 1);
    prefix.extend_from_slice(SKIPLIST_KEY_PREFIX);
    prefix.extend_from_slice(table.as_bytes());
    prefix.push(b':');
    prefix
}

// ============================================================
// Read operations
// ============================================================

/// 点查：使用 skiplist CF 做精确匹配
/// 注意：大多数点查应优先使用 pk_idx (hash 索引)，此接口用于 skiplist CF 的回退或验证
pub fn get_pk_index(db: &DB, table: &str, pk: &[u8]) -> Result<Option<IndexEntry>> {
    let cf = db.cf_handle("pk_skiplist")
        .ok_or_else(|| crate::RockDuckError::Metadata("pk_skiplist column family not found".to_string()))?;

    let key = pk_skiplist_key(table, pk);

    match db.get_cf(&cf, &key)? {
        Some(value) => {
            let entry: IndexEntry = decode(&value)?;
            Ok(Some(entry))
        }
        None => Ok(None),
    }
}

/// 范围扫描：Skiplist CF 的核心操作
/// RocksDB 的 skiplist memtable 和 SST 文件中的 skiplist 索引天然按 key 排序，
/// 因此 raw_iterator 可以高效地做有序遍历。
pub fn scan_pk_range(
    db: &DB,
    table: &str,
    start_pk: Option<&[u8]>,
    end_pk: Option<&[u8]>,
) -> Result<Vec<(Vec<u8>, IndexEntry)>> {
    let cf = db.cf_handle("pk_skiplist")
        .ok_or_else(|| crate::RockDuckError::Metadata("pk_skiplist column family not found".to_string()))?;

    let prefix = skiplist_prefix(table);
    let start_key = match start_pk {
        Some(pk) => {
            let mut key = prefix.clone();
            key.extend_from_slice(pk);
            key
        }
        None => prefix.clone(),
    };

    let end_key = end_pk.map(|pk| {
        let mut key = prefix.clone();
        key.extend_from_slice(pk);
        key
    });

    let mut results = Vec::new();

    let mut iter = db.raw_iterator_cf(&cf);
    iter.seek(&start_key);

    while iter.valid() {
        if let Some(key) = iter.key() {
            if !key.starts_with(&prefix) {
                break;
            }

            if let Some(ref end) = end_key {
                if key >= end.as_slice() {
                    break;
                }
            }

            if let Some(value) = iter.value() {
                if let Ok(entry) = decode::<IndexEntry>(value) {
                    let pk = key[prefix.len()..].to_vec();
                    results.push((pk, entry));
                }
            }
        } else {
            break;
        }

        iter.next();
    }

    Ok(results)
}

/// 批量范围扫描（用于多 segment 并行读取前的 pk 收集）
pub fn batch_scan_pk(
    db: &DB,
    table: &str,
    pk_range: Option<(Vec<u8>, Vec<u8>)>,
) -> Result<Vec<IndexEntry>> {
    match pk_range {
        Some((start, end)) => {
            let results = scan_pk_range(db, table, Some(start.as_slice()), Some(end.as_slice()))?;
            Ok(results.into_iter().map(|(_, entry)| entry).collect())
        }
        None => {
            let results = scan_pk_range(db, table, None, None)?;
            Ok(results.into_iter().map(|(_, entry)| entry).collect())
        }
    }
}

// ============================================================
// Write operations
// ============================================================

/// 双写：同时写入 hash 索引 (pk_idx CF) 和 skiplist 索引 (pk_skiplist CF)
/// 用于新插入记录时保持双索引一致性。
pub fn put_pk_index_double(
    db: &DB,
    table: &str,
    pk: &[u8],
    entry: &IndexEntry,
) -> Result<()> {
    let hash_cf = db.cf_handle("pk_idx")
        .ok_or_else(|| crate::RockDuckError::Metadata("pk_idx column family not found".to_string()))?;
    let skiplist_cf = db.cf_handle("pk_skiplist")
        .ok_or_else(|| crate::RockDuckError::Metadata("pk_skiplist column family not found".to_string()))?;

    let key_hash = pk_index_key(table, pk);
    let key_skip = pk_skiplist_key(table, pk);
    let value = encode(entry)?;

    let mut batch = WriteBatch::default();
    batch.put_cf(&hash_cf, &key_hash, &value);
    batch.put_cf(&skiplist_cf, &key_skip, &value);

    db.write(batch)?;
    Ok(())
}

/// 双写（WriteBatch variant）：将双写操作加入已有的 WriteBatch
/// 用于批量插入时减少事务数量。
pub fn put_pk_index_double_into_batch(
    batch: &mut WriteBatch,
    db: &DB,
    table: &str,
    pk: &[u8],
    entry: &IndexEntry,
) -> Result<()> {
    let hash_cf = db.cf_handle("pk_idx")
        .ok_or_else(|| crate::RockDuckError::Metadata("pk_idx column family not found".to_string()))?;
    let skiplist_cf = db.cf_handle("pk_skiplist")
        .ok_or_else(|| crate::RockDuckError::Metadata("pk_skiplist column family not found".to_string()))?;

    let key_hash = pk_index_key(table, pk);
    let key_skip = pk_skiplist_key(table, pk);
    let value = encode(entry)?;

    batch.put_cf(&hash_cf, &key_hash, &value);
    batch.put_cf(&skiplist_cf, &key_skip, &value);
    Ok(())
}

/// 双删：同时从 hash 索引和 skiplist 索引中删除主键
pub fn delete_pk_index_double(
    db: &DB,
    table: &str,
    pk: &[u8],
) -> Result<()> {
    let hash_cf = db.cf_handle("pk_idx")
        .ok_or_else(|| crate::RockDuckError::Metadata("pk_idx column family not found".to_string()))?;
    let skiplist_cf = db.cf_handle("pk_skiplist")
        .ok_or_else(|| crate::RockDuckError::Metadata("pk_skiplist column family not found".to_string()))?;

    let key_hash = pk_index_key(table, pk);
    let key_skip = pk_skiplist_key(table, pk);

    let mut batch = WriteBatch::default();
    batch.delete_cf(&hash_cf, &key_hash);
    batch.delete_cf(&skiplist_cf, &key_skip);

    db.write(batch)?;
    Ok(())
}

/// 双删（WriteBatch variant）
pub fn delete_pk_index_double_into_batch(
    batch: &mut WriteBatch,
    db: &DB,
    table: &str,
    pk: &[u8],
) -> Result<()> {
    let hash_cf = db.cf_handle("pk_idx")
        .ok_or_else(|| crate::RockDuckError::Metadata("pk_idx column family not found".to_string()))?;
    let skiplist_cf = db.cf_handle("pk_skiplist")
        .ok_or_else(|| crate::RockDuckError::Metadata("pk_skiplist column family not found".to_string()))?;

    let key_hash = pk_index_key(table, pk);
    let key_skip = pk_skiplist_key(table, pk);

    batch.delete_cf(&hash_cf, &key_hash);
    batch.delete_cf(&skiplist_cf, &key_skip);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_skiplist_key_format() {
        let key = pk_skiplist_key("users", b"user_123");
        // sk:users:user_123
        assert_eq!(key, b"sk:users:user_123");
    }

    #[test]
    fn test_skiplist_key_empty_pk() {
        let key = pk_skiplist_key("t", b"");
        assert_eq!(key, b"sk:t:");
    }

    #[test]
    fn test_skiplist_prefix() {
        let prefix = skiplist_prefix("orders");
        assert_eq!(prefix, b"sk:orders:");
    }

    #[test]
    fn test_extract_pk_from_skiplist_key() {
        let key = b"sk:users:user_123" as &[u8];
        let prefix = b"sk:users:" as &[u8];
        let pk = extract_pk_from_skiplist_key(key, prefix);
        assert_eq!(pk, Some(b"user_123".to_vec()));
    }

    #[test]
    fn test_extract_pk_from_skiplist_key_no_match() {
        let key = b"pk:users:user_123" as &[u8];
        let prefix = b"sk:users:" as &[u8];
        let pk = extract_pk_from_skiplist_key(key, prefix);
        assert_eq!(pk, None);
    }

    #[test]
    fn test_extract_pk_from_skiplist_key_special_chars() {
        let key = b"sk:users:user:123" as &[u8];
        let prefix = b"sk:users:" as &[u8];
        let pk = extract_pk_from_skiplist_key(key, prefix);
        // 包含冒号在内的完整 pk
        assert_eq!(pk, Some(b"user:123".to_vec()));
    }

    // ============================================================
    // Integration tests with real RocksDB
    // ============================================================

    fn make_test_db_with_skiplist() -> rocksdb::DB {
        use tempfile::TempDir;
        use rocksdb::{Options, DB};

        let temp = TempDir::new().unwrap();
        let mut opts = Options::default();
        opts.create_if_missing(true);
        opts.create_missing_column_families(true);

        // 确保 pk_idx 和 pk_skiplist 两个 CF 都存在
        DB::open_cf(&opts, temp.path(), &["pk_idx", "pk_skiplist"]).unwrap()
    }

    #[test]
    fn test_double_write_and_read() {
        let db = make_test_db_with_skiplist();

        let table = "users";
        let pk = b"user_001";
        let entry = IndexEntry::new("seg_abc".to_string(), 3, 42, 100);

        put_pk_index_double(&db, table, pk, &entry).unwrap();

        // Hash 索引点查
        let hash_result = {
            let cf = db.cf_handle("pk_idx").unwrap();
            let key = pk_index_key(table, pk);
            let value = db.get_cf(&cf, &key).unwrap().unwrap();
            decode::<IndexEntry>(&value).unwrap()
        };

        // Skiplist 索引点查
        let skiplist_result = get_pk_index(&db, table, pk).unwrap().unwrap();

        assert_eq!(hash_result.seg_id, "seg_abc");
        assert_eq!(skiplist_result.seg_id, "seg_abc");
        assert_eq!(hash_result.offset, 42);
        assert_eq!(skiplist_result.offset, 42);
    }

    // ============================================================
    // Byte-level double-write consistency: critical business rule
    // ============================================================

    #[test]
    fn test_double_write_consistency() {
        let db = make_test_db_with_skiplist();
        let table = "consistency_test";
        let pk = b"user_bytecheck";
        let entry = IndexEntry::new("seg_final".to_string(), 7, 123, 999);

        put_pk_index_double(&db, table, pk, &entry).unwrap();

        // Read raw bytes from both CFs
        let hash_cf = db.cf_handle("pk_idx").unwrap();
        let skiplist_cf = db.cf_handle("pk_skiplist").unwrap();
        let hash_key = pk_index_key(table, pk);
        let skip_key = pk_skiplist_key(table, pk);

        let hash_bytes = db.get_cf(&hash_cf, &hash_key)
            .unwrap()
            .expect("Hash CF entry must exist");
        let skip_bytes = db.get_cf(&skiplist_cf, &skip_key)
            .unwrap()
            .expect("Skiplist CF entry must exist");

        // Byte-level consistency assertion
        assert_eq!(
            hash_bytes, skip_bytes,
            "Hash and skiplist entries must be byte-for-byte identical for key {:?}",
            pk
        );
        assert!(!hash_bytes.is_empty(), "Encoded bytes must not be empty");
    }

    #[test]
    fn test_double_write_and_delete() {
        let db = make_test_db_with_skiplist();

        let table = "orders";
        let pk = b"order_999";
        let entry = IndexEntry::new("seg_xyz".to_string(), 1, 7, 200);

        put_pk_index_double(&db, table, pk, &entry).unwrap();

        // 确认写入成功
        assert!(get_pk_index(&db, table, pk).unwrap().is_some());

        // 双删
        delete_pk_index_double(&db, table, pk).unwrap();

        // 两个索引都应该为空
        assert!(get_pk_index(&db, table, pk).unwrap().is_none());
        let hash_cf = db.cf_handle("pk_idx").unwrap();
        let key = pk_index_key(table, pk);
        assert!(db.get_cf(&hash_cf, &key).unwrap().is_none());
    }

    #[test]
    fn test_skiplist_range_scan() {
        let db = make_test_db_with_skiplist();

        let table = "events";
        // 写入有序的 pk
        for i in 0u32..10 {
            let pk = format!("event_{:05}", i);
            let entry = IndexEntry::new(
                format!("seg_{}", i),
                0,
                i,
                i as u64,
            );
            put_pk_index_double(&db, table, pk.as_bytes(), &entry).unwrap();
        }

        // 范围查询 [event_00003, event_00007)
        let results = scan_pk_range(
            &db,
            table,
            Some(b"event_00003"),
            Some(b"event_00007"),
        ).unwrap();

        assert_eq!(results.len(), 4);
        // 结果应该按键有序
        let pks: Vec<_> = results.iter().map(|(pk, _)| pk.clone()).collect();
        assert_eq!(pks[0], b"event_00003".to_vec());
        assert_eq!(pks[3], b"event_00006".to_vec());
    }

    #[test]
    fn test_skiplist_range_scan_no_bounds() {
        let db = make_test_db_with_skiplist();

        let table = "test";
        for i in 0u32..5 {
            let pk = format!("k{:02}", i);
            let entry = IndexEntry::new("seg".to_string(), 0, i, i as u64);
            put_pk_index_double(&db, table, pk.as_bytes(), &entry).unwrap();
        }

        // 全量扫描
        let results = scan_pk_range(&db, table, None, None).unwrap();
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn test_skiplist_range_scan_only_start() {
        let db = make_test_db_with_skiplist();

        let table = "items";
        for i in 0u32..20 {
            let pk = format!("item_{:03}", i);
            let entry = IndexEntry::new("seg".to_string(), 0, i, i as u64);
            put_pk_index_double(&db, table, pk.as_bytes(), &entry).unwrap();
        }

        // 只设置起始边界
        let results = scan_pk_range(
            &db,
            table,
            Some(b"item_010"),
            None,
        ).unwrap();
        assert_eq!(results.len(), 10);
        assert_eq!(results[0].0, b"item_010".to_vec());
    }

    #[test]
    fn test_skiplist_range_scan_only_end() {
        let db = make_test_db_with_skiplist();

        let table = "rows";
        for i in 0u32..15 {
            let pk = format!("row_{:03}", i);
            let entry = IndexEntry::new("seg".to_string(), 0, i, i as u64);
            put_pk_index_double(&db, table, pk.as_bytes(), &entry).unwrap();
        }

        // 只设置结束边界
        let results = scan_pk_range(
            &db,
            table,
            None,
            Some(b"row_005"),
        ).unwrap();
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn test_double_write_batch_variant() {
        let db = make_test_db_with_skiplist();

        let table = "batch_test";
        let mut batch = WriteBatch::default();

        for i in 0u32..5 {
            let pk = format!("pk_{}", i);
            let entry = IndexEntry::new(format!("seg_{}", i), 0, i, i as u64);
            put_pk_index_double_into_batch(&mut batch, &db, table, pk.as_bytes(), &entry).unwrap();
        }

        db.write(batch).unwrap();

        let results = scan_pk_range(&db, table, None, None).unwrap();
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn test_double_delete_batch_variant() {
        let db = make_test_db_with_skiplist();

        let table = "del_test";
        let pk = b"to_delete";
        let entry = IndexEntry::new("seg_1".to_string(), 0, 1, 1);
        put_pk_index_double(&db, table, pk, &entry).unwrap();

        // 确认存在
        assert!(get_pk_index(&db, table, pk).unwrap().is_some());

        // 使用 batch variant 删除
        let mut batch = WriteBatch::default();
        delete_pk_index_double_into_batch(&mut batch, &db, table, pk).unwrap();
        db.write(batch).unwrap();

        assert!(get_pk_index(&db, table, pk).unwrap().is_none());
    }

    #[test]
    fn test_batch_scan_pk() {
        let db = make_test_db_with_skiplist();

        let table = "batch_scan";
        for i in 0u32..8 {
            let pk = format!("key_{:02}", i);
            let entry = IndexEntry::new("seg".to_string(), 0, i, i as u64);
            put_pk_index_double(&db, table, pk.as_bytes(), &entry).unwrap();
        }

        // 部分范围
        let entries = batch_scan_pk(&db, table, Some((b"key_02".to_vec(), b"key_06".to_vec()))).unwrap();
        assert_eq!(entries.len(), 4);

        // 全量
        let all = batch_scan_pk(&db, table, None).unwrap();
        assert_eq!(all.len(), 8);
    }

    #[test]
    fn test_skiplist_scan_empty_result() {
        let db = make_test_db_with_skiplist();

        let results = scan_pk_range(&db, "empty_table", None, None).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_skiplist_key_unicode_pk() {
        let key = pk_skiplist_key("t", "用户名".as_bytes());
        // sk:t:用户名
        assert_eq!(key, b"sk:t:\xe7\x94\xa8\xe6\x88\xb7\xe5\x90\x8d");
    }

    #[test]
    fn test_batch_scan_pk_empty_range() {
        let db = make_test_db_with_skiplist();

        let table = "range_test";
        for i in 0u32..5 {
            let pk = format!("key_{:02}", i);
            let entry = IndexEntry::new("seg".to_string(), 0, i, i as u64);
            put_pk_index_double(&db, table, pk.as_bytes(), &entry).unwrap();
        }

        // 范围不存在数据
        let entries = batch_scan_pk(&db, table, Some((b"z999".to_vec(), b"z999".to_vec()))).unwrap();
        assert!(entries.is_empty());
    }
}
