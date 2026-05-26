//! 主键索引操作

use rocksdb::DB;
use crate::error::Result;
use crate::metadata::{IndexEntry, rocksdb::pk_index_key};
use crate::codec::decode;

/// 在主键索引中查找记录
pub fn lookup_pk(db: &DB, table: &str, pk: &[u8]) -> Result<Option<IndexEntry>> {
    let cf = db.cf_handle("pk_idx")
        .ok_or_else(|| crate::RockDuckError::Metadata("pk_idx column family not found".to_string()))?;
    
    let key = pk_index_key(table, pk);
    
    match db.get_cf(&cf, &key)? {
        Some(value) => {
            let entry: IndexEntry = decode(&value)?;
            Ok(Some(entry))
        }
        None => Ok(None),
    }
}

/// 批量查找主键
pub fn batch_lookup_pk<'a>(
    db: &DB,
    table: &str,
    pks: impl IntoIterator<Item = &'a [u8]>,
) -> Result<Vec<Option<IndexEntry>>> {
    let cf = db.cf_handle("pk_idx")
        .ok_or_else(|| crate::RockDuckError::Metadata("pk_idx column family not found".to_string()))?;
    
    let mut results = Vec::new();
    
    for pk in pks {
        let key = pk_index_key(table, pk);
        let value = db.get_cf(&cf, &key)?;
        
        let entry = match value {
            Some(v) => Some(decode::<IndexEntry>(&v)?),
            None => None,
        };
        results.push(entry);
    }
    
    Ok(results)
}

/// 范围扫描主键索引
pub fn scan_pk_range(
    db: &DB,
    table: &str,
    start_pk: Option<&[u8]>,
    end_pk: Option<&[u8]>,
) -> Result<Vec<(Vec<u8>, IndexEntry)>> {
    let cf = db.cf_handle("pk_idx")
        .ok_or_else(|| crate::RockDuckError::Metadata("pk_idx column family not found".to_string()))?;
    
    let prefix = format!("pk:{}:", table);
    let prefix_bytes = prefix.as_bytes().to_vec();
    
    let start_key = match start_pk {
        Some(pk) => {
            let mut key = prefix_bytes.clone();
            key.extend_from_slice(pk);
            key
        }
        None => prefix_bytes.clone(),
    };
    
    let end_key = end_pk.map(|pk| {
        let mut key = prefix_bytes.clone();
        key.extend_from_slice(pk);
        key
    });
    
    let mut results = Vec::new();
    
    let mut iter = db.raw_iterator_cf(&cf);
    iter.seek(&start_key);
    
    while iter.valid() {
        if let Some(key) = iter.key() {
            if !key.starts_with(&prefix_bytes) {
                break;
            }
            
            if let Some(ref end) = end_key {
                if key >= end.as_slice() {
                    break;
                }
            }
            
            if let Some(value) = iter.value() {
                if let Ok(entry) = decode::<IndexEntry>(value) {
                    let pk = key[prefix_bytes.len()..].to_vec();
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

#[cfg(test)]
mod tests {
    use crate::metadata::IndexEntry;
    use crate::codec::encode;
    use tempfile::tempdir;

    // ============================================================
    // pk_index_key already tested via metadata::rocksdb
    // ============================================================

    #[test]
    fn test_cf_name_constant() {
        assert_eq!(crate::metadata::rocksdb::CF_PK_IDX, "pk_idx");
    }

    #[test]
    fn test_lookup_pk_roundtrip() {
        let temp_dir = tempdir().unwrap();
        let db = crate::RockDuck::open(temp_dir.path()).unwrap();

        let table = "users";
        let pk = b"user_123";
        let entry = IndexEntry::new("seg_001".to_string(), 0, 5, 100);

        let cf = db.db.cf_handle("pk_idx").unwrap();
        let key = crate::metadata::rocksdb::pk_index_key(table, pk);
        let value = encode(&entry).unwrap();
        db.db.put_cf(&cf, &key, &value).unwrap();

        let result = super::lookup_pk(&db.db, table, pk).unwrap();
        assert!(result.is_some());
        let found = result.unwrap();
        assert_eq!(found.seg_id, "seg_001");
        assert_eq!(found.offset, 5);
    }

    #[test]
    fn test_lookup_pk_not_found() {
        let temp_dir = tempdir().unwrap();
        let db = crate::RockDuck::open(temp_dir.path()).unwrap();

        let result = super::lookup_pk(&db.db, "users", b"nonexistent").unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_batch_lookup_pk_multiple() {
        let temp_dir = tempdir().unwrap();
        let db = crate::RockDuck::open(temp_dir.path()).unwrap();
        let table = "users";

        let cf = db.db.cf_handle("pk_idx").unwrap();
        let pks_vec: Vec<Vec<u8>> = vec![
            b"pk_a".to_vec(),
            b"pk_b".to_vec(),
            b"pk_c".to_vec(),
        ];

        for (i, pk) in pks_vec.iter().enumerate() {
            let entry = IndexEntry::new(format!("seg_{:03}", i), 0, i as u32, i as u64);
            let key = crate::metadata::rocksdb::pk_index_key(table, pk);
            let value = encode(&entry).unwrap();
            db.db.put_cf(&cf, &key, &value).unwrap();
        }

        let results = super::batch_lookup_pk(&db.db, table, pks_vec.iter().map(|p| p.as_slice())).unwrap();
        assert_eq!(results.len(), 3);
        assert!(results[0].is_some());
        assert!(results[1].is_some());
        assert!(results[2].is_some());
    }

    #[test]
    fn test_batch_lookup_pk_mixed_found_not_found() {
        let temp_dir = tempdir().unwrap();
        let db = crate::RockDuck::open(temp_dir.path()).unwrap();
        let table = "users";

        let cf = db.db.cf_handle("pk_idx").unwrap();
        let entry = IndexEntry::new("seg_001".to_string(), 0, 0, 1);
        let key = crate::metadata::rocksdb::pk_index_key(table, b"pk_exists");
        let value = encode(&entry).unwrap();
        db.db.put_cf(&cf, &key, &value).unwrap();

        let results = super::batch_lookup_pk(&db.db, table, [
            b"pk_exists" as &[u8],
            b"pk_missing" as &[u8],
            b"pk_also_miss" as &[u8],
        ].iter().copied()).unwrap();
        assert_eq!(results.len(), 3);
        assert!(results[0].is_some());
        assert!(results[1].is_none());
        assert!(results[2].is_none());
    }

    #[test]
    fn test_scan_pk_range_full() {
        let temp_dir = tempdir().unwrap();
        let db = crate::RockDuck::open(temp_dir.path()).unwrap();
        let table = "users";

        let cf = db.db.cf_handle("pk_idx").unwrap();
        for i in 0..5 {
            let entry = IndexEntry::new(format!("seg_{}", i), 0, i as u32, i as u64);
            let key = crate::metadata::rocksdb::pk_index_key(table, &format!("pk_{:02}", i).into_bytes());
            let value = encode(&entry).unwrap();
            db.db.put_cf(&cf, &key, &value).unwrap();
        }

        let results = super::scan_pk_range(&db.db, table, None, None).unwrap();
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn test_scan_pk_range_with_start() {
        let temp_dir = tempdir().unwrap();
        let db = crate::RockDuck::open(temp_dir.path()).unwrap();
        let table = "users";

        let cf = db.db.cf_handle("pk_idx").unwrap();
        for i in 0..5 {
            let entry = IndexEntry::new(format!("seg_{}", i), 0, i as u32, i as u64);
            let key = crate::metadata::rocksdb::pk_index_key(table, &format!("pk_{:02}", i).into_bytes());
            let value = encode(&entry).unwrap();
            db.db.put_cf(&cf, &key, &value).unwrap();
        }

        let results = super::scan_pk_range(&db.db, table, Some(b"pk_02"), None).unwrap();
        assert!(results.len() >= 3);
    }

    #[test]
    fn test_scan_pk_range_with_end() {
        let temp_dir = tempdir().unwrap();
        let db = crate::RockDuck::open(temp_dir.path()).unwrap();
        let table = "users";

        let cf = db.db.cf_handle("pk_idx").unwrap();
        for i in 0..5 {
            let entry = IndexEntry::new(format!("seg_{}", i), 0, i as u32, i as u64);
            let key = crate::metadata::rocksdb::pk_index_key(table, &format!("pk_{:02}", i).into_bytes());
            let value = encode(&entry).unwrap();
            db.db.put_cf(&cf, &key, &value).unwrap();
        }

        let results = super::scan_pk_range(&db.db, table, None, Some(b"pk_02")).unwrap();
        assert!(results.len() <= 3);
    }

    #[test]
    fn test_scan_pk_range_empty() {
        let temp_dir = tempdir().unwrap();
        let db = crate::RockDuck::open(temp_dir.path()).unwrap();
        let table = "users";

        let results = super::scan_pk_range(&db.db, table, None, None).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_scan_pk_range_bounds() {
        let temp_dir = tempdir().unwrap();
        let db = crate::RockDuck::open(temp_dir.path()).unwrap();
        let table = "users";

        let cf = db.db.cf_handle("pk_idx").unwrap();
        for i in 0..5 {
            let entry = IndexEntry::new(format!("seg_{}", i), 0, i as u32, i as u64);
            let key = crate::metadata::rocksdb::pk_index_key(table, &format!("pk_{:02}", i).into_bytes());
            let value = encode(&entry).unwrap();
            db.db.put_cf(&cf, &key, &value).unwrap();
        }

        let results = super::scan_pk_range(&db.db, table, Some(b"pk_01"), Some(b"pk_03")).unwrap();
        assert!(results.len() <= 3);
    }
}
