//! Segment 元数据操作

use rocksdb::DB;
use crate::error::Result;
use crate::metadata::SegmentMeta;
use crate::segment::meta::SegmentStatus;
use crate::codec::{decode, encode};

/// 列出表的所有 segments
pub fn list_table_segments(db: &DB, table: &str) -> Result<Vec<String>> {
    let cf = db.cf_handle("seg_meta")
        .ok_or_else(|| crate::RockDuckError::Metadata("seg_meta column family not found".to_string()))?;
    
    let prefix = format!("seg:");
    let mut segments = Vec::new();
    
    let mut iter = db.raw_iterator_cf(&cf);
    iter.seek(prefix.as_bytes());
    
    while iter.valid() {
        if let Some(_key) = iter.key() {
            if let Some(value) = iter.value() {
                if let Ok(meta) = decode::<SegmentMeta>(value) {
                    if meta.table == table {
                        segments.push(meta.seg_id);
                    }
                }
            }
        }
        iter.next();
    }
    
    segments.sort();
    Ok(segments)
}

/// 获取 segment 元数据
pub fn get_segment(db: &DB, seg_id: &str) -> Result<Option<SegmentMeta>> {
    let cf = db.cf_handle("seg_meta")
        .ok_or_else(|| crate::RockDuckError::Metadata("seg_meta column family not found".to_string()))?;
    
    let key = format!("seg:{}", seg_id);
    
    match db.get_cf(&cf, key.as_bytes())? {
        Some(value) => {
            let meta: SegmentMeta = decode::<SegmentMeta>(&value)?;
            Ok(Some(meta))
        }
        None => Ok(None),
    }
}

/// 获取所有活跃的 segments
pub fn get_active_segments(db: &DB) -> Result<Vec<SegmentMeta>> {
    let cf = db.cf_handle("seg_meta")
        .ok_or_else(|| crate::RockDuckError::Metadata("seg_meta column family not found".to_string()))?;
    
    let mut segments = Vec::new();
    
    let mut iter = db.raw_iterator_cf(&cf);
    iter.seek(b"seg:");
    
    while iter.valid() {
        if let Some(value) = iter.value() {
            if let Ok(meta) = decode::<SegmentMeta>(value) {
                if meta.status == SegmentStatus::Active {
                    segments.push(meta);
                }
            }
        }
        iter.next();
    }
    
    Ok(segments)
}

/// 更新 segment 状态
pub fn update_segment_status(db: &DB, seg_id: &str, status: SegmentStatus) -> Result<()> {
    let cf = db.cf_handle("seg_meta")
        .ok_or_else(|| crate::RockDuckError::Metadata("seg_meta column family not found".to_string()))?;
    
    let key = format!("seg:{}", seg_id);
    
    if let Some(value) = db.get_cf(&cf, key.as_bytes())? {
        let mut meta: SegmentMeta = decode(&value)?;
        meta.status = status;
        meta.updated_at = crate::codec::current_timestamp_secs();
        
        let new_value = encode(&meta)?;
        db.put_cf(&cf, key.as_bytes(), &new_value)?;
    }
    
    Ok(())
}

/// 检查 segment 是否需要 compaction
pub fn needs_compaction(db: &DB, seg_id: &str, threshold: f64) -> Result<bool> {
    if let Some(meta) = get_segment(db, seg_id)? {
        Ok(meta.needs_compaction(threshold))
    } else {
        Ok(false)
    }
}

#[cfg(test)]
mod tests {
    use crate::segment::meta::SegmentStatus;
    use crate::codec::encode;
    use tempfile::tempdir;

    #[test]
    fn test_functions() {}

    #[test]
    fn test_segment_status_variants() {
        assert_eq!(SegmentStatus::Active, SegmentStatus::Active);
        assert_eq!(SegmentStatus::Compactable, SegmentStatus::Compactable);
        assert_eq!(SegmentStatus::Frozen, SegmentStatus::Frozen);
    }

    #[test]
    fn test_list_table_segments_empty() {
        let temp_dir = tempdir().unwrap();
        let db = crate::RockDuck::open(temp_dir.path()).unwrap();
        let segments = super::list_table_segments(&db.db, "users").unwrap();
        assert!(segments.is_empty());
    }

    #[test]
    fn test_list_table_segments_filters_by_table() {
        let temp_dir = tempdir().unwrap();
        let db = crate::RockDuck::open(temp_dir.path()).unwrap();

        let cf = db.db.cf_handle("seg_meta").unwrap();

        let meta_a = crate::metadata::SegmentMeta::new(
            "seg_a".to_string(),
            "table_a".to_string(),
            vec![],
        );
        let meta_b = crate::metadata::SegmentMeta::new(
            "seg_b".to_string(),
            "table_b".to_string(),
            vec![],
        );

        db.db.put_cf(&cf, "seg:seg_a", &encode(&meta_a).unwrap()).unwrap();
        db.db.put_cf(&cf, "seg:seg_b", &encode(&meta_b).unwrap()).unwrap();

        let segs_a = super::list_table_segments(&db.db, "table_a").unwrap();
        let segs_b = super::list_table_segments(&db.db, "table_b").unwrap();

        assert_eq!(segs_a, vec!["seg_a".to_string()]);
        assert_eq!(segs_b, vec!["seg_b".to_string()]);
    }

    #[test]
    fn test_get_segment_roundtrip() {
        let temp_dir = tempdir().unwrap();
        let db = crate::RockDuck::open(temp_dir.path()).unwrap();

        let cf = db.db.cf_handle("seg_meta").unwrap();
        let meta = crate::metadata::SegmentMeta::new(
            "seg_test".to_string(),
            "users".to_string(),
            vec![],
        );
        db.db.put_cf(&cf, "seg:seg_test", &encode(&meta).unwrap()).unwrap();

        let result = super::get_segment(&db.db, "seg_test").unwrap();
        assert!(result.is_some());
    }

    #[test]
    fn test_get_segment_nonexistent() {
        let temp_dir = tempdir().unwrap();
        let db = crate::RockDuck::open(temp_dir.path()).unwrap();

        let result = super::get_segment(&db.db, "nonexistent").unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_get_active_segments() {
        let temp_dir = tempdir().unwrap();
        let db = crate::RockDuck::open(temp_dir.path()).unwrap();

        let cf = db.db.cf_handle("seg_meta").unwrap();

        let active = crate::metadata::SegmentMeta::new(
            "seg_active".to_string(),
            "users".to_string(),
            vec![],
        );
        let mut frozen = crate::metadata::SegmentMeta::new(
            "seg_frozen".to_string(),
            "users".to_string(),
            vec![],
        );
        frozen.status = SegmentStatus::Frozen;

        db.db.put_cf(&cf, "seg:seg_active", &encode(&active).unwrap()).unwrap();
        db.db.put_cf(&cf, "seg:seg_frozen", &encode(&frozen).unwrap()).unwrap();

        let active_segs = super::get_active_segments(&db.db).unwrap();
        assert_eq!(active_segs.len(), 1);
        assert_eq!(active_segs[0].seg_id, "seg_active");
    }

    #[test]
    fn test_update_segment_status() {
        let temp_dir = tempdir().unwrap();
        let db = crate::RockDuck::open(temp_dir.path()).unwrap();

        let cf = db.db.cf_handle("seg_meta").unwrap();
        let original = crate::metadata::SegmentMeta::new(
            "seg_update".to_string(),
            "users".to_string(),
            vec![],
        );
        db.db.put_cf(&cf, "seg:seg_update", &encode(&original).unwrap()).unwrap();

        super::update_segment_status(&db.db, "seg_update", SegmentStatus::Frozen).unwrap();

        let result = super::get_segment(&db.db, "seg_update").unwrap();
        assert!(result.is_some());
        assert_eq!(result.unwrap().status, SegmentStatus::Frozen);
    }

    #[test]
    fn test_update_segment_status_nonexistent() {
        let temp_dir = tempdir().unwrap();
        let db = crate::RockDuck::open(temp_dir.path()).unwrap();

        super::update_segment_status(&db.db, "nonexistent", SegmentStatus::Frozen).unwrap();
    }

    #[test]
    fn test_needs_compaction_true() {
        let temp_dir = tempdir().unwrap();
        let db = crate::RockDuck::open(temp_dir.path()).unwrap();

        let cf = db.db.cf_handle("seg_meta").unwrap();
        let mut meta = crate::metadata::SegmentMeta::new(
            "seg_dirty".to_string(),
            "users".to_string(),
            vec![],
        );
        meta.row_count = 100;
        meta.deleted_rows = 60;
        meta.del_ratio = 0.6;

        db.db.put_cf(&cf, "seg:seg_dirty", &encode(&meta).unwrap()).unwrap();

        let result = super::needs_compaction(&db.db, "seg_dirty", 0.3).unwrap();
        assert!(result);
    }

    #[test]
    fn test_needs_compaction_false() {
        let temp_dir = tempdir().unwrap();
        let db = crate::RockDuck::open(temp_dir.path()).unwrap();

        let cf = db.db.cf_handle("seg_meta").unwrap();
        let meta = crate::metadata::SegmentMeta::new(
            "seg_clean".to_string(),
            "users".to_string(),
            vec![],
        );
        db.db.put_cf(&cf, "seg:seg_clean", &encode(&meta).unwrap()).unwrap();

        let result = super::needs_compaction(&db.db, "seg_clean", 0.3).unwrap();
        assert!(!result);
    }

    #[test]
    fn test_needs_compaction_nonexistent() {
        let temp_dir = tempdir().unwrap();
        let db = crate::RockDuck::open(temp_dir.path()).unwrap();

        let result = super::needs_compaction(&db.db, "nonexistent", 0.3).unwrap();
        assert!(!result);
    }
}
