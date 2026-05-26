//! Zone Map 实现
//! 
//! Granule 级别的统计信息，用于查询裁剪

use rocksdb::DB;
use std::collections::HashMap;
use bincode_next::{Encode, Decode};
use crate::error::Result;
use crate::codec::{encode, decode};
use crate::segment::meta::{ColumnStats, ZoneMapStats, CompareOp};

/// Zone Map key
pub fn zone_map_key(seg_id: &str, granule_id: u32) -> String {
    format!("{}:{}", seg_id, granule_id)
}

/// Zone Map value
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, Default, Encode, Decode)]
pub struct ZoneMapValue {
    /// 列名到统计的映射
    pub stats: HashMap<String, ColumnStats>,
}

/// 写入 Zone Map
pub fn put_zone_map(db: &DB, seg_id: &str, granule_id: u32, zm: &ZoneMapValue) -> Result<()> {
    let cf = db.cf_handle("zone")
        .ok_or_else(|| crate::RockDuckError::Metadata("zone column family not found".to_string()))?;
    
    let key = zone_map_key(seg_id, granule_id);
    let value = encode(zm)?;
    db.put_cf(&cf, key.as_bytes(), &value)?;
    
    Ok(())
}

/// 读取 Zone Map
pub fn get_zone_map(db: &DB, seg_id: &str, granule_id: u32) -> Result<Option<ZoneMapValue>> {
    let cf = db.cf_handle("zone")
        .ok_or_else(|| crate::RockDuckError::Metadata("zone column family not found".to_string()))?;
    
    let key = zone_map_key(seg_id, granule_id);
    
    match db.get_cf(&cf, key.as_bytes())? {
        Some(value) => {
            let zm: ZoneMapValue = decode(&value)?;
            Ok(Some(zm))
        }
        None => Ok(None),
    }
}

/// 检查 Zone Map 是否可以裁剪
pub fn can_prune_by_zone_map(
    db: &DB,
    seg_id: &str,
    granule_id: u32,
    column: &str,
    op: CompareOp,
    value: &[u8],
) -> Result<bool> {
    let Some(zm) = get_zone_map(db, seg_id, granule_id)? else {
        return Ok(false);
    };
    
    if let Some(stats) = zm.stats.get(column) {
        Ok(ZoneMapStats::can_prune_static(stats, &op, value))
    } else {
        Ok(false)
    }
}

impl ZoneMapStats {
    /// 静态版本的 can_prune
    pub fn can_prune_static(stats: &ColumnStats, op: &CompareOp, val: &[u8]) -> bool {
        match op {
            CompareOp::Eq => false, // 不能裁剪
            CompareOp::Ne => false, // 不能裁剪
            CompareOp::Lt => {
                stats.min.as_ref().map_or(false, |min| min.as_slice() >= val.as_ref())
            }
            CompareOp::Le => {
                stats.min.as_ref().map_or(false, |min| min.as_slice() > val.as_ref())
            }
            CompareOp::Gt => {
                stats.max.as_ref().map_or(false, |max| max.as_slice() <= val.as_ref())
            }
            CompareOp::Ge => {
                stats.max.as_ref().map_or(false, |max| max.as_slice() < val.as_ref())
            }
        }
    }
}

/// 批量获取 Zone Map
pub fn batch_get_zone_maps(
    db: &DB,
    keys: &[(String, u32)], // (seg_id, granule_id)
) -> Result<HashMap<String, ZoneMapValue>> {
    let cf = db.cf_handle("zone")
        .ok_or_else(|| crate::RockDuckError::Metadata("zone column family not found".to_string()))?;
    
    let mut results = HashMap::new();
    
    for (seg_id, granule_id) in keys {
        let key = zone_map_key(seg_id, *granule_id);
        
        if let Some(value) = db.get_cf(&cf, key.as_bytes())? {
            if let Ok(zm) = decode::<ZoneMapValue>(&value) {
                results.insert(key, zm);
            }
        }
    }
    
    Ok(results)
}

/// 删除 Zone Map
pub fn delete_zone_map(db: &DB, seg_id: &str, granule_id: u32) -> Result<()> {
    let cf = db.cf_handle("zone")
        .ok_or_else(|| crate::RockDuckError::Metadata("zone column family not found".to_string()))?;
    
    let key = zone_map_key(seg_id, granule_id);
    db.delete_cf(&cf, key.as_bytes())?;
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zone_map_key() {
        let key = zone_map_key("seg_001", 5);
        assert_eq!(key, "seg_001:5");
    }

    #[test]
    fn test_zone_map_key_special_chars() {
        let key = zone_map_key("seg_with-dash_underscore", 999);
        assert_eq!(key, "seg_with-dash_underscore:999");
    }

    #[test]
    fn test_zone_map_key_zero() {
        let key = zone_map_key("seg_001", 0);
        assert_eq!(key, "seg_001:0");
    }

    // ============================================================
    // ZoneMapValue tests
    // ============================================================

    #[test]
    fn test_zone_map_value_default() {
        let value = ZoneMapValue::default();
        assert!(value.stats.is_empty());
    }

    #[test]
    fn test_zone_map_value_with_stats() {
        let mut value = ZoneMapValue::default();
        value.stats.insert(
            "age".to_string(),
            ColumnStats {
                min: Some(vec![10u8]),
                max: Some(vec![100u8]),
                null_count: 5,
                sum: None,
                distinct_count: Some(50),
            },
        );
        assert_eq!(value.stats.len(), 1);
        assert_eq!(value.stats.get("age").unwrap().null_count, 5);
    }

    // ============================================================
    // ZoneMapStats::can_prune_static tests
    // ============================================================

    #[test]
    fn test_can_prune_static_lt_true() {
        let stats = ColumnStats {
            min: Some(vec![50u8]),
            max: Some(vec![100u8]),
            null_count: 0,
            sum: None,
            distinct_count: None,
        };
        assert!(ZoneMapStats::can_prune_static(&stats, &CompareOp::Lt, &[20u8]));
    }

    #[test]
    fn test_can_prune_static_lt_false() {
        let stats = ColumnStats {
            min: Some(vec![50u8]),
            max: Some(vec![100u8]),
            null_count: 0,
            sum: None,
            distinct_count: None,
        };
        assert!(!ZoneMapStats::can_prune_static(&stats, &CompareOp::Lt, &[80u8]));
    }

    #[test]
    fn test_can_prune_static_le_true() {
        let stats = ColumnStats {
            min: Some(vec![60u8]),
            max: Some(vec![100u8]),
            null_count: 0,
            sum: None,
            distinct_count: None,
        };
        assert!(ZoneMapStats::can_prune_static(&stats, &CompareOp::Le, &[40u8]));
    }

    #[test]
    fn test_can_prune_static_le_false() {
        let stats = ColumnStats {
            min: Some(vec![60u8]),
            max: Some(vec![100u8]),
            null_count: 0,
            sum: None,
            distinct_count: None,
        };
        assert!(!ZoneMapStats::can_prune_static(&stats, &CompareOp::Le, &[70u8]));
    }

    #[test]
    fn test_can_prune_static_gt_true() {
        let stats = ColumnStats {
            min: Some(vec![10u8]),
            max: Some(vec![40u8]),
            null_count: 0,
            sum: None,
            distinct_count: None,
        };
        assert!(ZoneMapStats::can_prune_static(&stats, &CompareOp::Gt, &[60u8]));
    }

    #[test]
    fn test_can_prune_static_gt_false() {
        let stats = ColumnStats {
            min: Some(vec![10u8]),
            max: Some(vec![40u8]),
            null_count: 0,
            sum: None,
            distinct_count: None,
        };
        assert!(!ZoneMapStats::can_prune_static(&stats, &CompareOp::Gt, &[30u8]));
    }

    #[test]
    fn test_can_prune_static_ge_true() {
        let stats = ColumnStats {
            min: Some(vec![10u8]),
            max: Some(vec![80u8]),
            null_count: 0,
            sum: None,
            distinct_count: None,
        };
        assert!(ZoneMapStats::can_prune_static(&stats, &CompareOp::Ge, &[90u8]));
    }

    #[test]
    fn test_can_prune_static_ge_false() {
        let stats = ColumnStats {
            min: Some(vec![10u8]),
            max: Some(vec![80u8]),
            null_count: 0,
            sum: None,
            distinct_count: None,
        };
        assert!(!ZoneMapStats::can_prune_static(&stats, &CompareOp::Ge, &[70u8]));
    }

    #[test]
    fn test_can_prune_static_eq_always_false() {
        let stats = ColumnStats {
            min: Some(vec![10u8]),
            max: Some(vec![100u8]),
            null_count: 0,
            sum: None,
            distinct_count: None,
        };
        // Eq never prunes
        assert!(!ZoneMapStats::can_prune_static(&stats, &CompareOp::Eq, &[50u8]));
        assert!(!ZoneMapStats::can_prune_static(&stats, &CompareOp::Eq, &[5u8]));
    }

    #[test]
    fn test_can_prune_static_ne_always_false() {
        let stats = ColumnStats {
            min: Some(vec![10u8]),
            max: Some(vec![100u8]),
            null_count: 0,
            sum: None,
            distinct_count: None,
        };
        assert!(!ZoneMapStats::can_prune_static(&stats, &CompareOp::Ne, &[50u8]));
    }

    #[test]
    fn test_can_prune_static_none_min() {
        let stats = ColumnStats {
            min: None,
            max: Some(vec![100u8]),
            null_count: 5,
            sum: None,
            distinct_count: None,
        };
        // No min -> cannot prune Lt
        assert!(!ZoneMapStats::can_prune_static(&stats, &CompareOp::Lt, &[50u8]));
    }

    #[test]
    fn test_can_prune_static_none_max() {
        let stats = ColumnStats {
            min: Some(vec![10u8]),
            max: None,
            null_count: 5,
            sum: None,
            distinct_count: None,
        };
        // No max -> cannot prune Gt
        assert!(!ZoneMapStats::can_prune_static(&stats, &CompareOp::Gt, &[50u8]));
    }
}
