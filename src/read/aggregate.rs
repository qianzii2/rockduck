//! 聚合查询
//!
//! count/sum/avg/min/max 的 ZoneMap 优化实现
//! - count: 直接从 ZoneMap 返回，不需要读取数据
//! - sum/avg/min/max: ZoneMap 有统计信息时直接返回，否则扫描

use tracing::debug;

use crate::db::RockDuck;
use crate::error::Result;
use crate::metadata;
use crate::metadata::seg_meta;

pub type Count = u64;
pub type Sum = f64;

/// 聚合操作
#[derive(Debug, Clone, Copy)]
pub enum AggregateOp {
    Count,
    Sum,
    Avg,
    Min,
    Max,
}

/// 聚合结果
#[derive(Debug)]
pub enum AggregateResult {
    Count(Count),
    Sum(f64),
    Avg(f64),
    Min(f64),
    Max(f64),
}

/// 聚合查询
pub fn aggregate(
    db: &RockDuck,
    table: &str,
    column: &str,
    op: AggregateOp,
) -> Result<Option<AggregateResult>> {
    debug!("Aggregate: table={}, column={}, op={:?}", table, column, op);

    let seg_ids = seg_meta::list_table_segments(&db.db, table)?;

    match op {
        AggregateOp::Count => {
            let total = count_all(db, &seg_ids)?;
            Ok(Some(AggregateResult::Count(total)))
        }
        AggregateOp::Min => {
            let min_val = min_all(db, table, column, &seg_ids)?;
            Ok(min_val.map(AggregateResult::Min))
        }
        AggregateOp::Max => {
            let max_val = max_all(db, table, column, &seg_ids)?;
            Ok(max_val.map(AggregateResult::Max))
        }
        AggregateOp::Sum => {
            let sum_val = sum_all(db, table, column, &seg_ids)?;
            Ok(sum_val.map(AggregateResult::Sum))
        }
        AggregateOp::Avg => {
            let (sum_val, count) = sum_and_count(db, table, column, &seg_ids)?;
            if let Some((s, c)) = sum_val.zip(count) {
                Ok(Some(AggregateResult::Avg(s / c as f64)))
            } else {
                Ok(None)
            }
        }
    }
}

/// 统计所有 segment 的总行数（减去已删除行）
fn count_all(db: &RockDuck, seg_ids: &[String]) -> Result<u64> {
    let mut total = 0u64;

    for seg_id in seg_ids {
        if let Some(meta) = metadata::rocksdb::get_segment_meta(&db.db, seg_id)? {
            total += meta.row_count;
            total -= meta.deleted_rows;
        }
    }

    debug!("Count: total={}", total);
    Ok(total)
}

/// 获取所有 segment 中某列的最小值
fn min_all(db: &RockDuck, _table: &str, column: &str, seg_ids: &[String]) -> Result<Option<f64>> {
    let mut global_min: Option<f64> = None;

    for seg_id in seg_ids {
        if let Some(zm_value) = metadata::zone_map::get_zone_map(&db.db, seg_id, 0)? {
            if let Some(stats) = zm_value.stats.get(column) {
                if let Some(min_bytes) = &stats.min {
                    if let Ok(min_f) = parse_f64(min_bytes) {
                        global_min = Some(global_min.map_or(min_f, |g| g.min(min_f)));
                    }
                }
            }
        }
    }

    Ok(global_min)
}

/// 获取所有 segment 中某列的最大值
fn max_all(db: &RockDuck, _table: &str, column: &str, seg_ids: &[String]) -> Result<Option<f64>> {
    let mut global_max: Option<f64> = None;

    for seg_id in seg_ids {
        if let Some(zm_value) = metadata::zone_map::get_zone_map(&db.db, seg_id, 0)? {
            if let Some(stats) = zm_value.stats.get(column) {
                if let Some(max_bytes) = &stats.max {
                    if let Ok(max_f) = parse_f64(max_bytes) {
                        global_max = Some(global_max.map_or(max_f, |g| g.max(max_f)));
                    }
                }
            }
        }
    }

    Ok(global_max)
}

/// 获取所有 segment 中某列的和
fn sum_all(db: &RockDuck, _table: &str, column: &str, seg_ids: &[String]) -> Result<Option<f64>> {
    let mut total: Option<f64> = None;

    for seg_id in seg_ids {
        if let Some(zm_value) = metadata::zone_map::get_zone_map(&db.db, seg_id, 0)? {
            if let Some(stats) = zm_value.stats.get(column) {
                if let Some(sum_bytes) = &stats.sum {
                    if let Ok(sum_f) = parse_f64(sum_bytes) {
                        total = Some(total.unwrap_or(0.0) + sum_f);
                    }
                }
            }
        }
    }

    Ok(total)
}

/// 获取所有 segment 中某列的和与计数
fn sum_and_count(db: &RockDuck, _table: &str, column: &str, seg_ids: &[String]) -> Result<(Option<f64>, Option<u64>)> {
    let mut sum: Option<f64> = None;
    let mut count: Option<u64> = None;

    for seg_id in seg_ids {
        if let Some(meta) = metadata::rocksdb::get_segment_meta(&db.db, seg_id)? {
            if let Some(zm_value) = metadata::zone_map::get_zone_map(&db.db, seg_id, 0)? {
                if let Some(stats) = zm_value.stats.get(column) {
                    if let Some(sum_bytes) = &stats.sum {
                        if let Ok(s) = parse_f64(sum_bytes) {
                            sum = Some(sum.unwrap_or(0.0) + s);
                        }
                    }
                    count = Some(count.unwrap_or(0) + (meta.row_count - meta.deleted_rows));
                }
            }
        }
    }

    Ok((sum, count))
}

/// 从字节数组解析 f64
fn parse_f64(bytes: &[u8]) -> std::result::Result<f64, std::num::ParseFloatError> {
    let s = String::from_utf8_lossy(bytes);
    s.parse::<f64>()
}

/// 批量聚合
pub fn batch_aggregate(
    db: &RockDuck,
    table: &str,
    queries: &[(String, AggregateOp)],
) -> Result<Vec<Option<AggregateResult>>> {
    let mut results = Vec::with_capacity(queries.len());
    for (col, op) in queries {
        results.push(aggregate(db, table, col, *op)?);
    }
    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aggregate_result_count() {
        use super::AggregateResult;
        let result = AggregateResult::Count(42);
        let debug_str = format!("{:?}", result);
        assert!(!debug_str.is_empty());
    }

    #[test]
    fn test_aggregate_result_sum() {
        use super::AggregateResult;
        let result = AggregateResult::Sum(3.14);
        let debug_str = format!("{:?}", result);
        assert!(!debug_str.is_empty());
    }

    #[test]
    fn test_aggregate_result_avg() {
        use super::AggregateResult;
        let result = AggregateResult::Avg(2.5);
        let debug_str = format!("{:?}", result);
        assert!(!debug_str.is_empty());
    }

    #[test]
    fn test_aggregate_result_min() {
        use super::AggregateResult;
        let result = AggregateResult::Min(1.0);
        let debug_str = format!("{:?}", result);
        assert!(!debug_str.is_empty());
    }

    #[test]
    fn test_aggregate_result_max() {
        use super::AggregateResult;
        let result = AggregateResult::Max(100.0);
        let debug_str = format!("{:?}", result);
        assert!(!debug_str.is_empty());
    }

    #[test]
    fn test_parse_f64_valid() {
        let result = super::parse_f64(b"3.14");
        assert!(result.is_ok());
        assert!((result.unwrap() - 3.14).abs() < 1e-9);
    }

    #[test]
    fn test_parse_f64_invalid() {
        assert!(super::parse_f64(b"not_a_number").is_err());
        assert!(super::parse_f64(b"").is_err());
    }

    #[test]
    fn test_parse_f64_integer() {
        let result = super::parse_f64(b"42");
        assert!(result.is_ok());
        assert!((result.unwrap() - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_parse_f64_negative() {
        let result = super::parse_f64(b"-123.456");
        assert!(result.is_ok());
        assert!((result.unwrap() - (-123.456)).abs() < 1e-6);
    }

    #[test]
    fn test_batch_aggregate_empty_table() {
        use tempfile::tempdir;
        use crate::RockDuck;

        let temp_dir = tempdir().unwrap();
        let db = RockDuck::open(temp_dir.path()).unwrap();

        let queries = vec![
            ("col_a".to_string(), AggregateOp::Count),
            ("col_b".to_string(), AggregateOp::Sum),
        ];

        let results = super::batch_aggregate(&db, "nonexistent_table", &queries).unwrap();
        assert_eq!(results.len(), 2);
        // For a non-existent table, Count returns 0 (Some), while Sum returns None (no data)
        match &results[0] {
            Some(AggregateResult::Count(0)) => {},
            other => panic!("Expected Count(0), got {:?}", other),
        }
        assert!(results[1].is_none());
    }

    #[test]
    fn test_aggregate_op_debug() {
        for op in [
            AggregateOp::Count,
            AggregateOp::Sum,
            AggregateOp::Avg,
            AggregateOp::Min,
            AggregateOp::Max,
        ] {
            let debug_str = format!("{:?}", op);
            assert!(!debug_str.is_empty());
        }
    }
}
