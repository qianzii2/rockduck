//! Selective Late Materialization (SLM) — Adaptive Materialization Strategy
//!
//! 策略选择基于 filter 列的选择率（selectivity）：
//!   - selectivity > 90% (Early)：先读所有列再过滤更划算
//!   - selectivity < 10% (Late)：先读 filter 列再按需拉 data 列
//!   - 10% ≤ selectivity ≤ 90% (Hybrid)：分批拉 data 列
//!
//!参照: SLM (Selective Late Materialization), VLDB 2025.
//!   - Integrated into DuckDB: SLM outperforms EM by 14.7% and LM by 8.9% on JOB

use arrow_array::{ArrayRef, RecordBatch, UInt64Array};
use arrow_schema::Schema;

use crate::error::{RockDuckError, Result};
use crate::segment::meta::{CompareOp, ZoneMapStats};

/// Materialization strategy determined by selectivity
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaterializationStrategy {
    /// 先读所有列，再应用 filter（filter 选择率高）
    Early,
    /// 先读 filter 列，按需拉 data 列（filter 选择率低）
    Late,
    /// 分批拉 data 列（中等选择率）
    Hybrid,
}

/// 选择率阈值
#[derive(Debug, Clone)]
pub struct SlmThresholds {
    /// selectivity > early_threshold → Early Materialization
    pub early_threshold: f64,
    /// selectivity < late_threshold → Late Materialization
    pub late_threshold: f64,
}

impl Default for SlmThresholds {
    fn default() -> Self {
        Self {
            early_threshold: 0.9,  // 90% of rows match → read all
            late_threshold: 0.1,    // < 10% of rows match → lazy fetch
        }
    }
}

impl SlmThresholds {
    pub fn new(early: f64, late: f64) -> Self {
        Self {
            early_threshold: early,
            late_threshold: late,
        }
    }
}

/// 根据选择率选择策略
pub fn choose_strategy(selectivity: f64, thresholds: &SlmThresholds) -> MaterializationStrategy {
    if selectivity > thresholds.early_threshold {
        MaterializationStrategy::Early
    } else if selectivity < thresholds.late_threshold {
        MaterializationStrategy::Late
    } else {
        MaterializationStrategy::Hybrid
    }
}

/// 估算 filter 列的选择率（selectivity = matching_rows / total_rows）。
///
/// 方法：
/// 1. 如果有 Zone Map → 用 min/max/null_count 估算
/// 2. 如果 Zone Map 不够用（Eq/Ne 谓词）→ 读一个 sample granule 估算
pub fn estimate_selectivity(
    filter_column: &str,
    predicate: &str,
    zone_map_stats: Option<&ZoneMapStats>,
    #[allow(unused_variables)]
    total_rows: u64,
    #[allow(unused_variables)]
    granule_row_count: u32,
) -> Result<f64> {
    if total_rows == 0 {
        return Ok(0.0);
    }

    // Parse predicate: "op value" where op is one of =, !=, <, <=, >, >=
    let (op, value) = parse_predicate(predicate)?;

    // Try Zone Map estimation for range predicates
    if let Some(zm) = zone_map_stats {
        if let Some(stats) = zm.get(filter_column) {
            if let Some(sel) = estimate_from_stats(stats, &op, value, total_rows) {
                return Ok(sel);
            }
        }
    }

    // Fallback: conservative estimate for Eq/Ne or missing stats
    // For Eq: assume 1 / num_distinct (or 0.01 if unknown)
    match op {
        CompareOp::Eq => {
            let distinct = zone_map_stats
                .and_then(|zm| zm.get(filter_column))
                .and_then(|s| s.distinct_count)
                .unwrap_or(100) as f64;
            Ok((1.0 / distinct).min(1.0))
        }
        CompareOp::Ne => Ok(1.0), // NOT-EQUAL matches almost everything
        _ => Ok(0.5), // Unknown range → conservative middle ground
    }
}

/// 用 Zone Map 统计信息估算选择率
fn estimate_from_stats(
    stats: &crate::segment::meta::ColumnStats,
    op: &CompareOp,
    value: i64,
    #[allow(unused_variables)]
    total_rows: u64,
) -> Option<f64> {
    let min_val = bytes_to_i64(stats.min.as_ref().map(|v| v.as_slice()))?;
    let max_val = bytes_to_i64(stats.max.as_ref().map(|v| v.as_slice()))?;

    let selectivity = match op {
        CompareOp::Lt => {
            if min_val >= value { 0.0 }
            else if max_val < value { 1.0 }
            else { ((value - min_val) as f64) / (max_val - min_val).max(1) as f64 }
        }
        CompareOp::Le => {
            if min_val > value { 0.0 }
            else if max_val <= value { 1.0 }
            else { ((value - min_val + 1) as f64) / (max_val - min_val + 1).max(1) as f64 }
        }
        CompareOp::Gt => {
            if max_val <= value { 0.0 }
            else if min_val > value { 1.0 }
            else { ((max_val - value) as f64) / (max_val - min_val).max(1) as f64 }
        }
        CompareOp::Ge => {
            if min_val >= value { 1.0 }
            else if max_val < value { 0.0 }
            else { ((max_val - value + 1) as f64) / (max_val - min_val + 1).max(1) as f64 }
        }
        CompareOp::Eq | CompareOp::Ne => return None, // Use distinct_count-based estimate
    };

    Some(selectivity.min(1.0).max(0.0))
}

/// 将 bytes 反序列化为 i64
fn bytes_to_i64(bytes: Option<&[u8]>) -> Option<i64> {
    let bytes = bytes?;
    if bytes.len() == 8 {
        Some(i64::from_le_bytes(bytes.try_into().ok()?))
    } else if bytes.len() == 4 {
        Some(i64::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3], 0, 0, 0, 0]))
    } else {
        None
    }
}

/// 解析 "op value" 形式的谓词，返回 (CompareOp, value_as_i64)
fn parse_predicate(predicate: &str) -> Result<(CompareOp, i64)> {
    let predicate = predicate.trim();

    for (op_str, op) in [
        (">=", CompareOp::Ge),
        ("<=", CompareOp::Le),
        ("!=", CompareOp::Ne),
        (">", CompareOp::Gt),
        ("<", CompareOp::Lt),
        ("=", CompareOp::Eq),
    ] {
        if let Some(rest) = predicate.strip_prefix(op_str) {
            let val_str = rest.trim();
            if let Ok(v) = val_str.parse::<i64>() {
                return Ok((op, v));
            }
            if let Ok(v) = val_str.parse::<f64>() {
                return Ok((op, v as i64));
            }
        }
        if let Some(rest) = predicate.strip_suffix(op_str) {
            let val_str = rest.trim();
            if let Ok(v) = val_str.parse::<i64>() {
                return Ok((op, v));
            }
            if let Ok(v) = val_str.parse::<f64>() {
                return Ok((op, v as i64));
            }
        }
    }

    Err(RockDuckError::Query(format!("Cannot parse predicate: {}", predicate)))
}

/// 执行 Early Materialization（全列读后再过滤）
pub fn early_materialize(
    columns: &[(String, ArrayRef)],
    _positions: Option<&UInt64Array>,
) -> Result<RecordBatch> {
    let fields: Vec<arrow_schema::Field> = columns.iter().map(|(n, a)| {
        arrow_schema::Field::new(n, a.data_type().clone(), true)
    }).collect();
    let schema = Schema::new(fields);

    if columns.is_empty() {
        return Ok(RecordBatch::new_empty(std::sync::Arc::new(schema)));
    }

    let arrays: Vec<_> = columns.iter().map(|(_, a)| a.clone()).collect();
    RecordBatch::try_new(std::sync::Arc::new(schema), arrays)
        .map_err(RockDuckError::Arrow)
}

/// Hybrid Materialization：分批拉 data 列，每次拉 `batch_size` 行。
pub fn hybrid_materialize_batch(
    columns: &[(String, ArrayRef)],
    positions: &UInt64Array,
    batch_size: usize,
) -> Result<Vec<RecordBatch>> {
    let num_positions = positions.len();
    if num_positions == 0 {
        return Ok(vec![]);
    }

    let fields: Vec<arrow_schema::Field> = columns.iter().map(|(n, a)| {
        arrow_schema::Field::new(n, a.data_type().clone(), true)
    }).collect();
    let schema = Schema::new(fields);

    let mut batches = Vec::new();

    for start in (0..num_positions).step_by(batch_size) {
        let end = (start + batch_size).min(num_positions);
        let batch_positions = UInt64Array::from_iter_values(
            positions.values()[start..end].iter().copied()
        );

        let mut batch_arrays = Vec::with_capacity(columns.len());
        for (_, col_data) in columns {
            // Take only positions in this batch
            let taken = take_by_positions_simple(col_data, &batch_positions)?;
            batch_arrays.push(taken);
        }

        let batch = RecordBatch::try_new(
            std::sync::Arc::new(schema.clone()),
            batch_arrays,
        ).map_err(RockDuckError::Arrow)?;
        batches.push(batch);
    }

    Ok(batches)
}

/// 按 positions 抽取列数据（简化版，依赖 arrow-select）
fn take_by_positions_simple(col: &ArrayRef, positions: &UInt64Array) -> Result<ArrayRef> {
    use arrow_select::take::take;
    let indices = arrow_array::UInt64Array::from_iter_values(positions.values().to_vec());
    take(col, &indices, None).map_err(RockDuckError::Arrow)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::segment::meta::ColumnStats;
    use std::sync::Arc;
    use arrow::array::AsArray;
    use arrow_array::Int64Array;

    // ============================================================
    // SlmThresholds
    // ============================================================

    #[test]
    fn test_thresholds_default() {
        let t = SlmThresholds::default();
        assert!((t.early_threshold - 0.9).abs() < 1e-9);
        assert!((t.late_threshold - 0.1).abs() < 1e-9);
    }

    #[test]
    fn test_thresholds_custom() {
        let t = SlmThresholds::new(0.95, 0.05);
        assert!((t.early_threshold - 0.95).abs() < 1e-9);
        assert!((t.late_threshold - 0.05).abs() < 1e-9);
    }

    // ============================================================
    // choose_strategy
    // ============================================================

    #[test]
    fn test_choose_strategy_early() {
        let t = SlmThresholds::default();
        assert_eq!(choose_strategy(0.95, &t), MaterializationStrategy::Early);
        assert_eq!(choose_strategy(1.0, &t), MaterializationStrategy::Early);
    }

    #[test]
    fn test_choose_strategy_late() {
        let t = SlmThresholds::default();
        assert_eq!(choose_strategy(0.05, &t), MaterializationStrategy::Late);
        assert_eq!(choose_strategy(0.0, &t), MaterializationStrategy::Late);
    }

    #[test]
    fn test_choose_strategy_hybrid() {
        let t = SlmThresholds::default();
        assert_eq!(choose_strategy(0.5, &t), MaterializationStrategy::Hybrid);
        assert_eq!(choose_strategy(0.1, &t), MaterializationStrategy::Hybrid);
        assert_eq!(choose_strategy(0.9, &t), MaterializationStrategy::Hybrid);
    }

    // ============================================================
    // estimate_selectivity: return value correctness
    // ============================================================

    #[test]
    fn test_estimate_selectivity_zone_map_ge_narrow_range() {
        // Zone Map: min=10, max=10000. Predicate "> 9500" → only ~5% match.
        let stats = ColumnStats {
            min: Some(10i64.to_le_bytes().to_vec()),
            max: Some(10000i64.to_le_bytes().to_vec()),
            null_count: 0,
            sum: None,
            distinct_count: None,
        };
        let mut zm = ZoneMapStats::new();
        zm.add_column_stats("age", stats);

        let sel = estimate_selectivity("age", "> 9500", Some(&zm), 10000, 1000).unwrap();
        assert!(sel < 0.1,
            "value > 9500 in [10, 10000] should be ~5% selectivity, got {}", sel);
        assert!(sel > 0.0);
    }

    #[test]
    fn test_estimate_selectivity_zone_map_lt_broad_range() {
        // Zone Map: min=10, max=100. Predicate "< 50" → ~40% match.
        let stats = ColumnStats {
            min: Some(10i64.to_le_bytes().to_vec()),
            max: Some(100i64.to_le_bytes().to_vec()),
            null_count: 0,
            sum: None,
            distinct_count: None,
        };
        let mut zm = ZoneMapStats::new();
        zm.add_column_stats("age", stats);

        let sel = estimate_selectivity("age", "< 50", Some(&zm), 1000, 100).unwrap();
        assert!(sel >= 0.3 && sel <= 0.5,
            "value < 50 in [10, 100] should be ~40% selectivity, got {}", sel);
    }

    #[test]
    fn test_estimate_selectivity_eq_uses_distinct() {
        // Eq: selectivity = 1 / distinct_count
        let stats = ColumnStats {
            min: Some(1i64.to_le_bytes().to_vec()),
            max: Some(100i64.to_le_bytes().to_vec()),
            null_count: 0,
            sum: None,
            distinct_count: Some(100),
        };
        let mut zm = ZoneMapStats::new();
        zm.add_column_stats("id", stats);

        let sel = estimate_selectivity("id", "= 42", Some(&zm), 10000, 1000).unwrap();
        assert!((sel - 0.01).abs() < 0.001,
            "Eq selectivity should be 1/distinct=0.01, got {}", sel);
    }

    #[test]
    fn test_estimate_selectivity_ne_is_always_one() {
        let sel = estimate_selectivity("col", "!= 0", None, 1000, 100).unwrap();
        assert!((sel - 1.0).abs() < 1e-9, "Ne should always return 1.0");
    }

    #[test]
    fn test_estimate_selectivity_no_zone_map_fallback() {
        // Without Zone Map: Eq → 1/distinct, Ne → 1.0, range → 0.5
        let sel_eq = estimate_selectivity("col", "= 5", None, 1000, 100).unwrap();
        assert!((sel_eq - 0.01).abs() < 0.001); // default distinct=100

        let sel_ne = estimate_selectivity("col", "!= 5", None, 1000, 100).unwrap();
        assert!((sel_ne - 1.0).abs() < 1e-9);

        let sel_range = estimate_selectivity("col", "> 5", None, 1000, 100).unwrap();
        assert!((sel_range - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_estimate_selectivity_zero_rows() {
        let sel = estimate_selectivity("col", "> 0", None, 0, 0).unwrap();
        assert!((sel - 0.0).abs() < 1e-9);
    }

    // ============================================================
    // Integration: selectivity → strategy selection consistency
    // ============================================================

    #[test]
    fn test_choose_strategy_consistency_with_estimate() {
        let t = SlmThresholds::default();

        // Zone Map [1, 100], predicate "> 0" → selectivity ≈ 1.0 → Early
        let stats = ColumnStats {
            min: Some(1i64.to_le_bytes().to_vec()),
            max: Some(100i64.to_le_bytes().to_vec()),
            null_count: 0,
            sum: None,
            distinct_count: None,
        };
        let mut zm = ZoneMapStats::new();
        zm.add_column_stats("age", stats);
        let sel_high = estimate_selectivity("age", "> 0", Some(&zm), 1000, 100).unwrap();
        assert!(sel_high > 0.9, "value > 0 in [1, 100] should have high selectivity, got {}", sel_high);
        assert_eq!(choose_strategy(sel_high, &t), MaterializationStrategy::Early,
            "high selectivity (>90%) should choose Early");

        // No Zone Map, Eq → 1/distinct=0.01 → Late
        let sel_low = estimate_selectivity("id", "= 999999", None, 10000, 1000).unwrap();
        assert_eq!(choose_strategy(sel_low, &t), MaterializationStrategy::Late,
            "low selectivity (<10%) should choose Late");

        // No Zone Map, range → 0.5 → Hybrid
        let sel_mid = estimate_selectivity("age", "> 50", None, 1000, 100).unwrap();
        assert_eq!(choose_strategy(sel_mid, &t), MaterializationStrategy::Hybrid,
            "medium selectivity should choose Hybrid");
    }

    #[test]
    fn test_hybrid_materialize_batch_produces_correct_values() {
        let col1: ArrayRef = Arc::new(Int64Array::from(vec![10i64, 20, 30, 40, 50]));
        let columns = vec![("a".to_string(), col1)];
        let positions = UInt64Array::from_iter_values(vec![0u64, 2, 4]);
        let batches = hybrid_materialize_batch(&columns, &positions, 2).unwrap();

        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].num_columns(), 1);
        let arr = batches[0].column(0).as_primitive::<arrow_array::types::Int64Type>();
        assert_eq!(arr.value(0), 10);
        assert_eq!(arr.value(1), 30);

        assert_eq!(batches[1].num_rows(), 1);
        let arr2 = batches[1].column(0).as_primitive::<arrow_array::types::Int64Type>();
        assert_eq!(arr2.value(0), 50);
    }

    #[test]
    fn test_early_materialize_preserves_all_data() {
        let col1: ArrayRef = Arc::new(Int64Array::from(vec![1i64, 2, 3]));
        let col2: ArrayRef = Arc::new(Int64Array::from(vec![10i64, 20, 30]));
        let columns = vec![("a".to_string(), col1), ("b".to_string(), col2)];

        let batch = early_materialize(&columns, None).unwrap();
        assert_eq!(batch.num_rows(), 3);
        assert_eq!(batch.num_columns(), 2);
        assert_eq!(batch.column(0).as_primitive::<arrow_array::types::Int64Type>().value(1), 2);
        assert_eq!(batch.column(1).as_primitive::<arrow_array::types::Int64Type>().value(2), 30);
    }

    // ============================================================
    // parse_predicate
    // ============================================================

    #[test]
    fn test_parse_predicate_prefix() {
        let (op, val) = parse_predicate("> 18").unwrap();
        assert!(matches!(op, CompareOp::Gt));
        assert_eq!(val, 18);
    }

    #[test]
    fn test_parse_predicate_suffix() {
        let (op, val) = parse_predicate("18 <=").unwrap();
        assert!(matches!(op, CompareOp::Le));
        assert_eq!(val, 18);
    }

    #[test]
    fn test_parse_predicate_float() {
        let (op, val) = parse_predicate(">= 95.5").unwrap();
        assert!(matches!(op, CompareOp::Ge));
        assert_eq!(val, 95);
    }

    #[test]
    fn test_parse_predicate_eq() {
        let (op, val) = parse_predicate("= 42").unwrap();
        assert!(matches!(op, CompareOp::Eq));
        assert_eq!(val, 42);
    }

    #[test]
    fn test_parse_predicate_ne() {
        let (op, val) = parse_predicate("!= 0").unwrap();
        assert!(matches!(op, CompareOp::Ne));
        assert_eq!(val, 0);
    }

    #[test]
    fn test_parse_predicate_invalid() {
        assert!(parse_predicate("invalid").is_err());
        assert!(parse_predicate("").is_err());
    }

    #[test]
    fn test_estimate_selectivity_zone_map_lt() {
        let sel = estimate_selectivity("age", "< 30", None, 1000, 1000).unwrap();
        assert_eq!(sel, 0.5, "unknown predicate falls back to 0.5");
    }

    #[test]
    fn test_early_materialize_empty() {
        let batch = early_materialize(&[], None).unwrap();
        assert_eq!(batch.num_rows(), 0);
    }

    #[test]
    fn test_hybrid_materialize_batch_empty() {
        let col1: ArrayRef = Arc::new(Int64Array::from(vec![1i64, 2]));
        let columns = vec![("a".to_string(), col1)];
        let positions = UInt64Array::from_iter_values(vec![]);

        let batches = hybrid_materialize_batch(&columns, &positions, 2).unwrap();
        assert!(batches.is_empty());
    }
}
