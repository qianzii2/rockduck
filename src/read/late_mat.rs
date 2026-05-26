//! Late Materialization 延迟物化
//!
//! 查询执行策略：
//! 1. 只读 filter 列 → predicate 过滤 → 得到 positions
//! 2. 只读 needed_cols 的 positions（Vortex 支持按 position 读取）
//! 3. 应用 Del Mask（过滤删除行）
//! 4. 拼接 Arrow RecordBatch
//! 5. 交给 DuckDB 执行剩余 SQL 逻辑

use std::collections::HashSet;
use arrow_array::{ArrayRef, RecordBatch, UInt64Array};
use arrow::array::AsArray;
use arrow_schema::Schema;
use tracing::debug;

use crate::error::{RockDuckError, Result};

/// 延迟物化上下文
pub struct LateMaterialization {
    pub needed_columns: HashSet<String>,
    pub filter_column: Option<String>,
    pub filter_predicate: Option<String>,
}

impl LateMaterialization {
    pub fn new() -> Self {
        Self {
            needed_columns: HashSet::new(),
            filter_column: None,
            filter_predicate: None,
        }
    }

    pub fn add_needed_column(mut self, col: impl Into<String>) -> Self {
        self.needed_columns.insert(col.into());
        self
    }

    pub fn with_filter(mut self, col: impl Into<String>, predicate: impl Into<String>) -> Self {
        self.filter_column = Some(col.into());
        self.filter_predicate = Some(predicate.into());
        self
    }

    pub fn execute(&self) -> Result<RecordBatch> {
        debug!("LateMaterialization: needed={:?}, filter={:?}",
            self.needed_columns, self.filter_predicate);

        let schema = Schema::new(Vec::<arrow_schema::Field>::new());
        Ok(RecordBatch::new_empty(std::sync::Arc::new(schema)))
    }
}

impl Default for LateMaterialization {
    fn default() -> Self {
        Self::new()
    }
}

/// 估算 filter 列的选择率（Feature 2: Adaptive Late Materialization）。
///
/// 使用 Zone Map 统计信息来估算谓词匹配的行数比例。
pub fn selectivity_estimator(
    column: &str,
    predicate: &str,
    zone_map_stats: Option<&crate::segment::meta::ZoneMapStats>,
    total_rows: u64,
) -> Result<f64> {
    crate::read::adaptive_lm::estimate_selectivity(
        column,
        predicate,
        zone_map_stats,
        total_rows,
        total_rows as u32,
    )
}

/// 比较操作
#[derive(Debug, Clone, Copy)]
pub enum CompareOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

impl CompareOp {
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "=" | "==" | "eq" => Some(CompareOp::Eq),
            "!=" | "<>" | "ne" => Some(CompareOp::Ne),
            "<" | "lt" => Some(CompareOp::Lt),
            "<=" | "le" => Some(CompareOp::Le),
            ">" | "gt" => Some(CompareOp::Gt),
            ">=" | "ge" => Some(CompareOp::Ge),
            _ => None,
        }
    }
}

/// 从 filter 列和 predicate 提取 positions
pub fn extract_positions(
    _filter_column: &str,
    predicate: &str,
    data: &ArrayRef,
) -> Result<UInt64Array> {
    let (op, value) = parse_simple_predicate(predicate)?;
    filter_by_comparison(data, &op, &value)
}

/// 解析简单的 "col op value" 谓词
fn parse_simple_predicate(predicate: &str) -> Result<(CompareOp, i64)> {
    let predicate = predicate.trim();
    
    for (op_str, op) in [
        (">=", CompareOp::Ge),
        ("<=", CompareOp::Le),
        ("!=", CompareOp::Ne),
        ("==", CompareOp::Eq),
        ("<>", CompareOp::Ne),
        (">", CompareOp::Gt),
        ("<", CompareOp::Lt),
        ("=", CompareOp::Eq),
    ] {
        if let Some(rest) = predicate.strip_prefix(op_str) {
            let val_str = rest.trim();
            if let Ok(val) = val_str.parse::<i64>() {
                return Ok((op, val));
            }
            if let Ok(val) = val_str.parse::<f64>() {
                return Ok((op, val as i64));
            }
        }
        if let Some(rest) = predicate.strip_suffix(op_str) {
            let val_str = rest.trim();
            if let Ok(val) = val_str.parse::<i64>() {
                return Ok((op, val));
            }
            if let Ok(val) = val_str.parse::<f64>() {
                return Ok((op, val as i64));
            }
        }
    }
    
    Err(RockDuckError::Query(format!("Cannot parse predicate: {}", predicate)))
}

/// 根据比较操作过滤数据，返回匹配的行号
fn filter_by_comparison(data: &ArrayRef, op: &CompareOp, value: &i64) -> Result<UInt64Array> {
    let mut positions = Vec::new();
    
    if let Some(arr) = data.as_primitive_opt::<arrow_array::types::Int64Type>() {
        for i in 0..arr.len() {
            let v = arr.value(i);
            let keep = match op {
                CompareOp::Eq => v == *value,
                CompareOp::Ne => v != *value,
                CompareOp::Lt => v < *value,
                CompareOp::Le => v <= *value,
                CompareOp::Gt => v > *value,
                CompareOp::Ge => v >= *value,
            };
            if keep {
                positions.push(i as u64);
            }
        }
    } else {
        for i in 0..data.len() {
            positions.push(i as u64);
        }
    }
    
    Ok(UInt64Array::from_iter_values(positions))
}

/// 按 positions 读取列（take 操作）
pub fn take_by_positions(
    column_data: &ArrayRef,
    positions: &UInt64Array,
) -> Result<ArrayRef> {
    use arrow_select::take::take;
    
    let indices = arrow_array::UInt64Array::from_iter_values(positions.values().to_vec());
    let taken = take(column_data, &indices, None)?;
    Ok(taken)
}

/// 应用 Del Mask 到 positions
pub fn apply_del_mask(
    positions: &UInt64Array,
    del_mask: &crate::segment::del_mask::DelMask,
) -> UInt64Array {
    let kept: Vec<u64> = positions
        .values()
        .iter()
        .filter(|&&pos| !del_mask.is_deleted(pos))
        .copied()
        .collect();

    UInt64Array::from_iter_values(kept)
}

/// 拼接多个 Array 为 RecordBatch
pub fn materialize(
    positions: &UInt64Array,
    columns: &[(String, ArrayRef)],
    del_mask: Option<&crate::segment::del_mask::DelMask>,
) -> Result<RecordBatch> {
    let positions = if let Some(mask) = del_mask {
        apply_del_mask(positions, mask)
    } else {
        positions.clone()
    };

    let mut taken_cols = Vec::new();
    for (name, data) in columns {
        let taken = take_by_positions(data, &positions)?;
        taken_cols.push((name.clone(), taken));
    }

    if taken_cols.is_empty() {
        let schema = Schema::new(Vec::<arrow_schema::Field>::new());
        return Ok(RecordBatch::new_empty(std::sync::Arc::new(schema)));
    }

    let schema = Schema::new(
        taken_cols.iter().map(|(n, d)| {
            arrow_schema::Field::new(n, d.data_type().clone(), true)
        }).collect::<Vec<_>>()
    );
    let arrays: Vec<_> = taken_cols.into_iter().map(|(_, a)| a).collect();
    RecordBatch::try_new(std::sync::Arc::new(schema), arrays)
        .map_err(|e| RockDuckError::Arrow(e))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use crate::segment::del_mask::DelMask;

    // ============================================================
    // LateMaterialization construction
    // ============================================================

    #[test]
    fn test_late_materialization_new() {
        let lm = LateMaterialization::new();
        assert!(lm.needed_columns.is_empty());
        assert!(lm.filter_column.is_none());
        assert!(lm.filter_predicate.is_none());
    }

    #[test]
    fn test_late_materialization_default() {
        let lm = LateMaterialization::default();
        assert!(lm.needed_columns.is_empty());
        assert!(lm.filter_column.is_none());
    }

    #[test]
    fn test_late_materialization_add_needed_column() {
        let lm = LateMaterialization::new().add_needed_column("age");
        assert!(lm.needed_columns.contains("age"));
    }

    #[test]
    fn test_late_materialization_add_needed_column_chaining() {
        let lm = LateMaterialization::new()
            .add_needed_column("id")
            .add_needed_column("name")
            .add_needed_column("value");
        assert!(lm.needed_columns.contains("id"));
        assert!(lm.needed_columns.contains("name"));
        assert!(lm.needed_columns.contains("value"));
        assert_eq!(lm.needed_columns.len(), 3);
    }

    #[test]
    fn test_late_materialization_with_filter() {
        let lm = LateMaterialization::new()
            .add_needed_column("id")
            .with_filter("age", "> 18");
        assert_eq!(lm.filter_column, Some("age".to_string()));
        assert_eq!(lm.filter_predicate, Some("> 18".to_string()));
    }

    #[test]
    fn test_late_materialization_execute() {
        let lm = LateMaterialization::new();
        let batch = lm.execute().unwrap();
        assert_eq!(batch.num_rows(), 0);
    }

    #[test]
    fn test_late_materialization_execute_with_columns() {
        let lm = LateMaterialization::new()
            .add_needed_column("id")
            .add_needed_column("name");
        let batch = lm.execute().unwrap();
        assert_eq!(batch.num_rows(), 0);
    }

    // ============================================================
    // CompareOp from_str
    // ============================================================

    #[test]
    fn test_compare_op_from_str_eq() {
        assert!(matches!(CompareOp::from_str("="), Some(CompareOp::Eq)));
        assert!(matches!(CompareOp::from_str("=="), Some(CompareOp::Eq)));
        assert!(matches!(CompareOp::from_str("eq"), Some(CompareOp::Eq)));
    }

    #[test]
    fn test_compare_op_from_str_ne() {
        assert!(matches!(CompareOp::from_str("!="), Some(CompareOp::Ne)));
        assert!(matches!(CompareOp::from_str("<>"), Some(CompareOp::Ne)));
        assert!(matches!(CompareOp::from_str("ne"), Some(CompareOp::Ne)));
    }

    #[test]
    fn test_compare_op_from_str_lt() {
        assert!(matches!(CompareOp::from_str("<"), Some(CompareOp::Lt)));
        assert!(matches!(CompareOp::from_str("lt"), Some(CompareOp::Lt)));
    }

    #[test]
    fn test_compare_op_from_str_le() {
        assert!(matches!(CompareOp::from_str("<="), Some(CompareOp::Le)));
        assert!(matches!(CompareOp::from_str("le"), Some(CompareOp::Le)));
    }

    #[test]
    fn test_compare_op_from_str_gt() {
        assert!(matches!(CompareOp::from_str(">"), Some(CompareOp::Gt)));
        assert!(matches!(CompareOp::from_str("gt"), Some(CompareOp::Gt)));
    }

    #[test]
    fn test_compare_op_from_str_ge() {
        assert!(matches!(CompareOp::from_str(">="), Some(CompareOp::Ge)));
        assert!(matches!(CompareOp::from_str("ge"), Some(CompareOp::Ge)));
    }

    #[test]
    fn test_compare_op_from_str_invalid() {
        assert!(CompareOp::from_str("invalid").is_none());
        assert!(CompareOp::from_str("===").is_none());
        assert!(CompareOp::from_str("").is_none());
        assert!(CompareOp::from_str("between").is_none());
    }

    // ============================================================
    // parse_simple_predicate
    // ============================================================

    #[test]
    fn test_parse_simple_predicate_int_ge() {
        let result = parse_simple_predicate(">=18");
        assert!(result.is_ok());
        let (op, val) = result.unwrap();
        assert!(matches!(op, CompareOp::Ge));
        assert_eq!(val, 18);
    }

    #[test]
    fn test_parse_simple_predicate_int_le() {
        let result = parse_simple_predicate("<=50000");
        assert!(result.is_ok());
        let (op, val) = result.unwrap();
        assert!(matches!(op, CompareOp::Le));
        assert_eq!(val, 50000);
    }

    #[test]
    fn test_parse_simple_predicate_int_ne() {
        let result = parse_simple_predicate("!=0");
        assert!(result.is_ok());
        let (op, val) = result.unwrap();
        assert!(matches!(op, CompareOp::Ne));
        assert_eq!(val, 0);
    }

    #[test]
    fn test_parse_simple_predicate_int_eq() {
        let result = parse_simple_predicate("=10");
        assert!(result.is_ok());
        let (op, val) = result.unwrap();
        assert!(matches!(op, CompareOp::Eq));
        assert_eq!(val, 10);
    }

    #[test]
    fn test_parse_simple_predicate_int_eq_double() {
        let result = parse_simple_predicate("==10");
        assert!(result.is_ok());
        let (op, val) = result.unwrap();
        assert!(matches!(op, CompareOp::Eq));
        assert_eq!(val, 10);
    }

    #[test]
    fn test_parse_simple_predicate_int_gt() {
        let result = parse_simple_predicate(">5");
        assert!(result.is_ok());
        let (op, val) = result.unwrap();
        assert!(matches!(op, CompareOp::Gt));
        assert_eq!(val, 5);
    }

    #[test]
    fn test_parse_simple_predicate_int_lt() {
        let result = parse_simple_predicate("<100");
        assert!(result.is_ok());
        let (op, val) = result.unwrap();
        assert!(matches!(op, CompareOp::Lt));
        assert_eq!(val, 100);
    }

    #[test]
    fn test_parse_simple_predicate_float() {
        let result = parse_simple_predicate(">=95.5");
        assert!(result.is_ok());
        let (op, val) = result.unwrap();
        assert!(matches!(op, CompareOp::Ge));
        assert_eq!(val, 95); // Truncated
    }

    #[test]
    fn test_parse_simple_predicate_float_suffix() {
        let result = parse_simple_predicate("18>=");
        assert!(result.is_ok());
        let (op, val) = result.unwrap();
        assert!(matches!(op, CompareOp::Ge));
        assert_eq!(val, 18);
    }

    #[test]
    fn test_parse_simple_predicate_whitespace() {
        let result = parse_simple_predicate(">=   18");
        assert!(result.is_ok());
        let (_, val) = result.unwrap();
        assert_eq!(val, 18);
    }

    #[test]
    fn test_parse_simple_predicate_negative() {
        let result = parse_simple_predicate("<-10");
        assert!(result.is_ok());
        let (_, val) = result.unwrap();
        assert_eq!(val, -10);
    }

    #[test]
    fn test_parse_simple_predicate_invalid() {
        assert!(parse_simple_predicate("invalid").is_err());
        assert!(parse_simple_predicate("x between 1 and 5").is_err());
        assert!(parse_simple_predicate("age").is_err());
    }

    #[test]
    fn test_parse_simple_predicate_no_value() {
        assert!(parse_simple_predicate(">=").is_err());
        assert!(parse_simple_predicate(">18<=").is_err());
    }

    // ============================================================
    // filter_by_comparison with Int64Array
    // ============================================================

    #[test]
    fn test_filter_by_comparison_int64_eq() {
        use arrow_array::Int64Array;
        let arr: ArrayRef = Arc::new(Int64Array::from(vec![10, 20, 30, 40, 50]));
        let positions = filter_by_comparison(&arr, &CompareOp::Eq, &20).unwrap();
        assert_eq!(positions.len(), 1);
        assert_eq!(positions.value(0), 1);
    }

    #[test]
    fn test_filter_by_comparison_int64_ne() {
        use arrow_array::Int64Array;
        let arr: ArrayRef = Arc::new(Int64Array::from(vec![10, 20, 20, 40]));
        let positions = filter_by_comparison(&arr, &CompareOp::Ne, &20).unwrap();
        assert_eq!(positions.len(), 2);
        assert_eq!(positions.value(0), 0);
        assert_eq!(positions.value(1), 3);
    }

    #[test]
    fn test_filter_by_comparison_int64_lt() {
        use arrow_array::Int64Array;
        let arr: ArrayRef = Arc::new(Int64Array::from(vec![1, 5, 10, 15, 20]));
        let positions = filter_by_comparison(&arr, &CompareOp::Lt, &10).unwrap();
        assert_eq!(positions.len(), 2);
        assert_eq!(positions.value(0), 0);
        assert_eq!(positions.value(1), 1);
    }

    #[test]
    fn test_filter_by_comparison_int64_le() {
        use arrow_array::Int64Array;
        let arr: ArrayRef = Arc::new(Int64Array::from(vec![1, 5, 10, 15]));
        let positions = filter_by_comparison(&arr, &CompareOp::Le, &10).unwrap();
        assert_eq!(positions.len(), 3);
    }

    #[test]
    fn test_filter_by_comparison_int64_gt() {
        use arrow_array::Int64Array;
        let arr: ArrayRef = Arc::new(Int64Array::from(vec![1, 5, 10, 15]));
        let positions = filter_by_comparison(&arr, &CompareOp::Gt, &5).unwrap();
        assert_eq!(positions.len(), 2);
        assert_eq!(positions.value(0), 2);
        assert_eq!(positions.value(1), 3);
    }

    #[test]
    fn test_filter_by_comparison_int64_ge() {
        use arrow_array::Int64Array;
        let arr: ArrayRef = Arc::new(Int64Array::from(vec![1, 5, 10, 15]));
        let positions = filter_by_comparison(&arr, &CompareOp::Ge, &10).unwrap();
        assert_eq!(positions.len(), 2);
    }

    #[test]
    fn test_filter_by_comparison_int64_no_matches() {
        use arrow_array::Int64Array;
        let arr: ArrayRef = Arc::new(Int64Array::from(vec![1, 2, 3]));
        let positions = filter_by_comparison(&arr, &CompareOp::Eq, &999).unwrap();
        assert_eq!(positions.len(), 0);
    }

    #[test]
    fn test_filter_by_comparison_int64_all_match() {
        use arrow_array::Int64Array;
        let arr: ArrayRef = Arc::new(Int64Array::from(vec![5, 5, 5]));
        let positions = filter_by_comparison(&arr, &CompareOp::Ne, &0).unwrap();
        assert_eq!(positions.len(), 3);
    }

    #[test]
    fn test_filter_by_comparison_int64_empty() {
        use arrow_array::Int64Array;
        let arr: ArrayRef = Arc::new(Int64Array::from(Vec::<i64>::new()));
        let positions = filter_by_comparison(&arr, &CompareOp::Eq, &0).unwrap();
        assert_eq!(positions.len(), 0);
    }

    // ============================================================
    // filter_by_comparison with non-Int64 data (fallback)
    // ============================================================

    #[test]
    fn test_filter_by_comparison_string_fallback() {
        use arrow_array::StringArray;
        let arr: ArrayRef = Arc::new(StringArray::from(vec!["a", "b", "c"]));
        // String array falls back to returning all positions
        let positions = filter_by_comparison(&arr, &CompareOp::Eq, &0).unwrap();
        assert_eq!(positions.len(), 3);
    }

    // ============================================================
    // take_by_positions
    // ============================================================

    #[test]
    fn test_take_by_positions_int64() {
        use arrow_array::Int64Array;
        let arr: ArrayRef = Arc::new(Int64Array::from(vec![10, 20, 30, 40, 50]));
        let positions = UInt64Array::from_iter_values(vec![1, 3]);
        let taken = take_by_positions(&arr, &positions).unwrap();
        let taken_arr = taken.as_primitive::<arrow_array::types::Int64Type>();
        assert_eq!(taken_arr.len(), 2);
        assert_eq!(taken_arr.value(0), 20);
        assert_eq!(taken_arr.value(1), 40);
    }

    #[test]
    fn test_take_by_positions_empty() {
        use arrow_array::Int64Array;
        let arr: ArrayRef = Arc::new(Int64Array::from(vec![10, 20, 30]));
        let positions = UInt64Array::from_iter_values(Vec::<u64>::new());
        let taken = take_by_positions(&arr, &positions).unwrap();
        assert_eq!(taken.len(), 0);
    }

    #[test]
    fn test_take_by_positions_out_of_order() {
        use arrow_array::Int64Array;
        let arr: ArrayRef = Arc::new(Int64Array::from(vec![10, 20, 30, 40]));
        let positions = UInt64Array::from_iter_values(vec![3, 1, 2]);
        let taken = take_by_positions(&arr, &positions).unwrap();
        let taken_arr = taken.as_primitive::<arrow_array::types::Int64Type>();
        assert_eq!(taken_arr.value(0), 40);
        assert_eq!(taken_arr.value(1), 20);
        assert_eq!(taken_arr.value(2), 30);
    }

    // ============================================================
    // apply_del_mask
    // ============================================================

    #[test]
    fn test_apply_del_mask_filters_deleted() {
        let positions = UInt64Array::from_iter_values(vec![0, 1, 2, 3, 4]);

        let mut del_mask = DelMask::new(5);
        del_mask.add_delete(1);
        del_mask.add_delete(3);

        let result = apply_del_mask(&positions, &del_mask);
        assert_eq!(result.len(), 3);
        assert!(result.values().contains(&0));
        assert!(result.values().contains(&2));
        assert!(result.values().contains(&4));
    }

    #[test]
    fn test_apply_del_mask_no_deletes() {
        let positions = UInt64Array::from_iter_values(vec![0, 1, 2]);
        let del_mask = DelMask::new(3);
        let result = apply_del_mask(&positions, &del_mask);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_apply_del_mask_all_deleted() {
        let positions = UInt64Array::from_iter_values(vec![0, 1, 2]);
        let mut del_mask = DelMask::new(3);
        del_mask.add_delete(0);
        del_mask.add_delete(1);
        del_mask.add_delete(2);
        let result = apply_del_mask(&positions, &del_mask);
        assert_eq!(result.len(), 0);
    }

    // ============================================================
    // materialize
    // ============================================================

    #[test]
    fn test_materialize_empty_columns() {
        let positions = UInt64Array::from_iter_values(vec![0, 1]);
        let columns: Vec<(String, ArrayRef)> = vec![];
        let batch = materialize(&positions, &columns, None).unwrap();
        assert_eq!(batch.num_rows(), 0);
    }

    #[test]
    fn test_materialize_with_columns() {
        use arrow_array::Int64Array;
        let positions = UInt64Array::from_iter_values(vec![0, 2]);
        let col1: ArrayRef = Arc::new(Int64Array::from(vec![10, 20, 30]));
        let col2: ArrayRef = Arc::new(Int64Array::from(vec![100, 200, 300]));
        let columns = vec![
            ("id".to_string(), col1),
            ("val".to_string(), col2),
        ];
        let batch = materialize(&positions, &columns, None).unwrap();
        assert_eq!(batch.num_rows(), 2);
        assert_eq!(batch.num_columns(), 2);
    }

    #[test]
    fn test_materialize_with_del_mask() {
        use arrow_array::Int64Array;
        let positions = UInt64Array::from_iter_values(vec![0, 1, 2, 3, 4]);
        let col1: ArrayRef = Arc::new(Int64Array::from(vec![1, 2, 3, 4, 5]));
        let columns = vec![("id".to_string(), col1)];

        let mut del_mask = DelMask::new(5);
        del_mask.add_delete(1);
        del_mask.add_delete(3);

        let batch = materialize(&positions, &columns, Some(&del_mask)).unwrap();
        assert_eq!(batch.num_rows(), 3);
    }
}
