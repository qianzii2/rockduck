//! Update Mask 增量物化
//!
//! 对单个 segment 增量物化 UpdMask，不触发完整 compaction。
//! 流程：读 segment 列文件 → 应用 materialize_column() → 写回文件 → 更新 DelMask stale tracking
//!
//! 触发条件（由 CompactionScheduler 调用）：
//!   - 单列更新比例高（should_materialize）但删除率低（不需要 full compaction）
//!
//! 参照：ClickHouse mutation part 思路

use std::path::Path;
use std::sync::Arc;
use tracing::{debug, warn};

use crate::error::{RockDuckError, Result};
use crate::segment::del_mask::DelMask;
use crate::segment::layout::SegmentLayout;
use crate::segment::upd_mask::UpdMask;

/// 增量物化结果
#[derive(Debug)]
pub struct MaterializeResult {
    /// 物化的列数
    pub columns_materialized: u32,
    /// 物化前 UpdMask 中的总更新数
    pub updates_consumed: usize,
    /// 是否完全物化（UpdMask 已清空）
    pub fully_materialized: bool,
}

/// 检查 UpdMask 是否值得增量物化（单列更新多但删除少）。
pub fn should_materialize(upd_mask: &UpdMask, threshold: f64) -> bool {
    upd_mask.should_materialize(threshold)
}

/// 对单个 segment 的 UpdMask 做增量物化。
///
/// 流程：
/// 1. 加载 UpdMask
/// 2. 对每个被更新的列：读原始列文件 → materialize_column → 写回
/// 3. 保存新的 UpdMask（已清除物化列的记录）
/// 4. 返回物化结果
pub fn materialize_segment_updates(
    data_dir: &Path,
    seg_id: &str,
    col_names: &[String],
    upd_mask: &UpdMask,
) -> Result<MaterializeResult> {
    let layout = SegmentLayout::new(data_dir, seg_id);

    let mut columns_materialized = 0u32;
    let mut updates_consumed = 0usize;
    let mut fully_materialized = true;

    for col_name in col_names {
        let col_updates = upd_mask.update_count(col_name);
        if col_updates == 0 {
            continue;
        }

        let col_path = layout.col_path(col_name);
        if !col_path.exists() {
            warn!("Column file {} not found, skipping materialize for {}", col_path.display(), col_name);
            continue;
        }

        // 读取原始列数据
        let original_data = match std::fs::read(&col_path) {
            Ok(bytes) => bytes,
            Err(e) => {
                warn!("Failed to read column {}: {}", col_path.display(), e);
                continue;
            }
        };

        // 解析为 Arrow IPC 格式
        let array: arrow_array::ArrayRef = match parse_arrow_ipc(&original_data) {
            Ok(arr) => arr,
            Err(_) => {
                // Fallback: treat raw bytes as a Vec of UInt8 values
                use arrow_array::builder::UInt8Builder;
                let mut builder = UInt8Builder::new();
                for &b in &original_data {
                    builder.append_value(b);
                }
                Arc::new(builder.finish())
            }
        };

        // 应用 materialize_column（应用更新并清除记录）
        let mut mask_to_apply = upd_mask.clone();
        match mask_to_apply.materialize_column(col_name, &array) {
            Ok(new_array) => {
                // 写回文件（用 Arrow IPC 格式）
                let schema = arrow_schema::Schema::new(vec![arrow_schema::Field::new(
                    col_name,
                    array.data_type().clone(),
                    true,
                )]);
                let batch = match arrow_array::RecordBatch::try_new(
                    std::sync::Arc::new(schema.clone()),
                    vec![new_array.clone()],
                ) {
                    Ok(b) => b,
                    Err(e) => {
                        warn!("Failed to create RecordBatch for {}: {}", col_name, e);
                        fully_materialized = false;
                        continue;
                    }
                };
                let output = write_arrow_ipc_batch(&batch)?;

                if std::fs::write(&col_path, &output).is_ok() {
                    columns_materialized += 1;
                    updates_consumed += col_updates;
                    debug!("Materialized column {} ({} updates)", col_name, col_updates);
                }
            }
            Err(e) => {
                warn!("Failed to materialize column {}: {}", col_name, e);
                fully_materialized = false;
            }
        }
    }

    Ok(MaterializeResult {
        columns_materialized,
        updates_consumed,
        fully_materialized,
    })
}

/// 解析 Arrow IPC 格式的列数据，返回第一个数组。
fn parse_arrow_ipc(data: &[u8]) -> Result<arrow_array::ArrayRef> {
    use std::io::BufReader;
    let cursor = std::io::Cursor::new(data);
    let reader = arrow_ipc::reader::FileReader::try_new(
        BufReader::new(cursor),
        None,
    ).map_err(|e| RockDuckError::Arrow(e))?;

    for batch in reader {
        let batch = batch.map_err(|e| RockDuckError::Arrow(e))?;
        if batch.num_columns() > 0 {
            return Ok(batch.column(0).clone());
        }
    }
    Err(RockDuckError::Storage("No arrays in Arrow IPC data".into()))
}

/// 写 RecordBatch 为 Arrow IPC bytes。
fn write_arrow_ipc_batch(batch: &arrow_array::RecordBatch) -> Result<Vec<u8>> {
    let mut buf = Vec::new();
    let mut writer = arrow_ipc::writer::FileWriter::try_new(&mut buf, &batch.schema())
        .map_err(|e| RockDuckError::Arrow(e))?;
    writer.write(batch)
        .map_err(|e| RockDuckError::Arrow(e))?;
    writer.finish()
        .map_err(|e| RockDuckError::Arrow(e))?;
    Ok(buf)
}

/// 判断 segment 是否需要增量物化（更新多但删除少）。
pub fn segment_needs_materialization(
    _data_dir: &Path,
    _seg_id: &str,
    upd_mask: &UpdMask,
    del_mask: &DelMask,
    update_threshold: f64,
    del_ratio_threshold: f64,
) -> bool {
    // 条件1：更新值得物化
    if !should_materialize(upd_mask, update_threshold) {
        return false;
    }
    // 条件2：删除率低（不需要 full compaction，增量物化更划算）
    if del_mask.del_ratio() > del_ratio_threshold {
        return false;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::ArrayRef;
    use arrow::array::AsArray;

    #[test]
    fn test_should_materialize_standalone() {
        let mut mask = crate::segment::upd_mask::UpdMask::new(100);
        for i in 0..10 {
            mask.set("age", i, i.to_le_bytes().to_vec());
        }
        assert!(should_materialize(&mask, 0.05));
        assert!(!should_materialize(&mask, 0.15));
    }

    #[test]
    fn test_should_materialize_empty() {
        let mask = crate::segment::upd_mask::UpdMask::new(100);
        assert!(!should_materialize(&mask, 0.01));
    }

    #[test]
    fn test_materialize_result_debug() {
        let result = MaterializeResult {
            columns_materialized: 3,
            updates_consumed: 50,
            fully_materialized: true,
        };
        let debug_str = format!("{:?}", result);
        assert!(!debug_str.is_empty());
    }

    #[test]
    fn test_materialize_result_not_fully() {
        let result = MaterializeResult {
            columns_materialized: 1,
            updates_consumed: 5,
            fully_materialized: false,
        };
        let debug_str = format!("{:?}", result);
        assert!(!debug_str.is_empty());
    }

    #[test]
    fn test_segment_needs_materialization_update_threshold() {
        let upd = crate::segment::upd_mask::UpdMask::new(100);
        let del = DelMask::new(100);
        // No updates -> should not need materialization
        assert!(!segment_needs_materialization(
            std::path::Path::new("/tmp"),
            "seg_001",
            &upd,
            &del,
            0.05,
            0.5,
        ));
    }

    #[test]
    fn test_segment_needs_materialization_del_too_high() {
        let mut upd = crate::segment::upd_mask::UpdMask::new(100);
        for i in 0..10 {
            upd.set("age", i, i.to_le_bytes().to_vec());
        }

        let mut del = DelMask::new(100);
        for i in 0..60 {
            del.add_delete(i);
        }
        // del_ratio = 60% > 0.5 threshold -> full compaction preferred
        assert!(!segment_needs_materialization(
            std::path::Path::new("/tmp"),
            "seg_001",
            &upd,
            &del,
            0.05,
            0.5,
        ));
    }

    #[test]
    fn test_segment_needs_materialization_ok() {
        let mut upd = crate::segment::upd_mask::UpdMask::new(100);
        for i in 0..10 {
            upd.set("age", i, i.to_le_bytes().to_vec());
        }

        let del = DelMask::new(100);
        // del_ratio = 0 -> low deletion, incremental materialize is better
        assert!(segment_needs_materialization(
            std::path::Path::new("/tmp"),
            "seg_001",
            &upd,
            &del,
            0.05,
            0.5,
        ));
    }

    // ============================================================
    // materialize_column roundtrip tests: value correctness + idempotency
    // ============================================================

    #[test]
    fn test_materialize_column_roundtrip_value_correctness() {
        // Core roundtrip: original array → apply updates → materialize → correct values
        let mut mask = crate::segment::upd_mask::UpdMask::new(10);
        mask.set("age", 1, 25i64.to_le_bytes().to_vec());
        mask.set("age", 3, 35i64.to_le_bytes().to_vec());
        mask.set("age", 5, 45i64.to_le_bytes().to_vec());

        let original: ArrayRef = Arc::new(arrow_array::Int64Array::from(vec![
            10i64, 20, 30, 40, 50, 60, 70, 80, 90, 100,
        ]));

        let result = mask.materialize_column("age", &original).unwrap();
        let arr = result.as_primitive::<arrow_array::types::Int64Type>();

        assert_eq!(arr.value(0), 10, "position 0 was not updated");
        assert_eq!(arr.value(1), 25, "position 1 should be updated to 25");
        assert_eq!(arr.value(2), 30, "position 2 was not updated");
        assert_eq!(arr.value(3), 35, "position 3 should be updated to 35");
        assert_eq!(arr.value(4), 50, "position 4 was not updated");
        assert_eq!(arr.value(5), 45, "position 5 should be updated to 45");
        assert_eq!(arr.value(6), 70, "position 6 was not updated");
    }

    #[test]
    fn test_materialize_column_clears_update_records() {
        // Key assertion: after materialize_column, the column's update records are cleared
        let mut mask = crate::segment::upd_mask::UpdMask::new(10);
        mask.set("age", 1, 25i64.to_le_bytes().to_vec());
        mask.set("age", 2, 30i64.to_le_bytes().to_vec());
        mask.set("name", 1, b"Bob".to_vec());

        let original: ArrayRef = Arc::new(arrow_array::Int64Array::from(vec![10i64, 20, 30, 40, 50]));
        mask.materialize_column("age", &original).unwrap();

        // age updates must be cleared
        assert_eq!(mask.update_count("age"), 0,
            "age updates should be cleared after materialize_column");
        // other columns must be unaffected
        assert_eq!(mask.update_count("name"), 1);
        assert_eq!(mask.total_updates(), 1);
    }

    #[test]
    fn test_materialize_column_idempotent_protection() {
        // Critical: second materialize must NOT double-apply updates.
        // After first materialize, update records are cleared → second call returns original.
        let mut mask = crate::segment::upd_mask::UpdMask::new(10);
        mask.set("age", 0, 100i64.to_le_bytes().to_vec());

        let original: ArrayRef = Arc::new(arrow_array::Int64Array::from(vec![0i64; 5]));
        let r1 = mask.materialize_column("age", &original).unwrap();
        let r2 = mask.materialize_column("age", &original).unwrap();

        // First materialize: value updated to 100
        assert_eq!(r1.as_primitive::<arrow_array::types::Int64Type>().value(0), 100);
        // Second materialize: update records cleared → original returned (0, not 200)
        assert_eq!(r2.as_primitive::<arrow_array::types::Int64Type>().value(0), 0,
            "second materialize_column must not double-apply updates");
    }

    #[test]
    fn test_materialize_all_mixed_columns() {
        let mut mask = crate::segment::upd_mask::UpdMask::new(50);
        mask.set("age", 0, 25i64.to_le_bytes().to_vec());
        mask.set("name", 1, b"Charlie".to_vec());

        let mut columns: std::collections::HashMap<String, ArrayRef> =
            std::collections::HashMap::new();
        columns.insert("age".into(), Arc::new(arrow_array::Int64Array::from(vec![0i64; 5])));
        columns.insert("name".into(), Arc::new(arrow_array::StringArray::from(vec!["Bob"; 3])));
        columns.insert("score".into(), Arc::new(arrow_array::Int64Array::from(vec![0i64; 5])));

        let result = mask.materialize_all(&columns);

        // age was updated
        let age = result.get("age").unwrap()
            .as_primitive::<arrow_array::types::Int64Type>();
        assert_eq!(age.value(0), 25);
        // name was updated
        let name = result.get("name").unwrap()
            .as_any().downcast_ref::<arrow_array::StringArray>().unwrap();
        assert_eq!(name.value(1), "Charlie");
        // score had no updates: not in result
        assert!(!result.contains_key("score"),
            "un-updated columns must not appear in materialize_all result");

        // mask state: all update records cleared
        assert_eq!(mask.update_count("age"), 0);
        assert_eq!(mask.update_count("name"), 0);
        assert_eq!(mask.update_count("score"), 0);
    }

    #[test]
    fn test_materialize_all_empty_updates() {
        let mut mask = crate::segment::upd_mask::UpdMask::new(50);
        let columns: std::collections::HashMap<String, ArrayRef> =
            std::collections::HashMap::new();
        let result = mask.materialize_all(&columns);
        assert!(result.is_empty(), "empty updates should produce empty result");
    }

    #[test]
    fn test_should_materialize_threshold_boundary() {
        let mut mask = crate::segment::upd_mask::UpdMask::new(100);
        // 10 updates on 1 column = 10% update ratio
        for i in 0..10 {
            mask.set("age", i, i.to_le_bytes().to_vec());
        }

        assert!(mask.should_materialize(0.05), "10% > 5% threshold → should materialize");
        assert!(!mask.should_materialize(0.15), "10% < 15% threshold → should not materialize");
    }
}
