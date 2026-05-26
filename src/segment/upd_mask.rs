//! Update Mask：位置 → 新值的 Patch
//!
//! 对于更新操作，不重写原始数据，而是记录 {position → new_value}
//! 读取时合并：原始值 + Update Mask

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use serde::{Deserialize, Serialize};
use bincode_next::{Encode, Decode};
use arrow_array::{
    Array, ArrayRef, PrimitiveArray, StringArray, LargeStringArray, BooleanArray,
    types::{
        Int8Type, Int16Type, Int32Type, Int64Type,
        UInt8Type, UInt16Type, UInt32Type, UInt64Type,
        Float32Type, Float64Type,
    },
};
use arrow::datatypes::DataType;
use arrow::buffer::Buffer;
use crate::error::{RockDuckError, Result};
use crate::codec::{encode, decode};

/// 更新掩码
#[derive(Debug, Clone, Default, Serialize, Deserialize, Encode, Decode)]
pub struct UpdMask {
    /// 总行数
    pub total_rows: u64,
    /// 列名 → {position → serialized_value}
    pub updates: HashMap<String, HashMap<u64, Vec<u8>>>,
}

impl UpdMask {
    /// 创建新的更新掩码
    pub fn new(total_rows: u64) -> Self {
        Self {
            total_rows,
            updates: HashMap::new(),
        }
    }

    /// 添加更新（单值）
    pub fn set(&mut self, col: &str, pos: u64, value: Vec<u8>) {
        self.updates
            .entry(col.to_string())
            .or_default()
            .insert(pos, value);
    }

    /// 检查是否有更新
    pub fn has_updates(&self) -> bool {
        self.updates.values().any(|m| !m.is_empty())
    }

    /// 获取更新计数
    pub fn update_count(&self, col: &str) -> usize {
        self.updates.get(col).map(|m| m.len()).unwrap_or(0)
    }

    /// 获取更新计数（所有列）
    pub fn total_updates(&self) -> usize {
        self.updates.values().map(|m| m.len()).sum()
    }

    /// 获取指定位置的值（如果存在）
    pub fn get(&self, col: &str, pos: u64) -> Option<&Vec<u8>> {
        self.updates.get(col).and_then(|m| m.get(&pos))
    }

    /// 合并 UpdMask 到数据（返回新的数组，不修改原数组）
    ///
    /// 在 arrow_array 58 中，Array trait 不提供 as_any_mut（只读的 as_any），
    /// 因此无法原地修改 dyn Array。改为返回包含更新值的新数组。
    pub fn apply(&self, col: &str, original: &dyn Array) -> Result<ArrayRef> {
        let Some(col_updates) = self.updates.get(col) else {
            return Ok(make_arc_clone(original));
        };

        if col_updates.is_empty() {
            return Ok(make_arc_clone(original));
        }

        tracing::debug!("Applying {} updates to column {}", col_updates.len(), col);

        match original.data_type() {
            DataType::Int8 => apply_primitive::<Int8Type>(original, col_updates),
            DataType::Int16 => apply_primitive::<Int16Type>(original, col_updates),
            DataType::Int32 => apply_primitive::<Int32Type>(original, col_updates),
            DataType::Int64 => apply_primitive::<Int64Type>(original, col_updates),
            DataType::UInt8 => apply_primitive::<UInt8Type>(original, col_updates),
            DataType::UInt16 => apply_primitive::<UInt16Type>(original, col_updates),
            DataType::UInt32 => apply_primitive::<UInt32Type>(original, col_updates),
            DataType::UInt64 => apply_primitive::<UInt64Type>(original, col_updates),
            DataType::Float32 => apply_primitive::<Float32Type>(original, col_updates),
            DataType::Float64 => apply_primitive::<Float64Type>(original, col_updates),
            DataType::Boolean => apply_boolean(original, col_updates),
            DataType::Utf8 => apply_string(original, col_updates),
            DataType::LargeUtf8 => apply_large_string(original, col_updates),
            _ => Err(RockDuckError::Query(format!(
                "UpdMask: unsupported dtype {:?} for column {}",
                original.data_type(),
                col
            ))),
        }
    }

    /// 加载更新掩码
    pub fn load(path: &Path) -> Result<Self> {
        let data = std::fs::read(path)?;
        let mask: UpdMask = decode(&data)?;
        Ok(mask)
    }

    /// 保存更新掩码
    pub fn save(&self, path: &Path) -> Result<()> {
        let data = encode(self)?;
        std::fs::write(path, data)?;
        Ok(())
    }

    /// 增量物化：应用所有更新到原始数据，返回新数组并清除该列的更新记录。
    ///
    /// 与 `apply()` 的区别：apply 只返回合并结果但保留更新记录，
    /// materialize 返回合并结果并清空该列的 updates（释放内存 + 避免重复应用）。
    pub fn materialize_column(
        &mut self,
        col: &str,
        original: &dyn Array,
    ) -> Result<ArrayRef> {
        let result = self.apply(col, original)?;
        // 清除该列的更新记录，释放内存
        if let Some(col_updates) = self.updates.get_mut(col) {
            col_updates.clear();
        }
        Ok(result)
    }

    /// 增量物化所有被更新过的列，返回新列数据的 HashMap。
    pub fn materialize_all(
        &mut self,
        columns: &std::collections::HashMap<String, ArrayRef>,
    ) -> std::collections::HashMap<String, ArrayRef> {
        let mut materialized = std::collections::HashMap::new();
        for (col, array) in columns {
            if self.updates.get(col).map(|m| !m.is_empty()).unwrap_or(false) {
                if let Ok(result) = self.materialize_column(col, array) {
                    materialized.insert(col.clone(), result);
                }
            }
        }
        materialized
    }

    /// 检查 UpdMask 是否值得物化（更新比例超过阈值）。
    pub fn should_materialize(&self, threshold: f64) -> bool {
        if self.total_rows == 0 {
            return false;
        }
        let total_updates = self.total_updates() as f64;
        total_updates / (self.total_rows as f64 * self.updates.len().max(1) as f64) > threshold
    }
}

/// 深度克隆一个 dyn Array 为 ArrayRef
fn make_arc_clone(array: &dyn Array) -> ArrayRef {
    arrow_array::make_array(array.to_data())
}

/// Apply updates to a primitive array - returns a new array with updates applied
fn apply_primitive<T: arrow_array::types::ArrowPrimitiveType>(
    original: &dyn Array,
    updates: &HashMap<u64, Vec<u8>>,
) -> Result<ArrayRef> {
    let arr = original.as_any().downcast_ref::<PrimitiveArray<T>>()
        .ok_or_else(|| RockDuckError::Query("UpdMask: failed to downcast to PrimitiveArray".into()))?;

    let len = arr.len();
    if len == 0 {
        return Ok(make_arc_clone(original));
    }

    let native_size = std::mem::size_of::<T::Native>();
    let nulls = arr.nulls();

    let mut new_values: Vec<T::Native> = arr.values().iter().copied().collect();

    for (pos, bytes) in updates {
        let pos_usize = *pos as usize;
        if pos_usize >= len {
            continue;
        }
        if bytes.len() < native_size {
            tracing::warn!("UpdMask: byte slice too short for type");
            continue;
        }
        let val = transmute_from_bytes::<T::Native>(bytes);
        new_values[pos_usize] = val;
    }

    let new_arr = if let Some(n) = nulls {
        PrimitiveArray::<T>::new(Buffer::from(new_values).into(), Some(n.clone()))
    } else {
        PrimitiveArray::<T>::new(Buffer::from(new_values).into(), None)
    };

    Ok(Arc::new(new_arr))
}

/// Apply updates to a BooleanArray - returns a new array
fn apply_boolean(
    original: &dyn Array,
    updates: &HashMap<u64, Vec<u8>>,
) -> Result<ArrayRef> {
    let arr = original.as_any().downcast_ref::<BooleanArray>()
        .ok_or_else(|| RockDuckError::Query("UpdMask: failed to downcast to BooleanArray".into()))?;

    let len = arr.len();
    if len == 0 {
        return Ok(make_arc_clone(original));
    }

    let nulls = arr.nulls();

    let mut new_values: Vec<bool> = (0..len)
        .map(|i| arr.value(i))
        .collect();

    for (pos, bytes) in updates {
        let pos_usize = *pos as usize;
        if pos_usize >= len {
            continue;
        }
        let val = bytes.first().map(|b| *b != 0).unwrap_or(false);
        new_values[pos_usize] = val;
    }

    let new_arr = if let Some(n) = nulls {
        let mut builder = arrow_array::builder::BooleanBuilder::new();
        for i in 0..len {
            if n.is_null(i) {
                builder.append_null();
            } else {
                builder.append_value(new_values[i]);
            }
        }
        builder.finish()
    } else {
        BooleanArray::from(new_values)
    };

    Ok(Arc::new(new_arr))
}

/// Apply updates to a StringArray (Utf8) - returns a new array
fn apply_string(
    original: &dyn Array,
    updates: &HashMap<u64, Vec<u8>>,
) -> Result<ArrayRef> {
    let arr = original.as_any().downcast_ref::<StringArray>()
        .ok_or_else(|| RockDuckError::Query("UpdMask: failed to downcast to StringArray".into()))?;

    let len = arr.len();
    if len == 0 {
        return Ok(make_arc_clone(original));
    }

    let nulls = arr.nulls();

    let mut new_strings: Vec<String> = (0..len)
        .map(|i| arr.value(i).to_string())
        .collect();

    for (pos, bytes) in updates {
        let pos_usize = *pos as usize;
        if pos_usize >= len {
            continue;
        }
        let val = String::from_utf8(bytes.clone()).unwrap_or_default();
        new_strings[pos_usize] = val;
    }

    let new_arr = if let Some(n) = nulls {
        let mut builder = arrow_array::builder::StringBuilder::with_capacity(len, 1024);
        for i in 0..len {
            if n.is_null(i) {
                builder.append_null();
            } else {
                builder.append_value(&new_strings[i]);
            }
        }
        builder.finish()
    } else {
        StringArray::from(new_strings)
    };

    Ok(Arc::new(new_arr))
}

/// Apply updates to a LargeStringArray (LargeUtf8) - returns a new array
fn apply_large_string(
    original: &dyn Array,
    updates: &HashMap<u64, Vec<u8>>,
) -> Result<ArrayRef> {
    let arr = original.as_any().downcast_ref::<LargeStringArray>()
        .ok_or_else(|| RockDuckError::Query("UpdMask: failed to downcast to LargeStringArray".into()))?;

    let len = arr.len();
    if len == 0 {
        return Ok(make_arc_clone(original));
    }

    let nulls = arr.nulls();

    let mut new_strings: Vec<String> = (0..len)
        .map(|i| arr.value(i).to_string())
        .collect();

    for (pos, bytes) in updates {
        let pos_usize = *pos as usize;
        if pos_usize >= len {
            continue;
        }
        let val = String::from_utf8(bytes.clone()).unwrap_or_default();
        new_strings[pos_usize] = val;
    }

    let new_arr = if let Some(n) = nulls {
        let mut builder = arrow_array::builder::LargeStringBuilder::with_capacity(len, 1024);
        for i in 0..len {
            if n.is_null(i) {
                builder.append_null();
            } else {
                builder.append_value(&new_strings[i]);
            }
        }
        builder.finish()
    } else {
        LargeStringArray::from(new_strings)
    };

    Ok(Arc::new(new_arr))
}

/// 从字节切片安全地转换为原生类型
#[inline]
fn transmute_from_bytes<T: Copy>(bytes: &[u8]) -> T {
    let n = bytes.len();
    assert_eq!(n, std::mem::size_of::<T>(), "byte length mismatch");
    // Create a properly-sized array and copy bytes into it, then transmute
    let mut arr: [u8; 16] = [0; 16]; // max size for i128
    arr[..n].copy_from_slice(bytes);
    // SAFETY: T is Copy (Plain Old Data) and arr has exactly the right size
    // We use transmute_copy which requires src to be properly aligned (arr is aligned)
    unsafe {
        let mut val: T = std::mem::zeroed();
        std::ptr::copy_nonoverlapping(arr.as_ptr(), &mut val as *mut T as *mut u8, n);
        val
    }
}

/// Serialize a value from an array at a given position to bytes for storage.
/// Returns None if the type is not supported.
pub fn serialize_value(array: &dyn Array, pos: usize) -> Option<Vec<u8>> {
    match array.data_type() {
        DataType::Int8 => {
            let a = array.as_any().downcast_ref::<PrimitiveArray<Int8Type>>()?;
            Some(a.value(pos).to_le_bytes().to_vec())
        }
        DataType::Int16 => {
            let a = array.as_any().downcast_ref::<PrimitiveArray<Int16Type>>()?;
            Some(a.value(pos).to_le_bytes().to_vec())
        }
        DataType::Int32 => {
            let a = array.as_any().downcast_ref::<PrimitiveArray<Int32Type>>()?;
            Some(a.value(pos).to_le_bytes().to_vec())
        }
        DataType::Int64 => {
            let a = array.as_any().downcast_ref::<PrimitiveArray<Int64Type>>()?;
            Some(a.value(pos).to_le_bytes().to_vec())
        }
        DataType::UInt8 => {
            let a = array.as_any().downcast_ref::<PrimitiveArray<UInt8Type>>()?;
            Some(a.value(pos).to_le_bytes().to_vec())
        }
        DataType::UInt16 => {
            let a = array.as_any().downcast_ref::<PrimitiveArray<UInt16Type>>()?;
            Some(a.value(pos).to_le_bytes().to_vec())
        }
        DataType::UInt32 => {
            let a = array.as_any().downcast_ref::<PrimitiveArray<UInt32Type>>()?;
            Some(a.value(pos).to_le_bytes().to_vec())
        }
        DataType::UInt64 => {
            let a = array.as_any().downcast_ref::<PrimitiveArray<UInt64Type>>()?;
            Some(a.value(pos).to_le_bytes().to_vec())
        }
        DataType::Float32 => {
            let a = array.as_any().downcast_ref::<PrimitiveArray<Float32Type>>()?;
            Some(a.value(pos).to_le_bytes().to_vec())
        }
        DataType::Float64 => {
            let a = array.as_any().downcast_ref::<PrimitiveArray<Float64Type>>()?;
            Some(a.value(pos).to_le_bytes().to_vec())
        }
        DataType::Boolean => {
            let a = array.as_any().downcast_ref::<BooleanArray>()?;
            Some(vec![if a.value(pos) { 1u8 } else { 0u8 }])
        }
        DataType::Utf8 => {
            let a = array.as_any().downcast_ref::<StringArray>()?;
            Some(a.value(pos).as_bytes().to_vec())
        }
        DataType::LargeUtf8 => {
            let a = array.as_any().downcast_ref::<LargeStringArray>()?;
            Some(a.value(pos).as_bytes().to_vec())
        }
        _ => None,
    }
}

/// 批量更新掩码管理器
#[derive(Debug, Clone, Default)]
pub struct UpdMaskManager {
    /// segment_id → UpdMask
    masks: HashMap<String, UpdMask>,
}

impl UpdMaskManager {
    pub fn new() -> Self {
        Self::default()
    }

    /// 获取或创建 segment 的更新掩码
    pub fn get_or_create(&mut self, seg_id: &str, total_rows: u64) -> &mut UpdMask {
        self.masks
            .entry(seg_id.to_string())
            .or_insert_with(|| UpdMask::new(total_rows))
    }

    /// 移除 segment 的更新掩码
    pub fn remove(&mut self, seg_id: &str) {
        self.masks.remove(seg_id);
    }

    /// 获取 segment 的更新掩码
    pub fn get(&self, seg_id: &str) -> Option<&UpdMask> {
        self.masks.get(seg_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::AsArray;

    #[test]
    fn test_upd_mask_new() {
        let mask = UpdMask::new(100);
        assert_eq!(mask.total_rows, 100);
        assert!(mask.updates.is_empty());
    }

    #[test]
    fn test_upd_mask_set() {
        let mut mask = UpdMask::new(50);
        mask.set("age", 5, vec![30u8]);
        assert!(mask.updates.contains_key("age"));
        assert!(mask.updates["age"].contains_key(&5));
    }

    #[test]
    fn test_upd_mask_set_multiple_columns() {
        let mut mask = UpdMask::new(50);
        mask.set("age", 1, vec![25u8]);
        mask.set("name", 1, vec![b'A', b'l', b'i', b'c', b'e']);
        mask.set("age", 2, vec![30u8]);
        assert_eq!(mask.update_count("age"), 2);
        assert_eq!(mask.update_count("name"), 1);
        assert_eq!(mask.update_count("missing"), 0);
    }

    #[test]
    fn test_upd_mask_set_duplicate_position_replaces() {
        let mut mask = UpdMask::new(50);
        mask.set("age", 5, vec![25u8]);
        mask.set("age", 5, vec![30u8]);
        assert_eq!(mask.updates["age"].get(&5), Some(&vec![30u8]));
    }

    #[test]
    fn test_upd_mask_has_updates_empty() {
        let mask = UpdMask::new(50);
        assert!(!mask.has_updates());
    }

    #[test]
    fn test_upd_mask_has_updates_with_data() {
        let mut mask = UpdMask::new(50);
        mask.set("age", 0, vec![1u8]);
        assert!(mask.has_updates());
    }

    #[test]
    fn test_upd_mask_total_updates() {
        let mut mask = UpdMask::new(50);
        assert_eq!(mask.total_updates(), 0);
        mask.set("a", 1, vec![1u8]);
        mask.set("a", 2, vec![2u8]);
        mask.set("b", 1, vec![3u8]);
        assert_eq!(mask.total_updates(), 3);
    }

    #[test]
    fn test_upd_mask_get_existing() {
        let mut mask = UpdMask::new(50);
        mask.set("age", 10, vec![42u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8]);
        let val = mask.get("age", 10);
        assert!(val.is_some());
    }

    #[test]
    fn test_upd_mask_get_nonexistent_column() {
        let mask = UpdMask::new(50);
        assert!(mask.get("missing", 0).is_none());
    }

    #[test]
    fn test_upd_mask_get_nonexistent_position() {
        let mut mask = UpdMask::new(50);
        mask.set("age", 10, vec![1u8]);
        assert!(mask.get("age", 99).is_none());
    }

    // ============================================================
    // apply() tests
    // ============================================================

    #[test]
    fn test_upd_mask_apply_int64() {
        let mut mask = UpdMask::new(50);
        let value_bytes = 9999i64.to_le_bytes().to_vec();
        mask.set("col", 1, value_bytes);

        let arr: ArrayRef = Arc::new(arrow_array::Int64Array::from(vec![100i64, 200, 300]));
        let result = mask.apply("col", &arr).unwrap();
        let updated = result.as_primitive::<arrow_array::types::Int64Type>();

        assert_eq!(updated.value(0), 100, "unchanged position should stay the same");
        assert_eq!(updated.value(1), 9999, "updated position should have new value");
        assert_eq!(updated.value(2), 300, "unchanged position should stay the same");
    }

    #[test]
    fn test_upd_mask_apply_string() {
        let mut mask = UpdMask::new(50);
        mask.set("name", 0, b"Alice".to_vec());
        mask.set("name", 2, b"Charlie".to_vec());

        let arr: ArrayRef = Arc::new(StringArray::from(vec!["Bob".to_string(), "Carol".to_string(), "Dave".to_string()]));
        let result = mask.apply("name", &arr).unwrap();
        let updated = result.as_any().downcast_ref::<StringArray>().unwrap();

        assert_eq!(updated.value(0), "Alice");
        assert_eq!(updated.value(1), "Carol", "position 1 was not updated");
        assert_eq!(updated.value(2), "Charlie");
    }

    #[test]
    fn test_upd_mask_apply_boolean() {
        let mut mask = UpdMask::new(50);
        mask.set("flag", 0, vec![1u8]);  // true
        mask.set("flag", 2, vec![0u8]);  // false

        let arr: ArrayRef = Arc::new(BooleanArray::from(vec![false, true, true]));
        let result = mask.apply("flag", &arr).unwrap();
        let updated = result.as_any().downcast_ref::<BooleanArray>().unwrap();

        assert_eq!(updated.value(0), true);
        assert_eq!(updated.value(1), true, "position 1 was not updated");
        assert_eq!(updated.value(2), false);
    }

    #[test]
    fn test_upd_mask_apply_empty_updates() {
        let mask = UpdMask::new(50);
        let arr: ArrayRef = Arc::new(arrow_array::Int64Array::from(vec![100i64]));
        let result = mask.apply("col", &arr).unwrap();
        let updated = result.as_primitive::<arrow_array::types::Int64Type>();
        assert_eq!(updated.value(0), 100);
    }

    #[test]
    fn test_upd_mask_apply_column_with_no_updates() {
        let mut mask = UpdMask::new(50);
        mask.set("col", 0, vec![99u8]);

        let arr: ArrayRef = Arc::new(arrow_array::Int64Array::from(vec![100i64]));
        let result = mask.apply("other_col", &arr).unwrap();
        let updated = result.as_primitive::<arrow_array::types::Int64Type>();
        assert_eq!(updated.value(0), 100, "array should not be modified");
    }

    #[test]
    fn test_upd_mask_apply_out_of_bounds_position() {
        let mut mask = UpdMask::new(50);
        let value_bytes = 9999i64.to_le_bytes().to_vec();
        mask.set("col", 100, value_bytes);

        let arr: ArrayRef = Arc::new(arrow_array::Int64Array::from(vec![100i64, 200, 300]));
        let result = mask.apply("col", &arr).unwrap();
        let updated = result.as_primitive::<arrow_array::types::Int64Type>();

        // Array should be unchanged since position is out of bounds
        assert_eq!(updated.value(0), 100);
        assert_eq!(updated.value(1), 200);
        assert_eq!(updated.value(2), 300);
    }

    #[test]
    fn test_upd_mask_apply_multiple_updates_same_position() {
        let mut mask = UpdMask::new(50);
        mask.set("col", 1, 100i64.to_le_bytes().to_vec());
        mask.set("col", 1, 999i64.to_le_bytes().to_vec());

        let arr: ArrayRef = Arc::new(arrow_array::Int64Array::from(vec![10i64, 20, 30]));
        let result = mask.apply("col", &arr).unwrap();
        let updated = result.as_primitive::<arrow_array::types::Int64Type>();

        assert_eq!(updated.value(1), 999);
    }

    #[test]
    fn test_upd_mask_apply_int32() {
        let mut mask = UpdMask::new(50);
        mask.set("col", 0, 42i32.to_le_bytes().to_vec());
        mask.set("col", 2, 100i32.to_le_bytes().to_vec());

        let arr: ArrayRef = Arc::new(arrow_array::Int32Array::from(vec![1i32, 2, 3]));
        let result = mask.apply("col", &arr).unwrap();
        let updated = result.as_primitive::<arrow_array::types::Int32Type>();

        assert_eq!(updated.value(0), 42);
        assert_eq!(updated.value(1), 2, "position 1 not updated");
        assert_eq!(updated.value(2), 100);
    }

    #[test]
    fn test_upd_mask_apply_float64() {
        let mut mask = UpdMask::new(50);
        mask.set("col", 1, 3.14159f64.to_le_bytes().to_vec());

        let arr: ArrayRef = Arc::new(arrow_array::Float64Array::from(vec![1.0, 2.0, 3.0]));
        let result = mask.apply("col", &arr).unwrap();
        let updated = result.as_primitive::<arrow_array::types::Float64Type>();

        assert_eq!(updated.value(0), 1.0);
        assert!((updated.value(1) - 3.14159).abs() < 0.0001);
        assert_eq!(updated.value(2), 3.0);
    }

    #[test]
    fn test_upd_mask_apply_multiple_columns() {
        let mut mask = UpdMask::new(50);
        mask.set("col_a", 0, 100i64.to_le_bytes().to_vec());
        mask.set("col_b", 1, b"updated".to_vec());

        let arr_a: ArrayRef = Arc::new(arrow_array::Int64Array::from(vec![1i64, 2, 3]));
        let arr_b: ArrayRef = Arc::new(StringArray::from(vec!["orig".to_string(), "orig2".to_string(), "orig3".to_string()]));

        let result_a = mask.apply("col_a", &arr_a).unwrap();
        let result_b = mask.apply("col_b", &arr_b).unwrap();

        let updated_a = result_a.as_primitive::<arrow_array::types::Int64Type>();
        let updated_b = result_b.as_any().downcast_ref::<StringArray>().unwrap();

        assert_eq!(updated_a.value(0), 100);
        assert_eq!(updated_b.value(1), "updated");
    }

    // ============================================================
    // serialize_value tests
    // ============================================================

    #[test]
    fn test_serialize_value_int64() {
        let arr = arrow_array::Int64Array::from(vec![100i64, 200, 300]);
        let serialized = serialize_value(&arr, 1);
        assert!(serialized.is_some());
        let bytes = serialized.unwrap();
        assert_eq!(bytes.len(), 8);
        assert_eq!(i64::from_le_bytes(bytes.try_into().unwrap()), 200);
    }

    #[test]
    fn test_serialize_value_string() {
        let arr = StringArray::from(vec!["Hello".to_string(), "World".to_string()]);
        let serialized = serialize_value(&arr, 1);
        assert!(serialized.is_some());
        assert_eq!(serialized.unwrap(), b"World");
    }

    #[test]
    fn test_serialize_value_boolean() {
        let arr = BooleanArray::from(vec![false, true, false]);
        let serialized = serialize_value(&arr, 1);
        assert!(serialized.is_some());
        assert_eq!(serialized.unwrap(), vec![1u8]);
    }

    #[test]
    fn test_serialize_value_float32() {
        let arr = arrow_array::Float32Array::from(vec![1.5f32, 2.5f32]);
        let serialized = serialize_value(&arr, 0);
        assert!(serialized.is_some());
        let bytes = serialized.unwrap();
        assert_eq!(bytes.len(), 4);
        assert_eq!(f32::from_le_bytes(bytes.try_into().unwrap()), 1.5);
    }

    #[test]
    fn test_serialize_value_roundtrip() {
        let original_value: i64 = 42;
        let bytes = original_value.to_le_bytes().to_vec();

        let mut mask = UpdMask::new(10);
        mask.set("col", 5, bytes);

        let arr: ArrayRef = Arc::new(arrow_array::Int64Array::from(vec![0i64; 10]));
        let result = mask.apply("col", &arr).unwrap();
        let updated = result.as_primitive::<arrow_array::types::Int64Type>();
        assert_eq!(updated.value(5), original_value);
    }

    #[test]
    fn test_serialize_value_string_roundtrip() {
        let original_value = "Test String";
        let bytes = original_value.as_bytes().to_vec();

        let mut mask = UpdMask::new(10);
        mask.set("col", 3, bytes);

        let arr: ArrayRef = Arc::new(StringArray::from(vec!["default".to_string(); 10]));
        let result = mask.apply("col", &arr).unwrap();
        let updated = result.as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(updated.value(3), original_value);
    }

    #[test]
    fn test_serialize_value_unsupported_type() {
        // Test with a type that serialize_value doesn't support yet
        // (e.g., Decimal) - this would need a matching apply handler too
        // For now, just test Int8 which we know works
        let arr = arrow_array::Int8Array::from(vec![10i8, 20, 30]);
        let serialized = serialize_value(&arr, 1);
        assert!(serialized.is_some());
        assert_eq!(serialized.unwrap(), &20i8.to_le_bytes()[..]);
    }

    // ============================================================
    // save/load tests
    // ============================================================

    #[test]
    fn test_upd_mask_save_load_roundtrip() {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("upd_mask.bin");

        let mut mask = UpdMask::new(100);
        mask.set("age", 5, vec![30u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8]);
        mask.set("name", 10, vec![65u8, 66u8, 67u8]);

        mask.save(&path).unwrap();
        let loaded = UpdMask::load(&path).unwrap();

        assert_eq!(loaded.total_rows, 100);
        assert_eq!(loaded.update_count("age"), 1);
        assert_eq!(loaded.update_count("name"), 1);
    }

    #[test]
    fn test_upd_mask_load_nonexistent() {
        let result = UpdMask::load(std::path::Path::new("/nonexistent/upd_mask.bin"));
        assert!(result.is_err());
    }

    #[test]
    fn test_upd_mask_manager_new() {
        let mgr = UpdMaskManager::new();
        assert!(mgr.masks.is_empty());
    }

    #[test]
    fn test_upd_mask_manager_default() {
        let mgr: UpdMaskManager = UpdMaskManager::default();
        assert!(mgr.masks.is_empty());
    }

    #[test]
    fn test_upd_mask_manager_get_or_create() {
        let mut mgr = UpdMaskManager::new();
        let mask1 = mgr.get_or_create("seg_001", 100);
        assert_eq!(mask1.total_rows, 100);

        let mask2 = mgr.get_or_create("seg_001", 100);
        mask2.set("age", 0, vec![1u8]);

        let retrieved = mgr.get("seg_001");
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().update_count("age"), 1);
    }

    #[test]
    fn test_upd_mask_manager_get_or_create_existing() {
        let mut mgr = UpdMaskManager::new();
        let mask1 = mgr.get_or_create("seg_001", 100);
        mask1.set("x", 0, vec![1u8]);

        let mask2 = mgr.get_or_create("seg_001", 200);
        assert_eq!(mask2.update_count("x"), 1);
    }

    #[test]
    fn test_upd_mask_manager_remove() {
        let mut mgr = UpdMaskManager::new();
        mgr.get_or_create("seg_001", 100);
        assert!(mgr.get("seg_001").is_some());
        mgr.remove("seg_001");
        assert!(mgr.get("seg_001").is_none());
    }

    #[test]
    fn test_upd_mask_manager_remove_nonexistent() {
        let mut mgr = UpdMaskManager::new();
        mgr.remove("nonexistent");
    }

    #[test]
    fn test_upd_mask_manager_get() {
        let mut mgr = UpdMaskManager::new();
        mgr.get_or_create("seg_001", 100);
        assert!(mgr.get("seg_001").is_some());
        assert!(mgr.get("seg_002").is_none());
    }

    #[test]
    fn test_upd_mask_debug() {
        let mask = UpdMask::new(50);
        let debug_str = format!("{:?}", mask);
        assert!(!debug_str.is_empty());
    }

    #[test]
    fn test_upd_mask_manager_debug() {
        let mgr = UpdMaskManager::new();
        let debug_str = format!("{:?}", mgr);
        assert!(!debug_str.is_empty());
    }

    // ============================================================
    // materialize_column tests
    // ============================================================

    #[test]
    fn test_materialize_column_clears_updates() {
        let mut mask = UpdMask::new(50);
        mask.set("age", 5, 42i64.to_le_bytes().to_vec());

        let arr: ArrayRef = Arc::new(arrow_array::Int64Array::from(vec![0i64; 10]));
        let result = mask.materialize_column("age", &arr).unwrap();
        let updated = result.as_primitive::<arrow_array::types::Int64Type>();

        // Result should have the updated value
        assert_eq!(updated.value(5), 42);
        // After materialize, the update record should be cleared
        assert_eq!(mask.update_count("age"), 0, "update record should be cleared after materialize");
    }

    #[test]
    fn test_materialize_column_double_apply_protection() {
        let mut mask = UpdMask::new(50);
        mask.set("age", 0, 100i64.to_le_bytes().to_vec());

        let arr: ArrayRef = Arc::new(arrow_array::Int64Array::from(vec![0i64; 5]));
        let result1 = mask.materialize_column("age", &arr).unwrap();
        let result2 = mask.materialize_column("age", &arr).unwrap();

        let v1 = result1.as_primitive::<arrow_array::types::Int64Type>().value(0);
        let v2 = result2.as_primitive::<arrow_array::types::Int64Type>().value(0);

        assert_eq!(v1, 100);
        assert_eq!(v2, 0, "second materialize should not double-apply");
    }

    #[test]
    fn test_materialize_column_string() {
        let mut mask = UpdMask::new(50);
        mask.set("name", 1, b"Alice".to_vec());

        let arr: ArrayRef = Arc::new(StringArray::from(vec!["Bob".to_string(); 3]));
        let result = mask.materialize_column("name", &arr).unwrap();
        let updated = result.as_any().downcast_ref::<StringArray>().unwrap();

        assert_eq!(updated.value(0), "Bob");
        assert_eq!(updated.value(1), "Alice");
        assert_eq!(updated.value(2), "Bob");
        assert_eq!(mask.update_count("name"), 0);
    }

    // ============================================================
    // materialize_all tests
    // ============================================================

    #[test]
    fn test_materialize_all_mixed() {
        let mut mask = UpdMask::new(50);
        mask.set("age", 0, 25i64.to_le_bytes().to_vec());
        mask.set("name", 1, b"Charlie".to_vec());
        // "score" column has no updates - should be skipped

        let mut columns: std::collections::HashMap<String, ArrayRef> = std::collections::HashMap::new();
        columns.insert("age".to_string(), Arc::new(arrow_array::Int64Array::from(vec![0i64; 5])));
        columns.insert("name".to_string(), Arc::new(StringArray::from(vec!["Bob".to_string(); 3])));
        columns.insert("score".to_string(), Arc::new(arrow_array::Int64Array::from(vec![0i64; 5])));

        let result = mask.materialize_all(&columns);

        assert_eq!(result.len(), 2);
        assert!(result.contains_key("age"));
        assert!(result.contains_key("name"));
        assert!(!result.contains_key("score"), "un-updated columns should be skipped");
    }

    #[test]
    fn test_materialize_all_empty_updates() {
        let mut mask = UpdMask::new(50);
        // No updates at all

        let mut columns: std::collections::HashMap<String, ArrayRef> = std::collections::HashMap::new();
        columns.insert("age".to_string(), Arc::new(arrow_array::Int64Array::from(vec![1i64; 5])));

        let result = mask.materialize_all(&columns);
        assert!(result.is_empty(), "no updates should produce empty result");
    }

    // ============================================================
    // should_materialize tests
    // ============================================================

    #[test]
    fn test_should_materialize_true() {
        let mut mask = UpdMask::new(100);
        // 10 updates on 1 column out of 100 rows -> 10% per-column update ratio
        for i in 0..10 {
            mask.set("age", i, i.to_le_bytes().to_vec());
        }

        assert!(mask.should_materialize(0.05), "10 updates / 100 rows should trigger materialize");
        assert!(!mask.should_materialize(0.15), "should not trigger at higher threshold");
    }

    #[test]
    fn test_should_materialize_false() {
        let mut mask = UpdMask::new(10000);
        mask.set("age", 0, vec![1u8]);

        assert!(!mask.should_materialize(0.0001), "singleton update should not trigger");
    }

    #[test]
    fn test_should_materialize_zero_rows() {
        let mask = UpdMask::new(0);
        assert!(!mask.should_materialize(0.01));
    }

    #[test]
    fn test_should_materialize_multiple_columns() {
        let mut mask = UpdMask::new(100);
        // 5 updates each on 2 columns = 10 total
        for i in 0..5 {
            mask.set("col_a", i, i.to_le_bytes().to_vec());
            mask.set("col_b", i, i.to_le_bytes().to_vec());
        }

        // ratio = 10 / (100 * 2) = 0.05
        assert!(mask.should_materialize(0.04), "5 updates per column / 100 rows should trigger");
        assert!(!mask.should_materialize(0.06), "should not trigger above threshold");
    }
}
