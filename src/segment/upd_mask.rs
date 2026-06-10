//! Update Mask：位置 → 新值的 Patch
//!
//! 对于更新操作，不重写原始数据，而是记录 {position → new_value}
//! 读取时合并：原始值 + Update Mask
//!
//! ## Code Generation via Macro
//!
//! The `apply()` method uses `apply_primitive_body!` and `apply_variable_body!` macros
//! to generate DataType branches. Each branch follows an identical pattern — the macro
//! eliminates copy-paste across 14 numeric DataType variants.

use crate::codec::{decode, encode};
use crate::error::Result;
use arrow_array::{
    builder::{
        BinaryBuilder, BooleanBuilder, Float32Builder, Float64Builder, Int16Builder, Int32Builder,
        Int64Builder, Int8Builder, LargeBinaryBuilder, LargeStringBuilder, StringBuilder,
        UInt16Builder, UInt32Builder, UInt64Builder, UInt8Builder,
    },
    cast::AsArray,
    types::{
        Float32Type, Float64Type, Int16Type, Int32Type, Int64Type, Int8Type, UInt16Type,
        UInt32Type, UInt64Type, UInt8Type,
    },
    Array, ArrayRef,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

#[allow(dead_code)]
static UPDMASK_DISCARDED_TOTAL: AtomicU64 = AtomicU64::new(0);

#[allow(dead_code)]
const MAX_UPDATES_BYTES: usize = 64 * 1024 * 1024;

// =============================================================================
// Deserialization helpers (used by the apply() macro)
// =============================================================================

#[allow(dead_code)]
fn bytes_to_i64(b: &[u8]) -> Option<i64> {
    if b.len() < 8 {
        return None;
    }
    let mut buf = [0u8; 8];
    buf.copy_from_slice(&b[..8]);
    Some(i64::from_le_bytes(buf))
}

#[allow(dead_code)]
fn bytes_to_i32(b: &[u8]) -> Option<i32> {
    if b.len() < 4 {
        return None;
    }
    let mut buf = [0u8; 4];
    buf.copy_from_slice(&b[..4]);
    Some(i32::from_le_bytes(buf))
}

#[allow(dead_code)]
fn bytes_to_i16(b: &[u8]) -> Option<i16> {
    if b.is_empty() {
        return None;
    }
    // For single byte, just cast. For 2+ bytes, use full 16-bit read.
    Some(if b.len() >= 2 {
        let mut buf = [0u8; 2];
        buf.copy_from_slice(&b[..2]);
        i16::from_le_bytes(buf)
    } else {
        b[0] as i16
    })
}

#[allow(dead_code)]
fn bytes_to_i8(b: &[u8]) -> Option<i8> {
    b.first().map(|&b| b as i8)
}

#[allow(dead_code)]
fn bytes_to_u64(b: &[u8]) -> Option<u64> {
    if b.len() < 8 {
        return None;
    }
    let mut buf = [0u8; 8];
    buf.copy_from_slice(&b[..8]);
    Some(u64::from_le_bytes(buf))
}

#[allow(dead_code)]
fn bytes_to_u32(b: &[u8]) -> Option<u32> {
    if b.len() < 4 {
        return None;
    }
    let mut buf = [0u8; 4];
    buf.copy_from_slice(&b[..4]);
    Some(u32::from_le_bytes(buf))
}

#[allow(dead_code)]
fn bytes_to_u16(b: &[u8]) -> Option<u16> {
    if b.len() < 2 {
        return None;
    }
    let mut buf = [0u8; 2];
    buf.copy_from_slice(&b[..2]);
    Some(u16::from_le_bytes(buf))
}

#[allow(dead_code)]
fn bytes_to_u8(b: &[u8]) -> Option<u8> {
    b.first().copied()
}

#[allow(dead_code)]
fn bytes_to_f64(b: &[u8]) -> Option<f64> {
    if b.len() < 8 {
        return None;
    }
    let mut buf = [0u8; 8];
    buf.copy_from_slice(&b[..8]);
    Some(f64::from_le_bytes(buf))
}

#[allow(dead_code)]
fn bytes_to_f32(b: &[u8]) -> Option<f32> {
    if b.len() < 4 {
        return None;
    }
    let mut buf = [0u8; 4];
    buf.copy_from_slice(&b[..4]);
    Some(f32::from_le_bytes(buf))
}

// =============================================================================
// UpdMask struct
// =============================================================================

/// 更新掩码
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct UpdMask {
    /// 总行数
    pub total_rows: u64,
    /// 列名 → {position → serialized_value}
    pub updates: HashMap<String, HashMap<u64, Vec<u8>>>,
    /// Current total serialized size of all updates in bytes.
    /// Maintained incrementally on set/remove for O(1) budget checks.
    /// Recomputed on deserialization via `recompute_bytes()`.
    #[serde(skip, default = "UpdMask::zero_current_bytes")]
    current_bytes: usize,
}

impl Default for UpdMask {
    #[allow(dead_code)]
    fn default() -> Self {
        Self {
            total_rows: 0,
            updates: HashMap::new(),
            current_bytes: 0,
        }
    }
}

impl UpdMask {
    #[allow(dead_code)]
    fn zero_current_bytes() -> usize {
        0
    }

    /// 创建新的更新掩码
    #[allow(dead_code)]
    pub fn new(total_rows: u64) -> Self {
        Self {
            total_rows,
            updates: HashMap::new(),
            current_bytes: 0,
        }
    }

    /// 添加更新（单值）
    ///
    /// ## Budget Protection (Intentional Design)
    ///
    /// When the memory budget is exceeded (MAX_UPDATES_BYTES = 64MB), this function
    /// drops the update silently and emits a `tracing::warn!`. This is NOT a bug —
    /// it's an intentional guard against unbounded memory growth from runaway update patterns.
    /// Monitor for these warnings in production to detect anomalous workloads.
    ///
    /// ## O(1) Budget Check
    ///
    /// `current_bytes` is maintained incrementally via a counter. Each `set()` call:
    /// 1. Computes the entry's serialized byte size once.
    /// 2. Checks the pre-computed `current_bytes` counter (O(1)) against the budget.
    /// 3. On successful insert, adds the new entry's size and subtracts the old value's size (if replaced).
    #[allow(dead_code)]
    pub fn set(&mut self, col: &str, pos: u64, value: Vec<u8>) -> bool {
        let entry_size = std::mem::size_of::<String>()
            + col.len()
            + std::mem::size_of::<u64>()
            + std::mem::size_of::<Vec<u8>>()
            + value.len();

        // O(1) budget check using the pre-maintained counter
        if self.current_bytes + entry_size > MAX_UPDATES_BYTES {
            let discarded_total = UPDMASK_DISCARDED_TOTAL.fetch_add(1, Ordering::Relaxed) + 1;
            tracing::warn!(
                target: "upd_mask",
                discarded_total,
                total_rows = self.total_rows,
                "update dropped: budget exceeded ({}/{} bytes) for column '{}' at position {}",
                self.current_bytes + entry_size,
                MAX_UPDATES_BYTES,
                col,
                pos
            );
            return false;
        }

        // Compute old entry size for decrement (if position already has an update)
        let old_entry_size: usize = self
            .updates
            .get(col)
            .and_then(|m| m.get(&pos))
            .map(|v| {
                std::mem::size_of::<String>()
                    + col.len()
                    + std::mem::size_of::<u64>()
                    + std::mem::size_of::<Vec<u8>>()
                    + v.len()
            })
            .unwrap_or(0);

        self.updates
            .entry(col.to_string())
            .or_default()
            .insert(pos, value);

        // Incremental update: add new entry, subtract old (if replaced)
        self.current_bytes += entry_size;
        self.current_bytes -= old_entry_size;
        true
    }

    /// 检查是否有更新
    #[allow(dead_code)]
    pub fn has_updates(&self) -> bool {
        self.updates.values().any(|m| !m.is_empty())
    }

    /// 获取更新计数
    #[allow(dead_code)]
    pub fn update_count(&self, col: &str) -> usize {
        self.updates.get(col).map(|m| m.len()).unwrap_or(0)
    }

    /// 获取更新计数（所有列）
    #[allow(dead_code)]
    pub fn total_updates(&self) -> usize {
        self.updates.values().map(|m| m.len()).sum()
    }

    /// 获取指定位置的值（如果存在）
    #[allow(dead_code)]
    pub fn get(&self, col: &str, pos: u64) -> Option<&Vec<u8>> {
        self.updates.get(col).and_then(|m| m.get(&pos))
    }

    /// Apply updates to a column array.
    ///
    /// Returns a new array with updates applied, or `None` if there are no updates
    /// for this column. Uses declarative macros to generate type-specific branches
    /// without copy-paste for numeric types; Boolean and variable-width types use
    /// explicit downcast branches.
    #[allow(dead_code)]
    pub fn apply(&self, col: &str, original: &dyn Array) -> Option<ArrayRef> {
        let col_updates = self.updates.get(col)?;
        if col_updates.is_empty() {
            return None;
        }
        tracing::debug!("Applying {} updates to column {}", col_updates.len(), col);

        // Macros defined here have full access to `original` and `col_updates`.
        // They handle numeric primitive types (which all share the same pattern).
        macro_rules! apply_primitive_body {
            ($arrow_type:ty, $builder:ty, $deser:expr) => {{
                let arr = original.as_primitive::<$arrow_type>();
                let mut b = <$builder>::with_capacity(arr.len());
                for i in 0..arr.len() {
                    let pos = i as u64;
                    if let Some(bytes) = col_updates.get(&pos) {
                        // Deserialize bytes; if invalid (wrong length), fall back to original
                        if let Some(val) = $deser(bytes) {
                            b.append_value(val);
                        } else if arr.is_null(i) {
                            b.append_null();
                        } else {
                            b.append_value(arr.value(i));
                        }
                    } else if arr.is_null(i) {
                        b.append_null();
                    } else {
                        b.append_value(arr.value(i));
                    }
                }
                arrow_array::make_array(b.finish().to_data())
            }};
        }

        let new_array: ArrayRef = match original.data_type() {
            // Numeric primitives: handled by macro (avoids 10 near-identical branches)
            arrow_schema::DataType::Int64 => {
                apply_primitive_body!(Int64Type, Int64Builder, bytes_to_i64)
            }
            arrow_schema::DataType::Float64 => {
                apply_primitive_body!(Float64Type, Float64Builder, bytes_to_f64)
            }
            arrow_schema::DataType::Float32 => {
                apply_primitive_body!(Float32Type, Float32Builder, bytes_to_f32)
            }
            arrow_schema::DataType::Int32 => {
                apply_primitive_body!(Int32Type, Int32Builder, bytes_to_i32)
            }
            arrow_schema::DataType::Int16 => {
                apply_primitive_body!(Int16Type, Int16Builder, bytes_to_i16)
            }
            arrow_schema::DataType::Int8 => {
                apply_primitive_body!(Int8Type, Int8Builder, bytes_to_i8)
            }
            arrow_schema::DataType::UInt64 => {
                apply_primitive_body!(UInt64Type, UInt64Builder, bytes_to_u64)
            }
            arrow_schema::DataType::UInt32 => {
                apply_primitive_body!(UInt32Type, UInt32Builder, bytes_to_u32)
            }
            arrow_schema::DataType::UInt16 => {
                apply_primitive_body!(UInt16Type, UInt16Builder, bytes_to_u16)
            }
            arrow_schema::DataType::UInt8 => {
                apply_primitive_body!(UInt8Type, UInt8Builder, bytes_to_u8)
            }

            // Boolean: handled via explicit downcast (BooleanType is not ArrowPrimitiveType)
            // SAFETY: downcast is guaranteed safe because we reached this branch only after
            // matching `original.data_type() == Boolean`, which enforces runtime type matching.
            arrow_schema::DataType::Boolean => {
                let arr = original
                    .as_any()
                    .downcast_ref::<arrow_array::BooleanArray>()
                    .expect("Boolean downcast failed — data_type() and runtime array type must match");
                let mut b = BooleanBuilder::with_capacity(arr.len());
                for i in 0..arr.len() {
                    let pos = i as u64;
                    if let Some(bytes) = col_updates.get(&pos) {
                        b.append_value(bytes.first().is_some_and(|&b| b != 0));
                    } else if arr.is_null(i) {
                        b.append_null();
                    } else {
                        b.append_value(arr.value(i));
                    }
                }
                arrow_array::make_array(b.finish().to_data())
            }

            // Variable-width types: handled via explicit downcast (macros can't handle turbofish)
            arrow_schema::DataType::Utf8 => {
                let arr = original.as_string::<i32>();
                let mut b = StringBuilder::new();
                for i in 0..arr.len() {
                    let pos = i as u64;
                    if let Some(bytes) = col_updates.get(&pos) {
                        if let Ok(s) = std::str::from_utf8(bytes) {
                            b.append_value(s);
                        } else {
                            b.append_null();
                        }
                    } else if arr.is_null(i) {
                        b.append_null();
                    } else {
                        b.append_value(arr.value(i));
                    }
                }
                arrow_array::make_array(b.finish().to_data())
            }

            arrow_schema::DataType::LargeUtf8 => {
                let arr = original.as_string::<i64>();
                let mut b = LargeStringBuilder::new();
                for i in 0..arr.len() {
                    let pos = i as u64;
                    if let Some(bytes) = col_updates.get(&pos) {
                        if let Ok(s) = std::str::from_utf8(bytes) {
                            b.append_value(s);
                        } else {
                            b.append_null();
                        }
                    } else if arr.is_null(i) {
                        b.append_null();
                    } else {
                        b.append_value(arr.value(i));
                    }
                }
                arrow_array::make_array(b.finish().to_data())
            }

            arrow_schema::DataType::Binary => {
                let arr = original.as_binary::<i32>();
                let mut b = BinaryBuilder::new();
                for i in 0..arr.len() {
                    let pos = i as u64;
                    if let Some(bytes) = col_updates.get(&pos) {
                        b.append_value(bytes.as_slice());
                    } else if arr.is_null(i) {
                        b.append_null();
                    } else {
                        b.append_value(arr.value(i));
                    }
                }
                arrow_array::make_array(b.finish().to_data())
            }

            arrow_schema::DataType::LargeBinary => {
                let arr = original.as_binary::<i64>();
                let mut b = LargeBinaryBuilder::new();
                for i in 0..arr.len() {
                    let pos = i as u64;
                    if let Some(bytes) = col_updates.get(&pos) {
                        b.append_value(bytes.as_slice());
                    } else if arr.is_null(i) {
                        b.append_null();
                    } else {
                        b.append_value(arr.value(i));
                    }
                }
                arrow_array::make_array(b.finish().to_data())
            }

            // Unsupported types: pass through unchanged
            _ => arrow_array::make_array(original.to_data()),
        };

        Some(new_array)
    }

    /// 加载更新掩码
    #[allow(dead_code)]
    pub fn load(path: &Path) -> Result<Self> {
        let data = std::fs::read(path)?;
        let mut mask: UpdMask = decode(&data)?;
        // Recompute current_bytes after deserialization since it's skip_serialize
        mask.recompute_bytes();
        Ok(mask)
    }

    /// Recompute `current_bytes` by summing all entry sizes.
    /// Call after deserialization or when restoring from a snapshot.
    #[allow(dead_code)]
    pub fn recompute_bytes(&mut self) {
        self.current_bytes = self
            .updates
            .iter()
            .map(|(col, m)| {
                m.values()
                    .map(|v| {
                        std::mem::size_of::<String>()
                            + col.len()
                            + std::mem::size_of::<u64>()
                            + std::mem::size_of::<Vec<u8>>()
                            + v.len()
                    })
                    .sum::<usize>()
            })
            .sum();
    }

    /// Returns the current memory usage in bytes (from the maintained counter).
    #[allow(dead_code)]
    pub fn current_bytes(&self) -> usize {
        self.current_bytes
    }

    /// 保存更新掩码
    #[allow(dead_code)]
    pub fn save(&self, path: &Path) -> Result<()> {
        let data = encode(self)?;
        std::fs::write(path, data)?;
        Ok(())
    }
}

/// 批量更新掩码管理器
#[allow(dead_code)]
#[derive(Debug, Clone, Default)]
pub struct UpdMaskManager {
    /// segment_id → UpdMask
    masks: HashMap<String, UpdMask>,
}

impl UpdMaskManager {
    #[allow(dead_code)]
    pub fn new() -> Self {
        Self::default()
    }

    /// 获取或创建 segment 的更新掩码
    #[allow(dead_code)]
    pub fn get_or_create(&mut self, seg_id: &str, total_rows: u64) -> &mut UpdMask {
        self.masks
            .entry(seg_id.to_string())
            .or_insert_with(|| UpdMask::new(total_rows))
    }

    /// 移除 segment 的更新掩码
    #[allow(dead_code)]
    pub fn remove(&mut self, seg_id: &str) {
        self.masks.remove(seg_id);
    }

    /// 获取 segment 的更新掩码
    #[allow(dead_code)]
    pub fn get(&self, seg_id: &str) -> Option<&UpdMask> {
        self.masks.get(seg_id)
    }

    #[allow(dead_code)]
    pub fn discarded_total() -> u64 {
        UPDMASK_DISCARDED_TOTAL.load(Ordering::Relaxed)
    }
}
