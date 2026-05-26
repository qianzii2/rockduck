//! DeltaStore: Cell-level Update Tracking with Before Image
//!
//! Provides transaction-aware cell-level updates that track both before and after values.
//! Supports:
//! - Cell-level updates with full before/after image
//! - MVCC-aware visibility (latest committed txn per cell)
//! - Compaction rollback using before_image
//! - Delta overlay merging during reads

use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use std::sync::Arc;
use bincode_next::{Encode, Decode};
use arrow_array::{ArrayRef, Array};
use arrow::datatypes::DataType;
use crate::error::{RockDuckError, Result};
use crate::codec::{encode, decode};

/// Transaction ID type
pub type TxnId = u64;

/// Delta operation types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Encode, Decode)]
pub enum DeltaOpType {
    Update,
    Delete,
    Insert,
}

/// Single cell delta entry
#[derive(Debug, Clone, Encode, Decode)]
pub struct CellDelta {
    pub row: u64,
    pub col: String,
    pub op: DeltaOpType,
    pub before: Option<Vec<u8>>,
    pub after: Option<Vec<u8>>,
    pub txn_id: TxnId,
}

impl CellDelta {
    pub fn new_update(col: String, row: u64, before: Vec<u8>, after: Vec<u8>, txn_id: TxnId) -> Self {
        Self { row, col, op: DeltaOpType::Update, before: Some(before), after: Some(after), txn_id }
    }
    pub fn new_delete(col: String, row: u64, before: Vec<u8>, txn_id: TxnId) -> Self {
        Self { row, col, op: DeltaOpType::Delete, before: Some(before), after: None, txn_id }
    }
    pub fn new_insert(col: String, row: u64, after: Vec<u8>, txn_id: TxnId) -> Self {
        Self { row, col, op: DeltaOpType::Insert, before: None, after: Some(after), txn_id }
    }
}

/// Persisted delta store data (what gets serialized)
#[derive(Debug, Clone, Encode, Decode)]
struct PersistedDeltas {
    pub seg_id: String,
    pub deltas: BTreeMap<TxnId, HashMap<String, HashMap<u64, CellDelta>>>,
}

/// DeltaStore: tracks all cell-level changes for a segment.
pub struct DeltaStore {
    pub seg_id: String,
    /// Per-transaction deltas: txn_id → { col → { row → CellDelta } }
    deltas: BTreeMap<TxnId, HashMap<String, HashMap<u64, CellDelta>>>,
    /// Transient: committed transactions (not serialized)
    committed_txns: Option<BTreeMap<TxnId, ()>>,
}

impl std::fmt::Debug for DeltaStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeltaStore")
            .field("seg_id", &self.seg_id)
            .field("deltas", &self.deltas)
            .field("committed_txns", &self.committed_txns)
            .finish()
    }
}

impl Default for DeltaStore {
    fn default() -> Self {
        Self::new("".to_string())
    }
}

impl Clone for DeltaStore {
    fn clone(&self) -> Self {
        Self {
            seg_id: self.seg_id.clone(),
            deltas: self.deltas.clone(),
            committed_txns: self.committed_txns.clone(),
        }
    }
}

impl DeltaStore {
    pub fn new(seg_id: String) -> Self {
        Self { seg_id, deltas: BTreeMap::new(), committed_txns: None }
    }

    /// Deserialize from bytes (committed_txns reconstructed separately from WAL)
    pub fn from_bytes(seg_id: String, data: &[u8]) -> Result<Self> {
        let persisted: PersistedDeltas = decode(data)?;
        Ok(Self {
            seg_id,
            deltas: persisted.deltas,
            committed_txns: None,
        })
    }

    /// Serialize to bytes
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let persisted = PersistedDeltas { seg_id: self.seg_id.clone(), deltas: self.deltas.clone() };
        Ok(encode(&persisted)?)
    }

    pub fn begin_txn(&mut self, txn_id: TxnId) {
        self.deltas.entry(txn_id).or_default();
    }

    pub fn record_update(&mut self, txn_id: TxnId, col: String, row: u64, before: Vec<u8>, after: Vec<u8>) {
        self.deltas.entry(txn_id).or_default()
            .entry(col.clone()).or_default()
            .insert(row, CellDelta::new_update(col, row, before, after, txn_id));
    }

    pub fn record_delete(&mut self, txn_id: TxnId, col: String, row: u64, before: Vec<u8>) {
        self.deltas.entry(txn_id).or_default()
            .entry(col.clone()).or_default()
            .insert(row, CellDelta::new_delete(col, row, before, txn_id));
    }

    pub fn record_insert(&mut self, txn_id: TxnId, col: String, row: u64, after: Vec<u8>) {
        self.deltas.entry(txn_id).or_default()
            .entry(col.clone()).or_default()
            .insert(row, CellDelta::new_insert(col, row, after, txn_id));
    }

    pub fn commit_txn(&mut self, txn_id: TxnId) {
        self.committed_txns.get_or_insert_with(Default::default).insert(txn_id, ());
    }

    pub fn rollback_txn(&mut self, txn_id: TxnId) {
        self.deltas.remove(&txn_id);
        if let Some(c) = &mut self.committed_txns { c.remove(&txn_id); }
    }

    fn latest_committed_txn(&self) -> Option<TxnId> {
        self.committed_txns.as_ref().and_then(|c| c.last_key_value().map(|(k, _)| *k))
    }

    fn is_txn_committed(&self, txn_id: TxnId) -> bool {
        self.committed_txns.as_ref().map_or(false, |c| c.contains_key(&txn_id))
    }

    /// Get the latest visible delta for a specific cell (col, row).
    pub fn get_visible_delta(&self, col: &str, row: u64) -> Option<&CellDelta> {
        let latest = self.latest_committed_txn()?;
        for (txn_id, txn_deltas) in self.deltas.range(..=latest).rev() {
            if !self.is_txn_committed(*txn_id) { continue; }
            if let Some(col_deltas) = txn_deltas.get(col) {
                if let Some(delta) = col_deltas.get(&row) { return Some(delta); }
            }
        }
        None
    }

    /// Get all visible deltas for a column, sorted by row.
    pub fn get_visible_deltas_for_column(&self, col: &str) -> Vec<(u64, &CellDelta)> {
        let latest = self.latest_committed_txn().unwrap_or(0);
        let mut result = Vec::new();
        let mut seen: HashMap<u64, TxnId> = HashMap::new();
        for (txn_id, txn_deltas) in self.deltas.range(..=latest).rev() {
            if !self.is_txn_committed(*txn_id) { continue; }
            if let Some(col_deltas) = txn_deltas.get(col) {
                for (row, delta) in col_deltas {
                    let should_insert = match seen.get(row) {
                        None => true,
                        Some(&seen_txn) => txn_id > &seen_txn,
                    };
                    if should_insert {
                        seen.insert(*row, *txn_id);
                        result.push((*row, delta));
                    }
                }
            }
        }
        result.sort_by_key(|(row, _)| *row);
        result
    }

    /// Get all visible deltas for a segment, keyed by column name.
    pub fn get_all_visible_deltas(&self) -> HashMap<String, HashMap<u64, CellDelta>> {
        let latest = self.latest_committed_txn().unwrap_or(0);
        let mut result: HashMap<String, HashMap<u64, CellDelta>> = HashMap::new();
        let mut seen: HashMap<(String, u64), TxnId> = HashMap::new();
        for (txn_id, txn_deltas) in self.deltas.range(..=latest).rev() {
            if !self.is_txn_committed(*txn_id) { continue; }
            for (col, col_deltas) in txn_deltas {
                let col_map = result.entry(col.clone()).or_default();
                for (row, delta) in col_deltas {
                    let key = (col.clone(), *row);
                    let should_insert = match seen.get(&key) {
                        None => true,
                        Some(&seen_txn) => txn_id > &seen_txn,
                    };
                    if should_insert {
                        seen.insert(key, *txn_id);
                        col_map.insert(*row, delta.clone());
                    }
                }
            }
        }
        result
    }

    /// Collect all before_images for compaction rollback.
    pub fn get_before_images(&self) -> Vec<CellDelta> {
        let mut images = Vec::new();
        for txn_deltas in self.deltas.values() {
            for col_deltas in txn_deltas.values() {
                for delta in col_deltas.values() {
                    if delta.before.is_some() { images.push(delta.clone()); }
                }
            }
        }
        images
    }

    pub fn is_empty(&self) -> bool {
        self.deltas.is_empty() || self.deltas.values().all(|t| t.values().all(|c| c.is_empty()))
    }

    pub fn delta_count(&self) -> usize {
        self.deltas.values().map(|t| t.values().map(|c| c.len()).sum::<usize>()).sum()
    }

    pub fn committed_txn_count(&self) -> usize {
        self.committed_txns.as_ref().map_or(0, |c| c.len())
    }

    /// 扫描指定行范围内的所有可见 deltas
    ///
    /// 用于 HTAP 双存储路由（D-1）：DeltaStore 范围查询支持。
    /// 遍历 [start_row, end_row) 范围内有变更的行，返回 (row, CellDelta) 列表。
    ///
    /// 与 `get_all_visible_deltas()` 的区别：
    /// - `get_all_visible_deltas()` 返回所有列的 deltas，按列名分组
    /// - `scan_deltas_in_range()` 只返回指定行范围内的 deltas，不按列分组
    pub fn scan_deltas_in_range(&self, start_row: u64, end_row: u64) -> Vec<(u64, CellDelta)> {
        let latest = self.latest_committed_txn().unwrap_or(0);
        let mut result = Vec::new();
        let mut seen: HashMap<u64, TxnId> = HashMap::new();

        for (txn_id, txn_deltas) in self.deltas.range(..=latest).rev() {
            if !self.is_txn_committed(*txn_id) {
                continue;
            }
            for (_col, col_deltas) in txn_deltas {
                for (row, delta) in col_deltas {
                    if *row < start_row || *row >= end_row {
                        continue;
                    }
                    let should_insert = match seen.get(row) {
                        None => true,
                        Some(&seen_txn) => txn_id > &seen_txn,
                    };
                    if should_insert {
                        seen.insert(*row, *txn_id);
                        result.push((*row, delta.clone()));
                    }
                }
            }
        }
        result.sort_by_key(|(row, _)| *row);
        result
    }
}

/// DeltaStoreManager: manages DeltaStores for all segments
#[derive(Debug, Clone, Default)]
pub struct DeltaStoreManager {
    stores: HashMap<String, DeltaStore>,
}

impl DeltaStoreManager {
    pub fn new() -> Self { Self::default() }

    pub fn get_or_create(&mut self, seg_id: &str) -> &mut DeltaStore {
        self.stores.entry(seg_id.to_string()).or_insert_with(|| DeltaStore::new(seg_id.to_string()))
    }

    pub fn get(&self, seg_id: &str) -> Option<&DeltaStore> { self.stores.get(seg_id) }
    pub fn get_mut(&mut self, seg_id: &str) -> Option<&mut DeltaStore> { self.stores.get_mut(seg_id) }
    pub fn remove(&mut self, seg_id: &str) { self.stores.remove(seg_id); }

    pub fn persist(&self, data_dir: &Path, seg_id: &str) -> Result<()> {
        let store = self.stores.get(seg_id).ok_or_else(|| {
            RockDuckError::Storage(format!("DeltaStore not found for segment {}", seg_id))
        })?;
        let path = Self::delta_store_path(data_dir, seg_id);
        if let Some(p) = path.parent() { std::fs::create_dir_all(p)?; }
        std::fs::write(&path, store.to_bytes()?)?;
        Ok(())
    }

    #[allow(dead_code)]
    pub fn load_from_disk(&mut self, data_dir: &Path, seg_id: &str) -> Result<()> {
        let path = Self::delta_store_path(data_dir, seg_id);
        if path.exists() {
            let data = std::fs::read(&path)?;
            let store = DeltaStore::from_bytes(seg_id.to_string(), &data)?;
            self.stores.insert(seg_id.to_string(), store);
        }
        Ok(())
    }

    fn delta_store_path(data_dir: &Path, seg_id: &str) -> std::path::PathBuf {
        data_dir.join("_deltas").join(seg_id).with_extension("delta")
    }

    pub fn prune_before_txn(&mut self, seg_id: &str, cutoff_txn: TxnId) -> Result<usize> {
        let store = self.stores.get_mut(seg_id).ok_or_else(|| {
            RockDuckError::Storage(format!("DeltaStore not found for segment {}", seg_id))
        })?;
        let before = store.delta_count();
        store.deltas.retain(|&txn_id, _| txn_id > cutoff_txn);
        if let Some(c) = &mut store.committed_txns { c.retain(|&txn_id, _| txn_id > cutoff_txn); }
        Ok(before.saturating_sub(store.delta_count()))
    }

    pub fn segments_with_deltas(&self) -> Vec<String> {
        self.stores.iter().filter(|(_, s)| !s.is_empty()).map(|(id, _)| id.clone()).collect()
    }

    /// 获取所有 segment 中 deltas 的总数（用于 HTAP 路由）
    pub fn delta_count(&self) -> usize {
        self.stores.values().map(|s| s.delta_count()).sum()
    }

    /// 扫描指定 segment 在行范围内的所有可见 deltas（用于 HTAP 路由）
    pub fn scan_segment_deltas(&self, seg_id: &str, start_row: u64, end_row: u64) -> Vec<(u64, CellDelta)> {
        self.stores
            .get(seg_id)
            .map(|store| store.scan_deltas_in_range(start_row, end_row))
            .unwrap_or_default()
    }
}

/// Apply deltas as an overlay on top of original column arrays.
pub fn apply_deltas_overlay(
    deltas: &HashMap<u64, CellDelta>,
    original: &dyn Array,
    col_name: &str,
) -> Result<ArrayRef> {
    if deltas.is_empty() { return Ok(arrow_array::make_array(original.to_data())); }

    match original.data_type() {
        DataType::Int8 => apply_deltas_primitive::<arrow_array::types::Int8Type>(original, deltas),
        DataType::Int16 => apply_deltas_primitive::<arrow_array::types::Int16Type>(original, deltas),
        DataType::Int32 => apply_deltas_primitive::<arrow_array::types::Int32Type>(original, deltas),
        DataType::Int64 => apply_deltas_primitive::<arrow_array::types::Int64Type>(original, deltas),
        DataType::UInt8 => apply_deltas_primitive::<arrow_array::types::UInt8Type>(original, deltas),
        DataType::UInt16 => apply_deltas_primitive::<arrow_array::types::UInt16Type>(original, deltas),
        DataType::UInt32 => apply_deltas_primitive::<arrow_array::types::UInt32Type>(original, deltas),
        DataType::UInt64 => apply_deltas_primitive::<arrow_array::types::UInt64Type>(original, deltas),
        DataType::Float32 => apply_deltas_primitive::<arrow_array::types::Float32Type>(original, deltas),
        DataType::Float64 => apply_deltas_primitive::<arrow_array::types::Float64Type>(original, deltas),
        DataType::Boolean => apply_deltas_boolean(original, deltas),
        DataType::Utf8 => apply_deltas_string(original, deltas),
        DataType::LargeUtf8 => apply_deltas_large_string(original, deltas),
        _ => Err(RockDuckError::Query(format!(
            "DeltaStore: unsupported dtype {:?} for column {}", original.data_type(), col_name
        ))),
    }
}

fn apply_deltas_primitive<T: arrow_array::types::ArrowPrimitiveType>(
    original: &dyn Array,
    deltas: &HashMap<u64, CellDelta>,
) -> Result<ArrayRef> {
    use arrow_array::PrimitiveArray;

    let arr = original.as_any().downcast_ref::<PrimitiveArray<T>>()
        .ok_or_else(|| RockDuckError::Query("DeltaStore: failed to downcast to PrimitiveArray".into()))?;

    let len = arr.len();
    if len == 0 { return Ok(arrow_array::make_array(original.to_data())); }

    let native_size = std::mem::size_of::<T::Native>();
    let nulls = arr.nulls();
    let mut new_values: Vec<T::Native> = arr.values().iter().copied().collect();

    for (row, delta) in deltas {
        let pos_usize = *row as usize;
        if pos_usize >= len { continue; }
        match delta.op {
            DeltaOpType::Update | DeltaOpType::Insert => {
                if let Some(ref after) = delta.after {
                    if after.len() >= native_size {
                        let mut arr_bytes: [u8; 16] = [0; 16];
                        let copy_len = after.len().min(16);
                        arr_bytes[..copy_len].copy_from_slice(&after[..copy_len]);
                        unsafe {
                            let mut val: T::Native = std::mem::zeroed();
                            std::ptr::copy_nonoverlapping(arr_bytes.as_ptr(), &mut val as *mut T::Native as *mut u8, copy_len);
                            new_values[pos_usize] = val;
                        }
                    }
                }
            }
            DeltaOpType::Delete => {
                if let Some(_n) = nulls {
                    let mut builder = arrow_array::builder::PrimitiveBuilder::<T>::new();
                    for i in 0..len {
                        if _n.is_null(i) || i == pos_usize {
                            builder.append_null();
                        } else {
                            builder.append_value(new_values[i]);
                        }
                    }
                    return Ok(Arc::new(builder.finish()));
                } else {
                    let mut builder = arrow_array::builder::PrimitiveBuilder::<T>::new();
                    for i in 0..len {
                        if i == pos_usize {
                            builder.append_null();
                        } else {
                            builder.append_value(new_values[i]);
                        }
                    }
                    return Ok(Arc::new(builder.finish()));
                }
            }
        }
    }

    let new_arr = if let Some(n) = nulls {
        PrimitiveArray::<T>::new(arrow::buffer::ScalarBuffer::from(new_values), Some(n.clone()))
    } else {
        PrimitiveArray::<T>::new(arrow::buffer::ScalarBuffer::from(new_values), None)
    };
    Ok(Arc::new(new_arr))
}

fn apply_deltas_boolean(original: &dyn Array, deltas: &HashMap<u64, CellDelta>) -> Result<ArrayRef> {
    use arrow_array::BooleanArray;
    use arrow_array::builder::BooleanBuilder;

    let arr = original.as_any().downcast_ref::<BooleanArray>()
        .ok_or_else(|| RockDuckError::Query("DeltaStore: failed to downcast to BooleanArray".into()))?;

    let len = arr.len();
    if len == 0 { return Ok(arrow_array::make_array(original.to_data())); }

    let nulls = arr.nulls();
    let mut new_values: Vec<bool> = (0..len).map(|i| arr.value(i)).collect();

    for (row, delta) in deltas {
        let pos_usize = *row as usize;
        if pos_usize >= len { continue; }
        match delta.op {
            DeltaOpType::Update | DeltaOpType::Insert => {
                if let Some(ref after) = delta.after { new_values[pos_usize] = !after.is_empty() && after[0] != 0; }
            }
            DeltaOpType::Delete => { new_values[pos_usize] = false; }
        }
    }

    let new_arr = if let Some(n) = nulls {
        let mut builder = BooleanBuilder::new();
        for i in 0..len {
            if n.is_null(i) { builder.append_null(); } else { builder.append_value(new_values[i]); }
        }
        builder.finish()
    } else {
        BooleanArray::from(new_values)
    };
    Ok(Arc::new(new_arr))
}

fn apply_deltas_string(original: &dyn Array, deltas: &HashMap<u64, CellDelta>) -> Result<ArrayRef> {
    use arrow_array::StringArray;
    use arrow_array::builder::StringBuilder;

    let arr = original.as_any().downcast_ref::<StringArray>()
        .ok_or_else(|| RockDuckError::Query("DeltaStore: failed to downcast to StringArray".into()))?;

    let len = arr.len();
    if len == 0 { return Ok(arrow_array::make_array(original.to_data())); }

    let nulls = arr.nulls();
    let mut new_strings: Vec<String> = (0..len).map(|i| arr.value(i).to_string()).collect();

    for (row, delta) in deltas {
        let pos_usize = *row as usize;
        if pos_usize >= len { continue; }
        match delta.op {
            DeltaOpType::Update | DeltaOpType::Insert => {
                if let Some(ref after) = delta.after { new_strings[pos_usize] = String::from_utf8(after.clone()).unwrap_or_default(); }
            }
            DeltaOpType::Delete => { new_strings[pos_usize].clear(); }
        }
    }

    let new_arr = if let Some(n) = nulls {
        let mut builder = StringBuilder::with_capacity(len, 1024);
        for i in 0..len {
            if n.is_null(i) { builder.append_null(); } else { builder.append_value(&new_strings[i]); }
        }
        builder.finish()
    } else {
        StringArray::from(new_strings)
    };
    Ok(Arc::new(new_arr))
}

fn apply_deltas_large_string(original: &dyn Array, deltas: &HashMap<u64, CellDelta>) -> Result<ArrayRef> {
    use arrow_array::LargeStringArray;
    use arrow_array::builder::LargeStringBuilder;

    let arr = original.as_any().downcast_ref::<LargeStringArray>()
        .ok_or_else(|| RockDuckError::Query("DeltaStore: failed to downcast to LargeStringArray".into()))?;

    let len = arr.len();
    if len == 0 { return Ok(arrow_array::make_array(original.to_data())); }

    let nulls = arr.nulls();
    let mut new_strings: Vec<String> = (0..len).map(|i| arr.value(i).to_string()).collect();

    for (row, delta) in deltas {
        let pos_usize = *row as usize;
        if pos_usize >= len { continue; }
        match delta.op {
            DeltaOpType::Update | DeltaOpType::Insert => {
                if let Some(ref after) = delta.after { new_strings[pos_usize] = String::from_utf8(after.clone()).unwrap_or_default(); }
            }
            DeltaOpType::Delete => { new_strings[pos_usize].clear(); }
        }
    }

    let new_arr = if let Some(n) = nulls {
        let mut builder = LargeStringBuilder::with_capacity(len, 1024);
        for i in 0..len {
            if n.is_null(i) { builder.append_null(); } else { builder.append_value(&new_strings[i]); }
        }
        builder.finish()
    } else {
        LargeStringArray::from(new_strings)
    };
    Ok(Arc::new(new_arr))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::AsArray;

    #[test] fn test_delta_store_new() { let s = DeltaStore::new("seg_001".into()); assert_eq!(s.seg_id, "seg_001"); assert!(s.is_empty()); }
    #[test] fn test_delta_store_record_update() { let mut s = DeltaStore::new("s".into()); s.record_update(100, "age".into(), 5, vec![25u8], vec![30u8]); assert_eq!(s.delta_count(), 1); }
    #[test] fn test_delta_store_commit_rollback() {
        let mut s = DeltaStore::new("s".into());
        s.record_update(100, "age".into(), 5, vec![1u8], vec![2u8]);
        assert!(s.get_visible_delta("age", 5).is_none());
        s.commit_txn(100);
        assert!(s.get_visible_delta("age", 5).is_some());
        s.rollback_txn(100);
        assert!(s.get_visible_delta("age", 5).is_none());
        assert!(s.is_empty());
    }
    #[test] fn test_delta_store_latest_wins() {
        let mut s = DeltaStore::new("s".into());
        s.record_update(100, "age".into(), 5, vec![20u8], vec![25u8]);
        s.record_update(200, "age".into(), 5, vec![25u8], vec![30u8]);
        s.commit_txn(100); s.commit_txn(200);
        let delta = s.get_visible_delta("age", 5).unwrap();
        assert_eq!(delta.txn_id, 200);
    }
    #[test] fn test_delta_store_get_all_visible_deltas() {
        let mut s = DeltaStore::new("s".into());
        s.record_update(100, "age".into(), 0, vec![20u8], vec![25u8]);
        s.record_update(100, "name".into(), 0, vec![1u8], vec![2u8]);
        s.commit_txn(100);
        let all = s.get_all_visible_deltas();
        assert_eq!(all.len(), 2);
    }
    #[test] fn test_delta_store_insert_delete() {
        let mut s = DeltaStore::new("s".into());
        s.record_insert(100, "new_col".into(), 0, vec![99u8]);
        s.record_delete(200, "age".into(), 5, vec![30u8]);
        s.commit_txn(100); s.commit_txn(200);
        assert_eq!(s.get_visible_delta("new_col", 0).unwrap().op, DeltaOpType::Insert);
        assert_eq!(s.get_visible_delta("age", 5).unwrap().op, DeltaOpType::Delete);
    }
    #[test] fn test_delta_store_serialize_roundtrip() {
        let mut s = DeltaStore::new("seg_001".into());
        s.record_update(100, "age".into(), 5, vec![25u8], vec![30u8]);
        let bytes = s.to_bytes().unwrap();
        let loaded = DeltaStore::from_bytes("seg_001".into(), &bytes).unwrap();
        assert_eq!(loaded.delta_count(), 1);
    }
    #[test] fn test_delta_store_manager() {
        let mut mgr = DeltaStoreManager::new();
        mgr.get_or_create("seg_001").record_update(1, "a".into(), 0, vec![1u8], vec![2u8]);
        assert_eq!(mgr.get("seg_001").unwrap().delta_count(), 1);
        mgr.remove("seg_001");
        assert!(mgr.get("seg_001").is_none());
    }
    #[test] fn test_delta_store_prune() {
        let mut s = DeltaStore::new("s".into());
        s.record_update(100, "age".into(), 5, vec![25u8], vec![30u8]);
        s.record_update(200, "name".into(), 0, vec![1u8], vec![2u8]);
        s.commit_txn(100); s.commit_txn(200);
        let mut mgr = DeltaStoreManager::new();
        mgr.stores.insert("s".into(), s);
        let pruned = mgr.prune_before_txn("s", 150).unwrap();
        assert_eq!(pruned, 1);
        assert_eq!(mgr.get("s").unwrap().delta_count(), 1);
    }
    #[test] fn test_apply_deltas_overlay_int64() {
        let mut s = DeltaStore::new("s".into());
        s.record_update(100, "age".into(), 1, 20i64.to_le_bytes().to_vec(), 25i64.to_le_bytes().to_vec());
        s.record_update(100, "age".into(), 3, 40i64.to_le_bytes().to_vec(), 50i64.to_le_bytes().to_vec());
        s.commit_txn(100);
        let all = s.get_all_visible_deltas();
        let deltas = all.get("age").unwrap();
        let arr64 = arrow_array::Int64Array::from(vec![10i64, 20, 30, 40, 50, 60]);
        let original: ArrayRef = std::sync::Arc::new(arr64);
        let merged = apply_deltas_overlay(deltas, original.as_ref(), "age").unwrap();
        let arr = merged.as_primitive::<arrow_array::types::Int64Type>();
        assert_eq!(arr.value(0), 10);
        assert_eq!(arr.value(1), 25);
        assert_eq!(arr.value(3), 50);
    }

    // ============================================================
    // Float64 overlay: exercises apply_deltas_primitive<Float64Type> path
    // ============================================================

    #[test]
    fn test_apply_deltas_overlay_float64() {
        let mut s = DeltaStore::new("s".into());
        // Record: row=1, "price" column updated from 99.5 to 149.95
        s.record_update(
            100,
            "price".into(),
            1,
            99.5f64.to_le_bytes().to_vec(),
            149.95f64.to_le_bytes().to_vec(),
        );
        s.commit_txn(100);

        let all = s.get_all_visible_deltas();
        let deltas = all.get("price").unwrap();
        let arr = arrow_array::Float64Array::from(vec![10.0, 99.5, 30.0, 40.0, 50.0]);
        let original: ArrayRef = std::sync::Arc::new(arr);
        let merged = apply_deltas_overlay(deltas, original.as_ref(), "price").unwrap();
        let arr = merged.as_primitive::<arrow_array::types::Float64Type>();

        assert_eq!(arr.value(0), 10.0, "Row 0 unchanged");
        assert!(
            (arr.value(1) - 149.95).abs() < 1e-9,
            "Row 1 must apply delta: expected 149.95, got {}",
            arr.value(1)
        );
        assert_eq!(arr.value(2), 30.0, "Row 2 unchanged");
    }
    #[test] fn test_apply_deltas_overlay_string() {
        let mut s = DeltaStore::new("s".into());
        s.record_update(100, "name".into(), 0, b"Bob".to_vec(), b"Alice".to_vec());
        s.commit_txn(100);
        let all = s.get_all_visible_deltas();
        let deltas = all.get("name").unwrap();
        let arr_str = arrow_array::StringArray::from(vec!["Bob", "Carol"]);
        let original: ArrayRef = std::sync::Arc::new(arr_str);
        let merged = apply_deltas_overlay(deltas, original.as_ref(), "name").unwrap();
        let arr = merged.as_any().downcast_ref::<arrow_array::StringArray>().unwrap();
        assert_eq!(arr.value(0), "Alice");
        assert_eq!(arr.value(1), "Carol");
    }
    #[test] fn test_apply_deltas_overlay_empty() {
        let deltas: HashMap<u64, CellDelta> = HashMap::new();
        let arr64 = arrow_array::Int64Array::from(vec![1i64, 2, 3]);
        let original: ArrayRef = std::sync::Arc::new(arr64);
        let merged = apply_deltas_overlay(&deltas, original.as_ref(), "age").unwrap();
        let arr = merged.as_primitive::<arrow_array::types::Int64Type>();
        assert_eq!(arr.value(0), 1);
    }
    #[test] fn test_get_visible_deltas_for_column() {
        let mut s = DeltaStore::new("s".into());
        s.record_update(100, "age".into(), 1, vec![20u8], vec![25u8]);
        s.record_update(100, "age".into(), 5, vec![40u8], vec![50u8]);
        s.record_update(100, "age".into(), 3, vec![30u8], vec![35u8]);
        s.commit_txn(100);
        let deltas = s.get_visible_deltas_for_column("age");
        assert_eq!(deltas.len(), 3);
        assert_eq!(deltas[0].0, 1);
        assert_eq!(deltas[1].0, 3);
        assert_eq!(deltas[2].0, 5);
    }

    // ============================================================
    // Additional tests for missing coverage
    // ============================================================

    #[test]
    fn test_uncommitted_txn_invisible() {
        let mut s = DeltaStore::new("s".into());
        s.record_update(100, "age".into(), 5, vec![25u8], vec![30u8]);

        let result = s.get_visible_delta("age", 5);
        assert!(
            result.is_none(),
            "Uncommitted txn should not be visible"
        );
    }

    #[test]
    fn test_get_visible_delta_nonexistent_cell() {
        let mut s = DeltaStore::new("s".into());
        s.record_update(100, "age".into(), 5, vec![25u8], vec![30u8]);
        s.commit_txn(100);

        let result = s.get_visible_delta("age", 99);
        assert!(result.is_none(), "Nonexistent cell should return None");

        let result = s.get_visible_delta("nonexistent_col", 0);
        assert!(result.is_none(), "Nonexistent column should return None");
    }

    #[test]
    fn test_get_visible_delta_uncommitted_txn_ignored() {
        let mut s = DeltaStore::new("s".into());
        s.record_update(100, "age".into(), 5, vec![25u8], vec![30u8]);
        s.commit_txn(100);

        s.record_update(200, "age".into(), 5, vec![30u8], vec![40u8]);

        let result = s.get_visible_delta("age", 5);
        assert!(result.is_some());
        assert_eq!(result.unwrap().txn_id, 100);
    }

    #[test]
    fn test_delta_store_commit_order_matters() {
        let mut s = DeltaStore::new("s".into());

        s.record_update(100, "age".into(), 5, vec![20u8], vec![25u8]);
        s.commit_txn(100);

        s.record_update(200, "age".into(), 5, vec![25u8], vec![35u8]);
        s.commit_txn(200);

        let result = s.get_visible_delta("age", 5);
        assert!(result.is_some());
        assert_eq!(result.unwrap().txn_id, 200);
        assert_eq!(result.unwrap().after.as_ref().map(|v| v.as_slice()), Some(vec![35u8].as_slice()));
    }

    #[test]
    fn test_apply_deltas_overlay_out_of_bounds_row() {
        use arrow_array::Int64Array;
        use arrow::array::AsArray;

        let mut s = DeltaStore::new("s".into());
        s.record_update(100, "val".into(), 9999, vec![0u8], vec![99u8]);
        s.commit_txn(100);
        let all = s.get_all_visible_deltas();
        let deltas = all.get("val").unwrap();

        let arr = Int64Array::from(vec![1i64, 2, 3]);
        let original: ArrayRef = std::sync::Arc::new(arr);
        let merged = apply_deltas_overlay(deltas, original.as_ref(), "val").unwrap();
        let arr = merged.as_primitive::<arrow_array::types::Int64Type>();

        assert_eq!(arr.value(0), 1);
        assert_eq!(arr.value(1), 2);
        assert_eq!(arr.value(2), 3);
    }

    #[test]
    fn test_delta_store_manager_persist_and_load() {
        use tempfile::TempDir;

        let temp = TempDir::new().unwrap();

        let mut mgr = DeltaStoreManager::new();
        {
            let store = mgr.get_or_create("seg_persist");
            store.record_update(100, "col_a".into(), 0, vec![10u8], vec![20u8]);
            store.record_update(100, "col_b".into(), 1, vec![30u8], vec![40u8]);
            store.commit_txn(100);
        }

        // Persist to disk
        mgr.persist(temp.path(), "seg_persist").unwrap();

        // Load into a new manager
        let mut mgr2 = DeltaStoreManager::new();
        mgr2.load_from_disk(temp.path(), "seg_persist").unwrap();

        let loaded = mgr2.get("seg_persist").unwrap();
        // committed_txns is not persisted (reconstructed from WAL on recovery)
        // so delta_count is preserved but visibility requires committed_txns to be re-established
        assert_eq!(loaded.delta_count(), 2);
        assert_eq!(loaded.committed_txn_count(), 0); // committed_txns not serialized
    }

    #[test]
    fn test_delta_store_get_before_images() {
        let mut s = DeltaStore::new("s".into());

        s.record_update(100, "age".into(), 5, vec![25u8], vec![30u8]);
        s.record_delete(200, "name".into(), 0, vec![65u8, 66u8, 67u8]);
        s.record_insert(300, "new_col".into(), 10, vec![99u8]);

        let images = s.get_before_images();

        assert_eq!(images.len(), 2);
        for img in &images {
            assert!(img.before.is_some());
        }

        let before_vals: Vec<_> = images.iter().filter_map(|img| img.before.as_ref()).collect();
        assert!(before_vals.iter().any(|v| v.as_slice() == vec![25u8].as_slice()));
        assert!(before_vals.iter().any(|v| v.as_slice() == vec![65u8, 66u8, 67u8].as_slice()));
    }

    #[test]
    fn test_delta_store_empty_after_rollback() {
        let mut s = DeltaStore::new("s".into());
        s.record_update(100, "age".into(), 5, vec![25u8], vec![30u8]);
        s.commit_txn(100);
        assert!(!s.is_empty());

        s.rollback_txn(200);
        assert!(!s.is_empty());

        s.rollback_txn(100);
        assert!(s.is_empty(), "Rolling back all deltas should make store empty");
    }

    #[test]
    fn test_delta_store_manager_segments_with_deltas() {
        let mut mgr = DeltaStoreManager::new();

        let store_a = mgr.get_or_create("seg_a");
        store_a.record_update(100, "col".into(), 0, vec![1u8], vec![2u8]);
        store_a.commit_txn(100);

        let store_b = mgr.get_or_create("seg_b");
        store_b.record_update(200, "col".into(), 0, vec![3u8], vec![4u8]);
        store_b.commit_txn(200);

        mgr.get_or_create("seg_c");

        let with_deltas = mgr.segments_with_deltas();
        assert_eq!(with_deltas.len(), 2);
        assert!(with_deltas.contains(&"seg_a".to_string()));
        assert!(with_deltas.contains(&"seg_b".to_string()));
        assert!(!with_deltas.contains(&"seg_c".to_string()));
    }

    #[test]
    fn test_delta_store_txn_committed_count() {
        let mut s = DeltaStore::new("s".into());
        assert_eq!(s.committed_txn_count(), 0);

        s.record_update(100, "a".into(), 0, vec![1u8], vec![2u8]);
        s.commit_txn(100);
        assert_eq!(s.committed_txn_count(), 1);

        s.record_update(200, "b".into(), 0, vec![3u8], vec![4u8]);
        s.commit_txn(200);
        assert_eq!(s.committed_txn_count(), 2);

        s.rollback_txn(300);
        assert_eq!(s.committed_txn_count(), 2);
    }

    #[test]
    fn test_delta_store_prune_before_txn() {
        let mut s = DeltaStore::new("s".into());
        s.record_update(100, "col".into(), 0, vec![1u8], vec![2u8]);
        s.record_update(200, "col".into(), 1, vec![3u8], vec![4u8]);
        s.record_update(300, "col".into(), 2, vec![5u8], vec![6u8]);
        s.commit_txn(100);
        s.commit_txn(200);
        s.commit_txn(300);

        let mut mgr = DeltaStoreManager::new();
        mgr.stores.insert("s".to_string(), s);

        let pruned = mgr.prune_before_txn("s", 200).unwrap();
        assert_eq!(pruned, 2);

        let remaining = mgr.get("s").unwrap();
        assert_eq!(remaining.delta_count(), 1);

        let pruned = mgr.prune_before_txn("s", 500).unwrap();
        assert_eq!(pruned, 1);
    }
}
