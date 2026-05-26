//! Granule (1MB) 读写单元
//! 
//! Granule 是最小的读写单元，包含：
//! - 列数据块
//! - Zone Map 统计
//! - 压缩信息

use std::path::PathBuf;
use arrow_array::Array;
use arrow_array::ArrayRef;
use arrow_array::{
    Int8Array, Int16Array, Int32Array, Int64Array,
    UInt8Array, UInt16Array, UInt32Array, UInt64Array,
    Float32Array, Float64Array, BooleanArray,
    StringArray, LargeStringArray,
};
use arrow::datatypes::DataType;
use crate::error::Result;
use crate::segment::meta::{ColumnStats, GranuleMeta, ZoneMapStats};

/// Granule 大小常量
pub const GRANULE_SIZE: usize = 1024 * 1024; // 1MB

/// 从 Arrow 数组提取最小值字节
fn extract_min_bytes(array: &ArrayRef) -> Option<Vec<u8>> {
    match array.data_type() {
        DataType::Int8 => {
            let a = array.as_any().downcast_ref::<Int8Array>()?;
            a.values().iter().min().copied().map(|v| v.to_le_bytes().to_vec())
        }
        DataType::Int16 => {
            let a = array.as_any().downcast_ref::<Int16Array>()?;
            a.values().iter().min().copied().map(|v| v.to_le_bytes().to_vec())
        }
        DataType::Int32 => {
            let a = array.as_any().downcast_ref::<Int32Array>()?;
            a.values().iter().min().copied().map(|v| v.to_le_bytes().to_vec())
        }
        DataType::Int64 => {
            let a = array.as_any().downcast_ref::<Int64Array>()?;
            a.values().iter().min().copied().map(|v| v.to_le_bytes().to_vec())
        }
        DataType::UInt8 => {
            let a = array.as_any().downcast_ref::<UInt8Array>()?;
            a.values().iter().min().copied().map(|v| v.to_le_bytes().to_vec())
        }
        DataType::UInt16 => {
            let a = array.as_any().downcast_ref::<UInt16Array>()?;
            a.values().iter().min().copied().map(|v| v.to_le_bytes().to_vec())
        }
        DataType::UInt32 => {
            let a = array.as_any().downcast_ref::<UInt32Array>()?;
            a.values().iter().min().copied().map(|v| v.to_le_bytes().to_vec())
        }
        DataType::UInt64 => {
            let a = array.as_any().downcast_ref::<UInt64Array>()?;
            a.values().iter().min().copied().map(|v| v.to_le_bytes().to_vec())
        }
        DataType::Float32 => {
            let a = array.as_any().downcast_ref::<Float32Array>()?;
            a.values().iter().copied().reduce(|a, b| a.min(b)).map(|v| v.to_le_bytes().to_vec())
        }
        DataType::Float64 => {
            let a = array.as_any().downcast_ref::<Float64Array>()?;
            a.values().iter().copied().reduce(|a, b| a.min(b)).map(|v| v.to_le_bytes().to_vec())
        }
        DataType::Boolean => {
            let a = array.as_any().downcast_ref::<BooleanArray>()?;
            if a.len() == 0 {
                return None;
            }
            // For boolean: false (0) < true (1), so min=false, max=true
            // min_val = m.map_or(v, |m| m && v) - only true if ALL are true
            // max_val = m.map_or(v, |m| m || v) - true if ANY is true
            let mut min_val: Option<bool> = None;
            let mut max_val: Option<bool> = None;
            for i in 0..a.len() {
                let v = a.value(i);
                min_val = Some(min_val.map_or(v, |m| m && v));
                max_val = Some(max_val.map_or(v, |m| m || v));
            }
            min_val.map(|v| vec![if v { 0u8 } else { 1u8 }])
        }
        DataType::Utf8 => {
            let a = array.as_any().downcast_ref::<StringArray>()?;
            (0..a.len()).map(|i| a.value(i)).min().map(|v| v.as_bytes().to_vec())
        }
        DataType::LargeUtf8 => {
            let a = array.as_any().downcast_ref::<LargeStringArray>()?;
            (0..a.len()).map(|i| a.value(i)).min().map(|v| v.as_bytes().to_vec())
        }
        _ => None,
    }
}

/// 从 Arrow 数组提取最大值字节
fn extract_max_bytes(array: &ArrayRef) -> Option<Vec<u8>> {
    match array.data_type() {
        DataType::Int8 => {
            let a = array.as_any().downcast_ref::<Int8Array>()?;
            a.values().iter().max().copied().map(|v| v.to_le_bytes().to_vec())
        }
        DataType::Int16 => {
            let a = array.as_any().downcast_ref::<Int16Array>()?;
            a.values().iter().max().copied().map(|v| v.to_le_bytes().to_vec())
        }
        DataType::Int32 => {
            let a = array.as_any().downcast_ref::<Int32Array>()?;
            a.values().iter().max().copied().map(|v| v.to_le_bytes().to_vec())
        }
        DataType::Int64 => {
            let a = array.as_any().downcast_ref::<Int64Array>()?;
            a.values().iter().max().copied().map(|v| v.to_le_bytes().to_vec())
        }
        DataType::UInt8 => {
            let a = array.as_any().downcast_ref::<UInt8Array>()?;
            a.values().iter().max().copied().map(|v| v.to_le_bytes().to_vec())
        }
        DataType::UInt16 => {
            let a = array.as_any().downcast_ref::<UInt16Array>()?;
            a.values().iter().max().copied().map(|v| v.to_le_bytes().to_vec())
        }
        DataType::UInt32 => {
            let a = array.as_any().downcast_ref::<UInt32Array>()?;
            a.values().iter().max().copied().map(|v| v.to_le_bytes().to_vec())
        }
        DataType::UInt64 => {
            let a = array.as_any().downcast_ref::<UInt64Array>()?;
            a.values().iter().max().copied().map(|v| v.to_le_bytes().to_vec())
        }
        DataType::Float32 => {
            let a = array.as_any().downcast_ref::<Float32Array>()?;
            a.values().iter().copied().reduce(|a, b| a.max(b)).map(|v| v.to_le_bytes().to_vec())
        }
        DataType::Float64 => {
            let a = array.as_any().downcast_ref::<Float64Array>()?;
            a.values().iter().copied().reduce(|a, b| a.max(b)).map(|v| v.to_le_bytes().to_vec())
        }
        DataType::Boolean => {
            let a = array.as_any().downcast_ref::<BooleanArray>()?;
            if a.len() == 0 {
                return None;
            }
            // For boolean: false (0) < true (1), so min=false, max=true
            let mut min_val: Option<bool> = None;
            let mut max_val: Option<bool> = None;
            for i in 0..a.len() {
                let v = a.value(i);
                min_val = Some(min_val.map_or(v, |m| m && v));
                max_val = Some(max_val.map_or(v, |m| m || v));
            }
            max_val.map(|v| vec![if v { 1u8 } else { 0u8 }])
        }
        DataType::Utf8 => {
            let a = array.as_any().downcast_ref::<StringArray>()?;
            (0..a.len()).map(|i| a.value(i)).max().map(|v| v.as_bytes().to_vec())
        }
        DataType::LargeUtf8 => {
            let a = array.as_any().downcast_ref::<LargeStringArray>()?;
            (0..a.len()).map(|i| a.value(i)).max().map(|v| v.as_bytes().to_vec())
        }
        _ => None,
    }
}

/// 计算数组的 null 计数
fn compute_null_count(array: &ArrayRef) -> u32 {
    array.null_count() as u32
}

/// 从 Arrow 数组计算列统计信息
fn compute_column_stats(array: &ArrayRef) -> ColumnStats {
    ColumnStats {
        min: extract_min_bytes(array),
        max: extract_max_bytes(array),
        null_count: compute_null_count(array),
        sum: None,
        distinct_count: None,
    }
}

/// 合并两个 ColumnStats
fn merge_column_stats(existing: &ColumnStats, new: &ColumnStats) -> ColumnStats {
    let merged_min = match (&existing.min, &new.min) {
        (Some(e), Some(n)) => {
            Some(if e.as_slice() < n.as_slice() { e.clone() } else { n.clone() })
        }
        (Some(e), None) => Some(e.clone()),
        (None, Some(n)) => Some(n.clone()),
        (None, None) => None,
    };

    let merged_max = match (&existing.max, &new.max) {
        (Some(e), Some(n)) => {
            Some(if e.as_slice() > n.as_slice() { e.clone() } else { n.clone() })
        }
        (Some(e), None) => Some(e.clone()),
        (None, Some(n)) => Some(n.clone()),
        (None, None) => None,
    };

    ColumnStats {
        min: merged_min,
        max: merged_max,
        null_count: existing.null_count + new.null_count,
        sum: None,
        distinct_count: None,
    }
}

/// Granule 读取器
pub struct GranuleReader {
    /// Granule 元数据
    pub meta: GranuleMeta,
    /// 列数据路径
    col_paths: std::collections::HashMap<String, PathBuf>,
}

impl GranuleReader {
    /// 创建新的 GranuleReader
    pub fn new(meta: GranuleMeta) -> Self {
        Self {
            meta,
            col_paths: std::collections::HashMap::new(),
        }
    }

    /// 添加列路径
    pub fn add_column(&mut self, col_name: &str, path: PathBuf) {
        self.col_paths.insert(col_name.to_string(), path);
    }

    /// 获取 Zone Map
    pub fn zone_map(&self) -> &ZoneMapStats {
        &self.meta.zone_map
    }

    /// 获取行数
    pub fn row_count(&self) -> u32 {
        self.meta.row_count
    }

    /// 获取行偏移
    pub fn row_offset(&self) -> u64 {
        self.meta.row_offset
    }
}

/// Granule 写入器
pub struct GranuleWriter {
    /// Granule ID
    pub granule_id: u32,
    /// 起始行号
    row_offset: u64,
    /// 列数据
    columns: std::collections::HashMap<String, Vec<ArrayRef>>,
    /// Zone Map
    zone_map: ZoneMapStats,
    /// 当前行数
    current_rows: u32,
}

impl GranuleWriter {
    /// 创建新的 GranuleWriter
    pub fn new(granule_id: u32, row_offset: u64) -> Self {
        Self {
            granule_id,
            row_offset,
            columns: std::collections::HashMap::new(),
            zone_map: ZoneMapStats::new(),
            current_rows: 0,
        }
    }

    /// 添加列数据
    pub fn add_column(&mut self, col_name: &str, array: ArrayRef) -> Result<()> {
        let rows = array.len() as u32;

        if self.current_rows == 0 {
            self.current_rows = rows;
        } else if self.current_rows != rows {
            return Err(crate::RockDuckError::Storage(format!(
                "Column {} has {} rows, expected {}",
                col_name, rows, self.current_rows
            )));
        }

        self.columns.entry(col_name.to_string()).or_insert_with(Vec::new).push(array.clone());

        // 更新 Zone Map (合并到现有统计)
        self.update_zone_map(col_name, &array)?;

        Ok(())
    }

    /// 更新 Zone Map
    fn update_zone_map(&mut self, col_name: &str, array: &ArrayRef) -> Result<()> {
        let new_stats = compute_column_stats(array);

        if let Some(existing_stats) = self.zone_map.stats.get(col_name) {
            let merged = merge_column_stats(existing_stats, &new_stats);
            self.zone_map.stats.insert(col_name.to_string(), merged);
        } else {
            self.zone_map.stats.insert(col_name.to_string(), new_stats);
        }

        Ok(())
    }

    /// 获取元数据
    pub fn build_meta(&self) -> GranuleMeta {
        let mut meta = GranuleMeta::new(
            self.granule_id,
            self.row_offset,
            self.current_rows,
        );
        meta.zone_map = self.zone_map.clone();
        meta
    }

    /// 获取当前行数
    pub fn current_rows(&self) -> u32 {
        self.current_rows
    }

    /// 获取当前大小估计
    pub fn estimated_size(&self) -> usize {
        self.current_rows as usize * 8 // 假设每行 8 字节
    }

    /// 是否达到 Granule 大小
    pub fn is_full(&self) -> bool {
        self.estimated_size() >= GRANULE_SIZE
    }

    /// 清空数据
    pub fn clear(&mut self) {
        self.columns.clear();
        self.zone_map = ZoneMapStats::new();
        self.current_rows = 0;
    }
}

/// Granule 缓冲区管理器
pub struct GranuleBuffer {
    /// Granule 写入器
    writers: std::collections::HashMap<String, GranuleWriter>,
    /// 当前的 granule ID
    current_granule_id: u32,
    /// 当前行偏移
    current_row_offset: u64,
}

impl GranuleBuffer {
    /// 创建新的 GranuleBuffer
    pub fn new() -> Self {
        Self {
            writers: std::collections::HashMap::new(),
            current_granule_id: 0,
            current_row_offset: 0,
        }
    }

    /// 获取或创建 granule 写入器
    fn get_or_create_writer(&mut self, seg_id: &str) -> &mut GranuleWriter {
        if !self.writers.contains_key(seg_id) {
            self.writers.insert(seg_id.to_string(), GranuleWriter::new(
                self.current_granule_id,
                self.current_row_offset,
            ));
        }
        self.writers.get_mut(seg_id).unwrap()
    }

    /// 添加数据到 segment
    pub fn add_row(&mut self, seg_id: &str, columns: &std::collections::HashMap<String, ArrayRef>) -> Result<()> {
        let writer = self.get_or_create_writer(seg_id);
        
        for (name, array) in columns {
            writer.add_column(name, array.clone())?;
        }
        
        Ok(())
    }

    /// 检查是否需要刷新
    pub fn needs_flush(&self, seg_id: &str) -> bool {
        self.writers.get(seg_id)
            .map(|w| w.is_full())
            .unwrap_or(false)
    }

    /// 获取 granule 元数据并准备新 granule
    pub fn flush_and_next(&mut self, seg_id: &str) -> Result<Option<GranuleMeta>> {
        if let Some(writer) = self.writers.get_mut(seg_id) {
            if writer.current_rows() > 0 {
                let meta = writer.build_meta();
                
                // 更新状态
                self.current_granule_id += 1;
                self.current_row_offset += writer.current_rows() as u64;
                
                // 清空写入器
                writer.clear();
                
                return Ok(Some(meta));
            }
        }
        
        Ok(None)
    }

    /// 完成 segment，刷新所有数据
    pub fn finish_segment(&mut self, seg_id: &str) -> Result<Vec<GranuleMeta>> {
        let mut metas = Vec::new();
        
        while let Some(meta) = self.flush_and_next(seg_id)? {
            metas.push(meta);
        }
        
        // 如果还有最后的数据
        if let Some(writer) = self.writers.get_mut(seg_id) {
            if writer.current_rows() > 0 {
                metas.push(writer.build_meta());
            }
        }
        
        self.writers.remove(seg_id);
        
        Ok(metas)
    }

    /// 获取当前 granule 的行数
    pub fn current_row_count(&self, seg_id: &str) -> u32 {
        self.writers.get(seg_id)
            .map(|w| w.current_rows())
            .unwrap_or(0)
    }
}

impl Default for GranuleBuffer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_granule_writer() {
        let writer = GranuleWriter::new(0, 0);
        assert_eq!(writer.current_rows(), 0);
        assert!(!writer.is_full());
    }

    #[test]
    fn test_granule_buffer() {
        let buffer = GranuleBuffer::new();
        assert_eq!(buffer.current_row_count("test"), 0);
    }

    // ============================================================
    // GranuleReader tests
    // ============================================================

    #[test]
    fn test_granule_reader_new() {
        let meta = GranuleMeta::new(0, 0, 100);
        let reader = GranuleReader::new(meta.clone());
        assert_eq!(reader.row_count(), 100);
        assert_eq!(reader.row_offset(), 0);
        assert_eq!(reader.zone_map().stats.len(), 0);
    }

    #[test]
    fn test_granule_reader_add_column() {
        let meta = GranuleMeta::new(5, 1000, 500);
        let mut reader = GranuleReader::new(meta);
        use std::path::PathBuf;
        reader.add_column("age", PathBuf::from("/data/age.vortex"));
        assert_eq!(reader.meta.granule_id, 5);
        assert_eq!(reader.meta.row_offset, 1000);
    }

    // ============================================================
    // GranuleWriter comprehensive tests
    // ============================================================

    #[test]
    fn test_granule_writer_current_rows() {
        let writer = GranuleWriter::new(1, 0);
        assert_eq!(writer.current_rows(), 0);
    }

    #[test]
    fn test_granule_writer_estimated_size() {
        let writer = GranuleWriter::new(0, 0);
        assert_eq!(writer.estimated_size(), 0);
    }

    #[test]
    fn test_granule_writer_is_full() {
        let writer = GranuleWriter::new(0, 0);
        assert!(!writer.is_full());
    }

    #[test]
    fn test_granule_writer_build_meta() {
        let mut writer = GranuleWriter::new(3, 1000);
        let arr: ArrayRef = std::sync::Arc::new(arrow_array::Int64Array::from(vec![1i64, 2, 3]));
        writer.add_column("id", arr).unwrap();
        let meta = writer.build_meta();
        assert_eq!(meta.granule_id, 3);
        assert_eq!(meta.row_offset, 1000);
        assert_eq!(meta.row_count, 3);
    }

    #[test]
    fn test_granule_writer_add_column_rejects_mismatch() {
        let mut writer = GranuleWriter::new(0, 0);
        let arr1: ArrayRef = std::sync::Arc::new(arrow_array::Int64Array::from(vec![1i64, 2, 3]));
        writer.add_column("id", arr1).unwrap();

        let arr2: ArrayRef = std::sync::Arc::new(arrow_array::Int64Array::from(vec![1i64, 2])); // 2 rows != 3
        let result = writer.add_column("age", arr2);
        assert!(result.is_err());
    }

    #[test]
    fn test_granule_writer_clear() {
        let mut writer = GranuleWriter::new(0, 0);
        let arr: ArrayRef = std::sync::Arc::new(arrow_array::Int64Array::from(vec![1i64, 2, 3]));
        writer.add_column("id", arr).unwrap();
        assert_eq!(writer.current_rows(), 3);

        writer.clear();
        assert_eq!(writer.current_rows(), 0);
    }

    #[test]
    fn test_granule_writer_update_zone_map_no_panic() {
        let mut writer = GranuleWriter::new(0, 0);
        let arr: ArrayRef = std::sync::Arc::new(arrow_array::Int64Array::from(vec![1i64, 2, 3]));
        let result = writer.add_column("id", arr);
        // Should not panic even if zone_map update is no-op
        assert!(result.is_ok());
    }

    // ============================================================
    // GranuleBuffer comprehensive tests
    // ============================================================

    #[test]
    fn test_granule_buffer_new() {
        let buffer = GranuleBuffer::new();
        assert_eq!(buffer.current_row_count("any"), 0);
    }

    #[test]
    fn test_granule_buffer_multiple_segments() {
        let mut buffer = GranuleBuffer::new();
        let arr: ArrayRef = std::sync::Arc::new(arrow_array::Int64Array::from(vec![1i64, 2, 3]));
        let mut cols = std::collections::HashMap::new();
        cols.insert("id".to_string(), arr);

        buffer.add_row("seg_a", &cols).unwrap();
        buffer.add_row("seg_b", &cols).unwrap();

        assert_eq!(buffer.current_row_count("seg_a"), 3);
        assert_eq!(buffer.current_row_count("seg_b"), 3);
    }

    #[test]
    fn test_granule_buffer_needs_flush() {
        let mut buffer = GranuleBuffer::new();
        assert!(!buffer.needs_flush("seg_001"));

        let arr: ArrayRef = std::sync::Arc::new(arrow_array::Int64Array::from(vec![1i64]));
        let mut cols = std::collections::HashMap::new();
        cols.insert("id".to_string(), arr);
        buffer.add_row("seg_001", &cols).unwrap();

        assert!(!buffer.needs_flush("seg_001")); // Not full with default 1MB threshold
    }

    #[test]
    fn test_granule_buffer_flush_and_next() {
        let mut buffer = GranuleBuffer::new();
        let arr: ArrayRef = std::sync::Arc::new(arrow_array::Int64Array::from(vec![1i64, 2, 3]));
        let mut cols = std::collections::HashMap::new();
        cols.insert("id".to_string(), arr);
        buffer.add_row("seg_001", &cols).unwrap();

        let result = buffer.flush_and_next("seg_001");
        assert!(result.is_ok());
    }

    #[test]
    fn test_granule_buffer_flush_nonexistent() {
        let mut buffer = GranuleBuffer::new();
        let result = buffer.flush_and_next("seg_nonexistent");
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
    }

    #[test]
    fn test_granule_buffer_finish_segment() {
        let mut buffer = GranuleBuffer::new();
        let arr: ArrayRef = std::sync::Arc::new(arrow_array::Int64Array::from(vec![1i64, 2]));
        let mut cols = std::collections::HashMap::new();
        cols.insert("id".to_string(), arr);
        buffer.add_row("seg_001", &cols).unwrap();

        let metas = buffer.finish_segment("seg_001");
        assert!(metas.is_ok());
        assert!(!metas.unwrap().is_empty());
    }

    #[test]
    fn test_granule_buffer_finish_empty() {
        let mut buffer = GranuleBuffer::new();
        let metas = buffer.finish_segment("seg_empty");
        assert!(metas.is_ok());
        assert!(metas.unwrap().is_empty());
    }

    #[test]
    fn test_granule_buffer_current_row_count() {
        let mut buffer = GranuleBuffer::new();
        assert_eq!(buffer.current_row_count("seg_001"), 0);

        let arr: ArrayRef = std::sync::Arc::new(arrow_array::Int64Array::from(vec![1i64, 2, 3]));
        let mut cols = std::collections::HashMap::new();
        cols.insert("id".to_string(), arr);
        buffer.add_row("seg_001", &cols).unwrap();

        assert_eq!(buffer.current_row_count("seg_001"), 3);
    }

    // ============================================================
    // GranuleBuffer Default trait
    // ============================================================

    #[test]
    fn test_granule_buffer_default() {
        let buffer: GranuleBuffer = Default::default();
        assert_eq!(buffer.current_row_count("any"), 0);
    }

    // ============================================================
    // GRANULE_SIZE constant
    // ============================================================

    #[test]
    fn test_granule_size_constant() {
        assert_eq!(GRANULE_SIZE, 1024 * 1024);
    }
}
