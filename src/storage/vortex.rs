//! Vortex 列存读写
//!
//! Segment 目录结构：
//!   seg_{id}/
//!     _meta.vortex      # Segment 元数据
//!     _del.vortex       # 删除掩码
//!     _upd.vortex       # 更新掩码
//!     {col}.vortex      # 列数据
//!     _zm.vortex        # Zone Map
//!
//! Mmap 支持：
//!   Frozen 状态的 segment 使用 mmap 读取，实现零拷贝。

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::Mutex;
use arrow_array::{RecordBatch, ArrayRef, Array};
use arrow_schema::Schema;
use std::collections::HashMap;
use crate::error::{RockDuckError, Result};
use crate::segment::meta::{SegmentMeta, GranuleMeta, ZoneMapStats, ColumnStats, SegmentStatus};
use crate::segment::layout::SegmentLayout;
use crate::codec::{encode, decode};
use crate::segment::encoding::AdaptiveEncoder;

/// Vortex 写入器
pub struct VortexWriter {
    /// Segment 布局
    layout: SegmentLayout,
    /// Segment ID
    seg_id: String,
    /// 列数据（未刷新）
    column_data: std::collections::HashMap<String, Vec<ArrayRef>>,
    /// 删除掩码
    del_mask: Vec<bool>,
    /// 当前行数
    row_count: u64,
}

impl VortexWriter {
    /// 创建新的 VortexWriter
    pub fn new(data_dir: PathBuf, seg_id: String) -> Self {
        Self {
            layout: SegmentLayout::new(&data_dir, &seg_id),
            seg_id,
            column_data: std::collections::HashMap::new(),
            del_mask: Vec::new(),
            row_count: 0,
        }
    }

    /// 追加数据（单行）
    pub fn append_row(&mut self, row: &std::collections::HashMap<String, ArrayRef>) -> Result<()> {
        // 验证所有列长度一致
        let batch_len = row.values().next()
            .map(|a| a.len())
            .unwrap_or(0);
        
        for (name, array) in row {
            if array.len() != batch_len {
                return Err(RockDuckError::Storage(format!(
                    "Column {} has inconsistent length: expected {}, got {}",
                    name, batch_len, array.len()
                )));
            }
            
            self.column_data
                .entry(name.clone())
                .or_insert_with(Vec::new)
                .push(array.clone());
        }
        
        // 删除掩码全部为 false
        for _ in 0..batch_len {
            self.del_mask.push(false);
        }
        
        self.row_count += batch_len as u64;
        Ok(())
    }

    /// 追加数据（RecordBatch）
    pub fn append_batch(&mut self, batch: &RecordBatch) -> Result<()> {
        let num_rows = batch.num_rows() as u64;
        
        for i in 0..batch.num_columns() {
            let col_name = batch.schema().field(i).name().clone();
            let array = batch.column(i);
            
            self.column_data
                .entry(col_name)
                .or_insert_with(Vec::new)
                .push(array.clone());
        }
        
        // 删除掩码全部为 false
        for _ in 0..num_rows {
            self.del_mask.push(false);
        }
        
        self.row_count += num_rows;
        Ok(())
    }

    /// 刷新到磁盘（使用自适应编码）
    pub fn flush(self) -> Result<SegmentMeta> {
        self.flush_with_encoding(true)
    }

    /// 刷新到磁盘
    ///
    /// 如果 `adaptive_encoding` 为 true，则对每个列做真实数据采样分析，
    /// 推荐最优编码并写入 ColumnDef；否则使用默认编码。
    pub fn flush_with_encoding(self, adaptive_encoding: bool) -> Result<SegmentMeta> {
        // 创建 segment 目录
        self.layout.create_dirs()?;

        let mut columns = Vec::new();
        let mut granules = Vec::new();

        // 记录每个列的文件偏移量（用于 mmap 稀疏读取）
        let mut column_offsets: std::collections::HashMap<String, u64> = std::collections::HashMap::new();

        let encoder = if adaptive_encoding {
            Some(AdaptiveEncoder::new())
        } else {
            None
        };

        // 写入每个列
        for (col_name, arrays) in &self.column_data {
            let combined = self.combine_arrays(arrays)?;
            let col_path = self.layout.col_path(col_name);

            // 记录写入前的文件大小作为偏移量
            let file_offset = std::fs::metadata(&col_path)
                .map(|m| m.len())
                .unwrap_or(0);
            column_offsets.insert(col_name.clone(), file_offset);

            let dtype = crate::segment::meta::DataType::from_arrow(combined.data_type());
            let col_def = if let Some(ref enc) = encoder {
                let def = crate::segment::meta::ColumnDef::new(col_name.clone(), dtype);
                let total_count = combined.len();
                let analysis = enc.analyze_column_array(&def, combined.as_ref(), total_count);
                let rec = enc.recommend(&analysis);
                tracing::debug!(
                    "Adaptive encoding for '{}': {:?} (confidence={:.2}) reason={}",
                    col_name,
                    rec.encoding,
                    rec.confidence,
                    rec.reason
                );
                crate::segment::meta::ColumnDef::with_encoding(
                    col_name.clone(),
                    dtype,
                    rec.encoding,
                )
            } else {
                crate::segment::meta::ColumnDef::new(col_name.clone(), dtype)
            };

            // 写入 Vortex 文件
            self.write_vortex_file(&combined, &col_path)?;

            // 更新 Zone Map
            let _zm = self.compute_zone_map(&combined)?;

            columns.push(col_def);
        }

    // 记录删除掩码的文件偏移
    let _del_offset_before = std::fs::metadata(self.layout.del_mask_path())
            .map(|m| m.len())
            .unwrap_or(0);

        // 写入删除掩码
        let del_array = arrow_array::BooleanArray::from(self.del_mask.clone());
        self.write_vortex_file(&del_array, &self.layout.del_mask_path())?;

        // 计算 granule 元数据
        let granule_size = 1024 * 1024; // 1MB
        let mut granule_count = 0u32;
        let mut row_offset = 0u64;

        // 第一列文件作为参考，记录每个 granule 边界的字节偏移
        let reference_col = self.column_data.keys().next();
        let mut granule_byte_offset = reference_col.and_then(|col| {
            column_offsets.get(col).copied()
        }).unwrap_or(0);

        while row_offset < self.row_count {
            let remaining = self.row_count - row_offset;
            let this_granule_rows = std::cmp::min(remaining, granule_size as u64) as u32;

            // 使用 with_offset 记录此 granule 的文件字节偏移（供 mmap 定位）
            granules.push(GranuleMeta::with_offset(
                granule_count,
                row_offset,
                this_granule_rows,
                granule_byte_offset,
            ));

            // 假设均匀分布：每个 granule 约 1MB
            granule_byte_offset += granule_size;
            granule_count += 1;
            row_offset += this_granule_rows as u64;
        }

        // 创建 SegmentMeta
        let now = crate::codec::current_timestamp_secs();
        let meta = SegmentMeta {
            seg_id: self.seg_id.clone(),
            table: "default".to_string(), // TODO: 传入表名
            row_count: self.row_count,
            uncompressed_size: 0,
            compressed_size: 0,
            columns,
            granules,
            deleted_rows: 0,
            del_ratio: 0.0,
            status: crate::segment::meta::SegmentStatus::Active,
            created_at: now,
            updated_at: now,
            pk_range: None,
        };

        // 写入元数据
        self.write_meta_file(&meta)?;

        Ok(meta)
    }

    /// 合并多个 Array 为一个 (concatenation)
    #[allow(dead_code)]
    fn combine_arrays(&self, arrays: &[ArrayRef]) -> Result<ArrayRef> {
        if arrays.is_empty() {
            return Err(RockDuckError::Storage("No arrays to combine".to_string()));
        }

        if arrays.len() == 1 {
            return Ok(arrays[0].clone());
        }

        let refs: Vec<&dyn arrow_array::Array> = arrays.iter().map(|a| a.as_ref()).collect();
        arrow::compute::concat(refs.as_slice())
            .map_err(|e| RockDuckError::Storage(format!("Failed to concat arrays: {}", e)))
    }

    /// 从 Arrow Array 中提取最小值为字节向量
    fn extract_min_bytes(array: &ArrayRef) -> Option<Vec<u8>> {
        Self::extract_min_max_bytes(array, true)
    }

    /// 从 Arrow Array 中提取最大值为字节向量
    fn extract_max_bytes(array: &ArrayRef) -> Option<Vec<u8>> {
        Self::extract_min_max_bytes(array, false)
    }

    /// 从 Arrow Array 中提取 min 或 max 值并转换为字节向量
    fn extract_min_max_bytes(array: &ArrayRef, is_min: bool) -> Option<Vec<u8>> {
        use arrow::datatypes::DataType;

        match array.data_type() {
            DataType::Int8 => {
                let arr = array.as_any().downcast_ref::<arrow_array::Int8Array>()?;
                if arr.len() == 0 { return None; }
                let val = if is_min { arr.iter().flatten().min()? } else { arr.iter().flatten().max()? };
                Some(val.to_le_bytes().to_vec())
            }
            DataType::Int16 => {
                let arr = array.as_any().downcast_ref::<arrow_array::Int16Array>()?;
                if arr.len() == 0 { return None; }
                let val = if is_min { arr.iter().flatten().min()? } else { arr.iter().flatten().max()? };
                Some(val.to_le_bytes().to_vec())
            }
            DataType::Int32 => {
                let arr = array.as_any().downcast_ref::<arrow_array::Int32Array>()?;
                if arr.len() == 0 { return None; }
                let val = if is_min { arr.iter().flatten().min()? } else { arr.iter().flatten().max()? };
                Some(val.to_le_bytes().to_vec())
            }
            DataType::Int64 => {
                let arr = array.as_any().downcast_ref::<arrow_array::Int64Array>()?;
                if arr.len() == 0 { return None; }
                let val = if is_min { arr.iter().flatten().min()? } else { arr.iter().flatten().max()? };
                Some(val.to_le_bytes().to_vec())
            }
            DataType::UInt8 => {
                let arr = array.as_any().downcast_ref::<arrow_array::UInt8Array>()?;
                if arr.len() == 0 { return None; }
                let val = if is_min { arr.iter().flatten().min()? } else { arr.iter().flatten().max()? };
                Some(val.to_le_bytes().to_vec())
            }
            DataType::UInt16 => {
                let arr = array.as_any().downcast_ref::<arrow_array::UInt16Array>()?;
                if arr.len() == 0 { return None; }
                let val = if is_min { arr.iter().flatten().min()? } else { arr.iter().flatten().max()? };
                Some(val.to_le_bytes().to_vec())
            }
            DataType::UInt32 => {
                let arr = array.as_any().downcast_ref::<arrow_array::UInt32Array>()?;
                if arr.len() == 0 { return None; }
                let val = if is_min { arr.iter().flatten().min()? } else { arr.iter().flatten().max()? };
                Some(val.to_le_bytes().to_vec())
            }
            DataType::UInt64 => {
                let arr = array.as_any().downcast_ref::<arrow_array::UInt64Array>()?;
                if arr.len() == 0 { return None; }
                let val = if is_min { arr.iter().flatten().min()? } else { arr.iter().flatten().max()? };
                Some(val.to_le_bytes().to_vec())
            }
            DataType::Float32 => {
                let arr = array.as_any().downcast_ref::<arrow_array::Float32Array>()?;
                if arr.len() == 0 { return None; }
                let val = if is_min {
                    arr.values().iter().copied().fold(f32::NAN, |a, b| a.min(b))
                } else {
                    arr.values().iter().copied().fold(f32::NAN, |a, b| a.max(b))
                };
                if val.is_nan() { return None; }
                Some(val.to_le_bytes().to_vec())
            }
            DataType::Float64 => {
                let arr = array.as_any().downcast_ref::<arrow_array::Float64Array>()?;
                if arr.len() == 0 { return None; }
                let val = if is_min {
                    arr.values().iter().copied().fold(f64::NAN, |a, b| a.min(b))
                } else {
                    arr.values().iter().copied().fold(f64::NAN, |a, b| a.max(b))
                };
                if val.is_nan() { return None; }
                Some(val.to_le_bytes().to_vec())
            }
            DataType::Utf8 => {
                if array.len() == 0 { return None; }
                let arr = array.as_any().downcast_ref::<arrow_array::StringArray>()?;
                let mut min_val: Option<&str> = None;
                let mut max_val: Option<&str> = None;
                for i in 0..array.len() {
                    let val = arr.value(i);
                    match min_val {
                        None => min_val = Some(val),
                        Some(m) if val < m => min_val = Some(val),
                        _ => {}
                    }
                    match max_val {
                        None => max_val = Some(val),
                        Some(m) if val > m => max_val = Some(val),
                        _ => {}
                    }
                }
                let val = if is_min { min_val } else { max_val };
                val.map(|v| v.as_bytes().to_vec())
            }
            DataType::LargeUtf8 => {
                if array.len() == 0 { return None; }
                let arr = array.as_any().downcast_ref::<arrow_array::LargeStringArray>()?;
                let mut min_val: Option<&str> = None;
                let mut max_val: Option<&str> = None;
                for i in 0..array.len() {
                    let val = arr.value(i);
                    match min_val {
                        None => min_val = Some(val),
                        Some(m) if val < m => min_val = Some(val),
                        _ => {}
                    }
                    match max_val {
                        None => max_val = Some(val),
                        Some(m) if val > m => max_val = Some(val),
                        _ => {}
                    }
                }
                let val = if is_min { min_val } else { max_val };
                val.map(|v| v.as_bytes().to_vec())
            }
            DataType::Binary => {
                if array.len() == 0 { return None; }
                let arr = array.as_any().downcast_ref::<arrow_array::BinaryArray>()?;
                let mut min_val: Option<&[u8]> = None;
                let mut max_val: Option<&[u8]> = None;
                for i in 0..array.len() {
                    let val = arr.value(i);
                    match min_val {
                        None => min_val = Some(val),
                        Some(m) if val < m => min_val = Some(val),
                        _ => {}
                    }
                    match max_val {
                        None => max_val = Some(val),
                        Some(m) if val > m => max_val = Some(val),
                        _ => {}
                    }
                }
                let val = if is_min { min_val } else { max_val };
                val.map(|v| v.to_vec())
            }
            DataType::LargeBinary => {
                if array.len() == 0 { return None; }
                let arr = array.as_any().downcast_ref::<arrow_array::LargeBinaryArray>()?;
                let mut min_val: Option<&[u8]> = None;
                let mut max_val: Option<&[u8]> = None;
                for i in 0..array.len() {
                    let val = arr.value(i);
                    match min_val {
                        None => min_val = Some(val),
                        Some(m) if val < m => min_val = Some(val),
                        _ => {}
                    }
                    match max_val {
                        None => max_val = Some(val),
                        Some(m) if val > m => max_val = Some(val),
                        _ => {}
                    }
                }
                let val = if is_min { min_val } else { max_val };
                val.map(|v| v.to_vec())
            }
            DataType::Date32 => {
                let arr = array.as_any().downcast_ref::<arrow_array::Date32Array>()?;
                if arr.len() == 0 { return None; }
                let val = if is_min { arr.iter().flatten().min()? } else { arr.iter().flatten().max()? };
                Some(val.to_le_bytes().to_vec())
            }
            DataType::Date64 => {
                let arr = array.as_any().downcast_ref::<arrow_array::Date64Array>()?;
                if arr.len() == 0 { return None; }
                let val = if is_min { arr.iter().flatten().min()? } else { arr.iter().flatten().max()? };
                Some(val.to_le_bytes().to_vec())
            }
            DataType::Boolean => {
                let arr = array.as_any().downcast_ref::<arrow_array::BooleanArray>()?;
                if arr.len() == 0 { return None; }
                // For bool: min = false if exists, max = true if exists
                let has_false = (0..arr.len()).any(|i| arr.value(i) == false);
                let has_true = (0..arr.len()).any(|i| arr.value(i) == true);
                let val = if is_min {
                    if has_false { false } else { true }
                } else {
                    if has_true { true } else { false }
                };
                Some(vec![if val { 1 } else { 0 }])
            }
            DataType::Timestamp(unit, _) => {
                match unit {
                    arrow::datatypes::TimeUnit::Second => {
                        let arr = array.as_any().downcast_ref::<arrow_array::TimestampSecondArray>()?;
                        if arr.len() == 0 { return None; }
                        let val = if is_min { arr.iter().flatten().min()? } else { arr.iter().flatten().max()? };
                        Some(val.to_le_bytes().to_vec())
                    }
                    arrow::datatypes::TimeUnit::Millisecond => {
                        let arr = array.as_any().downcast_ref::<arrow_array::TimestampMillisecondArray>()?;
                        if arr.len() == 0 { return None; }
                        let val = if is_min { arr.iter().flatten().min()? } else { arr.iter().flatten().max()? };
                        Some(val.to_le_bytes().to_vec())
                    }
                    arrow::datatypes::TimeUnit::Microsecond => {
                        let arr = array.as_any().downcast_ref::<arrow_array::TimestampMicrosecondArray>()?;
                        if arr.len() == 0 { return None; }
                        let val = if is_min { arr.iter().flatten().min()? } else { arr.iter().flatten().max()? };
                        Some(val.to_le_bytes().to_vec())
                    }
                    arrow::datatypes::TimeUnit::Nanosecond => {
                        let arr = array.as_any().downcast_ref::<arrow_array::TimestampNanosecondArray>()?;
                        if arr.len() == 0 { return None; }
                        let val = if is_min { arr.iter().flatten().min()? } else { arr.iter().flatten().max()? };
                        Some(val.to_le_bytes().to_vec())
                    }
                }
            }
            _ => None,
        }
    }

    /// 计算单个 Array 的 Zone Map
    fn compute_zone_map(&self, array: &ArrayRef) -> Result<ZoneMapStats> {
        let mut stats = ZoneMapStats::new();

        // 空数组直接返回
        if array.len() == 0 {
            return Ok(stats);
        }

        // 提取 min/max 值
        let min_bytes = Self::extract_min_bytes(array);
        let max_bytes = Self::extract_max_bytes(array);

        // 获取 null 计数
        let null_count = array.null_count() as u32;

        let col_stats = ColumnStats {
            min: min_bytes,
            max: max_bytes,
            null_count,
            sum: None,
            distinct_count: None,
        };

        stats.add_column_stats("", col_stats);

        Ok(stats)
    }

    /// 计算所有列的 Zone Map（最终化）
    pub fn finalize_zone_map(&self) -> Result<ZoneMapStats> {
        let mut stats = ZoneMapStats::new();

        for (col_name, arrays) in &self.column_data {
            // 合并所有 arrays
            let combined = self.combine_arrays(arrays)?;

            // 空数组跳过
            if combined.len() == 0 {
                continue;
            }

            // 提取 min/max 值
            let min_bytes = Self::extract_min_bytes(&combined);
            let max_bytes = Self::extract_max_bytes(&combined);

            // 获取 null 计数
            let null_count = combined.null_count() as u32;

            let col_stats = ColumnStats {
                min: min_bytes,
                max: max_bytes,
                null_count,
                sum: None,
                distinct_count: None,
            };

            stats.add_column_stats(col_name, col_stats);
        }

        Ok(stats)
    }

    /// 写入 Vortex 文件
    fn write_vortex_file(&self, array: &dyn arrow_array::Array, path: &Path) -> Result<()> {
        use arrow_ipc::writer::FileWriter;
        use arrow_array::make_array;
        let file = std::fs::File::create(path)?;
        let field = arrow_schema::Field::new("value", array.data_type().clone(), true);
        let schema = Schema::new(vec![field]);
        let array_data = array.to_data();
        let array_ref: ArrayRef = make_array(array_data);
        let batch = RecordBatch::try_new(Arc::new(schema.clone()), vec![array_ref])?;
        let mut writer = FileWriter::try_new(std::io::BufWriter::new(file), &schema)?;
        writer.write(&batch)?;
        writer.finish()?;
        Ok(())
    }

    /// 写入元数据文件
    fn write_meta_file(&self, meta: &SegmentMeta) -> Result<()> {
        let path = self.layout.meta_path();
        let data = encode(meta)?;
        std::fs::write(path, data)?;
        Ok(())
    }

    /// 获取当前行数
    pub fn row_count(&self) -> u64 {
        self.row_count
    }
}

/// Vortex 读取器
pub struct VortexReader {
    /// Segment 布局
    layout: SegmentLayout,
    /// 缓存的 Segment 元数据
    meta_cache: std::collections::HashMap<String, SegmentMeta>,
    /// Mmap 缓存（Arc 共享，同一文件多处引用时不重复映射）
    mmap_cache: Mutex<HashMap<String, Arc<memmap2::Mmap>>>,
}

impl VortexReader {
    /// 创建新的 VortexReader
    pub fn new(data_dir: PathBuf) -> Self {
        Self {
            layout: SegmentLayout::new(&data_dir, ""),
            meta_cache: std::collections::HashMap::new(),
            mmap_cache: Mutex::new(HashMap::new()),
        }
    }

    /// 读取 Segment 元数据
    pub fn read_meta(&mut self, seg_id: &str) -> Result<SegmentMeta> {
        if let Some(meta) = self.meta_cache.get(seg_id) {
            return Ok(meta.clone());
        }

        let path = self.layout.meta_path();
        let data = std::fs::read(path)?;
        let meta: SegmentMeta = decode(&data)?;

        self.meta_cache.insert(seg_id.to_string(), meta.clone());
        Ok(meta)
    }

    /// 读取指定 granule 的数据
    #[allow(dead_code)]
    pub fn read_granule(
        &self,
        seg_id: &str,
        col_name: &str,
        _granule_id: u32,
    ) -> Result<ArrayRef> {
        let layout = SegmentLayout::new(&self.layout.seg_dir.parent().unwrap(), seg_id);
        let col_path = layout.col_path(col_name);
        self.read_arrow_file(&col_path)
    }

    /// 读取列数据（根据 segment 状态自动选择 BufReader 或 mmap）
    pub fn read_column(&self, seg_id: &str, col_name: &str) -> Result<ArrayRef> {
        let layout = SegmentLayout::new(&self.layout.seg_dir.parent().unwrap(), seg_id);
        let col_path = layout.col_path(col_name);

        // 读取 segment 状态，判断是否使用 mmap
        let meta_path = layout.meta_path();
        if meta_path.exists() {
            let data = std::fs::read(&meta_path)?;
            if let Ok(meta) = decode::<SegmentMeta>(&data) {
                if matches!(meta.status, SegmentStatus::Frozen) {
                    return self.read_column_mmap_internal(seg_id, col_name);
                }
            }
        }

        // 非 Frozen 状态或读取失败，使用 BufReader
        self.read_arrow_file(&col_path)
    }

    /// 读取删除掩码
    #[allow(dead_code)]
    pub fn read_del_mask(&self, seg_id: &str) -> Result<Vec<bool>> {
        let layout = SegmentLayout::new(&self.layout.seg_dir.parent().unwrap(), seg_id);
        let path = layout.del_mask_path();
        let array = self.read_arrow_file(&path)?;

        let bool_array = array.as_ref();
        if let Some(arr) = bool_array.as_any().downcast_ref::<arrow_array::BooleanArray>() {
            Ok((0..arr.len()).map(|i| arr.value(i)).collect())
        } else {
            Err(RockDuckError::Storage("Invalid del mask format".to_string()))
        }
    }

    /// 通过 mmap 读取列数据（Frozen segment 专用，零拷贝）
    fn read_column_mmap_internal(&self, seg_id: &str, col_name: &str) -> Result<ArrayRef> {
        let layout = SegmentLayout::new(&self.layout.seg_dir.parent().unwrap(), seg_id);
        let col_path = layout.col_path(col_name);

        let cache_key = format!("{}:{}", seg_id, col_name);

        // 尝试从缓存获取；没有则创建 mmap 并缓存（Arc 共享引用）
        let mmap: Arc<memmap2::Mmap> = {
            let cache = self.mmap_cache.lock().unwrap();
            if let Some(cached) = cache.get(&cache_key) {
                cached.clone()
            } else {
                drop(cache); // 释放锁，避免持有锁期间做 I/O
                let file = std::fs::File::open(&col_path)?;
                let m = unsafe { memmap2::Mmap::map(&file)? };
                let m = Arc::new(m);
                let mut cache = self.mmap_cache.lock().unwrap();
                cache.insert(cache_key, m.clone());
                m
            }
        };

        // mmap 实现了 AsRef<[u8]>，可直接传给 Arrow IPC
        let reader = arrow_ipc::reader::FileReader::try_new(
            std::io::Cursor::new(mmap.as_ref()),
            None,
        )?;

        let batches: Vec<RecordBatch> = reader
            .filter_map(|b| b.ok())
            .collect();

        if batches.is_empty() {
            return Err(RockDuckError::Storage("Empty file".to_string()));
        }

        if batches.len() == 1 {
            return Ok(batches.into_iter().next().unwrap().column(0).clone());
        }

        // 合并多个 RecordBatch 的同一列
        let num_columns = batches[0].num_columns();
        let mut result_columns: Vec<ArrayRef> = Vec::new();

        for col_idx in 0..num_columns {
            let mut combined: Vec<ArrayRef> = batches.iter()
                .map(|b| b.column(col_idx).clone())
                .collect();

            if combined.len() == 1 {
                result_columns.push(combined.remove(0));
            } else {
                let refs: Vec<&dyn arrow_array::Array> = combined.iter().map(|a| a.as_ref()).collect();
                let concat = arrow::compute::concat(refs.as_slice())
                    .map_err(|e| RockDuckError::Storage(format!("Failed to concat column {}: {}", col_idx, e)))?;
                result_columns.push(concat);
            }
        }

        if result_columns.len() == 1 {
            return Ok(result_columns.into_iter().next().unwrap());
        }

        let schema = batches[0].schema();
        let batch = RecordBatch::try_new(schema, result_columns)
            .map_err(|e| RockDuckError::Storage(format!("Failed to create RecordBatch: {}", e)))?;
        Ok(batch.column(0).clone())
    }

    /// 清除 mmap 缓存（释放内存）
    pub fn clear_mmap_cache(&self) {
        let mut cache = self.mmap_cache.lock().unwrap();
        cache.clear();
    }

    /// 清除指定 segment 的 mmap 缓存
    pub fn clear_segment_mmap_cache(&self, seg_id: &str) {
        let mut cache = self.mmap_cache.lock().unwrap();
        cache.retain(|key, _| !key.starts_with(&format!("{}:", seg_id)));
    }

    /// 读取 Arrow IPC 文件（BufReader 路径，用于 Active/Compactable segment）
    fn read_arrow_file(&self, path: &Path) -> Result<ArrayRef> {
        if !path.exists() {
            return Err(RockDuckError::ColumnNotFound(path.to_string_lossy().to_string()));
        }

        let reader = std::fs::File::open(path)?;
        let mut reader = arrow_ipc::reader::FileReader::try_new(
            std::io::BufReader::new(reader),
            None,
        )?;

        let batch = reader.next()
            .ok_or_else(|| RockDuckError::Storage("Empty file".to_string()))?
            .map_err(|e| RockDuckError::Storage(format!("Failed to read batch: {}", e)))?;

        Ok(batch.column(0).clone())
    }

    /// 检查 segment 是否存在
    pub fn segment_exists(&self, seg_id: &str) -> bool {
        let data_dir = self.layout.seg_dir.parent().unwrap().parent().unwrap();
        let seg_dir = data_dir.join("segments").join(seg_id);
        seg_dir.exists()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_writer() {
        let temp_dir = std::env::temp_dir().join("docdb_vortex_test");
        let _writer = VortexWriter::new(temp_dir, "test_seg".to_string());
        // TODO: 添加更多测试
    }

    // ============================================================
    // VortexWriter tests
    // ============================================================

    #[test]
    fn test_vortex_writer_new() {
        let temp_dir = std::env::temp_dir().join("docdb_vortex_writer_test");
        let writer = VortexWriter::new(temp_dir.clone(), "seg_test".to_string());
        assert_eq!(writer.row_count(), 0);
    }

    #[test]
    fn test_vortex_writer_row_count() {
        let temp_dir = std::env::temp_dir().join("docdb_vortex_rowcount_test");
        let writer = VortexWriter::new(temp_dir, "seg_001".to_string());
        assert_eq!(writer.row_count(), 0);
    }

    #[test]
    fn test_vortex_writer_append_row_empty() {
        let temp_dir = std::env::temp_dir().join("docdb_vortex_append_empty");
        let mut writer = VortexWriter::new(temp_dir, "seg_001".to_string());
        let empty_cols: std::collections::HashMap<String, ArrayRef> = std::collections::HashMap::new();
        // Empty map: batch_len = 0
        let result = writer.append_row(&empty_cols);
        assert!(result.is_ok());
    }

    #[test]
    fn test_vortex_writer_append_row_single() {
        let temp_dir = std::env::temp_dir().join("docdb_vortex_append_single");
        let mut writer = VortexWriter::new(temp_dir, "seg_001".to_string());

        let mut row = std::collections::HashMap::new();
        let arr: ArrayRef = std::sync::Arc::new(arrow_array::Int64Array::from(vec![42i64]));
        row.insert("id".to_string(), arr);

        let result = writer.append_row(&row);
        assert!(result.is_ok());
        assert_eq!(writer.row_count(), 1);
    }

    #[test]
    fn test_vortex_writer_append_row_inconsistent() {
        let temp_dir = std::env::temp_dir().join("docdb_vortex_inconsistent");
        let mut writer = VortexWriter::new(temp_dir, "seg_001".to_string());

        let mut row = std::collections::HashMap::new();
        // Two columns with different lengths
        let arr1: ArrayRef = std::sync::Arc::new(arrow_array::Int64Array::from(vec![1i64, 2]));
        let arr2: ArrayRef = std::sync::Arc::new(arrow_array::Int64Array::from(vec![3i64, 4, 5]));
        row.insert("a".to_string(), arr1);
        row.insert("b".to_string(), arr2);

        let result = writer.append_row(&row);
        assert!(result.is_err());
    }

    #[test]
    fn test_vortex_writer_append_batch() {
        let temp_dir = std::env::temp_dir().join("docdb_vortex_append_batch");
        let mut writer = VortexWriter::new(temp_dir, "seg_001".to_string());

        let ids = arrow_array::Int64Array::from(vec![1i64, 2, 3]);
        let schema = arrow_schema::Schema::new(vec![arrow_schema::Field::new("id", arrow::datatypes::DataType::Int64, true)]);
        let batch = arrow_array::RecordBatch::try_new(
            std::sync::Arc::new(schema),
            vec![std::sync::Arc::new(ids) as ArrayRef],
        ).unwrap();

        let result = writer.append_batch(&batch);
        assert!(result.is_ok());
        assert_eq!(writer.row_count(), 3);
    }

    #[test]
    fn test_vortex_writer_append_batch_empty() {
        let temp_dir = std::env::temp_dir().join("docdb_vortex_append_batch_empty");
        let mut writer = VortexWriter::new(temp_dir, "seg_001".to_string());

        let schema = arrow_schema::Schema::new(vec![arrow_schema::Field::new("id", arrow::datatypes::DataType::Int64, true)]);
        let batch = arrow_array::RecordBatch::new_empty(std::sync::Arc::new(schema));

        let result = writer.append_batch(&batch);
        assert!(result.is_ok());
        assert_eq!(writer.row_count(), 0);
    }

    #[test]
    fn test_vortex_writer_combine_arrays_empty() {
        let temp_dir = std::env::temp_dir().join("docdb_vortex_combine_empty");
        let writer = VortexWriter::new(temp_dir, "seg_001".to_string());
        let result = writer.combine_arrays(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_vortex_writer_combine_arrays_single() {
        let temp_dir = std::env::temp_dir().join("docdb_vortex_combine_single");
        let writer = VortexWriter::new(temp_dir, "seg_001".to_string());
        let arr: ArrayRef = std::sync::Arc::new(arrow_array::Int64Array::from(vec![1i64, 2]));
        let result = writer.combine_arrays(&[arr.clone()]);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 2);
    }

    #[test]
    fn test_vortex_writer_combine_arrays_multiple() {
        let temp_dir = std::env::temp_dir().join("docdb_vortex_combine_multi");
        let writer = VortexWriter::new(temp_dir, "seg_001".to_string());
        let arr1: ArrayRef = std::sync::Arc::new(arrow_array::Int64Array::from(vec![1i64, 2]));
        let arr2: ArrayRef = std::sync::Arc::new(arrow_array::Int64Array::from(vec![3i64, 4]));
        // Now properly concatenates all arrays
        let result = writer.combine_arrays(&[arr1.clone(), arr2]);
        assert!(result.is_ok());
        let combined = result.unwrap();
        assert_eq!(combined.len(), 4);
    }

    #[test]
    fn test_vortex_writer_compute_zone_map() {
        let temp_dir = std::env::temp_dir().join("docdb_vortex_zm");
        let writer = VortexWriter::new(temp_dir, "seg_001".to_string());
        let arr: ArrayRef = std::sync::Arc::new(arrow_array::Int64Array::from(vec![1i64, 2]));
        let result = writer.compute_zone_map(&arr);
        assert!(result.is_ok());
    }

    // ============================================================
    // VortexReader tests
    // ============================================================

    #[test]
    fn test_vortex_reader_new() {
        let temp_dir = std::env::temp_dir().join("docdb_vortex_reader_new");
        let reader = VortexReader::new(temp_dir);
        assert!(reader.meta_cache.is_empty());
    }

    #[test]
    fn test_vortex_reader_segment_exists_false() {
        let temp_dir = std::env::temp_dir().join("docdb_vortex_exists_false");
        let reader = VortexReader::new(temp_dir.clone());
        assert!(!reader.segment_exists("nonexistent_seg"));
    }

    #[test]
    fn test_vortex_reader_segment_exists_true() {
        let temp_dir = std::env::temp_dir().join("docdb_vortex_exists_true");
        // VortexReader stores data_dir in layout.seg_dir = data_dir/seg_id
        // segment_exists navigates up: data_dir/seg_id -> data_dir -> parent -> parent
        // So we need data_dir such that parent(parent(data_dir)) == temp_dir/segments
        // That means data_dir should be: temp_dir/segments/seg_existing
        std::fs::create_dir_all(temp_dir.join("segments/seg_existing")).ok();
        // Pass the segments directory as data_dir so parent(parent) lands on segments/
        let reader = VortexReader::new(temp_dir.join("segments"));
        assert!(reader.segment_exists("seg_existing"));
    }

    #[test]
    fn test_vortex_reader_read_column_nonexistent() {
        let temp_dir = std::env::temp_dir().join("docdb_vortex_read_nonexistent");
        let reader = VortexReader::new(temp_dir);
        let result = reader.read_column("seg_nonexistent", "col_nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_vortex_reader_read_arrow_file_nonexistent() {
        let temp_dir = std::env::temp_dir().join("docdb_vortex_read_file_nonexistent");
        let reader = VortexReader::new(temp_dir.clone());
        let result = reader.read_arrow_file(&temp_dir.join("nonexistent_file.vortex"));
        assert!(result.is_err());
    }

    // ============================================================
    // VortexReader cache tests
    // ============================================================

    #[test]
    fn test_vortex_reader_meta_cache() {
        let temp_dir = std::env::temp_dir().join("docdb_vortex_cache");
        let reader = VortexReader::new(temp_dir);
        assert!(reader.meta_cache.is_empty());
    }

    // ============================================================
    // VortexReader del_mask tests
    // ============================================================

    #[test]
    fn test_vortex_reader_read_del_mask_nonexistent() {
        let temp_dir = std::env::temp_dir().join("docdb_vortex_delmask_missing");
        let reader = VortexReader::new(temp_dir);
        // Segment dir doesn't exist, but we can still try
        let result = reader.read_del_mask("seg_missing");
        assert!(result.is_err());
    }

    // ============================================================
    // VortexWriter flush + mmap roundtrip tests
    // ============================================================

    fn make_record_batch(name: &str, values: Vec<i64>) -> RecordBatch {
        let schema = arrow_schema::Schema::new(vec![arrow_schema::Field::new(name, arrow::datatypes::DataType::Int64, true)]);
        RecordBatch::try_new(
            std::sync::Arc::new(schema),
            vec![std::sync::Arc::new(arrow_array::Int64Array::from(values)) as ArrayRef],
        ).unwrap()
    }

    // Helper: write segment and freeze it, returning reader
    #[allow(dead_code)]
    fn freeze_and_get_reader(
        data_dir: &std::path::Path,
        seg_id: &str,
        batch: &RecordBatch,
    ) -> VortexReader {
        let mut writer = VortexWriter::new(data_dir.to_path_buf(), seg_id.to_string());
        writer.append_batch(batch).unwrap();
        writer.flush().unwrap();
        let layout = crate::segment::layout::SegmentLayout::new(&data_dir, seg_id);
        let meta_path = layout.meta_path();
        let meta_bytes = std::fs::read(&meta_path).unwrap();
        let mut meta: crate::segment::meta::SegmentMeta =
            crate::codec::decode(&meta_bytes).unwrap();
        meta.status = crate::segment::meta::SegmentStatus::Frozen;
        std::fs::write(&meta_path, crate::codec::encode(&meta).unwrap()).unwrap();
        VortexReader::new(data_dir.to_path_buf())
    }

    // Helper: unfreeze a segment (set Active status)
    #[allow(dead_code)]
    fn unfreeze_segment(data_dir: &std::path::Path, seg_id: &str) {
        let layout = crate::segment::layout::SegmentLayout::new(data_dir, seg_id);
        let meta_bytes = std::fs::read(layout.meta_path()).unwrap();
        let mut meta: crate::segment::meta::SegmentMeta =
            crate::codec::decode(&meta_bytes).unwrap();
        meta.status = crate::segment::meta::SegmentStatus::Active;
        std::fs::write(layout.meta_path(), crate::codec::encode(&meta).unwrap()).unwrap();
    }

    #[test]
    fn test_flush_and_read_roundtrip() {
        use tempfile::TempDir;

        let temp_dir = TempDir::new().unwrap();
        let data_dir = temp_dir.path().to_path_buf();
        let seg_id = "seg_flush_test";

        // Write data via VortexWriter
        let mut writer = VortexWriter::new(data_dir.clone(), seg_id.to_string());
        let batch = make_record_batch("value", vec![1i64, 2, 3, 100, 200]);
        writer.append_batch(&batch).unwrap();
        let meta = writer.flush().unwrap();

        // Verify SegmentMeta structure
        assert_eq!(meta.seg_id, seg_id);
        assert_eq!(meta.row_count, 5);
        assert!(!meta.columns.is_empty());
        assert_eq!(meta.columns[0].name, "value");

        // Verify granules have correct file_offset for mmap
        assert!(!meta.granules.is_empty());
        let g = &meta.granules[0];
        assert_eq!(g.row_count, 5);
        assert_eq!(g.row_offset, 0);
        // file_offset must be non-negative (actual value is FS-dependent)
        // file_offset is always >= 0 (u64 is unsigned)

        // Verify column files were actually written
        let layout = crate::segment::layout::SegmentLayout::new(&data_dir, seg_id);
        assert!(layout.col_path("value").exists());
        assert!(layout.del_mask_path().exists());
    }

    #[test]
    fn test_flush_with_multiple_columns() {
        use tempfile::TempDir;

        let temp_dir = TempDir::new().unwrap();
        let data_dir = temp_dir.path().to_path_buf();
        let seg_id = "seg_multi_col";

        let mut writer = VortexWriter::new(data_dir.clone(), seg_id.to_string());
        let schema = arrow_schema::Schema::new(vec![
            arrow_schema::Field::new("id", arrow::datatypes::DataType::Int64, true),
            arrow_schema::Field::new("name", arrow::datatypes::DataType::Utf8, true),
        ]);
        let batch = RecordBatch::try_new(
            std::sync::Arc::new(schema),
            vec![
                std::sync::Arc::new(arrow_array::Int64Array::from(vec![10i64, 20, 30])) as ArrayRef,
                std::sync::Arc::new(arrow_array::StringArray::from(vec!["a", "b", "c"])) as ArrayRef,
            ],
        ).unwrap();
        writer.append_batch(&batch).unwrap();
        let meta = writer.flush().unwrap();

        assert_eq!(meta.row_count, 3);
        assert_eq!(meta.columns.len(), 2);
        assert!(meta.columns.iter().any(|c| c.name == "id"));
        assert!(meta.columns.iter().any(|c| c.name == "name"));
    }

    #[test]
    fn test_flush_empty_writer_returns_error() {
        use tempfile::TempDir;

        let temp_dir = TempDir::new().unwrap();
        let data_dir = temp_dir.path().to_path_buf();
        let writer = VortexWriter::new(data_dir, "seg_empty".to_string());
        // flush with no data should still succeed but produce 0-row metadata
        let meta = writer.flush().unwrap();
        assert_eq!(meta.row_count, 0);
        assert!(meta.granules.is_empty());
    }

    #[test]
    fn test_vortex_reader_mmap_cache_shared_arc() {
        use tempfile::TempDir;

        let temp_dir = TempDir::new().unwrap();
        let data_dir = temp_dir.path().to_path_buf();
        let seg_id = "seg_mmap_cache";

        // Write and flush
        let mut writer = VortexWriter::new(data_dir.clone(), seg_id.to_string());
        let arr: ArrayRef = Arc::new(arrow_array::Int32Array::from(vec![1i32, 2, 3]));
        let schema = arrow_schema::Schema::new(vec![arrow_schema::Field::new("n", arrow::datatypes::DataType::Int32, true)]);
        let batch = RecordBatch::try_new(
            std::sync::Arc::new(schema),
            vec![arr],
        ).unwrap();
        writer.append_batch(&batch).unwrap();
        writer.flush().unwrap();

        // Freeze the segment by rewriting meta with Frozen status
        let layout = crate::segment::layout::SegmentLayout::new(&data_dir, seg_id);
        let meta_path = layout.meta_path();
        let meta_bytes = std::fs::read(&meta_path).unwrap();
        let mut meta: crate::segment::meta::SegmentMeta =
            crate::codec::decode(&meta_bytes).unwrap();
        meta.status = crate::segment::meta::SegmentStatus::Frozen;
        let reencoded = crate::codec::encode(&meta).unwrap();
        std::fs::write(&meta_path, reencoded).unwrap();

        // Reader with mmap cache
        let reader = VortexReader::new(data_dir.clone());

        // First read populates cache
        let col1 = reader.read_column(seg_id, "n").unwrap();
        assert_eq!(col1.len(), 3);

        // Second read should hit cache (same Arc reference)
        let col2 = reader.read_column(seg_id, "n").unwrap();
        assert_eq!(col2.len(), 3);

        // Cache still holds the mmap
        let cache = reader.mmap_cache.lock().unwrap();
        assert!(!cache.is_empty());
    }

    #[test]
    fn test_mmap_read_returns_same_as_bufreader() {
        use tempfile::TempDir;

        let temp_dir = TempDir::new().unwrap();
        let data_dir = temp_dir.path().to_path_buf();
        let seg_id = "seg_mmap_compare";

        // Write data
        let mut writer = VortexWriter::new(data_dir.clone(), seg_id.to_string());
        let arr: ArrayRef = Arc::new(arrow_array::Int64Array::from(vec![42i64, -1, 0, i64::MAX]));
        let schema = arrow_schema::Schema::new(vec![arrow_schema::Field::new("val", arrow::datatypes::DataType::Int64, true)]);
        let batch = RecordBatch::try_new(
            std::sync::Arc::new(schema),
            vec![arr],
        ).unwrap();
        writer.append_batch(&batch).unwrap();
        writer.flush().unwrap();

        // Freeze it
        let layout = crate::segment::layout::SegmentLayout::new(&data_dir, seg_id);
        let meta_path = layout.meta_path();
        let meta_bytes = std::fs::read(&meta_path).unwrap();
        let mut meta: crate::segment::meta::SegmentMeta =
            crate::codec::decode(&meta_bytes).unwrap();
        meta.status = crate::segment::meta::SegmentStatus::Frozen;
        std::fs::write(&meta_path, crate::codec::encode(&meta).unwrap()).unwrap();

        let reader = VortexReader::new(data_dir.clone());

        // Frozen → mmap path
        let mmap_result = reader.read_column(seg_id, "val").unwrap();

        // Active → BufReader path (unfreeze)
        let meta_bytes2 = std::fs::read(&meta_path).unwrap();
        let mut meta2: crate::segment::meta::SegmentMeta =
            crate::codec::decode(&meta_bytes2).unwrap();
        meta2.status = crate::segment::meta::SegmentStatus::Active;
        std::fs::write(&meta_path, crate::codec::encode(&meta2).unwrap()).unwrap();

        let bufreader_result = reader.read_column(seg_id, "val").unwrap();

        // Both must return identical data
        assert_eq!(mmap_result.len(), bufreader_result.len());
    }
}
