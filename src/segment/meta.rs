//! Segment 和 Granule 元数据定义

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use bincode_next::{Encode, Decode};
use strum::{EnumString, Display};
use arrow_array::Array;

/// Segment 元数据
#[derive(Debug, Clone, Serialize, Deserialize, Encode, Decode)]
pub struct SegmentMeta {
    /// Segment ID
    pub seg_id: String,
    /// 表名
    pub table: String,
    /// 行数
    pub row_count: u64,
    /// 未压缩大小（字节）
    pub uncompressed_size: u64,
    /// 压缩后大小（字节）
    pub compressed_size: u64,
    /// 列定义
    pub columns: Vec<ColumnDef>,
    /// Granule 列表
    pub granules: Vec<GranuleMeta>,
    /// 删除行数
    pub deleted_rows: u64,
    /// 删除率
    pub del_ratio: f64,
    /// 状态: Active, Compactable, Frozen
    pub status: SegmentStatus,
    /// 创建时间戳
    pub created_at: u64,
    /// 最后更新时间戳
    pub updated_at: u64,
    /// 主键范围 (min, max)
    pub pk_range: Option<(Vec<u8>, Vec<u8>)>,
}

impl SegmentMeta {
    /// 创建新的 SegmentMeta
    pub fn new(seg_id: String, table: String, columns: Vec<ColumnDef>) -> Self {
        let now = crate::codec::current_timestamp_secs();
            
        Self {
            seg_id,
            table,
            row_count: 0,
            uncompressed_size: 0,
            compressed_size: 0,
            columns,
            granules: Vec::new(),
            deleted_rows: 0,
            del_ratio: 0.0,
            status: SegmentStatus::Active,
            created_at: now,
            updated_at: now,
            pk_range: None,
        }
    }

    /// 添加 granule
    pub fn add_granule(&mut self, granule: GranuleMeta) {
        self.row_count += granule.row_count as u64;
        self.granules.push(granule);
        self.updated_at = crate::codec::current_timestamp_secs();
    }

    /// 更新删除统计
    pub fn update_del_stats(&mut self, deleted_count: u64) {
        self.deleted_rows += deleted_count;
        if self.row_count > 0 {
            self.del_ratio = (self.deleted_rows as f64 / self.row_count as f64).min(1.0);
        }
    }

    /// 检查是否需要 compaction
    pub fn needs_compaction(&self, threshold: f64) -> bool {
        self.del_ratio > threshold
    }

    /// 计算所有列文件的总大小（字节）。
    /// 直接从磁盘读取实际文件大小，不依赖内部的 uncompressed_size / compressed_size（这两个字段当前为 0）。
    pub fn compute_file_sizes(&self, data_dir: &std::path::Path) -> std::io::Result<(u64, u64)> {
        use std::fs;
        let seg_path = data_dir.join("segments").join(&self.seg_id);
        let mut total = 0u64;
        for col in &self.columns {
            let path = seg_path.join(format!("{}.vortex", col.name));
            if path.exists() {
                total += fs::metadata(&path)?.len();
            }
        }
        Ok((total, total))
    }
}

/// Segment 状态
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Encode, Decode, EnumString, Display)]
#[serde(rename_all = "snake_case")]
pub enum SegmentStatus {
    /// 活跃状态，可接受写入
    Active,
    /// 可压缩状态
    Compactable,
    /// 冻结状态，只读
    Frozen,
    /// Compaction 进行中（三阶段中的 Prepare/Commit 阶段）
    Compacting,
}

/// Granule 元数据
#[derive(Debug, Clone, Serialize, Deserialize, Encode, Decode)]
pub struct GranuleMeta {
    /// Granule ID (在 segment 内递增)
    pub granule_id: u32,
    /// 在 segment 内的起始行号
    pub row_offset: u64,
    /// 该 granule 的行数
    pub row_count: u32,
    /// 在文件中的偏移量
    pub file_offset: u64,
    /// 压缩后大小（字节）
    pub compressed_size: u32,
    /// Zone Map 统计信息（granule 级）
    pub zone_map: ZoneMapStats,
    /// Block 级统计信息（每 1024 行一个 block 的 min/max）
    pub block_stats: Vec<BlockStats>,
}

impl GranuleMeta {
    /// 创建新的 GranuleMeta
    pub fn new(granule_id: u32, row_offset: u64, row_count: u32) -> Self {
        Self {
            granule_id,
            row_offset,
            row_count,
            file_offset: 0,
            compressed_size: 0,
            zone_map: ZoneMapStats::default(),
            block_stats: Vec::new(),
        }
    }

    /// 创建带文件偏移量的 GranuleMeta（用于 mmap 稀疏读取）
    pub fn with_offset(granule_id: u32, row_offset: u64, row_count: u32, file_offset: u64) -> Self {
        Self {
            granule_id,
            row_offset,
            row_count,
            file_offset,
            compressed_size: 0,
            zone_map: ZoneMapStats::default(),
            block_stats: Vec::new(),
        }
    }
}

/// 列定义
#[derive(Debug, Clone, Serialize, Deserialize, Encode, Decode)]
pub struct ColumnDef {
    /// 列名
    pub name: String,
    /// 数据类型
    pub dtype: DataType,
    /// 编码类型
    pub encoding: EncodingType,
}

impl ColumnDef {
    /// 创建新的列定义
    pub fn new(name: String, dtype: DataType) -> Self {
        Self {
            name,
            dtype,
            encoding: EncodingType::default_for_dtype(&dtype),
        }
    }

    /// 创建带编码的列定义
    pub fn with_encoding(name: String, dtype: DataType, encoding: EncodingType) -> Self {
        Self { name, dtype, encoding }
    }
}

/// 数据类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Encode, Decode)]
#[serde(rename_all = "snake_case")]
pub enum DataType {
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Float32,
    Float64,
    Bool,
    Utf8,
    LargeUtf8,
    Binary,
    LargeBinary,
    Date32,
    Date64,
    TimestampSecond,
    TimestampMillisecond,
    TimestampMicrosecond,
    TimestampNanosecond,
}

impl DataType {
    /// 转换为 Arrow 数据类型
    pub fn to_arrow(&self) -> arrow::datatypes::DataType {
        use arrow::datatypes::{DataType as ArrowDt, TimeUnit};
        match self {
            DataType::Int8 => ArrowDt::Int8,
            DataType::Int16 => ArrowDt::Int16,
            DataType::Int32 => ArrowDt::Int32,
            DataType::Int64 => ArrowDt::Int64,
            DataType::UInt8 => ArrowDt::UInt8,
            DataType::UInt16 => ArrowDt::UInt16,
            DataType::UInt32 => ArrowDt::UInt32,
            DataType::UInt64 => ArrowDt::UInt64,
            DataType::Float32 => ArrowDt::Float32,
            DataType::Float64 => ArrowDt::Float64,
            DataType::Bool => ArrowDt::Boolean,
            DataType::Utf8 => ArrowDt::Utf8,
            DataType::LargeUtf8 => ArrowDt::LargeUtf8,
            DataType::Binary => ArrowDt::Binary,
            DataType::LargeBinary => ArrowDt::LargeBinary,
            DataType::Date32 => ArrowDt::Date32,
            DataType::Date64 => ArrowDt::Date64,
            DataType::TimestampSecond => ArrowDt::Timestamp(TimeUnit::Second, None),
            DataType::TimestampMillisecond => ArrowDt::Timestamp(TimeUnit::Millisecond, None),
            DataType::TimestampMicrosecond => ArrowDt::Timestamp(TimeUnit::Microsecond, None),
            DataType::TimestampNanosecond => ArrowDt::Timestamp(TimeUnit::Nanosecond, None),
        }
    }

    /// 从 Arrow 数据类型创建
    pub fn from_arrow(dtype: &arrow::datatypes::DataType) -> Self {
        use arrow::datatypes::{DataType as ArrowDt, TimeUnit};
        match dtype {
            ArrowDt::Int8 => DataType::Int8,
            ArrowDt::Int16 => DataType::Int16,
            ArrowDt::Int32 => DataType::Int32,
            ArrowDt::Int64 => DataType::Int64,
            ArrowDt::UInt8 => DataType::UInt8,
            ArrowDt::UInt16 => DataType::UInt16,
            ArrowDt::UInt32 => DataType::UInt32,
            ArrowDt::UInt64 => DataType::UInt64,
            ArrowDt::Float32 => DataType::Float32,
            ArrowDt::Float64 => DataType::Float64,
            ArrowDt::Boolean => DataType::Bool,
            ArrowDt::Utf8 => DataType::Utf8,
            ArrowDt::LargeUtf8 => DataType::LargeUtf8,
            ArrowDt::Binary => DataType::Binary,
            ArrowDt::LargeBinary => DataType::LargeBinary,
            ArrowDt::Date32 => DataType::Date32,
            ArrowDt::Date64 => DataType::Date64,
            ArrowDt::Timestamp(TimeUnit::Second, _) => DataType::TimestampSecond,
            ArrowDt::Timestamp(TimeUnit::Millisecond, _) => DataType::TimestampMillisecond,
            ArrowDt::Timestamp(TimeUnit::Microsecond, _) => DataType::TimestampMicrosecond,
            ArrowDt::Timestamp(TimeUnit::Nanosecond, _) => DataType::TimestampNanosecond,
            _ => DataType::Binary,
        }
    }
}

/// 编码类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Encode, Decode)]
#[serde(rename_all = "snake_case")]
pub enum EncodingType {
    /// 不编码
    Raw,
    /// Delta 编码
    Delta,
    /// RLE 编码
    Rle,
    /// Dictionary 编码
    Dict,
    /// ALP 编码 (Adaptive Lossless compression for integers)
    Alp,
    /// Gorilla 编码 (用于浮点数)
    Gorilla,
    /// Bitpacked 编码
    Bitpacked,
    /// Zstd 压缩
    Zstd,
    /// LZ4 压缩
    Lz4,
}

impl EncodingType {
    /// 根据数据类型返回默认编码
    pub fn default_for_dtype(dtype: &DataType) -> Self {
        match dtype {
            DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64 |
            DataType::UInt8 | DataType::UInt16 | DataType::UInt32 | DataType::UInt64 => {
                EncodingType::Delta
            }
            DataType::Float32 | DataType::Float64 => EncodingType::Gorilla,
            DataType::Bool => EncodingType::Rle,
            _ => EncodingType::Raw,
        }
    }
}

/// Zone Map 统计信息
#[derive(Debug, Clone, Default, Serialize, Deserialize, Encode, Decode)]
pub struct ZoneMapStats {
    /// 列名到统计信息的映射
    pub stats: HashMap<String, ColumnStats>,
}

impl ZoneMapStats {
    /// 创建空的 ZoneMapStats
    pub fn new() -> Self {
        Self {
            stats: HashMap::new(),
        }
    }

    /// 添加列统计
    pub fn add_column_stats(&mut self, col: &str, stats: ColumnStats) {
        self.stats.insert(col.to_string(), stats);
    }

    /// 获取列统计
    pub fn get(&self, col: &str) -> Option<&ColumnStats> {
        self.stats.get(col)
    }

    /// 检查是否可以裁剪
    pub fn can_prune(&self, col: &str, op: &CompareOp, val: &[u8]) -> bool {
        let Some(stats) = self.stats.get(col) else {
            return false; // 没有统计信息，不能裁剪
        };
        
        match op {
            CompareOp::Eq => {
                // 不能裁剪，因为可能有相等的值
                false
            }
            CompareOp::Ne => {
                // 不能裁剪
                false
            }
            CompareOp::Lt => {
                // 如果最小值 >= val，则全部 < val
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

/// Block 级统计信息（nodedb per-block stats）
///
/// 每个 granule 内部再细分为多个 block，每个 block 记录热点列的 min/max。
/// 用于 granule 内谓词下推：如果 block 的 min/max 与查询条件不匹配，整 block 跳过。
///
/// 每 1024 行一个 block。存储空间：~32 bytes/block。
#[derive(Debug, Clone, Default, Serialize, Deserialize, Encode, Decode)]
pub struct BlockStats {
    /// Block 序号（在 granule 内递增）
    pub block_id: u32,
    /// 起始行号（相对 granule 起始位置）
    pub row_offset: u64,
    /// 行数
    pub row_count: u32,
    /// 热点列的 min/max 统计（列名 → (min_bytes, max_bytes)）
    pub column_stats: HashMap<String, (Vec<u8>, Vec<u8>)>,
}

impl BlockStats {
    /// 创建新的 BlockStats
    pub fn new(block_id: u32, row_offset: u64, row_count: u32) -> Self {
        Self {
            block_id,
            row_offset,
            row_count,
            column_stats: HashMap::new(),
        }
    }

    /// 添加列统计
    pub fn add_column(&mut self, col: &str, min: Vec<u8>, max: Vec<u8>) {
        self.column_stats.insert(col.to_string(), (min, max));
    }

    /// 检查 block 是否可能包含满足条件的行
    /// 返回 true = 可能包含，false = 肯定不包含
    pub fn may_contain(&self, col: &str, op: &CompareOp, val: &[u8]) -> bool {
        let Some((ref block_min, ref block_max)) = self.column_stats.get(col) else {
            return true;
        };

        match op {
            CompareOp::Eq | CompareOp::Ne => true,
            CompareOp::Lt => block_min.as_slice() < val,
            CompareOp::Le => block_min.as_slice() <= val,
            CompareOp::Gt => block_max.as_slice() > val,
            CompareOp::Ge => block_max.as_slice() >= val,
        }
    }
}

/// 从 RecordBatch 计算 BlockStats（每 1024 行一个 block）
pub fn compute_block_stats(batch: &arrow_array::RecordBatch, block_size: u32) -> Vec<BlockStats> {
    let num_rows = batch.num_rows() as u32;
    let num_blocks = (num_rows + block_size - 1) / block_size;

    (0..num_blocks)
        .map(|block_id| {
            let start = (block_id * block_size) as usize;
            let end = (start + block_size as usize).min(batch.num_rows());
            let count = (end - start) as u32;
            let row_offset = (block_id * block_size) as u64;

            let mut stats = BlockStats::new(block_id, row_offset, count);

            // 对数值列计算统计
            let schema = batch.schema();
            for i in 0..batch.num_columns() {
                let field_ref = schema.field(i);
                let dtype = field_ref.data_type();
                if dtype == &arrow::datatypes::DataType::Int32
                    || dtype == &arrow::datatypes::DataType::Int64
                    || dtype == &arrow::datatypes::DataType::Float64
                    || dtype == &arrow::datatypes::DataType::Float32
                    || dtype == &arrow::datatypes::DataType::UInt32
                    || dtype == &arrow::datatypes::DataType::UInt64
                {
                    let (min, max) = compute_column_min_max(batch.column(i), start, end);
                    if let (Some(min_v), Some(max_v)) = (min, max) {
                        stats.add_column(field_ref.name(), min_v, max_v);
                    }
                }
            }

            stats
        })
        .collect()
}

/// 计算数组片段的 min/max
fn compute_column_min_max(
    array: &dyn arrow::array::Array,
    start: usize,
    end: usize,
) -> (Option<Vec<u8>>, Option<Vec<u8>>) {
    use arrow::array::{Int32Array, Int64Array, Float64Array, Float32Array, UInt32Array, UInt64Array};

    if let Some(arr) = array.as_any().downcast_ref::<Int32Array>() {
        let slice = arr.slice(start, end - start);
        let mut min_v: Option<i32> = None;
        let mut max_v: Option<i32> = None;
        for i in 0..slice.len() {
            if !slice.is_null(i) {
                let v = slice.value(i);
                min_v = Some(min_v.map(|m| m.min(v)).unwrap_or(v));
                max_v = Some(max_v.map(|m| m.max(v)).unwrap_or(v));
            }
        }
        return (min_v.map(|v| v.to_le_bytes().to_vec()), max_v.map(|v| v.to_le_bytes().to_vec()));
    }
    if let Some(arr) = array.as_any().downcast_ref::<Int64Array>() {
        let slice = arr.slice(start, end - start);
        let mut min_v: Option<i64> = None;
        let mut max_v: Option<i64> = None;
        for i in 0..slice.len() {
            if !slice.is_null(i) {
                let v = slice.value(i);
                min_v = Some(min_v.map(|m| m.min(v)).unwrap_or(v));
                max_v = Some(max_v.map(|m| m.max(v)).unwrap_or(v));
            }
        }
        return (min_v.map(|v| v.to_le_bytes().to_vec()), max_v.map(|v| v.to_le_bytes().to_vec()));
    }
    if let Some(arr) = array.as_any().downcast_ref::<Float64Array>() {
        let slice = arr.slice(start, end - start);
        let mut min_v: Option<f64> = None;
        let mut max_v: Option<f64> = None;
        for i in 0..slice.len() {
            if !slice.is_null(i) {
                let v = slice.value(i);
                min_v = Some(min_v.map(|m| m.min(v)).unwrap_or(v));
                max_v = Some(max_v.map(|m| m.max(v)).unwrap_or(v));
            }
        }
        return (min_v.map(|v| v.to_le_bytes().to_vec()), max_v.map(|v| v.to_le_bytes().to_vec()));
    }
    if let Some(arr) = array.as_any().downcast_ref::<Float32Array>() {
        let slice = arr.slice(start, end - start);
        let mut min_v: Option<f32> = None;
        let mut max_v: Option<f32> = None;
        for i in 0..slice.len() {
            if !slice.is_null(i) {
                let v = slice.value(i);
                min_v = Some(min_v.map(|m| m.min(v)).unwrap_or(v));
                max_v = Some(max_v.map(|m| m.max(v)).unwrap_or(v));
            }
        }
        return (min_v.map(|v| v.to_le_bytes().to_vec()), max_v.map(|v| v.to_le_bytes().to_vec()));
    }
    if let Some(arr) = array.as_any().downcast_ref::<UInt32Array>() {
        let slice = arr.slice(start, end - start);
        let mut min_v: Option<u32> = None;
        let mut max_v: Option<u32> = None;
        for i in 0..slice.len() {
            if !slice.is_null(i) {
                let v = slice.value(i);
                min_v = Some(min_v.map(|m| m.min(v)).unwrap_or(v));
                max_v = Some(max_v.map(|m| m.max(v)).unwrap_or(v));
            }
        }
        return (min_v.map(|v| v.to_le_bytes().to_vec()), max_v.map(|v| v.to_le_bytes().to_vec()));
    }
    if let Some(arr) = array.as_any().downcast_ref::<UInt64Array>() {
        let slice = arr.slice(start, end - start);
        let mut min_v: Option<u64> = None;
        let mut max_v: Option<u64> = None;
        for i in 0..slice.len() {
            if !slice.is_null(i) {
                let v = slice.value(i);
                min_v = Some(min_v.map(|m| m.min(v)).unwrap_or(v));
                max_v = Some(max_v.map(|m| m.max(v)).unwrap_or(v));
            }
        }
        return (min_v.map(|v| v.to_le_bytes().to_vec()), max_v.map(|v| v.to_le_bytes().to_vec()));
    }

    (None, None)
}

/// 列统计信息
#[derive(Debug, Clone, Serialize, Deserialize, Encode, Decode)]
pub struct ColumnStats {
    /// 最小值 (bytes)
    pub min: Option<Vec<u8>>,
    /// 最大值 (bytes)
    pub max: Option<Vec<u8>>,
    /// null 计数
    pub null_count: u32,
    /// 值的总数 (用于 sum)
    pub sum: Option<Vec<u8>>,
    /// 不同的值数量
    pub distinct_count: Option<u32>,
}

impl Default for ColumnStats {
    fn default() -> Self {
        Self {
            min: None,
            max: None,
            null_count: 0,
            sum: None,
            distinct_count: None,
        }
    }
}

/// 比较操作
#[derive(Debug, Clone, Copy, PartialEq, Eq, EnumString)]
pub enum CompareOp {
    Eq,  // =
    Ne,  // !=
    Lt,  // <
    Le,  // <=
    Gt,  // >
    Ge,  // >=
}

impl CompareOp {
    /// 从字符串创建
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_segment_meta() {
        let columns = vec![
            ColumnDef::new("id".to_string(), DataType::Int64),
            ColumnDef::new("name".to_string(), DataType::Utf8),
        ];

        let mut meta = SegmentMeta::new("seg_001".to_string(), "users".to_string(), columns);

        assert_eq!(meta.row_count, 0);
        assert_eq!(meta.status, SegmentStatus::Active);
        assert_eq!(meta.seg_id, "seg_001");
        assert_eq!(meta.table, "users");
        assert_eq!(meta.deleted_rows, 0);
        assert!((meta.del_ratio - 0.0).abs() < 1e-9);
        assert!(meta.pk_range.is_none());
        assert!(meta.granules.is_empty());
        assert!(meta.created_at > 0);
        assert_eq!(meta.created_at, meta.updated_at);

        let granule = GranuleMeta::new(0, 0, 1000);
        meta.add_granule(granule);

        assert_eq!(meta.row_count, 1000);
        assert_eq!(meta.granules.len(), 1);
    }

    #[test]
    fn test_del_ratio() {
        let columns = vec![ColumnDef::new("id".to_string(), DataType::Int64)];
        let mut meta = SegmentMeta::new("seg_001".to_string(), "users".to_string(), columns);

        meta.row_count = 100;
        meta.update_del_stats(50);

        assert_eq!(meta.del_ratio, 0.5);
        assert!(meta.needs_compaction(0.3));
        assert!(!meta.needs_compaction(0.6));
    }

    // ============================================================
    // SegmentMeta methods
    // ============================================================

    #[test]
    fn test_segment_meta_add_multiple_granules() {
        let columns = vec![ColumnDef::new("id".to_string(), DataType::Int64)];
        let mut meta = SegmentMeta::new("seg_001".to_string(), "users".to_string(), columns);

        meta.add_granule(GranuleMeta::new(0, 0, 500));
        meta.add_granule(GranuleMeta::new(1, 500, 500));
        meta.add_granule(GranuleMeta::new(2, 1000, 500));

        assert_eq!(meta.row_count, 1500);
        assert_eq!(meta.granules.len(), 3);
        assert!(meta.updated_at >= meta.created_at);
    }

    #[test]
    fn test_segment_meta_update_del_stats_zero_rows() {
        let columns = vec![ColumnDef::new("id".to_string(), DataType::Int64)];
        let mut meta = SegmentMeta::new("seg_001".to_string(), "users".to_string(), columns);

        meta.update_del_stats(0);
        assert_eq!(meta.del_ratio, 0.0);

        meta.row_count = 100;
        meta.update_del_stats(0);
        assert_eq!(meta.del_ratio, 0.0);
    }

    #[test]
    fn test_segment_meta_update_del_stats_saturating() {
        let columns = vec![ColumnDef::new("id".to_string(), DataType::Int64)];
        let mut meta = SegmentMeta::new("seg_001".to_string(), "users".to_string(), columns);

        meta.row_count = 100;
        meta.update_del_stats(150); // More than row_count
        assert!(meta.del_ratio <= 1.0, "del_ratio should be clamped to 1.0");
        assert!((meta.del_ratio - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_del_ratio_capped_at_one() {
        let columns = vec![ColumnDef::new("id".to_string(), DataType::Int64)];
        let mut meta = SegmentMeta::new("seg_001".to_string(), "t".to_string(), columns);
        meta.row_count = 100;
        meta.update_del_stats(150); // More deleted than total rows
        assert!(meta.del_ratio <= 1.0);
        assert!((meta.del_ratio - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_segment_meta_needs_compaction_threshold() {
        let columns = vec![ColumnDef::new("id".to_string(), DataType::Int64)];
        let mut meta = SegmentMeta::new("seg_001".to_string(), "users".to_string(), columns);

        meta.row_count = 100;
        meta.update_del_stats(25); // 25%
        assert!(meta.needs_compaction(0.20)); // 25% > 20%
        assert!(!meta.needs_compaction(0.30)); // 25% < 30%
        assert!(!meta.needs_compaction(0.25)); // 25% == 25% (strict >)
    }

    // ============================================================
    // GranuleMeta tests
    // ============================================================

    #[test]
    fn test_granule_meta_new() {
        let granule = GranuleMeta::new(5, 1000, 500);
        assert_eq!(granule.granule_id, 5);
        assert_eq!(granule.row_offset, 1000);
        assert_eq!(granule.row_count, 500);
        assert_eq!(granule.file_offset, 0);
        assert_eq!(granule.compressed_size, 0);
    }

    #[test]
    fn test_granule_meta_zero_values() {
        let granule = GranuleMeta::new(0, 0, 0);
        assert_eq!(granule.granule_id, 0);
        assert_eq!(granule.row_offset, 0);
        assert_eq!(granule.row_count, 0);
    }

    // ============================================================
    // ColumnDef tests
    // ============================================================

    #[test]
    fn test_column_def_new_all_types() {
        for dtype in [
            DataType::Int8, DataType::Int16, DataType::Int32, DataType::Int64,
            DataType::UInt8, DataType::UInt16, DataType::UInt32, DataType::UInt64,
            DataType::Float32, DataType::Float64, DataType::Bool,
            DataType::Utf8, DataType::LargeUtf8,
            DataType::Binary, DataType::LargeBinary,
            DataType::Date32, DataType::Date64,
            DataType::TimestampSecond, DataType::TimestampMillisecond,
            DataType::TimestampMicrosecond, DataType::TimestampNanosecond,
        ] {
            let col = ColumnDef::new("test_col".to_string(), dtype);
            assert_eq!(col.name, "test_col");
            assert_eq!(col.dtype, dtype);
            // Encoding should be set
            assert!(matches!(col.encoding, crate::segment::meta::EncodingType::Raw | crate::segment::meta::EncodingType::Delta | crate::segment::meta::EncodingType::Rle | crate::segment::meta::EncodingType::Gorilla));
        }
    }

    #[test]
    fn test_column_def_with_encoding() {
        let col = ColumnDef::with_encoding(
            "my_col".to_string(),
            DataType::Int64,
            EncodingType::Alp,
        );
        assert_eq!(col.name, "my_col");
        assert_eq!(col.dtype, DataType::Int64);
        assert_eq!(col.encoding, EncodingType::Alp);
    }

    // ============================================================
    // DataType to_arrow / from_arrow
    // ============================================================

    #[test]
    fn test_datatype_to_arrow_all_variants() {
        use arrow::datatypes::DataType as ArrowDt;
        use arrow::datatypes::TimeUnit;

        let cases: &[(DataType, ArrowDt)] = &[
            (DataType::Int8, ArrowDt::Int8),
            (DataType::Int16, ArrowDt::Int16),
            (DataType::Int32, ArrowDt::Int32),
            (DataType::Int64, ArrowDt::Int64),
            (DataType::UInt8, ArrowDt::UInt8),
            (DataType::UInt16, ArrowDt::UInt16),
            (DataType::UInt32, ArrowDt::UInt32),
            (DataType::UInt64, ArrowDt::UInt64),
            (DataType::Float32, ArrowDt::Float32),
            (DataType::Float64, ArrowDt::Float64),
            (DataType::Bool, ArrowDt::Boolean),
            (DataType::Utf8, ArrowDt::Utf8),
            (DataType::LargeUtf8, ArrowDt::LargeUtf8),
            (DataType::Binary, ArrowDt::Binary),
            (DataType::LargeBinary, ArrowDt::LargeBinary),
            (DataType::Date32, ArrowDt::Date32),
            (DataType::Date64, ArrowDt::Date64),
            (DataType::TimestampSecond, ArrowDt::Timestamp(TimeUnit::Second, None)),
            (DataType::TimestampMillisecond, ArrowDt::Timestamp(TimeUnit::Millisecond, None)),
            (DataType::TimestampMicrosecond, ArrowDt::Timestamp(TimeUnit::Microsecond, None)),
            (DataType::TimestampNanosecond, ArrowDt::Timestamp(TimeUnit::Nanosecond, None)),
        ];

        for (dtype, expected) in cases {
            assert_eq!(dtype.to_arrow(), *expected, "Failed for {:?}", dtype);
        }
    }

    #[test]
    fn test_datatype_from_arrow_all_variants() {
        use arrow::datatypes::DataType as ArrowDt;
        use arrow::datatypes::TimeUnit;

        let cases: &[(ArrowDt, DataType)] = &[
            (ArrowDt::Int8, DataType::Int8),
            (ArrowDt::Int16, DataType::Int16),
            (ArrowDt::Int32, DataType::Int32),
            (ArrowDt::Int64, DataType::Int64),
            (ArrowDt::UInt8, DataType::UInt8),
            (ArrowDt::UInt16, DataType::UInt16),
            (ArrowDt::UInt32, DataType::UInt32),
            (ArrowDt::UInt64, DataType::UInt64),
            (ArrowDt::Float32, DataType::Float32),
            (ArrowDt::Float64, DataType::Float64),
            (ArrowDt::Boolean, DataType::Bool),
            (ArrowDt::Utf8, DataType::Utf8),
            (ArrowDt::LargeUtf8, DataType::LargeUtf8),
            (ArrowDt::Binary, DataType::Binary),
            (ArrowDt::LargeBinary, DataType::LargeBinary),
            (ArrowDt::Date32, DataType::Date32),
            (ArrowDt::Date64, DataType::Date64),
            (ArrowDt::Timestamp(TimeUnit::Second, None), DataType::TimestampSecond),
            (ArrowDt::Timestamp(TimeUnit::Millisecond, None), DataType::TimestampMillisecond),
            (ArrowDt::Timestamp(TimeUnit::Microsecond, None), DataType::TimestampMicrosecond),
            (ArrowDt::Timestamp(TimeUnit::Nanosecond, None), DataType::TimestampNanosecond),
        ];

        for (arrow_dt, expected) in cases {
            assert_eq!(DataType::from_arrow(arrow_dt), *expected, "Failed for {:?}", arrow_dt);
        }
    }

    #[test]
    fn test_datatype_from_arrow_fallback() {
        use arrow::datatypes::DataType as ArrowDt;
        // LargeList, FixedSizeBinary, etc. should fallback to Binary
        assert_eq!(DataType::from_arrow(&ArrowDt::LargeList(std::sync::Arc::new(arrow::datatypes::Field::new("item", ArrowDt::Int32, true)))), DataType::Binary);
        assert_eq!(DataType::from_arrow(&ArrowDt::FixedSizeBinary(16)), DataType::Binary);
        assert_eq!(DataType::from_arrow(&ArrowDt::Duration(arrow::datatypes::TimeUnit::Second)), DataType::Binary);
        assert_eq!(DataType::from_arrow(&ArrowDt::Null), DataType::Binary);
        assert_eq!(DataType::from_arrow(&ArrowDt::Interval(arrow::datatypes::IntervalUnit::YearMonth)), DataType::Binary);
    }

    // ============================================================
    // EncodingType tests
    // ============================================================

    #[test]
    fn test_encoding_type_default_for_dtype() {
        // Integer types -> Delta
        for dtype in [DataType::Int8, DataType::Int16, DataType::Int32, DataType::Int64,
                      DataType::UInt8, DataType::UInt16, DataType::UInt32, DataType::UInt64] {
            assert_eq!(EncodingType::default_for_dtype(&dtype), EncodingType::Delta);
        }
        // Float types -> Gorilla
        assert_eq!(EncodingType::default_for_dtype(&DataType::Float32), EncodingType::Gorilla);
        assert_eq!(EncodingType::default_for_dtype(&DataType::Float64), EncodingType::Gorilla);
        // Bool -> Rle
        assert_eq!(EncodingType::default_for_dtype(&DataType::Bool), EncodingType::Rle);
        // Others -> Raw
        for dtype in [DataType::Utf8, DataType::LargeUtf8, DataType::Binary, DataType::LargeBinary,
                      DataType::Date32, DataType::Date64, DataType::TimestampSecond,
                      DataType::TimestampMillisecond, DataType::TimestampMicrosecond, DataType::TimestampNanosecond] {
            assert_eq!(EncodingType::default_for_dtype(&dtype), EncodingType::Raw, "Failed for {:?}", dtype);
        }
    }

    // ============================================================
    // ZoneMapStats tests
    // ============================================================

    #[test]
    fn test_zone_map_stats_new() {
        let zm = ZoneMapStats::new();
        assert!(zm.stats.is_empty());
    }

    #[test]
    fn test_zone_map_stats_add_and_get() {
        let mut zm = ZoneMapStats::new();

        let stats = ColumnStats {
            min: Some(vec![10u8]),
            max: Some(vec![100u8]),
            null_count: 0,
            sum: Some(vec![]),
            distinct_count: Some(50),
        };

        zm.add_column_stats("age", stats.clone());
        assert!(zm.get("age").is_some());
        assert_eq!(zm.get("age").unwrap().null_count, 0);

        assert!(zm.get("nonexistent").is_none());
    }

    #[test]
    fn test_zone_map_stats_can_prune_lt() {
        let mut zm = ZoneMapStats::new();
        zm.add_column_stats("age", ColumnStats {
            min: Some(vec![20u8]),
            max: Some(vec![80u8]),
            null_count: 0,
            sum: None,
            distinct_count: None,
        });

        // age < 15: all values are >= 20, so entire zone can be pruned
        let can = zm.can_prune("age", &CompareOp::Lt, &[15u8]);
        assert!(can);

        // age < 30: min is 20, which is NOT >= 30, so cannot prune
        let can = zm.can_prune("age", &CompareOp::Lt, &[30u8]);
        assert!(!can);
    }

    #[test]
    fn test_zone_map_stats_can_prune_le() {
        let mut zm = ZoneMapStats::new();
        zm.add_column_stats("score", ColumnStats {
            min: Some(vec![50u8]),
            max: Some(vec![100u8]),
            null_count: 0,
            sum: None,
            distinct_count: None,
        });

        // score <= 40: min is 50, 50 > 40, so can prune
        let can = zm.can_prune("score", &CompareOp::Le, &[40u8]);
        assert!(can);

        // score <= 55: min is 50, 50 is NOT > 55, so cannot prune
        let can = zm.can_prune("score", &CompareOp::Le, &[55u8]);
        assert!(!can);
    }

    #[test]
    fn test_zone_map_stats_can_prune_gt() {
        let mut zm = ZoneMapStats::new();
        zm.add_column_stats("val", ColumnStats {
            min: Some(vec![10u8]),
            max: Some(vec![50u8]),
            null_count: 0,
            sum: None,
            distinct_count: None,
        });

        // val > 60: max is 50, 50 <= 60, so can prune
        let can = zm.can_prune("val", &CompareOp::Gt, &[60u8]);
        assert!(can);

        // val > 40: max is 50, 50 is NOT < 40, so cannot prune
        let can = zm.can_prune("val", &CompareOp::Gt, &[40u8]);
        assert!(!can);
    }

    #[test]
    fn test_zone_map_stats_can_prune_ge() {
        let mut zm = ZoneMapStats::new();
        zm.add_column_stats("num", ColumnStats {
            min: Some(vec![30u8]),
            max: Some(vec![90u8]),
            null_count: 0,
            sum: None,
            distinct_count: None,
        });

        // num >= 95: max is 90, 90 < 95, so can prune
        let can = zm.can_prune("num", &CompareOp::Ge, &[95u8]);
        assert!(can);

        // num >= 85: max is 90, 90 is NOT < 85, so cannot prune
        let can = zm.can_prune("num", &CompareOp::Ge, &[85u8]);
        assert!(!can);
    }

    #[test]
    fn test_zone_map_stats_can_prune_eq_ne() {
        let mut zm = ZoneMapStats::new();
        zm.add_column_stats("col", ColumnStats {
            min: Some(vec![10u8]),
            max: Some(vec![50u8]),
            null_count: 0,
            sum: None,
            distinct_count: None,
        });

        // Eq never prunes (values might exist)
        assert!(!zm.can_prune("col", &CompareOp::Eq, &[25u8]));
        // Ne never prunes
        assert!(!zm.can_prune("col", &CompareOp::Ne, &[25u8]));
    }

    #[test]
    fn test_zone_map_stats_can_prune_no_stats() {
        let zm = ZoneMapStats::new();
        assert!(!zm.can_prune("nonexistent", &CompareOp::Lt, &[100u8]));
    }

    #[test]
    fn test_zone_map_stats_can_prune_none_min() {
        let mut zm = ZoneMapStats::new();
        zm.add_column_stats("col", ColumnStats {
            min: None,
            max: Some(vec![100u8]),
            null_count: 5,
            sum: None,
            distinct_count: None,
        });
        // No min value -> cannot prune Lt
        assert!(!zm.can_prune("col", &CompareOp::Lt, &[50u8]));
    }

    #[test]
    fn test_zone_map_stats_can_prune_none_max() {
        let mut zm = ZoneMapStats::new();
        zm.add_column_stats("col", ColumnStats {
            min: Some(vec![10u8]),
            max: None,
            null_count: 5,
            sum: None,
            distinct_count: None,
        });
        // No max value -> cannot prune Gt
        assert!(!zm.can_prune("col", &CompareOp::Gt, &[50u8]));
    }

    // ============================================================
    // ColumnStats tests
    // ============================================================

    #[test]
    fn test_column_stats_default() {
        let stats = ColumnStats::default();
        assert!(stats.min.is_none());
        assert!(stats.max.is_none());
        assert_eq!(stats.null_count, 0);
        assert!(stats.sum.is_none());
        assert!(stats.distinct_count.is_none());
    }

    // ============================================================
    // CompareOp tests
    // ============================================================

    #[test]
    fn test_compare_op_from_str() {
        assert_eq!(CompareOp::from_str("="), Some(CompareOp::Eq));
        assert_eq!(CompareOp::from_str("=="), Some(CompareOp::Eq));
        assert_eq!(CompareOp::from_str("eq"), Some(CompareOp::Eq));
        assert_eq!(CompareOp::from_str("!="), Some(CompareOp::Ne));
        assert_eq!(CompareOp::from_str("<>"), Some(CompareOp::Ne));
        assert_eq!(CompareOp::from_str("ne"), Some(CompareOp::Ne));
        assert_eq!(CompareOp::from_str("<"), Some(CompareOp::Lt));
        assert_eq!(CompareOp::from_str("lt"), Some(CompareOp::Lt));
        assert_eq!(CompareOp::from_str("<="), Some(CompareOp::Le));
        assert_eq!(CompareOp::from_str("le"), Some(CompareOp::Le));
        assert_eq!(CompareOp::from_str(">"), Some(CompareOp::Gt));
        assert_eq!(CompareOp::from_str("gt"), Some(CompareOp::Gt));
        assert_eq!(CompareOp::from_str(">="), Some(CompareOp::Ge));
        assert_eq!(CompareOp::from_str("ge"), Some(CompareOp::Ge));
        assert_eq!(CompareOp::from_str("invalid"), None);
        assert_eq!(CompareOp::from_str(""), None);
        assert_eq!(CompareOp::from_str("==="), None);
    }

    // ============================================================
    // SegmentStatus tests
    // ============================================================

    #[test]
    fn test_segment_status_all_variants() {
        assert_eq!(format!("{}", SegmentStatus::Active), "Active");
        assert_eq!(format!("{}", SegmentStatus::Compactable), "Compactable");
        assert_eq!(format!("{}", SegmentStatus::Frozen), "Frozen");
    }

    #[test]
    fn test_segment_status_from_str() {
        use std::str::FromStr;
        assert_eq!(SegmentStatus::from_str("Active"), Ok(SegmentStatus::Active));
        assert_eq!(SegmentStatus::from_str("Compactable"), Ok(SegmentStatus::Compactable));
        assert_eq!(SegmentStatus::from_str("Frozen"), Ok(SegmentStatus::Frozen));
        assert!(SegmentStatus::from_str("InvalidStatus").is_err());
    }

    // ============================================================
    // GranuleMeta::with_offset tests
    // ============================================================

    #[test]
    fn test_granule_meta_with_offset() {
        let g = GranuleMeta::with_offset(2, 100, 50, 0x1000);
        assert_eq!(g.granule_id, 2);
        assert_eq!(g.row_offset, 100);
        assert_eq!(g.row_count, 50);
        assert_eq!(g.file_offset, 0x1000);
        assert_eq!(g.zone_map.stats.len(), 0);
    }

    #[test]
    fn test_granule_meta_new_and_with_offset_equivalence() {
        // with_offset(file_offset=0) must be identical to new()
        let g_new = GranuleMeta::new(1, 10, 20);
        let g_offset = GranuleMeta::with_offset(1, 10, 20, 0);
        assert_eq!(g_new.granule_id, g_offset.granule_id);
        assert_eq!(g_new.row_offset, g_offset.row_offset);
        assert_eq!(g_new.row_count, g_offset.row_count);
        assert_eq!(g_new.file_offset, g_offset.file_offset);
        assert_eq!(g_new.zone_map.stats.len(), g_offset.zone_map.stats.len());
    }

    #[test]
    fn test_granule_meta_with_offset_preserves_offset() {
        // file_offset must not be overwritten or truncated
        let offsets = [0u64, 1, 1024, 1_000_000, u64::MAX];
        for (i, &offset) in offsets.iter().enumerate() {
            let g = GranuleMeta::with_offset(i as u32, 0, 1, offset);
            assert_eq!(g.file_offset, offset, "offset {} not preserved", offset);
        }
    }

    // ============================================================
    // EncodingType Debug and Display
    // ============================================================

    #[test]
    fn test_encoding_type_debug() {
        for enc in [
            EncodingType::Raw, EncodingType::Delta, EncodingType::Rle,
            EncodingType::Dict, EncodingType::Alp, EncodingType::Gorilla,
            EncodingType::Bitpacked, EncodingType::Zstd, EncodingType::Lz4,
        ] {
            let debug_str = format!("{:?}", enc);
            assert!(!debug_str.is_empty());
        }
    }

    // ============================================================
    // BlockStats tests (per-block predicate)
    // ============================================================

    #[test]
    fn test_block_stats_new() {
        let bs = BlockStats::new(3, 1024, 500);
        assert_eq!(bs.block_id, 3);
        assert_eq!(bs.row_offset, 1024);
        assert_eq!(bs.row_count, 500);
        assert!(bs.column_stats.is_empty());
    }

    #[test]
    fn test_block_stats_add_column() {
        let mut bs = BlockStats::new(0, 0, 1024);
        bs.add_column("age", 10i32.to_le_bytes().to_vec(), 50i32.to_le_bytes().to_vec());
        assert!(bs.column_stats.contains_key("age"));
        let (min, max) = bs.column_stats.get("age").unwrap();
        assert_eq!(min.as_slice(), 10i32.to_le_bytes().as_slice());
        assert_eq!(max.as_slice(), 50i32.to_le_bytes().as_slice());
    }

    #[test]
    fn test_block_stats_may_contain_eq_true() {
        let mut bs = BlockStats::new(0, 0, 1024);
        bs.add_column("col", 10i32.to_le_bytes().to_vec(), 50i32.to_le_bytes().to_vec());

        // col = 30 is within [10, 50], so block may contain it
        let may = bs.may_contain("col", &CompareOp::Eq, &30i32.to_le_bytes());
        assert!(may, "block min=10, max=50, col=30 should be possibly present");
    }

    #[test]
    fn test_block_stats_may_contain_eq_false() {
        let mut bs = BlockStats::new(0, 0, 1024);
        bs.add_column("col", 10i32.to_le_bytes().to_vec(), 50i32.to_le_bytes().to_vec());

        // col = 5 is outside [10, 50], but Eq always returns true (conservative)
        // Note: per implementation, Eq always returns true (can't prune on equality)
        let may = bs.may_contain("col", &CompareOp::Eq, &5i32.to_le_bytes());
        assert!(may, "Eq is conservative, always returns true");
    }

    #[test]
    fn test_block_stats_may_contain_missing_column() {
        let bs = BlockStats::new(0, 0, 1024);
        // Query column that doesn't exist in block stats -> conservative (may contain)
        let may = bs.may_contain("nonexistent", &CompareOp::Eq, &100i32.to_le_bytes());
        assert!(may, "missing column should return true (conservative)");
    }

    #[test]
    fn test_block_stats_may_contain_gt_true() {
        let mut bs = BlockStats::new(0, 0, 1024);
        bs.add_column("score", 10i32.to_le_bytes().to_vec(), 50i32.to_le_bytes().to_vec());

        // score > 45: max is 50, and 50 > 45, so block may contain matching rows
        let may = bs.may_contain("score", &CompareOp::Gt, &45i32.to_le_bytes());
        assert!(may, "max=50 > 45, block may contain score > 45");
    }

    #[test]
    fn test_block_stats_may_contain_gt_false() {
        let mut bs = BlockStats::new(0, 0, 1024);
        bs.add_column("score", 10i32.to_le_bytes().to_vec(), 50i32.to_le_bytes().to_vec());

        // score > 60: max is 50, and 50 <= 60, so block definitely does NOT contain matching rows
        let may = bs.may_contain("score", &CompareOp::Gt, &60i32.to_le_bytes());
        assert!(!may, "max=50 <= 60, block cannot contain score > 60");
    }

    #[test]
    fn test_block_stats_may_contain_lt_true() {
        let mut bs = BlockStats::new(0, 0, 1024);
        bs.add_column("val", 20i32.to_le_bytes().to_vec(), 80i32.to_le_bytes().to_vec());

        // val < 15: min is 20, and 20 >= 15, so block definitely does NOT contain
        let may = bs.may_contain("val", &CompareOp::Lt, &15i32.to_le_bytes());
        assert!(!may, "min=20 >= 15, block cannot contain val < 15");
    }

    #[test]
    fn test_block_stats_may_contain_lt_false() {
        let mut bs = BlockStats::new(0, 0, 1024);
        bs.add_column("val", 20i32.to_le_bytes().to_vec(), 80i32.to_le_bytes().to_vec());

        // val < 90: min is 20, and 20 < 90, so block may contain matching rows
        let may = bs.may_contain("val", &CompareOp::Lt, &90i32.to_le_bytes());
        assert!(may, "min=20 < 90, block may contain val < 90");
    }

    #[test]
    fn test_block_stats_may_contain_ne_always_true() {
        let mut bs = BlockStats::new(0, 0, 1024);
        bs.add_column("col", 10i32.to_le_bytes().to_vec(), 50i32.to_le_bytes().to_vec());
        // Ne is always conservative
        assert!(bs.may_contain("col", &CompareOp::Ne, &30i32.to_le_bytes()));
        assert!(bs.may_contain("col", &CompareOp::Ne, &5i32.to_le_bytes()));
    }

    #[test]
    fn test_compute_block_stats_single_block() {
        use arrow_array::RecordBatch;
        use arrow_schema::Schema;

        let schema = Schema::new(vec![
            arrow_schema::Field::new("id", arrow::datatypes::DataType::Int32, false),
            arrow_schema::Field::new("val", arrow::datatypes::DataType::Int64, false),
        ]);

        // 500 rows, block_size=1024 -> should produce 1 block
        let ids = arrow_array::Int32Array::from_iter_values(0..500i32);
        let vals = arrow_array::Int64Array::from_iter_values((0..500i64).map(|i| i * 10));

        let batch = RecordBatch::try_new(
            std::sync::Arc::new(schema),
            vec![std::sync::Arc::new(ids), std::sync::Arc::new(vals)],
        ).unwrap();

        let blocks = compute_block_stats(&batch, 1024);

        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].block_id, 0);
        assert_eq!(blocks[0].row_count, 500);
        assert_eq!(blocks[0].row_offset, 0);
    }

    #[test]
    fn test_compute_block_stats_multiple_blocks() {
        use arrow_array::RecordBatch;
        use arrow_schema::Schema;

        let schema = Schema::new(vec![
            arrow_schema::Field::new("x", arrow::datatypes::DataType::Int32, false),
        ]);

        // 2048 rows, block_size=1024 -> should produce 2 blocks
        let values = arrow_array::Int32Array::from_iter_values(0..2048i32);
        let batch = RecordBatch::try_new(
            std::sync::Arc::new(schema),
            vec![std::sync::Arc::new(values)],
        ).unwrap();

        let blocks = compute_block_stats(&batch, 1024);

        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].block_id, 0);
        assert_eq!(blocks[0].row_count, 1024);
        assert_eq!(blocks[0].row_offset, 0);
        assert_eq!(blocks[1].block_id, 1);
        assert_eq!(blocks[1].row_count, 1024);
        assert_eq!(blocks[1].row_offset, 1024);
    }

    #[test]
    fn test_compute_block_stats_exact_boundary() {
        use arrow_array::RecordBatch;
        use arrow_schema::Schema;

        let schema = Schema::new(vec![
            arrow_schema::Field::new("x", arrow::datatypes::DataType::Int32, false),
        ]);

        // Exactly 1024 rows = 1 block (boundary case)
        let values = arrow_array::Int32Array::from_iter_values(0..1024i32);
        let batch = RecordBatch::try_new(
            std::sync::Arc::new(schema),
            vec![std::sync::Arc::new(values)],
        ).unwrap();

        let blocks = compute_block_stats(&batch, 1024);

        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].row_count, 1024);
    }

    #[test]
    fn test_compute_block_stats_last_block_smaller() {
        use arrow_array::RecordBatch;
        use arrow_schema::Schema;

        let schema = Schema::new(vec![
            arrow_schema::Field::new("x", arrow::datatypes::DataType::Int32, false),
        ]);

        // 1500 rows, block_size=1024 -> 2 blocks: 1024 + 476
        let values = arrow_array::Int32Array::from_iter_values(0..1500i32);
        let batch = RecordBatch::try_new(
            std::sync::Arc::new(schema),
            vec![std::sync::Arc::new(values)],
        ).unwrap();

        let blocks = compute_block_stats(&batch, 1024);

        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].row_count, 1024);
        assert_eq!(blocks[1].row_count, 476);
        assert_eq!(blocks[0].row_offset + blocks[0].row_count as u64, blocks[1].row_offset);
    }

    #[test]
    fn test_compute_block_stats_includes_column_stats() {
        use arrow_array::RecordBatch;
        use arrow_schema::Schema;

        let schema = Schema::new(vec![
            arrow_schema::Field::new("val", arrow::datatypes::DataType::Int64, false),
        ]);

        // Rows with known range: [100, 200]
        let values = arrow_array::Int64Array::from_iter_values((100..1100i64).map(|i| i * 10));
        let batch = RecordBatch::try_new(
            std::sync::Arc::new(schema),
            vec![std::sync::Arc::new(values)],
        ).unwrap();

        let blocks = compute_block_stats(&batch, 1024);

        assert_eq!(blocks.len(), 1);
        // Block should contain stats for the "val" column (numeric type)
        assert!(blocks[0].column_stats.contains_key("val"));
    }
}
