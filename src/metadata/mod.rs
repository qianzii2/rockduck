//! Metadata 模块
//!
//! RocksDB 元数据管理

pub mod rocksdb;
pub mod pk_index;
pub mod pk_skiplist;
pub mod seg_meta;
pub mod zone_map;
pub mod layer;
pub mod projection;
pub mod lbf;

pub use rocksdb::*;
pub use pk_index::*;
pub use seg_meta::*;
pub use zone_map::*;
// Note: CF_LAYER is re-exported from rocksdb, not layer, to avoid conflict
pub use rocksdb::CF_LAYER;

// Re-export SegmentMeta from segment module
pub use crate::segment::meta::SegmentMeta;

// 时间戳工具函数
pub use crate::codec::timestamp::{current_timestamp_secs, current_timestamp_millis};

/// 表统计信息
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, Default, bincode_next::Encode, bincode_next::Decode)]
pub struct TableStats {
    /// 表名
    pub table: String,
    /// 行数
    pub row_count: u64,
    /// Segment 数量
    pub segment_count: u32,
    /// 总大小（字节）
    pub total_size: u64,
    /// 压缩后大小（字节）
    pub compressed_size: u64,
    /// 删除行数
    pub deleted_rows: u64,
    /// 最后更新时间戳
    pub last_updated: u64,
}

impl TableStats {
    /// 创建新的 TableStats
    pub fn new(table: String) -> Self {
        let now = crate::codec::current_timestamp_secs();
            
        Self {
            table,
            row_count: 0,
            segment_count: 0,
            total_size: 0,
            compressed_size: 0,
            deleted_rows: 0,
            last_updated: now,
        }
    }

    /// 更新行数
    pub fn add_rows(&mut self, count: u64) {
        self.row_count += count;
        self.last_updated = crate::codec::current_timestamp_secs();
    }

    /// 更新删除行数
    pub fn add_deleted(&mut self, count: u64) {
        self.deleted_rows += count;
        self.last_updated = crate::codec::current_timestamp_secs();
    }

    /// 添加一个 segment
    pub fn add_segment(&mut self) {
        self.segment_count += 1;
        self.last_updated = crate::codec::current_timestamp_secs();
    }

    /// 获取有效行数
    pub fn alive_rows(&self) -> u64 {
        self.row_count.saturating_sub(self.deleted_rows)
    }

    /// 获取删除率
    pub fn del_ratio(&self) -> f64 {
        if self.row_count == 0 {
            return 0.0;
        }
        self.deleted_rows as f64 / self.row_count as f64
    }
}

/// Segment 索引条目（用于主键索引）
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, bincode_next::Encode, bincode_next::Decode)]
pub struct IndexEntry {
    /// Segment ID
    pub seg_id: String,
    /// Granule ID（在 segment 内）
    pub granule_id: u32,
    /// 在 granule 内的行偏移
    pub offset: u32,
    /// 事务 ID
    pub txn_id: u64,
    /// 时间戳
    pub timestamp: u64,
}

impl IndexEntry {
    /// 创建新的索引条目
    pub fn new(seg_id: String, granule_id: u32, offset: u32, txn_id: u64) -> Self {
        Self {
            seg_id,
            granule_id,
            offset,
            txn_id,
            timestamp: crate::codec::current_timestamp_millis(),
        }
    }
}
