//! Segment metadata module
//!
//! Defines segment and granule metadata structures used throughout RockDuck.

use arrow_schema::TimeUnit;
use rkyv::{Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};
use serde::{Deserialize, Serialize};

/// Number of rows per granule. A granule is the atomic unit of compaction and caching.
/// Each Vortex column file batch corresponds to one granule.
pub const GRANULE_SIZE: u32 = 8192;

/// A granule ID — the index of a granule within a segment.
///
/// A granule groups GRANULE_SIZE rows together. The granule ID is `row_offset / GRANULE_SIZE`.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    Default,
    Serialize,
    Deserialize,
    Archive,
    RkyvSerialize,
    RkyvDeserialize,
)]
pub struct GranuleId(pub u32);

impl GranuleId {
    pub const fn new(id: u32) -> Self {
        Self(id)
    }

    pub const fn zero() -> Self {
        Self(0)
    }

    pub const fn get(self) -> u32 {
        self.0
    }

    pub fn to_le_bytes(self) -> [u8; 4] {
        self.0.to_le_bytes()
    }

    pub fn from_le_bytes(bytes: [u8; 4]) -> Self {
        Self(u32::from_le_bytes(bytes))
    }
}

impl From<u32> for GranuleId {
    fn from(id: u32) -> Self {
        Self(id)
    }
}

impl From<GranuleId> for u32 {
    fn from(id: GranuleId) -> Self {
        id.0
    }
}

/// Segment metadata (stored in RocksDB via rkyv, with bytecheck validation)
#[derive(Debug, Clone, Serialize, Deserialize, Archive, RkyvSerialize, RkyvDeserialize)]
pub struct SegmentMeta {
    /// Segment ID
    pub seg_id: String,
    /// Table ID
    pub table_id: String,
    /// Segment status
    pub status: SegmentStatus,
    /// Segment type
    pub seg_type: SegmentType,
    /// Column definitions
    pub columns: Vec<ColumnDef>,
    /// Min key (for range partitioning)
    pub min_key: Vec<u8>,
    /// Max key (for range partitioning)
    pub max_key: Vec<u8>,
    /// Row count
    pub row_count: u64,
    /// Alive row count (after compaction)
    pub alive_row_count: u64,
    /// Delete ratio
    pub del_ratio: f64,
    /// Size in bytes
    pub size_bytes: u64,
    /// Creation transaction ID
    pub created_txn: u64,
    /// Last update transaction ID
    pub updated_txn: u64,
    /// Last updated timestamp (unix epoch micros)
    pub updated_at: u64,
    /// File paths (stored as Strings for cross-platform rkyv compatibility)
    pub file_paths: Vec<String>,
    /// Granules
    pub granules: Vec<GranuleMeta>,
    /// Whether this segment has MVCC shadow columns (__created_txn, __deleted_txn).
    /// Old segments written before Shadow Column support do NOT have these columns.
    pub has_visibility_columns: bool,
    /// Delta file ID for this segment (references L2 delta file).
    /// None if no delta has been flushed for this segment yet.
    pub delta_file_id: Option<String>,
    /// Number of rows in the delta layer (L1 + L2 + L3).
    pub delta_row_count: u64,
    /// Size in bytes of the L1 memstore for this segment.
    pub delta_l1_bytes: u64,
}

impl SegmentMeta {
    /// Create a new segment with current timestamp
    pub fn new(seg_id: String, table_id: String, columns: Vec<ColumnDef>) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_micros() as u64)
            .unwrap_or(0);
        Self {
            seg_id,
            table_id,
            status: SegmentStatus::Active,
            seg_type: SegmentType::Delta,
            columns,
            min_key: Vec::new(),
            max_key: Vec::new(),
            row_count: 0,
            alive_row_count: 0,
            del_ratio: 0.0,
            size_bytes: 0,
            created_txn: 0,
            updated_txn: 0,
            updated_at: now,
            file_paths: Vec::new(),
            granules: Vec::new(),
            has_visibility_columns: true, // 新 segment 默认开启 Shadow Column
            delta_file_id: None,
            delta_row_count: 0,
            delta_l1_bytes: 0,
        }
    }

    /// Calculate alive row count from total row count and delete ratio
    pub fn alive_rows(&self) -> u64 {
        (self.row_count as f64 * (1.0 - self.del_ratio)) as u64
    }

    /// Recalculate delete statistics from granule metadata
    pub fn update_del_stats(&mut self) {
        let total_rows: u64 = self.granules.iter().map(|g| g.row_count).sum();
        let total_deleted: u64 = self.granules.iter().map(|g| g.del_count).sum();
        self.row_count = total_rows;
        self.del_ratio = if total_rows > 0 {
            total_deleted as f64 / total_rows as f64
        } else {
            0.0
        };
        self.alive_row_count = self.alive_rows();
    }
}

/// Segment status
#[derive(
    Debug,
    Clone,
    Copy,
    Serialize,
    Deserialize,
    Archive,
    RkyvSerialize,
    RkyvDeserialize,
    PartialEq,
    Eq,
    Default,
)]
pub enum SegmentStatus {
    /// Active/writable segment
    #[default]
    Active,
    /// Frozen/immutable segment
    Frozen,
    /// Being compacted
    Compacting,
    /// Ready for cleanup
    Garbage,
}

/// Segment type
#[derive(
    Debug,
    Clone,
    Copy,
    Serialize,
    Deserialize,
    Archive,
    RkyvSerialize,
    RkyvDeserialize,
    PartialEq,
    Eq,
    Default,
)]
pub enum SegmentType {
    /// Delta store segment (transactional)
    #[default]
    Delta,
    /// Vortex columnar segment (analytical)
    Vortex,
    /// Frozen segment (compacted)
    Frozen,
}

/// Granule metadata (subset of segment)
#[derive(
    Debug, Clone, Serialize, Deserialize, Archive, RkyvSerialize, RkyvDeserialize, Default,
)]
pub struct GranuleMeta {
    /// Granule ID within segment
    pub granule_id: GranuleId,
    /// Min key
    pub min_key: Vec<u8>,
    /// Max key
    pub max_key: Vec<u8>,
    /// Row count
    pub row_count: u64,
    /// Delete count
    pub del_count: u64,
    /// Zone map statistics
    pub zone_map: Option<ZoneMapStats>,
    /// Whether this granule has block-level zone maps in sidecar files.
    /// When true, scan should load `.{col}._zm.bin` for block-level pruning.
    /// Defaults to false for backwards compatibility with pre-existing granules.
    pub has_block_zm: bool,
    /// Whether this granule's visibility data may be stale and needs validation during compaction.
    ///
    /// Set to `true` when new writes create a granule with potentially unvalidated vis data.
    /// Set to `false` after a successful compaction pass has validated and rewritten the granule.
    ///
    /// Defaults to `false` for backwards compatibility with pre-existing granules
    /// (old granules are assumed to have been validated by previous compaction runs).
    #[serde(default)]
    pub vis_dirty: bool,
}

/// Zone map statistics for columns
#[derive(Debug, Clone, Serialize, Deserialize, Archive, RkyvSerialize, RkyvDeserialize)]
pub struct ZoneMapStats {
    /// Column statistics
    pub columns: Vec<ColumnStats>,
}

/// Statistics for a single column
#[derive(Debug, Clone, Serialize, Deserialize, Archive, RkyvSerialize, RkyvDeserialize)]
pub struct ColumnStats {
    /// Column name
    pub name: String,
    /// Column data type
    pub data_type: DataType,
    /// Min value (as bytes)
    pub min: Vec<u8>,
    /// Max value (as bytes)
    pub max: Vec<u8>,
    /// Null count
    pub null_count: u64,
    /// Number of distinct values
    pub distinct_count: u64,
}

/// Column data type
#[derive(
    Debug,
    Clone,
    Copy,
    Serialize,
    Deserialize,
    Archive,
    RkyvSerialize,
    RkyvDeserialize,
    PartialEq,
    Eq,
)]
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
    TimestampMicros,
    TimestampMillis,
}

impl DataType {
    /// Convert from Arrow DataType
    pub fn from_arrow(dt: &arrow_schema::DataType) -> Self {
        use arrow_schema::DataType::*;
        match dt {
            Int8 => DataType::Int8,
            Int16 => DataType::Int16,
            Int32 => DataType::Int32,
            Int64 => DataType::Int64,
            UInt8 => DataType::UInt8,
            UInt16 => DataType::UInt16,
            UInt32 => DataType::UInt32,
            UInt64 => DataType::UInt64,
            Float32 => DataType::Float32,
            Float64 => DataType::Float64,
            Boolean => DataType::Bool,
            Utf8 => DataType::Utf8,
            LargeUtf8 => DataType::LargeUtf8,
            Binary => DataType::Binary,
            LargeBinary => DataType::LargeBinary,
            Date32 => DataType::Date32,
            Date64 => DataType::Date64,
            Timestamp(TimeUnit::Microsecond, _) => DataType::TimestampMicros,
            Timestamp(TimeUnit::Millisecond, _) => DataType::TimestampMillis,
            _ => DataType::Binary,
        }
    }

    /// Convert to Arrow DataType
    pub fn to_arrow(&self) -> arrow_schema::DataType {
        use arrow_schema::DataType as A;
        use arrow_schema::TimeUnit;
        match self {
            DataType::Int8 => A::Int8,
            DataType::Int16 => A::Int16,
            DataType::Int32 => A::Int32,
            DataType::Int64 => A::Int64,
            DataType::UInt8 => A::UInt8,
            DataType::UInt16 => A::UInt16,
            DataType::UInt32 => A::UInt32,
            DataType::UInt64 => A::UInt64,
            DataType::Float32 => A::Float32,
            DataType::Float64 => A::Float64,
            DataType::Bool => A::Boolean,
            DataType::Utf8 => A::Utf8,
            DataType::LargeUtf8 => A::LargeUtf8,
            DataType::Binary => A::Binary,
            DataType::LargeBinary => A::LargeBinary,
            DataType::Date32 => A::Date32,
            DataType::Date64 => A::Date64,
            DataType::TimestampMicros => A::Timestamp(TimeUnit::Microsecond, None),
            DataType::TimestampMillis => A::Timestamp(TimeUnit::Millisecond, None),
        }
    }
}

/// Column definition
#[derive(Debug, Clone, Serialize, Deserialize, Archive, RkyvSerialize, RkyvDeserialize)]
pub struct ColumnDef {
    /// Column name
    pub name: String,
    /// Data type
    pub data_type: DataType,
    /// Nullable
    pub nullable: bool,
    /// Default value expression
    pub default_expr: Option<String>,
}

impl ColumnDef {
    /// Create a new column definition
    pub fn new(name: String, data_type: DataType) -> Self {
        Self {
            name,
            data_type,
            nullable: true,
            default_expr: None,
        }
    }

    /// Create a non-nullable column
    pub fn not_null(mut self) -> Self {
        self.nullable = false;
        self
    }
}
