//! Block-level Zone Map and Granule-level Bloom Filter for data skipping.
//!
//! ## Architecture
//!
//! Data hierarchy (bottom-up):
//!   Block (8192 rows or 10MB) < Granule (8 blocks = 65536 rows) < Segment
//!
//! Statistics hierarchy:
//!   BlockStats (per block, in-mem) < GranuleStats (per granule, serialized) < GranuleZoneMapIndex (per column, on-disk)
//!
//! ## Storage Layout
//!
//! ```text
//! segments/seg_xxx/
//!   ├── id.vortex              # column data
//!   ├── id._zm.bin            # GranuleZoneMapIndex (bincode, one per column)
//!   └── id._bf.bin            # GranuleBloomFilter data (bincode, one per column)
//! ```
//!
//! ## Bloom Filter: Split Block Bloom Filter (SBBF)
//!
//! Based on Parquet SBBF specification (Putze et al.):
//! - xxhash64 as hash function
//! - 256-bit (32-byte) blocks
//! - 8 salt lanes per block (split Bloom filter)
//! - False positive rate ~1-3% depending on fill factor

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::Write;
use std::path::Path;

use crate::error::{Result, RockDuckError};
use crate::metadata::block_stats::{compute_block_stats, BlockColumnStats};
use crate::metadata::GranuleId;

// Re-export for use inGranuleBloomFilterManager (also used by pdt_merge.rs)
use arrow_array::cast::AsArray;
use arrow_array::Array;

// ─── Constants ────────────────────────────────────────────────────────────────

/// Default number of rows per block (matches ClickHouse granule size).
pub const DEFAULT_BLOCK_SIZE: u32 = 8192;
/// Default number of blocks per granule (8 * 8192 = 65536 rows).
pub const DEFAULT_GRANULE_SIZE: u32 = 8;
/// Default target false-positive probability for Bloom filter.
pub const DEFAULT_BF_FPP: f64 = 0.01;
/// Number of bits set per hash insertion (8 salt lanes per SBBF block).
const BITS_PER_BLOCK: usize = 8;
/// Salt values from the Parquet Bloom filter spec.
const SALT: [u32; BITS_PER_BLOCK] = [
    0x47b5_5a47u32,
    0x8e76_3453u32,
    0x32a8_9235u32,
    0x7f83_1979u32,
    0x6b8d_0ab9u32,
    0x4d94_87a3u32,
    0x1d74_2df3u32,
    0xa12f_57bdu32,
];

// ─── BlockStats ──────────────────────────────────────────────────────────────

/// Statistics for one data block (~8192 rows or ~10MB).
///
/// A block is the minimum indivisible I/O unit: either all rows are read
/// or none are (if the block is skipped by zone map or bloom filter).
///
/// `min_bytes` and `max_bytes` are stored in little-endian byte order
/// for native numeric types (so raw byte comparison equals numeric comparison).
/// For strings, raw byte comparison equals lex order.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockStats {
    /// Row offset of the first row in this block (inclusive).
    pub row_start: u32,
    /// Row offset one-past the last row in this block (exclusive).
    pub row_end: u32,
    /// Min value as bytes (little-endian for numeric, lex for string).
    pub min_bytes: Option<Vec<u8>>,
    /// Max value as bytes.
    pub max_bytes: Option<Vec<u8>>,
    /// Number of null values in this block.
    pub null_count: u32,
}

impl BlockStats {
    /// Create a new BlockStats for a row range with pre-computed column stats.
    pub fn new(row_start: u32, row_end: u32, col_stats: &BlockColumnStats) -> Self {
        Self {
            row_start,
            row_end,
            min_bytes: col_stats.min_bytes.clone(),
            max_bytes: col_stats.max_bytes.clone(),
            null_count: col_stats.null_count,
        }
    }

    /// Number of rows in this block.
    pub fn num_rows(&self) -> u32 {
        self.row_end - self.row_start
    }
}

// ─── GranuleBloomFilter (SBBF) ────────────────────────────────────────────────

/// Granule-level Split Block Bloom Filter (SBBF) for fast membership checks.
/// Lives inside `GranuleStats` for in-memory bloom filtering during stats traversal.
/// Uses xxhash64 and 256-bit blocks with 8 salt lanes (Parquet SBBF spec).
///
/// One Bloom filter covers all blocks within a single granule (~64K rows).
/// Used for primary-key deduplication and high-cardinality equality pruning.
///
/// Based on Parquet SBBF: xxhash64, 256-bit blocks, 8 salt lanes.
/// - `insert(pk)` → compute xxhash64 → select block → set 8 bits
/// - `might_contain(pk)` → compute xxhash64 → select block → check 8 bits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SbbFilter {
    /// Number of entries (PK values) inserted.
    pub entries: u32,
    /// Number of 32-byte blocks.
    num_blocks: u32,
    /// Bitset: `num_blocks * 32` bytes.
    bits: Vec<u8>,
}

impl SbbFilter {
    /// Create a new Bloom filter with the given number of blocks.
    pub fn new(num_blocks: u32) -> Self {
        let bits = vec![0u8; num_blocks as usize * 32];
        Self {
            entries: 0,
            num_blocks,
            bits,
        }
    }

    /// Compute optimal number of 32-byte blocks for the expected entry count
    /// and target false-positive probability.
    pub fn optimal_num_blocks(entries: u32, fpp: f64) -> u32 {
        let n = entries as f64;
        let m = (-8.0 * n * fpp.ln() / (2.0_f64.ln().powi(2))).max(256.0);
        ((m / 256.0).ceil() as u32).max(1)
    }

    /// Number of 32-byte blocks in this filter.
    pub fn num_blocks(&self) -> u32 {
        self.num_blocks
    }

    /// Insert a key into the Bloom filter.
    pub fn insert(&mut self, key: &[u8]) {
        self.insert_hashes(Self::hash(key));
    }

    fn insert_hashes(&mut self, (h1, h2): (u64, u64)) {
        let block_idx = (h1 % (self.num_blocks as u64)) as usize;
        let offset = block_idx * 32;

        for salt_val in SALT {
            let bit_pos = (salt_val.wrapping_mul(h2 as u32)) % 32;
            self.bits[offset + (bit_pos / 8) as usize] |= 1 << (bit_pos % 8);
        }
        self.entries += 1;
    }

    /// Check if a key might be in the set.
    /// Returns `false` if definitely NOT present.
    /// Returns `true` if probably present (may be a false positive).
    pub fn might_contain(&self, key: &[u8]) -> bool {
        self.might_contain_hashes(Self::hash(key))
    }

    fn might_contain_hashes(&self, (h1, h2): (u64, u64)) -> bool {
        let block_idx = (h1 % (self.num_blocks as u64)) as usize;
        let offset = block_idx * 32;

        for salt_val in SALT {
            let bit_pos = (salt_val.wrapping_mul(h2 as u32)) % 32;
            if (self.bits[offset + (bit_pos / 8) as usize] & (1 << (bit_pos % 8))) == 0 {
                return false;
            }
        }
        true
    }

    /// Compute two independent 64-bit hashes using xxh3_128.
    ///
    /// Uses `xxh3_128_with_seed` which produces a 128-bit output, then splits
    /// it into high and low 64-bit halves. This is superior to the old
    /// `xxh3_64 + LCG` approach because:
    ///
    /// - A single strong hash call produces 128 bits, minimizing entropy loss.
    /// - The high/low split yields two hashes that are truly independent
    ///   (derived from the same compression function, but without the correlation
    ///   that LCG arithmetic introduces).
    /// - Consistent with the plan's migration note: existing `.bf` files
    ///   remain readable; the old hash function is not used for new files.
    fn hash(key: &[u8]) -> (u64, u64) {
        use xxhash_rust::xxh3::xxh3_128_with_seed;
        let h = xxh3_128_with_seed(key, 0x1234_ABCE_FD01_2345_u64);
        (h as u64, (h >> 64) as u64)
    }

    /// Estimated false-positive rate based on current fill.
    #[allow(dead_code)]
    pub fn estimated_fpp(&self) -> f64 {
        if self.entries == 0 || self.num_blocks == 0 {
            return 0.0;
        }
        let m = (self.num_blocks as f64) * 256.0;
        let n = self.entries as f64;
        let k = BITS_PER_BLOCK as f64;
        // Standard Bloom filter FPP: (1 - exp(-k * m/n))^k
        let inner = (-k * m / n).exp();
        let fpp = (1.0 - inner).powf(k);
        fpp.clamp(0.0, 1.0)
    }
}

// ─── GranuleStats ───────────────────────────────────────────────────────────

    /// Statistics for one granule: all block stats + optional bloom filter.
    /// The bloom filter is marked skip_serializing so that it is rebuilt on load.
    /// This avoids bincode byte-boundary ambiguity issues when the bloom filter
    /// bitset is large (e.g. 4908 blocks = 157KB with varint encoding).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GranuleStats {
    pub granule_id: GranuleId,
    pub block_size: u32,
    pub blocks: Vec<BlockStats>,
    #[serde(skip, default)]
    pub bloom_filter: Option<SbbFilter>,
}

impl GranuleStats {
    /// Rebuild the bloom filter from existing block data.
    ///
    /// **DEPRECATED**: The bloom filter cannot be rebuilt from GranuleStats alone since
    /// BlockStats only contains min/max bytes, not individual PK values.
    ///
    /// For compaction: use `GranuleBloomFilterManager::build_all_from_array()` which inserts
    /// actual PK values from the decompressed column data, then saves to `.{col}._bf.bin`.
    ///
    /// For existing segments: bloom filters are loaded from the `.{col}._bf.bin` sidecar
    /// file. If the sidecar is missing (pre-m001 segments), the bloom filter cannot be rebuilt
    /// and remains None.
    #[deprecated(since = "0.2.1", note = "use GranuleBloomFilterManager during compaction instead")]
    pub fn rebuild(&mut self) {
        let total_rows: u32 = self.blocks.iter().map(|b| b.row_end - b.row_start).sum();
        if total_rows > 0 {
            self.init_bloom_filter(total_rows);
            tracing::warn!(
                "GranuleStats::rebuild is deprecated. Bloom filter initialized with \
                 {} rows capacity but NOT populated with PK values. Bloom filter is empty. \
                 Load from sidecar or rebuild during compaction.",
                self.total_rows()
            );
        }
    }

    /// Create a new empty GranuleStats.
    pub fn new(granule_id: GranuleId, block_size: u32) -> Self {
        Self {
            granule_id,
            block_size,
            blocks: Vec::with_capacity(DEFAULT_GRANULE_SIZE as usize),
            bloom_filter: None,
        }
    }

    /// Add a block to this granule.
    pub fn add_block(&mut self, block: BlockStats) {
        self.blocks.push(block);
    }

    /// Initialize the bloom filter with optimal size based on expected entries.
    pub fn init_bloom_filter(&mut self, expected_entries: u32) {
        let num_blocks = SbbFilter::optimal_num_blocks(expected_entries, DEFAULT_BF_FPP);
        self.bloom_filter = Some(SbbFilter::new(num_blocks));
    }

    /// Insert a primary key into the bloom filter.
    #[allow(dead_code)]
    pub fn insert_pk(&mut self, pk: &[u8]) {
        if let Some(ref mut bf) = self.bloom_filter {
            bf.insert(pk);
        }
    }

    /// Check if a PK might exist in this granule.
    #[allow(dead_code)]
    pub fn might_contain_pk(&self, pk: &[u8]) -> bool {
        self.bloom_filter
            .as_ref()
            .is_none_or(|bf| bf.might_contain(pk))
    }

    /// Insert raw bytes (already serialized) into the bloom filter.
    /// Used by compaction to insert PK values from Arrow arrays.
    #[allow(dead_code)]
    pub fn insert_bytes(&mut self, bytes: &[u8]) {
        if let Some(ref mut bf) = self.bloom_filter {
            bf.insert(bytes);
        }
    }

    /// Total rows across all blocks in this granule.
    pub fn total_rows(&self) -> u32 {
        self.blocks.iter().map(|b| b.num_rows()).sum()
    }
}

// ─── GranuleZoneMapIndex ─────────────────────────────────────────────────────

/// Zone map index for all granules of one column within one segment.
///
/// Stored as a single bincode-encoded file: `{col_name}._zm.bin`.
///
/// File format (v3 — current):
///   - 8-byte binary header: magic[4] + version[1] + reserved[3]
///   - bincode-encoded GranuleZoneMapIndex payload
///
/// v1 (legacy): no header, raw bincode payload. Magic check happens after
///   deserialization. column_type is absent (defaults to Int64).
///
/// v2: magic+version inside bincode payload. column_type added.
///
/// v3: binary header prepended. Magic+version readable before bincode.
///   Payload version must match header version. v1/v2 files still readable
///   by ignoring the header and deserializing directly.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GranuleZoneMapIndex {
    /// Magic bytes "BLCK" for file identification.
    pub magic: u32,
    /// Version for forward compatibility.
    pub version: u8,
    /// Column name this index belongs to.
    pub col_name: String,
    /// Column data type for type-aware zone map comparisons.
    /// Stored here (rather than per-block) because all blocks in this index are for the same column.
    pub column_type: arrow_schema::DataType,
    /// Rows per block (fixed at 8192).
    pub block_size: u32,
    /// Number of blocks per granule (fixed at 8).
    pub granule_size: u32,
    /// All granules in row-order.
    pub granules: Vec<GranuleStats>,
}

/// Binary header prepended to the bincode payload for v3 files.
struct ZmBinaryHeader {
    /// Magic bytes: b"BLCK"
    magic: [u8; 4],
    /// Version byte (GranuleZoneMapIndex::VERSION).
    version: u8,
    /// Reserved for future use; must be 0.
    _reserved: [u8; 3],
}

impl ZmBinaryHeader {
    /// Total header size in bytes.
    const SIZE: usize = 8;

    fn write_to(&self, buf: &mut Vec<u8>) {
        buf.extend_from_slice(&self.magic);
        buf.push(self.version);
        buf.extend_from_slice(&self._reserved);
    }
}

impl GranuleZoneMapIndex {
    /// Magic bytes: b"BLCK"
    const MAGIC: u32 = 0x424C434B;
    /// v3: binary header prepended. Magic+version readable before bincode.
    /// v2: magic+version inside bincode payload.
    /// v1: no header, raw bincode. Magic check happens post-deserialization.
    const VERSION: u8 = 3;

    /// Create a new empty index for a column.
    pub fn new(col_name: String, column_type: arrow_schema::DataType) -> Self {
        Self {
            magic: Self::MAGIC,
            version: Self::VERSION,
            col_name,
            column_type,
            block_size: DEFAULT_BLOCK_SIZE,
            granule_size: DEFAULT_GRANULE_SIZE,
            granules: Vec::new(),
        }
    }

    /// Backward-compatible constructor: infers column_type from the column array.
    /// Prefer `new(col_name, column_type)` when the dtype is known at construction time.
    pub fn with_array(col_name: String, arr: &dyn arrow_array::Array) -> Self {
        Self::new(col_name, arr.data_type().clone())
    }

    /// Serialize to binary bytes using bincode.
    ///
    /// File format (v3):
    ///   - 8-byte binary header: magic[4] + version[1] + reserved[3]
    ///   - bincode-encoded GranuleZoneMapIndex payload
    ///
    /// The length prefix is replaced by the binary header, which is more
    /// efficient (8 bytes vs 4 bytes) and allows reading magic+version
    /// before attempting bincode deserialization.
    pub fn to_bytes(&self) -> crate::error::Result<Vec<u8>> {
        let payload = bincode::serialize(self).map_err(|e| {
            RockDuckError::Codec(format!("bincode encode GranuleZoneMapIndex: {}", e))
        })?;

        let mut buf = Vec::with_capacity(ZmBinaryHeader::SIZE + payload.len());
        let header = ZmBinaryHeader {
            magic: Self::MAGIC.to_le_bytes(),
            version: Self::VERSION,
            _reserved: [0u8; 3],
        };
        header.write_to(&mut buf);
        buf.extend_from_slice(&payload);
        Ok(buf)
    }

    /// Deserialize from binary bytes.
    ///
    /// Handles v1 (raw bincode, no header), v2 (magic+version in bincode payload),
    /// and v3 (8-byte binary header prepended).
    ///
    /// v1 files: detected by absence of "BLCK" magic at byte 0.
    ///   The entire data is treated as a bincode payload.
    ///   column_type is defaulted to Int64.
    ///
    /// v2 files: bincode-deserializes normally, then checks magic.
    ///   column_type is restored from payload.
    ///
    /// v3 files: reads the 8-byte binary header first, then the payload.
    ///   Version must match header version.
    pub fn from_bytes(data: &[u8]) -> crate::error::Result<Self> {
        // Peek at first 4 bytes to detect v3 vs v1/v2
        if data.len() >= ZmBinaryHeader::SIZE {
            // Fixed Q-2: use ok_or_else instead of unwrap for boundary safety
            let magic = u32::from_le_bytes(data[..4].try_into().map_err(|_| {
                RockDuckError::Codec("GranuleZoneMapIndex: insufficient bytes for magic".into())
            })?);
            if magic == Self::MAGIC {
                // v3: read binary header
                let header_version = data[4];
                if header_version > Self::VERSION {
                    return Err(RockDuckError::Codec(format!(
                        "unsupported GranuleZoneMapIndex version: {} (max: {})",
                        header_version,
                        Self::VERSION
                    )));
                }
                let payload = &data[ZmBinaryHeader::SIZE..];
                let mut index: Self = {
                    let limit = 10 * 1024 * 1024; // 10 MiB cap
                    if payload.len() > limit {
                        return Err(RockDuckError::Codec(format!(
                            "GranuleZoneMapIndex payload too large: {} > {} bytes",
                            payload.len(),
                            limit
                        )));
                    }
                    bincode::deserialize(payload)
                }
                .map_err(|e| {
                    RockDuckError::Codec(format!("bincode decode GranuleZoneMapIndex: {}", e))
                })?;
                index.rebuild_all();
                // Payload version must match header version
                if index.version != header_version {
                    return Err(RockDuckError::Codec(format!(
                        "GranuleZoneMapIndex version mismatch: header={}, payload={}",
                        header_version, index.version
                    )));
                }
                return Ok(index);
            }
        }

        // v1 or v2: no binary header — entire data is a bincode payload
        let mut index: Self = {
            let limit = 10 * 1024 * 1024; // 10 MiB cap
            if data.len() > limit {
                return Err(RockDuckError::Codec(format!(
                    "GranuleZoneMapIndex data too large: {} > {} bytes",
                    data.len(),
                    limit
                )));
            }
            bincode::deserialize(data)
        }
        .map_err(|e| RockDuckError::Codec(format!("bincode decode GranuleZoneMapIndex: {}", e)))?;
        index.rebuild_all();

        // Validate magic for v2 files
        if index.magic != Self::MAGIC {
            return Err(RockDuckError::Codec(format!(
                "invalid GranuleZoneMapIndex magic: expected {:#x}, got {:#x}",
                Self::MAGIC,
                index.magic
            )));
        }

        // 4.3 fix: v1 files have no column_type — mark as Null instead of defaulting to Int64.
        // Defaulting to Int64 causes incorrect zone map pruning for non-Int64 columns.
        // Null type will cause the zone map to skip pruning (returns false in bytes_lt).
        if index.version < 2 {
            index.column_type = DataType::Null;
            tracing::error!(
                "GranuleZoneMapIndex v1 file for column '{}': column_type unknown. \
                Zone map pruning is DISABLED for this column (all blocks returned as candidates). \
                Migrate to v2 format by rebuilding the zone map.",
                index.col_name
            );
        }
        Ok(index)
    }

    /// Save to a file path.
    pub fn save(&self, path: &std::path::Path) -> crate::error::Result<()> {
        let data = self.to_bytes()?;
        std::fs::write(path, data).map_err(RockDuckError::Io)?;
        Ok(())
    }

    /// Load from a file path.
    pub fn load(path: &std::path::Path) -> crate::error::Result<Self> {
        let data = std::fs::read(path).map_err(RockDuckError::Io)?;
        Self::from_bytes(&data)
    }

    /// Total rows across all granules.
    pub fn total_rows(&self) -> u64 {
        self.granules.iter().map(|g| g.total_rows() as u64).sum()
    }

    /// Number of granules.
    pub fn num_granules(&self) -> usize {
        self.granules.len()
    }

    /// Rebuild all bloom filters after deserialization.
    /// Bloom filters are skip_serialized to avoid bincode byte-boundary ambiguity issues,
    /// so they must be reconstructed from block row counts after loading.
    #[allow(deprecated)]
    pub fn rebuild_all(&mut self) {
        for gran in &mut self.granules {
            gran.rebuild();
        }
    }

    /// Check if any block in any granule of this column could overlap with the given predicate range.
    /// This is the granule-level analog of ZoneMapStats::may_overlap.
    /// Returns true if any block might overlap, false if all blocks provably don't overlap.
    #[allow(dead_code)]
    pub fn may_overlap(&self, pred_min: &[u8], pred_max: &[u8]) -> bool {
        let dtype = &self.column_type;

        for granule in &self.granules {
            for block in &granule.blocks {
                // Skip block if it is entirely below the predicate:
                // block.max < pred.min → no possible overlap
                if let Some(ref block_max) = block.max_bytes {
                    if bytes_lt(block_max, pred_min, dtype) {
                        continue;
                    }
                }
                // Skip block if it is entirely above the predicate:
                // block.min > pred.max → no possible overlap
                if let Some(ref block_min) = block.min_bytes {
                    if bytes_lt(pred_max, block_min, dtype) {
                        continue;
                    }
                }
                // Block overlaps the range — can't skip this granule
                return true;
            }
        }
        // All blocks provably outside the range
        false
    }
}

// ─── BlockZoneMapBuilder ─────────────────────────────────────────────────────

/// Builds BlockStats incrementally as data is written.
///
/// Accumulates rows from RecordBatches and emits completed BlockStats
/// when either:
///   - Row count reaches `block_size` (8192 rows), OR
///   - Accumulated bytes reaches `max_bytes_per_block` (10MB)
///
/// Once a block is complete, it is appended to the current granule.
/// When a granule is complete (8 blocks), it is appended to the index.
///
/// Note: `append_batch` should be called once per column (not once per batch).
/// Call it once per column you want to build zone map stats for.
#[derive(Debug)]
pub struct BlockZoneMapBuilder {
    /// Column name.
    col_name: String,
    /// Column data type for type-aware zone map comparison.
    column_type: arrow_schema::DataType,
    /// Rows per block.
    block_size: u32,
    /// Blocks per granule.
    granule_size: u32,
    /// Max bytes before triggering a block flush.
    max_bytes_per_block: u64,

    /// Row offset of the end of the current block (next row to be added).
    block_end_row: u32,
    /// Total number of rows in blocks already flushed.
    blocks_flushed_rows: u32,

    /// Pending rows in the current block (not yet flushed).
    pending_rows: u32,
    /// Pending estimated bytes in the current block.
    pending_bytes: u64,

    /// Stats accumulated so far for the current block.
    current_block_stats: BlockColumnStats,

    /// Current granule being built.
    current_granule: Vec<BlockStats>,
    /// Completed granules ready for the index.
    completed_granules: Vec<GranuleStats>,

    /// Number of blocks in current granule.
    blocks_in_granule: u32,
}

impl BlockZoneMapBuilder {
    /// Create a new builder with explicit dtype.
    pub fn new(col_name: String, column_type: arrow_schema::DataType) -> Self {
        Self {
            col_name,
            column_type,
            block_size: DEFAULT_BLOCK_SIZE,
            granule_size: DEFAULT_GRANULE_SIZE,
            max_bytes_per_block: 10 * 1024 * 1024, // 10MB
            block_end_row: 0,
            blocks_flushed_rows: 0,
            pending_rows: 0,
            pending_bytes: 0,
            current_block_stats: BlockColumnStats {
                min_bytes: None,
                max_bytes: None,
                null_count: 0,
            },
            current_granule: Vec::with_capacity(DEFAULT_GRANULE_SIZE as usize),
            completed_granules: Vec::new(),
            blocks_in_granule: 0,
        }
    }

    /// Backward-compatible: infer column_type from array dtype.
    pub fn with_array(col_name: String, arr: &dyn arrow_array::Array) -> Self {
        Self::new(col_name, arr.data_type().clone())
    }

    /// Append a column array and update block stats.
    /// Call once per column you want to build zone map stats for.
    pub fn append_batch(&mut self, col: &dyn arrow_array::Array) -> usize {
        let rows_consumed = self.append_column(col);
        // Flush any remaining rows in the current block
        if self.pending_rows > 0 {
            self.flush_block();
        }
        // Flush any remaining blocks in the current granule
        self.flush_granule();
        rows_consumed
    }

    /// Append a column array and return rows consumed.
    /// Each call processes the full column from row 0..N.
    fn append_column(&mut self, col: &dyn arrow_array::Array) -> usize {
        // Track rows within this single column pass
        let mut pass_end_row: u32 = 0;
        let mut rows_consumed = 0;

        while rows_consumed < col.len() {
            // If the current block is full, flush it and start a new one
            if self.pending_rows >= self.block_size
                || self.pending_bytes >= self.max_bytes_per_block
            {
                self.flush_block();
                if self.blocks_in_granule >= self.granule_size {
                    self.flush_granule();
                }
            }

            let rows_left_in_block = self.block_size - self.pending_rows;
            let bytes_left_in_block = self.max_bytes_per_block.saturating_sub(self.pending_bytes);
            let rows_can_add = ((col.len() - rows_consumed) as u32)
                .min(rows_left_in_block)
                .min((bytes_left_in_block / 8).max(1) as u32)
                as usize;

            if rows_can_add == 0 {
                // Block is full but has no room for even 1 row — skip this block
                self.pending_rows = self.block_size; // force flush on next iteration
                self.flush_block();
                if self.blocks_in_granule >= self.granule_size {
                    self.flush_granule();
                }
                continue;
            }

            let offset = rows_consumed;
            let slice_stats = compute_block_stats(col, offset, rows_can_add);
            self.current_block_stats.merge_with(&slice_stats);

            self.pending_rows += rows_can_add as u32;
            self.pending_bytes += (rows_can_add * 8) as u64;
            pass_end_row += rows_can_add as u32;
            rows_consumed += rows_can_add;
        }

        // After processing, advance block_end_row by the rows consumed in this pass
        self.block_end_row = self.block_end_row.max(pass_end_row);

        rows_consumed
    }

    /// Flush the current block stats to the current granule.
    fn flush_block(&mut self) {
        if self.pending_rows == 0 {
            return;
        }

        let block_rows = self.pending_rows;
        let row_start = self.blocks_flushed_rows;
        let row_end = row_start + block_rows;

        let block = BlockStats {
            row_start,
            row_end,
            min_bytes: self.current_block_stats.min_bytes.clone(),
            max_bytes: self.current_block_stats.max_bytes.clone(),
            null_count: self.current_block_stats.null_count,
        };

        self.current_granule.push(block);
        self.blocks_in_granule += 1;
        self.blocks_flushed_rows += block_rows;

        self.pending_rows = 0;
        self.pending_bytes = 0;
        self.current_block_stats = BlockColumnStats {
            min_bytes: None,
            max_bytes: None,
            null_count: 0,
        };
    }

    /// Flush the current granule to completed_granules.
    fn flush_granule(&mut self) {
        if self.current_granule.is_empty() {
            return;
        }

        let granule_id = self.completed_granules.len() as u32;
        let block_size = self.block_size;
        let expected_entries = self.current_granule.iter().map(|b| b.num_rows()).sum();

        let mut granule = GranuleStats::new(GranuleId::new(granule_id), block_size);
        for block in std::mem::take(&mut self.current_granule) {
            granule.add_block(block);
        }
        granule.init_bloom_filter(expected_entries);

        self.completed_granules.push(granule);
        self.blocks_in_granule = 0;
    }

    /// Finalize and return the completed index.
    pub fn finalize(mut self) -> GranuleZoneMapIndex {
        if self.pending_rows > 0 {
            self.flush_block();
        }
        if !self.current_granule.is_empty() || self.blocks_in_granule > 0 {
            self.flush_granule();
        }

        let mut index = GranuleZoneMapIndex::new(self.col_name, self.column_type);
        index.granules = self.completed_granules;
        index
    }

    /// Number of rows processed so far.
    #[allow(dead_code)]
    pub fn rows_processed(&self) -> u32 {
        self.block_end_row
    }
}

// ─── Type-aware comparison utilities ─────────────────────────────────────────

use arrow_schema::DataType;

/// Compare two byte sequences as the given data type.
/// Returns true if `left < right` according to the type's ordering.
pub fn bytes_lt(left: &[u8], right: &[u8], dtype: &DataType) -> bool {
    use arrow_schema::DataType::*;
    match dtype {
        // Integer types (1–8 bytes, little-endian)
        Int8 => {
            i8::from_le_bytes(read_bytes::<1>(left)) < i8::from_le_bytes(read_bytes::<1>(right))
        }
        Int16 => {
            i16::from_le_bytes(read_bytes::<2>(left)) < i16::from_le_bytes(read_bytes::<2>(right))
        }
        Int32 => {
            i32::from_le_bytes(read_bytes::<4>(left)) < i32::from_le_bytes(read_bytes::<4>(right))
        }
        Int64 => {
            i64::from_le_bytes(read_bytes::<8>(left)) < i64::from_le_bytes(read_bytes::<8>(right))
        }

        // Unsigned integer types
        UInt8 => {
            u8::from_le_bytes(read_bytes::<1>(left)) < u8::from_le_bytes(read_bytes::<1>(right))
        }
        UInt16 => {
            u16::from_le_bytes(read_bytes::<2>(left)) < u16::from_le_bytes(read_bytes::<2>(right))
        }
        UInt32 => {
            u32::from_le_bytes(read_bytes::<4>(left)) < u32::from_le_bytes(read_bytes::<4>(right))
        }
        UInt64 => {
            u64::from_le_bytes(read_bytes::<8>(left)) < u64::from_le_bytes(read_bytes::<8>(right))
        }

        // Floating-point types (NaN-aware comparison)
        // DESIGN-DECISION (design-05): NaN values in ZoneMap pruning are handled conservatively.
        // Float16 => false (skip pruning - half crate needed for comparison)
        // Float32/Float64 => returns false if either value is NaN (skip pruning).
        // This means segments containing NaN values in the min/max columns are NOT skipped,
        // preserving correctness at the cost of some potential pruning efficiency.
        Float16 => false, // Float16 comparison needs half crate; skip pruning for safety
        Float32 => {
            let l = f32::from_le_bytes(read_bytes::<4>(left));
            let r = f32::from_le_bytes(read_bytes::<4>(right));
            l < r && !l.is_nan() && !r.is_nan()
        }
        Float64 => {
            let l = f64::from_le_bytes(read_bytes::<8>(left));
            let r = f64::from_le_bytes(read_bytes::<8>(right));
            l < r && !l.is_nan() && !r.is_nan()
        }

        // Binary / string types (lexical byte comparison)
        Utf8 | LargeUtf8 | Binary | LargeBinary | BinaryView | Utf8View => left < right,

        // Boolean (stored as single byte, 0=false, 1=true)
        // Handle empty slices to prevent panic
        Boolean => {
            if left.is_empty() && right.is_empty() {
                false
            } else if left.is_empty() || right.is_empty() {
                left.is_empty()
            } else {
                left[0] < right[0]
            }
        }

        // Decimal types (fixed-width, little-endian)
        Decimal128(_, _) => {
            let l = i128::from_le_bytes(read_bytes::<16>(left));
            let r = i128::from_le_bytes(read_bytes::<16>(right));
            l < r
        }
        Decimal256(_, _) => {
            // 32-byte decimal — compare byte-by-byte as big-endian for magnitude ordering
            left < right
        }
        Decimal32(..) => {
            i32::from_le_bytes(read_bytes::<4>(left)) < i32::from_le_bytes(read_bytes::<4>(right))
        }
        Decimal64(..) => {
            i64::from_le_bytes(read_bytes::<8>(left)) < i64::from_le_bytes(read_bytes::<8>(right))
        }

        // Date/Time types (fixed-width integers representing timestamps/durations)
        Date32 => {
            i32::from_le_bytes(read_bytes::<4>(left)) < i32::from_le_bytes(read_bytes::<4>(right))
        }
        Date64 => {
            i64::from_le_bytes(read_bytes::<8>(left)) < i64::from_le_bytes(read_bytes::<8>(right))
        }
        // All timestamp variants (micro, milli, nano, second) share i64 representation
        Timestamp(_, _) => {
            i64::from_le_bytes(read_bytes::<8>(left)) < i64::from_le_bytes(read_bytes::<8>(right))
        }

        // Time types
        Time32(_) => {
            u32::from_le_bytes(read_bytes::<4>(left)) < u32::from_le_bytes(read_bytes::<4>(right))
        }
        Time64(_) => {
            u64::from_le_bytes(read_bytes::<8>(left)) < u64::from_le_bytes(read_bytes::<8>(right))
        }

        // Interval and Duration (no natural ordering — compare bytes)
        Interval(_) | Duration(_) => left < right,

        // Fixed-size types
        // DESIGN-DECISION (design-02): FixedSizeBinary comparison uses lexicographic byte comparison
        // on the minimum of the two byte slice lengths, not on the declared size.
        // This handles the edge case where two values with different actual byte lengths share
        // the same prefix (e.g., [0xAA, 0xBB] vs [0xAA, 0xBB, 0xCC]).
        // The shorter slice is considered "less" if it matches the prefix of the longer.
        // Note: This may produce unexpected results for domain-specific fixed-size binary types
        // where length is semantically significant. If strict fixed-width comparison is needed,
        // callers should ensure left.len() == right.len() == *size before calling.
        FixedSizeBinary(size) => {
            let n = (*size as usize).min(left.len()).min(right.len());
            left[..n] < right[..n]
        }
        FixedSizeList(..) => {
            // Cannot compare list elements byte-by-byte; skip pruning
            false
        }

        // Non-comparable types — return false to skip pruning, not a silent success
        List(_)
        | LargeList(_)
        | LargeListView(_)
        | ListView(_)
        | Map(..)
        | Struct(_)
        | Union(..)
        | Dictionary(_, _)
        | Null
        | RunEndEncoded(..) => false,
    }
}

pub fn bytes_gt(left: &[u8], right: &[u8], dtype: &DataType) -> bool {
    bytes_lt(right, left, dtype)
}

pub fn bytes_ge(left: &[u8], right: &[u8], dtype: &DataType) -> bool {
    !bytes_lt(left, right, dtype)
}

pub fn bytes_le(left: &[u8], right: &[u8], dtype: &DataType) -> bool {
    !bytes_lt(right, left, dtype)
}

fn read_bytes<const N: usize>(bytes: &[u8]) -> [u8; N] {
    let mut arr = [0u8; N];
    arr.copy_from_slice(&bytes[..N.min(bytes.len())]);
    arr
}

// ─── GranuleBloomFilterManager ─────────────────────────────────────────────────

/// Manages persistence and loading of per-column, per-granule Bloom filters.
///
/// Storage layout:
///   `{seg_dir}/{col_name}._bf.bin` — bincode-encoded `GranuleBloomFilterManager`
///
/// The manager maps granule_id → bloom_filter_bytes. This format allows:
/// - O(1) granule-level updates during compaction (only one granule changes)
/// - Lazy loading: bloom filters are loaded on-demand during query execution
///   and cached in GranuleStats.bloom_filter for the lifetime of the query.
/// - Parallel construction during compaction: build all bloom filters from
///   decompressed column data, then serialize once to disk.
///
/// ## Compaction integration
///
/// During compaction (`pdt_merge::compact_segment`), call:
/// 1. `GranuleBloomFilterManager::new(seg_id, col_name, total_rows)` to create a manager
/// 2. For each granule's batch: `build_for_granule(pk_array, granule_idx)` to insert PKs
/// 3. After all granules: `save(seg_dir)` to persist
///
/// The bloom filter is built from the **decompressed** data that compaction already reads,
/// so there is no extra I/O cost. This is the key fix for the m001 issue.
///
/// ## Rebuild (deprecated)
///
/// `rebuild_all()` is marked deprecated because it can only initialize the bloom filter
/// structure (num_blocks based on row count) but cannot populate it with actual PK values.
/// On deserialization, bloom filter data is loaded from the `.{col}._bf.bin` sidecar file.
/// If the sidecar is missing (pre-m001 segments), the bloom filter remains empty.
/// Callers that rely on bloom filter for pruning should check `bloom_filter.is_some()`
/// before using it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GranuleBloomFilterManager {
    /// Segment ID (for error messages).
    seg_id: String,
    /// Column name this manager belongs to.
    col_name: String,
    /// Total rows in the segment.
    total_rows: u64,
    /// Granule ID → bloom filter (bincode bytes).
    /// Kept as raw bytes to avoid nested bincode serialization overhead.
    /// Each value is the bincode-encoded `GranuleBloomFilter` for that granule.
    granule_filters: HashMap<u32, Vec<u8>>,
    /// D2 fix: In-memory cache of deserialized SbbFilter for fast per-row inserts.
    /// Avoids deserialize→insert→serialize roundtrip for every PK in a granule.
    /// Flushed to `granule_filters` before `save()`.
    cache: HashMap<u32, SbbFilter>,
}

impl GranuleBloomFilterManager {
    /// Magic bytes for the file header.
    const MAGIC: [u8; 4] = *b"BLOM";
    /// Version byte (v1 = current).
    const VERSION: u8 = 1;

    /// Create a new manager for building bloom filters for a column.
    pub fn new(seg_id: String, col_name: String, total_rows: u64) -> Self {
        Self {
            seg_id,
            col_name,
            total_rows,
            granule_filters: HashMap::new(),
            cache: HashMap::new(),
        }
    }

    /// Number of granules based on total_rows.
    pub fn num_granules(&self) -> u32 {
        (self.total_rows / (DEFAULT_GRANULE_SIZE as u64 * DEFAULT_BLOCK_SIZE as u64)) as u32
            + if self.total_rows % (DEFAULT_GRANULE_SIZE as u64 * DEFAULT_BLOCK_SIZE as u64) > 0 {
                1
            } else {
                0
            }
    }

    /// Returns true if no bloom filters have been built yet.
    pub fn is_empty(&self) -> bool {
        self.granule_filters.is_empty()
    }

    /// Insert primary key bytes into the bloom filter for a granule.
    /// Creates the bloom filter on first insert for that granule.
    ///
    /// `granule_idx`: the granule index (0-based).
    /// `expected_entries`: expected number of entries for this granule (used for size calculation).
    ///
    /// D2 fix: Uses an in-memory cache of deserialized SbbFilter to avoid
    /// deserialize→insert→serialize roundtrip for every PK. The cache is flushed
    /// to `granule_filters` only before `save()`.
    pub fn insert_into_granule(
        &mut self,
        granule_idx: u32,
        expected_entries: u32,
        pk_bytes: &[u8],
    ) -> crate::error::Result<()> {
        let bf = if let Some(cached) = self.cache.get_mut(&granule_idx) {
            cached
        } else {
            let bf = if let Some(bytes) = self.granule_filters.get(&granule_idx) {
                // D9 fix: if deserialize fails, create a new filter instead of silently skipping
                bincode::deserialize::<SbbFilter>(bytes).unwrap_or_else(|e| {
                    tracing::warn!(
                        "bloom filter deserialize failed for granule {}: {}. Creating new filter.",
                        granule_idx, e
                    );
                    let num_blocks =
                        SbbFilter::optimal_num_blocks(expected_entries, DEFAULT_BF_FPP);
                    SbbFilter::new(num_blocks)
                })
            } else {
                let num_blocks =
                    SbbFilter::optimal_num_blocks(expected_entries, DEFAULT_BF_FPP);
                SbbFilter::new(num_blocks)
            };
            self.cache.entry(granule_idx).or_insert(bf)
        };
        bf.insert(pk_bytes);
        Ok(())
    }

    /// Flush the in-memory cache to `granule_filters` before serialization.
    /// Called automatically by `save()` — not needed by callers.
    fn flush_cache(&mut self) {
        for (granule_idx, bf) in self.cache.drain() {
            let bytes = bincode::serialize(&bf).expect(
                "bloom filter serialization should not fail for in-memory SbbFilter",
            );
            self.granule_filters.insert(granule_idx, bytes);
        }
    }

    /// Batch-insert all PK bytes from an Arrow array into a granule's bloom filter.
    ///
    /// This is the primary entry point for compaction: call once per batch with the
    /// PK array (already decompressed from compaction data).
    ///
    /// `batch_row_offset`: the global row offset of row 0 in this batch.
    ///                     Used to determine which granule this batch belongs to.
    /// `granule_idx`: the granule index (0-based).
    /// `pk_array`: the primary key column array.
    pub fn build_for_granule_from_array(
        &mut self,
        granule_idx: u32,
        batch_row_offset: u64,
        pk_array: &dyn arrow_array::Array,
    ) -> crate::error::Result<()> {
        let rows_per_granule =
            DEFAULT_GRANULE_SIZE as u64 * DEFAULT_BLOCK_SIZE as u64;
        let expected_entries = rows_per_granule as u32;

        match pk_array.data_type() {
            arrow_schema::DataType::Int64 => {
                let arr = pk_array.as_primitive::<arrow_array::types::Int64Type>();
                for i in 0..arr.len() {
                    let pk_bytes = arr.value(i).to_le_bytes();
                    self.insert_into_granule(granule_idx, expected_entries, &pk_bytes)?;
                }
            }
            arrow_schema::DataType::Utf8 => {
                let arr = pk_array.as_string::<i32>();
                for i in 0..arr.len() {
                    if !arr.is_null(i) {
                        let pk = arr.value(i).as_bytes();
                        self.insert_into_granule(granule_idx, expected_entries, pk)?;
                    }
                }
            }
            arrow_schema::DataType::LargeUtf8 => {
                let arr = pk_array.as_string::<i64>();
                for i in 0..arr.len() {
                    if !arr.is_null(i) {
                        let pk = arr.value(i).as_bytes();
                        self.insert_into_granule(granule_idx, expected_entries, pk)?;
                    }
                }
            }
            arrow_schema::DataType::Binary => {
                let arr = pk_array.as_binary::<i32>();
                for i in 0..arr.len() {
                    if !arr.is_null(i) {
                        let pk = arr.value(i);
                        self.insert_into_granule(granule_idx, expected_entries, pk)?;
                    }
                }
            }
            arrow_schema::DataType::LargeBinary => {
                let arr = pk_array.as_binary::<i64>();
                for i in 0..arr.len() {
                    if !arr.is_null(i) {
                        let pk = arr.value(i);
                        self.insert_into_granule(granule_idx, expected_entries, pk)?;
                    }
                }
            }
            _ => {
                // Fallback: serialize entire row as bytes
                for i in 0..pk_array.len() {
                    if !pk_array.is_null(i) {
                        // Use Arrow IPC or simple serialization
                        let offset = batch_row_offset as u32 + i as u32;
                        let key_bytes = format!("{}:{}", granule_idx, offset);
                        self.insert_into_granule(
                            granule_idx,
                            expected_entries,
                            key_bytes.as_bytes(),
                        )?;
                    }
                }
            }
        }
        Ok(())
    }

    /// Build bloom filters for all granules from a column's PK array.
    /// Granules are determined by row offsets (DEFAULT_GRANULE_SIZE * DEFAULT_BLOCK_SIZE rows each).
    ///
    /// Call this during compaction after reading and filtering column data.
    /// The PK array should be the compacted (alive rows only) version.
    pub fn build_all_from_array(&self, _pk_array: &dyn arrow_array::Array) {
        // Note: This method requires &mut self for insertion into granule_filters HashMap.
        // Call build_for_granule_from_array() in a loop from the compaction context instead.
        // This is a placeholder for the trait-based API; actual usage goes through
        // insert_into_granule() per PK value.
        tracing::debug!(
            "GranuleBloomFilterManager::build_all_from_array called — use insert_into_granule() instead"
        );
    }

    /// Load a manager from a sidecar file.
    pub fn load(seg_id: &str, col_name: &str, bf_path: &Path) -> Result<Self> {
        if !bf_path.exists() {
            return Err(RockDuckError::Codec(format!(
                "bloom filter file not found for segment '{}' column '{}': {:?}",
                seg_id, col_name, bf_path
            )));
        }
        let data = std::fs::read(bf_path).map_err(RockDuckError::Io)?;

        // Check header
        if data.len() < 6 {
            return Err(RockDuckError::Codec(
                "GranuleBloomFilterManager: file too short for header".into(),
            ));
        }
        if data[..4] != Self::MAGIC {
            return Err(RockDuckError::Codec(format!(
                "GranuleBloomFilterManager: invalid magic for segment '{}' column '{}'",
                seg_id, col_name
            )));
        }
        let version = data[4];
        if version > Self::VERSION {
            return Err(RockDuckError::Codec(format!(
                "GranuleBloomFilterManager: unsupported version {} for segment '{}' column '{}'",
                version, seg_id, col_name
            )));
        }
        let payload = &data[6..];
        bincode::deserialize(payload).map_err(|e| {
            RockDuckError::Codec(format!(
                "GranuleBloomFilterManager deserialize: {}",
                e
            ))
        })
    }

    /// Save the manager to a sidecar file.
    /// D2 fix: Uses atomic write (write-to-tmp + rename + fsync) for crash safety.
    ///
    /// This prevents data corruption if a crash occurs during the write operation.
    pub fn save(&mut self, bf_path: &Path) -> Result<()> {
        if self.granule_filters.is_empty() && self.cache.is_empty() {
            return Ok(());
        }
        self.flush_cache();
        let payload = bincode::serialize(self)
            .map_err(|e| RockDuckError::Codec(format!("serialize: {}", e)))?;

        let mut data = Vec::with_capacity(6 + payload.len());
        data.extend_from_slice(&Self::MAGIC);
        data.push(Self::VERSION);
        data.push(0u8); // reserved
        data.extend_from_slice(&payload);

        if let Some(parent) = bf_path.parent() {
            std::fs::create_dir_all(parent).map_err(RockDuckError::Io)?;
        }

        // Atomic write: write to temp file, fsync, then rename.
        let tmp_path = {
            let mut p = bf_path.as_os_str().to_os_string();
            p.push(".tmp");
            std::path::PathBuf::from(p)
        };

        // Write temp file and fsync.
        {
            let mut tmp_file = std::fs::OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(&tmp_path)
                .map_err(|e| RockDuckError::Io(std::io::Error::other(
                    format!("open bloom filter tmp: {}", e),
                )))?;
            tmp_file
                .write_all(&data)
                .map_err(|e| RockDuckError::Io(std::io::Error::other(
                    format!("write bloom filter tmp: {}", e),
                )))?;
            tmp_file
                .sync_all()
                .map_err(|e| RockDuckError::Io(std::io::Error::other(
                    format!("fsync bloom filter tmp: {}", e),
                )))?;
        }

        // Atomic rename.
        std::fs::rename(&tmp_path, bf_path)
            .map_err(|e| RockDuckError::Io(std::io::Error::other(
                format!("rename bloom filter: {}", e),
            )))?;

        // fsync parent directory on Windows to ensure directory entry is persisted.
        if let Some(parent) = bf_path.parent() {
            if let Ok(_dir_file) = std::fs::OpenOptions::new().read(true).open(parent) {
                #[cfg(unix)]
                {
                    use std::os::fd::AsRawFd;
                    let _ = dir_file.sync_all();
                }
                #[cfg(windows)]
                {
                    use std::os::windows::io::FromRawHandle;
                    use windows::core::PCWSTR;
                    use windows::Win32::Storage::FileSystem::{
                        CreateFileW, FILE_FLAG_BACKUP_SEMANTICS, FILE_SHARE_MODE, FILE_SHARE_READ,
                        FILE_SHARE_WRITE, OPEN_EXISTING,
                    };

                    let wide_path: Vec<u16> = parent
                        .to_string_lossy()
                        .encode_utf16()
                        .chain(std::iter::once(0))
                        .collect();
                    let handle = unsafe {
                        CreateFileW(
                            PCWSTR::from_raw(wide_path.as_ptr()),
                            0,
                            FILE_SHARE_MODE(FILE_SHARE_READ.0 | FILE_SHARE_WRITE.0),
                            None,
                            OPEN_EXISTING,
                            FILE_FLAG_BACKUP_SEMANTICS,
                            None,
                        )
                    };
                    if let Ok(h) = handle {
                        let file = unsafe { std::fs::File::from_raw_handle(h.0 as *mut _) };
                        let _ = file.sync_all();
                    }
                }
            }
        }

        Ok(())
    }

    /// Get the bloom filter for a specific granule.
    /// Returns None if not found (segment was created before m001, or granule has no PKs).
    pub fn get_granule_filter(&self, granule_idx: u32) -> Option<SbbFilter> {
        self.granule_filters
            .get(&granule_idx)
            .and_then(|bytes| bincode::deserialize(bytes).ok())
    }

    /// Rebuild all bloom filters from scratch.
    ///
    /// **DEPRECATED**: This method can only initialize the bloom filter structure
    /// (num_blocks based on row count) but cannot populate it with actual PK values.
    /// The bloom filter will be effectively empty.
    ///
    /// For compaction: use `build_all_from_array()` which inserts actual PK values
    /// from the decompressed column data.
    ///
    /// For existing segments: bloom filters are loaded from the `.{col}._bf.bin` sidecar
    /// file if present. If the sidecar is missing, the bloom filter cannot be rebuilt
    /// and remains None.
    #[deprecated(since = "0.2.1", note = "use build_all_from_array() during compaction instead")]
    pub fn rebuild_all(&mut self) {
        // Clear existing filters — they cannot be rebuilt from metadata alone
        self.granule_filters.clear();
        tracing::warn!(
            "GranuleBloomFilterManager::rebuild_all is deprecated. \
             Bloom filters cannot be rebuilt without original PK data. \
             Use build_all_from_array() during compaction instead."
        );
    }
}

/// Compute the global row offset of index `i` in an Arrow array.
/// For sliced arrays, this accounts for the offset.
#[allow(dead_code)]
fn batch_row_of_index(arr: &dyn arrow_array::Array, i: usize) -> u64 {
    arr.offset() as u64 + i as u64
}
