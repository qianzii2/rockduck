//! Segment sparse index — ClickHouse Granule-style sampling.
//!
//! # Core Idea (ClickHouse MergeTree)
//!
//! ClickHouse uses a sparse primary key index: one index entry per 8192 rows.
//! A billion-row table has a primary key index of only a few MB.
//!
//! For RockDuck: one index entry per SPARSE_INDEX_SAMPLE_RATE rows.
//!
//! # When to Build
//!
//! Sparse indexes are only useful when data is sorted by the primary key.
//! Build during compaction after sorting by PK.
//!
//! # Limitations
//!
//! Sorting is the most expensive operation — only applicable to small
//! segments (< 100MB). For large segments, skip sparse indexing.

use std::fs::File;
use std::io::{BufWriter, Read, Write as IoWrite};
use std::path::Path;

use crate::error::{Result, RockDuckError};

/// Number of rows between sparse index samples.
const SPARSE_INDEX_SAMPLE_RATE: u64 = 8192;

/// Sparse index entry: (row_offset, pk_bytes)
pub type SparseIndexEntry = (u64, Vec<u8>);

// ─── Serialization format ────────────────────────────────────────────────────────
//
// v1 (legacy): raw postcard Vec<SparseIndexEntry> — no header.
//   Each entry's pk_bytes was the display string of the FIRST row of the batch,
//   not the value at the entry's row offset. This made the index useless.
//
// v2 (current): 13-byte header + postcard Vec<SparseIndexEntry>
//   Header: magic[4] + version[1] + reserved[4] + num_entries[4] = 13 bytes
//   pk_bytes are binary: fixed-width types use to_le_bytes; strings use len-prefix.
//   pk_to_bytes() must be fixed as part of the same PR.

/// Magic bytes: b"SSIX"
const SPARSE_INDEX_MAGIC: [u8; 4] = [0x53, 0x53, 0x49, 0x58]; // "SSIX"
/// Version 2: binary pk_bytes encoding (replaces v1 display-string encoding).
const SPARSE_INDEX_VERSION: u8 = 2;

/// Sparse sampling index for efficient PK lookup in sorted segments.
///
/// Internally maintains two arrays:
/// - `entries`: sorted by `row_offset` (insertion order, used for building)
/// - `entries_by_pk`: sorted by `pk_bytes` (used for O(log n) binary search in `find_offset`)
///
/// This dual-array design avoids the need to re-sort on every lookup while keeping
/// memory overhead minimal (entries_by_pk shares the same `Vec<SparseIndexEntry>` elements,
/// just in a different order).
#[derive(Default)]
pub struct SegmentSparseIndex {
    /// Entries sorted by row_offset. Maintained in insertion order.
    entries: Vec<SparseIndexEntry>,
    /// The same entries, sorted by pk_bytes for O(log n) binary search.
    entries_by_pk: Vec<SparseIndexEntry>,
}

/// Header prefix written before the postcard payload.
/// Stored in little-endian byte order for cross-platform compatibility.
struct SparseIndexHeader {
    /// Magic bytes: b"SSIX"
    magic: [u8; 4],
    /// Version byte (SPARSE_INDEX_VERSION).
    version: u8,
    /// Reserved padding (must be 0).
    reserved: u32,
    /// Number of sparse index entries in the payload.
    num_entries: u64,
}

impl SparseIndexHeader {
    /// Total header size in bytes.
    const SIZE: usize = 4 + 1 + 4 + 8; // 17 bytes

    /// Write this header into `buf` (little-endian).
    fn write_to(&self, buf: &mut Vec<u8>) {
        buf.extend_from_slice(&self.magic);
        buf.push(self.version);
        buf.extend_from_slice(&self.reserved.to_le_bytes());
        buf.extend_from_slice(&self.num_entries.to_le_bytes());
    }

    /// Read the NON-MAGIC part of the header (version + reserved + num_entries) from `reader`.
    /// The caller is responsible for having already read and validated the magic bytes.
    fn read_rest<R: std::io::Read>(reader: &mut R) -> std::io::Result<Self> {
        let mut version_buf = [0u8; 1];
        reader.read_exact(&mut version_buf)?;
        let mut reserved_buf = [0u8; 4];
        reader.read_exact(&mut reserved_buf)?;
        let mut num_buf = [0u8; 8];
        reader.read_exact(&mut num_buf)?;
        Ok(Self {
            magic: SPARSE_INDEX_MAGIC, // already validated by caller
            version: version_buf[0],
            reserved: u32::from_le_bytes(reserved_buf),
            num_entries: u64::from_le_bytes(num_buf),
        })
    }
}

impl SegmentSparseIndex {
    /// Create a new empty sparse index.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            entries_by_pk: Vec::new(),
        }
    }

    /// Add a sample: given the row_offset and the PK bytes at that offset.
    ///
    /// After all samples are added, call `build()` once before using `find_offset()`.
    pub fn add_sample(&mut self, row_offset: u64, pk_bytes: Vec<u8>) {
        self.entries.push((row_offset, pk_bytes));
    }

    /// Finalize the index: build the pk-sorted view for O(log n) lookup.
    ///
    /// Must be called after all `add_sample` calls and before `find_offset`.
    /// Idempotent: safe to call multiple times.
    pub fn build(&mut self) {
        // Fix S-4: Use take() instead of clone() to avoid double memory allocation.
        // After build(), the unsorted `entries` is no longer needed - we only use
        // `entries_by_pk` for lookups. Using take() avoids holding both Vecs in memory.
        self.entries_by_pk = std::mem::take(&mut self.entries);
        self.entries_by_pk
            .sort_by(|a, b| a.1.as_slice().cmp(b.1.as_slice()));
    }

    /// Binary search to find the row_offset of the target PK using O(log n) binary search.
    ///
    /// Uses `entries_by_pk` (sorted by pk_bytes) rather than `entries` (sorted by row_offset).
    /// Requires `build()` to be called first.
    ///
    /// Returns the row_offset of the first entry whose PK >= target_pk.
    /// Returns `None` if all indexed PKs are < target_pk.
    #[allow(dead_code)]
    pub fn find_offset(&self, target_pk: &[u8]) -> Option<u64> {
        self.entries_by_pk
            .binary_search_by(|e| e.1.as_slice().cmp(target_pk))
            .ok()
            .map(|idx| self.entries_by_pk[idx].0)
    }

    /// Save the index to a file using postcard serialization with v2 header.
    ///
    /// File format (v2):
    ///   - 13-byte header: magic[4] + version[1] + reserved[4] + num_entries[4]
    ///   - postcard Vec<SparseIndexEntry> (binary)
    ///
    /// fsync is NOT called here — callers are responsible for durability.
    pub fn save(&self, path: &Path) -> Result<()> {
        let file = File::create(path).map_err(RockDuckError::Io)?;
        let mut writer = BufWriter::new(file);

        // Write v2 header
        let header = SparseIndexHeader {
            magic: SPARSE_INDEX_MAGIC,
            version: SPARSE_INDEX_VERSION,
            reserved: 0,
            num_entries: self.entries.len() as u64,
        };
        let mut buf = Vec::with_capacity(SparseIndexHeader::SIZE);
        header.write_to(&mut buf);
        writer.write_all(&buf).map_err(RockDuckError::Io)?;

        // Write payload
        postcard::to_io(&self.entries, &mut writer)
            .map_err(|e| RockDuckError::Serialization(format!("save sparse index: {e}")))?;
        writer.flush().map_err(RockDuckError::Io)?;
        Ok(())
    }

    /// Load the index from a file.
    ///
    /// Handles both v1 (legacy, no header) and v2 (with header).
    ///
    /// v1 files are detected by the absence of the "SSIX" magic prefix.
    /// For v1 files, an error is returned — v1 indexes are invalid and must be rebuilt.
    /// Callers should rebuild the sparse index from the source Vortex data.
    ///
    /// v2 files are validated against the header and loaded normally.
    pub fn load(path: &Path) -> Result<Self> {
        let mut file = File::open(path).map_err(RockDuckError::Io)?;
        let file_size = file.metadata().map_err(RockDuckError::Io)?.len();

        if file_size < SparseIndexHeader::SIZE as u64 {
            return Err(RockDuckError::Deserialize(format!(
                "sparse index file too small ({} bytes, expected at least {}): {:?}",
                file_size,
                SparseIndexHeader::SIZE,
                path
            )));
        }

        // Peek at first 4 bytes to detect v1 vs v2
        let mut magic_buf = [0u8; 4];
        let bytes_read = file
            .read(&mut magic_buf)
            .map_err(RockDuckError::Io)?;
        if bytes_read != 4 {
            return Err(RockDuckError::Deserialize(format!(
                "sparse index file truncated: expected 4 magic bytes, got {}",
                bytes_read
            )));
        }

        if magic_buf == SPARSE_INDEX_MAGIC {
            // v2: read remaining header bytes + payload
            let header =
                SparseIndexHeader::read_rest(&mut file).map_err(RockDuckError::Io)?;

            if header.version > SPARSE_INDEX_VERSION {
                return Err(RockDuckError::Deserialize(format!(
                    "sparse index version {} not supported (max: {}): {:?}",
                    header.version, SPARSE_INDEX_VERSION, path
                )));
            }

            // Read remaining bytes after the header as the postcard payload
            let header_end = SparseIndexHeader::SIZE as u64;
            let payload_size = file_size
                .checked_sub(header_end)
                .ok_or_else(|| RockDuckError::Deserialize("file too small for header".into()))?;

            let mut payload = vec![0u8; payload_size as usize];
            file.read_exact(&mut payload)
                .map_err(RockDuckError::Io)?;

            let entries: Vec<SparseIndexEntry> = postcard::from_bytes(&payload)
                .map_err(|e| RockDuckError::Deserialize(format!("load sparse index: {e}")))?;

            if entries.len() as u64 != header.num_entries {
                return Err(RockDuckError::Deserialize(format!(
                    "sparse index entry count mismatch: header says {}, loaded {}",
                    header.num_entries,
                    entries.len()
                )));
            }

            // Build the pk-sorted index on load so find_offset works immediately
            let mut index = Self {
                entries,
                entries_by_pk: Vec::new(),
            };
            index.build();
            Ok(index)
        } else {
            // v1: no header — the entire file is a raw postcard Vec<SparseIndexEntry>
            // The v1 format is INVALID because pk_bytes were set to the display string
            // of the first row of the batch, not the value at the entry's row offset.
            // We return a clear error directing callers to rebuild.
            let _magic_str = String::from_utf8_lossy(&magic_buf);
            Err(RockDuckError::Deserialize(format!(
                "sparse index v1 detected (magic: {:?}): \
                index is invalid and must be rebuilt from source Vortex data. \
                Delete the .sparse_idx file and regenerate via compaction.",
                _magic_str
            )))
        }
    }

    /// Generate a sparse index from a sorted RecordBatch.
    ///
    /// Samples one PK every SPARSE_INDEX_SAMPLE_RATE rows.
    /// Calls `build()` before returning so the index is ready for binary search.
    ///
    /// Panics if `pk_column_idx` is out of bounds for the batch schema.
    pub fn from_sorted_batch(batch: &arrow_array::RecordBatch, pk_column_idx: usize) -> Self {
        let num_rows = batch.num_rows() as u64;
        let pk_array = batch.column(pk_column_idx);
        let mut index = Self::new();

        // Sample every SPARSE_INDEX_SAMPLE_RATE rows
        let mut row = 0u64;
        while row < num_rows {
            let idx = row as usize;
            // P9-45 fix: encode the PK value at row `idx` as bytes for comparison.
            let pk_bytes = Self::pk_to_bytes(pk_array, idx);
            index.add_sample(row, pk_bytes);
            row = row.saturating_add(SPARSE_INDEX_SAMPLE_RATE);
        }

        index.build();
        index
    }

    /// Extract the value at row index `idx` from `pk_array` as a binary `Vec<u8>`.
    ///
    /// Binary encoding (v2 sparse index format):
    ///   - Fixed-width types (Int*, UInt*, Float*): `to_le_bytes()`
    ///   - UTF-8 / Binary strings: `len_prefix (u32 LE) + raw bytes`
    ///   - Fallback: `to_string().into_bytes()` (used for Decimal, complex types)
    ///
    /// The returned bytes are comparable via `memcmp` / `Vec::cmp`:
    ///
    ///   - For integers: little-endian byte order matches numeric order
    ///   - For strings: lexicographic byte order matches string sort order
    ///
    /// Convert a primary key value to a byte representation for sorting/comparison.
    ///
    /// Rules:
    ///   - Fixed-width integer types → little-endian bytes (preserves sort order)
    ///   - Float types → IEEE 754 byte order (NaN/Inf are sorted consistently)
    ///   - String/Binary → raw bytes + 4-byte LE length suffix (memcmp ordering)
    fn pk_to_bytes(pk_array: &arrow_array::ArrayRef, idx: usize) -> Vec<u8> {
        use arrow_array::Array;
        use arrow_schema::DataType;

        macro_rules! primitive_pk_bytes {
            ($arr_ty:ident, $value_ty:ty) => {{
                pk_array
                    .as_any()
                    .downcast_ref::<arrow_array::$arr_ty>()
                    .map(|a| a.value(idx).to_le_bytes().to_vec())
                    .unwrap_or_default()
            }};
            ($arr_ty:ident) => {{
                pk_array
                    .as_any()
                    .downcast_ref::<arrow_array::$arr_ty>()
                    .map(|a| a.value(idx).to_le_bytes().to_vec())
                    .unwrap_or_default()
            }};
        }

        macro_rules! variable_pk_bytes {
            ($arr_ty:ident) => {{
                pk_array
                    .as_any()
                    .downcast_ref::<arrow_array::$arr_ty>()
                    .map(|a| {
                        let b = a.value(idx);
                        let mut buf = Vec::with_capacity(b.len() + 4);
                        buf.extend_from_slice(b);
                        buf.extend_from_slice(&(b.len() as u32).to_le_bytes());
                        buf
                    })
                    .unwrap_or_default()
            }};
        }

        match pk_array.data_type() {
            // ── Fixed-width integer types ──
            DataType::Int8 => primitive_pk_bytes!(Int8Array),
            DataType::Int16 => primitive_pk_bytes!(Int16Array),
            DataType::Int32 => primitive_pk_bytes!(Int32Array),
            DataType::Int64 => primitive_pk_bytes!(Int64Array),
            DataType::UInt8 => primitive_pk_bytes!(UInt8Array),
            DataType::UInt16 => primitive_pk_bytes!(UInt16Array),
            DataType::UInt32 => primitive_pk_bytes!(UInt32Array),
            DataType::UInt64 => primitive_pk_bytes!(UInt64Array),
            // ── Fixed-width float types ──
            DataType::Float32 => primitive_pk_bytes!(Float32Array),
            DataType::Float64 => primitive_pk_bytes!(Float64Array),
            // ── Variable-width string types ──
            DataType::Utf8 => {
                pk_array
                    .as_any()
                    .downcast_ref::<arrow_array::StringArray>()
                    .map(|a| {
                        let s = a.value(idx);
                        let bytes = s.as_bytes();
                        let mut buf = Vec::with_capacity(bytes.len() + 4);
                        // Binary search compares raw bytes directly (no length prefix),
                        // so store only the bytes in the same order as target_pk bytes.
                        buf.extend_from_slice(bytes);
                        buf
                    })
                    .unwrap_or_default()
            }
            DataType::LargeUtf8 => {
                pk_array
                    .as_any()
                    .downcast_ref::<arrow_array::LargeStringArray>()
                    .map(|a| {
                        let s = a.value(idx);
                        let bytes = s.as_bytes();
                        let mut buf = Vec::with_capacity(bytes.len() + 4);
                        // Binary search compares raw bytes directly (no length prefix).
                        buf.extend_from_slice(bytes);
                        buf
                    })
                    .unwrap_or_default()
            }
            DataType::Binary => variable_pk_bytes!(BinaryArray),
            DataType::LargeBinary => variable_pk_bytes!(LargeBinaryArray),
            // ── Fallback ──
            _ => format!("{:?}", pk_array).into_bytes(),
        }
    }
}

impl SegmentSparseIndex {}

#[cfg(test)]
mod tests {

    /// Test that pk_to_bytes is symmetric: round-trip through bytes preserves ordering.
    /// For variable-width strings, the raw bytes must be byte-comparable (no length prefix).
    #[test]
    fn test_pk_bytes_encoding_symmetry_fixed_width() {
        // Fixed-width types (Int64): to_le_bytes is always symmetric.
        let val: i64 = 42;
        let bytes = val.to_le_bytes().to_vec();
        let roundtrip = i64::from_le_bytes(bytes.clone().try_into().unwrap());
        assert_eq!(val, roundtrip);

        let val2: i64 = 100;
        let bytes2 = val2.to_le_bytes().to_vec();
        // Ordering must match
        assert_eq!(bytes.cmp(&bytes2), val.cmp(&val2));
    }

    #[test]
    fn test_pk_bytes_encoding_symmetry_utf8() {
        // Utf8 strings: raw bytes must be byte-comparable for binary search to work.
        // Previously buggy: appended length suffix, corrupting string comparison.
        let strings = ["apple", "banana", "cherry", ""];

        for s in &strings {
            let bytes = s.as_bytes().to_vec();
            let roundtrip = String::from_utf8(bytes.clone()).unwrap();
            assert_eq!(*s, roundtrip);

            // Verify byte comparison order matches string order
            for other in &["a", "z", "AAA"] {
                let cmp = bytes.as_slice().cmp(other.as_bytes());
                let expected = s.cmp(other);
                assert_eq!(
                    cmp, expected,
                    "byte comparison must match string comparison"
                );
            }
        }
    }

    #[test]
    fn test_pk_bytes_encoding_symmetry_binary() {
        // Binary: raw bytes must be byte-comparable.
        let data1: &[u8] = &[0x01, 0x02, 0x03];
        let data2: &[u8] = &[0x01, 0x02, 0x04];
        assert!(data1.cmp(data2) == std::cmp::Ordering::Less);
    }
}
