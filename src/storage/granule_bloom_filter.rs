//! GranuleBloomFilter — fastbloom Bloom filter with serde persistence and auto-save on Drop.
//!
//! `fastbloom` is used over `quickbloom` because its `&(impl Hash + ?Sized)` trait bounds
//! allow direct `&[u8]` slice hashing (no wrapper types needed). This is critical for
//! granule-level bloom filters that index primary keys and record identifiers as raw bytes.
//!
//! Serialization uses `bincode` (compact binary format, ~10x smaller than JSON).

use std::io::Write;
use std::path::{Path, PathBuf};

use crate::error::{Result, RockDuckError};
use crate::metadata::GranuleId;
use fastbloom::{AtomicBloomFilter, BloomFilter, DefaultHasher};
use serde::{Deserialize, Serialize};

// DEFAULT_FPP: f64 = 0.01; // false positive probability for bloom filter

// Number of bits per expected entry (optimal for ~1% FPP).
const BITS_PER_ENTRY: usize = 10;

fn optimal_bits(entries: usize) -> usize {
    // m / n ≈ -1.44 * log2(FPP) ≈ 9.6 bits/entry for 1% FPP
    (entries * BITS_PER_ENTRY).max(64)
}

// ─── FastbloomGranuleFilter ────────────────────────────────────────────────────

/// Granule-level Bloom filter using fastbloom (fastbloom crate).
/// Persisted to `.{col}._bf.bin` sidecar files.
///
/// Used by `GranuleBloomFilterManager` during compaction to build bloom filters
/// from decompressed column data. The manager persists filters to the sidecar file
/// so they survive deserialization (the in-memory `SbbFilter` in `GranuleStats`
/// is skip-serialized).
///
/// NOTE: This is a different implementation from `SbbFilter` in `block_zone_map.rs`
/// (Parquet SBBF spec). Both serve the same purpose but have different storage formats.
/// The fastbloom filter is more efficient for random insertions; the SBBF is more compact
/// for sequential data.
#[derive(Serialize, Deserialize)]
pub struct FastbloomGranuleFilter {
    filter: BloomFilter<DefaultHasher>,
    #[serde(skip)]
    seg_id: String,
    #[serde(skip)]
    granule_id: GranuleId,
    #[serde(skip)]
    path: PathBuf,
    #[serde(skip)]
    dirty: bool,
    /// Number of items inserted into the filter. Incremented on insert.
    #[serde(skip)]
    entries: u32,
}

impl FastbloomGranuleFilter {
    /// Create a new Bloom filter with auto-persistence on Drop.
    pub fn create(
        seg_id: String,
        granule_id: GranuleId,
        expected_entries: u32,
        path: PathBuf,
    ) -> Result<Self> {
        let num_bits = optimal_bits(expected_entries as usize);
        let filter = BloomFilter::with_num_bits(num_bits).expected_items(expected_entries as usize);
        Ok(Self {
            filter,
            seg_id,
            granule_id,
            path,
            dirty: true,
            entries: 0,
        })
    }

    /// Load a Bloom filter from disk.
    pub fn load(path: &Path) -> Result<Self> {
        if !path.exists() {
            return Err(RockDuckError::Codec(format!(
                "FastbloomGranuleFilter not found: {:?}",
                path
            )));
        }
        let bytes = std::fs::read(path).map_err(RockDuckError::Io)?;
        let mut gbf: FastbloomGranuleFilter = {
            let limit = 10 * 1024 * 1024; // 10 MiB cap
            if bytes.len() > limit {
                return Err(RockDuckError::Codec(format!(
                    "FastbloomGranuleFilter data too large: {} > {} bytes",
                    bytes.len(),
                    limit
                )));
            }
            bincode::deserialize(&bytes)
        }
        .map_err(|e| RockDuckError::Codec(format!("bloom deserialize: {}", e)))?;
        gbf.path = path.to_path_buf();
        gbf.dirty = false;
        Ok(gbf)
    }

    pub fn load_or_create(
        seg_id: String,
        granule_id: GranuleId,
        expected_entries: u32,
        path: PathBuf,
    ) -> Result<Self> {
        if path.exists() {
            Self::load(&path)
        } else {
            Self::create(seg_id, granule_id, expected_entries, path)
        }
    }

    pub fn insert(&mut self, key: &[u8]) {
        self.filter.insert(&key);
        self.dirty = true;
        self.entries += 1;
    }

    /// Remove a key from the filter.
    ///
    /// Note: Bloom filters do not support true deletion. This method returns an error
    /// because silently decrementing the entry count would lead to inconsistent state
    /// where the bits remain set but item_count() returns incorrect values.
    /// For true deletion support, a counting bloom filter or cuckoo filter would be needed.
    pub fn remove(&mut self, _key: &[u8]) -> Result<()> {
        Err(RockDuckError::Internal(
            "FastbloomGranuleFilter does not support removal (bloom filters have no delete operation)".into(),
        ))
    }

    pub fn might_contain(&self, key: &[u8]) -> bool {
        self.filter.contains(&key)
    }

    pub fn fill_ratio(&self) -> f64 {
        let bits = self.filter.num_bits();
        if bits == 0 {
            return 0.0;
        }
        let ones: usize = self.filter.iter().map(|w| w.count_ones() as usize).sum();
        ones as f64 / bits as f64
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
    /// Returns the number of items inserted into this filter.
    /// O(1) via the `entries` counter (instead of iterating the bitset).
    pub fn item_count(&self) -> usize {
        self.entries as usize
    }

    /// Manually persist to disk.
    ///
    /// Writes the bincode-encoded bloom filter bytes, then flushes to OS page
    /// cache, then calls `sync_all()` to force the OS to persist the data to
    /// disk. This ensures the bloom filter survives a power failure or crash.
    pub fn save(&mut self) -> Result<()> {
        if !self.dirty {
            return Ok(());
        }
        let bytes = bincode::serialize(&self.filter)
            .map_err(|e| RockDuckError::Codec(format!("bloom serialize: {}", e)))?;
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent).map_err(RockDuckError::Io)?;
        }
        let mut file = std::fs::File::create(&self.path).map_err(RockDuckError::Io)?;
        file.write_all(&bytes).map_err(RockDuckError::Io)?;
        file.flush().map_err(RockDuckError::Io)?; // flush to page cache
        file.sync_all().map_err(RockDuckError::Io)?; // force to disk
        self.dirty = false;
        Ok(())
    }
}

impl Drop for FastbloomGranuleFilter {
    fn drop(&mut self) {
        if self.dirty {
            if let Err(e) = self.save() {
                tracing::warn!("failed to persist FastbloomGranuleFilter to {:?}: {}", self.path, e);
            }
        }
    }
}

impl std::fmt::Debug for FastbloomGranuleFilter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FastbloomGranuleFilter")
            .field("seg_id", &self.seg_id)
            .field("granule_id", &self.granule_id)
            .field("fill_ratio", &self.fill_ratio())
            .field("num_bits", &self.filter.num_bits())
            .field("num_hashes", &self.filter.num_hashes())
            .field("path", &self.path)
            .finish()
    }
}

// ─── AtomicGranuleBloomFilter ─────────────────────────────────────────────────

pub struct AtomicGranuleBloomFilter {
    filter: AtomicBloomFilter<DefaultHasher>,
    #[allow(dead_code)]
    path: PathBuf,
}

impl AtomicGranuleBloomFilter {
    pub fn create(_granule_id: u32, expected_entries: u32, path: PathBuf) -> Result<Self> {
        let num_bits = optimal_bits(expected_entries as usize);
        let filter =
            AtomicBloomFilter::with_num_bits(num_bits).expected_items(expected_entries as usize);
        Ok(Self { filter, path })
    }

    pub fn insert(&self, key: &[u8]) {
        self.filter.insert(&key);
    }
    pub fn might_contain(&self, key: &[u8]) -> bool {
        self.filter.contains(&key)
    }
}
