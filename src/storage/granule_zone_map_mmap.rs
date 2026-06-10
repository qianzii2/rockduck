//! Mmapped GranuleZoneMap - zero-copy zone map access via mmap.
//!
//! Uses memmap2 for cross-platform memory mapping and postcard for serialization.
//! The zone map data is small enough that full deserialization is fast.

use std::path::Path;
use std::sync::Arc;

use crate::error::{Result, RockDuckError};
use crate::codec::serialize::{MAGIC_ZONE, extract_payload_from_mmap};
use crate::codec::{encode, decode};
use crate::metadata::GranuleId;
use crate::storage::MmapReader;

const MAX_INLINE_BYTES: usize = 32;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct InlineBytes {
    pub len: u8,
    data: [u8; MAX_INLINE_BYTES],
}

impl InlineBytes {
    pub fn new() -> Self { Self { len: 0, data: [0u8; MAX_INLINE_BYTES] } }
    pub fn from_slice(slice: &[u8]) -> Self {
        let len = slice.len().min(MAX_INLINE_BYTES) as u8;
        let mut data = [0u8; MAX_INLINE_BYTES];
        data[..len as usize].copy_from_slice(&slice[..len as usize]);
        Self { len, data }
    }
    pub fn as_slice(&self) -> &[u8] { &self.data[..self.len as usize] }
    pub fn is_empty(&self) -> bool { self.len == 0 }
    pub fn len(&self) -> usize { self.len as usize }
}

impl Default for InlineBytes { fn default() -> Self { Self::new() } }

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct InlineGranuleStats {
    pub granule_id: GranuleId,
    pub block_size: u32,
    pub row_count: u32,
    pub null_count: u64,
    pub min_value: InlineBytes,
    pub max_value: InlineBytes,
    pub has_min: bool,
    pub has_max: bool,
}

impl InlineGranuleStats {
    pub fn new(granule_id: GranuleId, block_size: u32) -> Self {
        Self { granule_id, block_size, row_count: 0, null_count: 0,
            min_value: InlineBytes::new(), max_value: InlineBytes::new(),
            has_min: false, has_max: false }
    }
    pub fn update_min(&mut self, value: &[u8]) {
        if !self.has_min || value < self.min_value.as_slice() {
            self.min_value = InlineBytes::from_slice(value);
            self.has_min = true;
        }
    }
    pub fn update_max(&mut self, value: &[u8]) {
        if !self.has_max || value > self.max_value.as_slice() {
            self.max_value = InlineBytes::from_slice(value);
            self.has_max = true;
        }
    }
    pub fn update_null_count(&mut self, count: u64) { self.null_count += count; }
    pub fn min_value(&self) -> Option<&[u8]> {
        if self.has_min { Some(self.min_value.as_slice()) } else { None }
    }
    pub fn max_value(&self) -> Option<&[u8]> {
        if self.has_max { Some(self.max_value.as_slice()) } else { None }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct GranuleZoneMapData {
    pub column_name: String,
    pub num_granules: u32,
    pub block_size: u32,
    pub granules: Vec<InlineGranuleStats>,
}

impl GranuleZoneMapData {
    pub fn total_rows(&self) -> u64 { self.granules.iter().map(|g| g.row_count as u64).sum() }
    pub fn num_granules(&self) -> usize { self.granules.len() }

    pub fn may_overlap(&self, pred_min: &[u8], pred_max: &[u8]) -> bool {
        for g in &self.granules {
            if g.has_max {
                let gmax = g.max_value.as_slice();
                if !gmax.is_empty() && gmax < pred_min { continue; }
            }
            if g.has_min {
                let gmin = g.min_value.as_slice();
                if !gmin.is_empty() && gmin > pred_max { continue; }
            }
            return true;
        }
        false
    }

    pub fn granule_may_overlap(&self, granule_idx: usize, pred_min: &[u8], pred_max: &[u8]) -> bool {
        let g = match self.granules.get(granule_idx) {
            Some(g) => g,
            None => return true,
        };
        if g.has_max {
            let gmax = g.max_value.as_slice();
            if !gmax.is_empty() && gmax < pred_min { return false; }
        }
        if g.has_min {
            let gmin = g.min_value.as_slice();
            if !gmin.is_empty() && gmin > pred_max { return false; }
        }
        true
    }
}

pub struct MmappedZoneMap {
    reader: Arc<MmapReader>,
    data: GranuleZoneMapData,
}

impl MmappedZoneMap {
    pub fn open(path: &Path) -> Result<Self> {
        let reader = MmapReader::open(path)?;
        let file_len = reader.len() as u64;
        let payload = extract_payload_from_mmap(reader.slice_at(0, file_len)?, MAGIC_ZONE)
            .map_err(|e| RockDuckError::Codec(e.to_string()))?;

        let data: GranuleZoneMapData = decode(payload)
            .map_err(|e| RockDuckError::Codec(e.to_string()))?;

        Ok(Self { reader: Arc::new(reader), data })
    }

    pub fn may_overlap(&self, pred_min: &[u8], pred_max: &[u8]) -> bool {
        self.data.may_overlap(pred_min, pred_max)
    }

    pub fn granule_may_overlap(&self, granule_idx: usize, pred_min: &[u8], pred_max: &[u8]) -> bool {
        self.data.granule_may_overlap(granule_idx, pred_min, pred_max)
    }

    pub fn num_granules(&self) -> usize { self.data.num_granules() }
    pub fn granule_min(&self, granule_idx: usize) -> Option<&[u8]> {
        self.data.granules.get(granule_idx).and_then(|g| g.min_value())
    }
    pub fn granule_max(&self, granule_idx: usize) -> Option<&[u8]> {
        self.data.granules.get(granule_idx).and_then(|g| g.max_value())
    }
    pub fn granule_null_count(&self, granule_idx: usize) -> u64 {
        self.data.granules.get(granule_idx).map(|g| g.null_count).unwrap_or(0)
    }
    pub fn granule_row_count(&self, granule_idx: usize) -> u32 {
        self.data.granules.get(granule_idx).map(|g| g.row_count).unwrap_or(0)
    }
    pub fn reader(&self) -> &Arc<MmapReader> { &self.reader }
}

pub struct GranuleZoneMapBuilder {
    column_name: String,
    block_size: u32,
    granules: Vec<InlineGranuleStats>,
}

impl GranuleZoneMapBuilder {
    pub fn new(column_name: String, block_size: u32) -> Self {
        Self { column_name, block_size, granules: Vec::new() }
    }
    pub fn add_granule(&mut self, stats: InlineGranuleStats) { self.granules.push(stats); }
    pub fn finalize(mut self) -> GranuleZoneMapData {
        GranuleZoneMapData {
            column_name: self.column_name,
            num_granules: self.granules.len() as u32,
            block_size: self.block_size,
            granules: self.granules,
        }
    }
}

impl From<crate::metadata::zone_map::ZoneMapStats> for GranuleZoneMapData {
    fn from(stats: crate::metadata::zone_map::ZoneMapStats) -> Self {
        let granules: Vec<_> = stats.columns.iter().enumerate().map(|(i, col)| {
            let mut g = InlineGranuleStats::new(GranuleId::new(i as u32), 8192);
            if let Some(ref min) = col.min_value { g.update_min(min); }
            if let Some(ref max) = col.max_value { g.update_max(max); }
            g.update_null_count(col.null_count as u64);
            g
        }).collect();
        Self { column_name: "unknown".to_string(), num_granules: granules.len() as u32,
            block_size: 8192, granules }
    }
}

