//! L3: Frozen Patch Part — compacted delta storage.
//!
//! After L2 accumulates too many patches, they are compacted into a single
//! frozen patch part that is merged into the base column files.
//!
//! File layout:
//! ```text
//! {data_dir}/
//!   {seg_id}/
//!     l3/
//!       {col}.patch.{n}    # Compacting patch files
//! ```

use std::collections::HashMap;
use std::fs::{self, File};
use std::path::PathBuf;

use std::sync::Arc;

use parking_lot::RwLock;

use super::types::{DeltaCell, DeltaPatch, DeltaPatchFormat, ZoneMap};
use crate::error::{Result, RockDuckError};
use crate::storage::delta::sparsity::extract_value_at;

/// L3: Frozen compacted delta storage.
/// Compaction writes go here; the base column files are updated in place.
pub struct DeltaL3Frozen {
    /// Root directory for L3 files.
    root_dir: PathBuf,
    /// Compaction index: (seg_id, col) → path to frozen patch file.
    index: RwLock<HashMap<(String, String), PathBuf>>,
}

impl DeltaL3Frozen {
    /// Create a new L3 frozen store.
    pub fn new(root_dir: PathBuf) -> Self {
        let _ = fs::create_dir_all(&root_dir);
        Self {
            root_dir,
            index: RwLock::new(HashMap::new()),
        }
    }

    /// Get the path for a frozen patch file.
    fn patch_path(&self, seg_id: &str, col: &str, version: u64) -> PathBuf {
        self.root_dir
            .join(seg_id)
            .join("l3")
            .join(format!("{col}.patch.{version}"))
    }

    /// Write a compacted patch to L3.
    /// The compacted data is the result of merging patches from L2 into a single file.
    pub fn write_compacted(
        &self,
        seg_id: &str,
        col: &str,
        version: u64,
        format: DeltaPatchFormat,
        zone_map: ZoneMap,
    ) -> Result<PathBuf> {
        let path = self.patch_path(seg_id, col, version);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }

        let patch = DeltaPatch {
            seg_id: seg_id.to_string(),
            column: col.to_string(),
            patch_id: version,
            txn_range: (zone_map.min_txn, zone_map.max_txn),
            format,
            zone_map: zone_map.clone(),
        };

        let bytes = patch.format.to_bytes();

        // Atomic write
        let tmp_path = {
            let mut p = path.clone();
            p.set_extension("patch.tmp");
            p
        };
        fs::write(&tmp_path, &bytes)?;
        let f = File::open(&tmp_path)?;
        f.sync_all()?;
        drop(f);
        fs::rename(&tmp_path, &path)?;
        if let Some(parent) = path.parent() {
            let dir = File::open(parent)?;
            dir.sync_all()?;
        }

        // Update index
        self.index
            .write()
            .insert((seg_id.to_string(), col.to_string()), path.clone());

        Ok(path)
    }

    /// Get all visible deltas for a segment at a given snapshot.
    pub fn get_visible(&self, seg_id: &str, snapshot_txn: u64) -> Result<Vec<DeltaCell>> {
        let index = self.index.read();
        let columns: Vec<String> = index
            .keys()
            .filter(|(indexed_seg_id, _)| indexed_seg_id == seg_id)
            .map(|(_, col)| col.clone())
            .collect();
        drop(index);

        if columns.is_empty() {
            let seg_dir = self.root_dir.join(seg_id).join("l3");
            if !seg_dir.exists() {
                return Ok(Vec::new());
            }
            return Ok(Vec::new());
        }

        let mut deltas = Vec::new();
        for col in columns {
            let patch_deltas = self.load_patch(seg_id, &col, snapshot_txn)?;
            deltas.extend(patch_deltas);
        }

        Ok(deltas)
    }

    /// Load a single patch file.
    fn load_patch(&self, seg_id: &str, col: &str, snapshot_txn: u64) -> Result<Vec<DeltaCell>> {
        let index = self.index.read();
        let path = index.get(&(seg_id.to_string(), col.to_string())).cloned();

        let path = match path {
            Some(p) => p,
            None => {
                // Fallback: scan directory for the latest numeric patch extension.
                let seg_dir = self.root_dir.join(seg_id).join("l3");
                let mut latest: Option<PathBuf> = None;
                let mut latest_ver = 0u64;
                for entry in fs::read_dir(&seg_dir)? {
                    let entry = entry?;
                    let p = entry.path();
                    // The version number is a pure numeric extension (e.g. "5" for "price.patch.5").
                    // Fixed: was checking ext.starts_with("patch.") which always failed for "5".
                    if let Some(ext) = p.extension().and_then(|e| e.to_str()) {
                        if ext.chars().all(|c| c.is_ascii_digit()) {
                            if let Ok(ver) = ext.parse::<u64>() {
                                if ver > latest_ver {
                                    latest_ver = ver;
                                    latest = Some(p);
                                }
                            }
                        }
                    }
                }
                latest.ok_or_else(|| RockDuckError::Internal("No patch file found".into()))?
            }
        };

        if !path.exists() {
            return Ok(Vec::new());
        }

        let bytes = fs::read(&path)?;
        let format = DeltaPatchFormat::from_bytes(&bytes)
            .ok_or_else(|| RockDuckError::Internal("Invalid patch format".into()))?;

        // Check ZoneMap prune
        if let Some(zm) = self.get_zone_map_from_patch(&bytes) {
            if zm.max_txn < snapshot_txn {
                return Ok(Vec::new());
            }
        }

        patch_format_to_deltas(seg_id, col, &format)
    }

    /// Extract ZoneMap from patch bytes (without full parsing).
    fn get_zone_map_from_patch(&self, bytes: &[u8]) -> Option<ZoneMap> {
        // ZoneMap is at the end of the patch
        // For now, return None (conservative — always load the patch)
        let _ = bytes;
        None
    }

    /// Get a single cell delta.
    pub fn get_cell(
        &self,
        seg_id: &str,
        row_offset: u64,
        column: &str,
        snapshot_txn: u64,
    ) -> Result<Option<DeltaCell>> {
        let deltas = self.get_visible(seg_id, snapshot_txn)?;
        let candidates: Vec<_> = deltas
            .into_iter()
            .filter(|d| d.row_offset == row_offset && d.column == column)
            .collect();

        Ok(candidates.into_iter().max_by_key(|d| d.txn_id))
    }

    /// Get all segment IDs that have frozen patches in L3.
    pub fn get_segment_ids(&self) -> Vec<String> {
        let index = self.index.read();
        let mut seg_ids = std::collections::HashSet::new();
        for (seg_id, _) in index.keys() {
            seg_ids.insert(seg_id.clone());
        }
        seg_ids.into_iter().collect()
    }

    /// Get the number of frozen patches.
    pub fn num_frozen(&self) -> usize {
        self.index.read().len()
    }

    /// Get all patch paths for a segment.
    pub fn get_patch_paths(&self, seg_id: &str) -> Vec<PathBuf> {
        self.index
            .read()
            .iter()
            .filter(|((s, _), _)| s == seg_id)
            .map(|(_, p)| p.clone())
            .collect()
    }

    /// Delete a frozen patch file.
    pub fn delete_patch(&self, seg_id: &str, col: &str) -> Result<()> {
        let mut index = self.index.write();
        if let Some(path) = index.remove(&(seg_id.to_string(), col.to_string())) {
            if path.exists() {
                fs::remove_file(&path)?;
            }
        }
        Ok(())
    }
}

/// Convert a DeltaPatchFormat to individual DeltaCells.
///
/// For Dense patches, the Arrow IPC array is decoded and each row is converted
/// to a DeltaCell. All rows get `txn_id` from the Dense header (highest txn in patch).
///
/// Note: for Dense patches, every row in the segment is emitted as a DeltaCell.
/// This means even "unchanged" rows (null values in the IPC array) get a DeltaCell.
/// Callers should handle null `after` values appropriately.
fn patch_format_to_deltas(
    seg_id: &str,
    col: &str,
    format: &DeltaPatchFormat,
) -> Result<Vec<DeltaCell>> {
    match format {
        DeltaPatchFormat::Sparse { positions, .. } => {
            let bitmap = croaring::Bitmap::deserialize::<croaring::Portable>(&positions[..]);
            let positions: Vec<u64> = bitmap.iter().map(|p| p as u64).collect();
            Ok(positions
                .into_iter()
                .map(|row| DeltaCell {
                    seg_id: seg_id.to_string(),
                    row_offset: row,
                    column: col.to_string(),
                    txn_id: 0,
                    before: None,
                    after: None,
                    committed: true,
                    ts: 0,
                })
                .collect())
        }
        DeltaPatchFormat::Dense {
            values,
            total_rows,
            txn_id,
        } => {
            let arr = crate::storage::delta::sparsity::decode_arrow_array(values)?;
            let arr_len = arr.len() as u64;
            let seg_total = (*total_rows).max(arr_len);
            let mut cells = Vec::with_capacity(seg_total as usize);
            for row_offset in 0..seg_total {
                let after_bytes = extract_value_at(values, row_offset as usize)?;
                cells.push(DeltaCell {
                    seg_id: seg_id.to_string(),
                    row_offset,
                    column: col.to_string(),
                    txn_id: *txn_id,
                    before: None,
                    after: after_bytes.map(Arc::new),
                    committed: true,
                    ts: 0,
                });
            }
            Ok(cells)
        }
    }
}
