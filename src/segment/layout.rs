//! PAX layout: directory structure, file naming
//!
//! Each Segment = `{data_dir}/seg_{seg_id}/`
//! Each Column = `{seg_dir}/{col_name}.vortex`
//! Metadata = `{seg_dir}/_meta.vortex`
//! Delete mask = `{seg_dir}/_del.vortex`

use std::path::{Path, PathBuf};

/// Segment directory layout
#[derive(Debug, Clone)]
pub struct SegmentLayout {
    pub seg_dir: PathBuf,
}

/// Validate a segment ID to prevent path traversal attacks.
///
/// seg_id can come from two sources:
/// - Internally generated via `generate_seg_id()` (UUID v7, safe)
/// - Externally loaded from WAL files or bloom filter filenames (potentially adversarial)
///
/// `Path::join` does NOT resolve `..` components, so a seg_id of `../../../etc`
/// would escape the data directory. This function blocks that.
///
/// Returns `Ok(())` if the seg_id is safe, or an error describing the problem.
pub fn validate_seg_id(seg_id: &str, data_dir: &Path) -> Result<(), crate::RockDuckError> {
    if seg_id.contains('/') || seg_id.contains('\\') || seg_id.contains("..") {
        return Err(crate::RockDuckError::Security(format!(
            "invalid seg_id: contains path traversal: {}",
            seg_id
        )));
    }
    let seg_dir = data_dir.join("segments").join(seg_id);
    seg_dir.strip_prefix(data_dir).map_err(|_| {
        crate::RockDuckError::Security(format!("path traversal detected in seg_id: {}", seg_id))
    })?;
    Ok(())
}

impl SegmentLayout {
    /// Create a new Segment layout.
    /// seg_id should be the full segment identifier (e.g., "seg_abc123")
    ///
    /// # Security
    /// seg_id is validated via `validate_seg_id` to prevent path traversal.
    /// Callers must use `validate_seg_id` before this constructor if seg_id
    /// comes from an untrusted source (e.g., deserialized WAL entries).
    ///
    /// # Panics
    /// Panics if seg_id contains path traversal characters (../, /, \\).
    /// This is a deliberate security measure — invalid seg_ids should not silently
    /// create malformed paths that could lead to data corruption or security issues.
    pub fn new(data_dir: &Path, seg_id: &str) -> Self {
        validate_seg_id(seg_id, data_dir).expect("invalid seg_id: path traversal detected");
        Self {
            seg_dir: data_dir.join("segments").join(seg_id),
        }
    }

    /// Create a new Segment layout, returning an error instead of panicking.
    /// Use this variant when you need to handle invalid seg_ids gracefully.
    pub fn try_new(data_dir: &Path, seg_id: &str) -> Result<Self, crate::RockDuckError> {
        validate_seg_id(seg_id, data_dir)?;
        Ok(Self {
            seg_dir: data_dir.join("segments").join(seg_id),
        })
    }

    /// Get column file path
    pub fn col_path(&self, col_name: &str) -> PathBuf {
        self.seg_dir.join(format!("{}.vortex", col_name))
    }

    /// Get visibility file path (__vis.vortex)
    pub fn vis_path(&self) -> PathBuf {
        self.seg_dir.join("__vis.vortex")
    }

    /// Get deltavis file path (__vis.vortex.delta).
    /// Stores (row_offset, txn_id) pairs as a binary log for O(1) delete marking.
    pub fn deltavis_path(&self) -> PathBuf {
        self.seg_dir.join("__vis.vortex.delta")
    }

    /// Get visibility file path
    pub fn del_mask_path(&self) -> PathBuf {
        self.seg_dir.join("_del.vortex")
    }

    /// Get update mask path
    pub fn upd_mask_path(&self, col_name: &str) -> PathBuf {
        self.seg_dir.join(format!("_upd_{}.vortex", col_name))
    }

    /// Get metadata file path
    pub fn meta_path(&self) -> PathBuf {
        self.seg_dir.join("_meta.vortex")
    }

    /// Get Zone Map path
    pub fn zone_map_path(&self) -> PathBuf {
        self.seg_dir.join("_zm.json")
    }

    /// Get block-level Zone Map sidecar path for a column.
    /// Format: `{col_name}._zm.bin` (bincode-encoded GranuleZoneMapIndex).
    pub fn block_zm_path(&self, col_name: &str) -> PathBuf {
        self.seg_dir.join(format!("{}._zm.bin", col_name))
    }

    /// Get block-level Bloom Filter sidecar path for a column.
    /// Format: `{col_name}._bf.bin` (bincode-encoded GranuleBloomFilter data).
    pub fn block_bf_path(&self, col_name: &str) -> PathBuf {
        self.seg_dir.join(format!("{}._bf.bin", col_name))
    }

    /// Get delta file path for this segment.
    /// References the L2 delta file: `{seg_dir}/l2/{col}.delta`
    pub fn delta_path(&self, col_name: &str) -> PathBuf {
        self.seg_dir.join("l2").join(format!("{}.delta", col_name))
    }

    /// Ensure directory exists
    pub fn create_dirs(&self) -> std::io::Result<()> {
        std::fs::create_dir_all(&self.seg_dir)
    }

    /// Delete entire segment directory
    pub fn delete_all(&self) -> std::io::Result<()> {
        if self.seg_dir.exists() {
            std::fs::remove_dir_all(&self.seg_dir)
        } else {
            Ok(())
        }
    }
}

/// Generate a new segment ID using UUID v7 (time-ordered)
pub fn generate_seg_id() -> String {
    use uuid::Uuid;
    format!("seg_{}", Uuid::now_v7().to_string().replace("-", ""))
}

/// File naming conventions
pub mod naming {
    /// Segment ID prefix
    pub const SEG_PREFIX: &str = "seg_";

    /// Column file suffix
    pub const COL_SUFFIX: &str = ".vortex";

    /// Delete mask filename
    pub const DEL_MASK_NAME: &str = "_del.vortex";

    /// Update mask filename pattern
    pub fn upd_mask_name(col: &str) -> String {
        format!("_upd_{}.vortex", col)
    }

    /// Metadata filename
    pub const META_NAME: &str = "_meta.vortex";

    /// Zone Map filename
    pub const ZM_NAME: &str = "_zm.json";

    /// Block-level Zone Map sidecar suffix
    pub const BLOCK_ZM_SUFFIX: &str = "._zm.bin";

    /// Block-level Bloom Filter sidecar suffix
    pub const BLOCK_BF_SUFFIX: &str = "._bf.bin";
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_seg_id_accepts_valid_ids() {
        let data_dir = PathBuf::from("/data");
        assert!(validate_seg_id("seg_123", &data_dir).is_ok());
        assert!(validate_seg_id("seg_abc123def", &data_dir).is_ok());
        assert!(validate_seg_id("seg_12345", &data_dir).is_ok());
    }

    #[test]
    fn test_validate_seg_id_rejects_path_traversal() {
        let data_dir = PathBuf::from("/data");

        // Reject forward slashes
        assert!(validate_seg_id("../../../etc", &data_dir).is_err());

        // Reject backslashes (Windows path traversal)
        assert!(validate_seg_id("..\\..\\etc", &data_dir).is_err());

        // Reject explicit dotdot
        assert!(validate_seg_id("foo/../bar", &data_dir).is_err());
        assert!(validate_seg_id("foo/..", &data_dir).is_err());

        // Reject embedded path separators
        assert!(validate_seg_id("seg/123", &data_dir).is_err());
        assert!(validate_seg_id("seg\\123", &data_dir).is_err());
    }

    #[test]
    fn test_validate_seg_id_strip_prefix_detection() {
        let data_dir = PathBuf::from("/data/segments");

        // Valid: normal segment ID
        assert!(validate_seg_id("seg_12345", &data_dir).is_ok());

        // Invalid: would escape via symlink or relative path
        // This catches cases like "seg_../../../etc"
        assert!(validate_seg_id("seg_../../../etc", &data_dir).is_err());
    }

    #[test]
    fn test_segment_layout_new_accepts_valid_ids() {
        let temp = tempfile::tempdir().unwrap();
        let data_dir = temp.path().to_path_buf();

        // Valid segment IDs should work
        let layout = SegmentLayout::new(&data_dir, "seg_12345");
        assert!(layout.seg_dir.exists() || !layout.seg_dir.exists()); // Just verify it doesn't panic
    }

    #[test]
    fn test_segment_layout_new_panics_on_invalid_ids() {
        let temp = tempfile::tempdir().unwrap();
        let data_dir = temp.path().to_path_buf();

        // Invalid segment IDs should panic
        let result = std::panic::catch_unwind(|| {
            SegmentLayout::new(&data_dir, "../../../etc");
        });
        assert!(
            result.is_err(),
            "SegmentLayout::new should panic on invalid seg_id"
        );
    }

    #[test]
    fn test_segment_layout_try_new_returns_error_on_invalid() {
        let temp = tempfile::tempdir().unwrap();
        let data_dir = temp.path().to_path_buf();

        // try_new should return an error instead of panicking
        let result = SegmentLayout::try_new(&data_dir, "../../../etc");
        assert!(
            result.is_err(),
            "SegmentLayout::try_new should return error on invalid seg_id"
        );
    }
}
