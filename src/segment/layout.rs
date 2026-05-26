//! PAX 布局：目录结构、文件命名
//!
//! 每个 Segment = `{data_dir}/seg_{seg_id}/`
//! 每个 Column = `{seg_dir}/{col_name}.vortex`
//! 元数据 = `{seg_dir}/_meta.vortex`
//! 删除掩码 = `{seg_dir}/_del.vortex`

use std::path::{Path, PathBuf};
use uuid::Uuid;

/// Segment 目录布局
#[derive(Debug, Clone)]
pub struct SegmentLayout {
    pub seg_dir: PathBuf,
}

impl SegmentLayout {
    /// 创建新的 Segment 布局
    /// seg_id should be the full segment identifier (e.g., "seg_abc123")
    pub fn new(data_dir: &Path, seg_id: &str) -> Self {
        Self {
            seg_dir: data_dir.join("segments").join(seg_id),
        }
    }

    /// 获取列文件路径
    pub fn col_path(&self, col_name: &str) -> PathBuf {
        self.seg_dir.join(format!("{}.vortex", col_name))
    }

    /// 获取删除掩码路径
    pub fn del_mask_path(&self) -> PathBuf {
        self.seg_dir.join("_del.vortex")
    }

    /// 获取更新掩码路径
    pub fn upd_mask_path(&self, col_name: &str) -> PathBuf {
        self.seg_dir.join(format!("_upd_{}.vortex", col_name))
    }

    /// 获取元数据文件路径
    pub fn meta_path(&self) -> PathBuf {
        self.seg_dir.join("_meta.vortex")
    }

    /// 获取 Zone Map 路径
    pub fn zone_map_path(&self) -> PathBuf {
        self.seg_dir.join("_zm.json")
    }

    /// 确保目录存在
    pub fn create_dirs(&self) -> std::io::Result<()> {
        std::fs::create_dir_all(&self.seg_dir)
    }

    /// 删除整个 segment 目录
    pub fn delete_all(&self) -> std::io::Result<()> {
        if self.seg_dir.exists() {
            std::fs::remove_dir_all(&self.seg_dir)
        } else {
            Ok(())
        }
    }
}

/// 生成新的 segment ID
pub fn generate_seg_id() -> String {
    format!("seg_{}", Uuid::new_v4().to_string().replace("-", ""))
}

/// 文件命名规范
pub mod naming {
    /// Segment ID 前缀
    pub const SEG_PREFIX: &str = "seg_";

    /// 列文件后缀
    pub const COL_SUFFIX: &str = ".vortex";

    /// 删除掩码文件名
    pub const DEL_MASK_NAME: &str = "_del.vortex";

    /// 更新掩码文件名模式
    pub fn upd_mask_name(col: &str) -> String {
        format!("_upd_{}.vortex", col)
    }

    /// 元数据文件名
    pub const META_NAME: &str = "_meta.vortex";

    /// Zone Map 文件名
    pub const ZM_NAME: &str = "_zm.json";
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_segment_layout_new_constructs_paths() {
        let layout = SegmentLayout::new(std::path::Path::new("/data"), "seg_001");
        assert_eq!(layout.seg_dir, std::path::Path::new("/data/segments/seg_001"));
    }

    #[test]
    fn test_segment_layout_col_path() {
        let layout = SegmentLayout::new(std::path::Path::new("/data"), "seg_001");
        assert_eq!(layout.col_path("age"), std::path::Path::new("/data/segments/seg_001/age.vortex"));
        assert_eq!(layout.col_path("name"), std::path::Path::new("/data/segments/seg_001/name.vortex"));
    }

    #[test]
    fn test_segment_layout_meta_path() {
        let layout = SegmentLayout::new(std::path::Path::new("/data"), "seg_001");
        assert_eq!(layout.meta_path(), std::path::Path::new("/data/segments/seg_001/_meta.vortex"));
    }

    #[test]
    fn test_segment_layout_del_mask_path() {
        let layout = SegmentLayout::new(std::path::Path::new("/data"), "seg_001");
        assert_eq!(layout.del_mask_path(), std::path::Path::new("/data/segments/seg_001/_del.vortex"));
    }

    #[test]
    fn test_segment_layout_upd_mask_path() {
        let layout = SegmentLayout::new(std::path::Path::new("/data"), "seg_001");
        assert_eq!(layout.upd_mask_path("age"), std::path::Path::new("/data/segments/seg_001/_upd_age.vortex"));
        assert_eq!(layout.upd_mask_path("name"), std::path::Path::new("/data/segments/seg_001/_upd_name.vortex"));
    }

    #[test]
    fn test_segment_layout_zone_map_path() {
        let layout = SegmentLayout::new(std::path::Path::new("/data"), "seg_001");
        assert_eq!(layout.zone_map_path(), std::path::Path::new("/data/segments/seg_001/_zm.json"));
    }

    #[test]
    fn test_segment_layout_nested_seg_id() {
        let layout = SegmentLayout::new(std::path::Path::new("/data"), "users/seg_abc123");
        assert_eq!(layout.seg_dir, std::path::Path::new("/data/segments/users/seg_abc123"));
    }

    #[test]
    fn test_segment_layout_create_and_delete_dirs() -> std::io::Result<()> {
        let temp = tempfile::tempdir()?;
        let layout = SegmentLayout::new(temp.path(), "test_seg");
        layout.create_dirs()?;
        assert!(layout.seg_dir.exists());
        assert!(layout.seg_dir.is_dir());
        layout.delete_all()?;
        assert!(!layout.seg_dir.exists());
        Ok(())
    }

    #[test]
    fn test_segment_layout_delete_all_nonexistent() {
        let temp = tempfile::tempdir().unwrap();
        let layout = SegmentLayout::new(temp.path(), "nonexistent_seg");
        let result = layout.delete_all();
        assert!(result.is_ok());
    }

    #[test]
    fn test_generate_seg_id_format() {
        let id = generate_seg_id();
        assert!(id.starts_with("seg_"));
        assert!(id.len() > 4);
        assert!(!id.contains("-"), "UUID hyphens should be removed");
    }

    #[test]
    fn test_generate_seg_id_uniqueness() {
        let id1 = generate_seg_id();
        let id2 = generate_seg_id();
        assert_ne!(id1, id2, "Generated segment IDs should be unique");
    }

    #[test]
    fn test_naming_seg_prefix() {
        assert_eq!(naming::SEG_PREFIX, "seg_");
    }

    #[test]
    fn test_naming_col_suffix() {
        assert_eq!(naming::COL_SUFFIX, ".vortex");
    }

    #[test]
    fn test_naming_del_mask_name() {
        assert_eq!(naming::DEL_MASK_NAME, "_del.vortex");
    }

    #[test]
    fn test_naming_upd_mask_name() {
        assert_eq!(naming::upd_mask_name("age"), "_upd_age.vortex");
        assert_eq!(naming::upd_mask_name("x"), "_upd_x.vortex");
    }

    #[test]
    fn test_naming_meta_name() {
        assert_eq!(naming::META_NAME, "_meta.vortex");
    }

    #[test]
    fn test_naming_zm_name() {
        assert_eq!(naming::ZM_NAME, "_zm.json");
    }
}
