//! Iceberg export orchestrator.
//!
//! On-demand export of RockDuck data to a spec-compliant Iceberg v2 table.
//!
//! Usage:
//!   let metadata_path = rockduck.export_iceberg("users", "/path/to/iceberg_table").await?;
//!   // Now DuckDB can read it:
//!   //   INSTALL vortex; LOAD vortex;
//!   //   SELECT * FROM read_vortex('/path/to/iceberg_table/segments/*/*.vortex');
//!
//! Directory layout produced:
//!   target/
//!   ├── version-hint.txt              ("2" for Iceberg v2)
//!   ├── metadata/
//!   │   ├── v{snapshot_id}.metadata.json
//!   │   └── snap-{id}-{seq}-{uuid8}.avro  (manifest-list)
//!   └── data/
//!       └── segments/
//!           └── {seg_id}/
//!               └── {col}.vortex

use std::path::{Path, PathBuf};
use crate::db::RockDuck;
use crate::iceberg::avro_writer::{self, ManifestFileInfo};
use crate::iceberg::catalog;
use crate::iceberg::translate;
use crate::iceberg::IcebergExport;
use crate::error::{RockDuckError, Result};
use crate::metadata::seg_meta;
use crate::segment::meta::SegmentStatus;

/// Export the current RockDuck table state as an Iceberg v2 table.
///
/// Writes spec-compliant Iceberg v2 artifacts to `target_dir`:
///   - `metadata/v{N}.metadata.json` — TableMetadata JSON
///   - `metadata/snap-{id}-{seq}-{uuid8}.avro` — manifest-list
///   - `data/segments/{seg_id}/{col}.vortex` — copied data files
///   - `version-hint.txt` — "2"
///
/// Returns the path to the written metadata JSON file.
pub async fn export_to_iceberg(
    rockduck: &RockDuck,
    table: &str,
    target_dir: impl AsRef<Path>,
    snapshot_id: Option<i64>,
) -> Result<PathBuf> {
    let target = target_dir.as_ref();
    let data_dir = &rockduck.data_dir;

    // 1. Collect all Frozen segments for this table
    let seg_ids = seg_meta::list_table_segments(&rockduck.db, table)?;
    let mut frozen_segments: Vec<_> = Vec::new();

    for seg_id in &seg_ids {
        if let Some(meta) = seg_meta::get_segment(&rockduck.db, seg_id)? {
            if meta.status == SegmentStatus::Frozen {
                frozen_segments.push(meta);
            }
        }
    }

    if frozen_segments.is_empty() {
        return Err(RockDuckError::Metadata(
            "No frozen segments found for export; freeze segments first".to_string()
        ).into());
    }

    // 2. Build column -> field_id map
    let columns = frozen_segments.first().map(|m| m.columns.clone())
        .ok_or_else(|| RockDuckError::Metadata("No columns found".to_string()))?;

    let field_id_map: std::collections::HashMap<String, i32> = columns
        .iter()
        .enumerate()
        .map(|(i, col)| (col.name.clone(), (i as i32) + 1))
        .collect();

    // 3. Build data file entries from segments + granules
    let entries = translate::build_data_file_entries(frozen_segments.iter(), &field_id_map, data_dir);

    // 4. Load or create native manifest
    let mut manifest = catalog::load_manifest(&rockduck.db)?
        .unwrap_or_else(|| IcebergExport::new(table.to_string(), snapshot_id.unwrap_or(1)));

    // 5. Update snapshot
    let new_snapshot_id = snapshot_id.unwrap_or(manifest.snapshot_id + 1);
    manifest.advance_snapshot(new_snapshot_id);
    manifest.table_name = table.to_string();
    manifest.entries = entries;

    // 6. Create directory structure
    let metadata_dir = target.join("metadata");
    let data_dir_target = target.join("data");
    std::fs::create_dir_all(&metadata_dir)?;
    std::fs::create_dir_all(&data_dir_target)?;

    // 7. Copy Vortex data files to data/
    copy_vortex_files(&frozen_segments, data_dir, &data_dir_target)?;

    // 8. Build Iceberg schema and sort order JSON
    let schema_json = translate::to_iceberg_schema(&columns);
    let sort_order_json = translate::to_sort_order(&columns);

    // 9. Write manifest Avro (contains data-file entries)
    let manifest_uuid = uuid::Uuid::new_v4().to_string().replace("-", "");
    let manifest_path = metadata_dir.join(format!("{}-m0.avro", manifest_uuid));
    let manifest_info = avro_writer::write_manifest_avro_sync(&manifest, &manifest_path)?;

    // 10. Write manifest-list Avro
    let manifest_list_path = metadata_dir.join(format!(
        "snap-{}-{}-{}.avro",
        manifest.snapshot_id,
        manifest.sequence_number,
        &manifest_uuid[..8]
    ));

    let manifest_list_infos = vec![ManifestFileInfo {
        manifest_path: manifest_path.to_string_lossy().to_string(),
        manifest_length: manifest_info.manifest_length,
        added_files_count: manifest_info.added_files_count,
        existing_files_count: 0,
        deleted_files_count: 0,
        added_rows_count: manifest_info.added_rows_count,
        existing_rows_count: 0,
        deleted_rows_count: 0,
    }];

    avro_writer::write_manifest_list_avro_sync(
        manifest.snapshot_id,
        manifest.sequence_number,
        manifest.parent_snapshot_id,
        manifest_list_infos,
        &manifest_list_path,
    )?;

    // 11. Write TableMetadata JSON
    let metadata_json = translate::to_iceberg_table_metadata(
        &manifest,
        &target.to_string_lossy(),
        &schema_json,
        &sort_order_json,
    );

    let metadata_file_name = format!("v{}.metadata.json", manifest.snapshot_id);
    let metadata_path = metadata_dir.join(&metadata_file_name);
    let json_str = serde_json::to_string_pretty(&metadata_json)?;
    std::fs::write(&metadata_path, json_str)?;

    // 12. Write version-hint.txt
    std::fs::write(target.join("version-hint.txt"), "2")?;
    std::fs::write(metadata_dir.join("version-hint.txt"), "2")?;

    // 13. Sync all written files
    sync_file(&metadata_path)?;
    sync_file(&manifest_list_path)?;
    sync_file(&manifest_path)?;
    sync_dir(&metadata_dir)?;
    sync_dir(target)?;

    // 14. Save native manifest back to RocksDB
    catalog::save_manifest(&rockduck.db, &manifest)?;

    tracing::info!(
        "Iceberg export complete: {} entries, snapshot_id={}, metadata={}",
        manifest.entries.len(),
        manifest.snapshot_id,
        metadata_path.display()
    );

    Ok(metadata_path)
}

/// Copy Vortex segment files to the export data directory.
fn copy_vortex_files(
    segments: &[crate::metadata::SegmentMeta],
    src_data_dir: &Path,
    dst_data_dir: &Path,
) -> Result<()> {
    for seg in segments {
        let src_seg_dir = src_data_dir.join("segments").join(&seg.seg_id);
        let dst_seg_dir = dst_data_dir.join("segments").join(&seg.seg_id);
        std::fs::create_dir_all(&dst_seg_dir)?;

        for col in &seg.columns {
            let src_file = src_seg_dir.join(format!("{}.vortex", col.name));
            if src_file.exists() {
                let dst_file = dst_seg_dir.join(format!("{}.vortex", col.name));
                std::fs::copy(&src_file, &dst_file)?;
            }
        }
    }
    Ok(())
}

/// Sync a file to disk.
fn sync_file(path: &Path) -> std::io::Result<()> {
    #[cfg(windows)]
    {
        use std::os::windows::io::AsRawHandle;
        let file = std::fs::OpenOptions::new().read(true).open(path)?;
        let handle = file.as_raw_handle();
        let ok = unsafe {
            windows_sys::Win32::Storage::FileSystem::FlushFileBuffers(handle as *mut _)
        };
        if ok == 0 {
            return Err(std::io::Error::last_os_error());
        }
        Ok(())
    }
    #[cfg(not(windows))]
    {
        let file = std::fs::OpenOptions::new().read(true).open(path)?;
        file.sync_all()
    }
}

/// Sync a directory to disk.
fn sync_dir(path: &Path) -> std::io::Result<()> {
    #[cfg(windows)]
    {
        use std::os::windows::io::AsRawHandle;
        let dir = std::fs::OpenOptions::new().read(true).open(path)?;
        let handle = dir.as_raw_handle();
        let ok = unsafe {
            windows_sys::Win32::Storage::FileSystem::FlushFileBuffers(handle as *mut _)
        };
        if ok == 0 {
            return Err(std::io::Error::last_os_error());
        }
        Ok(())
    }
    #[cfg(not(windows))]
    {
        let dir = std::fs::OpenOptions::new().read(true).open(path)?;
        dir.sync_all()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sync_file_nonexistent_returns_error() {
        let result = sync_file(std::path::Path::new("/this/file/does/not/exist/anywhere"));
        assert!(result.is_err(), "sync_file on non-existent path should return an error, not panic");
    }

    #[test]
    fn test_sync_file_on_nested_nonexistent_file() {
        // sync_file returns error when file does not exist, even if parent dir exists
        let tmp = tempfile::TempDir::new().unwrap();
        let nested = tmp.path().join("a").join("b").join("file.txt");
        std::fs::create_dir_all(nested.parent().unwrap()).unwrap();
        let result = sync_file(&nested);
        assert!(result.is_err(), "sync_file should fail when file does not exist");
    }
}
