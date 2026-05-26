//! Native Iceberg manifest storage in RocksDB.
//!
//! Stores the native `IcebergExport` as bincode in RocksDB column family
//! `iceberg_manifest`. This is the hot path — written on every compaction commit.
//!
//! Key layout:
//!   `iceberg:latest`     → latest IcebergExport
//!   `iceberg:history`    → Vec<SnapshotRef> (append-only log of snapshots)
//!
//! Recovery: on DB open, load the latest manifest from RocksDB.

use rocksdb::DB;
use crate::iceberg::IcebergExport;
use crate::error::{RockDuckError, Result};
use crate::codec::{encode, decode};

/// Column family name for Iceberg manifest storage.
pub const CF_ICEBERG: &str = crate::metadata::rocksdb::CF_ICEBERG;

/// Key for the latest snapshot manifest.
const KEY_LATEST: &[u8] = b"iceberg:latest";

/// Key prefix for snapshot history.
const KEY_HISTORY_PREFIX: &[u8] = b"iceberg:history";

/// Append this CF name to the global list of column families.
pub fn cf_name() -> &'static str {
    CF_ICEBERG
}

/// Add the iceberg column family to the list of all CFs.
pub fn register_cf(all_cfs: &mut Vec<&'static str>) {
    if !all_cfs.contains(&CF_ICEBERG) {
        all_cfs.push(CF_ICEBERG);
    }
}

/// Save the Iceberg export to RocksDB (bincode-serialised).
///
/// Called after every compaction commit, on the hot path.
pub fn save_manifest(db: &DB, manifest: &IcebergExport) -> Result<()> {
    let cf = db.cf_handle(CF_ICEBERG)
        .ok_or_else(|| RockDuckError::Metadata(format!(
            "Column family '{}' not found; call register_cf() first", CF_ICEBERG
        )))?;

    let value = encode(manifest)?;
    db.put_cf(&cf, KEY_LATEST, &value)?;
    Ok(())
}

/// Load the latest Iceberg export from RocksDB.
///
/// Returns `None` if no export has been performed yet.
pub fn load_manifest(db: &DB) -> Result<Option<IcebergExport>> {
    let cf = db.cf_handle(CF_ICEBERG)
        .ok_or_else(|| RockDuckError::Metadata(format!(
            "Column family '{}' not found; call register_cf() first", CF_ICEBERG
        )))?;

    match db.get_cf(&cf, KEY_LATEST)? {
        Some(value) => {
            let manifest: IcebergExport = decode(&value)?;
            Ok(Some(manifest))
        }
        None => Ok(None),
    }
}

/// Append a snapshot entry to the history log (append-only).
pub fn append_history(db: &DB, ref_: SnapshotRef) -> Result<()> {
    let cf = db.cf_handle(CF_ICEBERG)
        .ok_or_else(|| RockDuckError::Metadata(format!(
            "Column family '{}' not found", CF_ICEBERG
        )))?;

    // Load existing history
    let existing: Vec<SnapshotRef> = match db.get_cf(&cf, KEY_HISTORY_PREFIX)? {
        Some(data) => decode(&data).unwrap_or_default(),
        None => Vec::new(),
    };

    let mut history = existing;
    history.push(ref_);

    let value = encode(&history)?;
    db.put_cf(&cf, KEY_HISTORY_PREFIX, &value)?;
    Ok(())
}

/// Load the full snapshot history.
pub fn load_history(db: &DB) -> Result<Vec<SnapshotRef>> {
    let cf = db.cf_handle(CF_ICEBERG)
        .ok_or_else(|| RockDuckError::Metadata(format!(
            "Column family '{}' not found", CF_ICEBERG
        )))?;

    match db.get_cf(&cf, KEY_HISTORY_PREFIX)? {
        Some(data) => {
            let history: Vec<SnapshotRef> = decode(&data)?;
            Ok(history)
        }
        None => Ok(Vec::new()),
    }
}

/// A named reference to a snapshot (Iceberg "refs" table).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, bincode_next::Encode, bincode_next::Decode)]
pub struct SnapshotRef {
    pub name: String,
    pub snapshot_id: i64,
    pub type_: String, // "branch" or "tag"
    pub min_snapshots_to_keep: Option<i32>,
    pub max_snapshot_age_ms: Option<i64>,
    pub max_ref_age_ms: Option<i64>,
}

impl SnapshotRef {
    /// Create a "main" branch pointing to a snapshot.
    pub fn main_branch(snapshot_id: i64) -> Self {
        Self {
            name: "main".to_string(),
            snapshot_id,
            type_: "branch".to_string(),
            min_snapshots_to_keep: None,
            max_snapshot_age_ms: None,
            max_ref_age_ms: None,
        }
    }
}

/// Update the native Iceberg manifest when a segment is frozen.
///
/// Reads the frozen segment, creates a DataFileEntry for each column file,
/// and appends it to the current Iceberg manifest in RocksDB.
pub fn update_iceberg_manifest_on_freeze(db: &DB, seg_id: &str) -> Result<Option<i64>> {
    use crate::metadata::rocksdb::get_segment_meta;
    use crate::segment::meta::SegmentStatus;

    let meta = match get_segment_meta(db, seg_id)? {
        Some(m) if m.status == SegmentStatus::Frozen => m,
        _ => return Ok(None),
    };

    let data_dir = db.path()
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_default();

    let field_id_map: std::collections::HashMap<String, i32> = meta.columns
        .iter()
        .enumerate()
        .map(|(i, col)| (col.name.clone(), (i as i32) + 1))
        .collect();

    let entries = crate::iceberg::translate::build_data_file_entries(
        std::iter::once(&meta),
        &field_id_map,
        &data_dir,
    );

    // Load or create manifest
    let mut manifest = load_manifest(db)?
        .unwrap_or_else(|| IcebergExport::new(meta.table.clone(), 1));

    manifest.advance_snapshot(manifest.snapshot_id + 1);
    for entry in entries {
        manifest.add_entry(entry);
    }

    save_manifest(db, &manifest)?;
    Ok(Some(manifest.snapshot_id))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_snapshot_ref_main_branch() {
        let ref_ = SnapshotRef::main_branch(12345);
        assert_eq!(ref_.name, "main");
        assert_eq!(ref_.snapshot_id, 12345);
        assert_eq!(ref_.type_, "branch");
        assert!(ref_.min_snapshots_to_keep.is_none());
    }

    #[test]
    fn test_snapshot_ref_encode_decode() {
        let ref_ = SnapshotRef::main_branch(999);
        let data = encode(&ref_).unwrap();
        let decoded: SnapshotRef = decode(&data).unwrap();
        assert_eq!(decoded.name, "main");
        assert_eq!(decoded.snapshot_id, 999);
    }

    // ===================== Save/Load round-trip tests =====================

    fn open_test_db(tmp: &std::path::Path) -> rocksdb::DB {
        let db_path = tmp.join("meta");
        std::fs::create_dir_all(&db_path).unwrap();
        let mut opts = rocksdb::Options::default();
        opts.create_if_missing(true);
        opts.create_missing_column_families(true);

        let cf_names: Vec<&str> = vec![
            "default",
            "pk_idx",
            "pk_skiplist",
            "seg_meta",
            "stat",
            "zone",
            "layer",
            "lbf",
            "bf",
            "proj_meta",
            "mvcc",
            "sys",
            "iceberg_manifest",
        ];

        rocksdb::DB::open_cf(&opts, &db_path, &cf_names).unwrap()
    }

    #[test]
    fn test_save_and_load_manifest() {
        use crate::iceberg::{DataFileEntry, IcebergExport};
        use std::collections::HashMap;

        let tmp = tempfile::TempDir::new().unwrap();
        let db = open_test_db(tmp.path());

        let mut manifest = IcebergExport::new("users".to_string(), 1);
        let mut entry = DataFileEntry::new("segments/seg1/id.vortex".to_string(), 1000, 8192);
        let mut null_counts = HashMap::new();
        null_counts.insert(1, 5u64);
        entry.null_counts = null_counts;
        entry.split_offsets = vec![0, 4096];
        manifest.add_entry(entry);

        save_manifest(&db, &manifest).unwrap();
        let loaded = load_manifest(&db).unwrap().unwrap();

        assert_eq!(loaded.table_name, "users");
        assert_eq!(loaded.snapshot_id, 1);
        assert_eq!(loaded.entries.len(), 1);
        assert_eq!(loaded.entries[0].file_path, "segments/seg1/id.vortex");
        assert_eq!(loaded.entries[0].record_count, 1000);
        assert_eq!(loaded.entries[0].file_size, 8192);
        assert_eq!(loaded.entries[0].null_counts.get(&1), Some(&5u64));
        assert_eq!(loaded.entries[0].split_offsets, &[0, 4096]);
    }

    #[test]
    fn test_load_manifest_nonexistent() {
        let tmp = tempfile::TempDir::new().unwrap();
        let db = open_test_db(tmp.path());
        let result = load_manifest(&db).unwrap();
        assert!(result.is_none(), "load_manifest should return None when no manifest has been saved");
    }

    #[test]
    fn test_save_manifest_updates_existing() {
        use crate::iceberg::{DataFileEntry, IcebergExport};

        let tmp = tempfile::TempDir::new().unwrap();
        let db = open_test_db(tmp.path());

        let mut v1 = IcebergExport::new("users".to_string(), 1);
        v1.add_entry(DataFileEntry::new("seg1.vortex".into(), 100, 1024));
        save_manifest(&db, &v1).unwrap();

        let mut v2 = IcebergExport::new("users".to_string(), 2);
        v2.add_entry(DataFileEntry::new("seg1.vortex".into(), 100, 1024));
        v2.add_entry(DataFileEntry::new("seg2.vortex".into(), 200, 2048));
        save_manifest(&db, &v2).unwrap();

        let loaded = load_manifest(&db).unwrap().unwrap();
        assert_eq!(loaded.snapshot_id, 2);
        assert_eq!(loaded.entries.len(), 2, "second save should overwrite first");
    }

    #[test]
    fn test_append_and_load_history() {
        let tmp = tempfile::TempDir::new().unwrap();
        let db = open_test_db(tmp.path());

        append_history(&db, SnapshotRef::main_branch(10)).unwrap();
        append_history(&db, SnapshotRef {
            name: "backup".to_string(),
            snapshot_id: 20,
            type_: "tag".to_string(),
            min_snapshots_to_keep: Some(3),
            max_snapshot_age_ms: None,
            max_ref_age_ms: None,
        }).unwrap();
        append_history(&db, SnapshotRef::main_branch(30)).unwrap();

        let history = load_history(&db).unwrap();
        assert_eq!(history.len(), 3);
        assert_eq!(history[0].snapshot_id, 10);
        assert_eq!(history[0].name, "main");
        assert_eq!(history[1].snapshot_id, 20);
        assert_eq!(history[1].name, "backup");
        assert_eq!(history[1].type_, "tag");
        assert_eq!(history[2].snapshot_id, 30);
    }

    #[test]
    fn test_history_empty() {
        let tmp = tempfile::TempDir::new().unwrap();
        let db = open_test_db(tmp.path());
        let history = load_history(&db).unwrap();
        assert!(history.is_empty(), "load_history on fresh DB should return empty vec");
    }
}
