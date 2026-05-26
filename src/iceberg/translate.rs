//! Translation from RockDuck segment metadata to Iceberg v2 spec artifacts.
//!
//! This module translates our native `IcebergExport` + `SegmentMeta` into:
//!   - Iceberg TableMetadata JSON (`metadata.json`)
//!   - Manifest file entries (consumed by avro_writer.rs)
//!
//! Iceberg spec reference: <https://iceberg.apache.org/spec/>

use std::collections::HashMap;
use crate::iceberg::{DataFileEntry, IcebergExport, DEFAULT_SORT_ORDER_ID};
use crate::segment::meta::{ColumnDef, DataType, SegmentMeta};

/// Mapping from column name → Iceberg field ID (used in DataFile lower/upper bounds).
type FieldIdMap = HashMap<String, i32>;

/// Convert RockDuck `DataType` to Iceberg v2 type string.
///
/// Iceberg uses a nested struct for the type representation.
/// We return the inner type string for the Iceberg `Type` union.
pub fn data_type_to_iceberg(dt: &DataType) -> serde_json::Value {
    use serde_json::json;
    match dt {
        DataType::Int8 => json!({"type": "int", "bitWidth": 8, "isSigned": true}),
        DataType::Int16 => json!({"type": "int", "bitWidth": 16, "isSigned": true}),
        DataType::Int32 => json!({"type": "int", "bitWidth": 32, "isSigned": true}),
        DataType::Int64 => json!({"type": "long"}),
        DataType::UInt8 => json!({"type": "int", "bitWidth": 8, "isSigned": false}),
        DataType::UInt16 => json!({"type": "int", "bitWidth": 16, "isSigned": false}),
        DataType::UInt32 => json!({"type": "int", "bitWidth": 32, "isSigned": false}),
        DataType::UInt64 => json!({"type": "long", "isSigned": false}),
        DataType::Float32 => json!({"type": "float", "bitWidth": 32}),
        DataType::Float64 => json!({"type": "double", "bitWidth": 64}),
        DataType::Bool => json!({"type": "boolean"}),
        DataType::Utf8 | DataType::LargeUtf8 => json!({"type": "string"}),
        DataType::Binary | DataType::LargeBinary => json!({"type": "binary"}),
        DataType::Date32 => json!({"type": "date"}),
        DataType::Date64 => json!({"type": "timestamp", "unit": "micro", "adjustToUTC": false}),
        DataType::TimestampSecond => json!({"type": "timestamp", "unit": "second", "adjustToUTC": false}),
        DataType::TimestampMillisecond => json!({"type": "timestamp", "unit": "millisecond", "adjustToUTC": false}),
        DataType::TimestampMicrosecond => json!({"type": "timestamp", "unit": "micro", "adjustToUTC": false}),
        DataType::TimestampNanosecond => json!({"type": "timestamp", "unit": "nanosecond", "adjustToUTC": false}),
    }
}

/// Build the Iceberg v2 schema JSON from a list of column definitions.
pub fn to_iceberg_schema(columns: &[ColumnDef]) -> serde_json::Value {
    use serde_json::json;

    let fields: Vec<serde_json::Value> = columns
        .iter()
        .enumerate()
        .map(|(i, col)| {
            let field_id = (i as i32) + 1; // Iceberg field IDs are 1-based
            json!({
                "id": field_id,
                "name": col.name,
                "type": data_type_to_iceberg(&col.dtype),
                "required": true
            })
        })
        .collect();

    json!({
        "type": "struct",
        "schema-id": 0,
        "fields": fields
    })
}

/// Build the Iceberg v2 sort-order JSON from a list of column definitions.
/// The default order is primary-key ascending (sort by all columns).
pub fn to_sort_order(columns: &[ColumnDef]) -> serde_json::Value {
    use serde_json::json;

    let transforms: Vec<serde_json::Value> = columns
        .iter()
        .enumerate()
        .map(|(i, _col)| {
            let field_id = (i as i32) + 1;
            json!({
                "transform": "identity",
                "source-id": field_id,
                "direction": "asc",
                "null-order": "nulls-first"
            })
        })
        .collect();

    json!({
        "sort-order-id": DEFAULT_SORT_ORDER_ID,
        "fields": transforms
    })
}

/// Encode a byte slice as a lowercase hex string (Iceberg convention for binary bounds).
pub fn bytes_to_hex(bytes: &[u8]) -> String {
    hex::encode(bytes)
}

/// Decode a lowercase hex string back to bytes.
pub fn hex_to_bytes(hex: &str) -> Result<Vec<u8>, hex::FromHexError> {
    hex::decode(hex)
}

/// Build the Iceberg v2 TableMetadata JSON.
///
/// `table_location` is the root path of the Iceberg table (e.g. "s3://bucket/table/").
/// `current_snapshot_id` is the snapshot ID of the current manifest list.
pub fn to_iceberg_table_metadata(
    manifest: &IcebergExport,
    table_location: &str,
    schema_json: &serde_json::Value,
    sort_order_json: &serde_json::Value,
) -> serde_json::Value {
    use serde_json::json;

    let manifest_list_path = format!(
        "{}/metadata/snap-{}-{}.avro",
        table_location.trim_end_matches('/'),
        manifest.snapshot_id,
        manifest.sequence_number
    );

    json!({
        "format-version": 2,
        "table-uuid": manifest.table_uuid,
        "location": table_location,
        "last-sequence-number": manifest.sequence_number,
        "last-updated-ms": manifest.last_updated_ms,
        "schemas": [
            {
                "schema-id": 0,
                "type": schema_json["type"],
                "schema-id": schema_json["schema-id"],
                "fields": schema_json["fields"]
            }
        ],
        "default-sort-order": {
            "order-id": sort_order_json["sort-order-id"],
            "fields": sort_order_json["fields"]
        },
        "current-schema-id": 0,
        "current-sort-order-id": DEFAULT_SORT_ORDER_ID,
        "properties": manifest.properties,
        "snapshots": [
            {
                "snapshot-id": manifest.snapshot_id,
                "sequence-number": manifest.sequence_number,
                "summary": {
                    "operation": "dynamic",
                    "spark.app-initialCommitTime": manifest.last_updated_ms.to_string()
                },
                "manifest-list": manifest_list_path,
                "schema-id": 0
            }
        ],
        "refs": {
            "main": {
                "snapshot-id": manifest.snapshot_id,
                "type": "branch"
            }
        }
    })
}

/// Build DataFileEntry objects from a list of segments and their granules.
///
/// For each column in each segment, we create one `DataFileEntry` that points to the
/// column's `.vortex` file. The per-granule statistics (block_stats, zone_map) are
/// aggregated into the entry's bounds.
///
/// `field_id_map` maps column names to Iceberg field IDs (column_index + 1).
pub fn build_data_file_entries<'a>(
    segments: impl Iterator<Item = &'a SegmentMeta>,
    field_id_map: &FieldIdMap,
    data_dir: &std::path::Path,
) -> Vec<DataFileEntry> {
    let mut entries = Vec::new();

    for seg in segments {
        for col in &seg.columns {
            let Some(&field_id) = field_id_map.get(&col.name) else {
                continue;
            };

            // Build per-granule split offsets + aggregate bounds
            let mut split_offsets: Vec<u64> = Vec::new();
            let mut global_lower: HashMap<i32, Vec<u8>> = HashMap::new();
            let mut global_upper: HashMap<i32, Vec<u8>> = HashMap::new();
            let mut null_counts: HashMap<i32, u64> = HashMap::new();
            let mut total_records: u64 = 0;

            for granule in &seg.granules {
                split_offsets.push(granule.file_offset);
                total_records += granule.row_count as u64;

                // Aggregate null counts from zone_map
                if let Some(col_stats) = granule.zone_map.stats.get(&col.name) {
                    *null_counts.entry(field_id).or_insert(0) += col_stats.null_count as u64;

                    // Update lower bounds
                    if let Some(ref min_bytes) = col_stats.min {
                        global_lower
                            .entry(field_id)
                            .or_insert_with(|| min_bytes.clone())
                            .clone_from(min_bytes);
                    }

                    // Update upper bounds
                    if let Some(ref max_bytes) = col_stats.max {
                        global_upper
                            .entry(field_id)
                            .or_insert_with(|| max_bytes.clone())
                            .clone_from(max_bytes);
                    }
                }

                // Also check block_stats for more fine-grained bounds
                for block in &granule.block_stats {
                    if let Some((ref block_min, ref block_max)) = block.column_stats.get(&col.name) {
                        let lower_entry = global_lower.entry(field_id).or_insert_with(|| block_min.clone());
                        let upper_entry = global_upper.entry(field_id).or_insert_with(|| block_max.clone());

                        if block_min.as_slice() < lower_entry.as_slice() {
                            *lower_entry = block_min.clone();
                        }
                        if block_max.as_slice() > upper_entry.as_slice() {
                            *upper_entry = block_max.clone();
                        }
                    }
                }
            }

            // Compute actual file size from disk
            let seg_path = data_dir.join("segments").join(&seg.seg_id);
            let col_path = seg_path.join(format!("{}.vortex", col.name));
            let file_size = std::fs::metadata(&col_path)
                .map(|m| m.len())
                .unwrap_or(0);

            let file_path = format!("segments/{}/{}", seg.seg_id, col.name);

            let mut entry = DataFileEntry::new(file_path, total_records, file_size);
            entry.lower_bounds = global_lower;
            entry.upper_bounds = global_upper;
            entry.null_counts = null_counts;
            entry.split_offsets = split_offsets;
            entry.sort_order_id = DEFAULT_SORT_ORDER_ID;

            entries.push(entry);
        }
    }

    entries
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_type_to_iceberg_all_types() {
        use crate::segment::meta::DataType;
        let types = [
            (DataType::Int8, "int"),
            (DataType::Int32, "int"),
            (DataType::Int64, "long"),
            (DataType::Float32, "float"),
            (DataType::Float64, "double"),
            (DataType::Bool, "boolean"),
            (DataType::Utf8, "string"),
            (DataType::Binary, "binary"),
            (DataType::Date32, "date"),
            (DataType::TimestampMillisecond, "timestamp"),
        ];
        for (dtype, expected_type) in types {
            let result = data_type_to_iceberg(&dtype);
            assert_eq!(result["type"], expected_type, "Failed for {:?}", dtype);
        }
    }

    #[test]
    fn test_to_iceberg_schema() {
        use crate::segment::meta::{ColumnDef, DataType};
        let columns = vec![
            ColumnDef::new("id".to_string(), DataType::Int64),
            ColumnDef::new("name".to_string(), DataType::Utf8),
            ColumnDef::new("age".to_string(), DataType::Int32),
        ];
        let schema = to_iceberg_schema(&columns);
        assert_eq!(schema["type"], "struct");
        let fields = schema["fields"].as_array().unwrap();
        assert_eq!(fields.len(), 3);
        assert_eq!(fields[0]["id"], 1);
        assert_eq!(fields[0]["name"], "id");
        assert_eq!(fields[1]["id"], 2);
        assert_eq!(fields[2]["id"], 3);
    }

    #[test]
    fn test_bytes_to_hex_roundtrip() {
        let original = vec![0xDE, 0xAD, 0xBE, 0xEF];
        let hex_str = bytes_to_hex(&original);
        assert_eq!(hex_str, "deadbeef");
        let decoded = hex_to_bytes(&hex_str).unwrap();
        assert_eq!(decoded, original);
    }

    #[test]
    fn test_hex_to_bytes_invalid() {
        let result = hex_to_bytes("not_hex");
        assert!(result.is_err());
    }

    #[test]
    fn test_to_sort_order() {
        use crate::segment::meta::{ColumnDef, DataType};
        let columns = vec![
            ColumnDef::new("id".to_string(), DataType::Int64),
            ColumnDef::new("name".to_string(), DataType::Utf8),
        ];
        let order = to_sort_order(&columns);
        assert_eq!(order["sort-order-id"], DEFAULT_SORT_ORDER_ID);
        let fields = order["fields"].as_array().unwrap();
        assert_eq!(fields.len(), 2);
        assert_eq!(fields[0]["transform"], "identity");
        assert_eq!(fields[0]["source-id"], 1);
        assert_eq!(fields[0]["direction"], "asc");
    }

    #[test]
    fn test_data_file_entry_new() {
        use crate::iceberg::VORTEX_FORMAT;
        let entry = DataFileEntry::new("segments/seg_abc/id.vortex".to_string(), 1000, 4096);
        assert_eq!(entry.file_format, VORTEX_FORMAT);
        assert_eq!(entry.record_count, 1000);
        assert_eq!(entry.file_size, 4096);
        assert!(entry.lower_bounds.is_empty());
        assert!(entry.upper_bounds.is_empty());
        assert_eq!(entry.sort_order_id, DEFAULT_SORT_ORDER_ID);
    }

    #[test]
    fn test_iceberg_export_new() {
        let manifest = IcebergExport::new("users".to_string(), 100);
        assert_eq!(manifest.table_name, "users");
        assert_eq!(manifest.snapshot_id, 100);
        assert_eq!(manifest.sequence_number, 1);
        assert!(manifest.parent_snapshot_id.is_none());
        assert!(!manifest.table_uuid.is_empty());
        assert!(manifest.entries.is_empty());
    }

    #[test]
    fn test_iceberg_export_add_entry() {
        let mut manifest = IcebergExport::new("users".to_string(), 1);
        let entry = DataFileEntry::new("test.vortex".to_string(), 100, 1024);
        manifest.add_entry(entry);
        assert_eq!(manifest.entries.len(), 1);
        assert_eq!(manifest.sequence_number, 2); // 1 initial + 1 added
    }

    #[test]
    fn test_iceberg_export_advance_snapshot() {
        let mut manifest = IcebergExport::new("users".to_string(), 1);
        manifest.advance_snapshot(2);
        assert_eq!(manifest.snapshot_id, 2);
        assert_eq!(manifest.parent_snapshot_id, Some(1));
        assert_eq!(manifest.sequence_number, 1);
    }

    // ===================== build_data_file_entries tests =====================

    #[test]
    fn test_build_data_file_entries_empty() {
        let segments: Vec<SegmentMeta> = vec![];
        let field_id_map: FieldIdMap = HashMap::new();
        let entries = build_data_file_entries(segments.iter(), &field_id_map, std::path::Path::new("/tmp"));
        assert!(entries.is_empty());
    }

    #[test]
    fn test_build_data_file_entries_skips_non_frozen() {
        use crate::segment::meta::{ColumnDef, DataType, SegmentStatus};
        let seg = SegmentMeta {
            seg_id: "seg1".to_string(),
            table: "users".to_string(),
            row_count: 100,
            uncompressed_size: 4096,
            compressed_size: 4096,
            columns: vec![
                ColumnDef::new("id".to_string(), DataType::Int64),
                ColumnDef::new("name".to_string(), DataType::Utf8),
            ],
            granules: vec![],
            deleted_rows: 0,
            del_ratio: 0.0,
            status: SegmentStatus::Active,
            created_at: 0,
            updated_at: 0,
            pk_range: None,
        };

        let mut field_id_map = FieldIdMap::new();
        field_id_map.insert("id".to_string(), 1);
        field_id_map.insert("name".to_string(), 2);

        let entries = build_data_file_entries([&seg].into_iter(), &field_id_map, std::path::Path::new("/tmp"));
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].file_path, "segments/seg1/id");
        assert_eq!(entries[1].file_path, "segments/seg1/name");
    }

    #[test]
    fn test_build_data_file_entries_single_segment_two_columns() {
        use crate::segment::meta::{ColumnDef, DataType, GranuleMeta, ZoneMapStats, ColumnStats};

        let mut zone_map = ZoneMapStats::new();
        zone_map.add_column_stats("id", ColumnStats {
            min: Some(0i64.to_le_bytes().to_vec()),
            max: Some(99i64.to_le_bytes().to_vec()),
            null_count: 0,
            sum: None,
            distinct_count: None,
        });

        let seg = SegmentMeta {
            seg_id: "seg_abc".to_string(),
            table: "users".to_string(),
            row_count: 100,
            uncompressed_size: 4096,
            compressed_size: 4096,
            columns: vec![
                ColumnDef::new("id".to_string(), DataType::Int64),
                ColumnDef::new("name".to_string(), DataType::Utf8),
            ],
            granules: vec![
                GranuleMeta {
                    granule_id: 0,
                    row_offset: 0,
                    row_count: 100,
                    file_offset: 0,
                    compressed_size: 4096,
                    zone_map,
                    block_stats: vec![],
                }
            ],
            deleted_rows: 0,
            del_ratio: 0.0,
            status: crate::segment::meta::SegmentStatus::Frozen,
            created_at: 0,
            updated_at: 0,
            pk_range: None,
        };

        let mut field_id_map = FieldIdMap::new();
        field_id_map.insert("id".to_string(), 1);
        field_id_map.insert("name".to_string(), 2);

        let entries = build_data_file_entries([&seg].into_iter(), &field_id_map, std::path::Path::new("/tmp"));
        assert_eq!(entries.len(), 2);
        let id_entry = entries.iter().find(|e| e.file_path.contains("id")).unwrap();
        assert_eq!(id_entry.record_count, 100);
        assert!(!id_entry.split_offsets.is_empty());
        assert_eq!(id_entry.split_offsets[0], 0);
    }

    #[test]
    fn test_build_data_file_entries_field_id_mapping() {
        use crate::segment::meta::{ColumnDef, DataType, GranuleMeta, ZoneMapStats, ColumnStats};

        let mut zone_map = ZoneMapStats::new();
        zone_map.add_column_stats("score", ColumnStats {
            min: Some(1i32.to_le_bytes().to_vec()),
            max: Some(100i32.to_le_bytes().to_vec()),
            null_count: 3,
            sum: None,
            distinct_count: None,
        });

        let seg = SegmentMeta {
            seg_id: "seg1".to_string(),
            table: "t".to_string(),
            row_count: 50,
            uncompressed_size: 1024,
            compressed_size: 1024,
            columns: vec![ColumnDef::new("score".to_string(), DataType::Int32)],
            granules: vec![
                GranuleMeta {
                    granule_id: 0,
                    row_offset: 0,
                    row_count: 50,
                    file_offset: 0,
                    compressed_size: 1024,
                    zone_map,
                    block_stats: vec![],
                }
            ],
            deleted_rows: 0,
            del_ratio: 0.0,
            status: crate::segment::meta::SegmentStatus::Frozen,
            created_at: 0,
            updated_at: 0,
            pk_range: None,
        };

        let mut field_id_map = FieldIdMap::new();
        field_id_map.insert("score".to_string(), 5);

        let entries = build_data_file_entries([&seg].into_iter(), &field_id_map, std::path::Path::new("/tmp"));
        assert_eq!(entries.len(), 1);
        let entry = &entries[0];
        assert!(entry.lower_bounds.contains_key(&5));
        assert!(entry.upper_bounds.contains_key(&5));
        assert!(entry.null_counts.contains_key(&5));
    }

    #[test]
    fn test_build_data_file_entries_vortex_format() {
        use crate::segment::meta::{ColumnDef, DataType};
        let seg = SegmentMeta {
            seg_id: "seg1".to_string(),
            table: "t".to_string(),
            row_count: 10,
            uncompressed_size: 100,
            compressed_size: 100,
            columns: vec![ColumnDef::new("col1".to_string(), DataType::Int32)],
            granules: vec![],
            deleted_rows: 0,
            del_ratio: 0.0,
            status: crate::segment::meta::SegmentStatus::Frozen,
            created_at: 0,
            updated_at: 0,
            pk_range: None,
        };
        let mut field_id_map = FieldIdMap::new();
        field_id_map.insert("col1".to_string(), 1);
        let entries = build_data_file_entries([&seg].into_iter(), &field_id_map, std::path::Path::new("/tmp"));
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].file_format, "VORTEX");
    }

    // ===================== to_iceberg_table_metadata tests =====================

    #[test]
    fn test_to_iceberg_table_metadata_structure() {
        let manifest = IcebergExport::new("users".to_string(), 1);
        let schema_json = to_iceberg_schema(&[]);
        let sort_json = to_sort_order(&[]);
        let meta = to_iceberg_table_metadata(&manifest, "/tmp/table", &schema_json, &sort_json);

        // Required top-level keys per Iceberg spec v2
        assert!(meta.get("format-version").is_some(), "missing format-version");
        assert!(meta.get("table-uuid").is_some(), "missing table-uuid");
        assert!(meta.get("location").is_some(), "missing location");
        assert!(meta.get("schemas").is_some(), "missing schemas");
        assert!(meta.get("current-schema-id").is_some(), "missing current-schema-id");
        assert!(meta.get("snapshots").is_some(), "missing snapshots");
        assert!(meta.get("default-sort-order").is_some(), "missing default-sort-order");
        assert!(meta.get("refs").is_some(), "missing refs");
        assert!(meta.get("last-sequence-number").is_some(), "missing last-sequence-number");
        assert!(meta.get("last-updated-ms").is_some(), "missing last-updated-ms");
    }

    #[test]
    fn test_to_iceberg_table_metadata_format_version() {
        let manifest = IcebergExport::new("users".to_string(), 1);
        let schema_json = to_iceberg_schema(&[]);
        let sort_json = to_sort_order(&[]);
        let meta = to_iceberg_table_metadata(&manifest, "/tmp/table", &schema_json, &sort_json);
        assert_eq!(meta["format-version"], 2, "Iceberg format version must be exactly 2");
    }

    #[test]
    fn test_to_iceberg_table_metadata_contains_snapshot_ref() {
        let manifest = IcebergExport::new("users".to_string(), 42);
        let schema_json = to_iceberg_schema(&[]);
        let sort_json = to_sort_order(&[]);
        let meta = to_iceberg_table_metadata(&manifest, "/tmp/table", &schema_json, &sort_json);

        let refs = meta["refs"].as_object().unwrap();
        assert!(refs.contains_key("main"), "refs must contain 'main' branch");
        let main = &refs["main"];
        assert_eq!(main["snapshot-id"], 42, "main branch must point to snapshot_id");
        assert_eq!(main["type"], "branch", "main ref must be type=branch");
    }

    #[test]
    fn test_to_iceberg_table_metadata_location() {
        let manifest = IcebergExport::new("users".to_string(), 1);
        let schema_json = to_iceberg_schema(&[]);
        let sort_json = to_sort_order(&[]);
        let meta = to_iceberg_table_metadata(&manifest, "s3://bucket/table", &schema_json, &sort_json);
        assert_eq!(meta["location"], "s3://bucket/table");
        // manifest-list path should contain the table location
        let snapshots = meta["snapshots"].as_array().unwrap();
        let manifest_list = &snapshots[0]["manifest-list"];
        let path_str = manifest_list.as_str().unwrap();
        assert!(path_str.starts_with("s3://bucket/table/"), "manifest-list path must start with table location");
    }
}
