//! Iceberg table format translation
//!
//! Translates internal segment metadata to Iceberg table format for export.
//!
//! ## Type Mapping Safety (ICE-06)
//!
//! - **UInt32 → Int (i32)**: Arrow UInt32 values > 2,147,483,647 overflow in Iceberg Int.
//!   This codebase does not use UInt32 columns — no actual risk. If added later, MUST revisit.
//! - **Date64 → Timestamp**: Arrow Date64 stores ms-since-epoch, NOT calendar date.
//!   Only Arrow Date32 (days-since-epoch) maps to Iceberg Date.

use crate::iceberg::DataFileEntry;
use crate::segment::meta::{ColumnDef, DataType, SegmentMeta};
use serde_json::json;

/// Translate data type to Iceberg format
pub fn data_type_to_iceberg(dt: &DataType) -> serde_json::Value {
    match dt {
        DataType::Int8 => json!({"type": "int", "signed": true}),
        DataType::Int16 => json!({"type": "int", "signed": true}),
        DataType::Int32 => json!({"type": "int", "signed": true}),
        DataType::Int64 => json!({"type": "long", "signed": true}),
        DataType::UInt8 => json!({"type": "int", "signed": false}),
        DataType::UInt16 => json!({"type": "int", "signed": false}),
        DataType::UInt32 => json!({"type": "int", "signed": false}),
        DataType::UInt64 => json!({"type": "long", "signed": false}),
        DataType::Float32 => json!({"type": "float"}),
        DataType::Float64 => json!({"type": "double"}),
        DataType::Bool => json!({"type": "boolean"}),
        DataType::Utf8 => json!({"type": "string"}),
        DataType::LargeUtf8 => json!({"type": "string"}),
        DataType::Binary => json!({"type": "binary"}),
        DataType::LargeBinary => json!({"type": "binary"}),
        DataType::Date32 => json!({"type": "date"}),
        // Date64 -> Timestamp (NOT Date):
        // Arrow Date64 stores milliseconds-since-epoch, which is a timestamp
        // representation. Iceberg Date stores year-month-day (days-since-epoch).
        // Mapping Date64 to Date would result in completely wrong dates.
        DataType::Date64 => json!({"type": "timestamp-micros"}),
        DataType::TimestampMicros => json!({"type": "long", "logicalType": "timestamp(6)"}),
        DataType::TimestampMillis => json!({"type": "long", "logicalType": "timestamp(3)"}),
    }
}

/// Translate a column definition to Iceberg format.
/// Accepts a unique field_id per column instead of hardcoding to 1.
/// Iceberg requires unique field IDs across the schema for proper schema evolution.
pub fn column_to_iceberg(col: &ColumnDef, field_id: i32) -> serde_json::Value {
    json!({
        "id": field_id,
        "name": col.name,
        "type": data_type_to_iceberg(&col.data_type),
        "required": !col.nullable,
    })
}

/// Translate segment to Iceberg data files (stub).
/// TODO[ICEBERG]: implement once VortexFileWriter is available.
#[allow(dead_code)]
pub fn segment_to_data_files(_seg: &SegmentMeta) -> Vec<DataFileEntry> {
    Vec::new()
}
