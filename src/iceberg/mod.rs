//! Iceberg module for RockDuck
//!
//! Provides Iceberg table format export functionality.
//!
//! ## Migration: apache-avro -> apache/iceberg-rust
//!
//! This module has been migrated from hand-written Avro to `apache/iceberg-rust`.
//! The iceberg-rust crate handles:
//! - Manifest file Avro encoding (no apache-avro dependency)
//! - Manifest list Avro encoding
//! - Table metadata JSON serialization
//!
//! ## Vortex Integration (pending)
//!
//! VortexFileWriter is scaffolded and will be fully implemented once
//! Iceberg 1.11.0 adds native Vortex FormatModel support.
//! See: `src/iceberg/vortex_writer.rs`
//!
//! ## Feature Gate

//! This module is only available when the `iceberg_export` feature is enabled.

use std::collections::HashMap;

// Required for Array trait methods (null_count, len, is_null, etc.)
use arrow_array::Array;

#[cfg(feature = "iceberg_export")]
use iceberg::spec::{Datum, Schema};

#[cfg(feature = "iceberg_export")]
pub mod export;
pub mod translate;
#[cfg(feature = "iceberg_export")]
pub mod vortex_writer;

#[cfg(feature = "iceberg_export")]
pub use export::*;
#[cfg(feature = "iceberg_export")]
pub use translate::*;

/// Default sort order ID for Iceberg tables
pub const DEFAULT_SORT_ORDER_ID: i32 = 0;

/// Manifest file info for Iceberg manifest list
/// (moved from avro_writer.rs which is being replaced by apache/iceberg-rust)
/// TODO[ICEBERG]: 此结构将被 iceberg-rust 内置 manifest writer 替代
#[derive(Debug, Clone)]
pub struct ManifestFileInfo {
    /// Path to the manifest file
    pub manifest_path: String,
    /// Length of the manifest file in bytes
    pub manifest_length: i64,
    /// Number of added data files
    pub added_files_count: i64,
    /// Number of existing data files
    pub existing_files_count: i64,
    /// Number of deleted data files
    pub deleted_files_count: i64,
    /// Number of rows in added data files
    pub added_rows_count: i64,
    /// Number of rows in existing data files
    pub existing_rows_count: i64,
    /// Number of rows in deleted data files
    pub deleted_rows_count: i64,
}

/// Data file entry for Iceberg manifest
#[derive(
    Debug,
    Clone,
    serde::Serialize,
    serde::Deserialize,
    rkyv::Archive,
    rkyv::Serialize,
    rkyv::Deserialize,
)]
pub struct DataFileEntry {
    /// File path
    pub file_path: String,
    /// File format (parquet, orc, avro)
    pub file_format: String,
    /// Partition values (stored as JSON string for cross-platform rkyv compatibility)
    pub partition: String,
    /// Record count
    pub record_count: i64,
    /// File size in bytes
    pub file_size_bytes: i64,
    /// Column sizes
    pub column_sizes: HashMap<i32, i64>,
    /// Value counts per column
    pub value_counts: HashMap<i32, i64>,
    /// Null counts per column
    pub null_counts: HashMap<i32, i64>,
    /// Nan counts per column
    pub nan_counts: HashMap<i32, i64>,
    /// Lower bounds per column
    pub lower_bounds: HashMap<i32, Vec<u8>>,
    /// Upper bounds per column
    pub upper_bounds: HashMap<i32, Vec<u8>>,
    /// Key metadata
    pub key_metadata: Option<Vec<u8>>,
    /// Split offsets
    pub split_offsets: Vec<i64>,
}

impl DataFileEntry {
    /// Create a new data file entry
    pub fn new(file_path: String, record_count: i64, file_size_bytes: i64) -> Self {
        Self {
            file_path,
            file_format: "parquet".to_string(),
            partition: serde_json::Value::Null.to_string(),
            record_count,
            file_size_bytes,
            column_sizes: Default::default(),
            value_counts: Default::default(),
            null_counts: Default::default(),
            nan_counts: Default::default(),
            lower_bounds: Default::default(),
            upper_bounds: Default::default(),
            key_metadata: None,
            split_offsets: vec![],
        }
    }
}

/// Iceberg export configuration
#[derive(Debug, Clone)]
pub struct IcebergExport {
    /// Export format
    pub format: ExportFormat,
    /// Sort order ID
    pub sort_order_id: i32,
    /// Partition spec
    pub partition_spec: Vec<PartitionField>,
    /// Snapshot ID (optional)
    pub snapshot_id: Option<i64>,
    /// Sequence number (optional)
    pub sequence_number: Option<i64>,
    /// Table UUID (optional)
    pub table_uuid: Option<String>,
    /// Last updated timestamp (optional)
    pub last_updated_ms: Option<i64>,
    /// Properties (optional)
    pub properties: Option<std::collections::HashMap<String, String>>,
    /// Data file entries in this export
    pub entries: Vec<DataFileEntry>,
}

impl IcebergExport {
    /// Create a new Iceberg export for a given table name and snapshot ID.
    pub fn new(_table_name: String, snapshot_id: i64) -> Self {
        Self {
            format: ExportFormat::Vortex,
            sort_order_id: DEFAULT_SORT_ORDER_ID,
            partition_spec: Vec::new(),
            snapshot_id: Some(snapshot_id),
            sequence_number: None,
            table_uuid: Some(uuid::Uuid::new_v4().to_string()),
            last_updated_ms: Some(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_millis() as i64)
                    .unwrap_or(0),
            ),
            properties: None,
            entries: Vec::new(),
        }
    }

    /// Add a data file entry to this export.
    pub fn add_entry(&mut self, entry: DataFileEntry) {
        self.entries.push(entry);
    }
}

impl Default for IcebergExport {
    fn default() -> Self {
        Self {
            format: ExportFormat::Parquet,
            sort_order_id: DEFAULT_SORT_ORDER_ID,
            partition_spec: vec![],
            snapshot_id: None,
            sequence_number: None,
            table_uuid: None,
            last_updated_ms: None,
            properties: None,
            entries: vec![],
        }
    }
}

/// Export format
#[derive(Debug, Clone, Copy)]
pub enum ExportFormat {
    Parquet,
    Orc,
    Avro,
    Vortex,
}

/// Partition field definition
#[derive(Debug, Clone)]
pub struct PartitionField {
    /// Source column ID
    pub source_id: i32,
    /// Partition field ID
    pub field_id: i32,
    /// Transform type
    pub transform: String,
    /// Partition field name
    pub name: String,
}

// =============================================================================
// Column Statistics Accumulator
// =============================================================================

/// Accumulates per-column statistics from Arrow `RecordBatch`es during export.
///
/// Used to populate Iceberg data file column stats:
/// - `column_sizes`: total bytes per column
/// - `value_counts`: total non-null values per column
/// - `null_counts`: total null values per column
/// - `lower_bounds`: minimum value per column
/// - `upper_bounds`: maximum value per column
///
/// ## Iceberg Compatibility
/// Lower/upper bounds are stored as Iceberg `Datum` values, which are
/// serialized according to the Iceberg spec (big-endian for fixed-width types).
#[derive(Debug, Clone)]
pub struct ColumnStatsAccumulator {
    /// Iceberg schema reference
    pub schema: Schema,
    /// Column ID → total byte size (sum of all array buffers)
    pub column_sizes: HashMap<i32, u64>,
    /// Column ID → total non-null value count
    pub value_counts: HashMap<i32, u64>,
    /// Column ID → total null count
    pub null_counts: HashMap<i32, u64>,
    /// Column ID → NaN count (only for float columns; u64 for accumulation across batches)
    pub nan_counts: HashMap<i32, u64>,
    /// Column ID → lower bound (as Iceberg Datum)
    pub lower_bounds: HashMap<i32, Datum>,
    /// Column ID → upper bound (as Iceberg Datum)
    pub upper_bounds: HashMap<i32, Datum>,
}

impl ColumnStatsAccumulator {
    /// Create a new accumulator for the given Iceberg schema.
    pub fn new(schema: Schema) -> Self {
        Self {
            schema,
            column_sizes: HashMap::new(),
            value_counts: HashMap::new(),
            null_counts: HashMap::new(),
            nan_counts: HashMap::new(),
            lower_bounds: HashMap::new(),
            upper_bounds: HashMap::new(),
        }
    }

    /// Update statistics for one column's array data.
    ///
    /// Called once per column per `RecordBatch` during the export scan.
    pub fn update(&mut self, field_id: i32, arr: &dyn arrow_array::Array) {
        // Column size: total byte length of the array
        let col_size = arr.get_buffer_memory_size() as u64;
        *self.column_sizes.entry(field_id).or_insert(0) += col_size;

        // Value count: non-null rows
        let non_null = (arr.len() - arr.null_count()) as u64;
        *self.value_counts.entry(field_id).or_insert(0) += non_null;

        // Null count
        *self.null_counts.entry(field_id).or_insert(0) += arr.null_count() as u64;

        // Min/max — downcast to concrete array types for arrow-arith kernels
        if non_null > 0 {
            update_min_max_for_array(arr, field_id, self);
        }

        // nan_counts: only for floating-point arrays, accumulated across batches.
        // B7 fix: Iceberg spec requires nan_counts per column; these were previously always zero.
        count_nan_for_array(arr, field_id, self);
    }

    /// Update the lower bound for a column using an f64 value.
    ///
    /// Note: Iceberg spec requires `Datum::double` for Float64 columns.
    /// NaN values are skipped — min of {x, NaN} = x (NaN comparisons always return false).
    /// If an existing bound is a different type (shouldn't happen in practice), skip the update.
    fn update_min_f64(&mut self, field_id: i32, value: f64) {
        if value.is_nan() {
            return;
        }
        let datum = Datum::double(value);
        let entry = self.lower_bounds.entry(field_id);
        match entry {
            std::collections::hash_map::Entry::Vacant(e) => {
                e.insert(datum);
            }
            std::collections::hash_map::Entry::Occupied(mut e) => {
                if matches!(
                    e.get().literal(),
                    iceberg::spec::PrimitiveLiteral::Double(_)
                ) {
                    let existing = match e.get().literal() {
                        iceberg::spec::PrimitiveLiteral::Double(v) => **v,
                        _ => unreachable!(),
                    };
                    if value < existing {
                        e.insert(datum);
                    }
                }
                // Otherwise: existing bound is a different type — can't compare, leave it.
            }
        }
    }

    /// Update the upper bound for a column using an f64 value.
    fn update_max_f64(&mut self, field_id: i32, value: f64) {
        if value.is_nan() {
            return;
        }
        let datum = Datum::double(value);
        let entry = self.upper_bounds.entry(field_id);
        match entry {
            std::collections::hash_map::Entry::Vacant(e) => {
                e.insert(datum);
            }
            std::collections::hash_map::Entry::Occupied(mut e) => {
                if matches!(
                    e.get().literal(),
                    iceberg::spec::PrimitiveLiteral::Double(_)
                ) {
                    let existing = match e.get().literal() {
                        iceberg::spec::PrimitiveLiteral::Double(v) => **v,
                        _ => unreachable!(),
                    };
                    if value > existing {
                        e.insert(datum);
                    }
                }
            }
        }
    }

    /// Update the lower bound for a column using an f32 value.
    ///
    /// Iceberg spec requires `Datum::float` for Float32 columns.
    fn update_min_f32(&mut self, field_id: i32, value: f32) {
        if value.is_nan() {
            return;
        }
        let datum = Datum::float(value);
        let entry = self.lower_bounds.entry(field_id);
        match entry {
            std::collections::hash_map::Entry::Vacant(e) => {
                e.insert(datum);
            }
            std::collections::hash_map::Entry::Occupied(mut e) => {
                if matches!(e.get().literal(), iceberg::spec::PrimitiveLiteral::Float(_)) {
                    let existing = match e.get().literal() {
                        iceberg::spec::PrimitiveLiteral::Float(v) => **v,
                        _ => unreachable!(),
                    };
                    if value < existing {
                        e.insert(datum);
                    }
                }
            }
        }
    }

    /// Update the upper bound for a column using an f32 value.
    fn update_max_f32(&mut self, field_id: i32, value: f32) {
        if value.is_nan() {
            return;
        }
        let datum = Datum::float(value);
        let entry = self.upper_bounds.entry(field_id);
        match entry {
            std::collections::hash_map::Entry::Vacant(e) => {
                e.insert(datum);
            }
            std::collections::hash_map::Entry::Occupied(mut e) => {
                if matches!(e.get().literal(), iceberg::spec::PrimitiveLiteral::Float(_)) {
                    let existing = match e.get().literal() {
                        iceberg::spec::PrimitiveLiteral::Float(v) => **v,
                        _ => unreachable!(),
                    };
                    if value > existing {
                        e.insert(datum);
                    }
                }
            }
        }
    }

    /// Update the lower bound for a column using an i64 value.
    fn update_min_i64(&mut self, field_id: i32, value: i64) {
        match self.lower_bounds.entry(field_id) {
            std::collections::hash_map::Entry::Vacant(e) => {
                e.insert(Datum::long(value));
            }
            std::collections::hash_map::Entry::Occupied(mut e) => {
                let existing = match e.get().literal() {
                    iceberg::spec::PrimitiveLiteral::Long(v) => *v,
                    _ => i64::MAX,
                };
                if value < existing {
                    e.insert(Datum::long(value));
                }
            }
        }
    }

    /// Update the upper bound for a column using an i64 value.
    fn update_max_i64(&mut self, field_id: i32, value: i64) {
        match self.upper_bounds.entry(field_id) {
            std::collections::hash_map::Entry::Vacant(e) => {
                e.insert(Datum::long(value));
            }
            std::collections::hash_map::Entry::Occupied(mut e) => {
                let existing = match e.get().literal() {
                    iceberg::spec::PrimitiveLiteral::Long(v) => *v,
                    _ => i64::MIN,
                };
                if value > existing {
                    e.insert(Datum::long(value));
                }
            }
        }
    }

    /// Update the lower bound for a column using a string value.
    fn update_min_string(&mut self, field_id: i32, value: String) {
        let entry = self.lower_bounds.entry(field_id);
        match entry {
            std::collections::hash_map::Entry::Vacant(e) => {
                e.insert(Datum::string(&value));
            }
            std::collections::hash_map::Entry::Occupied(mut e) => {
                if let iceberg::spec::PrimitiveLiteral::String(existing) = e.get().literal() {
                    if &value < existing {
                        e.insert(Datum::string(&value));
                    }
                }
            }
        }
    }

    /// Update the upper bound for a column using a string value.
    fn update_max_string(&mut self, field_id: i32, value: String) {
        let entry = self.upper_bounds.entry(field_id);
        match entry {
            std::collections::hash_map::Entry::Vacant(e) => {
                e.insert(Datum::string(&value));
            }
            std::collections::hash_map::Entry::Occupied(mut e) => {
                if let iceberg::spec::PrimitiveLiteral::String(existing) = e.get().literal() {
                    if &value > existing {
                        e.insert(Datum::string(&value));
                    }
                }
            }
        }
    }
}

/// Downcast a generic Arrow array to concrete numeric types and update min/max stats.
fn update_min_max_for_array(
    arr: &dyn arrow_array::Array,
    field_id: i32,
    stats: &mut ColumnStatsAccumulator,
) {
    // Try Int64
    if let Some(col) = arr.as_any().downcast_ref::<arrow_array::Int64Array>() {
        if col.null_count() < col.len() {
            if let Some(min) = arrow_arith::aggregate::min(col) {
                stats.update_min_i64(field_id, min);
            }
            if let Some(max) = arrow_arith::aggregate::max(col) {
                stats.update_max_i64(field_id, max);
            }
        }
        return;
    }

    // Try Int32
    if let Some(col) = arr.as_any().downcast_ref::<arrow_array::Int32Array>() {
        if col.null_count() < col.len() {
            if let Some(min) = arrow_arith::aggregate::min(col) {
                stats.update_min_i64(field_id, min as i64);
            }
            if let Some(max) = arrow_arith::aggregate::max(col) {
                stats.update_max_i64(field_id, max as i64);
            }
        }
        return;
    }

    // Try Float64
    if let Some(col) = arr.as_any().downcast_ref::<arrow_array::Float64Array>() {
        if col.null_count() < col.len() {
            if let Some(min) = arrow_arith::aggregate::min(col) {
                stats.update_min_f64(field_id, min);
            }
            if let Some(max) = arrow_arith::aggregate::max(col) {
                stats.update_max_f64(field_id, max);
            }
        }
        return;
    }

    // Try Float32
    if let Some(col) = arr.as_any().downcast_ref::<arrow_array::Float32Array>() {
        if col.null_count() < col.len() {
            if let Some(min) = arrow_arith::aggregate::min(col) {
                stats.update_min_f32(field_id, min);
            }
            if let Some(max) = arrow_arith::aggregate::max(col) {
                stats.update_max_f32(field_id, max);
            }
        }
        return;
    }

    // Try StringArray — use vectorized min_string / max_string
    if let Some(col) = arr.as_any().downcast_ref::<arrow_array::StringArray>() {
        if col.null_count() < col.len() {
            if let Some(min) = arrow::compute::min_string(col) {
                stats.update_min_string(field_id, min);
            }
            if let Some(max) = arrow::compute::max_string(col) {
                stats.update_max_string(field_id, max);
            }
        }
    }
}

/// Count NaN values in a float array and accumulate into `nan_counts`.
///
/// B7 fix: Iceberg spec requires `nan_counts` per column for statistics-based
/// partition pruning. Previously always zero. Accumulates across batches by adding
/// to the existing count (each batch adds its own NaN count).
fn count_nan_for_array(
    arr: &dyn arrow_array::Array,
    field_id: i32,
    stats: &mut ColumnStatsAccumulator,
) {
    use arrow_arith::arithmetic::is_nan;

    if let Some(col) = arr.as_any().downcast_ref::<arrow_array::Float64Array>() {
        let nan_mask = is_nan(col).unwrap_or_else(|_| {
            arrow_array::BooleanArray::from(vec![false; col.len()])
        });
        let nan_count = nan_mask.true_count() as u64;
        *stats.nan_counts.entry(field_id).or_insert(0) += nan_count;
        return;
    }

    if let Some(col) = arr.as_any().downcast_ref::<arrow_array::Float32Array>() {
        let nan_mask = is_nan(col).unwrap_or_else(|_| {
            arrow_array::BooleanArray::from(vec![false; col.len()])
        });
        let nan_count = nan_mask.true_count() as u64;
        *stats.nan_counts.entry(field_id).or_insert(0) += nan_count;
        return;
    }

    // Non-float columns: nan_count is always 0 (no action needed).
}

// =============================================================================
// Export Configuration
// =============================================================================

/// Configuration for Iceberg export operations.
#[derive(Debug, Clone)]
pub struct ExportConfig {
    /// Target file format for data files
    pub format: ExportFormat,
    /// Sort order ID (0 = unsorted)
    pub sort_order_id: i32,
    /// Snapshot ID (optional, auto-generated if None)
    pub snapshot_id: Option<i64>,
    /// Table UUID (optional, auto-generated if None)
    pub table_uuid: Option<String>,
}

impl Default for ExportConfig {
    fn default() -> Self {
        Self {
            format: ExportFormat::Vortex,
            sort_order_id: DEFAULT_SORT_ORDER_ID,
            snapshot_id: None,
            table_uuid: None,
        }
    }
}
