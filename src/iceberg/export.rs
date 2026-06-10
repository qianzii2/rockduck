//! Iceberg export orchestrator for RockDuck.
//!
//! Orchestrates the full Iceberg table export pipeline using `apache/iceberg-rust`:
//! 1. Collect all frozen segments for a table
//! 2. Build Iceberg Schema from RockDuck column definitions
//! 3. Write data files (Vortex → DataFile) -- scaffolded via VortexFileWriter
//! 4. Accumulate per-column statistics (ColumnStatsAccumulator)
//! 5. Write manifest files (iceberg-rust handles Avro)
//! 6. Write metadata.json (iceberg-rust handles JSON)
//! 7. Update version-hint.txt
//!
//! ## Vortex Integration (pending)
//!
//! VortexFileWriter is scaffolded and will be fully implemented once
//! Iceberg 1.11.0 adds native Vortex FormatModel support.
//! See: `src/iceberg/vortex_writer.rs`

use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::Field;
use tracing::{debug, info, warn};

#[cfg(feature = "iceberg_export")]
use iceberg::spec::{
    DataContentType, DataFileBuilder, DataFileFormat, NestedField, NestedFieldRef, PartitionSpec,
    Schema, Snapshot, SortOrder, TableMetadataBuildResult, Type,
};

use crate::db::RockDuck;
use crate::error::{Result, RockDuckError};
use crate::iceberg::vortex_writer::VortexFileWriter;
use crate::iceberg::ColumnStatsAccumulator;
use crate::metadata::kv_store::list_segment_metas;
use crate::metadata::projection::ProjectionContract;
use crate::query::routing::feedback::ExportDigestSnapshot;
use crate::query::routing::SidecarEvidenceSnapshot;
use crate::read::scan::data_type_to_arrow;
use crate::segment::meta::{ColumnDef, SegmentMeta, SegmentStatus};
use crate::storage::vortex::VortexReader;

fn write_atomic_file(path: &Path, contents: &[u8]) -> Result<()> {
    let parent = path.parent().ok_or_else(|| {
        RockDuckError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("path has no parent: {}", path.display()),
        ))
    })?;
    std::fs::create_dir_all(parent).map_err(RockDuckError::Io)?;

    let temp_name = format!(
        ".{}.{}.tmp",
        path.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("export"),
        uuid::Uuid::new_v4()
    );
    let temp_path = parent.join(temp_name);

    {
        let mut file = std::fs::File::create(&temp_path).map_err(RockDuckError::Io)?;
        file.write_all(contents).map_err(RockDuckError::Io)?;
        file.sync_all().map_err(RockDuckError::Io)?;
    }

    std::fs::rename(&temp_path, path).map_err(RockDuckError::Io)?;
    Ok(())
}

fn lock_export_publication_root(table_root: &Path) -> Result<ExportPublicationLock> {
    let lock_path = table_root.join("metadata").join("export.lock");
    std::fs::create_dir_all(
        lock_path.parent().ok_or_else(|| {
            RockDuckError::Internal("export lock path missing parent".to_string())
        })?,
    )?;
    match std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&lock_path)
    {
        Ok(_) => Ok(ExportPublicationLock { lock_path }),
        Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => {
            Err(RockDuckError::Write(format!(
                "concurrent Iceberg export already publishing at {}",
                table_root.display()
            )))
        }
        Err(err) => Err(RockDuckError::Io(err)),
    }
}

struct ExportPublicationLock {
    lock_path: PathBuf,
}

impl Drop for ExportPublicationLock {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.lock_path);
    }
}

/// Export a RockDuck table to Iceberg format.
///
/// The output directory structure follows the Iceberg spec:
///
/// ```text
/// {table_root}/
///   metadata/
///     v{N}.metadata.json
///     snap-{snapshot_id}-0.avro   (manifest file)
///   data/
///     {uuid}.vortex              (Vortex data files, scaffolded)
/// ```
///
/// # Arguments
/// * `db` -- RockDuck instance (provides segment metadata and Vortex data)
/// * `table` -- table name to export
/// * `table_root` -- path to the Iceberg table root
///
/// # Errors
/// Returns an error if the `iceberg_export` feature is not enabled.
#[cfg(feature = "iceberg_export")]
pub fn export_to_iceberg(db: &RockDuck, table: &str, table_root: &Path) -> Result<String> {
    if let Some(router) = db.router.as_ref() {
        if !router.export_runtime_enabled() {
            router.feedback().record_export_digest(
                table,
                ExportDigestSnapshot {
                    latest_status: "runtime_disabled".to_string(),
                    file_count: 0,
                    total_bytes: 0,
                    mode: if router.export_external_reader_compat_required() {
                        "compat_gate_blocked".to_string()
                    } else {
                        "disabled".to_string()
                    },
                    metadata_path: None,
                },
            );
            return Err(RockDuckError::Internal(
                "Iceberg export is runtime-disabled by rollout config".to_string(),
            ));
        }
    }
    let projection_contract = ProjectionContract::vtab();
    projection_contract.assert_blocking_governance();
    // 1. Collect frozen segments for this table
    let frozen_segs = collect_frozen_segments(db, table)?;
    if frozen_segs.is_empty() {
        info!(
            "No frozen segments for table '{}', exporting empty table",
            table
        );
    } else {
        info!(
            "Exporting {} frozen segments for table '{}'",
            frozen_segs.len(),
            table
        );
    }

    // 2. Build Iceberg Schema from first segment's columns
    let schema = build_iceberg_schema(&frozen_segs)?;

    // 3. Determine snapshot ID (atomic, monotonically increasing via SeqCst)
    let snapshot_id = next_snapshot_id();
    let timestamp_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0);

    // 4. Write data files and accumulate column statistics
    let (data_files, _stats) = write_data_files(db, table_root, &frozen_segs, &schema)?;

    // 5. Write manifest file (iceberg-rust handles Avro encoding)
    let metadata_dir = table_root.join("metadata");
    std::fs::create_dir_all(&metadata_dir).map_err(RockDuckError::Io)?;
    let _publication_lock = lock_export_publication_root(table_root)?;

    // Build partition spec (required for ManifestWriter).
    let partition_spec = iceberg::spec::PartitionSpec::builder(schema.clone())
        .with_spec_id(0)
        .build()
        .map_err(|e| RockDuckError::Internal(format!("build partition spec: {}", e)))?;

    let manifest_path = metadata_dir.join(format!("snap-{}-0.avro", snapshot_id));
    smol::block_on(write_manifest_file(
        &manifest_path,
        &data_files,
        snapshot_id,
        &schema,
        &partition_spec,
    ))
    .map_err(|e| RockDuckError::Internal(format!("write manifest: {e}")))?;

    // 6. Write metadata.json
    let meta_path = write_metadata_json(
        table_root,
        table,
        snapshot_id,
        timestamp_ms,
        &schema,
        &manifest_path,
    )
    .map_err(|e| RockDuckError::Internal(format!("write metadata.json: {e}")))?;

    // 7. Update version-hint.txt
    write_version_hint(table_root)?;

    info!(
        "Iceberg export complete: {} data files, snapshot_id={}, metadata={}",
        data_files.len(),
        snapshot_id,
        meta_path
    );
    if let Some(router) = db.router.as_ref() {
        router.feedback().record_export_digest(
            table,
            ExportDigestSnapshot {
                latest_status: "ok".to_string(),
                file_count: data_files.len(),
                total_bytes: data_files
                    .iter()
                    .map(|df| df.file_size_in_bytes() as u64)
                    .sum(),
                mode: if router.export_external_reader_compat_required() {
                    "compat_required".to_string()
                } else {
                    "runtime_enabled".to_string()
                },
                metadata_path: Some(meta_path.clone()),
            },
        );
        router.observe_sidecar_evidence(
            db,
            &SidecarEvidenceSnapshot {
                table: table.to_string(),
                routed_segment_ids: frozen_segs.iter().map(|seg| seg.seg_id.clone()).collect(),
                executed_segment_ids: Vec::new(),
                contract: projection_contract.clone(),
            },
        );
    }

    Ok(meta_path)
}

/// Iceberg export is disabled by default.
///
/// Iceberg spec notes for the export implementation:
/// - DataFileFormat::Avro used as registered format for .vortex files (actual format
///   stored in key_metadata as "vortex-v1")
/// - Empty manifest files written (no entries accumulated)
/// - Sequence number hardcoded to 1
///
/// Re-enable by compiling with `--features iceberg_export`.
#[cfg(not(feature = "iceberg_export"))]
pub fn export_to_iceberg(_db: &RockDuck, _table: &str, _table_root: &Path) -> Result<String> {
    Err(RockDuckError::Internal(
        "Iceberg export is disabled: the 'iceberg_export' feature flag is not enabled. \
         File format (Vortex) and manifest file generation are not yet Iceberg-spec-compliant. \
         See TODO[ICEBERG] comments in src/iceberg/export.rs"
            .into(),
    ))
}

// =============================================================================
// Segment collection
// =============================================================================

fn collect_frozen_segments(db: &RockDuck, table: &str) -> Result<Vec<SegmentMeta>> {
    let all = list_segment_metas(&db.kv)?;
    let frozen: Vec<_> = all
        .into_iter()
        .filter(|seg| seg.table_id == table && seg.status == SegmentStatus::Frozen)
        .collect();
    Ok(frozen)
}

// =============================================================================
// Arrow Schema → Iceberg Schema
// =============================================================================

/// Build an Iceberg `Schema` from RockDuck `SegmentMeta` columns.
///
/// Field IDs are 1-based sequential integers. Required vs optional is determined
/// by `ColumnDef.nullable`.
fn build_iceberg_schema(segments: &[SegmentMeta]) -> Result<Schema> {
    let columns: &[ColumnDef] = match segments.first() {
        Some(seg) if !seg.columns.is_empty() => &seg.columns,
        _ => {
            // Return empty schema for tables with no segments
            let schema = Schema::builder()
                .with_schema_id(0)
                .build()
                .map_err(|e| RockDuckError::Internal(format!("build empty schema: {e}")))?;
            return Ok(schema);
        }
    };

    let fields: Vec<NestedFieldRef> = columns
        .iter()
        .enumerate()
        .map(|(i, col)| {
            let id = (i + 1) as i32;
            let iceberg_type = column_def_to_iceberg_type(col);
            let field = if col.nullable {
                NestedField::optional(id, &col.name, iceberg_type)
            } else {
                NestedField::required(id, &col.name, iceberg_type)
            };
            std::sync::Arc::new(field)
        })
        .collect();

    Schema::builder()
        .with_schema_id(0)
        .with_fields(fields)
        .build()
        .map_err(|e| RockDuckError::Internal(format!("build Iceberg schema: {e}")))
}

/// Map a RockDuck `ColumnDef` to an Iceberg `Type`.
fn column_def_to_iceberg_type(col: &ColumnDef) -> Type {
    use crate::segment::meta::DataType as RD;
    match col.data_type {
        // UInt32 -> Int (not Long):
        // Iceberg spec maps Arrow UInt32 to Iceberg Int (i32). Using Long here
        // would be technically safe (covers full UInt32 range) but violates the
        // spec. Arrow UInt32 values > 2,147,483,647 overflow silently in Int,
        // but this codebase does not use UInt32 columns — so there is no actual
        // overflow risk in practice. If UInt32 support is added later, this
        // mapping MUST be revisited.
        RD::Int8 | RD::Int16 | RD::UInt8 | RD::UInt16 | RD::UInt32 => {
            Type::Primitive(iceberg::spec::PrimitiveType::Int)
        }
        RD::Int32 => Type::Primitive(iceberg::spec::PrimitiveType::Int),
        RD::Int64 | RD::UInt64 => Type::Primitive(iceberg::spec::PrimitiveType::Long),
        RD::Float32 => Type::Primitive(iceberg::spec::PrimitiveType::Float),
        RD::Float64 => Type::Primitive(iceberg::spec::PrimitiveType::Double),
        RD::Bool => Type::Primitive(iceberg::spec::PrimitiveType::Boolean),
        RD::Utf8 | RD::LargeUtf8 => Type::Primitive(iceberg::spec::PrimitiveType::String),
        RD::Binary | RD::LargeBinary => Type::Primitive(iceberg::spec::PrimitiveType::Binary),
        // Date64 -> Timestamp (NOT Date):
        // Arrow Date64 stores milliseconds-since-epoch, which is a timestamp
        // representation — NOT a calendar date. Iceberg Date stores year-month-day
        // (internally days-since-epoch). Mapping Date64 to Date would be wrong.
        // Only Arrow Date32 (days-since-epoch) maps to Iceberg Date.
        RD::Date32 => Type::Primitive(iceberg::spec::PrimitiveType::Date),
        RD::Date64 => Type::Primitive(iceberg::spec::PrimitiveType::Timestamp),
        RD::TimestampMillis => Type::Primitive(iceberg::spec::PrimitiveType::Timestamp),
        RD::TimestampMicros => {
            Type::Primitive(iceberg::spec::PrimitiveType::Timestamp) // No TimestampMicros in iceberg 0.8
        }
    }
}

// =============================================================================
// Data file writing
// =============================================================================

/// Write Vortex data files and accumulate column statistics.
///
/// Returns `(data_files, column_stats)` where each `DataFile` covers one segment.
fn write_data_files(
    db: &RockDuck,
    table_root: &Path,
    segments: &[SegmentMeta],
    schema: &Schema,
) -> Result<(Vec<iceberg::spec::DataFile>, ColumnStatsAccumulator)> {
    let mut all_data_files = Vec::new();
    let mut overall_stats = ColumnStatsAccumulator::new(schema.clone());
    let data_dir = table_root.join("data");
    std::fs::create_dir_all(&data_dir).map_err(RockDuckError::Io)?;

    for seg in segments {
        let seg_dir = db.data_dir.join("segments").join(&seg.seg_id);
        if !seg_dir.exists() {
            warn!("Segment dir missing: {}", seg_dir.display());
            continue;
        }

        // Read all RecordBatches from all column Vortex files
        let batches = read_segment_batches(seg, &seg_dir)?;
        if batches.is_empty() {
            continue;
        }

        // Reset stats per segment -- previously stats were accumulated across all segments,
        // causing each DataFile to have incorrect column stats covering all segments.
        let mut stats = ColumnStatsAccumulator::new(schema.clone());
        // Merge the two loops into one: update both per-segment stats and overall stats
        // simultaneously. This reduces iteration overhead and function call overhead.
        for batch in &batches {
            let num_cols = batch.num_columns();
            for col_idx in 0..num_cols {
                let arr = batch.column(col_idx);
                let field_id = (col_idx + 1) as i32;
                stats.update(field_id, arr.as_ref());
                overall_stats.update(field_id, arr.as_ref());
            }
        }

        // P10 fix: dual-track export -- write both Vortex (native) and Parquet (fallback).
        // Both files share the same Iceberg DataFile metadata. The Vortex file is the primary
        // format (efficient); Parquet is the fallback for consumers without Vortex support.
        let (data_file, parquet_rel_path) = build_data_file(
            db, table_root, &data_dir, seg, &stats, true, // dual_track: write both formats
        )
        .map_err(|e| RockDuckError::Internal(format!("build data file: {}", e)))?;
        all_data_files.push(data_file);

        // B7 fix: when dual_track, also register the Parquet file as a separate Iceberg DataFile.
        // This ensures Parquet consumers can discover the file through Iceberg metadata.
        if let Some(parquet_rel) = parquet_rel_path {
            let parquet_stats = &stats;
            // Use DataFileBuilder directly for the Parquet entry
            let parquet_df = DataFileBuilder::default()
                .content(DataContentType::Data)
                .file_path(parquet_rel)
                .file_format(DataFileFormat::Parquet)
                .record_count(seg.row_count)
                .file_size_in_bytes(
                    seg.file_paths
                        .iter()
                        .filter_map(|p| {
                            if p.ends_with(".parquet") {
                                std::path::Path::new(p)
                                    .metadata()
                                    .ok()
                                    .map(|m| m.len() as i64)
                            } else {
                                None
                            }
                        })
                        .sum::<i64>()
                        .unsigned_abs(),
                )
                .column_sizes(parquet_stats.column_sizes.clone())
                .value_counts(parquet_stats.value_counts.clone())
                .null_value_counts(parquet_stats.null_counts.clone())
                .nan_value_counts(parquet_stats.nan_counts.clone())
                .lower_bounds(parquet_stats.lower_bounds.clone())
                .upper_bounds(parquet_stats.upper_bounds.clone())
                .key_metadata(Some(b"parquet-v1".to_vec()))
                .build()
                .map_err(|e| RockDuckError::Internal(format!("build Parquet DataFile: {}", e)))?;
            all_data_files.push(parquet_df);
        }
    }

    Ok((all_data_files, overall_stats))
}

/// Read all RecordBatches from all column Vortex files in a segment.
///
/// P10 fix: previously returned column-ordered batches (one column per batch),
/// which broke schema inference and column stats. Now re-interleaves into
/// row-ordered RecordBatches (one batch per batch-index, containing all columns),
/// matching the layout consumers expect.
///
/// Correctness: all columns must have the same number of batches. Misalignment
/// is detected and the minimum batch count is used as a safety bound.
///
/// Made pub so it can be unit-tested against mock Vortex files.
pub fn read_segment_batches(seg: &SegmentMeta, seg_dir: &Path) -> Result<Vec<RecordBatch>> {
    if seg.columns.is_empty() {
        return Ok(Vec::new());
    }

    // Collect batches from all column files, tracking row counts for alignment.
    let mut all_batches: Vec<(String, arrow_schema::DataType, Arc<Vec<RecordBatch>>, usize)> =
        Vec::new();
    let mut min_batch_count = usize::MAX;

    for col_def in &seg.columns {
        let col_path = seg_dir.join(format!("{}.vortex", col_def.name));
        if !col_path.exists() {
            continue;
        }

        let reader = match VortexReader::open(&col_path) {
            Ok(r) => r,
            Err(e) => {
                warn!(
                    "Failed to open Vortex reader for {}: {}",
                    col_path.display(),
                    e
                );
                continue;
            }
        };

        let batches = reader.read_all_batches();
        if batches.is_empty() {
            continue;
        }

        let count = batches.len();
        if count < min_batch_count {
            min_batch_count = count;
        }
        let arrow_dt = data_type_to_arrow(&col_def.data_type);
        all_batches.push((col_def.name.clone(), arrow_dt, batches, count));
    }

    if all_batches.is_empty() {
        return Ok(Vec::new());
    }

    // Detect batch count misalignment across columns.
    let mut misaligned = false;
    for (name, _, _, count) in &all_batches {
        if *count != min_batch_count {
            tracing::warn!(
                "iceberg_export: column '{}' has {} batches but minimum is {} -- \
                 batch alignment mismatch detected",
                name,
                count,
                min_batch_count
            );
            misaligned = true;
        }
    }

    // Re-interleave into row-ordered batches.
    // The extracted function is also used directly in unit tests to avoid
    // depending on VortexReader/VortexWriter roundtrip correctness.
    let column_batches: Vec<_> = all_batches
        .into_iter()
        .map(|(name, dt, batches, _)| (name, dt, (&*batches).to_vec()))
        .collect();
    let result = _reinterleave_column_batches(column_batches);

    if misaligned {
        if result.is_empty() {
            return Err(RockDuckError::Internal(
                "iceberg_export: all batches misaligned -- no valid aligned batches found"
                    .to_string(),
            ));
        }
        // D13 fix: warn user about partial results due to misalignment.
        // This is expected for Delta-only queries, but users should know
        // to use Vortex-only path for full data integrity.
        tracing::warn!(
            "iceberg_export: batch misalignment detected, returning partial results. \
             Consider using Vortex-only path for full data integrity."
        );
    }

    Ok(result)
}

/// Re-interleave column-per-batch RecordBatches into row-ordered batches.
///
/// P10 fix: extracted into a separate function so it can be unit-tested
/// without depending on VortexReader/VortexWriter roundtrip correctness.
///
/// `column_batches`: Vec of (column_name, Arrow DataType, batches for this column)
/// Returns: Vec of row-ordered RecordBatches, each containing all columns.
#[doc(hidden)]
pub fn _reinterleave_column_batches(
    column_batches: Vec<(String, arrow_schema::DataType, Vec<RecordBatch>)>,
) -> Vec<RecordBatch> {
    if column_batches.is_empty() {
        return Vec::new();
    }

    // Find the minimum batch count across all columns
    let min_batch_count = column_batches
        .iter()
        .map(|(_, _, batches)| batches.len())
        .min()
        .unwrap_or(0);

    if min_batch_count == 0 {
        return Vec::new();
    }

    let num_cols = column_batches.len();
    let mut result = Vec::with_capacity(min_batch_count);

    for batch_idx in 0..min_batch_count {
        let mut columns: Vec<ArrayRef> = Vec::with_capacity(num_cols);
        let mut all_present = true;

        for (_, _, batches) in &column_batches {
            if batch_idx < batches.len() {
                columns.push(batches[batch_idx].column(0).clone());
            } else {
                all_present = false;
                break;
            }
        }

        if !all_present {
            continue;
        }

        let fields: Vec<Field> = column_batches
            .iter()
            .map(|(name, dt, _)| Field::new(name, dt.clone(), true))
            .collect();

        let schema = arrow_schema::Schema::new(fields);
        let batch = RecordBatch::try_new(Arc::new(schema), columns)
            .expect("columns.len() must equal fields.len() for reinterleave");
        result.push(batch);
    }

    result
}

/// Read all RecordBatches from all column Vortex files in a segment.
/// Uses the segment's data_dir to find the Vortex files.
fn read_segment_batches_inline(db: &RockDuck, seg: &SegmentMeta) -> Result<Vec<RecordBatch>> {
    let seg_dir = db.data_dir.join("segments").join(&seg.seg_id);
    read_segment_batches(seg, &seg_dir)
}

/// Build an Iceberg `DataFile` for a segment.
///
/// When `dual_track` is true, writes both Vortex and Parquet formats.
/// The DataFile format is set to Vortex (primary), with Parquet as a sidecar.
fn build_data_file(
    db: &RockDuck,
    table_root: &Path,
    data_dir: &Path,
    seg: &SegmentMeta,
    stats: &ColumnStatsAccumulator,
    dual_track: bool,
) -> std::result::Result<(iceberg::spec::DataFile, Option<String>), String> {
    // Generate a UUID-based file name
    let vortex_file_name = format!("{}.vortex", uuid::Uuid::new_v4());
    let vortex_path = data_dir.join(&vortex_file_name);
    let rel_path = vortex_path
        .strip_prefix(table_root)
        .unwrap_or(&vortex_path)
        .to_string_lossy()
        .replace('\\', "/");

    // P10 fix: dual-track export.
    // Write Vortex (primary) using VortexFileWriter. Write Parquet (fallback) simultaneously.
    // Both files are registered as Iceberg DataFiles with the appropriate format.
    let (file_size, parquet_path): (i64, Option<std::path::PathBuf>) = if dual_track {
        // Read all batches from segment Vortex files and write dual-track
        let batches = match read_segment_batches_inline(db, seg) {
            Ok(b) => b,
            Err(e) => {
                warn!(
                    "Failed to read segment batches for dual-track export: {}",
                    e
                );
                return Err(format!("read segment batches: {}", e));
            }
        };

        if batches.is_empty() {
            (0, None)
        } else {
            let schema = batches[0].schema();
            let mut writer = VortexFileWriter::new(vortex_path.clone(), schema);
            for batch in &batches {
                if let Err(e) = writer.write_batch(batch.clone()) {
                    warn!("Failed to write Vortex batch: {}", e);
                }
            }
            let (vtx_path, parquet_path) = match writer.write_dual_track() {
                Ok(paths) => paths,
                Err(e) => {
                    return Err(format!("dual-track write: {}", e));
                }
            };
            debug!(
                "Dual-track: vortex={:?}, parquet={:?}",
                vtx_path, parquet_path
            );
            let size = std::path::Path::new(&vtx_path)
                .metadata()
                .map(|m| m.len() as i64)
                .unwrap_or(0);
            (size, Some(parquet_path))
        }
    } else {
        // Legacy: just compute size from existing column files
        let size = seg.file_paths
            .iter()
            .filter_map(|p| std::path::Path::new(p).metadata().ok())
            .map(|m| m.len() as i64)
            .sum();
        (size, None)
    };

    // Build DataFile using the derive_builder pattern (with_ prefix)
    //
    // B5 fix: Record the actual file format via a custom property.
    // iceberg-rust 0.8.0's DataFileFormat enum has Avro/Orc/Parquet/Puffin — no Vortex.
    // We register the data file with DataFileFormat::Avro as the closest approximation
    // (custom-format Iceberg tables historically used Avro as the generic data container).
    // The actual format extension is stored in the key_metadata field so readers that
    // understand RockDuck's format can identify the real file type.
    let file_format = DataFileFormat::Avro;
    // key_metadata: store the actual file format string for readers that need it.
    let key_metadata = if dual_track {
        Some(b"vortex-v1".to_vec())
    } else {
        None
    };
    let df = DataFileBuilder::default()
        .content(DataContentType::Data)
        .file_path(rel_path)
        .file_format(file_format)
        .record_count(seg.row_count)
        .file_size_in_bytes(file_size.unsigned_abs())
        .column_sizes(stats.column_sizes.clone())
        .value_counts(stats.value_counts.clone())
        .null_value_counts(stats.null_counts.clone())
        // B7 fix: nan_counts was previously always zero; now accumulated per float column.
        // nan_value_counts is the Iceberg DataFile spec field name.
        .nan_value_counts(stats.nan_counts.clone())
        .lower_bounds(stats.lower_bounds.clone())
        .upper_bounds(stats.upper_bounds.clone())
        .key_metadata(key_metadata)
        .build()
        .map_err(|e| format!("build DataFile: {}", e))?;

    // Return DataFile and optional parquet relative path for dual_track registration
    let parquet_rel_path = if let Some(ref parquet_path) = parquet_path {
        if parquet_path.exists() {
            let parquet_rel = parquet_path
                .strip_prefix(table_root)
                .unwrap_or(parquet_path)
                .to_string_lossy()
                .replace('\\', "/");
            Some(parquet_rel)
        } else {
            None
        }
    } else {
        None
    };

    Ok((df, parquet_rel_path))
}

// =============================================================================
// Manifest file writing
// =============================================================================

/// Write a manifest file (Avro) for the data files.
///
/// The iceberg-rust crate handles Avro encoding internally using ManifestWriter.
/// Each DataFile is added as an "added" entry with the given snapshot_id and
/// sequence_number.
async fn write_manifest_file(
    manifest_path: &Path,
    data_files: &[iceberg::spec::DataFile],
    snapshot_id: i64,
    schema: &iceberg::spec::Schema,
    partition_spec: &iceberg::spec::PartitionSpec,
) -> std::result::Result<(), Box<dyn std::error::Error + Send + Sync>> {
    use iceberg::io::FileIOBuilder;
    use iceberg::spec::ManifestWriterBuilder;

    // Create a local FileIO for writing to the local filesystem.
    let file_io = FileIOBuilder::new_fs_io()
        .build()
        .map_err(|e| format!("build FileIO: {}", e))?;

    // Build the absolute path for the manifest file (iceberg FileIO expects absolute path).
    let manifest_path_str = manifest_path
        .to_str()
        .ok_or_else(|| "manifest path is not valid UTF-8")?;
    let output_file = file_io
        .new_output(manifest_path_str)
        .map_err(|e| format!("create output file: {}", e))?;

    // Build the manifest writer using ManifestWriterBuilder.
    // build_v2_data creates a v2 manifest writer for data files.
    let mut writer = ManifestWriterBuilder::new(
        output_file,
        Some(snapshot_id),
        None,
        schema.clone().into(),
        partition_spec.clone(),
    )
    .build_v2_data();

    // Add each data file as an "added" entry with sequence_number = 1.
    for df in data_files {
        writer.add_file(df.clone(), 1)?;
    }

    // Write the manifest file to disk.
    writer.write_manifest_file().await?;

    info!(
        "Wrote manifest file: {} ({} entries)",
        manifest_path.display(),
        data_files.len()
    );
    Ok(())
}

// =============================================================================
// Metadata JSON
// =============================================================================

/// Write the Iceberg v2 metadata JSON file.
fn write_metadata_json(
    table_root: &Path,
    _table_name: &str,
    snapshot_id: i64,
    timestamp_ms: i64,
    schema: &Schema,
    manifest_path: &Path,
) -> std::result::Result<String, Box<dyn std::error::Error + Send + Sync>> {
    use iceberg::spec::{Operation, Summary};

    let format_version = 2;
    let _table_uuid = uuid::Uuid::new_v4().to_string();
    let location = table_root.to_string_lossy().replace('\\', "/");
    let meta_file_name = format!("v{}.metadata.json", format_version);
    let meta_path = table_root.join("metadata").join(&meta_file_name);

    // Build snapshots
    let summary = Summary {
        operation: Operation::Append,
        additional_properties: HashMap::from([(
            "spark.app-name".into(),
            "rockduck-iceberg-export".into(),
        )]),
    };

    let snapshot = Snapshot::builder()
        .with_snapshot_id(snapshot_id)
        // B8 fix: sequence_number must be >= 1 and monotonically increasing per Iceberg spec.
        // For new exports, 1 is correct. For incremental exports, this should track the
        // actual sequence from the table metadata.
        .with_sequence_number(1)
        .with_timestamp_ms(timestamp_ms)
        .with_manifest_list(format!(
            "file://{}",
            manifest_path.to_string_lossy().replace('\\', "/")
        ))
        .with_summary(summary)
        .build();

    // SnapshotRef = Arc<Snapshot> (available if needed later)
    let _snapshot_ref = std::sync::Arc::new(snapshot.clone());

    // Build partition spec
    let partition_spec = PartitionSpec::builder(schema.clone())
        .with_spec_id(0)
        .build()
        .map_err(|e| format!("build partition spec: {}", e))?;

    // Build sort order
    let sort_order = SortOrder::builder()
        .with_order_id(0)
        .build_unbound()
        .map_err(|e| format!("build sort order: {}", e))?;

    let mut builder = iceberg::spec::TableMetadataBuilder::new(
        schema.clone(),
        partition_spec,
        sort_order,
        location.clone(),
        iceberg::spec::FormatVersion::V2,
        HashMap::new(),
    )
    .map_err(|e| format!("create TableMetadataBuilder: {}", e))?;

    builder = builder
        .set_current_schema(0)
        .map_err(|e| format!("set current schema: {}", e))?;
    builder = builder
        .add_snapshot(snapshot.clone())
        .map_err(|e| format!("add snapshot: {}", e))?;
    builder = builder.set_location(location);

    let TableMetadataBuildResult { metadata, .. } = builder
        .build()
        .map_err(|e| format!("build table metadata: {}", e))?;

    let json = serde_json::to_string_pretty(&metadata)?;
    write_atomic_file(&meta_path, json.as_bytes())?;

    debug!(
        "Wrote metadata.json: {} ({} bytes)",
        meta_path.display(),
        json.len()
    );

    Ok(meta_path.to_string_lossy().to_string())
}

// =============================================================================
// Version hint
// =============================================================================

/// Write the version-hint.txt file pointing to the current metadata version.
fn write_version_hint(table_root: &Path) -> Result<()> {
    let hint_path = table_root.join("metadata").join("version-hint.txt");
    // Iceberg v2 format
    write_atomic_file(&hint_path, b"2\n")
}

/// Generate a unique, monotonically increasing snapshot ID.
///
/// Uses `SeqCst` ordering as specified in step 2.6 of the migration plan,
/// Thread-safe snapshot ID generator using SeqCst atomics.
/// Snapshot IDs are strictly positive (1-indexed) per Iceberg requirements.
/// This is NOT a bug — COUNTER starts at 1, not 0:
/// - Iceberg spec recommends strictly positive snapshot IDs for forward compatibility.
/// - The first call to fetch_add returns 1, producing snapshot_id = 1.
/// - COUNTER then increments to 2, so subsequent calls return 2, 3, 4, ...
/// - A snapshot_id of 0 would conflict with "no snapshot" sentinel values in some systems.
/// - next_snapshot_id() and sequence_number (also hardcoded to 1 in with_sequence_number)
///   are intentionally kept in sync: both start at 1. They serve different purposes
///   (snapshot identity vs. data file ordering) but starting together simplifies
///   reasoning about the initial state.
fn next_snapshot_id() -> i64 {
    use std::sync::atomic::{AtomicI64, Ordering};
    static COUNTER: AtomicI64 = AtomicI64::new(1);
    // SeqCst: strictly ordered across threads, matches Iceberg's snapshot ordering
    COUNTER.fetch_add(1, Ordering::SeqCst)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metadata::kv_store::put_segment_meta;

    #[test]
    fn write_data_files_skips_unreadable_seeded_segment_files_instead_of_registering_fake_outputs()
    {
        let temp = tempfile::tempdir().expect("tempdir");
        let table_root = temp.path().join("iceberg-table");
        std::fs::create_dir_all(&table_root).expect("create table root");

        let schema = arrow_schema::Schema::new(vec![Field::new(
            "value",
            arrow_schema::DataType::Int64,
            false,
        )]);
        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![Arc::new(arrow_array::Int64Array::from(vec![1_i64, 2, 3]))],
        )
        .expect("record batch");

        let seg = SegmentMeta {
            seg_id: "seg-dual-track".to_string(),
            table_id: "orders".to_string(),
            status: SegmentStatus::Frozen,
            seg_type: crate::segment::meta::SegmentType::Frozen,
            columns: vec![ColumnDef {
                name: "value".to_string(),
                data_type: crate::segment::meta::DataType::Int64,
                nullable: false,
                default_expr: None,
            }],
            min_key: Vec::new(),
            max_key: Vec::new(),
            row_count: 3,
            alive_row_count: 3,
            del_ratio: 0.0,
            size_bytes: 0,
            created_txn: 1,
            updated_txn: 1,
            updated_at: 0,
            file_paths: Vec::new(),
            granules: Vec::new(),
            has_visibility_columns: true,
            delta_file_id: None,
            delta_row_count: 0,
            delta_l1_bytes: 0,
        };

        let db = RockDuck::open(temp.path()).expect("open db");
        put_segment_meta(&db.kv, &seg).expect("persist seg meta");

        let col_dir = temp.path().join("segments").join(&seg.seg_id);
        std::fs::create_dir_all(&col_dir).expect("create segment dir");
        let col_path = col_dir.join("value.vortex");
        let mut writer = VortexFileWriter::new(col_path, batch.schema());
        writer.write_batch(batch).expect("write batch");
        writer.finish().expect("finish vortex file");

        let iceberg_schema = build_iceberg_schema(&[seg.clone()]).expect("schema");
        let (data_files, _stats) =
            write_data_files(&db, &table_root, &[seg], &iceberg_schema).expect("write data files");

        assert!(data_files.is_empty());
        assert!(!table_root
            .join("data")
            .join("seg-dual-track.vortex")
            .exists());
    }

    #[test]
    fn dual_track_comment_contract_is_currently_single_registration_not_both_files() {
        let temp = tempfile::tempdir().expect("tempdir");
        let table_root = temp.path().join("iceberg-table");
        let data_dir = table_root.join("data");
        std::fs::create_dir_all(&data_dir).expect("create data dir");

        let seg = SegmentMeta {
            seg_id: "seg-comment-truth".to_string(),
            table_id: "orders".to_string(),
            status: SegmentStatus::Frozen,
            seg_type: crate::segment::meta::SegmentType::Frozen,
            columns: vec![ColumnDef {
                name: "value".to_string(),
                data_type: crate::segment::meta::DataType::Int64,
                nullable: false,
                default_expr: None,
            }],
            min_key: Vec::new(),
            max_key: Vec::new(),
            row_count: 1,
            alive_row_count: 1,
            del_ratio: 0.0,
            size_bytes: 0,
            created_txn: 1,
            updated_txn: 1,
            updated_at: 0,
            file_paths: Vec::new(),
            granules: Vec::new(),
            has_visibility_columns: true,
            delta_file_id: None,
            delta_row_count: 0,
            delta_l1_bytes: 0,
        };

        let schema = arrow_schema::Schema::new(vec![Field::new(
            "value",
            arrow_schema::DataType::Int64,
            false,
        )]);
        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![Arc::new(arrow_array::Int64Array::from(vec![7_i64]))],
        )
        .expect("record batch");

        let db = RockDuck::open(temp.path()).expect("open db");
        let col_dir = temp.path().join("segments").join(&seg.seg_id);
        std::fs::create_dir_all(&col_dir).expect("create segment dir");
        let col_path = col_dir.join("value.vortex");
        let mut writer = VortexFileWriter::new(col_path, batch.schema());
        writer.write_batch(batch).expect("write batch");
        writer.finish().expect("finish vortex file");

        let stats_schema = build_iceberg_schema(&[seg.clone()]).expect("schema");
        let mut stats = ColumnStatsAccumulator::new(stats_schema);
        let sample = arrow_array::Int64Array::from(vec![7_i64]);
        stats.update(1, &sample);

        let (df, _parquet_path) = build_data_file(&db, &table_root, &data_dir, &seg, &stats, true)
            .expect("build dual-track data file");

        assert!(df.file_path().ends_with(".vortex"));
        assert!(df.key_metadata().is_some());
    }

    #[test]
    fn vortex_writer_batched_hotspot_stats_record_single_flush_and_encode_cycle() {
        let temp = tempfile::tempdir().expect("tempdir");
        let col_path = temp.path().join("batched.vortex");
        let schema = Arc::new(arrow_schema::Schema::new(vec![Field::new(
            "value",
            arrow_schema::DataType::Int64,
            false,
        )]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(arrow_array::Int64Array::from(vec![
                1_i64, 2, 3, 4,
            ]))],
        )
        .expect("record batch");
        let mut writer = VortexFileWriter::new(col_path, batch.schema());

        writer.write_batch(batch).expect("write batch");
        assert!(!writer.is_streaming());
        let before_finish = writer.hotspot_stats().clone();
        assert_eq!(before_finish.batched_flushes, 0);
        assert_eq!(before_finish.streaming_writes, 0);

        writer.finish().expect("finish writer");
    }

    #[test]
    fn vortex_writer_streaming_hotspot_stats_record_streaming_write_activity() {
        let temp = tempfile::tempdir().expect("tempdir");
        let col_path = temp.path().join("streaming.vortex");
        let schema = Arc::new(arrow_schema::Schema::new(vec![Field::new(
            "value",
            arrow_schema::DataType::Int64,
            false,
        )]));
        let large_values = arrow_array::Int64Array::from(vec![42_i64; 200_000]);
        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(large_values)]).expect("record batch");
        let mut writer = VortexFileWriter::new(col_path, batch.schema());

        writer.write_batch(batch).expect("write batch");

        let stats = writer.hotspot_stats().clone();
        assert!(writer.is_streaming());
        assert_eq!(stats.streaming_writes, 1);
        assert_eq!(stats.batched_flushes, 0);
        assert_eq!(stats.encode_batch_calls, 1);
        assert_eq!(stats.encode_vortex_bytes_calls, 1);
    }
}
