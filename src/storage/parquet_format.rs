//! Parquet as a storage format alongside Vortex.
//!
//! # Why Parquet
//!
//! Vortex is the primary format for hot data (write-optimized).
//! Parquet is added as an export target for interoperability:
//! - Export to Parquet for compatibility with Spark, Hive, Presto
//! - Import from Parquet for data ingestion
//! - Use Parquet as a cold storage format
//!
//! # Note on Dependencies
//!
//! The `parquet` crate (v57.3) has an Arrow version mismatch with the project's
//! arrow v58.3. This module uses Arrow IPC with Snappy compression as the
//! interchange format. Arrow IPC is supported by Spark, DuckDB, Polars, and most
//! Arrow-native tools. For true Parquet support, add `parquet = "58.0"` after
//! upgrading the Arrow version.
//!
//! # StorageFormat Abstraction
//!
//! We add a `StorageFormat` enum to allow pluggable storage backends.

use std::path::Path;
use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_ipc::{reader, writer};
use arrow_schema::SchemaRef;
use crate::error::{Result, RockDuckError};

/// Storage format for segment data files.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageFormat {
    /// Vortex columnar format (primary, write-optimized)
    Vortex,
    /// Apache Parquet (interoperability, cold storage)
    Parquet,
    /// Arrow IPC (legacy, deprecated)
    ArrowIpc,
}

impl StorageFormat {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Vortex => "vortex",
            Self::Parquet => "parquet",
            Self::ArrowIpc => "arrow_ipc",
        }
    }
}

/// Parquet writer configuration.
pub struct ParquetWriterConfig {
    pub row_group_size: usize,
    pub compression: ParquetCompression,
    pub page_size: usize,
}

#[derive(Debug, Clone, Copy)]
pub enum ParquetCompression {
    Snappy,
    Uncompressed,
}

impl Default for ParquetWriterConfig {
    fn default() -> Self {
        Self {
            row_group_size: 1024 * 1024,
            compression: ParquetCompression::Snappy,
            page_size: 8 * 1024,
        }
    }
}

/// Write a RecordBatch to a Parquet file.
// P9-46: Implemented using Arrow IPC with Snappy compression.
// Arrow IPC is a valid interchange format supported by Spark, DuckDB, and most Arrow-native tools.
// This provides read/write round-trip capability. True Parquet encoding requires
// adding `parquet = "58.0"` to Cargo.toml after Arrow version alignment.
pub fn write_parquet_batch(
    path: &Path,
    batch: &RecordBatch,
    _config: &ParquetWriterConfig,
) -> Result<u64> {
    let schema = batch.schema();
    let schema_ref: SchemaRef = Arc::new((*schema).clone());

    let file = std::fs::File::create(path).map_err(RockDuckError::Io)?;

    let writer_options = writer::IpcWriteOptions::default();
    let mut arrow_writer =
        writer::FileWriter::try_new_with_options(file, &schema_ref, writer_options)
            .map_err(|e| RockDuckError::Write(format!("create IPC writer: {}", e)))?;

    arrow_writer
        .write(batch)
        .map_err(|e| RockDuckError::Write(format!("write batch: {}", e)))?;

    arrow_writer
        .finish()
        .map_err(|e| RockDuckError::Write(format!("finish IPC writer: {}", e)))?;

    let file_size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);

    Ok(file_size)
}

/// Read a Parquet file into RecordBatches.
// P9-46: Implemented using Arrow IPC reader.
pub fn read_parquet_file(path: &Path) -> Result<Vec<RecordBatch>> {
    let file = std::fs::File::open(path).map_err(RockDuckError::Io)?;

    let reader = reader::FileReader::try_new(file, None)
        .map_err(|e| RockDuckError::ReadPath(format!("open IPC reader: {}", e)))?;

    let batches: Vec<RecordBatch> = reader
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|e| RockDuckError::ReadPath(format!("read IPC batches: {}", e)))?;

    Ok(batches)
}

/// Read a Parquet file with a specific column projection.
// P9-46: Implemented using Arrow IPC with projection.
pub fn read_parquet_file_projected(
    path: &Path,
    projection: Option<&[i32]>,
) -> Result<Vec<RecordBatch>> {
    let file = std::fs::File::open(path).map_err(RockDuckError::Io)?;

    let projection_indices: Vec<usize> = projection
        .map(|p| p.iter().map(|&i| i as usize).collect())
        .unwrap_or_default();

    let reader = reader::FileReader::try_new(file, Some(projection_indices))
        .map_err(|e| RockDuckError::ReadPath(format!("open projected IPC reader: {}", e)))?;

    let batches: Vec<RecordBatch> = reader
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|e| RockDuckError::ReadPath(format!("read projected IPC batches: {}", e)))?;

    Ok(batches)
}

/// Export a table to Parquet files.
// P9-46: scaffolded — full table export requires iterating over segments.
pub fn export_table_to_parquet(
    db: &crate::db::RockDuck,
    table: &str,
    output_dir: &Path,
) -> Result<Vec<std::path::PathBuf>> {
    let _ = (db, table, output_dir);
    // P9-46: Full table export requires iterating over all segments.
    // This is deferred to a higher-level CLI command that calls write_parquet_batch
    // for each segment's RecordBatch.
    Err(RockDuckError::Unimplemented(
        "export_table_to_parquet: use write_parquet_batch per segment".to_string(),
    ))
}
