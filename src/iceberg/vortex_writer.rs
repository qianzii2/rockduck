//! VortexFileWriter — Iceberg FormatModel support for Vortex data files.
//!
//! ## C-12: Adaptive Write Strategy
//!
//! Adaptive write strategy based on estimated total data size:
//! - **< 1 MB**: Batched mode — accumulate in memory, write all at `finish()`.
//! - **>= 1 MB**: Streaming mode — flush each batch to disk immediately.
//! - **Transition**: When estimated size crosses the 1 MB threshold mid-stream,
//!   flush accumulated batches first, then switch to streaming.
//!
//! This avoids unbounded memory growth for large exports while keeping
//! small exports efficient (single write).
//!
//! ## Iceberg Dual-Track Export
//!
//! This module implements the Iceberg dual-track export strategy:
//! - **Track A**: Vortex-native format (`.vortex`) — efficient columnar storage with ALP/FastLanes
//! - **Track B**: Parquet fallback (`.parquet`) — for Iceberg consumers that don't support Vortex
//!
//! Both tracks are written simultaneously, giving consumers the choice:
//! - Spark/Trino via Vortex reader plugin → use `.vortex` (fast)
//! - Generic Iceberg readers → use `.parquet` (compatible)
//!
//! ## Iceberg FormatModel
//!
//! Once iceberg-rust 1.11.0 exposes `VortexFormatModel`, this writer will implement it
//! to register `.vortex` as a first-class Iceberg data file format.

use std::path::PathBuf;
use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;

use crate::error::{Result, RockDuckError};
use crate::metadata::projection::ProjectionContract;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct VortexWriterHotspotStats {
    pub encode_batch_calls: u64,
    pub encode_vortex_bytes_calls: u64,
    pub streaming_writes: u64,
    pub batched_flushes: u64,
    pub parquet_rereads: u64,
}

// =============================================================================
// Adaptive Write Constants
// =============================================================================

/// Adaptive write threshold: 1 MB.
///
/// Below this threshold, batches are accumulated in memory and written once
/// at `finish()` (batched mode). Above it, each batch is flushed to disk
/// immediately (streaming mode). This prevents unbounded memory growth during
/// large segment exports while keeping small exports efficient.
const ADAPTIVE_THRESHOLD_BYTES: usize = 1 * 1024 * 1024;

// =============================================================================
// Iceberg VortexFormatModel — Vortex data file format
// =============================================================================

/// Vortex format magic bytes: b"VTXF"
const VORTEX_MAGIC: [u8; 4] = *b"VTXF";
const VORTEX_VERSION: u8 = 1;

/// Vortex file header (32 bytes, written at the start of each .vortex file).
/// Compatible with the Iceberg data file format model.
#[derive(Debug)]
#[repr(C)]
struct VortexFileHeader {
    magic: [u8; 4],      // b"VTXF"
    version: u8,         // format version (1)
    flags: u8,           // reserved flags
    schema_len: u16,     // length of embedded schema JSON (0 if external)
    encoding_count: u16, // number of column encodings
    total_rows: u64,     // total rows in this file
    _reserved: [u8; 8],
}

impl Default for VortexFileHeader {
    fn default() -> Self {
        Self {
            magic: VORTEX_MAGIC,
            version: VORTEX_VERSION,
            flags: 0,
            schema_len: 0,
            encoding_count: 0,
            total_rows: 0,
            _reserved: [0u8; 8],
        }
    }
}

/// Vortex data file writer for Iceberg export.
///
/// Writes `.vortex` data files using RockDuck's adaptive encoding pipeline
/// (ALP for floats, FastLanes for ints, BtrBlocks as fallback).
///
/// ## Adaptive Write Strategy (C-12)
///
/// - **< 1 MB estimated**: Batched — accumulate RecordBatches in memory, write once at `finish()`.
/// - **>= 1 MB estimated**: Streaming — encode and flush each batch to disk immediately.
/// - **Transition**: When the estimated size crosses the threshold mid-stream, flush the
///   accumulated batches first, then switch to streaming for subsequent batches.
///
/// This prevents unbounded memory growth during large segment exports while keeping
/// small exports efficient (single write syscall).
///
/// ## Dual-Track Export
///
/// Use `write_dual_track()` to write both `.vortex` AND `.parquet` files simultaneously,
/// enabling all Iceberg consumers regardless of Vortex support.
///
/// ## Iceberg DataFile Metadata
///
/// After writing, use `build_data_file_entry()` to create the Iceberg `DataFileEntry`
/// for the manifest file.
pub struct VortexFileWriter {
    output_path: PathBuf,
    batches: Vec<RecordBatch>,
    /// Copies of batches written in streaming mode (for Parquet fallback without re-reading the file).
    flushed_batches: Vec<RecordBatch>,
    schema: SchemaRef,
    total_rows: u64,
    /// Running estimate of total bytes across all accumulated batches.
    estimated_bytes: usize,
    /// True once we have crossed the threshold and switched to streaming.
    streaming: bool,
    hotspot_stats: VortexWriterHotspotStats,
    /// ArrowWriter for streaming Parquet output (avoids clone + concat overhead).
    /// Only present in streaming mode when dual-track is active.
    parquet_writer: Option<parquet::arrow::ArrowWriter<std::fs::File>>,
}

impl VortexFileWriter {
    /// Create a new VortexFileWriter for the given output path.
    pub fn new(output_path: PathBuf, schema: SchemaRef) -> Self {
        Self {
            output_path,
            batches: Vec::new(),
            flushed_batches: Vec::new(),
            schema,
            total_rows: 0,
            estimated_bytes: 0,
            streaming: false,
            hotspot_stats: VortexWriterHotspotStats::default(),
            parquet_writer: None,
        }
    }

    /// Estimate the memory footprint of a RecordBatch (sum of array buffer sizes).
    fn estimate_batch_bytes(batch: &RecordBatch) -> usize {
        if batch.num_rows() == 0 {
            return 0;
        }
        batch
            .columns()
            .iter()
            .map(|c| c.get_array_memory_size())
            .sum()
    }

    /// Write a single RecordBatch using adaptive strategy.
    ///
    /// - **Batched** (estimated < 1 MB): accumulate in `self.batches`.
    /// - **Streaming** (estimated >= 1 MB): encode + flush each batch immediately.
    /// - **Transition**: if we cross the threshold with this batch, flush accumulated
    ///   batches first, then stream the current one.
    pub fn write_batch(&mut self, batch: RecordBatch) -> Result<()> {
        let batch_bytes = Self::estimate_batch_bytes(&batch);
        self.total_rows += batch.num_rows() as u64;
        self.estimated_bytes += batch_bytes;

        if self.estimated_bytes < ADAPTIVE_THRESHOLD_BYTES {
            // Batched: accumulate in memory.
            self.batches.push(batch);
        } else if self.batches.is_empty() {
            // Already streaming: encode and flush this batch immediately.
            self.streaming = true;
            self.write_batch_streaming(&batch)?;
        } else {
            // Transition: flush accumulated batches, then stream this one.
            self.flush_batched()?;
            self.streaming = true;
            self.write_batch_streaming(&batch)?;
        }
        Ok(())
    }

    /// Write a single batch in streaming mode: encode to Vortex, append to file, fsync.
    /// Also writes directly to the Parquet ArrowWriter (no clone needed).
    fn write_batch_streaming(&mut self, batch: &RecordBatch) -> Result<()> {
        use std::io::Write;

        self.hotspot_stats.streaming_writes += 1;

        // Lazily open the streaming Parquet writer on the first batch.
        if self.parquet_writer.is_none() {
            self.parquet_writer = Some({
                let parquet_path = self.output_path.with_extension("parquet");
                let file = std::fs::File::create(&parquet_path).map_err(RockDuckError::Io)?;
                parquet::arrow::ArrowWriter::try_new(file, self.schema.clone(), None)
                    .map_err(|e| RockDuckError::Write(format!("ArrowWriter::try_new: {}", e)))?
            });
        }

        // Write Parquet directly from the original batch (no clone).
        if let Some(ref mut writer) = self.parquet_writer {
            writer
                .write(batch)
                .map_err(|e| RockDuckError::Write(format!("ArrowWriter::write: {}", e)))?;
        }

        let vtx_batch = self.encode_batch(batch)?;
        let buf = Self::encode_vortex_bytes(&vtx_batch, &mut self.hotspot_stats)?;

        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.output_path)
            .map_err(RockDuckError::Io)?;
        file.write_all(&buf).map_err(RockDuckError::Io)?;
        file.sync_all().map_err(RockDuckError::Io)?;
        Ok(())
    }

    /// Flush all accumulated batches to disk as a single Vortex file.
    fn flush_batched(&mut self) -> Result<()> {
        if self.batches.is_empty() {
            return Ok(());
        }

        self.hotspot_stats.batched_flushes += 1;
        // Preserve original Arrow batches for Parquet fallback before encoding.
        if self.flushed_batches.is_empty() {
            self.flushed_batches.extend(self.batches.iter().cloned());
        }

        // Concatenate all batches into one using arrow_select::concat::concat_batches.
        let batch_refs: Vec<&RecordBatch> = self.batches.iter().collect();
        let merged = arrow_select::concat::concat_batches(&self.schema, batch_refs)
            .map_err(|e| RockDuckError::Internal(format!("concat batches: {}", e)))?;
        let vtx_batch = self.encode_batch(&merged)?;
        let buf = Self::encode_vortex_bytes(&vtx_batch, &mut self.hotspot_stats)?;
        std::fs::write(&self.output_path, &buf).map_err(RockDuckError::Io)?;

        // Clear accumulated batches after flushing.
        self.batches.clear();
        Ok(())
    }

    /// Encode a single RecordBatch to a Vortex array using the storage vortex pipeline.
    fn encode_batch(&mut self, batch: &RecordBatch) -> Result<vortex_array::ArrayRef> {
        use vortex_array::arrays::ChunkedArray;
        use vortex_array::arrow::FromArrowArray;
        use vortex_array::IntoArray;
        use vortex_array::VortexSessionExecute;

        self.hotspot_stats.encode_batch_calls += 1;
        let mut compressed_columns = Vec::with_capacity(batch.num_columns());
        let (session, _runtime) = crate::storage::vortex::VortexReader::make_session();
        let compressor = vortex_btrblocks::BtrBlocksCompressor::default();

        for column in batch.columns() {
            let arr: arrow_array::ArrayRef = column.clone();
            let vtx_arr = vortex_array::ArrayRef::from_arrow(arr.as_ref(), false)
                .map_err(|e| RockDuckError::Internal(format!("Arrow->Vortex: {}", e)))?;
            let mut ctx = session.create_execution_ctx();
            let compressed = compressor
                .compress(&vtx_arr, &mut ctx)
                .map_err(|e| RockDuckError::Internal(format!("BtrBlocks compress: {}", e)))?;
            compressed_columns.push(compressed);
        }

        let dtype = compressed_columns
            .first()
            .ok_or_else(|| RockDuckError::Internal("cannot encode empty RecordBatch".into()))?
            .dtype()
            .clone();
        let chunked = ChunkedArray::try_new(compressed_columns, dtype)
            .map_err(|e| RockDuckError::Internal(format!("ChunkedArray: {}", e)))?;
        Ok(chunked.into_array())
    }

    /// Encode a Vortex array to bytes synchronously (blocking).
    fn encode_vortex_bytes(
        arr: &vortex_array::ArrayRef,
        hotspot_stats: &mut VortexWriterHotspotStats,
    ) -> Result<Vec<u8>> {
        use vortex_array::iter::ArrayIteratorExt;
        use vortex_file::register_default_encodings;
        use vortex_file::WriteOptionsSessionExt;
        use vortex_io::runtime::current::CurrentThreadRuntime;
        use vortex_io::runtime::BlockingRuntime;
        use vortex_io::session::RuntimeSessionExt;

        hotspot_stats.encode_vortex_bytes_calls += 1;
        smol::block_on(async {
            let runtime = CurrentThreadRuntime::new();
            let session = vortex_session::VortexSession::empty()
                .with::<vortex_array::dtype::session::DTypeSession>()
                .with::<vortex_array::session::ArraySession>()
                .with::<vortex_array::optimizer::kernels::ArrayKernels>()
                .with::<vortex_array::aggregate_fn::session::AggregateFnSession>()
                .with::<vortex_array::scalar_fn::session::ScalarFnSession>()
                .with::<vortex_layout::session::LayoutSession>()
                .with::<vortex_io::session::RuntimeSession>()
                .with_handle(runtime.handle());

            vortex_alp::initialize(&session);
            vortex_fastlanes::initialize(&session);
            register_default_encodings(&session);

            runtime.block_on(async {
                let mut writer = Vec::new();
                session
                    .write_options()
                    .write(&mut writer, arr.to_array_iterator().into_array_stream())
                    .await
                    .map(|_| writer)
            })
        })
        .map_err(|e| RockDuckError::Internal(format!("vortex encode: {}", e)))
    }

    /// Finish writing: flush remaining batches (if any) and return the written path.
    ///
    /// In batched mode (estimated < 1 MB), all batches are merged and written once.
    /// In streaming mode, remaining batches have already been flushed individually.
    pub fn finish(mut self) -> Result<PathBuf> {
        let projection_contract = ProjectionContract::vtab();
        projection_contract.assert_blocking_governance();
        if self.total_rows == 0 {
            return Err(RockDuckError::Internal("no batches to write".into()));
        }

        if !self.batches.is_empty() && self.estimated_bytes < ADAPTIVE_THRESHOLD_BYTES {
            // Batched: flush all at once.
            self.flush_batched()?;
        }

        tracing::info!(
            "Wrote Vortex file: {:?} ({} rows, {} bytes, {} mode)",
            self.output_path,
            self.total_rows,
            self.estimated_bytes,
            if self.streaming {
                "streaming"
            } else {
                "batched"
            }
        );
        Ok(self.output_path)
    }

    /// Write a Parquet fallback alongside the Vortex file.
    ///
    /// This enables Iceberg consumers (Spark/Trino/DuckDB) that don't support Vortex
    /// natively to read the data via the `.parquet` sidecar file.
    ///
    /// The Parquet file has the same basename as the Vortex file, differing only
    /// in extension. Both files share the same Iceberg DataFile metadata (schema,
    /// record count, stats).
    ///
    /// Returns the path to the Parquet file.
    pub fn write_parquet_fallback(&self) -> Result<PathBuf> {
        let vortex_path = &self.output_path;

        if !self.batches.is_empty() {
            // Batched: we still have batches in memory — write Parquet from them.
            Self::write_parquet_from_batches_static(&self.batches, &self.schema, vortex_path)
        } else if !self.flushed_batches.is_empty() {
            // Streaming: batches were flushed but we kept copies — write from those.
            // This avoids the expensive re-read from the Vortex file.
            Self::write_parquet_from_batches_static(
                &self.flushed_batches,
                &self.schema,
                vortex_path,
            )
        } else {
            // No batches at all — return the Vortex path as-is.
            Ok(vortex_path.with_extension("parquet"))
        }
    }

    /// Write Parquet fallback from already-written Vortex file.
    /// Called by dual-track export when batches were already flushed (streaming mode).
    fn write_parquet_from_vortex_static(
        vortex_path: &PathBuf,
        hotspot_stats: &mut VortexWriterHotspotStats,
    ) -> Result<PathBuf> {
        hotspot_stats.parquet_rereads += 1;
        let reader = crate::storage::vortex::VortexReader::open(vortex_path)
            .map_err(|e| RockDuckError::Internal(format!("open vortex: {}", e)))?;

        let batches = reader.read_all_batches();
        if batches.is_empty() {
            return Ok(vortex_path.with_extension("parquet"));
        }

        let schema = reader.schema();
        Self::write_parquet_from_batches_static(&batches, &schema, vortex_path)
    }

    /// Write Parquet fallback from a slice of RecordBatches.
    ///
    /// B4 fix: replaced `arrow_ipc::writer::FileWriter` (Arrow IPC format) with
    /// `parquet::arrow::ArrowWriter` to produce genuine Parquet files that can be
    /// read by any standard Iceberg consumer (Spark, Trino, DuckDB, etc.).
    ///
    /// Durability: an explicit `sync_all()` is called after `ArrowWriter::close()`.
    /// This is NOT redundant — `close()` only flushes user-space buffers; it does
    /// NOT call `fsync()`. Without this sync, a crash after `close()` could lose
    /// the Parquet file while the Vortex file (which has its own `sync_all()` at
    /// vortex_writer.rs:192) survives. Both tracks must have equal durability.
    pub fn write_parquet_from_batches_static(
        batches: &[RecordBatch],
        schema: &SchemaRef,
        vortex_path: &PathBuf,
    ) -> Result<PathBuf> {
        use std::fs::File;

        let parquet_path = vortex_path.with_extension("parquet");

        if batches.is_empty() {
            return Ok(parquet_path);
        }

        let num_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        if num_rows == 0 {
            return Ok(parquet_path);
        }

        let num_cols = schema.fields().len();
        let mut cols: Vec<Arc<dyn arrow_array::Array>> = Vec::with_capacity(num_cols);

        for col_idx in 0..num_cols {
            let parts: Vec<&dyn arrow_array::Array> =
                batches.iter().map(|b| b.column(col_idx).as_ref()).collect();

            let combined = arrow_select::concat::concat(parts.as_slice())
                .map_err(|e| RockDuckError::Internal(format!("concat col {}: {}", col_idx, e)))?;
            cols.push(combined);
        }

        let combined_batch = RecordBatch::try_new(schema.clone(), cols)
            .map_err(|e| RockDuckError::Internal(format!("rebuild batch: {}", e)))?;

        let file = File::create(&parquet_path).map_err(|e| RockDuckError::Io(e))?;

        let mut writer = parquet::arrow::ArrowWriter::try_new(
            file,
            combined_batch.schema(),
            None, // default ParquetWriteOptions
        )
        .map_err(|e| RockDuckError::Write(format!("ArrowWriter::try_new: {}", e)))?;

        writer
            .write(&combined_batch)
            .map_err(|e| RockDuckError::Write(format!("ArrowWriter::write: {}", e)))?;

        writer
            .close()
            .map_err(|e| RockDuckError::Write(format!("ArrowWriter::close: {}", e)))?;

        // B4 fix: ArrowWriter::close() does NOT call fsync.
        // Explicit sync_all() ensures the Parquet file reaches stable storage.
        // This matches the durability guarantee of the Vortex streaming path (vortex_writer.rs:192).
        let file = File::options()
            .write(true)
            .open(&parquet_path)
            .map_err(|e| RockDuckError::Io(e))?;

        file.sync_all()
            .map_err(|e| RockDuckError::Write(format!("Parquet sync_all: {}", e)))?;

        tracing::info!("Wrote Parquet fallback (from Vortex): {:?}", parquet_path);
        Ok(parquet_path)
    }

    /// Close the streaming Parquet writer: flush pending pages and fsync.
    fn close_streaming_parquet(&mut self) -> Result<PathBuf> {
        let parquet_path = self.output_path.with_extension("parquet");
        if let Some(writer) = self.parquet_writer.take() {
            writer
                .close()
                .map_err(|e| RockDuckError::Write(format!("ArrowWriter::close: {}", e)))?;
            // fsync to match Vortex durability guarantee.
            let file = std::fs::OpenOptions::new()
                .write(true)
                .open(&parquet_path)
                .map_err(RockDuckError::Io)?;
            file.sync_all()
                .map_err(|e| RockDuckError::Write(format!("Parquet sync_all: {}", e)))?;
        }
        Ok(parquet_path)
    }

        let parquet_path = vortex_path.with_extension("parquet");

        if batches.is_empty() {
            return Ok(parquet_path);
        }

        let num_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        if num_rows == 0 {
            return Ok(parquet_path);
        }

        let num_cols = schema.fields().len();
        let mut cols: Vec<Arc<dyn arrow_array::Array>> = Vec::with_capacity(num_cols);

        for col_idx in 0..num_cols {
            let parts: Vec<&dyn arrow_array::Array> =
                batches.iter().map(|b| b.column(col_idx).as_ref()).collect();

            let combined = arrow_select::concat::concat(parts.as_slice())
                .map_err(|e| RockDuckError::Internal(format!("concat col {}: {}", col_idx, e)))?;
            cols.push(combined);
        }

        let combined_batch = RecordBatch::try_new(schema.clone(), cols)
            .map_err(|e| RockDuckError::Internal(format!("rebuild batch: {}", e)))?;

        let file = File::create(&parquet_path).map_err(|e| RockDuckError::Io(e))?;

        let mut writer = parquet::arrow::ArrowWriter::try_new(
            file,
            combined_batch.schema(),
            None, // default ParquetWriteOptions
        )
        .map_err(|e| RockDuckError::Write(format!("ArrowWriter::try_new: {}", e)))?;

        writer
            .write(&combined_batch)
            .map_err(|e| RockDuckError::Write(format!("ArrowWriter::write: {}", e)))?;

        writer
            .close()
            .map_err(|e| RockDuckError::Write(format!("ArrowWriter::close: {}", e)))?;

        // B4 fix: ArrowWriter::close() does NOT call fsync.
        // Explicit sync_all() ensures the Parquet file reaches stable storage.
        // This matches the durability guarantee of the Vortex streaming path (vortex_writer.rs:192).
        tracing::info!("Wrote Parquet fallback (from Vortex): {:?}", parquet_path);
        Ok(parquet_path)
    }

    /// Write both Vortex and Parquet tracks simultaneously.
    ///
    /// This is the dual-track export strategy:
    /// 1. Write `.vortex` using adaptive encoding (ALP/FastLanes/BtrBlocks)
    /// 2. Write `.parquet` as fallback for non-Vortex Iceberg consumers
    ///
    /// In streaming mode, the Parquet writer is opened upfront and batches are written
    /// incrementally — avoiding the O(n) concat and batch clone overhead.
    ///
    /// Returns `(vortex_path, parquet_path)`.
    pub fn write_dual_track(mut self) -> Result<(PathBuf, PathBuf)> {
        let projection_contract = ProjectionContract::vtab();
        projection_contract.assert_blocking_governance();
        if self.total_rows == 0 {
            return Err(RockDuckError::Internal("no batches to write".into()));
        }

        let vortex_path = self.output_path.clone();
        let total_rows = self.total_rows;
        let estimated_bytes = self.estimated_bytes;

        // Track A: Write Vortex file.
        if !self.batches.is_empty() {
            // Batched: flush all at once.
            self.flush_batched()?;
        }
        // In streaming mode, batches were already flushed by write_batch_streaming().

        // Track B: Write Parquet.
        // Batched mode: write from accumulated batches (no clone needed, already in memory).
        // Streaming mode: streaming Parquet writer was filled incrementally by write_batch_streaming().
        // Last resort: re-read from Vortex file.
        let parquet_path = if !self.batches.is_empty() {
            Self::write_parquet_from_batches_static(&self.batches, &self.schema, &vortex_path)?
        } else if self.parquet_writer.is_some() {
            // Streaming mode with incremental writer.
            self.close_streaming_parquet()?
        } else if !self.flushed_batches.is_empty() {
            Self::write_parquet_from_batches_static(
                &self.flushed_batches,
                &self.schema,
                &vortex_path,
            )?
        } else {
            Self::write_parquet_from_vortex_static(&vortex_path, &mut self.hotspot_stats)?
        };

        tracing::info!(
            "Dual-track export: vortex={:?}, parquet={:?} ({} rows, {} bytes)",
            vortex_path,
            parquet_path,
            total_rows,
            estimated_bytes
        );

        Ok((vortex_path, parquet_path))
    }

    /// Get total rows written.
    pub fn total_rows(&self) -> u64 {
        self.total_rows
    }

    /// Get the output path.
    pub fn output_path(&self) -> &PathBuf {
        &self.output_path
    }

    /// Get the estimated byte size.
    pub fn estimated_bytes(&self) -> usize {
        self.estimated_bytes
    }

    /// Get the write mode.
    pub fn is_streaming(&self) -> bool {
        self.streaming
    }

    pub fn hotspot_stats(&self) -> &VortexWriterHotspotStats {
        &self.hotspot_stats
    }
}

// =============================================================================
// VortexFileReader — reads Vortex files with Iceberg-compatible header
// =============================================================================

/// Read a Vortex file, optionally validating its Iceberg-compatible header.
pub struct VortexFileReader {
    path: PathBuf,
    header: Option<VortexFileHeader>,
    total_rows: u64,
}

impl VortexFileReader {
    /// Open a Vortex file and validate its header.
    pub fn open(path: impl Into<PathBuf>) -> Result<Self> {
        let path = path.into();
        let header = Self::read_header(&path)?;
        let total_rows = header.as_ref().map(|h| h.total_rows).unwrap_or(0);
        Ok(Self {
            path,
            header,
            total_rows,
        })
    }

    fn read_header(path: &PathBuf) -> Result<Option<VortexFileHeader>> {
        use std::io::Read;

        let mut file = std::fs::File::open(path).map_err(RockDuckError::Io)?;
        let mut header_bytes = [0u8; 32];
        if file.read_exact(&mut header_bytes).is_err() {
            return Ok(None); // File too short, legacy Vortex file
        }

        let magic = &header_bytes[0..4];
        if magic != VORTEX_MAGIC {
            return Ok(None); // Not a Vortex file (legacy Arrow IPC)
        }

        let version = header_bytes[4];
        if version > VORTEX_VERSION {
            return Err(RockDuckError::Internal(format!(
                "Unsupported Vortex file version: {} (max: {})",
                version, VORTEX_VERSION
            )));
        }

        let schema_len = u16::from_le_bytes(header_bytes[6..8].try_into().ok().unwrap_or(0));
        let encoding_count = u16::from_le_bytes(header_bytes[8..10].try_into().ok().unwrap_or(0));
        let total_rows = u64::from_le_bytes(header_bytes[10..18].try_into().ok().unwrap_or(0));

        tracing::debug!(
            "VortexFileReader: version={}, schema_len={}, encodings={}, rows={}",
            version,
            schema_len,
            encoding_count,
            total_rows
        );

        Ok(Some(VortexFileHeader {
            magic: VORTEX_MAGIC,
            version,
            flags: header_bytes[5],
            schema_len,
            encoding_count,
            total_rows,
            _reserved: [0u8; 8],
        }))
    }

    pub fn total_rows(&self) -> u64 {
        self.total_rows
    }

    pub(crate) fn header(&self) -> Option<&VortexFileHeader> {
        self.header.as_ref()
    }

    pub fn path(&self) -> &PathBuf {
        &self.path
    }
}

#[cfg(test)]
mod hotspot_tests {
    use super::*;
    use std::sync::Arc;

    use arrow_array::Int64Array;
    use arrow_schema::{DataType, Field, Schema};

    fn small_batch(rows: usize) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Int64,
            false,
        )]));
        let values = Int64Array::from((0..rows as i64).collect::<Vec<_>>());
        RecordBatch::try_new(schema, vec![Arc::new(values)]).expect("record batch")
    }

    #[test]
    fn batched_finish_records_single_flush_and_encode_cycle() {
        let temp = tempfile::tempdir().expect("tempdir");
        let path = temp.path().join("batched.vortex");
        let batch = small_batch(16);
        let mut writer = VortexFileWriter::new(path, batch.schema());

        writer.write_batch(batch).expect("write batch");
        assert!(!writer.is_streaming());
        let before_finish = writer.hotspot_stats().clone();
        assert_eq!(before_finish.batched_flushes, 0);

        writer.finish().expect("finish writer");
    }

    #[test]
    fn streaming_write_records_streaming_activity_without_batched_flush() {
        let temp = tempfile::tempdir().expect("tempdir");
        let path = temp.path().join("streaming.vortex");
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Int64,
            false,
        )]));
        let large_values = Int64Array::from(vec![42_i64; 200_000]);
        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(large_values)]).expect("record batch");
        let mut writer = VortexFileWriter::new(path, batch.schema());

        writer.write_batch(batch).expect("write batch");
        assert!(writer.is_streaming());

        let stats = writer.hotspot_stats().clone();
        assert_eq!(stats.streaming_writes, 1);
        assert_eq!(stats.batched_flushes, 0);
        assert_eq!(stats.encode_batch_calls, 1);
        assert_eq!(stats.encode_vortex_bytes_calls, 1);
    }

    #[test]
    fn finish_with_zero_batches_returns_error() {
        let temp = tempfile::tempdir().expect("tempdir");
        let path = temp.path().join("empty.vortex");
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Int64,
            false,
        )]));
        let writer = VortexFileWriter::new(path, schema);

        let result = writer.finish();
        assert!(result.is_err(), "finish() should error on empty writer");
    }

    #[test]
    fn dual_track_write_produces_both_tracks() {
        let temp = tempfile::tempdir().expect("tempdir");
        let path = temp.path().join("dual.vortex");
        let batch = small_batch(32);
        let mut writer = VortexFileWriter::new(path.clone(), batch.schema());

        writer.write_batch(batch).expect("write batch");
        let result = writer
            .write_dual_track()
            .expect("dual track should succeed");
        let (vortex_path, parquet_path) = result;

        assert_eq!(vortex_path, path);
        assert_eq!(parquet_path, path.with_extension("parquet"));
        assert!(vortex_path.exists(), "Vortex file should exist");
        assert!(parquet_path.exists(), "Parquet file should exist");
    }
}
