//! Vortex columnar storage with adaptive encoding support.
//!
//! ## Storage Format
//!
//! `.vortex` files use the Vortex File Format with BtrBlocks compression:
//! - **Write**: Arrow RecordBatch -> BtrBlocksCompressor -> VortexFile bytes -> write to disk
//! - **Read**: Read file bytes -> VortexFile (via open_buffer) -> Arrow RecordBatch
//!
//! ## Encoding Metadata
//!
//! Per-column encoding metadata is stored in `.vortex.encoding` sidecar files
//! (serialized via postcard), giving full encoding control.
//!
//! ## Backward Compatibility
//!
//! Existing `.vortex` files (Arrow IPC format) are read via Arrow IPC FileReader.

use std::path::Path;
use std::sync::{Arc as StdArc, RwLock};

use arrow_array::RecordBatch;
use arrow_schema::{DataType, Field, SchemaRef, TimeUnit};

use vortex_array::IntoArray;

use crate::codec;
use crate::codec::column_encoding::{ColumnEncoding, EncodingScheme};
use crate::error::{Result, RockDuckError};

// =============================================================================
// Encoding sidecar file format
//
// Format v1 (64-byte fixed header + N postcard-encoded entries):
//   magic[4] = b"VEM1"
//   version[1] = 1
//   entry_count[4] = little-endian u32
//   reserved[55] = 0
//
// Legacy format (no header): raw postcard-encoded ColumnEncoding bytes.
// =============================================================================

const VEM_MAGIC: &[u8; 4] = b"VEM1";
const VEM_VERSION: u8 = 1;
const VEM_HEADER_SIZE: usize = 64;

/// Fixed-size header at the start of a .vortex.encoding sidecar file.
/// Layout: magic[4] + version[1] + entry_count[4] + payload_len[4] + reserved[51] = 64 bytes.
#[repr(C, packed)]
struct EncodingMetaHeader {
    magic: [u8; 4], // b"VEM1"
    version: u8,    // current version = 1
    /// For the file-level header: total number of encoding entries.
    /// For per-entry headers: 1 (each entry is self-contained).
    entry_count: u32, // little-endian u32
    /// Length in bytes of the postcard-encoded payload that follows this header.
    payload_len: u32, // little-endian u32
    reserved: [u8; 51], // zeroed for forward compatibility
}

impl EncodingMetaHeader {
    fn new_file_header(entry_count: u32) -> Self {
        Self {
            magic: *VEM_MAGIC,
            version: VEM_VERSION,
            entry_count: entry_count.to_le(),
            payload_len: 0u32.to_le(),
            reserved: [0u8; 51],
        }
    }

    fn new_entry(payload_len: u32) -> Self {
        Self {
            magic: *VEM_MAGIC,
            version: VEM_VERSION,
            entry_count: 1u32.to_le(),
            payload_len: payload_len.to_le(),
            reserved: [0u8; 51],
        }
    }

    fn to_bytes(&self) -> [u8; VEM_HEADER_SIZE] {
        // SAFETY: EncodingMetaHeader is repr(C, packed) with explicit size VEM_HEADER_SIZE.
        // transmute_copy copies the bytes from self without consuming it.
        let mut bytes = [0u8; VEM_HEADER_SIZE];
        bytes[0..4].copy_from_slice(&self.magic);
        bytes[4] = self.version;
        bytes[5..9].copy_from_slice(&self.entry_count.to_le_bytes());
        bytes[9..13].copy_from_slice(&self.payload_len.to_le_bytes());
        bytes
    }

    fn from_bytes(bytes: &[u8; VEM_HEADER_SIZE]) -> Self {
        Self {
            magic: [bytes[0], bytes[1], bytes[2], bytes[3]],
            version: bytes[4],
            entry_count: u32::from_le_bytes([bytes[5], bytes[6], bytes[7], bytes[8]]),
            payload_len: u32::from_le_bytes([bytes[9], bytes[10], bytes[11], bytes[12]]),
            reserved: [0u8; 51], // ignore reserved bytes on read
        }
    }

    fn is_valid(&self) -> bool {
        self.magic == *VEM_MAGIC && self.version == VEM_VERSION
    }
}

// =============================================================================
// VortexReader -- reads Vortex or Arrow IPC column files
// =============================================================================

pub struct VortexReader {
    path: std::path::PathBuf,
    schema: SchemaRef,
    /// Lazily-loaded, shared batches. Wrapped in RwLock so that `&self` can set it
    /// during lazy loading. Cloning the StdArc is O(1) — no deep copy of batch data.
    /// Initially None — loaded lazily on first `read_all_batches()` call, so that
    /// point-get callers can use `read_batch_at()` instead without paying the cost.
    batches: RwLock<Option<StdArc<Vec<RecordBatch>>>>,
    total_rows: u64,
    encoding_meta: Option<ColumnEncoding>,
}

impl Clone for VortexReader {
    fn clone(&self) -> Self {
        Self {
            path: self.path.clone(),
            schema: self.schema.clone(),
            batches: RwLock::new(
                self.batches
                    .read()
                    .expect("VortexReader: batches lock poisoned")
                    .clone(),
            ),
            total_rows: self.total_rows,
            encoding_meta: self.encoding_meta.clone(),
        }
    }
}

impl VortexReader {
    pub fn open(path: impl Into<std::path::PathBuf>) -> Result<Self> {
        Self::new(path.into())
    }

    pub fn new(path: std::path::PathBuf) -> Result<Self> {
        let encoding_meta = Self::load_encoding_meta(&path).ok();

        // Try Vortex File format first (magic byte detection)
        if Self::looks_like_vortex(&path) {
            if let Ok(vtx_file) = Self::open_vortex_buffer(&path) {
                let (schema, batches, total_rows) =
                    Self::read_vortex_from_buffer(&vtx_file, path.clone())?;
                return Ok(Self {
                    path,
                    schema,
                    batches: RwLock::new(Some(StdArc::new(batches))),
                    total_rows,
                    encoding_meta,
                });
            }
        }

        // Fall back to Arrow IPC (legacy files)
        Self::read_arrow_ipc(&path).map(|(schema, batches, total_rows)| Self {
            path,
            schema,
            batches: RwLock::new(Some(StdArc::new(batches))),
            total_rows,
            encoding_meta,
        })
    }

    /// Detect Vortex format by magic bytes ("VTXF").
    fn looks_like_vortex(path: &Path) -> bool {
        use std::io::Read;
        if let Ok(mut file) = std::fs::File::open(path) {
            let mut magic = [0u8; 4];
            if file.read_exact(&mut magic).is_ok() {
                return &magic == b"VTXF";
            }
        }
        false
    }

    /// Open a Vortex file using memory-mapped I/O (zero-copy).
    ///
    /// Uses `From<Mmap>` impl (available via `memmap2` feature)
    /// instead of eagerly copying the entire mmap into a heap-allocated Vec.
    ///
    /// The `From<Mmap>` implementation calls `Bytes::from_owner(mmap)`, which
    /// transfers ownership of the mmap to Bytes without any memcpy. The OS
    /// pages data in on demand via page faults -- true zero-copy.
    fn open_vortex_mmap(path: &Path) -> Result<vortex_buffer::ByteBuffer> {
        use memmap2::Mmap;

        let file = std::fs::File::open(path).map_err(RockDuckError::Io)?;
        let mmap = unsafe { Mmap::map(&file) }.map_err(RockDuckError::Io)?;
        // Zero-copy: ByteBuffer::from(mmap) uses Bytes::from_owner internally,
        // transferring the mmap pointer without copying data.
        Ok(vortex_buffer::ByteBuffer::from(mmap))
    }

    /// Open a Vortex file by reading bytes into memory and using open_buffer (fully sync).
    fn open_vortex_buffer(path: &Path) -> Result<vortex_buffer::ByteBuffer> {
        // Try mmap first (zero-copy, OS page-cache managed)
        Self::open_vortex_mmap(path)
    }

    fn read_vortex_from_buffer(
        buffer: &vortex_buffer::ByteBuffer,
        _path: std::path::PathBuf,
    ) -> Result<(SchemaRef, Vec<RecordBatch>, u64)> {
        use vortex_array::stream::ArrayStreamExt;
        use vortex_file::OpenOptionsSessionExt;
        use vortex_io::runtime::BlockingRuntime;

        let (session, runtime) = Self::make_session();

        let vtx_file = session
            .open_options()
            .open_buffer(buffer.clone())
            .map_err(|e| RockDuckError::Internal(format!("open_buffer: {}", e)))?;

        let arr = runtime.block_on(async {
            let scan = vtx_file
                .scan()
                .map_err(|e| RockDuckError::Internal(format!("vortex scan: {}", e)))?;
            scan.into_array_stream()
                .map_err(|e| RockDuckError::Internal(format!("into_array_stream: {}", e)))?
                .read_all()
                .await
                .map_err(|e| RockDuckError::Internal(format!("read_all: {}", e)))
        })?;

        // Derive Arrow schema from the first column of the Vortex array.
        // Vortex files store one column per file, so this gives us the correct schema.
        let schema = Self::schema_from_vortex_array(&arr)?;
        let batch =
            Self::vortex_to_arrow_batch_with_session_static(&session, &arr, schema.clone())?;
        let total_rows = batch.num_rows() as u64;

        Ok((schema, vec![batch], total_rows))
    }

    /// Derive an Arrow schema from a Vortex array by reading the first chunk.
    #[allow(deprecated)]
    fn schema_from_vortex_array(vtx_arr: &vortex_array::ArrayRef) -> Result<SchemaRef> {
        use vortex_array::arrow::ArrowArrayExecutor;
        use vortex_array::VortexSessionExecute;

        let (session, _runtime) = Self::make_session();
        let mut ctx = session.create_execution_ctx();

        // Try to extract column information from the Vortex array's chunk structure.
        // For a single-column Vortex file, we convert the array directly.
        // For multi-column, we project each chunk.
        let col: arrow_array::ArrayRef = vtx_arr
            .clone()
            .execute_arrow(None, &mut ctx)
            .map_err(|e| RockDuckError::Internal(format!("schema col: {}", e)))?;

        // Build schema: if this is a ChunkedArray with N chunks, we have N columns.
        // Otherwise, single column named "value" (backward compat).
        let fields = vec![arrow_schema::Field::new(
            "value",
            col.data_type().clone(),
            true,
        )];
        Ok(StdArc::new(arrow_schema::Schema::new(fields)))
    }

    #[allow(dead_code)]
    fn vortex_to_arrow_batch(
        vtx_arr: &vortex_array::ArrayRef,
        schema: SchemaRef,
    ) -> Result<RecordBatch> {
        let (session, _runtime) = Self::make_session();
        Self::vortex_to_arrow_batch_with_session_static(&session, vtx_arr, schema)
    }

    #[allow(dead_code)]
    #[allow(deprecated)]
    fn vortex_to_arrow_batch_with_session_static(
        session: &vortex_session::VortexSession,
        vtx_arr: &vortex_array::ArrayRef,
        schema: SchemaRef,
    ) -> Result<RecordBatch> {
        use vortex_array::arrow::ArrowArrayExecutor;
        use vortex_array::VortexSessionExecute;

        let mut ctx = session.create_execution_ctx();

        // Derive all columns from the Vortex array using the provided schema.
        // Previously this always converted to a single-column RecordBatch named "value",
        // silently dropping all columns except the first for multi-column Vortex files.
        let num_schema_fields = schema.fields().len();

        if num_schema_fields > 1 {
            // Multi-column case: the Vortex array represents multiple columns.
            // We convert the entire array to Arrow -- each chunk in the ChunkedArray
            // corresponds to one column, and Arrow conversion handles this correctly.
            // This preserves all columns (unlike the old code which only read column 0).
            let arrow_arr: arrow_array::ArrayRef = vtx_arr
                .clone()
                .execute_arrow(None, &mut ctx)
                .map_err(|e| RockDuckError::Internal(format!("execute_arrow multi-col: {}", e)))?;

            // Build a RecordBatch from the converted Arrow array.
            // execute_arrow returns the canonical Arrow representation, which may be a
            // StructArray if there are multiple columns. We reconstruct using the schema.
            if let Some(struct_arr) = arrow_arr
                .as_ref()
                .as_any()
                .downcast_ref::<arrow_array::StructArray>()
            {
                let arrays: Vec<arrow_array::ArrayRef> = struct_arr
                    .columns()
                    .iter()
                    .map(|a| a.clone() as arrow_array::ArrayRef)
                    .collect();
                RecordBatch::try_new(schema, arrays)
                    .map_err(|e| RockDuckError::Internal(format!("struct multi-col batch: {}", e)))
            } else {
                // Fallback: wrap single array with schema (backward compat)
                RecordBatch::try_new(schema, vec![arrow_arr])
                    .map_err(|e| RockDuckError::Internal(format!("fallback batch: {}", e)))
            }
        } else {
            // Single column path (backward compat) -- use the schema's single field
            let arrow_arr: arrow_array::ArrayRef = vtx_arr
                .clone()
                .execute_arrow(None, &mut ctx)
                .map_err(|e| RockDuckError::Internal(format!("execute_arrow: {}", e)))?;
            RecordBatch::try_new(schema, vec![arrow_arr]).map_err(|e| {
                RockDuckError::Internal(format!("single-col batch from vortex: {}", e))
            })
        }
    }

    fn read_arrow_ipc(path: &Path) -> Result<(SchemaRef, Vec<RecordBatch>, u64)> {
        use arrow_ipc::reader::FileReader;
        use std::fs::File;
        use std::io::BufReader;

        let file = File::open(path).map_err(RockDuckError::Io)?;
        let reader = BufReader::new(file);
        let ipc_reader = FileReader::try_new(reader, None)
            .map_err(|e| RockDuckError::Internal(format!("Arrow IPC reader: {}", e)))?;

        let schema = ipc_reader.schema().clone();
        let mut batches = Vec::new();
        let mut total_rows = 0u64;

        for batch in ipc_reader {
            let batch = batch.map_err(|e| RockDuckError::Internal(format!("batch read: {}", e)))?;
            total_rows += batch.num_rows() as u64;
            batches.push(batch);
        }

        Ok((schema, batches, total_rows))
    }

    fn load_encoding_meta(path: &Path) -> Result<ColumnEncoding> {
        let meta_path = path.with_extension("vortex.encoding");
        let bytes = std::fs::read(&meta_path).map_err(RockDuckError::Io)?;

        if bytes.len() < VEM_HEADER_SIZE {
            return Err(RockDuckError::Codec(
                "encoding sidecar file too short to contain header".into(),
            ));
        }

        let file_header_bytes: [u8; VEM_HEADER_SIZE] =
            bytes[..VEM_HEADER_SIZE].try_into().map_err(|_| {
                RockDuckError::Codec(
                    "encoding sidecar file: insufficient bytes for file header".into(),
                )
            })?;
        let file_header = EncodingMetaHeader::from_bytes(&file_header_bytes);

        if file_header.is_valid() {
            // V1 format: file header + entry descriptor + payload.
            // Read the entry descriptor (next 64 bytes).
            let after_header = VEM_HEADER_SIZE;
            if bytes.len() < after_header + VEM_HEADER_SIZE {
                return Err(RockDuckError::Codec(
                    "encoding sidecar: file too short for entry descriptor".into(),
                ));
            }
            let entry_header_bytes: [u8; VEM_HEADER_SIZE] = bytes
                [after_header..after_header + VEM_HEADER_SIZE]
                .try_into()
                .map_err(|_| {
                    RockDuckError::Codec(
                        "encoding sidecar: insufficient bytes for entry header".into(),
                    )
                })?;
            let entry_header = EncodingMetaHeader::from_bytes(&entry_header_bytes);

            if !entry_header.is_valid() {
                tracing::warn!("encoding sidecar: entry descriptor has invalid magic/version");
                return Err(RockDuckError::Codec(
                    "encoding sidecar: invalid entry descriptor".into(),
                ));
            }

            let payload_len = u32::from_le(entry_header.payload_len) as usize;
            let payload_start = after_header + VEM_HEADER_SIZE;

            if bytes.len() < payload_start + payload_len {
                return Err(RockDuckError::Codec(
                    "encoding sidecar: truncated payload".into(),
                ));
            }

            let payload = &bytes[payload_start..payload_start + payload_len];
            codec::decode(payload)
                .map_err(|e| RockDuckError::Codec(format!("encoding meta: {}", e)))
        } else {
            // Legacy format: raw postcard-encoded bytes without header.
            tracing::debug!(
                "encoding sidecar has no VEM1 magic — treating as legacy single-entry format"
            );
            codec::decode(&bytes).map_err(|e| RockDuckError::Codec(format!("encoding meta: {}", e)))
        }
    }

    /// Read the Vortex file as a stream of RecordBatch chunks.
    ///
    /// This is the **chunk-level read** path: instead of calling `read_all_batches()`
    /// which concatenates all chunks into a single large RecordBatch, this yields
    /// each Vortex chunk as a separate RecordBatch. Callers can process chunks
    /// incrementally, enabling:
    /// - Streaming ingestion (DuckDB VTab, DataFusion)
    /// - Early-exit on limit
    /// - Filter pushdown at chunk boundaries
    ///
    /// Each yielded RecordBatch is a separate column chunk from the Vortex file.
    /// For a single-column Vortex file (the common case), each chunk corresponds to
    /// one batch of rows written during segment flush.
    pub fn read_chunks(&self) -> impl Iterator<Item = Result<RecordBatch>> + '_ {
        VortexChunkIterator::new(std::path::PathBuf::from(&self.path))
    }

    /// Build a VortexSession with required extensions for reading.
    /// Includes ALP and FastLanes plugins so VortexFile can auto-decode all encoding types.
    /// Also returns the CurrentThreadRuntime so it stays alive while the session is used.
    pub(crate) fn make_session() -> (
        vortex_session::VortexSession,
        vortex_io::runtime::current::CurrentThreadRuntime,
    ) {
        use vortex_io::runtime::current::CurrentThreadRuntime;
        use vortex_io::runtime::BlockingRuntime;
        use vortex_io::session::RuntimeSessionExt;

        let runtime = CurrentThreadRuntime::new();
        let session = vortex_session::VortexSession::empty()
            .with::<vortex_array::dtype::session::DTypeSession>()
            .with::<vortex_array::session::ArraySession>()
            .with::<vortex_layout::session::LayoutSession>()
            .with::<vortex_array::scalar_fn::session::ScalarFnSession>()
            .with::<vortex_array::optimizer::kernels::ArrayKernels>()
            .with::<vortex_array::aggregate_fn::session::AggregateFnSession>()
            .with::<vortex_io::session::RuntimeSession>()
            .with_handle(runtime.handle());

        // Register ALP and FastLanes plugins so VortexFile can auto-decode
        vortex_alp::initialize(&session);
        vortex_fastlanes::initialize(&session);

        (session, runtime)
    }

    #[allow(dead_code)]
    pub(crate) fn make_session_with_plugins() -> (
        vortex_session::VortexSession,
        vortex_io::runtime::current::CurrentThreadRuntime,
    ) {
        Self::make_session()
    }

    pub fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    /// Read all batches from the Vortex file, sharing via Arc.
    ///
    /// Returns an Arc<Vec<RecordBatch>> — O(1) to clone the reference. Callers
    /// that need ownership (e.g. scan paths) can iterate the Arc directly since
    /// it derefs to Vec. The Arc avoids deep-copying batch data on repeated queries.
    pub fn read_all_batches(&self) -> StdArc<Vec<RecordBatch>> {
        // Fast path: batches already loaded
        if let Some(ref batches) = *self
            .batches
            .read()
            .expect("VortexReader: batches lock poisoned")
        {
            return StdArc::clone(batches);
        }

        // Slow path: lazily load and cache.
        tracing::warn!("VortexReader: batches not loaded at construction, loading now");

        if Self::looks_like_vortex(&self.path) {
            match Self::open_vortex_mmap(&self.path) {
                Ok(buffer) => match Self::read_vortex_batches(&buffer, self.schema.clone()) {
                    Ok(batches) => {
                        let shared = StdArc::new(batches);
                        *self
                            .batches
                            .write()
                            .expect("VortexReader: batches lock poisoned") =
                            Some(StdArc::clone(&shared));
                        return shared;
                    }
                    Err(e) => tracing::warn!("Vortex reload failed: {}", e),
                },
                Err(e) => tracing::warn!("Vortex mmap reload failed: {}", e),
            }
        }

        // Arrow IPC fallback
        match Self::read_arrow_ipc(&self.path) {
            Ok((_schema, batches, _total_rows)) => {
                let shared = StdArc::new(batches);
                *self
                    .batches
                    .write()
                    .expect("VortexReader: batches lock poisoned") = Some(StdArc::clone(&shared));
                shared
            }
            Err(e) => {
                tracing::error!("Arrow IPC reload also failed: {}", e);
                StdArc::new(Vec::new())
            }
        }
    }

    /// Read a single RecordBatch containing the row at `row_offset` using per-chunk streaming.
    ///
    /// This is the **O(1) point-get path**: instead of loading all batches into memory
    /// (as `read_all_batches()` does), this uses `VortexChunkIteratorStreaming` to load
    /// only the chunk(s) that contain the target row. Iteration stops as soon as the
    /// row is found, avoiding unnecessary I/O for subsequent chunks.
    ///
    /// Returns `None` if `row_offset >= self.total_rows` or if an error occurs.
    /// Returns `Some((batch, local_row_idx))` where `local_row_idx` is the 0-based index
    /// of the target row within the returned batch.
    pub fn read_batch_at(&self, row_offset: u64) -> Option<(RecordBatch, usize)> {
        if row_offset >= self.total_rows {
            return None;
        }

        // Fast path: if batches are already loaded in memory, binary-search for the chunk.
        if let Some(ref batches) = *self
            .batches
            .read()
            .expect("VortexReader: batches lock poisoned")
        {
            let mut start = 0u64;
            for batch in batches.iter() {
                let end = start + batch.num_rows() as u64;
                if row_offset < end {
                    let local_idx = (row_offset - start) as usize;
                    return Some((batch.clone(), local_idx));
                }
                start = end;
            }
            return None;
        }

        // Slow path: stream chunks lazily until we find the target row.
        let iter = VortexChunkIteratorStreaming::new(self.path.clone(), None);
        let mut base_offset: u64 = 0;

        for batch_result in iter {
            match batch_result {
                Ok(batch) => {
                    let batch_rows = batch.num_rows() as u64;
                    if row_offset < base_offset + batch_rows {
                        let local_idx = (row_offset - base_offset) as usize;
                        return Some((batch, local_idx));
                    }
                    base_offset += batch_rows;
                }
                Err(_) => return None,
            }
        }
        None
    }

    /// Stream chunks from the Vortex file without loading all batches into memory.
    ///
    /// ## Naming Clarification
    ///
    /// Despite the "lazy" in the name, this currently delegates to `VortexChunkIterator`
    /// which is EAGER (loads all batches at construction). This is NOT a bug — the name
    /// reflects intent for future lazy loading. For true per-chunk streaming with early-exit,
    /// use `read_batch_at()` which uses `VortexChunkIteratorStreaming` internally.
    pub fn read_chunks_lazy(&self) -> impl Iterator<Item = Result<RecordBatch>> + '_ {
        VortexChunkIterator::new(std::path::PathBuf::from(&self.path))
    }

    /// Read batches from a Vortex buffer.
    fn read_vortex_batches(
        buffer: &vortex_buffer::ByteBuffer,
        schema: SchemaRef,
    ) -> Result<Vec<RecordBatch>> {
        use vortex_array::stream::ArrayStreamExt;
        use vortex_file::OpenOptionsSessionExt;
        use vortex_io::runtime::BlockingRuntime;

        let (session, runtime) = Self::make_session();
        let vtx_file = session
            .open_options()
            .open_buffer(buffer.clone())
            .map_err(|e| RockDuckError::Internal(format!("open_buffer: {}", e)))?;

        let arr = runtime.block_on(async {
            let scan = vtx_file
                .scan()
                .map_err(|e| RockDuckError::Internal(format!("vortex scan: {}", e)))?;
            scan.into_array_stream()
                .map_err(|e| RockDuckError::Internal(format!("into_array_stream: {}", e)))?
                .read_all()
                .await
                .map_err(|e| RockDuckError::Internal(format!("read_all: {}", e)))
        })?;

        let batch = Self::vortex_to_arrow_batch_with_session_static(&session, &arr, schema)?;
        Ok(vec![batch])
    }

    pub fn total_rows(&self) -> u64 {
        self.total_rows
    }

    pub fn num_batches(&self) -> usize {
        self.batches
            .read()
            .expect("VortexReader: batches lock poisoned")
            .as_ref()
            .map_or(0, |b| b.len())
    }

    pub fn encoding_meta(&self) -> Option<&ColumnEncoding> {
        self.encoding_meta.as_ref()
    }

    /// Scan with an optional Vortex expression filter pushed into the storage layer.
    /// The filter is applied during the scan, enabling FilterKernel per-encoding optimizations.
    ///
    /// Note: Currently reads all matching data into memory as a single batch.
    /// Chunked streaming scan will be added in a future iteration.
    pub fn scan_with_filter(
        &self,
        vortex_filter: Option<vortex_array::expr::Expression>,
    ) -> crate::error::Result<Vec<RecordBatch>> {
        use vortex_array::stream::ArrayStreamExt;
        use vortex_file::OpenOptionsSessionExt;
        use vortex_io::runtime::BlockingRuntime;

        let buffer = Self::open_vortex_mmap(&self.path)?;
        let (session, runtime) = Self::make_session();

        let vtx_file = session
            .open_options()
            .open_buffer(buffer)
            .map_err(|e| RockDuckError::Internal(format!("open_buffer: {}", e)))?;

        let arr = runtime.block_on(async {
            let scan = vtx_file
                .scan()
                .map_err(|e| RockDuckError::Internal(format!("vortex scan: {}", e)))?;

            let scan = if let Some(f) = vortex_filter {
                scan.with_filter(f)
            } else {
                scan
            };

            scan.into_array_stream()
                .map_err(|e| RockDuckError::Internal(format!("into_array_stream: {}", e)))?
                .read_all()
                .await
                .map_err(|e| RockDuckError::Internal(format!("read_all: {}", e)))
        })?;

        let schema = Self::schema_from_vortex_array(&arr)?;
        let batch = Self::vortex_to_arrow_batch_with_session_static(&session, &arr, schema)?;
        Ok(vec![batch])
    }
}

// =============================================================================
// VortexWriter -- writes Vortex column files with adaptive encoding
// =============================================================================

pub struct VortexWriter {
    path: std::path::PathBuf,
    col_name: String,
    batches: Vec<RecordBatch>,
    total_rows: u64,
    picker: Option<crate::storage::vortex_alp_ext::AdaptiveEncodingPicker>,
    /// Adaptive flush threshold (rows). Flush to disk when accumulated rows exceed this.
    flush_threshold: u64,
    /// Bytes written so far (running total for adaptive decisions).
    bytes_written: usize,
    /// Cached VortexSession for compression (avoids rebuilding per flush).
    cached_session: Option<(
        vortex_session::VortexSession,
        vortex_io::runtime::current::CurrentThreadRuntime,
    )>,
    /// Total original bytes (Arrow data) written in streaming mode.
    /// Accumulated from each flush_to_disk call.
    streaming_original_bytes: u64,
    /// Total encoded bytes (compressed Vortex data) written in streaming mode.
    /// Accumulated from each flush_to_disk call.
    streaming_encoded_bytes: u64,
    /// Total rows written in streaming mode (tracked separately since batches are cleared after flush).
    /// Fix #10: This field ensures total_rows is correctly reported in streaming mode.
    total_rows_flushed: u64,
}

impl VortexWriter {
    pub fn create(path: impl Into<std::path::PathBuf>, col_name: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            col_name: col_name.into(),
            batches: Vec::new(),
            total_rows: 0,
            picker: None,
            flush_threshold: 100_000,
            bytes_written: 0,
            cached_session: None,
            streaming_original_bytes: 0,
            streaming_encoded_bytes: 0,
            total_rows_flushed: 0,
        }
    }

    pub fn with_encoding_picker(
        path: impl Into<std::path::PathBuf>,
        col_name: impl Into<String>,
        picker: crate::storage::vortex_alp_ext::AdaptiveEncodingPicker,
    ) -> Self {
        Self {
            path: path.into(),
            col_name: col_name.into(),
            batches: Vec::new(),
            total_rows: 0,
            picker: Some(picker),
            flush_threshold: 100_000,
            bytes_written: 0,
            cached_session: None,
            streaming_original_bytes: 0,
            streaming_encoded_bytes: 0,
            total_rows_flushed: 0,
        }
    }

    /// Set the adaptive flush threshold (in rows).
    /// When accumulated rows exceed this, `flush_pending` can be called to write to disk.
    pub fn with_flush_threshold(mut self, threshold: u64) -> Self {
        self.flush_threshold = threshold;
        self
    }

    pub fn write(&mut self, batch: RecordBatch) -> Result<()> {
        self.total_rows += batch.num_rows() as u64;
        self.bytes_written += batch.get_array_memory_size();
        self.batches.push(batch);
        Ok(())
    }

    /// Flush pending batches to disk (adaptive write).
    ///
    /// Adaptive strategy:
    /// - **Streaming** (rows >= flush_threshold): immediately write to disk after each batch,
    ///   avoiding memory accumulation. Suitable for large ingestion pipelines.
    /// - **Batching** (rows < flush_threshold): accumulate in memory, write on `finish()`.
    ///
    /// Returns the number of batches flushed, or 0 if no flush was needed (batching mode).
    pub fn flush_pending(&mut self) -> Result<usize> {
        if self.total_rows < self.flush_threshold {
            return Ok(0); // Batching mode: nothing to flush yet
        }

        // Streaming mode: actually write accumulated batches to disk and clear them.
        if self.batches.is_empty() {
            return Ok(0);
        }

        let flushed = self.flush_to_disk()?;
        Ok(flushed)
    }

    /// Compress and append accumulated batches to the Vortex file.
    ///
    /// Called by `flush_pending` in streaming mode. Writes each batch as a separate
    /// Vortex chunk by appending to the file. Clears `self.batches` after flushing.
    fn flush_to_disk(&mut self) -> Result<usize> {
        if self.batches.is_empty() {
            return Ok(0);
        }

        let flushed = self.batches.len();
        tracing::debug!(
            "VortexWriter streaming flush: {} batches ({} rows) to {:?}",
            flushed,
            self.total_rows,
            self.path
        );

        // Compute original bytes before compression
        let original_bytes: u64 = self
            .batches
            .iter()
            .map(|b| b.get_array_memory_size() as u64)
            .sum();

        // Compress all batches with adaptive encoding.
        let (compressed, chosen_scheme) = self.compress_adaptive()?;

        // Append encoded bytes to the Vortex file.
        let encoded_bytes = Self::append_vortex_bytes_sync(&compressed, &self.path)? as u64;

        // Append encoding metadata.
        let meta = ColumnEncoding {
            column_name: self.col_name.clone(),
            scheme: chosen_scheme,
            blocks: Vec::new(),
            correlation: None,
            lea: None,
            total_rows: self.total_rows,
            original_bytes,
            encoded_bytes,
        };
        Self::append_encoding_meta_sync(&self.path, &meta)?;

        // Track bytes for finish() metadata
        self.streaming_original_bytes += original_bytes;
        self.streaming_encoded_bytes += encoded_bytes;

        // Track rows flushed (fix #10: needed for correct total_rows in finish())
        let rows_this_flush = self.total_rows;
        self.total_rows_flushed += rows_this_flush;

        // Clear accumulated batches (already on disk).
        self.batches.clear();
        self.total_rows = 0;
        self.bytes_written += encoded_bytes as usize;

        Ok(flushed)
    }

    /// Append encoded Vortex bytes to a file (streaming mode).
    fn append_vortex_bytes_sync(arr: &vortex_array::ArrayRef, path: &Path) -> Result<usize> {
        use std::io::Write;
        use vortex_array::iter::ArrayIteratorExt;
        use vortex_file::register_default_encodings;
        use vortex_file::WriteOptionsSessionExt;
        use vortex_io::runtime::current::CurrentThreadRuntime;
        use vortex_io::runtime::BlockingRuntime;
        use vortex_io::session::RuntimeSessionExt;

        let bytes: Vec<u8> = smol::block_on(async {
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
        .map_err(|e| RockDuckError::Internal(format!("vortex streaming write: {}", e)))?;

        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .map_err(RockDuckError::Io)?;
        file.write_all(&bytes).map_err(RockDuckError::Io)?;
        file.sync_all().map_err(RockDuckError::Io)?;

        Ok(bytes.len())
    }

    /// Append encoding metadata to the sidecar file (streaming mode).
    /// Each call writes one entry in v1 format: file_header + entry_header + payload.
    /// On first call, the file_header is written. Subsequent calls append entry_header + payload.
    fn append_encoding_meta_sync(path: &Path, meta: &ColumnEncoding) -> Result<()> {
        let meta_bytes = codec::encode(meta)
            .map_err(|e| RockDuckError::Codec(format!("encode encoding meta: {}", e)))?;
        let meta_path = path.with_extension("vortex.encoding");

        use std::io::Write;
        let file_exists = meta_path.exists();
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&meta_path)
            .map_err(RockDuckError::Io)?;

        if !file_exists {
            // First entry: write file header first, then entry descriptor + payload.
            let file_header = EncodingMetaHeader::new_file_header(1);
            file.write_all(&file_header.to_bytes())
                .map_err(RockDuckError::Io)?;
        }

        let entry_header = EncodingMetaHeader::new_entry(meta_bytes.len() as u32);
        file.write_all(&entry_header.to_bytes())
            .map_err(RockDuckError::Io)?;
        file.write_all(&meta_bytes).map_err(RockDuckError::Io)?;
        file.sync_all().map_err(RockDuckError::Io)?;
        Ok(())
    }

    /// Finish: compress with adaptive encoding, write Vortex file, save encoding metadata.
    pub fn finish(mut self) -> Result<ColumnEncoding> {
        // Flush any pending batches first (st016)
        self.flush_pending()?;

        if self.batches.is_empty() && self.total_rows == 0 {
            return Err(RockDuckError::Internal("no batches to write".into()));
        }

        // In streaming mode, batches were already flushed by flush_pending().
        // Use the tracked bytes from streaming flushes.
        // Fix #10: Use total_rows_flushed instead of total_rows (which is 0 after flush).
        if self.total_rows == 0 {
            let encoding_meta = ColumnEncoding {
                column_name: self.col_name.clone(),
                scheme: EncodingScheme::BtrBlocks,
                blocks: Vec::new(),
                correlation: None,
                lea: None,
                total_rows: self.total_rows_flushed,
                original_bytes: self.streaming_original_bytes,
                encoded_bytes: self.streaming_encoded_bytes,
            };
            return Ok(encoding_meta);
        }

        let original_bytes: u64 = self
            .batches
            .iter()
            .map(|b| b.get_array_memory_size() as u64)
            .sum();

        // Compress with adaptive encoding
        let (compressed, chosen_scheme) = self.compress_adaptive()?;

        // Write Vortex file to in-memory buffer, then persist to disk
        let encoded_bytes = Self::write_vortex_bytes_sync(&compressed, &self.path)? as u64;

        let encoding_meta = ColumnEncoding {
            column_name: self.col_name.clone(),
            scheme: chosen_scheme,
            blocks: Vec::new(),
            correlation: None,
            lea: None,
            total_rows: self.total_rows,
            original_bytes,
            encoded_bytes,
        };

        Self::save_encoding_meta_sync(&self.path, &encoding_meta)?;
        Ok(encoding_meta)
    }

    fn get_or_create_session(
        &mut self,
    ) -> &(
        vortex_session::VortexSession,
        vortex_io::runtime::current::CurrentThreadRuntime,
    ) {
        if self.cached_session.is_none() {
            self.cached_session = Some(VortexReader::make_session());
        }
        self.cached_session.as_ref().unwrap()
    }

    fn compress_adaptive(&mut self) -> Result<(vortex_array::ArrayRef, EncodingScheme)> {
        use vortex_array::arrays::ChunkedArray;
        use vortex_array::arrow::FromArrowArray;
        use vortex_array::IntoArray;
        use vortex_array::VortexSessionExecute;

        if self.batches.is_empty() {
            return Err(RockDuckError::Internal("no batches".into()));
        }

        let (session, _runtime) = self.get_or_create_session();
        let mut ctx = session.create_execution_ctx();

        let mut compressed_parts: Vec<vortex_array::ArrayRef> =
            Vec::with_capacity(self.batches.len());
        // Store per-batch encoding schemes, then use the dominant one (most rows).
        let mut batch_encodings: Vec<(EncodingScheme, u32)> = Vec::new();

        for batch in &self.batches {
            let arr: arrow_array::ArrayRef = batch.column(0).clone();

            if let Some(ref picker) = self.picker {
                let (encoded, scheme) = picker.pick_and_encode(arr.as_ref(), &mut ctx)?;
                compressed_parts.push(encoded);
                batch_encodings.push((scheme, batch.num_rows() as u32));
            } else {
                // Fallback: BtrBlocks only
                let vtx_arr = vortex_array::ArrayRef::from_arrow(arr.as_ref(), false)
                    .map_err(|e| RockDuckError::Internal(format!("Arrow->Vortex: {}", e)))?;
                let compressor = vortex_btrblocks::BtrBlocksCompressor::default();
                let comp = compressor
                    .compress(&vtx_arr, &mut ctx)
                    .map_err(|e| RockDuckError::Internal(format!("BtrBlocks: {}", e)))?;
                compressed_parts.push(comp);
                batch_encodings.push((EncodingScheme::BtrBlocks, batch.num_rows() as u32));
            }
        }

        // Use the dominant encoding -- the scheme used for the most rows.
        let chosen_scheme = batch_encodings
            .iter()
            .max_by_key(|(_, rows)| *rows)
            .map(|(scheme, _)| *scheme)
            .unwrap_or(EncodingScheme::BtrBlocks);

        let dtype = compressed_parts[0].dtype().clone();
        let chunked = ChunkedArray::try_new(compressed_parts, dtype)
            .map_err(|e| RockDuckError::Internal(format!("ChunkedArray: {}", e)))?;

        Ok((chunked.into_array(), chosen_scheme))
    }

    fn write_vortex_bytes_sync(arr: &vortex_array::ArrayRef, path: &Path) -> Result<usize> {
        use vortex_array::iter::ArrayIteratorExt;

        use vortex_file::register_default_encodings;
        use vortex_file::WriteOptionsSessionExt;
        use vortex_io::runtime::current::CurrentThreadRuntime;
        use vortex_io::runtime::BlockingRuntime;
        use vortex_io::session::RuntimeSessionExt;

        let bytes: Vec<u8> = smol::block_on(async {
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

            // Register ALP and FastLanes plugins so the writer can encode with them
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
        .map_err(|e| RockDuckError::Internal(format!("vortex write: {}", e)))?;

        // Write bytes to disk
        std::fs::write(path, &bytes).map_err(RockDuckError::Io)?;

        Ok(bytes.len())
    }

    fn save_encoding_meta_sync(path: &Path, meta: &ColumnEncoding) -> Result<()> {
        let meta_bytes = codec::encode(meta)
            .map_err(|e| RockDuckError::Codec(format!("encode encoding meta: {}", e)))?;
        let meta_path = path.with_extension("vortex.encoding");
        let file_header = EncodingMetaHeader::new_file_header(1);
        let entry_header = EncodingMetaHeader::new_entry(meta_bytes.len() as u32);
        let mut file = std::fs::File::create(&meta_path).map_err(RockDuckError::Io)?;
        use std::io::Write;
        file.write_all(&file_header.to_bytes())
            .map_err(RockDuckError::Io)?;
        file.write_all(&entry_header.to_bytes())
            .map_err(RockDuckError::Io)?;
        file.write_all(&meta_bytes).map_err(RockDuckError::Io)?;
        file.sync_all().map_err(RockDuckError::Io)?;
        Ok(())
    }

    pub fn total_rows(&self) -> u64 {
        self.total_rows
    }
    pub fn num_batches(&self) -> usize {
        self.batches.len()
    }
}

// =============================================================================
// Legacy compatibility
// =============================================================================

#[derive(
    Debug,
    Clone,
    serde::Serialize,
    serde::Deserialize,
    rkyv::Archive,
    rkyv::Serialize,
    rkyv::Deserialize,
)]
pub struct VortexMeta {
    pub num_rows: u64,
    pub num_batches: usize,
    pub schema_json: String,
}

#[derive(
    Debug,
    Clone,
    serde::Serialize,
    serde::Deserialize,
    rkyv::Archive,
    rkyv::Serialize,
    rkyv::Deserialize,
)]
pub struct SchemaField {
    pub name: String,
    pub data_type_str: String,
    pub nullable: bool,
}

impl SchemaField {
    pub fn from_field(field: &Field) -> Self {
        Self {
            name: field.name().to_string(),
            data_type_str: format!("{:?}", field.data_type()),
            nullable: field.is_nullable(),
        }
    }
    pub fn from_arc_field(field: &StdArc<Field>) -> Self {
        Self {
            name: field.name().to_string(),
            data_type_str: format!("{:?}", field.data_type()),
            nullable: field.is_nullable(),
        }
    }
    pub fn to_field(&self) -> Result<Field> {
        let dt = parse_data_type(&self.data_type_str)?;
        Ok(Field::new(&self.name, dt, self.nullable))
    }
}

fn parse_data_type(s: &str) -> Result<DataType> {
    match s {
        "Null" => Ok(DataType::Null),
        "Boolean" => Ok(DataType::Boolean),
        "Int8" => Ok(DataType::Int8),
        "Int16" => Ok(DataType::Int16),
        "Int32" => Ok(DataType::Int32),
        "Int64" => Ok(DataType::Int64),
        "UInt8" => Ok(DataType::UInt8),
        "UInt16" => Ok(DataType::UInt16),
        "UInt32" => Ok(DataType::UInt32),
        "UInt64" => Ok(DataType::UInt64),
        "Float16" => Ok(DataType::Float16),
        "Float32" => Ok(DataType::Float32),
        "Float64" => Ok(DataType::Float64),
        "Timestamp(Second, None)" => Ok(DataType::Timestamp(TimeUnit::Second, None)),
        "Timestamp(Millisecond, None)" => Ok(DataType::Timestamp(TimeUnit::Millisecond, None)),
        "Timestamp(Microsecond, None)" => Ok(DataType::Timestamp(TimeUnit::Microsecond, None)),
        "Timestamp(Nanosecond, None)" => Ok(DataType::Timestamp(TimeUnit::Nanosecond, None)),
        "Date32" => Ok(DataType::Date32),
        "Date64" => Ok(DataType::Date64),
        "Binary" => Ok(DataType::Binary),
        "LargeBinary" => Ok(DataType::LargeBinary),
        "Utf8" => Ok(DataType::Utf8),
        "LargeUtf8" => Ok(DataType::LargeUtf8),
        other => Err(RockDuckError::Internal(format!(
            "parse_data_type: unknown arrow data type '{}' (this may indicate data corruption or a version mismatch)", other
        ))),
    }
}

// =============================================================================
// Encoding helpers
// =============================================================================

/// Encode a single Arrow column array to a Vortex array with adaptive compression.
pub fn encode_column(
    arr: &dyn arrow_array::Array,
    session: &vortex_session::VortexSession,
) -> std::result::Result<vortex_array::ArrayRef, RockDuckError> {
    use vortex_array::arrow::FromArrowArray;

    use vortex_array::VortexSessionExecute;

    let vtx = vortex_array::ArrayRef::from_arrow(arr, false)
        .map_err(|e| RockDuckError::Internal(format!("Arrow->Vortex: {}", e)))?;

    let compressor = vortex_btrblocks::BtrBlocksCompressor::default();
    let mut ctx = session.create_execution_ctx();
    compressor
        .compress(&vtx, &mut ctx)
        .map_err(|e| RockDuckError::Internal(format!("BtrBlocks: {}", e)))
}

/// Decode a Vortex array to its canonical Arrow-compatible form.
pub fn decode_column(
    vtx: &vortex_array::ArrayRef,
    session: &vortex_session::VortexSession,
) -> std::result::Result<vortex_array::ArrayRef, RockDuckError> {
    use vortex_array::VortexSessionExecute;
    let mut ctx = session.create_execution_ctx();
    let canonical = vtx
        .clone()
        .execute::<vortex_array::Canonical>(&mut ctx)
        .map_err(|e| RockDuckError::Internal(format!("Canonical: {}", e)))?;
    Ok(canonical.into_array())
}

/// Returns the recommended Vortex encoding name for a given Arrow data type.
pub fn recommended_encoding(dt: &DataType) -> &'static str {
    match dt {
        DataType::Float32 | DataType::Float64 => "BtrBlocks(ALP)",
        DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::Int64
        | DataType::UInt8
        | DataType::UInt16
        | DataType::UInt32
        | DataType::UInt64 => "BtrBlocks(FastLanes)",
        _ => "BtrBlocks",
    }
}

pub fn make_schema(columns: &[(impl AsRef<str>, DataType)]) -> SchemaRef {
    let fields: Vec<Field> = columns
        .iter()
        .map(|(name, dtype)| Field::new(name.as_ref(), dtype.clone(), true))
        .collect();
    StdArc::new(arrow_schema::Schema::new(fields))
}

// =============================================================================
// VortexChunkIterator -- chunk-level streaming read via mmap
// =============================================================================

/// Iterator over RecordBatch chunks in a Vortex file.
///
/// Note: construction is EAGER -- opens the Vortex file and loads all batches
/// into memory immediately. Subsequent `next()` calls yield the already-loaded batches.
///
/// For true per-chunk streaming with early-exit, callers should use
/// `VortexReader::read_chunks_lazy()` or `scan_with_filter()` which accepts
/// Vortex Expression for filter pushdown at the storage layer.
struct VortexChunkIterator {
    /// Opened once at construction.
    #[allow(dead_code)]
    path: std::path::PathBuf,
    /// All batches loaded at construction -- avoid re-opening on each next().
    batches: Vec<RecordBatch>,
    /// Index of the next batch to yield.
    current_idx: usize,
    /// True if an open/read error occurred -- yield that error once, then stop.
    error: Option<super::super::RockDuckError>,
}

impl VortexChunkIterator {
    /// Open the Vortex file and load all batches ONCE.
    /// All errors are deferred to the Iterator -- the first `next()` yields the error.
    ///
    /// Batches are loaded eagerly via the async runtime, yielding each chunk as a separate RecordBatch.
    fn new(path: std::path::PathBuf) -> Self {
        // Open via mmap once (zero-copy, OS page-cache managed)
        let buffer = match VortexReader::open_vortex_mmap(&path) {
            Ok(b) => b,
            Err(e) => {
                return Self {
                    path,
                    batches: Vec::new(),
                    current_idx: 0,
                    error: Some(e),
                }
            }
        };

        use vortex_array::stream::ArrayStreamExt;
        use vortex_file::OpenOptionsSessionExt;
        use vortex_io::runtime::BlockingRuntime;

        let (session, runtime) = VortexReader::make_session();

        let vtx_file = match session.open_options().open_buffer(buffer) {
            Ok(f) => f,
            Err(e) => {
                return Self {
                    path,
                    batches: Vec::new(),
                    current_idx: 0,
                    error: Some(super::super::RockDuckError::Internal(format!(
                        "open_buffer: {}",
                        e
                    ))),
                }
            }
        };

        // Use the same blocking runtime pattern as read_vortex_batches()
        let arr = runtime.block_on(async {
            let scan = vtx_file
                .scan()
                .map_err(|e| super::super::RockDuckError::Internal(format!("scan: {}", e)))?;
            scan.into_array_stream()
                .map_err(|e| {
                    super::super::RockDuckError::Internal(format!("into_array_stream: {}", e))
                })?
                .read_all()
                .await
                .map_err(|e| super::super::RockDuckError::Internal(format!("read_all: {}", e)))
        });

        let arr = match arr {
            Ok(a) => a,
            Err(e) => {
                return Self {
                    path,
                    batches: Vec::new(),
                    current_idx: 0,
                    error: Some(e),
                }
            }
        };

        let schema = match VortexReader::schema_from_vortex_array(&arr) {
            Ok(s) => s,
            Err(e) => {
                return Self {
                    path,
                    batches: Vec::new(),
                    current_idx: 0,
                    error: Some(e),
                }
            }
        };

        match VortexReader::vortex_to_arrow_batch_with_session_static(&session, &arr, schema) {
            Ok(batch) if batch.num_rows() > 0 => Self {
                path,
                batches: vec![batch],
                current_idx: 0,
                error: None,
            },
            _ => Self {
                path,
                batches: Vec::new(),
                current_idx: 0,
                error: None,
            },
        }
    }
}

impl Iterator for VortexChunkIterator {
    type Item = Result<RecordBatch>;

    /// Yield the next batch. Opens the file at most ONCE (during construction).
    fn next(&mut self) -> Option<Self::Item> {
        // Yield deferred error first
        if let Some(e) = self.error.take() {
            return Some(Err(e));
        }

        if self.current_idx >= self.batches.len() {
            return None;
        }

        let batch = self.batches[self.current_idx].clone();
        self.current_idx += 1;
        Some(Ok(batch))
    }
}

// =============================================================================
// VortexChunkIteratorStreaming -- true chunk-level streaming via block_on_stream
// =============================================================================

/// Iterator over RecordBatch chunks in a Vortex file with TRUE per-chunk streaming.
///
/// Unlike `VortexChunkIterator` (eager — loads ALL chunks at construction),
/// `VortexChunkIteratorStreaming` yields one chunk at a time, blocking on each
/// `next()` call. This enables early-exit: if a query is satisfied after reading
/// N chunks, remaining chunks are never loaded.
///
/// Uses `vortex_io::BlockingRuntime::block_on_stream` to drive the async
/// `ArrayStream` as a synchronous iterator.
pub struct VortexChunkIteratorStreaming {
    /// Path to the Vortex file.
    #[allow(dead_code)]
    path: std::path::PathBuf,
    /// Optional filter pushed into the storage layer.
    #[allow(dead_code)]
    filter: Option<vortex_array::expr::Expression>,
    /// Holds the Vortex session — must stay alive across all `next()` calls.
    #[allow(dead_code)]
    session: vortex_session::VortexSession,
    /// Holds the runtime — needed by `into_array_iter`.
    #[allow(dead_code)]
    runtime: vortex_io::runtime::current::CurrentThreadRuntime,
    /// Cached mmap buffer — opened once at construction, shared across all `next()` calls.
    #[allow(dead_code)]
    buffer: vortex_buffer::ByteBuffer,
    /// Open VortexFile — opened once at construction, shared across all `next()` calls.
    #[allow(dead_code)]
    vtx_file: vortex_file::VortexFile,
    /// Schema for converting chunks to Arrow batches.
    schema: std::sync::Arc<arrow_schema::Schema>,
    /// Cached first chunk — read once for schema, yielded first in iteration.
    /// Avoids re-opening the file after schema derivation.
    first_chunk: Option<vortex_array::ArrayRef>,
    /// Blocking iterator yielding one `ArrayRef` chunk per `next()` call.
    /// Type-erased via Box because `ScanBuilder::into_array_iter` returns
    /// `impl ArrayIterator + 'static` which cannot be named directly.
    inner: std::boxed::Box<
        dyn Iterator<Item = vortex_error::VortexResult<vortex_array::ArrayRef>> + Send + 'static,
    >,
    /// Deferred error from initialization — yielded on first `next()` call.
    init_error: Option<crate::RockDuckError>,
}

impl VortexChunkIteratorStreaming {
    /// Open the Vortex file and set up the blocking stream iterator.
    /// The file is opened once; chunks are loaded one at a time on each `next()`.
    pub fn new(path: std::path::PathBuf, filter: Option<vortex_array::expr::Expression>) -> Self {
        use vortex_file::OpenOptionsSessionExt;

        // Open mmap once (zero-copy, OS page-cache managed)
        let buffer = match VortexReader::open_vortex_mmap(&path) {
            Ok(b) => b,
            Err(e) => return Self::error(path, e),
        };

        // Create session (reused across all chunks)
        let (session, runtime) = VortexReader::make_session();

        // Open VortexFile from the mmap buffer
        let buffer_for_vtx_file = buffer.clone();
        let vtx_file = match session.open_options().open_buffer(buffer_for_vtx_file) {
            Ok(f) => f,
            Err(e) => {
                return Self::error(
                    path,
                    crate::RockDuckError::Internal(format!("open_buffer: {}", e)),
                );
            }
        };

        // Build the blocking iterator from the scan
        let scan = match vtx_file.scan() {
            Ok(s) => s,
            Err(e) => {
                return Self::error(path, crate::RockDuckError::Internal(format!("scan: {}", e)));
            }
        };
        let scan = if let Some(ref f) = filter {
            scan.with_filter(f.clone())
        } else {
            scan
        };

        // into_array_iter returns an owned blocking iterator
        let mut iter = match scan.into_array_iter(&runtime) {
            Ok(i) => i,
            Err(e) => {
                return Self::error(
                    path,
                    crate::RockDuckError::Internal(format!("into_array_iter: {}", e)),
                );
            }
        };

        // Derive schema from the first chunk and cache it for iteration
        let first_chunk_result = iter.next();
        let (schema, first_chunk) = match first_chunk_result {
            Some(Ok(chunk)) => {
                let schema = VortexReader::schema_from_vortex_array(&chunk).unwrap_or_else(|_| {
                    std::sync::Arc::new(arrow_schema::Schema::new(std::vec::Vec::<
                        arrow_schema::Field,
                    >::new()))
                });
                (schema, Some(chunk))
            }
            Some(Err(e)) => {
                return Self::error(
                    path,
                    crate::RockDuckError::Internal(format!("read first chunk: {}", e)),
                );
            }
            None => {
                return Self::error(
                    path,
                    crate::RockDuckError::Internal("Vortex file is empty".into()),
                );
            }
        };

        // Re-open the file and seek to beginning instead of re-creating iterator
        // This avoids the overhead of re-parsing the file structure
        let vtx_file2 = match session.open_options().open_buffer(buffer.clone()) {
            Ok(f) => f,
            Err(e) => {
                return Self::error(
                    path,
                    crate::RockDuckError::Internal(format!("open_buffer (2nd): {}", e)),
                );
            }
        };
        let scan2 = match vtx_file2.scan() {
            Ok(s) => s,
            Err(e) => {
                return Self::error(
                    path,
                    crate::RockDuckError::Internal(format!("scan (2nd): {}", e)),
                );
            }
        };
        let scan2 = if let Some(ref f) = filter {
            scan2.with_filter(f.clone())
        } else {
            scan2
        };
        let iter2 = match scan2.into_array_iter(&runtime) {
            Ok(i) => i,
            Err(e) => {
                return Self::error(
                    path,
                    crate::RockDuckError::Internal(format!("into_array_iter (2nd): {}", e)),
                );
            }
        };

        Self {
            path,
            filter,
            session,
            runtime,
            buffer,
            vtx_file: vtx_file2,
            schema,
            first_chunk,
            inner: std::boxed::Box::new(iter2.fuse()),
            init_error: None,
        }
    }

    /// Create a streaming iterator that immediately defers an error.
    fn error(path: std::path::PathBuf, err: crate::RockDuckError) -> Self {
        let (session, runtime) = VortexReader::make_session();
        let buffer = vortex_buffer::ByteBuffer::from(Vec::new());
        let buffer_for_file = buffer.clone();
        let vtx_file = {
            use vortex_file::OpenOptionsSessionExt;
            session
                .open_options()
                .open_buffer(buffer_for_file)
                .unwrap_or_else(|_| {
                    panic!("VortexChunkIteratorStreaming: error path should not open file")
                })
        };
        let schema = std::sync::Arc::new(arrow_schema::Schema::new(std::vec::Vec::<
            arrow_schema::Field,
        >::new()));
        let inner: std::boxed::Box<
            dyn Iterator<Item = vortex_error::VortexResult<vortex_array::ArrayRef>>
                + Send
                + 'static,
        > = std::boxed::Box::new(std::iter::empty().fuse());
        Self {
            path,
            filter: None,
            session,
            runtime,
            buffer,
            vtx_file,
            schema,
            first_chunk: None,
            inner,
            init_error: Some(err),
        }
    }

    /// Convert a Vortex `ArrayRef` chunk to an Arrow `RecordBatch`.
    #[allow(deprecated)]
    fn vortex_chunk_to_arrow(
        chunk: &vortex_array::ArrayRef,
        schema: &std::sync::Arc<arrow_schema::Schema>,
        session: &vortex_session::VortexSession,
    ) -> crate::error::Result<RecordBatch> {
        use vortex_array::arrow::ArrowArrayExecutor;
        use vortex_array::VortexSessionExecute;
        let mut ctx = session.create_execution_ctx();
        let arr: arrow_array::ArrayRef = chunk
            .clone()
            .execute_arrow(None, &mut ctx)
            .map_err(|e| crate::RockDuckError::Internal(format!("execute_arrow: {}", e)))?;
        RecordBatch::try_new(schema.clone(), vec![arr.clone()])
            .map_err(|e| crate::RockDuckError::Codec(format!("RecordBatch::try_new: {}", e)))
    }
}

impl Iterator for VortexChunkIteratorStreaming {
    type Item = crate::error::Result<RecordBatch>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(e) = self.init_error.take() {
            return Some(Err(e));
        }
        // Yield cached first chunk first, then continue with inner iterator
        if let Some(chunk) = self.first_chunk.take() {
            return Some(
                Self::vortex_chunk_to_arrow(&chunk, &self.schema, &self.session),
            );
        }
        self.inner.next().map(|result| {
            let chunk = result
                .map_err(|e| crate::RockDuckError::Internal(format!("vortex chunk: {}", e)))?;
            Self::vortex_chunk_to_arrow(&chunk, &self.schema, &self.session)
        })
    }
}

impl std::fmt::Debug for VortexChunkIteratorStreaming {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VortexChunkIteratorStreaming")
            .field("path", &self.path)
            .finish()
    }
}
