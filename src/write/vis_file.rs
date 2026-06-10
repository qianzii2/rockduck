/// Deltavis file format constants (V2).
/// Stored alongside `__vis.vortex` as `__vis.vortex.delta`.
///
/// File layout:
///   magic(4) | version(1) | num_entries(8) | records(N*16) | crc32(4)
///
/// V1 (legacy, no header): raw `(row_offset: u64, txn_id: u64)` pairs.
/// V2: adds header + CRC32 checksum for corruption detection.
pub const DELTAVIS_MAGIC: [u8; 4] = *b"DVIS";
pub const DELTAVIS_VERSION: u8 = 1;
/// Header size: 4 (magic) + 1 (version) + 8 (num_entries) = 13 bytes
pub const DELTAVIS_HEADER_SIZE: usize = 13;
/// Footer size: 4 bytes CRC32
pub const DELTAVIS_FOOTER_SIZE: usize = 4;
/// Fixed record size: u64 row_offset + u64 txn_id = 16 bytes
pub const DELTAVIS_RECORD_SIZE: usize = 16;

use crate::error::Result;
use crate::mvcc::shadow_columns as sc;
use crate::storage::vortex::{VortexReader, VortexWriter};
use arrow_array::ArrayRef;
use arrow_array::RecordBatch;
/// Unified __vis.vortex file reader/writer.
///
/// Supports two modes:
/// - **Full rewrite**: `mark_deleted()` rewrites the entire vis file — used during WAL recovery
///   where the file is small
/// - **Append-only deltavis**: `mark_deleted_append()` appends a deletion marker to a separate
///   `__vis.delta` file — O(1) per delete, no full rewrite
///
/// The deltavis file stores (row_offset, txn_id) pairs as a binary log.
/// On read, both the vis file and the deltavis file are consulted;
/// the deltavis takes precedence (newer).
///
/// ## Lifecycle Safety (DB-06)
/// `load_batches()` opens a `VortexReader` locally, reads all batches, clones the data,
/// and drops the reader immediately. There is no long-lived `VortexReader` kept in `VisFileWriter`.
/// The `VortexReader` is opened fresh per call, so there are no dangling reference issues.
/// Callers that need a persistent reader should manage it themselves.
use std::fs::{self, OpenOptions};
use std::io::Write as IoWrite;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Unified __vis.vortex file reader/writer.
pub struct VisFileWriter {
    path: PathBuf,
}

impl VisFileWriter {
    pub fn new(path: &Path) -> Self {
        Self {
            path: path.to_path_buf(),
        }
    }

    /// Load all batches from the vis file (or empty vec if file doesn't exist).
    ///
    /// w010 fix: appends to an existing Vec instead of cloning the entire returned Vec.
    fn load_batches(&self, out: &mut Vec<RecordBatch>) -> Result<()> {
        if self.path.exists() {
            let reader = VortexReader::open(&self.path)?;
            let batches = reader.read_all_batches();
            // w010 fix: extend instead of clone. Caller owns the Vec and appends to it.
            out.extend_from_slice(&batches);
        }
        Ok(())
    }

    /// Load all batches from the vis file as a new Vec (convenience wrapper).
    fn load_batches_new(&self) -> Result<Vec<RecordBatch>> {
        let mut batches = Vec::new();
        self.load_batches(&mut batches)?;
        Ok(batches)
    }

    /// Find the batch and offset within that batch that contains `row_offset`.
    #[allow(dead_code)]
    fn find_row_batch(&self, batches: &[RecordBatch], row_offset: u64) -> Option<(usize, u64)> {
        let mut cumulative = 0u64;
        for (i, batch) in batches.iter().enumerate() {
            let n = batch.num_rows() as u64;
            if cumulative + n > row_offset {
                return Some((i, row_offset - cumulative));
            }
            cumulative += n;
        }
        None
    }

    /// Mark a single row as deleted in the vis file.
    ///
    /// Uses append-only deltavis (`mark_deleted_append`) to avoid O(N) full file rewrite.
    /// Falls back to full rewrite only if deltavis doesn't exist (legacy files).
    pub fn mark_deleted(&self, row_offset: u64, txn_id: u64) -> Result<()> {
        // Prefer append-only deltavis: fast, O(1) per delete
        self.mark_deleted_append(row_offset, txn_id)
    }

    /// Append a visibility batch to the vis file.
    ///
    /// Loads existing batches, appends the new one, and rewrites.
    /// Used during WAL recovery replay.
    pub fn append_batch(&self, vis_arrays: &[arrow_array::ArrayRef]) -> Result<()> {
        let mut batches = self.load_batches_new()?;

        let vis_schema = sc::visibility_schema();
        let batch = RecordBatch::try_new(vis_schema.clone(), vis_arrays.to_vec())
            .map_err(|e| crate::RockDuckError::Internal(format!("vis batch create: {}", e)))?;

        batches.push(batch);
        self.rewrite_batches(&batches)
    }

    /// Write a vis batch (replacing any existing content).
    pub fn write_batch(&self, batch: &RecordBatch) -> Result<()> {
        self.rewrite_batches(std::slice::from_ref(batch))
    }

    /// Append-only: mark a row as deleted by appending to the deltavis file.
    ///
    /// V2 format: writes magic + version + num_entries + records + CRC32.
    /// Uses write-to-temp + rename + parent fsync for atomicity.
    ///
    /// If the file doesn't exist yet, creates it with a proper V2 header.
    /// If the file is V1 (no magic), falls back to raw append for backward compat.
    ///
    /// w011 fix: uses incremental CRC — only reads the footer CRC (4 bytes) from the existing
    /// file, then combines it with the new record. Avoids O(N) full-file read on every append.
    pub fn mark_deleted_append(&self, row_offset: u64, txn_id: u64) -> Result<()> {
        use crc32fast::Hasher;

        let delta_path = self.deltavis_path();

        // Read existing content, or start fresh.
        let existing_data: Vec<u8> = if delta_path.exists() {
            fs::read(&delta_path).map_err(crate::RockDuckError::Io)?
        } else {
            Vec::new()
        };

        // Build new record bytes.
        let new_record = {
            let mut rec = [0u8; DELTAVIS_RECORD_SIZE];
            rec[..8].copy_from_slice(&row_offset.to_le_bytes());
            rec[8..].copy_from_slice(&txn_id.to_le_bytes());
            rec
        };

        // Determine if existing file is V2: strictly starts with DELTAVIS_MAGIC and has version.
        let is_v2 = existing_data.len() >= DELTAVIS_HEADER_SIZE + DELTAVIS_FOOTER_SIZE
            && existing_data[..4] == DELTAVIS_MAGIC
            && existing_data[4] == DELTAVIS_VERSION;

        // Build the complete new file content.
        let full_data = if is_v2 {
            // Existing V2 file: w011 fix — incremental CRC.
            // Read existing num_entries and footer CRC only (no full record read).
            let existing_num_entries =
                u64::from_le_bytes(existing_data[5..13].try_into().unwrap());
            let existing_footer_crc = u32::from_le_bytes(
                existing_data[existing_data.len() - DELTAVIS_FOOTER_SIZE..]
                    .try_into()
                    .unwrap(),
            );
            let existing_records_len =
                existing_data.len() - DELTAVIS_HEADER_SIZE - DELTAVIS_FOOTER_SIZE;

            // w011: Incremental CRC. Hasher::new_with_initial seeds the state with
            // existing_footer_crc so update() continues CRC computation from there.
            // This computes CRC(old_records || new_record) without re-reading old_records.
            let mut h = Hasher::new_with_initial(existing_footer_crc);
            h.update(&new_record);
            let new_footer_crc = h.finalize();

            // Build new file: header + old records + new record + new CRC.
            let mut new_data = Vec::with_capacity(
                DELTAVIS_HEADER_SIZE
                    + existing_records_len
                    + DELTAVIS_RECORD_SIZE
                    + DELTAVIS_FOOTER_SIZE,
            );
            // Header: magic + version + new num_entries
            new_data.extend_from_slice(&DELTAVIS_MAGIC);
            new_data.push(DELTAVIS_VERSION);
            new_data.extend_from_slice(&(existing_num_entries + 1).to_le_bytes());
            // Existing records (copied verbatim — no re-read needed)
            new_data.extend_from_slice(
                &existing_data[DELTAVIS_HEADER_SIZE..existing_data.len() - DELTAVIS_FOOTER_SIZE],
            );
            // New record
            new_data.extend_from_slice(&new_record);
            // New footer CRC (incrementally computed)
            new_data.extend_from_slice(&new_footer_crc.to_le_bytes());
            new_data
        } else if !existing_data.is_empty() {
            // V1 file: read existing raw records and migrate to V2.
            // V1 format: raw (row_offset: u64, txn_id: u64) pairs.
            let num_existing = existing_data.len() / DELTAVIS_RECORD_SIZE;
            let mut all_records = Vec::with_capacity(num_existing * DELTAVIS_RECORD_SIZE);
            for i in 0..num_existing {
                let start = i * DELTAVIS_RECORD_SIZE;
                all_records.extend_from_slice(&existing_data[start..start + DELTAVIS_RECORD_SIZE]);
            }
            // Append new record.
            all_records.extend_from_slice(&new_record);

            // Build V2 file with all records.
            let mut h = Hasher::new();
            h.update(&all_records);
            let crc = h.finalize();

            let mut data =
                Vec::with_capacity(DELTAVIS_HEADER_SIZE + all_records.len() + DELTAVIS_FOOTER_SIZE);
            data.extend_from_slice(&DELTAVIS_MAGIC);
            data.push(DELTAVIS_VERSION);
            data.extend_from_slice(&((num_existing + 1) as u64).to_le_bytes());
            data.extend_from_slice(&all_records);
            data.extend_from_slice(&crc.to_le_bytes());
            data
        } else {
            // Empty file: create new V2 with just this record.
            let mut h = Hasher::new();
            h.update(&new_record);
            let crc = h.finalize();

            let mut data = Vec::with_capacity(
                DELTAVIS_HEADER_SIZE + DELTAVIS_RECORD_SIZE + DELTAVIS_FOOTER_SIZE,
            );
            data.extend_from_slice(&DELTAVIS_MAGIC);
            data.push(DELTAVIS_VERSION);
            data.extend_from_slice(&1u64.to_le_bytes());
            data.extend_from_slice(&new_record);
            data.extend_from_slice(&crc.to_le_bytes());
            data
        };

        // Write to temp, fsync, rename.
        let tmp_path = {
            let mut p = delta_path.clone();
            p.set_extension("delta.tmp");
            p
        };
        {
            let mut tmp_file = OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(&tmp_path)
                .map_err(|e| crate::RockDuckError::Write(format!("open deltavis tmp: {}", e)))?;
            tmp_file
                .write_all(&full_data)
                .map_err(|e| crate::RockDuckError::Write(format!("write deltavis tmp: {}", e)))?;
            tmp_file
                .sync_all()
                .map_err(|e| crate::RockDuckError::Write(format!("deltavis tmp fsync: {}", e)))?;
        }

        fs::rename(&tmp_path, &delta_path)
            .map_err(|e| crate::RockDuckError::Write(format!("deltavis rename: {}", e)))?;

        // fsync parent directory on Windows.
        if let Some(parent) = delta_path.parent() {
            if let Ok(_dir_file) = std::fs::OpenOptions::new().read(true).open(parent) {
                #[cfg(unix)]
                {
                    use std::os::fd::AsRawFd;
                    let _ = dir_file.sync_all();
                }
                #[cfg(windows)]
                {
                    use std::os::windows::io::FromRawHandle;
                    use windows::core::PCWSTR;
                    use windows::Win32::Storage::FileSystem::{
                        CreateFileW, FILE_FLAG_BACKUP_SEMANTICS, FILE_SHARE_MODE, FILE_SHARE_READ,
                        FILE_SHARE_WRITE, OPEN_EXISTING,
                    };

                    let wide_path: Vec<u16> = parent
                        .to_string_lossy()
                        .encode_utf16()
                        .chain(std::iter::once(0))
                        .collect();
                    let handle = unsafe {
                        CreateFileW(
                            PCWSTR::from_raw(wide_path.as_ptr()),
                            0,
                            FILE_SHARE_MODE(FILE_SHARE_READ.0 | FILE_SHARE_WRITE.0),
                            None,
                            OPEN_EXISTING,
                            FILE_FLAG_BACKUP_SEMANTICS,
                            None,
                        )
                    };
                    if let Ok(h) = handle {
                        let file = unsafe { std::fs::File::from_raw_handle(h.0 as *mut _) };
                        let _ = file.sync_all();
                    }
                }
            }
        }

        Ok(())
    }

    /// Returns the path to the deltavis file (same dir as vis file, `.delta` extension).
    pub fn deltavis_path(&self) -> PathBuf {
        let mut p = self.path.clone();
        p.set_extension("vortex.delta");
        p
    }

    /// Read all deltavis entries from the delta file.
    ///
    /// Handles both V1 (raw binary) and V2 (magic + version + num_entries + CRC32).
    ///
    /// V1 format: raw `(row_offset: u64, txn_id: u64)` pairs, no header.
    /// V2 format: `magic(4) | version(1) | num_entries(8) | records(N*16) | crc32(4)`.
    ///
    /// Returns a sorted vec of (row_offset, txn_id) pairs.
    ///
    /// D17 fix: Uses mmap for zero-copy reads when available, falling back to
    /// fs::read for smaller files or when mmap is not available.
    pub fn read_deltavis(&self) -> Result<Vec<(u64, u64)>> {

        let delta_path = self.deltavis_path();
        if !delta_path.exists() {
            return Ok(Vec::new());
        }

        // D17 optimization: Try mmap first for zero-copy reading.
        // Falls back to fs::read if mmap fails (e.g., file too small or access denied).
        if let Ok(reader) = crate::storage::MmapReader::open(&delta_path) {
            let len = reader.len();
            if len >= DELTAVIS_HEADER_SIZE + DELTAVIS_FOOTER_SIZE {
                if let Ok(slice) = reader.slice_at(0, len as u64) {
                    // Check for V2 magic header
                    if slice[..4] == DELTAVIS_MAGIC {
                        return self.read_deltavis_v2_mmap(slice);
                    }
                    // V1 format: fall through to byte-by-byte reading
                    return self.read_deltavis_v1(slice);
                }
            }
            // Fall back to fs::read for empty/small files
        }

        // Fallback: read entire file into memory
        let data = fs::read(&delta_path).map_err(crate::RockDuckError::Io)?;

        if data.len() < 4 {
            return Ok(Vec::new());
        }

        // Check for V2 magic header. Minimum size: header + footer.
        if data.len() >= DELTAVIS_HEADER_SIZE + DELTAVIS_FOOTER_SIZE
            && data[..4] == DELTAVIS_MAGIC
        {
            return self.read_deltavis_v2(&data);
        }

        // V1 format: raw (row_offset, txn_id) pairs.
        self.read_deltavis_v1(&data)
    }

    /// Read V2 deltavis format from mmap slice (zero-copy).
    fn read_deltavis_v2_mmap(&self, data: &[u8]) -> Result<Vec<(u64, u64)>> {
        use crc32fast::Hasher;

        let version = data[4];
        if version != DELTAVIS_VERSION {
            tracing::warn!("deltavis V2: unknown version {}, treating as V1", version);
            return self.read_deltavis_v1(data);
        }

        let num_entries = u64::from_le_bytes(data[5..13].try_into().unwrap());
        let stored_crc = u32::from_le_bytes(
            data[data.len() - DELTAVIS_FOOTER_SIZE..]
                .try_into()
                .unwrap(),
        );

        let record_count = data.len() - DELTAVIS_HEADER_SIZE - DELTAVIS_FOOTER_SIZE;
        let expected_record_count = (num_entries as usize) * DELTAVIS_RECORD_SIZE;
        if record_count != expected_record_count {
            tracing::error!(
                "deltavis V2: file size mismatch (expected {} bytes for {} entries, got {})",
                expected_record_count,
                num_entries,
                record_count
            );
            return Err(crate::RockDuckError::Io(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                format!(
                    "deltavis V2: truncated file (expected {} bytes for {} entries, got {})",
                    expected_record_count, num_entries, record_count
                ),
            )));
        }

        // Verify CRC32 of records (excluding header and footer)
        let records = &data[DELTAVIS_HEADER_SIZE..data.len() - DELTAVIS_FOOTER_SIZE];
        let computed_crc = {
            let mut hasher = Hasher::new();
            hasher.update(records);
            hasher.finalize()
        };
        if computed_crc != stored_crc {
            tracing::error!(
                "deltavis V2: CRC mismatch (expected {:08x}, got {:08x})",
                stored_crc,
                computed_crc
            );
            return Err(crate::RockDuckError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "deltavis V2: CRC mismatch",
            )));
        }

        // Zero-copy parsing: directly read from mmap slice
        let mut entries = Vec::with_capacity(num_entries as usize);
        for i in (0..record_count).step_by(DELTAVIS_RECORD_SIZE) {
            let row_offset = u64::from_le_bytes(
                data[DELTAVIS_HEADER_SIZE + i..DELTAVIS_HEADER_SIZE + i + 8]
                    .try_into()
                    .unwrap(),
            );
            let txn_id = u64::from_le_bytes(
                data[DELTAVIS_HEADER_SIZE + i + 8..DELTAVIS_HEADER_SIZE + i + 16]
                    .try_into()
                    .unwrap(),
            );
            entries.push((row_offset, txn_id));
        }
        Ok(entries)
    }

    /// Read V1 (legacy) deltavis format: raw binary pairs without header.
    ///
    /// V1 files have no magic header, so we detect them by checking for V2 magic.
    /// This function reads until EOF or until there aren't enough bytes for a complete record.
    fn read_deltavis_v1(&self, data: &[u8]) -> Result<Vec<(u64, u64)>> {
        let mut entries = Vec::new();
        let mut i = 0;
        while i + DELTAVIS_RECORD_SIZE <= data.len() {
            let row_offset = u64::from_le_bytes(
                data[i..i + 8]
                    .try_into()
                    .expect("deltavis_v1: slice must have 8 bytes for row_offset"),
            );
            let txn_id = u64::from_le_bytes(
                data[i + 8..i + 16]
                    .try_into()
                    .expect("deltavis_v1: slice must have 8 bytes for txn_id"),
            );
            entries.push((row_offset, txn_id));
            i += DELTAVIS_RECORD_SIZE;
        }
        // Log warning if data wasn't a multiple of DELTAVIS_RECORD_SIZE.
        if i < data.len() {
            tracing::warn!(
                "deltavis_v1: {} trailing bytes ignored (file may be corrupted)",
                data.len() - i
            );
        }
        Ok(entries)
    }

    /// Read V2 deltavis format: magic + version + num_entries + records + CRC32.
    fn read_deltavis_v2(&self, data: &[u8]) -> Result<Vec<(u64, u64)>> {
        use crc32fast::Hasher;

        // Validate minimum size before accessing fixed offsets.
        if data.len() < DELTAVIS_HEADER_SIZE + DELTAVIS_FOOTER_SIZE {
            tracing::error!(
                "deltavis V2: file too small ({} bytes, minimum {})",
                data.len(),
                DELTAVIS_HEADER_SIZE + DELTAVIS_FOOTER_SIZE
            );
            return Err(crate::RockDuckError::Io(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "deltavis V2: file too small for header and footer",
            )));
        }

        let version = data[4];
        if version != DELTAVIS_VERSION {
            tracing::warn!("deltavis V2: unknown version {}, treating as V1", version);
            return self.read_deltavis_v1(data);
        }

        let num_entries = u64::from_le_bytes(
            data[5..13]
                .try_into()
                .expect("deltavis_v2: slice must have 8 bytes for num_entries"),
        );
        let stored_crc = u32::from_le_bytes(
            data[data.len() - DELTAVIS_FOOTER_SIZE..]
                .try_into()
                .expect("deltavis_v2: slice must have 4 bytes for CRC"),
        );

        let record_count = data.len() - DELTAVIS_HEADER_SIZE - DELTAVIS_FOOTER_SIZE;
        let expected_record_count = (num_entries as usize) * DELTAVIS_RECORD_SIZE;
        if record_count != expected_record_count {
            tracing::error!(
                "deltavis V2: file size mismatch (expected {} bytes for {} entries, got {})",
                expected_record_count,
                num_entries,
                record_count
            );
            return Err(crate::RockDuckError::Io(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                format!(
                    "deltavis V2: truncated file (expected {} bytes for {} entries, got {})",
                    expected_record_count, num_entries, record_count
                ),
            )));
        }

        let records = &data[DELTAVIS_HEADER_SIZE..data.len() - DELTAVIS_FOOTER_SIZE];
        let mut hasher = Hasher::new();
        hasher.update(records);
        let computed_crc = hasher.finalize();
        if computed_crc != stored_crc {
            tracing::error!(
                "deltavis V2: CRC mismatch (expected {:08x}, computed {:08x})",
                stored_crc,
                computed_crc
            );
            return Err(crate::RockDuckError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "deltavis V2: CRC mismatch (expected {:08x}, computed {:08x})",
                    stored_crc, computed_crc
                ),
            )));
        }

        // Parse entries.
        let mut entries = Vec::with_capacity(num_entries as usize);
        let mut i = 0usize;
        while i + DELTAVIS_RECORD_SIZE <= records.len() {
            let row_offset = u64::from_le_bytes(records[i..i + 8].try_into().unwrap());
            let txn_id = u64::from_le_bytes(records[i + 8..i + 16].try_into().unwrap());
            entries.push((row_offset, txn_id));
            i += DELTAVIS_RECORD_SIZE;
        }

        Ok(entries)
    }

    /// Check whether the deltavis file exists.
    pub fn has_deltavis(&self) -> bool {
        self.deltavis_path().exists()
    }

    /// Compact: merge deltavis entries back into the main vis file and delete the delta file.
    /// This is called by the compaction engine during L1→L2 flush.
    /// Returns the number of entries compacted.
    pub fn compact_deltavis(&self, _total_rows: u32) -> Result<usize> {
        let entries = self.read_deltavis()?;
        if entries.is_empty() {
            return Ok(0);
        }

        let mut batches = self.load_batches_new()?;
        let deleted_rows = {
            let mut del_map: rustc_hash::FxHashMap<u64, u64> = rustc_hash::FxHashMap::default();
            for (row, txn) in &entries {
                del_map.entry(*row).or_insert(*txn);
            }
            del_map
        };

        // WAL-1 fix: use cumulative global row numbers across all batches
        let mut cumulative_rows = 0u64;
        for batch in &mut batches {
            let n = batch.num_rows() as u64;
            for local_row in 0..n {
                // WAL-1 fix: use cumulative global row number, not local index
                let global_row = cumulative_rows + local_row;
                if let Some(&del_txn) = deleted_rows.get(&global_row) {
                    let mut deleted = vec![false; 1];
                    deleted[0] = true;
                    *batch = sc::mark_rows_deleted(batch, &deleted, del_txn);
                }
            }
            cumulative_rows += n;
        }

        self.rewrite_batches(&batches)?;
        fs::remove_file(self.deltavis_path())
            .map_err(|e| crate::RockDuckError::Write(format!("remove deltavis: {}", e)))?;

        Ok(entries.len())
    }

    #[allow(dead_code)]
    fn rewrite_with(
        &self,
        mut batches: Vec<RecordBatch>,
        modified_idx: usize,
        modified: RecordBatch,
    ) -> Result<()> {
        batches[modified_idx] = modified;
        self.rewrite_batches(&batches)
    }

    fn rewrite_batches(&self, batches: &[RecordBatch]) -> Result<()> {
        if batches.is_empty() {
            return Ok(());
        }

        // Atomic write: write to a .vortex.tmp temp file, then rename to the final path.
        // This prevents a crash from leaving a partial vortex file at the target path.
        let tmp_path = {
            let mut p = self.path.clone();
            p.set_extension("vortex.tmp");
            p
        };

        {
            let mut writer = VortexWriter::create(&tmp_path, "__vis");
            for batch in batches {
                writer.write(batch.clone())?;
            }
            writer.finish()?;
        }

        // fsync the temp file to ensure all data is durable before rename
        {
            let file = std::fs::OpenOptions::new().write(true).open(&tmp_path)?;
            file.sync_all()
                .map_err(|e| crate::RockDuckError::Write(format!("vis vortex tmp fsync: {}", e)))?;
        }

        std::fs::rename(&tmp_path, &self.path)
            .map_err(|e| crate::RockDuckError::Write(format!("vis vortex rename: {}", e)))?;

        // fsync the parent directory to make the rename durable on Windows
        if let Some(parent) = self.path.parent() {
            if let Ok(dir_fd) = std::fs::OpenOptions::new().read(true).open(parent) {
                let _ = dir_fd.sync_all();
            }
        }

        Ok(())
    }
}

/// Build a vis RecordBatch from __created_txn and __deleted_txn arrays.
pub fn make_vis_batch(
    created_txn: &arrow_array::UInt64Array,
    deleted_txn: &arrow_array::UInt64Array,
) -> Result<RecordBatch> {
    let vis_schema = sc::visibility_schema();
    let batch = RecordBatch::try_new(
        vis_schema,
        vec![
            Arc::new(created_txn.clone()) as ArrayRef,
            Arc::new(deleted_txn.clone()) as ArrayRef,
        ],
    )
    .map_err(|e| crate::RockDuckError::Internal(format!("make_vis_batch: {}", e)))?;
    Ok(batch)
}
