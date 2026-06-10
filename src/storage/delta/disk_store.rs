//! L2: Guard-Indexed Delta Storage on disk.
//!
//! # Architecture
//!
//! Replaces the fixed 1000-row bucket with PebblesDB-style Guards:
//! each Guard covers a key range [start_key, end_key) and contains multiple
//! overlapping patch files. This avoids full key-range rewrites during compaction,
//! reducing write amplification.
//!
//! ```text
//! File layout:
//!   {data_dir}/
//!     {seg_id}/
//!       {col}/
//!         {guard_id}.delta    # Patch file (append-only)
//!         {guard_id}.delta.zm # ZoneMap index
//! ```
//!
//! File format (compatible with existing DELT format):
//! ```ignore
//! Header (64 bytes):
//!   magic:       u32  = 0x44454C54 ("DELT")
//!   version:      u16  = 2 (guard-aware)
//!   patch_count:  u32
//!   total_rows:   u64
//!   min_txn:      u64
//!   max_txn:      u64
//!   guard_start:  u64  (row offset range start)
//!   guard_end:    u64  (row offset range end)
//!   reserved:     [0; 8]
//!
//! Patch (variable):
//!   patch_id:     u64
//!   txn_start:    u64
//!   txn_end:      u64
//!   format_flag:  u8   (0=sparse, 1=dense, 2=mask)
//!   affected:     u32
//!   payload_len:  u32
//!   payload:      [payload_len bytes]
//! ```
//!
//! References:
//! - PebblesDB FLSM (SOSP 2017): fragmented log-structured merge trees
//! - F1 Lightning (VLDB 2020): two-phase merge plan generation

use std::collections::BTreeMap;
use std::collections::HashMap;
use std::collections::HashSet;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write as IoWrite};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering as AtomicOrdering};
use std::sync::Arc;

use moka::sync::Cache;
use parking_lot::RwLock;
use rayon::prelude::*;

use super::sparsity::SparsitySelector;
use super::types::{DeltaCell, DeltaPatch, DeltaPatchFormat, ZoneMap};
use crate::error::{Result, RockDuckError};

/// Magic bytes at the start of every delta file: "DLTP" in ASCII.
const DELTA_FILE_MAGIC: u32 = 0x44504C54;

/// Maximum reasonable payload size: 256 MB.
/// Prevents OOM from corrupted payload_size fields in malicious or truncated files.
const MAX_PAYLOAD_SIZE: usize = 256 * 1024 * 1024;

// =============================================================================
// Guard -- PebblesDB-style key-range partition unit
// =============================================================================

/// A Guard partitions the key space by (row_offset range, column).
/// Guards are stored in a BTreeMap keyed by GuardKey, which is ordered,
/// enabling O(log n) guard lookup for any (seg_id, col, row_offset).
///
/// Unlike fixed-row buckets, guard boundaries are determined by the data
/// distribution -- reducing fragmentation for sparse or clustered workloads.
#[derive(Debug)]
pub struct Guard {
    /// Guard identifier: (seg_id, col, start_row)
    pub key: GuardKey,
    /// File path for this guard's patch file.
    pub file_path: PathBuf,
    /// ZoneMap index for this guard.
    pub zone_map: RwLock<GuardZoneMap>,
    /// Number of patch files currently in this guard.
    pub patch_count: AtomicU32,
    /// File handle (opened lazily).
    #[allow(dead_code)]
    handle: RwLock<Option<File>>,
}

/// Key for guard lookup and ordering.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GuardKey {
    pub seg_id: String,
    pub col: String,
    /// Inclusive start row of the guard's range.
    pub start_row: u64,
    /// Exclusive end row of the guard's range.
    pub end_row: u64,
}

impl PartialOrd for GuardKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for GuardKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Guards partition the key space by (seg_id, col, start_row).
        // end_row is metadata (derived from start_row), not part of the key.
        // Two guards with the same start_row but different end_rows should
        // compare equal for ordering purposes -- end_row is deduped.
        self.seg_id
            .cmp(&other.seg_id)
            .then(self.col.cmp(&other.col))
            .then(self.start_row.cmp(&other.start_row))
    }
}

impl Guard {
    pub fn new(key: GuardKey, file_path: PathBuf) -> Self {
        Self {
            key,
            file_path,
            zone_map: RwLock::new(GuardZoneMap::default()),
            patch_count: AtomicU32::new(0),
            handle: RwLock::new(None),
        }
    }

    pub fn contains_row(&self, row: u64) -> bool {
        row >= self.key.start_row && row < self.key.end_row
    }

    pub fn range(&self) -> (u64, u64) {
        (self.key.start_row, self.key.end_row)
    }
}

/// Per-guard ZoneMap and statistics.
#[derive(Debug, Clone, Default)]
pub struct GuardZoneMap {
    pub min_txn: u64,
    pub max_txn: u64,
    pub patch_count: u64,
    pub total_affected: u64,
    pub min_value: Option<Vec<u8>>,
    pub max_value: Option<Vec<u8>>,
}

impl GuardZoneMap {
    pub fn update(&mut self, patch_zm: &ZoneMap, patch_count: u64) {
        // Handle uninitialized state (all zeros) -- use patch values directly
        if self.min_txn == 0 || patch_zm.min_txn < self.min_txn {
            self.min_txn = patch_zm.min_txn;
        }
        if self.max_txn == 0 || patch_zm.max_txn > self.max_txn {
            self.max_txn = patch_zm.max_txn;
        }
        self.patch_count = patch_count;
        self.total_affected += patch_zm.affected_rows;
        if let (Some(min), Some(new_min)) = (&self.min_value, &patch_zm.min_value) {
            if new_min < min {
                self.min_value = Some(new_min.clone());
            }
        } else if patch_zm.min_value.is_some() {
            self.min_value = patch_zm.min_value.clone();
        }
        if let (Some(max), Some(new_max)) = (&self.max_value, &patch_zm.max_value) {
            if new_max > max {
                self.max_value = Some(new_max.clone());
            }
        } else if patch_zm.max_value.is_some() {
            self.max_value = patch_zm.max_value.clone();
        }
    }
}

// =============================================================================
// MergePlan -- F1 Lightning two-phase merge
// =============================================================================

/// F1 Lightning-style merge plan: Phase 1 (plan generation) + Phase 2 (apply).
///
/// Phase 1: Read blocks of keys from k input patches, generate a plan that
/// identifies which versions to keep and in what order.
///
/// Phase 2: Apply the plan column-by-column, producing a compacted patch.
#[derive(Debug)]
pub struct MergePlan {
    /// The guard being compacted.
    pub guard_key: GuardKey,
    /// Patches being merged.
    pub patches: Vec<MergePatchRef>,
    /// Per-column merge specs.
    pub column_plans: Vec<ColumnMergePlan>,
    /// Txn range of the merged output.
    pub output_txn_range: (u64, u64),
}

/// Reference to a patch for merge planning.
#[derive(Debug, Clone)]
pub struct MergePatchRef {
    pub patch_id: u64,
    pub seg_id: String,
    pub col: String,
    pub file_path: PathBuf,
    pub format: DeltaPatchFormat,
    pub txn_range: (u64, u64),
    /// Deduplication key: (guard_start_row, patch_id).
    /// Two patches with the same dedup key represent the same logical update.
    /// Used by CDC to avoid emitting duplicate changes from compacted guards.
    pub dedup_key: (u64, u64),
}

/// Per-column merge specification.
#[derive(Debug)]
pub struct ColumnMergePlan {
    pub col: String,
    /// Sorted row offsets from patches selected during merge planning.
    /// These are the rows that need to appear in the compacted output.
    pub kept_rows: Vec<u64>,
    pub output_format: DeltaPatchFormat,
}

// =============================================================================
// DeltaL2Disk -- Guard-indexed disk store
// =============================================================================

/// L2: Guard-indexed disk-resident delta storage.
///
/// ## Key design choices
///
/// - **Guard indexing**: Guards partition the key space by (seg_id, col, row_range).
///   Guard lookup is O(log n) via BTreeMap. Guards never need to be split
///   or merged -- new patches are simply appended to the appropriate guard.
///
/// - **Append-only patches**: Each patch is appended to its guard's file.
///   No in-place modification. Compaction creates new files.
///
/// - **Adaptive patch format**: Sparse (RoaringBitmap + Arrow IPC) vs
///   Dense (full Arrow column) selected by `SparsitySelector`.
///
/// - **Merge plan (F1 Lightning)**: Two-phase merge:
///   1. Generate plan (dedup by txn_id per (col, row))
///   2. Apply plan (vectorized Arrow kernels)
///
/// - **Async compaction**: Guard merge can run in background threads
///   (controlled by `async_compaction` config).
pub struct DeltaL2Disk {
    /// Guard index: sorted by GuardKey, O(log n) lookup.
    /// Guards are partitioned by (seg_id, col, row_range).
    guards: RwLock<BTreeMap<GuardKey, Guard>>,

    /// Auxiliary index: (seg_id, col, row_offset) → GuardKey.
    /// Provides O(1) row_offset → guard mapping.
    row_to_guard: RwLock<rustc_hash::FxHashMap<(String, String, u64), GuardKey>>,

    /// Guard compaction threshold (patches per guard before merge).
    /// Stored as AtomicU32 for interior mutability — can be updated at runtime
    /// via `set_compaction_threshold()` without requiring &mut self.
    compaction_threshold: std::sync::atomic::AtomicU32,

    /// Root data directory.
    root_dir: PathBuf,

    /// Sparsity selector for adaptive patch format.
    #[allow(dead_code)]
    sparsity_selector: SparsitySelector,

    /// File handles cache (opened lazily).
    handles: RwLock<rustc_hash::FxHashMap<GuardKey, Arc<RwLock<File>>>>,

    /// Configurable async compaction.
    async_compaction: bool,

    /// Compaction thread pool (when async).
    compaction_pool: RwLock<Option<Arc<rayon::ThreadPool>>>,

    /// Tracks in-progress guard merges to enable cleanup on failure.
    /// Key = GuardKey, Value = (merged_path, tmp_path).
    /// Inserted at merge start, removed on success or after cleanup.
    in_progress_merges: RwLock<HashSet<GuardKey>>,

    /// d016: LRU cache for `load_patches_from_guard` results.
    /// Key: GuardKey, Value: Vec<DeltaCell> decoded at `snapshot_txn`.
    /// Evicted on compaction (guard delete) to prevent stale reads.
    patch_cache: Cache<GuardKey, Vec<DeltaCell>, std::hash::BuildHasherDefault<rustc_hash::FxHasher>>,

    /// d017: Secondary index: (seg_id, col) → GuardKey.
    /// Provides O(log n) lookup when iterating guards for a segment in `get_visible`,
    /// replacing the previous O(n) BTreeMap range scan.
    seg_col_index: RwLock<BTreeMap<(String, String), Vec<GuardKey>>>,
}

impl DeltaL2Disk {
    /// Create a new L2 disk store.
    pub fn new(root_dir: PathBuf) -> Self {
        let _ = fs::create_dir_all(&root_dir);
        Self {
            guards: RwLock::new(BTreeMap::new()),
            row_to_guard: RwLock::new(rustc_hash::FxHashMap::default()),
            compaction_threshold: std::sync::atomic::AtomicU32::new(64),
            root_dir,
            sparsity_selector: SparsitySelector::new(),
            handles: RwLock::new(rustc_hash::FxHashMap::default()),
            async_compaction: false,
            compaction_pool: RwLock::new(None),
            in_progress_merges: RwLock::new(HashSet::new()),
            // d016: 64-entry LRU cache, ~50 MB max (assume avg 800 KB per guard result)
            patch_cache: Cache::builder()
                .max_capacity(64)
                .weigher(|_key, value: &Vec<DeltaCell>| value.len() as u32)
                .build_with_hasher(Default::default()),
            // d017: secondary index for O(log n) seg_id+col → guards lookup
            seg_col_index: RwLock::new(BTreeMap::new()),
        }
    }

    /// Get all segment IDs that have guards in L2.
    pub fn get_segment_ids(&self) -> Vec<String> {
        let guards = self.guards.read();
        let mut seg_ids = std::collections::HashSet::new();
        for key in guards.keys() {
            seg_ids.insert(key.seg_id.clone());
        }
        seg_ids.into_iter().collect()
    }

    /// Configure async compaction with a thread pool.
    ///
    /// Returns `Ok(self)` on success, or an error if the thread pool could not be built.
    /// The error is logged and the scheduler falls back to sync (no compaction) mode.
    pub fn with_async_compaction(
        mut self,
        async_enabled: bool,
        num_threads: usize,
    ) -> std::result::Result<Self, RockDuckError> {
        self.async_compaction = async_enabled;
        if async_enabled && num_threads > 0 {
            match rayon::ThreadPoolBuilder::new()
                .num_threads(num_threads)
                .thread_name(|i| format!("delta-l2-compact-{}", i))
                .build()
            {
                Ok(pool) => {
                    self.compaction_pool = RwLock::new(Some(Arc::new(pool)));
                }
                Err(e) => {
                    tracing::error!(
                        "DeltaL2Disk: failed to build compaction thread pool ({} threads): {}. \
                         Compaction will run synchronously in the caller thread.",
                        num_threads,
                        e
                    );
                    self.async_compaction = false;
                }
            }
        }
        Ok(self)
    }

    /// Set the compaction threshold (patches per guard before triggering merge).
    ///
    /// Uses interior mutability via `AtomicU32`, so no `&mut self` required.
    /// This allows runtime reconfiguration of the compaction policy without
    /// restarting the store.
    pub fn set_compaction_threshold(&self, threshold: u32) {
        self.compaction_threshold
            .store(threshold, std::sync::atomic::Ordering::Relaxed);
    }

    /// Get the path for a guard's delta file.
    pub fn guard_path(&self, guard_key: &GuardKey) -> PathBuf {
        self.root_dir
            .join(&guard_key.seg_id)
            .join(&guard_key.col)
            .join(format!(
                "guard_{}_{}.delta",
                guard_key.start_row, guard_key.end_row
            ))
    }

    /// Find the guard that contains the given (seg_id, col, row_offset).
    /// O(log n) guard lookup + O(1) auxiliary index.
    fn find_guard(&self, seg_id: &str, col: &str, row: u64) -> Option<GuardKey> {
        // First try the auxiliary index (O(1))
        let idx_key = (seg_id.to_string(), col.to_string(), row);
        let guard_key = self.row_to_guard.read().get(&idx_key).cloned();

        if guard_key.is_some() {
            return guard_key;
        }

        // Fall back to BTreeMap range scan: find guard with start_row <= row < end_row
        let guards = self.guards.read();
        for (gk, _) in guards.iter() {
            if gk.seg_id == seg_id && gk.col == col && row >= gk.start_row && row < gk.end_row {
                return Some(gk.clone());
            }
        }
        None
    }

    /// Find or create the guard for a given (seg_id, col, row_offset).
    /// Guard size: 10,000 rows by default.
    pub fn find_or_create_guard(&self, seg_id: &str, col: &str, row: u64) -> GuardKey {
        let guard_size = 10_000u64;
        let start_row = (row / guard_size) * guard_size;
        let end_row = start_row.saturating_add(guard_size);
        let key = GuardKey {
            seg_id: seg_id.to_string(),
            col: col.to_string(),
            start_row,
            end_row,
        };

        // Check if guard exists — acquire write lock immediately to avoid TOCTOU race
        let mut guards = self.guards.write();
        if let Some(existing) = guards.get(&key) {
            return existing.key.clone();
        }

        // Create new guard
        let file_path = self.guard_path(&key);
        let guard = Guard::new(key.clone(), file_path.clone());

        if let Some(parent) = file_path.parent() {
            let _ = fs::create_dir_all(parent);
        }

        guards.insert(key.clone(), guard);

        // Populate auxiliary index for all rows in this guard's range
        let mut idx = self.row_to_guard.write();
        let (start, end) = (key.start_row, key.end_row);
        for r in start..end {
            idx.insert((key.seg_id.clone(), key.col.clone(), r), key.clone());
        }
        drop(idx);

        // d017: maintain secondary index (seg_id, col) → [GuardKey]
        {
            let mut sidx = self.seg_col_index.write();
            sidx
                .entry((key.seg_id.clone(), key.col.clone()))
                .or_default()
                .push(key.clone());
        }

        key
    }

    /// Append a patch to the L2 store.
    ///
    /// Finds or creates the appropriate guard and appends the patch to its file.
    /// Uses atomic write: create .tmp → write → fsync → rename.
    pub fn append_patch(&self, seg_id: &str, col: &str, patch: &DeltaPatch) -> Result<()> {
        // Guard is partitioned by (seg_id, col, row_range). Use seg_id+col to find or create
        // the guard -- the actual row_offset lives inside the patch's positions/values, not in
        // the guard key itself. Using min_txn as u32 would truncate u64 txn_ids.
        let guard_key = self.find_or_create_guard(seg_id, col, 0);
        let guard_path = self.guard_path(&guard_key);
        let tmp_path = {
            let mut p = guard_path.clone();
            p.set_extension("delta.tmp");
            p
        };

        if let Some(parent) = guard_path.parent() {
            fs::create_dir_all(parent)?;
        }

        // Serialize the patch payload
        let payload = patch.format.to_bytes();
        let affected = patch.zone_map.affected_rows;

        // Build patch binary
        let mut patch_bytes = Vec::with_capacity(32 + payload.len());
        patch_bytes.extend_from_slice(&patch.patch_id.to_le_bytes());
        patch_bytes.extend_from_slice(&patch.txn_range.0.to_le_bytes());
        patch_bytes.extend_from_slice(&patch.txn_range.1.to_le_bytes());
        patch_bytes.push(if patch.format.is_sparse() { 0x00 } else { 0x01 });
        patch_bytes.extend_from_slice(&affected.to_le_bytes());
        patch_bytes.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        patch_bytes.extend_from_slice(&payload);

        let mut f = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&tmp_path)?;

        // On new file creation, write a magic header (64 bytes) so readers can validate the format.
        if f.metadata()?.len() == 0 {
            let mut header = [0u8; 64];
            header[0..4].copy_from_slice(&DELTA_FILE_MAGIC.to_le_bytes());
            f.write_all(&header)?;
        }

        f.write_all(&patch_bytes)?;
        f.sync_all()?;
        fs::rename(&tmp_path, &guard_path)?;
        //: sync parent directory after rename to ensure durability.
        // Without this, a crash after rename but before directory fsync can lose the file.
        if let Some(parent) = tmp_path.parent() {
            if let Ok(_dir_fd) = std::fs::OpenOptions::new().read(true).open(parent) {
                #[cfg(unix)]
                {
                    use std::os::fd::AsRawFd;
                    let _ = std::fs::File::from(dir_fd).sync_all();
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
                            0, // dwDesiredAccess: no read/write needed
                            FILE_SHARE_MODE(FILE_SHARE_READ.0 | FILE_SHARE_WRITE.0),
                            None, // lpSecurityAttributes
                            OPEN_EXISTING,
                            FILE_FLAG_BACKUP_SEMANTICS,
                            None, // hTemplateFile
                        )
                    };
                    if let Ok(h) = handle {
                        let file = unsafe { File::from_raw_handle(h.0 as *mut _) };
                        let _ = file.sync_all(); // non-fatal
                    }
                }
            }
        }

        // Update guard ZoneMap
        self.update_guard_zone_map(&guard_key, &patch.zone_map, affected);

        // Increment patch count and check threshold
        self.increment_patch_count(&guard_key);

        Ok(())
    }

    fn update_guard_zone_map(&self, guard_key: &GuardKey, patch_zm: &ZoneMap, affected: u64) {
        let guards = self.guards.read();
        if let Some(guard) = guards.get(guard_key) {
            let mut zm = guard.zone_map.write();
            zm.update(patch_zm, affected);
        }
    }

    fn increment_patch_count(&self, guard_key: &GuardKey) {
        let guards = self.guards.read();
        if let Some(guard) = guards.get(guard_key) {
            let new_count = guard.patch_count.fetch_add(1, AtomicOrdering::Relaxed) + 1;
            if new_count
                >= self
                    .compaction_threshold
                    .load(std::sync::atomic::Ordering::Relaxed)
            {
                tracing::debug!(
                    "Guard {:?} has {} patches, threshold {}",
                    guard_key,
                    new_count,
                    self.compaction_threshold
                        .load(std::sync::atomic::Ordering::Relaxed)
                );
            }
        }
    }

    /// Get all visible deltas for a segment at a given snapshot.
    pub fn get_visible(&self, seg_id: &str, snapshot_txn: u64) -> Result<Vec<DeltaCell>> {
        // d017: O(log n) lookup via secondary index instead of O(n) BTreeMap scan.
        // Collect all guard keys for this seg_id (across all columns).
        let guard_keys: Vec<GuardKey> = {
            let sidx = self.seg_col_index.read();
            let mut keys = Vec::new();
            for ((sid, _col), gkeys) in sidx.iter() {
                if sid == seg_id {
                    keys.extend(gkeys.iter().cloned());
                }
            }
            keys
        };

        if guard_keys.is_empty() {
            return Ok(Vec::new());
        }

        let mut all_deltas = Vec::new();
        let mut corruption_err: Option<RockDuckError> = None;

        // Iterate over guards for this segment (now O(#cols) lookups instead of O(n))
        let guards = self.guards.read();
        for guard_key in guard_keys {
            let guard = match guards.get(&guard_key) {
                Some(g) => g,
                None => continue,
            };

            // ZoneMap prune: skip if all patches are older than snapshot
            let zm = guard.zone_map.read();
            if zm.max_txn < snapshot_txn {
                continue;
            }
            drop(zm);

            // d016: load from cache if available
            if let Some(cached) = self.patch_cache.get(&guard_key) {
                all_deltas.extend(cached.iter().cloned());
                continue;
            }

            // Load patches from this guard — propagate decode errors up
            match self.load_patches_from_guard(seg_id, guard, snapshot_txn) {
                Ok(deltas) => {
                    // d016: cache the result for future calls
                    if !deltas.is_empty() {
                        self.patch_cache.insert(guard_key.clone(), deltas.clone());
                    }
                    all_deltas.extend(deltas);
                }
                Err(e) => {
                    // Record the first corruption error but keep processing other guards.
                    // This lets the caller see partial results while still propagating the error.
                    if corruption_err.is_none() {
                        corruption_err = Some(e);
                    }
                }
            }
        }

        match corruption_err {
            Some(err) => Err(err),
            None => Ok(all_deltas),
        }
    }

    fn load_patches_from_guard(
        &self,
        _seg_id: &str,
        guard: &Guard,
        snapshot_txn: u64,
    ) -> Result<Vec<DeltaCell>> {
        if !guard.file_path.exists() {
            return Ok(Vec::new());
        }

        let mut f = File::open(&guard.file_path)?;
        let file_size = f.metadata()?.len();

        // Validate magic header on file open.
        if file_size >= 64 {
            let mut magic_buf = [0u8; 4];
            f.seek(SeekFrom::Start(0))?;
            if f.read_exact(&mut magic_buf).is_ok() {
                let magic = u32::from_le_bytes(magic_buf);
                if magic != DELTA_FILE_MAGIC {
                    tracing::error!(
                        "load_patches_from_guard: invalid magic in {:?}: expected {:08x}, got {:08x}",
                        guard.file_path, DELTA_FILE_MAGIC, magic
                    );
                    return Err(RockDuckError::Delta(format!(
                        "invalid delta file magic: expected {:08x}, got {:08x}",
                        DELTA_FILE_MAGIC, magic
                    )));
                }
            }
        }

        let mut deltas = Vec::new();
        let mut offset: u64 = 64; // Skip header

        loop {
            if offset >= file_size {
                break;
            }

            if f.seek(SeekFrom::Start(offset)).is_err() {
                break;
            }

            let mut ph = [0u8; 32];
            match f.read_exact(&mut ph) {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => {
                    tracing::warn!(
                        "load_patches_from_guard: failed to read patch header at offset {}: {}",
                        offset,
                        e
                    );
                    break;
                }
            }

            let txn_end = u64::from_le_bytes(ph[16..24].try_into().unwrap());
            let payload_size = u32::from_le_bytes(ph[24..28].try_into().unwrap()) as usize;

            // OOM guard: reject implausibly large payload sizes.
            if payload_size > MAX_PAYLOAD_SIZE {
                tracing::error!(
                    "load_patches_from_guard: payload_size {} exceeds maximum {} at offset {}",
                    payload_size,
                    MAX_PAYLOAD_SIZE,
                    offset
                );
                return Err(RockDuckError::Delta(format!(
                    "implausible payload_size {} at offset {} (max {})",
                    payload_size, offset, MAX_PAYLOAD_SIZE
                )));
            }

            // Sanity bound: payload_size should not exceed remaining file bytes.
            let remaining = file_size.saturating_sub(offset + 32);
            if payload_size as u64 > remaining {
                tracing::error!(
                    "load_patches_from_guard: payload_size {} exceeds remaining file bytes {} at offset {}",
                    payload_size, remaining, offset
                );
                return Err(RockDuckError::Delta(format!(
                    "payload_size {} exceeds remaining file bytes {} at offset {}",
                    payload_size, remaining, offset
                )));
            }

            if txn_end > snapshot_txn {
                offset += 32 + payload_size as u64;
                continue;
            }

            let mut payload = vec![0u8; payload_size];
            if f.read_exact(&mut payload).is_err() {
                tracing::warn!(
                    "load_patches_from_guard: truncated payload at offset {}, size {}",
                    offset,
                    payload_size
                );
                break;
            }

            let format = match DeltaPatchFormat::from_bytes(&payload) {
                Some(f) => f,
                None => {
                    tracing::error!(
                        "load_patches_from_guard: failed to parse DeltaPatchFormat at offset {} in {:?}",
                        offset, guard.file_path
                    );
                    return Err(RockDuckError::Delta(format!(
                        "failed to parse DeltaPatchFormat at offset {}",
                        offset
                    )));
                }
            };

            let patch_deltas = patch_to_deltas(&format)?;
            deltas.extend(patch_deltas);

            offset += 32 + payload_size as u64;
        }

        Ok(deltas)
    }

    /// D20 fix: True batch query — get visible deltas for multiple segments with parallel I/O.
    ///
    /// Strategy:
    /// 1. Collect all GuardKeys for all segments via seg_col_index (O(log n) per segment)
    /// 2. ZoneMap prune: skip guards where `max_txn < snapshot_txn` (no file I/O needed)
    /// 3. LRU cache check: skip file I/O for cache hits
    /// 4. Open remaining guard files in parallel using rayon thread pool
    /// 5. Decode patches and cache results
    /// 6. Return per-segment results grouped by seg_id
    pub fn get_visible_batch(
        &self,
        seg_ids: &[String],
        snapshot_txn: u64,
    ) -> Result<HashMap<String, Vec<DeltaCell>>> {
        // Step 1: Collect all GuardKeys for all segments via seg_col_index
        let all_guard_keys: Vec<GuardKey> = {
            let sidx = self.seg_col_index.read();
            let mut keys = Vec::new();
            for seg_id in seg_ids {
                for ((sid, _col), gkeys) in sidx.iter() {
                    if sid == seg_id {
                        keys.extend(gkeys.iter().cloned());
                    }
                }
            }
            keys
        };

        if all_guard_keys.is_empty() {
            return Ok(seg_ids.iter().map(|s| (s.clone(), Vec::new())).collect());
        }

        // Step 2: Classify guards — cache hit, zone-pruned, or needs file I/O
        let guards = self.guards.read();
        let mut needs_io: Vec<GuardKey> = Vec::new();
        let mut cached: HashMap<GuardKey, Vec<DeltaCell>> = HashMap::default();
        let mut skipped: HashSet<GuardKey> = HashSet::default();

        for guard_key in &all_guard_keys {
            let guard = match guards.get(guard_key) {
                Some(g) => g,
                None => continue,
            };

            // ZoneMap prune: skip if all patches are older than snapshot
            let zm = guard.zone_map.read();
            if zm.max_txn < snapshot_txn {
                skipped.insert(guard_key.clone());
                continue;
            }
            drop(zm);

            // LRU cache hit
            if let Some(cached_deltas) = self.patch_cache.get(guard_key) {
                cached.insert(guard_key.clone(), cached_deltas.to_vec());
            } else {
                needs_io.push(guard_key.clone());
            }
        }
        drop(guards);

        // Step 3: Parallel file I/O for uncached guards
        if !needs_io.is_empty() {
            // Read guard files in parallel using rayon global pool
            let loaded: Vec<(GuardKey, Vec<DeltaCell>)> = needs_io
                .par_iter()
                .filter_map(|guard_key| {
                    // Get file_path and zone_map_max_txn under read lock, then release lock
                    let file_path: std::path::PathBuf;
                    let zone_map_max_txn: u64;
                    
                    {
                        let guards = self.guards.read();
                        match guards.get(guard_key) {
                            Some(g) => {
                                file_path = g.file_path.clone();
                                let zm = g.zone_map.read();
                                zone_map_max_txn = zm.max_txn;
                            }
                            None => return None,
                        }
                    }

                    // Re-check zone map pruning
                    if zone_map_max_txn < snapshot_txn {
                        return Some((guard_key.clone(), Vec::new()));
                    }

                    // Create a minimal guard with just the file path for loading
                    let guard = Guard::new(guard_key.clone(), file_path);

                    match self.load_patches_from_guard(&guard_key.seg_id, &guard, snapshot_txn) {
                        Ok(deltas) => Some((guard_key.clone(), deltas)),
                        Err(e) => {
                            tracing::warn!(
                                "get_visible_batch: failed to load guard {:?}: {}",
                                guard_key,
                                e
                            );
                            Some((guard_key.clone(), Vec::new()))
                        }
                    }
                })
                .collect();

            // Step 4: Cache results and merge into cached map
            for (guard_key, deltas) in loaded {
                if !deltas.is_empty() {
                    self.patch_cache.insert(guard_key.clone(), deltas.clone());
                }
                cached.insert(guard_key, deltas);
            }
        }

        // Step 5: Group by segment and return
        let mut results: HashMap<String, Vec<DeltaCell>> =
            seg_ids.iter().map(|s| (s.clone(), Vec::new())).collect();
        for guard_key in &all_guard_keys {
            if let Some(deltas) = cached.get(guard_key) {
                if let Some(seg_result) = results.get_mut(&guard_key.seg_id) {
                    seg_result.extend(deltas.iter().cloned());
                }
            }
        }

        Ok(results)
    }

    /// Get a single cell delta.
    pub fn get_cell(
        &self,
        seg_id: &str,
        row_offset: u64,
        column: &str,
        snapshot_txn: u64,
    ) -> Result<Option<DeltaCell>> {
        let guard_key = match self.find_guard(seg_id, column, row_offset) {
            Some(k) => k,
            None => return Ok(None),
        };

        let guards = self.guards.read();
        let guard = match guards.get(&guard_key) {
            Some(g) => g,
            None => return Ok(None),
        };

        let deltas = self.load_patches_from_guard(seg_id, guard, snapshot_txn)?;
        let candidates: Vec<_> = deltas
            .into_iter()
            .filter(|d| d.row_offset == row_offset && d.column == column)
            .collect();

        Ok(candidates.into_iter().max_by_key(|d| d.txn_id))
    }

    /// Generate a merge plan for a guard (F1 Lightning Phase 1).
    ///
    /// Reads all patches in the guard, deduplicates by (col, row_offset),
    /// keeping the highest txn_id version.
    pub fn generate_merge_plan(&self, guard_key: &GuardKey) -> Result<Option<MergePlan>> {
        let guards = self.guards.read();
        let guard = match guards.get(guard_key) {
            Some(g) => g,
            None => return Ok(None),
        };

        if !guard.file_path.exists() {
            return Ok(None);
        }

        // Read all patches from the guard file
        let patches = self.read_all_patches(guard)?;
        if patches.is_empty() {
            return Ok(None);
        }

        // Deduplicate by (col, row_offset) -- keep highest txn_id.
        // Per-row dedup: extract actual row positions from each patch.
        let mut dedup: BTreeMap<(String, u64), (u64, MergePatchRef)> = BTreeMap::new();
        let mut min_txn = u64::MAX;
        let mut max_txn = u64::MIN;

        for patch in &patches {
            min_txn = min_txn.min(patch.txn_range.0);
            max_txn = max_txn.max(patch.txn_range.1);

            // Extract row positions from the patch for per-row dedup
            let rows = patch_row_positions(&patch.format);
            for row in rows {
                let key = (patch.col.clone(), row);
                // Keep the patch with the highest txn_id for this (col, row)
                if let Some((_, existing)) = dedup.get(&key) {
                    if patch.txn_range.1 <= existing.txn_range.1 {
                        continue;
                    }
                }
                dedup.insert(key, (patch.txn_range.1, patch.clone()));
            }
        }

        // Group by column: for each column, collect the row positions from the winning patches.
        let mut col_groups: BTreeMap<String, Vec<u64>> = BTreeMap::new();
        for ((col, row), (_, _patch)) in dedup {
            col_groups.entry(col).or_default().push(row);
        }

        let column_plans: Vec<ColumnMergePlan> = col_groups
            .into_iter()
            .map(|(col, mut kept_rows)| {
                kept_rows.sort_unstable();
                // kept_rows is the list of row positions to include in the merged output.
                // These are sorted so the compacted patch has deterministic row order.
                ColumnMergePlan {
                    col,
                    kept_rows,
                    output_format: DeltaPatchFormat::Dense {
                        values: Arc::new(Vec::new()),
                        total_rows: 0,
                        txn_id: 0,
                    },
                }
            })
            .collect();

        Ok(Some(MergePlan {
            guard_key: guard_key.clone(),
            patches,
            column_plans,
            output_txn_range: (min_txn, max_txn),
        }))
    }

    pub fn read_all_patches(&self, guard: &Guard) -> Result<Vec<MergePatchRef>> {
        if !guard.file_path.exists() {
            return Ok(Vec::new());
        }

        let mut f = File::open(&guard.file_path)?;
        let file_size = f.metadata()?.len();

        // Validate magic header.
        if file_size >= 64 {
            let mut magic_buf = [0u8; 4];
            f.seek(SeekFrom::Start(0))?;
            if f.read_exact(&mut magic_buf).is_ok() {
                let magic = u32::from_le_bytes(magic_buf);
                if magic != DELTA_FILE_MAGIC {
                    tracing::error!(
                        "read_all_patches: invalid magic in {:?}: expected {:08x}, got {:08x}",
                        guard.file_path,
                        DELTA_FILE_MAGIC,
                        magic
                    );
                    return Err(RockDuckError::Delta(format!(
                        "invalid delta file magic: expected {:08x}, got {:08x}",
                        DELTA_FILE_MAGIC, magic
                    )));
                }
            }
        }

        let mut patches = Vec::new();
        let mut offset: u64 = 64;

        loop {
            if offset >= file_size {
                break;
            }
            if f.seek(SeekFrom::Start(offset)).is_err() {
                break;
            }

            let mut ph = [0u8; 32];
            match f.read_exact(&mut ph) {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => {
                    tracing::warn!(
                        "read_all_patches: failed to read header at offset {}: {}",
                        offset,
                        e
                    );
                    break;
                }
            }

            let payload_size = u32::from_le_bytes(ph[24..28].try_into().unwrap()) as usize;

            // OOM guard.
            if payload_size > MAX_PAYLOAD_SIZE {
                tracing::error!(
                    "read_all_patches: payload_size {} exceeds maximum {} at offset {}",
                    payload_size,
                    MAX_PAYLOAD_SIZE,
                    offset
                );
                return Err(RockDuckError::Delta(format!(
                    "implausible payload_size {} at offset {} (max {})",
                    payload_size, offset, MAX_PAYLOAD_SIZE
                )));
            }

            let remaining = file_size.saturating_sub(offset + 32);
            if payload_size as u64 > remaining {
                tracing::error!(
                    "read_all_patches: payload_size {} exceeds remaining {} at offset {}",
                    payload_size,
                    remaining,
                    offset
                );
                return Err(RockDuckError::Delta(format!(
                    "payload_size {} exceeds remaining file bytes {} at offset {}",
                    payload_size, remaining, offset
                )));
            }

            let mut payload = vec![0u8; payload_size];
            if f.read_exact(&mut payload).is_err() {
                tracing::warn!(
                    "read_all_patches: truncated payload at offset {}, size {}",
                    offset,
                    payload_size
                );
                break;
            }

            // B3.9 fix: propagate format parse errors instead of silently falling back to a
            // bogus Dense patch with total_rows=0, txn_id=0. A corrupted patch is skipped with
            // a clear error message rather than producing incorrect query results downstream.
            let format = match DeltaPatchFormat::from_bytes(&payload) {
                Some(f) => f,
                None => {
                    tracing::error!(
                        "read_all_patches: corrupted delta patch format at offset {} (invalid bytes). \
                         Skipping {} bytes of invalid payload.",
                        offset, payload_size
                    );
                    offset += payload_size as u64;
                    continue;
                }
            };

            let patch_id = u64::from_le_bytes(ph[0..8].try_into().unwrap());
            let txn_start = u64::from_le_bytes(ph[8..16].try_into().unwrap());
            let txn_end = u64::from_le_bytes(ph[16..24].try_into().unwrap());

            patches.push(MergePatchRef {
                patch_id,
                seg_id: guard.key.seg_id.clone(),
                col: guard.key.col.clone(),
                file_path: guard.file_path.clone(),
                format,
                txn_range: (txn_start, txn_end),
                dedup_key: (guard.key.start_row, patch_id),
            });

            offset += 32 + payload_size as u64;
        }

        Ok(patches)
    }

    /// Execute a merge plan (F1 Lightning Phase 2).
    ///
    /// This is called after `generate_merge_plan`. The plan specifies which
    /// (col, row_offset) versions to keep. This implementation writes a new,
    /// compacted patch file.
    ///
    /// When `async_compaction` is enabled, this runs in a background thread pool.
    pub fn execute_merge_plan(&self, plan: MergePlan) -> Result<()> {
        self.execute_merge_plan_sync(&plan)
    }

    fn execute_merge_plan_sync(&self, plan: &MergePlan) -> Result<()> {
        use super::merge::DeltaMerger;
        use std::collections::BTreeMap;
        use std::sync::Arc;

        tracing::info!(
            "Executing merge plan for guard {:?}: {} columns, txn range {:?}",
            plan.guard_key,
            plan.column_plans.len(),
            plan.output_txn_range
        );

        let _merger = DeltaMerger::new();

        // D5 optimization: Local cache to avoid repeated decoding of the same patch.
        // Key: Arc of the patch values data (the IPC bytes), Value: decoded Arrow array.
        // This significantly reduces CPU when merging dense patches with many rows.
        let mut patch_decode_cache: std::collections::HashMap<
            Arc<Vec<u8>>,
            std::sync::Arc<dyn arrow_array::Array>,
            std::collections::hash_map::RandomState,
        > = std::collections::HashMap::with_hasher(Default::default());

        // Read all source patches (column → patch → row_positions)
        let mut col_patch_data: BTreeMap<String, Vec<(MergePatchRef, Vec<u64>)>> = BTreeMap::new();
        for patch in &plan.patches {
            let positions = patch_row_positions(&patch.format);
            col_patch_data
                .entry(patch.col.clone())
                .or_default()
                .push((patch.clone(), positions));
        }

        for col_plan in &plan.column_plans {
            // Phase 2: merge patches for this column using kept rows
            let kept_rows = &col_plan.kept_rows;
            if kept_rows.is_empty() {
                continue;
            }

            // Read patch data for this column
            let patches_data = col_patch_data.get(&col_plan.col);

            // Build merged sparse patch for this column
            // Strategy: build a bitmap of kept rows and gather their values
            let mut merged_positions: Vec<u64> = Vec::with_capacity(kept_rows.len());
            let mut merged_values: Vec<Vec<u8>> = Vec::with_capacity(kept_rows.len());

            for &row in kept_rows {
                // Find the newest patch that covers this row
                let mut found_value: Option<Vec<u8>> = None;
                if let Some(patches_for_col) = patches_data {
                    // Sort by txn_range descending (newest first)
                    for (patch, positions) in patches_for_col {
                        if positions.contains(&row) {
                            if let Some(value) =
                                self.read_patch_value_cached(patch, row, &mut patch_decode_cache)?
                            {
                                found_value = Some(value);
                                break; // Newest patch wins
                            }
                        }
                    }
                }

                merged_positions.push(row);
                merged_values.push(found_value.unwrap_or_default());
            }

            // Build sparse bitmap from merged positions
            let mut bitmap = croaring::Bitmap::new();
            for pos in &merged_positions {
                bitmap.add(*pos as u32);
            }

            // Encode values as Arrow IPC
            let values_ipc = self.encode_values_ipc(&merged_values)?;

            // Write compacted patch for this column
            let compacted = DeltaPatchFormat::Sparse {
                positions: Arc::new(bitmap.serialize::<croaring::Portable>()),
                values: Arc::new(values_ipc),
                affected_count: merged_positions.len() as u64,
            };

            // Write to a new guard file for this column.
            // Uses the same path convention as `guard_path` so the guards map lookup in
            // `find_guard`/`find_or_create_guard` finds the merged file directly.
            // Old patch files are cleaned up after successful merge.
            let merged_guard_key = GuardKey {
                seg_id: plan.guard_key.seg_id.clone(),
                col: col_plan.col.clone(),
                start_row: plan.guard_key.start_row,
                end_row: plan.guard_key.end_row,
            };
            let merged_guard_path = self.guard_path(&merged_guard_key);
            let tmp_path = {
                let mut p = merged_guard_path.clone();
                p.set_extension("delta.tmp");
                p
            };

            if let Some(parent) = tmp_path.parent() {
                fs::create_dir_all(parent)?;
            }

            let mut guard_keys_set = std::collections::HashSet::new();
            let mut file_paths_set = std::collections::HashSet::new();
            for patch in &plan.patches {
                guard_keys_set.insert(GuardKey {
                    seg_id: patch.seg_id.clone(),
                    col: patch.col.clone(),
                    start_row: patch.dedup_key.0,
                    end_row: patch.dedup_key.0 + 1,
                });
                file_paths_set.insert(patch.file_path.clone());

                // Also clean up the zone-map sidecar file.
                let mut zm_path = patch.file_path.clone();
                zm_path.set_extension("delta.zm");
                file_paths_set.insert(zm_path);
            }
            let old_guard_keys: Vec<_> = guard_keys_set.into_iter().collect();
            let old_file_paths: Vec<_> = file_paths_set.into_iter().collect();

            // RAII guard: marks this merged guard as in-progress and cleans up tmp file
            // on any early return (success, error, or panic). The guard tracks its own
            // (guard_key, tmp_path) so cleanup on failure is always correct.
            let _ig = InProgressGuard::new(
                self,
                merged_guard_key.clone(),
                tmp_path.clone(),
                old_guard_keys.clone(),
                old_file_paths.clone(),
            );

            // Serialize the compacted patch (match the binary format used by append_patch)
            let payload = compacted.to_bytes();
            let merged_patch_id = plan.patches.iter().map(|p| p.patch_id).max().unwrap_or(0);
            let mut patch_bytes = Vec::with_capacity(32 + payload.len());
            patch_bytes.extend_from_slice(&merged_patch_id.to_le_bytes());
            patch_bytes.extend_from_slice(&plan.output_txn_range.0.to_le_bytes());
            patch_bytes.extend_from_slice(&plan.output_txn_range.1.to_le_bytes());
            patch_bytes.push(0x00u8); // sparse format
            patch_bytes.extend_from_slice(&(merged_positions.len() as u32).to_le_bytes());
            patch_bytes.extend_from_slice(&(payload.len() as u32).to_le_bytes());
            patch_bytes.extend_from_slice(&payload);

            let mut f = OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(&tmp_path)?;
            f.write_all(&patch_bytes)?;
            f.sync_all()?;

            // Atomic rename: write-to-tmp is done above; now rename to final path.
            fs::rename(&tmp_path, &merged_guard_path)?;

            // fsync parent directory to make the rename durable on Windows.
            // Without this, a crash after rename but before directory fsync can lose
            // the file from directory listings.
            if let Some(parent) = merged_guard_path.parent() {
                if let Ok(_dir_fd) = std::fs::OpenOptions::new().read(true).open(parent) {
                    #[cfg(unix)]
                    {
                        use std::os::fd::AsRawFd;
                        let _ = std::fs::File::from(dir_fd).sync_all();
                    }
                    #[cfg(windows)]
                    {
                        use std::os::windows::io::FromRawHandle;
                        use windows::core::PCWSTR;
                        use windows::Win32::Storage::FileSystem::{
                            CreateFileW, FILE_FLAG_BACKUP_SEMANTICS, FILE_SHARE_MODE,
                            FILE_SHARE_READ, FILE_SHARE_WRITE, OPEN_EXISTING,
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
                            let file = unsafe { File::from_raw_handle(h.0 as *mut _) };
                            let _ = file.sync_all(); // non-fatal
                        }
                    }
                }
            }

            // Success: explicitly unregister from in_progress_merges. The tmp file has already
            // been renamed to the final merged path, so InProgressGuard::drop won't find it.
            _ig.unregister();

            // Register the merged guard in the guards map (overwrites old entry if present)
            let merged_guard = Guard::new(merged_guard_key.clone(), merged_guard_path.clone());
            self.guards
                .write()
                .insert(merged_guard_key.clone(), merged_guard);

            // Update row_to_guard auxiliary index for the merged guard's range
            {
                let mut idx = self.row_to_guard.write();
                for r in merged_guard_key.start_row..merged_guard_key.end_row {
                    idx.insert(
                        (
                            merged_guard_key.seg_id.clone(),
                            merged_guard_key.col.clone(),
                            r,
                        ),
                        merged_guard_key.clone(),
                    );
                }
            }

            // d017: register merged guard in secondary index
            {
                let mut sidx = self.seg_col_index.write();
                sidx
                    .entry((
                        merged_guard_key.seg_id.clone(),
                        merged_guard_key.col.clone(),
                    ))
                    .or_default()
                    .push(merged_guard_key.clone());
            }

            tracing::debug!(
                "Guard merge: wrote {} rows for column '{}' to {:?}",
                merged_positions.len(),
                col_plan.col,
                &merged_guard_path
            );
        }

        // Old patch files are left on disk. Future: schedule them for async cleanup
        // once the merged guard is fully registered and readers will use the new file.
        tracing::debug!(
            "Guard merge: {} old patch files left on disk for future cleanup",
            plan.patches.len()
        );

        // Cleanup old patch files now that the merged guard is registered (G8 fix).
        // Collect unique guard keys and file paths, then delete them.
        //
        // We deliberately do NOT look up paths from the guards map here, because
        // the merged guard was already inserted into the map at the same key,
        // so the old-path lookup would return the new .delta path and we
        // would accidentally delete the freshly-written merged file.
        //
        // Instead we collect the original patch file paths directly from the plan,
        // deduplicate them, and also include the corresponding .delta.zm zone-map
        // files.
        use std::collections::BTreeSet;

        let mut guard_keys_set: BTreeSet<GuardKey> = BTreeSet::new();
        let mut file_paths_set: BTreeSet<std::path::PathBuf> = BTreeSet::new();

        for patch in &plan.patches {
            guard_keys_set.insert(GuardKey {
                seg_id: patch.seg_id.clone(),
                col: patch.col.clone(),
                start_row: patch.dedup_key.0,
                end_row: patch.dedup_key.0 + 1,
            });
            file_paths_set.insert(patch.file_path.clone());

            // Also clean up the zone-map sidecar file.
            let mut zm_path = patch.file_path.clone();
            zm_path.set_extension("delta.zm");
            file_paths_set.insert(zm_path);
        }

        if !guard_keys_set.is_empty() {
            let num_guard_keys = guard_keys_set.len();
            let num_file_paths = file_paths_set.len();
            self.cleanup_guard_patches(
                &guard_keys_set.into_iter().collect::<Vec<_>>(),
                &file_paths_set.into_iter().collect::<Vec<_>>(),
            );
            tracing::debug!(
                "Guard merge cleanup: removed {} old patch files ({} zone maps)",
                num_guard_keys,
                num_file_paths - num_guard_keys
            );
        }

        tracing::info!("Merge plan executed for guard {:?}", plan.guard_key);
        Ok(())
    }

    /// Read a single value from a patch at a specific row offset.
    /// D5 optimization: Uses a cache to avoid repeated decoding of the same patch.
    fn read_patch_value_cached(
        &self,
        patch: &MergePatchRef,
        row: u64,
        cache: &mut std::collections::HashMap<
            Arc<Vec<u8>>,
            std::sync::Arc<dyn arrow_array::Array>,
            std::collections::hash_map::RandomState,
        >,
    ) -> Result<Option<Vec<u8>>> {
        use super::sparsity::decode_arrow_array;
        use arrow_array::Array;

        match &patch.format {
            DeltaPatchFormat::Sparse {
                positions,
                values,
                affected_count: _,
            } => {
                let bitmap = croaring::Bitmap::deserialize::<croaring::Portable>(positions);
                if !bitmap.contains(row as u32) {
                    return Ok(None);
                }
                // Find the index of `row` in the bitmap to get value index
                let all_positions: Vec<u32> = bitmap.iter().collect();
                let value_idx = all_positions.iter().position(|&r| r == row as u32);

                if let Some(idx) = value_idx {
                    // D5 optimization: use cache to avoid repeated decoding
                    let arr = match cache.get(values) {
                        Some(a) => a.clone(),
                        None => {
                            let decoded = decode_arrow_array(values)?;
                            let arr: std::sync::Arc<dyn arrow_array::Array> = decoded;
                            cache.entry(values.clone()).or_insert(arr.clone());
                            arr
                        }
                    };
                    if idx < arr.len() {
                        let val = arr
                            .as_any()
                            .downcast_ref::<arrow_array::BinaryArray>()
                            .map(|a| a.value(idx).to_vec());
                        return Ok(val);
                    }
                }
                Ok(None)
            }
            DeltaPatchFormat::Dense {
                values, total_rows, ..
            } => {
                if row >= *total_rows {
                    return Ok(None);
                }
                // D5 optimization: use cache to avoid repeated decoding
                let arr = match cache.get(values) {
                    Some(a) => a.clone(),
                    None => {
                        let decoded = decode_arrow_array(values)?;
                        let arr: std::sync::Arc<dyn arrow_array::Array> = decoded;
                        cache.entry(values.clone()).or_insert(arr.clone());
                        arr
                    }
                };
                if (row as usize) < arr.len() {
                    let val = arr
                        .as_any()
                        .downcast_ref::<arrow_array::BinaryArray>()
                        .map(|a| a.value(row as usize).to_vec());
                    return Ok(val);
                }
                Ok(None)
            }
        }
    }

    /// Read a single value from a patch at a specific row offset (non-cached version).
    #[allow(dead_code)]
    fn read_patch_value(&self, patch: &MergePatchRef, row: u64) -> Result<Option<Vec<u8>>> {
        use super::sparsity::decode_arrow_array;
        use arrow_array::Array;

        match &patch.format {
            DeltaPatchFormat::Sparse {
                positions,
                values,
                affected_count: _,
            } => {
                let bitmap = croaring::Bitmap::deserialize::<croaring::Portable>(positions);
                if !bitmap.contains(row as u32) {
                    return Ok(None);
                }
                // Find the index of `row` in the bitmap to get value index
                let all_positions: Vec<u32> = bitmap.iter().collect();
                let value_idx = all_positions.iter().position(|&r| r == row as u32);

                if let Some(idx) = value_idx {
                    let arr = decode_arrow_array(values)?;
                    if idx < arr.len() {
                        let val = arr
                            .as_any()
                            .downcast_ref::<arrow_array::BinaryArray>()
                            .map(|a| a.value(idx).to_vec());
                        return Ok(val);
                    }
                }
                Ok(None)
            }
            DeltaPatchFormat::Dense {
                values, total_rows, ..
            } => {
                if row >= *total_rows {
                    return Ok(None);
                }
                let arr = decode_arrow_array(values)?;
                if (row as usize) < arr.len() {
                    let val = arr
                        .as_any()
                        .downcast_ref::<arrow_array::BinaryArray>()
                        .map(|a| a.value(row as usize).to_vec());
                    return Ok(val);
                }
                Ok(None)
            }
        }
    }

    /// Encode a list of binary values as Arrow IPC format.
    fn encode_values_ipc(&self, values: &[Vec<u8>]) -> Result<Vec<u8>> {
        use arrow_array::{BinaryArray, RecordBatch};
        use arrow_schema::{Field, Schema};

        let schema = Schema::new(vec![Field::new("v", arrow_schema::DataType::Binary, true)]);
        let batch = RecordBatch::try_new(
            std::sync::Arc::new(schema),
            vec![Arc::new(BinaryArray::from_iter_values(
                values.iter().map(|v| v.as_slice()),
            ))],
        )
        .map_err(|e| RockDuckError::Internal(format!("Arrow batch build: {}", e)))?;

        let mut buf = Vec::new();
        {
            let mut writer = arrow_ipc::writer::FileWriter::try_new(&mut buf, &batch.schema())
                .map_err(|e| RockDuckError::Internal(format!("IPC writer: {}", e)))?;
            writer
                .write(&batch)
                .map_err(|e| RockDuckError::Internal(format!("IPC write: {}", e)))?;
            writer
                .finish()
                .map_err(|e| RockDuckError::Internal(format!("IPC finish: {}", e)))?;
        }
        Ok(buf)
    }

    /// Schedule a guard merge (async or sync).
    pub fn schedule_guard_merge(&self, guard_key: &GuardKey) -> Result<()> {
        if let Some(plan) = self.generate_merge_plan(guard_key)? {
            self.execute_merge_plan(plan)?;
        }
        Ok(())
    }

    /// Cleanup an in-progress merge that failed. Removes tmp files and unregisters the key.
    /// Safe to call multiple times (idempotent).
    #[allow(dead_code)]
    fn cleanup_in_progress_merge(
        &self,
        guard_key: &GuardKey,
        tmp_path: &PathBuf,
        old_guard_keys: &[GuardKey],
        old_file_paths: &[PathBuf],
    ) {
        // Remove tmp file if it exists.
        if tmp_path.exists() {
            if let Err(e) = fs::remove_file(tmp_path) {
                tracing::warn!(
                    "cleanup_in_progress_merge: failed to remove tmp {:?}: {}",
                    tmp_path,
                    e
                );
            }
        }
        self.cleanup_guard_patches(old_guard_keys, old_file_paths);
        // Unregister from in_progress_merges.
        self.in_progress_merges.write().remove(guard_key);
        tracing::debug!("cleanup_in_progress_merge: unregistered {:?}", guard_key);
    }

    /// Recovery: clean up orphan .tmp and .delta.merged files on startup.
    pub fn recover(&self) -> Result<()> {
        if !self.root_dir.exists() {
            return Ok(());
        }
        // Use simple recursive walk to clean up .tmp and .delta.merged files
        fn walk_dir(dir: &PathBuf) -> std::io::Result<Vec<std::path::PathBuf>> {
            let mut results = Vec::new();
            for entry in fs::read_dir(dir)? {
                let entry = entry?;
                let path = entry.path();
                if path.is_dir() {
                    results.extend(walk_dir(&path)?);
                } else {
                    results.push(path);
                }
            }
            Ok(results)
        }
        for path in walk_dir(&self.root_dir)? {
            if path.extension().and_then(|s| s.to_str()) == Some("tmp") {
                tracing::info!("Cleaning up orphan delta tmp file: {:?}", path);
                fs::remove_file(path)?;
            } else if path
                .file_name()
                .and_then(|s| s.to_str())
                .is_some_and(|s| s.ends_with(".delta.merged"))
            {
                tracing::info!("Cleaning up orphan delta merged file: {:?}", path);
                fs::remove_file(path)?;
            }
        }
        Ok(())
    }

    /// Get the number of guards.
    pub fn num_guards(&self) -> usize {
        self.guards.read().len()
    }

    /// Get a read guard over the guards map for test inspection.
    pub fn guards_map(&self) -> &parking_lot::RwLock<std::collections::BTreeMap<GuardKey, Guard>> {
        &self.guards
    }

    /// Get a read guard over the row_to_guard map for test inspection.
    pub fn row_to_guard_map(
        &self,
    ) -> &parking_lot::RwLock<
        rustc_hash::FxHashMap<(std::string::String, std::string::String, u64), GuardKey>,
    > {
        &self.row_to_guard
    }

    /// Get the number of L2 entries.
    pub fn num_entries(&self) -> usize {
        self.guards.read().len()
    }

    /// Get the compaction threshold.
    pub fn compaction_threshold(&self) -> u32 {
        self.compaction_threshold
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Returns an iterator over all guards with their zone maps, filtered by seg_id.
    pub fn guards_by_seg(&self, seg_id: &str) -> Vec<(GuardKey, u64)> {
        let guards = self.guards.read();
        guards
            .iter()
            .filter(|(gk, _)| gk.seg_id == seg_id)
            .map(|(gk, guard)| {
                let zm = guard.zone_map.read();
                (gk.clone(), zm.min_txn)
            })
            .collect()
    }

    /// Find a guard that has exceeded the compaction threshold.
    /// Returns the guard key if found.
    pub fn find_overloaded_guard(&self) -> Option<GuardKey> {
        let guards = self.guards.read();
        for (gk, guard) in guards.iter() {
            if guard.patch_count.load(AtomicOrdering::Relaxed)
                >= self
                    .compaction_threshold
                    .load(std::sync::atomic::Ordering::Relaxed)
            {
                return Some(gk.clone());
            }
        }
        None
    }

    /// Read all patches from a guard by its GuardKey.
    pub fn read_all_patches_from_guard(&self, guard_key: &GuardKey) -> Result<Vec<MergePatchRef>> {
        let file_path = {
            let guards = self.guards.read();
            match guards.get(guard_key) {
                Some(g) => g.file_path.clone(),
                None => return Ok(Vec::new()),
            }
        };
        let guard = Guard::new(guard_key.clone(), file_path);
        self.read_all_patches(&guard)
    }

    /// Delete old patch files from the guards that were merged away.
    ///
    /// Called after a successful guard merge to clean up orphaned patch files (G8 fix).
    ///
    /// This function is idempotent and safe to call multiple times.
    ///
    /// - `old_guard_keys`: keys to remove from guards map and row_to_guard index
    /// - `old_file_paths`: explicit file paths to delete (patch files and their zone maps).
    ///   Unlike the previous implementation, we do NOT look up paths from the guards map,
    ///   because the guards map may already contain the new merged guard at the same key,
    ///   causing us to accidentally delete the merged file instead of the old ones.
    pub fn cleanup_guard_patches(
        &self,
        old_guard_keys: &[GuardKey],
        old_file_paths: &[std::path::PathBuf],
    ) {
        // d016: invalidate LRU cache entries for deleted guards
        for guard_key in old_guard_keys {
            self.patch_cache.invalidate(guard_key);
        }

        // Deduplicate file paths so we don't attempt to delete the same file twice.
        let unique_paths: std::collections::HashSet<_> = old_file_paths.iter().collect();

        // Step 1: delete files directly from the explicit paths provided.
        for path in &unique_paths {
            if path.exists() {
                match fs::remove_file(path) {
                    Ok(()) => {
                        tracing::debug!("cleanup_guard_patches: removed file {:?}", path);
                    }
                    Err(e) => {
                        tracing::warn!("cleanup_guard_patches: failed to remove {:?}: {}", path, e);
                    }
                }
            }
        }

        // Step 2: clean up in-memory handles, guard metadata, and auxiliary indexes.
        // We do this under write locks to avoid leaving stale cache entries behind.
        let mut handles_write = self.handles.write();
        let mut guards_write = self.guards.write();
        {
            let mut idx_write = self.row_to_guard.write();
            let mut in_progress_write = self.in_progress_merges.write();

            for guard_key in old_guard_keys {
                handles_write.remove(guard_key);
                guards_write.remove(guard_key);
                in_progress_write.remove(guard_key);

                idx_write.retain(|_key, guard| {
                    !(guard.seg_id == guard_key.seg_id
                        && guard.col == guard_key.col
                        && guard.start_row >= guard_key.start_row
                        && guard.start_row < guard_key.end_row)
                });

                // d017: clean up secondary index entries for this guard
                let index_key = (guard_key.seg_id.clone(), guard_key.col.clone());
                if let Some(keys) = self.seg_col_index.write().get_mut(&index_key) {
                    keys.retain(|k| k != guard_key);
                    if keys.is_empty() {
                        self.seg_col_index.write().remove(&index_key);
                    }
                }
            }
        }
    }
}

    /// RAII guard: marks a guard key as in-progress at creation and unregisters on drop.
    /// On SUCCESS, the caller must call `unregister()` explicitly before the guard drops.
    /// On FAILURE (early return), the guard drops automatically and cleans up the tmp file.
    /// The final merged path is `guard_path`, renamed from tmp on success.
struct InProgressGuard<'a> {
    store: &'a DeltaL2Disk,
    guard_key: GuardKey,
    tmp_path: PathBuf,
    old_guard_keys: Vec<GuardKey>,
    old_file_paths: Vec<PathBuf>,
    unregistered: bool,
}

impl<'a> InProgressGuard<'a> {
    fn new(
        store: &'a DeltaL2Disk,
        guard_key: GuardKey,
        tmp_path: PathBuf,
        old_guard_keys: Vec<GuardKey>,
        old_file_paths: Vec<PathBuf>,
    ) -> Self {
        store.in_progress_merges.write().insert(guard_key.clone());
        Self {
            store,
            guard_key,
            tmp_path,
            old_guard_keys,
            old_file_paths,
            unregistered: false,
        }
    }
    /// Call on success path: unregister from in_progress_merges but do NOT delete tmp
    /// (it has already been renamed to the final merged path).
    fn unregister(mut self) {
        self.store
            .in_progress_merges
            .write()
            .remove(&self.guard_key);
        self.unregistered = true;
    }
}

impl<'a> Drop for InProgressGuard<'a> {
    fn drop(&mut self) {
        if self.unregistered {
            return;
        }
        // Failure path: tmp file still exists, remove it.
        if self.tmp_path.exists() {
            if let Err(e) = fs::remove_file(&self.tmp_path) {
                tracing::warn!(
                    "InProgressGuard: failed to remove tmp {:?}: {}",
                    self.tmp_path,
                    e
                );
            }
        }
        self.store
            .cleanup_guard_patches(&self.old_guard_keys, &self.old_file_paths);
        self.store
            .in_progress_merges
            .write()
            .remove(&self.guard_key);
        tracing::debug!("InProgressGuard: cleaned up {:?}", self.guard_key);
    }
}

impl DeltaL2Disk {
    /// Delete the guard's patch file from disk and remove it from the guard index.
    pub fn delete_guard_file(&self, guard_key: &GuardKey) -> Result<()> {
        let path = {
            let guards = self.guards.read();
            match guards.get(guard_key) {
                Some(g) => g.file_path.clone(),
                None => return Ok(()),
            }
        };
        if path.exists() {
            fs::remove_file(&path)?;
        }
        self.guards.write().remove(guard_key);
        Ok(())
    }
}

// =============================================================================
// Patch → DeltaCell conversion
// =============================================================================

/// Extract row positions from a DeltaPatchFormat.
/// Returns the set of row offsets affected by this patch.
/// Used for per-row deduplication in merge planning.
pub fn patch_row_positions(format: &DeltaPatchFormat) -> Vec<u64> {
    match format {
        DeltaPatchFormat::Sparse { positions, .. } => {
            let bitmap = croaring::Bitmap::deserialize::<croaring::Portable>(positions);
            bitmap.iter().map(|p| p as u64).collect()
        }
        DeltaPatchFormat::Dense {
            values: _,
            total_rows,
            ..
        } => (0..*total_rows).collect(),
    }
}

/// Convert a DeltaPatchFormat to individual DeltaCells.
/// Used by L2 and L3 query paths.
pub fn patch_to_deltas(format: &DeltaPatchFormat) -> Result<Vec<DeltaCell>> {
    match format {
        DeltaPatchFormat::Sparse {
            positions,
            values,
            affected_count: _,
        } => {
            let bitmap = croaring::Bitmap::deserialize::<croaring::Portable>(positions);
            let positions: Vec<u64> = bitmap.iter().map(|p| p as u64).collect();
            let arr = super::sparsity::decode_arrow_array(values)?;
            let arr_ref = arr.as_ref();
            let binary = arr_ref
                .as_any()
                .downcast_ref::<arrow_array::BinaryArray>()
                .ok_or_else(|| {
                    RockDuckError::Delta("sparse patch values are not BinaryArray".into())
                })?;

            Ok(positions
                .iter()
                .enumerate()
                .map(|(i, &row)| {
                    let after = if arrow_array::Array::is_null(&*binary, i) {
                        None
                    } else {
                        Some(Vec::from(binary.value(i)))
                    };
                    DeltaCell {
                        seg_id: String::new(),
                        row_offset: row,
                        column: String::new(),
                        txn_id: 0,
                        before: None,
                        after: after.map(Arc::new),
                        committed: true,
                        ts: 0,
                    }
                })
                .collect())
        }
        DeltaPatchFormat::Dense {
            values,
            total_rows,
            txn_id,
        } => {
            let arr = super::sparsity::decode_arrow_array(values)?;
            let arr_len = arr.len() as u64;
            // Use the larger of the two: total_rows is the segment size, arr_len is
            // the number of encoded values. The segment may be larger than the patch.
            let seg_total = (*total_rows).max(arr_len);
            let mut cells = Vec::with_capacity(seg_total as usize);
            for row_offset in 0..seg_total {
                let after_bytes = super::sparsity::extract_value_at(values, row_offset as usize)?;
                cells.push(DeltaCell {
                    seg_id: String::new(),
                    row_offset,
                    column: String::new(),
                    txn_id: *txn_id,
                    before: None,
                    after: after_bytes.map(Arc::new),
                    committed: true,
                    ts: 0,
                });
            }
            Ok(cells)
        }
    }
}

// =============================================================================
// Helpers
// =============================================================================

#[allow(dead_code)]
fn read_bytes<R: Read>(reader: &mut R, n: usize) -> Result<Vec<u8>> {
    let mut buf = vec![0u8; n];
    reader.read_exact(&mut buf).map_err(RockDuckError::Io)?;
    Ok(buf)
}
