//! DuckDB VTab extension for RockDuck -- streams Vortex data to DuckDB.
//!
//! Provides `docdb_scan()` as a DuckDB table function, enabling SQL queries
//! over RockDuck's Vortex-backed storage with full MVCC visibility, filter pushdown,
//! and projection pushdown support.
//!
//! Uses DuckDB's native `VTab` trait (`duckdb::vtab::VTab`) which gives us
//! `duckdb::core::DataChunkHandle` in the `func()` callback, enabling direct use of
//! `record_batch_to_duckdb_data_chunk` for Arrow RecordBatch -> DuckDB conversion.
//!
//! ## Thread-safety design
//! All mutable state lives in `BindData` behind `Sync` primitives so that `func()`
//! can work with just a `&BindData` shared reference (no UB from casting to `&mut`).

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};

use arrow_array::{Array, ArrayRef, RecordBatch};
use arrow_schema::{Schema as ArrowSchema, SchemaRef as ArrowSchemaRef};
use duckdb::core::{DataChunkHandle, LogicalTypeHandle, LogicalTypeId};
use duckdb::vtab::arrow::{record_batch_to_duckdb_data_chunk, to_duckdb_logical_type};
use duckdb::vtab::{BindInfo, InitInfo, TableFunctionInfo, VTab};
use tracing::{debug, info, warn};

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct VtabReaderHotspotStats {
    pub lazy_init_calls: u64,
    pub column_reader_open_attempts: u64,
    pub column_reader_open_successes: u64,
    pub vis_reader_open_attempts: u64,
    pub vis_reader_open_successes: u64,
}

use crate::db::RockDuck;
use crate::metadata::projection::ProjectionContract;
use crate::metadata::{get_segment_meta, CF_MVCC};
use crate::mvcc::visibility::TxnSnapshot;
use crate::query::filter_expr::{self, ScanFilter};
use crate::query::routing::SidecarEvidenceSnapshot;
use crate::segment::meta::{SegmentMeta, SegmentStatus};
use crate::storage::vortex::VortexReader;

/// store the RockDuck instance in BindData so that
/// `lazy_init_readers` and `load_segment_meta` can reuse it instead of
/// re-opening a new RockDuck for every segment (which was O(segments) disk I/O).
/// The configured data root for vTab path validation.
/// Set once at registration time; all subsequent docdb_scan calls must resolve
/// within this root.
static VTAB_DATA_ROOT: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();

// Per-thread RockDuck instance for VTab.
// Uses `thread_local` instead of `OnceLock` to provide per-thread isolation.
// When a thread registers a different database via `set_rockduck_for_vtab`, only
// that thread's queries are affected. Other threads retain their own RockDuck
// reference. This prevents cross-database interference when multiple databases are
// queried concurrently (e.g., from different DuckDB connections).
// The `RefCell<Option<Arc<RockDuck>>` pattern allows both set and get per-thread.
thread_local! {
    static VTAB_ROCKDUCK: std::cell::RefCell<Option<Arc<RockDuck>>> =
        const { std::cell::RefCell::new(None) };
}

/// Configure the data root for docdb_scan path validation.
/// Must be called before any DuckDB query using docdb_scan.
pub fn set_vtab_data_root(path: PathBuf) -> std::result::Result<(), String> {
    VTAB_DATA_ROOT
        .set(path.clone())
        .map_err(|_| format!("vTab data root already configured: {}", path.display()))?;
    info!(root = %path.display(), "vTab data root configured");
    Ok(())
}

/// Store the RockDuck instance for the current thread's VTab to reuse.
/// Returns `true` always (TLS guarantees no cross-thread interference).
pub fn set_rockduck_for_vtab(db: Arc<RockDuck>) -> bool {
    VTAB_ROCKDUCK.with(|cell| {
        cell.borrow_mut().replace(db);
        true
    })
}

/// Get the current thread's RockDuck instance for the VTab.
pub fn get_rockduck_for_vtab() -> Option<Arc<RockDuck>> {
    VTAB_ROCKDUCK.with(|cell| cell.borrow().clone())
}

/// Clear the current thread's RockDuck instance from TLS.
///
/// Called by `VtabScope::drop` to prevent cross-connection contamination when
/// the same thread creates multiple DuckDB connections sequentially.
pub fn clear_rockduck_for_vtab() {
    VTAB_ROCKDUCK.with(|cell| cell.borrow_mut().take());
}

// =============================================================================
// VTab Types
// =============================================================================

/// BindData -- shared across all func() calls via `get_bind_data()`.
pub struct BindData {
    /// the RockDuck instance, opened once at bind time and reused
    /// by all lazy_init_readers calls instead of re-opening per segment.
    pub rockduck: Arc<RockDuck>,
    /// Canonical, validated table root path
    pub table_root: PathBuf,
    /// Table ID being scanned through the sidecar.
    pub table_id: String,
    /// Segment IDs for this table (from KV)
    pub segment_ids: Vec<String>,
    /// Column names (from first segment's schema)
    pub column_names: Vec<String>,
    /// MVCC committed txn ID at bind time
    pub committed_txn: u64,
    /// Optional parsed filter expression
    pub filter: Option<ScanFilter>,
    /// Estimated total rows for cardinality hint
    pub estimated_rows: u64,
    /// Time travel: read data as of this transaction ID.
    /// When set, uses TimeTravelScanner instead of normal Vortex reads.
    pub as_of_txn: Option<u64>,

    /// --- Mutable scan state (Sync, so &BindData works in func) ---
    /// Index of the currently active segment
    current_seg_idx: AtomicUsize,
    /// Lazily-initialized VortexReaders per segment
    readers: RwLock<HashMap<String, Vec<VortexReader>>>,
    /// Per-segment chunk index (position within the current reader's batches)
    chunk_idx_in_seg: RwLock<HashMap<String, usize>>,
    /// Flag: have readers been lazily initialized?
    readers_initialized: AtomicUsize,
    hotspot_stats: RwLock<VtabReaderHotspotStats>,
    /// per-segment nullable flags, parallel to column_names indexing.
    /// seg_nullable[seg_id][col_idx] == true if the column is nullable.
    seg_nullable: RwLock<HashMap<String, Vec<bool>>>,
    /// per-segment visibility readers for __vis.vortex.
    /// Maps seg_id -> VortexReader that can read both created_txn and deleted_txn columns.
    /// Populated lazily during reader initialization.
    vis_readers: RwLock<HashMap<String, VortexReader>>,
    /// Time travel batches (pre-loaded when as_of_txn is set)
    tt_batches: RwLock<Vec<RecordBatch>>,
    /// Current index into tt_batches
    tt_batch_idx: AtomicUsize,

    /// eagerly-captured MVCC snapshot for visibility filtering.
    /// Captured at bind time (equivalent to BatchScanIterator's eager snapshot).
    /// Contains committed_txn and active_txns for visibility checks.
    snapshot: TxnSnapshot,
    /// Explicit sidecar projection contract for this VTab bind.
    projection_contract: ProjectionContract,
}

impl BindData {
    pub fn hotspot_stats(&self) -> VtabReaderHotspotStats {
        self.hotspot_stats
            .read()
            .expect("BindData hotspot_stats lock poisoned")
            .clone()
    }

    /// Advance `current_seg_idx` to the next segment that has unread batches.
    fn advance_to_next_segment(&self) {
        let seg_ids = &self.segment_ids;
        let current = self.current_seg_idx.load(Ordering::Relaxed);

        for seg_id in seg_ids.iter().skip(current) {
            let readers = self.readers.read().expect("BindData readers lock poisoned");
            let chunk_idx_map = self
                .chunk_idx_in_seg
                .read()
                .expect("BindData chunk_idx lock poisoned");

            let num_batches = readers
                .get(seg_id)
                .and_then(|r| r.first()) // Option<&VortexReader> -> r is &VortexReader
                .map(|r| r.num_batches())
                .unwrap_or(0);

            let chunk_idx = *chunk_idx_map.get(seg_id).unwrap_or(&0);

            if num_batches > 0 && chunk_idx < num_batches {
                return;
            }
            self.current_seg_idx.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Load the visibility batch (__vis.vortex) for the given segment and chunk index.
    /// Returns Ok(None) only when no vis reader is available for the segment.
    fn load_vis_batch(
        &self,
        seg_id: &str,
        chunk_idx: usize,
    ) -> std::result::Result<Option<RecordBatch>, String> {
        let vis_readers = self
            .vis_readers
            .read()
            .expect("BindData vis_readers lock poisoned");
        let Some(vis_reader) = vis_readers.get(seg_id) else {
            return Ok(None);
        };

        let mut batches = vis_reader.read_chunks_lazy();
        for idx in 0..=chunk_idx {
            match batches.next() {
                Some(Ok(batch)) if idx == chunk_idx => return Ok(Some(batch)),
                Some(Ok(_)) => continue,
                Some(Err(e)) => {
                    return Err(format!(
                        "read visibility batch {} for segment {}: {}",
                        idx, seg_id, e
                    ));
                }
                None => return Ok(None),
            }
        }
        Ok(None)
    }

    /// Filter a RecordBatch by MVCC visibility using the snapshot's committed_txn.
    ///
    /// ## Sanctioned Surface
    ///
    /// This is a **sanctioned** visibility surface. It delegates to
    /// `TxnSnapshot::is_row_visible` via the `VisFilter` trait, which applies
    /// Rule 1-4 of strict Snapshot Isolation — **equivalent** to
    /// `VisibilityManager::is_row_visible`.
    ///
    /// This was previously an independent inline implementation with a behavioral gap
    /// (absent from `commit_ts_map` was treated as visible — D12). Delegating to
    /// `TxnSnapshot::is_row_visible` closes that gap and ensures VTab uses the same
    /// visibility semantics as the scan and point_get paths.
    ///
    /// ## mv011 Batch Optimization
    ///
    /// Uses a HashSet-backed approach for active_txns to reduce per-row lookup overhead.
    /// For large batches, the active_txns HashSet provides O(1) membership checks
    /// (vs O(log n) for BTreeSet). Commit_ts_map lookups are also batched via local
    /// variable access to avoid repeated borrow checker overhead.
    fn filter_by_visibility(
        batch: &RecordBatch,
        vis_batch: &RecordBatch,
        snapshot: &TxnSnapshot,
    ) -> std::result::Result<RecordBatch, String> {
        use arrow_array::Int64Array;
        use arrow_select::filter::filter_record_batch;

        let created_col = vis_batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or("vis_batch col 0 is not Int64Array")?;
        let deleted_col = vis_batch
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or("vis_batch col 1 is not Int64Array")?;

        // mv011 fix: Pre-extract snapshot data into local variables to avoid
        // repeated borrow overhead in the per-row loop.
        // active_txns is cloned to HashSet for O(1) contains lookups (vs O(log n) for BTreeSet).
        let snapshot_id = snapshot.snapshot_id;
        let active_txns: std::collections::HashSet<u64> = snapshot
            .active_txns
            .iter()
            .copied()
            .collect();
        let commit_ts_map = &snapshot.commit_ts_map;

        let num_rows = batch.num_rows();
        let mut mask = Vec::with_capacity(num_rows);

        for i in 0..num_rows {
            let created_txn = created_col.value(i) as u64;
            let deleted_txn_val = deleted_col.value(i);
            let deleted_txn = if deleted_txn_val == crate::mvcc::shadow_columns::NOT_DELETED as i64 {
                None
            } else {
                Some(deleted_txn_val as u64)
            };

            // mv011 batch optimization: use pre-collected HashSet for O(1) active_txns lookup
            // Rule 1: not a future transaction
            if created_txn > snapshot_id {
                mask.push(false);
                continue;
            }
            // Rule 2: not created by a still-active transaction (HashSet O(1) lookup)
            if active_txns.contains(&created_txn) {
                mask.push(false);
                continue;
            }
            // Rule 3: D12 strict Snapshot Isolation — if created_txn is absent from
            // commit_ts_map, it means the transaction was aborted. Treat as invisible.
            let Some(&commit_ts) = commit_ts_map.get(&created_txn) else {
                mask.push(false);
                continue;
            };
            // Rule 3 continued: committed txns must have commit_ts <= snapshot_id
            if commit_ts > snapshot_id {
                mask.push(false);
                continue;
            }
            // Rule 4: deletion visibility
            if let Some(del) = deleted_txn {
                if active_txns.contains(&del) {
                    // Delete txn is still active → treat as not deleted
                } else if let Some(&del_commit_ts) = commit_ts_map.get(&del) {
                    if del_commit_ts <= snapshot_id {
                        mask.push(false);
                        continue;
                    }
                }
            }
            mask.push(true);
        }

        let filter_array = arrow_array::BooleanArray::from(mask);
        filter_record_batch(batch, &filter_array).map_err(|e| format!("visibility filter: {}", e))
    }

    /// Consume the next `RecordBatch` for DuckDB, or None if all data is exhausted.
    fn next_batch(&self) -> Option<RecordBatch> {
        loop {
            let seg_ids = &self.segment_ids;
            let current = self.current_seg_idx.load(Ordering::Relaxed);

            if current >= seg_ids.len() {
                return None;
            }

            let seg_id = &seg_ids[current];

            let num_total_batches = {
                let readers = self.readers.read().expect("BindData readers lock poisoned");
                readers
                    .get(seg_id)
                    .and_then(|r| r.first())
                    .map(|r| r.num_batches())
                    .unwrap_or(0)
            };

            let chunk_idx = {
                let map = self
                    .chunk_idx_in_seg
                    .read()
                    .expect("BindData chunk_idx lock poisoned");
                *map.get(seg_id).unwrap_or(&0)
            };

            if chunk_idx >= num_total_batches {
                self.advance_to_next_segment();
                continue;
            }

            // Build a RecordBatch from all column readers for this chunk.
            // take the minimum batch count across all column readers
            // and only access up to min_batches to avoid index out of bounds.
            // NOTE: clone reader refs under read lock, release the lock, then call
            // read_all_batches() (requires &mut self) on the clones.
            let reader_clones: Vec<VortexReader> = {
                let readers = self.readers.read().expect("BindData readers lock poisoned");
                readers.get(seg_id).cloned().unwrap_or_default()
            };

            // Build a RecordBatch from all column readers for this chunk.
            // take the minimum batch count across all column readers
            // and only access up to min_batches to avoid index out of bounds.
            //
            // Optimization: one pass loads all batches and extracts the needed column
            // immediately. VortexReader clones share the internal Arc<RwLock<Option<Arc<Vec<RecordBatch>>>>>,
            // so once the first clone calls read_all_batches(), subsequent clones
            // of the same reader see the cached Arc and don't re-read from disk.
            let batch_cols: Vec<ArrayRef> = {
                let mut cols = Vec::with_capacity(reader_clones.len());
                for reader in reader_clones.into_iter() {
                    // read_all_batches() is O(1) if this reader's Arc is already cached;
                    // otherwise triggers one mmap+parse and caches the result.
                    let batches = reader.read_all_batches();
                    if chunk_idx < batches.len() {
                        cols.push(batches[chunk_idx].column(0).clone());
                    }
                    // reader is dropped; its Arc share is released
                }
                cols
            };

            // Advance chunk index
            {
                let mut map = self
                    .chunk_idx_in_seg
                    .write()
                    .expect("BindData chunk_idx lock poisoned");
                *map.entry(seg_id.clone()).or_insert(0) += 1;
            }

            if batch_cols.is_empty() {
                continue;
            }

            // look up nullable flags for this segment from seg_nullable map.
            let nullable_flags = {
                let map = self
                    .seg_nullable
                    .read()
                    .expect("BindData seg_nullable lock poisoned");
                map.get(seg_id).cloned().unwrap_or_default()
            };

            let batch_fields: Vec<arrow_schema::Field> = batch_cols
                .iter()
                .enumerate()
                .map(|(i, arr)| {
                    let nullable = nullable_flags.get(i).copied().unwrap_or(true);
                    arrow_schema::Field::new(
                        self.column_names.get(i).map(|s| s.as_str()).unwrap_or("?"),
                        arr.data_type().clone(),
                        nullable,
                    )
                })
                .collect();

            let schema = ArrowSchemaRef::new(ArrowSchema::new(batch_fields));
            let Some(batch) = RecordBatch::try_new(schema, batch_cols).ok() else {
                continue;
            };

            if batch.num_rows() == 0 {
                continue;
            }

            let filtered = if let Some(ref f) = self.filter {
                match apply_scan_filter(&batch, f) {
                    Ok(b) => b,
                    Err(e) => {
                        // Classify error severity:
                        // - "schema mismatch" / "type error": internal bug, should not happen
                        // - "overflow" / "division by zero": data corruption or malformed data
                        // - "file not found" / "io error": transient infrastructure issue
                        let err_str = e.to_lowercase();
                        let is_fatal = err_str.contains("schema mismatch")
                            || err_str.contains("type error")
                            || err_str.contains("internal");
                        if is_fatal {
                            tracing::error!("Fatal filter error (aborting scan): {}", e);
                            break None;
                        } else {
                            warn!("Filter error (dropping batch): {}", e);
                            continue;
                        }
                    }
                }
            } else {
                batch
            };

            // apply MVCC visibility filter.
            // Load created_txn and deleted_txn columns from __vis.vortex for this batch,
            // then filter out rows that are not visible at the snapshot's committed_txn.
            let filtered = match self.load_vis_batch(seg_id, chunk_idx) {
                Ok(Some(vis_batch)) => {
                    match Self::filter_by_visibility(&filtered, &vis_batch, &self.snapshot) {
                        Ok(f) => f,
                        Err(e) => {
                            warn!("Visibility filter error (dropping batch): {}", e);
                            continue;
                        }
                    }
                }
                Ok(None) => {
                    warn!(
                        "Missing visibility reader or batch for segment {} chunk {}; dropping batch to avoid leaking rows",
                        seg_id,
                        chunk_idx
                    );
                    continue;
                }
                Err(e) => {
                    warn!(
                        "Visibility batch read failed for segment {} chunk {}: {}; dropping batch to avoid leaking rows",
                        seg_id,
                        chunk_idx,
                        e
                    );
                    continue;
                }
            };

            if filtered.num_rows() == 0 {
                continue;
            }

            return Some(filtered);
        }
    }
}

/// InitData -- minimal index passed from DuckDB's init phase.
/// Holds which segment batch to resume from (index into BindData::segment_ids).
pub struct InitData {
    /// Index into BindData::segment_ids to start from
    pub seg_start_idx: usize,
}

// =============================================================================
// VTab Implementation
// =============================================================================

pub struct RockDuckVTab;

impl VTab for RockDuckVTab {
    type BindData = BindData;
    type InitData = InitData;

    fn bind(
        bind: &BindInfo,
    ) -> std::result::Result<Self::BindData, Box<dyn std::error::Error + 'static>> {
        let param_count = bind.get_parameter_count();
        if param_count == 0 {
            bind.set_error("docdb_scan requires at least 1 argument: table_root_path");
            return Err("missing path parameter".into());
        }

        let raw_path = bind.get_parameter(0).to_string();
        debug!("docdb_scan bind: path='{}'", raw_path);

        // Path validation -- blocks traversal attacks
        let table_root = match validate_and_canonicalize_path(&raw_path) {
            Ok(p) => p,
            Err(e) => {
                bind.set_error(&e);
                return Err(e.into());
            }
        };

        // Try to reuse the current thread's RockDuck instance from TLS.
        // When provided via set_rockduck_for_vtab(), bind() uses it instead of re-opening.
        // This avoids O(segments) disk I/O from repeated RockDuck::open() calls.
        let rockduck_arc = match get_rockduck_for_vtab() {
            Some(db) => db,
            None => {
                let rockduck =
                    RockDuck::open(&table_root).map_err(|e| format!("open RockDuck: {}", e))?;
                Arc::new(rockduck)
            }
        };

        // Load all segment metadata for this table
        let all_metas = crate::metadata::list_segment_metas(&rockduck_arc.kv)
            .map_err(|e| format!("list segments: {}", e))?;

        let relevant: Vec<SegmentMeta> = all_metas
            .into_iter()
            .filter(|m| m.status == SegmentStatus::Active || m.status == SegmentStatus::Frozen)
            .collect();

        if relevant.is_empty() {
            info!(
                "docdb_scan bind: no segments found at '{}'",
                table_root.display()
            );
            return Ok(BindData {
                rockduck: rockduck_arc.clone(),
                table_root,
                table_id: String::new(),
                segment_ids: Vec::new(),
                column_names: Vec::new(),
                committed_txn: 0,
                filter: None,
                estimated_rows: 0,
                as_of_txn: None,
                current_seg_idx: AtomicUsize::new(0),
                readers: RwLock::new(HashMap::new()),
                chunk_idx_in_seg: RwLock::new(HashMap::new()),
                readers_initialized: AtomicUsize::new(0),
                hotspot_stats: RwLock::new(VtabReaderHotspotStats::default()),
                seg_nullable: RwLock::new(HashMap::new()),
                vis_readers: RwLock::new(HashMap::new()),
                tt_batches: RwLock::new(Vec::new()),
                tt_batch_idx: AtomicUsize::new(0),
                snapshot: rockduck_arc.snapshot(),
                projection_contract: ProjectionContract::vtab(),
            });
        }

        let schema_source = &relevant[0];
        let table_id = schema_source.table_id.clone();
        let segment_ids: Vec<String> = relevant.iter().map(|m| m.seg_id.clone()).collect();
        let column_names: Vec<String> = schema_source
            .columns
            .iter()
            .map(|c| c.name.clone())
            .collect();

        // Register result columns with DuckDB's schema
        for col_def in &schema_source.columns {
            let arrow_dtype = column_def_to_arrow_dtype(col_def);
            let logical_type = to_duckdb_logical_type(&arrow_dtype)
                .map_err(|e| format!("unsupported type {:?}: {}", col_def.data_type, e))?;
            bind.add_result_column(&col_def.name, logical_type);
        }

        // Filter expression (optional 2nd parameter)
        let filter = if param_count > 1 {
            let raw = bind.get_parameter(1).to_string();
            if raw.is_empty() {
                None
            } else {
                match filter_expr::parse_filter_expr(&raw) {
                    Ok(f) => {
                        debug!("Parsed filter: {:?}", f);
                        filter_expr::Expr::to_scan_filter(&f)
                    }
                    Err(e) => {
                        warn!("Filter parse error: {}", e);
                        bind.set_error(&format!("invalid filter expression: {}", e));
                        return Err(format!("invalid filter expression: {}", e).into());
                    }
                }
            }
        } else {
            None
        };

        // Cardinality hint
        let estimated_rows: u64 = relevant.iter().map(|m| m.row_count).sum();
        bind.set_cardinality(estimated_rows, true);

        // MVCC committed txn
        let committed_txn = rockduck_arc
            .kv
            .get(CF_MVCC, b"committed_txn")
            .ok()
            .flatten()
            .and_then(|v| postcard::from_bytes::<u64>(&v).ok())
            .unwrap_or(0);

        debug!(
            "docdb_scan bind: {} segments, ~{} rows, committed_txn={}",
            segment_ids.len(),
            estimated_rows,
            committed_txn
        );

        // Time travel: check parameter 2 for AS OF TxnId syntax
        // DuckDB passes this via get_extra_info() which contains the full SQL
        let as_of_txn = Self::parse_time_travel_from_bind(bind);

        // If time travel is requested, pre-load batches via TimeTravelScanner
        let tt_batches = if let Some(txn_id) = as_of_txn {
            // Derive the table name from segment metadata to pass to TimeTravelScanner.
            // This is the actual table_id stored in SegmentMeta (e.g. "orders"),
            // not the data directory path. The previous bug was passing data_dir
            // as the table name, which caused collect_segments to always return
            // empty (since no segment's table_id matches a filesystem path).
            let table_name = relevant
                .first()
                .map(|m| m.table_id.clone())
                .unwrap_or_default();
            info!(
                "docdb_scan bind: time travel to txn={}, table='{}'",
                txn_id, table_name
            );
            let scanner = crate::query::TimeTravelScanner::new(
                Arc::clone(&rockduck_arc.kv),
                table_name,
                txn_id,
                rockduck_arc.data_dir.clone(),
            )?;
            scanner.scan()?
        } else {
            Vec::new()
        };

        Ok(BindData {
            rockduck: rockduck_arc.clone(),
            table_root,
            table_id,
            segment_ids,
            column_names,
            committed_txn,
            filter,
            estimated_rows,
            as_of_txn,
            current_seg_idx: AtomicUsize::new(0),
            readers: RwLock::new(HashMap::new()),
            chunk_idx_in_seg: RwLock::new(HashMap::new()),
            readers_initialized: AtomicUsize::new(0),
            hotspot_stats: RwLock::new(VtabReaderHotspotStats::default()),
            seg_nullable: RwLock::new(HashMap::new()),
            vis_readers: RwLock::new(HashMap::new()),
            tt_batches: RwLock::new(tt_batches),
            tt_batch_idx: AtomicUsize::new(0),
            snapshot: rockduck_arc.snapshot(),
            projection_contract: ProjectionContract::vtab(),
        })
    }

    fn init(
        init: &InitInfo,
    ) -> std::result::Result<Self::InitData, Box<dyn std::error::Error + 'static>> {
        init.set_max_threads(1);

        let bind_ptr = init.get_bind_data::<Self::BindData>();
        if bind_ptr.is_null() {
            init.set_error("docdb_scan init missing bind data");
            return Err("docdb_scan init missing bind data".into());
        }

        // SAFETY: bind_ptr is valid for the lifetime of the VTab call
        let bind_data = unsafe { &*bind_ptr };
        let seg_start_idx = bind_data.current_seg_idx.load(Ordering::Relaxed);

        debug!("docdb_scan init: seg_start_idx={}", seg_start_idx);
        Ok(InitData { seg_start_idx })
    }

    fn func(
        func: &TableFunctionInfo<Self>,
        output: &mut DataChunkHandle,
    ) -> std::result::Result<(), Box<dyn std::error::Error + 'static>> {
        // SAFETY: references are valid for the lifetime of the TableFunctionInfo
        let bind_data: &BindData = func.get_bind_data();
        debug!(
            surface = ?bind_data.projection_contract.surface,
            visibility = ?bind_data.projection_contract.visibility,
            sidecar_class = ?bind_data.projection_contract.sidecar_class,
            evidence_hook = bind_data.projection_contract.evidence_hook,
            "docdb_scan projection contract"
        );
        bind_data.projection_contract.assert_blocking_governance();
        if let Some(router) = bind_data.rockduck.router.as_ref() {
            let executed_segment_ids = if bind_data.as_of_txn.is_some() {
                bind_data.segment_ids.clone()
            } else {
                Vec::new()
            };
            let sidecar_evidence = SidecarEvidenceSnapshot {
                table: bind_data.table_id.clone(),
                routed_segment_ids: bind_data.segment_ids.clone(),
                executed_segment_ids,
                contract: bind_data.projection_contract.clone(),
            };
            if bind_data.as_of_txn.is_none() {
                assert!(
                    sidecar_evidence.executed_segment_ids.is_empty(),
                    "live vtab sidecar path must stay metadata-only until execution attribution is authoritative"
                );
            }
            router.observe_sidecar_evidence(&bind_data.rockduck, &sidecar_evidence);
        }

        // Time travel path: pull from pre-loaded tt_batches
        if bind_data.as_of_txn.is_some() {
            let batch = {
                let batches = bind_data
                    .tt_batches
                    .read()
                    .expect("BindData tt_batches lock poisoned");
                let idx = bind_data.tt_batch_idx.load(Ordering::Relaxed);
                if idx < batches.len() {
                    Some(batches[idx].clone())
                } else {
                    None
                }
            };

            if let Some(batch) = batch {
                bind_data.tt_batch_idx.fetch_add(1, Ordering::Relaxed);
                let num_rows = batch.num_rows();
                if num_rows == 0 {
                    output.set_len(0);
                    return Ok(());
                }
                record_batch_to_duckdb_data_chunk(&batch, output)
                    .map_err(|e| format!("tt record_batch_to_duckdb_data_chunk: {}", e))?;
                debug!("docdb_scan func [time-travel]: emitted {} rows", num_rows);
                return Ok(());
            } else {
                output.set_len(0);
                return Ok(());
            }
        }

        // Normal path: lazy initialization of VortexReaders
        if bind_data
            .readers_initialized
            .compare_exchange(0, 1, Ordering::Acquire, Ordering::Relaxed)
            .is_ok()
        {
            // This thread won the race -- perform lazy init
            Self::lazy_init_readers(bind_data);
        }

        let batch = match bind_data.next_batch() {
            Some(b) => b,
            None => {
                output.set_len(0);
                return Ok(());
            }
        };

        let num_rows = batch.num_rows();
        if num_rows == 0 {
            output.set_len(0);
            return Ok(());
        }

        record_batch_to_duckdb_data_chunk(&batch, output)
            .map_err(|e| format!("record_batch_to_duckdb_data_chunk: {}", e))?;

        debug!("docdb_scan func: emitted {} rows", num_rows);
        Ok(())
    }

    fn parameters() -> Option<Vec<LogicalTypeHandle>> {
        Some(vec![LogicalTypeHandle::from(LogicalTypeId::Varchar)])
    }
}

impl RockDuckVTab {
    /// Parse `AS OF TxnId N` from the extra_info string (DuckDB passes the full SQL).
    ///
    /// DuckDB provides the raw SQL via `BindInfo::get_extra_info()` which contains
    /// the full query including the `AS OF TxnId N` clause. We delegate to the
    /// existing `parse_time_travel` function from `time_travel.rs`.
    fn parse_time_travel_from_bind(bind: &BindInfo) -> Option<u64> {
        let extra_ptr: *const std::ffi::c_char = bind.get_extra_info();
        if extra_ptr.is_null() {
            debug!("docdb_scan bind: no extra SQL info available for time travel parsing");
            return None;
        }

        // SAFETY: DuckDB returns a null-terminated string when extra info is present.
        let extra_cstr = unsafe { std::ffi::CStr::from_ptr(extra_ptr) };
        let extra_str = extra_cstr.to_string_lossy();

        match crate::query::time_travel::parse_time_travel(&extra_str) {
            Ok((_stripped, txn_id)) => txn_id,
            Err(err) => {
                warn!(%err, sql = %extra_str, "docdb_scan bind: invalid time-travel clause");
                None
            }
        }
    }

    /// Lazily open VortexReaders for all segments.
    fn lazy_init_readers(bind_data: &BindData) {
        bind_data
            .hotspot_stats
            .write()
            .expect("BindData hotspot_stats lock poisoned")
            .lazy_init_calls += 1;
        for seg_id in &bind_data.segment_ids {
            let seg_dir = bind_data.table_root.join("segments").join(seg_id);
            if !seg_dir.exists() {
                continue;
            }

            let seg_meta = match load_segment_meta(&bind_data.rockduck, seg_id) {
                Ok(m) => m,
                Err(e) => {
                    warn!("load seg meta for {}: {}", seg_id, e);
                    continue;
                }
            };

            let mut readers = Vec::new();
            for col_def in &seg_meta.columns {
                let col_path = seg_dir.join(format!("{}.vortex", col_def.name));
                if col_path.exists() {
                    bind_data
                        .hotspot_stats
                        .write()
                        .expect("BindData hotspot_stats lock poisoned")
                        .column_reader_open_attempts += 1;
                    match VortexReader::open(&col_path) {
                        Ok(r) => {
                            bind_data
                                .hotspot_stats
                                .write()
                                .expect("BindData hotspot_stats lock poisoned")
                                .column_reader_open_successes += 1;
                            readers.push(r)
                        }
                        Err(e) => {
                            warn!(
                                "Failed to open Vortex reader for {}: {}",
                                col_path.display(),
                                e
                            );
                        }
                    }
                }
            }

            if !readers.is_empty() {
                // collect nullable flags from segment metadata,
                // parallel to the reader order (column order on disk).
                let nullable_flags: Vec<bool> =
                    seg_meta.columns.iter().map(|c| c.nullable).collect();
                bind_data
                    .readers
                    .write()
                    .expect("BindData readers lock poisoned")
                    .insert(seg_id.clone(), readers);
                bind_data
                    .chunk_idx_in_seg
                    .write()
                    .expect("BindData chunk_idx lock poisoned")
                    .insert(seg_id.clone(), 0);
                bind_data
                    .seg_nullable
                    .write()
                    .expect("BindData seg_nullable lock poisoned")
                    .insert(seg_id.clone(), nullable_flags);

                // open __vis.vortex visibility reader for MVCC filtering.
                let vis_path = seg_dir.join("__vis.vortex");
                if vis_path.exists() {
                    bind_data
                        .hotspot_stats
                        .write()
                        .expect("BindData hotspot_stats lock poisoned")
                        .vis_reader_open_attempts += 1;
                    match VortexReader::open(&vis_path) {
                        Ok(vis_reader) => {
                            bind_data
                                .hotspot_stats
                                .write()
                                .expect("BindData hotspot_stats lock poisoned")
                                .vis_reader_open_successes += 1;
                            bind_data
                                .vis_readers
                                .write()
                                .expect("BindData vis_readers lock poisoned")
                                .insert(seg_id.clone(), vis_reader);
                        }
                        Err(e) => {
                            warn!("Failed to open vis reader for {}: {}", seg_id, e);
                        }
                    }
                }
            }
        }
    }
}

// =============================================================================
// Filter application
// =============================================================================

/// Apply a parsed `ScanFilter` to a `RecordBatch` using Arrow's vectorized filter kernel.
fn apply_scan_filter(
    batch: &RecordBatch,
    filter: &ScanFilter,
) -> std::result::Result<RecordBatch, String> {
    use arrow_select::filter::filter_record_batch;

    let filter_array = build_filter_mask(batch, filter)?;
    filter_record_batch(batch, &filter_array).map_err(|e| format!("apply filter: {}", e))
}

/// Build a BooleanArray mask from a `ScanFilter`.
fn build_filter_mask(
    batch: &RecordBatch,
    filter: &ScanFilter,
) -> std::result::Result<arrow_array::BooleanArray, String> {
    use arrow::compute::kernels::cmp::eq as arrow_eq;
    use arrow::compute::kernels::cmp::gt_eq as arrow_gt_eq;
    use arrow::compute::kernels::cmp::lt_eq as arrow_lt_eq;
    use arrow_arith::boolean;

    match filter {
        ScanFilter::Eq { column, value } => {
            let col = batch
                .column_by_name(column)
                .ok_or_else(|| format!("column not found: {}", column))?;
            let scalar = bytes_to_scalar(value, col.data_type())?;
            arrow_eq(col, &scalar).map_err(|e| format!("filter eq: {}", e))
        }
        ScanFilter::Range { column, min, max } => {
            let col = batch
                .column_by_name(column)
                .ok_or_else(|| format!("column not found: {}", column))?;
            let min_arr = bytes_to_scalar(min, col.data_type())?;
            let max_arr = bytes_to_scalar(max, col.data_type())?;
            let above_min = arrow_gt_eq(col, &min_arr).map_err(|e| format!("filter ge: {}", e))?;
            let below_max = arrow_lt_eq(col, &max_arr).map_err(|e| format!("filter le: {}", e))?;
            boolean::and(&above_min, &below_max).map_err(|e| format!("filter and: {}", e))
        }
        ScanFilter::Like { column, pattern } => {
            let col = batch
                .column_by_name(column)
                .ok_or_else(|| format!("column not found: {}", column))?;
            let contains = pattern.trim_matches('%');
            if let Some(str_arr) = col.as_any().downcast_ref::<arrow_array::StringArray>() {
                Ok(arrow_array::BooleanArray::from(
                    (0..str_arr.len())
                        .map(|i| str_arr.value(i).contains(contains))
                        .collect::<Vec<_>>(),
                ))
            } else {
                Ok(arrow_array::BooleanArray::from(vec![true; col.len()]))
            }
        }
        ScanFilter::Between { column, min, max } => {
            let col = batch
                .column_by_name(column)
                .ok_or_else(|| format!("column not found: {}", column))?;
            let min_arr = bytes_to_scalar(min, col.data_type())?;
            let max_arr = bytes_to_scalar(max, col.data_type())?;
            let ge = arrow_gt_eq(col, &min_arr).map_err(|e| format!("filter ge: {}", e))?;
            let le = arrow_lt_eq(col, &max_arr).map_err(|e| format!("filter le: {}", e))?;
            boolean::and(&ge, &le).map_err(|e| format!("filter and: {}", e))
        }
        ScanFilter::IsNull { column } => {
            let col = batch
                .column_by_name(column)
                .ok_or_else(|| format!("column not found: {}", column))?;
            use arrow::compute::is_null;
            is_null(col).map_err(|e| format!("filter is_null: {}", e))
        }
        ScanFilter::IsNotNull { column } => {
            let col = batch
                .column_by_name(column)
                .ok_or_else(|| format!("column not found: {}", column))?;
            use arrow::compute::is_not_null;
            is_not_null(col).map_err(|e| format!("filter is_not_null: {}", e))
        }
        ScanFilter::IsTrue { column } => {
            let col = batch
                .column_by_name(column)
                .ok_or_else(|| format!("column not found: {}", column))?;
            let scalar = bytes_to_scalar(&[1u8], &arrow_schema::DataType::Boolean)?;
            arrow_eq(col, &scalar).map_err(|e| format!("filter is_true: {}", e))
        }
        ScanFilter::IsNotTrue { column } => {
            let col = batch
                .column_by_name(column)
                .ok_or_else(|| format!("column not found: {}", column))?;
            let scalar = bytes_to_scalar(&[1u8], &arrow_schema::DataType::Boolean)?;
            let eq = arrow_eq(col, &scalar).map_err(|e| format!("filter is_not_true eq: {}", e))?;
            boolean::not(&eq).map_err(|e| format!("filter is_not_true not: {}", e))
        }
        ScanFilter::And(a, b) => {
            let a_mask = build_filter_mask(batch, a)?;
            let b_mask = build_filter_mask(batch, b)?;
            boolean::and(&a_mask, &b_mask).map_err(|e| format!("filter and: {}", e))
        }
        ScanFilter::Or(a, b) => {
            let a_mask = build_filter_mask(batch, a)?;
            let b_mask = build_filter_mask(batch, b)?;
            boolean::or(&a_mask, &b_mask).map_err(|e| format!("filter or: {}", e))
        }
        ScanFilter::Not(inner) => {
            let inner_mask = build_filter_mask(batch, inner)?;
            boolean::not(&inner_mask).map_err(|e| format!("filter not: {}", e))
        }
        ScanFilter::In { column, values } => {
            let col = batch
                .column_by_name(column)
                .ok_or_else(|| format!("column not found: {}", column))?;

            // Parse each value bytes into scalar arrays
            let scalars: std::result::Result<Vec<_>, _> = values
                .iter()
                .map(|v| bytes_to_scalar(v, col.data_type()))
                .collect();
            let scalars = scalars?;

            // Build OR mask: row is in the IN set if it equals any scalar value
            let mut combined_mask: arrow_array::BooleanArray = arrow_array::BooleanArray::from_iter(std::iter::repeat_n(false, col.len()));
            for scalar_arr in &scalars {
                let eq_arr = arrow_eq(col, scalar_arr)
                    .map_err(|e| format!("filter in eq: {}", e))?;
                combined_mask = boolean::or(&combined_mask, &eq_arr)
                    .map_err(|e| format!("filter in or: {}", e))?;
            }
            Ok(combined_mask)
        }
        ScanFilter::Gt { column, value } => {
            let col = batch
                .column_by_name(column)
                .ok_or_else(|| format!("column not found: {}", column))?;
            let scalar = bytes_to_scalar(value, col.data_type())?;
            use arrow::compute::kernels::cmp::gt;
            gt(col, &scalar).map_err(|e| format!("filter gt: {}", e))
        }
        ScanFilter::Ge { column, value } => {
            let col = batch
                .column_by_name(column)
                .ok_or_else(|| format!("column not found: {}", column))?;
            let scalar = bytes_to_scalar(value, col.data_type())?;
            arrow_gt_eq(col, &scalar).map_err(|e| format!("filter ge: {}", e))
        }
        ScanFilter::Lt { column, value } => {
            let col = batch
                .column_by_name(column)
                .ok_or_else(|| format!("column not found: {}", column))?;
            let scalar = bytes_to_scalar(value, col.data_type())?;
            use arrow::compute::kernels::cmp::lt;
            lt(col, &scalar).map_err(|e| format!("filter lt: {}", e))
        }
        ScanFilter::Le { column, value } => {
            let col = batch
                .column_by_name(column)
                .ok_or_else(|| format!("column not found: {}", column))?;
            let scalar = bytes_to_scalar(value, col.data_type())?;
            arrow_lt_eq(col, &scalar).map_err(|e| format!("filter le: {}", e))
        }
    }
}

fn bytes_to_scalar(
    bytes: &[u8],
    dtype: &arrow_schema::DataType,
) -> std::result::Result<ArrayRef, String> {
    use arrow_array::*;
    use arrow_schema::DataType;

    match dtype {
        DataType::Int8 => {
            let v = i8::from_le_bytes(bytes.try_into().map_err(|_| "bad i8 bytes")?);
            Ok(Arc::new(Int8Array::from(vec![Some(v)])))
        }
        DataType::Int16 => {
            let v = i16::from_le_bytes(bytes.try_into().map_err(|_| "bad i16 bytes")?);
            Ok(Arc::new(Int16Array::from(vec![Some(v)])))
        }
        DataType::Int32 => {
            let v = i32::from_le_bytes(bytes.try_into().map_err(|_| "bad i32 bytes")?);
            Ok(Arc::new(Int32Array::from(vec![Some(v)])))
        }
        DataType::Int64 => {
            let v = i64::from_le_bytes(bytes.try_into().map_err(|_| "bad i64 bytes")?);
            Ok(Arc::new(Int64Array::from(vec![Some(v)])))
        }
        DataType::UInt8 => {
            let v = u8::from_le_bytes(bytes.try_into().map_err(|_| "bad u8 bytes")?);
            Ok(Arc::new(UInt8Array::from(vec![Some(v)])))
        }
        DataType::UInt16 => {
            let v = u16::from_le_bytes(bytes.try_into().map_err(|_| "bad u16 bytes")?);
            Ok(Arc::new(UInt16Array::from(vec![Some(v)])))
        }
        DataType::UInt32 => {
            let v = u32::from_le_bytes(bytes.try_into().map_err(|_| "bad u32 bytes")?);
            Ok(Arc::new(UInt32Array::from(vec![Some(v)])))
        }
        DataType::UInt64 => {
            let v = u64::from_le_bytes(bytes.try_into().map_err(|_| "bad u64 bytes")?);
            Ok(Arc::new(UInt64Array::from(vec![Some(v)])))
        }
        DataType::Float32 => {
            let v = f32::from_le_bytes(bytes.try_into().map_err(|_| "bad f32 bytes")?);
            Ok(Arc::new(Float32Array::from(vec![Some(v)])))
        }
        DataType::Float64 => {
            let v = f64::from_le_bytes(bytes.try_into().map_err(|_| "bad f64 bytes")?);
            Ok(Arc::new(Float64Array::from(vec![Some(v)])))
        }
        DataType::Utf8 => {
            let s = String::from_utf8_lossy(bytes).to_string();
            Ok(Arc::new(StringArray::from(vec![s])))
        }
        DataType::Boolean => {
            let v = bytes.first().copied().unwrap_or(0) != 0;
            Ok(Arc::new(BooleanArray::from(vec![v])))
        }
        DataType::Date32 => {
            let v = i32::from_le_bytes(bytes.try_into().map_err(|_| "bad Date32 bytes")?);
            Ok(Arc::new(Date32Array::from(vec![v])))
        }
        DataType::Date64 => {
            let v = i64::from_le_bytes(bytes.try_into().map_err(|_| "bad Date64 bytes")?);
            Ok(Arc::new(Date64Array::from(vec![v])))
        }
        _ => Err(format!("unsupported filter type: {:?}", dtype)),
    }
}

#[allow(dead_code)]
fn arrays_equal(a: &dyn Array, b: &dyn Array) -> bool {
    if let (Some(a_int), Some(b_int)) = (
        a.as_any().downcast_ref::<arrow_array::Int64Array>(),
        b.as_any().downcast_ref::<arrow_array::Int64Array>(),
    ) {
        return a_int.value(0) == b_int.value(0);
    }
    if let (Some(a_str), Some(b_str)) = (
        a.as_any().downcast_ref::<arrow_array::StringArray>(),
        b.as_any().downcast_ref::<arrow_array::StringArray>(),
    ) {
        return a_str.value(0) == b_str.value(0);
    }
    if let (Some(a_f64), Some(b_f64)) = (
        a.as_any().downcast_ref::<arrow_array::Float64Array>(),
        b.as_any().downcast_ref::<arrow_array::Float64Array>(),
    ) {
        return (a_f64.value(0) - b_f64.value(0)).abs() < 1e-9;
    }
    if let (Some(a_f32), Some(b_f32)) = (
        a.as_any().downcast_ref::<arrow_array::Float32Array>(),
        b.as_any().downcast_ref::<arrow_array::Float32Array>(),
    ) {
        return (a_f32.value(0) - b_f32.value(0)).abs() < 1e-6;
    }
    if let (Some(a_u64), Some(b_u64)) = (
        a.as_any().downcast_ref::<arrow_array::UInt64Array>(),
        b.as_any().downcast_ref::<arrow_array::UInt64Array>(),
    ) {
        return a_u64.value(0) == b_u64.value(0);
    }
    if let (Some(a_u32), Some(b_u32)) = (
        a.as_any().downcast_ref::<arrow_array::UInt32Array>(),
        b.as_any().downcast_ref::<arrow_array::UInt32Array>(),
    ) {
        return a_u32.value(0) == b_u32.value(0);
    }
    if let (Some(a_i32), Some(b_i32)) = (
        a.as_any().downcast_ref::<arrow_array::Int32Array>(),
        b.as_any().downcast_ref::<arrow_array::Int32Array>(),
    ) {
        return a_i32.value(0) == b_i32.value(0);
    }
    false
}

// =============================================================================
// Helpers
// =============================================================================

fn validate_and_canonicalize_path(raw_path: &str) -> std::result::Result<PathBuf, String> {
    let root = VTAB_DATA_ROOT
        .get()
        .ok_or_else(|| "vTab data root is not configured".to_string())?;

    let joined = if PathBuf::from(raw_path).is_absolute() {
        PathBuf::from(raw_path)
    } else {
        root.join(raw_path)
    };

    let canonical = std::fs::canonicalize(&joined)
        .map_err(|e| format!("canonicalize '{}': {}", joined.display(), e))?;

    let root_str = root
        .canonicalize()
        .map_err(|e| format!("canonicalize root: {}", e))?
        .to_string_lossy()
        .trim_end_matches(['/', '\\'])
        .to_lowercase();

    let canonical_str = canonical
        .to_string_lossy()
        .trim_end_matches(['/', '\\'])
        .to_lowercase();

    if !canonical_str.starts_with(&root_str) {
        return Err(format!(
            "Path '{}' escapes data root '{}' -- traversal blocked",
            canonical.display(),
            root.display()
        ));
    }

    if !canonical.is_dir() {
        return Err(format!("'{}' is not a directory", canonical.display()));
    }

    debug!(
        "Validated vTab path: {} -> {}",
        raw_path,
        canonical.display()
    );
    Ok(canonical)
}

/// now takes Arc<RockDuck> instead of re-opening for each segment.
/// Uses the stored RockDuck instance from BindData instead of re-opening per segment.
fn load_segment_meta(
    rockduck: &Arc<RockDuck>,
    seg_id: &str,
) -> std::result::Result<SegmentMeta, String> {
    get_segment_meta(&rockduck.kv, seg_id)
        .map_err(|e| format!("get seg meta: {}", e))?
        .ok_or_else(|| format!("segment not found: {}", seg_id))
}

fn column_def_to_arrow_dtype(col: &crate::segment::meta::ColumnDef) -> arrow_schema::DataType {
    use crate::segment::meta::DataType as RD;
    match col.data_type {
        RD::Int8 | RD::Int16 | RD::UInt8 | RD::UInt16 | RD::UInt32 => arrow_schema::DataType::Int32,
        RD::Int32 => arrow_schema::DataType::Int32,
        RD::UInt64 | RD::Int64 => arrow_schema::DataType::Int64,
        RD::Float32 => arrow_schema::DataType::Float32,
        RD::Float64 => arrow_schema::DataType::Float64,
        RD::Bool => arrow_schema::DataType::Boolean,
        RD::Utf8 | RD::LargeUtf8 => arrow_schema::DataType::Utf8,
        RD::Binary | RD::LargeBinary => arrow_schema::DataType::Binary,
        RD::Date32 => arrow_schema::DataType::Date32,
        RD::Date64 => arrow_schema::DataType::Date64,
        RD::TimestampMicros => {
            arrow_schema::DataType::Timestamp(arrow_schema::TimeUnit::Microsecond, None)
        }
        RD::TimestampMillis => {
            arrow_schema::DataType::Timestamp(arrow_schema::TimeUnit::Millisecond, None)
        }
    }
}

#[cfg(test)]
mod governance_contract_tests {
    use super::*;

    #[test]
    fn vtab_path_validation_rejects_traversal_outside_root() {
        let temp_root = std::env::temp_dir().join(format!(
            "rockduck-vtab-root-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system time before unix epoch")
                .as_nanos()
        ));
        let allowed_dir = temp_root.join("allowed");
        let outside_dir = temp_root.join("outside");
        std::fs::create_dir_all(&allowed_dir).expect("create allowed dir");
        std::fs::create_dir_all(&outside_dir).expect("create outside dir");
        let _ = set_vtab_data_root(allowed_dir.clone());

        let err = validate_and_canonicalize_path(outside_dir.to_string_lossy().as_ref())
            .expect_err("outside path should be rejected");
        assert!(
            err.contains("escapes data root") || err.contains("outside configured data root"),
            "unexpected error: {}",
            err
        );

        let _ = std::fs::remove_dir_all(&temp_root);
    }
}
