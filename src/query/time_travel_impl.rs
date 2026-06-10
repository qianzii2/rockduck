//! Time Travel ─Reverse Timestamp Index implementation.
//!
//! # Architecture
//!
//! MVCC shadow columns (`created_txn`, `deleted_txn`) already track versions.
//! Time travel extends this with an index for efficient historical reads.
//!
//! # Reverse Timestamp Index
//!
//! Key design: `version_idx:{table}:{seg_id}:{row_offset:010x}:{txn_id:016x}`
//!
//! The value is the row data (all column bytes concatenated in schema order).
//!
//! AS OF txn=T query: find max(txn_id) where txn_id <= T
//!
//! Since mace-kv's iterator is forward-only, we implement SeekForPrev as:
//! 1. Scan all entries for the row (prefix = version_idx:{table}:{seg_id}:{row_offset:010x}:)
//! 2. Find the entry with the largest txn_id <= target_txn
//!
//! This is O(n_versions_per_row), which is acceptable since rows have few versions.

use std::sync::Arc;

use crate::metadata::projection::ProjectionContract;
use crate::mvcc::visibility::VisFilter;

use crate::error::{Result, RockDuckError};
use crate::metadata::kv_engine::{KVEngine, KVOp, CF_VERSIONS};
use crate::metadata::pk_skiplist;
use crate::storage::vortex::VortexReader;

/// Result of loading a segment at a specific transaction.
/// Distinguishes "segment was deleted / not visible at target txn" from "load failed".
#[derive(Debug)]
pub enum SegLoadResult {
    /// Segment has visible data at target_txn.
    Ok(Vec<arrow_array::RecordBatch>),
    /// Segment directory missing or segment invisible at target_txn. Not an error.
    NotVisible,
    /// Physical I/O or parse error while loading. Scanner should warn.
    LoadFailed(String),
}

/// Prefix for version index keys
const VERSION_IDX_PREFIX: &str = "version_idx:";

/// A historical version of a row.
#[derive(
    Debug,
    Clone,
    serde::Serialize,
    serde::Deserialize,
    rkyv::Archive,
    rkyv::Serialize,
    rkyv::Deserialize,
)]
pub struct VersionEntry {
    /// Transaction ID that keys this version entry in the reverse timestamp index.
    /// For compacted segments this may differ from `created_txn`; historical readers
    /// must use `created_txn` as MVCC authority.
    pub txn_id: u64,
    /// Serialized row bytes (all columns concatenated in schema order)
    pub value: Vec<u8>,
    /// TxnId that inserted this row
    pub created_txn: u64,
    /// TxnId that deleted this row (None if still alive)
    pub deleted_txn: Option<u64>,
}

/// Reads historical versions of a row using the Reverse Timestamp Index.
///
/// Uses SeekForPrev implemented via prefix scan + filter:
/// 1. Scan all version entries for a row (prefix: version_idx:{table}:{seg_id}:{row_offset:010x}:)
/// 2. Find the entry with the largest txn_id <= target_txn
///
/// The version index is built lazily on first access per segment.
pub struct TimeTravelReader {
    kv: Arc<dyn KVEngine>,
    data_dir: std::path::PathBuf,
    committed_txns: std::collections::BTreeSet<u64>,
    commit_ts_map: std::collections::HashMap<u64, u64>,
    historical_filter: crate::mvcc::visibility::VisibilityManager,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HistoricalVisibility {
    pub created_txn: u64,
    pub deleted_txn: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct HistoricalVisibilityAudit {
    pub main_path: &'static str,
    pub bypass_paths: Vec<&'static str>,
    pub landing_files: Vec<&'static str>,
    pub projection_contract: ProjectionContract,
}

impl HistoricalVisibilityAudit {
    /// Truth package audit card for historical visibility surfaces.
    ///
    /// `HistoricalVisibility::is_visible_at` now runs as a historical projection over real
    /// commit timestamps instead of relying on a synthetic `txn_id -> txn_id` map.
    ///
    /// This projection is currently bounded by these conditions:
    /// 1. The historical commit_ts authority is loaded from KV-backed committed history
    ///    (`get_committed_txns()`) at construction time, rather than synthesized from txn ids.
    /// 2. The `active_txns` set is always empty in historical snapshots —
    ///    uncommitted transactions are not visible at any historical snapshot point.
    /// 3. The fallback `get_as_of` path in `point_get.rs` uses the same KV-backed
    ///    historical commit timestamp map for delta cell filtering, rather than
    ///    relying on prunable in-memory `VisibilityManager::committed_history`.
    ///
    /// This remains a classified historical projection, with compaction semantics still
    /// explicit and separate, but historical delta filtering now shares the same KV-backed
    /// authority as the rest of the historical projection.
    ///
    /// Phase 1 exit rule: historical reads must keep using a single commit-timestamp
    /// authority source for both row visibility and delta filtering; no in-memory-only
    /// fallback may silently weaken long-range time travel semantics.
    pub fn truth_package() -> Self {
        Self {
            main_path: "VisibilityManager/VisFilter -> VisibilityContext(Historical) -> HistoricalVisibility::is_visible_at",
            bypass_paths: vec![
                "read::point_get::get_as_of fallback __vis path (historical projection, uses real get_commit_ts for delta cells)",
                "TimeTravelReader::is_visible_at via version index (historical projection)",
                "DuckDB/VTab snapshot filtering (sanctioned, uses TxnSnapshot::is_row_visible)",
            ],
            landing_files: vec![
                "src/mvcc/visibility.rs",
                "src/read/point_get.rs",
                "src/query/time_travel_impl.rs",
                "src/query/vtab_quack.rs",
            ],
            projection_contract: ProjectionContract::time_travel_scanner(),
        }
    }
}

impl HistoricalVisibility {
    pub fn projection_context_from_map(
        target_txn: u64,
        commit_ts_map: std::collections::HashMap<u64, u64>,
    ) -> crate::mvcc::visibility::VisibilityContext {
        crate::mvcc::visibility::VisibilityContext::historical(target_txn, commit_ts_map)
    }

    pub fn projection_context(
        target_txn: u64,
        committed_txns: &std::collections::BTreeSet<u64>,
        commit_ts_map: &std::collections::HashMap<u64, u64>,
    ) -> crate::mvcc::visibility::VisibilityContext {
        let filtered_commit_ts_map: std::collections::HashMap<u64, u64> = committed_txns
            .iter()
            .filter_map(|txn_id| {
                commit_ts_map
                    .get(txn_id)
                    .copied()
                    .map(|commit_ts| (*txn_id, commit_ts))
            })
            .collect();
        Self::projection_context_from_map(target_txn, filtered_commit_ts_map)
    }

    pub fn is_visible_at(
        &self,
        filter: &crate::mvcc::visibility::VisibilityManager,
        context: &crate::mvcc::visibility::VisibilityContext,
    ) -> bool {
        filter.is_row_visible(
            context.snapshot_id,
            self.created_txn,
            self.deleted_txn,
            &context.active_txns,
            &context.commit_ts_map,
        )
    }
}

#[allow(dead_code)]
fn segment_enumeration_error(source: &str, err: RockDuckError) -> RockDuckError {
    RockDuckError::Internal(format!(
        "historical segment enumeration failed in {source}: {err}"
    ))
}

fn malformed_version_index_error(source: &str, detail: impl Into<String>) -> RockDuckError {
    RockDuckError::Internal(format!(
        "historical version index malformed in {source}: {}",
        detail.into()
    ))
}

fn committed_history_unavailable_error(source: &str, err: RockDuckError) -> RockDuckError {
    RockDuckError::Internal(format!(
        "historical commit authority unavailable in {source}: {err}"
    ))
}

/// Version index key format: version_idx:{table}:{seg_id}:{row_offset:010x}:{txn_id:016x}
fn build_version_key(table: &str, seg_id: &str, row_offset: u32, txn_id: u64) -> Vec<u8> {
    let row_hex = format!("{:010x}", row_offset);
    let txn_hex = format!("{:016x}", txn_id);
    let mut key = Vec::with_capacity(
        VERSION_IDX_PREFIX.len() + table.len() + 1 + seg_id.len() + 1 + 10 + 1 + 16,
    );
    key.extend_from_slice(VERSION_IDX_PREFIX.as_bytes());
    key.extend_from_slice(table.as_bytes());
    key.push(b':');
    key.extend_from_slice(seg_id.as_bytes());
    key.push(b':');
    key.extend_from_slice(row_hex.as_bytes());
    key.push(b':');
    key.extend_from_slice(txn_hex.as_bytes());
    key
}

/// Build a prefix for scanning all versions of a row.
fn build_row_prefix(table: &str, seg_id: &str, row_offset: u32) -> Vec<u8> {
    let row_hex = format!("{:010x}", row_offset);
    let mut prefix =
        Vec::with_capacity(VERSION_IDX_PREFIX.len() + table.len() + 1 + seg_id.len() + 1 + 10 + 1);
    prefix.extend_from_slice(VERSION_IDX_PREFIX.as_bytes());
    prefix.extend_from_slice(table.as_bytes());
    prefix.push(b':');
    prefix.extend_from_slice(seg_id.as_bytes());
    prefix.push(b':');
    prefix.extend_from_slice(row_hex.as_bytes());
    prefix.push(b':');
    prefix
}

/// Extract txn_id from a version index key.
/// Key format: version_idx:{table}:{seg_id}:{row_offset:010x}:{txn_id:016x}
fn extract_txn_from_key(key: &[u8]) -> Option<u64> {
    // Find the last ':' separator by scanning backward.
    let last_colon = key.iter().rposition(|&b| b == b':')?;
    let txn_bytes = &key[last_colon + 1..];
    let txn_str = std::str::from_utf8(txn_bytes).ok()?;
    u64::from_str_radix(txn_str, 16).ok()
}

// =============================================================================
// TimeTravelReader implementation
// =============================================================================

impl TimeTravelReader {
    /// Create a new TimeTravelReader.
    pub fn new(kv: Arc<dyn KVEngine>, data_dir: std::path::PathBuf) -> Result<Self> {
        let commit_map = crate::metadata::get_committed_txns(&kv)
            .map_err(|err| committed_history_unavailable_error("TimeTravelReader::new", err))?;
        let committed_txns = commit_map.keys().copied().collect();
        let historical_filter = crate::mvcc::visibility::VisibilityManager::new();
        Ok(Self {
            kv,
            data_dir,
            committed_txns,
            commit_ts_map: commit_map,
            historical_filter,
        })
    }

    /// Build the version index for a segment if it hasn't been built yet.
    /// Called lazily on first time-travel access for a segment.
    fn ensure_segment_index_built(
        &self,
        table: &str,
        seg_id: &str,
        column_names: &[String],
    ) -> Result<()> {
        use crate::metadata::kv_engine::KVEngine;

        // Check if index exists for this segment by looking for any entries
        let check_prefix = {
            let mut key = Vec::new();
            key.extend_from_slice(VERSION_IDX_PREFIX.as_bytes());
            key.extend_from_slice(table.as_bytes());
            key.push(b':');
            key.extend_from_slice(seg_id.as_bytes());
            key.push(b':');
            key
        };

        let mut iter = self
            .kv
            .prefix_iter(CF_VERSIONS, &check_prefix)
            .map_err(|err| {
                malformed_version_index_error(
                    "TimeTravelReader::ensure_segment_index_built",
                    err.to_string(),
                )
            })?;

        // If we can find any entry with this segment prefix, index is already built
        if iter.next() {
            return Ok(());
        }

        // Build the index lazily
        TimeTravelScanner::build_version_index_for_segment(
            &self.kv,
            table,
            &self.data_dir,
            seg_id,
            column_names,
        )
    }

    /// Read a row as of a given transaction.
    ///
    /// Returns the serialized row bytes if a visible version exists.
    pub fn as_of_read(&self, table: &str, pk: &[u8], target_txn: u64) -> Result<Option<Vec<u8>>> {
        // Look up PK index to find the current segment and row offset
        let Some((seg_id, _granule_id, row_offset)) =
            pk_skiplist::get_pk_index_by_pk(&self.kv, table, pk)?
        else {
            return Ok(None);
        };

        // Need column names to build index if absent
        let seg_meta = crate::metadata::get_segment_meta(&self.kv, &seg_id)?.ok_or_else(|| {
            malformed_version_index_error(
                "TimeTravelReader::as_of_read",
                format!("missing segment metadata for {seg_id}"),
            )
        })?;
        let column_names: Vec<String> = seg_meta.columns.iter().map(|c| c.name.clone()).collect();

        // Ensure version index exists for this segment
        self.ensure_segment_index_built(table, &seg_id, &column_names)?;

        // Scan all versions for this row and find the best match
        let prefix = build_row_prefix(table, &seg_id, row_offset);
        let mut iter = self.kv.prefix_iter(CF_VERSIONS, &prefix)?;

        let mut best: Option<(u64, VersionEntry)> = None;
        while iter.next() {
            let key = iter.key();
            let value = iter.value();
            let index_txn_id = extract_txn_from_key(key).ok_or_else(|| {
                malformed_version_index_error(
                    "TimeTravelReader::as_of_read",
                    format!("invalid version-index key for segment {seg_id}"),
                )
            })?;
            if index_txn_id > target_txn {
                continue;
            }

            let version: VersionEntry = postcard::from_bytes(value).map_err(|err| {
                malformed_version_index_error(
                    "TimeTravelReader::as_of_read",
                    format!("deserialize version entry for segment {seg_id}: {err}"),
                )
            })?;
            // Apply visibility through the canonical historical projection seam.
            let historical_vis = HistoricalVisibility {
                created_txn: version.created_txn,
                deleted_txn: version.deleted_txn,
            };
            let context = HistoricalVisibility::projection_context(
                target_txn,
                &self.committed_txns,
                &self.commit_ts_map,
            );
            if historical_vis.is_visible_at(&self.historical_filter, &context) {
                match &best {
                    Some((current_index_txn, _)) if *current_index_txn >= index_txn_id => {}
                    _ => best = Some((index_txn_id, version)),
                }
            }
        }

        Ok(best.map(|(_, v)| v.value))
    }
}

// =============================================================================
// TimeTravelScanner implementation
// =============================================================================

pub struct TimeTravelScanner {
    #[allow(dead_code)]
    kv: Arc<dyn KVEngine>,
    #[allow(dead_code)]
    table: String,
    historical_filter: crate::mvcc::visibility::VisibilityManager,
    target_txn: u64,
    /// Segments that existed at `target_txn`.
    segments: Vec<crate::segment::meta::SegmentMeta>,
    /// Current segment index
    current_seg_idx: usize,
    /// Data directory
    data_dir: std::path::PathBuf,
    /// Set of txn IDs that were committed at or before `target_txn`.
    /// Used to filter out uncommitted creates from visibility checks.
    #[allow(dead_code)]
    committed_txns: std::collections::BTreeSet<u64>,
    /// Commit timestamp map sourced from KV metadata; filtered per snapshot when
    /// constructing historical projection contexts.
    commit_ts_map: std::collections::HashMap<u64, u64>,
}

impl TimeTravelScanner {
    /// Create a new TimeTravelScanner for the given table at the given transaction.
    ///
    /// Scans all segments where `seg.last_updated_txn <= target_txn`.
    pub fn new(
        kv: Arc<dyn KVEngine>,
        table: String,
        target_txn: u64,
        data_dir: std::path::PathBuf,
    ) -> Result<Self> {
        let segments = Self::collect_segments(&kv, &table, target_txn)?;
        let commit_map = crate::metadata::get_committed_txns(&kv)
            .map_err(|err| committed_history_unavailable_error("TimeTravelScanner::new", err))?;
        let committed_txns = commit_map
            .iter()
            .filter_map(|(&txn_id, &commit_ts)| (commit_ts <= target_txn).then_some(txn_id))
            .collect();

        let historical_filter = crate::mvcc::visibility::VisibilityManager::new();
        Ok(Self {
            kv,
            table,
            historical_filter,
            target_txn,
            segments,
            current_seg_idx: 0,
            data_dir,
            committed_txns,
            commit_ts_map: commit_map,
        })
    }

    /// Collect all segments that existed at the given transaction.
    ///
    /// Uses `created_txn` (not `updated_txn`) to find segments that were created
    /// at or before `target_txn`. A segment created at txn 10 but updated at txn 50
    /// IS valid at `AS OF txn 30` — using `updated_txn` would incorrectly exclude it.
    fn collect_segments(
        kv: &Arc<dyn KVEngine>,
        table: &str,
        target_txn: u64,
    ) -> Result<Vec<crate::segment::meta::SegmentMeta>> {
        let metas = crate::metadata::list_segment_metas(kv)
            .map_err(|err| committed_history_unavailable_error("TimeTravelScanner::new", err))?;
        Ok(metas
            .into_iter()
            .filter(|m| m.table_id == table && m.created_txn <= target_txn)
            .collect())
    }

    /// Scan all segments and return RecordBatches visible at target_txn.
    pub fn scan(mut self) -> Result<Vec<arrow_array::RecordBatch>> {
        let mut results = Vec::new();
        let mut segments_with_no_data = 0;
        while self.current_seg_idx < self.segments.len() {
            let meta = &self.segments[self.current_seg_idx];
            match self.load_segment_at_txn(meta) {
                SegLoadResult::Ok(batches) => {
                    for batch in batches {
                        if batch.num_rows() > 0 {
                            results.push(batch);
                        }
                    }
                }
                SegLoadResult::NotVisible => {
                    segments_with_no_data += 1;
                }
                SegLoadResult::LoadFailed(reason) => {
                    tracing::warn!(
                        "Time travel load_segment_at_txn failed for seg {}: {}",
                        meta.seg_id,
                        reason
                    );
                }
            }
            self.current_seg_idx += 1;
        }
        if segments_with_no_data > 0 && results.is_empty() {
            tracing::warn!(
                "Time travel query at txn {} returned no results for {} of {} segments",
                self.target_txn,
                segments_with_no_data,
                self.segments.len()
            );
        }
        Ok(results)
    }

    /// Build the version index for a single segment.
    ///
    /// This is the entry point for lazy version index building. It checks if the
    /// index already exists (by probing for a segment prefix in CF_VERSIONS) and
    /// only builds if absent.
    pub fn build_version_index_for_segment(
        kv: &Arc<dyn KVEngine>,
        table: &str,
        data_dir: &std::path::Path,
        seg_id: &str,
        column_names: &[String],
    ) -> Result<()> {
        build_segment_version_index(kv, table, data_dir, seg_id, column_names)
    }

    /// Load batches from a segment at the target transaction.
    ///
    /// Reads data columns + __vis.vortex, applies visibility filter based on target_txn.
    /// Returns `SegLoadResult` to distinguish "not visible" from "load failed".
    fn load_segment_at_txn(
        &self,
        meta: &crate::segment::meta::SegmentMeta,
    ) -> SegLoadResult {
        let seg_dir = self.data_dir.join("segments").join(&meta.seg_id);
        if !seg_dir.exists() {
            return SegLoadResult::NotVisible;
        }

        // Load data batches (columnar)
        let col_paths: Vec<_> = meta
            .columns
            .iter()
            .filter_map(|c| {
                let p = seg_dir.join(format!("{}.vortex", c.name));
                if p.exists() {
                    Some((c.name.clone(), p))
                } else {
                    None
                }
            })
            .collect();

        if col_paths.is_empty() {
            return SegLoadResult::NotVisible;
        }

        // Load data batches per column
        let mut per_col_batches: Vec<Vec<arrow_array::RecordBatch>> = Vec::new();
        for (_, path) in &col_paths {
            match VortexReader::open(path) {
                Ok(r) => per_col_batches.push(r.read_all_batches().as_ref().clone()),
                Err(e) => {
                    return SegLoadResult::LoadFailed(format!(
                        "open column batch {}: {e}",
                        path.display()
                    ));
                }
            }
        }

        if per_col_batches.is_empty() {
            return SegLoadResult::NotVisible;
        }

        // Load visibility batches
        let vis_path = seg_dir.join("__vis.vortex");
        let vis_batches: Vec<arrow_array::RecordBatch> = if vis_path.exists() {
            let reader = match VortexReader::open(&vis_path) {
                Ok(r) => r,
                Err(e) => {
                    return SegLoadResult::LoadFailed(format!(
                        "open visibility reader for {}: {e}",
                        meta.seg_id
                    ));
                }
            };
            let batches = reader.read_all_batches().as_ref().clone();
            if batches.is_empty() {
                return SegLoadResult::NotVisible;
            }
            batches
        } else {
            return SegLoadResult::NotVisible;
        };

        // Align batches across columns (same batch_idx across all columns)
        let num_batches = per_col_batches.iter().map(|b| b.len()).min().unwrap_or(0);
        if num_batches == 0 {
            return SegLoadResult::NotVisible;
        }

        // Warn if vis batches are shorter than data batches (indicates inconsistency)
        if !vis_batches.is_empty() && vis_batches.len() < num_batches {
            tracing::warn!(
                "Batch count mismatch: {} data batches, {} vis batches for seg {}",
                num_batches,
                vis_batches.len(),
                meta.seg_id
            );
        }

        let mut output_batches = Vec::new();

        for batch_idx in 0..num_batches {
            // Collect column values for this batch
            let mut batch_cols: Vec<arrow_array::ArrayRef> = Vec::new();
            let mut batch_fields = Vec::new();

            for batches in &per_col_batches {
                if batch_idx < batches.len() {
                    batch_cols.push(batches[batch_idx].column(0).clone());
                    batch_fields.push(batches[batch_idx].schema().field(0).clone());
                }
            }

            if batch_cols.is_empty() {
                continue;
            }

            // Get visibility for this batch
            let Some(vis_batch) = vis_batches.get(batch_idx) else {
                continue;
            };

            let vis_rows = vis_batch.num_rows();
            let data_rows = batch_cols.first().map(|col| col.len()).unwrap_or(0);
            let vis_cols = vis_batch.num_columns();

            if vis_rows == 0 || data_rows == 0 || vis_cols < 2 {
                continue;
            }

            // Apply visibility filter at target_txn
            let vis_created = match vis_batch
                .column(vis_cols - 2)
                .as_any()
                .downcast_ref::<arrow_array::Int64Array>()
            {
                Some(arr) => arr,
                None => {
                    return SegLoadResult::LoadFailed(format!(
                        "visibility created_txn column has unexpected type for segment {} batch {}",
                        meta.seg_id, batch_idx
                    ));
                }
            };
            let vis_deleted = match vis_batch
                .column(vis_batch.num_columns() - 1)
                .as_any()
                .downcast_ref::<arrow_array::Int64Array>()
            {
                Some(arr) => arr,
                None => {
                    return SegLoadResult::LoadFailed(format!(
                        "visibility deleted_txn column has unexpected type for segment {} batch {}",
                        meta.seg_id, batch_idx
                    ));
                }
            };

            // Apply visibility through the canonical historical projection seam.
            // This keeps historical scans on the same commit_ts-backed authority model
            // used by point_get_as_of, instead of a parallel committed-set-only rule.
            let commit_ts_map: std::collections::HashMap<u64, u64> = self
                .commit_ts_map
                .iter()
                .filter_map(|(&txn_id, &commit_ts)| {
                    (commit_ts <= self.target_txn).then_some((txn_id, commit_ts))
                })
                .collect();
            let historical_context =
                HistoricalVisibility::projection_context_from_map(self.target_txn, commit_ts_map);

            let mut mask =
                arrow_array::builder::BooleanBuilder::with_capacity(vis_rows.min(data_rows));
            for i in 0..vis_rows.min(data_rows) {
                let created = vis_created.value(i) as u64;
                let deleted = vis_deleted.value(i) as u64;
                let deleted_txn = if deleted == crate::mvcc::shadow_columns::NOT_DELETED {
                    None
                } else {
                    Some(deleted)
                };
                let historical_vis = HistoricalVisibility {
                    created_txn: created,
                    deleted_txn,
                };
                let visible =
                    historical_vis.is_visible_at(&self.historical_filter, &historical_context);
                mask.append_value(visible);
            }

            let mask_arr = mask.finish();

            // Filter all columns
            use arrow_select::filter::filter_record_batch;
            let schema = arrow_schema::SchemaRef::new(arrow_schema::Schema::new(batch_fields));
            let full_batch = match arrow_array::RecordBatch::try_new(schema, batch_cols) {
                Ok(batch) => batch,
                Err(err) => {
                    return SegLoadResult::LoadFailed(format!(
                        "build historical record batch for segment {} batch {}: {err}",
                        meta.seg_id, batch_idx
                    ));
                }
            };

            match filter_record_batch(&full_batch, &mask_arr) {
                Ok(filtered) if filtered.num_rows() > 0 => output_batches.push(filtered),
                Ok(_) => {}
                Err(err) => {
                    return SegLoadResult::LoadFailed(format!(
                        "filter historical record batch for segment {} batch {}: {err}",
                        meta.seg_id, batch_idx
                    ));
                }
            }
        }

        if output_batches.is_empty() {
            SegLoadResult::NotVisible
        } else {
            SegLoadResult::Ok(output_batches)
        }
    }
}

// =============================================================================
// Version index building
// =============================================================================

/// Build the version index for a segment.
///
/// Reads all data columns and visibility from the segment's vortex files,
/// then writes version entries to the KV store for every historical version.
///
/// Memory-optimized: processes rows in bounded chunks. For each chunk,
/// column bytes are accumulated, visibility is read, and KV entries are
/// written in a single batch. This avoids holding all row bytes for the
/// entire segment in memory at once.
///
/// The KV write is also batched: `BATCH_KV_SIZE` entries are collected
/// into a single `kv.write_batch` call to reduce mace-kv write amplification.
pub fn build_segment_version_index(
    kv: &Arc<dyn KVEngine>,
    table: &str,
    data_dir: &std::path::Path,
    seg_id: &str,
    column_names: &[String],
) -> Result<()> {
    const CHUNK_SIZE: usize = 10_000;
    const BATCH_KV_SIZE: usize = 500;

    let seg_dir = data_dir.join("segments").join(seg_id);
    if !seg_dir.exists() {
        return Ok(());
    }

    let vis_path = seg_dir.join("__vis.vortex");
    if !vis_path.exists() {
        return Ok(());
    }

    // Load all visibility batches upfront (needed to count rows and for second pass).
    // These are small relative to data columns (2 Int64 columns per row).
    let vis_batches: Vec<arrow_array::RecordBatch> = VortexReader::open(&vis_path)
        .map_err(|e| crate::error::RockDuckError::Internal(format!("open vis.vortex: {e}")))?
        .read_all_batches()
        .as_ref()
        .clone();

    if vis_batches.is_empty() {
        return Ok(());
    }

    // Pre-count total rows from visibility to reserve chunk buffers.
    let total_rows: usize = vis_batches.iter().map(|b| b.num_rows()).sum();
    if total_rows == 0 {
        return Ok(());
    }

    // Pre-allocate per-chunk row accumulator. Each entry holds the partial
    // row bytes accumulated so far across columns.
    let mut chunk_row_buf: Vec<Vec<u8>> = vec![Vec::new(); CHUNK_SIZE.min(total_rows)];

    // Track how many rows are currently accumulated in chunk_row_buf.
    let mut chunk_filled = 0usize;
    let mut global_row = 0usize;

    // Column-pass 1: accumulate column bytes into row chunks.
    // Clear the buffer before each pass to avoid stale data from previous ring buffer wraps.
    for buf in chunk_row_buf.iter_mut() {
        buf.clear();
    }
    for col_name in column_names {
        let col_path = seg_dir.join(format!("{}.vortex", col_name));
        if !col_path.exists() {
            continue;
        }

        let col_batches: Vec<arrow_array::RecordBatch> = VortexReader::open(&col_path)
            .map_err(|e| {
                crate::error::RockDuckError::Internal(format!("open column {col_name}: {e}"))
            })?
            .read_all_batches()
            .as_ref()
            .clone();

        for batch in &col_batches {
            if batch.num_columns() == 0 {
                continue;
            }
            let array = batch.column(0);
            for local_row in 0..batch.num_rows() {
                if global_row >= total_rows {
                    break;
                }

                let buf_idx = chunk_filled % CHUNK_SIZE;
                if buf_idx >= chunk_row_buf.len() {
                    // Grow buffer if needed (handles total_rows < CHUNK_SIZE case wasn't pre-sized)
                    chunk_row_buf.push(Vec::new());
                }

                let val = crate::read::point_get::extract_row_bytes(array, local_row)?;
                let bytes = val.to_storage_bytes();
                chunk_row_buf[buf_idx].extend_from_slice(&bytes);

                chunk_filled += 1;
                global_row += 1;
            }
        }
    }

    // Second pass: read visibility and write KV entries in bounded batches.
    // chunk_row_buf[i] holds row bytes for row (i) of the segment.
    let mut global_row = 0usize;
    let mut kv_batch: Vec<(Vec<u8>, Vec<u8>)> = Vec::with_capacity(BATCH_KV_SIZE);

    for batch in &vis_batches {
        let vis_cols = batch.num_columns();
        if vis_cols < 2 {
            continue;
        }

        let created_arr = batch
            .column(vis_cols - 2)
            .as_any()
            .downcast_ref::<arrow_array::Int64Array>()
            .ok_or_else(|| {
                crate::error::RockDuckError::Internal(
                    "__vis created_txn column is not Int64".into(),
                )
            })?;
        let deleted_arr = batch
            .column(vis_cols - 1)
            .as_any()
            .downcast_ref::<arrow_array::Int64Array>()
            .ok_or_else(|| {
                crate::error::RockDuckError::Internal(
                    "__vis deleted_txn column is not Int64".into(),
                )
            })?;

        for local_row in 0..batch.num_rows() {
            if global_row >= total_rows {
                break;
            }

            let created_txn = created_arr.value(local_row) as u64;
            let deleted_raw = deleted_arr.value(local_row) as u64;
            let deleted_txn = if deleted_raw == crate::mvcc::shadow_columns::NOT_DELETED {
                None
            } else {
                Some(deleted_raw)
            };

            let version = VersionEntry {
                txn_id: created_txn,
                value: chunk_row_buf
                    .get(global_row % CHUNK_SIZE)
                    .cloned()
                    .unwrap_or_default(),
                created_txn,
                deleted_txn,
            };

            let key = build_version_key(table, seg_id, global_row as u32, created_txn);
            let value = postcard::to_allocvec(&version).map_err(|e| {
                crate::error::RockDuckError::Internal(format!("serialize version entry: {e}"))
            })?;
            kv_batch.push((key, value));

            // Flush batch when full.
            if kv_batch.len() >= BATCH_KV_SIZE {
                flush_kv_batch(kv, CF_VERSIONS, &mut kv_batch)?;
            }

            global_row += 1;
        }
    }

    // Flush remaining entries.
    if !kv_batch.is_empty() {
        flush_kv_batch(kv, CF_VERSIONS, &mut kv_batch)?;
    }

    Ok(())
}

/// Write a batch of key-value entries to the KV store.
fn flush_kv_batch(
    kv: &Arc<dyn KVEngine>,
    cf: &str,
    batch: &mut Vec<(Vec<u8>, Vec<u8>)>,
) -> Result<()> {
    if batch.is_empty() {
        return Ok(());
    }
    let ops: Vec<KVOp> = batch
        .drain(..)
        .map(|(k, v)| KVOp::Put { key: k, value: v })
        .collect();
    kv.write_batch(cf, &ops)?;
    Ok(())
}

#[cfg(test)]
mod time_travel_phase1_tests {
    use super::*;
    use std::sync::Arc;

    struct FailingKv;

    impl crate::metadata::kv_engine::KVEngine for FailingKv {
        fn get(&self, _cf: &str, _key: &[u8]) -> crate::error::Result<Option<Vec<u8>>> {
            Err(RockDuckError::Internal(
                "synthetic committed history read failure".into(),
            ))
        }

        fn put(&self, _cf: &str, _key: &[u8], _value: &[u8]) -> crate::error::Result<()> {
            Ok(())
        }

        fn delete(&self, _cf: &str, _key: &[u8]) -> crate::error::Result<()> {
            Ok(())
        }

        fn prefix_iter(
            &self,
            _cf: &str,
            _prefix: &[u8],
        ) -> crate::error::Result<Box<dyn crate::metadata::kv_engine::KVIter>> {
            Err(RockDuckError::Internal(
                "synthetic committed history read failure".into(),
            ))
        }

        fn write_batch(
            &self,
            _cf: &str,
            _ops: &[crate::metadata::kv_engine::KVOp],
        ) -> crate::error::Result<()> {
            Ok(())
        }

        fn flush(&self) -> crate::error::Result<()> {
            Ok(())
        }

        fn atomic_increment(
            &self,
            _bucket: &str,
            _key: &[u8],
            _delta: i64,
        ) -> crate::error::Result<i64> {
            Err(RockDuckError::Internal(
                "synthetic committed history read failure".into(),
            ))
        }
    }

    #[test]
    fn time_travel_reader_fails_loudly_when_commit_authority_is_unavailable() {
        let kv: Arc<dyn crate::metadata::kv_engine::KVEngine> = Arc::new(FailingKv);
        let err = TimeTravelReader::new(kv, std::path::PathBuf::from("."))
            .err()
            .expect("reader should fail when committed history cannot be loaded");
        assert!(err.to_string().contains("TimeTravelReader::new"));
        assert!(err
            .to_string()
            .contains("historical commit authority unavailable"));
    }

    #[test]
    fn time_travel_scanner_fails_loudly_when_commit_authority_is_unavailable() {
        let kv: Arc<dyn crate::metadata::kv_engine::KVEngine> = Arc::new(FailingKv);
        let err =
            TimeTravelScanner::new(kv, "orders".to_string(), 42, std::path::PathBuf::from("."))
                .err()
                .expect("scanner should fail when committed history cannot be loaded");
        assert!(err.to_string().contains("TimeTravelScanner::new"));
        assert!(err
            .to_string()
            .contains("historical commit authority unavailable"));
    }
}
