//! PDT Positional Merge implementation
//!
//! # Rewrite Framework Summary
//!
//! This module is the canonical **physical rewrite engine** for RockDuck.
//! The only fully implemented physical rewrite action is PDT merge (filter deleted rows).
//!
//! ## Rewrite Action Inventory
//!
//! | Action | File | Status | Budget Model | Truth Impact |
//! |---------|------|--------|-------------|-------------|
//! | PDT merge | `pdt_merge.rs` | **Production** | `min_del_ratio` threshold | Increases recovery complexity (new segment) |
//! | Small file merge | `nonblocking.rs` | **Prod** | No separate budget | Delegates to PDT merge |
//! | Query-driven | `nonblocking.rs` | **Stub** | None | N/A (not wired) |
//!
//! Phase 4 closure note:
//! - maintenance outcomes now carry an explicit `CompactionQualitySignalKind` so governance can
//!   distinguish measured delete-resolution evidence from bootstrap proxy callback signals.
//!
//! ## Key Properties
//!
//! - PDT merge is the **only** production rewrite that changes physical layout.
//! - It does NOT increase recovery complexity beyond new segment registration.
//! - `min_del_ratio` is the **only explicit budget gate** before a rewrite is dispatched.
//! - All other signals (staleness, miss penalty, age, size) are **soft** — they affect
//!   priority ordering, not whether a rewrite fires.

use crate::error::Result;
use crate::error::RockDuckError;
use crate::metadata::block_zone_map::GranuleBloomFilterManager;
use crate::mvcc::shadow_columns as sc;
use crate::segment::{
    overlay::compaction_overlay_filter, SegmentLayout, SegmentMeta, SegmentStatus,
};
use crate::storage::vortex::{VortexReader, VortexWriter};
use arrow_array::cast::AsArray;
use arrow_array::RecordBatch;
use arrow_select::filter::filter_record_batch;
use std::fs;
use std::path::Path;
use std::sync::Arc;

/// Progress callback type
pub type ProgressCallback = Box<dyn Fn(f32) + Send + Sync>;

/// PDT Merge configuration
#[derive(Debug, Clone)]
pub struct PdtMergeConfig {
    pub min_del_ratio: f64,
    pub target_granule_rows: u32,
    pub parallel_io: bool,
    pub parallelism: usize,
}

impl Default for PdtMergeConfig {
    fn default() -> Self {
        Self {
            min_del_ratio: 0.05,
            target_granule_rows: 1024 * 1024,
            parallel_io: true,
            parallelism: num_cpus::get(),
        }
    }
}

/// PDT Merge results
#[derive(Debug, Clone, Default)]
pub struct MergeStats {
    pub rows_read: u64,
    pub rows_written: u64,
    pub rows_dropped: u64,
    pub bytes_read: u64,
    pub bytes_written: u64,
    pub granules_created: u32,
}

/// Per-column writer: accumulates RecordBatches for one column, then writes and finishes
/// in a single call. This avoids the O(n) overhead of creating and finishing a new
/// VortexWriter for each batch.
struct PerColWriter {
    col_name: String,
    path: std::path::PathBuf,
    batches: Vec<RecordBatch>,
}

impl PerColWriter {
    fn finish(mut self) -> Result<()> {
        if self.batches.is_empty() {
            return Ok(());
        }
        let mut writer = VortexWriter::create(&self.path, self.col_name.clone());
        for batch in self.batches.drain(..) {
            writer.write(batch)?;
        }
        writer.finish()?;
        Ok(())
    }
}

/// Compact a single segment: read column files, filter deleted rows, write new segment.
/// Outputs a new segment with `has_visibility_columns: true`.
pub fn compact_segment(
    data_dir: &Path,
    old_seg_id: &str,
    old_meta: &SegmentMeta,
    config: &PdtMergeConfig,
    progress: Option<ProgressCallback>,
) -> Result<(SegmentMeta, MergeStats)> {
    let old_layout = SegmentLayout::new(data_dir, old_seg_id);

    if !old_layout.seg_dir.exists() {
        return Err(RockDuckError::Compaction(format!(
            "Segment directory not found: {}",
            old_layout.seg_dir.display()
        )));
    }

    let mut stats = MergeStats::default();
    let vis_path = old_layout.vis_path();
    let vis_exists = vis_path.exists();

    let total_rows = old_meta.row_count;
    let alive_rows = old_meta.alive_row_count;

    // Compute actual delete ratio instead of always returning true.
    // If there are no rows, skip compaction (no work to do).
    let del_ratio_threshold_exceeded = if total_rows == 0 {
        false
    } else {
        let dead_rows = total_rows.saturating_sub(alive_rows);
        let del_ratio = dead_rows as f64 / total_rows as f64;
        del_ratio >= config.min_del_ratio
    };

    stats.rows_read = total_rows;
    stats.rows_dropped = total_rows.saturating_sub(alive_rows);

    if !del_ratio_threshold_exceeded {
        return Err(RockDuckError::Compaction(format!(
            "del_ratio < threshold {}",
            config.min_del_ratio
        )));
    }

    if let Some(cb) = &progress {
        cb(0.1);
    }

    let new_seg_id = format!("{}_c{}", old_seg_id, uuid::Uuid::new_v4());
    let new_layout = SegmentLayout::new(data_dir, &new_seg_id);
    fs::create_dir_all(&new_layout.seg_dir)?;

    if let Some(cb) = &progress {
        cb(0.2);
    }

    // Read visibility batches if available
    let old_vis_batches: Vec<RecordBatch> = if vis_exists {
        match VortexReader::open(&vis_path) {
            Ok(r) => {
                let batches: Vec<RecordBatch> = (*r.read_all_batches()).clone();
                if batches.is_empty() && vis_path.exists() {
                    tracing::warn!(
                        "Vortex VIS file '{}' opened but returned no batches for seg {}. \
                         File may be corrupted or empty.",
                        vis_path.display(),
                        old_seg_id
                    );
                }
                batches
            }
            Err(e) => {
                tracing::warn!("Vortex VIS file open failed for seg {}: {}", old_seg_id, e);
                Vec::new()
            }
        }
    } else {
        Vec::new()
    };

    // Collect all user column batches per batch index
    let mut col_data: Vec<(String, Vec<RecordBatch>)> = Vec::new();
    for col_def in &old_meta.columns {
        let old_path = old_layout.seg_dir.join(format!("{}.vortex", col_def.name));
        if old_path.exists() {
            let batches = VortexReader::open(&old_path)
                .ok()
                .map(|r| r.read_all_batches())
                .map(|a| (*a).clone())
                .unwrap_or_default();
            if !batches.is_empty() {
                col_data.push((col_def.name.clone(), batches));
            }
        }
    }

    if col_data.is_empty() {
        return Err(RockDuckError::Compaction(
            "No column data found".to_string(),
        ));
    }

    let num_batch_indices = col_data.iter().map(|(_, b)| b.len()).max().unwrap_or(0);

    // m001 fix: Build bloom filters for ALL columns during compaction.
    // The data is already decompressed (compaction reads it to filter deleted rows).
    // We create one GranuleBloomFilterManager per column, insert PKs from each batch,
    // then save to .{col}._bf.bin sidecar files.
    // The PK is assumed to be the first column (id column). If no PK is defined,
    // we use column 0 as a proxy for uniqueness.
    #[allow(unused_variables)]
    let mut bloom_managers: Vec<(String, GranuleBloomFilterManager)> = Vec::new();
    for (col_name, _) in &col_data {
        let manager = GranuleBloomFilterManager::new(
            new_seg_id.clone(),
            col_name.clone(),
            alive_rows,
        );
        bloom_managers.push((col_name.clone(), manager));
    }

    // 3.5 fix: build the compaction visibility filter once outside the batch loop,
    // instead of calling compaction_overlay_filter() inside the loop for every batch.
    let (overlay, vis_filter) = compaction_overlay_filter();

    let mut new_vis_batches: Vec<RecordBatch> = Vec::new();
    let mut column_writers: Option<Vec<PerColWriter>> = None;

    for batch_idx in 0..num_batch_indices {
        let batch_n = col_data
            .first()
            .and_then(|(_, b)| b.get(batch_idx))
            .map(|b| b.num_rows())
            .unwrap_or(0);
        if batch_n == 0 {
            continue;
        }

        // Determine which rows are alive based on visibility columns
        let mut alive_mask: Vec<bool> = Vec::with_capacity(batch_n);
        let old_vis = old_vis_batches.get(batch_idx);

        for row_i in 0..batch_n {
            let alive = if let Some(vis_batch) = old_vis {
                if sc::has_visibility_columns(&vis_batch.schema()) {
                    let (_, deleted_arr) = sc::extract_visibility_columns(vis_batch);
                    let d_raw = deleted_arr
                        .as_primitive::<arrow_array::types::Int64Type>()
                        .value(row_i) as u64;
                    let d = if d_raw == sc::NOT_DELETED {
                        None
                    } else {
                        Some(d_raw)
                    };
                    let c = vis_batch
                        .column(0)
                        .as_primitive::<arrow_array::types::Int64Type>()
                        .value(row_i) as u64;
                    overlay.is_row_visible(&vis_filter, c, d)
                } else {
                    true
                }
            } else {
                true
            };
            alive_mask.push(alive);
        }

        if alive_mask.iter().all(|&a| !a) {
            continue;
        }

        // Build compacted visibility batch
        let vis_n = alive_mask.iter().filter(|&&a| a).count();
        let vis_schema = sc::visibility_schema();
        let mut new_c = arrow_array::Int64Array::builder(vis_n);
        let mut new_d = arrow_array::Int64Array::builder(vis_n);

        if let Some(vis_batch) = old_vis {
            let c_arr = vis_batch
                .column(0)
                .as_primitive::<arrow_array::types::Int64Type>();
            let d_arr = vis_batch
                .column(1)
                .as_primitive::<arrow_array::types::Int64Type>();
            for (row_i, &alive) in alive_mask.iter().enumerate() {
                if alive {
                    new_c.append_value(c_arr.value(row_i));
                    new_d.append_value(d_arr.value(row_i));
                }
            }
        } else {
            // Old segment without vis columns: fill with 0 / NOT_DELETED
            for &alive in &alive_mask {
                if alive {
                    new_c.append_value(0i64);
                    new_d.append_value(sc::NOT_DELETED as i64);
                }
            }
        }

        let vis_batch = RecordBatch::try_new(
            vis_schema,
            vec![Arc::new(new_c.finish()), Arc::new(new_d.finish())],
        )
        .map_err(|e| RockDuckError::Compaction(format!("vis batch create: {}", e)))?;
        new_vis_batches.push(vis_batch);

        // Write compacted user data batches using per-column writers.
        // Each writer accumulates all its batches then finishes once — avoids
        // the overhead of creating/finishing N writers per column.
        for (col_idx, (col_name, batches)) in col_data.iter().enumerate() {
            if let Some(batch) = batches.get(batch_idx) {
                let predicate = arrow_array::BooleanArray::from(alive_mask.clone());
                if let Ok(filtered) = filter_record_batch(batch, &predicate) {
                    if filtered.num_rows() > 0 {
                        // m001 fix: Build bloom filter from the compacted (alive) PK values.
                        // The PK is assumed to be the first column (column index 0).
                        // For the PK column (col_idx == 0), insert all PK values into the bloom filter.
                        if col_idx == 0 {
                            let granule_idx = batch_idx as u32;
                            // For bloom filter, we just need PK values — offset within batch doesn't affect
                            // the bloom filter (it's per-granule). We pass 0 as offset.
                            for (manager_col_name, manager) in &mut bloom_managers {
                                if manager_col_name == col_name {
                                    manager.build_for_granule_from_array(granule_idx, 0, filtered.column(0).as_ref())?;
                                }
                            }
                        }

                        let writers = column_writers.get_or_insert_with(Vec::new);
                        if writers.len() <= col_idx {
                            writers.push(PerColWriter {
                                col_name: col_name.clone(),
                                path: new_layout.seg_dir.join(format!("{}.vortex", col_name)),
                                batches: Vec::new(),
                            });
                        }
                        writers[col_idx].batches.push(filtered);
                    }
                }
            }
        }
    }

    // Finish all per-column writers.
    if let Some(writers) = column_writers {
        for writer in writers {
            writer.finish()?;
        }
    }

    // m001 fix: Save bloom filters to .{col}._bf.bin sidecar files.
    // Bloom filters were built during the compaction loop from decompressed column data.
    // Saving happens after all data is written, before the fsync.
    for (col_name, mut manager) in bloom_managers {
        if !manager.is_empty() {
            let bf_path = new_layout.seg_dir.join(format!("{}._bf.bin", col_name));
            manager.save(&bf_path).map_err(|e| {
                std::io::Error::other(
                    format!(
                        "compaction: failed to save bloom filter for column '{}' in seg {}: {}",
                        col_name, new_seg_id, e
                    ),
                )
            })?;
            tracing::debug!(
                "compaction: saved bloom filter for column '{}' in seg {}",
                col_name, new_seg_id
            );
        }
    }

    // Write __vis.vortex for compacted segment
    if !new_vis_batches.is_empty() {
        let _vis_schema = sc::visibility_schema();
        let mut vis_writer = VortexWriter::create(new_layout.vis_path(), "__vis");
        for vb in &new_vis_batches {
            vis_writer.write(vb.clone())?;
        }
        vis_writer.finish()?;
    }

    if let Some(seg_dir) = new_layout.vis_path().parent() {
        if let Ok(dir) = std::fs::OpenOptions::new().read(true).open(seg_dir) {
            dir.sync_all().map_err(|e| {
                RockDuckError::Write(format!("pdt merge dir fsync {}: {}", seg_dir.display(), e))
            })?;
        }
    }

    if let Some(cb) = &progress {
        cb(0.8);
    }

    // Estimate bytes_written from column files
    let mut bytes_written: u64 = 0;
    for col_def in &old_meta.columns {
        let new_path = new_layout.seg_dir.join(format!("{}.vortex", col_def.name));
        if new_path.exists() {
            bytes_written += fs::metadata(&new_path).map(|m| m.len()).unwrap_or(0);
        }
    }
    stats.bytes_written = bytes_written;

    let new_meta = SegmentMeta {
        seg_id: new_seg_id.clone(),
        table_id: old_meta.table_id.clone(),
        status: SegmentStatus::Frozen,
        seg_type: old_meta.seg_type,
        columns: old_meta.columns.clone(),
        min_key: old_meta.min_key.clone(),
        max_key: old_meta.max_key.clone(),
        row_count: alive_rows,
        alive_row_count: alive_rows,
        del_ratio: 0.0,
        size_bytes: bytes_written,
        created_txn: old_meta.created_txn,
        updated_txn: old_meta.updated_txn,
        updated_at: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_micros() as u64)
            .unwrap_or(0),
        file_paths: vec![],
        granules: old_meta.granules.clone(),
        has_visibility_columns: true,
        delta_file_id: None,
        delta_row_count: 0,
        delta_l1_bytes: 0,
    };

    stats.rows_written = alive_rows;
    stats.granules_created = 1;

    if let Some(cb) = &progress {
        cb(1.0);
    }

    Ok((new_meta, stats))
}

/// Multi-way merge: merge multiple segments into one with a single pass.
///
/// Reads all segments' column data once (O(N) I/O per segment) and produces
/// a single compacted output segment, instead of calling `compact_segment` N times
/// (which would re-read the same data for every segment).
///
/// Assumes all segments share the same schema (column names). If segment schemas
/// diverge, the last segment's column definitions are used for the output.
pub fn multiway_merge(
    data_dir: &Path,
    seg_ids: &[String],
    metas: &[SegmentMeta],
    config: &PdtMergeConfig,
) -> Result<(SegmentMeta, MergeStats)> {
    if seg_ids.is_empty() {
        return Err(RockDuckError::Compaction(
            "No segments to merge".to_string(),
        ));
    }
    if seg_ids.len() == 1 {
        return compact_segment(data_dir, &seg_ids[0], &metas[0], config, None);
    }

    // --- Accumulate total stats and verify all segments exist ---
    let mut total_stats = MergeStats::default();
    let mut total_rows: u64 = 0;
    let mut total_alive: u64 = 0;
    #[allow(unused_variables, unused_mut)]
    let total_bytes: u64 = 0;

    for (seg_id, meta) in seg_ids.iter().zip(metas.iter()) {
        let old_layout = SegmentLayout::new(data_dir, seg_id);
        if !old_layout.seg_dir.exists() {
            return Err(RockDuckError::Compaction(format!(
                "Segment directory not found: {}",
                old_layout.seg_dir.display()
            )));
        }

        total_rows += meta.row_count;
        let dead_rows = meta.row_count.saturating_sub(meta.alive_row_count);
        total_alive += meta.alive_row_count;
    let mut total_bytes: u64 = 0;
    {
        #[allow(unused_assignments)]
        let total_bytes = &mut total_bytes;
        *total_bytes += meta.size_bytes;
    }
        total_stats.rows_read += meta.row_count;
        total_stats.rows_dropped += dead_rows;
    }

    let first_meta = metas.first().ok_or_else(|| {
        RockDuckError::Compaction(format!(
            "No segments to merge (attempted {} segments)",
            seg_ids.len()
        ))
    })?;

    // --- Build per-segment column readers and visibility readers ---
    #[allow(dead_code)]
    struct SegmentReaders {
        vis_batches: Vec<RecordBatch>,
        col_batches: Vec<(String, Vec<RecordBatch>)>,
    }

    let mut seg_readers: Vec<SegmentReaders> = Vec::with_capacity(seg_ids.len());
    for (seg_id, meta) in seg_ids.iter().zip(metas.iter()) {
        let old_layout = SegmentLayout::new(data_dir, seg_id);

        // Visibility batches.
        let vis_path = old_layout.vis_path();
        let vis_batches: Vec<RecordBatch> = if vis_path.exists() {
            VortexReader::open(&vis_path)
                .ok()
                .map(|r| (*r.read_all_batches()).clone())
                .unwrap_or_default()
        } else {
            Vec::new()
        };

        // Per-column batches.
        let mut col_batches: Vec<(String, Vec<RecordBatch>)> = Vec::new();
        for col_def in &meta.columns {
            let col_path = old_layout.seg_dir.join(format!("{}.vortex", col_def.name));
            if col_path.exists() {
                if let Ok(r) = VortexReader::open(&col_path) {
                    col_batches.push((col_def.name.clone(), (*r.read_all_batches()).clone()));
                }
            }
        }

        seg_readers.push(SegmentReaders {
            vis_batches,
            col_batches,
        });
    }

    // Determine max batch index across all segments.
    let max_batch_idx = seg_readers
        .iter()
        .map(|s| {
            s.col_batches
                .iter()
                .map(|(_, b)| b.len())
                .max()
                .unwrap_or(0)
        })
        .max()
        .unwrap_or(0);

    if max_batch_idx == 0 {
        return Err(RockDuckError::Compaction(
            "No column data found across segments".to_string(),
        ));
    }

    // --- Precompute overlay once for all batches ---
    let (overlay, vis_filter) = compaction_overlay_filter();

    // --- Create output segment ---
    let new_seg_id = format!(
        "merge_{}_{}",
        seg_ids.first().unwrap(),
        uuid::Uuid::new_v4()
    );
    let new_layout = SegmentLayout::new(data_dir, &new_seg_id);
    fs::create_dir_all(&new_layout.seg_dir)?;

    // Collect column names from first segment (assumed consistent across all).
    let col_names: Vec<String> = first_meta
        .columns
        .iter()
        .map(|c| c.name.clone())
        .collect();

    // --- Single-pass: iterate batches once, merge across all segments per batch index ---
    let mut new_vis_batches: Vec<RecordBatch> = Vec::new();

    for batch_idx in 0..max_batch_idx {
        // Compute total alive rows across all segments for this batch index.
        let mut total_alive_this_batch: usize = 0;

        for seg in &seg_readers {
            let vis_batches = &seg.vis_batches;
            let batch_n = seg
                .col_batches
                .first()
                .and_then(|(_, b)| b.get(batch_idx))
                .map(|b| b.num_rows())
                .unwrap_or(0);

            let alive_n = if batch_n == 0 {
                0
            } else {
                let old_vis = vis_batches.get(batch_idx);
                let mut alive = 0usize;
                for row_i in 0..batch_n {
                    let visible = if let Some(vis_batch) = old_vis {
                        if sc::has_visibility_columns(&vis_batch.schema()) {
                            let (_, deleted_arr) = sc::extract_visibility_columns(vis_batch);
                            let d_raw = deleted_arr
                                .as_primitive::<arrow_array::types::Int64Type>()
                                .value(row_i) as u64;
                            let d = if d_raw == sc::NOT_DELETED {
                                None
                            } else {
                                Some(d_raw)
                            };
                            let c = vis_batch
                                .column(0)
                                .as_primitive::<arrow_array::types::Int64Type>()
                                .value(row_i) as u64;
                            overlay.is_row_visible(&vis_filter, c, d)
                        } else {
                            true
                        }
                    } else {
                        true
                    };
                    if visible {
                        alive += 1;
                    }
                }
                alive
            };

            total_alive_this_batch += alive_n;
        }

        if total_alive_this_batch == 0 {
            continue;
        }

        // Build compacted visibility batch.
        let vis_schema = sc::visibility_schema();
        let mut new_c = arrow_array::Int64Array::builder(total_alive_this_batch);
        let mut new_d = arrow_array::Int64Array::builder(total_alive_this_batch);

        for seg in &seg_readers {
            let vis_batches = &seg.vis_batches;
            let batch_n = seg
                .col_batches
                .first()
                .and_then(|(_, b)| b.get(batch_idx))
                .map(|b| b.num_rows())
                .unwrap_or(0);

            if batch_n == 0 {
                continue;
            }

            let old_vis = vis_batches.get(batch_idx);
            for row_i in 0..batch_n {
                let visible = if let Some(vis_batch) = old_vis {
                    if sc::has_visibility_columns(&vis_batch.schema()) {
                        let (_, deleted_arr) = sc::extract_visibility_columns(vis_batch);
                        let d_raw = deleted_arr
                            .as_primitive::<arrow_array::types::Int64Type>()
                            .value(row_i) as u64;
                        let d = if d_raw == sc::NOT_DELETED {
                            None
                        } else {
                            Some(d_raw)
                        };
                        let c = vis_batch
                            .column(0)
                            .as_primitive::<arrow_array::types::Int64Type>()
                            .value(row_i) as u64;
                        overlay.is_row_visible(&vis_filter, c, d)
                    } else {
                        true
                    }
                } else {
                    true
                };

                if visible {
                    if let Some(vis_batch) = old_vis {
                        let c_arr = vis_batch
                            .column(0)
                            .as_primitive::<arrow_array::types::Int64Type>();
                        let d_arr = vis_batch
                            .column(1)
                            .as_primitive::<arrow_array::types::Int64Type>();
                        new_c.append_value(c_arr.value(row_i));
                        new_d.append_value(d_arr.value(row_i));
                    } else {
                        new_c.append_value(0i64);
                        new_d.append_value(sc::NOT_DELETED as i64);
                    }
                }
            }
        }

        let vis_batch = RecordBatch::try_new(
            vis_schema,
            vec![Arc::new(new_c.finish()), Arc::new(new_d.finish())],
        )
        .map_err(|e| RockDuckError::Compaction(format!("vis batch create: {}", e)))?;
        new_vis_batches.push(vis_batch);
    }

    // Write compacted visibility.
    if !new_vis_batches.is_empty() {
        let mut vis_writer = VortexWriter::create(new_layout.vis_path(), "__vis");
        for vb in &new_vis_batches {
            vis_writer.write(vb.clone())?;
        }
        vis_writer.finish()?;
    }

    // Per-column writers: accumulate all batches for each column, then write+finish once.
    let mut col_writers: Vec<PerColWriter> = col_names
        .iter()
        .map(|col_name| PerColWriter {
            col_name: col_name.clone(),
            path: new_layout.seg_dir.join(format!("{}.vortex", col_name)),
            batches: Vec::new(),
        })
        .collect();

    for batch_idx in 0..max_batch_idx {
        // === Precompute visibility once per batch (hoisted out of col_idx loop) ===
        let mut all_alive_mask: Vec<bool> = Vec::new();

        for seg in &seg_readers {
            let vis_batches = &seg.vis_batches;
            let batch_n = seg
                .col_batches
                .first()
                .and_then(|(_, b)| b.get(batch_idx))
                .map(|b| b.num_rows())
                .unwrap_or(0);

            // Extract visibility columns once per vis_batch (not per row)
            let vis_arrays = if let Some(vis_batch) = vis_batches.get(batch_idx) {
                if sc::has_visibility_columns(&vis_batch.schema()) {
                    let (c_arr, d_arr) = sc::extract_visibility_columns(vis_batch);
                    Some((c_arr, d_arr))
                } else {
                    None
                }
            } else {
                None
            };

            for row_i in 0..batch_n {
                let visible = if let Some((c_arr, d_arr)) = &vis_arrays {
                    let d_raw = d_arr
                        .as_primitive::<arrow_array::types::Int64Type>()
                        .value(row_i) as u64;
                    let d = if d_raw == sc::NOT_DELETED { None } else { Some(d_raw) };
                    let c = c_arr
                        .as_primitive::<arrow_array::types::Int64Type>()
                        .value(row_i) as u64;
                    overlay.is_row_visible(&vis_filter, c, d)
                } else {
                    true
                };
                all_alive_mask.push(visible);
            }
        }

        if all_alive_mask.is_empty() || all_alive_mask.iter().all(|&a| !a) {
            continue;
        }

        let predicate = arrow_array::BooleanArray::from(all_alive_mask);

        // === col_idx loop only does filter (visibility already computed above) ===
        for (col_idx, col_name) in col_names.iter().enumerate() {
            let predicate_clone = predicate.clone();
            for seg in &seg_readers {
                if let Some((_, batches)) = seg.col_batches.iter().find(|(n, _)| n == col_name) {
                    if let Some(batch) = batches.get(batch_idx) {
                        if batch.num_rows() > 0 {
                        if let Ok(filtered) = filter_record_batch(batch, &predicate_clone) {
                            if filtered.num_rows() > 0 {
                                let filtered_rows = filtered.num_rows();
                                col_writers[col_idx].batches.push(filtered);
                                total_stats.rows_written += filtered_rows as u64;
                            }
                        }
                        }
                    }
                }
            }
        }
    }

    for writer in col_writers {
        writer.finish()?;
    }

    if let Some(seg_dir) = new_layout.vis_path().parent() {
        if let Ok(dir) = std::fs::OpenOptions::new().read(true).open(seg_dir) {
            dir.sync_all().map_err(|e| {
                RockDuckError::Write(format!("multiway_merge dir fsync {}: {}", seg_dir.display(), e))
            })?;
        }
    }

    // Compute bytes_written.
    let mut bytes_written: u64 = 0;
    for col_name in &col_names {
        let path = new_layout.seg_dir.join(format!("{}.vortex", col_name));
        if path.exists() {
            bytes_written += fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
        }
    }
    total_stats.bytes_written = bytes_written;
    total_stats.granules_created = 1;

    // Build output metadata.
    let new_meta = SegmentMeta {
        seg_id: new_seg_id.clone(),
        table_id: first_meta.table_id.clone(),
        status: SegmentStatus::Frozen,
        seg_type: first_meta.seg_type,
        columns: first_meta.columns.clone(),
        min_key: first_meta.min_key.clone(),
        max_key: first_meta.max_key.clone(),
        row_count: total_alive,
        alive_row_count: total_alive,
        del_ratio: if total_rows > 0 {
            (total_rows - total_alive) as f64 / total_rows as f64
        } else {
            0.0
        },
        size_bytes: bytes_written,
        created_txn: metas.iter().map(|m| m.created_txn).min().unwrap_or(0),
        updated_txn: metas.iter().map(|m| m.updated_txn).max().unwrap_or(0),
        updated_at: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_micros() as u64)
            .unwrap_or(0),
        file_paths: vec![],
        granules: first_meta.granules.clone(),
        has_visibility_columns: true,
        delta_file_id: None,
        delta_row_count: 0,
        delta_l1_bytes: 0,
    };

    Ok((new_meta, total_stats))
}
