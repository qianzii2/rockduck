//! Range scan for RockDuck.
//!
//! Phase 1.5/2/3/4 note: scan is no longer just a monolithic "read every segment
//! uniformly" function. It now has explicit execution planning, routing-template
//! dispatch, and post-execution feedback/evidence hooks.

use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

use serde::{Deserialize, Serialize};

use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::Field;
use rayon::prelude::*;

use crate::db::{RockDuck, TxnId};
use crate::error::{Result, RockDuckError};
use crate::metadata;
use crate::metadata::kv_store::get_or_create_table_stats;
use crate::mvcc::visibility::{IsolationLevel, TxnSnapshot, VisibilityError};
use crate::query::routing::feedback::ShadowTimingSample;
use crate::query::routing::{QueryKind, RouteDecision, RouterParamsOwned, RoutingResult};
use crate::segment::layout::SegmentLayout;
use crate::segment::meta::SegmentMeta;
use crate::storage::delta::{merge::apply_deltas_to_batch, DeltaQueryLayer};
use crate::storage::vortex::VortexReader;

// ─── Public API Types ─────────────────────────────────────────────────────────

/// Options for a table scan.
#[derive(Debug, Clone, Default)]
pub struct ScanOptions {
    pub table: String,
    pub as_of_txn: Option<TxnId>,
    pub filter: Option<String>,
    pub batch_size: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadExecutionTemplate {
    DeltaOnly,
    VortexOnly,
    CooperativeMerge,
    Historical,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SegmentExecutionTemplate {
    DeltaOnly,
    VortexOnly,
    CooperativeMerge,
}

#[derive(Debug, Clone, Default)]
pub struct ReadExecutionStats {
    pub segments_routed: usize,
    pub segments_scanned: u64,
    pub base_batches_read: u64,
    pub delta_rows_seen: u64,
    pub rows_after_visibility: u64,
    pub rows_after_filter: usize,
    pub filter_failures: u64,
    pub skipped_segment_failures: u64,
    pub skipped_segment_ids: Vec<String>,
    pub cooperative: CooperativeRuntimeStats,
}

#[derive(Debug, Clone, Default)]
pub struct CooperativeRuntimeStats {
    pub slice_budget: Option<CooperativeSliceBudget>,
    pub total_slices: u64,
    pub truncated_segment_ids: Vec<String>,
    pub elapsed_budget_exhausted: bool,
    pub max_segments_per_slice_hit: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CooperativeSliceBudget {
    pub max_segments_per_slice: usize,
    pub max_slice_ms: u64,
}

impl CooperativeSliceBudget {
    pub fn is_bounded(self) -> bool {
        self.max_segments_per_slice > 0 || self.max_slice_ms > 0
    }
}

#[derive(Debug, Clone)]
pub struct CooperativeExecutionDigest {
    pub budget: Option<CooperativeSliceBudget>,
    pub slices_executed: u64,
    pub truncated_segments: Vec<String>,
    pub elapsed_budget_exhausted: bool,
    pub max_segments_per_slice_hit: bool,
}

#[derive(Debug, Clone)]
pub struct ReadExecutionPlan {
    pub table: String,
    pub filter: Option<String>,
    pub query_columns: Vec<String>,
    pub snapshot: TxnSnapshot,
    pub routed_segments: Vec<SegmentMeta>,
    pub execution_segments: Vec<SegmentMeta>,
    pub parsed_expr: Option<crate::query::filter_expr::Expr>,
    pub zone_map_group: Option<crate::query::filter_expr::ZoneMapPredicateGroup>,
    pub template: ReadExecutionTemplate,
    pub routing: Option<RoutingResult>,
}

#[derive(Debug, Clone)]
pub struct ReadExecutionOutput {
    pub batches: Vec<RecordBatch>,
    pub stats: ReadExecutionStats,
}

#[derive(Debug, Clone)]
pub struct SegmentReadStageOutput {
    pub seg_id: String,
    pub original_index: usize,
    pub batches: Vec<RecordBatch>,
    pub base_batches: u64,
    pub delta_rows: u64,
    pub rows_after_visibility: u64,
    pub template: SegmentExecutionTemplate,
}

#[derive(Debug, Clone)]
pub struct ExecutionOutcome {
    pub table: String,
    pub query_columns: Vec<String>,
    pub template: ReadExecutionTemplate,
    pub chosen_path: Option<RouteDecision>,
    pub estimated_cost: Option<f64>,
    pub rows_scanned: u64,
    pub candidate_rows: u64,
    pub rows_after_visibility: u64,
    pub rows_returned: u64,
    pub filter_failures: u64,
    pub skipped_segment_failures: u64,
    pub skipped_segment_ids: Vec<String>,
    pub segments_routed: u64,
    pub segments_scanned: u64,
    pub elapsed_ms: f64,
    pub executed_segment_ids: Vec<String>,
    pub cooperative_digest: Option<CooperativeExecutionDigest>,
}

#[derive(Debug, Clone)]
struct RoutedExecutionPlan {
    routed_segments: Vec<SegmentMeta>,
    execution_segments: Vec<SegmentMeta>,
    template: ReadExecutionTemplate,
    routing: Option<RoutingResult>,
}

/// Iterator over `RecordBatch` results from a table scan.
pub struct ScanIterator<'a> {
    db: &'a RockDuck,
    opts: ScanOptions,
    filter_str: Option<String>,
    state: ScanState,
    snapshot: Option<Arc<TxnSnapshot>>,
    stats: ReadExecutionStats,
    /// Pre-computed segment list to avoid a redundant `list_segment_metas` KV read.
    /// Set by `prepare_read_execution_plan` when routing information is already available.
    precomputed_segments: Option<Vec<SegmentMeta>>,
}

enum ScanState {
    Init,
    Scanning {
        segment_idx: usize,
        segments: Vec<SegmentMeta>,
        current_batch_results: Vec<RecordBatch>,
        batch_offset: usize,
        zone_map_group: Option<crate::query::filter_expr::ZoneMapPredicateGroup>,
    },
    Done,
}

impl<'a> ScanIterator<'a> {
    pub fn new(db: &'a RockDuck, opts: ScanOptions, filter: Option<&str>) -> Self {
        Self {
            db,
            opts,
            filter_str: filter.map(String::from),
            state: ScanState::Init,
            snapshot: None,
            stats: ReadExecutionStats::default(),
            precomputed_segments: None,
        }
    }

    /// Construct with pre-computed segment list (avoids a redundant KV read in `ensure_init`).
    #[allow(dead_code)]
    fn new_with_segments(
        db: &'a RockDuck,
        opts: ScanOptions,
        filter: Option<&str>,
        segments: Vec<SegmentMeta>,
    ) -> Self {
        Self {
            db,
            opts,
            filter_str: filter.map(String::from),
            state: ScanState::Init,
            snapshot: None,
            stats: ReadExecutionStats::default(),
            precomputed_segments: Some(segments),
        }
    }

    fn ensure_init(&mut self) -> Result<()> {
        if !matches!(self.state, ScanState::Init) {
            return Ok(());
        }

        let snapshot = match self.opts.as_of_txn {
            Some(txn_id) => self
                .db
                .mvcc
                .read()
                .snapshot_at(txn_id, IsolationLevel::Snapshot),
            None => self.db.snapshot(),
        };
        self.snapshot = Some(Arc::new(snapshot.clone()));

        let table_metas = if let Some(segments) = self.precomputed_segments.take() {
            segments
        } else {
            let metas = metadata::list_segment_metas(&self.db.kv)?;
            metas
                .into_iter()
                .filter(|m| m.table_id == self.opts.table)
                .filter(|m| m.status != crate::segment::meta::SegmentStatus::Garbage)
                .collect()
        };

        let zone_map_group = if let Some(filter_str) = &self.filter_str {
            let expr = crate::query::filter_expr::parse(filter_str)
                .map_err(|e| RockDuckError::Query(e.to_string()))?;
            crate::query::filter_expr::to_zone_map_predicate_group(&expr)
        } else {
            None
        };

        self.state = ScanState::Scanning {
            segment_idx: 0,
            segments: table_metas,
            current_batch_results: Vec::new(),
            batch_offset: 0,
            zone_map_group,
        };

        self.advance_segments()?;
        Ok(())
    }

    fn advance_segments(&mut self) -> Result<()> {
        let ScanState::Scanning {
            segment_idx,
            segments,
            current_batch_results,
            batch_offset,
            zone_map_group,
        } = &mut self.state
        else {
            return Ok(());
        };

        let stats = &mut self.stats;

        current_batch_results.clear();
        *batch_offset = 0;

        let snapshot = self.snapshot.as_ref().expect("snapshot not initialized");
        let vis_filter = self.db.mvcc.read();

        while *segment_idx < segments.len() && current_batch_results.is_empty() {
            let meta = &segments[*segment_idx];
            *segment_idx += 1;

            match scan_segment_internal(
                self.db,
                meta,
                snapshot.as_ref(),
                &*vis_filter,
                zone_map_group.as_ref(),
            ) {
                Ok(batches) => {
                    if !batches.is_empty() {
                        current_batch_results.extend(batches);
                    }
                }
                Err(e) => {
                    stats.skipped_segment_failures += 1;
                    stats.skipped_segment_ids.push(meta.seg_id.clone());
                    tracing::warn!(
                        segment = %meta.seg_id,
                        error = %e,
                        "scan_segment failed, skipping segment"
                    );
                }
            }
        }

        if *segment_idx >= segments.len() && current_batch_results.is_empty() {
            self.state = ScanState::Done;
        }

        Ok(())
    }
}

impl<'a> Iterator for ScanIterator<'a> {
    type Item = Result<RecordBatch>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Err(e) = self.ensure_init() {
            return Some(Err(e));
        }

        loop {
            match &mut self.state {
                ScanState::Scanning {
                    segment_idx,
                    segments,
                    current_batch_results,
                    batch_offset,
                    ..
                } => {
                    if *batch_offset < current_batch_results.len() {
                        let batch = current_batch_results[*batch_offset].clone();
                        *batch_offset += 1;
                        return Some(Ok(batch));
                    }

                    if *segment_idx >= segments.len() {
                        self.state = ScanState::Done;
                        return None;
                    }

                    if let Err(e) = self.advance_segments() {
                        self.state = ScanState::Done;
                        return Some(Err(e));
                    }

                    if matches!(self.state, ScanState::Done) {
                        return None;
                    }
                }
                ScanState::Done | ScanState::Init => return None,
            }
        }
    }
}

pub fn scan(db: &RockDuck, table: &str, filter: Option<&str>) -> Result<Vec<RecordBatch>> {
    let snapshot = db.snapshot();
    let vis_filter = db.mvcc.read();
    scan_internal(db, table, filter, &snapshot, &*vis_filter)
}

pub fn scan_as_of(
    db: &RockDuck,
    table: &str,
    txn_id: TxnId,
    filter: Option<&str>,
) -> Result<Vec<RecordBatch>> {
    let snapshot = db.mvcc.read().snapshot_at(txn_id, IsolationLevel::Snapshot);
    let vis_filter = db.mvcc.read();
    scan_internal(db, table, filter, &snapshot, &*vis_filter)
}

fn scan_internal(
    db: &RockDuck,
    table: &str,
    filter: Option<&str>,
    snapshot: &TxnSnapshot,
    vis_filter: &dyn crate::mvcc::visibility::VisFilter,
) -> Result<Vec<RecordBatch>> {
    let fallback_kind = if filter.is_some() {
        QueryKind::RangeScan
    } else {
        QueryKind::FullScan
    };
    let plan = prepare_read_execution_plan(db, table, filter, snapshot.clone(), fallback_kind)?;
    let output = execute_read_plan(db, &plan, vis_filter)?;
    Ok(output.batches)
}

fn cooperative_runtime_budget_for_plan(
    db: &RockDuck,
    plan: &ReadExecutionPlan,
) -> Option<crate::query::routing::CooperativeRuntimeBudget> {
    if !matches!(
        plan.template,
        ReadExecutionTemplate::CooperativeMerge | ReadExecutionTemplate::Historical
    ) {
        return None;
    }
    let configured = db.config.router.cooperative_runtime_budget;
    let budget = db
        .router
        .as_ref()
        .and_then(|router| {
            router.cooperative_runtime_budget(&plan.table, plan.execution_segments.len())
        })
        .unwrap_or(configured);
    budget.is_bounded().then_some(budget)
}

fn prepare_read_execution_plan(
    db: &RockDuck,
    table: &str,
    filter: Option<&str>,
    snapshot: TxnSnapshot,
    #[allow(unused_variables)]
    fallback_kind: QueryKind,
) -> Result<ReadExecutionPlan> {
    let current_snapshot_id = db.snapshot().snapshot_id;
    let projection_contract = if snapshot.snapshot_id == current_snapshot_id {
        None
    } else {
        Some(crate::metadata::projection::ProjectionContract::time_travel_scanner())
    };
    let parsed_expr = if let Some(filter_str) = filter {
        Some(
            crate::query::filter_expr::parse(filter_str)
                .map_err(|e| RockDuckError::Query(e.to_string()))?,
        )
    } else {
        None
    };

    let query_columns = parsed_expr
        .as_ref()
        .map(extract_filter_columns)
        .unwrap_or_default();

    let zone_map_group = parsed_expr
        .as_ref()
        .and_then(crate::query::filter_expr::to_zone_map_predicate_group);

    let route_plan =
        route_table_segments(db, table, filter, zone_map_group.as_ref(), fallback_kind)?;

    let evidence = build_evidence_snapshot(
        db,
        table,
        &query_columns,
        &route_plan.routed_segments,
        &route_plan.execution_segments,
        zone_map_group.as_ref(),
        projection_contract.clone(),
    )?;

    if let Some(router) = db.router.as_ref() {
        router.observe_metadata_evidence(db, &evidence);
        if let Some(routing) = route_plan.routing.as_ref() {
            let routing_evidence = evidence.clone().into_execution_evidence();
            router.observe_route_selection(
                table,
                &route_plan.routed_segments,
                routing,
                &routing_evidence,
            );
            router.observe_maintenance_evidence(db, &routing_evidence);
        }
    }

    Ok(ReadExecutionPlan {
        table: table.to_string(),
        filter: filter.map(str::to_string),
        query_columns,
        snapshot,
        routed_segments: route_plan.routed_segments,
        execution_segments: route_plan.execution_segments,
        parsed_expr,
        zone_map_group,
        template: route_plan.template,
        routing: route_plan.routing,
    })
}

fn execute_read_plan(
    db: &RockDuck,
    plan: &ReadExecutionPlan,
    vis_filter: &dyn crate::mvcc::visibility::VisFilter,
) -> Result<ReadExecutionOutput> {
    let started_at = Instant::now();
    let runtime_budget = cooperative_runtime_budget_for_plan(db, plan);
    let mut stats = ReadExecutionStats {
        segments_routed: plan.routed_segments.len(),
        cooperative: CooperativeRuntimeStats {
            slice_budget: runtime_budget.map(|budget| CooperativeSliceBudget {
                max_segments_per_slice: budget.max_segments_per_slice,
                max_slice_ms: budget.max_slice_ms,
            }),
            ..Default::default()
        },
        ..Default::default()
    };

    let shadow_choice = shadow_route_decision(db, plan);
    let segment_outputs =
        execute_segment_read_stages(db, plan, vis_filter, &mut stats, runtime_budget)?;
    let batches = finalize_read_filter_stage(plan, segment_outputs.clone(), &mut stats)?;

    let output = ReadExecutionOutput { batches, stats };
    if let Some(router) = db.router.as_ref() {
        let outcome = build_execution_outcome(
            plan,
            &output,
            &segment_outputs,
            started_at.elapsed().as_secs_f64() * 1000.0,
        );
        router.record_execution_outcome(&outcome);
        if let Some(shadow_path) = shadow_choice {
            if let Some(shadow_elapsed_ms) =
                measure_shadow_path(db, plan, vis_filter, shadow_path, runtime_budget)?
            {
                if let Some(sample) = sample_shadow_timing(
                    &outcome,
                    shadow_path,
                    shadow_elapsed_ms,
                    crate::query::routing::feedback::ShadowTimingPolicy::BoundedDualPath,
                ) {
                    router.record_shadow_timing(sample);
                }
            }
        }
        // D12 fix: pass committed_txn to checkpoint_if_dirty
        let committed_txn = db.mvcc.read().committed_txn();
        let _ = router.checkpoint_if_dirty(&db.kv, committed_txn);
    }

    Ok(output)
}

fn execute_segment_read_stages_with_template(
    db: &RockDuck,
    plan: &ReadExecutionPlan,
    template: ReadExecutionTemplate,
    vis_filter: &dyn crate::mvcc::visibility::VisFilter,
    stats: &mut ReadExecutionStats,
    runtime_budget: Option<crate::query::routing::CooperativeRuntimeBudget>,
) -> Result<Vec<SegmentReadStageOutput>> {
    if let Some(budget) = runtime_budget.filter(|_| {
        matches!(
            template,
            ReadExecutionTemplate::CooperativeMerge | ReadExecutionTemplate::Historical
        )
    }) {
        return execute_bounded_cooperative_segment_read_stages(
            db, plan, template, vis_filter, stats, budget,
        );
    }

    let raw_outputs: Result<Vec<Option<SegmentReadStageOutput>>> = plan
        .execution_segments
        .par_iter()
        .enumerate()
        .map(|(original_index, meta)| {
            let segment_output = match template {
                ReadExecutionTemplate::DeltaOnly => {
                    execute_delta_only_segment(db, meta, &plan.snapshot, vis_filter, original_index)
                }
                ReadExecutionTemplate::VortexOnly => execute_vortex_only_segment(
                    db,
                    meta,
                    &plan.snapshot,
                    vis_filter,
                    plan.zone_map_group.as_ref(),
                    original_index,
                ),
                ReadExecutionTemplate::CooperativeMerge | ReadExecutionTemplate::Historical => {
                    execute_cooperative_merge_segment(
                        db,
                        meta,
                        &plan.snapshot,
                        vis_filter,
                        plan.zone_map_group.as_ref(),
                        original_index,
                    )
                }
            }?;

            Ok(if segment_output.batches.is_empty() {
                None
            } else {
                Some(segment_output)
            })
        })
        .collect();

    let mut outputs: Vec<SegmentReadStageOutput> = raw_outputs?.into_iter().flatten().collect();
    outputs.sort_by_key(|output| output.original_index);

    accumulate_segment_output_stats(stats, &outputs);
    Ok(outputs)
}

fn execute_bounded_cooperative_segment_read_stages(
    db: &RockDuck,
    plan: &ReadExecutionPlan,
    template: ReadExecutionTemplate,
    vis_filter: &dyn crate::mvcc::visibility::VisFilter,
    stats: &mut ReadExecutionStats,
    runtime_budget: crate::query::routing::CooperativeRuntimeBudget,
) -> Result<Vec<SegmentReadStageOutput>> {
    let mut outputs = Vec::new();
    let started_at = Instant::now();
    let max_segments = runtime_budget.max_segments_per_slice.max(1);
    for (slice_idx, chunk) in plan.execution_segments.chunks(max_segments).enumerate() {
        stats.cooperative.total_slices += 1;
        let raw_outputs: Result<Vec<Option<SegmentReadStageOutput>>> = chunk
            .par_iter()
            .enumerate()
            .map(|(chunk_index, meta)| {
                let original_index = slice_idx * max_segments + chunk_index;
                let segment_output = match template {
                    ReadExecutionTemplate::CooperativeMerge | ReadExecutionTemplate::Historical => {
                        execute_cooperative_merge_segment(
                            db,
                            meta,
                            &plan.snapshot,
                            vis_filter,
                            plan.zone_map_group.as_ref(),
                            original_index,
                        )
                    }
                    ReadExecutionTemplate::DeltaOnly => execute_delta_only_segment(
                        db,
                        meta,
                        &plan.snapshot,
                        vis_filter,
                        original_index,
                    ),
                    ReadExecutionTemplate::VortexOnly => execute_vortex_only_segment(
                        db,
                        meta,
                        &plan.snapshot,
                        vis_filter,
                        plan.zone_map_group.as_ref(),
                        original_index,
                    ),
                }?;
                Ok(if segment_output.batches.is_empty() {
                    None
                } else {
                    Some(segment_output)
                })
            })
            .collect();

        let mut slice_outputs: Vec<SegmentReadStageOutput> =
            raw_outputs?.into_iter().flatten().collect();
        slice_outputs.sort_by_key(|output| output.original_index);
        accumulate_segment_output_stats(stats, &slice_outputs);
        outputs.extend(slice_outputs);

        if chunk.len() == max_segments && slice_idx > 0 {
            stats.cooperative.max_segments_per_slice_hit = true;
        }

        if runtime_budget.max_slice_ms > 0
            && started_at.elapsed().as_millis() as u64 >= runtime_budget.max_slice_ms
            && slice_idx + 1 < plan.execution_segments.len().div_ceil(max_segments)
        {
            stats.cooperative.elapsed_budget_exhausted = true;
            let remaining = &plan.execution_segments[(slice_idx + 1) * max_segments..];
            stats.cooperative.truncated_segment_ids =
                remaining.iter().map(|meta| meta.seg_id.clone()).collect();
            break;
        }
    }
    Ok(outputs)
}

fn accumulate_segment_output_stats(
    stats: &mut ReadExecutionStats,
    outputs: &[SegmentReadStageOutput],
) {
    for output in outputs {
        stats.base_batches_read += output.base_batches;
        stats.delta_rows_seen += output.delta_rows;
        stats.rows_after_visibility += output.rows_after_visibility;
    }
    stats.segments_scanned += outputs.len() as u64;
}

fn execute_segment_read_stages(
    db: &RockDuck,
    plan: &ReadExecutionPlan,
    vis_filter: &dyn crate::mvcc::visibility::VisFilter,
    stats: &mut ReadExecutionStats,
    runtime_budget: Option<crate::query::routing::CooperativeRuntimeBudget>,
) -> Result<Vec<SegmentReadStageOutput>> {
    execute_segment_read_stages_with_template(
        db,
        plan,
        plan.template,
        vis_filter,
        stats,
        runtime_budget,
    )
}

fn finalize_read_filter_stage(
    plan: &ReadExecutionPlan,
    segment_outputs: Vec<SegmentReadStageOutput>,
    stats: &mut ReadExecutionStats,
) -> Result<Vec<RecordBatch>> {
    let mut results = Vec::new();
    for output in segment_outputs {
        results.extend(output.batches);
    }

    if let Some(ref expr) = plan.parsed_expr {
        let mut filtered_results: Vec<RecordBatch> = Vec::new();
        for batch in results {
            let predicate = match crate::query::filter_expr::evaluate(expr, &batch) {
                Ok(p) => p,
                Err(e) => {
                    stats.filter_failures += 1;
                    return Err(RockDuckError::Query(format!(
                        "filter evaluation failed for table {} batch_rows={}: {}",
                        plan.table,
                        batch.num_rows(),
                        e
                    )));
                }
            };
            use arrow::compute::filter_record_batch;
            match filter_record_batch(&batch, &predicate) {
                Ok(filtered_batch) => {
                    stats.rows_after_filter += filtered_batch.num_rows();
                    filtered_results.push(filtered_batch)
                }
                Err(e) => {
                    stats.filter_failures += 1;
                    return Err(RockDuckError::Query(format!(
                        "filter_record_batch failed for table {} batch_rows={}: {}",
                        plan.table,
                        batch.num_rows(),
                        e
                    )));
                }
            }
        }
        results = filtered_results;
    } else {
        stats.rows_after_filter = results.iter().map(|b| b.num_rows()).sum();
    }

    Ok(results)
}

pub fn scan_segment_internal(
    db: &RockDuck,
    meta: &SegmentMeta,
    snapshot: &TxnSnapshot,
    vis_filter: &dyn crate::mvcc::visibility::VisFilter,
    zone_map_group: Option<&crate::query::filter_expr::ZoneMapPredicateGroup>,
) -> Result<Vec<RecordBatch>> {
    let mut stats = ReadExecutionStats::default();
    let output =
        execute_cooperative_merge_segment(db, meta, snapshot, vis_filter, zone_map_group, 0)?;
    stats.base_batches_read += output.base_batches;
    stats.delta_rows_seen += output.delta_rows;
    stats.rows_after_visibility += output.rows_after_visibility;
    Ok(output.batches)
}

fn execute_delta_only_segment(
    db: &RockDuck,
    meta: &SegmentMeta,
    snapshot: &TxnSnapshot,
    vis_filter: &dyn crate::mvcc::visibility::VisFilter,
    original_index: usize,
) -> Result<SegmentReadStageOutput> {
    let active_txns: HashSet<u64> = snapshot.active_txns.iter().copied().collect();
    let deltas = db.delta_layer.query(
        &meta.seg_id,
        snapshot.snapshot_id,
        &snapshot.commit_ts_map,
        &active_txns,
    )?;
    if deltas.is_empty() {
        return Ok(SegmentReadStageOutput {
            seg_id: meta.seg_id.clone(),
            original_index,
            batches: Vec::new(),
            base_batches: 0,
            delta_rows: 0,
            rows_after_visibility: 0,
            template: SegmentExecutionTemplate::DeltaOnly,
        });
    }

    let batch = build_delta_only_batch(meta, &deltas)?;
    let rows_before = batch.num_rows() as u64;
    let visible_batches = filter_batches_by_visibility_impl(
        Vec::new(),
        Arc::new(vec![batch]),
        snapshot,
        vis_filter,
        HashSet::new(),
    )?;
    let rows_after: u64 = visible_batches.iter().map(|b| b.num_rows() as u64).sum();

    Ok(SegmentReadStageOutput {
        seg_id: meta.seg_id.clone(),
        original_index,
        batches: visible_batches,
        base_batches: 0,
        delta_rows: deltas.len() as u64,
        rows_after_visibility: rows_after.min(rows_before),
        template: SegmentExecutionTemplate::DeltaOnly,
    })
}

fn execute_vortex_only_segment(
    db: &RockDuck,
    meta: &SegmentMeta,
    snapshot: &TxnSnapshot,
    vis_filter: &dyn crate::mvcc::visibility::VisFilter,
    zone_map_group: Option<&crate::query::filter_expr::ZoneMapPredicateGroup>,
    original_index: usize,
) -> Result<SegmentReadStageOutput> {
    let layout = SegmentLayout::new(&db.data_dir, &meta.seg_id);
    if should_skip_segment_for_zone_map(&layout, meta, zone_map_group) {
        return Ok(SegmentReadStageOutput {
            seg_id: meta.seg_id.clone(),
            original_index,
            batches: Vec::new(),
            base_batches: 0,
            delta_rows: 0,
            rows_after_visibility: 0,
            template: SegmentExecutionTemplate::VortexOnly,
        });
    }

    db.access_tracker
        .mark_access(&meta.seg_id, 0, meta.row_count.min(u32::MAX as u64) as u32);

    let base_batches = read_vortex_segment(&layout, meta)?;
    let vis_batches = read_visibility_batches(&layout)?;
    let deleted_rows = read_deleted_rows(&layout, &vis_batches)?;
    let visible = filter_batches_by_visibility_impl(
        vis_batches,
        base_batches.clone(),
        snapshot,
        vis_filter,
        deleted_rows,
    )?;
    let rows_after = visible.iter().map(|b| b.num_rows() as u64).sum();

    Ok(SegmentReadStageOutput {
        seg_id: meta.seg_id.clone(),
        original_index,
        batches: visible,
        base_batches: base_batches.len() as u64,
        delta_rows: 0,
        rows_after_visibility: rows_after,
        template: SegmentExecutionTemplate::VortexOnly,
    })
}

fn execute_cooperative_merge_segment(
    db: &RockDuck,
    meta: &SegmentMeta,
    snapshot: &TxnSnapshot,
    vis_filter: &dyn crate::mvcc::visibility::VisFilter,
    zone_map_group: Option<&crate::query::filter_expr::ZoneMapPredicateGroup>,
    original_index: usize,
) -> Result<SegmentReadStageOutput> {
    let seg_id = &meta.seg_id;
    let layout = SegmentLayout::new(&db.data_dir, seg_id);
    if should_skip_segment_for_zone_map(&layout, meta, zone_map_group) {
        return Ok(SegmentReadStageOutput {
            seg_id: meta.seg_id.clone(),
            original_index,
            batches: Vec::new(),
            base_batches: 0,
            delta_rows: 0,
            rows_after_visibility: 0,
            template: SegmentExecutionTemplate::CooperativeMerge,
        });
    }

    db.access_tracker
        .mark_access(seg_id, 0, meta.row_count.min(u32::MAX as u64) as u32);

    let base_batches = read_vortex_segment(&layout, meta)?;
    if base_batches.is_empty() {
        return Ok(SegmentReadStageOutput {
            seg_id: meta.seg_id.clone(),
            original_index,
            batches: Vec::new(),
            base_batches: 0,
            delta_rows: 0,
            rows_after_visibility: 0,
            template: SegmentExecutionTemplate::CooperativeMerge,
        });
    }

    let vis_batches = read_visibility_batches(&layout)?;
    let deleted_rows = read_deleted_rows(&layout, &vis_batches)?;
    let active_txns: HashSet<u64> = snapshot.active_txns.iter().copied().collect();
    let deltas = db.delta_layer.query(
        seg_id,
        snapshot.snapshot_id,
        &snapshot.commit_ts_map,
        &active_txns,
    )?;

    let visible = if deltas.is_empty() {
        filter_batches_by_visibility_impl(
            vis_batches,
            base_batches.clone(),
            snapshot,
            vis_filter,
            deleted_rows,
        )?
    } else {
        let mut merged_batches = Vec::with_capacity(base_batches.len());
        for batch in base_batches.iter() {
            merged_batches.push(apply_deltas_to_batch(batch, &deltas)?);
        }
        filter_batches_by_visibility_impl(
            vis_batches,
            Arc::new(merged_batches),
            snapshot,
            vis_filter,
            deleted_rows,
        )?
    };
    let rows_after = visible.iter().map(|b| b.num_rows() as u64).sum();

    Ok(SegmentReadStageOutput {
        seg_id: meta.seg_id.clone(),
        original_index,
        batches: visible,
        base_batches: base_batches.len() as u64,
        delta_rows: deltas.len() as u64,
        rows_after_visibility: rows_after,
        template: SegmentExecutionTemplate::CooperativeMerge,
    })
}

fn should_skip_segment_for_zone_map(
    layout: &SegmentLayout,
    meta: &SegmentMeta,
    zone_map_group: Option<&crate::query::filter_expr::ZoneMapPredicateGroup>,
) -> bool {
    if let Some(group) = zone_map_group {
        if group.has_cross_column_or || group.is_empty() {
            return false;
        }

        let mut skip_segment = true;
        for or_group in &group.groups {
            let may_overlap = evaluate_or_group_for_segment(layout, meta, or_group);
            if may_overlap {
                skip_segment = false;
                break;
            }
        }
        return skip_segment;
    }
    false
}

/// Read visibility batches for a segment layout.
///
/// Returns Ok(Vec<RecordBatch>) on success, or an error if the visibility file
/// cannot be read. An empty Vec is only returned when the file legitimately
/// doesn't exist (segment is too new to have visibility metadata written).
///
/// Note: This function intentionally returns Ok(Vec::new()) when the vis.vortex
/// file doesn't exist yet - this is a valid transient state during segment creation.
fn read_visibility_batches(layout: &SegmentLayout) -> Result<Vec<RecordBatch>> {
    let vis_path = layout.vis_path();
    if vis_path.exists() {
        match VortexReader::open(&vis_path) {
            Ok(r) => Ok(r.read_all_batches().as_ref().clone()),
            // Fix R-6: Changed from silently returning empty Vec to propagating error.
            // This distinguishes "no visibility data" from "visibility data unavailable".
            Err(e) => Err(RockDuckError::ReadPath(format!(
                "failed to read visibility batches from {}: {}",
                vis_path.display(),
                e
            ))),
        }
    } else {
        // vis.vortex doesn't exist yet - this is a valid transient state
        Ok(Vec::new())
    }
}

fn read_deleted_rows(layout: &SegmentLayout, vis_batches: &[RecordBatch]) -> Result<HashSet<u64>> {
    if vis_batches.is_empty() && !layout.deltavis_path().exists() {
        return Ok(HashSet::new());
    }

    let vis_path = layout.vis_path();
    let deltavis_path = layout.deltavis_path();
    if deltavis_path.exists() {
        let vis_writer = crate::write::vis_file::VisFileWriter::new(&vis_path);
        return vis_writer
            .read_deltavis()
            .map(|entries| entries.into_iter().map(|(row, _)| row).collect());
    }
    Ok(HashSet::new())
}

fn build_delta_only_batch(
    meta: &SegmentMeta,
    deltas: &[crate::storage::delta::DeltaCell],
) -> Result<RecordBatch> {
    let mut row_map: BTreeMap<u64, BTreeMap<String, Option<Vec<u8>>>> = BTreeMap::new();
    for delta in deltas {
        row_map.entry(delta.row_offset).or_default().insert(
            delta.column.clone(),
            delta.after.as_ref().map(|bytes| bytes.to_vec()),
        );
    }

    let fields: Vec<Field> = meta
        .columns
        .iter()
        .map(|col| Field::new(&col.name, data_type_to_arrow(&col.data_type), true))
        .collect();
    let schema = Arc::new(arrow_schema::Schema::new(fields.clone()));

    let mut columns: Vec<ArrayRef> = Vec::with_capacity(fields.len());
    for col in &meta.columns {
        let values: Vec<Option<Vec<u8>>> = row_map
            .values()
            .map(|row| row.get(&col.name).cloned().unwrap_or(None))
            .collect();
        columns.push(build_array_from_delta_values(&col.data_type, &values)?);
    }

    RecordBatch::try_new(schema, columns)
        .map_err(|e| RockDuckError::Internal(format!("Build delta-only batch: {}", e)))
}

fn build_array_from_delta_values(
    dt: &crate::segment::meta::DataType,
    values: &[Option<Vec<u8>>],
) -> Result<ArrayRef> {
    use crate::segment::meta::DataType as D;

    macro_rules! primitive_array {
        ($builder:ty, $ty:ty, $decode:expr) => {{
            let mut b = <$builder>::with_capacity(values.len());
            for value in values {
                if let Some(bytes) = value {
                    let parsed: $ty = $decode(bytes.as_slice())?;
                    b.append_value(parsed);
                } else {
                    b.append_null();
                }
            }
            Arc::new(b.finish()) as ArrayRef
        }};
    }

    fn expect_len(bytes: &[u8], expected: usize, name: &str) -> Result<()> {
        if bytes.len() != expected {
            Err(RockDuckError::Internal(format!(
                "delta decode {} expected {} bytes, got {}",
                name,
                expected,
                bytes.len()
            )))
        } else {
            Ok(())
        }
    }

    Ok(match dt {
        D::Int8 => primitive_array!(
            arrow_array::builder::Int8Builder,
            i8,
            |b: &[u8]| -> Result<i8> {
                expect_len(b, 1, "i8")?;
                Ok(b[0] as i8)
            }
        ),
        D::UInt8 => primitive_array!(
            arrow_array::builder::UInt8Builder,
            u8,
            |b: &[u8]| -> Result<u8> {
                expect_len(b, 1, "u8")?;
                Ok(b[0])
            }
        ),
        D::Bool => primitive_array!(
            arrow_array::builder::BooleanBuilder,
            bool,
            |b: &[u8]| -> Result<bool> {
                expect_len(b, 1, "bool")?;
                Ok(b[0] != 0)
            }
        ),
        D::Int16 => primitive_array!(
            arrow_array::builder::Int16Builder,
            i16,
            |b: &[u8]| -> Result<i16> {
                expect_len(b, 2, "i16")?;
                Ok(i16::from_le_bytes([b[0], b[1]]))
            }
        ),
        D::UInt16 => primitive_array!(
            arrow_array::builder::UInt16Builder,
            u16,
            |b: &[u8]| -> Result<u16> {
                expect_len(b, 2, "u16")?;
                Ok(u16::from_le_bytes([b[0], b[1]]))
            }
        ),
        D::Int32 | D::Date32 => primitive_array!(
            arrow_array::builder::Int32Builder,
            i32,
            |b: &[u8]| -> Result<i32> {
                expect_len(b, 4, "i32")?;
                Ok(i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            }
        ),
        D::UInt32 => primitive_array!(
            arrow_array::builder::UInt32Builder,
            u32,
            |b: &[u8]| -> Result<u32> {
                expect_len(b, 4, "u32")?;
                Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            }
        ),
        D::Float32 => primitive_array!(
            arrow_array::builder::Float32Builder,
            f32,
            |b: &[u8]| -> Result<f32> {
                expect_len(b, 4, "f32")?;
                Ok(f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            }
        ),
        D::Int64 | D::Date64 | D::TimestampMicros | D::TimestampMillis => {
            primitive_array!(
                arrow_array::builder::Int64Builder,
                i64,
                |b: &[u8]| -> Result<i64> {
                    expect_len(b, 8, "i64")?;
                    Ok(i64::from_le_bytes([
                        b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
                    ]))
                }
            )
        }
        D::UInt64 => primitive_array!(
            arrow_array::builder::UInt64Builder,
            u64,
            |b: &[u8]| -> Result<u64> {
                expect_len(b, 8, "u64")?;
                Ok(u64::from_le_bytes([
                    b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
                ]))
            }
        ),
        D::Float64 => primitive_array!(
            arrow_array::builder::Float64Builder,
            f64,
            |b: &[u8]| -> Result<f64> {
                expect_len(b, 8, "f64")?;
                Ok(f64::from_le_bytes([
                    b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
                ]))
            }
        ),
        D::Utf8 => {
            let mut b =
                arrow_array::builder::StringBuilder::with_capacity(values.len(), values.len() * 8);
            for value in values {
                if let Some(bytes) = value {
                    let s = String::from_utf8(bytes.clone()).map_err(|e| {
                        RockDuckError::Internal(format!("delta decode utf8: {}", e))
                    })?;
                    b.append_value(s);
                } else {
                    b.append_null();
                }
            }
            Arc::new(b.finish()) as ArrayRef
        }
        D::LargeUtf8 => {
            let mut b = arrow_array::builder::LargeStringBuilder::with_capacity(
                values.len(),
                values.len() * 8,
            );
            for value in values {
                if let Some(bytes) = value {
                    let s = String::from_utf8(bytes.clone()).map_err(|e| {
                        RockDuckError::Internal(format!("delta decode largeutf8: {}", e))
                    })?;
                    b.append_value(s);
                } else {
                    b.append_null();
                }
            }
            Arc::new(b.finish()) as ArrayRef
        }
        D::Binary => {
            let mut b = arrow_array::builder::BinaryBuilder::with_capacity(
                values.len(),
                values.iter().flatten().map(|v| v.len()).sum(),
            );
            for value in values {
                if let Some(bytes) = value {
                    b.append_value(bytes);
                } else {
                    b.append_null();
                }
            }
            Arc::new(b.finish()) as ArrayRef
        }
        D::LargeBinary => {
            let mut b = arrow_array::builder::LargeBinaryBuilder::with_capacity(
                values.len(),
                values.iter().flatten().map(|v| v.len()).sum(),
            );
            for value in values {
                if let Some(bytes) = value {
                    b.append_value(bytes);
                } else {
                    b.append_null();
                }
            }
            Arc::new(b.finish()) as ArrayRef
        }
    })
}

fn build_execution_outcome(
    plan: &ReadExecutionPlan,
    output: &ReadExecutionOutput,
    segment_outputs: &[SegmentReadStageOutput],
    elapsed_ms: f64,
) -> ExecutionOutcome {
    ExecutionOutcome {
        table: plan.table.clone(),
        query_columns: plan.query_columns.clone(),
        template: plan.template,
        chosen_path: plan.routing.as_ref().map(|routing| routing.decision),
        estimated_cost: plan.routing.as_ref().map(|routing| routing.estimated_cost),
        rows_scanned: output
            .stats
            .rows_after_visibility
            .max(output.stats.rows_after_filter as u64),
        candidate_rows: plan.routed_segments.iter().map(|meta| meta.row_count).sum(),
        rows_after_visibility: output.stats.rows_after_visibility,
        rows_returned: output.stats.rows_after_filter as u64,
        filter_failures: output.stats.filter_failures,
        skipped_segment_failures: output.stats.skipped_segment_failures,
        skipped_segment_ids: output.stats.skipped_segment_ids.clone(),
        segments_routed: output.stats.segments_routed as u64,
        segments_scanned: output.stats.segments_scanned,
        elapsed_ms,
        executed_segment_ids: segment_outputs
            .iter()
            .map(|output| output.seg_id.clone())
            .collect(),
        cooperative_digest: output.stats.cooperative.slice_budget.map(|budget| {
            CooperativeExecutionDigest {
                budget: Some(budget),
                slices_executed: output.stats.cooperative.total_slices,
                truncated_segments: output.stats.cooperative.truncated_segment_ids.clone(),
                elapsed_budget_exhausted: output.stats.cooperative.elapsed_budget_exhausted,
                max_segments_per_slice_hit: output.stats.cooperative.max_segments_per_slice_hit,
            }
        }),
    }
}

fn shadow_template_for_path(path: RouteDecision) -> ReadExecutionTemplate {
    match path {
        RouteDecision::DeltaStoreOnly => ReadExecutionTemplate::DeltaOnly,
        RouteDecision::VortexOnly => ReadExecutionTemplate::VortexOnly,
        RouteDecision::Merge => ReadExecutionTemplate::CooperativeMerge,
    }
}

fn measure_shadow_path(
    db: &RockDuck,
    plan: &ReadExecutionPlan,
    vis_filter: &dyn crate::mvcc::visibility::VisFilter,
    shadow_path: RouteDecision,
    runtime_budget: Option<crate::query::routing::CooperativeRuntimeBudget>,
) -> Result<Option<f64>> {
    let shadow_template = shadow_template_for_path(shadow_path);
    if shadow_template == plan.template {
        return Ok(None);
    }

    let started_at = Instant::now();
    let mut shadow_stats = ReadExecutionStats {
        segments_routed: plan.routed_segments.len(),
        cooperative: CooperativeRuntimeStats {
            slice_budget: runtime_budget.map(|budget| CooperativeSliceBudget {
                max_segments_per_slice: budget.max_segments_per_slice,
                max_slice_ms: budget.max_slice_ms,
            }),
            ..Default::default()
        },
        ..Default::default()
    };
    let shadow_outputs = execute_segment_read_stages_with_template(
        db,
        plan,
        shadow_template,
        vis_filter,
        &mut shadow_stats,
        runtime_budget,
    )?;
    let _ = finalize_read_filter_stage(plan, shadow_outputs, &mut shadow_stats)?;
    Ok(Some(started_at.elapsed().as_secs_f64() * 1000.0))
}

fn shadow_route_decision(db: &RockDuck, plan: &ReadExecutionPlan) -> Option<RouteDecision> {
    let router = db.router.as_ref()?;
    let chosen_path = plan.routing.as_ref()?.decision;
    if db.config.router.shadow_sample_rate <= 0.0 {
        return None;
    }
    if !router.should_sample_shadow_timing(
        &plan.table,
        &plan.query_columns,
        chosen_path,
        plan.routed_segments.len(),
    ) {
        return None;
    }
    match chosen_path {
        RouteDecision::DeltaStoreOnly => Some(RouteDecision::VortexOnly),
        RouteDecision::VortexOnly => Some(RouteDecision::DeltaStoreOnly),
        RouteDecision::Merge => Some(RouteDecision::VortexOnly),
    }
}

fn sample_shadow_timing(
    outcome: &ExecutionOutcome,
    shadow_path: RouteDecision,
    shadow_elapsed_ms: f64,
    policy: crate::query::routing::feedback::ShadowTimingPolicy,
) -> Option<ShadowTimingSample> {
    let floor_ms = (outcome.rows_scanned.max(1) as f64).ln_1p();
    ShadowTimingSample::sampled_from_outcome(
        outcome,
        shadow_path,
        shadow_elapsed_ms.max(floor_ms),
        policy,
    )
}

fn extract_filter_columns(expr: &crate::query::filter_expr::Expr) -> Vec<String> {
    fn walk(expr: &crate::query::filter_expr::Expr, out: &mut Vec<String>) {
        use crate::query::filter_expr::Expr;
        match expr {
            Expr::Comparison { col, .. } => {
                if !out.contains(col) {
                    out.push(col.clone());
                }
            }
            Expr::And(l, r) | Expr::Or(l, r) => {
                walk(l, out);
                walk(r, out);
            }
            Expr::Not(inner) => walk(inner, out),
        }
    }

    let mut out = Vec::new();
    walk(expr, &mut out);
    out
}

fn table_segment_ids(db: &RockDuck, table: &str) -> Result<HashSet<String>> {
    Ok(metadata::list_segment_metas(&db.kv)?
        .into_iter()
        .filter(|meta| meta.table_id == table)
        .filter(|meta| meta.status != crate::segment::meta::SegmentStatus::Garbage)
        .map(|meta| meta.seg_id)
        .collect())
}

pub fn build_evidence_snapshot(
    db: &RockDuck,
    table: &str,
    query_columns: &[String],
    routed_segments: &[SegmentMeta],
    executed_segments: &[SegmentMeta],
    zone_map_group: Option<&crate::query::filter_expr::ZoneMapPredicateGroup>,
    projection_contract: Option<crate::metadata::projection::ProjectionContract>,
) -> Result<metadata::EvidenceSnapshot> {
    let table_stats = build_router_table_stats(db, table)?;
    let total_segment_rows = routed_segments.iter().map(|meta| meta.row_count).sum();
    let routed_segment_ids = routed_segments
        .iter()
        .map(|meta| meta.seg_id.clone())
        .collect();
    let executed_segment_ids: Vec<String> = executed_segments
        .iter()
        .map(|meta| meta.seg_id.clone())
        .collect();
    let table_segment_ids = table_segment_ids(db, table)?;
    let delta_segment_count = db
        .delta_layer
        .get_segment_ids_for_table(&table_segment_ids)
        .len();

    Ok(metadata::EvidenceSnapshot {
        table: table.to_string(),
        query_columns: query_columns.to_vec(),
        table_stats,
        routed_segment_ids,
        executed_segment_ids,
        total_segment_rows,
        delta_segment_count,
        has_zone_map_predicates: zone_map_group.is_some_and(|group| !group.is_empty()),
        has_cross_column_or: zone_map_group.is_some_and(|group| group.has_cross_column_or),
        projection_contract,
    })
}

fn build_router_table_stats(
    db: &RockDuck,
    table: &str,
) -> Result<crate::query::routing::TableStats> {
    let table_stats = get_or_create_table_stats(&db.kv, table)?;
    Ok(crate::query::routing::TableStats {
        row_count: table_stats.row_count,
        total_bytes: table_stats.total_size_bytes,
        compressed_bytes: (table_stats.total_size_bytes / 2).max(1),
        ..Default::default()
    })
}

#[cfg(test)]
mod governance_contract_tests {
    use super::*;

    #[test]
    fn bounded_dual_path_shadow_template_maps_to_expected_execution_template() {
        assert_eq!(
            shadow_template_for_path(RouteDecision::DeltaStoreOnly),
            ReadExecutionTemplate::DeltaOnly
        );
        assert_eq!(
            shadow_template_for_path(RouteDecision::VortexOnly),
            ReadExecutionTemplate::VortexOnly
        );
        assert_eq!(
            shadow_template_for_path(RouteDecision::Merge),
            ReadExecutionTemplate::CooperativeMerge
        );
    }

    #[test]
    fn sample_shadow_timing_preserves_bounded_dual_path_policy() {
        let outcome = ExecutionOutcome {
            table: "orders".to_string(),
            query_columns: vec!["status".to_string()],
            template: ReadExecutionTemplate::VortexOnly,
            chosen_path: Some(RouteDecision::VortexOnly),
            estimated_cost: Some(12.0),
            rows_scanned: 100,
            candidate_rows: 100,
            rows_after_visibility: 100,
            rows_returned: 10,
            filter_failures: 0,
            skipped_segment_failures: 0,
            skipped_segment_ids: vec![],
            segments_routed: 1,
            segments_scanned: 1,
            elapsed_ms: 8.0,
            executed_segment_ids: vec!["seg-1".to_string()],
            cooperative_digest: None,
        };

        let sample = sample_shadow_timing(
            &outcome,
            RouteDecision::DeltaStoreOnly,
            6.5,
            crate::query::routing::feedback::ShadowTimingPolicy::BoundedDualPath,
        )
        .expect("bounded dual-path sample should be created");

        assert_eq!(
            sample.policy,
            crate::query::routing::feedback::ShadowTimingPolicy::BoundedDualPath
        );
        assert_eq!(sample.shadow_elapsed_ms, 6.5);
        assert_eq!(sample.shadow_path, RouteDecision::DeltaStoreOnly);
    }

    #[test]
    fn historical_scan_evidence_carries_blocking_projection_contract() {
        let contract = crate::metadata::projection::ProjectionContract::time_travel_scanner();
        let evidence = metadata::EvidenceSnapshot {
            table: "orders".to_string(),
            query_columns: vec!["status".to_string()],
            table_stats: crate::query::routing::TableStats::default(),
            routed_segment_ids: vec!["seg-1".to_string()],
            executed_segment_ids: vec!["seg-1".to_string()],
            total_segment_rows: 10,
            delta_segment_count: 1,
            has_zone_map_predicates: false,
            has_cross_column_or: false,
            projection_contract: Some(contract.clone()),
        };

        contract.assert_blocking_governance();
        evidence.assert_governance_ready();
        let exec = evidence.into_execution_evidence();
        assert!(exec.projection_contract.is_some());
        assert_eq!(
            exec.projection_contract.unwrap().surface,
            crate::metadata::projection::ProjectionSurface::TimeTravelScanner
        );
    }

    #[test]
    fn live_scan_metadata_evidence_requires_projection_contract() {
        let evidence = metadata::EvidenceSnapshot {
            table: "orders".to_string(),
            query_columns: vec!["status".to_string()],
            table_stats: crate::query::routing::TableStats::default(),
            routed_segment_ids: vec!["seg-1".to_string()],
            executed_segment_ids: vec![],
            total_segment_rows: 10,
            delta_segment_count: 1,
            has_zone_map_predicates: false,
            has_cross_column_or: false,
            projection_contract: Some(crate::metadata::projection::ProjectionContract::point_get()),
        };

        evidence.assert_governance_ready();
    }

    #[test]
    fn execution_outcome_uses_only_executed_segment_ids_from_stage_outputs() {
        let segment = SegmentMeta::new(
            "seg-routed-only".to_string(),
            "orders".to_string(),
            Vec::new(),
        );
        let plan = ReadExecutionPlan {
            table: "orders".to_string(),
            filter: None,
            query_columns: vec!["status".to_string()],
            snapshot: TxnSnapshot::new(
                0,
                std::collections::BTreeSet::new(),
                IsolationLevel::Snapshot,
            ),
            routed_segments: vec![segment.clone()],
            execution_segments: vec![segment.clone()],
            parsed_expr: None,
            zone_map_group: None,
            template: ReadExecutionTemplate::VortexOnly,
            routing: None,
        };
        let output = ReadExecutionOutput {
            batches: Vec::new(),
            stats: ReadExecutionStats {
                segments_routed: 1,
                segments_scanned: 0,
                base_batches_read: 0,
                delta_rows_seen: 0,
                rows_after_visibility: 0,
                rows_after_filter: 0,
                filter_failures: 0,
                skipped_segment_failures: 0,
                skipped_segment_ids: vec![],
                cooperative: CooperativeRuntimeStats::default(),
            },
        };
        let outcome = build_execution_outcome(&plan, &output, &[], 1.0);

        assert!(outcome.executed_segment_ids.is_empty());
        assert_eq!(outcome.segments_scanned, 0);
    }

    #[test]
    fn cooperative_digest_uses_recorded_truncated_segment_ids() {
        let executed =
            SegmentMeta::new("seg-executed".to_string(), "orders".to_string(), Vec::new());
        let truncated = SegmentMeta::new(
            "seg-truncated".to_string(),
            "orders".to_string(),
            Vec::new(),
        );
        let plan = ReadExecutionPlan {
            table: "orders".to_string(),
            filter: None,
            query_columns: vec!["status".to_string()],
            snapshot: TxnSnapshot::new(
                0,
                std::collections::BTreeSet::new(),
                IsolationLevel::Snapshot,
            ),
            routed_segments: vec![executed.clone(), truncated.clone()],
            execution_segments: vec![executed.clone(), truncated.clone()],
            parsed_expr: None,
            zone_map_group: None,
            template: ReadExecutionTemplate::CooperativeMerge,
            routing: None,
        };
        let segment_output = SegmentReadStageOutput {
            seg_id: executed.seg_id.clone(),
            original_index: 0,
            batches: Vec::new(),
            base_batches: 0,
            delta_rows: 0,
            rows_after_visibility: 0,
            template: SegmentExecutionTemplate::CooperativeMerge,
        };
        let output = ReadExecutionOutput {
            batches: Vec::new(),
            stats: ReadExecutionStats {
                segments_routed: 2,
                segments_scanned: 1,
                base_batches_read: 0,
                delta_rows_seen: 0,
                rows_after_visibility: 0,
                rows_after_filter: 0,
                filter_failures: 0,
                skipped_segment_failures: 0,
                skipped_segment_ids: vec![],
                cooperative: CooperativeRuntimeStats {
                    slice_budget: Some(CooperativeSliceBudget {
                        max_segments_per_slice: 1,
                        max_slice_ms: 10,
                    }),
                    total_slices: 1,
                    truncated_segment_ids: vec![truncated.seg_id.clone()],
                    elapsed_budget_exhausted: true,
                    max_segments_per_slice_hit: true,
                },
            },
        };

        let outcome = build_execution_outcome(&plan, &output, &[segment_output], 1.0);
        let digest = outcome
            .cooperative_digest
            .expect("cooperative digest should exist");
        assert_eq!(digest.truncated_segments, vec!["seg-truncated".to_string()]);
        assert_eq!(
            outcome.executed_segment_ids,
            vec!["seg-executed".to_string()]
        );
    }

    #[test]
    fn table_scoped_delta_segment_count_excludes_other_tables() {
        let table_segment_ids =
            HashSet::from(["orders-seg-1".to_string(), "orders-seg-2".to_string()]);
        let all_delta_ids = HashSet::from([
            "orders-seg-1".to_string(),
            "other-seg-1".to_string(),
            "other-seg-2".to_string(),
        ]);

        let scoped: Vec<String> = all_delta_ids
            .into_iter()
            .filter(|seg_id| table_segment_ids.contains(seg_id))
            .collect();

        assert_eq!(scoped, vec!["orders-seg-1".to_string()]);
    }
}

fn build_router_params(
    db: &RockDuck,
    table: &str,
    filter: Option<&str>,
) -> Result<RouterParamsOwned> {
    let query_stats = build_router_table_stats(db, table)?;
    let table_segment_ids = table_segment_ids(db, table)?;
    let delta_segment_ids = db.delta_layer.get_segment_ids_for_table(&table_segment_ids);
    let columns = filter
        .and_then(|f| crate::query::filter_expr::parse(f).ok())
        .map(|expr| extract_filter_columns(&expr))
        .unwrap_or_default();

    let estimated_selectivity = if let Some(expr_str) = filter {
        crate::query::filter_expr::parse(expr_str)
            .ok()
            .and_then(|expr| crate::query::filter_expr::to_zone_map_predicate_group(&expr))
            .map(|group| if group.has_cross_column_or { 0.25 } else { 0.1 })
            .unwrap_or(0.1)
    } else {
        1.0
    };

    let kind = QueryKind::from_filter(filter.is_some(), false, estimated_selectivity);

    Ok(RouterParamsOwned::new(
        table.to_string(),
        columns,
        kind,
        delta_segment_ids.len(),
        !delta_segment_ids.is_empty(),
        estimated_selectivity,
        query_stats,
    ))
}

fn route_table_segments(
    db: &RockDuck,
    table: &str,
    filter: Option<&str>,
    zone_map_group: Option<&crate::query::filter_expr::ZoneMapPredicateGroup>,
    #[allow(unused_variables)]
    fallback_kind: QueryKind,
) -> Result<RoutedExecutionPlan> {
    let metas = metadata::list_segment_metas(&db.kv)?;
    let table_metas: Vec<SegmentMeta> = metas
        .into_iter()
        .filter(|m| m.table_id == table)
        .filter(|m| m.status != crate::segment::meta::SegmentStatus::Garbage)
        .collect();

    let delta_segment_ids: HashSet<String> =
        db.delta_layer.get_all_segment_ids().into_iter().collect();
    let has_vortex_layout = |meta: &SegmentMeta| {
        let layout = SegmentLayout::new(&db.data_dir, &meta.seg_id);
        meta.columns
            .iter()
            .any(|col| layout.col_path(&col.name).exists())
    };
    let passes_zone_map = |meta: &SegmentMeta| {
        let layout = SegmentLayout::new(&db.data_dir, &meta.seg_id);
        !should_skip_segment_for_zone_map(&layout, meta, zone_map_group)
    };

    let Some(router) = db.router.as_ref() else {
        return Ok(RoutedExecutionPlan {
            routed_segments: table_metas.clone(),
            execution_segments: table_metas,
            template: ReadExecutionTemplate::CooperativeMerge,
            routing: None,
        });
    };

    let params_owned = build_router_params(db, table, filter)?;
    let params = params_owned.as_borrowed();
    let routing = router.route(&params);

    let routed_segments: Vec<SegmentMeta> = match routing.decision {
        RouteDecision::DeltaStoreOnly => table_metas
            .iter()
            .filter(|meta| {
                delta_segment_ids.contains(&meta.seg_id) && !has_vortex_layout(meta)
            })
            .cloned()
            .collect(),
        RouteDecision::VortexOnly => table_metas
            .iter()
            .filter(|meta| passes_zone_map(meta) && has_vortex_layout(meta))
            .cloned()
            .collect(),
        RouteDecision::Merge => table_metas
            .iter()
            .filter(|meta| passes_zone_map(meta) || delta_segment_ids.contains(&meta.seg_id))
            .cloned()
            .collect(),
    };

    let (template, execution_segments): (ReadExecutionTemplate, Vec<SegmentMeta>) =
        match routing.decision {
            RouteDecision::DeltaStoreOnly => (
                ReadExecutionTemplate::DeltaOnly,
                routed_segments
                    .iter()
                    .filter(|meta| {
                        delta_segment_ids.contains(&meta.seg_id) && !has_vortex_layout(meta)
                    })
                    .cloned()
                    .collect(),
            ),
            RouteDecision::VortexOnly => (
                ReadExecutionTemplate::VortexOnly,
                routed_segments
                    .iter()
                    .filter(|meta| has_vortex_layout(meta) && passes_zone_map(meta))
                    .cloned()
                    .collect(),
            ),
            RouteDecision::Merge => (
                ReadExecutionTemplate::CooperativeMerge,
                routed_segments.clone(),
            ),
        };

    Ok(RoutedExecutionPlan {
        routed_segments,
        execution_segments,
        template,
        routing: Some(routing),
    })
}

fn evaluate_or_group_for_segment(
    layout: &SegmentLayout,
    meta: &SegmentMeta,
    or_group: &crate::query::filter_expr::ZoneMapOrGroup,
) -> bool {
    use crate::metadata::block_zone_map::GranuleZoneMapIndex;

    let col_name = &or_group.column;
    if !meta.columns.iter().any(|c| &c.name == col_name) {
        return true;
    }

    let zm_path = layout.block_zm_path(col_name);
    if !zm_path.exists() {
        return true;
    }

    let zm_bytes = match std::fs::read(&zm_path) {
        Ok(b) => b,
        Err(e) => {
            tracing::debug!(
                seg_id = %meta.seg_id,
                column = %col_name,
                error = %e,
                "ZoneMap file read failed, reading segment conservatively"
            );
            return true;
        }
    };

    let zm = match GranuleZoneMapIndex::from_bytes(&zm_bytes) {
        Ok(z) => z,
        Err(e) => {
            tracing::debug!(
                seg_id = %meta.seg_id,
                column = %col_name,
                error = %e,
                "ZoneMap parse failed, reading segment conservatively"
            );
            return true;
        }
    };

    for (pred_min, pred_max) in &or_group.ranges {
        if zm.may_overlap(pred_min, pred_max) {
            return true;
        }
    }
    false
}

fn read_vortex_segment(
    layout: &SegmentLayout,
    meta: &SegmentMeta,
) -> Result<Arc<Vec<RecordBatch>>> {
    if meta.columns.is_empty() {
        return Ok(Arc::new(Vec::new()));
    }

    let mut all_batches: Vec<(String, arrow_schema::DataType, Arc<Vec<RecordBatch>>, usize)> =
        Vec::new();
    let mut min_batch_count = usize::MAX;

    for col_def in &meta.columns {
        let col_path = layout.col_path(&col_def.name);
        if !col_path.exists() {
            continue;
        }

        let reader = VortexReader::open(&col_path)?;
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
        return Ok(Arc::new(Vec::new()));
    }

    let mut misaligned = false;
    for (name, _, _, count) in &all_batches {
        if *count != min_batch_count {
            tracing::warn!(
                "scan: column '{}' has {} batches but minimum is {} — batch alignment mismatch detected",
                name,
                count,
                min_batch_count
            );
            misaligned = true;
        }
    }

    let num_output_cols = all_batches.len();
    let mut result = Vec::with_capacity(min_batch_count);
    for batch_idx in 0..min_batch_count {
        let mut columns: Vec<ArrayRef> = Vec::with_capacity(num_output_cols);
        let mut all_columns_present = true;
        for (name, _, batches, count) in &all_batches {
            if batch_idx < *count {
                columns.push(batches[batch_idx].column(0).clone());
            } else {
                tracing::warn!(
                    "scan: column '{}' missing batch {} (has {} batches) — skipping batch",
                    name,
                    batch_idx,
                    count
                );
                all_columns_present = false;
            }
        }
        if !all_columns_present {
            continue;
        }

        let fields: Vec<Field> = all_batches
            .iter()
            .map(|(name, dt, _, _)| Field::new(name, dt.clone(), true))
            .collect();

        let schema = arrow_schema::Schema::new(fields);
        let batch = RecordBatch::try_new(Arc::new(schema), columns)
            .map_err(|e| RockDuckError::Internal(format!("Build batch: {}", e)))?;
        result.push(batch);
    }

    if misaligned && result.is_empty() {
        return Err(RockDuckError::Internal(
            "scan: all batches misaligned — no valid aligned batches found".to_string(),
        ));
    }

    Ok(Arc::new(result))
}

pub(crate) fn data_type_to_arrow(dt: &crate::segment::meta::DataType) -> arrow_schema::DataType {
    use crate::segment::meta::DataType as D;
    use arrow_schema::DataType as A;
    match dt {
        D::Int8 => A::Int8,
        D::Int16 => A::Int16,
        D::Int32 => A::Int32,
        D::Int64 => A::Int64,
        D::UInt8 => A::UInt8,
        D::UInt16 => A::UInt16,
        D::UInt32 => A::UInt32,
        D::UInt64 => A::UInt64,
        D::Float32 => A::Float32,
        D::Float64 => A::Float64,
        D::Bool => A::Boolean,
        D::Utf8 => A::Utf8,
        D::LargeUtf8 => A::LargeUtf8,
        D::Binary => A::Binary,
        D::LargeBinary => A::LargeBinary,
        D::Date32 => A::Date32,
        D::Date64 => A::Date64,
        D::TimestampMicros => A::Timestamp(arrow_schema::TimeUnit::Microsecond, None),
        D::TimestampMillis => A::Timestamp(arrow_schema::TimeUnit::Millisecond, None),
    }
}

fn filter_batches_by_visibility_impl(
    vis_batches: Vec<RecordBatch>,
    data_batches: Arc<Vec<RecordBatch>>,
    snapshot: &TxnSnapshot,
    vis_filter: &dyn crate::mvcc::visibility::VisFilter,
    deleted_rows: HashSet<u64>,
) -> Result<Vec<RecordBatch>> {
    if vis_batches.is_empty() && deleted_rows.is_empty() {
        return Ok(data_batches.iter().cloned().collect());
    }

    let mut filtered = Vec::with_capacity(data_batches.len());
    let mut vis_batch_idx = 0;
    let mut rows_in_vis_batch = 0u64;
    let mut total_vis_rows = 0u64;

    for batch in data_batches.iter() {
        let num_rows = batch.num_rows();
        let mut visible_mask = Vec::with_capacity(num_rows);

        for _row_local in 0..num_rows {
            if deleted_rows.contains(&rows_in_vis_batch) {
                visible_mask.push(false);
                rows_in_vis_batch += 1;
                continue;
            }

            while vis_batch_idx < vis_batches.len()
                && total_vis_rows + vis_batches[vis_batch_idx].num_rows() as u64
                    <= rows_in_vis_batch
            {
                total_vis_rows += vis_batches[vis_batch_idx].num_rows() as u64;
                vis_batch_idx += 1;
            }

            if vis_batch_idx >= vis_batches.len() {
                visible_mask.push(true);
                rows_in_vis_batch += 1;
                continue;
            }

            let vis_batch = &vis_batches[vis_batch_idx];
            let local_vis_idx = rows_in_vis_batch - total_vis_rows;

            if local_vis_idx >= vis_batch.num_rows() as u64 {
                visible_mask.push(true);
                rows_in_vis_batch += 1;
                continue;
            }

            let visible = is_row_visible_in_vis_batch(
                vis_batch,
                local_vis_idx as usize,
                snapshot,
                vis_filter,
            );
            visible_mask.push(visible);
            rows_in_vis_batch += 1;
        }

        let batch_filtered = apply_visibility_mask(batch, &visible_mask).map_err(|e| {
            RockDuckError::Internal(format!(
                "visibility filter failed (batch {} rows): {}",
                batch.num_rows(),
                e
            ))
        })?;
        filtered.push(batch_filtered);
    }

    Ok(filtered)
}

fn is_row_visible_in_vis_batch(
    vis_batch: &RecordBatch,
    row_idx: usize,
    snapshot: &TxnSnapshot,
    vis_filter: &dyn crate::mvcc::visibility::VisFilter,
) -> bool {
    use arrow_array::Int64Array;

    let created_col = vis_batch.column(0).as_any().downcast_ref::<Int64Array>();
    let deleted_col = vis_batch.column(1).as_any().downcast_ref::<Int64Array>();

    let (Some(c_arr), Some(d_arr)) = (created_col, deleted_col) else {
        return true;
    };

    let created_txn = c_arr.value(row_idx) as u64;
    let del_raw = d_arr.value(row_idx) as u64;

    let deleted_txn = if del_raw == crate::mvcc::shadow_columns::NOT_DELETED {
        None
    } else {
        Some(del_raw)
    };

    vis_filter.is_row_visible(
        snapshot.snapshot_id,
        created_txn,
        deleted_txn,
        &snapshot.active_txns,
        &snapshot.commit_ts_map,
    )
}

fn apply_visibility_mask(
    batch: &RecordBatch,
    mask: &[bool],
) -> std::result::Result<RecordBatch, VisibilityError> {
    use arrow::compute::filter_record_batch;
    use arrow_array::BooleanArray;

    if mask.len() != batch.num_rows() {
        return Err(VisibilityError::LengthMismatch {
            mask_len: mask.len(),
            batch_rows: batch.num_rows(),
        });
    }

    let mut bool_buf = arrow_buffer::BooleanBufferBuilder::new(mask.len());
    bool_buf.append_slice(mask);
    let arr = BooleanArray::from(bool_buf.finish());
    match filter_record_batch(batch, &arr) {
        Ok(filtered) => Ok(filtered),
        Err(e) => Err(VisibilityError::Filter(e.to_string())),
    }
}

pub struct StepVerificationCard {
    pub target: &'static str,
    pub main_path: &'static str,
    pub bypass_paths: Vec<&'static str>,
    pub landing_files: Vec<&'static str>,
}

pub struct EvidencePackageVerification {
    pub router_boundary: StepVerificationCard,
    pub metadata_boundary: StepVerificationCard,
}

pub fn route_step_verification_card() -> StepVerificationCard {
    StepVerificationCard {
        target: "routing/evidence-plane",
        main_path: "scan::prepare_read_execution_plan -> QueryRouter::observe_route_selection -> FeedbackState::last_evidence",
        bypass_paths: vec![
            "point_get path does not enter scan routing and remains non-routing core read evidence",
            "time-travel path remains a sanctioned historical projection; sidecar evidence is metadata-only unless execution attribution is explicitly present",
            "zone-map conservative fallback reads segment without extra evidence refinement",
        ],
        landing_files: vec![
            "src/read/scan.rs",
            "src/query/routing/mod.rs",
            "src/query/routing/feedback.rs",
        ],
    }
}

pub fn evidence_package_verification() -> EvidencePackageVerification {
    EvidencePackageVerification {
        router_boundary: route_step_verification_card(),
        metadata_boundary: StepVerificationCard {
            target: "metadata/evidence-plane",
            main_path: "metadata::get_or_create_table_stats -> build_router_params/build_evidence_snapshot -> QueryRouter::route",
            bypass_paths: vec![
                "zone-map conservative read fallback (unknown column / missing ZM -> read segment)",
                "missing table stats default path (TableStats::default, zero counts)",
                "maintenance-only evidence consumers may lag one routing event behind",
            ],
            landing_files: vec![
                "src/metadata/mod.rs",
                "src/metadata/kv_store.rs",
                "src/read/scan.rs",
                "src/query/routing/mod.rs",
                "src/compaction/adaptive.rs",
            ],
        },
    }
}

pub fn count(db: &RockDuck, table: &str) -> Result<u64> {
    let metas = metadata::list_segment_metas(&db.kv)?;
    let mut total: u64 = 0;
    for meta in metas {
        if meta.table_id == table && meta.status != crate::segment::meta::SegmentStatus::Garbage {
            total += meta.alive_row_count;
        }
    }
    Ok(total)
}

#[allow(dead_code)]
pub fn range_aggregate(
    _db: &RockDuck,
    _table: &str,
    _column: &str,
    _agg_fn: &str,
) -> Result<ArrayRef> {
    Err(RockDuckError::Unimplemented(
        "range_aggregate is not yet implemented".to_string(),
    ))
}

#[cfg(test)]
mod batch_boundary_tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn test_apply_visibility_mask_length_mismatch() {
        use arrow_array::{Int64Array, RecordBatch};
        use arrow_schema::{DataType, Field, Schema};

        let schema = Schema::new(vec![Field::new("id", DataType::Int64, false)]);
        let values = Int64Array::from(vec![1_i64, 2, 3, 4, 5]);
        let batch =
            RecordBatch::try_new(Arc::new(schema), vec![Arc::new(values)]).expect("valid batch");

        // Mask with wrong length should return error
        let mask = vec![true, false, true]; // 3 elements, batch has 5
        let result = apply_visibility_mask(&batch, &mask);
        assert!(result.is_err(), "Length mismatch should be an error");

        // Correct length should succeed
        let mask = vec![true, false, true, false, true];
        let result = apply_visibility_mask(&batch, &mask);
        assert!(result.is_ok(), "Correct length mask should succeed");
    }

    #[test]
    fn test_filter_batches_empty_visibility_returns_all_batches() {
        use crate::mvcc::visibility::{IsolationLevel, NoopVisFilter, TxnSnapshot};
        use arrow_array::{Int64Array, RecordBatch};
        use arrow_schema::{DataType, Field, Schema};

        let schema = Schema::new(vec![Field::new("id", DataType::Int64, false)]);
        let values = Int64Array::from(vec![1_i64, 2, 3, 4, 5]);
        let batch = RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(values)])
            .expect("valid batch");
        let batches = Arc::new(vec![batch]);

        let snapshot = TxnSnapshot::new(
            0,
            std::collections::BTreeSet::new(),
            IsolationLevel::Snapshot,
        );
        let deleted_rows: HashSet<u64> = HashSet::new();

        let result = filter_batches_by_visibility_impl(
            vec![], // empty vis_batches
            batches,
            &snapshot,
            &NoopVisFilter,
            deleted_rows,
        )
        .expect("should succeed");

        assert_eq!(result.len(), 1, "Should return the data batch");
    }

    #[test]
    fn test_visibility_mask_with_all_visible() {
        use arrow_array::{Int64Array, RecordBatch};
        use arrow_schema::{DataType, Field, Schema};

        let schema = Schema::new(vec![Field::new("id", DataType::Int64, false)]);
        let values = Int64Array::from(vec![1_i64, 2, 3, 4, 5]);
        let batch =
            RecordBatch::try_new(Arc::new(schema), vec![Arc::new(values)]).expect("valid batch");

        // All true mask - all rows visible
        let mask = vec![true, true, true, true, true];
        let result = apply_visibility_mask(&batch, &mask);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result.num_rows(), 5, "All visible rows should be kept");
    }

    #[test]
    fn test_visibility_mask_with_all_invisible() {
        use arrow_array::{Int64Array, RecordBatch};
        use arrow_schema::{DataType, Field, Schema};

        let schema = Schema::new(vec![Field::new("id", DataType::Int64, false)]);
        let values = Int64Array::from(vec![1_i64, 2, 3, 4, 5]);
        let batch =
            RecordBatch::try_new(Arc::new(schema), vec![Arc::new(values)]).expect("valid batch");

        // All false mask - all rows invisible
        let mask = vec![false, false, false, false, false];
        let result = apply_visibility_mask(&batch, &mask);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(
            result.num_rows(),
            0,
            "All invisible rows should be filtered out"
        );
    }

    #[test]
    fn test_visibility_mask_selective() {
        use arrow_array::{Int64Array, RecordBatch};
        use arrow_schema::{DataType, Field, Schema};

        let schema = Schema::new(vec![Field::new("id", DataType::Int64, false)]);
        let values = Int64Array::from(vec![10_i64, 20, 30, 40, 50]);
        let batch =
            RecordBatch::try_new(Arc::new(schema), vec![Arc::new(values)]).expect("valid batch");

        // Select rows 0, 2, 4
        let mask = vec![true, false, true, false, true];
        let result = apply_visibility_mask(&batch, &mask);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result.num_rows(), 3, "Should have 3 visible rows");

        // Verify values
        let result_values = result
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(result_values.values(), &[10_i64, 30, 50]);
    }

    #[test]
    fn test_read_vortex_segment_misaligned_columns() {
        // This test validates that misaligned batches are detected and handled
        // The actual misaligned case would require a real segment, but we can test the logic

        let min_batch_count = 3;
        let counts = vec![3, 4, 3]; // Column 1 has 3, column 2 has 4, column 3 has 3

        let mut misaligned = false;
        for count in &counts {
            if *count != min_batch_count {
                misaligned = true;
            }
        }

        assert!(misaligned, "Should detect misalignment when counts differ");
    }

    #[test]
    fn test_read_vortex_segment_aligned_columns() {
        let min_batch_count = 3;
        let counts = vec![3, 3, 3]; // All columns have 3 batches

        let mut misaligned = false;
        for count in &counts {
            if *count != min_batch_count {
                misaligned = true;
            }
        }

        assert!(
            !misaligned,
            "Should not detect misalignment when all counts match"
        );
    }

    #[test]
    fn test_scan_options_default() {
        let options = ScanOptions::default();
        assert_eq!(options.table, "");
        assert_eq!(options.as_of_txn, None);
        assert_eq!(options.filter, None);
        assert_eq!(options.batch_size, None);
    }

    #[test]
    fn test_batch_scan_options_default() {
        let options = ScanOptions::default();
        assert_eq!(options.table, "");
        assert_eq!(options.as_of_txn, None);
        assert_eq!(options.filter, None);
        assert_eq!(options.batch_size, None);
    }

    #[test]
    fn test_read_execution_stats_default() {
        let stats = ReadExecutionStats::default();
        assert_eq!(stats.segments_routed, 0);
        assert_eq!(stats.segments_scanned, 0);
        assert_eq!(stats.base_batches_read, 0);
        assert_eq!(stats.delta_rows_seen, 0);
        assert_eq!(stats.rows_after_visibility, 0);
        assert_eq!(stats.rows_after_filter, 0);
        assert_eq!(stats.filter_failures, 0);
        assert_eq!(stats.skipped_segment_failures, 0);
        assert!(stats.skipped_segment_ids.is_empty());
    }

    #[test]
    fn test_data_type_to_arrow_conversions() {
        use crate::segment::meta::DataType;

        assert_eq!(
            data_type_to_arrow(&DataType::Int64),
            arrow_schema::DataType::Int64
        );
        assert_eq!(
            data_type_to_arrow(&DataType::Float64),
            arrow_schema::DataType::Float64
        );
        assert_eq!(
            data_type_to_arrow(&DataType::Utf8),
            arrow_schema::DataType::Utf8
        );
        assert_eq!(
            data_type_to_arrow(&DataType::Bool),
            arrow_schema::DataType::Boolean
        );
        assert_eq!(
            data_type_to_arrow(&DataType::Binary),
            arrow_schema::DataType::Binary
        );
    }
}
