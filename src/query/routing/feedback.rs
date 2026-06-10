//! Feedback state for the router -- selectivity tracking and execution history.
//!
//! Closes the feedback loop between query execution and routing decisions.
//!
//! ## Feedback Authority Tiers
//!
//! | Surface | Authority Tier | Actual Callers | Contract |
//! |---------|----------------|---------------|----------|
//! | `observe_route_selection` | **Derived** | `scan::prepare_read_execution_plan` | Records pre-execution routing intent and candidate evidence; may inform analysis and attribution, but not measured runtime correction on its own |
//! | `record_execution_outcome` | **Measured** | `scan::execute_read_plan` | Authoritative runtime correction surface for elapsed time, scanned work, and executed segment attribution |
//! | `observe_selectivity` | **Measured-lower-bound** | `QueryRouter::record_execution_outcome`, compatibility `record_feedback` | Runtime-derived lower-bound selectivity from returned rows vs routed candidate rows |
//! | `record_regret_sample` / `record_regret_feedback` | **Proxy** | `QueryRouter::record_execution_outcome`, compatibility `record_feedback` | Historical alternative-path averages only; never authoritative for live routing |
//! | `record_shadow_timing` | **Observation-only** | `scan::execute_read_plan` same-query shadow sampling | Same-query shadow store for analysis and bounded dual-path evidence; must not mutate routing authority directly |
//! | `record_ml_feedback` | **Stub / ephemeral** | `QueryRouter::record_execution_outcome` | Advisory-only comparison samples, excluded from checkpoint and blocked from authority |
//!
//! ## Selectivity Learning
//!
//! `SelectivityTracker` maintains a sliding-window average of observed selectivities
//! per column and per (col1, col2) pair. When a query executes, we observe:
//! actual_selectivity = actual_rows_returned / routed_candidate_rows
//!
//! This updates the tracker, so future queries on the same column get a better estimate.
//!
//! ## Execution History
//!
//! `ExecutionRecord` tracks the actual wall-clock time for each path (Delta / Vortex).
//! Used by Tier 2 to blend estimates with reality and by `CostParams::update()` to
//! calibrate the cost model online.
//!
//! Measured timing history intentionally excludes proxy regret records and shadow-only
//! samples from `avg_time()` so live routing correction remains bound to real execution.
//!
//! ## Persistence
//!
//! Selectivity and execution-history state are checkpointed to KV via `persist_to_kv()` / `load_from_kv()`.
//! Checkpoint is triggered by `QueryRouter::checkpoint_if_dirty()` on a timer.
//! `ml_collector` is intentionally excluded from `FeedbackCheckpoint`: ML samples remain
//! ephemeral and non-authoritative until durability, replay semantics, and evaluation
//! discipline are explicitly ratified. Shadow timing samples are the only extra durable
//! observability surface today, and they remain observation-only.
//!
//! ## Known Gaps
//!
//! - `record_regret_feedback`: **Proxy-only instrumentation**. The function is called from
//!   router feedback paths, but it uses historical average alternative-path timing rather than
//!   concurrent same-query dual-path measurement. Bounded dual-path shadow samples now exist via
//!   `record_shadow_timing`, so regret records remain explicitly non-authoritative fallback data.
//!
//! - `record_shadow_timing`: **Observation-only classified seam**. The storage surface now supports
//!   bounded same-query dual-path samples when `ShadowTimingPolicy::BoundedDualPath` is enabled,
//!   while `SyntheticEstimate` samples remain analysis-only. Both stay fenced away from direct
//!   routing authority and flow through `EvidenceQuality` gates before any cost blending.
//!
//! - `record_ml_feedback`: **Ephemeral advisory collector**. The function now has a real
//!   measured caller, but samples remain memory-only, excluded from `FeedbackCheckpoint`, and
//!   structurally fenced away from live authority. This surface exists for bounded ML evaluation,
//!   not for direct route ownership.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use crate::error::{Result, RockDuckError};
use crate::metadata::kv_engine::KVEngine;
use crate::query::routing::cost::CostParamsCheckpoint;
use crate::query::routing::RouteDecision;
use crate::read::scan::ExecutionOutcome;
/// Observation-only shadow timing policy.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize, PartialEq, Eq, Default)]
pub enum ShadowTimingPolicy {
    #[default]
    Disabled,
    SyntheticEstimate,
    BoundedDualPath,
}

impl ShadowTimingPolicy {
    pub fn is_authoritative(self) -> bool {
        false
    }

    pub fn is_measured(self) -> bool {
        matches!(self, Self::BoundedDualPath)
    }
}

/// Column key for selectivity tracking.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct SelectivityKey {
    pub table: String,
    pub column: String,
}

/// Joint selectivity key for multi-column predicates.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct JointKey {
    pub table: String,
    pub col1: String,
    pub col2: String,
}

/// A sliding window of f64 values with bounded memory.
#[derive(Debug, Clone)]
struct SlidingWindow {
    sum: f64,
    count: usize,
    values: VecDeque<f64>,
    max_size: usize,
}

impl SlidingWindow {
    fn new(max_size: usize) -> Self {
        Self {
            sum: 0.0,
            count: 0,
            values: VecDeque::with_capacity(max_size),
            max_size,
        }
    }

    fn push(&mut self, value: f64) {
        if self.values.len() >= self.max_size {
            if let Some(old) = self.values.pop_front() {
                self.sum -= old;
                self.count -= 1;
            }
        }
        self.values.push_back(value);
        self.sum += value;
        self.count += 1;
    }

    fn mean(&self) -> f64 {
        if self.count == 0 {
            0.1 // conservative default: 10% selectivity
        } else {
            self.sum / self.count as f64
        }
    }

    fn sample_count(&self) -> usize {
        self.count
    }
}

#[cfg(test)]
mod sliding_window_tests {
    use super::SlidingWindow;

    #[test]
    fn sliding_window_preserves_internal_invariants() {
        let mut window = SlidingWindow::new(3);
        for value in [0.2, 0.4, 0.6, 0.8] {
            window.push(value);
            let expected_sum: f64 = window.values.iter().sum();
            assert!(window.values.len() <= window.max_size);
            assert_eq!(window.count, window.values.len());
            assert!((window.sum - expected_sum).abs() < 1e-9);
        }
    }
}

/// Tracks per-column observed selectivities using a sliding window.
/// This is the core data structure for Tier 2's selectivity estimation.
pub struct SelectivityTracker {
    /// Per-column: sliding window of observed selectivities.
    columns: RwLock<HashMap<SelectivityKey, SlidingWindow>>,
    /// Per table pair: sliding window of joint selectivities.
    joints: RwLock<HashMap<JointKey, SlidingWindow>>,
    max_size: usize,
    default_selectivity: f64,
}

impl Clone for SelectivityTracker {
    fn clone(&self) -> Self {
        Self {
            columns: RwLock::new(self.columns.read().clone()),
            joints: RwLock::new(self.joints.read().clone()),
            max_size: self.max_size,
            default_selectivity: self.default_selectivity,
        }
    }
}

impl SelectivityTracker {
    pub fn new(max_size: usize) -> Self {
        Self {
            columns: RwLock::new(HashMap::new()),
            joints: RwLock::new(HashMap::new()),
            max_size,
            default_selectivity: 0.1,
        }
    }

    /// Observe a query result and update selectivity estimates.
    ///
    /// `actual_rows` = number of rows returned by the query.
    /// `total_rows` = number of rows in the scanned segment.
    pub fn observe(&self, table: &str, column: &str, actual_rows: u64, total_rows: u64) {
        if total_rows == 0 {
            return;
        }
        let selectivity = (actual_rows as f64 / total_rows as f64).clamp(0.0, 1.0);
        let key = SelectivityKey {
            table: table.to_string(),
            column: column.to_string(),
        };
        let mut cols = self.columns.write();
        let window = cols
            .entry(key)
            .or_insert_with(|| SlidingWindow::new(self.max_size));
        window.push(selectivity);
    }

    /// Observe a two-column predicate result.
    pub fn observe_joint(
        &self,
        table: &str,
        col1: &str,
        col2: &str,
        actual_rows: u64,
        total_rows: u64,
    ) {
        if total_rows == 0 {
            return;
        }
        let selectivity = (actual_rows as f64 / total_rows as f64).clamp(0.0, 1.0);
        let key = JointKey {
            table: table.to_string(),
            col1: col1.to_string(),
            col2: col2.to_string(),
        };
        let mut joints = self.joints.write();
        let window = joints
            .entry(key)
            .or_insert_with(|| SlidingWindow::new(self.max_size));
        window.push(selectivity);
    }

    /// Get the estimated selectivity for a column.
    /// Returns the sliding-window mean, or `default_selectivity` if no history.
    pub fn estimate(&self, table: &str, column: &str) -> f64 {
        let cols = self.columns.read();
        cols.get(&SelectivityKey {
            table: table.to_string(),
            column: column.to_string(),
        })
        .map(|w| w.mean())
        .unwrap_or(self.default_selectivity)
    }

    /// Get the number of observations for a column.
    pub fn sample_count(&self, table: &str, column: &str) -> usize {
        let cols = self.columns.read();
        cols.get(&SelectivityKey {
            table: table.to_string(),
            column: column.to_string(),
        })
        .map(|w| w.sample_count())
        .unwrap_or(0)
    }

    /// Get the estimated joint selectivity for two columns.
    pub fn estimate_joint(&self, table: &str, col1: &str, col2: &str) -> f64 {
        let joints = self.joints.read();
        joints
            .get(&JointKey {
                table: table.to_string(),
                col1: col1.to_string(),
                col2: col2.to_string(),
            })
            .map(|w| w.mean())
            .unwrap_or(self.default_selectivity)
    }

    /// Get per-column estimates for all known columns of a table.
    pub fn estimates_for_table(&self, table: &str) -> HashMap<String, f64> {
        let cols = self.columns.read();
        cols.iter()
            .filter(|(k, _)| k.table == table)
            .map(|(k, w)| (k.column.clone(), w.mean()))
            .collect()
    }

    /// Prune entries for tables that no longer exist.
    pub fn prune_for_tables(&self, alive_tables: &std::collections::HashSet<String>) {
        let mut cols = self.columns.write();
        cols.retain(|k, _| alive_tables.contains(&k.table));

        let mut joints = self.joints.write();
        joints.retain(|k, _| alive_tables.contains(&k.table));
    }

    /// Number of columns being tracked.
    pub fn len(&self) -> usize {
        self.columns.read().len()
    }

    /// Returns true if no columns are tracked.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Default for SelectivityTracker {
    fn default() -> Self {
        Self::new(1000)
    }
}

/// Record of a single query execution.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ExecutionRecord {
    pub table: String,
    pub chosen_path: RouteDecision,
    pub elapsed_ms: f64,
    pub estimated_cost: f64,
    pub rows_scanned: u64,
    pub candidate_rows: u64,
    pub rows_returned: u64,
    pub segments_routed: u64,
    pub segments_scanned: u64,
    pub template: Option<String>,
    pub executed_segment_ids: Vec<String>,
    pub alternative_path: Option<RouteDecision>,
    pub alternative_elapsed_ms: Option<f64>,
    pub is_shadow_timing: bool,
    pub is_measured_dual_path: bool,
}

/// Tracks recent query execution times per table and path.
#[derive(Default)]
pub struct ExecutionHistory {
    authoritative_records: RwLock<HashMap<String, Vec<ExecutionRecord>>>,
    proxy_records: RwLock<HashMap<String, Vec<ExecutionRecord>>>,
    max_per_table: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct MeasuredCostSample {
    pub avg_elapsed_ms: f64,
    pub sample_count: usize,
    pub quality: EvidenceQuality,
}

impl ExecutionHistory {
    pub fn new(max_per_table: usize) -> Self {
        Self {
            authoritative_records: RwLock::new(HashMap::new()),
            proxy_records: RwLock::new(HashMap::new()),
            max_per_table,
        }
    }

    fn push_record(
        records: &mut HashMap<String, Vec<ExecutionRecord>>,
        max_per_table: usize,
        rec: ExecutionRecord,
    ) {
        let entries = records.entry(rec.table.clone()).or_default();
        entries.push(rec);
        if entries.len() > max_per_table {
            let overflow = entries.len() - max_per_table;
            entries.drain(0..overflow);
        }
    }

    /// Record a query execution.
    pub fn record(&self, rec: ExecutionRecord) {
        if rec.is_shadow_timing || rec.alternative_path.is_some() {
            let mut records = self.proxy_records.write();
            Self::push_record(&mut records, self.max_per_table, rec);
        } else {
            let mut records = self.authoritative_records.write();
            Self::push_record(&mut records, self.max_per_table, rec);
        }
    }

    /// Get recent execution times for a table and path.
    pub fn recent_times(&self, table: &str, path: RouteDecision) -> Vec<f64> {
        let records = self.authoritative_records.read();
        records
            .get(table)
            .map(|v| {
                v.iter()
                    .filter(|r| {
                        r.chosen_path == path && !r.is_shadow_timing && r.alternative_path.is_none()
                    })
                    .map(|r| r.elapsed_ms)
                    .collect()
            })
            .unwrap_or_default()
    }

    pub fn measured_execution_count(&self, table: &str) -> usize {
        let records = self.authoritative_records.read();
        records
            .get(table)
            .map(|entries| {
                entries
                    .iter()
                    .filter(|record| {
                        !record.executed_segment_ids.is_empty() && !record.is_shadow_timing
                    })
                    .count()
            })
            .unwrap_or(0)
    }

    pub fn measured_cost_sample(
        &self,
        table: &str,
        path: RouteDecision,
    ) -> Option<MeasuredCostSample> {
        let authoritative = self.authoritative_records.read();
        let proxy = self.proxy_records.read();

        let authoritative_matches: Vec<&ExecutionRecord> = authoritative
            .get(table)
            .into_iter()
            .flat_map(|records| records.iter())
            .filter(|r| r.chosen_path == path)
            .collect();
        let bounded_dual_path: Vec<&ExecutionRecord> = proxy
            .get(table)
            .into_iter()
            .flat_map(|records| records.iter())
            .filter(|r| r.chosen_path == path && r.is_shadow_timing && r.is_measured_dual_path)
            .collect();
        let proxy_regret: Vec<&ExecutionRecord> = proxy
            .get(table)
            .into_iter()
            .flat_map(|records| records.iter())
            .filter(|r| {
                r.chosen_path == path
                    && !r.is_shadow_timing
                    && r.alternative_path.is_some()
                    && !r.is_measured_dual_path
            })
            .collect();

        if !authoritative_matches.is_empty() {
            let avg_elapsed_ms = authoritative_matches
                .iter()
                .map(|r| r.elapsed_ms)
                .sum::<f64>()
                / authoritative_matches.len() as f64;
            return Some(MeasuredCostSample {
                avg_elapsed_ms,
                sample_count: authoritative_matches.len(),
                quality: EvidenceQuality::Measured,
            });
        }

        if !bounded_dual_path.is_empty() {
            let avg_elapsed_ms = bounded_dual_path.iter().map(|r| r.elapsed_ms).sum::<f64>()
                / bounded_dual_path.len() as f64;
            return Some(MeasuredCostSample {
                avg_elapsed_ms,
                sample_count: bounded_dual_path.len(),
                quality: EvidenceQuality::BoundedDualPath,
            });
        }

        if proxy_regret.is_empty() {
            return None;
        }

        let avg_elapsed_ms =
            proxy_regret.iter().map(|r| r.elapsed_ms).sum::<f64>() / proxy_regret.len() as f64;
        Some(MeasuredCostSample {
            avg_elapsed_ms,
            sample_count: proxy_regret.len(),
            quality: EvidenceQuality::Synthetic,
        })
    }

    /// Get the average execution time for a table and path.
    pub fn avg_time(&self, table: &str, path: RouteDecision) -> Option<f64> {
        self.measured_cost_sample(table, path)
            .map(|sample| sample.avg_elapsed_ms)
    }
}

#[derive(Debug, Clone)]
pub struct ExecutionEvidenceSnapshot {
    pub table: String,
    pub query_columns: Vec<String>,
    pub total_segment_rows: u64,
    pub routed_segment_ids: Vec<String>,
    pub executed_segment_ids: Vec<String>,
    pub delta_segment_count: usize,
    pub has_zone_map_predicates: bool,
    pub has_cross_column_or: bool,
    pub table_row_count: u64,
    pub total_bytes: u64,
    pub compressed_bytes: u64,
    pub projection_contract: Option<crate::metadata::projection::ProjectionContract>,
}

impl ExecutionEvidenceSnapshot {
    pub fn compression_ratio(&self) -> f64 {
        if self.total_bytes == 0 {
            1.0
        } else {
            (self.compressed_bytes as f64 / self.total_bytes as f64).clamp(0.0, 1.0)
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvidenceQuality {
    Measured,
    BoundedDualPath,
    Proxy,
    Synthetic,
}

impl EvidenceQuality {
    pub fn allows_cost_blend(self) -> bool {
        matches!(self, Self::Measured | Self::BoundedDualPath)
    }

    pub fn is_measured_family(self) -> bool {
        matches!(self, Self::Measured | Self::BoundedDualPath)
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ShadowTimingSample {
    pub table: String,
    pub chosen_path: RouteDecision,
    pub shadow_path: RouteDecision,
    pub chosen_elapsed_ms: f64,
    pub shadow_elapsed_ms: f64,
    pub rows_scanned: u64,
    pub candidate_rows: u64,
    pub rows_returned: u64,
    pub segments_routed: u64,
    pub segments_scanned: u64,
    pub template: Option<String>,
    pub policy: ShadowTimingPolicy,
}

impl ShadowTimingSample {
    pub fn sampled_from_outcome(
        outcome: &ExecutionOutcome,
        shadow_path: RouteDecision,
        shadow_elapsed_ms: f64,
        policy: ShadowTimingPolicy,
    ) -> Option<Self> {
        let chosen_path = outcome.chosen_path?;
        Some(Self {
            table: outcome.table.clone(),
            chosen_path,
            shadow_path,
            chosen_elapsed_ms: outcome.elapsed_ms,
            shadow_elapsed_ms,
            rows_scanned: outcome.rows_scanned,
            candidate_rows: outcome.candidate_rows,
            rows_returned: outcome.rows_returned,
            segments_routed: outcome.segments_routed,
            segments_scanned: outcome.segments_scanned,
            template: Some(format!("{:?}", outcome.template)),
            policy,
        })
    }
}

/// Complete feedback state -- checkpointed to KV.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MlShadowEvaluationSample {
    pub table: String,
    pub tier2_decision: RouteDecision,
    pub ml_decision: RouteDecision,
    pub ml_confidence: f64,
    pub agreed_with_tier2: bool,
    pub observed_at_epoch_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MlPromotionSnapshot {
    pub shadow_samples: Vec<MlShadowEvaluationSample>,
    pub measured_training_sample_count: usize,
    pub export_batches_persisted: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SidecarDigestSnapshot {
    pub routed_segment_count: usize,
    pub executed_segment_count: usize,
    pub sidecar_class: String,
    pub surface: String,
    pub evidence_hook: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ExportDigestSnapshot {
    pub latest_status: String,
    pub file_count: usize,
    pub total_bytes: u64,
    pub mode: String,
    pub metadata_path: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ReplayDigestSnapshot {
    pub latest_status: String,
    pub accepted: u64,
    pub dropped: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SinkDigestSnapshot {
    pub latest_status: String,
    pub success_count: u64,
    pub failure_count: u64,
    pub latest_failure: Option<String>,
    pub sink_active: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GovernanceDigestSnapshot {
    pub blocking_assertion_count: u64,
}

#[derive(Default, serde::Serialize, serde::Deserialize)]
pub struct FeedbackCheckpoint {
    /// Serialized selectivity tracker state.
    pub selectivity_columns: HashMap<String, Vec<f64>>,
    pub selectivity_joints: HashMap<String, Vec<f64>>,
    /// Recent authoritative execution records by table.
    pub authoritative_execution_history: HashMap<String, Vec<ExecutionRecord>>,
    /// Recent proxy / shadow execution records by table.
    pub proxy_execution_history: HashMap<String, Vec<ExecutionRecord>>,
    /// Observation-only same-query shadow timings.
    pub shadow_timings: Vec<ShadowTimingSample>,
    /// Last cooperative runtime observations by table.
    pub cooperative_runtime: HashMap<String, crate::query::routing::CooperativeRuntimeSnapshot>,
    /// Latest sidecar evidence summaries by table.
    pub sidecar_digests: HashMap<String, SidecarDigestSnapshot>,
    /// Latest Iceberg export summaries by table.
    pub export_digests: HashMap<String, ExportDigestSnapshot>,
    /// Latest CDC replay summaries by table.
    pub replay_digests: HashMap<String, ReplayDigestSnapshot>,
    /// Latest CDC sink summaries by table.
    pub sink_digests: HashMap<String, SinkDigestSnapshot>,
    /// Latest governance assertion summaries by table.
    pub governance_digests: HashMap<String, GovernanceDigestSnapshot>,
    /// Session-adapted cost parameters that remain authority-safe.
    pub cost_params: Option<CostParamsCheckpoint>,
    /// Durable ML export batches collected under advisory mode.
    pub durable_ml_exports: Vec<MlExportBatch>,
    /// Bounded ML promotion evidence gathered in shadow mode.
    pub ml_promotion_snapshot: MlPromotionSnapshot,
    /// Last txn at which checkpoint was written.
    pub last_checkpoint_txn: u64,
}

/// Aggregated feedback state for the router.
///
/// Cloned cheaply via `Arc` and passed to `ScanIterator` so that
/// `QueryRouter` and `ScanIterator` can both access it without a shared reference.
#[derive(Clone)]
pub struct FeedbackState {
    pub selectivity: SelectivityTracker,
    pub execution: Arc<ExecutionHistory>,
    pub shadow_timings: Arc<RwLock<Vec<ShadowTimingSample>>>,
    /// Phase 10: ML routing feedback collector — stores (features, delta_ms, vortex_ms) tuples.
    pub ml_collector: Arc<QueryFeedbackCollector>,
    pub last_evidence: Arc<RwLock<HashMap<String, ExecutionEvidenceSnapshot>>>,
    pub last_cooperative_runtime:
        Arc<RwLock<HashMap<String, crate::query::routing::CooperativeRuntimeSnapshot>>>,
    pub last_sidecar_digest: Arc<RwLock<HashMap<String, SidecarDigestSnapshot>>>,
    pub last_export_digest: Arc<RwLock<HashMap<String, ExportDigestSnapshot>>>,
    pub last_replay_digest: Arc<RwLock<HashMap<String, ReplayDigestSnapshot>>>,
    pub last_sink_digest: Arc<RwLock<HashMap<String, SinkDigestSnapshot>>>,
    pub last_governance_digest: Arc<RwLock<HashMap<String, GovernanceDigestSnapshot>>>,
    pub cost_params_checkpoint: Arc<RwLock<Option<CostParamsCheckpoint>>>,
    pub durable_ml_exports: Arc<RwLock<Vec<MlExportBatch>>>,
    pub ml_promotion_snapshot: Arc<RwLock<MlPromotionSnapshot>>,
    checkpoint_key: String,
}

impl Default for FeedbackState {
    fn default() -> Self {
        Self {
            selectivity: SelectivityTracker::default(),
            execution: Arc::new(ExecutionHistory::default()),
            shadow_timings: Arc::new(RwLock::new(Vec::new())),
            ml_collector: Arc::new(QueryFeedbackCollector::default()),
            last_evidence: Arc::new(RwLock::new(HashMap::new())),
            last_cooperative_runtime: Arc::new(RwLock::new(HashMap::new())),
            last_sidecar_digest: Arc::new(RwLock::new(HashMap::new())),
            last_export_digest: Arc::new(RwLock::new(HashMap::new())),
            last_replay_digest: Arc::new(RwLock::new(HashMap::new())),
            last_sink_digest: Arc::new(RwLock::new(HashMap::new())),
            last_governance_digest: Arc::new(RwLock::new(HashMap::new())),
            cost_params_checkpoint: Arc::new(RwLock::new(None)),
            durable_ml_exports: Arc::new(RwLock::new(Vec::new())),
            ml_promotion_snapshot: Arc::new(RwLock::new(MlPromotionSnapshot::default())),
            checkpoint_key: "router_feedback_checkpoint".to_string(),
        }
    }
}

impl FeedbackState {
    pub fn new(window_size: usize, max_execution_records: usize) -> Self {
        Self {
            selectivity: SelectivityTracker::new(window_size),
            execution: Arc::new(ExecutionHistory::new(max_execution_records)),
            shadow_timings: Arc::new(RwLock::new(Vec::new())),
            ml_collector: Arc::new(QueryFeedbackCollector::default()),
            last_evidence: Arc::new(RwLock::new(HashMap::new())),
            last_cooperative_runtime: Arc::new(RwLock::new(HashMap::new())),
            last_sidecar_digest: Arc::new(RwLock::new(HashMap::new())),
            last_export_digest: Arc::new(RwLock::new(HashMap::new())),
            last_replay_digest: Arc::new(RwLock::new(HashMap::new())),
            last_sink_digest: Arc::new(RwLock::new(HashMap::new())),
            last_governance_digest: Arc::new(RwLock::new(HashMap::new())),
            cost_params_checkpoint: Arc::new(RwLock::new(None)),
            durable_ml_exports: Arc::new(RwLock::new(Vec::new())),
            ml_promotion_snapshot: Arc::new(RwLock::new(MlPromotionSnapshot::default())),
            checkpoint_key: "router_feedback_checkpoint".to_string(),
        }
    }

    /// Observe selectivity for multiple columns from a single query result.
    pub fn observe_selectivity(
        &self,
        table: &str,
        columns: &[String],
        actual_rows: u64,
        total_rows: u64,
    ) {
        if total_rows == 0 {
            return;
        }
        for col in columns {
            self.selectivity
                .observe(table, col, actual_rows, total_rows);
        }
        // Also track joint selectivity for pairs (if 2+ columns).
        if columns.len() >= 2 {
            self.selectivity.observe_joint(
                table,
                &columns[0],
                &columns[1],
                actual_rows,
                total_rows,
            );
        }
    }

    /// Record actual execution time for a completed query.
    pub fn record_execution_aggregate(
        &self,
        table: &str,
        path: RouteDecision,
        elapsed_ms: f64,
        estimated_cost: f64,
        rows_scanned: u64,
        rows_returned: u64,
    ) {
        let rec = ExecutionRecord {
            table: table.to_string(),
            chosen_path: path,
            elapsed_ms,
            estimated_cost,
            rows_scanned,
            candidate_rows: rows_scanned,
            rows_returned,
            segments_routed: 0,
            segments_scanned: 0,
            template: None,
            executed_segment_ids: Vec::new(),
            alternative_path: None,
            alternative_elapsed_ms: None,
            is_shadow_timing: false,
            is_measured_dual_path: false,
        };
        self.execution.record(rec);
    }

    pub fn record_execution_outcome(&self, outcome: &ExecutionOutcome, path: RouteDecision) {
        let rec = ExecutionRecord {
            table: outcome.table.clone(),
            chosen_path: path,
            elapsed_ms: outcome.elapsed_ms,
            estimated_cost: outcome.estimated_cost.unwrap_or(0.0),
            rows_scanned: outcome.rows_scanned,
            candidate_rows: outcome.candidate_rows,
            rows_returned: outcome.rows_returned,
            segments_routed: outcome.segments_routed,
            segments_scanned: outcome.segments_scanned,
            template: Some(format!("{:?}", outcome.template)),
            executed_segment_ids: outcome.executed_segment_ids.clone(),
            alternative_path: None,
            alternative_elapsed_ms: None,
            is_shadow_timing: false,
            is_measured_dual_path: false,
        };
        self.execution.record(rec);
    }

    pub fn observe_route_selection(
        &self,
        table: &str,
        path: RouteDecision,
        estimated_cost: f64,
        rows_scanned_upper_bound: u64,
        segments_routed: u64,
        evidence: &ExecutionEvidenceSnapshot,
    ) {
        let rec = ExecutionRecord {
            table: table.to_string(),
            chosen_path: path,
            elapsed_ms: 0.0,
            estimated_cost,
            rows_scanned: rows_scanned_upper_bound,
            candidate_rows: rows_scanned_upper_bound,
            rows_returned: 0,
            segments_routed,
            segments_scanned: 0,
            template: Some("route-selection".to_string()),
            executed_segment_ids: evidence.executed_segment_ids.clone(),
            alternative_path: None,
            alternative_elapsed_ms: None,
            is_shadow_timing: false,
            is_measured_dual_path: false,
        };
        self.execution.record(rec);
        self.last_evidence
            .write()
            .insert(table.to_string(), evidence.clone());
    }

    /// Record actual execution time for a completed query.
    pub fn record_execution_time(&self, table: &str, path: RouteDecision, elapsed_ms: f64) {
        self.record_execution_aggregate(table, path, elapsed_ms, 0.0, 0, 0);
    }

    pub fn record_regret_sample(
        &self,
        table: &str,
        chosen_path: RouteDecision,
        elapsed_ms: f64,
        alternative_path: RouteDecision,
        alternative_elapsed_ms: f64,
    ) {
        let rec = ExecutionRecord {
            table: table.to_string(),
            chosen_path,
            elapsed_ms,
            estimated_cost: 0.0,
            rows_scanned: 0,
            candidate_rows: 0,
            rows_returned: 0,
            segments_routed: 0,
            segments_scanned: 0,
            template: None,
            executed_segment_ids: Vec::new(),
            alternative_path: Some(alternative_path),
            alternative_elapsed_ms: Some(alternative_elapsed_ms),
            is_shadow_timing: false,
            is_measured_dual_path: false,
        };
        self.execution.record(rec);
    }

    pub fn record_shadow_timing(&self, sample: ShadowTimingSample) {
        if sample.shadow_path == sample.chosen_path {
            return;
        }
        assert!(
            !sample.policy.is_authoritative(),
            "observation-only shadow timing must never be promoted into authoritative routing evidence"
        );
        let shadow_record = ExecutionRecord {
            table: sample.table.clone(),
            chosen_path: sample.shadow_path,
            elapsed_ms: sample.shadow_elapsed_ms,
            estimated_cost: 0.0,
            rows_scanned: sample.rows_scanned,
            candidate_rows: sample.candidate_rows,
            rows_returned: sample.rows_returned,
            segments_routed: sample.segments_routed,
            segments_scanned: sample.segments_scanned,
            template: sample.template.clone(),
            executed_segment_ids: Vec::new(),
            alternative_path: Some(sample.chosen_path),
            alternative_elapsed_ms: Some(sample.chosen_elapsed_ms),
            is_shadow_timing: true,
            is_measured_dual_path: sample.policy.is_measured(),
        };
        self.execution.record(shadow_record);
        self.shadow_timings.write().push(sample);
    }

    pub fn shadow_timings(&self) -> Vec<ShadowTimingSample> {
        self.shadow_timings.read().clone()
    }

    /// Phase 10: Record ML routing feedback (features + actual delta/vortex times).
    /// This enables online model improvement when N >= 1000 samples.
    ///
    /// Classified Phase 4 status: stub surface with no live callers yet.
    #[allow(dead_code)]
    pub fn record_ml_sample(
        &self,
        features: crate::query::routing::ml::QueryFeatureVector,
        delta_ms: f64,
        vortex_ms: f64,
    ) {
        self.ml_collector.record(MlSample {
            features,
            delta_actual_ms: delta_ms,
            vortex_actual_ms: vortex_ms,
        });
    }

    pub fn record_cooperative_runtime(
        &self,
        table: &str,
        snapshot: crate::query::routing::CooperativeRuntimeSnapshot,
    ) {
        self.last_cooperative_runtime
            .write()
            .insert(table.to_string(), snapshot);
    }

    pub fn record_sidecar_digest(&self, table: &str, snapshot: SidecarDigestSnapshot) {
        self.last_sidecar_digest
            .write()
            .insert(table.to_string(), snapshot);
    }

    pub fn last_sidecar_digest(&self, table: &str) -> Option<SidecarDigestSnapshot> {
        self.last_sidecar_digest.read().get(table).cloned()
    }

    pub fn record_export_digest(&self, table: &str, snapshot: ExportDigestSnapshot) {
        self.last_export_digest
            .write()
            .insert(table.to_string(), snapshot);
    }

    pub fn last_export_digest(&self, table: &str) -> Option<ExportDigestSnapshot> {
        self.last_export_digest.read().get(table).cloned()
    }

    pub fn record_replay_digest(&self, table: &str, snapshot: ReplayDigestSnapshot) {
        self.last_replay_digest
            .write()
            .insert(table.to_string(), snapshot);
    }

    pub fn last_replay_digest(&self, table: &str) -> Option<ReplayDigestSnapshot> {
        self.last_replay_digest.read().get(table).cloned()
    }

    pub fn record_sink_digest(&self, table: &str, snapshot: SinkDigestSnapshot) {
        self.last_sink_digest
            .write()
            .insert(table.to_string(), snapshot);
    }

    pub fn last_sink_digest(&self, table: &str) -> Option<SinkDigestSnapshot> {
        self.last_sink_digest.read().get(table).cloned()
    }

    pub fn record_governance_digest(&self, table: &str, snapshot: GovernanceDigestSnapshot) {
        self.last_governance_digest
            .write()
            .insert(table.to_string(), snapshot);
    }

    pub fn last_governance_digest(&self, table: &str) -> Option<GovernanceDigestSnapshot> {
        self.last_governance_digest.read().get(table).cloned()
    }

    pub fn last_cooperative_runtime(
        &self,
        table: &str,
    ) -> Option<crate::query::routing::CooperativeRuntimeSnapshot> {
        self.last_cooperative_runtime.read().get(table).cloned()
    }

    pub fn set_cost_params_checkpoint(&self, checkpoint: CostParamsCheckpoint) {
        *self.cost_params_checkpoint.write() = Some(checkpoint);
    }

    pub fn cost_params_checkpoint(&self) -> Option<CostParamsCheckpoint> {
        self.cost_params_checkpoint.read().clone()
    }

    pub fn durable_ml_exports(&self) -> Vec<MlExportBatch> {
        self.durable_ml_exports.read().clone()
    }

    pub fn push_durable_ml_export(&self, batch: MlExportBatch) {
        self.durable_ml_exports.write().push(batch);
    }

    pub fn record_ml_shadow_evaluation(
        &self,
        sample: MlShadowEvaluationSample,
        max_samples: usize,
    ) {
        let mut snapshot = self.ml_promotion_snapshot.write();
        snapshot.shadow_samples.push(sample);
        if snapshot.shadow_samples.len() > max_samples {
            let drop_count = snapshot.shadow_samples.len() - max_samples;
            snapshot.shadow_samples.drain(0..drop_count);
        }
    }

    pub fn set_measured_ml_training_sample_count(&self, count: usize) {
        self.ml_promotion_snapshot
            .write()
            .measured_training_sample_count = count;
    }

    pub fn measured_ml_training_sample_count(&self) -> usize {
        self.ml_promotion_snapshot
            .read()
            .measured_training_sample_count
    }

    pub fn ml_promotion_snapshot(&self) -> MlPromotionSnapshot {
        self.ml_promotion_snapshot.read().clone()
    }

    pub fn ml_shadow_agreement_ratio(&self) -> Option<f64> {
        let snapshot = self.ml_promotion_snapshot.read();
        let total = snapshot.shadow_samples.len();
        if total == 0 {
            return None;
        }
        let agreed = snapshot
            .shadow_samples
            .iter()
            .filter(|sample| sample.agreed_with_tier2)
            .count();
        Some(agreed as f64 / total as f64)
    }

    pub fn ml_shadow_disagreement_ratio(&self) -> Option<f64> {
        self.ml_shadow_agreement_ratio().map(|ratio| 1.0 - ratio)
    }

    /// Load feedback state from a KV checkpoint.
    pub fn load_from_kv(kv: &Arc<dyn KVEngine>) -> Result<Self> {
        let kv_bucket = crate::metadata::CF_SYS;
        let ck_key = "router_feedback_checkpoint";
        let kv_key = ck_key.as_bytes();
        match kv.get(kv_bucket, kv_key) {
            Ok(Some(data)) => {
                match postcard::from_bytes::<FeedbackCheckpoint>(&data) {
                    Ok(cp) => {
                        tracing::info!(
                            "Loaded router feedback checkpoint ({} column stats, last_txn={})",
                            cp.selectivity_columns.len(),
                            cp.last_checkpoint_txn
                        );
                        let tracker = SelectivityTracker::default();
                        // Restore columns from checkpoint.
                        for (key_ser, values) in cp.selectivity_columns {
                            let mut cols = tracker.columns.write();
                            let parts: Vec<&str> = key_ser.splitn(2, ':').collect();
                            if parts.len() == 2 {
                                let key = SelectivityKey {
                                    table: parts[0].to_string(),
                                    column: parts[1].to_string(),
                                };
                                let mut window = SlidingWindow::new(1000);
                                for v in values {
                                    window.push(v);
                                }
                                cols.insert(key, window);
                            }
                        }
                        // Restore joints from checkpoint.
                        for (key_ser, values) in cp.selectivity_joints {
                            let mut joints = tracker.joints.write();
                            let parts: Vec<&str> = key_ser.splitn(3, ':').collect();
                            if parts.len() == 3 {
                                let key = JointKey {
                                    table: parts[0].to_string(),
                                    col1: parts[1].to_string(),
                                    col2: parts[2].to_string(),
                                };
                                let mut window = SlidingWindow::new(1000);
                                for v in values {
                                    window.push(v);
                                }
                                joints.insert(key, window);
                            }
                        }
                        let execution = Arc::new(ExecutionHistory::new(1000));
                        {
                            let mut authoritative = execution.authoritative_records.write();
                            *authoritative = cp.authoritative_execution_history;
                        }
                        {
                            let mut proxy = execution.proxy_records.write();
                            *proxy = cp.proxy_execution_history;
                        }
                        Ok(Self {
                            selectivity: tracker,
                            execution,
                            shadow_timings: Arc::new(RwLock::new(cp.shadow_timings)),
                            ml_collector: Arc::new(QueryFeedbackCollector::from_export_batches(
                                &cp.durable_ml_exports,
                                10_000,
                            )),
                            last_evidence: Arc::new(RwLock::new(HashMap::new())),
                            last_cooperative_runtime: Arc::new(RwLock::new(cp.cooperative_runtime)),
                            last_sidecar_digest: Arc::new(RwLock::new(cp.sidecar_digests)),
                            last_export_digest: Arc::new(RwLock::new(cp.export_digests)),
                            last_replay_digest: Arc::new(RwLock::new(cp.replay_digests)),
                            last_sink_digest: Arc::new(RwLock::new(cp.sink_digests)),
                            last_governance_digest: Arc::new(RwLock::new(cp.governance_digests)),
                            cost_params_checkpoint: Arc::new(RwLock::new(cp.cost_params)),
                            durable_ml_exports: Arc::new(RwLock::new(cp.durable_ml_exports)),
                            ml_promotion_snapshot: Arc::new(RwLock::new(cp.ml_promotion_snapshot)),
                            checkpoint_key: ck_key.to_string(),
                        })
                    }
                    Err(e) => {
                        tracing::warn!(
                            "Failed to deserialize feedback checkpoint: {}, starting fresh",
                            e
                        );
                        Ok(Self::default())
                    }
                }
            }
            Ok(None) => {
                tracing::debug!("No router feedback checkpoint found, starting fresh");
                Ok(Self::default())
            }
            Err(e) => {
                tracing::warn!(
                    "KV error loading feedback checkpoint: {}, starting fresh",
                    e
                );
                Ok(Self::default())
            }
        }
    }

    /// Persist feedback state to a KV checkpoint.
    pub fn persist_to_kv(&self, kv: &Arc<dyn KVEngine>, committed_txn: u64) -> Result<()> {
        let cols = self.selectivity.columns.read();
        let joints = self.selectivity.joints.read();

        let mut selectivity_columns = HashMap::new();
        for (key, window) in cols.iter() {
            let key_ser = format!("{}:{}", key.table, key.column);
            selectivity_columns.insert(key_ser, window.values.iter().copied().collect::<Vec<_>>());
        }

        let mut selectivity_joints = HashMap::new();
        for (key, window) in joints.iter() {
            let key_ser = format!("{}:{}:{}", key.table, key.col1, key.col2);
            selectivity_joints.insert(key_ser, window.values.iter().copied().collect::<Vec<_>>());
        }

        let cp_len = selectivity_columns.len();
        let authoritative_execution_history = self.execution.authoritative_records.read().clone();
        let proxy_execution_history = self.execution.proxy_records.read().clone();
        let shadow_timings = self.shadow_timings.read().clone();
        let cooperative_runtime = self.last_cooperative_runtime.read().clone();
        let sidecar_digests = self.last_sidecar_digest.read().clone();
        let export_digests = self.last_export_digest.read().clone();
        let replay_digests = self.last_replay_digest.read().clone();
        let sink_digests = self.last_sink_digest.read().clone();
        let governance_digests = self.last_governance_digest.read().clone();
        let cost_params = self.cost_params_checkpoint();
        let durable_ml_exports = self.durable_ml_exports();
        let ml_promotion_snapshot = self.ml_promotion_snapshot();
        let cp = FeedbackCheckpoint {
            selectivity_columns,
            selectivity_joints,
            authoritative_execution_history,
            proxy_execution_history,
            shadow_timings,
            cooperative_runtime,
            sidecar_digests,
            export_digests,
            replay_digests,
            sink_digests,
            governance_digests,
            cost_params,
            durable_ml_exports,
            ml_promotion_snapshot,
            // D12 fix: use the actual committed txn at checkpoint time.
            last_checkpoint_txn: committed_txn,
        };

        let data = postcard::to_allocvec(&cp)
            .map_err(|e| RockDuckError::Internal(format!("serialize feedback: {}", e)))?;

        let kv_bucket = crate::metadata::CF_SYS;
        let kv_key = self.checkpoint_key.as_bytes();
        kv.put(kv_bucket, kv_key, &data)
            .map_err(|e| RockDuckError::Write(format!("persist feedback: {}", e)))?;

        tracing::debug!("Router feedback checkpoint persisted ({} columns)", cp_len);
        Ok(())
    }
}

// =============================================================================
// ML Routing Feedback Collector — Phase 10
// =============================================================================

/// Sample for ML routing feedback: (query_features, delta_cost, vortex_cost).
/// Used to train/improve the ML model over time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MlExportSample {
    pub features: Vec<f32>,
    pub delta_actual_ms: f64,
    pub vortex_actual_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MlExportBatch {
    pub samples: Vec<MlExportSample>,
}

#[derive(Debug, Clone)]
pub struct MlSample {
    pub features: crate::query::routing::ml::QueryFeatureVector,
    pub delta_actual_ms: f64,
    pub vortex_actual_ms: f64,
}

impl MlSample {
    pub fn to_export_sample(&self) -> MlExportSample {
        MlExportSample {
            features: self.features.as_slice().to_vec(),
            delta_actual_ms: self.delta_actual_ms,
            vortex_actual_ms: self.vortex_actual_ms,
        }
    }
}

/// Collects ML routing samples for online model retraining.
///
/// When N >= `train_threshold` samples are collected, the collected data can be
/// used to retrain the ML model weights (e.g., via a Python script or a simple
/// online learner in Rust).
///
/// Samples are stored in a ring buffer to avoid unbounded memory growth.
pub struct QueryFeedbackCollector {
    samples: parking_lot::RwLock<Vec<MlSample>>,
    max_samples: usize,
}

impl QueryFeedbackCollector {
    pub fn new(max_samples: usize) -> Self {
        Self {
            samples: parking_lot::RwLock::new(Vec::with_capacity(max_samples)),
            max_samples,
        }
    }

    pub fn from_export_batches(batches: &[MlExportBatch], max_samples: usize) -> Self {
        let collector = Self::new(max_samples);
        for batch in batches {
            for sample in &batch.samples {
                collector.record(MlSample {
                    features: crate::query::routing::ml::QueryFeatureVector::from_slice(
                        &sample.features,
                    ),
                    delta_actual_ms: sample.delta_actual_ms,
                    vortex_actual_ms: sample.vortex_actual_ms,
                });
            }
        }
        collector
    }

    /// Add a new sample after query execution.
    pub fn record(&self, sample: MlSample) {
        let mut samples = self.samples.write();
        if samples.len() >= self.max_samples {
            // Ring buffer: drop the oldest half when full.
            let half = samples.len() / 2;
            samples.drain(0..half);
        }
        samples.push(sample);
    }

    /// Get all collected samples.
    pub fn samples(&self) -> Vec<MlSample> {
        self.samples.read().clone()
    }

    /// Number of samples currently stored.
    pub fn len(&self) -> usize {
        self.samples.read().len()
    }

    /// Returns true if no samples are stored.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns true when enough samples are available for retraining.
    pub fn ready_for_training(&self, threshold: usize) -> bool {
        self.len() >= threshold
    }

    /// Drain all samples (used during training export).
    pub fn drain(&self) -> Vec<MlSample> {
        let mut samples = self.samples.write();
        let drained: Vec<MlSample> = samples.drain(..).collect();
        drained
    }
}

impl Default for QueryFeedbackCollector {
    fn default() -> Self {
        Self::new(10_000)
    }
}

#[cfg(test)]
mod persistence_tests {
    use super::*;
    use crate::metadata::kv_engine::{KVIter, KVOp};
    use crate::query::routing::RouteDecision;

    #[test]
    fn measured_dual_path_samples_are_counted_but_proxy_regret_samples_are_not() {
        let history = ExecutionHistory::new(16);
        history.record(ExecutionRecord {
            table: "orders".to_string(),
            chosen_path: RouteDecision::DeltaStoreOnly,
            elapsed_ms: 12.0,
            estimated_cost: 0.0,
            rows_scanned: 10,
            candidate_rows: 10,
            rows_returned: 3,
            segments_routed: 1,
            segments_scanned: 1,
            template: None,
            executed_segment_ids: vec!["seg-1".to_string()],
            alternative_path: None,
            alternative_elapsed_ms: None,
            is_shadow_timing: false,
            is_measured_dual_path: false,
        });
        history.record(ExecutionRecord {
            table: "orders".to_string(),
            chosen_path: RouteDecision::DeltaStoreOnly,
            elapsed_ms: 18.0,
            estimated_cost: 0.0,
            rows_scanned: 10,
            candidate_rows: 10,
            rows_returned: 3,
            segments_routed: 1,
            segments_scanned: 1,
            template: None,
            executed_segment_ids: Vec::new(),
            alternative_path: Some(RouteDecision::VortexOnly),
            alternative_elapsed_ms: Some(19.0),
            is_shadow_timing: false,
            is_measured_dual_path: false,
        });
        history.record(ExecutionRecord {
            table: "orders".to_string(),
            chosen_path: RouteDecision::DeltaStoreOnly,
            elapsed_ms: 14.0,
            estimated_cost: 0.0,
            rows_scanned: 10,
            candidate_rows: 10,
            rows_returned: 3,
            segments_routed: 1,
            segments_scanned: 1,
            template: None,
            executed_segment_ids: Vec::new(),
            alternative_path: Some(RouteDecision::VortexOnly),
            alternative_elapsed_ms: Some(15.0),
            is_shadow_timing: true,
            is_measured_dual_path: true,
        });

        assert_eq!(
            history.recent_times("orders", RouteDecision::DeltaStoreOnly),
            vec![12.0]
        );
        let sample = history
            .measured_cost_sample("orders", RouteDecision::DeltaStoreOnly)
            .expect("authoritative samples must stay isolated from bounded dual-path evidence");
        assert_eq!(sample.sample_count, 1);
        assert_eq!(sample.quality, EvidenceQuality::Measured);
    }

    #[test]
    fn mixed_authoritative_and_bounded_dual_path_samples_do_not_co_mingle_into_cost_authority() {
        let history = ExecutionHistory::new(16);
        history.record(ExecutionRecord {
            table: "orders".to_string(),
            chosen_path: RouteDecision::DeltaStoreOnly,
            elapsed_ms: 12.0,
            estimated_cost: 0.0,
            rows_scanned: 10,
            candidate_rows: 10,
            rows_returned: 3,
            segments_routed: 1,
            segments_scanned: 1,
            template: None,
            executed_segment_ids: vec!["seg-1".to_string()],
            alternative_path: None,
            alternative_elapsed_ms: None,
            is_shadow_timing: false,
            is_measured_dual_path: false,
        });
        history.record(ExecutionRecord {
            table: "orders".to_string(),
            chosen_path: RouteDecision::DeltaStoreOnly,
            elapsed_ms: 14.0,
            estimated_cost: 0.0,
            rows_scanned: 10,
            candidate_rows: 10,
            rows_returned: 3,
            segments_routed: 1,
            segments_scanned: 1,
            template: None,
            executed_segment_ids: Vec::new(),
            alternative_path: Some(RouteDecision::VortexOnly),
            alternative_elapsed_ms: Some(15.0),
            is_shadow_timing: true,
            is_measured_dual_path: true,
        });

        let sample = history
            .measured_cost_sample("orders", RouteDecision::DeltaStoreOnly)
            .expect("authoritative samples should remain the live routing authority");
        assert_eq!(sample.quality, EvidenceQuality::Measured);
        assert!(sample.quality.allows_cost_blend());
    }

    #[test]
    fn measured_dual_path_only_samples_upgrade_to_bounded_measured_quality() {
        let history = ExecutionHistory::new(16);
        history.record(ExecutionRecord {
            table: "orders".to_string(),
            chosen_path: RouteDecision::DeltaStoreOnly,
            elapsed_ms: 14.0,
            estimated_cost: 0.0,
            rows_scanned: 10,
            candidate_rows: 10,
            rows_returned: 3,
            segments_routed: 1,
            segments_scanned: 1,
            template: None,
            executed_segment_ids: Vec::new(),
            alternative_path: Some(RouteDecision::VortexOnly),
            alternative_elapsed_ms: Some(15.0),
            is_shadow_timing: true,
            is_measured_dual_path: true,
        });

        let sample = history
            .measured_cost_sample("orders", RouteDecision::DeltaStoreOnly)
            .expect("expected bounded dual-path sample");
        assert_eq!(sample.sample_count, 1);
        assert_eq!(sample.quality, EvidenceQuality::BoundedDualPath);
        assert!(sample.quality.allows_cost_blend());
    }

    struct MemoryKv {
        inner: parking_lot::RwLock<HashMap<(String, Vec<u8>), Vec<u8>>>,
    }

    struct EmptyIter;

    impl KVIter for EmptyIter {
        fn next(&mut self) -> bool {
            false
        }
        fn key(&self) -> &[u8] {
            &[]
        }
        fn value(&self) -> &[u8] {
            &[]
        }
        fn seek(&mut self, _key: &[u8]) {}
    }

    impl MemoryKv {
        fn new() -> Self {
            Self {
                inner: parking_lot::RwLock::new(HashMap::new()),
            }
        }
    }

    impl KVEngine for MemoryKv {
        fn get(&self, bucket: &str, key: &[u8]) -> crate::error::Result<Option<Vec<u8>>> {
            Ok(self
                .inner
                .read()
                .get(&(bucket.to_string(), key.to_vec()))
                .cloned())
        }

        fn put(&self, bucket: &str, key: &[u8], value: &[u8]) -> crate::error::Result<()> {
            self.inner
                .write()
                .insert((bucket.to_string(), key.to_vec()), value.to_vec());
            Ok(())
        }

        fn delete(&self, bucket: &str, key: &[u8]) -> crate::error::Result<()> {
            self.inner
                .write()
                .remove(&(bucket.to_string(), key.to_vec()));
            Ok(())
        }

        fn prefix_iter(
            &self,
            _bucket: &str,
            _prefix: &[u8],
        ) -> crate::error::Result<Box<dyn KVIter>> {
            Ok(Box::new(EmptyIter))
        }

        fn write_batch(&self, bucket: &str, ops: &[KVOp]) -> crate::error::Result<()> {
            let mut inner = self.inner.write();
            for op in ops {
                match op {
                    KVOp::Put { key, value } => {
                        inner.insert((bucket.to_string(), key.clone()), value.clone());
                    }
                    KVOp::Delete { key } => {
                        inner.remove(&(bucket.to_string(), key.clone()));
                    }
                }
            }
            Ok(())
        }

        fn flush(&self) -> crate::error::Result<()> {
            Ok(())
        }

        fn atomic_increment(
            &self,
            bucket: &str,
            key: &[u8],
            delta: i64,
        ) -> crate::error::Result<i64> {
            let mut inner = self.inner.write();
            let composite = (bucket.to_string(), key.to_vec());
            let current = inner
                .get(&composite)
                .and_then(|bytes| String::from_utf8(bytes.clone()).ok())
                .and_then(|s| s.parse::<i64>().ok())
                .unwrap_or(0);
            let next = current + delta;
            inner.insert(composite, next.to_string().into_bytes());
            Ok(next)
        }
    }

    #[test]
    fn feedback_checkpoint_round_trips_execution_history_but_not_ml_samples() {
        let kv: Arc<dyn KVEngine> = Arc::new(MemoryKv::new());
        let feedback = FeedbackState::new(32, 32);

        feedback.observe_selectivity("orders", &["status".to_string()], 3, 10);
        feedback
            .selectivity
            .observe_joint("orders", "status", "region", 2, 10);
        feedback.record_execution_aggregate(
            "orders",
            RouteDecision::DeltaStoreOnly,
            12.5,
            7.0,
            20,
            3,
        );
        feedback.record_regret_sample(
            "orders",
            RouteDecision::DeltaStoreOnly,
            12.5,
            RouteDecision::VortexOnly,
            18.0,
        );

        let sample = crate::query::routing::ml::QueryFeatureVector::from_query(
            crate::query::routing::ml::OpType::PointGet,
            0.3,
            100,
            1,
            1,
            0,
        );
        feedback.record_ml_sample(sample, 12.0, 8.0);
        assert_eq!(feedback.ml_collector.samples().len(), 1);

        let shadow = ShadowTimingSample::sampled_from_outcome(
            &ExecutionOutcome {
                table: "orders".to_string(),
                query_columns: vec!["status".to_string()],
                template: crate::read::scan::ReadExecutionTemplate::DeltaOnly,
                chosen_path: Some(RouteDecision::DeltaStoreOnly),
                estimated_cost: Some(7.0),
                rows_scanned: 20,
                candidate_rows: 25,
                rows_returned: 20,
                rows_after_visibility: 20,
                filter_failures: 0,
                skipped_segment_failures: 0,
                skipped_segment_ids: Vec::new(),
                segments_routed: 1,
                segments_scanned: 1,
                elapsed_ms: 12.5,
                executed_segment_ids: vec!["seg-1".to_string()],
                cooperative_digest: None,
            },
            RouteDecision::VortexOnly,
            18.0,
            ShadowTimingPolicy::SyntheticEstimate,
        )
        .expect("chosen path available");
        feedback.record_shadow_timing(shadow);

        feedback.record_sidecar_digest(
            "orders",
            SidecarDigestSnapshot {
                routed_segment_count: 2,
                executed_segment_count: 1,
                sidecar_class: "SanctionedSidecar".to_string(),
                surface: "Vtab".to_string(),
                evidence_hook: "hook".to_string(),
            },
        );
        feedback.record_export_digest(
            "orders",
            ExportDigestSnapshot {
                latest_status: "ok".to_string(),
                file_count: 3,
                total_bytes: 4096,
                mode: "dual_track".to_string(),
                metadata_path: Some("metadata/v1.metadata.json".to_string()),
            },
        );
        feedback.record_replay_digest(
            "orders",
            ReplayDigestSnapshot {
                latest_status: "delta_only".to_string(),
                accepted: 7,
                dropped: 2,
            },
        );
        feedback.record_sink_digest(
            "orders",
            SinkDigestSnapshot {
                latest_status: "degraded".to_string(),
                success_count: 5,
                failure_count: 1,
                latest_failure: Some("topic unavailable".to_string()),
                sink_active: true,
            },
        );
        feedback.record_governance_digest(
            "orders",
            GovernanceDigestSnapshot {
                blocking_assertion_count: 9,
            },
        );

        feedback
            .persist_to_kv(&kv, 42)
            .expect("persist feedback checkpoint");
        let restored = FeedbackState::load_from_kv(&kv).expect("load feedback checkpoint");

        assert_eq!(restored.selectivity.estimate("orders", "status"), 0.3);
        assert_eq!(
            restored
                .last_sidecar_digest("orders")
                .expect("sidecar digest")
                .routed_segment_count,
            2
        );
        assert_eq!(
            restored
                .last_export_digest("orders")
                .expect("export digest")
                .file_count,
            3
        );
        assert_eq!(
            restored
                .last_replay_digest("orders")
                .expect("replay digest")
                .accepted,
            7
        );
        assert_eq!(
            restored
                .last_sink_digest("orders")
                .expect("sink digest")
                .failure_count,
            1
        );
        assert_eq!(
            restored
                .last_governance_digest("orders")
                .expect("governance digest")
                .blocking_assertion_count,
            9
        );
        assert_eq!(
            restored
                .selectivity
                .estimate_joint("orders", "status", "region"),
            0.2
        );
        assert_eq!(
            restored
                .execution
                .recent_times("orders", RouteDecision::DeltaStoreOnly),
            vec![12.5]
        );
        let restored_shadow = restored.shadow_timings();
        assert_eq!(restored_shadow.len(), 1);
        assert_eq!(restored_shadow[0].shadow_path, RouteDecision::VortexOnly);
        assert!(
            restored.ml_collector.samples().is_empty(),
            "ML collector should remain ephemeral"
        );
    }

    #[test]
    fn proxy_only_history_does_not_produce_avg_time() {
        let history = ExecutionHistory::new(16);
        history.record(ExecutionRecord {
            table: "orders".to_string(),
            chosen_path: RouteDecision::DeltaStoreOnly,
            elapsed_ms: 18.0,
            estimated_cost: 0.0,
            rows_scanned: 0,
            candidate_rows: 0,
            rows_returned: 0,
            segments_routed: 0,
            segments_scanned: 0,
            template: None,
            executed_segment_ids: Vec::new(),
            alternative_path: Some(RouteDecision::VortexOnly),
            alternative_elapsed_ms: Some(21.0),
            is_shadow_timing: false,
            is_measured_dual_path: false,
        });

        assert_eq!(
            history
                .measured_cost_sample("orders", RouteDecision::DeltaStoreOnly)
                .expect("proxy history should remain inspectable")
                .quality,
            EvidenceQuality::Synthetic
        );
        assert_eq!(
            history.avg_time("orders", RouteDecision::DeltaStoreOnly),
            Some(18.0)
        );
    }
}
