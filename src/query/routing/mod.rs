//! Router module - cost-model-driven HTAP read path selection
//!
//! This is the new unified router, superseding the legacy `router.rs` and
//! `router_demain.rs` files which are now deprecated.
//!
//! Execution protocol for every implementation step touching routing:
//! 1. Confirm the main path (`QueryRouter::route` -> `scan::route_table_segments`).
//! 2. Confirm bypass paths (`point_get`, `DuckDB/VTab`, time-travel special paths).
//! 3. Confirm landing files where decisions are consumed and persisted.
//!
//! ## Architecture
//!
//! Three-tier cascade:
//! - **Tier 1** (`tier1.rs`): Rule-based fast path - sub-millisecond, no statistics needed
//! - **Tier 2** (`tier2.rs`): Statistical cost model - scan cost = rows x (cpu + io)
//! - **Tier 3** (`ml.rs`): Tree-CNN ML routing - candle CPU inference for complex queries
//!
//! Feedback loop closes the gap between estimates and reality:
//! - `feedback.rs`: SelectivityTracker learns from actual query results
//! - `cost.rs`: CostParams auto-tuned via online regression (ACM-style)
//!
//! ## Feature Governance: Admission Rules
//!
//! Any new feature that touches routing, compaction, or data layout must pass these gates
//! before being merged. These are the **minimum** criteria — additional gates may be
//! required for specific feature categories.
//!
//! ### G1: Sanctioned Bypass Classification
//!
//! Every new read path or data access path must be classified as one of:
//! - **SANCTIONED bypass**: Known, documented, bounded risk. Proceed.
//! - **UNSANCTIONED gap**: Not yet bounded. **BLOCK** until gap is analyzed and documented.
//!
//!### G2: Truth Package Integrity
//!
//! Any change that introduces a new visibility surface must:
//! - Delegate to `VisFilter` trait, OR
//! - Be explicitly added to the sanctioned exception list with documented constraints, OR
//! - Be **BLOCKED** until the exception is ratified.
//!
//!### G3: Recovery Boundary
//!
//! Any change that introduces new physical state (new file types, new column families,
//! new index structures) must:
//! - Be recoverable from the existing CheckpointManager + WAL replay protocol, OR
//! - Extend the recovery protocol with a documented recovery path, OR
//! - Be **BLOCKED** until recovery is verified.
//!
//!### G4: Maintenance Signal Classification
//!
//! Any change that affects compaction signals must classify its signal into:
//! - **Debt signal**: Directly represents storage/read debt (e.g., `del_ratio`). Can fire rewrite.
//! - **Heuristic factor**: Proxy for benefit, not debt itself. Affects priority, not rewrite dispatch.
//! - **New**: Introduces a new type of signal not in the existing model. Requires design doc before merging.
//!
//!### G5: Feedback Surface Verification
//!
//! Any new feedback path (selectivity, regret, ML) must have at least one real caller
//! before merging. Stub implementations without callers must be marked with `#[allow(dead_code)]`
//! and a tracking issue link.
//!
//!### G6: Sidecar Classification
//!
//! Any external integration (Iceberg, CDC, DuckDB extension, ML) must be classified:
//! - **Core**: Required for DB correctness. Any regression blocks the DB from opening.
//! - **Optional**: Can degrade gracefully. Failures must not propagate to core DB state.
//! - Hybrid persistence: memory + KV checkpoint
//!
//! ## Usage
//!
//! ```ignore
//! let result = router.route(&query, &stats, &feedback);
//! match result.path {
//!     RouteDecision::DeltaStoreOnly { .. } => { /* row-store path */ }
//!     RouteDecision::VortexOnly => { /* columnar scan path */ }
//!     RouteDecision::Merge { .. } => { /* delta overlay + vortex */ }
//! }
//! ```

pub mod config;
pub mod cost;
pub mod feedback;
pub mod ml;
pub mod plan_encoder;
pub mod tier1;
pub mod tier2;

use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::metadata::kv_engine::KVEngine;
use crate::metadata::projection::ProjectionContract;
use crate::query::routing::config::RouterConfig;
use crate::query::routing::cost::CostParams;
use crate::query::routing::feedback::{
    ExecutionEvidenceSnapshot, ExportDigestSnapshot, FeedbackState, GovernanceDigestSnapshot,
    MlExportBatch, MlSample, MlShadowEvaluationSample, ReplayDigestSnapshot, SidecarDigestSnapshot,
    SinkDigestSnapshot,
};
use crate::query::routing::ml::TreeCnnRouter;
use crate::query::routing::plan_encoder::PlanEncoder;
use crate::query::routing::tier1::RuleRouter;
use crate::query::routing::tier2::StatisticalRouter;

use crate::read::scan::ExecutionOutcome;
use arrow_schema::DataType;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CheckpointHotspotStats {
    pub calls: u64,
    pub writes: u64,
    pub skips: u64,
}

/// Routing decision returned by all tiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum RouteDecision {
    /// Pure point/short-range query - only read DeltaStore (row delta overlay).
    /// Fast for recent writes, no columnar decompression.
    DeltaStoreOnly,
    /// Large aggregation / full scan - only read Vortex (columnar storage).
    /// Efficient for scanning large historical datasets.
    VortexOnly,
    /// Mixed workload - DeltaStore overlay merged with Vortex results.
    /// Necessary when recent writes are in Delta and historical data is in Vortex.
    Merge,
}

/// Metadata about the chosen routing path.
#[derive(Debug, Clone)]
pub struct RoutingResult {
    /// Which storage path was chosen.
    pub decision: RouteDecision,
    /// Confidence score in [0.0, 1.0].
    /// - 1.0 = Tier 1 rule, completely confident
    /// - 0.8 = Tier 2 cost model
    /// - < 0.8 = Tier 3 ML (may still be overridden by config threshold)
    pub confidence: f64,
    /// Human-readable explanation of why this path was chosen.
    pub reasoning: String,
    /// Estimated cost of the chosen path (relative units).
    pub estimated_cost: f64,
    /// Which tier made the decision: "tier1_rule", "tier2_cost", "tier3_ml".
    pub tier: &'static str,
}

/// Query type classification used throughout the router.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryKind {
    /// Primary-key point get - always goes to DeltaStore.
    PointGet,
    /// Range scan with a filter - the main routing battleground.
    RangeScan,
    /// Full table scan without filters - Vortex is almost always better.
    FullScan,
    /// Aggregation (COUNT, SUM, AVG, etc.) - depends on selectivity.
    Aggregate,
}

impl QueryKind {
    pub fn from_filter(has_filter: bool, is_aggregate: bool, selectivity: f64) -> Self {
        if is_aggregate {
            Self::Aggregate
        } else if !has_filter {
            Self::FullScan
        } else if selectivity < 0.01 {
            Self::PointGet
        } else {
            Self::RangeScan
        }
    }
}

/// Unified query router - all tiers + feedback state + config.
pub struct QueryRouter {
    pub config: Arc<RouterConfig>,
    pub cost_params: Arc<RwLock<CostParams>>,

    tier1: RuleRouter,
    tier2: StatisticalRouter,
    ml_router: RwLock<Option<TreeCnnRouter>>,

    feedback: Arc<FeedbackState>,
    plan_encoder: PlanEncoder,

    /// Whether the feedback state has changed since the last checkpoint.
    /// Changed from RwLock<bool> to AtomicBool to avoid deadlock.
    dirty: AtomicBool,
    checkpoint_hotspot_stats: RwLock<CheckpointHotspotStats>,
    last_ml_export: RwLock<Vec<MlSample>>,
    shadow_sample_epoch: AtomicU64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MlRoutingMode {
    Disabled,
    AdvisoryShadow,
    LiveAuthority,
}

impl MlRoutingMode {
    pub fn allows_live_authority(self) -> bool {
        matches!(self, Self::LiveAuthority)
    }
}

#[cfg(test)]
struct TestMemoryKv {
    inner: parking_lot::RwLock<HashMap<(String, Vec<u8>), Vec<u8>>>,
}

#[cfg(test)]
struct TestEmptyIter;

#[cfg(test)]
impl crate::metadata::kv_engine::KVIter for TestEmptyIter {
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

#[cfg(test)]
impl TestMemoryKv {
    fn new() -> Self {
        Self {
            inner: parking_lot::RwLock::new(HashMap::new()),
        }
    }
}

#[cfg(test)]
impl crate::metadata::kv_engine::KVEngine for TestMemoryKv {
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
    ) -> crate::error::Result<Box<dyn crate::metadata::kv_engine::KVIter>> {
        Ok(Box::new(TestEmptyIter))
    }

    fn write_batch(
        &self,
        bucket: &str,
        ops: &[crate::metadata::kv_engine::KVOp],
    ) -> crate::error::Result<()> {
        let mut inner = self.inner.write();
        for op in ops {
            match op {
                crate::metadata::kv_engine::KVOp::Put { key, value } => {
                    inner.insert((bucket.to_string(), key.clone()), value.clone());
                }
                crate::metadata::kv_engine::KVOp::Delete { key } => {
                    inner.remove(&(bucket.to_string(), key.clone()));
                }
            }
        }
        Ok(())
    }

    fn flush(&self) -> crate::error::Result<()> {
        Ok(())
    }

    fn atomic_increment(&self, bucket: &str, key: &[u8], delta: i64) -> crate::error::Result<i64> {
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

#[derive(Debug, Clone)]
pub struct SubagentVerificationTemplate {
    pub goal: &'static str,
    pub required_output: [&'static str; 3],
    pub stop_conditions: [&'static str; 3],
}

#[derive(Debug, Clone)]
pub struct PackageVerificationTemplates {
    pub call_boundary: SubagentVerificationTemplate,
    pub authority_state: SubagentVerificationTemplate,
    pub exception_path: SubagentVerificationTemplate,
}

#[derive(Debug, Clone)]
pub struct GovernancePosture {
    pub current_reality: &'static str,
    pub ascension_target: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CooperativeRuntimeBudget {
    pub max_segments_per_slice: usize,
    pub max_slice_ms: u64,
}

impl CooperativeRuntimeBudget {
    pub fn is_bounded(self) -> bool {
        self.max_segments_per_slice > 0 || self.max_slice_ms > 0
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CooperativeRuntimeSnapshot {
    pub budget: Option<CooperativeRuntimeBudget>,
    pub slices_executed: u64,
    pub truncated_segments: Vec<String>,
    pub elapsed_budget_exhausted: bool,
    pub max_segments_per_slice_hit: bool,
}

#[derive(Debug, Clone)]
pub struct SidecarEvidenceSnapshot {
    pub table: String,
    pub routed_segment_ids: Vec<String>,
    pub executed_segment_ids: Vec<String>,
    pub contract: ProjectionContract,
}

#[derive(Debug, Clone)]
pub struct QueryRouterStatusDigest {
    pub governance: GovernanceDigestSnapshot,
    pub sidecar: Option<SidecarDigestSnapshot>,
    pub replay: Option<ReplayDigestSnapshot>,
    pub sink: Option<SinkDigestSnapshot>,
    pub export: Option<ExportDigestSnapshot>,
    pub ml_disagreement_ratio: Option<f64>,
    pub cooperative_runtime: Option<CooperativeRuntimeSnapshot>,
    pub ml_mode: MlRoutingMode,
    pub ml_disabled_reason: Option<String>,
    pub export_runtime_enabled: bool,
    pub export_external_reader_compat_required: bool,
    pub kafka_runtime_enabled: bool,
}

impl QueryRouter {
    pub fn status_digest(&self, table: &str) -> QueryRouterStatusDigest {
        let ml_disabled_reason = self
            .ml_router
            .read()
            .as_ref()
            .and_then(|router| router.disabled_reason.clone());
        QueryRouterStatusDigest {
            governance: self
                .feedback
                .last_governance_digest(table)
                .unwrap_or_default(),
            sidecar: self.feedback.last_sidecar_digest(table),
            replay: self.feedback.last_replay_digest(table),
            sink: self.feedback.last_sink_digest(table),
            export: self.feedback.last_export_digest(table),
            ml_disagreement_ratio: self.feedback.ml_shadow_disagreement_ratio(),
            cooperative_runtime: self.feedback.last_cooperative_runtime(table),
            ml_mode: self.ml_routing_mode(),
            ml_disabled_reason,
            export_runtime_enabled: self.config.export_rollout.runtime_enabled,
            export_external_reader_compat_required: self
                .config
                .export_rollout
                .external_reader_compat_required,
            kafka_runtime_enabled: self.config.cdc_sink_rollout.kafka_runtime_enabled,
        }
    }

    pub fn cooperative_runtime_budget(
        &self,
        table: &str,
        routed_segments: usize,
    ) -> Option<CooperativeRuntimeBudget> {
        let last = self.feedback.last_cooperative_runtime(table)?;
        let budget = last.budget?;
        let observed_segments = last.truncated_segments.len();
        let route_size = routed_segments.max(1);
        let capped_segments = if last.max_segments_per_slice_hit {
            budget.max_segments_per_slice.max(1).min(route_size)
        } else {
            budget
                .max_segments_per_slice
                .max(route_size.min(4))
                .min(route_size)
        };
        let bounded_ms = if last.elapsed_budget_exhausted {
            budget.max_slice_ms.max(1)
        } else {
            budget.max_slice_ms.max(1).saturating_mul(2).min(250)
        };
        let adaptive_segments = if observed_segments == 0 {
            capped_segments
        } else {
            capped_segments.min(observed_segments.max(1))
        };
        Some(CooperativeRuntimeBudget {
            max_segments_per_slice: adaptive_segments.max(1),
            max_slice_ms: bounded_ms.max(1),
        })
    }

    pub fn should_sample_shadow_timing(
        &self,
        table: &str,
        query_columns: &[String],
        chosen_path: RouteDecision,
        routed_segments: usize,
    ) -> bool {
        let threshold = (self.config.shadow_sample_rate.clamp(0.0, 1.0) * 10_000.0).round() as u64;
        if threshold == 0 {
            return false;
        }
        let epoch = self.shadow_sample_epoch.fetch_add(1, Ordering::Relaxed);
        let mut hasher = DefaultHasher::new();
        table.hash(&mut hasher);
        query_columns.hash(&mut hasher);
        chosen_path.hash(&mut hasher);
        routed_segments.hash(&mut hasher);
        epoch.hash(&mut hasher);
        (hasher.finish() % 10_000) < threshold
    }

    pub fn ml_routing_mode(&self) -> MlRoutingMode {
        if !self.config.ml_enabled {
            MlRoutingMode::Disabled
        } else if self.config.ml_promotion_gate.live_authority_enabled {
            MlRoutingMode::LiveAuthority
        } else {
            MlRoutingMode::AdvisoryShadow
        }
    }

    pub fn export_runtime_enabled(&self) -> bool {
        self.config.export_rollout.runtime_enabled
    }

    pub fn export_external_reader_compat_required(&self) -> bool {
        self.config.export_rollout.external_reader_compat_required
    }

    pub fn kafka_runtime_enabled(&self) -> bool {
        self.config.cdc_sink_rollout.kafka_runtime_enabled
    }

    pub fn feedback(&self) -> &Arc<FeedbackState> {
        &self.feedback
    }

    pub fn call_boundary_template() -> SubagentVerificationTemplate {
        SubagentVerificationTemplate {
            goal: "调用边界确认",
            required_output: ["Files", "ShortestChain", "MissingConfirmations"],
            stop_conditions: ["主链明确", "旁路明确", "落地面明确"],
        }
    }

    pub fn authority_state_template() -> SubagentVerificationTemplate {
        SubagentVerificationTemplate {
            goal: "权威状态确认",
            required_output: ["Files", "VerifiedAssumptions", "MissingConfirmations"],
            stop_conditions: ["source of truth 明确", "覆盖顺序明确", "恢复顺序明确"],
        }
    }

    pub fn exception_path_template() -> SubagentVerificationTemplate {
        SubagentVerificationTemplate {
            goal: "例外路径确认",
            required_output: ["Files", "ShortestChain", "MissingConfirmations"],
            stop_conditions: [
                "bypass 明确",
                "fallback 明确",
                "special-case semantics 明确",
            ],
        }
    }

    pub fn verification_templates() -> PackageVerificationTemplates {
        PackageVerificationTemplates {
            call_boundary: Self::call_boundary_template(),
            authority_state: Self::authority_state_template(),
            exception_path: Self::exception_path_template(),
        }
    }
    pub fn governance_posture() -> GovernancePosture {
        GovernancePosture {
            current_reality: "routing governance is currently expressed as classified verification templates plus documentation-backed hint-layer checks",
            ascension_target: "routing governance graduates to layered enforcement only after execution templates, feedback outcomes, and sidecar classifications are structurally enforced",
        }
    }

    /// Open a router from an existing KV store, restoring feedback state from checkpoint.
    pub fn open(kv: &Arc<dyn KVEngine>, config: RouterConfig) -> Result<Self> {
        let feedback = FeedbackState::load_from_kv(kv)?;
        let mut cost_params = CostParams::from_config(&config);
        if let Some(checkpoint) = feedback.cost_params_checkpoint() {
            cost_params.apply_checkpoint(&checkpoint);
        }

        let ml_router = if config.ml_enabled {
            match TreeCnnRouter::new(&config) {
                Ok(r) => Some(r),
                Err(e) => {
                    tracing::warn!(
                        "Tree-CNN ML router failed to load, disabling ML tier: {}",
                        e
                    );
                    None
                }
            }
        } else {
            None
        };

        Ok(Self {
            config: Arc::new(config.clone()),
            cost_params: Arc::new(RwLock::new(cost_params)),
            tier1: RuleRouter::new(),
            tier2: StatisticalRouter::new(),
            ml_router: RwLock::new(ml_router),
            feedback: Arc::new(feedback),
            plan_encoder: PlanEncoder::new(),
            dirty: AtomicBool::new(false),
            checkpoint_hotspot_stats: RwLock::new(CheckpointHotspotStats::default()),
            last_ml_export: RwLock::new(Vec::new()),
            shadow_sample_epoch: AtomicU64::new(0),
        })
    }

    /// Create a router with default configuration (for tests).
    pub fn default_for_test() -> Self {
        let config = RouterConfig::default();
        let cost_params = CostParams::from_config(&config);
        Self {
            config: Arc::new(config),
            cost_params: Arc::new(RwLock::new(cost_params)),
            tier1: RuleRouter::new(),
            tier2: StatisticalRouter::new(),
            ml_router: RwLock::new(None),
            feedback: Arc::new(FeedbackState::default()),
            plan_encoder: PlanEncoder::new(),
            dirty: AtomicBool::new(false),
            checkpoint_hotspot_stats: RwLock::new(CheckpointHotspotStats::default()),
            last_ml_export: RwLock::new(Vec::new()),
            shadow_sample_epoch: AtomicU64::new(0),
        }
    }

    /// Make a routing decision for the given query.
    ///
    /// Tiers cascade: Tier 1 -> Tier 2 -> Tier 3. Each tier returns `Some` if it can
    /// make a confident decision, falling through to the next tier otherwise.
    pub fn route(&self, params: &RouterParams) -> RoutingResult {
        let cfg = &self.config;

        // ---- Tier 1: Rule fast path ----
        if let Some(decision) = self.tier1.decide(params, cfg) {
            return RoutingResult {
                decision,
                confidence: 1.0,
                reasoning: format!("tier1: matched rule for {:?}", params.kind),
                estimated_cost: 0.0,
                tier: "tier1_rule",
            };
        }

        // ---- Tier 2: Statistical cost model ----
        let sel_tracker = &self.feedback.selectivity;
        let exec_feedback = &self.feedback;
        let cost_guard = self.cost_params.read();
        let tier2_result = self
            .tier2
            .decide(params, cfg, sel_tracker, exec_feedback, &cost_guard);
        drop(cost_guard);

        // If Tier 2 is very confident (> 0.85), return it without consulting Tier 3.
        if tier2_result.confidence > 0.85 {
            return tier2_result;
        }

        // ---- Tier 3: ML routing ----
        let ml_guard = self.ml_router.read();
        if let Some(ref ml) = *ml_guard {
            if cfg.ml_enabled {
                let plan_vec = self.plan_encoder.encode(params);
                let ml_result = ml.predict(&plan_vec);

                let confidence = {
                    let has_any = ml_result.delta_score > 0.0
                        || ml_result.vortex_score > 0.0
                        || ml_result.merge_score > 0.0;
                    if !has_any {
                        0.0
                    } else {
                        let sum =
                            ml_result.delta_score + ml_result.vortex_score + ml_result.merge_score;
                        if sum <= 0.0 {
                            0.0
                        } else {
                            let max_score = ml_result
                                .delta_score
                                .max(ml_result.vortex_score)
                                .max(ml_result.merge_score);
                            let min_score = ml_result
                                .delta_score
                                .min(ml_result.vortex_score)
                                .min(ml_result.merge_score);
                            (max_score - min_score) / sum
                        }
                    }
                };

                self.feedback.record_ml_shadow_evaluation(
                    MlShadowEvaluationSample {
                        table: params.table.to_string(),
                        tier2_decision: tier2_result.decision,
                        ml_decision: ml_result.routing_path,
                        ml_confidence: confidence as f64,
                        agreed_with_tier2: ml_result.routing_path == tier2_result.decision,
                        observed_at_epoch_ms: std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .map(|duration| duration.as_millis() as u64)
                            .unwrap_or(0),
                    },
                    self.config.ml_promotion_gate.shadow_sample_window.max(1),
                );

                tracing::debug!(
                    table = params.table,
                    ml_mode = ?self.ml_routing_mode(),
                    tier2_decision = ?tier2_result.decision,
                    ml_decision = ?ml_result.routing_path,
                    ml_confidence = confidence,
                    "tier3 ML evaluated in advisory mode"
                );

                if let Some(override_decision) = ml.live_override_decision(
                    &self.config.ml_promotion_gate,
                    &self.feedback,
                    &ml_result,
                ) {
                    if override_decision.confidence >= cfg.ml_confidence_thresh {
                        drop(ml_guard);
                        return RoutingResult {
                            decision: override_decision.routing_path,
                            confidence: override_decision.confidence,
                            reasoning: format!(
                                "tier3_ml_live_authority: disagreement_ratio={:.3}",
                                override_decision.disagreement_ratio
                            ),
                            estimated_cost: tier2_result.estimated_cost,
                            tier: "tier3_ml_live_authority",
                        };
                    }
                }
            }
        }
        drop(ml_guard);

        // ---- Fallback: Tier 2 result ----
        tier2_result
    }

    /// Record an observed route selection before execution completes.
    /// This preserves the current segment-pruning evidence without pretending it is a
    /// full execution outcome.
    pub fn observe_route_selection(
        &self,
        table: &str,
        routed_segments: &[crate::segment::meta::SegmentMeta],
        routing: &RoutingResult,
        evidence: &ExecutionEvidenceSnapshot,
    ) {
        self.feedback.observe_route_selection(
            table,
            routing.decision,
            routing.estimated_cost,
            routed_segments.iter().map(|m| m.row_count).sum(),
            routed_segments.len() as u64,
            evidence,
        );
        self.dirty.store(true, Ordering::Relaxed);
    }

    pub fn observe_maintenance_evidence(
        &self,
        db: &crate::db::RockDuck,
        evidence: &ExecutionEvidenceSnapshot,
    ) {
        db.compaction_scheduler.write().observe_evidence(evidence);
    }

    pub fn observe_metadata_evidence(
        &self,
        db: &crate::db::RockDuck,
        evidence: &crate::metadata::EvidenceSnapshot,
    ) {
        evidence.assert_governance_ready();
        db.compaction_scheduler
            .write()
            .observe_metadata_evidence(evidence);
    }

    pub fn observe_sidecar_evidence(
        &self,
        db: &crate::db::RockDuck,
        evidence: &SidecarEvidenceSnapshot,
    ) {
        let routed_segment_ids = if evidence.routed_segment_ids.is_empty() {
            db.delta_layer.get_all_segment_ids()
        } else {
            evidence.routed_segment_ids.clone()
        };
        let metadata_evidence = crate::metadata::EvidenceSnapshot {
            table: evidence.table.clone(),
            query_columns: Vec::new(),
            table_stats: crate::query::routing::TableStats::default(),
            routed_segment_ids,
            executed_segment_ids: evidence.executed_segment_ids.clone(),
            total_segment_rows: 0,
            delta_segment_count: db.delta_layer.get_all_segment_ids().len(),
            has_zone_map_predicates: false,
            has_cross_column_or: false,
            projection_contract: Some(evidence.contract.clone()),
        };
        tracing::debug!(
            surface = ?evidence.contract.surface,
            visibility = ?evidence.contract.visibility,
            sidecar_class = ?evidence.contract.sidecar_class,
            evidence_hook = evidence.contract.evidence_hook,
            table = evidence.table,
            routed_segments = metadata_evidence.routed_segment_ids.len(),
            "sidecar evidence observed"
        );
        self.feedback.record_sidecar_digest(
            &evidence.table,
            SidecarDigestSnapshot {
                routed_segment_count: metadata_evidence.routed_segment_ids.len(),
                executed_segment_count: metadata_evidence.executed_segment_ids.len(),
                sidecar_class: format!("{:?}", evidence.contract.sidecar_class),
                surface: format!("{:?}", evidence.contract.surface),
                evidence_hook: evidence.contract.evidence_hook.to_string(),
            },
        );
        let current_assertions = self
            .feedback
            .last_governance_digest(&evidence.table)
            .unwrap_or_default()
            .blocking_assertion_count;
        self.feedback.record_governance_digest(
            &evidence.table,
            GovernanceDigestSnapshot {
                blocking_assertion_count: current_assertions + 1,
            },
        );
        metadata_evidence.assert_governance_ready();
        db.compaction_scheduler
            .write()
            .observe_metadata_evidence(&metadata_evidence);
    }

    /// Record an observed query execution for feedback learning.
    ///
    /// Measured-authority path: only this surface contributes authoritative runtime
    /// correction because it is sourced from a completed execution with explicit
    /// executed-segment attribution.
    ///
    /// Current semantics: selectivity remains a lower-bound runtime observation.
    /// `rows_returned` is real post-filter output, while `candidate_rows` is the routed
    /// segment-row upper bound used for selectivity learning. `rows_scanned` remains the
    /// observed post-visibility work signal used for execution-history and cost calibration.
    /// This improves evidence quality without granting selectivity authority over routing decisions.
    ///
    /// Call this after each query completes with the actual number of rows returned.
    pub fn record_execution_outcome(&self, outcome: &ExecutionOutcome) {
        let Some(chosen_path) = outcome.chosen_path else {
            return;
        };

        self.feedback.observe_selectivity(
            &outcome.table,
            &outcome.query_columns,
            outcome.rows_returned,
            outcome.candidate_rows.max(outcome.rows_returned),
        );
        self.feedback.record_execution_outcome(outcome, chosen_path);

        let alternative_path = match chosen_path {
            RouteDecision::DeltaStoreOnly => Some(RouteDecision::VortexOnly),
            RouteDecision::VortexOnly => Some(RouteDecision::DeltaStoreOnly),
            RouteDecision::Merge => Some(RouteDecision::VortexOnly),
        };
        if let Some(alt_path) = alternative_path {
            if let Some(avg_alt_ms) = self.feedback.execution.avg_time(&outcome.table, alt_path) {
                self.record_regret_feedback(
                    &outcome.table,
                    chosen_path,
                    outcome.elapsed_ms,
                    alt_path,
                    avg_alt_ms,
                );
            }
        }

        if let Some(estimated_cost) = outcome.estimated_cost {
            self.update_cost_params(
                estimated_cost,
                outcome.elapsed_ms,
                outcome.rows_scanned.max(1),
            );
        }

        if let Some(digest) = outcome.cooperative_digest.as_ref() {
            self.feedback.record_cooperative_runtime(
                &outcome.table,
                CooperativeRuntimeSnapshot {
                    budget: digest.budget.map(|budget| CooperativeRuntimeBudget {
                        max_segments_per_slice: budget.max_segments_per_slice,
                        max_slice_ms: budget.max_slice_ms,
                    }),
                    slices_executed: digest.slices_executed,
                    truncated_segments: digest.truncated_segments.clone(),
                    elapsed_budget_exhausted: digest.elapsed_budget_exhausted,
                    max_segments_per_slice_hit: digest.max_segments_per_slice_hit,
                },
            );
        }

        let alternative_path = match chosen_path {
            RouteDecision::DeltaStoreOnly => Some(RouteDecision::VortexOnly),
            RouteDecision::VortexOnly => Some(RouteDecision::DeltaStoreOnly),
            RouteDecision::Merge => None,
        };
        if let Some(alt_path) = alternative_path {
            let features = self.plan_encoder.encode(&RouterParams::new(
                &outcome.table,
                &outcome.query_columns,
                crate::query::routing::QueryKind::FullScan,
                outcome.executed_segment_ids.len(),
                false,
                if outcome.candidate_rows == 0 {
                    0.0
                } else {
                    outcome.rows_returned as f64 / outcome.candidate_rows as f64
                },
                &crate::query::routing::TableStats {
                    row_count: outcome.candidate_rows.max(outcome.rows_scanned),
                    ..Default::default()
                },
            ));
            match alt_path {
                RouteDecision::DeltaStoreOnly => {
                    self.record_ml_feedback(
                        features,
                        self.feedback.execution.avg_time(&outcome.table, alt_path),
                        Some(outcome.elapsed_ms),
                    );
                }
                RouteDecision::VortexOnly => {
                    self.record_ml_feedback(
                        features,
                        Some(outcome.elapsed_ms),
                        self.feedback.execution.avg_time(&outcome.table, alt_path),
                    );
                }
                RouteDecision::Merge => {}
            }
        }

        self.dirty.store(true, Ordering::Relaxed);
    }

    /// Record an observed query execution for feedback learning.
    ///
    /// Deprecated path: prefers `record_execution_outcome`, but kept as a compatibility
    /// adapter for callers that still only have aggregate counters.
    pub fn record_feedback(
        &self,
        table: &str,
        columns: &[String],
        actual_rows: u64,
        total_rows: u64,
        elapsed_ms: f64,
        chosen_path: RouteDecision,
    ) {
        self.feedback
            .observe_selectivity(table, columns, actual_rows, total_rows);
        self.feedback.record_execution_aggregate(
            table,
            chosen_path,
            elapsed_ms,
            0.0,
            total_rows,
            actual_rows,
        );

        let alternative_path = match chosen_path {
            RouteDecision::DeltaStoreOnly => Some(RouteDecision::VortexOnly),
            RouteDecision::VortexOnly => Some(RouteDecision::DeltaStoreOnly),
            RouteDecision::Merge => Some(RouteDecision::VortexOnly),
        };
        if let Some(alt_path) = alternative_path {
            if let Some(avg_alt_ms) = self.feedback.execution.avg_time(table, alt_path) {
                self.record_regret_feedback(table, chosen_path, elapsed_ms, alt_path, avg_alt_ms);
                tracing::trace!(
                    table = table,
                    chosen = ?chosen_path,
                    chosen_ms = elapsed_ms,
                    alternative = ?alt_path,
                    alternative_avg_ms = avg_alt_ms,
                    "proxy regret feedback recorded"
                );
            }
        }

        self.dirty.store(true, Ordering::Relaxed);
    }

    /// Phase 10: Record ML routing feedback sample for online model retraining.
    /// Call this after query execution when you have the feature vector and actual times.
    ///
    /// Classified status: advisory-only caller with memory-only storage.
    /// Governance gate G5 keeps this non-authoritative until durability, replay semantics,
    /// and bounded rollout evaluation are explicitly ratified.
    /// Samples are accepted only under advisory/shadow mode and must never imply live authority.
    #[allow(dead_code)]
    pub fn record_ml_feedback(
        &self,
        features: crate::query::routing::ml::QueryFeatureVector,
        delta_ms: Option<f64>,
        vortex_ms: Option<f64>,
    ) {
        let mode = self.ml_routing_mode();
        if mode.allows_live_authority() {
            tracing::warn!(
                ml_mode = ?mode,
                "ML feedback remains non-authoritative even when live override is enabled"
            );
        }
        // Only record when we have both path times for meaningful comparison.
        if let (Some(dm), Some(vm)) = (delta_ms, vortex_ms) {
            self.feedback.record_ml_sample(features, dm, vm);
            self.feedback
                .set_measured_ml_training_sample_count(self.feedback.ml_collector.len());
            tracing::trace!(
                ml_sample=true,
                delta_ms=dm,
                vortex_ms=vm,
                ml_mode=?mode,
                "ML feedback sample recorded"
            );
        }
        self.dirty.store(true, Ordering::Relaxed);
    }

    /// Record observed regret instrumentation when both chosen-path and comparison-path
    /// timings are available. This keeps Phase 4 instrumentation-first until shadow reads
    /// become broadly available.
    pub fn record_regret_feedback(
        &self,
        table: &str,
        chosen_path: RouteDecision,
        elapsed_ms: f64,
        alternative_path: RouteDecision,
        alternative_elapsed_ms: f64,
    ) {
        self.feedback.record_regret_sample(
            table,
            chosen_path,
            elapsed_ms,
            alternative_path,
            alternative_elapsed_ms,
        );
        self.dirty.store(true, Ordering::Relaxed);
    }

    /// Record observation-only same-query shadow timing.
    /// Samples stored under `SyntheticEstimate` remain non-authoritative and may only support
    /// analysis/reporting until a real same-query dual-path collection model is proven.
    pub fn record_shadow_timing(
        &self,
        sample: crate::query::routing::feedback::ShadowTimingSample,
    ) {
        self.feedback.record_shadow_timing(sample);
        self.dirty.store(true, Ordering::Relaxed);
    }

    /// Update adaptive cost parameters from actual execution feedback.
    pub fn update_cost_params(&self, estimated: f64, actual: f64, rows: u64) {
        let mut params = self.cost_params.write();
        params.update(estimated, actual, rows);
        self.dirty.store(true, Ordering::Relaxed);
    }

    /// Checkpoint dirty feedback state to KV store.
    /// Returns true if a checkpoint was written.
    /// D12 fix: accepts committed_txn to record the checkpoint position accurately.
    pub fn checkpoint_if_dirty(&self, kv: &Arc<dyn KVEngine>, committed_txn: u64) -> Result<bool> {
        let mut stats = self.checkpoint_hotspot_stats.write();
        stats.calls += 1;
        let is_dirty = self.dirty.load(Ordering::Acquire);
        if !is_dirty {
            stats.skips += 1;
            return Ok(false);
        }

        self.feedback
            .set_cost_params_checkpoint(self.cost_params.read().to_checkpoint());
        self.flush_ml_export_if_ready();
        self.feedback.persist_to_kv(kv, committed_txn)?;
        self.dirty.store(false, Ordering::Release);
        stats.writes += 1;
        Ok(true)
    }

    pub fn checkpoint_hotspot_stats(&self) -> CheckpointHotspotStats {
        self.checkpoint_hotspot_stats.read().clone()
    }

    pub fn take_last_ml_export(&self) -> Vec<MlSample> {
        std::mem::take(&mut *self.last_ml_export.write())
    }

    fn flush_ml_export_if_ready(&self) {
        if !self
            .feedback
            .ml_collector
            .ready_for_training(self.config.ml_export_threshold)
        {
            return;
        }
        let drained = self.feedback.ml_collector.drain();
        if drained.is_empty() {
            return;
        }
        let export_batch = MlExportBatch {
            samples: drained.iter().map(MlSample::to_export_sample).collect(),
        };
        self.feedback.push_durable_ml_export(export_batch);
        self.feedback
            .set_measured_ml_training_sample_count(self.feedback.ml_collector.len());
        *self.last_ml_export.write() = drained;
    }

    /// Get a clone of the feedback handle for passing to ScanIterator.
    pub fn feedback_handle(&self) -> Arc<FeedbackState> {
        self.feedback.clone()
    }
}

/// Owned variant of `RouterParams` for convenience.
#[derive(Debug, Clone)]
pub struct RouterParamsOwned {
    pub table: String,
    pub columns: Vec<String>,
    pub kind: QueryKind,
    pub delta_count: usize,
    pub has_pending_writes: bool,
    pub estimated_selectivity: f64,
    pub stats: TableStats,
}

impl RouterParamsOwned {
    pub fn new(
        table: String,
        columns: Vec<String>,
        kind: QueryKind,
        delta_count: usize,
        has_pending_writes: bool,
        estimated_selectivity: f64,
        stats: TableStats,
    ) -> Self {
        Self {
            table,
            columns,
            kind,
            delta_count,
            has_pending_writes,
            estimated_selectivity,
            stats,
        }
    }

    /// Convert to a borrowed `RouterParams` for use with the router.
    pub fn as_borrowed(&self) -> RouterParams<'_> {
        RouterParams {
            table: &self.table,
            columns: &self.columns,
            kind: self.kind,
            delta_count: self.delta_count,
            has_pending_writes: self.has_pending_writes,
            estimated_selectivity: self.estimated_selectivity,
            stats: &self.stats,
        }
    }
}

/// Parameters required for a routing decision.
/// Produced by scanning the query AST (from filter_expr) and table metadata.
#[derive(Debug, Clone)]
pub struct RouterParams<'a> {
    /// Target table name.
    pub table: &'a str,
    /// Columns referenced in the query.
    pub columns: &'a [String],
    /// Query classification.
    pub kind: QueryKind,
    /// Number of deltas in DeltaStore for this table.
    pub delta_count: usize,
    /// Whether there are any pending writes (non-empty DeltaStore).
    pub has_pending_writes: bool,
    /// Estimated selectivity from predicate analysis.
    /// Tier 1 can compute this without statistics.
    pub estimated_selectivity: f64,
    /// Column statistics for this table (from SegmentMetaCache).
    pub stats: &'a TableStats,
}

impl<'a> RouterParams<'a> {
    pub fn new(
        table: &'a str,
        columns: &'a [String],
        kind: QueryKind,
        delta_count: usize,
        has_pending_writes: bool,
        estimated_selectivity: f64,
        stats: &'a TableStats,
    ) -> Self {
        Self {
            table,
            columns,
            kind,
            delta_count,
            has_pending_writes,
            estimated_selectivity,
            stats,
        }
    }
}

#[cfg(test)]
mod governance_guard_tests {
    use super::*;
    use crate::query::routing::feedback::{SelectivityTracker, ShadowTimingSample};

    #[test]
    fn low_sample_observations_do_not_blend_into_costs() {
        let router = StatisticalRouter::new();
        let cfg = RouterConfig::default();
        let feedback = FeedbackState::new(32, 32);
        let tracker = SelectivityTracker::new(32);
        let cost_params = CostParams::from_config(&cfg);
        let stats = TableStats {
            row_count: 1_000,
            total_bytes: 10_000,
            compressed_bytes: 5_000,
            column_stats: Default::default(),
        };
        let columns = vec!["status".to_string()];
        let params = RouterParams::new(
            "orders",
            &columns,
            QueryKind::FullScan,
            4,
            true,
            0.2,
            &stats,
        );

        feedback.record_execution_aggregate(
            "orders",
            RouteDecision::DeltaStoreOnly,
            200.0,
            0.0,
            100,
            10,
        );
        feedback.record_execution_aggregate(
            "orders",
            RouteDecision::VortexOnly,
            10.0,
            0.0,
            100,
            10,
        );

        let result = router.decide(&params, &cfg, &tracker, &feedback, &cost_params);

        let sel = params.estimated_selectivity;
        let rows = params.stats.estimated_scanned_rows(sel);
        let expected_delta =
            crate::query::routing::cost::delta_scan_cost(rows, params.stats, &cost_params);
        let expected_vortex =
            crate::query::routing::cost::vortex_scan_cost(rows, params.stats, &cost_params);
        let expected_merge = crate::query::routing::cost::merge_cost(
            params.delta_count,
            rows,
            params.stats,
            &cost_params,
        );
        let (_, expected_min) = crate::query::routing::cost::argmin_path(
            expected_delta,
            expected_vortex,
            expected_merge,
        );

        assert_eq!(result.estimated_cost, expected_min);
        assert!(
            result.confidence < 0.5,
            "low-sample routing should stay bounded: {}",
            result.confidence
        );
    }

    #[test]
    fn shadow_sampling_uses_varying_epoch_for_same_query_shape() {
        let router = QueryRouter::default_for_test();
        let table = "orders";
        let columns = vec!["status".to_string()];
        let chosen_path = RouteDecision::VortexOnly;
        let routed_segments = 8;

        let mut outcomes = std::collections::BTreeSet::new();
        for _ in 0..64 {
            outcomes.insert(router.should_sample_shadow_timing(
                table,
                &columns,
                chosen_path,
                routed_segments,
            ));
        }

        assert!(outcomes.contains(&true));
        assert!(outcomes.contains(&false));
    }

    #[test]
    fn cold_table_estimated_cost_gap_does_not_raise_confidence_without_measured_history() {
        let router = StatisticalRouter::new();
        let cfg = RouterConfig::default();
        let feedback = FeedbackState::new(32, 32);
        let tracker = SelectivityTracker::new(32);
        let cost_params = CostParams::from_config(&cfg);
        let columns = vec!["status".to_string()];
        let stats = TableStats {
            row_count: 10_000,
            total_bytes: 10_240_000,
            compressed_bytes: 512_000,
            column_stats: Default::default(),
        };
        let params = RouterParams::new(
            "orders",
            &columns,
            QueryKind::RangeScan,
            0,
            true,
            0.001,
            &stats,
        );

        let result = router.decide(&params, &cfg, &tracker, &feedback, &cost_params);

        assert!(
            result.confidence <= 0.35,
            "cold-table confidence should not be inflated by estimated-only cost gap: {}",
            result.confidence
        );
    }

    #[test]
    fn single_path_measured_history_does_not_raise_cost_gap_confidence() {
        let router = StatisticalRouter::new();
        let cfg = RouterConfig::default();
        let feedback = FeedbackState::new(32, 32);
        let tracker = SelectivityTracker::new(32);
        let cost_params = CostParams::from_config(&cfg);
        let columns = vec!["status".to_string()];
        let stats = TableStats {
            row_count: 10_000,
            total_bytes: 10_240_000,
            compressed_bytes: 512_000,
            column_stats: Default::default(),
        };
        let params = RouterParams::new(
            "orders",
            &columns,
            QueryKind::RangeScan,
            0,
            true,
            0.001,
            &stats,
        );

        for elapsed in [5.0, 6.0, 4.0] {
            feedback.record_execution_aggregate(
                "orders",
                RouteDecision::DeltaStoreOnly,
                elapsed,
                0.0,
                100,
                1,
            );
        }

        let result = router.decide(&params, &cfg, &tracker, &feedback, &cost_params);

        assert!(
            result.confidence <= 0.50,
            "single-path measured history should not claim comparative cost-gap authority: {}",
            result.confidence
        );
    }

    #[test]
    fn measured_samples_gate_cost_blending_and_confidence() {
        let router = StatisticalRouter::new();
        let cfg = RouterConfig::default();
        let feedback = FeedbackState::new(32, 32);
        let tracker = SelectivityTracker::new(32);
        let cost_params = CostParams::from_config(&cfg);
        let columns = vec!["status".to_string()];
        let stats = TableStats {
            row_count: 1_000,
            total_bytes: 10_000,
            compressed_bytes: 5_000,
            column_stats: Default::default(),
        };
        let params = RouterParams::new(
            "orders",
            &columns,
            QueryKind::FullScan,
            4,
            true,
            0.2,
            &stats,
        );

        for elapsed in [120.0, 110.0, 100.0] {
            feedback.record_execution_aggregate(
                "orders",
                RouteDecision::DeltaStoreOnly,
                elapsed,
                0.0,
                100,
                10,
            );
        }
        for elapsed in [40.0, 45.0, 50.0] {
            feedback.record_execution_aggregate(
                "orders",
                RouteDecision::VortexOnly,
                elapsed,
                0.0,
                100,
                10,
            );
        }

        let result = router.decide(&params, &cfg, &tracker, &feedback, &cost_params);
        assert!(
            result.confidence >= 0.4,
            "measured samples should raise confidence: {}",
            result.confidence
        );
    }

    #[test]
    fn shadow_timing_samples_do_not_count_as_measured_execution_history() {
        let feedback = FeedbackState::new(32, 32);
        feedback.record_execution_aggregate(
            "orders",
            RouteDecision::DeltaStoreOnly,
            120.0,
            0.0,
            100,
            10,
        );
        feedback.record_execution_aggregate(
            "orders",
            RouteDecision::DeltaStoreOnly,
            110.0,
            0.0,
            100,
            10,
        );
        feedback.record_execution_aggregate(
            "orders",
            RouteDecision::DeltaStoreOnly,
            100.0,
            0.0,
            100,
            10,
        );

        feedback.record_shadow_timing(ShadowTimingSample {
            table: "orders".to_string(),
            chosen_path: RouteDecision::DeltaStoreOnly,
            shadow_path: RouteDecision::VortexOnly,
            chosen_elapsed_ms: 100.0,
            shadow_elapsed_ms: 80.0,
            rows_scanned: 100,
            candidate_rows: 100,
            rows_returned: 10,
            segments_routed: 1,
            segments_scanned: 1,
            template: Some("FullScan".to_string()),
            policy: crate::query::routing::feedback::ShadowTimingPolicy::BoundedDualPath,
        });

        assert_eq!(feedback.execution.measured_execution_count("orders"), 0);
    }

    #[test]
    fn proxy_samples_do_not_raise_cost_gap_confidence() {
        let router = StatisticalRouter::new();
        let cfg = RouterConfig::default();
        let feedback = FeedbackState::new(32, 32);
        let tracker = SelectivityTracker::new(32);
        let cost_params = CostParams::from_config(&cfg);
        let columns = vec!["status".to_string()];
        let stats = TableStats {
            row_count: 10_000,
            total_bytes: 10_240_000,
            compressed_bytes: 512_000,
            column_stats: Default::default(),
        };
        let params = RouterParams::new(
            "orders",
            &columns,
            QueryKind::RangeScan,
            0,
            true,
            0.001,
            &stats,
        );

        for elapsed in [120.0, 110.0, 100.0] {
            feedback.record_execution_aggregate(
                "orders",
                RouteDecision::DeltaStoreOnly,
                elapsed,
                0.0,
                100,
                10,
            );
        }
        for shadow_elapsed in [40.0, 45.0, 50.0] {
            feedback.record_shadow_timing(ShadowTimingSample {
                table: "orders".to_string(),
                chosen_path: RouteDecision::DeltaStoreOnly,
                shadow_path: RouteDecision::VortexOnly,
                chosen_elapsed_ms: 100.0,
                shadow_elapsed_ms: shadow_elapsed,
                rows_scanned: 100,
                candidate_rows: 100,
                rows_returned: 10,
                segments_routed: 1,
                segments_scanned: 1,
                template: Some("RangeScan".to_string()),
                policy: crate::query::routing::feedback::ShadowTimingPolicy::BoundedDualPath,
            });
        }

        let result = router.decide(&params, &cfg, &tracker, &feedback, &cost_params);
        assert!(
            result.confidence <= 0.50,
            "proxy dual-path samples must not claim measured comparative authority: {}",
            result.confidence
        );
    }

    #[test]
    fn proxy_samples_do_not_appear_as_measured_cost_samples() {
        let feedback = FeedbackState::new(32, 32);
        feedback.record_regret_sample(
            "orders",
            RouteDecision::DeltaStoreOnly,
            30.0,
            RouteDecision::VortexOnly,
            200.0,
        );

        let sample = feedback
            .execution
            .measured_cost_sample("orders", RouteDecision::DeltaStoreOnly)
            .expect("proxy regret samples should remain synthetic-quality observations");
        assert_eq!(sample.sample_count, 1);
        assert_eq!(
            sample.quality,
            crate::query::routing::feedback::EvidenceQuality::Synthetic
        );
        assert!(!sample.quality.allows_cost_blend());
    }

    #[test]
    fn single_selectivity_sample_does_not_raise_confidence() {
        let router = StatisticalRouter::new();
        let cfg = RouterConfig::default();
        let feedback = FeedbackState::new(32, 32);
        let tracker = SelectivityTracker::new(32);
        let cost_params = CostParams::from_config(&cfg);
        let stats = TableStats {
            row_count: 1_000,
            total_bytes: 10_000,
            compressed_bytes: 5_000,
            column_stats: Default::default(),
        };
        let columns = vec!["status".to_string()];
        let params = RouterParams::new(
            "orders",
            &columns,
            QueryKind::RangeScan,
            4,
            true,
            0.2,
            &stats,
        );

        tracker.observe("orders", "status", 5, 100);

        let result = router.decide(&params, &cfg, &tracker, &feedback, &cost_params);
        assert!(
            result.confidence < 0.35,
            "single-sample selectivity should stay bounded: {}",
            result.confidence
        );
    }

    #[test]
    fn live_sidecar_metadata_evidence_requires_projection_contract() {
        let evidence = crate::metadata::EvidenceSnapshot {
            table: "orders".to_string(),
            query_columns: vec!["status".to_string()],
            table_stats: TableStats::default(),
            routed_segment_ids: vec!["seg-1".to_string()],
            executed_segment_ids: Vec::new(),
            total_segment_rows: 10,
            delta_segment_count: 1,
            has_zone_map_predicates: false,
            has_cross_column_or: false,
            projection_contract: Some(ProjectionContract::vtab()),
        };

        evidence.assert_governance_ready();
    }

    #[test]
    fn ml_feedback_stays_non_authoritative_in_test_router() {
        let router = QueryRouter::default_for_test();
        let features = crate::query::routing::ml::QueryFeatureVector::from_query(
            crate::query::routing::ml::OpType::Scan,
            0.25,
            1024,
            4,
            8,
            16,
        );

        router.record_ml_feedback(features, Some(12.0), Some(9.5));

        assert_eq!(router.ml_routing_mode(), MlRoutingMode::Disabled);
        assert!(!router.ml_routing_mode().allows_live_authority());
    }

    #[test]
    fn checkpoint_hotspot_stats_track_skip_and_write_paths() {
        let router = QueryRouter::default_for_test();
        let kv: Arc<dyn KVEngine> = Arc::new(TestMemoryKv::new());

        assert!(!router
            .checkpoint_if_dirty(&kv, 0)
            .expect("clean checkpoint call"));
        let clean_stats = router.checkpoint_hotspot_stats();
        assert_eq!(clean_stats.calls, 1);
        assert_eq!(clean_stats.skips, 1);
        assert_eq!(clean_stats.writes, 0);

        router.update_cost_params(10.0, 12.0, 100);
        assert!(router
            .checkpoint_if_dirty(&kv, 0)
            .expect("dirty checkpoint call"));
        let dirty_stats = router.checkpoint_hotspot_stats();
        assert_eq!(dirty_stats.calls, 2);
        assert_eq!(dirty_stats.skips, 1);
        assert_eq!(dirty_stats.writes, 1);
    }
}

/// Table-level statistics used by the cost model.
#[derive(Debug, Clone, Default)]
pub struct TableStats {
    pub row_count: u64,
    pub column_stats: HashMap<String, ColumnStats>,
    pub total_bytes: u64,
    pub compressed_bytes: u64,
}

/// Column-level statistics for selectivity estimation.
#[derive(Debug, Clone, Default)]
pub struct ColumnStats {
    pub min: Option<Vec<u8>>,
    pub max: Option<Vec<u8>>,
    pub null_count: u64,
    pub distinct_count: u64,
    pub avg_width_bytes: f64,
    /// The Arrow data type of this column. Required for type-aware selectivity
    /// estimation in `estimate_position` — without it, byte sequences are ambiguous
    /// (e.g. an 8-byte integer value and an 8-byte float have the same byte length
    /// but different semantic values when interpreted as IEEE-754 f64).
    pub column_type: Option<DataType>,
}

impl TableStats {
    pub fn avg_row_bytes(&self) -> f64 {
        if self.row_count == 0 {
            16.0
        } else {
            self.total_bytes as f64 / self.row_count as f64
        }
    }

    pub fn compression_ratio(&self) -> f64 {
        if self.total_bytes == 0 {
            0.5
        } else {
            self.compressed_bytes as f64 / self.total_bytes as f64
        }
    }

    pub fn estimated_scanned_rows(&self, selectivity: f64) -> u64 {
        let sel = if selectivity.is_nan() {
            0.1
        } else {
            selectivity.clamp(0.0, 1.0)
        };
        (self.row_count as f64 * sel) as u64
    }
}
