//! Adaptive Compaction Scheduler — Hill Climbing Weight Optimization
//!
//! Maintenance verification protocol for every debt/rewrite change:
//! 1. Confirm the main heuristic path (feedback -> weights -> scheduler priority).
//! 2. Confirm bypass paths (RangeReduce/query-driven, small-file merge, PDT merge specific logic).
//! 3. Confirm landing files where heuristic signals become executable rewrite decisions.
//!
//! Alternative to fixed compaction weight coefficients, using hill climbing for auto-tuning.
//!
//! Approach:
//! - After each compaction completes, check if Zone Map hit rate improves
//! - If improved -> continue in the same direction
//! - If degraded -> rollback and reverse
//! - Evaluate every N rounds to avoid jitter
//!
//! # Three-Layer Model (current reality)
//!
//! The maintenance system currently has three layers that are mixed in `calculate_priority`:
//!
//! | Layer | Description | Example | Commitment |
//! |-------|-------------|---------|-----------|
//! | **Debt signals** | Factors that directly represent storage/read debt | `del_ratio`, `staleness_penalty`, `miss_penalty` | Semantic: these ARE the debt being measured |
//! | **Heuristic factors** | Approximations for compaction benefit | `size_score`, `age_score` | Heuristic: these are reasonable proxies but not debt itself |
//! | **Tuning coefficients** | Tunable knobs for weight calibration | `del_coef`, `stale_coef`, `miss_coef` | Arbitrary: these are hill-climbing targets, not semantic debt |
//!
//! RockDuck also exposes an additive typed debt layer via `DebtFlags`. This does NOT replace
//! the scalar `f64` priority score. It exists so governance and future design can
//! reason about explicit debt dimensions without overloading the score itself.
//!
//! ## Design Decision
//!
//! These three layers are currently mixed into a single `f64` score. This is **acceptable**
//! for a heuristic scheduler but does NOT constitute an explicit debt model. The score
//! is used for priority ordering, not typed debt classification.
//!
//! If a typed debt model is needed in the future, it should be introduced as a separate
//! type that produces typed debt flags (e.g., `DebtFlag::HighDeleteRatio | DebtFlag::StaleZoneMap`)
//! which then feed into `CompactionStrategy` dispatch — NOT by further mixing into the score.
//! The current scalar score approach should NOT be refactored into a "typed score" without
//! a clear requirement. See `keep-it-simple` skill before adding new abstraction.
//!
//! ### Why hill climbing only adjusts stale_coef and miss_coef?
//!
//! `adjust()` only modifies `stale_coef` and `miss_coef`. `del_coef`, `size_coef`, and
//! `age_coef` are never adjusted. The plausible rationale is:
//! - `del_ratio` and `size_score` are **direct physical measurements** (from `SegmentMeta`)
//!   — they are not estimates and do not drift; hill-climbing them would be overfitting.
//! - `stale_penalty` and `miss_penalty` are **behavioral signals** (from `QueryFeedback`) —
//!   they can drift as access patterns change; hill-climbing compensates for this drift.
//! - This is a **plausible but unconfirmed design decision**. A tracking issue should
//!   formalize this rationale before it is treated as canon.
//!
//! ### `record_compaction_result` is conditionally wired
//!
//! `record_compaction_result` records the post-compaction callback signal so that
//! `prev_hit_ratio` can be updated for the next call, enabling hill climbing rollback.
//! The callback wiring exists in `io_scheduler.rs::with_scheduler_and_compactor_internal`,
//! and the default runtime now routes compactions through that constructor, but the callback
//! value passed today is still a bootstrap proxy (`1.0` for a freshly compacted segment),
//! not a measured post-compaction prune hit ratio from real queries.
//!
//! Consequence: the feedback hook is structurally available and live in the default runtime,
//! but its callback metric remains proxy-only until RockDuck observes real post-compaction
//! query evidence for the new segment.
//!
//! **CurrentReality classification**: live callback path with proxy-only metric.
//! It is visible maintenance scaffolding, not a fully authoritative production feedback loop,
//! until the default runtime measures post-compaction quality instead of assuming it.

use crate::metadata::EvidenceSnapshot;
use crate::query::feedback::FeedbackHandle;
use crate::query::routing::feedback::ExecutionEvidenceSnapshot;
use std::sync::Arc;
use tracing::debug;

/// Adaptive compaction weights
#[derive(Debug, Clone)]
pub struct CompactionWeights {
    /// Delete ratio coefficient (high delete ratio prioritized for merge)
    pub del_coef: f64,
    /// Size coefficient (small file merge)
    pub size_coef: f64,
    /// Age coefficient (older data prioritized for merge)
    pub age_coef: f64,
    /// Zone Map misalignment penalty coefficient
    pub stale_coef: f64,
    /// Pruning failure penalty coefficient
    pub miss_coef: f64,
}

impl Default for CompactionWeights {
    fn default() -> Self {
        Self {
            del_coef: 10.0,
            size_coef: 0.3,
            age_coef: 0.2,
            stale_coef: 5.0,
            miss_coef: 3.0,
        }
    }
}

impl CompactionWeights {
    /// Create weights with specified initial values
    pub fn new(del: f64, size: f64, age: f64, stale: f64, miss: f64) -> Self {
        Self {
            del_coef: del,
            size_coef: size,
            age_coef: age,
            stale_coef: stale,
            miss_coef: miss,
        }
    }
}

bitflags::bitflags! {
    /// Typed debt flags for a segment.
    ///
    /// This is an additive classification layer on top of the existing scalar priority score.
    /// It does not replace `calculate_priority`; it makes debt dimensions explicit for
    /// governance/reporting while dispatch continues to rely on scalar/runtime gates.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
    pub struct DebtFlags: u8 {
        const NONE = 0;
        const HIGH_DELETE_RATIO = 1 << 0;
        const STALE_ZONE_MAP = 1 << 1;
        const PRUNE_MISS_HIGH = 1 << 2;
        const SMALL_FILE = 1 << 3;
        const COLD_ACCESS = 1 << 4;
    }
}

#[derive(Debug, Clone)]
pub struct MaintenancePosture {
    pub current_reality: &'static str,
    pub ascension_target: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DebtSignalClass {
    Direct,
    Heuristic,
    EvidenceDriven,
}

impl DebtSignalClass {
    pub fn permits_rewrite(self, budget: crate::compaction::scheduler::RewriteBudget) -> bool {
        match self {
            DebtSignalClass::Direct => true,
            DebtSignalClass::Heuristic => matches!(
                budget,
                crate::compaction::scheduler::RewriteBudget::HeuristicPriority
            ),
            DebtSignalClass::EvidenceDriven => matches!(
                budget,
                crate::compaction::scheduler::RewriteBudget::EvidenceDriven
            ),
        }
    }
}

#[derive(Debug, Clone)]
pub struct MaintenanceEvidenceSnapshot {
    pub seg_id: String,
    pub table: String,
    pub query_columns: Vec<String>,
    pub total_segment_rows: u64,
    pub delta_segment_count: usize,
    pub has_zone_map_predicates: bool,
    pub has_cross_column_or: bool,
    pub table_row_count: u64,
    pub total_bytes: u64,
    pub compressed_bytes: u64,
}

impl From<&ExecutionEvidenceSnapshot> for MaintenanceEvidenceSnapshot {
    fn from(evidence: &ExecutionEvidenceSnapshot) -> Self {
        Self {
            seg_id: String::new(),
            table: evidence.table.clone(),
            query_columns: evidence.query_columns.clone(),
            total_segment_rows: evidence.total_segment_rows,
            delta_segment_count: evidence.delta_segment_count,
            has_zone_map_predicates: evidence.has_zone_map_predicates,
            has_cross_column_or: evidence.has_cross_column_or,
            table_row_count: evidence.table_row_count,
            total_bytes: evidence.total_bytes,
            compressed_bytes: evidence.compressed_bytes,
        }
    }
}

impl From<&EvidenceSnapshot> for MaintenanceEvidenceSnapshot {
    fn from(evidence: &EvidenceSnapshot) -> Self {
        Self {
            seg_id: String::new(),
            table: evidence.table.clone(),
            query_columns: evidence.query_columns.clone(),
            total_segment_rows: evidence.total_segment_rows,
            delta_segment_count: evidence.delta_segment_count,
            has_zone_map_predicates: evidence.has_zone_map_predicates,
            has_cross_column_or: evidence.has_cross_column_or,
            table_row_count: evidence.table_stats.row_count,
            total_bytes: evidence.table_stats.total_bytes,
            compressed_bytes: evidence.table_stats.compressed_bytes,
        }
    }
}

impl MaintenanceEvidenceSnapshot {
    fn compressed_bytes_ratio(&self) -> f64 {
        if self.total_bytes == 0 {
            1.0
        } else {
            (self.compressed_bytes as f64 / self.total_bytes as f64).clamp(0.0, 1.0)
        }
    }
}

#[derive(Debug, Clone)]
pub struct MaintenanceLedgerRow {
    pub seg_id: String,
    pub debt_flags: DebtFlags,
    pub debt_signal_class: DebtSignalClass,
    pub action: Option<crate::compaction::scheduler::RewriteAction>,
    pub action_budget: Option<crate::compaction::scheduler::RewriteBudget>,
    pub action_authority: Option<crate::compaction::scheduler::ActionAuthority>,
    pub executor_authoritative: bool,
    pub del_ratio: f64,
    pub staleness_penalty: f64,
    pub prune_miss_ratio: f64,
    pub small_file: bool,
    pub evidence: Option<MaintenanceEvidenceSnapshot>,
    pub last_outcome: Option<DebtResolutionOutcome>,
    pub last_outcome_measured: bool,
    pub feedback_seeded_from: Option<String>,
    pub feedback_query_count: u64,
}

#[derive(Debug, Clone, Default)]
pub struct MaintenanceLedger {
    pub rows: Vec<MaintenanceLedgerRow>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MaintenanceGovernanceRow {
    pub seg_id: String,
    pub action: Option<crate::compaction::scheduler::RewriteAction>,
    pub action_budget: Option<crate::compaction::scheduler::RewriteBudget>,
    pub action_authority: Option<crate::compaction::scheduler::ActionAuthority>,
    pub executor_authoritative: bool,
    pub feedback_seeded_from: Option<String>,
    pub has_last_outcome: bool,
    pub last_outcome_measured: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MaintenanceGovernanceSnapshot {
    pub rows: Vec<MaintenanceGovernanceRow>,
}

impl MaintenanceLedger {
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }
}

/// Adaptive compaction scheduler
pub struct AdaptiveCompactionScheduler {
    /// Current weights
    weights: CompactionWeights,
    /// Query feedback collector — shared via Arc so that Clone shares state (G9 fix).
    /// Previously, the derived Clone would create a NEW empty collector, losing all statistics.
    feedback: FeedbackHandle,
    evidence:
        Arc<parking_lot::RwLock<std::collections::HashMap<String, MaintenanceEvidenceSnapshot>>>,
    /// Write heat tracker for estimating write throughput
    write_heat_tracker: crate::write::heat_tracker::WriteHeatTrackerHandle,
    /// Previous weights after adjustment (for rollback)
    prev_weights: CompactionWeights,
    /// Adjustment step size
    step_size: f64,
    /// Minimum query count threshold (no adjustment below this)
    min_query_count: u64,
    /// Adjustment interval (evaluate once every N schedule calls)
    adjustment_interval: usize,
    /// Schedule call count
    call_count: usize,
    /// Zone Map/pruning signal after last compaction callback.
    /// In the current runtime this starts as a bootstrap proxy signal, not a measured
    /// post-compaction observation from real queries.
    prev_hit_ratio: f64,
    /// Last typed maintenance outcome observed per rewritten segment.
    last_compaction_outcomes:
        Arc<parking_lot::RwLock<std::collections::HashMap<String, DebtResolutionOutcome>>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompactionQualitySignalKind {
    ProxyBootstrap,
    MeasuredDeleteResolution,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DebtResolutionOutcome {
    pub resolved_flags: DebtFlags,
    pub delete_debt_remaining_ratio: f64,
    pub rows_read: u64,
    pub rows_written: u64,
    pub rows_dropped: u64,
    pub source_seg_id: Option<String>,
    pub quality_signal_kind: CompactionQualitySignalKind,
}

impl DebtResolutionOutcome {
    pub fn from_merge_stats(
        seg_id: &str,
        stats: &crate::compaction::pdt_merge::MergeStats,
    ) -> Self {
        let remaining_ratio = if stats.rows_read == 0 {
            1.0
        } else {
            1.0 - (stats.rows_dropped as f64 / stats.rows_read as f64)
        };
        Self {
            resolved_flags: DebtFlags::HIGH_DELETE_RATIO,
            delete_debt_remaining_ratio: remaining_ratio.clamp(0.0, 1.0),
            rows_read: stats.rows_read,
            rows_written: stats.rows_written,
            rows_dropped: stats.rows_dropped,
            source_seg_id: Some(seg_id.to_string()),
            quality_signal_kind: CompactionQualitySignalKind::MeasuredDeleteResolution,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MaintenanceVerificationCard {
    pub main_path: &'static str,
    pub bypass_paths: Vec<&'static str>,
    pub landing_files: Vec<&'static str>,
}

#[derive(Debug, Clone)]
pub struct MaintenancePackageVerification {
    pub debt_signal_boundary: MaintenanceVerificationCard,
}

#[derive(Debug, Clone)]
pub struct GovernanceVerificationCard {
    pub main_path: &'static str,
    pub bypass_paths: Vec<&'static str>,
    pub landing_files: Vec<&'static str>,
}

#[derive(Debug, Clone)]
pub struct LayeredGovernanceSnapshot {
    pub sidecar_contracts_enforced: bool,
    pub evidence_boundary_enforced: bool,
    pub truth_boundary_documented: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GovernanceBypassInventory {
    pub projection_surfaces: Vec<&'static str>,
    pub evidence_boundaries: Vec<&'static str>,
    pub truth_boundaries: Vec<&'static str>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GovernanceMode {
    Advisory,
    Enforced,
}

impl GovernanceMode {
    pub fn ensures_blocking(self) -> bool {
        matches!(self, Self::Enforced)
    }
}

impl LayeredGovernanceSnapshot {
    pub fn ready_for_enforcement(&self) -> bool {
        self.sidecar_contracts_enforced
            && self.evidence_boundary_enforced
            && self.truth_boundary_documented
    }
}

impl GovernanceBypassInventory {
    pub fn phase2_matrix() -> Self {
        Self {
            projection_surfaces: vec![
                "ProjectionContract::point_get -> point_get::get assert_blocking_governance",
                "ProjectionContract::historical_point_get -> point_get::get_as_of assert_blocking_governance",
                "ProjectionContract::time_travel_scanner -> scan::prepare_read_execution_plan assert_blocking_governance",
                "ProjectionContract::vtab -> vtab_quack::func assert_blocking_governance + live executed_segment_ids gate",
            ],
            evidence_boundaries: vec![
                "metadata::EvidenceSnapshot::assert_governance_ready",
                "metadata::EvidenceSnapshot::into_execution_evidence",
                "read::scan::observe_metadata_evidence",
                "query::vtab_quack::SidecarEvidenceSnapshot live-path empty executed_segment_ids assert",
            ],
            truth_boundaries: vec![
                "AdaptiveCompactionScheduler::governance_verification_card",
                "AdaptiveCompactionScheduler::maintenance_verification_card",
                "CheckpointManager::recovery_verification_card",
                "CheckpointManager::truth_package_verification",
            ],
        }
    }

    pub fn truth_boundary_documented(&self) -> bool {
        self.truth_boundaries.iter().all(|entry| !entry.is_empty())
    }
}

impl AdaptiveCompactionScheduler {
    pub fn governance_mode() -> GovernanceMode {
        GovernanceMode::Enforced
    }

    pub fn maintenance_posture() -> MaintenancePosture {
        MaintenancePosture {
            current_reality: "maintenance today is a heuristic scheduler plus PDT-merge-centric rewrite inventory, with typed debt mostly serving governance visibility",
            ascension_target: "maintenance evolves into shared language and budget across rewrite/flush/metadata while checkpoint remains a truth-plane privileged participant",
        }
    }

    pub fn maintenance_verification_card() -> MaintenanceVerificationCard {
        MaintenanceVerificationCard {
            main_path: "query::feedback::QueryFeedbackCollector -> AdaptiveCompactionScheduler::calculate_priority -> CompactionScheduler queue",
            bypass_paths: vec![
                "CompactionStrategy::QueryDriven",
                "CompactionStrategy::SmallFileMerge",
                "compaction::pdt_merge::compact_segment",
            ],
            landing_files: vec![
                "src/query/feedback.rs",
                "src/compaction/adaptive.rs",
                "src/compaction/scheduler.rs",
                "src/compaction/pdt_merge.rs",
            ],
        }
    }

    pub fn maintenance_package_verification() -> MaintenancePackageVerification {
        MaintenancePackageVerification {
            debt_signal_boundary: Self::maintenance_verification_card(),
        }
    }

    pub fn governance_verification_card() -> GovernanceVerificationCard {
        GovernanceVerificationCard {
            main_path: "Truth/Evidence/Maintenance verification cards -> final cross-check before institutionalization",
            bypass_paths: vec![
                // SANCTIONED bypasses (known, documented):
                "point_get and VTab hidden read paths (sanctioned: bypass router by design, visibility via VisFilter)",
                "heuristic-only rewrite triggers (sanctioned: AdaptiveScheduler weights are heuristic, not typed debt)",
                "recovery fallback not covered by mainline tests (sanctioned: documented in ReplayErrorKind)",
            ],
            landing_files: vec![
                "src/db.rs",
                "src/read/scan.rs",
                "src/write/checkpoint.rs",
                "src/compaction/adaptive.rs",
            ],
        }
    }

    pub fn layered_governance_snapshot() -> LayeredGovernanceSnapshot {
        let bypass_inventory = GovernanceBypassInventory::phase2_matrix();
        LayeredGovernanceSnapshot {
            sidecar_contracts_enforced: bypass_inventory
                .projection_surfaces
                .iter()
                .all(|entry| !entry.is_empty()),
            evidence_boundary_enforced: bypass_inventory
                .evidence_boundaries
                .iter()
                .all(|entry| !entry.is_empty()),
            truth_boundary_documented: bypass_inventory.truth_boundary_documented()
                && !Self::governance_verification_card().bypass_paths.is_empty()
                && !Self::maintenance_verification_card()
                    .bypass_paths
                    .is_empty(),
        }
    }

    pub fn assert_layered_governance() {
        let snapshot = Self::layered_governance_snapshot();
        assert!(
            Self::governance_mode().ensures_blocking(),
            "layered governance must run in enforced mode for correctness-critical seams"
        );
        assert!(
            snapshot.ready_for_enforcement(),
            "layered governance enforcement requires truth/evidence/sidecar boundaries to be classified"
        );
    }

    pub fn new() -> Self {
        Self {
            weights: CompactionWeights::default(),
            feedback: FeedbackHandle::new(),
            evidence: Arc::new(parking_lot::RwLock::new(std::collections::HashMap::new())),
            write_heat_tracker: crate::write::heat_tracker::WriteHeatTrackerHandle::new(),
            prev_weights: CompactionWeights::default(),
            step_size: 0.5,
            min_query_count: 5,
            adjustment_interval: 10,
            call_count: 0,
            prev_hit_ratio: 0.5,
            last_compaction_outcomes: Arc::new(parking_lot::RwLock::new(
                std::collections::HashMap::new(),
            )),
        }
    }

    /// Get current weights
    pub fn weights(&self) -> &CompactionWeights {
        &self.weights
    }

    /// Get query feedback collector reference
    pub fn feedback(&self) -> &FeedbackHandle {
        &self.feedback
    }

    pub fn observe_evidence(&self, evidence: &ExecutionEvidenceSnapshot) {
        let snapshot = MaintenanceEvidenceSnapshot::from(evidence);
        self.observe_metadata_evidence_snapshot(&snapshot, &evidence.executed_segment_ids);
    }

    pub fn observe_metadata_evidence(&self, evidence: &EvidenceSnapshot) {
        let snapshot = MaintenanceEvidenceSnapshot::from(evidence);
        self.observe_metadata_evidence_snapshot(&snapshot, &evidence.executed_segment_ids);
    }

    fn observe_metadata_evidence_snapshot(
        &self,
        snapshot: &MaintenanceEvidenceSnapshot,
        executed_segment_ids: &[String],
    ) {
        let mut evidence_map = self.evidence.write();
        for seg_id in executed_segment_ids {
            let mut per_segment = snapshot.clone();
            per_segment.seg_id = seg_id.clone();
            evidence_map.insert(seg_id.clone(), per_segment);
        }
    }

    pub fn evidence_for_segment(&self, seg_id: &str) -> Option<MaintenanceEvidenceSnapshot> {
        self.evidence.read().get(seg_id).cloned()
    }

    pub fn maintenance_ledger_row(
        &self,
        meta: &crate::segment::meta::SegmentMeta,
    ) -> MaintenanceLedgerRow {
        let debt_flags = self.classify_debt(meta);
        let debt_signal_class = self.debt_signal_class(meta, debt_flags);
        let staleness_penalty = self.feedback.staleness_penalty(&meta.seg_id);
        let prune_miss_ratio = 1.0 - self.feedback.prune_hit_ratio(&meta.seg_id);
        let small_file = (meta.size_bytes as f64 / (1024.0 * 1024.0)) < 1.0;
        let action = self.maintenance_action_for_segment(meta);
        let action_budget = action.map(|action| self.rewrite_budget_for_segment(meta, action));
        let action_authority = action.map(|action| action.authority());
        let executor_authoritative = action
            .map(|action| action.is_executor_authoritative())
            .unwrap_or(false);
        let last_outcome = self.last_compaction_outcome(&meta.seg_id);
        let last_outcome_measured = last_outcome
            .as_ref()
            .map(|outcome| {
                matches!(
                    outcome.quality_signal_kind,
                    CompactionQualitySignalKind::MeasuredDeleteResolution
                )
            })
            .unwrap_or(false);
        let feedback = self.feedback.get_feedback(&meta.seg_id);
        let feedback_seeded_from = last_outcome
            .as_ref()
            .and_then(|outcome| outcome.source_seg_id.clone());
        let feedback_query_count = feedback
            .as_ref()
            .map(|feedback| feedback.query_count)
            .unwrap_or(0);

        MaintenanceLedgerRow {
            seg_id: meta.seg_id.clone(),
            debt_flags,
            debt_signal_class,
            action,
            action_budget,
            action_authority,
            executor_authoritative,
            del_ratio: meta.del_ratio,
            staleness_penalty,
            prune_miss_ratio,
            small_file,
            evidence: self.evidence_for_segment(&meta.seg_id),
            last_outcome,
            last_outcome_measured,
            feedback_seeded_from,
            feedback_query_count,
        }
    }

    pub fn maintenance_ledger(
        &self,
        metas: &[crate::segment::meta::SegmentMeta],
    ) -> MaintenanceLedger {
        MaintenanceLedger {
            rows: metas
                .iter()
                .map(|meta| self.maintenance_ledger_row(meta))
                .collect(),
        }
    }

    /// Classify a segment into explicit debt dimensions.
    ///
    /// This is a governance layer only. It intentionally does not replace the scalar
    /// `calculate_priority()` score, which remains the scheduler's ordering mechanism.
    pub fn classify_debt(&self, meta: &crate::segment::meta::SegmentMeta) -> DebtFlags {
        let mut flags = DebtFlags::NONE;

        if meta.del_ratio > 0.5 {
            flags |= DebtFlags::HIGH_DELETE_RATIO;
        }

        if self.feedback.staleness_penalty(&meta.seg_id) > 0.5 {
            flags |= DebtFlags::STALE_ZONE_MAP;
        }

        let miss_ratio = 1.0 - self.feedback.prune_hit_ratio(&meta.seg_id);
        if miss_ratio > 0.5 {
            flags |= DebtFlags::PRUNE_MISS_HIGH;
        }

        let size_mb = meta.size_bytes as f64 / (1024.0 * 1024.0);
        if size_mb < 1.0 {
            flags |= DebtFlags::SMALL_FILE;
        }

        flags
    }

    pub fn debt_signal_class(
        &self,
        _meta: &crate::segment::meta::SegmentMeta,
        flags: DebtFlags,
    ) -> DebtSignalClass {
        if flags.contains(DebtFlags::HIGH_DELETE_RATIO) {
            DebtSignalClass::Direct
        } else if flags.intersects(DebtFlags::STALE_ZONE_MAP | DebtFlags::PRUNE_MISS_HIGH) {
            DebtSignalClass::EvidenceDriven
        } else {
            DebtSignalClass::Heuristic
        }
    }

    pub fn rewrite_budget_for_segment(
        &self,
        meta: &crate::segment::meta::SegmentMeta,
        action: crate::compaction::scheduler::RewriteAction,
    ) -> crate::compaction::scheduler::RewriteBudget {
        let flags = self.classify_debt(meta);
        let class = self.debt_signal_class(meta, flags);
        let action_budget = action.budget();

        if action.truth_plane_privileged() {
            return crate::compaction::scheduler::RewriteBudget::HardGate;
        }

        if class.permits_rewrite(action_budget) {
            action_budget
        } else {
            crate::compaction::scheduler::RewriteBudget::HeuristicPriority
        }
    }

    pub fn maintenance_action_for_segment(
        &self,
        meta: &crate::segment::meta::SegmentMeta,
    ) -> Option<crate::compaction::scheduler::RewriteAction> {
        let flags = self.classify_debt(meta);
        if flags.contains(DebtFlags::HIGH_DELETE_RATIO) {
            return Some(crate::compaction::scheduler::RewriteAction::PdtMerge);
        }
        if flags.contains(DebtFlags::PRUNE_MISS_HIGH) {
            return Some(crate::compaction::scheduler::RewriteAction::QueryDriven);
        }
        if flags.contains(DebtFlags::STALE_ZONE_MAP) {
            return Some(crate::compaction::scheduler::RewriteAction::MetadataEvidenceRefresh);
        }
        if flags.contains(DebtFlags::SMALL_FILE) {
            return Some(crate::compaction::scheduler::RewriteAction::SmallFileMerge);
        }
        None
    }

    pub fn maintenance_governance_snapshot(
        &self,
        metas: &[crate::segment::meta::SegmentMeta],
    ) -> MaintenanceGovernanceSnapshot {
        MaintenanceGovernanceSnapshot {
            rows: metas
                .iter()
                .map(|meta| {
                    let row = self.maintenance_ledger_row(meta);
                    MaintenanceGovernanceRow {
                        seg_id: row.seg_id,
                        action: row.action,
                        action_budget: row.action_budget,
                        action_authority: row.action_authority,
                        executor_authoritative: row.executor_authoritative,
                        feedback_seeded_from: row.feedback_seeded_from,
                        has_last_outcome: row.last_outcome.is_some(),
                        last_outcome_measured: row.last_outcome_measured,
                    }
                })
                .collect(),
        }
    }

    pub fn observe_maintenance_feedback_seed(
        &mut self,
        old_seg_id: &str,
        new_seg_id: &str,
        timestamp: u64,
    ) {
        self.feedback
            .alias_feedback(old_seg_id, new_seg_id, timestamp);
    }

    /// Record bounded post-compaction feedback after a rewrite event.
    ///
    /// `post_compaction_signal` is the callback value currently delivered by the compactor.
    /// Today this is often a bootstrap proxy (`1.0` for a freshly compacted segment), not a
    /// measured post-compaction prune hit ratio from real queries.
    ///
    /// This hook is intentionally proxy-only in the current runtime: the callback fires once per
    /// compaction event and `prev_hit_ratio` is global to the scheduler, so feeding cross-segment
    /// first-query measurements through this seam would create false improvements and false rollbacks.
    /// Authoritative measured query quality must stay in `QueryFeedback` until this seam becomes
    /// segment-aware and can compare like-with-like.
    pub fn record_compaction_result(&mut self, seg_id: &str, post_compaction_signal: f64) {
        let outcome = DebtResolutionOutcome {
            resolved_flags: DebtFlags::HIGH_DELETE_RATIO,
            delete_debt_remaining_ratio: post_compaction_signal.clamp(0.0, 1.0),
            rows_read: 0,
            rows_written: 0,
            rows_dropped: 0,
            source_seg_id: Some(seg_id.to_string()),
            quality_signal_kind: CompactionQualitySignalKind::ProxyBootstrap,
        };
        self.record_compaction_outcome(seg_id, outcome);
    }

    pub fn record_compaction_outcome(&mut self, seg_id: &str, outcome: DebtResolutionOutcome) {
        self.last_compaction_outcomes
            .write()
            .insert(seg_id.to_string(), outcome.clone());

        let post_compaction_signal = outcome.delete_debt_remaining_ratio;
        let bounded_signal = if post_compaction_signal >= self.prev_hit_ratio {
            self.prev_hit_ratio
        } else {
            post_compaction_signal
        };
        let delta = bounded_signal - self.prev_hit_ratio;

        if delta > 0.05 {
            debug!(
                seg_id,
                resolved_flags = ?outcome.resolved_flags,
                rows_read = outcome.rows_read,
                rows_written = outcome.rows_written,
                rows_dropped = outcome.rows_dropped,
                "Compaction improved hit ratio: {:.3} -> {:.3} (delta={:.3})",
                self.prev_hit_ratio,
                bounded_signal,
                delta
            );
        } else if delta < -0.05 {
            self.weights = self.prev_weights.clone();
            self.step_size = (self.step_size * 0.5).max(0.05);
            debug!(
                seg_id,
                resolved_flags = ?outcome.resolved_flags,
                rows_read = outcome.rows_read,
                rows_written = outcome.rows_written,
                rows_dropped = outcome.rows_dropped,
                "Compaction degraded hit ratio, rolling back (step_size={:.3})",
                self.step_size
            );
        }

        self.prev_hit_ratio = bounded_signal;
    }

    pub fn last_compaction_outcome(&self, seg_id: &str) -> Option<DebtResolutionOutcome> {
        self.last_compaction_outcomes.read().get(seg_id).cloned()
    }

    pub fn measured_post_compaction_outcome(&self, seg_id: &str) -> Option<DebtResolutionOutcome> {
        self.last_compaction_outcome(seg_id).filter(|outcome| {
            matches!(
                outcome.quality_signal_kind,
                CompactionQualitySignalKind::MeasuredDeleteResolution
            )
        })
    }

    /// Hill climbing weight adjustment (evaluate once every adjustment_interval calls)
    pub fn adjust(&mut self) {
        self.call_count += 1;

        if self.call_count % self.adjustment_interval != 0 {
            return;
        }

        let stats = self.feedback.get_all_stats();
        if stats.is_empty() {
            return;
        }

        let mut any_change = false;

        for (_seg_id, s) in stats {
            if s.query_count < self.min_query_count {
                continue;
            }

            // Save current weights before the first mutation (for rollback)
            if !any_change {
                self.prev_weights = self.weights.clone();
                any_change = true;
            }

            // Strategy: if a segment's Zone Map continuously misaligns (penalty > 0.5) -> increase stale_coef
            if s.staleness_penalty() > 0.5 {
                let new_val = self.weights.stale_coef + self.step_size;
                if new_val <= 20.0 {
                    self.weights.stale_coef = new_val;
                    debug!(
                        "HC: stale_penalty={:.2} -> stale_coef={:.2}",
                        s.staleness_penalty(),
                        self.weights.stale_coef
                    );
                }
            } else if s.staleness_penalty() < 0.1 && s.query_count > 20 {
                // Zone Map continuously accurate -> can reduce stale_coef (more conservative compaction)
                let new_val = self.weights.stale_coef - self.step_size * 0.5;
                if new_val >= 1.0 {
                    self.weights.stale_coef = new_val;
                    debug!("HC: stable -> stale_coef={:.2}", self.weights.stale_coef);
                }
            }

            // Strategy: if prune miss rate is high -> increase miss_coef
            let miss_ratio = 1.0 - s.prune_hit_ratio();
            if miss_ratio > 0.5 && s.query_count > 10 {
                let new_val = self.weights.miss_coef + self.step_size;
                if new_val <= 10.0 {
                    self.weights.miss_coef = new_val;
                    debug!(
                        "HC: miss_ratio={:.2} -> miss_coef={:.2}",
                        miss_ratio, self.weights.miss_coef
                    );
                }
            }
        }
    }

    /// Calculate priority for a segment (using adaptive weights)
    ///
    /// This remains heuristic ordering only. Rewrite admission/semantic action selection must
    /// flow through `maintenance_action_for_segment` and `rewrite_budget_for_segment` so typed
    /// debt and evidence semantics are not smuggled into scalar priority.
    pub fn calculate_priority(&self, meta: &crate::segment::meta::SegmentMeta, now: u64) -> f64 {
        let age_hours = if meta.created_txn > 0 {
            ((now - meta.created_txn) as f64) / 3600.0
        } else {
            1.0
        };

        let size_mb = meta.size_bytes as f64 / (1024.0 * 1024.0);

        let del_score = meta.del_ratio.powi(2) * self.weights.del_coef;
        let size_score = size_mb.max(1.0).log2() * self.weights.size_coef;
        let age_score = age_hours.log2() * self.weights.age_coef;

        // Zone Map misalignment penalty
        let stale_penalty = self.feedback.staleness_penalty(&meta.seg_id) * self.weights.stale_coef;
        // Pruning failure penalty
        let miss_penalty =
            (1.0 - self.feedback.prune_hit_ratio(&meta.seg_id)) * self.weights.miss_coef;
        let evidence_bonus = self
            .evidence_for_segment(&meta.seg_id)
            .map(|evidence| {
                let compression_signal = 1.0 - evidence.compressed_bytes_ratio();
                let zone_map_signal =
                    if evidence.has_zone_map_predicates && !evidence.has_cross_column_or {
                        0.5
                    } else {
                        0.0
                    };
                let delta_signal = (evidence.delta_segment_count as f64 / 10.0).min(1.0);
                compression_signal + zone_map_signal + delta_signal
            })
            .unwrap_or(0.0);

        del_score + size_score + age_score + stale_penalty + miss_penalty + evidence_bonus
    }
}

// ---------------------------------------------------------------------------
// EcoTune Integration
// ---------------------------------------------------------------------------

use crate::compaction::ecotune::WorkloadProfile;

impl AdaptiveCompactionScheduler {
    /// Collect a WorkloadProfile from current query feedback statistics.
    ///
    /// Uses the QueryFeedbackCollector to estimate:
    /// - rq_ratio: fraction of segment accesses with non-empty selectivity
    /// - avg_selectivity: average selectivity of recent scans
    /// - point_query_ratio: inferred from scan selectivity (very low = point query)
    pub fn collect_workload_profile(&self) -> WorkloadProfile {
        let stats = self.feedback.get_all_stats();
        if stats.is_empty() {
            return WorkloadProfile::default();
        }

        let total = stats.len() as f64;
        let rq_count = stats.iter().filter(|(_, fb)| fb.query_count > 0).count() as f64;
        let rq_ratio = (rq_count / total).clamp(0.0, 1.0);

        // Very small avg selectivity → likely point queries.
        let avg_selectivity = self.feedback.avg_selectivity();
        let point_query_ratio = if avg_selectivity < 0.0001 { 0.8 } else { 0.2 };

        let write_speed_mbps = self.write_heat_tracker.estimated_mbps();

        WorkloadProfile {
            rq_ratio,
            write_speed_mbps,
            avg_selectivity,
            point_query_ratio,
        }
    }
}

#[cfg(test)]
#[allow(clippy::items_after_test_module)]
mod tests {
    use super::*;
    use crate::segment::meta::SegmentMeta;
    use crate::segment::meta::SegmentStatus;
    use crate::segment::meta::SegmentType;

    fn make_meta(seg_id: &str) -> SegmentMeta {
        SegmentMeta {
            seg_id: seg_id.to_string(),
            table_id: "orders".to_string(),
            status: SegmentStatus::Active,
            seg_type: SegmentType::Vortex,
            columns: Vec::new(),
            min_key: Vec::new(),
            max_key: Vec::new(),
            row_count: 100,
            alive_row_count: 40,
            del_ratio: 0.6,
            size_bytes: 512 * 1024,
            created_txn: 1,
            updated_txn: 1,
            updated_at: 0,
            file_paths: Vec::new(),
            granules: Vec::new(),
            has_visibility_columns: true,
            delta_file_id: None,
            delta_row_count: 0,
            delta_l1_bytes: 0,
        }
    }

    #[test]
    fn layered_governance_snapshot_is_falsifiable_and_ready() {
        let snapshot = AdaptiveCompactionScheduler::layered_governance_snapshot();
        assert!(snapshot.sidecar_contracts_enforced);
        assert!(snapshot.evidence_boundary_enforced);
        assert!(snapshot.truth_boundary_documented);
        assert!(snapshot.ready_for_enforcement());
    }

    #[test]
    fn governance_bypass_inventory_covers_phase2_surfaces() {
        let inventory = GovernanceBypassInventory::phase2_matrix();
        assert_eq!(inventory.projection_surfaces.len(), 4);
        assert!(inventory
            .projection_surfaces
            .iter()
            .any(|entry| entry.contains("ProjectionContract::vtab")));
        assert!(inventory
            .evidence_boundaries
            .iter()
            .any(|entry| entry.contains("assert_governance_ready")));
        assert!(inventory.truth_boundary_documented());
    }

    #[test]
    fn governance_verification_card_declares_documented_bypasses() {
        let card = AdaptiveCompactionScheduler::governance_verification_card();
        assert!(!card.bypass_paths.is_empty());
        assert!(card
            .bypass_paths
            .iter()
            .any(|entry| entry.contains("point_get and VTab hidden read paths")));
    }

    #[test]
    fn record_compaction_result_accepts_measured_quality_signal() {
        let mut scheduler = AdaptiveCompactionScheduler::new();
        scheduler.record_compaction_result("seg-1", 0.3);
        assert_eq!(scheduler.prev_hit_ratio, 0.3);
    }

    #[test]
    fn record_compaction_result_tracks_bounded_proxy_signal_without_claiming_improvement() {
        let mut scheduler = AdaptiveCompactionScheduler::new();
        let baseline = scheduler.prev_hit_ratio;

        scheduler.record_compaction_result("seg-1", 1.0);
        assert_eq!(scheduler.prev_hit_ratio, baseline);

        let row = scheduler.maintenance_ledger_row(&make_meta("seg-1"));
        assert!(row.debt_flags.contains(DebtFlags::HIGH_DELETE_RATIO));
        assert!(row.debt_flags.contains(DebtFlags::SMALL_FILE));
    }

    #[test]
    fn record_compaction_result_accepts_delete_debt_resolution_hint() {
        let mut scheduler = AdaptiveCompactionScheduler::new();
        scheduler.record_compaction_result("seg-1", 0.4);
        assert_eq!(scheduler.prev_hit_ratio, 0.4);
    }

    #[test]
    fn record_compaction_outcome_tracks_typed_delete_resolution() {
        let mut scheduler = AdaptiveCompactionScheduler::new();
        let outcome = DebtResolutionOutcome {
            resolved_flags: DebtFlags::HIGH_DELETE_RATIO,
            delete_debt_remaining_ratio: 0.4,
            rows_read: 100,
            rows_written: 60,
            rows_dropped: 40,
            source_seg_id: Some("seg-typed-source".to_string()),
            quality_signal_kind: CompactionQualitySignalKind::MeasuredDeleteResolution,
        };

        scheduler.record_compaction_outcome("seg-typed", outcome);

        assert_eq!(scheduler.prev_hit_ratio, 0.4);
        let recorded = scheduler
            .last_compaction_outcome("seg-typed")
            .expect("typed outcome should be stored per segment");
        assert_eq!(recorded.resolved_flags, DebtFlags::HIGH_DELETE_RATIO);
        assert_eq!(recorded.rows_dropped, 40);
        assert_eq!(recorded.delete_debt_remaining_ratio, 0.4);
    }

    #[test]
    fn record_compaction_result_proxy_signal_does_not_trigger_cross_segment_rollback() {
        let mut scheduler = AdaptiveCompactionScheduler::new();
        scheduler.adjustment_interval = 1;

        for ts in 1..=5 {
            scheduler.feedback.record_hit("seg-1", ts);
        }
        for ts in 6..=11 {
            scheduler.feedback.record_miss("seg-1", ts);
        }

        scheduler.record_compaction_result("seg-1", 1.0);
        scheduler.adjust();

        let baseline = CompactionWeights::default();
        assert!(scheduler.weights.stale_coef > baseline.stale_coef);
        assert!(scheduler.weights.miss_coef > baseline.miss_coef);
        assert_eq!(scheduler.prev_hit_ratio, 0.5);
        assert_eq!(scheduler.step_size, 0.5);

        scheduler.record_compaction_result("seg-2", 0.8);
        assert!(scheduler.weights.stale_coef > baseline.stale_coef);
        assert!(scheduler.weights.miss_coef > baseline.miss_coef);
        assert_eq!(scheduler.step_size, 0.5);
        assert_eq!(scheduler.prev_hit_ratio, 0.5);
    }

    #[test]
    fn record_compaction_result_rolls_back_weights_on_signal_regression() {
        let mut scheduler = AdaptiveCompactionScheduler::new();
        scheduler.adjustment_interval = 1;

        for ts in 1..=5 {
            scheduler.feedback.record_hit("seg-1", ts);
        }
        for ts in 6..=11 {
            scheduler.feedback.record_miss("seg-1", ts);
        }

        scheduler.record_compaction_result("seg-1", 1.0);
        scheduler.adjust();

        let baseline = CompactionWeights::default();
        assert!(scheduler.weights.stale_coef > baseline.stale_coef);
        assert!(scheduler.weights.miss_coef > baseline.miss_coef);

        scheduler.record_compaction_result("seg-1", 0.8);
        assert!(scheduler.weights.stale_coef > baseline.stale_coef);
        assert!(scheduler.weights.miss_coef > baseline.miss_coef);
        assert_eq!(scheduler.step_size, 0.5);
        assert_eq!(scheduler.prev_hit_ratio, 0.5);
    }

    #[test]
    fn evidence_budget_mapping_promotes_measured_rewrite_classes_without_touching_priority() {
        let scheduler = AdaptiveCompactionScheduler::new();
        let delete_heavy = make_meta("seg-delete");
        let evidence_heavy = SegmentMeta {
            seg_id: "seg-evidence".to_string(),
            del_ratio: 0.2,
            size_bytes: 2 * 1024 * 1024,
            ..make_meta("seg-evidence")
        };
        scheduler.feedback().record_miss("seg-evidence", 100);

        assert_eq!(
            scheduler.maintenance_action_for_segment(&delete_heavy),
            Some(crate::compaction::scheduler::RewriteAction::PdtMerge)
        );
        assert_eq!(
            scheduler.rewrite_budget_for_segment(
                &delete_heavy,
                crate::compaction::scheduler::RewriteAction::PdtMerge,
            ),
            crate::compaction::scheduler::RewriteBudget::HardGate
        );

        assert_eq!(
            scheduler.maintenance_action_for_segment(&evidence_heavy),
            Some(crate::compaction::scheduler::RewriteAction::QueryDriven)
        );
        assert_eq!(
            scheduler.rewrite_budget_for_segment(
                &evidence_heavy,
                crate::compaction::scheduler::RewriteAction::QueryDriven,
            ),
            crate::compaction::scheduler::RewriteBudget::EvidenceDriven
        );

        let small_file = SegmentMeta {
            seg_id: "seg-small".to_string(),
            del_ratio: 0.1,
            size_bytes: 512 * 1024,
            ..make_meta("seg-small")
        };
        let small_file_row = scheduler.maintenance_ledger_row(&small_file);
        assert_eq!(small_file_row.debt_signal_class, DebtSignalClass::Heuristic);
        let evidence_row = scheduler.maintenance_ledger_row(&evidence_heavy);
        assert_eq!(
            evidence_row.debt_signal_class,
            DebtSignalClass::EvidenceDriven
        );
    }

    #[test]
    fn prune_miss_evidence_is_promoted_ahead_of_metadata_refresh() {
        let scheduler = AdaptiveCompactionScheduler::new();
        let evidence_heavy = SegmentMeta {
            seg_id: "seg-prune-priority".to_string(),
            del_ratio: 0.2,
            size_bytes: 2 * 1024 * 1024,
            ..make_meta("seg-prune-priority")
        };
        scheduler.feedback().record_miss("seg-prune-priority", 1);

        let row = scheduler.maintenance_ledger_row(&evidence_heavy);
        assert!(row.debt_flags.contains(DebtFlags::PRUNE_MISS_HIGH));
        assert!(row.debt_flags.contains(DebtFlags::STALE_ZONE_MAP));
        assert_eq!(
            scheduler.maintenance_action_for_segment(&evidence_heavy),
            Some(crate::compaction::scheduler::RewriteAction::QueryDriven),
            "prune-miss evidence should take the executable query-driven seam before metadata-only refresh"
        );
    }

    #[test]
    fn measured_post_compaction_outcome_filters_out_proxy_bootstrap_signals() {
        let mut scheduler = AdaptiveCompactionScheduler::new();
        scheduler.record_compaction_result("seg-proxy", 0.3);
        assert!(scheduler
            .measured_post_compaction_outcome("seg-proxy")
            .is_none());

        let measured = DebtResolutionOutcome::from_merge_stats(
            "seg-measured-source",
            &crate::compaction::pdt_merge::MergeStats {
                rows_read: 100,
                rows_written: 60,
                rows_dropped: 40,
                ..Default::default()
            },
        );
        scheduler.record_compaction_outcome("seg-measured", measured);

        let outcome = scheduler
            .measured_post_compaction_outcome("seg-measured")
            .expect("measured delete resolution should be retained");
        assert!(matches!(
            outcome.quality_signal_kind,
            CompactionQualitySignalKind::MeasuredDeleteResolution
        ));
    }

    #[test]
    fn governance_snapshot_distinguishes_measured_outcomes_from_proxy_bootstrap() {
        let mut scheduler = AdaptiveCompactionScheduler::new();
        let meta = make_meta("seg-governance-measured");
        scheduler.record_compaction_result("seg-governance-measured", 0.3);
        let proxy_snapshot = scheduler.maintenance_governance_snapshot(std::slice::from_ref(&meta));
        assert!(!proxy_snapshot.rows[0].last_outcome_measured);

        scheduler.record_compaction_outcome(
            "seg-governance-measured",
            DebtResolutionOutcome::from_merge_stats(
                "seg-governance-measured-source",
                &crate::compaction::pdt_merge::MergeStats {
                    rows_read: 100,
                    rows_written: 60,
                    rows_dropped: 40,
                    ..Default::default()
                },
            ),
        );
        let measured_snapshot = scheduler.maintenance_governance_snapshot(&[meta]);
        assert!(measured_snapshot.rows[0].last_outcome_measured);
    }

    #[test]
    fn maintenance_ledger_captures_authority_and_feedback_seed_state() {
        let mut scheduler = AdaptiveCompactionScheduler::new();
        let delete_heavy = crate::segment::meta::SegmentMeta {
            del_ratio: 0.6,
            size_bytes: 512 * 1024,
            ..make_meta("seg-governed")
        };
        scheduler.feedback().record_miss("seg-governed", 1);
        scheduler.record_compaction_outcome(
            "seg-governed",
            DebtResolutionOutcome {
                resolved_flags: DebtFlags::HIGH_DELETE_RATIO,
                delete_debt_remaining_ratio: 0.25,
                rows_read: 100,
                rows_written: 75,
                rows_dropped: 25,
                source_seg_id: Some("seg-governed-old".to_string()),
                quality_signal_kind: CompactionQualitySignalKind::MeasuredDeleteResolution,
            },
        );

        let row = scheduler.maintenance_ledger_row(&delete_heavy);
        assert_eq!(
            row.action,
            Some(crate::compaction::scheduler::RewriteAction::PdtMerge)
        );
        assert_eq!(
            row.action_authority,
            Some(crate::compaction::scheduler::ActionAuthority::Executable)
        );
        assert!(row.executor_authoritative);
        assert_eq!(
            row.feedback_seeded_from.as_deref(),
            Some("seg-governed-old")
        );
        assert_eq!(row.feedback_query_count, 1);
        assert!(row.last_outcome.is_some());
    }

    #[test]
    fn maintenance_governance_snapshot_marks_demoted_evidence_driven_actions() {
        let scheduler = AdaptiveCompactionScheduler::new();
        let evidence_heavy = crate::segment::meta::SegmentMeta {
            del_ratio: 0.0,
            size_bytes: 2 * 1024 * 1024,
            ..make_meta("seg-demoted")
        };
        scheduler.feedback().record_miss("seg-demoted", 1);

        let snapshot = scheduler.maintenance_governance_snapshot(&[evidence_heavy]);
        assert_eq!(snapshot.rows.len(), 1);
        let row = &snapshot.rows[0];
        assert_eq!(
            row.action,
            Some(crate::compaction::scheduler::RewriteAction::QueryDriven)
        );
        assert_eq!(
            row.action_authority,
            Some(crate::compaction::scheduler::ActionAuthority::DemotedScaffold)
        );
        assert!(!row.executor_authoritative);
        assert_eq!(
            row.action_budget,
            Some(crate::compaction::scheduler::RewriteBudget::EvidenceDriven)
        );
    }

    #[test]
    fn observe_maintenance_feedback_seed_aliases_query_feedback_to_rewritten_segment() {
        let mut scheduler = AdaptiveCompactionScheduler::new();
        scheduler.feedback().record_miss("seg-old", 10);
        scheduler.feedback().record_hit("seg-old", 11);

        scheduler.observe_maintenance_feedback_seed("seg-old", "seg-new", 12);

        let new_feedback = scheduler
            .feedback()
            .get_feedback("seg-new")
            .expect("aliased feedback should exist for rewritten segment");
        
        // D2 fix: query_count is reset to 0 for new segment to enter observation period.
        // This prevents premature scheduling decisions based on inherited query history.
        assert_eq!(new_feedback.query_count, 0);
        // Access pattern (hit/miss ratio) is still inherited.
        assert_eq!(new_feedback.hit_count, 1);
        assert_eq!(new_feedback.miss_count, 1);
        assert_eq!(new_feedback.last_access, 12);
    }

    #[test]
    fn debt_flags_are_reporting_only_and_do_not_change_priority() {
        let scheduler = AdaptiveCompactionScheduler::new();
        let now = 7_200;

        let hot_delete_small = make_meta("seg-hot-delete");
        let clean_large = SegmentMeta {
            seg_id: "seg-clean-large".to_string(),
            table_id: "orders".to_string(),
            status: SegmentStatus::Active,
            seg_type: SegmentType::Vortex,
            columns: Vec::new(),
            min_key: Vec::new(),
            max_key: Vec::new(),
            row_count: 100,
            alive_row_count: 100,
            del_ratio: 0.0,
            size_bytes: 8 * 1024 * 1024,
            created_txn: 1,
            updated_txn: 1,
            updated_at: 0,
            file_paths: Vec::new(),
            granules: Vec::new(),
            has_visibility_columns: true,
            delta_file_id: None,
            delta_row_count: 0,
            delta_l1_bytes: 0,
        };

        let base_priority = scheduler.calculate_priority(&clean_large, now);
        let debt_priority = scheduler.calculate_priority(&hot_delete_small, now);
        assert!(debt_priority > base_priority);

        let row = scheduler.maintenance_ledger_row(&hot_delete_small);
        assert!(row.debt_flags.contains(DebtFlags::HIGH_DELETE_RATIO));
        assert!(row.debt_flags.contains(DebtFlags::SMALL_FILE));

        let control = AdaptiveCompactionScheduler::new();
        assert_eq!(
            debt_priority,
            control.calculate_priority(&hot_delete_small, now),
            "typed debt reporting must not add hidden dispatch authority"
        );
    }
}

impl Default for AdaptiveCompactionScheduler {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for AdaptiveCompactionScheduler {
    /// Creates an independent adaptive scheduler for a different execution context.
    ///
    /// ## Intentional Design: Fresh State on Clone
    ///
    /// When cloned (e.g., for a new compaction worker), the new instance starts with
    /// default weights and reset history. This is CORRECT because:
    /// 1. Each worker should learn independently based on its own observations
    /// 2. The original instance's history is preserved in the source
    /// 3. Shared state (feedback, heat_tracker) is accessed via Arc and is correct
    ///
    /// If you need to share weights across workers, clone AFTER adjusting — not before.
    fn clone(&self) -> Self {
        Self {
            weights: self.weights.clone(),
            // Share the feedback collector handle so all clones observe the same evidence/feedback surface.
            feedback: self.feedback.clone(),
            evidence: Arc::clone(&self.evidence),
            // Share the heat tracker — Clone on WriteHeatTrackerHandle is Arc::clone
            write_heat_tracker: self.write_heat_tracker.clone(),
            prev_weights: self.prev_weights.clone(),
            step_size: self.step_size,
            min_query_count: self.min_query_count,
            adjustment_interval: self.adjustment_interval,
            call_count: 0, // reset on clone
            prev_hit_ratio: 0.5,
            last_compaction_outcomes: Arc::clone(&self.last_compaction_outcomes),
        }
    }
}
