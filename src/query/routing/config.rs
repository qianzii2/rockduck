//! Router configuration - all magic thresholds are now parameters.
//!
//! See veDB-HTAP (VLDB 2025) for the production-validated threshold ranges.
//! See PolarDB IMCI `loose_cost_threshold_for_imci` for a proven single-threshold approach.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MlRoutingPromotionGate {
    pub live_authority_enabled: bool,
    pub min_measured_samples: usize,
    pub min_shadow_agreement_ratio: f64,
    pub max_disagreement_ratio: f64,
    pub shadow_sample_window: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ExportRolloutConfig {
    pub runtime_enabled: bool,
    pub external_reader_compat_required: bool,
}

impl Default for ExportRolloutConfig {
    fn default() -> Self {
        Self {
            runtime_enabled: false,
            external_reader_compat_required: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct CdcSinkRolloutConfig {
    pub kafka_runtime_enabled: bool,
}

impl Default for MlRoutingPromotionGate {
    fn default() -> Self {
        Self {
            live_authority_enabled: false,
            min_measured_samples: 32,
            min_shadow_agreement_ratio: 0.80,
            max_disagreement_ratio: 0.20,
            shadow_sample_window: 500,
        }
    }
}

/// All router thresholds and parameters. Wired into `RockDuckConfig`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterConfig {
    // === Tier 1 thresholds ===
    /// Selectivity below this -> DeltaStore (point/short-range query).
    /// Default: 0.01 (1% of rows).
    pub point_selectivity_thresh: f64,
    /// Selectivity above this -> Vortex (full columnar scan).
    /// Default: 0.10 (10% of rows).
    pub full_scan_selectivity_thresh: f64,
    /// Delta count above this -> Merge instead of pure DeltaStore.
    /// Default: 100 deltas (beyond this the PK-tree lookup overhead dominates).
    pub delta_overload_thresh: usize,
    /// Aggregate selectivity threshold: above this -> Vortex.
    /// Default: 0.10.
    pub aggregate_selectivity_thresh: f64,

    // === Tier 2 cost model parameters ===
    /// CPU cost per row (in relative cost units).
    /// Represents the CPU work to decode/filter one row.
    /// Calibrated against TPC-H Q1 on this hardware.
    /// Default: 1.0.
    pub cpu_cost_per_row: f64,
    /// I/O cost per byte (in relative cost units).
    /// Represents the cost of reading 1 byte from storage.
    /// Columnar (Vortex): benefits from compression -> effective bytes < raw bytes.
    /// Default: 0.001 (calibrated for NVMe).
    pub io_cost_per_byte: f64,
    /// Fixed cost of one PK-tree lookup in DeltaStore (in relative units).
    /// Each delta in DeltaStore requires a PK-tree lookup during Merge.
    /// Default: 10.0 (one PK lookup approx 10 row scans).
    pub delta_lookup_cost: f64,
    /// Overhead multiplier for Merge vs pure Vortex.
    /// Merge must sort/merge results - extra O(N log N) work.
    /// Default: 1.5 (50% overhead over pure Vortex scan).
    pub merge_overhead_factor: f64,
    /// Default compression ratio for Vortex columnar storage.
    /// Updated dynamically from BlockStats by Tier 2.
    /// Default: 0.5 (2x compression).
    pub columnar_compression_ratio: f64,

    // === Tier 3 ML parameters ===
    /// Enable ML routing via Tree-CNN (candle CPU inference).
    /// Default: false (Tier 1 + 2 only until model is trained).
    pub ml_enabled: bool,
    /// Minimum ML confidence to trust the ML decision.
    /// Below this threshold, Tier 2 result is used instead.
    /// Default: 0.7.
    ///
    /// Governance note: while `MlRoutingMode` remains advisory-only, this threshold only
    /// shapes observability/reporting and must not grant live routing authority.
    pub ml_confidence_thresh: f64,
    /// Maximum time allowed for ML inference (ms).
    /// If exceeded, fall back to Tier 2.
    /// Default: 5ms.
    pub ml_inference_timeout_ms: u64,
    /// Path to pre-trained Tree-CNN weight file.
    /// Loaded at startup if `ml_enabled = true`.
    /// Default: "assets/tree_cnn_weights.bin".
    pub ml_weight_path: String,
    /// Bounded rollout gate for ML authority promotion.
    /// Defaults to advisory-only until measured evidence and disagreement gates are satisfied.
    pub ml_promotion_gate: MlRoutingPromotionGate,

    // === Feedback / adaptive learning parameters ===
    /// Number of recent observations to keep per column for selectivity estimation.
    /// Sliding window prevents stale distributions from dominating.
    /// Default: 1000.
    pub selectivity_window_size: usize,
    /// How much to weight actual feedback vs. estimate: 0.0 = pure estimate, 1.0 = pure feedback.
    /// ACM (arXiv 2024) suggests 0.3-0.5 for stable convergence.
    /// Default: 0.3.
    pub cost_feedback_weight: f64,
    /// How often to write feedback checkpoint to KV (seconds).
    /// Default: 300 (5 minutes).
    pub checkpoint_interval_secs: u64,
    /// Maximum number of query execution records to keep per table.
    /// Prevents unbounded memory growth for hot tables.
    /// Default: 10_000.
    pub max_feedback_entries: usize,
    /// Observation-only shadow timing sample rate in [0.0, 1.0].
    /// Default: 0.05 to enable bounded low-rate same-query comparison collection.
    pub shadow_sample_rate: f64,
    /// Observation-only shadow timing policy.
    /// `Disabled` means no same-query shadow evidence is collected.
    /// `SyntheticEstimate` means we may record synthetic comparison samples for analysis only.
    /// `BoundedDualPath` collects same-query alternate-path timings as bounded measured evidence.
    pub shadow_timing_policy: crate::query::routing::feedback::ShadowTimingPolicy,
    /// Default bounded cooperative runtime slice for Merge/Historical scans.
    pub cooperative_runtime_budget: crate::query::routing::CooperativeRuntimeBudget,
    /// Maximum number of durable ML samples buffered before export.
    pub ml_export_threshold: usize,
    /// Runtime rollout controls for outward Iceberg export.
    pub export_rollout: ExportRolloutConfig,
    /// Runtime rollout controls for outward CDC sinks.
    pub cdc_sink_rollout: CdcSinkRolloutConfig,
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self {
            point_selectivity_thresh: 0.01,
            full_scan_selectivity_thresh: 0.10,
            delta_overload_thresh: 100,
            aggregate_selectivity_thresh: 0.10,
            cpu_cost_per_row: 1.0,
            io_cost_per_byte: 0.001,
            delta_lookup_cost: 10.0,
            merge_overhead_factor: 1.5,
            columnar_compression_ratio: 0.5,
            ml_enabled: false,
            ml_confidence_thresh: 0.7,
            ml_inference_timeout_ms: 5,
            ml_weight_path: "assets/tree_cnn_weights.bin".to_string(),
            ml_promotion_gate: MlRoutingPromotionGate::default(),
            selectivity_window_size: 1000,
            cost_feedback_weight: 0.3,
            checkpoint_interval_secs: 300,
            max_feedback_entries: 10_000,
            shadow_sample_rate: 0.05,
            shadow_timing_policy:
                crate::query::routing::feedback::ShadowTimingPolicy::BoundedDualPath,
            cooperative_runtime_budget: crate::query::routing::CooperativeRuntimeBudget {
                max_segments_per_slice: 8,
                max_slice_ms: 25,
            },
            ml_export_threshold: 1_000,
            export_rollout: ExportRolloutConfig::default(),
            cdc_sink_rollout: CdcSinkRolloutConfig::default(),
        }
    }
}

impl RouterConfig {
    pub fn builder() -> RouterConfigBuilder {
        RouterConfigBuilder::new()
    }
}

#[derive(Default)]
pub struct RouterConfigBuilder {
    inner: RouterConfig,
}

impl RouterConfigBuilder {
    pub fn new() -> Self {
        Self {
            inner: RouterConfig::default(),
        }
    }

    pub fn point_selectivity_thresh(mut self, v: f64) -> Self {
        self.inner.point_selectivity_thresh = v;
        self
    }
    pub fn full_scan_selectivity_thresh(mut self, v: f64) -> Self {
        self.inner.full_scan_selectivity_thresh = v;
        self
    }
    pub fn delta_overload_thresh(mut self, v: usize) -> Self {
        self.inner.delta_overload_thresh = v;
        self
    }
    pub fn aggregate_selectivity_thresh(mut self, v: f64) -> Self {
        self.inner.aggregate_selectivity_thresh = v;
        self
    }
    pub fn cpu_cost_per_row(mut self, v: f64) -> Self {
        self.inner.cpu_cost_per_row = v;
        self
    }
    pub fn io_cost_per_byte(mut self, v: f64) -> Self {
        self.inner.io_cost_per_byte = v;
        self
    }
    pub fn delta_lookup_cost(mut self, v: f64) -> Self {
        self.inner.delta_lookup_cost = v;
        self
    }
    pub fn merge_overhead_factor(mut self, v: f64) -> Self {
        self.inner.merge_overhead_factor = v;
        self
    }
    pub fn columnar_compression_ratio(mut self, v: f64) -> Self {
        self.inner.columnar_compression_ratio = v;
        self
    }
    pub fn ml_enabled(mut self, v: bool) -> Self {
        self.inner.ml_enabled = v;
        self
    }
    pub fn ml_confidence_thresh(mut self, v: f64) -> Self {
        self.inner.ml_confidence_thresh = v;
        self
    }
    pub fn ml_inference_timeout_ms(mut self, v: u64) -> Self {
        self.inner.ml_inference_timeout_ms = v;
        self
    }
    pub fn ml_weight_path(mut self, v: String) -> Self {
        self.inner.ml_weight_path = v;
        self
    }
    pub fn ml_promotion_gate(mut self, v: MlRoutingPromotionGate) -> Self {
        self.inner.ml_promotion_gate = v;
        self
    }
    pub fn selectivity_window_size(mut self, v: usize) -> Self {
        self.inner.selectivity_window_size = v;
        self
    }
    pub fn cost_feedback_weight(mut self, v: f64) -> Self {
        self.inner.cost_feedback_weight = v;
        self
    }
    pub fn checkpoint_interval_secs(mut self, v: u64) -> Self {
        self.inner.checkpoint_interval_secs = v;
        self
    }
    pub fn max_feedback_entries(mut self, v: usize) -> Self {
        self.inner.max_feedback_entries = v;
        self
    }
    pub fn shadow_sample_rate(mut self, v: f64) -> Self {
        self.inner.shadow_sample_rate = v.clamp(0.0, 1.0);
        self
    }
    pub fn shadow_timing_policy(
        mut self,
        v: crate::query::routing::feedback::ShadowTimingPolicy,
    ) -> Self {
        self.inner.shadow_timing_policy = v;
        self
    }
    pub fn cooperative_runtime_budget(
        mut self,
        v: crate::query::routing::CooperativeRuntimeBudget,
    ) -> Self {
        self.inner.cooperative_runtime_budget = v;
        self
    }
    pub fn ml_export_threshold(mut self, v: usize) -> Self {
        self.inner.ml_export_threshold = v;
        self
    }
    pub fn export_rollout(mut self, v: ExportRolloutConfig) -> Self {
        self.inner.export_rollout = v;
        self
    }
    pub fn cdc_sink_rollout(mut self, v: CdcSinkRolloutConfig) -> Self {
        self.inner.cdc_sink_rollout = v;
        self
    }

    pub fn build(self) -> RouterConfig {
        self.inner
    }
}
