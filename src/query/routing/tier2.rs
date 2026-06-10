//! Tier 2: Statistical cost model router.
//!
//! Estimates relative execution cost for each path using:
//! - Observed selectivity from `FeedbackState::SelectivityTracker`
//! - Table statistics (`TableStats`)
//! - `CostParams` (adaptively tuned via online feedback)
//!
//! Cost formula (see `cost.rs`):
//!   delta_cost = rows * (cpu_per_row + bytes_per_row * io_per_byte)
//!   vortex_cost = rows * (cpu_per_row * 1.5 + bytes_per_row * comp_ratio * io_per_byte)
//!   merge_cost = delta_count * lookup_cost + vortex_rows * overhead_factor
//!
//! ## Confidence
//!
//! Confidence is derived from:
//! - Selectivity: high confidence if we have many observations for this column.
//! - Delta overload: high confidence if delta_count is very large or zero.
//! - Execution history: high confidence if recent actual times are consistent.

use crate::query::routing::config::RouterConfig;
use crate::query::routing::cost::{
    argmin_path, delta_scan_cost, merge_cost, vortex_scan_cost, CostParams,
};
use crate::query::routing::feedback::{
    ExecutionEvidenceSnapshot, ExecutionHistory, MeasuredCostSample, SelectivityTracker,
};
use crate::query::routing::{RouteDecision, RouterParams, RoutingResult};

/// Statistical cost model router.
pub struct StatisticalRouter;

const MIN_MEASURED_EXECUTION_SAMPLES: usize = 3;
const MIN_SELECTIVITY_SAMPLES: usize = 5;

impl StatisticalRouter {
    pub fn new() -> Self {
        Self
    }

    /// Make a routing decision using the cost model.
    ///
    /// Returns `RoutingResult` with confidence derived from:
    /// - How many selectivity observations we have for this table/column.
    /// - How much the actual execution history agrees with estimates.
    pub fn decide(
        &self,
        params: &RouterParams,
        cfg: &RouterConfig,
        selectivity_tracker: &SelectivityTracker,
        execution_feedback: &crate::query::routing::feedback::FeedbackState,
        cost_params: &CostParams,
    ) -> RoutingResult {
        // Step 1: Get selectivity estimate (from history or from predicate analysis).
        let sel = self.get_selectivity(params, selectivity_tracker);

        // Step 2: Compute estimated scanned rows.
        let rows = params.stats.estimated_scanned_rows(sel);

        // Step 3: Compute cost for each path.
        let delta_cost = delta_scan_cost(rows, params.stats, cost_params);
        let vortex_cost = vortex_scan_cost(rows, params.stats, cost_params);

        // Merge: delta overlay must scan all rows for completeness, but the Vortex base
        // scan only needs to scan the estimated selective rows. The delta lookup cost
        // accounts for the overlay overhead (PK tree lookups for each delta entry).
        let merge_cost_val = merge_cost(params.delta_count, rows, params.stats, cost_params);

        // Step 4: Get actual observed costs from feedback (if available).
        let evidence_guard = execution_feedback.last_evidence.read();
        let evidence = evidence_guard.get(params.table);
        let (delta_actual, vortex_actual) = self.get_actual_costs(
            params.table,
            &execution_feedback.execution,
            evidence,
            execution_feedback,
        );

        // Step 5: Blend estimates with actual observations.
        let weight = cfg.cost_feedback_weight;
        let delta_final = self.blend_cost(cost_params, delta_cost, delta_actual, weight);
        let vortex_final = self.blend_cost(cost_params, vortex_cost, vortex_actual, weight);

        // Step 6: Choose minimum cost.
        let (decision, min_cost) = argmin_path(delta_final, vortex_final, merge_cost_val);

        // Step 7: Compute confidence.
        let confidence = self.compute_confidence(
            params,
            sel,
            selectivity_tracker,
            execution_feedback
                .execution
                .measured_execution_count(params.table),
            delta_cost,
            vortex_cost,
            delta_actual.as_ref(),
            vortex_actual.as_ref(),
        );

        let reasoning = format!(
            "tier2: sel={:.4}, rows={}, delta={:.2}, vortex={:.2}, merge={:.2} -> {:?}",
            sel, rows, delta_final, vortex_final, merge_cost_val, decision
        );

        RoutingResult {
            decision,
            confidence,
            reasoning,
            estimated_cost: min_cost,
            tier: "tier2_cost",
        }
    }

    /// Get selectivity from history or fall back to the estimated value.
    fn get_selectivity<'a>(&self, params: &RouterParams<'a>, tracker: &SelectivityTracker) -> f64 {
        // Try to get per-column estimates and average them.
        let col_estimates: Vec<f64> = params
            .columns
            .iter()
            .filter_map(|col| {
                let est = tracker.estimate(params.table, col);
                let sample_count = tracker.sample_count(params.table, col);
                if est != 0.1 && sample_count >= MIN_SELECTIVITY_SAMPLES {
                    Some(est)
                } else {
                    None
                }
            })
            .collect();

        let result = if col_estimates.is_empty() {
            // No sufficiently-sampled history -- use the predicate-analyzed estimate from Tier 1.
            params.estimated_selectivity
        } else {
            // Blend historical estimates with the predicate estimate only after minimum sample depth.
            let hist_avg = col_estimates.iter().sum::<f64>() / col_estimates.len() as f64;
            (hist_avg + params.estimated_selectivity) / 2.0
        };

        // Defensive NaN guard: if either the input or the blend produced NaN,
        // fall back to the conservative default. NaN can only come from the
        // Tier 1 estimated_selectivity (this function and SlidingWindow::mean
        // never produce NaN by construction).
        if result.is_nan() || result <= 0.0 {
            0.1
        } else {
            result
        }
    }

    /// Get actual observed execution times from feedback history.
    fn get_actual_costs(
        &self,
        table: &str,
        execution: &ExecutionHistory,
        evidence: Option<&ExecutionEvidenceSnapshot>,
        feedback: &crate::query::routing::feedback::FeedbackState,
    ) -> (Option<MeasuredCostSample>, Option<MeasuredCostSample>) {
        let delta_time = execution.measured_cost_sample(table, RouteDecision::DeltaStoreOnly);
        let vortex_time = execution.measured_cost_sample(table, RouteDecision::VortexOnly);

        if let Some(ev) = evidence {
            let selectivity = if ev.table_row_count == 0 {
                None
            } else {
                Some((ev.total_segment_rows as f64 / ev.table_row_count as f64).clamp(0.0, 1.0))
            };
            let measured_exec_count = feedback.execution.measured_execution_count(table);
            tracing::trace!(
                table,
                query_columns = ?ev.query_columns,
                routed_segments = ev.routed_segment_ids.len(),
                executed_segments = ev.executed_segment_ids.len(),
                delta_segment_count = ev.delta_segment_count,
                has_zone_map_predicates = ev.has_zone_map_predicates,
                has_cross_column_or = ev.has_cross_column_or,
                observed_selectivity = ?selectivity,
                measured_exec_count,
                "tier2 using evidence snapshot"
            );
        }

        (delta_time, vortex_time)
    }

    fn blend_cost(
        &self,
        cost_params: &CostParams,
        estimated_cost: f64,
        observed: Option<MeasuredCostSample>,
        weight: f64,
    ) -> f64 {
        let Some(sample) = observed else {
            return estimated_cost;
        };
        if sample.sample_count < MIN_MEASURED_EXECUTION_SAMPLES
            || !sample.quality.allows_cost_blend()
        {
            return estimated_cost;
        }
        debug_assert!(
            sample.quality.is_measured_family(),
            "tier2 cost blending must only consume measured evidence families"
        );
        cost_params.blend(estimated_cost, sample.avg_elapsed_ms, weight)
    }

    /// Compute confidence score [0.0, 1.0].
    ///
    /// High confidence: many selectivity observations, clear cost ordering, consistent history.
    /// Low confidence: few observations, similar costs, conflicting history.
    #[allow(clippy::too_many_arguments)]
    fn compute_confidence(
        &self,
        params: &RouterParams<'_>,
        sel: f64,
        tracker: &SelectivityTracker,
        measured_execution_count: usize,
        delta_cost: f64,
        vortex_cost: f64,
        delta_actual: Option<&MeasuredCostSample>,
        vortex_actual: Option<&MeasuredCostSample>,
    ) -> f64 {
        let mut confidence = 0.25_f64;

        let n_obs = params
            .columns
            .iter()
            .filter(|col| tracker.sample_count(params.table, col) >= MIN_SELECTIVITY_SAMPLES)
            .count();
        if n_obs > 0 {
            confidence += (n_obs as f64 * 0.05).min(0.15);
        }

        if measured_execution_count >= MIN_MEASURED_EXECUTION_SAMPLES {
            confidence += 0.15;
        }

        let has_measured_delta = delta_actual.is_some_and(|sample| {
            sample.sample_count >= MIN_MEASURED_EXECUTION_SAMPLES
                && sample.quality.allows_cost_blend()
        });
        let has_measured_vortex = vortex_actual.is_some_and(|sample| {
            sample.sample_count >= MIN_MEASURED_EXECUTION_SAMPLES
                && sample.quality.allows_cost_blend()
        });
        let has_measured_history = has_measured_delta && has_measured_vortex;
        let cost_gap = (delta_cost - vortex_cost).abs() / (delta_cost + vortex_cost + 1e-6);
        if has_measured_history && cost_gap > 0.5 {
            confidence += 0.15;
        }

        if let (Some(d), Some(v)) = (delta_actual, vortex_actual) {
            if d.sample_count >= MIN_MEASURED_EXECUTION_SAMPLES
                && v.sample_count >= MIN_MEASURED_EXECUTION_SAMPLES
                && d.quality.allows_cost_blend()
                && v.quality.allows_cost_blend()
            {
                let actual_gap = (d.avg_elapsed_ms - v.avg_elapsed_ms).abs()
                    / (d.avg_elapsed_ms + v.avg_elapsed_ms + 1e-6);
                if actual_gap > 0.3 {
                    confidence += 0.15;
                }
            }
        }

        if !(0.01..=0.5).contains(&sel) && n_obs > 0 {
            confidence += 0.1;
        }

        confidence.clamp(0.0, 1.0)
    }
}

impl Default for StatisticalRouter {
    fn default() -> Self {
        Self::new()
    }
}
