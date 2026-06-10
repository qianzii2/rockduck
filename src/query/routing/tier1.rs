//! Tier 1: Rule-based fast path router.
//!
//! Makes routing decisions using simple, hardcoded rules that require zero statistics.
//! Target latency: < 0.1ms (no KV reads, no cost computation).
//!
//! ## Coverage
//!
//! Tier 1 handles these cases with 100% confidence:
//! - PK point-get -> DeltaStoreOnly (one PK lookup is always faster than a columnar scan)
//! - No pending writes -> VortexOnly (no need for delta overlay)
//! - Very low selectivity + few deltas -> DeltaStoreOnly
//! - High selectivity -> VortexOnly (columnar compression helps on large scans)
//!
//! If none of the rules match, returns `None` and defers to Tier 2.

use crate::query::routing::config::RouterConfig;
use crate::query::routing::{QueryKind, RouteDecision, RouterParams};

/// Rule-based router -- no statistics needed, deterministic.
pub struct RuleRouter;

impl RuleRouter {
    pub fn new() -> Self {
        Self
    }

    /// Decide using Tier 1 rules.
    /// Returns `Some(decision)` if a rule applies, `None` if uncertain.
    pub fn decide(&self, params: &RouterParams, cfg: &RouterConfig) -> Option<RouteDecision> {
        // Rule 1: Point-get is always DeltaStoreOnly.
        if params.kind == QueryKind::PointGet {
            return Some(RouteDecision::DeltaStoreOnly);
        }

        // Rule 2: No pending writes AND no delta files on disk -> VortexOnly.
        // Both conditions must be met: no pending writes in memstore AND no deltas
        // on disk. If there are pending writes, they may not yet be in Vortex
        // (only flushed during compaction), so we must defer to Tier 2/3.
        if !params.has_pending_writes && params.delta_count == 0 {
            return Some(RouteDecision::VortexOnly);
        }

        // Rule 3: Very low selectivity -> DeltaStoreOnly.
        // Small result sets benefit from row-store PK lookups.
        if params.estimated_selectivity < cfg.point_selectivity_thresh {
            return Some(RouteDecision::DeltaStoreOnly);
        }

        // Rule 4: High selectivity -> VortexOnly.
        // Large scans benefit from columnar compression and vectorized execution.
        if params.estimated_selectivity > cfg.full_scan_selectivity_thresh {
            return Some(RouteDecision::VortexOnly);
        }

        // Rule 5: Aggregate with high selectivity -> VortexOnly.
        // Aggregations on large result sets are much faster with columnar.
        if params.kind == QueryKind::Aggregate
            && params.estimated_selectivity > cfg.aggregate_selectivity_thresh
        {
            return Some(RouteDecision::VortexOnly);
        }

        // Rule 6: Many deltas -> Merge (delta overload).
        // When there are too many deltas, a pure delta scan becomes a full table scan anyway.
        if params.delta_count > cfg.delta_overload_thresh {
            return Some(RouteDecision::Merge);
        }

        // Cannot decide confidently -- defer to Tier 2.
        None
    }
}

impl Default for RuleRouter {
    fn default() -> Self {
        Self::new()
    }
}
