//! Query plan encoder -- converts parsed queries into Tree-CNN input features.
//!
//! This bridges the gap between `filter_expr.rs` (or future SQL parser) and
//! `ml.rs` (Tree-CNN inference).
//!
//! ## Input
//!
//! - `RouterParams`: query type, columns, selectivity, row count.
//!
//! ## Output
//!
//! - `QueryFeatureVector`: fixed-dimension f32 vector fed into `TreeCnnRouter`.
//!
//! ## Feature Dimensions
//!
//! Total: 25 dimensions
//! - [0-19]  OpType one-hot encoding (20 types)
//! - [20]    Selectivity (0.0-1.0)
//! - [21]    Row count log10 (normalized 0-1)
//! - [22]    Number of filter predicates (log scale)
//! - [23]    Number of projected columns (log scale)
//! - [24]    Delta count (log scale)
//!
//! See veDB-HTAP (VLDB 2025) Section 4.3.2 for the Tree-CNN encoding approach.

use crate::query::routing::ml::{OpType, QueryFeatureVector};
use crate::query::routing::RouterParams;

/// Encodes query features for ML-based routing.
pub struct PlanEncoder;

impl PlanEncoder {
    pub fn new() -> Self {
        Self
    }

    /// Encode a query's routing parameters into a fixed-dimension feature vector.
    ///
    /// This is the primary integration point with the query parser.
    /// Once a full SQL parser is integrated, extract the plan tree and call this
    /// per-node, then aggregate node vectors into a single query vector.
    pub fn encode(&self, params: &RouterParams) -> QueryFeatureVector {
        let op_type = OpType::from_kind(params.kind);
        let num_filters = Self::count_filters_from_selectivity(params.estimated_selectivity);

        QueryFeatureVector::from_query(
            op_type,
            params.estimated_selectivity,
            params.stats.row_count,
            num_filters,
            params.columns.len(),
            params.delta_count,
        )
    }

    /// Infer the number of filter predicates from the estimated selectivity.
    /// This is a heuristic: low selectivity -> many filters needed to achieve it.
    fn count_filters_from_selectivity(selectivity: f64) -> usize {
        if selectivity >= 0.9 {
            0
        } else if selectivity >= 0.5 {
            1
        } else if selectivity >= 0.1 {
            2
        } else if selectivity >= 0.01 {
            3
        } else if selectivity >= 0.001 {
            4
        } else {
            5
        }
    }

    /// Encode multiple plan nodes into a single vector.
    ///
    /// Used when the query has a complex plan tree (e.g., JOINs, subqueries).
    /// Strategy: concatenate node vectors and apply mean pooling.
    pub fn encode_aggregate<'a, I>(&self, nodes: I) -> QueryFeatureVector
    where
        I: Iterator<Item = &'a QueryFeatureVector>,
    {
        let mut sum = [0.0_f32; 25];
        let mut count = 0usize;

        for node in nodes {
            for (i, &v) in node.as_slice().iter().enumerate() {
                sum[i] += v;
            }
            count += 1;
        }

        if count > 0 {
            for v in &mut sum {
                *v /= count as f32;
            }
        }

        QueryFeatureVector { data: sum }
    }
}

impl Default for PlanEncoder {
    fn default() -> Self {
        Self::new()
    }
}
