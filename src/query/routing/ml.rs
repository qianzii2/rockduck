//! Tier 3: ML-based query router — veDB-HTAP style.
//!
//! Implements the ByteDance veDB-HTAP approach (VLDB 2025) for complex queries.
//!
//! ## Architecture
//!
//! Uses a three-head linear model for Delta vs Vortex vs Merge routing:
//!   Input(25) -> DeltaScore / VortexScore / MergeScore -> route decision
//!
//! The canonical production model is trained via `scripts/train_tree_cnn.py`
//! and exported as a little-endian `f32` binary weight file.
//!
//! Weight file format:
//!   [delta_w0, ..., delta_w24, delta_b,
//!    vortex_w0, ..., vortex_w24, vortex_b,
//!    merge_w0, ..., merge_w24, merge_b]
//!   = 3 * (25 + 1) = 78 f32 = 312 bytes.
//!
//! ## Loading weights
//!
//! If `ml_weight_path` is missing, malformed, or sized for an older format,
//! ML routing is disabled and the router falls back to Tier 2 decisions. Use
//! `scripts/train_tree_cnn.py` to generate a compatible weight file.

use crate::query::routing::config::{MlRoutingPromotionGate, RouterConfig};
use crate::query::routing::RouteDecision;

const FEATURE_DIM: usize = 25;
const OP_TYPE_COUNT: usize = 20;

#[derive(Debug, Clone)]
pub enum OpType {
    Scan = 0,
    PointGet = 1,
    RangeScan = 2,
    Filter = 3,
    Aggregate = 4,
    GroupBy = 5,
    Sort = 6,
    Join = 7,
    Limit = 8,
    Projection = 9,
    Union = 10,
    Distinct = 11,
    WindowFn = 12,
    Subquery = 13,
    Insert = 14,
    Update = 15,
    Delete = 16,
    SetOp = 17,
    Other = 18,
    Unknown = 19,
}

impl OpType {
    pub fn from_kind(kind: crate::query::routing::QueryKind) -> Self {
        match kind {
            crate::query::routing::QueryKind::PointGet => OpType::PointGet,
            crate::query::routing::QueryKind::FullScan => OpType::Scan,
            crate::query::routing::QueryKind::RangeScan => OpType::RangeScan,
            crate::query::routing::QueryKind::Aggregate => OpType::Aggregate,
        }
    }
}

fn encode_op_type(op: OpType) -> [f32; OP_TYPE_COUNT] {
    let mut v = [0.0_f32; OP_TYPE_COUNT];
    let idx = op as usize;
    if idx < OP_TYPE_COUNT {
        v[idx] = 1.0;
    }
    v
}

#[derive(Debug, Clone)]
pub struct QueryFeatureVector {
    pub(crate) data: [f32; FEATURE_DIM],
}

impl QueryFeatureVector {
    pub fn from_query(
        op_type: OpType,
        selectivity: f64,
        row_count: u64,
        num_filters: usize,
        num_columns: usize,
        delta_count: usize,
    ) -> Self {
        let mut data = [0.0_f32; FEATURE_DIM];
        let oh = encode_op_type(op_type);
        data[0..OP_TYPE_COUNT].copy_from_slice(&oh);
        data[OP_TYPE_COUNT] = selectivity.clamp(0.0, 1.0) as f32;
        // Row count: log10(rows+1), normalized to [0,1] assuming max 1B rows.
        data[OP_TYPE_COUNT + 1] = ((row_count as f64 + 1.0).log10() / 9.0).clamp(0.0, 1.0) as f32;
        // Number of filters: log scale, max 20.
        let max_filters = 20.0_f64;
        data[OP_TYPE_COUNT + 2] =
            ((num_filters as f64 + 1.0).log2() / max_filters.log2()).clamp(0.0, 1.0) as f32;
        // Number of columns: log scale, max 64.
        let max_cols = 64.0_f64;
        data[OP_TYPE_COUNT + 3] =
            ((num_columns as f64 + 1.0).log2() / max_cols.log2()).clamp(0.0, 1.0) as f32;
        // Delta count: log scale, max 1024 deltas.
        let max_deltas = 1024.0_f64;
        data[OP_TYPE_COUNT + 4] =
            ((delta_count as f64 + 1.0).log2() / max_deltas.log2()).clamp(0.0, 1.0) as f32;
        Self { data }
    }

    pub fn from_slice(values: &[f32]) -> Self {
        let mut data = [0.0_f32; FEATURE_DIM];
        let len = values.len().min(FEATURE_DIM);
        data[..len].copy_from_slice(&values[..len]);
        Self { data }
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }
}

#[derive(Debug, Clone)]
pub struct MlRoutingResult {
    pub delta_score: f32,
    pub vortex_score: f32,
    pub merge_score: f32,
    pub routing_path: RouteDecision,
    pub confidence: f32,
}

#[derive(Debug, Clone)]
pub struct MlLiveOverrideDecision {
    pub routing_path: RouteDecision,
    pub confidence: f64,
    pub disagreement_ratio: f64,
}

/// Twin-network linear model for HTAP routing.
///
/// Three independent linear heads (Delta, Vortex, Merge):
///   score = dot(weights, features) + bias
///
/// Weights are loaded from `assets/tree_cnn_weights.bin`.
/// Format: [delta_w0..24, delta_b, vortex_w0..24, vortex_b, merge_w0..24, merge_b]
/// = 3 * (25 + 1) = 78 f32s = 312 bytes.
pub struct TreeCnnRouter {
    /// Whether ML inference is enabled and weights loaded successfully.
    pub enabled: bool,
    /// Why ML is disabled when `enabled` is false.
    pub disabled_reason: Option<String>,
    /// Delta head: 25 weights + 1 bias.
    delta_weights: [f32; FEATURE_DIM],
    delta_bias: f32,
    /// Vortex head: 25 weights + 1 bias.
    vortex_weights: [f32; FEATURE_DIM],
    vortex_bias: f32,
    /// Merge head: 25 weights + 1 bias.
    merge_weights: [f32; FEATURE_DIM],
    merge_bias: f32,
}

impl TreeCnnRouter {
    /// Load weights from binary file.
    /// Format: [delta_w0..24, delta_b, vortex_w0..24, vortex_b, merge_w0..24, merge_b]
    /// = 3 * (25 + 1) = 78 f32s = 312 bytes.
    #[allow(clippy::type_complexity)]
    fn load_weights(
        path: &str,
    ) -> std::io::Result<(
        [f32; FEATURE_DIM],
        f32,
        [f32; FEATURE_DIM],
        f32,
        [f32; FEATURE_DIM],
        f32,
    )> {
        let data = std::fs::read(path)?;
        let exact_len = (FEATURE_DIM * 3 + 3) * 4;
        if data.len() != exact_len {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "expected exactly {} bytes / 78 little-endian f32s, got {} bytes",
                    exact_len,
                    data.len()
                ),
            ));
        }
        fn read_f32(data: &[u8], i: usize) -> std::io::Result<f32> {
            let value = f32::from_le_bytes([data[i], data[i + 1], data[i + 2], data[i + 3]]);
            if !value.is_finite() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("non-finite weight at byte offset {i}"),
                ));
            }
            Ok(value)
        }
        let mut dw = [0.0_f32; FEATURE_DIM];
        for (i, slot) in dw.iter_mut().enumerate() {
            *slot = read_f32(&data, i * 4)?;
        }
        let db = read_f32(&data, FEATURE_DIM * 4)?;
        let mut vw = [0.0_f32; FEATURE_DIM];
        for (i, slot) in vw.iter_mut().enumerate() {
            *slot = read_f32(&data, (FEATURE_DIM + 1 + i) * 4)?;
        }
        let vb = read_f32(&data, (FEATURE_DIM * 2 + 1) * 4)?;
        let mut mw = [0.0_f32; FEATURE_DIM];
        for (i, slot) in mw.iter_mut().enumerate() {
            *slot = read_f32(&data, (FEATURE_DIM * 2 + 2 + i) * 4)?;
        }
        let mb = read_f32(&data, (FEATURE_DIM * 3 + 2) * 4)?;
        Ok((dw, db, vw, vb, mw, mb))
    }

    pub fn new(config: &RouterConfig) -> std::io::Result<Self> {
        if !config.ml_enabled {
            return Ok(Self {
                enabled: false,
                disabled_reason: Some("ml routing disabled by config".to_string()),
                delta_weights: [0.0; FEATURE_DIM],
                delta_bias: 0.5,
                vortex_weights: [0.0; FEATURE_DIM],
                vortex_bias: 0.5,
                merge_weights: [0.0; FEATURE_DIM],
                merge_bias: 0.0,
            });
        }

        match Self::load_weights(&config.ml_weight_path) {
            Ok((dw, db, vw, vb, mw, mb)) => {
                tracing::info!(
                    "Loaded ML routing weights from {} ({} bytes)",
                    config.ml_weight_path,
                    (FEATURE_DIM * 3 + 3) * 4
                );
                Ok(Self {
                    enabled: true,
                    disabled_reason: None,
                    delta_weights: dw,
                    delta_bias: db,
                    vortex_weights: vw,
                    vortex_bias: vb,
                    merge_weights: mw,
                    merge_bias: mb,
                })
            }
            Err(err) => {
                let expected_bytes = (FEATURE_DIM * 3 + 3) * 4;
                tracing::warn!(
                    "ML weight file '{}' invalid for current format (expected exactly {} bytes / 78 little-endian f32s from scripts/train_tree_cnn.py): {}; ML routing disabled.",
                    config.ml_weight_path,
                    expected_bytes,
                    err
                );
                Ok(Self {
                    enabled: false,
                    disabled_reason: Some(format!(
                        "failed to load ML routing weights from {}: {}",
                        config.ml_weight_path, err
                    )),
                    delta_weights: [0.0; FEATURE_DIM],
                    delta_bias: 0.5,
                    vortex_weights: [0.0; FEATURE_DIM],
                    vortex_bias: 0.5,
                    merge_weights: [0.0; FEATURE_DIM],
                    merge_bias: 0.0,
                })
            }
        }
    }

    /// Run inference: compute Delta, Vortex, and Merge scores from the feature vector.
    ///
    /// Three-way routing:
    /// - merge_score highest: route to Merge (Delta overlay on Vortex)
    /// - delta_score highest: route to DeltaStoreOnly
    /// - vortex_score highest: route to VortexOnly
    ///
    /// When disabled, returns equal Delta/Vortex scores and Merge=0 (defer to Tier 2).
    pub fn predict(&self, features: &QueryFeatureVector) -> MlRoutingResult {
        let (delta_score, vortex_score, merge_score) = if self.enabled {
            let feat = features.as_slice();
            let ds = feat
                .iter()
                .zip(self.delta_weights.iter())
                .map(|(f, w)| f * w)
                .sum::<f32>()
                + self.delta_bias;
            let vs = feat
                .iter()
                .zip(self.vortex_weights.iter())
                .map(|(f, w)| f * w)
                .sum::<f32>()
                + self.vortex_bias;
            let ms = feat
                .iter()
                .zip(self.merge_weights.iter())
                .map(|(f, w)| f * w)
                .sum::<f32>()
                + self.merge_bias;
            (ds, vs, ms)
        } else {
            (0.5, 0.5, 0.0)
        };

        let sum = delta_score + vortex_score + merge_score;
        let confidence = if sum > 0.0 {
            let max_score = delta_score.max(vortex_score).max(merge_score);
            let min_score = delta_score.min(vortex_score).min(merge_score);
            (max_score - min_score) / sum
        } else {
            0.0
        };

        // Merge is chosen only when it is strictly best AND confidence is meaningful.
        // Conservative threshold: merge must beat the second-best by at least 10%.
        let routing_path =
            if merge_score > delta_score && merge_score > vortex_score && confidence > 0.1 {
                RouteDecision::Merge
            } else if delta_score < vortex_score {
                RouteDecision::DeltaStoreOnly
            } else {
                RouteDecision::VortexOnly
            };

        MlRoutingResult {
            delta_score,
            vortex_score,
            merge_score,
            routing_path,
            confidence,
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    pub fn live_override_decision(
        &self,
        gate: &MlRoutingPromotionGate,
        feedback: &crate::query::routing::feedback::FeedbackState,
        ml_result: &MlRoutingResult,
    ) -> Option<MlLiveOverrideDecision> {
        if !self.enabled || !gate.live_authority_enabled {
            return None;
        }
        let measured = feedback.measured_ml_training_sample_count();
        if measured < gate.min_measured_samples {
            return None;
        }
        let agreement = feedback.ml_shadow_agreement_ratio()?;
        let disagreement = 1.0 - agreement;
        if agreement < gate.min_shadow_agreement_ratio || disagreement > gate.max_disagreement_ratio
        {
            return None;
        }
        Some(MlLiveOverrideDecision {
            routing_path: ml_result.routing_path,
            confidence: ml_result.confidence as f64,
            disagreement_ratio: disagreement,
        })
    }
}

#[cfg(test)]
mod ml_governance_tests {
    use super::*;

    #[test]
    fn ml_mode_never_allows_live_authority() {
        assert!(!crate::query::routing::MlRoutingMode::Disabled.allows_live_authority());
        assert!(!crate::query::routing::MlRoutingMode::AdvisoryShadow.allows_live_authority());
    }

    #[test]
    fn missing_weights_keep_router_disabled_even_when_ml_is_enabled() {
        let config = RouterConfig::builder()
            .ml_enabled(true)
            .ml_weight_path("assets/definitely-missing-tree-cnn-weights.bin".to_string())
            .build();

        let router =
            TreeCnnRouter::new(&config).expect("router construction should degrade gracefully");
        assert!(
            !router.enabled,
            "missing weights must not produce an authoritative ML router"
        );
    }
}
