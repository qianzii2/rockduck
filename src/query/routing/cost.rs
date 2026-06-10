//! Cost estimation engine for the Tier 2 statistical router.
//!
//! Estimates relative execution cost for DeltaStore, Vortex, and Merge paths.
//! Costs are relative units (not wall-clock time) -- only ordering matters.
//!
//! ## Cost Model
//!
//! Total cost = CPU_cost + IO_cost
//!
//! **DeltaStore**: rows * cpu_cost_per_row + rows * avg_row_bytes * io_cost_per_byte
//!     (row-oriented: no compression benefit, must read full rows for PK lookups)
//!
//! **Vortex**: rows * cpu_cost_per_row * col_decompression_factor
//!             + rows * avg_row_bytes * compression_ratio * io_cost_per_byte
//!     (columnar: only reads needed columns; compression reduces I/O)
//!
//! **Merge**: delta_count * lookup_cost + vortex_rows * overhead_factor
//!     (delta PK-tree lookups + vortex scan + overlay merge)
//!
//! See ACM (Adaptive Cost Model, arXiv 2024) for the online-tuning approach
//! used to adapt these parameters at runtime.

use serde::{Deserialize, Serialize};

use crate::query::routing::config::RouterConfig;
use crate::query::routing::{RouteDecision, TableStats};

/// Cost parameters -- can be updated online via `CostParams::update()`.
#[derive(Debug, Clone)]
pub struct CostParams {
    /// CPU cost per row (relative units).
    pub cpu_cost_per_row: f64,
    /// I/O cost per byte (relative units).
    pub io_cost_per_byte: f64,
    /// Fixed cost of one PK-tree lookup in DeltaStore.
    pub delta_lookup_cost: f64,
    /// Merge overhead multiplier over pure Vortex.
    pub merge_overhead_factor: f64,
    /// Columnar compression ratio (compressed / uncompressed bytes).
    /// Updated from BlockStats by Tier 2.
    pub columnar_compression_ratio: f64,

    /// Online regression state for adaptive CPU cost tuning.
    /// Tracks the recent error between estimated and actual execution times.
    error_history: Vec<f64>,
    /// Maximum error history size before pruning.
    max_history: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostParamsCheckpoint {
    pub cpu_cost_per_row: f64,
    pub io_cost_per_byte: f64,
    pub delta_lookup_cost: f64,
    pub merge_overhead_factor: f64,
    pub columnar_compression_ratio: f64,
    pub error_history: Vec<f64>,
    pub max_history: usize,
}

impl CostParams {
    /// Build cost params from static config. Call `update()` at runtime to adapt.
    pub fn from_config(cfg: &RouterConfig) -> Self {
        Self {
            cpu_cost_per_row: cfg.cpu_cost_per_row,
            io_cost_per_byte: cfg.io_cost_per_byte,
            delta_lookup_cost: cfg.delta_lookup_cost,
            merge_overhead_factor: cfg.merge_overhead_factor,
            columnar_compression_ratio: cfg.columnar_compression_ratio,
            error_history: Vec::new(),
            max_history: 500,
        }
    }

    /// Update cost parameters using observed execution feedback.
    ///
    /// Implements a lightweight online least-squares adjustment inspired by ACM.
    /// The adjustment is proportional to the error and inversely proportional to row count
    /// (large scans have more stable per-row costs).
    pub fn update(&mut self, estimated: f64, actual: f64, rows: u64) {
        if rows == 0 || estimated <= 0.0 {
            return;
        }
        let error = actual - estimated;
        let error_pct = error / estimated;

        self.error_history.push(error_pct);
        if self.error_history.len() > self.max_history {
            self.error_history.drain(0..self.error_history.len() / 2);
        }

        // Gradient descent style update: adjust cpu_cost_per_row by the average error fraction.
        // Positive avg_error = underestimation (actual > estimated) -> INCREASE cpu_cost.
        // Negative avg_error = overestimation (actual < estimated) -> DECREASE cpu_cost.
        let avg_error = self.error_history.iter().sum::<f64>() / self.error_history.len() as f64;
        let learning_rate = 0.001 / (1.0 + rows as f64 * 1e-6).sqrt();
        let adjustment = avg_error * learning_rate;

        self.cpu_cost_per_row = (self.cpu_cost_per_row * (1.0 + adjustment)).clamp(0.001, 100.0);

        // Adapt compression ratio based on observed I/O patterns.
        // If actual I/O was higher than expected, compression is worse than modeled.
        let io_weight = 0.0001;
        if error > 0.0 {
            self.columnar_compression_ratio =
                (self.columnar_compression_ratio + io_weight).min(1.0);
        } else {
            self.columnar_compression_ratio =
                (self.columnar_compression_ratio - io_weight).max(0.1);
        }

        tracing::debug!(
            "CostParams updated: cpu={:.4}, io={:.5}, comp_ratio={:.3}, avg_error={:.4}",
            self.cpu_cost_per_row,
            self.io_cost_per_byte,
            self.columnar_compression_ratio,
            avg_error
        );
    }

    /// Reset the error history for online calibration.
    /// Useful after schema changes or major operational shifts that invalidate historical cost data.
    pub fn reset_history(&mut self) {
        self.error_history.clear();
    }

    /// Blend an estimated cost with an observed real cost.
    /// `weight` is how much to trust the observation (0.0 = pure estimate, 1.0 = pure observation).
    pub fn blend(&self, estimated: f64, observed: f64, weight: f64) -> f64 {
        estimated * (1.0 - weight) + observed * weight
    }

    pub fn to_checkpoint(&self) -> CostParamsCheckpoint {
        CostParamsCheckpoint {
            cpu_cost_per_row: self.cpu_cost_per_row,
            io_cost_per_byte: self.io_cost_per_byte,
            delta_lookup_cost: self.delta_lookup_cost,
            merge_overhead_factor: self.merge_overhead_factor,
            columnar_compression_ratio: self.columnar_compression_ratio,
            error_history: self.error_history.clone(),
            max_history: self.max_history,
        }
    }

    pub fn apply_checkpoint(&mut self, checkpoint: &CostParamsCheckpoint) {
        self.cpu_cost_per_row = checkpoint.cpu_cost_per_row.clamp(0.001, 100.0);
        self.io_cost_per_byte = checkpoint.io_cost_per_byte;
        self.delta_lookup_cost = checkpoint.delta_lookup_cost;
        self.merge_overhead_factor = checkpoint.merge_overhead_factor;
        self.columnar_compression_ratio = checkpoint.columnar_compression_ratio.clamp(0.1, 1.0);
        self.error_history = checkpoint.error_history.clone();
        self.max_history = checkpoint.max_history.max(1);
    }
}

/// Scan engine type -- determines which cost formula to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScanEngine {
    /// Row-store delta overlay (DeltaStore).
    Delta,
    /// Columnar storage (Vortex).
    Vortex,
    /// Delta + Vortex merge.
    Merge,
}

/// Estimate the CPU cost of scanning `rows` rows.
fn cpu_cost(rows: u64, cpu_per_row: f64) -> f64 {
    rows as f64 * cpu_per_row
}

/// Estimate the I/O cost of scanning `rows` rows with given bytes-per-row and I/O cost.
fn io_cost(rows: u64, bytes_per_row: f64, io_per_byte: f64) -> f64 {
    rows as f64 * bytes_per_row * io_per_byte
}

/// Estimate the cost of a DeltaStore scan.
///
/// Delta is row-oriented: must read full rows for PK lookups, no column pruning.
/// Cost = CPU(rows) + IO(rows)
pub fn delta_scan_cost(rows: u64, stats: &TableStats, params: &CostParams) -> f64 {
    let avg_bytes = stats.avg_row_bytes();
    cpu_cost(rows, params.cpu_cost_per_row) + io_cost(rows, avg_bytes, params.io_cost_per_byte)
}

/// Estimate the cost of a Vortex (columnar) scan.
///
/// Columnar is more efficient: only reads needed columns, benefits from compression.
/// Cost = CPU(rows * col_factor) + IO(rows * bytes * compression_ratio)
pub fn vortex_scan_cost(rows: u64, stats: &TableStats, params: &CostParams) -> f64 {
    // Columnar decompression is typically 2-5x more CPU-intensive per row than row-store.
    let col_cpu_factor = 1.5;
    let avg_bytes = stats.avg_row_bytes();

    let cpu = cpu_cost(rows, params.cpu_cost_per_row * col_cpu_factor);
    let io = io_cost(
        rows,
        avg_bytes * params.columnar_compression_ratio,
        params.io_cost_per_byte,
    );
    cpu + io
}

/// Estimate the cost of a Delta + Vortex Merge scan.
///
/// Cost = delta_lookup_cost * delta_count + vortex_scan * merge_overhead_factor
pub fn merge_cost(
    delta_count: usize,
    vortex_rows: u64,
    stats: &TableStats,
    params: &CostParams,
) -> f64 {
    let delta_lookup = delta_count as f64 * params.delta_lookup_cost;
    let vortex = vortex_scan_cost(vortex_rows, stats, params);
    delta_lookup + vortex * params.merge_overhead_factor
}

/// Choose the minimum-cost path from three cost estimates.
pub fn argmin_path(delta_cost: f64, vortex_cost: f64, merge_cost: f64) -> (RouteDecision, f64) {
    if delta_cost <= vortex_cost && delta_cost <= merge_cost {
        (RouteDecision::DeltaStoreOnly, delta_cost)
    } else if vortex_cost <= merge_cost {
        (RouteDecision::VortexOnly, vortex_cost)
    } else {
        (RouteDecision::Merge, merge_cost)
    }
}
