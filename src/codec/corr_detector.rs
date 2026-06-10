//! CORRA — Correlation-Aware Column Compression.
//!
//! Detects and encodes correlated column pairs:
//! - **Peer DIFF**: Highly correlated numeric columns.
//! - **Subaltern**: Hierarchical one-to-many relationships.

use arrow_array::types::{Float32Type, Float64Type, Int32Type, Int64Type, UInt32Type, UInt64Type};
use arrow_array::{Array, PrimitiveArray};
use std::collections::HashMap;

/// Result of CORRA detection for a column pair.
#[derive(Debug, Clone)]
pub enum CorrResult {
    PeerDiff {
        corr: f64,
        est_ratio: f64,
    },
    Subaltern {
        num_parents: usize,
        avg_children: f64,
    },
    None,
}

/// Detect correlation between two columns and return encoding recommendation.
pub fn detect_correlation(col_a: &dyn Array, col_b: &dyn Array) -> Option<CorrResult> {
    let corr = pearson_correlation(col_a, col_b)?;
    if corr > 0.9 {
        let est_ratio = estimate_diff_ratio(col_a, col_b);
        return Some(CorrResult::PeerDiff { corr, est_ratio });
    }
    if let Some(sub) = detect_subaltern(col_a, col_b) {
        return Some(sub);
    }
    None
}

/// Pearson correlation coefficient between two numeric columns.
/// PR 7-B fix (CODEC-2): supports Int32, Int64, UInt32, UInt64, Float32, Float64.
pub fn pearson_correlation(a: &dyn Array, b: &dyn Array) -> Option<f64> {
    if a.len() != b.len() || a.is_empty() {
        return None;
    }

    let n = a.len();
    let sample_size = n.min(10000);
    let step = if n <= sample_size { 1 } else { n / sample_size };

    let mut sum_a = 0.0;
    let mut sum_b = 0.0;
    let mut sum_ab = 0.0;
    let mut sum_a2 = 0.0;
    let mut sum_b2 = 0.0;
    let mut count = 0i64;

    // Try Int64 columns
    if let (Some(da), Some(db)) = (
        a.as_any().downcast_ref::<PrimitiveArray<Int64Type>>(),
        b.as_any().downcast_ref::<PrimitiveArray<Int64Type>>(),
    ) {
        for i in (0..n).step_by(step).take(sample_size) {
            if da.is_valid(i) && db.is_valid(i) {
                let va = da.value(i) as f64;
                let vb = db.value(i) as f64;
                sum_a += va;
                sum_b += vb;
                sum_ab += va * vb;
                sum_a2 += va * va;
                sum_b2 += vb * vb;
                count += 1;
            }
        }
    }

    // Try Float64 columns
    if count == 0 {
        if let (Some(da), Some(db)) = (
            a.as_any().downcast_ref::<PrimitiveArray<Float64Type>>(),
            b.as_any().downcast_ref::<PrimitiveArray<Float64Type>>(),
        ) {
            for i in (0..n).step_by(step).take(sample_size) {
                if da.is_valid(i) && db.is_valid(i) {
                    let va = da.value(i);
                    let vb = db.value(i);
                    sum_a += va;
                    sum_b += vb;
                    sum_ab += va * vb;
                    sum_a2 += va * va;
                    sum_b2 += vb * vb;
                    count += 1;
                }
            }
        }
    }

    // Try Int32 columns
    if count == 0 {
        if let (Some(da), Some(db)) = (
            a.as_any().downcast_ref::<PrimitiveArray<Int32Type>>(),
            b.as_any().downcast_ref::<PrimitiveArray<Int32Type>>(),
        ) {
            for i in (0..n).step_by(step).take(sample_size) {
                if da.is_valid(i) && db.is_valid(i) {
                    let va = da.value(i) as f64;
                    let vb = db.value(i) as f64;
                    sum_a += va;
                    sum_b += vb;
                    sum_ab += va * vb;
                    sum_a2 += va * va;
                    sum_b2 += vb * vb;
                    count += 1;
                }
            }
        }
    }

    // Try UInt32 columns
    if count == 0 {
        if let (Some(da), Some(db)) = (
            a.as_any().downcast_ref::<PrimitiveArray<UInt32Type>>(),
            b.as_any().downcast_ref::<PrimitiveArray<UInt32Type>>(),
        ) {
            for i in (0..n).step_by(step).take(sample_size) {
                if da.is_valid(i) && db.is_valid(i) {
                    let va = da.value(i) as f64;
                    let vb = db.value(i) as f64;
                    sum_a += va;
                    sum_b += vb;
                    sum_ab += va * vb;
                    sum_a2 += va * va;
                    sum_b2 += vb * vb;
                    count += 1;
                }
            }
        }
    }

    // Try UInt64 columns
    if count == 0 {
        if let (Some(da), Some(db)) = (
            a.as_any().downcast_ref::<PrimitiveArray<UInt64Type>>(),
            b.as_any().downcast_ref::<PrimitiveArray<UInt64Type>>(),
        ) {
            for i in (0..n).step_by(step).take(sample_size) {
                if da.is_valid(i) && db.is_valid(i) {
                    let va = da.value(i) as f64;
                    let vb = db.value(i) as f64;
                    sum_a += va;
                    sum_b += vb;
                    sum_ab += va * vb;
                    sum_a2 += va * va;
                    sum_b2 += vb * vb;
                    count += 1;
                }
            }
        }
    }

    // Try Float32 columns
    if count == 0 {
        if let (Some(da), Some(db)) = (
            a.as_any().downcast_ref::<PrimitiveArray<Float32Type>>(),
            b.as_any().downcast_ref::<PrimitiveArray<Float32Type>>(),
        ) {
            for i in (0..n).step_by(step).take(sample_size) {
                if da.is_valid(i) && db.is_valid(i) {
                    let va = da.value(i) as f64;
                    let vb = db.value(i) as f64;
                    sum_a += va;
                    sum_b += vb;
                    sum_ab += va * vb;
                    sum_a2 += va * va;
                    sum_b2 += vb * vb;
                    count += 1;
                }
            }
        }
    }

    if count < 2 {
        return None;
    }

    let n_f = count as f64;
    let numerator = n_f * sum_ab - sum_a * sum_b;
    let denominator = ((n_f * sum_a2 - sum_a * sum_a) * (n_f * sum_b2 - sum_b * sum_b)).sqrt();

    if denominator == 0.0 {
        // All x values are identical OR all y values are identical (zero variance).
        // In this degenerate case, there is no meaningful correlation — by convention we return
        // 1.0 to indicate "perfect ordering" when x is constant (y can be anything).
        // This is a design decision: alternative would be 0.0 (no correlation) or NaN.
        // Returning 1.0 biases toward over-correlation but is conservative for encoding decisions.
        return Some(1.0);
    }

    let corr = numerator / denominator;
    Some(corr.clamp(-1.0, 1.0))
}

/// Detect Subaltern relationship.
pub fn detect_subaltern(parent: &dyn Array, child: &dyn Array) -> Option<CorrResult> {
    if parent.len() != child.len() || parent.is_empty() {
        return None;
    }

    let n = parent.len();
    let step = if n <= 10000 { 1 } else { n / 10000 };

    let mut parent_to_children: HashMap<i64, Vec<i64>> = HashMap::new();

    let mut success = false;
    if let (Some(dp), Some(dc)) = (
        parent.as_any().downcast_ref::<PrimitiveArray<Int64Type>>(),
        child.as_any().downcast_ref::<PrimitiveArray<Int64Type>>(),
    ) {
        success = true;
        for i in (0..n).step_by(step) {
            if dp.is_valid(i) && dc.is_valid(i) {
                let pv = dp.value(i);
                let cv = dc.value(i);
                parent_to_children.entry(pv).or_default().push(cv);
            }
        }
    }

    if !success || parent_to_children.is_empty() {
        return None;
    }

    let num_parents = parent_to_children.len();
    let total_children: usize = parent_to_children.values().map(|v| v.len()).sum();
    let avg_children = total_children as f64 / num_parents as f64;

    if avg_children > 1.2 {
        Some(CorrResult::Subaltern {
            num_parents,
            avg_children,
        })
    } else {
        None
    }
}

fn estimate_diff_ratio(a: &dyn Array, b: &dyn Array) -> f64 {
    let mut sum_orig = 0.0f64;
    let mut sum_diff = 0.0f64;
    let mut count = 0i32;

    // PR 7-B fix (CODEC-2): support Int32, Int64, UInt32, UInt64, Float32, Float64.
    // Try Int64 first (most common in DB systems)
    if let (Some(da), Some(db)) = (
        a.as_any().downcast_ref::<PrimitiveArray<Int64Type>>(),
        b.as_any().downcast_ref::<PrimitiveArray<Int64Type>>(),
    ) {
        let n = a.len().min(1000);
        let step = if a.len() <= n { 1 } else { a.len() / n };
        for i in (0..a.len()).step_by(step).take(n) {
            if da.is_valid(i) && db.is_valid(i) {
                let va = da.value(i) as f64;
                let vb = db.value(i) as f64;
                sum_orig += va.abs() + vb.abs();
                sum_diff += (va - vb).abs();
                count += 1;
            }
        }
    }

    // Try Float64
    if count == 0 {
        if let (Some(da), Some(db)) = (
            a.as_any().downcast_ref::<PrimitiveArray<Float64Type>>(),
            b.as_any().downcast_ref::<PrimitiveArray<Float64Type>>(),
        ) {
            let n = a.len().min(1000);
            let step = if a.len() <= n { 1 } else { a.len() / n };
            for i in (0..a.len()).step_by(step).take(n) {
                if da.is_valid(i) && db.is_valid(i) {
                    let va = da.value(i);
                    let vb = db.value(i);
                    sum_orig += va.abs() + vb.abs();
                    sum_diff += (va - vb).abs();
                    count += 1;
                }
            }
        }
    }

    // Try Int32
    if count == 0 {
        if let (Some(da), Some(db)) = (
            a.as_any().downcast_ref::<PrimitiveArray<Int32Type>>(),
            b.as_any().downcast_ref::<PrimitiveArray<Int32Type>>(),
        ) {
            let n = a.len().min(1000);
            let step = if a.len() <= n { 1 } else { a.len() / n };
            for i in (0..a.len()).step_by(step).take(n) {
                if da.is_valid(i) && db.is_valid(i) {
                    let va = da.value(i) as f64;
                    let vb = db.value(i) as f64;
                    sum_orig += va.abs() + vb.abs();
                    sum_diff += (va - vb).abs();
                    count += 1;
                }
            }
        }
    }

    // Try UInt32
    if count == 0 {
        if let (Some(da), Some(db)) = (
            a.as_any().downcast_ref::<PrimitiveArray<UInt32Type>>(),
            b.as_any().downcast_ref::<PrimitiveArray<UInt32Type>>(),
        ) {
            let n = a.len().min(1000);
            let step = if a.len() <= n { 1 } else { a.len() / n };
            for i in (0..a.len()).step_by(step).take(n) {
                if da.is_valid(i) && db.is_valid(i) {
                    let va = da.value(i) as f64;
                    let vb = db.value(i) as f64;
                    sum_orig += va.abs() + vb.abs();
                    sum_diff += (va - vb).abs();
                    count += 1;
                }
            }
        }
    }

    // Try UInt64
    if count == 0 {
        if let (Some(da), Some(db)) = (
            a.as_any().downcast_ref::<PrimitiveArray<UInt64Type>>(),
            b.as_any().downcast_ref::<PrimitiveArray<UInt64Type>>(),
        ) {
            let n = a.len().min(1000);
            let step = if a.len() <= n { 1 } else { a.len() / n };
            for i in (0..a.len()).step_by(step).take(n) {
                if da.is_valid(i) && db.is_valid(i) {
                    let va = da.value(i) as f64;
                    let vb = db.value(i) as f64;
                    sum_orig += va.abs() + vb.abs();
                    sum_diff += (va - vb).abs();
                    count += 1;
                }
            }
        }
    }

    // Try Float32
    if count == 0 {
        if let (Some(da), Some(db)) = (
            a.as_any().downcast_ref::<PrimitiveArray<Float32Type>>(),
            b.as_any().downcast_ref::<PrimitiveArray<Float32Type>>(),
        ) {
            let n = a.len().min(1000);
            let step = if a.len() <= n { 1 } else { a.len() / n };
            for i in (0..a.len()).step_by(step).take(n) {
                if da.is_valid(i) && db.is_valid(i) {
                    let va = da.value(i) as f64;
                    let vb = db.value(i) as f64;
                    sum_orig += va.abs() + vb.abs();
                    sum_diff += (va - vb).abs();
                    count += 1;
                }
            }
        }
    }

    if count == 0 {
        // Return NaN when no valid pairs found (all NaN or empty).
        // This signals "no correlation detectable" to the caller.
        return f64::NAN;
    }

    // Avoid division by zero when sum_diff is 0 (identical values in both columns).
    if sum_diff == 0.0 {
        // When both columns are identical, correlation is "perfect" for DIFF encoding.
        return 10.0; // Clamped maximum to indicate high DIFF benefit.
    }

    (sum_orig / sum_diff).clamp(1.0, 10.0)
}

/// Encode a column as DIFF from a reference column (i64).
pub fn peer_diff_encode_i64(data: &[i64], reference: &[i64]) -> Vec<i64> {
    data.iter()
        .zip(reference.iter())
        .map(|(d, r)| d - r)
        .collect()
}

/// Decode a DIFF column back to original values (i64).
pub fn peer_diff_decode_i64(diff: &[i64], reference: &[i64]) -> Vec<i64> {
    diff.iter()
        .zip(reference.iter())
        .map(|(d, r)| d + r)
        .collect()
}

/// Encode a column as DIFF from a reference column (f64).
pub fn peer_diff_encode_f64(data: &[f64], reference: &[f64]) -> Vec<f64> {
    data.iter()
        .zip(reference.iter())
        .map(|(d, r)| d - r)
        .collect()
}

/// Decode a DIFF column back to original values (f64).
pub fn peer_diff_decode_f64(diff: &[f64], reference: &[f64]) -> Vec<f64> {
    diff.iter()
        .zip(reference.iter())
        .map(|(d, r)| d + r)
        .collect()
}

/// Build an offset table for Subaltern encoding.
pub fn build_subaltern_offset_table(child: &[i64], parent: &[i64]) -> (HashMap<i64, i64>, i64) {
    let mut parent_to_children: HashMap<i64, Vec<i64>> = HashMap::new();
    for (p, c) in parent.iter().zip(child.iter()) {
        parent_to_children.entry(*p).or_default().push(*c);
    }

    let mut global_min = i64::MAX;
    for children in parent_to_children.values() {
        if let Some(&min_child) = children.iter().min() {
            if min_child < global_min {
                global_min = min_child;
            }
        }
    }
    if global_min == i64::MAX {
        global_min = 0;
    }

    let offset_table: HashMap<i64, i64> = parent_to_children
        .iter()
        .map(|(&p, children)| {
            (
                p,
                children
                    .iter()
                    .min()
                    .copied()
                    .unwrap_or(0)
                    .saturating_sub(global_min),
            )
        })
        .collect();

    (offset_table, global_min)
}

/// Encode using Subaltern.
pub fn subaltern_encode(
    child: &[i64],
    parent: &[i64],
    offset_table: &HashMap<i64, i64>,
    global_min: i64,
) -> Vec<i64> {
    parent
        .iter()
        .zip(child.iter())
        .map(|(&p, &c)| {
            let offset = offset_table.get(&p).copied().unwrap_or(0);
            c.saturating_sub(offset + global_min)
        })
        .collect()
}

/// Decode using Subaltern.
pub fn subaltern_decode(
    diff: &[i64],
    parent: &[i64],
    offset_table: &HashMap<i64, i64>,
    global_min: i64,
) -> Vec<i64> {
    parent
        .iter()
        .zip(diff.iter())
        .map(|(&p, &d)| {
            let offset = offset_table.get(&p).copied().unwrap_or(0);
            d + offset + global_min
        })
        .collect()
}
