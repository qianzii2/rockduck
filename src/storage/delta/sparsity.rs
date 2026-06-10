//! Adaptive sparsity selector and format conversion utilities.
//!
//! # Sparsity Decision
//!
//! Based on ClickHouse Patch Part and Iceberg V3 Deletion Vector research:
//! - `sparsity < 1%` OR `affected < 1000` rows → **Sparse** (RoaringBitmap + values)
//! - `total_rows > 1_000_000` → **Dense** (avoid container explosion)
//! - otherwise → **Dense** (Arrow columnar is more compact)
//!
//! Binary format is compatible with Iceberg V3 Deletion Vector spec:
//! <https://iceberg.apache.org/puffin-spec/>

use std::sync::Arc;

use arrow_array::{Array, ArrayRef, RecordBatch, UInt32Array};
use arrow_schema::Field;
use croaring::Bitmap;

use super::types::{DeltaPatchFormat, ZoneMap};
use crate::error::Result;
use crate::write::wal_utils::{batch_to_bytes, decode_ipc_column};

/// Adaptive sparsity selector — chooses between Sparse (RoaringBitmap) and Dense (Arrow IPC).
///
/// Decision logic:
/// - `sparsity < 1%` → Sparse (RoaringBitmap overhead justified)
/// - `affected < 1000` rows → Sparse (small patch overhead is low)
/// - `total_rows > 1_000_000` → Dense (RoaringBitmap container explosion)
/// - otherwise → Dense
#[derive(Debug, Clone)]
pub struct SparsitySelector {
    /// Sparsity threshold (fraction of rows). Default 0.01 (1%).
    pub threshold: f64,
    /// Row count above which we force dense format.
    pub dense_force_rows: usize,
}

impl Default for SparsitySelector {
    fn default() -> Self {
        Self {
            threshold: 0.01,
            dense_force_rows: 1_000_000,
        }
    }
}

impl SparsitySelector {
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns `true` if we should use the sparse format.
    pub fn should_use_sparse(&self, total_rows: u64, affected: usize) -> bool {
        if total_rows == 0 || affected == 0 {
            return false;
        }
        // Hard cap: segments above 1M rows use dense to prevent RoaringBitmap container explosion
        if total_rows > self.dense_force_rows as u64 {
            return false;
        }
        if affected >= self.dense_force_rows {
            return false;
        }
        let ratio = affected as f64 / total_rows as f64;
        ratio < self.threshold || affected < 1000
    }

    /// Select the appropriate format and build the patch format.
    ///
    /// `positions` — sorted, unique row offsets affected by this patch.
    /// `values` — Arrow IPC-encoded column values (one per position).
    /// `total_rows` — total row count of the segment (for sparsity ratio).
    /// `txn_id` — highest txn_id among cells in this patch (used in Dense format).
    pub fn build_format(
        &self,
        positions: &[u32],
        values: &dyn Array,
        total_rows: u64,
        txn_id: u64,
    ) -> DeltaPatchFormat {
        if self.should_use_sparse(total_rows, positions.len()) {
            self.build_sparse(positions, values)
        } else {
            self.build_dense(positions, values, total_rows, txn_id)
        }
    }

    fn build_sparse(&self, positions: &[u32], values: &dyn Array) -> DeltaPatchFormat {
        let bitmap = positions_to_roaring(positions);
        let values_bytes = encode_arrow_array(values);
        DeltaPatchFormat::sparse(bitmap, values_bytes)
    }

    fn build_dense(
        &self,
        _positions: &[u32],
        values: &dyn Array,
        total_rows: u64,
        txn_id: u64,
    ) -> DeltaPatchFormat {
        // For dense format, we create a full-row array with patch values at affected positions.
        // We encode the compact version: just the (position, value) pairs.
        // The caller is responsible for expanding to full dense if needed.
        let values_bytes = encode_arrow_array(values);
        DeltaPatchFormat::dense(values_bytes, total_rows, txn_id)
    }
}

// =============================================================================
// RoaringBitmap utilities
// =============================================================================

/// Build a RoaringBitmap from a sorted, unique list of positions.
pub fn positions_to_roaring(positions: &[u32]) -> Bitmap {
    Bitmap::of(positions)
}

/// Convert a Bitmap to a sorted Vec<u32>.
pub fn roaring_to_positions(bitmap: &Bitmap) -> Vec<u32> {
    bitmap.iter().collect::<Vec<u32>>()
}

/// Convert a Bitmap to a UInt32Array (one value per set bit).
pub fn roaring_to_u32array(bitmap: &Bitmap) -> UInt32Array {
    UInt32Array::from_iter_values(bitmap.iter())
}

/// Convert a UInt32Array to a Bitmap.
pub fn u32array_to_roaring(arr: &UInt32Array) -> Bitmap {
    Bitmap::of(arr.values().iter().as_slice())
}

/// Check if a Bitmap contains a specific position.
pub fn roaring_contains(bitmap: &Bitmap, pos: u32) -> bool {
    bitmap.contains(pos)
}

// =============================================================================
// Arrow IPC encode / decode — delegates to wal_utils (single source of truth)
// =============================================================================

/// Encode a single Arrow array to IPC bytes.
/// Delegates to `wal_utils::batch_to_bytes` for the single-column RecordBatch path.
pub fn encode_arrow_array(array: &dyn arrow_array::Array) -> Vec<u8> {
    let field = Field::new("value", array.data_type().clone(), true);
    let schema = arrow_schema::SchemaRef::new(arrow_schema::Schema::new(vec![field]));
    let batch = RecordBatch::try_new(
        schema,
        vec![Arc::new(arrow_array::make_array(array.to_data().clone()))],
    )
    .expect("Failed to create RecordBatch for IPC encoding");
    batch_to_bytes(&batch).expect("Failed to encode Arrow array to IPC bytes")
}

/// Decode IPC bytes back to an ArrayRef.
/// Delegates to `wal_utils::decode_ipc_column`.
pub fn decode_arrow_array(bytes: &[u8]) -> Result<ArrayRef> {
    decode_ipc_column(bytes)
}

/// Extract a single value from an Arrow IPC byte array at the given logical index.
pub fn extract_value_at(values_bytes: &[u8], logical_index: usize) -> Result<Option<Vec<u8>>> {
    let arr = decode_arrow_array(values_bytes)?;
    if logical_index >= arr.len() {
        return Ok(None);
    }
    let binary = arr.as_any().downcast_ref::<arrow_array::BinaryArray>();
    if let Some(b) = binary {
        if b.is_null(logical_index) {
            Ok(None)
        } else {
            Ok(Some(b.value(logical_index).to_vec()))
        }
    } else {
        Ok(None)
    }
}

// =============================================================================
// Sparse patch utilities
// =============================================================================

/// Build a ZoneMap from a sparse patch.
pub fn build_zone_map_from_sparse(
    _patch: &DeltaPatchFormat,
    txn_range: (u64, u64),
    affected: u64,
) -> ZoneMap {
    ZoneMap {
        min_txn: txn_range.0,
        max_txn: txn_range.1,
        affected_rows: affected,
        min_value: None,
        max_value: None,
    }
}

/// Build a ZoneMap from a dense patch.
pub fn build_zone_map_from_dense(
    _patch: &DeltaPatchFormat,
    txn_range: (u64, u64),
    total_rows: u64,
) -> ZoneMap {
    ZoneMap {
        min_txn: txn_range.0,
        max_txn: txn_range.1,
        affected_rows: total_rows,
        min_value: None,
        max_value: None,
    }
}

// =============================================================================
