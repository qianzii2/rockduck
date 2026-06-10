//! Core types for the delta storage module.
//!
//! Key types:
//! - [`DeltaCell`] — unified point-update unit
//! - [`DeltaPatch`] — persistent patch unit with adaptive format
//! - [`DeltaPatchFormat`] — either Sparse (RoaringBitmap) or Dense (Arrow IPC bytes)
//! - [`ZoneMap`] — txn-range + row-count stats for patch-level ZoneMap pruning
//!
//! ## File Format
//!
//! Each `.patch` file starts with a 6-byte magic header `RDELTA00`
//! followed by the format payload:
//! - `0x00` — Sparse format (RoaringBitmap positions + Arrow IPC values)
//! - `0x01` — Dense format (Arrow IPC values for all rows)
//!
//! Files written before this header was added (D2 fix) lack the magic prefix
//! and start directly with the format byte. `from_bytes` detects this by
//! checking whether the file starts with the magic bytes.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use croaring::Bitmap;
use serde::{Deserialize, Serialize};

/// File magic bytes for `.patch` files, used to detect partial writes.
///
/// Bytes: `R D E L T A 00` (6 bytes header + 1 byte version).
/// Old files written before the D2 fix lack this prefix; `from_bytes` handles
/// backward compatibility.
const PATCH_MAGIC: &[u8; 6] = b"RDELTA";

/// Unified delta cell — the atomic unit of incremental update.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeltaCell {
    /// Segment ID — top-level partitioning key.
    pub seg_id: String,
    /// Row offset within the segment.
    pub row_offset: u64,
    /// Column name.
    pub column: String,
    /// Transaction ID (globally monotonically increasing).
    pub txn_id: u64,
    /// Before-image. `None` for inserts.
    pub before: Option<Arc<Vec<u8>>>,
    /// After-image. `None` for deletes.
    pub after: Option<Arc<Vec<u8>>>,
    /// Whether this delta has been committed.
    pub committed: bool,
    /// Wall-clock timestamp (for time travel).
    pub ts: i64,
}

impl DeltaCell {
    /// Returns true if this delta is semantically committed for a snapshot using
    /// authoritative MVCC commit timestamps. Absent commit history means invisible.
    pub fn is_visible_at_commit(
        &self,
        snapshot_txn: u64,
        commit_ts_by_txn: &HashMap<u64, u64>,
        active_txns: &HashSet<u64>,
    ) -> bool {
        if self.txn_id > snapshot_txn {
            return false;
        }
        if active_txns.contains(&self.txn_id) {
            return false;
        }
        commit_ts_by_txn
            .get(&self.txn_id)
            .copied()
            .map(|commit_ts| commit_ts <= snapshot_txn)
            .unwrap_or(false)
    }

    /// Legacy committed flag is retained only as a transitional materialization field.
    /// Authoritative visibility must come from commit history + active txn state.
    pub fn legacy_committed_flag(&self) -> bool {
        self.committed
    }

    /// Returns true if this is an insert.
    pub fn is_insert(&self) -> bool {
        self.before.is_none() && self.after.is_some()
    }

    /// Returns true if this is a delete.
    pub fn is_delete(&self) -> bool {
        self.before.is_some() && self.after.is_none()
    }

    /// Returns true if this is an update.
    pub fn is_update(&self) -> bool {
        self.before.is_some() && self.after.is_some()
    }
}

/// A persistent delta patch — the unit stored in L2 and L3.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeltaPatch {
    /// Segment ID.
    pub seg_id: String,
    /// Column name.
    pub column: String,
    /// Monotonically increasing patch version.
    pub patch_id: u64,
    /// Transaction range covered by this patch.
    pub txn_range: (u64, u64),
    /// The patch payload — either sparse or dense.
    pub format: DeltaPatchFormat,
    /// ZoneMap stats for this patch.
    pub zone_map: ZoneMap,
}

/// ZoneMap statistics for a delta patch — used for ZoneMap-level pruning.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ZoneMap {
    /// Minimum transaction ID in this patch.
    pub min_txn: u64,
    /// Maximum transaction ID in this patch.
    pub max_txn: u64,
    /// Number of rows affected by this patch.
    pub affected_rows: u64,
    /// Min value (optional, for column-level ZoneMap).
    pub min_value: Option<Vec<u8>>,
    /// Max value (optional, for column-level ZoneMap).
    pub max_value: Option<Vec<u8>>,
}

/// The format of a delta patch.
///
/// We use an adaptive format that selects between Sparse (RoaringBitmap + values)
/// and Dense (Arrow IPC bytes) based on the sparsity ratio. This mirrors
/// the approach taken by ClickHouse Patch Parts and Iceberg V3 Deletion Vectors.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeltaPatchFormat {
    /// Sparse: only affected rows stored, referenced by RoaringBitmap.
    /// Best when < 1% of rows are affected.
    Sparse {
        /// Bitmap of affected row offsets (little-endian RoaringBitmap).
        positions: Arc<Vec<u8>>,
        /// Binary values in Arrow IPC format, one per set bit in positions.
        values: Arc<Vec<u8>>,
        /// Cached cardinality of the bitmap — avoids repeated deserialization on each
        /// `affected_count()` call. Computed once at construction time and stored.
        affected_count: u64,
    },
    /// Dense: every row has a value (including unchanged rows as nulls).
    /// Best when >= 1% of rows are affected.
    Dense {
        /// Arrow IPC encoded array — one value per row in the segment.
        values: Arc<Vec<u8>>,
        /// Total number of rows in the segment — stored so affected_count() can return it.
        total_rows: u64,
        /// Transaction ID of the patch — the highest txn_id among all cells in this patch.
        /// Used for visibility filtering in k_way_merge (txn_id > snapshot_txn → skip).
        txn_id: u64,
    },
}

impl DeltaPatchFormat {
    /// Build a sparse format from positions and values.
    ///
    /// The `affected_count` is computed once here and cached, so callers that call
    /// `affected_count()` multiple times pay the deserialization cost only once.
    pub fn sparse(positions: croaring::Bitmap, values: Vec<u8>) -> Self {
        let affected_count = positions.cardinality();
        Self::Sparse {
            positions: Arc::new(positions.serialize::<croaring::Portable>()),
            values: Arc::new(values),
            affected_count,
        }
    }

    /// Build a dense format from Arrow IPC bytes, total row count, and patch txn_id.
    ///
    /// `txn_id` is the highest txn_id among all cells in the patch — used for
    /// visibility filtering (k_way_merge skips patches with txn_id > snapshot_txn).
    pub fn dense(values: Vec<u8>, total_rows: u64, txn_id: u64) -> Self {
        Self::Dense {
            values: Arc::new(values),
            total_rows,
            txn_id,
        }
    }

    /// Returns the number of affected rows.
    ///
    /// For `Sparse` patches this returns the cached cardinality — no re-deserialization.
    /// For `Dense` patches it returns the stored `total_rows`.
    pub fn affected_count(&self) -> u64 {
        match self {
            Self::Sparse { affected_count, .. } => *affected_count,
            Self::Dense { total_rows, .. } => *total_rows,
        }
    }

    /// Returns true if this is a sparse patch.
    pub fn is_sparse(&self) -> bool {
        matches!(self, Self::Sparse { .. })
    }

    /// Returns true if this is a dense patch.
    pub fn is_dense(&self) -> bool {
        matches!(self, Self::Dense { .. })
    }

    /// Serialize to bytes. Format:
    /// - u8 flag: 0x00 = sparse, 0x01 = dense
    /// - u32 payload size
    /// - payload bytes
    ///
    /// Dense payload: u64 txn_id + u64 total_rows + u32 val_len + Arrow IPC bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let payload = match self {
            Self::Sparse {
                positions,
                values,
                affected_count: _,
            } => {
                let mut buf = Vec::with_capacity(9 + positions.len() + values.len());
                buf.push(0x00u8);
                let pos_len = positions.len() as u32;
                buf.extend_from_slice(&pos_len.to_le_bytes());
                buf.extend_from_slice(positions);
                let val_len = values.len() as u32;
                buf.extend_from_slice(&val_len.to_le_bytes());
                buf.extend_from_slice(values);
                buf
            }
            Self::Dense {
                values,
                total_rows,
                txn_id,
            } => {
                let mut buf = Vec::with_capacity(25 + values.len());
                buf.push(0x01u8);
                buf.extend_from_slice(&txn_id.to_le_bytes());
                buf.extend_from_slice(&total_rows.to_le_bytes());
                let val_len = values.len() as u32;
                buf.extend_from_slice(&val_len.to_le_bytes());
                buf.extend_from_slice(values);
                buf
            }
        };
        let mut buf = Vec::with_capacity(7 + payload.len());
        buf.extend_from_slice(PATCH_MAGIC);
        buf.push(0x00);
        buf.extend_from_slice(&payload);
        buf
    }

    /// Deserialize from bytes.
    ///
    /// Handles backward compatibility with old `.patch` files written before the D2 fix
    /// (those lack the magic header and start directly with the format byte).
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.is_empty() {
            return None;
        }
        let (offset, format_start) = if bytes.len() >= 7 && &bytes[..6] == PATCH_MAGIC {
            (7, 7)
        } else {
            (0, 0)
        };
        match bytes[offset] {
            0x00 => {
                if bytes.len() < format_start + 9 {
                    return None;
                }
                let pos_len = u32::from_le_bytes(
                    bytes[format_start + 1..format_start + 5].try_into().ok()?,
                ) as usize;
                if bytes.len() < format_start + 5 + pos_len + 4 {
                    return None;
                }
                let positions = bytes[format_start + 5..format_start + 5 + pos_len].to_vec();
                let val_len_offset = format_start + 5 + pos_len;
                let val_len = u32::from_le_bytes(
                    bytes[val_len_offset..val_len_offset + 4].try_into().ok()?,
                ) as usize;
                let values_offset = val_len_offset + 4;
                if bytes.len() < values_offset + val_len {
                    return None;
                }
                let values = bytes[values_offset..values_offset + val_len].to_vec();
                let affected_count =
                    Bitmap::deserialize::<croaring::Portable>(&positions).cardinality();
                Some(Self::Sparse {
                    positions: Arc::new(positions),
                    values: Arc::new(values),
                    affected_count,
                })
            }
            0x01 => {
                if bytes.len() < format_start + 21 {
                    return None;
                }
                let txn_id = u64::from_le_bytes(
                    bytes[format_start + 1..format_start + 9].try_into().ok()?,
                );
                let total_rows = u64::from_le_bytes(
                    bytes[format_start + 9..format_start + 17].try_into().ok()?,
                );
                let val_len = u32::from_le_bytes(
                    bytes[format_start + 17..format_start + 21].try_into().ok()?,
                ) as usize;
                if bytes.len() < format_start + 21 + val_len {
                    return None;
                }
                let values = bytes[format_start + 21..format_start + 21 + val_len].to_vec();
                Some(Self::Dense {
                    values: Arc::new(values),
                    total_rows,
                    txn_id,
                })
            }
            _ => None,
        }
    }
}

/// Delta checkpoint state for durability / recovery.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeltaCheckpointState {
    pub committed_txn: u64,
    pub l1_entry_count: usize,
    pub l1_size_bytes: u64,
    pub l2_patch_counts: u64,
    pub l3_patch_counts: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn visible_cell(txn_id: u64) -> DeltaCell {
        DeltaCell {
            seg_id: "seg".to_string(),
            row_offset: 7,
            column: "value".to_string(),
            txn_id,
            before: None,
            after: Some(Arc::new(vec![txn_id as u8])),
            committed: false,
            ts: 0,
        }
    }

    #[test]
    fn visibility_requires_commit_history_not_legacy_flag() {
        let cell = visible_cell(42);
        let active_txns = HashSet::new();

        assert!(!cell.is_visible_at_commit(100, &HashMap::new(), &active_txns));

        let mut commit_ts_by_txn = HashMap::new();
        commit_ts_by_txn.insert(42, 40);
        assert!(cell.is_visible_at_commit(100, &commit_ts_by_txn, &active_txns));
    }

    #[test]
    fn active_txn_is_never_visible_even_if_legacy_flag_true() {
        let mut cell = visible_cell(42);
        cell.committed = true;

        let mut commit_ts_by_txn = HashMap::new();
        commit_ts_by_txn.insert(42, 40);
        let active_txns = HashSet::from([42]);

        assert!(!cell.is_visible_at_commit(100, &commit_ts_by_txn, &active_txns));
        assert!(cell.legacy_committed_flag());
    }
}
