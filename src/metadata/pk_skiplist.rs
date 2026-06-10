//! Primary key skiplist with double-buffering support
//!
//! Key format: pk_idx:<seg_id>:<granule_id>:<pk_bytes>
//!
//! This design enables efficient prefix scans by segment (compaction, segment-level queries).
//! The seg_id and granule_id are embedded in the key so that
//! `pk_skiplist_key(seg_id, 0)` works as a prefix scan for all PKs in a segment.

use crate::error::{Result, RockDuckError};
use crate::metadata::granule_id::GranuleId;
use crate::metadata::kv_engine::{KVEngine, CF_PK_IDX};
use std::sync::Arc;

/// Double-buffered primary key index entry
#[derive(Debug, Clone)]
pub struct SkiplistEntry {
    /// Primary key bytes — extracted from the KV key
    pub pk: Vec<u8>,
    pub seg_id: String,
    pub granule_id: GranuleId,
    pub row_offset: u32,
}

/// Build a primary key index key: pk_idx:<seg_id>:<granule_id>:<pk_bytes>
///
/// This format enables `list_skiplist_entries` to scan all PKs for a segment
/// using `pk_skiplist_key(seg_id, 0)` as a prefix.
pub fn pk_index_key(seg_id: &str, granule_id: GranuleId, pk: &[u8]) -> Vec<u8> {
    let mut key = Vec::with_capacity(8 + seg_id.len() + 1 + 4 + 1 + pk.len());
    key.extend_from_slice(b"pk_idx:");
    key.extend_from_slice(seg_id.as_bytes());
    key.push(b':');
    key.extend_from_slice(&granule_id.get().to_le_bytes());
    key.push(b':');
    key.extend_from_slice(pk);
    key
}

/// Build a reverse lookup key: pk_lookup:<pk_bytes>
/// This enables O(1) lookup by PK alone, without needing seg_id/granule_id.
/// The value stored is: seg_id:granule_id:row_offset (same as pk_index value).
pub fn pk_lookup_key(pk: &[u8]) -> Vec<u8> {
    let mut key = Vec::with_capacity(10 + pk.len());
    key.extend_from_slice(b"pk_lookup:");
    key.extend_from_slice(pk);
    key
}

/// Insert a primary key entry into the PK index (in-memory + KV store).
/// Writes two KV entries atomically in a single write_batch:
///   pk_idx:<seg_id>:<granule_id>:<pk>  -> (seg_id, granule_id, row_offset)
///   pk_lookup:<pk>                    -> (seg_id, granule_id, row_offset, idx_key)
pub fn put_pk_index_double(
    kv: &Arc<dyn KVEngine>,
    _table: &str,
    pk: &[u8],
    seg_id: &str,
    granule_id: GranuleId,
    row_offset: u32,
) -> Result<()> {
    let idx_key = pk_index_key(seg_id, granule_id, pk);
    let lookup_key = pk_lookup_key(pk);
    let value = crate::codec::encode(&(seg_id.to_string(), granule_id.get(), row_offset))
        .map_err(|e| RockDuckError::Codec(e.to_string()))?;
    let lookup_value = crate::codec::encode(&(
        seg_id.to_string(),
        granule_id.get(),
        row_offset,
        idx_key.clone(),
    ))
    .map_err(|e| RockDuckError::Codec(e.to_string()))?;

    // Atomic write: both entries succeed or both fail
    kv.write_batch(
        CF_PK_IDX,
        &[
            crate::metadata::kv_engine::KVOp::Put {
                key: idx_key,
                value,
            },
            crate::metadata::kv_engine::KVOp::Put {
                key: lookup_key,
                value: lookup_value,
            },
        ],
    )?;

    Ok(())
}

/// Delete a primary key entry from the PK index.
pub fn delete_pk_index_double(
    kv: &Arc<dyn KVEngine>,
    _table: &str,
    pk: &[u8],
    seg_id: &str,
    granule_id: GranuleId,
) -> Result<()> {
    // Delete primary index entry
    let idx_key = pk_index_key(seg_id, granule_id, pk);
    kv.delete(CF_PK_IDX, &idx_key)?;
    // Delete reverse lookup entry
    let lookup_key = pk_lookup_key(pk);
    kv.delete(CF_PK_IDX, &lookup_key)?;
    Ok(())
}

/// Delete a primary key entry from the PK index (RocksDB WriteBatch variant — kept for compatibility)
pub fn delete_pk_index_double_into_batch(
    kv: &Arc<dyn KVEngine>,
    _table: &str,
    pk: &[u8],
    seg_id: &str,
    granule_id: GranuleId,
) -> Result<()> {
    // Delete primary index entry
    let idx_key = pk_index_key(seg_id, granule_id, pk);
    kv.delete(CF_PK_IDX, &idx_key)?;
    // Delete reverse lookup entry
    let lookup_key = pk_lookup_key(pk);
    kv.delete(CF_PK_IDX, &lookup_key)?;
    Ok(())
}

/// Get a primary key entry from the PK index by (seg_id, granule_id, pk).
pub fn get_pk_index(
    kv: &Arc<dyn KVEngine>,
    _table: &str,
    pk: &[u8],
    seg_id: &str,
    granule_id: GranuleId,
) -> Result<Option<(String, GranuleId, u32)>> {
    let key = pk_index_key(seg_id, granule_id, pk);
    match kv.get(CF_PK_IDX, &key)? {
        Some(value) => {
            let entry: (String, u32, u32) =
                crate::codec::decode(&value).map_err(|e| RockDuckError::Codec(e.to_string()))?;
            Ok(Some((entry.0, GranuleId::new(entry.1), entry.2)))
        }
        None => Ok(None),
    }
}

/// Get a primary key entry by PK alone (uses pk_lookup reverse index).
/// This is the primary lookup path for delete/update operations.
///
/// If the returned seg_id has been aliased (segment was compacted), this function
/// transparently follows the alias and returns the entry from the new seg_id.
pub fn get_pk_index_by_pk(
    kv: &Arc<dyn KVEngine>,
    _table: &str,
    pk: &[u8],
) -> Result<Option<(String, GranuleId, u32)>> {
    let lookup_key = pk_lookup_key(pk);
    match kv.get(CF_PK_IDX, &lookup_key)? {
        Some(value) => {
            let (seg_id, granule_raw, row_offset, idx_key): (String, u32, u32, Vec<u8>) =
                crate::codec::decode(&value).map_err(|e| RockDuckError::Codec(e.to_string()))?;

            let (final_seg_id, is_redirected) =
                crate::metadata::seg_alias::resolve_seg_id(kv.as_ref(), &seg_id)?;

            if !is_redirected {
                return Ok(Some((seg_id, GranuleId::new(granule_raw), row_offset)));
            }

            tracing::debug!(
                "pk_skiplist: pk {:x?} redirected {} -> {}",
                pk,
                seg_id,
                final_seg_id
            );

            match kv.get(CF_PK_IDX, &idx_key)? {
                Some(v) => {
                    let new_entry: (String, u32, u32) = crate::codec::decode(&v)
                        .map_err(|e| RockDuckError::Codec(e.to_string()))?;
                    Ok(Some((
                        final_seg_id,
                        GranuleId::new(new_entry.1),
                        new_entry.2,
                    )))
                }
                None => Ok(None),
            }
        }
        None => Ok(None),
    }
}

/// Resolve a seg_id by following any segment alias.
///
/// Returns `(final_seg_id, granule_id, row_offset)` where final_seg_id is the current
/// (possibly redirected) segment ID.
pub fn resolve_pk_entry_seg_id(
    kv: &dyn KVEngine,
    seg_id: &str,
    granule_id: GranuleId,
    row_offset: u32,
) -> Result<(String, GranuleId, u32)> {
    let (final_seg_id, _is_redirected) = crate::metadata::seg_alias::resolve_seg_id(kv, seg_id)?;
    Ok((final_seg_id, granule_id, row_offset))
}

/// Build a skiplist key from a segment ID and granule ID.
/// Used as a prefix for KV prefix iteration to list all PKs in a segment.
///
/// Format: <seg_id>:<granule_id_le_bytes>
/// When combined with the pk_idx: prefix in pk_index_key:
///   pk_idx:<seg_id>:<granule_id>:<pk_bytes>
/// starts with pk_idx:<seg_id>: so scanning with prefix `pk_idx:<seg_id>:` finds all entries.
pub fn pk_skiplist_key(seg_id: &str, granule_id: GranuleId) -> Vec<u8> {
    let mut key = Vec::with_capacity(8 + seg_id.len() + 1 + 4);
    key.extend_from_slice(b"pk_idx:");
    key.extend_from_slice(seg_id.as_bytes());
    key.push(b':');
    // granule_id as prefix to narrow scan to a specific granule (0 = all granules)
    key.extend_from_slice(&granule_id.get().to_le_bytes());
    key
}

/// Parse a pk_idx key back into its components: (seg_id, granule_id, pk_bytes).
/// Returns None if the key doesn't start with the pk_idx prefix.
fn find_byte(haystack: &[u8], byte: u8) -> Option<usize> {
    haystack.iter().position(|&b| b == byte)
}

/// Parse a pk_idx key back into its components: (seg_id, granule_id, pk_bytes).
/// Returns None if the key doesn't start with the pk_idx prefix.
pub fn parse_pk_index_key(key: &[u8]) -> Option<(String, GranuleId, Vec<u8>)> {
    if !key.starts_with(b"pk_idx:") {
        return None;
    }
    let rest = &key[7..]; // skip "pk_idx:"
                          // Find first colon = end of seg_id
    let first_colon = find_byte(rest, b':')?;
    let seg_id = String::from_utf8_lossy(&rest[..first_colon]).to_string();
    let after_seg = &rest[first_colon + 1..];
    // Next 4 bytes = granule_id (little-endian u32)
    if after_seg.len() < 4 {
        return None;
    }
    let granule_id = GranuleId::from_le_bytes(after_seg[..4].try_into().ok()?);
    let pk = after_seg[4..].to_vec();
    Some((seg_id, granule_id, pk))
}

/// List all skiplist entries for a segment using KV prefix iteration.
///
/// The prefix `pk_idx:<seg_id>:` scans all PKs in that segment across all granules.
/// For each entry, the pk is extracted from the key bytes (previously discarded).
pub fn list_skiplist_entries(kv: &Arc<dyn KVEngine>, seg_id: &str) -> Result<Vec<SkiplistEntry>> {
    // Prefix: pk_idx:<seg_id>: (all granule_ids)
    let prefix = pk_skiplist_key(seg_id, GranuleId::zero());
    let mut entries = Vec::new();
    let mut iter = kv.prefix_iter(CF_PK_IDX, &prefix)?;

    while iter.next() {
        let key = iter.key();
        let value = iter.value().to_vec();

        // Parse seg_id, granule_id, pk from the key
        let Some((_, _, pk)) = parse_pk_index_key(key) else {
            tracing::warn!("Malformed pk_idx key for seg {}, skipping", seg_id);
            continue;
        };

        let entry: (String, u32, u32, Vec<u8>) =
            crate::codec::decode(&value).map_err(|e| RockDuckError::Codec(e.to_string()))?;
        entries.push(SkiplistEntry {
            pk,
            seg_id: entry.0,
            granule_id: GranuleId::new(entry.1),
            row_offset: entry.2,
        });
    }

    Ok(entries)
}
