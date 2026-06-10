//! Generic metadata storage using the KVEngine trait
//!
//! Provides all metadata operations (segment meta, table stats, MVCC, etc.)
//! via the generic `KVEngine` interface, supporting both RocksDB and mace-kv backends.
//!
//! Column families / buckets:
//! - pk_idx: Primary key index
//! - seg_meta: Segment metadata
//! - stat: Statistics and counters
//! - zone: Zone maps for fast filtering
//! - layer: Layer/L0 tracking
//! - bf: Bloom filter index
//! - sys: System metadata
//! - mvcc: MVCC transaction state
//! - iceberg: Iceberg table metadata

use crate::db::TxnId;
use crate::error::{Result, RockDuckError};
use crate::metadata::kv_engine::KVEngine;
use crate::metadata::kv_engine::{CF_MVCC, CF_SEG_META, CF_STAT};
use crate::segment::meta::SegmentMeta;
use std::collections::HashMap;
use std::sync::Arc;

/// Segment alias column family — for redirecting PK lookups after compaction.
/// When a segment is compacted, its old seg_id is aliased to the new seg_id.
/// PK lookups transparently follow the alias chain.
pub const CF_SEG_ALIAS: &str = "seg_alias";

/// Key prefix for segment alias records: seg_alias:<old_seg_id> -> <new_seg_id>
const KEY_SEG_ALIAS: &[u8] = b"seg_alias:";

/// Key prefix constants
const KEY_COMMITTED_TXN: &[u8] = b"committed_txn";
const KEY_COMMITTED_TXN_HISTORY: &[u8] = b"committed:";
const KEY_SEGMENT_META: &[u8] = b"seg_meta:";
const KEY_TABLE_STATS: &[u8] = b"table_stats:";
const KEY_ACTIVE_TXNS: &[u8] = b"active_txns:";
const KEY_SEG_ROW_COUNT: &[u8] = b"seg_row_count:";

/// Build a key for atomic row count storage: seg_row_count:<seg_id>
pub fn segment_row_count_key(seg_id: &str) -> Vec<u8> {
    let mut key = Vec::with_capacity(KEY_SEG_ROW_COUNT.len() + seg_id.len());
    key.extend_from_slice(KEY_SEG_ROW_COUNT);
    key.extend_from_slice(seg_id.as_bytes());
    key
}

/// Table statistics stored in the stat bucket
#[derive(
    Debug,
    Clone,
    Default,
    serde::Serialize,
    serde::Deserialize,
    rkyv::Archive,
    rkyv::Serialize,
    rkyv::Deserialize,
)]
pub struct TableStats {
    pub table_name: String,
    pub row_count: u64,
    pub segment_count: u64,
    pub total_size_bytes: u64,
    pub last_updated_txn: u64,
}

/// Get or create table statistics
pub fn get_or_create_table_stats(kv: &Arc<dyn KVEngine>, table: &str) -> Result<TableStats> {
    let key = table_stats_key(table);
    match kv.get(CF_STAT, &key)? {
        Some(value) => {
            crate::codec::decode(&value).map_err(|e| RockDuckError::Codec(e.to_string()))
        }
        None => Ok(TableStats {
            table_name: table.to_string(),
            ..Default::default()
        }),
    }
}

/// Persist table statistics
pub fn put_table_stats(kv: &Arc<dyn KVEngine>, stats: &TableStats) -> Result<()> {
    let key = table_stats_key(&stats.table_name);
    let value = crate::codec::encode(stats).map_err(|e| RockDuckError::Codec(e.to_string()))?;
    kv.put(CF_STAT, &key, &value)
}

/// Persist committed transaction ID (for crash recovery)
pub fn put_committed_txn(kv: &Arc<dyn KVEngine>, txn_id: TxnId) -> Result<()> {
    let value = crate::codec::encode(&txn_id).map_err(|e| RockDuckError::Codec(e.to_string()))?;
    kv.put(CF_MVCC, KEY_COMMITTED_TXN, &value)
}

/// Get the last committed transaction ID
pub fn get_committed_txn(kv: &Arc<dyn KVEngine>) -> Result<TxnId> {
    match kv.get(CF_MVCC, KEY_COMMITTED_TXN)? {
        Some(value) => {
            crate::codec::decode(&value).map_err(|e| RockDuckError::Codec(e.to_string()))
        }
        None => Ok(0),
    }
}

/// Add active transaction record
pub fn add_active_txn(kv: &Arc<dyn KVEngine>, txn_id: TxnId, begin_ts: u64) -> Result<()> {
    let key = active_txn_key(txn_id);
    let value = crate::codec::encode(&(txn_id, begin_ts))
        .map_err(|e| RockDuckError::Codec(e.to_string()))?;
    kv.put(CF_MVCC, &key, &value)
}

/// Remove active transaction record
pub fn remove_active_txn(kv: &Arc<dyn KVEngine>, txn_id: TxnId) -> Result<()> {
    let key = active_txn_key(txn_id);
    kv.delete(CF_MVCC, &key)
}

/// Get all active transactions
pub fn get_active_txns(kv: &Arc<dyn KVEngine>) -> Result<Vec<(TxnId, u64)>> {
    let mut active = Vec::new();
    let mut iter = kv.prefix_iter(CF_MVCC, KEY_ACTIVE_TXNS)?;
    while iter.next() {
        let value = iter.value().to_vec();
        let txn: (TxnId, u64) =
            crate::codec::decode(&value).map_err(|e| RockDuckError::Codec(e.to_string()))?;
        active.push(txn);
    }
    Ok(active)
}

/// Store segment metadata
pub fn put_segment_meta(kv: &Arc<dyn KVEngine>, meta: &SegmentMeta) -> Result<()> {
    let key = segment_meta_key(&meta.seg_id);
    let value = crate::codec::encode(meta).map_err(|e| RockDuckError::Codec(e.to_string()))?;
    kv.put(CF_SEG_META, &key, &value)
}

/// Store segment metadata and invalidate the cache entry if present.
///
/// Call this instead of `put_segment_meta` when the segment metadata may already
/// be cached, to prevent stale cache reads.
pub fn put_segment_meta_and_invalidate(
    kv: &Arc<dyn KVEngine>,
    cache: &crate::metadata::seg_meta::SegmentMetaCache,
    meta: &SegmentMeta,
) -> Result<()> {
    let key = segment_meta_key(&meta.seg_id);
    let value = crate::codec::encode(meta).map_err(|e| RockDuckError::Codec(e.to_string()))?;
    kv.put(CF_SEG_META, &key, &value)?;
    // Invalidate cache to prevent stale reads after metadata update.
    cache.invalidate(&meta.seg_id);
    Ok(())
}

/// Get segment metadata
pub fn get_segment_meta(kv: &Arc<dyn KVEngine>, seg_id: &str) -> Result<Option<SegmentMeta>> {
    let key = segment_meta_key(seg_id);
    match kv.get(CF_SEG_META, &key)? {
        Some(value) => {
            let meta: SegmentMeta =
                crate::codec::decode(&value).map_err(|e| RockDuckError::Codec(e.to_string()))?;
            Ok(Some(meta))
        }
        None => Ok(None),
    }
}

/// Update segment status
///
/// Returns `Ok(true)` if the segment was found and updated,
/// `Ok(false)` if the segment doesn't exist.
///
/// # Cache Invalidation
/// This function invalidates the segment's cache entry to prevent stale reads.
pub fn update_segment_status(
    kv: &Arc<dyn KVEngine>,
    cache: &crate::metadata::seg_meta::SegmentMetaCache,
    seg_id: &str,
    status: crate::segment::meta::SegmentStatus,
) -> Result<bool> {
    if let Some(mut meta) = get_segment_meta(kv, seg_id)? {
        meta.status = status;
        put_segment_meta(kv, &meta)?;
        // CRITICAL: invalidate cache to prevent stale reads.
        // Without this, reads after update could return the old cached metadata.
        cache.invalidate(seg_id);
        Ok(true)
    } else {
        Ok(false)
    }
}

/// List all segment metadata entries by scanning the seg_meta bucket
pub fn list_segment_metas(kv: &Arc<dyn KVEngine>) -> Result<Vec<SegmentMeta>> {
    let mut metas = Vec::new();
    let mut iter = kv.prefix_iter(CF_SEG_META, KEY_SEGMENT_META)?;
    while iter.next() {
        let value = iter.value().to_vec();
        let meta: SegmentMeta =
            crate::codec::decode(&value).map_err(|e| RockDuckError::Codec(e.to_string()))?;
        metas.push(meta);
    }
    Ok(metas)
}

// =============================================================================
// Private key helpers
// =============================================================================

fn segment_meta_key(seg_id: &str) -> Vec<u8> {
    let mut key = Vec::with_capacity(KEY_SEGMENT_META.len() + seg_id.len());
    key.extend_from_slice(KEY_SEGMENT_META);
    key.extend_from_slice(seg_id.as_bytes());
    key
}

fn table_stats_key(table: &str) -> Vec<u8> {
    let mut key = Vec::with_capacity(KEY_TABLE_STATS.len() + table.len());
    key.extend_from_slice(KEY_TABLE_STATS);
    key.extend_from_slice(table.as_bytes());
    key
}

/// Key for the atomic row count counter of a table.
/// Stored separately from TableStats to enable O(1) atomic_increment instead of
/// read-modify-write of the full blob.
fn table_row_count_key(table: &str) -> Vec<u8> {
    let mut key = Vec::with_capacity(KEY_TABLE_STATS.len() + table.len() + 10);
    key.extend_from_slice(KEY_TABLE_STATS);
    key.extend_from_slice(table.as_bytes());
    key.extend_from_slice(b":row_count");
    key
}

/// Atomically increment the row count of a table.
/// Uses KVEngine::atomic_increment for O(1) instead of read-modify-write.
/// Returns the new value after increment.
pub fn increment_row_count(kv: &Arc<dyn KVEngine>, table: &str, delta: i64) -> Result<u64> {
    kv.atomic_increment(CF_STAT, &table_row_count_key(table), delta)
        .map(|v| v as u64)
}

fn active_txn_key(txn_id: TxnId) -> Vec<u8> {
    let mut key = Vec::with_capacity(KEY_ACTIVE_TXNS.len() + 8);
    key.extend_from_slice(KEY_ACTIVE_TXNS);
    key.extend_from_slice(&txn_id.to_le_bytes());
    key
}

/// Persist a committed transaction's commit timestamp for strict MVCC visibility checks.
///
/// Stores "committed:<txn_id>" -> commit_ts in the MVCC column family.
/// This enables WAL recovery to reconstruct the committed_history map after crash.
pub fn commit_txn_record(kv: &Arc<dyn KVEngine>, txn_id: TxnId, commit_ts: u64) -> Result<()> {
    let key = committed_txn_key(txn_id);
    let value = crate::codec::encode(&(txn_id, commit_ts))
        .map_err(|e| RockDuckError::Codec(e.to_string()))?;
    kv.put(CF_MVCC, &key, &value)
}

/// Recover all committed transaction commit timestamps from the KV store.
///
/// Scans the "committed:<txn_id>" entries and returns a HashMap of txn_id -> commit_ts.
/// Called during `RockDuck::open_with_config` after WAL recovery to restore
/// `VisibilityManager::committed_history` so strict MVCC visibility checks work correctly.
pub fn get_committed_txns(kv: &Arc<dyn KVEngine>) -> Result<HashMap<TxnId, u64>> {
    let mut history = HashMap::new();
    let mut iter = kv.prefix_iter(CF_MVCC, KEY_COMMITTED_TXN_HISTORY)?;
    while iter.next() {
        let value = iter.value().to_vec();
        if let Ok((txn_id, commit_ts)) = crate::codec::decode::<(TxnId, u64)>(&value) {
            history.insert(txn_id, commit_ts);
        }
    }
    Ok(history)
}

fn committed_txn_key(txn_id: TxnId) -> Vec<u8> {
    let mut key = Vec::with_capacity(KEY_COMMITTED_TXN_HISTORY.len() + 8);
    key.extend_from_slice(KEY_COMMITTED_TXN_HISTORY);
    key.extend_from_slice(&txn_id.to_le_bytes());
    key
}

// =============================================================================
// Segment aliasing — PK index remapping after compaction
// =============================================================================

/// Build the alias key: seg_alias:<old_seg_id>
fn seg_alias_key_impl(old_seg_id: &str) -> Vec<u8> {
    let mut key = Vec::with_capacity(KEY_SEG_ALIAS.len() + old_seg_id.len());
    key.extend_from_slice(KEY_SEG_ALIAS);
    key.extend_from_slice(old_seg_id.as_bytes());
    key
}

/// Write a segment alias (old_seg_id -> new_seg_id) after successful compaction.
/// Overwrites if an alias already exists (compaction is idempotent).
pub fn put_segment_alias(kv: &Arc<dyn KVEngine>, old_seg_id: &str, new_seg_id: &str) -> Result<()> {
    let key = seg_alias_key_impl(old_seg_id);
    kv.put(CF_SEG_ALIAS, &key, new_seg_id.as_bytes())
}

/// Resolve a segment alias. Returns Some(new_seg_id) if redirected, None otherwise.
pub fn resolve_segment_alias(kv: &dyn KVEngine, seg_id: &str) -> Result<Option<String>> {
    let key = seg_alias_key_impl(seg_id);
    match kv.get(CF_SEG_ALIAS, &key)? {
        Some(v) => String::from_utf8(v)
            .map(Some)
            .map_err(|e| RockDuckError::Internal(format!("seg_alias value is not UTF-8: {}", e))),
        None => Ok(None),
    }
}

/// Resolve a segment ID, following the alias chain.
/// Returns (final_seg_id, is_redirected).
pub fn resolve_seg_id(kv: &dyn KVEngine, seg_id: &str) -> Result<(String, bool)> {
    match resolve_segment_alias(kv, seg_id)? {
        Some(redirected_id) => {
            tracing::debug!("Segment alias redirect: {} -> {}", seg_id, redirected_id);
            Ok((redirected_id, true))
        }
        None => Ok((seg_id.to_string(), false)),
    }
}
