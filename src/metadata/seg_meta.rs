//! Segment metadata cache
//!
//! Provides a bounded cache for SegmentMeta to reduce KV store lookups.
//! Cache stores both decoded `SegmentMeta` and encoded bytes to avoid
//! re-encoding on writes back to the KV store.

use moka::sync::Cache;
use parking_lot::RwLock;
use std::sync::Arc;

use crate::codec::encode;
use crate::error::{Result, RockDuckError};
use crate::segment::meta::SegmentMeta;

/// Cache entry storing both decoded and encoded forms.
#[derive(Clone)]
struct CacheEntry {
    meta: SegmentMeta,
    #[allow(dead_code)]
    encoded: Vec<u8>,
}

/// Thread-safe bounded cache for SegmentMeta.
///
/// Stores both the decoded struct and its encoded bytes to avoid
/// re-serializing on every write. Entries are invalidated when
/// the underlying data changes.
///
/// Maintains a secondary index by table_id to accelerate find_active_segment_for_table
/// queries, avoiding a full KV scan when the cache is warm.
pub struct SegmentMetaCache {
    cache: Cache<String, CacheEntry>,
    present: RwLock<std::collections::HashSet<String>>,
    /// Secondary index: table_id -> ordered list of seg_ids for that table.
    /// Kept in sync with `cache` via put() and invalidate().
    table_index: RwLock<std::collections::HashMap<String, Vec<String>>>,
}

impl SegmentMetaCache {
    /// Create a new cache with the given capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            cache: Cache::new(capacity as u64),
            present: RwLock::new(std::collections::HashSet::new()),
            table_index: RwLock::new(std::collections::HashMap::new()),
        }
    }

    /// Get a cached segment by ID. Returns None if not in cache.
    pub fn get(&self, seg_id: &str) -> Option<SegmentMeta> {
        self.cache.get(seg_id).map(|e| e.meta.clone())
    }

    /// Read-through get: try cache first, fall back to KV if not present.
    ///
    /// On cache miss, loads from KV and populates the cache so subsequent calls
    /// hit the cache without another KV lookup. This avoids the cache-invalidation window
    /// where `put_segment_meta` invalidates the cache and the next read goes KV-direct
    /// instead of re-populating the cache.
    pub fn get_with_repopulate(
        &self,
        kv: &Arc<dyn crate::metadata::KVEngine>,
        seg_id: &str,
    ) -> Result<Option<SegmentMeta>> {
        // Fast path: cache hit
        if let Some(meta) = self.get(seg_id) {
            return Ok(Some(meta));
        }

        // Slow path: load from KV
        let key = {
            let prefix = crate::metadata::kv_engine::CF_SEG_META;
            let mut k = Vec::with_capacity(prefix.len() + seg_id.len());
            k.extend_from_slice(prefix.as_bytes());
            k.extend_from_slice(seg_id.as_bytes());
            k
        };
        let value = match kv.get(crate::metadata::kv_engine::CF_SEG_META, &key)? {
            Some(v) => v,
            None => return Ok(None),
        };
        let meta: SegmentMeta =
            crate::codec::decode(&value).map_err(|e| RockDuckError::Codec(e.to_string()))?;

        // Repopulate cache so future reads are fast
        self.put(&meta)?;

        Ok(Some(meta))
    }

    /// Put a segment into the cache. Overwrites existing entry.
    /// Also updates the secondary table_index so find_active_segment_for_table stays fast.
    pub fn put(&self, meta: &SegmentMeta) -> Result<()> {
        let encoded = encode(meta).map_err(|e| RockDuckError::Codec(e.to_string()))?;

        let entry = CacheEntry {
            meta: meta.clone(),
            encoded,
        };

        self.cache.insert(meta.seg_id.clone(), entry);
        self.present.write().insert(meta.seg_id.clone());

        // Update secondary index (dedup: don't push if seg_id already exists)
        let mut idx = self.table_index.write();
        let entry = idx.entry(meta.table_id.clone()).or_default();
        if !entry.contains(&meta.seg_id) {
            entry.push(meta.seg_id.clone());
        }
        Ok(())
    }

    /// Invalidate a cached segment. Call this when the segment is updated.
    /// Removes from both the primary cache and the table_index.
    pub fn invalidate(&self, seg_id: &str) {
        self.cache.invalidate(seg_id);
        self.present.write().remove(seg_id);
        // Also remove from table_index (scan and remove from all table lists).
        let mut idx = self.table_index.write();
        for seg_ids in idx.values_mut() {
            seg_ids.retain(|s| s != seg_id);
        }
    }

    /// Check if a segment is in cache.
    pub fn contains(&self, seg_id: &str) -> bool {
        self.present.read().contains(seg_id)
    }

    /// Get the number of cached entries.
    pub fn len(&self) -> usize {
        self.present.read().len()
    }

    /// Check if the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.present.read().is_empty()
    }

    /// Clear the entire cache.
    pub fn clear(&self) {
        self.cache.invalidate_all();
        self.present.write().clear();
        self.table_index.write().clear();
    }

    /// Get all segment IDs for a given table from the secondary index.
    /// Returns an empty vec if no segments for this table are cached.
    pub fn get_seg_ids_for_table(&self, table_id: &str) -> Vec<String> {
        self.table_index
            .read()
            .get(table_id)
            .cloned()
            .unwrap_or_default()
    }

    /// Get all cached SegmentMeta for a given table, filtered by status.
    /// Returns an empty vec on cache miss.
    pub fn get_metas_for_table(&self, table_id: &str) -> Vec<SegmentMeta> {
        let seg_ids = self.get_seg_ids_for_table(table_id);
        seg_ids
            .into_iter()
            .filter_map(|seg_id| self.get(&seg_id))
            .collect()
    }
}

impl Default for SegmentMetaCache {
    fn default() -> Self {
        Self::new(1024)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_get_returns_none_when_empty() {
        let cache = SegmentMetaCache::new(10);
        assert!(cache.get("nonexistent").is_none());
    }

    #[test]
    fn test_cache_contains() {
        let cache = SegmentMetaCache::new(10);
        assert!(!cache.contains("seg_123"));

        // Create a dummy segment meta
        let meta = SegmentMeta::new("seg_123".to_string(), "test_table".to_string(), vec![]);
        cache.put(&meta).unwrap();

        assert!(cache.contains("seg_123"));
    }

    #[test]
    fn test_cache_invalidate() {
        let cache = SegmentMetaCache::new(10);

        let meta = SegmentMeta::new("seg_123".to_string(), "test_table".to_string(), vec![]);
        cache.put(&meta).unwrap();
        assert!(cache.contains("seg_123"));

        cache.invalidate("seg_123");
        assert!(!cache.contains("seg_123"));
    }

    #[test]
    fn test_cache_clear() {
        let cache = SegmentMetaCache::new(10);

        for i in 0..5 {
            let meta = SegmentMeta::new(format!("seg_{}", i), "test_table".to_string(), vec![]);
            cache.put(&meta).unwrap();
        }

        assert_eq!(cache.len(), 5);
        cache.clear();
        assert!(cache.is_empty());
    }

    #[test]
    fn test_table_index() {
        let cache = SegmentMetaCache::new(10);

        let meta1 = SegmentMeta::new("seg_1".to_string(), "table_a".to_string(), vec![]);
        let meta2 = SegmentMeta::new("seg_2".to_string(), "table_a".to_string(), vec![]);
        let meta3 = SegmentMeta::new("seg_3".to_string(), "table_b".to_string(), vec![]);

        cache.put(&meta1).unwrap();
        cache.put(&meta2).unwrap();
        cache.put(&meta3).unwrap();

        let table_a_segs = cache.get_seg_ids_for_table("table_a");
        assert_eq!(table_a_segs.len(), 2);
        assert!(table_a_segs.contains(&"seg_1".to_string()));
        assert!(table_a_segs.contains(&"seg_2".to_string()));

        let table_b_segs = cache.get_seg_ids_for_table("table_b");
        assert_eq!(table_b_segs.len(), 1);
        assert!(table_b_segs.contains(&"seg_3".to_string()));

        let table_c_segs = cache.get_seg_ids_for_table("table_c");
        assert!(table_c_segs.is_empty());
    }

    #[test]
    fn test_invalidate_removes_from_table_index() {
        let cache = SegmentMetaCache::new(10);

        let meta = SegmentMeta::new("seg_1".to_string(), "table_x".to_string(), vec![]);
        cache.put(&meta).unwrap();

        assert!(!cache.get_seg_ids_for_table("table_x").is_empty());

        cache.invalidate("seg_1");

        assert!(cache.get_seg_ids_for_table("table_x").is_empty());
    }

    #[test]
    fn test_cache_put_overwrites() {
        let cache = SegmentMetaCache::new(10);

        let meta1 = SegmentMeta::new("seg_1".to_string(), "test_table".to_string(), vec![]);
        cache.put(&meta1).unwrap();

        let meta2 = SegmentMeta::new("seg_1".to_string(), "test_table".to_string(), vec![]);
        cache.put(&meta2).unwrap();

        assert_eq!(
            cache.len(),
            1,
            "Cache should have only one entry after overwrite"
        );
    }

    #[test]
    fn test_default_cache_capacity() {
        let cache = SegmentMetaCache::default();
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_get_metas_for_table() {
        let cache = SegmentMetaCache::new(10);

        let meta1 = SegmentMeta::new("seg_1".to_string(), "orders".to_string(), vec![]);
        let meta2 = SegmentMeta::new("seg_2".to_string(), "orders".to_string(), vec![]);
        let meta3 = SegmentMeta::new("seg_3".to_string(), "users".to_string(), vec![]);

        cache.put(&meta1).unwrap();
        cache.put(&meta2).unwrap();
        cache.put(&meta3).unwrap();

        let orders_metas = cache.get_metas_for_table("orders");
        assert_eq!(orders_metas.len(), 2);

        let users_metas = cache.get_metas_for_table("users");
        assert_eq!(users_metas.len(), 1);
    }

    #[test]
    fn test_invalidate_nonexistent_is_noop() {
        let cache = SegmentMetaCache::new(10);
        cache.invalidate("nonexistent_seg");
        assert!(cache.is_empty());
    }
}
