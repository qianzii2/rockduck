//! Key-Value engine abstraction layer
//!
//! Provides a unified interface over different KV store backends (RocksDB, mace-kv).
//! This allows the storage backend to be swapped without changing metadata code.

use crate::error::Result;

/// Key-value engine trait for storage abstraction
pub trait KVEngine: Send + Sync {
    /// Get a value by key in a bucket/column family
    fn get(&self, bucket: &str, key: &[u8]) -> Result<Option<Vec<u8>>>;

    /// Put a key-value pair into a bucket
    fn put(&self, bucket: &str, key: &[u8], value: &[u8]) -> Result<()>;

    /// Delete a key from a bucket
    fn delete(&self, bucket: &str, key: &[u8]) -> Result<()>;

    /// Get an iterator over keys matching a prefix in a bucket
    fn prefix_iter(&self, bucket: &str, prefix: &[u8]) -> Result<Box<dyn KVIter>>;

    /// Execute a batch write operation
    fn write_batch(&self, bucket: &str, ops: &[KVOp]) -> Result<()>;

    /// Flush any pending writes
    fn flush(&self) -> Result<()>;

    /// Atomic increment: reads the current i64 value, adds `delta`, and writes back.
    /// All within a single ACID transaction. Returns the NEW value after increment.
    /// If the key does not exist, treats it as 0.
    fn atomic_increment(&self, bucket: &str, key: &[u8], delta: i64) -> Result<i64>;
}

/// Single operation in a KV batch
#[derive(Debug, Clone)]
pub enum KVOp {
    Put { key: Vec<u8>, value: Vec<u8> },
    Delete { key: Vec<u8> },
}

/// Key-value iterator trait
pub trait KVIter {
    /// Move to the next key-value pair
    fn next(&mut self) -> bool;
    /// Get the current key
    fn key(&self) -> &[u8];
    /// Get the current value
    fn value(&self) -> &[u8];

    /// Seek to the first key >= `key`.
    /// After calling seek, the next call to `next()` returns the first matching key.
    /// The initial position is unspecified after seek — caller must call `next()` to advance.
    fn seek(&mut self, key: &[u8]);
}

// =============================================================================
// Bucket/Column Family name constants (shared between RocksDB and mace-kv backends)
// =============================================================================

/// Primary key index bucket
pub const CF_PK_IDX: &str = "pk_idx";
/// Segment metadata bucket
pub const CF_SEG_META: &str = "seg_meta";
/// Table statistics bucket
pub const CF_STAT: &str = "stat";
/// Zone map bucket
pub const CF_ZONE: &str = "zone";
/// Layer/L0 tracking bucket
pub const CF_LAYER: &str = "layer";
/// Bloom filter bucket
pub const CF_BF: &str = "bf";
/// System metadata bucket
pub const CF_LBF: &str = "lbf";
pub const CF_SYS: &str = "sys";
/// MVCC transaction state bucket
pub const CF_MVCC: &str = "mvcc";
/// Iceberg table metadata bucket
pub const CF_ICEBERG: &str = "iceberg";
/// Version index bucket — stores reverse timestamp index for time travel
pub const CF_VERSIONS: &str = "versions";
/// Delta layer metadata bucket — stores (seg_id, col) → DeltaFileId mapping
pub const CF_DELTA: &str = "delta";

// =============================================================================
// Blanket impl for Arc<dyn KVEngine> — delegates to inner trait object
// =============================================================================

impl KVEngine for std::sync::Arc<dyn KVEngine> {
    fn get(&self, bucket: &str, key: &[u8]) -> Result<Option<Vec<u8>>> {
        (**self).get(bucket, key)
    }

    fn put(&self, bucket: &str, key: &[u8], value: &[u8]) -> Result<()> {
        (**self).put(bucket, key, value)
    }

    fn delete(&self, bucket: &str, key: &[u8]) -> Result<()> {
        (**self).delete(bucket, key)
    }

    fn prefix_iter(&self, bucket: &str, prefix: &[u8]) -> Result<Box<dyn KVIter>> {
        (**self).prefix_iter(bucket, prefix)
    }

    fn write_batch(&self, bucket: &str, ops: &[KVOp]) -> Result<()> {
        (**self).write_batch(bucket, ops)
    }

    fn flush(&self) -> Result<()> {
        (**self).flush()
    }

    fn atomic_increment(&self, bucket: &str, key: &[u8], delta: i64) -> Result<i64> {
        (**self).atomic_increment(bucket, key, delta)
    }
}
