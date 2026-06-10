//! mace-kv backend adapter for the KVEngine trait
//!
//! mace-kv is a Rust-native ACID+MVCC embedded key-value store with flash-optimized
//! storage. Each column family maps to a mace-kv bucket.
//!
//! ## D1 fix: Cross-process atomicity
//!
//! mace-kv's MVCC transaction (`txn.get` + `txn.upsert` in `atomic_increment`) has a TOCTOU
//! gap: two processes can read the same value and both increment, producing a lost update.
//!
//! We address this with per-process exclusive file locks on a lock directory inside the mace-kv
//! DB path. Each write operation acquires the lock before opening a mace-kv transaction.
//! LockFileEx (Windows) and flock (Unix) are used for cross-process exclusion.

use crate::codec::{decode, encode};
use crate::error::{Result, RockDuckError};
use crate::metadata::kv_engine::{KVEngine, KVIter, KVOp};
use rustc_hash::FxHashMap;
use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[cfg(windows)]
use std::os::windows::io::{AsRawHandle, RawHandle};

#[cfg(windows)]
#[link(name = "kernel32")]
extern "system" {
    fn LockFileEx(
        hFile: RawHandle,
        dwFlags: u32,
        dwReserved: u32,
        nNumberOfBytesToLockLow: u32,
        nNumberOfBytesToLockHigh: u32,
        lpOverlapped: *mut std::ffi::c_void,
    ) -> i32;
}

/// mace-kv backed KV engine
pub struct MaceKVEngine {
    db: Arc<mace::Mace>,
    /// Bucket handles, cached for performance.
    buckets: parking_lot::RwLock<FxHashMap<String, Arc<mace::Bucket>>>,
    /// Per-bucket write locks — mace-kv does not support concurrent write txns per bucket.
    /// Use same key space as `buckets` so we can look up the right lock.
    bucket_write_locks: parking_lot::RwLock<FxHashMap<String, Arc<parking_lot::Mutex<()>>>>,
    /// D1 fix: Directory for cross-process file locks.
    /// Each process acquires an exclusive lock on its own lock file before opening a mace-kv txn.
    lock_dir: PathBuf,
}

impl MaceKVEngine {
    pub fn open(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let opts = mace::Options::new(path.as_ref());
        let parsed = opts
            .validate()
            .map_err(|e| RockDuckError::Storage(format!("mace-kv init failed: {e}")))?;
        let db = mace::Mace::new(parsed)
            .map_err(|e| RockDuckError::Storage(format!("mace-kv open failed: {e}")))?;
        // D1 fix: create per-process lock directory
        let lock_dir = path.as_ref().join("_process_locks");
        std::fs::create_dir_all(&lock_dir)
            .map_err(|e| RockDuckError::Io(std::io::Error::other(format!(
                "create lock dir: {e}"))))?;
        Ok(Self {
            db: Arc::new(db),
            buckets: parking_lot::RwLock::new(FxHashMap::default()),
            bucket_write_locks: parking_lot::RwLock::new(FxHashMap::default()),
            lock_dir,
        })
    }

    /// Get a bucket, with in-memory caching of handles and write lock.
    fn get_bucket(&self, name: &str) -> Result<Arc<mace::Bucket>> {
        {
            let buckets = self.buckets.read();
            if let Some(b) = buckets.get(name) {
                return Ok(Arc::clone(b));
            }
        }
        let mut buckets = self.buckets.write();
        if let Some(b) = buckets.get(name) {
            return Ok(Arc::clone(b));
        }
        let bucket = self
            .db
            .get_bucket(name)
            .or_else(|_| self.db.new_bucket(name))
            .map_err(|e| RockDuckError::Storage(format!("bucket '{name}': {e}")))?;
        let bucket = Arc::new(bucket);
        buckets.insert(name.to_string(), Arc::clone(&bucket));

        // Also create the write lock for this bucket.
        let mut locks = self.bucket_write_locks.write();
        locks
            .entry(name.to_string())
            .or_insert_with(|| Arc::new(parking_lot::Mutex::new(())));

        Ok(Arc::clone(&bucket))
    }

    /// Get the write lock for a bucket.
    ///
    /// IMPORTANT: Must be called AFTER `get_bucket()` in all callers to ensure
    /// consistent lock ordering (buckets before bucket_write_locks) and prevent deadlocks.
    #[allow(dead_code)]
    fn get_bucket_write_lock(&self, name: &str) -> Arc<parking_lot::Mutex<()>> {
        // get_bucket ensures the bucket entry exists in bucket_write_locks.
        // We only need to acquire the lock here.
        let _ = self.get_bucket(name); // ensures consistent ordering
        let locks = self.bucket_write_locks.read();
        locks.get(name).map(Arc::clone).unwrap_or_else(|| {
            drop(locks);
            let mut locks = self.bucket_write_locks.write();
            locks
                .entry(name.to_string())
                .or_insert_with(|| Arc::new(parking_lot::Mutex::new(())))
                .clone()
        })
    }

    /// D1 fix: Acquire an exclusive cross-process file lock for a bucket write operation.
    ///
    /// Uses LockFileEx on Windows (exclusive, non-blocking) and flock on Unix.
    /// The lock is released when the returned guard is dropped.
    fn acquire_cross_process_lock(&self, bucket: &str) -> Result<CrossProcessLockGuard> {
        CrossProcessLockGuard::acquire(&self.lock_dir, bucket)
    }
}

/// D1 fix: RAII guard for cross-process file locks.
///
/// Holds an exclusive lock on a per-bucket lock file for the duration of a mace-kv
/// write transaction. The lock is released when this guard is dropped.
struct CrossProcessLockGuard {
    #[cfg(windows)]
    file: File,
    #[cfg(unix)]
    _file: File,
}

impl CrossProcessLockGuard {
    fn acquire(lock_dir: &Path, bucket: &str) -> Result<Self> {
        let sanitized = bucket.replace(['/', '\\', ':', '.'], "_");
        let lock_file_path = lock_dir.join(format!("{}.lock", sanitized));

        let file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .read(true)
            .open(&lock_file_path)
            .map_err(RockDuckError::Io)?;

        #[cfg(windows)]
        {
            const LOCKFILE_EXCLUSIVE_LOCK: u32 = 0x00000002;
            const LOCKFILE_FAIL_IMMEDIATELY: u32 = 0x00000001;

            // SAFETY: OVERLAPPED is a simple C struct with no Drop implementation.
            // We zero-initialize it before passing to LockFileEx, which reads/writes
            // the structure. The c_void type is used as a placeholder for the
            // OVERLAPPED structure required by the Windows API.
            let result = unsafe {
                let mut overlapped = std::mem::zeroed::<std::ffi::c_void>();
                LockFileEx(
                    file.as_raw_handle(),
                    LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY,
                    0,
                    1,
                    0,
                    &mut overlapped,
                )
            };
            if result == 0 {
                return Err(RockDuckError::Storage(
                    "cross-process lock acquisition failed: file is locked by another process"
                        .into(),
                ));
            }
        }

        #[cfg(unix)]
        {
            use std::os::unix::io::AsRawFd;
            let fd = file.as_raw_fd();
            let result = unsafe { libc::flock(fd, libc::LOCK_EX | libc::LOCK_NB) };
            if result != 0 {
                let errno = std::io::Error::last_os_error();
                return Err(RockDuckError::Storage(format!(
                    "cross-process lock acquisition failed: {}",
                    errno
                )));
            }
        }

        Ok(Self {
            #[cfg(windows)]
            file,
            #[cfg(unix)]
            _file: file,
        })
    }
}

impl Drop for CrossProcessLockGuard {
    fn drop(&mut self) {
        #[cfg(windows)]
        {
            let result = unsafe {
                let mut overlapped = std::mem::zeroed::<std::ffi::c_void>();
                LockFileEx(
                    self.file.as_raw_handle(),
                    0x00000001,
                    0,
                    1,
                    0,
                    &mut overlapped,
                )
            };
            let _ = result; // ignore unlock errors during drop
        }
        #[cfg(unix)]
        {
            use std::os::unix::io::AsRawFd;
            let _ = unsafe { libc::flock(self._file.as_raw_fd(), libc::LOCK_UN) };
        }
    }
}

impl KVEngine for MaceKVEngine {
    fn get(&self, bucket: &str, key: &[u8]) -> Result<Option<Vec<u8>>> {
        let bucket = self.get_bucket(bucket)?;
        let txn = bucket
            .view()
            .map_err(|e| RockDuckError::Storage(format!("mace-kv view: {e}")))?;
        match txn.get(key) {
            Ok(val) => Ok(Some(val.to_vec())),
            Err(mace::OpCode::NotFound) => Ok(None),
            Err(e) => Err(RockDuckError::Storage(format!("mace-kv get: {e}"))),
        }
    }

    fn put(&self, bucket: &str, key: &[u8], value: &[u8]) -> Result<()> {
        let bucket_arc = self.get_bucket(bucket)?;
        // D1 fix: cross-process file lock — serializes writes across processes
        let _cross_lock = self.acquire_cross_process_lock(bucket)?;
        let txn = bucket_arc
            .begin()
            .map_err(|e| RockDuckError::Storage(format!("mace-kv begin: {e}")))?;
        // mace-kv's `put` is strict insert-only (fails if key exists).
        // Use `upsert` which inserts or updates — matching the KVEngine trait semantics.
        txn.upsert(key, value)
            .map_err(|e| RockDuckError::Storage(format!("mace-kv upsert: {e}")))?;
        txn.commit()
            .map_err(|e| RockDuckError::Storage(format!("mace-kv commit: {e}")))?;
        Ok(())
    }

    fn delete(&self, bucket: &str, key: &[u8]) -> Result<()> {
        let bucket_arc = self.get_bucket(bucket)?;
        // D1 fix: cross-process file lock
        let _cross_lock = self.acquire_cross_process_lock(bucket)?;
        let txn = bucket_arc
            .begin()
            .map_err(|e| RockDuckError::Storage(format!("mace-kv begin: {e}")))?;
        txn.del(key)
            .map_err(|e| RockDuckError::Storage(format!("mace-kv del: {e}")))?;
        txn.commit()
            .map_err(|e| RockDuckError::Storage(format!("mace-kv commit: {e}")))?;
        Ok(())
    }

    fn prefix_iter(&self, bucket: &str, prefix: &[u8]) -> Result<Box<dyn KVIter>> {
        let bucket = self.get_bucket(bucket)?;
        Ok(Box::new(MaceKVIter::new(bucket, prefix.to_vec())))
    }

    fn write_batch(&self, bucket: &str, ops: &[KVOp]) -> Result<()> {
        let bucket_arc = self.get_bucket(bucket)?;
        // D1 fix: cross-process file lock
        let _cross_lock = self.acquire_cross_process_lock(bucket)?;
        let txn = bucket_arc
            .begin()
            .map_err(|e| RockDuckError::Storage(format!("mace-kv begin: {e}")))?;
        for op in ops {
            match op {
                KVOp::Put { key, value } => {
                    txn.upsert(key.as_slice(), value.as_slice())
                        .map_err(|e| RockDuckError::Storage(format!("mace-kv upsert: {e}")))?;
                }
                KVOp::Delete { key } => {
                    txn.del(key.as_slice())
                        .map_err(|e| RockDuckError::Storage(format!("mace-kv del: {e}")))?;
                }
            }
        }
        txn.commit()
            .map_err(|e| RockDuckError::Storage(format!("mace-kv commit: {e}")))?;
        Ok(())
    }

    fn flush(&self) -> Result<()> {
        self.db
            .sync()
            .map_err(|e| RockDuckError::Storage(format!("mace-kv sync: {e}")))?;
        Ok(())
    }

    fn atomic_increment(&self, bucket: &str, key: &[u8], delta: i64) -> Result<i64> {
        let bucket_arc = self.get_bucket(bucket)?;
        // D1 fix: cross-process file lock — closes the TOCTOU gap in txn.get + txn.upsert
        let _cross_lock = self.acquire_cross_process_lock(bucket)?;
        let txn = bucket_arc
            .begin()
            .map_err(|e| RockDuckError::Storage(format!("mace-kv begin: {e}")))?;

        // Read current value
        let old_val: i64 = match txn.get(key) {
            Ok(val) => decode::<i64>(&val.to_vec())
                .map_err(|e| RockDuckError::Codec(format!("atomic_increment decode: {e}")))?,
            Err(mace::OpCode::NotFound) => 0,
            Err(e) => return Err(RockDuckError::Storage(format!("mace-kv get: {e}"))),
        };

        let new_val = old_val.saturating_add(delta);
        let new_bytes = encode(&new_val)
            .map_err(|e| RockDuckError::Codec(format!("atomic_increment encode: {e}")))?;

        txn.upsert(key, &new_bytes)
            .map_err(|e| RockDuckError::Storage(format!("mace-kv upsert: {e}")))?;
        txn.commit()
            .map_err(|e| RockDuckError::Storage(format!("mace-kv commit: {e}")))?;

        Ok(new_val)
    }
}

/// mace-kv iterator that implements the KVIter trait.
///
/// Since mace::Iter has lifetime constraints that prevent storing it in a trait object,
/// we re-create the view+iter on each call to next(). We use a seek-key approach:
///
/// - First call: seek to prefix, return first matching key
/// - Subsequent calls: seek to last_key + 1, return next matching key
///
///   This requires keys to be comparable (which they are as bytes).
struct MaceKVIter {
    bucket: Arc<mace::Bucket>,
    prefix: Vec<u8>,
    /// The last key we returned. On next() we seek past this to avoid duplicates.
    last_key: Option<Vec<u8>>,
    current_key: Vec<u8>,
    current_value: Vec<u8>,
}

impl MaceKVIter {
    fn new(bucket: Arc<mace::Bucket>, prefix: Vec<u8>) -> Self {
        Self {
            bucket,
            prefix,
            last_key: None,
            current_key: Vec::new(),
            current_value: Vec::new(),
        }
    }

    /// Compute the lexicographic successor of a key (key + 1 as bytes).
    fn next_key(key: &[u8]) -> Vec<u8> {
        let mut succ = key.to_vec();
        succ.push(0);
        succ
    }

    /// Advance to the next key matching prefix, starting from `seek_key`.
    /// Returns true if a matching key was found.
    fn advance(&mut self, seek_key: &[u8]) -> bool {
        let view = match self.bucket.view() {
            Ok(v) => v,
            Err(_) => return false,
        };
        // seek() returns Iter directly, not Result
        let mut iter = view.seek(seek_key);
        if let Some(item) = iter.next() {
            let k = item.key();
            // Only return if key starts with our prefix
            if k.starts_with(&self.prefix) {
                self.current_key = k.to_vec();
                self.current_value = item.val().to_vec();
                return true;
            }
        }
        false
    }
}

impl KVIter for MaceKVIter {
    fn next(&mut self) -> bool {
        let seek_key = match &self.last_key {
            None => self.prefix.clone(),
            Some(last) => Self::next_key(last),
        };

        if self.advance(&seek_key) {
            self.last_key = Some(self.current_key.clone());
            true
        } else {
            false
        }
    }

    fn seek(&mut self, key: &[u8]) {
        self.last_key = None;
        // Seek to the provided key and position at the first matching entry.
        // The caller will call next() to advance.
        let _ = self.advance(key);
    }

    fn key(&self) -> &[u8] {
        &self.current_key
    }

    fn value(&self) -> &[u8] {
        &self.current_value
    }
}
