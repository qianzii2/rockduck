//! File manager for storage layer
//!
//! Manages file lifecycle for delta, vortex, and frozen segments.
//!
//! ## mmap-io Integration
//!
//! FileManager provides:
//! - `open()`: regular file handle with reference counting
//! - `open_mmap()`: zero-copy memory-mapped access via `MmapFile`
//!
//! All mmap operations go through `MmapFile` (in `mmap_file.rs`).
//! `mmap-io` features (chunked iteration, atomic views, advise) are available
//! through the `MmapFile` API.

use parking_lot::RwLock;
use rustc_hash::FxHashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use super::mmap_file::MmapFile;
use crate::error::Result;
use crate::storage::MmapReader;

// ─── FileHandle ─────────────────────────────────────────────────────────────

/// File handle with reference counting and optional mmap support.
pub struct FileHandle {
    /// File path.
    pub path: PathBuf,
    /// Reference count (protected by Arc).
    ref_count: RwLock<usize>,
    /// File size.
    pub size: u64,
    /// Is memory-mapped.
    pub is_mmap: bool,
    /// Optional memory-mapped file for zero-copy access.
    mmap: Option<Arc<MmapFile>>,
}

impl FileHandle {
    /// Create a new file handle (regular file, not mmap'd).
    pub fn new(path: PathBuf, size: u64) -> Self {
        Self {
            path,
            ref_count: RwLock::new(1),
            size,
            is_mmap: false,
            mmap: None,
        }
    }

    /// Create a new file handle with mmap support.
    pub fn new_mmap(path: PathBuf, mmap: Arc<MmapFile>) -> Self {
        Self {
            path: path.clone(),
            ref_count: RwLock::new(1),
            size: mmap.len() as u64,
            is_mmap: true,
            mmap: Some(mmap),
        }
    }

    /// Increment reference count.
    pub fn acquire(&self) {
        *self.ref_count.write() += 1;
    }

    /// Decrement reference count and return the new value.
    pub fn release(&self) -> usize {
        let mut count = self.ref_count.write();
        *count -= 1;
        *count
    }

    /// Get current reference count.
    pub fn ref_count(&self) -> usize {
        *self.ref_count.read()
    }

    /// Get the mmap file if available (zero-copy reads).
    pub fn get_mmap(&self) -> Option<&Arc<MmapFile>> {
        self.mmap.as_ref()
    }

    /// Get a `MmapReader` for zero-copy access.
    ///
    /// Returns the existing mmap as a reader, or opens the file as a new reader.
    pub fn get_reader(&self) -> Result<MmapReader> {
        if let Some(ref mmap_arc) = self.mmap {
            return Ok(MmapReader::from_mmap(mmap_arc.clone()));
        }
        // Open as a new reader if not mmap'd
        MmapReader::open(&self.path)
    }
}

// ─── FileManager ─────────────────────────────────────────────────────────────

/// File manager for managing segment file lifecycle.
///
/// Provides:
/// - Reference-counted file handle cache
/// - Zero-copy mmap access for read-heavy workloads
/// - Automatic handle cleanup when reference count reaches zero
pub struct FileManager {
    /// Data directory.
    data_dir: PathBuf,
    /// Open file handles (cached).
    handles: RwLock<FxHashMap<PathBuf, Arc<FileHandle>>>,
    /// Open mmap readers (cached separately for reference counting).
    readers: RwLock<FxHashMap<PathBuf, Arc<MmapReader>>>,
}

impl FileManager {
    /// Create a new file manager.
    pub fn new(data_dir: PathBuf) -> Self {
        Self {
            data_dir,
            handles: RwLock::new(FxHashMap::default()),
            readers: RwLock::new(FxHashMap::default()),
        }
    }

    /// Get the data directory.
    pub fn data_dir(&self) -> &Path {
        &self.data_dir
    }

    /// Open or get a cached file handle.
    ///
    /// Fast path (cache hit): checks the read-lock cache first and returns immediately.
    /// Slow path (cache miss): acquires write lock, opens the file, and caches it.
    /// This avoids unnecessary `std::fs::metadata()` syscalls on cache hits.
    pub fn open(&self, path: &Path) -> Result<Arc<FileHandle>> {
        // Fast path: check cache under read lock first
        {
            let handles = self.handles.read();
            if let Some(existing) = handles.get(path) {
                return Ok(Arc::clone(existing));
            }
        }

        // Slow path: cache miss — open the file and add to cache
        let metadata = std::fs::metadata(path)?;
        let handle = Arc::new(FileHandle::new(path.to_path_buf(), metadata.len()));

        let mut handles = self.handles.write();
        // Use entry().and_modify() + or_insert_with() to avoid re-opening
        // if another thread added it while we were waiting for the write lock
        let entry = handles.entry(path.to_path_buf());
        Ok(entry
            .and_modify(|h| {
                h.acquire();
            })
            .or_insert_with(|| {
                handle.acquire();
                Arc::clone(&handle)
            })
            .clone())
    }

    /// Open a file with memory mapping for zero-copy reads.
    /// Uses the same atomic check-and-insert pattern as `open()`.
    pub fn open_mmap(&self, path: &Path) -> Result<Arc<FileHandle>> {
        let mmap = Arc::new(MmapFile::open(path)?);
        let handle = Arc::new(FileHandle::new_mmap(path.to_path_buf(), mmap));

        let mut handles = self.handles.write();
        let entry = handles.entry(path.to_path_buf());
        Ok(entry
            .or_insert_with(|| {
                handle.acquire();
                handle.clone()
            })
            .clone())
    }

    /// Open a file as a zero-copy `MmapReader`.
    /// Returns the cached `Arc<MmapReader>` instead of the original clone.
    /// Previously returned `reader` (original) while caching `Arc::new(reader.clone())`,
    /// so each call to `open_reader` returned a different instance — defeating the cache.
    pub fn open_reader(&self, path: &Path) -> Result<Arc<MmapReader>> {
        let reader = MmapReader::open(path)?;

        let mut readers = self.readers.write();
        let cached = readers
            .entry(path.to_path_buf())
            .or_insert_with(|| Arc::new(reader));

        Ok(cached.clone())
    }

    /// Close a file handle, decrementing its reference count.
    ///
    /// If the refcount reaches zero, the handle is removed from the cache.
    /// If the handle was mmap'd, the mmap is automatically unmapped.
    pub fn close(&self, path: &Path) -> Result<()> {
        let mut handles = self.handles.write();
        if let Some(handle) = handles.remove(path) {
            let remaining = handle.release();
            tracing::debug!("closed file {:?}, remaining refcount={}", path, remaining);
        }
        Ok(())
    }

    /// Close a reader from the cache.
    pub fn close_reader(&self, path: &Path) -> Result<()> {
        let mut readers = self.readers.write();
        if readers.remove(path).is_some() {
            tracing::debug!("closed mmap reader for {:?}", path);
        }
        Ok(())
    }

    /// Get all open file paths.
    pub fn open_files(&self) -> Vec<PathBuf> {
        let handles = self.handles.read();
        handles.keys().cloned().collect()
    }

    /// Get total size of open files.
    pub fn total_open_size(&self) -> u64 {
        let handles = self.handles.read();
        handles.values().map(|h| h.size).sum()
    }

    /// Get the number of open handles (for monitoring).
    pub fn handle_count(&self) -> usize {
        self.handles.read().len()
    }

    /// Get the number of open mmap readers (for monitoring).
    pub fn reader_count(&self) -> usize {
        self.readers.read().len()
    }

    /// Flush all open mmap handles to the OS page cache.
    ///
    /// This flushes modified pages to the OS page cache but does NOT call fsync —
    /// the OS may still hold data in its buffer cache. For durable flush, use
    /// `flush_all_to_disk()` instead.
    pub fn flush_all(&self) -> std::io::Result<()> {
        let handles = self.handles.read();
        for handle in handles.values() {
            if let Some(ref mmap) = handle.mmap {
                mmap.flush_to_disk()?;
            }
        }
        Ok(())
    }

    /// Flush and sync all open mmap handles to physical disk.
    ///
    /// Internally calls `mmap.flush_to_disk()` which maps to `mmap.flush()` + `file.sync_all()`
    /// on POSIX, achieving true durability. This is expensive — do not call on every write.
    pub fn flush_all_to_disk(&self) -> std::io::Result<()> {
        let handles = self.handles.read();
        for handle in handles.values() {
            if let Some(ref mmap) = handle.mmap {
                mmap.flush_to_disk()?;
            }
        }
        Ok(())
    }
}
