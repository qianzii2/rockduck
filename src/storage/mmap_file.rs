//! Memory-mapped file access for zero-copy storage reads and writes.
//!
//! Uses `memmap2` for cross-platform mmap. All slices borrowed from mmap
//! are zero-copy. Flushing is done via `fsync` on the underlying file.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use memmap2::{Mmap, MmapMut};

use crate::error::{Result, RockDuckError};

// ─── MmapFile ───────────────────────────────────────────────────────────────

#[derive(Debug)]
pub struct MmapFile {
    mmap: Mmap,
    path: PathBuf,
}

impl MmapFile {
    /// Open an existing mmap file for reading and writing.
    ///
    /// # Panics
    /// Panics if the file does not exist.
    #[track_caller]
    pub fn open(path: &Path) -> Result<Self> {
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
            .map_err(RockDuckError::Io)?;
        let mmap = unsafe { Mmap::map(&file) }.map_err(RockDuckError::Io)?;
        Ok(Self {
            mmap,
            path: path.to_path_buf(),
        })
    }

    /// Open an existing file for read-only access.
    ///
    /// # Panics
    /// Panics if the file does not exist.
    #[track_caller]
    pub fn open_ro(path: &Path) -> Result<Self> {
        let file = std::fs::OpenOptions::new()
            .read(true)
            .open(path)
            .map_err(RockDuckError::Io)?;
        let mmap = unsafe { Mmap::map(&file) }.map_err(RockDuckError::Io)?;
        Ok(Self {
            mmap,
            path: path.to_path_buf(),
        })
    }

    pub fn open_with_size(path: &Path, size: u64) -> Result<Self> {
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)
            .map_err(RockDuckError::Io)?;
        file.set_len(size).map_err(RockDuckError::Io)?;
        let mmap = unsafe { Mmap::map(&file) }.map_err(RockDuckError::Io)?;
        Ok(Self {
            mmap,
            path: path.to_path_buf(),
        })
    }

    /// Open an existing file and return as Arc<MmapFile>.
    #[track_caller]
    pub fn open_arc(path: &Path) -> Result<Arc<Self>> {
        Ok(Arc::new(Self::open(path)?))
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
    pub fn len(&self) -> usize {
        self.mmap.len()
    }
    pub fn is_empty(&self) -> bool {
        self.mmap.is_empty()
    }

    pub fn slice_at(&self, offset: u64, len: u64) -> Result<&[u8]> {
        let start = offset as usize;
        let end = (start + len as usize).min(self.mmap.len());
        self.mmap.get(start..end).ok_or_else(|| {
            RockDuckError::Internal(format!(
                "mmap slice out of bounds: offset={}, len={}, mmap_len={}",
                offset,
                len,
                self.mmap.len()
            ))
        })
    }

    pub fn chunks(&self, chunk_size: usize) -> impl Iterator<Item = &[u8]> {
        self.mmap.chunks(chunk_size)
    }

    pub fn as_mmap(&self) -> &Mmap {
        &self.mmap
    }

    /// Durable flush via fsync on the underlying file.
    ///
    /// This works for both read-only and read-write mmaps.
    /// On Unix: calls fsync. On Windows: calls FlushFileBuffers.
    pub fn flush_to_disk(&self) -> std::io::Result<()> {
        let file = std::fs::OpenOptions::new().write(true).open(&self.path)?;
        file.sync_all()
    }
}

// ─── MmapWriter ─────────────────────────────────────────────────────────────

/// A RAII guard for writing via mmap with pre-allocated space.
///
/// Writes through MmapMut (via DerefMut), flush via fsync on the file.
#[derive(Debug)]
pub struct MmapWriter {
    file: std::fs::File,
    mmap: Option<MmapMut>,
    path: PathBuf,
    data_len: usize,
}

impl MmapWriter {
    /// Create a new file with `size` pre-allocated (truncates existing file).
    #[track_caller]
    pub fn create(path: &Path, size: u64) -> Result<Self> {
        let file = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .read(true)
            .truncate(true)
            .open(path)
            .map_err(RockDuckError::Io)?;
        file.set_len(size).map_err(RockDuckError::Io)?;
        let mmap = unsafe { MmapMut::map_mut(&file) }.map_err(RockDuckError::Io)?;
        Ok(Self {
            file,
            mmap: Some(mmap),
            path: path.to_path_buf(),
            data_len: 0,
        })
    }

    fn mmap(&self) -> &MmapMut {
        self.mmap
            .as_ref()
            .expect("MmapWriter: mmap not available (already finalized)")
    }

    fn mmap_mut(&mut self) -> &mut MmapMut {
        self.mmap
            .as_mut()
            .expect("MmapWriter: mmap not available (already finalized)")
    }

    /// Write data at the given offset.
    pub fn write_at(&mut self, offset: u64, data: &[u8]) -> Result<()> {
        let start = offset as usize;
        let end = start + data.len();
        if end > self.mmap().len() {
            return Err(RockDuckError::Internal(format!(
                "MmapWriter: write overflow: offset={}, len={}, mmap_len={}",
                offset,
                data.len(),
                self.mmap().len()
            )));
        }
        self.mmap_mut()[start..end].copy_from_slice(data);
        self.data_len = self.data_len.max(end);
        Ok(())
    }

    pub fn slice_at_mut(&mut self, offset: u64, len: u64) -> Result<&mut [u8]> {
        let start = offset as usize;
        let end = (start + len as usize).min(self.mmap().len());
        self.mmap_mut()
            .get_mut(start..end)
            .ok_or_else(|| RockDuckError::Internal("mmap slice out of bounds".into()))
    }

    pub fn data_len(&self) -> usize {
        self.data_len
    }

    pub fn finalize(mut self) -> Result<usize> {
        // Flush dirty pages to OS page cache before unmapping.
        self.flush().map_err(RockDuckError::Io)?;
        // Unmap the file view. On Windows, further file operations fail while
        // the file is memory-mapped (ERROR_INVALID_HANDLE / OS error 1224).
        self.mmap = None;
        // Truncate the file to the actual data length using a fresh handle.
        // This reflects the written data length in the file metadata.
        let file_path = self.path.clone();
        let file = std::fs::OpenOptions::new()
            .write(true)
            .open(&file_path)
            .map_err(RockDuckError::Io)?;
        file.set_len(self.data_len as u64)?;
        // Sync: ensure file size metadata and all data are durable.
        file.sync_all()?;
        Ok(self.data_len)
    }

    /// Flush modified pages to OS page cache.
    pub fn flush(&self) -> std::io::Result<()> {
        self.mmap().flush()
    }

    /// Durable flush: flush pages + fsync to physical disk.
    pub fn flush_to_disk(&self) -> std::io::Result<()> {
        self.mmap().flush()?;
        self.file.sync_all()
    }
}

impl std::ops::Deref for MmapWriter {
    type Target = MmapMut;
    fn deref(&self) -> &MmapMut {
        self.mmap
            .as_ref()
            .expect("MmapWriter: mmap not available (already finalized)")
    }
}

impl std::ops::DerefMut for MmapWriter {
    fn deref_mut(&mut self) -> &mut MmapMut {
        self.mmap
            .as_mut()
            .expect("MmapWriter: mmap not available (already finalized)")
    }
}

// ─── MmapReader ─────────────────────────────────────────────────────────────

#[derive(Debug)]
pub struct MmapReader {
    inner: Arc<MmapFile>,
}

impl MmapReader {
    /// Open an existing file as a read-only mmap.
    #[track_caller]
    pub fn open(path: &Path) -> Result<Self> {
        Ok(Self {
            inner: Arc::new(MmapFile::open_ro(path)?),
        })
    }

    /// Construct a MmapReader from an existing Arc<MmapFile> (used by FileHandle).
    pub fn from_mmap(mmap: Arc<MmapFile>) -> Self {
        Self { inner: mmap }
    }

    pub fn open_shared(path: &Path) -> Result<Arc<Self>> {
        Ok(Arc::new(Self::open(path)?))
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
    pub fn slice_at(&self, offset: u64, len: u64) -> Result<&[u8]> {
        self.inner.slice_at(offset, len)
    }
    pub fn chunks(&self, chunk_size: usize) -> impl Iterator<Item = &[u8]> {
        self.inner.chunks(chunk_size)
    }
    pub fn path(&self) -> &Path {
        self.inner.path()
    }
}

impl Clone for MmapReader {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}
