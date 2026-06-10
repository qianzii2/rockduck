//! Storage module for RockDuck
//!
//! Provides storage layer abstractions for:
//! - Delta store (transactional updates with before-image)
//! - Vortex columnar storage (analytical reads)
//! - Segment file management

pub mod delta;
pub mod fastlanes; // FastLanes BitPacking/Delta/FoR encoders
pub mod file_manager;
pub mod granule_bloom_filter;
pub mod mmap_file; // Zero-copy mmap via memmap2 (cross-platform)
pub mod parquet_format; // TODO[PARQUET]: Parquet as export format
pub mod vortex;
pub mod vortex_alp_ext; // ALP + FastLanes Phase B // fastbloom Blocked(64) + auto-persist

pub use delta::*;
pub use file_manager::*;
pub use granule_bloom_filter::{AtomicGranuleBloomFilter, FastbloomGranuleFilter};
pub use mmap_file::{MmapFile, MmapReader};
pub use vortex::*;
