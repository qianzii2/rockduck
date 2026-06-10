//! RockDuck -- HTAP Embedded Database
//!
//! Core components:
//! - Delta Store for transactional updates with before-image
//! - Vortex columnar storage for analytical reads
//! - DuckDB extension for SQL execution
//! - MVCC visibility management
//! - Compaction with PDT merge
//! - Iceberg table format export
//! - CDC change data capture

pub mod cdc;
pub mod codec;
pub mod compaction;
pub mod config;
pub mod db;
pub mod error;
pub mod metadata;
pub mod mvcc;
pub mod query;
pub mod read;
pub mod segment;
pub mod storage;
pub mod write;

// =============================================================================
// Iceberg export (ICE-1/2/3/4)
// Iceberg spec violations found: Vortex as Parquet, empty manifests, hardcoded seq numbers.
// This module is disabled by default. Enable via `iceberg_export` feature.
// =============================================================================
#[cfg(feature = "iceberg_export")]
pub mod iceberg;

pub mod typed_arrow_records;

// Re-exports
pub use config::RockDuckConfig;
pub use db::RockDuck;
pub use db::RockDuck as DocDB;
pub use error::{DocDBError, Result, RockDuckError};
