//! RockDuck - 嵌入式列存数据库
//!
//! 基于 RocksDB（元数据）+ Vortex（列式存储）+ DuckDB（查询执行）
//! 核心思想来自 Iceberg Delete Files / ClickHouse MergeTree / Snowflake Zone Map

pub mod error;
pub mod codec;
pub mod config;
pub mod db;

pub mod segment;
pub mod storage;
pub mod metadata;
pub mod read;
pub mod write;
pub mod compaction;
pub mod query;
pub mod mvcc;
pub mod typed_arrow_records;
pub mod iceberg;

pub use error::{RockDuckError, Result};
pub use config::RockDuckConfig;
pub use db::RockDuck;
