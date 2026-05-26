//! Write 模块
//!
//! 插入、删除、更新操作（都集成在 insert.rs 中）
//! WAL 模块提供崩溃安全的写入保证。

pub mod insert;
pub mod wal;
