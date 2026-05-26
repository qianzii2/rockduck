//! MVCC 模块
//!
//! 提供多版本并发控制（MVCC）能力：
//! - 活跃事务追踪（Shadow Column 方式）
//! - 可见性判断（Read Committed / Repeatable Read / Snapshot Isolation）
//! - 快照生成

pub mod visibility;

pub use visibility::{VisibilityManager, TxnSnapshot, IsolationLevel};
