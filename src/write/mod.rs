//! Write module for RockDuck
//!
//! Provides write path functionality including:
//! - WAL durability
//! - Write buffering
//! - Transaction management
//! - Checkpoint and Group Commit

pub mod checkpoint;
pub mod durability_wal;
pub mod group_commit;
pub mod heat_tracker; // TRIAD: hot/cold key separation for WAL flush (VLDB 2022)
pub mod insert;
pub mod vis_file;
pub mod wal_recovery;
pub mod wal_utils;

// Re-export WAL types with a unified interface.
pub use durability_wal::{OpPayload, OpType, WalConfig, WalWriter};
pub use wal_recovery::{replay_committed_ops, RecoveryResult};
pub use wal_utils::{
    batch_to_bytes, batch_to_ipc_stream, batch_to_raw_bytes, bytes_to_batch, ipc_stream_to_batch,
    raw_bytes_to_batch,
};

pub use insert::*;
