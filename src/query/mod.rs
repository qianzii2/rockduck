//! Query module for RockDuck
//!
//! Provides query execution functionality including:
//! - DuckDB extension integration
//! - Query routing (cost-model-driven HTAP read path selection)
//! - Time travel queries
//! - Virtual table support
//! - Query feedback collection

pub mod duckdb_ext;
pub mod feedback;
pub mod filter_expr;
pub mod routing; // NEW: unified three-tier router (Tier1 rule / Tier2 cost / Tier3 ML)
pub mod time_travel;
pub mod time_travel_impl;
pub mod vtab_quack;

pub use duckdb_ext::*;
pub use feedback::*;
pub use routing::*; // Re-exports QueryRouter, RouteDecision, RoutingResult, QueryKind, RouterParams, TableStats
pub use time_travel::*;
pub use time_travel_impl::{
    build_segment_version_index, TimeTravelReader, TimeTravelScanner, VersionEntry,
};
