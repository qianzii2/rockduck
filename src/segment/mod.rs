//! Segment module for RockDuck
//!
//! Provides segment layout, metadata, and delete mask operations.

pub mod adaptive_del_mask;
pub mod layout;
pub mod meta;
pub mod overlay;
pub mod sparse_index;
mod upd_mask; // Used only within segment layout — internal implementation detail

pub use adaptive_del_mask::*;
pub use layout::*;
pub use meta::*;
pub use overlay::*;
pub use sparse_index::*;
