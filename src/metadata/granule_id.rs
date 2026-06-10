//! GranuleId re-export from segment::meta.
//!
//! GranuleId is defined in `segment::meta` to avoid circular dependencies
//! (segment::meta is used by metadata, but metadata also needs GranuleId).
//! This module re-exports it for ergonomic access from the metadata subtree.

pub use crate::segment::meta::GranuleId;
