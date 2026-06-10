//! Segment aliasing — transparent PK lookup redirect after compaction.
//!
//! When a segment is compacted via PDT merge, its old seg_id becomes stale.
//! PK lookups for rows in the compacted segment must be redirected to the new seg_id.
//!
//! ## Alias chain
//!
//! After compaction:
//! 1. Data is written to a new segment with a new seg_id
//! 2. A `seg_alias:<old_seg_id>` entry is written to the KV store, pointing to the new seg_id
//! 3. All subsequent PK lookups follow the alias transparently
//!
//! ## PK lookup flow
//!
//! ```text
//! get_pk_index_by_pk(pk)
//!   -> resolve_seg_id(kv, old_seg_id)
//!     -> resolve_segment_alias(kv, old_seg_id)
//!       -> kv.get("seg_alias:old_seg_id") -> Some(new_seg_id)
//!   -> return (new_seg_id, granule_id, row_offset)
//! ```
//!
//! ## Cleanup
//!
//! Alias entries are not automatically cleaned up. Old alias entries pointing to
//! compacted-away segments can accumulate over time. The cleanup strategy is:
//! - After a successful compaction, the old segment's data directory is deleted
//! - The alias entry remains (harmless, just points to a non-existent old seg)
//! - A background task can periodically purge alias entries whose target segments no longer exist

use crate::error::Result;
use crate::metadata::kv_engine::KVEngine;
use crate::metadata::kv_store;
use std::sync::Arc;

/// Resolve a segment ID, following the alias chain if the segment was compacted.
///
/// Returns `(final_seg_id, is_redirected)`:
///   - `is_redirected = true` if an alias was found and followed
///   - `is_redirected = false` if no alias exists (seg_id is current)
pub fn resolve_seg_id(kv: &dyn KVEngine, seg_id: &str) -> Result<(String, bool)> {
    kv_store::resolve_seg_id(kv, seg_id)
}

/// Write a segment alias after successful compaction.
///
/// Call this AFTER the compacted segment data is fully written and durable.
/// On failure, the alias is not written — callers must handle partial state.
pub fn write_alias(kv: &Arc<dyn KVEngine>, old_seg_id: &str, new_seg_id: &str) -> Result<()> {
    kv_store::put_segment_alias(kv, old_seg_id, new_seg_id)
}

/// Check if a segment ID has been redirected by an alias.
pub fn is_redirected(kv: &dyn KVEngine, seg_id: &str) -> Result<bool> {
    Ok(kv_store::resolve_segment_alias(kv, seg_id)?.is_some())
}
