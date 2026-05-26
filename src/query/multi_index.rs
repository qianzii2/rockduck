//! Multi-index AND intersection for accelerated query planning.
//!
//! When a query has multiple indexable predicates (e.g., `age > 30 AND score > 50`),
//! this module uses Zone Maps + Bloom Filters to find matching row IDs and
//! intersect them with RoaringBitmap, avoiding full table scans.
//!
//! Architecture:
//!   filter AST (Expr)
//!     → extract_indexable_preds()  (pulls out all Comparison nodes from AND)
//!     → multi_index_intersection()  (evaluates each predicate, returns intersection)
//!     → read_records_by_row_ids()   (batch reads matching rows)

use roaring::RoaringBitmap;
use crate::error::Result;
use crate::metadata::rocksdb;
use crate::segment::meta::CompareOp;
use crate::query::filter_expr::{Expr, ScalarVal};

/// Pruning strategy for a predicate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PruningType {
    /// Lt / Le / Gt / Ge — Zone Map range pruning
    ZoneMapRange,
    /// Eq — Bloom Filter fast negative check
    BloomFilter,
    /// Secondary index (Projection) — sorted column lookup
    SecondaryIndex,
}

/// Result of evaluating a single predicate with index acceleration.
#[derive(Debug)]
#[allow(dead_code)]
struct PredicateResult {
    column: String,
    row_ids: RoaringBitmap,
    pruning_type: PruningType,
}

/// Extract all indexable Comparison predicates from an AND expression.
///
/// Example: `age > 30 AND score > 50 AND name = 'Alice'`
/// Returns all three Comparison nodes (but NOT the AND itself).
///
/// Returns empty for OR / NOT expressions (not purely indexable).
pub fn extract_indexable_preds<'a>(expr: &'a Expr) -> Vec<&'a Expr> {
    match expr {
        Expr::And(left, right) => {
            let mut preds = extract_indexable_preds(left);
            preds.extend(extract_indexable_preds(right));
            preds
        }
        Expr::Comparison { .. } => vec![expr],
        Expr::Or(..) | Expr::Not(..) => vec![],
    }
}

/// Check if a predicate can benefit from Zone Map pruning.
fn is_zone_map_prunable(op: &CompareOp) -> bool {
    matches!(op, CompareOp::Lt | CompareOp::Le | CompareOp::Gt | CompareOp::Ge)
}

/// Check if a predicate can benefit from Bloom Filter (Eq only).
fn is_bloom_prunable(op: &CompareOp) -> bool {
    matches!(op, CompareOp::Eq)
}

/// Multi-index AND intersection.
///
/// Takes a list of predicates (all AND-connected) and evaluates each one
/// using the best available index. Returns the intersection of all matching
/// row ID sets, or `None` if no index acceleration is available.
pub fn multi_index_intersection(
    db: &crate::RockDuck,
    table: &str,
    preds: &[&Expr],
) -> Result<Option<RoaringBitmap>> {
    if preds.is_empty() {
        return Ok(None);
    }

    let mut results = Vec::with_capacity(preds.len());

    for pred in preds {
        if let Some(result) = evaluate_predicate(db, table, pred)? {
            results.push(result);
        } else {
            // Predicate not indexable — fall back to full scan
            return Ok(None);
        }
    }

    // Intersection of all predicate results
    let intersection = results
        .iter()
        .fold(None, |acc: Option<RoaringBitmap>, r| {
            match acc {
                None => Some(r.row_ids.clone()),
                Some(cur) => Some(&cur & &r.row_ids),
            }
        });

    Ok(intersection)
}

/// Evaluate a single predicate using the best available index.
/// Returns `None` if no index is available (full scan needed).
fn evaluate_predicate(
    db: &crate::RockDuck,
    table: &str,
    pred: &Expr,
) -> Result<Option<PredicateResult>> {
    let Expr::Comparison { col, op, val } = pred else {
        return Ok(None);
    };

    // Try Secondary Projection first (sorted column → most selective)
    if let Some(pks) = try_projection_lookup(db, table, col, op, val)? {
        // Convert pk values to row IDs via segment metadata
        let row_ids = resolve_pks_to_row_ids(db, table, col, op, val, &pks)?;
        return Ok(Some(PredicateResult {
            column: col.clone(),
            row_ids,
            pruning_type: PruningType::SecondaryIndex,
        }));
    }

    // Try Zone Map pruning for range predicates
    if is_zone_map_prunable(op) {
        if let Some(row_ids) = prune_by_zone_map(db, table, col, op, val)? {
            return Ok(Some(PredicateResult {
                column: col.clone(),
                row_ids,
                pruning_type: PruningType::ZoneMapRange,
            }));
        }
    }

    // Try Bloom Filter for Eq predicates
    if is_bloom_prunable(op) {
        if let Some(row_ids) = prune_by_bloom_filter(db, table, col, val)? {
            return Ok(Some(PredicateResult {
                column: col.clone(),
                row_ids,
                pruning_type: PruningType::BloomFilter,
            }));
        }
    }

    Ok(None)
}

/// Try to accelerate a predicate using a Secondary Projection.
fn try_projection_lookup(
    _db: &crate::RockDuck,
    _table: &str,
    _col: &str,
    _op: &CompareOp,
    _val: &ScalarVal,
) -> Result<Option<Vec<u64>>> {
    // TODO: Implement ProjectionReader loading and search
    // Once ProjectionReader::open() is implemented, use it here to:
    // 1. Find a projection sorted by `col`
    // 2. Binary search the sort column for matching positions
    // 3. Return pk values at those positions
    Ok(None)
}

/// Prune segments using Zone Map statistics at the granule level.
///
/// Returns a RoaringBitmap of row IDs that might satisfy the predicate.
/// (False positives are possible; actual filtering happens in scan.)
fn prune_by_zone_map(
    db: &crate::RockDuck,
    table: &str,
    col: &str,
    op: &CompareOp,
    val: &ScalarVal,
) -> Result<Option<RoaringBitmap>> {
    let val_bytes = scalar_to_bytes(val)?;
    let seg_ids = get_table_segments(db, table)?;

    let mut matching_rows = RoaringBitmap::new();

    for seg_id in seg_ids {
        let meta = match rocksdb::get_segment_meta(&db.db, &seg_id)? {
            Some(m) => m,
            None => continue,
        };

        for granule in &meta.granules {
            if granule.zone_map.can_prune(col, op, &val_bytes) {
                // Zone Map says no values in this granule match → skip entire granule
                continue;
            }
            // Granule may contain matching rows → add all row IDs
            for i in 0..granule.row_count {
                let row_id = granule.row_offset + i as u64;
                matching_rows.insert(row_id as u32);
            }
        }
    }

    if matching_rows.is_empty() {
        Ok(None)
    } else {
        Ok(Some(matching_rows))
    }
}

/// Prune using Bloom Filter for Eq predicates.
/// Returns matching row IDs (Bloom Filter can have false positives).
fn prune_by_bloom_filter(
    db: &crate::RockDuck,
    _table: &str,
    _col: &str,
    val: &ScalarVal,
) -> Result<Option<RoaringBitmap>> {
    let pk_bytes = scalar_to_bytes(val)?;
    let seg_ids = get_table_segments(db, _table)?;

    let mut matching_rows = RoaringBitmap::new();

    for seg_id in &seg_ids {
        let bfs = db.segment_bloom_filters.read();
        if let Some(bf) = bfs.get(seg_id) {
            if crate::read::point_get::bloom_contains(bf, &pk_bytes) {
                // Bloom Filter says pk might be in this segment
                // Add all row IDs from this segment (conservative)
                if let Some(meta) = rocksdb::get_segment_meta(&db.db, seg_id)? {
                    for granule in &meta.granules {
                        for i in 0..granule.row_count {
                            let row_id = granule.row_offset + i as u64;
                            matching_rows.insert(row_id as u32);
                        }
                    }
                }
            }
            // Bloom Filter says definitely not present → skip segment
        } else {
            // No Bloom Filter → conservatively include all rows
            if let Some(meta) = rocksdb::get_segment_meta(&db.db, seg_id)? {
                for granule in &meta.granules {
                    for i in 0..granule.row_count {
                        let row_id = granule.row_offset + i as u64;
                        matching_rows.insert(row_id as u32);
                    }
                }
            }
        }
    }

    if matching_rows.is_empty() {
        Ok(None)
    } else {
        Ok(Some(matching_rows))
    }
}

/// Convert a ScalarVal to bytes for Bloom Filter lookup.
fn scalar_to_bytes(val: &ScalarVal) -> Result<Vec<u8>> {
    match val {
        ScalarVal::Int64(n) => Ok(n.to_le_bytes().to_vec()),
        ScalarVal::String(s) => Ok(s.as_bytes().to_vec()),
        ScalarVal::Bool(b) => Ok(vec![if *b { 1 } else { 0 }]),
        _ => Err(crate::RockDuckError::Query(
            "Bloom filter only supports Int64/String/Bool values".into(),
        )),
    }
}

/// Get all segment IDs for a table.
fn get_table_segments(db: &crate::RockDuck, table: &str) -> Result<Vec<String>> {
    let cf = db.db.cf_handle("seg_meta")
        .ok_or_else(|| crate::RockDuckError::Metadata("seg_meta column family not found".to_string()))?;

    let prefix = format!("seg:{}:", table);
    let prefix_bytes = prefix.as_bytes();
    let prefix_len = prefix_bytes.len();
    let mut seg_ids = Vec::new();

    let mut iter = db.db.raw_iterator_cf(&cf);
    iter.seek(prefix_bytes);

    while iter.valid() {
        if let Some(key) = iter.key() {
            if !key.starts_with(prefix_bytes) {
                break;
            }
            let seg_id_bytes = &key[prefix_len..];
            seg_ids.push(String::from_utf8_lossy(seg_id_bytes).into_owned());
        }
        iter.next();
    }

    Ok(seg_ids)
}

/// Resolve pk values to row IDs using segment metadata.
fn resolve_pks_to_row_ids(
    _db: &crate::RockDuck,
    _table: &str,
    _col: &str,
    _op: &CompareOp,
    _val: &ScalarVal,
    _pks: &[u64],
) -> Result<RoaringBitmap> {
    // TODO: Use RocksDB pk_idx to look up each pk → IndexEntry → granule row offset
    // For now, return empty (projection lookup is not yet implemented)
    Ok(RoaringBitmap::new())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_indexable_preds_single() {
        let expr = Expr::Comparison {
            col: "age".into(),
            op: CompareOp::Gt,
            val: ScalarVal::Int64(30),
        };
        let preds = extract_indexable_preds(&expr);
        assert_eq!(preds.len(), 1);
    }

    #[test]
    fn test_extract_indexable_preds_nested_and() {
        // age > 30 AND score > 50 AND city = 'NYC'
        let expr = Expr::And(
            Box::new(Expr::And(
                Box::new(Expr::Comparison {
                    col: "age".into(),
                    op: CompareOp::Gt,
                    val: ScalarVal::Int64(30),
                }),
                Box::new(Expr::Comparison {
                    col: "score".into(),
                    op: CompareOp::Gt,
                    val: ScalarVal::Int64(50),
                }),
            )),
            Box::new(Expr::Comparison {
                col: "city".into(),
                op: CompareOp::Eq,
                val: ScalarVal::String("NYC".into()),
            }),
        );
        let preds = extract_indexable_preds(&expr);
        assert_eq!(preds.len(), 3);
    }

    #[test]
    fn test_extract_indexable_preds_or_returns_empty() {
        let expr = Expr::Or(
            Box::new(Expr::Comparison {
                col: "age".into(),
                op: CompareOp::Gt,
                val: ScalarVal::Int64(30),
            }),
            Box::new(Expr::Comparison {
                col: "score".into(),
                op: CompareOp::Gt,
                val: ScalarVal::Int64(50),
            }),
        );
        let preds = extract_indexable_preds(&expr);
        assert!(preds.is_empty());
    }

    #[test]
    fn test_extract_indexable_preds_not_returns_empty() {
        let expr = Expr::Not(Box::new(Expr::Comparison {
            col: "age".into(),
            op: CompareOp::Eq,
            val: ScalarVal::Int64(30),
        }));
        let preds = extract_indexable_preds(&expr);
        assert!(preds.is_empty());
    }

    #[test]
    fn test_is_zone_map_prunable() {
        assert!(is_zone_map_prunable(&CompareOp::Lt));
        assert!(is_zone_map_prunable(&CompareOp::Le));
        assert!(is_zone_map_prunable(&CompareOp::Gt));
        assert!(is_zone_map_prunable(&CompareOp::Ge));
        assert!(!is_zone_map_prunable(&CompareOp::Eq));
        assert!(!is_zone_map_prunable(&CompareOp::Ne));
    }

    #[test]
    fn test_is_bloom_prunable() {
        assert!(is_bloom_prunable(&CompareOp::Eq));
        assert!(!is_bloom_prunable(&CompareOp::Ne));
        assert!(!is_bloom_prunable(&CompareOp::Lt));
    }

    #[test]
    fn test_scalar_to_bytes_int64() {
        let val = ScalarVal::Int64(42);
        let bytes = scalar_to_bytes(&val).unwrap();
        assert_eq!(bytes, 42i64.to_le_bytes());
    }

    #[test]
    fn test_scalar_to_bytes_string() {
        let val = ScalarVal::String("hello".into());
        let bytes = scalar_to_bytes(&val).unwrap();
        assert_eq!(bytes, b"hello".to_vec());
    }

    // ============================================================
    // extract_indexable_preds strict assertion tests
    // ============================================================

    #[test]
    fn test_extract_indexable_preds_deeply_nested_and() {
        // age > 30 AND score > 50 AND city = 'NYC' AND status != 'X'
        let expr = Expr::And(
            Box::new(Expr::And(
                Box::new(Expr::Comparison { col: "age".into(), op: CompareOp::Gt, val: ScalarVal::Int64(30) }),
                Box::new(Expr::Comparison { col: "score".into(), op: CompareOp::Gt, val: ScalarVal::Int64(50) }),
            )),
            Box::new(Expr::And(
                Box::new(Expr::Comparison { col: "city".into(), op: CompareOp::Eq, val: ScalarVal::String("NYC".into()) }),
                Box::new(Expr::Comparison { col: "status".into(), op: CompareOp::Ne, val: ScalarVal::String("X".into()) }),
            )),
        );
        let preds = extract_indexable_preds(&expr);
        // Must extract exactly 4 predicates (neither more nor less)
        assert_eq!(preds.len(), 4);

        // Verify each predicate's column name and operator
        assert!(matches!(&preds[0], Expr::Comparison { col, op: CompareOp::Gt, .. } if col == "age"));
        assert!(matches!(&preds[1], Expr::Comparison { col, op: CompareOp::Gt, .. } if col == "score"));
        assert!(matches!(&preds[2], Expr::Comparison { col, op: CompareOp::Eq, .. } if col == "city"));
        assert!(matches!(&preds[3], Expr::Comparison { col, op: CompareOp::Ne, .. } if col == "status"));
    }

    #[test]
    fn test_extract_indexable_preds_mixed_and_or_returns_empty() {
        // (a AND b) OR (c AND d) — OR is not purely indexable
        let expr = Expr::Or(
            Box::new(Expr::And(
                Box::new(Expr::Comparison { col: "a".into(), op: CompareOp::Eq, val: ScalarVal::Int64(1) }),
                Box::new(Expr::Comparison { col: "b".into(), op: CompareOp::Eq, val: ScalarVal::Int64(2) }),
            )),
            Box::new(Expr::And(
                Box::new(Expr::Comparison { col: "c".into(), op: CompareOp::Eq, val: ScalarVal::Int64(3) }),
                Box::new(Expr::Comparison { col: "d".into(), op: CompareOp::Eq, val: ScalarVal::Int64(4) }),
            )),
        );
        let preds = extract_indexable_preds(&expr);
        assert!(preds.is_empty(), "OR expressions must return empty — they cannot be purely indexed");
    }

    #[test]
    fn test_extract_indexable_preds_single_comparison() {
        let expr = Expr::Comparison { col: "x".into(), op: CompareOp::Eq, val: ScalarVal::Int64(1) };
        let preds = extract_indexable_preds(&expr);
        assert_eq!(preds.len(), 1);
        assert!(matches!(&preds[0], Expr::Comparison { col, .. } if col == "x"));
    }

    #[test]
    fn test_extract_indexable_preds_three_level_nesting() {
        // Single nested: A AND (B AND C)
        let expr = Expr::And(
            Box::new(Expr::Comparison { col: "a".into(), op: CompareOp::Eq, val: ScalarVal::Int64(1) }),
            Box::new(Expr::And(
                Box::new(Expr::Comparison { col: "b".into(), op: CompareOp::Eq, val: ScalarVal::Int64(2) }),
                Box::new(Expr::Comparison { col: "c".into(), op: CompareOp::Eq, val: ScalarVal::Int64(3) }),
            )),
        );
        let preds = extract_indexable_preds(&expr);
        assert_eq!(preds.len(), 3);
        assert!(matches!(&preds[0], Expr::Comparison { col, .. } if col == "a"));
        assert!(matches!(&preds[1], Expr::Comparison { col, .. } if col == "b"));
        assert!(matches!(&preds[2], Expr::Comparison { col, .. } if col == "c"));
    }

    // ============================================================
    // Boundary and exception tests
    // ============================================================

    #[test]
    fn test_scalar_to_bytes_bool() {
        assert_eq!(scalar_to_bytes(&ScalarVal::Bool(true)).unwrap(), vec![1u8]);
        assert_eq!(scalar_to_bytes(&ScalarVal::Bool(false)).unwrap(), vec![0u8]);
    }

    #[test]
    fn test_scalar_to_bytes_null_returns_err() {
        assert!(scalar_to_bytes(&ScalarVal::Null).is_err());
    }

    #[test]
    fn test_scalar_to_bytes_float_returns_err() {
        // Float/Double cannot be converted to Bloom Filter bytes
        assert!(scalar_to_bytes(&ScalarVal::Float64(3.14)).is_err());
    }

    #[test]
    fn test_pruning_helpers_all_ops() {
        // Zone Map prunable: Lt Le Gt Ge
        assert!(is_zone_map_prunable(&CompareOp::Lt));
        assert!(is_zone_map_prunable(&CompareOp::Le));
        assert!(is_zone_map_prunable(&CompareOp::Gt));
        assert!(is_zone_map_prunable(&CompareOp::Ge));
        // Zone Map NOT prunable: Eq Ne
        assert!(!is_zone_map_prunable(&CompareOp::Eq));
        assert!(!is_zone_map_prunable(&CompareOp::Ne));

        // Bloom prunable: Eq only
        assert!(is_bloom_prunable(&CompareOp::Eq));
        // Bloom NOT prunable: all others
        assert!(!is_bloom_prunable(&CompareOp::Ne));
        assert!(!is_bloom_prunable(&CompareOp::Lt));
        assert!(!is_bloom_prunable(&CompareOp::Le));
        assert!(!is_bloom_prunable(&CompareOp::Gt));
        assert!(!is_bloom_prunable(&CompareOp::Ge));
    }
}
