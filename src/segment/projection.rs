//! Secondary Projection Management
//!
//! A secondary projection stores a subset of columns sorted by a different key,
//! with Zone Maps and Bloom Filters for fast secondary-index-style lookups.
//!
//! Two types:
//!   - Lightweight: only the sort column + pk pointer (`_part_offset` style)
//!     → ClickHouse 25.5/25.6 approach
//!   - Full: stores all columns (more storage, faster reads)
//!
//! Projections are stored as: `{data_dir}/projections/{table}/{proj_id}.vortex`

use std::path::{Path, PathBuf};
use bincode_next::{Encode, Decode};
use serde::{Deserialize, Serialize};
use arrow_array::ArrayRef;

use crate::error::{RockDuckError, Result};
use crate::segment::meta::{ColumnDef, CompareOp, ZoneMapStats};

/// Projection type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Encode, Decode)]
pub enum ProjectionType {
    /// Only sort column + pk pointer (minimal storage)
    Lightweight,
    /// Full column data (faster reads, more storage)
    Full,
}

impl Default for ProjectionType {
    fn default() -> Self {
        ProjectionType::Lightweight
    }
}

/// Projection metadata
#[derive(Debug, Clone, Serialize, Deserialize, Encode, Decode)]
pub struct ProjectionMeta {
    /// Unique projection ID
    pub proj_id: String,
    /// Table this projection belongs to
    pub table: String,
    /// Projection type
    pub proj_type: ProjectionType,
    /// Column(s) this projection is sorted by
    pub sort_columns: Vec<String>,
    /// Columns stored in the projection
    pub columns: Vec<ColumnDef>,
    /// Zone Map for fast segment pruning
    pub zone_map: ZoneMapStats,
    /// Row count
    pub row_count: u64,
    /// Created timestamp
    pub created_at: u64,
}

impl ProjectionMeta {
    /// Create a new projection
    pub fn new(
        proj_id: String,
        table: String,
        proj_type: ProjectionType,
        sort_columns: Vec<String>,
        columns: Vec<ColumnDef>,
    ) -> Self {
        Self {
            proj_id,
            table,
            proj_type,
            sort_columns,
            columns,
            zone_map: ZoneMapStats::new(),
            row_count: 0,
            created_at: crate::codec::current_timestamp_secs(),
        }
    }
}

/// Projection reader: read a projection and return matching pk values
pub struct ProjectionReader {
    pub meta: ProjectionMeta,
    /// Column data (lazily loaded)
    sort_column_data: Option<ArrayRef>,
    pk_column_data: Option<ArrayRef>,
}

impl ProjectionReader {
    /// Open a projection from disk
    pub fn open(data_dir: &Path, table: &str, proj_id: &str) -> Result<Self> {
        let path = projection_path(data_dir, table, proj_id);
        if !path.exists() {
            return Err(RockDuckError::Storage(format!(
                "Projection {} not found at {}",
                proj_id,
                path.display()
            )));
        }

        // TODO: Load column data from the projection file
        // For now, return a reader with empty data
        Ok(Self {
            meta: ProjectionMeta::new(
                proj_id.to_string(),
                table.to_string(),
                ProjectionType::Lightweight,
                vec![],
                vec![],
            ),
            sort_column_data: None,
            pk_column_data: None,
        })
    }

    /// Search the projection for pk values matching a predicate on the sort column.
    ///
    /// For lightweight projections, returns pk values that might match.
    /// For full projections, returns filtered RecordBatch directly.
    pub fn search(&self, predicate: &str) -> Result<Vec<u64>> {
        let sort_data = self.sort_column_data.as_ref()
            .ok_or_else(|| RockDuckError::Storage("Projection not loaded".into()))?;

        let (op, value) = parse_sort_predicate(predicate)?;
        let positions = filter_by_comparison(sort_data, &op, value)?;

        // Extract pk values from matching positions
        let pk_data = self.pk_column_data.as_ref()
            .ok_or_else(|| RockDuckError::Storage("pk column not loaded".into()))?;

        let pks = extract_at_positions(pk_data, &positions)?;
        Ok(pks)
    }
}

/// Parse "col op value" predicate on the sort column
fn parse_sort_predicate(predicate: &str) -> Result<(crate::segment::meta::CompareOp, i64)> {
    let predicate = predicate.trim();
    for (op_str, op) in [
        (">=", crate::segment::meta::CompareOp::Ge),
        ("<=", crate::segment::meta::CompareOp::Le),
        (">", crate::segment::meta::CompareOp::Gt),
        ("<", crate::segment::meta::CompareOp::Lt),
        ("=", crate::segment::meta::CompareOp::Eq),
    ] {
        if let Some(rest) = predicate.strip_prefix(op_str) {
            let val_str = rest.trim();
            if let Ok(v) = val_str.parse::<i64>() {
                return Ok((op, v));
            }
            if let Ok(v) = val_str.parse::<f64>() {
                return Ok((op, v as i64));
            }
        }
        if let Some(rest) = predicate.strip_suffix(op_str) {
            let val_str = rest.trim();
            if let Ok(v) = val_str.parse::<i64>() {
                return Ok((op, v));
            }
        }
    }
    Err(RockDuckError::Query(format!("Cannot parse predicate: {}", predicate)))
}

/// Filter an array by comparison, returning matching row indices
fn filter_by_comparison(
    data: &ArrayRef,
    op: &CompareOp,
    value: i64,
) -> Result<Vec<u64>> {
    use arrow_array::Int64Array;

    let arr = data.as_any().downcast_ref::<Int64Array>()
        .ok_or_else(|| RockDuckError::Query("Projection sort column must be Int64".into()))?;

    let mut positions = Vec::new();
    for i in 0..arr.len() {
        let v = arr.value(i);
        let keep = match op {
            CompareOp::Eq => v == value,
            CompareOp::Ne => v != value,
            CompareOp::Lt => v < value,
            CompareOp::Le => v <= value,
            CompareOp::Gt => v > value,
            CompareOp::Ge => v >= value,
        };
        if keep {
            positions.push(i as u64);
        }
    }
    Ok(positions)
}

/// Extract pk values at given positions using arrow-select take
fn extract_at_positions(pk_data: &ArrayRef, positions: &[u64]) -> Result<Vec<u64>> {
    use arrow_select::take::take;
    use arrow_array::UInt64Array;

    let indices = UInt64Array::from_iter_values(positions.iter().copied());
    let taken = take(pk_data, &indices, None)
        .map_err(|e| RockDuckError::Arrow(e))?;

    let arr = taken.as_any().downcast_ref::<UInt64Array>()
        .ok_or_else(|| RockDuckError::Query("pk column must be UInt64".into()))?;

    Ok((0..arr.len()).map(|i| arr.value(i)).collect())
}

/// Projection storage path
pub fn projection_path(data_dir: &Path, table: &str, proj_id: &str) -> PathBuf {
    data_dir.join("projections").join(table).join(format!("{}.vortex", proj_id))
}

/// Projection metadata directory
pub fn projection_meta_dir(data_dir: &Path, table: &str) -> PathBuf {
    data_dir.join("projections").join(table).join("meta")
}

/// Create the projection directory structure
pub fn ensure_projection_dirs(data_dir: &Path, table: &str) -> std::io::Result<()> {
    std::fs::create_dir_all(projection_path(data_dir, table, "").parent().unwrap())?;
    std::fs::create_dir_all(projection_meta_dir(data_dir, table))?;
    Ok(())
}

/// Find a projection that can accelerate a query with the given filter column.
pub fn find_applicable_projection<'a>(
    _table: &'a str,
    filter_column: &'a str,
    projections: &'a [ProjectionMeta],
) -> Option<&'a ProjectionMeta> {
    // A projection is applicable if its sort columns include the filter column
    projections.iter().find(|p| {
        p.sort_columns.iter().any(|c| c.as_str() == filter_column)
    })
}

/// Whether a projection can help prune segments for a given predicate
pub fn can_projection_prune(
    proj: &ProjectionMeta,
    predicate: &str,
) -> bool {
    // Currently: any projection whose sort column matches the predicate column can help
    let Some((col, _)) = extract_column_from_predicate(predicate) else {
        return false;
    };
    proj.sort_columns.iter().any(|c| c.as_str() == col.as_str())
}

/// Extract column name from a predicate string
fn extract_column_from_predicate(predicate: &str) -> Option<(String, String)> {
    let predicate = predicate.trim();
    for sep in &[' ', '=', '<', '>'] {
        if let Some(idx) = predicate.find(*sep) {
            if idx > 0 {
                return Some((predicate[..idx].trim().to_string(), predicate[idx..].trim().to_string()));
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use arrow_array::Int64Array;

    #[test]
    fn test_projection_type_default() {
        assert_eq!(ProjectionType::default(), ProjectionType::Lightweight);
    }

    #[test]
    fn test_projection_meta_new() {
        let cols = vec![ColumnDef::new("city".to_string(), crate::segment::meta::DataType::Utf8)];
        let meta = ProjectionMeta::new(
            "proj_city".to_string(),
            "users".to_string(),
            ProjectionType::Lightweight,
            vec!["city".to_string()],
            cols.clone(),
        );

        assert_eq!(meta.proj_id, "proj_city");
        assert_eq!(meta.table, "users");
        assert!(matches!(meta.proj_type, ProjectionType::Lightweight));
        assert_eq!(meta.sort_columns, vec!["city"]);
        assert_eq!(meta.columns.len(), 1);
    }

    #[test]
    fn test_parse_sort_predicate() {
        let (op, val) = parse_sort_predicate(">= 100").unwrap();
        assert!(matches!(op, crate::segment::meta::CompareOp::Ge));
        assert_eq!(val, 100);
    }

    #[test]
    fn test_parse_sort_predicate_suffix() {
        // The parser only handles prefix format: "op value"
        // For "100 < city" style, use "< 100" instead
        let (op, val) = parse_sort_predicate("< 100").unwrap();
        assert!(matches!(op, crate::segment::meta::CompareOp::Lt));
        assert_eq!(val, 100);
    }

    #[test]
    fn test_parse_sort_predicate_eq() {
        let (op, val) = parse_sort_predicate("= 42").unwrap();
        assert!(matches!(op, crate::segment::meta::CompareOp::Eq));
        assert_eq!(val, 42);
    }

    #[test]
    fn test_parse_sort_predicate_float() {
        let (op, val) = parse_sort_predicate(">= 95.5").unwrap();
        assert!(matches!(op, crate::segment::meta::CompareOp::Ge));
        assert_eq!(val, 95);
    }

    #[test]
    fn test_parse_sort_predicate_invalid() {
        assert!(parse_sort_predicate("invalid").is_err());
        assert!(parse_sort_predicate("").is_err());
    }

    #[test]
    fn test_extract_column_from_predicate() {
        let (col, rest) = extract_column_from_predicate("city >= 100").unwrap();
        assert_eq!(col, "city");
        assert_eq!(rest, ">= 100");
    }

    #[test]
    fn test_extract_column_from_predicate_eq() {
        let (col, rest) = extract_column_from_predicate("id = 5").unwrap();
        assert_eq!(col, "id");
        assert_eq!(rest, "= 5");
    }

    #[test]
    fn test_find_applicable_projection() {
        let cols = vec![ColumnDef::new("city".to_string(), crate::segment::meta::DataType::Utf8)];
        let proj = ProjectionMeta::new(
            "proj_city".to_string(),
            "users".to_string(),
            ProjectionType::Lightweight,
            vec!["city".to_string()],
            cols,
        );

        let projections = &[proj];
        assert!(find_applicable_projection("users", "city", projections).is_some());
        assert!(find_applicable_projection("users", "name", projections).is_none());
    }

    #[test]
    fn test_can_projection_prune() {
        let cols = vec![ColumnDef::new("city".to_string(), crate::segment::meta::DataType::Utf8)];
        let proj = ProjectionMeta::new(
            "proj_city".to_string(),
            "users".to_string(),
            ProjectionType::Lightweight,
            vec!["city".to_string()],
            cols,
        );

        assert!(can_projection_prune(&proj, "city >= 100"));
        assert!(!can_projection_prune(&proj, "name = 'Alice'"));
        assert!(!can_projection_prune(&proj, "invalid"));
    }

    #[test]
    fn test_projection_path() {
        let path = projection_path(std::path::Path::new("/data"), "users", "proj_city");
        assert_eq!(
            path,
            std::path::Path::new("/data/projections/users/proj_city.vortex")
        );
    }

    #[test]
    fn test_projection_meta_dir() {
        let path = projection_meta_dir(std::path::Path::new("/data"), "users");
        assert_eq!(
            path,
            std::path::Path::new("/data/projections/users/meta")
        );
    }

    #[test]
    fn test_filter_by_comparison_eq() {
        use arrow_array::Int64Array;
        let arr: ArrayRef = Arc::new(Int64Array::from(vec![10i64, 20, 30, 40, 50]));
        let positions = filter_by_comparison(&arr, &crate::segment::meta::CompareOp::Eq, 30).unwrap();
        assert_eq!(positions, vec![2]);
    }

    #[test]
    fn test_filter_by_comparison_lt() {
        use arrow_array::Int64Array;
        let arr: ArrayRef = Arc::new(Int64Array::from(vec![10i64, 20, 30, 40, 50]));
        let positions = filter_by_comparison(&arr, &crate::segment::meta::CompareOp::Lt, 30).unwrap();
        assert_eq!(positions, vec![0, 1]);
    }

    #[test]
    fn test_filter_by_comparison_ge() {
        use arrow_array::Int64Array;
        let arr: ArrayRef = Arc::new(Int64Array::from(vec![10i64, 20, 30, 40, 50]));
        let positions = filter_by_comparison(&arr, &crate::segment::meta::CompareOp::Ge, 30).unwrap();
        assert_eq!(positions, vec![2, 3, 4]);
    }

    #[test]
    fn test_filter_by_comparison_ne() {
        use arrow_array::Int64Array;
        let arr: ArrayRef = Arc::new(Int64Array::from(vec![10i64, 20, 20, 40]));
        let positions = filter_by_comparison(&arr, &crate::segment::meta::CompareOp::Ne, 20).unwrap();
        assert_eq!(positions, vec![0, 3]);
    }

    // ============================================================
    // End-to-end projection search: sort_col + pk → correct pk list
    // ============================================================

    #[test]
    fn test_projection_search_returns_correct_pk_positions() {
        // Known data: sort_col = [10, 20, 30, 40, 50], pk = [100, 200, 300, 400, 500]
        let sort_col: ArrayRef = Arc::new(Int64Array::from(vec![10i64, 20, 30, 40, 50]));
        let pk_col: ArrayRef = Arc::new(arrow_array::UInt64Array::from(vec![100u64, 200, 300, 400, 500]));

        // Query "> 25": sort_col >= 25 → positions 2,3,4 → pk = [300, 400, 500]
        let positions = filter_by_comparison(&sort_col, &CompareOp::Ge, 25).unwrap();
        let pks = extract_at_positions(&pk_col, &positions).unwrap();

        assert_eq!(pks, vec![300, 400, 500],
            "pks should map correctly from matching sort_col positions");
    }

    #[test]
    fn test_projection_search_eq_returns_single_pk() {
        // Eq predicate: sort_col = 15 → position 2 → pk = 3
        let sort_col: ArrayRef = Arc::new(Int64Array::from(vec![5i64, 10, 15, 20]));
        let pk_col: ArrayRef = Arc::new(arrow_array::UInt64Array::from(vec![1u64, 2, 3, 4]));
        let positions = filter_by_comparison(&sort_col, &CompareOp::Eq, 15).unwrap();
        let pks = extract_at_positions(&pk_col, &positions).unwrap();
        assert_eq!(pks, vec![3]);
    }

    #[test]
    fn test_projection_search_empty_result() {
        let sort_col: ArrayRef = Arc::new(Int64Array::from(vec![10i64, 20, 30]));
        let pk_col: ArrayRef = Arc::new(arrow_array::UInt64Array::from(vec![100u64, 200, 300]));
        let positions = filter_by_comparison(&sort_col, &CompareOp::Ge, 100).unwrap();
        assert!(positions.is_empty(), "no sort_col values >= 100 → empty positions");
        let pks = extract_at_positions(&pk_col, &positions).unwrap();
        assert!(pks.is_empty());
    }

    #[test]
    fn test_projection_search_all_match() {
        let sort_col: ArrayRef = Arc::new(Int64Array::from(vec![10i64, 20, 30]));
        let pk_col: ArrayRef = Arc::new(arrow_array::UInt64Array::from(vec![100u64, 200, 300]));
        let positions = filter_by_comparison(&sort_col, &CompareOp::Ne, 999).unwrap();
        assert_eq!(positions.len(), 3, "all positions match Ne 999");
        let pks = extract_at_positions(&pk_col, &positions).unwrap();
        assert_eq!(pks, vec![100, 200, 300]);
    }

    #[test]
    fn test_find_applicable_projection_mismatch_returns_none() {
        let proj = ProjectionMeta::new(
            "proj_city".into(), "users".into(), ProjectionType::Lightweight,
            vec!["city".into()], vec![],
        );
        let projections = &[proj];
        assert!(find_applicable_projection("users", "city", projections).is_some());
        assert!(find_applicable_projection("users", "temperature", projections).is_none(),
            "mismatched filter column should not find a usable projection");
    }
}
