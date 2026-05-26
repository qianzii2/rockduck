//! DuckDB Extension integration for RockDuck.
//!
//! Registers these DuckDB functions:
//!   - `docdb_scan(path)`           — scan RockDuck, return table stats
//!   - `docdb_iceberg_info(path)`    — read Iceberg metadata.json, return table info
//!   - `docdb_iceberg_entries(path)` — list all Vortex data files in an Iceberg export
//!
//! DuckDB queries Vortex data directly (requires `LOAD vortex`):
//! ```sql
//! SELECT * FROM read_vortex('/path/to/exported/data/segments/*/*.vortex');
//! ```
//!
//! For streaming scan results from RockDuck to DuckDB, use:
//! ```sql
//! SELECT * FROM docdb_scan('/path/to/rockduck/data');
//! ```
//! This uses a custom VTab that streams RecordBatches in multiple batches,
//! avoiding the concat overhead of ArrowVTab.

use std::path::Path;
use duckdb::{Connection, Result as DuckDBResult};
use tracing::{debug, info};

use crate::db::RockDuck;
use crate::error::Result;

// ---------------------------------------------------------------------------
// DuckDBConnection
// ---------------------------------------------------------------------------

/// DuckDB connection wrapper that shares a RockDuck instance.
pub struct DuckDBConnection {
    conn: Connection,
    rockduck: std::sync::Arc<RockDuck>,
}

impl DuckDBConnection {
    /// Create a new DuckDB connection with shared RockDuck state.
    pub fn new(rockduck: std::sync::Arc<RockDuck>) -> Result<Self> {
        let conn = Connection::open_in_memory()
            .map_err(|e| crate::RockDuckError::DuckDB(e))?;
        Ok(Self { conn, rockduck })
    }

    /// Execute a SQL query and return rows.
    pub fn query(&self, sql: &str) -> DuckDBResult<Vec<Vec<duckdb::types::Value>>> {
        let mut results = Vec::new();
        let mut stmt = self.conn.prepare(sql)?;
        let col_count = stmt.column_count() as usize;
        let mut rows = stmt.query([])?;
        while let Some(row) = rows.next()? {
            let mut row_data = Vec::with_capacity(col_count);
            for i in 0..col_count {
                row_data.push(row.get_unwrap(i));
            }
            results.push(row_data);
        }
        Ok(results)
    }

    /// Register a DuckDB table that scans a RockDuck path.
    pub fn register_docdb_function(&self, name: &str, docdb_path: &str) -> DuckDBResult<()> {
        let sql = format!(
            "CREATE TABLE {} AS SELECT * FROM docdb_scan('{}')",
            name, docdb_path
        );
        self.conn.execute(&sql, [])?;
        Ok(())
    }

    /// Get a reference to the underlying RockDuck instance.
    pub fn rockduck(&self) -> &RockDuck {
        &self.rockduck
    }
}

// ---------------------------------------------------------------------------
// Implementation functions (used by DuckDB table functions / direct Rust calls)
// ---------------------------------------------------------------------------

/// Scan a RockDuck database and return table statistics.
///
/// Returns one row with the alive row count.
pub fn docdb_scan_impl(path: &str) -> Result<Vec<Vec<duckdb::types::Value>>> {
    debug!("docdb_scan: path={}", path);

    let rockduck = RockDuck::open(path)?;
    let table = "default";
    let stats = rockduck.get_table_stats(table)?;
    let alive_rows = stats.as_ref().map(|s| s.alive_rows()).unwrap_or(0);

    debug!("docdb_scan: found {} alive rows", alive_rows);
    Ok(vec![vec![duckdb::types::Value::UBigInt(alive_rows)]])
}

/// Read an Iceberg TableMetadata JSON and return table info as rows.
///
/// Usage:
/// ```sql
/// SELECT * FROM docdb_iceberg_info('/path/to/metadata/v1.metadata.json');
/// ```
///
/// Returns: format_version, table_uuid, location, last_sequence_number,
///          last_updated_ms, current_snapshot_id, schema_id, sort_order_id
pub fn docdb_iceberg_info_impl(path: &str) -> Result<Vec<Vec<duckdb::types::Value>>> {
    debug!("docdb_iceberg_info: path={}", path);

    let json_str = std::fs::read_to_string(path)
        .map_err(|e| crate::RockDuckError::Io(e))?;
    let meta: serde_json::Value = serde_json::from_str(&json_str)?;

    let format_version = meta["format-version"].as_i64().unwrap_or(2);
    let table_uuid = meta["table-uuid"].as_str().unwrap_or("").to_string();
    let location = meta["location"].as_str().unwrap_or("").to_string();
    let last_seq = meta["last-sequence-number"].as_i64().unwrap_or(0);
    let last_updated = meta["last-updated-ms"].as_i64().unwrap_or(0);
    let schema_id = meta["current-schema-id"].as_i64().unwrap_or(0);
    let sort_order_id = meta["current-sort-order-id"].as_i64().unwrap_or(1);
    let snapshot_id = meta["refs"]["main"]["snapshot-id"].as_i64()
        .unwrap_or(0);

    let rows = vec![vec![
        duckdb::types::Value::BigInt(format_version),
        duckdb::types::Value::Text(table_uuid),
        duckdb::types::Value::Text(location),
        duckdb::types::Value::BigInt(last_seq),
        duckdb::types::Value::BigInt(last_updated),
        duckdb::types::Value::BigInt(snapshot_id),
        duckdb::types::Value::BigInt(schema_id),
        duckdb::types::Value::BigInt(sort_order_id),
    ]];

    debug!("docdb_iceberg_info: snapshot_id={}", snapshot_id);
    Ok(rows)
}

/// List all Vortex data files in an exported Iceberg table directory.
///
/// Usage:
/// ```sql
/// SELECT * FROM docdb_iceberg_entries('/path/to/exported/data');
/// ```
///
/// Returns: file_path, file_format, record_count, file_size
pub fn docdb_iceberg_entries_impl(data_dir: &str) -> Result<Vec<Vec<duckdb::types::Value>>> {
    debug!("docdb_iceberg_entries: data_dir={}", data_dir);

    let base_path = Path::new(data_dir);
    let segments_dir = base_path.join("segments");
    if !segments_dir.exists() {
        return Ok(Vec::new());
    }

    let mut all_rows: Vec<Vec<duckdb::types::Value>> = Vec::new();

    for seg_entry in std::fs::read_dir(&segments_dir)? {
        let seg_entry = seg_entry?;
        let seg_path = seg_entry.path();
        if !seg_path.is_dir() {
            continue;
        }

        let seg_id = seg_path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("");

        for col_entry in std::fs::read_dir(&seg_path)? {
            let col_entry = col_entry?;
            let col_path = col_entry.path();

            let file_name = col_path.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("");
            // Skip internal files (_del.vortex, _upd_*.vortex, _meta.vortex)
            if !file_name.ends_with(".vortex") || file_name.starts_with('_') {
                continue;
            }

            let file_size = std::fs::metadata(&col_path)
                .map(|m| m.len() as i64)
                .unwrap_or(0);

            let file_path = format!("segments/{}/{}", seg_id, file_name);

            all_rows.push(vec![
                duckdb::types::Value::Text(file_path),
                duckdb::types::Value::Text("VORTEX".to_string()),
                duckdb::types::Value::BigInt(0), // record_count — requires reading file
                duckdb::types::Value::BigInt(file_size),
            ]);
        }
    }

    debug!("docdb_iceberg_entries: found {} files", all_rows.len());
    Ok(all_rows)
}

// ---------------------------------------------------------------------------
// DuckDB extension registration
// ---------------------------------------------------------------------------

/// Register RockDuck table functions in a DuckDB connection.
pub fn register_extension(conn: &Connection, _rockduck: &RockDuck) -> DuckDBResult<()> {
    info!("Registering RockDuck DuckDB extension");

    // Register the streaming VTab (multi-batch, no concat overhead).
    conn.register_table_function::<crate::query::vtab::RockDuckVTab>("docdb_scan")?;

    info!("RockDuck DuckDB extension registered");
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_docdb_scan_impl() {
        let temp_dir = tempfile::tempdir().unwrap();
        let results = docdb_scan_impl(temp_dir.path().to_str().unwrap()).unwrap();
        assert!(!results.is_empty());
    }

    #[test]
    fn test_docdb_iceberg_entries_empty() {
        let temp_dir = tempfile::tempdir().unwrap();
        let result = docdb_iceberg_entries_impl(temp_dir.path().to_str().unwrap()).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_docdb_iceberg_entries_with_vortex_file() {
        let temp_dir = tempfile::tempdir().unwrap();
        let seg_dir = temp_dir.path().join("segments").join("seg_abc");
        std::fs::create_dir_all(&seg_dir).unwrap();
        std::fs::write(seg_dir.join("id.vortex"), b"ARROW1\0").unwrap();

        let result = docdb_iceberg_entries_impl(temp_dir.path().to_str().unwrap()).unwrap();
        assert_eq!(result.len(), 1);
        assert!(matches!(result[0][0], duckdb::types::Value::Text(_)));
    }

    #[test]
    fn test_docdb_iceberg_entries_skips_internal_files() {
        let temp_dir = tempfile::tempdir().unwrap();
        let seg_dir = temp_dir.path().join("segments").join("seg_x");
        std::fs::create_dir_all(&seg_dir).unwrap();
        std::fs::write(seg_dir.join("id.vortex"), b"data").unwrap();
        std::fs::write(seg_dir.join("_del.vortex"), b"del").unwrap();
        std::fs::write(seg_dir.join("_upd_age.vortex"), b"upd").unwrap();

        let result = docdb_iceberg_entries_impl(temp_dir.path().to_str().unwrap()).unwrap();
        assert_eq!(result.len(), 1);
        assert!(matches!(result[0][0], duckdb::types::Value::Text(_)));
    }

    #[test]
    fn test_docdb_iceberg_info_nonexistent() {
        let result = docdb_iceberg_info_impl("/nonexistent/v1.metadata.json");
        assert!(result.is_err());
    }

    #[test]
    fn test_duckdb_connection_new() {
        let temp_dir = tempfile::tempdir().unwrap();
        let db = crate::RockDuck::open(temp_dir.path()).unwrap();
        let conn = DuckDBConnection::new(std::sync::Arc::new(db));
        assert!(conn.is_ok());
    }

    #[test]
    fn test_duckdb_connection_rockduck_reference() {
        let temp_dir = tempfile::tempdir().unwrap();
        let db = crate::RockDuck::open(temp_dir.path()).unwrap();
        let conn = DuckDBConnection::new(std::sync::Arc::new(db)).unwrap();
        let _rd = conn.rockduck();
    }
}
