//! Integration tests for RockDuckVTab via DuckDB SQL.
//!
//! These tests verify the VTab produces correct results from DuckDB's perspective.
//! The assertions focus on WHAT is returned (row counts, column values) rather than
//! HOW it is streamed (batch boundaries, internal state).
//!
//! Key constraint: RocksDB on Windows does not allow two open handles to the same DB.
//! The RockDuck instance that holds the RocksDB lock MUST be dropped before DuckDB queries.

use std::collections::HashMap;
use std::sync::Arc;
use tempfile::tempdir;

use arrow_array::{ArrayRef, Float64Array, Int64Array, StringArray};
use rockduck::{Result, RockDuck, RockDuckError};

use rockduck::query::duckdb_ext::register_extension;

// ============================================================
// Test Utilities
// ============================================================

fn make_data(id: i64, name: &str, value: f64) -> HashMap<String, ArrayRef> {
    let mut data = HashMap::new();
    data.insert(
        "id".to_string(),
        Arc::new(Int64Array::from(vec![id])) as ArrayRef,
    );
    data.insert(
        "name".to_string(),
        Arc::new(StringArray::from(vec![name])) as ArrayRef,
    );
    data.insert(
        "value".to_string(),
        Arc::new(Float64Array::from(vec![value])) as ArrayRef,
    );
    data
}

// ============================================================
// Normal Path: Data is returned correctly
// ============================================================

/// VTab returns exactly the rows that were inserted.
#[test]
fn test_vtab_returns_inserted_rows() -> Result<()> {
    let tmp = tempdir()?;
    let path = tmp.path().to_str().unwrap().to_string();

    // Write data with RockDuck, then DROP the instance.
    {
        let db = RockDuck::open(tmp.path())?;
        db.insert("default", b"pk1", &make_data(10, "Alice", 1.5))?;
        db.insert("default", b"pk2", &make_data(20, "Bob", 2.5))?;
    }

    // DuckDB queries — must open its own RockDuck handle.
    {
        let db = Arc::new(RockDuck::open(&path)?);
        let conn = duckdb::Connection::open_in_memory()?;
        register_extension(&conn, &db)?;
        drop(db);

        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM docdb_scan($1)",
                [&path],
                |row| row.get(0),
            )
            .map_err(RockDuckError::DuckDB)?;
        assert_eq!(count, 2, "VTab must return exactly 2 rows for 2 inserted records");
    }

    Ok(())
}

/// VTab returns correct column values (type mapping correctness).
#[test]
fn test_vtab_type_mapping() -> Result<()> {
    let tmp = tempdir()?;
    let path = tmp.path().to_str().unwrap().to_string();

    {
        let db = RockDuck::open(tmp.path())?;
        db.insert("default", b"pk1", &make_data(42, "Alice", 99.5))?;
    }

    {
        let db = Arc::new(RockDuck::open(&path)?);
        let conn = duckdb::Connection::open_in_memory()?;
        register_extension(&conn, &db)?;
        drop(db);

        let (id, name, value): (i64, String, f64) = conn
            .query_row(
                "SELECT * FROM docdb_scan($1)",
                [&path],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .map_err(RockDuckError::DuckDB)?;

        assert_eq!(id, 42, "id column must return inserted Int64 value");
        assert_eq!(name, "Alice", "name column must return inserted String value");
        assert!(
            (value - 99.5).abs() < 1e-6,
            "value column must return inserted Float64 value"
        );
    }

    Ok(())
}

// ============================================================
// Edge Case: Empty table
// ============================================================

/// VTab returns zero rows when the table is empty.
#[test]
fn test_vtab_empty_table() -> Result<()> {
    let tmp = tempdir()?;
    let path = tmp.path().to_str().unwrap().to_string();

    // Create empty DB.
    {
        let _db = RockDuck::open(tmp.path())?;
    }

    {
        let db = Arc::new(RockDuck::open(&path)?);
        let conn = duckdb::Connection::open_in_memory()?;
        register_extension(&conn, &db)?;
        drop(db);

        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM docdb_scan($1)",
                [&path],
                |row| row.get(0),
            )
            .map_err(RockDuckError::DuckDB)?;
        assert_eq!(count, 0, "VTab must return 0 rows for empty table");
    }

    Ok(())
}

// ============================================================
// Edge Case: Single row
// ============================================================

/// VTab works correctly with exactly one row.
#[test]
fn test_vtab_single_row() -> Result<()> {
    let tmp = tempdir()?;
    let path = tmp.path().to_str().unwrap().to_string();

    {
        let db = RockDuck::open(tmp.path())?;
        db.insert("default", b"pk1", &make_data(1, "Solo", 3.14))?;
    }

    {
        let db = Arc::new(RockDuck::open(&path)?);
        let conn = duckdb::Connection::open_in_memory()?;
        register_extension(&conn, &db)?;
        drop(db);

        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM docdb_scan($1)",
                [&path],
                |row| row.get(0),
            )
            .map_err(RockDuckError::DuckDB)?;
        assert_eq!(count, 1, "VTab must return exactly 1 row");

        let (id,): (i64,) = conn
            .query_row(
                "SELECT id FROM docdb_scan($1)",
                [&path],
                |row| Ok((row.get(0)?,)),
            )
            .map_err(RockDuckError::DuckDB)?;
        assert_eq!(id, 1, "id must be 1");
    }

    Ok(())
}

// ============================================================
// Edge Case: Each query is independent
// ============================================================

/// Each query to docdb_scan opens and scans independently.
#[test]
fn test_vtab_each_query_is_independent() -> Result<()> {
    let tmp = tempdir()?;
    let path = tmp.path().to_str().unwrap().to_string();

    {
        let db = RockDuck::open(tmp.path())?;
        db.insert("default", b"pk1", &make_data(1, "A", 1.0))?;
    }

    {
        let db = Arc::new(RockDuck::open(&path)?);
        let conn = duckdb::Connection::open_in_memory()?;
        register_extension(&conn, &db)?;
        drop(db);

        let count1: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM docdb_scan($1)",
                [&path],
                |row| row.get(0),
            )
            .map_err(RockDuckError::DuckDB)?;
        assert_eq!(count1, 1);

        let count2: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM docdb_scan($1)",
                [&path],
                |row| row.get(0),
            )
            .map_err(RockDuckError::DuckDB)?;
        assert_eq!(count2, 1, "Each query must independently scan and return data");
    }

    Ok(())
}

// ============================================================
// Edge Case: Non-existent table in VTab path
// ============================================================

/// VTab handles a path with no RockDuck data gracefully (returns empty).
#[test]
fn test_vtab_nonexistent_path_returns_empty() -> Result<()> {
    let tmp = tempdir()?;
    let path = tmp.path().to_str().unwrap().to_string();

    {
        let _db = RockDuck::open(tmp.path())?;
    }

    {
        let db = Arc::new(RockDuck::open(&path)?);
        let conn = duckdb::Connection::open_in_memory()?;
        register_extension(&conn, &db)?;
        drop(db);

        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM docdb_scan($1)",
                [&path],
                |row| row.get(0),
            )
            .map_err(RockDuckError::DuckDB)?;
        assert_eq!(count, 0, "VTab must return 0 rows for table with no inserts");
    }

    Ok(())
}

// ============================================================
// Edge Case: Deleted rows excluded from VTab output
// ============================================================

/// Deleted records must not appear in VTab output.
#[test]
fn test_vtab_excludes_deleted_rows() -> Result<()> {
    let tmp = tempdir()?;
    let path = tmp.path().to_str().unwrap().to_string();

    {
        let db = RockDuck::open(tmp.path())?;
        db.insert("default", b"pk1", &make_data(1, "Keep", 1.0))?;
        db.insert("default", b"pk2", &make_data(2, "Delete", 2.0))?;
        db.insert("default", b"pk3", &make_data(3, "AlsoKeep", 3.0))?;
        db.delete("default", b"pk2")?;
    }

    {
        let db = Arc::new(RockDuck::open(&path)?);
        let conn = duckdb::Connection::open_in_memory()?;
        register_extension(&conn, &db)?;
        drop(db);

        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM docdb_scan($1)",
                [&path],
                |row| row.get(0),
            )
            .map_err(RockDuckError::DuckDB)?;
        assert_eq!(count, 2, "Deleted row must not appear in VTab output");

        let mut stmt = conn
            .prepare("SELECT id FROM docdb_scan($1) ORDER BY id")
            .map_err(RockDuckError::DuckDB)?;
        let mut rows = stmt.query([&path]).map_err(RockDuckError::DuckDB)?;

        let mut ids: Vec<i64> = Vec::new();
        while let Some(row) = rows.next().map_err(RockDuckError::DuckDB)? {
            ids.push(row.get(0).map_err(RockDuckError::DuckDB)?);
        }
        assert_eq!(ids, vec![1, 3], "Only non-deleted rows should be present");
    }

    Ok(())
}

// ============================================================
// Data correctness: Multiple rows with mixed data types
// ============================================================

/// VTab preserves data across all columns for multiple rows.
#[test]
fn test_vtab_multiple_rows_data_integrity() -> Result<()> {
    let tmp = tempdir()?;
    let path = tmp.path().to_str().unwrap().to_string();

    {
        let db = RockDuck::open(tmp.path())?;
        let rows = vec![
            (100, "row_one", 10.5),
            (200, "row_two", 20.5),
            (300, "row_three", 30.5),
        ];
        for (id, name, value) in rows {
            db.insert(
                "default",
                format!("pk_{}", id).as_bytes(),
                &make_data(id, name, value),
            )?;
        }
    }

    {
        let db = Arc::new(RockDuck::open(&path)?);
        let conn = duckdb::Connection::open_in_memory()?;
        register_extension(&conn, &db)?;
        drop(db);

        let mut stmt = conn
            .prepare("SELECT id, name, value FROM docdb_scan($1) ORDER BY id")
            .map_err(RockDuckError::DuckDB)?;
        let mut duckdb_rows = stmt.query([&path]).map_err(RockDuckError::DuckDB)?;

        let expected = vec![
            (100, "row_one", 10.5),
            (200, "row_two", 20.5),
            (300, "row_three", 30.5),
        ];

        let mut seen: Vec<(i64, String, f64)> = Vec::new();
        while let Some(row) = duckdb_rows.next().map_err(RockDuckError::DuckDB)? {
            seen.push((
                row.get(0).map_err(RockDuckError::DuckDB)?,
                row.get(1).map_err(RockDuckError::DuckDB)?,
                row.get(2).map_err(RockDuckError::DuckDB)?,
            ));
        }

        assert_eq!(seen.len(), 3, "Must return exactly 3 rows");
        for (i, (row, exp)) in seen.iter().zip(expected.iter()).enumerate() {
            assert_eq!(row.0, exp.0, "Row {}: id mismatch", i);
            assert_eq!(row.1, exp.1, "Row {}: name mismatch", i);
            assert!(
                (row.2 - exp.2).abs() < 1e-6,
                "Row {}: value mismatch — got {}, want {}",
                i, row.2, exp.2
            );
        }
    }

    Ok(())
}
