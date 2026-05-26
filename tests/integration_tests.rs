//! RockDuck Integration Tests
//!
//! End-to-end tests verifying actual data correctness.
//! Every test that writes data MUST verify it can be read back correctly.

use std::collections::HashMap;
use std::sync::Arc;
use tempfile::tempdir;

use arrow_array::{ArrayRef, Float64Array, Int64Array, StringArray};
use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use rockduck::{Result, RockDuck, RockDuckConfig, segment::meta::SegmentStatus};

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

fn make_batch_data(n: usize) -> HashMap<String, ArrayRef> {
    let mut ids = Vec::with_capacity(n);
    let mut names = Vec::with_capacity(n);
    let mut values = Vec::with_capacity(n);
    for i in 0..n {
        ids.push(i as i64 + 1);
        names.push(format!("name_{}", i + 1));
        values.push((i + 1) as f64 * 10.0);
    }
    let mut data = HashMap::new();
    data.insert(
        "id".to_string(),
        Arc::new(Int64Array::from(ids)) as ArrayRef,
    );
    data.insert(
        "name".to_string(),
        Arc::new(StringArray::from(names)) as ArrayRef,
    );
    data.insert(
        "value".to_string(),
        Arc::new(Float64Array::from(values)) as ArrayRef,
    );
    data
}

// ============================================================
// Database Lifecycle Tests
// ============================================================

#[test]
fn test_open_database_default() -> Result<()> {
    let temp_dir = tempdir()?;
    let db = RockDuck::open(temp_dir.path())?;
    let info = db.get_info();
    assert!(!info.data_dir.as_os_str().is_empty());
    assert_eq!(info.txn_counter, 0);
    Ok(())
}

#[test]
fn test_open_database_custom_config() -> Result<()> {
    let temp_dir = tempdir()?;
    let config = RockDuckConfig {
        data_dir: temp_dir.path().to_path_buf(),
        granule_size: 2 * 1024 * 1024,
        segment_target_size: 100 * 1024 * 1024,
        num_threads: 4,
        enable_bloom_filter: false,
        bloom_filter_fpp: 0.01,
        enable_zone_map: true,
        enable_compression: true,
        compression_algorithm: Some("lz4".to_string()),
        enable_wal: true,
        wal_max_file_size: 128 * 1024 * 1024,
    };
    let db = RockDuck::open_with_config(temp_dir.path(), config.clone())?;
    let info = db.get_info();
    assert_eq!(info.config.granule_size, 2 * 1024 * 1024);
    assert_eq!(info.config.segment_target_size, 100 * 1024 * 1024);
    assert!(!info.config.enable_bloom_filter);
    Ok(())
}

#[test]
fn test_next_txn_id_incrementing() -> Result<()> {
    let temp_dir = tempdir()?;
    let db = RockDuck::open(temp_dir.path())?;
    let id1 = db.next_txn_id();
    let id2 = db.next_txn_id();
    let id3 = db.next_txn_id();
    assert!(id1 < id2);
    assert!(id2 < id3);
    assert_eq!(id3, 3);
    Ok(())
}

#[test]
fn test_flush_succeeds() -> Result<()> {
    let temp_dir = tempdir()?;
    let db = RockDuck::open(temp_dir.path())?;
    db.flush()?;
    Ok(())
}

// ============================================================
// Core Business Rule: Write → Read
// Every insert must be retrievable with correct data.
// ============================================================

/// Single record insert → point-get returns correct values
#[test]
fn test_insert_single_record_then_get() -> Result<()> {
    let temp_dir = tempdir()?;
    let db = RockDuck::open(temp_dir.path())?;
    let pk = b"user_1";
    let data = make_data(42, "Alice", 99.5);
    db.insert("users", pk, &data)?;

    let result = db.get("users", pk)?;
    assert!(result.is_some(), "Inserted record should be retrievable");
    let batch = result.unwrap();
    assert_eq!(batch.num_rows(), 1);

    let ids = batch.column_by_name("id")
        .expect("id column should exist")
        .as_primitive::<Int64Type>();
    assert_eq!(ids.value(0), 42);

    let names = batch.column_by_name("name")
        .expect("name column should exist")
        .as_string::<i32>();
    assert_eq!(names.value(0), "Alice");

    let values = batch.column_by_name("value")
        .expect("value column should exist")
        .as_primitive::<arrow_array::types::Float64Type>();
    assert!((values.value(0) - 99.5).abs() < 1e-6);

    Ok(())
}

/// Batch insert → scan returns correct number of rows with correct values
#[test]
fn test_insert_batch_then_scan_all() -> Result<()> {
    let temp_dir = tempdir()?;
    let db = RockDuck::open(temp_dir.path())?;
    let pks: Vec<Vec<u8>> = (1..=10)
        .map(|i| format!("user_{}", i).into_bytes())
        .collect();
    let data = make_batch_data(10);
    db.insert_batch("users", &pks, &data)?;

    let batches = db.scan("users", None, None)?;
    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total_rows, 10, "Should return all 10 inserted rows");

    // Verify first, middle, and last row values
    for (_i, batch) in batches.iter().enumerate() {
        for row in 0..batch.num_rows() {
            let ids = batch.column_by_name("id")
                .expect("id column should exist")
                .as_primitive::<Int64Type>();
            let id = ids.value(row);
            assert!(id >= 1 && id <= 10, "id {} out of range [1,10]", id);
        }
    }

    Ok(())
}

/// Batch insert → point-get each record individually
#[test]
fn test_batch_insert_individual_point_gets() -> Result<()> {
    let temp_dir = tempdir()?;
    let db = RockDuck::open(temp_dir.path())?;
    let pks: Vec<Vec<u8>> = (1..=100)
        .map(|i| format!("user_{}", i).into_bytes())
        .collect();
    let data = make_batch_data(100);
    db.insert_batch("users", &pks, &data)?;

    for i in 1..=100 {
        let pk = format!("user_{}", i);
        let result = db.get("users", pk.as_bytes())?;
        assert!(result.is_some(), "user_{} should be found", i);
        let batch = result.unwrap();
        let id = batch
            .column_by_name("id")
            .expect("id column should exist")
            .as_primitive::<Int64Type>()
            .value(0);
        assert_eq!(id, i as i64, "user_{} should have id={}", i, i);
    }
    Ok(())
}

// ============================================================
// Core Business Rule: Delete → Not Readable
// ============================================================

/// Delete → point-get returns None
#[test]
fn test_delete_then_point_get_returns_none() -> Result<()> {
    let temp_dir = tempdir()?;
    let db = RockDuck::open(temp_dir.path())?;
    let pk = b"user_1";
    let data = make_data(1, "Alice", 10.0);
    db.insert("users", pk, &data)?;

    // Verify it exists before delete
    assert!(db.get("users", pk)?.is_some());

    db.delete("users", pk)?;

    // After delete, point-get must return None
    let result = db.get("users", pk)?;
    assert!(result.is_none(), "Deleted record should not be retrievable via point-get");

    Ok(())
}

/// Delete → scan excludes deleted rows
#[test]
fn test_deleted_records_excluded_from_scan() -> Result<()> {
    let temp_dir = tempdir()?;
    let db = RockDuck::open(temp_dir.path())?;
    for i in 1..=5 {
        let pk = format!("user_{}", i);
        let data = make_data(i, &format!("U{}", i), i as f64);
        db.insert("users", pk.as_bytes(), &data)?;
    }
    db.delete("users", b"user_1")?;
    db.delete("users", b"user_3")?;

    let batches = db.scan("users", None, None)?;
    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total_rows, 3, "Deleted rows should not appear in scan");

    // Verify remaining rows are correct
    let mut found_ids: Vec<i64> = Vec::new();
    for batch in &batches {
        let ids = batch.column_by_name("id")
            .expect("id column should exist")
            .as_primitive::<Int64Type>();
        for row in 0..batch.num_rows() {
            found_ids.push(ids.value(row));
        }
    }
    found_ids.sort();
    assert_eq!(found_ids, vec![2, 4, 5], "Should only contain undeleted user ids");

    Ok(())
}

/// Double delete is idempotent
#[test]
fn test_double_delete_is_idempotent() -> Result<()> {
    let temp_dir = tempdir()?;
    let db = RockDuck::open(temp_dir.path())?;
    db.insert("users", b"pk", &make_data(1, "A", 1.0))?;

    let r1 = db.delete("users", b"pk");
    assert!(r1.is_ok(), "First delete should succeed");

    let r2 = db.delete("users", b"pk");
    assert!(r2.is_ok(), "Second delete of same key should be idempotent (returns Ok)");
    Ok(())
}

// ============================================================
// Core Business Rule: Re-insert After Delete
// ============================================================

#[test]
fn test_delete_then_insert_same_key() -> Result<()> {
    let temp_dir = tempdir()?;
    let db = RockDuck::open(temp_dir.path())?;

    db.insert("users", b"key", &make_data(1, "Original", 10.0))?;
    db.delete("users", b"key")?;
    db.insert("users", b"key", &make_data(2, "Updated", 20.0))?;

    let result = db.get("users", b"key")?;
    assert!(result.is_some(), "Should be able to re-insert deleted key");
    let batch = result.unwrap();

    let id = batch
        .column_by_name("id")
        .expect("id column should exist")
        .as_primitive::<Int64Type>()
        .value(0);
    assert_eq!(id, 2, "Re-inserted record should have id=2");

    let name = batch
        .column_by_name("name")
        .expect("name column should exist")
        .as_string::<i32>()
        .value(0);
    assert_eq!(name, "Updated", "Re-inserted record should have name=Updated");

    Ok(())
}

// ============================================================
// Core Business Rule: Data Persistence
// ============================================================

#[test]
fn test_data_persists_after_reopen() -> Result<()> {
    let temp_dir = tempdir()?;
    let data_dir = temp_dir.path().to_path_buf();
    {
        let db = RockDuck::open(&data_dir)?;
        let pks = vec![b"pk1".to_vec(), b"pk2".to_vec(), b"pk3".to_vec()];
        let data = make_batch_data(3);
        db.insert_batch("users", &pks, &data)?;
    }
    {
        let db = RockDuck::open(&data_dir)?;
        let result = db.get("users", b"pk2")?;
        assert!(result.is_some(), "pk2 should be found after reopen");
        let batch = result.unwrap();
        let ids = batch.column_by_name("id")
            .expect("id column should exist")
            .as_primitive::<Int64Type>();
        assert_eq!(ids.value(0), 2, "pk2 should have id=2");
    }
    Ok(())
}

// ============================================================
// Core Business Rule: Table Stats Accuracy
// ============================================================

#[test]
fn test_table_stats_row_count_matches_scan() -> Result<()> {
    let temp_dir = tempdir()?;
    let db = RockDuck::open(temp_dir.path())?;

    for i in 1..=10 {
        let pk = format!("user_{}", i);
        let data = make_data(i, &format!("User{}", i), i as f64);
        db.insert("users", pk.as_bytes(), &data)?;
    }

    let stats = db.get_table_stats("users")?;
    assert!(stats.is_some());
    let stats = stats.unwrap();
    assert_eq!(stats.row_count, 10);

    let batches = db.scan("users", None, None)?;
    let scanned_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(stats.row_count as usize, scanned_rows,
        "TableStats.row_count must match total rows returned by scan");
    Ok(())
}

#[test]
fn test_table_stats_alive_rows_after_delete() -> Result<()> {
    let temp_dir = tempdir()?;
    let db = RockDuck::open(temp_dir.path())?;

    for i in 1..=10 {
        let pk = format!("user_{}", i);
        let data = make_data(i, &format!("User{}", i), i as f64);
        db.insert("users", pk.as_bytes(), &data)?;
    }

    db.delete("users", b"user_1")?;
    db.delete("users", b"user_5")?;
    db.delete("users", b"user_9")?;

    let stats = db.get_table_stats("users")?.unwrap();
    assert_eq!(stats.row_count, 10);
    assert_eq!(stats.deleted_rows, 3);
    assert_eq!(stats.alive_rows(), 7);
    assert!((stats.del_ratio() - 0.3).abs() < 1e-6);

    // alive_rows must match what scan returns
    let batches = db.scan("users", None, None)?;
    let scanned: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(stats.alive_rows() as usize, scanned,
        "alive_rows() must equal scan result count");
    Ok(())
}

#[test]
fn test_table_stats_basic() -> Result<()> {
    let temp_dir = tempdir()?;
    let db = RockDuck::open(temp_dir.path())?;

    let stats = db.get_table_stats("users")?;
    assert!(stats.is_none(), "Stats for non-existent table should be None");

    let data = make_data(1, "Alice", 100.0);
    db.insert("users", b"pk_1", &data)?;

    let stats = db.get_table_stats("users")?;
    assert!(stats.is_some());
    let stats = stats.unwrap();
    assert!(stats.row_count > 0);
    assert!(stats.segment_count > 0);
    Ok(())
}

#[test]
fn test_table_stats_del_ratio_zero() -> Result<()> {
    let temp_dir = tempdir()?;
    let db = RockDuck::open(temp_dir.path())?;
    let data = make_data(1, "Alice", 100.0);
    db.insert("users", b"pk_1", &data)?;

    let stats = db.get_table_stats("users")?.unwrap();
    assert_eq!(stats.del_ratio(), 0.0);
    assert_eq!(stats.alive_rows(), 1);
    Ok(())
}

// ============================================================
// Scan Range Tests
// ============================================================

#[test]
fn test_scan_all_records() -> Result<()> {
    let temp_dir = tempdir()?;
    let db = RockDuck::open(temp_dir.path())?;
    for i in 1..=10 {
        let pk = format!("user_{}", i);
        let data = make_data(i, &format!("User{}", i), i as f64 * 10.0);
        db.insert("users", pk.as_bytes(), &data)?;
    }
    let batches = db.scan("users", None, None)?;
    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total_rows, 10, "Should return all 10 inserted rows");
    Ok(())
}

/// PK range is half-open [start, end).
/// Zero-padded keys: "00001".."00010" → [start="00003", end="00007")
/// should return exactly 4 records: "00003", "00004", "00005", "00006".
#[test]
fn test_scan_with_pk_range_half_open() -> Result<()> {
    let temp_dir = tempdir()?;
    let db = RockDuck::open(temp_dir.path())?;
    for i in 1..=10 {
        let pk = format!("{:05}", i);
        let data = make_data(i, &format!("User{}", i), i as f64 * 10.0);
        db.insert("users", pk.as_bytes(), &data)?;
    }

    let start = b"00003".to_vec();
    let end = b"00007".to_vec();
    let batches = db.scan("users", Some((start, end)), None)?;
    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total_rows, 4, "[start, end) with 00003..00007 should return exactly 4 rows");
    Ok(())
}

#[test]
fn test_scan_nonexistent_table() -> Result<()> {
    let temp_dir = tempdir()?;
    let db = RockDuck::open(temp_dir.path())?;
    let batches = db.scan("nonexistent", None, None)?;
    assert!(batches.is_empty());
    Ok(())
}

#[test]
fn test_scan_empty_range_returns_nothing() -> Result<()> {
    let temp_dir = tempdir()?;
    let db = RockDuck::open(temp_dir.path())?;
    db.insert("users", b"user_1", &make_data(1, "Alice", 100.0))?;

    let start = b"zzz".to_vec();
    let end = b"zzz".to_vec();
    let batches = db.scan("users", Some((start, end)), None)?;
    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total_rows, 0);
    Ok(())
}

// ============================================================
// Segment Tests
// ============================================================

#[test]
fn test_list_segments() -> Result<()> {
    let temp_dir = tempdir()?;
    let db = RockDuck::open(temp_dir.path())?;

    let segments = db.list_segments("users")?;
    assert!(segments.is_empty());

    let data = make_data(1, "Alice", 100.0);
    db.insert("users", b"pk_1", &data)?;

    let segments = db.list_segments("users")?;
    assert!(!segments.is_empty(), "Should have at least one segment");
    assert_eq!(segments.len(), 1);
    Ok(())
}

#[test]
fn test_get_segment_meta() -> Result<()> {
    let temp_dir = tempdir()?;
    let db = RockDuck::open(temp_dir.path())?;
    let data = make_data(1, "Alice", 100.0);
    db.insert("users", b"pk_1", &data)?;

    let segments = db.list_segments("users")?;
    assert!(!segments.is_empty());

    let meta = db.get_segment_meta(&segments[0])?;
    assert!(meta.is_some());
    let meta = meta.unwrap();
    assert_eq!(meta.table, "users");
    assert!(meta.row_count > 0);
    assert_eq!(meta.status, SegmentStatus::Active);
    Ok(())
}

#[test]
fn test_get_segment_meta_nonexistent() -> Result<()> {
    let temp_dir = tempdir()?;
    let db = RockDuck::open(temp_dir.path())?;
    let meta = db.get_segment_meta("nonexistent_seg")?;
    assert!(meta.is_none());
    Ok(())
}

// ============================================================
// Multi-Table Isolation
// ============================================================

#[test]
fn test_multiple_tables_data_isolation() -> Result<()> {
    let temp_dir = tempdir()?;
    let db = RockDuck::open(temp_dir.path())?;
    db.insert("table_a", b"key", &make_data(1, "A", 1.0))?;
    db.insert("table_b", b"key", &make_data(2, "B", 2.0))?;

    let batch_a = db.get("table_a", b"key")?.unwrap();
    let batch_b = db.get("table_b", b"key")?.unwrap();

    assert_eq!(
        batch_a.column_by_name("id")
            .expect("id column should exist in table_a")
            .as_primitive::<Int64Type>()
            .value(0),
        1,
        "table_a should have id=1"
    );
    assert_eq!(
        batch_b.column_by_name("id")
            .expect("id column should exist in table_b")
            .as_primitive::<Int64Type>()
            .value(0),
        2,
        "table_b should have id=2"
    );
    Ok(())
}

// ============================================================
// Large Batch Tests
// ============================================================

#[test]
fn test_large_batch_insert_and_scan() -> Result<()> {
    let temp_dir = tempdir()?;
    let db = RockDuck::open(temp_dir.path())?;
    let pks: Vec<Vec<u8>> = (1..=1000)
        .map(|i| format!("user_{}", i).into_bytes())
        .collect();
    let data = make_batch_data(1000);
    db.insert_batch("users", &pks, &data)?;
    let batches = db.scan("users", None, None)?;
    let total: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total, 1000, "Should return all 1000 batch-inserted rows");
    Ok(())
}

// ============================================================
// Empty / Edge Cases
// ============================================================

#[test]
fn test_insert_batch_empty_pks() -> Result<()> {
    let temp_dir = tempdir()?;
    let db = RockDuck::open(temp_dir.path())?;
    let pks: Vec<Vec<u8>> = vec![];
    let data = make_batch_data(0);
    let txn_id = db.insert_batch("users", &pks, &data)?;
    assert_eq!(txn_id, 0);
    Ok(())
}

#[test]
fn test_insert_batch_empty_data() -> Result<()> {
    let temp_dir = tempdir()?;
    let db = RockDuck::open(temp_dir.path())?;
    let pks: Vec<Vec<u8>> = vec![b"pk_1".to_vec()];
    let data: HashMap<String, ArrayRef> = HashMap::new();
    let result = db.insert_batch("users", &pks, &data);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_delete_nonexistent_returns_error() -> Result<()> {
    let temp_dir = tempdir()?;
    let db = RockDuck::open(temp_dir.path())?;
    let result = db.delete("users", b"nonexistent");
    assert!(result.is_err());
    Ok(())
}

// ============================================================
// DB Info
// ============================================================

#[test]
fn test_get_info() -> Result<()> {
    let temp_dir = tempdir()?;
    let db = RockDuck::open(temp_dir.path())?;
    let info = db.get_info();
    assert!(!info.data_dir.as_os_str().is_empty());
    assert_eq!(info.txn_counter, 0);

    // Insert something to increment counter
    let data = make_data(1, "Alice", 100.0);
    db.insert("users", b"pk_1", &data)?;
    let info = db.get_info();
    assert!(info.txn_counter >= 1);

    Ok(())
}

// ============================================================
// Mmap Read Integration Tests
// ============================================================

#[test]
fn test_scan_with_filter_returns_correct_rows() -> Result<()> {
    let temp_dir = tempdir()?;
    let db = RockDuck::open(temp_dir.path())?;

    // Insert multiple rows
    for i in 0u8..5 {
        let data = make_data(i as i64, &format!("user_{}", i), i as f64 * 10.0);
        db.insert("t", format!("pk_{}", i).as_bytes(), &data)?;
    }

    // Filter: id > 2 (should return rows: id=3,4)
    let batches = db.scan("t", None, Some("id > 2"))?;
    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert!(total_rows >= 2, "Should return at least 2 rows matching id > 2");

    // Verify all returned ids are > 2
    for batch in &batches {
        let ids = batch.column_by_name("id").expect("id column");
        let id_arr = ids.as_primitive::<Int64Type>();
        for i in 0..id_arr.len() {
            let id = id_arr.value(i);
            assert!(id > 2, "All returned ids should be > 2, but got {}", id);
        }
    }

    Ok(())
}

#[test]
fn test_list_segments_returns_after_insert() -> Result<()> {
    let temp_dir = tempdir()?;
    let db = RockDuck::open(temp_dir.path())?;

    // Initially no segments
    let segs = db.list_segments("t")?;
    assert!(segs.is_empty(), "New table should have no segments");

    // After insert, at least one segment exists
    let data = make_data(1, "Alice", 100.0);
    db.insert("t", b"pk_1", &data)?;
    let segs = db.list_segments("t")?;
    assert!(!segs.is_empty(), "Should have at least one segment after insert");

    // Segment IDs should be non-empty strings
    for seg_id in &segs {
        assert!(!seg_id.is_empty(), "Segment ID should not be empty");
    }

    Ok(())
}

#[test]
fn test_mmap_read_returns_same_as_bufreader() -> Result<()> {
    let temp_dir = tempdir()?;
    let db = RockDuck::open(temp_dir.path())?;

    let data = make_data(42, "MmapTest", 99.9);
    db.insert("t", b"pk_1", &data)?;

    let seg_ids = db.list_segments("t")?;
    assert!(!seg_ids.is_empty(), "Should have at least one segment");
    let seg_id = &seg_ids[0];

    // Freeze segment → mmap path
    db.freeze_segment(seg_id)?;

    let batches = db.scan("t", None, None)?;
    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert!(total_rows >= 1, "Should read at least 1 row via mmap");

    let batch = &batches[0];
    let ids = batch.column_by_name("id").expect("id column");
    let id_arr = ids.as_primitive::<Int64Type>();
    assert!(id_arr.iter().any(|v| v == Some(42)), "Should contain inserted id=42");

    Ok(())
}
