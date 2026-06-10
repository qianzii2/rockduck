//! Benchmarks for vis/deltavis file I/O operations.
//!
//! Run with: `cargo bench --bench vis_deltavis`
//!
//! Measures:
//! - `mark_deleted_append`: append-only deltavis write latency (typical write path)
//! - `read_deltavis`: sequential read of all deltavis entries (typical read path)
//! - `append_batch` (full rewrite): legacy full-file rewrite latency (fallback path)
//! - File size growth: how deltavis grows with sequential deletes vs segment size
//!
//! Typical expectations:
//! - Append: < 1ms for a single record (16 bytes + header/CRC)
//! - Read: < 10ms for 10k entries (160 KB + CRC)
//! - Full rewrite: grows with num_batches — used only during WAL recovery
//! - If deltavis > ~10MB, compaction should fire to merge back into main vis file

use std::path::PathBuf;
use std::sync::Arc;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use tempfile::TempDir;

use rockduck::mvcc::shadow_columns as sc;
use rockduck::write::vis_file::{
    VisFileWriter, DELTAVIS_FOOTER_SIZE, DELTAVIS_HEADER_SIZE, DELTAVIS_RECORD_SIZE,
};

/// Build a single visibility RecordBatch with `n_rows` rows.
fn make_vis_batch(n_rows: usize) -> arrow_array::RecordBatch {
    let schema = sc::visibility_schema();
    let created: arrow_array::UInt64Array = (0u64..n_rows as u64).collect();
    let deleted: arrow_array::UInt64Array = arrow_array::UInt64Array::from(vec![0u64; n_rows]); // 0 = NOT_DELETED
    let batch = arrow_array::RecordBatch::try_new(
        schema,
        vec![
            Arc::new(created) as arrow_array::ArrayRef,
            Arc::new(deleted) as arrow_array::ArrayRef,
        ],
    )
    .unwrap();
    batch
}

/// Create a vis file with `num_batches` batches of `rows_per_batch` rows each.
fn setup_vis_file(tmp: &TempDir, num_batches: usize, rows_per_batch: usize) -> PathBuf {
    let vis_path = tmp.path().join("__vis.vortex");
    let writer = VisFileWriter::new(&vis_path);
    for _ in 0..num_batches {
        let batch = make_vis_batch(rows_per_batch);
        writer.write_batch(&batch).unwrap();
    }
    vis_path
}

// ── mark_deleted_append benchmarks ─────────────────────────────────────────────

fn bench_mark_deleted_append(c: &mut Criterion) {
    let mut group = c.benchmark_group("mark_deleted_append");

    // Vary by number of existing deltavis entries (simulating increasing file size)
    for num_existing_entries in [0, 100, 1_000, 10_000, 100_000].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(num_existing_entries),
            num_existing_entries,
            |b, &num_existing| {
                let tmp = TempDir::new().unwrap();
                let vis_path = setup_vis_file(&tmp, 1, 1000);
                let writer = VisFileWriter::new(&vis_path);

                // Pre-populate deltavis with `num_existing` entries
                for i in 0..num_existing {
                    writer
                        .mark_deleted_append(i as u64, i as u64 + 1000)
                        .unwrap();
                }

                // Measure one append
                b.iter(|| black_box(writer.mark_deleted_append(num_existing as u64 + 1, 2000)));
            },
        );
    }

    group.finish();
}

// ── read_deltavis benchmarks ─────────────────────────────────────────────────

fn bench_read_deltavis(c: &mut Criterion) {
    let mut group = c.benchmark_group("read_deltavis");

    for num_entries in [0, 100, 1_000, 10_000, 100_000].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(num_entries),
            num_entries,
            |b, &num_entries| {
                let tmp = TempDir::new().unwrap();
                let vis_path = setup_vis_file(&tmp, 1, 1000);
                let writer = VisFileWriter::new(&vis_path);

                // Populate deltavis
                for i in 0..num_entries {
                    writer
                        .mark_deleted_append(i as u64, i as u64 + 1000)
                        .unwrap();
                }

                // Measure read
                b.iter(|| black_box(writer.read_deltavis()));
            },
        );
    }

    group.finish();
}

// ── append_batch (full rewrite) benchmarks ────────────────────────────────────

fn bench_append_batch_full_rewrite(c: &mut Criterion) {
    let mut group = c.benchmark_group("append_batch_full_rewrite");

    // Vary by number of existing batches in vis file (determines rewrite size)
    for (num_batches, rows_per_batch) in [(1, 100), (10, 100), (100, 100)].iter() {
        let id = format!("{}x{}", num_batches, rows_per_batch);
        group.bench_with_input(
            BenchmarkId::new("vis_batches", id.as_str()),
            &(num_batches, rows_per_batch),
            |b, &(num_batches, rows_per_batch)| {
                let tmp = TempDir::new().unwrap();
                let vis_path = setup_vis_file(&tmp, *num_batches, *rows_per_batch);
                let writer = VisFileWriter::new(&vis_path);
                let batch = make_vis_batch(*rows_per_batch);
                // append_batch expects &[ArrayRef], extract columns from RecordBatch
                let arrays: Vec<arrow_array::ArrayRef> = (0..batch.num_columns())
                    .map(|i| batch.column(i).clone())
                    .collect();

                // Measure full rewrite: load all + append new + rewrite
                b.iter(|| {
                    let _ = writer.append_batch(&arrays);
                    black_box(());
                });
            },
        );
    }

    group.finish();
}

// ── File size growth benchmarks ───────────────────────────────────────────────

fn bench_deltavis_file_size_growth(c: &mut Criterion) {
    let mut group = c.benchmark_group("deltavis_file_size");

    for num_deletes in [1, 10, 100, 1_000, 10_000].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(num_deletes),
            num_deletes,
            |_b, &num_deletes| {
                let tmp = TempDir::new().unwrap();
                let vis_path = setup_vis_file(&tmp, 1, 1000);
                let writer = VisFileWriter::new(&vis_path);

                for i in 0..num_deletes {
                    writer
                        .mark_deleted_append(i as u64, i as u64 + 1000)
                        .unwrap();
                }

                // Read back and check size
                let entries = writer.read_deltavis().unwrap();
                let file_size = std::fs::metadata(writer.deltavis_path()).unwrap().len();
                let theoretical_size = (entries.len() as u64)
                    * (DELTAVIS_RECORD_SIZE as u64)
                    + (DELTAVIS_HEADER_SIZE as u64)
                    + (DELTAVIS_FOOTER_SIZE as u64);

                // Assert size matches theoretical (no growth overhead)
                assert_eq!(
                    file_size,
                    theoretical_size,
                    "file size mismatch for {} entries",
                    entries.len()
                );
                let _ = black_box(entries.len());
            },
        );
    }

    group.finish();
}

// ── Compact threshold benchmarks ───────────────────────────────────────────────

fn bench_compact_threshold(c: &mut Criterion) {
    // Measure the "sweet spot" for when deltavis compaction should fire.
    // Compaction rewrites vis file + deletes deltavis.
    // Heuristic: fire when deltavis_file_size > vis_file_size * compaction_ratio_threshold

    let mut group = c.benchmark_group("compaction_threshold");

    // Test with 1000-entry deltavis at various vis file sizes
    for vis_batches in [1, 10, 100].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(vis_batches),
            vis_batches,
            |_b, &vis_batches| {
                let tmp = TempDir::new().unwrap();
                let vis_path = setup_vis_file(&tmp, vis_batches, 1000);
                let writer = VisFileWriter::new(&vis_path);

                // Populate 1000 deltavis entries
                for i in 0..1000 {
                    writer
                        .mark_deleted_append(i as u64, i as u64 + 1000)
                        .unwrap();
                }

                let vis_size = std::fs::metadata(&vis_path).unwrap().len();
                let delta_size = std::fs::metadata(writer.deltavis_path()).unwrap().len();
                let ratio = delta_size as f64 / vis_size.max(1) as f64;

                // Heuristic: compact when delta > 10% of vis
                if ratio > 0.1 {
                    // Simulate compaction: rewrite vis from reader + delete deltavis
                    let total_rows = vis_batches as u32 * 1000;
                    let _compact = writer.compact_deltavis(total_rows);
                }

                let _ = black_box(ratio);
            },
        );
    }

    group.finish();
}

criterion_group!(
    name = vis_bench;
    config = Criterion::default();
    targets =
        bench_mark_deleted_append,
        bench_read_deltavis,
        bench_append_batch_full_rewrite,
        bench_deltavis_file_size_growth,
        bench_compact_threshold,
);
criterion_main!(vis_bench);
