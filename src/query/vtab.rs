//! Custom VTab for RockDuck — streams Vec<RecordBatch> to DuckDB in multiple batches.
//!
//! Unlike ArrowVTab (one-shot, single batch), this implementation maintains
//! a batch index across func() calls, allowing DuckDB to consume RockDuck scan
//! results incrementally without a full concat into one giant RecordBatch.

pub use inner::RockDuckVTab;

mod inner {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    use arrow_array::RecordBatch;

    use duckdb::core::{DataChunkHandle, LogicalTypeHandle, LogicalTypeId};
    use duckdb::vtab::arrow::{record_batch_to_duckdb_data_chunk, to_duckdb_logical_type};
    use duckdb::vtab::{BindInfo, InitInfo, TableFunctionInfo, VTab};
    use tracing::{debug, info};

    use crate::db::RockDuck;
    use crate::read::scan;

    // BindData — holds all RecordBatches from scan()
    pub struct BindData {
        pub(crate) batches: Vec<RecordBatch>,
        pub(crate) total_batches: usize,
    }

    // InitData — tracks which batch to emit next across func() calls
    pub struct InitData {
        pub(crate) batch_index: AtomicUsize,
        pub(crate) total_batches: usize,
    }

    pub struct RockDuckVTab;

    impl VTab for RockDuckVTab {
        type BindData = BindData;
        type InitData = InitData;

        fn bind(bind: &BindInfo) -> Result<Self::BindData, Box<dyn std::error::Error>> {
            let param_count = bind.get_parameter_count();
            if param_count == 0 {
                bind.set_error("docdb_scan requires at least one parameter: path");
                return Err("missing path parameter".into());
            }

            let path = bind.get_parameter(0).to_string();

            debug!("docdb_scan bind: path={}", path);

            let rockduck = RockDuck::open(&path)?;
            let batches = scan::scan(&rockduck, "default", None, None)
                .map_err(|e| format!("scan failed: {}", e))?;

            let total_batches = batches.len();
            if batches.is_empty() {
                info!("docdb_scan: empty result set for path={}", path);
            } else {
                let total_rows = batches.iter().map(|b| b.num_rows() as u64).sum::<u64>();
                info!("docdb_scan bind: loaded {} batches, {} total rows",
                    total_batches, total_rows);
            }

            let schema = batches.first()
                .map(|b| b.schema())
                .unwrap_or_else(|| Arc::new(arrow_schema::Schema::new(
                    Vec::<arrow_schema::Field>::new()
                )));

            // Handle empty tables: must register at least one column for DuckDB schema validity.
            if schema.fields().is_empty() {
                bind.add_result_column("__docdb_scan_dummy", LogicalTypeHandle::from(LogicalTypeId::UBigint));
            }

            for field in schema.fields() {
                let logical_type = to_duckdb_logical_type(field.data_type())
                    .map_err(|e| format!("unsupported Arrow type {:?}: {}", field.data_type(), e))?;
                bind.add_result_column(field.name(), logical_type);
            }

            let total_rows: u64 = batches.iter().map(|b| b.num_rows() as u64).sum();
            bind.set_cardinality(total_rows, true);

            Ok(BindData { batches, total_batches })
        }

        fn init(init: &InitInfo) -> Result<Self::InitData, Box<dyn std::error::Error>> {
            init.set_max_threads(1);

            let bind_ptr = init.get_bind_data::<Self::BindData>();
            let total_batches = if bind_ptr.is_null() {
                0
            } else {
                let total = unsafe { (*bind_ptr).total_batches };
                total
            };

            debug!("docdb_scan init: total_batches={}", total_batches);

            Ok(InitData {
                batch_index: AtomicUsize::new(0),
                total_batches,
            })
        }

        fn func(
            func: &TableFunctionInfo<Self>,
            output: &mut DataChunkHandle,
        ) -> Result<(), Box<dyn std::error::Error>> {
            let bind_ptr = func.get_bind_data();
            let init_ptr = func.get_init_data();

            let bind = &*bind_ptr;
            let init = &*init_ptr;

            let idx = init.batch_index.fetch_add(1, Ordering::Relaxed);

            if idx >= init.total_batches || init.total_batches == 0 {
                debug!("docdb_scan func: all batches consumed (idx={})", idx);
                output.set_len(0);
                return Ok(());
            }

            let batch = &bind.batches[idx];
            let num_rows = batch.num_rows();

            if num_rows == 0 {
                output.set_len(0);
                return Ok(());
            }

            record_batch_to_duckdb_data_chunk(batch, output)
                .map_err(|e| format!("record_batch_to_duckdb_data_chunk failed: {}", e))?;

            debug!("docdb_scan func: emitted batch {} ({} rows)", idx, num_rows);
            Ok(())
        }

        fn parameters() -> Option<Vec<LogicalTypeHandle>> {
            Some(vec![LogicalTypeHandle::from(LogicalTypeId::Varchar)])
        }
    }
}
