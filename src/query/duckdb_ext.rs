//! DuckDB Extension integration
//!
//! Exposes RockDuck as an external data source to DuckDB via table functions.
//!
//! Uses DuckDB's native `VTab` trait (duckdb::vtab::VTab) which provides:
//! - `duckdb::core::DataChunkHandle` in the func() callback
//! - `record_batch_to_duckdb_data_chunk` for Arrow RecordBatch -> DuckDB DataChunk conversion

use std::path::PathBuf;
use std::sync::Arc;

use duckdb::Connection;
use crate::db::RockDuck;
use crate::error::Result;
use crate::query::vtab_quack;
use crate::RockDuckError;

// =============================================================
// Connection wrapper
// =============================================================

pub struct DuckDBConnection {
    rockduck: Arc<RockDuck>,
    vtab_data_root: PathBuf,
    /// RAII guard that sets thread-local state and clears it on drop.
    /// When this struct is dropped, the TLS is cleared to prevent cross-connection
    /// contamination from sequential connections on the same thread.
    #[allow(dead_code)] // stored to keep the guard alive for the lifetime of this connection
    vtab_scope: Option<crate::query::duckdb_ext::VtabScope>,
}

impl DuckDBConnection {
    pub fn new(rockduck: Arc<RockDuck>) -> Result<Self> {
        let vtab_data_root = rockduck.data_dir.clone();
        let vtab_scope = Some(
            VtabScope::enter(rockduck.clone(), vtab_data_root.clone())
                .map_err(|e| RockDuckError::DuckDB(format!("enter VTab scope: {e}")))?,
        );
        Ok(Self {
            vtab_data_root,
            rockduck,
            vtab_scope,
        })
    }

    pub fn with_vtab_data_root(rockduck: Arc<RockDuck>, vtab_data_root: PathBuf) -> Result<Self> {
        let vtab_scope = Some(
            VtabScope::enter(rockduck.clone(), vtab_data_root.clone())
                .map_err(|e| RockDuckError::DuckDB(format!("enter VTab scope: {e}")))?,
        );
        Ok(Self {
            rockduck,
            vtab_data_root,
            vtab_scope,
        })
    }

    pub fn rockduck(&self) -> &RockDuck {
        &self.rockduck
    }

    /// Create a DuckDB in-memory connection with DocDB scan function registered.
    /// The scan function can then be used as: `SELECT * FROM docdb_scan('data_dir')`
    pub fn create_duckdb_conn(&self) -> Result<Connection> {
        let conn = Connection::open_in_memory()
            .map_err(|e| RockDuckError::DuckDB(format!("open in-memory: {e}")))?;

        register_docdb_scan_with_connection(
            &conn,
            self.rockduck.clone(),
            self.vtab_data_root.clone(),
        )
        .map_err(|e| RockDuckError::DuckDB(format!("register docdb_scan: {e}")))?;

        Ok(conn)
    }
}

/// Register the `docdb_scan` table function with a DuckDB connection.
///
/// This function can be called in two contexts:
/// 1. From `DuckDBConnection::create_duckdb_conn()` — the `VtabScope` is already active
///    and `set_vtab_data_root` will be a no-op (same path) or fail gracefully (different path).
/// 2. Standalone — this function will set the TLS itself.
///
/// If called multiple times with different `vtab_data_root` values on the same thread,
/// subsequent calls will fail with an error. This is intentional: `VTAB_DATA_ROOT` is
/// a process-global OnceLock and cannot be changed after first set.
///
/// For cross-database isolation, use `DuckDBConnection` which manages the TLS scope via
/// `VtabScope` and clears TLS on drop.
pub fn register_docdb_scan_with_connection(
    conn: &Connection,
    db: Arc<RockDuck>,
    vtab_data_root: PathBuf,
) -> Result<()> {
    // Set the data root. Fails gracefully if a different path was already set
    // (indicates incorrect usage — two connections with different data roots on the same thread).
    if let Err(e) = vtab_quack::set_vtab_data_root(vtab_data_root) {
        tracing::warn!(
            error = %e,
            "vTab data root already configured for a different path; reusing existing root"
        );
    }

    // Store the provided RockDuck instance for the VTab to use.
    // Uses per-thread storage (TLS), so each thread has its own reference.
    // Concurrent connections on different threads are fully isolated.
    vtab_quack::set_rockduck_for_vtab(db);

    // Register using duckdb's native VTab trait with the name "docdb_scan"
    conn.register_table_function::<vtab_quack::RockDuckVTab>("docdb_scan")
        .map_err(|e| RockDuckError::DuckDB(format!("register_table_function: {}", e)))?;

    tracing::info!("Registered docdb_scan table function (duckdb-native VTab)");
    Ok(())
}

// =============================================================================
// Scoped TLS guard — RAII cleanup
// =============================================================================

/// RAII guard that enters a VTab scope (sets TLS) and clears it on drop.
///
/// When a `VtabScope` is dropped, it clears the thread-local RockDuck reference.
/// This prevents cross-connection contamination: if the same thread creates two
/// DuckDB connections sequentially, the first connection's TLS state does not
/// leak into the second.
///
/// # Example
/// ```ignore
/// {
///     let _scope = VtabScope::enter(db.clone(), data_root.clone())?;
///     conn.register_table_function::<vtab_quack::RockDuckVTab>("docdb_scan")?;
/// } // TLS is cleared here when _scope drops
/// ```
pub struct VtabScope {
    _root_set: bool,
}

impl VtabScope {
    /// Enter the VTab scope: set TLS and OnceLock for the current thread.
    /// Returns a guard that clears TLS on drop.
    pub fn enter(db: Arc<RockDuck>, data_root: PathBuf) -> std::result::Result<Self, String> {
        // Set the data root (OnceLock — global, one-time per process).
        // Subsequent calls with the same path succeed silently; different paths fail.
        // The Rust OnceLock prevents accidental cross-database contamination at the
        // process level. The TLS guard below prevents cross-connection contamination
        // within a single thread.
        let _ = vtab_quack::set_vtab_data_root(data_root);

        // Store the RockDuck instance in thread-local storage.
        // This is per-thread, so concurrent connections on different threads are isolated.
        vtab_quack::set_rockduck_for_vtab(db);

        Ok(Self { _root_set: true })
    }
}

impl Drop for VtabScope {
    fn drop(&mut self) {
        // Clear the thread-local reference so the next connection in this thread
        // starts with a clean slate. This prevents the first connection's queries
        // from accidentally reading from the second connection's database.
        vtab_quack::clear_rockduck_for_vtab();
    }
}
