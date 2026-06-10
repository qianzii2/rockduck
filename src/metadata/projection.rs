//! Column projection utilities
//!
//! Projects columns from serialized Arrow data.

use std::io::Cursor;
use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_schema::Schema;

use crate::error::{Result, RockDuckError};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContractEnforcement {
    Advisory,
    Blocking,
}

impl ContractEnforcement {
    pub fn blocks_regressions(self) -> bool {
        matches!(self, Self::Blocking)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProjectionSurface {
    PointGet,
    HistoricalPointGet,
    TimeTravelScanner,
    Vtab,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SidecarClass {
    CoreRead,
    SanctionedSidecar,
}

#[derive(Debug, Clone)]
pub struct ProjectionContract {
    pub surface: ProjectionSurface,
    pub visibility: crate::mvcc::visibility::VisibilityProjection,
    pub sidecar_class: SidecarClass,
    pub evidence_hook: &'static str,
    pub enforcement: ContractEnforcement,
}

impl ProjectionContract {
    pub fn assert_blocking_governance(&self) {
        assert!(
            self.enforcement.blocks_regressions(),
            "projection contract must opt into blocking enforcement for correctness-critical seams"
        );
        assert!(
            !self.evidence_hook.is_empty(),
            "projection contract must declare an evidence hook before entering enforced mode"
        );
    }

    pub fn point_get() -> Self {
        Self {
            surface: ProjectionSurface::PointGet,
            visibility: crate::mvcc::visibility::VisibilityProjection::Online,
            sidecar_class: SidecarClass::CoreRead,
            evidence_hook: "point_get direct read path currently has no routing evidence hook",
            enforcement: ContractEnforcement::Blocking,
        }
    }

    pub fn historical_point_get() -> Self {
        Self {
            surface: ProjectionSurface::HistoricalPointGet,
            visibility: crate::mvcc::visibility::VisibilityProjection::Historical,
            sidecar_class: SidecarClass::CoreRead,
            evidence_hook:
                "historical point_get uses historical visibility projection without router evidence",
            enforcement: ContractEnforcement::Blocking,
        }
    }

    pub fn time_travel_scanner() -> Self {
        Self {
            surface: ProjectionSurface::TimeTravelScanner,
            visibility: crate::mvcc::visibility::VisibilityProjection::Historical,
            sidecar_class: SidecarClass::CoreRead,
            evidence_hook: "TimeTravelScanner uses historical projection; evidence remains local to time-travel surfaces",
            enforcement: ContractEnforcement::Blocking,
        }
    }

    pub fn vtab() -> Self {
        Self {
            surface: ProjectionSurface::Vtab,
            visibility: crate::mvcc::visibility::VisibilityProjection::Vtab,
            sidecar_class: SidecarClass::SanctionedSidecar,
            evidence_hook:
                "DuckDB VTab is a sanctioned sidecar and bypasses router evidence by design",
            enforcement: ContractEnforcement::Blocking,
        }
    }
}

#[cfg(test)]
mod governance_contract_tests {
    use super::*;

    #[test]
    fn advisory_contracts_fail_blocking_governance_assertion() {
        let contract = ProjectionContract {
            surface: ProjectionSurface::PointGet,
            visibility: crate::mvcc::visibility::VisibilityProjection::Online,
            sidecar_class: SidecarClass::CoreRead,
            evidence_hook: "advisory hook",
            enforcement: ContractEnforcement::Advisory,
        };

        let panic = std::panic::catch_unwind(|| contract.assert_blocking_governance());
        assert!(
            panic.is_err(),
            "advisory contracts must fail blocking governance"
        );
    }

    #[test]
    fn empty_evidence_hook_fails_blocking_governance_assertion() {
        let contract = ProjectionContract {
            surface: ProjectionSurface::PointGet,
            visibility: crate::mvcc::visibility::VisibilityProjection::Online,
            sidecar_class: SidecarClass::CoreRead,
            evidence_hook: "",
            enforcement: ContractEnforcement::Blocking,
        };

        let panic = std::panic::catch_unwind(|| contract.assert_blocking_governance());
        assert!(
            panic.is_err(),
            "blocking contracts must declare evidence hooks"
        );
    }
}

/// Projection format for serialized data.
#[derive(Debug, Clone, Copy)]
pub enum ProjectionFormat {
    /// Arrow IPC (aka Feather / IPC) format
    ArrowIpc,
}

/// Project specific columns from serialized Arrow IPC data.
// P9-47: Deserializes Arrow IPC bytes, extracts requested columns, re-serializes.
pub fn project_columns(
    columns: &[String],
    data: &[u8],
    format: ProjectionFormat,
) -> Result<Vec<u8>> {
    match format {
        ProjectionFormat::ArrowIpc => project_columns_arrow_ipc(columns, data),
    }
}

fn project_columns_arrow_ipc(columns: &[String], data: &[u8]) -> Result<Vec<u8>> {
    use arrow_ipc::{reader, writer};

    let cursor = Cursor::new(data);
    let reader = reader::FileReader::try_new(cursor, None)
        .map_err(|e| RockDuckError::ReadPath(format!("Arrow IPC reader: {}", e)))?;

    let mut projected_batches = Vec::new();
    for batch_result in reader {
        let batch: RecordBatch =
            batch_result.map_err(|e| RockDuckError::ReadPath(format!("Arrow IPC batch: {}", e)))?;

        let projected = project_batch_columns(batch, columns)?;
        if projected.num_rows() > 0 {
            projected_batches.push(projected);
        }
    }

    if projected_batches.is_empty() {
        return Ok(Vec::new());
    }

    let schema = projected_batches[0].schema();
    let mut buf = Vec::new();
    let writer_options = writer::IpcWriteOptions::default();
    let mut writer = writer::FileWriter::try_new_with_options(&mut buf, &schema, writer_options)
        .map_err(|e| RockDuckError::Write(format!("Arrow IPC writer: {}", e)))?;

    for batch in projected_batches {
        writer
            .write(&batch)
            .map_err(|e| RockDuckError::Write(format!("write projected batch: {}", e)))?;
    }

    writer
        .finish()
        .map_err(|e| RockDuckError::Write(format!("finish IPC writer: {}", e)))?;

    Ok(buf)
}

fn project_batch_columns(batch: RecordBatch, columns: &[String]) -> Result<RecordBatch> {
    let schema = batch.schema();
    let fields = schema.fields();

    let mut projected_cols = Vec::new();
    let mut projected_fields = Vec::new();

    for col_name in columns {
        if let Some(idx) = fields.iter().position(|f| f.name() == col_name) {
            projected_cols.push(batch.column(idx).clone());
            projected_fields.push(fields[idx].clone());
        }
    }

    if projected_cols.is_empty() {
        return Ok(RecordBatch::new_empty(Arc::new(Schema::new(
            projected_fields,
        ))));
    }

    RecordBatch::try_new(Arc::new(Schema::new(projected_fields)), projected_cols)
        .map_err(|e| RockDuckError::ReadPath(format!("project batch: {}", e)))
}
