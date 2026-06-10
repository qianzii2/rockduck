//! WAL serialization utilities: Arrow Array -> bytes, bytes -> Arrow Array
//!
//! Provides three tiers of serialization:
//! - Arrow IPC File (legacy): per-column with schema header per column
//! - Arrow IPC Stream (Level 1): whole batch, schema encoded once
//! - Raw Arrow Buffers (Level 3): zero-encoding overhead (stub — requires arrow 58 API work)

use arrow_array::RecordBatch;
use std::io::Cursor;
use std::sync::Arc;

use crate::error::{Result, RockDuckError};

// =============================================================================
// Legacy: Arrow IPC File — per-column (used for single-row CDC before_images)
// =============================================================================

/// Serialize a RecordBatch to bytes using Arrow IPC file format.
/// The schema header + all data buffers are included in the output.
/// Can be deserialized with `bytes_to_batch`.
pub fn batch_to_bytes(batch: &RecordBatch) -> Result<Vec<u8>> {
    let mut buf = Vec::new();
    let mut writer =
        arrow_ipc::writer::FileWriter::try_new(Cursor::new(&mut buf), batch.schema().as_ref())
            .map_err(|e| RockDuckError::Codec(format!("IPC write error: {e}")))?;
    writer
        .write(batch)
        .map_err(|e| RockDuckError::Codec(format!("IPC write error: {e}")))?;
    writer
        .finish()
        .map_err(|e| RockDuckError::Codec(format!("IPC finish error: {e}")))?;
    Ok(buf)
}

/// Deserialize bytes back to a RecordBatch using Arrow IPC.
/// The input must be a valid Arrow IPC file produced by `batch_to_bytes`.
pub fn bytes_to_batch(buf: &[u8]) -> Result<RecordBatch> {
    let cursor = std::io::Cursor::new(buf);
    let mut reader = arrow_ipc::reader::FileReader::try_new(cursor, None)
        .map_err(|e| RockDuckError::Codec(format!("IPC read error: {e}")))?;
    match reader.next() {
        Some(Ok(batch)) => Ok(batch),
        Some(Err(e)) => Err(RockDuckError::Codec(format!("IPC read error: {e}"))),
        None => Err(RockDuckError::Codec("Empty IPC file".to_string())),
    }
}

// =============================================================================
// Level 1: Arrow IPC Stream — schema encoded once per batch
// =============================================================================

/// Serialize a RecordBatch to Arrow IPC Stream bytes (schema encoded once).
/// Level 1: eliminates N × schema header by encoding the schema once.
pub fn batch_to_ipc_stream(batch: &RecordBatch) -> Result<Vec<u8>> {
    let mut buf = Vec::new();
    let mut writer = arrow_ipc::writer::StreamWriter::try_new(
        std::io::Cursor::new(&mut buf),
        batch.schema().as_ref(),
    )
    .map_err(|e| RockDuckError::Codec(format!("IPC stream write: {e}")))?;
    writer
        .write(batch)
        .map_err(|e| RockDuckError::Codec(format!("IPC stream write: {e}")))?;
    writer
        .finish()
        .map_err(|e| RockDuckError::Codec(format!("IPC stream finish: {e}")))?;
    Ok(buf)
}

/// Deserialize Arrow IPC Stream bytes back to a RecordBatch.
pub fn ipc_stream_to_batch(buf: &[u8]) -> Result<RecordBatch> {
    if buf.is_empty() {
        return Err(RockDuckError::Codec("Empty IPC stream".to_string()));
    }
    let mut reader = arrow_ipc::reader::StreamReader::try_new(std::io::Cursor::new(buf), None)
        .map_err(|e| RockDuckError::Codec(format!("IPC stream read error: {e}")))?;
    match reader.next() {
        Some(Ok(batch)) => Ok(batch),
        Some(Err(e)) => Err(RockDuckError::Codec(format!("IPC stream batch error: {e}"))),
        None => Err(RockDuckError::Codec("Empty IPC stream".to_string())),
    }
}

// =============================================================================
// Level 3: Raw Arrow Buffer — zero encoding overhead (stub)
// =============================================================================
//
// Level 3 is deferred: arrow-array 58 removed the public `ArrayData::try_new` API
// and restructured the ArrayData internals. The correct approach for arrow 58 is to
// use `ArrowIPC` as the schema container (store schema + raw buffers separately)
// or pin to a compatible arrow version.
// A proper Level 3 implementation requires:
// 1. Using `ArrowIPC` as the schema container (store schema + raw buffers separately)
// 2. Or pinning to arrow < 58 which has public ArrayData constructors
// 3. Or using typed builders (PrimitiveBuilder, BinaryBuilder, etc.)
//
// For now, Level 1 (Arrow IPC Stream) provides the primary benefit — eliminating
// N × schema header redundancy. Level 3 can be revisited in a follow-up PR.

/// Serialize a RecordBatch to raw Arrow buffer bytes.
/// Level 3: zero encoding overhead — stores raw Arrow buffers directly.
///
/// # Deferred
/// This function is a stub. The arrow-array 58 release removed the public
/// `ArrayData::try_new` API needed to reconstruct arrays from raw buffers.
/// See the module-level docs for the full roadmap.
pub fn batch_to_raw_bytes(_batch: &RecordBatch) -> Result<Vec<u8>> {
    Err(RockDuckError::Codec(
        "Level 3 raw bytes (batch_to_raw_bytes) is deferred: requires arrow 58 API work".into(),
    ))
}

/// Deserialize raw Arrow buffer bytes back to a RecordBatch.
/// Requires schema to reconstruct Array objects.
///
/// # Deferred
/// See `batch_to_raw_bytes` for details.
pub fn raw_bytes_to_batch(_buf: &[u8], _schema: &arrow_schema::SchemaRef) -> Result<RecordBatch> {
    Err(RockDuckError::Codec(
        "Level 3 raw bytes (raw_bytes_to_batch) is deferred: requires arrow 58 API work".into(),
    ))
}

// =============================================================================
// Shared IPC helpers
// =============================================================================

/// Decode Arrow IPC file bytes to a single ArrayRef (first column).
/// Used by merge.rs and sparsity.rs to decode column patch values.
pub fn decode_ipc_column(bytes: &[u8]) -> Result<arrow_array::ArrayRef> {
    if bytes.is_empty() {
        return Ok(Arc::new(arrow_array::BooleanArray::new_null(0)));
    }
    let batch = bytes_to_batch(bytes)?;
    batch
        .columns()
        .first()
        .cloned()
        .ok_or_else(|| RockDuckError::Codec("Empty Arrow IPC payload".into()))
}

/// Serialize a single-row column array to IPC bytes for CDC before/after images.
/// Wraps the array in a single-row RecordBatch and serializes it.
pub fn arrow_to_bytes(arr: &arrow_array::ArrayRef) -> Option<Vec<u8>> {
    use std::sync::Arc as Arc2;

    let field = arrow_schema::Field::new("val", arr.data_type().clone(), true);
    let schema = Arc2::new(arrow_schema::Schema::new(vec![field]));

    RecordBatch::try_new(schema, vec![arr.slice(0, 1)])
        .ok()
        .and_then(|batch| batch_to_bytes(&batch).ok())
}
