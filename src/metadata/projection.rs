//! Projection Metadata CRUD in RocksDB
//!
//! Stores ProjectionMeta in CF "proj_meta": key = "{table}:{proj_id}", value = bincode(ProjectionMeta)

use rocksdb::DB;
use crate::codec::{encode, decode};
use crate::error::{RockDuckError, Result};
use crate::segment::projection::ProjectionMeta;
use crate::metadata::rocksdb::CF_PROJ_META;

/// Store a projection's metadata
pub fn put_projection_meta(db: &DB, proj: &ProjectionMeta) -> Result<()> {
    let key = format!("{}:{}", proj.table, proj.proj_id);
    let value = encode(proj)?;
    let cf = db.cf_handle(CF_PROJ_META)
        .ok_or_else(|| RockDuckError::Storage("CF proj_meta not found".into()))?;

    db.put_cf(&cf, key.as_bytes(), &value)
        .map_err(RockDuckError::RocksDB)
}

/// Retrieve a projection's metadata
pub fn get_projection_meta(db: &DB, table: &str, proj_id: &str) -> Result<Option<ProjectionMeta>> {
    let key = format!("{}:{}", table, proj_id);
    let cf = db.cf_handle(CF_PROJ_META)
        .ok_or_else(|| RockDuckError::Storage("CF proj_meta not found".into()))?;

    match db.get_cf(cf, key.as_bytes())? {
        Some(bytes) => {
            let proj: ProjectionMeta = decode(&bytes)?;
            Ok(Some(proj))
        }
        None => Ok(None),
    }
}

/// List all projection IDs for a table
pub fn list_table_projections(db: &DB, table: &str) -> Result<Vec<ProjectionMeta>> {
    let cf = db.cf_handle(CF_PROJ_META)
        .ok_or_else(|| RockDuckError::Storage("CF proj_meta not found".into()))?;

    let prefix = format!("{}:", table);
    let mut projections = Vec::new();

    let mut iter = db.raw_iterator_cf(cf);
    iter.seek(prefix.as_bytes());

    while iter.valid() {
        if let Some(key) = iter.key() {
            let key_str = String::from_utf8_lossy(key);
            if key_str.starts_with(&prefix) {
                if let Some(value) = iter.value() {
                    if let Ok(proj) = decode::<ProjectionMeta>(value) {
                        projections.push(proj);
                    }
                }
            } else {
                break;
            }
        }
        iter.next();
    }

    Ok(projections)
}

/// Delete a projection's metadata
pub fn delete_projection_meta(db: &DB, table: &str, proj_id: &str) -> Result<()> {
    let key = format!("{}:{}", table, proj_id);
    let cf = db.cf_handle(CF_PROJ_META)
        .ok_or_else(|| RockDuckError::Storage("CF proj_meta not found".into()))?;

    db.delete_cf(&cf, key.as_bytes())
        .map_err(RockDuckError::RocksDB)
}

#[cfg(test)]
mod tests {
    // RocksDB tests would require a real DB instance.
    // Integration tests live in tests/projection_tests.rs
    #[test]
    fn test_projection_meta_key_format() {
        let key = format!("{}:{}", "users", "proj_city");
        assert_eq!(key, "users:proj_city");
    }
}
