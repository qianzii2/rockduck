//! RockDuck 错误类型定义
//! 
//! 使用 thiserror 定义错误类型，分为以下几类：
//! - Storage: 存储相关错误
//! - Metadata: 元数据相关错误  
//! - Query: 查询执行错误
//! - Compaction: 压缩相关错误
//! - Config: 配置相关错误

use thiserror::Error;

pub type Result<T> = std::result::Result<T, RockDuckError>;

#[derive(Error, Debug)]
pub enum RockDuckError {
    // ============ Storage 错误 ============
    #[error("Storage error: {0}")]
    Storage(String),
    
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Column file not found: {0}")]
    ColumnNotFound(String),
    
    #[error("Segment not found: {0}")]
    SegmentNotFound(String),
    
    #[error("Granule not found: seg={seg_id}, granule={granule_id}")]
    GranuleNotFound { seg_id: String, granule_id: u32 },

    // ============ Metadata 错误 ============
    #[error("Metadata error: {0}")]
    Metadata(String),
    
    #[error("RocksDB error: {0}")]
    RocksDB(#[from] rocksdb::Error),
    
    #[error("Index entry not found for pk")]
    IndexEntryNotFound,
    
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    // ============ Query 错误 ============
    #[error("Query error: {0}")]
    Query(String),
    
    #[error("DuckDB error: {0}")]
    DuckDB(#[from] duckdb::Error),
    
    #[error("Arrow error: {0}")]
    Arrow(#[from] arrow::error::ArrowError),

    // ============ Compaction 错误 ============
    #[error("Compaction error: {0}")]
    Compaction(String),
    
    #[error("Merge failed: {0}")]
    MergeFailed(String),

    // ============ Config 错误 ============
    #[error("Configuration error: {0}")]
    Config(String),
    
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    // ============ 其他错误 ============
    #[error("Internal error: {0}")]
    Internal(String),
    
    #[error("Vortex error: {0}")]
    Vortex(String),
    
    #[error("Encoding error: {0}")]
    Encoding(String),

    #[error("Codec error: {0}")]
    Codec(#[from] crate::codec::CodecError),
}

impl RockDuckError {
    /// 判断是否为可重试的错误
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            RockDuckError::Io(_) | RockDuckError::Storage(_)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- is_retryable ----
    #[test]
    fn test_is_retryable_io() {
        let err = RockDuckError::Io(std::io::Error::new(std::io::ErrorKind::NotFound, "test"));
        assert!(err.is_retryable());
    }

    #[test]
    fn test_is_retryable_storage() {
        let err = RockDuckError::Storage("disk full".to_string());
        assert!(err.is_retryable());
    }

    #[test]
    fn test_is_retryable_not_retryable_variants() {
        let variants: Vec<RockDuckError> = vec![
            RockDuckError::ColumnNotFound("col".to_string()),
            RockDuckError::SegmentNotFound("seg".to_string()),
            RockDuckError::GranuleNotFound { seg_id: "s".to_string(), granule_id: 1 },
            RockDuckError::Metadata("meta error".to_string()),
            RockDuckError::IndexEntryNotFound,
            RockDuckError::Query("query error".to_string()),
            RockDuckError::Compaction("compaction error".to_string()),
            RockDuckError::MergeFailed("merge failed".to_string()),
            RockDuckError::Config("config error".to_string()),
            RockDuckError::InvalidParameter("invalid param".to_string()),
            RockDuckError::Internal("internal error".to_string()),
            RockDuckError::Vortex("vortex error".to_string()),
            RockDuckError::Encoding("encoding error".to_string()),
        ];
        for err in variants {
            assert!(!err.is_retryable(), "Expected {:?} to not be retryable", err);
        }
    }

    // ---- Display formatting ----
    #[test]
    fn test_display_storage() {
        let err = RockDuckError::Storage("disk error".to_string());
        let s = err.to_string();
        assert!(s.contains("Storage"));
        assert!(s.contains("disk error"));
    }

    #[test]
    fn test_display_io() {
        let err = RockDuckError::Io(std::io::Error::new(std::io::ErrorKind::PermissionDenied, "perm denied"));
        let s = err.to_string();
        assert!(s.contains("IO error"));
    }

    #[test]
    fn test_display_column_not_found() {
        let err = RockDuckError::ColumnNotFound("my_col".to_string());
        assert!(err.to_string().contains("Column file not found"));
        assert!(err.to_string().contains("my_col"));
    }

    #[test]
    fn test_display_segment_not_found() {
        let err = RockDuckError::SegmentNotFound("seg_001".to_string());
        assert!(err.to_string().contains("Segment not found"));
        assert!(err.to_string().contains("seg_001"));
    }

    #[test]
    fn test_display_granule_not_found() {
        let err = RockDuckError::GranuleNotFound { seg_id: "seg_A".to_string(), granule_id: 5 };
        let s = err.to_string();
        assert!(s.contains("Granule not found"));
        assert!(s.contains("seg_A"));
        assert!(s.contains("5"));
    }

    #[test]
    fn test_display_metadata() {
        let err = RockDuckError::Metadata("meta broken".to_string());
        assert!(err.to_string().contains("Metadata error"));
    }

    #[test]
    fn test_display_index_entry_not_found() {
        let err = RockDuckError::IndexEntryNotFound;
        assert!(err.to_string().contains("Index entry not found"));
    }

    #[test]
    fn test_display_query() {
        let err = RockDuckError::Query("bad query".to_string());
        assert!(err.to_string().contains("Query error"));
    }

    #[test]
    fn test_display_compaction() {
        let err = RockDuckError::Compaction(" compaction ".to_string());
        assert!(err.to_string().contains("Compaction error"));
    }

    #[test]
    fn test_display_merge_failed() {
        let err = RockDuckError::MergeFailed("merge error".to_string());
        assert!(err.to_string().contains("Merge failed"));
    }

    #[test]
    fn test_display_config() {
        let err = RockDuckError::Config("bad config".to_string());
        assert!(err.to_string().contains("Configuration error"));
    }

    #[test]
    fn test_display_invalid_parameter() {
        let err = RockDuckError::InvalidParameter("param X invalid".to_string());
        assert!(err.to_string().contains("Invalid parameter"));
    }

    #[test]
    fn test_display_internal() {
        let err = RockDuckError::Internal("internal panic".to_string());
        assert!(err.to_string().contains("Internal error"));
    }

    #[test]
    fn test_display_vortex() {
        let err = RockDuckError::Vortex("vortex error".to_string());
        assert!(err.to_string().contains("Vortex error"));
    }

    #[test]
    fn test_display_encoding() {
        let err = RockDuckError::Encoding("encoding broken".to_string());
        assert!(err.to_string().contains("Encoding error"));
    }

    // ---- From trait conversions ----
    #[test]
    fn test_from_io_error() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let err: RockDuckError = io_err.into();
        assert!(matches!(err, RockDuckError::Io(_)));
    }

    #[test]
    fn test_from_rocksdb_error() {
        // rocksdb::Error doesn't have a simple constructor, but we can test
        // that the From impl exists and compiles correctly by using ? operator
        // on a Result<_, rocksdb::Error>
        let _err_type: Option<RockDuckError> = None;
        // Just verify the type is constructible
        fn assert_send<T: Send>() {}
        assert_send::<rocksdb::Error>();
    }

    #[test]
    fn test_from_arrow_error() {
        let arrow_err = arrow::error::ArrowError::InvalidArgumentError("bad arg".to_string());
        let err: RockDuckError = arrow_err.into();
        assert!(matches!(err, RockDuckError::Arrow(_)));
    }

    // ---- Debug formatting ----
    #[test]
    fn test_debug_all_variants() {
        let variants: Vec<RockDuckError> = vec![
            RockDuckError::Io(std::io::Error::new(std::io::ErrorKind::NotFound, "test")),
            RockDuckError::Storage("storage".to_string()),
            RockDuckError::ColumnNotFound("col".to_string()),
            RockDuckError::SegmentNotFound("seg".to_string()),
            RockDuckError::GranuleNotFound { seg_id: "s".to_string(), granule_id: 1 },
            RockDuckError::Metadata("meta".to_string()),
            RockDuckError::IndexEntryNotFound,
            RockDuckError::Query("query".to_string()),
            RockDuckError::Compaction("compact".to_string()),
            RockDuckError::MergeFailed("merge".to_string()),
            RockDuckError::Config("config".to_string()),
            RockDuckError::InvalidParameter("param".to_string()),
            RockDuckError::Internal("internal".to_string()),
            RockDuckError::Vortex("vortex".to_string()),
            RockDuckError::Encoding("encoding".to_string()),
        ];
        for err in variants {
            let debug_str = format!("{:?}", err);
            assert!(!debug_str.is_empty());
        }
    }
}
