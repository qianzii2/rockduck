//! Read 模块

pub mod point_get;
pub mod scan;
pub mod late_mat;
pub mod aggregate;
pub mod adaptive_lm;
pub mod staleness;
pub mod update_materializer;

pub use point_get::*;
pub use scan::*;
pub use late_mat::*;
pub use aggregate::*;
pub use adaptive_lm::*;
pub use staleness::*;

/// RecordBatch 类型别名
pub type RecordBatch = arrow_array::RecordBatch;
