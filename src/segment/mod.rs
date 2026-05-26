//! Segment 模块
//!
//! 管理 Segment 和 Granule 的元数据、布局、删除掩码、更新掩码

pub mod meta;
pub mod layout;
pub mod del_mask;
pub mod upd_mask;
pub mod delta_store;
pub mod encoding;
pub mod projection;
