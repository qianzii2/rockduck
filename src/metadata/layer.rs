//! Layer metadata for tiered storage

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerMetadata {
    pub layer_id: u32,
    pub compression: String,
    pub file_count: u32,
}
