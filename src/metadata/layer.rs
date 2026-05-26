//! Layer：Immutable Snapshot Layer
//!
//! 用于管理不可变的 snapshot layer，支持 differential storage
//! 每个 layer 包含一组 segment 的引用

use serde::{Deserialize, Serialize};
use bincode_next::{Encode, Decode};
use rocksdb::{DB, ColumnFamily};
use crate::error::{RockDuckError, Result};
use crate::codec::{encode, decode};

/// Layer 版本
#[derive(Debug, Clone, Serialize, Deserialize, Encode, Decode)]
pub struct Layer {
    pub layer_id: String,
    pub created_at: u64,
    pub segments: Vec<SegmentRef>,
    pub row_count: u64,
    pub size_bytes: u64,
}

impl Layer {
    pub fn new(layer_id: String) -> Self {
        let now = crate::codec::current_timestamp_secs();
        Self {
            layer_id,
            created_at: now,
            segments: Vec::new(),
            row_count: 0,
            size_bytes: 0,
        }
    }

    pub fn add_segment(&mut self, seg_ref: SegmentRef) {
        self.row_count += seg_ref.row_count;
        self.size_bytes += seg_ref.size_bytes;
        self.segments.push(seg_ref);
    }
}

/// Segment 引用（属于某个 Layer）
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, bincode_next::Encode, bincode_next::Decode)]
pub struct SegmentRef {
    pub seg_id: String,
    pub row_count: u64,
    pub size_bytes: u64,
}

/// Layer CF 名称
pub const CF_LAYER: &str = crate::metadata::rocksdb::CF_LAYER;

/// 构建 layer key
pub fn layer_key(layer_id: &str) -> Vec<u8> {
    let mut key = Vec::with_capacity(layer_id.len() + 1);
    key.extend_from_slice(layer_id.as_bytes());
    key
}

/// 获取 layer CF
pub fn layer_cf(db: &DB) -> Result<&ColumnFamily> {
    db.cf_handle(CF_LAYER)
        .ok_or_else(|| RockDuckError::Metadata(format!("Column family {} not found", CF_LAYER)))
}

/// 写入 layer
pub fn put_layer(db: &DB, layer: &Layer) -> Result<()> {
    let cf = layer_cf(db)?;
    let key = layer_key(&layer.layer_id);
    let value = encode(layer)?;
    db.put_cf(&cf, &key, &value)?;
    Ok(())
}

/// 读取 layer
pub fn get_layer(db: &DB, layer_id: &str) -> Result<Option<Layer>> {
    let cf = layer_cf(db)?;
    let key = layer_key(layer_id);
    match db.get_cf(&cf, &key)? {
        Some(data) => {
            let layer: Layer = decode(&data)?;
            Ok(Some(layer))
        }
        None => Ok(None),
    }
}

/// 列出所有 layer
pub fn list_layers(db: &DB) -> Result<Vec<Layer>> {
    let cf = layer_cf(db)?;
    let mut layers = Vec::new();

    let mut iter = db.raw_iterator_cf(&cf);
    iter.seek(b"");

    while iter.valid() {
        if let Some(value) = iter.value() {
            if let Ok(layer) = decode::<Layer>(value) {
                layers.push(layer);
            }
        }
        iter.next();
    }

    layers.sort_by_key(|l| l.created_at);
    Ok(layers)
}

/// 删除 layer
pub fn delete_layer(db: &DB, layer_id: &str) -> Result<()> {
    let cf = layer_cf(db)?;
    let key = layer_key(layer_id);
    db.delete_cf(&cf, &key)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_creation() {
        let mut layer = Layer::new("layer_001".to_string());
        layer.add_segment(SegmentRef {
            seg_id: "seg_001".to_string(),
            row_count: 1000,
            size_bytes: 1024 * 1024,
        });
        assert_eq!(layer.row_count, 1000);
        assert_eq!(layer.segments.len(), 1);
    }

    // ============================================================
    // Layer::new tests
    // ============================================================

    #[test]
    fn test_layer_new_id() {
        let layer = Layer::new("layer_x".to_string());
        assert_eq!(layer.layer_id, "layer_x");
        assert!(layer.created_at > 0);
        assert!(layer.segments.is_empty());
        assert_eq!(layer.row_count, 0);
        assert_eq!(layer.size_bytes, 0);
    }

    #[test]
    fn test_layer_add_segment_multiple() {
        let mut layer = Layer::new("layer_001".to_string());
        layer.add_segment(SegmentRef {
            seg_id: "seg_001".to_string(),
            row_count: 100,
            size_bytes: 1024,
        });
        layer.add_segment(SegmentRef {
            seg_id: "seg_002".to_string(),
            row_count: 200,
            size_bytes: 2048,
        });
        layer.add_segment(SegmentRef {
            seg_id: "seg_003".to_string(),
            row_count: 50,
            size_bytes: 512,
        });

        assert_eq!(layer.row_count, 350);
        assert_eq!(layer.size_bytes, 3584);
        assert_eq!(layer.segments.len(), 3);
    }

    #[test]
    fn test_layer_add_segment_accumulates() {
        let mut layer = Layer::new("layer_001".to_string());
        layer.add_segment(SegmentRef {
            seg_id: "seg_001".to_string(),
            row_count: 1000,
            size_bytes: 1_000_000,
        });
        layer.add_segment(SegmentRef {
            seg_id: "seg_002".to_string(),
            row_count: 1000,
            size_bytes: 1_000_000,
        });
        assert_eq!(layer.row_count, 2000);
        assert_eq!(layer.size_bytes, 2_000_000);
    }

    // ============================================================
    // SegmentRef tests
    // ============================================================

    #[test]
    fn test_segment_ref_construction() {
        let seg_ref = SegmentRef {
            seg_id: "seg_abc".to_string(),
            row_count: 5000,
            size_bytes: 5_000_000,
        };
        assert_eq!(seg_ref.seg_id, "seg_abc");
        assert_eq!(seg_ref.row_count, 5000);
        assert_eq!(seg_ref.size_bytes, 5_000_000);
    }

    // ============================================================
    // layer_key tests
    // ============================================================

    #[test]
    fn test_layer_key_format() {
        let key = layer_key("layer_123");
        assert_eq!(key, b"layer_123");
    }

    #[test]
    fn test_layer_key_empty() {
        let key = layer_key("");
        assert_eq!(key, b"");
    }

    // ============================================================
    // Layer Debug
    // ============================================================

    #[test]
    fn test_layer_debug() {
        let layer = Layer::new("layer_001".to_string());
        let debug_str = format!("{:?}", layer);
        assert!(!debug_str.is_empty());
    }

    #[test]
    fn test_segment_ref_debug() {
        let seg_ref = SegmentRef {
            seg_id: "seg_001".to_string(),
            row_count: 100,
            size_bytes: 1024,
        };
        let debug_str = format!("{:?}", seg_ref);
        assert!(!debug_str.is_empty());
    }
}
