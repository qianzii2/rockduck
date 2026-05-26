//! Del Mask：自适应编码的删除掩码
//!
//! 编码策略：
//! - del_ratio < 1%  → SkipList(Vec<u64>)：只存已删除位置
//! - del_ratio 1-50% → RoaringBitmap：roaring crate
//! - del_ratio > 50% → FullBitmap(Vec<u8>) + 触发 compaction

use std::path::Path;
use serde::{Deserialize, Serialize};
use roaring::RoaringBitmap;
use bincode_next::{Encode, Decode};
use crate::error::Result;
use crate::codec::{encode, decode};

/// 删除掩码模式
#[derive(Debug, Clone, Serialize, Deserialize, Encode, Decode)]
pub enum DelMaskMode {
    Empty,
    SkipList(Vec<u32>),
    Roaring(Vec<u8>),
    FullBitmap(Vec<u8>),
}

impl Default for DelMaskMode {
    fn default() -> Self {
        Self::Empty
    }
}

/// 删除掩码
#[derive(Debug, Clone, Default, Serialize, Deserialize, Encode, Decode)]
pub struct DelMask {
    total_rows: u32,
    #[serde(flatten)]
    mode: DelMaskMode,
    compaction_threshold: f64,
}

impl DelMask {
    pub fn new(total_rows: u64) -> Self {
        Self {
            total_rows: total_rows as u32,
            mode: DelMaskMode::Empty,
            compaction_threshold: 0.5,
        }
    }

    pub fn is_deleted(&self, pos: u64) -> bool {
        if pos >= self.total_rows as u64 {
            return false;
        }
        let pos32 = pos as u32;
        match &self.mode {
            DelMaskMode::Empty => false,
            DelMaskMode::SkipList(list) => list.contains(&pos32),
            DelMaskMode::Roaring(data) => {
                let bitmap = RoaringBitmap::deserialize_from(data.as_slice()).unwrap_or_default();
                bitmap.contains(pos32)
            }
            DelMaskMode::FullBitmap(data) => {
                let byte_idx = pos as usize / 8;
                let bit_idx = pos as usize % 8;
                byte_idx < data.len() && (data[byte_idx] & (1 << bit_idx)) != 0
            }
        }
    }

    pub fn add_delete(&mut self, pos: u64) {
        let pos32 = pos as u32;
        match &mut self.mode {
            DelMaskMode::Empty => {
                self.mode = DelMaskMode::SkipList(vec![pos32]);
            }
            DelMaskMode::SkipList(list) => {
                if !list.contains(&pos32) {
                    list.push(pos32);
                    list.sort_unstable();
                    let del_ratio = list.len() as f64 / self.total_rows as f64;
                    if del_ratio > 0.01 && del_ratio < 0.5 {
                        let mut bitmap = RoaringBitmap::new();
                        for &p in list.iter() {
                            bitmap.insert(p);
                        }
                        let mut buf = Vec::new();
                        bitmap.serialize_into(&mut buf).map_err(|_| ()).ok();
                        self.mode = DelMaskMode::Roaring(buf);
                    }
                }
            }
            DelMaskMode::Roaring(data) => {
                let mut bitmap = RoaringBitmap::deserialize_from(data.as_slice()).unwrap_or_else(|_| RoaringBitmap::new());
                bitmap.insert(pos32);
                let mut buf = Vec::new();
                bitmap.serialize_into(&mut buf).map_err(|_| ()).ok();
                self.mode = DelMaskMode::Roaring(buf);
                let del_ratio = bitmap.len() as f64 / self.total_rows as f64;
                if del_ratio > self.compaction_threshold {
                    self.to_full_bitmap();
                }
            }
            DelMaskMode::FullBitmap(data) => {
                let byte_idx = pos as usize / 8;
                let bit_idx = pos as usize % 8;
                if byte_idx >= data.len() {
                    data.resize(byte_idx + 1, 0);
                }
                data[byte_idx] |= 1 << bit_idx;
            }
        }
    }

    fn to_full_bitmap(&mut self) {
        let byte_count = (self.total_rows as usize + 7) / 8;
        let mut full = vec![0u8; byte_count];

        match &self.mode {
            DelMaskMode::SkipList(list) => {
                for &pos in list {
                    let idx = pos as usize / 8;
                    let bit = pos as usize % 8;
                    if idx < byte_count {
                        full[idx] |= 1 << bit;
                    }
                }
            }
            DelMaskMode::Roaring(data) => {
                let bitmap = RoaringBitmap::deserialize_from(data.as_slice()).unwrap_or_default();
                for pos in bitmap {
                    let idx = pos as usize / 8;
                    let bit = pos as usize % 8;
                    if idx < byte_count {
                        full[idx] |= 1 << bit;
                    }
                }
            }
            DelMaskMode::FullBitmap(old) => {
                full[..old.len().min(byte_count)].copy_from_slice(&old[..old.len().min(byte_count)]);
            }
            DelMaskMode::Empty => {}
        }
        self.mode = DelMaskMode::FullBitmap(full);
    }

    pub fn deleted_count(&self) -> u64 {
        match &self.mode {
            DelMaskMode::Empty => 0,
            DelMaskMode::SkipList(list) => list.len() as u64,
            DelMaskMode::Roaring(data) => {
                RoaringBitmap::deserialize_from(data.as_slice())
                    .map(|b| b.len())
                    .unwrap_or(0)
            }
            DelMaskMode::FullBitmap(data) => {
                data.iter().map(|&b| b.count_ones() as u64).sum()
            }
        }
    }

    pub fn del_ratio(&self) -> f64 {
        if self.total_rows == 0 { return 0.0; }
        self.deleted_count() as f64 / self.total_rows as f64
    }

    pub fn needs_compaction(&self) -> bool {
        self.del_ratio() > self.compaction_threshold
    }

    pub fn add_deletes(&mut self, positions: &[u64]) {
        for &pos in positions {
            self.add_delete(pos);
        }
    }

    pub fn deleted_positions(&self) -> Box<dyn Iterator<Item = u64> + '_> {
        match &self.mode {
            DelMaskMode::Empty => Box::new(std::iter::empty()),
            DelMaskMode::SkipList(list) => Box::new(list.iter().map(|&p| p as u64)),
            DelMaskMode::Roaring(data) => {
                let bitmap = RoaringBitmap::deserialize_from(data.as_slice())
                    .unwrap_or_else(|_| RoaringBitmap::new());
                Box::new(bitmap.into_iter().map(|p| p as u64))
            }
            DelMaskMode::FullBitmap(data) => {
                Box::new(data.iter().enumerate().flat_map(|(byte_idx, &byte)| {
                    let base = (byte_idx * 8) as u64;
                    (0..8u64).filter(move |i| (byte & (1 << i)) != 0).map(move |i| base + i)
                }))
            }
        }
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let data = encode(self)?;
        std::fs::write(path, data)?;
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self> {
        let data = std::fs::read(path)?;
        let mask: DelMask = decode(&data)?;
        Ok(mask)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_mask() {
        let mask = DelMask::new(100);
        assert!(!mask.is_deleted(5));
        assert_eq!(mask.deleted_count(), 0);
        assert_eq!(mask.del_ratio(), 0.0);
    }

    #[test]
    fn test_skiplist() {
        let mut mask = DelMask::new(10000);
        mask.add_delete(10);
        mask.add_delete(20);
        assert!(mask.is_deleted(10));
        assert!(mask.is_deleted(20));
        assert!(!mask.is_deleted(30));
        assert_eq!(mask.deleted_count(), 2);
    }

    #[test]
    fn test_roaring_conversion() {
        let mut mask = DelMask::new(1000);
        for i in 0..50 {
            mask.add_delete(i * 2);
        }
        assert!(matches!(mask.mode, DelMaskMode::Roaring(_)));
    }

    #[test]
    fn test_full_bitmap() {
        let mut mask = DelMask::new(100);
        for i in 0..60 {
            mask.add_delete(i);
        }
        assert!(matches!(mask.mode, DelMaskMode::FullBitmap(_)));
        assert!(mask.needs_compaction());
    }

    #[test]
    fn test_serialization() {
        let mut mask = DelMask::new(1000);
        mask.add_delete(10);
        mask.add_delete(20);
        mask.add_delete(30);

        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("del_mask_test.bin");
        mask.save(&path).unwrap();

        let loaded = DelMask::load(&path).unwrap();
        assert!(loaded.is_deleted(10));
        assert!(loaded.is_deleted(20));
        assert!(loaded.is_deleted(30));
        assert!(!loaded.is_deleted(40));

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_is_deleted_out_of_bounds() {
        let mask = DelMask::new(100);
        assert!(!mask.is_deleted(100));
        assert!(!mask.is_deleted(200));
        assert!(!mask.is_deleted(u64::MAX));
    }

    #[test]
    fn test_is_deleted_zero_rows() {
        let mask = DelMask::new(0);
        assert!(!mask.is_deleted(0));
        assert!(!mask.is_deleted(100));
    }

    #[test]
    fn test_add_delete_empty_to_skiplist() {
        let mut mask = DelMask::new(10000);
        assert!(matches!(mask.mode, DelMaskMode::Empty));
        mask.add_delete(5);
        assert!(matches!(mask.mode, DelMaskMode::SkipList(_)));
        assert!(mask.is_deleted(5));
    }

    #[test]
    fn test_add_delete_skiplist_to_roaring() {
        let mut mask = DelMask::new(10000);
        for i in 0..100 {
            mask.add_delete(i);
        }
        for i in 100..150 {
            mask.add_delete(i);
        }
        assert!(matches!(mask.mode, DelMaskMode::Roaring(_)));
    }

    #[test]
    fn test_add_delete_roaring_to_full_bitmap() {
        let mut mask = DelMask::new(100);
        for i in 0..50 {
            mask.add_delete(i);
        }
        assert!(matches!(mask.mode, DelMaskMode::Roaring(_)));
        for i in 50..60 {
            mask.add_delete(i);
        }
        assert!(matches!(mask.mode, DelMaskMode::FullBitmap(_)));
    }

    #[test]
    fn test_add_delete_duplicate_no_op() {
        let mut mask = DelMask::new(1000);
        mask.add_delete(50);
        mask.add_delete(50);
        assert!(mask.is_deleted(50));
        assert_eq!(mask.deleted_count(), 1);
    }

    #[test]
    fn test_deleted_positions_empty() {
        let mask = DelMask::new(100);
        let positions: Vec<u64> = mask.deleted_positions().collect();
        assert!(positions.is_empty());
    }

    #[test]
    fn test_deleted_positions_skiplist() {
        let mut mask = DelMask::new(1000);
        mask.add_delete(5);
        mask.add_delete(10);
        mask.add_delete(15);
        let positions: Vec<u64> = mask.deleted_positions().collect();
        assert_eq!(positions.len(), 3);
        assert!(positions.contains(&5));
        assert!(positions.contains(&10));
        assert!(positions.contains(&15));
    }

    #[test]
    fn test_deleted_positions_roaring() {
        let mut mask = DelMask::new(1000);
        for i in (0..1000).step_by(2) {
            mask.add_delete(i);
        }
        let positions: Vec<u64> = mask.deleted_positions().collect();
        assert_eq!(positions.len(), 500);
        assert!(positions.contains(&0));
        assert!(positions.contains(&998));
    }

    #[test]
    fn test_deleted_positions_full_bitmap() {
        let mut mask = DelMask::new(100);
        for i in 0..60 {
            mask.add_delete(i);
        }
        let positions: Vec<u64> = mask.deleted_positions().collect();
        assert_eq!(positions.len(), 60);
        assert!(positions.contains(&0));
        assert!(positions.contains(&59));
    }

    #[test]
    fn test_add_deletes_batch() {
        let mut mask = DelMask::new(1000);
        mask.add_deletes(&[10, 20, 30, 40, 50]);
        assert_eq!(mask.deleted_count(), 5);
        assert!(mask.is_deleted(10));
        assert!(mask.is_deleted(20));
        assert!(mask.is_deleted(50));
        assert!(!mask.is_deleted(60));
    }

    #[test]
    fn test_del_ratio_zero_total_rows() {
        let mask = DelMask::new(0);
        assert_eq!(mask.del_ratio(), 0.0);
    }

    #[test]
    fn test_del_ratio_all_deleted() {
        let mut mask = DelMask::new(10);
        for i in 0..10 {
            mask.add_delete(i);
        }
        assert!((mask.del_ratio() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_needs_compaction_threshold() {
        let mut mask = DelMask::new(100);
        for i in 0..60 {
            mask.add_delete(i);
        }
        assert!(mask.needs_compaction());
    }

    #[test]
    fn test_needs_compaction_below_threshold() {
        let mut mask = DelMask::new(100);
        for i in 0..30 {
            mask.add_delete(i);
        }
        assert!(!mask.needs_compaction());
    }

    #[test]
    fn test_full_bitmap_bits_correct() {
        let mut mask = DelMask::new(16);
        for i in 0..16 {
            mask.add_delete(i);
        }
        let positions: Vec<u64> = mask.deleted_positions().collect();
        assert_eq!(positions.len(), 16);
    }

    #[test]
    fn test_to_full_bitmap_explicit() {
        let mut mask = DelMask::new(64);
        for i in 0..32 {
            mask.add_delete(i);
        }
        mask.to_full_bitmap();
        assert!(matches!(mask.mode, DelMaskMode::FullBitmap(_)));
        assert_eq!(mask.deleted_count(), 32);
    }

    #[test]
    fn test_to_full_bitmap_from_empty() {
        let mut mask = DelMask::new(64);
        mask.to_full_bitmap();
        assert!(matches!(mask.mode, DelMaskMode::FullBitmap(ref data) if data.len() == 8));
        assert_eq!(mask.deleted_count(), 0);
    }

    #[test]
    fn test_mode_is_empty() {
        let mask = DelMask::new(100);
        assert!(matches!(mask.mode, DelMaskMode::Empty));
    }

    #[test]
    fn test_mode_is_skiplist_after_first_delete() {
        let mut mask = DelMask::new(10000);
        mask.add_delete(1);
        assert!(matches!(mask.mode, DelMaskMode::SkipList(_)));
    }

    #[test]
    fn test_mode_after_save_load() {
        let mut mask = DelMask::new(500);
        for i in 0..50 {
            mask.add_delete(i);
        }

        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("del_mask_mode_test.bin");
        mask.save(&path).unwrap();

        let loaded = DelMask::load(&path).unwrap();
        assert_eq!(loaded.deleted_count(), 50);

        std::fs::remove_file(path).ok();
    }
}
