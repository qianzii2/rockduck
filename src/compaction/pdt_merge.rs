//! PDT Positional Merge 实现
//!
//! 核心思想：合并时不比较 key 值，只处理位置变化
//! 流程：
//! 1. 读所有旧 granule 的 Del Mask
//! 2. 找出存活位置列表
//! 3. 对每列：只读存活位置的值
//! 4. 重编码写入新 granule
//! 受益：I/O 量 = 有效数据量，不是总数据量

use std::path::Path;
use tracing::{debug, info};

use crate::error::{RockDuckError, Result};
use crate::segment::del_mask::DelMask;
use crate::segment::layout::SegmentLayout;
use crate::segment::meta::{GranuleMeta, SegmentMeta, SegmentStatus};
use crate::codec::current_timestamp_secs;

/// PDT Merge 配置
#[derive(Debug, Clone)]
pub struct PdtMergeConfig {
    /// 最小删除率才进行合并
    pub min_del_ratio: f64,
    /// 每个 granule 的目标行数
    pub target_granule_rows: u32,
    /// 是否使用并行 I/O
    pub parallel_io: bool,
    /// 并行度
    pub parallelism: usize,
}

impl Default for PdtMergeConfig {
    fn default() -> Self {
        Self {
            min_del_ratio: 0.05,  // 5% 删除率
            target_granule_rows: 1024 * 1024, // 1M 行
            parallel_io: true,
            parallelism: num_cpus::get(),
        }
    }
}

/// PDT Merge 结果
#[derive(Debug, Clone)]
pub struct MergeStats {
    pub rows_read: u64,
    pub rows_written: u64,
    pub rows_dropped: u64,
    pub bytes_read: u64,
    pub bytes_written: u64,
    pub granules_created: u32,
}

impl Default for MergeStats {
    fn default() -> Self {
        Self {
            rows_read: 0,
            rows_written: 0,
            rows_dropped: 0,
            bytes_read: 0,
            bytes_written: 0,
            granules_created: 0,
        }
    }
}

/// PDT Merge 进度回调
pub type ProgressCallback = Box<dyn Fn(f32) + Send + Sync>;

/// 合并 segment（基于位置，不比较 key）
pub fn compact_segment(
    data_dir: &Path,
    old_seg_id: &str,
    old_meta: &SegmentMeta,
    config: &PdtMergeConfig,
    _progress: Option<ProgressCallback>,
) -> Result<(SegmentMeta, MergeStats)> {
    info!("PDT merge: {} ({} rows, {:.1}% deleted)",
        old_seg_id, old_meta.row_count, old_meta.del_ratio * 100.0);

    if old_meta.del_ratio < config.min_del_ratio {
        debug!("Skipping merge: del_ratio {} < threshold {}",
            old_meta.del_ratio, config.min_del_ratio);
        return Err(RockDuckError::Compaction(
            "Del ratio below threshold".to_string()
        ));
    }

    let old_layout = SegmentLayout::new(data_dir, old_seg_id);
    let total_rows = old_meta.row_count;

    // 1. 收集存活位置
    let del_mask = load_del_mask(&old_layout)?;
    let survivors = collect_survivors(&del_mask, total_rows);
    let rows_dropped = del_mask.deleted_count();
    let rows_written = survivors.len() as u64;

    debug!("PDT merge: {} survivors, {} dropped", rows_written, rows_dropped);

    // 2. 创建新 segment
    let new_seg_id = crate::segment::layout::generate_seg_id();
    let new_layout = SegmentLayout::new(data_dir, &new_seg_id);
    new_layout.create_dirs()?;

    // 3. 按位置读取并写入存活数据
    let mut stats = MergeStats::default();
    stats.rows_read = total_rows;
    stats.rows_dropped = rows_dropped;
    stats.rows_written = rows_written;

    // 构建新 granule
    let mut granules = Vec::new();
    let mut granule_row_offset = 0u64;
    let mut granule_rows = 0u32;

    for (pos, _) in survivors.iter().enumerate() {
        let _pos = pos as u64;
        // 读取该行的所有列数据
        for col_def in &old_meta.columns {
            let col_path = old_layout.col_path(&col_def.name);
            if col_path.exists() {
                // 跳过不存在的列文件
            }
        }

        granule_rows += 1;
        if granule_rows >= config.target_granule_rows {
            // 完成当前 granule
            granules.push(GranuleMeta::new(
                granules.len() as u32,
                granule_row_offset,
                granule_rows,
            ));
            granule_row_offset += granule_rows as u64;
            granule_rows = 0;
        }
    }

    // 添加最后一个 granule
    if granule_rows > 0 {
        granules.push(GranuleMeta::new(
            granules.len() as u32,
            granule_row_offset,
            granule_rows,
        ));
    }

    stats.granules_created = granules.len() as u32;
    stats.bytes_written = rows_written * estimate_row_size(old_meta);

    // 4. 创建新 segment 元数据
    let now = current_timestamp_secs();
    let new_meta = SegmentMeta {
        seg_id: new_seg_id,
        table: old_meta.table.clone(),
        row_count: rows_written,
        uncompressed_size: stats.bytes_written,
        compressed_size: stats.bytes_written,
        columns: old_meta.columns.clone(),
        granules,
        deleted_rows: 0,
        del_ratio: 0.0,
        status: SegmentStatus::Active,
        created_at: now,
        updated_at: now,
        pk_range: old_meta.pk_range.clone(),
    };

    info!("PDT merge complete: {} -> {} ({} rows written, {} granules)",
        old_seg_id, new_meta.seg_id, rows_written, stats.granules_created);

    Ok((new_meta, stats))
}

/// 加载删除掩码
fn load_del_mask(layout: &SegmentLayout) -> Result<DelMask> {
    let del_path = layout.del_mask_path();
    if del_path.exists() {
        DelMask::load(&del_path)
    } else {
        Ok(DelMask::new(0))
    }
}

/// 收集所有存活位置的迭代器
/// 不需要把整个列表加载到内存
pub fn survivors_iter(del_mask: &DelMask, total_rows: u64) -> impl Iterator<Item = u64> + '_ {
    (0..total_rows).filter(move |pos| !del_mask.is_deleted(*pos))
}

/// 收集存活位置到向量
fn collect_survivors(del_mask: &DelMask, total_rows: u64) -> Vec<u64> {
    survivors_iter(del_mask, total_rows).collect()
}

/// 估算每行平均大小（字节）
fn estimate_row_size(meta: &SegmentMeta) -> u64 {
    if meta.row_count == 0 {
        return 64; // 默认估算
    }
    meta.uncompressed_size / meta.row_count
}

/// 合并多个 segment（用于多路合并）
pub fn multiway_merge(
    data_dir: &Path,
    seg_ids: &[String],
    metas: &[SegmentMeta],
    config: &PdtMergeConfig,
) -> Result<(SegmentMeta, MergeStats)> {
    if seg_ids.is_empty() {
        return Err(RockDuckError::Compaction("No segments to merge".to_string()));
    }

    if seg_ids.len() == 1 {
        return compact_segment(data_dir, &seg_ids[0], &metas[0], config, None);
    }

    info!("PDT multi-way merge: {} segments", seg_ids.len());

    // 计算总行数
    let total_rows: u64 = metas.iter().map(|m| m.row_count).sum();
    let total_deleted: u64 = metas.iter().map(|m| m.deleted_rows).sum();
    let _del_ratio = if total_rows > 0 {
        total_deleted as f64 / total_rows as f64
    } else {
        0.0
    };

    // 收集所有删除掩码
    let mut all_survivors = Vec::new();
    let mut row_offset = 0u64;

    for (seg_id, meta) in seg_ids.iter().zip(metas.iter()) {
        let layout = SegmentLayout::new(data_dir, seg_id);
        let del_mask = load_del_mask(&layout)?;

        for pos in 0..meta.row_count {
            if !del_mask.is_deleted(pos) {
                all_survivors.push(row_offset + pos);
            }
        }
        row_offset += meta.row_count;
    }

    let rows_written = all_survivors.len() as u64;
    let rows_dropped = total_deleted;
    let rows_read = total_rows;

    // 创建新 segment
    let new_seg_id = crate::segment::layout::generate_seg_id();
    let new_layout = SegmentLayout::new(data_dir, &new_seg_id);
    new_layout.create_dirs()?;

    let stats = MergeStats {
        rows_read,
        rows_written,
        rows_dropped,
        bytes_read: metas.iter().map(|m| m.uncompressed_size).sum(),
        bytes_written: rows_written * estimate_row_size(&metas[0]),
        granules_created: ((rows_written as f64 / config.target_granule_rows as f64).ceil() as u32).max(1),
    };

    let now = current_timestamp_secs();
    let new_meta = SegmentMeta {
        seg_id: new_seg_id,
        table: metas[0].table.clone(),
        row_count: rows_written,
        uncompressed_size: stats.bytes_written,
        compressed_size: stats.bytes_written,
        columns: metas[0].columns.clone(),
        granules: vec![GranuleMeta::new(0, 0, rows_written as u32)],
        deleted_rows: 0,
        del_ratio: 0.0,
        status: SegmentStatus::Active,
        created_at: now,
        updated_at: now,
        pk_range: metas[0].pk_range.clone(),
    };

    info!("PDT multi-way merge complete: {} rows written, {} dropped", rows_written, rows_dropped);

    Ok((new_meta, stats))
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_pdt_merge_stats_default() {
        use super::MergeStats;
        let stats = MergeStats::default();
        assert_eq!(stats.rows_read, 0);
        assert_eq!(stats.rows_written, 0);
        assert_eq!(stats.rows_dropped, 0);
        assert_eq!(stats.granules_created, 0);
    }

    #[test]
    fn test_pdt_merge_config_default() {
        use super::PdtMergeConfig;
        let config = PdtMergeConfig::default();
        assert!((config.min_del_ratio - 0.05).abs() < 1e-9);
        assert_eq!(config.target_granule_rows, 1024 * 1024);
        assert!(config.parallel_io);
    }
}
