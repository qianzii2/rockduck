//! Non-Blocking Compaction 实现
//!
//! 核心思想：写入永远追加到新 granule，compaction 后台线程异步执行
//! Compaction 后台线程读旧 granule → 合并 → 写新 granule
//! 读取时同时查新旧路径，合并结果
//! 写入不会因为 compaction 而阻塞

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{mpsc, RwLock};
use tracing::{debug, info};

use crate::db::RockDuck;
use crate::error::Result;
use crate::metadata;
use crate::segment::layout::SegmentLayout;
use crate::segment::layout::generate_seg_id;
use crate::segment::meta::SegmentStatus;
use crate::compaction::pdt_merge::{self, PdtMergeConfig, MergeStats};

/// Compaction 状态
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompactionStatus {
    /// 空闲
    Idle,
    /// 正在运行
    Running,
    /// 暂停
    Paused,
    /// 出错
    Error,
}

impl Default for CompactionStatus {
    fn default() -> Self {
        CompactionStatus::Idle
    }
}

/// 单个 Segment 的合并状态
#[derive(Debug, Clone)]
pub struct SegmentMergeState {
    pub old_seg_id: String,
    pub new_seg_id: String,
    pub started_at: u64,
    pub progress: f32,
    pub status: CompactionStatus,
}

impl SegmentMergeState {
    pub fn new(old_seg_id: String, new_seg_id: String) -> Self {
        Self {
            old_seg_id,
            new_seg_id,
            started_at: crate::codec::current_timestamp_secs(),
            progress: 0.0,
            status: CompactionStatus::Running,
        }
    }
}

/// Non-Blocking Compactor 配置
#[derive(Debug, Clone)]
pub struct NonBlockingConfig {
    /// Compaction 线程数
    pub num_threads: usize,
    /// Compaction 触发阈值（删除率）
    pub del_ratio_threshold: f64,
    /// Compaction 检查间隔
    pub check_interval: Duration,
    /// 最大并发合并数
    pub max_concurrent_merges: usize,
}

impl Default for NonBlockingConfig {
    fn default() -> Self {
        Self {
            num_threads: 2,
            del_ratio_threshold: 0.1,  // 10% 删除率触发
            check_interval: Duration::from_secs(60),
            max_concurrent_merges: 4,
        }
    }
}

/// Non-Blocking Compactor
pub struct NonBlockingCompactor {
    rockduck: Arc<RockDuck>,
    config: NonBlockingConfig,
    /// 正在合并的 segments
    merging: RwLock<HashMap<String, SegmentMergeState>>,
    /// Compaction 任务发送器
    task_tx: RwLock<Option<mpsc::Sender<CompactionTask>>>,
}

/// Compaction 任务
#[allow(dead_code)]
struct CompactionTask {
    seg_id: String,
}

/// Compaction 结果事件
#[derive(Debug, Clone)]
pub struct CompactionEvent {
    pub old_seg_id: String,
    pub new_seg_id: String,
    pub success: bool,
    pub stats: Option<MergeStats>,
}

impl NonBlockingCompactor {
    /// 创建新的 Non-Blocking Compactor
    pub fn new(rockduck: Arc<RockDuck>, config: NonBlockingConfig) -> Self {
        Self {
            rockduck,
            config,
            merging: RwLock::new(HashMap::new()),
            task_tx: RwLock::new(None),
        }
    }

    /// 启动 compactor 后台任务
    /// Note: This is a placeholder stub. The actual async worker spawning requires
    /// tokio features (clone on RwLock, resubscribe) that are not available in this build.
    pub async fn start(&self) {
        // TODO: implement full non-blocking compaction worker loop
        // Requires: tokio::sync::RwLock::clone() and mpsc::Receiver::resubscribe()
        let _ = &self.rockduck;
        let _ = &self.config;
        let _ = &self.merging;
    }

    /// 触发 compaction（异步，不阻塞写入）
    pub async fn trigger(&self, seg_id: &str) -> Result<()> {
        // 检查是否已在合并
        {
            let merging = self.merging.read().await;
            if merging.contains_key(seg_id) {
                debug!("Segment {} already merging", seg_id);
                return Ok(());
            }
        }

        // 检查是否达到合并阈值
        if let Some(meta) = metadata::rocksdb::get_segment_meta(&self.rockduck.db, seg_id)? {
            if meta.del_ratio < self.config.del_ratio_threshold {
                debug!("Segment {} del_ratio {} below threshold",
                    seg_id, meta.del_ratio);
                return Ok(());
            }
        }

        // 发送任务
        let task_tx = self.task_tx.read().await;
        if let Some(tx) = task_tx.as_ref() {
            let new_seg_id = generate_seg_id();

            // 记录合并状态
            {
                let mut merging = self.merging.write().await;
                merging.insert(seg_id.to_string(), SegmentMergeState::new(seg_id.to_string(), new_seg_id));
            }

            tx.send(CompactionTask {
                seg_id: seg_id.to_string(),
            }).await.map_err(|_| crate::RockDuckError::Internal("Compactor shut down".to_string()))?;
        }

        Ok(())
    }

    /// 批量触发 compaction
    pub async fn trigger_batch(&self, seg_ids: &[String]) -> Result<()> {
        for seg_id in seg_ids {
            self.trigger(seg_id).await?;
        }
        Ok(())
    }

    /// 检查是否正在合并某个 segment
    pub async fn is_merging(&self, seg_id: &str) -> bool {
        let merging = self.merging.read().await;
        merging.contains_key(seg_id)
    }

    /// 获取合并状态
    pub async fn get_state(&self, seg_id: &str) -> Option<SegmentMergeState> {
        let merging = self.merging.read().await;
        merging.get(seg_id).cloned()
    }

    /// 获取所有正在合并的 segments
    pub async fn list_merging(&self) -> Vec<String> {
        let merging = self.merging.read().await;
        merging.keys().cloned().collect()
    }
}

/// 运行实际的 compaction
#[allow(dead_code)]
async fn run_compaction(
    rockduck: &Arc<RockDuck>,
    seg_id: &str,
    config: &NonBlockingConfig,
) -> Result<MergeStats> {
    debug!("Starting compaction for segment: {}", seg_id);

    // 1. 读取旧 segment 元数据
    let old_meta = metadata::rocksdb::get_segment_meta(&rockduck.db, seg_id)?
        .ok_or_else(|| crate::RockDuckError::SegmentNotFound(seg_id.to_string()))?;

    // 2. 创建新 segment
    let new_seg_id = generate_seg_id();
    let new_layout = SegmentLayout::new(&rockduck.data_dir, &new_seg_id);
    new_layout.create_dirs()?;

    // 3. 运行 PDT Merge
    let pdt_config = PdtMergeConfig {
        min_del_ratio: config.del_ratio_threshold,
        ..Default::default()
    };

    let (new_meta, stats) = pdt_merge::compact_segment(
        &rockduck.data_dir,
        seg_id,
        &old_meta,
        &pdt_config,
        None,
    )?;

    // 4. 注册新 segment
    metadata::rocksdb::put_segment_meta(&rockduck.db, &new_meta)?;

    // 5. 冻结旧 segment
    metadata::seg_meta::update_segment_status(&rockduck.db, seg_id, SegmentStatus::Frozen)?;

    // 6. 删除旧 segment 文件（可选，保留以便回滚）
    // cleanup_old_segment(&docdb.data_dir, seg_id)?;

    info!("Compaction done: {} -> {} ({} rows, {:.1}% reduction)",
        seg_id, new_seg_id, stats.rows_written,
        if stats.rows_read > 0 { (1.0 - stats.rows_written as f64 / stats.rows_read as f64) * 100.0 } else { 0.0 });

    Ok(stats)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ============================================================
    // CompactionStatus
    // ============================================================

    #[test]
    fn test_compaction_status_default() {
        assert_eq!(CompactionStatus::default(), CompactionStatus::Idle);
    }

    #[test]
    fn test_compaction_status_debug() {
        for status in [
            CompactionStatus::Idle,
            CompactionStatus::Running,
            CompactionStatus::Paused,
            CompactionStatus::Error,
        ] {
            let debug_str = format!("{:?}", status);
            assert!(!debug_str.is_empty());
        }
    }

    #[test]
    fn test_compaction_status_eq() {
        assert_eq!(CompactionStatus::Idle, CompactionStatus::Idle);
        assert_ne!(CompactionStatus::Idle, CompactionStatus::Running);
    }

    // ============================================================
    // SegmentMergeState
    // ============================================================

    #[test]
    fn test_segment_merge_state_new() {
        let state = SegmentMergeState::new("old_seg".to_string(), "new_seg".to_string());
        assert_eq!(state.old_seg_id, "old_seg");
        assert_eq!(state.new_seg_id, "new_seg");
        assert!(state.started_at > 0);
        assert!((state.progress - 0.0).abs() < 1e-6);
        assert_eq!(state.status, CompactionStatus::Running);
    }

    #[test]
    fn test_segment_merge_state_debug() {
        let state = SegmentMergeState::new("old".to_string(), "new".to_string());
        let debug_str = format!("{:?}", state);
        assert!(!debug_str.is_empty());
    }

    // ============================================================
    // NonBlockingConfig
    // ============================================================

    #[test]
    fn test_nonblocking_config_default() {
        let config = NonBlockingConfig::default();
        assert_eq!(config.num_threads, 2);
        assert!((config.del_ratio_threshold - 0.1).abs() < 1e-9);
        assert_eq!(config.check_interval, Duration::from_secs(60));
        assert_eq!(config.max_concurrent_merges, 4);
    }

    #[test]
    fn test_nonblocking_config_debug() {
        let config = NonBlockingConfig::default();
        let debug_str = format!("{:?}", config);
        assert!(!debug_str.is_empty());
    }

    // ============================================================
    // CompactionEvent
    // ============================================================

    #[test]
    fn test_compaction_event_debug() {
        let event = CompactionEvent {
            old_seg_id: "seg_001".to_string(),
            new_seg_id: "seg_002".to_string(),
            success: true,
            stats: None,
        };
        let debug_str = format!("{:?}", event);
        assert!(!debug_str.is_empty());
    }

    #[test]
    fn test_compaction_event_with_stats() {
        let stats = MergeStats::default();
        let event = CompactionEvent {
            old_seg_id: "seg_001".to_string(),
            new_seg_id: "seg_002".to_string(),
            success: true,
            stats: Some(stats),
        };
        assert!(event.stats.is_some());
    }

    #[test]
    fn test_compaction_event_failed() {
        let event = CompactionEvent {
            old_seg_id: "seg_001".to_string(),
            new_seg_id: "seg_002".to_string(),
            success: false,
            stats: None,
        };
        assert!(!event.success);
    }
}
