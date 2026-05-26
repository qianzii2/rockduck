//! Crash-Safe Compaction — Three-Phase State Machine (nodedb design)
//!
//! 确保 compaction 崩溃后不丢数据：
//!
//! Phase::Idle
//!   ↓ [scheduler 选择目标 segs]
//! Phase::Prepare { new_seg_id, old_seg_ids }
//!   - 锁定旧 segs（写 Compacting 状态到 RocksDB）
//!   - 开始生成新 seg 数据
//!   ↓ [新 seg 数据生成完成]
//! Phase::Commit { new_seg_id, old_seg_ids }
//!   - 原子更新 manifest（所有 seg 状态切换）
//!   ↓ [manifest 更新成功]
//! Phase::Cleanup { new_seg_id, old_seg_ids }
//!   - 删除旧 seg 文件
//!   ↓ [清理完成]
//! Phase::Idle
//!
//! Recovery 时（db.rs init）：
//!   - 扫描所有处于 Compacting 状态的 segment
//!   - 读取 phase 状态
//!   - 按状态继续或回退

use crate::db::TxnId;
use crate::error::Result;
use crate::segment::meta::SegmentStatus;
use bincode_next::{Encode, Decode};

/// Compaction 阶段
#[derive(Debug, Clone, PartialEq, Eq, Encode, Decode)]
pub enum CompactionPhase {
    /// 空闲，无 compaction 进行
    Idle,
    /// Prepare：正在生成新 segment，旧 seg 已被锁定
    Prepare {
        new_seg_id: String,
        old_seg_ids: Vec<String>,
    },
    /// Commit：新 seg 已生成完成，等待 manifest 原子切换
    Commit {
        new_seg_id: String,
        old_seg_ids: Vec<String>,
    },
    /// Cleanup：manifest 已更新，清理旧 seg 文件
    Cleanup {
        new_seg_id: String,
        old_seg_ids: Vec<String>,
    },
}

impl CompactionPhase {
    pub fn is_idle(&self) -> bool {
        matches!(self, Self::Idle)
    }

    pub fn phase_name(&self) -> &'static str {
        match self {
            Self::Idle => "Idle",
            Self::Prepare { .. } => "Prepare",
            Self::Commit { .. } => "Commit",
            Self::Cleanup { .. } => "Cleanup",
        }
    }

    /// 序列化（存入 RocksDB）
    pub fn serialize(&self) -> Vec<u8> {
        bincode_next::encode_to_vec(self, bincode_next::config::standard()).unwrap_or_default()
    }

    /// 反序列化
    pub fn deserialize(data: &[u8]) -> Option<Self> {
        bincode_next::decode_from_slice::<Self, _>(data, bincode_next::config::standard())
            .ok()
            .map(|(v, _)| v)
    }
}

/// Compaction 状态记录（存于 RocksDB）
#[derive(Debug, Clone, Encode, Decode)]
pub struct CompactionState {
    /// 事务 ID（用于 WAL 关联）
    pub txn_id: TxnId,
    /// 当前阶段
    pub phase: CompactionPhase,
    /// 开始时间
    pub start_time: u64,
}

impl CompactionState {
    pub fn new(txn_id: TxnId, phase: CompactionPhase) -> Self {
        Self {
            txn_id,
            phase,
            start_time: crate::codec::current_timestamp_millis(),
        }
    }

    pub fn serialize(&self) -> Vec<u8> {
        bincode_next::encode_to_vec(self, bincode_next::config::standard()).unwrap_or_default()
    }

    pub fn deserialize(data: &[u8]) -> Option<Self> {
        bincode_next::decode_from_slice::<Self, _>(data, bincode_next::config::standard())
            .ok()
            .map(|(v, _)| v)
    }
}

/// CompactionPhase 状态机
pub struct CompactionPhaseMachine {
    state: Option<CompactionState>,
}

impl CompactionPhaseMachine {
    pub fn new() -> Self {
        Self { state: None }
    }

    pub fn is_idle(&self) -> bool {
        self.state.as_ref().map_or(true, |s| s.phase.is_idle())
    }

    pub fn current_phase(&self) -> Option<&CompactionPhase> {
        self.state.as_ref().map(|s| &s.phase)
    }

    pub fn current_state(&self) -> Option<&CompactionState> {
        self.state.as_ref()
    }

    /// 开始 compaction：进入 Prepare 阶段
    pub fn begin_compaction(
        &mut self,
        txn_id: TxnId,
        new_seg_id: String,
        old_seg_ids: Vec<String>,
    ) -> Result<&CompactionPhase> {
        if !self.is_idle() {
            return Err(crate::RockDuckError::Compaction(
                "Compaction already in progress".to_string()
            ).into());
        }

        self.state = Some(CompactionState::new(
            txn_id,
            CompactionPhase::Prepare {
                new_seg_id,
                old_seg_ids,
            },
        ));

        Ok(&self.state.as_ref().unwrap().phase)
    }

    /// 从 Prepare 进入 Commit 阶段
    pub fn commit_phase(&mut self) -> Result<&CompactionPhase> {
        let state = self.state.as_mut().ok_or_else(|| {
            crate::RockDuckError::Compaction("No compaction in progress".to_string())
        })?;

        match &state.phase {
            CompactionPhase::Prepare { new_seg_id, old_seg_ids } => {
                state.phase = CompactionPhase::Commit {
                    new_seg_id: new_seg_id.clone(),
                    old_seg_ids: old_seg_ids.clone(),
                };
                Ok(&state.phase)
            }
            _ => Err(crate::RockDuckError::Compaction(
                format!("Cannot commit from phase {:?}", state.phase)
            ).into()),
        }
    }

    /// 从 Commit 进入 Cleanup 阶段
    pub fn cleanup_phase(&mut self) -> Result<&CompactionPhase> {
        let state = self.state.as_mut().ok_or_else(|| {
            crate::RockDuckError::Compaction("No compaction in progress".to_string())
        })?;

        match &state.phase {
            CompactionPhase::Commit { new_seg_id, old_seg_ids } => {
                state.phase = CompactionPhase::Cleanup {
                    new_seg_id: new_seg_id.clone(),
                    old_seg_ids: old_seg_ids.clone(),
                };
                Ok(&state.phase)
            }
            _ => Err(crate::RockDuckError::Compaction(
                format!("Cannot cleanup from phase {:?}", state.phase)
            ).into()),
        }
    }

    /// 完成 compaction：回到 Idle
    pub fn finish(&mut self) -> Result<()> {
        self.state = None;
        Ok(())
    }

    /// 从 RocksDB 恢复 compaction 状态
    pub fn restore_from_db(&mut self, data: &[u8]) {
        if let Some(state) = CompactionState::deserialize(data) {
            if !state.phase.is_idle() {
                self.state = Some(state);
            }
        }
    }
}

impl Default for CompactionPhaseMachine {
    fn default() -> Self {
        Self::new()
    }
}

/// 获取处于 Compacting 状态的 segment IDs
pub fn get_compacting_segments(
    db: &rocksdb::DB,
) -> Result<Vec<String>> {
    let cf = db.cf_handle("seg_meta")
        .ok_or_else(|| crate::RockDuckError::Metadata("seg_meta not found".to_string()))?;

    let prefix = b"seg:".to_vec();
    let mut compacting = Vec::new();

    let mut iter = db.raw_iterator_cf(&cf);
    iter.seek(&prefix);

    while iter.valid() {
        if let Some(value) = iter.value() {
            if let Ok(meta) = crate::codec::decode::<crate::segment::meta::SegmentMeta>(value) {
                if meta.status == SegmentStatus::Compacting {
                    if let Some(key) = iter.key() {
                        if let Ok(seg_id) = std::str::from_utf8(&key[prefix.len()..]) {
                            compacting.push(seg_id.to_string());
                        }
                    }
                }
            }
        }
        iter.next();
    }

    Ok(compacting)
}

/// Recovery：处理崩溃后的 compaction 状态
pub fn recover_compaction(
    db: &rocksdb::DB,
    phase_machine: &mut CompactionPhaseMachine,
) -> Result<()> {
    let compacting = get_compacting_segments(db)?;
    if compacting.is_empty() {
        return Ok(());
    }

    // 从 RocksDB 读取 phase 状态
    // 状态存储在单独的 key 下
    let phase_state = db.get("compaction_phase")
        .map_err(|e| crate::RockDuckError::RocksDB(e))?
        .and_then(|data| CompactionState::deserialize(&data));

    if let Some(state) = phase_state {
        let phase_name = state.phase.phase_name();
        match &state.phase {
            CompactionPhase::Prepare { new_seg_id, old_seg_ids } => {
                // 新 seg 半成品 → 删除
                tracing::warn!(
                    "Recovery: Prepare phase for compaction of {:?}, rolling back",
                    old_seg_ids
                );
                for old_id in old_seg_ids {
                    let _ = update_segment_status_unsafe(db, &old_id, SegmentStatus::Compactable);
                }
                // 删除新 seg 文件（如果存在）
                let data_dir = db.path()
                    .parent()
                    .map(|p| p.to_path_buf())
                    .unwrap_or_default();
                let new_path = data_dir.join("segments").join("active").join(new_seg_id);
                let _ = std::fs::remove_dir_all(&new_path);
            }
            CompactionPhase::Commit { new_seg_id, old_seg_ids } => {
                // 新 seg 已存在 → 继续 Cleanup
                tracing::warn!(
                    "Recovery: Commit phase for compaction, resuming cleanup"
                );
                phase_machine.state = Some(state);
            }
            CompactionPhase::Cleanup { new_seg_id, old_seg_ids } => {
                // 旧 seg 可能未删除 → 继续删除
                tracing::warn!(
                    "Recovery: Cleanup phase for compaction, resuming"
                );
                phase_machine.state = Some(state);
            }
            CompactionPhase::Idle => {}
        }
    } else {
        // 无 phase 记录但有 Compacting 状态的 segment → 回退
        tracing::warn!(
            "Recovery: Found {} compacting segments without phase state, rolling back",
            compacting.len()
        );
        for seg_id in &compacting {
            let _ = update_segment_status_unsafe(db, seg_id, SegmentStatus::Compactable);
        }
    }

    // 清理 phase 状态 key
    let _ = db.delete("compaction_phase");

    Ok(())
}

/// 不通过元数据锁直接更新 segment 状态（recovery 专用）
fn update_segment_status_unsafe(
    db: &rocksdb::DB,
    seg_id: &str,
    status: SegmentStatus,
) -> Result<()> {
    let cf = db.cf_handle("seg_meta")
        .ok_or_else(|| crate::RockDuckError::Metadata("seg_meta not found".to_string()))?;

    let key = format!("seg:{}", seg_id);
    if let Some(value) = db.get_cf(&cf, key.as_bytes())? {
        let mut meta: crate::segment::meta::SegmentMeta = crate::codec::decode(&value)?;
        meta.status = status;
        meta.updated_at = crate::codec::current_timestamp_secs();
        let new_value = crate::codec::encode(&meta)?;
        db.put_cf(&cf, key.as_bytes(), &new_value)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compaction_phase_idle() {
        let machine = CompactionPhaseMachine::new();
        assert!(machine.is_idle());
    }

    #[test]
    fn test_compaction_phase_transitions() {
        let mut machine = CompactionPhaseMachine::new();

        // Begin
        let phase = machine.begin_compaction(1, "new_seg".to_string(), vec!["old_1".to_string()]).unwrap();
        assert!(matches!(phase, CompactionPhase::Prepare { .. }));
        assert!(!machine.is_idle());

        // Commit
        let phase = machine.commit_phase().unwrap();
        assert!(matches!(phase, CompactionPhase::Commit { .. }));

        // Cleanup
        let phase = machine.cleanup_phase().unwrap();
        assert!(matches!(phase, CompactionPhase::Cleanup { .. }));

        // Finish
        machine.finish().unwrap();
        assert!(machine.is_idle());
    }

    #[test]
    fn test_compaction_phase_serialize() {
        let phase = CompactionPhase::Prepare {
            new_seg_id: "seg_new".to_string(),
            old_seg_ids: vec!["seg_old".to_string()],
        };
        let data = phase.serialize();
        let restored = CompactionPhase::deserialize(&data).unwrap();
        assert_eq!(phase, restored);
    }

    #[test]
    fn test_compaction_state_serialize() {
        let state = CompactionState::new(42, CompactionPhase::Commit {
            new_seg_id: "new".to_string(),
            old_seg_ids: vec!["old1".to_string(), "old2".to_string()],
        });
        let data = state.serialize();
        let restored = CompactionState::deserialize(&data).unwrap();
        assert_eq!(state.txn_id, restored.txn_id);
        assert!(matches!(restored.phase, CompactionPhase::Commit { .. }));
    }

    #[test]
    fn test_phase_name() {
        assert_eq!(CompactionPhase::Idle.phase_name(), "Idle");
        assert!(!CompactionPhase::Idle.is_idle() == false);
        let p = CompactionPhase::Prepare { new_seg_id: "a".into(), old_seg_ids: vec![] };
        assert_eq!(p.phase_name(), "Prepare");
    }

    #[test]
    fn test_double_begin_fails() {
        let mut machine = CompactionPhaseMachine::new();
        machine.begin_compaction(1, "s1".to_string(), vec![]).unwrap();
        let result = machine.begin_compaction(2, "s2".to_string(), vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn test_phase_machine_invalid_transition() {
        let mut machine = CompactionPhaseMachine::new();

        // commit_phase() from Idle state should return error
        let result = machine.commit_phase();
        assert!(result.is_err(), "commit_phase() from Idle should fail");

        // cleanup_phase() from Idle state should return error
        let result = machine.cleanup_phase();
        assert!(result.is_err(), "cleanup_phase() from Idle should fail");

        // finish() from Idle state should be fine (idempotent)
        machine.finish().unwrap();
    }

    #[test]
    fn test_phase_machine_invalid_transition_from_commit() {
        let mut machine = CompactionPhaseMachine::new();
        machine.begin_compaction(1, "new".to_string(), vec!["old".to_string()]).unwrap();

        // commit_phase() from Idle should fail
        let result = machine.commit_phase();
        assert!(result.is_ok(), "commit_phase() from Prepare should succeed");

        // commit_phase() again from Commit should fail
        let result = machine.commit_phase();
        assert!(result.is_err(), "commit_phase() from Commit should fail");
    }

    #[test]
    fn test_phase_machine_full_cycle_serialize() {
        let mut machine = CompactionPhaseMachine::new();

        // Idle -> Prepare
        machine.begin_compaction(1, "new_seg".to_string(), vec!["old_1".to_string(), "old_2".to_string()]).unwrap();
        let phase = machine.current_phase().unwrap();
        let serialized = phase.serialize();
        let restored = CompactionPhase::deserialize(&serialized).unwrap();
        assert_eq!(*phase, restored);

        // Prepare -> Commit
        machine.commit_phase().unwrap();
        let phase = machine.current_phase().unwrap();
        let serialized = phase.serialize();
        let restored = CompactionPhase::deserialize(&serialized).unwrap();
        assert_eq!(*phase, restored);

        // Commit -> Cleanup
        machine.cleanup_phase().unwrap();
        let phase = machine.current_phase().unwrap();
        let serialized = phase.serialize();
        let restored = CompactionPhase::deserialize(&serialized).unwrap();
        assert_eq!(*phase, restored);

        // Cleanup -> Idle
        machine.finish().unwrap();
        let phase = machine.current_phase();
        assert!(phase.is_none());
    }

    #[test]
    fn test_compaction_phase_commit_preserves_ids() {
        let mut machine = CompactionPhaseMachine::new();

        machine.begin_compaction(42, "seg_new".to_string(), vec!["seg_old_a".to_string(), "seg_old_b".to_string()]).unwrap();
        machine.commit_phase().unwrap();

        let state = machine.current_state().unwrap();
        assert_eq!(state.txn_id, 42);

        let phase = machine.current_phase().unwrap();
        match phase {
            CompactionPhase::Commit { new_seg_id, old_seg_ids } => {
                assert_eq!(new_seg_id, "seg_new");
                assert_eq!(old_seg_ids, &["seg_old_a", "seg_old_b"]);
            }
            _ => panic!("expected Commit phase"),
        }

        // After serialize -> deserialize, IDs should be preserved
        let phase_serialized = phase.serialize();
        let phase_restored = CompactionPhase::deserialize(&phase_serialized).unwrap();
        match phase_restored {
            CompactionPhase::Commit { new_seg_id, old_seg_ids } => {
                assert_eq!(new_seg_id, "seg_new");
                assert_eq!(old_seg_ids, vec!["seg_old_a", "seg_old_b"]);
            }
            _ => panic!("expected Commit phase after deserialization"),
        }
    }

    #[test]
    fn test_compaction_state_new_has_timestamp() {
        let state = CompactionState::new(1, CompactionPhase::Idle);
        assert_eq!(state.txn_id, 1);
        assert!(
            state.start_time > 0,
            "start_time should be set to current timestamp"
        );
    }

    #[test]
    fn test_phase_deserialize_invalid_data() {
        let invalid_data = vec![0xFF, 0xFE, 0xFD, 0xFC, 0xFB];
        let result = CompactionPhase::deserialize(&invalid_data);
        assert!(
            result.is_none(),
            "Invalid data should deserialize to None"
        );

        let empty_data = vec![];
        let result = CompactionPhase::deserialize(&empty_data);
        assert!(
            result.is_none(),
            "Empty data should deserialize to None"
        );
    }

    #[test]
    fn test_compaction_state_serialize_all_phases() {
        // Test that CompactionState serializes correctly for all phase variants
        for phase in [
            CompactionPhase::Idle,
            CompactionPhase::Prepare { new_seg_id: "a".into(), old_seg_ids: vec!["b".into()] },
            CompactionPhase::Commit { new_seg_id: "c".into(), old_seg_ids: vec!["d".into(), "e".into()] },
            CompactionPhase::Cleanup { new_seg_id: "f".into(), old_seg_ids: vec![] },
        ] {
            let state = CompactionState::new(99, phase.clone());
            let data = state.serialize();
            let restored = CompactionState::deserialize(&data).unwrap();
            assert_eq!(restored.txn_id, 99);
            assert_eq!(restored.phase, phase);
        }
    }

    #[test]
    fn test_phase_machine_invalid_transition_from_cleanup() {
        let mut machine = CompactionPhaseMachine::new();
        machine.begin_compaction(1, "n".to_string(), vec!["o".to_string()]).unwrap();
        machine.commit_phase().unwrap();
        machine.cleanup_phase().unwrap();

        // cleanup_phase() again from Cleanup should fail
        let result = machine.cleanup_phase();
        assert!(result.is_err(), "cleanup_phase() from Cleanup should fail");
    }
}
