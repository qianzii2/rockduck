//! MVCC 可见性管理器
//!
//! 负责：
//! - 追踪活跃事务（事务开始时注册，提交/回滚时移除）
//! - 生成一致性快照（用于 Repeatable Read / Snapshot Isolation）
//! - 活跃事务持久化到 RocksDB（崩溃后可恢复）
//!
//! MVCC 设计（Shadow Column 方式）：
//! - 每个数据行记录 created_by_txn 和 deleted_by_txn
//! - 读取时根据快照判断可见性

use std::collections::BTreeSet;
use std::sync::Arc;

use crate::db::TxnId;
use crate::error::Result;

/// 隔离级别
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IsolationLevel {
    ReadCommitted,
    RepeatableRead,
    Snapshot,
}

impl Default for IsolationLevel {
    fn default() -> Self {
        Self::ReadCommitted
    }
}

/// 事务快照
#[derive(Debug, Clone)]
pub struct TxnSnapshot {
    /// 快照ID（创建快照时的最大已提交 txn_id）
    pub snapshot_id: TxnId,
    /// 活跃事务集合（这些事务创建的行对当前快照不可见）
    pub active_txns: BTreeSet<TxnId>,
    /// 隔离级别
    pub isolation: IsolationLevel,
}

impl TxnSnapshot {
    pub fn new(snapshot_id: TxnId, active_txns: BTreeSet<TxnId>, isolation: IsolationLevel) -> Self {
        Self { snapshot_id, active_txns, isolation }
    }
}

/// MVCC 可见性管理器
///
/// 管理活跃事务的生命周期：
/// - begin_txn(txn_id) → 注册到 RocksDB
/// - commit_txn(txn_id) → 从 RocksDB 移除
/// - rollback_txn(txn_id) → 从 RocksDB 移除
/// - snapshot() → 生成一致性快照
pub struct VisibilityManager {
    db: Arc<rocksdb::DB>,
    committed_txn: parking_lot::RwLock<TxnId>,
}

impl VisibilityManager {
    pub fn new(db: Arc<rocksdb::DB>, committed_txn: TxnId) -> Self {
        Self {
            db,
            committed_txn: parking_lot::RwLock::new(committed_txn),
        }
    }

    pub fn set_committed_txn(&self, txn_id: TxnId) {
        *self.committed_txn.write() = txn_id;
    }

    /// 从 RocksDB 加载已提交事务 ID（用于重启后恢复）
    pub fn load_committed_txn(db: &rocksdb::DB) -> Result<TxnId> {
        let cf = match db.cf_handle(crate::metadata::rocksdb::CF_SYS) {
            Some(cf) => cf,
            None => return Ok(0), // CF not available (e.g., in tests or old DBs)
        };
        let key = b"__system__:committed_txn".to_vec();
        match db.get_cf(cf, &key)? {
            Some(data) => {
                let txn_id: TxnId = crate::codec::decode(&data)?;
                Ok(txn_id)
            }
            None => Ok(0),
        }
    }

    /// 事务开始：注册到活跃事务表
    pub fn begin_txn(&self, txn_id: TxnId) -> Result<()> {
        let begin_ts = crate::codec::current_timestamp_millis();
        crate::metadata::rocksdb::add_active_txn(&self.db, txn_id, begin_ts)
    }

    /// 事务提交：从活跃事务表移除
    pub fn commit_txn(&self, txn_id: TxnId) -> Result<()> {
        {
            let mut committed = self.committed_txn.write();
            if txn_id > *committed {
                *committed = txn_id;
            }
        }
        // Persist committed_txn to RocksDB for durability (skip if CF not available, e.g., in tests)
        if let Some(cf) = self.db.cf_handle(crate::metadata::rocksdb::CF_SYS) {
            let key = b"__system__:committed_txn".to_vec();
            let value = {
                let committed = self.committed_txn.read();
                crate::codec::encode(&*committed)?
            };
            self.db.put_cf(cf, &key, &value)?;
        }
        crate::metadata::rocksdb::remove_active_txn(&self.db, txn_id)
    }

    /// 事务回滚：从活跃事务表移除
    pub fn rollback_txn(&self, txn_id: TxnId) -> Result<()> {
        crate::metadata::rocksdb::remove_active_txn(&self.db, txn_id)
    }

    /// 生成一致性快照
    pub fn snapshot(&self, isolation: IsolationLevel) -> Result<TxnSnapshot> {
        let active = crate::metadata::rocksdb::get_active_txns(&self.db)?;
        Ok(TxnSnapshot::new(
            *self.committed_txn.read(),
            active.into_iter().collect(),
            isolation,
        ))
    }

    /// 获取当前最大已提交事务 ID
    pub fn committed_txn(&self) -> TxnId {
        *self.committed_txn.read()
    }

    /// 在指定事务 ID 处创建快照（用于 Time-Travel 查询 AS OF TxnId）
    ///
    /// 与 `snapshot()` 不同，`snapshot_at()` 不查询当前活跃事务，
    /// 而是直接使用传入的 `txn_id` 作为 snapshot_id，
    /// 这样可以在任意历史时间点重建可见性状态。
    ///
    /// 活跃事务集合为空 — 这是 Time-Travel 查询的安全近似：
    /// 重建历史快照时，无法准确知道哪些事务在历史时间点处于活跃状态。
    /// 对于 `AS OF TxnId <committed>` 的查询，只要 committed_txn >= txn_id，
    /// 查询的就是已提交的快照，活跃事务集合为空是安全的。
    pub fn snapshot_at(&self, txn_id: TxnId, isolation: IsolationLevel) -> Result<TxnSnapshot> {
        Ok(TxnSnapshot::new(txn_id, BTreeSet::new(), isolation))
    }

    /// 检查某条数据对给定快照是否可见
    pub fn is_visible(&self, snapshot: &TxnSnapshot, created_txn: TxnId, deleted_txn: Option<TxnId>) -> bool {
        match snapshot.isolation {
            IsolationLevel::ReadCommitted => {
                if created_txn > *self.committed_txn.read() {
                    return false;
                }
                if let Some(del) = deleted_txn {
                    if del <= *self.committed_txn.read() {
                        return false;
                    }
                }
                true
            }
            IsolationLevel::RepeatableRead | IsolationLevel::Snapshot => {
                if created_txn > snapshot.snapshot_id {
                    return false;
                }
                if let Some(del) = deleted_txn {
                    if del <= snapshot.snapshot_id {
                        return false;
                    }
                }
                if snapshot.active_txns.contains(&created_txn) {
                    return false;
                }
                true
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn make_test_db() -> Arc<rocksdb::DB> {
        use tempfile::TempDir;
        use rocksdb::{Options, DB};

        let temp = TempDir::new().unwrap();
        let mut opts = Options::default();
        opts.create_if_missing(true);
        opts.create_missing_column_families(true);

        let db = DB::open_cf(&opts, temp.path(), &["mvcc"]).unwrap();
        Arc::new(db)
    }

    #[test]
    fn test_txn_snapshot_contains_active() {
        let db = make_test_db();
        let mgr = VisibilityManager::new(db.clone(), 0);
        mgr.set_committed_txn(50);

        mgr.begin_txn(100).unwrap();
        let snap = mgr.snapshot(IsolationLevel::RepeatableRead).unwrap();

        assert!(snap.active_txns.contains(&100));
        assert_eq!(snap.snapshot_id, 50);
        assert_eq!(snap.isolation, IsolationLevel::RepeatableRead);
    }

    #[test]
    fn test_is_visible_committed_row() {
        let db = make_test_db();
        let mgr = VisibilityManager::new(db.clone(), 0);

        // Txn 10 creates a row, then commits
        mgr.begin_txn(10).unwrap();
        mgr.commit_txn(10).unwrap();
        mgr.set_committed_txn(10);

        // Snapshot taken after commit
        let snap = mgr.snapshot(IsolationLevel::RepeatableRead).unwrap();

        // Row created by Txn 10 should be visible
        assert!(mgr.is_visible(&snap, 10, None));
    }

    #[test]
    fn test_is_visible_active_txn_row() {
        let db = make_test_db();
        let mgr = VisibilityManager::new(db.clone(), 0);

        // Txn 10 active (has begun but not committed)
        mgr.begin_txn(10).unwrap();
        mgr.set_committed_txn(5); // Committed up to txn 5

        // Snapshot with active txn 10
        let snap = mgr.snapshot(IsolationLevel::RepeatableRead).unwrap();
        assert!(snap.active_txns.contains(&10));

        // Row created by active Txn 10 should NOT be visible to this snapshot
        assert!(!mgr.is_visible(&snap, 10, None));
    }

    #[test]
    fn test_snapshot_isolation() {
        let db = make_test_db();
        let mgr = VisibilityManager::new(db.clone(), 0);

        // Txn 10 creates rows and commits
        mgr.begin_txn(10).unwrap();
        mgr.commit_txn(10).unwrap();
        mgr.set_committed_txn(10);

        // Snapshot taken after Txn 10
        let snap1 = mgr.snapshot(IsolationLevel::RepeatableRead).unwrap();
        assert!(mgr.is_visible(&snap1, 10, None));

        // New active Txn 20 starts
        mgr.begin_txn(20).unwrap();

        // Snapshot1 should still see Txn 10's rows, but NOT Txn 20's
        assert!(mgr.is_visible(&snap1, 10, None));
        assert!(!mgr.is_visible(&snap1, 20, None));

        // New snapshot sees both committed Txn 10 and Txn 20 is invisible
        let snap2 = mgr.snapshot(IsolationLevel::RepeatableRead).unwrap();
        assert!(mgr.is_visible(&snap2, 10, None));
        assert!(!mgr.is_visible(&snap2, 20, None));
    }

    #[test]
    fn test_committed_txn_tracking() {
        let db = make_test_db();
        let mgr = VisibilityManager::new(db.clone(), 0);

        // Txn 10 begins
        mgr.begin_txn(10).unwrap();

        // Before commit, snapshot should have active_txns containing 10
        let snap_before = mgr.snapshot(IsolationLevel::RepeatableRead).unwrap();
        assert!(snap_before.active_txns.contains(&10));
        assert!(!mgr.is_visible(&snap_before, 10, None));

        // After commit
        mgr.commit_txn(10).unwrap();
        mgr.set_committed_txn(10);

        let snap_after = mgr.snapshot(IsolationLevel::RepeatableRead).unwrap();
        assert!(!snap_after.active_txns.contains(&10));
        assert!(mgr.is_visible(&snap_after, 10, None));

        // Rollback: Txn 20
        mgr.begin_txn(20).unwrap();
        let snap_rollback = mgr.snapshot(IsolationLevel::RepeatableRead).unwrap();
        assert!(snap_rollback.active_txns.contains(&20));

        mgr.rollback_txn(20).unwrap();

        let snap_after_rollback = mgr.snapshot(IsolationLevel::RepeatableRead).unwrap();
        assert!(!snap_after_rollback.active_txns.contains(&20));
        assert!(!mgr.is_visible(&snap_after_rollback, 20, None));
    }

    #[test]
    fn test_is_visible_deleted_row() {
        let db = make_test_db();
        let mgr = VisibilityManager::new(db.clone(), 0);

        // Txn 10 creates row (deleted_by = None)
        mgr.begin_txn(10).unwrap();
        mgr.commit_txn(10).unwrap();
        mgr.set_committed_txn(10);

        let snap = mgr.snapshot(IsolationLevel::RepeatableRead).unwrap();
        assert!(mgr.is_visible(&snap, 10, None));

        // Txn 11 deletes the row
        mgr.begin_txn(11).unwrap();
        mgr.commit_txn(11).unwrap();
        mgr.set_committed_txn(11);

        // Snapshot taken after delete
        let snap_after_delete = mgr.snapshot(IsolationLevel::RepeatableRead).unwrap();
        // Row created by 10, deleted by 11 -> both <= snapshot_id=11, so NOT visible
        assert!(!mgr.is_visible(&snap_after_delete, 10, Some(11)));
    }

    #[test]
    fn test_is_visible_read_committed_vs_repeatable_read() {
        let db = make_test_db();
        let mgr = VisibilityManager::new(db.clone(), 0);

        // Txn 10 commits
        mgr.begin_txn(10).unwrap();
        mgr.commit_txn(10).unwrap();
        mgr.set_committed_txn(10);

        // Txn 20 is active (created row but not committed)
        mgr.begin_txn(20).unwrap();

        // ReadCommitted: visible if created_txn <= committed_txn
        let snap_rc = mgr.snapshot(IsolationLevel::ReadCommitted).unwrap();
        // Txn 20 is NOT committed (committed_txn=10, created_txn=20 > 10), so invisible
        assert!(!mgr.is_visible(&snap_rc, 20, None));
        // Txn 10 is committed, so visible
        assert!(mgr.is_visible(&snap_rc, 10, None));

        // RepeatableRead: visible if created_txn <= snapshot_id AND created_txn not in active_txns
        let snap_rr = mgr.snapshot(IsolationLevel::RepeatableRead).unwrap();
        assert!(!mgr.is_visible(&snap_rr, 20, None)); // NOT visible (in active_txns)
        assert!(mgr.is_visible(&snap_rr, 10, None)); // visible (committed, not in active_txns)
    }

    #[test]
    fn test_txn_snapshot_default() {
        let snap = TxnSnapshot::new(0, BTreeSet::new(), IsolationLevel::ReadCommitted);
        assert_eq!(snap.snapshot_id, 0);
        assert!(snap.active_txns.is_empty());
        assert_eq!(snap.isolation, IsolationLevel::ReadCommitted);
    }

    #[test]
    fn test_isolation_level_default() {
        assert_eq!(IsolationLevel::default(), IsolationLevel::ReadCommitted);
    }
}
