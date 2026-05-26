//! RockDuck 主结构体和入口点

use std::path::Path;
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{info, debug};

use crate::config::RockDuckConfig;
use crate::error::Result;
use crate::metadata;
use crate::read;
use crate::write;
use crate::write::wal::{WalWriter, WalConfig, WalReader, OpPayload};
use crate::mvcc::{VisibilityManager, IsolationLevel};

/// Bloom Filter per segment
type SegmentBloomFilters = HashMap<String, quickbloom::BloomFilter>;

/// Transaction ID 类型
pub type TxnId = u64;

/// RockDuck 主结构体
    pub struct RockDuck {
        /// RocksDB 实例（Arc 共享给 MVCC）
        pub(crate) db: Arc<rocksdb::DB>,
    /// 数据目录
    pub(crate) data_dir: std::path::PathBuf,
    /// 配置
    pub(crate) config: RockDuckConfig,
    /// 事务计数器
    pub(crate) txn_counter: parking_lot::RwLock<u64>,
    /// 每个 segment 的 Bloom Filter 缓存
    pub(crate) segment_bloom_filters: parking_lot::RwLock<SegmentBloomFilters>,
    /// WAL 写入器（崩溃恢复保证）
    pub(crate) wal: Option<WalWriter>,
    /// MVCC 可见性管理器
    pub(crate) mvcc: VisibilityManager,
    /// DeltaStore 管理器（cell-level 更新追踪）
    pub(crate) delta_store: parking_lot::RwLock<crate::segment::delta_store::DeltaStoreManager>,
}

/// Debug impl that avoids printing the RocksDB instance
impl std::fmt::Debug for RockDuck {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RockDuck")
            .field("data_dir", &self.data_dir)
            .field("config", &self.config)
            .finish()
    }
}

impl RockDuck {
    /// 打开或创建 RockDuck 实例
    pub fn open(path: impl Into<std::path::PathBuf>) -> Result<Self> {
        let config = RockDuckConfig::default();
        Self::open_with_config(path, config)
    }

    /// 使用指定配置打开或创建 RockDuck
    pub fn open_with_config(path: impl Into<std::path::PathBuf>, config: RockDuckConfig) -> Result<Self> {
        let data_dir = path.into();

        // 初始化目录结构
        Self::init_directories(&data_dir)?;

        // 初始化 RocksDB
        let db = Arc::new(metadata::rocksdb::init_rocksdb(&data_dir, &config)?);

        // 从 RocksDB 恢复 committed_txn
        let committed_txn = VisibilityManager::load_committed_txn(&db).unwrap_or(0);

        // 初始化 MVCC manager（放在 WAL 恢复之前）
        let mvcc = VisibilityManager::new(Arc::clone(&db), committed_txn);

        // WAL 恢复（崩溃后重放已提交事务）
        if config.enable_wal {
            let committed_from_wal = Self::recover_from_wal(&db, &data_dir)?;
            // 取 WAL 恢复的最大值和已存储值的较大者
            if committed_from_wal > committed_txn {
                mvcc.set_committed_txn(committed_from_wal);
            }
        }

        // 初始化 WAL writer
        let wal = if config.enable_wal {
            let wal_config = WalConfig {
                wal_dir: std::path::PathBuf::from("wal"),
                max_file_size: config.wal_max_file_size,
                enabled: true,
            };
            Some(WalWriter::new(&data_dir, wal_config)?)
        } else {
            None
        };

        info!("RockDuck opened at {:?}", data_dir);

        Ok(Self {
            db,
            data_dir,
            config,
            txn_counter: parking_lot::RwLock::new(0),
            segment_bloom_filters: parking_lot::RwLock::new(HashMap::new()),
            wal,
            mvcc,
            delta_store: parking_lot::RwLock::new(crate::segment::delta_store::DeltaStoreManager::new()),
        })
    }

    /// 从 WAL 恢复已提交事务，返回最大已提交事务 ID
    fn recover_from_wal(db: &rocksdb::DB, data_dir: &Path) -> Result<u64> {
        let reader = WalReader::new(data_dir);
        let files = reader.list_wal_files()?;

        if files.is_empty() {
            debug!("No WAL files found, skipping recovery");
            return Ok(0);
        }

        info!("Found {} WAL files, scanning for committed transactions", files.len());
        let committed = reader.scan_committed_records()?;
        info!("WAL recovery: {} committed operations to replay", committed.len());

        let mut max_committed_txn = 0u64;

        // 重放每条已提交的操作到 RocksDB
        for rec in &committed {
            if rec.txn_id > max_committed_txn {
                max_committed_txn = rec.txn_id;
            }
            match &rec.payload {
                OpPayload::Insert { table, pk, seg_id, granule_id, offset, .. } => {
                    let entry = crate::metadata::IndexEntry::new(
                        seg_id.clone(),
                        *granule_id,
                        *offset,
                        rec.txn_id,
                    );
                    // 双写：同时写入 pk_idx (hash) 和 pk_skiplist (有序)
                    let _ = crate::metadata::pk_skiplist::put_pk_index_double(
                        db, table.as_str(), pk.as_slice(), &entry
                    );
                }
                OpPayload::Delete { table, pk, seg_id, .. } => {
                    if let Some(mut meta) = metadata::rocksdb::get_segment_meta(db, seg_id.as_str())? {
                        meta.update_del_stats(1);
                        let _ = metadata::rocksdb::put_segment_meta(db, &meta);
                    }
                    // 双删：同时删除 pk_idx 和 pk_skiplist
                    let _ = crate::metadata::pk_skiplist::delete_pk_index_double(
                        db, table.as_str(), pk.as_slice()
                    );
                }
                OpPayload::Update { .. } => {
                    // 更新暂不重放（依赖 UpdMask）
                }
                OpPayload::Begin | OpPayload::Commit | OpPayload::Rollback => {
                    // 这些不重放
                }
            }
        }

        info!("WAL recovery complete, max committed txn_id={}", max_committed_txn);
        Ok(max_committed_txn)
    }

    /// 初始化目录结构
    fn init_directories(data_dir: &Path) -> Result<()> {
        // 创建必要的目录
        let dirs = [
            data_dir,
            &data_dir.join("segments"),
            &data_dir.join("segments").join("active"),
            &data_dir.join("segments").join("immutable"),
            &data_dir.join("wal"),
            &data_dir.join("meta"),
            &data_dir.join("temp"),
        ];

        for dir in &dirs {
            if !dir.exists() {
                std::fs::create_dir_all(dir)?;
                info!("Created directory: {:?}", dir);
            }
        }

        Ok(())
    }

    /// 获取下一个事务 ID
    pub fn next_txn_id(&self) -> TxnId {
        let mut counter = self.txn_counter.write();
        *counter += 1;
        *counter
    }

    // ================== 事务接口 ==================

    /// 开始一个事务：注册到活跃事务表，写 WAL Begin
    pub fn begin_txn(&self) -> Result<TxnId> {
        let txn_id = self.next_txn_id();

        // 注册到 MVCC 活跃事务表
        self.mvcc.begin_txn(txn_id)?;

        // 写 WAL Begin
        if let Some(ref wal) = self.wal {
            wal.append(crate::write::wal::OpType::Begin, txn_id, &crate::write::wal::OpPayload::Begin)?;
        }

        Ok(txn_id)
    }

    /// 提交一个事务：写 WAL Commit，移除活跃事务
    pub fn commit_txn(&self, txn_id: TxnId) -> Result<()> {
        // 写 WAL Commit
        if let Some(ref wal) = self.wal {
            wal.append(crate::write::wal::OpType::Commit, txn_id, &crate::write::wal::OpPayload::Commit)?;
            wal.flush()?;
        }

        // 更新 MVCC committed 事务
        self.mvcc.commit_txn(txn_id)?;

        Ok(())
    }

    /// 回滚一个事务：写 WAL Rollback，移除活跃事务
    pub fn rollback_txn(&self, txn_id: TxnId) -> Result<()> {
        // 写 WAL Rollback
        if let Some(ref wal) = self.wal {
            wal.append(crate::write::wal::OpType::Rollback, txn_id, &crate::write::wal::OpPayload::Rollback)?;
            wal.flush()?;
        }

        // 移除活跃事务
        self.mvcc.rollback_txn(txn_id)?;

        Ok(())
    }

    /// 获取一致性快照（用于读取）
    pub fn snapshot(&self, isolation: IsolationLevel) -> Result<crate::mvcc::TxnSnapshot> {
        self.mvcc.snapshot(isolation)
    }

    /// 在指定事务 ID 处创建快照（用于 Time-Travel 查询）
    pub fn snapshot_at(&self, txn_id: TxnId, isolation: IsolationLevel) -> Result<crate::mvcc::TxnSnapshot> {
        self.mvcc.snapshot_at(txn_id, isolation)
    }

    /// 检查数据是否对给定快照可见
    pub fn is_visible(&self, snapshot: &crate::mvcc::TxnSnapshot, created_txn: TxnId, deleted_txn: Option<TxnId>) -> bool {
        self.mvcc.is_visible(snapshot, created_txn, deleted_txn)
    }

    // ================== 读取接口 ==================

    /// 点查：通过主键获取记录
    pub fn get(&self, table: &str, pk: &[u8]) -> Result<Option<read::RecordBatch>> {
        tracing::debug!("get called: table={}, pk={:?}", table, pk);
        read::point_get::get(self, table, pk)
    }

    /// 范围扫描
    pub fn scan(
        &self,
        table: &str,
        pk_range: Option<(Vec<u8>, Vec<u8>)>,
        filter: Option<&str>,
    ) -> Result<Vec<read::RecordBatch>> {
        read::scan::scan(self, table, pk_range, filter)
    }

    /// Time-Travel 点查：在指定事务 ID 的快照下查询
    pub fn get_as_of(&self, table: &str, pk: &[u8], txn_id: TxnId) -> Result<Option<read::RecordBatch>> {
        read::point_get::get_as_of(self, table, pk, txn_id)
    }

    /// Time-Travel 范围扫描：在指定事务 ID 的快照下扫描
    pub fn scan_as_of(
        &self,
        table: &str,
        txn_id: TxnId,
        pk_range: Option<(Vec<u8>, Vec<u8>)>,
        filter: Option<&str>,
    ) -> Result<Vec<read::RecordBatch>> {
        read::scan::scan_as_of(self, table, txn_id, pk_range, filter)
    }

    // ================== 写入接口 ==================

    /// 插入单条记录
    pub fn insert(
        &self,
        table: &str,
        pk: &[u8],
        columns: &std::collections::HashMap<String, arrow_array::ArrayRef>,
    ) -> Result<TxnId> {
        let txn_id = write::insert::insert(self, table, pk, columns)?;
        self.mvcc.commit_txn(txn_id)?;
        Ok(txn_id)
    }

    /// 批量插入
    pub fn insert_batch(
        &self,
        table: &str,
        pks: &[Vec<u8>],
        columns: &std::collections::HashMap<String, arrow_array::ArrayRef>,
    ) -> Result<TxnId> {
        let txn_id = write::insert::insert_batch(self, table, pks, columns)?;
        self.mvcc.commit_txn(txn_id)?;
        Ok(txn_id)
    }

    /// 删除记录
    pub fn delete(&self, table: &str, pk: &[u8]) -> Result<TxnId> {
        let txn_id = write::insert::delete(self, table, pk)?;
        self.mvcc.commit_txn(txn_id)?;
        Ok(txn_id)
    }

    /// 更新记录
    pub fn update(
        &self,
        table: &str,
        pk: &[u8],
        columns: &std::collections::HashMap<String, arrow_array::ArrayRef>,
    ) -> Result<TxnId> {
        let txn_id = write::insert::update(self, table, pk, columns)?;
        self.mvcc.commit_txn(txn_id)?;
        Ok(txn_id)
    }

    // ================== 元数据接口 ==================

    /// 获取表统计信息
    pub fn get_table_stats(&self, table: &str) -> Result<Option<metadata::TableStats>> {
        metadata::rocksdb::get_table_stats(&self.db, table)
    }

    /// 获取 segment 列表
    pub fn list_segments(&self, table: &str) -> Result<Vec<String>> {
        metadata::seg_meta::list_table_segments(&self.db, table)
    }

    /// 获取 segment 元数据
    pub fn get_segment_meta(&self, seg_id: &str) -> Result<Option<metadata::SegmentMeta>> {
        metadata::rocksdb::get_segment_meta(&self.db, seg_id)
    }

    /// Freeze a segment so its data is read via mmap instead of BufReader.
    pub fn freeze_segment(&self, seg_id: &str) -> Result<()> {
        metadata::seg_meta::update_segment_status(&self.db, seg_id, crate::segment::meta::SegmentStatus::Frozen)
    }

    // ================== 工具接口 ==================

    /// Flush all pending data.
    pub fn flush(&self) -> Result<()> {
        // Flush RocksDB
        self.db.flush()?;
        Ok(())
    }

    /// Get database information.
    pub fn get_info(&self) -> RockDuckInfo {
        RockDuckInfo {
            data_dir: self.data_dir.clone(),
            config: self.config.clone(),
            txn_counter: *self.txn_counter.read(),
        }
    }

    // ================== Iceberg Export ==================

    /// Export the current table state as an Iceberg v2 table.
    ///
    /// Produces spec-compliant Iceberg v2 artifacts:
    ///   - `metadata/v{N}.metadata.json` — TableMetadata JSON
    ///   - `metadata/snap-{id}-{seq}-{uuid8}.avro` — manifest-list Avro
    ///   - `data/segments/{seg_id}/{col}.vortex` — Vortex (Arrow IPC) data files
    ///   - `version-hint.text` — "2"
    ///
    /// # Arguments
    /// * `table` — table name to export
    /// * `target_dir` — output directory for the Iceberg table
    /// * `snapshot_id` — optional snapshot ID (auto-increments if None)
    ///
    /// # Returns
    /// Path to the written `v{N}.metadata.json` file.
    ///
    /// # Example
    /// ```ignore
    /// let metadata_path = rockduck.export_iceberg("users", "/tmp/iceberg_users", None).await?;
    /// // DuckDB: INSTALL vortex; LOAD vortex;
    /// // SELECT * FROM read_vortex('/tmp/iceberg_users/segments/*/*.vortex');
    /// ```
    pub async fn export_iceberg(
        &self,
        table: &str,
        target_dir: impl AsRef<std::path::Path>,
        snapshot_id: Option<i64>,
    ) -> Result<std::path::PathBuf> {
        crate::iceberg::export::export_to_iceberg(self, table, target_dir, snapshot_id).await
    }

    /// Get the current Iceberg snapshot ID, if an export has been performed.
    pub fn iceberg_snapshot_id(&self) -> Result<Option<i64>> {
        let manifest = crate::iceberg::catalog::load_manifest(&self.db)?;
        Ok(manifest.map(|m| m.snapshot_id))
    }

    /// Freeze a segment and optionally export it as Iceberg.
    ///
    /// This is a convenience method that freezes the segment, updates the native
    /// Iceberg manifest in RocksDB, and returns the updated snapshot ID.
    ///
    /// To produce spec-compliant Iceberg files, call `export_iceberg()` separately.
    pub fn freeze_for_iceberg(&self, seg_id: &str) -> Result<Option<i64>> {
        crate::iceberg::catalog::update_iceberg_manifest_on_freeze(&self.db, seg_id)
    }
}

/// RockDuck 信息
#[derive(Debug)]
pub struct RockDuckInfo {
    pub data_dir: std::path::PathBuf,
    pub config: RockDuckConfig,
    pub txn_counter: TxnId,
}

impl std::fmt::Display for RockDuck {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "RockDuck(data_dir={:?}, config={:?})",
            self.data_dir, self.config
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Transaction ID sequence ----

    #[test]
    fn test_next_txn_id_sequence() {
        let temp_dir = tempfile::tempdir().unwrap();
        let db = RockDuck::open(temp_dir.path()).unwrap();

        let ids: Vec<u64> = (0..5).map(|_| db.next_txn_id()).collect();
        assert_eq!(ids, vec![1, 2, 3, 4, 5]);
    }

    // ---- Configuration preservation ----

    #[test]
    fn test_rockduck_config_preserved() {
        let temp_dir = tempfile::tempdir().unwrap();
        let config = RockDuckConfig {
            data_dir: temp_dir.path().to_path_buf(),
            granule_size: 999999,
            segment_target_size: 888888,
            num_threads: 7,
            enable_bloom_filter: false,
            bloom_filter_fpp: 0.05,
            enable_zone_map: false,
            enable_compression: false,
            compression_algorithm: Some("zstd".to_string()),
            enable_wal: false,
            wal_max_file_size: 64 * 1024 * 1024,
        };

        let db = RockDuck::open_with_config(temp_dir.path(), config.clone()).unwrap();
        let info = db.get_info();

        assert_eq!(info.config.granule_size, 999999);
        assert_eq!(info.config.segment_target_size, 888888);
        assert_eq!(info.config.num_threads, 7);
        assert!(!info.config.enable_bloom_filter);
        assert!((info.config.bloom_filter_fpp - 0.05).abs() < 1e-9);
        assert!(!info.config.enable_zone_map);
        assert!(!info.config.enable_compression);
        assert_eq!(info.config.compression_algorithm, Some("zstd".to_string()));
    }

    #[test]
    fn test_init_directories_creates_all_subdirs() {
        let temp_dir = tempfile::tempdir().unwrap();
        let data_dir = temp_dir.path();

        let config = RockDuckConfig {
            data_dir: data_dir.to_path_buf(),
            ..RockDuckConfig::default()
        };
        let _db = RockDuck::open_with_config(data_dir, config).unwrap();

        assert!(data_dir.exists());
        assert!(data_dir.join("segments").exists());
        assert!(data_dir.join("segments").join("active").exists());
        assert!(data_dir.join("segments").join("immutable").exists());
        assert!(data_dir.join("wal").exists());
        assert!(data_dir.join("meta").exists());
        assert!(data_dir.join("temp").exists());
    }

    #[test]
    fn test_get_info_returns_all_fields() {
        let temp_dir = tempfile::tempdir().unwrap();
        let config = RockDuckConfig {
            data_dir: temp_dir.path().to_path_buf(),
            granule_size: 1_000_000,
            segment_target_size: 50_000_000,
            num_threads: 8,
            enable_bloom_filter: true,
            bloom_filter_fpp: 0.001,
            enable_zone_map: true,
            enable_compression: true,
            compression_algorithm: Some("zstd".to_string()),
            enable_wal: true,
            wal_max_file_size: 128 * 1024 * 1024,
        };
        let db = RockDuck::open_with_config(temp_dir.path(), config.clone()).unwrap();

        db.next_txn_id();
        db.next_txn_id();

        let info = db.get_info();
        assert_eq!(info.data_dir, temp_dir.path());
        assert_eq!(info.config.granule_size, 1_000_000);
        assert_eq!(info.config.segment_target_size, 50_000_000);
        assert_eq!(info.config.num_threads, 8);
        assert!(info.config.enable_bloom_filter);
        assert!((info.config.bloom_filter_fpp - 0.001).abs() < 1e-6);
        assert!(info.config.enable_zone_map);
        assert!(info.config.enable_compression);
        assert_eq!(info.txn_counter, 2);
    }

    // ---- Default configuration values ----

    #[test]
    fn test_rockduck_default_config() {
        let temp_dir = tempfile::tempdir().unwrap();
        let db = RockDuck::open(temp_dir.path()).unwrap();
        let info = db.get_info();

        let defaults = RockDuckConfig::default();
        assert_eq!(info.config.granule_size, defaults.granule_size);
        assert_eq!(info.config.segment_target_size, defaults.segment_target_size);
        assert_eq!(info.config.num_threads, defaults.num_threads);
        assert_eq!(info.config.enable_bloom_filter, defaults.enable_bloom_filter);
    }
}
