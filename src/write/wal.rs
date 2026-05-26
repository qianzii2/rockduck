//! Write-Ahead Log (WAL) 实现
//!
//! 基于 32KiB block + CRC32 校验的顺序写入日志。
//! 支持事务边界记录、崩溃恢复、WAL rotation。

use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write as IoWrite, BufWriter};
use std::path::{Path, PathBuf};
use parking_lot::RwLock;
use crc32fast::Hasher;
use tracing::{debug, info, warn};
use bincode_next::{Encode, Decode};

use crate::error::Result;
use crate::codec::{encode, decode};

/// WAL block 大小（32KiB）
pub const WAL_BLOCK_SIZE: usize = 32 * 1024;

/// WAL block header 大小
pub const WAL_BLOCK_HEADER_SIZE: usize = 16;

/// WAL 最大单条记录大小
const WAL_MAX_RECORD_SIZE: usize = 1024 * 1024;

/// WAL 操作类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Encode, Decode)]
pub enum OpType {
    Begin = 0x01,
    Insert = 0x02,
    Delete = 0x03,
    Update = 0x04,
    Commit = 0x10,
    Rollback = 0x11,
}

impl OpType {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0x01 => Some(Self::Begin),
            0x02 => Some(Self::Insert),
            0x03 => Some(Self::Delete),
            0x04 => Some(Self::Update),
            0x10 => Some(Self::Commit),
            0x11 => Some(Self::Rollback),
            _ => None,
        }
    }
}

/// WAL 记录载荷
#[derive(Debug, Clone, Encode, Decode)]
pub enum OpPayload {
    Begin,
    Insert {
        table: String,
        pk: Vec<u8>,
        columns: Vec<(String, Vec<u8>)>,
        seg_id: String,
        granule_id: u32,
        offset: u32,
    },
    Delete {
        table: String,
        pk: Vec<u8>,
        seg_id: String,
        offset: u32,
    },
    Update {
        table: String,
        pk: Vec<u8>,
        columns: Vec<(String, Vec<u8>)>,
        seg_id: String,
        offset: u32,
    },
    Commit,
    Rollback,
}

/// WAL 记录（解析后的内存表示）
#[derive(Debug, Clone)]
pub struct WalRecord {
    pub op_type: OpType,
    pub txn_id: u64,
    pub payload: OpPayload,
}

/// WAL block header（16 bytes）
#[repr(C)]
struct WalBlockHeader {
    block_seq: u64,
    used_bytes: u32,
    header_crc: u32,
}

impl WalBlockHeader {
    fn new(block_seq: u64, used_bytes: u32) -> Self {
        let header_crc = Self::calc_header_crc(block_seq, used_bytes);
        Self { block_seq, used_bytes, header_crc }
    }

    fn calc_header_crc(block_seq: u64, used_bytes: u32) -> u32 {
        let mut h = Hasher::new();
        h.update(&block_seq.to_le_bytes());
        h.update(&used_bytes.to_le_bytes());
        h.finalize()
    }

    fn is_zeroed(&self) -> bool {
        self.block_seq == 0 && self.used_bytes == 0 && self.header_crc == 0
    }

    fn is_valid(&self) -> bool {
        self.header_crc == Self::calc_header_crc(self.block_seq, self.used_bytes)
            && self.used_bytes <= (WAL_BLOCK_SIZE - WAL_BLOCK_HEADER_SIZE) as u32
    }
}

/// WAL 配置
#[derive(Debug, Clone)]
pub struct WalConfig {
    pub wal_dir: PathBuf,
    pub max_file_size: u64,
    pub enabled: bool,
}

impl Default for WalConfig {
    fn default() -> Self {
        Self {
            wal_dir: PathBuf::from("wal"),
            max_file_size: 128 * 1024 * 1024,
            enabled: true,
        }
    }
}

/// WAL 写入器
///
/// 每个 WAL 文件由连续的 32KiB blocks 组成：
///   block = [16-byte header] [up to 32752 bytes of records]
///
/// append() 流程：
///   1. 计算 record 大小
///   2. 如果当前 block 不够，填零并 start_new_block()
///   3. start_new_block()：更新旧 block header，刷盘，打开新文件
///   4. 写入 record
///   5. 更新 block_used
///   6. 检查 rotation
pub struct WalWriter {
    wal_dir: PathBuf,
    file: RwLock<Option<BufWriter<File>>>,
    file_path: RwLock<Option<PathBuf>>,
    block_seq: RwLock<u64>,
    /// 当前 block 内已用字节数
    block_used: RwLock<u32>,
    /// 当前 block 在文件中的起始偏移（用于更新 header）
    block_file_offset: RwLock<u64>,
    /// 当前文件已刷到磁盘的最大偏移（用于 recovery）
    flushed_offset: RwLock<u64>,
    config: WalConfig,
    pub committed_txn: RwLock<u64>,
}

impl WalWriter {
    pub fn new(data_dir: &Path, config: WalConfig) -> Result<Self> {
        let wal_dir = if config.wal_dir.is_absolute() {
            config.wal_dir.clone()
        } else {
            data_dir.join(&config.wal_dir)
        };
        if !wal_dir.exists() {
            std::fs::create_dir_all(&wal_dir)?;
        }

        let w = Self {
            wal_dir,
            file: RwLock::new(None),
            file_path: RwLock::new(None),
            block_seq: RwLock::new(0),
            block_used: RwLock::new(0),
            block_file_offset: RwLock::new(0),
            flushed_offset: RwLock::new(0),
            config,
            committed_txn: RwLock::new(0),
        };
        w.start_new_file()?;
        Ok(w)
    }

    /// 开始一个新的 WAL 文件
    fn start_new_file(&self) -> Result<()> {
        if let Some(ref mut f) = *self.file.write() {
            f.flush()?;
        }
        *self.file.write() = None;
        *self.file_path.write() = None;

        let seq = *self.block_seq.read();
        let filename = format!("wal_{:06}.bin", seq);
        let path = self.wal_dir.join(&filename);

        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)?;

        let file_size = file.metadata()?.len();
        *self.file.write() = Some(BufWriter::with_capacity(WAL_BLOCK_SIZE, file));
        *self.file_path.write() = Some(path);
        *self.block_used.write() = 0;
        *self.block_file_offset.write() = file_size;
        *self.flushed_offset.write() = file_size;

        debug!("Opened WAL file: wal_{:06}.bin (size={})", seq, file_size);
        Ok(())
    }

    /// 开始新的 block：更新旧 block header，刷盘，确保有空间
    fn start_new_block(&self) -> Result<()> {
        // 更新当前 block 的 header（写入真实 used_bytes）
        self.update_block_header()?;

        // 刷盘，确保旧 block 数据持久化
        if let Some(ref mut f) = *self.file.write() {
            f.flush()?;
        }
        let fs = *self.flushed_offset.read();

        // 计算新 block 起始偏移（必须是 32KB 对齐）
        let new_block_offset = (fs + WAL_BLOCK_SIZE as u64) & !(WAL_BLOCK_SIZE as u64 - 1);
        let padding = (new_block_offset - fs) as usize;

        if padding > 0 {
            // 填零对齐
            if let Some(ref mut f) = *self.file.write() {
                let zeros = vec![0u8; padding];
                f.write_all(&zeros)?;
                f.flush()?;
            }
            *self.flushed_offset.write() = new_block_offset;
        }

        *self.block_used.write() = 0;
        *self.block_file_offset.write() = new_block_offset;
        Ok(())
    }

    /// 更新当前 block 的 header（seek + write + flush）
    fn update_block_header(&self) -> Result<()> {
        let used = *self.block_used.read();
        let offset = *self.block_file_offset.read();
        if used == 0 {
            return Ok(());
        }

        if let Some(ref p) = *self.file_path.read() {
            let mut file = OpenOptions::new()
                .read(true)
                .write(true)
                .open(p)?;

            file.seek(SeekFrom::Start(offset))?;
            let header = WalBlockHeader::new(*self.block_seq.read(), used);
            let header_bytes = unsafe {
                std::slice::from_raw_parts(
                    &header as *const _ as *const u8,
                    WAL_BLOCK_HEADER_SIZE,
                )
            };
            file.write_all(header_bytes)?;
            file.flush()?;
        }
        Ok(())
    }

    fn rotate(&self) -> Result<()> {
        // 更新当前 block header 并刷盘
        if *self.block_used.read() > 0 {
            self.update_block_header()?;
            if let Some(ref mut f) = *self.file.write() {
                f.flush()?;
            }
        }

        *self.block_seq.write() += 1;
        self.start_new_file()?;
        info!("WAL rotated to seq={}", *self.block_seq.read());
        Ok(())
    }

    fn calc_crc(op_type: u8, txn_id: u64, payload_len: u32, payload: &[u8]) -> u32 {
        let mut h = Hasher::new();
        h.update(&[op_type]);
        h.update(&txn_id.to_le_bytes());
        h.update(&payload_len.to_le_bytes());
        h.update(payload);
        h.finalize()
    }

    /// 追加一条 WAL 记录
    pub fn append(&self, op_type: OpType, txn_id: u64, payload: &OpPayload) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }

        let payload_bytes = encode(payload)?;
        let payload_len = payload_bytes.len() as u32;
        if payload_len > WAL_MAX_RECORD_SIZE as u32 {
            return Err(crate::RockDuckError::Internal(
                format!("WAL record too large: {} bytes", payload_len)
            ).into());
        }

        // record = type(1) + txn_id(8) + len(4) + payload + crc(4)
        let record_len = 1 + 8 + 4 + payload_bytes.len() + 4;

        // 检查当前 block 剩余空间
        let remaining = (WAL_BLOCK_SIZE - WAL_BLOCK_HEADER_SIZE) as u32 - *self.block_used.read();
        if record_len as u32 > remaining {
            self.start_new_block()?;
        }

        // 写入 record
        {
            let mut file = self.file.write();
            if let Some(ref mut f) = *file {
                let crc = Self::calc_crc(op_type as u8, txn_id, payload_len, &payload_bytes);

                let mut hdr = Vec::with_capacity(13);
                hdr.push(op_type as u8);
                hdr.extend_from_slice(&txn_id.to_le_bytes());
                hdr.extend_from_slice(&payload_len.to_le_bytes());

                f.write_all(&hdr)?;
                f.write_all(&payload_bytes)?;
                f.write_all(&crc.to_le_bytes())?;
            }
        }

        *self.block_used.write() += record_len as u32;

        if op_type == OpType::Commit {
            let mut committed = self.committed_txn.write();
            if txn_id > *committed {
                *committed = txn_id;
            }
        }

        // 检查 rotation
        let padded_offset = (*self.block_file_offset.read()
            + WAL_BLOCK_HEADER_SIZE as u64
            + *self.block_used.read() as u64) as u64;
        if padded_offset >= self.config.max_file_size {
            self.rotate()?;
        }

        Ok(())
    }

    /// 刷盘
    pub fn flush(&self) -> Result<()> {
        self.update_block_header()?;
        if let Some(ref mut f) = *self.file.write() {
            f.flush()?;
        }
        Ok(())
    }

    pub fn active_files(&self) -> Vec<PathBuf> {
        let mut paths = Vec::new();
        if let Some(ref p) = *self.file_path.read() {
            paths.push(p.clone());
        }
        paths
    }

    pub fn get_committed_txn(&self) -> u64 {
        *self.committed_txn.read()
    }
}

impl Drop for WalWriter {
    fn drop(&mut self) {
        if let Some(ref mut f) = *self.file.write() {
            let _ = f.flush();
        }
    }
}

// ============================================================
// WAL Reader
// ============================================================

/// WAL 读取器（用于崩溃恢复）
pub struct WalReader {
    data_dir: PathBuf,
}

impl WalReader {
    pub fn new(data_dir: &Path) -> Self {
        Self {
            data_dir: data_dir.to_path_buf(),
        }
    }

    pub fn list_wal_files(&self) -> Result<Vec<PathBuf>> {
        let wal_dir = self.data_dir.join("wal");
        if !wal_dir.exists() {
            return Ok(Vec::new());
        }
        let mut files: Vec<_> = std::fs::read_dir(&wal_dir)?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map_or(false, |ext| ext == "bin"))
            .filter(|e| e.file_name().to_string_lossy().starts_with("wal_"))
            .map(|e| e.path())
            .collect();
        files.sort();
        Ok(files)
    }

    /// 扫描所有 WAL 文件，返回已提交事务的记录
    pub fn scan_committed_records(&self) -> Result<Vec<WalRecord>> {
        let files = self.list_wal_files()?;
        let mut committed: Vec<WalRecord> = Vec::new();
        let mut txn_states: std::collections::HashMap<u64, Vec<WalRecord>> =
            std::collections::HashMap::new();

        for path in &files {
            for rec in self.scan_file(path)? {
                match rec.op_type {
                    OpType::Begin | OpType::Insert | OpType::Delete | OpType::Update => {
                        txn_states.entry(rec.txn_id).or_default().push(rec);
                    }
                    OpType::Commit => {
                        if let Some(recs) = txn_states.remove(&rec.txn_id) {
                            committed.extend(recs);
                        }
                    }
                    OpType::Rollback => {
                        txn_states.remove(&rec.txn_id);
                    }
                }
            }
        }
        Ok(committed)
    }

    fn scan_file(&self, path: &Path) -> Result<Vec<WalRecord>> {
        let mut file = File::open(path)?;
        let file_size = file.metadata()?.len() as usize;
        let mut records = Vec::new();
        let mut pos = 0usize;

        while pos < file_size {
            // 跳过未对齐的位置
            let block_start = (pos + WAL_BLOCK_SIZE - 1) & !(WAL_BLOCK_SIZE - 1);
            if block_start > file_size {
                break;
            }
            if block_start != pos {
                pos = block_start;
                continue;
            }

            // 读 block header
            let mut hdr_buf = [0u8; WAL_BLOCK_HEADER_SIZE];
            file.seek(SeekFrom::Start(pos as u64))?;
            file.read_exact(&mut hdr_buf)?;

            let header: WalBlockHeader = unsafe {
                std::ptr::read_unaligned(hdr_buf.as_ptr() as *const _)
            };

            // 完全为零的 block：跳过
            if header.is_zeroed() {
                pos += WAL_BLOCK_SIZE;
                continue;
            }

            if !header.is_valid() {
                warn!("Invalid WAL block header at {:?}: pos={}", path, pos);
                break;
            }

            let data_start = pos + WAL_BLOCK_HEADER_SIZE;
            let data_end = data_start + header.used_bytes as usize;
            let mut block_pos = 0usize;

            // 解析 block 内的 records
            while block_pos < header.used_bytes as usize {
                let remaining = (header.used_bytes as usize) - block_pos;
                if remaining < 13 {
                    break;
                }

                let mut meta_buf = [0u8; 13];
                file.seek(SeekFrom::Start((data_start + block_pos) as u64))?;
                file.read_exact(&mut meta_buf)?;

                let op_type = OpType::from_u8(meta_buf[0]).unwrap_or(OpType::Begin);
                let txn_id = u64::from_le_bytes([
                    meta_buf[1], meta_buf[2], meta_buf[3],
                    meta_buf[4], meta_buf[5], meta_buf[6],
                    meta_buf[7], meta_buf[8],
                ]);
                let payload_len = u32::from_le_bytes([
                    meta_buf[9], meta_buf[10], meta_buf[11], meta_buf[12],
                ]) as usize;

                if payload_len == 0 || payload_len > WAL_MAX_RECORD_SIZE {
                    break;
                }
                if data_start + block_pos + 13 + payload_len + 4 > file_size {
                    break;
                }

                let mut payload_buf = vec![0u8; payload_len];
                let mut crc_buf = [0u8; 4];
                file.read_exact(&mut payload_buf)?;
                file.read_exact(&mut crc_buf)?;
                let expected_crc = u32::from_le_bytes(crc_buf);

                let actual_crc = Self::calc_crc(
                    meta_buf[0], txn_id, payload_len as u32, &payload_buf
                );
                if actual_crc != expected_crc {
                    warn!("WAL CRC mismatch at {:?}: expected={}, actual={}",
                        path, expected_crc, actual_crc);
                    break;
                }

                let payload: OpPayload = decode(&payload_buf)?;
                records.push(WalRecord {
                    op_type,
                    txn_id,
                    payload,
                });

                let rec_len = 1 + 8 + 4 + payload_len + 4;
                block_pos += rec_len;
            }

            pos += WAL_BLOCK_SIZE;
        }

        Ok(records)
    }

    fn calc_crc(op_type: u8, txn_id: u64, payload_len: u32, payload: &[u8]) -> u32 {
        let mut h = Hasher::new();
        h.update(&[op_type]);
        h.update(&txn_id.to_le_bytes());
        h.update(&payload_len.to_le_bytes());
        h.update(payload);
        h.finalize()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_wal_block_header_valid() {
        let h = WalBlockHeader::new(42, 1024);
        assert!(h.is_valid());
        assert!(!h.is_zeroed());
    }

    #[test]
    fn test_wal_block_header_zeroed() {
        let h: WalBlockHeader = unsafe {
            std::ptr::read_unaligned([0u8; WAL_BLOCK_HEADER_SIZE].as_ptr() as *const _)
        };
        assert!(h.is_zeroed());
    }

    #[test]
    fn test_optype_from_u8() {
        assert_eq!(OpType::from_u8(0x01), Some(OpType::Begin));
        assert_eq!(OpType::from_u8(0x02), Some(OpType::Insert));
        assert_eq!(OpType::from_u8(0x10), Some(OpType::Commit));
        assert_eq!(OpType::from_u8(0xff), None);
    }

    #[test]
    fn test_wal_writer_basic() {
        let temp = TempDir::new().unwrap();
        let config = WalConfig {
            wal_dir: temp.path().join("wal"),
            max_file_size: 1024 * 1024,
            enabled: true,
        };

        let wal = WalWriter::new(temp.path(), config).unwrap();

        wal.append(OpType::Begin, 1, &OpPayload::Begin).unwrap();
        wal.append(OpType::Insert, 1, &OpPayload::Insert {
            table: "t1".to_string(),
            pk: b"pk1".to_vec(),
            columns: vec![],
            seg_id: "seg_001".to_string(),
            granule_id: 0,
            offset: 0,
        }).unwrap();
        wal.append(OpType::Commit, 1, &OpPayload::Commit).unwrap();
        wal.flush().unwrap();

        let reader = WalReader::new(temp.path());
        let files = reader.list_wal_files().unwrap();
        assert_eq!(files.len(), 1);

        let records = reader.scan_file(&files[0]).unwrap();
        assert_eq!(records.len(), 3);
        assert_eq!(records[0].op_type, OpType::Begin);
        assert_eq!(records[1].op_type, OpType::Insert);
        assert_eq!(records[2].op_type, OpType::Commit);

        let committed = reader.scan_committed_records().unwrap();
        assert_eq!(committed.len(), 2);
    }

    #[test]
    fn test_wal_rollback_discarded() {
        let temp = TempDir::new().unwrap();
        let config = WalConfig {
            wal_dir: temp.path().join("wal"),
            max_file_size: 1024 * 1024,
            enabled: true,
        };

        let wal = WalWriter::new(temp.path(), config).unwrap();

        wal.append(OpType::Begin, 1, &OpPayload::Begin).unwrap();
        wal.append(OpType::Insert, 1, &OpPayload::Insert {
            table: "t1".to_string(),
            pk: b"pk1".to_vec(),
            columns: vec![],
            seg_id: "seg_001".to_string(),
            granule_id: 0,
            offset: 0,
        }).unwrap();
        wal.append(OpType::Commit, 1, &OpPayload::Commit).unwrap();

        wal.append(OpType::Begin, 2, &OpPayload::Begin).unwrap();
        wal.append(OpType::Insert, 2, &OpPayload::Insert {
            table: "t1".to_string(),
            pk: b"pk2".to_vec(),
            columns: vec![],
            seg_id: "seg_001".to_string(),
            granule_id: 0,
            offset: 1,
        }).unwrap();
        wal.append(OpType::Rollback, 2, &OpPayload::Rollback).unwrap();
        wal.flush().unwrap();

        let reader = WalReader::new(temp.path());
        let committed = reader.scan_committed_records().unwrap();
        assert_eq!(committed.len(), 2);
        for r in &committed {
            assert_eq!(r.txn_id, 1);
        }
    }

    #[test]
    fn test_wal_rotation() {
        let temp = TempDir::new().unwrap();
        let config = WalConfig {
            wal_dir: temp.path().join("wal"),
            max_file_size: 512, // 非常小的 rotation 阈值
            enabled: true,
        };

        let wal = WalWriter::new(temp.path(), config).unwrap();

        for i in 0..10 {
            wal.append(OpType::Begin, i, &OpPayload::Begin).unwrap();
            wal.append(OpType::Insert, i, &OpPayload::Insert {
                table: "t1".to_string(),
                pk: format!("pk{}", i).into_bytes(),
                columns: vec![],
                seg_id: "seg_001".to_string(),
                granule_id: 0,
                offset: i as u32,
            }).unwrap();
            wal.append(OpType::Commit, i, &OpPayload::Commit).unwrap();
        }
        wal.flush().unwrap();

        let reader = WalReader::new(temp.path());
        let files = reader.list_wal_files().unwrap();
        assert!(files.len() >= 2);

        // 验证所有已提交记录都能读回
        let committed = reader.scan_committed_records().unwrap();
        assert_eq!(committed.len(), 20); // 10 个事务 × 2 条记录每个
    }

    #[test]
    fn test_wal_block_crossing_record() {
        let temp = TempDir::new().unwrap();
        let config = WalConfig {
            wal_dir: temp.path().join("wal"),
            max_file_size: 256, // 每次都 rotation
            enabled: true,
        };

        let wal = WalWriter::new(temp.path(), config).unwrap();

        // 写一个长 payload（超过 256 bytes）
        let long_data = vec![0xAB; 200];
        wal.append(OpType::Begin, 1, &OpPayload::Begin).unwrap();
        wal.append(OpType::Insert, 1, &OpPayload::Insert {
            table: "t1".to_string(),
            pk: b"long_pk".to_vec(),
            columns: vec![("data".to_string(), long_data)],
            seg_id: "seg_001".to_string(),
            granule_id: 0,
            offset: 0,
        }).unwrap();
        wal.append(OpType::Commit, 1, &OpPayload::Commit).unwrap();
        wal.flush().unwrap();

        let reader = WalReader::new(temp.path());
        let committed = reader.scan_committed_records().unwrap();
        assert_eq!(committed.len(), 2);
    }

    #[test]
    fn test_wal_commit_preserves_txn_ids() {
        let temp = TempDir::new().unwrap();
        let config = WalConfig {
            wal_dir: temp.path().join("wal"),
            max_file_size: 1024 * 1024,
            enabled: true,
        };

        let wal = WalWriter::new(temp.path(), config).unwrap();

        // Txn 10
        wal.append(OpType::Begin, 10, &OpPayload::Begin).unwrap();
        wal.append(OpType::Insert, 10, &OpPayload::Insert {
            table: "t1".to_string(),
            pk: b"pk_a".to_vec(),
            columns: vec![],
            seg_id: "seg_001".to_string(),
            granule_id: 0,
            offset: 0,
        }).unwrap();
        wal.append(OpType::Commit, 10, &OpPayload::Commit).unwrap();

        // Txn 20
        wal.append(OpType::Begin, 20, &OpPayload::Begin).unwrap();
        wal.append(OpType::Insert, 20, &OpPayload::Insert {
            table: "t1".to_string(),
            pk: b"pk_b".to_vec(),
            columns: vec![],
            seg_id: "seg_002".to_string(),
            granule_id: 0,
            offset: 0,
        }).unwrap();
        wal.append(OpType::Commit, 20, &OpPayload::Commit).unwrap();

        wal.flush().unwrap();

        let reader = WalReader::new(temp.path());
        let committed = reader.scan_committed_records().unwrap();
        assert_eq!(committed.len(), 4);
        let txn_ids: Vec<u64> = committed.iter().map(|r| r.txn_id).collect();
        assert_eq!(txn_ids, vec![10, 10, 20, 20]);
    }

    #[test]
    fn test_wal_uncommitted_txn_filtered() {
        let temp = TempDir::new().unwrap();
        let config = WalConfig {
            wal_dir: temp.path().join("wal"),
            max_file_size: 1024 * 1024,
            enabled: true,
        };

        let wal = WalWriter::new(temp.path(), config).unwrap();

        // Txn 1: commits
        wal.append(OpType::Begin, 1, &OpPayload::Begin).unwrap();
        wal.append(OpType::Insert, 1, &OpPayload::Insert {
            table: "t1".to_string(),
            pk: b"pk1".to_vec(),
            columns: vec![],
            seg_id: "seg_001".to_string(),
            granule_id: 0,
            offset: 0,
        }).unwrap();
        wal.append(OpType::Commit, 1, &OpPayload::Commit).unwrap();

        // Txn 2: rolls back
        wal.append(OpType::Begin, 2, &OpPayload::Begin).unwrap();
        wal.append(OpType::Insert, 2, &OpPayload::Insert {
            table: "t1".to_string(),
            pk: b"pk2".to_vec(),
            columns: vec![],
            seg_id: "seg_001".to_string(),
            granule_id: 0,
            offset: 1,
        }).unwrap();
        wal.append(OpType::Rollback, 2, &OpPayload::Rollback).unwrap();

        wal.flush().unwrap();

        let reader = WalReader::new(temp.path());
        let committed = reader.scan_committed_records().unwrap();
        // Only Txn 1 records appear
        assert_eq!(committed.len(), 2);
        for r in &committed {
            assert_eq!(r.txn_id, 1);
        }
    }

    #[test]
    fn test_wal_crc_error_skips_record() {
        let temp = TempDir::new().unwrap();
        let config = WalConfig {
            wal_dir: temp.path().join("wal"),
            max_file_size: 1024 * 1024,
            enabled: true,
        };

        let wal = WalWriter::new(temp.path(), config).unwrap();

        // Write Txn 1 (valid)
        wal.append(OpType::Begin, 1, &OpPayload::Begin).unwrap();
        wal.append(OpType::Insert, 1, &OpPayload::Insert {
            table: "t1".to_string(),
            pk: b"pk1".to_vec(),
            columns: vec![],
            seg_id: "seg_001".to_string(),
            granule_id: 0,
            offset: 0,
        }).unwrap();
        wal.append(OpType::Commit, 1, &OpPayload::Commit).unwrap();

        // Write Txn 2
        wal.append(OpType::Begin, 2, &OpPayload::Begin).unwrap();
        wal.append(OpType::Insert, 2, &OpPayload::Insert {
            table: "t1".to_string(),
            pk: b"pk2".to_vec(),
            columns: vec![],
            seg_id: "seg_001".to_string(),
            granule_id: 0,
            offset: 1,
        }).unwrap();
        wal.flush().unwrap();

        // Corrupt Txn 2's first record CRC (last 4 bytes of that record)
        // We need to find where Txn 2's Begin record starts.
        // Txn 1 records: Begin(1+8+4+0+4=17) + Insert(payload_len+17) + Commit(17)
        // The CRC is the last 4 bytes of each record.
        let wal_path = temp.path().join("wal").join("wal_000000.bin");
        let data = std::fs::read(&wal_path).unwrap();

        // Find the start of Txn 2 Begin: scan for 0x01 (Begin op type) + txn_id 2
        let mut txn2_begin_pos = None;
        for i in 0..data.len().saturating_sub(13) {
            if data[i] == 0x01 {
                let txn_id = u64::from_le_bytes([
                    data[i+1], data[i+2], data[i+3],
                    data[i+4], data[i+5], data[i+6],
                    data[i+7], data[i+8],
                ]);
                if txn_id == 2 {
                    txn2_begin_pos = Some(i);
                    break;
                }
            }
        }
        let pos = txn2_begin_pos.expect("Should find Txn 2 Begin");

        // CRC is at pos + 13 + payload_len, last 4 bytes of the record
        let payload_len = u32::from_le_bytes([
            data[pos+9], data[pos+10], data[pos+11], data[pos+12]
        ]) as usize;
        let crc_pos = pos + 13 + payload_len;
        assert!(crc_pos + 4 <= data.len());

        // Corrupt CRC
        let mut corrupted = data.clone();
        corrupted[crc_pos] = corrupted[crc_pos].wrapping_add(1);
        std::fs::write(&wal_path, &corrupted).unwrap();

        let reader = WalReader::new(temp.path());
        let files = reader.list_wal_files().unwrap();
        let all_records = reader.scan_file(&files[0]).unwrap();

        // Txn 1 records should be readable (valid CRC)
        let txn1_records: Vec<_> = all_records.iter().filter(|r| r.txn_id == 1).collect();
        assert_eq!(txn1_records.len(), 3); // Begin + Insert + Commit

        // Txn 2 record should NOT be in results (CRC mismatch causes break)
        let txn2_records: Vec<_> = all_records.iter().filter(|r| r.txn_id == 2).collect();
        assert_eq!(txn2_records.len(), 0, "Txn 2 skipped due to CRC mismatch");
    }

    #[test]
    fn test_wal_partial_block_read() {
        let temp = TempDir::new().unwrap();
        let config = WalConfig {
            wal_dir: temp.path().join("wal"),
            max_file_size: WAL_BLOCK_SIZE as u64 * 4, // 4 blocks
            enabled: true,
        };

        let wal = WalWriter::new(temp.path(), config).unwrap();

        // Write many small records to span multiple blocks
        for i in 0..20u64 {
            wal.append(OpType::Begin, i, &OpPayload::Begin).unwrap();
            wal.append(OpType::Insert, i, &OpPayload::Insert {
                table: "t1".to_string(),
                pk: format!("pk{}", i).into_bytes(),
                columns: vec![],
                seg_id: "seg_001".to_string(),
                granule_id: 0,
                offset: i as u32,
            }).unwrap();
            wal.append(OpType::Commit, i, &OpPayload::Commit).unwrap();
        }
        wal.flush().unwrap();

        let reader = WalReader::new(temp.path());
        let committed = reader.scan_committed_records().unwrap();
        assert_eq!(committed.len(), 40); // 20 txns * 2 records each

        // Verify we have records from multiple transactions
        let unique_txn_ids: std::collections::HashSet<u64> = committed.iter().map(|r| r.txn_id).collect();
        assert_eq!(unique_txn_ids.len(), 20);
    }

    #[test]
    fn test_wal_empty_dir_returns_empty() {
        let temp = TempDir::new().unwrap();
        // Do NOT create the wal subdirectory
        let reader = WalReader::new(temp.path());
        let files = reader.list_wal_files().unwrap();
        assert_eq!(files.len(), 0);
        assert!(files.is_empty());
    }
}
