//! Configuration for RockDuck database

use serde::{Deserialize, Serialize};

use crate::query::routing::config::RouterConfig;
use crate::write::durability_wal::WalConfig;

#[derive(Clone)]
pub struct RockDuckConfig {
    pub compaction: CompactionConfig,
    pub storage: StorageConfig,
    pub bloom_filter_fpp: f64,
    pub router: RouterConfig,
    pub checkpoint: CheckpointConfig,
    pub cdc: CdcConfig,
    pub visibility: VisibilityConfig,
    /// Optional WAL config override. If None, uses the default.
    pub wal_config: Option<WalConfig>,
}

impl RockDuckConfig {
    /// Default bucket names for mace-kv (column families in RocksDB terms).
    /// mace-kv auto-creates buckets on first access, so this is informational only.
    pub fn default_buckets() -> Vec<&'static str> {
        vec![
            crate::metadata::CF_PK_IDX,
            crate::metadata::CF_SEG_META,
            crate::metadata::CF_STAT,
            crate::metadata::CF_ZONE,
            crate::metadata::CF_LAYER,
            crate::metadata::CF_BF,
            crate::metadata::CF_SYS,
            crate::metadata::CF_MVCC,
            crate::metadata::CF_ICEBERG,
        ]
    }
}

impl Default for RockDuckConfig {
    fn default() -> Self {
        Self {
            compaction: CompactionConfig::default(),
            storage: StorageConfig::default(),
            bloom_filter_fpp: 0.01,
            router: RouterConfig::default(),
            checkpoint: CheckpointConfig::default(),
            cdc: CdcConfig::default(),
            visibility: VisibilityConfig::default(),
            wal_config: None,
        }
    }
}

impl RockDuckConfigBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn compaction(mut self, v: CompactionConfig) -> Self {
        self.compaction = Some(v);
        self
    }

    pub fn storage(mut self, v: StorageConfig) -> Self {
        self.storage = Some(v);
        self
    }

    pub fn bloom_filter_fpp(mut self, v: f64) -> Self {
        if !(v > 0.0 && v <= 1.0) {
            panic!("bloom_filter_fpp must be in (0.0, 1.0], got {v}");
        }
        self.bloom_filter_fpp = Some(v);
        self
    }

    pub fn router(mut self, v: RouterConfig) -> Self {
        self.router = Some(v);
        self
    }

    pub fn checkpoint(mut self, v: CheckpointConfig) -> Self {
        self.checkpoint = Some(v);
        self
    }

    pub fn cdc(mut self, v: CdcConfig) -> Self {
        self.cdc = Some(v);
        self
    }

    pub fn visibility(mut self, v: VisibilityConfig) -> Self {
        self.visibility = Some(v);
        self
    }

    pub fn wal_config(mut self, v: WalConfig) -> Self {
        self.wal_config = Some(v);
        self
    }

    pub fn build(self) -> RockDuckConfig {
        RockDuckConfig {
            compaction: self.compaction.unwrap_or_default(),
            storage: self.storage.unwrap_or_default(),
            bloom_filter_fpp: self.bloom_filter_fpp.unwrap_or(0.01),
            router: self.router.unwrap_or_default(),
            checkpoint: self.checkpoint.unwrap_or_default(),
            cdc: self.cdc.unwrap_or_default(),
            visibility: self.visibility.unwrap_or_default(),
            wal_config: self.wal_config,
        }
    }
}

#[derive(Default)]
pub struct RockDuckConfigBuilder {
    compaction: Option<CompactionConfig>,
    storage: Option<StorageConfig>,
    bloom_filter_fpp: Option<f64>,
    router: Option<RouterConfig>,
    checkpoint: Option<CheckpointConfig>,
    cdc: Option<CdcConfig>,
    visibility: Option<VisibilityConfig>,
    wal_config: Option<WalConfig>,
}

/// MVCC visibility manager configuration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VisibilityConfig {
    /// Maximum number of entries to retain in the committed history.
    /// When exceeded, oldest 20% of entries are evicted.
    pub max_history_entries: usize,
    /// Time-to-live for committed history entries, in seconds.
    /// Entries older than (now - ttl) are evicted on each commit.
    pub history_ttl_secs: u64,
}

impl Default for VisibilityConfig {
    fn default() -> Self {
        Self {
            max_history_entries: 100_000,
            history_ttl_secs: 3600,
        }
    }
}

impl VisibilityConfigBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn max_history_entries(mut self, v: usize) -> Self {
        self.max_history_entries = Some(v);
        self
    }

    pub fn history_ttl_secs(mut self, v: u64) -> Self {
        self.history_ttl_secs = Some(v);
        self
    }

    pub fn build(self) -> VisibilityConfig {
        VisibilityConfig {
            max_history_entries: self.max_history_entries.unwrap_or(100_000),
            history_ttl_secs: self.history_ttl_secs.unwrap_or(3600),
        }
    }
}

#[derive(Default)]
pub struct VisibilityConfigBuilder {
    max_history_entries: Option<usize>,
    history_ttl_secs: Option<u64>,
}

#[derive(Clone, Debug)]
pub struct CompactionConfig {
    pub del_ratio_threshold: f64,
    pub max_concurrent_tasks: usize,
    pub target_granule_rows: u32,
    /// Enable background compaction thread (FlushEngine). Default: false.
    pub background_enabled: bool,
    /// Memstore threshold for FlushEngine L1 flush decisions. Default: 64MB.
    pub memstore_threshold: usize,
}

impl Default for CompactionConfig {
    fn default() -> Self {
        Self {
            del_ratio_threshold: 0.1,
            max_concurrent_tasks: 4,
            target_granule_rows: 1024,
            background_enabled: false,
            memstore_threshold: 64 * 1024 * 1024,
        }
    }
}

impl CompactionConfigBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn del_ratio_threshold(mut self, v: f64) -> Self {
        self.del_ratio_threshold = Some(v);
        self
    }

    pub fn max_concurrent_tasks(mut self, v: usize) -> Self {
        self.max_concurrent_tasks = Some(v);
        self
    }

    pub fn target_granule_rows(mut self, v: u32) -> Self {
        self.target_granule_rows = Some(v);
        self
    }

    pub fn background_enabled(mut self, v: bool) -> Self {
        self.background_enabled = Some(v);
        self
    }

    pub fn memstore_threshold(mut self, v: usize) -> Self {
        self.memstore_threshold = Some(v);
        self
    }

    pub fn build(self) -> CompactionConfig {
        CompactionConfig {
            del_ratio_threshold: self.del_ratio_threshold.unwrap_or(0.1),
            max_concurrent_tasks: self.max_concurrent_tasks.unwrap_or(4),
            target_granule_rows: self.target_granule_rows.unwrap_or(1024),
            background_enabled: self.background_enabled.unwrap_or(false),
            memstore_threshold: self.memstore_threshold.unwrap_or(64 * 1024 * 1024),
        }
    }
}

#[derive(Default)]
pub struct CompactionConfigBuilder {
    del_ratio_threshold: Option<f64>,
    max_concurrent_tasks: Option<usize>,
    target_granule_rows: Option<u32>,
    background_enabled: Option<bool>,
    memstore_threshold: Option<usize>,
}

#[derive(Clone, Debug)]
pub struct StorageConfig {
    pub compression: CompressionType,
    pub target_file_size: u64,
    pub use_mmap: bool,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            compression: CompressionType::Zstd,
            target_file_size: 128 * 1024 * 1024,
            use_mmap: true,
        }
    }
}

impl StorageConfigBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn compression(mut self, v: CompressionType) -> Self {
        self.compression = Some(v);
        self
    }

    pub fn target_file_size(mut self, v: u64) -> Self {
        self.target_file_size = Some(v);
        self
    }

    pub fn use_mmap(mut self, v: bool) -> Self {
        self.use_mmap = Some(v);
        self
    }

    pub fn build(self) -> StorageConfig {
        StorageConfig {
            compression: self.compression.unwrap_or(CompressionType::Zstd),
            target_file_size: self.target_file_size.unwrap_or(128 * 1024 * 1024),
            use_mmap: self.use_mmap.unwrap_or(true),
        }
    }
}

#[derive(Default)]
pub struct StorageConfigBuilder {
    compression: Option<CompressionType>,
    target_file_size: Option<u64>,
    use_mmap: Option<bool>,
}

#[derive(Clone, Copy, Debug, Default, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub enum CompressionType {
    None,
    #[default]
    Zstd,
    Lz4,
    Snappy,
}

/// Checkpoint trigger configuration.
#[derive(Clone, Debug)]
pub struct CheckpointConfig {
    /// WAL size threshold in bytes. Checkpoint triggers when WAL exceeds this.
    pub wal_size_threshold: u64,
    /// Time interval threshold in seconds. Checkpoint triggers after this many seconds
    /// since the last checkpoint.
    pub time_threshold_secs: u64,
    /// Number of committed transactions threshold. Checkpoint triggers after this many
    /// commits since the last checkpoint.
    pub txn_count_threshold: u64,
    /// Whether checkpointing is enabled.
    pub enabled: bool,
}

impl Default for CheckpointConfig {
    fn default() -> Self {
        Self {
            wal_size_threshold: 64 * 1024 * 1024, // 64 MB
            time_threshold_secs: 300,             // 5 minutes
            txn_count_threshold: 10_000,
            enabled: true,
        }
    }
}

/// CDC (Change Data Capture) configuration.
#[derive(Clone, Debug)]
pub struct CdcConfig {
    /// Whether CDC is enabled.
    pub enabled: bool,
    /// Granularity of CDC output (Cell, Row, or Both).
    pub granularity: crate::cdc::CdcGranularity,
    /// Kafka bootstrap servers (e.g., "localhost:9092"). Used if any kafka sink is configured.
    pub kafka_bootstrap_servers: Option<String>,
    /// Default topic for CDC events when no table-specific routing is configured.
    pub kafka_default_topic: Option<String>,
    /// Maximum number of CDC log entries to retain in the in-memory ring buffer.
    pub log_buffer_size: usize,
    /// Client ID prefix for Kafka producers.
    pub kafka_client_id: Option<String>,
    /// Message timeout in milliseconds for Kafka sends.
    pub kafka_message_timeout_ms: u64,
}

impl Default for CdcConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            granularity: crate::cdc::CdcGranularity::Cell,
            kafka_bootstrap_servers: None,
            kafka_default_topic: None,
            log_buffer_size: 100_000,
            kafka_client_id: None,
            kafka_message_timeout_ms: 5000,
        }
    }
}
