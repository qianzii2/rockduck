//! RockDuck ??????

use derive_builder::Builder;
use std::path::PathBuf;

/// RockDuck ??
#[derive(Debug, Clone, Builder)]
pub struct RockDuckConfig {
    /// ????
    #[builder(setter(into))]
    pub data_dir: PathBuf,
    /// Granule ????????? 1MB
    #[builder(default = "1024 * 1024")]
    pub granule_size: usize,
    /// Segment ??????????? 1GB
    #[builder(default = "1024 * 1024 * 1024")]
    pub segment_target_size: usize,
    /// ?????
    #[builder(default = "num_cpus::get()")]
    pub num_threads: usize,
    /// ???? Bloom Filter
    #[builder(default = "true")]
    pub enable_bloom_filter: bool,
    /// Bloom Filter ????
    #[builder(default = "0.01")]
    pub bloom_filter_fpp: f64,
    /// ???? Zone Map
    #[builder(default = "true")]
    pub enable_zone_map: bool,
    /// ??????
    #[builder(default = "true")]
    pub enable_compression: bool,
    /// ???? ("lz4", "zstd", "snappy")
    #[builder(setter(into, strip_option), default = "Some(\"lz4\".to_string())")]
    pub compression_algorithm: Option<String>,
    /// 是否启用 WAL（崩溃恢复保证）
    #[builder(default = "true")]
    pub enable_wal: bool,
    /// WAL 单文件最大大小（字节），默认 128MB
    #[builder(default = "128 * 1024 * 1024")]
    pub wal_max_file_size: u64,
}

impl Default for RockDuckConfig {
    fn default() -> Self {
        Self {
            data_dir: PathBuf::from("./rockduck_data"),
            granule_size: 1024 * 1024,
            segment_target_size: 1024 * 1024 * 1024,
            num_threads: num_cpus::get(),
            enable_bloom_filter: true,
            bloom_filter_fpp: 0.01,
            enable_zone_map: true,
            enable_compression: true,
            compression_algorithm: Some("lz4".to_string()),
            enable_wal: true,
            wal_max_file_size: 128 * 1024 * 1024,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Default values ----

    #[test]
    fn test_default_config_values() {
        let config = RockDuckConfig::default();
        assert_eq!(config.granule_size, 1024 * 1024);
        assert_eq!(config.segment_target_size, 1024 * 1024 * 1024);
        assert!(config.enable_bloom_filter);
        assert!((config.bloom_filter_fpp - 0.01).abs() < 1e-9);
        assert!(config.enable_zone_map);
        assert!(config.enable_compression);
        assert_eq!(config.compression_algorithm, Some("lz4".to_string()));
    }

    // ---- Builder - all fields ----

    #[test]
    fn test_builder_all_fields() {
        use std::path::PathBuf;
        let config = RockDuckConfig {
            data_dir: PathBuf::from("/test/path"),
            granule_size: 512 * 1024,
            segment_target_size: 512 * 1024 * 1024,
            num_threads: 8,
            enable_bloom_filter: false,
            bloom_filter_fpp: 0.05,
            enable_zone_map: false,
            enable_compression: false,
            compression_algorithm: Some("zstd".to_string()),
            enable_wal: false,
            wal_max_file_size: 64 * 1024 * 1024,
        };

        assert_eq!(config.data_dir, PathBuf::from("/test/path"));
        assert_eq!(config.granule_size, 512 * 1024);
        assert_eq!(config.segment_target_size, 512 * 1024 * 1024);
        assert_eq!(config.num_threads, 8);
        assert!(!config.enable_bloom_filter);
        assert!((config.bloom_filter_fpp - 0.05).abs() < 1e-9);
        assert!(!config.enable_zone_map);
        assert!(!config.enable_compression);
        assert_eq!(config.compression_algorithm, Some("zstd".to_string()));
    }

    // ---- Clone equality ----

    #[test]
    fn test_config_clone() {
        let config = RockDuckConfig::default();
        let cloned = config.clone();
        assert_eq!(config.data_dir, cloned.data_dir);
        assert_eq!(config.granule_size, cloned.granule_size);
        assert_eq!(config.segment_target_size, cloned.segment_target_size);
        assert_eq!(config.num_threads, cloned.num_threads);
        assert_eq!(config.enable_bloom_filter, cloned.enable_bloom_filter);
        assert_eq!(config.bloom_filter_fpp, cloned.bloom_filter_fpp);
        assert_eq!(config.enable_zone_map, cloned.enable_zone_map);
        assert_eq!(config.enable_compression, cloned.enable_compression);
        assert_eq!(config.compression_algorithm, cloned.compression_algorithm);
    }

    // ---- Debug format ----

    #[test]
    fn test_config_debug() {
        let config = RockDuckConfig::default();
        let debug_str = format!("{:?}", config);
        assert!(!debug_str.is_empty());
        assert!(debug_str.contains("RockDuckConfig"));
    }

    // ---- Compression variants ----

    #[test]
    fn test_compression_algorithm_variants() {
        for algo in &["lz4", "zstd", "snappy"] {
            let config = RockDuckConfig {
                data_dir: std::path::PathBuf::from("/tmp/test"),
                compression_algorithm: Some(algo.to_string()),
                ..RockDuckConfig::default()
            };
            assert_eq!(config.compression_algorithm, Some(algo.to_string()));
        }

        let config = RockDuckConfig {
            data_dir: std::path::PathBuf::from("/tmp/test"),
            compression_algorithm: None,
            ..RockDuckConfig::default()
        };
        assert_eq!(config.compression_algorithm, None);
    }
}
