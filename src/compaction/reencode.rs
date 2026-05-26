//! Adaptive 列编码重选
//!
//! Compaction 时分析每列数据分布：
//! - 单调递增 → DELTA 编码
//! - 低基数 → DICT 编码
//! - 浮点数方差小 → ALP/GORILLA
//! - 高基数无模式 → RAW
//! 重新编码写入，利用 compaction 机会优化压缩率

use std::collections::HashMap;
use tracing::debug;
use crate::segment::encoding::AdaptiveEncoder;
use crate::segment::meta::{ColumnDef, DataType, EncodingType};

/// 推荐编码
pub use crate::segment::encoding::EncodingRecommendation;

/// 分析结果
#[derive(Debug)]
pub struct EncodingAnalysis {
    pub col_name: String,
    pub dtype: DataType,
    pub cardinality: Option<u64>,
    pub min_value: Option<Vec<u8>>,
    pub max_value: Option<Vec<u8>>,
    pub null_count: u32,
    pub is_sorted: bool,
    pub compression_ratio_hint: f64,
}

/// 自适应重编码器
pub struct AdaptiveReEncoder {
    encoder: AdaptiveEncoder,
}

impl AdaptiveReEncoder {
    pub fn new() -> Self {
        Self {
            encoder: AdaptiveEncoder::new(),
        }
    }

    /// 分析列数据并推荐编码
    pub fn analyze_and_recommend(
        &self,
        col: &ColumnDef,
        values: &[u8],
    ) -> EncodingRecommendation {
        self.encoder.analyze_and_recommend(col, values)
    }

    /// 分析多列
    pub fn analyze_columns(
        &self,
        columns: &[ColumnDef],
        data: &HashMap<String, Vec<u8>>,
    ) -> HashMap<String, EncodingRecommendation> {
        let mut results = HashMap::new();

        for col in columns {
            if let Some(values) = data.get(&col.name) {
                let rec = self.analyze_and_recommend(col, values);
                debug!("Column {}: recommended {:?} (confidence: {:.2})",
                    col.name, rec.encoding, rec.confidence);
                results.insert(col.name.clone(), rec);
            }
        }

        results
    }

    /// 为所有列选择最佳编码
    pub fn select_best_encoding(
        &self,
        columns: &[ColumnDef],
        data: &HashMap<String, Vec<u8>>,
    ) -> Vec<ColumnDef> {
        let analyses = self.analyze_columns(columns, data);
        
        columns.iter().map(|col| {
            if let Some(rec) = analyses.get(&col.name) {
                ColumnDef::with_encoding(
                    col.name.clone(),
                    col.dtype,
                    rec.encoding,
                )
            } else {
                col.clone()
            }
        }).collect()
    }
}

impl Default for AdaptiveReEncoder {
    fn default() -> Self {
        Self::new()
    }
}

/// 估算压缩率
pub fn estimate_compression_ratio(
    encoding: EncodingType,
    values: &[u8],
    dtype: &DataType,
) -> f64 {
    let _raw_size = values.len();
    
    match encoding {
        EncodingType::Raw => 1.0,
        EncodingType::Delta => {
            if matches!(dtype, DataType::Int64 | DataType::UInt64) {
                // Delta 编码通常能达到 2-5x 压缩
                0.3
            } else {
                0.8
            }
        }
        EncodingType::Dict => {
            // Dictionary 编码取决于基数
            0.2
        }
        EncodingType::Alp => {
            if matches!(dtype, DataType::Int64 | DataType::Float64) {
                0.25
            } else {
                0.7
            }
        }
        EncodingType::Gorilla => {
            if matches!(dtype, DataType::Float32 | DataType::Float64) {
                0.3
            } else {
                0.8
            }
        }
        EncodingType::Rle => 0.4,
        EncodingType::Bitpacked => 0.5,
        _ => 1.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ============================================================
    // AdaptiveReEncoder
    // ============================================================

    #[test]
    fn test_adaptive_re_encoder_new() {
        let _encoder = AdaptiveReEncoder::new();
        // Just verify it compiles and creates
    }

    #[test]
    fn test_adaptive_re_encoder_default() {
        let _encoder: AdaptiveReEncoder = Default::default();
        // Verify it creates
    }

    // ============================================================
    // analyze_and_recommend
    // ============================================================

    // Note: analyze_and_recommend delegates to segment::encoding::AdaptiveEncoder.
    // The encoding module has its own tests for this logic.

    // ============================================================
    // analyze_columns
    // ============================================================

    #[test]
    fn test_analyze_columns_empty_data() {
        let encoder = AdaptiveReEncoder::new();
        let cols = vec![
            ColumnDef::new("a".to_string(), DataType::Int64),
            ColumnDef::new("b".to_string(), DataType::Float64),
        ];
        let data: std::collections::HashMap<String, Vec<u8>> = std::collections::HashMap::new();
        let results = encoder.analyze_columns(&cols, &data);
        // Both columns have no data, results should be empty
        assert!(results.is_empty());
    }

    #[test]
    fn test_analyze_columns_with_data() {
        let encoder = AdaptiveReEncoder::new();
        let cols = vec![ColumnDef::new("id".to_string(), DataType::Int64)];
        let mut data: std::collections::HashMap<String, Vec<u8>> = std::collections::HashMap::new();
        data.insert("id".to_string(), vec![1u8, 2, 3, 4, 5]);

        let results = encoder.analyze_columns(&cols, &data);
        assert!(results.contains_key("id"));
    }

    // ============================================================
    // select_best_encoding
    // ============================================================

    #[test]
    fn test_select_best_encoding() {
        let encoder = AdaptiveReEncoder::new();
        let cols = vec![ColumnDef::new("id".to_string(), DataType::Int64)];

        let mut data: std::collections::HashMap<String, Vec<u8>> = std::collections::HashMap::new();
        data.insert("id".to_string(), vec![1u8, 2, 3, 4, 5]);

        let selected = encoder.select_best_encoding(&cols, &data);
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].name, "id");
    }

    // ============================================================
    // estimate_compression_ratio
    // ============================================================

    #[test]
    fn test_estimate_compression_ratio_raw() {
        let ratio = estimate_compression_ratio(EncodingType::Raw, &[1, 2, 3], &DataType::Binary);
        assert!((ratio - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_estimate_compression_ratio_delta_int64() {
        let ratio = estimate_compression_ratio(EncodingType::Delta, &[1, 2, 3], &DataType::Int64);
        assert!((ratio - 0.3).abs() < 1e-9);
    }

    #[test]
    fn test_estimate_compression_ratio_delta_non_int64() {
        let ratio = estimate_compression_ratio(EncodingType::Delta, &[1, 2, 3], &DataType::Utf8);
        assert!((ratio - 0.8).abs() < 1e-9);
    }

    #[test]
    fn test_estimate_compression_ratio_dict() {
        let ratio = estimate_compression_ratio(EncodingType::Dict, &[], &DataType::Int32);
        assert!((ratio - 0.2).abs() < 1e-9);
    }

    #[test]
    fn test_estimate_compression_ratio_alp_int64() {
        let ratio = estimate_compression_ratio(EncodingType::Alp, &[], &DataType::Int64);
        assert!((ratio - 0.25).abs() < 1e-9);
    }

    #[test]
    fn test_estimate_compression_ratio_alp_non_int() {
        let ratio = estimate_compression_ratio(EncodingType::Alp, &[], &DataType::Utf8);
        assert!((ratio - 0.7).abs() < 1e-9);
    }

    #[test]
    fn test_estimate_compression_ratio_gorilla_float() {
        let ratio = estimate_compression_ratio(EncodingType::Gorilla, &[], &DataType::Float64);
        assert!((ratio - 0.3).abs() < 1e-9);
    }

    #[test]
    fn test_estimate_compression_ratio_gorilla_non_float() {
        let ratio = estimate_compression_ratio(EncodingType::Gorilla, &[], &DataType::Int32);
        assert!((ratio - 0.8).abs() < 1e-9);
    }

    #[test]
    fn test_estimate_compression_ratio_rle() {
        let ratio = estimate_compression_ratio(EncodingType::Rle, &[], &DataType::Bool);
        assert!((ratio - 0.4).abs() < 1e-9);
    }

    #[test]
    fn test_estimate_compression_ratio_bitpacked() {
        let ratio = estimate_compression_ratio(EncodingType::Bitpacked, &[], &DataType::Int32);
        assert!((ratio - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_estimate_compression_ratio_unrecognized() {
        // All EncodingType variants should be covered, but test fallback
        let ratio = estimate_compression_ratio(EncodingType::Zstd, &[], &DataType::Binary);
        assert!((ratio - 1.0).abs() < 1e-9);
    }

    // ============================================================
    // EncodingAnalysis Debug
    // ============================================================

    #[test]
    fn test_encoding_analysis_debug() {
        let analysis = EncodingAnalysis {
            col_name: "id".to_string(),
            dtype: DataType::Int64,
            cardinality: Some(100),
            min_value: Some(vec![1]),
            max_value: Some(vec![100]),
            null_count: 0,
            is_sorted: true,
            compression_ratio_hint: 0.5,
        };
        let debug_str = format!("{:?}", analysis);
        assert!(!debug_str.is_empty());
    }

    // ============================================================
    // EncodingRecommendation Debug
    // ============================================================

    #[test]
    fn test_encoding_recommendation_debug() {
        let rec = EncodingRecommendation {
            encoding: EncodingType::Delta,
            confidence: 0.85,
            reason: "Test reason".to_string(),
        };
        let debug_str = format!("{:?}", rec);
        assert!(!debug_str.is_empty());
    }
}
