//! Column encoding metadata - persisted alongside each .vortex column file.
//!
//! This module defines the metadata structures that track which encoding scheme
//! was used for each column and each block within a column. This metadata is
//! stored in a sidecar `.vortex.encoding` file alongside each column's `.vortex`
//! data file.

use rkyv::{Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};
use serde::{Deserialize, Serialize};

/// Encoding scheme used for a column or block.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Serialize,
    Deserialize,
    Archive,
    RkyvSerialize,
    RkyvDeserialize,
    Default,
)]
pub enum EncodingScheme {
    Raw,
    #[default]
    BtrBlocks,
    Dictionary,
    FOR,
    Delta,
    RLE,
    BitPacking,
    Alp,
    AlpRD,
    Gorilla,
    CORRAPeerDiff,
    CORRASubaltern,
    CORRAHierarchical,
}

impl EncodingScheme {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Raw => "raw",
            Self::BtrBlocks => "btrblocks",
            Self::Dictionary => "dict",
            Self::FOR => "for",
            Self::Delta => "delta",
            Self::RLE => "rle",
            Self::BitPacking => "bitpack",
            Self::Alp => "alp",
            Self::AlpRD => "alp_rd",
            Self::Gorilla => "gorilla",
            Self::CORRAPeerDiff => "corra_peer",
            Self::CORRASubaltern => "corra_subaltern",
            Self::CORRAHierarchical => "corra_hier",
        }
    }

    pub fn try_from_str(s: &str) -> Option<Self> {
        match s {
            "raw" => Some(Self::Raw),
            "btrblocks" => Some(Self::BtrBlocks),
            "dict" => Some(Self::Dictionary),
            "for" => Some(Self::FOR),
            "delta" => Some(Self::Delta),
            "rle" => Some(Self::RLE),
            "bitpack" => Some(Self::BitPacking),
            "alp" => Some(Self::Alp),
            "alp_rd" => Some(Self::AlpRD),
            "gorilla" => Some(Self::Gorilla),
            "corra_peer" => Some(Self::CORRAPeerDiff),
            "corra_subaltern" => Some(Self::CORRASubaltern),
            "corra_hier" => Some(Self::CORRAHierarchical),
            _ => None,
        }
    }
}

/// Per-block metadata.
#[derive(Debug, Clone, Serialize, Deserialize, Archive, RkyvSerialize, RkyvDeserialize)]
pub struct BlockMeta {
    pub block_idx: u32,
    pub num_rows: u32,
    pub scheme: EncodingScheme,
    pub encoded_bytes: u64,
    pub dict_size: Option<u32>,
    pub min_value: Option<Vec<u8>>,
    pub max_value: Option<Vec<u8>>,
}

/// CORRA correlation information.
#[derive(Debug, Clone, Serialize, Deserialize, Archive, RkyvSerialize, RkyvDeserialize)]
pub struct ColumnCorrelation {
    pub ref_col: String,
    pub corr_type: CorrType,
    pub offset_table: Option<Vec<u8>>,
    pub correlation_strength: Option<f64>,
}

#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Serialize,
    Deserialize,
    Archive,
    RkyvSerialize,
    RkyvDeserialize,
)]
pub enum CorrType {
    PeerDiff,
    Subaltern,
    Hierarchical,
}

/// LEA statistics.
#[derive(Debug, Clone, Serialize, Deserialize, Archive, RkyvSerialize, RkyvDeserialize)]
pub struct LEAStats {
    pub predicted_scheme: EncodingScheme,
    pub confidence: f64,
    pub used_lea: bool,
    pub reencode_count: u32,
}

/// Per-column encoding metadata.
#[derive(Debug, Clone, Serialize, Deserialize, Archive, RkyvSerialize, RkyvDeserialize)]
pub struct ColumnEncoding {
    pub column_name: String,
    pub scheme: EncodingScheme,
    pub blocks: Vec<BlockMeta>,
    pub correlation: Option<ColumnCorrelation>,
    pub lea: Option<LEAStats>,
    pub total_rows: u64,
    pub original_bytes: u64,
    pub encoded_bytes: u64,
}

impl ColumnEncoding {
    pub fn compression_ratio(&self) -> f64 {
        if self.encoded_bytes == 0 {
            0.0 // No encoding data available; ratio is 0, not INFINITY
        } else {
            self.original_bytes as f64 / self.encoded_bytes as f64
        }
    }

    pub fn new(column_name: String, scheme: EncodingScheme, total_rows: u64) -> Self {
        Self {
            column_name,
            scheme,
            blocks: Vec::new(),
            correlation: None,
            lea: None,
            total_rows,
            original_bytes: 0,
            encoded_bytes: 0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Archive, RkyvSerialize, RkyvDeserialize)]
pub struct TableEncodingConfig {
    pub column_overrides: std::collections::HashMap<String, EncodingScheme>,
    pub btrblocks_enabled: bool,
    pub corra_enabled: bool,
    pub lea_enabled: bool,
    pub compression_threshold: f64,
}

impl Default for TableEncodingConfig {
    fn default() -> Self {
        Self {
            column_overrides: std::collections::HashMap::new(),
            btrblocks_enabled: true,
            corra_enabled: true,
            lea_enabled: true,
            compression_threshold: 1.0,
        }
    }
}
