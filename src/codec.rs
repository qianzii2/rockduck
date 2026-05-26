//! 二进制序列化工具模块
//!
//! 使用 bincode-next 进行高性能二进制序列化
//! 作为 serde_json 的高性能替代

pub mod timestamp;

pub use timestamp::{current_timestamp_secs, current_timestamp_millis};

use thiserror::Error;

#[derive(Error, Debug)]
pub enum CodecError {
    #[error("Encoding error: {0}")]
    Encode(#[from] bincode_next::error::EncodeError),
    #[error("Decoding error: {0}")]
    Decode(#[from] bincode_next::error::DecodeError),
}

/// 默认配置
#[inline]
pub fn config() -> bincode_next::config::Configuration {
    bincode_next::config::standard()
}

/// 将对象编码为二进制字节向量
pub fn encode<T: bincode_next::Encode>(value: &T) -> Result<Vec<u8>, CodecError> {
    Ok(bincode_next::encode_to_vec(value, config())?)
}

/// 从二进制字节向量解码为对象（忽略读取的字节数）
pub fn decode<T>(bytes: &[u8]) -> Result<T, CodecError>
where
    T: bincode_next::Decode<()>,
{
    let (value, _) = bincode_next::decode_from_slice::<T, _>(bytes, config())?;
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ============================================================
    // encode / decode roundtrip tests
    // ============================================================

    #[test]
    fn test_encode_decode_u32() {
        let original = vec![1u32, 2, 3, 4, 5];
        let encoded = encode(&original).unwrap();
        let decoded: Vec<u32> = decode(&encoded).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn test_encode_decode_u64() {
        let original: Vec<u64> = vec![0, 1, u64::MAX, 12345678901234];
        let encoded = encode(&original).unwrap();
        let decoded: Vec<u64> = decode(&encoded).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn test_encode_decode_i32() {
        let original: Vec<i32> = vec![-1, 0, 1, i32::MIN, i32::MAX];
        let encoded = encode(&original).unwrap();
        let decoded: Vec<i32> = decode(&encoded).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn test_encode_decode_i64() {
        let original: Vec<i64> = vec![-1, 0, 1, i64::MIN, i64::MAX, -999999999];
        let encoded = encode(&original).unwrap();
        let decoded: Vec<i64> = decode(&encoded).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn test_encode_decode_string() {
        let original = vec!["hello".to_string(), "world".to_string(), "".to_string(), "unicode: \u{1F600}".to_string()];
        let encoded = encode(&original).unwrap();
        let decoded: Vec<String> = decode(&encoded).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn test_encode_decode_vec_u8() {
        let original: Vec<u8> = (0u8..=255u8).collect();
        let encoded = encode(&original).unwrap();
        let decoded: Vec<u8> = decode(&encoded).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn test_encode_decode_empty_vec() {
        let original: Vec<u8> = vec![];
        let encoded = encode(&original).unwrap();
        let decoded: Vec<u8> = decode(&encoded).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn test_encode_decode_hashmap() {
        use std::collections::HashMap;
        let mut original: HashMap<String, i64> = HashMap::new();
        original.insert("key1".to_string(), 100i64);
        original.insert("key2".to_string(), -200i64);
        let encoded = encode(&original).unwrap();
        let decoded: HashMap<String, i64> = decode(&encoded).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn test_encode_decode_option() {
        let original: Option<i64> = Some(42);
        let encoded = encode(&original).unwrap();
        let decoded: Option<i64> = decode(&encoded).unwrap();
        assert_eq!(original, decoded);

        let none: Option<i64> = None;
        let encoded = encode(&none).unwrap();
        let decoded: Option<i64> = decode(&encoded).unwrap();
        assert_eq!(none, decoded);
    }

    #[test]
    fn test_encode_decode_tuple() {
        let original: (u32, String, i64) = (42u32, "hello".to_string(), -123i64);
        let encoded = encode(&original).unwrap();
        let decoded: (u32, String, i64) = decode(&encoded).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn test_encode_decode_nested_struct() {
        #[derive(bincode_next::Encode, bincode_next::Decode, PartialEq, Debug)]
        struct Inner {
            value: i32,
            name: String,
        }
        #[derive(bincode_next::Encode, bincode_next::Decode, PartialEq, Debug)]
        struct Outer {
            id: u64,
            inner: Inner,
        }
        let original = Outer {
            id: 123,
            inner: Inner { value: -456, name: "test".to_string() },
        };
        let encoded = encode(&original).unwrap();
        let decoded: Outer = decode(&encoded).unwrap();
        assert_eq!(original, decoded);
    }

    // ============================================================
    // Error handling tests
    // ============================================================

    #[test]
    fn test_decode_corrupted_bytes() {
        let corrupted = vec![0xFF, 0xFE, 0xFD];
        let result: Result<Vec<u32>, _> = decode(&corrupted);
        assert!(result.is_err());
    }

    #[test]
    fn test_decode_empty_bytes() {
        let result: Result<Vec<u32>, _> = decode(&[]);
        assert!(result.is_err());
    }

    // ============================================================
    // Config test
    // ============================================================

    #[test]
    fn test_config_returns_standard() {
        let config = config();
        // Verify it doesn't panic and returns a config
        let _ = format!("{:?}", config);
    }

    // ============================================================
    // Roundtrip with SegmentMeta-like struct
    // ============================================================

    #[test]
    fn test_encode_decode_with_custom_struct() {
        use std::collections::HashMap;
        #[derive(bincode_next::Encode, bincode_next::Decode, PartialEq, Debug)]
        struct TestMeta {
            id: String,
            count: u64,
            data: HashMap<String, Vec<u8>>,
        }

        let mut data: HashMap<String, Vec<u8>> = HashMap::new();
        data.insert("col1".to_string(), vec![1, 2, 3]);
        data.insert("col2".to_string(), vec![10, 20]);

        let original = TestMeta {
            id: "seg_001".to_string(),
            count: 1000,
            data,
        };

        let encoded = encode(&original).unwrap();
        let decoded: TestMeta = decode(&encoded).unwrap();
        assert_eq!(original.id, decoded.id);
        assert_eq!(original.count, decoded.count);
        assert_eq!(original.data.len(), decoded.data.len());
    }

    // ============================================================
    // encode_to_vec vs encode_to_slice consistency
    // ============================================================

    #[test]
    fn test_encode_large_data() {
        let large: Vec<u32> = (0u32..10000).collect();
        let encoded = encode(&large).unwrap();
        let decoded: Vec<u32> = decode(&encoded).unwrap();
        assert_eq!(large, decoded);
    }
}
