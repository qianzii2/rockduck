//! 时间戳工具函数
//!
//! 提供获取当前时间戳的统一接口

use std::time::{SystemTime, UNIX_EPOCH};

/// 获取当前时间戳（秒）
pub fn current_timestamp_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("Time went backwards")
        .as_secs()
}

/// 获取当前时间戳（毫秒）
pub fn current_timestamp_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("Time went backwards")
        .as_millis() as u64
}

/// 获取当前时间戳（微秒）
#[allow(dead_code)]
pub fn current_timestamp_micros() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("Time went backwards")
        .as_micros() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_current_timestamp_secs_positive() {
        let ts1 = current_timestamp_secs();
        std::thread::sleep(std::time::Duration::from_millis(10));
        let ts2 = current_timestamp_secs();
        assert!(ts2 > 0);
        assert!(ts2 >= ts1);
    }

    #[test]
    fn test_current_timestamp_millis_positive() {
        let ts1 = current_timestamp_millis();
        std::thread::sleep(std::time::Duration::from_millis(5));
        let ts2 = current_timestamp_millis();
        assert!(ts2 > 0);
        assert!(ts2 >= ts1);
    }

    #[test]
    fn test_current_timestamp_millis_greater_than_secs() {
        let secs = current_timestamp_secs();
        let millis = current_timestamp_millis();
        assert!(millis >= secs * 1000);
    }

    #[test]
    fn test_current_timestamp_micros_positive() {
        let ts1 = current_timestamp_micros();
        std::thread::sleep(std::time::Duration::from_micros(100));
        let ts2 = current_timestamp_micros();
        assert!(ts2 > 0);
        assert!(ts2 >= ts1);
    }

    #[test]
    fn test_current_timestamp_micros_greater_than_millis() {
        let millis = current_timestamp_millis();
        let micros = current_timestamp_micros();
        // micros should be >= millis * 1000
        assert!(micros >= millis * 1000);
    }

    #[test]
    fn test_timestamps_are_monotonic() {
        let secs = current_timestamp_secs();
        let millis = current_timestamp_millis();
        let micros = current_timestamp_micros();

        assert!(micros >= millis * 1000);
        assert!(millis >= secs * 1000);

        // All should be based on same epoch
        assert!(secs > 1_700_000_000); // Reasonable Unix timestamp
    }
}
