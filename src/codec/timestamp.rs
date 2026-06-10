//! 时间戳工具函数
//!
//! 提供获取当前时间戳的统一接口

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// D2 fix: Last valid timestamp for fallback when time goes backwards.
/// Using AtomicU64 for lock-free reads in hot paths.
static LAST_VALID_TS_SECS: AtomicU64 = AtomicU64::new(0);
static LAST_VALID_TS_MILLIS: AtomicU64 = AtomicU64::new(0);
static LAST_VALID_TS_MICROS: AtomicU64 = AtomicU64::new(0);

/// 获取当前时间戳（秒）
/// D2 fix: Falls back to last valid timestamp instead of panicking on time drift.
pub fn current_timestamp_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| {
            let ts = d.as_secs();
            LAST_VALID_TS_SECS.store(ts, Ordering::Relaxed);
            ts
        })
        .unwrap_or_else(|_| {
            tracing::warn!("Time went backwards, using last valid timestamp");
            LAST_VALID_TS_SECS.load(Ordering::Relaxed)
        })
}

/// 获取当前时间戳（毫秒）
/// D2 fix: Falls back to last valid timestamp instead of panicking on time drift.
pub fn current_timestamp_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| {
            let ts = d.as_millis() as u64;
            LAST_VALID_TS_MILLIS.store(ts, Ordering::Relaxed);
            ts
        })
        .unwrap_or_else(|_| {
            tracing::warn!("Time went backwards, using last valid timestamp");
            LAST_VALID_TS_MILLIS.load(Ordering::Relaxed)
        })
}

/// 获取当前时间戳（微秒）
#[allow(dead_code)]
/// D2 fix: Falls back to last valid timestamp instead of panicking on time drift.
pub fn current_timestamp_micros() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| {
            let ts = d.as_micros() as u64;
            LAST_VALID_TS_MICROS.store(ts, Ordering::Relaxed);
            ts
        })
        .unwrap_or_else(|_| {
            tracing::warn!("Time went backwards, using last valid timestamp");
            LAST_VALID_TS_MICROS.load(Ordering::Relaxed)
        })
}
