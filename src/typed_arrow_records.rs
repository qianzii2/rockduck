//! typed-arrow integration module
//!
//! Provides type-safe Arrow Record conversion utilities.
//!
//! This module contains stub type definitions for typed-arrow integration.
//! The actual Record definitions should be placed in business-specific modules.

/// Example user record for Arrow conversion (stub)
#[derive(Debug, Clone)]
pub struct UserRecord {
    pub id: i64,
    pub name: String,
    pub age: Option<i32>,
}

/// Example order record for Arrow conversion (stub)
#[derive(Debug, Clone)]
pub struct OrderRecord {
    pub order_id: i64,
    pub user_id: i64,
    pub amount: f64,
    pub created_at: i64,
}

/// Example log record for Arrow conversion (stub)
#[derive(Debug, Clone)]
pub struct LogRecord {
    pub timestamp: i64,
    pub level: i32,
    pub message: String,
}
