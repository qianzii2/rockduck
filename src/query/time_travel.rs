//! Time-Travel 查询解析
//!
//! 支持 `AS OF TxnId <id>` 语法解析，将 Time-Travel 子句从 SQL 中剥离，
//! 并返回 (stripped_sql, txn_id)。

/// 解析 Time-Travel 子句 `AS OF TxnId <id>`
///
/// 支持的格式：
/// - `SELECT * FROM t AS OF TxnId 12345`
/// - `SELECT * FROM t AS OF TxnId12345`（无空格）
/// - `SELECT * FROM t AS OF TxnId 0x1F3A`（十六进制）
///
/// Returns `(stripped_sql, txn_id)` if found, or `(sql.to_string(), None)` if not found.
pub fn parse_time_travel(sql: &str) -> (String, Option<u64>) {
    let upper = sql.to_uppercase();

    // 尝试匹配 "AS OF TxnId <number>"
    // 先找 "AS OF" 关键字的位置
    let as_of_pos = upper.find("AS OF TXNID");
    let (as_of_start, txn_id_start) = match as_of_pos {
        Some(pos) => (pos, pos + "AS OF TXNID".len()),
        None => return (sql.to_string(), None),
    };

    let after_txn_id = &sql[txn_id_start..];

    // 提取数字部分（跳过空格）
    let digits_start = after_txn_id.find(|c: char| !c.is_whitespace());
    if digits_start.is_none() {
        return (sql.to_string(), None);
    }
    let digits_start = txn_id_start + digits_start.unwrap();
    let remaining = &sql[digits_start..];

    // 解析数字（支持十进制和 0x 十六进制）
    let num_str = if remaining.starts_with("0x") || remaining.starts_with("0X") {
        let hex = &remaining[2..];
        let end = hex.len().saturating_sub(hex.trim_start_matches(|c: char| c.is_ascii_hexdigit()).len());
        &remaining[..2 + end]
    } else {
        let end = remaining.len().saturating_sub(remaining.trim_start_matches(|c: char| c.is_ascii_digit()).len());
        &remaining[..end]
    };

    if num_str.is_empty() {
        return (sql.to_string(), None);
    }

    let txn_id = if num_str.starts_with("0x") || num_str.starts_with("0X") {
        u64::from_str_radix(&num_str[2..], 16).ok()
    } else {
        num_str.parse::<u64>().ok()
    };

    let txn_id = match txn_id {
        Some(id) => id,
        None => return (sql.to_string(), None),
    };

    // 剥离 Time-Travel 子句
    let before = sql[..as_of_start].trim();
    let after_num = &sql[digits_start + num_str.len()..].trim();
    let stripped = if before.is_empty() {
        after_num.to_string()
    } else if after_num.is_empty() {
        before.to_string()
    } else {
        format!("{} {}", before, after_num)
    };

    (stripped, Some(txn_id))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_time_trail_basic() {
        let (sql, txn_id) = parse_time_travel("SELECT * FROM t AS OF TxnId 12345");
        assert_eq!(sql, "SELECT * FROM t");
        assert_eq!(txn_id, Some(12345));
    }

    #[test]
    fn test_parse_time_trail_no_space() {
        let (sql, txn_id) = parse_time_travel("SELECT * FROM t AS OF TxnId12345");
        assert_eq!(sql, "SELECT * FROM t");
        assert_eq!(txn_id, Some(12345));
    }

    #[test]
    fn test_parse_time_trail_hex() {
        let (sql, txn_id) = parse_time_travel("SELECT * FROM t AS OF TxnId 0x1F3A");
        assert_eq!(sql, "SELECT * FROM t");
        assert_eq!(txn_id, Some(0x1F3A));
    }

    #[test]
    fn test_parse_time_trail_uppercase_txnid() {
        let (sql, txn_id) = parse_time_travel("SELECT * FROM t AS OF TXNID 999");
        assert_eq!(sql, "SELECT * FROM t");
        assert_eq!(txn_id, Some(999));
    }

    #[test]
    fn test_parse_time_trail_no_clause() {
        let (sql, txn_id) = parse_time_travel("SELECT * FROM t WHERE id > 5");
        assert_eq!(sql, "SELECT * FROM t WHERE id > 5");
        assert_eq!(txn_id, None);
    }

    #[test]
    fn test_parse_time_trail_middle_of_sql() {
        let (sql, txn_id) = parse_time_travel("SELECT * FROM users AS OF TxnId 100 WHERE age > 30");
        assert_eq!(sql, "SELECT * FROM users WHERE age > 30");
        assert_eq!(txn_id, Some(100));
    }

    #[test]
    fn test_parse_time_trail_zero() {
        let (sql, txn_id) = parse_time_travel("SELECT * FROM t AS OF TxnId 0");
        assert_eq!(sql, "SELECT * FROM t");
        assert_eq!(txn_id, Some(0));
    }

    #[test]
    fn test_parse_time_trail_large_number() {
        let (sql, txn_id) = parse_time_travel("SELECT * FROM t AS OF TxnId 999999999999");
        assert_eq!(sql, "SELECT * FROM t");
        assert_eq!(txn_id, Some(999999999999));
    }

    #[test]
    fn test_parse_time_trail_case_insensitive() {
        let (sql, txn_id) = parse_time_travel("select * from t as of txnid 42");
        assert_eq!(sql, "select * from t");
        assert_eq!(txn_id, Some(42));
    }

    #[test]
    fn test_parse_time_trail_hex_uppercase() {
        let (sql, txn_id) = parse_time_travel("SELECT * FROM t AS OF TxnId 0XFF");
        assert!(sql.contains("SELECT * FROM t"));
        assert!(!sql.contains("AS OF TxnId"));
        assert_eq!(txn_id, Some(0xFF));
    }

    #[test]
    fn test_parse_time_trail_multiple_spaces() {
        let (sql, txn_id) = parse_time_travel("SELECT * FROM t AS OF TxnId   555");
        assert_eq!(sql, "SELECT * FROM t");
        assert_eq!(txn_id, Some(555));
    }

    #[test]
    fn test_parse_time_trail_trailing_query() {
        let (sql, txn_id) = parse_time_travel("SELECT * FROM t AS OF TxnId 10 GROUP BY x");
        assert_eq!(sql, "SELECT * FROM t GROUP BY x");
        assert_eq!(txn_id, Some(10));
    }

    // ============================================================
    // ORDER BY clause preservation
    // ============================================================

    #[test]
    fn test_parse_time_travel_preserves_order_by() {
        let (sql, txn_id) = parse_time_travel(
            "SELECT id, name FROM users WHERE age > 30 AS OF TxnId 42 ORDER BY id"
        );
        assert_eq!(txn_id, Some(42));
        assert!(
            sql.contains("ORDER BY id"),
            "ORDER BY clause must be preserved, got: {}",
            sql
        );
        assert!(
            sql.contains("SELECT id, name FROM users WHERE age > 30"),
            "SELECT and WHERE must be preserved, got: {}",
            sql
        );
        assert!(
            !sql.contains("AS OF TxnId"),
            "Time-Travel clause must be stripped, got: {}",
            sql
        );
    }

    #[test]
    fn test_parse_time_travel_preserves_order_by_with_limit() {
        let (sql, txn_id) = parse_time_travel(
            "SELECT * FROM t AS OF TxnId 7 ORDER BY score DESC LIMIT 10"
        );
        assert_eq!(txn_id, Some(7));
        assert!(sql.contains("ORDER BY score DESC"));
        assert!(sql.contains("LIMIT 10"));
        assert!(!sql.contains("AS OF TxnId"));
    }

    // ============================================================
    // Hex zero boundary
    // ============================================================

    #[test]
    fn test_parse_time_travel_hex_zero() {
        let (sql, txn_id) = parse_time_travel("SELECT * FROM t AS OF TxnId 0x0");
        assert_eq!(
            txn_id, Some(0),
            "0x0 must parse to TxnId 0, got {:?}",
            txn_id
        );
        assert_eq!(
            sql, "SELECT * FROM t",
            "SQL must be clean after stripping, got: {}",
            sql
        );
    }

    #[test]
    fn test_parse_time_trail_no_trailing_space() {
        let (sql, txn_id) = parse_time_travel("SELECT * FROM t AS OF TxnId 1;");
        // Stripping "AS OF TxnId 1" from "SELECT * FROM t AS OF TxnId 1;" leaves "SELECT * FROM t ;"
        // (space before semicolon because we normalize spacing)
        assert!(sql.contains("SELECT * FROM t"));
        assert!(sql.contains(';'));
        assert_eq!(txn_id, Some(1));
    }
}
