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
/// Returns `(stripped_sql, txn_id)` if found, or an error when malformed syntax is present.
pub fn parse_time_travel(sql: &str) -> Result<(String, Option<u64>), String> {
    let upper = sql.to_uppercase();

    // 尝试匹配 "AS OF TxnId <number>"
    // 先找 "AS OF" 关键字的位置
    let as_of_pos = match upper.find("AS OF TXNID") {
        Some(pos) => pos,
        None => return Ok((sql.to_string(), None)),
    };

    let before_as_of = &sql[..as_of_pos];
    if before_as_of
        .chars()
        .last()
        .is_some_and(|c| c.is_ascii_alphanumeric() || c == '_')
    {
        return Err(
            "malformed AS OF TxnId clause: missing token boundary before AS OF".to_string(),
        );
    }

    let as_of_start = as_of_pos;
    let txn_id_start = as_of_pos + "AS OF TXNID".len();
    let after_txn_id = &sql[txn_id_start..];

    // 提取数字部分（跳过空格）
    let Some(offset_to_digits) = after_txn_id.find(|c: char| !c.is_whitespace()) else {
        return Err("malformed AS OF TxnId clause: missing transaction id".to_string());
    };
    let digits_start = txn_id_start + offset_to_digits;
    let remaining = &sql[digits_start..];

    // 解析数字（支持十进制和 0x 十六进制）
    let (num_str, num_digits_len) = if remaining.starts_with("0x") || remaining.starts_with("0X") {
        let hex = &remaining[2..];
        let num_end = hex
            .len()
            .saturating_sub(hex.trim_end_matches(|c: char| c.is_ascii_hexdigit()).len());
        (&remaining[..2 + num_end], 2 + num_end)
    } else {
        let num_end = remaining.len().saturating_sub(
            remaining
                .trim_start_matches(|c: char| c.is_ascii_digit())
                .len(),
        );
        let num_digits = remaining[..num_end].len();
        (&remaining[..num_end], num_digits)
    };

    if num_str.is_empty() {
        return Err("malformed AS OF TxnId clause: missing transaction id".to_string());
    }

    let txn_id = if num_str.starts_with("0x") || num_str.starts_with("0X") {
        u64::from_str_radix(&num_str[2..], 16)
            .map_err(|_| format!("invalid AS OF TxnId hex literal: {num_str}"))?
    } else {
        num_str
            .parse::<u64>()
            .map_err(|_| format!("invalid AS OF TxnId literal: {num_str}"))?
    };

    let after_num = &sql[digits_start + num_digits_len..];
    if after_num
        .chars()
        .next()
        .is_some_and(|c| c.is_ascii_alphanumeric() || c == '_')
    {
        return Err("malformed AS OF TxnId clause: invalid trailing token".to_string());
    }

    // 剥离 Time-Travel 子句
    let before = sql[..as_of_start].trim();
    let after_num = after_num.trim();
    let stripped = if before.is_empty() {
        after_num.to_string()
    } else if after_num.is_empty() {
        before.to_string()
    } else if after_num.starts_with(';') || after_num.starts_with(',') || after_num.starts_with(')')
    {
        format!("{}{}", before, after_num)
    } else {
        format!("{} {}", before, after_num)
    };

    Ok((stripped, Some(txn_id)))
}
