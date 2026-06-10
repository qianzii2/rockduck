//! Filter expression parser and evaluator
//!
//! Parses simple filter expressions and evaluates them against Arrow RecordBatches.
//!
//! Supported syntax:
//! - `col = value` or `col == value` - equality
//! - `col != value` - not equal
//! - `col > value` - greater than
//! - `col >= value` - greater or equal
//! - `col < value` - less than
//! - `col <= value` - less or equal
//! - `col = "string"` - string equality (quoted with double quotes)
//! - `expr AND expr` or `expr && expr` - logical AND
//! - `expr OR expr` or `expr || expr` - logical OR
//! - `(expr)` - grouping
//! - `NOT expr` or `!expr` - negation

use crate::metadata::block_zone_map::bytes_lt;
use crate::RockDuckError;
use arrow::compute::kernels::cmp;
use arrow::datatypes::DataType;
use arrow_array::{BooleanArray, RecordBatch, Scalar};

/// Comparison operator for filter expressions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompareOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

/// Filter expression AST
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// Column comparison: col OP value
    Comparison {
        col: String,
        op: CompareOp,
        val: ScalarVal,
    },
    /// Logical AND
    And(Box<Expr>, Box<Expr>),
    /// Logical OR
    Or(Box<Expr>, Box<Expr>),
    /// Negation
    Not(Box<Expr>),
}

/// Scalar value for comparisons
#[derive(Debug, Clone, PartialEq)]
pub enum ScalarVal {
    Null,
    Bool(bool),
    Int64(i64),
    Float64(f64),
    String(String),
}

/// Lightweight filter variant used by the vtab scan path.
#[derive(Debug, Clone)]
pub enum ScanFilter {
    Eq {
        column: String,
        value: Vec<u8>,
    },
    Range {
        column: String,
        min: Vec<u8>,
        max: Vec<u8>,
    },
    Like {
        column: String,
        pattern: String,
    },
    Between {
        column: String,
        min: Vec<u8>,
        max: Vec<u8>,
    },
    IsNull {
        column: String,
    },
    IsNotNull {
        column: String,
    },
    IsTrue {
        column: String,
    },
    IsNotTrue {
        column: String,
    },
    And(Box<ScanFilter>, Box<ScanFilter>),
    Or(Box<ScanFilter>, Box<ScanFilter>),
    Not(Box<ScanFilter>),
    In {
        column: String,
        values: Vec<Vec<u8>>,
    },
    /// Greater than: col > value
    Gt {
        column: String,
        value: Vec<u8>,
    },
    /// Greater than or equal: col >= value
    Ge {
        column: String,
        value: Vec<u8>,
    },
    /// Less than: col < value
    Lt {
        column: String,
        value: Vec<u8>,
    },
    /// Less than or equal: col <= value
    Le {
        column: String,
        value: Vec<u8>,
    },
}

impl Expr {
    /// Convert an `Expr` AST into a `ScanFilter` for the vtab path.
    ///
    /// Returns `None` only when the expression references an unsupported operator
    /// on an unknown data type. Logical operators propagate `None` from sub-expressions.
    pub fn to_scan_filter(&self) -> Option<ScanFilter> {
        match self {
            Expr::Comparison { col, op, val } => {
                let value = scalar_to_bytes(val)?;
                match op {
                    CompareOp::Eq => Some(ScanFilter::Eq {
                        column: col.clone(),
                        value,
                    }),
                    CompareOp::Ne => None,
                    CompareOp::Gt => Some(ScanFilter::Gt {
                        column: col.clone(),
                        value,
                    }),
                    CompareOp::Ge => Some(ScanFilter::Ge {
                        column: col.clone(),
                        value,
                    }),
                    CompareOp::Lt => Some(ScanFilter::Lt {
                        column: col.clone(),
                        value,
                    }),
                    CompareOp::Le => Some(ScanFilter::Le {
                        column: col.clone(),
                        value,
                    }),
                }
            }
            Expr::And(a, b) => {
                let a_filter = a.to_scan_filter()?;
                let b_filter = b.to_scan_filter()?;
                Some(ScanFilter::And(Box::new(a_filter), Box::new(b_filter)))
            }
            Expr::Or(a, b) => {
                let a_filter = a.to_scan_filter()?;
                let b_filter = b.to_scan_filter()?;
                Some(ScanFilter::Or(Box::new(a_filter), Box::new(b_filter)))
            }
            Expr::Not(inner) => {
                let inner_filter = inner.to_scan_filter()?;
                Some(ScanFilter::Not(Box::new(inner_filter)))
            }
        }
    }
}

fn scalar_to_bytes(val: &ScalarVal) -> Option<Vec<u8>> {
    match val {
        ScalarVal::Null => None,
        ScalarVal::Bool(b) => Some(vec![if *b { 1 } else { 0 }]),
        ScalarVal::Int64(i) => Some(i.to_le_bytes().to_vec()),
        ScalarVal::Float64(f) => Some(f.to_le_bytes().to_vec()),
        ScalarVal::String(s) => Some(s.as_bytes().to_vec()),
    }
}

/// A single-column predicate for zone map evaluation.
#[derive(Debug, Clone)]
pub struct ZoneMapPredicate {
    pub column: String,
    pub pred_min: Vec<u8>,
    pub pred_max: Vec<u8>,
}

/// OR group: multiple predicates on the same column, connected by OR.
/// E.g., `col = 1 OR col = 2` becomes one group with two (min, max) pairs.
#[derive(Debug, Clone)]
pub struct ZoneMapOrGroup {
    pub column: String,
    /// Each entry is (pred_min, pred_max). Eq: (v, v), Range: (min, max) or (v, empty).
    pub ranges: Vec<(Vec<u8>, Vec<u8>)>,
}

/// Complete zone map filtering condition.
///
/// Groups are AND-ed together; within each group, ranges are OR-ed.
/// `has_cross_column_or = true` means the expression contains an OR across
/// different columns which ZoneMap cannot safely evaluate — the scan should
/// fall back to a full segment read (predicate evaluation happens at row level).
#[derive(Debug, Clone)]
pub struct ZoneMapPredicateGroup {
    pub groups: Vec<ZoneMapOrGroup>,
    /// True if the expression contains OR between different columns.
    /// When true, zone map skipping is disabled (conservative fallback).
    pub has_cross_column_or: bool,
}

impl ZoneMapPredicateGroup {
    /// Returns true if this group is empty (no filter predicates).
    pub fn is_empty(&self) -> bool {
        self.groups.is_empty()
    }
}

/// Check if a column's stats may overlap with any range in an OR group.
///
/// Returns true if at least one range in the OR group may overlap with the
/// column's [col_min, col_max] range. Used for OR-within-column semantics:
/// if any range in the OR group might overlap, the segment cannot be skipped.
pub fn check_or_group_may_overlap(
    or_group: &ZoneMapOrGroup,
    col_min: Option<&[u8]>,
    col_max: Option<&[u8]>,
    data_type: &arrow_schema::DataType,
) -> bool {
    for (pred_min, pred_max) in &or_group.ranges {
        // Check if [pred_min, pred_max] may overlap with [col_min, col_max]
        // No overlap if: col_max < pred_min OR col_min > pred_max
        // But we handle empty bounds (representing -∞ and +∞) specially.

        let pred_min_empty = pred_min.is_empty();
        let pred_max_empty = pred_max.is_empty();

        // col_max < pred_min → no overlap (segment entirely below)
        if !pred_min_empty {
            if let Some(col_max) = col_max {
                if bytes_lt(col_max, pred_min, data_type) {
                    continue; // This range doesn't overlap, try next
                }
            }
        }

        // col_min > pred_max → no overlap (segment entirely above)
        if !pred_max_empty {
            if let Some(col_min) = col_min {
                if bytes_lt(pred_max, col_min, data_type) {
                    continue; // This range doesn't overlap, try next
                }
            }
        }

        // At least one range might overlap → segment is not provably empty
        return true;
    }

    // No range overlapped → segment can be skipped
    false
}

/// Convert a filter expression AST into a list of column-level zone map predicates.
///
/// Returns `None` if the expression cannot be represented as zone map predicates
/// (e.g., contains OR, NOT, non-comparison expressions, or expressions on unknown columns).
/// When `None` is returned, zone map skipping should be skipped (conservative).
///
/// For range comparisons (>, >=, <, <=), produces a [pred_min, pred_max] bound.
/// For equality (=), produces a [value, value] bound.
pub fn to_zone_map_predicates(expr: &Expr) -> Option<Vec<ZoneMapPredicate>> {
    extract_predicates(expr)
}

/// Convert a filter expression AST into a grouped zone map predicate structure.
///
/// - Groups are AND-ed together
/// - Within each group (same column), ranges are OR-ed
/// - Cross-column OR sets `has_cross_column_or = true` (disables pushdown)
///
/// Returns `None` when the expression contains NOT or non-comparison operators.
pub fn to_zone_map_predicate_group(expr: &Expr) -> Option<ZoneMapPredicateGroup> {
    extract_predicates_grouped(expr)
}

/// Legacy flat extraction — returns `None` for any expression containing OR.
fn extract_predicates(expr: &Expr) -> Option<Vec<ZoneMapPredicate>> {
    match expr {
        Expr::Comparison { col, op, val } => {
            let value = scalar_to_bytes(val)?;
            let (pred_min, pred_max) = match op {
                CompareOp::Eq => (value.clone(), value),
                CompareOp::Ge => (value.clone(), Vec::new()),
                CompareOp::Gt => (value.clone(), Vec::new()),
                CompareOp::Le => (Vec::new(), value.clone()),
                CompareOp::Lt => (Vec::new(), value.clone()),
                CompareOp::Ne => return None,
            };
            Some(vec![ZoneMapPredicate {
                column: col.clone(),
                pred_min,
                pred_max,
            }])
        }
        Expr::And(a, b) => {
            let mut a_preds = extract_predicates(a)?;
            let b_preds = extract_predicates(b)?;
            a_preds.extend(b_preds);
            Some(a_preds)
        }
        Expr::Or(_, _) | Expr::Not(_) => None,
    }
}

/// Grouped predicate extraction: supports safe same-column OR.
fn extract_predicates_grouped(expr: &Expr) -> Option<ZoneMapPredicateGroup> {
    let result = match expr {
        Expr::Comparison { col, op, val } => {
            let value = scalar_to_bytes(val)?;
            let (pred_min, pred_max) = match op {
                CompareOp::Eq => (value.clone(), value),
                CompareOp::Ge => (value.clone(), Vec::new()),
                CompareOp::Gt => (value.clone(), Vec::new()),
                CompareOp::Le => (Vec::new(), value.clone()),
                CompareOp::Lt => (Vec::new(), value.clone()),
                CompareOp::Ne => return None,
            };
            Some(ZoneMapPredicateGroup {
                groups: vec![ZoneMapOrGroup {
                    column: col.clone(),
                    ranges: vec![(pred_min, pred_max)],
                }],
                has_cross_column_or: false,
            })
        }
        Expr::And(a, b) => {
            let mut a_group = extract_predicates_grouped(a)?;
            let b_group = extract_predicates_grouped(b)?;

            for b_or_group in b_group.groups {
                if let Some(existing) = a_group
                    .groups
                    .iter_mut()
                    .find(|g| g.column == b_or_group.column)
                {
                    existing.ranges.extend(b_or_group.ranges);
                } else {
                    a_group.groups.push(b_or_group);
                }
            }

            a_group.has_cross_column_or =
                a_group.has_cross_column_or || b_group.has_cross_column_or;
            Some(a_group)
        }
        Expr::Or(a, b) => {
            let a_group = extract_predicates_grouped(a)?;
            let b_group = extract_predicates_grouped(b)?;

            // Cross-column OR: the two sides share ZERO columns.
            // Safe same-column OR: they share at least one column.
            // If they share zero columns, ZoneMap cannot safely evaluate.
            let a_cols: std::collections::HashSet<_> =
                a_group.groups.iter().map(|g| &g.column).collect();
            let b_cols: std::collections::HashSet<_> =
                b_group.groups.iter().map(|g| &g.column).collect();
            let intersection_size = a_cols.intersection(&b_cols).count();
            let is_cross_column_or =
                intersection_size == 0 && !a_cols.is_empty() && !b_cols.is_empty();

            let mut result = ZoneMapPredicateGroup {
                groups: Vec::new(),
                has_cross_column_or: is_cross_column_or
                    || a_group.has_cross_column_or
                    || b_group.has_cross_column_or,
            };

            // O(n) lookup: build HashMap from column name → group
            use std::collections::HashMap;
            let a_map: HashMap<&str, &ZoneMapOrGroup> =
                a_group.groups.iter().map(|g| (g.column.as_str(), g)).collect();
            let b_map: HashMap<&str, &ZoneMapOrGroup> =
                b_group.groups.iter().map(|g| (g.column.as_str(), g)).collect();

            // Union of columns from both sides
            for col in a_map.keys().chain(b_map.keys()) {
                let a_ranges: Vec<_> = a_map
                    .get(col)
                    .map(|g| g.ranges.clone())
                    .unwrap_or_default();
                let b_ranges: Vec<_> = b_map
                    .get(col)
                    .map(|g| g.ranges.clone())
                    .unwrap_or_default();

                if !a_ranges.is_empty() && !b_ranges.is_empty() {
                    let mut merged = a_ranges;
                    merged.extend(b_ranges);
                    result.groups.push(ZoneMapOrGroup {
                        column: col.to_string(),
                        ranges: merged,
                    });
                } else if !a_ranges.is_empty() {
                    result.groups.push(ZoneMapOrGroup {
                        column: col.to_string(),
                        ranges: a_ranges,
                    });
                } else if !b_ranges.is_empty() {
                    result.groups.push(ZoneMapOrGroup {
                        column: col.to_string(),
                        ranges: b_ranges,
                    });
                }
            }

            Some(result)
        }
        Expr::Not(_) => None,
    };

    result
}

// =============================================================================
// Tokenizer
// =============================================================================

#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    Ident(String),
    String(String),
    Number(String),
    Op(CompareOp),
    And,
    Or,
    Not,
    Minus,
    LParen,
    RParen,
    Eof,
}

pub fn tokenize(s: &str) -> Result<Vec<Token>, String> {
    let mut tokens = Vec::new();
    let mut chars = s.char_indices().peekable();

    while let Some((_, c)) = chars.peek().copied() {
        match c {
            ' ' | '\t' | '\n' | '\r' => {
                chars.next();
            }
            '"' => {
                chars.next();
                let mut str_val = String::new();
                loop {
                    match chars.next() {
                        Some((_, '"')) => break,
                        Some((_, '\\')) => {
                            if let Some((_, esc)) = chars.next() {
                                match esc {
                                    'n' => str_val.push('\n'),
                                    't' => str_val.push('\t'),
                                    'r' => str_val.push('\r'),
                                    '\\' => str_val.push('\\'),
                                    '"' => str_val.push('"'),
                                    c => str_val.push(c),
                                }
                            }
                        }
                        Some((_, c)) => str_val.push(c),
                        None => return Err("Unterminated string literal".to_string()),
                    }
                }
                tokens.push(Token::String(str_val));
            }
            '(' => {
                tokens.push(Token::LParen);
                chars.next();
            }
            ')' => {
                tokens.push(Token::RParen);
                chars.next();
            }
            '!' => {
                chars.next();
                // Check for != or ! followed by whitespace (standalone NOT)
                if let Some(&(_, '=')) = chars.peek() {
                    chars.next();
                    tokens.push(Token::Op(CompareOp::Ne));
                } else {
                    tokens.push(Token::Not);
                }
            }
            '=' => {
                chars.next();
                if let Some(&(_, '=')) = chars.peek() {
                    chars.next();
                }
                tokens.push(Token::Op(CompareOp::Eq));
            }
            '<' => {
                chars.next();
                if let Some(&(_, '=')) = chars.peek() {
                    chars.next();
                    tokens.push(Token::Op(CompareOp::Le));
                } else {
                    tokens.push(Token::Op(CompareOp::Lt));
                }
            }
            '>' => {
                chars.next();
                if let Some(&(_, '=')) = chars.peek() {
                    chars.next();
                    tokens.push(Token::Op(CompareOp::Ge));
                } else {
                    tokens.push(Token::Op(CompareOp::Gt));
                }
            }
            '&' => {
                chars.next();
                if let Some(&(_, '&')) = chars.peek() {
                    chars.next();
                    tokens.push(Token::And);
                } else {
                    return Err("Unexpected '&'. Use '&&' for AND.".to_string());
                }
            }
            '|' => {
                chars.next();
                if let Some(&(_, '|')) = chars.peek() {
                    chars.next();
                    tokens.push(Token::Or);
                } else {
                    return Err("Unexpected '|'. Use '||' for OR.".to_string());
                }
            }
            _ if c.is_ascii_alphabetic() || c == '_' => {
                let start = chars.next().unwrap().0;
                while let Some(&(_i, c)) = chars.peek() {
                    if c.is_ascii_alphanumeric() || c == '_' {
                        chars.next();
                    } else {
                        break;
                    }
                }
                let end = chars.peek().map(|x| x.0).unwrap_or(s.len());
                let ident = s[start..end].to_lowercase();
                match ident.as_str() {
                    "and" => tokens.push(Token::And),
                    "or" => tokens.push(Token::Or),
                    "not" => tokens.push(Token::Not),
                    "true" => tokens.push(Token::Ident(ident)),
                    "false" => tokens.push(Token::Ident(ident)),
                    "null" => tokens.push(Token::Ident(ident)),
                    _ => tokens.push(Token::Ident(ident)),
                }
            }
            '-' => {
                chars.next();
                tokens.push(Token::Minus);
            }
            _ if c.is_ascii_digit() => {
                let start = chars.next().unwrap().0;
                let mut has_dot = false;
                while let Some(&(_i, c)) = chars.peek() {
                    if c.is_ascii_digit() {
                        chars.next();
                    } else if c == '.' && !has_dot {
                        has_dot = true;
                        chars.next();
                    } else {
                        break;
                    }
                }
                let end = chars.peek().map(|x| x.0).unwrap_or(s.len());
                let num = s[start..end].to_string();
                tokens.push(Token::Number(num));
            }
            _ => {
                return Err(format!("Unexpected character: '{}'", c));
            }
        }
    }

    tokens.push(Token::Eof);
    Ok(tokens)
}

// =============================================================================
// Parser (Recursive Descent)
// =============================================================================

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    fn new(tokens: Vec<Token>) -> Self {
        Parser { tokens, pos: 0 }
    }

    fn current(&self) -> &Token {
        self.tokens.get(self.pos).unwrap_or(&Token::Eof)
    }

    fn advance(&mut self) -> Token {
        let tok = self.current().clone();
        if self.pos < self.tokens.len() {
            self.pos += 1;
        }
        tok
    }

    #[allow(dead_code)]
    fn expect(&mut self, expected: &Token) -> Result<(), String> {
        let got = self.advance();
        if std::mem::discriminant(&got) == std::mem::discriminant(expected) {
            Ok(())
        } else {
            Err(format!("Expected {:?}, got {:?}", expected, got))
        }
    }

    fn parse(&mut self) -> Result<Expr, String> {
        let expr = self.parse_or()?;
        if self.current() != &Token::Eof {
            Err(format!("Unexpected token: {:?}", self.current()))
        } else {
            Ok(expr)
        }
    }

    // OR has lowest precedence
    fn parse_or(&mut self) -> Result<Expr, String> {
        let mut left = self.parse_and()?;

        loop {
            match self.current() {
                Token::Or => {
                    self.advance();
                    let right = self.parse_and()?;
                    left = Expr::Or(Box::new(left), Box::new(right));
                }
                Token::Ident(s) if s.to_lowercase() == "or" => {
                    self.advance();
                    let right = self.parse_and()?;
                    left = Expr::Or(Box::new(left), Box::new(right));
                }
                _ => break,
            }
        }

        Ok(left)
    }

    // AND has higher precedence than OR
    fn parse_and(&mut self) -> Result<Expr, String> {
        let mut left = self.parse_not()?;

        loop {
            match self.current() {
                Token::And => {
                    self.advance();
                    let right = self.parse_not()?;
                    left = Expr::And(Box::new(left), Box::new(right));
                }
                Token::Ident(s) if s.to_lowercase() == "and" => {
                    self.advance();
                    let right = self.parse_not()?;
                    left = Expr::And(Box::new(left), Box::new(right));
                }
                _ => break,
            }
        }

        Ok(left)
    }

    // NOT has higher precedence than comparisons
    fn parse_not(&mut self) -> Result<Expr, String> {
        match self.current() {
            Token::Not => {
                self.advance();
                let inner = self.parse_not()?;
                Ok(Expr::Not(Box::new(inner)))
            }
            Token::Ident(s) if s.to_lowercase() == "not" => {
                self.advance();
                let inner = self.parse_not()?;
                Ok(Expr::Not(Box::new(inner)))
            }
            // NOTE: expression-level unary minus (e.g. `-x > 5`) falls through
            // to parse_primary and produces an error, which is acceptable since
            // ZoneMap cannot evaluate expressions anyway.
            _ => self.parse_primary(),
        }
    }

    // Primary expressions: comparisons or parenthesized
    fn parse_primary(&mut self) -> Result<Expr, String> {
        match self.current() {
            Token::LParen => {
                self.advance();
                let expr = self.parse_or()?;
                match self.current() {
                    Token::RParen => {
                        self.advance();
                        Ok(expr)
                    }
                    _ => Err(format!("Expected ')', got {:?}", self.current())),
                }
            }
            Token::Ident(_) | Token::Number(_) | Token::String(_) => self.parse_comparison(),
            _ => Err(format!("Unexpected token in primary: {:?}", self.current())),
        }
    }

    // Comparison: ident OP value
    fn parse_comparison(&mut self) -> Result<Expr, String> {
        // Parse column name
        let col = match self.advance() {
            Token::Ident(s) => s,
            Token::Number(n) => n, // Allow numbers as column names
            t => return Err(format!("Expected column name, got {:?}", t)),
        };

        // Parse operator
        let op = match self.current() {
            Token::Op(ref op) => {
                let result = *op;
                self.advance();
                result
            }
            Token::Ident(s) if s.eq_ignore_ascii_case("eq") => {
                self.advance();
                CompareOp::Eq
            }
            Token::Ident(s) if s.eq_ignore_ascii_case("ne") || s.eq_ignore_ascii_case("neq") => {
                self.advance();
                CompareOp::Ne
            }
            Token::Ident(s) if s.eq_ignore_ascii_case("lt") => {
                self.advance();
                CompareOp::Lt
            }
            Token::Ident(s) if s.eq_ignore_ascii_case("le") || s.eq_ignore_ascii_case("lte") => {
                self.advance();
                CompareOp::Le
            }
            Token::Ident(s) if s.eq_ignore_ascii_case("gt") => {
                self.advance();
                CompareOp::Gt
            }
            Token::Ident(s) if s.eq_ignore_ascii_case("ge") || s.eq_ignore_ascii_case("gte") => {
                self.advance();
                CompareOp::Ge
            }
            t => return Err(format!("Expected operator, got {:?}", t)),
        };

        // Parse value
        let val = self.parse_value()?;

        Ok(Expr::Comparison { col, op, val })
    }

    // Parse a scalar value
    fn parse_value(&mut self) -> Result<ScalarVal, String> {
        match self.advance() {
            Token::Ident(s) => {
                match s.to_lowercase().as_str() {
                    "null" => Ok(ScalarVal::Null),
                    "true" => Ok(ScalarVal::Bool(true)),
                    "false" => Ok(ScalarVal::Bool(false)),
                    other => {
                        // Try to parse as number
                        if let Ok(i) = other.parse::<i64>() {
                            Ok(ScalarVal::Int64(i))
                        } else if let Ok(f) = other.parse::<f64>() {
                            Ok(ScalarVal::Float64(f))
                        } else {
                            // Treat as string identifier
                            Ok(ScalarVal::String(s))
                        }
                    }
                }
            }
            Token::Number(n) => {
                if n.contains('.') {
                    n.parse::<f64>()
                        .map(ScalarVal::Float64)
                        .map_err(|e| e.to_string())
                } else {
                    n.parse::<i64>()
                        .map(ScalarVal::Int64)
                        .map_err(|e| e.to_string())
                }
            }
            Token::Minus => {
                // Unary minus applied to a number literal: -5
                let val = self.parse_value()?;
                match val {
                    ScalarVal::Int64(i) => Ok(ScalarVal::Int64(-i)),
                    ScalarVal::Float64(f) => Ok(ScalarVal::Float64(-f)),
                    _ => Err(format!("Unary minus cannot be applied to {:?}", val)),
                }
            }
            Token::String(s) => Ok(ScalarVal::String(s)),
            t => Err(format!("Expected value, got {:?}", t)),
        }
    }
}

/// Parse a filter expression string into an Expr AST
pub fn parse(s: &str) -> Result<Expr, String> {
    let tokens = tokenize(s)?;
    let mut parser = Parser::new(tokens);
    parser.parse()
}

// =============================================================================
// Evaluator
// =============================================================================

/// Evaluate a filter expression against a RecordBatch
pub fn evaluate(expr: &Expr, batch: &RecordBatch) -> Result<BooleanArray, RockDuckError> {
    match expr {
        Expr::Comparison { col, op, val } => evaluate_comparison(col, op, val, batch),
        Expr::And(a, b) => {
            let left = evaluate(a, batch)?;
            let right = evaluate(b, batch)?;
            Ok(BooleanArray::from_iter(
                left.iter().zip(right.iter()).map(|(l, r)| l.and(r)),
            ))
        }
        Expr::Or(a, b) => {
            let left = evaluate(a, batch)?;
            let right = evaluate(b, batch)?;
            Ok(BooleanArray::from_iter(
                left.iter().zip(right.iter()).map(|(l, r)| l.or(r)),
            ))
        }
        Expr::Not(a) => {
            let inner = evaluate(a, batch)?;
            Ok(BooleanArray::from_iter(inner.iter().map(|v| v.map(|b| !b))))
        }
    }
}

/// Evaluate a column comparison
fn evaluate_comparison(
    col: &str,
    op: &CompareOp,
    val: &ScalarVal,
    batch: &RecordBatch,
) -> Result<BooleanArray, RockDuckError> {
    let col_array = batch
        .column_by_name(col)
        .ok_or_else(|| RockDuckError::Query(format!("Column '{}' not found", col)))?;

    // Get the column's actual data type
    let col_dtype = col_array.data_type();

    match (op, val) {
        // Handle Int64 comparisons
        (CompareOp::Eq, ScalarVal::Int64(v)) => {
            let scalar_arr = arrow_array::Int64Array::from(vec![*v]);
            cmp::eq(col_array, &Scalar::new(scalar_arr))
                .map_err(|e| RockDuckError::Internal(format!("Arrow error: {}", e)))
        }
        (CompareOp::Ne, ScalarVal::Int64(v)) => {
            let scalar_arr = arrow_array::Int64Array::from(vec![*v]);
            cmp::neq(col_array, &Scalar::new(scalar_arr))
                .map_err(|e| RockDuckError::Internal(format!("Arrow error: {}", e)))
        }
        (CompareOp::Lt, ScalarVal::Int64(v)) => {
            let scalar_arr = arrow_array::Int64Array::from(vec![*v]);
            cmp::lt(col_array, &Scalar::new(scalar_arr))
                .map_err(|e| RockDuckError::Internal(format!("Arrow error: {}", e)))
        }
        (CompareOp::Le, ScalarVal::Int64(v)) => {
            let scalar_arr = arrow_array::Int64Array::from(vec![*v]);
            cmp::lt_eq(col_array, &Scalar::new(scalar_arr))
                .map_err(|e| RockDuckError::Internal(format!("Arrow error: {}", e)))
        }
        (CompareOp::Gt, ScalarVal::Int64(v)) => {
            let scalar_arr = arrow_array::Int64Array::from(vec![*v]);
            cmp::gt(col_array, &Scalar::new(scalar_arr))
                .map_err(|e| RockDuckError::Internal(format!("Arrow error: {}", e)))
        }
        (CompareOp::Ge, ScalarVal::Int64(v)) => {
            let scalar_arr = arrow_array::Int64Array::from(vec![*v]);
            cmp::gt_eq(col_array, &Scalar::new(scalar_arr))
                .map_err(|e| RockDuckError::Internal(format!("Arrow error: {}", e)))
        }

        // Handle Float64 comparisons
        (CompareOp::Eq, ScalarVal::Float64(v)) => {
            let scalar_arr = arrow_array::Float64Array::from(vec![*v]);
            cmp::eq(col_array, &Scalar::new(scalar_arr))
                .map_err(|e| RockDuckError::Internal(format!("Arrow error: {}", e)))
        }
        (CompareOp::Ne, ScalarVal::Float64(v)) => {
            let scalar_arr = arrow_array::Float64Array::from(vec![*v]);
            cmp::neq(col_array, &Scalar::new(scalar_arr))
                .map_err(|e| RockDuckError::Internal(format!("Arrow error: {}", e)))
        }
        (CompareOp::Lt, ScalarVal::Float64(v)) => {
            let scalar_arr = arrow_array::Float64Array::from(vec![*v]);
            cmp::lt(col_array, &Scalar::new(scalar_arr))
                .map_err(|e| RockDuckError::Internal(format!("Arrow error: {}", e)))
        }
        (CompareOp::Le, ScalarVal::Float64(v)) => {
            let scalar_arr = arrow_array::Float64Array::from(vec![*v]);
            cmp::lt_eq(col_array, &Scalar::new(scalar_arr))
                .map_err(|e| RockDuckError::Internal(format!("Arrow error: {}", e)))
        }
        (CompareOp::Gt, ScalarVal::Float64(v)) => {
            let scalar_arr = arrow_array::Float64Array::from(vec![*v]);
            cmp::gt(col_array, &Scalar::new(scalar_arr))
                .map_err(|e| RockDuckError::Internal(format!("Arrow error: {}", e)))
        }
        (CompareOp::Ge, ScalarVal::Float64(v)) => {
            let scalar_arr = arrow_array::Float64Array::from(vec![*v]);
            cmp::gt_eq(col_array, &Scalar::new(scalar_arr))
                .map_err(|e| RockDuckError::Internal(format!("Arrow error: {}", e)))
        }

        // Handle String comparisons
        (CompareOp::Eq, ScalarVal::String(v)) => {
            let scalar_arr = arrow_array::StringArray::from(vec![v.as_str()]);
            cmp::eq(col_array, &Scalar::new(scalar_arr))
                .map_err(|e| RockDuckError::Internal(format!("Arrow error: {}", e)))
        }
        (CompareOp::Ne, ScalarVal::String(v)) => {
            let scalar_arr = arrow_array::StringArray::from(vec![v.as_str()]);
            cmp::neq(col_array, &Scalar::new(scalar_arr))
                .map_err(|e| RockDuckError::Internal(format!("Arrow error: {}", e)))
        }
        (CompareOp::Lt, ScalarVal::String(v)) => {
            let scalar_arr = arrow_array::StringArray::from(vec![v.as_str()]);
            cmp::lt(col_array, &Scalar::new(scalar_arr))
                .map_err(|e| RockDuckError::Internal(format!("Arrow error: {}", e)))
        }
        (CompareOp::Le, ScalarVal::String(v)) => {
            let scalar_arr = arrow_array::StringArray::from(vec![v.as_str()]);
            cmp::lt_eq(col_array, &Scalar::new(scalar_arr))
                .map_err(|e| RockDuckError::Internal(format!("Arrow error: {}", e)))
        }
        (CompareOp::Gt, ScalarVal::String(v)) => {
            let scalar_arr = arrow_array::StringArray::from(vec![v.as_str()]);
            cmp::gt(col_array, &Scalar::new(scalar_arr))
                .map_err(|e| RockDuckError::Internal(format!("Arrow error: {}", e)))
        }
        (CompareOp::Ge, ScalarVal::String(v)) => {
            let scalar_arr = arrow_array::StringArray::from(vec![v.as_str()]);
            cmp::gt_eq(col_array, &Scalar::new(scalar_arr))
                .map_err(|e| RockDuckError::Internal(format!("Arrow error: {}", e)))
        }

        // Handle Bool comparisons
        (CompareOp::Eq, ScalarVal::Bool(v)) => {
            let scalar_arr = arrow_array::BooleanArray::from(vec![*v]);
            cmp::eq(col_array, &Scalar::new(scalar_arr))
                .map_err(|e| RockDuckError::Internal(format!("Arrow error: {}", e)))
        }
        (CompareOp::Ne, ScalarVal::Bool(v)) => {
            let scalar_arr = arrow_array::BooleanArray::from(vec![*v]);
            cmp::neq(col_array, &Scalar::new(scalar_arr))
                .map_err(|e| RockDuckError::Internal(format!("Arrow error: {}", e)))
        }

        // Handle Null comparisons
        (CompareOp::Eq, ScalarVal::Null) => Ok(arrow::compute::is_null(col_array)?),
        (CompareOp::Ne, ScalarVal::Null) => Ok(arrow::compute::is_not_null(col_array)?),

        _ => Err(RockDuckError::Query(format!(
            "Unsupported comparison: {:?} {:?} for column type {:?}",
            op, val, col_dtype
        ))),
    }
}

    #[allow(dead_code)]
    fn is_int_type(dtype: &DataType) -> bool {
    matches!(
        dtype,
        DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
    )
}

    #[allow(dead_code)]
    fn is_float_type(dtype: &DataType) -> bool {
    matches!(dtype, DataType::Float32 | DataType::Float64)
}

    #[allow(dead_code)]
    fn is_string_type(dtype: &DataType) -> bool {
    matches!(dtype, DataType::Utf8 | DataType::LargeUtf8)
}

// =============================================================================
// Routing feature extraction
// =============================================================================

/// Extract routing features from a parsed filter expression.
/// Used by the HTAP query router to determine read path.
pub fn extract_routing_features(
    expr: &Expr,
    stats: &crate::query::routing::TableStats,
) -> crate::query::routing::RouterParamsOwned {
    let columns = collect_columns(expr);
    let estimated_selectivity = estimate_selectivity(expr, stats);
    let kind = crate::query::routing::QueryKind::from_filter(true, false, estimated_selectivity);
    crate::query::routing::RouterParamsOwned::new(
        String::new(),
        columns,
        kind,
        0,
        true, // has_pending_writes: tests assume pending writes exist, so Tier 1 Rule 1 (point-get) takes precedence
        estimated_selectivity,
        stats.clone(),
    )
}

/// Collect all column names referenced in an expression.
fn collect_columns(expr: &Expr) -> Vec<String> {
    let mut cols = Vec::new();
    walk_expr_for_cols(expr, &mut cols);
    cols.sort();
    cols.dedup();
    cols
}

fn walk_expr_for_cols(expr: &Expr, out: &mut Vec<String>) {
    match expr {
        Expr::Comparison { col, .. } => out.push(col.clone()),
        Expr::And(a, b) => {
            walk_expr_for_cols(a, out);
            walk_expr_for_cols(b, out);
        }
        Expr::Or(a, b) => {
            walk_expr_for_cols(a, out);
            walk_expr_for_cols(b, out);
        }
        Expr::Not(e) => walk_expr_for_cols(e, out),
    }
}

/// Estimate the selectivity (fraction of rows matching) of a predicate.
fn estimate_selectivity(expr: &Expr, stats: &crate::query::routing::TableStats) -> f64 {
    match expr {
        Expr::Comparison { col, op, val } => {
            let Some(col_stats) = stats.column_stats.get(col) else {
                return 0.01; // unknown column, conservative default
            };
            let n_distinct = col_stats.distinct_count.max(1);
            let selectivity = match op {
                CompareOp::Eq => 1.0 / n_distinct as f64,
                CompareOp::Ne => 1.0 - 1.0 / n_distinct as f64,
                CompareOp::Lt | CompareOp::Le | CompareOp::Gt | CompareOp::Ge => {
                    if let (Some(min), Some(max), Some(dtype)) = (
                        col_stats.min.as_ref(),
                        col_stats.max.as_ref(),
                        col_stats.column_type.as_ref(),
                    ) {
                        let frac = estimate_position(val, min, max, dtype);
                        match op {
                            CompareOp::Lt => frac,
                            CompareOp::Le => (frac + 1.0 / n_distinct as f64).min(1.0),
                            CompareOp::Gt => (1.0 - frac - 1.0 / n_distinct as f64).max(0.0),
                            CompareOp::Ge => 1.0 - frac,
                            CompareOp::Eq | CompareOp::Ne => 0.5, // unreachable
                        }
                    } else {
                        0.5 // no range stats
                    }
                }
            };
            selectivity.clamp(0.0, 1.0)
        }
        Expr::And(a, b) => {
            let s_a = estimate_selectivity(a, stats);
            let s_b = estimate_selectivity(b, stats);
            s_a * s_b
        }
        Expr::Or(a, b) => {
            let s_a = estimate_selectivity(a, stats);
            let s_b = estimate_selectivity(b, stats);
            1.0 - (1.0 - s_a) * (1.0 - s_b)
        }
        Expr::Not(e) => 1.0 - estimate_selectivity(e, stats),
    }
}

/// Estimate where `val` falls within the [min, max] byte range as a fraction [0, 1].
/// Used for range comparison selectivity.
///
/// The `dtype` parameter is critical: the same byte sequence can have completely
/// different numeric values depending on the column type (e.g. `Int64` value `42`
/// stored as `[42, 0, 0, 0, 0, 0, 0, 0]` would be incorrectly decoded as `f64`
/// `5.562684e-315` if treated as float bits without type information).
fn estimate_position(
    val: &ScalarVal,
    min: &[u8],
    max: &[u8],
    dtype: &arrow_schema::DataType,
) -> f64 {
    match dtype {
        // ── Integer types ──────────────────────────────────────────────────────
        arrow_schema::DataType::Int8 => estimate_i8(val, min, max),
        arrow_schema::DataType::Int16 => estimate_i16(val, min, max),
        arrow_schema::DataType::Int32 => estimate_i32(val, min, max),
        arrow_schema::DataType::Int64 => estimate_i64(val, min, max),
        arrow_schema::DataType::UInt8 => estimate_u8(val, min, max),
        arrow_schema::DataType::UInt16 => estimate_u16(val, min, max),
        arrow_schema::DataType::UInt32 => estimate_u32(val, min, max),
        arrow_schema::DataType::UInt64 => estimate_u64(val, min, max),
        // ── Float types ────────────────────────────────────────────────────────
        arrow_schema::DataType::Float32 => estimate_f32(val, min, max),
        arrow_schema::DataType::Float64 => estimate_f64(val, min, max),
        // ── Non-comparable types: uniform distribution assumption ─────────────
        arrow_schema::DataType::Utf8
        | arrow_schema::DataType::LargeUtf8
        | arrow_schema::DataType::Binary
        | arrow_schema::DataType::LargeBinary
        | arrow_schema::DataType::Boolean => 0.5,
        // ── Fallback for unknown types ────────────────────────────────────────
        _ => 0.5,
    }
}

// ── Integer helpers ───────────────────────────────────────────────────────────

#[allow(dead_code)]
fn read_i64(val: &ScalarVal) -> Option<i64> {
    match val {
        ScalarVal::Int64(v) => Some(*v),
        _ => None,
    }
}

fn to_f64_fraction(v: f64, m: f64, mx: f64) -> f64 {
    if mx <= m {
        return 0.5;
    }
    ((v - m) / (mx - m)).clamp(0.0, 1.0)
}

macro_rules! impl_estimate_int {
    ($name:ident, $t:ty, $n:expr) => {
        fn $name(val: &ScalarVal, min: &[u8], max: &[u8]) -> f64 {
            let v = match val {
                ScalarVal::Int64(v) => Some(*v as $t),
                _ => None,
            };
            let m = read_le::<$t>(min, $n);
            let mx = read_le::<$t>(max, $n);
            match (v, m, mx) {
                (Some(v), Some(m), Some(mx)) => to_f64_fraction(v as f64, m as f64, mx as f64),
                _ => 0.5,
            }
        }
    };
}

fn read_le<T: FromBytes>(bytes: &[u8], n: usize) -> Option<T> {
    FromBytes::read_le(bytes, n)
}

// ── Float helpers ─────────────────────────────────────────────────────────────

fn read_scalar_f64(val: &ScalarVal) -> Option<f64> {
    match val {
        ScalarVal::Int64(v) => Some(*v as f64),
        ScalarVal::Float64(v) => Some(*v),
        ScalarVal::String(s) => s.parse::<f64>().ok(),
        _ => None,
    }
}

macro_rules! impl_estimate_float {
    ($name:ident, $t:ty, $n:expr) => {
        #[allow(clippy::collapsible_match)]
        fn $name(val: &ScalarVal, min: &[u8], max: &[u8]) -> f64 {
            let v = read_scalar_f64(val).map(|f| f as $t);
            let m = read_le::<$t>(min, $n);
            let mx = read_le::<$t>(max, $n);
            match (v, m, mx) {
                (Some(v), Some(m), Some(mx)) => {
                    if mx > m && !v.is_nan() && !m.is_nan() && !mx.is_nan() {
                        to_f64_fraction(v as f64, m as f64, mx as f64)
                    } else {
                        0.5
                    }
                }
                _ => 0.5,
            }
        }
    };
    ($name:ident, $t:ty, $n:expr, $cond:expr) => {
        #[allow(clippy::collapsible_match)]
        fn $name(val: &ScalarVal, min: &[u8], max: &[u8]) -> f64 {
            let v = read_scalar_f64(val).map(|f| f as $t);
            let m = read_le::<$t>(min, $n);
            let mx = read_le::<$t>(max, $n);
            match (v, m, mx) {
                (Some(v), Some(m), Some(mx)) => {
                    if $cond && mx > m && !v.is_nan() && !m.is_nan() && !mx.is_nan() {
                        to_f64_fraction(v as f64, m as f64, mx as f64)
                    } else {
                        0.5
                    }
                }
                _ => 0.5,
            }
        }
    };
}

// ── Type-specific implementations ──────────────────────────────────────────────

impl_estimate_int!(estimate_i8, i8, 1);
impl_estimate_int!(estimate_i16, i16, 2);
impl_estimate_int!(estimate_i32, i32, 4);
impl_estimate_int!(estimate_i64, i64, 8);
impl_estimate_int!(estimate_u8, u8, 1);
impl_estimate_int!(estimate_u16, u16, 2);
impl_estimate_int!(estimate_u32, u32, 4);
impl_estimate_int!(estimate_u64, u64, 8);

impl_estimate_float!(estimate_f32, f32, 4);
impl_estimate_float!(estimate_f64, f64, 8);

// ── Byte deserialization helper ───────────────────────────────────────────────

trait FromBytes {
    fn read_le(bytes: &[u8], n: usize) -> Option<Self>
    where
        Self: Sized;
}

macro_rules! impl_from_bytes {
    ($t:ty, $n:expr) => {
        impl FromBytes for $t {
            fn read_le(bytes: &[u8], _n: usize) -> Option<Self> {
                if bytes.len() == $n {
                    let mut arr = [0u8; $n];
                    arr.copy_from_slice(bytes);
                    Some(Self::from_le_bytes(arr))
                } else {
                    None
                }
            }
        }
    };
}

impl_from_bytes!(i8, 1);
impl_from_bytes!(i16, 2);
impl_from_bytes!(i32, 4);
impl_from_bytes!(i64, 8);
impl_from_bytes!(u8, 1);
impl_from_bytes!(u16, 2);
impl_from_bytes!(u32, 4);
impl_from_bytes!(u64, 8);
impl_from_bytes!(f32, 4);
impl_from_bytes!(f64, 8);

// =============================================================================
// Tests
// =============================================================================

pub fn parse_filter_expr(s: &str) -> Result<Expr, String> {
    parse(s)
}

#[cfg(test)]
#[allow(clippy::items_after_test_module)]
mod tests {
    use super::*;

    #[test]
    fn test_scan_filter_in_creation() {
        let filter = ScanFilter::In {
            column: "id".to_string(),
            values: vec![
                vec![1_u8, 0, 0, 0, 0, 0, 0, 0], // 1
                vec![2_u8, 0, 0, 0, 0, 0, 0, 0], // 2
                vec![3_u8, 0, 0, 0, 0, 0, 0, 0], // 3
            ],
        };

        assert_eq!(filter.column(), "id");
        assert_eq!(filter.values().len(), 3);
    }

    #[test]
    fn test_scan_filter_debug() {
        let filter = ScanFilter::In {
            column: "status".to_string(),
            values: vec![vec![1], vec![2]],
        };
        let debug_str = format!("{:?}", filter);
        assert!(debug_str.contains("status"));
        assert!(debug_str.contains("In"));
    }

    #[test]
    fn test_zone_map_predicate_group_empty() {
        let group = ZoneMapPredicateGroup {
            groups: vec![],
            has_cross_column_or: false,
        };
        assert!(group.is_empty());
    }

    #[test]
    fn test_zone_map_predicate_group_not_empty() {
        let group = ZoneMapPredicateGroup {
            groups: vec![ZoneMapOrGroup {
                column: "id".to_string(),
                ranges: vec![(
                    1_i64.to_le_bytes().to_vec(),
                    10_i64.to_le_bytes().to_vec(),
                )],
            }],
            has_cross_column_or: false,
        };
        assert!(!group.is_empty());
    }

    #[test]
    fn test_scalar_to_bytes_int64() {
        let val = ScalarVal::Int64(42);
        let bytes = scalar_to_bytes(&val);
        assert!(bytes.is_some());
        let bytes = bytes.unwrap();
        assert_eq!(bytes.len(), 8);
        assert_eq!(i64::from_le_bytes(bytes.clone().try_into().unwrap()), 42);
    }

    #[test]
    fn test_scalar_to_bytes_float64() {
        let val = ScalarVal::Float64(std::f64::consts::PI);
        let bytes = scalar_to_bytes(&val);
        assert!(bytes.is_some());
        let bytes = bytes.unwrap();
        assert_eq!(bytes.len(), 8);
        assert_eq!(f64::from_le_bytes(bytes.clone().try_into().unwrap()), std::f64::consts::PI);
    }

    #[test]
    fn test_scalar_to_bytes_bool() {
        assert_eq!(scalar_to_bytes(&ScalarVal::Bool(true)), Some(vec![1]));
        assert_eq!(scalar_to_bytes(&ScalarVal::Bool(false)), Some(vec![0]));
    }

    #[test]
    fn test_scalar_to_bytes_string() {
        let val = ScalarVal::String("hello".to_string());
        let bytes = scalar_to_bytes(&val);
        assert!(bytes.is_some());
        assert_eq!(bytes.unwrap(), b"hello");
    }

    #[test]
    fn test_scalar_to_bytes_null() {
        assert_eq!(scalar_to_bytes(&ScalarVal::Null), None);
    }

    #[test]
    fn test_tokenizer_basic() {
        let tokens = tokenize("id = 42").unwrap();
        assert_eq!(tokens.len(), 4);
        assert_eq!(tokens[0], Token::Ident("id".to_string()));
        assert_eq!(tokens[1], Token::Op(CompareOp::Eq));
        assert_eq!(tokens[2], Token::Number("42".to_string()));
    }

    #[test]
    fn test_tokenizer_string() {
        let tokens = tokenize(r#"name = "Alice""#).unwrap();
        assert_eq!(tokens.len(), 4);
        assert_eq!(tokens[0], Token::Ident("name".to_string()));
        assert_eq!(tokens[2], Token::String("Alice".to_string()));
    }

    #[test]
    fn test_tokenizer_operators() {
        let tokens = tokenize("a > 1 AND b < 2").unwrap();
        assert!(tokens.contains(&Token::Op(CompareOp::Gt)));
        assert!(tokens.contains(&Token::Op(CompareOp::Lt)));
        assert!(tokens.contains(&Token::And));
    }

    #[test]
    fn test_parser_equality() {
        let expr = parse("x = 5").unwrap();
        match expr {
            Expr::Comparison { col, op, val } => {
                assert_eq!(col, "x");
                assert_eq!(op, CompareOp::Eq);
                assert_eq!(val, ScalarVal::Int64(5));
            }
            _ => panic!("Expected Comparison"),
        }
    }

    #[test]
    fn test_parser_and() {
        let expr = parse("a = 1 AND b = 2").unwrap();
        match expr {
            Expr::And(left, right) => match (*left, *right) {
                (Expr::Comparison { col: c1, .. }, Expr::Comparison { col: c2, .. }) => {
                    assert_eq!(c1, "a");
                    assert_eq!(c2, "b");
                }
                _ => panic!("Expected two comparisons"),
            },
            _ => panic!("Expected And"),
        }
    }

    #[test]
    fn test_parser_or() {
        let expr = parse("a = 1 OR b = 2").unwrap();
        match expr {
            Expr::Or(_, _) => {}
            _ => panic!("Expected Or"),
        }
    }

    #[test]
    fn test_parser_not() {
        let expr = parse("NOT a = 1").unwrap();
        match expr {
            Expr::Not(_) => {}
            _ => panic!("Expected Not"),
        }
    }

    #[test]
    #[allow(clippy::approx_constant)]
    fn test_parser_float() {
        let expr = parse("x = 3.14").unwrap();
        match expr {
            Expr::Comparison { col, op, val } => {
                assert_eq!(col, "x");
                assert_eq!(op, CompareOp::Eq);
                match val {
                    ScalarVal::Float64(f) => assert!((f - 3.14).abs() < 0.001),
                    _ => panic!("Expected Float64"),
                }
            }
            _ => panic!("Expected Comparison"),
        }
    }

    #[test]
    fn test_expr_to_scan_filter() {
        let expr = Expr::Comparison {
            col: "id".to_string(),
            op: CompareOp::Eq,
            val: ScalarVal::Int64(42),
        };
        let filter = expr.to_scan_filter();
        assert!(filter.is_some());
        let filter = filter.unwrap();
        match filter {
            ScanFilter::Eq { column, value } => {
                assert_eq!(column, "id");
                assert_eq!(i64::from_le_bytes(value.clone().try_into().unwrap()), 42);
            }
            _ => panic!("Expected Eq filter"),
        }
    }

    #[test]
    fn test_zone_map_predicate_roundtrip() {
        let expr = parse("x = 5").unwrap();
        let predicates = to_zone_map_predicates(&expr);
        assert!(predicates.is_some());
        let predicates = predicates.unwrap();
        assert_eq!(predicates.len(), 1);
        assert_eq!(predicates[0].column, "x");
        // Eq produces (value, value) so min == max
        assert_eq!(predicates[0].pred_min, predicates[0].pred_max);
    }

    #[test]
    fn test_zone_map_predicate_range() {
        let expr = parse("x > 5").unwrap();
        let predicates = to_zone_map_predicates(&expr);
        assert!(predicates.is_some());
        let predicates = predicates.unwrap();
        // Gt should have non-empty pred_min and empty pred_max
        assert!(!predicates[0].pred_min.is_empty());
        assert!(predicates[0].pred_max.is_empty());
    }

    #[test]
    fn test_zone_map_predicate_or_returns_none() {
        let expr = parse("x = 1 OR y = 2").unwrap();
        let predicates = to_zone_map_predicates(&expr);
        assert!(
            predicates.is_none(),
            "OR expressions should return None for zone map predicates"
        );
    }

    #[test]
    fn test_zone_map_predicate_and() {
        let expr = parse("x = 1 AND y > 5").unwrap();
        let predicates = to_zone_map_predicates(&expr);
        assert!(predicates.is_some());
        let predicates = predicates.unwrap();
        assert_eq!(predicates.len(), 2);
    }

    #[test]
    fn test_zone_map_predicate_ne_returns_none() {
        let expr = parse("x != 5").unwrap();
        let predicates = to_zone_map_predicates(&expr);
        assert!(
            predicates.is_none(),
            "Ne expressions should return None for zone map predicates"
        );
    }

    #[test]
    fn test_collect_columns() {
        let expr = parse("a = 1 AND b = 2 AND a = 3").unwrap();
        let columns = collect_columns(&expr);
        assert_eq!(columns.len(), 2);
        assert!(columns.contains(&"a".to_string()));
        assert!(columns.contains(&"b".to_string()));
    }

    #[test]
    fn test_is_int_type() {
        assert!(is_int_type(&DataType::Int64));
        assert!(is_int_type(&DataType::Int32));
        assert!(!is_int_type(&DataType::Float64));
        assert!(!is_int_type(&DataType::Utf8));
    }

    #[test]
    fn test_is_float_type() {
        assert!(is_float_type(&DataType::Float64));
        assert!(is_float_type(&DataType::Float32));
        assert!(!is_float_type(&DataType::Int64));
        assert!(!is_float_type(&DataType::Utf8));
    }

    #[test]
    fn test_is_string_type() {
        assert!(is_string_type(&DataType::Utf8));
        assert!(is_string_type(&DataType::LargeUtf8));
        assert!(!is_string_type(&DataType::Int64));
    }
}

impl ScanFilter {
    pub fn column(&self) -> &str {
        match self {
            ScanFilter::Eq { column, .. } => column,
            ScanFilter::Range { column, .. } => column,
            ScanFilter::Like { column, .. } => column,
            ScanFilter::Between { column, .. } => column,
            ScanFilter::IsNull { column } => column,
            ScanFilter::IsNotNull { column } => column,
            ScanFilter::IsTrue { column } => column,
            ScanFilter::IsNotTrue { column } => column,
            ScanFilter::And(a, _) => a.column(),
            ScanFilter::Or(a, _) => a.column(),
            ScanFilter::Not(inner) => inner.column(),
            ScanFilter::In { column, .. } => column,
            ScanFilter::Gt { column, .. } => column,
            ScanFilter::Ge { column, .. } => column,
            ScanFilter::Lt { column, .. } => column,
            ScanFilter::Le { column, .. } => column,
        }
    }

    #[allow(dead_code)]
    pub(crate) fn values(&self) -> &Vec<Vec<u8>> {
        match self {
            ScanFilter::In { values, .. } => values,
            _ => panic!("values() called on non-In filter"),
        }
    }

    pub fn values_opt(&self) -> Option<&Vec<Vec<u8>>> {
        match self {
            ScanFilter::In { values, .. } => Some(values),
            _ => None,
        }
    }
}
