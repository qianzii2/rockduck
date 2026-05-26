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

use crate::segment::meta::CompareOp;
use crate::RockDuckError;
use arrow_array::{RecordBatch, BooleanArray, Scalar};
use arrow::compute::kernels::cmp;

/// Filter expression AST
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// Column comparison: col OP value
    Comparison { col: String, op: CompareOp, val: ScalarVal },
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
                while let Some(&(_, c)) = chars.peek() {
                    if c.is_ascii_alphanumeric() || c == '_' {
                        chars.next();
                    } else {
                        break;
                    }
                }
                let end = chars.peek().map(|x| x.0).unwrap_or(s.len());
                let original = s[start..end].to_string();
                let lower = original.to_lowercase();
                match lower.as_str() {
                    "and" => tokens.push(Token::And),
                    "or" => tokens.push(Token::Or),
                    "not" => tokens.push(Token::Not),
                    "true" => tokens.push(Token::Ident(original)),
                    "false" => tokens.push(Token::Ident(original)),
                    "null" => tokens.push(Token::Ident(original)),
                    _ => tokens.push(Token::Ident(original)),
                }
            }
            _ if c.is_ascii_digit() || c == '-' => {
                let start = chars.next().unwrap().0;
                let mut has_dot = false;
                while let Some(&(_, c)) = chars.peek() {
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
            Token::Ident(_) | Token::Number(_) | Token::String(_) => {
                self.parse_comparison_or_bare_ident()
            }
            _ => Err(format!("Unexpected token in primary: {:?}", self.current())),
        }
    }

    // Comparison: ident OP value, OR bare identifier (treated as ident = true)
    fn parse_comparison_or_bare_ident(&mut self) -> Result<Expr, String> {
        // Parse column name
        let col = match self.advance() {
            Token::Ident(s) => s,
            Token::Number(n) => n, // Allow numbers as column names
            t => return Err(format!("Expected column name, got {:?}", t)),
        };

        // Check if there's an operator next
        if matches!(self.current(), Token::Op(_))
            || matches!(self.current(), Token::Ident(_))
        {
            // Parse operator - DON'T consume yet, just inspect
            let op = match self.current().clone() {
                Token::Op(op) => {
                    self.advance();
                    op
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
                _ => return Err(format!("Expected operator, got {:?}", self.current())),
            };

            // Parse value
            let val = self.parse_value()?;
            Ok(Expr::Comparison { col, op, val })
        } else {
            // Bare identifier: treat as column = true
            Ok(Expr::Comparison {
                col,
                op: CompareOp::Eq,
                val: ScalarVal::Bool(true),
            })
        }
    }

    // Parse a scalar value
    fn parse_value(&mut self) -> Result<ScalarVal, String> {
        match self.advance() {
            Token::Ident(s) => {
                let lower = s.to_lowercase();
                match lower.as_str() {
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
                            // Treat as string identifier - preserve original case
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
                left.iter()
                    .zip(right.iter())
                    .map(|(l, r)| l.and(r)),
            ))
        }
        Expr::Or(a, b) => {
            let left = evaluate(a, batch)?;
            let right = evaluate(b, batch)?;
            Ok(BooleanArray::from_iter(
                left.iter()
                    .zip(right.iter())
                    .map(|(l, r)| l.or(r)),
            ))
        }
        Expr::Not(a) => {
            let inner = evaluate(a, batch)?;
            Ok(BooleanArray::from_iter(
                inner.iter().map(|v| v.map(|b| !b)),
            ))
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
            let scalar = Scalar::new(scalar_arr);
            cmp::eq(col_array, &scalar).map_err(RockDuckError::Arrow)
        }
        (CompareOp::Ne, ScalarVal::Int64(v)) => {
            let scalar_arr = arrow_array::Int64Array::from(vec![*v]);
            let scalar = Scalar::new(scalar_arr);
            cmp::neq(col_array, &scalar).map_err(RockDuckError::Arrow)
        }
        (CompareOp::Lt, ScalarVal::Int64(v)) => {
            let scalar_arr = arrow_array::Int64Array::from(vec![*v]);
            let scalar = Scalar::new(scalar_arr);
            cmp::lt(col_array, &scalar).map_err(RockDuckError::Arrow)
        }
        (CompareOp::Le, ScalarVal::Int64(v)) => {
            let scalar_arr = arrow_array::Int64Array::from(vec![*v]);
            let scalar = Scalar::new(scalar_arr);
            cmp::lt_eq(col_array, &scalar).map_err(RockDuckError::Arrow)
        }
        (CompareOp::Gt, ScalarVal::Int64(v)) => {
            let scalar_arr = arrow_array::Int64Array::from(vec![*v]);
            let scalar = Scalar::new(scalar_arr);
            cmp::gt(col_array, &scalar).map_err(RockDuckError::Arrow)
        }
        (CompareOp::Ge, ScalarVal::Int64(v)) => {
            let scalar_arr = arrow_array::Int64Array::from(vec![*v]);
            let scalar = Scalar::new(scalar_arr);
            cmp::gt_eq(col_array, &scalar).map_err(RockDuckError::Arrow)
        }

        // Handle Float64 comparisons
        (CompareOp::Eq, ScalarVal::Float64(v)) => {
            let scalar_arr = arrow_array::Float64Array::from(vec![*v]);
            let scalar = Scalar::new(scalar_arr);
            cmp::eq(col_array, &scalar).map_err(RockDuckError::Arrow)
        }
        (CompareOp::Ne, ScalarVal::Float64(v)) => {
            let scalar_arr = arrow_array::Float64Array::from(vec![*v]);
            let scalar = Scalar::new(scalar_arr);
            cmp::neq(col_array, &scalar).map_err(RockDuckError::Arrow)
        }
        (CompareOp::Lt, ScalarVal::Float64(v)) => {
            let scalar_arr = arrow_array::Float64Array::from(vec![*v]);
            let scalar = Scalar::new(scalar_arr);
            cmp::lt(col_array, &scalar).map_err(RockDuckError::Arrow)
        }
        (CompareOp::Le, ScalarVal::Float64(v)) => {
            let scalar_arr = arrow_array::Float64Array::from(vec![*v]);
            let scalar = Scalar::new(scalar_arr);
            cmp::lt_eq(col_array, &scalar).map_err(RockDuckError::Arrow)
        }
        (CompareOp::Gt, ScalarVal::Float64(v)) => {
            let scalar_arr = arrow_array::Float64Array::from(vec![*v]);
            let scalar = Scalar::new(scalar_arr);
            cmp::gt(col_array, &scalar).map_err(RockDuckError::Arrow)
        }
        (CompareOp::Ge, ScalarVal::Float64(v)) => {
            let scalar_arr = arrow_array::Float64Array::from(vec![*v]);
            let scalar = Scalar::new(scalar_arr);
            cmp::gt_eq(col_array, &scalar).map_err(RockDuckError::Arrow)
        }

        // Handle String comparisons
        (CompareOp::Eq, ScalarVal::String(v)) => {
            let scalar_arr = arrow_array::StringArray::from(vec![v.as_str()]);
            let scalar = Scalar::new(scalar_arr);
            cmp::eq(col_array, &scalar).map_err(RockDuckError::Arrow)
        }
        (CompareOp::Ne, ScalarVal::String(v)) => {
            let scalar_arr = arrow_array::StringArray::from(vec![v.as_str()]);
            let scalar = Scalar::new(scalar_arr);
            cmp::neq(col_array, &scalar).map_err(RockDuckError::Arrow)
        }
        (CompareOp::Lt, ScalarVal::String(v)) => {
            let scalar_arr = arrow_array::StringArray::from(vec![v.as_str()]);
            let scalar = Scalar::new(scalar_arr);
            cmp::lt(col_array, &scalar).map_err(RockDuckError::Arrow)
        }
        (CompareOp::Le, ScalarVal::String(v)) => {
            let scalar_arr = arrow_array::StringArray::from(vec![v.as_str()]);
            let scalar = Scalar::new(scalar_arr);
            cmp::lt_eq(col_array, &scalar).map_err(RockDuckError::Arrow)
        }
        (CompareOp::Gt, ScalarVal::String(v)) => {
            let scalar_arr = arrow_array::StringArray::from(vec![v.as_str()]);
            let scalar = Scalar::new(scalar_arr);
            cmp::gt(col_array, &scalar).map_err(RockDuckError::Arrow)
        }
        (CompareOp::Ge, ScalarVal::String(v)) => {
            let scalar_arr = arrow_array::StringArray::from(vec![v.as_str()]);
            let scalar = Scalar::new(scalar_arr);
            cmp::gt_eq(col_array, &scalar).map_err(RockDuckError::Arrow)
        }

        // Handle Bool comparisons
        (CompareOp::Eq, ScalarVal::Bool(v)) => {
            let scalar_arr = arrow_array::BooleanArray::from(vec![*v]);
            let scalar = Scalar::new(scalar_arr);
            cmp::eq(col_array, &scalar).map_err(RockDuckError::Arrow)
        }
        (CompareOp::Ne, ScalarVal::Bool(v)) => {
            let scalar_arr = arrow_array::BooleanArray::from(vec![*v]);
            let scalar = Scalar::new(scalar_arr);
            cmp::neq(col_array, &scalar).map_err(RockDuckError::Arrow)
        }

        // Handle Null comparisons
        (CompareOp::Eq, ScalarVal::Null) => {
            Ok(arrow::compute::is_null(col_array)?)
        }
        (CompareOp::Ne, ScalarVal::Null) => {
            Ok(arrow::compute::is_not_null(col_array)?)
        }

        _ => Err(RockDuckError::Query(format!(
            "Unsupported comparison: {:?} {:?} for column type {:?}",
            op, val, col_dtype
        ))),
    }
}


// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ============ Tokenizer tests ============

    #[test]
    fn test_tokenize_simple_comparison() {
        let tokens = tokenize("age > 30").unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::Ident("age".to_string()),
                Token::Op(CompareOp::Gt),
                Token::Number("30".to_string()),
                Token::Eof
            ]
        );
    }

    #[test]
    fn test_tokenize_equality() {
        let tokens = tokenize("name = \"Alice\"").unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::Ident("name".to_string()),
                Token::Op(CompareOp::Eq),
                Token::String("Alice".to_string()),
                Token::Eof
            ]
        );
    }

    #[test]
    fn test_tokenize_and_or() {
        let tokens = tokenize("age > 30 AND status = 1").unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::Ident("age".to_string()),
                Token::Op(CompareOp::Gt),
                Token::Number("30".to_string()),
                Token::And,
                Token::Ident("status".to_string()),
                Token::Op(CompareOp::Eq),
                Token::Number("1".to_string()),
                Token::Eof
            ]
        );
    }

    #[test]
    fn test_tokenize_double_ampersand() {
        let tokens = tokenize("a > 1 && b < 2").unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::Ident("a".to_string()),
                Token::Op(CompareOp::Gt),
                Token::Number("1".to_string()),
                Token::And,
                Token::Ident("b".to_string()),
                Token::Op(CompareOp::Lt),
                Token::Number("2".to_string()),
                Token::Eof
            ]
        );
    }

    #[test]
    fn test_tokenize_double_bar() {
        let tokens = tokenize("a = 1 || b = 2").unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::Ident("a".to_string()),
                Token::Op(CompareOp::Eq),
                Token::Number("1".to_string()),
                Token::Or,
                Token::Ident("b".to_string()),
                Token::Op(CompareOp::Eq),
                Token::Number("2".to_string()),
                Token::Eof
            ]
        );
    }

    #[test]
    fn test_tokenize_not() {
        let tokens = tokenize("NOT active").unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::Not,
                Token::Ident("active".to_string()),
                Token::Eof
            ]
        );
    }

    #[test]
    fn test_tokenize_bang_not() {
        let tokens = tokenize("!active").unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::Not,
                Token::Ident("active".to_string()),
                Token::Eof
            ]
        );
    }

    #[test]
    fn test_tokenize_float() {
        let tokens = tokenize("price > 3.14").unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::Ident("price".to_string()),
                Token::Op(CompareOp::Gt),
                Token::Number("3.14".to_string()),
                Token::Eof
            ]
        );
    }

    #[test]
    fn test_tokenize_negative_number() {
        let tokens = tokenize("temp > -5").unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::Ident("temp".to_string()),
                Token::Op(CompareOp::Gt),
                Token::Number("-5".to_string()),
                Token::Eof
            ]
        );
    }

    #[test]
    fn test_tokenize_grouping() {
        let tokens = tokenize("( a > 1 )").unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::LParen,
                Token::Ident("a".to_string()),
                Token::Op(CompareOp::Gt),
                Token::Number("1".to_string()),
                Token::RParen,
                Token::Eof
            ]
        );
    }

    #[test]
    fn test_tokenize_bool_values() {
        let tokens = tokenize("active = true AND verified = false").unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::Ident("active".to_string()),
                Token::Op(CompareOp::Eq),
                Token::Ident("true".to_string()),
                Token::And,
                Token::Ident("verified".to_string()),
                Token::Op(CompareOp::Eq),
                Token::Ident("false".to_string()),
                Token::Eof
            ]
        );
    }

    #[test]
    fn test_tokenize_null() {
        let tokens = tokenize("col = null").unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::Ident("col".to_string()),
                Token::Op(CompareOp::Eq),
                Token::Ident("null".to_string()),
                Token::Eof
            ]
        );
    }

    #[test]
    fn test_tokenize_escape_sequences() {
        let tokens = tokenize("msg = \"hello\\nworld\"").unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::Ident("msg".to_string()),
                Token::Op(CompareOp::Eq),
                Token::String("hello\nworld".to_string()),
                Token::Eof
            ]
        );
    }

    // ============ Parser tests ============

    #[test]
    fn test_parse_simple_comparison() {
        let expr = parse("age > 30").unwrap();
        assert_eq!(
            expr,
            Expr::Comparison {
                col: "age".to_string(),
                op: CompareOp::Gt,
                val: ScalarVal::Int64(30)
            }
        );
    }

    #[test]
    fn test_parse_equality() {
        let expr = parse("name = \"Alice\"").unwrap();
        assert_eq!(
            expr,
            Expr::Comparison {
                col: "name".to_string(),
                op: CompareOp::Eq,
                val: ScalarVal::String("Alice".to_string())
            }
        );
    }

    #[test]
    fn test_parse_equality_double_eq() {
        let expr = parse("id == 100").unwrap();
        assert_eq!(
            expr,
            Expr::Comparison {
                col: "id".to_string(),
                op: CompareOp::Eq,
                val: ScalarVal::Int64(100)
            }
        );
    }

    #[test]
    fn test_parse_not_equal() {
        let expr = parse("status != 0").unwrap();
        assert_eq!(
            expr,
            Expr::Comparison {
                col: "status".to_string(),
                op: CompareOp::Ne,
                val: ScalarVal::Int64(0)
            }
        );
    }

    #[test]
    fn test_parse_all_operators() {
        assert!(matches!(
            parse("a > b").unwrap(),
            Expr::Comparison {
                op: CompareOp::Gt,
                ..
            }
        ));
        assert!(matches!(
            parse("a >= b").unwrap(),
            Expr::Comparison {
                op: CompareOp::Ge,
                ..
            }
        ));
        assert!(matches!(
            parse("a < b").unwrap(),
            Expr::Comparison {
                op: CompareOp::Lt,
                ..
            }
        ));
        assert!(matches!(
            parse("a <= b").unwrap(),
            Expr::Comparison {
                op: CompareOp::Le,
                ..
            }
        ));
    }

    #[test]
    fn test_parse_and() {
        let expr = parse("age > 30 AND status = 1").unwrap();
        assert_eq!(
            expr,
            Expr::And(
                Box::new(Expr::Comparison {
                    col: "age".to_string(),
                    op: CompareOp::Gt,
                    val: ScalarVal::Int64(30)
                }),
                Box::new(Expr::Comparison {
                    col: "status".to_string(),
                    op: CompareOp::Eq,
                    val: ScalarVal::Int64(1)
                })
            )
        );
    }

    #[test]
    fn test_parse_or() {
        let expr = parse("a = 1 OR b = 2").unwrap();
        assert_eq!(
            expr,
            Expr::Or(
                Box::new(Expr::Comparison {
                    col: "a".to_string(),
                    op: CompareOp::Eq,
                    val: ScalarVal::Int64(1)
                }),
                Box::new(Expr::Comparison {
                    col: "b".to_string(),
                    op: CompareOp::Eq,
                    val: ScalarVal::Int64(2)
                })
            )
        );
    }

    #[test]
    fn test_parse_double_ampersand() {
        let expr = parse("a > 1 && b < 2").unwrap();
        assert!(matches!(expr, Expr::And(_, _)));
    }

    #[test]
    fn test_parse_double_bar() {
        let expr = parse("a = 1 || b = 2").unwrap();
        assert!(matches!(expr, Expr::Or(_, _)));
    }

    #[test]
    fn test_parse_not() {
        let expr = parse("NOT active").unwrap();
        assert_eq!(
            expr,
            Expr::Not(Box::new(Expr::Comparison {
                col: "active".to_string(),
                op: CompareOp::Eq,
                val: ScalarVal::Bool(true)
            }))
        );
    }

    #[test]
    fn test_parse_bang_not() {
        let expr = parse("!active").unwrap();
        assert!(matches!(expr, Expr::Not(_)));
    }

    #[test]
    fn test_parse_complex_expression() {
        let expr = parse("a > 1 AND b < 2 OR c = 3").unwrap();
        // OR has lower precedence, so this is (a > 1 AND b < 2) OR c = 3
        assert!(matches!(expr, Expr::Or(_, _)));
    }

    #[test]
    fn test_parse_grouping() {
        let expr = parse("a > 1 AND (b < 2 OR c = 3)").unwrap();
        assert!(matches!(expr, Expr::And(_, _)));
    }

    #[test]
    fn test_parse_float_value() {
        let expr = parse("price > 3.14").unwrap();
        assert_eq!(
            expr,
            Expr::Comparison {
                col: "price".to_string(),
                op: CompareOp::Gt,
                val: ScalarVal::Float64(3.14)
            }
        );
    }

    #[test]
    fn test_parse_bool_values() {
        let expr = parse("active = true").unwrap();
        assert_eq!(
            expr,
            Expr::Comparison {
                col: "active".to_string(),
                op: CompareOp::Eq,
                val: ScalarVal::Bool(true)
            }
        );

        let expr = parse("deleted = false").unwrap();
        assert_eq!(
            expr,
            Expr::Comparison {
                col: "deleted".to_string(),
                op: CompareOp::Eq,
                val: ScalarVal::Bool(false)
            }
        );
    }

    #[test]
    fn test_parse_null() {
        let expr = parse("col = null").unwrap();
        assert_eq!(
            expr,
            Expr::Comparison {
                col: "col".to_string(),
                op: CompareOp::Eq,
                val: ScalarVal::Null
            }
        );
    }

    #[test]
    fn test_parse_negation_of_comparison() {
        let expr = parse("NOT age > 30").unwrap();
        assert!(matches!(expr, Expr::Not(_)));
    }

    #[test]
    fn test_parse_unquoted_string() {
        // Unquoted strings are treated as identifiers, then parsed as strings
        let expr = parse("name = Alice").unwrap();
        assert_eq!(
            expr,
            Expr::Comparison {
                col: "name".to_string(),
                op: CompareOp::Eq,
                val: ScalarVal::String("Alice".to_string())
            }
        );
    }

    #[test]
    fn test_parse_multiple_not() {
        let expr = parse("NOT NOT active").unwrap();
        // NOT NOT active = NOT (NOT active)
        assert!(matches!(expr, Expr::Not(_)));
    }

    #[test]
    fn test_parse_unterminated_string() {
        let result = tokenize("name = \"Alice");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Unterminated"));
    }

    #[test]
    fn test_parse_invalid_operator() {
        let result = parse("age ?? 30");
        assert!(result.is_err());
    }

    // ============ ScalarVal tests ============

    #[test]
    fn test_scalar_val_eq() {
        assert_eq!(ScalarVal::Int64(42), ScalarVal::Int64(42));
        assert_ne!(ScalarVal::Int64(42), ScalarVal::Int64(43));
        assert_eq!(
            ScalarVal::String("hello".to_string()),
            ScalarVal::String("hello".to_string())
        );
        assert_ne!(
            ScalarVal::String("hello".to_string()),
            ScalarVal::String("world".to_string())
        );
        assert_eq!(ScalarVal::Bool(true), ScalarVal::Bool(true));
        assert_eq!(ScalarVal::Null, ScalarVal::Null);
    }

    // ============================================================
    // Boundary and error-path tests
    // ============================================================

    #[test]
    fn test_parse_empty_string() {
        let result = parse("");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_column_with_underscore_and_digits() {
        // Column names with underscores and digits are valid
        let result = parse("user_id_123 > 10");
        assert!(result.is_ok());
        let expr = result.unwrap();
        // Should parse as a comparison
        assert!(matches!(expr, Expr::Comparison { .. }));
    }

    #[test]
    fn test_parse_multiple_ands() {
        // a > 1 AND b > 2 AND c > 3
        let result = parse("age > 18 AND score > 50 AND active = 1");
        assert!(result.is_ok());
        let expr = result.unwrap();
        // Should be a left-deep AND tree
        assert!(matches!(&expr, Expr::And(..)));
    }

    #[test]
    fn test_parse_comparison_all_ops() {
        for (op_str, op) in [
            ("=", CompareOp::Eq),
            ("==", CompareOp::Eq),
            ("!=", CompareOp::Ne),
            (">", CompareOp::Gt),
            (">=", CompareOp::Ge),
            ("<", CompareOp::Lt),
            ("<=", CompareOp::Le),
        ] {
            let input = format!("col {} 42", op_str);
            let result = parse(&input);
            assert!(result.is_ok(), "op '{}' should parse successfully", op_str);
            assert!(
                matches!(&result.unwrap(), Expr::Comparison { op: o, .. } if *o == op),
                "op '{}' should parse to {:?}",
                op_str,
                op
            );
        }
    }

    #[test]
    fn test_parse_string_value() {
        let result = parse("name = \"Alice\"");
        assert!(result.is_ok());
        let expr = result.unwrap();
        assert!(matches!(&expr, Expr::Comparison { .. }));
    }
}
