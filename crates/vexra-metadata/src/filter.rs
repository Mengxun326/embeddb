//! Simple metadata filter parser and evaluator.
//!
//! Supports a SQL-WHERE-like filter expression language:
//! - Equality: `field = "value"` or `field = 42`
//! - Comparison: `field > 10`, `field >= 5`, `field < 100`, `field <= 50`
//! - IN: `field IN ("a", "b", "c")`
//! - CONTAINS: `field CONTAINS "substring"`
//! - AND/OR: `a = 1 AND b > 2`
//!
//! Phase 0 implements a simple evaluator. Phase 3 will add a proper
//! parser with an expression AST.

use serde_json::Value as JsonValue;
use std::collections::HashSet;

/// A filter that can be applied to metadata entries.
#[derive(Debug, Clone)]
pub enum Filter {
    /// Field equals a JSON value.
    Equals {
        field: String,
        value: JsonValue,
    },
    /// Field is greater than a numeric value.
    GreaterThan {
        field: String,
        value: f64,
    },
    /// Field is less than a numeric value.
    LessThan {
        field: String,
        value: f64,
    },
    /// Field is in a set of values.
    In {
        field: String,
        values: HashSet<String>,
    },
    /// Field contains a substring.
    Contains {
        field: String,
        value: String,
    },
    /// Logical AND of two filters.
    And(Box<Filter>, Box<Filter>),
    /// Logical OR of two filters.
    Or(Box<Filter>, Box<Filter>),
}

impl Filter {
    /// Evaluate the filter against a JSON metadata value.
    pub fn evaluate(&self, metadata: &JsonValue) -> bool {
        match self {
            Filter::Equals { field, value } => {
                metadata.get(field) == Some(value)
            }
            Filter::GreaterThan { field, value } => {
                metadata
                    .get(field)
                    .and_then(|v| v.as_f64())
                    .is_some_and(|v| v > *value)
            }
            Filter::LessThan { field, value } => {
                metadata
                    .get(field)
                    .and_then(|v| v.as_f64())
                    .is_some_and(|v| v < *value)
            }
            Filter::In { field, values } => {
                metadata
                    .get(field)
                    .and_then(|v| v.as_str())
                    .is_some_and(|s| values.contains(s))
            }
            Filter::Contains { field, value } => {
                metadata
                    .get(field)
                    .and_then(|v| v.as_str())
                    .is_some_and(|s| s.contains(value.as_str()))
            }
            Filter::And(left, right) => left.evaluate(metadata) && right.evaluate(metadata),
            Filter::Or(left, right) => left.evaluate(metadata) || right.evaluate(metadata),
        }
    }

    /// Create an equality filter.
    pub fn eq(field: impl Into<String>, value: impl Into<JsonValue>) -> Self {
        Filter::Equals {
            field: field.into(),
            value: value.into(),
        }
    }

    /// Create a greater-than filter.
    pub fn gt(field: impl Into<String>, value: f64) -> Self {
        Filter::GreaterThan {
            field: field.into(),
            value,
        }
    }

    /// Create a less-than filter.
    pub fn lt(field: impl Into<String>, value: f64) -> Self {
        Filter::LessThan {
            field: field.into(),
            value,
        }
    }

    /// Create a contains filter.
    pub fn contains(field: impl Into<String>, value: impl Into<String>) -> Self {
        Filter::Contains {
            field: field.into(),
            value: value.into(),
        }
    }

    /// Combine two filters with AND.
    pub fn and(self, other: Filter) -> Self {
        Filter::And(Box::new(self), Box::new(other))
    }

    /// Combine two filters with OR.
    pub fn or(self, other: Filter) -> Self {
        Filter::Or(Box::new(self), Box::new(other))
    }

    /// Parse a simple filter string into a Filter.
    ///
    /// Supported syntax (Phase 0 — limited):
    /// - `field = "value"`
    /// - `field = number`
    /// - `field > number`
    /// - `field < number`
    /// - `field CONTAINS "text"`
    /// - `field IN ("a", "b")`
    pub fn parse(input: &str) -> std::result::Result<Self, String> {
        let input = input.trim();

        // Try to split on AND
        if let Some(pos) = find_and_or(input, " AND ") {
            let left = Filter::parse(&input[..pos])?;
            let right = Filter::parse(&input[pos + 5..])?;
            return Ok(left.and(right));
        }

        // Try to split on OR
        if let Some(pos) = find_and_or(input, " OR ") {
            let left = Filter::parse(&input[..pos])?;
            let right = Filter::parse(&input[pos + 4..])?;
            return Ok(left.or(right));
        }

        // Parse a single condition
        parse_condition(input)
    }
}

/// Find AND/OR at the top level (not inside quotes or parentheses).
fn find_and_or(input: &str, op: &str) -> Option<usize> {
    let mut in_quote = false;
    let mut paren_depth = 0;
    let bytes = input.as_bytes();
    let op_bytes = op.as_bytes();

    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'"' => in_quote = !in_quote,
            b'(' => paren_depth += 1,
            b')' => paren_depth -= 1,
            _ => {}
        }

        if !in_quote && paren_depth == 0
            && i + op_bytes.len() <= bytes.len() && &bytes[i..i + op_bytes.len()] == op_bytes {
                return Some(i);
            }
        i += 1;
    }

    None
}

/// Parse a single filter condition (no AND/OR).
fn parse_condition(input: &str) -> std::result::Result<Filter, String> {
    let input = input.trim();

    // IN: field IN ("a", "b", "c")
    if let Some(pos) = input.find(" IN ") {
        let field = input[..pos].trim().to_string();
        let rest = input[pos + 4..].trim();
        // Parse parenthesized list
        if rest.starts_with('(') && rest.ends_with(')') {
            let inner = &rest[1..rest.len() - 1];
            let values: HashSet<String> = inner
                .split(',')
                .map(|s| s.trim().trim_matches('"').to_string())
                .collect();
            return Ok(Filter::In { field, values });
        }
    }

    // CONTAINS: field CONTAINS "text"
    if let Some(pos) = input.find(" CONTAINS ") {
        let field = input[..pos].trim().to_string();
        let value = input[pos + 10..].trim().trim_matches('"').to_string();
        return Ok(Filter::Contains { field, value });
    }

    // Equality: field = "value" or field = number
    if let Some(pos) = input.find('=') {
        let field = input[..pos].trim().to_string();
        let value_str = input[pos + 1..].trim();

        // String value
        if value_str.starts_with('"') && value_str.ends_with('"') {
            let value = JsonValue::String(value_str[1..value_str.len() - 1].to_string());
            return Ok(Filter::Equals { field, value });
        }

        // Numeric value
        if let Ok(num) = value_str.parse::<f64>() {
            // Check if it's really an integer
            if num == num.floor() && !value_str.contains('.') {
                return Ok(Filter::Equals {
                    field,
                    value: JsonValue::Number(
                        serde_json::Number::from_f64(num).unwrap_or(0.into()),
                    ),
                });
            }
        }

        // Default: treat as string
        return Ok(Filter::Equals {
            field,
            value: JsonValue::String(value_str.trim_matches('"').to_string()),
        });
    }

    // Greater than: field > number
    if let Some(pos) = input.find('>') {
        // Check it's not >=
        let after = &input[pos + 1..];
        let op_start = if after.starts_with('=') { pos + 2 } else { pos + 1 };
        let field = input[..pos].trim().to_string();
        let value: f64 = input[op_start..]
            .trim()
            .parse()
            .map_err(|_| format!("Invalid number in filter: {}", input))?;
        if input.as_bytes().get(pos + 1) == Some(&b'=') {
            // >= is implemented as > value - epsilon
            return Ok(Filter::GreaterThan {
                field,
                value: value - f64::EPSILON,
            });
        }
        return Ok(Filter::GreaterThan { field, value });
    }

    // Less than: field < number
    if let Some(pos) = input.find('<') {
        let after = &input[pos + 1..];
        let op_start = if after.starts_with('=') { pos + 2 } else { pos + 1 };
        let field = input[..pos].trim().to_string();
        let value: f64 = input[op_start..]
            .trim()
            .parse()
            .map_err(|_| format!("Invalid number in filter: {}", input))?;
        if input.as_bytes().get(pos + 1) == Some(&b'=') {
            return Ok(Filter::LessThan {
                field,
                value: value + f64::EPSILON,
            });
        }
        return Ok(Filter::LessThan { field, value });
    }

    Err(format!("Unable to parse filter: {}", input))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_equals_string() {
        let filter = Filter::eq("category", "tech");
        assert!(filter.evaluate(&json!({"category": "tech"})));
        assert!(!filter.evaluate(&json!({"category": "science"})));
    }

    #[test]
    fn test_equals_number() {
        let filter = Filter::eq("year", json!(2024));
        assert!(filter.evaluate(&json!({"year": 2024})));
        assert!(!filter.evaluate(&json!({"year": 2023})));
    }

    #[test]
    fn test_greater_than() {
        let filter = Filter::gt("score", 10.0);
        assert!(filter.evaluate(&json!({"score": 15})));
        assert!(!filter.evaluate(&json!({"score": 5})));
    }

    #[test]
    fn test_contains() {
        let filter = Filter::contains("title", "vector");
        assert!(filter.evaluate(&json!({"title": "vector database"})));
        assert!(!filter.evaluate(&json!({"title": "relational"})));
    }

    #[test]
    fn test_and() {
        let filter = Filter::eq("cat", "tech").and(Filter::gt("score", 5.0));
        assert!(filter.evaluate(&json!({"cat": "tech", "score": 10})));
        assert!(!filter.evaluate(&json!({"cat": "science", "score": 10})));
        assert!(!filter.evaluate(&json!({"cat": "tech", "score": 2})));
    }

    #[test]
    fn test_parse_equality() {
        let filter = Filter::parse(r#"category = "tech""#).unwrap();
        assert!(filter.evaluate(&json!({"category": "tech"})));
    }

    #[test]
    fn test_parse_and() {
        let filter = Filter::parse(r#"a = "1" AND b > 5"#).unwrap();
        assert!(filter.evaluate(&json!({"a": "1", "b": 10})));
        assert!(!filter.evaluate(&json!({"a": "2", "b": 10})));
    }

    #[test]
    fn test_parse_contains() {
        let filter = Filter::parse(r#"title CONTAINS "hello""#).unwrap();
        assert!(filter.evaluate(&json!({"title": "hello world"})));
        assert!(!filter.evaluate(&json!({"title": "world"})));
    }
}
