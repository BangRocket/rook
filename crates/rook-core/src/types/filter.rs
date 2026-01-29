//! Filter types for memory queries.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Filter operator for metadata queries.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FilterOperator {
    /// Equal to.
    Eq(serde_json::Value),
    /// Not equal to.
    Ne(serde_json::Value),
    /// Greater than.
    Gt(serde_json::Value),
    /// Greater than or equal to.
    Gte(serde_json::Value),
    /// Less than.
    Lt(serde_json::Value),
    /// Less than or equal to.
    Lte(serde_json::Value),
    /// In list.
    In(Vec<serde_json::Value>),
    /// Not in list.
    Nin(Vec<serde_json::Value>),
    /// Contains substring.
    Contains(String),
    /// Contains substring (case-insensitive).
    Icontains(String),
    /// Between range.
    Between {
        min: serde_json::Value,
        max: serde_json::Value,
    },
    /// Is null.
    IsNull,
    /// Is not null.
    IsNotNull,
    /// Exists.
    Exists,
    /// Does not exist.
    NotExists,
    /// Wildcard match (matches any value).
    Wildcard,
}

/// A single filter condition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterCondition {
    /// Field name to filter on.
    pub field: String,
    /// Operator to apply.
    pub operator: FilterOperator,
}

impl FilterCondition {
    /// Create an equality filter.
    pub fn eq(field: impl Into<String>, value: impl Into<serde_json::Value>) -> Self {
        Self {
            field: field.into(),
            operator: FilterOperator::Eq(value.into()),
        }
    }

    /// Create an inequality filter.
    pub fn ne(field: impl Into<String>, value: impl Into<serde_json::Value>) -> Self {
        Self {
            field: field.into(),
            operator: FilterOperator::Ne(value.into()),
        }
    }

    /// Create a greater than filter.
    pub fn gt(field: impl Into<String>, value: impl Into<serde_json::Value>) -> Self {
        Self {
            field: field.into(),
            operator: FilterOperator::Gt(value.into()),
        }
    }

    /// Create a greater than or equal filter.
    pub fn gte(field: impl Into<String>, value: impl Into<serde_json::Value>) -> Self {
        Self {
            field: field.into(),
            operator: FilterOperator::Gte(value.into()),
        }
    }

    /// Create a contains filter.
    pub fn contains(field: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            field: field.into(),
            operator: FilterOperator::Contains(value.into()),
        }
    }

    /// Create an in-list filter.
    pub fn in_list(field: impl Into<String>, values: Vec<serde_json::Value>) -> Self {
        Self {
            field: field.into(),
            operator: FilterOperator::In(values),
        }
    }

    /// Create a less than filter.
    pub fn lt(field: impl Into<String>, value: impl Into<serde_json::Value>) -> Self {
        Self {
            field: field.into(),
            operator: FilterOperator::Lt(value.into()),
        }
    }

    /// Create a less than or equal filter.
    pub fn lte(field: impl Into<String>, value: impl Into<serde_json::Value>) -> Self {
        Self {
            field: field.into(),
            operator: FilterOperator::Lte(value.into()),
        }
    }

    /// Create a case-insensitive contains filter.
    pub fn icontains(field: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            field: field.into(),
            operator: FilterOperator::Icontains(value.into()),
        }
    }

    /// Create an is null filter.
    pub fn is_null(field: impl Into<String>) -> Self {
        Self {
            field: field.into(),
            operator: FilterOperator::IsNull,
        }
    }

    /// Create an is not null filter.
    pub fn is_not_null(field: impl Into<String>) -> Self {
        Self {
            field: field.into(),
            operator: FilterOperator::IsNotNull,
        }
    }

    /// Create an exists filter.
    pub fn exists(field: impl Into<String>) -> Self {
        Self {
            field: field.into(),
            operator: FilterOperator::Exists,
        }
    }
}

/// Composite filter with AND/OR/NOT logic.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Filter {
    /// Single condition.
    Condition(FilterCondition),
    /// AND of multiple filters.
    And(Vec<Filter>),
    /// OR of multiple filters.
    Or(Vec<Filter>),
    /// NOT of a filter.
    Not(Box<Filter>),
}

impl Filter {
    /// Create an equality filter.
    pub fn eq(field: impl Into<String>, value: impl Into<serde_json::Value>) -> Self {
        Filter::Condition(FilterCondition::eq(field, value))
    }

    /// Create an AND filter.
    pub fn and(filters: Vec<Filter>) -> Self {
        Filter::And(filters)
    }

    /// Create an OR filter.
    pub fn or(filters: Vec<Filter>) -> Self {
        Filter::Or(filters)
    }

    /// Create a NOT filter.
    pub fn not(filter: Filter) -> Self {
        Filter::Not(Box::new(filter))
    }

    /// Create a range filter.
    pub fn between(
        field: impl Into<String>,
        min: impl Into<serde_json::Value>,
        max: impl Into<serde_json::Value>,
    ) -> Self {
        Filter::Condition(FilterCondition {
            field: field.into(),
            operator: FilterOperator::Between {
                min: min.into(),
                max: max.into(),
            },
        })
    }

    /// Create a contains filter.
    pub fn contains(field: impl Into<String>, value: impl Into<String>) -> Self {
        Filter::Condition(FilterCondition::contains(field, value))
    }

    /// Create an in-list filter.
    pub fn in_list(field: impl Into<String>, values: Vec<serde_json::Value>) -> Self {
        Filter::Condition(FilterCondition::in_list(field, values))
    }

    /// Create an inequality filter.
    pub fn ne(field: impl Into<String>, value: impl Into<serde_json::Value>) -> Self {
        Filter::Condition(FilterCondition::ne(field, value))
    }

    /// Create a greater than filter.
    pub fn gt(field: impl Into<String>, value: impl Into<serde_json::Value>) -> Self {
        Filter::Condition(FilterCondition::gt(field, value))
    }

    /// Create a greater than or equal filter.
    pub fn gte(field: impl Into<String>, value: impl Into<serde_json::Value>) -> Self {
        Filter::Condition(FilterCondition::gte(field, value))
    }

    /// Create a less than filter.
    pub fn lt(field: impl Into<String>, value: impl Into<serde_json::Value>) -> Self {
        Filter::Condition(FilterCondition::lt(field, value))
    }

    /// Create a less than or equal filter.
    pub fn lte(field: impl Into<String>, value: impl Into<serde_json::Value>) -> Self {
        Filter::Condition(FilterCondition::lte(field, value))
    }

    /// Create a case-insensitive contains filter.
    pub fn icontains(field: impl Into<String>, value: impl Into<String>) -> Self {
        Filter::Condition(FilterCondition::icontains(field, value))
    }

    /// Create an is null filter.
    pub fn is_null(field: impl Into<String>) -> Self {
        Filter::Condition(FilterCondition::is_null(field))
    }

    /// Create an is not null filter.
    pub fn is_not_null(field: impl Into<String>) -> Self {
        Filter::Condition(FilterCondition::is_not_null(field))
    }

    /// Create an exists filter.
    pub fn exists(field: impl Into<String>) -> Self {
        Filter::Condition(FilterCondition::exists(field))
    }
}

/// Trait for translating filters to backend-specific formats.
pub trait FilterTranslator {
    type Output;
    type Error;

    /// Translate a filter to the backend-specific format.
    fn translate(&self, filter: &Filter) -> Result<Self::Output, Self::Error>;
}

/// Check if a JSON value contains advanced filter operators.
pub fn has_advanced_operators(filter: &serde_json::Value) -> bool {
    if let Some(obj) = filter.as_object() {
        for (key, value) in obj {
            // Check for logical operators
            if matches!(key.as_str(), "AND" | "OR" | "NOT") {
                return true;
            }

            // Check for wildcard
            if value.as_str() == Some("*") {
                return true;
            }

            // Check for comparison operators
            if let Some(inner_obj) = value.as_object() {
                for op in inner_obj.keys() {
                    if matches!(
                        op.as_str(),
                        "eq" | "ne" | "gt" | "gte" | "lt" | "lte" | "in" | "nin" | "contains" | "icontains"
                    ) {
                        return true;
                    }
                }
            }
        }
    }
    false
}

/// Convert a simple key-value map to a Filter.
pub fn from_simple_filters(filters: &HashMap<String, serde_json::Value>) -> Filter {
    let conditions: Vec<Filter> = filters
        .iter()
        .map(|(k, v)| Filter::eq(k.clone(), v.clone()))
        .collect();

    if conditions.len() == 1 {
        conditions.into_iter().next().unwrap()
    } else {
        Filter::And(conditions)
    }
}
