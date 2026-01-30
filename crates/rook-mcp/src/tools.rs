//! MCP tool input/output type definitions.
//!
//! These types are used with `schemars::JsonSchema` to generate the JSON Schema
//! that MCP clients use to understand tool parameters.

use rmcp::schemars::{self, JsonSchema};
use serde::{Deserialize, Serialize};

/// Input for memory_add tool.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct AddMemoryInput {
    /// The content to store as a memory.
    pub content: String,

    /// User ID for scoping the memory.
    /// Memories are isolated by user_id - searches only return memories for the same user.
    #[serde(default)]
    pub user_id: Option<String>,

    /// Optional metadata as a JSON object.
    /// Can include arbitrary key-value pairs for filtering and context.
    #[serde(default)]
    pub metadata: Option<serde_json::Value>,
}

/// Input for memory_search tool.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct SearchMemoryInput {
    /// The search query.
    /// The system uses semantic similarity to find the most relevant memories.
    pub query: String,

    /// User ID to scope search.
    /// Only returns memories belonging to this user.
    #[serde(default)]
    pub user_id: Option<String>,

    /// Maximum results to return.
    #[serde(default = "default_limit")]
    pub limit: usize,
}

fn default_limit() -> usize {
    10
}

/// Input for memory_get tool.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct GetMemoryInput {
    /// The memory ID to retrieve.
    pub id: String,
}

/// Input for memory_delete tool.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct DeleteMemoryInput {
    /// The memory ID to delete.
    pub id: String,
}

/// A single memory search result.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct MemorySearchResult {
    /// Unique memory identifier.
    pub id: String,

    /// The memory content text.
    pub memory: String,

    /// Similarity score (0.0 to 1.0, higher is more similar).
    pub score: f32,

    /// Optional metadata associated with the memory.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
}

/// Result of adding a memory.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct AddMemoryResult {
    /// The ID of the created memory.
    pub id: String,

    /// The stored memory content.
    pub memory: String,

    /// Event type (ADD, UPDATE, etc.).
    pub event: String,
}

/// Result of getting a memory.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct GetMemoryResult {
    /// Unique memory identifier.
    pub id: String,

    /// The memory content text.
    pub memory: String,

    /// Optional metadata associated with the memory.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,

    /// Creation timestamp.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub created_at: Option<String>,
}

/// Result of deleting a memory.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct DeleteMemoryResult {
    /// The ID of the deleted memory.
    pub deleted: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_memory_input_schema() {
        let schema = rmcp::schemars::schema_for!(AddMemoryInput);
        let json = serde_json::to_string_pretty(&schema).unwrap();
        assert!(json.contains("content"));
        assert!(json.contains("user_id"));
        assert!(json.contains("metadata"));
    }

    #[test]
    fn test_search_memory_input_default_limit() {
        let input: SearchMemoryInput = serde_json::from_str(r#"{"query": "test"}"#).unwrap();
        assert_eq!(input.limit, 10);
    }

    #[test]
    fn test_search_memory_input_custom_limit() {
        let input: SearchMemoryInput =
            serde_json::from_str(r#"{"query": "test", "limit": 5}"#).unwrap();
        assert_eq!(input.limit, 5);
    }
}
