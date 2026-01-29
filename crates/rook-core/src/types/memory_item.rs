//! Memory item types.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A memory item stored in the system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryItem {
    /// Unique identifier for the memory.
    pub id: String,
    /// The memory content/text.
    pub memory: String,
    /// MD5 hash of the memory content.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hash: Option<String>,
    /// Similarity score (from search).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score: Option<f32>,
    /// Custom metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, serde_json::Value>>,
    /// Creation timestamp.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub created_at: Option<String>,
    /// Last update timestamp.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub updated_at: Option<String>,
}

impl MemoryItem {
    /// Create a new memory item.
    pub fn new(id: impl Into<String>, memory: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            memory: memory.into(),
            hash: None,
            score: None,
            metadata: None,
            created_at: None,
            updated_at: None,
        }
    }

    /// Set the hash.
    pub fn with_hash(mut self, hash: impl Into<String>) -> Self {
        self.hash = Some(hash.into());
        self
    }

    /// Set the score.
    pub fn with_score(mut self, score: f32) -> Self {
        self.score = Some(score);
        self
    }

    /// Set the metadata.
    pub fn with_metadata(mut self, metadata: HashMap<String, serde_json::Value>) -> Self {
        self.metadata = Some(metadata);
        self
    }
}

/// Event type for memory operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum MemoryEvent {
    Add,
    Update,
    Delete,
    None,
}

impl Default for MemoryEvent {
    fn default() -> Self {
        Self::None
    }
}

/// Result of a memory add operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryResult {
    /// Memory ID.
    pub id: String,
    /// Memory content.
    pub memory: String,
    /// Event type (ADD, UPDATE, DELETE, NONE).
    pub event: MemoryEvent,
    /// Previous memory content (for UPDATE).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous_memory: Option<String>,
}

/// Result of an add operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AddResult {
    /// Memory results.
    pub results: Vec<MemoryResult>,
    /// Graph relations (if graph store enabled).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub relations: Option<Vec<GraphRelation>>,
}

/// Result of a search operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// Memory items.
    pub results: Vec<MemoryItem>,
    /// Graph relations (if graph store enabled).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub relations: Option<Vec<GraphRelation>>,
}

/// A graph relation between entities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphRelation {
    /// Source entity.
    pub source: String,
    /// Relationship type.
    pub relationship: String,
    /// Target entity.
    pub target: String,
}

/// Memory type enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryType {
    SemanticMemory,
    EpisodicMemory,
    ProceduralMemory,
}
