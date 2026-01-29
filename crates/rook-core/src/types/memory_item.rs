//! Memory item types.

use super::fsrs::{DualStrength, FsrsState};
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

    // FSRS-6 memory dynamics fields

    /// FSRS memory state (stability, difficulty, reps, lapses).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_state: Option<FsrsState>,
    /// Dual-strength model (storage and retrieval strength).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dual_strength: Option<DualStrength>,
    /// Whether this is a key/important memory (higher priority in retrieval).
    #[serde(default, skip_serializing_if = "is_false")]
    pub is_key: bool,
}

/// Helper function to skip serializing false booleans.
fn is_false(b: &bool) -> bool {
    !*b
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
            memory_state: None,
            dual_strength: None,
            is_key: false,
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

    /// Set the FSRS memory state.
    pub fn with_memory_state(mut self, state: FsrsState) -> Self {
        self.memory_state = Some(state);
        self
    }

    /// Set the dual-strength model state.
    pub fn with_dual_strength(mut self, strength: DualStrength) -> Self {
        self.dual_strength = Some(strength);
        self
    }

    /// Mark this memory as a key/important memory.
    pub fn with_is_key(mut self, is_key: bool) -> Self {
        self.is_key = is_key;
        self
    }

    /// Set created_at timestamp.
    pub fn with_created_at(mut self, created_at: impl Into<String>) -> Self {
        self.created_at = Some(created_at.into());
        self
    }

    /// Set updated_at timestamp.
    pub fn with_updated_at(mut self, updated_at: impl Into<String>) -> Self {
        self.updated_at = Some(updated_at.into());
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_item_with_fsrs_state() {
        let state = FsrsState::new();
        let item = MemoryItem::new("id1", "test memory")
            .with_memory_state(state.clone());

        assert!(item.memory_state.is_some());
        let mem_state = item.memory_state.unwrap();
        assert_eq!(mem_state.stability, 0.0);
        assert_eq!(mem_state.difficulty, 5.0);
    }

    #[test]
    fn test_memory_item_with_dual_strength() {
        let strength = DualStrength {
            storage_strength: 0.5,
            retrieval_strength: 0.8,
        };
        let item = MemoryItem::new("id1", "test memory")
            .with_dual_strength(strength);

        assert!(item.dual_strength.is_some());
        let ds = item.dual_strength.unwrap();
        assert_eq!(ds.storage_strength, 0.5);
        assert_eq!(ds.retrieval_strength, 0.8);
    }

    #[test]
    fn test_memory_item_is_key() {
        let item = MemoryItem::new("id1", "important memory")
            .with_is_key(true);

        assert!(item.is_key);
    }

    #[test]
    fn test_memory_item_serialization() {
        let item = MemoryItem::new("id1", "test memory")
            .with_is_key(true)
            .with_memory_state(FsrsState::new());

        let json = serde_json::to_string(&item).unwrap();

        // Verify JSON contains the new fields
        assert!(json.contains("memory_state"));
        assert!(json.contains("is_key"));

        // Verify roundtrip
        let deserialized: MemoryItem = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.id, "id1");
        assert!(deserialized.is_key);
        assert!(deserialized.memory_state.is_some());
    }

    #[test]
    fn test_memory_item_serialization_omits_none() {
        let item = MemoryItem::new("id1", "test memory");

        let json = serde_json::to_string(&item).unwrap();

        // None fields should be omitted
        assert!(!json.contains("memory_state"));
        assert!(!json.contains("dual_strength"));
        assert!(!json.contains("is_key")); // false is also omitted
    }

    #[test]
    fn test_memory_item_builder_chain() {
        let item = MemoryItem::new("id1", "memory content")
            .with_hash("abc123")
            .with_score(0.95)
            .with_created_at("2024-01-01T00:00:00Z")
            .with_updated_at("2024-01-02T00:00:00Z")
            .with_memory_state(FsrsState::new())
            .with_dual_strength(DualStrength::new())
            .with_is_key(true);

        assert_eq!(item.id, "id1");
        assert_eq!(item.memory, "memory content");
        assert_eq!(item.hash, Some("abc123".to_string()));
        assert_eq!(item.score, Some(0.95));
        assert_eq!(item.created_at, Some("2024-01-01T00:00:00Z".to_string()));
        assert_eq!(item.updated_at, Some("2024-01-02T00:00:00Z".to_string()));
        assert!(item.memory_state.is_some());
        assert!(item.dual_strength.is_some());
        assert!(item.is_key);
    }
}
