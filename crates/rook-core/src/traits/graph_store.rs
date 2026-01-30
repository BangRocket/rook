//! Graph store trait and related types.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::RookResult;
use crate::types::{GraphRelation, Message};

/// Entity in a knowledge graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    /// Entity name.
    pub name: String,
    /// Entity type/label.
    pub entity_type: String,
    /// Additional properties.
    pub properties: serde_json::Value,
}

/// Relationship between entities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Relationship {
    /// Source entity name.
    pub source: String,
    /// Target entity name.
    pub target: String,
    /// Relationship type.
    pub relationship_type: String,
    /// Additional properties.
    pub properties: serde_json::Value,
}

/// Search filters for graph operations.
#[derive(Debug, Clone, Default)]
pub struct GraphFilters {
    /// Filter by user ID.
    pub user_id: Option<String>,
    /// Filter by agent ID.
    pub agent_id: Option<String>,
    /// Filter by run ID.
    pub run_id: Option<String>,
}

/// Core GraphStore trait - all graph store backends implement this.
#[async_trait]
pub trait GraphStore: Send + Sync {
    /// Add entities and relationships from messages.
    async fn add(
        &self,
        messages: &[Message],
        filters: &GraphFilters,
    ) -> RookResult<Vec<GraphRelation>>;

    /// Search for related entities.
    async fn search(
        &self,
        query: &str,
        filters: &GraphFilters,
        limit: usize,
    ) -> RookResult<Vec<GraphRelation>>;

    /// Delete all data for the given filters.
    async fn delete_all(&self, filters: &GraphFilters) -> RookResult<()>;

    /// Get all entities for the given filters.
    async fn get_all(&self, filters: &GraphFilters) -> RookResult<Vec<Entity>>;

    /// Add an entity directly to the graph store.
    ///
    /// This method is used for LLM-extracted entities. The entity extraction
    /// happens in the Memory layer, and the extracted entities are stored
    /// via this method.
    ///
    /// Returns the entity ID (implementation-specific).
    async fn add_entity(
        &self,
        name: &str,
        entity_type: &str,
        properties: &serde_json::Value,
        filters: &GraphFilters,
    ) -> RookResult<i64> {
        // Default implementation: no-op, returns 0
        // Implementations can override to provide actual storage
        let _ = (name, entity_type, properties, filters);
        Ok(0)
    }

    /// Add a relationship directly to the graph store.
    ///
    /// Creates a relationship between two entities by name.
    /// If the entities don't exist, implementations may create them.
    ///
    /// Returns the relationship ID (implementation-specific).
    async fn add_relationship(
        &self,
        source_name: &str,
        target_name: &str,
        relationship_type: &str,
        properties: &serde_json::Value,
        filters: &GraphFilters,
    ) -> RookResult<i64> {
        // Default implementation: no-op, returns 0
        let _ = (source_name, target_name, relationship_type, properties, filters);
        Ok(0)
    }

    /// Get entities with their embeddings for entity merging.
    ///
    /// This is used by the entity merger to find similar entities.
    /// Returns entities that match the filters, with their embedding vectors
    /// if available.
    async fn get_entities_for_merge(
        &self,
        filters: &GraphFilters,
    ) -> RookResult<Vec<EntityWithEmbedding>> {
        // Default implementation: return empty vec (no merge support)
        let _ = filters;
        Ok(vec![])
    }
}

/// Entity with embedding for merge operations.
#[derive(Debug, Clone)]
pub struct EntityWithEmbedding {
    /// Database ID.
    pub id: i64,
    /// Entity name.
    pub name: String,
    /// Entity type.
    pub entity_type: String,
    /// Embedding vector (if available).
    pub embedding: Option<Vec<f32>>,
}

/// Graph store configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphStoreConfig {
    /// Provider type.
    pub provider: GraphStoreProvider,
    /// Connection URL.
    pub url: String,
    /// Username for authentication.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub username: Option<String>,
    /// Password for authentication.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub password: Option<String>,
    /// Database name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub database: Option<String>,
}

impl Default for GraphStoreConfig {
    fn default() -> Self {
        Self {
            provider: GraphStoreProvider::Embedded,
            url: ":memory:".to_string(),
            username: None,
            password: None,
            database: None,
        }
    }
}

impl GraphStoreConfig {
    /// Create a new embedded graph store config with in-memory database.
    pub fn embedded_memory() -> Self {
        Self {
            provider: GraphStoreProvider::Embedded,
            url: ":memory:".to_string(),
            username: None,
            password: None,
            database: None,
        }
    }

    /// Create a new embedded graph store config with file-based database.
    pub fn embedded_file(path: impl Into<String>) -> Self {
        Self {
            provider: GraphStoreProvider::Embedded,
            url: path.into(),
            username: None,
            password: None,
            database: None,
        }
    }
}

/// Graph store provider type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum GraphStoreProvider {
    /// Embedded graph store using petgraph + SQLite (default, no external dependencies).
    #[default]
    Embedded,
    /// Neo4j graph database.
    Neo4j,
    /// Memgraph (Neo4j-compatible).
    Memgraph,
    /// AWS Neptune.
    Neptune,
}
