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
            provider: GraphStoreProvider::Neo4j,
            url: "bolt://localhost:7687".to_string(),
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
    #[default]
    Neo4j,
    Memgraph,
    Neptune,
    Kuzu,
}
