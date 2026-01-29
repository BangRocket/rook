//! Embedded graph store using petgraph + SQLite hybrid architecture.
//!
//! This module provides a graph store implementation that:
//! - Uses SQLite for persistent storage
//! - Uses petgraph DiGraph for O(1) in-memory neighbor lookups
//! - Synchronizes between the two on startup and updates
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────┐
//! │          EmbeddedGraphStore             │
//! ├─────────────────────────────────────────┤
//! │  ┌─────────────┐    ┌────────────────┐  │
//! │  │   SQLite    │    │   petgraph     │  │
//! │  │ (persistent)│◄──►│  (in-memory)   │  │
//! │  │             │    │  DiGraph       │  │
//! │  └─────────────┘    └────────────────┘  │
//! └─────────────────────────────────────────┘
//! ```

pub mod petgraph_ops;
pub mod schema;
pub mod sync;

use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;

use async_trait::async_trait;
use petgraph::graph::DiGraph;
use petgraph::visit::EdgeRef;
use rusqlite::Connection;

use rook_core::error::{RookError, RookResult};
use rook_core::traits::{Entity, GraphFilters, GraphStore, GraphStoreConfig};
use rook_core::types::{GraphRelation, Message};

use petgraph_ops::{DbIdIndex, EntityNode, NameIndex, RelationshipEdge};

/// Embedded graph store using petgraph + SQLite.
///
/// Thread-safe via Mutex on the connection and graph.
pub struct EmbeddedGraphStore {
    /// SQLite connection (wrapped in Mutex for Send + Sync).
    conn: Mutex<Connection>,
    /// In-memory graph for fast traversal.
    graph: Mutex<DiGraph<EntityNode, RelationshipEdge>>,
    /// Index from database ID to node index.
    db_id_index: Mutex<DbIdIndex>,
    /// Index from name to node index.
    name_index: Mutex<NameIndex>,
}

impl EmbeddedGraphStore {
    /// Create a new embedded graph store with the given database path.
    pub fn new(db_path: impl AsRef<Path>) -> RookResult<Self> {
        let conn = Connection::open(db_path)?;
        schema::init_schema(&conn)?;

        let mut graph = DiGraph::new();
        let mut db_id_index = HashMap::new();
        let mut name_index = HashMap::new();

        // Load existing data
        sync::load_graph(&conn, &mut graph, &mut db_id_index, &mut name_index)?;

        Ok(Self {
            conn: Mutex::new(conn),
            graph: Mutex::new(graph),
            db_id_index: Mutex::new(db_id_index),
            name_index: Mutex::new(name_index),
        })
    }

    /// Create a new in-memory embedded graph store.
    pub fn in_memory() -> RookResult<Self> {
        let conn = Connection::open_in_memory()?;
        schema::init_schema(&conn)?;

        Ok(Self {
            conn: Mutex::new(conn),
            graph: Mutex::new(DiGraph::new()),
            db_id_index: Mutex::new(HashMap::new()),
            name_index: Mutex::new(HashMap::new()),
        })
    }

    /// Create from a GraphStoreConfig.
    pub async fn from_config(config: &GraphStoreConfig) -> RookResult<Self> {
        // URL is the database path for embedded store
        if config.url.is_empty() || config.url == ":memory:" {
            Self::in_memory()
        } else {
            Self::new(&config.url)
        }
    }

    /// Add an entity to the store.
    pub fn add_entity(
        &self,
        name: &str,
        entity_type: &str,
        properties: &serde_json::Value,
        filters: &GraphFilters,
    ) -> RookResult<i64> {
        let conn = self.conn.lock().map_err(|e| RookError::internal(e.to_string()))?;

        // Save to SQLite
        let db_id = sync::save_entity(&conn, name, entity_type, properties, filters)?;

        // Update in-memory graph
        let mut graph = self.graph.lock().map_err(|e| RookError::internal(e.to_string()))?;
        let mut db_id_index = self.db_id_index.lock().map_err(|e| RookError::internal(e.to_string()))?;
        let mut name_index = self.name_index.lock().map_err(|e| RookError::internal(e.to_string()))?;

        // Only add if not already present
        if !db_id_index.contains_key(&db_id) {
            let mut entity = EntityNode::new(db_id, name, entity_type);
            entity.properties = properties.clone();
            entity.user_id = filters.user_id.clone();
            entity.agent_id = filters.agent_id.clone();
            entity.run_id = filters.run_id.clone();

            let name_key = format!(
                "{}:{}:{}:{}",
                name,
                filters.user_id.as_deref().unwrap_or(""),
                filters.agent_id.as_deref().unwrap_or(""),
                filters.run_id.as_deref().unwrap_or("")
            );

            let idx = graph.add_node(entity);
            db_id_index.insert(db_id, idx);
            name_index.insert(name_key, idx);
        }

        Ok(db_id)
    }

    /// Add a relationship between entities.
    pub fn add_relationship(
        &self,
        source_name: &str,
        target_name: &str,
        relationship_type: &str,
        properties: &serde_json::Value,
        filters: &GraphFilters,
    ) -> RookResult<i64> {
        // First ensure both entities exist
        let source_id = self.get_or_create_entity(source_name, "entity", filters)?;
        let target_id = self.get_or_create_entity(target_name, "entity", filters)?;

        let conn = self.conn.lock().map_err(|e| RookError::internal(e.to_string()))?;

        // Save to SQLite
        let db_id = sync::save_relationship(&conn, source_id, target_id, relationship_type, properties, 1.0)?;

        // Update in-memory graph
        let mut graph = self.graph.lock().map_err(|e| RookError::internal(e.to_string()))?;
        let db_id_index = self.db_id_index.lock().map_err(|e| RookError::internal(e.to_string()))?;

        if let (Some(&source_idx), Some(&target_idx)) =
            (db_id_index.get(&source_id), db_id_index.get(&target_id))
        {
            let edge = RelationshipEdge::new(db_id, relationship_type)
                .with_properties(properties.clone());
            graph.add_edge(source_idx, target_idx, edge);
        }

        Ok(db_id)
    }

    /// Get or create an entity by name.
    fn get_or_create_entity(
        &self,
        name: &str,
        entity_type: &str,
        filters: &GraphFilters,
    ) -> RookResult<i64> {
        let name_key = format!(
            "{}:{}:{}:{}",
            name,
            filters.user_id.as_deref().unwrap_or(""),
            filters.agent_id.as_deref().unwrap_or(""),
            filters.run_id.as_deref().unwrap_or("")
        );

        // Check if already in memory
        let name_index = self.name_index.lock().map_err(|e| RookError::internal(e.to_string()))?;
        let db_id_index = self.db_id_index.lock().map_err(|e| RookError::internal(e.to_string()))?;

        if let Some(&idx) = name_index.get(&name_key) {
            let graph = self.graph.lock().map_err(|e| RookError::internal(e.to_string()))?;
            if let Some(entity) = graph.node_weight(idx) {
                return Ok(entity.db_id);
            }
        }
        drop(name_index);
        drop(db_id_index);

        // Create new entity
        self.add_entity(name, entity_type, &serde_json::json!({}), filters)
    }

    /// Get neighbors of an entity.
    pub fn get_neighbors(&self, entity_name: &str, filters: &GraphFilters) -> RookResult<Vec<GraphRelation>> {
        let name_key = format!(
            "{}:{}:{}:{}",
            entity_name,
            filters.user_id.as_deref().unwrap_or(""),
            filters.agent_id.as_deref().unwrap_or(""),
            filters.run_id.as_deref().unwrap_or("")
        );

        let name_index = self.name_index.lock().map_err(|e| RookError::internal(e.to_string()))?;
        let graph = self.graph.lock().map_err(|e| RookError::internal(e.to_string()))?;

        let node_idx = match name_index.get(&name_key) {
            Some(&idx) => idx,
            None => return Ok(vec![]),
        };

        let mut relations = Vec::new();

        // Get outgoing edges
        for edge in graph.edges(node_idx) {
            let target_node = graph.node_weight(edge.target());
            if let Some(target) = target_node {
                relations.push(GraphRelation {
                    source: entity_name.to_string(),
                    relationship: edge.weight().relationship_type.clone(),
                    target: target.name.clone(),
                });
            }
        }

        // Get incoming edges
        for edge in graph.edges_directed(node_idx, petgraph::Direction::Incoming) {
            let source_node = graph.node_weight(edge.source());
            if let Some(source) = source_node {
                relations.push(GraphRelation {
                    source: source.name.clone(),
                    relationship: edge.weight().relationship_type.clone(),
                    target: entity_name.to_string(),
                });
            }
        }

        Ok(relations)
    }

    /// Get entity count.
    pub fn entity_count(&self) -> RookResult<usize> {
        let graph = self.graph.lock().map_err(|e| RookError::internal(e.to_string()))?;
        Ok(graph.node_count())
    }

    /// Get relationship count.
    pub fn relationship_count(&self) -> RookResult<usize> {
        let graph = self.graph.lock().map_err(|e| RookError::internal(e.to_string()))?;
        Ok(graph.edge_count())
    }
}

#[async_trait]
impl GraphStore for EmbeddedGraphStore {
    /// Add entities and relationships from messages.
    ///
    /// For the embedded store, this is a simplified implementation.
    /// Full entity extraction would be done by an LLM in the memory layer.
    async fn add(
        &self,
        messages: &[Message],
        filters: &GraphFilters,
    ) -> RookResult<Vec<GraphRelation>> {
        let relations = Vec::new();

        // Simple implementation: extract entities from message content
        // In production, this would use an LLM for entity extraction
        for msg in messages {
            // For now, just store the message content as an entity
            let entity_type = match msg.role {
                rook_core::types::MessageRole::User => "user_message",
                rook_core::types::MessageRole::Assistant => "assistant_message",
                _ => "message",
            };

            // Create a hash-based ID for the entity
            let entity_name = format!("msg_{}", md5::compute(&msg.content).0[..8].iter()
                .map(|b| format!("{:02x}", b))
                .collect::<String>());

            self.add_entity(
                &entity_name,
                entity_type,
                &serde_json::json!({
                    "content": msg.content,
                    "role": format!("{:?}", msg.role)
                }),
                filters,
            )?;
        }

        Ok(relations)
    }

    /// Search for related entities.
    async fn search(
        &self,
        query: &str,
        filters: &GraphFilters,
        limit: usize,
    ) -> RookResult<Vec<GraphRelation>> {
        // Search for entities matching the query
        let graph = self.graph.lock().map_err(|e| RookError::internal(e.to_string()))?;
        let mut relations = Vec::new();

        let query_lower = query.to_lowercase();

        // Simple text match search
        for node_idx in graph.node_indices() {
            if relations.len() >= limit {
                break;
            }

            let node = match graph.node_weight(node_idx) {
                Some(n) => n,
                None => continue,
            };

            // Check filters
            if !node.matches_filters(
                filters.user_id.as_deref(),
                filters.agent_id.as_deref(),
                filters.run_id.as_deref(),
            ) {
                continue;
            }

            // Check if name contains query
            if node.name.to_lowercase().contains(&query_lower) {
                // Get all relationships for this entity
                for edge in graph.edges(node_idx) {
                    if relations.len() >= limit {
                        break;
                    }
                    if let Some(target) = graph.node_weight(edge.target()) {
                        relations.push(GraphRelation {
                            source: node.name.clone(),
                            relationship: edge.weight().relationship_type.clone(),
                            target: target.name.clone(),
                        });
                    }
                }
            }
        }

        Ok(relations)
    }

    /// Delete all data for the given filters.
    async fn delete_all(&self, filters: &GraphFilters) -> RookResult<()> {
        let conn = self.conn.lock().map_err(|e| RookError::internal(e.to_string()))?;

        // Get IDs to delete
        let ids = sync::get_entity_ids_by_filters(&conn, filters)?;

        // Delete from SQLite
        sync::delete_entities_by_filters(&conn, filters)?;

        // Remove from in-memory graph
        let mut graph = self.graph.lock().map_err(|e| RookError::internal(e.to_string()))?;
        let mut db_id_index = self.db_id_index.lock().map_err(|e| RookError::internal(e.to_string()))?;
        let mut name_index = self.name_index.lock().map_err(|e| RookError::internal(e.to_string()))?;

        for id in ids {
            if let Some(idx) = db_id_index.remove(&id) {
                // Get name key before removing
                if let Some(entity) = graph.node_weight(idx) {
                    let name_key = format!(
                        "{}:{}:{}:{}",
                        entity.name,
                        entity.user_id.as_deref().unwrap_or(""),
                        entity.agent_id.as_deref().unwrap_or(""),
                        entity.run_id.as_deref().unwrap_or("")
                    );
                    name_index.remove(&name_key);
                }
                graph.remove_node(idx);
            }
        }

        Ok(())
    }

    /// Get all entities for the given filters.
    async fn get_all(&self, filters: &GraphFilters) -> RookResult<Vec<Entity>> {
        let graph = self.graph.lock().map_err(|e| RookError::internal(e.to_string()))?;
        let mut entities = Vec::new();

        for node_idx in graph.node_indices() {
            let node = match graph.node_weight(node_idx) {
                Some(n) => n,
                None => continue,
            };

            if node.matches_filters(
                filters.user_id.as_deref(),
                filters.agent_id.as_deref(),
                filters.run_id.as_deref(),
            ) {
                entities.push(Entity {
                    name: node.name.clone(),
                    entity_type: node.entity_type.clone(),
                    properties: node.properties.clone(),
                });
            }
        }

        Ok(entities)
    }
}

// Implement Debug for EmbeddedGraphStore
impl std::fmt::Debug for EmbeddedGraphStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EmbeddedGraphStore")
            .field("entity_count", &self.entity_count().unwrap_or(0))
            .field("relationship_count", &self.relationship_count().unwrap_or(0))
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_embedded_store_crud() {
        let store = EmbeddedGraphStore::in_memory().unwrap();
        let filters = GraphFilters {
            user_id: Some("user1".to_string()),
            agent_id: None,
            run_id: None,
        };

        // Add entities
        store
            .add_entity("Alice", "person", &serde_json::json!({"age": 30}), &filters)
            .unwrap();
        store
            .add_entity("Bob", "person", &serde_json::json!({"age": 25}), &filters)
            .unwrap();

        assert_eq!(store.entity_count().unwrap(), 2);

        // Add relationship
        store
            .add_relationship("Alice", "Bob", "knows", &serde_json::json!({}), &filters)
            .unwrap();

        assert_eq!(store.relationship_count().unwrap(), 1);

        // Get neighbors
        let neighbors = store.get_neighbors("Alice", &filters).unwrap();
        assert_eq!(neighbors.len(), 1);
        assert_eq!(neighbors[0].source, "Alice");
        assert_eq!(neighbors[0].target, "Bob");
        assert_eq!(neighbors[0].relationship, "knows");
    }

    #[tokio::test]
    async fn test_embedded_store_get_all() {
        let store = EmbeddedGraphStore::in_memory().unwrap();
        let filters = GraphFilters {
            user_id: Some("user1".to_string()),
            agent_id: None,
            run_id: None,
        };

        store
            .add_entity("Alice", "person", &serde_json::json!({}), &filters)
            .unwrap();
        store
            .add_entity("Acme", "company", &serde_json::json!({}), &filters)
            .unwrap();

        let entities = store.get_all(&filters).await.unwrap();
        assert_eq!(entities.len(), 2);
    }

    #[tokio::test]
    async fn test_embedded_store_search() {
        let store = EmbeddedGraphStore::in_memory().unwrap();
        let filters = GraphFilters::default();

        store
            .add_entity("Alice Smith", "person", &serde_json::json!({}), &filters)
            .unwrap();
        store
            .add_entity("Bob Jones", "person", &serde_json::json!({}), &filters)
            .unwrap();
        store
            .add_relationship("Alice Smith", "Bob Jones", "knows", &serde_json::json!({}), &filters)
            .unwrap();

        // Search for Alice
        let results = store.search("Alice", &filters, 10).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].source, "Alice Smith");
    }

    #[tokio::test]
    async fn test_embedded_store_delete_all() {
        let store = EmbeddedGraphStore::in_memory().unwrap();
        let filters1 = GraphFilters {
            user_id: Some("user1".to_string()),
            agent_id: None,
            run_id: None,
        };
        let filters2 = GraphFilters {
            user_id: Some("user2".to_string()),
            agent_id: None,
            run_id: None,
        };

        store
            .add_entity("Alice", "person", &serde_json::json!({}), &filters1)
            .unwrap();
        store
            .add_entity("Bob", "person", &serde_json::json!({}), &filters2)
            .unwrap();

        assert_eq!(store.entity_count().unwrap(), 2);

        // Delete user1's data
        store.delete_all(&filters1).await.unwrap();

        // Only user2's data should remain
        assert_eq!(store.entity_count().unwrap(), 1);
        let entities = store.get_all(&filters2).await.unwrap();
        assert_eq!(entities.len(), 1);
        assert_eq!(entities[0].name, "Bob");
    }

    #[tokio::test]
    async fn test_embedded_store_persistence() {
        let temp_dir = tempfile::tempdir().unwrap();
        let db_path = temp_dir.path().join("test_graph.db");

        // Create store and add data
        {
            let store = EmbeddedGraphStore::new(&db_path).unwrap();
            let filters = GraphFilters::default();

            store
                .add_entity("Alice", "person", &serde_json::json!({"age": 30}), &filters)
                .unwrap();
            store
                .add_entity("Bob", "person", &serde_json::json!({}), &filters)
                .unwrap();
            store
                .add_relationship("Alice", "Bob", "knows", &serde_json::json!({}), &filters)
                .unwrap();
        }

        // Reopen store and verify data persisted
        {
            let store = EmbeddedGraphStore::new(&db_path).unwrap();
            let filters = GraphFilters::default();

            assert_eq!(store.entity_count().unwrap(), 2);
            assert_eq!(store.relationship_count().unwrap(), 1);

            let entities = store.get_all(&filters).await.unwrap();
            let names: Vec<_> = entities.iter().map(|e| e.name.as_str()).collect();
            assert!(names.contains(&"Alice"));
            assert!(names.contains(&"Bob"));
        }
    }

    #[tokio::test]
    async fn test_embedded_store_filter_isolation() {
        let store = EmbeddedGraphStore::in_memory().unwrap();

        let user1 = GraphFilters {
            user_id: Some("user1".to_string()),
            agent_id: None,
            run_id: None,
        };
        let user2 = GraphFilters {
            user_id: Some("user2".to_string()),
            agent_id: None,
            run_id: None,
        };

        // Both users have an entity named "Alice"
        store
            .add_entity("Alice", "person", &serde_json::json!({"owner": "user1"}), &user1)
            .unwrap();
        store
            .add_entity("Alice", "person", &serde_json::json!({"owner": "user2"}), &user2)
            .unwrap();

        // Total entities should be 2
        assert_eq!(store.entity_count().unwrap(), 2);

        // Each user should see only their Alice
        let user1_entities = store.get_all(&user1).await.unwrap();
        assert_eq!(user1_entities.len(), 1);
        assert_eq!(user1_entities[0].properties["owner"], "user1");

        let user2_entities = store.get_all(&user2).await.unwrap();
        assert_eq!(user2_entities.len(), 1);
        assert_eq!(user2_entities[0].properties["owner"], "user2");
    }
}
