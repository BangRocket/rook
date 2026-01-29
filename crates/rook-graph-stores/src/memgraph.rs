//! Memgraph graph store implementation.
//! Memgraph is compatible with Neo4j protocol.

use async_trait::async_trait;
use neo4rs::{query, Graph};

use rook_core::error::{RookError, RookResult};
use rook_core::traits::{Entity, GraphFilters, GraphStore, GraphStoreConfig};
use rook_core::types::{GraphRelation, Message};

/// Memgraph graph store implementation.
/// Uses the same protocol as Neo4j.
pub struct MemgraphStore {
    graph: Graph,
    #[allow(dead_code)]
    config: GraphStoreConfig,
}

impl MemgraphStore {
    /// Create a new Memgraph store.
    pub async fn new(config: GraphStoreConfig) -> RookResult<Self> {
        let uri = config.url.clone();
        let username = config.username.clone().unwrap_or_else(|| "memgraph".to_string());
        let password = config.password.clone().unwrap_or_default();

        let graph = Graph::new(&uri, &username, &password)
            .await
            .map_err(|e| RookError::graph_store(format!("Failed to connect to Memgraph: {}", e)))?;

        Ok(Self { graph, config })
    }
}

#[async_trait]
impl GraphStore for MemgraphStore {
    async fn add(
        &self,
        messages: &[Message],
        filters: &GraphFilters,
    ) -> RookResult<Vec<GraphRelation>> {
        let mut relations = Vec::new();

        for message in messages {
            let user_id = filters.user_id.clone().unwrap_or_else(|| "default".to_string());

            let q = query(
                r#"
                MERGE (u:User {id: $user_id})
                CREATE (m:Memory {content: $content})
                MERGE (u)-[r:HAS_MEMORY]->(m)
                RETURN m.content as content
                "#,
            )
            .param("user_id", user_id.clone())
            .param("content", message.content.clone());

            self.graph
                .run(q)
                .await
                .map_err(|e| RookError::graph_store(format!("Failed to add to graph: {}", e)))?;

            relations.push(GraphRelation {
                source: user_id,
                relationship: "HAS_MEMORY".to_string(),
                target: message.content.clone(),
            });
        }

        Ok(relations)
    }

    async fn search(
        &self,
        query_str: &str,
        filters: &GraphFilters,
        limit: usize,
    ) -> RookResult<Vec<GraphRelation>> {
        let user_id = filters.user_id.clone().unwrap_or_else(|| "default".to_string());

        let q = query(
            r#"
            MATCH (u:User {id: $user_id})-[r]->(m)
            WHERE m.content CONTAINS $query
            RETURN u.id as source, type(r) as relationship, m.content as target
            LIMIT $limit
            "#,
        )
        .param("user_id", user_id)
        .param("query", query_str.to_string())
        .param("limit", limit as i64);

        let mut result = self
            .graph
            .execute(q)
            .await
            .map_err(|e| RookError::graph_store(format!("Failed to search: {}", e)))?;

        let mut relations = Vec::new();
        while let Some(row) = result
            .next()
            .await
            .map_err(|e| RookError::graph_store(format!("Failed to fetch row: {}", e)))?
        {
            relations.push(GraphRelation {
                source: row.get("source").unwrap_or_default(),
                relationship: row.get("relationship").unwrap_or_default(),
                target: row.get("target").unwrap_or_default(),
            });
        }

        Ok(relations)
    }

    async fn delete_all(&self, filters: &GraphFilters) -> RookResult<()> {
        let user_id = filters.user_id.clone().unwrap_or_else(|| "default".to_string());

        let q = query(
            r#"
            MATCH (u:User {id: $user_id})-[r]->(m)
            DETACH DELETE m
            "#,
        )
        .param("user_id", user_id);

        self.graph
            .run(q)
            .await
            .map_err(|e| RookError::graph_store(format!("Failed to delete: {}", e)))?;

        Ok(())
    }

    async fn get_all(&self, filters: &GraphFilters) -> RookResult<Vec<Entity>> {
        let user_id = filters.user_id.clone().unwrap_or_else(|| "default".to_string());

        let q = query(
            r#"
            MATCH (u:User {id: $user_id})-[r]->(m)
            RETURN labels(m)[0] as entity_type, m.content as name
            "#,
        )
        .param("user_id", user_id);

        let mut result = self
            .graph
            .execute(q)
            .await
            .map_err(|e| RookError::graph_store(format!("Failed to get entities: {}", e)))?;

        let mut entities = Vec::new();
        while let Some(row) = result
            .next()
            .await
            .map_err(|e| RookError::graph_store(format!("Failed to fetch row: {}", e)))?
        {
            entities.push(Entity {
                name: row.get("name").unwrap_or_default(),
                entity_type: row.get("entity_type").unwrap_or_else(|| "Memory".to_string()),
                properties: serde_json::json!({}),
            });
        }

        Ok(entities)
    }
}