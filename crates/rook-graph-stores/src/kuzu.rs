//! Kuzu graph store implementation.
//! Kuzu is an embedded graph database.

use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

use rook_core::error::RookResult;
use rook_core::traits::{Entity, GraphFilters, GraphStore, GraphStoreConfig};
use rook_core::types::{GraphRelation, Message};

/// In-memory graph store (placeholder for Kuzu).
/// Kuzu requires native bindings which are not yet stable.
pub struct KuzuGraphStore {
    /// In-memory storage of relations.
    relations: Arc<RwLock<HashMap<String, Vec<GraphRelation>>>>,
    /// In-memory storage of entities.
    entities: Arc<RwLock<HashMap<String, Vec<Entity>>>>,
    #[allow(dead_code)]
    config: GraphStoreConfig,
}

impl KuzuGraphStore {
    /// Create a new Kuzu graph store.
    pub async fn new(config: GraphStoreConfig) -> RookResult<Self> {
        Ok(Self {
            relations: Arc::new(RwLock::new(HashMap::new())),
            entities: Arc::new(RwLock::new(HashMap::new())),
            config,
        })
    }
}

#[async_trait]
impl GraphStore for KuzuGraphStore {
    async fn add(
        &self,
        messages: &[Message],
        filters: &GraphFilters,
    ) -> RookResult<Vec<GraphRelation>> {
        let user_id = filters.user_id.clone().unwrap_or_else(|| "default".to_string());
        let mut new_relations = Vec::new();

        for message in messages {
            let relation = GraphRelation {
                source: user_id.clone(),
                relationship: "HAS_MEMORY".to_string(),
                target: message.content.clone(),
            };
            new_relations.push(relation.clone());

            // Store entity
            let entity = Entity {
                name: message.content.clone(),
                entity_type: "Memory".to_string(),
                properties: serde_json::json!({}),
            };

            let mut entities = self.entities.write().await;
            entities
                .entry(user_id.clone())
                .or_default()
                .push(entity);
        }

        // Store relations
        let mut relations = self.relations.write().await;
        relations
            .entry(user_id)
            .or_default()
            .extend(new_relations.clone());

        Ok(new_relations)
    }

    async fn search(
        &self,
        query: &str,
        filters: &GraphFilters,
        limit: usize,
    ) -> RookResult<Vec<GraphRelation>> {
        let user_id = filters.user_id.clone().unwrap_or_else(|| "default".to_string());
        let relations = self.relations.read().await;

        let results: Vec<GraphRelation> = relations
            .get(&user_id)
            .map(|rels| {
                rels.iter()
                    .filter(|r| r.target.contains(query))
                    .take(limit)
                    .cloned()
                    .collect()
            })
            .unwrap_or_default();

        Ok(results)
    }

    async fn delete_all(&self, filters: &GraphFilters) -> RookResult<()> {
        let user_id = filters.user_id.clone().unwrap_or_else(|| "default".to_string());

        let mut relations = self.relations.write().await;
        relations.remove(&user_id);

        let mut entities = self.entities.write().await;
        entities.remove(&user_id);

        Ok(())
    }

    async fn get_all(&self, filters: &GraphFilters) -> RookResult<Vec<Entity>> {
        let user_id = filters.user_id.clone().unwrap_or_else(|| "default".to_string());
        let entities = self.entities.read().await;

        Ok(entities.get(&user_id).cloned().unwrap_or_default())
    }
}
