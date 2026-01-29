//! In-memory graph operations using petgraph DiGraph.
//!
//! Provides O(1) neighbor lookups and efficient traversal operations
//! for the knowledge graph.

use std::collections::HashMap;

use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::EdgeRef;
use serde::{Deserialize, Serialize};

/// Node data in the memory graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityNode {
    /// Database ID (from SQLite).
    pub db_id: i64,
    /// Entity name.
    pub name: String,
    /// Entity type/label (e.g., "person", "location", "concept").
    pub entity_type: String,
    /// Additional properties as JSON.
    pub properties: serde_json::Value,
    /// User ID filter.
    pub user_id: Option<String>,
    /// Agent ID filter.
    pub agent_id: Option<String>,
    /// Run ID filter.
    pub run_id: Option<String>,
}

impl EntityNode {
    /// Create a new entity node.
    pub fn new(
        db_id: i64,
        name: impl Into<String>,
        entity_type: impl Into<String>,
    ) -> Self {
        Self {
            db_id,
            name: name.into(),
            entity_type: entity_type.into(),
            properties: serde_json::Value::Object(serde_json::Map::new()),
            user_id: None,
            agent_id: None,
            run_id: None,
        }
    }

    /// Set properties.
    pub fn with_properties(mut self, properties: serde_json::Value) -> Self {
        self.properties = properties;
        self
    }

    /// Set user ID filter.
    pub fn with_user_id(mut self, user_id: impl Into<String>) -> Self {
        self.user_id = Some(user_id.into());
        self
    }

    /// Set agent ID filter.
    pub fn with_agent_id(mut self, agent_id: impl Into<String>) -> Self {
        self.agent_id = Some(agent_id.into());
        self
    }

    /// Set run ID filter.
    pub fn with_run_id(mut self, run_id: impl Into<String>) -> Self {
        self.run_id = Some(run_id.into());
        self
    }

    /// Check if this node matches the given filters.
    pub fn matches_filters(
        &self,
        user_id: Option<&str>,
        agent_id: Option<&str>,
        run_id: Option<&str>,
    ) -> bool {
        // If filter is specified, node must match it (or have no restriction)
        let user_match = match user_id {
            Some(uid) => self.user_id.as_deref() == Some(uid) || self.user_id.is_none(),
            None => true,
        };
        let agent_match = match agent_id {
            Some(aid) => self.agent_id.as_deref() == Some(aid) || self.agent_id.is_none(),
            None => true,
        };
        let run_match = match run_id {
            Some(rid) => self.run_id.as_deref() == Some(rid) || self.run_id.is_none(),
            None => true,
        };
        user_match && agent_match && run_match
    }
}

/// Edge data in the memory graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationshipEdge {
    /// Database ID (from SQLite).
    pub db_id: i64,
    /// Relationship type (e.g., "knows", "works_at", "located_in").
    pub relationship_type: String,
    /// Edge weight (for spreading activation).
    pub weight: f64,
    /// Additional properties as JSON.
    pub properties: serde_json::Value,
}

impl RelationshipEdge {
    /// Create a new relationship edge.
    pub fn new(db_id: i64, relationship_type: impl Into<String>) -> Self {
        Self {
            db_id,
            relationship_type: relationship_type.into(),
            weight: 1.0,
            properties: serde_json::Value::Object(serde_json::Map::new()),
        }
    }

    /// Set edge weight.
    pub fn with_weight(mut self, weight: f64) -> Self {
        self.weight = weight;
        self
    }

    /// Set properties.
    pub fn with_properties(mut self, properties: serde_json::Value) -> Self {
        self.properties = properties;
        self
    }
}

/// The in-memory graph type using petgraph.
pub type MemoryGraph = DiGraph<EntityNode, RelationshipEdge>;

/// Index for O(1) lookups by database ID.
pub type DbIdIndex = HashMap<i64, NodeIndex>;

/// Index for O(1) lookups by name (scoped by user/agent/run).
pub type NameIndex = HashMap<String, NodeIndex>;

/// Graph operations on the in-memory graph.
pub struct GraphOps<'a> {
    graph: &'a mut MemoryGraph,
    db_id_index: &'a mut DbIdIndex,
    name_index: &'a mut NameIndex,
}

impl<'a> GraphOps<'a> {
    /// Create a new GraphOps instance.
    pub fn new(
        graph: &'a mut MemoryGraph,
        db_id_index: &'a mut DbIdIndex,
        name_index: &'a mut NameIndex,
    ) -> Self {
        Self {
            graph,
            db_id_index,
            name_index,
        }
    }

    /// Add an entity node to the graph.
    ///
    /// Returns the NodeIndex for the new node.
    pub fn add_entity(&mut self, entity: EntityNode) -> NodeIndex {
        let db_id = entity.db_id;
        let name_key = Self::make_name_key(
            &entity.name,
            entity.user_id.as_deref(),
            entity.agent_id.as_deref(),
            entity.run_id.as_deref(),
        );

        let idx = self.graph.add_node(entity);
        self.db_id_index.insert(db_id, idx);
        self.name_index.insert(name_key, idx);
        idx
    }

    /// Add a relationship edge between two entities.
    ///
    /// Returns true if the edge was added, false if source or target not found.
    pub fn add_relationship(
        &mut self,
        source_db_id: i64,
        target_db_id: i64,
        edge: RelationshipEdge,
    ) -> bool {
        let source_idx = match self.db_id_index.get(&source_db_id) {
            Some(idx) => *idx,
            None => return false,
        };
        let target_idx = match self.db_id_index.get(&target_db_id) {
            Some(idx) => *idx,
            None => return false,
        };

        self.graph.add_edge(source_idx, target_idx, edge);
        true
    }

    /// Find a node by database ID.
    pub fn find_by_db_id(&self, db_id: i64) -> Option<NodeIndex> {
        self.db_id_index.get(&db_id).copied()
    }

    /// Find a node by name and filters.
    pub fn find_by_name(
        &self,
        name: &str,
        user_id: Option<&str>,
        agent_id: Option<&str>,
        run_id: Option<&str>,
    ) -> Option<NodeIndex> {
        let key = Self::make_name_key(name, user_id, agent_id, run_id);
        self.name_index.get(&key).copied()
    }

    /// Get outgoing neighbors (entities this entity points to).
    ///
    /// O(1) lookup via petgraph adjacency list.
    pub fn outgoing_neighbors(&self, node_idx: NodeIndex) -> Vec<(NodeIndex, &RelationshipEdge)> {
        self.graph
            .edges(node_idx)
            .map(|e| (e.target(), e.weight()))
            .collect()
    }

    /// Get incoming neighbors (entities pointing to this entity).
    ///
    /// O(degree) lookup via petgraph.
    pub fn incoming_neighbors(
        &self,
        node_idx: NodeIndex,
    ) -> Vec<(NodeIndex, &RelationshipEdge)> {
        self.graph
            .edges_directed(node_idx, petgraph::Direction::Incoming)
            .map(|e| (e.source(), e.weight()))
            .collect()
    }

    /// Get all neighbors (both directions).
    pub fn all_neighbors(&self, node_idx: NodeIndex) -> Vec<(NodeIndex, &RelationshipEdge)> {
        let mut neighbors = self.outgoing_neighbors(node_idx);
        neighbors.extend(self.incoming_neighbors(node_idx));
        neighbors
    }

    /// Get the entity node at the given index.
    pub fn get_entity(&self, node_idx: NodeIndex) -> Option<&EntityNode> {
        self.graph.node_weight(node_idx)
    }

    /// Remove a node and all its edges.
    ///
    /// Returns the removed entity if it existed.
    pub fn remove_entity(&mut self, db_id: i64) -> Option<EntityNode> {
        let node_idx = self.db_id_index.remove(&db_id)?;

        // Get entity to build name key for removal
        if let Some(entity) = self.graph.node_weight(node_idx) {
            let name_key = Self::make_name_key(
                &entity.name,
                entity.user_id.as_deref(),
                entity.agent_id.as_deref(),
                entity.run_id.as_deref(),
            );
            self.name_index.remove(&name_key);
        }

        self.graph.remove_node(node_idx)
    }

    /// Clear all nodes and edges from the graph.
    pub fn clear(&mut self) {
        self.graph.clear();
        self.db_id_index.clear();
        self.name_index.clear();
    }

    /// Get the number of nodes in the graph.
    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Get the number of edges in the graph.
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// Create a name key for indexing.
    fn make_name_key(
        name: &str,
        user_id: Option<&str>,
        agent_id: Option<&str>,
        run_id: Option<&str>,
    ) -> String {
        format!(
            "{}:{}:{}:{}",
            name,
            user_id.unwrap_or(""),
            agent_id.unwrap_or(""),
            run_id.unwrap_or("")
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entity_node_creation() {
        let node = EntityNode::new(1, "Alice", "person")
            .with_user_id("user1")
            .with_properties(serde_json::json!({"age": 30}));

        assert_eq!(node.db_id, 1);
        assert_eq!(node.name, "Alice");
        assert_eq!(node.entity_type, "person");
        assert_eq!(node.user_id, Some("user1".to_string()));
        assert_eq!(node.properties["age"], 30);
    }

    #[test]
    fn test_entity_filter_matching() {
        let node = EntityNode::new(1, "Alice", "person").with_user_id("user1");

        // Matching filter
        assert!(node.matches_filters(Some("user1"), None, None));

        // Non-matching filter
        assert!(!node.matches_filters(Some("user2"), None, None));

        // No filter should match
        assert!(node.matches_filters(None, None, None));
    }

    #[test]
    fn test_graph_add_and_find() {
        let mut graph = MemoryGraph::new();
        let mut db_id_index = DbIdIndex::new();
        let mut name_index = NameIndex::new();
        let mut ops = GraphOps::new(&mut graph, &mut db_id_index, &mut name_index);

        let entity = EntityNode::new(1, "Alice", "person").with_user_id("user1");
        let idx = ops.add_entity(entity);

        // Find by db_id
        assert_eq!(ops.find_by_db_id(1), Some(idx));

        // Find by name
        assert_eq!(ops.find_by_name("Alice", Some("user1"), None, None), Some(idx));

        // Not found
        assert_eq!(ops.find_by_db_id(999), None);
    }

    #[test]
    fn test_graph_relationships() {
        let mut graph = MemoryGraph::new();
        let mut db_id_index = DbIdIndex::new();
        let mut name_index = NameIndex::new();
        let mut ops = GraphOps::new(&mut graph, &mut db_id_index, &mut name_index);

        // Add entities
        let alice = EntityNode::new(1, "Alice", "person");
        let bob = EntityNode::new(2, "Bob", "person");
        let alice_idx = ops.add_entity(alice);
        let _bob_idx = ops.add_entity(bob);

        // Add relationship
        let edge = RelationshipEdge::new(1, "knows");
        assert!(ops.add_relationship(1, 2, edge));

        // Check neighbors
        let neighbors = ops.outgoing_neighbors(alice_idx);
        assert_eq!(neighbors.len(), 1);
        assert_eq!(neighbors[0].1.relationship_type, "knows");
    }

    #[test]
    fn test_graph_remove_entity() {
        let mut graph = MemoryGraph::new();
        let mut db_id_index = DbIdIndex::new();
        let mut name_index = NameIndex::new();
        let mut ops = GraphOps::new(&mut graph, &mut db_id_index, &mut name_index);

        let entity = EntityNode::new(1, "Alice", "person");
        ops.add_entity(entity);

        assert_eq!(ops.node_count(), 1);

        let removed = ops.remove_entity(1);
        assert!(removed.is_some());
        assert_eq!(ops.node_count(), 0);
        assert_eq!(ops.find_by_db_id(1), None);
    }
}
