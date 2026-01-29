//! SQLite <-> petgraph synchronization.
//!
//! Provides functions to load graph data from SQLite into petgraph
//! and persist changes back to SQLite.

use rusqlite::{params, Connection};

use rook_core::error::RookResult;
use rook_core::traits::GraphFilters;

use super::petgraph_ops::{DbIdIndex, EntityNode, MemoryGraph, NameIndex, RelationshipEdge};

/// Load the entire graph from SQLite into petgraph.
///
/// This is called on startup to hydrate the in-memory graph.
pub fn load_graph(
    conn: &Connection,
    graph: &mut MemoryGraph,
    db_id_index: &mut DbIdIndex,
    name_index: &mut NameIndex,
) -> RookResult<()> {
    // Clear existing data
    graph.clear();
    db_id_index.clear();
    name_index.clear();

    // Load all entities
    let mut stmt = conn.prepare(
        "SELECT id, name, entity_type, properties, user_id, agent_id, run_id FROM entities",
    )?;

    let entity_iter = stmt.query_map([], |row| {
        let id: i64 = row.get(0)?;
        let name: String = row.get(1)?;
        let entity_type: String = row.get(2)?;
        let properties_str: String = row.get(3)?;
        let user_id: Option<String> = row.get(4)?;
        let agent_id: Option<String> = row.get(5)?;
        let run_id: Option<String> = row.get(6)?;

        let properties = serde_json::from_str(&properties_str).unwrap_or_default();

        Ok(EntityNode {
            db_id: id,
            name,
            entity_type,
            properties,
            user_id,
            agent_id,
            run_id,
        })
    })?;

    for entity_result in entity_iter {
        let entity = entity_result?;
        let db_id = entity.db_id;
        let name_key = make_name_key(
            &entity.name,
            entity.user_id.as_deref(),
            entity.agent_id.as_deref(),
            entity.run_id.as_deref(),
        );

        let idx = graph.add_node(entity);
        db_id_index.insert(db_id, idx);
        name_index.insert(name_key, idx);
    }

    // Load all relationships
    let mut stmt = conn.prepare(
        "SELECT id, source_id, target_id, relationship_type, properties, weight FROM relationships",
    )?;

    let relationship_iter = stmt.query_map([], |row| {
        let id: i64 = row.get(0)?;
        let source_id: i64 = row.get(1)?;
        let target_id: i64 = row.get(2)?;
        let relationship_type: String = row.get(3)?;
        let properties_str: String = row.get(4)?;
        let weight: f64 = row.get(5)?;

        let properties = serde_json::from_str(&properties_str).unwrap_or_default();

        Ok((
            source_id,
            target_id,
            RelationshipEdge {
                db_id: id,
                relationship_type,
                weight,
                properties,
            },
        ))
    })?;

    for rel_result in relationship_iter {
        let (source_id, target_id, edge) = rel_result?;

        if let (Some(&source_idx), Some(&target_idx)) =
            (db_id_index.get(&source_id), db_id_index.get(&target_id))
        {
            graph.add_edge(source_idx, target_idx, edge);
        }
    }

    Ok(())
}

/// Save an entity to SQLite.
///
/// Uses INSERT OR REPLACE to handle both new and existing entities.
/// Returns the database ID of the entity.
pub fn save_entity(
    conn: &Connection,
    name: &str,
    entity_type: &str,
    properties: &serde_json::Value,
    filters: &GraphFilters,
) -> RookResult<i64> {
    let properties_str = serde_json::to_string(properties)?;

    // Try to insert, on conflict update
    conn.execute(
        r#"
        INSERT INTO entities (name, entity_type, properties, user_id, agent_id, run_id, updated_at)
        VALUES (?1, ?2, ?3, ?4, ?5, ?6, datetime('now'))
        ON CONFLICT(name, user_id, agent_id, run_id) DO UPDATE SET
            entity_type = excluded.entity_type,
            properties = excluded.properties,
            updated_at = datetime('now')
        "#,
        params![
            name,
            entity_type,
            properties_str,
            filters.user_id,
            filters.agent_id,
            filters.run_id
        ],
    )?;

    // Get the ID (either newly inserted or existing)
    let id: i64 = conn.query_row(
        "SELECT id FROM entities WHERE name = ?1 AND user_id IS ?2 AND agent_id IS ?3 AND run_id IS ?4",
        params![name, filters.user_id, filters.agent_id, filters.run_id],
        |row| row.get(0),
    )?;

    Ok(id)
}

/// Save a relationship to SQLite.
///
/// Returns the database ID of the relationship.
pub fn save_relationship(
    conn: &Connection,
    source_id: i64,
    target_id: i64,
    relationship_type: &str,
    properties: &serde_json::Value,
    weight: f64,
) -> RookResult<i64> {
    let properties_str = serde_json::to_string(properties)?;

    conn.execute(
        r#"
        INSERT INTO relationships (source_id, target_id, relationship_type, properties, weight, updated_at)
        VALUES (?1, ?2, ?3, ?4, ?5, datetime('now'))
        ON CONFLICT(source_id, target_id, relationship_type) DO UPDATE SET
            properties = excluded.properties,
            weight = excluded.weight,
            updated_at = datetime('now')
        "#,
        params![source_id, target_id, relationship_type, properties_str, weight],
    )?;

    let id: i64 = conn.query_row(
        "SELECT id FROM relationships WHERE source_id = ?1 AND target_id = ?2 AND relationship_type = ?3",
        params![source_id, target_id, relationship_type],
        |row| row.get(0),
    )?;

    Ok(id)
}

/// Delete an entity from SQLite.
///
/// Also deletes all related relationships (via CASCADE).
pub fn delete_entity(conn: &Connection, entity_id: i64) -> RookResult<bool> {
    let rows = conn.execute("DELETE FROM entities WHERE id = ?1", params![entity_id])?;
    Ok(rows > 0)
}

/// Delete entities matching filters.
pub fn delete_entities_by_filters(conn: &Connection, filters: &GraphFilters) -> RookResult<usize> {
    let rows = conn.execute(
        "DELETE FROM entities WHERE user_id IS ?1 AND agent_id IS ?2 AND run_id IS ?3",
        params![filters.user_id, filters.agent_id, filters.run_id],
    )?;
    Ok(rows)
}

/// Get all entity IDs matching filters.
pub fn get_entity_ids_by_filters(conn: &Connection, filters: &GraphFilters) -> RookResult<Vec<i64>> {
    let mut stmt = conn.prepare(
        "SELECT id FROM entities WHERE user_id IS ?1 AND agent_id IS ?2 AND run_id IS ?3",
    )?;

    let ids: Vec<i64> = stmt
        .query_map(params![filters.user_id, filters.agent_id, filters.run_id], |row| {
            row.get(0)
        })?
        .filter_map(|r| r.ok())
        .collect();

    Ok(ids)
}

/// Log an entity access for spreading activation.
pub fn log_entity_access(
    conn: &Connection,
    entity_id: i64,
    access_type: &str,
    activation_score: f64,
) -> RookResult<()> {
    conn.execute(
        r#"
        INSERT INTO entity_access_log (entity_id, access_type, activation_score)
        VALUES (?1, ?2, ?3)
        "#,
        params![entity_id, access_type, activation_score],
    )?;
    Ok(())
}

/// Link a memory to an entity.
pub fn link_memory_to_entity(
    conn: &Connection,
    memory_id: &str,
    entity_id: i64,
    role: &str,
) -> RookResult<()> {
    conn.execute(
        r#"
        INSERT INTO memory_entities (memory_id, entity_id, role)
        VALUES (?1, ?2, ?3)
        ON CONFLICT(memory_id, entity_id) DO UPDATE SET role = excluded.role
        "#,
        params![memory_id, entity_id, role],
    )?;
    Ok(())
}

/// Get all entity IDs linked to a memory.
pub fn get_entities_for_memory(conn: &Connection, memory_id: &str) -> RookResult<Vec<i64>> {
    let mut stmt = conn.prepare("SELECT entity_id FROM memory_entities WHERE memory_id = ?1")?;
    let ids: Vec<i64> = stmt
        .query_map(params![memory_id], |row| row.get(0))?
        .filter_map(|r| r.ok())
        .collect();
    Ok(ids)
}

/// Get all memory IDs linked to an entity.
pub fn get_memories_for_entity(conn: &Connection, entity_id: i64) -> RookResult<Vec<String>> {
    let mut stmt = conn.prepare("SELECT memory_id FROM memory_entities WHERE entity_id = ?1")?;
    let ids: Vec<String> = stmt
        .query_map(params![entity_id], |row| row.get(0))?
        .filter_map(|r| r.ok())
        .collect();
    Ok(ids)
}

/// Helper to create name key for indexing.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedded::schema::init_schema;

    fn setup_test_db() -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        init_schema(&conn).unwrap();
        conn
    }

    #[test]
    fn test_save_and_load_entity() {
        let conn = setup_test_db();
        let filters = GraphFilters {
            user_id: Some("user1".to_string()),
            agent_id: None,
            run_id: None,
        };

        let id = save_entity(
            &conn,
            "Alice",
            "person",
            &serde_json::json!({"age": 30}),
            &filters,
        )
        .unwrap();

        assert!(id > 0);

        // Load graph and verify
        let mut graph = MemoryGraph::new();
        let mut db_id_index = DbIdIndex::new();
        let mut name_index = NameIndex::new();

        load_graph(&conn, &mut graph, &mut db_id_index, &mut name_index).unwrap();

        assert_eq!(graph.node_count(), 1);
        let node = graph.node_weight(db_id_index[&id]).unwrap();
        assert_eq!(node.name, "Alice");
        assert_eq!(node.entity_type, "person");
    }

    #[test]
    fn test_save_and_load_relationship() {
        let conn = setup_test_db();
        let filters = GraphFilters::default();

        let alice_id = save_entity(&conn, "Alice", "person", &serde_json::json!({}), &filters).unwrap();
        let bob_id = save_entity(&conn, "Bob", "person", &serde_json::json!({}), &filters).unwrap();

        let rel_id = save_relationship(
            &conn,
            alice_id,
            bob_id,
            "knows",
            &serde_json::json!({"since": "2020"}),
            1.0,
        )
        .unwrap();

        assert!(rel_id > 0);

        // Load graph and verify
        let mut graph = MemoryGraph::new();
        let mut db_id_index = DbIdIndex::new();
        let mut name_index = NameIndex::new();

        load_graph(&conn, &mut graph, &mut db_id_index, &mut name_index).unwrap();

        assert_eq!(graph.node_count(), 2);
        assert_eq!(graph.edge_count(), 1);
    }

    #[test]
    fn test_delete_entity_cascades() {
        let conn = setup_test_db();
        let filters = GraphFilters::default();

        let alice_id = save_entity(&conn, "Alice", "person", &serde_json::json!({}), &filters).unwrap();
        let bob_id = save_entity(&conn, "Bob", "person", &serde_json::json!({}), &filters).unwrap();

        save_relationship(&conn, alice_id, bob_id, "knows", &serde_json::json!({}), 1.0).unwrap();

        // Delete Alice
        delete_entity(&conn, alice_id).unwrap();

        // Relationship should be gone too
        let count: i32 = conn
            .query_row("SELECT COUNT(*) FROM relationships", [], |row| row.get(0))
            .unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn test_memory_entity_linking() {
        let conn = setup_test_db();
        let filters = GraphFilters::default();

        let alice_id = save_entity(&conn, "Alice", "person", &serde_json::json!({}), &filters).unwrap();

        link_memory_to_entity(&conn, "mem-123", alice_id, "subject").unwrap();

        let entities = get_entities_for_memory(&conn, "mem-123").unwrap();
        assert_eq!(entities, vec![alice_id]);

        let memories = get_memories_for_entity(&conn, alice_id).unwrap();
        assert_eq!(memories, vec!["mem-123".to_string()]);
    }
}
