//! SQLite schema for embedded graph store.
//!
//! Provides persistent storage for the knowledge graph with four tables:
//! - `entities`: Node data (name, type, properties)
//! - `relationships`: Edge data (source, target, type, properties)
//! - `memory_entities`: Links memories to entities
//! - `entity_access_log`: Tracks entity access patterns for spreading activation

use rusqlite::Connection;

use rook_core::error::RookResult;

/// SQL statements for creating the graph schema.
pub const CREATE_ENTITIES_TABLE: &str = r#"
CREATE TABLE IF NOT EXISTS entities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    properties TEXT NOT NULL DEFAULT '{}',
    user_id TEXT,
    agent_id TEXT,
    run_id TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(name, user_id, agent_id, run_id)
)
"#;

/// Index for efficient entity lookups by name.
pub const CREATE_ENTITIES_NAME_INDEX: &str = r#"
CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name)
"#;

/// Index for efficient entity lookups by type.
pub const CREATE_ENTITIES_TYPE_INDEX: &str = r#"
CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type)
"#;

/// Index for filtering by user/agent/run.
pub const CREATE_ENTITIES_FILTER_INDEX: &str = r#"
CREATE INDEX IF NOT EXISTS idx_entities_filter ON entities(user_id, agent_id, run_id)
"#;

/// SQL for relationships table.
pub const CREATE_RELATIONSHIPS_TABLE: &str = r#"
CREATE TABLE IF NOT EXISTS relationships (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    target_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    relationship_type TEXT NOT NULL,
    properties TEXT NOT NULL DEFAULT '{}',
    weight REAL NOT NULL DEFAULT 1.0,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(source_id, target_id, relationship_type)
)
"#;

/// Index for efficient traversal from source.
pub const CREATE_RELATIONSHIPS_SOURCE_INDEX: &str = r#"
CREATE INDEX IF NOT EXISTS idx_relationships_source ON relationships(source_id)
"#;

/// Index for efficient traversal to target.
pub const CREATE_RELATIONSHIPS_TARGET_INDEX: &str = r#"
CREATE INDEX IF NOT EXISTS idx_relationships_target ON relationships(target_id)
"#;

/// Index for relationship type filtering.
pub const CREATE_RELATIONSHIPS_TYPE_INDEX: &str = r#"
CREATE INDEX IF NOT EXISTS idx_relationships_type ON relationships(relationship_type)
"#;

/// SQL for memory-entity links.
pub const CREATE_MEMORY_ENTITIES_TABLE: &str = r#"
CREATE TABLE IF NOT EXISTS memory_entities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id TEXT NOT NULL,
    entity_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    role TEXT NOT NULL DEFAULT 'mentioned',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(memory_id, entity_id)
)
"#;

/// Index for finding entities in a memory.
pub const CREATE_MEMORY_ENTITIES_MEMORY_INDEX: &str = r#"
CREATE INDEX IF NOT EXISTS idx_memory_entities_memory ON memory_entities(memory_id)
"#;

/// Index for finding memories containing an entity.
pub const CREATE_MEMORY_ENTITIES_ENTITY_INDEX: &str = r#"
CREATE INDEX IF NOT EXISTS idx_memory_entities_entity ON memory_entities(entity_id)
"#;

/// SQL for entity access log (for spreading activation).
pub const CREATE_ENTITY_ACCESS_LOG_TABLE: &str = r#"
CREATE TABLE IF NOT EXISTS entity_access_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    access_type TEXT NOT NULL,
    activation_score REAL NOT NULL DEFAULT 1.0,
    accessed_at TEXT NOT NULL DEFAULT (datetime('now'))
)
"#;

/// Index for access log by entity.
pub const CREATE_ACCESS_LOG_ENTITY_INDEX: &str = r#"
CREATE INDEX IF NOT EXISTS idx_access_log_entity ON entity_access_log(entity_id)
"#;

/// Index for access log by time (for decay calculations).
pub const CREATE_ACCESS_LOG_TIME_INDEX: &str = r#"
CREATE INDEX IF NOT EXISTS idx_access_log_time ON entity_access_log(accessed_at)
"#;

/// Initialize the graph schema in the given database connection.
///
/// Creates all tables and indexes if they don't exist.
/// Safe to call multiple times (idempotent).
pub fn init_schema(conn: &Connection) -> RookResult<()> {
    // Enable foreign keys
    conn.execute("PRAGMA foreign_keys = ON", [])?;

    // Create tables
    conn.execute(CREATE_ENTITIES_TABLE, [])?;
    conn.execute(CREATE_RELATIONSHIPS_TABLE, [])?;
    conn.execute(CREATE_MEMORY_ENTITIES_TABLE, [])?;
    conn.execute(CREATE_ENTITY_ACCESS_LOG_TABLE, [])?;

    // Create indexes for entities
    conn.execute(CREATE_ENTITIES_NAME_INDEX, [])?;
    conn.execute(CREATE_ENTITIES_TYPE_INDEX, [])?;
    conn.execute(CREATE_ENTITIES_FILTER_INDEX, [])?;

    // Create indexes for relationships
    conn.execute(CREATE_RELATIONSHIPS_SOURCE_INDEX, [])?;
    conn.execute(CREATE_RELATIONSHIPS_TARGET_INDEX, [])?;
    conn.execute(CREATE_RELATIONSHIPS_TYPE_INDEX, [])?;

    // Create indexes for memory_entities
    conn.execute(CREATE_MEMORY_ENTITIES_MEMORY_INDEX, [])?;
    conn.execute(CREATE_MEMORY_ENTITIES_ENTITY_INDEX, [])?;

    // Create indexes for access log
    conn.execute(CREATE_ACCESS_LOG_ENTITY_INDEX, [])?;
    conn.execute(CREATE_ACCESS_LOG_TIME_INDEX, [])?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init_schema_creates_tables() {
        let conn = Connection::open_in_memory().unwrap();
        init_schema(&conn).unwrap();

        // Verify tables exist
        let tables: Vec<String> = conn
            .prepare("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            .unwrap()
            .query_map([], |row| row.get(0))
            .unwrap()
            .filter_map(|r| r.ok())
            .collect();

        assert!(tables.contains(&"entities".to_string()));
        assert!(tables.contains(&"relationships".to_string()));
        assert!(tables.contains(&"memory_entities".to_string()));
        assert!(tables.contains(&"entity_access_log".to_string()));
    }

    #[test]
    fn test_init_schema_idempotent() {
        let conn = Connection::open_in_memory().unwrap();

        // Call init_schema multiple times
        init_schema(&conn).unwrap();
        init_schema(&conn).unwrap();
        init_schema(&conn).unwrap();

        // Should still work fine
        let count: i32 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='entities'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(count, 1);
    }

    #[test]
    fn test_entity_unique_constraint() {
        let conn = Connection::open_in_memory().unwrap();
        init_schema(&conn).unwrap();

        // Insert first entity
        conn.execute(
            "INSERT INTO entities (name, entity_type, user_id) VALUES ('Alice', 'person', 'user1')",
            [],
        )
        .unwrap();

        // Same entity with same filters should fail
        let result = conn.execute(
            "INSERT INTO entities (name, entity_type, user_id) VALUES ('Alice', 'person', 'user1')",
            [],
        );
        assert!(result.is_err());

        // Same entity with different user should succeed
        conn.execute(
            "INSERT INTO entities (name, entity_type, user_id) VALUES ('Alice', 'person', 'user2')",
            [],
        )
        .unwrap();
    }

    #[test]
    fn test_relationship_cascade_delete() {
        let conn = Connection::open_in_memory().unwrap();
        init_schema(&conn).unwrap();

        // Insert entities
        conn.execute(
            "INSERT INTO entities (name, entity_type) VALUES ('Alice', 'person')",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO entities (name, entity_type) VALUES ('Bob', 'person')",
            [],
        )
        .unwrap();

        // Insert relationship
        conn.execute(
            "INSERT INTO relationships (source_id, target_id, relationship_type) VALUES (1, 2, 'knows')",
            [],
        )
        .unwrap();

        // Verify relationship exists
        let count: i32 = conn
            .query_row("SELECT COUNT(*) FROM relationships", [], |row| row.get(0))
            .unwrap();
        assert_eq!(count, 1);

        // Delete source entity
        conn.execute("DELETE FROM entities WHERE id = 1", [])
            .unwrap();

        // Relationship should be cascaded
        let count: i32 = conn
            .query_row("SELECT COUNT(*) FROM relationships", [], |row| row.get(0))
            .unwrap();
        assert_eq!(count, 0);
    }
}
