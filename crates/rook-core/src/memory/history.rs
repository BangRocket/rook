//! History tracking using SQLite.

use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::{Arc, Mutex};
use uuid::Uuid;

use crate::error::{RookError, RookResult};

/// Event type for history records.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum HistoryEvent {
    Add,
    Update,
    Delete,
}

impl HistoryEvent {
    fn as_str(&self) -> &'static str {
        match self {
            HistoryEvent::Add => "ADD",
            HistoryEvent::Update => "UPDATE",
            HistoryEvent::Delete => "DELETE",
        }
    }
}

/// A history record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryRecord {
    /// Record ID.
    pub id: String,
    /// Memory ID this record refers to.
    pub memory_id: String,
    /// Previous memory content.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub old_memory: Option<String>,
    /// New memory content.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub new_memory: Option<String>,
    /// Event type.
    pub event: String,
    /// Creation timestamp.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub created_at: Option<String>,
    /// Update timestamp.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub updated_at: Option<String>,
    /// Whether this memory is deleted.
    pub is_deleted: bool,
    /// Actor who made the change.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub actor_id: Option<String>,
    /// Role of the message that created this.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
}

/// SQLite-based history store.
pub struct HistoryStore {
    conn: Arc<Mutex<Connection>>,
}

impl HistoryStore {
    /// Create a new history store.
    pub fn new(db_path: impl AsRef<Path>) -> RookResult<Self> {
        // Ensure parent directory exists
        if let Some(parent) = db_path.as_ref().parent() {
            std::fs::create_dir_all(parent)?;
        }

        let conn = if db_path.as_ref().to_str() == Some(":memory:") {
            Connection::open_in_memory()
        } else {
            Connection::open(db_path.as_ref())
        }
        .map_err(|e| RookError::database(e.to_string()))?;

        let store = Self {
            conn: Arc::new(Mutex::new(conn)),
        };

        store.create_table()?;

        Ok(store)
    }

    /// Create the history table if it doesn't exist.
    fn create_table(&self) -> RookResult<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            r#"
            CREATE TABLE IF NOT EXISTS history (
                id           TEXT PRIMARY KEY,
                memory_id    TEXT,
                old_memory   TEXT,
                new_memory   TEXT,
                event        TEXT,
                created_at   DATETIME,
                updated_at   DATETIME,
                is_deleted   INTEGER DEFAULT 0,
                actor_id     TEXT,
                role         TEXT
            )
            "#,
            [],
        )
        .map_err(|e| RookError::database(e.to_string()))?;

        // Create index on memory_id
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_history_memory_id ON history(memory_id)",
            [],
        )
        .map_err(|e| RookError::database(e.to_string()))?;

        Ok(())
    }

    /// Add a history record.
    pub fn add(
        &self,
        memory_id: &str,
        old_memory: Option<&str>,
        new_memory: Option<&str>,
        event: HistoryEvent,
        created_at: Option<&str>,
        updated_at: Option<&str>,
        actor_id: Option<&str>,
        role: Option<&str>,
    ) -> RookResult<String> {
        let conn = self.conn.lock().unwrap();
        let id = Uuid::new_v4().to_string();
        let is_deleted = matches!(event, HistoryEvent::Delete);

        conn.execute(
            r#"
            INSERT INTO history (
                id, memory_id, old_memory, new_memory, event,
                created_at, updated_at, is_deleted, actor_id, role
            )
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
            "#,
            params![
                id,
                memory_id,
                old_memory,
                new_memory,
                event.as_str(),
                created_at,
                updated_at,
                is_deleted as i32,
                actor_id,
                role,
            ],
        )
        .map_err(|e| RookError::database(e.to_string()))?;

        Ok(id)
    }

    /// Get history for a memory.
    pub fn get(&self, memory_id: &str) -> RookResult<Vec<HistoryRecord>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn
            .prepare(
                r#"
            SELECT id, memory_id, old_memory, new_memory, event,
                   created_at, updated_at, is_deleted, actor_id, role
            FROM history
            WHERE memory_id = ?1
            ORDER BY created_at ASC, updated_at ASC
            "#,
            )
            .map_err(|e| RookError::database(e.to_string()))?;

        let records = stmt
            .query_map([memory_id], |row| {
                Ok(HistoryRecord {
                    id: row.get(0)?,
                    memory_id: row.get(1)?,
                    old_memory: row.get(2)?,
                    new_memory: row.get(3)?,
                    event: row.get(4)?,
                    created_at: row.get(5)?,
                    updated_at: row.get(6)?,
                    is_deleted: row.get::<_, i32>(7)? != 0,
                    actor_id: row.get(8)?,
                    role: row.get(9)?,
                })
            })
            .map_err(|e| RookError::database(e.to_string()))?;

        records
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| RookError::database(e.to_string()))
    }

    /// Reset (clear) all history.
    pub fn reset(&self) -> RookResult<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute("DELETE FROM history", [])
            .map_err(|e| RookError::database(e.to_string()))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_history_store() {
        let store = HistoryStore::new(":memory:").unwrap();

        // Add a record
        let id = store
            .add(
                "mem1",
                None,
                Some("new memory"),
                HistoryEvent::Add,
                Some("2024-01-01T00:00:00Z"),
                None,
                Some("user1"),
                Some("user"),
            )
            .unwrap();

        assert!(!id.is_empty());

        // Get history
        let history = store.get("mem1").unwrap();
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].event, "ADD");
        assert_eq!(history[0].new_memory, Some("new memory".to_string()));
    }

    #[test]
    fn test_history_reset() {
        let store = HistoryStore::new(":memory:").unwrap();

        store
            .add("mem1", None, Some("test"), HistoryEvent::Add, None, None, None, None)
            .unwrap();

        let history = store.get("mem1").unwrap();
        assert_eq!(history.len(), 1);

        store.reset().unwrap();

        let history = store.get("mem1").unwrap();
        assert!(history.is_empty());
    }
}
