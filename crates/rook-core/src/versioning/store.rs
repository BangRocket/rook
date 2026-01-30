//! Version storage layer with point-in-time query support.
//!
//! Provides SQLite-backed persistence for memory version snapshots.

use crate::error::RookResult;
use crate::versioning::{FsrsStateSnapshot, MemoryVersion, VersionEventType, VersionSummary};
use chrono::{DateTime, Utc};
use rusqlite::{params, Connection, OptionalExtension};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;
use uuid::Uuid;

/// Trait for memory version storage operations
pub trait VersionStore: Send + Sync {
    /// Store a new version
    fn add_version(&self, version: &MemoryVersion) -> RookResult<()>;

    /// Get memory state at a specific point in time (INT-09)
    fn get_at_time(
        &self,
        memory_id: &str,
        timestamp: DateTime<Utc>,
    ) -> RookResult<Option<MemoryVersion>>;

    /// Get a specific version by number
    fn get_version(&self, memory_id: &str, version_number: u32)
        -> RookResult<Option<MemoryVersion>>;

    /// Get the latest version of a memory
    fn get_latest(&self, memory_id: &str) -> RookResult<Option<MemoryVersion>>;

    /// Get all versions of a memory (ordered by version number)
    fn get_all_versions(&self, memory_id: &str) -> RookResult<Vec<MemoryVersion>>;

    /// Get versions within a time range
    fn get_versions_in_range(
        &self,
        memory_id: &str,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> RookResult<Vec<MemoryVersion>>;

    /// Get version summary for a memory
    fn get_summary(&self, memory_id: &str) -> RookResult<Option<VersionSummary>>;

    /// Get the next version number for a memory
    fn get_next_version_number(&self, memory_id: &str) -> RookResult<u32>;

    /// Delete all versions for a memory
    fn delete_versions(&self, memory_id: &str) -> RookResult<usize>;

    /// Prune old versions (keep last N versions per memory)
    fn prune_old_versions(&self, memory_id: &str, keep_count: usize) -> RookResult<usize>;

    /// Count total versions in store
    fn count_all(&self) -> RookResult<usize>;
}

/// SQLite-backed version store
pub struct SqliteVersionStore {
    conn: Mutex<Connection>,
}

impl SqliteVersionStore {
    /// Create a new store at the given path
    pub fn new(path: impl AsRef<Path>) -> RookResult<Self> {
        let conn = Connection::open(path)?;
        let store = Self {
            conn: Mutex::new(conn),
        };
        store.init_schema()?;
        Ok(store)
    }

    /// Create an in-memory store (for testing)
    pub fn in_memory() -> RookResult<Self> {
        let conn = Connection::open_in_memory()?;
        let store = Self {
            conn: Mutex::new(conn),
        };
        store.init_schema()?;
        Ok(store)
    }

    fn init_schema(&self) -> RookResult<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS memory_versions (
                version_id TEXT PRIMARY KEY,
                memory_id TEXT NOT NULL,
                version_number INTEGER NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT,
                fsrs_state TEXT,
                created_at TEXT NOT NULL,
                event_type TEXT NOT NULL,
                change_description TEXT,
                changed_by TEXT,
                UNIQUE(memory_id, version_number)
            );

            -- Index for point-in-time queries (INT-09)
            CREATE INDEX IF NOT EXISTS idx_versions_memory_time
                ON memory_versions(memory_id, created_at);

            -- Index for getting latest version
            CREATE INDEX IF NOT EXISTS idx_versions_memory_num
                ON memory_versions(memory_id, version_number DESC);

            -- Index for version lookups
            CREATE INDEX IF NOT EXISTS idx_versions_id
                ON memory_versions(version_id);
        "#,
        )?;
        Ok(())
    }

    fn serialize_metadata(metadata: &HashMap<String, serde_json::Value>) -> RookResult<String> {
        Ok(serde_json::to_string(metadata)?)
    }

    fn deserialize_metadata(data: &str) -> RookResult<HashMap<String, serde_json::Value>> {
        Ok(serde_json::from_str(data)?)
    }

    fn serialize_fsrs(fsrs: &Option<FsrsStateSnapshot>) -> RookResult<Option<String>> {
        match fsrs {
            Some(state) => Ok(Some(serde_json::to_string(state)?)),
            None => Ok(None),
        }
    }

    fn deserialize_fsrs(data: &Option<String>) -> RookResult<Option<FsrsStateSnapshot>> {
        match data {
            Some(s) => Ok(Some(serde_json::from_str(s)?)),
            None => Ok(None),
        }
    }

    fn row_to_version(row: &rusqlite::Row<'_>) -> RookResult<MemoryVersion> {
        let version_id: String = row.get(0)?;
        let memory_id: String = row.get(1)?;
        let version_number: u32 = row.get(2)?;
        let content: String = row.get(3)?;
        let metadata: String = row.get(4)?;
        let fsrs_state: Option<String> = row.get(5)?;
        let created_at: String = row.get(6)?;
        let event_type: String = row.get(7)?;
        let change_description: Option<String> = row.get(8)?;
        let changed_by: Option<String> = row.get(9)?;

        Ok(MemoryVersion {
            version_id: Uuid::parse_str(&version_id)
                .map_err(|e| crate::error::RookError::parse(e.to_string()))?,
            memory_id,
            version_number,
            content,
            metadata: Self::deserialize_metadata(&metadata)?,
            fsrs_state: Self::deserialize_fsrs(&fsrs_state)?,
            created_at: DateTime::parse_from_rfc3339(&created_at)
                .map(|dt| dt.with_timezone(&Utc))
                .map_err(|e| crate::error::RookError::parse(e.to_string()))?,
            event_type: VersionEventType::from_str(&event_type)
                .unwrap_or(VersionEventType::ContentUpdated),
            change_description,
            changed_by,
        })
    }
}

impl VersionStore for SqliteVersionStore {
    fn add_version(&self, version: &MemoryVersion) -> RookResult<()> {
        let conn = self.conn.lock().unwrap();
        let metadata = Self::serialize_metadata(&version.metadata)?;
        let fsrs_state = Self::serialize_fsrs(&version.fsrs_state)?;

        conn.execute(
            r#"INSERT INTO memory_versions
               (version_id, memory_id, version_number, content, metadata, fsrs_state,
                created_at, event_type, change_description, changed_by)
               VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)"#,
            params![
                version.version_id.to_string(),
                version.memory_id,
                version.version_number,
                version.content,
                metadata,
                fsrs_state,
                version.created_at.to_rfc3339(),
                version.event_type.as_str(),
                version.change_description,
                version.changed_by,
            ],
        )?;
        Ok(())
    }

    fn get_at_time(
        &self,
        memory_id: &str,
        timestamp: DateTime<Utc>,
    ) -> RookResult<Option<MemoryVersion>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            r#"SELECT version_id, memory_id, version_number, content, metadata, fsrs_state,
                      created_at, event_type, change_description, changed_by
               FROM memory_versions
               WHERE memory_id = ?1 AND created_at <= ?2
               ORDER BY version_number DESC
               LIMIT 1"#,
        )?;

        stmt.query_row(params![memory_id, timestamp.to_rfc3339()], |row| {
            Ok(Self::row_to_version(row))
        })
        .optional()?
        .transpose()
    }

    fn get_version(
        &self,
        memory_id: &str,
        version_number: u32,
    ) -> RookResult<Option<MemoryVersion>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            r#"SELECT version_id, memory_id, version_number, content, metadata, fsrs_state,
                      created_at, event_type, change_description, changed_by
               FROM memory_versions
               WHERE memory_id = ?1 AND version_number = ?2"#,
        )?;

        stmt.query_row(params![memory_id, version_number], |row| {
            Ok(Self::row_to_version(row))
        })
        .optional()?
        .transpose()
    }

    fn get_latest(&self, memory_id: &str) -> RookResult<Option<MemoryVersion>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            r#"SELECT version_id, memory_id, version_number, content, metadata, fsrs_state,
                      created_at, event_type, change_description, changed_by
               FROM memory_versions
               WHERE memory_id = ?1
               ORDER BY version_number DESC
               LIMIT 1"#,
        )?;

        stmt.query_row(params![memory_id], |row| Ok(Self::row_to_version(row)))
            .optional()?
            .transpose()
    }

    fn get_all_versions(&self, memory_id: &str) -> RookResult<Vec<MemoryVersion>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            r#"SELECT version_id, memory_id, version_number, content, metadata, fsrs_state,
                      created_at, event_type, change_description, changed_by
               FROM memory_versions
               WHERE memory_id = ?1
               ORDER BY version_number ASC"#,
        )?;

        let results = stmt.query_map(params![memory_id], |row| Ok(Self::row_to_version(row)))?;

        results
            .map(|r| r.map_err(|e| e.into()).and_then(|inner| inner))
            .collect()
    }

    fn get_versions_in_range(
        &self,
        memory_id: &str,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> RookResult<Vec<MemoryVersion>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            r#"SELECT version_id, memory_id, version_number, content, metadata, fsrs_state,
                      created_at, event_type, change_description, changed_by
               FROM memory_versions
               WHERE memory_id = ?1 AND created_at >= ?2 AND created_at <= ?3
               ORDER BY version_number ASC"#,
        )?;

        let results = stmt.query_map(
            params![memory_id, start.to_rfc3339(), end.to_rfc3339()],
            |row| Ok(Self::row_to_version(row)),
        )?;

        results
            .map(|r| r.map_err(|e| e.into()).and_then(|inner| inner))
            .collect()
    }

    fn get_summary(&self, memory_id: &str) -> RookResult<Option<VersionSummary>> {
        let conn = self.conn.lock().unwrap();

        let mut stmt = conn.prepare(
            r#"SELECT
                COUNT(*) as total,
                MAX(version_number) as latest,
                MIN(created_at) as first,
                MAX(created_at) as last,
                SUM(CASE WHEN event_type = 'content_updated' THEN 1 ELSE 0 END) as content_updates,
                SUM(CASE WHEN event_type = 'metadata_updated' THEN 1 ELSE 0 END) as metadata_updates,
                SUM(CASE WHEN event_type = 'fsrs_updated' THEN 1 ELSE 0 END) as fsrs_updates
               FROM memory_versions
               WHERE memory_id = ?1"#,
        )?;

        let result = stmt
            .query_row(params![memory_id], |row| {
                let total: u32 = row.get(0)?;
                if total == 0 {
                    return Ok(None);
                }

                let latest: u32 = row.get(1)?;
                let first: String = row.get(2)?;
                let last: String = row.get(3)?;
                let content_updates: u32 = row.get(4)?;
                let metadata_updates: u32 = row.get(5)?;
                let fsrs_updates: u32 = row.get(6)?;

                Ok(Some(VersionSummary {
                    memory_id: memory_id.to_string(),
                    total_versions: total,
                    latest_version: latest,
                    first_created: DateTime::parse_from_rfc3339(&first)
                        .map(|dt| dt.with_timezone(&Utc))
                        .unwrap_or_else(|_| Utc::now()),
                    last_modified: DateTime::parse_from_rfc3339(&last)
                        .map(|dt| dt.with_timezone(&Utc))
                        .unwrap_or_else(|_| Utc::now()),
                    content_updates,
                    metadata_updates,
                    fsrs_updates,
                }))
            })
            .optional()?
            .flatten();

        Ok(result)
    }

    fn get_next_version_number(&self, memory_id: &str) -> RookResult<u32> {
        let conn = self.conn.lock().unwrap();
        let max: Option<u32> = conn.query_row(
            "SELECT MAX(version_number) FROM memory_versions WHERE memory_id = ?1",
            params![memory_id],
            |row| row.get(0),
        )?;
        Ok(max.unwrap_or(0) + 1)
    }

    fn delete_versions(&self, memory_id: &str) -> RookResult<usize> {
        let conn = self.conn.lock().unwrap();
        let count = conn.execute(
            "DELETE FROM memory_versions WHERE memory_id = ?1",
            params![memory_id],
        )?;
        Ok(count)
    }

    fn prune_old_versions(&self, memory_id: &str, keep_count: usize) -> RookResult<usize> {
        let conn = self.conn.lock().unwrap();

        // Delete versions older than the Nth most recent
        let count = conn.execute(
            r#"DELETE FROM memory_versions
               WHERE memory_id = ?1
               AND version_number NOT IN (
                   SELECT version_number
                   FROM memory_versions
                   WHERE memory_id = ?1
                   ORDER BY version_number DESC
                   LIMIT ?2
               )"#,
            params![memory_id, keep_count as i64],
        )?;
        Ok(count)
    }

    fn count_all(&self) -> RookResult<usize> {
        let conn = self.conn.lock().unwrap();
        let count: i64 = conn.query_row("SELECT COUNT(*) FROM memory_versions", [], |row| {
            row.get(0)
        })?;
        Ok(count as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_store_crud() {
        let store = SqliteVersionStore::in_memory().unwrap();

        // Create initial version
        let v1 = MemoryVersion::initial("mem-1", "Hello world");
        store.add_version(&v1).unwrap();

        // Get latest
        let latest = store.get_latest("mem-1").unwrap().unwrap();
        assert_eq!(latest.content, "Hello world");
        assert_eq!(latest.version_number, 1);

        // Add content update
        let v2 = MemoryVersion::from_content_update(&v1, "Hello updated world");
        store.add_version(&v2).unwrap();

        // Get latest should be v2
        let latest = store.get_latest("mem-1").unwrap().unwrap();
        assert_eq!(latest.content, "Hello updated world");
        assert_eq!(latest.version_number, 2);

        // Get all versions
        let all = store.get_all_versions("mem-1").unwrap();
        assert_eq!(all.len(), 2);
        assert_eq!(all[0].version_number, 1);
        assert_eq!(all[1].version_number, 2);
    }

    #[test]
    fn test_point_in_time_query() {
        let store = SqliteVersionStore::in_memory().unwrap();

        // Create versions at different times
        let mut v1 = MemoryVersion::initial("mem-1", "Version 1");
        v1.created_at = Utc::now() - chrono::Duration::days(2);
        store.add_version(&v1).unwrap();

        let mut v2 = MemoryVersion::from_content_update(&v1, "Version 2");
        v2.created_at = Utc::now() - chrono::Duration::days(1);
        store.add_version(&v2).unwrap();

        let mut v3 = MemoryVersion::from_content_update(&v2, "Version 3");
        v3.created_at = Utc::now();
        store.add_version(&v3).unwrap();

        // Query at different points
        let at_day_ago = store
            .get_at_time("mem-1", Utc::now() - chrono::Duration::hours(36))
            .unwrap()
            .unwrap();
        assert_eq!(at_day_ago.content, "Version 1");

        let at_12h_ago = store
            .get_at_time("mem-1", Utc::now() - chrono::Duration::hours(12))
            .unwrap()
            .unwrap();
        assert_eq!(at_12h_ago.content, "Version 2");

        let now = store.get_at_time("mem-1", Utc::now()).unwrap().unwrap();
        assert_eq!(now.content, "Version 3");
    }

    #[test]
    fn test_version_summary() {
        let store = SqliteVersionStore::in_memory().unwrap();

        let v1 = MemoryVersion::initial("mem-1", "Content");
        store.add_version(&v1).unwrap();

        let v2 = MemoryVersion::from_content_update(&v1, "Updated");
        store.add_version(&v2).unwrap();

        let summary = store.get_summary("mem-1").unwrap().unwrap();
        assert_eq!(summary.total_versions, 2);
        assert_eq!(summary.latest_version, 2);
        assert_eq!(summary.content_updates, 1);
    }

    #[test]
    fn test_prune_old_versions() {
        let store = SqliteVersionStore::in_memory().unwrap();

        // Create 5 versions
        let mut prev = MemoryVersion::initial("mem-1", "V1");
        store.add_version(&prev).unwrap();

        for i in 2..=5 {
            let next = MemoryVersion::from_content_update(&prev, format!("V{}", i));
            store.add_version(&next).unwrap();
            prev = next;
        }

        assert_eq!(store.get_all_versions("mem-1").unwrap().len(), 5);

        // Prune to keep only 2 most recent
        let pruned = store.prune_old_versions("mem-1", 2).unwrap();
        assert_eq!(pruned, 3);

        let remaining = store.get_all_versions("mem-1").unwrap();
        assert_eq!(remaining.len(), 2);
        assert_eq!(remaining[0].version_number, 4);
        assert_eq!(remaining[1].version_number, 5);
    }

    #[test]
    fn test_get_version_by_number() {
        let store = SqliteVersionStore::in_memory().unwrap();

        let v1 = MemoryVersion::initial("mem-1", "Version 1");
        store.add_version(&v1).unwrap();

        let v2 = MemoryVersion::from_content_update(&v1, "Version 2");
        store.add_version(&v2).unwrap();

        let retrieved = store.get_version("mem-1", 1).unwrap().unwrap();
        assert_eq!(retrieved.content, "Version 1");

        let retrieved = store.get_version("mem-1", 2).unwrap().unwrap();
        assert_eq!(retrieved.content, "Version 2");

        assert!(store.get_version("mem-1", 3).unwrap().is_none());
    }

    #[test]
    fn test_delete_versions() {
        let store = SqliteVersionStore::in_memory().unwrap();

        let v1 = MemoryVersion::initial("mem-1", "Content");
        store.add_version(&v1).unwrap();

        let v2 = MemoryVersion::from_content_update(&v1, "Updated");
        store.add_version(&v2).unwrap();

        assert_eq!(store.count_all().unwrap(), 2);

        let deleted = store.delete_versions("mem-1").unwrap();
        assert_eq!(deleted, 2);

        assert_eq!(store.count_all().unwrap(), 0);
    }

    #[test]
    fn test_versions_in_range() {
        let store = SqliteVersionStore::in_memory().unwrap();

        let mut v1 = MemoryVersion::initial("mem-1", "V1");
        v1.created_at = Utc::now() - chrono::Duration::hours(3);
        store.add_version(&v1).unwrap();

        let mut v2 = MemoryVersion::from_content_update(&v1, "V2");
        v2.created_at = Utc::now() - chrono::Duration::hours(2);
        store.add_version(&v2).unwrap();

        let mut v3 = MemoryVersion::from_content_update(&v2, "V3");
        v3.created_at = Utc::now() - chrono::Duration::hours(1);
        store.add_version(&v3).unwrap();

        let start = Utc::now() - chrono::Duration::hours(2) - chrono::Duration::minutes(30);
        let end = Utc::now() - chrono::Duration::minutes(30);
        let in_range = store.get_versions_in_range("mem-1", start, end).unwrap();

        assert_eq!(in_range.len(), 2);
        assert_eq!(in_range[0].content, "V2");
        assert_eq!(in_range[1].content, "V3");
    }

    #[test]
    fn test_get_next_version_number() {
        let store = SqliteVersionStore::in_memory().unwrap();

        // No versions yet
        assert_eq!(store.get_next_version_number("mem-1").unwrap(), 1);

        let v1 = MemoryVersion::initial("mem-1", "Content");
        store.add_version(&v1).unwrap();

        assert_eq!(store.get_next_version_number("mem-1").unwrap(), 2);

        let v2 = MemoryVersion::from_content_update(&v1, "Updated");
        store.add_version(&v2).unwrap();

        assert_eq!(store.get_next_version_number("mem-1").unwrap(), 3);
    }
}
