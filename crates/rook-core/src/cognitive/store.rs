//! FSRS state persistence store.
//!
//! Provides SQLite-backed storage for FSRS memory states,
//! enabling state retrieval, persistence, and archival candidate queries.

use crate::error::{RookError, RookResult};
use crate::types::{ArchivalConfig, FsrsState};
use chrono::{DateTime, Utc};
use rusqlite::{params, Connection, OptionalExtension};
use std::path::Path;
use std::sync::{Arc, Mutex};

/// SQLite-backed store for FSRS cognitive state.
///
/// Stores memory states with their FSRS parameters for scheduling
/// and archival candidate identification.
pub struct CognitiveStore {
    conn: Arc<Mutex<Connection>>,
}

impl CognitiveStore {
    /// Create a new cognitive store with the given database path.
    ///
    /// Creates the database file and schema if it doesn't exist.
    pub fn new<P: AsRef<Path>>(path: P) -> RookResult<Self> {
        let conn = Connection::open(path)?;
        let store = Self {
            conn: Arc::new(Mutex::new(conn)),
        };
        store.init_schema()?;
        Ok(store)
    }

    /// Create an in-memory cognitive store (useful for testing).
    pub fn in_memory() -> RookResult<Self> {
        let conn = Connection::open_in_memory()?;
        let store = Self {
            conn: Arc::new(Mutex::new(conn)),
        };
        store.init_schema()?;
        Ok(store)
    }

    /// Initialize the database schema.
    fn init_schema(&self) -> RookResult<()> {
        let conn = self.conn.lock().map_err(|e| RookError::database(e.to_string()))?;

        conn.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS fsrs_states (
                memory_id TEXT PRIMARY KEY,
                stability REAL NOT NULL,
                difficulty REAL NOT NULL,
                last_review TEXT,
                reps INTEGER NOT NULL DEFAULT 0,
                lapses INTEGER NOT NULL DEFAULT 0,
                is_key INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_fsrs_states_is_key ON fsrs_states(is_key);
            CREATE INDEX IF NOT EXISTS idx_fsrs_states_stability ON fsrs_states(stability);
            CREATE INDEX IF NOT EXISTS idx_fsrs_states_last_review ON fsrs_states(last_review);
            CREATE INDEX IF NOT EXISTS idx_fsrs_states_created_at ON fsrs_states(created_at);
            ",
        )?;

        Ok(())
    }

    /// Get the FSRS state for a memory.
    ///
    /// Returns None if the memory doesn't have a stored state.
    pub fn get_state(&self, memory_id: &str) -> RookResult<Option<(FsrsState, bool, DateTime<Utc>)>> {
        let conn = self.conn.lock().map_err(|e| RookError::database(e.to_string()))?;

        let result = conn
            .query_row(
                "SELECT stability, difficulty, last_review, reps, lapses, is_key, created_at
                 FROM fsrs_states WHERE memory_id = ?1",
                params![memory_id],
                |row| {
                    let stability: f32 = row.get(0)?;
                    let difficulty: f32 = row.get(1)?;
                    let last_review_str: Option<String> = row.get(2)?;
                    let reps: u32 = row.get(3)?;
                    let lapses: u32 = row.get(4)?;
                    let is_key: i32 = row.get(5)?;
                    let created_at_str: String = row.get(6)?;

                    let last_review = last_review_str.and_then(|s| {
                        DateTime::parse_from_rfc3339(&s)
                            .map(|dt| dt.with_timezone(&Utc))
                            .ok()
                    });

                    let created_at = DateTime::parse_from_rfc3339(&created_at_str)
                        .map(|dt| dt.with_timezone(&Utc))
                        .unwrap_or_else(|_| Utc::now());

                    let state = FsrsState {
                        stability,
                        difficulty,
                        last_review,
                        reps,
                        lapses,
                    };

                    Ok((state, is_key != 0, created_at))
                },
            )
            .optional()?;

        Ok(result)
    }

    /// Save the FSRS state for a memory.
    ///
    /// Creates or updates the state for the given memory ID.
    pub fn save_state(
        &self,
        memory_id: &str,
        state: &FsrsState,
        is_key: bool,
        created_at: Option<DateTime<Utc>>,
    ) -> RookResult<()> {
        let conn = self.conn.lock().map_err(|e| RookError::database(e.to_string()))?;

        let now = Utc::now();
        let created_at = created_at.unwrap_or(now);
        let last_review_str = state.last_review.map(|dt| dt.to_rfc3339());

        conn.execute(
            "INSERT OR REPLACE INTO fsrs_states
             (memory_id, stability, difficulty, last_review, reps, lapses, is_key, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7,
                     COALESCE((SELECT created_at FROM fsrs_states WHERE memory_id = ?1), ?8),
                     ?9)",
            params![
                memory_id,
                state.stability,
                state.difficulty,
                last_review_str,
                state.reps,
                state.lapses,
                if is_key { 1 } else { 0 },
                created_at.to_rfc3339(),
                now.to_rfc3339(),
            ],
        )?;

        Ok(())
    }

    /// Delete the FSRS state for a memory.
    pub fn delete_state(&self, memory_id: &str) -> RookResult<bool> {
        let conn = self.conn.lock().map_err(|e| RookError::database(e.to_string()))?;

        let deleted = conn.execute(
            "DELETE FROM fsrs_states WHERE memory_id = ?1",
            params![memory_id],
        )?;

        Ok(deleted > 0)
    }

    /// Get archival candidates based on configuration.
    ///
    /// Returns memory IDs that:
    /// 1. Are NOT key memories (is_key = false)
    /// 2. Are older than min_age_days
    /// 3. Have stability below what would give archive_threshold retrievability
    ///    (Note: actual R check should be done by caller using FsrsScheduler)
    ///
    /// Returns up to archive_limit candidates.
    pub fn get_archival_candidates(
        &self,
        config: &ArchivalConfig,
        now: DateTime<Utc>,
    ) -> RookResult<Vec<ArchivalCandidate>> {
        let conn = self.conn.lock().map_err(|e| RookError::database(e.to_string()))?;

        // Calculate the cutoff date for min_age_days
        let min_age_cutoff = now - chrono::Duration::days(config.min_age_days as i64);

        // Get candidates that are:
        // - Not key memories
        // - Old enough (created before cutoff)
        // - Ordered by stability (lowest first - most likely forgotten)
        let mut stmt = conn.prepare(
            "SELECT memory_id, stability, difficulty, last_review, reps, lapses, created_at
             FROM fsrs_states
             WHERE is_key = 0
               AND created_at <= ?1
             ORDER BY stability ASC, last_review ASC
             LIMIT ?2",
        )?;

        let candidates = stmt
            .query_map(params![min_age_cutoff.to_rfc3339(), config.archive_limit], |row| {
                let memory_id: String = row.get(0)?;
                let stability: f32 = row.get(1)?;
                let difficulty: f32 = row.get(2)?;
                let last_review_str: Option<String> = row.get(3)?;
                let reps: u32 = row.get(4)?;
                let lapses: u32 = row.get(5)?;
                let created_at_str: String = row.get(6)?;

                let last_review = last_review_str.and_then(|s| {
                    DateTime::parse_from_rfc3339(&s)
                        .map(|dt| dt.with_timezone(&Utc))
                        .ok()
                });

                let created_at = DateTime::parse_from_rfc3339(&created_at_str)
                    .map(|dt| dt.with_timezone(&Utc))
                    .unwrap_or(now);

                let state = FsrsState {
                    stability,
                    difficulty,
                    last_review,
                    reps,
                    lapses,
                };

                Ok(ArchivalCandidate {
                    memory_id,
                    state,
                    created_at,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(candidates)
    }

    /// Update is_key flag for a memory.
    pub fn set_key(&self, memory_id: &str, is_key: bool) -> RookResult<bool> {
        let conn = self.conn.lock().map_err(|e| RookError::database(e.to_string()))?;

        let updated = conn.execute(
            "UPDATE fsrs_states SET is_key = ?1, updated_at = ?2 WHERE memory_id = ?3",
            params![if is_key { 1 } else { 0 }, Utc::now().to_rfc3339(), memory_id],
        )?;

        Ok(updated > 0)
    }

    /// Get count of stored states.
    pub fn count(&self) -> RookResult<usize> {
        let conn = self.conn.lock().map_err(|e| RookError::database(e.to_string()))?;

        let count: i64 = conn.query_row("SELECT COUNT(*) FROM fsrs_states", [], |row| row.get(0))?;

        Ok(count as usize)
    }

    /// Get count of key memories.
    pub fn count_key_memories(&self) -> RookResult<usize> {
        let conn = self.conn.lock().map_err(|e| RookError::database(e.to_string()))?;

        let count: i64 =
            conn.query_row("SELECT COUNT(*) FROM fsrs_states WHERE is_key = 1", [], |row| {
                row.get(0)
            })?;

        Ok(count as usize)
    }
}

/// A candidate for archival with its state information.
#[derive(Debug, Clone)]
pub struct ArchivalCandidate {
    /// Memory ID.
    pub memory_id: String,
    /// Current FSRS state.
    pub state: FsrsState,
    /// When the memory was created.
    pub created_at: DateTime<Utc>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;

    fn create_test_state(stability: f32, days_since_review: i64) -> FsrsState {
        FsrsState {
            stability,
            difficulty: 5.0,
            last_review: Some(Utc::now() - Duration::days(days_since_review)),
            reps: 3,
            lapses: 0,
        }
    }

    #[test]
    fn test_store_creation() {
        let store = CognitiveStore::in_memory().unwrap();
        assert_eq!(store.count().unwrap(), 0);
    }

    #[test]
    fn test_save_and_get_state() {
        let store = CognitiveStore::in_memory().unwrap();

        let state = create_test_state(10.0, 5);
        store.save_state("mem1", &state, false, None).unwrap();

        let (retrieved, is_key, _created_at) = store.get_state("mem1").unwrap().unwrap();

        assert!((retrieved.stability - 10.0).abs() < 0.001);
        assert!((retrieved.difficulty - 5.0).abs() < 0.001);
        assert_eq!(retrieved.reps, 3);
        assert!(!is_key);
    }

    #[test]
    fn test_save_key_memory() {
        let store = CognitiveStore::in_memory().unwrap();

        let state = create_test_state(10.0, 5);
        store.save_state("key_mem", &state, true, None).unwrap();

        let (_, is_key, _) = store.get_state("key_mem").unwrap().unwrap();
        assert!(is_key);
    }

    #[test]
    fn test_update_state() {
        let store = CognitiveStore::in_memory().unwrap();

        let state1 = create_test_state(5.0, 3);
        store.save_state("mem1", &state1, false, None).unwrap();

        // Update with new state
        let state2 = create_test_state(10.0, 1);
        store.save_state("mem1", &state2, false, None).unwrap();

        let (retrieved, _, _) = store.get_state("mem1").unwrap().unwrap();
        assert!((retrieved.stability - 10.0).abs() < 0.001);
    }

    #[test]
    fn test_delete_state() {
        let store = CognitiveStore::in_memory().unwrap();

        let state = create_test_state(10.0, 5);
        store.save_state("mem1", &state, false, None).unwrap();

        assert!(store.delete_state("mem1").unwrap());
        assert!(store.get_state("mem1").unwrap().is_none());

        // Deleting non-existent returns false
        assert!(!store.delete_state("nonexistent").unwrap());
    }

    #[test]
    fn test_get_state_not_found() {
        let store = CognitiveStore::in_memory().unwrap();

        let result = store.get_state("nonexistent").unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_archival_candidates_excludes_key_memories() {
        let store = CognitiveStore::in_memory().unwrap();
        let now = Utc::now();
        let old_date = now - Duration::days(60);

        // Create a regular memory and a key memory
        let state = create_test_state(0.5, 30);
        store.save_state("regular", &state, false, Some(old_date)).unwrap();
        store.save_state("key_mem", &state, true, Some(old_date)).unwrap();

        let config = ArchivalConfig {
            archive_threshold: 0.5,
            min_age_days: 30,
            archive_limit: 100,
        };

        let candidates = store.get_archival_candidates(&config, now).unwrap();

        // Only regular memory should be in candidates
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].memory_id, "regular");
    }

    #[test]
    fn test_archival_candidates_respects_min_age() {
        let store = CognitiveStore::in_memory().unwrap();
        let now = Utc::now();

        let state = create_test_state(0.5, 5);

        // Old memory
        let old_date = now - Duration::days(60);
        store.save_state("old", &state, false, Some(old_date)).unwrap();

        // Young memory
        let young_date = now - Duration::days(10);
        store.save_state("young", &state, false, Some(young_date)).unwrap();

        let config = ArchivalConfig {
            archive_threshold: 0.5,
            min_age_days: 30,
            archive_limit: 100,
        };

        let candidates = store.get_archival_candidates(&config, now).unwrap();

        // Only old memory should be in candidates
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].memory_id, "old");
    }

    #[test]
    fn test_archival_candidates_respects_limit() {
        let store = CognitiveStore::in_memory().unwrap();
        let now = Utc::now();
        let old_date = now - Duration::days(60);

        // Create many memories
        for i in 0..10 {
            let state = create_test_state(0.5, 30);
            store
                .save_state(&format!("mem{}", i), &state, false, Some(old_date))
                .unwrap();
        }

        let config = ArchivalConfig {
            archive_threshold: 0.5,
            min_age_days: 30,
            archive_limit: 3, // Only get 3
        };

        let candidates = store.get_archival_candidates(&config, now).unwrap();
        assert_eq!(candidates.len(), 3);
    }

    #[test]
    fn test_archival_candidates_ordered_by_stability() {
        let store = CognitiveStore::in_memory().unwrap();
        let now = Utc::now();
        let old_date = now - Duration::days(60);

        // Create memories with different stabilities
        store
            .save_state("high_s", &create_test_state(10.0, 30), false, Some(old_date))
            .unwrap();
        store
            .save_state("low_s", &create_test_state(0.1, 30), false, Some(old_date))
            .unwrap();
        store
            .save_state("mid_s", &create_test_state(5.0, 30), false, Some(old_date))
            .unwrap();

        let config = ArchivalConfig::default();
        let candidates = store.get_archival_candidates(&config, now).unwrap();

        // Should be ordered by stability ascending
        assert_eq!(candidates[0].memory_id, "low_s");
        assert_eq!(candidates[1].memory_id, "mid_s");
        assert_eq!(candidates[2].memory_id, "high_s");
    }

    #[test]
    fn test_set_key() {
        let store = CognitiveStore::in_memory().unwrap();

        let state = create_test_state(10.0, 5);
        store.save_state("mem1", &state, false, None).unwrap();

        // Promote to key
        assert!(store.set_key("mem1", true).unwrap());

        let (_, is_key, _) = store.get_state("mem1").unwrap().unwrap();
        assert!(is_key);

        // Demote from key
        assert!(store.set_key("mem1", false).unwrap());

        let (_, is_key, _) = store.get_state("mem1").unwrap().unwrap();
        assert!(!is_key);
    }

    #[test]
    fn test_count_methods() {
        let store = CognitiveStore::in_memory().unwrap();

        let state = create_test_state(10.0, 5);
        store.save_state("mem1", &state, false, None).unwrap();
        store.save_state("mem2", &state, true, None).unwrap();
        store.save_state("mem3", &state, true, None).unwrap();

        assert_eq!(store.count().unwrap(), 3);
        assert_eq!(store.count_key_memories().unwrap(), 2);
    }
}
