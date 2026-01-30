//! Intention storage trait and implementations.
//!
//! Provides persistence for intentions with query capabilities.

use crate::error::RookResult;
use crate::intentions::{FiredIntention, Intention, TriggerCondition};
use chrono::{DateTime, Utc};
use rusqlite::{params, Connection, OptionalExtension};
use std::path::Path;
use std::sync::Mutex;
use uuid::Uuid;

/// Trait for intention storage operations
pub trait IntentionStore: Send + Sync {
    /// Add a new intention
    fn add(&self, intention: &Intention) -> RookResult<()>;

    /// Get intention by ID
    fn get(&self, id: Uuid) -> RookResult<Option<Intention>>;

    /// Update an existing intention
    fn update(&self, intention: &Intention) -> RookResult<()>;

    /// Delete an intention
    fn delete(&self, id: Uuid) -> RookResult<()>;

    /// Get all active intentions
    fn get_active(&self) -> RookResult<Vec<Intention>>;

    /// Get intentions by trigger type
    fn get_by_trigger_type(&self, trigger_type: &str) -> RookResult<Vec<Intention>>;

    /// Get intentions for a specific user
    fn get_for_user(&self, user_id: &str) -> RookResult<Vec<Intention>>;

    /// Get intentions associated with a memory
    fn get_for_memory(&self, memory_id: &str) -> RookResult<Vec<Intention>>;

    /// Record that an intention fired
    fn record_fire(&self, intention_id: Uuid, fired: &FiredIntention) -> RookResult<()>;

    /// Get fire history for an intention
    fn get_fire_history(&self, intention_id: Uuid, limit: usize) -> RookResult<Vec<FiredIntention>>;

    /// Clean up expired intentions
    fn cleanup_expired(&self) -> RookResult<usize>;
}

/// SQLite-backed intention store
pub struct SqliteIntentionStore {
    conn: Mutex<Connection>,
}

impl SqliteIntentionStore {
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
            CREATE TABLE IF NOT EXISTS intentions (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                memory_id TEXT,
                user_id TEXT,
                trigger_type TEXT NOT NULL,
                trigger_data TEXT NOT NULL,
                action_type TEXT NOT NULL,
                action_data TEXT NOT NULL,
                expires_at TEXT,
                active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                last_fired_at TEXT,
                fire_count INTEGER NOT NULL DEFAULT 0,
                max_fires INTEGER,
                metadata TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_intentions_active ON intentions(active);
            CREATE INDEX IF NOT EXISTS idx_intentions_user ON intentions(user_id);
            CREATE INDEX IF NOT EXISTS idx_intentions_memory ON intentions(memory_id);
            CREATE INDEX IF NOT EXISTS idx_intentions_trigger ON intentions(trigger_type);
            CREATE INDEX IF NOT EXISTS idx_intentions_expires ON intentions(expires_at);

            CREATE TABLE IF NOT EXISTS intention_fires (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                intention_id TEXT NOT NULL,
                fired_at TEXT NOT NULL,
                reason_type TEXT NOT NULL,
                reason_data TEXT NOT NULL,
                action_result TEXT NOT NULL,
                FOREIGN KEY (intention_id) REFERENCES intentions(id)
            );

            CREATE INDEX IF NOT EXISTS idx_fires_intention ON intention_fires(intention_id);
            CREATE INDEX IF NOT EXISTS idx_fires_time ON intention_fires(fired_at);
        "#,
        )?;
        Ok(())
    }

    fn trigger_type_name(trigger: &TriggerCondition) -> &'static str {
        match trigger {
            TriggerCondition::KeywordMention { .. } => "keyword_mention",
            TriggerCondition::TopicDiscussed { .. } => "topic_discussed",
            TriggerCondition::TimeElapsed { .. } => "time_elapsed",
            TriggerCondition::ScheduledTime { .. } => "scheduled_time",
        }
    }

    fn serialize_trigger(trigger: &TriggerCondition) -> RookResult<String> {
        Ok(serde_json::to_string(trigger)?)
    }

    fn deserialize_trigger(data: &str) -> RookResult<TriggerCondition> {
        Ok(serde_json::from_str(data)?)
    }

    fn serialize_action(action: &crate::intentions::IntentionAction) -> RookResult<String> {
        Ok(serde_json::to_string(action)?)
    }

    fn deserialize_action(data: &str) -> RookResult<crate::intentions::IntentionAction> {
        Ok(serde_json::from_str(data)?)
    }

    fn action_type_name(action: &crate::intentions::IntentionAction) -> &'static str {
        match action {
            crate::intentions::IntentionAction::SurfaceMemory { .. } => "surface_memory",
            crate::intentions::IntentionAction::Notify { .. } => "notify",
            crate::intentions::IntentionAction::Callback { .. } => "callback",
            crate::intentions::IntentionAction::Log { .. } => "log",
        }
    }

    fn row_to_intention(row: &rusqlite::Row<'_>) -> RookResult<Intention> {
        let id: String = row.get(0)?;
        let name: String = row.get(1)?;
        let memory_id: Option<String> = row.get(2)?;
        let user_id: Option<String> = row.get(3)?;
        let trigger_data: String = row.get(4)?;
        let action_data: String = row.get(5)?;
        let expires_at: Option<String> = row.get(6)?;
        let active: i32 = row.get(7)?;
        let created_at: String = row.get(8)?;
        let last_fired_at: Option<String> = row.get(9)?;
        let fire_count: u32 = row.get(10)?;
        let max_fires: Option<u32> = row.get(11)?;
        let metadata: String = row.get(12)?;

        Ok(Intention {
            id: Uuid::parse_str(&id).map_err(|e| crate::error::RookError::parse(e.to_string()))?,
            name,
            memory_id,
            user_id,
            trigger: Self::deserialize_trigger(&trigger_data)?,
            action: Self::deserialize_action(&action_data)?,
            expires_at: expires_at
                .map(|s| {
                    DateTime::parse_from_rfc3339(&s)
                        .map(|dt| dt.with_timezone(&Utc))
                        .map_err(|e| crate::error::RookError::parse(e.to_string()))
                })
                .transpose()?,
            active: active != 0,
            created_at: DateTime::parse_from_rfc3339(&created_at)
                .map(|dt| dt.with_timezone(&Utc))
                .map_err(|e| crate::error::RookError::parse(e.to_string()))?,
            last_fired_at: last_fired_at
                .map(|s| {
                    DateTime::parse_from_rfc3339(&s)
                        .map(|dt| dt.with_timezone(&Utc))
                        .map_err(|e| crate::error::RookError::parse(e.to_string()))
                })
                .transpose()?,
            fire_count,
            max_fires,
            metadata: serde_json::from_str(&metadata)?,
        })
    }
}

impl IntentionStore for SqliteIntentionStore {
    fn add(&self, intention: &Intention) -> RookResult<()> {
        let conn = self.conn.lock().unwrap();
        let trigger_type = Self::trigger_type_name(&intention.trigger);
        let trigger_data = Self::serialize_trigger(&intention.trigger)?;
        let action_type = Self::action_type_name(&intention.action);
        let action_data = Self::serialize_action(&intention.action)?;
        let metadata = serde_json::to_string(&intention.metadata)?;

        conn.execute(
            r#"INSERT INTO intentions
               (id, name, memory_id, user_id, trigger_type, trigger_data, action_type, action_data,
                expires_at, active, created_at, last_fired_at, fire_count, max_fires, metadata)
               VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15)"#,
            params![
                intention.id.to_string(),
                intention.name,
                intention.memory_id,
                intention.user_id,
                trigger_type,
                trigger_data,
                action_type,
                action_data,
                intention.expires_at.map(|dt| dt.to_rfc3339()),
                intention.active as i32,
                intention.created_at.to_rfc3339(),
                intention.last_fired_at.map(|dt| dt.to_rfc3339()),
                intention.fire_count,
                intention.max_fires,
                metadata,
            ],
        )?;
        Ok(())
    }

    fn get(&self, id: Uuid) -> RookResult<Option<Intention>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            r#"SELECT id, name, memory_id, user_id, trigger_data, action_data,
                      expires_at, active, created_at, last_fired_at, fire_count, max_fires, metadata
               FROM intentions WHERE id = ?1"#,
        )?;

        stmt.query_row(params![id.to_string()], |row| Ok(Self::row_to_intention(row)))
            .optional()?
            .transpose()
    }

    fn update(&self, intention: &Intention) -> RookResult<()> {
        let conn = self.conn.lock().unwrap();
        let trigger_type = Self::trigger_type_name(&intention.trigger);
        let trigger_data = Self::serialize_trigger(&intention.trigger)?;
        let action_type = Self::action_type_name(&intention.action);
        let action_data = Self::serialize_action(&intention.action)?;
        let metadata = serde_json::to_string(&intention.metadata)?;

        conn.execute(
            r#"UPDATE intentions SET
               name = ?2, memory_id = ?3, user_id = ?4, trigger_type = ?5, trigger_data = ?6,
               action_type = ?7, action_data = ?8, expires_at = ?9, active = ?10,
               last_fired_at = ?11, fire_count = ?12, max_fires = ?13, metadata = ?14
               WHERE id = ?1"#,
            params![
                intention.id.to_string(),
                intention.name,
                intention.memory_id,
                intention.user_id,
                trigger_type,
                trigger_data,
                action_type,
                action_data,
                intention.expires_at.map(|dt| dt.to_rfc3339()),
                intention.active as i32,
                intention.last_fired_at.map(|dt| dt.to_rfc3339()),
                intention.fire_count,
                intention.max_fires,
                metadata,
            ],
        )?;
        Ok(())
    }

    fn delete(&self, id: Uuid) -> RookResult<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "DELETE FROM intentions WHERE id = ?1",
            params![id.to_string()],
        )?;
        Ok(())
    }

    fn get_active(&self) -> RookResult<Vec<Intention>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            r#"SELECT id, name, memory_id, user_id, trigger_data, action_data,
                      expires_at, active, created_at, last_fired_at, fire_count, max_fires, metadata
               FROM intentions
               WHERE active = 1 AND (expires_at IS NULL OR expires_at > ?1)"#,
        )?;

        let now = Utc::now().to_rfc3339();
        let results = stmt.query_map(params![now], |row| Ok(Self::row_to_intention(row)))?;

        results
            .map(|r| r.map_err(|e| e.into()).and_then(|inner| inner))
            .collect()
    }

    fn get_by_trigger_type(&self, trigger_type: &str) -> RookResult<Vec<Intention>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            r#"SELECT id, name, memory_id, user_id, trigger_data, action_data,
                      expires_at, active, created_at, last_fired_at, fire_count, max_fires, metadata
               FROM intentions WHERE trigger_type = ?1 AND active = 1"#,
        )?;

        let results = stmt.query_map(params![trigger_type], |row| {
            Ok(Self::row_to_intention(row))
        })?;

        results
            .map(|r| r.map_err(|e| e.into()).and_then(|inner| inner))
            .collect()
    }

    fn get_for_user(&self, user_id: &str) -> RookResult<Vec<Intention>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            r#"SELECT id, name, memory_id, user_id, trigger_data, action_data,
                      expires_at, active, created_at, last_fired_at, fire_count, max_fires, metadata
               FROM intentions WHERE user_id = ?1"#,
        )?;

        let results = stmt.query_map(params![user_id], |row| Ok(Self::row_to_intention(row)))?;

        results
            .map(|r| r.map_err(|e| e.into()).and_then(|inner| inner))
            .collect()
    }

    fn get_for_memory(&self, memory_id: &str) -> RookResult<Vec<Intention>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            r#"SELECT id, name, memory_id, user_id, trigger_data, action_data,
                      expires_at, active, created_at, last_fired_at, fire_count, max_fires, metadata
               FROM intentions WHERE memory_id = ?1"#,
        )?;

        let results = stmt.query_map(params![memory_id], |row| Ok(Self::row_to_intention(row)))?;

        results
            .map(|r| r.map_err(|e| e.into()).and_then(|inner| inner))
            .collect()
    }

    fn record_fire(&self, intention_id: Uuid, fired: &FiredIntention) -> RookResult<()> {
        let conn = self.conn.lock().unwrap();

        // Insert fire record
        let reason_type = match &fired.reason {
            crate::intentions::TriggerReason::Keyword { .. } => "keyword",
            crate::intentions::TriggerReason::Topic { .. } => "topic",
            crate::intentions::TriggerReason::TimeElapsed { .. } => "time_elapsed",
            crate::intentions::TriggerReason::ScheduledTime { .. } => "scheduled_time",
        };
        let reason_data = serde_json::to_string(&fired.reason)?;
        let action_result = serde_json::to_string(&fired.action_result)?;

        conn.execute(
            r#"INSERT INTO intention_fires (intention_id, fired_at, reason_type, reason_data, action_result)
               VALUES (?1, ?2, ?3, ?4, ?5)"#,
            params![
                intention_id.to_string(),
                fired.fired_at.to_rfc3339(),
                reason_type,
                reason_data,
                action_result,
            ],
        )?;

        // Update intention's fire count and last_fired_at
        conn.execute(
            r#"UPDATE intentions SET fire_count = fire_count + 1, last_fired_at = ?2 WHERE id = ?1"#,
            params![intention_id.to_string(), fired.fired_at.to_rfc3339()],
        )?;

        Ok(())
    }

    fn get_fire_history(&self, intention_id: Uuid, limit: usize) -> RookResult<Vec<FiredIntention>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            r#"SELECT intention_id, fired_at, reason_data, action_result
               FROM intention_fires
               WHERE intention_id = ?1
               ORDER BY fired_at DESC
               LIMIT ?2"#,
        )?;

        let results = stmt.query_map(params![intention_id.to_string(), limit as i64], |row| {
            let intention_id: String = row.get(0)?;
            let fired_at: String = row.get(1)?;
            let reason_data: String = row.get(2)?;
            let action_result: String = row.get(3)?;

            Ok((intention_id, fired_at, reason_data, action_result))
        })?;

        results
            .map(|r| {
                let (intention_id, fired_at, reason_data, action_result) = r?;
                Ok(FiredIntention {
                    intention_id: Uuid::parse_str(&intention_id)
                        .map_err(|e| crate::error::RookError::parse(e.to_string()))?,
                    fired_at: DateTime::parse_from_rfc3339(&fired_at)
                        .map(|dt| dt.with_timezone(&Utc))
                        .map_err(|e| crate::error::RookError::parse(e.to_string()))?,
                    reason: serde_json::from_str(&reason_data)?,
                    action_result: serde_json::from_str(&action_result)?,
                })
            })
            .collect()
    }

    fn cleanup_expired(&self) -> RookResult<usize> {
        let conn = self.conn.lock().unwrap();
        let now = Utc::now().to_rfc3339();

        // Deactivate expired intentions (keep for history)
        let count = conn.execute(
            "UPDATE intentions SET active = 0 WHERE expires_at IS NOT NULL AND expires_at < ?1 AND active = 1",
            params![now],
        )?;

        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intentions::IntentionAction;

    #[test]
    fn test_intention_store_crud() {
        let store = SqliteIntentionStore::in_memory().unwrap();

        // Create
        let intention = Intention::new(
            "test intention",
            TriggerCondition::keyword(vec!["test".to_string()]),
            IntentionAction::default(),
        );
        store.add(&intention).unwrap();

        // Read
        let retrieved = store.get(intention.id).unwrap().unwrap();
        assert_eq!(retrieved.name, "test intention");

        // Update
        let mut updated = retrieved;
        updated.name = "updated name".to_string();
        store.update(&updated).unwrap();

        let retrieved2 = store.get(intention.id).unwrap().unwrap();
        assert_eq!(retrieved2.name, "updated name");

        // Delete
        store.delete(intention.id).unwrap();
        assert!(store.get(intention.id).unwrap().is_none());
    }

    #[test]
    fn test_get_active_intentions() {
        let store = SqliteIntentionStore::in_memory().unwrap();

        // Active intention
        let active = Intention::new(
            "active",
            TriggerCondition::keyword(vec!["test".to_string()]),
            IntentionAction::default(),
        );
        store.add(&active).unwrap();

        // Expired intention
        let mut expired = Intention::new(
            "expired",
            TriggerCondition::keyword(vec!["test".to_string()]),
            IntentionAction::default(),
        );
        expired.expires_at = Some(Utc::now() - chrono::Duration::hours(1));
        store.add(&expired).unwrap();

        let actives = store.get_active().unwrap();
        assert_eq!(actives.len(), 1);
        assert_eq!(actives[0].name, "active");
    }

    #[test]
    fn test_get_by_trigger_type() {
        let store = SqliteIntentionStore::in_memory().unwrap();

        let keyword = Intention::new(
            "keyword intention",
            TriggerCondition::keyword(vec!["test".to_string()]),
            IntentionAction::default(),
        );
        store.add(&keyword).unwrap();

        let topic = Intention::new(
            "topic intention",
            TriggerCondition::topic("machine learning"),
            IntentionAction::default(),
        );
        store.add(&topic).unwrap();

        let keywords = store.get_by_trigger_type("keyword_mention").unwrap();
        assert_eq!(keywords.len(), 1);
        assert_eq!(keywords[0].name, "keyword intention");

        let topics = store.get_by_trigger_type("topic_discussed").unwrap();
        assert_eq!(topics.len(), 1);
        assert_eq!(topics[0].name, "topic intention");
    }

    #[test]
    fn test_get_for_user() {
        let store = SqliteIntentionStore::in_memory().unwrap();

        let intention1 = Intention::new(
            "user1 intention",
            TriggerCondition::keyword(vec!["test".to_string()]),
            IntentionAction::default(),
        )
        .with_user("user1");
        store.add(&intention1).unwrap();

        let intention2 = Intention::new(
            "user2 intention",
            TriggerCondition::keyword(vec!["test".to_string()]),
            IntentionAction::default(),
        )
        .with_user("user2");
        store.add(&intention2).unwrap();

        let user1_intentions = store.get_for_user("user1").unwrap();
        assert_eq!(user1_intentions.len(), 1);
        assert_eq!(user1_intentions[0].name, "user1 intention");
    }

    #[test]
    fn test_get_for_memory() {
        let store = SqliteIntentionStore::in_memory().unwrap();

        let intention = Intention::new(
            "memory intention",
            TriggerCondition::keyword(vec!["test".to_string()]),
            IntentionAction::default(),
        )
        .with_memory("mem-123");
        store.add(&intention).unwrap();

        let intentions = store.get_for_memory("mem-123").unwrap();
        assert_eq!(intentions.len(), 1);
        assert_eq!(intentions[0].name, "memory intention");
    }

    #[test]
    fn test_record_fire_and_history() {
        let store = SqliteIntentionStore::in_memory().unwrap();

        let intention = Intention::new(
            "test intention",
            TriggerCondition::keyword(vec!["test".to_string()]),
            IntentionAction::default(),
        );
        store.add(&intention).unwrap();

        // Record a fire
        let fired = FiredIntention::success(
            intention.id,
            crate::intentions::TriggerReason::Keyword {
                matched_keyword: "test".to_string(),
                context: "Testing the intention".to_string(),
            },
        );
        store.record_fire(intention.id, &fired).unwrap();

        // Check fire count updated
        let updated = store.get(intention.id).unwrap().unwrap();
        assert_eq!(updated.fire_count, 1);
        assert!(updated.last_fired_at.is_some());

        // Check fire history
        let history = store.get_fire_history(intention.id, 10).unwrap();
        assert_eq!(history.len(), 1);
        assert!(history[0].action_result.is_success());
    }

    #[test]
    fn test_cleanup_expired() {
        let store = SqliteIntentionStore::in_memory().unwrap();

        // Active intention
        let active = Intention::new(
            "active",
            TriggerCondition::keyword(vec!["test".to_string()]),
            IntentionAction::default(),
        );
        store.add(&active).unwrap();

        // Expired intention
        let mut expired = Intention::new(
            "expired",
            TriggerCondition::keyword(vec!["test".to_string()]),
            IntentionAction::default(),
        );
        expired.expires_at = Some(Utc::now() - chrono::Duration::hours(1));
        store.add(&expired).unwrap();

        // Clean up
        let count = store.cleanup_expired().unwrap();
        assert_eq!(count, 1);

        // Check that expired intention is now inactive
        let retrieved = store.get(expired.id).unwrap().unwrap();
        assert!(!retrieved.active);

        // Active intention should still be active
        let active_retrieved = store.get(active.id).unwrap().unwrap();
        assert!(active_retrieved.active);
    }
}
