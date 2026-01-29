//! FSRS state persistence store.
//!
//! Provides SQLite-backed storage for FSRS memory states,
//! enabling state retrieval, persistence, and archival candidate queries.
//!
//! Also provides storage for synaptic tags and consolidation phases
//! for the STC (Synaptic Tagging and Capture) memory consolidation model.

use crate::consolidation::{BehavioralTagger, ConsolidationPhase, NoveltyResult, SynapticTag};
use crate::error::{RookError, RookResult};
use crate::types::{ArchivalConfig, FsrsState};
use chrono::{DateTime, Utc};
use rusqlite::{params, Connection, OptionalExtension};
use std::path::Path;
use std::str::FromStr;
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
            -- FSRS states table for spaced repetition scheduling
            CREATE TABLE IF NOT EXISTS fsrs_states (
                memory_id TEXT PRIMARY KEY,
                stability REAL NOT NULL,
                difficulty REAL NOT NULL,
                last_review TEXT,
                reps INTEGER NOT NULL DEFAULT 0,
                lapses INTEGER NOT NULL DEFAULT 0,
                is_key INTEGER NOT NULL DEFAULT 0,
                consolidation_phase TEXT NOT NULL DEFAULT 'immediate',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_fsrs_states_is_key ON fsrs_states(is_key);
            CREATE INDEX IF NOT EXISTS idx_fsrs_states_stability ON fsrs_states(stability);
            CREATE INDEX IF NOT EXISTS idx_fsrs_states_last_review ON fsrs_states(last_review);
            CREATE INDEX IF NOT EXISTS idx_fsrs_states_created_at ON fsrs_states(created_at);
            CREATE INDEX IF NOT EXISTS idx_fsrs_states_consolidation_phase ON fsrs_states(consolidation_phase);

            -- Synaptic tags table for STC memory consolidation
            CREATE TABLE IF NOT EXISTS synaptic_tags (
                memory_id TEXT PRIMARY KEY,
                initial_strength REAL NOT NULL,
                tau REAL NOT NULL,
                tagged_at TEXT NOT NULL,
                prp_available INTEGER NOT NULL DEFAULT 0,
                prp_available_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_synaptic_tags_tagged_at ON synaptic_tags(tagged_at);
            CREATE INDEX IF NOT EXISTS idx_synaptic_tags_prp_available ON synaptic_tags(prp_available);
            ",
        )?;

        // Add dual_strength columns for consolidation (CON-07)
        // These are added via ALTER TABLE to support existing databases
        let has_storage: bool = conn
            .query_row(
                "SELECT COUNT(*) FROM pragma_table_info('fsrs_states') WHERE name = 'storage_strength'",
                [],
                |row| row.get::<_, i64>(0),
            )
            .unwrap_or(0)
            > 0;

        if !has_storage {
            conn.execute(
                "ALTER TABLE fsrs_states ADD COLUMN storage_strength REAL DEFAULT 0.5",
                [],
            )?;
        }

        let has_retrieval: bool = conn
            .query_row(
                "SELECT COUNT(*) FROM pragma_table_info('fsrs_states') WHERE name = 'retrieval_strength'",
                [],
                |row| row.get::<_, i64>(0),
            )
            .unwrap_or(0)
            > 0;

        if !has_retrieval {
            conn.execute(
                "ALTER TABLE fsrs_states ADD COLUMN retrieval_strength REAL DEFAULT 1.0",
                [],
            )?;
        }

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

    // =========================================================================
    // Synaptic Tag Methods
    // =========================================================================

    /// Save a synaptic tag for a memory.
    ///
    /// Creates or updates the tag for the given memory ID.
    pub fn save_synaptic_tag(&self, tag: &SynapticTag) -> RookResult<()> {
        let conn = self.conn.lock().map_err(|e| RookError::database(e.to_string()))?;

        let now = Utc::now();
        let prp_available_at_str = tag.prp_available_at.map(|dt| dt.to_rfc3339());

        conn.execute(
            "INSERT OR REPLACE INTO synaptic_tags
             (memory_id, initial_strength, tau, tagged_at, prp_available, prp_available_at, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6,
                     COALESCE((SELECT created_at FROM synaptic_tags WHERE memory_id = ?1), ?7),
                     ?8)",
            params![
                tag.memory_id,
                tag.initial_strength,
                tag.tau,
                tag.tagged_at.to_rfc3339(),
                if tag.prp_available { 1 } else { 0 },
                prp_available_at_str,
                now.to_rfc3339(),
                now.to_rfc3339(),
            ],
        )?;

        Ok(())
    }

    /// Get a synaptic tag for a memory.
    ///
    /// Returns None if the memory doesn't have a stored tag.
    pub fn get_synaptic_tag(&self, memory_id: &str) -> RookResult<Option<SynapticTag>> {
        let conn = self.conn.lock().map_err(|e| RookError::database(e.to_string()))?;

        let result = conn
            .query_row(
                "SELECT memory_id, initial_strength, tau, tagged_at, prp_available, prp_available_at
                 FROM synaptic_tags WHERE memory_id = ?1",
                params![memory_id],
                |row| {
                    let memory_id: String = row.get(0)?;
                    let initial_strength: f64 = row.get(1)?;
                    let tau: f64 = row.get(2)?;
                    let tagged_at_str: String = row.get(3)?;
                    let prp_available: i32 = row.get(4)?;
                    let prp_available_at_str: Option<String> = row.get(5)?;

                    let tagged_at = DateTime::parse_from_rfc3339(&tagged_at_str)
                        .map(|dt| dt.with_timezone(&Utc))
                        .unwrap_or_else(|_| Utc::now());

                    let prp_available_at = prp_available_at_str.and_then(|s| {
                        DateTime::parse_from_rfc3339(&s)
                            .map(|dt| dt.with_timezone(&Utc))
                            .ok()
                    });

                    Ok(SynapticTag {
                        memory_id,
                        initial_strength,
                        tau,
                        tagged_at,
                        prp_available: prp_available != 0,
                        prp_available_at,
                    })
                },
            )
            .optional()?;

        Ok(result)
    }

    /// Delete a synaptic tag for a memory.
    pub fn delete_synaptic_tag(&self, memory_id: &str) -> RookResult<bool> {
        let conn = self.conn.lock().map_err(|e| RookError::database(e.to_string()))?;

        let deleted = conn.execute(
            "DELETE FROM synaptic_tags WHERE memory_id = ?1",
            params![memory_id],
        )?;

        Ok(deleted > 0)
    }

    /// Get synaptic tags created within a time range.
    ///
    /// Useful for behavioral tagging queries - finding tags that might
    /// benefit from a recent novel/emotional event providing PRPs.
    ///
    /// # Arguments
    ///
    /// * `start` - Start of the time range (inclusive)
    /// * `end` - End of the time range (inclusive)
    ///
    /// Returns tags ordered by tagged_at descending (most recent first).
    pub fn get_tags_in_time_range(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> RookResult<Vec<SynapticTag>> {
        let conn = self.conn.lock().map_err(|e| RookError::database(e.to_string()))?;

        let mut stmt = conn.prepare(
            "SELECT memory_id, initial_strength, tau, tagged_at, prp_available, prp_available_at
             FROM synaptic_tags
             WHERE tagged_at >= ?1 AND tagged_at <= ?2
             ORDER BY tagged_at DESC",
        )?;

        let tags = stmt
            .query_map(params![start.to_rfc3339(), end.to_rfc3339()], |row| {
                let memory_id: String = row.get(0)?;
                let initial_strength: f64 = row.get(1)?;
                let tau: f64 = row.get(2)?;
                let tagged_at_str: String = row.get(3)?;
                let prp_available: i32 = row.get(4)?;
                let prp_available_at_str: Option<String> = row.get(5)?;

                let tagged_at = DateTime::parse_from_rfc3339(&tagged_at_str)
                    .map(|dt| dt.with_timezone(&Utc))
                    .unwrap_or_else(|_| Utc::now());

                let prp_available_at = prp_available_at_str.and_then(|s| {
                    DateTime::parse_from_rfc3339(&s)
                        .map(|dt| dt.with_timezone(&Utc))
                        .ok()
                });

                Ok(SynapticTag {
                    memory_id,
                    initial_strength,
                    tau,
                    tagged_at,
                    prp_available: prp_available != 0,
                    prp_available_at,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(tags)
    }

    /// Get all tags without PRP that are still valid (haven't decayed below threshold).
    ///
    /// Useful for finding tags that could benefit from behavioral tagging.
    pub fn get_tags_needing_prp(&self, validity_threshold: f64) -> RookResult<Vec<SynapticTag>> {
        let conn = self.conn.lock().map_err(|e| RookError::database(e.to_string()))?;

        let mut stmt = conn.prepare(
            "SELECT memory_id, initial_strength, tau, tagged_at, prp_available, prp_available_at
             FROM synaptic_tags
             WHERE prp_available = 0
             ORDER BY tagged_at DESC",
        )?;

        let tags: Vec<SynapticTag> = stmt
            .query_map([], |row| {
                let memory_id: String = row.get(0)?;
                let initial_strength: f64 = row.get(1)?;
                let tau: f64 = row.get(2)?;
                let tagged_at_str: String = row.get(3)?;
                let prp_available: i32 = row.get(4)?;
                let prp_available_at_str: Option<String> = row.get(5)?;

                let tagged_at = DateTime::parse_from_rfc3339(&tagged_at_str)
                    .map(|dt| dt.with_timezone(&Utc))
                    .unwrap_or_else(|_| Utc::now());

                let prp_available_at = prp_available_at_str.and_then(|s| {
                    DateTime::parse_from_rfc3339(&s)
                        .map(|dt| dt.with_timezone(&Utc))
                        .ok()
                });

                Ok(SynapticTag {
                    memory_id,
                    initial_strength,
                    tau,
                    tagged_at,
                    prp_available: prp_available != 0,
                    prp_available_at,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;

        // Filter to only valid tags (strength above threshold)
        Ok(tags
            .into_iter()
            .filter(|tag| tag.is_valid_with_threshold(validity_threshold))
            .collect())
    }

    // =========================================================================
    // Dual-Strength Methods (Bjork Model - CON-07)
    // =========================================================================

    /// Get DualStrength for a memory.
    ///
    /// Returns None if the memory doesn't have a stored state.
    pub fn get_dual_strength(
        &self,
        memory_id: &str,
    ) -> RookResult<Option<crate::types::DualStrength>> {
        let conn = self.conn.lock().map_err(|e| RookError::database(e.to_string()))?;

        let result = conn
            .query_row(
                "SELECT storage_strength, retrieval_strength FROM fsrs_states WHERE memory_id = ?1",
                params![memory_id],
                |row| {
                    Ok(crate::types::DualStrength {
                        storage_strength: row.get::<_, f32>(0).unwrap_or(0.5),
                        retrieval_strength: row.get::<_, f32>(1).unwrap_or(1.0),
                    })
                },
            )
            .optional()?;

        Ok(result)
    }

    /// Save DualStrength for a memory.
    ///
    /// Updates only the dual_strength columns, preserving other state.
    pub fn save_dual_strength(
        &self,
        memory_id: &str,
        dual: &crate::types::DualStrength,
    ) -> RookResult<bool> {
        let conn = self.conn.lock().map_err(|e| RookError::database(e.to_string()))?;

        let updated = conn.execute(
            "UPDATE fsrs_states SET storage_strength = ?1, retrieval_strength = ?2, updated_at = ?3
             WHERE memory_id = ?4",
            params![
                dual.storage_strength,
                dual.retrieval_strength,
                Utc::now().to_rfc3339(),
                memory_id
            ],
        )?;

        Ok(updated > 0)
    }

    // =========================================================================
    // Consolidation Phase Methods
    // =========================================================================

    /// Get the consolidation phase for a memory.
    ///
    /// Returns None if the memory doesn't exist.
    pub fn get_consolidation_phase(&self, memory_id: &str) -> RookResult<Option<ConsolidationPhase>> {
        let conn = self.conn.lock().map_err(|e| RookError::database(e.to_string()))?;

        let result = conn
            .query_row(
                "SELECT consolidation_phase FROM fsrs_states WHERE memory_id = ?1",
                params![memory_id],
                |row| {
                    let phase_str: String = row.get(0)?;
                    Ok(phase_str)
                },
            )
            .optional()?;

        match result {
            Some(phase_str) => {
                let phase = ConsolidationPhase::from_str(&phase_str)
                    .unwrap_or(ConsolidationPhase::Immediate);
                Ok(Some(phase))
            }
            None => Ok(None),
        }
    }

    /// Update the consolidation phase for a memory.
    ///
    /// Returns true if the memory was updated, false if it doesn't exist.
    pub fn update_consolidation_phase(
        &self,
        memory_id: &str,
        phase: ConsolidationPhase,
    ) -> RookResult<bool> {
        let conn = self.conn.lock().map_err(|e| RookError::database(e.to_string()))?;

        let updated = conn.execute(
            "UPDATE fsrs_states SET consolidation_phase = ?1, updated_at = ?2 WHERE memory_id = ?3",
            params![phase.to_string(), Utc::now().to_rfc3339(), memory_id],
        )?;

        Ok(updated > 0)
    }

    /// Get all memories in a specific consolidation phase.
    pub fn get_memories_in_phase(&self, phase: ConsolidationPhase) -> RookResult<Vec<String>> {
        let conn = self.conn.lock().map_err(|e| RookError::database(e.to_string()))?;

        let mut stmt = conn.prepare(
            "SELECT memory_id FROM fsrs_states WHERE consolidation_phase = ?1",
        )?;

        let ids = stmt
            .query_map(params![phase.to_string()], |row| row.get(0))?
            .collect::<Result<Vec<String>, _>>()?;

        Ok(ids)
    }

    /// Get count of memories in each consolidation phase.
    pub fn count_by_phase(&self) -> RookResult<std::collections::HashMap<ConsolidationPhase, usize>> {
        let conn = self.conn.lock().map_err(|e| RookError::database(e.to_string()))?;

        let mut stmt = conn.prepare(
            "SELECT consolidation_phase, COUNT(*) FROM fsrs_states GROUP BY consolidation_phase",
        )?;

        let mut counts = std::collections::HashMap::new();

        let rows = stmt.query_map([], |row| {
            let phase_str: String = row.get(0)?;
            let count: i64 = row.get(1)?;
            Ok((phase_str, count))
        })?;

        for row in rows {
            let (phase_str, count) = row?;
            if let Ok(phase) = ConsolidationPhase::from_str(&phase_str) {
                counts.insert(phase, count as usize);
            }
        }

        Ok(counts)
    }

    // =========================================================================
    // Behavioral Tagging Integration
    // =========================================================================

    /// Process a novel event and boost nearby memories' tags.
    ///
    /// This combines:
    /// 1. Querying tags in the behavioral window from the database
    /// 2. Applying PRP boost to valid tags
    /// 3. Saving updated tags back to database
    ///
    /// Returns the NoveltyResult indicating what happened.
    pub fn process_novelty_boost(
        &self,
        tagger: &BehavioralTagger,
        encoding_surprise: f32,
        novel_event_time: DateTime<Utc>,
        novel_memory_id: &str,
    ) -> RookResult<NoveltyResult> {
        // Check if this is actually a novel event
        if !tagger.is_novel_event(encoding_surprise) {
            return Ok(NoveltyResult::NotNovel {
                encoding_surprise,
                threshold: tagger.config().novelty_threshold,
            });
        }

        // Get the time window
        let (window_start, window_end) = tagger.get_tagging_window(novel_event_time);

        // Query tags in the window from database
        let mut tags = self.get_tags_in_time_range(window_start, window_end)?;

        if tags.is_empty() {
            return Ok(NoveltyResult::NoValidTags);
        }

        // Apply PRP boost
        let boosted_ids = tagger.apply_prp_boost(&mut tags, novel_event_time, Some(novel_memory_id));

        if boosted_ids.is_empty() {
            return Ok(NoveltyResult::NoValidTags);
        }

        // Save updated tags back to database
        for tag in tags.iter() {
            if boosted_ids.contains(&tag.memory_id) {
                self.save_synaptic_tag(tag)?;
            }
        }

        Ok(NoveltyResult::Boosted {
            count: boosted_ids.len(),
            boosted_ids,
        })
    }

    /// Get memories with valid tags (for consolidation processing).
    ///
    /// Returns (memory_id, SynapticTag) pairs for memories with tags
    /// that are still valid (above threshold).
    pub fn get_memories_with_valid_tags(
        &self,
        threshold: f32,
        now: DateTime<Utc>,
    ) -> RookResult<Vec<(String, SynapticTag)>> {
        let conn = self.conn.lock().map_err(|e| RookError::database(e.to_string()))?;

        let mut stmt = conn.prepare(
            "SELECT memory_id, initial_strength, tau, tagged_at, prp_available, prp_available_at
             FROM synaptic_tags",
        )?;

        let results: Vec<(String, SynapticTag)> = stmt
            .query_map([], |row| {
                let memory_id: String = row.get(0)?;
                let initial_strength: f64 = row.get(1)?;
                let tau: f64 = row.get(2)?;
                let tagged_at_str: String = row.get(3)?;
                let prp_available: i32 = row.get(4)?;
                let prp_available_at_str: Option<String> = row.get(5)?;

                let tagged_at = DateTime::parse_from_rfc3339(&tagged_at_str)
                    .map(|dt| dt.with_timezone(&Utc))
                    .unwrap_or_else(|_| Utc::now());

                let prp_available_at = prp_available_at_str.and_then(|s| {
                    DateTime::parse_from_rfc3339(&s)
                        .map(|dt| dt.with_timezone(&Utc))
                        .ok()
                });

                let tag = SynapticTag {
                    memory_id: memory_id.clone(),
                    initial_strength,
                    tau,
                    tagged_at,
                    prp_available: prp_available != 0,
                    prp_available_at,
                };

                Ok((memory_id, tag))
            })?
            .filter_map(|r| r.ok())
            .filter(|(_, tag)| tag.is_valid_at(now, threshold))
            .collect();

        Ok(results)
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

    // =========================================================================
    // Synaptic Tag Tests
    // =========================================================================

    #[test]
    fn test_save_and_get_synaptic_tag() {
        let store = CognitiveStore::in_memory().unwrap();

        let tag = SynapticTag::new("mem1".to_string(), 0.8);
        store.save_synaptic_tag(&tag).unwrap();

        let retrieved = store.get_synaptic_tag("mem1").unwrap().unwrap();

        assert_eq!(retrieved.memory_id, "mem1");
        assert!((retrieved.initial_strength - 0.8).abs() < 0.001);
        assert!((retrieved.tau - 60.0).abs() < 0.001);
        assert!(!retrieved.prp_available);
    }

    #[test]
    fn test_synaptic_tag_with_prp() {
        let store = CognitiveStore::in_memory().unwrap();

        let mut tag = SynapticTag::new("mem1".to_string(), 0.9);
        tag.set_prp_available();
        store.save_synaptic_tag(&tag).unwrap();

        let retrieved = store.get_synaptic_tag("mem1").unwrap().unwrap();

        assert!(retrieved.prp_available);
        assert!(retrieved.prp_available_at.is_some());
    }

    #[test]
    fn test_delete_synaptic_tag() {
        let store = CognitiveStore::in_memory().unwrap();

        let tag = SynapticTag::new("mem1".to_string(), 0.8);
        store.save_synaptic_tag(&tag).unwrap();

        assert!(store.delete_synaptic_tag("mem1").unwrap());
        assert!(store.get_synaptic_tag("mem1").unwrap().is_none());

        // Deleting non-existent returns false
        assert!(!store.delete_synaptic_tag("nonexistent").unwrap());
    }

    #[test]
    fn test_get_tags_in_time_range() {
        let store = CognitiveStore::in_memory().unwrap();
        let now = Utc::now();

        // Create tags at different times
        let tag1 = SynapticTag::with_timestamp("mem1".to_string(), 0.8, now - Duration::hours(2));
        let tag2 = SynapticTag::with_timestamp("mem2".to_string(), 0.8, now - Duration::hours(1));
        let tag3 = SynapticTag::with_timestamp("mem3".to_string(), 0.8, now);

        store.save_synaptic_tag(&tag1).unwrap();
        store.save_synaptic_tag(&tag2).unwrap();
        store.save_synaptic_tag(&tag3).unwrap();

        // Query for tags in the last 90 minutes
        let start = now - Duration::minutes(90);
        let end = now;
        let tags = store.get_tags_in_time_range(start, end).unwrap();

        // Should get mem2 and mem3 (mem1 is too old)
        assert_eq!(tags.len(), 2);
        // Results should be ordered by tagged_at descending
        assert_eq!(tags[0].memory_id, "mem3");
        assert_eq!(tags[1].memory_id, "mem2");
    }

    #[test]
    fn test_get_tags_needing_prp() {
        let store = CognitiveStore::in_memory().unwrap();
        let now = Utc::now();

        // Tag with PRP (shouldn't be returned)
        let mut tag1 = SynapticTag::new("mem1".to_string(), 0.8);
        tag1.set_prp_available();
        store.save_synaptic_tag(&tag1).unwrap();

        // Tag without PRP, still valid (should be returned)
        let tag2 = SynapticTag::new("mem2".to_string(), 0.8);
        store.save_synaptic_tag(&tag2).unwrap();

        // Tag without PRP, but expired (shouldn't be returned)
        let old_time = now - Duration::hours(4);
        let tag3 = SynapticTag::with_timestamp("mem3".to_string(), 0.8, old_time);
        store.save_synaptic_tag(&tag3).unwrap();

        let tags = store.get_tags_needing_prp(0.1).unwrap();

        assert_eq!(tags.len(), 1);
        assert_eq!(tags[0].memory_id, "mem2");
    }

    // =========================================================================
    // Dual-Strength Tests
    // =========================================================================

    #[test]
    fn test_dual_strength_default_values() {
        let store = CognitiveStore::in_memory().unwrap();

        let state = create_test_state(10.0, 5);
        store.save_state("mem1", &state, false, None).unwrap();

        let dual = store.get_dual_strength("mem1").unwrap().unwrap();

        // Default values: storage=0.5, retrieval=1.0
        assert!((dual.storage_strength - 0.5).abs() < 0.001);
        assert!((dual.retrieval_strength - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_save_and_get_dual_strength() {
        let store = CognitiveStore::in_memory().unwrap();

        let state = create_test_state(10.0, 5);
        store.save_state("mem1", &state, false, None).unwrap();

        // Update dual strength
        let updated = crate::types::DualStrength {
            storage_strength: 0.8,
            retrieval_strength: 0.6,
        };
        assert!(store.save_dual_strength("mem1", &updated).unwrap());

        // Verify update
        let dual = store.get_dual_strength("mem1").unwrap().unwrap();
        assert!((dual.storage_strength - 0.8).abs() < 0.001);
        assert!((dual.retrieval_strength - 0.6).abs() < 0.001);
    }

    #[test]
    fn test_dual_strength_nonexistent_memory() {
        let store = CognitiveStore::in_memory().unwrap();

        // Getting from nonexistent memory returns None
        assert!(store.get_dual_strength("nonexistent").unwrap().is_none());

        // Saving to nonexistent memory returns false (no rows updated)
        let dual = crate::types::DualStrength {
            storage_strength: 0.5,
            retrieval_strength: 1.0,
        };
        assert!(!store.save_dual_strength("nonexistent", &dual).unwrap());
    }

    // =========================================================================
    // Consolidation Phase Tests
    // =========================================================================

    #[test]
    fn test_default_consolidation_phase() {
        let store = CognitiveStore::in_memory().unwrap();

        let state = create_test_state(10.0, 5);
        store.save_state("mem1", &state, false, None).unwrap();

        let phase = store.get_consolidation_phase("mem1").unwrap().unwrap();
        assert_eq!(phase, ConsolidationPhase::Immediate);
    }

    #[test]
    fn test_update_consolidation_phase() {
        let store = CognitiveStore::in_memory().unwrap();

        let state = create_test_state(10.0, 5);
        store.save_state("mem1", &state, false, None).unwrap();

        // Update to Early
        assert!(store.update_consolidation_phase("mem1", ConsolidationPhase::Early).unwrap());

        let phase = store.get_consolidation_phase("mem1").unwrap().unwrap();
        assert_eq!(phase, ConsolidationPhase::Early);

        // Update non-existent returns false
        assert!(!store.update_consolidation_phase("nonexistent", ConsolidationPhase::Early).unwrap());
    }

    #[test]
    fn test_get_memories_in_phase() {
        let store = CognitiveStore::in_memory().unwrap();

        let state = create_test_state(10.0, 5);
        store.save_state("mem1", &state, false, None).unwrap();
        store.save_state("mem2", &state, false, None).unwrap();
        store.save_state("mem3", &state, false, None).unwrap();

        // Update phases
        store.update_consolidation_phase("mem1", ConsolidationPhase::Early).unwrap();
        store.update_consolidation_phase("mem3", ConsolidationPhase::Early).unwrap();

        let immediate = store.get_memories_in_phase(ConsolidationPhase::Immediate).unwrap();
        let early = store.get_memories_in_phase(ConsolidationPhase::Early).unwrap();

        assert_eq!(immediate.len(), 1);
        assert_eq!(immediate[0], "mem2");

        assert_eq!(early.len(), 2);
        assert!(early.contains(&"mem1".to_string()));
        assert!(early.contains(&"mem3".to_string()));
    }

    #[test]
    fn test_count_by_phase() {
        let store = CognitiveStore::in_memory().unwrap();

        let state = create_test_state(10.0, 5);
        store.save_state("mem1", &state, false, None).unwrap();
        store.save_state("mem2", &state, false, None).unwrap();
        store.save_state("mem3", &state, false, None).unwrap();
        store.save_state("mem4", &state, false, None).unwrap();

        // Update phases
        store.update_consolidation_phase("mem1", ConsolidationPhase::Early).unwrap();
        store.update_consolidation_phase("mem2", ConsolidationPhase::Early).unwrap();
        store.update_consolidation_phase("mem3", ConsolidationPhase::Consolidated).unwrap();

        let counts = store.count_by_phase().unwrap();

        assert_eq!(counts.get(&ConsolidationPhase::Immediate), Some(&1));
        assert_eq!(counts.get(&ConsolidationPhase::Early), Some(&2));
        assert_eq!(counts.get(&ConsolidationPhase::Consolidated), Some(&1));
        assert_eq!(counts.get(&ConsolidationPhase::Late), None); // No memories in Late
    }
}
