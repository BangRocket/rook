//! Memory consolidation manager for processing memories through lifecycle phases.
//!
//! Based on Synaptic Tagging and Capture (STC) theory:
//! - Immediate memories with valid tag + PRP consolidate to Early phase
//! - Early memories advance to Late after 24 hours
//! - Late memories advance to Consolidated after 72 hours
//! - Consolidated memories receive storage_strength boost

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::cognitive::CognitiveStore;
use crate::consolidation::ConsolidationPhase;
use crate::error::RookError;

/// Configuration for consolidation operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsolidationConfig {
    /// Minimum tag strength to consider for consolidation (default: 0.1)
    pub tag_threshold: f32,
    /// Storage strength boost on successful consolidation (default: 0.15)
    pub storage_boost: f32,
    /// Maximum memories to process per batch (default: 100)
    pub batch_size: usize,
    /// Whether to mark unconsolidated memories for faster decay (default: true)
    pub penalize_unconsolidated: bool,
    /// Storage strength penalty for unconsolidated memories (default: 0.05)
    pub unconsolidated_penalty: f32,
}

impl Default for ConsolidationConfig {
    fn default() -> Self {
        Self {
            tag_threshold: 0.1,
            storage_boost: 0.15,
            batch_size: 100,
            penalize_unconsolidated: true,
            unconsolidated_penalty: 0.05,
        }
    }
}

impl ConsolidationConfig {
    /// Create a new config with custom values.
    pub fn new(tag_threshold: f32, storage_boost: f32, batch_size: usize) -> Self {
        Self {
            tag_threshold: tag_threshold.clamp(0.0, 1.0),
            storage_boost: storage_boost.clamp(0.0, 1.0),
            batch_size: batch_size.max(1),
            ..Default::default()
        }
    }
}

/// Result of a consolidation run.
#[derive(Debug, Clone, Default)]
pub struct ConsolidationResult {
    /// Number of memories that successfully consolidated (Immediate -> Early)
    pub consolidated: usize,
    /// Number of memories whose tags expired without PRP (may decay faster)
    pub unconsolidated: usize,
    /// Number of memories that advanced phases (Early -> Late, Late -> Consolidated)
    pub advanced: usize,
    /// Number of memories skipped (not ready for transition)
    pub skipped: usize,
    /// Any errors encountered (non-fatal, processing continued)
    pub errors: Vec<String>,
    /// Timestamp when consolidation started
    pub started_at: DateTime<Utc>,
    /// Timestamp when consolidation completed
    pub completed_at: Option<DateTime<Utc>>,
}

impl ConsolidationResult {
    /// Create a new result with start timestamp.
    pub fn new() -> Self {
        Self {
            started_at: Utc::now(),
            ..Default::default()
        }
    }

    /// Mark the result as complete.
    pub fn complete(mut self) -> Self {
        self.completed_at = Some(Utc::now());
        self
    }

    /// Total memories processed.
    pub fn total_processed(&self) -> usize {
        self.consolidated + self.unconsolidated + self.advanced + self.skipped
    }

    /// Duration of the consolidation run.
    pub fn duration_ms(&self) -> Option<i64> {
        self.completed_at
            .map(|end| (end - self.started_at).num_milliseconds())
    }
}

/// Memory consolidation manager.
///
/// Processes memories through consolidation phases based on STC theory.
pub struct ConsolidationManager {
    store: Arc<CognitiveStore>,
    config: ConsolidationConfig,
}

impl ConsolidationManager {
    /// Create a new ConsolidationManager.
    pub fn new(store: Arc<CognitiveStore>, config: ConsolidationConfig) -> Self {
        Self { store, config }
    }

    /// Create a manager with default configuration.
    pub fn with_defaults(store: Arc<CognitiveStore>) -> Self {
        Self::new(store, ConsolidationConfig::default())
    }

    /// Get the configuration.
    pub fn config(&self) -> &ConsolidationConfig {
        &self.config
    }

    /// Main consolidation operation - processes memories through phases.
    ///
    /// This should be called periodically (e.g., every 5-15 minutes).
    ///
    /// Processing order:
    /// 1. Immediate phase: Check tag + PRP, consolidate or mark unconsolidated
    /// 2. Early phase: Time-based advancement to Late (24+ hours)
    /// 3. Late phase: Time-based advancement to Consolidated (72+ hours)
    pub fn consolidate(&self) -> Result<ConsolidationResult, RookError> {
        let mut result = ConsolidationResult::new();
        let now = Utc::now();

        // Phase 1: Process Immediate memories
        self.process_immediate_phase(&mut result, now)?;

        // Phase 2: Process Early memories (time-based)
        self.process_early_phase(&mut result, now)?;

        // Phase 3: Process Late memories (time-based)
        self.process_late_phase(&mut result, now)?;

        Ok(result.complete())
    }

    /// Process memories in Immediate phase.
    fn process_immediate_phase(
        &self,
        result: &mut ConsolidationResult,
        now: DateTime<Utc>,
    ) -> Result<(), RookError> {
        let memory_ids = self.store.get_memories_in_phase(ConsolidationPhase::Immediate)?;

        for memory_id in memory_ids.iter().take(self.config.batch_size) {
            match self.process_immediate_memory(memory_id, now) {
                Ok(ProcessResult::Consolidated) => result.consolidated += 1,
                Ok(ProcessResult::Unconsolidated) => result.unconsolidated += 1,
                Ok(ProcessResult::Skipped) => result.skipped += 1,
                Err(e) => result.errors.push(format!("{}: {}", memory_id, e)),
            }
        }

        Ok(())
    }

    /// Process a single Immediate phase memory.
    fn process_immediate_memory(
        &self,
        memory_id: &str,
        now: DateTime<Utc>,
    ) -> Result<ProcessResult, RookError> {
        // Get the synaptic tag for this memory
        let tag = match self.store.get_synaptic_tag(memory_id)? {
            Some(t) => t,
            None => return Ok(ProcessResult::Skipped), // No tag = not a candidate
        };

        // Check if tag can consolidate (valid + PRP available)
        if tag.can_consolidate_at(now, self.config.tag_threshold) {
            // Successful consolidation: boost storage_strength and advance phase
            self.boost_storage_strength(memory_id)?;
            self.store
                .update_consolidation_phase(memory_id, ConsolidationPhase::Early)?;
            self.store.delete_synaptic_tag(memory_id)?; // Tag no longer needed
            return Ok(ProcessResult::Consolidated);
        }

        // Check if tag has expired (no longer valid)
        if !tag.is_valid_at(now, self.config.tag_threshold) {
            // Tag expired without PRP - memory may decay faster
            if self.config.penalize_unconsolidated {
                self.penalize_storage_strength(memory_id)?;
            }
            // Still advance to Early (time has passed) but without boost
            self.store
                .update_consolidation_phase(memory_id, ConsolidationPhase::Early)?;
            self.store.delete_synaptic_tag(memory_id)?;
            return Ok(ProcessResult::Unconsolidated);
        }

        // Tag is still valid but no PRP yet - skip and check again later
        Ok(ProcessResult::Skipped)
    }

    /// Process memories in Early phase (time-based advancement).
    fn process_early_phase(
        &self,
        result: &mut ConsolidationResult,
        now: DateTime<Utc>,
    ) -> Result<(), RookError> {
        let memory_ids = self.store.get_memories_in_phase(ConsolidationPhase::Early)?;

        for memory_id in memory_ids.iter().take(self.config.batch_size) {
            match self.process_time_based_advancement(memory_id, ConsolidationPhase::Early, now) {
                Ok(true) => result.advanced += 1,
                Ok(false) => result.skipped += 1,
                Err(e) => result.errors.push(format!("{}: {}", memory_id, e)),
            }
        }

        Ok(())
    }

    /// Process memories in Late phase (time-based advancement).
    fn process_late_phase(
        &self,
        result: &mut ConsolidationResult,
        now: DateTime<Utc>,
    ) -> Result<(), RookError> {
        let memory_ids = self.store.get_memories_in_phase(ConsolidationPhase::Late)?;

        for memory_id in memory_ids.iter().take(self.config.batch_size) {
            match self.process_time_based_advancement(memory_id, ConsolidationPhase::Late, now) {
                Ok(true) => result.advanced += 1,
                Ok(false) => result.skipped += 1,
                Err(e) => result.errors.push(format!("{}: {}", memory_id, e)),
            }
        }

        Ok(())
    }

    /// Process time-based phase advancement.
    fn process_time_based_advancement(
        &self,
        memory_id: &str,
        current_phase: ConsolidationPhase,
        now: DateTime<Utc>,
    ) -> Result<bool, RookError> {
        // Get memory creation time from state
        let (_state, _, created_at) = match self.store.get_state(memory_id)? {
            Some(s) => s,
            None => return Ok(false),
        };

        // Calculate memory age
        let age = now.signed_duration_since(created_at);

        // Check if can advance based on time (tag/PRP not required for time-based phases)
        // For Early and Late phases, only time matters
        let can_advance = match current_phase {
            ConsolidationPhase::Early => age.num_hours() >= 24,
            ConsolidationPhase::Late => age.num_hours() >= 72,
            _ => false,
        };

        if can_advance {
            if let Some(next_phase) = current_phase.next() {
                self.store.update_consolidation_phase(memory_id, next_phase)?;
                return Ok(true);
            }
        }

        Ok(false)
    }

    /// Boost storage_strength for a consolidated memory (CON-07).
    fn boost_storage_strength(&self, memory_id: &str) -> Result<(), RookError> {
        if let Some((mut state, is_key, created_at)) = self.store.get_state(memory_id)? {
            // Also update the FsrsState stability as a consolidation bonus
            state.stability *= 1.0 + self.config.storage_boost;
            self.store.save_state(memory_id, &state, is_key, Some(created_at))?;
        }
        Ok(())
    }

    /// Penalize storage_strength for an unconsolidated memory.
    fn penalize_storage_strength(&self, memory_id: &str) -> Result<(), RookError> {
        if let Some((mut state, is_key, created_at)) = self.store.get_state(memory_id)? {
            // Reduce stability as a penalty for unconsolidated memory
            state.stability *= 1.0 - self.config.unconsolidated_penalty;
            self.store.save_state(memory_id, &state, is_key, Some(created_at))?;
        }
        Ok(())
    }
}

/// Internal result for single memory processing.
enum ProcessResult {
    Consolidated,
    Unconsolidated,
    Skipped,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consolidation::SynapticTag;
    use crate::types::FsrsState;
    use chrono::Duration;

    fn create_test_store() -> Arc<CognitiveStore> {
        Arc::new(CognitiveStore::in_memory().unwrap())
    }

    #[test]
    fn test_consolidation_config_defaults() {
        let config = ConsolidationConfig::default();
        assert!((config.tag_threshold - 0.1).abs() < 0.001);
        assert!((config.storage_boost - 0.15).abs() < 0.001);
        assert_eq!(config.batch_size, 100);
    }

    #[test]
    fn test_consolidation_result_tracking() {
        let mut result = ConsolidationResult::new();
        result.consolidated = 5;
        result.unconsolidated = 2;
        result.advanced = 3;
        result.skipped = 10;

        assert_eq!(result.total_processed(), 20);

        let completed = result.complete();
        assert!(completed.completed_at.is_some());
        assert!(completed.duration_ms().unwrap() >= 0);
    }

    #[test]
    fn test_consolidate_empty_store() {
        let store = create_test_store();
        let manager = ConsolidationManager::with_defaults(store);

        let result = manager.consolidate().unwrap();

        assert_eq!(result.consolidated, 0);
        assert_eq!(result.unconsolidated, 0);
        assert_eq!(result.advanced, 0);
    }

    #[test]
    fn test_consolidate_memory_with_prp() {
        let store = create_test_store();
        let now = Utc::now();

        // Create a memory state
        let state = FsrsState {
            stability: 10.0,
            difficulty: 5.0,
            last_review: Some(now),
            reps: 1,
            lapses: 0,
        };
        store.save_state("mem-1", &state, false, None).unwrap();

        // Create a tag with PRP (ready to consolidate)
        let mut tag = SynapticTag::default_for("mem-1".to_string(), now - Duration::minutes(30));
        tag.set_prp_available_at(now);
        store.save_synaptic_tag(&tag).unwrap();

        // Run consolidation
        let manager = ConsolidationManager::with_defaults(store.clone());
        let result = manager.consolidate().unwrap();

        // Should have consolidated the memory
        assert_eq!(result.consolidated, 1);
        assert_eq!(result.unconsolidated, 0);

        // Memory should be in Early phase now
        let phase = store.get_consolidation_phase("mem-1").unwrap().unwrap();
        assert_eq!(phase, ConsolidationPhase::Early);

        // Tag should be deleted
        assert!(store.get_synaptic_tag("mem-1").unwrap().is_none());

        // Stability should be boosted
        let (updated_state, _, _) = store.get_state("mem-1").unwrap().unwrap();
        assert!(updated_state.stability > 10.0);
    }

    #[test]
    fn test_consolidate_memory_with_expired_tag() {
        let store = create_test_store();
        let now = Utc::now();

        // Create a memory state
        let state = FsrsState {
            stability: 10.0,
            difficulty: 5.0,
            last_review: Some(now),
            reps: 1,
            lapses: 0,
        };
        store.save_state("mem-1", &state, false, None).unwrap();

        // Create a tag that's expired (old, no PRP)
        let tag = SynapticTag::default_for("mem-1".to_string(), now - Duration::hours(4));
        store.save_synaptic_tag(&tag).unwrap();

        // Run consolidation
        let manager = ConsolidationManager::with_defaults(store.clone());
        let result = manager.consolidate().unwrap();

        // Should be unconsolidated (tag expired)
        assert_eq!(result.consolidated, 0);
        assert_eq!(result.unconsolidated, 1);

        // Memory should still advance to Early phase
        let phase = store.get_consolidation_phase("mem-1").unwrap().unwrap();
        assert_eq!(phase, ConsolidationPhase::Early);

        // Stability should be penalized
        let (updated_state, _, _) = store.get_state("mem-1").unwrap().unwrap();
        assert!(updated_state.stability < 10.0);
    }

    #[test]
    fn test_consolidate_memory_with_valid_tag_no_prp() {
        let store = create_test_store();
        let now = Utc::now();

        // Create a memory state
        let state = FsrsState {
            stability: 10.0,
            difficulty: 5.0,
            last_review: Some(now),
            reps: 1,
            lapses: 0,
        };
        store.save_state("mem-1", &state, false, None).unwrap();

        // Create a fresh tag without PRP (not ready)
        let tag = SynapticTag::default_for("mem-1".to_string(), now - Duration::minutes(10));
        store.save_synaptic_tag(&tag).unwrap();

        // Run consolidation
        let manager = ConsolidationManager::with_defaults(store.clone());
        let result = manager.consolidate().unwrap();

        // Should be skipped (valid tag but no PRP)
        assert_eq!(result.consolidated, 0);
        assert_eq!(result.unconsolidated, 0);
        assert_eq!(result.skipped, 1);

        // Memory should still be in Immediate phase
        let phase = store.get_consolidation_phase("mem-1").unwrap().unwrap();
        assert_eq!(phase, ConsolidationPhase::Immediate);
    }

    #[test]
    fn test_time_based_advancement_early_to_late() {
        let store = create_test_store();
        let now = Utc::now();
        let old_date = now - Duration::hours(30); // 30 hours old

        // Create a memory in Early phase
        let state = FsrsState {
            stability: 10.0,
            difficulty: 5.0,
            last_review: Some(now),
            reps: 1,
            lapses: 0,
        };
        store.save_state("mem-1", &state, false, Some(old_date)).unwrap();
        store
            .update_consolidation_phase("mem-1", ConsolidationPhase::Early)
            .unwrap();

        // Run consolidation
        let manager = ConsolidationManager::with_defaults(store.clone());
        let result = manager.consolidate().unwrap();

        // Should advance to Late
        assert_eq!(result.advanced, 1);

        let phase = store.get_consolidation_phase("mem-1").unwrap().unwrap();
        assert_eq!(phase, ConsolidationPhase::Late);
    }

    #[test]
    fn test_time_based_advancement_late_to_consolidated() {
        let store = create_test_store();
        let now = Utc::now();
        let old_date = now - Duration::hours(80); // 80 hours old

        // Create a memory in Late phase
        let state = FsrsState {
            stability: 10.0,
            difficulty: 5.0,
            last_review: Some(now),
            reps: 1,
            lapses: 0,
        };
        store.save_state("mem-1", &state, false, Some(old_date)).unwrap();
        store
            .update_consolidation_phase("mem-1", ConsolidationPhase::Late)
            .unwrap();

        // Run consolidation
        let manager = ConsolidationManager::with_defaults(store.clone());
        let result = manager.consolidate().unwrap();

        // Should advance to Consolidated
        assert_eq!(result.advanced, 1);

        let phase = store.get_consolidation_phase("mem-1").unwrap().unwrap();
        assert_eq!(phase, ConsolidationPhase::Consolidated);
    }

    #[test]
    fn test_config_clamping() {
        let config = ConsolidationConfig::new(-0.5, 2.0, 0);

        // Values should be clamped
        assert!((config.tag_threshold - 0.0).abs() < 0.001);
        assert!((config.storage_boost - 1.0).abs() < 0.001);
        assert_eq!(config.batch_size, 1);
    }
}
