//! FSRS-6 memory state types.
//!
//! Implements cognitive memory dynamics following the FSRS-6 algorithm
//! with dual-strength model based on Bjork's theory.

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};

/// Configuration for memory archival.
///
/// Defines thresholds for identifying memories that should be archived
/// (moved to cold storage) due to low retrievability and age.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchivalConfig {
    /// Retrievability threshold below which memories become archival candidates.
    /// Default: 0.1 (10% recall probability)
    pub archive_threshold: f32,

    /// Minimum age in days before a memory can be considered for archival.
    /// Prevents archiving recently created memories.
    /// Default: 30 days
    pub min_age_days: u32,

    /// Maximum number of memories to archive in a single operation.
    /// Default: 100
    pub archive_limit: usize,
}

impl ArchivalConfig {
    /// Create a new archival config with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create archival config with custom threshold.
    pub fn with_threshold(threshold: f32) -> Self {
        Self {
            archive_threshold: threshold,
            ..Default::default()
        }
    }

    /// Check if a memory is an archival candidate based on this config.
    ///
    /// A memory is an archival candidate if:
    /// 1. Its retrievability is below the archive threshold
    /// 2. It's older than min_age_days
    /// 3. It's not marked as a key memory (is_key check done externally)
    ///
    /// # Arguments
    /// * `retrievability` - Current retrievability of the memory
    /// * `created_at` - When the memory was created
    /// * `now` - Current timestamp
    pub fn is_candidate(&self, retrievability: f32, created_at: DateTime<Utc>, now: DateTime<Utc>) -> bool {
        // Check retrievability threshold
        if retrievability >= self.archive_threshold {
            return false;
        }

        // Check minimum age
        let age = now.signed_duration_since(created_at);
        let min_age = Duration::days(self.min_age_days as i64);

        age >= min_age
    }
}

impl Default for ArchivalConfig {
    fn default() -> Self {
        Self {
            archive_threshold: 0.1,
            min_age_days: 30,
            archive_limit: 100,
        }
    }
}

/// FSRS-6 memory state tracking cognitive dynamics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FsrsState {
    /// Stability: days for retrievability to drop to 90%.
    pub stability: f32,
    /// Difficulty: 1.0-10.0 scale (higher = harder to remember).
    pub difficulty: f32,
    /// Last review/access timestamp.
    pub last_review: Option<DateTime<Utc>>,
    /// Successful retrievals count.
    pub reps: u32,
    /// Lapse (forgetting) count.
    pub lapses: u32,
}

impl FsrsState {
    /// Create a new FSRS state with default values.
    ///
    /// Default stability is 0.0 (new memory), difficulty is 5.0 (medium).
    pub fn new() -> Self {
        Self {
            stability: 0.0,
            difficulty: 5.0,
            last_review: None,
            reps: 0,
            lapses: 0,
        }
    }

    /// Convert to fsrs::MemoryState for use with fsrs crate functions.
    pub fn to_memory_state(&self) -> fsrs::MemoryState {
        fsrs::MemoryState {
            stability: self.stability,
            difficulty: self.difficulty,
        }
    }

    /// Create from fsrs::MemoryState.
    pub fn from_memory_state(state: fsrs::MemoryState) -> Self {
        Self {
            stability: state.stability,
            difficulty: state.difficulty,
            last_review: None,
            reps: 0,
            lapses: 0,
        }
    }
}

impl Default for FsrsState {
    fn default() -> Self {
        Self::new()
    }
}

/// Dual-strength model (Bjork theory).
///
/// Tracks separate storage and retrieval strengths:
/// - Storage strength grows with repetition (how well-encoded)
/// - Retrieval strength decays with time (current accessibility)
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DualStrength {
    /// Storage strength: how well-encoded, grows with repetition.
    pub storage_strength: f32,
    /// Retrieval strength: current accessibility, decays with time.
    pub retrieval_strength: f32,
}

impl DualStrength {
    /// Create a new dual-strength with zero values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Update storage strength based on grade and repetition count.
    ///
    /// Storage strength increases with each review, with diminishing returns.
    /// Higher grades (Easy > Good > Hard > Again) give larger increases.
    pub fn update_storage(&mut self, grade: Grade, reps: u32) {
        // Base increase depends on grade
        let grade_factor = match grade {
            Grade::Again => 0.1,
            Grade::Hard => 0.3,
            Grade::Good => 0.5,
            Grade::Easy => 0.8,
        };

        // Diminishing returns: each rep adds less (logarithmic growth)
        // Formula: increment = grade_factor / (1 + 0.1 * reps)
        let diminishing_factor = 1.0 / (1.0 + 0.1 * reps as f32);
        let increment = grade_factor * diminishing_factor;

        self.storage_strength += increment;
    }

    /// Update retrieval strength based on current retrievability and grade.
    ///
    /// Retrieval strength is reset based on recall quality:
    /// - Successful recall (Good/Easy) sets retrieval close to 1.0
    /// - Partial recall (Hard) sets retrieval to moderate level
    /// - Failed recall (Again) resets retrieval to low level
    pub fn update_retrieval(&mut self, retrievability: f32, grade: Grade) {
        // Base reset level depends on grade
        let base_level = match grade {
            Grade::Again => 0.3, // Failed recall, low reset
            Grade::Hard => 0.6,  // Difficult recall, moderate reset
            Grade::Good => 0.9,  // Normal recall, high reset
            Grade::Easy => 1.0,  // Perfect recall, full reset
        };

        // Blend with current retrievability (some memory of previous state)
        self.retrieval_strength = base_level * 0.8 + retrievability * 0.2;
    }
}

/// Grade for memory feedback (maps to fsrs rating values 1-4).
///
/// Used when recording memory access quality:
/// - Again (1): Complete failure to recall
/// - Hard (2): Successful but difficult recall
/// - Good (3): Normal successful recall
/// - Easy (4): Effortless recall
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum Grade {
    /// Complete failure to recall.
    Again = 1,
    /// Successful but difficult recall.
    Hard = 2,
    /// Normal successful recall.
    Good = 3,
    /// Effortless recall.
    Easy = 4,
}

impl Grade {
    /// Convert to fsrs rating value (u8).
    ///
    /// FSRS uses integer ratings 1-4 where:
    /// - 1 = Again (failed recall)
    /// - 2 = Hard
    /// - 3 = Good
    /// - 4 = Easy
    pub fn to_rating(self) -> u8 {
        self as u8
    }

    /// Create from fsrs rating value.
    ///
    /// Returns None for invalid rating values.
    pub fn from_rating(rating: u8) -> Option<Self> {
        match rating {
            1 => Some(Grade::Again),
            2 => Some(Grade::Hard),
            3 => Some(Grade::Good),
            4 => Some(Grade::Easy),
            _ => None,
        }
    }
}

impl From<Grade> for u8 {
    fn from(grade: Grade) -> Self {
        grade.to_rating()
    }
}

impl TryFrom<u8> for Grade {
    type Error = ();

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        Grade::from_rating(value).ok_or(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fsrs_state_new() {
        let state = FsrsState::new();
        assert_eq!(state.stability, 0.0);
        assert_eq!(state.difficulty, 5.0);
        assert!(state.last_review.is_none());
        assert_eq!(state.reps, 0);
        assert_eq!(state.lapses, 0);
    }

    #[test]
    fn test_fsrs_state_to_memory_state() {
        let mut state = FsrsState::new();
        state.stability = 10.0;
        state.difficulty = 7.5;

        let mem_state = state.to_memory_state();
        assert_eq!(mem_state.stability, 10.0);
        assert_eq!(mem_state.difficulty, 7.5);
    }

    #[test]
    fn test_dual_strength_update_storage() {
        let mut strength = DualStrength::new();

        // First rep with Good grade
        strength.update_storage(Grade::Good, 0);
        assert!(strength.storage_strength > 0.0);
        let after_first = strength.storage_strength;

        // Second rep should add less (diminishing returns)
        strength.update_storage(Grade::Good, 1);
        let increment_second = strength.storage_strength - after_first;

        // Third rep should add even less
        let before_third = strength.storage_strength;
        strength.update_storage(Grade::Good, 2);
        let increment_third = strength.storage_strength - before_third;

        assert!(increment_third < increment_second);
    }

    #[test]
    fn test_dual_strength_grade_ordering() {
        // Higher grades should give more storage strength
        let mut again = DualStrength::new();
        let mut hard = DualStrength::new();
        let mut good = DualStrength::new();
        let mut easy = DualStrength::new();

        again.update_storage(Grade::Again, 0);
        hard.update_storage(Grade::Hard, 0);
        good.update_storage(Grade::Good, 0);
        easy.update_storage(Grade::Easy, 0);

        assert!(again.storage_strength < hard.storage_strength);
        assert!(hard.storage_strength < good.storage_strength);
        assert!(good.storage_strength < easy.storage_strength);
    }

    #[test]
    fn test_dual_strength_update_retrieval() {
        let mut strength = DualStrength::new();

        // Easy recall should set retrieval high
        strength.update_retrieval(0.5, Grade::Easy);
        assert!(strength.retrieval_strength > 0.8);

        // Again should set retrieval low
        strength.update_retrieval(0.5, Grade::Again);
        assert!(strength.retrieval_strength < 0.5);
    }

    #[test]
    fn test_grade_to_rating() {
        assert_eq!(Grade::Again.to_rating(), 1);
        assert_eq!(Grade::Hard.to_rating(), 2);
        assert_eq!(Grade::Good.to_rating(), 3);
        assert_eq!(Grade::Easy.to_rating(), 4);
    }

    #[test]
    fn test_grade_from_rating() {
        assert_eq!(Grade::from_rating(1), Some(Grade::Again));
        assert_eq!(Grade::from_rating(2), Some(Grade::Hard));
        assert_eq!(Grade::from_rating(3), Some(Grade::Good));
        assert_eq!(Grade::from_rating(4), Some(Grade::Easy));
        assert_eq!(Grade::from_rating(0), None);
        assert_eq!(Grade::from_rating(5), None);
    }

    // ============================================================
    // ArchivalConfig tests
    // ============================================================

    #[test]
    fn test_archival_config_default() {
        let config = ArchivalConfig::default();
        assert_eq!(config.archive_threshold, 0.1);
        assert_eq!(config.min_age_days, 30);
        assert_eq!(config.archive_limit, 100);
    }

    #[test]
    fn test_archival_config_with_threshold() {
        let config = ArchivalConfig::with_threshold(0.05);
        assert_eq!(config.archive_threshold, 0.05);
        assert_eq!(config.min_age_days, 30); // Unchanged default
    }

    #[test]
    fn test_archival_candidate_low_retrievability_old_memory() {
        let config = ArchivalConfig::default();
        let now = Utc::now();
        let created_at = now - Duration::days(60); // 60 days old

        // Low retrievability (5%), old memory -> candidate
        assert!(config.is_candidate(0.05, created_at, now));
    }

    #[test]
    fn test_archival_candidate_high_retrievability() {
        let config = ArchivalConfig::default();
        let now = Utc::now();
        let created_at = now - Duration::days(60);

        // High retrievability (50%), even if old -> not candidate
        assert!(!config.is_candidate(0.5, created_at, now));
    }

    #[test]
    fn test_archival_candidate_too_young() {
        let config = ArchivalConfig::default();
        let now = Utc::now();
        let created_at = now - Duration::days(10); // Only 10 days old

        // Low retrievability but too young -> not candidate
        assert!(!config.is_candidate(0.05, created_at, now));
    }

    #[test]
    fn test_archival_candidate_at_threshold_boundary() {
        let config = ArchivalConfig::default(); // threshold = 0.1
        let now = Utc::now();
        let created_at = now - Duration::days(60);

        // Exactly at threshold -> not candidate (must be below)
        assert!(!config.is_candidate(0.1, created_at, now));

        // Just below threshold -> candidate
        assert!(config.is_candidate(0.099, created_at, now));
    }

    #[test]
    fn test_archival_candidate_at_age_boundary() {
        let config = ArchivalConfig::default(); // min_age_days = 30
        let now = Utc::now();

        // Exactly 30 days old -> candidate
        let created_at_30 = now - Duration::days(30);
        assert!(config.is_candidate(0.05, created_at_30, now));

        // 29 days old -> not candidate
        let created_at_29 = now - Duration::days(29);
        assert!(!config.is_candidate(0.05, created_at_29, now));
    }
}
