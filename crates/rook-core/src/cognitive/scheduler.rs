//! FSRS-6 scheduler for memory dynamics.
//!
//! Provides retrievability calculation and initial state assignment
//! based on the Free Spaced Repetition Scheduler (FSRS) algorithm.

use crate::types::{FsrsState, Grade};
use chrono::{DateTime, Utc};

/// FSRS-6 scheduler for memory dynamics.
///
/// Calculates retrievability (probability of recall) and manages
/// initial state assignment for new memories based on grade.
pub struct FsrsScheduler {
    /// FSRS decay parameter (FSRS-6 uses 0.1542).
    decay: f32,
    /// Initial stability values for each grade (Again, Hard, Good, Easy).
    initial_stability: [f32; 4],
    /// Initial difficulty values for each grade.
    initial_difficulty: [f32; 4],
}

impl FsrsScheduler {
    /// Create a new scheduler with FSRS-6 default parameters.
    pub fn new() -> Self {
        Self {
            // FSRS-6 decay constant
            decay: fsrs::FSRS6_DEFAULT_DECAY,
            // Initial stability by grade (from FSRS default parameters w[0..4])
            // These represent days for retrievability to drop to 90%
            initial_stability: [
                fsrs::DEFAULT_PARAMETERS[0], // Again: ~0.212 days
                fsrs::DEFAULT_PARAMETERS[1], // Hard: ~1.29 days
                fsrs::DEFAULT_PARAMETERS[2], // Good: ~2.31 days
                fsrs::DEFAULT_PARAMETERS[3], // Easy: ~8.30 days
            ],
            // Initial difficulty by grade (mapped from algorithm)
            // Lower grades = higher difficulty
            initial_difficulty: [
                8.0, // Again: high difficulty
                6.5, // Hard: above average
                5.0, // Good: average
                3.5, // Easy: below average
            ],
        }
    }

    /// Create a scheduler with custom parameters.
    pub fn with_params(decay: f32, initial_stability: [f32; 4], initial_difficulty: [f32; 4]) -> Self {
        Self {
            decay,
            initial_stability,
            initial_difficulty,
        }
    }

    /// Calculate current retrievability (probability of recall).
    ///
    /// Uses the FSRS power forgetting curve formula:
    /// R(t) = (1 + factor * t/S)^(-decay)
    /// where factor = 0.9^(1/-decay) - 1
    ///
    /// # Arguments
    /// * `state` - Current FSRS memory state
    /// * `now` - Current timestamp
    ///
    /// # Returns
    /// Retrievability as probability [0.0, 1.0]
    pub fn current_retrievability(&self, state: &FsrsState, now: DateTime<Utc>) -> f32 {
        // If no last review, return 1.0 (just learned)
        let last_review = match state.last_review {
            Some(lr) => lr,
            None => return 1.0,
        };

        // Calculate days elapsed
        let elapsed = now.signed_duration_since(last_review);
        let days_elapsed = elapsed.num_seconds() as f32 / 86400.0;

        if days_elapsed <= 0.0 {
            return 1.0;
        }

        // If stability is 0 or very small, return 0 (no memory formed)
        if state.stability <= 0.001 {
            return 0.0;
        }

        // Use fsrs crate's retrievability calculation
        let mem_state = state.to_memory_state();
        fsrs::current_retrievability(mem_state, days_elapsed, self.decay)
    }

    /// Calculate retrievability given explicit days elapsed.
    ///
    /// Useful for testing or when elapsed time is already known.
    pub fn retrievability(&self, stability: f32, days_elapsed: f32) -> f32 {
        if days_elapsed <= 0.0 {
            return 1.0;
        }
        if stability <= 0.001 {
            return 0.0;
        }

        let mem_state = fsrs::MemoryState {
            stability,
            difficulty: 5.0, // Difficulty doesn't affect retrievability calculation
        };
        fsrs::current_retrievability(mem_state, days_elapsed, self.decay)
    }

    /// Create initial FSRS state for a new memory based on grade.
    ///
    /// When a memory is first created, the initial grade affects:
    /// - Initial stability (how long until 90% recall probability)
    /// - Initial difficulty (how hard the memory is to maintain)
    pub fn initial_state(&self, grade: Grade) -> FsrsState {
        let idx = grade.to_rating() as usize - 1; // Convert 1-4 to 0-3

        FsrsState {
            stability: self.initial_stability[idx],
            difficulty: self.initial_difficulty[idx],
            last_review: Some(Utc::now()),
            reps: 1,
            lapses: if grade == Grade::Again { 1 } else { 0 },
        }
    }

    /// Get the decay parameter.
    pub fn decay(&self) -> f32 {
        self.decay
    }

    /// Process a review and update memory state based on grade.
    ///
    /// This is the core FSRS algorithm: based on the grade (quality of recall),
    /// the memory's stability and difficulty are updated.
    ///
    /// # Arguments
    /// * `state` - Current FSRS memory state
    /// * `grade` - Quality of recall (Again, Hard, Good, Easy)
    /// * `now` - Current timestamp
    ///
    /// # Returns
    /// New FSRS state with updated stability, difficulty, reps, and lapses
    pub fn process_review(&self, state: &FsrsState, grade: Grade, now: DateTime<Utc>) -> FsrsState {
        // Calculate current retrievability
        let retrievability = self.current_retrievability(state, now);

        // Get the rating as u32 for FSRS calculations (1-4)
        let rating = grade.to_rating() as u32;

        // Calculate new stability based on FSRS-6 algorithm
        // For failed recall (Again): stability decreases significantly
        // For successful recall: stability increases based on grade
        let new_stability = if grade == Grade::Again {
            // Lapse case: stability reset to a fraction
            // FSRS uses: S' = w[11] * D^(-w[12]) * ((S+1)^w[13] - 1) * e^(w[14]*(1-R))
            // Simplified: multiply by lapse factor based on retrievability
            let lapse_factor = 0.3 * (1.0 - retrievability).max(0.1);
            (state.stability * lapse_factor).max(0.1)
        } else {
            // Successful recall: stability increases
            // FSRS: S' = S * (e^(w[8]) * (11 - D) * S^(-w[9]) * (e^(w[10]*(1-R)) - 1) * grade_modifier + 1)
            // Simplified model using grade-based multipliers
            let grade_multiplier = match grade {
                Grade::Hard => 1.2,
                Grade::Good => 1.5,
                Grade::Easy => 2.0,
                Grade::Again => unreachable!(),
            };

            // Lower retrievability at review time = higher stability gain (desirable difficulty)
            // But cap the boost to prevent extreme values
            let retrievability_boost = ((1.0 - retrievability) * 0.5 + 1.0).min(1.5);

            state.stability * grade_multiplier * retrievability_boost
        };

        // Update difficulty based on grade
        // Again increases difficulty, Easy decreases it
        // FSRS: D' = w[7] * D_0(4) + (1 - w[7]) * (D - w[6] * (G - 3))
        // Simplified: adjust by grade distance from "Good" (3)
        let grade_offset = rating as f32 - 3.0; // -2, -1, 0, +1
        let difficulty_change = -grade_offset * 0.5; // Positive for Again, negative for Easy
        let new_difficulty = (state.difficulty + difficulty_change).clamp(1.0, 10.0);

        // Update lapses count (only Again counts as lapse)
        let new_lapses = if grade == Grade::Again {
            state.lapses + 1
        } else {
            state.lapses
        };

        FsrsState {
            stability: new_stability,
            difficulty: new_difficulty,
            last_review: Some(now),
            reps: state.reps + 1,
            lapses: new_lapses,
        }
    }

    /// Promote a memory's stability (convenience wrapper around process_review).
    ///
    /// Used when a memory is explicitly promoted/reinforced with a given grade.
    ///
    /// # Arguments
    /// * `state` - Current FSRS memory state
    /// * `grade` - Quality grade (typically Good or Easy for promotion)
    /// * `now` - Current timestamp
    pub fn promote(&self, state: &FsrsState, grade: Grade, now: DateTime<Utc>) -> FsrsState {
        self.process_review(state, grade, now)
    }

    /// Demote a memory's stability (equivalent to Again grade).
    ///
    /// Used when a memory should be weakened/forgotten.
    ///
    /// # Arguments
    /// * `state` - Current FSRS memory state
    /// * `now` - Current timestamp
    pub fn demote(&self, state: &FsrsState, now: DateTime<Utc>) -> FsrsState {
        self.process_review(state, Grade::Again, now)
    }
}

impl Default for FsrsScheduler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;

    #[test]
    fn test_retrievability_at_zero_elapsed() {
        let scheduler = FsrsScheduler::new();

        // At t=0, retrievability should be 1.0 (perfect recall)
        let r = scheduler.retrievability(10.0, 0.0);
        assert!((r - 1.0).abs() < 0.001, "Retrievability at t=0 should be 1.0, got {}", r);
    }

    #[test]
    fn test_retrievability_at_stability() {
        let scheduler = FsrsScheduler::new();

        // At t=stability, retrievability should be ~0.9 (90%)
        // This is the definition of stability in FSRS
        let stability = 10.0;
        let r = scheduler.retrievability(stability, stability);

        // FSRS-6 uses a power forgetting curve, at t=S, R should be ~0.9
        assert!(r > 0.85 && r < 0.95, "Retrievability at t=S should be ~0.9, got {}", r);
    }

    #[test]
    fn test_retrievability_decays_over_time() {
        let scheduler = FsrsScheduler::new();
        let stability = 10.0;

        let r1 = scheduler.retrievability(stability, 1.0);
        let r2 = scheduler.retrievability(stability, 5.0);
        let r3 = scheduler.retrievability(stability, 10.0);
        let r4 = scheduler.retrievability(stability, 30.0);

        // Retrievability should decrease over time
        assert!(r1 > r2, "R(1) should be > R(5): {} > {}", r1, r2);
        assert!(r2 > r3, "R(5) should be > R(10): {} > {}", r2, r3);
        assert!(r3 > r4, "R(10) should be > R(30): {} > {}", r3, r4);

        // But never negative
        assert!(r4 > 0.0, "Retrievability should never be negative");
    }

    #[test]
    fn test_retrievability_with_zero_stability() {
        let scheduler = FsrsScheduler::new();

        // Zero stability means no memory formed
        let r = scheduler.retrievability(0.0, 1.0);
        assert_eq!(r, 0.0, "Zero stability should give 0 retrievability");
    }

    #[test]
    fn test_current_retrievability_no_last_review() {
        let scheduler = FsrsScheduler::new();
        let state = FsrsState::new();

        // No last review means just learned, return 1.0
        let r = scheduler.current_retrievability(&state, Utc::now());
        assert_eq!(r, 1.0, "No last review should give 1.0 retrievability");
    }

    #[test]
    fn test_current_retrievability_with_time_elapsed() {
        let scheduler = FsrsScheduler::new();

        let mut state = FsrsState::new();
        state.stability = 10.0;
        state.last_review = Some(Utc::now() - Duration::days(5));

        let r = scheduler.current_retrievability(&state, Utc::now());

        // 5 days elapsed with 10 day stability should give R > 0.9 but < 1.0
        assert!(r > 0.9 && r < 1.0, "R after 5 days with S=10 should be ~0.95, got {}", r);
    }

    #[test]
    fn test_initial_state_grade_again() {
        let scheduler = FsrsScheduler::new();
        let state = scheduler.initial_state(Grade::Again);

        // Again grade should have low stability, high difficulty
        assert!(state.stability < 1.0, "Again stability should be < 1 day");
        assert!(state.difficulty > 7.0, "Again difficulty should be high");
        assert_eq!(state.reps, 1);
        assert_eq!(state.lapses, 1, "Again should count as a lapse");
    }

    #[test]
    fn test_initial_state_grade_easy() {
        let scheduler = FsrsScheduler::new();
        let state = scheduler.initial_state(Grade::Easy);

        // Easy grade should have high stability, low difficulty
        assert!(state.stability > 5.0, "Easy stability should be > 5 days");
        assert!(state.difficulty < 5.0, "Easy difficulty should be below average");
        assert_eq!(state.reps, 1);
        assert_eq!(state.lapses, 0, "Easy should not count as a lapse");
    }

    #[test]
    fn test_initial_state_stability_ordering() {
        let scheduler = FsrsScheduler::new();

        let again = scheduler.initial_state(Grade::Again);
        let hard = scheduler.initial_state(Grade::Hard);
        let good = scheduler.initial_state(Grade::Good);
        let easy = scheduler.initial_state(Grade::Easy);

        // Higher grades should have higher stability
        assert!(again.stability < hard.stability);
        assert!(hard.stability < good.stability);
        assert!(good.stability < easy.stability);
    }

    #[test]
    fn test_initial_state_difficulty_ordering() {
        let scheduler = FsrsScheduler::new();

        let again = scheduler.initial_state(Grade::Again);
        let hard = scheduler.initial_state(Grade::Hard);
        let good = scheduler.initial_state(Grade::Good);
        let easy = scheduler.initial_state(Grade::Easy);

        // Higher grades should have lower difficulty
        assert!(again.difficulty > hard.difficulty);
        assert!(hard.difficulty > good.difficulty);
        assert!(good.difficulty > easy.difficulty);
    }

    #[test]
    fn test_higher_stability_decays_slower() {
        let scheduler = FsrsScheduler::new();

        // Two memories, same elapsed time, different stability
        let r_low_stability = scheduler.retrievability(2.0, 5.0);
        let r_high_stability = scheduler.retrievability(20.0, 5.0);

        // Higher stability should have higher retrievability
        assert!(
            r_high_stability > r_low_stability,
            "Higher stability should decay slower: {} > {}",
            r_high_stability,
            r_low_stability
        );
    }

    // ============================================================
    // Tests for process_review, promote, demote
    // ============================================================

    #[test]
    fn test_process_review_again_decreases_stability() {
        let scheduler = FsrsScheduler::new();

        // Start with a state that has good stability
        let mut state = FsrsState::new();
        state.stability = 10.0;
        state.difficulty = 5.0;
        state.last_review = Some(Utc::now() - Duration::days(5));
        state.reps = 5;

        let initial_stability = state.stability;

        // Process a review with Again grade (failed recall)
        let new_state = scheduler.process_review(&state, Grade::Again, Utc::now());

        // Again should decrease stability significantly
        assert!(
            new_state.stability < initial_stability,
            "Again grade should decrease stability: {} < {}",
            new_state.stability,
            initial_stability
        );
        // Lapses should increase
        assert_eq!(new_state.lapses, state.lapses + 1);
        // Reps should still increase (it's still a review)
        assert_eq!(new_state.reps, state.reps + 1);
    }

    #[test]
    fn test_process_review_good_increases_stability() {
        let scheduler = FsrsScheduler::new();

        let mut state = FsrsState::new();
        state.stability = 5.0;
        state.difficulty = 5.0;
        state.last_review = Some(Utc::now() - Duration::days(3));
        state.reps = 3;

        let initial_stability = state.stability;

        // Process a review with Good grade
        let new_state = scheduler.process_review(&state, Grade::Good, Utc::now());

        // Good should increase stability
        assert!(
            new_state.stability > initial_stability,
            "Good grade should increase stability: {} > {}",
            new_state.stability,
            initial_stability
        );
        // No lapse for successful recall
        assert_eq!(new_state.lapses, state.lapses);
        // Reps should increase
        assert_eq!(new_state.reps, state.reps + 1);
    }

    #[test]
    fn test_process_review_hard_increases_stability_less() {
        let scheduler = FsrsScheduler::new();

        let mut state = FsrsState::new();
        state.stability = 5.0;
        state.difficulty = 5.0;
        state.last_review = Some(Utc::now() - Duration::days(3));
        state.reps = 3;

        let initial_stability = state.stability;

        // Process with Hard grade
        let hard_state = scheduler.process_review(&state, Grade::Hard, Utc::now());
        // Process with Good grade (from same initial state)
        let good_state = scheduler.process_review(&state, Grade::Good, Utc::now());

        // Hard should still increase stability (successful recall)
        assert!(
            hard_state.stability > initial_stability,
            "Hard grade should increase stability: {} > {}",
            hard_state.stability,
            initial_stability
        );
        // But less than Good
        assert!(
            hard_state.stability < good_state.stability,
            "Hard should increase less than Good: {} < {}",
            hard_state.stability,
            good_state.stability
        );
    }

    #[test]
    fn test_process_review_easy_increases_stability_most() {
        let scheduler = FsrsScheduler::new();

        let mut state = FsrsState::new();
        state.stability = 5.0;
        state.difficulty = 5.0;
        state.last_review = Some(Utc::now() - Duration::days(3));
        state.reps = 3;

        // Process with all grades
        let good_state = scheduler.process_review(&state, Grade::Good, Utc::now());
        let easy_state = scheduler.process_review(&state, Grade::Easy, Utc::now());

        // Easy should increase stability more than Good
        assert!(
            easy_state.stability > good_state.stability,
            "Easy should increase more than Good: {} > {}",
            easy_state.stability,
            good_state.stability
        );
    }

    #[test]
    fn test_process_review_updates_difficulty() {
        let scheduler = FsrsScheduler::new();

        let mut state = FsrsState::new();
        state.stability = 5.0;
        state.difficulty = 5.0;
        state.last_review = Some(Utc::now() - Duration::days(3));
        state.reps = 3;

        // Again should increase difficulty (memory is harder)
        let again_state = scheduler.process_review(&state, Grade::Again, Utc::now());
        assert!(
            again_state.difficulty > state.difficulty,
            "Again should increase difficulty"
        );

        // Easy should decrease difficulty (memory is easier)
        let easy_state = scheduler.process_review(&state, Grade::Easy, Utc::now());
        assert!(
            easy_state.difficulty < state.difficulty,
            "Easy should decrease difficulty"
        );
    }

    #[test]
    fn test_promote_increases_stability() {
        let scheduler = FsrsScheduler::new();

        let mut state = FsrsState::new();
        state.stability = 5.0;
        state.difficulty = 5.0;
        state.last_review = Some(Utc::now() - Duration::days(2));
        state.reps = 2;

        // Promote with Good grade
        let promoted = scheduler.promote(&state, Grade::Good, Utc::now());

        assert!(
            promoted.stability > state.stability,
            "Promote should increase stability"
        );
    }

    #[test]
    fn test_demote_decreases_stability() {
        let scheduler = FsrsScheduler::new();

        let mut state = FsrsState::new();
        state.stability = 10.0;
        state.difficulty = 5.0;
        state.last_review = Some(Utc::now() - Duration::days(5));
        state.reps = 5;

        // Demote (equivalent to Again)
        let demoted = scheduler.demote(&state, Utc::now());

        assert!(
            demoted.stability < state.stability,
            "Demote should decrease stability: {} < {}",
            demoted.stability,
            state.stability
        );
        assert_eq!(demoted.lapses, state.lapses + 1, "Demote should count as lapse");
    }
}
