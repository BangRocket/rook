//! Automatic strength signal system for memory feedback.
//!
//! This module provides a signal-based approach to updating memory strength
//! based on user interactions and system events.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::types::Grade;

/// Signals that trigger automatic memory strength adjustments
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StrengthSignal {
    /// Memory was used in generating a response
    /// Effect: Promote with Good grade (memory is relevant and accurate)
    UsedInResponse {
        memory_id: String,
        /// Optional context of how memory was used
        context: Option<String>,
    },

    /// User explicitly corrected information
    /// Effect: Demote old with Again grade (was wrong)
    UserCorrection {
        old_memory_id: String,
        /// New content that replaces old
        new_content: String,
    },

    /// User confirmed information is correct
    /// Effect: Promote with Easy grade (user verified accuracy)
    UserConfirmation { memory_id: String },

    /// Contradiction resolved between two memories
    /// Effect: Boost winner with Good, demote loser with Hard
    Contradiction {
        winner_id: String,
        loser_id: String,
    },

    /// Memory was retrieved but not used
    /// Effect: Neutral - neither promote nor demote
    RetrievedNotUsed { memory_id: String },

    /// User explicitly marked memory as incorrect/outdated
    /// Effect: Strong demote with Again grade
    MarkedIncorrect {
        memory_id: String,
        reason: Option<String>,
    },

    /// User explicitly marked memory as important
    /// Effect: Mark as key memory (always retrieve)
    MarkedImportant { memory_id: String },
}

impl StrengthSignal {
    /// Get the grade to apply for a given memory ID
    /// Returns (memory_id, grade) pairs to apply
    pub fn to_grade_updates(&self) -> Vec<(String, Grade)> {
        match self {
            StrengthSignal::UsedInResponse { memory_id, .. } => {
                vec![(memory_id.clone(), Grade::Good)]
            }
            StrengthSignal::UserCorrection { old_memory_id, .. } => {
                vec![(old_memory_id.clone(), Grade::Again)]
            }
            StrengthSignal::UserConfirmation { memory_id } => {
                vec![(memory_id.clone(), Grade::Easy)]
            }
            StrengthSignal::Contradiction { winner_id, loser_id } => {
                vec![
                    (winner_id.clone(), Grade::Good),
                    (loser_id.clone(), Grade::Hard),
                ]
            }
            StrengthSignal::RetrievedNotUsed { .. } => {
                // No grade change for neutral retrieval
                vec![]
            }
            StrengthSignal::MarkedIncorrect { memory_id, .. } => {
                vec![(memory_id.clone(), Grade::Again)]
            }
            StrengthSignal::MarkedImportant { .. } => {
                // Key memory marking is handled separately, not via grades
                vec![]
            }
        }
    }
}

/// Processes strength signals and applies them to memory state
///
/// This processor collects signals and batches the grade updates.
/// Integration with actual FSRS scheduler happens via CognitiveStore.
pub struct StrengthSignalProcessor {
    /// Pending grade updates (memory_id -> list of grades to apply)
    pending_updates: HashMap<String, Vec<Grade>>,
    /// Memories to mark as key (always retrieve)
    pending_key_marks: Vec<String>,
}

impl StrengthSignalProcessor {
    pub fn new() -> Self {
        Self {
            pending_updates: HashMap::new(),
            pending_key_marks: Vec::new(),
        }
    }

    /// Process a strength signal
    pub fn process(&mut self, signal: StrengthSignal) {
        // Check for key memory marking first
        if let StrengthSignal::MarkedImportant { memory_id } = &signal {
            self.pending_key_marks.push(memory_id.clone());
        }

        // Collect grade updates
        for (memory_id, grade) in signal.to_grade_updates() {
            self.pending_updates
                .entry(memory_id)
                .or_insert_with(Vec::new)
                .push(grade);
        }
    }

    /// Get pending grade updates
    /// Returns map of memory_id -> consolidated grade
    ///
    /// When multiple grades are pending for same memory, uses the most recent.
    /// Future enhancement: could average or apply all in sequence.
    pub fn get_pending_updates(&self) -> HashMap<String, Grade> {
        self.pending_updates
            .iter()
            .filter_map(|(id, grades)| grades.last().map(|g| (id.clone(), *g)))
            .collect()
    }

    /// Get memories to mark as key
    pub fn get_pending_key_marks(&self) -> &[String] {
        &self.pending_key_marks
    }

    /// Clear all pending updates (call after applying)
    pub fn clear(&mut self) {
        self.pending_updates.clear();
        self.pending_key_marks.clear();
    }

    /// Check if there are pending updates
    pub fn has_pending(&self) -> bool {
        !self.pending_updates.is_empty() || !self.pending_key_marks.is_empty()
    }
}

impl Default for StrengthSignalProcessor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_used_in_response_gives_good() {
        let signal = StrengthSignal::UsedInResponse {
            memory_id: "mem1".to_string(),
            context: Some("answered question about preferences".to_string()),
        };

        let updates = signal.to_grade_updates();
        assert_eq!(updates.len(), 1);
        assert_eq!(updates[0], ("mem1".to_string(), Grade::Good));
    }

    #[test]
    fn test_user_correction_gives_again() {
        let signal = StrengthSignal::UserCorrection {
            old_memory_id: "mem1".to_string(),
            new_content: "corrected info".to_string(),
        };

        let updates = signal.to_grade_updates();
        assert_eq!(updates.len(), 1);
        assert_eq!(updates[0], ("mem1".to_string(), Grade::Again));
    }

    #[test]
    fn test_user_confirmation_gives_easy() {
        let signal = StrengthSignal::UserConfirmation {
            memory_id: "mem1".to_string(),
        };

        let updates = signal.to_grade_updates();
        assert_eq!(updates.len(), 1);
        assert_eq!(updates[0], ("mem1".to_string(), Grade::Easy));
    }

    #[test]
    fn test_contradiction_gives_good_and_hard() {
        let signal = StrengthSignal::Contradiction {
            winner_id: "mem1".to_string(),
            loser_id: "mem2".to_string(),
        };

        let updates = signal.to_grade_updates();
        assert_eq!(updates.len(), 2);
        assert!(updates.contains(&("mem1".to_string(), Grade::Good)));
        assert!(updates.contains(&("mem2".to_string(), Grade::Hard)));
    }

    #[test]
    fn test_retrieved_not_used_gives_nothing() {
        let signal = StrengthSignal::RetrievedNotUsed {
            memory_id: "mem1".to_string(),
        };

        let updates = signal.to_grade_updates();
        assert!(updates.is_empty());
    }

    #[test]
    fn test_marked_incorrect_gives_again() {
        let signal = StrengthSignal::MarkedIncorrect {
            memory_id: "mem1".to_string(),
            reason: Some("outdated".to_string()),
        };

        let updates = signal.to_grade_updates();
        assert_eq!(updates.len(), 1);
        assert_eq!(updates[0], ("mem1".to_string(), Grade::Again));
    }

    #[test]
    fn test_marked_important_gives_nothing() {
        let signal = StrengthSignal::MarkedImportant {
            memory_id: "mem1".to_string(),
        };

        let updates = signal.to_grade_updates();
        assert!(updates.is_empty());
    }

    #[test]
    fn test_processor_collects_updates() {
        let mut processor = StrengthSignalProcessor::new();

        processor.process(StrengthSignal::UsedInResponse {
            memory_id: "mem1".to_string(),
            context: None,
        });
        processor.process(StrengthSignal::UserConfirmation {
            memory_id: "mem2".to_string(),
        });

        let updates = processor.get_pending_updates();
        assert_eq!(updates.len(), 2);
        assert_eq!(updates.get("mem1"), Some(&Grade::Good));
        assert_eq!(updates.get("mem2"), Some(&Grade::Easy));
    }

    #[test]
    fn test_processor_handles_key_marks() {
        let mut processor = StrengthSignalProcessor::new();

        processor.process(StrengthSignal::MarkedImportant {
            memory_id: "mem1".to_string(),
        });

        assert_eq!(processor.get_pending_key_marks(), &["mem1".to_string()]);
    }

    #[test]
    fn test_processor_consolidates_multiple_updates() {
        let mut processor = StrengthSignalProcessor::new();

        // Same memory gets multiple updates
        processor.process(StrengthSignal::UsedInResponse {
            memory_id: "mem1".to_string(),
            context: None,
        });
        processor.process(StrengthSignal::UserConfirmation {
            memory_id: "mem1".to_string(),
        });

        let updates = processor.get_pending_updates();
        assert_eq!(updates.len(), 1);
        // Most recent wins
        assert_eq!(updates.get("mem1"), Some(&Grade::Easy));
    }

    #[test]
    fn test_processor_clear() {
        let mut processor = StrengthSignalProcessor::new();

        processor.process(StrengthSignal::UsedInResponse {
            memory_id: "mem1".to_string(),
            context: None,
        });
        processor.process(StrengthSignal::MarkedImportant {
            memory_id: "mem2".to_string(),
        });

        assert!(processor.has_pending());
        processor.clear();
        assert!(!processor.has_pending());
    }
}
