//! Trigger reason and fired intention tracking.
//!
//! This module defines types for tracking why intentions fired:
//! - `TriggerReason`: Why an intention was triggered
//! - `FiredIntention`: Record of an intention that fired
//! - `ActionResult`: Result of executing an intention action

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Reason why an intention fired
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TriggerReason {
    /// Fired due to keyword match
    Keyword {
        /// The keyword that matched
        matched_keyword: String,
        /// Context where keyword appeared
        context: String,
    },
    /// Fired due to topic similarity
    Topic {
        /// Similarity score achieved
        similarity: f32,
        /// The topic that matched
        topic: String,
    },
    /// Fired due to time elapsed
    TimeElapsed {
        /// Time elapsed since reference
        elapsed_secs: u64,
    },
    /// Fired at scheduled time
    ScheduledTime {
        /// The scheduled time that triggered
        scheduled_at: DateTime<Utc>,
    },
}

/// Record of an intention that fired
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FiredIntention {
    /// The intention that fired
    pub intention_id: Uuid,
    /// When it fired
    pub fired_at: DateTime<Utc>,
    /// Why it fired
    pub reason: TriggerReason,
    /// The action that was taken
    pub action_result: ActionResult,
}

/// Result of executing an intention action
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum ActionResult {
    /// Action completed successfully
    Success {
        #[serde(skip_serializing_if = "Option::is_none")]
        details: Option<String>,
    },
    /// Action failed
    Failed {
        error: String,
    },
    /// Action was skipped (e.g., intention expired)
    Skipped {
        reason: String,
    },
}

impl FiredIntention {
    /// Create a new fired intention record
    pub fn new(intention_id: Uuid, reason: TriggerReason, result: ActionResult) -> Self {
        Self {
            intention_id,
            fired_at: Utc::now(),
            reason,
            action_result: result,
        }
    }

    /// Create a successful fired intention
    pub fn success(intention_id: Uuid, reason: TriggerReason) -> Self {
        Self::new(intention_id, reason, ActionResult::Success { details: None })
    }

    /// Create a failed fired intention
    pub fn failed(intention_id: Uuid, reason: TriggerReason, error: impl Into<String>) -> Self {
        Self::new(
            intention_id,
            reason,
            ActionResult::Failed {
                error: error.into(),
            },
        )
    }

    /// Create a skipped fired intention
    pub fn skipped(intention_id: Uuid, reason: TriggerReason, skip_reason: impl Into<String>) -> Self {
        Self::new(
            intention_id,
            reason,
            ActionResult::Skipped {
                reason: skip_reason.into(),
            },
        )
    }
}

impl ActionResult {
    /// Check if the action was successful
    pub fn is_success(&self) -> bool {
        matches!(self, ActionResult::Success { .. })
    }

    /// Check if the action failed
    pub fn is_failed(&self) -> bool {
        matches!(self, ActionResult::Failed { .. })
    }

    /// Check if the action was skipped
    pub fn is_skipped(&self) -> bool {
        matches!(self, ActionResult::Skipped { .. })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trigger_reason_keyword() {
        let reason = TriggerReason::Keyword {
            matched_keyword: "rust".to_string(),
            context: "I love programming in rust".to_string(),
        };

        let json = serde_json::to_string(&reason).unwrap();
        assert!(json.contains("keyword"));
        assert!(json.contains("rust"));

        let deserialized: TriggerReason = serde_json::from_str(&json).unwrap();
        match deserialized {
            TriggerReason::Keyword {
                matched_keyword,
                context,
            } => {
                assert_eq!(matched_keyword, "rust");
                assert!(context.contains("programming"));
            }
            _ => panic!("Wrong reason type"),
        }
    }

    #[test]
    fn test_trigger_reason_topic() {
        let reason = TriggerReason::Topic {
            similarity: 0.85,
            topic: "machine learning".to_string(),
        };

        let json = serde_json::to_string(&reason).unwrap();
        assert!(json.contains("topic"));
        assert!(json.contains("0.85"));
    }

    #[test]
    fn test_fired_intention_creation() {
        let id = Uuid::new_v4();
        let reason = TriggerReason::TimeElapsed { elapsed_secs: 3600 };
        let fired = FiredIntention::success(id, reason);

        assert_eq!(fired.intention_id, id);
        assert!(fired.action_result.is_success());
    }

    #[test]
    fn test_action_result_states() {
        let success = ActionResult::Success {
            details: Some("Completed".to_string()),
        };
        assert!(success.is_success());
        assert!(!success.is_failed());
        assert!(!success.is_skipped());

        let failed = ActionResult::Failed {
            error: "Network error".to_string(),
        };
        assert!(!failed.is_success());
        assert!(failed.is_failed());
        assert!(!failed.is_skipped());

        let skipped = ActionResult::Skipped {
            reason: "Intention expired".to_string(),
        };
        assert!(!skipped.is_success());
        assert!(!skipped.is_failed());
        assert!(skipped.is_skipped());
    }

    #[test]
    fn test_action_result_serialization() {
        let success = ActionResult::Success {
            details: Some("Memory surfaced".to_string()),
        };
        let json = serde_json::to_string(&success).unwrap();
        assert!(json.contains("success"));
        assert!(json.contains("Memory surfaced"));

        let failed = ActionResult::Failed {
            error: "Connection timeout".to_string(),
        };
        let json = serde_json::to_string(&failed).unwrap();
        assert!(json.contains("failed"));
        assert!(json.contains("Connection timeout"));
    }
}
