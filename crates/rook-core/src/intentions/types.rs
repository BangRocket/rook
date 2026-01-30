//! Intention types for proactive memory behaviors.
//!
//! This module defines the core types for the intention system:
//! - `Intention`: A proactive memory behavior with trigger and action
//! - `TriggerCondition`: Conditions that cause an intention to fire
//! - `IntentionAction`: Actions to perform when an intention fires

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// An intention that can fire when conditions are met (INT-01)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Intention {
    /// Unique identifier
    pub id: Uuid,
    /// Human-readable name
    pub name: String,
    /// Memory ID this intention is associated with (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_id: Option<String>,
    /// User scope (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_id: Option<String>,
    /// Condition that triggers this intention
    pub trigger: TriggerCondition,
    /// Action to take when triggered
    pub action: IntentionAction,
    /// When this intention expires (None = never)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expires_at: Option<DateTime<Utc>>,
    /// Whether this intention is currently active
    pub active: bool,
    /// When this intention was created
    pub created_at: DateTime<Utc>,
    /// Last time this intention fired (if ever)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_fired_at: Option<DateTime<Utc>>,
    /// Number of times this intention has fired
    pub fire_count: u32,
    /// Maximum times to fire (None = unlimited)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_fires: Option<u32>,
    /// Custom metadata
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

impl Intention {
    /// Create a new intention with the given trigger and action
    pub fn new(name: impl Into<String>, trigger: TriggerCondition, action: IntentionAction) -> Self {
        Self {
            id: Uuid::new_v4(),
            name: name.into(),
            memory_id: None,
            user_id: None,
            trigger,
            action,
            expires_at: None,
            active: true,
            created_at: Utc::now(),
            last_fired_at: None,
            fire_count: 0,
            max_fires: None,
            metadata: HashMap::new(),
        }
    }

    /// Check if this intention has expired
    pub fn is_expired(&self) -> bool {
        self.expires_at.is_some_and(|exp| Utc::now() > exp)
    }

    /// Check if this intention can still fire
    pub fn can_fire(&self) -> bool {
        if !self.active || self.is_expired() {
            return false;
        }
        self.max_fires.map_or(true, |max| self.fire_count < max)
    }

    /// Builder method to set memory_id
    pub fn with_memory(mut self, memory_id: impl Into<String>) -> Self {
        self.memory_id = Some(memory_id.into());
        self
    }

    /// Builder method to set user_id
    pub fn with_user(mut self, user_id: impl Into<String>) -> Self {
        self.user_id = Some(user_id.into());
        self
    }

    /// Builder method to set expiration
    pub fn expires_at(mut self, dt: DateTime<Utc>) -> Self {
        self.expires_at = Some(dt);
        self
    }

    /// Builder method to set max fires
    pub fn max_fires(mut self, max: u32) -> Self {
        self.max_fires = Some(max);
        self
    }
}

/// Trigger conditions for intentions (INT-02 through INT-05)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TriggerCondition {
    /// Fire when any of the keywords are mentioned in conversation (INT-02)
    KeywordMention {
        /// Keywords to match (case-insensitive)
        keywords: Vec<String>,
        /// Whether to require exact word match (vs substring)
        #[serde(default)]
        exact_match: bool,
    },
    /// Fire when the topic is semantically similar to conversation (INT-03)
    TopicDiscussed {
        /// Topic description for semantic matching
        topic: String,
        /// Pre-computed embedding of the topic (for fast comparison)
        #[serde(skip_serializing_if = "Option::is_none")]
        topic_embedding: Option<Vec<f32>>,
        /// Similarity threshold (0.0-1.0, default 0.75)
        #[serde(default = "default_topic_threshold")]
        threshold: f32,
    },
    /// Fire after specified duration since creation or last fire (INT-04)
    TimeElapsed {
        /// Duration in seconds
        duration_secs: u64,
        /// Whether to repeat after firing
        #[serde(default)]
        recurring: bool,
        /// Reference time (creation time if None, last fire time if recurring)
        #[serde(skip_serializing_if = "Option::is_none")]
        reference_time: Option<DateTime<Utc>>,
    },
    /// Fire at a specific scheduled time (INT-05)
    ScheduledTime {
        /// Scheduled datetime in UTC
        scheduled_at: DateTime<Utc>,
        /// Optional cron expression for recurring schedules
        #[serde(skip_serializing_if = "Option::is_none")]
        cron: Option<String>,
        /// Timezone for cron interpretation
        #[serde(skip_serializing_if = "Option::is_none")]
        timezone: Option<String>,
    },
}

fn default_topic_threshold() -> f32 {
    0.75
}

impl TriggerCondition {
    /// Create a keyword mention trigger
    pub fn keyword(keywords: Vec<String>) -> Self {
        Self::KeywordMention {
            keywords,
            exact_match: false,
        }
    }

    /// Create a topic discussed trigger
    pub fn topic(topic: impl Into<String>) -> Self {
        Self::TopicDiscussed {
            topic: topic.into(),
            topic_embedding: None,
            threshold: default_topic_threshold(),
        }
    }

    /// Create a time elapsed trigger
    pub fn time_elapsed(duration: Duration) -> Self {
        Self::TimeElapsed {
            duration_secs: duration.num_seconds() as u64,
            recurring: false,
            reference_time: None,
        }
    }

    /// Create a scheduled time trigger
    pub fn scheduled_at(dt: DateTime<Utc>) -> Self {
        Self::ScheduledTime {
            scheduled_at: dt,
            cron: None,
            timezone: None,
        }
    }
}

/// Action to perform when intention fires
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum IntentionAction {
    /// Surface the associated memory in search results
    SurfaceMemory {
        /// Boost factor for the memory (1.0 = normal, >1.0 = higher)
        boost: f32,
    },
    /// Send a notification via webhook
    Notify {
        /// Webhook URL to call
        webhook_url: String,
        /// Custom payload to include
        #[serde(skip_serializing_if = "Option::is_none")]
        payload: Option<serde_json::Value>,
    },
    /// Execute a custom callback (for programmatic use)
    Callback {
        /// Callback identifier
        callback_id: String,
        /// Arguments to pass
        #[serde(default)]
        args: HashMap<String, serde_json::Value>,
    },
    /// Log for debugging/testing
    Log {
        message: String,
    },
}

impl Default for IntentionAction {
    fn default() -> Self {
        Self::SurfaceMemory { boost: 1.5 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_intention_creation() {
        let intention = Intention::new(
            "test intention",
            TriggerCondition::keyword(vec!["test".to_string()]),
            IntentionAction::default(),
        );

        assert_eq!(intention.name, "test intention");
        assert!(intention.active);
        assert_eq!(intention.fire_count, 0);
        assert!(intention.can_fire());
    }

    #[test]
    fn test_intention_builder() {
        let intention = Intention::new(
            "test",
            TriggerCondition::topic("machine learning"),
            IntentionAction::default(),
        )
        .with_memory("mem-123")
        .with_user("user-456")
        .max_fires(5);

        assert_eq!(intention.memory_id, Some("mem-123".to_string()));
        assert_eq!(intention.user_id, Some("user-456".to_string()));
        assert_eq!(intention.max_fires, Some(5));
    }

    #[test]
    fn test_intention_can_fire() {
        let mut intention = Intention::new(
            "test",
            TriggerCondition::keyword(vec!["test".to_string()]),
            IntentionAction::default(),
        )
        .max_fires(2);

        assert!(intention.can_fire());

        intention.fire_count = 1;
        assert!(intention.can_fire());

        intention.fire_count = 2;
        assert!(!intention.can_fire());
    }

    #[test]
    fn test_intention_expired() {
        let mut intention = Intention::new(
            "test",
            TriggerCondition::keyword(vec!["test".to_string()]),
            IntentionAction::default(),
        );

        // Not expired by default
        assert!(!intention.is_expired());
        assert!(intention.can_fire());

        // Set expiration in the past
        intention.expires_at = Some(Utc::now() - Duration::hours(1));
        assert!(intention.is_expired());
        assert!(!intention.can_fire());
    }

    #[test]
    fn test_trigger_condition_serialization() {
        let keyword = TriggerCondition::keyword(vec!["rust".to_string(), "programming".to_string()]);
        let json = serde_json::to_string(&keyword).unwrap();
        assert!(json.contains("keyword_mention"));
        assert!(json.contains("rust"));

        let topic = TriggerCondition::topic("machine learning");
        let json = serde_json::to_string(&topic).unwrap();
        assert!(json.contains("topic_discussed"));
        assert!(json.contains("machine learning"));
    }

    #[test]
    fn test_action_serialization() {
        let action = IntentionAction::SurfaceMemory { boost: 2.0 };
        let json = serde_json::to_string(&action).unwrap();
        assert!(json.contains("surface_memory"));
        assert!(json.contains("2.0"));

        let deserialized: IntentionAction = serde_json::from_str(&json).unwrap();
        match deserialized {
            IntentionAction::SurfaceMemory { boost } => assert!((boost - 2.0).abs() < f32::EPSILON),
            _ => panic!("Wrong action type"),
        }
    }
}
