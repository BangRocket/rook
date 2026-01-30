//! Intention system for proactive memory behaviors.
//!
//! Intentions allow memories to trigger actions when specific conditions are met:
//! - `KeywordMention`: fires when keywords appear in conversation (INT-02)
//! - `TopicDiscussed`: fires when semantically similar topics are discussed (INT-03)
//! - `TimeElapsed`: fires after a duration since creation/last fire (INT-04)
//! - `ScheduledTime`: fires at specific datetime or cron schedule (INT-05)
//!
//! # Example
//!
//! ```
//! use rook_core::intentions::{Intention, TriggerCondition, IntentionAction};
//!
//! // Create an intention that fires when "rust" is mentioned
//! let intention = Intention::new(
//!     "Rust mention alert",
//!     TriggerCondition::keyword(vec!["rust".to_string(), "programming".to_string()]),
//!     IntentionAction::SurfaceMemory { boost: 1.5 },
//! );
//!
//! // Create an intention that fires 24 hours after creation
//! let reminder = Intention::new(
//!     "Daily reminder",
//!     TriggerCondition::time_elapsed(chrono::Duration::hours(24)),
//!     IntentionAction::Log { message: "Time to review!".to_string() },
//! );
//! ```

mod bloom;
mod store;
mod triggers;
mod types;

pub use store::{IntentionStore, SqliteIntentionStore};
pub use triggers::{ActionResult, FiredIntention, TriggerReason};
pub use types::{Intention, IntentionAction, TriggerCondition};
