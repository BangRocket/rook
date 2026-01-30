//! Memory lifecycle events (INT-10 through INT-13)
//!
//! Events are emitted when memories are created, updated, deleted, or accessed.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Memory lifecycle events
///
/// These events are emitted to the event bus when memory operations occur,
/// allowing external systems to react via webhooks or internal subscribers.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MemoryLifecycleEvent {
    /// Memory was created (INT-10)
    Created(MemoryCreatedEvent),
    /// Memory was updated (INT-11)
    Updated(MemoryUpdatedEvent),
    /// Memory was deleted (INT-12)
    Deleted(MemoryDeletedEvent),
    /// Memory was accessed (INT-13)
    Accessed(MemoryAccessedEvent),
}

impl MemoryLifecycleEvent {
    /// Get the event type as a string for filtering
    pub fn event_type(&self) -> &'static str {
        match self {
            Self::Created(_) => "memory.created",
            Self::Updated(_) => "memory.updated",
            Self::Deleted(_) => "memory.deleted",
            Self::Accessed(_) => "memory.accessed",
        }
    }

    /// Get the memory ID this event relates to
    pub fn memory_id(&self) -> &str {
        match self {
            Self::Created(e) => &e.memory_id,
            Self::Updated(e) => &e.memory_id,
            Self::Deleted(e) => &e.memory_id,
            Self::Accessed(e) => &e.memory_id,
        }
    }

    /// Get the timestamp of this event
    pub fn timestamp(&self) -> DateTime<Utc> {
        match self {
            Self::Created(e) => e.timestamp,
            Self::Updated(e) => e.timestamp,
            Self::Deleted(e) => e.timestamp,
            Self::Accessed(e) => e.timestamp,
        }
    }

    /// Get user ID if present
    pub fn user_id(&self) -> Option<&str> {
        match self {
            Self::Created(e) => e.user_id.as_deref(),
            Self::Updated(e) => e.user_id.as_deref(),
            Self::Deleted(e) => e.user_id.as_deref(),
            Self::Accessed(e) => e.user_id.as_deref(),
        }
    }
}

/// Event payload for memory creation (INT-10)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryCreatedEvent {
    /// Unique event ID
    pub event_id: String,
    /// Memory that was created
    pub memory_id: String,
    /// User who owns the memory (if scoped)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_id: Option<String>,
    /// Content of the memory
    pub content: String,
    /// Metadata attached to memory
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
    /// When the event occurred
    pub timestamp: DateTime<Utc>,
}

/// Event payload for memory update (INT-11)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryUpdatedEvent {
    /// Unique event ID
    pub event_id: String,
    /// Memory that was updated
    pub memory_id: String,
    /// User who owns the memory (if scoped)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_id: Option<String>,
    /// Previous content (for diff purposes)
    pub old_content: String,
    /// New content
    pub new_content: String,
    /// What type of update occurred
    pub update_type: UpdateType,
    /// New version number
    pub version: u32,
    /// When the event occurred
    pub timestamp: DateTime<Utc>,
}

/// Type of update that occurred
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UpdateType {
    /// Content was modified
    Content,
    /// Metadata was modified
    Metadata,
    /// FSRS state was updated (after review)
    FsrsState,
    /// Memory was superseded
    Superseded,
    /// Memory was merged
    Merged,
}

/// Event payload for memory deletion (INT-12)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryDeletedEvent {
    /// Unique event ID
    pub event_id: String,
    /// Memory that was deleted
    pub memory_id: String,
    /// User who owned the memory (if scoped)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_id: Option<String>,
    /// Whether it was a soft delete (archived) or hard delete
    pub soft_delete: bool,
    /// Reason for deletion (if provided)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    /// When the event occurred
    pub timestamp: DateTime<Utc>,
}

/// Event payload for memory access (INT-13)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryAccessedEvent {
    /// Unique event ID
    pub event_id: String,
    /// Memory that was accessed
    pub memory_id: String,
    /// User context (if scoped)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_id: Option<String>,
    /// Type of access
    pub access_type: AccessType,
    /// Search query if access was via search
    #[serde(skip_serializing_if = "Option::is_none")]
    pub query: Option<String>,
    /// Relevance score if access was via search
    #[serde(skip_serializing_if = "Option::is_none")]
    pub relevance_score: Option<f32>,
    /// When the event occurred
    pub timestamp: DateTime<Utc>,
}

/// Type of memory access
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AccessType {
    /// Direct retrieval by ID
    DirectGet,
    /// Retrieved via search
    Search,
    /// Retrieved via spreading activation
    SpreadingActivation,
    /// Used in response generation
    UsedInResponse,
    /// Reviewed (FSRS interaction)
    Reviewed,
}

impl MemoryCreatedEvent {
    pub fn new(memory_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            event_id: uuid::Uuid::new_v4().to_string(),
            memory_id: memory_id.into(),
            user_id: None,
            content: content.into(),
            metadata: HashMap::new(),
            timestamp: Utc::now(),
        }
    }

    pub fn with_user(mut self, user_id: impl Into<String>) -> Self {
        self.user_id = Some(user_id.into());
        self
    }

    pub fn with_metadata(mut self, metadata: HashMap<String, serde_json::Value>) -> Self {
        self.metadata = metadata;
        self
    }
}

impl MemoryUpdatedEvent {
    pub fn new(
        memory_id: impl Into<String>,
        old_content: impl Into<String>,
        new_content: impl Into<String>,
        update_type: UpdateType,
        version: u32,
    ) -> Self {
        Self {
            event_id: uuid::Uuid::new_v4().to_string(),
            memory_id: memory_id.into(),
            user_id: None,
            old_content: old_content.into(),
            new_content: new_content.into(),
            update_type,
            version,
            timestamp: Utc::now(),
        }
    }

    pub fn with_user(mut self, user_id: impl Into<String>) -> Self {
        self.user_id = Some(user_id.into());
        self
    }
}

impl MemoryDeletedEvent {
    pub fn new(memory_id: impl Into<String>, soft_delete: bool) -> Self {
        Self {
            event_id: uuid::Uuid::new_v4().to_string(),
            memory_id: memory_id.into(),
            user_id: None,
            soft_delete,
            reason: None,
            timestamp: Utc::now(),
        }
    }

    pub fn with_user(mut self, user_id: impl Into<String>) -> Self {
        self.user_id = Some(user_id.into());
        self
    }

    pub fn with_reason(mut self, reason: impl Into<String>) -> Self {
        self.reason = Some(reason.into());
        self
    }
}

impl MemoryAccessedEvent {
    pub fn new(memory_id: impl Into<String>, access_type: AccessType) -> Self {
        Self {
            event_id: uuid::Uuid::new_v4().to_string(),
            memory_id: memory_id.into(),
            user_id: None,
            access_type,
            query: None,
            relevance_score: None,
            timestamp: Utc::now(),
        }
    }

    pub fn with_user(mut self, user_id: impl Into<String>) -> Self {
        self.user_id = Some(user_id.into());
        self
    }

    pub fn with_search_context(mut self, query: impl Into<String>, score: f32) -> Self {
        self.query = Some(query.into());
        self.relevance_score = Some(score);
        self
    }
}
