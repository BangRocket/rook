//! Memory version types for audit trails and point-in-time queries.
//!
//! Provides immutable snapshots of memory state at each mutation,
//! enabling queries like "what did this memory contain last week?"

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Event type that created this version (INT-08)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VersionEventType {
    /// Memory was created
    Created,
    /// Memory content was updated
    ContentUpdated,
    /// Memory metadata was changed
    MetadataUpdated,
    /// FSRS state was updated (after review)
    FsrsUpdated,
    /// Memory was superseded by new information
    Superseded,
    /// Memory was merged with another
    Merged,
    /// Memory was restored from archive
    Restored,
}

impl VersionEventType {
    /// Convert to string for storage
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Created => "created",
            Self::ContentUpdated => "content_updated",
            Self::MetadataUpdated => "metadata_updated",
            Self::FsrsUpdated => "fsrs_updated",
            Self::Superseded => "superseded",
            Self::Merged => "merged",
            Self::Restored => "restored",
        }
    }

    /// Parse from string
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "created" => Some(Self::Created),
            "content_updated" => Some(Self::ContentUpdated),
            "metadata_updated" => Some(Self::MetadataUpdated),
            "fsrs_updated" => Some(Self::FsrsUpdated),
            "superseded" => Some(Self::Superseded),
            "merged" => Some(Self::Merged),
            "restored" => Some(Self::Restored),
            _ => None,
        }
    }
}

/// Snapshot of FSRS cognitive state at a point in time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FsrsStateSnapshot {
    pub stability: f32,
    pub difficulty: f32,
    pub retrievability: f32,
    pub storage_strength: f32,
    pub retrieval_strength: f32,
    pub last_review: Option<DateTime<Utc>>,
}

/// A snapshot of memory state at a point in time (INT-08)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryVersion {
    /// Unique version identifier
    pub version_id: Uuid,
    /// Memory this version belongs to
    pub memory_id: String,
    /// Sequential version number within this memory (1, 2, 3...)
    pub version_number: u32,
    /// Content at this version
    pub content: String,
    /// Metadata at this version
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
    /// FSRS state at this version (if cognitive features enabled)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fsrs_state: Option<FsrsStateSnapshot>,
    /// When this version was created
    pub created_at: DateTime<Utc>,
    /// What type of change created this version
    pub event_type: VersionEventType,
    /// Optional description of the change
    #[serde(skip_serializing_if = "Option::is_none")]
    pub change_description: Option<String>,
    /// User who made the change (if applicable)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub changed_by: Option<String>,
}

impl MemoryVersion {
    /// Create a new version for memory creation
    pub fn initial(memory_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            version_id: Uuid::new_v4(),
            memory_id: memory_id.into(),
            version_number: 1,
            content: content.into(),
            metadata: HashMap::new(),
            fsrs_state: None,
            created_at: Utc::now(),
            event_type: VersionEventType::Created,
            change_description: None,
            changed_by: None,
        }
    }

    /// Create a new version from previous version with content update
    pub fn from_content_update(
        previous: &MemoryVersion,
        new_content: impl Into<String>,
    ) -> Self {
        Self {
            version_id: Uuid::new_v4(),
            memory_id: previous.memory_id.clone(),
            version_number: previous.version_number + 1,
            content: new_content.into(),
            metadata: previous.metadata.clone(),
            fsrs_state: previous.fsrs_state.clone(),
            created_at: Utc::now(),
            event_type: VersionEventType::ContentUpdated,
            change_description: None,
            changed_by: None,
        }
    }

    /// Create a new version from previous version with metadata update
    pub fn from_metadata_update(
        previous: &MemoryVersion,
        new_metadata: HashMap<String, serde_json::Value>,
    ) -> Self {
        Self {
            version_id: Uuid::new_v4(),
            memory_id: previous.memory_id.clone(),
            version_number: previous.version_number + 1,
            content: previous.content.clone(),
            metadata: new_metadata,
            fsrs_state: previous.fsrs_state.clone(),
            created_at: Utc::now(),
            event_type: VersionEventType::MetadataUpdated,
            change_description: None,
            changed_by: None,
        }
    }

    /// Create a new version from previous version with FSRS state update
    pub fn from_fsrs_update(
        previous: &MemoryVersion,
        new_fsrs: FsrsStateSnapshot,
    ) -> Self {
        Self {
            version_id: Uuid::new_v4(),
            memory_id: previous.memory_id.clone(),
            version_number: previous.version_number + 1,
            content: previous.content.clone(),
            metadata: previous.metadata.clone(),
            fsrs_state: Some(new_fsrs),
            created_at: Utc::now(),
            event_type: VersionEventType::FsrsUpdated,
            change_description: None,
            changed_by: None,
        }
    }

    /// Builder: set change description
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.change_description = Some(desc.into());
        self
    }

    /// Builder: set changed_by
    pub fn changed_by(mut self, user: impl Into<String>) -> Self {
        self.changed_by = Some(user.into());
        self
    }

    /// Builder: set FSRS state
    pub fn with_fsrs(mut self, fsrs: FsrsStateSnapshot) -> Self {
        self.fsrs_state = Some(fsrs);
        self
    }

    /// Builder: set metadata
    pub fn with_metadata(mut self, metadata: HashMap<String, serde_json::Value>) -> Self {
        self.metadata = metadata;
        self
    }
}

/// Summary of version changes for a memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionSummary {
    pub memory_id: String,
    pub total_versions: u32,
    pub latest_version: u32,
    pub first_created: DateTime<Utc>,
    pub last_modified: DateTime<Utc>,
    pub content_updates: u32,
    pub metadata_updates: u32,
    pub fsrs_updates: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_event_type_round_trip() {
        let types = [
            VersionEventType::Created,
            VersionEventType::ContentUpdated,
            VersionEventType::MetadataUpdated,
            VersionEventType::FsrsUpdated,
            VersionEventType::Superseded,
            VersionEventType::Merged,
            VersionEventType::Restored,
        ];

        for event_type in types {
            let s = event_type.as_str();
            let parsed = VersionEventType::from_str(s);
            assert_eq!(parsed, Some(event_type));
        }
    }

    #[test]
    fn test_initial_version() {
        let v = MemoryVersion::initial("mem-1", "Hello world");

        assert_eq!(v.memory_id, "mem-1");
        assert_eq!(v.content, "Hello world");
        assert_eq!(v.version_number, 1);
        assert_eq!(v.event_type, VersionEventType::Created);
        assert!(v.metadata.is_empty());
        assert!(v.fsrs_state.is_none());
    }

    #[test]
    fn test_content_update_version() {
        let v1 = MemoryVersion::initial("mem-1", "Original");
        let v2 = MemoryVersion::from_content_update(&v1, "Updated");

        assert_eq!(v2.memory_id, "mem-1");
        assert_eq!(v2.content, "Updated");
        assert_eq!(v2.version_number, 2);
        assert_eq!(v2.event_type, VersionEventType::ContentUpdated);
    }

    #[test]
    fn test_metadata_update_version() {
        let v1 = MemoryVersion::initial("mem-1", "Content");
        let mut metadata = HashMap::new();
        metadata.insert("key".to_string(), serde_json::json!("value"));

        let v2 = MemoryVersion::from_metadata_update(&v1, metadata.clone());

        assert_eq!(v2.version_number, 2);
        assert_eq!(v2.event_type, VersionEventType::MetadataUpdated);
        assert_eq!(v2.metadata.get("key"), Some(&serde_json::json!("value")));
    }

    #[test]
    fn test_fsrs_update_version() {
        let v1 = MemoryVersion::initial("mem-1", "Content");
        let fsrs = FsrsStateSnapshot {
            stability: 10.0,
            difficulty: 5.0,
            retrievability: 0.9,
            storage_strength: 0.5,
            retrieval_strength: 1.0,
            last_review: Some(Utc::now()),
        };

        let v2 = MemoryVersion::from_fsrs_update(&v1, fsrs);

        assert_eq!(v2.version_number, 2);
        assert_eq!(v2.event_type, VersionEventType::FsrsUpdated);
        assert!(v2.fsrs_state.is_some());
    }

    #[test]
    fn test_builder_methods() {
        let v = MemoryVersion::initial("mem-1", "Content")
            .with_description("Initial creation")
            .changed_by("user@example.com");

        assert_eq!(v.change_description, Some("Initial creation".to_string()));
        assert_eq!(v.changed_by, Some("user@example.com".to_string()));
    }
}
