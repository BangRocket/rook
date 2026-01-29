//! Entity merging with configurable similarity threshold.
//!
//! This module provides entity deduplication by finding similar
//! existing entities using embedding-based similarity.

use serde::{Deserialize, Serialize};

use super::extractor::ExtractedEntity;

/// Configuration for entity merging.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeConfig {
    /// Similarity threshold for merging (0.0 - 1.0).
    /// Higher values require more similar entities to merge.
    /// Default: 0.85
    pub similarity_threshold: f32,
    /// Whether to require same entity type for merging.
    /// Default: true
    pub require_same_type: bool,
}

impl Default for MergeConfig {
    fn default() -> Self {
        Self {
            similarity_threshold: 0.85,
            require_same_type: true,
        }
    }
}

impl MergeConfig {
    /// Create a new merge config with custom threshold.
    pub fn with_threshold(similarity_threshold: f32) -> Self {
        Self {
            similarity_threshold,
            ..Default::default()
        }
    }

    /// Set whether same type is required.
    pub fn require_same_type(mut self, require: bool) -> Self {
        self.require_same_type = require;
        self
    }

    /// Get the merge threshold.
    pub fn merge_threshold(&self) -> f32 {
        self.similarity_threshold
    }
}

/// Result of entity merge attempt.
#[derive(Debug, Clone)]
pub struct MergeResult {
    /// Whether a match was found.
    pub matched: bool,
    /// The matched entity ID (if found).
    pub matched_entity_id: Option<i64>,
    /// The matched entity name (if found).
    pub matched_entity_name: Option<String>,
    /// The similarity score (if found).
    pub similarity: Option<f32>,
}

impl MergeResult {
    /// Create a result indicating no match.
    pub fn no_match() -> Self {
        Self {
            matched: false,
            matched_entity_id: None,
            matched_entity_name: None,
            similarity: None,
        }
    }

    /// Create a result indicating a match.
    pub fn matched(entity_id: i64, entity_name: String, similarity: f32) -> Self {
        Self {
            matched: true,
            matched_entity_id: Some(entity_id),
            matched_entity_name: Some(entity_name),
            similarity: Some(similarity),
        }
    }
}

/// Entity merger for deduplication.
pub struct EntityMerger {
    config: MergeConfig,
}

impl EntityMerger {
    /// Create a new entity merger.
    pub fn new(config: MergeConfig) -> Self {
        Self { config }
    }

    /// Get the merge config.
    pub fn config(&self) -> &MergeConfig {
        &self.config
    }

    /// Placeholder for find_match - will be implemented in Task 3.
    pub fn find_match_placeholder(&self, _entity: &ExtractedEntity) -> MergeResult {
        MergeResult::no_match()
    }
}
