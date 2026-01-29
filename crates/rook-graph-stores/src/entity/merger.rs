//! Entity merging with configurable similarity threshold.
//!
//! This module provides entity deduplication by finding similar
//! existing entities using embedding-based similarity.
//!
//! # Architecture
//!
//! Entity merging prevents duplicate entities in the graph by:
//! 1. Computing embedding similarity between new and existing entities
//! 2. Optionally requiring same entity type for merge candidates
//! 3. Returning the best match above the configurable threshold
//!
//! # Example
//!
//! ```ignore
//! let config = MergeConfig::default(); // threshold 0.85
//! let merger = EntityMerger::new(config);
//!
//! // Find similar entity
//! let result = merger.find_match(
//!     &new_entity,
//!     &existing_entities,
//!     &embeddings,
//!     new_embedding,
//! );
//!
//! if result.matched {
//!     println!("Found match: {:?}", result.matched_entity_name);
//! }
//! ```

use serde::{Deserialize, Serialize};

use super::extractor::ExtractedEntity;
use super::types::EntityType;

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

/// An existing entity with its embedding for merge comparison.
#[derive(Debug, Clone)]
pub struct ExistingEntity {
    /// Database ID of the entity.
    pub id: i64,
    /// Name of the entity.
    pub name: String,
    /// Type of the entity.
    pub entity_type: EntityType,
    /// Precomputed embedding vector.
    pub embedding: Vec<f32>,
}

impl ExistingEntity {
    /// Create a new existing entity.
    pub fn new(
        id: i64,
        name: impl Into<String>,
        entity_type: EntityType,
        embedding: Vec<f32>,
    ) -> Self {
        Self {
            id,
            name: name.into(),
            entity_type,
            embedding,
        }
    }
}

/// Compute cosine similarity between two vectors.
///
/// Returns a value between -1.0 and 1.0, where:
/// - 1.0 means identical direction
/// - 0.0 means orthogonal
/// - -1.0 means opposite direction
///
/// For entity matching, we typically expect positive values.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let mut dot_product = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;

    for (x, y) in a.iter().zip(b.iter()) {
        dot_product += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }

    let norm_a = norm_a.sqrt();
    let norm_b = norm_b.sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot_product / (norm_a * norm_b)
}

/// Entity merger for deduplication.
///
/// Finds existing entities that are similar enough to merge with
/// a newly extracted entity, based on embedding similarity.
pub struct EntityMerger {
    config: MergeConfig,
}

impl EntityMerger {
    /// Create a new entity merger.
    pub fn new(config: MergeConfig) -> Self {
        Self { config }
    }

    /// Create a merger with default config.
    pub fn default_merger() -> Self {
        Self::new(MergeConfig::default())
    }

    /// Get the merge config.
    pub fn config(&self) -> &MergeConfig {
        &self.config
    }

    /// Find a matching entity from existing entities.
    ///
    /// This method:
    /// 1. Filters by entity type (if require_same_type is true)
    /// 2. Computes cosine similarity between embeddings
    /// 3. Returns the best match above the threshold
    ///
    /// # Arguments
    ///
    /// * `entity` - The newly extracted entity to match
    /// * `existing` - Slice of existing entities with embeddings
    /// * `entity_embedding` - Embedding vector for the new entity
    ///
    /// # Returns
    ///
    /// A `MergeResult` indicating whether a match was found.
    pub fn find_match(
        &self,
        entity: &ExtractedEntity,
        existing: &[ExistingEntity],
        entity_embedding: &[f32],
    ) -> MergeResult {
        if existing.is_empty() || entity_embedding.is_empty() {
            return MergeResult::no_match();
        }

        let mut best_match: Option<(i64, String, f32)> = None;

        for candidate in existing {
            // Check type constraint
            if self.config.require_same_type && candidate.entity_type != entity.entity_type {
                continue;
            }

            // Compute similarity
            let similarity = cosine_similarity(entity_embedding, &candidate.embedding);

            // Check threshold
            if similarity < self.config.similarity_threshold {
                continue;
            }

            // Update best match
            match &best_match {
                None => {
                    best_match = Some((candidate.id, candidate.name.clone(), similarity));
                }
                Some((_, _, best_sim)) if similarity > *best_sim => {
                    best_match = Some((candidate.id, candidate.name.clone(), similarity));
                }
                _ => {}
            }
        }

        match best_match {
            Some((id, name, sim)) => MergeResult::matched(id, name, sim),
            None => MergeResult::no_match(),
        }
    }

    /// Check if two entities should be merged based on their names and types.
    ///
    /// This is a quick check without embeddings, using exact/fuzzy name matching.
    /// Returns true if the names are exactly equal (case-insensitive) and
    /// type requirements are met.
    pub fn should_merge_by_name(
        &self,
        entity: &ExtractedEntity,
        existing: &ExistingEntity,
    ) -> bool {
        // Check type constraint first
        if self.config.require_same_type && existing.entity_type != entity.entity_type {
            return false;
        }

        // Exact match (case-insensitive)
        entity.name.to_lowercase() == existing.name.to_lowercase()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merge_config_default() {
        let config = MergeConfig::default();
        assert_eq!(config.similarity_threshold, 0.85);
        assert!(config.require_same_type);
    }

    #[test]
    fn test_merge_config_with_threshold() {
        let config = MergeConfig::with_threshold(0.90);
        assert_eq!(config.similarity_threshold, 0.90);
        assert!(config.require_same_type);
    }

    #[test]
    fn test_merge_config_builder() {
        let config = MergeConfig::with_threshold(0.80).require_same_type(false);
        assert_eq!(config.similarity_threshold, 0.80);
        assert!(!config.require_same_type);
    }

    #[test]
    fn test_merge_threshold() {
        let config = MergeConfig::with_threshold(0.75);
        assert_eq!(config.merge_threshold(), 0.75);
    }

    #[test]
    fn test_merge_result_no_match() {
        let result = MergeResult::no_match();
        assert!(!result.matched);
        assert!(result.matched_entity_id.is_none());
        assert!(result.matched_entity_name.is_none());
        assert!(result.similarity.is_none());
    }

    #[test]
    fn test_merge_result_matched() {
        let result = MergeResult::matched(123, "Alice".to_string(), 0.95);
        assert!(result.matched);
        assert_eq!(result.matched_entity_id, Some(123));
        assert_eq!(result.matched_entity_name.as_deref(), Some("Alice"));
        assert_eq!(result.similarity, Some(0.95));
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 0.0001);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![-1.0, 0.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim + 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_cosine_similarity_partial() {
        let a = vec![1.0, 1.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        // Expected: 1 / (sqrt(2) * 1) = 0.7071
        assert!((sim - 0.7071).abs() < 0.001);
    }

    #[test]
    fn test_cosine_similarity_empty() {
        let a: Vec<f32> = vec![];
        let b: Vec<f32> = vec![];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn test_cosine_similarity_different_lengths() {
        let a = vec![1.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn test_find_match_empty_existing() {
        let merger = EntityMerger::new(MergeConfig::default());
        let entity = ExtractedEntity::new("Alice", EntityType::Person);
        let embedding = vec![1.0, 0.0, 0.0];

        let result = merger.find_match(&entity, &[], &embedding);
        assert!(!result.matched);
    }

    #[test]
    fn test_find_match_empty_embedding() {
        let merger = EntityMerger::new(MergeConfig::default());
        let entity = ExtractedEntity::new("Alice", EntityType::Person);
        let existing = vec![ExistingEntity::new(
            1,
            "Alice",
            EntityType::Person,
            vec![1.0, 0.0, 0.0],
        )];

        let result = merger.find_match(&entity, &existing, &[]);
        assert!(!result.matched);
    }

    #[test]
    fn test_find_match_exact_match() {
        let merger = EntityMerger::new(MergeConfig::with_threshold(0.80));
        let entity = ExtractedEntity::new("Alice", EntityType::Person);
        let existing = vec![ExistingEntity::new(
            1,
            "Alice",
            EntityType::Person,
            vec![1.0, 0.0, 0.0],
        )];
        let embedding = vec![1.0, 0.0, 0.0];

        let result = merger.find_match(&entity, &existing, &embedding);
        assert!(result.matched);
        assert_eq!(result.matched_entity_id, Some(1));
        assert_eq!(result.matched_entity_name.as_deref(), Some("Alice"));
        assert!((result.similarity.unwrap() - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_find_match_below_threshold() {
        let merger = EntityMerger::new(MergeConfig::with_threshold(0.90));
        let entity = ExtractedEntity::new("Alice", EntityType::Person);
        let existing = vec![ExistingEntity::new(
            1,
            "Alice",
            EntityType::Person,
            vec![1.0, 1.0, 0.0],
        )];
        // Similarity ~0.707, below 0.90 threshold
        let embedding = vec![1.0, 0.0, 0.0];

        let result = merger.find_match(&entity, &existing, &embedding);
        assert!(!result.matched);
    }

    #[test]
    fn test_find_match_type_mismatch() {
        let merger = EntityMerger::new(MergeConfig::default());
        let entity = ExtractedEntity::new("Alice", EntityType::Person);
        let existing = vec![ExistingEntity::new(
            1,
            "Alice Corp",
            EntityType::Organization, // Different type
            vec![1.0, 0.0, 0.0],
        )];
        let embedding = vec![1.0, 0.0, 0.0];

        let result = merger.find_match(&entity, &existing, &embedding);
        assert!(!result.matched);
    }

    #[test]
    fn test_find_match_type_mismatch_allowed() {
        let merger = EntityMerger::new(MergeConfig::default().require_same_type(false));
        let entity = ExtractedEntity::new("Alice", EntityType::Person);
        let existing = vec![ExistingEntity::new(
            1,
            "Alice Corp",
            EntityType::Organization, // Different type but allowed
            vec![1.0, 0.0, 0.0],
        )];
        let embedding = vec![1.0, 0.0, 0.0];

        let result = merger.find_match(&entity, &existing, &embedding);
        assert!(result.matched);
    }

    #[test]
    fn test_find_match_best_of_multiple() {
        let merger = EntityMerger::new(MergeConfig::with_threshold(0.50));
        let entity = ExtractedEntity::new("Alice", EntityType::Person);
        let existing = vec![
            ExistingEntity::new(1, "Alice1", EntityType::Person, vec![0.6, 0.8, 0.0]),
            ExistingEntity::new(2, "Alice2", EntityType::Person, vec![0.9, 0.1, 0.0]),
            ExistingEntity::new(3, "Alice3", EntityType::Person, vec![0.5, 0.5, 0.5]),
        ];
        // Embedding most similar to Alice2
        let embedding = vec![1.0, 0.0, 0.0];

        let result = merger.find_match(&entity, &existing, &embedding);
        assert!(result.matched);
        // Alice2 should be best match (highest similarity to [1,0,0])
        assert_eq!(result.matched_entity_id, Some(2));
        assert_eq!(result.matched_entity_name.as_deref(), Some("Alice2"));
    }

    #[test]
    fn test_should_merge_by_name_exact() {
        let merger = EntityMerger::new(MergeConfig::default());
        let entity = ExtractedEntity::new("Alice", EntityType::Person);
        let existing = ExistingEntity::new(1, "Alice", EntityType::Person, vec![]);

        assert!(merger.should_merge_by_name(&entity, &existing));
    }

    #[test]
    fn test_should_merge_by_name_case_insensitive() {
        let merger = EntityMerger::new(MergeConfig::default());
        let entity = ExtractedEntity::new("ALICE", EntityType::Person);
        let existing = ExistingEntity::new(1, "alice", EntityType::Person, vec![]);

        assert!(merger.should_merge_by_name(&entity, &existing));
    }

    #[test]
    fn test_should_merge_by_name_different_names() {
        let merger = EntityMerger::new(MergeConfig::default());
        let entity = ExtractedEntity::new("Alice", EntityType::Person);
        let existing = ExistingEntity::new(1, "Bob", EntityType::Person, vec![]);

        assert!(!merger.should_merge_by_name(&entity, &existing));
    }

    #[test]
    fn test_should_merge_by_name_type_mismatch() {
        let merger = EntityMerger::new(MergeConfig::default());
        let entity = ExtractedEntity::new("Alice", EntityType::Person);
        let existing = ExistingEntity::new(1, "Alice", EntityType::Organization, vec![]);

        assert!(!merger.should_merge_by_name(&entity, &existing));
    }

    #[test]
    fn test_should_merge_by_name_type_mismatch_allowed() {
        let merger = EntityMerger::new(MergeConfig::default().require_same_type(false));
        let entity = ExtractedEntity::new("Alice", EntityType::Person);
        let existing = ExistingEntity::new(1, "Alice", EntityType::Organization, vec![]);

        assert!(merger.should_merge_by_name(&entity, &existing));
    }
}
