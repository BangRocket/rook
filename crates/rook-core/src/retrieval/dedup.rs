//! Result deduplication using embedding similarity.
//!
//! Removes near-duplicate results based on cosine similarity threshold.
//! Keeps the higher-scored item when duplicates are found.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// Configuration for deduplication.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeduplicationConfig {
    /// Cosine similarity threshold above which items are considered duplicates.
    /// Range: 0.0-1.0. Default: 0.95 (95% similar = duplicate)
    pub similarity_threshold: f32,
}

impl Default for DeduplicationConfig {
    fn default() -> Self {
        Self {
            similarity_threshold: 0.95,
        }
    }
}

impl DeduplicationConfig {
    /// Create config with custom threshold.
    pub fn with_threshold(threshold: f32) -> Self {
        Self {
            similarity_threshold: threshold.clamp(0.0, 1.0),
        }
    }

    /// Strict deduplication (lower threshold = more aggressive).
    pub fn strict() -> Self {
        Self {
            similarity_threshold: 0.90,
        }
    }

    /// Lenient deduplication (higher threshold = fewer removals).
    pub fn lenient() -> Self {
        Self {
            similarity_threshold: 0.98,
        }
    }
}

/// A retrieval result with embedding for deduplication.
#[derive(Debug, Clone)]
pub struct DeduplicatableResult {
    /// Unique identifier.
    pub id: String,
    /// Relevance score (higher = more relevant).
    pub score: f32,
    /// Optional embedding for similarity comparison.
    /// If None, result is kept by default (no dedup possible).
    pub embedding: Option<Vec<f32>>,
}

/// Deduplicator removes near-duplicate results based on embedding similarity.
pub struct Deduplicator {
    config: DeduplicationConfig,
}

impl Deduplicator {
    /// Create a new deduplicator with the given configuration.
    pub fn new(config: DeduplicationConfig) -> Self {
        Self { config }
    }

    /// Create a deduplicator with default configuration.
    pub fn default_config() -> Self {
        Self::new(DeduplicationConfig::default())
    }

    /// Deduplicate results based on embedding similarity.
    ///
    /// Results should be pre-sorted by score descending.
    /// Keeps the first (higher-scored) item when duplicates are found.
    ///
    /// # Arguments
    /// * `results` - Pre-sorted results (highest score first)
    ///
    /// # Returns
    /// Deduplicated results preserving original order.
    pub fn deduplicate(&self, results: Vec<DeduplicatableResult>) -> Vec<DeduplicatableResult> {
        if results.len() <= 1 {
            return results;
        }

        let mut kept: Vec<DeduplicatableResult> = Vec::with_capacity(results.len());
        let mut kept_embeddings: Vec<Vec<f32>> = Vec::with_capacity(results.len());

        for result in results {
            let should_keep = match &result.embedding {
                None => true, // No embedding, keep by default
                Some(embedding) => {
                    // Check if this is a duplicate of any kept result
                    !kept_embeddings.iter().any(|kept_emb| {
                        cosine_similarity(embedding, kept_emb) >= self.config.similarity_threshold
                    })
                }
            };

            if should_keep {
                if let Some(emb) = &result.embedding {
                    kept_embeddings.push(emb.clone());
                }
                kept.push(result);
            }
            // If duplicate, skip (higher-scored version already kept)
        }

        kept
    }

    /// Deduplicate using a separate embedding lookup.
    ///
    /// Useful when embeddings are stored separately from results.
    ///
    /// # Arguments
    /// * `results` - (id, score) pairs, pre-sorted by score descending
    /// * `embeddings` - Map of id -> embedding
    ///
    /// # Returns
    /// Deduplicated (id, score) pairs.
    pub fn deduplicate_with_lookup(
        &self,
        results: Vec<(String, f32)>,
        embeddings: &HashMap<String, Vec<f32>>,
    ) -> Vec<(String, f32)> {
        let dedup_results: Vec<DeduplicatableResult> = results
            .into_iter()
            .map(|(id, score)| DeduplicatableResult {
                embedding: embeddings.get(&id).cloned(),
                id,
                score,
            })
            .collect();

        self.deduplicate(dedup_results)
            .into_iter()
            .map(|r| (r.id, r.score))
            .collect()
    }
}

/// Calculate cosine similarity between two vectors.
///
/// Returns 0.0 if either vector has zero magnitude.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a > f32::EPSILON && norm_b > f32::EPSILON {
        dot / (norm_a * norm_b)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_result(id: &str, score: f32, embedding: Vec<f32>) -> DeduplicatableResult {
        DeduplicatableResult {
            id: id.to_string(),
            score,
            embedding: Some(embedding),
        }
    }

    #[test]
    fn test_no_duplicates() {
        let dedup = Deduplicator::default_config();
        let results = vec![
            make_result("a", 1.0, vec![1.0, 0.0, 0.0]),
            make_result("b", 0.8, vec![0.0, 1.0, 0.0]),
            make_result("c", 0.6, vec![0.0, 0.0, 1.0]),
        ];

        let deduped = dedup.deduplicate(results);
        assert_eq!(deduped.len(), 3);
    }

    #[test]
    fn test_removes_duplicates() {
        let dedup = Deduplicator::new(DeduplicationConfig::with_threshold(0.99));

        // a and b are identical embeddings
        let results = vec![
            make_result("a", 1.0, vec![1.0, 0.0, 0.0]),
            make_result("b", 0.8, vec![1.0, 0.0, 0.0]), // Duplicate of a
            make_result("c", 0.6, vec![0.0, 1.0, 0.0]),
        ];

        let deduped = dedup.deduplicate(results);
        assert_eq!(deduped.len(), 2);
        assert_eq!(deduped[0].id, "a"); // Higher scored kept
        assert_eq!(deduped[1].id, "c");
    }

    #[test]
    fn test_near_duplicates() {
        let dedup = Deduplicator::new(DeduplicationConfig::with_threshold(0.95));

        // a and b are very similar but not identical
        let results = vec![
            make_result("a", 1.0, vec![1.0, 0.0, 0.0]),
            make_result("b", 0.8, vec![0.99, 0.1, 0.0]), // ~99% similar
            make_result("c", 0.6, vec![0.0, 1.0, 0.0]),
        ];

        let deduped = dedup.deduplicate(results);
        assert_eq!(deduped.len(), 2); // b removed as near-duplicate
    }

    #[test]
    fn test_keeps_results_without_embeddings() {
        let dedup = Deduplicator::default_config();

        let results = vec![
            DeduplicatableResult {
                id: "a".to_string(),
                score: 1.0,
                embedding: None,
            },
            make_result("b", 0.8, vec![1.0, 0.0, 0.0]),
        ];

        let deduped = dedup.deduplicate(results);
        assert_eq!(deduped.len(), 2); // Both kept
    }

    #[test]
    fn test_empty_and_single() {
        let dedup = Deduplicator::default_config();

        let empty: Vec<DeduplicatableResult> = vec![];
        assert!(dedup.deduplicate(empty).is_empty());

        let single = vec![make_result("a", 1.0, vec![1.0, 0.0])];
        assert_eq!(dedup.deduplicate(single).len(), 1);
    }

    #[test]
    fn test_cosine_similarity() {
        // Identical vectors
        assert!((cosine_similarity(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 0.01);

        // Orthogonal vectors
        assert!(cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]).abs() < 0.01);

        // Opposite vectors
        assert!((cosine_similarity(&[1.0, 0.0], &[-1.0, 0.0]) - (-1.0)).abs() < 0.01);
    }

    #[test]
    fn test_deduplicate_with_lookup() {
        let dedup = Deduplicator::new(DeduplicationConfig::with_threshold(0.99));

        let results = vec![
            ("a".to_string(), 1.0),
            ("b".to_string(), 0.8),
            ("c".to_string(), 0.6),
        ];

        let mut embeddings = HashMap::new();
        embeddings.insert("a".to_string(), vec![1.0, 0.0]);
        embeddings.insert("b".to_string(), vec![1.0, 0.0]); // Same as a
        embeddings.insert("c".to_string(), vec![0.0, 1.0]);

        let deduped = dedup.deduplicate_with_lookup(results, &embeddings);
        assert_eq!(deduped.len(), 2);
        assert_eq!(deduped[0].0, "a");
        assert_eq!(deduped[1].0, "c");
    }
}
