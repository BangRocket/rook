//! Score fusion strategies for hybrid retrieval.
//!
//! Combines ranked results from multiple retrieval methods into a single ranking.
//! Supports Reciprocal Rank Fusion (RRF) for robust fusion without tuning,
//! and Linear Weighted Fusion for tunable, weighted combination.

use std::collections::HashMap;

use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};

/// Input scores for linear fusion (all scores should be normalized 0-1).
#[derive(Debug, Clone, Default)]
pub struct FusionInputs {
    /// Vector similarity score (cosine similarity, already 0-1).
    pub vector: f32,
    /// FSRS retrievability score (already 0-1).
    pub fsrs_retrievability: f32,
    /// Spreading activation score (already 0-1).
    pub activation: f32,
    /// BM25 score (must be normalized before fusion).
    pub bm25_normalized: f32,
}

/// Reciprocal Rank Fusion for combining ranked lists.
///
/// RRF is robust without parameter tuning and handles score scale mismatches.
/// Formula: score(d) = sum(1 / (k + rank_i(d))) for each ranker i
///
/// Reference: Cormack, Clarke & Buettcher (2009)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RrfFusion {
    /// k parameter controls how much to favor top-ranked items.
    /// Higher k = more even distribution; lower k = more weight to top ranks.
    /// Default: 60 (standard value from literature)
    pub k: f32,
}

impl Default for RrfFusion {
    fn default() -> Self {
        Self { k: 60.0 }
    }
}

impl RrfFusion {
    /// Create RRF fusion with custom k value.
    pub fn new(k: f32) -> Self {
        Self { k }
    }

    /// Fuse multiple ranked result lists into a single ranking.
    ///
    /// # Arguments
    /// * `ranked_lists` - Vec of ranked lists, each containing (id, score) pairs
    ///   sorted by score descending.
    ///
    /// # Returns
    /// Combined ranking as (id, rrf_score) pairs sorted by RRF score descending.
    pub fn fuse(&self, ranked_lists: Vec<Vec<(String, f32)>>) -> Vec<(String, f32)> {
        let mut rrf_scores: HashMap<String, f32> = HashMap::new();

        for ranked_list in ranked_lists {
            for (rank, (id, _original_score)) in ranked_list.iter().enumerate() {
                // RRF formula: 1 / (k + rank + 1)
                // rank is 0-indexed, so we add 1 to make it 1-indexed
                let rrf_contribution = 1.0 / (self.k + (rank as f32) + 1.0);
                *rrf_scores.entry(id.clone()).or_insert(0.0) += rrf_contribution;
            }
        }

        let mut results: Vec<_> = rrf_scores.into_iter().collect();
        results.sort_by(|a, b| OrderedFloat(b.1).cmp(&OrderedFloat(a.1)));
        results
    }
}

/// Linear weighted fusion for combining normalized scores.
///
/// Requires scores to be normalized to 0-1 range. Offers higher accuracy
/// when weights are properly tuned for the domain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearFusion {
    /// Weight for vector similarity score.
    pub vector_weight: f32,
    /// Weight for FSRS retrievability score.
    pub fsrs_weight: f32,
    /// Weight for spreading activation score.
    pub activation_weight: f32,
    /// Weight for BM25 text search score.
    pub bm25_weight: f32,
}

impl Default for LinearFusion {
    fn default() -> Self {
        // Standard mode weights (balanced)
        Self {
            vector_weight: 0.4,
            fsrs_weight: 0.2,
            activation_weight: 0.2,
            bm25_weight: 0.2,
        }
    }
}

impl LinearFusion {
    /// Create fusion weights optimized for Cognitive mode (FSRS-dominant).
    ///
    /// Emphasizes FSRS retrievability for human-like memory retrieval.
    pub fn cognitive() -> Self {
        Self {
            vector_weight: 0.4,
            fsrs_weight: 0.4,
            activation_weight: 0.2,
            bm25_weight: 0.0,
        }
    }

    /// Create fusion weights optimized for Precise mode (all signals).
    ///
    /// Balances all retrieval signals for maximum accuracy.
    pub fn precise() -> Self {
        Self {
            vector_weight: 0.35,
            fsrs_weight: 0.2,
            activation_weight: 0.2,
            bm25_weight: 0.25,
        }
    }

    /// Calculate fused score from normalized input scores.
    ///
    /// # Arguments
    /// * `inputs` - Normalized scores (all should be 0-1)
    ///
    /// # Returns
    /// Weighted sum of scores, clamped to 0-1.
    pub fn fuse(&self, inputs: &FusionInputs) -> f32 {
        let score = inputs.vector * self.vector_weight
            + inputs.fsrs_retrievability * self.fsrs_weight
            + inputs.activation * self.activation_weight
            + inputs.bm25_normalized * self.bm25_weight;
        score.clamp(0.0, 1.0)
    }

    /// Fuse a batch of results with their individual scores.
    ///
    /// # Arguments
    /// * `results` - Vec of (id, FusionInputs) pairs
    ///
    /// # Returns
    /// Vec of (id, fused_score) pairs sorted by score descending.
    pub fn fuse_batch(&self, results: Vec<(String, FusionInputs)>) -> Vec<(String, f32)> {
        let mut fused: Vec<_> = results
            .into_iter()
            .map(|(id, inputs)| (id, self.fuse(&inputs)))
            .collect();

        fused.sort_by(|a, b| OrderedFloat(b.1).cmp(&OrderedFloat(a.1)));
        fused
    }

    /// Validate that weights sum to approximately 1.0.
    pub fn validate(&self) -> Result<(), &'static str> {
        let sum =
            self.vector_weight + self.fsrs_weight + self.activation_weight + self.bm25_weight;
        if (sum - 1.0).abs() > 0.01 {
            return Err("Fusion weights should sum to 1.0");
        }
        if self.vector_weight < 0.0
            || self.fsrs_weight < 0.0
            || self.activation_weight < 0.0
            || self.bm25_weight < 0.0
        {
            return Err("Fusion weights must be non-negative");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rrf_single_list() {
        let rrf = RrfFusion::default();
        let results = rrf.fuse(vec![vec![
            ("a".to_string(), 1.0),
            ("b".to_string(), 0.8),
            ("c".to_string(), 0.5),
        ]]);

        // First item should have highest RRF score
        assert_eq!(results[0].0, "a");
        assert!(results[0].1 > results[1].1);
        assert!(results[1].1 > results[2].1);
    }

    #[test]
    fn test_rrf_multiple_lists() {
        let rrf = RrfFusion::default();
        let results = rrf.fuse(vec![
            vec![("a".to_string(), 1.0), ("b".to_string(), 0.5)],
            vec![("b".to_string(), 1.0), ("a".to_string(), 0.5)],
        ]);

        // Both a and b appear in both lists, should have equal scores
        let a_score = results.iter().find(|(id, _)| id == "a").unwrap().1;
        let b_score = results.iter().find(|(id, _)| id == "b").unwrap().1;
        assert!((a_score - b_score).abs() < 0.01);
    }

    #[test]
    fn test_rrf_unique_items() {
        let rrf = RrfFusion::default();
        let results = rrf.fuse(vec![
            vec![("a".to_string(), 1.0)],
            vec![("b".to_string(), 1.0)],
        ]);

        // Both should be present
        assert_eq!(results.len(), 2);
        // Scores should be equal (both rank 1 in their respective lists)
        assert!((results[0].1 - results[1].1).abs() < 0.01);
    }

    #[test]
    fn test_linear_fusion() {
        let fusion = LinearFusion::default();

        let inputs = FusionInputs {
            vector: 0.8,
            fsrs_retrievability: 0.6,
            activation: 0.5,
            bm25_normalized: 0.7,
        };

        let score = fusion.fuse(&inputs);

        // Expected: 0.8*0.4 + 0.6*0.2 + 0.5*0.2 + 0.7*0.2 = 0.32 + 0.12 + 0.10 + 0.14 = 0.68
        assert!((score - 0.68).abs() < 0.01);
    }

    #[test]
    fn test_cognitive_weights() {
        let fusion = LinearFusion::cognitive();
        fusion.validate().unwrap();

        // FSRS should have high weight
        assert!(fusion.fsrs_weight >= 0.4);
        // BM25 should be 0 in cognitive mode
        assert!(fusion.bm25_weight < 0.01);
    }

    #[test]
    fn test_linear_batch_fusion() {
        let fusion = LinearFusion::default();

        let batch = vec![
            (
                "a".to_string(),
                FusionInputs {
                    vector: 0.9,
                    ..Default::default()
                },
            ),
            (
                "b".to_string(),
                FusionInputs {
                    vector: 0.5,
                    ..Default::default()
                },
            ),
        ];

        let results = fusion.fuse_batch(batch);

        // Higher vector score should rank first
        assert_eq!(results[0].0, "a");
        assert!(results[0].1 > results[1].1);
    }
}
