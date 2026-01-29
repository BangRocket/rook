//! Embedding similarity layer (Layer 1) for prediction error gating.

use crate::error::RookResult;
use crate::ingestion::types::{GatingThresholds, IngestDecision};
use crate::traits::{Embedder, EmbeddingAction, VectorRecord};

/// A memory candidate with similarity score
#[derive(Debug, Clone)]
pub struct SimilarityCandidate {
    pub memory_id: String,
    pub content: String,
    pub embedding: Vec<f32>,
    pub similarity: f32,
}

/// Result from embedding similarity check
#[derive(Debug)]
pub struct EmbeddingResult {
    /// Maximum similarity found
    pub max_similarity: f32,
    /// Candidate memories above related_threshold, sorted by similarity desc
    pub candidates: Vec<SimilarityCandidate>,
    /// Decision if clear from embedding alone (duplicate or novel)
    pub decision: Option<IngestDecision>,
    /// Reason for decision
    pub reason: Option<String>,
}

/// Layer 1: Embedding similarity detection
///
/// Fastest layer (~1ms). Calculates cosine similarity between new content
/// and existing memories to identify:
/// - Clear duplicates (similarity > duplicate_threshold) -> Skip
/// - Clearly novel (similarity < novel_threshold) -> Create
/// - Related memories that need deeper analysis -> pass to next layer
pub struct EmbeddingSimilarityLayer {
    thresholds: GatingThresholds,
}

impl EmbeddingSimilarityLayer {
    pub fn new(thresholds: GatingThresholds) -> Self {
        Self { thresholds }
    }

    pub fn with_default_thresholds() -> Self {
        Self::new(GatingThresholds::default())
    }

    /// Check new content against existing memories
    pub async fn check(
        &self,
        new_content: &str,
        existing_memories: &[VectorRecord],
        embedder: &dyn Embedder,
    ) -> RookResult<EmbeddingResult> {
        // Generate embedding for new content
        let new_embedding = embedder
            .embed(new_content, Some(EmbeddingAction::Search))
            .await?;

        let mut candidates = Vec::new();
        let mut max_similarity = 0.0f32;

        for memory in existing_memories {
            let similarity = cosine_similarity(&new_embedding, &memory.vector);
            max_similarity = max_similarity.max(similarity);

            if similarity >= self.thresholds.related_threshold {
                let content = memory
                    .payload
                    .get("data")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();

                candidates.push(SimilarityCandidate {
                    memory_id: memory.id.clone(),
                    content,
                    embedding: memory.vector.clone(),
                    similarity,
                });
            }
        }

        // Sort by similarity descending
        candidates.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Determine if we can short-circuit
        let (decision, reason) = if max_similarity >= self.thresholds.duplicate_threshold {
            (
                Some(IngestDecision::Skip),
                Some(format!(
                    "Duplicate detected (similarity: {:.3})",
                    max_similarity
                )),
            )
        } else if max_similarity < self.thresholds.novel_threshold {
            (
                Some(IngestDecision::Create),
                Some(format!(
                    "Novel content (max similarity: {:.3})",
                    max_similarity
                )),
            )
        } else {
            (None, None) // Ambiguous, continue to next layer
        };

        Ok(EmbeddingResult {
            max_similarity,
            candidates,
            decision,
            reason,
        })
    }

    /// Calculate surprise value from similarity
    /// Higher similarity = lower surprise, lower similarity = higher surprise
    pub fn calculate_surprise(&self, max_similarity: f32) -> f32 {
        1.0 - max_similarity.clamp(0.0, 1.0)
    }
}

/// Calculate cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "Vectors must have same dimension");

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        assert!((cosine_similarity(&a, &b)).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![-1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_surprise_calculation() {
        let layer = EmbeddingSimilarityLayer::with_default_thresholds();

        // High similarity = low surprise
        assert!((layer.calculate_surprise(0.9) - 0.1).abs() < 1e-6);

        // Low similarity = high surprise
        assert!((layer.calculate_surprise(0.1) - 0.9).abs() < 1e-6);

        // Zero similarity = maximum surprise
        assert!((layer.calculate_surprise(0.0) - 1.0).abs() < 1e-6);
    }
}
