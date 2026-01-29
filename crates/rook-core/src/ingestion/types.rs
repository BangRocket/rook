//! Core types for smart ingestion and prediction error gating.
//!
//! This module provides the type system for intelligent memory ingestion decisions,
//! determining whether new information should be skipped (duplicate), created (novel),
//! updated (additive), or supersede existing memories (contradiction).

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Decision made by prediction error gating.
///
/// Based on prediction error theory: high surprise (prediction error) indicates
/// novel information worth encoding, while low surprise indicates redundancy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IngestDecision {
    /// Skip: Information is redundant/duplicate (very low prediction error)
    Skip,
    /// Create: Novel information, encode as new memory (high prediction error)
    Create,
    /// Update: Refines existing memory with additive information (moderate prediction error)
    Update,
    /// Supersede: Contradicts and replaces existing memory (high prediction error + contradiction)
    Supersede,
}

/// Which detection layer made the decision.
///
/// Layers are ordered by computational cost, from fastest to slowest:
/// 1. EmbeddingSimilarity (~1ms) - vector cosine similarity
/// 2. KeywordPattern (~5ms) - keyword/pattern matching
/// 3. TemporalConflict (~10ms) - time-based contradiction detection
/// 4. SemanticLlm (~500ms+) - LLM-based semantic analysis
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DetectionLayer {
    /// Fast embedding similarity check (Layer 1)
    EmbeddingSimilarity,
    /// Keyword/pattern matching (Layer 2)
    KeywordPattern,
    /// Temporal conflict detection (Layer 3)
    TemporalConflict,
    /// Deep semantic analysis via LLM (Layer 4)
    SemanticLlm,
    /// Default layer when no specific detection was performed
    Default,
}

/// Configurable thresholds for prediction error gating.
///
/// These thresholds control the sensitivity of duplicate detection and novelty
/// assessment. The "gray zone" between `novel_threshold` and `duplicate_threshold`
/// requires deeper analysis by subsequent layers.
#[derive(Debug, Clone)]
pub struct GatingThresholds {
    /// Above this: definitely duplicate, Skip (default: 0.95)
    pub duplicate_threshold: f32,
    /// Above this: likely related, needs deeper analysis (default: 0.70)
    pub related_threshold: f32,
    /// Below this: clearly novel, Create directly (default: 0.50)
    pub novel_threshold: f32,
}

impl Default for GatingThresholds {
    fn default() -> Self {
        Self {
            duplicate_threshold: 0.95,
            related_threshold: 0.70,
            novel_threshold: 0.50,
        }
    }
}

/// Timing information for each detection layer.
///
/// Used for performance monitoring and optimization. Each layer records
/// how long it took, and the total is the sum of all layers invoked.
#[derive(Debug, Clone, Default)]
pub struct LayerTimings {
    /// Time spent in embedding similarity layer
    pub embedding_layer: Option<Duration>,
    /// Time spent in keyword/pattern layer
    pub keyword_layer: Option<Duration>,
    /// Time spent in temporal conflict layer
    pub temporal_layer: Option<Duration>,
    /// Time spent in semantic LLM layer
    pub semantic_layer: Option<Duration>,
    /// Total time across all layers
    pub total: Duration,
}

/// Result of smart_ingest() operation.
///
/// Captures the decision, associated memory IDs, surprise value (prediction error),
/// which layer made the decision, and the reasoning for transparency.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestResult {
    /// Decision taken (Skip/Create/Update/Supersede)
    pub decision: IngestDecision,

    /// Memory ID (for Create/Update/Supersede: new ID, for Skip: existing duplicate ID)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_id: Option<String>,

    /// Previous memory content (for Update/Supersede operations)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous_content: Option<String>,

    /// ID of related/superseded memory
    #[serde(skip_serializing_if = "Option::is_none")]
    pub related_memory_id: Option<String>,

    /// Prediction error / surprise value (0.0-1.0)
    /// - 0.0 = completely expected (duplicate)
    /// - 1.0 = completely unexpected (novel)
    pub surprise: f32,

    /// Which detection layer made the decision
    pub decided_at_layer: DetectionLayer,

    /// Reasoning for the decision (for debugging/transparency)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

impl IngestResult {
    /// Create a new IngestResult for a Skip decision.
    pub fn skip(existing_memory_id: String, similarity: f32, reason: impl Into<String>) -> Self {
        Self {
            decision: IngestDecision::Skip,
            memory_id: Some(existing_memory_id.clone()),
            previous_content: None,
            related_memory_id: Some(existing_memory_id),
            surprise: 1.0 - similarity, // Low surprise for duplicates
            decided_at_layer: DetectionLayer::EmbeddingSimilarity,
            reason: Some(reason.into()),
        }
    }

    /// Create a new IngestResult for a Create decision.
    pub fn create(new_memory_id: String, max_similarity: f32, reason: impl Into<String>) -> Self {
        Self {
            decision: IngestDecision::Create,
            memory_id: Some(new_memory_id),
            previous_content: None,
            related_memory_id: None,
            surprise: 1.0 - max_similarity, // High surprise for novel content
            decided_at_layer: DetectionLayer::EmbeddingSimilarity,
            reason: Some(reason.into()),
        }
    }

    /// Create a new IngestResult for an Update decision.
    pub fn update(
        new_memory_id: String,
        related_memory_id: String,
        previous_content: String,
        similarity: f32,
        layer: DetectionLayer,
        reason: impl Into<String>,
    ) -> Self {
        Self {
            decision: IngestDecision::Update,
            memory_id: Some(new_memory_id),
            previous_content: Some(previous_content),
            related_memory_id: Some(related_memory_id),
            surprise: 1.0 - similarity,
            decided_at_layer: layer,
            reason: Some(reason.into()),
        }
    }

    /// Create a new IngestResult for a Supersede decision.
    pub fn supersede(
        new_memory_id: String,
        superseded_memory_id: String,
        previous_content: String,
        layer: DetectionLayer,
        reason: impl Into<String>,
    ) -> Self {
        Self {
            decision: IngestDecision::Supersede,
            memory_id: Some(new_memory_id),
            previous_content: Some(previous_content),
            related_memory_id: Some(superseded_memory_id),
            surprise: 1.0, // Maximum surprise for contradictions
            decided_at_layer: layer,
            reason: Some(reason.into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_thresholds() {
        let thresholds = GatingThresholds::default();
        assert!((thresholds.duplicate_threshold - 0.95).abs() < f32::EPSILON);
        assert!((thresholds.related_threshold - 0.70).abs() < f32::EPSILON);
        assert!((thresholds.novel_threshold - 0.50).abs() < f32::EPSILON);
    }

    #[test]
    fn test_ingest_decision_serialization() {
        let decision = IngestDecision::Create;
        let json = serde_json::to_string(&decision).unwrap();
        assert_eq!(json, "\"Create\"");

        let parsed: IngestDecision = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, IngestDecision::Create);
    }

    #[test]
    fn test_ingest_result_skip() {
        let result = IngestResult::skip(
            "existing-123".to_string(),
            0.98,
            "Duplicate detected",
        );
        assert_eq!(result.decision, IngestDecision::Skip);
        assert_eq!(result.memory_id, Some("existing-123".to_string()));
        assert!((result.surprise - 0.02).abs() < 0.001);
    }

    #[test]
    fn test_ingest_result_create() {
        let result = IngestResult::create(
            "new-456".to_string(),
            0.3,
            "Novel content",
        );
        assert_eq!(result.decision, IngestDecision::Create);
        assert_eq!(result.memory_id, Some("new-456".to_string()));
        assert!((result.surprise - 0.7).abs() < 0.001);
    }

    #[test]
    fn test_layer_timings_default() {
        let timings = LayerTimings::default();
        assert!(timings.embedding_layer.is_none());
        assert!(timings.keyword_layer.is_none());
        assert!(timings.temporal_layer.is_none());
        assert!(timings.semantic_layer.is_none());
        assert_eq!(timings.total, Duration::ZERO);
    }
}
