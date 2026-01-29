//! Prediction Error Gate - orchestrates multi-layer contradiction detection.
//!
//! Executes detection layers in order of speed, short-circuiting when a clear
//! decision is reached. This implements the cognitive science concept of
//! "prediction error" - surprising information gets encoded more strongly.
//!
//! Layer cascade:
//! 1. Embedding similarity (~1ms) - duplicate/novel detection
//! 2. Keyword patterns (~1ms) - explicit contradiction patterns
//! 3. Temporal conflicts (~1ms) - date-based conflicts
//! 4. Semantic LLM (~500ms) - fallback for nuanced cases

use std::sync::Arc;
use std::time::Instant;

use crate::error::RookResult;
use crate::ingestion::layers::{
    EmbeddingSimilarityLayer, KeywordNegationLayer, SemanticLayer, TemporalConflictLayer,
};
use crate::ingestion::types::{DetectionLayer, GatingThresholds, IngestDecision, LayerTimings};
use crate::traits::{Embedder, Llm, VectorRecord};

/// Result from prediction error gate evaluation
#[derive(Debug)]
pub struct GateResult {
    /// The decision made
    pub decision: IngestDecision,
    /// Which layer made the decision
    pub layer: DetectionLayer,
    /// ID of related memory (for Update/Supersede/Skip)
    pub related_memory_id: Option<String>,
    /// Surprise value (0.0-1.0)
    pub surprise: f32,
    /// Reasoning for the decision
    pub reason: Option<String>,
    /// Timing information per layer
    pub timings: LayerTimings,
}

/// Prediction Error Gate - orchestrates multi-layer contradiction detection
///
/// Executes detection layers in order of speed, short-circuiting when a clear
/// decision is reached. This implements the cognitive science concept of
/// "prediction error" - surprising information gets encoded more strongly.
///
/// Layer cascade:
/// 1. Embedding similarity (~1ms) - duplicate/novel detection
/// 2. Keyword patterns (~1ms) - explicit contradiction patterns
/// 3. Temporal conflicts (~1ms) - date-based conflicts
/// 4. Semantic LLM (~500ms) - fallback for nuanced cases
pub struct PredictionErrorGate {
    embedding_layer: EmbeddingSimilarityLayer,
    keyword_layer: KeywordNegationLayer,
    temporal_layer: TemporalConflictLayer,
    semantic_layer: Option<SemanticLayer>,
    thresholds: GatingThresholds,
}

impl PredictionErrorGate {
    /// Create a new gate with default thresholds
    pub fn new(llm: Option<Arc<dyn Llm>>) -> Self {
        Self::with_thresholds(GatingThresholds::default(), llm)
    }

    /// Create a new gate with custom thresholds
    pub fn with_thresholds(thresholds: GatingThresholds, llm: Option<Arc<dyn Llm>>) -> Self {
        Self {
            embedding_layer: EmbeddingSimilarityLayer::new(thresholds.clone()),
            keyword_layer: KeywordNegationLayer::new(),
            temporal_layer: TemporalConflictLayer::new(),
            semantic_layer: llm.map(SemanticLayer::new),
            thresholds,
        }
    }

    /// Evaluate new content against existing memories
    ///
    /// Returns a GateResult with the decision, which layer made it,
    /// and associated metadata (related memory ID, surprise, reasoning).
    pub async fn evaluate(
        &self,
        new_content: &str,
        existing_memories: &[VectorRecord],
        embedder: &dyn Embedder,
    ) -> RookResult<GateResult> {
        let total_start = Instant::now();
        let mut timings = LayerTimings::default();

        // Layer 1: Embedding similarity (~1ms)
        let layer1_start = Instant::now();
        let embedding_result = self
            .embedding_layer
            .check(new_content, existing_memories, embedder)
            .await?;
        timings.embedding_layer = Some(layer1_start.elapsed());

        // Short-circuit on clear duplicate or novel content
        if let Some(decision) = embedding_result.decision {
            timings.total = total_start.elapsed();

            let surprise = self
                .embedding_layer
                .calculate_surprise(embedding_result.max_similarity);

            return Ok(GateResult {
                decision,
                layer: DetectionLayer::EmbeddingSimilarity,
                related_memory_id: embedding_result
                    .candidates
                    .first()
                    .map(|c| c.memory_id.clone()),
                surprise,
                reason: embedding_result.reason,
                timings,
            });
        }

        // Layer 2: Keyword/negation patterns (~1ms)
        let layer2_start = Instant::now();
        let keyword_result = self
            .keyword_layer
            .check(new_content, &embedding_result.candidates);
        timings.keyword_layer = Some(layer2_start.elapsed());

        // Short-circuit on explicit contradiction pattern
        if let Some(decision) = keyword_result.decision {
            timings.total = total_start.elapsed();

            return Ok(GateResult {
                decision,
                layer: DetectionLayer::KeywordPattern,
                related_memory_id: keyword_result.contradicted_id,
                surprise: 0.8, // Contradictions are surprising
                reason: keyword_result.contradiction_evidence,
                timings,
            });
        }

        // Layer 3: Temporal conflict detection
        let layer3_start = Instant::now();
        let temporal_result = self
            .temporal_layer
            .check(new_content, &embedding_result.candidates);
        timings.temporal_layer = Some(layer3_start.elapsed());

        // Short-circuit on temporal conflict
        if let Some(decision) = temporal_result.decision {
            timings.total = total_start.elapsed();

            return Ok(GateResult {
                decision,
                layer: DetectionLayer::TemporalConflict,
                related_memory_id: temporal_result.conflicting_id,
                surprise: 0.7, // Temporal updates are moderately surprising
                reason: temporal_result.conflict_reason,
                timings,
            });
        }

        // Layer 4: Semantic LLM evaluation (~500ms, only if configured)
        if let Some(ref semantic) = self.semantic_layer {
            let layer4_start = Instant::now();
            let semantic_result = semantic
                .evaluate(new_content, &embedding_result.candidates)
                .await?;
            timings.semantic_layer = Some(layer4_start.elapsed());
            timings.total = total_start.elapsed();

            let surprise = match semantic_result.decision {
                IngestDecision::Skip => 0.0,
                IngestDecision::Create => self
                    .embedding_layer
                    .calculate_surprise(embedding_result.max_similarity),
                IngestDecision::Update => 0.4,
                IngestDecision::Supersede => 0.8,
            };

            return Ok(GateResult {
                decision: semantic_result.decision,
                layer: DetectionLayer::SemanticLlm,
                related_memory_id: semantic_result.related_id,
                surprise,
                reason: Some(semantic_result.reasoning),
                timings,
            });
        }

        // Default: If we reach here with related candidates, Update; else Create
        timings.total = total_start.elapsed();

        let (decision, related_id) = if !embedding_result.candidates.is_empty()
            && embedding_result.max_similarity >= 0.7
        {
            // Related content found, but no contradiction - likely an update
            (
                IngestDecision::Update,
                embedding_result
                    .candidates
                    .first()
                    .map(|c| c.memory_id.clone()),
            )
        } else {
            // No strong relation - create new memory
            (IngestDecision::Create, None)
        };

        let surprise = self
            .embedding_layer
            .calculate_surprise(embedding_result.max_similarity);

        Ok(GateResult {
            decision,
            layer: DetectionLayer::Default,
            related_memory_id: related_id,
            surprise,
            reason: Some("No layer made definitive decision".to_string()),
            timings,
        })
    }

    /// Get current thresholds
    pub fn thresholds(&self) -> &GatingThresholds {
        &self.thresholds
    }

    /// Check if semantic layer is configured
    pub fn has_semantic_layer(&self) -> bool {
        self.semantic_layer.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: Full async tests require mock Embedder
    // These are structural tests

    #[test]
    fn test_gate_creation() {
        let gate = PredictionErrorGate::new(None);
        assert!(!gate.has_semantic_layer());

        let thresholds = gate.thresholds();
        assert!((thresholds.duplicate_threshold - 0.95).abs() < 0.01);
    }

    #[test]
    fn test_custom_thresholds() {
        let thresholds = GatingThresholds {
            duplicate_threshold: 0.90,
            related_threshold: 0.60,
            novel_threshold: 0.40,
        };

        let gate = PredictionErrorGate::with_thresholds(thresholds.clone(), None);
        assert!((gate.thresholds().duplicate_threshold - 0.90).abs() < 0.01);
    }

    #[test]
    fn test_gate_with_llm() {
        use crate::traits::{GenerationOptions, LlmResponse, LlmStream, Tool, ToolChoice};
        use crate::types::Message;

        struct MockLlm;

        #[async_trait::async_trait]
        impl Llm for MockLlm {
            async fn generate(
                &self,
                _messages: &[Message],
                _options: Option<GenerationOptions>,
            ) -> RookResult<LlmResponse> {
                Ok(LlmResponse::default())
            }

            async fn generate_with_tools(
                &self,
                _messages: &[Message],
                _tools: &[Tool],
                _tool_choice: ToolChoice,
                _options: Option<GenerationOptions>,
            ) -> RookResult<LlmResponse> {
                Ok(LlmResponse::default())
            }

            async fn generate_stream(
                &self,
                _messages: &[Message],
                _options: Option<GenerationOptions>,
            ) -> RookResult<LlmStream> {
                unimplemented!()
            }

            fn model_name(&self) -> &str {
                "mock"
            }
        }

        let gate = PredictionErrorGate::new(Some(Arc::new(MockLlm)));
        assert!(gate.has_semantic_layer());
    }

    #[test]
    fn test_gate_result_fields() {
        // Test GateResult struct construction
        let timings = LayerTimings::default();
        let result = GateResult {
            decision: IngestDecision::Create,
            layer: DetectionLayer::EmbeddingSimilarity,
            related_memory_id: None,
            surprise: 0.8,
            reason: Some("Novel content".to_string()),
            timings,
        };

        assert!(matches!(result.decision, IngestDecision::Create));
        assert!(matches!(result.layer, DetectionLayer::EmbeddingSimilarity));
        assert!(result.related_memory_id.is_none());
        assert!((result.surprise - 0.8).abs() < 0.01);
    }
}
