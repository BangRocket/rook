//! Semantic LLM layer (Layer 4) for prediction error gating.
//!
//! This layer uses an LLM as a fallback for nuanced contradiction detection.
//! It is the slowest layer (~500ms) and should only be used when faster layers
//! are inconclusive.
//!
//! CAUTION: Research shows LLMs perform poorly at contradiction detection
//! (barely better than random). Use as last resort.

use std::sync::Arc;

use crate::error::RookResult;
use crate::ingestion::layers::embedding::SimilarityCandidate;
use crate::ingestion::types::IngestDecision;
use crate::traits::Llm;
use crate::types::Message;

/// Result from semantic LLM evaluation
#[derive(Debug)]
pub struct SemanticResult {
    /// Decision from LLM analysis
    pub decision: IngestDecision,
    /// Reasoning from LLM
    pub reasoning: String,
    /// ID of related memory (if Update/Supersede)
    pub related_id: Option<String>,
}

/// Layer 4: Semantic evaluation using LLM
///
/// CAUTION: This is the slowest layer (~500ms) and LLMs perform poorly at
/// contradiction detection (barely better than random per research).
/// Only use as last resort when faster layers are inconclusive.
///
/// This layer asks the LLM to:
/// 1. Compare new content with candidate memories
/// 2. Determine relationship: duplicate, update, contradiction, or unrelated
/// 3. Provide reasoning
pub struct SemanticLayer {
    llm: Arc<dyn Llm>,
}

impl SemanticLayer {
    pub fn new(llm: Arc<dyn Llm>) -> Self {
        Self { llm }
    }

    /// Evaluate semantic relationship between new content and candidates
    pub async fn evaluate(
        &self,
        new_content: &str,
        candidates: &[SimilarityCandidate],
    ) -> RookResult<SemanticResult> {
        if candidates.is_empty() {
            // No candidates to compare, default to Create
            return Ok(SemanticResult {
                decision: IngestDecision::Create,
                reasoning: "No existing memories to compare".to_string(),
                related_id: None,
            });
        }

        // Build prompt for LLM
        let prompt = self.build_prompt(new_content, candidates);

        let messages = vec![
            Message::system(SEMANTIC_SYSTEM_PROMPT.to_string()),
            Message::user(prompt),
        ];

        let response = self.llm.generate(&messages, None).await?;
        let content = response.content_or_empty();

        // Parse LLM response
        self.parse_response(content, candidates)
    }

    fn build_prompt(&self, new_content: &str, candidates: &[SimilarityCandidate]) -> String {
        let mut prompt = format!(
            "NEW INFORMATION:\n\"{}\"\n\nEXISTING MEMORIES:\n",
            new_content
        );

        for (i, candidate) in candidates.iter().enumerate() {
            prompt.push_str(&format!(
                "[{}] \"{}\" (similarity: {:.2})\n",
                i, candidate.content, candidate.similarity
            ));
        }

        prompt.push_str("\nAnalyze the relationship and respond with a single JSON object.");
        prompt
    }

    fn parse_response(
        &self,
        response: &str,
        candidates: &[SimilarityCandidate],
    ) -> RookResult<SemanticResult> {
        // Try to extract JSON from response
        let json_start = response.find('{').unwrap_or(0);
        let json_end = response.rfind('}').map(|i| i + 1).unwrap_or(response.len());
        let json_str = &response[json_start..json_end];

        // Parse the JSON response
        #[derive(serde::Deserialize)]
        struct LlmDecision {
            decision: String,
            reasoning: String,
            #[serde(default)]
            related_index: Option<usize>,
        }

        let parsed: LlmDecision = serde_json::from_str(json_str).unwrap_or_else(|_| {
            // Fallback if JSON parsing fails
            LlmDecision {
                decision: "create".to_string(),
                reasoning: format!("Could not parse LLM response: {}", response),
                related_index: None,
            }
        });

        let decision = match parsed.decision.to_lowercase().as_str() {
            "skip" | "duplicate" => IngestDecision::Skip,
            "update" | "refine" => IngestDecision::Update,
            "supersede" | "contradict" | "contradiction" => IngestDecision::Supersede,
            _ => IngestDecision::Create,
        };

        let related_id = parsed
            .related_index
            .and_then(|idx| candidates.get(idx))
            .map(|c| c.memory_id.clone());

        Ok(SemanticResult {
            decision,
            reasoning: parsed.reasoning,
            related_id,
        })
    }
}

const SEMANTIC_SYSTEM_PROMPT: &str = r#"You are a memory contradiction detector. Given new information and existing memories, determine their relationship.

Respond with a JSON object:
{
  "decision": "skip" | "create" | "update" | "supersede",
  "reasoning": "brief explanation",
  "related_index": null | <index of related memory>
}

Decisions:
- "skip": New info is essentially duplicate of existing memory
- "create": New info is distinct, store as new memory
- "update": New info adds to/refines existing memory (non-contradictory)
- "supersede": New info contradicts existing memory (newer info wins)

Be conservative with "supersede" - only use for clear contradictions, not just different topics.
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::{GenerationOptions, LlmResponse, LlmStream, Tool, ToolChoice};

    // Mock LLM for tests - won't actually be called in unit tests
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

    #[test]
    fn test_parse_create_response() {
        let layer = SemanticLayer::new(Arc::new(MockLlm));
        let candidates = vec![];

        let response = r#"{"decision": "create", "reasoning": "Novel information", "related_index": null}"#;
        let result = layer.parse_response(response, &candidates).unwrap();

        assert!(matches!(result.decision, IngestDecision::Create));
        assert_eq!(result.reasoning, "Novel information");
        assert!(result.related_id.is_none());
    }

    #[test]
    fn test_parse_supersede_response() {
        let layer = SemanticLayer::new(Arc::new(MockLlm));
        let candidates = vec![SimilarityCandidate {
            memory_id: "mem1".to_string(),
            content: "test".to_string(),
            embedding: vec![],
            similarity: 0.8,
        }];

        let response = r#"{"decision": "supersede", "reasoning": "Contradicts", "related_index": 0}"#;
        let result = layer.parse_response(response, &candidates).unwrap();

        assert!(matches!(result.decision, IngestDecision::Supersede));
        assert_eq!(result.related_id, Some("mem1".to_string()));
    }

    #[test]
    fn test_parse_skip_response() {
        let layer = SemanticLayer::new(Arc::new(MockLlm));
        let candidates = vec![SimilarityCandidate {
            memory_id: "mem1".to_string(),
            content: "User likes pizza".to_string(),
            embedding: vec![],
            similarity: 0.95,
        }];

        let response = r#"{"decision": "skip", "reasoning": "Duplicate of existing memory", "related_index": 0}"#;
        let result = layer.parse_response(response, &candidates).unwrap();

        assert!(matches!(result.decision, IngestDecision::Skip));
        assert_eq!(result.related_id, Some("mem1".to_string()));
    }

    #[test]
    fn test_parse_update_response() {
        let layer = SemanticLayer::new(Arc::new(MockLlm));
        let candidates = vec![SimilarityCandidate {
            memory_id: "mem1".to_string(),
            content: "User likes Italian food".to_string(),
            embedding: vec![],
            similarity: 0.75,
        }];

        let response = r#"{"decision": "update", "reasoning": "Adds detail to existing preference", "related_index": 0}"#;
        let result = layer.parse_response(response, &candidates).unwrap();

        assert!(matches!(result.decision, IngestDecision::Update));
        assert_eq!(result.related_id, Some("mem1".to_string()));
    }

    #[test]
    fn test_parse_malformed_response() {
        let layer = SemanticLayer::new(Arc::new(MockLlm));
        let candidates = vec![];

        let response = "This is not JSON at all";
        let result = layer.parse_response(response, &candidates).unwrap();

        // Should fallback to Create
        assert!(matches!(result.decision, IngestDecision::Create));
    }

    #[test]
    fn test_parse_response_with_extra_text() {
        let layer = SemanticLayer::new(Arc::new(MockLlm));
        let candidates = vec![];

        let response = r#"Here is my analysis: {"decision": "create", "reasoning": "New info", "related_index": null} That's my answer."#;
        let result = layer.parse_response(response, &candidates).unwrap();

        assert!(matches!(result.decision, IngestDecision::Create));
    }

    #[test]
    fn test_build_prompt() {
        let layer = SemanticLayer::new(Arc::new(MockLlm));
        let candidates = vec![
            SimilarityCandidate {
                memory_id: "1".to_string(),
                content: "User lives in Boston".to_string(),
                embedding: vec![],
                similarity: 0.85,
            },
            SimilarityCandidate {
                memory_id: "2".to_string(),
                content: "User works at Google".to_string(),
                embedding: vec![],
                similarity: 0.72,
            },
        ];

        let prompt = layer.build_prompt("I now live in New York", &candidates);

        assert!(prompt.contains("NEW INFORMATION:"));
        assert!(prompt.contains("I now live in New York"));
        assert!(prompt.contains("[0]"));
        assert!(prompt.contains("User lives in Boston"));
        assert!(prompt.contains("[1]"));
        assert!(prompt.contains("User works at Google"));
    }

    #[tokio::test]
    async fn test_evaluate_empty_candidates() {
        let layer = SemanticLayer::new(Arc::new(MockLlm));
        let candidates: Vec<SimilarityCandidate> = vec![];

        let result = layer.evaluate("Some new content", &candidates).await.unwrap();

        assert!(matches!(result.decision, IngestDecision::Create));
        assert_eq!(result.reasoning, "No existing memories to compare");
    }
}
