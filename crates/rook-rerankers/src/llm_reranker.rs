//! LLM-based reranker implementation.

use async_trait::async_trait;

use rook_core::error::{RookError, RookResult};
use rook_core::traits::{Reranker, RerankerConfig};
use rook_core::types::MemoryItem;

use reqwest::Client;
use serde_json::json;

/// LLM-based reranker implementation.
/// Uses an LLM to score document relevance.
pub struct LlmReranker {
    client: Client,
    api_key: String,
    model: String,
    base_url: String,
    #[allow(dead_code)]
    config: RerankerConfig,
}

impl LlmReranker {
    /// Create a new LLM reranker.
    pub fn new(config: RerankerConfig) -> RookResult<Self> {
        let api_key = config
            .api_key
            .clone()
            .or_else(|| std::env::var("OPENAI_API_KEY").ok())
            .ok_or_else(|| {
                RookError::Configuration("API key required for LLM reranker".to_string())
            })?;

        let model = if config.model.is_empty() {
            "gpt-4.1-nano".to_string()
        } else {
            config.model.clone()
        };

        let base_url = "https://api.openai.com/v1".to_string();

        let client = Client::new();

        Ok(Self {
            client,
            api_key,
            model,
            base_url,
            config,
        })
    }

    fn build_prompt(&self, query: &str, memories: &[MemoryItem]) -> String {
        let mut prompt = format!(
            "You are a document relevance scorer. Score each document's relevance to the query on a scale of 0.0 to 1.0.\n\n\
            Query: {}\n\n\
            Documents to score:\n",
            query
        );

        for (i, mem) in memories.iter().enumerate() {
            prompt.push_str(&format!("\n[Document {}]: {}\n", i, mem.memory));
        }

        prompt.push_str("\n\nRespond with a JSON array of scores in order, e.g., [0.8, 0.2, 0.5].\n\
            Only output the JSON array, nothing else.");

        prompt
    }
}

#[async_trait]
impl Reranker for LlmReranker {
    async fn rerank(
        &self,
        query: &str,
        memories: Vec<MemoryItem>,
        limit: Option<usize>,
    ) -> RookResult<Vec<MemoryItem>> {
        if memories.is_empty() {
            return Ok(vec![]);
        }

        let prompt = self.build_prompt(query, &memories);

        let response = self
            .client
            .post(format!("{}/chat/completions", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&json!({
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant that scores document relevance."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.0
            }))
            .send()
            .await
            .map_err(|e| RookError::reranker(format!("Failed to call LLM API: {}", e)))?;

        if !response.status().is_success() {
            let error = response.text().await.unwrap_or_default();
            return Err(RookError::reranker(format!("LLM API error: {}", error)));
        }

        let result: serde_json::Value = response
            .json()
            .await
            .map_err(|e| RookError::reranker(format!("Failed to parse response: {}", e)))?;

        let content = result["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("[]");

        // Parse the scores from the LLM response
        let scores: Vec<f32> = serde_json::from_str(content.trim()).unwrap_or_else(|_| {
            // Try to extract numbers from the response
            content
                .chars()
                .filter(|c| c.is_numeric() || *c == '.' || *c == ',')
                .collect::<String>()
                .split(',')
                .filter_map(|s| s.trim().parse().ok())
                .collect()
        });

        // Create results with scores
        let mut results: Vec<MemoryItem> = memories
            .into_iter()
            .enumerate()
            .map(|(i, mut mem)| {
                mem.score = Some(scores.get(i).copied().unwrap_or(0.5));
                mem
            })
            .collect();

        // Sort by score descending
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Apply limit
        if let Some(n) = limit {
            results.truncate(n);
        }

        Ok(results)
    }

    fn model_name(&self) -> &str {
        &self.model
    }
}