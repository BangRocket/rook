//! Cohere reranker implementation.

use async_trait::async_trait;

use rook_core::error::{RookError, RookResult};
use rook_core::traits::{Reranker, RerankerConfig};
use rook_core::types::MemoryItem;

use reqwest::Client;
use serde::{Deserialize, Serialize};

/// Cohere reranker implementation.
pub struct CohereReranker {
    client: Client,
    api_key: String,
    model: String,
    #[allow(dead_code)]
    config: RerankerConfig,
}

#[derive(Debug, Serialize)]
struct CohereRerankRequest {
    model: String,
    query: String,
    documents: Vec<String>,
    top_n: Option<usize>,
    return_documents: bool,
}

#[derive(Debug, Deserialize)]
struct CohereRerankResponse {
    results: Vec<CohereRerankResult>,
}

#[derive(Debug, Deserialize)]
struct CohereRerankResult {
    index: usize,
    relevance_score: f32,
}

impl CohereReranker {
    /// Create a new Cohere reranker.
    pub fn new(config: RerankerConfig) -> RookResult<Self> {
        let api_key = config
            .api_key
            .clone()
            .or_else(|| std::env::var("COHERE_API_KEY").ok())
            .ok_or_else(|| {
                RookError::Configuration(
                    "Cohere API key required. Set COHERE_API_KEY or provide api_key.".to_string(),
                )
            })?;

        let model = config.model.clone();
        let client = Client::new();

        Ok(Self {
            client,
            api_key,
            model,
            config,
        })
    }
}

#[async_trait]
impl Reranker for CohereReranker {
    async fn rerank(
        &self,
        query: &str,
        memories: Vec<MemoryItem>,
        limit: Option<usize>,
    ) -> RookResult<Vec<MemoryItem>> {
        if memories.is_empty() {
            return Ok(vec![]);
        }

        let documents: Vec<String> = memories.iter().map(|m| m.memory.clone()).collect();

        let request = CohereRerankRequest {
            model: self.model.clone(),
            query: query.to_string(),
            documents,
            top_n: limit,
            return_documents: false,
        };

        let response = self
            .client
            .post("https://api.cohere.ai/v1/rerank")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .map_err(|e| RookError::reranker(format!("Failed to call Cohere API: {}", e)))?;

        if !response.status().is_success() {
            let error = response.text().await.unwrap_or_default();
            return Err(RookError::reranker(format!(
                "Cohere API error: {}",
                error
            )));
        }

        let result: CohereRerankResponse = response
            .json()
            .await
            .map_err(|e| RookError::reranker(format!("Failed to parse response: {}", e)))?;

        let mut reranked: Vec<MemoryItem> = result
            .results
            .into_iter()
            .filter_map(|r| {
                memories.get(r.index).map(|m| {
                    let mut item = m.clone();
                    item.score = Some(r.relevance_score);
                    item
                })
            })
            .collect();

        // Sort by score descending
        reranked.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(reranked)
    }

    fn model_name(&self) -> &str {
        &self.model
    }
}