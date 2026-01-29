//! Reranker trait and related types.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::RookResult;
use crate::types::MemoryItem;

/// Core Reranker trait - all reranker providers implement this.
#[async_trait]
pub trait Reranker: Send + Sync {
    /// Rerank memories based on relevance to query.
    async fn rerank(
        &self,
        query: &str,
        memories: Vec<MemoryItem>,
        limit: Option<usize>,
    ) -> RookResult<Vec<MemoryItem>>;

    /// Get the model name.
    fn model_name(&self) -> &str;
}

/// Reranker configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankerConfig {
    /// Provider type.
    pub provider: RerankerProvider,
    /// Model name/identifier.
    pub model: String,
    /// API key (if not using environment variable).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    /// Top-N results to rerank.
    #[serde(default = "default_top_n")]
    pub top_n: usize,
}

fn default_top_n() -> usize {
    5
}

impl Default for RerankerConfig {
    fn default() -> Self {
        Self {
            provider: RerankerProvider::Cohere,
            model: "rerank-english-v2.0".to_string(),
            api_key: None,
            top_n: default_top_n(),
        }
    }
}

/// Reranker provider type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum RerankerProvider {
    #[default]
    Cohere,
    HuggingFace,
    Llm,
    SentenceTransformer,
    ZeroEntropy,
}
