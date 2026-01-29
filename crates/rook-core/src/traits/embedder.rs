//! Embedder trait and related types.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::RookResult;

/// The action context for embedding (some models use different embeddings).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EmbeddingAction {
    /// Adding to the store.
    #[default]
    Add,
    /// Searching the store.
    Search,
    /// Updating in the store.
    Update,
}

/// Core Embedder trait - all embedding providers implement this.
#[async_trait]
pub trait Embedder: Send + Sync {
    /// Generate embedding for a single text.
    async fn embed(&self, text: &str, action: Option<EmbeddingAction>) -> RookResult<Vec<f32>>;

    /// Generate embeddings for multiple texts (batch).
    async fn embed_batch(
        &self,
        texts: &[String],
        action: Option<EmbeddingAction>,
    ) -> RookResult<Vec<Vec<f32>>> {
        // Default implementation: sequential embedding
        let mut embeddings = Vec::with_capacity(texts.len());
        for text in texts {
            embeddings.push(self.embed(text, action).await?);
        }
        Ok(embeddings)
    }

    /// Get the dimension of the embeddings.
    fn dimension(&self) -> usize;

    /// Get the model name.
    fn model_name(&self) -> &str;
}

/// Embedder configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbedderConfig {
    /// Model name/identifier.
    pub model: String,
    /// Embedding dimensions.
    #[serde(default = "default_embedding_dims")]
    pub embedding_dims: usize,
    /// API key (if not using environment variable).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    /// Base URL for API.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
}

fn default_embedding_dims() -> usize {
    1536
}

impl Default for EmbedderConfig {
    fn default() -> Self {
        Self {
            model: "text-embedding-3-small".to_string(),
            embedding_dims: default_embedding_dims(),
            api_key: None,
            base_url: None,
        }
    }
}

/// Embedder provider type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum EmbedderProvider {
    #[default]
    OpenAI,
    Anthropic,
    Ollama,
    HuggingFace,
    Cohere,
    VertexAI,
    AzureOpenAI,
    AwsBedrock,
}
