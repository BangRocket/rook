//! Ollama embedding provider implementation.

use async_trait::async_trait;

use rook_core::error::{RookError, RookResult};
use rook_core::traits::{Embedder, EmbedderConfig, EmbeddingAction};

#[cfg(feature = "ollama")]
use ollama_rs::{generation::embeddings::request::GenerateEmbeddingsRequest, Ollama};

/// Ollama embedding provider.
pub struct OllamaEmbedder {
    #[cfg(feature = "ollama")]
    client: Ollama,
    config: EmbedderConfig,
}

impl OllamaEmbedder {
    /// Create a new Ollama embedder.
    pub fn new(config: EmbedderConfig) -> RookResult<Self> {
        let base_url = config
            .base_url
            .clone()
            .unwrap_or_else(|| "http://localhost:11434".to_string());

        let url = url::Url::parse(&base_url)
            .map_err(|e| RookError::Configuration(format!("Invalid Ollama URL: {}", e)))?;

        let host = url.host_str().unwrap_or("localhost").to_string();
        let port = url.port().unwrap_or(11434);

        #[cfg(feature = "ollama")]
        let client = Ollama::new(format!("http://{}", host), port);

        Ok(Self {
            #[cfg(feature = "ollama")]
            client,
            config,
        })
    }
}

#[async_trait]
impl Embedder for OllamaEmbedder {
    #[cfg(feature = "ollama")]
    async fn embed(&self, text: &str, _action: Option<EmbeddingAction>) -> RookResult<Vec<f32>> {
        let request = GenerateEmbeddingsRequest::new(self.config.model.clone(), text.into());

        let response = self
            .client
            .generate_embeddings(request)
            .await
            .map_err(|e| RookError::embedding(format!("Ollama embedding error: {}", e)))?;

        // Convert f64 to f32
        let embedding: Vec<f32> = response.embeddings.into_iter().map(|v| v as f32).collect();

        Ok(embedding)
    }

    #[cfg(not(feature = "ollama"))]
    async fn embed(&self, _text: &str, _action: Option<EmbeddingAction>) -> RookResult<Vec<f32>> {
        Err(RookError::Configuration(
            "Ollama feature not enabled. Enable the 'ollama' feature.".to_string(),
        ))
    }

    fn dimension(&self) -> usize {
        self.config.embedding_dims
    }

    fn model_name(&self) -> &str {
        &self.config.model
    }
}
