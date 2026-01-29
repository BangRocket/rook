//! OpenAI embedding provider implementation.

use async_trait::async_trait;

use rook_core::error::{RookError, RookResult};
use rook_core::traits::{Embedder, EmbedderConfig, EmbeddingAction};

#[cfg(feature = "openai")]
use async_openai::{
    config::OpenAIConfig,
    types::{CreateEmbeddingRequest, EmbeddingInput},
    Client,
};

/// OpenAI embedding provider.
pub struct OpenAIEmbedder {
    #[cfg(feature = "openai")]
    client: Client<OpenAIConfig>,
    config: EmbedderConfig,
}

impl OpenAIEmbedder {
    /// Create a new OpenAI embedder.
    pub fn new(config: EmbedderConfig) -> RookResult<Self> {
        let api_key = config
            .api_key
            .clone()
            .or_else(|| std::env::var("OPENAI_API_KEY").ok())
            .ok_or_else(|| {
                RookError::Configuration("OpenAI API key not found. Set OPENAI_API_KEY environment variable or provide api_key in config.".to_string())
            })?;

        #[cfg(feature = "openai")]
        let openai_config = if let Some(ref base_url) = config.base_url {
            OpenAIConfig::new()
                .with_api_key(api_key)
                .with_api_base(base_url)
        } else {
            OpenAIConfig::new().with_api_key(api_key)
        };

        #[cfg(feature = "openai")]
        let client = Client::with_config(openai_config);

        Ok(Self {
            #[cfg(feature = "openai")]
            client,
            config,
        })
    }
}

#[async_trait]
impl Embedder for OpenAIEmbedder {
    #[cfg(feature = "openai")]
    async fn embed(&self, text: &str, _action: Option<EmbeddingAction>) -> RookResult<Vec<f32>> {
        let request = CreateEmbeddingRequest {
            model: self.config.model.clone(),
            input: EmbeddingInput::String(text.to_string()),
            ..Default::default()
        };

        let response = self
            .client
            .embeddings()
            .create(request)
            .await
            .map_err(|e| RookError::embedding(format!("OpenAI embedding error: {}", e)))?;

        let embedding = response
            .data
            .first()
            .ok_or_else(|| RookError::embedding("No embedding returned"))?;

        Ok(embedding.embedding.clone())
    }

    #[cfg(not(feature = "openai"))]
    async fn embed(&self, _text: &str, _action: Option<EmbeddingAction>) -> RookResult<Vec<f32>> {
        Err(RookError::Configuration(
            "OpenAI feature not enabled. Enable the 'openai' feature.".to_string(),
        ))
    }

    #[cfg(feature = "openai")]
    async fn embed_batch(
        &self,
        texts: &[String],
        _action: Option<EmbeddingAction>,
    ) -> RookResult<Vec<Vec<f32>>> {
        let request = CreateEmbeddingRequest {
            model: self.config.model.clone(),
            input: EmbeddingInput::StringArray(texts.to_vec()),
            ..Default::default()
        };

        let response = self
            .client
            .embeddings()
            .create(request)
            .await
            .map_err(|e| RookError::embedding(format!("OpenAI embedding error: {}", e)))?;

        let embeddings: Vec<Vec<f32>> = response.data.into_iter().map(|e| e.embedding).collect();

        Ok(embeddings)
    }

    fn dimension(&self) -> usize {
        self.config.embedding_dims
    }

    fn model_name(&self) -> &str {
        &self.config.model
    }
}
