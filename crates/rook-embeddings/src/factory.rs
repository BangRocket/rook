//! Factory for creating embedding providers.

use std::sync::Arc;

use rook_core::error::{RookError, RookResult};
use rook_core::traits::{Embedder, EmbedderConfig, EmbedderProvider};

use crate::ollama::OllamaEmbedder;
use crate::openai::OpenAIEmbedder;

/// Factory for creating embedding providers.
pub struct EmbedderFactory;

impl EmbedderFactory {
    /// Create an embedder from the given configuration.
    pub fn create(provider: EmbedderProvider, config: EmbedderConfig) -> RookResult<Arc<dyn Embedder>> {
        match provider {
            EmbedderProvider::OpenAI => {
                let embedder = OpenAIEmbedder::new(config)?;
                Ok(Arc::new(embedder))
            }
            EmbedderProvider::Ollama => {
                let embedder = OllamaEmbedder::new(config)?;
                Ok(Arc::new(embedder))
            }
            _ => Err(RookError::UnsupportedProvider {
                provider: format!("{:?}", provider),
            }),
        }
    }

    /// Create an OpenAI embedder with default configuration.
    pub fn openai() -> RookResult<Arc<dyn Embedder>> {
        Self::create(EmbedderProvider::OpenAI, EmbedderConfig::default())
    }

    /// Create an OpenAI embedder with a specific model.
    pub fn openai_with_model(model: impl Into<String>, dims: usize) -> RookResult<Arc<dyn Embedder>> {
        let config = EmbedderConfig {
            model: model.into(),
            embedding_dims: dims,
            ..Default::default()
        };
        Self::create(EmbedderProvider::OpenAI, config)
    }

    /// Create an Ollama embedder with default configuration.
    pub fn ollama() -> RookResult<Arc<dyn Embedder>> {
        let config = EmbedderConfig {
            model: "nomic-embed-text".to_string(),
            embedding_dims: 768,
            ..Default::default()
        };
        Self::create(EmbedderProvider::Ollama, config)
    }

    /// Create an Ollama embedder with a specific model.
    pub fn ollama_with_model(model: impl Into<String>, dims: usize) -> RookResult<Arc<dyn Embedder>> {
        let config = EmbedderConfig {
            model: model.into(),
            embedding_dims: dims,
            ..Default::default()
        };
        Self::create(EmbedderProvider::Ollama, config)
    }
}
