//! Factory for creating LLM providers.

use std::sync::Arc;

use rook_core::config::LlmProvider;
use rook_core::error::{RookError, RookResult};
use rook_core::traits::{Llm, LlmConfig};

use crate::anthropic::AnthropicLlm;
use crate::ollama::OllamaLlm;
use crate::openai::OpenAIProvider;

/// Factory for creating LLM providers.
pub struct LlmFactory;

impl LlmFactory {
    /// Create an LLM provider from the given configuration.
    pub fn create(provider: LlmProvider, config: LlmConfig) -> RookResult<Arc<dyn Llm>> {
        match provider {
            LlmProvider::OpenAI => {
                let llm = OpenAIProvider::new(config)?;
                Ok(Arc::new(llm))
            }
            LlmProvider::Anthropic => {
                let llm = AnthropicLlm::new(config)?;
                Ok(Arc::new(llm))
            }
            LlmProvider::Ollama => {
                let llm = OllamaLlm::new(config)?;
                Ok(Arc::new(llm))
            }
            _ => Err(RookError::UnsupportedProvider {
                provider: format!("{:?}", provider),
            }),
        }
    }

    /// Create an OpenAI LLM provider with default configuration.
    pub fn openai() -> RookResult<Arc<dyn Llm>> {
        Self::create(LlmProvider::OpenAI, LlmConfig::default())
    }

    /// Create an OpenAI LLM provider with a specific model.
    pub fn openai_with_model(model: impl Into<String>) -> RookResult<Arc<dyn Llm>> {
        let config = LlmConfig {
            model: model.into(),
            ..Default::default()
        };
        Self::create(LlmProvider::OpenAI, config)
    }

    /// Create an Anthropic LLM provider with default configuration.
    pub fn anthropic() -> RookResult<Arc<dyn Llm>> {
        Self::create(LlmProvider::Anthropic, LlmConfig::default())
    }

    /// Create an Anthropic LLM provider with a specific model.
    pub fn anthropic_with_model(model: impl Into<String>) -> RookResult<Arc<dyn Llm>> {
        let config = LlmConfig {
            model: model.into(),
            ..Default::default()
        };
        Self::create(LlmProvider::Anthropic, config)
    }

    /// Create an Ollama LLM provider with default configuration.
    pub fn ollama() -> RookResult<Arc<dyn Llm>> {
        Self::create(LlmProvider::Ollama, LlmConfig::default())
    }

    /// Create an Ollama LLM provider with a specific model.
    pub fn ollama_with_model(model: impl Into<String>) -> RookResult<Arc<dyn Llm>> {
        let config = LlmConfig {
            model: model.into(),
            ..Default::default()
        };
        Self::create(LlmProvider::Ollama, config)
    }
}
