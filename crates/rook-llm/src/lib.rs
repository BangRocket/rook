//! rook-llm - LLM provider implementations for rook.
//!
//! This crate provides LLM (Large Language Model) provider implementations
//! for use with the rook memory layer.
//!
//! # Supported Providers
//!
//! - **OpenAI** (feature: `openai`) - GPT-4, GPT-3.5, etc.
//! - **Anthropic** (feature: `anthropic`) - Claude 3.5, Claude 3, etc.
//! - **Ollama** (feature: `ollama`) - Local models via Ollama
//!
//! # Example
//!
//! ```ignore
//! use rook_llm::LlmFactory;
//!
//! // Create an OpenAI LLM
//! let llm = LlmFactory::openai()?;
//!
//! // Or with a specific model
//! let llm = LlmFactory::openai_with_model("gpt-4-turbo")?;
//!
//! // Create an Anthropic LLM
//! let llm = LlmFactory::anthropic_with_model("claude-3-5-sonnet-20240620")?;
//! ```

mod anthropic;
mod factory;
mod ollama;
mod openai;

pub use anthropic::AnthropicLlm;
pub use factory::LlmFactory;
pub use ollama::OllamaLlm;
pub use openai::OpenAIProvider;

// Re-export core types for convenience
pub use rook_core::config::LlmProvider;
pub use rook_core::traits::{GenerationOptions, Llm, LlmConfig, LlmResponse, ResponseFormat};
