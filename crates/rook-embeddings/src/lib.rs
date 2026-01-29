//! rook-embeddings - Embedding provider implementations for rook.
//!
//! This crate provides embedding provider implementations for use with
//! the rook memory layer.
//!
//! # Supported Providers
//!
//! - **OpenAI** (feature: `openai`) - text-embedding-3-small, text-embedding-3-large, etc.
//! - **Ollama** (feature: `ollama`) - Local embedding models via Ollama
//!
//! # Example
//!
//! ```ignore
//! use rook_embeddings::EmbedderFactory;
//!
//! // Create an OpenAI embedder
//! let embedder = EmbedderFactory::openai()?;
//!
//! // Or with a specific model
//! let embedder = EmbedderFactory::openai_with_model("text-embedding-3-large", 3072)?;
//!
//! // Create an Ollama embedder
//! let embedder = EmbedderFactory::ollama_with_model("nomic-embed-text", 768)?;
//! ```

mod factory;
mod ollama;
mod openai;

pub use factory::EmbedderFactory;
pub use ollama::OllamaEmbedder;
pub use openai::OpenAIEmbedder;

// Re-export core types for convenience
pub use rook_core::traits::{Embedder, EmbedderConfig, EmbedderProvider, EmbeddingAction};
