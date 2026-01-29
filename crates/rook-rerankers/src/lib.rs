//! rook-rerankers - Reranker implementations for rook.
//!
//! This crate provides reranker implementations for reordering
//! search results based on relevance.
//!
//! # Supported Backends
//!
//! - **Cohere** (feature: `cohere`) - Cohere Rerank API
//! - **LLM** (feature: `llm`) - LLM-based reranking

mod factory;

#[cfg(feature = "cohere")]
mod cohere;

#[cfg(feature = "llm")]
mod llm_reranker;

pub use factory::RerankerFactory;

#[cfg(feature = "cohere")]
pub use cohere::CohereReranker;

#[cfg(feature = "llm")]
pub use llm_reranker::LlmReranker;

// Re-export core types
pub use rook_core::traits::{Reranker, RerankerConfig, RerankerProvider};
