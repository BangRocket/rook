//! Retrieval module for advanced memory search.
//!
//! This module provides:
//! - Spreading activation for discovering related memories
//!   by propagating activation from seed nodes through the knowledge graph
//! - Full-text search with BM25 scoring (Tantivy)

mod activation;
mod config;
mod tantivy_search;

pub use activation::{spread_activation, spread_activation_by_id, ActivatedMemory, ActivationEdge, ActivationNode};
pub use config::SpreadingConfig;
pub use tantivy_search::{TantivySearcher, TextSearchResult};
