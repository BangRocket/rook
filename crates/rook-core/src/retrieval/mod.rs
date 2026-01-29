//! Retrieval module for advanced memory search.
//!
//! This module provides:
//! - Spreading activation for discovering related memories
//!   by propagating activation from seed nodes through the knowledge graph
//! - Full-text search with BM25 scoring (Tantivy)
//! - Score fusion (RRF and linear weighted)

mod activation;
mod config;
mod fusion;
mod tantivy_search;

pub use activation::{
    spread_activation, spread_activation_by_id, ActivatedMemory, ActivationEdge, ActivationNode,
};
pub use config::SpreadingConfig;
pub use fusion::{FusionInputs, LinearFusion, RrfFusion};
pub use tantivy_search::{TantivySearcher, TextSearchResult};
