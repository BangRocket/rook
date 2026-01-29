//! Retrieval module for advanced memory search.
//!
//! Provides multi-signal hybrid retrieval with four modes:
//! - Quick: Vector-only (fastest)
//! - Standard: Vector + BM25 + Activation with RRF fusion
//! - Precise: All signals with linear fusion
//! - Cognitive: Activation + FSRS weighting

mod activation;
mod config;
mod dedup;
mod engine;
mod fusion;
mod modes;
mod tantivy_search;

pub use activation::{
    spread_activation, spread_activation_by_id, ActivatedMemory, ActivationEdge, ActivationNode,
};
pub use config::SpreadingConfig;
pub use dedup::{DeduplicatableResult, DeduplicationConfig, Deduplicator};
pub use engine::{
    ActivationGraph, FsrsMemoryState, FsrsStateProvider, RetrievalEngine, RetrievalResult,
    RetrievalSignals, VectorSearcher,
};
pub use fusion::{FusionInputs, LinearFusion, RrfFusion};
pub use modes::{RetrievalConfig, RetrievalMode};
pub use tantivy_search::{TantivySearcher, TextSearchResult};
