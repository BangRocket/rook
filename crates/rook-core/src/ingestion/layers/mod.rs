//! Detection layers for smart ingestion.
//!
//! Each layer represents a different strategy for detecting duplicates,
//! updates, and contradictions. Layers are ordered by computational cost:
//!
//! 1. Embedding similarity (~1ms) - vector cosine similarity
//! 2. Keyword pattern (~1ms) - regex-based negation/contradiction detection
//! 3. Temporal conflict (~1ms) - date-based conflict detection
//! 4. Semantic LLM (~500ms) - LLM-based semantic analysis

pub mod embedding;
pub mod keyword;
pub mod temporal;

pub use embedding::{EmbeddingResult, EmbeddingSimilarityLayer, SimilarityCandidate};
pub use keyword::{KeywordNegationLayer, KeywordResult};
pub use temporal::{TemporalConflictLayer, TemporalResult};
