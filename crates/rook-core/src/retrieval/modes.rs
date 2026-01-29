//! Retrieval mode definitions and configurations.
//!
//! Four retrieval modes optimize for different use cases:
//! - Quick: Vector-only, fastest
//! - Standard: Vector + BM25 + Activation with RRF fusion
//! - Precise: All signals with linear fusion
//! - Cognitive: Spreading activation + FSRS weighting

use serde::{Deserialize, Serialize};

use super::config::SpreadingConfig;
use super::dedup::DeduplicationConfig;
use super::fusion::{LinearFusion, RrfFusion};

/// Retrieval mode determines which signals are combined and how.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum RetrievalMode {
    /// Vector search only - fastest retrieval.
    /// Use when: Speed is critical, query is well-formed for embedding.
    Quick,

    /// Vector + BM25 + Spreading Activation with RRF fusion.
    /// Use when: Balanced retrieval quality and speed (default).
    #[default]
    Standard,

    /// All signals (vector, BM25, activation, FSRS) with linear fusion.
    /// Use when: Maximum accuracy is needed, latency is acceptable.
    Precise,

    /// Spreading activation + FSRS retrievability weighting.
    /// Use when: Human-like memory retrieval with decay patterns.
    Cognitive,
}

impl RetrievalMode {
    /// Check if this mode uses vector search.
    pub fn uses_vector(&self) -> bool {
        true // All modes use vector search
    }

    /// Check if this mode uses BM25 text search.
    pub fn uses_bm25(&self) -> bool {
        matches!(self, Self::Standard | Self::Precise)
    }

    /// Check if this mode uses spreading activation.
    pub fn uses_activation(&self) -> bool {
        matches!(self, Self::Standard | Self::Precise | Self::Cognitive)
    }

    /// Check if this mode uses FSRS retrievability.
    pub fn uses_fsrs(&self) -> bool {
        matches!(self, Self::Precise | Self::Cognitive)
    }

    /// Check if this mode uses RRF fusion.
    pub fn uses_rrf(&self) -> bool {
        matches!(self, Self::Standard)
    }

    /// Check if this mode uses linear fusion.
    pub fn uses_linear(&self) -> bool {
        matches!(self, Self::Precise | Self::Cognitive)
    }
}

/// Configuration for retrieval operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalConfig {
    /// Retrieval mode.
    pub mode: RetrievalMode,
    /// Maximum number of results to return.
    pub limit: usize,
    /// Spreading activation configuration.
    pub spreading: SpreadingConfig,
    /// RRF fusion configuration (used in Standard mode).
    pub rrf: RrfFusion,
    /// Linear fusion weights (used in Precise/Cognitive modes).
    pub linear: LinearFusion,
    /// Deduplication configuration.
    pub dedup: DeduplicationConfig,
    /// Whether to deduplicate results.
    pub enable_dedup: bool,
    /// Oversample factor for fusion (fetch more results before fusion).
    pub oversample_factor: usize,
}

impl Default for RetrievalConfig {
    fn default() -> Self {
        Self {
            mode: RetrievalMode::Standard,
            limit: 10,
            spreading: SpreadingConfig::default(),
            rrf: RrfFusion::default(),
            linear: LinearFusion::default(),
            dedup: DeduplicationConfig::default(),
            enable_dedup: true,
            oversample_factor: 2,
        }
    }
}

impl RetrievalConfig {
    /// Create config for Quick mode.
    pub fn quick(limit: usize) -> Self {
        Self {
            mode: RetrievalMode::Quick,
            limit,
            enable_dedup: false, // Speed optimization
            oversample_factor: 1,
            ..Default::default()
        }
    }

    /// Create config for Standard mode.
    pub fn standard(limit: usize) -> Self {
        Self {
            mode: RetrievalMode::Standard,
            limit,
            ..Default::default()
        }
    }

    /// Create config for Precise mode.
    pub fn precise(limit: usize) -> Self {
        Self {
            mode: RetrievalMode::Precise,
            limit,
            linear: LinearFusion::precise(),
            oversample_factor: 3, // More oversampling for accuracy
            ..Default::default()
        }
    }

    /// Create config for Cognitive mode.
    pub fn cognitive(limit: usize) -> Self {
        Self {
            mode: RetrievalMode::Cognitive,
            limit,
            linear: LinearFusion::cognitive(),
            spreading: SpreadingConfig::wide(), // Wider activation spread
            ..Default::default()
        }
    }

    /// Set custom limit.
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }

    /// Enable or disable deduplication.
    pub fn with_dedup(mut self, enable: bool) -> Self {
        self.enable_dedup = enable;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_retrieval_mode_helpers() {
        // Quick mode
        assert!(RetrievalMode::Quick.uses_vector());
        assert!(!RetrievalMode::Quick.uses_bm25());
        assert!(!RetrievalMode::Quick.uses_activation());
        assert!(!RetrievalMode::Quick.uses_fsrs());
        assert!(!RetrievalMode::Quick.uses_rrf());
        assert!(!RetrievalMode::Quick.uses_linear());

        // Standard mode
        assert!(RetrievalMode::Standard.uses_vector());
        assert!(RetrievalMode::Standard.uses_bm25());
        assert!(RetrievalMode::Standard.uses_activation());
        assert!(!RetrievalMode::Standard.uses_fsrs());
        assert!(RetrievalMode::Standard.uses_rrf());
        assert!(!RetrievalMode::Standard.uses_linear());

        // Precise mode
        assert!(RetrievalMode::Precise.uses_vector());
        assert!(RetrievalMode::Precise.uses_bm25());
        assert!(RetrievalMode::Precise.uses_activation());
        assert!(RetrievalMode::Precise.uses_fsrs());
        assert!(!RetrievalMode::Precise.uses_rrf());
        assert!(RetrievalMode::Precise.uses_linear());

        // Cognitive mode
        assert!(RetrievalMode::Cognitive.uses_vector());
        assert!(!RetrievalMode::Cognitive.uses_bm25());
        assert!(RetrievalMode::Cognitive.uses_activation());
        assert!(RetrievalMode::Cognitive.uses_fsrs());
        assert!(!RetrievalMode::Cognitive.uses_rrf());
        assert!(RetrievalMode::Cognitive.uses_linear());
    }

    #[test]
    fn test_config_presets() {
        let quick = RetrievalConfig::quick(10);
        assert_eq!(quick.mode, RetrievalMode::Quick);
        assert!(!quick.enable_dedup);
        assert_eq!(quick.oversample_factor, 1);

        let standard = RetrievalConfig::standard(10);
        assert_eq!(standard.mode, RetrievalMode::Standard);
        assert!(standard.enable_dedup);

        let precise = RetrievalConfig::precise(10);
        assert_eq!(precise.mode, RetrievalMode::Precise);
        assert_eq!(precise.oversample_factor, 3);

        let cognitive = RetrievalConfig::cognitive(10);
        assert_eq!(cognitive.mode, RetrievalMode::Cognitive);
    }

    #[test]
    fn test_default_mode() {
        let default_mode = RetrievalMode::default();
        assert_eq!(default_mode, RetrievalMode::Standard);
    }
}
