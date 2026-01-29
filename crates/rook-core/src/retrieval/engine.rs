//! Retrieval engine orchestrating multi-signal hybrid search.
//!
//! Combines vector similarity, BM25 text search, spreading activation,
//! and FSRS retrievability into unified retrieval with configurable modes.

use std::collections::HashMap;
use std::sync::Arc;

use chrono::Utc;
use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};

use crate::error::RookResult;

use super::activation::ActivatedMemory;
use super::config::SpreadingConfig;
use super::dedup::{DeduplicatableResult, Deduplicator};
use super::fusion::{FusionInputs, LinearFusion};
use super::modes::{RetrievalConfig, RetrievalMode};
use super::tantivy_search::{TantivySearcher, TextSearchResult};

/// A retrieval result with combined scores.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalResult {
    /// Memory ID.
    pub id: String,
    /// Final combined score (0-1).
    pub score: f32,
    /// Individual signal scores for debugging/analysis.
    pub signals: RetrievalSignals,
}

/// Individual signal scores contributing to a result.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RetrievalSignals {
    /// Vector similarity score (cosine, 0-1).
    pub vector: Option<f32>,
    /// BM25 score (normalized 0-1).
    pub bm25: Option<f32>,
    /// Spreading activation score (0-1).
    pub activation: Option<f32>,
    /// FSRS retrievability score (0-1).
    pub fsrs: Option<f32>,
}

/// FSRS memory state for retrievability calculation.
#[derive(Debug, Clone)]
pub struct FsrsMemoryState {
    /// Memory stability (days until ~90% forgetting).
    pub stability: f32,
    /// Memory difficulty (0.0-1.0).
    #[allow(dead_code)]
    pub difficulty: f32,
    /// Last review timestamp.
    pub last_review: Option<chrono::DateTime<Utc>>,
}

/// Trait for vector search backend.
#[async_trait::async_trait]
pub trait VectorSearcher: Send + Sync {
    /// Search for similar vectors.
    async fn search(
        &self,
        query_embedding: &[f32],
        limit: usize,
    ) -> RookResult<Vec<(String, f32)>>; // (id, similarity)

    /// Get embedding for a memory ID.
    async fn get_embedding(&self, id: &str) -> RookResult<Option<Vec<f32>>>;

    /// Get embeddings for multiple IDs (batch).
    async fn get_embeddings(&self, ids: &[String]) -> RookResult<HashMap<String, Vec<f32>>>;
}

/// Trait for graph backend providing activation data.
pub trait ActivationGraph: Send + Sync {
    /// Get memory IDs connected to seed memories.
    fn spread_from_seeds(
        &self,
        seeds: &[(String, f32)],
        config: &SpreadingConfig,
    ) -> Vec<ActivatedMemory>;
}

/// Trait for FSRS state backend.
pub trait FsrsStateProvider: Send + Sync {
    /// Get FSRS state for a memory.
    fn get_state(&self, id: &str) -> Option<FsrsMemoryState>;
}

/// Main retrieval engine combining all search methods.
pub struct RetrievalEngine<V, G, F>
where
    V: VectorSearcher,
    G: ActivationGraph,
    F: FsrsStateProvider,
{
    vector_searcher: Arc<V>,
    text_searcher: Option<Arc<TantivySearcher>>,
    graph: Option<Arc<G>>,
    fsrs_provider: Option<Arc<F>>,
}

impl<V, G, F> RetrievalEngine<V, G, F>
where
    V: VectorSearcher,
    G: ActivationGraph,
    F: FsrsStateProvider,
{
    /// Create a new retrieval engine with vector search only.
    pub fn new(vector_searcher: Arc<V>) -> Self {
        Self {
            vector_searcher,
            text_searcher: None,
            graph: None,
            fsrs_provider: None,
        }
    }

    /// Add Tantivy text search capability.
    pub fn with_text_search(mut self, searcher: Arc<TantivySearcher>) -> Self {
        self.text_searcher = Some(searcher);
        self
    }

    /// Add graph for spreading activation.
    pub fn with_graph(mut self, graph: Arc<G>) -> Self {
        self.graph = Some(graph);
        self
    }

    /// Add FSRS state provider.
    pub fn with_fsrs(mut self, provider: Arc<F>) -> Self {
        self.fsrs_provider = Some(provider);
        self
    }

    /// Retrieve memories matching a query.
    ///
    /// # Arguments
    /// * `query` - Text query for search
    /// * `query_embedding` - Pre-computed embedding for the query
    /// * `config` - Retrieval configuration (mode, limits, etc.)
    ///
    /// # Returns
    /// Vector of RetrievalResult sorted by score descending.
    pub async fn retrieve(
        &self,
        query: &str,
        query_embedding: &[f32],
        config: &RetrievalConfig,
    ) -> RookResult<Vec<RetrievalResult>> {
        let fetch_limit = config.limit * config.oversample_factor;

        match config.mode {
            RetrievalMode::Quick => self.retrieve_quick(query_embedding, config.limit).await,
            RetrievalMode::Standard => {
                self.retrieve_standard(query, query_embedding, config, fetch_limit)
                    .await
            }
            RetrievalMode::Precise => {
                self.retrieve_precise(query, query_embedding, config, fetch_limit)
                    .await
            }
            RetrievalMode::Cognitive => {
                self.retrieve_cognitive(query_embedding, config, fetch_limit)
                    .await
            }
        }
    }

    /// Quick mode: Vector search only.
    async fn retrieve_quick(
        &self,
        query_embedding: &[f32],
        limit: usize,
    ) -> RookResult<Vec<RetrievalResult>> {
        let vector_results = self.vector_searcher.search(query_embedding, limit).await?;

        Ok(vector_results
            .into_iter()
            .map(|(id, score)| RetrievalResult {
                id,
                score,
                signals: RetrievalSignals {
                    vector: Some(score),
                    ..Default::default()
                },
            })
            .collect())
    }

    /// Standard mode: Vector + BM25 + Activation with RRF fusion.
    async fn retrieve_standard(
        &self,
        query: &str,
        query_embedding: &[f32],
        config: &RetrievalConfig,
        fetch_limit: usize,
    ) -> RookResult<Vec<RetrievalResult>> {
        // Fetch from all sources
        let vector_results = self.vector_searcher.search(query_embedding, fetch_limit).await?;

        let text_results = if let Some(searcher) = &self.text_searcher {
            searcher.search(query, fetch_limit)?
        } else {
            vec![]
        };

        let activation_results = if let Some(graph) = &self.graph {
            // Use top vector results as seeds
            let seeds: Vec<_> = vector_results
                .iter()
                .take(5)
                .map(|(id, score)| (id.clone(), *score))
                .collect();
            graph.spread_from_seeds(&seeds, &config.spreading)
        } else {
            vec![]
        };

        // Convert to ranked lists for RRF
        let vector_ranked: Vec<_> = vector_results.clone();
        let text_ranked: Vec<_> = text_results
            .iter()
            .map(|r| (r.id.clone(), r.normalized_score))
            .collect();
        let activation_ranked: Vec<_> = activation_results
            .iter()
            .map(|r| (r.memory_id.clone(), r.activation))
            .collect();

        // RRF fusion
        let fused = config
            .rrf
            .fuse(vec![vector_ranked, text_ranked, activation_ranked]);

        // Build results with signals
        let mut results =
            self.build_results_with_signals(fused, &vector_results, &text_results, &activation_results, None);

        // Apply deduplication if enabled
        if config.enable_dedup {
            results = self.deduplicate_results(results, &config.dedup).await?;
        }

        // Limit results
        results.truncate(config.limit);
        Ok(results)
    }

    /// Precise mode: All signals with linear fusion.
    async fn retrieve_precise(
        &self,
        query: &str,
        query_embedding: &[f32],
        config: &RetrievalConfig,
        fetch_limit: usize,
    ) -> RookResult<Vec<RetrievalResult>> {
        // Fetch from all sources
        let vector_results = self.vector_searcher.search(query_embedding, fetch_limit).await?;

        let text_results = if let Some(searcher) = &self.text_searcher {
            searcher.search(query, fetch_limit)?
        } else {
            vec![]
        };

        let activation_results = if let Some(graph) = &self.graph {
            let seeds: Vec<_> = vector_results
                .iter()
                .take(5)
                .map(|(id, score)| (id.clone(), *score))
                .collect();
            graph.spread_from_seeds(&seeds, &config.spreading)
        } else {
            vec![]
        };

        // Collect FSRS scores
        let fsrs_scores = self.collect_fsrs_scores(&vector_results);

        // Linear fusion
        let mut results = self.linear_fuse_all(
            &vector_results,
            &text_results,
            &activation_results,
            &fsrs_scores,
            &config.linear,
        );

        // Apply deduplication
        if config.enable_dedup {
            results = self.deduplicate_results(results, &config.dedup).await?;
        }

        results.truncate(config.limit);
        Ok(results)
    }

    /// Cognitive mode: Activation + FSRS with linear fusion.
    async fn retrieve_cognitive(
        &self,
        query_embedding: &[f32],
        config: &RetrievalConfig,
        fetch_limit: usize,
    ) -> RookResult<Vec<RetrievalResult>> {
        // Vector search for seeds and as one signal
        let vector_results = self.vector_searcher.search(query_embedding, fetch_limit).await?;

        // Spreading activation (emphasized in cognitive mode)
        let activation_results = if let Some(graph) = &self.graph {
            let seeds: Vec<_> = vector_results
                .iter()
                .take(10) // More seeds for cognitive
                .map(|(id, score)| (id.clone(), *score))
                .collect();
            graph.spread_from_seeds(&seeds, &config.spreading)
        } else {
            vec![]
        };

        // FSRS retrievability (emphasized in cognitive mode)
        let fsrs_scores = self.collect_fsrs_scores(&vector_results);

        // Linear fusion with cognitive weights
        let mut results = self.linear_fuse_all(
            &vector_results,
            &[], // No BM25 in cognitive mode
            &activation_results,
            &fsrs_scores,
            &config.linear,
        );

        // Apply deduplication
        if config.enable_dedup {
            results = self.deduplicate_results(results, &config.dedup).await?;
        }

        results.truncate(config.limit);
        Ok(results)
    }

    /// Collect FSRS retrievability scores for memories.
    fn collect_fsrs_scores(&self, results: &[(String, f32)]) -> HashMap<String, f32> {
        let mut scores = HashMap::new();

        if let Some(provider) = &self.fsrs_provider {
            let now = Utc::now();

            for (id, _) in results {
                if let Some(state) = provider.get_state(id) {
                    let days_elapsed = state
                        .last_review
                        .map(|lr| (now - lr).num_seconds() as f32 / 86400.0)
                        .unwrap_or(0.0);

                    // FSRS retrievability formula: R = (1 + t/S)^(-1/D)
                    // where t=days_elapsed, S=stability, D=decay (~0.2 for FSRS-6)
                    let retrievability = (1.0 + days_elapsed / state.stability).powf(-1.0 / 0.2);
                    scores.insert(id.clone(), retrievability.clamp(0.0, 1.0));
                }
            }
        }

        scores
    }

    /// Build result structs with signal information.
    fn build_results_with_signals(
        &self,
        fused: Vec<(String, f32)>,
        vector_results: &[(String, f32)],
        text_results: &[TextSearchResult],
        activation_results: &[ActivatedMemory],
        fsrs_scores: Option<&HashMap<String, f32>>,
    ) -> Vec<RetrievalResult> {
        let vector_map: HashMap<_, _> = vector_results.iter().cloned().collect();
        let text_map: HashMap<_, _> = text_results
            .iter()
            .map(|r| (r.id.clone(), r.normalized_score))
            .collect();
        let activation_map: HashMap<_, _> = activation_results
            .iter()
            .map(|r| (r.memory_id.clone(), r.activation))
            .collect();

        fused
            .into_iter()
            .map(|(id, score)| RetrievalResult {
                signals: RetrievalSignals {
                    vector: vector_map.get(&id).copied(),
                    bm25: text_map.get(&id).copied(),
                    activation: activation_map.get(&id).copied(),
                    fsrs: fsrs_scores.and_then(|m| m.get(&id).copied()),
                },
                id,
                score,
            })
            .collect()
    }

    /// Linear fusion of all signals.
    fn linear_fuse_all(
        &self,
        vector_results: &[(String, f32)],
        text_results: &[TextSearchResult],
        activation_results: &[ActivatedMemory],
        fsrs_scores: &HashMap<String, f32>,
        weights: &LinearFusion,
    ) -> Vec<RetrievalResult> {
        // Collect all unique IDs
        let mut all_ids: HashMap<String, FusionInputs> = HashMap::new();

        for (id, score) in vector_results {
            all_ids.entry(id.clone()).or_default().vector = *score;
        }

        for r in text_results {
            all_ids
                .entry(r.id.clone())
                .or_default()
                .bm25_normalized = r.normalized_score;
        }

        for r in activation_results {
            all_ids
                .entry(r.memory_id.clone())
                .or_default()
                .activation = r.activation;
        }

        for (id, score) in fsrs_scores {
            all_ids
                .entry(id.clone())
                .or_default()
                .fsrs_retrievability = *score;
        }

        // Fuse and sort
        let mut results: Vec<_> = all_ids
            .into_iter()
            .map(|(id, inputs)| {
                let score = weights.fuse(&inputs);
                RetrievalResult {
                    id,
                    score,
                    signals: RetrievalSignals {
                        vector: if inputs.vector > 0.0 {
                            Some(inputs.vector)
                        } else {
                            None
                        },
                        bm25: if inputs.bm25_normalized > 0.0 {
                            Some(inputs.bm25_normalized)
                        } else {
                            None
                        },
                        activation: if inputs.activation > 0.0 {
                            Some(inputs.activation)
                        } else {
                            None
                        },
                        fsrs: if inputs.fsrs_retrievability > 0.0 {
                            Some(inputs.fsrs_retrievability)
                        } else {
                            None
                        },
                    },
                }
            })
            .collect();

        results.sort_by(|a, b| OrderedFloat(b.score).cmp(&OrderedFloat(a.score)));
        results
    }

    /// Apply deduplication to results.
    async fn deduplicate_results(
        &self,
        results: Vec<RetrievalResult>,
        config: &super::dedup::DeduplicationConfig,
    ) -> RookResult<Vec<RetrievalResult>> {
        // Get embeddings for deduplication
        let ids: Vec<_> = results.iter().map(|r| r.id.clone()).collect();
        let embeddings = self.vector_searcher.get_embeddings(&ids).await?;

        // Store signals for reconstruction after dedup
        let signal_map: HashMap<_, _> = results
            .iter()
            .map(|r| (r.id.clone(), r.signals.clone()))
            .collect();

        let dedup_results: Vec<_> = results
            .into_iter()
            .map(|r| DeduplicatableResult {
                embedding: embeddings.get(&r.id).cloned(),
                id: r.id,
                score: r.score,
            })
            .collect();

        let deduplicator = Deduplicator::new(config.clone());
        let deduped = deduplicator.deduplicate(dedup_results);

        // Reconstruct full results with signals preserved
        Ok(deduped
            .into_iter()
            .map(|r| RetrievalResult {
                signals: signal_map.get(&r.id).cloned().unwrap_or_default(),
                id: r.id,
                score: r.score,
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Mock implementations for testing
    struct MockVectorSearcher {
        results: Vec<(String, f32)>,
        embeddings: HashMap<String, Vec<f32>>,
    }

    #[async_trait::async_trait]
    impl VectorSearcher for MockVectorSearcher {
        async fn search(&self, _: &[f32], limit: usize) -> RookResult<Vec<(String, f32)>> {
            Ok(self.results.iter().take(limit).cloned().collect())
        }

        async fn get_embedding(&self, id: &str) -> RookResult<Option<Vec<f32>>> {
            Ok(self.embeddings.get(id).cloned())
        }

        async fn get_embeddings(&self, ids: &[String]) -> RookResult<HashMap<String, Vec<f32>>> {
            Ok(ids
                .iter()
                .filter_map(|id| self.embeddings.get(id).map(|e| (id.clone(), e.clone())))
                .collect())
        }
    }

    struct MockGraph;
    impl ActivationGraph for MockGraph {
        fn spread_from_seeds(
            &self,
            seeds: &[(String, f32)],
            _: &SpreadingConfig,
        ) -> Vec<ActivatedMemory> {
            seeds
                .iter()
                .map(|(id, score)| ActivatedMemory {
                    memory_id: id.clone(),
                    activation: *score * 0.8,
                    depth: 1,
                })
                .collect()
        }
    }

    struct MockFsrs;
    impl FsrsStateProvider for MockFsrs {
        fn get_state(&self, _: &str) -> Option<FsrsMemoryState> {
            Some(FsrsMemoryState {
                stability: 10.0,
                difficulty: 0.5,
                last_review: Some(Utc::now()),
            })
        }
    }

    #[tokio::test]
    async fn test_quick_mode() {
        let searcher = MockVectorSearcher {
            results: vec![("a".to_string(), 0.9), ("b".to_string(), 0.7)],
            embeddings: HashMap::new(),
        };

        let engine: RetrievalEngine<MockVectorSearcher, MockGraph, MockFsrs> =
            RetrievalEngine::new(Arc::new(searcher));
        let config = RetrievalConfig::quick(10);

        let results = engine.retrieve("test", &[1.0], &config).await.unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "a");
        assert!(results[0].signals.vector.is_some());
    }

    #[tokio::test]
    async fn test_standard_mode_with_text_search() {
        let searcher = MockVectorSearcher {
            results: vec![("a".to_string(), 0.9), ("b".to_string(), 0.7)],
            embeddings: HashMap::new(),
        };

        let engine: RetrievalEngine<MockVectorSearcher, MockGraph, MockFsrs> =
            RetrievalEngine::new(Arc::new(searcher));
        let config = RetrievalConfig::standard(10);

        // Standard mode should work even without text searcher
        let results = engine.retrieve("test", &[1.0], &config).await.unwrap();

        assert!(!results.is_empty());
    }

    #[tokio::test]
    async fn test_cognitive_mode_with_fsrs() {
        let searcher = MockVectorSearcher {
            results: vec![("a".to_string(), 0.9), ("b".to_string(), 0.7)],
            embeddings: HashMap::new(),
        };

        let engine: RetrievalEngine<MockVectorSearcher, MockGraph, MockFsrs> =
            RetrievalEngine::new(Arc::new(searcher))
                .with_graph(Arc::new(MockGraph))
                .with_fsrs(Arc::new(MockFsrs));
        let config = RetrievalConfig::cognitive(10);

        let results = engine.retrieve("test", &[1.0], &config).await.unwrap();

        // Should have FSRS and activation signals
        assert!(!results.is_empty());
        // With FSRS provider, at least some results should have FSRS score
        let has_fsrs = results.iter().any(|r| r.signals.fsrs.is_some());
        assert!(has_fsrs);
    }

    #[tokio::test]
    async fn test_precise_mode() {
        let searcher = MockVectorSearcher {
            results: vec![("a".to_string(), 0.9)],
            embeddings: HashMap::new(),
        };

        let engine: RetrievalEngine<MockVectorSearcher, MockGraph, MockFsrs> =
            RetrievalEngine::new(Arc::new(searcher))
                .with_graph(Arc::new(MockGraph))
                .with_fsrs(Arc::new(MockFsrs));
        let config = RetrievalConfig::precise(10);

        let results = engine.retrieve("test", &[1.0], &config).await.unwrap();

        assert!(!results.is_empty());
    }

    #[test]
    fn test_retrieval_mode_helpers() {
        assert!(RetrievalMode::Quick.uses_vector());
        assert!(!RetrievalMode::Quick.uses_bm25());
        assert!(!RetrievalMode::Quick.uses_activation());

        assert!(RetrievalMode::Standard.uses_vector());
        assert!(RetrievalMode::Standard.uses_bm25());
        assert!(RetrievalMode::Standard.uses_activation());
        assert!(RetrievalMode::Standard.uses_rrf());

        assert!(RetrievalMode::Cognitive.uses_fsrs());
        assert!(!RetrievalMode::Cognitive.uses_bm25());
    }
}
