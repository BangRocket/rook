//! Tiered intention checking (INT-06, INT-07).
//!
//! Two-phase evaluation:
//! 1. Fast bloom filter scan for keyword mentions (every message)
//! 2. Expensive semantic similarity check (at configurable interval)

use crate::error::RookResult;
use crate::intentions::{
    bloom::KeywordBloomFilter,
    store::IntentionStore,
    triggers::{ActionResult, FiredIntention, TriggerReason},
    types::{Intention, TriggerCondition},
};
use crate::traits::Embedder;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use tokio::sync::RwLock;

/// Configuration for the intention checker.
#[derive(Debug, Clone)]
pub struct CheckerConfig {
    /// How often to run semantic checks (every N messages).
    /// Default: 10 (check semantics every 10th message).
    pub semantic_check_interval: u32,
    /// Similarity threshold for topic matching (0.0 to 1.0).
    /// Default: 0.75.
    pub topic_similarity_threshold: f32,
    /// Maximum number of intentions to check per message.
    /// Default: 100.
    pub max_intentions_per_check: usize,
}

impl Default for CheckerConfig {
    fn default() -> Self {
        Self {
            semantic_check_interval: 10,
            topic_similarity_threshold: 0.75,
            max_intentions_per_check: 100,
        }
    }
}

/// Tiered intention checker.
///
/// Evaluates intentions efficiently using a two-tier approach:
/// - Tier 1: Bloom filter for keyword mentions (fast, every message)
/// - Tier 2: Embedding similarity for topics (expensive, interval-based)
pub struct IntentionChecker<S: IntentionStore, E: Embedder> {
    /// Intention store for loading intentions.
    store: Arc<S>,
    /// Embedder for semantic similarity.
    embedder: Arc<E>,
    /// Bloom filter for keyword pre-screening.
    keyword_bloom: RwLock<KeywordBloomFilter>,
    /// Configuration.
    config: CheckerConfig,
    /// Message counter for semantic check interval.
    message_counter: AtomicU32,
    /// Cached keyword intentions (refreshed periodically).
    keyword_intentions: RwLock<Vec<Intention>>,
    /// Cached topic intentions with embeddings.
    topic_intentions: RwLock<Vec<(Intention, Vec<f32>)>>,
}

impl<S: IntentionStore, E: Embedder> IntentionChecker<S, E> {
    /// Create a new intention checker.
    pub fn new(store: Arc<S>, embedder: Arc<E>, config: CheckerConfig) -> Self {
        Self {
            store,
            embedder,
            keyword_bloom: RwLock::new(KeywordBloomFilter::new()),
            config,
            message_counter: AtomicU32::new(0),
            keyword_intentions: RwLock::new(Vec::new()),
            topic_intentions: RwLock::new(Vec::new()),
        }
    }

    /// Load/refresh intentions from store.
    pub async fn refresh_intentions(&self) -> RookResult<()> {
        // Load keyword intentions and rebuild bloom filter
        let keyword_intentions = self.store.get_by_trigger_type("keyword_mention")?;
        let mut bloom = self.keyword_bloom.write().await;
        bloom.clear();

        for intention in &keyword_intentions {
            if let TriggerCondition::KeywordMention { keywords, .. } = &intention.trigger {
                bloom.add_many(keywords);
            }
        }

        *self.keyword_intentions.write().await = keyword_intentions;

        // Load topic intentions and compute embeddings
        let topic_intentions = self.store.get_by_trigger_type("topic_discussed")?;
        let mut with_embeddings = Vec::new();

        for intention in topic_intentions {
            if let TriggerCondition::TopicDiscussed {
                topic,
                topic_embedding,
                ..
            } = &intention.trigger
            {
                let embedding = if let Some(emb) = topic_embedding {
                    emb.clone()
                } else {
                    // Compute embedding if not cached
                    match self.embedder.embed(topic, None).await {
                        Ok(emb) => emb,
                        Err(_) => continue, // Skip if embedding fails
                    }
                };
                with_embeddings.push((intention, embedding));
            }
        }

        *self.topic_intentions.write().await = with_embeddings;

        Ok(())
    }

    /// Check message for triggered intentions.
    ///
    /// Returns list of intentions that fired, with reasons.
    pub async fn check(
        &self,
        message: &str,
        user_id: Option<&str>,
    ) -> RookResult<Vec<FiredIntention>> {
        let mut fired = Vec::new();

        // Tier 1: Keyword checking (every message) - INT-06
        fired.extend(self.check_keywords(message, user_id).await?);

        // Tier 2: Semantic checking (at interval) - INT-07
        let count = self.message_counter.fetch_add(1, Ordering::Relaxed);
        if count % self.config.semantic_check_interval == 0 {
            fired.extend(self.check_topics(message, user_id).await?);
        }

        Ok(fired)
    }

    /// Check keyword intentions using bloom filter pre-screening.
    async fn check_keywords(
        &self,
        message: &str,
        user_id: Option<&str>,
    ) -> RookResult<Vec<FiredIntention>> {
        let mut fired = Vec::new();

        // Phase 1: Bloom filter scan
        let bloom = self.keyword_bloom.read().await;
        let potential_matches = bloom.scan_message(message);

        if potential_matches.is_empty() {
            return Ok(fired);
        }

        // Phase 2: Verify against actual intentions
        let intentions = self.keyword_intentions.read().await;
        for intention in intentions.iter() {
            // Skip if wrong user scope
            if let (Some(user), Some(intention_user)) = (user_id, &intention.user_id) {
                if user != intention_user {
                    continue;
                }
            }

            if !intention.can_fire() {
                continue;
            }

            if let TriggerCondition::KeywordMention {
                keywords,
                exact_match,
            } = &intention.trigger
            {
                // Check if any keyword matches
                for keyword in keywords {
                    let normalized_keyword = keyword.to_lowercase();
                    let matches = if *exact_match {
                        // Exact word match
                        message
                            .to_lowercase()
                            .split_whitespace()
                            .any(|word| {
                                word.trim_matches(|c: char| !c.is_alphanumeric())
                                    == normalized_keyword
                            })
                    } else {
                        // Substring match
                        message.to_lowercase().contains(&normalized_keyword)
                    };

                    if matches {
                        fired.push(FiredIntention::new(
                            intention.id,
                            TriggerReason::Keyword {
                                matched_keyword: keyword.clone(),
                                context: Self::extract_context(message, keyword),
                            },
                            ActionResult::Success { details: None },
                        ));
                        break; // Only fire once per intention
                    }
                }
            }
        }

        Ok(fired)
    }

    /// Check topic intentions using semantic similarity.
    async fn check_topics(
        &self,
        message: &str,
        user_id: Option<&str>,
    ) -> RookResult<Vec<FiredIntention>> {
        let mut fired = Vec::new();

        // Get message embedding
        let message_embedding = self.embedder.embed(message, None).await?;

        let intentions = self.topic_intentions.read().await;
        for (intention, topic_embedding) in intentions.iter() {
            // Skip if wrong user scope
            if let (Some(user), Some(intention_user)) = (user_id, &intention.user_id) {
                if user != intention_user {
                    continue;
                }
            }

            if !intention.can_fire() {
                continue;
            }

            if let TriggerCondition::TopicDiscussed {
                topic, threshold, ..
            } = &intention.trigger
            {
                let similarity = cosine_similarity(&message_embedding, topic_embedding);

                if similarity >= *threshold {
                    fired.push(FiredIntention::new(
                        intention.id,
                        TriggerReason::Topic {
                            similarity,
                            topic: topic.clone(),
                        },
                        ActionResult::Success { details: None },
                    ));
                }
            }
        }

        Ok(fired)
    }

    /// Extract context around matched keyword.
    fn extract_context(message: &str, keyword: &str) -> String {
        let lower_message = message.to_lowercase();
        let lower_keyword = keyword.to_lowercase();

        if let Some(pos) = lower_message.find(&lower_keyword) {
            let start = pos.saturating_sub(30);
            let end = (pos + keyword.len() + 30).min(message.len());

            let mut context = String::new();
            if start > 0 {
                context.push_str("...");
            }
            context.push_str(&message[start..end]);
            if end < message.len() {
                context.push_str("...");
            }
            context
        } else {
            message.chars().take(60).collect()
        }
    }

    /// Get the current message count.
    pub fn message_count(&self) -> u32 {
        self.message_counter.load(Ordering::Relaxed)
    }

    /// Reset the message counter.
    pub fn reset_counter(&self) {
        self.message_counter.store(0, Ordering::Relaxed);
    }
}

/// Compute cosine similarity between two vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.001);

        let c = vec![0.0, 1.0, 0.0];
        assert!((cosine_similarity(&a, &c) - 0.0).abs() < 0.001);

        let d = vec![0.707, 0.707, 0.0];
        assert!((cosine_similarity(&a, &d) - 0.707).abs() < 0.01);
    }

    #[test]
    fn test_cosine_similarity_edge_cases() {
        // Empty vectors
        let empty: Vec<f32> = vec![];
        assert_eq!(cosine_similarity(&empty, &empty), 0.0);

        // Different lengths
        let a = vec![1.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);

        // Zero vectors
        let zero = vec![0.0, 0.0, 0.0];
        assert_eq!(cosine_similarity(&zero, &zero), 0.0);
    }

    #[test]
    fn test_extract_context() {
        let message = "I've been learning about Rust programming and really enjoying it.";
        let context =
            IntentionChecker::<crate::intentions::SqliteIntentionStore, MockEmbedder>::extract_context(
                message, "Rust",
            );
        assert!(context.contains("Rust"));

        // Test short message
        let short = "Rust is great";
        let context_short =
            IntentionChecker::<crate::intentions::SqliteIntentionStore, MockEmbedder>::extract_context(
                short, "Rust",
            );
        assert_eq!(context_short, "Rust is great");
    }

    #[test]
    fn test_checker_config_default() {
        let config = CheckerConfig::default();
        assert_eq!(config.semantic_check_interval, 10);
        assert!((config.topic_similarity_threshold - 0.75).abs() < f32::EPSILON);
        assert_eq!(config.max_intentions_per_check, 100);
    }

    // Mock embedder for tests
    struct MockEmbedder;

    #[async_trait::async_trait]
    impl Embedder for MockEmbedder {
        async fn embed(
            &self,
            _text: &str,
            _action: Option<crate::traits::EmbeddingAction>,
        ) -> RookResult<Vec<f32>> {
            Ok(vec![1.0, 0.0, 0.0])
        }

        fn dimension(&self) -> usize {
            3
        }

        fn model_name(&self) -> &str {
            "mock"
        }
    }
}
