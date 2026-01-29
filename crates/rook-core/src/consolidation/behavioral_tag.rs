//! Behavioral tagging for novelty-based memory consolidation boost.
//!
//! Based on PMC4562088 research: when a novel event occurs (high prediction error),
//! memories created within a time window around that event receive a consolidation
//! boost via plasticity-related proteins (PRPs).
//!
//! The time window is asymmetric:
//! - 30 minutes BEFORE the novel event (retroactive boost)
//! - 2 hours AFTER the novel event (proactive boost)

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};

use super::SynapticTag;

/// Configuration for behavioral tagging.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehavioralTagConfig {
    /// Time window before novel event for boost (default: 30 minutes)
    pub window_before: Duration,
    /// Time window after novel event for boost (default: 2 hours)
    pub window_after: Duration,
    /// Novelty threshold: encoding_surprise > threshold = novel (default: 0.7)
    pub novelty_threshold: f32,
    /// Minimum tag strength required for PRP boost (default: 0.05)
    pub min_tag_strength: f32,
}

impl Default for BehavioralTagConfig {
    fn default() -> Self {
        Self {
            window_before: Duration::minutes(30),
            window_after: Duration::hours(2),
            novelty_threshold: 0.7,
            min_tag_strength: 0.05,
        }
    }
}

impl BehavioralTagConfig {
    /// Create a new config with custom values.
    pub fn new(
        window_before_minutes: i64,
        window_after_minutes: i64,
        novelty_threshold: f32,
        min_tag_strength: f32,
    ) -> Self {
        Self {
            window_before: Duration::minutes(window_before_minutes),
            window_after: Duration::minutes(window_after_minutes),
            novelty_threshold: novelty_threshold.clamp(0.0, 1.0),
            min_tag_strength: min_tag_strength.clamp(0.0, 1.0),
        }
    }

    /// Get the full window duration (before + after).
    pub fn total_window(&self) -> Duration {
        self.window_before + self.window_after
    }
}

/// Result of processing a novel event.
#[derive(Debug, Clone)]
pub enum NoveltyResult {
    /// Event was not novel (below threshold)
    NotNovel {
        encoding_surprise: f32,
        threshold: f32,
    },
    /// Event was novel, tags were boosted
    Boosted {
        /// Number of tags that received PRP
        count: usize,
        /// Memory IDs that were boosted
        boosted_ids: Vec<String>,
    },
    /// Event was novel but no valid tags in window
    NoValidTags,
}

/// Behavioral tagger for novelty-based consolidation boost.
pub struct BehavioralTagger {
    config: BehavioralTagConfig,
}

impl BehavioralTagger {
    /// Create a new BehavioralTagger with the given config.
    pub fn new(config: BehavioralTagConfig) -> Self {
        Self { config }
    }

    /// Create a BehavioralTagger with default config.
    pub fn with_defaults() -> Self {
        Self::new(BehavioralTagConfig::default())
    }

    /// Get the configuration.
    pub fn config(&self) -> &BehavioralTagConfig {
        &self.config
    }

    /// Check if an encoding represents a novel event.
    ///
    /// Uses prediction error (encoding_surprise) as novelty indicator.
    /// Phase 3's smart_ingest computes this from embedding dissimilarity.
    pub fn is_novel_event(&self, encoding_surprise: f32) -> bool {
        encoding_surprise > self.config.novelty_threshold
    }

    /// Get the time window for behavioral tagging around a novel event.
    ///
    /// Returns (window_start, window_end) timestamps.
    pub fn get_tagging_window(
        &self,
        novel_event_time: DateTime<Utc>,
    ) -> (DateTime<Utc>, DateTime<Utc>) {
        let window_start = novel_event_time - self.config.window_before;
        let window_end = novel_event_time + self.config.window_after;
        (window_start, window_end)
    }

    /// Filter tags to those within the behavioral tagging window.
    ///
    /// Returns tags that:
    /// 1. Were created within the time window
    /// 2. Are still valid (above min_tag_strength threshold)
    pub fn filter_tags_in_window<'a>(
        &self,
        tags: &'a [SynapticTag],
        novel_event_time: DateTime<Utc>,
    ) -> Vec<&'a SynapticTag> {
        let (window_start, window_end) = self.get_tagging_window(novel_event_time);

        tags.iter()
            .filter(|tag| {
                // Check if tag creation time is within window
                tag.tagged_at >= window_start && tag.tagged_at <= window_end
            })
            .filter(|tag| {
                // Check if tag is still valid (above threshold)
                tag.is_valid_at(novel_event_time, self.config.min_tag_strength)
            })
            .collect()
    }

    /// Apply PRP availability to tags in the behavioral window.
    ///
    /// This is a mutable operation that modifies tags in place.
    /// Returns the IDs of tags that were boosted.
    pub fn apply_prp_boost(
        &self,
        tags: &mut [SynapticTag],
        novel_event_time: DateTime<Utc>,
        exclude_memory_id: Option<&str>,
    ) -> Vec<String> {
        let (window_start, window_end) = self.get_tagging_window(novel_event_time);
        let mut boosted_ids = Vec::new();

        for tag in tags.iter_mut() {
            // Skip the novel memory itself
            if let Some(exclude_id) = exclude_memory_id {
                if tag.memory_id == exclude_id {
                    continue;
                }
            }

            // Check if in time window
            if tag.tagged_at < window_start || tag.tagged_at > window_end {
                continue;
            }

            // Check if tag is still valid
            if !tag.is_valid_at(novel_event_time, self.config.min_tag_strength) {
                continue;
            }

            // Skip if already has PRP
            if tag.prp_available {
                continue;
            }

            // Apply PRP boost
            tag.set_prp_available_at(novel_event_time);
            boosted_ids.push(tag.memory_id.clone());
        }

        boosted_ids
    }

    /// Process a potential novel event and boost nearby memories.
    ///
    /// This is the main entry point for behavioral tagging.
    pub fn process_novel_event(
        &self,
        encoding_surprise: f32,
        novel_event_time: DateTime<Utc>,
        novel_memory_id: &str,
        tags: &mut [SynapticTag],
    ) -> NoveltyResult {
        // Check if this is actually a novel event
        if !self.is_novel_event(encoding_surprise) {
            return NoveltyResult::NotNovel {
                encoding_surprise,
                threshold: self.config.novelty_threshold,
            };
        }

        // Apply PRP boost to tags in window
        let boosted_ids = self.apply_prp_boost(tags, novel_event_time, Some(novel_memory_id));

        if boosted_ids.is_empty() {
            NoveltyResult::NoValidTags
        } else {
            NoveltyResult::Boosted {
                count: boosted_ids.len(),
                boosted_ids,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_tag_at(memory_id: &str, time: DateTime<Utc>) -> SynapticTag {
        SynapticTag::default_for(memory_id.to_string(), time)
    }

    #[test]
    fn test_is_novel_event() {
        let tagger = BehavioralTagger::with_defaults();

        // Below threshold - not novel
        assert!(!tagger.is_novel_event(0.5));
        assert!(!tagger.is_novel_event(0.7)); // At threshold, not novel

        // Above threshold - novel
        assert!(tagger.is_novel_event(0.71));
        assert!(tagger.is_novel_event(0.9));
    }

    #[test]
    fn test_tagging_window_asymmetric() {
        let tagger = BehavioralTagger::with_defaults();
        let now = Utc::now();

        let (start, end) = tagger.get_tagging_window(now);

        // Window should be 30 min before
        let before_diff = now - start;
        assert_eq!(before_diff, Duration::minutes(30));

        // Window should be 2 hours after
        let after_diff = end - now;
        assert_eq!(after_diff, Duration::hours(2));
    }

    #[test]
    fn test_filter_tags_in_window() {
        let tagger = BehavioralTagger::with_defaults();
        let novel_time = Utc::now();

        let tags = vec![
            create_tag_at("too-old", novel_time - Duration::hours(1)), // Before window
            create_tag_at("in-before", novel_time - Duration::minutes(15)), // In window (before)
            create_tag_at("at-novel", novel_time),                     // At novel time
            create_tag_at("in-after", novel_time + Duration::minutes(30)), // In window (after)
            create_tag_at("too-new", novel_time + Duration::hours(3)), // After window
        ];

        let filtered = tagger.filter_tags_in_window(&tags, novel_time);
        let ids: Vec<&str> = filtered.iter().map(|t| t.memory_id.as_str()).collect();

        assert!(
            ids.contains(&"in-before"),
            "Should include memory 15min before"
        );
        assert!(
            ids.contains(&"at-novel"),
            "Should include memory at novel time"
        );
        assert!(
            !ids.contains(&"too-old"),
            "Should exclude memory 1hr before (outside 30min window)"
        );
        assert!(
            !ids.contains(&"too-new"),
            "Should exclude memory 3hr after (outside 2hr window)"
        );
    }

    #[test]
    fn test_apply_prp_boost_excludes_novel_memory() {
        let tagger = BehavioralTagger::with_defaults();
        let novel_time = Utc::now();

        let mut tags = vec![
            create_tag_at("mem-1", novel_time - Duration::minutes(10)),
            create_tag_at("novel-mem", novel_time), // The novel memory itself
            create_tag_at("mem-2", novel_time - Duration::minutes(5)),
        ];

        let boosted = tagger.apply_prp_boost(&mut tags, novel_time, Some("novel-mem"));

        // Should boost mem-1 and mem-2 but not novel-mem
        assert!(boosted.contains(&"mem-1".to_string()));
        assert!(boosted.contains(&"mem-2".to_string()));
        assert!(!boosted.contains(&"novel-mem".to_string()));

        // Verify PRP state
        assert!(tags
            .iter()
            .find(|t| t.memory_id == "mem-1")
            .unwrap()
            .prp_available);
        assert!(!tags
            .iter()
            .find(|t| t.memory_id == "novel-mem")
            .unwrap()
            .prp_available);
    }

    #[test]
    fn test_apply_prp_boost_skips_already_boosted() {
        let tagger = BehavioralTagger::with_defaults();
        let novel_time = Utc::now();

        let mut tag1 = create_tag_at("mem-1", novel_time - Duration::minutes(10));
        tag1.set_prp_available_at(novel_time - Duration::minutes(5)); // Already boosted earlier

        let mut tags = vec![tag1, create_tag_at("mem-2", novel_time - Duration::minutes(5))];

        let boosted = tagger.apply_prp_boost(&mut tags, novel_time, None);

        // Should only boost mem-2 (mem-1 already has PRP)
        assert_eq!(boosted.len(), 1);
        assert!(boosted.contains(&"mem-2".to_string()));
        assert!(!boosted.contains(&"mem-1".to_string()));
    }

    #[test]
    fn test_apply_prp_boost_skips_decayed_tags() {
        let tagger = BehavioralTagger::new(BehavioralTagConfig {
            window_before: Duration::hours(3), // Extended window
            window_after: Duration::hours(2),
            novelty_threshold: 0.7,
            min_tag_strength: 0.1,
        });
        let novel_time = Utc::now();

        // Create a tag that's 2.5 hours old - within extended window but decayed
        // At 2.5 hours (150 min) with tau=60, strength = e^(-150/60) = e^(-2.5) = ~0.08
        let mut tags = vec![
            create_tag_at("decayed", novel_time - Duration::minutes(150)),
            create_tag_at("fresh", novel_time - Duration::minutes(10)),
        ];

        let boosted = tagger.apply_prp_boost(&mut tags, novel_time, None);

        // Should only boost fresh tag (decayed is below 0.1 threshold)
        assert_eq!(boosted.len(), 1);
        assert!(boosted.contains(&"fresh".to_string()));
        assert!(!boosted.contains(&"decayed".to_string()));
    }

    #[test]
    fn test_process_novel_event_not_novel() {
        let tagger = BehavioralTagger::with_defaults();
        let novel_time = Utc::now();

        let mut tags = vec![create_tag_at("mem-1", novel_time - Duration::minutes(10))];

        let result = tagger.process_novel_event(0.5, novel_time, "novel-mem", &mut tags);

        match result {
            NoveltyResult::NotNovel {
                encoding_surprise,
                threshold,
            } => {
                assert!((encoding_surprise - 0.5).abs() < 0.001);
                assert!((threshold - 0.7).abs() < 0.001);
            }
            _ => panic!("Expected NotNovel result"),
        }

        // Tags should not have PRP
        assert!(!tags[0].prp_available);
    }

    #[test]
    fn test_process_novel_event_boosted() {
        let tagger = BehavioralTagger::with_defaults();
        let novel_time = Utc::now();

        let mut tags = vec![
            create_tag_at("mem-1", novel_time - Duration::minutes(10)),
            create_tag_at("mem-2", novel_time - Duration::minutes(20)),
        ];

        let result = tagger.process_novel_event(0.9, novel_time, "novel-mem", &mut tags);

        match result {
            NoveltyResult::Boosted { count, boosted_ids } => {
                assert_eq!(count, 2);
                assert!(boosted_ids.contains(&"mem-1".to_string()));
                assert!(boosted_ids.contains(&"mem-2".to_string()));
            }
            _ => panic!("Expected Boosted result"),
        }

        // Both tags should have PRP
        assert!(tags.iter().all(|t| t.prp_available));
    }
}
