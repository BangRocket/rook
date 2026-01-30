//! Bloom filter wrapper for fast keyword intention pre-screening (INT-06).
//!
//! Uses fastbloom for O(1) membership testing. False positives are acceptable
//! (we verify with actual string matching), but false negatives are not.

use fastbloom::BloomFilter;
use std::collections::HashSet;

/// Configuration for the keyword bloom filter.
#[derive(Debug, Clone)]
pub struct BloomConfig {
    /// Target false positive rate (0.0 to 1.0, default 0.001 = 0.1%).
    pub false_positive_rate: f64,
    /// Expected number of keywords (used for sizing).
    pub expected_items: usize,
}

impl Default for BloomConfig {
    fn default() -> Self {
        Self {
            false_positive_rate: 0.001, // 0.1% FPR
            expected_items: 1000,
        }
    }
}

/// Bloom filter wrapper for keyword intention checking.
///
/// Provides fast O(1) pre-screening of keywords. When a keyword might be
/// in the filter, we do the more expensive string comparison.
pub struct KeywordBloomFilter {
    filter: BloomFilter,
    /// Original keywords (for rebuilding filter and multi-word matching).
    keywords: HashSet<String>,
    config: BloomConfig,
}

impl KeywordBloomFilter {
    /// Create a new bloom filter with default config.
    pub fn new() -> Self {
        Self::with_config(BloomConfig::default())
    }

    /// Create a new bloom filter with custom config.
    pub fn with_config(config: BloomConfig) -> Self {
        let filter = BloomFilter::with_false_pos(config.false_positive_rate)
            .expected_items(config.expected_items);
        Self {
            filter,
            keywords: HashSet::new(),
            config,
        }
    }

    /// Add a keyword to the filter (case-insensitive).
    pub fn add(&mut self, keyword: &str) {
        let normalized = keyword.to_lowercase();
        self.filter.insert(&normalized);
        self.keywords.insert(normalized);
    }

    /// Add multiple keywords.
    pub fn add_many(&mut self, keywords: &[String]) {
        for keyword in keywords {
            self.add(keyword);
        }
    }

    /// Check if a keyword might be in the filter (may have false positives).
    pub fn might_contain(&self, keyword: &str) -> bool {
        self.filter.contains(&keyword.to_lowercase())
    }

    /// Check if any word in the message might match a keyword.
    /// Returns potential matches (may include false positives).
    pub fn scan_message(&self, message: &str) -> Vec<String> {
        let normalized = message.to_lowercase();
        let mut potential_matches = Vec::new();

        // Check each word
        for word in normalized.split_whitespace() {
            // Clean punctuation from word edges
            let cleaned = word.trim_matches(|c: char| !c.is_alphanumeric());
            if !cleaned.is_empty() && self.filter.contains(cleaned) {
                potential_matches.push(cleaned.to_string());
            }
        }

        // Also check for multi-word keywords by sliding window
        // (e.g., "machine learning" in "I'm learning about machine learning")
        for keyword in &self.keywords {
            if keyword.contains(' ') && normalized.contains(keyword) {
                potential_matches.push(keyword.clone());
            }
        }

        potential_matches
    }

    /// Verify potential matches against actual string content.
    /// This is the second phase after bloom filter pre-screening.
    pub fn verify_matches(&self, message: &str, potential_matches: &[String]) -> Vec<String> {
        let normalized = message.to_lowercase();
        potential_matches
            .iter()
            .filter(|keyword| normalized.contains(keyword.as_str()))
            .cloned()
            .collect()
    }

    /// Get count of keywords in filter.
    pub fn keyword_count(&self) -> usize {
        self.keywords.len()
    }

    /// Clear all keywords and rebuild filter.
    pub fn clear(&mut self) {
        self.filter = BloomFilter::with_false_pos(self.config.false_positive_rate)
            .expected_items(self.config.expected_items);
        self.keywords.clear();
    }

    /// Rebuild filter from current keywords (useful if config changed).
    pub fn rebuild(&mut self) {
        let keywords: Vec<String> = self.keywords.drain().collect();
        self.filter = BloomFilter::with_false_pos(self.config.false_positive_rate)
            .expected_items(self.config.expected_items.max(keywords.len()));
        for keyword in keywords {
            self.add(&keyword);
        }
    }
}

impl Default for KeywordBloomFilter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bloom_filter_basic() {
        let mut filter = KeywordBloomFilter::new();
        filter.add("rust");
        filter.add("programming");

        assert!(filter.might_contain("rust"));
        assert!(filter.might_contain("programming"));
        assert!(filter.might_contain("RUST")); // case-insensitive
    }

    #[test]
    fn test_scan_message() {
        let mut filter = KeywordBloomFilter::new();
        filter.add("rust");
        filter.add("machine learning");

        let matches = filter.scan_message("I love Rust and machine learning!");
        assert!(matches.contains(&"rust".to_string()));
        assert!(matches.contains(&"machine learning".to_string()));
    }

    #[test]
    fn test_verify_matches() {
        let filter = KeywordBloomFilter::new();
        let potential = vec!["rust".to_string(), "python".to_string()];

        let verified = filter.verify_matches("I love Rust programming", &potential);
        assert!(verified.contains(&"rust".to_string()));
        assert!(!verified.contains(&"python".to_string()));
    }

    #[test]
    fn test_clear_and_rebuild() {
        let mut filter = KeywordBloomFilter::new();
        filter.add("rust");
        filter.add("programming");

        assert_eq!(filter.keyword_count(), 2);

        filter.clear();
        assert_eq!(filter.keyword_count(), 0);
        assert!(!filter.might_contain("rust"));

        filter.add("new_keyword");
        filter.rebuild();
        assert!(filter.might_contain("new_keyword"));
    }

    #[test]
    fn test_add_many() {
        let mut filter = KeywordBloomFilter::new();
        filter.add_many(&["rust".to_string(), "go".to_string(), "python".to_string()]);

        assert_eq!(filter.keyword_count(), 3);
        assert!(filter.might_contain("rust"));
        assert!(filter.might_contain("go"));
        assert!(filter.might_contain("python"));
    }
}
