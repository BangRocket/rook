//! Keyword negation layer (Layer 2) for prediction error gating.
//!
//! This layer detects explicit contradiction patterns using regex matching:
//! - Negation patterns: "no longer", "stopped", "quit", "doesn't", etc.
//! - Temporal override patterns: "now", "currently", "as of", etc.
//! - State change patterns: "used to... now is"

use once_cell::sync::Lazy;
use regex::Regex;

use crate::ingestion::layers::embedding::SimilarityCandidate;
use crate::ingestion::types::IngestDecision;

/// Result from keyword pattern check
#[derive(Debug)]
pub struct KeywordResult {
    /// Decision if contradiction pattern found
    pub decision: Option<IngestDecision>,
    /// ID of contradicted memory
    pub contradicted_id: Option<String>,
    /// Evidence of contradiction
    pub contradiction_evidence: Option<String>,
}

// Compiled regex patterns for negation detection
static NEGATION_PATTERNS: Lazy<Vec<Regex>> = Lazy::new(|| {
    vec![
        Regex::new(r"(?i)\b(no longer|not|never|doesn't|don't|won't|isn't|aren't|wasn't|weren't|can't|cannot|couldn't)\b").unwrap(),
        Regex::new(r"(?i)\b(stopped|quit|ended|left|divorced|moved from|resigned|retired)\b").unwrap(),
        Regex::new(r"(?i)\b(formerly|previously|used to|was|were)\b.*\b(now|currently|is|are|has become|became)\b").unwrap(),
    ]
});

// Patterns that indicate temporal override (new info supersedes old)
static TEMPORAL_OVERRIDE_PATTERNS: Lazy<Vec<Regex>> = Lazy::new(|| {
    vec![
        Regex::new(r"(?i)\b(now|currently|as of|since|starting|effective|today)\b").unwrap(),
        Regex::new(r"(?i)\b(changed|updated|corrected|actually|in fact)\b").unwrap(),
        Regex::new(r"(?i)\b(no longer|not anymore|stopped)\b").unwrap(),
    ]
});

/// Layer 2: Keyword and negation pattern detection
///
/// Fast layer (~1ms). Detects explicit contradiction patterns like:
/// - Negation: "no longer", "stopped", "quit", "doesn't"
/// - Temporal override: "now", "currently", "as of"
/// - State change: "used to... now is"
pub struct KeywordNegationLayer;

impl KeywordNegationLayer {
    pub fn new() -> Self {
        Self
    }

    /// Check for contradiction patterns between new content and candidates
    pub fn check(
        &self,
        new_content: &str,
        candidates: &[SimilarityCandidate],
    ) -> KeywordResult {
        let new_lower = new_content.to_lowercase();

        // Check for negation or temporal override patterns in new content
        let has_negation = NEGATION_PATTERNS.iter().any(|p| p.is_match(&new_lower));
        let has_temporal_override = TEMPORAL_OVERRIDE_PATTERNS.iter().any(|p| p.is_match(&new_lower));

        // If no contradiction indicators, pass to next layer
        if !has_negation && !has_temporal_override {
            return KeywordResult {
                decision: None,
                contradicted_id: None,
                contradiction_evidence: None,
            };
        }

        // Look for direct contradictions in candidates
        for candidate in candidates {
            let existing_lower = candidate.content.to_lowercase();

            // Check for key term contradictions
            if let Some(evidence) = self.find_contradiction(&new_lower, &existing_lower) {
                return KeywordResult {
                    decision: Some(IngestDecision::Supersede),
                    contradicted_id: Some(candidate.memory_id.clone()),
                    contradiction_evidence: Some(evidence),
                };
            }
        }

        // Has negation patterns but no clear contradiction found
        // Pass to next layer for deeper analysis
        KeywordResult {
            decision: None,
            contradicted_id: None,
            contradiction_evidence: None,
        }
    }

    /// Find specific contradiction between two texts
    fn find_contradiction(&self, new_text: &str, existing_text: &str) -> Option<String> {
        // Extract key entities/terms and compare
        // This is a heuristic approach - look for common patterns

        // Pattern: "lives in X" vs "lives in Y"
        let location_pattern = Regex::new(r"(?i)\b(lives? in|moved to|relocated to|resides? in)\s+(\w+(?:\s+\w+)?)\b").ok()?;

        if let (Some(new_match), Some(existing_match)) = (
            location_pattern.captures(new_text),
            location_pattern.captures(existing_text),
        ) {
            let new_loc = new_match.get(2)?.as_str().to_lowercase();
            let existing_loc = existing_match.get(2)?.as_str().to_lowercase();
            if new_loc != existing_loc {
                return Some(format!(
                    "Location change: '{}' -> '{}'",
                    existing_loc, new_loc
                ));
            }
        }

        // Pattern: "works at X" vs "works at Y"
        let work_pattern = Regex::new(r"(?i)\b(works? (?:at|for)|employed (?:at|by)|joined)\s+(\w+(?:\s+\w+)?)\b").ok()?;

        if let (Some(new_match), Some(existing_match)) = (
            work_pattern.captures(new_text),
            work_pattern.captures(existing_text),
        ) {
            let new_work = new_match.get(2)?.as_str().to_lowercase();
            let existing_work = existing_match.get(2)?.as_str().to_lowercase();
            if new_work != existing_work {
                return Some(format!(
                    "Employment change: '{}' -> '{}'",
                    existing_work, new_work
                ));
            }
        }

        // Pattern: "is married" vs "is divorced" / "is single"
        // Allow optional "now" between verb and status
        let marital_pattern = Regex::new(r"(?i)\b(is|got|became)\s+(?:now\s+)?(married|divorced|single|engaged|widowed)\b").ok()?;

        if let (Some(new_match), Some(existing_match)) = (
            marital_pattern.captures(new_text),
            marital_pattern.captures(existing_text),
        ) {
            let new_status = new_match.get(2)?.as_str().to_lowercase();
            let existing_status = existing_match.get(2)?.as_str().to_lowercase();
            if new_status != existing_status {
                return Some(format!(
                    "Marital status change: '{}' -> '{}'",
                    existing_status, new_status
                ));
            }
        }

        None
    }
}

impl Default for KeywordNegationLayer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_candidate(id: &str, content: &str) -> SimilarityCandidate {
        SimilarityCandidate {
            memory_id: id.to_string(),
            content: content.to_string(),
            embedding: vec![],
            similarity: 0.8,
        }
    }

    #[test]
    fn test_no_negation_passes_through() {
        let layer = KeywordNegationLayer::new();
        let candidates = vec![make_candidate("1", "User lives in Boston")];

        let result = layer.check("I enjoy hiking on weekends", &candidates);
        assert!(result.decision.is_none());
    }

    #[test]
    fn test_location_change_detected() {
        let layer = KeywordNegationLayer::new();
        let candidates = vec![make_candidate("1", "User lives in Boston")];

        let result = layer.check("I now live in New York", &candidates);
        assert!(matches!(result.decision, Some(IngestDecision::Supersede)));
        assert_eq!(result.contradicted_id, Some("1".to_string()));
    }

    #[test]
    fn test_employment_change_detected() {
        let layer = KeywordNegationLayer::new();
        let candidates = vec![make_candidate("1", "User works at Google")];

        let result = layer.check("I currently work at Apple", &candidates);
        assert!(matches!(result.decision, Some(IngestDecision::Supersede)));
    }

    #[test]
    fn test_negation_without_match() {
        let layer = KeywordNegationLayer::new();
        let candidates = vec![make_candidate("1", "User likes pizza")];

        // Has negation pattern but no clear contradiction
        let result = layer.check("I no longer eat meat", &candidates);
        assert!(result.decision.is_none());
    }

    #[test]
    fn test_marital_status_change() {
        let layer = KeywordNegationLayer::new();
        let candidates = vec![make_candidate("1", "User is married")];

        let result = layer.check("User is now divorced", &candidates);
        assert!(matches!(result.decision, Some(IngestDecision::Supersede)));
        assert!(result.contradiction_evidence.is_some());
    }

    #[test]
    fn test_empty_candidates() {
        let layer = KeywordNegationLayer::new();
        let candidates: Vec<SimilarityCandidate> = vec![];

        let result = layer.check("I no longer work there", &candidates);
        assert!(result.decision.is_none());
    }
}
