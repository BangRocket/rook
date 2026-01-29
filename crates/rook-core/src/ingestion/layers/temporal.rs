//! Temporal conflict layer (Layer 3) for prediction error gating.
//!
//! This layer detects when new information has a date that conflicts with existing memories:
//! - Overlapping timeframe conflicts
//! - Sequential updates (newer supersedes older)

use chrono::NaiveDate;
use once_cell::sync::Lazy;
use regex::Regex;

use crate::ingestion::layers::embedding::SimilarityCandidate;
use crate::ingestion::types::IngestDecision;

/// Result from temporal conflict check
#[derive(Debug)]
pub struct TemporalResult {
    /// Decision if temporal conflict found
    pub decision: Option<IngestDecision>,
    /// ID of conflicting memory
    pub conflicting_id: Option<String>,
    /// Explanation of temporal conflict
    pub conflict_reason: Option<String>,
}

// Patterns for extracting dates from text
static DATE_PATTERNS: Lazy<Vec<Regex>> = Lazy::new(|| {
    vec![
        // ISO dates: 2024-01-15
        Regex::new(r"\b(\d{4})-(\d{2})-(\d{2})\b").unwrap(),
        // US format: 01/15/2024 or 1/15/2024
        Regex::new(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b").unwrap(),
        // Written: January 15, 2024
        Regex::new(r"(?i)\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})\b").unwrap(),
        // Year only: in 2024, since 2020
        Regex::new(r"(?i)\b(?:in|since|from|during)\s+(\d{4})\b").unwrap(),
    ]
});

/// Layer 3: Temporal conflict detection
///
/// Detects when new information has a date that conflicts with existing memories.
/// For example:
/// - New: "Started at Apple in 2023" vs Existing: "Started at Google in 2023"
///   -> Both claim the same timeframe, conflict
/// - New: "Started at Apple in 2024" vs Existing: "Started at Google in 2020"
///   -> Sequential, likely an update not a conflict
pub struct TemporalConflictLayer;

impl TemporalConflictLayer {
    pub fn new() -> Self {
        Self
    }

    /// Check for temporal conflicts between new content and candidates
    pub fn check(
        &self,
        new_content: &str,
        candidates: &[SimilarityCandidate],
    ) -> TemporalResult {
        // Extract dates from new content
        let new_dates = self.extract_dates(new_content);

        if new_dates.is_empty() {
            // No dates in new content, can't determine temporal conflict
            return TemporalResult {
                decision: None,
                conflicting_id: None,
                conflict_reason: None,
            };
        }

        for candidate in candidates {
            let existing_dates = self.extract_dates(&candidate.content);

            if existing_dates.is_empty() {
                continue;
            }

            // Check for overlapping dates
            if let Some(conflict) = self.check_date_conflict(&new_dates, &existing_dates) {
                // Same timeframe = likely conflict, supersede
                return TemporalResult {
                    decision: Some(IngestDecision::Supersede),
                    conflicting_id: Some(candidate.memory_id.clone()),
                    conflict_reason: Some(conflict),
                };
            }

            // Check for clear temporal ordering
            if let Some(update_reason) = self.check_temporal_update(&new_dates, &existing_dates) {
                // New info is more recent, update/supersede
                return TemporalResult {
                    decision: Some(IngestDecision::Supersede),
                    conflicting_id: Some(candidate.memory_id.clone()),
                    conflict_reason: Some(update_reason),
                };
            }
        }

        // No temporal conflicts found
        TemporalResult {
            decision: None,
            conflicting_id: None,
            conflict_reason: None,
        }
    }

    /// Extract dates from text
    fn extract_dates(&self, text: &str) -> Vec<NaiveDate> {
        let mut dates = Vec::new();

        // ISO format: 2024-01-15
        if let Some(cap) = DATE_PATTERNS[0].captures(text) {
            if let (Ok(y), Ok(m), Ok(d)) = (
                cap[1].parse::<i32>(),
                cap[2].parse::<u32>(),
                cap[3].parse::<u32>(),
            ) {
                if let Some(date) = NaiveDate::from_ymd_opt(y, m, d) {
                    dates.push(date);
                }
            }
        }

        // US format: MM/DD/YYYY
        if let Some(cap) = DATE_PATTERNS[1].captures(text) {
            if let (Ok(m), Ok(d), Ok(y)) = (
                cap[1].parse::<u32>(),
                cap[2].parse::<u32>(),
                cap[3].parse::<i32>(),
            ) {
                if let Some(date) = NaiveDate::from_ymd_opt(y, m, d) {
                    dates.push(date);
                }
            }
        }

        // Year only: use January 1 of that year
        for cap in DATE_PATTERNS[3].captures_iter(text) {
            if let Ok(year) = cap[1].parse::<i32>() {
                if let Some(date) = NaiveDate::from_ymd_opt(year, 1, 1) {
                    dates.push(date);
                }
            }
        }

        dates
    }

    /// Check if dates overlap/conflict
    fn check_date_conflict(&self, new_dates: &[NaiveDate], existing_dates: &[NaiveDate]) -> Option<String> {
        for new_date in new_dates {
            for existing_date in existing_dates {
                // Same year for year-level comparisons
                if new_date.year() == existing_date.year() {
                    // If dates are within same year, might be a conflict
                    let days_apart = (*new_date - *existing_date).num_days().abs();
                    if days_apart < 180 {
                        return Some(format!(
                            "Overlapping timeframe: {} vs {}",
                            new_date, existing_date
                        ));
                    }
                }
            }
        }
        None
    }

    /// Check if new dates indicate an update to existing info
    fn check_temporal_update(&self, new_dates: &[NaiveDate], existing_dates: &[NaiveDate]) -> Option<String> {
        let newest_new = new_dates.iter().max()?;
        let newest_existing = existing_dates.iter().max()?;

        if newest_new > newest_existing {
            // New content has more recent date
            let years_diff = newest_new.year() - newest_existing.year();
            if years_diff >= 1 {
                return Some(format!(
                    "More recent information: {} supersedes {}",
                    newest_new, newest_existing
                ));
            }
        }

        None
    }
}

impl Default for TemporalConflictLayer {
    fn default() -> Self {
        Self::new()
    }
}

// Need to import Datelike trait for .year() method
use chrono::Datelike;

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
    fn test_no_dates_passes_through() {
        let layer = TemporalConflictLayer::new();
        let candidates = vec![make_candidate("1", "User works at Google")];

        let result = layer.check("I work at Apple", &candidates);
        assert!(result.decision.is_none());
    }

    #[test]
    fn test_overlapping_year_conflict() {
        let layer = TemporalConflictLayer::new();
        let candidates = vec![make_candidate("1", "Joined Google in 2023")];

        let result = layer.check("Joined Apple in 2023", &candidates);
        assert!(matches!(result.decision, Some(IngestDecision::Supersede)));
    }

    #[test]
    fn test_sequential_years_update() {
        let layer = TemporalConflictLayer::new();
        let candidates = vec![make_candidate("1", "Joined Google in 2020")];

        let result = layer.check("Now at Apple since 2023", &candidates);
        assert!(matches!(result.decision, Some(IngestDecision::Supersede)));
    }

    #[test]
    fn test_extract_iso_date() {
        let layer = TemporalConflictLayer::new();
        let dates = layer.extract_dates("Started on 2024-01-15");
        assert_eq!(dates.len(), 1);
        assert_eq!(dates[0], NaiveDate::from_ymd_opt(2024, 1, 15).unwrap());
    }

    #[test]
    fn test_extract_us_date() {
        let layer = TemporalConflictLayer::new();
        let dates = layer.extract_dates("Started on 1/15/2024");
        assert_eq!(dates.len(), 1);
        assert_eq!(dates[0], NaiveDate::from_ymd_opt(2024, 1, 15).unwrap());
    }

    #[test]
    fn test_extract_year_only() {
        let layer = TemporalConflictLayer::new();
        let dates = layer.extract_dates("Working there since 2020");
        assert_eq!(dates.len(), 1);
        assert_eq!(dates[0], NaiveDate::from_ymd_opt(2020, 1, 1).unwrap());
    }

    #[test]
    fn test_no_existing_dates() {
        let layer = TemporalConflictLayer::new();
        let candidates = vec![make_candidate("1", "User works at Google")];

        let result = layer.check("Joined Apple in 2023", &candidates);
        assert!(result.decision.is_none());
    }

    #[test]
    fn test_empty_candidates() {
        let layer = TemporalConflictLayer::new();
        let candidates: Vec<SimilarityCandidate> = vec![];

        let result = layer.check("Joined Apple in 2023", &candidates);
        assert!(result.decision.is_none());
    }
}
