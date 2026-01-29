//! Consolidation phases representing memory stability stages.
//!
//! Memory consolidation progresses through distinct phases:
//! - Immediate: Highly labile, requires ongoing synaptic support
//! - Early: Beginning cellular consolidation
//! - Late: Systems consolidation begins (hippocampus -> neocortex)
//! - Consolidated: Stable long-term storage

use chrono::Duration;
use serde::{Deserialize, Serialize};
use strum::{Display, EnumString};

/// Phase boundaries in hours.
pub const IMMEDIATE_HOURS: i64 = 6;
pub const EARLY_HOURS: i64 = 24;
pub const LATE_HOURS: i64 = 72;

/// Consolidation phase representing memory stability stage.
///
/// Based on neuroscience research on memory consolidation timecourses:
///
/// - **Immediate (0-6h)**: Memory is highly labile and depends on synaptic
///   tag strength. Can be disrupted by interference or lack of PRPs.
///
/// - **Early (6-24h)**: Cellular consolidation underway. Memory requires
///   both valid tag and PRP availability to advance.
///
/// - **Late (24-72h)**: Systems consolidation begins. Memory traces are
///   being transferred from hippocampus to neocortical storage.
///
/// - **Consolidated (72h+)**: Memory has achieved stable long-term storage.
///   Less vulnerable to interference but still subject to normal forgetting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Display, EnumString)]
#[serde(rename_all = "snake_case")]
#[strum(serialize_all = "snake_case")]
pub enum ConsolidationPhase {
    /// 0-6 hours: Highly labile, depends on tag strength
    Immediate,
    /// 6-24 hours: Cellular consolidation, requires tag + PRP
    Early,
    /// 24-72 hours: Systems consolidation begins
    Late,
    /// 72+ hours: Stable long-term storage
    Consolidated,
}

impl ConsolidationPhase {
    /// Determine the consolidation phase based on memory age.
    ///
    /// # Arguments
    ///
    /// * `age` - Time since memory was created
    ///
    /// # Example
    ///
    /// ```
    /// use rook_core::consolidation::ConsolidationPhase;
    /// use chrono::Duration;
    ///
    /// let phase = ConsolidationPhase::from_age(Duration::hours(3));
    /// assert_eq!(phase, ConsolidationPhase::Immediate);
    ///
    /// let phase = ConsolidationPhase::from_age(Duration::hours(100));
    /// assert_eq!(phase, ConsolidationPhase::Consolidated);
    /// ```
    pub fn from_age(age: Duration) -> Self {
        let hours = age.num_hours();

        if hours < IMMEDIATE_HOURS {
            ConsolidationPhase::Immediate
        } else if hours < EARLY_HOURS {
            ConsolidationPhase::Early
        } else if hours < LATE_HOURS {
            ConsolidationPhase::Late
        } else {
            ConsolidationPhase::Consolidated
        }
    }

    /// Get the next phase in the consolidation sequence.
    ///
    /// Returns None if already at Consolidated phase.
    pub fn next(&self) -> Option<ConsolidationPhase> {
        match self {
            ConsolidationPhase::Immediate => Some(ConsolidationPhase::Early),
            ConsolidationPhase::Early => Some(ConsolidationPhase::Late),
            ConsolidationPhase::Late => Some(ConsolidationPhase::Consolidated),
            ConsolidationPhase::Consolidated => None,
        }
    }

    /// Check if memory can advance to the next phase.
    ///
    /// # Arguments
    ///
    /// * `age` - Current age of the memory
    /// * `has_valid_tag` - Whether the synaptic tag is still valid
    /// * `has_prp` - Whether PRPs are available
    ///
    /// # Transition Rules
    ///
    /// - Immediate -> Early: Requires time (>6h) + valid tag + PRP
    /// - Early -> Late: Requires time (>24h) only
    /// - Late -> Consolidated: Requires time (>72h) only
    pub fn can_advance(&self, age: Duration, has_valid_tag: bool, has_prp: bool) -> bool {
        let hours = age.num_hours();

        match self {
            ConsolidationPhase::Immediate => {
                // Critical transition: needs tag + PRP
                hours >= IMMEDIATE_HOURS && has_valid_tag && has_prp
            }
            ConsolidationPhase::Early => {
                // Past the critical window, time-based only
                hours >= EARLY_HOURS
            }
            ConsolidationPhase::Late => {
                // Time-based transition
                hours >= LATE_HOURS
            }
            ConsolidationPhase::Consolidated => {
                // Already at final phase
                false
            }
        }
    }

    /// Check if memory in this phase is vulnerable to loss.
    ///
    /// Memories in Immediate and Early phases are more susceptible to
    /// interference and can be lost if consolidation fails.
    pub fn is_vulnerable(&self) -> bool {
        matches!(
            self,
            ConsolidationPhase::Immediate | ConsolidationPhase::Early
        )
    }

    /// Get a human-readable description of this phase.
    pub fn description(&self) -> &'static str {
        match self {
            ConsolidationPhase::Immediate => {
                "Immediate phase (0-6h): Memory is highly labile. \
                 Requires synaptic tag and PRPs for stabilization."
            }
            ConsolidationPhase::Early => {
                "Early consolidation (6-24h): Cellular consolidation underway. \
                 Memory becoming more stable but still vulnerable."
            }
            ConsolidationPhase::Late => {
                "Late consolidation (24-72h): Systems consolidation in progress. \
                 Memory traces being integrated into long-term storage."
            }
            ConsolidationPhase::Consolidated => {
                "Consolidated (72h+): Memory has achieved stable long-term storage. \
                 Subject to normal forgetting dynamics."
            }
        }
    }

    /// Get the minimum age (in hours) for this phase.
    pub fn min_age_hours(&self) -> i64 {
        match self {
            ConsolidationPhase::Immediate => 0,
            ConsolidationPhase::Early => IMMEDIATE_HOURS,
            ConsolidationPhase::Late => EARLY_HOURS,
            ConsolidationPhase::Consolidated => LATE_HOURS,
        }
    }

    /// Get the maximum age (in hours) for this phase.
    ///
    /// Returns None for Consolidated phase (no upper bound).
    pub fn max_age_hours(&self) -> Option<i64> {
        match self {
            ConsolidationPhase::Immediate => Some(IMMEDIATE_HOURS),
            ConsolidationPhase::Early => Some(EARLY_HOURS),
            ConsolidationPhase::Late => Some(LATE_HOURS),
            ConsolidationPhase::Consolidated => None,
        }
    }
}

impl Default for ConsolidationPhase {
    fn default() -> Self {
        ConsolidationPhase::Immediate
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    #[test]
    fn test_from_age_immediate() {
        assert_eq!(
            ConsolidationPhase::from_age(Duration::hours(0)),
            ConsolidationPhase::Immediate
        );
        assert_eq!(
            ConsolidationPhase::from_age(Duration::hours(5)),
            ConsolidationPhase::Immediate
        );
    }

    #[test]
    fn test_from_age_early() {
        assert_eq!(
            ConsolidationPhase::from_age(Duration::hours(6)),
            ConsolidationPhase::Early
        );
        assert_eq!(
            ConsolidationPhase::from_age(Duration::hours(12)),
            ConsolidationPhase::Early
        );
        assert_eq!(
            ConsolidationPhase::from_age(Duration::hours(23)),
            ConsolidationPhase::Early
        );
    }

    #[test]
    fn test_from_age_late() {
        assert_eq!(
            ConsolidationPhase::from_age(Duration::hours(24)),
            ConsolidationPhase::Late
        );
        assert_eq!(
            ConsolidationPhase::from_age(Duration::hours(48)),
            ConsolidationPhase::Late
        );
        assert_eq!(
            ConsolidationPhase::from_age(Duration::hours(71)),
            ConsolidationPhase::Late
        );
    }

    #[test]
    fn test_from_age_consolidated() {
        assert_eq!(
            ConsolidationPhase::from_age(Duration::hours(72)),
            ConsolidationPhase::Consolidated
        );
        assert_eq!(
            ConsolidationPhase::from_age(Duration::hours(1000)),
            ConsolidationPhase::Consolidated
        );
    }

    #[test]
    fn test_next_phase() {
        assert_eq!(
            ConsolidationPhase::Immediate.next(),
            Some(ConsolidationPhase::Early)
        );
        assert_eq!(
            ConsolidationPhase::Early.next(),
            Some(ConsolidationPhase::Late)
        );
        assert_eq!(
            ConsolidationPhase::Late.next(),
            Some(ConsolidationPhase::Consolidated)
        );
        assert_eq!(ConsolidationPhase::Consolidated.next(), None);
    }

    #[test]
    fn test_can_advance_immediate() {
        let phase = ConsolidationPhase::Immediate;

        // Too early
        assert!(!phase.can_advance(Duration::hours(3), true, true));

        // Old enough but missing tag
        assert!(!phase.can_advance(Duration::hours(7), false, true));

        // Old enough but missing PRP
        assert!(!phase.can_advance(Duration::hours(7), true, false));

        // All conditions met
        assert!(phase.can_advance(Duration::hours(7), true, true));
    }

    #[test]
    fn test_can_advance_early() {
        let phase = ConsolidationPhase::Early;

        // Too early
        assert!(!phase.can_advance(Duration::hours(12), true, true));

        // Old enough (tag/PRP don't matter for Early->Late)
        assert!(phase.can_advance(Duration::hours(25), false, false));
    }

    #[test]
    fn test_can_advance_late() {
        let phase = ConsolidationPhase::Late;

        // Too early
        assert!(!phase.can_advance(Duration::hours(48), true, true));

        // Old enough
        assert!(phase.can_advance(Duration::hours(73), false, false));
    }

    #[test]
    fn test_can_advance_consolidated() {
        let phase = ConsolidationPhase::Consolidated;

        // Can never advance past consolidated
        assert!(!phase.can_advance(Duration::hours(1000), true, true));
    }

    #[test]
    fn test_is_vulnerable() {
        assert!(ConsolidationPhase::Immediate.is_vulnerable());
        assert!(ConsolidationPhase::Early.is_vulnerable());
        assert!(!ConsolidationPhase::Late.is_vulnerable());
        assert!(!ConsolidationPhase::Consolidated.is_vulnerable());
    }

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", ConsolidationPhase::Immediate), "immediate");
        assert_eq!(format!("{}", ConsolidationPhase::Early), "early");
        assert_eq!(format!("{}", ConsolidationPhase::Late), "late");
        assert_eq!(format!("{}", ConsolidationPhase::Consolidated), "consolidated");
    }

    #[test]
    fn test_from_str() {
        assert_eq!(
            ConsolidationPhase::from_str("immediate").unwrap(),
            ConsolidationPhase::Immediate
        );
        assert_eq!(
            ConsolidationPhase::from_str("early").unwrap(),
            ConsolidationPhase::Early
        );
        assert_eq!(
            ConsolidationPhase::from_str("late").unwrap(),
            ConsolidationPhase::Late
        );
        assert_eq!(
            ConsolidationPhase::from_str("consolidated").unwrap(),
            ConsolidationPhase::Consolidated
        );
    }

    #[test]
    fn test_serialization() {
        let phase = ConsolidationPhase::Early;
        let json = serde_json::to_string(&phase).unwrap();
        assert_eq!(json, "\"early\"");

        let restored: ConsolidationPhase = serde_json::from_str(&json).unwrap();
        assert_eq!(restored, phase);
    }

    #[test]
    fn test_min_max_age() {
        assert_eq!(ConsolidationPhase::Immediate.min_age_hours(), 0);
        assert_eq!(ConsolidationPhase::Immediate.max_age_hours(), Some(6));

        assert_eq!(ConsolidationPhase::Early.min_age_hours(), 6);
        assert_eq!(ConsolidationPhase::Early.max_age_hours(), Some(24));

        assert_eq!(ConsolidationPhase::Late.min_age_hours(), 24);
        assert_eq!(ConsolidationPhase::Late.max_age_hours(), Some(72));

        assert_eq!(ConsolidationPhase::Consolidated.min_age_hours(), 72);
        assert_eq!(ConsolidationPhase::Consolidated.max_age_hours(), None);
    }

    #[test]
    fn test_default() {
        assert_eq!(ConsolidationPhase::default(), ConsolidationPhase::Immediate);
    }
}
