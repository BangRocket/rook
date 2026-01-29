//! Synaptic tagging mechanism with exponential decay.
//!
//! Implements the transient marking of synapses during learning events.
//! Tags decay exponentially over time unless stabilized by PRPs.

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};

/// Default time constant (tau) for tag decay in minutes.
/// Based on biological findings that synaptic tags last ~60 minutes.
pub const DEFAULT_TAU_MINUTES: f64 = 60.0;

/// Default threshold below which tags are considered invalid.
pub const DEFAULT_VALIDITY_THRESHOLD: f64 = 0.1;

/// A synaptic tag marking a memory for potential consolidation.
///
/// Tags are set during learning and decay exponentially. If plasticity-related
/// proteins (PRPs) become available before the tag decays below threshold,
/// the memory can be consolidated into long-term storage.
///
/// # Decay Formula
///
/// S(t) = S_0 * e^(-t/tau)
///
/// Where:
/// - S(t) = current strength at time t
/// - S_0 = initial strength
/// - tau = time constant (~60 minutes)
/// - t = time elapsed since tagging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynapticTag {
    /// The memory this tag is associated with.
    pub memory_id: String,

    /// Initial tag strength (0.0-1.0).
    pub initial_strength: f64,

    /// Time constant for decay in minutes (default: 60).
    pub tau: f64,

    /// When the tag was created.
    pub tagged_at: DateTime<Utc>,

    /// Whether PRPs are available for capture.
    pub prp_available: bool,

    /// When PRPs became available (if any).
    pub prp_available_at: Option<DateTime<Utc>>,
}

impl SynapticTag {
    /// Create a new synaptic tag for a memory.
    ///
    /// # Arguments
    ///
    /// * `memory_id` - The memory this tag is associated with
    /// * `initial_strength` - Initial tag strength (clamped to 0.0-1.0)
    ///
    /// # Example
    ///
    /// ```
    /// use rook_core::consolidation::SynapticTag;
    ///
    /// let tag = SynapticTag::new("mem123".to_string(), 0.8);
    /// assert!(tag.current_strength() > 0.7); // Just created, still strong
    /// ```
    pub fn new(memory_id: String, initial_strength: f64) -> Self {
        Self {
            memory_id,
            initial_strength: initial_strength.clamp(0.0, 1.0),
            tau: DEFAULT_TAU_MINUTES,
            tagged_at: Utc::now(),
            prp_available: false,
            prp_available_at: None,
        }
    }

    /// Create a tag with a custom time constant.
    pub fn with_tau(memory_id: String, initial_strength: f64, tau: f64) -> Self {
        Self {
            tau: tau.max(1.0), // Minimum 1 minute tau
            ..Self::new(memory_id, initial_strength)
        }
    }

    /// Create a tag at a specific time (useful for testing and persistence).
    pub fn with_timestamp(
        memory_id: String,
        initial_strength: f64,
        tagged_at: DateTime<Utc>,
    ) -> Self {
        Self {
            tagged_at,
            ..Self::new(memory_id, initial_strength)
        }
    }

    /// Calculate the current tag strength using exponential decay.
    ///
    /// S(t) = S_0 * e^(-t/tau)
    ///
    /// Returns a value between 0.0 and initial_strength.
    pub fn current_strength(&self) -> f64 {
        self.strength_at(Utc::now())
    }

    /// Calculate tag strength at a specific time.
    ///
    /// Returns 0.0 if the time is before the tag was created.
    pub fn strength_at(&self, at: DateTime<Utc>) -> f64 {
        let elapsed = at.signed_duration_since(self.tagged_at);
        if elapsed < Duration::zero() {
            return 0.0;
        }

        let minutes = elapsed.num_milliseconds() as f64 / 60_000.0;
        self.initial_strength * (-minutes / self.tau).exp()
    }

    /// Check if the tag is still valid (above threshold).
    ///
    /// A tag is valid if its current strength is above the validity threshold.
    /// Tags below threshold are considered expired and cannot support consolidation.
    pub fn is_valid(&self) -> bool {
        self.is_valid_with_threshold(DEFAULT_VALIDITY_THRESHOLD)
    }

    /// Check validity against a custom threshold.
    pub fn is_valid_with_threshold(&self, threshold: f64) -> bool {
        self.current_strength() >= threshold
    }

    /// Check if this memory can be consolidated.
    ///
    /// Consolidation requires:
    /// 1. Tag is still valid (above threshold)
    /// 2. PRPs are available for capture
    pub fn can_consolidate(&self) -> bool {
        self.is_valid() && self.prp_available
    }

    /// Mark PRPs as available for this tag.
    ///
    /// PRPs can come from:
    /// - Behavioral tagging (novelty, emotional arousal)
    /// - Dopaminergic modulation
    /// - Sleep-related protein synthesis
    pub fn set_prp_available(&mut self) {
        self.prp_available = true;
        self.prp_available_at = Some(Utc::now());
    }

    /// Mark PRPs as available at a specific time.
    pub fn set_prp_available_at(&mut self, at: DateTime<Utc>) {
        self.prp_available = true;
        self.prp_available_at = Some(at);
    }

    /// Calculate time remaining until tag drops below threshold.
    ///
    /// Returns None if the tag is already below threshold.
    ///
    /// # Formula
    ///
    /// From S(t) = S_0 * e^(-t/tau), solving for t:
    /// t = -tau * ln(threshold/S_0)
    pub fn time_to_threshold(&self) -> Option<Duration> {
        self.time_to_threshold_with(DEFAULT_VALIDITY_THRESHOLD)
    }

    /// Calculate time to a custom threshold.
    pub fn time_to_threshold_with(&self, threshold: f64) -> Option<Duration> {
        if threshold <= 0.0 || threshold >= self.initial_strength {
            return None;
        }

        let current = self.current_strength();
        if current < threshold {
            return None; // Already below threshold
        }

        // Time from now until threshold
        // S(t) = current * e^(-t/tau) = threshold
        // -t/tau = ln(threshold/current)
        // t = -tau * ln(threshold/current)
        let t_minutes = -self.tau * (threshold / current).ln();

        if t_minutes <= 0.0 {
            return None;
        }

        Some(Duration::milliseconds((t_minutes * 60_000.0) as i64))
    }

    /// Get the age of this tag.
    pub fn age(&self) -> Duration {
        Utc::now().signed_duration_since(self.tagged_at)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_tag_has_full_strength() {
        let tag = SynapticTag::new("mem1".to_string(), 1.0);

        // Just created, should be at ~100% strength
        let strength = tag.current_strength();
        assert!(
            strength > 0.99,
            "New tag should be near initial strength: {}",
            strength
        );
    }

    #[test]
    fn test_strength_decays_at_tau() {
        let past = Utc::now() - Duration::minutes(60);
        let tag = SynapticTag::with_timestamp("mem1".to_string(), 1.0, past);

        // At t = tau, strength should be ~37% (1/e)
        let strength = tag.current_strength();
        let expected = 1.0 / std::f64::consts::E;

        assert!(
            (strength - expected).abs() < 0.01,
            "At tau, strength should be ~{:.3} but was {:.3}",
            expected,
            strength
        );
    }

    #[test]
    fn test_strength_at_3_tau() {
        let past = Utc::now() - Duration::minutes(180); // 3 * tau
        let tag = SynapticTag::with_timestamp("mem1".to_string(), 1.0, past);

        // At t = 3*tau, strength should be ~5% (e^-3)
        let strength = tag.current_strength();
        let expected = (-3.0_f64).exp();

        assert!(
            (strength - expected).abs() < 0.01,
            "At 3*tau, strength should be ~{:.3} but was {:.3}",
            expected,
            strength
        );
    }

    #[test]
    fn test_initial_strength_clamped() {
        let tag_high = SynapticTag::new("mem1".to_string(), 1.5);
        assert_eq!(tag_high.initial_strength, 1.0);

        let tag_low = SynapticTag::new("mem2".to_string(), -0.5);
        assert_eq!(tag_low.initial_strength, 0.0);
    }

    #[test]
    fn test_custom_tau() {
        let past = Utc::now() - Duration::minutes(30);
        let tag = SynapticTag {
            memory_id: "mem1".to_string(),
            initial_strength: 1.0,
            tau: 30.0, // Faster decay
            tagged_at: past,
            prp_available: false,
            prp_available_at: None,
        };

        // At t = tau (30 min with tau=30), strength should be ~37%
        let strength = tag.current_strength();
        let expected = 1.0 / std::f64::consts::E;

        assert!(
            (strength - expected).abs() < 0.01,
            "At custom tau, strength should be ~{:.3} but was {:.3}",
            expected,
            strength
        );
    }

    #[test]
    fn test_is_valid() {
        // Fresh tag should be valid
        let tag = SynapticTag::new("mem1".to_string(), 1.0);
        assert!(tag.is_valid());

        // Old tag should be invalid
        let old = Utc::now() - Duration::hours(4); // Well past decay
        let old_tag = SynapticTag::with_timestamp("mem2".to_string(), 1.0, old);
        assert!(!old_tag.is_valid());
    }

    #[test]
    fn test_can_consolidate() {
        // Fresh tag without PRP cannot consolidate
        let mut tag = SynapticTag::new("mem1".to_string(), 1.0);
        assert!(!tag.can_consolidate());

        // Fresh tag with PRP can consolidate
        tag.set_prp_available();
        assert!(tag.can_consolidate());

        // Old tag with PRP cannot consolidate (tag expired)
        let old = Utc::now() - Duration::hours(4);
        let mut old_tag = SynapticTag::with_timestamp("mem2".to_string(), 1.0, old);
        old_tag.set_prp_available();
        assert!(!old_tag.can_consolidate());
    }

    #[test]
    fn test_prp_timestamp() {
        let mut tag = SynapticTag::new("mem1".to_string(), 1.0);
        assert!(tag.prp_available_at.is_none());

        tag.set_prp_available();
        assert!(tag.prp_available_at.is_some());
    }

    #[test]
    fn test_time_to_threshold() {
        let tag = SynapticTag::new("mem1".to_string(), 1.0);

        let time_to = tag.time_to_threshold();
        assert!(time_to.is_some());

        // Time should be positive
        let duration = time_to.unwrap();
        assert!(duration > Duration::zero());

        // Should be roughly: t = -60 * ln(0.1) = ~138 minutes
        let expected_minutes = -60.0 * (0.1_f64).ln();
        let actual_minutes = duration.num_milliseconds() as f64 / 60_000.0;

        assert!(
            (actual_minutes - expected_minutes).abs() < 1.0,
            "Expected ~{:.1} min, got {:.1} min",
            expected_minutes,
            actual_minutes
        );
    }

    #[test]
    fn test_time_to_threshold_already_expired() {
        let old = Utc::now() - Duration::hours(4);
        let tag = SynapticTag::with_timestamp("mem1".to_string(), 1.0, old);

        // Tag already expired, should return None
        assert!(tag.time_to_threshold().is_none());
    }

    #[test]
    fn test_strength_at_specific_time() {
        let tag = SynapticTag::new("mem1".to_string(), 1.0);
        let future = tag.tagged_at + Duration::minutes(60);

        let strength = tag.strength_at(future);
        let expected = 1.0 / std::f64::consts::E;

        assert!(
            (strength - expected).abs() < 0.01,
            "At t=tau, strength should be ~{:.3}",
            expected
        );

        // Before creation returns 0
        let before = tag.tagged_at - Duration::minutes(10);
        assert_eq!(tag.strength_at(before), 0.0);
    }

    #[test]
    fn test_serialization() {
        let tag = SynapticTag::new("mem1".to_string(), 0.8);
        let json = serde_json::to_string(&tag).unwrap();
        let restored: SynapticTag = serde_json::from_str(&json).unwrap();

        assert_eq!(restored.memory_id, tag.memory_id);
        assert!((restored.initial_strength - tag.initial_strength).abs() < 0.001);
        assert!((restored.tau - tag.tau).abs() < 0.001);
    }
}
