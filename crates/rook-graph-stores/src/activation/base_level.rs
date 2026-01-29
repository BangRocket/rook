//! ACT-R base-level activation calculation.
//!
//! Implements the base-level learning equation from ACT-R theory:
//!
//! ```text
//! B_i = ln(sum(t_j^(-d))) + beta_i
//! ```
//!
//! Where:
//! - `t_j` is the time (in seconds) since the j-th presentation/access
//! - `d` is the decay parameter (typically 0.5)
//! - `beta_i` is the base-level constant
//!
//! This models how memories become more accessible with recent and frequent
//! access, while naturally decaying over time when not accessed.

use chrono::{DateTime, Utc};

use super::config::ActivationConfig;

/// Access record for activation calculation.
#[derive(Debug, Clone)]
pub struct AccessRecord {
    /// When the access occurred.
    pub timestamp: DateTime<Utc>,
    /// Type of access (e.g., "retrieval", "creation", "update").
    pub access_type: String,
    /// Activation boost from this access (default 1.0).
    pub activation_score: f64,
}

impl AccessRecord {
    /// Create a new access record at the current time.
    pub fn now(access_type: impl Into<String>) -> Self {
        Self {
            timestamp: Utc::now(),
            access_type: access_type.into(),
            activation_score: 1.0,
        }
    }

    /// Create an access record at a specific time.
    pub fn at(timestamp: DateTime<Utc>, access_type: impl Into<String>) -> Self {
        Self {
            timestamp,
            access_type: access_type.into(),
            activation_score: 1.0,
        }
    }

    /// Set the activation score for this access.
    pub fn with_score(mut self, score: f64) -> Self {
        self.activation_score = score;
        self
    }
}

/// Calculate base-level activation from access history.
///
/// Implements: B_i = ln(sum(t_j^(-d))) + beta_i
///
/// # Arguments
///
/// * `access_history` - List of access timestamps
/// * `config` - Activation configuration parameters
/// * `now` - Current time for calculating time differences
///
/// # Returns
///
/// The base-level activation value (can be negative).
/// Returns `config.default_activation` if access history is empty.
///
/// # Example
///
/// ```
/// use chrono::{Utc, Duration};
/// use rook_graph_stores::activation::{base_level_activation, AccessRecord, ActivationConfig};
///
/// let config = ActivationConfig::default();
/// let now = Utc::now();
///
/// // Access 10 seconds ago
/// let access = AccessRecord::at(now - Duration::seconds(10), "retrieval");
/// let activation = base_level_activation(&[access], &config, now);
///
/// // More recent access = higher activation
/// let recent_access = AccessRecord::at(now - Duration::seconds(1), "retrieval");
/// let recent_activation = base_level_activation(&[recent_access], &config, now);
/// assert!(recent_activation > activation);
/// ```
pub fn base_level_activation(
    access_history: &[AccessRecord],
    config: &ActivationConfig,
    now: DateTime<Utc>,
) -> f64 {
    // Handle empty history
    if access_history.is_empty() {
        return config.default_activation;
    }

    // Calculate sum of t_j^(-d) weighted by activation scores
    let mut sum = 0.0;

    for access in access_history {
        // Calculate time since access in seconds
        let duration = now.signed_duration_since(access.timestamp);
        let mut t_seconds = duration.num_milliseconds() as f64 / 1000.0;

        // Guard against t <= 0 (access in future or exactly now)
        if t_seconds < config.min_time_seconds {
            t_seconds = config.min_time_seconds;
        }

        // t_j^(-d) * activation_score
        let contribution = t_seconds.powf(-config.decay) * access.activation_score;
        sum += contribution;
    }

    // Handle edge case where sum is 0 or negative (shouldn't happen with valid data)
    if sum <= 0.0 {
        return config.default_activation;
    }

    // B_i = ln(sum) + beta_i
    sum.ln() + config.base_constant
}

/// Calculate base-level activation from raw timestamps.
///
/// Convenience function when you only have timestamps without metadata.
pub fn base_level_activation_from_timestamps(
    timestamps: &[DateTime<Utc>],
    config: &ActivationConfig,
    now: DateTime<Utc>,
) -> f64 {
    let access_history: Vec<AccessRecord> = timestamps
        .iter()
        .map(|&ts| AccessRecord::at(ts, "access"))
        .collect();

    base_level_activation(&access_history, config, now)
}

/// Calculate the time until activation drops below a threshold.
///
/// Useful for predicting when a memory will become "forgotten" (hard to retrieve).
///
/// # Arguments
///
/// * `current_activation` - Current base-level activation
/// * `threshold` - Activation threshold to predict crossing
/// * `config` - Activation configuration
///
/// # Returns
///
/// Approximate seconds until activation drops below threshold, or None if
/// activation is already below threshold or would take extremely long.
pub fn time_until_threshold(
    current_activation: f64,
    threshold: f64,
    config: &ActivationConfig,
) -> Option<f64> {
    // If already below threshold
    if current_activation <= threshold {
        return None;
    }

    // For a single access at time t:
    // B = ln(t^(-d)) + beta = -d*ln(t) + beta
    // Solving for t when B = threshold:
    // threshold = -d*ln(t) + beta
    // ln(t) = (beta - threshold) / d
    // t = exp((beta - threshold) / d)
    //
    // This is an approximation assuming single-access decay pattern

    let activation_without_beta = current_activation - config.base_constant;
    let threshold_without_beta = threshold - config.base_constant;

    if activation_without_beta <= threshold_without_beta {
        return None;
    }

    // Time at which single-access activation equals threshold
    let ln_t = -threshold_without_beta / config.decay;
    let t = ln_t.exp();

    // Sanity check - don't return extremely large values
    if t > 365.0 * 24.0 * 60.0 * 60.0 {
        // More than a year
        return None;
    }

    Some(t)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;

    #[test]
    fn test_empty_history() {
        let config = ActivationConfig::default();
        let now = Utc::now();

        let activation = base_level_activation(&[], &config, now);
        assert_eq!(activation, config.default_activation);
    }

    #[test]
    fn test_single_recent_access() {
        let config = ActivationConfig::default();
        let now = Utc::now();

        // Access 1 second ago
        let access = AccessRecord::at(now - Duration::seconds(1), "retrieval");
        let activation = base_level_activation(&[access], &config, now);

        // With d=0.5, t=1: ln(1^-0.5) + 0 = ln(1) = 0
        // So activation should be around 0
        assert!(activation.abs() < 1.0, "Expected activation near 0, got {}", activation);
    }

    #[test]
    fn test_recency_effect() {
        let config = ActivationConfig::default();
        let now = Utc::now();

        // Recent access (1 second ago)
        let recent = AccessRecord::at(now - Duration::seconds(1), "retrieval");
        let recent_activation = base_level_activation(&[recent], &config, now);

        // Old access (100 seconds ago)
        let old = AccessRecord::at(now - Duration::seconds(100), "retrieval");
        let old_activation = base_level_activation(&[old], &config, now);

        // Recent access should have higher activation
        assert!(
            recent_activation > old_activation,
            "Recent {} should be > old {}",
            recent_activation,
            old_activation
        );
    }

    #[test]
    fn test_frequency_effect() {
        let config = ActivationConfig::default();
        let now = Utc::now();

        // Single access 10 seconds ago
        let single = vec![AccessRecord::at(now - Duration::seconds(10), "retrieval")];
        let single_activation = base_level_activation(&single, &config, now);

        // Multiple accesses at the same time
        let multiple = vec![
            AccessRecord::at(now - Duration::seconds(10), "retrieval"),
            AccessRecord::at(now - Duration::seconds(10), "retrieval"),
            AccessRecord::at(now - Duration::seconds(10), "retrieval"),
        ];
        let multiple_activation = base_level_activation(&multiple, &config, now);

        // More accesses should mean higher activation
        assert!(
            multiple_activation > single_activation,
            "Multiple {} should be > single {}",
            multiple_activation,
            single_activation
        );
    }

    #[test]
    fn test_base_constant_effect() {
        let now = Utc::now();
        let access = AccessRecord::at(now - Duration::seconds(10), "retrieval");

        let config_no_boost = ActivationConfig::default();
        let activation_no_boost = base_level_activation(&[access.clone()], &config_no_boost, now);

        let config_with_boost = ActivationConfig::default().with_base_constant(1.0);
        let activation_with_boost = base_level_activation(&[access], &config_with_boost, now);

        // Base constant adds directly to activation
        assert!(
            (activation_with_boost - activation_no_boost - 1.0).abs() < 0.001,
            "Expected 1.0 difference, got {}",
            activation_with_boost - activation_no_boost
        );
    }

    #[test]
    fn test_decay_parameter_effect() {
        let now = Utc::now();
        let access = AccessRecord::at(now - Duration::seconds(100), "retrieval");

        let slow_decay = ActivationConfig::default().with_decay(0.3);
        let slow_activation = base_level_activation(&[access.clone()], &slow_decay, now);

        let fast_decay = ActivationConfig::default().with_decay(0.7);
        let fast_activation = base_level_activation(&[access], &fast_decay, now);

        // Slower decay should preserve more activation
        assert!(
            slow_activation > fast_activation,
            "Slow decay {} should be > fast decay {}",
            slow_activation,
            fast_activation
        );
    }

    #[test]
    fn test_min_time_guard() {
        let config = ActivationConfig::default();
        let now = Utc::now();

        // Access at exactly now (0 seconds ago)
        let access = AccessRecord::at(now, "retrieval");
        let activation = base_level_activation(&[access], &config, now);

        // Should not panic or return infinity
        assert!(activation.is_finite(), "Activation should be finite");
    }

    #[test]
    fn test_weighted_access() {
        let config = ActivationConfig::default();
        let now = Utc::now();

        // Normal access
        let normal = AccessRecord::at(now - Duration::seconds(10), "retrieval");
        let normal_activation = base_level_activation(&[normal], &config, now);

        // High-weight access (e.g., explicit user interest)
        let weighted = AccessRecord::at(now - Duration::seconds(10), "retrieval").with_score(2.0);
        let weighted_activation = base_level_activation(&[weighted], &config, now);

        // Higher weight should mean higher activation
        assert!(
            weighted_activation > normal_activation,
            "Weighted {} should be > normal {}",
            weighted_activation,
            normal_activation
        );
    }

    #[test]
    fn test_from_timestamps() {
        let config = ActivationConfig::default();
        let now = Utc::now();

        let timestamps = vec![
            now - Duration::seconds(1),
            now - Duration::seconds(10),
            now - Duration::seconds(100),
        ];

        let activation = base_level_activation_from_timestamps(&timestamps, &config, now);

        // Should be higher than single access due to multiple presentations
        let single = base_level_activation_from_timestamps(&timestamps[..1], &config, now);
        assert!(activation > single);
    }

    #[test]
    fn test_time_until_threshold() {
        let config = ActivationConfig::default();

        // High activation that will decay
        let result = time_until_threshold(1.0, -2.0, &config);
        assert!(result.is_some());

        // Already below threshold
        let result = time_until_threshold(-3.0, -2.0, &config);
        assert!(result.is_none());
    }
}
