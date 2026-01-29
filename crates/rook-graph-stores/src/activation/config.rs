//! Configuration for ACT-R activation model.
//!
//! Provides configuration parameters for base-level activation calculation
//! and retrieval probability using logistic noise.

use serde::{Deserialize, Serialize};

/// Configuration for ACT-R base-level activation.
///
/// The ACT-R activation model calculates memory strength based on:
/// - Recency: More recent accesses contribute more activation
/// - Frequency: More accesses contribute more activation
/// - Decay: Activation decays with time since access
///
/// The base-level activation formula is:
/// ```text
/// B_i = ln(sum(t_j^(-d))) + beta_i
/// ```
///
/// Where:
/// - `t_j` is the time (in seconds) since the j-th presentation
/// - `d` is the decay parameter (typically 0.5)
/// - `beta_i` is the base-level constant
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ActivationConfig {
    /// Decay parameter (d). Controls how fast activation decays.
    ///
    /// Higher values = faster decay.
    /// ACT-R default is 0.5.
    pub decay: f64,

    /// Base-level constant (beta_i). Added to all activation calculations.
    ///
    /// Can be used to give certain entities a baseline activation boost.
    /// Default is 0.0.
    pub base_constant: f64,

    /// Retrieval threshold (tau). Minimum activation for retrieval.
    ///
    /// Memories with activation below this threshold have low recall probability.
    /// Default is -2.0 (fairly accessible).
    pub retrieval_threshold: f64,

    /// Noise scale parameter (s). Controls variability in recall.
    ///
    /// Higher values = more random recall; lower values = more deterministic.
    /// ACT-R default is around 0.4.
    pub noise_scale: f64,

    /// Minimum time in seconds for activation calculation.
    ///
    /// Guards against t=0 which would cause ln(0) = -infinity.
    /// Access timestamps closer than this are treated as this value.
    /// Default is 0.05 seconds (50ms).
    pub min_time_seconds: f64,

    /// Default activation for entities with no access history.
    ///
    /// When an entity has never been accessed, this value is used.
    /// Default is -5.0 (very low activation, unlikely to be retrieved).
    pub default_activation: f64,
}

impl Default for ActivationConfig {
    fn default() -> Self {
        Self {
            decay: 0.5,
            base_constant: 0.0,
            retrieval_threshold: -2.0,
            noise_scale: 0.4,
            min_time_seconds: 0.05,
            default_activation: -5.0,
        }
    }
}

impl ActivationConfig {
    /// Create a new ActivationConfig with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a config with custom decay parameter.
    pub fn with_decay(mut self, decay: f64) -> Self {
        self.decay = decay;
        self
    }

    /// Create a config with custom base constant.
    pub fn with_base_constant(mut self, base_constant: f64) -> Self {
        self.base_constant = base_constant;
        self
    }

    /// Create a config with custom retrieval threshold.
    pub fn with_retrieval_threshold(mut self, threshold: f64) -> Self {
        self.retrieval_threshold = threshold;
        self
    }

    /// Create a config with custom noise scale.
    pub fn with_noise_scale(mut self, scale: f64) -> Self {
        self.noise_scale = scale;
        self
    }

    /// Create a config optimized for long-term memory (slower decay).
    pub fn long_term() -> Self {
        Self {
            decay: 0.3,
            retrieval_threshold: -3.0,
            ..Default::default()
        }
    }

    /// Create a config optimized for working memory (faster decay).
    pub fn working_memory() -> Self {
        Self {
            decay: 0.7,
            retrieval_threshold: -1.0,
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = ActivationConfig::default();
        assert_eq!(config.decay, 0.5);
        assert_eq!(config.base_constant, 0.0);
        assert_eq!(config.retrieval_threshold, -2.0);
        assert_eq!(config.noise_scale, 0.4);
        assert_eq!(config.min_time_seconds, 0.05);
    }

    #[test]
    fn test_builder_pattern() {
        let config = ActivationConfig::new()
            .with_decay(0.6)
            .with_base_constant(1.0)
            .with_retrieval_threshold(-1.5);

        assert_eq!(config.decay, 0.6);
        assert_eq!(config.base_constant, 1.0);
        assert_eq!(config.retrieval_threshold, -1.5);
    }

    #[test]
    fn test_presets() {
        let long_term = ActivationConfig::long_term();
        assert!(long_term.decay < 0.5); // Slower decay

        let working = ActivationConfig::working_memory();
        assert!(working.decay > 0.5); // Faster decay
    }

    #[test]
    fn test_serde() {
        let config = ActivationConfig::new().with_decay(0.6);
        let json = serde_json::to_string(&config).unwrap();
        let parsed: ActivationConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.decay, 0.6);
    }
}
