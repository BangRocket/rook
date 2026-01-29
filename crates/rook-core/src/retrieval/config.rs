//! Configuration for spreading activation algorithm.

use serde::{Deserialize, Serialize};

/// Configuration for spreading activation algorithm.
///
/// Based on Collins & Loftus (1975) spreading activation theory
/// with bounded propagation to prevent context explosion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpreadingConfig {
    /// Decay factor per hop (activation multiplier per edge traversal).
    /// Range: 0.0-1.0. Lower = faster decay.
    /// Default: 0.7 (30% loss per hop)
    pub decay_factor: f32,

    /// Minimum activation to continue spreading.
    /// Propagation stops when activation falls below this threshold.
    /// Range: 0.0-1.0. Default: 0.1 (10% of initial)
    pub firing_threshold: f32,

    /// Maximum propagation depth (number of hops from seeds).
    /// Prevents unbounded traversal. Default: 3
    pub max_depth: usize,

    /// Fan-out penalty for high-degree nodes.
    /// Activation is divided by (1 + fan_out_penalty * degree).
    /// Prevents hub nodes from flooding activation.
    /// Default: 0.1
    pub fan_out_penalty: f32,
}

impl Default for SpreadingConfig {
    fn default() -> Self {
        Self {
            decay_factor: 0.7,
            firing_threshold: 0.1,
            max_depth: 3,
            fan_out_penalty: 0.1,
        }
    }
}

impl SpreadingConfig {
    /// Create config optimized for quick retrieval (narrow spread).
    pub fn narrow() -> Self {
        Self {
            decay_factor: 0.5,
            firing_threshold: 0.2,
            max_depth: 2,
            fan_out_penalty: 0.15,
        }
    }

    /// Create config optimized for thorough retrieval (wide spread).
    pub fn wide() -> Self {
        Self {
            decay_factor: 0.8,
            firing_threshold: 0.05,
            max_depth: 4,
            fan_out_penalty: 0.05,
        }
    }

    /// Validate configuration values are in valid ranges.
    pub fn validate(&self) -> Result<(), &'static str> {
        if !(0.0..=1.0).contains(&self.decay_factor) {
            return Err("decay_factor must be between 0.0 and 1.0");
        }
        if !(0.0..=1.0).contains(&self.firing_threshold) {
            return Err("firing_threshold must be between 0.0 and 1.0");
        }
        if self.max_depth == 0 {
            return Err("max_depth must be at least 1");
        }
        if self.fan_out_penalty < 0.0 {
            return Err("fan_out_penalty must be non-negative");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = SpreadingConfig::default();
        assert!((config.decay_factor - 0.7).abs() < 0.01);
        assert!((config.firing_threshold - 0.1).abs() < 0.01);
        assert_eq!(config.max_depth, 3);
        assert!((config.fan_out_penalty - 0.1).abs() < 0.01);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_narrow_config() {
        let config = SpreadingConfig::narrow();
        assert!((config.decay_factor - 0.5).abs() < 0.01);
        assert_eq!(config.max_depth, 2);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_wide_config() {
        let config = SpreadingConfig::wide();
        assert!((config.decay_factor - 0.8).abs() < 0.01);
        assert_eq!(config.max_depth, 4);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_validation_errors() {
        let invalid_decay = SpreadingConfig {
            decay_factor: 1.5,
            ..Default::default()
        };
        assert!(invalid_decay.validate().is_err());

        let invalid_threshold = SpreadingConfig {
            firing_threshold: -0.1,
            ..Default::default()
        };
        assert!(invalid_threshold.validate().is_err());

        let invalid_depth = SpreadingConfig {
            max_depth: 0,
            ..Default::default()
        };
        assert!(invalid_depth.validate().is_err());

        let invalid_penalty = SpreadingConfig {
            fan_out_penalty: -0.5,
            ..Default::default()
        };
        assert!(invalid_penalty.validate().is_err());
    }
}
