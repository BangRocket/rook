//! Logistic noise and retrieval probability for ACT-R activation.
//!
//! This module implements the probabilistic retrieval component of ACT-R:
//!
//! ```text
//! P(recall) = 1 / (1 + exp((tau - A) / s))
//! ```
//!
//! Where:
//! - `tau` = retrieval threshold
//! - `A` = total activation (base-level + noise)
//! - `s` = noise scale parameter
//!
//! The noise follows a logistic distribution, which models the inherent
//! variability in human memory retrieval.

use rand::Rng;
use std::f64::consts::PI;

use super::config::ActivationConfig;

/// Generate activation noise from a logistic distribution.
///
/// The logistic distribution is used because:
/// 1. It has heavier tails than normal distribution, modeling occasional
///    surprising recalls/forgetting
/// 2. It's computationally efficient to sample via inverse CDF
/// 3. It's the standard in ACT-R literature
///
/// The noise has mean 0 and scale parameter `s` from the config.
///
/// # Arguments
///
/// * `config` - Configuration with noise_scale parameter
///
/// # Returns
///
/// Random noise value (can be positive or negative)
pub fn activation_noise(config: &ActivationConfig) -> f64 {
    let mut rng = rand::thread_rng();
    activation_noise_with_rng(&mut rng, config)
}

/// Generate activation noise with a provided RNG.
///
/// Useful for reproducible testing.
pub fn activation_noise_with_rng<R: Rng>(rng: &mut R, config: &ActivationConfig) -> f64 {
    // Sample from uniform (0, 1)
    let u: f64 = rng.gen_range(0.001..0.999);

    // Inverse CDF of logistic distribution: s * ln(u / (1 - u))
    config.noise_scale * (u / (1.0 - u)).ln()
}

/// Calculate retrieval probability given activation.
///
/// Implements the ACT-R retrieval probability formula:
/// ```text
/// P(recall) = 1 / (1 + exp((tau - A) / s))
/// ```
///
/// This is a sigmoidal function where:
/// - Activation well above threshold -> probability near 1
/// - Activation at threshold -> probability 0.5
/// - Activation well below threshold -> probability near 0
///
/// # Arguments
///
/// * `activation` - Total activation (base-level + any associative spreading)
/// * `config` - Configuration with retrieval_threshold and noise_scale
///
/// # Returns
///
/// Probability of successful retrieval, range [0, 1]
///
/// # Example
///
/// ```
/// use rook_graph_stores::activation::{ActivationConfig, retrieval_probability};
///
/// let config = ActivationConfig::default();
///
/// // High activation -> high probability
/// let prob_high = retrieval_probability(1.0, &config);
/// assert!(prob_high > 0.9);
///
/// // Low activation -> low probability
/// let prob_low = retrieval_probability(-5.0, &config);
/// assert!(prob_low < 0.1);
///
/// // At threshold -> probability around 0.5
/// let prob_threshold = retrieval_probability(config.retrieval_threshold, &config);
/// assert!((prob_threshold - 0.5).abs() < 0.1);
/// ```
pub fn retrieval_probability(activation: f64, config: &ActivationConfig) -> f64 {
    retrieval_probability_deterministic(activation, config)
}

/// Calculate deterministic retrieval probability (no noise).
///
/// This version is useful for testing and when you want predictable results.
/// In practice, you might want to add noise to the activation before calling this.
///
/// # Arguments
///
/// * `activation` - Activation value
/// * `config` - Configuration parameters
///
/// # Returns
///
/// Probability in range [0, 1]
pub fn retrieval_probability_deterministic(activation: f64, config: &ActivationConfig) -> f64 {
    // P = 1 / (1 + exp((tau - A) / s))
    let exponent = (config.retrieval_threshold - activation) / config.noise_scale;

    // Guard against overflow
    if exponent > 700.0 {
        return 0.0;
    }
    if exponent < -700.0 {
        return 1.0;
    }

    1.0 / (1.0 + exponent.exp())
}

/// Calculate retrieval probability with random noise.
///
/// Adds logistic noise to the activation before calculating probability.
/// This models the stochastic nature of human memory retrieval.
///
/// # Arguments
///
/// * `activation` - Base activation (typically from base_level_activation)
/// * `config` - Configuration parameters
///
/// # Returns
///
/// Probability in range [0, 1], will vary on each call due to noise
pub fn retrieval_probability_with_noise(activation: f64, config: &ActivationConfig) -> f64 {
    let noise = activation_noise(config);
    retrieval_probability_deterministic(activation + noise, config)
}

/// Perform a retrieval attempt with probabilistic success.
///
/// Returns true if the memory would be successfully retrieved.
///
/// # Arguments
///
/// * `activation` - Activation value
/// * `config` - Configuration parameters
///
/// # Returns
///
/// `true` if retrieval succeeds, `false` otherwise
pub fn attempt_retrieval(activation: f64, config: &ActivationConfig) -> bool {
    let mut rng = rand::thread_rng();
    attempt_retrieval_with_rng(&mut rng, activation, config)
}

/// Perform a retrieval attempt with provided RNG.
///
/// Useful for reproducible testing.
pub fn attempt_retrieval_with_rng<R: Rng>(
    rng: &mut R,
    activation: f64,
    config: &ActivationConfig,
) -> bool {
    // Add noise to activation
    let noisy_activation = activation + activation_noise_with_rng(rng, config);

    // Calculate probability with noisy activation
    let prob = retrieval_probability_deterministic(noisy_activation, config);

    // Roll the dice
    let roll: f64 = rng.gen();
    roll < prob
}

/// Calculate the expected latency for retrieval (in seconds).
///
/// ACT-R models retrieval time as inversely related to activation:
/// ```text
/// latency = F * exp(-f * A)
/// ```
///
/// Where F and f are scaling parameters.
///
/// Higher activation = faster retrieval (more accessible memory).
///
/// # Arguments
///
/// * `activation` - Activation value
/// * `latency_factor` - Base latency factor F (default 1.0)
/// * `latency_exponent` - Latency exponent f (default 1.0)
///
/// # Returns
///
/// Expected retrieval latency in seconds
pub fn retrieval_latency(activation: f64, latency_factor: f64, latency_exponent: f64) -> f64 {
    let latency = latency_factor * (-latency_exponent * activation).exp();

    // Clamp to reasonable bounds
    latency.clamp(0.01, 30.0) // 10ms to 30 seconds
}

/// Calculate the variance of the logistic distribution.
///
/// For a logistic distribution with scale s:
/// ```text
/// variance = (s * pi)^2 / 3
/// ```
///
/// This is useful for understanding the spread of activation noise.
pub fn logistic_variance(scale: f64) -> f64 {
    (scale * PI).powi(2) / 3.0
}

/// Calculate the standard deviation of the logistic distribution.
pub fn logistic_std_dev(scale: f64) -> f64 {
    logistic_variance(scale).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn test_config() -> ActivationConfig {
        ActivationConfig::default()
    }

    fn seeded_rng() -> StdRng {
        StdRng::seed_from_u64(42)
    }

    #[test]
    fn test_noise_distribution() {
        let config = test_config();
        let mut rng = seeded_rng();

        // Generate many samples
        let samples: Vec<f64> = (0..10000)
            .map(|_| activation_noise_with_rng(&mut rng, &config))
            .collect();

        // Mean should be approximately 0
        let mean: f64 = samples.iter().sum::<f64>() / samples.len() as f64;
        assert!(
            mean.abs() < 0.1,
            "Mean {} should be close to 0",
            mean
        );

        // Standard deviation should be close to theoretical value
        let variance: f64 = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
            / samples.len() as f64;
        let std_dev = variance.sqrt();
        let expected_std = logistic_std_dev(config.noise_scale);

        assert!(
            (std_dev - expected_std).abs() < 0.1,
            "Std dev {} should be close to theoretical {}",
            std_dev,
            expected_std
        );
    }

    #[test]
    fn test_retrieval_probability_high_activation() {
        let config = test_config();

        // Very high activation should give probability near 1
        let prob = retrieval_probability(5.0, &config);
        assert!(prob > 0.99, "High activation should give prob > 0.99, got {}", prob);
    }

    #[test]
    fn test_retrieval_probability_low_activation() {
        let config = test_config();

        // Very low activation should give probability near 0
        let prob = retrieval_probability(-10.0, &config);
        assert!(prob < 0.01, "Low activation should give prob < 0.01, got {}", prob);
    }

    #[test]
    fn test_retrieval_probability_at_threshold() {
        let config = test_config();

        // At threshold, probability should be exactly 0.5
        let prob = retrieval_probability(config.retrieval_threshold, &config);
        assert!(
            (prob - 0.5).abs() < 0.001,
            "At threshold, prob should be 0.5, got {}",
            prob
        );
    }

    #[test]
    fn test_retrieval_probability_monotonic() {
        let config = test_config();

        // Probability should increase with activation
        let activations = vec![-5.0, -3.0, -1.0, 0.0, 1.0, 3.0, 5.0];
        let probs: Vec<f64> = activations.iter().map(|a| retrieval_probability(*a, &config)).collect();

        for i in 1..probs.len() {
            assert!(
                probs[i] >= probs[i - 1],
                "Probability should be monotonically increasing: {:?}",
                probs
            );
        }
    }

    #[test]
    fn test_retrieval_probability_overflow_protection() {
        let config = test_config();

        // Extremely high activation should not cause overflow
        let prob_high = retrieval_probability(1000.0, &config);
        assert!((prob_high - 1.0).abs() < 0.001);

        // Extremely low activation should not cause overflow
        let prob_low = retrieval_probability(-1000.0, &config);
        assert!(prob_low.abs() < 0.001);
    }

    #[test]
    fn test_attempt_retrieval_statistical() {
        let config = test_config();
        let mut rng = seeded_rng();

        // With high activation, most attempts should succeed
        let high_activation = 3.0;
        let high_successes: usize = (0..1000)
            .filter(|_| attempt_retrieval_with_rng(&mut rng, high_activation, &config))
            .count();
        assert!(
            high_successes > 800,
            "High activation should succeed most of the time: {}",
            high_successes
        );

        // With low activation, few attempts should succeed
        let low_activation = -5.0;
        let low_successes: usize = (0..1000)
            .filter(|_| attempt_retrieval_with_rng(&mut rng, low_activation, &config))
            .count();
        assert!(
            low_successes < 200,
            "Low activation should rarely succeed: {}",
            low_successes
        );
    }

    #[test]
    fn test_retrieval_latency() {
        // Higher activation = faster retrieval
        let fast = retrieval_latency(2.0, 1.0, 1.0);
        let slow = retrieval_latency(-1.0, 1.0, 1.0);

        assert!(fast < slow, "Higher activation should be faster: {} vs {}", fast, slow);

        // Latency should be bounded
        assert!(fast >= 0.01);
        assert!(slow <= 30.0);
    }

    #[test]
    fn test_logistic_statistics() {
        let scale = 0.4;

        let variance = logistic_variance(scale);
        let std_dev = logistic_std_dev(scale);

        // Verify relationship
        assert!((std_dev.powi(2) - variance).abs() < 0.0001);

        // Verify approximate expected value
        // For s=0.4: variance = (0.4 * pi)^2 / 3 ≈ 0.526
        assert!(
            (variance - 0.526).abs() < 0.01,
            "Variance {} should be close to 0.526",
            variance
        );
    }

    #[test]
    fn test_noise_scale_effect() {
        let small_scale = ActivationConfig::default().with_noise_scale(0.1);
        let large_scale = ActivationConfig::default().with_noise_scale(1.0);
        let mut rng = seeded_rng();

        // Collect samples
        let small_samples: Vec<f64> = (0..1000)
            .map(|_| activation_noise_with_rng(&mut rng, &small_scale).abs())
            .collect();
        let large_samples: Vec<f64> = (0..1000)
            .map(|_| activation_noise_with_rng(&mut rng, &large_scale).abs())
            .collect();

        let small_mean: f64 = small_samples.iter().sum::<f64>() / small_samples.len() as f64;
        let large_mean: f64 = large_samples.iter().sum::<f64>() / large_samples.len() as f64;

        // Larger scale should produce larger absolute noise values
        assert!(
            large_mean > small_mean * 5.0,
            "Larger scale should produce larger noise: {} vs {}",
            large_mean,
            small_mean
        );
    }

    #[test]
    fn test_with_noise_variability() {
        let config = test_config();

        // Same activation should give varying results with noise
        let activation = 0.0;
        let probs: Vec<f64> = (0..100)
            .map(|_| retrieval_probability_with_noise(activation, &config))
            .collect();

        // Not all probabilities should be the same
        let first = probs[0];
        let all_same = probs.iter().all(|p| (*p - first).abs() < 0.001);
        assert!(!all_same, "With noise, probabilities should vary");

        // But they should all be valid probabilities
        assert!(probs.iter().all(|p| *p >= 0.0 && *p <= 1.0));
    }
}
