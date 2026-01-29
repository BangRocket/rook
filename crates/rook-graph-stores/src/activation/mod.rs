//! ACT-R activation model for cognitive-inspired memory retrieval.
//!
//! This module implements the base-level activation and retrieval probability
//! components from ACT-R (Adaptive Control of Thought-Rational) theory.
//!
//! # ACT-R Memory Model
//!
//! ACT-R models human memory as having activation levels that determine
//! how likely a memory is to be retrieved. Key principles:
//!
//! 1. **Recency**: Recently accessed memories are more active
//! 2. **Frequency**: Frequently accessed memories are more active
//! 3. **Decay**: Activation naturally decays over time
//! 4. **Noise**: Retrieval includes probabilistic noise (like human memory)
//!
//! # Base-Level Activation
//!
//! The core formula for base-level activation is:
//!
//! ```text
//! B_i = ln(sum(t_j^(-d))) + beta_i
//! ```
//!
//! Where:
//! - `t_j` = time since j-th access (seconds)
//! - `d` = decay parameter (typically 0.5)
//! - `beta_i` = base-level constant
//!
//! # Retrieval Probability
//!
//! The probability of successful retrieval follows a logistic function:
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
//! # Example
//!
//! ```
//! use chrono::{Utc, Duration};
//! use rook_graph_stores::activation::{
//!     ActivationConfig, AccessRecord,
//!     base_level_activation, retrieval_probability,
//! };
//!
//! let config = ActivationConfig::default();
//! let now = Utc::now();
//!
//! // Memory accessed 5 times over the past day
//! let accesses = vec![
//!     AccessRecord::at(now - Duration::hours(1), "retrieval"),
//!     AccessRecord::at(now - Duration::hours(4), "retrieval"),
//!     AccessRecord::at(now - Duration::hours(8), "retrieval"),
//!     AccessRecord::at(now - Duration::hours(16), "retrieval"),
//!     AccessRecord::at(now - Duration::hours(24), "creation"),
//! ];
//!
//! let activation = base_level_activation(&accesses, &config, now);
//! let prob = retrieval_probability(activation, &config);
//!
//! println!("Activation: {:.2}, Retrieval probability: {:.1}%", activation, prob * 100.0);
//! ```

pub mod base_level;
pub mod config;
pub mod noise;

// Re-export main types and functions
pub use base_level::{
    base_level_activation, base_level_activation_from_timestamps, time_until_threshold,
    AccessRecord,
};
pub use config::ActivationConfig;
pub use noise::{
    activation_noise, retrieval_probability, retrieval_probability_deterministic,
};
