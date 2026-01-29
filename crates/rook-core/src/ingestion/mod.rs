//! Smart ingestion with prediction error gating and strength signal processing.
//!
//! This module implements multi-layer detection for determining how new
//! information should be handled:
//! - Skip: Duplicate/redundant information
//! - Create: Novel information to store
//! - Update: Refines existing memory
//! - Supersede: Contradicts and replaces existing memory
//!
//! It also provides automatic memory strength adjustments based on
//! user actions and system events.

pub mod layers;
pub mod prediction_error;
pub mod strength_signals;
pub mod types;

pub use layers::*;
pub use prediction_error::{GateResult, PredictionErrorGate};
pub use strength_signals::{StrengthSignal, StrengthSignalProcessor};
pub use types::*;
