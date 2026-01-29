//! Memory consolidation module based on Synaptic Tagging and Capture (STC) hypothesis.
//!
//! This module implements the biological mechanisms of memory consolidation:
//!
//! - **Synaptic Tags**: Transient markers set at synapses during learning.
//!   Tags decay exponentially (tau ~60 min) unless captured by PRPs.
//!
//! - **Consolidation Phases**: Memory transitions through stages:
//!   - Immediate (0-6h): Highly labile, depends on tag strength
//!   - Early (6-24h): Requires tag + PRP for stabilization
//!   - Late (24-72h): Systems consolidation begins
//!   - Consolidated (72h+): Stable long-term storage
//!
//! # References
//!
//! - Frey & Morris (1997). Synaptic tagging and long-term potentiation.
//! - Redondo & Morris (2011). Making memories last: the synaptic tagging and capture hypothesis.

mod phases;
mod synaptic_tag;

pub use phases::ConsolidationPhase;
pub use synaptic_tag::SynapticTag;
