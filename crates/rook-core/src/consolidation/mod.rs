//! Memory consolidation module based on Synaptic Tagging and Capture (STC) hypothesis.
//!
//! This module implements the biological mechanisms of memory consolidation:
//!
//! - **Synaptic Tags**: Transient markers set at synapses during learning.
//!   Tags decay exponentially (tau ~60 min) unless captured by PRPs.
//!
//! - **Behavioral Tagging**: Novel events (high prediction error) boost
//!   consolidation of temporally-adjacent memories by providing PRPs.
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
//! - PMC4562088: Behavioral tagging research on novelty-based consolidation boost.

mod behavioral_tag;
pub mod manager;
mod phases;
pub mod scheduler;
mod synaptic_tag;

pub use behavioral_tag::{BehavioralTagConfig, BehavioralTagger, NoveltyResult};
pub use manager::{ConsolidationConfig, ConsolidationManager, ConsolidationResult};
pub use phases::ConsolidationPhase;
pub use scheduler::{ConsolidationScheduler, SchedulerConfig};
pub use synaptic_tag::SynapticTag;
