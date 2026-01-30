//! rook-core - Core library for rook.
//!
//! This crate provides the core types, traits, and Memory implementation
//! for the rook memory layer for AI agents.
//!
//! # Example
//!
//! ```ignore
//! use rook_core::{Memory, MemoryConfig};
//!
//! let config = MemoryConfig::default();
//! let memory = Memory::new(config, llm, embedder, vector_store, None, None)?;
//!
//! // Add a memory
//! let result = memory.add("I like pizza", Some("user1".to_string()), None, None, None, true, None).await?;
//!
//! // Search for memories
//! let results = memory.search("food preferences", Some("user1".to_string()), None, None, 10, None, None, true).await?;
//! ```

pub mod cognitive;
pub mod config;
pub mod consolidation;
pub mod error;
pub mod ingestion;
pub mod intentions;
pub mod memory;
pub mod retrieval;
pub mod traits;
pub mod types;
pub mod versioning;

// Re-export commonly used types
pub use cognitive::{ArchivalCandidate, CognitiveStore, FsrsScheduler};
pub use config::MemoryConfig;
pub use consolidation::{
    BehavioralTagConfig, BehavioralTagger, ConsolidationConfig, ConsolidationManager,
    ConsolidationPhase, ConsolidationResult, ConsolidationScheduler, NoveltyResult, SchedulerConfig,
    SynapticTag,
};
pub use error::{RookError, RookResult};
pub use ingestion::{
    DetectionLayer, GateResult, GatingThresholds, IngestDecision, IngestResult,
    PredictionErrorGate, StrengthSignal, StrengthSignalProcessor,
};
pub use memory::Memory;
pub use traits::{
    Embedder, EmbedderConfig, EmbeddingAction, Llm, LlmConfig, Reranker, VectorStore,
    VectorStoreConfig,
};
pub use retrieval::{
    spread_activation, spread_activation_by_id, ActivatedMemory, ActivationEdge, ActivationNode,
    SpreadingConfig,
};
pub use types::{
    AddResult, ArchivalConfig, DualStrength, Filter, FsrsState, Grade, MemoryEvent, MemoryItem,
    MemoryResult, MemoryType, Message, MessageInput, MessageRole, SearchResult,
};
pub use versioning::{
    FsrsStateSnapshot, MemoryVersion, SqliteVersionStore, VersionEventType, VersionStore,
    VersionSummary,
};
pub use intentions::{
    ActionResult, FiredIntention, Intention, IntentionAction, IntentionStore,
    SqliteIntentionStore, TriggerCondition, TriggerReason,
};
