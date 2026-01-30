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
pub mod events;
pub mod export;
pub mod import;
pub mod ingestion;
pub mod intentions;
pub mod memory;
pub mod migration;
#[cfg(feature = "multimodal")]
pub mod multimodal;
pub mod retrieval;
pub mod runtime;
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
    ActionResult, BloomConfig, CheckerConfig, FiredIntention, FiredIntentionReceiver, Intention,
    IntentionAction, IntentionChecker, IntentionScheduler, IntentionStore, KeywordBloomFilter,
    SqliteIntentionStore, TriggerCondition, TriggerReason,
};
pub use events::{
    AccessType, EventBus, EventSubscriber, MemoryAccessedEvent, MemoryCreatedEvent,
    MemoryDeletedEvent, MemoryLifecycleEvent, MemoryUpdatedEvent, RetryPolicy, UpdateType,
    WebhookConfig, WebhookDelivery, WebhookError, WebhookManager, verify_signature,
};
pub use runtime::{BackgroundRuntime, RuntimeConfig};

// Multimodal extraction (feature-gated)
#[cfg(feature = "multimodal")]
pub use multimodal::{MultimodalConfig, MultimodalIngester, MultimodalIngestResult, SourceProvenance};

// Export/Import utilities
pub use export::{export_jsonl, ExportStats, ExportableMemory};
#[cfg(feature = "export")]
pub use export::export_parquet;
pub use import::{import_jsonl, ImportStats, ImportableMemory};

// Migration utilities
pub use migration::{migrate_from_mem0, Mem0Memory, MigrationStats};
