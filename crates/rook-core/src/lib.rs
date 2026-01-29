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
pub mod error;
pub mod memory;
pub mod traits;
pub mod types;

// Re-export commonly used types
pub use cognitive::{ArchivalCandidate, CognitiveStore, FsrsScheduler};
pub use config::MemoryConfig;
pub use error::{RookError, RookResult};
pub use memory::Memory;
pub use traits::{
    Embedder, EmbedderConfig, EmbeddingAction, Llm, LlmConfig, Reranker, VectorStore,
    VectorStoreConfig,
};
pub use types::{
    AddResult, ArchivalConfig, DualStrength, Filter, FsrsState, Grade, MemoryEvent, MemoryItem,
    MemoryResult, MemoryType, Message, MessageInput, MessageRole, SearchResult,
};
