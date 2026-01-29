//! Entity extraction and management module.
//!
//! This module provides LLM-based entity extraction from text,
//! including entity type detection, relationship identification,
//! and entity merging to prevent duplicates.
//!
//! # Components
//!
//! - `types`: Entity and relationship type definitions
//! - `extractor`: LLM-based entity extraction
//! - `merger`: Entity merging with configurable similarity threshold
//!
//! # Example
//!
//! ```ignore
//! use rook_graph_stores::entity::{EntityExtractor, MergeConfig, EntityMerger};
//!
//! // Extract entities from text
//! let extractor = EntityExtractor::new(llm);
//! let result = extractor.extract("Alice works at Acme Corp").await?;
//!
//! // Merge similar entities
//! let config = MergeConfig::default();
//! let merger = EntityMerger::new(config);
//! let merged = merger.find_match(&new_entity, &existing_entities, &embedder).await?;
//! ```

mod types;
mod extractor;
mod merger;

pub use types::{EntityType, RelationshipType};
pub use extractor::{EntityExtractor, ExtractedEntity, ExtractedRelationship, ExtractionResult};
pub use merger::{EntityMerger, MergeConfig, MergeResult};
