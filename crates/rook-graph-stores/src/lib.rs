//! rook-graph-stores - Graph store implementations for rook.
//!
//! This crate provides graph database implementations for storing
//! and querying memory relationships.
//!
//! # Supported Backends
//!
//! - **Embedded** (feature: `embedded`, default) - petgraph + SQLite hybrid
//! - **Neo4j** (feature: `neo4j`) - Neo4j graph database
//! - **Memgraph** (feature: `memgraph`) - Memgraph (Neo4j-compatible)
//! - **Neptune** (feature: `neptune`) - AWS Neptune
//!
//! # Architecture
//!
//! The embedded store uses a hybrid architecture:
//! - **SQLite** for persistent storage (entities, relationships, access logs)
//! - **petgraph** for O(1) in-memory neighbor lookups and traversal
//!
//! This provides both durability and fast graph operations without
//! requiring external database dependencies.

mod factory;

#[cfg(feature = "embedded")]
pub mod embedded;

#[cfg(feature = "neo4j")]
mod neo4j;

#[cfg(feature = "memgraph")]
mod memgraph;

#[cfg(feature = "kuzu")]
mod kuzu;

#[cfg(feature = "neptune")]
mod neptune;

pub use factory::GraphStoreFactory;

#[cfg(feature = "embedded")]
pub use embedded::EmbeddedGraphStore;

#[cfg(feature = "neo4j")]
pub use neo4j::Neo4jGraphStore;

#[cfg(feature = "memgraph")]
pub use memgraph::MemgraphGraphStore;

#[cfg(feature = "kuzu")]
pub use kuzu::KuzuGraphStore;

#[cfg(feature = "neptune")]
pub use neptune::NeptuneGraphStore;

// Re-export core types
pub use rook_core::traits::{GraphStore, GraphStoreConfig, GraphStoreProvider};
