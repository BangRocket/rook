//! rook-graph-stores - Graph store implementations for rook.
//!
//! This crate provides graph database implementations for storing
//! and querying memory relationships.
//!
//! # Supported Backends
//!
//! - **Neo4j** (feature: `neo4j`) - Neo4j graph database
//! - **Memgraph** (feature: `memgraph`) - Memgraph (Neo4j-compatible)
//! - **Kuzu** (feature: `kuzu`) - Embedded graph database
//! - **Neptune** (feature: `neptune`) - AWS Neptune

mod factory;

#[cfg(feature = "neo4j")]
mod neo4j;

#[cfg(feature = "memgraph")]
mod memgraph;

#[cfg(feature = "kuzu")]
mod kuzu;

#[cfg(feature = "neptune")]
mod neptune;

pub use factory::GraphStoreFactory;

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
