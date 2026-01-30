//! Multimodal content ingestion for rook memory system.
//!
//! Provides unified ingestion API for documents (PDF, DOCX) and images,
//! storing extracted content as searchable memories with provenance tracking.

mod ingest;
mod types;

pub use ingest::MultimodalIngester;
pub use types::{MultimodalConfig, MultimodalIngestResult, SourceProvenance};
