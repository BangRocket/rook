//! Tantivy full-text search integration for BM25 keyword queries.
//!
//! Tantivy provides fast, Lucene-like full-text search with BM25 scoring.
//! This module wraps Tantivy to index memory content for keyword retrieval.

use std::path::Path;
use std::sync::Mutex;

use serde::{Deserialize, Serialize};
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::{Field, Schema, STORED, STRING, TEXT};
use tantivy::{doc, Index, IndexReader, IndexWriter, ReloadPolicy, TantivyDocument};

use crate::error::{RookError, RookResult};

/// Result from text search with BM25 score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextSearchResult {
    /// Memory ID that matched.
    pub id: String,
    /// BM25 relevance score (unbounded, higher = more relevant).
    pub bm25_score: f32,
    /// Normalized score (0.0-1.0) for fusion.
    pub normalized_score: f32,
}

/// Tantivy-based full-text search for memory content.
///
/// Manages a persistent Tantivy index with BM25 scoring.
/// Thread-safe: uses internal Mutex for IndexWriter.
pub struct TantivySearcher {
    index: Index,
    reader: IndexReader,
    writer: Mutex<IndexWriter>,
    #[allow(dead_code)]
    schema: Schema,
    id_field: Field,
    content_field: Field,
    #[allow(dead_code)]
    metadata_field: Field,
}

impl TantivySearcher {
    /// Create a new TantivySearcher with a persistent index.
    ///
    /// # Arguments
    /// * `index_path` - Directory path for the index (will be created if needed)
    ///
    /// # Errors
    /// Returns error if index creation or opening fails.
    pub fn new(index_path: &Path) -> RookResult<Self> {
        // Build schema
        let mut schema_builder = Schema::builder();

        // STRING for exact match ID (stored for retrieval)
        let id_field = schema_builder.add_text_field("id", STRING | STORED);

        // TEXT for full-text search on content
        let content_field = schema_builder.add_text_field("content", TEXT | STORED);

        // STRING for optional metadata (JSON string)
        let metadata_field = schema_builder.add_text_field("metadata", STRING | STORED);

        let schema = schema_builder.build();

        // Create or open index
        let index = if index_path.exists() {
            Index::open_in_dir(index_path)
                .map_err(|e| RookError::Configuration(format!("Failed to open Tantivy index: {}", e)))?
        } else {
            std::fs::create_dir_all(index_path)
                .map_err(|e| RookError::Configuration(format!("Failed to create index dir: {}", e)))?;
            Index::create_in_dir(index_path, schema.clone())
                .map_err(|e| RookError::Configuration(format!("Failed to create Tantivy index: {}", e)))?
        };

        // Create writer (heap size 50MB)
        let writer = index
            .writer(50_000_000)
            .map_err(|e| RookError::Configuration(format!("Failed to create index writer: {}", e)))?;

        // Create reader with reload policy
        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::OnCommitWithDelay)
            .try_into()
            .map_err(|e| RookError::Configuration(format!("Failed to create index reader: {}", e)))?;

        Ok(Self {
            index,
            reader,
            writer: Mutex::new(writer),
            schema,
            id_field,
            content_field,
            metadata_field,
        })
    }

    /// Create an in-memory TantivySearcher (for testing).
    pub fn in_memory() -> RookResult<Self> {
        let mut schema_builder = Schema::builder();
        let id_field = schema_builder.add_text_field("id", STRING | STORED);
        let content_field = schema_builder.add_text_field("content", TEXT | STORED);
        let metadata_field = schema_builder.add_text_field("metadata", STRING | STORED);
        let schema = schema_builder.build();

        let index = Index::create_in_ram(schema.clone());
        let writer = index
            .writer(15_000_000)
            .map_err(|e| RookError::Configuration(format!("Failed to create writer: {}", e)))?;
        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::Manual)
            .try_into()
            .map_err(|e| RookError::Configuration(format!("Failed to create reader: {}", e)))?;

        Ok(Self {
            index,
            reader,
            writer: Mutex::new(writer),
            schema,
            id_field,
            content_field,
            metadata_field,
        })
    }
}
