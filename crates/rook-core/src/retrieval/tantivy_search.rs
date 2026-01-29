//! Tantivy full-text search integration for BM25 keyword queries.
//!
//! Tantivy provides fast, Lucene-like full-text search with BM25 scoring.
//! This module wraps Tantivy to index memory content for keyword retrieval.

use std::path::Path;
use std::sync::Mutex;

use serde::{Deserialize, Serialize};
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::{Field, Schema, Value, STORED, STRING, TEXT};
use tantivy::{Index, IndexReader, IndexWriter, ReloadPolicy, TantivyDocument, Term};

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

// Make TantivySearcher Send + Sync safe
// IndexWriter is only accessed through Mutex
// Index and IndexReader are documented as thread-safe
unsafe impl Send for TantivySearcher {}
unsafe impl Sync for TantivySearcher {}

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

    /// Index a memory document.
    ///
    /// # Arguments
    /// * `id` - Unique memory ID
    /// * `content` - Text content to index
    /// * `metadata` - Optional JSON metadata string
    ///
    /// # Note
    /// Call `commit()` after adding documents to make them searchable.
    pub fn add(&self, id: &str, content: &str, metadata: Option<&str>) -> RookResult<()> {
        let writer = self
            .writer
            .lock()
            .map_err(|_| RookError::Configuration("Failed to acquire writer lock".to_string()))?;

        let mut doc = TantivyDocument::default();
        doc.add_text(self.id_field, id);
        doc.add_text(self.content_field, content);
        if let Some(meta) = metadata {
            doc.add_text(self.metadata_field, meta);
        }

        writer
            .add_document(doc)
            .map_err(|e| RookError::Configuration(format!("Failed to add document: {}", e)))?;

        Ok(())
    }

    /// Update a memory document (delete + add).
    ///
    /// # Arguments
    /// * `id` - Memory ID to update
    /// * `content` - New text content
    /// * `metadata` - Optional new JSON metadata
    pub fn update(&self, id: &str, content: &str, metadata: Option<&str>) -> RookResult<()> {
        self.delete(id)?;
        self.add(id, content, metadata)
    }

    /// Delete a memory document by ID.
    ///
    /// # Note
    /// Call `commit()` after deleting to persist changes.
    pub fn delete(&self, id: &str) -> RookResult<()> {
        let writer = self
            .writer
            .lock()
            .map_err(|_| RookError::Configuration("Failed to acquire writer lock".to_string()))?;

        // Create term for exact ID match
        let id_term = Term::from_field_text(self.id_field, id);
        writer.delete_term(id_term);

        Ok(())
    }

    /// Commit pending changes to make them searchable.
    ///
    /// # Important
    /// Documents are not searchable until commit() is called.
    /// Batch multiple add/delete operations before committing for efficiency.
    pub fn commit(&self) -> RookResult<()> {
        let mut writer = self
            .writer
            .lock()
            .map_err(|_| RookError::Configuration("Failed to acquire writer lock".to_string()))?;

        writer
            .commit()
            .map_err(|e| RookError::Configuration(format!("Failed to commit: {}", e)))?;

        // Reload reader to see committed changes
        self.reader
            .reload()
            .map_err(|e| RookError::Configuration(format!("Failed to reload reader: {}", e)))?;

        Ok(())
    }

    /// Get the number of indexed documents.
    pub fn num_docs(&self) -> u64 {
        self.reader.searcher().num_docs()
    }

    /// Search for memories matching a text query.
    ///
    /// # Arguments
    /// * `query_text` - The search query (supports AND, OR, phrases in quotes)
    /// * `limit` - Maximum number of results to return
    ///
    /// # Returns
    /// Vector of TextSearchResult with BM25 and normalized scores, sorted by score descending.
    ///
    /// # Query Syntax
    /// - Single terms: `rust` matches documents containing "rust"
    /// - Phrases: `"hello world"` matches exact phrase
    /// - AND: `rust AND memory` matches documents with both terms
    /// - OR: `rust OR python` matches documents with either term
    /// - Prefix: `mem*` matches "memory", "memo", etc.
    pub fn search(&self, query_text: &str, limit: usize) -> RookResult<Vec<TextSearchResult>> {
        if query_text.trim().is_empty() {
            return Ok(Vec::new());
        }

        let searcher = self.reader.searcher();

        // Parse query with content field as default
        let query_parser = QueryParser::for_index(&self.index, vec![self.content_field]);

        let query = query_parser.parse_query(query_text).map_err(|e| {
            RookError::Configuration(format!("Failed to parse query '{}': {}", query_text, e))
        })?;

        // Execute search
        let top_docs = searcher
            .search(&query, &TopDocs::with_limit(limit))
            .map_err(|e| RookError::Configuration(format!("Search failed: {}", e)))?;

        if top_docs.is_empty() {
            return Ok(Vec::new());
        }

        // Collect results
        let mut results: Vec<TextSearchResult> = top_docs
            .into_iter()
            .filter_map(|(score, doc_address)| {
                let doc: TantivyDocument = searcher.doc(doc_address).ok()?;
                let id = doc.get_first(self.id_field)?.as_str()?.to_string();

                Some(TextSearchResult {
                    id,
                    bm25_score: score,
                    normalized_score: 0.0, // Will be set below
                })
            })
            .collect();

        // Normalize scores to 0-1 range using min-max normalization
        Self::normalize_scores(&mut results);

        Ok(results)
    }

    /// Normalize BM25 scores to 0-1 range using min-max normalization.
    fn normalize_scores(results: &mut [TextSearchResult]) {
        if results.is_empty() {
            return;
        }

        let max_score = results
            .iter()
            .map(|r| r.bm25_score)
            .fold(f32::NEG_INFINITY, f32::max);
        let min_score = results
            .iter()
            .map(|r| r.bm25_score)
            .fold(f32::INFINITY, f32::min);

        let range = max_score - min_score;

        if range > f32::EPSILON {
            for r in results.iter_mut() {
                r.normalized_score = (r.bm25_score - min_score) / range;
            }
        } else {
            // All scores equal, normalize to 1.0
            for r in results.iter_mut() {
                r.normalized_score = 1.0;
            }
        }
    }

    /// Search and return only IDs (useful for fusion).
    pub fn search_ids(&self, query_text: &str, limit: usize) -> RookResult<Vec<String>> {
        let results = self.search(query_text, limit)?;
        Ok(results.into_iter().map(|r| r.id).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_and_search() {
        let searcher = TantivySearcher::in_memory().unwrap();

        searcher
            .add("mem1", "The quick brown fox jumps over the lazy dog", None)
            .unwrap();
        searcher
            .add("mem2", "A fast red fox runs through the forest", None)
            .unwrap();
        searcher
            .add("mem3", "The cat sleeps on the mat", None)
            .unwrap();
        searcher.commit().unwrap();

        let results = searcher.search("fox", 10).unwrap();
        assert_eq!(results.len(), 2);
        assert!(results.iter().any(|r| r.id == "mem1"));
        assert!(results.iter().any(|r| r.id == "mem2"));
    }

    #[test]
    fn test_phrase_search() {
        let searcher = TantivySearcher::in_memory().unwrap();

        searcher
            .add("mem1", "The quick brown fox jumps", None)
            .unwrap();
        searcher
            .add("mem2", "Quick fox brown jumps the", None)
            .unwrap();
        searcher.commit().unwrap();

        // Phrase search should match exact sequence
        let results = searcher.search("\"quick brown fox\"", 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "mem1");
    }

    #[test]
    fn test_delete_document() {
        let searcher = TantivySearcher::in_memory().unwrap();

        searcher.add("mem1", "fox in the forest", None).unwrap();
        searcher.add("mem2", "fox on the hill", None).unwrap();
        searcher.commit().unwrap();

        assert_eq!(searcher.num_docs(), 2);

        searcher.delete("mem1").unwrap();
        searcher.commit().unwrap();

        let results = searcher.search("fox", 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "mem2");
    }

    #[test]
    fn test_update_document() {
        let searcher = TantivySearcher::in_memory().unwrap();

        searcher
            .add("mem1", "original content about cats", None)
            .unwrap();
        searcher.commit().unwrap();

        let results = searcher.search("cats", 10).unwrap();
        assert_eq!(results.len(), 1);

        searcher
            .update("mem1", "updated content about dogs", None)
            .unwrap();
        searcher.commit().unwrap();

        let cats = searcher.search("cats", 10).unwrap();
        assert_eq!(cats.len(), 0);

        let dogs = searcher.search("dogs", 10).unwrap();
        assert_eq!(dogs.len(), 1);
        assert_eq!(dogs[0].id, "mem1");
    }

    #[test]
    fn test_score_normalization() {
        let searcher = TantivySearcher::in_memory().unwrap();

        // Add documents with varying relevance
        searcher.add("mem1", "fox fox fox fox fox", None).unwrap(); // High TF
        searcher.add("mem2", "fox in the forest", None).unwrap(); // Lower TF
        searcher.commit().unwrap();

        let results = searcher.search("fox", 10).unwrap();

        // All normalized scores should be 0-1
        for r in &results {
            assert!(
                r.normalized_score >= 0.0 && r.normalized_score <= 1.0,
                "Score {} not in [0,1]",
                r.normalized_score
            );
        }

        // Highest scorer should have normalized_score = 1.0
        let max_normalized = results
            .iter()
            .map(|r| r.normalized_score)
            .fold(0.0f32, f32::max);
        assert!((max_normalized - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_empty_query() {
        let searcher = TantivySearcher::in_memory().unwrap();
        searcher.add("mem1", "some content", None).unwrap();
        searcher.commit().unwrap();

        let results = searcher.search("", 10).unwrap();
        assert!(results.is_empty());

        let results = searcher.search("   ", 10).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_no_results() {
        let searcher = TantivySearcher::in_memory().unwrap();
        searcher.add("mem1", "cats and dogs", None).unwrap();
        searcher.commit().unwrap();

        let results = searcher.search("elephant", 10).unwrap();
        assert!(results.is_empty());
    }
}
