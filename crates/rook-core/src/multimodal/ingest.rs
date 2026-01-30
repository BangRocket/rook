//! Multimodal content ingestion orchestrator.
//!
//! Coordinates extraction, chunking, and memory storage for
//! documents and images.

use crate::error::{RookError, RookResult};
use crate::memory::Memory;
use crate::multimodal::types::{MultimodalConfig, MultimodalIngestResult, SourceProvenance};
use rook_extractors::{ExtractedContent, ExtractionPipeline, Modality};
use std::collections::HashMap;

/// Orchestrates multimodal content ingestion.
///
/// Workflow:
/// 1. Detect content type (MIME type)
/// 2. Extract text via appropriate extractor
/// 3. Optionally chunk large documents
/// 4. Store as memory with provenance metadata
///
/// # Cross-Modal Retrieval
///
/// When using multimodal ingestion, memories from different modalities
/// (PDF, DOCX, images) are all stored as text with metadata tracking
/// their source. This means:
///
/// - Queries find relevant content regardless of original format
/// - Results include `source_modality` in metadata for format awareness
/// - No special handling needed - just search normally
///
/// # Example
///
/// ```ignore
/// use rook_core::multimodal::MultimodalIngester;
///
/// let ingester = MultimodalIngester::new();
/// let pdf_bytes = std::fs::read("document.pdf")?;
///
/// let result = ingester.ingest(
///     &memory,
///     &pdf_bytes,
///     "application/pdf",
///     Some("document.pdf"),
///     "user_123",
///     None,
/// ).await?;
///
/// println!("Created {} memories from PDF", result.memory_ids.len());
/// ```
pub struct MultimodalIngester {
    pipeline: ExtractionPipeline,
    config: MultimodalConfig,
}

impl MultimodalIngester {
    /// Create new ingester with default configuration
    pub fn new() -> Self {
        Self {
            pipeline: ExtractionPipeline::with_defaults(),
            config: MultimodalConfig::default(),
        }
    }

    /// Create ingester with custom configuration
    pub fn with_config(config: MultimodalConfig) -> Self {
        Self {
            pipeline: ExtractionPipeline::with_defaults(),
            config,
        }
    }

    /// Create ingester with custom pipeline
    pub fn with_pipeline(pipeline: ExtractionPipeline, config: MultimodalConfig) -> Self {
        Self { pipeline, config }
    }

    /// Ingest content into memory system
    ///
    /// # Arguments
    /// * `memory` - Memory instance for storage
    /// * `content` - Raw bytes of the content
    /// * `mime_type` - MIME type (e.g., "application/pdf", "image/png")
    /// * `filename` - Optional original filename
    /// * `user_id` - User ID for memory scoping
    /// * `additional_metadata` - Extra metadata to include
    pub async fn ingest(
        &self,
        memory: &Memory,
        content: &[u8],
        mime_type: &str,
        filename: Option<&str>,
        user_id: &str,
        additional_metadata: Option<HashMap<String, serde_json::Value>>,
    ) -> RookResult<MultimodalIngestResult> {
        let content_len = content.len();
        let mut warnings = Vec::new();

        // Extract content
        let extracted = self
            .pipeline
            .extract(content, mime_type)
            .await
            .map_err(|e| RookError::extraction(format!("Failed to extract {}: {}", mime_type, e)))?;

        // Check minimum text length
        if extracted.text.len() < self.config.min_text_length {
            return Err(RookError::extraction(format!(
                "Extracted text too short ({} chars, minimum: {})",
                extracted.text.len(),
                self.config.min_text_length
            )));
        }

        // Build provenance
        let modality_str = match &extracted.modality {
            Modality::Text => "text",
            Modality::Pdf => "pdf",
            Modality::Docx => "docx",
            Modality::Image { format } => format.as_str(),
        };

        let mut provenance = SourceProvenance::new(modality_str)
            .with_size(content_len)
            .with_mime_type(mime_type);

        if let Some(name) = filename {
            provenance = provenance.with_filename(name);
        }

        // Get extraction method from metadata if available
        if let Some(method) = extracted.metadata.get("extraction_method") {
            if let Some(method_str) = method.as_str() {
                provenance = provenance.with_method(method_str);
            }
        }

        // Determine chunking strategy
        let chunks = self.chunk_content(&extracted, &mut warnings);

        // Store memories
        let mut memory_ids = Vec::new();
        let text_length = extracted.text.len();

        for (i, chunk) in chunks.iter().enumerate() {
            // Build metadata for this chunk
            let mut metadata = provenance.to_metadata();

            // Add chunk info if multiple chunks
            if chunks.len() > 1 {
                metadata.insert(
                    "chunk_index".to_string(),
                    serde_json::Value::Number(i.into()),
                );
                metadata.insert(
                    "chunk_total".to_string(),
                    serde_json::Value::Number(chunks.len().into()),
                );
            }

            // Add page info if available
            if let Some(page) = chunk.page_number {
                metadata.insert(
                    "source_page".to_string(),
                    serde_json::Value::Number(page.into()),
                );
            }

            // Merge additional metadata
            if let Some(ref extra) = additional_metadata {
                for (k, v) in extra {
                    metadata.insert(k.clone(), v.clone());
                }
            }

            // Add memory via Memory API
            // Use infer=false since we're ingesting raw extracted content
            let result = memory
                .add(
                    &chunk.text,
                    Some(user_id.to_string()),
                    None, // agent_id
                    None, // run_id
                    Some(metadata),
                    false, // infer - don't extract facts, just store raw text
                    None,  // memory_type
                )
                .await?;

            // Extract memory IDs from result
            for mem_result in &result.results {
                memory_ids.push(mem_result.id.clone());
            }
        }

        // Check for fallback usage
        let used_fallback = extracted
            .metadata
            .get("extraction_method")
            .and_then(|v| v.as_str())
            .map(|m| m == "vision_llm" || m == "combined")
            .unwrap_or(false);

        Ok(MultimodalIngestResult {
            memory_ids,
            provenance,
            chunks_created: chunks.len(),
            text_length,
            used_fallback,
            warnings,
        })
    }

    /// Chunk extracted content based on configuration
    fn chunk_content(
        &self,
        extracted: &ExtractedContent,
        warnings: &mut Vec<String>,
    ) -> Vec<ContentChunk> {
        let mut chunks = Vec::new();

        // Check if we should split by pages
        if self.config.split_by_page {
            if let Some(ref structure) = extracted.structure {
                if !structure.pages.is_empty() {
                    // Create chunk per page
                    for (i, page_text) in structure.pages.iter().enumerate() {
                        if !page_text.trim().is_empty() {
                            chunks.push(ContentChunk {
                                text: page_text.clone(),
                                page_number: Some(i + 1),
                            });
                        }
                    }

                    if !chunks.is_empty() {
                        return chunks;
                    }
                }
            }
            warnings.push("Page splitting requested but no page structure available".to_string());
        }

        // Fall back to character-based chunking
        let text = &extracted.text;

        if text.len() <= self.config.max_chunk_size {
            // Small enough to be single chunk
            chunks.push(ContentChunk {
                text: text.clone(),
                page_number: None,
            });
        } else {
            // Split into overlapping chunks
            let mut start = 0;
            let mut chunk_num = 0;

            while start < text.len() {
                let end = std::cmp::min(start + self.config.max_chunk_size, text.len());

                // Try to break at sentence boundary
                let chunk_end = self.find_sentence_boundary(text, start, end);

                let chunk_text = text[start..chunk_end].trim().to_string();
                if !chunk_text.is_empty() {
                    chunks.push(ContentChunk {
                        text: chunk_text,
                        page_number: None,
                    });
                    chunk_num += 1;
                }

                // Move start with overlap
                if chunk_end >= text.len() {
                    break;
                }
                start = chunk_end.saturating_sub(self.config.chunk_overlap);
            }

            if chunk_num > 10 {
                warnings.push(format!("Large document split into {} chunks", chunk_num));
            }
        }

        chunks
    }

    /// Find a good sentence boundary near the target end position
    fn find_sentence_boundary(&self, text: &str, start: usize, target_end: usize) -> usize {
        // Look for sentence-ending punctuation within the last 20% of the chunk
        let search_start = start + ((target_end - start) * 80 / 100);
        let search_region = &text[search_start..target_end];

        // Find last sentence boundary
        for (i, c) in search_region.char_indices().rev() {
            if c == '.' || c == '!' || c == '?' || c == '\n' {
                // Check if followed by whitespace (actual end of sentence)
                let pos = search_start + i + c.len_utf8();
                if pos < text.len() {
                    let next = text[pos..].chars().next();
                    if next.map(|c| c.is_whitespace()).unwrap_or(true) {
                        return pos;
                    }
                }
            }
        }

        // No good boundary found, use target end
        target_end
    }

    /// Check if ingester supports a given MIME type
    pub fn supports(&self, mime_type: &str) -> bool {
        self.pipeline.supports(mime_type)
    }

    /// Get list of supported MIME types
    pub fn supported_types(&self) -> Vec<&str> {
        self.pipeline.supported_types()
    }
}

impl Default for MultimodalIngester {
    fn default() -> Self {
        Self::new()
    }
}

/// Internal chunk representation
#[derive(Debug)]
struct ContentChunk {
    text: String,
    page_number: Option<usize>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ingester_creation() {
        let ingester = MultimodalIngester::new();
        // Check that it supports expected types based on enabled features
        #[cfg(feature = "pdf")]
        assert!(ingester.supports("application/pdf"));
    }

    #[test]
    fn test_sentence_boundary() {
        let ingester = MultimodalIngester::new();
        let text = "First sentence. Second sentence. Third sentence.";

        // Should find boundary at period
        let boundary = ingester.find_sentence_boundary(text, 0, 35);
        assert!(
            text[..boundary].ends_with(". ") || text[..boundary].ends_with('.'),
            "Boundary should be at sentence end"
        );
    }

    #[test]
    fn test_sentence_boundary_no_punctuation() {
        let ingester = MultimodalIngester::new();
        let text = "This is a long text without any punctuation in the region";

        // Should return target_end if no boundary found
        let boundary = ingester.find_sentence_boundary(text, 0, 30);
        assert_eq!(boundary, 30);
    }

    #[test]
    fn test_config_presets() {
        let page_config = MultimodalConfig::with_page_splitting();
        assert!(page_config.split_by_page);

        let large_config = MultimodalConfig::with_large_chunks();
        assert_eq!(large_config.max_chunk_size, 4000);
    }

    #[test]
    fn test_default_config() {
        let config = MultimodalConfig::default();
        assert_eq!(config.max_chunk_size, 2000);
        assert_eq!(config.chunk_overlap, 200);
        assert!(!config.split_by_page);
        assert!(config.preserve_structure);
        assert_eq!(config.min_text_length, 10);
    }
}
