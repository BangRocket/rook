//! Multimodal content ingestion orchestrator.
//!
//! Coordinates extraction, chunking, and memory storage for
//! documents and images.

use crate::error::RookResult;
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
