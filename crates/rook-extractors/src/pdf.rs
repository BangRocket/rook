//! PDF content extraction using pdf-extract.
//!
//! Implementation will be added in Task 2.

use crate::error::{ExtractError, ExtractResult};
use crate::types::{ContentSource, DocumentStructure, ExtractedContent, Modality};
use crate::Extractor;
use async_trait::async_trait;

/// PDF content extractor using pdf-extract library.
///
/// Extracts text from PDF files, wrapping synchronous pdf-extract
/// calls in spawn_blocking to avoid blocking the async runtime.
#[derive(Debug, Clone, Default)]
pub struct PdfExtractor {
    /// Minimum text length to consider extraction successful
    /// (helps detect image-based PDFs that need OCR fallback)
    min_text_length: usize,
}

impl PdfExtractor {
    /// Create new PDF extractor with default settings.
    pub fn new() -> Self {
        Self {
            min_text_length: 10,
        }
    }

    /// Create PDF extractor with custom minimum text threshold.
    pub fn with_min_text_length(min_text_length: usize) -> Self {
        Self { min_text_length }
    }
}

#[async_trait]
impl Extractor for PdfExtractor {
    async fn extract(&self, _content: &[u8]) -> ExtractResult<ExtractedContent> {
        // Stub implementation - will be completed in Task 2
        todo!("PDF extraction implementation")
    }

    fn supported_types(&self) -> &[&str] {
        &["application/pdf"]
    }

    fn name(&self) -> &str {
        "pdf-extract"
    }
}
