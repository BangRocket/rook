//! PDF content extraction using pdf-extract.

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
    /// (helps detect image-based PDFs that need OCR fallback).
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

    /// Extract text synchronously (called within spawn_blocking).
    fn extract_sync(content: Vec<u8>) -> Result<String, ExtractError> {
        pdf_extract::extract_text_from_mem(&content)
            .map_err(|e| ExtractError::Pdf(e.to_string()))
    }

    /// Estimate page count from content (heuristic based on page markers).
    fn estimate_page_count(text: &str) -> Option<usize> {
        // PDF page breaks often result in form feed characters
        let ff_count = text.matches('\x0c').count();
        if ff_count > 0 {
            Some(ff_count + 1)
        } else {
            None
        }
    }

    /// Split text into pages based on form feed characters.
    fn split_into_pages(text: &str) -> Vec<String> {
        text.split('\x0c')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect()
    }
}

#[async_trait]
impl Extractor for PdfExtractor {
    async fn extract(&self, content: &[u8]) -> ExtractResult<ExtractedContent> {
        let content = content.to_vec();
        let content_len = content.len();

        // Run synchronous PDF extraction in blocking task
        let text =
            tokio::task::spawn_blocking(move || Self::extract_sync(content)).await??;

        // Check for image-based PDF (minimal text extracted)
        if text.len() < self.min_text_length {
            return Err(ExtractError::ExtractionFailed(format!(
                "Extracted only {} chars (threshold: {}). PDF may be image-based and require OCR.",
                text.len(),
                self.min_text_length
            )));
        }

        // Build structure metadata
        let pages = Self::split_into_pages(&text);
        let page_count = if pages.len() > 1 {
            Some(pages.len())
        } else {
            Self::estimate_page_count(&text)
        };

        let structure = DocumentStructure {
            page_count,
            sections: Vec::new(), // Basic extraction doesn't preserve headings
            pages: if pages.len() > 1 { pages } else { Vec::new() },
        };

        // Clean up extracted text (normalize whitespace)
        let text = text
            .replace('\x0c', "\n\n") // Replace form feeds with paragraph breaks
            .lines()
            .map(|l| l.trim())
            .filter(|l| !l.is_empty())
            .collect::<Vec<_>>()
            .join("\n");

        let mut result = ExtractedContent::new(text, Modality::Pdf, ContentSource::Bytes);
        result = result.with_structure(structure);
        result = result.with_metadata("original_size", content_len);

        Ok(result)
    }

    fn supported_types(&self) -> &[&str] {
        &["application/pdf"]
    }

    fn name(&self) -> &str {
        "pdf-extract"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_pdf_extractor_creation() {
        let extractor = PdfExtractor::new();
        assert_eq!(extractor.name(), "pdf-extract");
        assert!(extractor.supports("application/pdf"));
        assert!(!extractor.supports("application/msword"));
    }

    #[tokio::test]
    async fn test_pdf_extractor_empty_content() {
        let extractor = PdfExtractor::new();
        let result = extractor.extract(&[]).await;
        assert!(result.is_err());
    }

    #[test]
    fn test_split_into_pages() {
        let text = "Page 1 content\x0cPage 2 content\x0cPage 3 content";
        let pages = PdfExtractor::split_into_pages(text);
        assert_eq!(pages.len(), 3);
        assert_eq!(pages[0], "Page 1 content");
        assert_eq!(pages[1], "Page 2 content");
        assert_eq!(pages[2], "Page 3 content");
    }

    #[test]
    fn test_estimate_page_count() {
        let text = "Page 1\x0cPage 2\x0cPage 3";
        assert_eq!(PdfExtractor::estimate_page_count(text), Some(3));

        let text = "No page breaks here";
        assert_eq!(PdfExtractor::estimate_page_count(text), None);
    }

    // Integration test with real PDF would go here
    // #[tokio::test]
    // async fn test_pdf_extraction_real_file() {
    //     let pdf_bytes = include_bytes!("../tests/fixtures/sample.pdf");
    //     let extractor = PdfExtractor::new();
    //     let result = extractor.extract(pdf_bytes).await.unwrap();
    //     assert!(!result.is_empty());
    //     assert_eq!(result.modality, Modality::Pdf);
    // }
}
