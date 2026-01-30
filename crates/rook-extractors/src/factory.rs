//! Factory for creating extractors.

use std::sync::Arc;

use crate::error::{ExtractError, ExtractResult};
use crate::Extractor;

#[cfg(feature = "pdf")]
use crate::PdfExtractor;

#[cfg(feature = "docx")]
use crate::DocxExtractor;

/// Factory for creating content extractors.
pub struct ExtractorFactory;

impl ExtractorFactory {
    /// Create a PDF extractor.
    #[cfg(feature = "pdf")]
    pub fn pdf() -> Arc<dyn Extractor> {
        Arc::new(PdfExtractor::new())
    }

    /// Create a PDF extractor with custom minimum text threshold.
    #[cfg(feature = "pdf")]
    pub fn pdf_with_threshold(min_text_length: usize) -> Arc<dyn Extractor> {
        Arc::new(PdfExtractor::with_min_text_length(min_text_length))
    }

    /// Create a DOCX extractor.
    #[cfg(feature = "docx")]
    pub fn docx() -> Arc<dyn Extractor> {
        Arc::new(DocxExtractor::new())
    }

    /// Create a DOCX extractor with custom configuration.
    #[cfg(feature = "docx")]
    pub fn docx_configured(preserve_tables: bool, extract_headings: bool) -> Arc<dyn Extractor> {
        Arc::new(
            DocxExtractor::new()
                .with_tables(preserve_tables)
                .with_headings(extract_headings),
        )
    }

    /// Create extractor for a given MIME type.
    pub fn for_mime_type(mime_type: &str) -> ExtractResult<Arc<dyn Extractor>> {
        match mime_type {
            #[cfg(feature = "pdf")]
            "application/pdf" => Ok(Self::pdf()),

            #[cfg(feature = "docx")]
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            | "application/docx" => Ok(Self::docx()),

            _ => Err(ExtractError::UnsupportedType(mime_type.to_string())),
        }
    }

    /// Get all available extractors.
    pub fn all() -> Vec<Arc<dyn Extractor>> {
        let mut extractors: Vec<Arc<dyn Extractor>> = Vec::new();

        #[cfg(feature = "pdf")]
        extractors.push(Self::pdf());

        #[cfg(feature = "docx")]
        extractors.push(Self::docx());

        extractors
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_factory_all_extractors() {
        let extractors = ExtractorFactory::all();

        #[cfg(all(feature = "pdf", feature = "docx"))]
        assert_eq!(extractors.len(), 2);

        #[cfg(all(feature = "pdf", not(feature = "docx")))]
        assert_eq!(extractors.len(), 1);

        #[cfg(all(feature = "docx", not(feature = "pdf")))]
        assert_eq!(extractors.len(), 1);
    }

    #[cfg(feature = "docx")]
    #[test]
    fn test_factory_docx() {
        let extractor = ExtractorFactory::docx();
        assert!(extractor.supports("application/docx"));
    }

    #[cfg(feature = "docx")]
    #[test]
    fn test_factory_for_mime_type_docx() {
        let extractor = ExtractorFactory::for_mime_type(
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        );
        assert!(extractor.is_ok());
    }
}
