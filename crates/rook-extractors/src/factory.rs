//! Factory for creating extractors.

use std::sync::Arc;

use crate::error::{ExtractError, ExtractResult};
use crate::Extractor;

#[cfg(feature = "pdf")]
use crate::PdfExtractor;

#[cfg(feature = "docx")]
use crate::DocxExtractor;

#[cfg(feature = "image")]
use crate::image::{ImageExtractionConfig, ImageExtractor};

#[cfg(feature = "vision")]
use crate::vision::{VisionConfig, VisionLlmExtractor};

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

    /// Create an image extractor (OCR + Vision combined).
    #[cfg(feature = "image")]
    pub fn image() -> Arc<dyn Extractor> {
        Arc::new(ImageExtractor::new())
    }

    /// Create an image extractor with custom configuration.
    #[cfg(feature = "image")]
    pub fn image_with_config(config: ImageExtractionConfig) -> Arc<dyn Extractor> {
        Arc::new(ImageExtractor::with_config(config))
    }

    /// Create a vision-only extractor (no OCR).
    #[cfg(feature = "vision")]
    pub fn vision() -> Arc<dyn Extractor> {
        Arc::new(VisionLlmExtractor::new())
    }

    /// Create a vision extractor with custom configuration.
    #[cfg(feature = "vision")]
    pub fn vision_with_config(config: VisionConfig) -> Arc<dyn Extractor> {
        Arc::new(VisionLlmExtractor::with_config(config))
    }

    /// Create extractor for a given MIME type.
    pub fn for_mime_type(mime_type: &str) -> ExtractResult<Arc<dyn Extractor>> {
        match mime_type {
            #[cfg(feature = "pdf")]
            "application/pdf" => Ok(Self::pdf()),

            #[cfg(feature = "docx")]
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            | "application/docx" => Ok(Self::docx()),

            #[cfg(feature = "image")]
            "image/png" | "image/jpeg" | "image/gif" | "image/webp" => Ok(Self::image()),

            _ => Err(ExtractError::UnsupportedType(mime_type.to_string())),
        }
    }

    /// Get all available extractors.
    #[allow(clippy::vec_init_then_push)]
    pub fn all() -> Vec<Arc<dyn Extractor>> {
        let mut extractors: Vec<Arc<dyn Extractor>> = Vec::new();

        #[cfg(feature = "pdf")]
        extractors.push(Self::pdf());

        #[cfg(feature = "docx")]
        extractors.push(Self::docx());

        #[cfg(feature = "image")]
        extractors.push(Self::image());

        extractors
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_factory_all_extractors() {
        let extractors = ExtractorFactory::all();

        #[cfg(all(feature = "pdf", feature = "docx", not(feature = "image")))]
        assert_eq!(extractors.len(), 2);

        #[cfg(all(feature = "pdf", feature = "docx", feature = "image"))]
        assert_eq!(extractors.len(), 3);

        #[cfg(all(feature = "pdf", not(feature = "docx"), not(feature = "image")))]
        assert_eq!(extractors.len(), 1);

        #[cfg(all(feature = "docx", not(feature = "pdf"), not(feature = "image")))]
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

    #[cfg(feature = "image")]
    #[test]
    fn test_factory_image() {
        let extractor = ExtractorFactory::image();
        assert!(extractor.supports("image/png"));
        assert!(extractor.supports("image/jpeg"));
    }

    #[cfg(feature = "image")]
    #[test]
    fn test_factory_for_mime_type_image() {
        let png = ExtractorFactory::for_mime_type("image/png");
        assert!(png.is_ok());

        let jpeg = ExtractorFactory::for_mime_type("image/jpeg");
        assert!(jpeg.is_ok());
    }

    #[cfg(feature = "vision")]
    #[test]
    fn test_factory_vision() {
        let extractor = ExtractorFactory::vision();
        assert!(extractor.supports("image/png"));
    }
}
