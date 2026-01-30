//! Extraction pipeline for processing content through appropriate extractors.

use std::sync::Arc;

use crate::error::{ExtractError, ExtractResult};
use crate::types::ExtractedContent;
use crate::Extractor;

/// Pipeline for extracting content using registered extractors.
///
/// Automatically routes content to the appropriate extractor based on MIME type.
pub struct ExtractionPipeline {
    extractors: Vec<Arc<dyn Extractor>>,
}

impl ExtractionPipeline {
    /// Create new empty pipeline.
    pub fn new() -> Self {
        Self {
            extractors: Vec::new(),
        }
    }

    /// Create pipeline with all available extractors.
    pub fn with_defaults() -> Self {
        Self {
            extractors: crate::ExtractorFactory::all(),
        }
    }

    /// Add an extractor to the pipeline.
    pub fn add_extractor(mut self, extractor: Arc<dyn Extractor>) -> Self {
        self.extractors.push(extractor);
        self
    }

    /// Extract content using the appropriate extractor for the MIME type.
    pub async fn extract(
        &self,
        content: &[u8],
        mime_type: &str,
    ) -> ExtractResult<ExtractedContent> {
        // Find matching extractor
        for extractor in &self.extractors {
            if extractor.supports(mime_type) {
                return extractor.extract(content).await;
            }
        }

        Err(ExtractError::UnsupportedType(mime_type.to_string()))
    }

    /// Check if pipeline can handle a given MIME type.
    pub fn supports(&self, mime_type: &str) -> bool {
        self.extractors.iter().any(|e| e.supports(mime_type))
    }

    /// List all supported MIME types.
    pub fn supported_types(&self) -> Vec<&str> {
        self.extractors
            .iter()
            .flat_map(|e| e.supported_types().iter().copied())
            .collect()
    }

    /// Get the number of registered extractors.
    pub fn len(&self) -> usize {
        self.extractors.len()
    }

    /// Check if the pipeline has no registered extractors.
    pub fn is_empty(&self) -> bool {
        self.extractors.is_empty()
    }
}

impl Default for ExtractionPipeline {
    fn default() -> Self {
        Self::with_defaults()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_creation() {
        let pipeline = ExtractionPipeline::with_defaults();

        #[cfg(feature = "pdf")]
        assert!(pipeline.supports("application/pdf"));

        #[cfg(feature = "docx")]
        assert!(pipeline.supports("application/docx"));
    }

    #[test]
    fn test_pipeline_unsupported_type() {
        let pipeline = ExtractionPipeline::new();
        assert!(!pipeline.supports("video/mp4"));
    }

    #[test]
    fn test_pipeline_empty() {
        let pipeline = ExtractionPipeline::new();
        assert!(pipeline.is_empty());
        assert_eq!(pipeline.len(), 0);
    }

    #[test]
    fn test_pipeline_with_defaults() {
        let pipeline = ExtractionPipeline::with_defaults();

        #[cfg(all(feature = "pdf", feature = "docx"))]
        assert_eq!(pipeline.len(), 2);

        #[cfg(all(feature = "pdf", not(feature = "docx")))]
        assert_eq!(pipeline.len(), 1);
    }

    #[tokio::test]
    async fn test_pipeline_unsupported_type_error() {
        let pipeline = ExtractionPipeline::new();
        let result = pipeline.extract(b"test", "video/mp4").await;
        assert!(matches!(result, Err(ExtractError::UnsupportedType(_))));
    }
}
