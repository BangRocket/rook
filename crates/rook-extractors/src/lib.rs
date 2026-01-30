//! rook-extractors - Content extraction for multimodal memory ingestion.
//!
//! Provides extractors for PDF, DOCX, and image content with a unified
//! trait-based interface following codebase patterns.

mod error;
mod factory;
mod types;

#[cfg(feature = "pdf")]
mod pdf;

#[cfg(feature = "docx")]
mod docx;

pub use error::{ExtractError, ExtractResult};
pub use factory::ExtractorFactory;
pub use types::{ContentSource, DocumentStructure, ExtractedContent, Modality};

#[cfg(feature = "pdf")]
pub use pdf::PdfExtractor;

#[cfg(feature = "docx")]
pub use docx::DocxExtractor;

use async_trait::async_trait;

/// Core Extractor trait - all content extractors implement this.
///
/// Similar pattern to Embedder/Llm traits in the codebase.
#[async_trait]
pub trait Extractor: Send + Sync {
    /// Extract text content from bytes.
    async fn extract(&self, content: &[u8]) -> ExtractResult<ExtractedContent>;

    /// Supported MIME types for this extractor.
    fn supported_types(&self) -> &[&str];

    /// Check if this extractor handles the given MIME type.
    fn supports(&self, mime_type: &str) -> bool {
        self.supported_types().contains(&mime_type)
    }

    /// Human-readable name for this extractor.
    fn name(&self) -> &str;
}
