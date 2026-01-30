//! rook-extractors - Content extraction for multimodal memory ingestion.
//!
//! Provides extractors for PDF, DOCX, and image content with a unified
//! trait-based interface following codebase patterns.
//!
//! # Features
//!
//! - `pdf` (default) - PDF text extraction via pdf-extract
//! - `docx` (default) - DOCX text extraction via docx-rs
//! - `image` - Image format detection and loading
//! - `ocr` - Image OCR via tesseract (requires tesseract installed)
//! - `vision` - Image description via vision LLM (GPT-4o)
//! - `full` - All extraction features
//!
//! # Example
//!
//! ```ignore
//! use rook_extractors::{ExtractionPipeline, ExtractorFactory};
//!
//! // Use pipeline for automatic MIME type routing
//! let pipeline = ExtractionPipeline::with_defaults();
//! let result = pipeline.extract(&pdf_bytes, "application/pdf").await?;
//!
//! // Document extractors
//! let pdf = ExtractorFactory::pdf();
//! let docx = ExtractorFactory::docx();
//!
//! // Image extraction (OCR + Vision combined)
//! let image = ExtractorFactory::image();
//! let result = image.extract(&png_bytes).await?;
//! ```

mod error;
mod factory;
mod pipeline;
mod types;

#[cfg(feature = "pdf")]
mod pdf;

#[cfg(feature = "docx")]
mod docx;

#[cfg(feature = "image")]
pub mod image;

#[cfg(feature = "vision")]
pub mod vision;

pub use error::{ExtractError, ExtractResult};
pub use factory::ExtractorFactory;
pub use pipeline::ExtractionPipeline;
pub use types::{ContentSource, DocumentStructure, ExtractedContent, Modality};

#[cfg(feature = "pdf")]
pub use pdf::PdfExtractor;

#[cfg(feature = "docx")]
pub use docx::DocxExtractor;

#[cfg(feature = "image")]
pub use image::{ImageExtractionConfig, ImageExtractor};

#[cfg(feature = "vision")]
pub use vision::{VisionConfig, VisionLlmExtractor};

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
