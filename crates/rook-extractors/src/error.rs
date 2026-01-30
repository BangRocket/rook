//! Extraction error types.

use thiserror::Error;

/// Errors that can occur during content extraction.
#[derive(Error, Debug)]
pub enum ExtractError {
    /// Content type is not supported by any extractor.
    #[error("Unsupported content type: {0}")]
    UnsupportedType(String),

    /// Extraction process failed.
    #[error("Extraction failed: {0}")]
    ExtractionFailed(String),

    /// Extracted content is empty.
    #[error("Empty content extracted")]
    EmptyContent,

    /// IO error during extraction.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// PDF-specific extraction error.
    #[cfg(feature = "pdf")]
    #[error("PDF extraction error: {0}")]
    Pdf(String),

    /// DOCX-specific extraction error.
    #[cfg(feature = "docx")]
    #[error("DOCX extraction error: {0}")]
    Docx(String),

    /// Task join error from spawn_blocking.
    #[error("Task join error: {0}")]
    TaskJoin(#[from] tokio::task::JoinError),
}

/// Result type for extraction operations.
pub type ExtractResult<T> = Result<T, ExtractError>;
