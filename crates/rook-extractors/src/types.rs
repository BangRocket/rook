//! Core types for content extraction.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Modality of original content.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Modality {
    /// Plain text content.
    Text,
    /// PDF document.
    Pdf,
    /// Microsoft Word document.
    Docx,
    /// Image with specified format.
    Image {
        /// Image format (e.g., "png", "jpeg").
        format: String,
    },
    // Future: Audio, Video (v2)
}

impl Default for Modality {
    fn default() -> Self {
        Modality::Text
    }
}

/// Source reference for original content.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContentSource {
    /// Content provided as bytes (no file reference).
    Bytes,
    /// Content from file path.
    Path(String),
    /// Content from URL.
    Url(String),
}

/// Document structure metadata (optional, for structured documents).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DocumentStructure {
    /// Total page count (for PDFs).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub page_count: Option<usize>,

    /// Extracted headings/sections.
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub sections: Vec<String>,

    /// Per-page text (for page-level retrieval).
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub pages: Vec<String>,
}

/// Extracted content with metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedContent {
    /// Extracted text for embedding/search.
    pub text: String,

    /// Original content modality.
    pub modality: Modality,

    /// Document structure (if preserved).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub structure: Option<DocumentStructure>,

    /// Reference to original content.
    pub source: ContentSource,

    /// Additional metadata (format-specific).
    #[serde(skip_serializing_if = "HashMap::is_empty", default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

impl ExtractedContent {
    /// Create new extracted content.
    pub fn new(text: String, modality: Modality, source: ContentSource) -> Self {
        Self {
            text,
            modality,
            structure: None,
            source,
            metadata: HashMap::new(),
        }
    }

    /// Add structure information.
    pub fn with_structure(mut self, structure: DocumentStructure) -> Self {
        self.structure = Some(structure);
        self
    }

    /// Add metadata entry.
    pub fn with_metadata(
        mut self,
        key: impl Into<String>,
        value: impl Into<serde_json::Value>,
    ) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Check if extraction produced meaningful content.
    pub fn is_empty(&self) -> bool {
        self.text.trim().is_empty()
    }

    /// Get content length.
    pub fn len(&self) -> usize {
        self.text.len()
    }
}
