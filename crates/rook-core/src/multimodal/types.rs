//! Types for multimodal content ingestion.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Source provenance - tracks where memory content originated
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceProvenance {
    /// Original content modality (pdf, docx, image, text)
    pub modality: String,

    /// Original file name if known
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filename: Option<String>,

    /// Original file size in bytes
    #[serde(skip_serializing_if = "Option::is_none")]
    pub original_size: Option<usize>,

    /// MIME type of original content
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,

    /// Extraction method used (pdf-extract, docx-rs, ocr, vision_llm)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extraction_method: Option<String>,

    /// Page number for paginated documents
    #[serde(skip_serializing_if = "Option::is_none")]
    pub page_number: Option<usize>,

    /// Section/heading if structure was preserved
    #[serde(skip_serializing_if = "Option::is_none")]
    pub section: Option<String>,

    /// Timestamp of extraction
    pub extracted_at: chrono::DateTime<chrono::Utc>,
}

impl SourceProvenance {
    /// Create new provenance record
    pub fn new(modality: impl Into<String>) -> Self {
        Self {
            modality: modality.into(),
            filename: None,
            original_size: None,
            mime_type: None,
            extraction_method: None,
            page_number: None,
            section: None,
            extracted_at: chrono::Utc::now(),
        }
    }

    /// Set original filename
    pub fn with_filename(mut self, filename: impl Into<String>) -> Self {
        self.filename = Some(filename.into());
        self
    }

    /// Set original size
    pub fn with_size(mut self, size: usize) -> Self {
        self.original_size = Some(size);
        self
    }

    /// Set MIME type
    pub fn with_mime_type(mut self, mime_type: impl Into<String>) -> Self {
        self.mime_type = Some(mime_type.into());
        self
    }

    /// Set extraction method
    pub fn with_method(mut self, method: impl Into<String>) -> Self {
        self.extraction_method = Some(method.into());
        self
    }

    /// Set page number
    pub fn with_page(mut self, page: usize) -> Self {
        self.page_number = Some(page);
        self
    }

    /// Set section
    pub fn with_section(mut self, section: impl Into<String>) -> Self {
        self.section = Some(section.into());
        self
    }

    /// Convert to metadata HashMap for memory storage
    pub fn to_metadata(&self) -> HashMap<String, serde_json::Value> {
        let mut metadata = HashMap::new();

        metadata.insert(
            "source_modality".to_string(),
            serde_json::Value::String(self.modality.clone()),
        );

        if let Some(ref filename) = self.filename {
            metadata.insert(
                "source_filename".to_string(),
                serde_json::Value::String(filename.clone()),
            );
        }

        if let Some(size) = self.original_size {
            metadata.insert(
                "source_size".to_string(),
                serde_json::Value::Number(size.into()),
            );
        }

        if let Some(ref mime) = self.mime_type {
            metadata.insert(
                "source_mime_type".to_string(),
                serde_json::Value::String(mime.clone()),
            );
        }

        if let Some(ref method) = self.extraction_method {
            metadata.insert(
                "extraction_method".to_string(),
                serde_json::Value::String(method.clone()),
            );
        }

        if let Some(page) = self.page_number {
            metadata.insert(
                "source_page".to_string(),
                serde_json::Value::Number(page.into()),
            );
        }

        if let Some(ref section) = self.section {
            metadata.insert(
                "source_section".to_string(),
                serde_json::Value::String(section.clone()),
            );
        }

        metadata.insert(
            "extracted_at".to_string(),
            serde_json::Value::String(self.extracted_at.to_rfc3339()),
        );

        metadata
    }
}

/// Result of multimodal ingestion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultimodalIngestResult {
    /// IDs of created memories
    pub memory_ids: Vec<String>,

    /// Source provenance for tracking
    pub provenance: SourceProvenance,

    /// Number of chunks/pages created
    pub chunks_created: usize,

    /// Total extracted text length
    pub text_length: usize,

    /// Whether extraction required fallback (e.g., vision for image-PDF)
    pub used_fallback: bool,

    /// Warnings/notes from extraction
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub warnings: Vec<String>,
}

/// Configuration for multimodal ingestion
#[derive(Debug, Clone)]
pub struct MultimodalConfig {
    /// Maximum chunk size for splitting large documents (in characters)
    pub max_chunk_size: usize,

    /// Overlap between chunks (in characters)
    pub chunk_overlap: usize,

    /// Whether to create separate memories per page (for PDFs)
    pub split_by_page: bool,

    /// Whether to preserve document structure in metadata
    pub preserve_structure: bool,

    /// Minimum extracted text length to consider successful
    pub min_text_length: usize,
}

impl Default for MultimodalConfig {
    fn default() -> Self {
        Self {
            max_chunk_size: 2000,
            chunk_overlap: 200,
            split_by_page: false,
            preserve_structure: true,
            min_text_length: 10,
        }
    }
}

impl MultimodalConfig {
    /// Create config with page-level splitting for PDFs
    pub fn with_page_splitting() -> Self {
        Self {
            split_by_page: true,
            ..Default::default()
        }
    }

    /// Create config with larger chunks for context
    pub fn with_large_chunks() -> Self {
        Self {
            max_chunk_size: 4000,
            chunk_overlap: 400,
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_source_provenance_creation() {
        let prov = SourceProvenance::new("pdf")
            .with_filename("test.pdf")
            .with_size(1024)
            .with_mime_type("application/pdf")
            .with_method("pdf-extract");

        assert_eq!(prov.modality, "pdf");
        assert_eq!(prov.filename, Some("test.pdf".to_string()));
        assert_eq!(prov.original_size, Some(1024));
    }

    #[test]
    fn test_provenance_to_metadata() {
        let prov = SourceProvenance::new("pdf")
            .with_filename("test.pdf")
            .with_page(1);

        let metadata = prov.to_metadata();

        assert_eq!(
            metadata.get("source_modality"),
            Some(&serde_json::Value::String("pdf".to_string()))
        );
        assert_eq!(
            metadata.get("source_filename"),
            Some(&serde_json::Value::String("test.pdf".to_string()))
        );
        assert_eq!(
            metadata.get("source_page"),
            Some(&serde_json::Value::Number(1.into()))
        );
    }

    #[test]
    fn test_config_presets() {
        let page_config = MultimodalConfig::with_page_splitting();
        assert!(page_config.split_by_page);

        let large_config = MultimodalConfig::with_large_chunks();
        assert_eq!(large_config.max_chunk_size, 4000);
    }
}
