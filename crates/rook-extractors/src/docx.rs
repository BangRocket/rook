//! DOCX content extraction using docx-rs.
//!
//! Extracts text from DOCX files including paragraphs, tables, and
//! basic structure information.

use crate::error::{ExtractError, ExtractResult};
use crate::types::{ContentSource, DocumentStructure, ExtractedContent, Modality};
use crate::Extractor;
use async_trait::async_trait;
use docx_rs::{DocumentChild, ParagraphChild, RunChild, TableChild, TableRowChild};

/// DOCX content extractor using docx-rs library.
///
/// Extracts text from DOCX files including paragraphs, tables, and
/// basic structure information. Wraps synchronous docx-rs calls in
/// spawn_blocking to avoid blocking the async runtime.
#[derive(Debug, Clone, Default)]
pub struct DocxExtractor {
    /// Whether to preserve table structure in output.
    preserve_tables: bool,
    /// Whether to extract headings as sections.
    extract_headings: bool,
}

impl DocxExtractor {
    /// Create new DOCX extractor with default settings.
    pub fn new() -> Self {
        Self {
            preserve_tables: true,
            extract_headings: true,
        }
    }

    /// Configure whether to preserve table structure.
    pub fn with_tables(mut self, preserve: bool) -> Self {
        self.preserve_tables = preserve;
        self
    }

    /// Configure whether to extract headings as sections.
    pub fn with_headings(mut self, extract: bool) -> Self {
        self.extract_headings = extract;
        self
    }

    /// Extract text synchronously (called within spawn_blocking).
    fn extract_sync(
        content: Vec<u8>,
        preserve_tables: bool,
        extract_headings: bool,
    ) -> Result<(String, Vec<String>), ExtractError> {
        let docx = docx_rs::read_docx(&content)
            .map_err(|e| ExtractError::Docx(format!("Failed to parse DOCX: {}", e)))?;

        let mut text_parts: Vec<String> = Vec::new();
        let mut headings: Vec<String> = Vec::new();

        // Process document children (paragraphs, tables, etc.)
        for child in docx.document.children {
            match child {
                DocumentChild::Paragraph(p) => {
                    let para_text = Self::extract_paragraph_text(&p);

                    // Check if this is a heading (by style)
                    if extract_headings {
                        if let Some(style) = &p.property.style {
                            let style_id = style.val.to_lowercase();
                            let is_heading = style_id.starts_with("heading")
                                || style_id.contains("title");
                            if is_heading && !para_text.trim().is_empty() {
                                headings.push(para_text.trim().to_string());
                            }
                        }
                    }

                    if !para_text.trim().is_empty() {
                        text_parts.push(para_text);
                    }
                }
                DocumentChild::Table(t) => {
                    if preserve_tables {
                        let table_text = Self::extract_table_text(&t);
                        if !table_text.trim().is_empty() {
                            text_parts.push(table_text);
                        }
                    } else {
                        // Just extract cell text without structure
                        for row in &t.rows {
                            let TableChild::TableRow(r) = row;
                            for cell in &r.cells {
                                let TableRowChild::TableCell(c) = cell;
                                for child in &c.children {
                                    if let docx_rs::TableCellContent::Paragraph(p) = child {
                                        let cell_text = Self::extract_paragraph_text(p);
                                        if !cell_text.trim().is_empty() {
                                            text_parts.push(cell_text);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                _ => {
                    // Skip other document children (bookmarks, etc.)
                }
            }
        }

        let text = text_parts.join("\n");
        Ok((text, headings))
    }

    /// Extract text from a paragraph.
    fn extract_paragraph_text(p: &docx_rs::Paragraph) -> String {
        let mut text = String::new();

        for child in &p.children {
            match child {
                ParagraphChild::Run(r) => {
                    for run_child in &r.children {
                        match run_child {
                            RunChild::Text(t) => {
                                text.push_str(&t.text);
                            }
                            RunChild::Tab(_) => {
                                text.push('\t');
                            }
                            RunChild::Break(_) => {
                                text.push('\n');
                            }
                            _ => {}
                        }
                    }
                }
                ParagraphChild::Hyperlink(h) => {
                    // Extract text from hyperlink runs
                    // Hyperlink children are ParagraphChild, so reuse same logic
                    for child in &h.children {
                        if let ParagraphChild::Run(r) = child {
                            for run_child in &r.children {
                                if let RunChild::Text(t) = run_child {
                                    text.push_str(&t.text);
                                }
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        text
    }

    /// Extract text from a table with structure.
    fn extract_table_text(t: &docx_rs::Table) -> String {
        let mut rows: Vec<Vec<String>> = Vec::new();

        for row in &t.rows {
            let TableChild::TableRow(r) = row;
            let mut cells: Vec<String> = Vec::new();
            for cell in &r.cells {
                let TableRowChild::TableCell(c) = cell;
                let mut cell_text = String::new();
                for child in &c.children {
                    if let docx_rs::TableCellContent::Paragraph(p) = child {
                        let para = Self::extract_paragraph_text(p);
                        if !cell_text.is_empty() && !para.is_empty() {
                            cell_text.push(' ');
                        }
                        cell_text.push_str(&para);
                    }
                }
                cells.push(cell_text.trim().to_string());
            }
            rows.push(cells);
        }

        // Format as a simple text table
        if rows.is_empty() {
            return String::new();
        }

        // Join cells with | and rows with newlines
        rows.iter()
            .map(|row| row.join(" | "))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

#[async_trait]
impl Extractor for DocxExtractor {
    async fn extract(&self, content: &[u8]) -> ExtractResult<ExtractedContent> {
        let content = content.to_vec();
        let content_len = content.len();
        let preserve_tables = self.preserve_tables;
        let extract_headings = self.extract_headings;

        // Run synchronous DOCX extraction in blocking task
        let (text, headings) = tokio::task::spawn_blocking(move || {
            Self::extract_sync(content, preserve_tables, extract_headings)
        })
        .await??;

        // Check for empty extraction
        if text.trim().is_empty() {
            return Err(ExtractError::EmptyContent);
        }

        // Build structure metadata
        let structure = DocumentStructure {
            page_count: None, // DOCX doesn't have inherent page structure
            sections: headings,
            pages: Vec::new(),
        };

        let mut result = ExtractedContent::new(text, Modality::Docx, ContentSource::Bytes);
        result = result.with_structure(structure);
        result = result.with_metadata("original_size", content_len);

        Ok(result)
    }

    fn supported_types(&self) -> &[&str] {
        &[
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/docx",
        ]
    }

    fn name(&self) -> &str {
        "docx-rs"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_docx_extractor_creation() {
        let extractor = DocxExtractor::new();
        assert_eq!(extractor.name(), "docx-rs");
        assert!(extractor
            .supports("application/vnd.openxmlformats-officedocument.wordprocessingml.document"));
        assert!(extractor.supports("application/docx"));
        assert!(!extractor.supports("application/pdf"));
    }

    #[tokio::test]
    async fn test_docx_extractor_empty_content() {
        let extractor = DocxExtractor::new();
        let result = extractor.extract(&[]).await;
        assert!(result.is_err());
    }

    #[test]
    fn test_docx_extractor_configuration() {
        let extractor = DocxExtractor::new().with_tables(false).with_headings(true);

        assert!(!extractor.preserve_tables);
        assert!(extractor.extract_headings);
    }
}
