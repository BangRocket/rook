//! Image content extraction with OCR and vision LLM strategies.
//!
//! Provides ImageExtractor that combines OCR (for text-heavy images)
//! and vision LLM (for scene understanding) with configurable fallback.

use crate::error::{ExtractError, ExtractResult};
use crate::types::{ContentSource, ExtractedContent, Modality};
use crate::Extractor;
use async_trait::async_trait;

#[cfg(feature = "vision")]
use crate::vision::VisionLlmExtractor;

/// Configuration for image extraction strategy.
#[derive(Debug, Clone)]
pub struct ImageExtractionConfig {
    /// Minimum OCR text length to consider successful (default: 50).
    /// If OCR extracts less than this, fall back to vision LLM.
    pub min_ocr_text_length: usize,
    /// Whether to prefer OCR over vision when both available.
    pub prefer_ocr: bool,
    /// Whether to combine OCR and vision results when both produce content.
    pub combine_results: bool,
}

impl Default for ImageExtractionConfig {
    fn default() -> Self {
        Self {
            min_ocr_text_length: 50,
            prefer_ocr: true,
            combine_results: false,
        }
    }
}

/// Image extractor combining OCR and vision LLM strategies.
///
/// Strategy:
/// 1. If OCR enabled and prefer_ocr=true: Try OCR first
/// 2. If OCR produces minimal text (<min_ocr_text_length) and vision available: Use vision
/// 3. If only vision available: Use vision directly
/// 4. combine_results=true: Merge OCR text with vision description
pub struct ImageExtractor {
    config: ImageExtractionConfig,
    #[cfg(feature = "vision")]
    vision_extractor: Option<VisionLlmExtractor>,
}

impl ImageExtractor {
    /// Create image extractor with default configuration.
    pub fn new() -> Self {
        Self {
            config: ImageExtractionConfig::default(),
            #[cfg(feature = "vision")]
            vision_extractor: Some(VisionLlmExtractor::new()),
        }
    }

    /// Create image extractor with custom configuration.
    pub fn with_config(config: ImageExtractionConfig) -> Self {
        Self {
            config,
            #[cfg(feature = "vision")]
            vision_extractor: Some(VisionLlmExtractor::new()),
        }
    }

    /// Create image extractor with custom vision extractor.
    #[cfg(feature = "vision")]
    pub fn with_vision(mut self, vision: VisionLlmExtractor) -> Self {
        self.vision_extractor = Some(vision);
        self
    }

    /// Disable vision LLM (OCR-only mode).
    #[cfg(feature = "vision")]
    pub fn without_vision(mut self) -> Self {
        self.vision_extractor = None;
        self
    }

    /// Detect image format from bytes.
    fn detect_format(content: &[u8]) -> Result<String, ExtractError> {
        if content.len() < 8 {
            return Err(ExtractError::Image("Content too short".to_string()));
        }

        if content.starts_with(&[0x89, 0x50, 0x4E, 0x47]) {
            Ok("png".to_string())
        } else if content.starts_with(&[0xFF, 0xD8, 0xFF]) {
            Ok("jpeg".to_string())
        } else if content.starts_with(b"GIF87a") || content.starts_with(b"GIF89a") {
            Ok("gif".to_string())
        } else if content.starts_with(b"RIFF") && content.len() > 12 && &content[8..12] == b"WEBP" {
            Ok("webp".to_string())
        } else {
            Err(ExtractError::Image("Unknown image format".to_string()))
        }
    }

    /// Try OCR extraction (if available).
    #[cfg(feature = "ocr")]
    async fn try_ocr(&self, content: &[u8]) -> Option<String> {
        use rusty_tesseract::{Args, Image};

        let content = content.to_vec();

        // Run Tesseract in blocking task to avoid blocking async runtime
        let result = tokio::task::spawn_blocking(move || {
            // Load image
            let img = match image::load_from_memory(&content) {
                Ok(img) => img,
                Err(_) => return None,
            };

            // Convert to grayscale format Tesseract expects
            let dynamic_image = img.to_luma8();
            let tesseract_image =
                match Image::from_dynamic_image(&image::DynamicImage::ImageLuma8(dynamic_image)) {
                    Ok(img) => img,
                    Err(_) => return None,
                };

            // Run OCR with default args
            let args = Args::default();
            rusty_tesseract::image_to_string(&tesseract_image, &args).ok()
        })
        .await
        .ok()
        .flatten();

        result
    }

    #[cfg(not(feature = "ocr"))]
    async fn try_ocr(&self, _content: &[u8]) -> Option<String> {
        None
    }

    /// Try vision LLM extraction (if available).
    #[cfg(feature = "vision")]
    async fn try_vision(&self, content: &[u8]) -> Option<ExtractedContent> {
        if let Some(ref vision) = self.vision_extractor {
            vision.extract(content).await.ok()
        } else {
            None
        }
    }

    #[cfg(not(feature = "vision"))]
    async fn try_vision(&self, _content: &[u8]) -> Option<ExtractedContent> {
        None
    }
}

impl Default for ImageExtractor {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Extractor for ImageExtractor {
    async fn extract(&self, content: &[u8]) -> ExtractResult<ExtractedContent> {
        let content_len = content.len();
        let format = Self::detect_format(content)?;

        // Strategy: Try OCR first if preferred and available
        // Note: Initial None values may not be read if features disabled at compile time
        #[allow(unused_assignments)]
        let mut ocr_text: Option<String> = None;
        #[allow(unused_assignments)]
        let mut vision_result: Option<ExtractedContent> = None;

        if self.config.prefer_ocr {
            ocr_text = self.try_ocr(content).await;

            // Check if OCR produced enough text
            let ocr_sufficient = ocr_text
                .as_ref()
                .map(|t| t.trim().len() >= self.config.min_ocr_text_length)
                .unwrap_or(false);

            if ocr_sufficient && !self.config.combine_results {
                // OCR successful, return OCR result
                let text = ocr_text.unwrap();
                let mut result = ExtractedContent::new(
                    text,
                    Modality::Image {
                        format: format.clone(),
                    },
                    ContentSource::Bytes,
                );
                result = result.with_metadata("original_size", content_len);
                result = result.with_metadata("extraction_method", "ocr");
                return Ok(result);
            }

            // OCR insufficient or combining, try vision
            vision_result = self.try_vision(content).await;
        } else {
            // Prefer vision, try it first
            vision_result = self.try_vision(content).await;

            if vision_result.is_none() {
                // Vision failed, try OCR as fallback
                ocr_text = self.try_ocr(content).await;
            }
        }

        // Combine or select result
        if self.config.combine_results {
            if let (Some(ocr), Some(vision)) = (ocr_text.take(), vision_result.take()) {
                // Combine both results
                let combined_text = format!(
                    "EXTRACTED TEXT (OCR):\n{}\n\nIMAGE ANALYSIS:\n{}",
                    ocr.trim(),
                    vision.text.trim()
                );

                let mut result = ExtractedContent::new(
                    combined_text,
                    Modality::Image { format },
                    ContentSource::Bytes,
                );
                result = result.with_metadata("original_size", content_len);
                result = result.with_metadata("extraction_method", "combined");
                return Ok(result);
            }
        }

        // Return vision result if available
        if let Some(mut vision) = vision_result {
            vision = vision.with_metadata("original_size", content_len);
            return Ok(vision);
        }

        // Return OCR result if available (even if short)
        if let Some(text) = ocr_text {
            if !text.trim().is_empty() {
                let mut result = ExtractedContent::new(
                    text,
                    Modality::Image { format },
                    ContentSource::Bytes,
                );
                result = result.with_metadata("original_size", content_len);
                result = result.with_metadata("extraction_method", "ocr");
                return Ok(result);
            }
        }

        // No extraction method succeeded
        Err(ExtractError::ExtractionFailed(
            "Neither OCR nor vision LLM produced results. \
             Ensure at least one extraction method is enabled and configured."
                .to_string(),
        ))
    }

    fn supported_types(&self) -> &[&str] {
        &["image/png", "image/jpeg", "image/gif", "image/webp"]
    }

    fn name(&self) -> &str {
        "image"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = ImageExtractionConfig::default();
        assert_eq!(config.min_ocr_text_length, 50);
        assert!(config.prefer_ocr);
        assert!(!config.combine_results);
    }

    #[test]
    fn test_format_detection_png() {
        let png = vec![0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
        assert_eq!(ImageExtractor::detect_format(&png).unwrap(), "png");
    }

    #[test]
    fn test_format_detection_jpeg() {
        let jpeg = vec![0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46];
        assert_eq!(ImageExtractor::detect_format(&jpeg).unwrap(), "jpeg");
    }

    #[test]
    fn test_format_detection_gif() {
        let gif = b"GIF89a\x00\x00";
        assert_eq!(ImageExtractor::detect_format(gif).unwrap(), "gif");
    }

    #[test]
    fn test_format_detection_webp() {
        // WEBP requires: RIFF (4) + size (4) + WEBP (4) = 12 bytes, check is > 12
        let mut webp = Vec::new();
        webp.extend_from_slice(b"RIFF");
        webp.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]); // size placeholder
        webp.extend_from_slice(b"WEBP");
        webp.extend_from_slice(&[0x00]); // extra byte to pass > 12 check
        assert_eq!(ImageExtractor::detect_format(&webp).unwrap(), "webp");
    }

    #[test]
    fn test_format_detection_unknown() {
        let unknown = vec![0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07];
        assert!(ImageExtractor::detect_format(&unknown).is_err());
    }

    #[test]
    fn test_format_detection_too_short() {
        let short = vec![0x89, 0x50];
        assert!(ImageExtractor::detect_format(&short).is_err());
    }

    #[test]
    fn test_extractor_supports() {
        let extractor = ImageExtractor::new();
        assert!(extractor.supports("image/png"));
        assert!(extractor.supports("image/jpeg"));
        assert!(extractor.supports("image/gif"));
        assert!(extractor.supports("image/webp"));
        assert!(!extractor.supports("application/pdf"));
    }

    #[test]
    fn test_extractor_name() {
        let extractor = ImageExtractor::new();
        assert_eq!(extractor.name(), "image");
    }

    #[test]
    fn test_config_custom() {
        let config = ImageExtractionConfig {
            min_ocr_text_length: 100,
            prefer_ocr: false,
            combine_results: true,
        };
        assert_eq!(config.min_ocr_text_length, 100);
        assert!(!config.prefer_ocr);
        assert!(config.combine_results);
    }
}
