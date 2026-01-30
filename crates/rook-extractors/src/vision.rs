//! Vision LLM content extraction using async-openai.
//!
//! Uses GPT-4o or compatible vision models to describe images
//! and extract any visible text.

use crate::error::{ExtractError, ExtractResult};
use crate::types::{ContentSource, ExtractedContent, Modality};
use crate::Extractor;
use async_openai::{
    config::OpenAIConfig,
    types::{
        ChatCompletionRequestMessage, ChatCompletionRequestMessageContentPartImage,
        ChatCompletionRequestUserMessage, ChatCompletionRequestUserMessageContent,
        ChatCompletionRequestUserMessageContentPart, CreateChatCompletionRequest, ImageDetail,
        ImageUrl,
    },
    Client,
};
use async_trait::async_trait;
use base64::{engine::general_purpose::STANDARD, Engine};

/// Configuration for vision LLM extraction.
#[derive(Debug, Clone)]
pub struct VisionConfig {
    /// Model to use (default: gpt-4o).
    pub model: String,
    /// Max tokens for response (default: 1000).
    pub max_tokens: u32,
    /// Image detail level (default: High).
    pub detail: ImageDetail,
    /// Custom system prompt (optional).
    pub prompt: Option<String>,
}

impl Default for VisionConfig {
    fn default() -> Self {
        Self {
            model: "gpt-4o".to_string(),
            max_tokens: 1000,
            detail: ImageDetail::High,
            prompt: None,
        }
    }
}

/// Vision LLM extractor for image description and text extraction.
///
/// Uses OpenAI's vision API (GPT-4o) to analyze images and produce
/// text descriptions. Can also transcribe visible text in images.
pub struct VisionLlmExtractor {
    client: Client<OpenAIConfig>,
    config: VisionConfig,
}

impl VisionLlmExtractor {
    /// Create new vision extractor with default OpenAI client.
    pub fn new() -> Self {
        Self {
            client: Client::new(),
            config: VisionConfig::default(),
        }
    }

    /// Create vision extractor with custom configuration.
    pub fn with_config(config: VisionConfig) -> Self {
        Self {
            client: Client::new(),
            config,
        }
    }

    /// Create vision extractor with custom OpenAI config (for API key, base URL).
    pub fn with_client(client: Client<OpenAIConfig>, config: VisionConfig) -> Self {
        Self { client, config }
    }

    /// Detect image format from bytes (magic number detection).
    fn detect_format(content: &[u8]) -> Result<&'static str, ExtractError> {
        if content.len() < 8 {
            return Err(ExtractError::Image(
                "Content too short to detect format".to_string(),
            ));
        }

        // Check magic numbers
        if content.starts_with(&[0x89, 0x50, 0x4E, 0x47]) {
            Ok("png")
        } else if content.starts_with(&[0xFF, 0xD8, 0xFF]) {
            Ok("jpeg")
        } else if content.starts_with(b"GIF87a") || content.starts_with(b"GIF89a") {
            Ok("gif")
        } else if content.starts_with(b"RIFF") && content.len() > 12 && &content[8..12] == b"WEBP" {
            Ok("webp")
        } else {
            Err(ExtractError::Image("Unknown image format".to_string()))
        }
    }

    /// Build the prompt for vision analysis.
    fn build_prompt(&self) -> String {
        self.config.prompt.clone().unwrap_or_else(|| {
            "Analyze this image thoroughly. \
             First, describe what you see in detail (objects, people, scene, context). \
             Then, if there is any text visible in the image, transcribe it exactly. \
             Format your response as:\n\n\
             DESCRIPTION:\n[Your detailed description]\n\n\
             TEXT (if any):\n[Transcribed text from the image]"
                .to_string()
        })
    }
}

impl Default for VisionLlmExtractor {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Extractor for VisionLlmExtractor {
    async fn extract(&self, content: &[u8]) -> ExtractResult<ExtractedContent> {
        let content_len = content.len();

        // Detect image format
        let format = Self::detect_format(content)?;

        // Encode image as base64 data URL
        let base64_image = STANDARD.encode(content);
        let data_url = format!("data:image/{};base64,{}", format, base64_image);

        // Build the vision request
        let prompt = self.build_prompt();

        let image_part = ChatCompletionRequestMessageContentPartImage {
            image_url: ImageUrl {
                url: data_url,
                detail: Some(self.config.detail.clone()),
            },
        };

        let request = CreateChatCompletionRequest {
            model: self.config.model.clone(),
            messages: vec![ChatCompletionRequestMessage::User(
                ChatCompletionRequestUserMessage {
                    content: ChatCompletionRequestUserMessageContent::Array(vec![
                        ChatCompletionRequestUserMessageContentPart::Text(prompt.into()),
                        ChatCompletionRequestUserMessageContentPart::ImageUrl(image_part),
                    ]),
                    name: None,
                },
            )],
            max_completion_tokens: Some(self.config.max_tokens),
            ..Default::default()
        };

        // Make the API call
        let response = self
            .client
            .chat()
            .create(request)
            .await
            .map_err(|e| ExtractError::Vision(format!("OpenAI API error: {}", e)))?;

        // Extract the response text
        let text = response
            .choices
            .first()
            .and_then(|c| c.message.content.clone())
            .ok_or_else(|| ExtractError::Vision("Empty response from vision API".to_string()))?;

        if text.trim().is_empty() {
            return Err(ExtractError::EmptyContent);
        }

        let mut result = ExtractedContent::new(
            text,
            Modality::Image {
                format: format.to_string(),
            },
            ContentSource::Bytes,
        );

        result = result.with_metadata("original_size", content_len);
        result = result.with_metadata("extraction_method", "vision_llm");
        result = result.with_metadata("model", self.config.model.clone());

        Ok(result)
    }

    fn supported_types(&self) -> &[&str] {
        &["image/png", "image/jpeg", "image/gif", "image/webp"]
    }

    fn name(&self) -> &str {
        "vision-llm"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vision_config_default() {
        let config = VisionConfig::default();
        assert_eq!(config.model, "gpt-4o");
        assert_eq!(config.max_tokens, 1000);
    }

    #[test]
    fn test_format_detection_png() {
        let png = vec![0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
        assert_eq!(VisionLlmExtractor::detect_format(&png).unwrap(), "png");
    }

    #[test]
    fn test_format_detection_jpeg() {
        let jpeg = vec![0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46];
        assert_eq!(VisionLlmExtractor::detect_format(&jpeg).unwrap(), "jpeg");
    }

    #[test]
    fn test_format_detection_gif() {
        let gif = b"GIF89a\x00\x00";
        assert_eq!(VisionLlmExtractor::detect_format(gif).unwrap(), "gif");
    }

    #[test]
    fn test_format_detection_webp() {
        // WEBP requires: RIFF (4) + size (4) + WEBP (4) = 12 bytes, check is > 12
        let mut webp = Vec::new();
        webp.extend_from_slice(b"RIFF");
        webp.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]); // size placeholder
        webp.extend_from_slice(b"WEBP");
        webp.extend_from_slice(&[0x00]); // extra byte to pass > 12 check
        assert_eq!(VisionLlmExtractor::detect_format(&webp).unwrap(), "webp");
    }

    #[test]
    fn test_format_detection_unknown() {
        let unknown = vec![0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07];
        assert!(VisionLlmExtractor::detect_format(&unknown).is_err());
    }

    #[test]
    fn test_format_detection_too_short() {
        let short = vec![0x89, 0x50];
        assert!(VisionLlmExtractor::detect_format(&short).is_err());
    }

    #[test]
    fn test_vision_extractor_supports() {
        let extractor = VisionLlmExtractor::new();
        assert!(extractor.supports("image/png"));
        assert!(extractor.supports("image/jpeg"));
        assert!(extractor.supports("image/gif"));
        assert!(extractor.supports("image/webp"));
        assert!(!extractor.supports("application/pdf"));
        assert!(!extractor.supports("text/plain"));
    }

    #[test]
    fn test_vision_extractor_name() {
        let extractor = VisionLlmExtractor::new();
        assert_eq!(extractor.name(), "vision-llm");
    }

    #[test]
    fn test_vision_config_custom_prompt() {
        let config = VisionConfig {
            prompt: Some("Custom prompt".to_string()),
            ..Default::default()
        };
        let extractor = VisionLlmExtractor::with_config(config);
        assert_eq!(extractor.build_prompt(), "Custom prompt");
    }
}
