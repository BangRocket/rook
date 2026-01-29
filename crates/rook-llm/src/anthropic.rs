//! Anthropic (Claude) LLM provider implementation.

use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};

use rook_core::error::{RookError, RookResult};
use rook_core::traits::{
    GenerationOptions, Llm, LlmConfig, LlmResponse, LlmStream, ResponseFormat, Tool, ToolCall,
    ToolChoice, TokenUsage,
};
use rook_core::types::{Message, MessageRole};

const ANTHROPIC_API_URL: &str = "https://api.anthropic.com/v1";
const ANTHROPIC_VERSION: &str = "2023-06-01";

/// Anthropic LLM provider.
pub struct AnthropicLlm {
    client: Client,
    config: LlmConfig,
    base_url: String,
}

#[derive(Debug, Serialize)]
struct AnthropicRequest {
    model: String,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    messages: Vec<AnthropicMessage>,
}

#[derive(Debug, Serialize)]
struct AnthropicMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct AnthropicResponse {
    content: Vec<AnthropicContent>,
    #[serde(default)]
    usage: Option<AnthropicUsage>,
}

#[derive(Debug, Deserialize)]
struct AnthropicContent {
    #[serde(rename = "type")]
    content_type: String,
    text: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AnthropicUsage {
    input_tokens: u32,
    output_tokens: u32,
}

#[derive(Debug, Deserialize)]
struct AnthropicError {
    error: AnthropicErrorDetail,
}

#[derive(Debug, Deserialize)]
struct AnthropicErrorDetail {
    message: String,
}

impl AnthropicLlm {
    /// Create a new Anthropic LLM provider.
    pub fn new(config: LlmConfig) -> RookResult<Self> {
        let api_key = config
            .api_key
            .clone()
            .or_else(|| std::env::var("ANTHROPIC_API_KEY").ok())
            .ok_or_else(|| {
                RookError::Configuration("Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable or provide api_key in config.".to_string())
            })?;

        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            "x-api-key",
            api_key
                .parse()
                .map_err(|_| RookError::Configuration("Invalid API key format".to_string()))?,
        );
        headers.insert(
            "anthropic-version",
            ANTHROPIC_VERSION
                .parse()
                .map_err(|_| RookError::Configuration("Invalid version header".to_string()))?,
        );
        headers.insert(
            "content-type",
            "application/json"
                .parse()
                .map_err(|_| RookError::Configuration("Invalid content type".to_string()))?,
        );

        let client = Client::builder()
            .default_headers(headers)
            .build()
            .map_err(|e| RookError::Configuration(format!("Failed to create HTTP client: {}", e)))?;

        let base_url = config
            .base_url
            .clone()
            .unwrap_or_else(|| ANTHROPIC_API_URL.to_string());

        let mut config = config;
        if config.model.is_empty() {
            config.model = "claude-3-5-sonnet-20240620".to_string();
        }

        Ok(Self {
            client,
            config,
            base_url,
        })
    }
}

#[async_trait]
impl Llm for AnthropicLlm {
    async fn generate(
        &self,
        messages: &[Message],
        options: Option<GenerationOptions>,
    ) -> RookResult<LlmResponse> {
        let options = options.unwrap_or_default();

        // Separate system message from conversation messages
        let system_msg = messages
            .iter()
            .find(|m| matches!(m.role, MessageRole::System))
            .map(|m| m.content.clone());

        let conversation_msgs: Vec<AnthropicMessage> = messages
            .iter()
            .filter(|m| !matches!(m.role, MessageRole::System))
            .map(|m| AnthropicMessage {
                role: match m.role {
                    MessageRole::User => "user".to_string(),
                    MessageRole::Assistant => "assistant".to_string(),
                    _ => "user".to_string(),
                },
                content: m.content.clone(),
            })
            .collect();

        let request = AnthropicRequest {
            model: self.config.model.clone(),
            max_tokens: options.max_tokens.unwrap_or(self.config.max_tokens),
            temperature: Some(options.temperature.unwrap_or(self.config.temperature)),
            system: system_msg,
            messages: conversation_msgs,
        };

        let response = self
            .client
            .post(format!("{}/messages", self.base_url))
            .json(&request)
            .send()
            .await
            .map_err(|e| RookError::llm(format!("Anthropic API request failed: {}", e)))?;

        let status = response.status();
        let body = response
            .text()
            .await
            .map_err(|e| RookError::llm(format!("Failed to read response body: {}", e)))?;

        if !status.is_success() {
            let error: Result<AnthropicError, _> = serde_json::from_str(&body);
            let message = error
                .map(|e| e.error.message)
                .unwrap_or_else(|_| body.clone());
            return Err(RookError::llm(format!(
                "Anthropic API error ({}): {}",
                status, message
            )));
        }

        let response: AnthropicResponse = serde_json::from_str(&body)
            .map_err(|e| RookError::llm(format!("Failed to parse response: {}", e)))?;

        let content = response
            .content
            .iter()
            .find(|c| c.content_type == "text")
            .and_then(|c| c.text.clone());

        let usage = response.usage.map(|u| TokenUsage {
            prompt_tokens: u.input_tokens,
            completion_tokens: u.output_tokens,
            total_tokens: u.input_tokens + u.output_tokens,
        });

        Ok(LlmResponse {
            content,
            tool_calls: vec![],
            usage,
        })
    }

    async fn generate_with_tools(
        &self,
        messages: &[Message],
        _tools: &[Tool],
        _tool_choice: ToolChoice,
        options: Option<GenerationOptions>,
    ) -> RookResult<LlmResponse> {
        // For now, just call generate without tools
        // TODO: Implement full tool calling support
        self.generate(messages, options).await
    }

    async fn generate_stream(
        &self,
        _messages: &[Message],
        _options: Option<GenerationOptions>,
    ) -> RookResult<LlmStream> {
        Err(RookError::llm("Streaming not yet implemented for Anthropic"))
    }

    fn model_name(&self) -> &str {
        &self.config.model
    }

    fn supports_vision(&self) -> bool {
        self.config.enable_vision
    }

    fn supports_json_mode(&self) -> bool {
        false // Anthropic doesn't have a native JSON mode
    }

    fn is_reasoning_model(&self) -> bool {
        false
    }
}
