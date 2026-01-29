//! LLM trait and related types.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::pin::Pin;

use crate::error::RookResult;
use crate::types::{Message, MessageRole};

/// Tool definition for function calling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    /// Tool name.
    pub name: String,
    /// Tool description.
    pub description: String,
    /// JSON Schema for parameters.
    pub parameters: serde_json::Value,
}

impl Tool {
    /// Create a new tool definition.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: serde_json::Value,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters,
        }
    }
}

/// Tool call returned by LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// Tool name.
    pub name: String,
    /// Tool arguments as key-value pairs.
    pub arguments: HashMap<String, serde_json::Value>,
}

/// Response from LLM generation.
#[derive(Debug, Clone, Default)]
pub struct LlmResponse {
    /// Generated text content.
    pub content: Option<String>,
    /// Tool calls (if any).
    pub tool_calls: Vec<ToolCall>,
    /// Token usage statistics.
    pub usage: Option<TokenUsage>,
}

impl LlmResponse {
    /// Get the content or an empty string.
    pub fn content_or_empty(&self) -> &str {
        self.content.as_deref().unwrap_or("")
    }
}

/// Token usage statistics.
#[derive(Debug, Clone, Default)]
pub struct TokenUsage {
    /// Tokens in the prompt.
    pub prompt_tokens: u32,
    /// Tokens in the completion.
    pub completion_tokens: u32,
    /// Total tokens.
    pub total_tokens: u32,
}

/// Configuration options for LLM generation.
#[derive(Debug, Clone, Default)]
pub struct GenerationOptions {
    /// Sampling temperature (0.0 - 2.0).
    pub temperature: Option<f32>,
    /// Maximum tokens to generate.
    pub max_tokens: Option<u32>,
    /// Top-p nucleus sampling.
    pub top_p: Option<f32>,
    /// Top-k sampling.
    pub top_k: Option<u32>,
    /// Response format.
    pub response_format: Option<ResponseFormat>,
}

/// Response format for LLM output.
#[derive(Debug, Clone)]
pub enum ResponseFormat {
    /// Plain text response.
    Text,
    /// JSON object response.
    Json,
    /// JSON with specific schema.
    JsonSchema(serde_json::Value),
}

/// Tool choice specification.
#[derive(Debug, Clone, Default)]
pub enum ToolChoice {
    /// Let the model decide.
    #[default]
    Auto,
    /// Don't use tools.
    None,
    /// Must use a tool.
    Required,
    /// Use a specific tool.
    Specific(String),
}

/// Stream item for streaming responses.
pub type StreamItem = Result<String, crate::error::RookError>;

/// Stream type for LLM streaming.
pub type LlmStream = Pin<Box<dyn futures::Stream<Item = StreamItem> + Send>>;

/// Core LLM trait - all LLM providers implement this.
#[async_trait]
pub trait Llm: Send + Sync {
    /// Generate a response from the LLM.
    async fn generate(
        &self,
        messages: &[Message],
        options: Option<GenerationOptions>,
    ) -> RookResult<LlmResponse>;

    /// Generate a response with tool calling support.
    async fn generate_with_tools(
        &self,
        messages: &[Message],
        tools: &[Tool],
        tool_choice: ToolChoice,
        options: Option<GenerationOptions>,
    ) -> RookResult<LlmResponse>;

    /// Generate a streaming response.
    async fn generate_stream(
        &self,
        messages: &[Message],
        options: Option<GenerationOptions>,
    ) -> RookResult<LlmStream>;

    /// Get the model name.
    fn model_name(&self) -> &str;

    /// Check if this model supports vision/images.
    fn supports_vision(&self) -> bool {
        false
    }

    /// Check if this model supports JSON mode.
    fn supports_json_mode(&self) -> bool {
        true
    }

    /// Check if this is a reasoning model (may not support certain params).
    fn is_reasoning_model(&self) -> bool {
        false
    }
}

/// LLM configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmConfig {
    /// Model name/identifier.
    pub model: String,
    /// Sampling temperature.
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    /// Maximum tokens to generate.
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    /// Top-p nucleus sampling.
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    /// API key (if not using environment variable).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    /// Base URL for API.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    /// Enable vision/image support.
    #[serde(default)]
    pub enable_vision: bool,
}

fn default_temperature() -> f32 {
    0.1
}

fn default_max_tokens() -> u32 {
    2000
}

fn default_top_p() -> f32 {
    0.1
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            model: String::new(),
            temperature: default_temperature(),
            max_tokens: default_max_tokens(),
            top_p: default_top_p(),
            api_key: None,
            base_url: None,
            enable_vision: false,
        }
    }
}
