//! Ollama LLM provider implementation.

use async_trait::async_trait;

use rook_core::error::{RookError, RookResult};
use rook_core::traits::{
    GenerationOptions, Llm, LlmConfig, LlmResponse, LlmStream, Tool, ToolChoice,
};
use rook_core::types::{Message, MessageRole};

#[cfg(feature = "ollama")]
use ollama_rs::{
    generation::chat::{ChatMessage, ChatMessageRequest, MessageRole as OllamaRole},
    Ollama,
};

/// Ollama LLM provider.
pub struct OllamaLlm {
    #[cfg(feature = "ollama")]
    client: Ollama,
    config: LlmConfig,
}

impl OllamaLlm {
    /// Create a new Ollama LLM provider.
    pub fn new(config: LlmConfig) -> RookResult<Self> {
        let base_url = config
            .base_url
            .clone()
            .unwrap_or_else(|| "http://localhost:11434".to_string());

        // Parse host and port from base_url
        let url = url::Url::parse(&base_url)
            .map_err(|e| RookError::Configuration(format!("Invalid Ollama URL: {}", e)))?;

        let host = url.host_str().unwrap_or("localhost").to_string();
        let port = url.port().unwrap_or(11434);

        #[cfg(feature = "ollama")]
        let client = Ollama::new(format!("http://{}", host), port);

        let mut config = config;
        if config.model.is_empty() {
            config.model = "llama3.1:70b".to_string();
        }

        Ok(Self {
            #[cfg(feature = "ollama")]
            client,
            config,
        })
    }

    #[cfg(feature = "ollama")]
    fn message_to_ollama(msg: &Message) -> ChatMessage {
        ChatMessage {
            role: match msg.role {
                MessageRole::System => OllamaRole::System,
                MessageRole::User => OllamaRole::User,
                MessageRole::Assistant => OllamaRole::Assistant,
                MessageRole::Tool => OllamaRole::User,
            },
            content: msg.content.clone(),
            images: None,
        }
    }
}

#[async_trait]
impl Llm for OllamaLlm {
    #[cfg(feature = "ollama")]
    async fn generate(
        &self,
        messages: &[Message],
        options: Option<GenerationOptions>,
    ) -> RookResult<LlmResponse> {
        let options = options.unwrap_or_default();

        let mut ollama_messages: Vec<ChatMessage> =
            messages.iter().map(Self::message_to_ollama).collect();

        // Add JSON instruction if JSON format requested
        if matches!(
            options.response_format,
            Some(rook_core::traits::ResponseFormat::Json)
        ) {
            if let Some(last) = ollama_messages.last_mut() {
                last.content
                    .push_str("\n\nPlease respond with valid JSON only.");
            }
        }

        let request = ChatMessageRequest::new(self.config.model.clone(), ollama_messages);

        let response = self
            .client
            .send_chat_messages(request)
            .await
            .map_err(|e| RookError::llm(format!("Ollama API error: {}", e)))?;

        let content = response.message.map(|m| m.content);

        Ok(LlmResponse {
            content,
            tool_calls: vec![],
            usage: None,
        })
    }

    #[cfg(not(feature = "ollama"))]
    async fn generate(
        &self,
        _messages: &[Message],
        _options: Option<GenerationOptions>,
    ) -> RookResult<LlmResponse> {
        Err(RookError::Configuration(
            "Ollama feature not enabled. Enable the 'ollama' feature.".to_string(),
        ))
    }

    async fn generate_with_tools(
        &self,
        messages: &[Message],
        _tools: &[Tool],
        _tool_choice: ToolChoice,
        options: Option<GenerationOptions>,
    ) -> RookResult<LlmResponse> {
        // Ollama doesn't support native tool calling, just call generate
        self.generate(messages, options).await
    }

    async fn generate_stream(
        &self,
        _messages: &[Message],
        _options: Option<GenerationOptions>,
    ) -> RookResult<LlmStream> {
        Err(RookError::llm("Streaming not yet implemented for Ollama"))
    }

    fn model_name(&self) -> &str {
        &self.config.model
    }

    fn supports_vision(&self) -> bool {
        // Some Ollama models support vision
        self.config.enable_vision
    }

    fn supports_json_mode(&self) -> bool {
        true // Ollama can be prompted to output JSON
    }

    fn is_reasoning_model(&self) -> bool {
        false
    }
}
