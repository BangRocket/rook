//! OpenAI LLM provider implementation.

use async_trait::async_trait;

use rook_core::error::{RookError, RookResult};
use rook_core::traits::{
    GenerationOptions, Llm, LlmConfig, LlmResponse, LlmStream, Tool, ToolChoice, TokenUsage,
};
use rook_core::types::Message;

#[cfg(feature = "openai")]
use async_openai::{
    config::OpenAIConfig,
    types::{
        ChatCompletionRequestAssistantMessage, ChatCompletionRequestMessage,
        ChatCompletionRequestSystemMessage, ChatCompletionRequestUserMessage,
        CreateChatCompletionRequest,
    },
    Client,
};

/// OpenAI LLM provider.
pub struct OpenAIProvider {
    #[cfg(feature = "openai")]
    client: Client<OpenAIConfig>,
    config: LlmConfig,
}

impl OpenAIProvider {
    /// Create a new OpenAI LLM provider.
    pub fn new(config: LlmConfig) -> RookResult<Self> {
        let api_key = config
            .api_key
            .clone()
            .or_else(|| std::env::var("OPENAI_API_KEY").ok())
            .ok_or_else(|| {
                RookError::Configuration("OpenAI API key not found. Set OPENAI_API_KEY environment variable or provide api_key in config.".to_string())
            })?;

        #[cfg(feature = "openai")]
        let openai_config = if let Some(ref base_url) = config.base_url {
            OpenAIConfig::new()
                .with_api_key(api_key)
                .with_api_base(base_url)
        } else {
            OpenAIConfig::new().with_api_key(api_key)
        };

        #[cfg(feature = "openai")]
        let client = Client::with_config(openai_config);

        let mut config = config;
        if config.model.is_empty() {
            config.model = "gpt-4.1-nano-2025-04-14".to_string();
        }

        Ok(Self {
            #[cfg(feature = "openai")]
            client,
            config,
        })
    }

    /// Check if this is a reasoning model that doesn't support certain params.
    fn is_reasoning_model_internal(&self) -> bool {
        let model_lower = self.config.model.to_lowercase();
        ["o1", "o3", "gpt-5"]
            .iter()
            .any(|m| model_lower.contains(m))
    }

    #[cfg(feature = "openai")]
    fn message_to_openai(msg: &Message) -> ChatCompletionRequestMessage {
        match msg.role {
            rook_core::types::MessageRole::System => {
                ChatCompletionRequestMessage::System(ChatCompletionRequestSystemMessage {
                    content: async_openai::types::ChatCompletionRequestSystemMessageContent::Text(
                        msg.content.clone(),
                    ),
                    name: msg.name.clone(),
                })
            }
            rook_core::types::MessageRole::User => {
                ChatCompletionRequestMessage::User(ChatCompletionRequestUserMessage {
                    content: async_openai::types::ChatCompletionRequestUserMessageContent::Text(
                        msg.content.clone(),
                    ),
                    name: msg.name.clone(),
                })
            }
            rook_core::types::MessageRole::Assistant => {
                ChatCompletionRequestMessage::Assistant(ChatCompletionRequestAssistantMessage {
                    content: Some(
                        async_openai::types::ChatCompletionRequestAssistantMessageContent::Text(
                            msg.content.clone(),
                        ),
                    ),
                    name: msg.name.clone(),
                    ..Default::default()
                })
            }
            rook_core::types::MessageRole::Tool => {
                // For tool messages, we'll treat them as user messages
                ChatCompletionRequestMessage::User(ChatCompletionRequestUserMessage {
                    content: async_openai::types::ChatCompletionRequestUserMessageContent::Text(
                        msg.content.clone(),
                    ),
                    name: msg.name.clone(),
                })
            }
        }
    }
}

#[async_trait]
impl Llm for OpenAIProvider {
    #[cfg(feature = "openai")]
    async fn generate(
        &self,
        messages: &[Message],
        options: Option<GenerationOptions>,
    ) -> RookResult<LlmResponse> {
        let chat_messages: Vec<ChatCompletionRequestMessage> =
            messages.iter().map(Self::message_to_openai).collect();

        let options = options.unwrap_or_default();

        let mut request = CreateChatCompletionRequest {
            model: self.config.model.clone(),
            messages: chat_messages,
            ..Default::default()
        };

        // Only add temperature/top_p for non-reasoning models
        if !self.is_reasoning_model_internal() {
            request.temperature = Some(options.temperature.unwrap_or(self.config.temperature));
            request.top_p = Some(options.top_p.unwrap_or(self.config.top_p));
            request.max_tokens = Some(
                options.max_tokens.unwrap_or(self.config.max_tokens),
            );
        }

        let response = self
            .client
            .chat()
            .create(request)
            .await
            .map_err(|e| RookError::llm(format!("OpenAI API error: {}", e)))?;

        let choice = response
            .choices
            .first()
            .ok_or_else(|| RookError::llm("No response choices returned"))?;

        let content = choice.message.content.clone();

        let usage = response.usage.map(|u| TokenUsage {
            prompt_tokens: u.prompt_tokens,
            completion_tokens: u.completion_tokens,
            total_tokens: u.total_tokens,
        });

        Ok(LlmResponse {
            content,
            tool_calls: vec![],
            usage,
        })
    }

    #[cfg(not(feature = "openai"))]
    async fn generate(
        &self,
        _messages: &[Message],
        _options: Option<GenerationOptions>,
    ) -> RookResult<LlmResponse> {
        Err(RookError::Configuration(
            "OpenAI feature not enabled. Enable the 'openai' feature.".to_string(),
        ))
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
        // TODO: Implement streaming
        Err(RookError::llm("Streaming not yet implemented for OpenAI"))
    }

    fn model_name(&self) -> &str {
        &self.config.model
    }

    fn supports_vision(&self) -> bool {
        self.config.enable_vision
    }

    fn supports_json_mode(&self) -> bool {
        true
    }

    fn is_reasoning_model(&self) -> bool {
        self.is_reasoning_model_internal()
    }
}
