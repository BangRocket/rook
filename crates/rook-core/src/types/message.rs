//! Message types for LLM interactions.

use serde::{Deserialize, Serialize};

/// Role of a message in a conversation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    System,
    User,
    Assistant,
    Tool,
}

impl Default for MessageRole {
    fn default() -> Self {
        Self::User
    }
}

/// A message in a conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: MessageRole,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl Message {
    /// Create a new user message.
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::User,
            content: content.into(),
            name: None,
            tool_call_id: None,
        }
    }

    /// Create a new assistant message.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::Assistant,
            content: content.into(),
            name: None,
            tool_call_id: None,
        }
    }

    /// Create a new system message.
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::System,
            content: content.into(),
            name: None,
            tool_call_id: None,
        }
    }

    /// Set the name field.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }
}

/// Input type for messages that can be converted to a list of messages.
#[derive(Debug, Clone)]
pub enum MessageInput {
    /// Simple string message (converted to user message).
    String(String),
    /// Single message.
    Single(Message),
    /// List of messages.
    List(Vec<Message>),
}

impl MessageInput {
    /// Normalize any input format to a list of messages.
    pub fn normalize(self) -> Vec<Message> {
        match self {
            MessageInput::String(s) => vec![Message::user(s)],
            MessageInput::Single(m) => vec![m],
            MessageInput::List(msgs) => msgs,
        }
    }
}

impl From<&str> for MessageInput {
    fn from(s: &str) -> Self {
        MessageInput::String(s.to_string())
    }
}

impl From<String> for MessageInput {
    fn from(s: String) -> Self {
        MessageInput::String(s)
    }
}

impl From<Message> for MessageInput {
    fn from(m: Message) -> Self {
        MessageInput::Single(m)
    }
}

impl From<Vec<Message>> for MessageInput {
    fn from(msgs: Vec<Message>) -> Self {
        MessageInput::List(msgs)
    }
}

/// Parse messages into a formatted string for LLM prompts.
pub fn format_messages(messages: &[Message]) -> String {
    messages
        .iter()
        .map(|msg| {
            let role = match msg.role {
                MessageRole::System => "system",
                MessageRole::User => "user",
                MessageRole::Assistant => "assistant",
                MessageRole::Tool => "tool",
            };
            format!("{}: {}", role, msg.content)
        })
        .collect::<Vec<_>>()
        .join("\n")
}
