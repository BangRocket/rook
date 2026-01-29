//! Error types for rook operations.
//!
//! This module provides a comprehensive error hierarchy with structured error codes,
//! suggestions for resolution, and debug information.

use std::collections::HashMap;
use thiserror::Error;

/// Result type alias for rook operations.
pub type RookResult<T> = Result<T, RookError>;

/// Main error type for all rook operations.
#[derive(Error, Debug)]
pub enum RookError {
    /// Authentication failed.
    #[error("Authentication error: {message}")]
    Authentication {
        message: String,
        code: ErrorCode,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// Input validation failed.
    #[error("Validation error: {message}")]
    Validation {
        message: String,
        code: ErrorCode,
        details: HashMap<String, String>,
        suggestion: Option<String>,
    },

    /// Memory not found.
    #[error("Memory not found: {message}")]
    NotFound {
        message: String,
        code: ErrorCode,
        memory_id: Option<String>,
    },

    /// Rate limit exceeded.
    #[error("Rate limit exceeded: {message}")]
    RateLimit {
        message: String,
        code: ErrorCode,
        retry_after: Option<u64>,
    },

    /// Vector store operation failed.
    #[error("Vector store error: {message}")]
    VectorStore {
        message: String,
        code: ErrorCode,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// LLM operation failed.
    #[error("LLM error: {message}")]
    Llm {
        message: String,
        code: ErrorCode,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// Embedding generation failed.
    #[error("Embedding error: {message}")]
    Embedding {
        message: String,
        code: ErrorCode,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// Graph store operation failed.
    #[error("Graph store error: {message}")]
    GraphStore {
        message: String,
        code: ErrorCode,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// Database operation failed.
    #[error("Database error: {message}")]
    Database {
        message: String,
        code: ErrorCode,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// Configuration error.
    #[error("Configuration error: {0}")]
    Configuration(String),

    /// Network error.
    #[error("Network error: {message}")]
    Network {
        message: String,
        code: ErrorCode,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// Quota exceeded.
    #[error("Quota exceeded: {message}")]
    QuotaExceeded {
        message: String,
        code: ErrorCode,
        current_usage: Option<u64>,
        limit: Option<u64>,
    },

    /// Provider not supported.
    #[error("Provider not supported: {provider}")]
    UnsupportedProvider { provider: String },

    /// Parse error.
    #[error("Parse error: {message}")]
    Parse {
        message: String,
        code: ErrorCode,
    },

    /// IO error.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON serialization error.
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// Internal error.
    #[error("Internal error: {0}")]
    Internal(String),
}

/// Error codes for programmatic handling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCode {
    // Authentication (AUTH_xxx)
    AuthInvalidKey,
    AuthExpiredToken,
    AuthMissingCredentials,

    // Validation (VAL_xxx)
    ValInvalidInput,
    ValMissingField,
    ValInvalidFormat,
    ValInvalidMemoryType,
    ValInvalidFilter,

    // Memory (MEM_xxx)
    MemNotFound,
    MemCorrupted,
    MemDuplicate,

    // Rate Limit (RATE_xxx)
    RateLimitExceeded,

    // Vector Store (VEC_xxx)
    VecConnectionFailed,
    VecOperationFailed,
    VecCollectionNotFound,

    // LLM (LLM_xxx)
    LlmConnectionFailed,
    LlmGenerationFailed,
    LlmInvalidResponse,

    // Embedding (EMB_xxx)
    EmbConnectionFailed,
    EmbGenerationFailed,

    // Graph (GRP_xxx)
    GrpConnectionFailed,
    GrpOperationFailed,

    // Database (DB_xxx)
    DbConnectionFailed,
    DbOperationFailed,

    // Network (NET_xxx)
    NetTimeout,
    NetConnectionFailed,

    // Quota (QTA_xxx)
    QtaExceeded,

    // Parse (PARSE_xxx)
    ParseInvalidJson,
    ParseMissingField,

    // Internal
    Internal,
}

impl ErrorCode {
    /// Get the string representation of the error code.
    pub fn as_str(&self) -> &'static str {
        match self {
            ErrorCode::AuthInvalidKey => "AUTH_001",
            ErrorCode::AuthExpiredToken => "AUTH_002",
            ErrorCode::AuthMissingCredentials => "AUTH_003",
            ErrorCode::ValInvalidInput => "VAL_001",
            ErrorCode::ValMissingField => "VAL_002",
            ErrorCode::ValInvalidFormat => "VAL_003",
            ErrorCode::ValInvalidMemoryType => "VAL_004",
            ErrorCode::ValInvalidFilter => "VAL_005",
            ErrorCode::MemNotFound => "MEM_001",
            ErrorCode::MemCorrupted => "MEM_002",
            ErrorCode::MemDuplicate => "MEM_003",
            ErrorCode::RateLimitExceeded => "RATE_001",
            ErrorCode::VecConnectionFailed => "VEC_001",
            ErrorCode::VecOperationFailed => "VEC_002",
            ErrorCode::VecCollectionNotFound => "VEC_003",
            ErrorCode::LlmConnectionFailed => "LLM_001",
            ErrorCode::LlmGenerationFailed => "LLM_002",
            ErrorCode::LlmInvalidResponse => "LLM_003",
            ErrorCode::EmbConnectionFailed => "EMB_001",
            ErrorCode::EmbGenerationFailed => "EMB_002",
            ErrorCode::GrpConnectionFailed => "GRP_001",
            ErrorCode::GrpOperationFailed => "GRP_002",
            ErrorCode::DbConnectionFailed => "DB_001",
            ErrorCode::DbOperationFailed => "DB_002",
            ErrorCode::NetTimeout => "NET_001",
            ErrorCode::NetConnectionFailed => "NET_002",
            ErrorCode::QtaExceeded => "QTA_001",
            ErrorCode::ParseInvalidJson => "PARSE_001",
            ErrorCode::ParseMissingField => "PARSE_002",
            ErrorCode::Internal => "INT_001",
        }
    }
}

impl RookError {
    /// Create a validation error.
    pub fn validation(message: impl Into<String>) -> Self {
        Self::Validation {
            message: message.into(),
            code: ErrorCode::ValInvalidInput,
            details: HashMap::new(),
            suggestion: None,
        }
    }

    /// Create a validation error with suggestion.
    pub fn validation_with_suggestion(message: impl Into<String>, suggestion: impl Into<String>) -> Self {
        Self::Validation {
            message: message.into(),
            code: ErrorCode::ValInvalidInput,
            details: HashMap::new(),
            suggestion: Some(suggestion.into()),
        }
    }

    /// Create a not found error.
    pub fn not_found(memory_id: impl Into<String>) -> Self {
        let id = memory_id.into();
        Self::NotFound {
            message: format!("Memory with id '{}' not found", id),
            code: ErrorCode::MemNotFound,
            memory_id: Some(id),
        }
    }

    /// Create an LLM error.
    pub fn llm(message: impl Into<String>) -> Self {
        Self::Llm {
            message: message.into(),
            code: ErrorCode::LlmGenerationFailed,
            source: None,
        }
    }

    /// Create a vector store error.
    pub fn vector_store(message: impl Into<String>) -> Self {
        Self::VectorStore {
            message: message.into(),
            code: ErrorCode::VecOperationFailed,
            source: None,
        }
    }

    /// Create an embedding error.
    pub fn embedding(message: impl Into<String>) -> Self {
        Self::Embedding {
            message: message.into(),
            code: ErrorCode::EmbGenerationFailed,
            source: None,
        }
    }

    /// Create a parse error.
    pub fn parse(message: impl Into<String>) -> Self {
        Self::Parse {
            message: message.into(),
            code: ErrorCode::ParseInvalidJson,
        }
    }

    /// Create a database error.
    pub fn database(message: impl Into<String>) -> Self {
        Self::Database {
            message: message.into(),
            code: ErrorCode::DbOperationFailed,
            source: None,
        }
    }

    /// Create a graph store error.
    pub fn graph_store(message: impl Into<String>) -> Self {
        Self::GraphStore {
            message: message.into(),
            code: ErrorCode::GrpOperationFailed,
            source: None,
        }
    }

    /// Create a reranker error.
    pub fn reranker(message: impl Into<String>) -> Self {
        Self::Internal(format!("Reranker error: {}", message.into()))
    }

    /// Create an API error.
    pub fn api(message: impl Into<String>) -> Self {
        Self::Network {
            message: message.into(),
            code: ErrorCode::NetConnectionFailed,
            source: None,
        }
    }

    /// Create an authentication error.
    pub fn authentication(message: impl Into<String>) -> Self {
        Self::Authentication {
            message: message.into(),
            code: ErrorCode::AuthInvalidKey,
            source: None,
        }
    }

    /// Create a rate limit error.
    pub fn rate_limit(message: impl Into<String>) -> Self {
        Self::RateLimit {
            message: message.into(),
            code: ErrorCode::RateLimitExceeded,
            retry_after: None,
        }
    }

    /// Get the error code.
    pub fn code(&self) -> ErrorCode {
        match self {
            Self::Authentication { code, .. } => *code,
            Self::Validation { code, .. } => *code,
            Self::NotFound { code, .. } => *code,
            Self::RateLimit { code, .. } => *code,
            Self::VectorStore { code, .. } => *code,
            Self::Llm { code, .. } => *code,
            Self::Embedding { code, .. } => *code,
            Self::GraphStore { code, .. } => *code,
            Self::Database { code, .. } => *code,
            Self::Network { code, .. } => *code,
            Self::QuotaExceeded { code, .. } => *code,
            Self::Parse { code, .. } => *code,
            _ => ErrorCode::Internal,
        }
    }

    /// Get a user-friendly suggestion for resolving this error.
    pub fn suggestion(&self) -> Option<&str> {
        match self {
            Self::Authentication { .. } => Some("Please check your API key and authentication credentials"),
            Self::RateLimit { .. } => Some("Please wait before making more requests"),
            Self::NotFound { .. } => Some("Please check the memory ID and ensure it exists"),
            Self::Validation { suggestion, .. } => suggestion.as_deref(),
            Self::VectorStore { .. } => Some("Please check your vector store connection settings"),
            Self::Llm { .. } => Some("Please check your LLM provider configuration"),
            Self::Embedding { .. } => Some("Please check your embedding provider configuration"),
            _ => None,
        }
    }

    /// Convert from HTTP status code (for client errors).
    pub fn from_http_status(status: u16, body: &str) -> Self {
        match status {
            400 => Self::Validation {
                message: body.to_string(),
                code: ErrorCode::ValInvalidInput,
                details: HashMap::new(),
                suggestion: Some("Please check your request parameters".to_string()),
            },
            401 | 403 => Self::Authentication {
                message: body.to_string(),
                code: ErrorCode::AuthInvalidKey,
                source: None,
            },
            404 => Self::NotFound {
                message: body.to_string(),
                code: ErrorCode::MemNotFound,
                memory_id: None,
            },
            429 => Self::RateLimit {
                message: body.to_string(),
                code: ErrorCode::RateLimitExceeded,
                retry_after: None,
            },
            _ => Self::Internal(format!("HTTP {}: {}", status, body)),
        }
    }
}

impl From<rusqlite::Error> for RookError {
    fn from(err: rusqlite::Error) -> Self {
        Self::Database {
            message: err.to_string(),
            code: ErrorCode::DbOperationFailed,
            source: Some(Box::new(err)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validation_error() {
        let err = RookError::validation("Invalid input");
        assert_eq!(err.code(), ErrorCode::ValInvalidInput);
        assert!(err.to_string().contains("Invalid input"));
    }

    #[test]
    fn test_not_found_error() {
        let err = RookError::not_found("test-id");
        assert_eq!(err.code(), ErrorCode::MemNotFound);
        assert!(err.suggestion().is_some());
    }

    #[test]
    fn test_error_code_as_str() {
        assert_eq!(ErrorCode::AuthInvalidKey.as_str(), "AUTH_001");
        assert_eq!(ErrorCode::MemNotFound.as_str(), "MEM_001");
    }
}
