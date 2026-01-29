//! Error handling for the REST API server.

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::Serialize;
use std::fmt;

/// API error type.
#[derive(Debug)]
pub struct ApiError {
    pub status: StatusCode,
    pub code: String,
    pub message: String,
    pub details: Option<serde_json::Value>,
}

impl ApiError {
    pub fn new(status: StatusCode, code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            status,
            code: code.into(),
            message: message.into(),
            details: None,
        }
    }

    pub fn with_details(mut self, details: serde_json::Value) -> Self {
        self.details = Some(details);
        self
    }

    // Common error constructors
    pub fn bad_request(message: impl Into<String>) -> Self {
        Self::new(StatusCode::BAD_REQUEST, "BAD_REQUEST", message)
    }

    pub fn not_found(message: impl Into<String>) -> Self {
        Self::new(StatusCode::NOT_FOUND, "NOT_FOUND", message)
    }

    pub fn unauthorized(message: impl Into<String>) -> Self {
        Self::new(StatusCode::UNAUTHORIZED, "UNAUTHORIZED", message)
    }

    pub fn internal(message: impl Into<String>) -> Self {
        Self::new(StatusCode::INTERNAL_SERVER_ERROR, "INTERNAL_ERROR", message)
    }

    pub fn validation(message: impl Into<String>) -> Self {
        Self::new(StatusCode::UNPROCESSABLE_ENTITY, "VALIDATION_ERROR", message)
    }

    pub fn conflict(message: impl Into<String>) -> Self {
        Self::new(StatusCode::CONFLICT, "CONFLICT", message)
    }

    pub fn rate_limit(message: impl Into<String>) -> Self {
        Self::new(StatusCode::TOO_MANY_REQUESTS, "RATE_LIMIT", message)
    }
}

impl fmt::Display for ApiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}: {}", self.status, self.code, self.message)
    }
}

impl std::error::Error for ApiError {}

/// Error response body.
#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: ErrorBody,
}

#[derive(Debug, Serialize)]
pub struct ErrorBody {
    pub code: String,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<serde_json::Value>,
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = ErrorResponse {
            error: ErrorBody {
                code: self.code,
                message: self.message,
                details: self.details,
            },
        };

        (self.status, Json(body)).into_response()
    }
}

// Convert from rook-core errors
impl From<rook_core::error::RookError> for ApiError {
    fn from(err: rook_core::error::RookError) -> Self {
        use rook_core::error::RookError;

        match err {
            RookError::Configuration(msg) => ApiError::bad_request(msg),
            RookError::Authentication { message, .. } => ApiError::unauthorized(message),
            RookError::NotFound { message, .. } => ApiError::not_found(message),
            RookError::Validation { message, .. } => ApiError::validation(message),
            RookError::RateLimit { message, .. } => ApiError::rate_limit(message),
            RookError::VectorStore { message, .. } => {
                ApiError::internal(format!("Vector store error: {}", message))
            }
            RookError::Llm { message, .. } => {
                ApiError::internal(format!("LLM error: {}", message))
            }
            RookError::Embedding { message, .. } => {
                ApiError::internal(format!("Embedding error: {}", message))
            }
            RookError::GraphStore { message, .. } => {
                ApiError::internal(format!("Graph store error: {}", message))
            }
            RookError::Database { message, .. } => {
                ApiError::internal(format!("Database error: {}", message))
            }
            RookError::Network { message, .. } => {
                ApiError::internal(format!("Network error: {}", message))
            }
            RookError::UnsupportedProvider { provider } => {
                ApiError::bad_request(format!("Unsupported provider: {}", provider))
            }
            RookError::Parse { message, .. } => {
                ApiError::internal(format!("Parse error: {}", message))
            }
            RookError::Serialization(e) => {
                ApiError::internal(format!("Serialization error: {}", e))
            }
            RookError::Io(e) => ApiError::internal(format!("IO error: {}", e)),
            RookError::Internal(msg) => ApiError::internal(msg),
            RookError::QuotaExceeded { message, .. } => {
                ApiError::rate_limit(format!("Quota exceeded: {}", message))
            }
        }
    }
}

/// Result type alias for API handlers.
pub type ApiResult<T> = Result<T, ApiError>;
