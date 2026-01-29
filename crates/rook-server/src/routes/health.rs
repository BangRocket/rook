//! Health check endpoint.

use axum::{extract::State, Json};
use serde::Serialize;

use crate::error::ApiResult;
use crate::state::AppState;

#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub configured: bool,
    pub version: String,
}

/// Health check endpoint.
/// GET /health
pub async fn health_check(State(state): State<AppState>) -> ApiResult<Json<HealthResponse>> {
    let configured = state.is_configured().await;

    Ok(Json(HealthResponse {
        status: "healthy".to_string(),
        configured,
        version: env!("CARGO_PKG_VERSION").to_string(),
    }))
}
