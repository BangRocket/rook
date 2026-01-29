//! Search endpoint.

use std::collections::HashMap;

use axum::{extract::State, Json};
use serde::{Deserialize, Serialize};

use crate::error::{ApiError, ApiResult};
use crate::state::AppState;
use rook_core::types::MemoryItem;

/// Request body for searching memories.
#[derive(Debug, Deserialize)]
pub struct SearchRequest {
    /// The search query.
    pub query: String,
    /// Optional user ID.
    pub user_id: Option<String>,
    /// Optional agent ID.
    pub agent_id: Option<String>,
    /// Optional run ID.
    pub run_id: Option<String>,
    /// Maximum number of results.
    pub limit: Option<usize>,
    /// Optional filters.
    pub filters: Option<HashMap<String, serde_json::Value>>,
    /// Score threshold.
    pub threshold: Option<f32>,
    /// Whether to rerank results.
    pub rerank: Option<bool>,
}

/// Response for searching memories.
#[derive(Debug, Serialize)]
pub struct SearchResponse {
    pub results: Vec<SearchResultItem>,
}

#[derive(Debug, Serialize)]
pub struct SearchResultItem {
    pub id: String,
    pub memory: String,
    pub score: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub agent_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub run_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, serde_json::Value>>,
}

impl From<MemoryItem> for SearchResultItem {
    fn from(item: MemoryItem) -> Self {
        Self {
            id: item.id,
            memory: item.memory,
            score: item.score.unwrap_or(0.0),
            user_id: item
                .metadata
                .as_ref()
                .and_then(|m| m.get("user_id"))
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
            agent_id: item
                .metadata
                .as_ref()
                .and_then(|m| m.get("agent_id"))
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
            run_id: item
                .metadata
                .as_ref()
                .and_then(|m| m.get("run_id"))
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
            metadata: item.metadata,
        }
    }
}

/// Search memories.
/// POST /search
pub async fn search_memories(
    State(state): State<AppState>,
    Json(request): Json<SearchRequest>,
) -> ApiResult<Json<SearchResponse>> {
    if !state.is_configured().await {
        return Err(ApiError::bad_request(
            "Memory not configured. Call /configure first.",
        ));
    }

    let limit = request.limit.unwrap_or(10);
    let rerank = request.rerank.unwrap_or(false);

    let results = {
        let guard = state.inner.read().await;
        let memory = guard
            .memory
            .as_ref()
            .ok_or_else(|| ApiError::bad_request("Memory not configured"))?;

        memory
            .search(
                &request.query,
                request.user_id,
                request.agent_id,
                request.run_id,
                limit,
                request.filters,
                request.threshold,
                rerank,
            )
            .await
            .map_err(ApiError::from)?
    };

    let response = SearchResponse {
        results: results.results.into_iter().map(Into::into).collect(),
    };

    Ok(Json(response))
}