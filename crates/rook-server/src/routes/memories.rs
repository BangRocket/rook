//! Memory CRUD endpoints.

use std::collections::HashMap;

use axum::{
    extract::{Path, Query, State},
    Json,
};
use serde::{Deserialize, Serialize};

use crate::error::{ApiError, ApiResult};
use crate::state::AppState;
use rook_core::types::{MemoryEvent, MemoryItem, MemoryResult as CoreMemoryResult};

/// Request body for adding a memory.
#[derive(Debug, Deserialize)]
pub struct AddMemoryRequest {
    /// The messages to process.
    pub messages: Vec<MessageInput>,
    /// Optional user ID.
    pub user_id: Option<String>,
    /// Optional agent ID.
    pub agent_id: Option<String>,
    /// Optional run ID.
    pub run_id: Option<String>,
    /// Optional metadata.
    pub metadata: Option<HashMap<String, serde_json::Value>>,
    /// Optional filters.
    pub filters: Option<HashMap<String, serde_json::Value>>,
    /// Whether to include fact extraction.
    pub infer: Option<bool>,
}

#[derive(Debug, Deserialize)]
pub struct MessageInput {
    pub role: String,
    pub content: String,
}

/// Response for adding a memory.
#[derive(Debug, Serialize)]
pub struct AddMemoryResponse {
    pub results: Vec<MemoryResultItem>,
}

#[derive(Debug, Serialize)]
pub struct MemoryResultItem {
    pub id: String,
    pub memory: String,
    pub event: String,
}

impl From<CoreMemoryResult> for MemoryResultItem {
    fn from(r: CoreMemoryResult) -> Self {
        Self {
            id: r.id,
            memory: r.memory,
            event: match r.event {
                MemoryEvent::Add => "ADD",
                MemoryEvent::Update => "UPDATE",
                MemoryEvent::Delete => "DELETE",
                MemoryEvent::None => "NONE",
            }
            .to_string(),
        }
    }
}

/// Add a memory.
/// POST /memories
pub async fn add_memory(
    State(state): State<AppState>,
    Json(request): Json<AddMemoryRequest>,
) -> ApiResult<Json<AddMemoryResponse>> {
    if !state.is_configured().await {
        return Err(ApiError::bad_request(
            "Memory not configured. Call /configure first.",
        ));
    }

    // Convert messages to string format
    let messages_str: String = request
        .messages
        .iter()
        .map(|m| format!("{}: {}", m.role, m.content))
        .collect::<Vec<_>>()
        .join("\n");

    let user_id = request.user_id.clone();
    let agent_id = request.agent_id.clone();
    let run_id = request.run_id.clone();
    let metadata = request.metadata.clone();
    let infer = request.infer.unwrap_or(true);

    let result = {
        let guard = state.inner.read().await;
        let memory = guard
            .memory
            .as_ref()
            .ok_or_else(|| ApiError::bad_request("Memory not configured"))?;

        memory
            .add(messages_str, user_id, agent_id, run_id, metadata, infer, None)
            .await
            .map_err(ApiError::from)?
    };

    let response = AddMemoryResponse {
        results: result.results.into_iter().map(Into::into).collect(),
    };

    Ok(Json(response))
}

/// Query parameters for getting memories.
#[derive(Debug, Deserialize)]
pub struct GetMemoriesQuery {
    pub user_id: Option<String>,
    pub agent_id: Option<String>,
    pub run_id: Option<String>,
}

/// Response for getting memories.
#[derive(Debug, Serialize)]
pub struct GetMemoriesResponse {
    pub results: Vec<MemoryItem>,
}

/// Get all memories.
/// GET /memories
pub async fn get_all_memories(
    State(state): State<AppState>,
    Query(query): Query<GetMemoriesQuery>,
) -> ApiResult<Json<GetMemoriesResponse>> {
    if !state.is_configured().await {
        return Err(ApiError::bad_request(
            "Memory not configured. Call /configure first.",
        ));
    }

    let results = {
        let guard = state.inner.read().await;
        let memory = guard
            .memory
            .as_ref()
            .ok_or_else(|| ApiError::bad_request("Memory not configured"))?;

        memory
            .get_all(query.user_id, query.agent_id, query.run_id, None)
            .await
            .map_err(ApiError::from)?
    };

    Ok(Json(GetMemoriesResponse { results }))
}

/// Get a specific memory by ID.
/// GET /memories/:id
pub async fn get_memory(
    State(state): State<AppState>,
    Path(memory_id): Path<String>,
) -> ApiResult<Json<MemoryItem>> {
    if !state.is_configured().await {
        return Err(ApiError::bad_request(
            "Memory not configured. Call /configure first.",
        ));
    }

    let result = {
        let guard = state.inner.read().await;
        let memory = guard
            .memory
            .as_ref()
            .ok_or_else(|| ApiError::bad_request("Memory not configured"))?;

        memory.get(&memory_id).await.map_err(ApiError::from)?
    };

    match result {
        Some(item) => Ok(Json(item)),
        None => Err(ApiError::not_found(format!(
            "Memory with id '{}' not found",
            memory_id
        ))),
    }
}

/// Request body for updating a memory.
#[derive(Debug, Deserialize)]
pub struct UpdateMemoryRequest {
    pub text: String,
}

/// Update a memory.
/// PUT /memories/:id
pub async fn update_memory(
    State(state): State<AppState>,
    Path(memory_id): Path<String>,
    Json(request): Json<UpdateMemoryRequest>,
) -> ApiResult<Json<MemoryItem>> {
    if !state.is_configured().await {
        return Err(ApiError::bad_request(
            "Memory not configured. Call /configure first.",
        ));
    }

    let result = {
        let guard = state.inner.read().await;
        let memory = guard
            .memory
            .as_ref()
            .ok_or_else(|| ApiError::bad_request("Memory not configured"))?;

        memory
            .update(&memory_id, &request.text)
            .await
            .map_err(ApiError::from)?
    };

    Ok(Json(result))
}

/// Delete a memory.
/// DELETE /memories/:id
pub async fn delete_memory(
    State(state): State<AppState>,
    Path(memory_id): Path<String>,
) -> ApiResult<Json<serde_json::Value>> {
    if !state.is_configured().await {
        return Err(ApiError::bad_request(
            "Memory not configured. Call /configure first.",
        ));
    }

    {
        let guard = state.inner.read().await;
        let memory = guard
            .memory
            .as_ref()
            .ok_or_else(|| ApiError::bad_request("Memory not configured"))?;

        memory.delete(&memory_id).await.map_err(ApiError::from)?;
    }

    Ok(Json(serde_json::json!({
        "message": "Memory deleted successfully"
    })))
}

/// Request body for deleting all memories.
#[derive(Debug, Deserialize)]
pub struct DeleteAllMemoriesRequest {
    pub user_id: Option<String>,
    pub agent_id: Option<String>,
    pub run_id: Option<String>,
}

/// Delete all memories.
/// DELETE /memories
pub async fn delete_all_memories(
    State(state): State<AppState>,
    Json(request): Json<DeleteAllMemoriesRequest>,
) -> ApiResult<Json<serde_json::Value>> {
    if !state.is_configured().await {
        return Err(ApiError::bad_request(
            "Memory not configured. Call /configure first.",
        ));
    }

    {
        let guard = state.inner.read().await;
        let memory = guard
            .memory
            .as_ref()
            .ok_or_else(|| ApiError::bad_request("Memory not configured"))?;

        memory
            .delete_all(request.user_id, request.agent_id, request.run_id)
            .await
            .map_err(ApiError::from)?;
    }

    Ok(Json(serde_json::json!({
        "message": "All memories deleted successfully"
    })))
}

/// Response for memory history.
#[derive(Debug, Serialize)]
pub struct MemoryHistoryResponse {
    pub history: Vec<serde_json::Value>,
}

/// Get memory history.
/// GET /memories/:id/history
pub async fn get_memory_history(
    State(state): State<AppState>,
    Path(memory_id): Path<String>,
) -> ApiResult<Json<MemoryHistoryResponse>> {
    if !state.is_configured().await {
        return Err(ApiError::bad_request(
            "Memory not configured. Call /configure first.",
        ));
    }

    let history = {
        let guard = state.inner.read().await;
        let memory = guard
            .memory
            .as_ref()
            .ok_or_else(|| ApiError::bad_request("Memory not configured"))?;

        memory
            .history(&memory_id)
            .await
            .map_err(ApiError::from)?
            .into_iter()
            .map(|h| serde_json::to_value(h).unwrap_or_default())
            .collect()
    };

    Ok(Json(MemoryHistoryResponse { history }))
}