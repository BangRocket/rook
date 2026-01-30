//! MCP server implementation for Rook memory system.
//!
//! Uses the rmcp SDK's macro-based approach for defining tools.

use std::collections::HashMap;
use std::sync::Arc;

use rmcp::{
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::*,
    schemars, tool, tool_handler, tool_router,
    ErrorData as McpError, RoleServer, ServerHandler,
};

use tokio::sync::RwLock;

use crate::tools::*;

/// MCP server for Rook memory operations.
///
/// Wraps a `rook_core::Memory` instance and exposes it as MCP tools.
#[derive(Clone)]
pub struct MemoryServer {
    memory: Arc<RwLock<rook_core::Memory>>,
    tool_router: ToolRouter<MemoryServer>,
}

#[tool_router]
impl MemoryServer {
    /// Create a new MemoryServer wrapping the given Memory instance.
    pub fn new(memory: Arc<RwLock<rook_core::Memory>>) -> Self {
        Self {
            memory,
            tool_router: Self::tool_router(),
        }
    }

    /// Add a new memory to the store.
    ///
    /// Memories are indexed for semantic search. The content will be
    /// processed by the LLM to extract facts and create embeddings.
    #[tool(
        name = "memory_add",
        description = "Add a new memory to the store. Memories are indexed for semantic search and can be retrieved later based on relevance to a query."
    )]
    async fn memory_add(
        &self,
        Parameters(input): Parameters<AddMemoryInput>,
    ) -> Result<CallToolResult, McpError> {
        let memory = self.memory.read().await;

        // Convert metadata to HashMap if provided
        let metadata: Option<HashMap<String, serde_json::Value>> =
            input.metadata.and_then(|v| {
                v.as_object()
                    .map(|m| m.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
            });

        let result = memory
            .add(
                input.content.as_str(),
                input.user_id,
                None, // agent_id
                None, // run_id
                metadata,
                true, // infer facts
                None, // memory_type
            )
            .await
            .map_err(|e| McpError::internal_error(e.to_string(), None))?;

        // Return first result (add typically returns one memory)
        if let Some(mem_result) = result.results.first() {
            let output = AddMemoryResult {
                id: mem_result.id.clone(),
                memory: mem_result.memory.clone(),
                event: format!("{:?}", mem_result.event),
            };
            Ok(CallToolResult::success(vec![Content::text(
                serde_json::to_string_pretty(&output).unwrap_or_default(),
            )]))
        } else {
            Ok(CallToolResult::success(vec![Content::text(
                "No memory created (content may have been filtered as duplicate)",
            )]))
        }
    }

    /// Search memories by semantic similarity.
    ///
    /// Returns the most relevant memories for the given query, ranked by
    /// similarity score.
    #[tool(
        name = "memory_search",
        description = "Search memories by semantic similarity. Returns the most relevant memories for the query, ranked by relevance score."
    )]
    async fn memory_search(
        &self,
        Parameters(input): Parameters<SearchMemoryInput>,
    ) -> Result<CallToolResult, McpError> {
        let memory = self.memory.read().await;

        let results = memory
            .search(
                &input.query,
                input.user_id,
                None, // agent_id
                None, // run_id
                input.limit,
                None,  // filters
                None,  // threshold
                false, // rerank
            )
            .await
            .map_err(|e| McpError::internal_error(e.to_string(), None))?;

        let output: Vec<MemorySearchResult> = results
            .results
            .into_iter()
            .map(|r| MemorySearchResult {
                id: r.id,
                memory: r.memory,
                score: r.score.unwrap_or(0.0),
                metadata: r.metadata.and_then(|m| serde_json::to_value(m).ok()),
            })
            .collect();

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&output).unwrap_or_default(),
        )]))
    }

    /// Get a specific memory by its ID.
    #[tool(
        name = "memory_get",
        description = "Get a specific memory by its ID. Returns the full memory content and metadata."
    )]
    async fn memory_get(
        &self,
        Parameters(input): Parameters<GetMemoryInput>,
    ) -> Result<CallToolResult, McpError> {
        let memory = self.memory.read().await;

        let result = memory
            .get(&input.id)
            .await
            .map_err(|e| McpError::internal_error(e.to_string(), None))?;

        match result {
            Some(mem) => {
                let output = GetMemoryResult {
                    id: mem.id,
                    memory: mem.memory,
                    metadata: mem.metadata.and_then(|m| serde_json::to_value(m).ok()),
                    created_at: mem.created_at,
                };
                Ok(CallToolResult::success(vec![Content::text(
                    serde_json::to_string_pretty(&output).unwrap_or_default(),
                )]))
            }
            None => Err(McpError::invalid_params(
                format!("Memory with id '{}' not found", input.id),
                None,
            )),
        }
    }

    /// Delete a memory by its ID.
    #[tool(
        name = "memory_delete",
        description = "Delete a memory by its ID. This permanently removes the memory from the store."
    )]
    async fn memory_delete(
        &self,
        Parameters(input): Parameters<DeleteMemoryInput>,
    ) -> Result<CallToolResult, McpError> {
        let memory = self.memory.read().await;

        memory
            .delete(&input.id)
            .await
            .map_err(|e| McpError::internal_error(e.to_string(), None))?;

        let output = DeleteMemoryResult {
            deleted: input.id,
        };
        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&output).unwrap_or_default(),
        )]))
    }
}

#[tool_handler]
impl ServerHandler for MemoryServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            protocol_version: ProtocolVersion::V_2024_11_05,
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            server_info: Implementation::from_build_env(),
            instructions: Some(
                "Rook Memory Server - A persistent memory layer for AI assistants. \
                 Use memory_add to store new memories, memory_search to find relevant \
                 memories based on a query, memory_get to retrieve a specific memory, \
                 and memory_delete to remove memories."
                    .to_string(),
            ),
        }
    }
}
