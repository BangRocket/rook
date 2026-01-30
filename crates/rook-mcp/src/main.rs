//! Rook MCP Server - Memory layer for Claude Code and other MCP clients.
//!
//! This binary provides an MCP server that exposes Rook's memory capabilities
//! as MCP tools. It communicates via stdio transport, which is the standard
//! for local MCP servers.
//!
//! # Configuration
//!
//! Set these environment variables before running:
//!
//! - `OPENAI_API_KEY` - Required for embeddings and LLM operations
//! - `ROOK_DATA_DIR` - Optional, defaults to `~/.rook`
//!
//! # Usage with Claude Code
//!
//! Add to `~/.config/claude/claude_desktop_config.json`:
//!
//! ```json
//! {
//!   "mcpServers": {
//!     "rook": {
//!       "command": "/path/to/rook-mcp"
//!     }
//!   }
//! }
//! ```

use std::sync::Arc;

use anyhow::Result;
use rmcp::{transport::stdio, ServiceExt};
use tokio::sync::RwLock;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

mod server;
mod tools;

use server::MemoryServer;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing to stderr (stdout is used for MCP transport)
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::fmt::layer()
                .with_writer(std::io::stderr)
                .with_ansi(false),
        )
        .with(EnvFilter::from_default_env().add_directive(tracing::Level::INFO.into()))
        .init();

    tracing::info!("Starting Rook MCP server");

    // Initialize memory system
    let memory = initialize_memory().await?;

    // Create MCP server
    let server = MemoryServer::new(Arc::new(RwLock::new(memory)));

    // Serve via stdio transport
    let service = server.serve(stdio()).await.inspect_err(|e| {
        tracing::error!("Server error: {:?}", e);
    })?;

    tracing::info!("MCP server running on stdio");

    service.waiting().await?;
    Ok(())
}

/// Initialize the Memory instance with providers.
///
/// Reads configuration from environment and sets up:
/// - OpenAI LLM for fact extraction
/// - OpenAI embeddings
/// - SQLite vector store (local)
async fn initialize_memory() -> Result<rook_core::Memory> {
    use std::path::PathBuf;

    // Determine data directory
    let data_dir = std::env::var("ROOK_DATA_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            dirs::home_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join(".rook")
        });

    // Ensure data directory exists
    std::fs::create_dir_all(&data_dir)?;

    tracing::info!("Data directory: {}", data_dir.display());

    // Get OpenAI API key from environment
    let api_key = std::env::var("OPENAI_API_KEY")
        .map_err(|_| anyhow::anyhow!("OPENAI_API_KEY environment variable not set"))?;

    // Initialize LLM using core config types
    let llm_config = rook_core::LlmConfig {
        model: "gpt-4o-mini".to_string(),
        api_key: Some(api_key.clone()),
        ..Default::default()
    };
    let llm: Arc<dyn rook_core::Llm> =
        Arc::new(rook_llm::OpenAIProvider::new(llm_config)?);

    // Initialize embedder using core config types
    let embedder_config = rook_core::EmbedderConfig {
        model: "text-embedding-3-small".to_string(),
        embedding_dims: 1536,
        api_key: Some(api_key),
        base_url: None,
    };
    let embedder: Arc<dyn rook_core::Embedder> =
        Arc::new(rook_embeddings::OpenAIEmbedder::new(embedder_config)?);

    // Initialize SQLite vector store
    let vector_db_path = data_dir.join("vectors.db");
    let history_db_path = data_dir.join("history.db");

    let vector_store: Arc<dyn rook_core::VectorStore> = Arc::new(
        rook_vector_stores::SqliteVecStore::new(
            &vector_db_path.to_string_lossy(),
            "rook",
            1536, // text-embedding-3-small dimension
        )?,
    );

    // Create memory config
    let memory_config = rook_core::MemoryConfig {
        history_db_path,
        ..Default::default()
    };

    // Create Memory instance
    let memory = rook_core::Memory::new(
        memory_config,
        llm,
        embedder,
        vector_store,
        None, // graph_store
        None, // reranker
    )?;

    Ok(memory)
}
