//! Configuration endpoints.

use axum::{extract::State, Json};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use crate::error::{ApiError, ApiResult};
use crate::state::AppState;
use rook_core::config::{
    EmbedderProviderConfig, LlmProvider, LlmProviderConfig, MemoryConfig,
};
use rook_core::traits::{
    EmbedderConfig, EmbedderProvider, GraphStoreConfig, GraphStoreProvider, LlmConfig,
    RerankerConfig, RerankerProvider, VectorStoreConfig, VectorStoreProvider,
};

/// Request body for configuring memory.
#[derive(Debug, Deserialize)]
pub struct ConfigureRequest {
    /// LLM configuration.
    pub llm: Option<LlmConfigInput>,
    /// Embedder configuration.
    pub embedder: Option<EmbedderConfigInput>,
    /// Vector store configuration.
    pub vector_store: Option<VectorStoreConfigInput>,
    /// Graph store configuration.
    pub graph_store: Option<GraphStoreConfigInput>,
    /// Reranker configuration.
    pub reranker: Option<RerankerConfigInput>,
    /// Collection name.
    pub collection_name: Option<String>,
    /// Embedding dimension.
    pub embedding_dims: Option<usize>,
}

#[derive(Debug, Deserialize)]
pub struct LlmConfigInput {
    pub provider: String,
    pub model: Option<String>,
    pub api_key: Option<String>,
    pub base_url: Option<String>,
    pub temperature: Option<f32>,
    pub max_tokens: Option<usize>,
}

#[derive(Debug, Deserialize)]
pub struct EmbedderConfigInput {
    pub provider: String,
    pub model: Option<String>,
    pub api_key: Option<String>,
    pub base_url: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct VectorStoreConfigInput {
    pub provider: String,
    pub url: Option<String>,
    pub api_key: Option<String>,
    pub collection_name: Option<String>,
    pub embedding_dims: Option<usize>,
}

#[derive(Debug, Deserialize)]
pub struct GraphStoreConfigInput {
    pub provider: String,
    pub url: Option<String>,
    pub username: Option<String>,
    pub password: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct RerankerConfigInput {
    pub provider: String,
    pub model: Option<String>,
    pub api_key: Option<String>,
}

/// Response for configuration.
#[derive(Debug, Serialize)]
pub struct ConfigureResponse {
    pub message: String,
    pub configured: bool,
}

/// Configure memory.
/// POST /configure
pub async fn configure(
    State(state): State<AppState>,
    Json(request): Json<ConfigureRequest>,
) -> ApiResult<Json<ConfigureResponse>> {
    // Build LLM config
    let llm_config = if let Some(llm) = request.llm {
        let provider = parse_llm_provider(&llm.provider)?;
        LlmProviderConfig {
            provider,
            config: LlmConfig {
                model: llm.model.unwrap_or_else(|| "gpt-4.1-nano".to_string()),
                api_key: llm.api_key,
                base_url: llm.base_url,
                temperature: llm.temperature.unwrap_or(0.7),
                max_tokens: llm.max_tokens.unwrap_or(4096) as u32,
                ..Default::default()
            },
        }
    } else {
        LlmProviderConfig::default()
    };

    // Build embedder config
    let embedder_config = if let Some(embedder) = request.embedder {
        let provider = parse_embedder_provider(&embedder.provider)?;
        EmbedderProviderConfig {
            provider,
            config: EmbedderConfig {
                model: embedder.model.unwrap_or_else(|| "text-embedding-3-small".to_string()),
                api_key: embedder.api_key,
                base_url: embedder.base_url,
                ..Default::default()
            },
        }
    } else {
        EmbedderProviderConfig::default()
    };

    // Build vector store config
    let collection_name = request
        .collection_name
        .clone()
        .unwrap_or_else(|| "rook".to_string());
    let embedding_dims = request.embedding_dims.unwrap_or(1536);

    let vector_store_config = if let Some(vs) = request.vector_store {
        let provider = parse_vector_store_provider(&vs.provider)?;
        let mut config_json = serde_json::json!({});
        if let Some(url) = vs.url {
            config_json["url"] = serde_json::Value::String(url);
        }
        if let Some(api_key) = vs.api_key {
            config_json["api_key"] = serde_json::Value::String(api_key);
        }

        VectorStoreConfig {
            provider,
            collection_name: vs.collection_name.unwrap_or(collection_name.clone()),
            embedding_model_dims: vs.embedding_dims.unwrap_or(embedding_dims),
            pool: None,
            config: config_json,
        }
    } else {
        VectorStoreConfig {
            collection_name: collection_name.clone(),
            embedding_model_dims: embedding_dims,
            ..Default::default()
        }
    };

    // Build graph store config (optional)
    let graph_store_config = if let Some(gs) = request.graph_store {
        let provider = parse_graph_store_provider(&gs.provider)?;
        Some(GraphStoreConfig {
            provider,
            url: gs.url.unwrap_or_else(|| "bolt://localhost:7687".to_string()),
            username: gs.username,
            password: gs.password,
            ..Default::default()
        })
    } else {
        None
    };

    // Build reranker config (optional)
    let reranker_config = if let Some(rr) = request.reranker {
        let provider = parse_reranker_provider(&rr.provider)?;
        Some(RerankerConfig {
            provider,
            model: rr.model.unwrap_or_default(),
            api_key: rr.api_key,
            ..Default::default()
        })
    } else {
        None
    };

    // Build memory config
    let config = MemoryConfig {
        llm: llm_config,
        embedder: embedder_config,
        vector_store: vector_store_config,
        graph_store: graph_store_config,
        reranker: reranker_config,
        history_db_path: PathBuf::from(".rook/history.db"),
        ..Default::default()
    };

    // Configure memory
    state.configure(config).await?;

    Ok(Json(ConfigureResponse {
        message: "Memory configured successfully".to_string(),
        configured: true,
    }))
}

/// Response for reset.
#[derive(Debug, Serialize)]
pub struct ResetResponse {
    pub message: String,
}

/// Reset memory.
/// POST /reset
pub async fn reset(State(state): State<AppState>) -> ApiResult<Json<ResetResponse>> {
    state.reset().await?;

    Ok(Json(ResetResponse {
        message: "Memory reset successfully".to_string(),
    }))
}

// Helper functions to parse provider types
fn parse_llm_provider(provider: &str) -> ApiResult<LlmProvider> {
    match provider.to_lowercase().as_str() {
        "openai" => Ok(LlmProvider::OpenAI),
        "anthropic" => Ok(LlmProvider::Anthropic),
        "ollama" => Ok(LlmProvider::Ollama),
        "azure" | "azure_openai" => Ok(LlmProvider::AzureOpenAI),
        "groq" => Ok(LlmProvider::Groq),
        "together" => Ok(LlmProvider::Together),
        "deepseek" => Ok(LlmProvider::DeepSeek),
        "gemini" => Ok(LlmProvider::Gemini),
        _ => Err(ApiError::validation(format!(
            "Unknown LLM provider: {}. Supported: openai, anthropic, ollama, azure_openai, groq, together, deepseek, gemini",
            provider
        ))),
    }
}

fn parse_embedder_provider(provider: &str) -> ApiResult<EmbedderProvider> {
    match provider.to_lowercase().as_str() {
        "openai" => Ok(EmbedderProvider::OpenAI),
        "ollama" => Ok(EmbedderProvider::Ollama),
        "huggingface" => Ok(EmbedderProvider::HuggingFace),
        "cohere" => Ok(EmbedderProvider::Cohere),
        "vertex" | "vertex_ai" => Ok(EmbedderProvider::VertexAI),
        "azure" | "azure_openai" => Ok(EmbedderProvider::AzureOpenAI),
        _ => Err(ApiError::validation(format!(
            "Unknown embedder provider: {}. Supported: openai, ollama, huggingface, cohere, vertex_ai, azure_openai",
            provider
        ))),
    }
}

fn parse_vector_store_provider(provider: &str) -> ApiResult<VectorStoreProvider> {
    match provider.to_lowercase().as_str() {
        "qdrant" => Ok(VectorStoreProvider::Qdrant),
        "redis" => Ok(VectorStoreProvider::Redis),
        "elasticsearch" => Ok(VectorStoreProvider::Elasticsearch),
        "opensearch" => Ok(VectorStoreProvider::Opensearch),
        "mongodb" => Ok(VectorStoreProvider::MongoDB),
        "pgvector" => Ok(VectorStoreProvider::Pgvector),
        "pgvector_pooled" | "postgres_pooled" | "postgresql_pooled" => {
            Ok(VectorStoreProvider::PgvectorPooled)
        }
        "pinecone" => Ok(VectorStoreProvider::Pinecone),
        "weaviate" => Ok(VectorStoreProvider::Weaviate),
        "chroma" => Ok(VectorStoreProvider::Chroma),
        "milvus" => Ok(VectorStoreProvider::Milvus),
        "upstash" => Ok(VectorStoreProvider::UpstashVector),
        "azure" | "azure_ai_search" => Ok(VectorStoreProvider::AzureAiSearch),
        "vertex" | "vertex_ai" => Ok(VectorStoreProvider::VertexAiVectorSearch),
        "supabase" => Ok(VectorStoreProvider::Supabase),
        "cassandra" => Ok(VectorStoreProvider::Cassandra),
        "neptune" => Ok(VectorStoreProvider::Neptune),
        "databricks" => Ok(VectorStoreProvider::Databricks),
        "faiss" => Ok(VectorStoreProvider::Faiss),
        "valkey" => Ok(VectorStoreProvider::Valkey),
        "s3" | "s3_vectors" => Ok(VectorStoreProvider::S3Vectors),
        _ => Err(ApiError::validation(format!(
            "Unknown vector store provider: {}",
            provider
        ))),
    }
}

fn parse_graph_store_provider(provider: &str) -> ApiResult<GraphStoreProvider> {
    match provider.to_lowercase().as_str() {
        "embedded" => Ok(GraphStoreProvider::Embedded),
        "neo4j" => Ok(GraphStoreProvider::Neo4j),
        "memgraph" => Ok(GraphStoreProvider::Memgraph),
        "neptune" => Ok(GraphStoreProvider::Neptune),
        _ => Err(ApiError::validation(format!(
            "Unknown graph store provider: {}. Supported: embedded, neo4j, memgraph, neptune",
            provider
        ))),
    }
}

fn parse_reranker_provider(provider: &str) -> ApiResult<RerankerProvider> {
    match provider.to_lowercase().as_str() {
        "cohere" => Ok(RerankerProvider::Cohere),
        "llm" => Ok(RerankerProvider::Llm),
        "huggingface" => Ok(RerankerProvider::HuggingFace),
        "sentence_transformer" => Ok(RerankerProvider::SentenceTransformer),
        _ => Err(ApiError::validation(format!(
            "Unknown reranker provider: {}. Supported: cohere, llm, huggingface, sentence_transformer",
            provider
        ))),
    }
}