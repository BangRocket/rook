//! Factory for creating Memory instances from configuration.

use std::sync::Arc;

use rook_core::config::{LlmProvider, MemoryConfig};
use rook_core::error::{RookError, RookResult};
use rook_core::memory::Memory;
use rook_core::traits::{
    Embedder, EmbedderProvider, GraphStore, GraphStoreConfig, GraphStoreProvider, Llm, Reranker,
    RerankerConfig, RerankerProvider, VectorStore, VectorStoreProvider,
};

use rook_embeddings::{OllamaEmbedder, OpenAIEmbedder};
use rook_llm::{AnthropicLlm, OllamaLlm, OpenAIProvider};
use rook_rerankers::CohereReranker;

// Import only the default-enabled vector store
use rook_vector_stores::QdrantVectorStore;

// Import only the default-enabled graph store
use rook_graph_stores::Neo4jGraphStore;

/// Create a Memory instance from configuration.
pub async fn create_memory(config: MemoryConfig) -> RookResult<Memory> {
    // Create LLM provider
    let llm = create_llm(&config)?;

    // Create embedder
    let embedder = create_embedder(&config)?;

    // Create vector store
    let vector_store = create_vector_store(&config).await?;

    // Create graph store (optional)
    let graph_store = if let Some(ref gs_config) = config.graph_store {
        Some(create_graph_store(gs_config).await?)
    } else {
        None
    };

    // Create reranker (optional)
    let reranker = if let Some(ref rr_config) = config.reranker {
        Some(create_reranker(rr_config)?)
    } else {
        None
    };

    Memory::new(config, llm, embedder, vector_store, graph_store, reranker)
}

fn create_llm(config: &MemoryConfig) -> RookResult<Arc<dyn Llm>> {
    match config.llm.provider {
        LlmProvider::OpenAI => {
            let provider = OpenAIProvider::new(config.llm.config.clone())?;
            Ok(Arc::new(provider))
        }
        LlmProvider::Anthropic => {
            let provider = AnthropicLlm::new(config.llm.config.clone())?;
            Ok(Arc::new(provider))
        }
        LlmProvider::Ollama => {
            let provider = OllamaLlm::new(config.llm.config.clone())?;
            Ok(Arc::new(provider))
        }
        _ => Err(RookError::Configuration(format!(
            "Unsupported LLM provider: {:?}",
            config.llm.provider
        ))),
    }
}

fn create_embedder(config: &MemoryConfig) -> RookResult<Arc<dyn Embedder>> {
    match config.embedder.provider {
        EmbedderProvider::OpenAI => {
            let embedder = OpenAIEmbedder::new(config.embedder.config.clone())?;
            Ok(Arc::new(embedder))
        }
        EmbedderProvider::Ollama => {
            let embedder = OllamaEmbedder::new(config.embedder.config.clone())?;
            Ok(Arc::new(embedder))
        }
        _ => Err(RookError::Configuration(format!(
            "Unsupported embedder provider: {:?}",
            config.embedder.provider
        ))),
    }
}

async fn create_vector_store(config: &MemoryConfig) -> RookResult<Arc<dyn VectorStore>> {
    let vs_config = &config.vector_store;

    match vs_config.provider {
        VectorStoreProvider::Qdrant => {
            let store = QdrantVectorStore::new(vs_config.clone()).await?;
            Ok(Arc::new(store))
        }
        // Additional providers can be enabled via features
        _ => Err(RookError::Configuration(format!(
            "Vector store provider {:?} is not enabled. Enable the corresponding feature.",
            vs_config.provider
        ))),
    }
}

async fn create_graph_store(config: &GraphStoreConfig) -> RookResult<Arc<dyn GraphStore>> {
    match config.provider {
        GraphStoreProvider::Neo4j => {
            let store = Neo4jGraphStore::new(config.clone()).await?;
            Ok(Arc::new(store))
        }
        _ => Err(RookError::Configuration(format!(
            "Graph store provider {:?} is not enabled. Enable the corresponding feature.",
            config.provider
        ))),
    }
}

fn create_reranker(config: &RerankerConfig) -> RookResult<Arc<dyn Reranker>> {
    match config.provider {
        RerankerProvider::Cohere => {
            let reranker = CohereReranker::new(config.clone())?;
            Ok(Arc::new(reranker))
        }
        // LLM reranker requires the llm feature
        _ => Err(RookError::Configuration(format!(
            "Unsupported reranker provider: {:?}",
            config.provider
        ))),
    }
}
