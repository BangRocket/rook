//! Configuration system for rook.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use crate::traits::{
    EmbedderConfig, EmbedderProvider, GraphStoreConfig, LlmConfig, RerankerConfig,
    VectorStoreConfig, VectorStoreProvider,
};

/// LLM provider type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum LlmProvider {
    #[default]
    OpenAI,
    Anthropic,
    Ollama,
    AzureOpenAI,
    Groq,
    Together,
    DeepSeek,
    Gemini,
    AwsBedrock,
    Vllm,
    LmStudio,
}

/// Provider configuration with type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmProviderConfig {
    /// Provider type.
    pub provider: LlmProvider,
    /// Provider-specific configuration.
    #[serde(flatten)]
    pub config: LlmConfig,
}

impl Default for LlmProviderConfig {
    fn default() -> Self {
        Self {
            provider: LlmProvider::OpenAI,
            config: LlmConfig {
                model: "gpt-4.1-nano-2025-04-14".to_string(),
                ..Default::default()
            },
        }
    }
}

/// Embedder provider configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbedderProviderConfig {
    /// Provider type.
    pub provider: EmbedderProvider,
    /// Provider-specific configuration.
    #[serde(flatten)]
    pub config: EmbedderConfig,
}

impl Default for EmbedderProviderConfig {
    fn default() -> Self {
        Self {
            provider: EmbedderProvider::OpenAI,
            config: EmbedderConfig::default(),
        }
    }
}

/// Main memory configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct MemoryConfig {
    /// Vector store configuration.
    pub vector_store: VectorStoreConfig,
    /// LLM configuration.
    pub llm: LlmProviderConfig,
    /// Embedder configuration.
    pub embedder: EmbedderProviderConfig,
    /// Graph store configuration (optional).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub graph_store: Option<GraphStoreConfig>,
    /// Reranker configuration (optional).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reranker: Option<RerankerConfig>,
    /// Path to history database.
    pub history_db_path: PathBuf,
    /// API version.
    pub version: String,
    /// Custom fact extraction prompt.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub custom_fact_extraction_prompt: Option<String>,
    /// Custom update memory prompt.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub custom_update_memory_prompt: Option<String>,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        let rook_dir = dirs::home_dir()
            .map(|h| h.join(".rook"))
            .unwrap_or_else(|| PathBuf::from(".rook"));

        Self {
            vector_store: VectorStoreConfig::default(),
            llm: LlmProviderConfig::default(),
            embedder: EmbedderProviderConfig::default(),
            graph_store: None,
            reranker: None,
            history_db_path: rook_dir.join("history.db"),
            version: "v1.1".to_string(),
            custom_fact_extraction_prompt: None,
            custom_update_memory_prompt: None,
        }
    }
}

impl MemoryConfig {
    /// Load configuration from a file (TOML, JSON, or YAML).
    pub fn from_file(path: impl AsRef<std::path::Path>) -> crate::error::RookResult<Self> {
        let content = std::fs::read_to_string(path.as_ref())?;
        let ext = path.as_ref().extension().and_then(|e| e.to_str());

        match ext {
            Some("toml") => toml::from_str(&content)
                .map_err(|e| crate::error::RookError::Configuration(e.to_string())),
            Some("json") => serde_json::from_str(&content)
                .map_err(|e| crate::error::RookError::Configuration(e.to_string())),
            Some("yaml" | "yml") => serde_yaml::from_str(&content)
                .map_err(|e| crate::error::RookError::Configuration(e.to_string())),
            _ => Err(crate::error::RookError::Configuration(
                "Unsupported config file format. Use .toml, .json, or .yaml".to_string(),
            )),
        }
    }

    /// Load configuration from environment variables.
    pub fn from_env() -> Self {
        let mut config = Self::default();

        // LLM configuration
        if let Ok(model) = std::env::var("MEM0_LLM_MODEL") {
            config.llm.config.model = model;
        }
        if let Ok(api_key) = std::env::var("OPENAI_API_KEY") {
            config.llm.config.api_key = Some(api_key.clone());
            config.embedder.config.api_key = Some(api_key);
        }

        // Vector store configuration
        if let Ok(provider) = std::env::var("MEM0_VECTOR_STORE_PROVIDER") {
            config.vector_store.provider = match provider.to_lowercase().as_str() {
                "qdrant" => VectorStoreProvider::Qdrant,
                "pinecone" => VectorStoreProvider::Pinecone,
                "pgvector" => VectorStoreProvider::Pgvector,
                "redis" => VectorStoreProvider::Redis,
                "mongodb" => VectorStoreProvider::MongoDB,
                _ => VectorStoreProvider::Qdrant,
            };
        }

        // History database path
        if let Ok(path) = std::env::var("MEM0_HISTORY_DB_PATH") {
            config.history_db_path = PathBuf::from(path);
        }

        config
    }

    /// Build configuration using builder pattern.
    pub fn builder() -> MemoryConfigBuilder {
        MemoryConfigBuilder::default()
    }
}

/// Builder for MemoryConfig.
#[derive(Default)]
pub struct MemoryConfigBuilder {
    config: MemoryConfig,
}

impl MemoryConfigBuilder {
    /// Set vector store configuration.
    pub fn vector_store(mut self, config: VectorStoreConfig) -> Self {
        self.config.vector_store = config;
        self
    }

    /// Set LLM configuration.
    pub fn llm(mut self, config: LlmProviderConfig) -> Self {
        self.config.llm = config;
        self
    }

    /// Set embedder configuration.
    pub fn embedder(mut self, config: EmbedderProviderConfig) -> Self {
        self.config.embedder = config;
        self
    }

    /// Set graph store configuration.
    pub fn graph_store(mut self, config: GraphStoreConfig) -> Self {
        self.config.graph_store = Some(config);
        self
    }

    /// Set reranker configuration.
    pub fn reranker(mut self, config: RerankerConfig) -> Self {
        self.config.reranker = Some(config);
        self
    }

    /// Set history database path.
    pub fn history_db_path(mut self, path: PathBuf) -> Self {
        self.config.history_db_path = path;
        self
    }

    /// Set custom fact extraction prompt.
    pub fn custom_fact_extraction_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.config.custom_fact_extraction_prompt = Some(prompt.into());
        self
    }

    /// Set custom update memory prompt.
    pub fn custom_update_memory_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.config.custom_update_memory_prompt = Some(prompt.into());
        self
    }

    /// Build the configuration.
    pub fn build(self) -> MemoryConfig {
        self.config
    }
}
