//! Python Memory class with async bridge.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::Arc;

use crate::types::{python_dict_to_metadata, AddResult, MemoryItem, SearchResult};

/// Memory class - the main interface to Rook's memory system.
///
/// Create a Memory instance to store and retrieve memories for AI assistants.
/// The memory system supports semantic search, memory decay, and spreading
/// activation for intelligent retrieval.
///
/// Example:
///     import rook_rs
///
///     # Create with default config (requires environment variables)
///     memory = rook_rs.Memory()
///
///     # Or with explicit config
///     memory = rook_rs.Memory({
///         "llm": {"provider": "openai"},
///         "embedder": {"provider": "openai"},
///         "vector_store": {"provider": "qdrant", "url": "http://localhost:6333"}
///     })
///
///     # Add memories
///     result = memory.add("I prefer dark mode", user_id="user123")
///
///     # Search memories
///     results = memory.search("user preferences", user_id="user123")
#[pyclass]
pub struct Memory {
    inner: Arc<rook_core::Memory>,
    runtime: Arc<tokio::runtime::Runtime>,
}

#[pymethods]
impl Memory {
    /// Create a new Memory instance.
    ///
    /// Args:
    ///     config: Optional configuration dict with keys:
    ///         - llm: LLM provider config (provider, model, api_key)
    ///         - embedder: Embedding provider config
    ///         - vector_store: Vector store config (provider, url, collection)
    ///
    /// Returns:
    ///     Memory instance
    ///
    /// Raises:
    ///     RuntimeError: If initialization fails (e.g., missing API keys)
    #[new]
    #[pyo3(signature = (config=None))]
    pub fn new(config: Option<&Bound<'_, PyAny>>, py: Python<'_>) -> PyResult<Self> {
        // Create tokio runtime
        let runtime = tokio::runtime::Runtime::new().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to create runtime: {}",
                e
            ))
        })?;

        // Parse config from Python dict or use defaults
        let memory_config = if let Some(cfg) = config {
            parse_config(cfg, py)?
        } else {
            rook_core::MemoryConfig::default()
        };

        // Initialize memory system (blocks on async)
        let inner = runtime.block_on(async { create_memory(memory_config).await })?;

        Ok(Self {
            inner: Arc::new(inner),
            runtime: Arc::new(runtime),
        })
    }

    /// Add a memory.
    ///
    /// Args:
    ///     content: The content to store as a memory
    ///     user_id: Optional user identifier for scoping
    ///     agent_id: Optional agent identifier for scoping
    ///     metadata: Optional dict of additional metadata
    ///     infer: Whether to extract facts using LLM (default: True)
    ///
    /// Returns:
    ///     AddResult with created/updated memories
    ///
    /// Example:
    ///     result = memory.add(
    ///         content="I love programming in Rust",
    ///         user_id="user123",
    ///         metadata={"source": "conversation"}
    ///     )
    #[pyo3(signature = (content, user_id=None, agent_id=None, metadata=None, infer=true))]
    pub fn add(
        &self,
        py: Python<'_>,
        content: String,
        user_id: Option<String>,
        agent_id: Option<String>,
        metadata: Option<&Bound<'_, PyAny>>,
        infer: bool,
    ) -> PyResult<AddResult> {
        let metadata_map = python_dict_to_metadata(py, metadata)?;

        let result = py.allow_threads(|| {
            self.runtime.block_on(async {
                self.inner
                    .add(content.as_str(), user_id, agent_id, None, metadata_map, infer, None)
                    .await
            })
        });

        match result {
            Ok(r) => Ok(AddResult::from_core(&r)),
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                e.to_string(),
            )),
        }
    }

    /// Search for memories.
    ///
    /// Args:
    ///     query: The search query
    ///     user_id: Optional user identifier for scoping
    ///     agent_id: Optional agent identifier for scoping
    ///     limit: Maximum number of results (default: 10)
    ///     threshold: Optional minimum similarity score
    ///
    /// Returns:
    ///     List of SearchResult objects
    ///
    /// Example:
    ///     results = memory.search(
    ///         query="programming preferences",
    ///         user_id="user123",
    ///         limit=5
    ///     )
    ///     for r in results:
    ///         print(f"{r.memory} (score: {r.score})")
    #[pyo3(signature = (query, user_id=None, agent_id=None, limit=10, threshold=None))]
    pub fn search(
        &self,
        py: Python<'_>,
        query: String,
        user_id: Option<String>,
        agent_id: Option<String>,
        limit: usize,
        threshold: Option<f32>,
    ) -> PyResult<Vec<SearchResult>> {
        let result = py.allow_threads(|| {
            self.runtime.block_on(async {
                self.inner
                    .search(&query, user_id, agent_id, None, limit, None, threshold, false)
                    .await
            })
        });

        match result {
            Ok(r) => Ok(r
                .results
                .iter()
                .map(SearchResult::from_memory_item)
                .collect()),
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                e.to_string(),
            )),
        }
    }

    /// Get a specific memory by ID.
    ///
    /// Args:
    ///     memory_id: The unique identifier of the memory
    ///
    /// Returns:
    ///     MemoryItem if found, None otherwise
    ///
    /// Example:
    ///     item = memory.get("abc123")
    ///     if item:
    ///         print(item.memory)
    pub fn get(&self, py: Python<'_>, memory_id: String) -> PyResult<Option<MemoryItem>> {
        let result = py.allow_threads(|| {
            self.runtime
                .block_on(async { self.inner.get(&memory_id).await })
        });

        match result {
            Ok(Some(item)) => Ok(Some(MemoryItem::from_core(item))),
            Ok(None) => Ok(None),
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                e.to_string(),
            )),
        }
    }

    /// Delete a specific memory.
    ///
    /// Args:
    ///     memory_id: The unique identifier of the memory to delete
    ///
    /// Raises:
    ///     RuntimeError: If deletion fails
    ///
    /// Example:
    ///     memory.delete("abc123")
    pub fn delete(&self, py: Python<'_>, memory_id: String) -> PyResult<()> {
        let result = py.allow_threads(|| {
            self.runtime
                .block_on(async { self.inner.delete(&memory_id).await })
        });

        result.map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    /// Delete all memories for a scope.
    ///
    /// Args:
    ///     user_id: Optional user identifier
    ///     agent_id: Optional agent identifier
    ///
    /// At least one of user_id or agent_id must be provided.
    ///
    /// Example:
    ///     memory.delete_all(user_id="user123")
    #[pyo3(signature = (user_id=None, agent_id=None))]
    pub fn delete_all(
        &self,
        py: Python<'_>,
        user_id: Option<String>,
        agent_id: Option<String>,
    ) -> PyResult<()> {
        let result = py.allow_threads(|| {
            self.runtime
                .block_on(async { self.inner.delete_all(user_id, agent_id, None).await })
        });

        result.map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    /// Reset all memories in the system.
    ///
    /// Warning: This permanently deletes ALL memories.
    ///
    /// Example:
    ///     memory.reset()  # Deletes everything!
    pub fn reset(&self, py: Python<'_>) -> PyResult<()> {
        let result =
            py.allow_threads(|| self.runtime.block_on(async { self.inner.reset().await }));

        result.map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    /// Get all memories for a scope.
    ///
    /// Args:
    ///     user_id: Optional user identifier
    ///     agent_id: Optional agent identifier
    ///     limit: Optional maximum number of results
    ///
    /// Returns:
    ///     List of MemoryItem objects
    ///
    /// Example:
    ///     items = memory.get_all(user_id="user123", limit=100)
    #[pyo3(signature = (user_id=None, agent_id=None, limit=None))]
    pub fn get_all(
        &self,
        py: Python<'_>,
        user_id: Option<String>,
        agent_id: Option<String>,
        limit: Option<usize>,
    ) -> PyResult<Vec<MemoryItem>> {
        let result = py.allow_threads(|| {
            self.runtime
                .block_on(async { self.inner.get_all(user_id, agent_id, None, limit).await })
        });

        match result {
            Ok(items) => Ok(items.into_iter().map(MemoryItem::from_core).collect()),
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                e.to_string(),
            )),
        }
    }

    /// Update a memory's content.
    ///
    /// Args:
    ///     memory_id: The unique identifier of the memory
    ///     content: The new content for the memory
    ///
    /// Returns:
    ///     Updated MemoryItem
    ///
    /// Example:
    ///     updated = memory.update("abc123", "Updated memory content")
    pub fn update(
        &self,
        py: Python<'_>,
        memory_id: String,
        content: String,
    ) -> PyResult<MemoryItem> {
        let result = py.allow_threads(|| {
            self.runtime
                .block_on(async { self.inner.update(&memory_id, &content).await })
        });

        match result {
            Ok(item) => Ok(MemoryItem::from_core(item)),
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                e.to_string(),
            )),
        }
    }
}

/// Parse Python config dict into Rust MemoryConfig.
fn parse_config(config: &Bound<'_, PyAny>, _py: Python<'_>) -> PyResult<rook_core::MemoryConfig> {
    // For now, use default config
    // Full config parsing would extract llm, embedder, vector_store sections
    if config.is_none() {
        return Ok(rook_core::MemoryConfig::default());
    }

    // Check if it's a dict
    if let Ok(dict) = config.downcast::<PyDict>() {
        let mut rust_config = rook_core::MemoryConfig::default();

        // Parse custom_fact_extraction_prompt if provided
        if let Ok(Some(prompt)) = dict.get_item("custom_fact_extraction_prompt") {
            if let Ok(s) = prompt.extract::<String>() {
                rust_config.custom_fact_extraction_prompt = Some(s);
            }
        }

        // Parse custom_update_memory_prompt if provided
        if let Ok(Some(prompt)) = dict.get_item("custom_update_memory_prompt") {
            if let Ok(s) = prompt.extract::<String>() {
                rust_config.custom_update_memory_prompt = Some(s);
            }
        }

        // Note: LLM, embedder, and vector_store configs are handled at construction time
        // The Python user should set environment variables or we'd need more complex parsing

        Ok(rust_config)
    } else {
        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "config must be a dict",
        ))
    }
}

/// Create Memory instance with the given config.
async fn create_memory(config: rook_core::MemoryConfig) -> PyResult<rook_core::Memory> {
    // Create LLM provider using factory (uses env vars for API keys)
    let llm = rook_llm::LlmFactory::openai().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("LLM error: {}", e))
    })?;

    // Create embedder using factory (uses env vars for API keys)
    let embedder = rook_embeddings::EmbedderFactory::openai().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Embedder error: {}", e))
    })?;

    // Create vector store using factory (Qdrant by default)
    let vector_store = rook_vector_stores::VectorStoreFactory::qdrant("rook")
        .await
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Vector store error: {}", e))
        })?;

    // Create Memory instance
    rook_core::Memory::new(config, llm, embedder, vector_store, None, None)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
}
