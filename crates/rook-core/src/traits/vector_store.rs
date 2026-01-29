//! Vector store trait and related types.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::error::RookResult;
use crate::types::Filter;

/// Distance metric for vector similarity.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum DistanceMetric {
    #[default]
    Cosine,
    Euclidean,
    DotProduct,
    Manhattan,
}

/// A vector record with payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorRecord {
    /// Unique identifier.
    pub id: String,
    /// Vector embedding.
    pub vector: Vec<f32>,
    /// Metadata payload.
    pub payload: HashMap<String, serde_json::Value>,
    /// Similarity score (from search).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score: Option<f32>,
}

impl VectorRecord {
    /// Create a new vector record.
    pub fn new(
        id: impl Into<String>,
        vector: Vec<f32>,
        payload: HashMap<String, serde_json::Value>,
    ) -> Self {
        Self {
            id: id.into(),
            vector,
            payload,
            score: None,
        }
    }

    /// Get a payload value as a string.
    pub fn get_string(&self, key: &str) -> Option<&str> {
        self.payload.get(key).and_then(|v| v.as_str())
    }

    /// Get the "data" field (memory content).
    pub fn get_data(&self) -> Option<&str> {
        self.get_string("data")
    }
}

/// Search result from vector store.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorSearchResult {
    /// Unique identifier.
    pub id: String,
    /// Similarity score.
    pub score: f32,
    /// Metadata payload.
    pub payload: HashMap<String, serde_json::Value>,
}

/// Collection information.
#[derive(Debug, Clone)]
pub struct CollectionInfo {
    /// Collection name.
    pub name: String,
    /// Number of vectors.
    pub vector_count: u64,
    /// Vector dimension.
    pub dimension: usize,
    /// Distance metric.
    pub distance: DistanceMetric,
}

/// Core VectorStore trait - all vector store backends implement this.
#[async_trait]
pub trait VectorStore: Send + Sync {
    /// Create a new collection.
    async fn create_collection(
        &self,
        name: &str,
        dimension: usize,
        distance: DistanceMetric,
    ) -> RookResult<()>;

    /// Insert vectors into the collection.
    async fn insert(&self, records: Vec<VectorRecord>) -> RookResult<()>;

    /// Search for similar vectors.
    async fn search(
        &self,
        query_vector: &[f32],
        limit: usize,
        filters: Option<Filter>,
    ) -> RookResult<Vec<VectorSearchResult>>;

    /// Get a vector by ID.
    async fn get(&self, id: &str) -> RookResult<Option<VectorRecord>>;

    /// Update a vector and/or its payload.
    async fn update(
        &self,
        id: &str,
        vector: Option<Vec<f32>>,
        payload: Option<HashMap<String, serde_json::Value>>,
    ) -> RookResult<()>;

    /// Delete a vector by ID.
    async fn delete(&self, id: &str) -> RookResult<()>;

    /// List vectors with optional filters.
    async fn list(
        &self,
        filters: Option<Filter>,
        limit: Option<usize>,
    ) -> RookResult<Vec<VectorRecord>>;

    /// List all collections.
    async fn list_collections(&self) -> RookResult<Vec<String>>;

    /// Delete a collection.
    async fn delete_collection(&self, name: &str) -> RookResult<()>;

    /// Get collection information.
    async fn collection_info(&self, name: &str) -> RookResult<CollectionInfo>;

    /// Reset (delete and recreate) the collection.
    async fn reset(&self) -> RookResult<()>;

    /// Get the collection name.
    fn collection_name(&self) -> &str;
}

/// Batch operations for high-throughput scenarios.
#[async_trait]
pub trait BatchVectorOperations: VectorStore {
    /// Batch insert with configurable batch size.
    async fn batch_insert(
        &self,
        records: Vec<VectorRecord>,
        batch_size: usize,
    ) -> RookResult<()>;

    /// Batch delete.
    async fn batch_delete(&self, ids: &[String]) -> RookResult<()>;

    /// Batch update.
    async fn batch_update(
        &self,
        updates: Vec<(String, Option<Vec<f32>>, Option<HashMap<String, serde_json::Value>>)>,
    ) -> RookResult<()>;
}

/// Hybrid search support (vector + keyword).
#[async_trait]
pub trait HybridSearch: VectorStore {
    /// Perform hybrid search combining vector and text search.
    async fn hybrid_search(
        &self,
        query_vector: &[f32],
        query_text: &str,
        limit: usize,
        filters: Option<Filter>,
        alpha: f32, // Weight between vector and text search
    ) -> RookResult<Vec<VectorSearchResult>>;
}

/// Vector store configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorStoreConfig {
    /// Provider type.
    pub provider: VectorStoreProvider,
    /// Collection name.
    pub collection_name: String,
    /// Embedding dimensions.
    #[serde(default = "default_embedding_dims")]
    pub embedding_model_dims: usize,
    /// Provider-specific configuration.
    #[serde(flatten)]
    pub config: serde_json::Value,
}

fn default_embedding_dims() -> usize {
    1536
}

impl Default for VectorStoreConfig {
    fn default() -> Self {
        Self {
            provider: VectorStoreProvider::Qdrant,
            collection_name: "rook".to_string(),
            embedding_model_dims: default_embedding_dims(),
            config: serde_json::json!({}),
        }
    }
}

/// Vector store provider type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum VectorStoreProvider {
    #[default]
    Qdrant,
    Pinecone,
    Pgvector,
    Redis,
    MongoDB,
    Elasticsearch,
    Opensearch,
    Chroma,
    Milvus,
    Weaviate,
    Faiss,
    AzureAiSearch,
    Supabase,
    UpstashVector,
    Databricks,
    VertexAiVectorSearch,
    Cassandra,
    Neptune,
    S3Vectors,
    Baidu,
    Valkey,
    AzureMysql,
    /// SQLite with sqlite-vec extension for embedded vector search.
    SqliteVec,
}
