//! Factory for creating vector store providers.

use std::sync::Arc;

use rook_core::error::{RookError, RookResult};
use rook_core::traits::{VectorStore, VectorStoreConfig, VectorStoreProvider};
use serde_json::json;

/// Factory for creating vector store providers.
pub struct VectorStoreFactory;

impl VectorStoreFactory {
    /// Create a vector store from the given configuration.
    pub async fn create(
        provider: VectorStoreProvider,
        config: VectorStoreConfig,
    ) -> RookResult<Arc<dyn VectorStore>> {
        match provider {
            #[cfg(feature = "qdrant")]
            VectorStoreProvider::Qdrant => {
                let store = crate::qdrant::QdrantVectorStore::new(config).await?;
                Ok(Arc::new(store))
            }

            #[cfg(feature = "redis")]
            VectorStoreProvider::Redis => {
                let store = crate::redis_store::RedisVectorStore::new(config).await?;
                Ok(Arc::new(store))
            }

            #[cfg(feature = "elasticsearch")]
            VectorStoreProvider::Elasticsearch => {
                let store = crate::elasticsearch::ElasticsearchVectorStore::new(config).await?;
                Ok(Arc::new(store))
            }

            #[cfg(feature = "opensearch")]
            VectorStoreProvider::Opensearch => {
                let store = crate::opensearch::OpenSearchVectorStore::new(config).await?;
                Ok(Arc::new(store))
            }

            #[cfg(feature = "mongodb")]
            VectorStoreProvider::MongoDB => {
                let store = crate::mongodb::MongoDBVectorStore::new(config).await?;
                Ok(Arc::new(store))
            }

            #[cfg(feature = "pgvector")]
            VectorStoreProvider::Pgvector => {
                let store = crate::pgvector::PgVectorStore::new(config).await?;
                Ok(Arc::new(store))
            }

            #[cfg(feature = "mysql")]
            VectorStoreProvider::AzureMysql => {
                let store = crate::mysql::MySQLVectorStore::new(config).await?;
                Ok(Arc::new(store))
            }

            #[cfg(feature = "sqlite")]
            VectorStoreProvider::Faiss => {
                // SQLite used as fallback for simple/faiss storage
                let store = crate::sqlite::SQLiteVectorStore::new(config).await?;
                Ok(Arc::new(store))
            }

            #[cfg(feature = "pinecone")]
            VectorStoreProvider::Pinecone => {
                let store = crate::pinecone::PineconeVectorStore::new(config).await?;
                Ok(Arc::new(store))
            }

            #[cfg(feature = "weaviate")]
            VectorStoreProvider::Weaviate => {
                let store = crate::weaviate::WeaviateVectorStore::new(config).await?;
                Ok(Arc::new(store))
            }

            #[cfg(feature = "chroma")]
            VectorStoreProvider::Chroma => {
                let store = crate::chroma::ChromaVectorStore::new(config).await?;
                Ok(Arc::new(store))
            }

            #[cfg(feature = "milvus")]
            VectorStoreProvider::Milvus => {
                let store = crate::milvus::MilvusVectorStore::new(config).await?;
                Ok(Arc::new(store))
            }

            #[cfg(feature = "upstash")]
            VectorStoreProvider::UpstashVector => {
                let store = crate::upstash::UpstashVectorStore::new(config).await?;
                Ok(Arc::new(store))
            }

            #[cfg(feature = "azure-ai-search")]
            VectorStoreProvider::AzureAiSearch => {
                let store = crate::azure_ai_search::AzureAISearchVectorStore::new(config).await?;
                Ok(Arc::new(store))
            }

            #[cfg(feature = "vertex-ai")]
            VectorStoreProvider::VertexAiVectorSearch => {
                let store = crate::vertex_ai::VertexAIVectorStore::new(config).await?;
                Ok(Arc::new(store))
            }

            #[cfg(feature = "supabase")]
            VectorStoreProvider::Supabase => {
                let store = crate::supabase::SupabaseVectorStore::new(config).await?;
                Ok(Arc::new(store))
            }

            #[cfg(feature = "cassandra")]
            VectorStoreProvider::Cassandra => {
                let store = crate::cassandra::CassandraVectorStore::new(config).await?;
                Ok(Arc::new(store))
            }

            #[cfg(feature = "valkey")]
            VectorStoreProvider::Valkey => {
                let store = crate::valkey::ValkeyVectorStore::new(config).await?;
                Ok(Arc::new(store))
            }

            #[cfg(feature = "s3-vectors")]
            VectorStoreProvider::S3Vectors => {
                let store = crate::s3_vectors::S3VectorStore::new(config).await?;
                Ok(Arc::new(store))
            }

            #[cfg(feature = "neptune")]
            VectorStoreProvider::Neptune => {
                let store = crate::neptune::NeptuneVectorStore::new(config).await?;
                Ok(Arc::new(store))
            }

            #[cfg(feature = "databricks")]
            VectorStoreProvider::Databricks => {
                let store = crate::databricks::DatabricksVectorStore::new(config).await?;
                Ok(Arc::new(store))
            }

            #[cfg(feature = "sqlite-vec")]
            VectorStoreProvider::SqliteVec => {
                let db_path = config
                    .config
                    .get("path")
                    .and_then(|v| v.as_str())
                    .unwrap_or(":memory:");
                let store = crate::sqlite_vec::SqliteVecStore::new(
                    db_path,
                    &config.collection_name,
                    config.embedding_model_dims,
                )?;
                Ok(Arc::new(store))
            }

            #[allow(unreachable_patterns)]
            _ => Err(RookError::UnsupportedProvider {
                provider: format!("{:?}", provider),
            }),
        }
    }

    /// Create a Qdrant vector store with default configuration.
    #[cfg(feature = "qdrant")]
    pub async fn qdrant(collection_name: &str) -> RookResult<Arc<dyn VectorStore>> {
        let config = VectorStoreConfig {
            provider: VectorStoreProvider::Qdrant,
            collection_name: collection_name.to_string(),
            embedding_model_dims: 1536,
            pool: None,
            config: json!({}),
        };
        Self::create(VectorStoreProvider::Qdrant, config).await
    }

    /// Create a Qdrant vector store with custom URL.
    #[cfg(feature = "qdrant")]
    pub async fn qdrant_with_url(
        collection_name: &str,
        url: &str,
    ) -> RookResult<Arc<dyn VectorStore>> {
        let config = VectorStoreConfig {
            provider: VectorStoreProvider::Qdrant,
            collection_name: collection_name.to_string(),
            embedding_model_dims: 1536,
            pool: None,
            config: json!({
                "url": url
            }),
        };
        Self::create(VectorStoreProvider::Qdrant, config).await
    }

    /// Create a Redis vector store.
    #[cfg(feature = "redis")]
    pub async fn redis(collection_name: &str, url: &str) -> RookResult<Arc<dyn VectorStore>> {
        let config = VectorStoreConfig {
            provider: VectorStoreProvider::Redis,
            collection_name: collection_name.to_string(),
            embedding_model_dims: 1536,
            pool: None,
            config: json!({
                "url": url
            }),
        };
        Self::create(VectorStoreProvider::Redis, config).await
    }

    /// Create a Pinecone vector store.
    #[cfg(feature = "pinecone")]
    pub async fn pinecone(
        index_name: &str,
        api_key: &str,
        environment: &str,
    ) -> RookResult<Arc<dyn VectorStore>> {
        let config = VectorStoreConfig {
            provider: VectorStoreProvider::Pinecone,
            collection_name: index_name.to_string(),
            embedding_model_dims: 1536,
            pool: None,
            config: json!({
                "api_key": api_key,
                "environment": environment
            }),
        };
        Self::create(VectorStoreProvider::Pinecone, config).await
    }

    /// Create a pgvector store.
    #[cfg(feature = "pgvector")]
    pub async fn pgvector(
        collection_name: &str,
        connection_string: &str,
    ) -> RookResult<Arc<dyn VectorStore>> {
        let config = VectorStoreConfig {
            provider: VectorStoreProvider::Pgvector,
            collection_name: collection_name.to_string(),
            embedding_model_dims: 1536,
            pool: None,
            config: json!({
                "url": connection_string
            }),
        };
        Self::create(VectorStoreProvider::Pgvector, config).await
    }

    /// Create a Chroma vector store.
    #[cfg(feature = "chroma")]
    pub async fn chroma(collection_name: &str, url: &str) -> RookResult<Arc<dyn VectorStore>> {
        let config = VectorStoreConfig {
            provider: VectorStoreProvider::Chroma,
            collection_name: collection_name.to_string(),
            embedding_model_dims: 1536,
            pool: None,
            config: json!({
                "url": url
            }),
        };
        Self::create(VectorStoreProvider::Chroma, config).await
    }

    /// Create an in-memory sqlite-vec vector store.
    ///
    /// This is ideal for testing and development. For persistence,
    /// use `sqlite_vec_with_path` instead.
    #[cfg(feature = "sqlite-vec")]
    pub fn sqlite_vec_memory(
        collection_name: &str,
        dimension: usize,
    ) -> RookResult<Arc<dyn VectorStore>> {
        let store = crate::sqlite_vec::SqliteVecStore::new(":memory:", collection_name, dimension)?;
        Ok(Arc::new(store))
    }

    /// Create a persistent sqlite-vec vector store.
    ///
    /// # Arguments
    ///
    /// * `collection_name` - Name of the collection
    /// * `db_path` - Path to SQLite database file
    /// * `dimension` - Vector dimension
    #[cfg(feature = "sqlite-vec")]
    pub fn sqlite_vec_with_path(
        collection_name: &str,
        db_path: &str,
        dimension: usize,
    ) -> RookResult<Arc<dyn VectorStore>> {
        let store = crate::sqlite_vec::SqliteVecStore::new(db_path, collection_name, dimension)?;
        Ok(Arc::new(store))
    }
}
