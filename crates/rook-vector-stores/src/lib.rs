//! rook-vector-stores - Vector store implementations for rook.
//!
//! This crate provides vector store implementations for use with
//! the rook memory layer.
//!
//! # Supported Backends
//!
//! ## Tier 1 - Native Rust Crates
//! - **Qdrant** (feature: `qdrant`) - High-performance vector database
//! - **Redis** (feature: `redis`) - Redis with vector search
//! - **Elasticsearch** (feature: `elasticsearch`) - Elasticsearch vector search
//! - **OpenSearch** (feature: `opensearch`) - OpenSearch vector search
//! - **MongoDB** (feature: `mongodb`) - MongoDB Atlas Vector Search
//!
//! ## Tier 2 - SQL-Based
//! - **pgvector** (feature: `pgvector`) - PostgreSQL with pgvector extension
//! - **MySQL** (feature: `mysql`) - Azure MySQL with vector support
//! - **SQLite** (feature: `sqlite`) - SQLite with vector extensions
//!
//! ## Tier 3 - REST API Backends
//! - **Pinecone** (feature: `pinecone`) - Pinecone vector database
//! - **Weaviate** (feature: `weaviate`) - Weaviate vector database
//! - **Chroma** (feature: `chroma`) - Chroma embedding database
//! - **Milvus** (feature: `milvus`) - Milvus vector database
//! - **Upstash** (feature: `upstash`) - Upstash Vector
//!
//! ## Tier 4 - Cloud Backends
//! - **Azure AI Search** (feature: `azure-ai-search`)
//! - **Vertex AI** (feature: `vertex-ai`)
//! - **Supabase** (feature: `supabase`)
//! - **S3 Vectors** (feature: `s3-vectors`)
//! - **Neptune** (feature: `neptune`)
//! - **Databricks** (feature: `databricks`)
//!
//! ## Tier 5 - Other Backends
//! - **LanceDB** (feature: `lancedb`)
//! - **DuckDB** (feature: `duckdb`)
//! - **Cassandra** (feature: `cassandra`)
//! - **Valkey** (feature: `valkey`)
//! - **FAISS** (feature: `faiss`)

mod factory;

#[cfg(feature = "qdrant")]
mod qdrant;

#[cfg(feature = "redis")]
mod redis_store;

#[cfg(feature = "elasticsearch")]
mod elasticsearch;

#[cfg(feature = "opensearch")]
mod opensearch;

#[cfg(feature = "mongodb")]
mod mongodb;

#[cfg(feature = "pgvector")]
mod pgvector;

#[cfg(feature = "mysql")]
mod mysql;

#[cfg(feature = "sqlite")]
mod sqlite;

#[cfg(feature = "pinecone")]
mod pinecone;

#[cfg(feature = "weaviate")]
mod weaviate;

#[cfg(feature = "chroma")]
mod chroma;

#[cfg(feature = "milvus")]
mod milvus;

#[cfg(feature = "upstash")]
mod upstash;

#[cfg(feature = "azure-ai-search")]
mod azure_ai_search;

#[cfg(feature = "vertex-ai")]
mod vertex_ai;

#[cfg(feature = "supabase")]
mod supabase;

#[cfg(feature = "lancedb")]
mod lancedb;

#[cfg(feature = "duckdb")]
mod duckdb;

#[cfg(feature = "cassandra")]
mod cassandra;

#[cfg(feature = "valkey")]
mod valkey;

#[cfg(feature = "s3-vectors")]
mod s3_vectors;

#[cfg(feature = "neptune")]
mod neptune;

#[cfg(feature = "databricks")]
mod databricks;

#[cfg(feature = "faiss")]
mod faiss;

#[cfg(feature = "sqlite-vec")]
mod sqlite_vec;

// Public exports
pub use factory::VectorStoreFactory;

#[cfg(feature = "qdrant")]
pub use qdrant::QdrantVectorStore;

#[cfg(feature = "redis")]
pub use redis_store::RedisVectorStore;

#[cfg(feature = "elasticsearch")]
pub use elasticsearch::ElasticsearchVectorStore;

#[cfg(feature = "opensearch")]
pub use opensearch::OpenSearchVectorStore;

#[cfg(feature = "mongodb")]
pub use mongodb::MongoDBVectorStore;

#[cfg(feature = "pgvector")]
pub use pgvector::PgVectorStore;

#[cfg(feature = "mysql")]
pub use mysql::MySQLVectorStore;

#[cfg(feature = "sqlite")]
pub use sqlite::SQLiteVectorStore;

#[cfg(feature = "pinecone")]
pub use pinecone::PineconeVectorStore;

#[cfg(feature = "weaviate")]
pub use weaviate::WeaviateVectorStore;

#[cfg(feature = "chroma")]
pub use chroma::ChromaVectorStore;

#[cfg(feature = "milvus")]
pub use milvus::MilvusVectorStore;

#[cfg(feature = "upstash")]
pub use upstash::UpstashVectorStore;

#[cfg(feature = "azure-ai-search")]
pub use azure_ai_search::AzureAISearchVectorStore;

#[cfg(feature = "vertex-ai")]
pub use vertex_ai::VertexAIVectorStore;

#[cfg(feature = "supabase")]
pub use supabase::SupabaseVectorStore;

#[cfg(feature = "lancedb")]
pub use lancedb::LanceDBVectorStore;

#[cfg(feature = "duckdb")]
pub use duckdb::DuckDBVectorStore;

#[cfg(feature = "cassandra")]
pub use cassandra::CassandraVectorStore;

#[cfg(feature = "valkey")]
pub use valkey::ValkeyVectorStore;

#[cfg(feature = "s3-vectors")]
pub use s3_vectors::S3VectorStore;

#[cfg(feature = "neptune")]
pub use neptune::NeptuneVectorStore;

#[cfg(feature = "databricks")]
pub use databricks::DatabricksVectorStore;

#[cfg(feature = "faiss")]
pub use faiss::FaissVectorStore;

#[cfg(feature = "sqlite-vec")]
pub use sqlite_vec::SqliteVecStore;

// Re-export core types for convenience
pub use rook_core::traits::{
    CollectionInfo, DistanceMetric, VectorRecord, VectorSearchResult, VectorStore,
    VectorStoreConfig, VectorStoreProvider,
};
