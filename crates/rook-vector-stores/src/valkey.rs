//! Valkey vector store implementation (Redis-compatible).
//! Valkey is a Redis fork with vector search capabilities.

use async_trait::async_trait;
use std::collections::HashMap;

use rook_core::error::{RookError, RookResult};
use rook_core::traits::{VectorStore, VectorStoreConfig};
use rook_core::types::{CollectionInfo, DistanceMetric, Filter, VectorRecord, VectorSearchResult};

use redis::aio::MultiplexedConnection;
use redis::{AsyncCommands, Client};

/// Valkey vector store implementation (Redis-compatible).
pub struct ValkeyVectorStore {
    client: Client,
    connection: MultiplexedConnection,
    config: VectorStoreConfig,
}

impl ValkeyVectorStore {
    pub async fn new(config: VectorStoreConfig) -> RookResult<Self> {
        let url = config.url.clone().unwrap_or_else(|| "redis://localhost:6379".to_string());
        let client = Client::open(url.as_str()).map_err(|e| RookError::vector_store(format!("Failed to create Valkey client: {}", e)))?;
        let connection = client.get_multiplexed_async_connection().await.map_err(|e| RookError::vector_store(format!("Failed to connect: {}", e)))?;
        Ok(Self { client, connection, config })
    }

    fn index_name(&self, collection_name: &str) -> String { format!("idx:{}", collection_name) }
    fn key_prefix(&self, collection_name: &str) -> String { format!("{}:", collection_name) }
    fn vector_to_bytes(vector: &[f32]) -> Vec<u8> { vector.iter().flat_map(|f| f.to_le_bytes()).collect() }
    fn bytes_to_vector(bytes: &[u8]) -> Vec<f32> { bytes.chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap_or([0; 4]))).collect() }
}

#[async_trait]
impl VectorStore for ValkeyVectorStore {
    async fn create_collection(&self, name: &str, dimension: usize, _distance_metric: Option<DistanceMetric>) -> RookResult<()> {
        let mut conn = self.connection.clone();
        let index_name = self.index_name(name);
        let prefix = self.key_prefix(name);

        let _: redis::RedisResult<()> = redis::cmd("FT.CREATE").arg(&index_name).arg("ON").arg("HASH").arg("PREFIX").arg(1).arg(&prefix)
            .arg("SCHEMA").arg("vector").arg("VECTOR").arg("FLAT").arg(6).arg("TYPE").arg("FLOAT32").arg("DIM").arg(dimension).arg("DISTANCE_METRIC").arg("COSINE")
            .arg("metadata").arg("TEXT").query_async(&mut conn).await;
        Ok(())
    }

    async fn insert(&self, collection_name: &str, records: Vec<VectorRecord>) -> RookResult<()> {
        let mut conn = self.connection.clone();
        let prefix = self.key_prefix(collection_name);
        for record in records {
            let key = format!("{}{}", prefix, record.id);
            let vector_bytes = Self::vector_to_bytes(&record.vector);
            let metadata_json = serde_json::to_string(&record.metadata).unwrap_or_default();
            let _: () = redis::cmd("HSET").arg(&key).arg("vector").arg(&vector_bytes).arg("metadata").arg(&metadata_json).arg("id").arg(&record.id)
                .query_async(&mut conn).await.map_err(|e| RookError::vector_store(format!("Insert failed: {}", e)))?;
        }
        Ok(())
    }

    async fn search(&self, collection_name: &str, query_vector: Vec<f32>, limit: usize, _filter: Option<Filter>) -> RookResult<Vec<VectorSearchResult>> {
        let mut conn = self.connection.clone();
        let index_name = self.index_name(collection_name);
        let vector_bytes = Self::vector_to_bytes(&query_vector);
        let query = format!("(*)=>[KNN {} @vector $vec AS score]", limit);

        let result: Vec<(String, Vec<(String, String)>)> = redis::cmd("FT.SEARCH").arg(&index_name).arg(&query).arg("PARAMS").arg(2).arg("vec").arg(&vector_bytes)
            .arg("SORTBY").arg("score").arg("RETURN").arg(3).arg("id").arg("metadata").arg("score").arg("DIALECT").arg(2)
            .query_async(&mut conn).await.map_err(|e| RookError::vector_store(format!("Search failed: {}", e)))?;

        let results = result.into_iter().map(|(key, fields)| {
            let mut id = String::new(); let mut metadata = HashMap::new(); let mut score = 0.0f32;
            for (name, value) in fields {
                match name.as_str() {
                    "id" => id = value,
                    "metadata" => { if let Ok(m) = serde_json::from_str(&value) { metadata = m; } }
                    "score" => { score = value.parse().unwrap_or(0.0); }
                    _ => {}
                }
            }
            if id.is_empty() { id = key.split(':').last().unwrap_or(&key).to_string(); }
            VectorSearchResult { id, score, vector: None, metadata }
        }).collect();
        Ok(results)
    }

    async fn get(&self, collection_name: &str, id: &str) -> RookResult<Option<VectorRecord>> {
        let mut conn = self.connection.clone();
        let key = format!("{}{}", self.key_prefix(collection_name), id);
        let result: Option<HashMap<String, Vec<u8>>> = conn.hgetall(&key).await.map_err(|e| RookError::vector_store(format!("Get failed: {}", e)))?;
        Ok(result.filter(|h| !h.is_empty()).map(|hash| {
            let vector = hash.get("vector").map(|v| Self::bytes_to_vector(v)).unwrap_or_default();
            let metadata: HashMap<String, serde_json::Value> = hash.get("metadata").and_then(|m| serde_json::from_str(&String::from_utf8_lossy(m)).ok()).unwrap_or_default();
            VectorRecord { id: id.to_string(), vector, metadata }
        }))
    }

    async fn update(&self, collection_name: &str, id: &str, vector: Option<Vec<f32>>, metadata: Option<HashMap<String, serde_json::Value>>) -> RookResult<()> {
        let existing = self.get(collection_name, id).await?.ok_or_else(|| RookError::not_found("Vector", id))?;
        let record = VectorRecord { id: id.to_string(), vector: vector.unwrap_or(existing.vector), metadata: { let mut m = existing.metadata; if let Some(meta) = metadata { m.extend(meta); } m } };
        self.insert(collection_name, vec![record]).await
    }

    async fn delete(&self, collection_name: &str, id: &str) -> RookResult<()> {
        let mut conn = self.connection.clone();
        let key = format!("{}{}", self.key_prefix(collection_name), id);
        let _: () = conn.del(&key).await.map_err(|e| RookError::vector_store(format!("Delete failed: {}", e)))?;
        Ok(())
    }

    async fn delete_collection(&self, name: &str) -> RookResult<()> {
        let mut conn = self.connection.clone();
        let _: redis::RedisResult<()> = redis::cmd("FT.DROPINDEX").arg(self.index_name(name)).arg("DD").query_async(&mut conn).await;
        Ok(())
    }

    async fn list(&self, collection_name: &str, _filter: Option<Filter>, limit: Option<usize>) -> RookResult<Vec<VectorRecord>> {
        let mut conn = self.connection.clone();
        let result: Vec<(String, Vec<(String, Vec<u8>)>)> = redis::cmd("FT.SEARCH").arg(self.index_name(collection_name)).arg("*").arg("LIMIT").arg(0).arg(limit.unwrap_or(100))
            .arg("RETURN").arg(3).arg("id").arg("vector").arg("metadata").query_async(&mut conn).await.map_err(|e| RookError::vector_store(format!("List failed: {}", e)))?;
        Ok(result.into_iter().filter_map(|(_, fields)| {
            let mut id = String::new(); let mut vector = Vec::new(); let mut metadata = HashMap::new();
            for (name, value) in fields {
                match name.as_str() {
                    "id" => id = String::from_utf8_lossy(&value).to_string(),
                    "vector" => vector = Self::bytes_to_vector(&value),
                    "metadata" => { if let Ok(m) = serde_json::from_str(&String::from_utf8_lossy(&value)) { metadata = m; } }
                    _ => {}
                }
            }
            if id.is_empty() { None } else { Some(VectorRecord { id, vector, metadata }) }
        }).collect())
    }

    async fn collection_info(&self, name: &str) -> RookResult<CollectionInfo> {
        let mut conn = self.connection.clone();
        let info: Vec<String> = redis::cmd("FT.INFO").arg(self.index_name(name)).query_async(&mut conn).await.map_err(|e| RookError::vector_store(format!("Info failed: {}", e)))?;
        let count = info.iter().position(|s| s == "num_docs").and_then(|i| info.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(0);
        Ok(CollectionInfo { name: name.to_string(), dimension: self.config.embedding_dims.unwrap_or(1536), count, distance_metric: DistanceMetric::Cosine })
    }

    async fn reset(&self, collection_name: &str) -> RookResult<()> {
        let info = self.collection_info(collection_name).await?;
        self.delete_collection(collection_name).await?;
        self.create_collection(collection_name, info.dimension, Some(info.distance_metric)).await
    }
}
