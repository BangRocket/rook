//! Redis vector store implementation using RediSearch.

use async_trait::async_trait;
use std::collections::HashMap;

use rook_core::error::{RookError, RookResult};
use rook_core::traits::{VectorStore, VectorStoreConfig};
use rook_core::types::{CollectionInfo, DistanceMetric, Filter, VectorRecord, VectorSearchResult};

use redis::aio::MultiplexedConnection;
use redis::{AsyncCommands, Client, RedisResult};

/// Redis vector store implementation.
pub struct RedisVectorStore {
    client: Client,
    connection: MultiplexedConnection,
    config: VectorStoreConfig,
}

impl RedisVectorStore {
    /// Create a new Redis vector store.
    pub async fn new(config: VectorStoreConfig) -> RookResult<Self> {
        let url = config
            .url
            .clone()
            .unwrap_or_else(|| "redis://localhost:6379".to_string());

        let client = Client::open(url.as_str())
            .map_err(|e| RookError::vector_store(format!("Failed to create Redis client: {}", e)))?;

        let connection = client
            .get_multiplexed_async_connection()
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to connect to Redis: {}", e)))?;

        Ok(Self {
            client,
            connection,
            config,
        })
    }

    fn index_name(&self, collection_name: &str) -> String {
        format!("idx:{}", collection_name)
    }

    fn key_prefix(&self, collection_name: &str) -> String {
        format!("{}:", collection_name)
    }

    fn distance_to_redis(metric: &DistanceMetric) -> &'static str {
        match metric {
            DistanceMetric::Cosine => "COSINE",
            DistanceMetric::Euclidean => "L2",
            DistanceMetric::DotProduct => "IP",
        }
    }

    fn vector_to_bytes(vector: &[f32]) -> Vec<u8> {
        vector
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect()
    }

    fn bytes_to_vector(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks(4)
            .map(|chunk| {
                let arr: [u8; 4] = chunk.try_into().unwrap_or([0; 4]);
                f32::from_le_bytes(arr)
            })
            .collect()
    }
}

#[async_trait]
impl VectorStore for RedisVectorStore {
    async fn create_collection(
        &self,
        name: &str,
        dimension: usize,
        distance_metric: Option<DistanceMetric>,
    ) -> RookResult<()> {
        let mut conn = self.connection.clone();
        let index_name = self.index_name(name);
        let prefix = self.key_prefix(name);
        let metric = distance_metric.unwrap_or(DistanceMetric::Cosine);
        let distance = Self::distance_to_redis(&metric);

        // Create RediSearch index with vector field
        let result: RedisResult<()> = redis::cmd("FT.CREATE")
            .arg(&index_name)
            .arg("ON")
            .arg("HASH")
            .arg("PREFIX")
            .arg(1)
            .arg(&prefix)
            .arg("SCHEMA")
            .arg("vector")
            .arg("VECTOR")
            .arg("FLAT")
            .arg(6)
            .arg("TYPE")
            .arg("FLOAT32")
            .arg("DIM")
            .arg(dimension)
            .arg("DISTANCE_METRIC")
            .arg(distance)
            .arg("metadata")
            .arg("TEXT")
            .query_async(&mut conn)
            .await;

        match result {
            Ok(()) => Ok(()),
            Err(e) => {
                // Index might already exist
                if e.to_string().contains("Index already exists") {
                    Ok(())
                } else {
                    Err(RookError::vector_store(format!(
                        "Failed to create Redis index: {}",
                        e
                    )))
                }
            }
        }
    }

    async fn insert(&self, collection_name: &str, records: Vec<VectorRecord>) -> RookResult<()> {
        let mut conn = self.connection.clone();
        let prefix = self.key_prefix(collection_name);

        for record in records {
            let key = format!("{}{}", prefix, record.id);
            let vector_bytes = Self::vector_to_bytes(&record.vector);
            let metadata_json = serde_json::to_string(&record.metadata)
                .map_err(|e| RookError::vector_store(format!("Failed to serialize metadata: {}", e)))?;

            let _: () = redis::cmd("HSET")
                .arg(&key)
                .arg("vector")
                .arg(&vector_bytes)
                .arg("metadata")
                .arg(&metadata_json)
                .arg("id")
                .arg(&record.id)
                .query_async(&mut conn)
                .await
                .map_err(|e| RookError::vector_store(format!("Failed to insert vector: {}", e)))?;
        }

        Ok(())
    }

    async fn search(
        &self,
        collection_name: &str,
        query_vector: Vec<f32>,
        limit: usize,
        filter: Option<Filter>,
    ) -> RookResult<Vec<VectorSearchResult>> {
        let mut conn = self.connection.clone();
        let index_name = self.index_name(collection_name);
        let vector_bytes = Self::vector_to_bytes(&query_vector);

        // Build filter query
        let filter_query = if let Some(f) = filter {
            Self::build_filter_query(&f)
        } else {
            "*".to_string()
        };

        let query = format!(
            "({})=>[KNN {} @vector $vec AS score]",
            filter_query, limit
        );

        let result: Vec<(String, Vec<(String, String)>)> = redis::cmd("FT.SEARCH")
            .arg(&index_name)
            .arg(&query)
            .arg("PARAMS")
            .arg(2)
            .arg("vec")
            .arg(&vector_bytes)
            .arg("SORTBY")
            .arg("score")
            .arg("RETURN")
            .arg(3)
            .arg("id")
            .arg("metadata")
            .arg("score")
            .arg("DIALECT")
            .arg(2)
            .query_async(&mut conn)
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to search vectors: {}", e)))?;

        let mut results = Vec::new();
        for (key, fields) in result {
            let mut id = String::new();
            let mut metadata = HashMap::new();
            let mut score = 0.0f32;

            for (field_name, field_value) in fields {
                match field_name.as_str() {
                    "id" => id = field_value,
                    "metadata" => {
                        if let Ok(m) = serde_json::from_str(&field_value) {
                            metadata = m;
                        }
                    }
                    "score" => {
                        score = field_value.parse().unwrap_or(0.0);
                    }
                    _ => {}
                }
            }

            if id.is_empty() {
                id = key.split(':').last().unwrap_or(&key).to_string();
            }

            results.push(VectorSearchResult {
                id,
                score,
                vector: None,
                metadata,
            });
        }

        Ok(results)
    }

    async fn get(&self, collection_name: &str, id: &str) -> RookResult<Option<VectorRecord>> {
        let mut conn = self.connection.clone();
        let key = format!("{}{}", self.key_prefix(collection_name), id);

        let result: Option<HashMap<String, Vec<u8>>> = conn
            .hgetall(&key)
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to get vector: {}", e)))?;

        match result {
            Some(hash) if !hash.is_empty() => {
                let vector = hash
                    .get("vector")
                    .map(|v| Self::bytes_to_vector(v))
                    .unwrap_or_default();

                let metadata: HashMap<String, serde_json::Value> = hash
                    .get("metadata")
                    .and_then(|m| {
                        let s = String::from_utf8_lossy(m);
                        serde_json::from_str(&s).ok()
                    })
                    .unwrap_or_default();

                Ok(Some(VectorRecord {
                    id: id.to_string(),
                    vector,
                    metadata,
                }))
            }
            _ => Ok(None),
        }
    }

    async fn update(
        &self,
        collection_name: &str,
        id: &str,
        vector: Option<Vec<f32>>,
        metadata: Option<HashMap<String, serde_json::Value>>,
    ) -> RookResult<()> {
        let existing = self.get(collection_name, id).await?;
        let existing = existing.ok_or_else(|| RookError::not_found("Vector", id))?;

        let new_vector = vector.unwrap_or(existing.vector);
        let new_metadata = if let Some(m) = metadata {
            let mut merged = existing.metadata;
            merged.extend(m);
            merged
        } else {
            existing.metadata
        };

        let record = VectorRecord {
            id: id.to_string(),
            vector: new_vector,
            metadata: new_metadata,
        };

        self.insert(collection_name, vec![record]).await
    }

    async fn delete(&self, collection_name: &str, id: &str) -> RookResult<()> {
        let mut conn = self.connection.clone();
        let key = format!("{}{}", self.key_prefix(collection_name), id);

        let _: () = conn
            .del(&key)
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to delete vector: {}", e)))?;

        Ok(())
    }

    async fn delete_collection(&self, name: &str) -> RookResult<()> {
        let mut conn = self.connection.clone();
        let index_name = self.index_name(name);

        // Drop the index
        let _: RedisResult<()> = redis::cmd("FT.DROPINDEX")
            .arg(&index_name)
            .arg("DD") // Delete documents as well
            .query_async(&mut conn)
            .await;

        Ok(())
    }

    async fn list(
        &self,
        collection_name: &str,
        filter: Option<Filter>,
        limit: Option<usize>,
    ) -> RookResult<Vec<VectorRecord>> {
        let mut conn = self.connection.clone();
        let index_name = self.index_name(collection_name);

        let filter_query = if let Some(f) = filter {
            Self::build_filter_query(&f)
        } else {
            "*".to_string()
        };

        let limit_val = limit.unwrap_or(100);

        let result: Vec<(String, Vec<(String, Vec<u8>)>)> = redis::cmd("FT.SEARCH")
            .arg(&index_name)
            .arg(&filter_query)
            .arg("LIMIT")
            .arg(0)
            .arg(limit_val)
            .arg("RETURN")
            .arg(3)
            .arg("id")
            .arg("vector")
            .arg("metadata")
            .query_async(&mut conn)
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to list vectors: {}", e)))?;

        let mut records = Vec::new();
        for (_key, fields) in result {
            let mut id = String::new();
            let mut vector = Vec::new();
            let mut metadata = HashMap::new();

            for (field_name, field_value) in fields {
                match field_name.as_str() {
                    "id" => id = String::from_utf8_lossy(&field_value).to_string(),
                    "vector" => vector = Self::bytes_to_vector(&field_value),
                    "metadata" => {
                        let s = String::from_utf8_lossy(&field_value);
                        if let Ok(m) = serde_json::from_str(&s) {
                            metadata = m;
                        }
                    }
                    _ => {}
                }
            }

            if !id.is_empty() {
                records.push(VectorRecord {
                    id,
                    vector,
                    metadata,
                });
            }
        }

        Ok(records)
    }

    async fn collection_info(&self, name: &str) -> RookResult<CollectionInfo> {
        let mut conn = self.connection.clone();
        let index_name = self.index_name(name);

        let info: Vec<String> = redis::cmd("FT.INFO")
            .arg(&index_name)
            .query_async(&mut conn)
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to get index info: {}", e)))?;

        // Parse info to get count
        let mut count = 0usize;
        for i in 0..info.len() {
            if info[i] == "num_docs" && i + 1 < info.len() {
                count = info[i + 1].parse().unwrap_or(0);
                break;
            }
        }

        Ok(CollectionInfo {
            name: name.to_string(),
            dimension: self.config.embedding_dims.unwrap_or(1536),
            count,
            distance_metric: DistanceMetric::Cosine,
        })
    }

    async fn reset(&self, collection_name: &str) -> RookResult<()> {
        let info = self.collection_info(collection_name).await?;
        self.delete_collection(collection_name).await?;
        self.create_collection(
            collection_name,
            info.dimension,
            Some(info.distance_metric),
        )
        .await?;
        Ok(())
    }
}

impl RedisVectorStore {
    fn build_filter_query(filter: &Filter) -> String {
        let conditions: Vec<String> = filter
            .conditions
            .iter()
            .filter_map(|cond| {
                match &cond.operator {
                    rook_core::types::FilterOperator::Eq => {
                        let value = match &cond.value {
                            serde_json::Value::String(s) => format!("\"{}\"", s),
                            serde_json::Value::Number(n) => n.to_string(),
                            serde_json::Value::Bool(b) => b.to_string(),
                            _ => return None,
                        };
                        Some(format!("@{}:{}", cond.field, value))
                    }
                    _ => None,
                }
            })
            .collect();

        if conditions.is_empty() {
            "*".to_string()
        } else {
            conditions.join(" ")
        }
    }
}
