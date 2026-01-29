//! Pinecone vector store implementation.

use async_trait::async_trait;
use std::collections::HashMap;

use rook_core::error::{RookError, RookResult};
use rook_core::traits::{VectorStore, VectorStoreConfig};
use rook_core::types::{CollectionInfo, DistanceMetric, Filter, VectorRecord, VectorSearchResult};

use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::json;

/// Pinecone vector store implementation.
pub struct PineconeVectorStore {
    client: Client,
    api_key: String,
    index_host: String,
    config: VectorStoreConfig,
}

#[derive(Debug, Serialize, Deserialize)]
struct PineconeVector {
    id: String,
    values: Vec<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    metadata: Option<HashMap<String, serde_json::Value>>,
}

#[derive(Debug, Deserialize)]
struct PineconeMatch {
    id: String,
    score: f32,
    #[serde(default)]
    values: Option<Vec<f32>>,
    #[serde(default)]
    metadata: Option<HashMap<String, serde_json::Value>>,
}

#[derive(Debug, Deserialize)]
struct PineconeQueryResponse {
    matches: Vec<PineconeMatch>,
}

#[derive(Debug, Deserialize)]
struct PineconeDescribeResponse {
    dimension: usize,
    index_fullness: f32,
    total_vector_count: usize,
}

impl PineconeVectorStore {
    /// Create a new Pinecone vector store.
    pub async fn new(config: VectorStoreConfig) -> RookResult<Self> {
        let api_key = config
            .api_key
            .clone()
            .or_else(|| std::env::var("PINECONE_API_KEY").ok())
            .ok_or_else(|| {
                RookError::Configuration(
                    "Pinecone API key required. Set PINECONE_API_KEY or provide api_key.".to_string(),
                )
            })?;

        let environment = config.environment.clone().ok_or_else(|| {
            RookError::Configuration("Pinecone environment required".to_string())
        })?;

        // Construct the index host
        let index_name = &config.collection_name;
        let index_host = config.url.clone().unwrap_or_else(|| {
            format!("https://{}-{}.svc.{}.pinecone.io", index_name, environment.split('-').next().unwrap_or(&environment), environment)
        });

        let client = Client::new();

        Ok(Self {
            client,
            api_key,
            index_host,
            config,
        })
    }

    fn headers(&self) -> reqwest::header::HeaderMap {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            "Api-Key",
            self.api_key.parse().unwrap(),
        );
        headers.insert(
            reqwest::header::CONTENT_TYPE,
            "application/json".parse().unwrap(),
        );
        headers
    }
}

#[async_trait]
impl VectorStore for PineconeVectorStore {
    async fn create_collection(
        &self,
        _name: &str,
        _dimension: usize,
        _distance_metric: Option<DistanceMetric>,
    ) -> RookResult<()> {
        // Pinecone indexes are created via the console or control plane API
        // This is a no-op for data plane operations
        tracing::info!("Pinecone indexes must be created via console or control plane API");
        Ok(())
    }

    async fn insert(&self, _collection_name: &str, records: Vec<VectorRecord>) -> RookResult<()> {
        if records.is_empty() {
            return Ok(());
        }

        let vectors: Vec<PineconeVector> = records
            .into_iter()
            .map(|r| PineconeVector {
                id: r.id,
                values: r.vector,
                metadata: if r.metadata.is_empty() {
                    None
                } else {
                    Some(r.metadata)
                },
            })
            .collect();

        let url = format!("{}/vectors/upsert", self.index_host);
        let body = json!({ "vectors": vectors });

        let response = self
            .client
            .post(&url)
            .headers(self.headers())
            .json(&body)
            .send()
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to upsert vectors: {}", e)))?;

        if !response.status().is_success() {
            let error = response.text().await.unwrap_or_default();
            return Err(RookError::vector_store(format!(
                "Failed to upsert vectors: {}",
                error
            )));
        }

        Ok(())
    }

    async fn search(
        &self,
        _collection_name: &str,
        query_vector: Vec<f32>,
        limit: usize,
        filter: Option<Filter>,
    ) -> RookResult<Vec<VectorSearchResult>> {
        let url = format!("{}/query", self.index_host);

        let mut body = json!({
            "vector": query_vector,
            "topK": limit,
            "includeMetadata": true
        });

        if let Some(f) = filter {
            body["filter"] = Self::build_filter(&f);
        }

        let response = self
            .client
            .post(&url)
            .headers(self.headers())
            .json(&body)
            .send()
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to query: {}", e)))?;

        if !response.status().is_success() {
            let error = response.text().await.unwrap_or_default();
            return Err(RookError::vector_store(format!("Failed to query: {}", error)));
        }

        let result: PineconeQueryResponse = response
            .json()
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to parse response: {}", e)))?;

        let results = result
            .matches
            .into_iter()
            .map(|m| VectorSearchResult {
                id: m.id,
                score: m.score,
                vector: m.values,
                metadata: m.metadata.unwrap_or_default(),
            })
            .collect();

        Ok(results)
    }

    async fn get(&self, _collection_name: &str, id: &str) -> RookResult<Option<VectorRecord>> {
        let url = format!("{}/vectors/fetch?ids={}", self.index_host, id);

        let response = self
            .client
            .get(&url)
            .headers(self.headers())
            .send()
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to fetch vector: {}", e)))?;

        if !response.status().is_success() {
            let error = response.text().await.unwrap_or_default();
            return Err(RookError::vector_store(format!(
                "Failed to fetch vector: {}",
                error
            )));
        }

        let body: serde_json::Value = response
            .json()
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to parse response: {}", e)))?;

        let vectors = body["vectors"].as_object();
        if let Some(vectors) = vectors {
            if let Some(v) = vectors.get(id) {
                let values: Vec<f32> = v["values"]
                    .as_array()
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_f64().map(|f| f as f32))
                            .collect()
                    })
                    .unwrap_or_default();

                let metadata: HashMap<String, serde_json::Value> = v["metadata"]
                    .as_object()
                    .map(|m| m.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
                    .unwrap_or_default();

                return Ok(Some(VectorRecord {
                    id: id.to_string(),
                    vector: values,
                    metadata,
                }));
            }
        }

        Ok(None)
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

    async fn delete(&self, _collection_name: &str, id: &str) -> RookResult<()> {
        let url = format!("{}/vectors/delete", self.index_host);
        let body = json!({ "ids": [id] });

        let response = self
            .client
            .post(&url)
            .headers(self.headers())
            .json(&body)
            .send()
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to delete vector: {}", e)))?;

        if !response.status().is_success() {
            let error = response.text().await.unwrap_or_default();
            return Err(RookError::vector_store(format!(
                "Failed to delete vector: {}",
                error
            )));
        }

        Ok(())
    }

    async fn delete_collection(&self, _name: &str) -> RookResult<()> {
        // Delete all vectors in the namespace
        let url = format!("{}/vectors/delete", self.index_host);
        let body = json!({ "deleteAll": true });

        let response = self
            .client
            .post(&url)
            .headers(self.headers())
            .json(&body)
            .send()
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to delete all vectors: {}", e)))?;

        if !response.status().is_success() {
            let error = response.text().await.unwrap_or_default();
            return Err(RookError::vector_store(format!(
                "Failed to delete all vectors: {}",
                error
            )));
        }

        Ok(())
    }

    async fn list(
        &self,
        collection_name: &str,
        filter: Option<Filter>,
        limit: Option<usize>,
    ) -> RookResult<Vec<VectorRecord>> {
        // Pinecone doesn't have a direct list operation
        // We use a search with a zero vector to get all vectors
        let dimension = self.config.embedding_dims.unwrap_or(1536);
        let zero_vector = vec![0.0f32; dimension];

        let results = self
            .search(collection_name, zero_vector, limit.unwrap_or(100), filter)
            .await?;

        // Fetch full vectors for each result
        let mut records = Vec::new();
        for result in results {
            if let Some(record) = self.get(collection_name, &result.id).await? {
                records.push(record);
            }
        }

        Ok(records)
    }

    async fn collection_info(&self, _name: &str) -> RookResult<CollectionInfo> {
        let url = format!("{}/describe_index_stats", self.index_host);

        let response = self
            .client
            .post(&url)
            .headers(self.headers())
            .json(&json!({}))
            .send()
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to describe index: {}", e)))?;

        if !response.status().is_success() {
            let error = response.text().await.unwrap_or_default();
            return Err(RookError::vector_store(format!(
                "Failed to describe index: {}",
                error
            )));
        }

        let stats: PineconeDescribeResponse = response
            .json()
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to parse response: {}", e)))?;

        Ok(CollectionInfo {
            name: self.config.collection_name.clone(),
            dimension: stats.dimension,
            count: stats.total_vector_count,
            distance_metric: DistanceMetric::Cosine,
        })
    }

    async fn reset(&self, collection_name: &str) -> RookResult<()> {
        self.delete_collection(collection_name).await
    }
}

impl PineconeVectorStore {
    fn build_filter(filter: &Filter) -> serde_json::Value {
        let conditions: Vec<serde_json::Value> = filter
            .conditions
            .iter()
            .filter_map(|cond| {
                match &cond.operator {
                    rook_core::types::FilterOperator::Eq => {
                        Some(json!({ cond.field.clone(): { "$eq": cond.value } }))
                    }
                    rook_core::types::FilterOperator::Ne => {
                        Some(json!({ cond.field.clone(): { "$ne": cond.value } }))
                    }
                    rook_core::types::FilterOperator::Gt => {
                        Some(json!({ cond.field.clone(): { "$gt": cond.value } }))
                    }
                    rook_core::types::FilterOperator::Gte => {
                        Some(json!({ cond.field.clone(): { "$gte": cond.value } }))
                    }
                    rook_core::types::FilterOperator::Lt => {
                        Some(json!({ cond.field.clone(): { "$lt": cond.value } }))
                    }
                    rook_core::types::FilterOperator::Lte => {
                        Some(json!({ cond.field.clone(): { "$lte": cond.value } }))
                    }
                    rook_core::types::FilterOperator::In => {
                        Some(json!({ cond.field.clone(): { "$in": cond.value } }))
                    }
                    rook_core::types::FilterOperator::NotIn => {
                        Some(json!({ cond.field.clone(): { "$nin": cond.value } }))
                    }
                    _ => None,
                }
            })
            .collect();

        if conditions.len() == 1 {
            conditions.into_iter().next().unwrap()
        } else {
            json!({ "$and": conditions })
        }
    }
}
