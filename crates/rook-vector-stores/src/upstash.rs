//! Upstash Vector store implementation.

use async_trait::async_trait;
use std::collections::HashMap;

use rook_core::error::{RookError, RookResult};
use rook_core::traits::{VectorStore, VectorStoreConfig};
use rook_core::types::{CollectionInfo, DistanceMetric, Filter, VectorRecord, VectorSearchResult};

use reqwest::Client;
use serde_json::json;

/// Upstash Vector store implementation.
pub struct UpstashVectorStore {
    client: Client,
    url: String,
    token: String,
    config: VectorStoreConfig,
}

impl UpstashVectorStore {
    /// Create a new Upstash vector store.
    pub async fn new(config: VectorStoreConfig) -> RookResult<Self> {
        let url = config
            .url
            .clone()
            .or_else(|| std::env::var("UPSTASH_VECTOR_REST_URL").ok())
            .ok_or_else(|| {
                RookError::Configuration(
                    "Upstash URL required. Set UPSTASH_VECTOR_REST_URL or provide url.".to_string(),
                )
            })?;

        let token = config
            .api_key
            .clone()
            .or_else(|| std::env::var("UPSTASH_VECTOR_REST_TOKEN").ok())
            .ok_or_else(|| {
                RookError::Configuration(
                    "Upstash token required. Set UPSTASH_VECTOR_REST_TOKEN or provide api_key.".to_string(),
                )
            })?;

        let client = Client::new();

        Ok(Self {
            client,
            url,
            token,
            config,
        })
    }

    fn headers(&self) -> reqwest::header::HeaderMap {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            "Authorization",
            format!("Bearer {}", self.token).parse().unwrap(),
        );
        headers.insert(
            reqwest::header::CONTENT_TYPE,
            "application/json".parse().unwrap(),
        );
        headers
    }

    fn api_url(&self, path: &str) -> String {
        format!("{}{}", self.url, path)
    }
}

#[async_trait]
impl VectorStore for UpstashVectorStore {
    async fn create_collection(
        &self,
        _name: &str,
        _dimension: usize,
        _distance_metric: Option<DistanceMetric>,
    ) -> RookResult<()> {
        // Upstash Vector is a single-index service, no collection creation needed
        Ok(())
    }

    async fn insert(&self, _collection_name: &str, records: Vec<VectorRecord>) -> RookResult<()> {
        if records.is_empty() {
            return Ok(());
        }

        let vectors: Vec<serde_json::Value> = records
            .into_iter()
            .map(|r| {
                json!({
                    "id": r.id,
                    "vector": r.vector,
                    "metadata": r.metadata
                })
            })
            .collect();

        let url = self.api_url("/upsert");
        let response = self
            .client
            .post(&url)
            .headers(self.headers())
            .json(&vectors)
            .send()
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to upsert: {}", e)))?;

        if !response.status().is_success() {
            let error = response.text().await.unwrap_or_default();
            return Err(RookError::vector_store(format!("Failed to upsert: {}", error)));
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
        let url = self.api_url("/query");

        let mut body = json!({
            "vector": query_vector,
            "topK": limit,
            "includeMetadata": true,
            "includeVectors": false
        });

        if let Some(f) = filter {
            body["filter"] = serde_json::Value::String(Self::build_filter(&f));
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

        let result: serde_json::Value = response
            .json()
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to parse response: {}", e)))?;

        let matches = result["result"]
            .as_array()
            .cloned()
            .unwrap_or_default();

        let results = matches
            .into_iter()
            .map(|m| {
                let id = m["id"].as_str().unwrap_or_default().to_string();
                let score = m["score"].as_f64().unwrap_or(0.0) as f32;

                let metadata: HashMap<String, serde_json::Value> = m["metadata"]
                    .as_object()
                    .map(|o| o.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
                    .unwrap_or_default();

                VectorSearchResult {
                    id,
                    score,
                    vector: None,
                    metadata,
                }
            })
            .collect();

        Ok(results)
    }

    async fn get(&self, _collection_name: &str, id: &str) -> RookResult<Option<VectorRecord>> {
        let url = self.api_url(&format!("/fetch/{}", id));

        let response = self
            .client
            .get(&url)
            .headers(self.headers())
            .send()
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to fetch: {}", e)))?;

        if response.status().as_u16() == 404 {
            return Ok(None);
        }

        if !response.status().is_success() {
            let error = response.text().await.unwrap_or_default();
            return Err(RookError::vector_store(format!("Failed to fetch: {}", error)));
        }

        let result: serde_json::Value = response
            .json()
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to parse response: {}", e)))?;

        if result["result"].is_null() {
            return Ok(None);
        }

        let vector: Vec<f32> = result["result"]["vector"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_f64().map(|f| f as f32))
                    .collect()
            })
            .unwrap_or_default();

        let metadata: HashMap<String, serde_json::Value> = result["result"]["metadata"]
            .as_object()
            .map(|o| o.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
            .unwrap_or_default();

        Ok(Some(VectorRecord {
            id: id.to_string(),
            vector,
            metadata,
        }))
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
        let url = self.api_url("/delete");
        let body = json!([id]);

        let response = self
            .client
            .post(&url)
            .headers(self.headers())
            .json(&body)
            .send()
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to delete: {}", e)))?;

        if !response.status().is_success() {
            let error = response.text().await.unwrap_or_default();
            return Err(RookError::vector_store(format!("Failed to delete: {}", error)));
        }

        Ok(())
    }

    async fn delete_collection(&self, _name: &str) -> RookResult<()> {
        let url = self.api_url("/reset");

        let response = self
            .client
            .post(&url)
            .headers(self.headers())
            .send()
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to reset: {}", e)))?;

        if !response.status().is_success() {
            let error = response.text().await.unwrap_or_default();
            return Err(RookError::vector_store(format!("Failed to reset: {}", error)));
        }

        Ok(())
    }

    async fn list(
        &self,
        collection_name: &str,
        filter: Option<Filter>,
        limit: Option<usize>,
    ) -> RookResult<Vec<VectorRecord>> {
        // Upstash doesn't have a direct list operation
        // Use range query to list vectors
        let url = self.api_url("/range");

        let body = json!({
            "cursor": "0",
            "limit": limit.unwrap_or(100),
            "includeMetadata": true,
            "includeVectors": true
        });

        let response = self
            .client
            .post(&url)
            .headers(self.headers())
            .json(&body)
            .send()
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to range: {}", e)))?;

        if !response.status().is_success() {
            let error = response.text().await.unwrap_or_default();
            return Err(RookError::vector_store(format!("Failed to range: {}", error)));
        }

        let result: serde_json::Value = response
            .json()
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to parse response: {}", e)))?;

        let vectors = result["result"]["vectors"]
            .as_array()
            .cloned()
            .unwrap_or_default();

        let mut records: Vec<VectorRecord> = vectors
            .into_iter()
            .map(|v| {
                let id = v["id"].as_str().unwrap_or_default().to_string();
                let vector: Vec<f32> = v["vector"]
                    .as_array()
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|x| x.as_f64().map(|f| f as f32))
                            .collect()
                    })
                    .unwrap_or_default();

                let metadata: HashMap<String, serde_json::Value> = v["metadata"]
                    .as_object()
                    .map(|o| o.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
                    .unwrap_or_default();

                VectorRecord {
                    id,
                    vector,
                    metadata,
                }
            })
            .collect();

        // Apply filter if present
        if let Some(f) = filter {
            records.retain(|r| Self::matches_filter(r, &f));
        }

        Ok(records)
    }

    async fn collection_info(&self, name: &str) -> RookResult<CollectionInfo> {
        let url = self.api_url("/info");

        let response = self
            .client
            .get(&url)
            .headers(self.headers())
            .send()
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to get info: {}", e)))?;

        if !response.status().is_success() {
            let error = response.text().await.unwrap_or_default();
            return Err(RookError::vector_store(format!("Failed to get info: {}", error)));
        }

        let result: serde_json::Value = response
            .json()
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to parse response: {}", e)))?;

        let dimension = result["result"]["dimension"].as_u64().unwrap_or(1536) as usize;
        let count = result["result"]["vectorCount"].as_u64().unwrap_or(0) as usize;

        Ok(CollectionInfo {
            name: name.to_string(),
            dimension,
            count,
            distance_metric: DistanceMetric::Cosine,
        })
    }

    async fn reset(&self, collection_name: &str) -> RookResult<()> {
        self.delete_collection(collection_name).await
    }
}

impl UpstashVectorStore {
    fn build_filter(filter: &Filter) -> String {
        let conditions: Vec<String> = filter
            .conditions
            .iter()
            .filter_map(|cond| {
                let value = match &cond.value {
                    serde_json::Value::String(s) => format!("'{}'", s),
                    serde_json::Value::Number(n) => n.to_string(),
                    serde_json::Value::Bool(b) => b.to_string(),
                    _ => return None,
                };

                match &cond.operator {
                    rook_core::types::FilterOperator::Eq => {
                        Some(format!("{} = {}", cond.field, value))
                    }
                    rook_core::types::FilterOperator::Ne => {
                        Some(format!("{} != {}", cond.field, value))
                    }
                    rook_core::types::FilterOperator::Gt => {
                        Some(format!("{} > {}", cond.field, value))
                    }
                    rook_core::types::FilterOperator::Gte => {
                        Some(format!("{} >= {}", cond.field, value))
                    }
                    rook_core::types::FilterOperator::Lt => {
                        Some(format!("{} < {}", cond.field, value))
                    }
                    rook_core::types::FilterOperator::Lte => {
                        Some(format!("{} <= {}", cond.field, value))
                    }
                    _ => None,
                }
            })
            .collect();

        conditions.join(" AND ")
    }

    fn matches_filter(record: &VectorRecord, filter: &Filter) -> bool {
        filter.conditions.iter().all(|cond| {
            match record.metadata.get(&cond.field) {
                Some(value) => match &cond.operator {
                    rook_core::types::FilterOperator::Eq => value == &cond.value,
                    rook_core::types::FilterOperator::Ne => value != &cond.value,
                    _ => true,
                },
                None => false,
            }
        })
    }
}
