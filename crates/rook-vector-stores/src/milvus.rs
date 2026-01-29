//! Milvus vector store implementation.

use async_trait::async_trait;
use std::collections::HashMap;

use rook_core::error::{RookError, RookResult};
use rook_core::traits::{VectorStore, VectorStoreConfig};
use rook_core::types::{CollectionInfo, DistanceMetric, Filter, VectorRecord, VectorSearchResult};

use reqwest::Client;
use serde_json::json;

/// Milvus vector store implementation using REST API.
pub struct MilvusVectorStore {
    client: Client,
    base_url: String,
    database: String,
    config: VectorStoreConfig,
}

impl MilvusVectorStore {
    /// Create a new Milvus vector store.
    pub async fn new(config: VectorStoreConfig) -> RookResult<Self> {
        let base_url = config
            .url
            .clone()
            .unwrap_or_else(|| "http://localhost:19530".to_string());

        let database = config
            .database
            .clone()
            .unwrap_or_else(|| "default".to_string());

        let client = Client::new();

        Ok(Self {
            client,
            base_url,
            database,
            config,
        })
    }

    fn api_url(&self, path: &str) -> String {
        format!("{}/v1{}", self.base_url, path)
    }

    fn distance_to_milvus(metric: &DistanceMetric) -> &'static str {
        match metric {
            DistanceMetric::Cosine => "COSINE",
            DistanceMetric::Euclidean => "L2",
            DistanceMetric::DotProduct => "IP",
        }
    }
}

#[async_trait]
impl VectorStore for MilvusVectorStore {
    async fn create_collection(
        &self,
        name: &str,
        dimension: usize,
        distance_metric: Option<DistanceMetric>,
    ) -> RookResult<()> {
        let metric = distance_metric.unwrap_or(DistanceMetric::Cosine);
        let metric_type = Self::distance_to_milvus(&metric);

        let url = self.api_url("/vector/collections/create");
        let body = json!({
            "dbName": self.database,
            "collectionName": name,
            "dimension": dimension,
            "metricType": metric_type,
            "primaryField": "id",
            "vectorField": "vector"
        });

        let response = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to create collection: {}", e)))?;

        if !response.status().is_success() {
            let error = response.text().await.unwrap_or_default();
            // Ignore if collection already exists
            if !error.contains("already exists") {
                return Err(RookError::vector_store(format!(
                    "Failed to create collection: {}",
                    error
                )));
            }
        }

        Ok(())
    }

    async fn insert(&self, collection_name: &str, records: Vec<VectorRecord>) -> RookResult<()> {
        if records.is_empty() {
            return Ok(());
        }

        let data: Vec<serde_json::Value> = records
            .into_iter()
            .map(|r| {
                let mut obj = json!({
                    "id": r.id,
                    "vector": r.vector
                });
                // Add metadata fields
                for (k, v) in r.metadata {
                    obj[k] = v;
                }
                obj
            })
            .collect();

        let url = self.api_url("/vector/insert");
        let body = json!({
            "dbName": self.database,
            "collectionName": collection_name,
            "data": data
        });

        let response = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to insert: {}", e)))?;

        if !response.status().is_success() {
            let error = response.text().await.unwrap_or_default();
            return Err(RookError::vector_store(format!("Failed to insert: {}", error)));
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
        let url = self.api_url("/vector/search");

        let mut body = json!({
            "dbName": self.database,
            "collectionName": collection_name,
            "vector": query_vector,
            "limit": limit,
            "outputFields": ["*"]
        });

        if let Some(f) = filter {
            body["filter"] = serde_json::Value::String(Self::build_filter(&f));
        }

        let response = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to search: {}", e)))?;

        if !response.status().is_success() {
            let error = response.text().await.unwrap_or_default();
            return Err(RookError::vector_store(format!("Failed to search: {}", error)));
        }

        let result: serde_json::Value = response
            .json()
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to parse response: {}", e)))?;

        let data = result["data"].as_array().cloned().unwrap_or_default();

        let results = data
            .into_iter()
            .map(|item| {
                let id = item["id"].as_str().unwrap_or_default().to_string();
                let distance = item["distance"].as_f64().unwrap_or(1.0) as f32;
                let score = 1.0 - distance;

                let mut metadata = HashMap::new();
                if let Some(obj) = item.as_object() {
                    for (k, v) in obj {
                        if k != "id" && k != "vector" && k != "distance" {
                            metadata.insert(k.clone(), v.clone());
                        }
                    }
                }

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

    async fn get(&self, collection_name: &str, id: &str) -> RookResult<Option<VectorRecord>> {
        let url = self.api_url("/vector/get");
        let body = json!({
            "dbName": self.database,
            "collectionName": collection_name,
            "id": [id],
            "outputFields": ["*"]
        });

        let response = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to get: {}", e)))?;

        if !response.status().is_success() {
            let error = response.text().await.unwrap_or_default();
            return Err(RookError::vector_store(format!("Failed to get: {}", error)));
        }

        let result: serde_json::Value = response
            .json()
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to parse response: {}", e)))?;

        let data = result["data"].as_array();
        if data.is_none() || data.unwrap().is_empty() {
            return Ok(None);
        }

        let item = &data.unwrap()[0];
        let vector: Vec<f32> = item["vector"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_f64().map(|f| f as f32))
                    .collect()
            })
            .unwrap_or_default();

        let mut metadata = HashMap::new();
        if let Some(obj) = item.as_object() {
            for (k, v) in obj {
                if k != "id" && k != "vector" {
                    metadata.insert(k.clone(), v.clone());
                }
            }
        }

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

        // Delete and re-insert
        self.delete(collection_name, id).await?;

        let record = VectorRecord {
            id: id.to_string(),
            vector: new_vector,
            metadata: new_metadata,
        };

        self.insert(collection_name, vec![record]).await
    }

    async fn delete(&self, collection_name: &str, id: &str) -> RookResult<()> {
        let url = self.api_url("/vector/delete");
        let body = json!({
            "dbName": self.database,
            "collectionName": collection_name,
            "id": [id]
        });

        let response = self
            .client
            .post(&url)
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

    async fn delete_collection(&self, name: &str) -> RookResult<()> {
        let url = self.api_url("/vector/collections/drop");
        let body = json!({
            "dbName": self.database,
            "collectionName": name
        });

        let response = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to drop collection: {}", e)))?;

        if !response.status().is_success() {
            let error = response.text().await.unwrap_or_default();
            return Err(RookError::vector_store(format!(
                "Failed to drop collection: {}",
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
        let url = self.api_url("/vector/query");

        let filter_expr = filter
            .map(|f| Self::build_filter(&f))
            .unwrap_or_else(|| "id != \"\"".to_string());

        let body = json!({
            "dbName": self.database,
            "collectionName": collection_name,
            "filter": filter_expr,
            "limit": limit.unwrap_or(100),
            "outputFields": ["*"]
        });

        let response = self
            .client
            .post(&url)
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

        let data = result["data"].as_array().cloned().unwrap_or_default();

        let records = data
            .into_iter()
            .map(|item| {
                let id = item["id"].as_str().unwrap_or_default().to_string();
                let vector: Vec<f32> = item["vector"]
                    .as_array()
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_f64().map(|f| f as f32))
                            .collect()
                    })
                    .unwrap_or_default();

                let mut metadata = HashMap::new();
                if let Some(obj) = item.as_object() {
                    for (k, v) in obj {
                        if k != "id" && k != "vector" {
                            metadata.insert(k.clone(), v.clone());
                        }
                    }
                }

                VectorRecord {
                    id,
                    vector,
                    metadata,
                }
            })
            .collect();

        Ok(records)
    }

    async fn collection_info(&self, name: &str) -> RookResult<CollectionInfo> {
        let url = self.api_url("/vector/collections/describe");
        let body = json!({
            "dbName": self.database,
            "collectionName": name
        });

        let response = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to describe: {}", e)))?;

        if !response.status().is_success() {
            return Err(RookError::not_found("Collection", name));
        }

        let result: serde_json::Value = response
            .json()
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to parse response: {}", e)))?;

        let dimension = result["data"]["fields"]
            .as_array()
            .and_then(|fields| {
                fields.iter().find_map(|f| {
                    if f["name"] == "vector" {
                        f["params"]["dim"].as_u64().map(|d| d as usize)
                    } else {
                        None
                    }
                })
            })
            .unwrap_or(self.config.embedding_dims.unwrap_or(1536));

        // Get count via stats
        let stats_url = self.api_url("/vector/collections/get_stats");
        let stats_body = json!({
            "dbName": self.database,
            "collectionName": name
        });

        let stats_response = self
            .client
            .post(&stats_url)
            .json(&stats_body)
            .send()
            .await;

        let count = stats_response
            .ok()
            .and_then(|r| r.json::<serde_json::Value>().ok())
            .and_then(|v| v["data"]["rowCount"].as_u64())
            .unwrap_or(0) as usize;

        Ok(CollectionInfo {
            name: name.to_string(),
            dimension,
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

impl MilvusVectorStore {
    fn build_filter(filter: &Filter) -> String {
        let conditions: Vec<String> = filter
            .conditions
            .iter()
            .filter_map(|cond| {
                let value = match &cond.value {
                    serde_json::Value::String(s) => format!("\"{}\"", s),
                    serde_json::Value::Number(n) => n.to_string(),
                    serde_json::Value::Bool(b) => b.to_string(),
                    _ => return None,
                };

                match &cond.operator {
                    rook_core::types::FilterOperator::Eq => Some(format!("{} == {}", cond.field, value)),
                    rook_core::types::FilterOperator::Ne => Some(format!("{} != {}", cond.field, value)),
                    rook_core::types::FilterOperator::Gt => Some(format!("{} > {}", cond.field, value)),
                    rook_core::types::FilterOperator::Gte => Some(format!("{} >= {}", cond.field, value)),
                    rook_core::types::FilterOperator::Lt => Some(format!("{} < {}", cond.field, value)),
                    rook_core::types::FilterOperator::Lte => Some(format!("{} <= {}", cond.field, value)),
                    rook_core::types::FilterOperator::In => {
                        Some(format!("{} in {}", cond.field, value))
                    }
                    _ => None,
                }
            })
            .collect();

        if conditions.is_empty() {
            "id != \"\"".to_string()
        } else {
            conditions.join(" and ")
        }
    }
}
