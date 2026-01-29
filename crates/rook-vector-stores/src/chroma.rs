//! Chroma vector store implementation.

use async_trait::async_trait;
use std::collections::HashMap;

use rook_core::error::{RookError, RookResult};
use rook_core::traits::{VectorStore, VectorStoreConfig};
use rook_core::types::{CollectionInfo, DistanceMetric, Filter, VectorRecord, VectorSearchResult};

use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::json;

/// Chroma vector store implementation.
pub struct ChromaVectorStore {
    client: Client,
    base_url: String,
    tenant: String,
    database: String,
    config: VectorStoreConfig,
}

#[derive(Debug, Serialize, Deserialize)]
struct ChromaCollection {
    id: String,
    name: String,
    metadata: Option<HashMap<String, serde_json::Value>>,
}

impl ChromaVectorStore {
    /// Create a new Chroma vector store.
    pub async fn new(config: VectorStoreConfig) -> RookResult<Self> {
        let base_url = config
            .url
            .clone()
            .unwrap_or_else(|| "http://localhost:8000".to_string());

        let tenant = config
            .tenant
            .clone()
            .unwrap_or_else(|| "default_tenant".to_string());

        let database = config
            .database
            .clone()
            .unwrap_or_else(|| "default_database".to_string());

        let client = Client::new();

        Ok(Self {
            client,
            base_url,
            tenant,
            database,
            config,
        })
    }

    fn api_url(&self, path: &str) -> String {
        format!("{}/api/v1{}", self.base_url, path)
    }

    fn distance_to_chroma(metric: &DistanceMetric) -> &'static str {
        match metric {
            DistanceMetric::Cosine => "cosine",
            DistanceMetric::Euclidean => "l2",
            DistanceMetric::DotProduct => "ip",
        }
    }

    async fn get_or_create_collection(&self, name: &str, metric: &DistanceMetric) -> RookResult<String> {
        let url = self.api_url(&format!(
            "/tenants/{}/databases/{}/collections",
            self.tenant, self.database
        ));

        let body = json!({
            "name": name,
            "metadata": {
                "hnsw:space": Self::distance_to_chroma(metric)
            }
        });

        let response = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to create collection: {}", e)))?;

        let collection: ChromaCollection = response
            .json()
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to parse response: {}", e)))?;

        Ok(collection.id)
    }

    async fn get_collection_id(&self, name: &str) -> RookResult<Option<String>> {
        let url = self.api_url(&format!(
            "/tenants/{}/databases/{}/collections/{}",
            self.tenant, self.database, name
        ));

        let response = self.client.get(&url).send().await;

        match response {
            Ok(resp) if resp.status().is_success() => {
                let collection: ChromaCollection = resp
                    .json()
                    .await
                    .map_err(|e| RookError::vector_store(format!("Failed to parse response: {}", e)))?;
                Ok(Some(collection.id))
            }
            Ok(resp) if resp.status().as_u16() == 404 => Ok(None),
            Ok(resp) => {
                let error = resp.text().await.unwrap_or_default();
                Err(RookError::vector_store(format!(
                    "Failed to get collection: {}",
                    error
                )))
            }
            Err(e) => Err(RookError::vector_store(format!(
                "Failed to get collection: {}",
                e
            ))),
        }
    }
}

#[async_trait]
impl VectorStore for ChromaVectorStore {
    async fn create_collection(
        &self,
        name: &str,
        _dimension: usize,
        distance_metric: Option<DistanceMetric>,
    ) -> RookResult<()> {
        let metric = distance_metric.unwrap_or(DistanceMetric::Cosine);
        self.get_or_create_collection(name, &metric).await?;
        Ok(())
    }

    async fn insert(&self, collection_name: &str, records: Vec<VectorRecord>) -> RookResult<()> {
        if records.is_empty() {
            return Ok(());
        }

        let collection_id = self
            .get_collection_id(collection_name)
            .await?
            .ok_or_else(|| RookError::not_found("Collection", collection_name))?;

        let ids: Vec<String> = records.iter().map(|r| r.id.clone()).collect();
        let embeddings: Vec<Vec<f32>> = records.iter().map(|r| r.vector.clone()).collect();
        let metadatas: Vec<HashMap<String, serde_json::Value>> =
            records.iter().map(|r| r.metadata.clone()).collect();

        let url = self.api_url(&format!("/collections/{}/upsert", collection_id));
        let body = json!({
            "ids": ids,
            "embeddings": embeddings,
            "metadatas": metadatas
        });

        let response = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to upsert: {}", e)))?;

        if !response.status().is_success() {
            let error = response.text().await.unwrap_or_default();
            return Err(RookError::vector_store(format!(
                "Failed to upsert: {}",
                error
            )));
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
        let collection_id = self
            .get_collection_id(collection_name)
            .await?
            .ok_or_else(|| RookError::not_found("Collection", collection_name))?;

        let url = self.api_url(&format!("/collections/{}/query", collection_id));

        let mut body = json!({
            "query_embeddings": [query_vector],
            "n_results": limit,
            "include": ["metadatas", "distances"]
        });

        if let Some(f) = filter {
            body["where"] = Self::build_filter(&f);
        }

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

        let ids = result["ids"][0].as_array().cloned().unwrap_or_default();
        let distances = result["distances"][0].as_array().cloned().unwrap_or_default();
        let metadatas = result["metadatas"][0].as_array().cloned().unwrap_or_default();

        let results: Vec<VectorSearchResult> = ids
            .into_iter()
            .zip(distances.into_iter())
            .zip(metadatas.into_iter())
            .map(|((id, distance), metadata)| {
                let id = id.as_str().unwrap_or_default().to_string();
                let distance = distance.as_f64().unwrap_or(1.0) as f32;
                let score = 1.0 - distance; // Convert distance to similarity

                let metadata: HashMap<String, serde_json::Value> = metadata
                    .as_object()
                    .map(|m| m.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
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

    async fn get(&self, collection_name: &str, id: &str) -> RookResult<Option<VectorRecord>> {
        let collection_id = self
            .get_collection_id(collection_name)
            .await?
            .ok_or_else(|| RookError::not_found("Collection", collection_name))?;

        let url = self.api_url(&format!("/collections/{}/get", collection_id));
        let body = json!({
            "ids": [id],
            "include": ["metadatas", "embeddings"]
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

        let ids = result["ids"].as_array();
        if ids.is_none() || ids.unwrap().is_empty() {
            return Ok(None);
        }

        let embeddings = result["embeddings"][0]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_f64().map(|f| f as f32))
                    .collect()
            })
            .unwrap_or_default();

        let metadata: HashMap<String, serde_json::Value> = result["metadatas"][0]
            .as_object()
            .map(|m| m.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
            .unwrap_or_default();

        Ok(Some(VectorRecord {
            id: id.to_string(),
            vector: embeddings,
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

    async fn delete(&self, collection_name: &str, id: &str) -> RookResult<()> {
        let collection_id = self
            .get_collection_id(collection_name)
            .await?
            .ok_or_else(|| RookError::not_found("Collection", collection_name))?;

        let url = self.api_url(&format!("/collections/{}/delete", collection_id));
        let body = json!({ "ids": [id] });

        let response = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to delete: {}", e)))?;

        if !response.status().is_success() {
            let error = response.text().await.unwrap_or_default();
            return Err(RookError::vector_store(format!(
                "Failed to delete: {}",
                error
            )));
        }

        Ok(())
    }

    async fn delete_collection(&self, name: &str) -> RookResult<()> {
        let url = self.api_url(&format!(
            "/tenants/{}/databases/{}/collections/{}",
            self.tenant, self.database, name
        ));

        let response = self
            .client
            .delete(&url)
            .send()
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to delete collection: {}", e)))?;

        if !response.status().is_success() && response.status().as_u16() != 404 {
            let error = response.text().await.unwrap_or_default();
            return Err(RookError::vector_store(format!(
                "Failed to delete collection: {}",
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
        let collection_id = self
            .get_collection_id(collection_name)
            .await?
            .ok_or_else(|| RookError::not_found("Collection", collection_name))?;

        let url = self.api_url(&format!("/collections/{}/get", collection_id));

        let mut body = json!({
            "include": ["metadatas", "embeddings"],
            "limit": limit.unwrap_or(100)
        });

        if let Some(f) = filter {
            body["where"] = Self::build_filter(&f);
        }

        let response = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to list: {}", e)))?;

        if !response.status().is_success() {
            let error = response.text().await.unwrap_or_default();
            return Err(RookError::vector_store(format!("Failed to list: {}", error)));
        }

        let result: serde_json::Value = response
            .json()
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to parse response: {}", e)))?;

        let ids = result["ids"].as_array().cloned().unwrap_or_default();
        let embeddings = result["embeddings"].as_array().cloned().unwrap_or_default();
        let metadatas = result["metadatas"].as_array().cloned().unwrap_or_default();

        let records: Vec<VectorRecord> = ids
            .into_iter()
            .zip(embeddings.into_iter())
            .zip(metadatas.into_iter())
            .map(|((id, embedding), metadata)| {
                let id = id.as_str().unwrap_or_default().to_string();
                let vector: Vec<f32> = embedding
                    .as_array()
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_f64().map(|f| f as f32))
                            .collect()
                    })
                    .unwrap_or_default();

                let metadata: HashMap<String, serde_json::Value> = metadata
                    .as_object()
                    .map(|m| m.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
                    .unwrap_or_default();

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
        let collection_id = self
            .get_collection_id(name)
            .await?
            .ok_or_else(|| RookError::not_found("Collection", name))?;

        let url = self.api_url(&format!("/collections/{}/count", collection_id));

        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to get count: {}", e)))?;

        let count: usize = response
            .json()
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to parse response: {}", e)))?;

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

impl ChromaVectorStore {
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
