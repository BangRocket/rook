//! Vertex AI Vector Search implementation.

use async_trait::async_trait;
use std::collections::HashMap;

use rook_core::error::{RookError, RookResult};
use rook_core::traits::{VectorStore, VectorStoreConfig};
use rook_core::types::{CollectionInfo, DistanceMetric, Filter, VectorRecord, VectorSearchResult};

use reqwest::Client;
use serde_json::json;

/// Vertex AI Vector Search implementation.
pub struct VertexAIVectorStore {
    client: Client,
    project_id: String,
    location: String,
    index_endpoint: String,
    deployed_index_id: String,
    config: VectorStoreConfig,
}

impl VertexAIVectorStore {
    /// Create a new Vertex AI vector store.
    pub async fn new(config: VectorStoreConfig) -> RookResult<Self> {
        let project_id = config
            .project_id
            .clone()
            .or_else(|| std::env::var("GOOGLE_CLOUD_PROJECT").ok())
            .ok_or_else(|| {
                RookError::Configuration("Google Cloud project ID required".to_string())
            })?;

        let location = config
            .region
            .clone()
            .unwrap_or_else(|| "us-central1".to_string());

        let index_endpoint = config
            .url
            .clone()
            .ok_or_else(|| {
                RookError::Configuration("Vertex AI index endpoint required".to_string())
            })?;

        let deployed_index_id = config.collection_name.clone();
        let client = Client::new();

        Ok(Self {
            client,
            project_id,
            location,
            index_endpoint,
            deployed_index_id,
            config,
        })
    }
}

#[async_trait]
impl VectorStore for VertexAIVectorStore {
    async fn create_collection(
        &self,
        _name: &str,
        _dimension: usize,
        _distance_metric: Option<DistanceMetric>,
    ) -> RookResult<()> {
        // Vertex AI indexes are created via the Google Cloud Console or gcloud CLI
        tracing::info!("Vertex AI indexes must be created via Google Cloud Console");
        Ok(())
    }

    async fn insert(&self, _collection_name: &str, records: Vec<VectorRecord>) -> RookResult<()> {
        if records.is_empty() {
            return Ok(());
        }

        let datapoints: Vec<serde_json::Value> = records
            .into_iter()
            .map(|r| {
                json!({
                    "datapoint_id": r.id,
                    "feature_vector": r.vector,
                    "restricts": r.metadata.iter().map(|(k, v)| {
                        json!({
                            "namespace": k,
                            "allow_list": [v.to_string().trim_matches('"')]
                        })
                    }).collect::<Vec<_>>()
                })
            })
            .collect();

        let url = format!(
            "https://{}-aiplatform.googleapis.com/v1/projects/{}/locations/{}/indexEndpoints/{}/deployedIndexes/{}:upsertDatapoints",
            self.location, self.project_id, self.location, self.index_endpoint, self.deployed_index_id
        );

        let body = json!({ "datapoints": datapoints });

        let response = self
            .client
            .post(&url)
            .json(&body)
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
        _filter: Option<Filter>,
    ) -> RookResult<Vec<VectorSearchResult>> {
        let url = format!(
            "https://{}-aiplatform.googleapis.com/v1/projects/{}/locations/{}/indexEndpoints/{}:findNeighbors",
            self.location, self.project_id, self.location, self.index_endpoint
        );

        let body = json!({
            "deployed_index_id": self.deployed_index_id,
            "queries": [{
                "datapoint": {
                    "feature_vector": query_vector
                },
                "neighbor_count": limit
            }]
        });

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

        let neighbors = result["nearestNeighbors"][0]["neighbors"]
            .as_array()
            .cloned()
            .unwrap_or_default();

        let results = neighbors
            .into_iter()
            .map(|n| {
                let id = n["datapoint"]["datapointId"]
                    .as_str()
                    .unwrap_or_default()
                    .to_string();
                let distance = n["distance"].as_f64().unwrap_or(1.0) as f32;

                VectorSearchResult {
                    id,
                    score: 1.0 - distance,
                    vector: None,
                    metadata: HashMap::new(),
                }
            })
            .collect();

        Ok(results)
    }

    async fn get(&self, _collection_name: &str, _id: &str) -> RookResult<Option<VectorRecord>> {
        // Vertex AI doesn't have a direct get by ID operation
        Ok(None)
    }

    async fn update(
        &self,
        collection_name: &str,
        id: &str,
        vector: Option<Vec<f32>>,
        metadata: Option<HashMap<String, serde_json::Value>>,
    ) -> RookResult<()> {
        let record = VectorRecord {
            id: id.to_string(),
            vector: vector.unwrap_or_default(),
            metadata: metadata.unwrap_or_default(),
        };
        self.insert(collection_name, vec![record]).await
    }

    async fn delete(&self, _collection_name: &str, id: &str) -> RookResult<()> {
        let url = format!(
            "https://{}-aiplatform.googleapis.com/v1/projects/{}/locations/{}/indexEndpoints/{}/deployedIndexes/{}:removeDatapoints",
            self.location, self.project_id, self.location, self.index_endpoint, self.deployed_index_id
        );

        let body = json!({ "datapoint_ids": [id] });

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

    async fn delete_collection(&self, _name: &str) -> RookResult<()> {
        tracing::info!("Vertex AI indexes must be deleted via Google Cloud Console");
        Ok(())
    }

    async fn list(
        &self,
        _collection_name: &str,
        _filter: Option<Filter>,
        _limit: Option<usize>,
    ) -> RookResult<Vec<VectorRecord>> {
        // Vertex AI doesn't have a direct list operation
        Ok(vec![])
    }

    async fn collection_info(&self, name: &str) -> RookResult<CollectionInfo> {
        Ok(CollectionInfo {
            name: name.to_string(),
            dimension: self.config.embedding_dims.unwrap_or(1536),
            count: 0,
            distance_metric: DistanceMetric::Cosine,
        })
    }

    async fn reset(&self, _collection_name: &str) -> RookResult<()> {
        tracing::info!("Vertex AI indexes must be reset via Google Cloud Console");
        Ok(())
    }
}
