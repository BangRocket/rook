//! Azure AI Search vector store implementation.

use async_trait::async_trait;
use std::collections::HashMap;

use rook_core::error::{RookError, RookResult};
use rook_core::traits::{VectorStore, VectorStoreConfig};
use rook_core::types::{CollectionInfo, DistanceMetric, Filter, VectorRecord, VectorSearchResult};

use reqwest::Client;
use serde_json::json;

/// Azure AI Search vector store implementation.
pub struct AzureAISearchVectorStore {
    client: Client,
    endpoint: String,
    api_key: String,
    api_version: String,
    config: VectorStoreConfig,
}

impl AzureAISearchVectorStore {
    /// Create a new Azure AI Search vector store.
    pub async fn new(config: VectorStoreConfig) -> RookResult<Self> {
        let endpoint = config
            .url
            .clone()
            .or_else(|| std::env::var("AZURE_SEARCH_ENDPOINT").ok())
            .ok_or_else(|| {
                RookError::Configuration(
                    "Azure Search endpoint required. Set AZURE_SEARCH_ENDPOINT or provide url.".to_string(),
                )
            })?;

        let api_key = config
            .api_key
            .clone()
            .or_else(|| std::env::var("AZURE_SEARCH_API_KEY").ok())
            .ok_or_else(|| {
                RookError::Configuration(
                    "Azure Search API key required. Set AZURE_SEARCH_API_KEY or provide api_key.".to_string(),
                )
            })?;

        let api_version = "2024-07-01".to_string();
        let client = Client::new();

        Ok(Self {
            client,
            endpoint,
            api_key,
            api_version,
            config,
        })
    }

    fn headers(&self) -> reqwest::header::HeaderMap {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert("api-key", self.api_key.parse().unwrap());
        headers.insert(
            reqwest::header::CONTENT_TYPE,
            "application/json".parse().unwrap(),
        );
        headers
    }

    fn index_url(&self, index_name: &str, path: &str) -> String {
        format!(
            "{}/indexes/{}/{}?api-version={}",
            self.endpoint, index_name, path, self.api_version
        )
    }
}

#[async_trait]
impl VectorStore for AzureAISearchVectorStore {
    async fn create_collection(
        &self,
        name: &str,
        dimension: usize,
        _distance_metric: Option<DistanceMetric>,
    ) -> RookResult<()> {
        let url = format!(
            "{}/indexes/{}?api-version={}",
            self.endpoint, name, self.api_version
        );

        let schema = json!({
            "name": name,
            "fields": [
                {"name": "id", "type": "Edm.String", "key": true, "filterable": true},
                {
                    "name": "vector",
                    "type": "Collection(Edm.Single)",
                    "dimensions": dimension,
                    "vectorSearchProfile": "vectorProfile"
                },
                {"name": "metadata", "type": "Edm.String", "filterable": true}
            ],
            "vectorSearch": {
                "profiles": [{
                    "name": "vectorProfile",
                    "algorithm": "hnsw",
                    "algorithmConfiguration": "hnswConfig"
                }],
                "algorithms": [{
                    "name": "hnswConfig",
                    "kind": "hnsw",
                    "hnswParameters": {
                        "m": 4,
                        "efConstruction": 400,
                        "efSearch": 500,
                        "metric": "cosine"
                    }
                }]
            }
        });

        let response = self
            .client
            .put(&url)
            .headers(self.headers())
            .json(&schema)
            .send()
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to create index: {}", e)))?;

        if !response.status().is_success() && response.status().as_u16() != 409 {
            let error = response.text().await.unwrap_or_default();
            return Err(RookError::vector_store(format!(
                "Failed to create index: {}",
                error
            )));
        }

        Ok(())
    }

    async fn insert(&self, collection_name: &str, records: Vec<VectorRecord>) -> RookResult<()> {
        if records.is_empty() {
            return Ok(());
        }

        let documents: Vec<serde_json::Value> = records
            .into_iter()
            .map(|r| {
                json!({
                    "@search.action": "mergeOrUpload",
                    "id": r.id,
                    "vector": r.vector,
                    "metadata": serde_json::to_string(&r.metadata).unwrap_or_default()
                })
            })
            .collect();

        let url = self.index_url(collection_name, "docs/index");
        let body = json!({ "value": documents });

        let response = self
            .client
            .post(&url)
            .headers(self.headers())
            .json(&body)
            .send()
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to index documents: {}", e)))?;

        if !response.status().is_success() {
            let error = response.text().await.unwrap_or_default();
            return Err(RookError::vector_store(format!(
                "Failed to index documents: {}",
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
        _filter: Option<Filter>,
    ) -> RookResult<Vec<VectorSearchResult>> {
        let url = self.index_url(collection_name, "docs/search");

        let body = json!({
            "vectorQueries": [{
                "vector": query_vector,
                "fields": "vector",
                "kind": "vector",
                "k": limit
            }],
            "select": "id,metadata",
            "top": limit
        });

        let response = self
            .client
            .post(&url)
            .headers(self.headers())
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

        let documents = result["value"].as_array().cloned().unwrap_or_default();

        let results = documents
            .into_iter()
            .map(|doc| {
                let id = doc["id"].as_str().unwrap_or_default().to_string();
                let score = doc["@search.score"].as_f64().unwrap_or(0.0) as f32;

                let metadata: HashMap<String, serde_json::Value> = doc["metadata"]
                    .as_str()
                    .and_then(|s| serde_json::from_str(s).ok())
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
        let url = format!(
            "{}/indexes/{}/docs/{}?api-version={}",
            self.endpoint, collection_name, id, self.api_version
        );

        let response = self
            .client
            .get(&url)
            .headers(self.headers())
            .send()
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to get document: {}", e)))?;

        if response.status().as_u16() == 404 {
            return Ok(None);
        }

        if !response.status().is_success() {
            let error = response.text().await.unwrap_or_default();
            return Err(RookError::vector_store(format!(
                "Failed to get document: {}",
                error
            )));
        }

        let doc: serde_json::Value = response
            .json()
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to parse response: {}", e)))?;

        let vector: Vec<f32> = doc["vector"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_f64().map(|f| f as f32))
                    .collect()
            })
            .unwrap_or_default();

        let metadata: HashMap<String, serde_json::Value> = doc["metadata"]
            .as_str()
            .and_then(|s| serde_json::from_str(s).ok())
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

    async fn delete(&self, collection_name: &str, id: &str) -> RookResult<()> {
        let url = self.index_url(collection_name, "docs/index");
        let body = json!({
            "value": [{
                "@search.action": "delete",
                "id": id
            }]
        });

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

    async fn delete_collection(&self, name: &str) -> RookResult<()> {
        let url = format!(
            "{}/indexes/{}?api-version={}",
            self.endpoint, name, self.api_version
        );

        let response = self
            .client
            .delete(&url)
            .headers(self.headers())
            .send()
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to delete index: {}", e)))?;

        if !response.status().is_success() && response.status().as_u16() != 404 {
            let error = response.text().await.unwrap_or_default();
            return Err(RookError::vector_store(format!(
                "Failed to delete index: {}",
                error
            )));
        }

        Ok(())
    }

    async fn list(
        &self,
        collection_name: &str,
        _filter: Option<Filter>,
        limit: Option<usize>,
    ) -> RookResult<Vec<VectorRecord>> {
        let url = self.index_url(collection_name, "docs/search");

        let body = json!({
            "search": "*",
            "select": "id,vector,metadata",
            "top": limit.unwrap_or(100)
        });

        let response = self
            .client
            .post(&url)
            .headers(self.headers())
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

        let documents = result["value"].as_array().cloned().unwrap_or_default();

        let records = documents
            .into_iter()
            .map(|doc| {
                let id = doc["id"].as_str().unwrap_or_default().to_string();
                let vector: Vec<f32> = doc["vector"]
                    .as_array()
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_f64().map(|f| f as f32))
                            .collect()
                    })
                    .unwrap_or_default();

                let metadata: HashMap<String, serde_json::Value> = doc["metadata"]
                    .as_str()
                    .and_then(|s| serde_json::from_str(s).ok())
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
        let url = format!(
            "{}/indexes/{}?api-version={}",
            self.endpoint, name, self.api_version
        );

        let response = self
            .client
            .get(&url)
            .headers(self.headers())
            .send()
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to get index: {}", e)))?;

        if !response.status().is_success() {
            return Err(RookError::not_found("Collection", name));
        }

        Ok(CollectionInfo {
            name: name.to_string(),
            dimension: self.config.embedding_dims.unwrap_or(1536),
            count: 0, // Azure AI Search doesn't easily expose document count
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
