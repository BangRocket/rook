//! Supabase vector store implementation (uses pgvector).

use async_trait::async_trait;
use std::collections::HashMap;

use rook_core::error::{RookError, RookResult};
use rook_core::traits::{VectorStore, VectorStoreConfig};
use rook_core::types::{CollectionInfo, DistanceMetric, Filter, VectorRecord, VectorSearchResult};

use reqwest::Client;
use serde_json::json;

/// Supabase vector store implementation.
pub struct SupabaseVectorStore {
    client: Client,
    url: String,
    api_key: String,
    config: VectorStoreConfig,
}

impl SupabaseVectorStore {
    /// Create a new Supabase vector store.
    pub async fn new(config: VectorStoreConfig) -> RookResult<Self> {
        let url = config
            .url
            .clone()
            .or_else(|| std::env::var("SUPABASE_URL").ok())
            .ok_or_else(|| {
                RookError::Configuration("Supabase URL required".to_string())
            })?;

        let api_key = config
            .api_key
            .clone()
            .or_else(|| std::env::var("SUPABASE_SERVICE_ROLE_KEY").ok())
            .ok_or_else(|| {
                RookError::Configuration("Supabase API key required".to_string())
            })?;

        let client = Client::new();

        Ok(Self {
            client,
            url,
            api_key,
            config,
        })
    }

    fn headers(&self) -> reqwest::header::HeaderMap {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert("apikey", self.api_key.parse().unwrap());
        headers.insert(
            "Authorization",
            format!("Bearer {}", self.api_key).parse().unwrap(),
        );
        headers.insert(
            reqwest::header::CONTENT_TYPE,
            "application/json".parse().unwrap(),
        );
        headers.insert("Prefer", "return=representation".parse().unwrap());
        headers
    }

    fn rest_url(&self, path: &str) -> String {
        format!("{}/rest/v1{}", self.url, path)
    }
}

#[async_trait]
impl VectorStore for SupabaseVectorStore {
    async fn create_collection(
        &self,
        name: &str,
        dimension: usize,
        _distance_metric: Option<DistanceMetric>,
    ) -> RookResult<()> {
        // Execute SQL via RPC to create table with vector column
        let sql = format!(
            r#"
            CREATE TABLE IF NOT EXISTS "{}" (
                id TEXT PRIMARY KEY,
                vector vector({}),
                metadata JSONB DEFAULT '{{}}'::jsonb
            )
            "#,
            name, dimension
        );

        let url = self.rest_url("/rpc/exec_sql");
        let body = json!({ "sql": sql });

        let response = self
            .client
            .post(&url)
            .headers(self.headers())
            .json(&body)
            .send()
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to create table: {}", e)))?;

        if !response.status().is_success() {
            // Table might already exist, that's OK
            let error = response.text().await.unwrap_or_default();
            if !error.contains("already exists") {
                tracing::warn!("Create table response: {}", error);
            }
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
                    "id": r.id,
                    "vector": format!("[{}]", r.vector.iter().map(|f| f.to_string()).collect::<Vec<_>>().join(",")),
                    "metadata": r.metadata
                })
            })
            .collect();

        let url = self.rest_url(&format!("/{}", collection_name));

        let response = self
            .client
            .post(&url)
            .headers(self.headers())
            .header("Prefer", "resolution=merge-duplicates")
            .json(&documents)
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
        _filter: Option<Filter>,
    ) -> RookResult<Vec<VectorSearchResult>> {
        // Use RPC function for similarity search
        let url = self.rest_url("/rpc/match_vectors");

        let body = json!({
            "table_name": collection_name,
            "query_vector": query_vector,
            "match_count": limit
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

        let results: Vec<serde_json::Value> = response
            .json()
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to parse response: {}", e)))?;

        let search_results = results
            .into_iter()
            .map(|r| {
                let id = r["id"].as_str().unwrap_or_default().to_string();
                let similarity = r["similarity"].as_f64().unwrap_or(0.0) as f32;

                let metadata: HashMap<String, serde_json::Value> = r["metadata"]
                    .as_object()
                    .map(|m| m.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
                    .unwrap_or_default();

                VectorSearchResult {
                    id,
                    score: similarity,
                    vector: None,
                    metadata,
                }
            })
            .collect();

        Ok(search_results)
    }

    async fn get(&self, collection_name: &str, id: &str) -> RookResult<Option<VectorRecord>> {
        let url = self.rest_url(&format!("/{}?id=eq.{}", collection_name, id));

        let response = self
            .client
            .get(&url)
            .headers(self.headers())
            .send()
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to get: {}", e)))?;

        if !response.status().is_success() {
            let error = response.text().await.unwrap_or_default();
            return Err(RookError::vector_store(format!("Failed to get: {}", error)));
        }

        let results: Vec<serde_json::Value> = response
            .json()
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to parse response: {}", e)))?;

        if results.is_empty() {
            return Ok(None);
        }

        let doc = &results[0];
        let vector_str = doc["vector"].as_str().unwrap_or("[]");
        let vector: Vec<f32> = vector_str
            .trim_matches(|c| c == '[' || c == ']')
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect();

        let metadata: HashMap<String, serde_json::Value> = doc["metadata"]
            .as_object()
            .map(|m| m.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
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
        let url = self.rest_url(&format!("/{}?id=eq.{}", collection_name, id));

        let response = self
            .client
            .delete(&url)
            .headers(self.headers())
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
        let sql = format!(r#"DROP TABLE IF EXISTS "{}" CASCADE"#, name);
        let url = self.rest_url("/rpc/exec_sql");
        let body = json!({ "sql": sql });

        let response = self
            .client
            .post(&url)
            .headers(self.headers())
            .json(&body)
            .send()
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to drop table: {}", e)))?;

        if !response.status().is_success() {
            let error = response.text().await.unwrap_or_default();
            return Err(RookError::vector_store(format!(
                "Failed to drop table: {}",
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
        let limit_param = limit.map(|l| format!("&limit={}", l)).unwrap_or_default();
        let url = self.rest_url(&format!("/{}?select=*{}", collection_name, limit_param));

        let response = self
            .client
            .get(&url)
            .headers(self.headers())
            .send()
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to list: {}", e)))?;

        if !response.status().is_success() {
            let error = response.text().await.unwrap_or_default();
            return Err(RookError::vector_store(format!("Failed to list: {}", error)));
        }

        let results: Vec<serde_json::Value> = response
            .json()
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to parse response: {}", e)))?;

        let records = results
            .into_iter()
            .map(|doc| {
                let id = doc["id"].as_str().unwrap_or_default().to_string();
                let vector_str = doc["vector"].as_str().unwrap_or("[]");
                let vector: Vec<f32> = vector_str
                    .trim_matches(|c| c == '[' || c == ']')
                    .split(',')
                    .filter_map(|s| s.trim().parse().ok())
                    .collect();

                let metadata: HashMap<String, serde_json::Value> = doc["metadata"]
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
        let url = self.rest_url(&format!("/{}?select=count", name));

        let response = self
            .client
            .get(&url)
            .headers(self.headers())
            .header("Prefer", "count=exact")
            .send()
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to get count: {}", e)))?;

        let count = response
            .headers()
            .get("content-range")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.split('/').last())
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);

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
