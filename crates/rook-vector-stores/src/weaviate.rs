//! Weaviate vector store implementation.

use async_trait::async_trait;
use std::collections::HashMap;

use rook_core::error::{RookError, RookResult};
use rook_core::traits::{VectorStore, VectorStoreConfig};
use rook_core::types::{CollectionInfo, DistanceMetric, Filter, VectorRecord, VectorSearchResult};

use reqwest::Client;
use serde_json::json;

/// Weaviate vector store implementation.
pub struct WeaviateVectorStore {
    client: Client,
    base_url: String,
    api_key: Option<String>,
    config: VectorStoreConfig,
}

impl WeaviateVectorStore {
    /// Create a new Weaviate vector store.
    pub async fn new(config: VectorStoreConfig) -> RookResult<Self> {
        let base_url = config
            .url
            .clone()
            .unwrap_or_else(|| "http://localhost:8080".to_string());

        let api_key = config.api_key.clone();
        let client = Client::new();

        Ok(Self {
            client,
            base_url,
            api_key,
            config,
        })
    }

    fn headers(&self) -> reqwest::header::HeaderMap {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            reqwest::header::CONTENT_TYPE,
            "application/json".parse().unwrap(),
        );
        if let Some(ref key) = self.api_key {
            headers.insert(
                "Authorization",
                format!("Bearer {}", key).parse().unwrap(),
            );
        }
        headers
    }

    fn class_name(collection: &str) -> String {
        // Weaviate class names must start with uppercase
        let mut chars: Vec<char> = collection.chars().collect();
        if let Some(first) = chars.first_mut() {
            *first = first.to_uppercase().next().unwrap_or(*first);
        }
        chars.into_iter().collect()
    }

    fn distance_to_weaviate(metric: &DistanceMetric) -> &'static str {
        match metric {
            DistanceMetric::Cosine => "cosine",
            DistanceMetric::Euclidean => "l2-squared",
            DistanceMetric::DotProduct => "dot",
        }
    }
}

#[async_trait]
impl VectorStore for WeaviateVectorStore {
    async fn create_collection(
        &self,
        name: &str,
        _dimension: usize,
        distance_metric: Option<DistanceMetric>,
    ) -> RookResult<()> {
        let class_name = Self::class_name(name);
        let metric = distance_metric.unwrap_or(DistanceMetric::Cosine);
        let distance = Self::distance_to_weaviate(&metric);

        let schema = json!({
            "class": class_name,
            "vectorizer": "none",
            "vectorIndexConfig": {
                "distance": distance
            },
            "properties": [
                {
                    "name": "doc_id",
                    "dataType": ["text"],
                    "description": "The document ID"
                },
                {
                    "name": "metadata",
                    "dataType": ["object"],
                    "description": "Document metadata"
                }
            ]
        });

        let url = format!("{}/v1/schema", self.base_url);
        let response = self
            .client
            .post(&url)
            .headers(self.headers())
            .json(&schema)
            .send()
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to create class: {}", e)))?;

        if !response.status().is_success() {
            let error = response.text().await.unwrap_or_default();
            // Ignore if class already exists
            if !error.contains("already exists") {
                return Err(RookError::vector_store(format!(
                    "Failed to create class: {}",
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

        let class_name = Self::class_name(collection_name);

        // Use batch endpoint
        let objects: Vec<serde_json::Value> = records
            .into_iter()
            .map(|r| {
                json!({
                    "class": class_name,
                    "id": format_uuid(&r.id),
                    "vector": r.vector,
                    "properties": {
                        "doc_id": r.id,
                        "metadata": r.metadata
                    }
                })
            })
            .collect();

        let url = format!("{}/v1/batch/objects", self.base_url);
        let body = json!({ "objects": objects });

        let response = self
            .client
            .post(&url)
            .headers(self.headers())
            .json(&body)
            .send()
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to batch insert: {}", e)))?;

        if !response.status().is_success() {
            let error = response.text().await.unwrap_or_default();
            return Err(RookError::vector_store(format!(
                "Failed to batch insert: {}",
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
        let class_name = Self::class_name(collection_name);

        let mut where_filter = String::new();
        if let Some(f) = filter {
            where_filter = format!(", where: {}", Self::build_filter(&f));
        }

        let query = format!(
            r#"{{
                Get {{
                    {}(
                        nearVector: {{
                            vector: {:?}
                        }}
                        limit: {}
                        {}
                    ) {{
                        doc_id
                        metadata
                        _additional {{
                            id
                            distance
                        }}
                    }}
                }}
            }}"#,
            class_name, query_vector, limit, where_filter
        );

        let url = format!("{}/v1/graphql", self.base_url);
        let body = json!({ "query": query });

        let response = self
            .client
            .post(&url)
            .headers(self.headers())
            .json(&body)
            .send()
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to search: {}", e)))?;

        let body: serde_json::Value = response
            .json()
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to parse response: {}", e)))?;

        let objects = body["data"]["Get"][&class_name]
            .as_array()
            .cloned()
            .unwrap_or_default();

        let results = objects
            .into_iter()
            .map(|obj| {
                let id = obj["doc_id"].as_str().unwrap_or_default().to_string();
                let distance = obj["_additional"]["distance"]
                    .as_f64()
                    .unwrap_or(1.0) as f32;
                let score = 1.0 - distance; // Convert distance to similarity

                let metadata: HashMap<String, serde_json::Value> = obj["metadata"]
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
        let class_name = Self::class_name(collection_name);
        let uuid = format_uuid(id);

        let url = format!("{}/v1/objects/{}/{}", self.base_url, class_name, uuid);

        let response = self
            .client
            .get(&url)
            .headers(self.headers())
            .query(&[("include", "vector")])
            .send()
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to get object: {}", e)))?;

        if response.status().as_u16() == 404 {
            return Ok(None);
        }

        if !response.status().is_success() {
            let error = response.text().await.unwrap_or_default();
            return Err(RookError::vector_store(format!(
                "Failed to get object: {}",
                error
            )));
        }

        let body: serde_json::Value = response
            .json()
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to parse response: {}", e)))?;

        let vector: Vec<f32> = body["vector"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_f64().map(|f| f as f32))
                    .collect()
            })
            .unwrap_or_default();

        let metadata: HashMap<String, serde_json::Value> = body["properties"]["metadata"]
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
        let class_name = Self::class_name(collection_name);
        let uuid = format_uuid(id);

        let url = format!("{}/v1/objects/{}/{}", self.base_url, class_name, uuid);

        let response = self
            .client
            .delete(&url)
            .headers(self.headers())
            .send()
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to delete object: {}", e)))?;

        if !response.status().is_success() && response.status().as_u16() != 404 {
            let error = response.text().await.unwrap_or_default();
            return Err(RookError::vector_store(format!(
                "Failed to delete object: {}",
                error
            )));
        }

        Ok(())
    }

    async fn delete_collection(&self, name: &str) -> RookResult<()> {
        let class_name = Self::class_name(name);
        let url = format!("{}/v1/schema/{}", self.base_url, class_name);

        let response = self
            .client
            .delete(&url)
            .headers(self.headers())
            .send()
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to delete class: {}", e)))?;

        if !response.status().is_success() && response.status().as_u16() != 404 {
            let error = response.text().await.unwrap_or_default();
            return Err(RookError::vector_store(format!(
                "Failed to delete class: {}",
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
        let class_name = Self::class_name(collection_name);
        let limit_val = limit.unwrap_or(100);

        let mut where_filter = String::new();
        if let Some(f) = filter {
            where_filter = format!(", where: {}", Self::build_filter(&f));
        }

        let query = format!(
            r#"{{
                Get {{
                    {}(limit: {}{}) {{
                        doc_id
                        metadata
                        _additional {{
                            id
                            vector
                        }}
                    }}
                }}
            }}"#,
            class_name, limit_val, where_filter
        );

        let url = format!("{}/v1/graphql", self.base_url);
        let body = json!({ "query": query });

        let response = self
            .client
            .post(&url)
            .headers(self.headers())
            .json(&body)
            .send()
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to list: {}", e)))?;

        let body: serde_json::Value = response
            .json()
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to parse response: {}", e)))?;

        let objects = body["data"]["Get"][&class_name]
            .as_array()
            .cloned()
            .unwrap_or_default();

        let records = objects
            .into_iter()
            .map(|obj| {
                let id = obj["doc_id"].as_str().unwrap_or_default().to_string();
                let vector: Vec<f32> = obj["_additional"]["vector"]
                    .as_array()
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_f64().map(|f| f as f32))
                            .collect()
                    })
                    .unwrap_or_default();

                let metadata: HashMap<String, serde_json::Value> = obj["metadata"]
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
        let class_name = Self::class_name(name);

        // Get schema info
        let url = format!("{}/v1/schema/{}", self.base_url, class_name);
        let response = self
            .client
            .get(&url)
            .headers(self.headers())
            .send()
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to get schema: {}", e)))?;

        if !response.status().is_success() {
            return Err(RookError::not_found("Collection", name));
        }

        // Count objects using aggregate
        let count_query = format!(
            r#"{{
                Aggregate {{
                    {}  {{
                        meta {{
                            count
                        }}
                    }}
                }}
            }}"#,
            class_name
        );

        let count_url = format!("{}/v1/graphql", self.base_url);
        let count_response = self
            .client
            .post(&count_url)
            .headers(self.headers())
            .json(&json!({ "query": count_query }))
            .send()
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to count: {}", e)))?;

        let count_body: serde_json::Value = count_response
            .json()
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to parse response: {}", e)))?;

        let count = count_body["data"]["Aggregate"][&class_name][0]["meta"]["count"]
            .as_u64()
            .unwrap_or(0) as usize;

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

impl WeaviateVectorStore {
    fn build_filter(filter: &Filter) -> String {
        let conditions: Vec<String> = filter
            .conditions
            .iter()
            .filter_map(|cond| {
                let path = format!("[\"metadata\", \"{}\"]", cond.field);
                let value = match &cond.value {
                    serde_json::Value::String(s) => format!("\"{}\"", s),
                    serde_json::Value::Number(n) => n.to_string(),
                    serde_json::Value::Bool(b) => b.to_string(),
                    _ => return None,
                };

                let operator = match &cond.operator {
                    rook_core::types::FilterOperator::Eq => "Equal",
                    rook_core::types::FilterOperator::Ne => "NotEqual",
                    rook_core::types::FilterOperator::Gt => "GreaterThan",
                    rook_core::types::FilterOperator::Gte => "GreaterThanEqual",
                    rook_core::types::FilterOperator::Lt => "LessThan",
                    rook_core::types::FilterOperator::Lte => "LessThanEqual",
                    rook_core::types::FilterOperator::Contains => "ContainsAny",
                    _ => return None,
                };

                Some(format!(
                    "{{ path: {}, operator: {}, value{}: {} }}",
                    path,
                    operator,
                    if cond.value.is_string() { "Text" } else { "Number" },
                    value
                ))
            })
            .collect();

        if conditions.len() == 1 {
            conditions.into_iter().next().unwrap()
        } else {
            format!("{{ operator: And, operands: [{}] }}", conditions.join(", "))
        }
    }
}

fn format_uuid(id: &str) -> String {
    // If already a valid UUID, return as-is
    if uuid::Uuid::parse_str(id).is_ok() {
        return id.to_string();
    }

    // Otherwise, create a UUID v5 from the id
    let namespace = uuid::Uuid::NAMESPACE_OID;
    uuid::Uuid::new_v5(&namespace, id.as_bytes()).to_string()
}
