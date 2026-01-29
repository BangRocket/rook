//! OpenSearch vector store implementation.

use async_trait::async_trait;
use std::collections::HashMap;

use rook_core::error::{RookError, RookResult};
use rook_core::traits::{VectorStore, VectorStoreConfig};
use rook_core::types::{CollectionInfo, DistanceMetric, Filter, VectorRecord, VectorSearchResult};

use opensearch::{
    http::transport::Transport,
    indices::{IndicesCreateParts, IndicesDeleteParts, IndicesExistsParts},
    OpenSearch, GetParts, IndexParts, SearchParts, DeleteParts, BulkParts,
};
use serde_json::json;

/// OpenSearch vector store implementation.
pub struct OpenSearchVectorStore {
    client: OpenSearch,
    config: VectorStoreConfig,
}

impl OpenSearchVectorStore {
    /// Create a new OpenSearch vector store.
    pub async fn new(config: VectorStoreConfig) -> RookResult<Self> {
        let url = config
            .url
            .clone()
            .unwrap_or_else(|| "http://localhost:9200".to_string());

        let transport = Transport::single_node(&url)
            .map_err(|e| RookError::vector_store(format!("Failed to create OS transport: {}", e)))?;

        let client = OpenSearch::new(transport);

        Ok(Self { client, config })
    }

    fn distance_to_os(metric: &DistanceMetric) -> &'static str {
        match metric {
            DistanceMetric::Cosine => "cosinesimil",
            DistanceMetric::Euclidean => "l2",
            DistanceMetric::DotProduct => "innerproduct",
        }
    }
}

#[async_trait]
impl VectorStore for OpenSearchVectorStore {
    async fn create_collection(
        &self,
        name: &str,
        dimension: usize,
        distance_metric: Option<DistanceMetric>,
    ) -> RookResult<()> {
        let metric = distance_metric.unwrap_or(DistanceMetric::Cosine);
        let space_type = Self::distance_to_os(&metric);

        let settings = json!({
            "settings": {
                "index": {
                    "knn": true,
                    "knn.algo_param.ef_search": 100
                }
            },
            "mappings": {
                "properties": {
                    "vector": {
                        "type": "knn_vector",
                        "dimension": dimension,
                        "method": {
                            "name": "hnsw",
                            "space_type": space_type,
                            "engine": "nmslib",
                            "parameters": {
                                "ef_construction": 128,
                                "m": 24
                            }
                        }
                    },
                    "metadata": {
                        "type": "object",
                        "enabled": true
                    },
                    "id": {
                        "type": "keyword"
                    }
                }
            }
        });

        let response = self
            .client
            .indices()
            .create(IndicesCreateParts::Index(name))
            .body(settings)
            .send()
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to create index: {}", e)))?;

        if !response.status_code().is_success() {
            let body = response.text().await.unwrap_or_default();
            if !body.contains("resource_already_exists_exception") {
                return Err(RookError::vector_store(format!(
                    "Failed to create index: {}",
                    body
                )));
            }
        }

        Ok(())
    }

    async fn insert(&self, collection_name: &str, records: Vec<VectorRecord>) -> RookResult<()> {
        if records.is_empty() {
            return Ok(());
        }

        let mut body: Vec<serde_json::Value> = Vec::new();
        for record in records {
            body.push(json!({ "index": { "_id": record.id } }));
            body.push(json!({
                "id": record.id,
                "vector": record.vector,
                "metadata": record.metadata
            }));
        }

        let response = self
            .client
            .bulk(BulkParts::Index(collection_name))
            .body(body)
            .send()
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to insert vectors: {}", e)))?;

        if !response.status_code().is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(RookError::vector_store(format!(
                "Failed to insert vectors: {}",
                body
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
        let mut query = json!({
            "size": limit,
            "query": {
                "knn": {
                    "vector": {
                        "vector": query_vector,
                        "k": limit
                    }
                }
            },
            "_source": ["id", "metadata"]
        });

        if let Some(f) = filter {
            query["query"] = json!({
                "bool": {
                    "must": {
                        "knn": {
                            "vector": {
                                "vector": query_vector,
                                "k": limit
                            }
                        }
                    },
                    "filter": Self::build_filter(&f)
                }
            });
        }

        let response = self
            .client
            .search(SearchParts::Index(&[collection_name]))
            .body(query)
            .send()
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to search: {}", e)))?;

        let body: serde_json::Value = response
            .json()
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to parse response: {}", e)))?;

        let hits = body["hits"]["hits"]
            .as_array()
            .cloned()
            .unwrap_or_default();

        let results = hits
            .into_iter()
            .map(|hit| {
                let id = hit["_source"]["id"]
                    .as_str()
                    .unwrap_or_default()
                    .to_string();
                let score = hit["_score"].as_f64().unwrap_or(0.0) as f32;
                let metadata: HashMap<String, serde_json::Value> = hit["_source"]["metadata"]
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
        let response = self
            .client
            .get(GetParts::IndexId(collection_name, id))
            .send()
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to get document: {}", e)))?;

        if response.status_code().as_u16() == 404 {
            return Ok(None);
        }

        let body: serde_json::Value = response
            .json()
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to parse response: {}", e)))?;

        if !body["found"].as_bool().unwrap_or(false) {
            return Ok(None);
        }

        let source = &body["_source"];
        let vector: Vec<f32> = source["vector"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_f64().map(|f| f as f32))
                    .collect()
            })
            .unwrap_or_default();

        let metadata: HashMap<String, serde_json::Value> = source["metadata"]
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
        let response = self
            .client
            .delete(DeleteParts::IndexId(collection_name, id))
            .send()
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to delete document: {}", e)))?;

        if !response.status_code().is_success() && response.status_code().as_u16() != 404 {
            let body = response.text().await.unwrap_or_default();
            return Err(RookError::vector_store(format!(
                "Failed to delete document: {}",
                body
            )));
        }

        Ok(())
    }

    async fn delete_collection(&self, name: &str) -> RookResult<()> {
        let response = self
            .client
            .indices()
            .delete(IndicesDeleteParts::Index(&[name]))
            .send()
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to delete index: {}", e)))?;

        if !response.status_code().is_success() && response.status_code().as_u16() != 404 {
            let body = response.text().await.unwrap_or_default();
            return Err(RookError::vector_store(format!(
                "Failed to delete index: {}",
                body
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
        let size = limit.unwrap_or(100);

        let mut query = json!({
            "query": {
                "match_all": {}
            },
            "size": size,
            "_source": ["id", "vector", "metadata"]
        });

        if let Some(f) = filter {
            query["query"] = json!({
                "bool": {
                    "filter": Self::build_filter(&f)
                }
            });
        }

        let response = self
            .client
            .search(SearchParts::Index(&[collection_name]))
            .body(query)
            .send()
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to list documents: {}", e)))?;

        let body: serde_json::Value = response
            .json()
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to parse response: {}", e)))?;

        let hits = body["hits"]["hits"]
            .as_array()
            .cloned()
            .unwrap_or_default();

        let records = hits
            .into_iter()
            .map(|hit| {
                let source = &hit["_source"];
                let id = source["id"].as_str().unwrap_or_default().to_string();
                let vector: Vec<f32> = source["vector"]
                    .as_array()
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_f64().map(|f| f as f32))
                            .collect()
                    })
                    .unwrap_or_default();
                let metadata: HashMap<String, serde_json::Value> = source["metadata"]
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
        let exists_response = self
            .client
            .indices()
            .exists(IndicesExistsParts::Index(&[name]))
            .send()
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to check index: {}", e)))?;

        if !exists_response.status_code().is_success() {
            return Err(RookError::not_found("Collection", name));
        }

        let count_response = self
            .client
            .search(SearchParts::Index(&[name]))
            .body(json!({ "query": { "match_all": {} }, "size": 0, "track_total_hits": true }))
            .send()
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to get count: {}", e)))?;

        let count_body: serde_json::Value = count_response
            .json()
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to parse response: {}", e)))?;

        let count = count_body["hits"]["total"]["value"].as_u64().unwrap_or(0) as usize;

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

impl OpenSearchVectorStore {
    fn build_filter(filter: &Filter) -> serde_json::Value {
        let conditions: Vec<serde_json::Value> = filter
            .conditions
            .iter()
            .filter_map(|cond| {
                let field = format!("metadata.{}", cond.field);
                match &cond.operator {
                    rook_core::types::FilterOperator::Eq => {
                        Some(json!({ "term": { &field: cond.value } }))
                    }
                    rook_core::types::FilterOperator::Ne => {
                        Some(json!({ "bool": { "must_not": { "term": { &field: cond.value } } } }))
                    }
                    rook_core::types::FilterOperator::Gt => {
                        Some(json!({ "range": { &field: { "gt": cond.value } } }))
                    }
                    rook_core::types::FilterOperator::Gte => {
                        Some(json!({ "range": { &field: { "gte": cond.value } } }))
                    }
                    rook_core::types::FilterOperator::Lt => {
                        Some(json!({ "range": { &field: { "lt": cond.value } } }))
                    }
                    rook_core::types::FilterOperator::Lte => {
                        Some(json!({ "range": { &field: { "lte": cond.value } } }))
                    }
                    _ => None,
                }
            })
            .collect();

        if conditions.len() == 1 {
            conditions.into_iter().next().unwrap()
        } else {
            json!({ "bool": { "must": conditions } })
        }
    }
}
