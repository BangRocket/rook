//! Databricks vector store implementation (stub).

use async_trait::async_trait;
use std::collections::HashMap;

use rook_core::error::{RookError, RookResult};
use rook_core::traits::{VectorStore, VectorStoreConfig};
use rook_core::types::{CollectionInfo, DistanceMetric, Filter, VectorRecord, VectorSearchResult};

pub struct DatabricksVectorStore {
    config: VectorStoreConfig,
    storage: tokio::sync::RwLock<HashMap<String, HashMap<String, VectorRecord>>>,
}

impl DatabricksVectorStore {
    pub async fn new(config: VectorStoreConfig) -> RookResult<Self> {
        Ok(Self { config, storage: tokio::sync::RwLock::new(HashMap::new()) })
    }
}

#[async_trait]
impl VectorStore for DatabricksVectorStore {
    async fn create_collection(&self, name: &str, _dimension: usize, _distance_metric: Option<DistanceMetric>) -> RookResult<()> {
        self.storage.write().await.entry(name.to_string()).or_insert_with(HashMap::new);
        Ok(())
    }

    async fn insert(&self, collection_name: &str, records: Vec<VectorRecord>) -> RookResult<()> {
        let mut storage = self.storage.write().await;
        let collection = storage.entry(collection_name.to_string()).or_insert_with(HashMap::new);
        for r in records { collection.insert(r.id.clone(), r); }
        Ok(())
    }

    async fn search(&self, collection_name: &str, query_vector: Vec<f32>, limit: usize, _filter: Option<Filter>) -> RookResult<Vec<VectorSearchResult>> {
        let storage = self.storage.read().await;
        let collection = storage.get(collection_name).ok_or_else(|| RookError::not_found("Collection", collection_name))?;
        let mut results: Vec<_> = collection.values().map(|r| VectorSearchResult { id: r.id.clone(), score: cosine_sim(&query_vector, &r.vector), vector: Some(r.vector.clone()), metadata: r.metadata.clone() }).collect();
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);
        Ok(results)
    }

    async fn get(&self, collection_name: &str, id: &str) -> RookResult<Option<VectorRecord>> {
        Ok(self.storage.read().await.get(collection_name).and_then(|c| c.get(id).cloned()))
    }

    async fn update(&self, collection_name: &str, id: &str, vector: Option<Vec<f32>>, metadata: Option<HashMap<String, serde_json::Value>>) -> RookResult<()> {
        if let Some(c) = self.storage.write().await.get_mut(collection_name) {
            if let Some(r) = c.get_mut(id) { if let Some(v) = vector { r.vector = v; } if let Some(m) = metadata { r.metadata.extend(m); } }
        }
        Ok(())
    }

    async fn delete(&self, collection_name: &str, id: &str) -> RookResult<()> {
        if let Some(c) = self.storage.write().await.get_mut(collection_name) { c.remove(id); }
        Ok(())
    }

    async fn delete_collection(&self, name: &str) -> RookResult<()> {
        self.storage.write().await.remove(name);
        Ok(())
    }

    async fn list(&self, collection_name: &str, _filter: Option<Filter>, limit: Option<usize>) -> RookResult<Vec<VectorRecord>> {
        let mut records: Vec<_> = self.storage.read().await.get(collection_name).map(|c| c.values().cloned().collect()).unwrap_or_default();
        if let Some(l) = limit { records.truncate(l); }
        Ok(records)
    }

    async fn collection_info(&self, name: &str) -> RookResult<CollectionInfo> {
        let count = self.storage.read().await.get(name).map(|c| c.len()).unwrap_or(0);
        Ok(CollectionInfo { name: name.to_string(), dimension: self.config.embedding_dims.unwrap_or(1536), count, distance_metric: DistanceMetric::Cosine })
    }

    async fn reset(&self, collection_name: &str) -> RookResult<()> {
        if let Some(c) = self.storage.write().await.get_mut(collection_name) { c.clear(); }
        Ok(())
    }
}

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() { return 0.0; }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 { 0.0 } else { dot / (na * nb) }
}
