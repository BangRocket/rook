//! MongoDB Atlas Vector Search implementation.

use async_trait::async_trait;
use std::collections::HashMap;

use rook_core::error::{RookError, RookResult};
use rook_core::traits::{VectorStore, VectorStoreConfig};
use rook_core::types::{CollectionInfo, DistanceMetric, Filter, VectorRecord, VectorSearchResult};

use mongodb::{
    bson::{doc, Document, Bson},
    options::ClientOptions,
    Client, Collection,
};
use serde::{Deserialize, Serialize};

/// MongoDB Atlas vector store implementation.
pub struct MongoDBVectorStore {
    client: Client,
    database: String,
    config: VectorStoreConfig,
}

#[derive(Debug, Serialize, Deserialize)]
struct VectorDocument {
    #[serde(rename = "_id")]
    id: String,
    vector: Vec<f32>,
    metadata: HashMap<String, serde_json::Value>,
}

impl MongoDBVectorStore {
    /// Create a new MongoDB vector store.
    pub async fn new(config: VectorStoreConfig) -> RookResult<Self> {
        let url = config
            .url
            .clone()
            .ok_or_else(|| RookError::Configuration("MongoDB connection string required".to_string()))?;

        let database = config
            .database
            .clone()
            .unwrap_or_else(|| "rook".to_string());

        let client_options = ClientOptions::parse(&url)
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to parse MongoDB URL: {}", e)))?;

        // Add driver info for rook-rs
        let mut client_options = client_options;
        client_options.app_name = Some("rook-rs".to_string());

        let client = Client::with_options(client_options)
            .map_err(|e| RookError::vector_store(format!("Failed to create MongoDB client: {}", e)))?;

        Ok(Self {
            client,
            database,
            config,
        })
    }

    fn get_collection(&self, name: &str) -> Collection<VectorDocument> {
        self.client
            .database(&self.database)
            .collection(name)
    }

    fn distance_to_mongo(metric: &DistanceMetric) -> &'static str {
        match metric {
            DistanceMetric::Cosine => "cosine",
            DistanceMetric::Euclidean => "euclidean",
            DistanceMetric::DotProduct => "dotProduct",
        }
    }
}

#[async_trait]
impl VectorStore for MongoDBVectorStore {
    async fn create_collection(
        &self,
        name: &str,
        dimension: usize,
        distance_metric: Option<DistanceMetric>,
    ) -> RookResult<()> {
        let db = self.client.database(&self.database);
        let metric = distance_metric.unwrap_or(DistanceMetric::Cosine);
        let similarity = Self::distance_to_mongo(&metric);

        // Create collection
        db.create_collection(name, None)
            .await
            .map_err(|e| {
                // Ignore if collection already exists
                if e.to_string().contains("already exists") {
                    return RookError::vector_store("".to_string());
                }
                RookError::vector_store(format!("Failed to create collection: {}", e))
            })
            .ok();

        // Create vector search index
        // Note: Atlas vector search indexes are created via Atlas UI or API
        // This is a placeholder for documentation purposes
        let _index_definition = doc! {
            "name": format!("{}_vector_index", name),
            "type": "vectorSearch",
            "definition": {
                "fields": [{
                    "type": "vector",
                    "path": "vector",
                    "numDimensions": dimension as i32,
                    "similarity": similarity
                }]
            }
        };

        tracing::info!(
            "MongoDB collection '{}' created. Note: Vector search index must be created via Atlas UI/API",
            name
        );

        Ok(())
    }

    async fn insert(&self, collection_name: &str, records: Vec<VectorRecord>) -> RookResult<()> {
        if records.is_empty() {
            return Ok(());
        }

        let collection = self.get_collection(collection_name);

        let docs: Vec<VectorDocument> = records
            .into_iter()
            .map(|r| VectorDocument {
                id: r.id,
                vector: r.vector,
                metadata: r.metadata,
            })
            .collect();

        // Use replace_one with upsert for each document to handle updates
        for doc in docs {
            let filter = doc! { "_id": &doc.id };
            let replacement = mongodb::bson::to_document(&doc)
                .map_err(|e| RookError::vector_store(format!("Failed to serialize document: {}", e)))?;

            collection
                .replace_one(filter, replacement, mongodb::options::ReplaceOptions::builder().upsert(true).build())
                .await
                .map_err(|e| RookError::vector_store(format!("Failed to insert document: {}", e)))?;
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
        let collection: Collection<Document> = self
            .client
            .database(&self.database)
            .collection(collection_name);

        // Build the vector search pipeline
        let mut pipeline = vec![
            doc! {
                "$vectorSearch": {
                    "index": format!("{}_vector_index", collection_name),
                    "path": "vector",
                    "queryVector": query_vector.iter().map(|&f| Bson::Double(f as f64)).collect::<Vec<_>>(),
                    "numCandidates": (limit * 10) as i32,
                    "limit": limit as i32
                }
            },
            doc! {
                "$project": {
                    "_id": 1,
                    "metadata": 1,
                    "score": { "$meta": "vectorSearchScore" }
                }
            },
        ];

        // Add filter if provided
        if let Some(f) = filter {
            let filter_doc = Self::build_filter(&f);
            pipeline.insert(1, doc! { "$match": filter_doc });
        }

        let mut cursor = collection
            .aggregate(pipeline, None)
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to search: {}", e)))?;

        let mut results = Vec::new();
        while cursor.advance().await.map_err(|e| RookError::vector_store(format!("Cursor error: {}", e)))? {
            let doc = cursor.deserialize_current()
                .map_err(|e| RookError::vector_store(format!("Failed to deserialize: {}", e)))?;

            let id = doc.get_str("_id").unwrap_or_default().to_string();
            let score = doc.get_f64("score").unwrap_or(0.0) as f32;

            let metadata: HashMap<String, serde_json::Value> = doc
                .get_document("metadata")
                .ok()
                .map(|m| {
                    m.iter()
                        .map(|(k, v)| (k.clone(), bson_to_json(v.clone())))
                        .collect()
                })
                .unwrap_or_default();

            results.push(VectorSearchResult {
                id,
                score,
                vector: None,
                metadata,
            });
        }

        Ok(results)
    }

    async fn get(&self, collection_name: &str, id: &str) -> RookResult<Option<VectorRecord>> {
        let collection = self.get_collection(collection_name);

        let result = collection
            .find_one(doc! { "_id": id }, None)
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to get document: {}", e)))?;

        Ok(result.map(|doc| VectorRecord {
            id: doc.id,
            vector: doc.vector,
            metadata: doc.metadata,
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
        let collection = self.get_collection(collection_name);

        collection
            .delete_one(doc! { "_id": id }, None)
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to delete document: {}", e)))?;

        Ok(())
    }

    async fn delete_collection(&self, name: &str) -> RookResult<()> {
        let collection = self.get_collection(name);

        collection
            .drop(None)
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to drop collection: {}", e)))?;

        Ok(())
    }

    async fn list(
        &self,
        collection_name: &str,
        filter: Option<Filter>,
        limit: Option<usize>,
    ) -> RookResult<Vec<VectorRecord>> {
        let collection = self.get_collection(collection_name);

        let filter_doc = filter.map(|f| Self::build_filter(&f)).unwrap_or_else(|| doc! {});
        let options = mongodb::options::FindOptions::builder()
            .limit(limit.map(|l| l as i64))
            .build();

        let mut cursor = collection
            .find(filter_doc, options)
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to list documents: {}", e)))?;

        let mut records = Vec::new();
        while cursor.advance().await.map_err(|e| RookError::vector_store(format!("Cursor error: {}", e)))? {
            let doc = cursor.deserialize_current()
                .map_err(|e| RookError::vector_store(format!("Failed to deserialize: {}", e)))?;

            records.push(VectorRecord {
                id: doc.id,
                vector: doc.vector,
                metadata: doc.metadata,
            });
        }

        Ok(records)
    }

    async fn collection_info(&self, name: &str) -> RookResult<CollectionInfo> {
        let collection = self.get_collection(name);

        let count = collection
            .count_documents(doc! {}, None)
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to count documents: {}", e)))?;

        Ok(CollectionInfo {
            name: name.to_string(),
            dimension: self.config.embedding_dims.unwrap_or(1536),
            count: count as usize,
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

impl MongoDBVectorStore {
    fn build_filter(filter: &Filter) -> Document {
        let conditions: Vec<Document> = filter
            .conditions
            .iter()
            .filter_map(|cond| {
                let field = format!("metadata.{}", cond.field);
                let value = json_to_bson(cond.value.clone());

                match &cond.operator {
                    rook_core::types::FilterOperator::Eq => Some(doc! { &field: value }),
                    rook_core::types::FilterOperator::Ne => Some(doc! { &field: { "$ne": value } }),
                    rook_core::types::FilterOperator::Gt => Some(doc! { &field: { "$gt": value } }),
                    rook_core::types::FilterOperator::Gte => Some(doc! { &field: { "$gte": value } }),
                    rook_core::types::FilterOperator::Lt => Some(doc! { &field: { "$lt": value } }),
                    rook_core::types::FilterOperator::Lte => Some(doc! { &field: { "$lte": value } }),
                    rook_core::types::FilterOperator::In => {
                        if let Bson::Array(arr) = value {
                            Some(doc! { &field: { "$in": arr } })
                        } else {
                            None
                        }
                    }
                    rook_core::types::FilterOperator::NotIn => {
                        if let Bson::Array(arr) = value {
                            Some(doc! { &field: { "$nin": arr } })
                        } else {
                            None
                        }
                    }
                    _ => None,
                }
            })
            .collect();

        if conditions.len() == 1 {
            conditions.into_iter().next().unwrap()
        } else if conditions.is_empty() {
            doc! {}
        } else {
            doc! { "$and": conditions }
        }
    }
}

fn json_to_bson(value: serde_json::Value) -> Bson {
    match value {
        serde_json::Value::Null => Bson::Null,
        serde_json::Value::Bool(b) => Bson::Boolean(b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Bson::Int64(i)
            } else if let Some(f) = n.as_f64() {
                Bson::Double(f)
            } else {
                Bson::Null
            }
        }
        serde_json::Value::String(s) => Bson::String(s),
        serde_json::Value::Array(arr) => {
            Bson::Array(arr.into_iter().map(json_to_bson).collect())
        }
        serde_json::Value::Object(obj) => {
            let doc: Document = obj
                .into_iter()
                .map(|(k, v)| (k, json_to_bson(v)))
                .collect();
            Bson::Document(doc)
        }
    }
}

fn bson_to_json(value: Bson) -> serde_json::Value {
    match value {
        Bson::Null => serde_json::Value::Null,
        Bson::Boolean(b) => serde_json::Value::Bool(b),
        Bson::Int32(i) => serde_json::Value::Number(i.into()),
        Bson::Int64(i) => serde_json::Value::Number(i.into()),
        Bson::Double(f) => serde_json::Number::from_f64(f)
            .map(serde_json::Value::Number)
            .unwrap_or(serde_json::Value::Null),
        Bson::String(s) => serde_json::Value::String(s),
        Bson::Array(arr) => serde_json::Value::Array(arr.into_iter().map(bson_to_json).collect()),
        Bson::Document(doc) => {
            serde_json::Value::Object(doc.into_iter().map(|(k, v)| (k, bson_to_json(v))).collect())
        }
        _ => serde_json::Value::Null,
    }
}
