//! SQLite vector store implementation.

use async_trait::async_trait;
use std::collections::HashMap;

use rook_core::error::{RookError, RookResult};
use rook_core::traits::{VectorStore, VectorStoreConfig};
use rook_core::types::{CollectionInfo, DistanceMetric, Filter, VectorRecord, VectorSearchResult};

use sqlx::{sqlite::SqlitePoolOptions, SqlitePool, Row};

/// SQLite vector store implementation.
pub struct SQLiteVectorStore {
    pool: SqlitePool,
    config: VectorStoreConfig,
}

impl SQLiteVectorStore {
    /// Create a new SQLite vector store.
    pub async fn new(config: VectorStoreConfig) -> RookResult<Self> {
        let url = config
            .url
            .clone()
            .unwrap_or_else(|| "sqlite::memory:".to_string());

        let pool = SqlitePoolOptions::new()
            .max_connections(5)
            .connect(&url)
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to connect to SQLite: {}", e)))?;

        Ok(Self { pool, config })
    }
}

#[async_trait]
impl VectorStore for SQLiteVectorStore {
    async fn create_collection(
        &self,
        name: &str,
        dimension: usize,
        _distance_metric: Option<DistanceMetric>,
    ) -> RookResult<()> {
        let create_table = format!(
            r#"
            CREATE TABLE IF NOT EXISTS "{}" (
                id TEXT PRIMARY KEY,
                vector TEXT NOT NULL,
                metadata TEXT DEFAULT '{{}}',
                dimension INTEGER DEFAULT {}
            )
            "#,
            name, dimension
        );

        sqlx::query(&create_table)
            .execute(&self.pool)
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to create table: {}", e)))?;

        Ok(())
    }

    async fn insert(&self, collection_name: &str, records: Vec<VectorRecord>) -> RookResult<()> {
        if records.is_empty() {
            return Ok(());
        }

        for record in records {
            let vector_json = serde_json::to_string(&record.vector)
                .map_err(|e| RookError::vector_store(format!("Failed to serialize vector: {}", e)))?;
            let metadata_json = serde_json::to_string(&record.metadata)
                .map_err(|e| RookError::vector_store(format!("Failed to serialize metadata: {}", e)))?;

            let query = format!(
                r#"
                INSERT INTO "{}" (id, vector, metadata)
                VALUES (?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    vector = excluded.vector,
                    metadata = excluded.metadata
                "#,
                collection_name
            );

            sqlx::query(&query)
                .bind(&record.id)
                .bind(&vector_json)
                .bind(&metadata_json)
                .execute(&self.pool)
                .await
                .map_err(|e| RookError::vector_store(format!("Failed to insert vector: {}", e)))?;
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
        let where_clause = if let Some(f) = filter {
            format!("WHERE {}", Self::build_filter(&f))
        } else {
            String::new()
        };

        let query = format!(
            r#"SELECT id, vector, metadata FROM "{}"{}"#,
            collection_name, where_clause
        );

        let rows = sqlx::query(&query)
            .fetch_all(&self.pool)
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to search: {}", e)))?;

        let mut results: Vec<VectorSearchResult> = rows
            .into_iter()
            .filter_map(|row| {
                let id: String = row.get("id");
                let vector_json: String = row.get("vector");
                let metadata_json: String = row.get("metadata");

                let vector: Vec<f32> = serde_json::from_str(&vector_json).ok()?;
                let metadata: HashMap<String, serde_json::Value> =
                    serde_json::from_str(&metadata_json).unwrap_or_default();

                let score = cosine_similarity(&query_vector, &vector);

                Some(VectorSearchResult {
                    id,
                    score,
                    vector: Some(vector),
                    metadata,
                })
            })
            .collect();

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);

        Ok(results)
    }

    async fn get(&self, collection_name: &str, id: &str) -> RookResult<Option<VectorRecord>> {
        let query = format!(
            r#"SELECT id, vector, metadata FROM "{}" WHERE id = ?"#,
            collection_name
        );

        let row = sqlx::query(&query)
            .bind(id)
            .fetch_optional(&self.pool)
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to get vector: {}", e)))?;

        Ok(row.map(|r| {
            let vector_json: String = r.get("vector");
            let metadata_json: String = r.get("metadata");

            let vector: Vec<f32> = serde_json::from_str(&vector_json).unwrap_or_default();
            let metadata: HashMap<String, serde_json::Value> =
                serde_json::from_str(&metadata_json).unwrap_or_default();

            VectorRecord {
                id: r.get("id"),
                vector,
                metadata,
            }
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
        let query = format!(r#"DELETE FROM "{}" WHERE id = ?"#, collection_name);

        sqlx::query(&query)
            .bind(id)
            .execute(&self.pool)
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to delete vector: {}", e)))?;

        Ok(())
    }

    async fn delete_collection(&self, name: &str) -> RookResult<()> {
        let query = format!(r#"DROP TABLE IF EXISTS "{}""#, name);

        sqlx::query(&query)
            .execute(&self.pool)
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to drop table: {}", e)))?;

        Ok(())
    }

    async fn list(
        &self,
        collection_name: &str,
        filter: Option<Filter>,
        limit: Option<usize>,
    ) -> RookResult<Vec<VectorRecord>> {
        let where_clause = if let Some(f) = filter {
            format!(" WHERE {}", Self::build_filter(&f))
        } else {
            String::new()
        };

        let limit_clause = limit.map(|l| format!(" LIMIT {}", l)).unwrap_or_default();

        let query = format!(
            r#"SELECT id, vector, metadata FROM "{}"{}{}"#,
            collection_name, where_clause, limit_clause
        );

        let rows = sqlx::query(&query)
            .fetch_all(&self.pool)
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to list vectors: {}", e)))?;

        let records = rows
            .into_iter()
            .map(|r| {
                let vector_json: String = r.get("vector");
                let metadata_json: String = r.get("metadata");

                let vector: Vec<f32> = serde_json::from_str(&vector_json).unwrap_or_default();
                let metadata: HashMap<String, serde_json::Value> =
                    serde_json::from_str(&metadata_json).unwrap_or_default();

                VectorRecord {
                    id: r.get("id"),
                    vector,
                    metadata,
                }
            })
            .collect();

        Ok(records)
    }

    async fn collection_info(&self, name: &str) -> RookResult<CollectionInfo> {
        let query = format!(r#"SELECT COUNT(*) as count FROM "{}""#, name);

        let row = sqlx::query(&query)
            .fetch_one(&self.pool)
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to get count: {}", e)))?;

        let count: i32 = row.get("count");

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

impl SQLiteVectorStore {
    fn build_filter(filter: &Filter) -> String {
        let conditions: Vec<String> = filter
            .conditions
            .iter()
            .filter_map(|cond| {
                let field = format!("json_extract(metadata, '$.{}')", cond.field);
                let value = match &cond.value {
                    serde_json::Value::String(s) => format!("'{}'", s.replace('\'', "''")),
                    serde_json::Value::Number(n) => n.to_string(),
                    serde_json::Value::Bool(b) => if *b { "1" } else { "0" }.to_string(),
                    _ => return None,
                };

                match &cond.operator {
                    rook_core::types::FilterOperator::Eq => Some(format!("{} = {}", field, value)),
                    rook_core::types::FilterOperator::Ne => Some(format!("{} != {}", field, value)),
                    rook_core::types::FilterOperator::Gt => Some(format!("{} > {}", field, value)),
                    rook_core::types::FilterOperator::Gte => Some(format!("{} >= {}", field, value)),
                    rook_core::types::FilterOperator::Lt => Some(format!("{} < {}", field, value)),
                    rook_core::types::FilterOperator::Lte => Some(format!("{} <= {}", field, value)),
                    rook_core::types::FilterOperator::Contains => {
                        Some(format!("{} LIKE '%{}%'", field, value.trim_matches('\'')))
                    }
                    _ => None,
                }
            })
            .collect();

        conditions.join(" AND ")
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot_product / (norm_a * norm_b)
}
