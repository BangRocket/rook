//! PostgreSQL with pgvector extension using deadpool connection pooling.
//!
//! This is the production-ready variant of the pgvector backend with
//! configurable connection pooling via deadpool-postgres.
//!
//! # Features
//!
//! - Configurable pool size and timeouts
//! - Connection recycling for efficient resource usage
//! - Clear error messages for pool exhaustion
//! - Compatible with the standard VectorStore trait
//!
//! # Example
//!
//! ```ignore
//! use rook_vector_stores::PgVectorStorePooled;
//! use rook_core::traits::{VectorStoreConfig, VectorStoreProvider, PostgresPoolConfig};
//!
//! let config = VectorStoreConfig {
//!     provider: VectorStoreProvider::PgvectorPooled,
//!     collection_name: "memories".to_string(),
//!     embedding_model_dims: 1536,
//!     pool: Some(PostgresPoolConfig {
//!         max_size: 16,
//!         wait_timeout_secs: 30,
//!         ..Default::default()
//!     }),
//!     config: serde_json::json!({
//!         "url": "postgres://user:pass@localhost/rook"
//!     }),
//! };
//!
//! let store = PgVectorStorePooled::new(config).await?;
//! ```

use async_trait::async_trait;
use std::collections::HashMap;
use std::str::FromStr;
use std::time::Duration;

use deadpool_postgres::{Manager, ManagerConfig, Pool, RecyclingMethod, Runtime};
use pgvector::Vector;
use tokio_postgres::types::ToSql;
use tokio_postgres::NoTls;

use rook_core::error::{RookError, RookResult};
use rook_core::traits::{
    CollectionInfo, DistanceMetric, VectorRecord, VectorSearchResult, VectorStore,
    VectorStoreConfig,
};
use rook_core::types::{Filter, FilterCondition, FilterOperator};

/// PostgreSQL with pgvector vector store using deadpool connection pooling.
///
/// Recommended for production deployments with high concurrency requirements.
/// Uses deadpool-postgres for efficient connection management with configurable
/// pool size, timeouts, and recycling behavior.
pub struct PgVectorStorePooled {
    pool: Pool,
    config: VectorStoreConfig,
}

impl PgVectorStorePooled {
    /// Create a new pooled pgvector store.
    ///
    /// # Arguments
    ///
    /// * `config` - Vector store configuration including connection URL and optional pool settings
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Connection URL is missing from config
    /// - URL cannot be parsed
    /// - Pool creation fails
    /// - pgvector extension cannot be enabled
    pub async fn new(config: VectorStoreConfig) -> RookResult<Self> {
        let url = config
            .config
            .get("url")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                RookError::Configuration("PostgreSQL connection string required".to_string())
            })?;

        // Parse the URL to extract connection parameters
        let pg_config = tokio_postgres::Config::from_str(url).map_err(|e| {
            RookError::Configuration(format!("Invalid PostgreSQL connection URL: {}", e))
        })?;

        // Get pool configuration or use defaults
        let pool_config = config.pool.clone().unwrap_or_default();

        // Configure pool manager
        let recycling_method = match pool_config.recycling_method.to_lowercase().as_str() {
            "verified" => RecyclingMethod::Verified,
            _ => RecyclingMethod::Fast,
        };

        let manager_config = ManagerConfig {
            recycling_method,
        };

        let manager = Manager::from_config(pg_config, NoTls, manager_config);

        // Build pool with custom configuration
        let pool = Pool::builder(manager)
            .max_size(pool_config.max_size)
            .wait_timeout(Some(Duration::from_secs(pool_config.wait_timeout_secs)))
            .create_timeout(Some(Duration::from_secs(pool_config.create_timeout_secs)))
            .recycle_timeout(Some(Duration::from_secs(pool_config.recycle_timeout_secs)))
            .runtime(Runtime::Tokio1)
            .build()
            .map_err(|e| RookError::vector_store(format!("Failed to create connection pool: {}", e)))?;

        // Ensure pgvector extension is enabled
        let client = pool.get().await.map_err(|e| {
            RookError::vector_store(format!("Failed to get connection from pool: {}", e))
        })?;

        client
            .execute("CREATE EXTENSION IF NOT EXISTS vector", &[])
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to enable pgvector: {}", e)))?;

        Ok(Self { pool, config })
    }

    /// Get a client from the connection pool.
    ///
    /// Returns a clear error message if the pool is exhausted.
    async fn get_client(&self) -> RookResult<deadpool_postgres::Client> {
        self.pool.get().await.map_err(|e| {
            let msg = if e.to_string().contains("timeout") {
                format!(
                    "Connection pool exhausted (timeout waiting for connection). \
                     Consider increasing pool_size (currently: {}). Error: {}",
                    self.config
                        .pool
                        .as_ref()
                        .map(|p| p.max_size)
                        .unwrap_or(16),
                    e
                )
            } else {
                format!("Failed to get connection from pool: {}", e)
            };
            RookError::vector_store(msg)
        })
    }

    fn distance_operator(metric: &DistanceMetric) -> &'static str {
        match metric {
            DistanceMetric::Cosine => "<=>",
            DistanceMetric::Euclidean => "<->",
            DistanceMetric::DotProduct => "<#>",
            DistanceMetric::Manhattan => "<->", // Fallback to Euclidean for Manhattan
        }
    }

    fn index_method(metric: &DistanceMetric) -> &'static str {
        match metric {
            DistanceMetric::Cosine => "vector_cosine_ops",
            DistanceMetric::Euclidean => "vector_l2_ops",
            DistanceMetric::DotProduct => "vector_ip_ops",
            DistanceMetric::Manhattan => "vector_l2_ops", // Fallback for Manhattan
        }
    }

    fn get_distance_metric(&self) -> DistanceMetric {
        self.config
            .config
            .get("distance_metric")
            .and_then(|v| v.as_str())
            .and_then(|s| match s.to_lowercase().as_str() {
                "cosine" => Some(DistanceMetric::Cosine),
                "euclidean" => Some(DistanceMetric::Euclidean),
                "dot_product" | "dotproduct" => Some(DistanceMetric::DotProduct),
                "manhattan" => Some(DistanceMetric::Manhattan),
                _ => None,
            })
            .unwrap_or(DistanceMetric::Cosine)
    }

    fn build_filter(filter: &Filter) -> (String, Vec<String>) {
        Self::build_filter_with_offset(filter, 2)
    }

    fn build_filter_with_offset(filter: &Filter, start_idx: usize) -> (String, Vec<String>) {
        match filter {
            Filter::Condition(cond) => Self::build_condition(cond, start_idx),
            Filter::And(filters) => {
                let mut conditions = Vec::new();
                let mut params = Vec::new();
                let mut param_idx = start_idx;

                for f in filters {
                    let (cond_str, cond_params) = Self::build_filter_with_offset(f, param_idx);
                    if !cond_str.is_empty() {
                        conditions.push(cond_str);
                        param_idx += cond_params.len();
                        params.extend(cond_params);
                    }
                }

                if conditions.is_empty() {
                    (String::new(), vec![])
                } else {
                    (format!("({})", conditions.join(" AND ")), params)
                }
            }
            Filter::Or(filters) => {
                let mut conditions = Vec::new();
                let mut params = Vec::new();
                let mut param_idx = start_idx;

                for f in filters {
                    let (cond_str, cond_params) = Self::build_filter_with_offset(f, param_idx);
                    if !cond_str.is_empty() {
                        conditions.push(cond_str);
                        param_idx += cond_params.len();
                        params.extend(cond_params);
                    }
                }

                if conditions.is_empty() {
                    (String::new(), vec![])
                } else {
                    (format!("({})", conditions.join(" OR ")), params)
                }
            }
            Filter::Not(inner) => {
                let (inner_str, inner_params) = Self::build_filter_with_offset(inner, start_idx);
                if inner_str.is_empty() {
                    (String::new(), vec![])
                } else {
                    (format!("NOT ({})", inner_str), inner_params)
                }
            }
        }
    }

    fn build_condition(cond: &FilterCondition, param_idx: usize) -> (String, Vec<String>) {
        let field = format!("payload->>'{}'", cond.field);

        match &cond.operator {
            FilterOperator::Eq(value) => {
                let param = value.to_string().trim_matches('"').to_string();
                (format!("{} = ${}", field, param_idx), vec![param])
            }
            FilterOperator::Ne(value) => {
                let param = value.to_string().trim_matches('"').to_string();
                (format!("{} != ${}", field, param_idx), vec![param])
            }
            FilterOperator::Gt(value) => {
                let param = value.to_string().trim_matches('"').to_string();
                (
                    format!("({})::numeric > ${}::numeric", field, param_idx),
                    vec![param],
                )
            }
            FilterOperator::Gte(value) => {
                let param = value.to_string().trim_matches('"').to_string();
                (
                    format!("({})::numeric >= ${}::numeric", field, param_idx),
                    vec![param],
                )
            }
            FilterOperator::Lt(value) => {
                let param = value.to_string().trim_matches('"').to_string();
                (
                    format!("({})::numeric < ${}::numeric", field, param_idx),
                    vec![param],
                )
            }
            FilterOperator::Lte(value) => {
                let param = value.to_string().trim_matches('"').to_string();
                (
                    format!("({})::numeric <= ${}::numeric", field, param_idx),
                    vec![param],
                )
            }
            FilterOperator::Contains(text) => {
                let param = format!("%{}%", text);
                (format!("{} ILIKE ${}", field, param_idx), vec![param])
            }
            FilterOperator::Icontains(text) => {
                let param = format!("%{}%", text);
                (
                    format!("LOWER({}) LIKE LOWER(${})", field, param_idx),
                    vec![param],
                )
            }
            FilterOperator::In(values) => {
                let params: Vec<String> = values
                    .iter()
                    .map(|v| v.to_string().trim_matches('"').to_string())
                    .collect();
                let placeholders: Vec<String> = (0..params.len())
                    .map(|i| format!("${}", param_idx + i))
                    .collect();
                (
                    format!("{} IN ({})", field, placeholders.join(", ")),
                    params,
                )
            }
            FilterOperator::Nin(values) => {
                let params: Vec<String> = values
                    .iter()
                    .map(|v| v.to_string().trim_matches('"').to_string())
                    .collect();
                let placeholders: Vec<String> = (0..params.len())
                    .map(|i| format!("${}", param_idx + i))
                    .collect();
                (
                    format!("{} NOT IN ({})", field, placeholders.join(", ")),
                    params,
                )
            }
            FilterOperator::Between { min, max } => {
                let min_param = min.to_string().trim_matches('"').to_string();
                let max_param = max.to_string().trim_matches('"').to_string();
                (
                    format!(
                        "({})::numeric BETWEEN ${}::numeric AND ${}::numeric",
                        field,
                        param_idx,
                        param_idx + 1
                    ),
                    vec![min_param, max_param],
                )
            }
            FilterOperator::IsNull => (format!("{} IS NULL", field), vec![]),
            FilterOperator::IsNotNull => (format!("{} IS NOT NULL", field), vec![]),
            FilterOperator::Exists => (format!("payload ? '{}'", cond.field), vec![]),
            FilterOperator::NotExists => (format!("NOT (payload ? '{}')", cond.field), vec![]),
            FilterOperator::Wildcard => (String::new(), vec![]), // Matches anything, no condition
        }
    }
}

#[async_trait]
impl VectorStore for PgVectorStorePooled {
    async fn create_collection(
        &self,
        name: &str,
        dimension: usize,
        distance: DistanceMetric,
    ) -> RookResult<()> {
        let client = self.get_client().await?;
        let ops = Self::index_method(&distance);

        // Create table
        let create_table = format!(
            r#"
            CREATE TABLE IF NOT EXISTS "{}" (
                id TEXT PRIMARY KEY,
                vector vector({}),
                payload JSONB DEFAULT '{{}}'::jsonb
            )
            "#,
            name, dimension
        );

        client
            .execute(&create_table, &[])
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to create table: {}", e)))?;

        // Create HNSW index for efficient similarity search
        let create_index = format!(
            r#"
            CREATE INDEX IF NOT EXISTS "{}_vector_idx"
            ON "{}" USING hnsw (vector {})
            WITH (m = 16, ef_construction = 64)
            "#,
            name, name, ops
        );

        client
            .execute(&create_index, &[])
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to create index: {}", e)))?;

        Ok(())
    }

    async fn insert(&self, records: Vec<VectorRecord>) -> RookResult<()> {
        if records.is_empty() {
            return Ok(());
        }

        let client = self.get_client().await?;
        let collection_name = self.collection_name();

        for record in records {
            let vector = Vector::from(record.vector);
            let payload = serde_json::to_value(&record.payload)
                .map_err(|e| RookError::vector_store(format!("Failed to serialize payload: {}", e)))?;

            let query = format!(
                r#"
                INSERT INTO "{}" (id, vector, payload)
                VALUES ($1, $2, $3)
                ON CONFLICT (id) DO UPDATE SET
                    vector = EXCLUDED.vector,
                    payload = EXCLUDED.payload
                "#,
                collection_name
            );

            client
                .execute(
                    &query,
                    &[&record.id, &vector, &payload],
                )
                .await
                .map_err(|e| RookError::vector_store(format!("Failed to insert vector: {}", e)))?;
        }

        Ok(())
    }

    async fn search(
        &self,
        query_vector: &[f32],
        limit: usize,
        filters: Option<Filter>,
    ) -> RookResult<Vec<VectorSearchResult>> {
        let client = self.get_client().await?;
        let collection_name = self.collection_name();
        let vector = Vector::from(query_vector.to_vec());
        let metric = self.get_distance_metric();
        let op = Self::distance_operator(&metric);

        let (where_clause, params) = if let Some(f) = filters {
            Self::build_filter(&f)
        } else {
            (String::new(), vec![])
        };

        let query = format!(
            r#"
            SELECT id, vector, payload, (vector {} $1) as distance
            FROM "{}"
            {}
            ORDER BY distance
            LIMIT ${}
            "#,
            op,
            collection_name,
            if where_clause.is_empty() {
                String::new()
            } else {
                format!("WHERE {}", where_clause)
            },
            params.len() + 2
        );

        // Build parameters array
        let limit_i64 = limit as i64;
        let mut query_params: Vec<&(dyn ToSql + Sync)> = vec![&vector];

        // Add filter params
        for param in &params {
            query_params.push(param);
        }
        query_params.push(&limit_i64);

        let rows = client
            .query(&query, &query_params)
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to search: {}", e)))?;

        let results = rows
            .into_iter()
            .map(|row| {
                let id: String = row.get("id");
                let distance: f64 = row.get("distance");
                let payload_value: serde_json::Value = row.get("payload");

                let payload: HashMap<String, serde_json::Value> = payload_value
                    .as_object()
                    .map(|m| m.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
                    .unwrap_or_default();

                // Convert distance to similarity score (1 - distance for cosine)
                let score = match metric {
                    DistanceMetric::Cosine => 1.0 - distance,
                    DistanceMetric::DotProduct => -distance, // Negate because <#> returns negative
                    DistanceMetric::Euclidean | DistanceMetric::Manhattan => 1.0 / (1.0 + distance),
                };

                VectorSearchResult {
                    id,
                    score: score as f32,
                    payload,
                }
            })
            .collect();

        Ok(results)
    }

    async fn get(&self, id: &str) -> RookResult<Option<VectorRecord>> {
        let client = self.get_client().await?;
        let collection_name = self.collection_name();

        let query = format!(
            r#"SELECT id, vector, payload FROM "{}" WHERE id = $1"#,
            collection_name
        );

        let row = client
            .query_opt(&query, &[&id])
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to get vector: {}", e)))?;

        Ok(row.map(|r| {
            let vector: Vector = r.get("vector");
            let payload_value: serde_json::Value = r.get("payload");

            let payload: HashMap<String, serde_json::Value> = payload_value
                .as_object()
                .map(|m| m.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
                .unwrap_or_default();

            VectorRecord {
                id: r.get("id"),
                vector: vector.to_vec(),
                payload,
                score: None,
            }
        }))
    }

    async fn update(
        &self,
        id: &str,
        vector: Option<Vec<f32>>,
        payload: Option<HashMap<String, serde_json::Value>>,
    ) -> RookResult<()> {
        let existing = self.get(id).await?;
        let existing = existing.ok_or_else(|| RookError::not_found(id))?;

        let new_vector = vector.unwrap_or(existing.vector);
        let new_payload = if let Some(p) = payload {
            let mut merged = existing.payload;
            merged.extend(p);
            merged
        } else {
            existing.payload
        };

        let record = VectorRecord {
            id: id.to_string(),
            vector: new_vector,
            payload: new_payload,
            score: None,
        };

        self.insert(vec![record]).await
    }

    async fn delete(&self, id: &str) -> RookResult<()> {
        let client = self.get_client().await?;
        let collection_name = self.collection_name();

        let query = format!(r#"DELETE FROM "{}" WHERE id = $1"#, collection_name);

        client
            .execute(&query, &[&id])
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to delete vector: {}", e)))?;

        Ok(())
    }

    async fn delete_collection(&self, name: &str) -> RookResult<()> {
        let client = self.get_client().await?;

        let query = format!(r#"DROP TABLE IF EXISTS "{}" CASCADE"#, name);

        client
            .execute(&query, &[])
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to drop table: {}", e)))?;

        Ok(())
    }

    async fn list(
        &self,
        filters: Option<Filter>,
        limit: Option<usize>,
    ) -> RookResult<Vec<VectorRecord>> {
        let client = self.get_client().await?;
        let collection_name = self.collection_name();

        let (where_clause, params) = if let Some(f) = filters {
            Self::build_filter(&f)
        } else {
            (String::new(), vec![])
        };

        let limit_clause = limit.map(|l| format!(" LIMIT {}", l)).unwrap_or_default();

        let query = format!(
            r#"SELECT id, vector, payload FROM "{}"{}{}"#,
            collection_name,
            if where_clause.is_empty() {
                String::new()
            } else {
                format!(" WHERE {}", where_clause)
            },
            limit_clause
        );

        let mut query_params: Vec<&(dyn ToSql + Sync)> = vec![];
        for param in &params {
            query_params.push(param);
        }

        let rows = client
            .query(&query, &query_params)
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to list vectors: {}", e)))?;

        let records = rows
            .into_iter()
            .map(|r| {
                let vector: Vector = r.get("vector");
                let payload_value: serde_json::Value = r.get("payload");

                let payload: HashMap<String, serde_json::Value> = payload_value
                    .as_object()
                    .map(|m| m.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
                    .unwrap_or_default();

                VectorRecord {
                    id: r.get("id"),
                    vector: vector.to_vec(),
                    payload,
                    score: None,
                }
            })
            .collect();

        Ok(records)
    }

    async fn list_collections(&self) -> RookResult<Vec<String>> {
        let client = self.get_client().await?;

        let query = r#"
            SELECT tablename FROM pg_tables
            WHERE schemaname = 'public'
            AND tablename NOT LIKE 'pg_%'
            AND tablename NOT LIKE 'sql_%'
        "#;

        let rows = client
            .query(query, &[])
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to list collections: {}", e)))?;

        let collections: Vec<String> = rows.into_iter().map(|r| r.get("tablename")).collect();

        Ok(collections)
    }

    async fn collection_info(&self, name: &str) -> RookResult<CollectionInfo> {
        let client = self.get_client().await?;

        let count_query = format!(r#"SELECT COUNT(*) as count FROM "{}""#, name);

        let row = client
            .query_one(&count_query, &[])
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to get count: {}", e)))?;

        let count: i64 = row.get("count");

        Ok(CollectionInfo {
            name: name.to_string(),
            dimension: self.config.embedding_model_dims,
            vector_count: count as u64,
            distance: self.get_distance_metric(),
        })
    }

    async fn reset(&self) -> RookResult<()> {
        let collection_name = self.collection_name();
        let info = self.collection_info(collection_name).await?;
        self.delete_collection(collection_name).await?;
        self.create_collection(collection_name, info.dimension, info.distance)
            .await?;
        Ok(())
    }

    fn collection_name(&self) -> &str {
        &self.config.collection_name
    }
}
