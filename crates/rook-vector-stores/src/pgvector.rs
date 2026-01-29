//! PostgreSQL with pgvector extension implementation.

use async_trait::async_trait;
use std::collections::HashMap;

use rook_core::error::{RookError, RookResult};
use rook_core::traits::{
    CollectionInfo, DistanceMetric, VectorRecord, VectorSearchResult, VectorStore,
    VectorStoreConfig,
};
use rook_core::types::{Filter, FilterCondition, FilterOperator};

use pgvector::Vector;
use sqlx::{postgres::PgPoolOptions, PgPool, Row};

/// PostgreSQL with pgvector vector store implementation.
pub struct PgVectorStore {
    pool: PgPool,
    config: VectorStoreConfig,
}

impl PgVectorStore {
    /// Create a new pgvector store.
    pub async fn new(config: VectorStoreConfig) -> RookResult<Self> {
        let url = config
            .config
            .get("url")
            .and_then(|v| v.as_str())
            .ok_or_else(|| RookError::Configuration("PostgreSQL connection string required".to_string()))?;

        let pool = PgPoolOptions::new()
            .max_connections(5)
            .connect(url)
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to connect to PostgreSQL: {}", e)))?;

        // Ensure pgvector extension is enabled
        sqlx::query("CREATE EXTENSION IF NOT EXISTS vector")
            .execute(&pool)
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to enable pgvector: {}", e)))?;

        Ok(Self { pool, config })
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
}

#[async_trait]
impl VectorStore for PgVectorStore {
    async fn create_collection(
        &self,
        name: &str,
        dimension: usize,
        distance: DistanceMetric,
    ) -> RookResult<()> {
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

        sqlx::query(&create_table)
            .execute(&self.pool)
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

        sqlx::query(&create_index)
            .execute(&self.pool)
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to create index: {}", e)))?;

        Ok(())
    }

    async fn insert(&self, records: Vec<VectorRecord>) -> RookResult<()> {
        if records.is_empty() {
            return Ok(());
        }

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

            sqlx::query(&query)
                .bind(&record.id)
                .bind(&vector)
                .bind(&payload)
                .execute(&self.pool)
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
            if where_clause.is_empty() { String::new() } else { format!("WHERE {}", where_clause) },
            params.len() + 2
        );

        let mut query_builder = sqlx::query(&query).bind(&vector);

        for param in &params {
            query_builder = query_builder.bind(param);
        }

        query_builder = query_builder.bind(limit as i64);

        let rows = query_builder
            .fetch_all(&self.pool)
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
        let collection_name = self.collection_name();
        let query = format!(
            r#"SELECT id, vector, payload FROM "{}" WHERE id = $1"#,
            collection_name
        );

        let row = sqlx::query(&query)
            .bind(id)
            .fetch_optional(&self.pool)
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
        let collection_name = self.collection_name();
        let query = format!(r#"DELETE FROM "{}" WHERE id = $1"#, collection_name);

        sqlx::query(&query)
            .bind(id)
            .execute(&self.pool)
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to delete vector: {}", e)))?;

        Ok(())
    }

    async fn delete_collection(&self, name: &str) -> RookResult<()> {
        let query = format!(r#"DROP TABLE IF EXISTS "{}" CASCADE"#, name);

        sqlx::query(&query)
            .execute(&self.pool)
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to drop table: {}", e)))?;

        Ok(())
    }

    async fn list(
        &self,
        filters: Option<Filter>,
        limit: Option<usize>,
    ) -> RookResult<Vec<VectorRecord>> {
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
            if where_clause.is_empty() { String::new() } else { format!(" WHERE {}", where_clause) },
            limit_clause
        );

        let mut query_builder = sqlx::query(&query);

        for param in &params {
            query_builder = query_builder.bind(param);
        }

        let rows = query_builder
            .fetch_all(&self.pool)
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
        let query = r#"
            SELECT tablename FROM pg_tables
            WHERE schemaname = 'public'
            AND tablename NOT LIKE 'pg_%'
            AND tablename NOT LIKE 'sql_%'
        "#;

        let rows = sqlx::query(query)
            .fetch_all(&self.pool)
            .await
            .map_err(|e| RookError::vector_store(format!("Failed to list collections: {}", e)))?;

        let collections: Vec<String> = rows
            .into_iter()
            .map(|r| r.get("tablename"))
            .collect();

        Ok(collections)
    }

    async fn collection_info(&self, name: &str) -> RookResult<CollectionInfo> {
        let count_query = format!(r#"SELECT COUNT(*) as count FROM "{}""#, name);

        let row = sqlx::query(&count_query)
            .fetch_one(&self.pool)
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

impl PgVectorStore {
    fn build_filter(filter: &Filter) -> (String, Vec<String>) {
        match filter {
            Filter::Condition(cond) => Self::build_condition(cond, 2),
            Filter::And(filters) => {
                let mut conditions = Vec::new();
                let mut params = Vec::new();
                let mut param_idx = 2;

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
                let mut param_idx = 2;

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
                let (inner_str, inner_params) = Self::build_filter(inner);
                if inner_str.is_empty() {
                    (String::new(), vec![])
                } else {
                    (format!("NOT ({})", inner_str), inner_params)
                }
            }
        }
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
        let field = format!("payload->>'{}'"  , cond.field);

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
                (format!("({})::numeric > ${}::numeric", field, param_idx), vec![param])
            }
            FilterOperator::Gte(value) => {
                let param = value.to_string().trim_matches('"').to_string();
                (format!("({})::numeric >= ${}::numeric", field, param_idx), vec![param])
            }
            FilterOperator::Lt(value) => {
                let param = value.to_string().trim_matches('"').to_string();
                (format!("({})::numeric < ${}::numeric", field, param_idx), vec![param])
            }
            FilterOperator::Lte(value) => {
                let param = value.to_string().trim_matches('"').to_string();
                (format!("({})::numeric <= ${}::numeric", field, param_idx), vec![param])
            }
            FilterOperator::Contains(text) => {
                let param = format!("%{}%", text);
                (format!("{} ILIKE ${}", field, param_idx), vec![param])
            }
            FilterOperator::Icontains(text) => {
                let param = format!("%{}%", text);
                (format!("LOWER({}) LIKE LOWER(${})", field, param_idx), vec![param])
            }
            FilterOperator::In(values) => {
                let params: Vec<String> = values
                    .iter()
                    .map(|v| v.to_string().trim_matches('"').to_string())
                    .collect();
                let placeholders: Vec<String> = (0..params.len())
                    .map(|i| format!("${}", param_idx + i))
                    .collect();
                (format!("{} IN ({})", field, placeholders.join(", ")), params)
            }
            FilterOperator::Nin(values) => {
                let params: Vec<String> = values
                    .iter()
                    .map(|v| v.to_string().trim_matches('"').to_string())
                    .collect();
                let placeholders: Vec<String> = (0..params.len())
                    .map(|i| format!("${}", param_idx + i))
                    .collect();
                (format!("{} NOT IN ({})", field, placeholders.join(", ")), params)
            }
            FilterOperator::Between { min, max } => {
                let min_param = min.to_string().trim_matches('"').to_string();
                let max_param = max.to_string().trim_matches('"').to_string();
                (
                    format!("({})::numeric BETWEEN ${}::numeric AND ${}::numeric", field, param_idx, param_idx + 1),
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
