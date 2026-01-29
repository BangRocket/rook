//! SQLite vector store implementation using sqlite-vec extension.
//!
//! This module provides a high-performance embedded vector store using
//! SQLite with the sqlite-vec extension for native ANN (Approximate Nearest Neighbor) search.
//!
//! # Features
//!
//! - Native vector similarity search using vec0 virtual tables
//! - Zero-copy vector serialization with zerocopy
//! - Support for L2 distance (Euclidean)
//! - Metadata storage as JSON
//!
//! # Example
//!
//! ```ignore
//! use rook_vector_stores::SqliteVecStore;
//!
//! let store = SqliteVecStore::new(":memory:", "embeddings", 1536)?;
//! ```

use std::collections::HashMap;
use std::sync::Mutex;

use async_trait::async_trait;
use rusqlite::Connection;
use serde_json::Value;
use zerocopy::IntoBytes;

use rook_core::error::{RookError, RookResult};
use rook_core::traits::{
    CollectionInfo, DistanceMetric, VectorRecord, VectorSearchResult, VectorStore,
};
use rook_core::types::Filter;

/// SQLite vector store using sqlite-vec extension.
///
/// This store uses SQLite's vec0 virtual table for efficient vector similarity search.
/// It is optimized for embedded use cases and provides native ANN search capabilities.
pub struct SqliteVecStore {
    /// SQLite connection (wrapped in Mutex for Send + Sync).
    conn: Mutex<Connection>,
    /// Collection name (table name).
    collection_name: String,
    /// Vector dimension.
    dimension: usize,
}

impl SqliteVecStore {
    /// Create a new SqliteVecStore.
    ///
    /// # Arguments
    ///
    /// * `db_path` - Path to SQLite database file (use ":memory:" for in-memory)
    /// * `collection_name` - Name of the collection (will be used as table name)
    /// * `dimension` - Dimension of vectors to store
    ///
    /// # Example
    ///
    /// ```ignore
    /// let store = SqliteVecStore::new(":memory:", "embeddings", 1536)?;
    /// ```
    pub fn new(db_path: &str, collection_name: &str, dimension: usize) -> RookResult<Self> {
        // Register sqlite-vec extension before opening connection.
        // SAFETY: sqlite3_auto_extension requires a function pointer cast.
        // This is the documented way to register sqlite-vec with rusqlite.
        unsafe {
            rusqlite::ffi::sqlite3_auto_extension(Some(std::mem::transmute(
                sqlite_vec::sqlite3_vec_init as *const (),
            )));
        }

        let conn = Connection::open(db_path).map_err(|e| RookError::VectorStore {
            message: format!("Failed to open SQLite database: {}", e),
            code: rook_core::error::ErrorCode::VecConnectionFailed,
            source: Some(Box::new(e)),
        })?;

        // Verify sqlite-vec is loaded.
        let version: String = conn
            .query_row("SELECT vec_version()", [], |row| row.get(0))
            .map_err(|e| RookError::VectorStore {
                message: format!("sqlite-vec extension not loaded: {}", e),
                code: rook_core::error::ErrorCode::VecConnectionFailed,
                source: Some(Box::new(e)),
            })?;

        tracing::debug!("sqlite-vec version: {}", version);

        Ok(Self {
            conn: Mutex::new(conn),
            collection_name: collection_name.to_string(),
            dimension,
        })
    }

    /// Create the vec0 virtual table for this collection.
    fn create_table(&self, conn: &Connection) -> RookResult<()> {
        // Create vec0 virtual table with embedding column and metadata.
        // The + prefix on columns makes them auxiliary (stored but not indexed).
        let sql = format!(
            r#"CREATE VIRTUAL TABLE IF NOT EXISTS "{}" USING vec0(
                embedding float[{}],
                +id TEXT PRIMARY KEY,
                +payload TEXT
            )"#,
            self.collection_name, self.dimension
        );

        conn.execute(&sql, []).map_err(|e| RookError::VectorStore {
            message: format!("Failed to create vec0 table: {}", e),
            code: rook_core::error::ErrorCode::VecOperationFailed,
            source: Some(Box::new(e)),
        })?;

        Ok(())
    }

    /// Convert Vec<f32> to bytes for sqlite-vec.
    fn vector_to_bytes(vector: &[f32]) -> Vec<u8> {
        vector.as_bytes().to_vec()
    }

    /// Convert bytes back to Vec<f32>.
    fn bytes_to_vector(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect()
    }

    /// Apply filter to records in memory (post-filter).
    /// sqlite-vec doesn't support native filtering, so we filter after retrieval.
    fn apply_filter(records: Vec<VectorSearchResult>, filter: &Filter) -> Vec<VectorSearchResult> {
        records
            .into_iter()
            .filter(|r| Self::matches_filter(&r.payload, filter))
            .collect()
    }

    /// Check if a payload matches a filter.
    fn matches_filter(payload: &HashMap<String, Value>, filter: &Filter) -> bool {
        match filter {
            Filter::Condition(cond) => {
                let field_value = payload.get(&cond.field);
                match &cond.operator {
                    rook_core::types::FilterOperator::Eq(v) => field_value == Some(v),
                    rook_core::types::FilterOperator::Ne(v) => field_value != Some(v),
                    rook_core::types::FilterOperator::In(values) => {
                        field_value.map_or(false, |fv| values.contains(fv))
                    }
                    rook_core::types::FilterOperator::Nin(values) => {
                        field_value.map_or(true, |fv| !values.contains(fv))
                    }
                    rook_core::types::FilterOperator::Contains(s) => field_value
                        .and_then(|v| v.as_str())
                        .map_or(false, |fv| fv.contains(s)),
                    rook_core::types::FilterOperator::Icontains(s) => field_value
                        .and_then(|v| v.as_str())
                        .map_or(false, |fv| fv.to_lowercase().contains(&s.to_lowercase())),
                    rook_core::types::FilterOperator::Gt(v) => {
                        Self::compare_values(field_value, v, |a, b| a > b)
                    }
                    rook_core::types::FilterOperator::Gte(v) => {
                        Self::compare_values(field_value, v, |a, b| a >= b)
                    }
                    rook_core::types::FilterOperator::Lt(v) => {
                        Self::compare_values(field_value, v, |a, b| a < b)
                    }
                    rook_core::types::FilterOperator::Lte(v) => {
                        Self::compare_values(field_value, v, |a, b| a <= b)
                    }
                    rook_core::types::FilterOperator::Between { min, max } => {
                        Self::compare_values(field_value, min, |a, b| a >= b)
                            && Self::compare_values(field_value, max, |a, b| a <= b)
                    }
                    rook_core::types::FilterOperator::IsNull => field_value.is_none(),
                    rook_core::types::FilterOperator::IsNotNull => field_value.is_some(),
                    rook_core::types::FilterOperator::Exists => payload.contains_key(&cond.field),
                    rook_core::types::FilterOperator::NotExists => {
                        !payload.contains_key(&cond.field)
                    }
                    rook_core::types::FilterOperator::Wildcard => true,
                }
            }
            Filter::And(filters) => filters.iter().all(|f| Self::matches_filter(payload, f)),
            Filter::Or(filters) => filters.iter().any(|f| Self::matches_filter(payload, f)),
            Filter::Not(filter) => !Self::matches_filter(payload, filter),
        }
    }

    /// Compare two JSON values using a comparison function.
    fn compare_values<F>(field_value: Option<&Value>, compare_to: &Value, cmp: F) -> bool
    where
        F: Fn(f64, f64) -> bool,
    {
        match (field_value, compare_to) {
            (Some(Value::Number(a)), Value::Number(b)) => {
                match (a.as_f64(), b.as_f64()) {
                    (Some(av), Some(bv)) => cmp(av, bv),
                    _ => false,
                }
            }
            _ => false,
        }
    }
}

#[async_trait]
impl VectorStore for SqliteVecStore {
    async fn create_collection(
        &self,
        name: &str,
        dimension: usize,
        _distance: DistanceMetric,
    ) -> RookResult<()> {
        let conn = self.conn.lock().map_err(|e| {
            RookError::vector_store(format!("Failed to acquire lock: {}", e))
        })?;

        // Create vec0 virtual table.
        // Note: sqlite-vec uses L2 distance by default, we ignore the distance parameter.
        let sql = format!(
            r#"CREATE VIRTUAL TABLE IF NOT EXISTS "{}" USING vec0(
                embedding float[{}],
                +id TEXT PRIMARY KEY,
                +payload TEXT
            )"#,
            name, dimension
        );

        conn.execute(&sql, []).map_err(|e| RookError::VectorStore {
            message: format!("Failed to create collection '{}': {}", name, e),
            code: rook_core::error::ErrorCode::VecOperationFailed,
            source: Some(Box::new(e)),
        })?;

        tracing::info!("Created collection '{}' with dimension {}", name, dimension);
        Ok(())
    }

    async fn insert(&self, records: Vec<VectorRecord>) -> RookResult<()> {
        let conn = self.conn.lock().map_err(|e| {
            RookError::vector_store(format!("Failed to acquire lock: {}", e))
        })?;

        // Ensure table exists.
        self.create_table(&conn)?;

        // Insert each record.
        let sql = format!(
            r#"INSERT OR REPLACE INTO "{}" (embedding, id, payload) VALUES (?, ?, ?)"#,
            self.collection_name
        );

        let mut stmt = conn.prepare(&sql).map_err(|e| RookError::VectorStore {
            message: format!("Failed to prepare insert statement: {}", e),
            code: rook_core::error::ErrorCode::VecOperationFailed,
            source: Some(Box::new(e)),
        })?;

        for record in records {
            let embedding_bytes = Self::vector_to_bytes(&record.vector);
            let payload_json = serde_json::to_string(&record.payload).map_err(|e| {
                RookError::VectorStore {
                    message: format!("Failed to serialize payload: {}", e),
                    code: rook_core::error::ErrorCode::VecOperationFailed,
                    source: Some(Box::new(e)),
                }
            })?;

            stmt.execute(rusqlite::params![embedding_bytes, record.id, payload_json])
                .map_err(|e| RookError::VectorStore {
                    message: format!("Failed to insert record '{}': {}", record.id, e),
                    code: rook_core::error::ErrorCode::VecOperationFailed,
                    source: Some(Box::new(e)),
                })?;
        }

        tracing::debug!("Inserted records into collection '{}'", self.collection_name);
        Ok(())
    }

    async fn search(
        &self,
        query_vector: &[f32],
        limit: usize,
        filters: Option<Filter>,
    ) -> RookResult<Vec<VectorSearchResult>> {
        let conn = self.conn.lock().map_err(|e| {
            RookError::vector_store(format!("Failed to acquire lock: {}", e))
        })?;

        // Use MATCH operator for KNN search.
        // We fetch more results if filtering is needed.
        let fetch_limit = if filters.is_some() { limit * 10 } else { limit };

        let sql = format!(
            r#"SELECT id, distance, payload
               FROM "{}"
               WHERE embedding MATCH ?
               ORDER BY distance
               LIMIT ?"#,
            self.collection_name
        );

        let query_bytes = Self::vector_to_bytes(query_vector);
        let mut stmt = conn.prepare(&sql).map_err(|e| RookError::VectorStore {
            message: format!("Failed to prepare search statement: {}", e),
            code: rook_core::error::ErrorCode::VecOperationFailed,
            source: Some(Box::new(e)),
        })?;

        let rows = stmt
            .query_map(rusqlite::params![query_bytes, fetch_limit], |row| {
                let id: String = row.get(0)?;
                let distance: f32 = row.get(1)?;
                let payload_str: String = row.get(2)?;
                Ok((id, distance, payload_str))
            })
            .map_err(|e| RookError::VectorStore {
                message: format!("Failed to execute search: {}", e),
                code: rook_core::error::ErrorCode::VecOperationFailed,
                source: Some(Box::new(e)),
            })?;

        let mut results = Vec::new();
        for row in rows {
            let (id, distance, payload_str) = row.map_err(|e| RookError::VectorStore {
                message: format!("Failed to read search result: {}", e),
                code: rook_core::error::ErrorCode::VecOperationFailed,
                source: Some(Box::new(e)),
            })?;

            let payload: HashMap<String, Value> =
                serde_json::from_str(&payload_str).unwrap_or_default();

            // Convert distance to similarity score.
            // For L2 distance, we use 1 / (1 + distance) to normalize to [0, 1].
            let score = 1.0 / (1.0 + distance);

            results.push(VectorSearchResult { id, score, payload });
        }

        // Apply post-filter if provided.
        let results = if let Some(filter) = filters {
            let filtered = Self::apply_filter(results, &filter);
            filtered.into_iter().take(limit).collect()
        } else {
            results
        };

        tracing::debug!(
            "Search returned {} results from collection '{}'",
            results.len(),
            self.collection_name
        );

        Ok(results)
    }

    async fn get(&self, id: &str) -> RookResult<Option<VectorRecord>> {
        let conn = self.conn.lock().map_err(|e| {
            RookError::vector_store(format!("Failed to acquire lock: {}", e))
        })?;

        let sql = format!(
            r#"SELECT embedding, id, payload FROM "{}" WHERE id = ?"#,
            self.collection_name
        );

        let result = conn.query_row(&sql, [id], |row| {
            let embedding_bytes: Vec<u8> = row.get(0)?;
            let id: String = row.get(1)?;
            let payload_str: String = row.get(2)?;
            Ok((embedding_bytes, id, payload_str))
        });

        match result {
            Ok((embedding_bytes, id, payload_str)) => {
                let vector = Self::bytes_to_vector(&embedding_bytes);
                let payload: HashMap<String, Value> =
                    serde_json::from_str(&payload_str).unwrap_or_default();

                Ok(Some(VectorRecord {
                    id,
                    vector,
                    payload,
                    score: None,
                }))
            }
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(RookError::VectorStore {
                message: format!("Failed to get record '{}': {}", id, e),
                code: rook_core::error::ErrorCode::VecOperationFailed,
                source: Some(Box::new(e)),
            }),
        }
    }

    async fn update(
        &self,
        id: &str,
        vector: Option<Vec<f32>>,
        payload: Option<HashMap<String, Value>>,
    ) -> RookResult<()> {
        // Get existing record.
        let existing = self.get(id).await?;
        let existing = existing.ok_or_else(|| RookError::not_found(id))?;

        // Merge updates.
        let new_vector = vector.unwrap_or(existing.vector);
        let new_payload = match payload {
            Some(p) => {
                let mut merged = existing.payload;
                merged.extend(p);
                merged
            }
            None => existing.payload,
        };

        // Delete old record first (vec0 doesn't support INSERT OR REPLACE well).
        self.delete(id).await?;

        // Insert updated record.
        let record = VectorRecord {
            id: id.to_string(),
            vector: new_vector,
            payload: new_payload,
            score: None,
        };

        self.insert(vec![record]).await
    }

    async fn delete(&self, id: &str) -> RookResult<()> {
        let conn = self.conn.lock().map_err(|e| {
            RookError::vector_store(format!("Failed to acquire lock: {}", e))
        })?;

        let sql = format!(r#"DELETE FROM "{}" WHERE id = ?"#, self.collection_name);

        conn.execute(&sql, [id]).map_err(|e| RookError::VectorStore {
            message: format!("Failed to delete record '{}': {}", id, e),
            code: rook_core::error::ErrorCode::VecOperationFailed,
            source: Some(Box::new(e)),
        })?;

        tracing::debug!("Deleted record '{}' from collection '{}'", id, self.collection_name);
        Ok(())
    }

    async fn list(
        &self,
        filters: Option<Filter>,
        limit: Option<usize>,
    ) -> RookResult<Vec<VectorRecord>> {
        let conn = self.conn.lock().map_err(|e| {
            RookError::vector_store(format!("Failed to acquire lock: {}", e))
        })?;

        // Fetch more if filtering is needed.
        let fetch_limit = match (filters.as_ref(), limit) {
            (Some(_), Some(l)) => l * 10,
            (None, Some(l)) => l,
            (_, None) => 10000, // Reasonable max
        };

        let sql = format!(
            r#"SELECT embedding, id, payload FROM "{}" LIMIT ?"#,
            self.collection_name
        );

        let mut stmt = conn.prepare(&sql).map_err(|e| RookError::VectorStore {
            message: format!("Failed to prepare list statement: {}", e),
            code: rook_core::error::ErrorCode::VecOperationFailed,
            source: Some(Box::new(e)),
        })?;

        let rows = stmt
            .query_map([fetch_limit], |row| {
                let embedding_bytes: Vec<u8> = row.get(0)?;
                let id: String = row.get(1)?;
                let payload_str: String = row.get(2)?;
                Ok((embedding_bytes, id, payload_str))
            })
            .map_err(|e| RookError::VectorStore {
                message: format!("Failed to execute list: {}", e),
                code: rook_core::error::ErrorCode::VecOperationFailed,
                source: Some(Box::new(e)),
            })?;

        let mut records = Vec::new();
        for row in rows {
            let (embedding_bytes, id, payload_str) = row.map_err(|e| RookError::VectorStore {
                message: format!("Failed to read list result: {}", e),
                code: rook_core::error::ErrorCode::VecOperationFailed,
                source: Some(Box::new(e)),
            })?;

            let vector = Self::bytes_to_vector(&embedding_bytes);
            let payload: HashMap<String, Value> =
                serde_json::from_str(&payload_str).unwrap_or_default();

            records.push(VectorRecord {
                id,
                vector,
                payload,
                score: None,
            });
        }

        // Apply post-filter if provided.
        let records = if let Some(filter) = filters {
            records
                .into_iter()
                .filter(|r| Self::matches_filter(&r.payload, &filter))
                .collect::<Vec<_>>()
        } else {
            records
        };

        // Apply limit.
        let records = match limit {
            Some(l) => records.into_iter().take(l).collect(),
            None => records,
        };

        Ok(records)
    }

    async fn list_collections(&self) -> RookResult<Vec<String>> {
        let conn = self.conn.lock().map_err(|e| {
            RookError::vector_store(format!("Failed to acquire lock: {}", e))
        })?;

        // Query for vec0 virtual tables.
        let sql = r#"SELECT name FROM sqlite_master WHERE type='table' AND sql LIKE '%vec0%'"#;

        let mut stmt = conn.prepare(sql).map_err(|e| RookError::VectorStore {
            message: format!("Failed to list collections: {}", e),
            code: rook_core::error::ErrorCode::VecOperationFailed,
            source: Some(Box::new(e)),
        })?;

        let rows = stmt
            .query_map([], |row| row.get(0))
            .map_err(|e| RookError::VectorStore {
                message: format!("Failed to execute list collections: {}", e),
                code: rook_core::error::ErrorCode::VecOperationFailed,
                source: Some(Box::new(e)),
            })?;

        let mut collections = Vec::new();
        for row in rows {
            collections.push(row.map_err(|e| RookError::VectorStore {
                message: format!("Failed to read collection name: {}", e),
                code: rook_core::error::ErrorCode::VecOperationFailed,
                source: Some(Box::new(e)),
            })?);
        }

        Ok(collections)
    }

    async fn delete_collection(&self, name: &str) -> RookResult<()> {
        let conn = self.conn.lock().map_err(|e| {
            RookError::vector_store(format!("Failed to acquire lock: {}", e))
        })?;

        let sql = format!(r#"DROP TABLE IF EXISTS "{}""#, name);

        conn.execute(&sql, []).map_err(|e| RookError::VectorStore {
            message: format!("Failed to delete collection '{}': {}", name, e),
            code: rook_core::error::ErrorCode::VecOperationFailed,
            source: Some(Box::new(e)),
        })?;

        tracing::info!("Deleted collection '{}'", name);
        Ok(())
    }

    async fn collection_info(&self, name: &str) -> RookResult<CollectionInfo> {
        let conn = self.conn.lock().map_err(|e| {
            RookError::vector_store(format!("Failed to acquire lock: {}", e))
        })?;

        // Count vectors.
        let count_sql = format!(r#"SELECT COUNT(*) FROM "{}""#, name);
        let count: u64 = conn
            .query_row(&count_sql, [], |row| row.get(0))
            .map_err(|e| RookError::VectorStore {
                message: format!("Failed to get collection info for '{}': {}", name, e),
                code: rook_core::error::ErrorCode::VecCollectionNotFound,
                source: Some(Box::new(e)),
            })?;

        Ok(CollectionInfo {
            name: name.to_string(),
            vector_count: count,
            dimension: self.dimension,
            distance: DistanceMetric::Euclidean, // sqlite-vec uses L2 distance
        })
    }

    async fn reset(&self) -> RookResult<()> {
        // Delete and recreate the collection.
        self.delete_collection(&self.collection_name.clone()).await?;

        let conn = self.conn.lock().map_err(|e| {
            RookError::vector_store(format!("Failed to acquire lock: {}", e))
        })?;

        self.create_table(&conn)?;

        tracing::info!("Reset collection '{}'", self.collection_name);
        Ok(())
    }

    fn collection_name(&self) -> &str {
        &self.collection_name
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_store() -> SqliteVecStore {
        SqliteVecStore::new(":memory:", "test_collection", 4).unwrap()
    }

    fn create_test_vector(id: &str, values: [f32; 4]) -> VectorRecord {
        let mut payload = HashMap::new();
        payload.insert("data".to_string(), Value::String(format!("data for {}", id)));
        payload.insert("category".to_string(), Value::String("test".to_string()));

        VectorRecord {
            id: id.to_string(),
            vector: values.to_vec(),
            payload,
            score: None,
        }
    }

    #[tokio::test]
    async fn test_create_collection() {
        let store = create_test_store();

        // Collection should be created automatically on first insert.
        let record = create_test_vector("1", [1.0, 0.0, 0.0, 0.0]);
        store.insert(vec![record]).await.unwrap();

        let info = store.collection_info("test_collection").await.unwrap();
        assert_eq!(info.name, "test_collection");
        assert_eq!(info.vector_count, 1);
        assert_eq!(info.dimension, 4);
    }

    #[tokio::test]
    async fn test_insert_and_search() {
        let store = create_test_store();

        // Insert 3 vectors.
        let records = vec![
            create_test_vector("1", [1.0, 0.0, 0.0, 0.0]),
            create_test_vector("2", [0.9, 0.1, 0.0, 0.0]),
            create_test_vector("3", [0.0, 1.0, 0.0, 0.0]),
        ];
        store.insert(records).await.unwrap();

        // Search for nearest to [1.0, 0.0, 0.0, 0.0].
        let query = vec![1.0, 0.0, 0.0, 0.0];
        let results = store.search(&query, 2, None).await.unwrap();

        assert_eq!(results.len(), 2);
        // First result should be exact match (id "1").
        assert_eq!(results[0].id, "1");
        // Second should be "2" (closest to query).
        assert_eq!(results[1].id, "2");
        // Scores should be in descending order (higher = more similar).
        assert!(results[0].score >= results[1].score);
    }

    #[tokio::test]
    async fn test_get_and_delete() {
        let store = create_test_store();

        // Insert a record.
        let record = create_test_vector("test-id", [1.0, 2.0, 3.0, 4.0]);
        store.insert(vec![record]).await.unwrap();

        // Get by ID.
        let retrieved = store.get("test-id").await.unwrap();
        assert!(retrieved.is_some());
        let retrieved = retrieved.unwrap();
        assert_eq!(retrieved.id, "test-id");
        assert_eq!(retrieved.vector, vec![1.0, 2.0, 3.0, 4.0]);

        // Delete.
        store.delete("test-id").await.unwrap();

        // Verify gone.
        let retrieved = store.get("test-id").await.unwrap();
        assert!(retrieved.is_none());
    }

    #[tokio::test]
    async fn test_update() {
        let store = create_test_store();

        // Insert a record.
        let mut payload = HashMap::new();
        payload.insert("key1".to_string(), Value::String("value1".to_string()));

        let record = VectorRecord {
            id: "update-test".to_string(),
            vector: vec![1.0, 0.0, 0.0, 0.0],
            payload,
            score: None,
        };
        store.insert(vec![record]).await.unwrap();

        // Update with new payload.
        let mut new_payload = HashMap::new();
        new_payload.insert("key2".to_string(), Value::String("value2".to_string()));

        store
            .update("update-test", None, Some(new_payload))
            .await
            .unwrap();

        // Verify update.
        let retrieved = store.get("update-test").await.unwrap().unwrap();
        assert_eq!(
            retrieved.payload.get("key1").and_then(|v| v.as_str()),
            Some("value1")
        );
        assert_eq!(
            retrieved.payload.get("key2").and_then(|v| v.as_str()),
            Some("value2")
        );
    }

    #[tokio::test]
    async fn test_list_with_filter() {
        let store = create_test_store();

        // Insert records with different categories.
        let mut records = Vec::new();
        for i in 0..5 {
            let mut payload = HashMap::new();
            payload.insert(
                "category".to_string(),
                Value::String(if i % 2 == 0 { "even" } else { "odd" }.to_string()),
            );
            records.push(VectorRecord {
                id: format!("record-{}", i),
                vector: vec![i as f32, 0.0, 0.0, 0.0],
                payload,
                score: None,
            });
        }
        store.insert(records).await.unwrap();

        // Filter by category "even".
        let filter = Filter::eq("category", "even");
        let results = store.list(Some(filter), None).await.unwrap();

        assert_eq!(results.len(), 3); // 0, 2, 4 are even indices.
        for result in results {
            assert_eq!(
                result.payload.get("category").and_then(|v| v.as_str()),
                Some("even")
            );
        }
    }

    #[tokio::test]
    async fn test_reset() {
        let store = create_test_store();

        // Insert some records.
        let records = vec![
            create_test_vector("1", [1.0, 0.0, 0.0, 0.0]),
            create_test_vector("2", [0.0, 1.0, 0.0, 0.0]),
        ];
        store.insert(records).await.unwrap();

        // Verify records exist.
        let info = store.collection_info("test_collection").await.unwrap();
        assert_eq!(info.vector_count, 2);

        // Reset.
        store.reset().await.unwrap();

        // Verify empty (note: reset recreates table, count may fail if table doesn't exist yet).
        // Insert one record to ensure table exists.
        store
            .insert(vec![create_test_vector("new", [1.0, 1.0, 1.0, 1.0])])
            .await
            .unwrap();
        let info = store.collection_info("test_collection").await.unwrap();
        assert_eq!(info.vector_count, 1);
    }
}
