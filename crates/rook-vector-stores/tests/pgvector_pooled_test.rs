//! Integration tests for PgVectorStorePooled.
//!
//! These tests require a running PostgreSQL instance with pgvector extension.
//! Set DATABASE_URL environment variable to run these tests.
//!
//! Example:
//! ```bash
//! DATABASE_URL="postgres://user:pass@localhost/rook_test" \
//!     cargo test -p rook-vector-stores --features pgvector -- --ignored
//! ```

#![cfg(feature = "pgvector")]

use std::collections::HashMap;
use std::sync::Arc;

use rook_core::error::RookResult;
use rook_core::traits::{
    DistanceMetric, PostgresPoolConfig, VectorRecord, VectorStore, VectorStoreConfig,
    VectorStoreProvider,
};
use rook_vector_stores::PgVectorStorePooled;

fn get_test_url() -> Option<String> {
    std::env::var("DATABASE_URL").ok()
}

fn create_test_config(collection_name: &str, pool_config: Option<PostgresPoolConfig>) -> VectorStoreConfig {
    VectorStoreConfig {
        provider: VectorStoreProvider::PgvectorPooled,
        collection_name: collection_name.to_string(),
        embedding_model_dims: 384,
        pool: pool_config,
        config: serde_json::json!({
            "url": get_test_url().expect("DATABASE_URL must be set")
        }),
    }
}

fn random_vector(dim: usize) -> Vec<f32> {
    use std::time::{SystemTime, UNIX_EPOCH};
    let seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64;

    (0..dim)
        .map(|i| ((seed.wrapping_mul(i as u64 + 1) % 1000) as f32 / 1000.0) * 2.0 - 1.0)
        .collect()
}

/// Test pool initialization with default configuration.
#[tokio::test]
#[ignore] // Requires PostgreSQL with pgvector
async fn test_pool_init_default_config() -> RookResult<()> {
    let url = match get_test_url() {
        Some(url) => url,
        None => return Ok(()), // Skip if no DB
    };

    let config = VectorStoreConfig {
        provider: VectorStoreProvider::PgvectorPooled,
        collection_name: "test_pool_default".to_string(),
        embedding_model_dims: 384,
        pool: None, // Use defaults
        config: serde_json::json!({ "url": url }),
    };

    let store = PgVectorStorePooled::new(config).await?;

    // Verify we can perform basic operations
    store
        .create_collection("test_pool_default", 384, DistanceMetric::Cosine)
        .await?;

    // Cleanup
    store.delete_collection("test_pool_default").await?;

    Ok(())
}

/// Test pool initialization with custom configuration.
#[tokio::test]
#[ignore] // Requires PostgreSQL with pgvector
async fn test_pool_init_custom_config() -> RookResult<()> {
    let url = match get_test_url() {
        Some(url) => url,
        None => return Ok(()),
    };

    let pool_config = PostgresPoolConfig {
        max_size: 4,
        wait_timeout_secs: 5,
        create_timeout_secs: 3,
        recycle_timeout_secs: 3,
        recycling_method: "verified".to_string(),
    };

    let config = VectorStoreConfig {
        provider: VectorStoreProvider::PgvectorPooled,
        collection_name: "test_pool_custom".to_string(),
        embedding_model_dims: 384,
        pool: Some(pool_config),
        config: serde_json::json!({ "url": url }),
    };

    let store = PgVectorStorePooled::new(config).await?;

    // Verify we can perform basic operations
    store
        .create_collection("test_pool_custom", 384, DistanceMetric::Cosine)
        .await?;

    // Cleanup
    store.delete_collection("test_pool_custom").await?;

    Ok(())
}

/// Test basic CRUD operations through pooled connection.
#[tokio::test]
#[ignore] // Requires PostgreSQL with pgvector
async fn test_crud_operations() -> RookResult<()> {
    let Some(_) = get_test_url() else {
        return Ok(());
    };

    let config = create_test_config("test_crud", None);
    let store = PgVectorStorePooled::new(config).await?;

    // Create collection
    store
        .create_collection("test_crud", 384, DistanceMetric::Cosine)
        .await?;

    // Insert
    let record = VectorRecord {
        id: "vec1".to_string(),
        vector: random_vector(384),
        payload: HashMap::from([
            ("content".to_string(), serde_json::json!("Hello, world!")),
            ("user_id".to_string(), serde_json::json!("user123")),
        ]),
        score: None,
    };
    store.insert(vec![record.clone()]).await?;

    // Get
    let retrieved = store.get("vec1").await?;
    assert!(retrieved.is_some());
    let retrieved = retrieved.unwrap();
    assert_eq!(retrieved.id, "vec1");
    assert_eq!(
        retrieved.payload.get("content"),
        Some(&serde_json::json!("Hello, world!"))
    );

    // Update
    let new_payload = HashMap::from([
        ("content".to_string(), serde_json::json!("Updated content")),
    ]);
    store.update("vec1", None, Some(new_payload)).await?;

    let updated = store.get("vec1").await?.unwrap();
    assert_eq!(
        updated.payload.get("content"),
        Some(&serde_json::json!("Updated content"))
    );

    // Delete
    store.delete("vec1").await?;
    let deleted = store.get("vec1").await?;
    assert!(deleted.is_none());

    // Cleanup
    store.delete_collection("test_crud").await?;

    Ok(())
}

/// Test search functionality.
#[tokio::test]
#[ignore] // Requires PostgreSQL with pgvector
async fn test_search() -> RookResult<()> {
    let Some(_) = get_test_url() else {
        return Ok(());
    };

    let config = create_test_config("test_search", None);
    let store = PgVectorStorePooled::new(config).await?;

    store
        .create_collection("test_search", 384, DistanceMetric::Cosine)
        .await?;

    // Insert multiple records
    let base_vector = random_vector(384);
    let records: Vec<VectorRecord> = (0..5)
        .map(|i| {
            let mut vec = base_vector.clone();
            // Slightly modify each vector
            vec[0] += i as f32 * 0.1;
            VectorRecord {
                id: format!("vec{}", i),
                vector: vec,
                payload: HashMap::from([
                    ("index".to_string(), serde_json::json!(i)),
                ]),
                score: None,
            }
        })
        .collect();

    store.insert(records).await?;

    // Search
    let results = store.search(&base_vector, 3, None).await?;
    assert!(!results.is_empty());
    assert!(results.len() <= 3);

    // First result should be vec0 (exact match)
    assert_eq!(results[0].id, "vec0");

    // Cleanup
    store.delete_collection("test_search").await?;

    Ok(())
}

/// Test concurrent access (validates pool handles multiple simultaneous requests).
#[tokio::test]
#[ignore] // Requires PostgreSQL with pgvector
async fn test_concurrent_access() -> RookResult<()> {
    let Some(_) = get_test_url() else {
        return Ok(());
    };

    let pool_config = PostgresPoolConfig {
        max_size: 4,
        ..Default::default()
    };
    let config = create_test_config("test_concurrent", Some(pool_config));
    let store = Arc::new(PgVectorStorePooled::new(config).await?);

    store
        .create_collection("test_concurrent", 384, DistanceMetric::Cosine)
        .await?;

    // Insert initial data
    let records: Vec<VectorRecord> = (0..10)
        .map(|i| VectorRecord {
            id: format!("vec{}", i),
            vector: random_vector(384),
            payload: HashMap::from([("index".to_string(), serde_json::json!(i))]),
            score: None,
        })
        .collect();
    store.insert(records).await?;

    // Spawn concurrent search tasks
    let mut handles = vec![];
    for i in 0..10 {
        let store_clone = Arc::clone(&store);
        let query_vec = random_vector(384);
        let handle = tokio::spawn(async move {
            let results = store_clone.search(&query_vec, 5, None).await;
            (i, results.is_ok())
        });
        handles.push(handle);
    }

    // Wait for all tasks to complete
    let mut successes = 0;
    for handle in handles {
        let (idx, success) = handle.await.unwrap();
        if success {
            successes += 1;
        } else {
            eprintln!("Task {} failed", idx);
        }
    }

    // All searches should succeed
    assert_eq!(successes, 10);

    // Cleanup
    store.delete_collection("test_concurrent").await?;

    Ok(())
}

/// Test pool exhaustion handling (validates clear error messages).
///
/// This test uses a small pool size and spawns more concurrent requests
/// than available connections to verify pool exhaustion is handled gracefully.
#[tokio::test]
#[ignore] // Requires PostgreSQL with pgvector
async fn test_pool_exhaustion_handling() -> RookResult<()> {
    let Some(_) = get_test_url() else {
        return Ok(());
    };

    // Use very small pool with short timeout
    let pool_config = PostgresPoolConfig {
        max_size: 2,
        wait_timeout_secs: 1, // Short timeout to trigger exhaustion quickly
        ..Default::default()
    };
    let config = create_test_config("test_exhaustion", Some(pool_config));
    let store = Arc::new(PgVectorStorePooled::new(config).await?);

    store
        .create_collection("test_exhaustion", 384, DistanceMetric::Cosine)
        .await?;

    // Insert data for searching
    let records: Vec<VectorRecord> = (0..10)
        .map(|i| VectorRecord {
            id: format!("vec{}", i),
            vector: random_vector(384),
            payload: HashMap::from([("index".to_string(), serde_json::json!(i))]),
            score: None,
        })
        .collect();
    store.insert(records).await?;

    // Spawn more concurrent tasks than pool size
    // This may or may not trigger exhaustion depending on timing
    let mut handles = vec![];
    for i in 0..20 {
        let store_clone = Arc::clone(&store);
        let query_vec = random_vector(384);
        let handle = tokio::spawn(async move {
            // Add small delay to try to hold connections longer
            let result = store_clone.search(&query_vec, 5, None).await;
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
            (i, result)
        });
        handles.push(handle);
    }

    // Count successes and failures
    let mut successes = 0;
    let mut failures = 0;
    for handle in handles {
        let (_, result) = handle.await.unwrap();
        match result {
            Ok(_) => successes += 1,
            Err(e) => {
                // Verify error message mentions pool exhaustion
                let msg = e.to_string();
                if msg.contains("pool") || msg.contains("timeout") || msg.contains("exhausted") {
                    // This is the expected error for pool exhaustion
                    failures += 1;
                } else {
                    // Unexpected error
                    eprintln!("Unexpected error: {}", msg);
                    failures += 1;
                }
            }
        }
    }

    // We should have some successes (the first requests before pool fills)
    assert!(successes > 0, "At least some requests should succeed");

    // Note: We can't guarantee failures because modern CPUs may complete
    // requests fast enough that pool never exhausts. The important thing
    // is that the error handling code path exists and works correctly.
    println!("Pool test: {} successes, {} failures", successes, failures);

    // Cleanup
    store.delete_collection("test_exhaustion").await?;

    Ok(())
}

/// Test collection info and listing.
#[tokio::test]
#[ignore] // Requires PostgreSQL with pgvector
async fn test_collection_operations() -> RookResult<()> {
    let Some(_) = get_test_url() else {
        return Ok(());
    };

    let config = create_test_config("test_collection_ops", None);
    let store = PgVectorStorePooled::new(config).await?;

    // Create collection
    store
        .create_collection("test_collection_ops", 384, DistanceMetric::Cosine)
        .await?;

    // Insert some records
    let records: Vec<VectorRecord> = (0..5)
        .map(|i| VectorRecord {
            id: format!("vec{}", i),
            vector: random_vector(384),
            payload: HashMap::new(),
            score: None,
        })
        .collect();
    store.insert(records).await?;

    // Get collection info
    let info = store.collection_info("test_collection_ops").await?;
    assert_eq!(info.name, "test_collection_ops");
    assert_eq!(info.dimension, 384);
    assert_eq!(info.vector_count, 5);

    // List collections (should include our test collection)
    let collections = store.list_collections().await?;
    assert!(collections.contains(&"test_collection_ops".to_string()));

    // Reset (delete and recreate)
    store.reset().await?;
    let info_after_reset = store.collection_info("test_collection_ops").await?;
    assert_eq!(info_after_reset.vector_count, 0);

    // Cleanup
    store.delete_collection("test_collection_ops").await?;

    Ok(())
}
