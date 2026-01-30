//! Migration from mem0 to Rook.
//!
//! Handles the differences between mem0 and Rook data formats,
//! converting mem0 exports to Rook-compatible memories.
//!
//! # mem0 Format
//!
//! mem0 exports memories in JSON Lines format with fields:
//! - `id`: Unique identifier
//! - `memory`: The memory text content
//! - `hash`: Optional hash of the content
//! - `metadata`: Optional key-value metadata
//! - `created_at`, `updated_at`: Timestamps
//! - `user_id`, `agent_id`: Scoping identifiers
//! - `categories`: Optional category tags
//!
//! # Conversion
//!
//! When converting to Rook format:
//! - `user_id`, `agent_id`, and `categories` are preserved in metadata
//! - All other fields map directly to `MemoryItem`

use std::collections::HashMap;
use std::future::Future;

use serde::Deserialize;
use tokio::io::{AsyncBufRead, AsyncBufReadExt};

use crate::error::RookResult;
use crate::types::MemoryItem;

/// mem0 memory format (v1.1 API).
///
/// This struct represents the memory format exported from mem0.
/// It captures the core memory data plus mem0-specific fields
/// that are preserved in Rook metadata during migration.
#[derive(Debug, Deserialize)]
pub struct Mem0Memory {
    /// Unique identifier for the memory.
    pub id: String,

    /// The memory text content.
    pub memory: String,

    /// Optional hash of the content.
    #[serde(default)]
    pub hash: Option<String>,

    /// Custom metadata as key-value pairs.
    #[serde(default)]
    pub metadata: Option<HashMap<String, serde_json::Value>>,

    /// Creation timestamp (ISO 8601).
    #[serde(default)]
    pub created_at: Option<String>,

    /// Last update timestamp (ISO 8601).
    #[serde(default)]
    pub updated_at: Option<String>,

    // mem0-specific fields (preserved in Rook metadata)

    /// User ID for scoping the memory.
    #[serde(default)]
    pub user_id: Option<String>,

    /// Agent ID for scoping the memory.
    #[serde(default)]
    pub agent_id: Option<String>,

    /// Category tags for the memory.
    #[serde(default)]
    pub categories: Option<Vec<String>>,
}

/// Statistics from a migration operation.
#[derive(Debug, Default, Clone)]
pub struct MigrationStats {
    /// Total lines/entries processed.
    pub total: u64,

    /// Successfully migrated memories.
    pub migrated: u64,

    /// Skipped entries (e.g., duplicates, filtered).
    pub skipped: u64,

    /// Error messages for failed migrations.
    pub errors: Vec<String>,
}

impl MigrationStats {
    /// Create new empty stats.
    pub fn new() -> Self {
        Self::default()
    }

    /// Check if migration completed without errors.
    pub fn is_success(&self) -> bool {
        self.errors.is_empty()
    }

    /// Get the success rate as a percentage.
    pub fn success_rate(&self) -> f64 {
        if self.total == 0 {
            100.0
        } else {
            (self.migrated as f64 / self.total as f64) * 100.0
        }
    }

    /// Get the error rate as a percentage.
    pub fn error_rate(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            (self.errors.len() as f64 / self.total as f64) * 100.0
        }
    }
}

/// Convert mem0 memory to Rook memory item.
///
/// Preserves mem0-specific fields (user_id, agent_id, categories) in metadata
/// for future reference and filtering.
fn convert_mem0_to_rook(mem0: Mem0Memory) -> MemoryItem {
    let mut metadata = mem0.metadata.unwrap_or_default();

    // Preserve mem0 user_id and agent_id in metadata
    if let Some(user_id) = mem0.user_id {
        metadata.insert("user_id".to_string(), serde_json::json!(user_id));
    }
    if let Some(agent_id) = mem0.agent_id {
        metadata.insert("agent_id".to_string(), serde_json::json!(agent_id));
    }
    if let Some(categories) = mem0.categories {
        metadata.insert("categories".to_string(), serde_json::json!(categories));
    }

    // Mark as migrated from mem0
    metadata.insert(
        "migrated_from".to_string(),
        serde_json::json!("mem0"),
    );

    MemoryItem {
        id: mem0.id,
        memory: mem0.memory,
        hash: mem0.hash,
        score: None,
        metadata: if metadata.is_empty() {
            None
        } else {
            Some(metadata)
        },
        created_at: mem0.created_at,
        updated_at: mem0.updated_at,
        category: None, // Will be classified during import
        is_key: false,
        memory_state: None,
        dual_strength: None,
    }
}

/// Migrate memories from mem0 JSON Lines export file.
///
/// Reads mem0 export data line-by-line, converts to Rook format,
/// and processes in batches using the provided import function.
///
/// # Arguments
///
/// * `reader` - AsyncBufRead source (file, stdin, network, etc.)
/// * `batch_size` - Number of memories to process per batch (default: 100)
/// * `import_fn` - Async function to import a batch of MemoryItems, returns count imported
///
/// # Returns
///
/// Migration statistics including total processed, migrated count, and any errors.
///
/// # Example
///
/// ```rust,ignore
/// use tokio::fs::File;
/// use tokio::io::BufReader;
/// use rook_core::migration::migrate_from_mem0;
///
/// async fn migrate() -> rook_core::RookResult<()> {
///     let file = File::open("mem0_export.jsonl").await?;
///     let reader = BufReader::new(file);
///
///     let stats = migrate_from_mem0(reader, 100, |batch| async {
///         // Import batch into Rook storage
///         // Returns number successfully imported
///         memory.import_batch(batch).await
///     }).await?;
///
///     println!("Migrated {}/{} memories", stats.migrated, stats.total);
///     if !stats.errors.is_empty() {
///         println!("Errors: {:?}", stats.errors);
///     }
///     Ok(())
/// }
/// ```
pub async fn migrate_from_mem0<R, F, Fut>(
    reader: R,
    batch_size: usize,
    mut import_fn: F,
) -> RookResult<MigrationStats>
where
    R: AsyncBufRead + Unpin,
    F: FnMut(Vec<MemoryItem>) -> Fut,
    Fut: Future<Output = RookResult<usize>>,
{
    let mut lines = reader.lines();
    let mut batch: Vec<MemoryItem> = Vec::with_capacity(batch_size);
    let mut stats = MigrationStats::default();

    while let Some(line) = lines.next_line().await? {
        // Skip empty lines
        if line.trim().is_empty() {
            continue;
        }

        stats.total += 1;

        match serde_json::from_str::<Mem0Memory>(&line) {
            Ok(mem0_item) => {
                let rook_item = convert_mem0_to_rook(mem0_item);
                batch.push(rook_item);

                // Process batch when full
                if batch.len() >= batch_size {
                    match import_fn(std::mem::take(&mut batch)).await {
                        Ok(count) => stats.migrated += count as u64,
                        Err(e) => stats.errors.push(format!("Batch error: {}", e)),
                    }
                    batch = Vec::with_capacity(batch_size);
                }
            }
            Err(e) => {
                stats.errors.push(format!("Line {}: Parse error: {}", stats.total, e));
                stats.skipped += 1;
            }
        }
    }

    // Process remaining batch
    if !batch.is_empty() {
        match import_fn(batch).await {
            Ok(count) => stats.migrated += count as u64,
            Err(e) => stats.errors.push(format!("Final batch error: {}", e)),
        }
    }

    Ok(stats)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    use std::sync::{Arc, Mutex};
    use tokio::io::BufReader;

    #[test]
    fn test_convert_mem0_to_rook_basic() {
        let mem0 = Mem0Memory {
            id: "test-id".to_string(),
            memory: "Test memory content".to_string(),
            hash: Some("abc123".to_string()),
            metadata: Some([("key".to_string(), serde_json::json!("value"))].into()),
            created_at: Some("2024-01-01T00:00:00Z".to_string()),
            updated_at: None,
            user_id: Some("user123".to_string()),
            agent_id: None,
            categories: Some(vec!["preference".to_string()]),
        };

        let rook = convert_mem0_to_rook(mem0);

        assert_eq!(rook.id, "test-id");
        assert_eq!(rook.memory, "Test memory content");
        assert_eq!(rook.hash, Some("abc123".to_string()));

        let metadata = rook.metadata.unwrap();
        assert_eq!(
            metadata.get("user_id"),
            Some(&serde_json::json!("user123"))
        );
        assert_eq!(
            metadata.get("categories"),
            Some(&serde_json::json!(["preference"]))
        );
        assert_eq!(
            metadata.get("migrated_from"),
            Some(&serde_json::json!("mem0"))
        );
    }

    #[test]
    fn test_convert_mem0_to_rook_minimal() {
        let mem0 = Mem0Memory {
            id: "id1".to_string(),
            memory: "Simple memory".to_string(),
            hash: None,
            metadata: None,
            created_at: None,
            updated_at: None,
            user_id: None,
            agent_id: None,
            categories: None,
        };

        let rook = convert_mem0_to_rook(mem0);

        assert_eq!(rook.id, "id1");
        assert_eq!(rook.memory, "Simple memory");

        // Should still have migrated_from marker
        let metadata = rook.metadata.unwrap();
        assert_eq!(
            metadata.get("migrated_from"),
            Some(&serde_json::json!("mem0"))
        );
    }

    #[tokio::test]
    async fn test_migrate_from_mem0_basic() {
        let jsonl = r#"{"id":"1","memory":"First memory","user_id":"user1"}
{"id":"2","memory":"Second memory","user_id":"user1"}
{"id":"3","memory":"Third memory","user_id":"user2"}"#;

        let reader = BufReader::new(Cursor::new(jsonl));

        let imported = Arc::new(Mutex::new(Vec::new()));
        let imported_clone = imported.clone();

        let stats = migrate_from_mem0(reader, 10, |batch| {
            let imported = imported_clone.clone();
            async move {
                let count = batch.len();
                imported.lock().unwrap().extend(batch);
                Ok(count)
            }
        })
        .await
        .unwrap();

        assert_eq!(stats.total, 3);
        assert_eq!(stats.migrated, 3);
        assert!(stats.errors.is_empty());

        let imported = imported.lock().unwrap();
        assert_eq!(imported.len(), 3);
        assert_eq!(imported[0].id, "1");
        assert_eq!(imported[1].id, "2");
        assert_eq!(imported[2].id, "3");
    }

    #[tokio::test]
    async fn test_migrate_from_mem0_with_errors() {
        let jsonl = r#"{"id":"1","memory":"Valid memory"}
invalid json here
{"id":"2","memory":"Another valid memory"}"#;

        let reader = BufReader::new(Cursor::new(jsonl));

        let stats = migrate_from_mem0(reader, 100, |batch| async move { Ok(batch.len()) })
            .await
            .unwrap();

        assert_eq!(stats.total, 3);
        assert_eq!(stats.migrated, 2);
        assert_eq!(stats.skipped, 1);
        assert_eq!(stats.errors.len(), 1);
        assert!(stats.errors[0].contains("Parse error"));
    }

    #[tokio::test]
    async fn test_migrate_from_mem0_batching() {
        let jsonl = (0..25)
            .map(|i| format!(r#"{{"id":"{}","memory":"Memory {}"}}"#, i, i))
            .collect::<Vec<_>>()
            .join("\n");

        let reader = BufReader::new(Cursor::new(jsonl));
        let mut batch_counts = Vec::new();

        let stats = migrate_from_mem0(reader, 10, |batch| {
            let count = batch.len();
            batch_counts.push(count);
            async move { Ok(count) }
        })
        .await
        .unwrap();

        assert_eq!(stats.total, 25);
        assert_eq!(stats.migrated, 25);

        // Should have 3 batches: 10, 10, 5
        assert_eq!(batch_counts.len(), 3);
        assert_eq!(batch_counts[0], 10);
        assert_eq!(batch_counts[1], 10);
        assert_eq!(batch_counts[2], 5);
    }

    #[tokio::test]
    async fn test_migrate_from_mem0_empty_lines() {
        let jsonl = r#"{"id":"1","memory":"First"}

{"id":"2","memory":"Second"}

"#;

        let reader = BufReader::new(Cursor::new(jsonl));

        let stats = migrate_from_mem0(reader, 100, |batch| async move { Ok(batch.len()) })
            .await
            .unwrap();

        // Empty lines should be skipped
        assert_eq!(stats.total, 2);
        assert_eq!(stats.migrated, 2);
    }

    #[tokio::test]
    async fn test_migrate_from_mem0_preserves_metadata() {
        let jsonl = r#"{"id":"1","memory":"Test","metadata":{"custom_key":"custom_value"},"user_id":"u1","agent_id":"a1","categories":["cat1","cat2"]}"#;

        let reader = BufReader::new(Cursor::new(jsonl));
        let mut captured: Vec<MemoryItem> = Vec::new();

        let stats = migrate_from_mem0(reader, 100, |batch| {
            captured.extend(batch.clone());
            async move { Ok(batch.len()) }
        })
        .await
        .unwrap();

        assert_eq!(stats.migrated, 1);
        assert_eq!(captured.len(), 1);

        let mem = &captured[0];
        let metadata = mem.metadata.as_ref().unwrap();

        // Check preserved fields
        assert_eq!(metadata.get("user_id"), Some(&serde_json::json!("u1")));
        assert_eq!(metadata.get("agent_id"), Some(&serde_json::json!("a1")));
        assert_eq!(
            metadata.get("categories"),
            Some(&serde_json::json!(["cat1", "cat2"]))
        );
        assert_eq!(
            metadata.get("custom_key"),
            Some(&serde_json::json!("custom_value"))
        );
        assert_eq!(
            metadata.get("migrated_from"),
            Some(&serde_json::json!("mem0"))
        );
    }

    #[test]
    fn test_migration_stats() {
        let mut stats = MigrationStats::new();
        stats.total = 100;
        stats.migrated = 95;
        stats.skipped = 3;
        stats.errors.push("Error 1".to_string());
        stats.errors.push("Error 2".to_string());

        assert!(!stats.is_success());
        assert!((stats.success_rate() - 95.0).abs() < 0.01);
        assert!((stats.error_rate() - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_migration_stats_empty() {
        let stats = MigrationStats::new();

        assert!(stats.is_success());
        assert!((stats.success_rate() - 100.0).abs() < 0.01);
        assert!((stats.error_rate() - 0.0).abs() < 0.01);
    }
}
