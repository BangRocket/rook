//! JSON Lines import for memory data.
//!
//! Provides streaming import from JSON Lines format with batched processing.
//! This format allows line-by-line reading without loading the entire file.

use crate::RookResult;
use serde::Deserialize;
use std::collections::HashMap;
use std::future::Future;
use tokio::io::{AsyncBufRead, AsyncBufReadExt};

/// Statistics from an import operation.
#[derive(Debug, Default, Clone)]
pub struct ImportStats {
    /// Total lines processed.
    pub total: u64,
    /// Successfully imported memories.
    pub imported: u64,
    /// Skipped (e.g., duplicates).
    pub skipped: u64,
    /// Error messages for failed imports.
    pub errors: Vec<String>,
}

impl ImportStats {
    /// Create new empty stats.
    pub fn new() -> Self {
        Self::default()
    }

    /// Check if import completed without errors.
    pub fn is_success(&self) -> bool {
        self.errors.is_empty()
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

/// Memory data in import format.
///
/// Mirrors ExportableMemory for deserialization during import.
#[derive(Debug, Clone, Deserialize)]
pub struct ImportableMemory {
    /// Unique identifier.
    pub id: String,
    /// Memory content text.
    pub memory: String,
    /// MD5 hash of content.
    #[serde(default)]
    pub hash: Option<String>,
    /// Custom metadata.
    #[serde(default)]
    pub metadata: Option<HashMap<String, serde_json::Value>>,
    /// Creation timestamp (ISO 8601).
    #[serde(default)]
    pub created_at: Option<String>,
    /// Last update timestamp (ISO 8601).
    #[serde(default)]
    pub updated_at: Option<String>,
    /// FSRS state as JSON value.
    #[serde(default)]
    pub fsrs_state: Option<serde_json::Value>,
    /// Dual-strength model state.
    #[serde(default)]
    pub dual_strength: Option<serde_json::Value>,
    /// Category classification.
    #[serde(default)]
    pub category: Option<String>,
    /// Whether this is a key memory.
    #[serde(default)]
    pub is_key: bool,
}

/// Import memories from JSON Lines format.
///
/// Reads lines from the input, parses each as JSON, and processes in batches.
/// Malformed lines are logged as errors but don't abort the import.
///
/// # Arguments
///
/// * `reader` - Async buffered reader source
/// * `batch_size` - Number of memories per batch (default: 100)
/// * `import_batch` - Callback to import each batch, returns count imported
///
/// # Returns
///
/// Import statistics including counts and any errors.
///
/// # Example
///
/// ```ignore
/// use tokio::fs::File;
/// use tokio::io::BufReader;
///
/// let file = File::open("memories.jsonl").await?;
/// let reader = BufReader::new(file);
/// let stats = import_jsonl(reader, 100, |batch| async move {
///     // Import batch and return count
///     Ok(batch.len())
/// }).await?;
/// ```
pub async fn import_jsonl<R, F, Fut>(
    reader: R,
    batch_size: usize,
    mut import_batch: F,
) -> RookResult<ImportStats>
where
    R: AsyncBufRead + Unpin,
    F: FnMut(Vec<ImportableMemory>) -> Fut,
    Fut: Future<Output = RookResult<usize>>,
{
    let mut stats = ImportStats::new();
    let mut batch = Vec::with_capacity(batch_size);
    let mut lines = reader.lines();

    while let Some(line_result) = lines.next_line().await? {
        let line = line_result.trim();

        // Skip empty lines
        if line.is_empty() {
            continue;
        }

        stats.total += 1;

        // Parse the JSON line
        match serde_json::from_str::<ImportableMemory>(line) {
            Ok(memory) => {
                batch.push(memory);

                // Process batch when full
                if batch.len() >= batch_size {
                    match import_batch(std::mem::take(&mut batch)).await {
                        Ok(count) => {
                            stats.imported += count as u64;
                        }
                        Err(e) => {
                            stats.errors.push(format!("Batch import error: {}", e));
                        }
                    }
                    batch = Vec::with_capacity(batch_size);
                }
            }
            Err(e) => {
                stats.errors.push(format!(
                    "Parse error at line {}: {}",
                    stats.total, e
                ));
            }
        }
    }

    // Process remaining batch
    if !batch.is_empty() {
        let batch_count = batch.len();
        match import_batch(batch).await {
            Ok(count) => {
                stats.imported += count as u64;
                stats.skipped += (batch_count - count) as u64;
            }
            Err(e) => {
                stats.errors.push(format!("Final batch import error: {}", e));
            }
        }
    }

    Ok(stats)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    use tokio::io::BufReader;

    #[tokio::test]
    async fn test_import_jsonl_basic() {
        let jsonl = r#"{"id":"1","memory":"First memory"}
{"id":"2","memory":"Second memory"}
{"id":"3","memory":"Third memory"}"#;

        let reader = BufReader::new(Cursor::new(jsonl));

        let stats = import_jsonl(reader, 100, |batch| async move {
            Ok(batch.len())
        })
        .await
        .unwrap();

        assert_eq!(stats.total, 3);
        assert_eq!(stats.imported, 3);
        assert!(stats.errors.is_empty());
    }

    #[tokio::test]
    async fn test_import_jsonl_with_batching() {
        let jsonl = (0..25)
            .map(|i| format!(r#"{{"id":"{}","memory":"Memory {}"}}"#, i, i))
            .collect::<Vec<_>>()
            .join("\n");

        let reader = BufReader::new(Cursor::new(jsonl));
        let mut batch_counts = Vec::new();

        let stats = import_jsonl(reader, 10, |batch| {
            let count = batch.len();
            batch_counts.push(count);
            async move { Ok(count) }
        })
        .await
        .unwrap();

        assert_eq!(stats.total, 25);
        assert_eq!(stats.imported, 25);
        // Should have 3 batches: 10, 10, 5
        assert_eq!(batch_counts.len(), 3);
        assert_eq!(batch_counts[0], 10);
        assert_eq!(batch_counts[1], 10);
        assert_eq!(batch_counts[2], 5);
    }

    #[tokio::test]
    async fn test_import_jsonl_with_errors() {
        let jsonl = r#"{"id":"1","memory":"Valid memory"}
invalid json here
{"id":"2","memory":"Another valid memory"}"#;

        let reader = BufReader::new(Cursor::new(jsonl));

        let stats = import_jsonl(reader, 100, |batch| async move {
            Ok(batch.len())
        })
        .await
        .unwrap();

        assert_eq!(stats.total, 3);
        assert_eq!(stats.imported, 2);
        assert_eq!(stats.errors.len(), 1);
        assert!(stats.errors[0].contains("Parse error"));
    }

    #[tokio::test]
    async fn test_import_jsonl_empty_lines() {
        let jsonl = r#"{"id":"1","memory":"First"}

{"id":"2","memory":"Second"}

"#;

        let reader = BufReader::new(Cursor::new(jsonl));

        let stats = import_jsonl(reader, 100, |batch| async move {
            Ok(batch.len())
        })
        .await
        .unwrap();

        // Empty lines should be skipped
        assert_eq!(stats.total, 2);
        assert_eq!(stats.imported, 2);
    }

    #[tokio::test]
    async fn test_import_jsonl_with_all_fields() {
        let jsonl = r#"{"id":"1","memory":"Test","hash":"abc","metadata":{"key":"value"},"created_at":"2024-01-01T00:00:00Z","category":"work","is_key":true}"#;

        let reader = BufReader::new(Cursor::new(jsonl));
        let mut captured: Vec<ImportableMemory> = Vec::new();

        let stats = import_jsonl(reader, 100, |batch| {
            captured.extend(batch.clone());
            async move { Ok(batch.len()) }
        })
        .await
        .unwrap();

        assert_eq!(stats.imported, 1);
        assert_eq!(captured.len(), 1);

        let mem = &captured[0];
        assert_eq!(mem.id, "1");
        assert_eq!(mem.memory, "Test");
        assert_eq!(mem.hash, Some("abc".to_string()));
        assert!(mem.metadata.is_some());
        assert_eq!(mem.category, Some("work".to_string()));
        assert!(mem.is_key);
    }

    #[tokio::test]
    async fn test_import_jsonl_empty_input() {
        let reader = BufReader::new(Cursor::new(""));

        let stats = import_jsonl(reader, 100, |batch| async move {
            Ok(batch.len())
        })
        .await
        .unwrap();

        assert_eq!(stats.total, 0);
        assert_eq!(stats.imported, 0);
        assert!(stats.is_success());
    }

    #[test]
    fn test_import_stats_error_rate() {
        let mut stats = ImportStats::new();
        stats.total = 100;
        stats.errors.push("Error 1".to_string());
        stats.errors.push("Error 2".to_string());

        assert!((stats.error_rate() - 2.0).abs() < 0.01);
    }
}
