//! JSON Lines export for memory data.
//!
//! Provides streaming export to JSON Lines format, where each line is a complete
//! JSON object representing a memory. This format is ideal for:
//! - Line-by-line streaming (no need to load all data)
//! - Human-readable debugging
//! - Incremental processing and appending
//! - Unix pipeline compatibility (grep, jq, etc.)

use crate::{MemoryItem, RookResult};
use futures::Stream;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tokio::io::{AsyncWrite, AsyncWriteExt, BufWriter};

/// Statistics from an export operation.
#[derive(Debug, Default, Clone)]
pub struct ExportStats {
    /// Total memories processed.
    pub total: u64,
    /// Successfully exported memories.
    pub exported: u64,
    /// Error messages for failed exports.
    pub errors: Vec<String>,
}

impl ExportStats {
    /// Create new empty stats.
    pub fn new() -> Self {
        Self::default()
    }

    /// Check if export completed without errors.
    pub fn is_success(&self) -> bool {
        self.errors.is_empty() && self.total == self.exported
    }
}

/// Memory data in export format.
///
/// This is a serialization-focused subset of MemoryItem that includes
/// all fields needed for complete backup/restore.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportableMemory {
    /// Unique identifier.
    pub id: String,
    /// Memory content text.
    pub memory: String,
    /// MD5 hash of content.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hash: Option<String>,
    /// Custom metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, serde_json::Value>>,
    /// Creation timestamp (ISO 8601).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub created_at: Option<String>,
    /// Last update timestamp (ISO 8601).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub updated_at: Option<String>,
    /// FSRS state as JSON value.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fsrs_state: Option<serde_json::Value>,
    /// Dual-strength model state.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dual_strength: Option<serde_json::Value>,
    /// Category classification.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub category: Option<String>,
    /// Whether this is a key memory.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub is_key: bool,
}

impl From<MemoryItem> for ExportableMemory {
    fn from(item: MemoryItem) -> Self {
        Self {
            id: item.id,
            memory: item.memory,
            hash: item.hash,
            metadata: item.metadata,
            created_at: item.created_at,
            updated_at: item.updated_at,
            fsrs_state: item.memory_state.map(|s| serde_json::to_value(s).ok()).flatten(),
            dual_strength: item.dual_strength.map(|s| serde_json::to_value(s).ok()).flatten(),
            category: item.category,
            is_key: item.is_key,
        }
    }
}

/// Export memories to JSON Lines format.
///
/// Each memory is serialized as a single JSON object per line.
/// Uses buffered writing for efficient I/O.
///
/// # Arguments
///
/// * `memories` - Stream of MemoryItem to export
/// * `writer` - Async writer destination (file, buffer, etc.)
///
/// # Returns
///
/// Export statistics including count and any errors.
///
/// # Example
///
/// ```ignore
/// use tokio::fs::File;
/// use futures::stream;
///
/// let file = File::create("memories.jsonl").await?;
/// let memories = stream::iter(vec![memory1, memory2]);
/// let stats = export_jsonl(memories, file).await?;
/// ```
pub async fn export_jsonl<W, S>(
    memories: S,
    writer: W,
) -> RookResult<ExportStats>
where
    W: AsyncWrite + Unpin,
    S: Stream<Item = MemoryItem>,
{
    use futures::StreamExt;

    let mut stats = ExportStats::new();
    let mut writer = BufWriter::new(writer);
    let mut memories = std::pin::pin!(memories);

    while let Some(memory) = memories.next().await {
        stats.total += 1;

        let exportable = ExportableMemory::from(memory);
        match serde_json::to_string(&exportable) {
            Ok(json) => {
                // Write JSON line
                if let Err(e) = writer.write_all(json.as_bytes()).await {
                    stats.errors.push(format!(
                        "Write error for memory {}: {}",
                        exportable.id, e
                    ));
                    continue;
                }

                // Write newline
                if let Err(e) = writer.write_all(b"\n").await {
                    stats.errors.push(format!(
                        "Write newline error for memory {}: {}",
                        exportable.id, e
                    ));
                    continue;
                }

                stats.exported += 1;
            }
            Err(e) => {
                stats.errors.push(format!(
                    "Serialization error for memory {}: {}",
                    exportable.id, e
                ));
            }
        }
    }

    // Flush the buffer
    if let Err(e) = writer.flush().await {
        stats.errors.push(format!("Final flush error: {}", e));
    }

    Ok(stats)
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::stream;

    #[tokio::test]
    async fn test_export_jsonl_basic() {
        let memories = vec![
            MemoryItem::new("id1", "First memory"),
            MemoryItem::new("id2", "Second memory"),
        ];

        let mut output = Vec::new();
        let stats = export_jsonl(stream::iter(memories), &mut output).await.unwrap();

        assert_eq!(stats.total, 2);
        assert_eq!(stats.exported, 2);
        assert!(stats.errors.is_empty());

        // Verify output is valid JSONL
        let content = String::from_utf8(output).unwrap();
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(lines.len(), 2);

        // Each line should be valid JSON
        for line in lines {
            let _: ExportableMemory = serde_json::from_str(line).unwrap();
        }
    }

    #[tokio::test]
    async fn test_export_jsonl_with_all_fields() {
        use crate::types::{DualStrength, FsrsState};
        use std::collections::HashMap;

        let mut metadata = HashMap::new();
        metadata.insert("key".to_string(), serde_json::json!("value"));

        let memory = MemoryItem::new("id1", "Test memory")
            .with_hash("abc123")
            .with_metadata(metadata)
            .with_created_at("2024-01-01T00:00:00Z")
            .with_updated_at("2024-01-02T00:00:00Z")
            .with_memory_state(FsrsState::new())
            .with_dual_strength(DualStrength::new())
            .with_category("professional")
            .with_is_key(true);

        let mut output = Vec::new();
        let stats = export_jsonl(stream::iter(vec![memory]), &mut output).await.unwrap();

        assert_eq!(stats.exported, 1);

        let content = String::from_utf8(output).unwrap();
        let exported: ExportableMemory = serde_json::from_str(content.trim()).unwrap();

        assert_eq!(exported.id, "id1");
        assert_eq!(exported.memory, "Test memory");
        assert_eq!(exported.hash, Some("abc123".to_string()));
        assert!(exported.metadata.is_some());
        assert!(exported.fsrs_state.is_some());
        assert!(exported.dual_strength.is_some());
        assert_eq!(exported.category, Some("professional".to_string()));
        assert!(exported.is_key);
    }

    #[tokio::test]
    async fn test_export_jsonl_empty_stream() {
        let memories: Vec<MemoryItem> = vec![];

        let mut output = Vec::new();
        let stats = export_jsonl(stream::iter(memories), &mut output).await.unwrap();

        assert_eq!(stats.total, 0);
        assert_eq!(stats.exported, 0);
        assert!(stats.is_success());
        assert!(output.is_empty());
    }

    #[test]
    fn test_exportable_memory_from_memory_item() {
        let item = MemoryItem::new("test-id", "test content")
            .with_category("personal");

        let exportable = ExportableMemory::from(item);

        assert_eq!(exportable.id, "test-id");
        assert_eq!(exportable.memory, "test content");
        assert_eq!(exportable.category, Some("personal".to_string()));
    }
}
