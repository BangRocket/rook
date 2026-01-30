//! Parquet export for memory data.
//!
//! Provides columnar export to Apache Parquet format with compression.
//! This format is ideal for:
//! - Analytics workloads (columnar storage)
//! - Long-term archival (efficient compression)
//! - Interoperability with data processing tools (Spark, DuckDB, etc.)

use crate::{MemoryItem, RookResult, RookError};
use crate::export::ExportStats;
use arrow::array::{ArrayRef, BooleanArray, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::AsyncArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use std::sync::Arc;
use tokio::io::AsyncWrite;

/// Create the Arrow schema for memory data.
fn memory_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("memory", DataType::Utf8, false),
        Field::new("hash", DataType::Utf8, true),
        Field::new("metadata", DataType::Utf8, true), // JSON string
        Field::new("created_at", DataType::Utf8, true),
        Field::new("updated_at", DataType::Utf8, true),
        Field::new("fsrs_state", DataType::Utf8, true), // JSON string
        Field::new("dual_strength", DataType::Utf8, true), // JSON string
        Field::new("category", DataType::Utf8, true),
        Field::new("is_key", DataType::Boolean, false),
    ]))
}

/// Export memories to Parquet format.
///
/// Creates a Parquet file with ZSTD compression (default).
/// All memories are loaded into a RecordBatch for columnar storage.
///
/// # Arguments
///
/// * `memories` - Vector of MemoryItem to export
/// * `writer` - Async writer destination
/// * `compression` - Optional compression (default: ZSTD)
///
/// # Returns
///
/// Export statistics including count and any errors.
///
/// # Example
///
/// ```ignore
/// use tokio::fs::File;
/// use parquet::basic::Compression;
///
/// let file = File::create("memories.parquet").await?;
/// let stats = export_parquet(memories, file, Some(Compression::ZSTD(Default::default()))).await?;
/// ```
pub async fn export_parquet<W>(
    memories: Vec<MemoryItem>,
    writer: W,
    compression: Option<Compression>,
) -> RookResult<ExportStats>
where
    W: AsyncWrite + Unpin + Send,
{
    let mut stats = ExportStats::new();
    stats.total = memories.len() as u64;

    if memories.is_empty() {
        return Ok(stats);
    }

    let schema = memory_schema();

    // Build Arrow arrays from memories
    let ids: Vec<&str> = memories.iter().map(|m| m.id.as_str()).collect();
    let memory_texts: Vec<&str> = memories.iter().map(|m| m.memory.as_str()).collect();
    let hashes: Vec<Option<&str>> = memories
        .iter()
        .map(|m| m.hash.as_deref())
        .collect();
    let metadata_jsons: Vec<Option<String>> = memories
        .iter()
        .map(|m| {
            m.metadata
                .as_ref()
                .and_then(|meta| serde_json::to_string(meta).ok())
        })
        .collect();
    let created_ats: Vec<Option<&str>> = memories
        .iter()
        .map(|m| m.created_at.as_deref())
        .collect();
    let updated_ats: Vec<Option<&str>> = memories
        .iter()
        .map(|m| m.updated_at.as_deref())
        .collect();
    let fsrs_states: Vec<Option<String>> = memories
        .iter()
        .map(|m| {
            m.memory_state
                .as_ref()
                .and_then(|s| serde_json::to_string(s).ok())
        })
        .collect();
    let dual_strengths: Vec<Option<String>> = memories
        .iter()
        .map(|m| {
            m.dual_strength
                .as_ref()
                .and_then(|s| serde_json::to_string(s).ok())
        })
        .collect();
    let categories: Vec<Option<&str>> = memories
        .iter()
        .map(|m| m.category.as_deref())
        .collect();
    let is_keys: Vec<bool> = memories.iter().map(|m| m.is_key).collect();

    // Create Arrow arrays
    let id_array: ArrayRef = Arc::new(StringArray::from(ids));
    let memory_array: ArrayRef = Arc::new(StringArray::from(memory_texts));
    let hash_array: ArrayRef = Arc::new(StringArray::from(hashes));
    let metadata_array: ArrayRef = Arc::new(StringArray::from(
        metadata_jsons
            .iter()
            .map(|s| s.as_deref())
            .collect::<Vec<_>>(),
    ));
    let created_at_array: ArrayRef = Arc::new(StringArray::from(created_ats));
    let updated_at_array: ArrayRef = Arc::new(StringArray::from(updated_ats));
    let fsrs_state_array: ArrayRef = Arc::new(StringArray::from(
        fsrs_states
            .iter()
            .map(|s| s.as_deref())
            .collect::<Vec<_>>(),
    ));
    let dual_strength_array: ArrayRef = Arc::new(StringArray::from(
        dual_strengths
            .iter()
            .map(|s| s.as_deref())
            .collect::<Vec<_>>(),
    ));
    let category_array: ArrayRef = Arc::new(StringArray::from(categories));
    let is_key_array: ArrayRef = Arc::new(BooleanArray::from(is_keys));

    // Create RecordBatch
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            id_array,
            memory_array,
            hash_array,
            metadata_array,
            created_at_array,
            updated_at_array,
            fsrs_state_array,
            dual_strength_array,
            category_array,
            is_key_array,
        ],
    )
    .map_err(|e| RookError::internal(format!("Failed to create RecordBatch: {}", e)))?;

    // Configure writer properties
    let props = WriterProperties::builder()
        .set_compression(compression.unwrap_or(Compression::ZSTD(Default::default())))
        .build();

    // Create AsyncArrowWriter and write
    let mut arrow_writer = AsyncArrowWriter::try_new(writer, schema, Some(props))
        .map_err(|e| RookError::internal(format!("Failed to create Parquet writer: {}", e)))?;

    arrow_writer
        .write(&batch)
        .await
        .map_err(|e| RookError::internal(format!("Failed to write Parquet batch: {}", e)))?;

    arrow_writer
        .close()
        .await
        .map_err(|e| RookError::internal(format!("Failed to close Parquet writer: {}", e)))?;

    stats.exported = memories.len() as u64;
    Ok(stats)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{DualStrength, FsrsState};
    use std::collections::HashMap;

    #[tokio::test]
    async fn test_export_parquet_basic() {
        let memories = vec![
            MemoryItem::new("id1", "First memory"),
            MemoryItem::new("id2", "Second memory"),
        ];

        let mut output = Vec::new();
        let stats = export_parquet(memories, &mut output, None).await.unwrap();

        assert_eq!(stats.total, 2);
        assert_eq!(stats.exported, 2);
        assert!(stats.errors.is_empty());

        // Verify Parquet magic bytes (PAR1)
        assert!(output.len() > 4);
        assert_eq!(&output[0..4], b"PAR1");
    }

    #[tokio::test]
    async fn test_export_parquet_with_all_fields() {
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
        let stats = export_parquet(vec![memory], &mut output, None).await.unwrap();

        assert_eq!(stats.exported, 1);
        assert!(output.len() > 4);
        assert_eq!(&output[0..4], b"PAR1");
    }

    #[tokio::test]
    async fn test_export_parquet_empty() {
        let memories: Vec<MemoryItem> = vec![];

        let mut output = Vec::new();
        let stats = export_parquet(memories, &mut output, None).await.unwrap();

        assert_eq!(stats.total, 0);
        assert_eq!(stats.exported, 0);
        assert!(output.is_empty());
    }

    #[tokio::test]
    async fn test_export_parquet_with_compression() {
        use parquet::basic::ZstdLevel;

        let memories = vec![
            MemoryItem::new("id1", "First memory with some longer content to compress"),
            MemoryItem::new("id2", "Second memory with more content to see compression effects"),
        ];

        let mut output = Vec::new();
        let stats = export_parquet(
            memories,
            &mut output,
            Some(Compression::ZSTD(ZstdLevel::try_new(3).unwrap())),
        )
        .await
        .unwrap();

        assert_eq!(stats.exported, 2);
        assert!(output.len() > 4);
        assert_eq!(&output[0..4], b"PAR1");
    }

    #[test]
    fn test_memory_schema() {
        let schema = memory_schema();
        assert_eq!(schema.fields().len(), 10);

        // Verify field names
        let field_names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
        assert!(field_names.contains(&"id"));
        assert!(field_names.contains(&"memory"));
        assert!(field_names.contains(&"hash"));
        assert!(field_names.contains(&"metadata"));
        assert!(field_names.contains(&"fsrs_state"));
        assert!(field_names.contains(&"category"));
        assert!(field_names.contains(&"is_key"));
    }
}
