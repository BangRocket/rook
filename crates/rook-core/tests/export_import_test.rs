//! Integration tests for export/import functionality.
//!
//! Tests round-trip export and import to verify data integrity.

use futures::stream;
use rook_core::{
    export_jsonl, import_jsonl, ExportStats, ExportableMemory, ImportStats, ImportableMemory,
    MemoryItem,
};
use std::collections::HashMap;
use std::io::Cursor;
use tokio::io::BufReader;

/// Test JSON Lines round-trip: export then import preserves data.
#[tokio::test]
async fn test_jsonl_round_trip() {
    // Create test memories with various fields
    let mut metadata = HashMap::new();
    metadata.insert("source".to_string(), serde_json::json!("test"));
    metadata.insert("confidence".to_string(), serde_json::json!(0.95));

    let original_memories = vec![
        MemoryItem::new("id1", "First memory - simple text"),
        MemoryItem::new("id2", "Second memory with category")
            .with_category("professional")
            .with_is_key(true),
        MemoryItem::new("id3", "Third memory with metadata")
            .with_metadata(metadata.clone())
            .with_hash("abc123"),
    ];

    // Export to buffer
    let mut export_buffer = Vec::new();
    let export_stats = export_jsonl(stream::iter(original_memories.clone()), &mut export_buffer)
        .await
        .unwrap();

    assert_eq!(export_stats.total, 3);
    assert_eq!(export_stats.exported, 3);
    assert!(export_stats.is_success());

    // Verify export output
    let content = String::from_utf8(export_buffer.clone()).unwrap();
    let lines: Vec<&str> = content.lines().collect();
    assert_eq!(lines.len(), 3, "Should have 3 lines of JSON");

    // Import from buffer
    let reader = BufReader::new(Cursor::new(export_buffer));
    let mut imported: Vec<ImportableMemory> = Vec::new();

    let import_stats = import_jsonl(reader, 100, |batch| {
        imported.extend(batch.clone());
        async move { Ok(batch.len()) }
    })
    .await
    .unwrap();

    assert_eq!(import_stats.total, 3);
    assert_eq!(import_stats.imported, 3);
    assert!(import_stats.is_success());

    // Verify imported data matches original
    assert_eq!(imported.len(), 3);

    // Check first memory
    assert_eq!(imported[0].id, "id1");
    assert_eq!(imported[0].memory, "First memory - simple text");

    // Check second memory with category
    assert_eq!(imported[1].id, "id2");
    assert_eq!(imported[1].category, Some("professional".to_string()));
    assert!(imported[1].is_key);

    // Check third memory with metadata
    assert_eq!(imported[2].id, "id3");
    assert_eq!(imported[2].hash, Some("abc123".to_string()));
    assert!(imported[2].metadata.is_some());
}

/// Test JSON Lines handles large datasets via batching.
#[tokio::test]
async fn test_jsonl_large_dataset_batching() {
    // Create 1000 memories
    let memories: Vec<MemoryItem> = (0..1000)
        .map(|i| MemoryItem::new(format!("id-{}", i), format!("Memory content number {}", i)))
        .collect();

    // Export
    let mut export_buffer = Vec::new();
    let export_stats = export_jsonl(stream::iter(memories.clone()), &mut export_buffer)
        .await
        .unwrap();

    assert_eq!(export_stats.exported, 1000);

    // Import with batch size of 100
    let reader = BufReader::new(Cursor::new(export_buffer));
    let mut batch_sizes: Vec<usize> = Vec::new();
    let mut total_imported = 0;

    let import_stats = import_jsonl(reader, 100, |batch| {
        batch_sizes.push(batch.len());
        total_imported += batch.len();
        async move { Ok(batch.len()) }
    })
    .await
    .unwrap();

    assert_eq!(import_stats.total, 1000);
    assert_eq!(import_stats.imported, 1000);

    // Should have 10 batches of 100
    assert_eq!(batch_sizes.len(), 10);
    for size in &batch_sizes {
        assert_eq!(*size, 100);
    }
}

/// Test JSON Lines handles malformed data gracefully.
#[tokio::test]
async fn test_jsonl_error_handling() {
    // Mix of valid and invalid JSON
    let jsonl = r#"{"id":"1","memory":"Valid memory 1"}
not valid json
{"id":"2","memory":"Valid memory 2"}
{"this is also invalid
{"id":"3","memory":"Valid memory 3"}"#;

    let reader = BufReader::new(Cursor::new(jsonl));
    let mut imported_count = 0;

    let stats = import_jsonl(reader, 100, |batch| {
        imported_count += batch.len();
        async move { Ok(batch.len()) }
    })
    .await
    .unwrap();

    // Should have processed 5 lines total
    assert_eq!(stats.total, 5);
    // Should have imported 3 valid memories
    assert_eq!(stats.imported, 3);
    // Should have 2 parse errors
    assert_eq!(stats.errors.len(), 2);
    // Error messages should indicate parse errors
    assert!(stats.errors[0].contains("Parse error"));
    assert!(stats.errors[1].contains("Parse error"));
}

/// Test ExportableMemory conversion preserves all fields.
#[test]
fn test_exportable_memory_all_fields() {
    use rook_core::types::{DualStrength, FsrsState};

    let mut metadata = HashMap::new();
    metadata.insert("key".to_string(), serde_json::json!("value"));

    let mut fsrs_state = FsrsState::new();
    fsrs_state.stability = 10.0;
    fsrs_state.difficulty = 7.5;
    fsrs_state.reps = 5;

    let mut dual_strength = DualStrength::new();
    dual_strength.storage_strength = 0.8;
    dual_strength.retrieval_strength = 0.6;

    let memory = MemoryItem::new("test-id", "test content")
        .with_hash("hash123")
        .with_metadata(metadata)
        .with_created_at("2024-01-01T00:00:00Z")
        .with_updated_at("2024-01-02T00:00:00Z")
        .with_memory_state(fsrs_state)
        .with_dual_strength(dual_strength)
        .with_category("professional")
        .with_is_key(true);

    let exportable = ExportableMemory::from(memory);

    assert_eq!(exportable.id, "test-id");
    assert_eq!(exportable.memory, "test content");
    assert_eq!(exportable.hash, Some("hash123".to_string()));
    assert!(exportable.metadata.is_some());
    assert_eq!(exportable.created_at, Some("2024-01-01T00:00:00Z".to_string()));
    assert_eq!(exportable.updated_at, Some("2024-01-02T00:00:00Z".to_string()));
    assert!(exportable.fsrs_state.is_some());
    assert!(exportable.dual_strength.is_some());
    assert_eq!(exportable.category, Some("professional".to_string()));
    assert!(exportable.is_key);
}

/// Test import statistics tracking.
#[test]
fn test_import_stats() {
    let mut stats = ImportStats::default();

    assert_eq!(stats.total, 0);
    assert_eq!(stats.imported, 0);
    assert_eq!(stats.skipped, 0);
    assert!(stats.errors.is_empty());
    assert!(stats.is_success());
    assert_eq!(stats.error_rate(), 0.0);

    // Add some stats
    stats.total = 100;
    stats.imported = 95;
    stats.skipped = 3;
    stats.errors.push("Error 1".to_string());
    stats.errors.push("Error 2".to_string());

    assert!(!stats.is_success());
    assert!((stats.error_rate() - 2.0).abs() < 0.01);
}

/// Test export statistics tracking.
#[test]
fn test_export_stats() {
    let mut stats = ExportStats::default();

    assert_eq!(stats.total, 0);
    assert_eq!(stats.exported, 0);
    assert!(stats.errors.is_empty());
    assert!(stats.is_success());

    // With data
    stats.total = 100;
    stats.exported = 100;
    assert!(stats.is_success());

    // With error
    stats.errors.push("Error".to_string());
    assert!(!stats.is_success());
}

// Feature-gated Parquet tests
#[cfg(feature = "export")]
mod parquet_tests {
    use super::*;
    use rook_core::export_parquet;

    /// Test Parquet export creates valid Parquet file.
    #[tokio::test]
    async fn test_parquet_magic_bytes() {
        let memories = vec![
            MemoryItem::new("id1", "First memory"),
            MemoryItem::new("id2", "Second memory"),
        ];

        let mut output = Vec::new();
        let stats = export_parquet(memories, &mut output, None).await.unwrap();

        assert_eq!(stats.exported, 2);

        // Parquet files start with "PAR1" magic bytes
        assert!(output.len() > 4);
        assert_eq!(&output[0..4], b"PAR1", "File should start with Parquet magic bytes");

        // Parquet files also end with "PAR1"
        let len = output.len();
        assert_eq!(&output[len - 4..], b"PAR1", "File should end with Parquet magic bytes");
    }

    /// Test Parquet export produces reasonable file size (compressed).
    #[tokio::test]
    async fn test_parquet_compression() {
        // Create memories with repeated content (compressible)
        let memories: Vec<MemoryItem> = (0..100)
            .map(|i| {
                MemoryItem::new(
                    format!("id-{}", i),
                    format!(
                        "This is a longer memory content with repeated phrases. \
                         Memory number {}. The quick brown fox jumps over the lazy dog. \
                         This content should compress well.",
                        i
                    ),
                )
            })
            .collect();

        let mut output = Vec::new();
        let stats = export_parquet(memories, &mut output, None).await.unwrap();

        assert_eq!(stats.exported, 100);

        // File should exist and be reasonably sized
        // 100 memories with ~150 bytes each = ~15KB uncompressed
        // With ZSTD compression should be much smaller
        assert!(output.len() > 0);
        // Parquet with ZSTD should compress significantly
        assert!(output.len() < 15000, "Compressed size should be less than uncompressed");
    }

    /// Test Parquet export with all memory fields.
    #[tokio::test]
    async fn test_parquet_all_fields() {
        use rook_core::types::{DualStrength, FsrsState};

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
        assert!(output.len() > 8); // At least header and footer
        assert_eq!(&output[0..4], b"PAR1");
    }
}
