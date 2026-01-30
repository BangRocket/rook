//! Export utilities for memory data.
//!
//! Supports JSON Lines (streaming, human-readable) and Parquet (columnar, compressed).
//!
//! # Example
//!
//! ```ignore
//! use rook_core::export::{export_jsonl, ExportStats};
//! use tokio::fs::File;
//!
//! // Export to JSON Lines
//! let file = File::create("memories.jsonl").await?;
//! let stats = export_jsonl(memory_stream, file).await?;
//! println!("Exported {} memories", stats.exported);
//! ```

pub mod jsonl;
#[cfg(feature = "export")]
pub mod parquet;

pub use jsonl::{export_jsonl, ExportStats, ExportableMemory};
#[cfg(feature = "export")]
pub use self::parquet::export_parquet;
