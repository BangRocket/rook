//! Import utilities for memory data.
//!
//! Supports JSON Lines format with batched processing.
//!
//! # Example
//!
//! ```ignore
//! use rook_core::import::{import_jsonl, ImportStats};
//! use tokio::fs::File;
//! use tokio::io::BufReader;
//!
//! // Import from JSON Lines
//! let file = File::open("memories.jsonl").await?;
//! let reader = BufReader::new(file);
//! let stats = import_jsonl(reader, 100, |batch| async {
//!     memory.import_batch(batch).await
//! }).await?;
//! println!("Imported {}/{}", stats.imported, stats.total);
//! ```

pub mod jsonl;

pub use jsonl::{import_jsonl, ImportStats, ImportableMemory};
