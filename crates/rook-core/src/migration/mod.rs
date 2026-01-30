//! Migration utilities for importing data from other memory systems.
//!
//! Currently supports:
//! - mem0: Import from mem0 export files
//!
//! # Example
//!
//! ```rust,ignore
//! use rook_core::migration::{migrate_from_mem0, MigrationStats};
//! use tokio::fs::File;
//! use tokio::io::BufReader;
//!
//! async fn migrate() -> rook_core::RookResult<()> {
//!     let file = File::open("mem0_export.jsonl").await?;
//!     let reader = BufReader::new(file);
//!
//!     let stats = migrate_from_mem0(reader, 100, |batch| async {
//!         // Import batch into Rook
//!         println!("Importing {} memories", batch.len());
//!         Ok(batch.len())
//!     }).await?;
//!
//!     println!("Migrated {}/{} memories", stats.migrated, stats.total);
//!     Ok(())
//! }
//! ```

pub mod mem0;

pub use mem0::{migrate_from_mem0, Mem0Memory, MigrationStats};
