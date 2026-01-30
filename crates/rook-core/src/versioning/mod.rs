//! Memory versioning system for audit trails and point-in-time queries
//!
//! Each memory mutation creates a new immutable version snapshot,
//! enabling historical queries like "what did this memory contain last week?"

mod store;
mod version;

pub use store::{SqliteVersionStore, VersionStore};
pub use version::{FsrsStateSnapshot, MemoryVersion, VersionEventType, VersionSummary};
