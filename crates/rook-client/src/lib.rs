//! rook-client - Client library for the Rook hosted API.
//!
//! This crate provides a client for interacting with the Rook hosted API.
//!
//! # Example
//!
//! ```ignore
//! use rook_client::MemoryClient;
//!
//! let client = MemoryClient::new("your-api-key")?;
//!
//! // Add a memory
//! let result = client.add("I love programming in Rust", "user-123", None).await?;
//!
//! // Search memories
//! let memories = client.search("programming", "user-123", 10).await?;
//! ```

mod client;

pub use client::MemoryClient;
pub use rook_core::types::MemoryItem;
