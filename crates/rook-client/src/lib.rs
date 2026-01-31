//! rook-client - Client library for the Rook hosted API.
//!
//! This crate provides a client for interacting with the Rook hosted API.
//!
//! # Example
//!
//! ```ignore
//! use rook_client::{MemoryClient, SignalInput};
//!
//! let client = MemoryClient::new("your-api-key")?;
//!
//! // Add a memory
//! let result = client.add("I love programming in Rust", "user-123", None).await?;
//!
//! // Search memories
//! let memories = client.search("programming", "user-123", 10).await?;
//!
//! // Send a signal that a memory was used
//! client.send_signal(SignalInput::UsedInResponse {
//!     memory_id: "mem-123".to_string(),
//!     context: Some("answered user question".to_string()),
//! }).await?;
//! ```

mod client;

pub use client::{MemoryClient, PendingUpdate, SignalInput, SignalsResponse};
pub use rook_core::types::MemoryItem;
