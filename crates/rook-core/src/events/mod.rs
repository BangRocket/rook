//! Event system for memory lifecycle events
//!
//! This module provides:
//! - Event types for memory operations (created, updated, deleted, accessed)
//! - Event bus for internal pub/sub
//! - Webhook delivery for external integrations

mod bus;
mod event;
mod webhook;

pub use bus::{EventBus, EventSubscriber};
pub use event::{
    AccessType, MemoryAccessedEvent, MemoryCreatedEvent, MemoryDeletedEvent, MemoryLifecycleEvent,
    MemoryUpdatedEvent, UpdateType,
};
pub use webhook::{verify_signature, RetryPolicy, WebhookConfig, WebhookDelivery, WebhookError, WebhookManager};
