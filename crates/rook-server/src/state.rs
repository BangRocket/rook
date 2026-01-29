//! Server state management.

use std::sync::Arc;

use rook_core::config::MemoryConfig;
use rook_core::error::RookResult;
use rook_core::memory::Memory;
use tokio::sync::RwLock;

use crate::factory::create_memory;

/// Shared application state.
#[derive(Clone)]
pub struct AppState {
    pub inner: Arc<RwLock<AppStateInner>>,
}

pub struct AppStateInner {
    pub memory: Option<Memory>,
    pub config: Option<MemoryConfig>,
}

impl AppState {
    /// Create a new application state.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(AppStateInner {
                memory: None,
                config: None,
            })),
        }
    }

    /// Create with pre-configured memory.
    pub fn new_with_memory(memory: Memory, config: MemoryConfig) -> Self {
        Self {
            inner: Arc::new(RwLock::new(AppStateInner {
                memory: Some(memory),
                config: Some(config),
            })),
        }
    }

    /// Check if memory is configured.
    pub async fn is_configured(&self) -> bool {
        self.inner.read().await.memory.is_some()
    }

    /// Get a reference to the memory instance for operations.
    /// Returns a guard that provides access to the memory.
    pub async fn with_memory<F, T>(&self, f: F) -> Option<T>
    where
        F: FnOnce(&Memory) -> T,
    {
        let guard = self.inner.read().await;
        guard.memory.as_ref().map(f)
    }

    /// Get a reference to the memory instance for async operations.
    pub async fn with_memory_async<F, Fut, T>(&self, f: F) -> Option<T>
    where
        F: FnOnce(&Memory) -> Fut,
        Fut: std::future::Future<Output = T>,
    {
        let guard = self.inner.read().await;
        if let Some(ref memory) = guard.memory {
            Some(f(memory).await)
        } else {
            None
        }
    }

    /// Configure the memory instance.
    pub async fn configure(&self, config: MemoryConfig) -> RookResult<()> {
        let memory = create_memory(config.clone()).await?;
        let mut guard = self.inner.write().await;
        guard.memory = Some(memory);
        guard.config = Some(config);
        Ok(())
    }

    /// Reset the memory instance.
    pub async fn reset(&self) -> RookResult<()> {
        let guard = self.inner.read().await;
        if let Some(ref memory) = guard.memory {
            memory.reset().await?;
        }
        Ok(())
    }
}

impl Default for AppState {
    fn default() -> Self {
        Self::new()
    }
}