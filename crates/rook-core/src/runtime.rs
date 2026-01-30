//! Background runtime for memory schedulers.
//!
//! Manages the lifecycle of ConsolidationScheduler and IntentionScheduler
//! as background tasks, providing unified startup and graceful shutdown.

use std::sync::Arc;

use tracing::{debug, info};

use crate::cognitive::CognitiveStore;
use crate::consolidation::{ConsolidationManager, ConsolidationScheduler, SchedulerConfig};
use crate::error::{RookError, RookResult};
use crate::intentions::{
    FiredIntentionReceiver, IntentionScheduler, IntentionStore, SqliteIntentionStore,
};

/// Configuration for the BackgroundRuntime.
#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    /// Interval between consolidation runs in minutes (default: 15).
    pub consolidation_interval_minutes: u64,
    /// Whether to run consolidation immediately on start (default: false).
    pub consolidation_run_on_start: bool,
    /// Whether to enable the consolidation scheduler (default: true).
    pub enable_consolidation: bool,
    /// Whether to enable the intention scheduler (default: true).
    pub enable_intentions: bool,
    /// Path to cognitive store SQLite database (default: None = in-memory).
    pub cognitive_db_path: Option<String>,
    /// Path to intention store SQLite database (default: None = in-memory).
    pub intention_db_path: Option<String>,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            consolidation_interval_minutes: 15,
            consolidation_run_on_start: false,
            enable_consolidation: true,
            enable_intentions: true,
            cognitive_db_path: None,
            intention_db_path: None,
        }
    }
}

impl RuntimeConfig {
    /// Create a new runtime config with custom consolidation interval.
    pub fn with_consolidation_interval(mut self, minutes: u64) -> Self {
        self.consolidation_interval_minutes = minutes.max(1);
        self
    }

    /// Enable running consolidation immediately on start.
    pub fn with_run_on_start(mut self) -> Self {
        self.consolidation_run_on_start = true;
        self
    }

    /// Disable consolidation scheduler.
    pub fn without_consolidation(mut self) -> Self {
        self.enable_consolidation = false;
        self
    }

    /// Disable intention scheduler.
    pub fn without_intentions(mut self) -> Self {
        self.enable_intentions = false;
        self
    }

    /// Set path for cognitive store database.
    pub fn with_cognitive_db_path(mut self, path: impl Into<String>) -> Self {
        self.cognitive_db_path = Some(path.into());
        self
    }

    /// Set path for intention store database.
    pub fn with_intention_db_path(mut self, path: impl Into<String>) -> Self {
        self.intention_db_path = Some(path.into());
        self
    }

    /// Create config from environment variables.
    ///
    /// Reads:
    /// - `ROOK_CONSOLIDATION_INTERVAL_MINUTES` (default: 15)
    /// - `ROOK_CONSOLIDATION_RUN_ON_START` (default: false)
    /// - `ROOK_ENABLE_CONSOLIDATION` (default: true)
    /// - `ROOK_ENABLE_INTENTIONS` (default: true)
    /// - `ROOK_COGNITIVE_DB_PATH` (default: None = in-memory)
    /// - `ROOK_INTENTION_DB_PATH` (default: None = in-memory)
    pub fn from_env() -> Self {
        let mut config = Self::default();

        if let Ok(interval) = std::env::var("ROOK_CONSOLIDATION_INTERVAL_MINUTES") {
            if let Ok(minutes) = interval.parse() {
                config.consolidation_interval_minutes = minutes;
            }
        }

        if std::env::var("ROOK_CONSOLIDATION_RUN_ON_START").is_ok() {
            config.consolidation_run_on_start = true;
        }

        if std::env::var("ROOK_DISABLE_CONSOLIDATION").is_ok() {
            config.enable_consolidation = false;
        }

        if std::env::var("ROOK_DISABLE_INTENTIONS").is_ok() {
            config.enable_intentions = false;
        }

        if let Ok(path) = std::env::var("ROOK_COGNITIVE_DB_PATH") {
            config.cognitive_db_path = Some(path);
        }

        if let Ok(path) = std::env::var("ROOK_INTENTION_DB_PATH") {
            config.intention_db_path = Some(path);
        }

        config
    }
}

/// Background runtime managing scheduler lifecycle.
///
/// Provides unified startup and shutdown for:
/// - ConsolidationScheduler (periodic memory consolidation)
/// - IntentionScheduler (time-based intention triggers)
///
/// # Example
///
/// ```ignore
/// use rook_core::{BackgroundRuntime, RuntimeConfig};
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let config = RuntimeConfig::default();
///     let mut runtime = BackgroundRuntime::new(config).await?;
///
///     // Start background schedulers
///     runtime.start().await?;
///
///     // ... application runs ...
///
///     // Graceful shutdown
///     runtime.shutdown().await?;
///     Ok(())
/// }
/// ```
pub struct BackgroundRuntime {
    /// Consolidation scheduler (optional based on config).
    consolidation_scheduler: Option<ConsolidationScheduler>,
    /// Intention scheduler (optional based on config).
    intention_scheduler: Option<IntentionScheduler>,
    /// Channel receiver for fired intentions (taken on first access).
    fired_intentions_rx: Option<FiredIntentionReceiver>,
    /// Cognitive store (shared with consolidation manager).
    cognitive_store: Arc<CognitiveStore>,
    /// Intention store (shared with intention scheduler).
    intention_store: Arc<dyn IntentionStore>,
    /// Runtime configuration.
    config: RuntimeConfig,
}

impl BackgroundRuntime {
    /// Create a new BackgroundRuntime with the given configuration.
    ///
    /// This creates the schedulers but does not start them.
    /// Call `start()` to begin background operations.
    pub async fn new(config: RuntimeConfig) -> RookResult<Self> {
        debug!(
            consolidation_enabled = config.enable_consolidation,
            intentions_enabled = config.enable_intentions,
            consolidation_interval = config.consolidation_interval_minutes,
            "Creating BackgroundRuntime"
        );

        // Create cognitive store
        let cognitive_store = match &config.cognitive_db_path {
            Some(path) => {
                debug!(path = %path, "Creating file-backed cognitive store");
                Arc::new(CognitiveStore::new(path)?)
            }
            None => {
                debug!("Creating in-memory cognitive store");
                Arc::new(CognitiveStore::in_memory()?)
            }
        };

        // Create intention store
        let intention_store: Arc<dyn IntentionStore> = match &config.intention_db_path {
            Some(path) => {
                debug!(path = %path, "Creating file-backed intention store");
                Arc::new(SqliteIntentionStore::new(path)?)
            }
            None => {
                debug!("Creating in-memory intention store");
                Arc::new(SqliteIntentionStore::in_memory()?)
            }
        };

        // Create consolidation scheduler if enabled
        let consolidation_scheduler = if config.enable_consolidation {
            let manager = Arc::new(ConsolidationManager::with_defaults(cognitive_store.clone()));
            let scheduler_config = SchedulerConfig {
                interval_minutes: config.consolidation_interval_minutes,
                run_on_start: config.consolidation_run_on_start,
            };
            let scheduler = ConsolidationScheduler::new(manager, scheduler_config)
                .await
                .map_err(|e| RookError::internal(format!("Failed to create consolidation scheduler: {}", e)))?;
            Some(scheduler)
        } else {
            None
        };

        // Create intention scheduler if enabled
        let (intention_scheduler, fired_intentions_rx) = if config.enable_intentions {
            let (scheduler, rx) = IntentionScheduler::new().await?;
            // Load existing time-based intentions from store
            let count = scheduler.load_from_store(intention_store.as_ref()).await?;
            if count > 0 {
                debug!(count, "Loaded time-based intentions from store");
            }
            (Some(scheduler), Some(rx))
        } else {
            (None, None)
        };

        Ok(Self {
            consolidation_scheduler,
            intention_scheduler,
            fired_intentions_rx,
            cognitive_store,
            intention_store,
            config,
        })
    }

    /// Start the background schedulers.
    ///
    /// Begins periodic execution of:
    /// - Consolidation (if enabled): runs every `consolidation_interval_minutes`
    /// - Intentions (if enabled): processes time-based triggers
    pub async fn start(&self) -> RookResult<()> {
        debug!("Starting background schedulers");

        // Start consolidation scheduler
        if let Some(ref scheduler) = self.consolidation_scheduler {
            scheduler.start().await.map_err(|e| {
                RookError::internal(format!("Failed to start consolidation scheduler: {}", e))
            })?;
            info!(
                interval_minutes = self.config.consolidation_interval_minutes,
                "Consolidation scheduler started"
            );
        }

        // Start intention scheduler
        if let Some(ref scheduler) = self.intention_scheduler {
            scheduler.start().await?;
            info!("Intention scheduler started");
        }

        info!("Background schedulers started");
        Ok(())
    }

    /// Shutdown the background schedulers gracefully.
    ///
    /// Stops all running schedulers and waits for them to complete.
    pub async fn shutdown(&mut self) -> RookResult<()> {
        debug!("Shutting down background schedulers");

        // Shutdown consolidation scheduler
        if let Some(ref mut scheduler) = self.consolidation_scheduler {
            scheduler.shutdown().await.map_err(|e| {
                RookError::internal(format!("Failed to shutdown consolidation scheduler: {}", e))
            })?;
            debug!("Consolidation scheduler stopped");
        }

        // Shutdown intention scheduler
        if let Some(ref mut scheduler) = self.intention_scheduler {
            scheduler.shutdown().await?;
            debug!("Intention scheduler stopped");
        }

        info!("Background schedulers stopped");
        Ok(())
    }

    /// Take the fired intentions receiver.
    ///
    /// Returns the mpsc receiver for consuming fired intentions.
    /// Can only be called once; subsequent calls return None.
    pub fn take_fired_intentions_rx(&mut self) -> Option<FiredIntentionReceiver> {
        self.fired_intentions_rx.take()
    }

    /// Get a reference to the cognitive store.
    pub fn cognitive_store(&self) -> Arc<CognitiveStore> {
        self.cognitive_store.clone()
    }

    /// Get a reference to the intention store.
    pub fn intention_store(&self) -> Arc<dyn IntentionStore> {
        self.intention_store.clone()
    }

    /// Get a reference to the consolidation scheduler.
    pub fn consolidation_scheduler(&self) -> Option<&ConsolidationScheduler> {
        self.consolidation_scheduler.as_ref()
    }

    /// Get a reference to the intention scheduler.
    pub fn intention_scheduler(&self) -> Option<&IntentionScheduler> {
        self.intention_scheduler.as_ref()
    }

    /// Get the runtime configuration.
    pub fn config(&self) -> &RuntimeConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_runtime_config_defaults() {
        let config = RuntimeConfig::default();
        assert_eq!(config.consolidation_interval_minutes, 15);
        assert!(!config.consolidation_run_on_start);
        assert!(config.enable_consolidation);
        assert!(config.enable_intentions);
        assert!(config.cognitive_db_path.is_none());
        assert!(config.intention_db_path.is_none());
    }

    #[test]
    fn test_runtime_config_builder() {
        let config = RuntimeConfig::default()
            .with_consolidation_interval(30)
            .with_run_on_start()
            .without_intentions();

        assert_eq!(config.consolidation_interval_minutes, 30);
        assert!(config.consolidation_run_on_start);
        assert!(config.enable_consolidation);
        assert!(!config.enable_intentions);
    }

    #[test]
    fn test_runtime_config_with_paths() {
        let config = RuntimeConfig::default()
            .with_cognitive_db_path("/tmp/cognitive.db")
            .with_intention_db_path("/tmp/intentions.db");

        assert_eq!(config.cognitive_db_path, Some("/tmp/cognitive.db".to_string()));
        assert_eq!(config.intention_db_path, Some("/tmp/intentions.db".to_string()));
    }

    #[test]
    fn test_runtime_config_interval_minimum() {
        let config = RuntimeConfig::default().with_consolidation_interval(0);
        assert_eq!(config.consolidation_interval_minutes, 1);
    }

    #[tokio::test]
    async fn test_runtime_creation_default() {
        let config = RuntimeConfig::default();
        let runtime = BackgroundRuntime::new(config).await.unwrap();

        assert!(runtime.consolidation_scheduler.is_some());
        assert!(runtime.intention_scheduler.is_some());
        assert!(runtime.fired_intentions_rx.is_some());
    }

    #[tokio::test]
    async fn test_runtime_creation_disabled_schedulers() {
        let config = RuntimeConfig::default()
            .without_consolidation()
            .without_intentions();
        let runtime = BackgroundRuntime::new(config).await.unwrap();

        assert!(runtime.consolidation_scheduler.is_none());
        assert!(runtime.intention_scheduler.is_none());
        assert!(runtime.fired_intentions_rx.is_none());
    }

    #[tokio::test]
    async fn test_runtime_start_and_shutdown() {
        let config = RuntimeConfig::default();
        let mut runtime = BackgroundRuntime::new(config).await.unwrap();

        // Start should succeed
        runtime.start().await.unwrap();

        // Shutdown should succeed
        runtime.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_runtime_take_fired_intentions_rx() {
        let config = RuntimeConfig::default();
        let mut runtime = BackgroundRuntime::new(config).await.unwrap();

        // First take should return Some
        let rx = runtime.take_fired_intentions_rx();
        assert!(rx.is_some());

        // Second take should return None
        let rx2 = runtime.take_fired_intentions_rx();
        assert!(rx2.is_none());
    }

    #[tokio::test]
    async fn test_runtime_store_access() {
        let config = RuntimeConfig::default();
        let runtime = BackgroundRuntime::new(config).await.unwrap();

        // Should be able to access stores
        let cognitive = runtime.cognitive_store();
        assert_eq!(cognitive.count().unwrap(), 0);

        let _intention = runtime.intention_store();
        // IntentionStore doesn't have a count method in the trait,
        // but we can verify it exists
    }
}
