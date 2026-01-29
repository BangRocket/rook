//! Periodic scheduler for memory consolidation.
//!
//! Uses tokio-cron-scheduler to run consolidate() at regular intervals.
//! The scheduler runs consolidation as a background task, processing memories
//! through their consolidation phases.

use std::sync::Arc;

use tokio_cron_scheduler::{Job, JobScheduler, JobSchedulerError};
use tracing::{debug, error, info};

use super::manager::ConsolidationManager;

/// Configuration for the consolidation scheduler.
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    /// Interval between consolidation runs in minutes (default: 15)
    pub interval_minutes: u64,
    /// Whether to run consolidation immediately on start (default: false)
    pub run_on_start: bool,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            interval_minutes: 15,
            run_on_start: false,
        }
    }
}

impl SchedulerConfig {
    /// Create config with custom interval.
    pub fn with_interval(interval_minutes: u64) -> Self {
        Self {
            interval_minutes: interval_minutes.max(1), // Minimum 1 minute
            ..Default::default()
        }
    }

    /// Enable running consolidation immediately on start.
    pub fn with_run_on_start(mut self) -> Self {
        self.run_on_start = true;
        self
    }
}

/// Scheduler for periodic consolidation operations.
///
/// Wraps tokio-cron-scheduler to run the ConsolidationManager's consolidate()
/// operation at regular intervals.
///
/// # Example
///
/// ```ignore
/// use rook_core::consolidation::{
///     ConsolidationManager, ConsolidationScheduler, SchedulerConfig,
/// };
/// use rook_core::cognitive::CognitiveStore;
/// use std::sync::Arc;
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let store = Arc::new(CognitiveStore::in_memory()?);
/// let manager = Arc::new(ConsolidationManager::with_defaults(store));
/// let config = SchedulerConfig::with_interval(5); // Every 5 minutes
///
/// let scheduler = ConsolidationScheduler::new(manager, config).await?;
/// scheduler.start().await?;
/// # Ok(())
/// # }
/// ```
pub struct ConsolidationScheduler {
    scheduler: JobScheduler,
    manager: Arc<ConsolidationManager>,
    config: SchedulerConfig,
}

impl ConsolidationScheduler {
    /// Create a new ConsolidationScheduler.
    ///
    /// Note: Call `start()` to begin periodic execution.
    pub async fn new(
        manager: Arc<ConsolidationManager>,
        config: SchedulerConfig,
    ) -> Result<Self, JobSchedulerError> {
        let scheduler = JobScheduler::new().await?;

        Ok(Self {
            scheduler,
            manager,
            config,
        })
    }

    /// Create a scheduler with default configuration (15 minute interval).
    pub async fn with_defaults(
        manager: Arc<ConsolidationManager>,
    ) -> Result<Self, JobSchedulerError> {
        Self::new(manager, SchedulerConfig::default()).await
    }

    /// Get the scheduler configuration.
    pub fn config(&self) -> &SchedulerConfig {
        &self.config
    }

    /// Start the scheduler.
    ///
    /// This begins periodic execution of consolidate() at the configured interval.
    pub async fn start(&self) -> Result<(), JobSchedulerError> {
        let manager = self.manager.clone();
        let interval_secs = self.config.interval_minutes * 60;

        // Create the periodic job
        let job = Job::new_repeated_async(
            std::time::Duration::from_secs(interval_secs),
            move |_uuid, _lock| {
                let manager = manager.clone();
                Box::pin(async move {
                    debug!("Starting periodic consolidation");
                    match manager.consolidate() {
                        Ok(result) => {
                            info!(
                                consolidated = result.consolidated,
                                unconsolidated = result.unconsolidated,
                                advanced = result.advanced,
                                skipped = result.skipped,
                                duration_ms = result.duration_ms().unwrap_or(0),
                                "Consolidation complete"
                            );
                        }
                        Err(e) => {
                            error!(error = %e, "Consolidation failed");
                        }
                    }
                })
            },
        )?;

        self.scheduler.add(job).await?;

        // Optionally run immediately
        if self.config.run_on_start {
            debug!("Running initial consolidation on start");
            if let Err(e) = self.manager.consolidate() {
                error!(error = %e, "Initial consolidation failed");
            }
        }

        self.scheduler.start().await?;

        info!(
            interval_minutes = self.config.interval_minutes,
            "Consolidation scheduler started"
        );

        Ok(())
    }

    /// Stop the scheduler gracefully.
    pub async fn shutdown(&mut self) -> Result<(), JobSchedulerError> {
        info!("Shutting down consolidation scheduler");
        self.scheduler.shutdown().await
    }

    /// Run consolidation manually (outside of scheduled interval).
    ///
    /// This is useful for triggering consolidation on-demand, such as
    /// during application shutdown or when a large batch of memories
    /// has been added.
    pub fn run_now(
        &self,
    ) -> Result<super::manager::ConsolidationResult, crate::error::RookError> {
        self.manager.consolidate()
    }

    /// Get the underlying manager.
    pub fn manager(&self) -> &Arc<ConsolidationManager> {
        &self.manager
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scheduler_config_defaults() {
        let config = SchedulerConfig::default();
        assert_eq!(config.interval_minutes, 15);
        assert!(!config.run_on_start);
    }

    #[test]
    fn test_scheduler_config_with_interval() {
        let config = SchedulerConfig::with_interval(5);
        assert_eq!(config.interval_minutes, 5);

        // Test minimum clamping
        let config_min = SchedulerConfig::with_interval(0);
        assert_eq!(config_min.interval_minutes, 1);
    }

    #[test]
    fn test_scheduler_config_with_run_on_start() {
        let config = SchedulerConfig::with_interval(10).with_run_on_start();
        assert_eq!(config.interval_minutes, 10);
        assert!(config.run_on_start);
    }

    // Note: Full async tests for scheduler would require tokio runtime
    // and are better suited for integration tests
}
