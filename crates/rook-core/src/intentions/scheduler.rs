//! Scheduler for time-based intention triggers (INT-04, INT-05).
//!
//! Uses tokio-cron-scheduler for scheduled and elapsed-time triggers.

use crate::error::{RookError, RookResult};
use crate::intentions::{
    store::IntentionStore,
    triggers::{ActionResult, FiredIntention, TriggerReason},
    types::{Intention, TriggerCondition},
};
use chrono::Utc;
use std::collections::HashMap;
use std::time::Duration;
use tokio::sync::{mpsc, RwLock};
use tokio_cron_scheduler::{Job, JobScheduler};
use uuid::Uuid;

/// Channel for receiving fired intentions from scheduled jobs.
pub type FiredIntentionReceiver = mpsc::Receiver<FiredIntention>;

/// Scheduler for time-based intention triggers.
pub struct IntentionScheduler {
    /// The job scheduler.
    scheduler: JobScheduler,
    /// Map of intention ID to job UUID.
    job_map: RwLock<HashMap<Uuid, uuid::Uuid>>,
    /// Channel for sending fired intentions.
    fire_sender: mpsc::Sender<FiredIntention>,
    /// Whether scheduler is running.
    running: RwLock<bool>,
}

impl IntentionScheduler {
    /// Create a new scheduler.
    ///
    /// Returns the scheduler and a receiver for fired intentions.
    pub async fn new() -> RookResult<(Self, FiredIntentionReceiver)> {
        let scheduler = JobScheduler::new()
            .await
            .map_err(|e| RookError::internal(format!("Failed to create scheduler: {}", e)))?;

        let (tx, rx) = mpsc::channel(100);

        Ok((
            Self {
                scheduler,
                job_map: RwLock::new(HashMap::new()),
                fire_sender: tx,
                running: RwLock::new(false),
            },
            rx,
        ))
    }

    /// Start the scheduler.
    pub async fn start(&self) -> RookResult<()> {
        let mut running = self.running.write().await;
        if !*running {
            self.scheduler
                .start()
                .await
                .map_err(|e| RookError::internal(format!("Failed to start scheduler: {}", e)))?;
            *running = true;
        }
        Ok(())
    }

    /// Stop the scheduler.
    pub async fn shutdown(&mut self) -> RookResult<()> {
        let mut running = self.running.write().await;
        if *running {
            self.scheduler
                .shutdown()
                .await
                .map_err(|e| RookError::internal(format!("Failed to shutdown scheduler: {}", e)))?;
            *running = false;
        }
        Ok(())
    }

    /// Check if scheduler is running.
    pub async fn is_running(&self) -> bool {
        *self.running.read().await
    }

    /// Schedule an intention based on its trigger condition.
    pub async fn schedule(&self, intention: &Intention) -> RookResult<()> {
        match &intention.trigger {
            TriggerCondition::TimeElapsed {
                duration_secs,
                recurring,
                ..
            } => {
                self.schedule_time_elapsed(intention.id, *duration_secs, *recurring)
                    .await
            }
            TriggerCondition::ScheduledTime {
                scheduled_at,
                cron,
                ..
            } => {
                if let Some(cron_expr) = cron {
                    self.schedule_cron(intention.id, cron_expr).await
                } else {
                    self.schedule_one_shot(intention.id, *scheduled_at).await
                }
            }
            // Keyword and Topic triggers are handled by IntentionChecker, not scheduler
            _ => Ok(()),
        }
    }

    /// Remove a scheduled intention.
    pub async fn unschedule(&self, intention_id: Uuid) -> RookResult<()> {
        let mut job_map = self.job_map.write().await;
        if let Some(job_id) = job_map.remove(&intention_id) {
            self.scheduler
                .remove(&job_id)
                .await
                .map_err(|e| RookError::internal(format!("Failed to remove job: {}", e)))?;
        }
        Ok(())
    }

    /// Get the number of scheduled jobs.
    pub async fn job_count(&self) -> usize {
        self.job_map.read().await.len()
    }

    /// Schedule a time-elapsed trigger.
    async fn schedule_time_elapsed(
        &self,
        intention_id: Uuid,
        duration_secs: u64,
        recurring: bool,
    ) -> RookResult<()> {
        let sender = self.fire_sender.clone();
        let duration = Duration::from_secs(duration_secs);

        let job = if recurring {
            Job::new_repeated_async(duration, move |_uuid, _lock| {
                let sender = sender.clone();
                Box::pin(async move {
                    let fired = FiredIntention::new(
                        intention_id,
                        TriggerReason::TimeElapsed {
                            elapsed_secs: duration_secs,
                        },
                        ActionResult::Success { details: None },
                    );
                    let _ = sender.send(fired).await;
                })
            })
            .map_err(|e| RookError::internal(format!("Failed to create repeated job: {}", e)))?
        } else {
            Job::new_one_shot_async(duration, move |_uuid, _lock| {
                let sender = sender.clone();
                Box::pin(async move {
                    let fired = FiredIntention::new(
                        intention_id,
                        TriggerReason::TimeElapsed {
                            elapsed_secs: duration_secs,
                        },
                        ActionResult::Success { details: None },
                    );
                    let _ = sender.send(fired).await;
                })
            })
            .map_err(|e| RookError::internal(format!("Failed to create one-shot job: {}", e)))?
        };

        let job_id = job.guid();
        self.scheduler
            .add(job)
            .await
            .map_err(|e| RookError::internal(format!("Failed to add job: {}", e)))?;

        self.job_map.write().await.insert(intention_id, job_id);
        Ok(())
    }

    /// Schedule a one-shot trigger at specific time.
    async fn schedule_one_shot(
        &self,
        intention_id: Uuid,
        scheduled_at: chrono::DateTime<Utc>,
    ) -> RookResult<()> {
        let now = Utc::now();
        if scheduled_at <= now {
            // Already past, fire immediately
            let fired = FiredIntention::new(
                intention_id,
                TriggerReason::ScheduledTime { scheduled_at },
                ActionResult::Success { details: None },
            );
            let _ = self.fire_sender.send(fired).await;
            return Ok(());
        }

        let duration = (scheduled_at - now)
            .to_std()
            .map_err(|e| RookError::internal(format!("Invalid duration: {}", e)))?;

        let sender = self.fire_sender.clone();
        let job = Job::new_one_shot_async(duration, move |_uuid, _lock| {
            let sender = sender.clone();
            Box::pin(async move {
                let fired = FiredIntention::new(
                    intention_id,
                    TriggerReason::ScheduledTime { scheduled_at },
                    ActionResult::Success { details: None },
                );
                let _ = sender.send(fired).await;
            })
        })
        .map_err(|e| RookError::internal(format!("Failed to create one-shot job: {}", e)))?;

        let job_id = job.guid();
        self.scheduler
            .add(job)
            .await
            .map_err(|e| RookError::internal(format!("Failed to add job: {}", e)))?;

        self.job_map.write().await.insert(intention_id, job_id);
        Ok(())
    }

    /// Schedule a cron-based recurring trigger.
    async fn schedule_cron(&self, intention_id: Uuid, cron_expr: &str) -> RookResult<()> {
        let sender = self.fire_sender.clone();

        let job = Job::new_async(cron_expr, move |_uuid, _lock| {
            let sender = sender.clone();
            Box::pin(async move {
                let fired = FiredIntention::new(
                    intention_id,
                    TriggerReason::ScheduledTime {
                        scheduled_at: Utc::now(),
                    },
                    ActionResult::Success { details: None },
                );
                let _ = sender.send(fired).await;
            })
        })
        .map_err(|e| RookError::internal(format!("Failed to create cron job: {}", e)))?;

        let job_id = job.guid();
        self.scheduler
            .add(job)
            .await
            .map_err(|e| RookError::internal(format!("Failed to add job: {}", e)))?;

        self.job_map.write().await.insert(intention_id, job_id);
        Ok(())
    }

    /// Load and schedule all time-based intentions from store.
    pub async fn load_from_store<S: IntentionStore + ?Sized>(&self, store: &S) -> RookResult<usize> {
        let mut count = 0;

        // Load time-elapsed intentions
        let elapsed = store.get_by_trigger_type("time_elapsed")?;
        for intention in elapsed {
            if intention.can_fire() {
                self.schedule(&intention).await?;
                count += 1;
            }
        }

        // Load scheduled-time intentions
        let scheduled = store.get_by_trigger_type("scheduled_time")?;
        for intention in scheduled {
            if intention.can_fire() {
                self.schedule(&intention).await?;
                count += 1;
            }
        }

        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration as ChronoDuration;

    #[tokio::test]
    async fn test_scheduler_creation() {
        let (scheduler, _rx) = IntentionScheduler::new().await.unwrap();
        assert!(!scheduler.is_running().await);
        assert_eq!(scheduler.job_count().await, 0);
    }

    #[tokio::test]
    async fn test_scheduler_start_stop() {
        let (mut scheduler, _rx) = IntentionScheduler::new().await.unwrap();

        scheduler.start().await.unwrap();
        assert!(scheduler.is_running().await);

        scheduler.shutdown().await.unwrap();
        assert!(!scheduler.is_running().await);
    }

    #[tokio::test]
    async fn test_schedule_time_elapsed() {
        let (scheduler, _rx) = IntentionScheduler::new().await.unwrap();

        let intention = crate::intentions::Intention::new(
            "test",
            TriggerCondition::TimeElapsed {
                duration_secs: 3600, // 1 hour
                recurring: false,
                reference_time: None,
            },
            crate::intentions::IntentionAction::default(),
        );

        scheduler.schedule(&intention).await.unwrap();
        assert_eq!(scheduler.job_count().await, 1);

        scheduler.unschedule(intention.id).await.unwrap();
        assert_eq!(scheduler.job_count().await, 0);
    }

    #[tokio::test]
    async fn test_schedule_past_time_fires_immediately() {
        let (scheduler, mut rx) = IntentionScheduler::new().await.unwrap();

        let past_time = Utc::now() - ChronoDuration::hours(1);
        let intention = crate::intentions::Intention::new(
            "test",
            TriggerCondition::ScheduledTime {
                scheduled_at: past_time,
                cron: None,
                timezone: None,
            },
            crate::intentions::IntentionAction::default(),
        );

        scheduler.schedule(&intention).await.unwrap();

        // Should receive fired intention immediately
        let fired = tokio::time::timeout(Duration::from_millis(100), rx.recv())
            .await
            .expect("Should receive within timeout")
            .expect("Should receive fired intention");

        assert_eq!(fired.intention_id, intention.id);
    }

    #[tokio::test]
    async fn test_schedule_keyword_intention_is_noop() {
        let (scheduler, _rx) = IntentionScheduler::new().await.unwrap();

        let intention = crate::intentions::Intention::new(
            "test",
            TriggerCondition::KeywordMention {
                keywords: vec!["test".to_string()],
                exact_match: false,
            },
            crate::intentions::IntentionAction::default(),
        );

        // Scheduling keyword intention should be a no-op
        scheduler.schedule(&intention).await.unwrap();
        assert_eq!(scheduler.job_count().await, 0);
    }
}
