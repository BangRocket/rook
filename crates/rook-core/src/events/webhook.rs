//! Webhook delivery with retry and signature (INT-14)
//!
//! Delivers memory events to external webhook endpoints with:
//! - HMAC-SHA256 payload signing for verification
//! - Exponential backoff retry on transient failures
//! - Event type filtering

use crate::events::{EventBus, MemoryLifecycleEvent};
use backon::{ExponentialBuilder, Retryable};
use hmac::{Hmac, Mac};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use sha2::Sha256;
use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;

/// Error type for webhook delivery
#[derive(Debug, Clone)]
pub enum WebhookError {
    /// Transient error (5xx, network) - should retry
    Transient(String),
    /// Permanent error (4xx) - should not retry
    Permanent(String),
    /// Configuration error
    Config(String),
}

impl std::fmt::Display for WebhookError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Transient(msg) => write!(f, "Transient error: {}", msg),
            Self::Permanent(msg) => write!(f, "Permanent error: {}", msg),
            Self::Config(msg) => write!(f, "Config error: {}", msg),
        }
    }
}

impl std::error::Error for WebhookError {}

/// Retry policy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryPolicy {
    /// Maximum number of retry attempts
    pub max_retries: u32,
    /// Initial delay before first retry (milliseconds)
    pub initial_delay_ms: u64,
    /// Maximum delay between retries (milliseconds)
    pub max_delay_ms: u64,
    /// Multiplier for exponential backoff
    pub multiplier: f32,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: 5,
            initial_delay_ms: 100,
            max_delay_ms: 30_000,
            multiplier: 2.0_f32,
        }
    }
}

/// Webhook endpoint configuration (INT-14)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebhookConfig {
    /// Unique identifier for this webhook
    pub id: String,
    /// Webhook endpoint URL
    pub url: String,
    /// Secret for HMAC signing (optional but recommended)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub secret: Option<String>,
    /// Event types to deliver (e.g., ["memory.created", "memory.updated"])
    /// Empty means all events
    pub events: HashSet<String>,
    /// Retry policy
    #[serde(default)]
    pub retry_policy: RetryPolicy,
    /// Request timeout in seconds
    #[serde(default = "default_timeout")]
    pub timeout_secs: u64,
    /// Whether this webhook is enabled
    #[serde(default = "default_enabled")]
    pub enabled: bool,
}

fn default_timeout() -> u64 {
    30
}

fn default_enabled() -> bool {
    true
}

impl WebhookConfig {
    /// Create a new webhook config
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            url: url.into(),
            secret: None,
            events: HashSet::new(),
            retry_policy: RetryPolicy::default(),
            timeout_secs: 30,
            enabled: true,
        }
    }

    /// Builder: set secret for signing
    pub fn with_secret(mut self, secret: impl Into<String>) -> Self {
        self.secret = Some(secret.into());
        self
    }

    /// Builder: subscribe to specific events
    pub fn with_events(mut self, events: Vec<&str>) -> Self {
        self.events = events.into_iter().map(String::from).collect();
        self
    }

    /// Builder: set retry policy
    pub fn with_retry(mut self, policy: RetryPolicy) -> Self {
        self.retry_policy = policy;
        self
    }

    /// Check if this webhook should receive the given event type
    pub fn should_receive(&self, event_type: &str) -> bool {
        self.enabled && (self.events.is_empty() || self.events.contains(event_type))
    }
}

/// Webhook delivery service
pub struct WebhookDelivery {
    client: Client,
    config: WebhookConfig,
}

impl WebhookDelivery {
    /// Create a new webhook delivery service
    pub fn new(config: WebhookConfig) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(config.timeout_secs))
            .build()
            .expect("Failed to create HTTP client");

        Self { client, config }
    }

    /// Deliver an event to the webhook endpoint
    pub async fn deliver(&self, event: &MemoryLifecycleEvent) -> Result<(), WebhookError> {
        let event_type = event.event_type();

        // Check if this event type should be delivered
        if !self.config.should_receive(event_type) {
            return Ok(());
        }

        let payload = serde_json::to_string(event)
            .map_err(|e| WebhookError::Config(format!("Serialization error: {}", e)))?;

        let signature = self.sign_payload(&payload);

        // Create delivery closure for retry
        let deliver_once = || async {
            let response = self
                .client
                .post(&self.config.url)
                .header("Content-Type", "application/json")
                .header("X-Rook-Signature", &signature)
                .header("X-Rook-Event", event_type)
                .header("X-Rook-Delivery", uuid::Uuid::new_v4().to_string())
                .body(payload.clone())
                .send()
                .await
                .map_err(|e| WebhookError::Transient(format!("Network error: {}", e)))?;

            let status = response.status();
            if status.is_success() {
                Ok(())
            } else if status.is_server_error() {
                // 5xx: transient, should retry
                Err(WebhookError::Transient(format!("Server error: {}", status)))
            } else {
                // 4xx: permanent, don't retry
                let body = response.text().await.unwrap_or_default();
                Err(WebhookError::Permanent(format!(
                    "Client error {}: {}",
                    status, body
                )))
            }
        };

        // Apply retry policy
        let policy = &self.config.retry_policy;
        deliver_once
            .retry(
                ExponentialBuilder::default()
                    .with_max_times(policy.max_retries as usize)
                    .with_min_delay(Duration::from_millis(policy.initial_delay_ms))
                    .with_max_delay(Duration::from_millis(policy.max_delay_ms))
                    .with_factor(policy.multiplier),
            )
            .when(|e| matches!(e, WebhookError::Transient(_)))
            .notify(|err, dur| {
                tracing::warn!(
                    "Webhook delivery to {} failed, retrying in {:?}: {:?}",
                    self.config.url,
                    dur,
                    err
                );
            })
            .await
    }

    /// Sign payload with HMAC-SHA256
    fn sign_payload(&self, payload: &str) -> String {
        match &self.config.secret {
            Some(secret) => {
                let mut mac =
                    Hmac::<Sha256>::new_from_slice(secret.as_bytes()).expect("HMAC accepts any key length");
                mac.update(payload.as_bytes());
                let result = mac.finalize();
                format!("sha256={}", hex::encode(result.into_bytes()))
            }
            None => String::new(),
        }
    }

    /// Get the webhook config
    pub fn config(&self) -> &WebhookConfig {
        &self.config
    }
}

impl Clone for WebhookDelivery {
    fn clone(&self) -> Self {
        Self {
            client: self.client.clone(),
            config: self.config.clone(),
        }
    }
}

/// Manager for multiple webhook deliveries
///
/// Spawns background task to consume events and deliver to all webhooks
pub struct WebhookManager {
    webhooks: Arc<RwLock<Vec<WebhookDelivery>>>,
    event_bus: EventBus,
}

impl WebhookManager {
    /// Create a new webhook manager
    pub fn new(event_bus: EventBus) -> Self {
        Self {
            webhooks: Arc::new(RwLock::new(Vec::new())),
            event_bus,
        }
    }

    /// Add a webhook
    pub async fn add_webhook(&self, config: WebhookConfig) {
        let mut webhooks = self.webhooks.write().await;
        webhooks.push(WebhookDelivery::new(config));
    }

    /// Remove a webhook by ID
    pub async fn remove_webhook(&self, id: &str) {
        let mut webhooks = self.webhooks.write().await;
        webhooks.retain(|w| w.config().id != id);
    }

    /// Start the delivery background task
    ///
    /// Returns a handle that can be used to stop delivery
    pub fn start(&self) -> tokio::task::JoinHandle<()> {
        let webhooks = self.webhooks.clone();
        let mut subscriber = self.event_bus.subscribe();

        tokio::spawn(async move {
            while let Some(event) = subscriber.recv().await {
                let webhooks = webhooks.read().await;

                // Deliver to all webhooks in parallel
                let futures: Vec<_> = webhooks
                    .iter()
                    .map(|webhook| {
                        let webhook = webhook.clone();
                        let event = event.clone();
                        async move {
                            if let Err(e) = webhook.deliver(&event).await {
                                tracing::error!(
                                    "Webhook delivery to {} failed: {:?}",
                                    webhook.config().url,
                                    e
                                );
                            }
                        }
                    })
                    .collect();

                futures::future::join_all(futures).await;
            }
        })
    }

    /// List all configured webhooks
    pub async fn list_webhooks(&self) -> Vec<WebhookConfig> {
        let webhooks = self.webhooks.read().await;
        webhooks.iter().map(|w| w.config().clone()).collect()
    }
}

/// Verify a webhook signature
///
/// Used by webhook receivers to verify the payload was sent by Rook
pub fn verify_signature(payload: &str, secret: &str, signature: &str) -> bool {
    let expected = {
        let mut mac =
            Hmac::<Sha256>::new_from_slice(secret.as_bytes()).expect("HMAC accepts any key length");
        mac.update(payload.as_bytes());
        let result = mac.finalize();
        format!("sha256={}", hex::encode(result.into_bytes()))
    };

    // Constant-time comparison to prevent timing attacks
    constant_time_eq(expected.as_bytes(), signature.as_bytes())
}

/// Constant-time equality comparison
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }

    let mut result = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        result |= x ^ y;
    }
    result == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_webhook_config_events_filter() {
        let config = WebhookConfig::new("https://example.com/webhook")
            .with_events(vec!["memory.created", "memory.updated"]);

        assert!(config.should_receive("memory.created"));
        assert!(config.should_receive("memory.updated"));
        assert!(!config.should_receive("memory.deleted"));
    }

    #[test]
    fn test_webhook_config_empty_events_receives_all() {
        let config = WebhookConfig::new("https://example.com/webhook");

        assert!(config.should_receive("memory.created"));
        assert!(config.should_receive("memory.updated"));
        assert!(config.should_receive("memory.deleted"));
        assert!(config.should_receive("memory.accessed"));
    }

    #[test]
    fn test_signature_verification() {
        let secret = "my-secret-key";
        let payload = r#"{"type":"memory.created","memory_id":"123"}"#;

        let delivery =
            WebhookDelivery::new(WebhookConfig::new("https://example.com").with_secret(secret));
        let signature = delivery.sign_payload(payload);

        assert!(verify_signature(payload, secret, &signature));
        assert!(!verify_signature(payload, "wrong-secret", &signature));
        assert!(!verify_signature("tampered", secret, &signature));
    }

    #[test]
    fn test_retry_policy_default() {
        let policy = RetryPolicy::default();
        assert_eq!(policy.max_retries, 5);
        assert_eq!(policy.initial_delay_ms, 100);
        assert_eq!(policy.max_delay_ms, 30_000);
    }
}
