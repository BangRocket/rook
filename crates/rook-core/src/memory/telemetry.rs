//! Telemetry for anonymous usage tracking.

use std::collections::HashMap;
use std::path::PathBuf;
use uuid::Uuid;

const PROJECT_API_KEY: &str = "phc_hgJkUVJFYtmaJqrvf6CYN67TIQ8yhXAkWzUn9AMU4yX";
const POSTHOG_HOST: &str = "https://us.i.posthog.com";

/// Telemetry client for anonymous usage tracking.
pub struct Telemetry {
    user_id: String,
    enabled: bool,
}

impl Telemetry {
    /// Create a new telemetry client.
    pub fn new(user_id: Option<String>) -> Self {
        let enabled = std::env::var("ROOK_TELEMETRY")
            .map(|v| !matches!(v.to_lowercase().as_str(), "false" | "0" | "no"))
            .unwrap_or(true);

        let user_id =
            user_id.unwrap_or_else(|| get_or_create_user_id().unwrap_or_else(|_| "anonymous".to_string()));

        Self { user_id, enabled }
    }

    /// Check if telemetry is enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Get the user ID.
    pub fn user_id(&self) -> &str {
        &self.user_id
    }

    /// Capture an event (async, fire-and-forget).
    pub async fn capture_event(
        &self,
        event_name: &str,
        mut properties: HashMap<String, serde_json::Value>,
    ) {
        if !self.enabled {
            return;
        }

        properties.insert(
            "client_source".to_string(),
            serde_json::Value::String("rust".to_string()),
        );
        properties.insert(
            "client_version".to_string(),
            serde_json::Value::String(env!("CARGO_PKG_VERSION").to_string()),
        );
        properties.insert(
            "os".to_string(),
            serde_json::Value::String(std::env::consts::OS.to_string()),
        );
        properties.insert(
            "arch".to_string(),
            serde_json::Value::String(std::env::consts::ARCH.to_string()),
        );

        let body = serde_json::json!({
            "api_key": PROJECT_API_KEY,
            "event": event_name,
            "distinct_id": self.user_id,
            "properties": properties,
        });

        // Fire and forget - don't block on this
        let _ = tokio::spawn(async move {
            let client = reqwest::Client::new();
            let _ = client
                .post(format!("{}/capture", POSTHOG_HOST))
                .json(&body)
                .send()
                .await;
        });
    }

    /// Capture an event synchronously (blocking).
    pub fn capture_event_sync(&self, event_name: &str, properties: HashMap<String, serde_json::Value>) {
        if !self.enabled {
            return;
        }

        // For sync context, we skip telemetry to avoid blocking
        // In production, you'd want to use a background thread
        let _ = (event_name, properties);
    }
}

impl Default for Telemetry {
    fn default() -> Self {
        Self::new(None)
    }
}

/// Get or create a persistent user ID.
fn get_or_create_user_id() -> std::io::Result<String> {
    let rook_dir = get_rook_dir()?;
    std::fs::create_dir_all(&rook_dir)?;

    let id_file = rook_dir.join("user_id");

    if id_file.exists() {
        std::fs::read_to_string(&id_file)
    } else {
        let id = Uuid::new_v4().to_string();
        std::fs::write(&id_file, &id)?;
        Ok(id)
    }
}

/// Get the rook directory.
fn get_rook_dir() -> std::io::Result<PathBuf> {
    dirs::home_dir()
        .map(|h| h.join(".rook"))
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::NotFound, "Home directory not found"))
}

/// Process telemetry filters to hash sensitive IDs.
pub fn process_telemetry_filters(
    filters: &HashMap<String, serde_json::Value>,
) -> (Vec<String>, HashMap<String, String>) {
    let mut encoded_ids = HashMap::new();
    let keys: Vec<String> = filters.keys().cloned().collect();

    for key in ["user_id", "agent_id", "run_id"] {
        if let Some(serde_json::Value::String(id)) = filters.get(key) {
            let hash = format!("{:x}", md5::compute(id.as_bytes()));
            encoded_ids.insert(key.to_string(), hash);
        }
    }

    (keys, encoded_ids)
}
