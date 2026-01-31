//! Memory client implementation for the Rook hosted API.

use std::collections::HashMap;

use rook_core::error::{RookError, RookResult};
use rook_core::types::MemoryItem;

use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::json;

/// Client for the Rook hosted API.
pub struct MemoryClient {
    client: Client,
    api_key: String,
    base_url: String,
    org_id: Option<String>,
    project_id: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct AddResponse {
    results: Vec<AddResultItem>,
}

#[derive(Debug, Serialize, Deserialize)]
struct AddResultItem {
    id: String,
    memory: String,
    event: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct SearchResponse {
    results: Vec<SearchResultItem>,
}

#[derive(Debug, Serialize, Deserialize)]
struct SearchResultItem {
    id: String,
    memory: String,
    score: f32,
    #[serde(default)]
    metadata: Option<HashMap<String, serde_json::Value>>,
    #[serde(default)]
    created_at: Option<String>,
    #[serde(default)]
    updated_at: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct GetResponse {
    results: Vec<MemoryItemResponse>,
}

#[derive(Debug, Serialize, Deserialize)]
struct MemoryItemResponse {
    id: String,
    memory: String,
    #[serde(default)]
    metadata: Option<HashMap<String, serde_json::Value>>,
    #[serde(default)]
    created_at: Option<String>,
    #[serde(default)]
    updated_at: Option<String>,
}

impl MemoryClient {
    /// Create a new memory client.
    pub fn new(api_key: &str) -> RookResult<Self> {
        Self::with_options(api_key, None, None, None)
    }

    /// Create a new memory client with options.
    pub fn with_options(
        api_key: &str,
        base_url: Option<&str>,
        org_id: Option<&str>,
        project_id: Option<&str>,
    ) -> RookResult<Self> {
        let client = Client::new();
        let base_url = base_url
            .map(|s| s.to_string())
            .unwrap_or_else(|| "https://api.mem0.ai/v1".to_string());

        Ok(Self {
            client,
            api_key: api_key.to_string(),
            base_url,
            org_id: org_id.map(|s| s.to_string()),
            project_id: project_id.map(|s| s.to_string()),
        })
    }

    /// Create a client from environment variables.
    pub fn from_env() -> RookResult<Self> {
        let api_key = std::env::var("ROOK_API_KEY")
            .map_err(|_| RookError::Configuration("ROOK_API_KEY not set".to_string()))?;

        let base_url = std::env::var("ROOK_BASE_URL").ok();
        let org_id = std::env::var("ROOK_ORG_ID").ok();
        let project_id = std::env::var("ROOK_PROJECT_ID").ok();

        Self::with_options(
            &api_key,
            base_url.as_deref(),
            org_id.as_deref(),
            project_id.as_deref(),
        )
    }

    fn headers(&self) -> reqwest::header::HeaderMap {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            "Authorization",
            format!("Token {}", self.api_key).parse().unwrap(),
        );
        headers.insert(
            reqwest::header::CONTENT_TYPE,
            "application/json".parse().unwrap(),
        );
        if let Some(ref org_id) = self.org_id {
            headers.insert("Mem0-Org-Id", org_id.parse().unwrap());
        }
        if let Some(ref project_id) = self.project_id {
            headers.insert("Mem0-Project-Id", project_id.parse().unwrap());
        }
        headers
    }

    /// Add a memory.
    pub async fn add(
        &self,
        messages: &str,
        user_id: Option<&str>,
        agent_id: Option<&str>,
        run_id: Option<&str>,
        metadata: Option<HashMap<String, serde_json::Value>>,
    ) -> RookResult<Vec<MemoryItem>> {
        let mut body = json!({
            "messages": [{"role": "user", "content": messages}]
        });

        if let Some(uid) = user_id {
            body["user_id"] = json!(uid);
        }
        if let Some(aid) = agent_id {
            body["agent_id"] = json!(aid);
        }
        if let Some(rid) = run_id {
            body["run_id"] = json!(rid);
        }
        if let Some(meta) = metadata {
            body["metadata"] = json!(meta);
        }

        let response = self
            .client
            .post(format!("{}/memories/", self.base_url))
            .headers(self.headers())
            .json(&body)
            .send()
            .await
            .map_err(|e| RookError::api(format!("Failed to add memory: {}", e)))?;

        if !response.status().is_success() {
            let error = response.text().await.unwrap_or_default();
            return Err(RookError::api(format!("Failed to add memory: {}", error)));
        }

        let result: AddResponse = response
            .json()
            .await
            .map_err(|e| RookError::api(format!("Failed to parse response: {}", e)))?;

        Ok(result
            .results
            .into_iter()
            .map(|r| MemoryItem::new(r.id, r.memory))
            .collect())
    }

    /// Search memories.
    pub async fn search(
        &self,
        query: &str,
        user_id: Option<&str>,
        agent_id: Option<&str>,
        run_id: Option<&str>,
        limit: Option<usize>,
    ) -> RookResult<Vec<MemoryItem>> {
        let mut body = json!({ "query": query });

        if let Some(uid) = user_id {
            body["user_id"] = json!(uid);
        }
        if let Some(aid) = agent_id {
            body["agent_id"] = json!(aid);
        }
        if let Some(rid) = run_id {
            body["run_id"] = json!(rid);
        }
        if let Some(l) = limit {
            body["limit"] = json!(l);
        }

        let response = self
            .client
            .post(format!("{}/memories/search/", self.base_url))
            .headers(self.headers())
            .json(&body)
            .send()
            .await
            .map_err(|e| RookError::api(format!("Failed to search: {}", e)))?;

        if !response.status().is_success() {
            let error = response.text().await.unwrap_or_default();
            return Err(RookError::api(format!("Failed to search: {}", error)));
        }

        let result: SearchResponse = response
            .json()
            .await
            .map_err(|e| RookError::api(format!("Failed to parse response: {}", e)))?;

        Ok(result
            .results
            .into_iter()
            .map(|r| {
                let mut item = MemoryItem::new(r.id, r.memory).with_score(r.score);
                if let Some(meta) = r.metadata {
                    item = item.with_metadata(meta);
                }
                item
            })
            .collect())
    }

    /// Get a specific memory by ID.
    pub async fn get(&self, memory_id: &str) -> RookResult<Option<MemoryItem>> {
        let response = self
            .client
            .get(format!("{}/memories/{}/", self.base_url, memory_id))
            .headers(self.headers())
            .send()
            .await
            .map_err(|e| RookError::api(format!("Failed to get memory: {}", e)))?;

        if response.status().as_u16() == 404 {
            return Ok(None);
        }

        if !response.status().is_success() {
            let error = response.text().await.unwrap_or_default();
            return Err(RookError::api(format!("Failed to get memory: {}", error)));
        }

        let result: MemoryItemResponse = response
            .json()
            .await
            .map_err(|e| RookError::api(format!("Failed to parse response: {}", e)))?;

        let mut item = MemoryItem::new(result.id, result.memory);
        if let Some(meta) = result.metadata {
            item = item.with_metadata(meta);
        }
        item.created_at = result.created_at;
        item.updated_at = result.updated_at;

        Ok(Some(item))
    }

    /// Get all memories for a user/agent/run.
    pub async fn get_all(
        &self,
        user_id: Option<&str>,
        agent_id: Option<&str>,
        run_id: Option<&str>,
    ) -> RookResult<Vec<MemoryItem>> {
        let mut params = vec![];

        if let Some(uid) = user_id {
            params.push(format!("user_id={}", uid));
        }
        if let Some(aid) = agent_id {
            params.push(format!("agent_id={}", aid));
        }
        if let Some(rid) = run_id {
            params.push(format!("run_id={}", rid));
        }

        let query_string = if params.is_empty() {
            String::new()
        } else {
            format!("?{}", params.join("&"))
        };

        let response = self
            .client
            .get(format!("{}/memories/{}", self.base_url, query_string))
            .headers(self.headers())
            .send()
            .await
            .map_err(|e| RookError::api(format!("Failed to get memories: {}", e)))?;

        if !response.status().is_success() {
            let error = response.text().await.unwrap_or_default();
            return Err(RookError::api(format!("Failed to get memories: {}", error)));
        }

        let result: GetResponse = response
            .json()
            .await
            .map_err(|e| RookError::api(format!("Failed to parse response: {}", e)))?;

        Ok(result
            .results
            .into_iter()
            .map(|r| {
                let mut item = MemoryItem::new(r.id, r.memory);
                if let Some(meta) = r.metadata {
                    item = item.with_metadata(meta);
                }
                item.created_at = r.created_at;
                item.updated_at = r.updated_at;
                item
            })
            .collect())
    }

    /// Update a memory.
    pub async fn update(&self, memory_id: &str, text: &str) -> RookResult<MemoryItem> {
        let body = json!({ "text": text });

        let response = self
            .client
            .put(format!("{}/memories/{}/", self.base_url, memory_id))
            .headers(self.headers())
            .json(&body)
            .send()
            .await
            .map_err(|e| RookError::api(format!("Failed to update memory: {}", e)))?;

        if !response.status().is_success() {
            let error = response.text().await.unwrap_or_default();
            return Err(RookError::api(format!(
                "Failed to update memory: {}",
                error
            )));
        }

        let result: MemoryItemResponse = response
            .json()
            .await
            .map_err(|e| RookError::api(format!("Failed to parse response: {}", e)))?;

        let mut item = MemoryItem::new(result.id, result.memory);
        if let Some(meta) = result.metadata {
            item = item.with_metadata(meta);
        }
        item.created_at = result.created_at;
        item.updated_at = result.updated_at;

        Ok(item)
    }

    /// Delete a memory.
    pub async fn delete(&self, memory_id: &str) -> RookResult<()> {
        let response = self
            .client
            .delete(format!("{}/memories/{}/", self.base_url, memory_id))
            .headers(self.headers())
            .send()
            .await
            .map_err(|e| RookError::api(format!("Failed to delete memory: {}", e)))?;

        if !response.status().is_success() {
            let error = response.text().await.unwrap_or_default();
            return Err(RookError::api(format!(
                "Failed to delete memory: {}",
                error
            )));
        }

        Ok(())
    }

    /// Delete all memories for a user/agent/run.
    pub async fn delete_all(
        &self,
        user_id: Option<&str>,
        agent_id: Option<&str>,
        run_id: Option<&str>,
    ) -> RookResult<()> {
        let mut body = json!({});

        if let Some(uid) = user_id {
            body["user_id"] = json!(uid);
        }
        if let Some(aid) = agent_id {
            body["agent_id"] = json!(aid);
        }
        if let Some(rid) = run_id {
            body["run_id"] = json!(rid);
        }

        let response = self
            .client
            .delete(format!("{}/memories/", self.base_url))
            .headers(self.headers())
            .json(&body)
            .send()
            .await
            .map_err(|e| RookError::api(format!("Failed to delete memories: {}", e)))?;

        if !response.status().is_success() {
            let error = response.text().await.unwrap_or_default();
            return Err(RookError::api(format!(
                "Failed to delete memories: {}",
                error
            )));
        }

        Ok(())
    }

    /// Get memory history.
    pub async fn history(&self, memory_id: &str) -> RookResult<Vec<serde_json::Value>> {
        let response = self
            .client
            .get(format!("{}/memories/{}/history/", self.base_url, memory_id))
            .headers(self.headers())
            .send()
            .await
            .map_err(|e| RookError::api(format!("Failed to get history: {}", e)))?;

        if !response.status().is_success() {
            let error = response.text().await.unwrap_or_default();
            return Err(RookError::api(format!(
                "Failed to get history: {}",
                error
            )));
        }

        let result: Vec<serde_json::Value> = response
            .json()
            .await
            .map_err(|e| RookError::api(format!("Failed to parse response: {}", e)))?;

        Ok(result)
    }

    /// Reset all memories.
    pub async fn reset(&self) -> RookResult<()> {
        let response = self
            .client
            .post(format!("{}/memories/reset/", self.base_url))
            .headers(self.headers())
            .send()
            .await
            .map_err(|e| RookError::api(format!("Failed to reset: {}", e)))?;

        if !response.status().is_success() {
            let error = response.text().await.unwrap_or_default();
            return Err(RookError::api(format!("Failed to reset: {}", error)));
        }

        Ok(())
    }

    /// Send strength signals for memory feedback.
    ///
    /// Signals indicate how memories were used (or not) and trigger FSRS updates:
    /// - `used_in_response`: Memory was used → Good grade
    /// - `user_confirmation`: User confirmed accuracy → Easy grade
    /// - `user_correction`: User corrected info → Again grade (demote)
    /// - `retrieved_not_used`: Retrieved but not used → Neutral
    /// - `marked_incorrect`: User marked as wrong → Again grade
    /// - `marked_important`: User marked as key → Key memory flag
    pub async fn send_signals(&self, signals: Vec<SignalInput>) -> RookResult<SignalsResponse> {
        let body = json!({ "signals": signals });

        let response = self
            .client
            .post(format!("{}/signals/", self.base_url))
            .headers(self.headers())
            .json(&body)
            .send()
            .await
            .map_err(|e| RookError::api(format!("Failed to send signals: {}", e)))?;

        if !response.status().is_success() {
            let error = response.text().await.unwrap_or_default();
            return Err(RookError::api(format!(
                "Failed to send signals: {}",
                error
            )));
        }

        let result: SignalsResponse = response
            .json()
            .await
            .map_err(|e| RookError::api(format!("Failed to parse response: {}", e)))?;

        Ok(result)
    }

    /// Send a single strength signal.
    pub async fn send_signal(&self, signal: SignalInput) -> RookResult<SignalsResponse> {
        self.send_signals(vec![signal]).await
    }
}

/// Input for a strength signal.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SignalInput {
    /// Memory was used in generating a response.
    UsedInResponse {
        memory_id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        context: Option<String>,
    },
    /// User explicitly corrected information.
    UserCorrection {
        old_memory_id: String,
        new_content: String,
    },
    /// User confirmed information is correct.
    UserConfirmation { memory_id: String },
    /// Contradiction resolved between two memories.
    Contradiction { winner_id: String, loser_id: String },
    /// Memory was retrieved but not used.
    RetrievedNotUsed { memory_id: String },
    /// User explicitly marked memory as incorrect.
    MarkedIncorrect {
        memory_id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        reason: Option<String>,
    },
    /// User explicitly marked memory as important.
    MarkedImportant { memory_id: String },
}

/// Response from sending signals.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalsResponse {
    /// Number of signals processed.
    pub processed: usize,
    /// Pending grade updates.
    pub pending_updates: Vec<PendingUpdate>,
    /// Message.
    pub message: String,
}

/// A pending grade update.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingUpdate {
    pub memory_id: String,
    pub grade: String,
}
