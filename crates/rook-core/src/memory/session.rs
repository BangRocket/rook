//! Session management for memory operations.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::error::{ErrorCode, RookError, RookResult};
use crate::types::{Message, MessageRole};

/// Session scope for memory operations.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SessionScope {
    /// User identifier.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_id: Option<String>,
    /// Agent identifier.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub agent_id: Option<String>,
    /// Run/conversation identifier.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub run_id: Option<String>,
}

impl SessionScope {
    /// Create a new session scope.
    pub fn new(
        user_id: Option<String>,
        agent_id: Option<String>,
        run_id: Option<String>,
    ) -> Self {
        Self {
            user_id,
            agent_id,
            run_id,
        }
    }

    /// Create a user-scoped session.
    pub fn user(user_id: impl Into<String>) -> Self {
        Self {
            user_id: Some(user_id.into()),
            agent_id: None,
            run_id: None,
        }
    }

    /// Create an agent-scoped session.
    pub fn agent(agent_id: impl Into<String>) -> Self {
        Self {
            user_id: None,
            agent_id: Some(agent_id.into()),
            run_id: None,
        }
    }

    /// Create a run-scoped session.
    pub fn run(run_id: impl Into<String>) -> Self {
        Self {
            user_id: None,
            agent_id: None,
            run_id: Some(run_id.into()),
        }
    }

    /// Check if at least one identifier is present.
    pub fn is_valid(&self) -> bool {
        self.user_id.is_some() || self.agent_id.is_some() || self.run_id.is_some()
    }

    /// Validate that at least one identifier is present.
    pub fn validate(&self) -> RookResult<()> {
        if !self.is_valid() {
            return Err(RookError::Validation {
                message: "At least one of 'user_id', 'agent_id', or 'run_id' must be specified"
                    .to_string(),
                code: ErrorCode::ValMissingField,
                details: HashMap::new(),
                suggestion: Some("Provide user_id, agent_id, or run_id".to_string()),
            });
        }
        Ok(())
    }

    /// Convert to filter map.
    pub fn to_filters(&self) -> HashMap<String, serde_json::Value> {
        let mut filters = HashMap::new();
        if let Some(ref id) = self.user_id {
            filters.insert("user_id".to_string(), serde_json::Value::String(id.clone()));
        }
        if let Some(ref id) = self.agent_id {
            filters.insert(
                "agent_id".to_string(),
                serde_json::Value::String(id.clone()),
            );
        }
        if let Some(ref id) = self.run_id {
            filters.insert("run_id".to_string(), serde_json::Value::String(id.clone()));
        }
        filters
    }

    /// Convert to metadata map with additional input metadata.
    pub fn to_metadata(
        &self,
        input_metadata: Option<HashMap<String, serde_json::Value>>,
    ) -> HashMap<String, serde_json::Value> {
        let mut metadata = input_metadata.unwrap_or_default();
        if let Some(ref id) = self.user_id {
            metadata.insert("user_id".to_string(), serde_json::Value::String(id.clone()));
        }
        if let Some(ref id) = self.agent_id {
            metadata.insert(
                "agent_id".to_string(),
                serde_json::Value::String(id.clone()),
            );
        }
        if let Some(ref id) = self.run_id {
            metadata.insert("run_id".to_string(), serde_json::Value::String(id.clone()));
        }
        metadata
    }

    /// Check if agent memory extraction should be used.
    pub fn should_use_agent_extraction(&self, messages: &[Message]) -> bool {
        // Use agent extraction if agent_id is present and messages contain assistant responses
        if self.agent_id.is_some() {
            return messages
                .iter()
                .any(|m| matches!(m.role, MessageRole::Assistant));
        }
        false
    }
}

/// Build filters and metadata from session scope and input.
pub fn build_filters_and_metadata(
    user_id: Option<String>,
    agent_id: Option<String>,
    run_id: Option<String>,
    input_metadata: Option<HashMap<String, serde_json::Value>>,
    input_filters: Option<HashMap<String, serde_json::Value>>,
) -> (
    HashMap<String, serde_json::Value>,
    HashMap<String, serde_json::Value>,
) {
    let scope = SessionScope::new(user_id, agent_id, run_id);

    let metadata = scope.to_metadata(input_metadata);

    let mut filters = scope.to_filters();
    if let Some(additional) = input_filters {
        filters.extend(additional);
    }

    (metadata, filters)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_session_scope_validation() {
        let empty = SessionScope::default();
        assert!(empty.validate().is_err());

        let valid = SessionScope::user("test");
        assert!(valid.validate().is_ok());
    }

    #[test]
    fn test_session_scope_to_filters() {
        let scope = SessionScope::new(
            Some("user1".to_string()),
            Some("agent1".to_string()),
            None,
        );
        let filters = scope.to_filters();
        assert_eq!(filters.len(), 2);
        assert!(filters.contains_key("user_id"));
        assert!(filters.contains_key("agent_id"));
    }
}
