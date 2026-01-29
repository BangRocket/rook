//! JSON parsing utilities for LLM responses.

use regex::Regex;
use serde::Deserialize;

use crate::error::{ErrorCode, RookError, RookResult};
use crate::types::MemoryEvent;

/// Extract JSON from potentially wrapped response (code blocks, etc.).
pub fn extract_json(text: &str) -> RookResult<String> {
    let text = text.trim();

    // Try to extract from code block
    let code_block_re = Regex::new(r"```(?:json)?\s*([\s\S]*?)\s*```").unwrap();
    if let Some(captures) = code_block_re.captures(text) {
        if let Some(content) = captures.get(1) {
            return Ok(content.as_str().trim().to_string());
        }
    }

    // Return as-is if no code block
    Ok(text.to_string())
}

/// Remove code blocks and thinking tags from response.
pub fn remove_code_blocks(content: &str) -> String {
    let content = content.trim();

    // Remove ```[language] ... ``` blocks
    let code_re = Regex::new(r"^```[a-zA-Z0-9]*\n?([\s\S]*?)\n?```$").unwrap();
    let content = code_re
        .captures(content)
        .map(|c| c.get(1).map(|m| m.as_str().trim()).unwrap_or(content))
        .unwrap_or(content);

    // Remove <think>...</think> tags
    let think_re = Regex::new(r"<think>.*?</think>").unwrap();
    think_re.replace_all(content, "").trim().to_string()
}

/// Response from facts extraction.
#[derive(Debug, Deserialize)]
pub struct FactsResponse {
    pub facts: Vec<String>,
}

/// Parse facts from LLM response.
pub fn parse_facts(response: &str) -> RookResult<Vec<String>> {
    let cleaned = remove_code_blocks(response);
    if cleaned.is_empty() {
        return Ok(vec![]);
    }

    let json_str = extract_json(&cleaned)?;
    if json_str.is_empty() {
        return Ok(vec![]);
    }

    let parsed: FactsResponse = serde_json::from_str(&json_str).map_err(|e| RookError::Parse {
        message: format!("Failed to parse facts JSON: {}", e),
        code: ErrorCode::ParseInvalidJson,
    })?;

    Ok(parsed.facts)
}

/// Memory action from LLM update decision.
#[derive(Debug, Clone, Deserialize)]
pub struct MemoryAction {
    pub id: String,
    pub text: String,
    pub event: MemoryEvent,
    #[serde(default)]
    pub old_memory: Option<String>,
}

/// Response from memory update.
#[derive(Debug, Deserialize)]
pub struct MemoryUpdateResponse {
    pub memory: Vec<MemoryAction>,
}

/// Parse memory update actions from LLM response.
pub fn parse_memory_actions(response: &str) -> RookResult<Vec<MemoryAction>> {
    let cleaned = remove_code_blocks(response);
    if cleaned.is_empty() {
        return Ok(vec![]);
    }

    let json_str = extract_json(&cleaned)?;
    if json_str.is_empty() {
        return Ok(vec![]);
    }

    let parsed: MemoryUpdateResponse =
        serde_json::from_str(&json_str).map_err(|e| RookError::Parse {
            message: format!("Failed to parse memory actions JSON: {}", e),
            code: ErrorCode::ParseInvalidJson,
        })?;

    Ok(parsed.memory)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_json_from_code_block() {
        let input = r#"```json
{"facts": ["likes pizza", "lives in SF"]}
```"#;
        let result = extract_json(input).unwrap();
        assert!(result.contains("facts"));
    }

    #[test]
    fn test_parse_facts() {
        let input = r#"{"facts": ["likes pizza", "prefers remote work"]}"#;
        let facts = parse_facts(input).unwrap();
        assert_eq!(facts.len(), 2);
        assert_eq!(facts[0], "likes pizza");
    }

    #[test]
    fn test_parse_empty_facts() {
        let input = r#"{"facts": []}"#;
        let facts = parse_facts(input).unwrap();
        assert!(facts.is_empty());
    }

    #[test]
    fn test_remove_code_blocks() {
        let input = r#"```json
{"key": "value"}
```"#;
        let result = remove_code_blocks(input);
        assert_eq!(result, r#"{"key": "value"}"#);
    }
}
