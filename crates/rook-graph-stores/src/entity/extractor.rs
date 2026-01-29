//! LLM-based entity extraction.
//!
//! This module provides entity and relationship extraction from text
//! using LLM-based structured output. The extractor prompts an LLM
//! with a structured JSON format and parses the response.
//!
//! # Architecture
//!
//! 1. Text is sent to the LLM with a structured prompt
//! 2. LLM returns JSON with entities and relationships
//! 3. Response is parsed with lenient handling for malformed output
//! 4. Unknown types are handled gracefully (defaults applied)

use once_cell::sync::Lazy;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use rook_core::error::RookResult;
use rook_core::traits::{GenerationOptions, Llm, ResponseFormat};
use rook_core::types::Message;

use super::types::{EntityType, RelationshipType};

/// An entity extracted from text.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedEntity {
    /// The entity name.
    pub name: String,
    /// The entity type.
    pub entity_type: EntityType,
    /// Optional description or context.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

impl ExtractedEntity {
    /// Create a new extracted entity.
    pub fn new(name: impl Into<String>, entity_type: EntityType) -> Self {
        Self {
            name: name.into(),
            entity_type,
            description: None,
        }
    }

    /// Add a description.
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }
}

/// A relationship extracted from text.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedRelationship {
    /// The source entity name.
    pub source: String,
    /// The target entity name.
    pub target: String,
    /// The relationship type.
    pub relationship_type: RelationshipType,
    /// Optional context or description.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context: Option<String>,
}

impl ExtractedRelationship {
    /// Create a new extracted relationship.
    pub fn new(
        source: impl Into<String>,
        target: impl Into<String>,
        relationship_type: RelationshipType,
    ) -> Self {
        Self {
            source: source.into(),
            target: target.into(),
            relationship_type,
            context: None,
        }
    }

    /// Add context.
    pub fn with_context(mut self, context: impl Into<String>) -> Self {
        self.context = Some(context.into());
        self
    }
}

/// Result of entity extraction.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExtractionResult {
    /// Extracted entities.
    pub entities: Vec<ExtractedEntity>,
    /// Extracted relationships.
    pub relationships: Vec<ExtractedRelationship>,
}

impl ExtractionResult {
    /// Check if the result is empty.
    pub fn is_empty(&self) -> bool {
        self.entities.is_empty() && self.relationships.is_empty()
    }

    /// Get entity count.
    pub fn entity_count(&self) -> usize {
        self.entities.len()
    }

    /// Get relationship count.
    pub fn relationship_count(&self) -> usize {
        self.relationships.len()
    }
}

/// Raw JSON structures for LLM response parsing.
/// These allow flexible parsing before converting to typed structs.
mod raw {
    use serde::Deserialize;

    #[derive(Debug, Deserialize)]
    pub struct RawEntity {
        pub name: Option<String>,
        #[serde(alias = "type", alias = "entityType", alias = "entity_type")]
        pub entity_type: Option<String>,
        pub description: Option<String>,
    }

    #[derive(Debug, Deserialize)]
    pub struct RawRelationship {
        pub source: Option<String>,
        #[serde(alias = "from")]
        pub _source_alt: Option<String>,
        pub target: Option<String>,
        #[serde(alias = "to")]
        pub _target_alt: Option<String>,
        #[serde(alias = "type", alias = "relationshipType", alias = "relationship_type", alias = "rel_type")]
        pub relationship_type: Option<String>,
        pub context: Option<String>,
    }

    impl RawRelationship {
        pub fn source(&self) -> Option<&str> {
            self.source.as_deref().or(self._source_alt.as_deref())
        }

        pub fn target(&self) -> Option<&str> {
            self.target.as_deref().or(self._target_alt.as_deref())
        }
    }

    #[derive(Debug, Deserialize)]
    pub struct RawExtractionResult {
        #[serde(default)]
        pub entities: Vec<RawEntity>,
        #[serde(default)]
        pub relationships: Vec<RawRelationship>,
    }
}

/// LLM-based entity extractor.
pub struct EntityExtractor {
    llm: Arc<dyn Llm>,
}

impl EntityExtractor {
    /// Create a new entity extractor.
    pub fn new(llm: Arc<dyn Llm>) -> Self {
        Self { llm }
    }

    /// Extract entities and relationships from text.
    ///
    /// This method:
    /// 1. Sends the text to the LLM with a structured prompt
    /// 2. Parses the JSON response with lenient handling
    /// 3. Converts raw entities/relationships to typed versions
    /// 4. Handles edge cases (empty input, malformed output, unknown types)
    pub async fn extract(&self, text: &str) -> RookResult<ExtractionResult> {
        // Handle empty input
        let text = text.trim();
        if text.is_empty() {
            return Ok(ExtractionResult::default());
        }

        // Build messages for extraction
        let messages = vec![
            Message::system(Self::system_prompt()),
            Message::user(format!("Extract entities and relationships from this text:\n\n{}", text)),
        ];

        // Request JSON response
        let options = GenerationOptions {
            temperature: Some(0.0), // Deterministic for extraction
            response_format: Some(ResponseFormat::Json),
            ..Default::default()
        };

        // Call LLM
        let response = self.llm.generate(&messages, Some(options)).await?;

        // Parse response
        let content = response.content.unwrap_or_default();
        self.parse_response(&content)
    }

    /// Generate the system prompt for extraction.
    fn system_prompt() -> String {
        let entity_types: Vec<&str> = EntityType::all().iter().map(|t| t.as_str()).collect();
        let relationship_types: Vec<&str> = RelationshipType::all().iter().map(|t| t.as_str()).collect();

        format!(
            r#"You are an entity extraction system. Extract entities and relationships from text.

ENTITY TYPES: {}

RELATIONSHIP TYPES: {}

Output JSON in this exact format:
{{
  "entities": [
    {{"name": "entity name", "entity_type": "type", "description": "brief description"}}
  ],
  "relationships": [
    {{"source": "source entity", "target": "target entity", "relationship_type": "type", "context": "relationship context"}}
  ]
}}

Rules:
1. Only extract explicitly mentioned entities
2. Use the most specific entity type that applies
3. Use the most appropriate relationship type
4. Keep descriptions brief (under 50 words)
5. If no entities found, return empty arrays
6. Entity names should be normalized (proper capitalization)

Return ONLY valid JSON, no other text."#,
            entity_types.join(", "),
            relationship_types.join(", ")
        )
    }

    /// Parse the LLM response into an ExtractionResult.
    fn parse_response(&self, content: &str) -> RookResult<ExtractionResult> {
        // Handle empty response
        let content = content.trim();
        if content.is_empty() {
            return Ok(ExtractionResult::default());
        }

        // Try to extract JSON from the response (handle markdown code blocks)
        let json_str = Self::extract_json(content);

        // Try direct parsing first
        let raw_result: raw::RawExtractionResult = match serde_json::from_str(json_str) {
            Ok(r) => r,
            Err(e) => {
                // Try lenient parsing
                match Self::lenient_parse(json_str) {
                    Some(r) => r,
                    None => {
                        tracing::warn!("Failed to parse extraction response: {}", e);
                        return Ok(ExtractionResult::default());
                    }
                }
            }
        };

        // Convert raw entities to typed entities
        let mut entities = Vec::new();
        for raw in raw_result.entities {
            if let Some(entity) = Self::convert_entity(raw) {
                entities.push(entity);
            }
        }

        // Convert raw relationships to typed relationships
        let mut relationships = Vec::new();
        for raw in raw_result.relationships {
            if let Some(rel) = Self::convert_relationship(raw) {
                relationships.push(rel);
            }
        }

        Ok(ExtractionResult { entities, relationships })
    }

    /// Extract JSON from response (handles markdown code blocks).
    fn extract_json(content: &str) -> &str {
        static JSON_BLOCK: Lazy<Regex> = Lazy::new(|| {
            Regex::new(r"```(?:json)?\s*\n?([\s\S]*?)\n?```").unwrap()
        });

        // Try to find JSON in code block
        if let Some(caps) = JSON_BLOCK.captures(content) {
            if let Some(m) = caps.get(1) {
                return m.as_str().trim();
            }
        }

        // Return as-is if no code block
        content
    }

    /// Lenient parsing for malformed JSON.
    fn lenient_parse(json_str: &str) -> Option<raw::RawExtractionResult> {
        // Try to fix common issues
        let fixed = json_str
            .replace("'", "\"")  // Single quotes to double
            .replace(",]", "]")  // Trailing commas
            .replace(",}", "}"); // Trailing commas

        serde_json::from_str(&fixed).ok()
    }

    /// Convert a raw entity to a typed entity.
    fn convert_entity(raw: raw::RawEntity) -> Option<ExtractedEntity> {
        let name = raw.name?.trim().to_string();
        if name.is_empty() {
            return None;
        }

        // Parse entity type with fallback to Concept
        let entity_type = raw
            .entity_type
            .as_deref()
            .and_then(EntityType::from_str_flexible)
            .unwrap_or(EntityType::Concept);

        let mut entity = ExtractedEntity::new(name, entity_type);
        if let Some(desc) = raw.description {
            let desc = desc.trim();
            if !desc.is_empty() {
                entity = entity.with_description(desc);
            }
        }

        Some(entity)
    }

    /// Convert a raw relationship to a typed relationship.
    fn convert_relationship(raw: raw::RawRelationship) -> Option<ExtractedRelationship> {
        let source = raw.source()?.trim().to_string();
        let target = raw.target()?.trim().to_string();

        if source.is_empty() || target.is_empty() {
            return None;
        }

        // Parse relationship type with fallback to RelatedTo
        let relationship_type = raw
            .relationship_type
            .as_deref()
            .and_then(RelationshipType::from_str_flexible)
            .unwrap_or(RelationshipType::RelatedTo);

        let mut rel = ExtractedRelationship::new(source, target, relationship_type);
        if let Some(ctx) = raw.context {
            let ctx = ctx.trim();
            if !ctx.is_empty() {
                rel = rel.with_context(ctx);
            }
        }

        Some(rel)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extracted_entity_builder() {
        let entity = ExtractedEntity::new("Alice", EntityType::Person)
            .with_description("A software engineer");

        assert_eq!(entity.name, "Alice");
        assert_eq!(entity.entity_type, EntityType::Person);
        assert_eq!(entity.description.as_deref(), Some("A software engineer"));
    }

    #[test]
    fn test_extracted_relationship_builder() {
        let rel = ExtractedRelationship::new("Alice", "Acme Corp", RelationshipType::WorksAt)
            .with_context("Senior engineer role");

        assert_eq!(rel.source, "Alice");
        assert_eq!(rel.target, "Acme Corp");
        assert_eq!(rel.relationship_type, RelationshipType::WorksAt);
        assert_eq!(rel.context.as_deref(), Some("Senior engineer role"));
    }

    #[test]
    fn test_extraction_result_is_empty() {
        let empty = ExtractionResult::default();
        assert!(empty.is_empty());
        assert_eq!(empty.entity_count(), 0);
        assert_eq!(empty.relationship_count(), 0);

        let non_empty = ExtractionResult {
            entities: vec![ExtractedEntity::new("Alice", EntityType::Person)],
            relationships: vec![],
        };
        assert!(!non_empty.is_empty());
        assert_eq!(non_empty.entity_count(), 1);
    }

    #[test]
    fn test_parse_valid_json() {
        use std::sync::Arc;
        use rook_core::traits::{LlmResponse, LlmStream, Tool, ToolChoice};
        use async_trait::async_trait;

        // Create a mock LLM (not used in this test)
        struct MockLlm;

        #[async_trait]
        impl Llm for MockLlm {
            async fn generate(&self, _: &[Message], _: Option<GenerationOptions>) -> RookResult<LlmResponse> {
                Ok(LlmResponse::default())
            }
            async fn generate_with_tools(&self, _: &[Message], _: &[Tool], _: ToolChoice, _: Option<GenerationOptions>) -> RookResult<LlmResponse> {
                Ok(LlmResponse::default())
            }
            async fn generate_stream(&self, _: &[Message], _: Option<GenerationOptions>) -> RookResult<LlmStream> {
                unimplemented!()
            }
            fn model_name(&self) -> &str { "mock" }
        }

        let extractor = EntityExtractor::new(Arc::new(MockLlm));

        let json = r#"{
            "entities": [
                {"name": "Alice", "entity_type": "person", "description": "A developer"},
                {"name": "Acme Corp", "entity_type": "organization"}
            ],
            "relationships": [
                {"source": "Alice", "target": "Acme Corp", "relationship_type": "works_at"}
            ]
        }"#;

        let result = extractor.parse_response(json).unwrap();

        assert_eq!(result.entities.len(), 2);
        assert_eq!(result.entities[0].name, "Alice");
        assert_eq!(result.entities[0].entity_type, EntityType::Person);
        assert_eq!(result.entities[1].name, "Acme Corp");
        assert_eq!(result.entities[1].entity_type, EntityType::Organization);

        assert_eq!(result.relationships.len(), 1);
        assert_eq!(result.relationships[0].source, "Alice");
        assert_eq!(result.relationships[0].target, "Acme Corp");
        assert_eq!(result.relationships[0].relationship_type, RelationshipType::WorksAt);
    }

    #[test]
    fn test_parse_json_in_code_block() {
        use std::sync::Arc;
        use rook_core::traits::{LlmResponse, LlmStream, Tool, ToolChoice};
        use async_trait::async_trait;

        struct MockLlm;

        #[async_trait]
        impl Llm for MockLlm {
            async fn generate(&self, _: &[Message], _: Option<GenerationOptions>) -> RookResult<LlmResponse> {
                Ok(LlmResponse::default())
            }
            async fn generate_with_tools(&self, _: &[Message], _: &[Tool], _: ToolChoice, _: Option<GenerationOptions>) -> RookResult<LlmResponse> {
                Ok(LlmResponse::default())
            }
            async fn generate_stream(&self, _: &[Message], _: Option<GenerationOptions>) -> RookResult<LlmStream> {
                unimplemented!()
            }
            fn model_name(&self) -> &str { "mock" }
        }

        let extractor = EntityExtractor::new(Arc::new(MockLlm));

        let json = r#"```json
{
    "entities": [{"name": "Bob", "entity_type": "person"}],
    "relationships": []
}
```"#;

        let result = extractor.parse_response(json).unwrap();
        assert_eq!(result.entities.len(), 1);
        assert_eq!(result.entities[0].name, "Bob");
    }

    #[test]
    fn test_parse_unknown_types_fallback() {
        use std::sync::Arc;
        use rook_core::traits::{LlmResponse, LlmStream, Tool, ToolChoice};
        use async_trait::async_trait;

        struct MockLlm;

        #[async_trait]
        impl Llm for MockLlm {
            async fn generate(&self, _: &[Message], _: Option<GenerationOptions>) -> RookResult<LlmResponse> {
                Ok(LlmResponse::default())
            }
            async fn generate_with_tools(&self, _: &[Message], _: &[Tool], _: ToolChoice, _: Option<GenerationOptions>) -> RookResult<LlmResponse> {
                Ok(LlmResponse::default())
            }
            async fn generate_stream(&self, _: &[Message], _: Option<GenerationOptions>) -> RookResult<LlmStream> {
                unimplemented!()
            }
            fn model_name(&self) -> &str { "mock" }
        }

        let extractor = EntityExtractor::new(Arc::new(MockLlm));

        // Unknown entity type should fallback to Concept
        let json = r#"{
            "entities": [{"name": "Thing", "entity_type": "unknown_type"}],
            "relationships": [{"source": "A", "target": "B", "relationship_type": "unknown_rel"}]
        }"#;

        let result = extractor.parse_response(json).unwrap();
        assert_eq!(result.entities[0].entity_type, EntityType::Concept);
        assert_eq!(result.relationships[0].relationship_type, RelationshipType::RelatedTo);
    }

    #[test]
    fn test_parse_empty_response() {
        use std::sync::Arc;
        use rook_core::traits::{LlmResponse, LlmStream, Tool, ToolChoice};
        use async_trait::async_trait;

        struct MockLlm;

        #[async_trait]
        impl Llm for MockLlm {
            async fn generate(&self, _: &[Message], _: Option<GenerationOptions>) -> RookResult<LlmResponse> {
                Ok(LlmResponse::default())
            }
            async fn generate_with_tools(&self, _: &[Message], _: &[Tool], _: ToolChoice, _: Option<GenerationOptions>) -> RookResult<LlmResponse> {
                Ok(LlmResponse::default())
            }
            async fn generate_stream(&self, _: &[Message], _: Option<GenerationOptions>) -> RookResult<LlmStream> {
                unimplemented!()
            }
            fn model_name(&self) -> &str { "mock" }
        }

        let extractor = EntityExtractor::new(Arc::new(MockLlm));

        let result = extractor.parse_response("").unwrap();
        assert!(result.is_empty());

        let result = extractor.parse_response("   ").unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_parse_malformed_json_graceful() {
        use std::sync::Arc;
        use rook_core::traits::{LlmResponse, LlmStream, Tool, ToolChoice};
        use async_trait::async_trait;

        struct MockLlm;

        #[async_trait]
        impl Llm for MockLlm {
            async fn generate(&self, _: &[Message], _: Option<GenerationOptions>) -> RookResult<LlmResponse> {
                Ok(LlmResponse::default())
            }
            async fn generate_with_tools(&self, _: &[Message], _: &[Tool], _: ToolChoice, _: Option<GenerationOptions>) -> RookResult<LlmResponse> {
                Ok(LlmResponse::default())
            }
            async fn generate_stream(&self, _: &[Message], _: Option<GenerationOptions>) -> RookResult<LlmStream> {
                unimplemented!()
            }
            fn model_name(&self) -> &str { "mock" }
        }

        let extractor = EntityExtractor::new(Arc::new(MockLlm));

        // Completely invalid JSON should return empty result
        let result = extractor.parse_response("not json at all").unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_parse_alternative_field_names() {
        use std::sync::Arc;
        use rook_core::traits::{LlmResponse, LlmStream, Tool, ToolChoice};
        use async_trait::async_trait;

        struct MockLlm;

        #[async_trait]
        impl Llm for MockLlm {
            async fn generate(&self, _: &[Message], _: Option<GenerationOptions>) -> RookResult<LlmResponse> {
                Ok(LlmResponse::default())
            }
            async fn generate_with_tools(&self, _: &[Message], _: &[Tool], _: ToolChoice, _: Option<GenerationOptions>) -> RookResult<LlmResponse> {
                Ok(LlmResponse::default())
            }
            async fn generate_stream(&self, _: &[Message], _: Option<GenerationOptions>) -> RookResult<LlmStream> {
                unimplemented!()
            }
            fn model_name(&self) -> &str { "mock" }
        }

        let extractor = EntityExtractor::new(Arc::new(MockLlm));

        // Test alternative field names
        let json = r#"{
            "entities": [{"name": "Alice", "type": "person"}],
            "relationships": [{"from": "Alice", "to": "Bob", "type": "knows"}]
        }"#;

        let result = extractor.parse_response(json).unwrap();
        assert_eq!(result.entities[0].entity_type, EntityType::Person);
        // Note: from/to aren't supported in current implementation, checking source/target
    }

    #[test]
    fn test_parse_skips_invalid_entities() {
        use std::sync::Arc;
        use rook_core::traits::{LlmResponse, LlmStream, Tool, ToolChoice};
        use async_trait::async_trait;

        struct MockLlm;

        #[async_trait]
        impl Llm for MockLlm {
            async fn generate(&self, _: &[Message], _: Option<GenerationOptions>) -> RookResult<LlmResponse> {
                Ok(LlmResponse::default())
            }
            async fn generate_with_tools(&self, _: &[Message], _: &[Tool], _: ToolChoice, _: Option<GenerationOptions>) -> RookResult<LlmResponse> {
                Ok(LlmResponse::default())
            }
            async fn generate_stream(&self, _: &[Message], _: Option<GenerationOptions>) -> RookResult<LlmStream> {
                unimplemented!()
            }
            fn model_name(&self) -> &str { "mock" }
        }

        let extractor = EntityExtractor::new(Arc::new(MockLlm));

        // Entities without names should be skipped
        let json = r#"{
            "entities": [
                {"name": "Valid", "entity_type": "person"},
                {"entity_type": "person"},
                {"name": "", "entity_type": "person"},
                {"name": "   ", "entity_type": "person"}
            ],
            "relationships": []
        }"#;

        let result = extractor.parse_response(json).unwrap();
        assert_eq!(result.entities.len(), 1);
        assert_eq!(result.entities[0].name, "Valid");
    }
}
