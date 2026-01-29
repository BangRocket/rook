//! Entity and relationship type definitions.
//!
//! This module defines the core types for entity extraction:
//! - `EntityType`: Categories of entities (person, organization, etc.)
//! - `RelationshipType`: Types of relationships between entities

use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

/// Entity types that can be extracted from text.
///
/// These represent the fundamental categories of entities in a knowledge graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EntityType {
    /// A person (e.g., "Alice", "John Smith").
    Person,
    /// An organization (e.g., "Acme Corp", "MIT").
    Organization,
    /// A physical location (e.g., "New York", "123 Main St").
    Location,
    /// A project or initiative (e.g., "Project Alpha", "Q4 Launch").
    Project,
    /// An abstract concept or topic (e.g., "machine learning", "democracy").
    Concept,
    /// An event (e.g., "meeting", "conference", "birthday").
    Event,
    /// A memory category node for classification grouping.
    Category,
}

impl EntityType {
    /// Parse entity type from string with flexible matching.
    ///
    /// This handles variations in LLM output like "PERSON", "Person", "person",
    /// "per", "people", etc.
    pub fn from_str_flexible(s: &str) -> Option<Self> {
        let normalized = s.trim().to_lowercase();

        match normalized.as_str() {
            // Person variants
            "person" | "per" | "people" | "individual" | "human" | "user" => Some(Self::Person),

            // Organization variants
            "organization" | "org" | "organisation" | "company" | "corporation"
            | "institution" | "business" | "firm" | "agency" => Some(Self::Organization),

            // Location variants
            "location" | "loc" | "place" | "address" | "city" | "country"
            | "region" | "area" | "venue" | "site" => Some(Self::Location),

            // Project variants
            "project" | "proj" | "initiative" | "program" | "programme"
            | "campaign" | "venture" => Some(Self::Project),

            // Concept variants
            "concept" | "idea" | "topic" | "theme" | "notion" | "theory"
            | "subject" | "field" | "discipline" => Some(Self::Concept),

            // Event variants
            "event" | "evt" | "meeting" | "conference" | "occasion"
            | "happening" | "occurrence" | "activity" => Some(Self::Event),

            // Category variants
            "category" | "cat" | "classification" | "group" | "type" | "class"
            | "taxonomy" | "tag" => Some(Self::Category),

            _ => None,
        }
    }

    /// Get all entity type variants.
    pub fn all() -> &'static [EntityType] {
        &[
            Self::Person,
            Self::Organization,
            Self::Location,
            Self::Project,
            Self::Concept,
            Self::Event,
            Self::Category,
        ]
    }

    /// Convert to string for prompts and display.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Person => "person",
            Self::Organization => "organization",
            Self::Location => "location",
            Self::Project => "project",
            Self::Concept => "concept",
            Self::Event => "event",
            Self::Category => "category",
        }
    }
}

impl fmt::Display for EntityType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl FromStr for EntityType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::from_str_flexible(s).ok_or_else(|| format!("Unknown entity type: {}", s))
    }
}

/// Relationship types between entities.
///
/// These represent the semantic connections in a knowledge graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RelationshipType {
    /// Person knows another person.
    Knows,
    /// Person works at organization.
    WorksAt,
    /// Person or entity lives in location.
    LivesIn,
    /// Entity is located in location.
    LocatedIn,
    /// Entity is part of another entity.
    PartOf,
    /// Generic relationship between entities.
    RelatedTo,
    /// Entity was created by person/organization.
    CreatedBy,
    /// Person participated in event.
    ParticipatedIn,
    /// Entity was mentioned in context.
    MentionedIn,
    /// Memory belongs to a category.
    BelongsToCategory,
    /// Category is a subcategory of another.
    SubcategoryOf,
}

impl RelationshipType {
    /// Parse relationship type from string with flexible matching.
    ///
    /// This handles variations in LLM output like "KNOWS", "knows", "know",
    /// "friends_with", etc.
    pub fn from_str_flexible(s: &str) -> Option<Self> {
        let normalized = s.trim().to_lowercase().replace(['-', ' '], "_");

        match normalized.as_str() {
            // Knows variants
            "knows" | "know" | "friends_with" | "acquainted_with" | "met"
            | "connected_to" | "knows_of" => Some(Self::Knows),

            // WorksAt variants
            "works_at" | "worksat" | "employed_by" | "employee_of" | "works_for"
            | "employed_at" | "member_of" | "affiliated_with" => Some(Self::WorksAt),

            // LivesIn variants
            "lives_in" | "livesin" | "resides_in" | "lives_at" | "resident_of"
            | "based_in" | "home_in" => Some(Self::LivesIn),

            // LocatedIn variants
            "located_in" | "locatedin" | "situated_in" | "found_in" | "in"
            | "at" | "headquartered_in" => Some(Self::LocatedIn),

            // PartOf variants
            "part_of" | "partof" | "belongs_to" | "component_of" | "subset_of"
            | "division_of" | "department_of" | "included_in" => Some(Self::PartOf),

            // RelatedTo variants
            "related_to" | "relatedto" | "associated_with" | "linked_to"
            | "connected_with" | "tied_to" | "involves" => Some(Self::RelatedTo),

            // CreatedBy variants
            "created_by" | "createdby" | "made_by" | "authored_by" | "built_by"
            | "developed_by" | "founded_by" | "invented_by" => Some(Self::CreatedBy),

            // ParticipatedIn variants
            "participated_in" | "participatedin" | "attended" | "joined"
            | "took_part_in" | "involved_in" | "present_at" => Some(Self::ParticipatedIn),

            // MentionedIn variants
            "mentioned_in" | "mentionedin" | "referenced_in" | "cited_in"
            | "appears_in" | "discussed_in" | "noted_in" => Some(Self::MentionedIn),

            // BelongsToCategory variants
            "belongs_to_category" | "in_category" | "categorized_as" | "classified_as"
            | "tagged_as" | "has_category" => Some(Self::BelongsToCategory),

            // SubcategoryOf variants
            "subcategory_of" | "child_of" | "parent_category" | "under_category"
            | "sub_category" | "nested_in" => Some(Self::SubcategoryOf),

            _ => None,
        }
    }

    /// Get all relationship type variants.
    pub fn all() -> &'static [RelationshipType] {
        &[
            Self::Knows,
            Self::WorksAt,
            Self::LivesIn,
            Self::LocatedIn,
            Self::PartOf,
            Self::RelatedTo,
            Self::CreatedBy,
            Self::ParticipatedIn,
            Self::MentionedIn,
            Self::BelongsToCategory,
            Self::SubcategoryOf,
        ]
    }

    /// Convert to string for prompts and display.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Knows => "knows",
            Self::WorksAt => "works_at",
            Self::LivesIn => "lives_in",
            Self::LocatedIn => "located_in",
            Self::PartOf => "part_of",
            Self::RelatedTo => "related_to",
            Self::CreatedBy => "created_by",
            Self::ParticipatedIn => "participated_in",
            Self::MentionedIn => "mentioned_in",
            Self::BelongsToCategory => "belongs_to_category",
            Self::SubcategoryOf => "subcategory_of",
        }
    }

    /// Get a human-readable description.
    pub fn description(&self) -> &'static str {
        match self {
            Self::Knows => "Person knows another person",
            Self::WorksAt => "Person works at organization",
            Self::LivesIn => "Person or entity lives in location",
            Self::LocatedIn => "Entity is located in location",
            Self::PartOf => "Entity is part of another entity",
            Self::RelatedTo => "Generic relationship between entities",
            Self::CreatedBy => "Entity was created by person/organization",
            Self::ParticipatedIn => "Person participated in event",
            Self::MentionedIn => "Entity was mentioned in context",
            Self::BelongsToCategory => "Memory belongs to a category",
            Self::SubcategoryOf => "Category is a subcategory of another",
        }
    }
}

impl fmt::Display for RelationshipType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl FromStr for RelationshipType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::from_str_flexible(s).ok_or_else(|| format!("Unknown relationship type: {}", s))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entity_type_from_str_flexible() {
        // Standard forms
        assert_eq!(EntityType::from_str_flexible("person"), Some(EntityType::Person));
        assert_eq!(EntityType::from_str_flexible("organization"), Some(EntityType::Organization));
        assert_eq!(EntityType::from_str_flexible("location"), Some(EntityType::Location));
        assert_eq!(EntityType::from_str_flexible("project"), Some(EntityType::Project));
        assert_eq!(EntityType::from_str_flexible("concept"), Some(EntityType::Concept));
        assert_eq!(EntityType::from_str_flexible("event"), Some(EntityType::Event));

        // Case insensitive
        assert_eq!(EntityType::from_str_flexible("PERSON"), Some(EntityType::Person));
        assert_eq!(EntityType::from_str_flexible("Person"), Some(EntityType::Person));

        // Variants
        assert_eq!(EntityType::from_str_flexible("company"), Some(EntityType::Organization));
        assert_eq!(EntityType::from_str_flexible("city"), Some(EntityType::Location));
        assert_eq!(EntityType::from_str_flexible("meeting"), Some(EntityType::Event));
        assert_eq!(EntityType::from_str_flexible("individual"), Some(EntityType::Person));

        // With whitespace
        assert_eq!(EntityType::from_str_flexible("  person  "), Some(EntityType::Person));

        // Unknown
        assert_eq!(EntityType::from_str_flexible("unknown"), None);
        assert_eq!(EntityType::from_str_flexible(""), None);
    }

    #[test]
    fn test_relationship_type_from_str_flexible() {
        // Standard forms
        assert_eq!(RelationshipType::from_str_flexible("knows"), Some(RelationshipType::Knows));
        assert_eq!(RelationshipType::from_str_flexible("works_at"), Some(RelationshipType::WorksAt));
        assert_eq!(RelationshipType::from_str_flexible("lives_in"), Some(RelationshipType::LivesIn));
        assert_eq!(RelationshipType::from_str_flexible("located_in"), Some(RelationshipType::LocatedIn));
        assert_eq!(RelationshipType::from_str_flexible("part_of"), Some(RelationshipType::PartOf));
        assert_eq!(RelationshipType::from_str_flexible("related_to"), Some(RelationshipType::RelatedTo));
        assert_eq!(RelationshipType::from_str_flexible("created_by"), Some(RelationshipType::CreatedBy));
        assert_eq!(RelationshipType::from_str_flexible("participated_in"), Some(RelationshipType::ParticipatedIn));
        assert_eq!(RelationshipType::from_str_flexible("mentioned_in"), Some(RelationshipType::MentionedIn));

        // Case insensitive
        assert_eq!(RelationshipType::from_str_flexible("KNOWS"), Some(RelationshipType::Knows));
        assert_eq!(RelationshipType::from_str_flexible("Works_At"), Some(RelationshipType::WorksAt));

        // Variants
        assert_eq!(RelationshipType::from_str_flexible("friends_with"), Some(RelationshipType::Knows));
        assert_eq!(RelationshipType::from_str_flexible("employed_by"), Some(RelationshipType::WorksAt));
        assert_eq!(RelationshipType::from_str_flexible("resides_in"), Some(RelationshipType::LivesIn));
        assert_eq!(RelationshipType::from_str_flexible("founded_by"), Some(RelationshipType::CreatedBy));

        // With hyphens/spaces (normalized)
        assert_eq!(RelationshipType::from_str_flexible("works-at"), Some(RelationshipType::WorksAt));
        assert_eq!(RelationshipType::from_str_flexible("works at"), Some(RelationshipType::WorksAt));

        // Unknown
        assert_eq!(RelationshipType::from_str_flexible("unknown"), None);
        assert_eq!(RelationshipType::from_str_flexible(""), None);
    }

    #[test]
    fn test_entity_type_display() {
        assert_eq!(EntityType::Person.to_string(), "person");
        assert_eq!(EntityType::Organization.to_string(), "organization");
    }

    #[test]
    fn test_relationship_type_display() {
        assert_eq!(RelationshipType::Knows.to_string(), "knows");
        assert_eq!(RelationshipType::WorksAt.to_string(), "works_at");
    }

    #[test]
    fn test_entity_type_all() {
        let all = EntityType::all();
        assert_eq!(all.len(), 7);
        assert!(all.contains(&EntityType::Person));
        assert!(all.contains(&EntityType::Event));
        assert!(all.contains(&EntityType::Category));
    }

    #[test]
    fn test_relationship_type_all() {
        let all = RelationshipType::all();
        assert_eq!(all.len(), 11);
        assert!(all.contains(&RelationshipType::Knows));
        assert!(all.contains(&RelationshipType::MentionedIn));
        assert!(all.contains(&RelationshipType::BelongsToCategory));
        assert!(all.contains(&RelationshipType::SubcategoryOf));
    }

    #[test]
    fn test_entity_type_serde() {
        let entity = EntityType::Person;
        let json = serde_json::to_string(&entity).unwrap();
        assert_eq!(json, "\"person\"");

        let parsed: EntityType = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, EntityType::Person);
    }

    #[test]
    fn test_relationship_type_serde() {
        let rel = RelationshipType::WorksAt;
        let json = serde_json::to_string(&rel).unwrap();
        assert_eq!(json, "\"works_at\"");

        let parsed: RelationshipType = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, RelationshipType::WorksAt);
    }
}
