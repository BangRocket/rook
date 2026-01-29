//! Category types for memory classification.
//!
//! This module provides the category system for classifying memories into
//! cognitive-science-based categories. Applications can use the default
//! categories, add custom categories, or completely override the category set.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use strum::{Display, EnumIter, EnumString, IntoEnumIterator, IntoStaticStr};

/// Default memory categories based on cognitive science research.
///
/// These 10 categories cover the primary domains of personal knowledge that
/// AI assistants typically need to remember about users. Categories serialize
/// to snake_case for storage compatibility.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    Serialize,
    Deserialize,
    Display,
    EnumString,
    EnumIter,
    IntoStaticStr,
)]
#[strum(serialize_all = "snake_case")]
#[serde(rename_all = "snake_case")]
pub enum DefaultCategory {
    /// Personal details: name, birthday, contact info, etc.
    PersonalDetails,
    /// Family members and family relationships.
    Family,
    /// Work, career, professional information.
    Professional,
    /// Likes, dislikes, preferences, styles.
    Preferences,
    /// Aspirations, plans, future intentions.
    Goals,
    /// Wellness, dietary restrictions, fitness, medical.
    Health,
    /// Active projects, endeavors, ongoing work.
    Projects,
    /// Friends, colleagues, social connections.
    Relationships,
    /// Life events, achievements, important dates.
    Milestones,
    /// Catch-all for uncategorized memories.
    Misc,
}

impl DefaultCategory {
    /// Returns a vector of all default category names as static strings.
    pub fn all_names() -> Vec<&'static str> {
        Self::iter().map(|c| c.into()).collect()
    }
}

/// Configuration for the category taxonomy.
///
/// Provides three modes of operation:
/// 1. **Defaults only**: `use_defaults = true`, empty `custom_categories`
/// 2. **Defaults + custom**: `use_defaults = true`, non-empty `custom_categories`
/// 3. **Override**: `allowed_categories = Some(...)` ignores other settings
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct CategoryConfig {
    /// Whether to include the default categories.
    pub use_defaults: bool,
    /// Additional custom categories to include.
    pub custom_categories: Vec<String>,
    /// If set, completely overrides both defaults and custom categories.
    /// Only these categories will be valid.
    pub allowed_categories: Option<Vec<String>>,
}

impl Default for CategoryConfig {
    fn default() -> Self {
        Self {
            use_defaults: true,
            custom_categories: Vec::new(),
            allowed_categories: None,
        }
    }
}

impl CategoryConfig {
    /// Returns the set of valid category names based on configuration.
    ///
    /// - If `allowed_categories` is set, returns only those categories
    /// - Otherwise, combines default categories (if enabled) with custom categories
    pub fn valid_categories(&self) -> HashSet<String> {
        if let Some(allowed) = &self.allowed_categories {
            return allowed.iter().cloned().collect();
        }

        let mut categories = HashSet::new();

        if self.use_defaults {
            for name in DefaultCategory::all_names() {
                categories.insert(name.to_string());
            }
        }

        for custom in &self.custom_categories {
            categories.insert(custom.clone());
        }

        categories
    }

    /// Checks if a category name is valid according to this configuration.
    pub fn is_valid(&self, category: &str) -> bool {
        self.valid_categories().contains(category)
    }

    /// Creates a config with only default categories.
    pub fn defaults_only() -> Self {
        Self::default()
    }

    /// Creates a config with default categories plus custom ones.
    pub fn with_custom(custom: Vec<String>) -> Self {
        Self {
            use_defaults: true,
            custom_categories: custom,
            allowed_categories: None,
        }
    }

    /// Creates a config that only allows specific categories.
    pub fn override_with(allowed: Vec<String>) -> Self {
        Self {
            use_defaults: false,
            custom_categories: Vec::new(),
            allowed_categories: Some(allowed),
        }
    }
}

/// Configuration for key memory handling.
///
/// Key memories are special high-importance memories that are exempt from
/// decay and archival. This config controls how many key memories can exist
/// per scope and whether they're included in search results.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct KeyMemoryConfig {
    /// Maximum number of key memories per scope (user/agent/session).
    pub max_key_memories: usize,
    /// Whether to include key memories in search results.
    pub include_in_search: bool,
}

impl Default for KeyMemoryConfig {
    fn default() -> Self {
        Self {
            max_key_memories: 15,
            include_in_search: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    #[test]
    fn test_default_category_display() {
        // Verify snake_case serialization
        assert_eq!(DefaultCategory::PersonalDetails.to_string(), "personal_details");
        assert_eq!(DefaultCategory::Family.to_string(), "family");
        assert_eq!(DefaultCategory::Professional.to_string(), "professional");
        assert_eq!(DefaultCategory::Preferences.to_string(), "preferences");
        assert_eq!(DefaultCategory::Goals.to_string(), "goals");
        assert_eq!(DefaultCategory::Health.to_string(), "health");
        assert_eq!(DefaultCategory::Projects.to_string(), "projects");
        assert_eq!(DefaultCategory::Relationships.to_string(), "relationships");
        assert_eq!(DefaultCategory::Milestones.to_string(), "milestones");
        assert_eq!(DefaultCategory::Misc.to_string(), "misc");
    }

    #[test]
    fn test_default_category_from_str() {
        // Verify parsing from snake_case strings
        assert_eq!(
            DefaultCategory::from_str("personal_details").unwrap(),
            DefaultCategory::PersonalDetails
        );
        assert_eq!(
            DefaultCategory::from_str("family").unwrap(),
            DefaultCategory::Family
        );
        assert_eq!(
            DefaultCategory::from_str("professional").unwrap(),
            DefaultCategory::Professional
        );
        assert_eq!(
            DefaultCategory::from_str("misc").unwrap(),
            DefaultCategory::Misc
        );

        // Invalid category should error
        assert!(DefaultCategory::from_str("invalid").is_err());
    }

    #[test]
    fn test_default_category_all_names() {
        let names = DefaultCategory::all_names();

        // Should have exactly 10 categories
        assert_eq!(names.len(), 10);

        // Verify all expected categories are present
        assert!(names.contains(&"personal_details"));
        assert!(names.contains(&"family"));
        assert!(names.contains(&"professional"));
        assert!(names.contains(&"preferences"));
        assert!(names.contains(&"goals"));
        assert!(names.contains(&"health"));
        assert!(names.contains(&"projects"));
        assert!(names.contains(&"relationships"));
        assert!(names.contains(&"milestones"));
        assert!(names.contains(&"misc"));
    }

    #[test]
    fn test_category_config_default() {
        let config = CategoryConfig::default();

        assert!(config.use_defaults);
        assert!(config.custom_categories.is_empty());
        assert!(config.allowed_categories.is_none());

        // Should include all default categories
        let valid = config.valid_categories();
        assert_eq!(valid.len(), 10);
        assert!(config.is_valid("personal_details"));
        assert!(config.is_valid("misc"));
        assert!(!config.is_valid("custom_category"));
    }

    #[test]
    fn test_category_config_custom() {
        let config = CategoryConfig::with_custom(vec![
            "research".to_string(),
            "hobbies".to_string(),
        ]);

        // Should include defaults + custom
        let valid = config.valid_categories();
        assert_eq!(valid.len(), 12); // 10 defaults + 2 custom

        assert!(config.is_valid("personal_details")); // default
        assert!(config.is_valid("research")); // custom
        assert!(config.is_valid("hobbies")); // custom
        assert!(!config.is_valid("invalid")); // not valid
    }

    #[test]
    fn test_category_config_allowed_override() {
        let config = CategoryConfig::override_with(vec![
            "work".to_string(),
            "personal".to_string(),
            "other".to_string(),
        ]);

        // Should only include allowed categories, ignoring defaults
        let valid = config.valid_categories();
        assert_eq!(valid.len(), 3);

        assert!(config.is_valid("work"));
        assert!(config.is_valid("personal"));
        assert!(config.is_valid("other"));
        assert!(!config.is_valid("personal_details")); // default, but not allowed
        assert!(!config.is_valid("family")); // default, but not allowed
    }

    #[test]
    fn test_key_memory_config_default() {
        let config = KeyMemoryConfig::default();

        assert_eq!(config.max_key_memories, 15);
        assert!(config.include_in_search);
    }

    #[test]
    fn test_category_serde() {
        // Test JSON serialization/deserialization
        let category = DefaultCategory::PersonalDetails;
        let json = serde_json::to_string(&category).unwrap();
        assert_eq!(json, "\"personal_details\"");

        let parsed: DefaultCategory = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, DefaultCategory::PersonalDetails);
    }
}
