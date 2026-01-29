//! Category nodes for memory classification in the knowledge graph.
//!
//! Categories are first-class graph citizens that enable:
//! - Hierarchical category structure (subcategories)
//! - Graph-based category navigation
//! - Memory-to-category relationships
//!
//! # Graph Structure
//!
//! ```text
//! [Category: professional]
//!        |
//!        +-- BELONGS_TO_CATEGORY -- [Memory: "User works at Acme Corp"]
//!        |
//!        +-- SUBCATEGORY_OF -- [Category: work_projects]
//!                                    |
//!                                    +-- BELONGS_TO_CATEGORY -- [Memory: "Working on Q4 launch"]
//! ```
//!
//! # Usage with EmbeddedGraphStore
//!
//! ```ignore
//! use rook_graph_stores::category::{CategoryNode, default_categories};
//! use rook_graph_stores::EmbeddedGraphStore;
//!
//! let store = EmbeddedGraphStore::in_memory()?;
//! let filters = GraphFilters::default();
//!
//! // Initialize default categories
//! store.initialize_default_categories(&filters)?;
//!
//! // Link a memory to a category
//! store.link_memory_to_category("mem-123", "professional", &filters)?;
//!
//! // Get all memories in a category
//! let memories = store.get_memories_in_category("professional", &filters)?;
//! ```

use serde::{Deserialize, Serialize};

use rook_core::types::DefaultCategory;

/// A category node in the knowledge graph.
///
/// Category nodes represent classification groups for memories.
/// They support hierarchical structures through parent_category.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategoryNode {
    /// The category name (e.g., "professional", "family").
    pub name: String,
    /// Human-readable description.
    pub description: String,
    /// Parent category name for hierarchy (None if root category).
    pub parent_category: Option<String>,
    /// Whether this is a system-defined category (vs user-created).
    pub is_system: bool,
}

impl CategoryNode {
    /// Create a new category node.
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parent_category: None,
            is_system: false,
        }
    }

    /// Create a system-defined category.
    pub fn system(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parent_category: None,
            is_system: true,
        }
    }

    /// Set the parent category for hierarchy.
    pub fn with_parent(mut self, parent: impl Into<String>) -> Self {
        self.parent_category = Some(parent.into());
        self
    }
}

/// Get the default categories as CategoryNodes.
///
/// Returns the 10 default categories from rook-core's DefaultCategory enum,
/// converted to CategoryNode instances ready for graph storage.
pub fn default_categories() -> Vec<CategoryNode> {
    vec![
        CategoryNode::system(
            DefaultCategory::PersonalDetails.to_string(),
            "Personal information like name, birthday, contact details",
        ),
        CategoryNode::system(
            DefaultCategory::Family.to_string(),
            "Family members and family relationships",
        ),
        CategoryNode::system(
            DefaultCategory::Professional.to_string(),
            "Work, career, and professional information",
        ),
        CategoryNode::system(
            DefaultCategory::Preferences.to_string(),
            "Likes, dislikes, preferences, and personal styles",
        ),
        CategoryNode::system(
            DefaultCategory::Goals.to_string(),
            "Aspirations, plans, and future intentions",
        ),
        CategoryNode::system(
            DefaultCategory::Health.to_string(),
            "Wellness, dietary restrictions, fitness, and medical info",
        ),
        CategoryNode::system(
            DefaultCategory::Projects.to_string(),
            "Active projects, endeavors, and ongoing work",
        ),
        CategoryNode::system(
            DefaultCategory::Relationships.to_string(),
            "Friends, colleagues, and social connections",
        ),
        CategoryNode::system(
            DefaultCategory::Milestones.to_string(),
            "Life events, achievements, and important dates",
        ),
        CategoryNode::system(
            DefaultCategory::Misc.to_string(),
            "Uncategorized memories and miscellaneous information",
        ),
    ]
}

/// Get a specific default category by name.
pub fn get_default_category(name: &str) -> Option<CategoryNode> {
    default_categories().into_iter().find(|c| c.name == name)
}

/// Check if a category name is a default system category.
pub fn is_default_category(name: &str) -> bool {
    DefaultCategory::all_names().contains(&name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_category_node_creation() {
        let category = CategoryNode::new("work_projects", "Work-related projects");

        assert_eq!(category.name, "work_projects");
        assert_eq!(category.description, "Work-related projects");
        assert!(category.parent_category.is_none());
        assert!(!category.is_system);
    }

    #[test]
    fn test_category_node_with_parent() {
        let category = CategoryNode::new("q4_launch", "Q4 product launch project")
            .with_parent("professional");

        assert_eq!(category.parent_category, Some("professional".to_string()));
    }

    #[test]
    fn test_system_category() {
        let category = CategoryNode::system("professional", "Work stuff");

        assert!(category.is_system);
    }

    #[test]
    fn test_default_categories() {
        let categories = default_categories();

        // Should have exactly 10 default categories
        assert_eq!(categories.len(), 10);

        // All should be system categories
        assert!(categories.iter().all(|c| c.is_system));

        // Check specific categories exist
        let names: Vec<&str> = categories.iter().map(|c| c.name.as_str()).collect();
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
    fn test_get_default_category() {
        let professional = get_default_category("professional");
        assert!(professional.is_some());
        assert_eq!(professional.unwrap().name, "professional");

        let unknown = get_default_category("not_a_category");
        assert!(unknown.is_none());
    }

    #[test]
    fn test_is_default_category() {
        assert!(is_default_category("professional"));
        assert!(is_default_category("family"));
        assert!(is_default_category("misc"));

        assert!(!is_default_category("custom_category"));
        assert!(!is_default_category("work_projects"));
    }

    #[test]
    fn test_category_serde() {
        let category = CategoryNode::new("test", "Test category")
            .with_parent("professional");

        let json = serde_json::to_string(&category).unwrap();
        let parsed: CategoryNode = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.name, "test");
        assert_eq!(parsed.parent_category, Some("professional".to_string()));
    }
}
