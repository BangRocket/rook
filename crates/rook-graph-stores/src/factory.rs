//! Factory for creating graph store providers.

use std::sync::Arc;

use rook_core::error::{RookError, RookResult};
use rook_core::traits::{GraphStore, GraphStoreConfig, GraphStoreProvider};

/// Factory for creating graph store providers.
pub struct GraphStoreFactory;

impl GraphStoreFactory {
    /// Create a graph store from the given configuration.
    pub async fn create(
        provider: GraphStoreProvider,
        config: GraphStoreConfig,
    ) -> RookResult<Arc<dyn GraphStore>> {
        match provider {
            #[cfg(feature = "embedded")]
            GraphStoreProvider::Embedded => {
                let store = crate::embedded::EmbeddedGraphStore::from_config(&config).await?;
                Ok(Arc::new(store))
            }

            #[cfg(feature = "neo4j")]
            GraphStoreProvider::Neo4j => {
                let store = crate::neo4j::Neo4jGraphStore::new(config).await?;
                Ok(Arc::new(store))
            }

            #[cfg(feature = "memgraph")]
            GraphStoreProvider::Memgraph => {
                let store = crate::memgraph::MemgraphGraphStore::new(config).await?;
                Ok(Arc::new(store))
            }

            #[cfg(feature = "neptune")]
            GraphStoreProvider::Neptune => {
                let store = crate::neptune::NeptuneGraphStore::new(config).await?;
                Ok(Arc::new(store))
            }

            #[allow(unreachable_patterns)]
            _ => Err(RookError::UnsupportedProvider {
                provider: format!("{:?}", provider),
            }),
        }
    }

    /// Create an embedded graph store (in-memory).
    ///
    /// This is the default and recommended option for most use cases.
    /// No external database dependencies required.
    #[cfg(feature = "embedded")]
    pub fn embedded_memory() -> RookResult<Arc<dyn GraphStore>> {
        let store = crate::embedded::EmbeddedGraphStore::in_memory()?;
        Ok(Arc::new(store))
    }

    /// Create an embedded graph store with file-based persistence.
    #[cfg(feature = "embedded")]
    pub fn embedded_file(path: impl AsRef<std::path::Path>) -> RookResult<Arc<dyn GraphStore>> {
        let store = crate::embedded::EmbeddedGraphStore::new(path)?;
        Ok(Arc::new(store))
    }

    /// Create a Neo4j graph store with default configuration.
    #[cfg(feature = "neo4j")]
    pub async fn neo4j(uri: &str, username: &str, password: &str) -> RookResult<Arc<dyn GraphStore>> {
        let config = GraphStoreConfig {
            provider: GraphStoreProvider::Neo4j,
            url: uri.to_string(),
            username: Some(username.to_string()),
            password: Some(password.to_string()),
            database: None,
        };
        Self::create(GraphStoreProvider::Neo4j, config).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rook_core::traits::GraphFilters;

    #[tokio::test]
    #[cfg(feature = "embedded")]
    async fn test_factory_create_embedded() {
        let config = GraphStoreConfig::embedded_memory();
        let store = GraphStoreFactory::create(GraphStoreProvider::Embedded, config)
            .await
            .unwrap();

        // Verify it works
        let entities = store.get_all(&GraphFilters::default()).await.unwrap();
        assert!(entities.is_empty());
    }

    #[test]
    #[cfg(feature = "embedded")]
    fn test_factory_embedded_memory() {
        let store = GraphStoreFactory::embedded_memory().unwrap();
        // Store exists - we just verify it was created without errors
        // The Arc<dyn GraphStore> is valid if we got here
        drop(store);
    }

    #[tokio::test]
    #[cfg(feature = "embedded")]
    async fn test_factory_embedded_file() {
        let temp_dir = tempfile::tempdir().unwrap();
        let db_path = temp_dir.path().join("test.db");

        let store = GraphStoreFactory::embedded_file(&db_path).unwrap();

        // Verify it works
        let entities = store.get_all(&GraphFilters::default()).await.unwrap();
        assert!(entities.is_empty());

        // File should exist
        assert!(db_path.exists());
    }
}
