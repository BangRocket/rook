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

            #[cfg(feature = "kuzu")]
            GraphStoreProvider::Kuzu => {
                let store = crate::kuzu::KuzuGraphStore::new(config).await?;
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
