//! Factory for creating reranker providers.

use std::sync::Arc;

use rook_core::error::{RookError, RookResult};
use rook_core::traits::{Reranker, RerankerConfig, RerankerProvider};

/// Factory for creating reranker providers.
pub struct RerankerFactory;

impl RerankerFactory {
    /// Create a reranker from the given configuration.
    pub async fn create(
        provider: RerankerProvider,
        config: RerankerConfig,
    ) -> RookResult<Arc<dyn Reranker>> {
        match provider {
            #[cfg(feature = "cohere")]
            RerankerProvider::Cohere => {
                let reranker = crate::cohere::CohereReranker::new(config)?;
                Ok(Arc::new(reranker))
            }

            #[cfg(feature = "llm")]
            RerankerProvider::Llm => {
                let reranker = crate::llm_reranker::LlmReranker::new(config)?;
                Ok(Arc::new(reranker))
            }

            #[allow(unreachable_patterns)]
            _ => Err(RookError::UnsupportedProvider {
                provider: format!("{:?}", provider),
            }),
        }
    }

    /// Create a Cohere reranker.
    #[cfg(feature = "cohere")]
    pub fn cohere(api_key: &str) -> RookResult<Arc<dyn Reranker>> {
        let config = RerankerConfig {
            api_key: Some(api_key.to_string()),
            model: "rerank-english-v3.0".to_string(),
            ..Default::default()
        };
        let reranker = crate::cohere::CohereReranker::new(config)?;
        Ok(Arc::new(reranker))
    }
}
