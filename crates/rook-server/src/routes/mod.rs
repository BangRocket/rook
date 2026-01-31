//! Route definitions for the REST API.

mod config;
mod health;
mod memories;
mod search;
mod signals;

use axum::{
    routing::{delete, get, post, put},
    Router,
};

use crate::state::AppState;

/// Create the main application router.
pub fn create_router(state: AppState) -> Router {
    Router::new()
        // Health check
        .route("/health", get(health::health_check))
        // Memory operations
        .route("/memories", post(memories::add_memory))
        .route("/memories", get(memories::get_all_memories))
        .route("/memories", delete(memories::delete_all_memories))
        .route("/memories/:id", get(memories::get_memory))
        .route("/memories/:id", put(memories::update_memory))
        .route("/memories/:id", delete(memories::delete_memory))
        .route("/memories/:id/history", get(memories::get_memory_history))
        // Search
        .route("/search", post(search::search_memories))
        // Strength signals
        .route("/signals", post(signals::process_signals))
        .route("/signals/apply", post(signals::apply_updates))
        // Configuration
        .route("/configure", post(config::configure))
        .route("/reset", post(config::reset))
        // Attach state
        .with_state(state)
}

pub use config::*;
pub use health::*;
pub use memories::*;
pub use search::*;
pub use signals::*;
