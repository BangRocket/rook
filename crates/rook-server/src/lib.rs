//! rook-server - REST API server for rook.
//!
//! This crate provides a REST API server for the rook memory layer.
//!
//! # Example
//!
//! ```ignore
//! use rook_server::{create_server, AppState};
//!
//! #[tokio::main]
//! async fn main() {
//!     let state = AppState::new();
//!     let app = create_server(state);
//!     
//!     let listener = tokio::net::TcpListener::bind("0.0.0.0:8080").await.unwrap();
//!     axum::serve(listener, app).await.unwrap();
//! }
//! ```

pub mod error;
pub mod factory;
pub mod middleware;
pub mod routes;
pub mod state;

pub use error::{ApiError, ApiResult};
pub use factory::create_memory;
pub use state::AppState;

use axum::{middleware as axum_middleware, Router};
use tower_http::trace::TraceLayer;

/// Create the server with all routes and middleware.
pub fn create_server(state: AppState) -> Router {
    routes::create_router(state)
        .layer(TraceLayer::new_for_http())
        .layer(middleware::cors_layer())
        .layer(axum_middleware::from_fn(middleware::logging_middleware))
}

/// Create the server with authentication middleware.
pub fn create_server_with_auth(state: AppState) -> Router {
    routes::create_router(state)
        .layer(TraceLayer::new_for_http())
        .layer(middleware::cors_layer())
        .layer(axum_middleware::from_fn(middleware::auth_middleware))
        .layer(axum_middleware::from_fn(middleware::logging_middleware))
}
