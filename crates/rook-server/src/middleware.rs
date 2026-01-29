//! Middleware for the REST API server.

use axum::{
    extract::Request,
    http::StatusCode,
    middleware::Next,
    response::Response,
};
use tower_http::cors::{Any, CorsLayer};
use tracing::info;

/// Create CORS middleware.
pub fn cors_layer() -> CorsLayer {
    CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any)
}

/// Request logging middleware.
pub async fn logging_middleware(request: Request, next: Next) -> Response {
    let method = request.method().clone();
    let uri = request.uri().clone();
    let start = std::time::Instant::now();

    let response = next.run(request).await;

    let duration = start.elapsed();
    let status = response.status();

    info!(
        method = %method,
        uri = %uri,
        status = %status.as_u16(),
        duration_ms = %duration.as_millis(),
        "Request completed"
    );

    response
}

/// API key authentication middleware (optional).
pub async fn auth_middleware(
    request: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    // Check for API key in header if ROOK_REQUIRE_AUTH is set
    if std::env::var("ROOK_REQUIRE_AUTH").is_ok() {
        let expected_key = std::env::var("ROOK_API_KEY").unwrap_or_default();

        if !expected_key.is_empty() {
            let auth_header = request
                .headers()
                .get("Authorization")
                .and_then(|v| v.to_str().ok());

            match auth_header {
                Some(header) if header.starts_with("Bearer ") || header.starts_with("Token ") => {
                    let token = header
                        .strip_prefix("Bearer ")
                        .or_else(|| header.strip_prefix("Token "))
                        .unwrap_or("");

                    if token != expected_key {
                        return Err(StatusCode::UNAUTHORIZED);
                    }
                }
                _ => return Err(StatusCode::UNAUTHORIZED),
            }
        }
    }

    Ok(next.run(request).await)
}
