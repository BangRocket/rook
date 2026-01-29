//! rook-server - REST API server binary.

use std::net::SocketAddr;

use rook_server::{create_server, create_server_with_auth, AppState};
use tracing::{info, Level};
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load environment variables
    dotenvy::dotenv().ok();

    // Initialize tracing
    tracing_subscriber::registry()
        .with(fmt::layer())
        .with(
            EnvFilter::from_default_env()
                .add_directive(Level::INFO.into())
                .add_directive("rook_server=debug".parse().unwrap()),
        )
        .init();

    // Get configuration from environment
    let host = std::env::var("MEM0_HOST").unwrap_or_else(|_| "0.0.0.0".to_string());
    let port: u16 = std::env::var("MEM0_PORT")
        .unwrap_or_else(|_| "8080".to_string())
        .parse()
        .expect("MEM0_PORT must be a valid port number");
    let require_auth = std::env::var("MEM0_REQUIRE_AUTH").is_ok();

    // Create application state
    let state = AppState::new();

    // Create server with or without auth
    let app = if require_auth {
        info!("Authentication enabled");
        create_server_with_auth(state)
    } else {
        info!("Authentication disabled");
        create_server(state)
    };

    // Start server
    let addr: SocketAddr = format!("{}:{}", host, port).parse()?;
    info!("Starting rook-server on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
