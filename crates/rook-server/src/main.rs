//! rook-server - REST API server binary.

use std::net::SocketAddr;

use rook_core::{BackgroundRuntime, RuntimeConfig};
use rook_server::{create_server, create_server_with_auth, AppState};
use tokio::signal;
use tracing::{info, Level};
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

/// Wait for shutdown signal (Ctrl+C or SIGTERM).
async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
}

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
    let host = std::env::var("ROOK_HOST").unwrap_or_else(|_| "0.0.0.0".to_string());
    let port: u16 = std::env::var("ROOK_PORT")
        .unwrap_or_else(|_| "8080".to_string())
        .parse()
        .expect("ROOK_PORT must be a valid port number");
    let require_auth = std::env::var("ROOK_REQUIRE_AUTH").is_ok();

    // Create BackgroundRuntime with config from environment
    let runtime_config = RuntimeConfig::from_env();
    let runtime = BackgroundRuntime::new(runtime_config).await?;

    // Start background schedulers
    runtime.start().await?;
    info!("Background schedulers started (consolidation + intentions)");

    // Create application state with runtime
    let state = AppState::new_with_runtime(runtime);

    // Create server with or without auth
    let app = if require_auth {
        info!("Authentication enabled");
        create_server_with_auth(state.clone())
    } else {
        info!("Authentication disabled");
        create_server(state.clone())
    };

    // Start server
    let addr: SocketAddr = format!("{}:{}", host, port).parse()?;
    info!("Starting rook-server on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;

    // Serve with graceful shutdown
    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            shutdown_signal().await;
            info!("Shutdown signal received, stopping schedulers...");
        })
        .await?;

    // Explicit shutdown of runtime
    if let Some(rt) = state.take_runtime() {
        rt.write().await.shutdown().await?;
    }

    info!("Server stopped cleanly");
    Ok(())
}
