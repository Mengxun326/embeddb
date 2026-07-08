//! EmbedDB Server — Web Dashboard HTTP API server.
//!
//! Provides a REST API for managing EmbedDB databases through a browser.
//! Used by the React dashboard and available for third-party integration.
//!
//! # API Endpoints
//!
//! - `GET  /api/health` — Health check
//! - `GET  /api/collections` — List all collections
//! - `POST /api/collections` — Create a new collection
//! - `GET  /api/collections/:name` — Get collection details
//! - `POST /api/collections/:name/search` — Search vectors
//! - `POST /api/collections/:name/insert` — Insert a document
//! - `DELETE /api/collections/:name/:id` — Delete a document
//! - `GET  /api/stats` — Database statistics

pub mod dashboard;
pub mod routes;
pub mod state;

use std::net::SocketAddr;
use std::sync::Arc;

/// Start the HTTP server on the given address.
pub async fn serve(db_path: &str, host: &str, port: u16) -> Result<(), Box<dyn std::error::Error>> {
    let app_state = state::AppState::new(db_path)?;

    let app = routes::create_router(Arc::new(app_state));

    let addr: SocketAddr = format!("{}:{}", host, port).parse()?;
    println!("EmbedDB Dashboard: http://{}", addr);
    println!("API Base: http://{}/api", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
