//! Shared application state for the HTTP server.
//!
//! # Note on synchronization
//!
//! Uses `parking_lot::RwLock` rather than `tokio::sync::RwLock` because the
//! underlying storage engine (page cache, WAL) internally uses parking_lot
//! primitives. Under typical read-heavy workloads, contention is minimal.
//! For high-write-load deployments, consider wrapping disk I/O calls in
//! `tokio::task::spawn_blocking`.

use vexra_core::db::Database;
use parking_lot::RwLock;

/// Shared application state for all HTTP handlers.
pub struct AppState {
    pub db: RwLock<Database>,
}

// Safety: Database is Send+Sync (all internal fields use Arc, parking_lot locks).
unsafe impl Send for AppState {}
unsafe impl Sync for AppState {}

impl AppState {
    /// Create a new AppState by opening the database at the given path.
    pub fn new(db_path: &str) -> Result<Self, vexra_core::Error> {
        let db = Database::open(db_path)?;
        Ok(Self {
            db: RwLock::new(db),
        })
    }
}
