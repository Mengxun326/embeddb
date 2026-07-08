//! Shared application state for the HTTP server.

use embeddb_core::db::Database;
use parking_lot::RwLock;

/// Shared application state for all HTTP handlers.
pub struct AppState {
    pub db: RwLock<Database>,
}

unsafe impl Send for AppState {}
unsafe impl Sync for AppState {}

impl AppState {
    /// Create a new AppState by opening the database at the given path.
    pub fn new(db_path: &str) -> Result<Self, embeddb_core::Error> {
        let db = Database::open(db_path)?;
        Ok(Self {
            db: RwLock::new(db),
        })
    }
}
