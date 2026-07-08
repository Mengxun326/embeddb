//! Database — the top-level handle for an EmbedDB database.
//!
//! A Database manages collections and coordinates the storage engine,
//! indexes, and metadata stores.

use crate::collection::Collection;
use crate::config::{CollectionConfig, CollectionStats, DatabaseConfig, DatabaseStats, Document, SearchHit, SearchQuery};
use crate::error::{Error, Result};
use embeddb_storage::page_cache::{CacheConfig, PageCache};
use embeddb_storage::wal::WalManager;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// The top-level database handle.
///
/// Thread-safe: can be shared across threads (all public methods take `&self`).
pub struct Database {
    /// Path to the database file.
    path: PathBuf,
    /// Page cache for disk I/O.
    page_cache: Arc<PageCache>,
    /// Write-Ahead Log manager.
    wal: Arc<WalManager>,
    /// Collections (name → collection).
    collections: RwLock<HashMap<String, Arc<RwLock<Collection>>>>,
    /// Database configuration.
    /// Database configuration.
    #[allow(dead_code)]
    config: DatabaseConfig,
}

impl Database {
    /// Open an existing database or create a new one at the given path.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::open_with_config(path, DatabaseConfig::default())
    }

    /// Open a database with custom configuration.
    pub fn open_with_config(path: impl AsRef<Path>, config: DatabaseConfig) -> Result<Self> {
        let path = path.as_ref().to_path_buf();

        // Initialize storage
        let page_cache = PageCache::open(
            &path,
            CacheConfig {
                max_pages: config.cache_max_pages,
                page_size: config.page_size,
            },
        ).map_err(Error::Storage)?;

        let page_cache = Arc::new(page_cache);

        // Initialize WAL
        let wal = WalManager::new(&path, config.page_size);

        // Try to recover from WAL if it exists
        if wal.exists() {
            if let Err(e) = wal.recover(&path) {
                // Log the error but continue — the database may still be usable
                log::warn!("WAL recovery failed: {}", e);
            }
        }

        // Always have an open WAL for crash safety
        wal.open().map_err(|e| Error::Other(format!("Failed to open WAL: {}", e)))?;

        let wal = Arc::new(wal);

        Ok(Self {
            path,
            page_cache,
            wal,
            collections: RwLock::new(HashMap::new()),
            config,
        })
    }

    /// Create a new collection.
    pub fn create_collection(&self, config: CollectionConfig) -> Result<()> {
        let mut collections = self.collections.write();

        if collections.contains_key(&config.name) {
            return Err(Error::CollectionAlreadyExists(config.name));
        }

        let collection = Collection::new(config);
        collections.insert(
            collection.name().to_string(),
            Arc::new(RwLock::new(collection)),
        );

        Ok(())
    }

    /// Get a collection by name.
    pub fn get_collection(&self, name: &str) -> Result<Arc<RwLock<Collection>>> {
        let collections = self.collections.read();
        collections
            .get(name)
            .cloned()
            .ok_or_else(|| Error::CollectionNotFound(name.to_string()))
    }

    /// Remove a collection by name.
    pub fn drop_collection(&self, name: &str) -> Result<()> {
        let mut collections = self.collections.write();
        collections
            .remove(name)
            .ok_or_else(|| Error::CollectionNotFound(name.to_string()))?;
        Ok(())
    }

    /// Check if a collection exists.
    pub fn collection_exists(&self, name: &str) -> bool {
        self.collections.read().contains_key(name)
    }

    /// List all collection names.
    pub fn list_collections(&self) -> Vec<String> {
        self.collections.read().keys().cloned().collect()
    }

    /// Get database statistics.
    pub fn stats(&self) -> Result<DatabaseStats> {
        let file_size = std::fs::metadata(&self.path)
            .map(|m| m.len())
            .unwrap_or(0);

        let header = self.page_cache.db_header().map_err(Error::Storage)?;

        let mut collection_stats = Vec::new();
        let collections = self.collections.read();
        for (_, col) in collections.iter() {
            let col = col.read();
            collection_stats.push(CollectionStats {
                name: col.name().to_string(),
                dimension: col.dimension(),
                distance: format!("{:?}", col.distance_metric()),
                vector_count: col.vector_count(),
                metadata_count: 0, // Will be populated when metadata is persisted
            });
        }

        Ok(DatabaseStats {
            path: self.path.display().to_string(),
            file_size,
            page_size: header.page_size,
            page_count: header.page_count,
            collection_count: collections.len(),
            collections: collection_stats,
        })
    }

    /// Close the database, flushing all data to disk.
    pub fn close(&self) -> Result<()> {
        // Flush page cache
        self.page_cache.flush_all().map_err(Error::Storage)?;

        // Checkpoint WAL and remove
        self.wal.checkpoint(&self.path).map_err(|e| {
            Error::Other(format!("WAL checkpoint failed: {}", e))
        })?;
        self.wal.remove().map_err(|e| {
            Error::Other(format!("WAL cleanup failed: {}", e))
        })?;

        Ok(())
    }

    /// Get the database file path.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Get the page cache (for advanced use).
    pub fn page_cache(&self) -> &Arc<PageCache> {
        &self.page_cache
    }
}

/// Convenience helper to insert a document into a collection.
pub fn insert(db: &Database, collection_name: &str, doc: Document) -> Result<String> {
    let col = db.get_collection(collection_name)?;
    let mut col = col.write();
    col.insert(doc)
}

/// Convenience helper to search a collection.
pub fn search(db: &Database, collection_name: &str, query: SearchQuery) -> Result<Vec<SearchHit>> {
    let col = db.get_collection(collection_name)?;
    let col = col.read();
    col.search(query)
}

impl Drop for Database {
    fn drop(&mut self) {
        let _ = self.close();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Document;
    use serde_json::json;

    fn setup_db() -> (tempfile::TempDir, Database) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.embeddb");
        let db = Database::open(&path).unwrap();
        (dir, db)
    }

    #[test]
    fn test_create_and_get_collection() {
        let (_dir, db) = setup_db();

        db.create_collection(CollectionConfig::new("docs", 4))
            .unwrap();
        assert!(db.collection_exists("docs"));

        let col = db.get_collection("docs").unwrap();
        let col = col.read();
        assert_eq!(col.name(), "docs");
        assert_eq!(col.dimension(), 4);
    }

    #[test]
    fn test_duplicate_collection() {
        let (_dir, db) = setup_db();
        db.create_collection(CollectionConfig::new("docs", 4))
            .unwrap();
        let err = db
            .create_collection(CollectionConfig::new("docs", 4))
            .unwrap_err();
        assert!(matches!(err, Error::CollectionAlreadyExists(_)));
    }

    #[test]
    fn test_insert_and_search_via_db() {
        let (_dir, db) = setup_db();
        db.create_collection(CollectionConfig::new("docs", 3))
            .unwrap();

        insert(
            &db,
            "docs",
            Document::with_vector_and_metadata("1", vec![1.0, 0.0, 0.0], json!({"title": "first"})),
        )
        .unwrap();

        insert(
            &db,
            "docs",
            Document::with_vector("2", vec![0.0, 1.0, 0.0]),
        )
        .unwrap();

        let results = search(&db, "docs", SearchQuery::with_vector(vec![1.0, 0.0, 0.0], 2))
            .unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "1");
        assert!((results[0].score - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_drop_collection() {
        let (_dir, db) = setup_db();
        db.create_collection(CollectionConfig::new("docs", 4))
            .unwrap();
        db.drop_collection("docs").unwrap();
        assert!(!db.collection_exists("docs"));
    }

    #[test]
    fn test_stats() {
        let (_dir, db) = setup_db();
        db.create_collection(CollectionConfig::new("docs", 3))
            .unwrap();
        insert(
            &db,
            "docs",
            Document::with_vector("1", vec![1.0, 0.0, 0.0]),
        )
        .unwrap();

        let stats = db.stats().unwrap();
        assert_eq!(stats.collection_count, 1);
        assert_eq!(stats.collections[0].name, "docs");
        assert_eq!(stats.collections[0].vector_count, 1);
    }
}
