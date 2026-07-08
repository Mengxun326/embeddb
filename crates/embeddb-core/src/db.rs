//! Database — the top-level handle for an EmbedDB database.
//!
//! A Database manages collections and coordinates the storage engine,
//! indexes, and metadata stores. Collections are persisted to the
//! catalog page so they survive process restarts.

use crate::collection::{Collection, IndexType};
use crate::config::{CollectionConfig, CollectionStats, DatabaseConfig, DatabaseStats, Document, SearchHit, SearchQuery};
use crate::error::{Error, Result};
use embeddb_storage::format::PageType;
use embeddb_storage::page_cache::{CacheConfig, PageCache};
use embeddb_storage::wal::WalManager;
use parking_lot::{Mutex, RwLock};
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
    /// Catalog mutex — prevents TOCTOU races during catalog page allocation.
    catalog_lock: Mutex<()>,
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
                log::warn!("WAL recovery failed: {}", e);
            }
        }

        // Always have an open WAL for crash safety
        wal.open().map_err(|e| Error::Other(format!("Failed to open WAL: {}", e)))?;

        let wal = Arc::new(wal);

        let db = Self {
            path,
            page_cache,
            wal,
            collections: RwLock::new(HashMap::new()),
            catalog_lock: Mutex::new(()),
            config,
        };

        // Load persisted collections from catalog
        db.load_catalog()?;
        init_id_counter(&db);

        Ok(db)
    }

    // ------------------------------------------------------------------
    // Collection management
    // ------------------------------------------------------------------

    /// Create a new collection with persistent storage.
    pub fn create_collection(&self, config: CollectionConfig) -> Result<()> {
        let mut collections = self.collections.write();

        if collections.contains_key(&config.name) {
            return Err(Error::CollectionAlreadyExists(config.name));
        }

        // Create with persistent storage
        let collection = Collection::new_persistent(
            config.clone(), IndexType::Flat, self.page_cache.clone(),
        )?;

        // Get updated config with allocated page IDs
        let updated_config = collection.config_snapshot();
        collections.insert(
            updated_config.name.clone(),
            Arc::new(RwLock::new(collection)),
        );

        // Persist the updated config (with page IDs) to catalog
        drop(collections);
        self.persist_collection(&updated_config)
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
        drop(collections);
        self.remove_collection_from_catalog(name)
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
                metadata_count: 0,
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
        self.page_cache.flush_all().map_err(Error::Storage)?;

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

    // ------------------------------------------------------------------
    // Catalog persistence
    // ------------------------------------------------------------------

    /// Ensure the catalog page exists (allocate if first collection).
    /// Protected by catalog_lock to prevent TOCTOU race.
    fn ensure_catalog_page(&self) -> Result<u64> {
        let _guard = self.catalog_lock.lock();

        // Double-check under lock
        let header = self.page_cache.db_header().map_err(Error::Storage)?;
        if header.catalog_root_page != 0 {
            return Ok(header.catalog_root_page);
        }

        // Allocate a new catalog page
        let page_id = self
            .page_cache
            .allocate_page(PageType::Catalog, 0)
            .map_err(Error::Storage)?;

        // Update DbHeader with catalog root
        let mut header = self.page_cache.db_header().map_err(Error::Storage)?;
        header.catalog_root_page = page_id;
        self.page_cache
            .update_db_header(&header)
            .map_err(Error::Storage)?;

        Ok(page_id)
    }

    /// Persist a single collection config to the catalog page.
    fn persist_collection(&self, config: &CollectionConfig) -> Result<()> {
        let catalog_page = self.ensure_catalog_page()?;

        // Serialize config to JSON
        let json = serde_json::to_vec(config).map_err(|e| {
            Error::Other(format!("Failed to serialize collection config: {}", e))
        })?;

        // Write as a cell on the catalog page; check the bool return value
        let written = self
            .page_cache
            .write_cell(catalog_page, &json, None)
            .map_err(Error::Storage)?;
        if !written {
            return Err(Error::Other("Catalog page full — cannot persist collection".into()));
        }

        self.page_cache.flush_page(catalog_page).map_err(Error::Storage)?;

        Ok(())
    }

    // TODO: Page leak — this allocates a new catalog page without freeing the old one.
    // Phase 3 should add a page free-list or in-place cell deletion to reclaim space.

    /// Remove a collection from the catalog by rebuilding the catalog without it.
    fn remove_collection_from_catalog(&self, name: &str) -> Result<()> {
        let header = self.page_cache.db_header().map_err(Error::Storage)?;
        if header.catalog_root_page == 0 {
            return Ok(());
        }

        // Read existing configurations
        let configs = self.read_catalog_configs(header.catalog_root_page)?;

        // Filter out the removed collection
        let remaining: Vec<&CollectionConfig> = configs
            .iter()
            .filter(|c| c.name != name)
            .collect();

        // Allocate a new catalog page
        let new_page = self
            .page_cache
            .allocate_page(PageType::Catalog, 0)
            .map_err(Error::Storage)?;

        // Write remaining configs
        for config in remaining {
            let json = serde_json::to_vec(config).map_err(|e| {
                Error::Other(format!("Failed to serialize: {}", e))
            })?;

            let written = self
                .page_cache
                .write_cell(new_page, &json, None)
                .map_err(Error::Storage)?;
            if !written {
                return Err(Error::Other("Catalog page full — cannot persist collection".into()));
            }
        }

        // Update DbHeader
        let mut header = self.page_cache.db_header().map_err(Error::Storage)?;
        header.catalog_root_page = new_page;
        self.page_cache
            .update_db_header(&header)
            .map_err(Error::Storage)?;

        self.page_cache.flush_page(new_page).map_err(Error::Storage)?;

        Ok(())
    }

    /// Load all collections from the catalog page (with persistent data).
    fn load_catalog(&self) -> Result<()> {
        let header = self.page_cache.db_header().map_err(Error::Storage)?;
        if header.catalog_root_page == 0 {
            return Ok(()); // No collections yet
        }

        let configs = self.read_catalog_configs(header.catalog_root_page)?;

        let mut collections = self.collections.write();
        for config in configs {
            let collection = Collection::new_persistent(
                config.clone(), IndexType::Flat, self.page_cache.clone(),
            )?;
            collections.insert(
                config.name.clone(),
                Arc::new(RwLock::new(collection)),
            );
        }

        Ok(())
    }

    /// Read all collection configs from a catalog page.
    fn read_catalog_configs(&self, page_id: u64) -> Result<Vec<CollectionConfig>> {
        let page = self.page_cache.get_page(page_id).map_err(Error::Storage)?;
        let pg = page.read();
        let header = pg.header();

        let mut configs = Vec::new();
        for i in 0..header.num_cells {
            if let Some(data) = pg.cell_data(i) {
                if let Ok(config) = serde_json::from_slice::<CollectionConfig>(&data) {
                    configs.push(config);
                }
            }
        }

        Ok(configs)
    }
}

/// Initialize the auto-increment ID counter from the database state.
/// Called after loading all collections to ensure IDs don't collide.
fn init_id_counter(db: &Database) {
    let collections = db.collections.read();
    let max_id: u64 = collections.values()
        .map(|c| c.read().vector_count() as u64)
        .max()
        .unwrap_or(0);
    // Start IDs well past the max loaded vector count
    crate::collection::reset_doc_id_counter(max_id + 1000);
    drop(collections);
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
    fn test_collection_persistence() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.embeddb");

        // Create a DB, add collection
        {
            let db = Database::open(&path).unwrap();
            db.create_collection(CollectionConfig::new("persistent", 128)).unwrap();
            db.close().unwrap();
        }

        // Reopen and verify collection still exists
        {
            let db = Database::open(&path).unwrap();
            assert!(db.collection_exists("persistent"));

            let col = db.get_collection("persistent").unwrap();
            let col = col.read();
            assert_eq!(col.name(), "persistent");
            assert_eq!(col.dimension(), 128);
        }
    }

    #[test]
    fn test_drop_collection_persists() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.embeddb");

        {
            let db = Database::open(&path).unwrap();
            db.create_collection(CollectionConfig::new("temp", 3)).unwrap();
            db.create_collection(CollectionConfig::new("keep", 3)).unwrap();
            db.drop_collection("temp").unwrap();
            db.close().unwrap();
        }

        {
            let db = Database::open(&path).unwrap();
            assert!(!db.collection_exists("temp"));
            assert!(db.collection_exists("keep"));
        }
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
