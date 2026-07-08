//! Collection — a named set of vectors with persistent storage.

use crate::config::{CollectionConfig, Document, SearchHit, SearchQuery};
use crate::error::{Error, Result};
use vexra_index::flat::FlatIndex;
use vexra_index::hnsw::graph::HnswGraph;
use vexra_index::hnsw::HnswConfig;
use vexra_index::{DistanceMetric, SearchResult, VectorIndex};
use vexra_metadata::filter::Filter;
use vexra_metadata::store::MetadataStore;
use vexra_storage::format::PageType;
use vexra_storage::page_cache::PageCache;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Index backend
// ---------------------------------------------------------------------------

/// Supported index backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum IndexType {
    Flat,
    Hnsw,
}

impl Default for IndexType {
    fn default() -> Self { IndexType::Flat }
}

/// Dispatch enum over FlatIndex or HnswGraph.
pub enum IndexBackend {
    Flat(FlatIndex),
    Hnsw(HnswGraph),
}

impl IndexBackend {
    pub fn new(dimension: usize, metric: DistanceMetric, index_type: IndexType) -> Self {
        match index_type {
            IndexType::Flat => IndexBackend::Flat(FlatIndex::new(dimension, metric)),
            IndexType::Hnsw => IndexBackend::Hnsw(HnswGraph::new(
                dimension, metric, HnswConfig::default(),
            )),
        }
    }

    pub fn get_vector(&self, id: u64) -> Option<Vec<f32>> {
        match self {
            IndexBackend::Flat(idx) => idx.find_idx(id)
                .and_then(|pos| idx.get_by_idx(pos))
                .map(|(_, v)| v.clone()),
            IndexBackend::Hnsw(graph) => graph.get_node(id).map(|n| n.vector.clone()),
        }
    }
}

impl VectorIndex for IndexBackend {
    fn search(&self, query: &[f32], k: usize) -> std::result::Result<Vec<SearchResult>, vexra_index::IndexError> {
        match self {
            IndexBackend::Flat(idx) => idx.search(query, k),
            IndexBackend::Hnsw(graph) => graph.search(query, k),
        }
    }
    fn insert(&mut self, id: u64, vector: &[f32]) -> std::result::Result<(), vexra_index::IndexError> {
        match self {
            IndexBackend::Flat(idx) => idx.insert(id, vector),
            IndexBackend::Hnsw(graph) => graph.insert(id, vector),
        }
    }
    fn remove(&mut self, id: u64) -> std::result::Result<(), vexra_index::IndexError> {
        match self {
            IndexBackend::Flat(idx) => idx.remove(id),
            IndexBackend::Hnsw(graph) => graph.remove(id),
        }
    }
    fn len(&self) -> usize {
        match self {
            IndexBackend::Flat(idx) => idx.len(),
            IndexBackend::Hnsw(graph) => graph.len(),
        }
    }
}

// ---------------------------------------------------------------------------
// Vector persistence helpers
// ---------------------------------------------------------------------------

/// Serialized form: [id_len:u32][doc_id:bytes][internal_id:u64][f32_count:u32][vector:bytes]
fn encode_vector_cell(doc_id: &str, internal_id: u64, vector: &[f32]) -> Vec<u8> {
    let id_bytes = doc_id.as_bytes();
    let mut buf = Vec::with_capacity(4 + id_bytes.len() + 8 + 4 + vector.len() * 4);
    buf.extend_from_slice(&(id_bytes.len() as u32).to_le_bytes());
    buf.extend_from_slice(id_bytes);
    buf.extend_from_slice(&internal_id.to_le_bytes());
    buf.extend_from_slice(&(vector.len() as u32).to_le_bytes());
    for &v in vector {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    buf
}

fn decode_vector_cell(data: &[u8]) -> Option<(String, u64, Vec<f32>)> {
    if data.len() < 12 { return None; }
    let id_len = u32::from_le_bytes([data[0],data[1],data[2],data[3]]) as usize;
    if data.len() < 4 + id_len + 8 + 4 { return None; }
    let doc_id = std::str::from_utf8(&data[4..4+id_len]).ok()?.to_string();
    let pos = 4 + id_len;
    let internal_id = u64::from_le_bytes([
        data[pos],data[pos+1],data[pos+2],data[pos+3],
        data[pos+4],data[pos+5],data[pos+6],data[pos+7],
    ]);
    let pos = pos + 8;
    let count = u32::from_le_bytes([data[pos],data[pos+1],data[pos+2],data[pos+3]]) as usize;
    let pos = pos + 4;
    let mut vector = Vec::with_capacity(count);
    for i in 0..count {
        let off = pos + i * 4;
        if off + 4 > data.len() { return None; }
        vector.push(f32::from_le_bytes([data[off],data[off+1],data[off+2],data[off+3]]));
    }
    Some((doc_id, internal_id, vector))
}

// ---------------------------------------------------------------------------
// Metadata persistence helpers
// ---------------------------------------------------------------------------

fn encode_metadata_cell(id: &str, metadata: &serde_json::Value) -> Vec<u8> {
    let mut buf = Vec::new();
    let id_bytes = id.as_bytes();
    buf.extend_from_slice(&(id_bytes.len() as u32).to_le_bytes());
    buf.extend_from_slice(id_bytes);
    let json_bytes = serde_json::to_vec(metadata).unwrap_or_default();
    buf.extend_from_slice(&json_bytes);
    buf
}

/// Create a tombstone cell (zero-length payload) marking a deleted document.
fn make_tombstone_cell(id: &str) -> Vec<u8> {
    let id_bytes = id.as_bytes();
    let mut buf = Vec::with_capacity(4 + id_bytes.len());
    buf.extend_from_slice(&(id_bytes.len() as u32).to_le_bytes());
    buf.extend_from_slice(id_bytes);
    buf
}

/// Check if a cell is a tombstone (payload matches id prefix with no extra data).
fn is_tombstone(data: &[u8], id_len: usize) -> bool {
    data.len() == 4 + id_len
}

fn decode_metadata_cell(data: &[u8]) -> Option<(String, serde_json::Value)> {
    if data.len() < 4 { return None; }
    let id_len = u32::from_le_bytes([data[0],data[1],data[2],data[3]]) as usize;
    if data.len() < 4 + id_len { return None; }
    let id = std::str::from_utf8(&data[4..4+id_len]).ok()?.to_string();
    let meta: serde_json::Value = serde_json::from_slice(&data[4+id_len..]).ok()?;
    Some((id, meta))
}

// ---------------------------------------------------------------------------
// Collection
// ---------------------------------------------------------------------------

static NEXT_DOC_ID: AtomicU64 = AtomicU64::new(1);
fn next_doc_id() -> u64 { NEXT_DOC_ID.fetch_add(1, Ordering::Relaxed) }
/// Reset the counter to avoid ID collisions with persisted data (called after loading).
pub fn reset_doc_id_counter(start: u64) { NEXT_DOC_ID.store(start, Ordering::Relaxed); }

pub struct Collection {
    config: CollectionConfig,
    index: IndexBackend,
    metadata: MetadataStore,
    id_map: HashMap<String, u64>,
    reverse_id_map: HashMap<u64, String>,
    /// Optional page cache for persistence. None = in-memory only.
    page_cache: Option<Arc<PageCache>>,
}

impl Collection {
    /// Collect all tombstone document IDs from a page.
    fn collect_tombstones(&self, pg: &vexra_storage::page_cache::CachedPage) -> std::collections::HashSet<String> {
        let mut set = std::collections::HashSet::new();
        let header = pg.header();
        for i in 0..header.num_cells {
            if let Some(data) = pg.cell_data(i) {
                if data.len() >= 4 {
                    let id_len = u32::from_le_bytes([data[0],data[1],data[2],data[3]]) as usize;
                    if is_tombstone(&data, id_len) {
                        if let Ok(id) = std::str::from_utf8(&data[4..4+id_len]) {
                            set.insert(id.to_string());
                        }
                    }
                }
            }
        }
        set
    }

    /// Create a new... (continues below)
    /// Create a new in-memory collection (legacy / test use).
    pub fn new(config: CollectionConfig) -> Self {
        Self::with_index(config, IndexType::default())
    }

    pub fn with_index(config: CollectionConfig, index_type: IndexType) -> Self {
        let dimension = config.dimension;
        let distance = config.distance;
        let name = config.name.clone();
        Self {
            config,
            index: IndexBackend::new(dimension, distance, index_type),
            metadata: MetadataStore::new(name),
            id_map: HashMap::new(),
            reverse_id_map: HashMap::new(),
            page_cache: None,
        }
    }

    /// Create a collection backed by persistent storage.
    pub fn new_persistent(
        mut config: CollectionConfig,
        index_type: IndexType,
        page_cache: Arc<PageCache>,
    ) -> Result<Self> {
        let dimension = config.dimension;
        let distance = config.distance;
        let name = config.name.clone();

        // Ensure data page exists
        if config.data_root_page == 0 {
            config.data_root_page = page_cache
                .allocate_page(PageType::Vector, 0)
                .map_err(Error::Storage)?;
        }

        // Ensure metadata page exists
        if config.metadata_root_page == 0 {
            config.metadata_root_page = page_cache
                .allocate_page(PageType::Metadata, 0)
                .map_err(Error::Storage)?;
        }

        // Ensure HNSW edge page exists for HNSW collections
        let hnsw_edge_page = if index_type == IndexType::Hnsw {
            if config.hnsw_edge_page == 0 {
                page_cache.allocate_page(PageType::HnswEdge, 0).map_err(Error::Storage)?
            } else { config.hnsw_edge_page }
        } else { 0 };
        config.hnsw_edge_page = hnsw_edge_page;

        let pc = page_cache.clone();
        let mut collection = Self {
            config,
            index: IndexBackend::new(dimension, distance, index_type),
            metadata: MetadataStore::new(name),
            id_map: HashMap::new(),
            reverse_id_map: HashMap::new(),
            page_cache: Some(pc.clone()),
        };

        // Load existing vectors and metadata
        collection.load_vectors()?;
        collection.load_metadata()?;

        // Load HNSW graph edges if applicable
        if index_type == IndexType::Hnsw && hnsw_edge_page != 0 {
            if let IndexBackend::Hnsw(ref mut graph) = collection.index {
                *graph = vexra_index::hnsw::graph::HnswGraph::load_from_page(
                    &pc, hnsw_edge_page, dimension, distance,
                    vexra_index::hnsw::HnswConfig::default(),
                ).map_err(|e| Error::Other(e))?;
            }
        }

        Ok(collection)
    }

    /// Load vectors from the data page into the index.
    fn load_vectors(&mut self) -> Result<()> {
        let pc = match self.page_cache.as_ref() {
            Some(pc) => pc,
            None => return Ok(()),
        };
        if self.config.data_root_page == 0 { return Ok(()); }

        let page = pc.get_page(self.config.data_root_page).map_err(Error::Storage)?;
        let pg = page.read();
        let header = pg.header();

        // Track tombstones to skip deleted documents
        let tombstones = self.collect_tombstones(&pg);

        for i in 0..header.num_cells {
            if let Some(data) = pg.cell_data(i) {
                // Skip tombstone cells (zero-length payload beyond id prefix)
                if data.len() >= 4 {
                    let id_len = u32::from_le_bytes([data[0],data[1],data[2],data[3]]) as usize;
                    if is_tombstone(&data, id_len) { continue; }
                }
                if let Some((doc_id, internal_id, vector)) = decode_vector_cell(&data) {
                    if tombstones.contains(&doc_id) { continue; }
                    let _ = self.index.insert(internal_id, &vector);
                    self.id_map.insert(doc_id.clone(), internal_id);
                    self.reverse_id_map.insert(internal_id, doc_id);
                }
            }
        }
        Ok(())
    }

    /// Load metadata from the metadata page.
    fn load_metadata(&mut self) -> Result<()> {
        let pc = match self.page_cache.as_ref() {
            Some(pc) => pc,
            None => return Ok(()),
        };
        if self.config.metadata_root_page == 0 { return Ok(()); }

        let page = pc.get_page(self.config.metadata_root_page).map_err(Error::Storage)?;
        let pg = page.read();
        let header = pg.header();

        for i in 0..header.num_cells {
            if let Some(data) = pg.cell_data(i) {
                if let Some((id, meta)) = decode_metadata_cell(&data) {
                    let _ = self.metadata.insert(&id, meta);
                }
            }
        }
        Ok(())
    }

    /// Get the data root page ID (for updating the catalog config).
    pub fn data_root_page(&self) -> u64 {
        self.config.data_root_page
    }

    /// Get the metadata root page ID.
    pub fn metadata_root_page(&self) -> u64 {
        self.config.metadata_root_page
    }

    /// Get a reference to the config (with updated page IDs).
    pub fn config_snapshot(&self) -> CollectionConfig {
        self.config.clone()
    }

    // --- Public API ---

    pub fn name(&self) -> &str { &self.config.name }
    pub fn dimension(&self) -> usize { self.config.dimension }
    pub fn distance_metric(&self) -> DistanceMetric { self.config.distance }
    pub fn vector_count(&self) -> usize { self.index.len() }
    pub fn config(&self) -> &CollectionConfig { &self.config }

    pub fn insert(&mut self, doc: Document) -> Result<String> {
        let doc_id = doc.id.unwrap_or_else(|| format!("doc_{}", next_doc_id()));

        let internal_id = if let Some(&existing) = self.id_map.get(&doc_id) {
            existing
        } else {
            let new_id = next_doc_id();
            self.id_map.insert(doc_id.clone(), new_id);
            self.reverse_id_map.insert(new_id, doc_id.clone());
            new_id
        };

        // Index vector
        if let Some(ref vector) = doc.vector {
            if vector.len() != self.config.dimension {
                return Err(Error::DimensionMismatch { expected: self.config.dimension, actual: vector.len() });
            }
            self.index.insert(internal_id, vector)?;

            // Persist vector to data page
            if let Some(ref pc) = self.page_cache {
                let cell = encode_vector_cell(&doc_id, internal_id, vector);
                let written = pc
                    .write_cell(self.config.data_root_page, &cell, None)
                    .map_err(Error::Storage)?;
                if !written {
                    return Err(Error::Other("Vector data page full".into()));
                }

                // Save HNSW graph edges if using HNSW
                if let IndexBackend::Hnsw(ref graph) = self.index {
                    if self.config.hnsw_edge_page != 0 {
                        let _ = graph.save_to_page(pc, self.config.hnsw_edge_page);
                    }
                }
            }
        }

        // Store + persist metadata
        if let Some(ref meta) = doc.metadata {
            self.metadata.insert(&doc_id, meta.clone())?;
            if let Some(ref pc) = self.page_cache {
                let cell = encode_metadata_cell(&doc_id, meta);
                let written = pc
                    .write_cell(self.config.metadata_root_page, &cell, None)
                    .map_err(Error::Storage)?;
                if !written {
                    return Err(Error::Other("Metadata page full".into()));
                }
            }
        }

        Ok(doc_id)
    }

    pub fn search(&self, query: SearchQuery) -> Result<Vec<SearchHit>> {
        let query_vector = query.vector
            .ok_or_else(|| Error::InvalidConfig("Query vector is required for search".into()))?;
        if query_vector.len() != self.config.dimension {
            return Err(Error::DimensionMismatch { expected: self.config.dimension, actual: query_vector.len() });
        }

        let filter = query.filter.as_ref()
            .map(|f| Filter::parse(f)).transpose()
            .map_err(|e| Error::Other(format!("Invalid filter: {}", e)))?;

        let raw_results = self.index.search(&query_vector, query.top_k)?;

        let mut hits = Vec::new();
        for result in raw_results {
            let doc_id = self.reverse_id_map.get(&result.id)
                .cloned()
                .unwrap_or_else(|| format!("unknown_{}", result.id));

            if let Some(ref filter) = filter {
                if let Some(entry) = self.metadata.get(&doc_id) {
                    if !filter.evaluate(&entry.data) { continue; }
                } else {
                    continue;
                }
            }

            let metadata = if query.include_metadata {
                self.metadata.get(&doc_id).map(|e| e.data.clone())
            } else { None };

            let vector = if query.include_vectors {
                self.index.get_vector(result.id)
            } else { None };

            hits.push(SearchHit { id: doc_id, score: result.score, vector, metadata });
        }
        Ok(hits)
    }

    pub fn delete(&mut self, id: &str) -> Result<()> {
        if let Some(&internal_id) = self.id_map.get(id) {
            let _ = self.index.remove(internal_id);
            self.reverse_id_map.remove(&internal_id);

            // Write zero-length tombstone cell to mark deletion on disk
            if let Some(ref pc) = self.page_cache {
                let tombstone = make_tombstone_cell(id);
                let _ = pc.write_cell(self.config.data_root_page, &tombstone, None);
            }
        }
        self.id_map.remove(id);
        let _ = self.metadata.remove(id);

        // Write metadata tombstone
        if let Some(ref pc) = self.page_cache {
            if self.config.metadata_root_page != 0 {
                let tombstone = make_tombstone_cell(id);
                let _ = pc.write_cell(self.config.metadata_root_page, &tombstone, None);
            }
        }
        Ok(())
    }

    pub fn get_metadata(&self, id: &str) -> Option<serde_json::Value> {
        self.metadata.get(id).map(|e| e.data.clone())
    }

    pub fn list_ids(&self) -> Vec<&str> {
        self.metadata.all_ids()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn make_collection() -> Collection {
        Collection::new(CollectionConfig::new("test", 3).with_distance(DistanceMetric::Euclidean))
    }

    fn make_hnsw_collection() -> Collection {
        Collection::with_index(
            CollectionConfig::new("test_hnsw", 3).with_distance(DistanceMetric::Euclidean),
            IndexType::Hnsw,
        )
    }

    #[test]
    fn test_vector_persistence() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.embeddb");
        let pc = Arc::new(PageCache::open(&path, Default::default()).unwrap());

        let config = CollectionConfig::new("persist", 3).with_distance(DistanceMetric::Euclidean);
        let saved_config;
        {
            let mut col = Collection::new_persistent(config, IndexType::Flat, pc.clone()).unwrap();
            col.insert(Document::with_vector("a", vec![1.0, 0.0, 0.0])).unwrap();
            col.insert(Document::with_vector("b", vec![0.0, 1.0, 0.0])).unwrap();
            col.insert(Document::with_vector_and_metadata("c", vec![0.0, 0.0, 1.0], json!({"key":"val"}))).unwrap();
            saved_config = col.config_snapshot();
        }

        // Reopen using saved config with correct page IDs
        let col = Collection::new_persistent(saved_config, IndexType::Flat, pc).unwrap();
        assert_eq!(col.vector_count(), 3);
        let results = col.search(SearchQuery::with_vector(vec![1.0, 0.0, 0.0], 3)).unwrap();
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].id, "a");

        // Check metadata survived
        let m = col.get_metadata("c");
        assert!(m.is_some());
        assert_eq!(m.unwrap()["key"], "val");
    }

    #[test]
    fn test_insert_and_search_flat() { /* ... existing tests unchanged ... */
        let mut col = make_collection();
        col.insert(Document::with_vector("a", vec![1.0, 0.0, 0.0])).unwrap();
        col.insert(Document::with_vector("b", vec![0.0, 1.0, 0.0])).unwrap();
        col.insert(Document::with_vector("c", vec![0.0, 0.0, 1.0])).unwrap();
        let results = col.search(SearchQuery::with_vector(vec![1.0, 0.0, 0.0], 3)).unwrap();
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].id, "a");
    }

    #[test]
    fn test_insert_and_search_hnsw() {
        let mut col = make_hnsw_collection();
        col.insert(Document::with_vector("a", vec![1.0, 0.0, 0.0])).unwrap();
        col.insert(Document::with_vector("b", vec![0.0, 1.0, 0.0])).unwrap();
        col.insert(Document::with_vector("c", vec![0.0, 0.0, 1.0])).unwrap();
        let results = col.search(SearchQuery::with_vector(vec![1.0, 0.0, 0.0], 3)).unwrap();
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].id, "a");
    }

    #[test]
    fn test_insert_with_metadata_and_filter() {
        let mut col = make_collection();
        col.insert(Document::with_vector_and_metadata("a", vec![1.0,0.0,0.0], json!({"category":"tech","score":10}))).unwrap();
        col.insert(Document::with_vector_and_metadata("b", vec![0.0,1.0,0.0], json!({"category":"science","score":5}))).unwrap();
        let results = col.search(SearchQuery::with_vector(vec![1.0,0.0,0.0], 3).with_filter(r#"category = "tech""#)).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "a");
    }

    #[test]
    fn test_delete() {
        let mut col = make_collection();
        col.insert(Document::with_vector("a", vec![1.0,0.0,0.0])).unwrap();
        assert_eq!(col.vector_count(), 1);
        col.delete("a").unwrap();
        assert_eq!(col.vector_count(), 0);
        assert!(col.search(SearchQuery::with_vector(vec![1.0,0.0,0.0],1)).unwrap().is_empty());
    }

    #[test]
    fn test_dimension_mismatch() {
        let mut col = make_collection();
        assert!(matches!(col.insert(Document::with_vector("a", vec![1.0,0.0])).unwrap_err(), Error::DimensionMismatch{..}));
    }
}
