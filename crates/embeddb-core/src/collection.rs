//! Collection — a named set of vectors with an associated index and metadata.

use crate::config::{CollectionConfig, Document, SearchHit, SearchQuery};
use crate::error::{Error, Result};
use embeddb_index::flat::FlatIndex;
use embeddb_index::hnsw::graph::HnswGraph;
use embeddb_index::hnsw::HnswConfig;
use embeddb_index::{DistanceMetric, SearchResult, VectorIndex};
use embeddb_metadata::filter::Filter;
use embeddb_metadata::store::MetadataStore;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

// ---------------------------------------------------------------------------
// Index backend — enum dispatching over available index types
// ---------------------------------------------------------------------------

/// Supported index backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum IndexType {
    /// Exact brute-force search. Best for small datasets (< 10k vectors).
    Flat,
    /// Approximate nearest neighbor via HNSW. 10-100x faster at scale.
    Hnsw,
}

impl Default for IndexType {
    fn default() -> Self {
        IndexType::Flat
    }
}

/// Dispatch enum that wraps either a FlatIndex or an HnswGraph.
pub enum IndexBackend {
    Flat(FlatIndex),
    Hnsw(HnswGraph),
}

impl IndexBackend {
    /// Create a new index of the given type.
    pub fn new(dimension: usize, metric: DistanceMetric, index_type: IndexType) -> Self {
        match index_type {
            IndexType::Flat => IndexBackend::Flat(FlatIndex::new(dimension, metric)),
            IndexType::Hnsw => IndexBackend::Hnsw(HnswGraph::new(
                dimension, metric, HnswConfig::default(),
            )),
        }
    }

    /// Get the vector for a given internal ID (if the index stores it).
    pub fn get_vector(&self, id: u64) -> Option<Vec<f32>> {
        match self {
            IndexBackend::Flat(idx) => idx
                .find_idx(id)
                .and_then(|pos| idx.get_by_idx(pos))
                .map(|(_, v)| v.clone()),
            IndexBackend::Hnsw(graph) => graph
                .get_node(id)
                .map(|n| n.vector.clone()),
        }
    }
}

impl VectorIndex for IndexBackend {
    fn search(&self, query: &[f32], k: usize) -> std::result::Result<Vec<SearchResult>, embeddb_index::IndexError> {
        match self {
            IndexBackend::Flat(idx) => idx.search(query, k),
            IndexBackend::Hnsw(graph) => graph.search(query, k),
        }
    }

    fn insert(&mut self, id: u64, vector: &[f32]) -> std::result::Result<(), embeddb_index::IndexError> {
        match self {
            IndexBackend::Flat(idx) => idx.insert(id, vector),
            IndexBackend::Hnsw(graph) => graph.insert(id, vector),
        }
    }

    fn remove(&mut self, id: u64) -> std::result::Result<(), embeddb_index::IndexError> {
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
// Collection
// ---------------------------------------------------------------------------

/// Internal ID counter for auto-generating document IDs.
static NEXT_DOC_ID: AtomicU64 = AtomicU64::new(1);

fn next_doc_id() -> u64 {
    NEXT_DOC_ID.fetch_add(1, Ordering::Relaxed)
}

/// A collection of vectors.
pub struct Collection {
    /// Collection configuration.
    config: CollectionConfig,
    /// Vector index (flat or HNSW).
    index: IndexBackend,
    /// Metadata store.
    metadata: MetadataStore,
    /// Mapping from string document IDs to internal u64 IDs.
    id_map: HashMap<String, u64>,
    /// Reverse mapping from u64 IDs to string document IDs.
    reverse_id_map: HashMap<u64, String>,
}

impl Collection {
    /// Create a new collection with the default index type (Flat).
    pub fn new(config: CollectionConfig) -> Self {
        Self::with_index(config, IndexType::default())
    }

    /// Create a new collection with a specific index type.
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
        }
    }

    /// Get the collection name.
    pub fn name(&self) -> &str {
        &self.config.name
    }

    /// Get the vector dimension.
    pub fn dimension(&self) -> usize {
        self.config.dimension
    }

    /// Get the distance metric.
    pub fn distance_metric(&self) -> DistanceMetric {
        self.config.distance
    }

    /// Get the number of vectors in the collection.
    pub fn vector_count(&self) -> usize {
        self.index.len()
    }

    /// Get the collection configuration.
    pub fn config(&self) -> &CollectionConfig {
        &self.config
    }

    /// Insert a document into the collection.
    ///
    /// If the document has a vector, it will be indexed for search.
    /// If the document has metadata, it will be stored for filtering.
    pub fn insert(&mut self, doc: Document) -> Result<String> {
        let doc_id = doc
            .id
            .unwrap_or_else(|| format!("doc_{}", next_doc_id()));

        // Map string ID to internal u64 ID
        let internal_id = if let Some(&existing) = self.id_map.get(&doc_id) {
            existing
        } else {
            let new_id = next_doc_id();
            self.id_map.insert(doc_id.clone(), new_id);
            self.reverse_id_map.insert(new_id, doc_id.clone());
            new_id
        };

        // Index the vector if provided
        if let Some(ref vector) = doc.vector {
            if vector.len() != self.config.dimension {
                return Err(Error::DimensionMismatch {
                    expected: self.config.dimension,
                    actual: vector.len(),
                });
            }
            self.index.insert(internal_id, vector)?;
        }

        // Store metadata if provided
        if let Some(meta) = doc.metadata {
            self.metadata.insert(&doc_id, meta)?;
        }

        Ok(doc_id)
    }

    /// Search for similar vectors.
    pub fn search(&self, query: SearchQuery) -> Result<Vec<SearchHit>> {
        let query_vector = query
            .vector
            .ok_or_else(|| Error::InvalidConfig("Query vector is required for search".into()))?;

        if query_vector.len() != self.config.dimension {
            return Err(Error::DimensionMismatch {
                expected: self.config.dimension,
                actual: query_vector.len(),
            });
        }

        // Parse filter if provided
        let filter = query
            .filter
            .as_ref()
            .map(|f| Filter::parse(f))
            .transpose()
            .map_err(|e| Error::Other(format!("Invalid filter: {}", e)))?;

        // Perform vector search
        let raw_results = self.index.search(&query_vector, query.top_k)?;

        // Convert to SearchHit, applying metadata filters
        let mut hits = Vec::new();
        for result in raw_results {
            let doc_id = self
                .reverse_id_map
                .get(&result.id)
                .cloned()
                .unwrap_or_else(|| format!("unknown_{}", result.id));

            // Apply metadata filter
            if let Some(ref filter) = filter {
                if let Some(entry) = self.metadata.get(&doc_id) {
                    if !filter.evaluate(&entry.data) {
                        continue;
                    }
                } else {
                    continue;
                }
            }

            let metadata = if query.include_metadata {
                self.metadata.get(&doc_id).map(|e| e.data.clone())
            } else {
                None
            };

            let vector = if query.include_vectors {
                self.index.get_vector(result.id)
            } else {
                None
            };

            hits.push(SearchHit {
                id: doc_id,
                score: result.score,
                vector,
                metadata,
            });
        }

        Ok(hits)
    }

    /// Delete a document from the collection.
    pub fn delete(&mut self, id: &str) -> Result<()> {
        if let Some(&internal_id) = self.id_map.get(id) {
            let _ = self.index.remove(internal_id);
            self.reverse_id_map.remove(&internal_id);
        }
        self.id_map.remove(id);
        let _ = self.metadata.remove(id);
        Ok(())
    }

    /// Get metadata for a document.
    pub fn get_metadata(&self, id: &str) -> Option<serde_json::Value> {
        self.metadata.get(id).map(|e| e.data.clone())
    }

    /// List all document IDs in the collection.
    pub fn list_ids(&self) -> Vec<&str> {
        self.metadata.all_ids()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn make_collection() -> Collection {
        let config = CollectionConfig::new("test", 3).with_distance(DistanceMetric::Euclidean);
        Collection::new(config)
    }

    fn make_hnsw_collection() -> Collection {
        let config = CollectionConfig::new("test_hnsw", 3).with_distance(DistanceMetric::Euclidean);
        Collection::with_index(config, IndexType::Hnsw)
    }

    #[test]
    fn test_insert_and_search_flat() {
        let mut col = make_collection();

        col.insert(Document::with_vector("a", vec![1.0, 0.0, 0.0])).unwrap();
        col.insert(Document::with_vector("b", vec![0.0, 1.0, 0.0])).unwrap();
        col.insert(Document::with_vector("c", vec![0.0, 0.0, 1.0])).unwrap();

        let results = col.search(SearchQuery::with_vector(vec![1.0, 0.0, 0.0], 3)).unwrap();
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].id, "a");
        assert!((results[0].score - 0.0).abs() < 1e-6);
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
        assert!((results[0].score - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_insert_with_metadata_and_filter() {
        let mut col = make_collection();

        col.insert(Document::with_vector_and_metadata(
            "a", vec![1.0, 0.0, 0.0], json!({"category": "tech", "score": 10}),
        )).unwrap();

        col.insert(Document::with_vector_and_metadata(
            "b", vec![0.0, 1.0, 0.0], json!({"category": "science", "score": 5}),
        )).unwrap();

        let results = col.search(
            SearchQuery::with_vector(vec![1.0, 0.0, 0.0], 3)
                .with_filter(r#"category = "tech""#),
        ).unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "a");
        assert_eq!(results[0].metadata.as_ref().unwrap()["score"], 10);
    }

    #[test]
    fn test_delete() {
        let mut col = make_collection();
        col.insert(Document::with_vector("a", vec![1.0, 0.0, 0.0])).unwrap();
        assert_eq!(col.vector_count(), 1);

        col.delete("a").unwrap();
        assert_eq!(col.vector_count(), 0);

        let results = col.search(SearchQuery::with_vector(vec![1.0, 0.0, 0.0], 1)).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_dimension_mismatch() {
        let mut col = make_collection();
        let err = col.insert(Document::with_vector("a", vec![1.0, 0.0])).unwrap_err();
        assert!(matches!(err, Error::DimensionMismatch { .. }));
    }
}
