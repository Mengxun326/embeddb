//! In-memory metadata store for document metadata.
//!
//! Phase 0: Simple in-memory HashMap-based storage.
//! Phase 3: Persistent storage with inverted indexes for filtering.

use crate::error::{MetadataError, Result};
use serde_json::Value as JsonValue;
use std::collections::HashMap;

/// A metadata entry associated with a document.
#[derive(Debug, Clone)]
pub struct MetadataEntry {
    /// Document ID.
    pub id: String,
    /// JSON metadata.
    pub data: JsonValue,
}

/// Simple in-memory metadata store.
pub struct MetadataStore {
    /// Map from document ID to metadata entry.
    entries: HashMap<String, MetadataEntry>,
    /// Collection name.
    collection_name: String,
}

impl MetadataStore {
    /// Create a new empty metadata store.
    pub fn new(collection_name: impl Into<String>) -> Self {
        Self {
            entries: HashMap::new(),
            collection_name: collection_name.into(),
        }
    }

    /// Insert or update metadata for a document.
    pub fn insert(&mut self, id: impl Into<String>, metadata: JsonValue) -> Result<()> {
        let id = id.into();
        self.entries.insert(
            id.clone(),
            MetadataEntry {
                id,
                data: metadata,
            },
        );
        Ok(())
    }

    /// Get metadata for a document.
    pub fn get(&self, id: &str) -> Option<&MetadataEntry> {
        self.entries.get(id)
    }

    /// Remove metadata for a document.
    pub fn remove(&mut self, id: &str) -> Result<()> {
        self.entries
            .remove(id)
            .ok_or_else(|| MetadataError::DocumentNotFound(id.to_string()))?;
        Ok(())
    }

    /// Get all entries.
    pub fn all(&self) -> Vec<&MetadataEntry> {
        self.entries.values().collect()
    }

    /// Get all document IDs.
    pub fn all_ids(&self) -> Vec<&str> {
        self.entries.keys().map(|s| s.as_str()).collect()
    }

    /// Return the number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Return true if the store is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Check if a document exists.
    pub fn contains(&self, id: &str) -> bool {
        self.entries.contains_key(id)
    }

    /// Get the collection name.
    pub fn collection_name(&self) -> &str {
        &self.collection_name
    }

    /// Filter entries by a predicate.
    pub fn filter<F>(&self, predicate: F) -> Vec<&MetadataEntry>
    where
        F: Fn(&MetadataEntry) -> bool,
    {
        self.entries.values().filter(|e| predicate(e)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_insert_and_get() {
        let mut store = MetadataStore::new("test");
        store
            .insert("doc1", json!({"title": "Hello", "year": 2024}))
            .unwrap();

        let entry = store.get("doc1").unwrap();
        assert_eq!(entry.id, "doc1");
        assert_eq!(entry.data["title"], "Hello");
        assert_eq!(entry.data["year"], 2024);
    }

    #[test]
    fn test_remove() {
        let mut store = MetadataStore::new("test");
        store.insert("doc1", json!({})).unwrap();
        store.remove("doc1").unwrap();
        assert!(store.get("doc1").is_none());
    }

    #[test]
    fn test_filter() {
        let mut store = MetadataStore::new("test");
        store.insert("a", json!({"score": 10})).unwrap();
        store.insert("b", json!({"score": 20})).unwrap();
        store.insert("c", json!({"score": 10})).unwrap();

        let filtered = store.filter(|e| e.data["score"] == json!(10));
        assert_eq!(filtered.len(), 2);
    }
}
