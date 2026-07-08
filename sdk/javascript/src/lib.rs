//! EmbedDB JavaScript/Node.js SDK — native napi-rs bindings.

use embeddb_core::collection::IndexType;
use embeddb_core::config::{CollectionConfig, Document, SearchQuery};
use embeddb_core::db::Database;
use embeddb_core::DistanceMetric;
use napi_derive::napi;
use std::sync::Mutex;

#[napi]
pub struct EmbedDb {
    db: Mutex<Option<Database>>,
}

#[napi]
impl EmbedDb {
    #[napi(constructor)]
    pub fn new(path: String) -> napi::Result<Self> {
        let db = Database::open(&path).map_err(|e| napi::Error::from_reason(e.to_string()))?;
        Ok(Self { db: Mutex::new(Some(db)) })
    }

    #[napi]
    pub fn create_collection(&self, name: String, dimension: u32, distance: Option<String>) -> napi::Result<()> {
        let db = self.db.lock().unwrap();
        let db = db.as_ref().ok_or_else(|| napi::Error::from_reason("Database closed"))?;
        let metric = match distance.as_deref() {
            Some("euclidean") => DistanceMetric::Euclidean,
            Some("dot") => DistanceMetric::DotProduct,
            _ => DistanceMetric::Cosine,
        };
        let config = CollectionConfig::new(&name, dimension as usize).with_distance(metric);
        db.create_collection(config).map_err(|e| napi::Error::from_reason(e.to_string()))
    }

    #[napi]
    pub fn insert(
        &self,
        collection: String,
        id: Option<String>,
        vector: Vec<f32>,
        metadata: Option<String>,
    ) -> napi::Result<String> {
        let db = self.db.lock().unwrap();
        let db = db.as_ref().ok_or_else(|| napi::Error::from_reason("Database closed"))?;
        let meta: Option<serde_json::Value> = metadata
            .map(|s| serde_json::from_str(&s).unwrap_or(serde_json::Value::String(s)));
        let doc = Document {
            id,
            vector: Some(vector),
            metadata: meta,
            text: None,
        };
        embeddb_core::insert(db, &collection, doc).map_err(|e| napi::Error::from_reason(e.to_string()))
    }

    #[napi]
    pub fn search(
        &self,
        collection: String,
        vector: Vec<f32>,
        top_k: Option<u32>,
        filter: Option<String>,
    ) -> napi::Result<Vec<SearchResultJs>> {
        let db = self.db.lock().unwrap();
        let db = db.as_ref().ok_or_else(|| napi::Error::from_reason("Database closed"))?;
        let mut query = SearchQuery::with_vector(vector, top_k.unwrap_or(10) as usize);
        if let Some(f) = filter { query = query.with_filter(f); }
        let hits = embeddb_core::search(db, &collection, query)
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;
        Ok(hits.into_iter().map(|h| SearchResultJs {
            id: h.id,
            score: h.score as f64,
            metadata: h.metadata.map(|m| m.to_string()),
        }).collect())
    }

    #[napi]
    pub fn list_collections(&self) -> napi::Result<Vec<String>> {
        let db = self.db.lock().unwrap();
        let db = db.as_ref().ok_or_else(|| napi::Error::from_reason("Database closed"))?;
        Ok(db.list_collections())
    }

    #[napi]
    pub fn close(&self) {
        if let Some(db) = self.db.lock().unwrap().take() {
            let _ = db.close();
        }
    }
}

#[napi(object)]
pub struct SearchResultJs {
    pub id: String,
    pub score: f64,
    pub metadata: Option<String>,
}
