//! EmbedDB Python SDK — native PyO3 bindings.
//!
//! Provides safe, idiomatic Python access to the EmbedDB embedded vector database.

use embeddb_core::collection::IndexType;
use embeddb_core::config::{CollectionConfig as CoreCollectionConfig, Document, SearchQuery};
use embeddb_core::db::Database;
use embeddb_core::DistanceMetric;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

// ---------------------------------------------------------------------------
// Python-facing classes
// ---------------------------------------------------------------------------

#[pyclass]
struct EmbedDB {
    db: Option<Database>,
    path: String,
}

#[pymethods]
impl EmbedDB {
    #[new]
    fn new(path: &str) -> PyResult<Self> {
        let db = Database::open(path).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { db: Some(db), path: path.to_string() })
    }

    fn create_collection(&self, name: &str, dimension: usize, distance: Option<&str>) -> PyResult<PyCollection> {
        let db = self.db.as_ref().ok_or_else(|| PyRuntimeError::new_err("Database closed"))?;
        let metric = match distance.unwrap_or("cosine") {
            "euclidean" => DistanceMetric::Euclidean,
            "dot" => DistanceMetric::DotProduct,
            _ => DistanceMetric::Cosine,
        };
        let config = CoreCollectionConfig::new(name, dimension).with_distance(metric);
        db.create_collection(config).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(PyCollection { db_path: self.path.clone(), name: name.to_string(), dimension, distance: metric })
    }

    fn get_collection(&self, name: &str) -> PyResult<PyCollection> {
        let db = self.db.as_ref().ok_or_else(|| PyRuntimeError::new_err("Database closed"))?;
        let col = db.get_collection(name).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let col = col.read();
        Ok(PyCollection {
            db_path: self.path.clone(),
            name: col.name().to_string(),
            dimension: col.dimension(),
            distance: col.distance_metric(),
        })
    }

    fn list_collections(&self) -> PyResult<Vec<String>> {
        let db = self.db.as_ref().ok_or_else(|| PyRuntimeError::new_err("Database closed"))?;
        Ok(db.list_collections())
    }

    fn close(&mut self) {
        if let Some(db) = self.db.take() {
            let _ = db.close();
        }
    }

    fn __enter__(slf: Py<Self>) -> Py<Self> { slf }
    fn __exit__(&mut self, _: PyObject, _: PyObject, _: PyObject) { self.close(); }
}

impl Drop for EmbedDB {
    fn drop(&mut self) { self.close(); }
}

// ---------------------------------------------------------------------------

#[pyclass]
#[derive(Clone)]
struct PyCollection {
    db_path: String,
    name: String,
    dimension: usize,
    distance: DistanceMetric,
}

#[pymethods]
impl PyCollection {
    #[getter]
    fn name(&self) -> &str { &self.name }

    #[getter]
    fn dimension(&self) -> usize { self.dimension }

    fn insert(&self, doc: &Bound<'_, PyDict>) -> PyResult<String> {
        let db = Database::open(&self.db_path).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let vector: Vec<f32> = doc.get_item("vector")
            .ok_or_else(|| PyRuntimeError::new_err("Missing 'vector' field"))?
            .ok_or_else(|| PyRuntimeError::new_err("'vector' is None"))?
            .extract()?;

        let id: Option<String> = doc.get_item("id").ok().flatten().map(|v| v.extract()).transpose()?;
        let metadata: Option<serde_json::Value> = doc.get_item("metadata").ok().flatten().map(|v| {
            let s: String = v.extract().map_err(|_| PyRuntimeError::new_err("metadata must be JSON string or dict"))?;
            serde_json::from_str(&s).unwrap_or(serde_json::Value::String(s))
        });

        let core_doc = Document {
            id: id.or_else(|| Some(format!("doc_{}", NEXT_ID.fetch_add(1, Ordering::Relaxed)))),
            vector: Some(vector),
            metadata,
            text: None,
        };

        embeddb_core::insert(&db, &self.name, core_doc)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    fn search(&self, vector: Vec<f32>, top_k: Option<usize>, filter: Option<&str>) -> PyResult<Vec<PyDict>> {
        let db = Database::open(&self.db_path).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let mut query = SearchQuery::with_vector(vector, top_k.unwrap_or(10));
        if let Some(f) = filter { query = query.with_filter(f); }

        let hits = embeddb_core::search(&db, &self.name, query)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        Python::with_gil(|py| {
            let results: Vec<PyDict> = hits.iter().map(|h| {
                let d = PyDict::new_bound(py);
                d.set_item("id", &h.id).unwrap();
                d.set_item("score", h.score).unwrap();
                if let Some(ref m) = h.metadata {
                    d.set_item("metadata", serde_json::to_string(m).unwrap_or_default()).unwrap();
                }
                d
            }).collect();
            Ok(results)
        })
    }

    fn delete(&self, id: &str) -> PyResult<()> {
        let db = Database::open(&self.db_path).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let col = db.get_collection(&self.name).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        col.write().delete(id).map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    fn __len__(&self) -> PyResult<usize> {
        let db = Database::open(&self.db_path).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let col = db.get_collection(&self.name).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(col.read().vector_count())
    }

    fn __repr__(&self) -> String {
        format!("Collection(name='{}', dim={})", self.name, self.dimension)
    }
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

#[pymodule]
fn embeddb(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<EmbedDB>()?;
    m.add_class::<PyCollection>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
