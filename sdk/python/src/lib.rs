//! EmbedDB Python SDK — native PyO3 bindings.
//!
//! Provides safe, idiomatic Python access to the EmbedDB embedded vector database.
//! The Database is opened once and shared across all Collection handles via Arc.

use embeddb_core::collection::IndexType;
use embeddb_core::config::{CollectionConfig as CoreCollectionConfig, Document, SearchQuery};
use embeddb_core::db::Database;
use embeddb_core::DistanceMetric;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Python-facing classes
// ---------------------------------------------------------------------------

#[pyclass]
struct EmbedDB {
    db: Arc<Database>,
    path: String,
}

#[pymethods]
impl EmbedDB {
    #[new]
    fn new(path: &str) -> PyResult<Self> {
        let db = Database::open(path).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { db: Arc::new(db), path: path.to_string() })
    }

    fn create_collection(&self, name: &str, dimension: usize, distance: Option<&str>) -> PyResult<PyCollection> {
        let metric = match distance.unwrap_or("cosine") {
            "euclidean" => DistanceMetric::Euclidean,
            "dot" => DistanceMetric::DotProduct,
            _ => DistanceMetric::Cosine,
        };
        let config = CoreCollectionConfig::new(name, dimension).with_distance(metric);
        self.db.create_collection(config).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(PyCollection { db: self.db.clone(), name: name.to_string(), dimension, distance: metric })
    }

    fn get_collection(&self, name: &str) -> PyResult<PyCollection> {
        let col = self.db.get_collection(name).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let col = col.read();
        Ok(PyCollection { db: self.db.clone(), name: col.name().to_string(), dimension: col.dimension(), distance: col.distance_metric() })
    }

    fn list_collections(&self) -> PyResult<Vec<String>> {
        self.db.list_collections().map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    fn stats(&self) -> PyResult<String> {
        let stats = self.db.stats().map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        serde_json::to_string(&stats).map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    fn drop_collection(&self, name: &str) -> PyResult<()> {
        self.db.drop_collection(name).map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    fn close(&self) -> PyResult<()> {
        self.db.close().map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    fn __enter__(slf: Py<Self>) -> Py<Self> { slf }
    fn __exit__(&self, _: PyObject, _: PyObject, _: PyObject) { let _ = self.close(); }
}

// ---------------------------------------------------------------------------

#[pyclass]
#[derive(Clone)]
struct PyCollection {
    db: Arc<Database>,
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
        let vector: Vec<f32> = doc.get_item("vector")
            .ok_or_else(|| PyRuntimeError::new_err("Missing 'vector' field"))?
            .ok_or_else(|| PyRuntimeError::new_err("'vector' is None"))?
            .extract()?;
        let id: Option<String> = doc.get_item("id").ok().flatten().map(|v| v.extract()).transpose()?;
        let metadata: Option<serde_json::Value> = doc.get_item("metadata").ok().flatten().map(|v| {
            if let Ok(dict) = v.downcast::<PyDict>() {
                // Convert Python dict to JSON Value
                let json_str = dict.repr().map_err(|_| PyRuntimeError::new_err("Cannot convert dict"))?.to_string();
                serde_json::from_str(&json_str.replace('\'', "\"")).unwrap_or_default()
            } else {
                let s: String = v.extract().unwrap_or_default();
                serde_json::from_str(&s).unwrap_or(serde_json::Value::String(s))
            }
        });
        let text: Option<String> = doc.get_item("text").ok().flatten().map(|v| v.extract()).transpose()?;

        let core_doc = Document { id, vector: Some(vector), metadata, text };
        embeddb_core::insert(&self.db, &self.name, core_doc)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    fn search(&self, vector: Vec<f32>, top_k: Option<usize>, filter: Option<&str>) -> PyResult<Vec<PyDict>> {
        let mut query = SearchQuery::with_vector(vector, top_k.unwrap_or(10));
        if let Some(f) = filter { query = query.with_filter(f); }
        let hits = embeddb_core::search(&self.db, &self.name, query)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Python::with_gil(|py| {
            let results: Vec<PyDict> = hits.iter().map(|h| {
                let d = PyDict::new_bound(py);
                d.set_item("id", &h.id).unwrap();
                d.set_item("score", h.score).unwrap();
                if let Some(ref m) = h.metadata { d.set_item("metadata", serde_json::to_string(m).unwrap_or_default()).unwrap(); }
                if let Some(ref v) = h.vector {
                    let list = PyList::new_bound(py, v.iter().map(|x| *x));
                    d.set_item("vector", list).unwrap();
                }
                d
            }).collect();
            Ok(results)
        })
    }

    fn delete(&self, id: &str) -> PyResult<()> {
        let col = self.db.get_collection(&self.name).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        col.write().delete(id).map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    fn __len__(&self) -> PyResult<usize> {
        let col = self.db.get_collection(&self.name).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(col.read().vector_count())
    }

    fn __repr__(&self) -> String { format!("Collection(name='{}', dim={})", self.name, self.dimension) }
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<EmbedDB>()?;
    m.add_class::<PyCollection>()?;
    Ok(())
}
