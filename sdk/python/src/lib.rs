use vexra_core::config::{CollectionConfig as CoreCollectionConfig, Document, SearchQuery};
use vexra_core::DistanceMetric;
use vexra_core::db::Database;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::Arc;

#[pyclass]
struct EmbedDB {
    db: Arc<Database>,
}

#[pymethods]
impl EmbedDB {
    #[new]
    fn new(path: &str) -> PyResult<Self> {
        let db = Database::open(path).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { db: Arc::new(db) })
    }

    #[pyo3(signature = (name, dim, distance=None))]
    fn create_collection(&self, name: &str, dim: usize, distance: Option<&str>) -> PyResult<PyCollection> {
        let metric = match distance.unwrap_or("cosine") {
            "euclidean" => DistanceMetric::Euclidean, "dot" => DistanceMetric::DotProduct, _ => DistanceMetric::Cosine,
        };
        self.db.create_collection(CoreCollectionConfig::new(name, dim).with_distance(metric))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(PyCollection { db: self.db.clone(), name: name.to_string(), dim, distance: metric })
    }

    fn get_collection(&self, name: &str) -> PyResult<PyCollection> {
        let col = self.db.get_collection(name).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let c = col.read();
        let r = PyCollection { db: self.db.clone(), name: c.name().to_string(), dim: c.dimension(), distance: c.distance_metric() };
        Ok(r)
    }

    fn list_collections(&self) -> Vec<String> {
        self.db.list_collections()
    }

    fn close(&self) -> PyResult<()> { self.db.close().map_err(|e| PyRuntimeError::new_err(e.to_string())) }
    fn __enter__(slf: Py<Self>) -> Py<Self> { slf }
    fn __exit__(&self, _exc_type: PyObject, _exc_val: PyObject, _exc_tb: PyObject) {}
}

#[pyclass]
#[derive(Clone)]
struct PyCollection {
    db: Arc<Database>,
    name: String,
    dim: usize,
    distance: DistanceMetric,
}

#[pymethods]
impl PyCollection {
    #[getter]
    fn name(&self) -> &str { &self.name }
    #[getter]
    fn dimension(&self) -> usize { self.dim }

    #[pyo3(signature = (vector, id=None))]
    fn insert(&self, vector: Vec<f32>, id: Option<String>) -> PyResult<String> {
        let doc = Document { id, vector: Some(vector), metadata: None, text: None };
        vexra_core::insert(&self.db, &self.name, doc)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    #[pyo3(signature = (vector, top_k=None))]
    fn search(&self, vector: Vec<f32>, top_k: Option<usize>) -> PyResult<Vec<PyObject>> {
        let hits = vexra_core::search(&self.db, &self.name, SearchQuery::with_vector(vector, top_k.unwrap_or(10)))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Python::with_gil(|py| {
            let results: Vec<PyObject> = hits.iter().map(|h| {
                let d = PyDict::new_bound(py);
                d.set_item("id", &h.id).unwrap();
                d.set_item("score", h.score).unwrap();
                d.into()
            }).collect();
            Ok(results)
        })
    }

    fn delete(&self, id: &str) -> PyResult<()> {
        let col = self.db.get_collection(&self.name).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let result = col.write().delete(id).map_err(|e| PyRuntimeError::new_err(e.to_string()));
        drop(col);
        result
    }

    fn __len__(&self) -> PyResult<usize> {
        let col_arc = self.db.get_collection(&self.name).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let count = { let g = col_arc.read(); g.vector_count() };
        Ok(count)
    }

    fn __repr__(&self) -> String { format!("Collection(name='{}', dim={})", self.name, self.dim) }
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<EmbedDB>()?;
    m.add_class::<PyCollection>()?;
    Ok(())
}
