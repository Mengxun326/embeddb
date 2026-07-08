//! REST API route handlers.

use crate::state::AppState;
use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::{Html, Json},
    routing::{delete, get, post},
    Router,
};
use crate::dashboard;
use embeddb_core::config::{CollectionConfig, Document, SearchQuery};
use embeddb_core::DistanceMetric;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Create the Axum router with all API routes.
pub fn create_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/", get(dashboard_index))
        .route("/api/health", get(health))
        .route("/api/collections", get(list_collections).post(create_collection))
        .route("/api/collections/:name", get(get_collection))
        .route("/api/collections/:name/search", post(search))
        .route("/api/collections/:name/documents", post(insert_document))
        .route("/api/collections/:name/documents/:id", delete(delete_document))
        .route("/api/stats", get(stats))
        .with_state(state)
}

// ---------------------------------------------------------------------------
// Response types
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct HealthResponse {
    status: String,
    version: String,
}

#[derive(Serialize)]
struct CollectionInfo {
    name: String,
    dimension: usize,
    distance: String,
    vector_count: usize,
}

#[derive(Deserialize)]
struct CreateCollectionRequest {
    name: String,
    dimension: usize,
    #[serde(default = "default_distance")]
    distance: String,
    #[serde(default)]
    description: String,
}

fn default_distance() -> String {
    "cosine".to_string()
}

#[derive(Deserialize)]
struct InsertRequest {
    #[serde(default)]
    id: Option<String>,
    vector: Vec<f32>,
    #[serde(default)]
    metadata: Option<serde_json::Value>,
    #[serde(default)]
    text: Option<String>,
}

#[derive(Deserialize)]
struct SearchRequest {
    vector: Vec<f32>,
    #[serde(default = "default_top_k")]
    top_k: usize,
    #[serde(default)]
    filter: Option<String>,
}

fn default_top_k() -> usize {
    10
}

#[derive(Serialize)]
struct ApiError {
    error: String,
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

async fn dashboard_index() -> Html<String> {
    Html(dashboard::dashboard_html())
}

async fn health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
    })
}

async fn list_collections(
    State(state): State<Arc<AppState>>,
) -> Result<Json<Vec<CollectionInfo>>, (StatusCode, Json<ApiError>)> {
    let db = state.db.read();
    let names = db.list_collections();
    let mut infos = Vec::new();
    for name in names {
        if let Ok(col) = db.get_collection(&name) {
            let col = col.read();
            infos.push(CollectionInfo {
                name: col.name().to_string(),
                dimension: col.dimension(),
                distance: format!("{:?}", col.distance_metric()),
                vector_count: col.vector_count(),
            });
        }
    }
    Ok(Json(infos))
}

async fn get_collection(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> Result<Json<CollectionInfo>, (StatusCode, Json<ApiError>)> {
    let db = state.db.read();
    let col = db.get_collection(&name).map_err(|e| {
        (
            StatusCode::NOT_FOUND,
            Json(ApiError { error: e.to_string() }),
        )
    })?;
    let col = col.read();
    Ok(Json(CollectionInfo {
        name: col.name().to_string(),
        dimension: col.dimension(),
        distance: format!("{:?}", col.distance_metric()),
        vector_count: col.vector_count(),
    }))
}

async fn create_collection(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CreateCollectionRequest>,
) -> Result<(StatusCode, Json<CollectionInfo>), (StatusCode, Json<ApiError>)> {
    let metric = match req.distance.as_str() {
        "euclidean" => DistanceMetric::Euclidean,
        "dot" | "dotproduct" => DistanceMetric::DotProduct,
        _ => DistanceMetric::Cosine,
    };

    let config = CollectionConfig::new(&req.name, req.dimension)
        .with_distance(metric)
        .with_description(req.description);

    let db = state.db.read();
    db.create_collection(config).map_err(|e| {
        (
            StatusCode::CONFLICT,
            Json(ApiError { error: e.to_string() }),
        )
    })?;
    drop(db);

    // Return the created collection info
    let db = state.db.read();
    let col = db.get_collection(&req.name).map_err(|e| {
        (
            StatusCode::NOT_FOUND,
            Json(ApiError { error: e.to_string() }),
        )
    })?;
    let col = col.read();
    Ok((StatusCode::CREATED, Json(CollectionInfo {
        name: col.name().to_string(),
        dimension: col.dimension(),
        distance: format!("{:?}", col.distance_metric()),
        vector_count: col.vector_count(),
    })))
}

async fn insert_document(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    Json(req): Json<InsertRequest>,
) -> Result<(StatusCode, Json<serde_json::Value>), (StatusCode, Json<ApiError>)> {
    let db = state.db.read();
    let doc = Document {
        id: req.id,
        vector: Some(req.vector),
        metadata: req.metadata,
        text: req.text,
    };

    let doc_id = embeddb_core::insert(&db, &name, doc).map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            Json(ApiError { error: e.to_string() }),
        )
    })?;

    let resp = serde_json::json!({"id": doc_id, "name": name});
    Ok((StatusCode::CREATED, Json(resp)))
}

async fn search(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    Json(req): Json<SearchRequest>,
) -> Result<Json<Vec<serde_json::Value>>, (StatusCode, Json<ApiError>)> {
    let db = state.db.read();
    let mut query = SearchQuery::with_vector(req.vector, req.top_k);
    if let Some(filter) = req.filter {
        query = query.with_filter(filter);
    }

    let hits = embeddb_core::search(&db, &name, query).map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            Json(ApiError { error: e.to_string() }),
        )
    })?;

    let results: Vec<serde_json::Value> = hits
        .iter()
        .map(|h| {
            serde_json::json!({
                "id": h.id,
                "score": h.score,
                "metadata": h.metadata,
                "vector": h.vector,
            })
        })
        .collect();

    Ok(Json(results))
}

async fn delete_document(
    State(state): State<Arc<AppState>>,
    Path((name, id)): Path<(String, String)>,
) -> Result<StatusCode, (StatusCode, Json<ApiError>)> {
    let db = state.db.read();
    let col = db.get_collection(&name).map_err(|e| {
        (
            StatusCode::NOT_FOUND,
            Json(ApiError { error: e.to_string() }),
        )
    })?;
    let mut col = col.write();
    col.delete(&id).map_err(|e| {
        (
            StatusCode::NOT_FOUND,
            Json(ApiError { error: e.to_string() }),
        )
    })?;

    Ok(StatusCode::NO_CONTENT)
}

async fn stats(
    State(state): State<Arc<AppState>>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ApiError>)> {
    let db = state.db.read();
    let stats = db.stats().map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiError { error: e.to_string() }),
        )
    })?;

    Ok(Json(serde_json::to_value(stats).unwrap_or_default()))
}
