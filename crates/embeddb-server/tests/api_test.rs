//! Integration tests for the HTTP API.

use axum::body::{to_bytes, Body};
use axum::http::{Request, StatusCode};
use embeddb_server::routes;
use embeddb_server::state::AppState;
use std::sync::Arc;
use tower::ServiceExt; // for `oneshot`

fn setup_app() -> (tempfile::TempDir, axum::Router) {
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("test.embeddb");
    let state = AppState::new(&db_path.display().to_string()).unwrap();
    let app = routes::create_router(Arc::new(state));
    (dir, app)
}

#[tokio::test]
async fn test_health() {
    let (_dir, app) = setup_app();
    let response = app
        .oneshot(
            Request::builder()
                .uri("/api/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["status"], "ok");
}

#[tokio::test]
async fn test_create_and_list_collections() {
    let (_dir, app) = setup_app();

    // List (empty)
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/api/collections")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    // Create
    let body = serde_json::json!({"name": "test", "dimension": 3, "distance": "cosine"});
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/api/collections")
                .method("POST")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::CREATED);
}

#[tokio::test]
async fn test_insert_and_search() {
    let (_dir, app) = setup_app();

    // Create collection
    let body = serde_json::json!({"name": "docs", "dimension": 3});
    app.clone()
        .oneshot(
            Request::builder()
                .uri("/api/collections")
                .method("POST")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    // Insert document
    let doc = serde_json::json!({"vector": [1.0, 0.0, 0.0], "id": "doc1"});
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/api/collections/docs/documents")
                .method("POST")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&doc).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::CREATED);

    // Search
    let query = serde_json::json!({"vector": [1.0, 0.1, 0.0], "top_k": 5});
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/api/collections/docs/search")
                .method("POST")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&query).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let results: Vec<serde_json::Value> = serde_json::from_slice(&body).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0]["id"], "doc1");
}

#[tokio::test]
async fn test_stats() {
    let (_dir, app) = setup_app();

    // Create a collection first
    let body = serde_json::json!({"name": "test", "dimension": 4});
    app.clone()
        .oneshot(
            Request::builder()
                .uri("/api/collections")
                .method("POST")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    let response = app
        .oneshot(
            Request::builder()
                .uri("/api/stats")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let stats: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(stats["collection_count"], 1);
}

#[tokio::test]
async fn test_404_collection() {
    let (_dir, app) = setup_app();

    let response = app
        .oneshot(
            Request::builder()
                .uri("/api/collections/nonexistent")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}
