//! EmbedDB benchmarks: Flat vs HNSW insert and search performance.
//!
//! Run with: cargo bench

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use vexra_core::collection::{Collection, IndexType};
use vexra_core::config::{CollectionConfig, SearchQuery};
use vexra_core::DistanceMetric;
use vexra_index::flat::FlatIndex;
use vexra_index::hnsw::graph::HnswGraph;
use vexra_index::hnsw::HnswConfig;
use vexra_index::{SearchResult, VectorIndex};
use rand::Rng;
use std::time::Duration;

const DIM: usize = 128;

/// Generate `n` random vectors of dimension `DIM`.
fn random_vectors(n: usize) -> Vec<Vec<f32>> {
    random_vectors_dim(n, DIM)
}

/// Generate `n` random vectors of a given dimension.
fn random_vectors_dim(n: usize, dim: usize) -> Vec<Vec<f32>> {
    let mut rng = rand::thread_rng();
    (0..n).map(|_| (0..dim).map(|_| rng.gen::<f32>()).collect()).collect()
}

/// Build a random FlatIndex with `n` vectors.
fn build_flat(n: usize) -> FlatIndex {
    let mut idx = FlatIndex::new(DIM, DistanceMetric::Euclidean);
    for (i, v) in random_vectors(n).into_iter().enumerate() {
        idx.insert(i as u64, &v).unwrap();
    }
    idx
}

/// Build a random HNSW graph with `n` vectors.
fn build_hnsw(n: usize) -> HnswGraph {
    let config = HnswConfig::new(16).with_ef_construction(200);
    let mut graph = HnswGraph::new(DIM, DistanceMetric::Euclidean, config);
    for (i, v) in random_vectors(n).into_iter().enumerate() {
        graph.insert(i as u64, &v).unwrap();
    }
    graph
}

// ---------------------------------------------------------------------------
// Insert benchmarks
// ---------------------------------------------------------------------------

fn bench_insert_flat(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert_flat");
    for n in [1000, 10000, 50000] {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            b.iter(|| build_flat(black_box(n)));
        });
    }
}

fn bench_insert_hnsw(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert_hnsw");
    for n in [1000, 10000, 50000] {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            b.iter(|| build_hnsw(black_box(n)));
        });
    }
}

// ---------------------------------------------------------------------------
// Search benchmarks (100 queries, avg latency)
// ---------------------------------------------------------------------------

fn bench_search_flat(c: &mut Criterion) {
    let mut group = c.benchmark_group("search_flat");
    for n in [1000, 10000, 50000] {
        let idx = build_flat(n);
        let queries = random_vectors(100);
        group.throughput(Throughput::Elements(100));
        group.bench_with_input(BenchmarkId::from_parameter(n), &(idx, queries), |b, (idx, queries)| {
            b.iter(|| {
                for q in queries {
                    black_box(idx.search(q, 10).unwrap());
                }
            });
        });
    }
}

fn bench_search_hnsw(c: &mut Criterion) {
    let mut group = c.benchmark_group("search_hnsw");
    for n in [1000, 10000, 50000] {
        let graph = build_hnsw(n);
        let queries = random_vectors(100);
        group.throughput(Throughput::Elements(100));
        group.bench_with_input(BenchmarkId::from_parameter(n), &(graph, queries), |b, (graph, queries)| {
            b.iter(|| {
                for q in queries {
                    black_box(graph.search(q, 10).unwrap());
                }
            });
        });
    }
}

// ---------------------------------------------------------------------------
// Recall benchmark
// ---------------------------------------------------------------------------

fn bench_recall_hnsw(c: &mut Criterion) {
    let mut group = c.benchmark_group("recall_hnsw");
    let n = 10000;
    let k = 10;

    // Build both indexes
    let flat = build_flat(n);
    let hnsw = build_hnsw(n);
    let queries = random_vectors(100);

    // Compute ground truth with Flat (exact)
    let ground_truth: Vec<Vec<SearchResult>> = queries
        .iter()
        .map(|q| flat.search(q, k).unwrap())
        .collect();

    group.bench_function("recall@10_vs_flat", |b| {
        b.iter(|| {
            let mut total_hits = 0usize;
            let mut total = 0usize;
            for (q, gt) in queries.iter().zip(ground_truth.iter()) {
                let approx = hnsw.search(q, k).unwrap();
                // Count intersection
                let gt_ids: std::collections::HashSet<u64> = gt.iter().map(|r| r.id).collect();
                total_hits += approx.iter().filter(|r| gt_ids.contains(&r.id)).count();
                total += k;
            }
            let recall = total_hits as f64 / total as f64;
            black_box(recall);
        });
    });
}

// ---------------------------------------------------------------------------
// Collection persistence benchmark
// ---------------------------------------------------------------------------

fn bench_persist_collection(c: &mut Criterion) {
    let mut group = c.benchmark_group("persist_collection");
    let n = 100;

    group.bench_function("insert_100_persist", |b| {
        b.iter(|| {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("bench.vexra");
            let pc = std::sync::Arc::new(
                vexra_storage::page_cache::PageCache::open(&path, Default::default()).unwrap(),
            );
            let config = CollectionConfig::new("bench", DIM);
            let mut col = Collection::new_persistent(config, IndexType::Flat, pc).unwrap();
            let vectors = random_vectors(n);
            for (i, v) in vectors.iter().enumerate() {
                col.insert(vexra_core::config::Document::with_vector(
                    format!("doc_{}", i), v.clone(),
                ))
                .unwrap();
            }
            black_box(col.vector_count());
        });
    });
}

criterion_group!(
    name = benches;
    config = Criterion::default()
        .sample_size(30)
        .measurement_time(Duration::from_secs(10))
        .warm_up_time(Duration::from_secs(2));
    targets =
        bench_insert_flat,
        bench_search_flat,
        bench_insert_hnsw,
        bench_search_hnsw,
        bench_recall_hnsw,
        bench_persist_collection,
);
criterion_main!(benches);
