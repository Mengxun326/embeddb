<p align="center">
  <img src="assets/vexra-logo.svg" width="132" alt="Vexra logo">
</p>

<h1 align="center">Vexra</h1>

<p align="center">
  <strong>SQLite for vectors.</strong><br>
  An embedded, single-file vector database for local AI applications, RAG prototypes, and edge services.
</p>

<p align="center">
  <a href="README.md">简体中文</a>
  ·
  <a href="README.en.md"><strong>English</strong></a>
</p>

<p align="center">
  <a href="https://github.com/Mengxun326/Vexra/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/Mengxun326/Vexra/ci.yml?branch=master&label=CI" alt="CI"></a>
  <a href="https://pypi.org/project/vexra/"><img src="https://img.shields.io/pypi/v/vexra?label=PyPI" alt="PyPI"></a>
  <a href="https://github.com/Mengxun326/Vexra/blob/master/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="License"></a>
  <img src="https://img.shields.io/badge/rust-2021-f74c00" alt="Rust 2021">
  <img src="https://img.shields.io/badge/status-release%20candidate-15b8a6" alt="Release candidate">
</p>

Vexra is a Rust-native embedded vector database. It runs inside your application process, stores vectors and metadata in a local database file, and exposes the same engine through a Rust API, CLI, HTTP server, Python bindings, JavaScript bindings, and a C FFI layer.

Use Vexra when you want vector search without operating a separate database service: local-first AI tools, desktop apps, test fixtures, small RAG systems, on-device semantic search, or edge deployments where one binary and one data file are easier to ship.

## Highlights

| Capability | What it gives you |
| --- | --- |
| Embedded deployment | Link the Rust crate, use the CLI, or start the built-in HTTP server. No external daemon is required. |
| Single-file persistence | Collections, vectors, metadata, and catalog state live in a Vexra database file. |
| WAL recovery | Write-ahead logging and checkpointing protect writes across process restarts. |
| Vector indexes | Exact Flat search for small collections and HNSW approximate nearest neighbor search for scale. |
| Distance metrics | Cosine, Euclidean, and dot product scoring. |
| Metadata filters | Store JSON metadata and filter searches with expressions such as `kind = "note" AND score > 0.8`. |
| Hybrid search modules | Workspace crates include simple embeddings, BM25 sparse retrieval, and reciprocal-rank fusion building blocks. |
| SIMD kernels | Distance kernels include scalar fallback plus AVX2 and NEON paths where available. |
| Multi-language surface | Rust, CLI, HTTP, Python, JavaScript, and C ABI entry points share the same core engine. |
| Dashboard | `vexra serve` starts a local management UI and REST API. |

## Quick Start

Install the CLI from a checkout:

```bash
git clone https://github.com/Mengxun326/Vexra.git
cd Vexra
cargo install --path crates/vexra-cli
```

Create a database, add a collection, insert a vector, and search it:

```bash
vexra --path data.vexra init
vexra --path data.vexra create-collection --name docs --dim 4 --distance cosine --index hnsw

vexra --path data.vexra insert \
  --collection docs \
  --id doc-1 \
  --vector 0.12,0.24,0.36,0.48 \
  --meta '{"kind":"note","title":"hello vectors"}'

vexra --path data.vexra search \
  --collection docs \
  --vector 0.10,0.20,0.30,0.40 \
  --top-k 5 \
  --format json
```

Start the local API and dashboard:

```bash
vexra --path data.vexra serve --host 127.0.0.1 --port 9020
```

Then open `http://127.0.0.1:9020`.

## Rust API

```rust
use serde_json::json;
use vexra_core::{
    CollectionConfig, Database, DistanceMetric, Document, SearchQuery,
};

fn main() -> vexra_core::Result<()> {
    let db = Database::open("data.vexra")?;

    if !db.collection_exists("docs") {
        let mut config = CollectionConfig::new("docs", 4)
            .with_distance(DistanceMetric::Cosine)
            .with_description("Example documents");
        config.index_type = "hnsw".to_string();
        db.create_collection(config)?;
    }

    vexra_core::insert(
        &db,
        "docs",
        Document::with_vector_and_metadata(
            "doc-1",
            vec![0.12, 0.24, 0.36, 0.48],
            json!({"kind": "note", "title": "hello vectors"}),
        ),
    )?;

    let hits = vexra_core::search(
        &db,
        "docs",
        SearchQuery::with_vector(vec![0.10, 0.20, 0.30, 0.40], 5)
            .with_filter(r#"kind = "note""#),
    )?;

    for hit in hits {
        println!("{} {}", hit.id, hit.score);
    }

    db.close()?;
    Ok(())
}
```

## Python API

The Python package is published as `vexra`.

```bash
pip install vexra
```

```python
import vexra

db = vexra.Database("data.vexra")
col = db.create_collection("docs", 4, "cosine")

col.insert([0.12, 0.24, 0.36, 0.48], id="doc-1")
results = col.search([0.10, 0.20, 0.30, 0.40], top_k=5)

for hit in results:
    print(hit["id"], hit["score"])

db.close()
```

## HTTP API

Run the server:

```bash
vexra --path data.vexra serve --port 9020
```

Create a collection:

```bash
curl -X POST http://127.0.0.1:9020/api/collections \
  -H "content-type: application/json" \
  -d '{"name":"docs","dimension":4,"distance":"cosine"}'
```

Insert a document:

```bash
curl -X POST http://127.0.0.1:9020/api/collections/docs/documents \
  -H "content-type: application/json" \
  -d '{"id":"doc-1","vector":[0.12,0.24,0.36,0.48],"metadata":{"kind":"note"}}'
```

Search:

```bash
curl -X POST http://127.0.0.1:9020/api/collections/docs/search \
  -H "content-type: application/json" \
  -d '{"vector":[0.10,0.20,0.30,0.40],"top_k":5,"filter":"kind = \"note\""}'
```

## Architecture

```text
Applications
  |-- Rust API
  |-- CLI
  |-- HTTP + Dashboard
  |-- Python SDK
  |-- JavaScript SDK
  `-- C FFI
        |
        v
vexra-core
  |-- collection catalog
  |-- document API
  |-- metadata filtering
  `-- search orchestration
        |
        +-- vexra-storage   pages, database header, WAL, cache
        +-- vexra-index     Flat, HNSW, distance metrics, SIMD kernels
        +-- vexra-metadata  JSON metadata store and filter parser
        +-- vexra-query     BM25 and rank fusion primitives
        `-- vexra-embedding simple local text embedding utilities
```

## Workspace Layout

| Path | Purpose |
| --- | --- |
| `crates/vexra-core` | Public Rust API, database handle, collections, document operations. |
| `crates/vexra-storage` | Page-based storage engine, database format, page cache, WAL. |
| `crates/vexra-index` | Flat and HNSW vector indexes plus distance kernels. |
| `crates/vexra-metadata` | JSON metadata storage and filter expressions. |
| `crates/vexra-query` | BM25 sparse retrieval and fusion utilities. |
| `crates/vexra-embedding` | Lightweight local embedding helpers. |
| `crates/vexra-cli` | `vexra` command-line interface. |
| `crates/vexra-server` | Axum REST API and embedded dashboard. |
| `crates/vexra-ffi` | C ABI for integrations with other runtimes. |
| `sdk/python` | PyO3-based Python package. |
| `sdk/javascript` | napi-rs based Node.js package. |

## Status

Vexra is in an early release-candidate phase. The core storage, collection, vector search, CLI, Python bindings, and HTTP API are usable, but APIs may still change before a stable 1.x line. For production workloads, benchmark with your own data, keep backups, and pin exact versions.

Near-term focus:

- Align all package metadata, docs, and SDK examples around the Vexra name.
- Expand query examples for metadata filters, HNSW tuning, and hybrid retrieval.
- Improve storage compaction and long-running write workloads.
- Publish deeper benchmarks for Flat vs HNSW search behavior.

## Brand Asset

The Vexra mark lives at [`assets/vexra-logo.svg`](assets/vexra-logo.svg). It combines a database cylinder, vector graph nodes, and a V-shaped search path so the icon still reads clearly at small sizes.

## Contributing

Contributions are welcome. Please read [`CONTRIBUTING.md`](CONTRIBUTING.md) before opening a pull request, and use GitHub Issues for bug reports, feature requests, and design discussions.

## License

Vexra is released under the [MIT License](LICENSE).
