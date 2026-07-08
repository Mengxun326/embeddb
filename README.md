# EmbedDB

<p align="center">
  <img src="https://img.shields.io/badge/version-0.1.0-blue.svg" alt="Version">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/rust-stable%201.80+-orange.svg" alt="Rust">
  <img src="https://img.shields.io/badge/tests-57%20passed-brightgreen.svg" alt="Tests">
  <img src="https://img.shields.io/badge/status-Phase%200%20MVP-yellow.svg" alt="Status">
</p>

<p align="center">
  <b>SQLite for vectors</b> — An embedded vector database.<br>
  One binary, one file, zero config. Runs anywhere from edge devices to cloud.
</p>

---

## What is EmbedDB?

EmbedDB is an **embedded vector database** — it runs inside your application process, stores everything in a single file, and requires zero configuration. Think SQLite, but purpose-built for AI workloads: semantic search, RAG (Retrieval-Augmented Generation), recommendations, and any task involving vector similarity.

### Why EmbedDB?

| | EmbedDB | Chroma | LanceDB | Qdrant |
|---|:---:|:---:|:---:|:---:|
| **Embedded (no server)** | ✅ | ✅ | ✅ | ❌ |
| **Single file** | ✅ | ❌ | ✅ | ❌ |
| **Built-in embedding** | 🔜 v0.2 | ✅ | ❌ | ❌ |
| **CLI tool** | ✅ | ❌ | ❌ | ❌ |
| **Web Dashboard** | 🔜 v0.2 | ❌ | ❌ | ✅ |
| **Multi-language SDK** | 🔜 v0.2 | Python | Python/Rust/JS | Python/Rust/Go |
| **Hybrid search** | 🔜 v0.3 | ❌ | ✅ | ✅ |
| **Crash safe (WAL)** | ✅ | ❌ | ✅ | ✅ |

✅ = supported &nbsp; 🔜 = planned &nbsp; ❌ = not available

## Features

### Phase 0 (v0.1.0)

- [x] **Embedded engine** — Single-file database, link as a Rust library or run the CLI
- [x] **Page-based storage** — SQLite-inspired 4KB page format with CRC integrity checks
- [x] **Write-Ahead Log** — Crash-safe writes with automatic recovery
- [x] **Flat (brute-force) search** — Exact nearest neighbors with Cosine, Euclidean, Dot Product
- [x] **Metadata filtering** — SQL-like expressions: `category = "tech" AND score > 5.0`
- [x] **C FFI layer** — Foundation for all language bindings (Python, JS, Go, Java)
- [x] **CLI tool** — `embeddb init | insert | search | info | stats | delete`

### Phase 1 (v0.2.0)

- [x] **HNSW index** — Approximate nearest neighbor search, 10-100x faster at scale
- [x] **SIMD acceleration** — AVX2 (x86_64) / NEON (aarch64) distance kernels
- [x] **Persistent collections** — Collections survive process restarts
- [x] **74 unit tests** — Comprehensive coverage across all crates

### Phase 2 (v0.3.0) — Current

- [x] **HTTP API Server** — Axum-based REST API with JSON endpoints
- [x] **Web Dashboard** — Built-in management UI (served from binary)
- [x] **Python SDK** — Native bindings via ctypes over C FFI
- [x] **JavaScript/TypeScript SDK** — napi-rs native module with type definitions
- [ ] **ONNX embedding engine** — Built-in text embedding (requires onnxruntime)
- [ ] **npm/pip publishing** — Publish to package registries

### Phase 3 (v1.0.0)

- [ ] **BM25 hybrid search** — Dense + sparse retrieval with RRF fusion
- [ ] **Go & Java SDKs** — Full multi-language support
- [ ] **IVF index** — For billion-scale datasets
- [ ] **Import/Export** — JSON, CSV, Parquet

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  embeddb CLI                      │
│         init / insert / search / serve            │
├─────────────────────────────────────────────────┤
│                 embeddb-core                      │
│           Database · Collection · Config          │
├──────────┬──────────┬──────────┬────────────────┤
│  storage │  index   │ metadata │    query        │
│ page fmt │  flat    │  store   │   parser        │
│   WAL    │  HNSW 🔜 │  filter  │   fusion 🔜     │
│  cache   │  SIMD 🔜 │ inverted │                 │
├──────────┴──────────┴──────────┴────────────────┤
│                  embeddb-ffi                      │
│            C ABI for multi-language               │
├─────────────────────────────────────────────────┤
│   Python · JavaScript · Go · Java (coming soon)  │
└─────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Rust 1.80+ ([rustup.rs](https://rustup.rs))

### Install from source

```bash
git clone https://github.com/Mengxun326/embeddb.git
cd embeddb
cargo build --release -p embeddb-cli
```

### Usage

```bash
# Create a new database
embeddb init

# Insert vectors (3-dimensional for this example)
embeddb insert --collection docs -v 1.0,0.0,0.0 -m '{"title":"first document"}'
embeddb insert --collection docs -v 0.0,1.0,0.0 -m '{"title":"second document"}'
embeddb insert --collection docs -v 0.0,0.0,1.0 -m '{"title":"third document"}'

# Search for similar vectors
embeddb search --collection docs -v 1.0,0.1,0.0 -k 2

# Search with metadata filter
embeddb search --collection docs -v 1.0,0.0,0.0 -k 10 -f 'title CONTAINS "first"'

# View database info
embeddb info

# Output as JSON
embeddb search --collection docs -v 1.0,0.0,0.0 -k 3 --format json
```

### Rust Library

Add to your `Cargo.toml`:

```toml
[dependencies]
embeddb-core = { git = "https://github.com/Mengxun326/embeddb.git" }
```

```rust
use embeddb_core::{Database, CollectionConfig, Document, SearchQuery};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let db = Database::open("data.embeddb")?;
    db.create_collection(CollectionConfig::new("docs", 384))?;

    let col = db.get_collection("docs")?;
    let mut col = col.write();

    // Insert
    col.insert(Document::with_vector("doc1", vec![0.1; 384]))?;
    col.insert(Document::with_vector_and_metadata(
        "doc2",
        vec![0.2; 384],
        serde_json::json!({"category": "tech"}),
    ))?;

    // Search
    let results = col.search(SearchQuery::with_vector(vec![0.15; 384], 10))?;
    for hit in results {
        println!("{}: score={:.4}", hit.id, hit.score);
    }

    Ok(())
}
```

## Project Structure

```
embeddb/
├── crates/
│   ├── embeddb-core/        # Public API (Database, Collection)
│   ├── embeddb-storage/     # Page format, WAL, page cache
│   ├── embeddb-index/       # Flat, HNSW, SIMD distance
│   ├── embeddb-metadata/    # JSON metadata, filter engine
│   ├── embeddb-query/       # Query planning, hybrid fusion (🔜)
│   ├── embeddb-embedding/   # ONNX inference engine (🔜)
│   ├── embeddb-ffi/         # C ABI for language bindings
│   ├── embeddb-cli/         # CLI binary
│   └── embeddb-server/      # Web dashboard (🔜)
├── sdk/                     # Python, JS, Go, Java (🔜)
├── dashboard/               # React SPA (🔜)
├── docs/                    # mdBook documentation (🔜)
└── tests/                   # Integration + benchmark suite
```

## Roadmap

| Version | Features | Status |
|---------|----------|--------|
| **v0.1.0** | Storage engine, Flat index, CLI, C FFI | ✅ Done |
| **v0.2.0** | HNSW, SIMD, Persistent collections | 🚧 In Progress |
| **v0.3.0** | ONNX embedding, Web Dashboard, Python/JS SDK | 📋 Planned |
| **v1.0.0** | BM25 hybrid search, Go/Java SDK, IVF index | 📋 Planned |

## Contributing

EmbedDB is in early development and welcomes contributions! Here are some ways to get involved:

- **Try it out** — Build from source, run the examples, and [open issues](https://github.com/Mengxun326/embeddb/issues) for bugs or feature requests
- **Good first issues** — Look for issues tagged [`good first issue`](https://github.com/Mengxun326/embeddb/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
- **Language bindings** — Help build Python, JavaScript, Go, or Java SDKs
- **Documentation** — Improve docs, write tutorials, create examples

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

MIT License — see [LICENSE](LICENSE) for full text.

---

<p align="center">
  <sub>Built with Rust 🦀 | Inspired by SQLite's elegant simplicity</sub>
</p>
