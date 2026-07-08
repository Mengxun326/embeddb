# EmbedDB

<p align="center">
  <strong>SQLite for vectors</strong> — an embedded vector database.<br>
  One binary, one file, zero config. Edge to cloud.
</p>

<p align="center">
  <img src="https://img.shields.io/github/v/tag/Mengxun326/embeddb?label=version&color=blue" alt="Version">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
  <img src="https://img.shields.io/badge/rust-stable%201.80+-orange" alt="Rust">
  <img src="https://img.shields.io/badge/tests-70%20passed-brightgreen" alt="Tests">
  <img src="https://img.shields.io/github/actions/workflow/status/Mengxun326/embeddb/ci.yml?branch=master" alt="CI">
</p>

---

## What is EmbedDB?

EmbedDB runs **inside your process**, stores everything in a **single file**, and requires **zero configuration**. No servers, no YAML files, no Docker containers. It's the database equivalent of SQLite — purpose-built for AI workloads: semantic search, RAG, recommendations, and vector similarity.

```bash
# In one terminal
embeddb serve

# In another
embeddb insert -c docs -v 0.1,0.2,...,0.384 -m '{"title":"Getting Started"}'
embeddb search -c docs --text "how to begin" -k 5
```

## Quick Start

### Install

```bash
git clone https://github.com/Mengxun326/embeddb.git
cd embeddb
cargo build --release -p embeddb-cli
./target/release/embeddb --help
```

### Use

```bash
embeddb init                                    # Create a new database
embeddb create-collection -n docs -d 3          # Create a 3-dim collection
embeddb insert -c docs -v 1.0,0.0,0.0          # Insert a vector
embeddb insert -c docs -v 0.0,1.0,0.0          # Insert another
embeddb search -c docs -v 1.0,0.1,0.3 -k 2     # Search nearest neighbors
embeddb serve                                    # Launch web dashboard
```

### Library (Rust)

```toml
[dependencies]
embeddb-core = { git = "https://github.com/Mengxun326/embeddb" }
```

```rust
use embeddb_core::{Database, CollectionConfig, Document, SearchQuery};

let db = Database::open("data.embeddb")?;
db.create_collection(CollectionConfig::new("docs", 384))?;

let col = db.get_collection("docs")?;
col.write().insert(Document::with_vector("doc1", vec![0.1; 384]))?;

let results = col.read().search(
    SearchQuery::with_vector(vec![0.2; 384], 10)
)?;
```

### Python

```bash
cd sdk/python && pip install maturin && maturin develop
```

```python
import embeddb

with embeddb.Database("data.embeddb") as db:
    col = db.create_collection("docs", 384)
    col.insert({"vector": [0.1] * 384, "metadata": {"title": "Hello"}})
    results = col.search(vector=[0.2] * 384, top_k=5)
    for hit in results:
        print(f"{hit['id']}: {hit['score']:.4f}")
```

### JavaScript / TypeScript

```typescript
const { Database } = require('embeddb');
const db = new Database('data.embeddb');
db.createCollection('docs', 384, 'cosine');
db.insert('docs', 'doc1', new Float32Array(384));
const results = db.search('docs', new Float32Array(384), 10);
db.close();
```

## Features

| Feature | Status | Description |
|---------|:------:|-------------|
| **Embedded engine** | ✅ | In-process, single-file, zero-config |
| **HNSW index** | ✅ | Approximate search, 10-100× faster at scale |
| **SIMD acceleration** | ✅ | AVX2 (x86_64) + NEON (aarch64) kernels |
| **Crash safety** | ✅ | Write-Ahead Log with frame checksums |
| **Collection persistence** | ✅ | Vectors + metadata survive restarts |
| **Metadata filtering** | ✅ | SQL-like: `category = "tech" AND score > 5.0` |
| **CLI tool** | ✅ | 8 commands: init, create, insert, search, info, stats, delete, serve |
| **HTTP API + Dashboard** | ✅ | Axum REST API + built-in management UI |
| **Python SDK** | ✅ | PyO3 native bindings, `pip install` via maturin |
| **JavaScript SDK** | ✅ | napi-rs native module, full TypeScript types |
| **Text embedding** | ✅ | SimpleEmbedder (hash n-grams) + ONNX interface |
| **BM25 + hybrid search** | ✅ | Tantivy sparse retrieval + RRF fusion |
| **C FFI** | ✅ | C ABI for Go, Java, Zig, etc. |
| **Benchmarks** | ✅ | Criterion suite: Flat vs HNSW insert/search/recall |
| **CI/CD** | ✅ | GitHub Actions: test, lint, release on 3 platforms |

## Why EmbedDB?

| | EmbedDB | Chroma | LanceDB | Qdrant | Milvus |
|---|:---:|:---:|:---:|:---:|:---:|
| **Serverless** | ✅ | ✅ | ✅ | ❌ | ❌ |
| **Single file** | ✅ | ❌ | ✅ | ❌ | ❌ |
| **Built-in embedding** | ✅ | ✅ | ❌ | ❌ | ❌ |
| **CLI** | ✅ | ❌ | ❌ | ✅ | ✅ |
| **Web Dashboard** | ✅ | ❌ | ❌ | ✅ | ✅ |
| **SIMD** | ✅ | ❌ | ✅ | ✅ | ✅ |
| **WAL crash safety** | ✅ | ❌ | ✅ | ✅ | ✅ |
| **Multi-language SDK** | 🐍⬡🟨 | 🐍 | 🐍⬡🟨 | 🐍⬡🟨 | 🐍⬡🟨 |
| **Hybrid search** | ✅ | ❌ | ✅ | ✅ | ✅ |

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                   embeddb CLI                         │
│        init · create-collection · insert · search     │
│               info · stats · delete · serve            │
├──────────────────────────────────────────────────────┤
│                  embeddb-core                         │
│             Database · Collection · Config             │
├──────────┬──────────┬──────────┬─────────────────────┤
│ storage  │  index   │ metadata │  query               │
│ page fmt │  HNSW    │  store   │  parser              │
│ WAL      │  Flat    │  filter  │  BM25 (Tantivy)       │
│ cache    │  SIMD    │ inverted │  RRF fusion          │
├──────────┴──────────┴──────────┴─────────────────────┤
│                embeddb-embedding                      │
│         SimpleEmbedder · ONNX interface               │
├──────────────────────────────────────────────────────┤
│                  embeddb-ffi                          │
│              C ABI for Python / JS / Go / Java        │
├──────────────────────────────────────────────────────┤
│      Python (PyO3) · JavaScript (napi-rs)             │
│      Go (CGO) · Java (JNI) [coming]                  │
└──────────────────────────────────────────────────────┘
```

## Performance

Benchmarks on Intel i7-13700H, AVX2 enabled, 128-dim vectors. Full suite: `cargo bench -p embeddb-core`.

| Operation | 1K vectors | 10K vectors | 50K vectors |
|-----------|-----------|------------|------------|
| **Flat insert** | ~0.3ms/vec | ~0.3ms/vec | ~0.3ms/vec |
| **HNSW insert** | ~1.2ms/vec | ~1.5ms/vec | ~1.8ms/vec |
| **Flat search (P50)** | ~0.05ms | ~0.4ms | ~2.0ms |
| **HNSW search (P50)** | ~0.02ms | ~0.04ms | ~0.08ms |
| **HNSW recall@10** | 99.8% | 99.2% | 98.5% |

> HNSW is 25× faster than brute-force at 50K vectors while maintaining >98% recall.

## Project Structure

```
embeddb/
├── crates/
│   ├── embeddb-core/        Database, Collection public API
│   ├── embeddb-storage/     Page format, WAL, page cache
│   ├── embeddb-index/       HNSW, Flat, SIMD distance
│   ├── embeddb-metadata/    JSON metadata, filter engine
│   ├── embeddb-query/       BM25, RRF fusion
│   ├── embeddb-embedding/   Text embedding engine
│   ├── embeddb-ffi/         C ABI for language bindings
│   ├── embeddb-cli/         CLI binary
│   └── embeddb-server/      HTTP API + Dashboard
├── sdk/
│   ├── python/              PyO3 native bindings
│   └── javascript/          napi-rs native module
├── benches/                 Criterion benchmarks
├── .github/                 Issue/PR templates, CI/CD
└── docs/                    Documentation
```

## Community

- **Issues**: [github.com/Mengxun326/embeddb/issues](https://github.com/Mengxun326/embeddb/issues)
- **Discussions**: [github.com/Mengxun326/embeddb/discussions](https://github.com/Mengxun326/embeddb/discussions)
- **Contributing**: [CONTRIBUTING.md](CONTRIBUTING.md)
- **Changelog**: [CHANGELOG.md](CHANGELOG.md)
- **Security**: [SECURITY.md](SECURITY.md)

## License

MIT © EmbedDB Contributors.

---

<p align="center">
  <sub>Built with Rust 🦀 · Inspired by SQLite's elegant simplicity · AI-native by design</sub>
</p>
