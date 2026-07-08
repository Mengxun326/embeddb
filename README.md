# EmbedDB

> **SQLite for vectors** — An embedded vector database. One binary, one file, zero config.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/rust-stable-brightgreen.svg)](https://www.rust-lang.org)

EmbedDB is a high-performance embedded vector database designed for local-first AI applications. Think of it as SQLite, but for vectors — no server, no configuration, just link the library or run the CLI.

## ✨ Features (Planned)

- 🚀 **Embedded by design** — Single file database, zero configuration, runs anywhere
- 🔍 **HNSW + Flat indexes** — Approximate and exact nearest neighbor search
- 🧠 **Built-in embedding** — ONNX Runtime powered, auto-embeds text locally
- 🔀 **Hybrid search** — Dense vectors + BM25 sparse retrieval + metadata filtering
- 🖥️ **Full-stack DX** — CLI for power users, Web Dashboard for visualization
- 🌍 **Multi-language** — Native bindings for Python, JavaScript, Go, Java, and Rust
- 🛡️ **Crash safe** — Write-Ahead Log with snapshot isolation

## 🚧 Development Status

**Current: Phase 0 — MVP Scaffolding**

- [x] Project skeleton
- [ ] Storage engine (page format + WAL)
- [ ] Flat (brute-force) index
- [ ] Metadata store
- [ ] Core API
- [ ] C FFI layer
- [ ] CLI tool
- [ ] Integration tests

## 🏗️ Architecture

```
embeddb/
├── crates/
│   ├── embeddb-core        # Public API, orchestration
│   ├── embeddb-storage     # Page format, WAL, page cache
│   ├── embeddb-index       # HNSW, Flat, IVF vector indexes
│   ├── embeddb-metadata    # JSON metadata + inverted index
│   ├── embeddb-query       # Query planning, hybrid search fusion
│   ├── embeddb-embedding   # ONNX embedding engine (optional)
│   ├── embeddb-ffi         # C-compatible ABI layer
│   ├── embeddb-cli         # Command-line interface
│   └── embeddb-server      # Web dashboard HTTP server
├── sdk/                    # Language bindings
├── dashboard/              # React SPA for web UI
└── docs/                   # Documentation
```

## 🔧 Quick Start (Coming Soon)

```bash
# Install EmbedDB CLI
cargo install embeddb-cli

# Create a new database
embeddb init --path mydata.embeddb

# Insert vectors
embeddb insert --collection docs --text "EmbedDB is an embedded vector database"

# Search
embeddb search --collection docs --text "vector database" --k 10

# Launch web dashboard
embeddb serve
```

## 📄 License

MIT — see [LICENSE](LICENSE) for details.
