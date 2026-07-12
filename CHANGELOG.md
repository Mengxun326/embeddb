# Changelog

All notable changes to Vexra will be documented in this file.

## [1.4.0] — 2026-07-10

### Added
- Multi-page data overflow: auto-allocate new pages when data/metadata pages are full
- 8 end-to-end integration tests (CRUD lifecycle, persistence, HNSW recall, bulk insert)
- GitHub Pages benchmark report with automated CI deployment
- Docker multi-arch image published to ghcr.io

### Fixed
- WAL checksums now verified correctly on recovery (salts stored in WAL header)
- delete() tombstones correctly written to overflow pages (prevents doc resurrection)
- insert() ordering: persist to disk before updating in-memory index
- Metadata page overflow support (was limited to single ~3KB page)
- Various branding references (EmbedDB → Vexra) in docs

### Changed
- Benchmarks: persist_collection now uses 128-dim vectors with page overflow
- CI: release workflow simplified, PyPI uses Python 3.12, npm/pypi use --skip-existing

## [1.0.0-rc1] — 2026-07-09

### Added
- **Vector & metadata persistence**: data survives process restarts via page-based storage
- **HNSW approximate nearest neighbor index**: 10-100x faster search at scale
- **SIMD-accelerated distance kernels**: AVX2 (x86_64) and NEON (aarch64) support
- **HTTP API server**: Axum-based REST API with 9 endpoints
- **Web Dashboard**: built-in management UI with overview, collections, and search playground
- **Python SDK**: native PyO3 bindings with `pip install` support via maturin
- **JavaScript SDK**: napi-rs native Node.js module with full TypeScript definitions
- **BM25 sparse retrieval**: Tantivy-backed full-text search index
- **RRF & weighted fusion**: hybrid search combining dense vectors and sparse text
- **SimpleEmbedder**: zero-dependency hash-based text embedding engine
- **WAL auto-checkpoint**: periodic checkpointing triggered after write operations
- **Collection catalog persistence**: collection definitions survive process restarts
- **C FFI layer**: C-compatible ABI for Python/Go/Java bindings
- **CLI tool**: 8 commands (init, create-collection, insert, search, info, stats, delete, serve)
- **Benchmark suite**: criterion benchmarks for Flat vs HNSW insert/search/recall
- **CI/CD**: GitHub Actions for test, lint, and release workflows

### Changed
- Python SDK: migrated from ctypes to PyO3 for native performance
- Collection: uses `IndexBackend` enum to dispatch Flat/HNSW at runtime
- Database: `create_collection` now uses persistent storage by default

### Fixed
- HNSW search heap direction (was returning worst matches instead of best)
- `random_level` panic on `ln(0)` when random value is exactly 0.0
- `write_cell` return value silently ignored (could lose data on full page)
- TOCTOU race in `ensure_catalog_page` (added catalog mutex)
- `f32::total_cmp` used for strict total ordering in HNSW BinaryHeap
- WAL frame checksums validated during recovery
- Python SDK: `PyCollection` now holds `Arc<Database>` instead of re-opening per operation
- `delete()` writes tombstone cells for durable deletion

## [0.3.0] — 2026-07-08

### Added
- HTTP API server with REST endpoints
- Web Dashboard (self-contained HTML)
- Python SDK (ctypes wrapper)
- JavaScript SDK (napi-rs structure + TypeScript definitions)
- Collection catalog persistence

### Fixed
- 10 code review findings: HNSW heap direction, NaN handling, page layout, WAL header size

## [0.2.0] — 2026-07-08

### Added
- HNSW approximate nearest neighbor index
- SIMD distance kernels (AVX2, NEON)
- Runtime CPU feature detection with scalar fallback
- Persistent collection catalog

## [0.1.0] — 2026-07-08

### Added
- Initial release: EmbedDB Phase 0 MVP
- Page-based storage engine with WAL crash safety
- Flat (brute-force) vector search
- JSON metadata storage with SQL-like filter expressions
- C FFI layer
- CLI tool: init, insert, search, info
