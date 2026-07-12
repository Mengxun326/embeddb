# Contributing to EmbedDB

Thanks for considering contributing to EmbedDB! This document outlines the process and guidelines.

## Code of Conduct

Be respectful, constructive, and inclusive. Harassment of any kind will not be tolerated.

## Getting Started

### Prerequisites

- Rust 1.80+ (install via [rustup.rs](https://rustup.rs))
- Git

### Setup

```bash
git clone https://github.com/Mengxun326/Vexra.git
cd Vexra
cargo build
cargo test
```

### Project Structure

Vexra is a Rust workspace with 9 crates. See [README.md](README.md#project-structure) for details.

## Development Workflow

1. **Find an issue** — Check [good first issue](https://github.com/Mengxun326/Vexra/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) or open a new one
2. **Fork & branch** — `git checkout -b feature/your-feature`
3. **Code & test** — Write code + tests. Run `cargo test` and `cargo clippy` before committing
4. **Commit** — Use descriptive commit messages
5. **Push & PR** — Open a Pull Request against `master`

### Code Style

- Run `cargo fmt` before committing
- Run `cargo clippy -- -D warnings` and fix all warnings
- Write doc comments (`///`) for all public APIs
- Add unit tests for new functionality

### Commit Messages

```
<area>: <short description>

<longer explanation if needed>
```

Examples:
- `storage: fix page header CRC calculation on big-endian`
- `index: add HNSW graph builder with parallel construction`
- `cli: support JSON output format in search command`

## Testing

```bash
# Run all tests
cargo test

# Run tests for a specific crate
cargo test -p vexra-storage

# Run with output
cargo test -- --nocapture

# Run benchmarks (Phase 1+)
cargo bench
```

## What to Work On

### Phase 0 → Phase 1 (current focus)

- HNSW approximate nearest neighbor index
- SIMD-accelerated distance kernels (AVX2, NEON)
- Persistent collection catalog
- WAL crash recovery tests
- Performance benchmarks (ANN-Benchmarks datasets)

### Language Bindings

- Python SDK (PyO3 / maturin)
- JavaScript/TypeScript SDK (napi-rs)
- Go SDK (CGO)
- Java/Kotlin SDK (JNI)

### Documentation

- API documentation examples
- Tutorial: "Building a semantic search app with EmbedDB"
- Architecture deep-dive: "How EmbedDB's storage engine works"
- Comparison guide vs Chroma, LanceDB, Qdrant

### Tooling

- GitHub Actions CI improvements
- Cross-compilation support
- Docker image for easy testing
- Benchmark dashboard

## Questions?

Open a [GitHub Discussion](https://github.com/Mengxun326/Vexra/discussions) or ask in an issue. We're happy to help!
