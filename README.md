# Vexra

<p align="center">
  <strong>SQLite for vectors</strong> — 嵌入式向量数据库<br>
  一个二进制 · 一个文件 · 零配置 · 边缘到云端
</p>

<p align="center">
  <a href="https://github.com/Mengxun326/embeddb/actions"><img src="https://img.shields.io/github/actions/workflow/status/Mengxun326/embeddb/ci.yml?branch=master" alt="CI"></a>
  <a href="https://crates.io/crates/vexra-core"><img src="https://img.shields.io/crates/v/vexra-core?label=crates.io" alt="crates.io"></a>
  <a href="https://www.npmjs.com/package/@mengxun326/vexra"><img src="https://img.shields.io/npm/v/@mengxun326/vexra?label=npm" alt="npm"></a>
  <a href="https://pypi.org/project/vexra/"><img src="https://img.shields.io/pypi/v/vexra?label=pypi" alt="PyPI"></a>
  <img src="https://img.shields.io/badge/tests-70%20passed-brightgreen" alt="tests">
  <img src="https://img.shields.io/badge/license-MIT-blue" alt="license">
</p>

---

## 安装

```bash
# CLI
cargo install vexra-cli

# Python
pip install vexra

# JavaScript
npm install @mengxun326/vexra
```

## 5 秒上手

```bash
vexra init
vexra create-collection -n docs -d 384
vexra insert -c docs -v 0.1,0.2,...,0.384
vexra search -c docs --text "hello" -k 5
vexra serve
```

```python
import vexra
db = vexra.Database("data.vexra")
col = db.create_collection("docs", 384)
col.insert([0.1]*384, id="doc1")
col.search([0.2]*384, top_k=5)
```

```rust
use vexra_core::{Database, CollectionConfig, Document, SearchQuery};
let db = Database::open("data.vexra")?;
db.create_collection(CollectionConfig::new("docs", 384))?;
```

## 功能

| | |
|---|---|
| **嵌入式引擎** | 进程内运行，单文件存储，零配置 |
| **HNSW 索引** | 近似搜索，比暴力快 10-100× |
| **SIMD 加速** | AVX2 (x86_64) + NEON (aarch64) |
| **WAL 崩溃安全** | 预写日志 + 帧校验和 |
| **向量 & 元数据持久化** | 重启后完整恢复 |
| **元数据过滤** | `category = "tech" AND score > 5.0` |
| **文本向量化** | SimpleEmbedder (哈希 n-gram) + ONNX 接口 |
| **BM25 混合搜索** | Tantivy 稀疏检索 + RRF 融合 |
| **CLI 工具** | 8 个命令 (init, create, insert, search, info, stats, delete, serve) |
| **HTTP API + Dashboard** | REST API + 内嵌 Web 管理面板 |
| **多语言 SDK** | Rust · Python (PyO3) · JavaScript (napi-rs) |
| **C FFI** | C ABI, 支持 Go, Java, Zig 等 |

## 架构

```
vexra/
├── crates/
│   ├── vexra-core/        公共 API
│   ├── vexra-storage/     页格式, WAL, 页缓存
│   ├── vexra-index/       HNSW, Flat, SIMD
│   ├── vexra-metadata/    JSON 元数据, 过滤
│   ├── vexra-query/       BM25, RRF 融合
│   ├── vexra-embedding/   文本向量化
│   ├── vexra-ffi/         C ABI
│   ├── vexra-cli/         CLI
│   └── vexra-server/      HTTP + Dashboard
├── sdk/
│   ├── python/            pip install vexra
│   └── javascript/        npm install @mengxun326/vexra
└── .github/               CI/CD
```

## 社区

- [Issues](https://github.com/Mengxun326/embeddb/issues)
- [Discussions](https://github.com/Mengxun326/embeddb/discussions)
- [CHANGELOG](CHANGELOG.md) · [CONTRIBUTING](CONTRIBUTING.md) · [SECURITY](SECURITY.md)

## 许可证

MIT
