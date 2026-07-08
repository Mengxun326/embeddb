# EmbedDB

<p align="center">
  <strong>「向量的 SQLite」</strong> — 嵌入式向量数据库。<br>
  一个二进制、一个文件、零配置。从边缘设备到云端，随处运行。
</p>

<p align="center">
  <img src="https://img.shields.io/github/v/tag/Mengxun326/embeddb?label=版本&color=blue" alt="版本">
  <img src="https://img.shields.io/badge/许可证-MIT-green" alt="许可证">
  <img src="https://img.shields.io/badge/Rust-1.80+-orange" alt="Rust">
  <img src="https://img.shields.io/badge/测试-70%20通过-brightgreen" alt="测试">
  <img src="https://img.shields.io/github/actions/workflow/status/Mengxun326/embeddb/ci.yml?branch=master" alt="CI">
</p>

---

## 这是什么？

EmbedDB 作为一个**嵌入式数据库**，直接运行在你的应用程序进程内部，所有数据存储在一个**单独的文件**中，**无需任何配置**。不需要服务器、不需要 YAML 配置文件、不需要 Docker 容器。它是 SQLite 在向量检索领域的对应物 —— 专为 AI 工作负载打造：语义搜索、RAG（检索增强生成）、推荐系统，以及任何涉及向量相似度的任务。

```bash
# 启动 Dashboard
embeddb serve

# 插入向量（384维）
embeddb insert -c docs -v 0.1,0.2,...,0.384 -m '{"title":"入门指南"}'

# 用自然语言搜索
embeddb search -c docs --text "如何开始" -k 5
```

## 快速开始

### 安装

```bash
git clone https://github.com/Mengxun326/embeddb.git
cd embeddb
cargo build --release -p embeddb-cli
./target/release/embeddb --help
```

### 命令行

```bash
embeddb init                                    # 创建新数据库
embeddb create-collection -n docs -d 3          # 创建3维集合
embeddb insert -c docs -v 1.0,0.0,0.0          # 插入向量
embeddb insert -c docs -v 0.0,1.0,0.0          # 再插一条
embeddb search -c docs -v 1.0,0.1,0.3 -k 2     # 搜索最近邻
embeddb serve                                    # 启动 Web 管理面板
```

### Rust 库

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
    col.insert({"vector": [0.1] * 384, "metadata": {"title": "你好"}})
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

## 功能特性

| 功能 | 状态 | 说明 |
|---------|:------:|-------------|
| **嵌入式引擎** | ✅ | 进程内运行、单文件、零配置 |
| **HNSW 索引** | ✅ | 近似搜索、大规模下比暴力搜索快 10-100 倍 |
| **SIMD 加速** | ✅ | AVX2 (x86_64) + NEON (aarch64) 向量化距离计算 |
| **崩溃安全** | ✅ | WAL 预写日志 + 帧校验和 |
| **集合持久化** | ✅ | 向量和元数据在进程重启后不丢失 |
| **元数据过滤** | ✅ | SQL 风格过滤：`category = "tech" AND score > 5.0` |
| **命令行工具** | ✅ | 8 个命令：init, create-collection, insert, search, info, stats, delete, serve |
| **HTTP API + Dashboard** | ✅ | Axum REST API + 内嵌 Web 管理界面 |
| **Python SDK** | ✅ | PyO3 原生绑定、`pip install` 通过 maturin |
| **JavaScript SDK** | ✅ | napi-rs 原生模块、完整的 TypeScript 类型 |
| **文本向量化** | ✅ | SimpleEmbedder（哈希 n-gram）+ ONNX 接口预留 |
| **BM25 + 混合搜索** | ✅ | Tantivy 稀疏检索 + RRF 融合排序 |
| **C FFI** | ✅ | C ABI 接口，支持 Go、Java、Zig 等语言 |
| **基准测试** | ✅ | Criterion 套件：Flat vs HNSW 插入/搜索/召回率 |
| **CI/CD** | ✅ | GitHub Actions：三大平台测试、检查、发布 |

## 竞品对比

| | EmbedDB | Chroma | LanceDB | Qdrant | Milvus |
|---|:---:|:---:|:---:|:---:|:---:|
| **无服务器** | ✅ | ✅ | ✅ | ❌ | ❌ |
| **单文件** | ✅ | ❌ | ✅ | ❌ | ❌ |
| **内置向量化** | ✅ | ✅ | ❌ | ❌ | ❌ |
| **命令行工具** | ✅ | ❌ | ❌ | ✅ | ✅ |
| **Web 管理面板** | ✅ | ❌ | ❌ | ✅ | ✅ |
| **SIMD** | ✅ | ❌ | ✅ | ✅ | ✅ |
| **WAL 崩溃安全** | ✅ | ❌ | ✅ | ✅ | ✅ |
| **多语言 SDK** | 🐍⬡🟨 | 🐍 | 🐍⬡🟨 | 🐍⬡🟨 | 🐍⬡🟨 |
| **混合搜索** | ✅ | ❌ | ✅ | ✅ | ✅ |

## 架构

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
│ 页式存储  │  HNSW    │  元数据   │  查询解析             │
│ WAL      │  Flat    │  过滤器   │  BM25 (Tantivy)       │
│ 页缓存    │  SIMD    │  倒排索引  │  RRF 融合           │
├──────────┴──────────┴──────────┴─────────────────────┤
│                embeddb-embedding                      │
│         SimpleEmbedder · ONNX 接口预留                │
├──────────────────────────────────────────────────────┤
│                  embeddb-ffi                          │
│              C ABI → Python / JS / Go / Java          │
├──────────────────────────────────────────────────────┤
│      Python (PyO3) · JavaScript (napi-rs)             │
│      Go (CGO) · Java (JNI) [即将推出]                  │
└──────────────────────────────────────────────────────┘
```

## 性能

在 Intel i7-13700H、AVX2 开启、128 维向量的环境下测试。完整基准：`cargo bench -p embeddb-core`。

| 操作 | 1K 向量 | 10K 向量 | 50K 向量 |
|-----------|-----------|------------|------------|
| **Flat 插入** | ~0.3ms/条 | ~0.3ms/条 | ~0.3ms/条 |
| **HNSW 插入** | ~1.2ms/条 | ~1.5ms/条 | ~1.8ms/条 |
| **Flat 搜索 (P50)** | ~0.05ms | ~0.4ms | ~2.0ms |
| **HNSW 搜索 (P50)** | ~0.02ms | ~0.04ms | ~0.08ms |
| **HNSW 召回率@10** | 99.8% | 99.2% | 98.5% |

> 在 5 万向量规模下，HNSW 比暴力搜索快 25 倍，同时保持 >98% 的召回率。

## 项目结构

```
embeddb/
├── crates/
│   ├── embeddb-core/        数据库、集合公共 API
│   ├── embeddb-storage/     页格式、WAL、页缓存
│   ├── embeddb-index/       HNSW、Flat、SIMD 距离计算
│   ├── embeddb-metadata/    JSON 元数据、过滤器引擎
│   ├── embeddb-query/       BM25、RRF 融合
│   ├── embeddb-embedding/   文本向量化引擎
│   ├── embeddb-ffi/         C ABI 多语言绑定基础
│   ├── embeddb-cli/         命令行工具
│   └── embeddb-server/       HTTP API + Dashboard
├── sdk/
│   ├── python/              PyO3 原生 Python 绑定
│   └── javascript/          napi-rs 原生 Node.js 模块
├── benches/                 Criterion 性能基准
├── .github/                 Issue/PR 模板、CI/CD
└── docs/                    文档
```

## 社区

- **Issues**: [github.com/Mengxun326/embeddb/issues](https://github.com/Mengxun326/embeddb/issues)
- **Discussions**: [github.com/Mengxun326/embeddb/discussions](https://github.com/Mengxun326/embeddb/discussions)
- **贡献指南**: [CONTRIBUTING.md](CONTRIBUTING.md)
- **变更日志**: [CHANGELOG.md](CHANGELOG.md)
- **安全政策**: [SECURITY.md](SECURITY.md)

## 许可证

MIT © EmbedDB 贡献者。

---

<p align="center">
  <sub>用 Rust 🦀 构建 · 受 SQLite 优雅简约启发 · AI 原生设计</sub>
</p>
