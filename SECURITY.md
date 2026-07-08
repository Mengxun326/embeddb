# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in EmbedDB, please report it responsibly.

**Please do NOT open a public GitHub issue.**

Instead, send an email to the project maintainer. We will respond within 48 hours.

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 1.0.x   | ✅ Active support  |
| 0.x     | ❌ No longer supported |

## Security Model

EmbedDB is an embedded database that runs in your process:

- **Data at rest**: Database files use CRC32 integrity checks and WAL checksums
- **Crash safety**: Write-Ahead Log provides atomicity and durability
- **Concurrency**: Single-writer, multiple-reader model with snapshot isolation

### Known Limitations

- There is no built-in encryption at rest. Use filesystem-level encryption
- The page cache is in-memory; sensitive vector data may remain in RAM
- The C FFI layer accepts raw pointers; language bindings should validate inputs

## Disclosure Policy

We follow a 90-day disclosure timeline. After the fix is released, we will publish a security advisory.
