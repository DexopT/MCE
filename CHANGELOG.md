# Changelog

All notable changes to MCE will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.1] - 2026-03-05

### Added

- **Proxy Server** — FastAPI-based JSON-RPC reverse proxy with `lifespan` lifecycle management.
- **MCP Client** — Async HTTP client using standard MCP protocol (`tools/list`, `tools/call`).
- **Token Economist** — Pre-flight token estimation using tiktoken with three-tier budget guardrails (`safe_limit`, `squeeze_trigger`, `absolute_max`).
- **Squeeze Engine** — 3-layer compression pipeline:
  - **Layer 1 (Pruner)** — HTML→Markdown, base64 stripping, null removal, array truncation, whitespace normalization.
  - **Layer 2 (Semantic Router)** — CPU-friendly RAG via sentence-transformers and in-memory numpy vector store.
  - **Layer 3 (Synthesizer)** — Optional Ollama-powered LLM summarization with graceful fallback.
- **Semantic Cache** — LRU cache with TTL for zero-token repeated request handling.
- **Policy Engine** — Regex-based command scanner with blocked patterns and Human-in-the-Loop (HitL) terminal prompts.
- **Circuit Breaker** — Sliding-window loop detector to prevent infinite tool-call cycles.
- **Lazy Registrar** — Just-in-Time tool schema injection via `discover_capabilities` meta-tool.
- **Context Manager** — Per-session token expenditure tracking and operational statistics.
- **TUI Dashboard** — Rich Live terminal dashboard for real-time observability (`--dashboard` flag).
- **Auto-Discovery** — Automatic upstream tool enumeration on startup via `tools/list`.
- **Concurrency Safety** — `asyncio.Lock` on shared state (cache, context manager, circuit breaker).
- **CLI Options** — `--dashboard`/`--tui` and `--config` command-line flags.
- **Test Suite** — 30 unit tests covering token economist, pruner, vector store, and chunker.

[0.0.1]: https://github.com/DexopT/MCE/releases/tag/v0.0.1
