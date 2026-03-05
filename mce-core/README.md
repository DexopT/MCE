# MCE — Model Context Engine

> **Token-aware transparent proxy for AI agents.**
> Eliminates context window bloat, reduces API latency, and runs efficiently on standard hardware.

## What is MCE?

MCE sits between your AI Agent (e.g., Claude Code, Cursor) and your MCP Tool Servers. It intercepts tool responses, evaluates their token cost, and applies a 3-layer **Squeeze Engine** to compress massive payloads before they ever reach your LLM's context window.

```
┌──────────┐     JSON-RPC     ┌──────────┐     JSON-RPC     ┌──────────────┐
│ AI Agent │ ───────────────→ │   MCE    │ ───────────────→ │  MCP Server  │
│          │ ←─── minified ── │  Proxy   │ ←─── raw ─────── │  (Tool)      │
└──────────┘                  └──────────┘                  └──────────────┘
                                  │
                          ┌───────┴───────┐
                          │ Squeeze Engine │
                          │  L1: Pruner   │
                          │  L2: Semantic │
                          │  L3: Synth.   │
                          └───────────────┘
```

## Quick Start

```bash
# 1. Install dependencies
cd mce-core
pip install -r requirements.txt

# 2. Edit config
#    → Set upstream MCP servers in config.yaml

# 3. Start the proxy
python main.py
```

MCE starts on `localhost:3025`. Point your AI agent's MCP config to this address.

## Architecture

| Component | File | Purpose |
|-----------|------|---------|
| **Proxy Server** | `core/proxy_server.py` | FastAPI JSON-RPC reverse proxy |
| **MCP Client** | `core/mcp_client.py` | Forwards calls to real tool servers |
| **Token Economist** | `engine/token_economist.py` | Budget guardrails (tiktoken) |
| **Policy Engine** | `engine/policy_engine.py` | Destructive command blocker |
| **Circuit Breaker** | `engine/circuit_breaker.py` | Infinite loop detector |
| **Lazy Registrar** | `engine/lazy_registrar.py` | Just-in-Time schema injection |
| **L1 Pruner** | `engine/squeeze/layer1_pruner.py` | HTML→MD, null strip, array truncation |
| **L2 Semantic** | `engine/squeeze/layer2_semantic.py` | CPU-friendly RAG filtering |
| **L3 Synthesizer** | `engine/squeeze/layer3_synthesizer.py` | Optional local LLM summary |
| **Semantic Cache** | `models/semantic_cache.py` | Zero-token repeated request cache |
| **TUI Dashboard** | `tui/dashboard.py` | Real-time observability |

## Configuration

Edit `config.yaml`:

```yaml
proxy:
  port: 3025

token_limits:
  safe_limit: 1000        # pass through if under
  squeeze_trigger: 2000   # route to squeeze if over

squeeze:
  layer1_pruner: true
  layer2_semantic: true
  layer3_synthesizer: false  # needs Ollama

upstream_servers:
  - name: "filesystem"
    url: "http://localhost:3001"
```

## Running Tests

```bash
cd mce-core
pytest tests/ -v
```

## License

MIT
