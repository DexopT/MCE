# MCE Project Context & Specification

## System Overview & Philosophy
The Model Context Engine (MCE) is an intelligent, token-aware proxy layer designed to sit between AI Agents (Clients) and Tool Servers (MCP Servers).
-   **Primary Directive:** Eliminate context window bloat, reduce API latency, and allow complex data operations to run efficiently on standard, low-VRAM hardware.
-   **Method:** Instead of passively passing massive JSON or HTML payloads back to the LLM, MCE intercepts, evaluates, prunes, and semantically chunks tool responses before the agent ever sees them.

## The "Transparent Proxy" Compatibility Layer
-   **Illusion:** The CLI agent (e.g., Claude Code) connects to `localhost:3025` (the MCE Port) as its sole MCP Server.
-   **Reality:** MCE connects to all actual MCP servers (Database, GitHub, File System, Web Scraper) in the background.
-   **Interception:** When the agent sends a standard JSON-RPC tool call, MCE intercepts it, executes the real tool, processes the massive return data using the MCE Pipeline, and hands a perfectly minified, token-optimized response back to the agent via standard JSON-RPC.

## Core Engine Components

### A. The Lazy Registrar (Dynamic Tool Management)
-   **Problem:** Current protocols dump all tool schemas into the system prompt at initialization (up-front token tax).
-   **Solution:**
    -   **Domain Grouping:** Tools categorized by domain (e.g., @database, @filesystem, @browser).
    -   **Just-in-Time Schema Injection:** MCE exposes a single meta-tool `discover_capabilities(domain)`.
    -   **Execution:** When the agent needs a domain, MCE temporarily injects schemas into the context window, removing them once the task is complete.

### B. The Token Economist (Budget & Cost Guardrails)
-   **Pre-Flight Estimation:** Runs a fast tokenizer (like `tiktoken`) on raw tool output.
-   **Strict Thresholds:**
    -   **Safe Limit:** If output < 1,000 tokens (example), pass directly.
    -   **Triggering the Squeeze:** If output exceeds limit, halt direct transfer and route to Squeeze Engine.

### C. The Squeeze Engine (The 3-Layer Filter)
Designed to run locally on standard hardware.
1.  **Layer 1: Deterministic Pruner (0ms Latency)**
    -   Uses standard logic (AST parsing, jq-style JSON manipulation) to strip guaranteed waste.
    -   Converts raw HTML to Markdown, removes base64 image strings, drops null values, truncates massive arrays.
    -   Appends metadata flag: `[MCE Notice: 4,000 identical rows truncated]`.
2.  **Layer 2: Semantic Router (CPU-Friendly RAG)**
    -   Chunks data in memory if pruned payload is still too large.
    -   Uses micro-embedding model (e.g., `all-MiniLM-L6-v2`) to embed chunks.
    -   Embeds agent's original query and performs localized vector search, extracting only relevant paragraphs/JSON objects.
3.  **Layer 3: The Synthesizer (Optional LLM Fallback)**
    -   Used only if explicitly requested.
    -   Routes chunked data to a local, small-parameter model (like `Qwen 2.5 3B`) to write a brief summary before sending to the master agent.

## Execution Flow Example
1.  **Request:** CLI Agent calls `read_entire_directory(path="./src")`.
2.  **Interception:** MCE catches JSON-RPC request.
3.  **Execution:** MCE queries local file system. Raw return = 85,000 tokens.
4.  **Economist Evaluation:** 85,000 tokens > 2,000 token budget. Route to Squeeze Engine.
5.  **Squeeze:**
    -   Layer 1 strips `.git`, `node_modules`, binaries (Size -> 30,000 tokens).
    -   Layer 2 detects task "fixing the auth bug", ranks files against "authentication logic", extracts only `auth.ts` and `middleware.ts`.
6.  **Return:** MCE returns standard JSON-RPC response with 1,200 tokens. Master agent unaware of hidden data.

## Additional Features

### The Policy Engine (Agent Firewall & Sandboxing)
-   **Action Blocking:** Inspects payloads (e.g., `bash_execution`). Detects `rm -rf`, `mkfs`, unauthorized network requests. Returns simulated error: `[MCE Blocked: Destructive command not permitted in current policy]`.
-   **Human-in-the-Loop (HitL):** Configurable pauses for high-risk commands (database DROP, git push) sending Y/N prompt to terminal.

### Semantic Caching (The "Zero-Token" Layer)
-   **Mechanism:** In-memory Semantic Cache. Hashes tool name and exact arguments before sending to real MCP server.
-   **Benefit:** If agent repeats request, MCE returns cached, pruned response instantly. Latency 0ms, zero tool computation.

### The Circuit Breaker (Infinite Loop Prevention)
-   **State Tracking:** Sliding window memory of last 5 tool calls.
-   **The Breaker:** If agent calls same tool with same failed output 3 times (e.g., `npm run build`), MCE trips breaker. Intercepts next call: `[MCE Alert: You are stuck in a loop. Previous 3 attempts failed identically. Pause execution and formulate a completely different approach, or ask the user for help.]`

### Observability TUI (Terminal User Interface)
-   **Implementation:** Lightweight TUI using Python's `Rich` or `Textual`.
-   **Display:** Real-time interception logs, total session token savings ($ saved), active cache hits vs. misses.

## MCE Codebase File Structure
**Stack:** Python (FastAPI, tiktoken, sentence-transformers, pydantic).

```plaintext
mce-core/
├── core/                           # The heart of the application
│   ├──  init .py
│   ├── proxy_server.py             # FastAPI/JSON-RPC reverse proxy
│   ├── mcp_client.py               # Internal client talking to actual tools
│   └── context_manager.py          # Tracks session token expenditure
│
├── engine/                         # The MCE Processing Pipelines
│   ├──  init .py
│   ├── lazy_registrar.py           # Dynamic injection/removal of tool schemas
│   ├── token_economist.py          # Calculates payload costs (tiktoken)
│   ├── policy_engine.py            # Rule definitions and Regex blockers
│   ├── circuit_breaker.py          # Loop detection state machine
│   └── squeeze/                    # The 3-Layer Filter System
│       ├──  init .py
│       ├── layer1_pruner.py        # Deterministic logic (jq, HTML->MD)
│       ├── layer2_semantic.py      # CPU-bound RAG (sentence-transformers)
│       └── layer3_synthesizer.py   # Optional local LLM wrapper (ollama/llama.cpp)
│
├── models/                         # Local AI model management
│   ├──  init .py
│   ├── embeddings.py               # Loads/caches lightweight embedding model
│   ├── vector_store.py             # In-memory vector store (FAISS/numpy)
│   └── semantic_cache.py           # Fast lookup for repeated requests
│
├── schemas/                        # Pydantic models
│   ├──  init .py
│   ├── json_rpc.py                 # Standard MCP JSON-RPC protocol shapes
│   └── mce_config.py               # User configuration schemas
│
├── utils/                           # Helper functions
│   ├──  init .py
│   ├── logger.py                   # Custom colorful terminal logging
│   └── chunker.py                  # Text splitting utilities
│
├── tui/
│   └── dashboard.py                # Live terminal interface
│
├── tests/                          # TDD
│   ├── test_economist.py
│   ├── test_pruner.py
│   └── test_semantic_router.py
│
├── config.yaml                      # User settings
├── main.py                         # Entry point
├── requirements.txt                # Dependencies
└── README.md                       # Documentation