"""
MCE — Proxy Server
FastAPI-based JSON-RPC reverse proxy. Orchestrates the full MCE pipeline:

  Cache Check → Forward to MCP → Economist Evaluation →
  Squeeze Engine → Policy Engine → Return Minified Response
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from core.context_manager import ContextManager
from core.mcp_client import MCPClient
from engine.circuit_breaker import CircuitBreaker
from engine.lazy_registrar import LazyRegistrar
from engine.policy_engine import PolicyEngine, PolicyDecision
from engine.squeeze.layer1_pruner import Layer1Pruner
from engine.squeeze.layer2_semantic import Layer2SemanticRouter
from engine.squeeze.layer3_synthesizer import Layer3Synthesizer
from engine.token_economist import TokenEconomist, Action
from models.semantic_cache import SemanticCache
from schemas.json_rpc import JsonRpcRequest, JsonRpcResponse, JsonRpcError
from schemas.mce_config import MCEConfig
from utils.logger import get_logger, log_token_savings

_log = get_logger("Proxy")


# ──────────────────────────────────────────────
# Proxy Server
# ──────────────────────────────────────────────

class ProxyServer:
    """
    The MCE reverse proxy.

    Exposes a single POST endpoint that accepts JSON-RPC requests,
    runs them through the full MCE pipeline, and returns minified
    responses to the AI agent.
    """

    def __init__(self, config: MCEConfig):
        self.config = config

        # ── Components ────────────────────────
        self.mcp_client = MCPClient(config.upstream_servers)
        self.economist = TokenEconomist(config.token_limits)
        self.policy = PolicyEngine(config.policy)
        self.breaker = CircuitBreaker(config.circuit_breaker)
        self.cache = SemanticCache(
            max_entries=config.cache.max_entries,
            ttl_seconds=config.cache.ttl_seconds,
        )
        self.registrar = LazyRegistrar()
        self.context = ContextManager()

        # Squeeze layers
        self.pruner = Layer1Pruner()
        self.semantic_router = (
            Layer2SemanticRouter(model_name=config.embeddings.model_name)
            if config.squeeze.layer2_semantic
            else None
        )
        self.synthesizer = (
            Layer3Synthesizer(config.synthesizer)
            if config.squeeze.layer3_synthesizer
            else None
        )

        # ── Build app with lifespan ───────────
        self.app = FastAPI(
            title="MCE — Model Context Engine",
            description="Token-aware transparent proxy for MCP servers",
            version="0.1.0",
            lifespan=self._lifespan,
        )

        # ── Routes ────────────────────────────
        self._register_routes()

    @asynccontextmanager
    async def _lifespan(self, app: FastAPI):
        """Modern FastAPI lifespan handler (replaces deprecated on_event)."""
        # Startup
        await self.mcp_client.start()

        # Auto-discover tools from upstream servers
        await self._discover_upstream_tools()

        # Warn if no upstream servers configured
        if not self.config.upstream_servers:
            _log.warning(
                "[mce.warning]No upstream MCP servers configured![/mce.warning] "
                "Add servers to config.yaml → upstream_servers"
            )

        _log.info(
            f"[mce.badge]\\[MCE][/mce.badge] "
            f"Proxy started on "
            f"[mce.info]{self.config.proxy.host}:{self.config.proxy.port}[/mce.info]"
        )

        yield

        # Shutdown
        await self.mcp_client.stop()
        _log.info("[mce.badge]\\[MCE][/mce.badge] Proxy shutting down")

    async def _discover_upstream_tools(self) -> None:
        """Auto-discover tools from upstream servers and register them."""
        for server in self.config.upstream_servers:
            try:
                schemas = await self.mcp_client.discover_tools(server.name)
                for schema in schemas:
                    self.registrar.register_tool(schema)
                    self.mcp_client.register_tool(schema.name, server.name)
                _log.info(
                    f"[mce.success]Discovered {len(schemas)} tools[/mce.success] "
                    f"from '{server.name}'"
                )
            except Exception as exc:
                _log.warning(
                    f"Failed to discover tools from '{server.name}': {exc}"
                )

    def _register_routes(self) -> None:
        """Register FastAPI routes."""

        @self.app.post("/")
        async def handle_jsonrpc(request: Request) -> Response:
            """Main JSON-RPC endpoint."""
            try:
                body = await request.json()
                rpc_request = JsonRpcRequest.model_validate(body)
            except Exception as exc:
                return JSONResponse(
                    content=JsonRpcResponse(
                        error=JsonRpcError(
                            code=-32700,
                            message=f"Parse error: {exc}",
                        )
                    ).model_dump(exclude_none=True),
                    status_code=200,
                )

            response = await self._process_request(rpc_request)
            return JSONResponse(
                content=response.model_dump(exclude_none=True),
                status_code=200,
            )

        @self.app.get("/health")
        async def health() -> dict:
            """Health check endpoint."""
            return {
                "status": "ok",
                "engine": "MCE",
                "version": "0.1.0",
                "stats": self.context.summary(),
            }

        @self.app.get("/stats")
        async def stats() -> dict:
            """Session statistics endpoint."""
            return self.context.summary()

    # ──────────────────────────────────────────
    # Main Pipeline
    # ──────────────────────────────────────────

    async def _process_request(self, request: JsonRpcRequest) -> JsonRpcResponse:
        """
        Execute the full MCE pipeline for a single JSON-RPC request.

        1. Handle meta-tools (discover_capabilities, tools/list)
        2. Check semantic cache
        3. Check policy engine (pre-execution on arguments)
        4. Forward to upstream MCP server
        5. Record in circuit breaker (with error status)
        6. Check policy engine (post-execution on response)
        7. Evaluate token budget
        8. Squeeze if over budget
        9. Cache the result
        10. Return minified response
        """
        tool_name = request.method
        arguments = request.params or {}

        _log.info(f"[mce.badge]\\[MCE][/mce.badge] ← {tool_name}")

        # 1a. Meta-tool: discover_capabilities
        if tool_name == "discover_capabilities":
            domain = arguments.get("domain", "")
            schemas = self.registrar.discover_capabilities(domain)
            return JsonRpcResponse(id=request.id, result=schemas)

        # 1b. MCP standard: tools/list
        if tool_name == "tools/list":
            tools = self._build_tools_list()
            return JsonRpcResponse(id=request.id, result={"tools": tools})

        # 2. Semantic cache check
        if self.config.cache.enabled:
            cached = self.cache.get(tool_name, arguments)
            if cached is not None:
                self.context.record_request(
                    tool_name=tool_name,
                    raw_tokens=cached.token_count,
                    squeezed_tokens=cached.token_count,
                    was_cached=True,
                )
                return JsonRpcResponse(id=request.id, result=cached.payload)

        # 3. Policy engine (pre-execution check on arguments)
        policy_result = self.policy.check(tool_name, arguments)
        if policy_result.decision == PolicyDecision.BLOCK:
            self.context.record_request(
                tool_name=tool_name,
                raw_tokens=0,
                squeezed_tokens=0,
                was_blocked=True,
            )
            return JsonRpcResponse(
                id=request.id,
                error=JsonRpcError(
                    code=-32001,
                    message=policy_result.reason,
                ),
            )

        if policy_result.decision == PolicyDecision.HITL:
            approved = await self.policy.prompt_human(tool_name, policy_result)
            if not approved:
                self.context.record_request(
                    tool_name=tool_name,
                    raw_tokens=0,
                    squeezed_tokens=0,
                    was_blocked=True,
                )
                return JsonRpcResponse(
                    id=request.id,
                    error=JsonRpcError(
                        code=-32001,
                        message=f"[MCE HitL: Command not approved] {policy_result.reason}",
                    ),
                )

        # 4. Forward to upstream MCP server
        upstream_response = await self.mcp_client.call_tool(
            tool_name, arguments, request.id
        )

        # 5. Record in circuit breaker AFTER execution, with correct error status
        is_error = upstream_response.error is not None
        breaker_state = self.breaker.record(tool_name, arguments, is_error=is_error)

        # Handle upstream errors
        if is_error:
            self.context.record_request(
                tool_name=tool_name, raw_tokens=0, squeezed_tokens=0
            )
            # If breaker tripped, return breaker alert instead of upstream error
            if breaker_state.is_tripped:
                self.context.record_request(
                    tool_name=tool_name,
                    raw_tokens=0,
                    squeezed_tokens=0,
                    breaker_tripped=True,
                )
                return JsonRpcResponse(
                    id=request.id,
                    error=JsonRpcError(
                        code=-32002,
                        message=breaker_state.alert_message,
                    ),
                )
            return upstream_response

        raw_result = upstream_response.result

        # 6. Policy engine (post-execution check on response payload)
        response_policy = self.policy.check(tool_name, raw_result)
        if response_policy.decision == PolicyDecision.BLOCK:
            self.context.record_request(
                tool_name=tool_name,
                raw_tokens=0,
                squeezed_tokens=0,
                was_blocked=True,
            )
            return JsonRpcResponse(
                id=request.id,
                error=JsonRpcError(
                    code=-32001,
                    message=response_policy.reason,
                ),
            )

        # 7. Evaluate token budget
        report = self.economist.evaluate(raw_result)
        raw_tokens = report.token_count

        # 8. Squeeze if over budget
        if report.recommended_action == Action.SQUEEZE:
            squeezed_result, notices = await self._squeeze(
                raw_result, tool_name, arguments
            )
            squeezed_tokens = self.economist.count_any(squeezed_result)
            log_token_savings(raw_tokens, squeezed_tokens)

            # Append MCE notices to the result
            if notices and isinstance(squeezed_result, str):
                squeezed_result = squeezed_result + "\n\n" + "\n".join(notices)
            elif notices and isinstance(squeezed_result, dict):
                squeezed_result["_mce_notices"] = notices
        else:
            squeezed_result = raw_result
            squeezed_tokens = raw_tokens

        # 9. Cache the processed result
        if self.config.cache.enabled:
            self.cache.put(tool_name, arguments, squeezed_result, squeezed_tokens)

        # 10. Record and return
        self.context.record_request(
            tool_name=tool_name,
            raw_tokens=raw_tokens,
            squeezed_tokens=squeezed_tokens,
        )

        return JsonRpcResponse(id=request.id, result=squeezed_result)

    # ──────────────────────────────────────────
    # Squeeze Pipeline
    # ──────────────────────────────────────────

    async def _squeeze(
        self,
        payload: Any,
        tool_name: str,
        arguments: dict,
    ) -> tuple[Any, list[str]]:
        """Run the 3-layer Squeeze Engine on a payload. Returns (result, notices)."""
        result = payload
        all_notices: list[str] = []

        # Layer 1: Deterministic Pruner
        if self.config.squeeze.layer1_pruner:
            result = self.pruner.prune(result)
            all_notices.extend(self.pruner.notices)
            _log.debug(f"L1 Pruner done — {len(self.pruner.notices)} notices")

        # Re-check tokens after pruning
        post_prune_tokens = self.economist.count_any(result)
        if post_prune_tokens <= self.config.token_limits.safe_limit:
            return result, all_notices

        # Layer 2: Semantic Router
        if self.config.squeeze.layer2_semantic and self.semantic_router is not None:
            # Use the tool name + arguments as the "agent query"
            agent_query = f"{tool_name} {json.dumps(arguments, default=str)}"
            result = self.semantic_router.route(result, agent_query)

        # Re-check tokens after semantic routing
        post_semantic_tokens = self.economist.count_any(result)
        if post_semantic_tokens <= self.config.token_limits.safe_limit:
            return result, all_notices

        # Layer 3: Synthesizer (optional)
        if self.config.squeeze.layer3_synthesizer and self.synthesizer is not None:
            agent_query = f"{tool_name} {json.dumps(arguments, default=str)}"
            result = await self.synthesizer.synthesize(
                result if isinstance(result, str) else json.dumps(result, default=str),
                agent_query,
            )

        return result, all_notices

    # ──────────────────────────────────────────
    # tools/list
    # ──────────────────────────────────────────

    def _build_tools_list(self) -> list[dict]:
        """Build the tools/list response: meta-tool + all active tools."""
        tools = [self.registrar.get_meta_tool_schema()]

        # Include all currently-active tool schemas
        for name in self.registrar.active_tool_names:
            schema = self.registrar.get_active_schema(name)
            if schema:
                tools.append({
                    "name": schema.name,
                    "description": schema.description,
                    "inputSchema": schema.input_schema,
                })

        return tools
