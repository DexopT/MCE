"""
MCE — MCP Client
Internal async HTTP client that forwards JSON-RPC requests to actual
upstream MCP servers. Manages per-server connections.

Uses the standard MCP protocol:
- tools/list  → enumerate available tools
- tools/call  → invoke a tool with { name, arguments }
"""

from __future__ import annotations

import json
from typing import Any, Optional

import httpx

from schemas.json_rpc import JsonRpcRequest, JsonRpcResponse, JsonRpcError, ToolSchema
from schemas.mce_config import UpstreamServer
from utils.logger import get_logger

_log = get_logger("MCPClient")


class MCPClient:
    """
    Async client that talks to real MCP tool servers.

    For each upstream server configured in config.yaml, the client
    maintains a connection and forwards JSON-RPC requests transparently
    using the MCP `tools/call` convention.
    """

    def __init__(self, upstream_servers: list[UpstreamServer] | None = None):
        self._servers: dict[str, UpstreamServer] = {}
        self._tool_to_server: dict[str, str] = {}  # tool_name → server_name
        self._client: Optional[httpx.AsyncClient] = None

        if upstream_servers:
            for srv in upstream_servers:
                self._servers[srv.name] = srv

    async def start(self) -> None:
        """Initialize the HTTP client pool."""
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=10.0),
            limits=httpx.Limits(max_connections=50, max_keepalive_connections=10),
        )

        if not self._servers:
            _log.warning(
                "[mce.warning]No upstream MCP servers configured![/mce.warning]"
            )
        else:
            _log.info(
                f"MCP Client started — {len(self._servers)} upstream server(s) configured"
            )

    async def stop(self) -> None:
        """Gracefully close connections."""
        if self._client:
            await self._client.aclose()
            self._client = None

    def register_tool(self, tool_name: str, server_name: str) -> None:
        """Map a tool name to the upstream server that handles it."""
        self._tool_to_server[tool_name] = server_name

    async def discover_tools(self, server_name: str) -> list[ToolSchema]:
        """
        Send `tools/list` to an upstream server and return ToolSchema objects.

        This is the standard MCP way to enumerate available tools.
        """
        server = self._servers.get(server_name)
        if server is None:
            _log.warning(f"Server '{server_name}' not found for tool discovery")
            return []

        request = JsonRpcRequest(
            method="tools/list",
            params={},
            id=f"mce-discover-{server_name}",
        )

        try:
            response = await self._send(server, request)
            if response.error is not None:
                _log.warning(
                    f"tools/list failed on '{server_name}': {response.error.message}"
                )
                return []

            # Parse result → list of ToolSchema
            tools_data = response.result
            if isinstance(tools_data, dict):
                tools_data = tools_data.get("tools", [])

            schemas = []
            for tool in (tools_data or []):
                if isinstance(tool, dict):
                    schemas.append(ToolSchema(
                        name=tool.get("name", ""),
                        description=tool.get("description", ""),
                        input_schema=tool.get("inputSchema", tool.get("input_schema", {})),
                        domain=self._infer_domain(tool.get("name", ""), server_name),
                    ))
            return schemas

        except Exception as exc:
            _log.warning(f"Tool discovery failed for '{server_name}': {exc}")
            return []

    async def call_tool(
        self,
        tool_name: str,
        arguments: Optional[dict[str, Any]] = None,
        request_id: Optional[int | str] = None,
    ) -> JsonRpcResponse:
        """
        Forward a tool call to the appropriate upstream MCP server.

        Uses the MCP standard: method="tools/call", params={name, arguments}.
        Returns the raw JsonRpcResponse from the server.
        """
        server_name = self._tool_to_server.get(tool_name)
        server = self._servers.get(server_name or "")

        if server is None:
            # Try any available server
            server = self._find_server_for_tool(tool_name)

        if server is None:
            return JsonRpcResponse(
                id=request_id,
                error=JsonRpcError(
                    code=-32601,
                    message=f"No upstream server found for tool '{tool_name}'",
                ),
            )

        # MCP standard: tools/call with name + arguments in params
        request = JsonRpcRequest(
            method="tools/call",
            params={
                "name": tool_name,
                "arguments": arguments or {},
            },
            id=request_id,
        )

        try:
            resp = await self._send(server, request)
            return resp
        except Exception as exc:
            _log.error(f"Upstream call failed: {server.name}/{tool_name} — {exc}")
            return JsonRpcResponse(
                id=request_id,
                error=JsonRpcError(
                    code=-32000,
                    message=f"Upstream server error: {exc}",
                ),
            )

    async def _send(
        self, server: UpstreamServer, request: JsonRpcRequest
    ) -> JsonRpcResponse:
        """Send JSON-RPC request to upstream and parse response."""
        if self._client is None:
            await self.start()

        assert self._client is not None

        _log.debug(f"→ {server.name}: {request.method}")

        resp = await self._client.post(
            server.url,
            json=request.model_dump(exclude_none=True),
            headers={"Content-Type": "application/json"},
        )
        resp.raise_for_status()

        data = resp.json()
        return JsonRpcResponse.model_validate(data)

    def _find_server_for_tool(self, tool_name: str) -> Optional[UpstreamServer]:
        """Attempt to find any server for an unknown tool (broadcasts first available)."""
        if self._servers:
            return next(iter(self._servers.values()))
        return None

    async def health_check(self) -> dict[str, bool]:
        """Check connectivity to all upstream servers."""
        results = {}
        if self._client is None:
            await self.start()

        assert self._client is not None

        for name, server in self._servers.items():
            try:
                resp = await self._client.get(server.url, timeout=5.0)
                results[name] = resp.status_code < 500
            except Exception:
                results[name] = False

        return results

    @staticmethod
    def _infer_domain(tool_name: str, server_name: str) -> str:
        """Infer a domain group from the tool/server name."""
        # Use the server name as the default domain
        return server_name
