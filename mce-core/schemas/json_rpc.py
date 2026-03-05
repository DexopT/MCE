"""
MCE — JSON-RPC 2.0 Protocol Schemas
Standard MCP-compatible request/response shapes.
"""

from __future__ import annotations

from typing import Any, Optional, Union

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────
# JSON-RPC 2.0 Core Models
# ──────────────────────────────────────────────

class JsonRpcRequest(BaseModel):
    """Incoming JSON-RPC 2.0 request from the AI agent."""
    jsonrpc: str = Field(default="2.0", pattern=r"^2\.0$")
    method: str = Field(..., description="Tool name / RPC method")
    params: Optional[dict[str, Any]] = Field(default=None, description="Tool arguments")
    id: Optional[Union[str, int]] = Field(default=None, description="Request identifier")


class JsonRpcError(BaseModel):
    """JSON-RPC 2.0 error object."""
    code: int = Field(..., description="Error code")
    message: str = Field(..., description="Human‑readable error string")
    data: Optional[Any] = Field(default=None, description="Additional error payload")


class JsonRpcResponse(BaseModel):
    """Outgoing JSON-RPC 2.0 response to the AI agent."""
    jsonrpc: str = Field(default="2.0", pattern=r"^2\.0$")
    result: Optional[Any] = Field(default=None, description="Successful result payload")
    error: Optional[JsonRpcError] = Field(default=None, description="Error payload")
    id: Optional[Union[str, int]] = Field(default=None, description="Must match request id")


# ──────────────────────────────────────────────
# MCP-Specific Models
# ──────────────────────────────────────────────

class ToolSchema(BaseModel):
    """Schema definition for a single MCP tool."""
    name: str
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict)
    domain: str = Field(default="general", description="Domain group (e.g. @filesystem)")


class ToolCallResult(BaseModel):
    """Internal wrapper for processed tool results."""
    tool_name: str
    raw_tokens: int = 0
    squeezed_tokens: int = 0
    was_cached: bool = False
    payload: Any = None
    mce_notices: list[str] = Field(default_factory=list)
