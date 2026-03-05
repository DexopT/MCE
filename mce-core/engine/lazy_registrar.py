"""
MCE — Lazy Registrar (Dynamic Tool Schema Management)
Just-in-Time schema injection: only load tool schemas when the agent
actually needs a specific domain, then remove them when done.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from schemas.json_rpc import ToolSchema
from utils.logger import get_logger

_log = get_logger("Registrar")


# ──────────────────────────────────────────────
# Domain Catalog
# ──────────────────────────────────────────────

@dataclass
class DomainGroup:
    """A domain (e.g. @filesystem) and its associated tool schemas."""
    name: str
    tools: list[ToolSchema] = field(default_factory=list)
    is_active: bool = False


class LazyRegistrar:
    """
    Manages domain-grouped tool schemas.

    Instead of dumping all tool schemas into the system prompt upfront
    (the "token tax"), MCE exposes a single meta-tool `discover_capabilities(domain)`
    that temporarily injects schemas on demand.
    """

    def __init__(self):
        self._domains: dict[str, DomainGroup] = {}
        self._active_schemas: dict[str, ToolSchema] = {}  # tool_name → schema

    # ── Registration ──────────────────────────

    def register_tool(self, schema: ToolSchema) -> None:
        """Register a tool under its domain group."""
        domain = schema.domain
        if domain not in self._domains:
            self._domains[domain] = DomainGroup(name=domain)
        self._domains[domain].tools.append(schema)
        _log.debug(f"Registered tool '{schema.name}' in domain '@{domain}'")

    def register_tools(self, schemas: list[ToolSchema]) -> None:
        """Bulk register multiple tools."""
        for s in schemas:
            self.register_tool(s)

    # ── Discovery (the meta-tool) ─────────────

    def discover_capabilities(self, domain: str) -> list[dict[str, Any]]:
        """
        The meta-tool that agents call to discover available tools
        in a specific domain.

        Activates the domain and returns lightweight schema summaries.
        """
        group = self._domains.get(domain)
        if group is None:
            _log.warning(f"Domain '@{domain}' not found")
            return []

        group.is_active = True
        for tool in group.tools:
            self._active_schemas[tool.name] = tool

        _log.info(
            f"[mce.success]Activated[/mce.success] domain '@{domain}' "
            f"({len(group.tools)} tools)"
        )

        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            }
            for t in group.tools
        ]

    def release_domain(self, domain: str) -> None:
        """Deactivate a domain and remove its schemas from active context."""
        group = self._domains.get(domain)
        if group is None:
            return

        group.is_active = False
        for tool in group.tools:
            self._active_schemas.pop(tool.name, None)

        _log.info(f"Released domain '@{domain}'")

    # ── Lookup ────────────────────────────────

    def get_active_schema(self, tool_name: str) -> Optional[ToolSchema]:
        """Look up a currently-active tool schema by name."""
        return self._active_schemas.get(tool_name)

    def is_tool_active(self, tool_name: str) -> bool:
        """Check if a tool is currently injected into the active context."""
        return tool_name in self._active_schemas

    @property
    def active_tool_names(self) -> list[str]:
        return list(self._active_schemas.keys())

    @property
    def available_domains(self) -> list[str]:
        return list(self._domains.keys())

    @property
    def active_domains(self) -> list[str]:
        return [name for name, grp in self._domains.items() if grp.is_active]

    def get_meta_tool_schema(self) -> dict[str, Any]:
        """Return the schema for the `discover_capabilities` meta-tool itself."""
        return {
            "name": "discover_capabilities",
            "description": (
                "Discover available tool capabilities in a specific domain. "
                f"Available domains: {', '.join(self.available_domains) or 'none registered'}."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "domain": {
                        "type": "string",
                        "description": "The domain to discover (e.g. 'filesystem', 'database')",
                    }
                },
                "required": ["domain"],
            },
        }
