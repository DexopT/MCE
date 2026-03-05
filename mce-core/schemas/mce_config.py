"""
MCE — Configuration Schema
Loads and validates config.yaml via Pydantic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field


# ──────────────────────────────────────────────
# Sub‑models
# ──────────────────────────────────────────────

class ProxyConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 3025


class TokenLimitsConfig(BaseModel):
    safe_limit: int = 1000
    squeeze_trigger: int = 2000
    absolute_max: int = 8000


class SqueezeConfig(BaseModel):
    layer1_pruner: bool = True
    layer2_semantic: bool = True
    layer3_synthesizer: bool = False


class CacheConfig(BaseModel):
    enabled: bool = True
    max_entries: int = 512
    ttl_seconds: int = 600


class UpstreamServer(BaseModel):
    name: str
    url: str


class PolicyConfig(BaseModel):
    blocked_commands: list[str] = Field(default_factory=list)
    blocked_network: list[str] = Field(default_factory=list)
    hitl_commands: list[str] = Field(default_factory=list)


class CircuitBreakerConfig(BaseModel):
    window_size: int = 5
    failure_threshold: int = 3


class SynthesizerConfig(BaseModel):
    model: str = "qwen2.5:3b"
    ollama_url: str = "http://localhost:11434"
    max_summary_tokens: int = 300


class EmbeddingsConfig(BaseModel):
    model_name: str = "all-MiniLM-L6-v2"


class LoggingConfig(BaseModel):
    level: str = "INFO"
    show_tokens: bool = True


# ──────────────────────────────────────────────
# Meridian Intelligence Configs (v1.0)
# ──────────────────────────────────────────────

class MemVaultConfig(BaseModel):
    enabled: bool = True
    storage_path: str = "~/.mce/projects"
    extraction_mode: str = "heuristic"  # "heuristic" | "ollama" | "api"
    injection_token_budget: int = 500
    auto_update_claude_md: bool = True
    memory_types: list[str] = Field(
        default_factory=lambda: ["decisions", "dead_ends", "constraints", "preferences", "file_patterns"]
    )


class ModelCost(BaseModel):
    input: float = 3.0
    output: float = 15.0


class CostWatchConfig(BaseModel):
    enabled: bool = True
    daily_budget_usd: float = 10.00
    session_budget_usd: float = 3.00
    alert_on_parallel_agents: bool = True
    token_rate_alert_per_min: int = 1000
    model_costs: dict[str, ModelCost] = Field(default_factory=lambda: {
        "claude-opus-4": ModelCost(input=15.0, output=75.0),
        "claude-sonnet-4": ModelCost(input=3.0, output=15.0),
        "gpt-4o": ModelCost(input=5.0, output=15.0),
    })


class TimeMachineConfig(BaseModel):
    enabled: bool = True
    auto_checkpoint_interval_mins: int = 10
    checkpoint_on_file_write: bool = True
    checkpoint_on_destructive_tool: bool = True
    max_checkpoints_per_session: int = 50
    capture_file_diffs: bool = True
    branch_on_hitl_decision: bool = True


class DriftSentinelConfig(BaseModel):
    enabled: bool = True
    alert_on_constraint_violation: bool = True
    block_on_critical_violation: bool = True
    load_constraints_from_memvault: bool = True


class PermissionProfile(BaseModel):
    file_read: str = "auto"     # "auto" | "prompt" | "block"
    file_write: str = "prompt"
    shell_exec: str = "prompt"
    destructive: str = "block"


class PermissionProfilesConfig(BaseModel):
    active: str = "focused_work"
    profiles: dict[str, PermissionProfile] = Field(default_factory=lambda: {
        "exploration": PermissionProfile(
            file_read="auto", file_write="prompt", shell_exec="prompt", destructive="block"
        ),
        "focused_work": PermissionProfile(
            file_read="auto", file_write="auto", shell_exec="prompt", destructive="prompt"
        ),
        "review": PermissionProfile(
            file_read="auto", file_write="prompt", shell_exec="prompt", destructive="block"
        ),
    })


class SkillsConfig(BaseModel):
    enabled: bool = True
    path: str = ".mce/skills"
    auto_trigger: bool = True
    team_sync_url: Optional[str] = None


# ──────────────────────────────────────────────
# Root Config
# ──────────────────────────────────────────────

class MCEConfig(BaseModel):
    """Root configuration model for the entire MCE system."""
    proxy: ProxyConfig = Field(default_factory=ProxyConfig)
    token_limits: TokenLimitsConfig = Field(default_factory=TokenLimitsConfig)
    squeeze: SqueezeConfig = Field(default_factory=SqueezeConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    upstream_servers: list[UpstreamServer] = Field(default_factory=list)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    circuit_breaker: CircuitBreakerConfig = Field(default_factory=CircuitBreakerConfig)
    synthesizer: SynthesizerConfig = Field(default_factory=SynthesizerConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    # Meridian Intelligence Layer (v1.0)
    memvault: MemVaultConfig = Field(default_factory=MemVaultConfig)
    cost_watch: CostWatchConfig = Field(default_factory=CostWatchConfig)
    time_machine: TimeMachineConfig = Field(default_factory=TimeMachineConfig)
    drift_sentinel: DriftSentinelConfig = Field(default_factory=DriftSentinelConfig)
    permission_profiles: PermissionProfilesConfig = Field(default_factory=PermissionProfilesConfig)
    skills: SkillsConfig = Field(default_factory=SkillsConfig)

    @classmethod
    def from_yaml(cls, path: Optional[str | Path] = None) -> "MCEConfig":
        """Load configuration from a YAML file. Falls back to defaults."""
        if path is None:
            path = Path(__file__).resolve().parent.parent / "config.yaml"
        path = Path(path)

        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
            return cls.model_validate(raw)
        return cls()
