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
