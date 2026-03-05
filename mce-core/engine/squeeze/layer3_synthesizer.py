"""
MCE — Squeeze Engine Layer 3: The Synthesizer (Optional LLM Fallback)
Routes chunked data to a local small-parameter model (e.g. Qwen 2.5 3B via Ollama)
to generate a brief summary before sending to the master agent.

Only used when explicitly enabled in config.
"""

from __future__ import annotations

from typing import Any, Optional

import httpx

from schemas.mce_config import SynthesizerConfig
from utils.logger import get_logger

_log = get_logger("Synthesizer")


class Layer3Synthesizer:
    """
    Optional local LLM summarizer.

    Calls the Ollama API to compress chunked data into a concise summary.
    Skipped if not configured or if the model is unavailable.
    """

    SYSTEM_PROMPT = (
        "You are a concise data summarizer for an AI coding assistant. "
        "Given raw tool output data, produce a brief, information-dense summary. "
        "Preserve all key facts, numbers, file names, and error messages. "
        "Do NOT add commentary or opinions. Be extremely terse."
    )

    def __init__(self, config: SynthesizerConfig | None = None):
        cfg = config or SynthesizerConfig()
        self._model = cfg.model
        self._ollama_url = cfg.ollama_url.rstrip("/")
        self._max_tokens = cfg.max_summary_tokens
        self._available: Optional[bool] = None

    async def synthesize(self, payload: str, agent_query: str = "") -> str:
        """
        Summarize *payload* using the local LLM.

        Falls back to returning *payload* unchanged if the model is
        unavailable or errors.
        """
        if not await self._check_available():
            _log.warning("Synthesizer skipped — Ollama not available")
            return payload

        prompt = self._build_prompt(payload, agent_query)

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    f"{self._ollama_url}/api/generate",
                    json={
                        "model": self._model,
                        "prompt": prompt,
                        "system": self.SYSTEM_PROMPT,
                        "stream": False,
                        "options": {
                            "num_predict": self._max_tokens,
                            "temperature": 0.1,
                        },
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                summary = data.get("response", "").strip()

                if summary:
                    _log.info(
                        f"[mce.success]Synthesized[/mce.success]: "
                        f"{len(payload):,} chars → {len(summary):,} chars"
                    )
                    return summary

        except Exception as exc:
            _log.warning(f"Synthesizer failed: {exc}")

        return payload

    async def _check_available(self) -> bool:
        """Ping Ollama to see if it's running."""
        if self._available is not None:
            return self._available

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self._ollama_url}/api/tags")
                self._available = resp.status_code == 200
        except Exception:
            self._available = False

        if not self._available:
            _log.debug("Ollama not reachable")
        return self._available

    @staticmethod
    def _build_prompt(payload: str, agent_query: str) -> str:
        parts = []
        if agent_query:
            parts.append(f"Agent's task: {agent_query}")
        parts.append(f"Raw tool output to summarize:\n\n{payload}")
        return "\n\n".join(parts)

    def reset_availability(self) -> None:
        """Force re-check of Ollama availability."""
        self._available = None
