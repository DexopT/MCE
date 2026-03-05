"""
MCE — Squeeze Engine Layer 1: Deterministic Pruner
Zero-latency deterministic transforms to strip guaranteed waste.

Operations:
- HTML → Markdown conversion
- Base64 string removal
- Null value elimination
- Array truncation with metadata flags
- Whitespace normalization
"""

from __future__ import annotations

import re
from typing import Any

from utils.logger import get_logger

_log = get_logger("Pruner")

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────

MAX_ARRAY_LENGTH = 50  # truncate arrays longer than this
MAX_STRING_LENGTH = 5000  # truncate individual strings longer than this


# ──────────────────────────────────────────────
# Layer 1 Pruner
# ──────────────────────────────────────────────

class Layer1Pruner:
    """
    Deterministic pruner — removes guaranteed waste with zero model calls.

    Runs at ~0ms latency using pure string/JSON manipulation.
    """

    def __init__(
        self,
        max_array_length: int = MAX_ARRAY_LENGTH,
        max_string_length: int = MAX_STRING_LENGTH,
    ):
        self._max_array = max_array_length
        self._max_string = max_string_length
        self._notices: list[str] = []

    @property
    def notices(self) -> list[str]:
        """MCE notices generated during the last prune operation."""
        return list(self._notices)

    def prune(self, payload: Any) -> Any:
        """
        Apply all deterministic pruning transforms.

        Returns the pruned payload and populates .notices.
        """
        self._notices.clear()

        if isinstance(payload, str):
            payload = self._prune_string(payload)
        elif isinstance(payload, dict):
            payload = self._prune_dict(payload)
        elif isinstance(payload, list):
            payload = self._prune_list(payload)

        return payload

    # ── String transforms ─────────────────────

    def _prune_string(self, text: str) -> str:
        # HTML → Markdown
        if self._looks_like_html(text):
            text = self._html_to_markdown(text)

        # Strip base64 blobs
        text = self._strip_base64(text)

        # Normalize whitespace
        text = self._normalize_whitespace(text)

        # Truncate overly long strings
        if len(text) > self._max_string:
            truncated = len(text) - self._max_string
            text = text[: self._max_string]
            self._notices.append(
                f"[MCE Notice: String truncated, {truncated:,} characters removed]"
            )

        return text

    # ── Dict transforms ───────────────────────

    def _prune_dict(self, d: dict) -> dict:
        pruned = {}
        for key, value in d.items():
            # Skip null values
            if value is None:
                continue

            # Recursively prune nested structures
            if isinstance(value, dict):
                value = self._prune_dict(value)
                if not value:  # skip empty dicts after pruning
                    continue
            elif isinstance(value, list):
                value = self._prune_list(value)
            elif isinstance(value, str):
                value = self._prune_string(value)

            pruned[key] = value

        return pruned

    # ── List transforms ───────────────────────

    def _prune_list(self, lst: list) -> list:
        # Truncate long arrays
        if len(lst) > self._max_array:
            truncated = len(lst) - self._max_array
            lst = lst[: self._max_array]
            self._notices.append(
                f"[MCE Notice: {truncated:,} identical rows truncated]"
            )

        # Recursively prune each element
        pruned = []
        for item in lst:
            if item is None:
                continue
            if isinstance(item, dict):
                item = self._prune_dict(item)
            elif isinstance(item, list):
                item = self._prune_list(item)
            elif isinstance(item, str):
                item = self._prune_string(item)
            pruned.append(item)

        return pruned

    # ── HTML → Markdown ───────────────────────

    @staticmethod
    def _looks_like_html(text: str) -> bool:
        """Quick heuristic check for HTML content."""
        return bool(re.search(r"<\s*(html|body|div|p|h[1-6]|table|ul|ol)\b", text, re.IGNORECASE))

    @staticmethod
    def _html_to_markdown(html: str) -> str:
        """Convert HTML to clean Markdown."""
        try:
            from markdownify import markdownify as md
            result = md(html, heading_style="ATX", strip=["script", "style", "img"])
            return result.strip()
        except ImportError:
            # Fallback: strip tags with regex
            clean = re.sub(r"<[^>]+>", "", html)
            return clean.strip()

    # ── Base64 stripping ──────────────────────

    # Match data-URI base64 blobs (require prefix) OR standalone blobs (require = padding)
    _B64_PATTERN = re.compile(
        r"data:[a-zA-Z0-9/+.-]+;base64,[A-Za-z0-9+/\n]{20,}={0,2}"
        r"|(?<![A-Za-z0-9/+])[A-Za-z0-9+/]{100,}={1,2}(?![A-Za-z0-9/+=])",
        re.MULTILINE,
    )

    def _strip_base64(self, text: str) -> str:
        """Remove base64-encoded blobs from text."""
        matches = self._B64_PATTERN.findall(text)
        if matches:
            count = len(matches)
            text = self._B64_PATTERN.sub("[MCE: base64 blob removed]", text)
            self._notices.append(
                f"[MCE Notice: {count} base64 blob(s) stripped]"
            )
        return text

    # ── Whitespace normalization ──────────────

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        """Collapse excessive whitespace while preserving structure."""
        # Collapse 3+ newlines to 2
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Collapse trailing whitespace on lines
        text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)
        return text.strip()
