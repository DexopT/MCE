"""
MCE — Custom Logger
Rich-powered colorful terminal logging with [MCE] badge prefix.
"""

from __future__ import annotations

import logging
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

# ──────────────────────────────────────────────
# Theme & Console
# ──────────────────────────────────────────────

_MCE_THEME = Theme(
    {
        "mce.info": "bold cyan",
        "mce.warning": "bold yellow",
        "mce.error": "bold red",
        "mce.success": "bold green",
        "mce.debug": "dim white",
        "mce.token": "bold magenta",
        "mce.badge": "bold white on blue",
    }
)

console = Console(theme=_MCE_THEME)


# ──────────────────────────────────────────────
# Logger Factory
# ──────────────────────────────────────────────

_loggers: dict[str, logging.Logger] = {}


def get_logger(name: str = "MCE", level: Optional[str] = None) -> logging.Logger:
    """Return a Rich-powered logger with the given name."""
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(f"mce.{name}")
    logger.setLevel(getattr(logging, (level or "INFO").upper(), logging.INFO))
    logger.propagate = False

    if not logger.handlers:
        handler = RichHandler(
            console=console,
            rich_tracebacks=True,
            tracebacks_show_locals=False,
            show_time=True,
            show_path=False,
            markup=True,
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)

    _loggers[name] = logger
    return logger


# ──────────────────────────────────────────────
# Convenience helpers
# ──────────────────────────────────────────────

def log_token_savings(raw: int, squeezed: int, logger: Optional[logging.Logger] = None) -> None:
    """Pretty-print token savings with Rich markup."""
    lg = logger or get_logger()
    saved = raw - squeezed
    pct = (saved / raw * 100) if raw > 0 else 0
    lg.info(
        "[mce.badge]\\[MCE][/mce.badge] "
        f"[mce.token]{raw:,}[/mce.token] → "
        f"[mce.success]{squeezed:,}[/mce.success] tokens  "
        f"([mce.success]-{pct:.0f}%[/mce.success]  |  "
        f"[mce.token]{saved:,} saved[/mce.token])"
    )


def log_cache_hit(tool_name: str, logger: Optional[logging.Logger] = None) -> None:
    """Log a semantic cache hit."""
    lg = logger or get_logger()
    lg.info(
        "[mce.badge]\\[MCE][/mce.badge] "
        f"[mce.success]CACHE HIT[/mce.success] → {tool_name}"
    )


def log_policy_block(action: str, reason: str, logger: Optional[logging.Logger] = None) -> None:
    """Log a policy-engine block event."""
    lg = logger or get_logger()
    lg.warning(
        "[mce.badge]\\[MCE][/mce.badge] "
        f"[mce.error]BLOCKED[/mce.error] → {action} — {reason}"
    )
