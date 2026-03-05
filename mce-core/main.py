"""
MCE — Application Entry Point
Loads config, initializes all components, and starts the proxy server.
Optionally launches the TUI dashboard alongside the server.
"""

from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

import uvicorn

# Ensure mce-core is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from schemas.mce_config import MCEConfig
from core.proxy_server import ProxyServer
from utils.logger import get_logger, console

_log = get_logger("Main")


# ──────────────────────────────────────────────
# Banner
# ──────────────────────────────────────────────

BANNER = """
[bold cyan]
 ███╗   ███╗ ██████╗███████╗
 ████╗ ████║██╔════╝██╔════╝
 ██╔████╔██║██║     █████╗
 ██║╚██╔╝██║██║     ██╔══╝
 ██║ ╚═╝ ██║╚██████╗███████╗
 ╚═╝     ╚═╝ ╚═════╝╚══════╝
[/bold cyan]
 [dim]Model Context Engine v0.1.0[/dim]
 [dim]Token-Aware Transparent Proxy[/dim]
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MCE — Model Context Engine")
    parser.add_argument(
        "--dashboard", "--tui",
        action="store_true",
        help="Launch the observability TUI dashboard alongside the server",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yaml (default: mce-core/config.yaml)",
    )
    return parser.parse_args()


def main() -> None:
    """Start the MCE proxy server."""
    args = parse_args()

    console.print(BANNER)

    # Load config
    config_path = Path(args.config) if args.config else Path(__file__).resolve().parent / "config.yaml"
    config = MCEConfig.from_yaml(config_path)

    _log.info(f"Config loaded from [mce.info]{config_path}[/mce.info]")
    _log.info(f"Token safe limit: [mce.token]{config.token_limits.safe_limit:,}[/mce.token]")
    _log.info(f"Squeeze trigger:  [mce.token]{config.token_limits.squeeze_trigger:,}[/mce.token]")
    _log.info(
        f"Squeeze layers:   "
        f"L1={'[mce.success]ON[/mce.success]' if config.squeeze.layer1_pruner else '[mce.error]OFF[/mce.error]'}  "
        f"L2={'[mce.success]ON[/mce.success]' if config.squeeze.layer2_semantic else '[mce.error]OFF[/mce.error]'}  "
        f"L3={'[mce.success]ON[/mce.success]' if config.squeeze.layer3_synthesizer else '[mce.error]OFF[/mce.error]'}"
    )
    _log.info(f"Upstream servers: {len(config.upstream_servers)}")

    # Build proxy
    proxy = ProxyServer(config)

    # Optionally start the TUI dashboard in a background thread
    if args.dashboard:
        _log.info("[mce.info]TUI Dashboard enabled[/mce.info] — launching in background")
        from tui.dashboard import Dashboard
        dashboard = Dashboard(proxy.context)
        tui_thread = threading.Thread(target=dashboard.start, daemon=True)
        tui_thread.start()

    # Start server
    _log.info(
        f"[mce.badge]\\[MCE][/mce.badge] Starting on "
        f"[mce.info]http://{config.proxy.host}:{config.proxy.port}[/mce.info]"
    )

    uvicorn.run(
        proxy.app,
        host=config.proxy.host,
        port=config.proxy.port,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
