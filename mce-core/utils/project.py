"""
MCE — Project Utilities
Project fingerprinting, session ID generation, and storage path management.
"""

from __future__ import annotations

import hashlib
import os
import uuid
from pathlib import Path


def get_project_id(project_path: str | Path | None = None) -> str:
    """
    Generate a deterministic project fingerprint from the directory path.

    Uses SHA-256 of the absolute, normalized path. This ensures the same
    project always maps to the same storage directory.
    """
    if project_path is None:
        project_path = Path.cwd()
    else:
        project_path = Path(project_path).resolve()

    raw = str(project_path).lower().replace("\\", "/")
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def get_project_label(project_path: str | Path | None = None) -> str:
    """Human-readable label from the project directory name."""
    if project_path is None:
        project_path = Path.cwd()
    return Path(project_path).resolve().name


def get_storage_root(storage_path: str = "~/.mce/projects") -> Path:
    """Resolve the storage root path (expands ~ and env vars)."""
    return Path(os.path.expanduser(os.path.expandvars(storage_path)))


def get_project_storage(
    project_id: str, storage_path: str = "~/.mce/projects"
) -> Path:
    """Return the storage directory for a specific project."""
    return get_storage_root(storage_path) / project_id


def get_session_id() -> str:
    """Generate a unique session ID (UUID4)."""
    return uuid.uuid4().hex[:16]


def ensure_storage_dirs(
    project_id: str,
    session_id: str,
    storage_path: str = "~/.mce/projects",
) -> dict[str, Path]:
    """
    Create the full storage directory structure for a project + session.

    Returns a dict of key paths:
        - project_dir: ~/.mce/projects/{project_id}/
        - session_dir: ~/.mce/projects/{project_id}/sessions/{session_id}/
        - checkpoints_dir: .../{session_id}/checkpoints/
        - memory_db: .../{project_id}/memory.db
    """
    root = get_storage_root(storage_path)
    project_dir = root / project_id
    session_dir = project_dir / "sessions" / session_id
    checkpoints_dir = session_dir / "checkpoints"

    # Create directories
    project_dir.mkdir(parents=True, exist_ok=True)
    session_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    return {
        "project_dir": project_dir,
        "session_dir": session_dir,
        "checkpoints_dir": checkpoints_dir,
        "memory_db": project_dir / "memory.db",
        "cost_db": root / "cost_ledger.db",
    }
