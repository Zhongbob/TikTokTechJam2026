"""Locate a model checkpoint across the different install layouts.

``SCRIPT_DIR.parent / "weights"`` only resolves for an *editable* checkout
(``.../<pkg>/src/<pkg>/detector.py`` next to ``.../<pkg>/src/weights/``). A plain
``pip install`` drops the package under ``site-packages/<pkg>/`` with no sibling
``weights/`` — which is the "tries to resolve from the python folder" bug.

`locate_checkpoint` searches a sensible set of places and can be pointed
anywhere with an env var or an explicit path.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence


def candidate_weight_dirs(script_dir: str | os.PathLike[str], *, env_var: str | None = None) -> list[Path]:
    """Directories to look in, most-specific first."""
    dirs: list[Path] = []
    seen: set[Path] = set()

    def add(path: Path | None) -> None:
        if path is None:
            return
        try:
            path = Path(path).expanduser()
        except (RuntimeError, OSError):
            return
        if path not in seen:
            seen.add(path)
            dirs.append(path)

    if env_var:
        raw = os.environ.get(env_var)
        if raw:
            env_path = Path(raw).expanduser()
            add(env_path if env_path.is_dir() else env_path.parent)

    script_dir = Path(script_dir).resolve()
    add(script_dir / "weights")           # weights bundled inside the package
    add(script_dir.parent / "weights")    # editable  <pkg>/src/weights  layout

    # walk up looking for a monorepo checkout (…/packages/… with a root pyproject)
    for ancestor in list(script_dir.parents)[:8]:
        add(ancestor / "weights")
        if (ancestor / "packages").is_dir() and (ancestor / "pyproject.toml").is_file():
            break

    cwd = Path.cwd()
    add(cwd / "weights")
    add(cwd)
    add(Path("/content/weights"))
    add(Path("/content"))
    return dirs


def locate_checkpoint(
    patterns: str | Sequence[str],
    *,
    script_dir: str | os.PathLike[str],
    env_var: str | None = None,
    allow_dir: bool = False,
) -> Path | None:
    """First file (or dir, if ``allow_dir``) matching any of ``patterns`` across
    `candidate_weight_dirs`. If ``$env_var`` points straight at an existing file
    (or dir), that wins. ``None`` if nothing matched."""
    pats = [patterns] if isinstance(patterns, str) else list(patterns)
    if env_var:
        raw = os.environ.get(env_var)
        if raw:
            env_path = Path(raw).expanduser()
            if env_path.is_file() or (allow_dir and env_path.is_dir()):
                return env_path
    for directory in candidate_weight_dirs(script_dir, env_var=env_var):
        if not directory.is_dir():
            continue
        for pattern in pats:
            for hit in sorted(directory.glob(pattern)):
                if hit.is_file() or (allow_dir and hit.is_dir()):
                    return hit
    return None
