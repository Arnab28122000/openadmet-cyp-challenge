"""Read the repo-root `.env.local` without a dependency, and without leaking it.

Secrets live in `/Users/arnab/Documents/talindrew/.env.local`, which is
gitignored (`.gitignore` line 59, `.env*`). Nothing in this package writes a
token to a log line, an argument list or a checkpoint — `ModelImpl.secret` in
the FoldRx registry takes the same care, and for the same reason: an argument
list is visible in `ps`, and a log line outlives the run.

The VM never sees this file. A token needed on the VM is passed through the
process environment of the `colab exec` call, not uploaded.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Walk up from this file to find the repo root, so the package works from any
#: working directory.
_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = _ROOT / ".env.local"


def load(path: str | Path | None = None, override: bool = False) -> list[str]:
    """Load KEY=VALUE lines into `os.environ`. Returns the key names loaded.

    Returns names, never values, so a caller can log what it found without
    logging what it is.
    """
    target = Path(path) if path else ENV_FILE
    if not target.exists():
        return []

    loaded: list[str] = []
    for line in target.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        if override or key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded


def require(name: str) -> str:
    """Fetch a variable, loading `.env.local` first, and fail clearly if absent."""
    if name not in os.environ:
        load()
    value = os.environ.get(name, "")
    if not value:
        raise RuntimeError(
            f"{name} is not set. Add it to {ENV_FILE} (gitignored) or export it."
        )
    return value
