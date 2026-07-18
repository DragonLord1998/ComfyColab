from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


class RuntimeStateError(ValueError):
    """Raised when persisted runtime state violates the versioned contract."""


def normalize_runtime_state(
    payload: Mapping[str, Any],
    *,
    session: str | None = None,
    gpu: str | None = None,
) -> dict[str, Any]:
    state = dict(payload)
    schema = state.get("schema", 1)
    if type(schema) is not int or schema != 1:
        raise RuntimeStateError(f"unsupported runtime state schema: {schema!r}")
    state["schema"] = 1
    state.setdefault("status", "ready" if state.get("comfyUrl") else "unknown")
    core = state.setdefault("core", {})
    packs = state.setdefault("packs", {})
    if not isinstance(core, dict):
        raise RuntimeStateError("runtime state 'core' must be an object")
    if not isinstance(packs, dict):
        raise RuntimeStateError("runtime state 'packs' must be an object")
    if session is not None:
        state["session"] = session
    if gpu is not None:
        state["gpu"] = gpu
    return state


def verify_lock_digest(payload: Mapping[str, Any], expected: str) -> None:
    actual = payload.get("lockSha256")
    if actual != expected:
        raise RuntimeStateError(
            f"runtime lock digest mismatch: expected {expected}, received {actual!r}"
        )


def write_runtime_state(path: Path, payload: Mapping[str, Any]) -> None:
    normalized = normalize_runtime_state(payload)
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(normalized, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_runtime_state(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.expanduser().read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    try:
        return normalize_runtime_state(payload)
    except RuntimeStateError:
        return None
