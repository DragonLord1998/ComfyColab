"""Canonical JSON helpers used for immutable pack locks."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json_bytes(value: Any) -> bytes:
    """Return deterministic UTF-8 JSON bytes without volatile formatting."""

    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    """Return the SHA-256 of :func:`canonical_json_bytes` for *value*."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()
