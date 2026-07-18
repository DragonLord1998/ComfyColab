from __future__ import annotations

import base64
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse


_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ConfigError(ValueError):
    """Raised when a versioned bootstrap configuration is invalid."""


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _validate_https_repository(value: str) -> None:
    parsed = urlparse(value)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ConfigError("core_repository must be an absolute HTTPS URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ConfigError("core_repository must not contain credentials, query, or fragment")


def _validate_relative_entrypoint(value: str) -> None:
    path = PurePosixPath(value)
    if path.is_absolute() or not value or any(part in {"", ".", ".."} for part in path.parts):
        raise ConfigError("stage1_entrypoint must be a safe relative POSIX path")


@dataclass(frozen=True)
class CoreStage0ConfigV1:
    core_repository: str
    core_commit: str
    stage1_entrypoint: str
    stage1_sha256: str
    lock_b64: str
    lock_sha256: str
    port: int = 8188
    refresh: bool = False
    colab_proxy: bool = False
    accepted_licenses: tuple[str, ...] = ()
    schema: int = 1

    @classmethod
    def create(
        cls,
        *,
        core_repository: str,
        core_commit: str,
        stage1_entrypoint: str,
        stage1_sha256: str,
        lock_bytes: bytes,
        port: int = 8188,
        refresh: bool = False,
        colab_proxy: bool = False,
        accepted_licenses: Sequence[str] = (),
    ) -> "CoreStage0ConfigV1":
        return cls(
            core_repository=core_repository,
            core_commit=core_commit,
            stage1_entrypoint=stage1_entrypoint,
            stage1_sha256=stage1_sha256,
            lock_b64=base64.b64encode(lock_bytes).decode("ascii"),
            lock_sha256=sha256_bytes(lock_bytes),
            port=port,
            refresh=refresh,
            colab_proxy=colab_proxy,
            accepted_licenses=tuple(sorted(set(accepted_licenses))),
        ).validated()

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CoreStage0ConfigV1":
        expected = {
            "schema",
            "core_repository",
            "core_commit",
            "stage1_entrypoint",
            "stage1_sha256",
            "lock_b64",
            "lock_sha256",
            "port",
            "refresh",
            "colab_proxy",
            "accepted_licenses",
        }
        unknown = set(payload) - expected
        missing = expected - set(payload)
        if missing:
            raise ConfigError(f"stage-0 config is missing fields: {', '.join(sorted(missing))}")
        if unknown:
            raise ConfigError(f"stage-0 config has unknown fields: {', '.join(sorted(unknown))}")
        try:
            config = cls(
                schema=payload["schema"],
                core_repository=payload["core_repository"],
                core_commit=payload["core_commit"],
                stage1_entrypoint=payload["stage1_entrypoint"],
                stage1_sha256=payload["stage1_sha256"],
                lock_b64=payload["lock_b64"],
                lock_sha256=payload["lock_sha256"],
                port=payload["port"],
                refresh=payload["refresh"],
                colab_proxy=payload["colab_proxy"],
                accepted_licenses=tuple(payload["accepted_licenses"]),
            )
        except TypeError as error:
            raise ConfigError(f"invalid stage-0 config: {error}") from error
        return config.validated()

    def lock_bytes(self) -> bytes:
        try:
            return base64.b64decode(self.lock_b64, validate=True)
        except (ValueError, TypeError) as error:
            raise ConfigError("lock_b64 is not valid base64") from error

    def validated(self) -> "CoreStage0ConfigV1":
        if type(self.schema) is not int or self.schema != 1:
            raise ConfigError("unsupported CoreStage0Config schema")
        if not isinstance(self.core_repository, str):
            raise ConfigError("core_repository must be a string")
        _validate_https_repository(self.core_repository)
        if not isinstance(self.core_commit, str) or not _COMMIT_RE.fullmatch(
            self.core_commit
        ):
            raise ConfigError("core_commit must be a lowercase 40-character Git commit")
        if not isinstance(self.stage1_entrypoint, str):
            raise ConfigError("stage1_entrypoint must be a string")
        _validate_relative_entrypoint(self.stage1_entrypoint)
        if not isinstance(self.stage1_sha256, str) or not _SHA256_RE.fullmatch(
            self.stage1_sha256
        ):
            raise ConfigError("stage1_sha256 must be a lowercase SHA-256 digest")
        if not isinstance(self.lock_sha256, str) or not _SHA256_RE.fullmatch(
            self.lock_sha256
        ):
            raise ConfigError("lock_sha256 must be a lowercase SHA-256 digest")
        lock_bytes = self.lock_bytes()
        if sha256_bytes(lock_bytes) != self.lock_sha256:
            raise ConfigError("embedded lock digest does not match lock_sha256")
        try:
            lock = json.loads(lock_bytes)
        except json.JSONDecodeError as error:
            raise ConfigError("embedded lock is not valid JSON") from error
        if not isinstance(lock, dict) or lock.get("schema") != 1:
            raise ConfigError("embedded lock must be a ComfyColabLockV1 object")
        if type(self.port) is not int or not 1 <= self.port <= 65535:
            raise ConfigError("port must be an integer between 1 and 65535")
        if type(self.refresh) is not bool or type(self.colab_proxy) is not bool:
            raise ConfigError("refresh and colab_proxy must be booleans")
        if (
            not isinstance(self.accepted_licenses, tuple)
            or len(self.accepted_licenses) != len(set(self.accepted_licenses))
            or tuple(sorted(self.accepted_licenses)) != self.accepted_licenses
            or any(
                not isinstance(item, str)
                or not re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,127}", item)
                for item in self.accepted_licenses
            )
        ):
            raise ConfigError(
                "accepted_licenses must be a sorted unique tuple of license-gate IDs"
            )
        return self

    def to_dict(self) -> dict[str, Any]:
        self.validated()
        return {
            "schema": self.schema,
            "core_repository": self.core_repository,
            "core_commit": self.core_commit,
            "stage1_entrypoint": self.stage1_entrypoint,
            "stage1_sha256": self.stage1_sha256,
            "lock_b64": self.lock_b64,
            "lock_sha256": self.lock_sha256,
            "port": self.port,
            "refresh": self.refresh,
            "colab_proxy": self.colab_proxy,
            "accepted_licenses": list(self.accepted_licenses),
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())
