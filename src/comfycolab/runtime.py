"""Authenticated stage-1 runtime for ComfyColab core and resolved packs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

from .packs.errors import PackContractError
from .packs.lock import ComfyColabLockV1


CONTENT = Path("/content")
COMFY_DIR = CONTENT / "ComfyUI"
STATE_DIR = CONTENT / ".comfycolab"
PACKS_DIR = STATE_DIR / "packs"
PACK_STATE_DIR = STATE_DIR / "pack-state"
DEPENDENCIES_DIR = STATE_DIR / "dependencies"
ENVIRONMENTS_DIR = STATE_DIR / "environments"
STATE_FILE = STATE_DIR / "runtime.json"
COMFY_LOG = STATE_DIR / "comfyui.log"
TUNNEL_LOG = STATE_DIR / "cloudflared.log"
PIP_BASELINE_FILE = STATE_DIR / "pip-baseline.json"
READY_PREFIX = "COMFYCOLAB_READY="
DEFAULT_COLAB_CORS_ORIGIN = "https://colab.research.google.com"
HUGGINGFACE_HUB_REQUIREMENT = "huggingface_hub[hf_xet]>=0.36.0,<1"
LEGACY_FULL_PACK_IDS = frozenset({"3d", "3dgs", "image", "video"})
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PACK_ID_RE = re.compile(r"^[a-z0-9]+(?:[._-][a-z0-9]+)*$")
_HOST_LABEL_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")
# Published upstream checksums:
# https://github.com/cloudflare/cloudflared/releases/tag/2026.7.2
CLOUDFLARED_VERSION = "2026.7.2"
CLOUDFLARED_ASSETS = {
    "amd64": (
        "cloudflared-linux-amd64",
        "ec905ea7b7e327ff8abdde8cb64697a2152de74dbcdbf6aec9db8364eb3886cd",
    ),
    "arm64": (
        "cloudflared-linux-arm64",
        "405df476437e027fc6d18729a5a77155c0a33a6082aeee60a799a688f3052e66",
    ),
}


class RuntimeContractError(RuntimeError):
    """Raised before runtime mutation when a resolved lock is invalid."""


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_relative_path(value: object, *, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise RuntimeContractError(f"{field} must be a non-empty relative path")
    posix = PurePosixPath(value)
    if posix.is_absolute() or any(part in {"", ".", ".."} for part in posix.parts):
        raise RuntimeContractError(f"{field} contains an unsafe path: {value!r}")
    return Path(*posix.parts)


def safe_pack_id(value: object) -> str:
    if not isinstance(value, str) or not _PACK_ID_RE.fullmatch(value):
        raise RuntimeContractError(f"invalid pack id: {value!r}")
    return value


def validate_repository(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise RuntimeContractError(f"{field} must be a string")
    parsed = urlparse(value)
    if parsed.scheme != "https" or not parsed.netloc:
        raise RuntimeContractError(f"{field} must be an absolute HTTPS URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise RuntimeContractError(f"{field} must not include credentials, query, or fragment")
    return value


def _trusted_colab_proxy_host(hostname: str) -> bool:
    return (
        hostname == "colab.research.google.com"
        or (
            hostname.endswith(".prod.colab.dev")
            and hostname != "prod.colab.dev"
        )
        or (
            hostname.endswith(".colab.googleusercontent.com")
            and hostname != "colab.googleusercontent.com"
        )
    )


def _validated_colab_url(value: object, *, field: str, allow_frontend: bool) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
        or "?" in value
        or "#" in value
    ):
        raise RuntimeContractError(f"{field} must be a trusted HTTPS Colab origin")
    try:
        parsed = urlparse(value)
        port = parsed.port
        hostname = (parsed.hostname or "").lower()
    except ValueError as error:
        raise RuntimeContractError(f"{field} must be a trusted HTTPS Colab origin") from error
    labels = hostname.split(".")
    trusted_host = _trusted_colab_proxy_host(hostname) and (
        allow_frontend or hostname != "colab.research.google.com"
    )
    if (
        parsed.scheme != "https"
        or not trusted_host
        or any(not _HOST_LABEL_RE.fullmatch(label) for label in labels)
        or parsed.username
        or parsed.password
        or parsed.path not in {"", "/"}
        or parsed.params
        or parsed.query
        or parsed.fragment
        or port not in {None, 443}
    ):
        raise RuntimeContractError(f"{field} must be a trusted HTTPS Colab origin")
    return f"https://{hostname}"


def validate_colab_cors_origin(value: object) -> str:
    return _validated_colab_url(
        value,
        field="COMFYCOLAB_CORS_ORIGIN",
        allow_frontend=True,
    )


def validate_colab_proxy_url(value: object) -> str:
    return (
        _validated_colab_url(
            value,
            field="COMFYCOLAB_PROXY_URL",
            allow_frontend=False,
        )
        + "/"
    )


def colab_cors_origin(inherited_environment: Mapping[str, str]) -> str:
    return validate_colab_cors_origin(
        inherited_environment.get(
            "COMFYCOLAB_CORS_ORIGIN",
            DEFAULT_COLAB_CORS_ORIGIN,
        )
    )


def comfy_launch_command(
    port: int,
    *,
    colab_proxy: bool,
    inherited_environment: Mapping[str, str],
) -> list[str]:
    command = [
        sys.executable,
        "main.py",
        "--listen",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    if colab_proxy:
        cors_origin = colab_cors_origin(inherited_environment)
        command.extend(["--enable-cors-header", cors_origin])
    return command


def validate_commit(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not _COMMIT_RE.fullmatch(value):
        raise RuntimeContractError(f"{field} must be an immutable 40-character commit")
    return value


def load_lock(path: Path, expected_sha256: str) -> dict[str, Any]:
    if not _SHA256_RE.fullmatch(expected_sha256):
        raise RuntimeContractError("expected lock digest is invalid")
    lock_bytes = path.read_bytes()
    actual = hashlib.sha256(lock_bytes).hexdigest()
    if actual != expected_sha256:
        raise RuntimeContractError(
            f"lock digest mismatch: expected {expected_sha256}, got {actual}"
        )
    try:
        parsed = ComfyColabLockV1.from_bytes(lock_bytes)
    except PackContractError as error:
        raise RuntimeContractError(f"lock contract is invalid: {error}") from error
    if parsed.canonical_bytes() != lock_bytes:
        raise RuntimeContractError("lock bytes are not canonical JSON")
    lock = parsed.to_dict()
    comfyui = lock.get("comfyui")
    packs = lock.get("packs")
    dependencies = lock.get("dependencies", [])
    if not isinstance(comfyui, dict):
        raise RuntimeContractError("lock.comfyui must be an object")
    validate_repository(comfyui.get("repository"), field="lock.comfyui.repository")
    validate_commit(comfyui.get("commit"), field="lock.comfyui.commit")
    if not isinstance(packs, list):
        raise RuntimeContractError("lock.packs must be an array")
    if not isinstance(dependencies, list):
        raise RuntimeContractError("lock.dependencies must be an array")
    seen_packs: set[str] = set()
    for pack in packs:
        if not isinstance(pack, dict):
            raise RuntimeContractError("lock pack entries must be objects")
        pack_id = safe_pack_id(pack.get("id"))
        if pack_id in seen_packs:
            raise RuntimeContractError(f"duplicate pack in lock: {pack_id}")
        seen_packs.add(pack_id)
        validate_repository(pack.get("repository"), field=f"pack {pack_id} repository")
        validate_commit(pack.get("commit"), field=f"pack {pack_id} commit")
        digest = pack.get("manifest_sha256")
        if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
            raise RuntimeContractError(f"pack {pack_id} manifest digest is invalid")
    return lock


def load_stage1_config(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_bytes())
    except json.JSONDecodeError as error:
        raise RuntimeContractError("stage-1 configuration is not valid JSON") from error
    if not isinstance(payload, dict) or payload.get("schema") != 1:
        raise RuntimeContractError("stage-1 configuration has an unsupported schema")
    expected = {
        "schema",
        "port",
        "refresh",
        "colab_proxy",
        "runtime_mode",
        "lock_path",
        "lock_sha256",
        "core_dir",
        "accepted_licenses",
    }
    if set(payload) != expected:
        raise RuntimeContractError("stage-1 configuration fields are invalid")
    if type(payload["port"]) is not int or not 1 <= payload["port"] <= 65535:
        raise RuntimeContractError("stage-1 port is invalid")
    if type(payload["refresh"]) is not bool or type(payload["colab_proxy"]) is not bool:
        raise RuntimeContractError("stage-1 flags must be booleans")
    if payload["runtime_mode"] not in {"generic", "legacy-full"}:
        raise RuntimeContractError("stage-1 runtime_mode is invalid")
    if not isinstance(payload["lock_path"], str) or not Path(payload["lock_path"]).is_absolute():
        raise RuntimeContractError("stage-1 lock_path must be absolute")
    if not isinstance(payload["core_dir"], str) or not Path(payload["core_dir"]).is_absolute():
        raise RuntimeContractError("stage-1 core_dir must be absolute")
    accepted = payload["accepted_licenses"]
    if (
        not isinstance(accepted, list)
        or accepted != sorted(set(accepted))
        or any(
            not isinstance(item, str)
            or not re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,127}", item)
            for item in accepted
        )
    ):
        raise RuntimeContractError("stage-1 accepted_licenses is invalid")
    digest = payload["lock_sha256"]
    if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
        raise RuntimeContractError("stage-1 lock_sha256 is invalid")
    return payload


def validate_runtime_support(lock: Mapping[str, Any]) -> None:
    """Reject lock features that the generic stage-1 installer cannot apply."""

    dependencies = lock.get("dependencies", [])
    environments = lock.get("environments", [])
    if not isinstance(dependencies, list) or not isinstance(environments, list):
        raise RuntimeContractError("lock dependency and environment arrays are invalid")
    for dependency in dependencies:
        if not isinstance(dependency, dict):
            raise RuntimeContractError("lock dependency entries must be objects")
        if (
            dependency.get("kind") == "git"
            and dependency.get("requirements_file") is not None
            and dependency.get("requirements_format", "requirements.txt")
            == "comfycolab-environment-toml"
        ):
            raise RuntimeContractError(
                f"dependency {dependency.get('id')} requires the pack environment-TOML "
                "installer, which is not available in the generic runtime yet"
            )
    for environment in environments:
        if not isinstance(environment, dict):
            raise RuntimeContractError("lock environment entries must be objects")
        environment_id = safe_pack_id(environment.get("id"))
        requirements = environment.get("python_requirements", [])
        systems = environment.get("system_dependencies", [])
        if not isinstance(requirements, list) or not isinstance(systems, list):
            raise RuntimeContractError(f"environment {environment_id} arrays are invalid")
        for dependency in systems:
            if not isinstance(dependency, dict) or dependency.get("manager") != "apt":
                manager = (
                    dependency.get("manager")
                    if isinstance(dependency, dict)
                    else None
                )
                raise RuntimeContractError(
                    f"environment {environment_id} requires unsupported "
                    f"system manager {manager!r}"
                )
        if (
            environment.get("kind") == "isolated"
            and environment.get("cache_profile")
            and not requirements
        ):
            raise RuntimeContractError(
                f"environment {environment_id} declares cache profile "
                f"{environment['cache_profile']!r} without a generic restore contract"
            )


def validate_pack_license_gates(
    lock: Mapping[str, Any],
    *,
    accepted_licenses: set[str],
) -> None:
    packs = lock.get("packs", [])
    if not isinstance(packs, list):
        raise RuntimeContractError("lock.packs must be an array")
    for pack in packs:
        if not isinstance(pack, dict):
            raise RuntimeContractError("lock pack entries must be objects")
        pack_id = safe_pack_id(pack.get("id"))
        _require_license_gate(
            pack,
            accepted_licenses=accepted_licenses,
            owner=f"pack {pack_id}",
        )


def run(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    input_text: str | None = None,
    capture_output: bool = False,
    timeout: int | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    rendered = [str(part) for part in command]
    print(f"[comfycolab] $ {' '.join(rendered)}", flush=True)
    return subprocess.run(
        rendered,
        cwd=cwd,
        env=dict(env) if env is not None else None,
        input=input_text,
        text=True,
        capture_output=capture_output,
        timeout=timeout,
        check=check,
    )


def pip_check_conflicts() -> tuple[str, ...]:
    """Return normalized conflicts without failing on a pre-conflicted host image."""

    result = run(
        [sys.executable, "-m", "pip", "check"],
        capture_output=True,
        check=False,
    )
    conflicts = tuple(
        sorted(
            {
                line.strip()
                for line in result.stdout.splitlines()
                if line.strip() and line.strip() != "No broken requirements found."
            }
        )
    )
    if result.returncode == 0:
        return ()
    if result.returncode == 1 and conflicts:
        return conflicts
    detail = (result.stderr or result.stdout or "no diagnostic output").strip()
    raise RuntimeContractError(
        "pip check could not complete: " + detail[-2000:]
    )


def reject_new_pip_conflicts(
    baseline: Sequence[str],
    current: Sequence[str],
) -> None:
    introduced = sorted(set(current) - set(baseline))
    if introduced:
        raise RuntimeContractError(
            "ComfyColab introduced Python package conflicts:\n- "
            + "\n- ".join(introduced)
        )


def load_or_create_pip_baseline() -> tuple[str, ...]:
    """Persist the clean-runtime conflict set so retries cannot redefine it."""

    path = PIP_BASELINE_FILE
    if path.is_symlink():
        raise RuntimeContractError(
            "persisted pip baseline is invalid. Restart the Colab runtime."
        )
    if path.exists():
        if not path.is_file():
            raise RuntimeContractError(
                "persisted pip baseline is invalid. Restart the Colab runtime."
            )
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeContractError(
                "persisted pip baseline is invalid. Restart the Colab runtime."
            ) from error
        expected = {"schema", "python", "conflicts"}
        python_identity = payload.get("python") if isinstance(payload, dict) else None
        conflicts = payload.get("conflicts") if isinstance(payload, dict) else None
        valid = (
            isinstance(payload, dict)
            and set(payload) == expected
            and payload.get("schema") == 1
            and isinstance(python_identity, dict)
            and set(python_identity) == {"executable", "version"}
            and python_identity.get("executable") == sys.executable
            and python_identity.get("version") == platform.python_version()
            and isinstance(conflicts, list)
            and all(isinstance(item, str) and item for item in conflicts)
            and conflicts == sorted(set(conflicts))
        )
        if not valid:
            raise RuntimeContractError(
                "persisted pip baseline is invalid. Restart the Colab runtime."
            )
        return tuple(conflicts)

    conflicts = pip_check_conflicts()
    payload = {
        "schema": 1,
        "python": {
            "executable": sys.executable,
            "version": platform.python_version(),
        },
        "conflicts": list(conflicts),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    if temporary.is_symlink() or temporary.exists():
        temporary.unlink()
    temporary.write_bytes(canonical_json_bytes(payload))
    temporary.replace(path)
    return conflicts


def clone_at_commit(repository: str, destination: Path, commit: str) -> None:
    validate_repository(repository, field="repository")
    validate_commit(commit, field="commit")
    remove_managed_path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            "git",
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            repository,
            str(destination),
        ]
    )
    run(["git", "fetch", "origin", commit, "--depth", "1"], cwd=destination)
    run(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=destination)
    result = run(["git", "rev-parse", "HEAD"], cwd=destination, capture_output=True)
    if result.stdout.strip() != commit:
        raise RuntimeContractError(f"checkout did not resolve requested commit: {destination}")


def dependency_destination(dependency: Mapping[str, Any]) -> Path:
    relative = safe_relative_path(
        dependency.get("destination"),
        field=f"dependency {dependency.get('id')!r} destination",
    )
    if dependency.get("scope") == "comfyui":
        destination = COMFY_DIR / relative
    else:
        destination = DEPENDENCIES_DIR / relative
    resolved_root = (COMFY_DIR if dependency.get("scope") == "comfyui" else DEPENDENCIES_DIR).resolve()
    resolved = destination.resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise RuntimeContractError("dependency destination escapes its allowed root")
    return destination


def verify_huggingface_artifacts(
    dependency_id: str,
    destination: Path,
    artifacts: object,
) -> None:
    if not isinstance(artifacts, list):
        raise RuntimeContractError(
            f"dependency {dependency_id} Hugging Face artifacts must be an array"
        )
    root = destination.resolve()
    for index, artifact in enumerate(artifacts):
        if not isinstance(artifact, dict):
            raise RuntimeContractError(
                f"dependency {dependency_id} artifact {index} must be an object"
            )
        relative = safe_relative_path(
            artifact.get("path"),
            field=f"dependency {dependency_id} artifact {index} path",
        )
        expected_size = artifact.get("bytes")
        if (
            type(expected_size) is not int
            or expected_size < 0
        ):
            raise RuntimeContractError(
                f"dependency {dependency_id} artifact {relative} byte count is invalid"
            )
        expected_sha256 = artifact.get("sha256")
        if (
            not isinstance(expected_sha256, str)
            or not _SHA256_RE.fullmatch(expected_sha256)
        ):
            raise RuntimeContractError(
                f"dependency {dependency_id} artifact {relative} digest is invalid"
            )
        path = (destination / relative).resolve()
        if path != root and root not in path.parents:
            raise RuntimeContractError(
                f"dependency {dependency_id} artifact {relative} escapes its destination"
            )
        if not path.is_file():
            raise RuntimeContractError(
                f"dependency {dependency_id} artifact is missing: {relative}"
            )
        actual_size = path.stat().st_size
        if actual_size != expected_size:
            raise RuntimeContractError(
                f"dependency {dependency_id} artifact size mismatch for {relative}: "
                f"expected {expected_size}, got {actual_size}"
            )
        actual_sha256 = sha256_file(path)
        if actual_sha256 != expected_sha256:
            raise RuntimeContractError(
                f"dependency {dependency_id} artifact digest mismatch for {relative}"
            )


def install_dependency(
    dependency: Mapping[str, Any],
    *,
    resolved_paths: dict[str, str],
    pack_roots: Mapping[str, Path],
    accepted_licenses: set[str],
) -> None:
    dependency_id = safe_pack_id(dependency.get("id"))
    kind = dependency.get("kind")
    install_phase = dependency.get("install_phase", "bootstrap")
    if install_phase not in {"bootstrap", "lazy"}:
        raise RuntimeContractError(
            f"dependency {dependency_id} install_phase is unsupported: {install_phase!r}"
        )
    _require_license_gate(
        dependency,
        accepted_licenses=accepted_licenses,
        owner=f"dependency {dependency_id}",
    )
    destination = dependency_destination(dependency)
    if install_phase == "lazy":
        resolved_paths[dependency_id] = str(destination)
        print(f"[comfycolab] Deferring lazy dependency {dependency_id}.", flush=True)
        return
    if kind == "git":
        repository = validate_repository(
            dependency.get("repository"),
            field=f"dependency {dependency_id} repository",
        )
        commit = validate_commit(
            dependency.get("ref") or dependency.get("commit"),
            field=f"dependency {dependency_id} ref",
        )
        clone_at_commit(repository, destination, commit)
        requirements = dependency.get("requirements_file")
        if requirements is not None:
            requirements_source = dependency.get("requirements_source", "dependency")
            if requirements_source == "dependency":
                requirements_root = destination
            elif requirements_source == "pack":
                requesters = dependency.get("requested_by", [])
                if not isinstance(requesters, list) or len(requesters) != 1:
                    raise RuntimeContractError(
                        f"dependency {dependency_id} pack requirements need exactly one owner"
                    )
                owner = safe_pack_id(requesters[0])
                try:
                    requirements_root = pack_roots[owner]
                except KeyError as error:
                    raise RuntimeContractError(
                        f"dependency {dependency_id} requirements owner is unavailable"
                    ) from error
            else:
                raise RuntimeContractError(
                    f"dependency {dependency_id} requirements_source is unsupported"
                )
            requirements_path = requirements_root / safe_relative_path(
                requirements,
                field=f"dependency {dependency_id} requirements_file",
            )
            if not requirements_path.is_file():
                raise RuntimeContractError(
                    f"dependency {dependency_id} requirements file is missing"
                )
            requirements_format = dependency.get("requirements_format", "requirements.txt")
            if requirements_format == "requirements.txt":
                run([sys.executable, "-m", "pip", "install", "-r", str(requirements_path)])
            elif requirements_format == "pyproject.toml":
                run([sys.executable, "-m", "pip", "install", str(requirements_path.parent)])
            elif requirements_format == "comfycolab-environment-toml":
                raise RuntimeContractError(
                    f"dependency {dependency_id} requires the pack environment-TOML "
                    "installer, which is not available in the generic runtime yet"
                )
            else:
                raise RuntimeContractError(
                    f"dependency {dependency_id} requirements_format is unsupported"
                )
    elif kind == "artifact":
        url = validate_repository(
            dependency.get("url"),
            field=f"dependency {dependency_id} url",
        )
        expected = dependency.get("sha256")
        if not isinstance(expected, str) or not _SHA256_RE.fullmatch(expected):
            raise RuntimeContractError(f"dependency {dependency_id} sha256 is invalid")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".part")
        urllib.request.urlretrieve(url, temporary)
        if sha256_file(temporary) != expected:
            temporary.unlink(missing_ok=True)
            raise RuntimeContractError(f"dependency {dependency_id} artifact digest mismatch")
        temporary.replace(destination)
    elif kind == "huggingface":
        repository = dependency.get("repository")
        commit = dependency.get("ref")
        if not isinstance(repository, str) or not repository or "/" not in repository:
            raise RuntimeContractError(f"dependency {dependency_id} HF repository is invalid")
        validate_commit(commit, field=f"dependency {dependency_id} ref")
        destination.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
        os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
        try:
            from huggingface_hub import snapshot_download
        except ImportError as error:
            raise RuntimeContractError(
                "huggingface_hub with hf-xet is required for Hugging Face dependencies"
            ) from error
        token = os.environ.get("HF_TOKEN") or os.environ.get(
            "HUGGING_FACE_HUB_TOKEN"
        )
        candidates: tuple[str | bool | None, ...] = (
            (token, False) if token else (False,)
        )
        last_error: Exception | None = None
        for candidate in candidates:
            try:
                snapshot_download(
                    repo_id=repository,
                    revision=commit,
                    local_dir=str(destination),
                    token=candidate,
                )
                last_error = None
                break
            except Exception as error:
                last_error = error
        if last_error is not None:
            raise RuntimeContractError(
                f"unable to download Hugging Face dependency "
                f"{repository}@{commit}"
            ) from last_error
        verify_huggingface_artifacts(
            dependency_id,
            destination,
            dependency.get("artifacts", []),
        )
    else:
        raise RuntimeContractError(
            f"dependency {dependency_id} uses unsupported kind {kind!r}"
        )
    resolved_paths[dependency_id] = str(destination)


def load_pack_manifest(pack_root: Path, pack: Mapping[str, Any]) -> dict[str, Any]:
    path = pack_root / "comfycolab-pack.json"
    if not path.is_file():
        raise RuntimeContractError(f"pack {pack.get('id')} has no comfycolab-pack.json")
    actual = sha256_file(path)
    if actual != pack.get("manifest_sha256"):
        raise RuntimeContractError(
            f"pack {pack.get('id')} manifest digest mismatch: {actual}"
        )
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RuntimeContractError(f"pack {pack.get('id')} manifest is invalid JSON") from error
    if not isinstance(manifest, dict) or manifest.get("schema") != 1:
        raise RuntimeContractError(f"pack {pack.get('id')} manifest schema is invalid")
    if manifest.get("id") != pack.get("id"):
        raise RuntimeContractError(f"pack {pack.get('id')} manifest ID mismatch")
    return manifest


def clone_packs(lock: Mapping[str, Any]) -> tuple[dict[str, Path], dict[str, dict[str, Any]]]:
    roots: dict[str, Path] = {}
    manifests: dict[str, dict[str, Any]] = {}
    PACKS_DIR.mkdir(parents=True, exist_ok=True)
    for pack in lock.get("packs", []):
        assert isinstance(pack, dict)
        pack_id = safe_pack_id(pack["id"])
        root = PACKS_DIR / pack_id
        clone_at_commit(str(pack["repository"]), root, str(pack["commit"]))
        roots[pack_id] = root
        manifests[pack_id] = load_pack_manifest(root, pack)
    return roots, manifests


def validate_manifest_compatibility(
    manifests: Mapping[str, Mapping[str, Any]],
    *,
    comfyui_commit: str,
) -> None:
    for pack_id, manifest in manifests.items():
        compatibility = manifest.get("compatibility")
        if not isinstance(compatibility, dict):
            raise RuntimeContractError(f"pack {pack_id} compatibility is invalid")
        comfyui = compatibility.get("comfyui")
        if not isinstance(comfyui, dict):
            raise RuntimeContractError(f"pack {pack_id} ComfyUI compatibility is invalid")
        compatible_refs = comfyui.get("compatible_refs")
        if not isinstance(compatible_refs, list) or comfyui_commit not in compatible_refs:
            raise RuntimeContractError(
                f"pack {pack_id} is incompatible with ComfyUI {comfyui_commit}"
            )


def install_dependencies(
    lock: Mapping[str, Any],
    *,
    pack_roots: Mapping[str, Path],
    accepted_licenses: set[str],
) -> dict[str, str]:
    resolved_paths: dict[str, str] = {}
    dependencies = lock.get("dependencies", [])
    if not isinstance(dependencies, list):
        raise RuntimeContractError("lock dependencies must be an array")
    for dependency in dependencies:
        if not isinstance(dependency, dict):
            raise RuntimeContractError("lock dependency entries must be objects")
        if dependency.get("id") == "comfyui":
            raise RuntimeContractError(
                "dependency ID 'comfyui' is reserved for the locked ComfyUI checkout"
            )
        if (
            dependency.get("requirements_file") is not None
            and dependency.get("requirements_source", "dependency") == "pack"
        ):
            requesters = dependency.get("requested_by", [])
            if not isinstance(requesters, list) or len(requesters) != 1:
                raise RuntimeContractError(
                    f"dependency {dependency.get('id')} pack requirements need "
                    "exactly one owner"
                )
            owner = safe_pack_id(requesters[0])
            if owner not in pack_roots:
                raise RuntimeContractError(
                    f"dependency {dependency.get('id')} requirements owner is unavailable"
                )
    for dependency in dependencies:
        assert isinstance(dependency, dict)
        install_dependency(
            dependency,
            resolved_paths=resolved_paths,
            pack_roots=pack_roots,
            accepted_licenses=accepted_licenses,
        )
    return resolved_paths


def _require_license_gate(
    specification: Mapping[str, Any],
    *,
    accepted_licenses: set[str],
    owner: str,
) -> None:
    gate = specification.get("license_gate")
    if gate is None:
        return
    if not isinstance(gate, str) or not gate:
        raise RuntimeContractError(f"{owner} has an invalid license gate")
    if gate not in accepted_licenses:
        raise RuntimeContractError(
            f"{owner} requires explicit license acceptance: {gate}. "
            f"Restart with --accept-license {gate} after reviewing the pack terms."
        )


def _python_requirement_argument(requirement: Mapping[str, Any]) -> str:
    name = requirement.get("name")
    specifier = requirement.get("specifier", "")
    if not isinstance(name, str) or not name:
        raise RuntimeContractError("Python requirement name is invalid")
    if not isinstance(specifier, str):
        raise RuntimeContractError(f"Python requirement {name} specifier is invalid")
    url = requirement.get("url")
    sha256 = requirement.get("sha256")
    if url is None:
        return f"{name}{specifier}"
    validate_repository(url, field=f"Python requirement {name} URL")
    if not isinstance(sha256, str) or not _SHA256_RE.fullmatch(sha256):
        raise RuntimeContractError(f"Python requirement {name} hash is invalid")
    return f"{name} @ {url}#sha256={sha256}"


def install_environments(
    lock: Mapping[str, Any],
    *,
    accepted_licenses: set[str],
) -> dict[str, str]:
    environments = lock.get("environments", [])
    if not isinstance(environments, list):
        raise RuntimeContractError("lock.environments must be an array")
    validated: list[tuple[str, dict[str, Any]]] = []
    seen_environment_ids: set[str] = set()
    for specification in environments:
        if not isinstance(specification, dict):
            raise RuntimeContractError("environment entries must be objects")
        environment_id = safe_pack_id(specification.get("id"))
        if environment_id in seen_environment_ids:
            raise RuntimeContractError(
                f"duplicate environment ID in lock: {environment_id}"
            )
        seen_environment_ids.add(environment_id)
        validated.append((environment_id, specification))

    resolved: dict[str, str] = {}
    for environment_id, specification in validated:
        _require_license_gate(
            specification,
            accepted_licenses=accepted_licenses,
            owner=f"environment {environment_id}",
        )
        kind = specification.get("kind")
        requirements = specification.get("python_requirements", [])
        systems = specification.get("system_dependencies", [])
        if not isinstance(requirements, list) or not isinstance(systems, list):
            raise RuntimeContractError(f"environment {environment_id} arrays are invalid")
        for dependency in systems:
            if not isinstance(dependency, dict):
                raise RuntimeContractError(
                    f"environment {environment_id} system dependency is invalid"
                )
            manager = dependency.get("manager")
            name = dependency.get("name")
            version = dependency.get("version")
            if not isinstance(name, str) or not name:
                raise RuntimeContractError(
                    f"environment {environment_id} system package is invalid"
                )
            if manager != "apt":
                raise RuntimeContractError(
                    f"environment {environment_id} requires unsupported "
                    f"system manager {manager!r}"
                )
            package = f"{name}={version}" if isinstance(version, str) else name
            run(["apt-get", "install", "-y", package])
        if kind == "main":
            python = Path(sys.executable)
        elif kind == "isolated":
            if specification.get("cache_profile") and not requirements:
                raise RuntimeContractError(
                    f"environment {environment_id} declares cache profile "
                    f"{specification['cache_profile']!r} without a generic restore contract"
                )
            environment_root = ENVIRONMENTS_DIR / environment_id
            python = environment_root / "bin" / "python"
            if not python.is_file():
                requested_python = specification.get("python")
                if requested_python not in {None, "", platform.python_version()}:
                    raise RuntimeContractError(
                        f"environment {environment_id} requires Python "
                        f"{requested_python}; generic venv creation cannot provide it"
                    )
                run([sys.executable, "-m", "venv", str(environment_root)])
        else:
            raise RuntimeContractError(
                f"environment {environment_id} uses unsupported kind {kind!r}"
            )
        arguments = [
            _python_requirement_argument(requirement)
            for requirement in requirements
            if isinstance(requirement, dict)
        ]
        if len(arguments) != len(requirements):
            raise RuntimeContractError(
                f"environment {environment_id} Python requirement is invalid"
            )
        if arguments:
            run([str(python), "-m", "pip", "install", *arguments])
        resolved[environment_id] = str(python)
    return resolved


def resolved_runtime_environment(lock: Mapping[str, Any]) -> dict[str, str]:
    entries = lock.get("runtime_env", [])
    if not isinstance(entries, list):
        raise RuntimeContractError("lock.runtime_env must be an array")
    environment: dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise RuntimeContractError("lock runtime_env entries must be objects")
        name = entry.get("name")
        value = entry.get("value")
        if (
            not isinstance(name, str)
            or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name)
            or not isinstance(value, str)
        ):
            raise RuntimeContractError("lock runtime_env entry is invalid")
        if name in environment and environment[name] != value:
            raise RuntimeContractError(f"runtime environment conflict for {name}")
        environment[name] = value
    return environment


def apply_content_addressed_patch(repository: Path, specification_path: Path) -> str:
    specification = json.loads(specification_path.read_text(encoding="utf-8"))
    if not isinstance(specification, dict) or specification.get("schema") != 1:
        raise RuntimeContractError(f"unsupported patch schema: {specification_path}")
    patch_id = specification.get("patch_id")
    expected_revision = specification.get("revision")
    if not isinstance(patch_id, str) or not patch_id:
        raise RuntimeContractError("patch ID is missing")
    validate_commit(expected_revision, field=f"patch {patch_id} revision")
    result = run(["git", "rev-parse", "HEAD"], cwd=repository, capture_output=True)
    if result.stdout.strip() != expected_revision:
        raise RuntimeContractError(f"patch {patch_id} target revision mismatch")
    repository_root = repository.resolve()
    prepared: list[tuple[Path, str, int]] = []
    states: set[str] = set()
    files = specification.get("files")
    if not isinstance(files, list) or not files:
        raise RuntimeContractError(f"patch {patch_id} has no files")
    for file_specification in files:
        if not isinstance(file_specification, dict):
            raise RuntimeContractError(f"patch {patch_id} file entry is invalid")
        relative = safe_relative_path(
            file_specification.get("path"),
            field=f"patch {patch_id} path",
        )
        path = (repository / relative).resolve()
        if path != repository_root and repository_root not in path.parents:
            raise RuntimeContractError(f"patch {patch_id} path escapes repository")
        before_sha256 = file_specification.get("before_sha256")
        after_sha256 = file_specification.get("after_sha256")
        actual_sha256 = sha256_file(path)
        if actual_sha256 == after_sha256:
            states.add("after")
            continue
        if actual_sha256 != before_sha256:
            raise RuntimeContractError(f"patch {patch_id} refused drifted file {relative}")
        states.add("before")
        content = path.read_text(encoding="utf-8")
        replacements = file_specification.get("replacements")
        if not isinstance(replacements, list) or not replacements:
            raise RuntimeContractError(f"patch {patch_id} has no replacements")
        for replacement in replacements:
            if not isinstance(replacement, dict):
                raise RuntimeContractError(f"patch {patch_id} replacement is invalid")
            before_lines = replacement.get("before_lines")
            after_lines = replacement.get("after_lines")
            occurrences = replacement.get("occurrences", 1)
            if (
                not isinstance(before_lines, list)
                or not isinstance(after_lines, list)
                or type(occurrences) is not int
                or occurrences < 1
            ):
                raise RuntimeContractError(f"patch {patch_id} replacement contract is invalid")
            before = "\n".join(str(line) for line in before_lines) + "\n"
            after = "\n".join(str(line) for line in after_lines)
            if after_lines:
                after += "\n"
            if not before or content.count(before) != occurrences:
                raise RuntimeContractError(f"patch {patch_id} replacement count mismatch")
            content = content.replace(before, after, occurrences)
        if hashlib.sha256(content.encode("utf-8")).hexdigest() != after_sha256:
            raise RuntimeContractError(f"patch {patch_id} produced an unexpected digest")
        prepared.append((path, content, path.stat().st_mode))
    if states == {"after"}:
        return patch_id
    if states != {"before"}:
        raise RuntimeContractError(f"patch {patch_id} is partially applied")
    for path, content, mode in prepared:
        temporary = path.with_suffix(path.suffix + ".comfycolab-patch")
        temporary.write_text(content, encoding="utf-8")
        temporary.chmod(mode)
        temporary.replace(path)
    return patch_id


def apply_patches(
    lock: Mapping[str, Any],
    *,
    pack_roots: Mapping[str, Path],
    resolved_paths: Mapping[str, str],
) -> list[str]:
    applied: list[str] = []
    for patch in lock.get("patches", []):
        if not isinstance(patch, dict):
            raise RuntimeContractError("lock patch entries must be objects")
        requested_by = patch.get("requested_by")
        if not isinstance(requested_by, list) or not requested_by:
            raise RuntimeContractError(f"patch {patch.get('id')} has no owner")
        owner = safe_pack_id(requested_by[0])
        pack_root = pack_roots.get(owner)
        if pack_root is None:
            raise RuntimeContractError(f"patch owner is absent: {owner}")
        specification = patch.get("specification") or patch.get("path")
        specification_path = pack_root / safe_relative_path(
            specification,
            field=f"patch {patch.get('id')} specification",
        )
        expected_sha = patch.get("specification_sha256") or patch.get("sha256")
        if expected_sha is not None and sha256_file(specification_path) != expected_sha:
            raise RuntimeContractError(f"patch {patch.get('id')} specification digest mismatch")
        target = patch.get("target")
        if not isinstance(target, str) or not resolved_paths.get(target):
            raise RuntimeContractError(f"patch {patch.get('id')} target is unresolved")
        applied.append(
            apply_content_addressed_patch(Path(resolved_paths[target]), specification_path)
        )
    return applied


def link_node_roots(
    manifests: Mapping[str, Mapping[str, Any]],
    *,
    pack_roots: Mapping[str, Path],
) -> None:
    custom_nodes = COMFY_DIR / "custom_nodes"
    custom_nodes.mkdir(parents=True, exist_ok=True)
    seen_targets: set[str] = set()
    for pack_id, manifest in manifests.items():
        node_roots = manifest.get("node_roots", [])
        if not isinstance(node_roots, list):
            raise RuntimeContractError(f"pack {pack_id} node_roots must be an array")
        for node_root in node_roots:
            if not isinstance(node_root, dict):
                raise RuntimeContractError(f"pack {pack_id} node root is invalid")
            source_relative = safe_relative_path(
                node_root.get("source"),
                field=f"pack {pack_id} node source",
            )
            target_relative = safe_relative_path(
                node_root.get("target"),
                field=f"pack {pack_id} node target",
            )
            if len(target_relative.parts) != 1:
                raise RuntimeContractError(f"pack {pack_id} node target must be one directory")
            target_name = target_relative.name
            if target_name in seen_targets:
                raise RuntimeContractError(f"duplicate node target: {target_name}")
            seen_targets.add(target_name)
            source = (pack_roots[pack_id] / source_relative).resolve()
            root = pack_roots[pack_id].resolve()
            if not source.is_dir() or (source != root and root not in source.parents):
                raise RuntimeContractError(f"pack {pack_id} node source is missing or unsafe")
            target = custom_nodes / target_name
            if target.is_symlink() or target.is_file():
                target.unlink()
            elif target.exists():
                shutil.rmtree(target)
            target.symlink_to(source, target_is_directory=True)


def run_post_clone_probes(
    manifests: Mapping[str, Mapping[str, Any]],
    *,
    pack_roots: Mapping[str, Path],
    resolved_paths: Mapping[str, str],
) -> None:
    for pack_id, manifest in manifests.items():
        probes = manifest.get("probes", [])
        if not isinstance(probes, list):
            raise RuntimeContractError(f"pack {pack_id} probes must be an array")
        for index, probe in enumerate(probes):
            if not isinstance(probe, dict):
                raise RuntimeContractError(f"pack {pack_id} probe {index} is invalid")
            if probe.get("phase") != "post_clone":
                continue
            target = probe.get("target", "pack")
            if target == "pack":
                root = pack_roots[pack_id]
            elif target == "comfyui":
                root = COMFY_DIR
            elif isinstance(target, str) and resolved_paths.get(target):
                root = Path(resolved_paths[target])
            else:
                raise RuntimeContractError(
                    f"pack {pack_id} probe {index} target is unresolved: {target!r}"
                )
            relative = safe_relative_path(
                probe.get("path"),
                field=f"pack {pack_id} probe {index} path",
            )
            path = (root / relative).resolve()
            resolved_root = root.resolve()
            if path != resolved_root and resolved_root not in path.parents:
                raise RuntimeContractError(f"pack {pack_id} probe {index} escapes target")
            kind = probe.get("type")
            if kind == "path_exists":
                if not path.exists():
                    raise RuntimeContractError(
                        f"pack {pack_id} required path is missing: {target}:{relative}"
                    )
            elif kind == "file_sha256":
                expected = probe.get("sha256")
                if not path.is_file() or sha256_file(path) != expected:
                    raise RuntimeContractError(
                        f"pack {pack_id} file digest probe failed: {target}:{relative}"
                    )
            elif kind in {"python_symbol", "file_symbols"}:
                if not path.is_file():
                    raise RuntimeContractError(
                        f"pack {pack_id} symbol probe file is missing: {target}:{relative}"
                    )
                symbols: list[object]
                if kind == "python_symbol":
                    symbols = [probe.get("symbol")]
                else:
                    raw_symbols = probe.get("symbols")
                    if not isinstance(raw_symbols, list):
                        raise RuntimeContractError(
                            f"pack {pack_id} file_symbols probe has invalid symbols"
                        )
                    symbols = raw_symbols
                source = path.read_text(encoding="utf-8")
                missing = [
                    symbol
                    for symbol in symbols
                    if not isinstance(symbol, str) or symbol not in source
                ]
                if missing:
                    raise RuntimeContractError(
                        f"pack {pack_id} symbol probe failed for {target}:{relative}: "
                        + ", ".join(repr(item) for item in missing)
                    )
            else:
                raise RuntimeContractError(
                    f"pack {pack_id} post-clone probe type is unsupported: {kind!r}"
                )


def run_hook(
    pack_id: str,
    pack_root: Path,
    hook_name: str,
    specification: Mapping[str, Any],
    context: Mapping[str, Any],
    arguments: Sequence[str] = (),
) -> dict[str, Any]:
    path = pack_root / safe_relative_path(
        specification.get("path"),
        field=f"pack {pack_id} hook {hook_name}",
    )
    root = pack_root.resolve()
    resolved = path.resolve()
    if resolved != root and root not in resolved.parents:
        raise RuntimeContractError(f"pack {pack_id} hook escapes pack root")
    if not path.is_file():
        raise RuntimeContractError(f"pack {pack_id} hook is missing: {hook_name}")
    timeout = specification.get("timeout_seconds", 60)
    if type(timeout) is not int or not 1 <= timeout <= 3600:
        raise RuntimeContractError(f"pack {pack_id} hook timeout is invalid")
    if not all(
        isinstance(argument, str) and argument and "\x00" not in argument
        for argument in arguments
    ):
        raise RuntimeContractError(f"pack {pack_id} hook {hook_name} arguments are invalid")
    environment = {
        name: os.environ[name]
        for name in ("HOME", "LANG", "LC_ALL", "PATH", "TMPDIR")
        if name in os.environ
    }
    environment["COMFYCOLAB_PACK_ID"] = pack_id
    environment["COMFYCOLAB_PACK_ROOT"] = str(pack_root)
    declared_write_roots = specification.get("write_roots", [])
    if (
        not isinstance(declared_write_roots, list)
        or not all(isinstance(item, str) for item in declared_write_roots)
        or len(declared_write_roots) != len(set(declared_write_roots))
    ):
        raise RuntimeContractError(f"pack {pack_id} hook {hook_name} write roots are invalid")
    supported_write_roots = {
        "pack_state": PACK_STATE_DIR / pack_id,
    }
    unknown_write_roots = sorted(
        set(declared_write_roots) - set(supported_write_roots)
    )
    if unknown_write_roots:
        raise RuntimeContractError(
            f"pack {pack_id} hook {hook_name} uses unsupported write roots: "
            + ", ".join(unknown_write_roots)
        )
    resolved_write_roots = {
        name: supported_write_roots[name].resolve()
        for name in declared_write_roots
    }
    for write_root in resolved_write_roots.values():
        write_root.mkdir(parents=True, exist_ok=True)
    hook_context = dict(context)
    hook_context["write_roots"] = {
        name: str(write_root)
        for name, write_root in sorted(resolved_write_roots.items())
    }
    environment["COMFYCOLAB_HOOK_WRITE_ROOTS"] = json.dumps(
        [str(path) for path in resolved_write_roots.values()],
        separators=(",", ":"),
    )
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    core_source_root = str(Path(__file__).resolve().parents[1])
    inherited_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        core_source_root
        if not inherited_pythonpath
        else os.pathsep.join((core_source_root, inherited_pythonpath))
    )
    if specification.get("network", "none") == "none":
        for name in (
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
            "http_proxy",
            "https_proxy",
            "all_proxy",
        ):
            environment.pop(name, None)
        environment["NO_PROXY"] = "*"
        environment["COMFYCOLAB_NETWORK"] = "none"
    try:
        result = run(
            [
                sys.executable,
                "-B",
                "-m",
                "comfycolab.hook_runner",
                str(path),
                *arguments,
            ],
            cwd=pack_root,
            env=environment,
            input_text=json.dumps(hook_context, separators=(",", ":")),
            capture_output=True,
            timeout=timeout,
        )
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or error.stdout or "").strip()
        suffix = f": {detail[-1000:]}" if detail else ""
        raise RuntimeContractError(
            f"pack {pack_id} hook {hook_name} was rejected{suffix}"
        ) from error
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeContractError(f"pack {pack_id} hook {hook_name} returned no JSON")
    try:
        payload = json.loads(lines[-1])
    except json.JSONDecodeError as error:
        raise RuntimeContractError(
            f"pack {pack_id} hook {hook_name} returned invalid JSON"
        ) from error
    if not isinstance(payload, dict):
        raise RuntimeContractError(f"pack {pack_id} hook {hook_name} returned a non-object")
    if payload.get("status") in {"error", "failed"}:
        raise RuntimeContractError(f"pack {pack_id} hook {hook_name} failed")
    for field in ("writes", "changes"):
        reported = payload.get(field, [])
        if not isinstance(reported, list):
            raise RuntimeContractError(
                f"pack {pack_id} hook {hook_name} {field} report is invalid"
            )
        for index, item in enumerate(reported):
            if not isinstance(item, dict):
                raise RuntimeContractError(
                    f"pack {pack_id} hook {hook_name} {field}[{index}] is invalid"
                )
            if set(item) != {"root", "path"}:
                raise RuntimeContractError(
                    f"pack {pack_id} hook {hook_name} {field}[{index}] fields are invalid"
                )
            root_name = item.get("root")
            if root_name not in resolved_write_roots:
                raise RuntimeContractError(
                    f"pack {pack_id} hook {hook_name} reported an undeclared write root"
                )
            safe_relative_path(
                item.get("path"),
                field=f"pack {pack_id} hook {hook_name} {field}[{index}] path",
            )
    return payload


def run_prestart_hooks(
    lock: Mapping[str, Any],
    *,
    lock_sha256: str,
    manifests: Mapping[str, Mapping[str, Any]],
    pack_roots: Mapping[str, Path],
    resolved_paths: Mapping[str, str],
) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    runtime_environment: dict[str, str] = {}
    readiness: dict[str, dict[str, Any]] = {}
    for pack_id, manifest in manifests.items():
        context = {
            "schema": 1,
            "pack": pack_id,
            "lock": lock,
            "lock_digest": lock_sha256,
            "pack_root": str(pack_roots[pack_id]),
            "comfyui_root": str(COMFY_DIR),
            "resolved_paths": resolved_paths,
        }
        hooks = manifest.get("hooks", {})
        if not isinstance(hooks, dict):
            raise RuntimeContractError(f"pack {pack_id} hooks must be an object")
        pack_readiness: dict[str, Any] = {}
        for hook_name in ("configure", "doctor", "runtime_env"):
            specification = hooks.get(hook_name)
            if specification is None:
                continue
            if not isinstance(specification, dict):
                raise RuntimeContractError(f"pack {pack_id} hook {hook_name} is invalid")
            result = run_hook(
                pack_id,
                pack_roots[pack_id],
                hook_name,
                specification,
                context,
            )
            pack_readiness[hook_name] = result
            environment = result.get("environment", {})
            if environment:
                if not isinstance(environment, dict):
                    raise RuntimeContractError(
                        f"pack {pack_id} runtime environment is invalid"
                    )
                for name, value in environment.items():
                    if not isinstance(name, str) or not isinstance(value, str):
                        raise RuntimeContractError(
                            f"pack {pack_id} runtime environment entry is invalid"
                        )
                    previous = runtime_environment.get(name)
                    if previous is not None and previous != value:
                        raise RuntimeContractError(
                            f"runtime environment conflict for {name}"
                        )
                    runtime_environment[name] = value
        declared = manifest.get("runtime_env", {})
        if declared:
            if not isinstance(declared, dict):
                raise RuntimeContractError(f"pack {pack_id} runtime_env must be an object")
            for name, value in declared.items():
                if not isinstance(name, str) or not isinstance(value, str):
                    raise RuntimeContractError(f"pack {pack_id} runtime_env entry is invalid")
                previous = runtime_environment.get(name)
                if previous is not None and previous != value:
                    raise RuntimeContractError(f"runtime environment conflict for {name}")
                runtime_environment[name] = value
        readiness[pack_id] = pack_readiness
    return runtime_environment, readiness


def install_core_requirements() -> None:
    requirements = COMFY_DIR / "requirements.txt"
    if not requirements.is_file():
        raise RuntimeContractError("ComfyUI requirements.txt is missing")
    run([sys.executable, "-m", "pip", "install", "-r", str(requirements)])
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            HUGGINGFACE_HUB_REQUIREMENT,
        ]
    )


def git_commit(directory: Path) -> str:
    return run(["git", "rev-parse", "HEAD"], cwd=directory, capture_output=True).stdout.strip()


def pid_alive(pid: object) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        waited, _ = os.waitpid(pid, os.WNOHANG)
        if waited == pid:
            return False
    except ChildProcessError:
        pass
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def stop_managed_process(pid: object) -> None:
    if not pid_alive(pid):
        return
    process_id = int(pid)
    try:
        os.killpg(os.getpgid(process_id), signal.SIGTERM)
    except OSError:
        try:
            os.kill(process_id, signal.SIGTERM)
        except OSError:
            return
    for _ in range(20):
        if not pid_alive(pid):
            return
        time.sleep(0.25)
    try:
        os.killpg(os.getpgid(process_id), signal.SIGKILL)
    except OSError:
        try:
            os.kill(process_id, signal.SIGKILL)
        except OSError:
            pass


def stop_started_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        process.wait()
        return
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    except OSError:
        process.terminate()
    try:
        process.wait(timeout=5)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except OSError:
        process.kill()
    process.wait(timeout=2)


def http_ready(port: int) -> bool:
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/object_info",
            timeout=2,
        ) as response:
            return 200 <= response.status < 500
    except Exception:
        return False


def wait_for_comfy(port: int, process: subprocess.Popen[bytes], timeout: int = 180) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            tail = COMFY_LOG.read_text(encoding="utf-8", errors="replace")[-4000:]
            raise RuntimeError(f"ComfyUI exited during startup.\n{tail}")
        if http_ready(port):
            return
        time.sleep(1)
    raise TimeoutError(f"ComfyUI did not become ready on port {port} within {timeout}s")


def cloudflared_path() -> Path:
    machine = platform.machine().lower()
    if machine in {"aarch64", "arm64"}:
        architecture = "arm64"
    elif machine in {"amd64", "x86_64"}:
        architecture = "amd64"
    else:
        raise RuntimeContractError(
            f"cloudflared has no authenticated asset for architecture {machine!r}"
        )
    asset, expected_sha256 = CLOUDFLARED_ASSETS[architecture]
    destination = STATE_DIR / f"cloudflared-{CLOUDFLARED_VERSION}-{architecture}"
    if destination.is_file() and sha256_file(destination) == expected_sha256:
        destination.chmod(0o755)
        return destination
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".part")
    url = (
        "https://github.com/cloudflare/cloudflared/releases/download/"
        f"{CLOUDFLARED_VERSION}/{asset}"
    )
    print(
        f"[comfycolab] Downloading cloudflared {CLOUDFLARED_VERSION} "
        f"({architecture})...",
        flush=True,
    )
    with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output)
    actual_sha256 = sha256_file(temporary)
    if actual_sha256 != expected_sha256:
        temporary.unlink(missing_ok=True)
        raise RuntimeContractError(
            "cloudflared digest mismatch: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    temporary.chmod(0o755)
    temporary.replace(destination)
    destination.chmod(0o755)
    return destination


def wait_for_tunnel(process: subprocess.Popen[bytes], timeout: int = 60) -> str:
    pattern = re.compile(r"https://[-a-z0-9]+\.trycloudflare\.com")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if TUNNEL_LOG.exists():
            content = TUNNEL_LOG.read_text(encoding="utf-8", errors="replace")
            if match := pattern.search(content):
                return match.group(0)
        if process.poll() is not None:
            tail = TUNNEL_LOG.read_text(encoding="utf-8", errors="replace")[-4000:]
            raise RuntimeError(f"cloudflared exited during startup.\n{tail}")
        time.sleep(0.5)
    raise TimeoutError("cloudflared did not publish a URL")


def start_cloudflare_tunnel(
    port: int,
) -> tuple[subprocess.Popen[bytes] | None, str | None, str | None]:
    process: subprocess.Popen[bytes] | None = None
    try:
        with TUNNEL_LOG.open("wb") as tunnel_log:
            process = subprocess.Popen(
                [
                    str(cloudflared_path()),
                    "tunnel",
                    "--url",
                    f"http://127.0.0.1:{port}",
                    "--no-autoupdate",
                ],
                stdout=tunnel_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        return process, wait_for_tunnel(process), None
    except Exception as error:
        if process is not None:
            stop_started_process(process)
        detail = f"{type(error).__name__}: {error}"[-2000:]
        print(
            f"[comfycolab] Cloudflare fallback unavailable ({detail}).",
            flush=True,
        )
        return None, None, detail


def request_colab_proxy_url(port: int) -> str | None:
    try:
        from google.colab.output import eval_js

        expression = f"""
(async () => {{
  if (!google.colab.kernel.accessAllowed) {{
    throw new Error("Colab kernel proxy access is not allowed");
  }}
  const proxy = await google.colab.kernel.proxyPort({port});
  return new URL("/", proxy).toString();
}})()
""".strip()
        value = eval_js(expression, timeout_sec=15)
        return validate_colab_proxy_url(value)
    except Exception as error:
        print(f"[comfycolab] Colab proxy unavailable ({error}).", flush=True)
        return None


def object_info(port: int) -> dict[str, Any]:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/object_info", timeout=15) as response:
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise RuntimeContractError("ComfyUI object_info returned a non-object")
    return payload


def validate_post_start_nodes(
    port: int,
    manifests: Mapping[str, Mapping[str, Any]],
    *,
    lock: Mapping[str, Any],
    lock_sha256: str,
    pack_roots: Mapping[str, Path],
    resolved_paths: Mapping[str, str],
) -> dict[str, dict[str, Any]]:
    inventory = object_info(port)
    readiness: dict[str, dict[str, Any]] = {}
    for pack_id, manifest in manifests.items():
        health = manifest.get("health_checks", {})
        expected = health.get("node_ids", []) if isinstance(health, dict) else []
        if not isinstance(expected, list) or not all(isinstance(item, str) for item in expected):
            raise RuntimeContractError(f"pack {pack_id} node health check is invalid")
        probes = manifest.get("probes", [])
        if not isinstance(probes, list):
            raise RuntimeContractError(f"pack {pack_id} probes must be an array")
        probe_count = 0
        probe_node_ids: list[str] = []
        for index, probe in enumerate(probes):
            if not isinstance(probe, dict):
                raise RuntimeContractError(f"pack {pack_id} probe {index} is invalid")
            if probe.get("phase") != "post_start":
                continue
            probe_count += 1
            kind = probe.get("type")
            if kind == "comfy_node_ids":
                values = probe.get("values")
                if not isinstance(values, list) or not all(
                    isinstance(item, str) for item in values
                ):
                    raise RuntimeContractError(
                        f"pack {pack_id} post-start node probe is invalid"
                    )
                probe_node_ids.extend(values)
                continue
            relative = safe_relative_path(
                probe.get("path"),
                field=f"pack {pack_id} post-start probe {index} path",
            )
            root = pack_roots[pack_id].resolve()
            path = (root / relative).resolve()
            if path != root and root not in path.parents:
                raise RuntimeContractError(
                    f"pack {pack_id} post-start probe {index} escapes pack root"
                )
            if kind == "path_exists":
                if not path.exists():
                    raise RuntimeContractError(
                        f"pack {pack_id} post-start path is missing: {relative}"
                    )
            elif kind == "file_sha256":
                expected_digest = probe.get("sha256")
                if not path.is_file() or sha256_file(path) != expected_digest:
                    raise RuntimeContractError(
                        f"pack {pack_id} post-start file digest probe failed: {relative}"
                    )
            elif kind == "python_symbol":
                symbol = probe.get("symbol")
                if (
                    not isinstance(symbol, str)
                    or not path.is_file()
                    or symbol not in path.read_text(encoding="utf-8")
                ):
                    raise RuntimeContractError(
                        f"pack {pack_id} post-start symbol probe failed: {relative}"
                    )
            else:
                raise RuntimeContractError(
                    f"pack {pack_id} post-start probe type is unsupported: {kind!r}"
                )
        required_node_ids = sorted(set(expected) | set(probe_node_ids))
        missing = sorted(set(required_node_ids) - set(inventory))
        if missing:
            raise RuntimeContractError(
                f"pack {pack_id} is missing ComfyUI nodes: {', '.join(missing)}"
            )
        command_result: dict[str, Any] | None = None
        command = health.get("command", []) if isinstance(health, dict) else []
        if command:
            if (
                not isinstance(command, list)
                or len(command) < 2
                or command[0] not in {"python", "python3"}
                or not all(isinstance(item, str) and item for item in command)
            ):
                raise RuntimeContractError(
                    f"pack {pack_id} health command is unsupported in manifest API v1"
                )
            command_result = run_hook(
                pack_id,
                pack_roots[pack_id],
                "health",
                {
                    "path": command[1],
                    "network": "none",
                    "write_roots": [],
                    "timeout_seconds": 300,
                },
                {
                    "schema": 1,
                    "pack": pack_id,
                    "lock": lock,
                    "lock_digest": lock_sha256,
                    "pack_root": str(pack_roots[pack_id]),
                    "comfyui_root": str(COMFY_DIR),
                    "resolved_paths": resolved_paths,
                },
                arguments=command[2:],
            )
        readiness[pack_id] = {
            "nodeIds": required_node_ids,
            "probeCount": probe_count,
            **({"command": command_result} if command_result is not None else {}),
        }
    return readiness


def load_state() -> dict[str, Any]:
    try:
        payload = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def remove_managed_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def reset_installation_roots() -> None:
    """Remove every lock-owned root before applying a non-reusable lock."""

    for path in (
        COMFY_DIR,
        PACKS_DIR,
        DEPENDENCIES_DIR,
        ENVIRONMENTS_DIR,
        PACK_STATE_DIR,
    ):
        remove_managed_path(path)


def load_existing_installation(
    lock: Mapping[str, Any],
) -> tuple[dict[str, Path], dict[str, dict[str, Any]], dict[str, str]]:
    comfyui = lock["comfyui"]
    assert isinstance(comfyui, dict)
    expected_comfyui_commit = str(comfyui["commit"])
    if not COMFY_DIR.is_dir() or git_commit(COMFY_DIR) != expected_comfyui_commit:
        raise RuntimeContractError("installed ComfyUI does not match the active lock")

    pack_roots: dict[str, Path] = {}
    manifests: dict[str, dict[str, Any]] = {}
    for pack in lock.get("packs", []):
        assert isinstance(pack, dict)
        pack_id = safe_pack_id(pack["id"])
        root = PACKS_DIR / pack_id
        if not root.is_dir() or git_commit(root) != pack.get("commit"):
            raise RuntimeContractError(
                f"installed pack {pack_id} does not match the active lock"
            )
        pack_roots[pack_id] = root
        manifests[pack_id] = load_pack_manifest(root, pack)

    resolved_paths = {"comfyui": str(COMFY_DIR)}
    for dependency in lock.get("dependencies", []):
        assert isinstance(dependency, dict)
        dependency_id = safe_pack_id(dependency.get("id"))
        destination = dependency_destination(dependency)
        if dependency.get("install_phase", "bootstrap") != "lazy" and not destination.exists():
            raise RuntimeContractError(
                f"installed dependency {dependency_id} is missing"
            )
        resolved_paths[dependency_id] = str(destination)

    for specification in lock.get("environments", []):
        assert isinstance(specification, dict)
        environment_id = safe_pack_id(specification.get("id"))
        kind = specification.get("kind")
        if kind == "main":
            python = Path(sys.executable)
        elif kind == "isolated":
            python = ENVIRONMENTS_DIR / environment_id / "bin" / "python"
        else:
            raise RuntimeContractError(
                f"environment {environment_id} uses unsupported kind {kind!r}"
            )
        if not python.is_file():
            raise RuntimeContractError(
                f"installed environment {environment_id} is missing"
            )
        resolved_paths[f"environment:{environment_id}"] = str(python)
    return pack_roots, manifests, resolved_paths


def running_comfy_matches(
    previous: Mapping[str, Any],
    *,
    lock_sha256: str,
    cors_origin: str | None,
    port: int,
    refresh: bool,
) -> bool:
    return (
        not refresh
        and previous.get("lockSha256") == lock_sha256
        and previous.get("corsOrigin") == cors_origin
        and pid_alive(previous.get("comfyPid"))
        and http_ready(port)
    )


def existing_cloudflare_endpoint(
    previous: Mapping[str, Any],
) -> tuple[int | None, str | None]:
    tunnel_pid = previous.get("tunnelPid")
    cloudflare_url = previous.get("cloudflareUrl")
    if (
        isinstance(tunnel_pid, int)
        and pid_alive(tunnel_pid)
        and isinstance(cloudflare_url, str)
    ):
        return tunnel_pid, cloudflare_url
    return None, None


def require_access_endpoint(
    *,
    colab_proxy: bool,
    cloudflare_url: str | None,
    tunnel_error: str | None,
) -> None:
    if colab_proxy or cloudflare_url is not None:
        return
    detail = f" ({tunnel_error})" if tunnel_error else ""
    raise RuntimeError(
        "Cloudflare fallback unavailable while Colab proxy mode is disabled"
        f"{detail}"
    )


def save_state(payload: Mapping[str, Any]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    temporary = STATE_FILE.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(STATE_FILE)


def emit_ready(payload: Mapping[str, Any]) -> None:
    print(READY_PREFIX + json.dumps(payload, separators=(",", ":")), flush=True)


def validate_legacy_full_lock(
    lock: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    packs = lock.get("packs")
    if not isinstance(packs, list):
        raise RuntimeContractError("legacy-full lock.packs must be an array")
    validated: dict[str, Mapping[str, Any]] = {}
    for pack in packs:
        if not isinstance(pack, dict):
            raise RuntimeContractError("legacy-full lock pack entries must be objects")
        pack_id = safe_pack_id(pack.get("id"))
        if pack_id in validated:
            raise RuntimeContractError(f"duplicate pack in legacy-full lock: {pack_id}")
        validate_repository(
            pack.get("repository"),
            field=f"legacy-full pack {pack_id} repository",
        )
        validate_commit(
            pack.get("commit"),
            field=f"legacy-full pack {pack_id} commit",
        )
        manifest_sha256 = pack.get("manifest_sha256")
        if (
            not isinstance(manifest_sha256, str)
            or not _SHA256_RE.fullmatch(manifest_sha256)
        ):
            raise RuntimeContractError(
                f"legacy-full pack {pack_id} manifest digest is invalid"
            )
        validated[pack_id] = pack
    if set(validated) != LEGACY_FULL_PACK_IDS:
        expected = ", ".join(sorted(LEGACY_FULL_PACK_IDS))
        actual = ", ".join(sorted(validated)) or "none"
        raise RuntimeContractError(
            "legacy-full requires exactly the node-bearing daughter packs "
            f"{expected}; resolved {actual}"
        )
    return tuple(validated[pack_id] for pack_id in sorted(validated))


def execute_legacy_full(
    config: Mapping[str, Any],
    lock: Mapping[str, Any],
) -> None:
    """Run the preserved full installer against authenticated daughter refs."""

    packs = validate_legacy_full_lock(lock)
    core = lock.get("core")
    if not isinstance(core, dict):
        raise RuntimeContractError("legacy-full lock.core must be an object")
    repository = validate_repository(
        core.get("repository"),
        field="legacy-full core repository",
    )
    commit = validate_commit(
        core.get("commit"),
        field="legacy-full core commit",
    )

    from . import remote_bootstrap

    remote_bootstrap.CONFIG = {
        "repository_url": repository,
        "repository_ref": commit,
        "port": int(config["port"]),
        "refresh": bool(config["refresh"]),
        "colab_proxy": bool(config["colab_proxy"]),
        "runtime_mode": "legacy-full",
        "lock_sha256": str(config["lock_sha256"]),
        "accepted_licenses": list(config.get("accepted_licenses", [])),
        "packs": [
            {
                "id": pack["id"],
                "repository": pack["repository"],
                "commit": pack["commit"],
                "manifest_sha256": pack["manifest_sha256"],
            }
            for pack in packs
        ],
    }
    remote_bootstrap.main()


def execute(config: Mapping[str, Any], lock: Mapping[str, Any]) -> None:
    runtime_mode = config.get("runtime_mode", "generic")
    if runtime_mode == "legacy-full":
        execute_legacy_full(config, lock)
        return
    if runtime_mode != "generic":
        raise RuntimeContractError(f"unsupported runtime mode: {runtime_mode!r}")
    validate_runtime_support(lock)
    accepted_licenses = set(config["accepted_licenses"])
    validate_pack_license_gates(
        lock,
        accepted_licenses=accepted_licenses,
    )
    port = int(config["port"])
    refresh = bool(config["refresh"])
    colab_proxy = bool(config["colab_proxy"])
    lock_sha256 = str(config["lock_sha256"])
    cors_origin = colab_cors_origin(os.environ) if colab_proxy else None
    reserved_proxy_url = None
    if colab_proxy and "COMFYCOLAB_PROXY_URL" in os.environ:
        reserved_proxy_url = validate_colab_proxy_url(
            os.environ["COMFYCOLAB_PROXY_URL"]
        )
    comfy_command = comfy_launch_command(
        port,
        colab_proxy=colab_proxy,
        inherited_environment=os.environ,
    )
    baseline_pip_conflicts = load_or_create_pip_baseline()
    if baseline_pip_conflicts:
        print(
            "[comfycolab] Detected pre-existing Colab package conflicts; "
            "only newly introduced conflicts will be fatal.",
            flush=True,
        )
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    previous = load_state()
    if running_comfy_matches(
        previous,
        lock_sha256=lock_sha256,
        cors_origin=cors_origin,
        port=port,
        refresh=refresh,
    ):
        proxy_url = (
            reserved_proxy_url or request_colab_proxy_url(port)
            if colab_proxy
            else None
        )
        tunnel_pid, cloudflare_url = existing_cloudflare_endpoint(previous)
        new_tunnel: subprocess.Popen[bytes] | None = None
        tunnel_error: str | None = None
        if cloudflare_url is None:
            stop_managed_process(previous.get("tunnelPid"))
            new_tunnel, cloudflare_url, tunnel_error = start_cloudflare_tunnel(port)
            tunnel_pid = new_tunnel.pid if new_tunnel is not None else None
        require_access_endpoint(
            colab_proxy=colab_proxy,
            cloudflare_url=cloudflare_url,
            tunnel_error=tunnel_error,
        )
        payload = dict(previous)
        payload.update(
            {
                "status": "ready",
                "comfyUrl": proxy_url or cloudflare_url,
                "cloudflareUrl": cloudflare_url,
                "colabProxyUrl": proxy_url,
                "corsOrigin": cors_origin,
                "tunnelPid": tunnel_pid,
                "tunnelError": tunnel_error,
            }
        )
        try:
            save_state(payload)
            emit_ready(payload)
        except BaseException:
            if new_tunnel is not None:
                stop_started_process(new_tunnel)
            raise
        return

    installation_reused = (
        not refresh and previous.get("lockSha256") == lock_sha256
    )
    if installation_reused:
        stop_managed_process(previous.get("comfyPid"))
    else:
        stop_managed_process(previous.get("tunnelPid"))
        stop_managed_process(previous.get("comfyPid"))
    if http_ready(port):
        raise RuntimeError(f"port {port} is occupied by an unmanaged process")

    comfyui = lock["comfyui"]
    assert isinstance(comfyui, dict)
    if installation_reused:
        try:
            pack_roots, manifests, resolved_paths = load_existing_installation(lock)
        except (OSError, subprocess.CalledProcessError, RuntimeContractError) as error:
            print(
                f"[comfycolab] Existing installation is not reusable ({error}); "
                "reinstalling the active lock.",
                flush=True,
            )
            stop_managed_process(previous.get("tunnelPid"))
            installation_reused = False

    if not installation_reused:
        STATE_FILE.unlink(missing_ok=True)
        reset_installation_roots()
        pack_roots, manifests = clone_packs(lock)
        validate_manifest_compatibility(
            manifests,
            comfyui_commit=str(comfyui["commit"]),
        )
        clone_at_commit(
            str(comfyui["repository"]),
            COMFY_DIR,
            str(comfyui["commit"]),
        )
        install_core_requirements()
        resolved_paths = install_dependencies(
            lock,
            pack_roots=pack_roots,
            accepted_licenses=accepted_licenses,
        )
        resolved_paths["comfyui"] = str(COMFY_DIR)
        environment_pythons = install_environments(
            lock,
            accepted_licenses=accepted_licenses,
        )
        resolved_paths.update(
            {
                f"environment:{name}": path
                for name, path in environment_pythons.items()
            }
        )
    else:
        validate_manifest_compatibility(
            manifests,
            comfyui_commit=str(comfyui["commit"]),
        )

    applied_patches = apply_patches(
        lock,
        pack_roots=pack_roots,
        resolved_paths=resolved_paths,
    )
    run_post_clone_probes(
        manifests,
        pack_roots=pack_roots,
        resolved_paths=resolved_paths,
    )
    link_node_roots(manifests, pack_roots=pack_roots)
    runtime_environment, hook_readiness = run_prestart_hooks(
        lock,
        lock_sha256=lock_sha256,
        manifests=manifests,
        pack_roots=pack_roots,
        resolved_paths=resolved_paths,
    )
    for name, value in resolved_runtime_environment(lock).items():
        previous_value = runtime_environment.get(name)
        if previous_value is not None and previous_value != value:
            raise RuntimeContractError(f"runtime environment conflict for {name}")
        runtime_environment[name] = value
    current_pip_conflicts = pip_check_conflicts()
    reject_new_pip_conflicts(
        baseline_pip_conflicts,
        current_pip_conflicts,
    )
    if current_pip_conflicts:
        print(
            f"[comfycolab] Ignoring {len(current_pip_conflicts)} unchanged "
            "pre-existing package conflict(s).",
            flush=True,
        )

    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    environment.update(runtime_environment)
    comfy: subprocess.Popen[bytes] | None = None
    new_tunnel: subprocess.Popen[bytes] | None = None
    ready = False
    try:
        with COMFY_LOG.open("wb") as comfy_log:
            comfy = subprocess.Popen(
                comfy_command,
                cwd=COMFY_DIR,
                stdout=comfy_log,
                stderr=subprocess.STDOUT,
                env=environment,
                start_new_session=True,
            )
        wait_for_comfy(port, comfy)
        post_start_readiness = validate_post_start_nodes(
            port,
            manifests,
            lock=lock,
            lock_sha256=lock_sha256,
            pack_roots=pack_roots,
            resolved_paths=resolved_paths,
        )
        for pack_id, result in post_start_readiness.items():
            hook_readiness.setdefault(pack_id, {})["post_start"] = result
        proxy_url = None
        if colab_proxy:
            proxy_url = reserved_proxy_url or request_colab_proxy_url(port)
        tunnel_pid, cloudflare_url = (
            existing_cloudflare_endpoint(previous)
            if installation_reused
            else (None, None)
        )
        tunnel_error: str | None = None
        if cloudflare_url is None:
            stop_managed_process(previous.get("tunnelPid"))
            new_tunnel, cloudflare_url, tunnel_error = start_cloudflare_tunnel(
                port
            )
            tunnel_pid = new_tunnel.pid if new_tunnel is not None else None
        require_access_endpoint(
            colab_proxy=colab_proxy,
            cloudflare_url=cloudflare_url,
            tunnel_error=tunnel_error,
        )
        pack_state: dict[str, Any] = {}
        lock_packs = {
            str(pack["id"]): pack
            for pack in lock.get("packs", [])
            if isinstance(pack, dict)
        }
        for pack_id in sorted(manifests):
            pack = lock_packs[pack_id]
            readiness_declaration = manifests[pack_id].get("readiness")
            pack_state[pack_id] = {
                "version": pack.get("version"),
                "commit": pack.get("commit"),
                "status": "ready",
                "hooks": hook_readiness.get(pack_id, {}),
                **(
                    {
                        "readinessDeclaration": {
                            "namespace": readiness_declaration.get("namespace"),
                            "fields": readiness_declaration.get("fields", []),
                            "status": "reserved-metadata",
                        }
                    }
                    if isinstance(readiness_declaration, dict)
                    else {}
                ),
            }
        core = lock.get("core", {})
        payload: dict[str, Any] = {
            "schema": 1,
            "status": "ready",
            "comfyUrl": proxy_url or cloudflare_url,
            "cloudflareUrl": cloudflare_url,
            "colabProxyUrl": proxy_url,
            "corsOrigin": cors_origin,
            "comfyPid": comfy.pid,
            "tunnelPid": tunnel_pid,
            "tunnelError": tunnel_error,
            "port": port,
            "storage": "temporary",
            "lockSha256": lock_sha256,
            "core": {
                "version": core.get("version") if isinstance(core, dict) else None,
                "commit": core.get("commit") if isinstance(core, dict) else None,
                "comfyuiCommit": git_commit(COMFY_DIR),
            },
            "packs": pack_state,
            "appliedPatches": applied_patches,
        }
        save_state(payload)
        ready = True
        emit_ready(payload)
    finally:
        if not ready:
            if new_tunnel is not None:
                stop_started_process(new_tunnel)
            if comfy is not None:
                stop_started_process(comfy)
            if not installation_reused:
                STATE_FILE.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="comfycolab-stage1")
    parser.add_argument("--config", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    args = build_parser().parse_args(argv)
    config = load_stage1_config(args.config)
    lock = load_lock(Path(config["lock_path"]), str(config["lock_sha256"]))
    execute(config, lock)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
