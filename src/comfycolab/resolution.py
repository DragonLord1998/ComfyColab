from __future__ import annotations

import hashlib
import json
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .config import CoreStage0ConfigV1
from .packs.io import (
    PackProfileV1,
    PackRegistryV1,
    load_pack_manifest,
    load_pack_ref,
    load_profile,
    load_registry,
)
from .packs.lock import ComfyColabLockV1
from .packs.resolver import resolve_lock
from .packs.schema import PackManifestV1, PackRefV1
from .repositories import checkout_repository, temporary_checkout
from .stage0 import render_stage0


STAGE1_ENTRYPOINT = "src/comfycolab/runtime.py"
_VERSION_RE = re.compile(r'^__version__\s*=\s*"([^"]+)"\s*$', re.MULTILINE)


class ResolutionError(RuntimeError):
    """Raised when a launch selection cannot become an immutable lock."""


@dataclass(frozen=True)
class PreparedLaunch:
    source: str
    lock: ComfyColabLockV1
    config: CoreStage0ConfigV1
    profile_id: str | None


def _load_engine(path: Path) -> tuple[str, str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ResolutionError(f"unable to load engine registry: {path}") from error
    if not isinstance(payload, dict) or set(payload) != {"schema", "comfyui"}:
        raise ResolutionError("engine registry fields are invalid")
    if payload["schema"] != 1 or not isinstance(payload["comfyui"], dict):
        raise ResolutionError("engine registry schema is invalid")
    comfyui = payload["comfyui"]
    if set(comfyui) != {"repository", "commit"}:
        raise ResolutionError("engine registry ComfyUI fields are invalid")
    repository = comfyui["repository"]
    commit = comfyui["commit"]
    if not isinstance(repository, str) or not repository.startswith("https://"):
        raise ResolutionError("engine registry repository is invalid")
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ResolutionError("engine registry commit is invalid")
    return repository, commit


def _core_version(core_checkout: Path) -> str:
    path = core_checkout / "src" / "comfycolab" / "__init__.py"
    try:
        source = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ResolutionError(f"core version source is unavailable: {path}") from error
    match = _VERSION_RE.search(source)
    if match is None:
        raise ResolutionError("core checkout does not declare __version__")
    return match.group(1)


def _profile_path(core_checkout: Path, profile: str) -> Path:
    candidate = Path(profile).expanduser()
    if candidate.is_file():
        return candidate
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,63}", profile):
        raise ResolutionError(f"invalid profile name: {profile!r}")
    path = core_checkout / "profiles" / f"{profile}.json"
    if not path.is_file():
        raise ResolutionError(f"unknown profile: {profile}")
    return path


def select_pack_refs(
    core_checkout: Path,
    *,
    pack_aliases: Sequence[str] = (),
    profile: str | None = None,
    pack_ref_files: Sequence[Path] = (),
) -> tuple[tuple[PackRefV1, ...], str | None]:
    registry = load_registry(core_checkout / "registry" / "official-packs.json")
    selected: list[PackRefV1] = []
    profile_id: str | None = None
    if profile is not None:
        parsed_profile = load_profile(
            _profile_path(core_checkout, profile),
            registry=registry,
        )
        selected.extend(parsed_profile.packs)
        profile_id = parsed_profile.id
    for alias in pack_aliases:
        try:
            selected.append(registry.packs[alias])
        except KeyError as error:
            raise ResolutionError(
                f"pack alias {alias!r} is not in the authenticated official registry"
            ) from error
    selected.extend(load_pack_ref(path) for path in pack_ref_files)
    by_id: dict[str, PackRefV1] = {}
    for pack_ref in selected:
        previous = by_id.get(pack_ref.id)
        if previous is not None and previous != pack_ref:
            raise ResolutionError(f"conflicting selections for pack {pack_ref.id!r}")
        by_id[pack_ref.id] = pack_ref
    return tuple(by_id[key] for key in sorted(by_id)), profile_id


def _materialize_manifests(
    pack_refs: Iterable[PackRefV1],
    workspace: Path,
) -> list[tuple[PackRefV1, PackManifestV1]]:
    resolved: list[tuple[PackRefV1, PackManifestV1]] = []
    for pack_ref in pack_refs:
        destination = workspace / "packs" / pack_ref.id
        commit = checkout_repository(
            pack_ref.repository,
            pack_ref.ref,
            destination,
        )
        if commit != pack_ref.ref:
            raise ResolutionError(
                f"pack {pack_ref.id!r} resolved {commit}, expected {pack_ref.ref}"
            )
        manifest = load_pack_manifest(
            destination / "comfycolab-pack.json",
            expected_sha256=pack_ref.manifest_sha256,
        )
        if manifest.id != pack_ref.id:
            raise ResolutionError(
                f"pack reference {pack_ref.id!r} loaded manifest {manifest.id!r}"
            )
        resolved.append((pack_ref, manifest))
    return resolved


def resolve_from_checkout(
    core_checkout: Path,
    *,
    core_repository: str,
    core_commit: str,
    pack_aliases: Sequence[str] = (),
    profile: str | None = "core",
    pack_ref_files: Sequence[Path] = (),
    workspace: Path,
) -> tuple[ComfyColabLockV1, str | None]:
    pack_refs, profile_id = select_pack_refs(
        core_checkout,
        pack_aliases=pack_aliases,
        profile=profile,
        pack_ref_files=pack_ref_files,
    )
    manifests = _materialize_manifests(pack_refs, workspace)
    comfyui_repository, comfyui_commit = _load_engine(
        core_checkout / "registry" / "engine.json"
    )
    lock = resolve_lock(
        core_repository=core_repository,
        core_commit=core_commit,
        core_version=_core_version(core_checkout),
        comfyui_repository=comfyui_repository,
        comfyui_commit=comfyui_commit,
        packs=manifests,
    )
    return lock, profile_id


def prepare_launch(
    *,
    core_repository: str,
    core_ref: str,
    pack_aliases: Sequence[str] = (),
    profile: str | None = "core",
    pack_ref_files: Sequence[Path] = (),
    port: int = 8188,
    refresh: bool = False,
    colab_proxy: bool = False,
    runtime_mode: str = "generic",
    accepted_licenses: Sequence[str] = (),
) -> PreparedLaunch:
    with temporary_checkout(core_repository, core_ref) as (core_checkout, core_commit):
        with tempfile.TemporaryDirectory(prefix="comfycolab-resolution-") as directory:
            lock, profile_id = resolve_from_checkout(
                core_checkout,
                core_repository=core_repository,
                core_commit=core_commit,
                pack_aliases=pack_aliases,
                profile=profile,
                pack_ref_files=pack_ref_files,
                workspace=Path(directory),
            )
        stage1_path = core_checkout / STAGE1_ENTRYPOINT
        if not stage1_path.is_file():
            raise ResolutionError(
                f"authenticated core checkout is missing {STAGE1_ENTRYPOINT}"
            )
        config = CoreStage0ConfigV1.create(
            core_repository=core_repository,
            core_commit=core_commit,
            stage1_entrypoint=STAGE1_ENTRYPOINT,
            stage1_sha256=hashlib.sha256(stage1_path.read_bytes()).hexdigest(),
            lock_bytes=lock.canonical_bytes(),
            port=port,
            refresh=refresh,
            colab_proxy=colab_proxy,
            runtime_mode=runtime_mode,
            accepted_licenses=accepted_licenses,
        )
        return PreparedLaunch(
            source=render_stage0(config),
            lock=lock,
            config=config,
            profile_id=profile_id,
        )


def prepare_launch_from_lock(
    lock: ComfyColabLockV1,
    *,
    port: int = 8188,
    refresh: bool = True,
    colab_proxy: bool = False,
    runtime_mode: str = "generic",
    accepted_licenses: Sequence[str] = (),
) -> PreparedLaunch:
    payload = lock.to_dict()
    core = payload.get("core")
    if not isinstance(core, dict):
        raise ResolutionError("saved lock is missing its core identity")
    repository = core.get("repository")
    commit = core.get("commit")
    if not isinstance(repository, str) or not isinstance(commit, str):
        raise ResolutionError("saved lock core identity is invalid")
    with temporary_checkout(repository, commit) as (core_checkout, actual_commit):
        if actual_commit != commit:
            raise ResolutionError(
                f"saved lock core commit resolved {actual_commit}, expected {commit}"
            )
        stage1_path = core_checkout / STAGE1_ENTRYPOINT
        if not stage1_path.is_file():
            raise ResolutionError(
                f"authenticated core checkout is missing {STAGE1_ENTRYPOINT}"
            )
        config = CoreStage0ConfigV1.create(
            core_repository=repository,
            core_commit=commit,
            stage1_entrypoint=STAGE1_ENTRYPOINT,
            stage1_sha256=hashlib.sha256(stage1_path.read_bytes()).hexdigest(),
            lock_bytes=lock.canonical_bytes(),
            port=port,
            refresh=refresh,
            colab_proxy=colab_proxy,
            runtime_mode=runtime_mode,
            accepted_licenses=accepted_licenses,
        )
    return PreparedLaunch(
        source=render_stage0(config),
        lock=lock,
        config=config,
        profile_id=None,
    )
