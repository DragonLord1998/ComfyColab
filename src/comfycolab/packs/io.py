"""Safe filesystem loading for pack manifests, references, registries, and profiles."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

from .errors import PackIntegrityError, PackSchemaError
from .schema import (
    SCHEMA_VERSION,
    PackManifestV1,
    PackRefV1,
    PatchFileV1,
    declared_pack_paths,
    resolve_declared_path,
)


MAX_DOCUMENT_BYTES = 2 * 1024 * 1024


def _reject_constant(value: str) -> None:
    raise PackSchemaError(f"JSON contains a non-finite number: {value}.")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PackSchemaError(f"JSON contains a duplicate object key: {key!r}.")
        result[key] = value
    return result


def _document_bytes(path: Path, *, maximum_bytes: int = MAX_DOCUMENT_BYTES) -> bytes:
    if path.is_symlink():
        raise PackSchemaError(f"Refusing a symlinked pack document: {path}.")
    try:
        stat = path.stat()
    except OSError as error:
        raise PackSchemaError(f"Unable to stat pack document {path}: {error}.") from error
    if not path.is_file():
        raise PackSchemaError(f"Pack document is not a regular file: {path}.")
    if stat.st_size > maximum_bytes:
        raise PackSchemaError(
            f"Pack document exceeds the {maximum_bytes}-byte limit: {path}."
        )
    try:
        return path.read_bytes()
    except OSError as error:
        raise PackSchemaError(f"Unable to read pack document {path}: {error}.") from error


def safe_load_json(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    maximum_bytes: int = MAX_DOCUMENT_BYTES,
) -> tuple[object, str]:
    """Load strict UTF-8 JSON and return the value plus raw-file SHA-256."""

    document_path = Path(path)
    raw = _document_bytes(document_path, maximum_bytes=maximum_bytes)
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise PackIntegrityError(
            f"SHA-256 mismatch for {document_path}: expected {expected_sha256}, got {digest}."
        )
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise PackSchemaError(f"Pack document is not valid UTF-8: {document_path}.") from error
    try:
        value = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except json.JSONDecodeError as error:
        raise PackSchemaError(
            f"Pack document is not valid JSON at line {error.lineno}, column {error.colno}: "
            f"{document_path}."
        ) from error
    return value, digest


def load_pack_ref(path: str | Path) -> PackRefV1:
    value, _ = safe_load_json(path)
    return PackRefV1.from_dict(value, f"pack ref {Path(path)}")


def load_lock(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
):
    """Load and schema-validate an immutable ComfyColab lock."""

    from .lock import ComfyColabLockV1

    value, _ = safe_load_json(path, expected_sha256=expected_sha256)
    return ComfyColabLockV1.from_dict(value)


def _hydrate_patch_metadata(
    manifest: PackManifestV1,
    root: Path,
) -> PackManifestV1:
    hydrated = []
    for patch in manifest.patches:
        specification_path = resolve_declared_path(root, patch.specification)
        value, digest = safe_load_json(
            specification_path,
            expected_sha256=patch.specification_sha256,
        )
        if not isinstance(value, Mapping):
            raise PackSchemaError(
                f"Patch specification must be an object: {specification_path}."
            )
        if value.get("schema") != SCHEMA_VERSION:
            raise PackSchemaError(
                f"Patch specification schema is unsupported: {specification_path}."
            )
        if value.get("patch_id") != patch.id:
            raise PackSchemaError(
                f"Patch ID mismatch in {specification_path}: expected {patch.id!r}."
            )
        if value.get("revision") != patch.target_ref:
            raise PackSchemaError(
                f"Patch revision mismatch in {specification_path}: "
                f"expected {patch.target_ref}."
            )
        raw_files = value.get("files")
        if not isinstance(raw_files, list) or not raw_files:
            raise PackSchemaError(
                f"Patch specification has no file metadata: {specification_path}."
            )
        files = tuple(
            PatchFileV1.from_dict(
                {
                    "path": item.get("path") if isinstance(item, Mapping) else None,
                    "before_sha256": (
                        item.get("before_sha256") if isinstance(item, Mapping) else None
                    ),
                    "after_sha256": (
                        item.get("after_sha256") if isinstance(item, Mapping) else None
                    ),
                },
                f"patch {patch.id}.files[{index}]",
            )
            for index, item in enumerate(raw_files)
        )
        if patch.files and patch.files != files:
            raise PackSchemaError(
                f"Inline patch file metadata disagrees with {specification_path}."
            )
        hydrated.append(
            replace(
                patch,
                specification_sha256=digest,
                files=files,
            )
        )
    return replace(manifest, patches=tuple(hydrated))


def _validate_declared_files(manifest: PackManifestV1, root: Path) -> None:
    for relative_path in declared_pack_paths(manifest):
        path = resolve_declared_path(root, relative_path)
        if not path.exists():
            raise PackSchemaError(
                f"Pack {manifest.id!r} declares a missing path: {relative_path}."
            )
    for node_root in manifest.node_roots:
        path = resolve_declared_path(root, node_root.source)
        if not path.is_dir():
            raise PackSchemaError(
                f"Pack {manifest.id!r} node root is not a directory: {node_root.source}."
            )
    for hook in manifest.hooks.values():
        path = resolve_declared_path(root, hook.path)
        if not path.is_file():
            raise PackSchemaError(
                f"Pack {manifest.id!r} hook is not a file: {hook.path}."
            )


def load_pack_manifest(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    validate_declared_paths: bool = True,
) -> PackManifestV1:
    """Load a pack manifest and validate all checkout-relative paths."""

    manifest_path = Path(path)
    value, digest = safe_load_json(manifest_path, expected_sha256=expected_sha256)
    manifest = PackManifestV1.from_dict(
        value,
        f"pack manifest {manifest_path}",
        source_sha256=digest,
    )
    root = manifest_path.resolve().parent
    if validate_declared_paths:
        _validate_declared_files(manifest, root)
    if manifest.patches:
        manifest = _hydrate_patch_metadata(manifest, root)
    return manifest


@dataclass(frozen=True)
class PackRegistryV1:
    packs: Mapping[str, PackRefV1]
    schema: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "packs": {
                alias: pack_ref.to_dict()
                for alias, pack_ref in sorted(self.packs.items())
            },
        }


@dataclass(frozen=True)
class PackProfileV1:
    id: str
    packs: tuple[PackRefV1, ...]
    schema: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "id": self.id,
            "packs": [pack_ref.to_dict() for pack_ref in self.packs],
        }


def _strict_keys(
    source: Mapping[str, object],
    *,
    required: set[str],
    location: str,
) -> None:
    missing = sorted(required - source.keys())
    unknown = sorted(source.keys() - required)
    if missing:
        raise PackSchemaError(f"{location} is missing: {', '.join(missing)}.")
    if unknown:
        raise PackSchemaError(f"{location} has unknown fields: {', '.join(unknown)}.")


def load_registry(path: str | Path) -> PackRegistryV1:
    value, _ = safe_load_json(path)
    if not isinstance(value, Mapping):
        raise PackSchemaError("Pack registry must be an object.")
    _strict_keys(value, required={"schema", "packs"}, location="pack registry")
    if value["schema"] != SCHEMA_VERSION:
        raise PackSchemaError(f"Pack registry schema must be {SCHEMA_VERSION}.")
    raw_packs = value["packs"]
    if not isinstance(raw_packs, Mapping):
        raise PackSchemaError("Pack registry packs must be an object.")
    packs: dict[str, PackRefV1] = {}
    for alias, raw_ref in raw_packs.items():
        if not isinstance(alias, str) or not re_full_alias(alias):
            raise PackSchemaError(f"Invalid pack registry alias: {alias!r}.")
        packs[alias] = PackRefV1.from_dict(raw_ref, f"pack registry.packs.{alias}")
    return PackRegistryV1(dict(sorted(packs.items())))


def re_full_alias(value: str) -> bool:
    return bool(value) and all(
        character.islower() or character.isdigit() or character in "._-"
        for character in value
    )


def load_profile(
    path: str | Path,
    *,
    registry: PackRegistryV1 | None = None,
) -> PackProfileV1:
    value, _ = safe_load_json(path)
    if not isinstance(value, Mapping):
        raise PackSchemaError("Pack profile must be an object.")
    _strict_keys(value, required={"schema", "id", "packs"}, location="pack profile")
    if value["schema"] != SCHEMA_VERSION:
        raise PackSchemaError(f"Pack profile schema must be {SCHEMA_VERSION}.")
    profile_id = value["id"]
    if not isinstance(profile_id, str) or not re_full_alias(profile_id):
        raise PackSchemaError("Pack profile id is invalid.")
    raw_packs = value["packs"]
    if not isinstance(raw_packs, list):
        raise PackSchemaError("Pack profile packs must be an array.")
    refs: list[PackRefV1] = []
    for index, item in enumerate(raw_packs):
        if isinstance(item, str):
            if registry is None:
                raise PackSchemaError(
                    f"Pack profile alias {item!r} requires an official registry."
                )
            try:
                pack_ref = registry.packs[item]
            except KeyError as error:
                raise PackSchemaError(f"Unknown pack profile alias: {item!r}.") from error
        else:
            pack_ref = PackRefV1.from_dict(item, f"pack profile.packs[{index}]")
        refs.append(pack_ref)
    seen: dict[str, PackRefV1] = {}
    for pack_ref in refs:
        previous = seen.get(pack_ref.id)
        if previous is not None and previous != pack_ref:
            raise PackSchemaError(
                f"Pack profile selects conflicting references for {pack_ref.id!r}."
            )
        seen[pack_ref.id] = pack_ref
    return PackProfileV1(profile_id, tuple(seen[key] for key in sorted(seen)))
