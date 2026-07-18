"""Immutable, deterministic lock representation for resolved pack selections."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping
from urllib.parse import urlsplit

from .canonical import canonical_json_bytes
from .errors import PackSchemaError
from .schema import (
    SCHEMA_VERSION,
    EnvironmentV1,
    GitDependencyV1,
    NodeRootV1,
    PatchV1,
    PythonRequirementV1,
    SystemDependencyV1,
    dependency_from_dict,
)

_PACK_ID = re.compile(r"^[a-z0-9]+(?:[._-][a-z0-9]+)*$")
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_LOGICAL_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9._-]*$")


def _object(value: object, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PackSchemaError(f"{location} must be an object.")
    if not all(isinstance(key, str) for key in value):
        raise PackSchemaError(f"{location} contains a non-string key.")
    return value


def _exact_keys(
    value: Mapping[str, Any],
    *,
    required: set[str],
    optional: set[str] = frozenset(),
    location: str,
) -> None:
    missing = sorted(required - value.keys())
    unknown = sorted(value.keys() - required - optional)
    if missing:
        raise PackSchemaError(f"{location} is missing: {', '.join(missing)}.")
    if unknown:
        raise PackSchemaError(f"{location} has unknown fields: {', '.join(unknown)}.")


def _array(value: object, location: str) -> list[Any]:
    if not isinstance(value, list):
        raise PackSchemaError(f"{location} must be an array.")
    return value


def _requested_by(value: object, location: str) -> list[str]:
    values = _array(value, location)
    result: list[str] = []
    for index, item in enumerate(values):
        if not isinstance(item, str) or not _PACK_ID.fullmatch(item):
            raise PackSchemaError(f"{location}[{index}] must be a pack ID.")
        result.append(item)
    if not result or len(result) != len(set(result)):
        raise PackSchemaError(f"{location} must be non-empty and unique.")
    return sorted(result)


def _identity(
    value: object,
    location: str,
    *,
    version: bool,
) -> dict[str, str]:
    source = _object(value, location)
    required = {"repository", "commit"}
    if version:
        required.add("version")
    _exact_keys(source, required=required, location=location)
    repository = source["repository"]
    commit = source["commit"]
    if not isinstance(repository, str):
        raise PackSchemaError(f"{location}.repository must be an HTTPS Git URL.")
    parsed = urlsplit(repository)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or not parsed.path.endswith(".git")
    ):
        raise PackSchemaError(
            f"{location}.repository must be an HTTPS .git URL without credentials "
            "or query data."
        )
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
    ):
        raise PackSchemaError(f"{location}.commit must be an immutable Git commit.")
    result = {"repository": repository, "commit": commit}
    if version:
        raw_version = source["version"]
        if not isinstance(raw_version, str) or not raw_version:
            raise PackSchemaError(f"{location}.version must be a non-empty string.")
        result["version"] = raw_version
    return result


def _validate_payload(value: object) -> dict[str, object]:
    source = _object(value, "ComfyColab lock")
    _exact_keys(
        source,
        required={
            "schema",
            "core",
            "comfyui",
            "packs",
            "dependencies",
            "patches",
            "environments",
            "runtime_env",
        },
        location="ComfyColab lock",
    )
    if source["schema"] != SCHEMA_VERSION:
        raise PackSchemaError(f"ComfyColab lock schema must be {SCHEMA_VERSION}.")

    packs: list[dict[str, object]] = []
    seen_packs: set[str] = set()
    for index, item in enumerate(_array(source["packs"], "ComfyColab lock.packs")):
        location = f"ComfyColab lock.packs[{index}]"
        pack = _object(item, location)
        _exact_keys(
            pack,
            required={
                "id",
                "version",
                "repository",
                "commit",
                "manifest_sha256",
                "node_roots",
            },
            optional={"license_gate"},
            location=location,
        )
        identifier = pack["id"]
        if not isinstance(identifier, str) or not _PACK_ID.fullmatch(identifier):
            raise PackSchemaError(f"{location}.id must be a pack ID.")
        if identifier in seen_packs:
            raise PackSchemaError(f"Duplicate pack in lock: {identifier}.")
        seen_packs.add(identifier)
        manifest_digest = pack["manifest_sha256"]
        if (
            not isinstance(manifest_digest, str)
            or len(manifest_digest) != 64
            or any(character not in "0123456789abcdef" for character in manifest_digest)
        ):
            raise PackSchemaError(f"{location}.manifest_sha256 is invalid.")
        identity = _identity(
            {
                "version": pack["version"],
                "repository": pack["repository"],
                "commit": pack["commit"],
            },
            location,
            version=True,
        )
        node_roots = [
            NodeRootV1.from_dict(root, f"{location}.node_roots[{root_index}]").to_dict()
            for root_index, root in enumerate(_array(pack["node_roots"], f"{location}.node_roots"))
        ]
        license_gate = pack.get("license_gate")
        if license_gate is not None and (
            not isinstance(license_gate, str)
            or not _LOGICAL_NAME.fullmatch(license_gate)
        ):
            raise PackSchemaError(f"{location}.license_gate must be a logical name.")
        packs.append(
            {
                "id": identifier,
                **identity,
                "manifest_sha256": manifest_digest,
                "node_roots": sorted(node_roots, key=lambda entry: entry["target"]),
                **(
                    {"license_gate": license_gate}
                    if license_gate is not None
                    else {}
                ),
            }
        )

    dependencies: list[dict[str, object]] = []
    for index, item in enumerate(
        _array(source["dependencies"], "ComfyColab lock.dependencies")
    ):
        location = f"ComfyColab lock.dependencies[{index}]"
        dependency = _object(item, location)
        requested = _requested_by(dependency.get("requested_by"), f"{location}.requested_by")
        raw = {key: item for key, item in dependency.items() if key != "requested_by"}
        parsed = dependency_from_dict(raw, location)
        if (
            isinstance(parsed, GitDependencyV1)
            and parsed.requirements_source == "pack"
            and len(requested) != 1
        ):
            raise PackSchemaError(
                f"{location} pack-sourced requirements require exactly one owner."
            )
        dependencies.append({**parsed.to_dict(), "requested_by": requested})

    patches: list[dict[str, object]] = []
    for index, item in enumerate(_array(source["patches"], "ComfyColab lock.patches")):
        location = f"ComfyColab lock.patches[{index}]"
        patch = _object(item, location)
        requested = _requested_by(patch.get("requested_by"), f"{location}.requested_by")
        raw = {key: item for key, item in patch.items() if key != "requested_by"}
        parsed = PatchV1.from_dict(raw, location)
        if parsed.specification_sha256 is None:
            raise PackSchemaError(f"{location}.specification_sha256 is required in a lock.")
        if not parsed.files:
            raise PackSchemaError(f"{location}.files is required in a lock.")
        patches.append({**parsed.to_dict(), "requested_by": requested})

    environments: list[dict[str, object]] = []
    seen_environment_ids: set[str] = set()
    for index, item in enumerate(
        _array(source["environments"], "ComfyColab lock.environments")
    ):
        location = f"ComfyColab lock.environments[{index}]"
        environment = _object(item, location)
        _exact_keys(
            environment,
            required={
                "id",
                "kind",
                "scope",
                "python_requirements",
                "system_dependencies",
                "requested_by",
            },
            optional={"owner", "python", "cache_profile", "license_gate"},
            location=location,
        )
        requested = _requested_by(
            environment["requested_by"], f"{location}.requested_by"
        )
        requirements: list[dict[str, object]] = []
        for req_index, raw_item in enumerate(
            _array(environment["python_requirements"], f"{location}.python_requirements")
        ):
            req_location = f"{location}.python_requirements[{req_index}]"
            requirement = _object(raw_item, req_location)
            req_requested = _requested_by(
                requirement.get("requested_by"), f"{req_location}.requested_by"
            )
            parsed = PythonRequirementV1.from_dict(
                {
                    key: value
                    for key, value in requirement.items()
                    if key != "requested_by"
                },
                req_location,
            )
            requirements.append({**parsed.to_dict(), "requested_by": req_requested})
        systems: list[dict[str, object]] = []
        for dep_index, raw_item in enumerate(
            _array(environment["system_dependencies"], f"{location}.system_dependencies")
        ):
            dep_location = f"{location}.system_dependencies[{dep_index}]"
            dependency = _object(raw_item, dep_location)
            dep_requested = _requested_by(
                dependency.get("requested_by"), f"{dep_location}.requested_by"
            )
            parsed = SystemDependencyV1.from_dict(
                {
                    key: value
                    for key, value in dependency.items()
                    if key != "requested_by"
                },
                dep_location,
            )
            systems.append({**parsed.to_dict(), "requested_by": dep_requested})
        base = EnvironmentV1.from_dict(
            {
                key: value
                for key, value in environment.items()
                if key
                not in {
                    "owner",
                    "requested_by",
                    "python_requirements",
                    "system_dependencies",
                }
            }
            | {
                "python_requirements": [
                    {key: value for key, value in item.items() if key != "requested_by"}
                    for item in requirements
                ],
                "system_dependencies": [
                    {key: value for key, value in item.items() if key != "requested_by"}
                    for item in systems
                ],
            },
            location,
        )
        owner = environment.get("owner")
        if owner is not None and (not isinstance(owner, str) or not owner):
            raise PackSchemaError(f"{location}.owner must be a pack ID.")
        normalized = base.to_dict()
        if base.id in seen_environment_ids:
            raise PackSchemaError(
                f"Duplicate ComfyColab lock environment ID: {base.id}."
            )
        seen_environment_ids.add(base.id)
        if owner is not None:
            normalized["owner"] = owner
        normalized["python_requirements"] = sorted(
            requirements, key=lambda entry: str(entry["name"])
        )
        normalized["system_dependencies"] = sorted(
            systems,
            key=lambda entry: (
                str(entry["manager"]),
                str(entry["name"]),
                str(entry["scope"]),
            ),
        )
        normalized["requested_by"] = requested
        environments.append(normalized)

    runtime_env: list[dict[str, object]] = []
    seen_env_names: set[str] = set()
    for index, item in enumerate(
        _array(source["runtime_env"], "ComfyColab lock.runtime_env")
    ):
        location = f"ComfyColab lock.runtime_env[{index}]"
        variable = _object(item, location)
        _exact_keys(
            variable,
            required={"name", "value", "requested_by"},
            location=location,
        )
        name = variable["name"]
        value_text = variable["value"]
        if not isinstance(name, str) or not _ENV_NAME.fullmatch(name):
            raise PackSchemaError(f"{location}.name must be an environment name.")
        if name in seen_env_names:
            raise PackSchemaError(f"Duplicate runtime environment variable: {name}.")
        seen_env_names.add(name)
        if not isinstance(value_text, str):
            raise PackSchemaError(f"{location}.value must be a string.")
        runtime_env.append(
            {
                "name": name,
                "value": value_text,
                "requested_by": _requested_by(
                    variable["requested_by"], f"{location}.requested_by"
                ),
            }
        )

    return {
        "schema": SCHEMA_VERSION,
        "core": _identity(source["core"], "ComfyColab lock.core", version=True),
        "comfyui": _identity(
            source["comfyui"], "ComfyColab lock.comfyui", version=False
        ),
        "packs": sorted(packs, key=lambda entry: str(entry["id"])),
        "dependencies": sorted(
            dependencies,
            key=lambda entry: (
                str(entry["scope"]),
                str(entry["destination"]),
                str(entry["id"]),
            ),
        ),
        "patches": sorted(
            patches,
            key=lambda entry: (
                str(entry["target"]),
                int(entry["order"]),
                str(entry["id"]),
            ),
        ),
        "environments": sorted(
            environments,
            key=lambda entry: (
                str(entry["kind"]),
                str(entry.get("owner", "")),
                str(entry["id"]),
            ),
        ),
        "runtime_env": sorted(runtime_env, key=lambda entry: str(entry["name"])),
    }


@dataclass(frozen=True)
class ComfyColabLockV1:
    """A schema-validated lock stored internally as canonical JSON bytes."""

    _canonical: bytes

    @classmethod
    def from_dict(cls, value: object) -> "ComfyColabLockV1":
        return cls(canonical_json_bytes(_validate_payload(value)))

    @classmethod
    def from_bytes(cls, value: bytes) -> "ComfyColabLockV1":
        def reject_constant(constant: str) -> None:
            raise PackSchemaError(
                f"ComfyColab lock contains a non-finite number: {constant}."
            )

        def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, item in pairs:
                if key in result:
                    raise PackSchemaError(
                        f"ComfyColab lock contains a duplicate key: {key!r}."
                    )
                result[key] = item
            return result

        try:
            payload = json.loads(
                value.decode("utf-8"),
                object_pairs_hook=unique_object,
                parse_constant=reject_constant,
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise PackSchemaError("ComfyColab lock bytes are not valid UTF-8 JSON.") from error
        return cls.from_dict(payload)

    def to_dict(self) -> dict[str, object]:
        value = json.loads(self._canonical)
        assert isinstance(value, dict)
        return value

    def canonical_bytes(self) -> bytes:
        return bytes(self._canonical)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self._canonical).hexdigest()

    @property
    def packs(self) -> tuple[dict[str, object], ...]:
        value = self.to_dict()["packs"]
        assert isinstance(value, list)
        return tuple(value)
