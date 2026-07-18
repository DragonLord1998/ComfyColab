"""Strict, standard-library-only models for the ComfyColab pack protocol."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping
from urllib.parse import urlsplit

from .canonical import canonical_sha256
from .errors import PackSchemaError


SCHEMA_VERSION = 1
CORE_MANIFEST_API = 1

_SLUG = re.compile(r"^[a-z0-9]+(?:[._-][a-z0-9]+)*$")
_LOGICAL_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9._-]*$")
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_PACKAGE_NAME = re.compile(r"^[A-Za-z0-9]+(?:[._-][A-Za-z0-9]+)*$")
_HF_REPOSITORY_PART = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SEMVER = re.compile(
    r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)"
    r"(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$"
)
_PYTHON_SPECIFIER = re.compile(
    r"^(==|!=|~=|<=|>=|<|>)"
    r"([0-9]+(?:\.[0-9]+)*(?:[A-Za-z0-9._+-]*)?|[0-9]+(?:\.[0-9]+)*\.\*)$"
)


def _mapping(value: object, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PackSchemaError(f"{location} must be an object.")
    if not all(isinstance(key, str) for key in value):
        raise PackSchemaError(f"{location} contains a non-string key.")
    return value


def _keys(
    value: Mapping[str, Any],
    location: str,
    *,
    required: Iterable[str],
    optional: Iterable[str] = (),
) -> None:
    required_set = set(required)
    allowed = required_set | set(optional)
    missing = sorted(required_set - value.keys())
    unknown = sorted(value.keys() - allowed)
    if missing:
        raise PackSchemaError(f"{location} is missing: {', '.join(missing)}.")
    if unknown:
        raise PackSchemaError(f"{location} has unknown fields: {', '.join(unknown)}.")


def _string(value: object, location: str, *, nonempty: bool = True) -> str:
    if not isinstance(value, str) or (nonempty and not value):
        qualifier = "a non-empty string" if nonempty else "a string"
        raise PackSchemaError(f"{location} must be {qualifier}.")
    if "\x00" in value:
        raise PackSchemaError(f"{location} contains a NUL byte.")
    return value


def _integer(value: object, location: str, *, minimum: int | None = None) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise PackSchemaError(f"{location} must be an integer.")
    if minimum is not None and value < minimum:
        raise PackSchemaError(f"{location} must be at least {minimum}.")
    return value


def _list(value: object, location: str) -> list[Any]:
    if not isinstance(value, list):
        raise PackSchemaError(f"{location} must be an array.")
    return value


def _unique_strings(value: object, location: str) -> tuple[str, ...]:
    items = tuple(_string(item, f"{location}[{index}]") for index, item in enumerate(_list(value, location)))
    if len(items) != len(set(items)):
        raise PackSchemaError(f"{location} must not contain duplicates.")
    return items


def _slug(value: object, location: str) -> str:
    text = _string(value, location)
    if not _SLUG.fullmatch(text):
        raise PackSchemaError(f"{location} must be a lowercase identifier.")
    return text


def _logical_name(value: object, location: str) -> str:
    text = _string(value, location)
    if not _LOGICAL_NAME.fullmatch(text):
        raise PackSchemaError(f"{location} is not a valid logical name.")
    return text


def _commit(value: object, location: str) -> str:
    text = _string(value, location)
    if not _COMMIT.fullmatch(text):
        raise PackSchemaError(f"{location} must be a lowercase 40-character Git commit.")
    return text


def _sha256(value: object, location: str) -> str:
    text = _string(value, location)
    if not _SHA256.fullmatch(text):
        raise PackSchemaError(f"{location} must be a lowercase SHA-256 digest.")
    return text


def _semver(value: object, location: str) -> str:
    text = _string(value, location)
    if not _SEMVER.fullmatch(text):
        raise PackSchemaError(f"{location} must be a semantic version.")
    return text


def _relative_path(value: object, location: str) -> str:
    text = _string(value, location)
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or text in {".", ".."}
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
        or "\\" in text
    ):
        raise PackSchemaError(f"{location} must be a safe POSIX relative path.")
    return path.as_posix()


def _target_directory(value: object, location: str) -> str:
    text = _string(value, location)
    if "/" in text or "\\" in text or text in {".", ".."}:
        raise PackSchemaError(f"{location} must be one directory name.")
    return text


def _https_url(value: object, location: str, *, git: bool = False) -> str:
    text = _string(value, location)
    parsed = urlsplit(text)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise PackSchemaError(f"{location} must be an HTTPS URL without credentials or query data.")
    if git and not parsed.path.endswith(".git"):
        raise PackSchemaError(f"{location} must identify a .git repository.")
    return text


def _optional_https_url(value: object, location: str) -> str | None:
    if value is None:
        return None
    return _https_url(value, location)


def _environment_name(value: object, location: str) -> str:
    text = _string(value, location)
    if not _ENV_NAME.fullmatch(text):
        raise PackSchemaError(f"{location} is not a valid environment-variable name.")
    return text


def normalize_package_name(value: object, location: str = "package name") -> str:
    text = _string(value, location)
    if not _PACKAGE_NAME.fullmatch(text):
        raise PackSchemaError(f"{location} is not a valid Python/system package name.")
    return re.sub(r"[-_.]+", "-", text).lower()


def normalize_python_specifier(value: object, location: str = "specifier") -> str:
    text = _string(value, location, nonempty=False).replace(" ", "")
    if not text:
        return ""
    normalized: list[str] = []
    for index, item in enumerate(text.split(",")):
        if not item or not _PYTHON_SPECIFIER.fullmatch(item):
            raise PackSchemaError(
                f"{location}[{index}] uses an unsupported version constraint: {item!r}."
            )
        if item.startswith("!=") and item.endswith(".*"):
            raise PackSchemaError(f"{location} does not support wildcard exclusions.")
        normalized.append(item)
    return ",".join(sorted(set(normalized)))


def _string_mapping(value: object, location: str) -> dict[str, str]:
    source = _mapping(value, location)
    result: dict[str, str] = {}
    for name, raw_value in source.items():
        result[_environment_name(name, f"{location}.{name}")] = _string(
            raw_value, f"{location}.{name}", nonempty=False
        )
    return dict(sorted(result.items()))


def _command(
    value: object,
    location: str,
    *,
    allow_empty: bool,
) -> tuple[str, ...]:
    parts = tuple(
        _string(item, f"{location}[{index}]")
        for index, item in enumerate(_list(value, location))
    )
    if not allow_empty and not parts:
        raise PackSchemaError(f"{location} must not be empty.")
    return parts


def _install_phase(value: object, location: str) -> str:
    phase = _string(value, location)
    if phase not in {"bootstrap", "lazy"}:
        raise PackSchemaError(f"{location} must be 'bootstrap' or 'lazy'.")
    return phase


@dataclass(frozen=True)
class PackRefV1:
    id: str
    repository: str
    ref: str
    manifest_sha256: str
    schema: int = field(default=SCHEMA_VERSION, init=False)

    @classmethod
    def from_dict(cls, value: object, location: str = "pack ref") -> "PackRefV1":
        source = _mapping(value, location)
        _keys(
            source,
            location,
            required=("schema", "id", "repository", "ref", "manifest_sha256"),
        )
        if source["schema"] != SCHEMA_VERSION:
            raise PackSchemaError(f"{location}.schema must be {SCHEMA_VERSION}.")
        return cls(
            id=_slug(source["id"], f"{location}.id"),
            repository=_https_url(source["repository"], f"{location}.repository", git=True),
            ref=_commit(source["ref"], f"{location}.ref"),
            manifest_sha256=_sha256(
                source["manifest_sha256"], f"{location}.manifest_sha256"
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "id": self.id,
            "repository": self.repository,
            "ref": self.ref,
            "manifest_sha256": self.manifest_sha256,
        }


@dataclass(frozen=True)
class CompatibilityV1:
    core_manifest_api: int
    compatible_refs: tuple[str, ...]
    tested_refs: tuple[str, ...]

    @classmethod
    def from_dict(cls, value: object, location: str) -> "CompatibilityV1":
        source = _mapping(value, location)
        _keys(source, location, required=("core_manifest_api", "comfyui"))
        api = _integer(source["core_manifest_api"], f"{location}.core_manifest_api", minimum=1)
        if api != CORE_MANIFEST_API:
            raise PackSchemaError(
                f"{location}.core_manifest_api {api} is unsupported; expected {CORE_MANIFEST_API}."
            )
        comfyui = _mapping(source["comfyui"], f"{location}.comfyui")
        if "probes" in comfyui:
            raise PackSchemaError(
                f"{location}.comfyui.probes must move to top-level probes using "
                "phase='post_clone', type='file_symbols', and target='comfyui'."
            )
        _keys(
            comfyui,
            f"{location}.comfyui",
            required=("compatible_refs", "tested_refs"),
        )
        compatible = tuple(
            _commit(item, f"{location}.comfyui.compatible_refs[{index}]")
            for index, item in enumerate(
                _list(comfyui["compatible_refs"], f"{location}.comfyui.compatible_refs")
            )
        )
        tested = tuple(
            _commit(item, f"{location}.comfyui.tested_refs[{index}]")
            for index, item in enumerate(
                _list(comfyui["tested_refs"], f"{location}.comfyui.tested_refs")
            )
        )
        if not compatible:
            raise PackSchemaError(f"{location}.comfyui.compatible_refs must not be empty.")
        if len(compatible) != len(set(compatible)) or len(tested) != len(set(tested)):
            raise PackSchemaError(f"{location}.comfyui revision arrays must not contain duplicates.")
        if not set(tested).issubset(compatible):
            raise PackSchemaError(f"{location}.comfyui.tested_refs must be compatible refs.")
        return cls(api, tuple(sorted(compatible)), tuple(sorted(tested)))

    def to_dict(self) -> dict[str, object]:
        return {
            "core_manifest_api": self.core_manifest_api,
            "comfyui": {
                "compatible_refs": list(self.compatible_refs),
                "tested_refs": list(self.tested_refs),
            },
        }


@dataclass(frozen=True)
class NodeRootV1:
    source: str
    target: str

    @classmethod
    def from_dict(cls, value: object, location: str) -> "NodeRootV1":
        source = _mapping(value, location)
        _keys(source, location, required=("source", "target"))
        return cls(
            _relative_path(source["source"], f"{location}.source"),
            _target_directory(source["target"], f"{location}.target"),
        )

    def to_dict(self) -> dict[str, str]:
        return {"source": self.source, "target": self.target}


@dataclass(frozen=True)
class GitDependencyV1:
    id: str
    repository: str
    ref: str
    destination: str
    scope: str
    requirements_file: str | None = None
    requirements_source: str | None = None
    requirements_format: str | None = None
    install_phase: str | None = None
    license_gate: str | None = None
    kind: str = field(default="git", init=False)

    @classmethod
    def from_dict(cls, value: object, location: str) -> "GitDependencyV1":
        source = _mapping(value, location)
        _keys(
            source,
            location,
            required=("kind", "id", "repository", "ref", "destination", "scope"),
            optional=(
                "requirements_file",
                "requirements_source",
                "requirements_format",
                "install_phase",
                "license_gate",
            ),
        )
        if source["kind"] != "git":
            raise PackSchemaError(f"{location}.kind must be 'git'.")
        scope = _logical_name(source["scope"], f"{location}.scope")
        requirements_value = source.get("requirements_file")
        requirements_file = (
            None
            if requirements_value is None
            else _relative_path(requirements_value, f"{location}.requirements_file")
        )
        source_value = source.get("requirements_source")
        requirements_source = (
            None
            if source_value is None
            else _string(source_value, f"{location}.requirements_source")
        )
        if requirements_source not in {None, "dependency", "pack"}:
            raise PackSchemaError(
                f"{location}.requirements_source must be 'dependency' or 'pack'."
            )
        format_value = source.get("requirements_format")
        requirements_format = (
            None
            if format_value is None
            else _string(format_value, f"{location}.requirements_format")
        )
        if requirements_format not in {
            None,
            "requirements.txt",
            "pyproject.toml",
            "comfycolab-environment-toml",
        }:
            raise PackSchemaError(
                f"{location}.requirements_format is unsupported."
            )
        if requirements_file is None and (
            requirements_source is not None or requirements_format is not None
        ):
            raise PackSchemaError(
                f"{location}.requirements_file is required when requirements "
                "source or format is declared."
            )
        phase_value = source.get("install_phase")
        gate_value = source.get("license_gate")
        license_gate = (
            None
            if gate_value is None
            else _logical_name(gate_value, f"{location}.license_gate")
        )
        return cls(
            id=_slug(source["id"], f"{location}.id"),
            repository=_https_url(source["repository"], f"{location}.repository", git=True),
            ref=_commit(source["ref"], f"{location}.ref"),
            destination=_relative_path(source["destination"], f"{location}.destination"),
            scope=scope,
            requirements_file=requirements_file,
            requirements_source=requirements_source,
            requirements_format=requirements_format,
            install_phase=(
                None
                if phase_value is None
                else _install_phase(phase_value, f"{location}.install_phase")
            ),
            license_gate=license_gate,
        )

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "kind": self.kind,
            "id": self.id,
            "repository": self.repository,
            "ref": self.ref,
            "destination": self.destination,
            "scope": self.scope,
        }
        if self.requirements_file is not None:
            result["requirements_file"] = self.requirements_file
        if self.requirements_source is not None:
            result["requirements_source"] = self.requirements_source
        if self.requirements_format is not None:
            result["requirements_format"] = self.requirements_format
        if self.install_phase is not None:
            result["install_phase"] = self.install_phase
        if self.license_gate is not None:
            result["license_gate"] = self.license_gate
        return result


@dataclass(frozen=True)
class HuggingFaceArtifactV1:
    path: str
    bytes: int
    sha256: str

    @classmethod
    def from_dict(cls, value: object, location: str) -> "HuggingFaceArtifactV1":
        source = _mapping(value, location)
        _keys(source, location, required=("path", "bytes", "sha256"))
        return cls(
            path=_relative_path(source["path"], f"{location}.path"),
            bytes=_integer(source["bytes"], f"{location}.bytes", minimum=1),
            sha256=_sha256(source["sha256"], f"{location}.sha256"),
        )

    def to_dict(self) -> dict[str, object]:
        return {"path": self.path, "bytes": self.bytes, "sha256": self.sha256}


@dataclass(frozen=True)
class HuggingFaceDependencyV1:
    id: str
    repository: str
    ref: str
    destination: str
    scope: str
    artifacts: tuple[HuggingFaceArtifactV1, ...] = ()
    install_phase: str | None = None
    license_gate: str | None = None
    kind: str = field(default="huggingface", init=False)

    @classmethod
    def from_dict(cls, value: object, location: str) -> "HuggingFaceDependencyV1":
        source = _mapping(value, location)
        _keys(
            source,
            location,
            required=("kind", "id", "repository", "ref", "destination", "scope"),
            optional=("artifacts", "install_phase", "license_gate"),
        )
        if source["kind"] != "huggingface":
            raise PackSchemaError(f"{location}.kind must be 'huggingface'.")
        repository = _string(source["repository"], f"{location}.repository")
        parts = repository.split("/")
        if len(parts) != 2 or not all(
            _HF_REPOSITORY_PART.fullmatch(part) for part in parts
        ):
            raise PackSchemaError(
                f"{location}.repository must be a Hugging Face owner/name identifier."
            )
        gate_value = source.get("license_gate")
        artifacts = tuple(
            HuggingFaceArtifactV1.from_dict(
                item, f"{location}.artifacts[{index}]"
            )
            for index, item in enumerate(
                _list(source.get("artifacts", []), f"{location}.artifacts")
            )
        )
        artifact_paths = [item.path for item in artifacts]
        if len(artifact_paths) != len(set(artifact_paths)):
            raise PackSchemaError(f"{location}.artifacts contains duplicate paths.")
        phase_value = source.get("install_phase")
        return cls(
            id=_slug(source["id"], f"{location}.id"),
            repository=repository,
            ref=_commit(source["ref"], f"{location}.ref"),
            destination=_relative_path(source["destination"], f"{location}.destination"),
            scope=_logical_name(source["scope"], f"{location}.scope"),
            artifacts=tuple(sorted(artifacts, key=lambda item: item.path)),
            install_phase=(
                None
                if phase_value is None
                else _install_phase(phase_value, f"{location}.install_phase")
            ),
            license_gate=(
                None
                if gate_value is None
                else _logical_name(gate_value, f"{location}.license_gate")
            ),
        )

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "kind": self.kind,
            "id": self.id,
            "repository": self.repository,
            "ref": self.ref,
            "destination": self.destination,
            "scope": self.scope,
        }
        if self.artifacts:
            result["artifacts"] = [item.to_dict() for item in self.artifacts]
        if self.install_phase is not None:
            result["install_phase"] = self.install_phase
        if self.license_gate is not None:
            result["license_gate"] = self.license_gate
        return result


@dataclass(frozen=True)
class ArtifactDependencyV1:
    id: str
    url: str
    sha256: str
    destination: str
    scope: str
    install_phase: str | None = None
    license_gate: str | None = None
    kind: str = field(default="artifact", init=False)

    @classmethod
    def from_dict(cls, value: object, location: str) -> "ArtifactDependencyV1":
        source = _mapping(value, location)
        _keys(
            source,
            location,
            required=("kind", "id", "url", "sha256", "destination", "scope"),
            optional=("install_phase", "license_gate"),
        )
        if source["kind"] != "artifact":
            raise PackSchemaError(f"{location}.kind must be 'artifact'.")
        phase_value = source.get("install_phase")
        gate_value = source.get("license_gate")
        return cls(
            id=_slug(source["id"], f"{location}.id"),
            url=_https_url(source["url"], f"{location}.url"),
            sha256=_sha256(source["sha256"], f"{location}.sha256"),
            destination=_relative_path(source["destination"], f"{location}.destination"),
            scope=_logical_name(source["scope"], f"{location}.scope"),
            install_phase=(
                None
                if phase_value is None
                else _install_phase(phase_value, f"{location}.install_phase")
            ),
            license_gate=(
                None
                if gate_value is None
                else _logical_name(gate_value, f"{location}.license_gate")
            ),
        )

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "kind": self.kind,
            "id": self.id,
            "url": self.url,
            "sha256": self.sha256,
            "destination": self.destination,
            "scope": self.scope,
        }
        if self.install_phase is not None:
            result["install_phase"] = self.install_phase
        if self.license_gate is not None:
            result["license_gate"] = self.license_gate
        return result


DependencyV1 = GitDependencyV1 | HuggingFaceDependencyV1 | ArtifactDependencyV1


def dependency_from_dict(value: object, location: str) -> DependencyV1:
    source = _mapping(value, location)
    kind = source.get("kind")
    if kind == "git":
        return GitDependencyV1.from_dict(source, location)
    if kind == "huggingface":
        return HuggingFaceDependencyV1.from_dict(source, location)
    if kind == "artifact":
        return ArtifactDependencyV1.from_dict(source, location)
    raise PackSchemaError(f"{location}.kind is unsupported: {kind!r}.")


@dataclass(frozen=True)
class PythonRequirementV1:
    name: str
    specifier: str
    url: str | None = None
    sha256: str | None = None

    @classmethod
    def from_dict(cls, value: object, location: str) -> "PythonRequirementV1":
        source = _mapping(value, location)
        _keys(
            source,
            location,
            required=("name", "specifier"),
            optional=("url", "sha256"),
        )
        url = _optional_https_url(source.get("url"), f"{location}.url")
        digest_value = source.get("sha256")
        digest = None if digest_value is None else _sha256(digest_value, f"{location}.sha256")
        if (url is None) != (digest is None):
            raise PackSchemaError(f"{location}.url and sha256 must be supplied together.")
        return cls(
            normalize_package_name(source["name"], f"{location}.name"),
            normalize_python_specifier(source["specifier"], f"{location}.specifier"),
            url,
            digest,
        )

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {"name": self.name, "specifier": self.specifier}
        if self.url is not None:
            result["url"] = self.url
            result["sha256"] = self.sha256
        return result


@dataclass(frozen=True)
class SystemDependencyV1:
    manager: str
    name: str
    version: str | None
    scope: str

    @classmethod
    def from_dict(cls, value: object, location: str) -> "SystemDependencyV1":
        source = _mapping(value, location)
        _keys(
            source,
            location,
            required=("manager", "name", "scope"),
            optional=("version",),
        )
        manager = _string(source["manager"], f"{location}.manager")
        if manager not in {"apt", "conda", "pixi", "binary"}:
            raise PackSchemaError(f"{location}.manager is unsupported.")
        scope = _string(source["scope"], f"{location}.scope")
        if scope not in {"runtime", "environment"}:
            raise PackSchemaError(f"{location}.scope must be 'runtime' or 'environment'.")
        version_value = source.get("version")
        version = None if version_value is None else _string(version_value, f"{location}.version")
        return cls(
            manager,
            normalize_package_name(source["name"], f"{location}.name"),
            version,
            scope,
        )

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "manager": self.manager,
            "name": self.name,
            "scope": self.scope,
        }
        if self.version is not None:
            result["version"] = self.version
        return result


@dataclass(frozen=True)
class EnvironmentV1:
    id: str
    kind: str
    scope: str
    python: str | None
    python_requirements: tuple[PythonRequirementV1, ...]
    system_dependencies: tuple[SystemDependencyV1, ...]
    cache_profile: str | None
    license_gate: str | None

    @classmethod
    def from_dict(cls, value: object, location: str) -> "EnvironmentV1":
        source = _mapping(value, location)
        _keys(
            source,
            location,
            required=("id",),
            optional=(
                "kind",
                "scope",
                "profile",
                "python",
                "python_requirements",
                "system_dependencies",
                "cache_profile",
                "license_gate",
            ),
        )
        identifier = _slug(source["id"], f"{location}.id")
        scope = _logical_name(source.get("scope", "main"), f"{location}.scope")
        kind = _string(
            source.get("kind", "main" if scope == "main" else "isolated"),
            f"{location}.kind",
        )
        if kind not in {"main", "isolated"}:
            raise PackSchemaError(f"{location}.kind must be 'main' or 'isolated'.")
        if kind == "main" and identifier != "main":
            raise PackSchemaError(f"{location}.id must be 'main' for a main environment.")
        python_value = source.get("python")
        python = None if python_value is None else _string(python_value, f"{location}.python")
        requirements = tuple(
            PythonRequirementV1.from_dict(item, f"{location}.python_requirements[{index}]")
            for index, item in enumerate(
                _list(source.get("python_requirements", []), f"{location}.python_requirements")
            )
        )
        systems = tuple(
            SystemDependencyV1.from_dict(item, f"{location}.system_dependencies[{index}]")
            for index, item in enumerate(
                _list(source.get("system_dependencies", []), f"{location}.system_dependencies")
            )
        )
        profile_value = source.get("profile")
        cache_value = source.get("cache_profile", profile_value)
        if (
            profile_value is not None
            and source.get("cache_profile") is not None
            and profile_value != source["cache_profile"]
        ):
            raise PackSchemaError(f"{location}.profile and cache_profile disagree.")
        cache = None if cache_value is None else _logical_name(cache_value, f"{location}.cache_profile")
        gate_value = source.get("license_gate")
        gate = (
            None
            if gate_value is None
            else _logical_name(gate_value, f"{location}.license_gate")
        )
        return cls(identifier, kind, scope, python, requirements, systems, cache, gate)

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "id": self.id,
            "kind": self.kind,
            "scope": self.scope,
            "python_requirements": [item.to_dict() for item in self.python_requirements],
            "system_dependencies": [item.to_dict() for item in self.system_dependencies],
        }
        if self.python is not None:
            result["python"] = self.python
        if self.cache_profile is not None:
            result["cache_profile"] = self.cache_profile
        if self.license_gate is not None:
            result["license_gate"] = self.license_gate
        return result


@dataclass(frozen=True)
class PatchFileV1:
    path: str
    before_sha256: str
    after_sha256: str

    @classmethod
    def from_dict(cls, value: object, location: str) -> "PatchFileV1":
        source = _mapping(value, location)
        _keys(source, location, required=("path", "before_sha256", "after_sha256"))
        before = _sha256(source["before_sha256"], f"{location}.before_sha256")
        after = _sha256(source["after_sha256"], f"{location}.after_sha256")
        if before == after:
            raise PackSchemaError(f"{location} must change the target file digest.")
        return cls(
            _relative_path(source["path"], f"{location}.path"),
            before,
            after,
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "path": self.path,
            "before_sha256": self.before_sha256,
            "after_sha256": self.after_sha256,
        }


@dataclass(frozen=True)
class PatchV1:
    id: str
    target: str
    target_ref: str
    specification: str
    specification_sha256: str | None
    order: int
    composition: tuple[str, ...]
    files: tuple[PatchFileV1, ...]

    @classmethod
    def from_dict(cls, value: object, location: str) -> "PatchV1":
        source = _mapping(value, location)
        _keys(
            source,
            location,
            required=("id", "target"),
            optional=(
                "target_ref",
                "source_ref",
                "specification",
                "path",
                "specification_sha256",
                "sha256",
                "files",
                "order",
                "composition",
            ),
        )
        target_ref_value = source.get("target_ref", source.get("source_ref"))
        if target_ref_value is None:
            raise PackSchemaError(f"{location} requires target_ref or source_ref.")
        if (
            source.get("target_ref") is not None
            and source.get("source_ref") is not None
            and source["target_ref"] != source["source_ref"]
        ):
            raise PackSchemaError(f"{location}.target_ref and source_ref disagree.")
        specification_value = source.get("specification", source.get("path"))
        if specification_value is None:
            raise PackSchemaError(f"{location} requires specification or path.")
        if (
            source.get("specification") is not None
            and source.get("path") is not None
            and source["specification"] != source["path"]
        ):
            raise PackSchemaError(f"{location}.specification and path disagree.")
        digest_value = source.get("specification_sha256", source.get("sha256"))
        if (
            source.get("specification_sha256") is not None
            and source.get("sha256") is not None
            and source["specification_sha256"] != source["sha256"]
        ):
            raise PackSchemaError(
                f"{location}.specification_sha256 and sha256 disagree."
            )
        files = tuple(
            PatchFileV1.from_dict(item, f"{location}.files[{index}]")
            for index, item in enumerate(
                _list(source.get("files", []), f"{location}.files")
            )
        )
        paths = [item.path for item in files]
        if len(paths) != len(set(paths)):
            raise PackSchemaError(f"{location}.files contains duplicate target paths.")
        composition = tuple(
            _slug(item, f"{location}.composition[{index}]")
            for index, item in enumerate(
                _list(source.get("composition", []), f"{location}.composition")
            )
        )
        if len(composition) != len(set(composition)):
            raise PackSchemaError(f"{location}.composition must not contain duplicates.")
        return cls(
            id=_slug(source["id"], f"{location}.id"),
            target=_slug(source["target"], f"{location}.target"),
            target_ref=_commit(target_ref_value, f"{location}.target_ref"),
            specification=_relative_path(
                specification_value, f"{location}.specification"
            ),
            specification_sha256=(
                None
                if digest_value is None
                else _sha256(digest_value, f"{location}.specification_sha256")
            ),
            order=_integer(source.get("order", 0), f"{location}.order", minimum=0),
            composition=composition,
            files=files,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "target": self.target,
            "target_ref": self.target_ref,
            "specification": self.specification,
            **(
                {"specification_sha256": self.specification_sha256}
                if self.specification_sha256 is not None
                else {}
            ),
            "order": self.order,
            "composition": list(self.composition),
            "files": [item.to_dict() for item in self.files],
        }


@dataclass(frozen=True)
class HookV1:
    path: str
    network: str
    write_roots: tuple[str, ...]
    timeout_seconds: int

    @classmethod
    def from_dict(cls, value: object, location: str) -> "HookV1":
        source = _mapping(value, location)
        _keys(
            source,
            location,
            required=("path", "network", "write_roots"),
            optional=("timeout_seconds",),
        )
        if source["network"] != "none":
            raise PackSchemaError(f"{location}.network must be 'none' in manifest API v1.")
        roots = tuple(
            _logical_name(item, f"{location}.write_roots[{index}]")
            for index, item in enumerate(
                _list(source["write_roots"], f"{location}.write_roots")
            )
        )
        if len(roots) != len(set(roots)):
            raise PackSchemaError(f"{location}.write_roots must not contain duplicates.")
        timeout = _integer(
            source.get("timeout_seconds", 300),
            f"{location}.timeout_seconds",
            minimum=1,
        )
        if timeout > 3600:
            raise PackSchemaError(f"{location}.timeout_seconds must not exceed 3600.")
        return cls(
            _relative_path(source["path"], f"{location}.path"),
            "none",
            roots,
            timeout,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "network": self.network,
            "write_roots": list(self.write_roots),
            "timeout_seconds": self.timeout_seconds,
        }


@dataclass(frozen=True)
class ProbeV1:
    phase: str
    type: str
    target: str | None = None
    path: str | None = None
    sha256: str | None = None
    symbol: str | None = None
    symbols: tuple[str, ...] = ()
    values: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, value: object, location: str) -> "ProbeV1":
        source = _mapping(value, location)
        _keys(
            source,
            location,
            required=("phase", "type"),
            optional=("target", "path", "sha256", "symbol", "symbols", "values"),
        )
        phase = _string(source["phase"], f"{location}.phase")
        if phase not in {"post_clone", "post_start"}:
            raise PackSchemaError(f"{location}.phase is unsupported.")
        kind = _string(source["type"], f"{location}.type")
        if kind not in {
            "path_exists",
            "file_sha256",
            "python_symbol",
            "file_symbols",
            "comfy_node_ids",
        }:
            raise PackSchemaError(f"{location}.type is unsupported.")
        target = (
            None
            if source.get("target") is None
            else _logical_name(source["target"], f"{location}.target")
        )
        path = (
            None
            if source.get("path") is None
            else _relative_path(source["path"], f"{location}.path")
        )
        digest = (
            None
            if source.get("sha256") is None
            else _sha256(source["sha256"], f"{location}.sha256")
        )
        symbol = (
            None
            if source.get("symbol") is None
            else _string(source["symbol"], f"{location}.symbol")
        )
        symbols = _unique_strings(source.get("symbols", []), f"{location}.symbols")
        values = _unique_strings(source.get("values", []), f"{location}.values")
        if kind == "path_exists" and (
            path is None or target or digest or symbol or symbols or values
        ):
            raise PackSchemaError(f"{location} path_exists requires only path.")
        if kind == "file_sha256" and (
            path is None or target or digest is None or symbol or symbols or values
        ):
            raise PackSchemaError(f"{location} file_sha256 requires path and sha256.")
        if kind == "python_symbol" and (
            path is None or target or symbol is None or digest or symbols or values
        ):
            raise PackSchemaError(f"{location} python_symbol requires path and symbol.")
        if kind == "file_symbols" and (
            phase != "post_clone"
            or target != "comfyui"
            or path is None
            or not symbols
            or digest
            or symbol
            or values
        ):
            raise PackSchemaError(
                f"{location} file_symbols requires phase='post_clone', "
                "target='comfyui', path, and symbols."
            )
        if kind == "comfy_node_ids" and (
            not values or target or path or digest or symbol or symbols
        ):
            raise PackSchemaError(f"{location} comfy_node_ids requires only values.")
        return cls(phase, kind, target, path, digest, symbol, symbols, values)

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {"phase": self.phase, "type": self.type}
        if self.target is not None:
            result["target"] = self.target
        if self.path is not None:
            result["path"] = self.path
        if self.sha256 is not None:
            result["sha256"] = self.sha256
        if self.symbol is not None:
            result["symbol"] = self.symbol
        if self.symbols:
            result["symbols"] = list(self.symbols)
        if self.values:
            result["values"] = list(self.values)
        return result


@dataclass(frozen=True)
class LicenseV1:
    id: str
    name: str | None
    url: str | None
    notice: str | None
    scope: str | None
    acceptance: str | None

    @classmethod
    def from_dict(cls, value: object, location: str) -> "LicenseV1":
        source = _mapping(value, location)
        _keys(
            source,
            location,
            required=("id",),
            optional=("name", "url", "notice", "path", "scope", "acceptance"),
        )
        notice_value = source.get("notice", source.get("path"))
        if (
            source.get("notice") is not None
            and source.get("path") is not None
            and source["notice"] != source["path"]
        ):
            raise PackSchemaError(f"{location}.notice and path disagree.")
        acceptance_value = source.get("acceptance")
        has_named_notice_record = (
            all(source.get(field) is not None for field in ("name", "url"))
            and notice_value is not None
        )
        has_path_notice_record = (
            notice_value is not None
            and source.get("name") is None
            and source.get("url") is None
        )
        has_notice_record = has_named_notice_record or has_path_notice_record
        has_gate_record = acceptance_value is not None
        if has_notice_record == has_gate_record:
            raise PackSchemaError(
                f"{location} must declare either a notice path, name/url/notice, "
                "or one acceptance gate."
            )
        if (source.get("name") is None) != (source.get("url") is None):
            raise PackSchemaError(f"{location}.name and url must be supplied together.")
        scope: str | None = None
        if has_named_notice_record:
            scope = _string(source.get("scope", "code"), f"{location}.scope")
            if scope not in {"code", "model", "data"}:
                raise PackSchemaError(f"{location}.scope is unsupported.")
        elif has_path_notice_record and source.get("scope") is not None:
            raise PackSchemaError(
                f"{location}.scope requires a named name/url/notice record."
            )
        elif source.get("scope") is not None:
            raise PackSchemaError(f"{location}.scope is invalid for an acceptance gate.")
        return cls(
            _slug(source["id"], f"{location}.id"),
            (
                _string(source["name"], f"{location}.name")
                if has_named_notice_record
                else None
            ),
            (
                _https_url(source["url"], f"{location}.url")
                if has_named_notice_record
                else None
            ),
            (
                _relative_path(notice_value, f"{location}.notice")
                if has_notice_record
                else None
            ),
            scope,
            (
                _logical_name(acceptance_value, f"{location}.acceptance")
                if has_gate_record
                else None
            ),
        )

    def to_dict(self) -> dict[str, str]:
        result = {"id": self.id}
        if self.acceptance is not None:
            result["acceptance"] = self.acceptance
        elif self.name is None and self.url is None:
            assert self.notice is not None
            result["path"] = self.notice
        else:
            assert self.name is not None
            assert self.url is not None
            assert self.notice is not None
            assert self.scope is not None
            result.update(
                {
                    "name": self.name,
                    "url": self.url,
                    "notice": self.notice,
                    "scope": self.scope,
                }
            )
        return result


@dataclass(frozen=True)
class PackManifestV1:
    id: str
    version: str
    display_name: str
    compatibility: CompatibilityV1
    node_roots: tuple[NodeRootV1, ...]
    dependencies: tuple[DependencyV1, ...]
    patches: tuple[PatchV1, ...]
    environments: tuple[EnvironmentV1, ...]
    hooks: Mapping[str, HookV1]
    runtime_env: Mapping[str, str]
    workflows: tuple[str, ...]
    probes: tuple[ProbeV1, ...]
    health_node_ids: tuple[str, ...]
    health_command: tuple[str, ...]
    licenses: tuple[LicenseV1, ...]
    cache_profiles: tuple[str, ...]
    accelerators: tuple[str, ...]
    validation_commands: tuple[tuple[str, ...], ...]
    readiness_namespace: str | None
    readiness_fields: tuple[str, ...]
    license_gate: str | None
    source_sha256: str | None = field(default=None, compare=False, repr=False)
    schema: int = field(default=SCHEMA_VERSION, init=False)

    @classmethod
    def from_dict(
        cls,
        value: object,
        location: str = "pack manifest",
        *,
        source_sha256: str | None = None,
    ) -> "PackManifestV1":
        source = _mapping(value, location)
        _keys(
            source,
            location,
            required=(
                "schema",
                "id",
                "version",
                "display_name",
                "compatibility",
                "node_roots",
                "dependencies",
                "patches",
                "environments",
                "hooks",
                "workflows",
                "probes",
                "health_checks",
                "licenses",
            ),
            optional=(
                "runtime_env",
                "cache_profiles",
                "accelerators",
                "validation_commands",
                "readiness",
                "license_gate",
            ),
        )
        if source["schema"] != SCHEMA_VERSION:
            raise PackSchemaError(f"{location}.schema must be {SCHEMA_VERSION}.")
        node_roots = tuple(
            NodeRootV1.from_dict(item, f"{location}.node_roots[{index}]")
            for index, item in enumerate(_list(source["node_roots"], f"{location}.node_roots"))
        )
        targets = [item.target for item in node_roots]
        if len(targets) != len(set(targets)):
            raise PackSchemaError(f"{location}.node_roots contains duplicate targets.")
        dependencies = tuple(
            dependency_from_dict(item, f"{location}.dependencies[{index}]")
            for index, item in enumerate(
                _list(source["dependencies"], f"{location}.dependencies")
            )
        )
        dependency_ids = [item.id for item in dependencies]
        if len(dependency_ids) != len(set(dependency_ids)):
            raise PackSchemaError(f"{location}.dependencies contains duplicate IDs.")
        patches = tuple(
            PatchV1.from_dict(item, f"{location}.patches[{index}]")
            for index, item in enumerate(_list(source["patches"], f"{location}.patches"))
        )
        patch_ids = [item.id for item in patches]
        if len(patch_ids) != len(set(patch_ids)):
            raise PackSchemaError(f"{location}.patches contains duplicate IDs.")
        environments = tuple(
            EnvironmentV1.from_dict(item, f"{location}.environments[{index}]")
            for index, item in enumerate(
                _list(source["environments"], f"{location}.environments")
            )
        )
        environment_ids = [item.id for item in environments]
        if len(environment_ids) != len(set(environment_ids)):
            raise PackSchemaError(f"{location}.environments contains duplicate IDs.")
        hook_source = _mapping(source["hooks"], f"{location}.hooks")
        unknown_hooks = sorted(set(hook_source) - {"configure", "doctor", "runtime_env"})
        if unknown_hooks:
            raise PackSchemaError(
                f"{location}.hooks has unsupported phases: {', '.join(unknown_hooks)}."
            )
        hooks = {
            name: HookV1.from_dict(item, f"{location}.hooks.{name}")
            for name, item in sorted(hook_source.items())
        }
        workflows = tuple(
            _relative_path(item, f"{location}.workflows[{index}]")
            for index, item in enumerate(_list(source["workflows"], f"{location}.workflows"))
        )
        if len(workflows) != len(set(workflows)):
            raise PackSchemaError(f"{location}.workflows must not contain duplicates.")
        probes = tuple(
            ProbeV1.from_dict(item, f"{location}.probes[{index}]")
            for index, item in enumerate(_list(source["probes"], f"{location}.probes"))
        )
        health = _mapping(source["health_checks"], f"{location}.health_checks")
        _keys(
            health,
            f"{location}.health_checks",
            required=("node_ids",),
            optional=("command",),
        )
        health_ids = _unique_strings(
            health["node_ids"], f"{location}.health_checks.node_ids"
        )
        health_command = _command(
            health.get("command", []), f"{location}.health_checks.command", allow_empty=True
        )
        licenses = tuple(
            LicenseV1.from_dict(item, f"{location}.licenses[{index}]")
            for index, item in enumerate(_list(source["licenses"], f"{location}.licenses"))
        )
        license_ids = [item.id for item in licenses]
        if len(license_ids) != len(set(license_ids)):
            raise PackSchemaError(f"{location}.licenses contains duplicate IDs.")
        cache_profiles = tuple(
            _relative_path(item, f"{location}.cache_profiles[{index}]")
            for index, item in enumerate(
                _list(source.get("cache_profiles", []), f"{location}.cache_profiles")
            )
        )
        if len(cache_profiles) != len(set(cache_profiles)):
            raise PackSchemaError(f"{location}.cache_profiles must not contain duplicates.")
        accelerators = tuple(
            _logical_name(item, f"{location}.accelerators[{index}]")
            for index, item in enumerate(
                _list(source.get("accelerators", []), f"{location}.accelerators")
            )
        )
        if len(accelerators) != len(set(accelerators)):
            raise PackSchemaError(f"{location}.accelerators must not contain duplicates.")
        validation_commands = tuple(
            _command(
                item,
                f"{location}.validation_commands[{index}]",
                allow_empty=False,
            )
            for index, item in enumerate(
                _list(
                    source.get("validation_commands", []),
                    f"{location}.validation_commands",
                )
            )
        )
        readiness_value = source.get("readiness")
        readiness_namespace: str | None = None
        readiness_fields: tuple[str, ...] = ()
        if readiness_value is not None:
            readiness = _mapping(readiness_value, f"{location}.readiness")
            _keys(
                readiness,
                f"{location}.readiness",
                required=("namespace", "fields"),
            )
            readiness_namespace = _string(
                readiness["namespace"], f"{location}.readiness.namespace"
            )
            if not readiness_namespace.startswith("packs."):
                raise PackSchemaError(
                    f"{location}.readiness.namespace must be under 'packs.'."
                )
            readiness_fields = tuple(
                _logical_name(item, f"{location}.readiness.fields[{index}]")
                for index, item in enumerate(
                    _list(readiness["fields"], f"{location}.readiness.fields")
                )
            )
            if len(readiness_fields) != len(set(readiness_fields)):
                raise PackSchemaError(
                    f"{location}.readiness.fields must not contain duplicates."
                )
        gate_value = source.get("license_gate")
        license_gate = (
            None
            if gate_value is None
            else _logical_name(gate_value, f"{location}.license_gate")
        )
        digest = None
        if source_sha256 is not None:
            digest = _sha256(source_sha256, f"{location} source SHA-256")
        return cls(
            id=_slug(source["id"], f"{location}.id"),
            version=_semver(source["version"], f"{location}.version"),
            display_name=_string(source["display_name"], f"{location}.display_name"),
            compatibility=CompatibilityV1.from_dict(
                source["compatibility"], f"{location}.compatibility"
            ),
            node_roots=node_roots,
            dependencies=dependencies,
            patches=patches,
            environments=environments,
            hooks=hooks,
            runtime_env=_string_mapping(
                source.get("runtime_env", {}), f"{location}.runtime_env"
            ),
            workflows=workflows,
            probes=probes,
            health_node_ids=health_ids,
            health_command=health_command,
            licenses=licenses,
            cache_profiles=cache_profiles,
            accelerators=accelerators,
            validation_commands=validation_commands,
            readiness_namespace=readiness_namespace,
            readiness_fields=readiness_fields,
            license_gate=license_gate,
            source_sha256=digest,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "id": self.id,
            "version": self.version,
            "display_name": self.display_name,
            "compatibility": self.compatibility.to_dict(),
            "node_roots": [item.to_dict() for item in self.node_roots],
            "dependencies": [item.to_dict() for item in self.dependencies],
            "patches": [item.to_dict() for item in self.patches],
            "environments": [item.to_dict() for item in self.environments],
            "hooks": {name: hook.to_dict() for name, hook in sorted(self.hooks.items())},
            "runtime_env": dict(sorted(self.runtime_env.items())),
            "workflows": list(self.workflows),
            "probes": [item.to_dict() for item in self.probes],
            "health_checks": {
                "node_ids": list(self.health_node_ids),
                **(
                    {"command": list(self.health_command)}
                    if self.health_command
                    else {}
                ),
            },
            "licenses": [item.to_dict() for item in self.licenses],
            **(
                {"cache_profiles": list(self.cache_profiles)}
                if self.cache_profiles
                else {}
            ),
            **(
                {"accelerators": list(self.accelerators)}
                if self.accelerators
                else {}
            ),
            **(
                {
                    "validation_commands": [
                        list(command) for command in self.validation_commands
                    ]
                }
                if self.validation_commands
                else {}
            ),
            **(
                {
                    "readiness": {
                        "namespace": self.readiness_namespace,
                        "fields": list(self.readiness_fields),
                    }
                }
                if self.readiness_namespace is not None
                else {}
            ),
            **(
                {"license_gate": self.license_gate}
                if self.license_gate is not None
                else {}
            ),
        }

    @property
    def canonical_sha256(self) -> str:
        return canonical_sha256(self.to_dict())


def declared_pack_paths(manifest: PackManifestV1) -> tuple[str, ...]:
    """Return pack-checkout paths that must be containment-checked by a loader."""

    values = [item.source for item in manifest.node_roots]
    values.extend(item.specification for item in manifest.patches)
    values.extend(item.path for item in manifest.hooks.values())
    values.extend(manifest.workflows)
    values.extend(item.notice for item in manifest.licenses if item.notice is not None)
    values.extend(manifest.cache_profiles)
    values.extend(
        item.path
        for item in manifest.probes
        if item.path is not None and item.target is None
    )
    values.extend(
        item.requirements_file
        for item in manifest.dependencies
        if isinstance(item, GitDependencyV1)
        and item.requirements_source == "pack"
        and item.requirements_file is not None
    )
    return tuple(values)


def resolve_declared_path(root: Path, relative_path: str) -> Path:
    """Resolve *relative_path* and reject symlink/path escapes from *root*."""

    resolved_root = root.resolve()
    candidate = (root / relative_path).resolve()
    if candidate != resolved_root and resolved_root not in candidate.parents:
        raise PackSchemaError(f"Declared path escapes the pack checkout: {relative_path}.")
    return candidate
