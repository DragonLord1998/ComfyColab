"""Deterministic pack composition and pre-install conflict detection."""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Iterable

from .errors import PackConflictError
from .lock import ComfyColabLockV1
from .schema import (
    SCHEMA_VERSION,
    ArtifactDependencyV1,
    EnvironmentV1,
    GitDependencyV1,
    HuggingFaceDependencyV1,
    PackManifestV1,
    PackRefV1,
    PatchV1,
    PythonRequirementV1,
    SystemDependencyV1,
)


ResolvedPackInput = tuple[PackRefV1, PackManifestV1]
Dependency = GitDependencyV1 | HuggingFaceDependencyV1 | ArtifactDependencyV1

_SPECIFIER = re.compile(r"^(==|!=|~=|<=|>=|<|>)(.+)$")


def _release(value: str) -> tuple[int, ...]:
    base = re.split(r"[A-Za-z+_-]", value, maxsplit=1)[0]
    if base.endswith(".*"):
        base = base[:-2]
    try:
        return tuple(int(part) for part in base.split("."))
    except ValueError as error:
        raise PackConflictError(f"Unsupported Python version in specifier: {value!r}.") from error


def _compare(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    width = max(len(left), len(right))
    normalized_left = left + (0,) * (width - len(left))
    normalized_right = right + (0,) * (width - len(right))
    return (normalized_left > normalized_right) - (normalized_left < normalized_right)


def _compatible_release_upper(value: str) -> tuple[int, ...]:
    release = _release(value)
    if len(release) < 2:
        raise PackConflictError(f"Compatible-release specifier is too broad: ~={value}.")
    prefix = list(release[:-1] if len(release) > 2 else release[:1])
    prefix[-1] += 1
    return tuple(prefix)


def _specifier_tokens(value: str) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(item for item in value.split(",") if item)


def merge_python_specifiers(left: str, right: str) -> str:
    """Return a stable conjunction or raise when the supported intersection is empty."""

    tokens = tuple(sorted(set((*_specifier_tokens(left), *_specifier_tokens(right)))))
    exact: str | None = None
    lower: tuple[tuple[int, ...], bool] | None = None
    upper: tuple[tuple[int, ...], bool] | None = None
    excluded: set[str] = set()

    def add_lower(version: tuple[int, ...], inclusive: bool) -> None:
        nonlocal lower
        if lower is None:
            lower = (version, inclusive)
            return
        comparison = _compare(version, lower[0])
        if comparison > 0 or (comparison == 0 and not inclusive):
            lower = (version, inclusive)

    def add_upper(version: tuple[int, ...], inclusive: bool) -> None:
        nonlocal upper
        if upper is None:
            upper = (version, inclusive)
            return
        comparison = _compare(version, upper[0])
        if comparison < 0 or (comparison == 0 and not inclusive):
            upper = (version, inclusive)

    for token in tokens:
        match = _SPECIFIER.fullmatch(token)
        if match is None:
            raise PackConflictError(f"Unsupported Python specifier: {token!r}.")
        operator, version = match.groups()
        if operator == "==" and version.endswith(".*"):
            base = _release(version)
            add_lower(base, True)
            upper_prefix = list(base)
            upper_prefix[-1] += 1
            add_upper(tuple(upper_prefix), False)
        elif operator == "==":
            if exact is not None and exact != version:
                raise PackConflictError(
                    f"Python exact-version constraints conflict: {exact} vs {version}."
                )
            exact = version
        elif operator == "!=":
            excluded.add(version)
        elif operator == ">=":
            add_lower(_release(version), True)
        elif operator == ">":
            add_lower(_release(version), False)
        elif operator == "<=":
            add_upper(_release(version), True)
        elif operator == "<":
            add_upper(_release(version), False)
        elif operator == "~=":
            add_lower(_release(version), True)
            add_upper(_compatible_release_upper(version), False)

    if exact is not None:
        exact_release = _release(exact)
        if any(
            _compare(exact_release, _release(excluded_version)) == 0
            for excluded_version in excluded
        ):
            raise PackConflictError(f"Python version {exact} is both required and excluded.")
        if lower is not None:
            comparison = _compare(exact_release, lower[0])
            if comparison < 0 or (comparison == 0 and not lower[1]):
                raise PackConflictError(f"Python version {exact} is below the required range.")
        if upper is not None:
            comparison = _compare(exact_release, upper[0])
            if comparison > 0 or (comparison == 0 and not upper[1]):
                raise PackConflictError(f"Python version {exact} is above the required range.")
    elif lower is not None and upper is not None:
        comparison = _compare(lower[0], upper[0])
        if comparison > 0 or (comparison == 0 and (not lower[1] or not upper[1])):
            raise PackConflictError("Python version constraints have an empty intersection.")
        if comparison == 0:
            if any(
                _compare(lower[0], _release(excluded_version)) == 0
                for excluded_version in excluded
            ):
                raise PackConflictError("Python version constraints exclude their only candidate.")
    return ",".join(tokens)


def _manifest_digest(manifest: PackManifestV1) -> str:
    return manifest.source_sha256 or manifest.canonical_sha256


def _dependency_identity(dependency: Dependency) -> tuple[object, ...]:
    value = dependency.to_dict()
    return tuple((key, repr(value[key])) for key in sorted(value))


def _resolve_dependencies(
    packs: tuple[ResolvedPackInput, ...],
) -> tuple[list[dict[str, object]], dict[str, Dependency]]:
    by_id: dict[str, tuple[Dependency, set[str]]] = {}
    by_destination: dict[tuple[str, str], tuple[str, Dependency]] = {}
    node_targets: dict[str, str] = {}

    for _, manifest in packs:
        for node_root in manifest.node_roots:
            destination = f"custom_nodes/{node_root.target}"
            previous_pack = node_targets.get(destination)
            if previous_pack is not None and previous_pack != manifest.id:
                raise PackConflictError(
                    f"Packs {previous_pack!r} and {manifest.id!r} both target {destination!r}."
                )
            node_targets[destination] = manifest.id

        for dependency in manifest.dependencies:
            if dependency.id == "comfyui":
                raise PackConflictError(
                    "Dependency ID 'comfyui' is reserved for the locked ComfyUI checkout."
                )
            previous = by_id.get(dependency.id)
            if previous is None:
                by_id[dependency.id] = (dependency, {manifest.id})
            elif _dependency_identity(previous[0]) != _dependency_identity(dependency):
                raise PackConflictError(
                    f"Dependency {dependency.id!r} conflicts between "
                    f"{sorted(previous[1])} and {manifest.id!r}."
                )
            else:
                if (
                    isinstance(dependency, GitDependencyV1)
                    and dependency.requirements_source == "pack"
                    and manifest.id not in previous[1]
                ):
                    raise PackConflictError(
                        f"Dependency {dependency.id!r} uses pack-sourced requirements "
                        "and cannot be shared by multiple packs in manifest API v1."
                    )
                previous[1].add(manifest.id)

            destination_namespace = (
                "comfyui" if dependency.scope == "comfyui" else "managed"
            )
            destination_key = (destination_namespace, dependency.destination)
            destination_previous = by_destination.get(destination_key)
            if (
                destination_previous is not None
                and destination_previous[0] != dependency.id
            ):
                raise PackConflictError(
                    f"Dependencies {destination_previous[0]!r} and {dependency.id!r} "
                    f"both target {dependency.scope}:{dependency.destination}."
                )
            by_destination[destination_key] = (dependency.id, dependency)
            if (
                dependency.scope == "comfyui"
                and dependency.destination in node_targets
            ):
                raise PackConflictError(
                    f"Dependency {dependency.id!r} collides with pack node root "
                    f"{dependency.destination!r}."
                )

    for (scope, destination), (dependency_id, _) in by_destination.items():
        if scope == "comfyui" and destination in node_targets:
            raise PackConflictError(
                f"Dependency {dependency_id!r} collides with pack node root "
                f"{destination!r}."
            )

    resolved = [
        {
            **dependency.to_dict(),
            "requested_by": sorted(requesters),
        }
        for dependency, requesters in by_id.values()
    ]
    resolved.sort(
        key=lambda item: (
            str(item["scope"]),
            str(item["destination"]),
            str(item["id"]),
        )
    )
    return resolved, {identifier: value[0] for identifier, value in by_id.items()}


def _merge_requirement(
    current: dict[str, object],
    incoming: PythonRequirementV1,
    pack_id: str,
    environment_name: str,
) -> None:
    current_url = current.get("url")
    if current_url != incoming.url:
        raise PackConflictError(
            f"Python package {incoming.name!r} uses conflicting sources in "
            f"environment {environment_name!r}."
        )
    if current.get("sha256") != incoming.sha256:
        raise PackConflictError(
            f"Python package {incoming.name!r} uses conflicting hashes in "
            f"environment {environment_name!r}."
        )
    try:
        current["specifier"] = merge_python_specifiers(
            str(current["specifier"]), incoming.specifier
        )
    except PackConflictError as error:
        raise PackConflictError(
            f"Python package {incoming.name!r} conflicts in environment "
            f"{environment_name!r}: {error}"
        ) from error
    requesters = current["requested_by"]
    assert isinstance(requesters, set)
    requesters.add(pack_id)


def _merge_system_dependency(
    current: dict[str, object],
    incoming: SystemDependencyV1,
    pack_id: str,
    environment_name: str,
) -> None:
    current_version = current.get("version")
    if (
        current_version is not None
        and incoming.version is not None
        and current_version != incoming.version
    ):
        raise PackConflictError(
            f"System package {incoming.manager}:{incoming.name} has conflicting "
            f"versions in {environment_name!r}: {current_version} vs {incoming.version}."
        )
    if current_version is None and incoming.version is not None:
        current["version"] = incoming.version
    requesters = current["requested_by"]
    assert isinstance(requesters, set)
    requesters.add(pack_id)


def _resolve_environments(
    packs: tuple[ResolvedPackInput, ...],
) -> list[dict[str, object]]:
    environments: dict[tuple[str, ...], dict[str, object]] = {}
    environment_id_owners: dict[str, str] = {}
    runtime_systems: dict[tuple[str, str], tuple[str | None, set[str]]] = {}

    for _, manifest in packs:
        for environment in manifest.environments:
            effective_owner = "main" if environment.kind == "main" else manifest.id
            previous_owner = environment_id_owners.get(environment.id)
            if previous_owner is not None and previous_owner != effective_owner:
                raise PackConflictError(
                    f"Environment ID {environment.id!r} collides between "
                    f"{previous_owner!r} and {effective_owner!r}; manifest API v1 "
                    "requires globally unique isolated environment IDs."
                )
            environment_id_owners[environment.id] = effective_owner
            key = (
                ("main",)
                if environment.kind == "main"
                else ("isolated", manifest.id, environment.id)
            )
            current = environments.get(key)
            if current is None:
                current = {
                    "id": environment.id,
                    "kind": environment.kind,
                    "scope": environment.scope,
                    "owner": None if environment.kind == "main" else manifest.id,
                    "python": environment.python,
                    "cache_profile": environment.cache_profile,
                    "license_gate": environment.license_gate,
                    "python_requirements": {},
                    "system_dependencies": {},
                    "requested_by": {manifest.id},
                }
                environments[key] = current
            else:
                requesters = current["requested_by"]
                assert isinstance(requesters, set)
                requesters.add(manifest.id)
                for field, incoming in (
                    ("scope", environment.scope),
                    ("python", environment.python),
                    ("cache_profile", environment.cache_profile),
                    ("license_gate", environment.license_gate),
                ):
                    existing = current.get(field)
                    if existing is not None and incoming is not None and existing != incoming:
                        raise PackConflictError(
                            f"Environment {environment.id!r} has conflicting {field}: "
                            f"{existing!r} vs {incoming!r}."
                        )
                    if existing is None:
                        current[field] = incoming

            requirements = current["python_requirements"]
            assert isinstance(requirements, dict)
            for requirement in environment.python_requirements:
                existing = requirements.get(requirement.name)
                if existing is None:
                    existing = {
                        **requirement.to_dict(),
                        "requested_by": {manifest.id},
                    }
                    requirements[requirement.name] = existing
                else:
                    _merge_requirement(existing, requirement, manifest.id, environment.id)

            systems = current["system_dependencies"]
            assert isinstance(systems, dict)
            for dependency in environment.system_dependencies:
                system_key = (dependency.manager, dependency.name, dependency.scope)
                existing = systems.get(system_key)
                if existing is None:
                    existing = {
                        **dependency.to_dict(),
                        "requested_by": {manifest.id},
                    }
                    systems[system_key] = existing
                else:
                    _merge_system_dependency(
                        existing, dependency, manifest.id, environment.id
                    )
                if dependency.scope == "runtime":
                    global_key = (dependency.manager, dependency.name)
                    previous = runtime_systems.get(global_key)
                    if previous is None:
                        runtime_systems[global_key] = (
                            dependency.version,
                            {manifest.id},
                        )
                    else:
                        previous_version, requesters = previous
                        if (
                            previous_version is not None
                            and dependency.version is not None
                            and previous_version != dependency.version
                        ):
                            raise PackConflictError(
                                f"Runtime system package {dependency.manager}:{dependency.name} "
                                f"conflicts between {sorted(requesters)} and {manifest.id!r}."
                            )
                        requesters.add(manifest.id)
                        runtime_systems[global_key] = (
                            previous_version or dependency.version,
                            requesters,
                        )

    output: list[dict[str, object]] = []
    for current in environments.values():
        requirements = current.pop("python_requirements")
        systems = current.pop("system_dependencies")
        requesters = current["requested_by"]
        assert isinstance(requirements, dict)
        assert isinstance(systems, dict)
        assert isinstance(requesters, set)
        entry = {
            key: value
            for key, value in current.items()
            if value is not None and key != "requested_by"
        }
        entry["python_requirements"] = [
            {
                **{
                    key: value
                    for key, value in requirement.items()
                    if key != "requested_by"
                },
                "requested_by": sorted(requirement["requested_by"]),
            }
            for requirement in sorted(
                requirements.values(), key=lambda item: str(item["name"])
            )
        ]
        entry["system_dependencies"] = [
            {
                **{
                    key: value
                    for key, value in dependency.items()
                    if key != "requested_by" and value is not None
                },
                "requested_by": sorted(dependency["requested_by"]),
            }
            for dependency in sorted(
                systems.values(),
                key=lambda item: (
                    str(item["manager"]),
                    str(item["name"]),
                    str(item["scope"]),
                ),
            )
        ]
        entry["requested_by"] = sorted(requesters)
        output.append(entry)
    output.sort(
        key=lambda item: (
            str(item["kind"]),
            str(item.get("owner", "")),
            str(item["id"]),
        )
    )
    return output


def _resolve_runtime_env(
    packs: tuple[ResolvedPackInput, ...],
) -> list[dict[str, object]]:
    values: dict[str, tuple[str, set[str]]] = {}
    for _, manifest in packs:
        for name, value in manifest.runtime_env.items():
            previous = values.get(name)
            if previous is None:
                values[name] = (value, {manifest.id})
            elif previous[0] != value:
                raise PackConflictError(
                    f"Runtime environment variable {name!r} conflicts between "
                    f"{sorted(previous[1])} and {manifest.id!r}."
                )
            else:
                previous[1].add(manifest.id)
    return [
        {"name": name, "value": value, "requested_by": sorted(requesters)}
        for name, (value, requesters) in sorted(values.items())
    ]


def _resolve_patches(
    packs: tuple[ResolvedPackInput, ...],
    dependencies: dict[str, Dependency],
    comfyui_commit: str,
) -> list[dict[str, object]]:
    by_id: dict[str, tuple[PatchV1, set[str]]] = {}
    declaration_order: dict[tuple[str, str], int] = {}

    for _, manifest in packs:
        for index, declared_patch in enumerate(manifest.patches):
            patch = declared_patch
            if patch.composition:
                if patch.id not in patch.composition:
                    raise PackConflictError(
                        f"Patch {patch.id!r} is omitted from its declared composition."
                    )
                composition_order = patch.composition.index(patch.id)
                if patch.order not in {0, composition_order}:
                    raise PackConflictError(
                        f"Patch {patch.id!r} order {patch.order} disagrees with its "
                        f"composition position {composition_order}."
                    )
                patch = replace(patch, order=composition_order)
            elif patch.order == 0 and index > 0:
                patch = replace(patch, order=index)
            if patch.target == "comfyui":
                expected_ref = comfyui_commit
            else:
                dependency = dependencies.get(patch.target)
                if dependency is None:
                    raise PackConflictError(
                        f"Patch {patch.id!r} targets undeclared dependency {patch.target!r}."
                    )
                if not isinstance(dependency, GitDependencyV1):
                    raise PackConflictError(
                        f"Patch {patch.id!r} target {patch.target!r} is not a Git dependency."
                    )
                expected_ref = dependency.ref
            if patch.target_ref != expected_ref:
                raise PackConflictError(
                    f"Patch {patch.id!r} expects {patch.target_ref}, but target "
                    f"{patch.target!r} resolves to {expected_ref}."
                )
            if patch.specification_sha256 is None or not patch.files:
                raise PackConflictError(
                    f"Patch {patch.id!r} lacks authenticated file metadata; "
                    "load it from a pack checkout before resolution."
                )
            previous = by_id.get(patch.id)
            if previous is None:
                by_id[patch.id] = (patch, {manifest.id})
                declaration_order[(manifest.id, patch.id)] = index
            elif previous[0] != patch:
                raise PackConflictError(
                    f"Patch ID {patch.id!r} has conflicting definitions."
                )
            else:
                previous[1].add(manifest.id)

    file_groups: dict[tuple[str, str, str], list[tuple[PatchV1, set[str]]]] = {}
    for patch, requesters in by_id.values():
        for file in patch.files:
            file_groups.setdefault(
                (patch.target, patch.target_ref, file.path), []
            ).append((patch, requesters))

    for (target, _, path), declarations in file_groups.items():
        if len(declarations) < 2:
            continue
        all_requesters = set().union(*(requesters for _, requesters in declarations))
        if len(all_requesters) == 1:
            owner = next(iter(all_requesters))
            ordered = sorted(
                declarations,
                key=lambda item: declaration_order[(owner, item[0].id)],
            )
        else:
            compositions = {patch.composition for patch, _ in declarations}
            if len(compositions) != 1 or not next(iter(compositions)):
                raise PackConflictError(
                    f"Cross-pack patches overlap at {target}:{path} without one "
                    "declared composition."
                )
            composition = next(iter(compositions))
            group_ids = {patch.id for patch, _ in declarations}
            if not group_ids.issubset(composition):
                raise PackConflictError(
                    f"Patch composition for {target}:{path} omits an overlapping patch."
                )
            ordered = sorted(
                declarations,
                key=lambda item: composition.index(item[0].id),
            )
        previous_after: str | None = None
        for patch, _ in ordered:
            file = next(item for item in patch.files if item.path == path)
            if previous_after is not None and file.before_sha256 != previous_after:
                raise PackConflictError(
                    f"Patch chain for {target}:{path} is not hash-contiguous at "
                    f"{patch.id!r}."
                )
            previous_after = file.after_sha256

    output = [
        {**patch.to_dict(), "requested_by": sorted(requesters)}
        for patch, requesters in by_id.values()
    ]
    output.sort(
        key=lambda item: (
            str(item["target"]),
            int(item["order"]),
            str(item["id"]),
        )
    )
    return output


def resolve_lock(
    *,
    core_repository: str,
    core_commit: str,
    core_version: str,
    comfyui_repository: str,
    comfyui_commit: str,
    packs: Iterable[ResolvedPackInput],
) -> ComfyColabLockV1:
    """Resolve validated manifests into an immutable, conflict-safe lock."""

    ordered = tuple(sorted(packs, key=lambda item: item[0].id))
    seen: dict[str, PackRefV1] = {}
    locked_packs: list[dict[str, object]] = []
    for pack_ref, manifest in ordered:
        if pack_ref.id != manifest.id:
            raise PackConflictError(
                f"Pack ref {pack_ref.id!r} points to manifest {manifest.id!r}."
            )
        digest = _manifest_digest(manifest)
        if pack_ref.manifest_sha256 != digest:
            raise PackConflictError(
                f"Manifest digest mismatch for pack {pack_ref.id!r}: "
                f"expected {pack_ref.manifest_sha256}, got {digest}."
            )
        previous = seen.get(pack_ref.id)
        if previous is not None:
            if previous != pack_ref:
                raise PackConflictError(
                    f"Conflicting pack references selected for {pack_ref.id!r}."
                )
            raise PackConflictError(f"Pack {pack_ref.id!r} is selected more than once.")
        seen[pack_ref.id] = pack_ref
        if comfyui_commit not in manifest.compatibility.compatible_refs:
            raise PackConflictError(
                f"Pack {manifest.id!r} is incompatible with ComfyUI {comfyui_commit}."
            )
        locked_packs.append(
            {
                "id": manifest.id,
                "version": manifest.version,
                "repository": pack_ref.repository,
                "commit": pack_ref.ref,
                "manifest_sha256": pack_ref.manifest_sha256,
                "node_roots": [item.to_dict() for item in manifest.node_roots],
                **(
                    {"license_gate": manifest.license_gate}
                    if manifest.license_gate is not None
                    else {}
                ),
            }
        )

    dependencies, dependency_index = _resolve_dependencies(ordered)
    environments = _resolve_environments(ordered)
    patches = _resolve_patches(ordered, dependency_index, comfyui_commit)
    runtime_env = _resolve_runtime_env(ordered)

    return ComfyColabLockV1.from_dict(
        {
            "schema": SCHEMA_VERSION,
            "core": {
                "version": core_version,
                "repository": core_repository,
                "commit": core_commit,
            },
            "comfyui": {
                "repository": comfyui_repository,
                "commit": comfyui_commit,
            },
            "packs": locked_packs,
            "dependencies": dependencies,
            "patches": patches,
            "environments": environments,
            "runtime_env": runtime_env,
        }
    )
