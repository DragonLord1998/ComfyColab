from __future__ import annotations

import hashlib
import unittest
from typing import Any

from comfycolab.packs import (
    ComfyColabLockV1,
    PackConflictError,
    PackManifestV1,
    PackRefV1,
    PackSchemaError,
    merge_python_specifiers,
    resolve_lock,
)


COMFYUI_COMMIT = "8b099de36acd81acd1afa3b5442951dc847e0a52"
CORE_COMMIT = "1111111111111111111111111111111111111111"
PACK_COMMIT = "2222222222222222222222222222222222222222"
DEPENDENCY_COMMIT = "3333333333333333333333333333333333333333"


def manifest(
    pack_id: str,
    *,
    compatible_ref: str = COMFYUI_COMMIT,
    dependencies: list[dict[str, Any]] | None = None,
    environments: list[dict[str, Any]] | None = None,
    patches: list[dict[str, Any]] | None = None,
    runtime_env: dict[str, str] | None = None,
    node_roots: list[dict[str, str]] | None = None,
    license_gate: str | None = None,
) -> PackManifestV1:
    digest = hashlib.sha256(pack_id.encode("utf-8")).hexdigest()
    return PackManifestV1.from_dict(
        {
            "schema": 1,
            "id": pack_id,
            "version": "1.0.0",
            "display_name": pack_id.title(),
            "compatibility": {
                "core_manifest_api": 1,
                "comfyui": {
                    "compatible_refs": [compatible_ref],
                    "tested_refs": [compatible_ref],
                },
            },
            "node_roots": node_roots or [],
            "dependencies": dependencies or [],
            "patches": patches or [],
            "environments": environments or [],
            "hooks": {},
            "runtime_env": runtime_env or {},
            "workflows": [],
            "probes": [],
            "health_checks": {"node_ids": []},
            "licenses": [],
            **({"license_gate": license_gate} if license_gate is not None else {}),
        },
        source_sha256=digest,
    )


def pack_ref(value: PackManifestV1) -> PackRefV1:
    assert value.source_sha256 is not None
    return PackRefV1.from_dict(
        {
            "schema": 1,
            "id": value.id,
            "repository": f"https://github.com/example/{value.id}.git",
            "ref": PACK_COMMIT,
            "manifest_sha256": value.source_sha256,
        }
    )


def git_dependency(
    identifier: str = "shared-node",
    *,
    ref: str = DEPENDENCY_COMMIT,
    destination: str = "custom_nodes/SharedNode",
    scope: str = "comfyui",
    requirements_file: str | None = None,
    requirements_source: str | None = None,
) -> dict[str, str]:
    dependency = {
        "kind": "git",
        "id": identifier,
        "repository": f"https://github.com/example/{identifier}.git",
        "ref": ref,
        "destination": destination,
        "scope": scope,
    }
    if requirements_file is not None:
        dependency["requirements_file"] = requirements_file
    if requirements_source is not None:
        dependency["requirements_source"] = requirements_source
    return dependency


def resolved(*values: PackManifestV1) -> ComfyColabLockV1:
    return resolve_lock(
        core_repository="https://github.com/example/ComfyColab.git",
        core_commit=CORE_COMMIT,
        core_version="0.3.0",
        comfyui_repository="https://github.com/comfyanonymous/ComfyUI.git",
        comfyui_commit=COMFYUI_COMMIT,
        packs=[(pack_ref(value), value) for value in values],
    )


class PackResolverV1Tests(unittest.TestCase):
    def test_lock_bytes_are_deterministic_and_have_no_volatile_metadata(self) -> None:
        shared = git_dependency()
        image = manifest("image", dependencies=[shared], runtime_env={"MODE": "safe"})
        video = manifest("video", dependencies=[shared], runtime_env={"MODE": "safe"})

        first = resolved(image, video)
        second = resolved(video, image)
        self.assertEqual(first.canonical_bytes(), second.canonical_bytes())
        self.assertEqual(first.sha256, second.sha256)
        self.assertNotIn(b"timestamp", first.canonical_bytes())
        dependency = first.to_dict()["dependencies"][0]
        self.assertEqual(dependency["requested_by"], ["image", "video"])

    def test_git_identity_and_destination_conflicts_fail_before_install(self) -> None:
        image = manifest("image", dependencies=[git_dependency()])
        changed_ref = manifest(
            "video",
            dependencies=[git_dependency(ref="4" * 40)],
        )
        with self.assertRaisesRegex(PackConflictError, "Dependency 'shared-node'"):
            resolved(image, changed_ref)

        changed_id = manifest(
            "video",
            dependencies=[
                git_dependency(
                    "other-node",
                    destination="custom_nodes/SharedNode",
                )
            ],
        )
        with self.assertRaisesRegex(PackConflictError, "both target"):
            resolved(image, changed_id)

        with self.assertRaisesRegex(PackConflictError, "both target"):
            resolved(
                manifest(
                    "image",
                    dependencies=[
                        git_dependency(
                            "image-source",
                            destination="sources/shared",
                            scope="image-environment",
                        )
                    ],
                ),
                manifest(
                    "video",
                    dependencies=[
                        git_dependency(
                            "video-source",
                            destination="sources/shared",
                            scope="video-environment",
                        )
                    ],
                ),
            )

    def test_pack_sourced_requirements_cannot_have_multiple_owners(self) -> None:
        shared = git_dependency(
            requirements_file="requirements/shared.txt",
            requirements_source="pack",
        )
        with self.assertRaisesRegex(PackConflictError, "pack-sourced requirements"):
            resolved(
                manifest("image", dependencies=[shared]),
                manifest("video", dependencies=[shared]),
            )

        lock = resolved(manifest("image", dependencies=[shared]))
        payload = lock.to_dict()
        payload["dependencies"][0]["requested_by"] = ["image", "video"]
        with self.assertRaisesRegex(PackSchemaError, "exactly one owner"):
            ComfyColabLockV1.from_dict(payload)

    def test_pack_license_gate_survives_resolution_and_lock_validation(self) -> None:
        lock = resolved(
            manifest("image", license_gate="accept_image_terms")
        )
        pack = lock.to_dict()["packs"][0]
        self.assertEqual(pack["license_gate"], "accept_image_terms")
        self.assertEqual(
            ComfyColabLockV1.from_bytes(lock.canonical_bytes()).to_dict()["packs"][0][
                "license_gate"
            ],
            "accept_image_terms",
        )

    def test_comfyui_dependency_id_is_reserved_for_engine_patches(self) -> None:
        value = manifest(
            "image",
            dependencies=[git_dependency("comfyui")],
        )
        with self.assertRaisesRegex(PackConflictError, "reserved"):
            resolved(value)

    def test_node_root_collision_is_detected_independent_of_pack_order(self) -> None:
        first = manifest(
            "aaa",
            dependencies=[
                git_dependency(destination="custom_nodes/OwnedByPack")
            ],
        )
        second = manifest(
            "zzz",
            node_roots=[
                {
                    "source": "custom_nodes/OwnedByPack",
                    "target": "OwnedByPack",
                }
            ],
        )
        with self.assertRaisesRegex(PackConflictError, "collides with pack node root"):
            resolved(first, second)

    def test_python_specifiers_merge_or_report_empty_intersections(self) -> None:
        self.assertEqual(
            merge_python_specifiers(">=2.1,<3", ">=2.5,!=2.7"),
            "!=2.7,<3,>=2.1,>=2.5",
        )
        with self.assertRaisesRegex(PackConflictError, "empty intersection"):
            merge_python_specifiers(">=3", "<3")

        image = manifest(
            "image",
            environments=[
                {
                    "id": "main",
                    "kind": "main",
                    "python_requirements": [
                        {"name": "Torch", "specifier": ">=2.1,<3"}
                    ],
                }
            ],
        )
        video = manifest(
            "video",
            environments=[
                {
                    "id": "main",
                    "kind": "main",
                    "python_requirements": [
                        {"name": "torch", "specifier": ">=2.5"}
                    ],
                }
            ],
        )
        requirement = resolved(image, video).to_dict()["environments"][0][
            "python_requirements"
        ][0]
        self.assertEqual(requirement["requested_by"], ["image", "video"])
        self.assertEqual(requirement["specifier"], "<3,>=2.1,>=2.5")

    def test_runtime_system_and_environment_variable_conflicts_are_global(self) -> None:
        environment = lambda identifier, version: [
            {
                "id": identifier,
                "kind": "isolated",
                "scope": "isolated",
                "system_dependencies": [
                    {
                        "manager": "apt",
                        "name": "ffmpeg",
                        "version": version,
                        "scope": "runtime",
                    }
                ],
            }
        ]
        with self.assertRaisesRegex(PackConflictError, "Runtime system package"):
            resolved(
                manifest("image", environments=environment("image-worker", "6")),
                manifest("video", environments=environment("video-worker", "7")),
            )
        with self.assertRaisesRegex(PackConflictError, "environment variable"):
            resolved(
                manifest("image", runtime_env={"ATTENTION": "sdpa"}),
                manifest("video", runtime_env={"ATTENTION": "flash"}),
            )

    def test_isolated_environment_ids_are_globally_unique(self) -> None:
        isolated = [
            {
                "id": "worker",
                "kind": "isolated",
                "scope": "isolated",
            }
        ]
        with self.assertRaisesRegex(PackConflictError, "globally unique"):
            resolved(
                manifest("image", environments=isolated),
                manifest("video", environments=isolated),
            )

        lock = resolved(
            manifest(
                "image",
                environments=[
                    {"id": "image-worker", "kind": "isolated", "scope": "isolated"}
                ],
            ),
            manifest(
                "video",
                environments=[
                    {"id": "video-worker", "kind": "isolated", "scope": "isolated"}
                ],
            ),
        )
        self.assertEqual(
            [entry["id"] for entry in lock.to_dict()["environments"]],
            ["image-worker", "video-worker"],
        )
        payload = lock.to_dict()
        payload["environments"][1]["id"] = "image-worker"
        with self.assertRaisesRegex(PackSchemaError, "Duplicate.*environment ID"):
            ComfyColabLockV1.from_dict(payload)

    def test_patch_overlap_requires_a_hash_contiguous_chain(self) -> None:
        dependency = git_dependency(
            "source",
            destination="sources/source",
            scope="isolated",
        )
        base = {
            "target": "source",
            "target_ref": DEPENDENCY_COMMIT,
            "specification_sha256": "9" * 64,
            "order": 0,
            "composition": [],
        }
        first = {
            **base,
            "id": "patch-one",
            "specification": "patch-one.json",
            "files": [
                {
                    "path": "source.py",
                    "before_sha256": "a" * 64,
                    "after_sha256": "b" * 64,
                }
            ],
        }
        second = {
            **base,
            "id": "patch-two",
            "specification": "patch-two.json",
            "files": [
                {
                    "path": "source.py",
                    "before_sha256": "b" * 64,
                    "after_sha256": "c" * 64,
                }
            ],
        }
        chained = manifest(
            "three",
            dependencies=[dependency],
            patches=[first, second],
        )
        self.assertEqual(len(resolved(chained).to_dict()["patches"]), 2)

        broken = {
            **second,
            "files": [
                {
                    "path": "source.py",
                    "before_sha256": "d" * 64,
                    "after_sha256": "e" * 64,
                }
            ],
        }
        with self.assertRaisesRegex(PackConflictError, "not hash-contiguous"):
            resolved(
                manifest(
                    "three",
                    dependencies=[dependency],
                    patches=[first, broken],
                )
            )

    def test_cross_pack_patch_overlap_requires_declared_composition(self) -> None:
        dependency = git_dependency(
            "source",
            destination="sources/source",
            scope="isolated",
        )
        common = {
            "target": "source",
            "target_ref": DEPENDENCY_COMMIT,
            "specification_sha256": "8" * 64,
            "order": 0,
            "composition": [],
        }
        first = {
            **common,
            "id": "patch-a",
            "specification": "patch-a.json",
            "files": [
                {
                    "path": "shared.py",
                    "before_sha256": "1" * 64,
                    "after_sha256": "2" * 64,
                }
            ],
        }
        second = {
            **common,
            "id": "patch-b",
            "specification": "patch-b.json",
            "files": [
                {
                    "path": "shared.py",
                    "before_sha256": "2" * 64,
                    "after_sha256": "3" * 64,
                }
            ],
        }
        with self.assertRaisesRegex(PackConflictError, "without one declared composition"):
            resolved(
                manifest("aaa", dependencies=[dependency], patches=[first]),
                manifest("bbb", dependencies=[dependency], patches=[second]),
            )

        composition = ["patch-a", "patch-b"]
        first["composition"] = composition
        second["composition"] = composition
        self.assertEqual(
            len(
                resolved(
                    manifest("aaa", dependencies=[dependency], patches=[first]),
                    manifest("bbb", dependencies=[dependency], patches=[second]),
                ).to_dict()["patches"]
            ),
            2,
        )

    def test_incompatible_comfyui_and_volatile_lock_fields_are_rejected(self) -> None:
        with self.assertRaisesRegex(PackConflictError, "incompatible"):
            resolved(manifest("image", compatible_ref="f" * 40))

        payload = resolved(manifest("image")).to_dict()
        payload["generated_at"] = "2026-07-18T00:00:00Z"
        with self.assertRaisesRegex(PackSchemaError, "unknown fields"):
            ComfyColabLockV1.from_dict(payload)

    def test_lock_bytes_reject_duplicate_keys_and_repository_credentials(self) -> None:
        with self.assertRaisesRegex(PackSchemaError, "duplicate key"):
            ComfyColabLockV1.from_bytes(b'{"schema":1,"schema":1}')

        payload = resolved(manifest("image")).to_dict()
        payload["core"]["repository"] = (
            "https://secret@example.com/core.git?token=unsafe"
        )
        with self.assertRaisesRegex(PackSchemaError, "without credentials"):
            ComfyColabLockV1.from_dict(payload)


if __name__ == "__main__":
    unittest.main()
