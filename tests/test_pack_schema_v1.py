from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

from comfycolab.packs import (
    PackIntegrityError,
    PackManifestV1,
    PackRefV1,
    PackSchemaError,
    load_pack_manifest,
    load_profile,
    load_registry,
    resolve_lock,
    safe_load_json,
)


COMFYUI_COMMIT = "8b099de36acd81acd1afa3b5442951dc847e0a52"
PACK_COMMIT = "0123456789abcdef0123456789abcdef01234567"


def manifest_payload(pack_id: str = "example") -> dict[str, object]:
    return {
        "schema": 1,
        "id": pack_id,
        "version": "1.2.3",
        "display_name": f"ComfyColab {pack_id.title()}",
        "compatibility": {
            "core_manifest_api": 1,
            "comfyui": {
                "compatible_refs": [COMFYUI_COMMIT],
                "tested_refs": [COMFYUI_COMMIT],
            },
        },
        "node_roots": [],
        "dependencies": [],
        "patches": [],
        "environments": [],
        "hooks": {},
        "workflows": [],
        "probes": [],
        "health_checks": {"node_ids": []},
        "licenses": [],
    }


class PackSchemaV1Tests(unittest.TestCase):
    def test_pack_ref_requires_an_immutable_commit_and_strict_https_url(self) -> None:
        valid = {
            "schema": 1,
            "id": "image",
            "repository": "https://github.com/example/image.git",
            "ref": PACK_COMMIT,
            "manifest_sha256": "a" * 64,
        }
        self.assertEqual(PackRefV1.from_dict(valid).to_dict(), valid)

        for invalid_ref in ("main", "v1.0.0", PACK_COMMIT.upper()):
            with self.subTest(ref=invalid_ref):
                with self.assertRaises(PackSchemaError):
                    PackRefV1.from_dict({**valid, "ref": invalid_ref})
        with self.assertRaises(PackSchemaError):
            PackRefV1.from_dict(
                {
                    **valid,
                    "repository": "https://token@example.com/image.git?branch=main",
                }
            )
        with self.assertRaises(PackSchemaError):
            PackRefV1.from_dict({**valid, "unexpected": True})

    def test_manifest_accepts_declarative_dependency_extensions(self) -> None:
        payload = manifest_payload()
        payload["dependencies"] = [
            {
                "kind": "git",
                "id": "source",
                "repository": "https://github.com/example/source.git",
                "ref": PACK_COMMIT,
                "destination": "sources/source",
                "scope": "isolated",
                "install_phase": "bootstrap",
                "requirements_file": "requirements/worker.txt",
                "requirements_source": "pack",
                "requirements_format": "requirements.txt",
            },
            {
                "kind": "huggingface",
                "id": "model",
                "repository": "example/model",
                "ref": PACK_COMMIT,
                "destination": "models/example",
                "scope": "isolated",
                "install_phase": "lazy",
                "artifacts": [
                    {
                        "path": "model.safetensors",
                        "bytes": 42,
                        "sha256": "b" * 64,
                    }
                ],
            },
            {
                "kind": "artifact",
                "id": "checkpoint",
                "url": "https://example.com/checkpoint.bin",
                "sha256": "c" * 64,
                "destination": "models/example/checkpoint.bin",
                "scope": "isolated",
                "install_phase": "lazy",
            },
        ]
        parsed = PackManifestV1.from_dict(payload)
        self.assertEqual(parsed.dependencies[0].to_dict()["requirements_source"], "pack")
        self.assertEqual(parsed.dependencies[1].to_dict()["artifacts"][0]["bytes"], 42)
        self.assertEqual(parsed.dependencies[2].to_dict()["install_phase"], "lazy")

    def test_file_symbol_probe_is_top_level_and_targets_comfyui(self) -> None:
        payload = manifest_payload()
        payload["probes"] = [
            {
                "phase": "post_clone",
                "type": "file_symbols",
                "target": "comfyui",
                "path": "comfy_extras/nodes_example.py",
                "symbols": ["ExampleNode", "ExamplePreview"],
            }
        ]
        parsed = PackManifestV1.from_dict(payload)
        self.assertEqual(parsed.probes[0].target, "comfyui")
        self.assertEqual(parsed.probes[0].symbols, ("ExampleNode", "ExamplePreview"))

        nested = manifest_payload()
        nested["compatibility"]["comfyui"]["probes"] = []
        with self.assertRaisesRegex(PackSchemaError, "top-level probes"):
            PackManifestV1.from_dict(nested)

    def test_safe_json_rejects_duplicates_nonfinite_values_and_digest_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            duplicate = root / "duplicate.json"
            duplicate.write_text('{"schema":1,"schema":1}', encoding="utf-8")
            with self.assertRaisesRegex(PackSchemaError, "duplicate"):
                safe_load_json(duplicate)

            nonfinite = root / "nonfinite.json"
            nonfinite.write_text('{"value":NaN}', encoding="utf-8")
            with self.assertRaisesRegex(PackSchemaError, "non-finite"):
                safe_load_json(nonfinite)

            valid = root / "valid.json"
            valid.write_text('{"schema":1}', encoding="utf-8")
            with self.assertRaises(PackIntegrityError):
                safe_load_json(valid, expected_sha256="0" * 64)

    def test_manifest_loader_hydrates_authenticated_patch_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            patch_spec = {
                "schema": 1,
                "patch_id": "fix-source",
                "revision": PACK_COMMIT,
                "files": [
                    {
                        "path": "source.py",
                        "before_sha256": "1" * 64,
                        "after_sha256": "2" * 64,
                    }
                ],
            }
            patch_path = root / "fix-source.json"
            patch_path.write_text(
                json.dumps(patch_spec, separators=(",", ":")),
                encoding="utf-8",
            )
            patch_digest = hashlib.sha256(patch_path.read_bytes()).hexdigest()
            payload = manifest_payload()
            payload["patches"] = [
                {
                    "id": "fix-source",
                    "target": "source",
                    "source_ref": PACK_COMMIT,
                    "path": "fix-source.json",
                    "sha256": patch_digest,
                }
            ]
            manifest_path = root / "comfycolab-pack.json"
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")

            manifest = load_pack_manifest(manifest_path)
            self.assertEqual(manifest.patches[0].specification_sha256, patch_digest)
            self.assertEqual(manifest.patches[0].files[0].path, "source.py")

    def test_registry_and_profile_resolve_aliases_without_floating_refs(self) -> None:
        reference = {
            "schema": 1,
            "id": "image",
            "repository": "https://github.com/example/image.git",
            "ref": PACK_COMMIT,
            "manifest_sha256": "d" * 64,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry_path = root / "registry.json"
            registry_path.write_text(
                json.dumps({"schema": 1, "packs": {"official-image": reference}}),
                encoding="utf-8",
            )
            profile_path = root / "profile.json"
            profile_path.write_text(
                json.dumps(
                    {
                        "schema": 1,
                        "id": "creator",
                        "packs": ["official-image"],
                    }
                ),
                encoding="utf-8",
            )
            registry = load_registry(registry_path)
            profile = load_profile(profile_path, registry=registry)
            self.assertEqual(profile.packs[0].ref, PACK_COMMIT)

    def test_current_daughter_manifests_validate_from_their_checkouts(self) -> None:
        workspace = Path(__file__).resolve().parents[1]
        configured_root = os.environ.get("COMFYCOLAB_DAUGHTER_REPOS_ROOT")
        candidate_roots = (
            (Path(configured_root).expanduser().resolve(),)
            if configured_root
            else (workspace / "daughter-repos", workspace.parent)
        )
        expected = {
            "ComfyColab-3D": "3d",
            "ComfyColab-3DGS": "3dgs",
            "ComfyColab-Image": "image",
            "ComfyColab-Video": "video",
            "ComfyColab-WorldModels": "world",
        }
        if not configured_root and not any(
            (root / repository / "comfycolab-pack.json").is_file()
            for root in candidate_roots
            for repository in expected
        ):
            self.skipTest(
                "daughter checkouts are absent; run the multi-repository integration lane"
            )
        for repository, pack_id in expected.items():
            with self.subTest(repository=repository):
                manifest_path = next(
                    (
                        root / repository / "comfycolab-pack.json"
                        for root in candidate_roots
                        if (root / repository / "comfycolab-pack.json").is_file()
                    ),
                    None,
                )
                self.assertIsNotNone(
                    manifest_path,
                    f"missing staged or sibling checkout for {repository}",
                )
                assert manifest_path is not None
                manifest = load_pack_manifest(manifest_path)
                self.assertEqual(manifest.id, pack_id)

    def test_exact_daughter_manifests_compose_into_one_lock(self) -> None:
        workspace = Path(__file__).resolve().parents[1]
        configured_root = os.environ.get("COMFYCOLAB_DAUGHTER_REPOS_ROOT")
        candidate_roots = (
            (Path(configured_root).expanduser().resolve(),)
            if configured_root
            else (workspace / "daughter-repos", workspace.parent)
        )
        repositories = {
            "3d": "ComfyColab-3D",
            "3dgs": "ComfyColab-3DGS",
            "image": "ComfyColab-Image",
            "video": "ComfyColab-Video",
            "world": "ComfyColab-WorldModels",
        }
        if not configured_root and not any(
            (root / repository / "comfycolab-pack.json").is_file()
            for root in candidate_roots
            for repository in repositories.values()
        ):
            self.skipTest(
                "daughter checkouts are absent; run the multi-repository integration lane"
            )

        registry = load_registry(workspace / "registry" / "published-packs.json")
        engine = json.loads(
            (workspace / "registry" / "engine.json").read_text(encoding="utf-8")
        )["comfyui"]
        packs = []
        for pack_id, repository in repositories.items():
            manifest_path = next(
                (
                    root / repository / "comfycolab-pack.json"
                    for root in candidate_roots
                    if (root / repository / "comfycolab-pack.json").is_file()
                ),
                None,
            )
            self.assertIsNotNone(manifest_path, f"missing checkout for {repository}")
            assert manifest_path is not None
            pack_ref = registry.packs[pack_id]
            manifest = load_pack_manifest(
                manifest_path,
                expected_sha256=pack_ref.manifest_sha256,
            )
            packs.append((pack_ref, manifest))

        lock = resolve_lock(
            core_repository="https://github.com/DragonLord1998/ComfyColab.git",
            core_commit="1" * 40,
            core_version="0.2.0.dev1",
            comfyui_repository=engine["repository"],
            comfyui_commit=engine["commit"],
            packs=packs,
        )
        self.assertEqual(
            [pack["id"] for pack in lock.to_dict()["packs"]],
            ["3d", "3dgs", "image", "video", "world"],
        )


if __name__ == "__main__":
    unittest.main()
