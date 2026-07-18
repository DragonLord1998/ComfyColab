from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "ci_exact_refs.py"
SPEC = importlib.util.spec_from_file_location("ci_exact_refs", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
ci_exact_refs = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ci_exact_refs
SPEC.loader.exec_module(ci_exact_refs)


PACKS = {
    "published-3d": ("3d", "ComfyColab-3D"),
    "published-3dgs": ("3dgs", "ComfyColab-3DGS"),
    "published-image": ("image", "ComfyColab-Image"),
    "published-video": ("video", "ComfyColab-Video"),
    "published-world": ("world", "ComfyColab-WM"),
}


def registry_payload() -> dict[str, object]:
    packs = {}
    for index, (alias, (pack_id, repository)) in enumerate(PACKS.items(), start=1):
        packs[alias] = {
            "schema": 1,
            "id": pack_id,
            "repository": f"https://github.com/example/{repository}.git",
            "ref": f"{index:040x}",
            "manifest_sha256": f"{index:064x}",
        }
    return {"schema": 1, "packs": packs}


def write_registry(root: Path, payload: dict[str, object] | None = None) -> Path:
    path = root / "published-packs.json"
    path.write_text(json.dumps(payload or registry_payload()), encoding="utf-8")
    return path


class ExactRefCITests(unittest.TestCase):
    def test_matrix_uses_registry_commits_and_stable_checkout_directories(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkouts = ci_exact_refs.load_checkouts(
                write_registry(Path(directory))
            )

        matrix = json.loads(ci_exact_refs.matrix(checkouts))["include"]
        by_id = {entry["id"]: entry for entry in matrix}
        self.assertEqual(
            by_id["world"]["directory"],
            "ComfyColab-WorldModels",
        )
        self.assertEqual(
            by_id["world"]["repository"],
            "example/ComfyColab-WM",
        )
        self.assertEqual(by_id["3d"]["install_target"], "./daughter[test]")
        self.assertEqual(by_id["image"]["install_target"], "./daughter")
        self.assertTrue(
            all(len(entry["ref"]) == 40 and entry["ref"] != "main" for entry in matrix)
        )

    def test_registry_must_name_exactly_one_of_each_published_daughter(self) -> None:
        payload = registry_payload()
        payload["packs"].pop("published-world")
        with tempfile.TemporaryDirectory() as directory:
            registry = write_registry(Path(directory), payload)
            with self.assertRaisesRegex(
                ci_exact_refs.ExactRefCIError,
                "missing: world",
            ):
                ci_exact_refs.load_checkouts(registry)

        payload = registry_payload()
        payload["packs"]["second-image"] = payload["packs"]["published-image"]
        with tempfile.TemporaryDirectory() as directory:
            registry = write_registry(Path(directory), payload)
            with self.assertRaisesRegex(
                ci_exact_refs.ExactRefCIError,
                "duplicated",
            ):
                ci_exact_refs.load_checkouts(registry)

    def test_registry_rejects_mutable_refs_and_non_github_repositories(self) -> None:
        payload = registry_payload()
        payload["packs"]["published-image"]["ref"] = "main"
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                ci_exact_refs.ExactRefCIError,
                "lowercase 40-character Git commit",
            ):
                ci_exact_refs.load_checkouts(
                    write_registry(Path(directory), payload)
                )

        payload = registry_payload()
        payload["packs"]["published-image"][
            "repository"
        ] = "https://example.com/example/ComfyColab-Image.git"
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                ci_exact_refs.ExactRefCIError,
                "GitHub HTTPS",
            ):
                ci_exact_refs.load_checkouts(
                    write_registry(Path(directory), payload)
                )

    def test_checkout_verification_authenticates_head_manifest_and_pack_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkout_root = root / "checkout"
            checkout_root.mkdir()
            subprocess.run(["git", "init", "-q"], cwd=checkout_root, check=True)
            subprocess.run(
                ["git", "config", "user.name", "CI Test"],
                cwd=checkout_root,
                check=True,
            )
            subprocess.run(
                ["git", "config", "user.email", "ci@example.invalid"],
                cwd=checkout_root,
                check=True,
            )
            manifest_path = checkout_root / "comfycolab-pack.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema": 1,
                        "id": "world",
                        "version": "1.0.0",
                        "display_name": "World",
                        "compatibility": {
                            "core_manifest_api": 1,
                            "comfyui": {
                                "compatible_refs": ["a" * 40],
                                "tested_refs": ["a" * 40],
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
                ),
                encoding="utf-8",
            )
            subprocess.run(["git", "add", "."], cwd=checkout_root, check=True)
            subprocess.run(
                ["git", "commit", "-qm", "fixture"],
                cwd=checkout_root,
                check=True,
            )
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=checkout_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            registry = registry_payload()
            registry["packs"]["published-world"]["ref"] = commit
            registry["packs"]["published-world"]["manifest_sha256"] = digest
            checkouts = ci_exact_refs.load_checkouts(
                write_registry(root, registry)
            )
            world = next(
                checkout for checkout in checkouts if checkout.pack_ref.id == "world"
            )

            ci_exact_refs.verify_checkout(world, checkout_root)

            manifest_path.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(
                ci_exact_refs.ExactRefCIError,
                "manifest verification failed",
            ):
                ci_exact_refs.verify_checkout(world, checkout_root)


if __name__ == "__main__":
    unittest.main()
