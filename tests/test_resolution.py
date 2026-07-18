from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from comfycolab.resolution import (
    ResolutionError,
    prepare_launch_from_lock,
    resolve_from_checkout,
    select_pack_refs,
)


class ResolutionTests(unittest.TestCase):
    def make_core(self, root: Path) -> Path:
        (root / "registry").mkdir(parents=True)
        (root / "profiles").mkdir()
        (root / "src" / "comfycolab").mkdir(parents=True)
        (root / "registry" / "official-packs.json").write_text(
            '{"schema":1,"packs":{}}',
            encoding="utf-8",
        )
        (root / "registry" / "engine.json").write_text(
            json.dumps(
                {
                    "schema": 1,
                    "comfyui": {
                        "repository": "https://github.com/Comfy-Org/ComfyUI.git",
                        "commit": "b" * 40,
                    },
                }
            ),
            encoding="utf-8",
        )
        (root / "profiles" / "core.json").write_text(
            '{"schema":1,"id":"core","packs":[]}',
            encoding="utf-8",
        )
        (root / "src" / "comfycolab" / "__init__.py").write_text(
            '__version__ = "0.2.0.dev1"\n',
            encoding="utf-8",
        )
        return root

    def test_core_profile_resolves_zero_pack_lock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = self.make_core(Path(directory) / "core")
            workspace = Path(directory) / "workspace"
            lock, profile = resolve_from_checkout(
                root,
                core_repository="https://github.com/example/ComfyColab.git",
                core_commit="a" * 40,
                workspace=workspace,
            )
        self.assertEqual(profile, "core")
        self.assertEqual(lock.to_dict()["packs"], [])
        self.assertEqual(lock.to_dict()["comfyui"]["commit"], "b" * 40)

    def test_unknown_pack_alias_fails_before_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = self.make_core(Path(directory) / "core")
            with self.assertRaisesRegex(ResolutionError, "authenticated official registry"):
                select_pack_refs(root, pack_aliases=["image"], profile=None)

    def test_refresh_reuses_exact_lock_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = self.make_core(Path(directory) / "core")
            lock, _ = resolve_from_checkout(
                root,
                core_repository="https://github.com/example/ComfyColab.git",
                core_commit="a" * 40,
                workspace=Path(directory) / "workspace",
            )
            stage1 = root / "src" / "comfycolab" / "runtime.py"
            stage1.write_text("# stage 1\n", encoding="utf-8")
            with mock.patch("comfycolab.resolution.temporary_checkout") as checkout:
                checkout.return_value.__enter__.return_value = (root, "a" * 40)
                prepared = prepare_launch_from_lock(lock)
        self.assertEqual(prepared.lock.canonical_bytes(), lock.canonical_bytes())
        self.assertTrue(prepared.config.refresh)


if __name__ == "__main__":
    unittest.main()
