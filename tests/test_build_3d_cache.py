from __future__ import annotations

import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_3d_cache.py"


def load_module():
    spec = importlib.util.spec_from_file_location("comfycolab_build_3d_cache", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class Build3DCacheTests(unittest.TestCase):
    def test_split_archive_is_complete_ordered_and_checksum_pinned(self) -> None:
        module = load_module()
        payload = bytes(range(251)) * 12_000
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "cache.tar.zst"
            archive.write_bytes(payload)
            parts = module.split_archive(archive, part_bytes=1_000_000)
            rebuilt = b"".join(
                (archive.parent / part["name"]).read_bytes() for part in parts
            )
        self.assertEqual(rebuilt, payload)
        self.assertEqual(sum(part["bytes"] for part in parts), len(payload))
        offset = 0
        for part in parts:
            end = offset + part["bytes"]
            self.assertEqual(
                part["sha256"],
                hashlib.sha256(payload[offset:end]).hexdigest(),
            )
            offset = end

    def test_parser_requires_explicit_install_overlay(self) -> None:
        module = load_module()
        args = module.parser().parse_args([])
        self.assertFalse(args.install_overlay)
        self.assertEqual(args.part_bytes, 1_900_000_000)


if __name__ == "__main__":
    unittest.main()
