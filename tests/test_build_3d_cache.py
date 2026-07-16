from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_3d_cache.py"


def load_module():
    spec = importlib.util.spec_from_file_location("comfycolab_build_3d_cache", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def fake_remote_bootstrap():
    return types.SimpleNamespace(
        COMFY_REF="comfy-ref",
        TRELLIS_REF="trellis-ref",
        GEOMETRY_REF="geometry-ref",
        ULTRASHAPE_REF="ultrashape-ref",
        PIXAL3D_REF="pixal3d-ref",
        ULTRASHAPE_CUBVH_REF="cubvh-ref",
        BIREFNET_MODEL_REF="birefnet-ref",
        COMFY_ENV_VERSION="comfy-env-version",
        TRELLIS_PATCH_ID="trellis-patch",
        TRELLIS_CATEGORY_PATCH_ID="trellis-category-patch",
        ULTRASHAPE_PATCH_ID="ultrashape-patch",
        PIXAL3D_PATCH_ID="pixal3d-patch",
    )


def passed_validation_record(module, remote, *, profile="combined-profile"):
    gates = {
        name: {"status": "passed", "evidence": f"evidence/{name}.json"}
        for name in module.REQUIRED_LIVE_GATES
    }
    benchmarks = {}
    for name, resolution in module.REQUIRED_LIVE_BENCHMARKS.items():
        benchmark = {
            "status": "passed",
            "actualResolution": resolution,
            "runtimeSeconds": 1.25,
            "peakVramBytes": 1024,
            "glbBytes": 2048,
            "faces": 128,
            "glbValidated": True,
        }
        if name.startswith("trellis_"):
            benchmark.update(tokens=256, textureSize=512)
        if name.startswith("pixal3d_"):
            benchmark.update(
                tokens=256,
                textureSize=2048,
                workerPeakVramBytes=1024,
                pipelineLoadCount=1,
                workerPid=4321,
            )
        benchmarks[name] = benchmark
    return {
        "schema": module.LIVE_VALIDATION_SCHEMA,
        "status": "passed",
        "profile": profile,
        "runId": "g4-live-run-001",
        "completedAt": "2026-07-13T12:34:56Z",
        "sources": module.expected_validation_sources(remote),
        "patches": module.expected_validation_patches(remote),
        "gates": gates,
        "benchmarks": benchmarks,
    }


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

    def test_parser_requires_explicit_validation_record(self) -> None:
        module = load_module()
        with mock.patch("sys.stderr"):
            with self.assertRaises(SystemExit):
                module.parser().parse_args([])
        args = module.parser().parse_args(
            ["--validation-record", "docs/3d-validation.json"]
        )
        self.assertFalse(args.install_overlay)
        self.assertEqual(args.part_bytes, 1_900_000_000)
        self.assertEqual(args.validation_record, Path("docs/3d-validation.json"))

    def test_live_validation_requires_passed_complete_evidence(self) -> None:
        module = load_module()
        remote = fake_remote_bootstrap()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "validation.json"
            record = passed_validation_record(module, remote)
            record["status"] = "pending"
            path.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "not passed"):
                module.load_live_validation_record(
                    path, expected_profile="combined-profile", remote_bootstrap=remote
                )

            record = passed_validation_record(module, remote)
            record["gates"][module.REQUIRED_LIVE_GATES[-1]]["evidence"] = None
            path.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "gates are incomplete"):
                module.load_live_validation_record(
                    path, expected_profile="combined-profile", remote_bootstrap=remote
                )

    def test_live_validation_rejects_wrong_profile_sources_and_benchmarks(self) -> None:
        module = load_module()
        remote = fake_remote_bootstrap()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "validation.json"
            cases = []
            wrong_profile = passed_validation_record(module, remote)
            wrong_profile["profile"] = "other-profile"
            cases.append((wrong_profile, "profile"))
            wrong_source = passed_validation_record(module, remote)
            wrong_source["sources"]["cubvh"] = "mutable-source"
            cases.append((wrong_source, "sources"))
            false_1536 = passed_validation_record(module, remote)
            false_1536["benchmarks"]["trellis_1536_cascade"]["actualResolution"] = 1408
            cases.append((false_1536, "benchmarks are incomplete"))
            empty_metric = passed_validation_record(module, remote)
            empty_metric["benchmarks"]["ultrashape_1024_run_2"]["peakVramBytes"] = 0
            cases.append((empty_metric, "benchmarks are incomplete"))

            for record, message in cases:
                with self.subTest(message=message):
                    path.write_text(json.dumps(record), encoding="utf-8")
                    with self.assertRaisesRegex(RuntimeError, message):
                        module.load_live_validation_record(
                            path,
                            expected_profile="combined-profile",
                            remote_bootstrap=remote,
                        )

    def test_passed_live_validation_is_checksum_bound_into_ready_manifest(self) -> None:
        module = load_module()
        remote = fake_remote_bootstrap()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            validation_path = root / "validation.json"
            record = passed_validation_record(module, remote)
            validation_path.write_text(
                json.dumps(record, indent=2) + "\n", encoding="utf-8"
            )
            loaded = module.load_live_validation_record(
                validation_path,
                expected_profile="combined-profile",
                remote_bootstrap=remote,
            )
            provenance = module.live_validation_provenance(validation_path, loaded)
            self.assertEqual(
                provenance["recordSha256"],
                hashlib.sha256(validation_path.read_bytes()).hexdigest(),
            )
            self.assertEqual(provenance["passedGates"], list(module.REQUIRED_LIVE_GATES))

            workspace = root / ".ce"
            workspace.mkdir()
            (workspace / "pixi.toml").write_text("toml", encoding="utf-8")
            (workspace / "pixi.lock").write_text("lock", encoding="utf-8")
            (workspace / "install.hash").write_text("install-hash\n", encoding="utf-8")
            archive = root / "cache.tar.zst"
            archive.write_bytes(b"archive")
            template = {
                "profile": "combined-profile",
                "releaseTag": "3d-cache-v2",
                "fallbackProfile": "trellis-v1",
            }
            with mock.patch.object(module, "runtime_metadata", return_value={"gpu": "G4"}):
                manifest = module.build_manifest(
                    template=template,
                    workspace=workspace,
                    archive=archive,
                    parts=[{"name": "part-000", "bytes": 7, "sha256": "digest"}],
                    unpacked_bytes=123,
                    live_validation=provenance,
                    remote_bootstrap=remote,
                )
            self.assertEqual(manifest["status"], "ready")
            self.assertEqual(manifest["liveValidation"], provenance)

    def test_expected_sources_and_manifest_are_revisioned_for_pixal3d(self) -> None:
        module = load_module()
        remote = fake_remote_bootstrap()
        sources = module.expected_validation_sources(remote)
        patches = module.expected_validation_patches(remote)

        self.assertEqual(sources["pixal3d"], "pixal3d-ref")
        self.assertEqual(patches["pixal3d"], "pixal3d-patch")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            validation_path = root / "validation.json"
            record = passed_validation_record(module, remote)
            validation_path.write_text(json.dumps(record), encoding="utf-8")
            loaded = module.load_live_validation_record(
                validation_path,
                expected_profile="combined-profile",
                remote_bootstrap=remote,
            )
            workspace = root / ".ce"
            workspace.mkdir()
            (workspace / "pixi.toml").write_text("toml", encoding="utf-8")
            (workspace / "pixi.lock").write_text("lock", encoding="utf-8")
            (workspace / "install.hash").write_text("install-hash\n", encoding="utf-8")
            archive = root / "cache.tar.zst"
            archive.write_bytes(b"archive")
            with mock.patch.object(module, "runtime_metadata", return_value={"gpu": "G4"}):
                manifest = module.build_manifest(
                    template={
                        "profile": "combined-profile",
                        "releaseTag": "3d-cache-v3",
                        "fallbackProfile": "3d-cache-v2",
                    },
                    workspace=workspace,
                    archive=archive,
                    parts=[],
                    unpacked_bytes=123,
                    live_validation=module.live_validation_provenance(validation_path, loaded),
                    remote_bootstrap=remote,
                )

        self.assertEqual(manifest["sources"]["pixal3d"], "pixal3d-ref")
        self.assertEqual(manifest["patches"]["pixal3d"], "pixal3d-patch")

    def test_real_pixal3d_validation_source_schema_matches_checked_in_record(self) -> None:
        module = load_module()
        from comfycolab import remote_bootstrap

        expected = module.expected_validation_sources(remote_bootstrap)
        expected_patches = module.expected_validation_patches(remote_bootstrap)
        record = json.loads((ROOT / "docs/3d-validation.json").read_text(encoding="utf-8"))
        pixal_keys = {key for key in expected if key.startswith("pixal3d")}
        self.assertTrue(pixal_keys)
        self.assertEqual(
            {key: record["sources"][key] for key in pixal_keys},
            {key: expected[key] for key in pixal_keys},
        )
        self.assertEqual(record["patches"]["pixal3d"], expected_patches["pixal3d"])


if __name__ == "__main__":
    unittest.main()
