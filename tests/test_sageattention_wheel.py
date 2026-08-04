from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_sageattention_wheel",
    ROOT / "scripts" / "build_sageattention_wheel.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SageAttentionWheelTests(unittest.TestCase):
    def test_sm120_patch_removes_debug_symbols_and_unused_sm80_extension(self) -> None:
        source = f"{MODULE._DEBUG_FLAGS}\n{MODULE._SM80_CONDITION}\n"
        with tempfile.TemporaryDirectory() as directory:
            setup_path = Path(directory) / "setup.py"
            setup_path.write_text(source, encoding="utf-8")
            expected_digest = MODULE.SAGE_ATTENTION_SETUP_SHA256
            MODULE.SAGE_ATTENTION_SETUP_SHA256 = MODULE.sha256_file(setup_path)
            try:
                MODULE.patch_sm120_build(setup_path)
            finally:
                MODULE.SAGE_ATTENTION_SETUP_SHA256 = expected_digest
            patched = setup_path.read_text(encoding="utf-8")
        self.assertIn(MODULE._RELEASE_FLAGS, patched)
        self.assertIn(MODULE._TARGET_CONDITION, patched)
        self.assertNotIn(MODULE._DEBUG_FLAGS, patched)
        self.assertNotIn(MODULE._SM80_CONDITION, patched)

    def test_build_environment_targets_cuda13_sm120_conservatively(self) -> None:
        environment = MODULE.build_environment()
        self.assertEqual(environment["CUDA_HOME"], "/usr/local/cuda-13.0")
        self.assertEqual(environment["TORCH_CUDA_ARCH_LIST"], "12.0")
        self.assertEqual(environment["EXT_PARALLEL"], "1")
        self.assertEqual(environment["MAX_JOBS"], "2")


if __name__ == "__main__":
    unittest.main()
