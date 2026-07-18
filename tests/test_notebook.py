from __future__ import annotations

import ast
import json
import unittest

from comfycolab.config import CoreStage0ConfigV1
from comfycolab.notebook import notebook_bytes, render_notebook


class NotebookTests(unittest.TestCase):
    def config(self) -> CoreStage0ConfigV1:
        return CoreStage0ConfigV1.create(
            core_repository="https://github.com/example/ComfyColab.git",
            core_commit="a" * 40,
            stage1_entrypoint="src/comfycolab/runtime.py",
            stage1_sha256="b" * 64,
            lock_bytes=b'{"packs":[],"schema":1}',
            colab_proxy=True,
        )

    def test_notebook_is_deterministic_and_cells_parse(self) -> None:
        first = notebook_bytes(self.config())
        second = notebook_bytes(self.config())
        self.assertEqual(first, second)
        payload = json.loads(first)
        self.assertEqual(payload["nbformat"], 4)
        self.assertEqual(len(payload["cells"]), 2)
        for cell in payload["cells"]:
            ast.parse("".join(cell["source"]))

    def test_notebook_embeds_same_stage0_config(self) -> None:
        config = self.config()
        notebook = render_notebook(config)
        bootstrap = "".join(notebook["cells"][1]["source"])
        self.assertIn("CONFIG_B64 =", bootstrap)
        self.assertIn("stage1_entrypoint", bootstrap)


if __name__ == "__main__":
    unittest.main()
