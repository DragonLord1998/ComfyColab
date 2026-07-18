from __future__ import annotations

import base64
import io
import json
import os
import re
import sys
import types
import unittest
from contextlib import redirect_stdout
from unittest import mock

from comfycolab.config import CoreStage0ConfigV1
from comfycolab.notebook import (
    _bootstrap_proxy_prelude,
    _proxy_helpers_source,
    notebook_bytes,
    render_notebook,
)


class NotebookTests(unittest.TestCase):
    def config(
        self,
        *,
        colab_proxy: bool = True,
        runtime_mode: str = "generic",
        lock_bytes: bytes = b'{"packs":[],"schema":1}',
    ) -> CoreStage0ConfigV1:
        return CoreStage0ConfigV1.create(
            core_repository="https://github.com/example/ComfyColab.git",
            core_commit="a" * 40,
            stage1_entrypoint="src/comfycolab/runtime.py",
            stage1_sha256="b" * 64,
            lock_bytes=lock_bytes,
            colab_proxy=colab_proxy,
            runtime_mode=runtime_mode,
        )

    def test_notebook_is_deterministic_and_cells_compile(self) -> None:
        first = notebook_bytes(self.config())
        second = notebook_bytes(self.config())
        self.assertEqual(first, second)
        payload = json.loads(first)
        self.assertEqual(payload["nbformat"], 4)
        self.assertEqual(len(payload["cells"]), 2)
        for cell in payload["cells"]:
            compile(
                "".join(cell["source"]),
                f"<{cell['id']}>",
                "exec",
            )

    def test_notebook_embeds_same_stage0_config(self) -> None:
        config = self.config()
        notebook = render_notebook(config)
        bootstrap = "".join(notebook["cells"][1]["source"])
        self.assertIn("CONFIG_B64 =", bootstrap)
        self.assertIn("stage1_entrypoint", bootstrap)

    def test_full_node_notebook_embeds_legacy_runtime_and_exact_pack_ids(self) -> None:
        lock = {
            "schema": 1,
            "packs": [
                {"id": pack_id}
                for pack_id in ("3d", "3dgs", "image", "video")
            ],
        }
        lock_bytes = json.dumps(
            lock,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        notebook = render_notebook(
            self.config(
                runtime_mode="legacy-full",
                lock_bytes=lock_bytes,
            )
        )
        bootstrap = "".join(notebook["cells"][1]["source"])
        match = re.search(
            r'CONFIG_B64 = "([A-Za-z0-9+/=]+)"',
            bootstrap,
        )
        self.assertIsNotNone(match)
        assert match is not None
        restored = CoreStage0ConfigV1.from_dict(
            json.loads(base64.b64decode(match.group(1)))
        )
        self.assertEqual(restored.runtime_mode, "legacy-full")
        self.assertEqual(
            [pack["id"] for pack in json.loads(restored.lock_bytes())["packs"]],
            ["3d", "3dgs", "image", "video"],
        )

    def test_proxy_cell_reserves_trusted_session_bound_primary_url(self) -> None:
        notebook = render_notebook(self.config())
        proxy = "".join(notebook["cells"][0]["source"])
        self.assertIn("google.colab.kernel.accessAllowed", proxy)
        self.assertIn("COMFYCOLAB_PORT = 8188", proxy)
        self.assertIn("google.colab.kernel.proxyPort({COMFYCOLAB_PORT})", proxy)
        self.assertIn(
            '("prod.colab.dev", "colab.googleusercontent.com")',
            proxy,
        )
        self.assertIn("COMFYCOLAB_PROXY_URL", proxy)
        self.assertIn("COMFYCOLAB_CORS_ORIGIN", proxy)
        self.assertIn('os.environ["COMFYCOLAB_PROXY_URL"]', proxy)
        self.assertIn('os.environ["COMFYCOLAB_CORS_ORIGIN"]', proxy)
        self.assertIn('os.environ.pop("COMFYCOLAB_PROXY_URL"', proxy)
        self.assertIn("Primary access path reserved", proxy)
        self.assertIn("verify the proxy HTTP/WebSocket path", proxy)
        self.assertNotIn("print(COMFYCOLAB_PROXY_URL)", proxy)
        self.assertNotIn("serve_kernel_port_as_iframe", proxy)
        self.assertNotIn("Open ComfyUI", proxy)

    def test_bootstrap_reserves_before_stage0_then_probes_and_embeds(self) -> None:
        notebook = render_notebook(self.config())
        bootstrap = "".join(notebook["cells"][1]["source"])
        reserve_at = bootstrap.index(
            "_comfycolab_reserve_proxy()",
            bootstrap.index("except (KeyError, RuntimeError)"),
        )
        stage0_at = bootstrap.index('if __name__ == "__main__":')
        state_at = bootstrap.index("state = _comfycolab_runtime_state()")
        reuse_at = bootstrap.index("proxy_url = COMFYCOLAB_PROXY_URL")
        probe_at = bootstrap.index("_comfycolab_probe_proxy(proxy_url)")
        primary_url_at = bootstrap.index("ComfyUI Colab proxy URL")
        iframe_at = bootstrap.index("output.serve_kernel_port_as_iframe")
        future_at = bootstrap.index("from __future__ import annotations")
        self.assertLess(reserve_at, stage0_at)
        self.assertLess(future_at, reserve_at)
        self.assertLess(stage0_at, state_at)
        self.assertLess(state_at, reuse_at)
        self.assertLess(reuse_at, primary_url_at)
        self.assertLess(primary_url_at, probe_at)
        self.assertLess(primary_url_at, iframe_at)
        self.assertIn(
            "if proxy_url is None:\n        proxy_url, _ = _comfycolab_reserve_proxy()",
            bootstrap,
        )
        self.assertIn("COMFYCOLAB_PORT = 8188", bootstrap)
        self.assertIn("google.colab.kernel.proxyPort({COMFYCOLAB_PORT})", bootstrap)
        self.assertIn('new URL("system_stats", baseUrl)', bootstrap)
        self.assertIn("new WebSocket(socketUrl)", bootstrap)
        self.assertIn('os.environ["COMFYCOLAB_CORS_ORIGIN"]', bootstrap)
        self.assertIn(
            '8188, path="/", width="100%", height="900"',
            bootstrap,
        )

    def test_bootstrap_always_surfaces_cloudflare_after_proxy_selection(self) -> None:
        notebook = render_notebook(self.config())
        bootstrap = "".join(notebook["cells"][1]["source"])
        fallback_at = bootstrap.index("cloudflare_url = _comfycolab_cloudflare_fallback(state)")
        probe_at = bootstrap.index("_comfycolab_probe_proxy(proxy_url)")
        primary_url_at = bootstrap.index("ComfyUI Colab proxy URL")
        iframe_at = bootstrap.index("output.serve_kernel_port_as_iframe")
        fallback_print_at = bootstrap.index("Cloudflare fallback URL")
        self.assertLess(fallback_at, primary_url_at)
        self.assertLess(primary_url_at, probe_at)
        self.assertLess(probe_at, iframe_at)
        self.assertLess(iframe_at, fallback_print_at)
        self.assertIn("direct session-bound Colab proxy link remains", bootstrap)
        self.assertNotIn("proxy WebSocket readiness probe failed", bootstrap)
        self.assertIn("Selected Cloudflare URL", bootstrap)
        self.assertIn("Both the Colab proxy and Cloudflare fallback", bootstrap)
        self.assertIn("/content/.comfycolab/runtime.json", bootstrap)
        self.assertNotIn('state.get("comfyUrl")', bootstrap)

    def test_proxy_validation_accepts_observed_host_and_rejects_confusion(self) -> None:
        namespace: dict[str, object] = {}
        with mock.patch.dict(sys.modules, self.fake_colab_modules()):
            exec(_proxy_helpers_source(8188), namespace)
        validate = namespace["_comfycolab_validate_proxy_url"]
        self.assertTrue(callable(validate))
        assert callable(validate)
        observed = (
            "https://8188-gpu-g4-s-kkb-use5c1-3m501x4q87ilk-c."
            "us-east5-1.prod.colab.dev/"
        )
        self.assertEqual(
            validate(observed),
            (observed, observed.rstrip("/")),
        )
        for value in (
            "https://prod.colab.dev/",
            "https://googleusercontent.com/",
            "https://colab.googleusercontent.com/",
            "https://attacker.googleusercontent.com/",
            "https://abc.prod.colab.dev.evil.example/",
            "https://user@abc.prod.colab.dev/",
            "https://abc.prod.colab.dev:8443/",
            "https://abc.prod.colab.dev/path",
            "https://abc.prod.colab.dev/?",
            "https://abc.prod.colab.dev/#",
        ):
            with self.subTest(value=value), self.assertRaises(RuntimeError):
                validate(value)

    def test_cell2_reuses_validated_cell1_proxy_without_reserving_again(self) -> None:
        observed = (
            "https://8188-gpu-g4-s-kkb-use5c1-3m501x4q87ilk-c."
            "us-east5-1.prod.colab.dev/"
        )
        origin = observed.rstrip("/")
        modules = self.fake_colab_modules(
            error=RuntimeError("proxy reservation should not be repeated")
        )
        namespace: dict[str, object] = {}
        with (
            mock.patch.dict(sys.modules, modules),
            mock.patch.dict(
                os.environ,
                {
                    "COMFYCOLAB_PROXY_URL": observed,
                    "COMFYCOLAB_CORS_ORIGIN": origin,
                },
                clear=False,
            ),
            redirect_stdout(io.StringIO()) as output,
        ):
            exec(_bootstrap_proxy_prelude(8188), namespace)
            self.assertEqual(os.environ["COMFYCOLAB_PROXY_URL"], observed)
            self.assertEqual(os.environ["COMFYCOLAB_CORS_ORIGIN"], origin)
        self.assertNotIn("reservation unavailable", output.getvalue())
        self.assertEqual(namespace["COMFYCOLAB_PROXY_URL"], observed)
        self.assertEqual(namespace["COMFYCOLAB_CORS_ORIGIN"], origin)

    def test_cell2_proxy_reservation_failure_keeps_cloudflare_path_available(self) -> None:
        modules = self.fake_colab_modules(error=RuntimeError("proxy denied"))
        namespace: dict[str, object] = {}
        with (
            mock.patch.dict(sys.modules, modules),
            mock.patch.dict(
                os.environ,
                {
                    "COMFYCOLAB_PROXY_URL": "https://stale.prod.colab.dev/?",
                    "COMFYCOLAB_CORS_ORIGIN": "https://stale.prod.colab.dev",
                },
                clear=False,
            ),
            redirect_stdout(io.StringIO()) as output,
        ):
            exec(_bootstrap_proxy_prelude(8188), namespace)
            self.assertNotIn("COMFYCOLAB_PROXY_URL", os.environ)
            self.assertNotIn("COMFYCOLAB_CORS_ORIGIN", os.environ)
        self.assertIn("continuing so Cloudflare can remain available", output.getvalue())

    def test_no_colab_proxy_configuration_omits_proxy_frontend_code(self) -> None:
        notebook = render_notebook(self.config(colab_proxy=False))
        proxy = "".join(notebook["cells"][0]["source"])
        bootstrap = "".join(notebook["cells"][1]["source"])
        self.assertIn("Colab proxy disabled", proxy)
        self.assertNotIn("proxyPort", proxy)
        self.assertNotIn("serve_kernel_port_as_iframe", bootstrap)
        self.assertNotIn("COMFYCOLAB_CORS_ORIGIN", bootstrap)
        self.assertIn("CONFIG_B64 =", bootstrap)

    @staticmethod
    def fake_colab_modules(
        *,
        value: str = "https://abc.prod.colab.dev/",
        error: Exception | None = None,
    ) -> dict[str, types.ModuleType]:
        google = types.ModuleType("google")
        colab = types.ModuleType("google.colab")
        output = types.ModuleType("google.colab.output")

        def eval_js(*_args: object, **_kwargs: object) -> str:
            if error is not None:
                raise error
            return value

        output.eval_js = eval_js  # type: ignore[attr-defined]
        colab.output = output  # type: ignore[attr-defined]
        google.colab = colab  # type: ignore[attr-defined]
        return {
            "google": google,
            "google.colab": colab,
            "google.colab.output": output,
        }


if __name__ == "__main__":
    unittest.main()
