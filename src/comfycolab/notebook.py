from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .config import CoreStage0ConfigV1
from .repositories import RepositoryError, validate_repository_url
from .stage0 import render_stage0


_PROFILE_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_PACK_ALIAS_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_LICENSE_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")


class NotebookConfigError(ValueError):
    """Raised when a notebook selection cannot be rendered safely."""


def _dedupe_sorted(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(sorted(set(values)))


@dataclass(frozen=True)
class RuntimeResolvedMainNotebookConfig:
    core_repository: str
    profile: str = "core"
    pack_aliases: tuple[str, ...] = ()
    port: int = 8188
    refresh: bool = False
    colab_proxy: bool = True
    runtime_mode: str = "generic"
    accepted_licenses: tuple[str, ...] = ()

    @classmethod
    def create(
        cls,
        *,
        core_repository: str,
        profile: str = "core",
        pack_aliases: Sequence[str] = (),
        port: int = 8188,
        refresh: bool = False,
        colab_proxy: bool = True,
        runtime_mode: str = "generic",
        accepted_licenses: Sequence[str] = (),
    ) -> "RuntimeResolvedMainNotebookConfig":
        return cls(
            core_repository=core_repository,
            profile=profile,
            pack_aliases=_dedupe_sorted(pack_aliases),
            port=port,
            refresh=refresh,
            colab_proxy=colab_proxy,
            runtime_mode=runtime_mode,
            accepted_licenses=_dedupe_sorted(accepted_licenses),
        ).validated()

    def validated(self) -> "RuntimeResolvedMainNotebookConfig":
        if not isinstance(self.core_repository, str):
            raise NotebookConfigError("core_repository must be a string")
        try:
            validate_repository_url(self.core_repository)
        except RepositoryError as error:
            raise NotebookConfigError(str(error)) from error
        if not isinstance(self.profile, str) or not _PROFILE_RE.fullmatch(self.profile):
            raise NotebookConfigError("profile must be a public profile name")
        if (
            not isinstance(self.pack_aliases, tuple)
            or tuple(sorted(self.pack_aliases)) != self.pack_aliases
            or len(self.pack_aliases) != len(set(self.pack_aliases))
            or any(
                not isinstance(item, str) or not _PACK_ALIAS_RE.fullmatch(item)
                for item in self.pack_aliases
            )
        ):
            raise NotebookConfigError("pack_aliases must be sorted public pack aliases")
        if type(self.port) is not int or not 1 <= self.port <= 65535:
            raise NotebookConfigError("port must be an integer between 1 and 65535")
        if type(self.refresh) is not bool or type(self.colab_proxy) is not bool:
            raise NotebookConfigError("refresh and colab_proxy must be booleans")
        if self.runtime_mode not in {"generic", "legacy-full"}:
            raise NotebookConfigError("runtime_mode must be 'generic' or 'legacy-full'")
        if (
            not isinstance(self.accepted_licenses, tuple)
            or tuple(sorted(self.accepted_licenses)) != self.accepted_licenses
            or len(self.accepted_licenses) != len(set(self.accepted_licenses))
            or any(
                not isinstance(item, str) or not _LICENSE_RE.fullmatch(item)
                for item in self.accepted_licenses
            )
        ):
            raise NotebookConfigError(
                "accepted_licenses must be sorted public license-gate IDs"
            )
        return self


def _proxy_helpers_source(port: int) -> str:
    return (
        "import json\n"
        "import os\n"
        "from urllib.parse import urlparse\n"
        "from google.colab import output\n"
        "\n"
        f"COMFYCOLAB_PORT = {port}\n"
        "\n"
        "\n"
        "def _comfycolab_validate_proxy_url(value):\n"
        "    if (\n"
        "        not isinstance(value, str)\n"
        "        or not value\n"
        "        or value != value.strip()\n"
        "        or any(ord(character) < 32 or ord(character) == 127 for character in value)\n"
        "        or \"?\" in value\n"
        "        or \"#\" in value\n"
        "    ):\n"
        "        raise RuntimeError(\"Colab returned an invalid proxy URL.\")\n"
        "    parsed = urlparse(value)\n"
        "    hostname = (parsed.hostname or \"\").lower().rstrip(\".\")\n"
        "    trusted_host = any(\n"
        "        hostname.endswith(f\".{suffix}\") and hostname != suffix\n"
        "        for suffix in (\"prod.colab.dev\", \"colab.googleusercontent.com\")\n"
        "    )\n"
        "    try:\n"
        "        trusted_port = parsed.port in {None, 443}\n"
        "    except ValueError:\n"
        "        trusted_port = False\n"
        "    if (\n"
        "        parsed.scheme != \"https\"\n"
        "        or not trusted_host\n"
        "        or not trusted_port\n"
        "        or parsed.username is not None\n"
        "        or parsed.password is not None\n"
        "        or parsed.path not in {\"\", \"/\"}\n"
        "        or parsed.params\n"
        "        or parsed.query\n"
        "        or parsed.fragment\n"
        "    ):\n"
        "        raise RuntimeError(\"Colab returned an untrusted proxy URL.\")\n"
        "    return f\"https://{hostname}/\", f\"https://{hostname}\"\n"
        "\n"
        "\n"
        "def _comfycolab_reserve_proxy():\n"
        "    value = output.eval_js(\n"
        "        f\"\"\"\n"
        "(async () => {{\n"
        "  if (!google.colab.kernel.accessAllowed) {{\n"
        "    throw new Error(\"Allow this notebook to access the Colab runtime first.\");\n"
        "  }}\n"
        "  const proxy = await google.colab.kernel.proxyPort({COMFYCOLAB_PORT});\n"
        "  return new URL(\"/\", proxy).toString();\n"
        "}})()\n"
        "\"\"\",\n"
        "        timeout_sec=30,\n"
        "    )\n"
        "    return _comfycolab_validate_proxy_url(value)\n"
        "\n"
        "\n"
        "def _comfycolab_probe_proxy(value):\n"
        "    base_url = json.dumps(value)\n"
        "    expression = f\"\"\"\n"
        "(async () => {{\n"
        "  const baseUrl = {base_url};\n"
        "  try {{\n"
        "    await fetch(new URL(\"system_stats\", baseUrl), {{\n"
        "      mode: \"no-cors\",\n"
        "      credentials: \"include\",\n"
        "      cache: \"no-store\",\n"
        "    }});\n"
        "    const socketUrl = new URL(\"ws\", baseUrl);\n"
        "    socketUrl.protocol = socketUrl.protocol === \"https:\" ? \"wss:\" : \"ws:\";\n"
        "    socketUrl.searchParams.set(\"clientId\", crypto.randomUUID());\n"
        "    return await new Promise((resolve) => {{\n"
        "      const socket = new WebSocket(socketUrl);\n"
        "      const timer = setTimeout(() => {{\n"
        "        socket.close();\n"
        "        resolve(false);\n"
        "      }}, 10000);\n"
        "      socket.onopen = () => {{\n"
        "        clearTimeout(timer);\n"
        "        socket.close();\n"
        "        resolve(true);\n"
        "      }};\n"
        "      socket.onerror = () => {{\n"
        "        clearTimeout(timer);\n"
        "        resolve(false);\n"
        "      }};\n"
        "    }});\n"
        "  }} catch (error) {{\n"
        "    return false;\n"
        "  }}\n"
        "}})()\n"
        "\"\"\"\n"
        "    return output.eval_js(expression, timeout_sec=15) is True\n"
    )


def _proxy_cell_source(port: int) -> str:
    return _proxy_helpers_source(port) + (
        "\n"
        "COMFYCOLAB_PROXY_URL = None\n"
        "COMFYCOLAB_CORS_ORIGIN = None\n"
        "try:\n"
        "    COMFYCOLAB_PROXY_URL, COMFYCOLAB_CORS_ORIGIN = _comfycolab_reserve_proxy()\n"
        "except Exception as proxy_error:\n"
        "    os.environ.pop(\"COMFYCOLAB_PROXY_URL\", None)\n"
        "    os.environ.pop(\"COMFYCOLAB_CORS_ORIGIN\", None)\n"
        "    print(f\"Colab proxy reservation will retry in Cell 2 ({proxy_error}).\")\n"
        "else:\n"
        "    os.environ[\"COMFYCOLAB_PROXY_URL\"] = COMFYCOLAB_PROXY_URL\n"
        "    os.environ[\"COMFYCOLAB_CORS_ORIGIN\"] = COMFYCOLAB_CORS_ORIGIN\n"
        "    print(\"Primary access path reserved: private, session-bound Colab proxy.\")\n"
        "print(\"Run Cell 2 to start ComfyUI and verify the proxy HTTP/WebSocket path.\")\n"
    )


def _bootstrap_proxy_prelude(port: int) -> str:
    return _proxy_helpers_source(port) + (
        "\n"
        "COMFYCOLAB_PROXY_URL = None\n"
        "COMFYCOLAB_CORS_ORIGIN = None\n"
        "try:\n"
        "    candidate_url = os.environ[\"COMFYCOLAB_PROXY_URL\"]\n"
        "    candidate_origin = os.environ[\"COMFYCOLAB_CORS_ORIGIN\"]\n"
        "    COMFYCOLAB_PROXY_URL, expected_origin = (\n"
        "        _comfycolab_validate_proxy_url(candidate_url)\n"
        "    )\n"
        "    if candidate_origin != expected_origin:\n"
        "        raise RuntimeError(\"Colab proxy URL and CORS origin do not match.\")\n"
        "    COMFYCOLAB_CORS_ORIGIN = expected_origin\n"
        "except (KeyError, RuntimeError):\n"
        "    os.environ.pop(\"COMFYCOLAB_PROXY_URL\", None)\n"
        "    os.environ.pop(\"COMFYCOLAB_CORS_ORIGIN\", None)\n"
        "    try:\n"
        "        COMFYCOLAB_PROXY_URL, COMFYCOLAB_CORS_ORIGIN = (\n"
        "            _comfycolab_reserve_proxy()\n"
        "        )\n"
        "    except Exception as proxy_error:\n"
        "        print(\n"
        "            f\"Colab proxy reservation unavailable ({proxy_error}); \"\n"
        "            \"continuing so Cloudflare can remain available.\"\n"
        "        )\n"
        "    else:\n"
        "        os.environ[\"COMFYCOLAB_PROXY_URL\"] = COMFYCOLAB_PROXY_URL\n"
        "        os.environ[\"COMFYCOLAB_CORS_ORIGIN\"] = COMFYCOLAB_CORS_ORIGIN\n"
        "else:\n"
        "    os.environ[\"COMFYCOLAB_PROXY_URL\"] = COMFYCOLAB_PROXY_URL\n"
        "    os.environ[\"COMFYCOLAB_CORS_ORIGIN\"] = COMFYCOLAB_CORS_ORIGIN\n"
    )


def _bootstrap_display_source(port: int) -> str:
    return (
        "def _comfycolab_runtime_state():\n"
        "    state_path = Path(\"/content/.comfycolab/runtime.json\")\n"
        "    try:\n"
        "        state = json.loads(state_path.read_text(encoding=\"utf-8\"))\n"
        "    except (FileNotFoundError, json.JSONDecodeError, OSError) as state_error:\n"
        "        raise RuntimeError(\n"
        "            \"ComfyColab runtime state is missing or invalid; rerun Cell 2.\"\n"
        "        ) from state_error\n"
        "    if not isinstance(state, dict) or state.get(\"status\") != \"ready\":\n"
        "        raise RuntimeError(\"ComfyColab is not ready; rerun Cell 2.\")\n"
        "    return state\n"
        "\n"
        "\n"
        "def _comfycolab_cloudflare_fallback(state):\n"
        "    value = state.get(\"cloudflareUrl\")\n"
        "    if not isinstance(value, str):\n"
        "        return None\n"
        "    parsed = urlparse(value)\n"
        "    hostname = (parsed.hostname or \"\").lower().rstrip(\".\")\n"
        "    if (\n"
        "        parsed.scheme == \"https\"\n"
        "        and hostname.endswith(\".trycloudflare.com\")\n"
        "        and parsed.username is None\n"
        "        and parsed.password is None\n"
        "    ):\n"
        "        return value\n"
        "    return None\n"
        "\n"
        "\n"
        "state = _comfycolab_runtime_state()\n"
        "cloudflare_url = _comfycolab_cloudflare_fallback(state)\n"
        "proxy_url = COMFYCOLAB_PROXY_URL\n"
        "proxy_error = None\n"
        "try:\n"
        "    if proxy_url is None:\n"
        "        proxy_url, _ = _comfycolab_reserve_proxy()\n"
        "    else:\n"
        "        proxy_url, _ = _comfycolab_validate_proxy_url(proxy_url)\n"
        "except Exception as error:\n"
        "    proxy_error = error\n"
        "\n"
        "if proxy_url is not None and proxy_error is None:\n"
        "    print(\"\\nComfyUI Colab proxy URL (primary, current user/session only):\")\n"
        "    print(proxy_url)\n"
        "    try:\n"
        "        proxy_probe_ready = _comfycolab_probe_proxy(proxy_url)\n"
        "    except Exception as probe_error:\n"
        "        proxy_probe_ready = False\n"
        "        print(f\"Colab proxy browser-side readiness probe was inconclusive ({probe_error}).\")\n"
        "    if not proxy_probe_ready:\n"
        "        print(\"The direct session-bound Colab proxy link remains the primary access path.\")\n"
        "    try:\n"
        "        output.serve_kernel_port_as_iframe(\n"
        f"            {port}, path=\"/\", width=\"100%\", height=\"900\"\n"
        "        )\n"
        "    except Exception as iframe_error:\n"
        "        print(f\"Colab inline preview unavailable ({iframe_error}); use the direct link above.\")\n"
        "\n"
        "if proxy_error is not None:\n"
        "    print(f\"Colab proxy unavailable ({proxy_error}).\")\n"
        "\n"
        "if cloudflare_url is not None:\n"
        "    label = (\n"
        "        \"Cloudflare fallback URL\"\n"
        "        if proxy_url is not None and proxy_error is None\n"
        "        else \"Selected Cloudflare URL\"\n"
        "    )\n"
        "    print(f\"\\n{label}:\")\n"
        "    print(cloudflare_url)\n"
        "elif proxy_error is not None:\n"
        "    raise RuntimeError(\n"
        "        \"Both the Colab proxy and Cloudflare fallback are unavailable.\"\n"
        "    ) from proxy_error\n"
    )


def _render_bootstrap_cell(config: CoreStage0ConfigV1) -> str:
    bootstrap = render_stage0(config)
    if not config.colab_proxy:
        return bootstrap
    future_import = "from __future__ import annotations\n"
    if bootstrap.count(future_import) != 1:
        raise RuntimeError("Stage-0 future import is missing or duplicated.")
    bootstrap_with_proxy = bootstrap.replace(
        future_import,
        future_import + "\n" + _bootstrap_proxy_prelude(config.port) + "\n\n",
        1,
    )
    return bootstrap_with_proxy + "\n\n" + _bootstrap_display_source(config.port)


def _runtime_resolved_main_source(config: RuntimeResolvedMainNotebookConfig) -> str:
    config.validated()
    return (
        "import json\n"
        "import re\n"
        "import shutil\n"
        "import subprocess\n"
        "import sys\n"
        "from pathlib import Path\n"
        "\n"
        f"COMFYCOLAB_CORE_REPOSITORY = {config.core_repository!r}\n"
        f"COMFYCOLAB_PROFILE = {config.profile!r}\n"
        f"COMFYCOLAB_PACK_ALIASES = {list(config.pack_aliases)!r}\n"
        f"COMFYCOLAB_PORT = {config.port!r}\n"
        f"COMFYCOLAB_REFRESH = {config.refresh!r}\n"
        f"COMFYCOLAB_COLAB_PROXY = {config.colab_proxy!r}\n"
        f"COMFYCOLAB_RUNTIME_MODE = {config.runtime_mode!r}\n"
        f"COMFYCOLAB_ACCEPTED_LICENSES = {list(config.accepted_licenses)!r}\n"
        "COMFYCOLAB_CORE_REF = \"main\"\n"
        "COMFYCOLAB_CHECKOUT = Path(\"/content/ComfyColab\")\n"
        "COMFYCOLAB_COMMIT_RE = re.compile(r\"^[0-9a-f]{40}$\")\n"
        "\n"
        "\n"
        "def _comfycolab_run(command, *, cwd=None):\n"
        "    return subprocess.run(\n"
        "        command,\n"
        "        cwd=cwd,\n"
        "        text=True,\n"
        "        capture_output=True,\n"
        "        check=True,\n"
        "    )\n"
        "\n"
        "\n"
        "if COMFYCOLAB_CHECKOUT.exists():\n"
        "    shutil.rmtree(COMFYCOLAB_CHECKOUT)\n"
        "_comfycolab_run([\n"
        "    \"git\",\n"
        "    \"clone\",\n"
        "    \"--branch\",\n"
        "    COMFYCOLAB_CORE_REF,\n"
        "    \"--single-branch\",\n"
        "    COMFYCOLAB_CORE_REPOSITORY,\n"
        "    str(COMFYCOLAB_CHECKOUT),\n"
        "])\n"
        "COMFYCOLAB_CORE_COMMIT = _comfycolab_run(\n"
        "    [\"git\", \"rev-parse\", \"HEAD\"],\n"
        "    cwd=COMFYCOLAB_CHECKOUT,\n"
        ").stdout.strip()\n"
        "if COMFYCOLAB_COMMIT_RE.fullmatch(COMFYCOLAB_CORE_COMMIT) is None:\n"
        "    raise RuntimeError(\n"
        "        f\"main resolved to an invalid commit: {COMFYCOLAB_CORE_COMMIT!r}\"\n"
        "    )\n"
        "sys.path.insert(0, str(COMFYCOLAB_CHECKOUT / \"src\"))\n"
        "from comfycolab.resolution import prepare_launch\n"
        "\n"
        "prepared = prepare_launch(\n"
        "    core_repository=COMFYCOLAB_CORE_REPOSITORY,\n"
        "    core_ref=COMFYCOLAB_CORE_COMMIT,\n"
        "    pack_aliases=COMFYCOLAB_PACK_ALIASES,\n"
        "    profile=COMFYCOLAB_PROFILE,\n"
        "    port=COMFYCOLAB_PORT,\n"
        "    refresh=COMFYCOLAB_REFRESH,\n"
        "    colab_proxy=COMFYCOLAB_COLAB_PROXY,\n"
        "    runtime_mode=COMFYCOLAB_RUNTIME_MODE,\n"
        "    accepted_licenses=COMFYCOLAB_ACCEPTED_LICENSES,\n"
        ")\n"
        "print(\n"
        "    \"Resolved latest main to immutable core commit \"\n"
        "    f\"{prepared.config.core_commit}.\"\n"
        ")\n"
        "print(f\"Embedded lock SHA-256: {prepared.lock.sha256}\")\n"
        "exec(compile(prepared.source, \"<comfycolab-runtime-resolved-stage0>\", \"exec\"), {\n"
        "    \"__name__\": \"__main__\",\n"
        "    \"__file__\": \"<comfycolab-runtime-resolved-stage0>\",\n"
        "})\n"
    )


def _render_runtime_resolved_bootstrap_cell(
    config: RuntimeResolvedMainNotebookConfig,
) -> str:
    config.validated()
    source = (
        "from __future__ import annotations\n\n"
        + (
            _bootstrap_proxy_prelude(config.port) + "\n\n"
            if config.colab_proxy
            else ""
        )
        + _runtime_resolved_main_source(config)
    )
    if config.colab_proxy:
        source += "\n\n" + _bootstrap_display_source(config.port)
    return source


def render_notebook(config: CoreStage0ConfigV1) -> dict[str, Any]:
    config.validated()
    proxy_source = (
        _proxy_cell_source(config.port)
        if config.colab_proxy
        else "print(\"Colab proxy disabled for this notebook.\")\n"
    )
    bootstrap_source = _render_bootstrap_cell(config)
    notebook: dict[str, Any] = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"name": "ComfyColab.ipynb", "provenance": []},
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "reserve-comfyui-proxy",
                "metadata": {},
                "outputs": [],
                "source": proxy_source.splitlines(keepends=True),
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "authenticated-comfycolab-bootstrap",
                "metadata": {},
                "outputs": [],
                "source": bootstrap_source.splitlines(keepends=True),
            },
        ],
    }
    for cell in notebook["cells"]:
        compile("".join(cell["source"]), f"<{cell['id']}>", "exec")
    return notebook


def render_runtime_resolved_main_notebook(
    config: RuntimeResolvedMainNotebookConfig,
) -> dict[str, Any]:
    config.validated()
    proxy_source = (
        _proxy_cell_source(config.port)
        if config.colab_proxy
        else "print(\"Colab proxy disabled for this notebook.\")\n"
    )
    bootstrap_source = _render_runtime_resolved_bootstrap_cell(config)
    notebook: dict[str, Any] = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"name": "ComfyColab.ipynb", "provenance": []},
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "reserve-comfyui-proxy",
                "metadata": {},
                "outputs": [],
                "source": proxy_source.splitlines(keepends=True),
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "runtime-resolved-comfycolab-bootstrap",
                "metadata": {},
                "outputs": [],
                "source": bootstrap_source.splitlines(keepends=True),
            },
        ],
    }
    for cell in notebook["cells"]:
        compile("".join(cell["source"]), f"<{cell['id']}>", "exec")
    return notebook


def notebook_bytes(config: CoreStage0ConfigV1) -> bytes:
    return (
        json.dumps(
            render_notebook(config),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")


def write_notebook(path: Path, config: CoreStage0ConfigV1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(notebook_bytes(config))
    temporary.replace(path)


def runtime_resolved_main_notebook_bytes(
    config: RuntimeResolvedMainNotebookConfig,
) -> bytes:
    return (
        json.dumps(
            render_runtime_resolved_main_notebook(config),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")


def write_runtime_resolved_main_notebook(
    path: Path,
    config: RuntimeResolvedMainNotebookConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(runtime_resolved_main_notebook_bytes(config))
    temporary.replace(path)
