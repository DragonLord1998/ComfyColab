from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import CoreStage0ConfigV1
from .stage0 import render_stage0


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
