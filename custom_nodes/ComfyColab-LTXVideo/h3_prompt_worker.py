from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import tarfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any

from .h3_prompt_policy import (
    MINIMAX_H3_GUIDE_REVISION,
    normalize_prompt_mode,
    system_policy,
    user_rewrite_request,
    validate_enhanced_prompt,
)


QWEN_GGUF_REPO = "unsloth/Qwen3.8-27B-GGUF"
QWEN_GGUF_REVISION = "f1bfb127c64f7072bdd2cad55f258b9c8b2910fe"
QWEN_GGUF_FILENAME = "Qwen3.8-27B-Q4_K_M.gguf"
QWEN_GGUF_SIZE = 17_106_775_008
QWEN_GGUF_SHA256 = "7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169"
QWEN_MODEL_ALIAS = "qwen3.8-27b-q4_k_m"

LLAMA_CPP_REPOSITORY = "https://github.com/ggml-org/llama.cpp.git"
LLAMA_CPP_RELEASE = "b10437"
LLAMA_CPP_REVISION = "16d222fc5ead59d20039501a37251c9ed457a454"
LLAMA_CPP_G4_CACHE_TAG = "llama-cpp-b10437-cu128-sm120-v1"
LLAMA_CPP_G4_CACHE_ASSET = "comfycolab-llama-b10437-g4-sm120.tar.gz"
LLAMA_CPP_G4_CACHE_SHA256 = (
    "f6b9ce7726f4346ab08660ce04b3e006c94d0cc46e147ab742f7603e12bda355"
)
LLAMA_CPP_G4_CACHE_URL = (
    "https://github.com/DragonLord1998/ComfyColab/releases/download/"
    f"{LLAMA_CPP_G4_CACHE_TAG}/{LLAMA_CPP_G4_CACHE_ASSET}"
)

_INSTALL_LOCK = threading.Lock()


def _runtime_root() -> Path:
    configured = os.environ.get("COMFYCOLAB_CACHE_DIR")
    root = Path(configured) if configured else Path(__file__).resolve().parents[2] / ".cache" / "comfycolab"
    target = root / "h3_prompt_enhancer"
    target.mkdir(parents=True, exist_ok=True)
    return target


def _sha256(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_qwen_gguf(*, force_redownload: bool = False) -> Path:
    override = os.environ.get("COMFYCOLAB_H3_PROMPT_GGUF")
    if override:
        path = Path(override).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(
                f"COMFYCOLAB_H3_PROMPT_GGUF does not point to a file: {path}"
            )
        return path

    marker = _runtime_root() / "qwen3.8-27b-q4_k_m.json"
    expected = {
        "repo_id": QWEN_GGUF_REPO,
        "revision": QWEN_GGUF_REVISION,
        "filename": QWEN_GGUF_FILENAME,
        "size_bytes": QWEN_GGUF_SIZE,
        "sha256": QWEN_GGUF_SHA256,
    }
    with _INSTALL_LOCK:
        if not force_redownload and marker.is_file():
            try:
                recorded = json.loads(marker.read_text(encoding="utf-8"))
                cached = Path(str(recorded.get("path", "")))
                if (
                    {key: recorded.get(key) for key in expected} == expected
                    and cached.is_file()
                    and cached.stat().st_size == QWEN_GGUF_SIZE
                ):
                    return cached
            except (OSError, json.JSONDecodeError):
                pass

        try:
            from huggingface_hub import hf_hub_download
        except ImportError as error:
            raise RuntimeError(
                "The H3 Prompt Enhancer requires huggingface_hub with hf-xet. "
                "Restart with `comfycolab start --refresh`."
            ) from error

        print(
            "[comfycolab] Downloading pinned Qwen3.8-27B Q4_K_M GGUF "
            f"({QWEN_GGUF_SIZE / 1_000_000_000:.1f} GB)...",
            flush=True,
        )
        downloaded = Path(
            hf_hub_download(
                repo_id=QWEN_GGUF_REPO,
                filename=QWEN_GGUF_FILENAME,
                revision=QWEN_GGUF_REVISION,
                force_download=bool(force_redownload),
            )
        ).resolve()
        actual_size = downloaded.stat().st_size
        if actual_size != QWEN_GGUF_SIZE:
            raise RuntimeError(
                "Qwen3.8 Q4 GGUF size mismatch: "
                f"expected {QWEN_GGUF_SIZE}, got {actual_size}."
            )
        actual_sha256 = _sha256(downloaded)
        if actual_sha256 != QWEN_GGUF_SHA256:
            raise RuntimeError(
                "Qwen3.8 Q4 GGUF checksum mismatch: "
                f"expected {QWEN_GGUF_SHA256}, got {actual_sha256}."
            )
        marker.write_text(
            json.dumps({**expected, "path": str(downloaded)}, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return downloaded


def _git_output(*args: str) -> str:
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()


def _cuda_compute_capability() -> str | None:
    try:
        output = _git_output(
            "nvidia-smi",
            "--query-gpu=compute_cap",
            "--format=csv,noheader",
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return output.splitlines()[0].strip() if output else None


def _validate_cache_members(members: list[tarfile.TarInfo]) -> None:
    root_files = {
        "llama.cpp-build.json",
        "llama.cpp-g4-sm120-cache.manifest.json",
    }
    for member in members:
        member_path = PurePosixPath(member.name)
        if member_path.is_absolute() or ".." in member_path.parts:
            raise RuntimeError(f"Unsafe llama.cpp cache member: {member.name}")
        allowed = (
            member.name in root_files
            or member.name == "llama.cpp-build/bin"
            or member.name.startswith("llama.cpp-build/bin/")
        )
        if not allowed:
            raise RuntimeError(f"Unexpected llama.cpp cache member: {member.name}")
        if member.issym() or member.islnk():
            link_path = PurePosixPath(member.linkname)
            if link_path.is_absolute() or ".." in link_path.parts:
                raise RuntimeError(
                    f"Unsafe llama.cpp cache link: {member.name} -> {member.linkname}"
                )


def _try_restore_g4_cache(
    runtime: Path,
    build: Path,
    marker: Path,
    expected: dict[str, str],
) -> bool:
    if (
        platform.system() != "Linux"
        or platform.machine() not in {"x86_64", "AMD64"}
        or _cuda_compute_capability() != "12.0"
    ):
        return False

    archive_path = runtime / LLAMA_CPP_G4_CACHE_ASSET
    partial_path = archive_path.with_suffix(archive_path.suffix + ".part")
    staging = runtime / ".llama.cpp-g4-cache-staging"
    try:
        if (
            not archive_path.is_file()
            or _sha256(archive_path) != LLAMA_CPP_G4_CACHE_SHA256
        ):
            archive_path.unlink(missing_ok=True)
            partial_path.unlink(missing_ok=True)
            print(
                "[comfycolab] Restoring pinned llama.cpp CUDA 12.8 SM120 cache "
                "from GitHub Releases...",
                flush=True,
            )
            request = urllib.request.Request(
                LLAMA_CPP_G4_CACHE_URL,
                headers={"User-Agent": "ComfyColab-H3-Prompt-Enhancer/1"},
            )
            with urllib.request.urlopen(request, timeout=120.0) as response:
                with partial_path.open("wb") as stream:
                    shutil.copyfileobj(response, stream, length=16 * 1024 * 1024)
            if _sha256(partial_path) != LLAMA_CPP_G4_CACHE_SHA256:
                raise RuntimeError("Downloaded llama.cpp G4 cache checksum mismatch")
            partial_path.replace(archive_path)

        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)
        with tarfile.open(archive_path, "r:gz") as archive:
            members = archive.getmembers()
            _validate_cache_members(members)
            archive.extractall(staging, members=members)

        staged_build = staging / "llama.cpp-build"
        staged_marker = staging / "llama.cpp-build.json"
        staged_binary = staged_build / "bin" / "llama-server"
        if not staged_binary.is_file() or not staged_marker.is_file():
            raise RuntimeError("Downloaded llama.cpp G4 cache is incomplete")
        if json.loads(staged_marker.read_text(encoding="utf-8")) != expected:
            raise RuntimeError("Downloaded llama.cpp G4 cache has the wrong build marker")

        shutil.rmtree(build, ignore_errors=True)
        marker.unlink(missing_ok=True)
        shutil.move(str(staged_build), str(build))
        shutil.copy2(staged_marker, marker)
        binary = build / "bin" / "llama-server"
        _git_output(str(binary), "--version")
        print(
            "COMFYCOLAB_H3_CUDA_CACHE_RESTORED="
            + json.dumps(
                {
                    "release": LLAMA_CPP_G4_CACHE_TAG,
                    "sha256": LLAMA_CPP_G4_CACHE_SHA256,
                    "computeCapability": "12.0",
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return True
    except Exception as error:
        marker.unlink(missing_ok=True)
        shutil.rmtree(build, ignore_errors=True)
        print(
            "[comfycolab] Pinned G4 CUDA cache restore failed; falling back to "
            f"a local build: {error}",
            flush=True,
        )
        return False
    finally:
        partial_path.unlink(missing_ok=True)
        shutil.rmtree(staging, ignore_errors=True)


def ensure_llama_server() -> Path:
    override = os.environ.get("COMFYCOLAB_H3_PROMPT_LLAMA_SERVER")
    if override:
        path = Path(override).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(
                f"COMFYCOLAB_H3_PROMPT_LLAMA_SERVER does not point to a file: {path}"
            )
        return path

    runtime = _runtime_root()
    source = runtime / "llama.cpp"
    build = runtime / "llama.cpp-build"
    binary = build / "bin" / "llama-server"
    marker = runtime / "llama.cpp-build.json"
    expected = {"release": LLAMA_CPP_RELEASE, "revision": LLAMA_CPP_REVISION}

    with _INSTALL_LOCK:
        if marker.is_file() and binary.is_file():
            try:
                if json.loads(marker.read_text(encoding="utf-8")) == expected:
                    print(
                        f"[comfycolab] Reusing cached llama.cpp {LLAMA_CPP_RELEASE} "
                        "CUDA server.",
                        flush=True,
                    )
                    return binary
            except (OSError, json.JSONDecodeError):
                pass

        if _try_restore_g4_cache(runtime, build, marker, expected):
            return binary

        if not shutil.which("git") or not shutil.which("cmake"):
            raise RuntimeError(
                "The H3 Prompt Enhancer needs git and cmake to build pinned llama.cpp."
            )
        if not (source / ".git").is_dir():
            if source.exists() and any(source.iterdir()):
                raise RuntimeError(
                    f"llama.cpp cache exists but is not a git checkout: {source}"
                )
            source.parent.mkdir(parents=True, exist_ok=True)
            subprocess.check_call(
                [
                    "git",
                    "clone",
                    "--filter=blob:none",
                    LLAMA_CPP_REPOSITORY,
                    str(source),
                ]
            )
        actual = _git_output("git", "-C", str(source), "rev-parse", "HEAD")
        if actual != LLAMA_CPP_REVISION:
            dirty = _git_output("git", "-C", str(source), "status", "--porcelain")
            if dirty:
                raise RuntimeError(
                    "The cached llama.cpp checkout has local changes. Remove or relocate "
                    f"that cache before retrying: {source}"
                )
            subprocess.check_call(
                [
                    "git",
                    "-C",
                    str(source),
                    "fetch",
                    "origin",
                    LLAMA_CPP_REVISION,
                    "--depth",
                    "1",
                ]
            )
            subprocess.check_call(
                ["git", "-C", str(source), "checkout", "--detach", LLAMA_CPP_REVISION]
            )

        print(
            f"[comfycolab] Building pinned llama.cpp {LLAMA_CPP_RELEASE} with CUDA...",
            flush=True,
        )
        subprocess.check_call(
            [
                "cmake",
                "-S",
                str(source),
                "-B",
                str(build),
                "-DGGML_CUDA=ON",
                "-DLLAMA_CURL=OFF",
                "-DLLAMA_BUILD_TESTS=OFF",
                "-DLLAMA_BUILD_EXAMPLES=OFF",
                "-DLLAMA_BUILD_SERVER=ON",
                "-DGGML_NATIVE=ON",
                "-DCMAKE_BUILD_TYPE=Release",
            ]
        )
        subprocess.check_call(
            [
                "cmake",
                "--build",
                str(build),
                "--config",
                "Release",
                "--target",
                "llama-server",
                "-j",
                str(min(os.cpu_count() or 1, 12)),
            ]
        )
        if not binary.is_file():
            raise RuntimeError(f"Pinned llama.cpp build did not create {binary}")
        marker.write_text(json.dumps(expected, sort_keys=True) + "\n", encoding="utf-8")
        return binary


def build_server_argv(server: Path, model: Path, port: int) -> list[str]:
    return [
        str(server),
        "--model",
        str(model),
        "--alias",
        QWEN_MODEL_ALIAS,
        "--host",
        "127.0.0.1",
        "--port",
        str(int(port)),
        "--ctx-size",
        "32768",
        "--parallel",
        "1",
        "--batch-size",
        "512",
        "--ubatch-size",
        "128",
        "--n-gpu-layers",
        "all",
        "--flash-attn",
        "on",
        "--cache-type-k",
        "q8_0",
        "--cache-type-v",
        "q8_0",
        "--jinja",
        "--reasoning",
        "on",
        "--reasoning-format",
        "deepseek",
    ]


def _free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(("127.0.0.1", 0))
        return int(server.getsockname()[1])


def _tail(path: Path, limit: int = 12_000) -> str:
    if not path.is_file():
        return ""
    data = path.read_bytes()
    return data[-limit:].decode("utf-8", errors="replace")


def _wait_until_ready(
    process: subprocess.Popen[Any],
    port: int,
    log_path: Path,
    *,
    timeout: float = 900.0,
) -> None:
    deadline = time.monotonic() + timeout
    url = f"http://127.0.0.1:{port}/health"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                "Qwen llama.cpp server exited during model load.\n" + _tail(log_path)
            )
        try:
            with urllib.request.urlopen(url, timeout=2.0) as response:
                payload = json.loads(response.read().decode("utf-8"))
                if response.status == 200 and payload.get("status") == "ok":
                    return
        except (OSError, urllib.error.URLError, json.JSONDecodeError):
            pass
        time.sleep(0.5)
    raise TimeoutError(
        f"Qwen llama.cpp server did not become ready within {timeout:.0f} seconds.\n"
        + _tail(log_path)
    )


def build_chat_request_body(
    *,
    source_prompt: str,
    mode: str,
    duration_seconds: float,
    seed: int,
    max_tokens: int,
    temperature: float,
    validation_errors: list[str],
) -> dict[str, Any]:
    return {
        "model": QWEN_MODEL_ALIAS,
        "messages": [
            {"role": "system", "content": system_policy(mode, duration_seconds)},
            {
                "role": "user",
                "content": user_rewrite_request(
                    source_prompt,
                    mode,
                    duration_seconds,
                    validation_errors=validation_errors,
                ),
            },
        ],
        "temperature": float(temperature),
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "repeat_penalty": 1.0,
        "seed": int(seed),
        "max_tokens": int(max_tokens),
        "reasoning_budget_tokens": min(4096, max(1024, int(max_tokens) // 2)),
        "stream": False,
        "reasoning_effort": "xhigh",
        "chat_template_kwargs": {"enable_thinking": True},
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "minimax_h3_prompt_rewrite",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "enhanced_prompt": {"type": "string", "minLength": 1}
                    },
                    "required": ["enhanced_prompt"],
                    "additionalProperties": False,
                },
            },
        },
    }


def _chat_request(
    port: int,
    *,
    source_prompt: str,
    mode: str,
    duration_seconds: float,
    seed: int,
    max_tokens: int,
    temperature: float,
    validation_errors: list[str],
) -> dict[str, Any]:
    body = build_chat_request_body(
        source_prompt=source_prompt,
        mode=mode,
        duration_seconds=duration_seconds,
        seed=seed,
        max_tokens=max_tokens,
        temperature=temperature,
        validation_errors=validation_errors,
    )
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=600.0) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Qwen llama.cpp completion failed with HTTP {error.code}: {details}"
        ) from error


def extract_enhanced_prompt(response: dict[str, Any]) -> str:
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise RuntimeError("Qwen response omitted choices[0].message.content") from error
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Qwen returned an empty prompt rewrite")
    try:
        decoded = json.loads(content)
    except json.JSONDecodeError as error:
        raise RuntimeError("Qwen returned invalid structured JSON") from error
    enhanced = decoded.get("enhanced_prompt") if isinstance(decoded, dict) else None
    if not isinstance(enhanced, str) or not enhanced.strip():
        raise RuntimeError("Qwen structured response omitted enhanced_prompt")
    return enhanced.strip()


def _terminate(process: subprocess.Popen[Any], timeout: float = 10.0) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=timeout)


def enhance_h3_prompt(
    source_prompt: str,
    mode: str,
    duration_seconds: float,
    *,
    seed: int = 0,
    max_tokens: int = 8192,
    temperature: float = 1.0,
    force_redownload: bool = False,
) -> str:
    source = str(source_prompt).strip()
    if not source:
        raise ValueError("MiniMax H3 Prompt Enhancer requires a non-empty prompt.")
    normalized_mode = normalize_prompt_mode(mode)
    if int(seed) < 0 or int(seed) > (2**31) - 1:
        raise ValueError("MiniMax H3 Prompt Enhancer seed must be between 0 and 2147483647.")
    if int(max_tokens) < 4096 or int(max_tokens) > 16384:
        raise ValueError(
            "MiniMax H3 Prompt Enhancer max_tokens must be between 4096 and 16384."
        )
    if not 0.0 <= float(temperature) <= 1.5:
        raise ValueError("MiniMax H3 Prompt Enhancer temperature must be between 0 and 1.5.")

    server = ensure_llama_server()
    model = ensure_qwen_gguf(force_redownload=bool(force_redownload))
    port = _free_local_port()
    log_dir = _runtime_root() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"llama-server-{int(time.time())}.log"
    started = time.monotonic()
    validation_errors: list[str] = []

    print(
        "[comfycolab] Loading Qwen3.8-27B Q4_K_M in isolated llama.cpp server...",
        flush=True,
    )
    with log_path.open("w", encoding="utf-8") as log_stream:
        process = subprocess.Popen(
            build_server_argv(server, model, port),
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            _wait_until_ready(process, port, log_path)
            for attempt in range(1, 3):
                response = _chat_request(
                    port,
                    source_prompt=source,
                    mode=normalized_mode,
                    duration_seconds=float(duration_seconds),
                    seed=int(seed),
                    max_tokens=int(max_tokens),
                    temperature=float(temperature),
                    validation_errors=validation_errors,
                )
                enhanced = extract_enhanced_prompt(response)
                validation_errors = validate_enhanced_prompt(
                    enhanced,
                    normalized_mode,
                    float(duration_seconds),
                )
                if not validation_errors:
                    elapsed = time.monotonic() - started
                    print(
                        "COMFYCOLAB_H3_PROMPT_ENHANCER_RESULT="
                        + json.dumps(
                            {
                                "attempt": attempt,
                                "durationSeconds": round(elapsed, 3),
                                "llamaCppRelease": LLAMA_CPP_RELEASE,
                                "minimaxH3GuideRevision": MINIMAX_H3_GUIDE_REVISION,
                                "mode": normalized_mode,
                                "model": QWEN_GGUF_REPO,
                                "promptCharacters": len(enhanced),
                                "promptSha256": hashlib.sha256(
                                    enhanced.encode("utf-8")
                                ).hexdigest(),
                                "quant": "Q4_K_M",
                                "thinking": True,
                                "validated": True,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                    return enhanced
            raise RuntimeError(
                "Qwen3.8 prompt rewrite failed MiniMax H3 validation after two "
                "attempts: "
                + "; ".join(validation_errors)
            )
        except Exception as error:
            details = _tail(log_path)
            if details:
                error.add_note("llama.cpp log tail:\n" + details)
            raise
        finally:
            _terminate(process)
            print(
                "[comfycolab] Qwen prompt server stopped; GPU memory released for H3.",
                flush=True,
            )
