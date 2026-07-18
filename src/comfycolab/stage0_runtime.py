"""Standard-library-only authenticated stage-0 bootstrap.

This file is rendered and sent to Colab. Keep imports in the Python standard
library and keep all pack/model behavior in the authenticated stage-1 module.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import shutil
import subprocess
import sys
import os
from collections import deque
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse


CONFIG_B64 = "__COMFYCOLAB_STAGE0_CONFIG_B64__"
CONTENT = Path("/content")
CORE_DIR = CONTENT / "ComfyColab"
STATE_DIR = CONTENT / ".comfycolab"
LOCK_FILE = STATE_DIR / "lock.json"
STAGE1_CONFIG_FILE = STATE_DIR / "stage1-config.json"
STAGE1_LOG_FILE = STATE_DIR / "stage1.log"
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def fail(message: str) -> RuntimeError:
    return RuntimeError(f"[comfycolab:stage0] {message}")


def canonical_json_bytes(payload: dict[str, object]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def decode_config(encoded: str) -> dict[str, object]:
    if encoded.startswith("__COMFYCOLAB_STAGE0_CONFIG_"):
        raise fail("configuration marker was not replaced")
    try:
        payload = json.loads(base64.b64decode(encoded, validate=True))
    except Exception as error:
        raise fail("configuration is not valid base64 JSON") from error
    if not isinstance(payload, dict):
        raise fail("configuration must be an object")
    expected = {
        "schema",
        "core_repository",
        "core_commit",
        "stage1_entrypoint",
        "stage1_sha256",
        "lock_b64",
        "lock_sha256",
        "port",
        "refresh",
        "colab_proxy",
        "accepted_licenses",
    }
    if set(payload) != expected:
        raise fail("configuration fields do not match CoreStage0ConfigV1")
    if payload["schema"] != 1:
        raise fail("unsupported configuration schema")
    repository = payload["core_repository"]
    if not isinstance(repository, str):
        raise fail("core_repository must be a string")
    parsed = urlparse(repository)
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise fail("core_repository must be a credential-free HTTPS URL")
    commit = payload["core_commit"]
    if not isinstance(commit, str) or not COMMIT_RE.fullmatch(commit):
        raise fail("core_commit must be an immutable 40-character commit")
    entrypoint = payload["stage1_entrypoint"]
    if not isinstance(entrypoint, str):
        raise fail("stage1_entrypoint must be a string")
    path = PurePosixPath(entrypoint)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise fail("stage1_entrypoint is unsafe")
    for field in ("stage1_sha256", "lock_sha256"):
        value = payload[field]
        if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
            raise fail(f"{field} must be a SHA-256 digest")
    if type(payload["port"]) is not int or not 1 <= payload["port"] <= 65535:
        raise fail("port is invalid")
    if type(payload["refresh"]) is not bool or type(payload["colab_proxy"]) is not bool:
        raise fail("refresh and colab_proxy must be booleans")
    accepted = payload["accepted_licenses"]
    if (
        not isinstance(accepted, list)
        or accepted != sorted(set(accepted))
        or any(
            not isinstance(item, str)
            or not re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,127}", item)
            for item in accepted
        )
    ):
        raise fail("accepted_licenses is invalid")
    return payload


def run(
    command: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    print(f"[comfycolab:stage0] $ {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def run_stage1(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
) -> None:
    """Stream stage-1 output while retaining an actionable failure tail."""

    print(f"[comfycolab:stage0] $ {' '.join(command)}", flush=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    tail: deque[str] = deque(maxlen=200)
    process: subprocess.Popen[str] | None = None
    try:
        with STAGE1_LOG_FILE.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                cwd=cwd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            if process.stdout is None:
                process.kill()
                raise fail("stage-1 output pipe was not created")
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
                tail.append(line)
            return_code = process.wait()
    except BaseException as error:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if isinstance(error, OSError):
            raise fail(f"stage-1 launch/output failed: {error}") from error
        raise

    if return_code == 0:
        return
    output_tail = "".join(tail)[-12000:].strip()
    detail = (
        "\n--- stage-1 output tail ---\n" + output_tail
        if output_tail
        else "\nStage-1 produced no output."
    )
    raise fail(
        f"stage-1 exited with status {return_code} under Python "
        f"{sys.version.split()[0]}. Full output: {STAGE1_LOG_FILE}.{detail}"
    )


def clone_authenticated_core(repository: str, commit: str) -> None:
    if CORE_DIR.exists():
        shutil.rmtree(CORE_DIR)
    run(
        [
            "git",
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            repository,
            str(CORE_DIR),
        ]
    )
    run(["git", "fetch", "origin", commit, "--depth", "1"], cwd=CORE_DIR)
    run(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=CORE_DIR)
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=CORE_DIR,
        check=True,
        text=True,
        capture_output=True,
    )
    if result.stdout.strip() != commit:
        raise fail("authenticated core checkout does not match core_commit")


def verify_file(path: Path, expected_sha256: str) -> None:
    if not path.is_file():
        raise fail(f"stage-1 entrypoint is missing: {path}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected_sha256:
        raise fail("stage-1 entrypoint digest mismatch")


def write_lock(config: dict[str, object]) -> None:
    try:
        lock_bytes = base64.b64decode(str(config["lock_b64"]), validate=True)
    except ValueError as error:
        raise fail("embedded lock is not valid base64") from error
    actual = hashlib.sha256(lock_bytes).hexdigest()
    if actual != config["lock_sha256"]:
        raise fail("embedded lock digest mismatch")
    try:
        lock = json.loads(lock_bytes)
    except json.JSONDecodeError as error:
        raise fail("embedded lock is not valid JSON") from error
    if not isinstance(lock, dict) or lock.get("schema") != 1:
        raise fail("embedded lock is not ComfyColabLockV1")
    if canonical_json_bytes(lock) != lock_bytes:
        raise fail("embedded lock is not canonical JSON")
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    temporary = LOCK_FILE.with_suffix(".tmp")
    temporary.write_bytes(lock_bytes)
    temporary.replace(LOCK_FILE)


def main() -> None:
    config = decode_config(CONFIG_B64)
    clone_authenticated_core(
        str(config["core_repository"]),
        str(config["core_commit"]),
    )
    entrypoint = CORE_DIR / str(config["stage1_entrypoint"])
    verify_file(entrypoint, str(config["stage1_sha256"]))
    write_lock(config)
    stage1_config = {
        "schema": 1,
        "port": config["port"],
        "refresh": config["refresh"],
        "colab_proxy": config["colab_proxy"],
        "lock_path": str(LOCK_FILE),
        "lock_sha256": config["lock_sha256"],
        "core_dir": str(CORE_DIR),
        "accepted_licenses": config["accepted_licenses"],
    }
    STAGE1_CONFIG_FILE.write_bytes(canonical_json_bytes(stage1_config))
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(CORE_DIR / "src")
    run_stage1(
        [
            sys.executable,
            "-m",
            "comfycolab.runtime",
            "--config",
            str(STAGE1_CONFIG_FILE),
        ],
        cwd=CORE_DIR / "src",
        env=environment,
    )


if __name__ == "__main__":
    main()
