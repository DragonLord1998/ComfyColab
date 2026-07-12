"""Executed by google-colab-cli inside a temporary Colab runtime."""

from __future__ import annotations

import base64
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


CONFIG_B64 = "__COMFYCOLAB_CONFIG_B64__"
DEFAULT_CONFIG = {
    "repository_url": "https://github.com/DragonLord1998/ComfyColab.git",
    "repository_ref": "main",
    "port": 8188,
    "refresh": False,
}
CONFIG = (
    DEFAULT_CONFIG
    if CONFIG_B64.startswith("__COMFYCOLAB_CONFIG_")
    else json.loads(base64.b64decode(CONFIG_B64).decode("utf-8"))
)

CONTENT = Path("/content")
COMFY_DIR = CONTENT / "ComfyUI"
REPO_DIR = CONTENT / "ComfyColab"
STATE_DIR = CONTENT / ".comfycolab"
STATE_FILE = STATE_DIR / "runtime.json"
COMFY_LOG = STATE_DIR / "comfyui.log"
TUNNEL_LOG = STATE_DIR / "cloudflared.log"
GGUF_DIR = COMFY_DIR / "custom_nodes" / "ComfyUI-GGUF"
NODE_TARGET = COMFY_DIR / "custom_nodes" / "ComfyColab-ZImage"
READY_PREFIX = "COMFYCOLAB_READY="
COMFY_REF = "8b099de36acd81acd1afa3b5442951dc847e0a52"
GGUF_REF = "6ea2651e7df66d7585f6ffee804b20e92fb38b8a"


def run(command: list[str], *, cwd: Path | None = None) -> None:
    print(f"[comfycolab] $ {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def clone_or_update(url: str, destination: Path, ref: str = "main") -> None:
    if not (destination / ".git").is_dir():
        if destination.exists():
            shutil.rmtree(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--filter=blob:none",
                "--no-checkout",
                url,
                str(destination),
            ]
        )

    run(["git", "fetch", "origin", ref, "--depth", "1"], cwd=destination)
    run(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=destination)


def git_commit(destination: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=destination,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def install_dependencies() -> None:
    run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], cwd=COMFY_DIR)
    requirements = GGUF_DIR / "requirements.txt"
    if requirements.exists():
        run([sys.executable, "-m", "pip", "install", "-r", str(requirements)])


def install_node_pack() -> None:
    source = REPO_DIR / "custom_nodes" / "ComfyColab-ZImage"
    if not source.is_dir():
        raise RuntimeError(f"Node pack is missing from repository: {source}")

    if NODE_TARGET.is_symlink() or NODE_TARGET.is_file():
        NODE_TARGET.unlink()
    elif NODE_TARGET.exists():
        shutil.rmtree(NODE_TARGET)
    NODE_TARGET.symlink_to(source, target_is_directory=True)


def cloudflared_path() -> Path:
    existing = shutil.which("cloudflared")
    if existing:
        return Path(existing)

    machine = platform.machine().lower()
    architecture = "arm64" if machine in {"aarch64", "arm64"} else "amd64"
    destination = STATE_DIR / "cloudflared"
    url = (
        "https://github.com/cloudflare/cloudflared/releases/latest/download/"
        f"cloudflared-linux-{architecture}"
    )
    print(f"[comfycolab] Downloading cloudflared ({architecture})...", flush=True)
    with urllib.request.urlopen(url, timeout=120) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output)
    destination.chmod(0o755)
    return destination


def pid_alive(pid: object) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        waited, _ = os.waitpid(pid, os.WNOHANG)
        if waited == pid:
            return False
    except ChildProcessError:
        pass
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def stop_managed_process(pid: object) -> None:
    if not pid_alive(pid):
        return
    process_id = int(pid)
    try:
        process_group = os.getpgid(process_id)
        os.killpg(process_group, signal.SIGTERM)
    except OSError:
        try:
            os.kill(process_id, signal.SIGTERM)
        except OSError:
            return
    for _ in range(20):
        if not pid_alive(pid):
            return
        time.sleep(0.25)
    try:
        os.killpg(os.getpgid(process_id), signal.SIGKILL)
    except OSError:
        try:
            os.kill(process_id, signal.SIGKILL)
        except OSError:
            pass
    for _ in range(20):
        if not pid_alive(pid):
            return
        time.sleep(0.1)
    raise RuntimeError(f"Managed process {process_id} did not terminate.")


def stop_started_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        process.wait()
        return
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    except OSError:
        process.terminate()
    try:
        process.wait(timeout=5)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except OSError:
        process.kill()
    process.wait(timeout=2)


def http_ready(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/object_info", timeout=2) as response:
            return 200 <= response.status < 500
    except Exception:
        return False


def wait_for_comfy(port: int, process: subprocess.Popen[bytes], timeout: int = 180) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            tail = COMFY_LOG.read_text(encoding="utf-8", errors="replace")[-4000:]
            raise RuntimeError(f"ComfyUI exited during startup.\n{tail}")
        if http_ready(port):
            return
        time.sleep(1)
    raise TimeoutError(f"ComfyUI did not become ready on port {port} within {timeout}s.")


def wait_for_tunnel(process: subprocess.Popen[bytes], timeout: int = 60) -> str:
    pattern = re.compile(r"https://[-a-z0-9]+\.trycloudflare\.com")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if TUNNEL_LOG.exists():
            content = TUNNEL_LOG.read_text(encoding="utf-8", errors="replace")
            if match := pattern.search(content):
                return match.group(0)
        if process.poll() is not None:
            tail = TUNNEL_LOG.read_text(encoding="utf-8", errors="replace")[-4000:]
            raise RuntimeError(f"cloudflared exited during startup.\n{tail}")
        time.sleep(0.5)
    raise TimeoutError("cloudflared did not publish a trycloudflare.com URL.")


def load_state() -> dict[str, object]:
    try:
        value = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def save_state(payload: dict[str, object]) -> None:
    temporary = STATE_FILE.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(STATE_FILE)


def emit_ready(payload: dict[str, object]) -> None:
    print(READY_PREFIX + json.dumps(payload, separators=(",", ":")), flush=True)


def main() -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    port = int(CONFIG["port"])
    refresh = bool(CONFIG.get("refresh", False))
    previous = load_state()

    if (
        not refresh
        and http_ready(port)
        and pid_alive(previous.get("comfyPid"))
        and pid_alive(previous.get("tunnelPid"))
        and previous.get("comfyUrl")
    ):
        emit_ready(previous)
        return

    stop_managed_process(previous.get("tunnelPid"))
    stop_managed_process(previous.get("comfyPid"))
    STATE_FILE.unlink(missing_ok=True)
    if http_ready(port):
        raise RuntimeError(
            f"Port {port} is already occupied by a process not managed by ComfyColab."
        )

    clone_or_update("https://github.com/Comfy-Org/ComfyUI.git", COMFY_DIR, COMFY_REF)
    clone_or_update("https://github.com/city96/ComfyUI-GGUF.git", GGUF_DIR, GGUF_REF)
    clone_or_update(
        str(CONFIG["repository_url"]),
        REPO_DIR,
        str(CONFIG["repository_ref"]),
    )
    install_node_pack()
    install_dependencies()

    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    comfy: subprocess.Popen[bytes] | None = None
    tunnel: subprocess.Popen[bytes] | None = None
    ready = False
    try:
        with COMFY_LOG.open("wb") as comfy_log:
            comfy = subprocess.Popen(
                [
                    sys.executable,
                    "main.py",
                    "--listen",
                    "127.0.0.1",
                    "--port",
                    str(port),
                ],
                cwd=COMFY_DIR,
                stdout=comfy_log,
                stderr=subprocess.STDOUT,
                env=environment,
                start_new_session=True,
            )
        save_state({"status": "starting_comfy", "comfyPid": comfy.pid, "port": port})
        wait_for_comfy(port, comfy)

        cloudflared = cloudflared_path()
        with TUNNEL_LOG.open("wb") as tunnel_log:
            tunnel = subprocess.Popen(
                [
                    str(cloudflared),
                    "tunnel",
                    "--url",
                    f"http://127.0.0.1:{port}",
                    "--no-autoupdate",
                ],
                stdout=tunnel_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        save_state(
            {
                "status": "starting_tunnel",
                "comfyPid": comfy.pid,
                "tunnelPid": tunnel.pid,
                "port": port,
            }
        )
        url = wait_for_tunnel(tunnel)

        payload: dict[str, object] = {
            "status": "ready",
            "comfyUrl": url,
            "comfyPid": comfy.pid,
            "tunnelPid": tunnel.pid,
            "port": port,
            "storage": "temporary",
            "repositoryUrl": CONFIG["repository_url"],
            "repositoryRef": CONFIG["repository_ref"],
            "repositoryCommit": git_commit(REPO_DIR),
            "comfyCommit": git_commit(COMFY_DIR),
            "ggufCommit": git_commit(GGUF_DIR),
        }
        save_state(payload)
        ready = True
        emit_ready(payload)
    finally:
        if not ready:
            if tunnel is not None:
                stop_started_process(tunnel)
            if comfy is not None:
                stop_started_process(comfy)
            STATE_FILE.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
