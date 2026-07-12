from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from .bootstrap import render_bootstrap
from .colab import ColabClient, parse_ready_payload


DEFAULT_CONFIG = Path("~/.config/comfycolab/colab-sessions.json")
DEFAULT_STATE = Path("~/.config/comfycolab/runtime.json")


def _git_remote(directory: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(directory), "remote", "get-url", "origin"],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip()
    return None


def resolve_repository_url(explicit: str | None) -> str:
    if explicit:
        return explicit
    if configured := os.environ.get("COMFYCOLAB_REPO_URL"):
        return configured

    candidates = [Path.cwd(), Path(__file__).resolve().parents[2]]
    for candidate in candidates:
        if remote := _git_remote(candidate):
            return remote

    raise RuntimeError(
        "Unable to determine this repository's public clone URL. Add an 'origin' Git "
        "remote or pass --repo-url."
    )


def _state_path(value: str) -> Path:
    return Path(value).expanduser()


def _write_state(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_state(path: Path) -> dict[str, object] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _client(args: argparse.Namespace) -> ColabClient:
    return ColabClient.create(
        executable=args.colab_bin,
        auth=args.auth,
        config_path=Path(args.config),
    )


def _start(args: argparse.Namespace) -> int:
    repository_url = resolve_repository_url(args.repo_url)
    client = _client(args)
    created = False

    try:
        if args.auth == "oauth2":
            client.authenticate_interactively()
        if not client.session_exists(args.session):
            client.new(args.session, args.gpu)
            created = True
        else:
            print(f"[comfycolab] Reusing active Colab session '{args.session}'.")

        if args.colab_proxy:
            print("[comfycolab] Opening the attached Colab page for the private proxy handshake...")
            try:
                result = client.open_url(args.session)
                if result.stdout:
                    print(result.stdout, end="")
            except Exception as error:
                print(
                    f"[comfycolab] Could not open the Colab page ({error}); "
                    "Cloudflare fallback will still be used.",
                    file=sys.stderr,
                )

        source = render_bootstrap(
            repository_url=repository_url,
            repository_ref=args.repo_ref,
            port=args.port,
            refresh=args.refresh,
            colab_proxy=args.colab_proxy,
        )
        result = client.exec_bootstrap(
            session=args.session,
            source=source,
            remote_timeout=args.bootstrap_timeout,
        )
        payload = parse_ready_payload(result.stdout)
        payload["session"] = args.session
        payload["gpu"] = args.gpu
        _write_state(_state_path(args.state), payload)
    except Exception:
        if created:
            client.stop(args.session)
        raise

    print(f"\nComfyUI: {payload['comfyUrl']}")
    print(f"Session: {args.session}")
    if payload.get("colabProxyUrl") and payload.get("cloudflareUrl"):
        print(f"Cloudflare fallback: {payload['cloudflareUrl']}")
    return 0


def _status(args: argparse.Namespace) -> int:
    result = _client(args).status(args.session)
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="")
    if result.returncode != 0:
        return result.returncode
    active = f"[{args.session}]" in result.stdout
    state = _read_state(_state_path(args.state))
    if active and state and state.get("session") == args.session and state.get("comfyUrl"):
        print(f"ComfyUI: {state['comfyUrl']}")
    elif not active and state and state.get("session") == args.session:
        _state_path(args.state).unlink(missing_ok=True)
    return 0 if active else 1


def _stop(args: argparse.Namespace) -> int:
    result = _client(args).stop(args.session)
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="")
    _state_path(args.state).unlink(missing_ok=True)
    return result.returncode


def _url(args: argparse.Namespace) -> int:
    state = _read_state(_state_path(args.state))
    if not state or state.get("session") != args.session or not state.get("comfyUrl"):
        print(f"No saved ComfyUI URL for session '{args.session}'.", file=sys.stderr)
        return 1
    print(state["comfyUrl"])
    return 0


def _add_connection_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--auth",
        choices=("oauth2", "adc"),
        default=os.environ.get("COMFYCOLAB_AUTH", "adc"),
        help="Authentication strategy passed to google-colab-cli.",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="google-colab-cli session-state file.",
    )
    parser.add_argument(
        "--colab-bin",
        default=os.environ.get("COMFYCOLAB_COLAB_BIN"),
        help="Path to the colab executable; defaults to PATH lookup.",
    )
    parser.add_argument("--session", default="comfycolab")
    parser.add_argument("--state", default=str(DEFAULT_STATE))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="comfycolab",
        description="Run ComfyUI with curated temporary model bundles on Google Colab.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    start = subparsers.add_parser("start", help="Start Colab, bootstrap ComfyUI, and print its URL.")
    _add_connection_options(start)
    start.add_argument("--gpu", choices=("T4", "L4", "G4", "A100", "H100"), default="G4")
    start.add_argument("--repo-url", help="Public clone URL for this repository.")
    start.add_argument("--repo-ref", default="main")
    start.add_argument("--port", type=int, default=8188)
    start.add_argument("--bootstrap-timeout", type=int, default=1800)
    start.add_argument(
        "--refresh",
        action="store_true",
        help="Update the remote repositories and restart managed UI processes.",
    )
    start.add_argument(
        "--colab-proxy",
        action="store_true",
        help="Prefer a private Google Colab proxy URL and retain Cloudflare as fallback.",
    )
    start.set_defaults(handler=_start)

    status = subparsers.add_parser("status", help="Show the Colab session and saved ComfyUI URL.")
    _add_connection_options(status)
    status.set_defaults(handler=_status)

    stop = subparsers.add_parser("stop", help="Stop the Colab session.")
    _add_connection_options(stop)
    stop.set_defaults(handler=_stop)

    url = subparsers.add_parser("url", help="Print the most recently saved ComfyUI URL.")
    url.add_argument("--session", default="comfycolab")
    url.add_argument("--state", default=str(DEFAULT_STATE))
    url.set_defaults(handler=_url)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"[comfycolab] {error}", file=sys.stderr)
        return 1
