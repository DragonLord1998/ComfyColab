from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

from .colab import ColabClient, parse_ready_payload
from .artifacts import sync_glb_artifacts, watch_glb_artifacts
from .notebook import RuntimeResolvedMainNotebookConfig, write_notebook
from .notebook import write_runtime_resolved_main_notebook
from .packs.io import load_registry
from .packs.lock import ComfyColabLockV1
from .repositories import temporary_checkout
from .resolution import PreparedLaunch, prepare_launch, prepare_launch_from_lock
from .runtime import (
    validate_legacy_full_lock,
    validate_pack_license_gates,
    validate_runtime_support,
)
from .state import (
    normalize_runtime_state,
    read_runtime_state,
    verify_lock_digest,
    write_runtime_state,
)


DEFAULT_CONFIG = Path("~/.config/comfycolab/colab-sessions.json")
DEFAULT_STATE = Path("~/.config/comfycolab/runtime.json")
DEFAULT_STATE_DIR = Path("~/.config/comfycolab")
DEFAULT_LOCK_DIR = Path("~/.config/comfycolab/locks")
DEFAULT_REPOSITORY_URL = "https://github.com/DragonLord1998/ComfyColab.git"


def _default_config_path() -> str:
    return os.environ.get("COMFYCOLAB_CONFIG", str(DEFAULT_CONFIG))


def _default_state_path() -> str:
    if explicit := os.environ.get("COMFYCOLAB_STATE"):
        return explicit
    if directory := os.environ.get("COMFYCOLAB_STATE_DIR"):
        return str(Path(directory) / "runtime.json")
    return str(DEFAULT_STATE)


def resolve_repository_url(explicit: str | None) -> str:
    if explicit:
        return explicit
    if configured := os.environ.get("COMFYCOLAB_REPO_URL"):
        return configured
    return DEFAULT_REPOSITORY_URL


def _state_path(value: str) -> Path:
    return Path(value).expanduser()


def _lock_path(args: argparse.Namespace) -> Path:
    return Path(args.lock_dir).expanduser() / f"{args.session}.lock.json"


def _previous_lock_path(path: Path) -> Path:
    suffix = ".lock.json"
    if not path.name.endswith(suffix):
        raise ValueError(f"lock path must end with {suffix}: {path}")
    session = path.name[: -len(suffix)]
    return path.with_name(f"{session}.previous.lock.json")


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _write_lock(path: Path, lock: ComfyColabLockV1) -> None:
    payload = lock.canonical_bytes()
    if path.is_file():
        current = path.read_bytes()
        ComfyColabLockV1.from_bytes(current)
        if current != payload:
            _atomic_write_bytes(_previous_lock_path(path), current)
    _atomic_write_bytes(path, payload)


def _write_state(path: Path, payload: dict[str, object]) -> None:
    write_runtime_state(path, payload)


def _read_state(path: Path) -> dict[str, object] | None:
    return read_runtime_state(path)


def _client(args: argparse.Namespace) -> ColabClient:
    return ColabClient.create(
        executable=args.colab_bin,
        auth=args.auth,
        config_path=Path(args.config),
    )


def _prepare_launch(args: argparse.Namespace) -> PreparedLaunch:
    runtime_mode = "legacy-full" if getattr(args, "legacy_full", False) else "generic"
    prepared: PreparedLaunch | None = None
    if args.refresh:
        existing = _lock_path(args)
        if existing.is_file():
            lock = ComfyColabLockV1.from_bytes(existing.read_bytes())
            prepared = prepare_launch_from_lock(
                lock,
                port=args.port,
                refresh=True,
                colab_proxy=args.colab_proxy,
                runtime_mode=runtime_mode,
                accepted_licenses=args.accept_license,
            )
    if prepared is None:
        prepared = prepare_launch(
            core_repository=resolve_repository_url(args.repo_url),
            core_ref=args.repo_ref,
            pack_aliases=args.pack,
            profile=args.profile,
            pack_ref_files=[Path(path).expanduser() for path in args.pack_ref],
            port=args.port,
            refresh=args.refresh,
            colab_proxy=args.colab_proxy,
            runtime_mode=runtime_mode,
            accepted_licenses=args.accept_license,
        )
    if runtime_mode == "legacy-full":
        validate_legacy_full_lock(prepared.lock.to_dict())
    return prepared


def _start(args: argparse.Namespace) -> int:
    prepared = _prepare_launch(args)
    lock_payload = prepared.lock.to_dict()
    validate_runtime_support(lock_payload)
    validate_pack_license_gates(
        lock_payload,
        accepted_licenses=set(args.accept_license),
    )
    _write_lock(_lock_path(args), prepared.lock)
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

        result = client.exec_bootstrap(
            session=args.session,
            source=prepared.source,
            remote_timeout=args.bootstrap_timeout,
        )
        payload = parse_ready_payload(result.stdout)
        verify_lock_digest(payload, prepared.lock.sha256)
        payload = normalize_runtime_state(
            payload,
            session=args.session,
            gpu=args.gpu,
        )
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


def _pack_list(args: argparse.Namespace) -> int:
    repository_url = resolve_repository_url(args.repo_url)
    with temporary_checkout(repository_url, args.repo_ref) as (checkout, commit):
        registry = load_registry(checkout / "registry" / "official-packs.json")
    print(f"Core commit: {commit}")
    if not registry.packs:
        print("No runtime-approved daughter packs are registered.")
        return 0
    for alias, pack_ref in registry.packs.items():
        print(f"{alias}\t{pack_ref.id}\t{pack_ref.ref}\t{pack_ref.repository}")
    return 0


def _pack_resolve(args: argparse.Namespace) -> int:
    prepared = _prepare_launch(args)
    path = _lock_path(args)
    _write_lock(path, prepared.lock)
    if args.output == "json":
        print(prepared.lock.canonical_bytes().decode("utf-8"))
    else:
        print(f"Lock: {path}")
        print(f"SHA-256: {prepared.lock.sha256}")
        print(f"Packs: {', '.join(item['id'] for item in prepared.lock.packs) or 'none'}")
    return 0


def _pack_doctor(args: argparse.Namespace) -> int:
    path = _lock_path(args)
    try:
        lock = ComfyColabLockV1.from_bytes(path.read_bytes())
    except FileNotFoundError:
        print(f"No saved lock for session '{args.session}'.", file=sys.stderr)
        return 1
    state = _read_state(_state_path(args.state))
    print(f"Lock: {path}")
    print(f"SHA-256: {lock.sha256}")
    print(f"Packs: {', '.join(item['id'] for item in lock.packs) or 'none'}")
    previous = _previous_lock_path(path)
    if previous.is_file():
        previous_lock = ComfyColabLockV1.from_bytes(previous.read_bytes())
        print(f"Rollback SHA-256: {previous_lock.sha256}")
    if state is None:
        print("Runtime: no saved state")
        return 0
    actual = state.get("lockSha256")
    if actual is not None and actual != lock.sha256:
        print(
            f"Runtime: lock mismatch (saved state has {actual})",
            file=sys.stderr,
        )
        return 1
    print(f"Runtime: {state.get('status', 'unknown')}")
    return 0


def _pack_rollback(args: argparse.Namespace) -> int:
    current_path = _lock_path(args)
    previous_path = _previous_lock_path(current_path)
    try:
        current_bytes = current_path.read_bytes()
        previous_bytes = previous_path.read_bytes()
    except FileNotFoundError:
        print(
            f"No rollback pair is available for session '{args.session}'.",
            file=sys.stderr,
        )
        return 1
    current = ComfyColabLockV1.from_bytes(current_bytes)
    previous = ComfyColabLockV1.from_bytes(previous_bytes)
    _atomic_write_bytes(previous_path, current.canonical_bytes())
    _atomic_write_bytes(current_path, previous.canonical_bytes())
    print(f"Current lock: {current_path}")
    print(f"Restored SHA-256: {previous.sha256}")
    print(f"Previous SHA-256: {current.sha256}")
    print("Run `comfycolab start --refresh` to apply the restored lock.")
    return 0


def _render_notebook(args: argparse.Namespace) -> int:
    if args.runtime_resolve_main:
        if args.repo_ref != "main":
            raise ValueError("--runtime-resolve-main always resolves the public main ref")
        if args.pack_ref:
            raise ValueError("--runtime-resolve-main does not support local --pack-ref paths")
        output = Path(args.output).expanduser().resolve()
        config = RuntimeResolvedMainNotebookConfig.create(
            core_repository=resolve_repository_url(args.repo_url),
            profile=args.profile,
            pack_aliases=args.pack,
            port=args.port,
            refresh=args.refresh,
            colab_proxy=args.colab_proxy,
            runtime_mode="legacy-full" if args.legacy_full else "generic",
            accepted_licenses=args.accept_license,
        )
        write_runtime_resolved_main_notebook(output, config)
        print(f"Notebook: {output}")
        print("Core ref: main (resolved inside Colab at runtime)")
        print("Lock: generated inside Colab after main resolves to an immutable commit")
        return 0
    prepared = _prepare_launch(args)
    lock_path = _lock_path(args)
    _write_lock(lock_path, prepared.lock)
    output = Path(args.output).expanduser().resolve()
    write_notebook(output, prepared.config)
    print(f"Notebook: {output}")
    print(f"Lock: {lock_path}")
    print(f"SHA-256: {prepared.lock.sha256}")
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


def _artifacts(args: argparse.Namespace) -> int:
    state = _read_state(_state_path(args.state))
    if not state or state.get("session") != args.session or not state.get("comfyUrl"):
        print(f"No saved ComfyUI URL for session '{args.session}'.", file=sys.stderr)
        return 1
    output = Path(args.output_dir).expanduser().resolve()
    if args.watch:
        print(f"[comfycolab] Watching GLB stages -> {output}", flush=True)
        watch_glb_artifacts(
            str(state["comfyUrl"]),
            output,
            interval=args.interval,
            on_saved=lambda path: print(f"[comfycolab] Saved: {path}", flush=True),
            on_error=lambda error: print(
                f"[comfycolab] Artifact sync retry: {error}",
                file=sys.stderr,
                flush=True,
            ),
        )
        return 0
    saved = sync_glb_artifacts(str(state["comfyUrl"]), output)
    for path in saved:
        print(f"Saved: {path}")
    if not saved:
        print("No new GLB artifacts.")
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
        default=_default_config_path(),
        help="google-colab-cli session-state file.",
    )
    parser.add_argument(
        "--colab-bin",
        default=os.environ.get("COMFYCOLAB_COLAB_BIN"),
        help="Path to the colab executable; defaults to PATH lookup.",
    )
    parser.add_argument("--session", default=os.environ.get("COMFYCOLAB_SESSION", "comfycolab"))
    parser.add_argument("--state", default=_default_state_path())


def _add_selection_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo-url", help="Public clone URL for the ComfyColab core repository.")
    parser.add_argument(
        "--repo-ref",
        default=os.environ.get("COMFYCOLAB_REPO_REF", "main"),
        help="Core author ref; it is resolved to an immutable commit before launch.",
    )
    parser.add_argument(
        "--profile",
        default=os.environ.get("COMFYCOLAB_PROFILE", "core"),
        help="Authenticated profile name or local profile JSON path.",
    )
    parser.add_argument(
        "--pack",
        action="append",
        default=[],
        help="Official daughter-pack alias; repeat to compose packs.",
    )
    parser.add_argument(
        "--pack-ref",
        action="append",
        default=[],
        help="Path to an explicit immutable PackRefV1 JSON file.",
    )
    parser.add_argument(
        "--accept-license",
        action="append",
        default=[],
        help="Explicitly accept a manifest-declared license gate.",
    )
    parser.add_argument(
        "--lock-dir",
        default=os.environ.get("COMFYCOLAB_LOCK_DIR", str(DEFAULT_LOCK_DIR)),
        help="Directory for immutable per-session lock files.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="comfycolab",
        description="Run pinned ComfyUI and optional daughter packs on Google Colab.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    start = subparsers.add_parser("start", help="Start Colab, bootstrap ComfyUI, and print its URL.")
    _add_connection_options(start)
    _add_selection_options(start)
    start.add_argument(
        "--gpu",
        choices=("T4", "L4", "G4", "A100", "H100"),
        default=os.environ.get("COMFYCOLAB_GPU", "G4"),
    )
    start.add_argument("--port", type=int, default=int(os.environ.get("COMFYCOLAB_PORT", "8188")))
    start.add_argument(
        "--bootstrap-timeout",
        type=int,
        default=int(os.environ.get("COMFYCOLAB_BOOTSTRAP_TIMEOUT", "1800")),
    )
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
    url.add_argument("--session", default=os.environ.get("COMFYCOLAB_SESSION", "comfycolab"))
    url.add_argument("--state", default=_default_state_path())
    url.set_defaults(handler=_url)

    artifacts = subparsers.add_parser(
        "artifacts",
        help="Download generated GLB stage artifacts from Colab to this Mac.",
    )
    artifacts.add_argument(
        "--session", default=os.environ.get("COMFYCOLAB_SESSION", "comfycolab")
    )
    artifacts.add_argument("--state", default=_default_state_path())
    artifacts.add_argument(
        "--output-dir",
        default=os.environ.get(
            "COMFYCOLAB_ARTIFACT_DIR",
            str(Path("~/Documents/ComfyColab-Meshes").expanduser()),
        ),
    )
    artifacts.add_argument("--watch", action="store_true")
    artifacts.add_argument("--interval", type=float, default=3.0)
    artifacts.set_defaults(handler=_artifacts)

    pack = subparsers.add_parser("pack", help="Inspect and resolve daughter packs.")
    pack_subparsers = pack.add_subparsers(dest="pack_command", required=True)

    pack_list = pack_subparsers.add_parser("list", help="List authenticated official packs.")
    pack_list.add_argument("--repo-url", help="Public clone URL for the ComfyColab core repository.")
    pack_list.add_argument(
        "--repo-ref",
        default=os.environ.get("COMFYCOLAB_REPO_REF", "main"),
    )
    pack_list.set_defaults(handler=_pack_list)

    for command, help_text in (
        ("resolve", "Resolve a profile/selection and persist its immutable lock."),
        ("update", "Explicitly resolve current aliases/profile into a new lock."),
    ):
        resolve = pack_subparsers.add_parser(command, help=help_text)
        _add_selection_options(resolve)
        resolve.add_argument("--session", default=os.environ.get("COMFYCOLAB_SESSION", "comfycolab"))
        resolve.add_argument("--port", type=int, default=int(os.environ.get("COMFYCOLAB_PORT", "8188")))
        resolve.add_argument("--refresh", action="store_true")
        resolve.add_argument("--colab-proxy", action="store_true")
        resolve.add_argument("--output", choices=("summary", "json"), default="summary")
        resolve.set_defaults(handler=_pack_resolve)

    doctor = pack_subparsers.add_parser("doctor", help="Validate the saved lock/state pair.")
    doctor.add_argument("--session", default=os.environ.get("COMFYCOLAB_SESSION", "comfycolab"))
    doctor.add_argument("--state", default=_default_state_path())
    doctor.add_argument(
        "--lock-dir",
        default=os.environ.get("COMFYCOLAB_LOCK_DIR", str(DEFAULT_LOCK_DIR)),
    )
    doctor.set_defaults(handler=_pack_doctor)

    rollback = pack_subparsers.add_parser(
        "rollback",
        help="Swap the current lock with the previous immutable lock.",
    )
    rollback.add_argument(
        "--session",
        default=os.environ.get("COMFYCOLAB_SESSION", "comfycolab"),
    )
    rollback.add_argument(
        "--lock-dir",
        default=os.environ.get("COMFYCOLAB_LOCK_DIR", str(DEFAULT_LOCK_DIR)),
    )
    rollback.set_defaults(handler=_pack_rollback)

    notebook = subparsers.add_parser(
        "notebook",
        help="Render a deterministic two-cell Colab notebook with an embedded lock.",
    )
    _add_selection_options(notebook)
    notebook.add_argument("--session", default=os.environ.get("COMFYCOLAB_SESSION", "comfycolab"))
    notebook.add_argument("--port", type=int, default=int(os.environ.get("COMFYCOLAB_PORT", "8188")))
    notebook.add_argument("--refresh", action="store_true")
    notebook.add_argument(
        "--colab-proxy",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    notebook.add_argument(
        "--legacy-full",
        action="store_true",
        help=(
            "Use the authenticated compatibility runtime for the exact Image, Video, "
            "3D, and 3DGS daughter refs selected by the profile."
        ),
    )
    notebook.add_argument(
        "--runtime-resolve-main",
        action="store_true",
        help=(
            "Render an opt-in public notebook that resolves the latest main commit "
            "inside Colab, then converts it to the normal immutable stage-0 lock."
        ),
    )
    notebook.add_argument("--output", default="ComfyColab.ipynb")
    notebook.set_defaults(handler=_render_notebook)

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
