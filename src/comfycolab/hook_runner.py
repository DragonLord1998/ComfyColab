"""Best-effort policy guard for trusted, authenticated ComfyColab pack hooks.

Python audit hooks catch accidental undeclared writes, subprocesses, and
network access. They are not an isolation boundary for hostile native code.
"""

from __future__ import annotations

import json
import os
import runpy
import socket
import sys
from pathlib import Path
from typing import Iterable


class HookSandboxViolation(RuntimeError):
    """Raised when a hook attempts an undeclared side effect."""


_WRITE_FLAGS = (
    os.O_WRONLY
    | os.O_RDWR
    | os.O_CREAT
    | os.O_TRUNC
    | os.O_APPEND
)
_PATH_MUTATIONS = {
    "os.chmod": (0,),
    "os.chown": (0,),
    "os.link": (0, 1),
    "os.mkdir": (0,),
    "os.remove": (0,),
    "os.rename": (0, 1),
    "os.rmdir": (0,),
    "os.symlink": (1,),
    "os.truncate": (0,),
    "os.unlink": (0,),
    "os.utime": (0,),
}


def _load_allowed_roots() -> tuple[Path, ...]:
    raw = os.environ.get("COMFYCOLAB_HOOK_WRITE_ROOTS", "[]")
    try:
        values = json.loads(raw)
    except json.JSONDecodeError as error:
        raise HookSandboxViolation("hook write-root configuration is invalid") from error
    if not isinstance(values, list) or not all(
        isinstance(item, str) and Path(item).is_absolute() for item in values
    ):
        raise HookSandboxViolation("hook write roots must be absolute paths")
    return tuple(Path(item).resolve() for item in values)


def _resolve_event_path(value: object) -> Path | None:
    if isinstance(value, int):
        return None
    if isinstance(value, bytes):
        value = os.fsdecode(value)
    if not isinstance(value, str):
        try:
            value = os.fspath(value)
        except TypeError:
            return None
    path = Path(value)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def _assert_writable(path_value: object, allowed_roots: Iterable[Path]) -> None:
    path = _resolve_event_path(path_value)
    if path is None:
        return
    for root in allowed_roots:
        if path == root or root in path.parents:
            return
    raise HookSandboxViolation(f"undeclared filesystem write blocked: {path}")


def _audit_hook(allowed_roots: tuple[Path, ...]):
    def audit(event: str, args: tuple[object, ...]) -> None:
        if event == "open":
            path = args[0] if args else None
            mode = args[1] if len(args) > 1 else None
            flags = args[2] if len(args) > 2 else 0
            writes = (
                isinstance(mode, str)
                and any(marker in mode for marker in ("w", "a", "x", "+"))
            ) or (
                isinstance(flags, int)
                and bool(flags & _WRITE_FLAGS)
            )
            if writes:
                _assert_writable(path, allowed_roots)
            return
        if event in _PATH_MUTATIONS:
            for index in _PATH_MUTATIONS[event]:
                if index < len(args):
                    _assert_writable(args[index], allowed_roots)
            return
        if (
            event.startswith("socket.")
            or event == "subprocess.Popen"
            or event == "os.system"
            or event.startswith("os.exec")
            or event.startswith("os.spawn")
            or event == "pty.spawn"
        ):
            raise HookSandboxViolation(f"undeclared process or network action blocked: {event}")
        if event == "os.chdir":
            raise HookSandboxViolation("hook working-directory changes are blocked")

    return audit


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments:
        raise SystemExit(
            "usage: python -m comfycolab.hook_runner HOOK_PATH [HOOK_ARGUMENT ...]"
        )
    hook_path = Path(arguments[0]).resolve()
    if not hook_path.is_file():
        raise SystemExit(f"hook is missing: {hook_path}")
    allowed_roots = _load_allowed_roots()
    sys.addaudithook(_audit_hook(allowed_roots))
    sys.argv = [str(hook_path), *arguments[1:]]
    try:
        runpy.run_path(str(hook_path), run_name="__main__")
    except HookSandboxViolation as error:
        print(f"[comfycolab:hook-sandbox] {error}", file=sys.stderr)
        return 97
    except SystemExit as error:
        if error.code is None:
            return 0
        if isinstance(error.code, int):
            return error.code
        print(str(error.code), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
