from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator, Sequence
from urllib.parse import urlparse


_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_AUTHOR_REF_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,254}$")


class RepositoryError(RuntimeError):
    """Raised when an immutable repository checkout cannot be established."""


Runner = Callable[..., subprocess.CompletedProcess[str]]


def validate_repository_url(repository: str) -> str:
    parsed = urlparse(repository)
    if parsed.scheme != "https" or not parsed.netloc:
        raise RepositoryError("repository must be an absolute HTTPS URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise RepositoryError("repository must not contain credentials, query, or fragment")
    if not parsed.path.endswith(".git"):
        raise RepositoryError("repository URL must end with .git")
    return repository


def validate_author_ref(ref: str) -> str:
    if not _AUTHOR_REF_RE.fullmatch(ref):
        raise RepositoryError(f"unsafe Git ref: {ref!r}")
    if ref.startswith("-") or ".." in ref or "//" in ref or ref.endswith("/"):
        raise RepositoryError(f"unsafe Git ref: {ref!r}")
    return ref


def _run_git(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    runner: Runner = subprocess.run,
) -> subprocess.CompletedProcess[str]:
    result = runner(
        list(command),
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RepositoryError(f"Git command failed: {' '.join(command)}\n{detail}")
    return result


def checkout_repository(
    repository: str,
    ref: str,
    destination: Path,
    *,
    runner: Runner = subprocess.run,
) -> str:
    validate_repository_url(repository)
    validate_author_ref(ref)
    if destination.exists():
        if destination.is_symlink() or destination.is_file():
            destination.unlink()
        else:
            shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _run_git(
        [
            "git",
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            repository,
            str(destination),
        ],
        runner=runner,
    )
    _run_git(
        ["git", "fetch", "origin", ref, "--depth", "1"],
        cwd=destination,
        runner=runner,
    )
    _run_git(
        ["git", "checkout", "--detach", "FETCH_HEAD"],
        cwd=destination,
        runner=runner,
    )
    commit = _run_git(
        ["git", "rev-parse", "HEAD"],
        cwd=destination,
        runner=runner,
    ).stdout.strip()
    if not _COMMIT_RE.fullmatch(commit):
        raise RepositoryError(f"checkout returned an invalid commit: {commit!r}")
    if _COMMIT_RE.fullmatch(ref) and commit != ref:
        raise RepositoryError(f"checkout resolved {commit}, expected immutable ref {ref}")
    return commit


@contextmanager
def temporary_checkout(repository: str, ref: str) -> Iterator[tuple[Path, str]]:
    with tempfile.TemporaryDirectory(prefix="comfycolab-checkout-") as directory:
        destination = Path(directory) / "repository"
        commit = checkout_repository(repository, ref, destination)
        yield destination, commit
