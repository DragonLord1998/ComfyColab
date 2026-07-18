#!/usr/bin/env python3
"""Prepare and verify exact-ref daughter checkouts for integration CI."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit


CORE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CORE_ROOT / "src"))

from comfycolab.packs import (  # noqa: E402
    PackContractError,
    PackRefV1,
    load_pack_manifest,
    load_registry,
)


PACK_CHECKOUTS = {
    "3d": ("ComfyColab-3D", ".[test]"),
    "3dgs": ("ComfyColab-3DGS", "."),
    "image": ("ComfyColab-Image", "."),
    "video": ("ComfyColab-Video", "."),
    "world": ("ComfyColab-WorldModels", "."),
}


class ExactRefCIError(RuntimeError):
    """Raised when an integration checkout is not exactly registry-authenticated."""


@dataclass(frozen=True)
class Checkout:
    alias: str
    pack_ref: PackRefV1
    repository_slug: str
    directory: str
    install_suffix: str

    def matrix_entry(self) -> dict[str, str]:
        return {
            "alias": self.alias,
            "id": self.pack_ref.id,
            "repository": self.repository_slug,
            "ref": self.pack_ref.ref,
            "manifest_sha256": self.pack_ref.manifest_sha256,
            "directory": self.directory,
            "install_target": f"./daughter{self.install_suffix[1:]}",
        }


def _github_repository_slug(repository: str) -> str:
    parsed = urlsplit(repository)
    parts = tuple(part for part in parsed.path.split("/") if part)
    if parsed.hostname != "github.com" or len(parts) != 2 or not parts[1].endswith(".git"):
        raise ExactRefCIError(
            f"CI daughter repository must be a GitHub HTTPS .git URL: {repository}"
        )
    return f"{parts[0]}/{parts[1][:-4]}"


def load_checkouts(registry_path: Path) -> tuple[Checkout, ...]:
    try:
        registry = load_registry(registry_path)
    except (OSError, PackContractError) as error:
        raise ExactRefCIError(f"Invalid published pack registry: {error}") from error

    by_id: dict[str, tuple[str, PackRefV1]] = {}
    for alias, pack_ref in registry.packs.items():
        if pack_ref.id in by_id:
            previous_alias = by_id[pack_ref.id][0]
            raise ExactRefCIError(
                f"Pack id {pack_ref.id!r} is duplicated by {previous_alias!r} and {alias!r}."
            )
        by_id[pack_ref.id] = (alias, pack_ref)

    expected = set(PACK_CHECKOUTS)
    actual = set(by_id)
    if actual != expected:
        missing = ", ".join(sorted(expected - actual)) or "none"
        unexpected = ", ".join(sorted(actual - expected)) or "none"
        raise ExactRefCIError(
            f"Published integration registry must contain exactly five daughter packs; "
            f"missing: {missing}; unexpected: {unexpected}."
        )

    checkouts = []
    for pack_id, (directory, install_suffix) in PACK_CHECKOUTS.items():
        alias, pack_ref = by_id[pack_id]
        checkouts.append(
            Checkout(
                alias=alias,
                pack_ref=pack_ref,
                repository_slug=_github_repository_slug(pack_ref.repository),
                directory=directory,
                install_suffix=install_suffix,
            )
        )
    return tuple(checkouts)


def matrix(checkouts: tuple[Checkout, ...]) -> str:
    return json.dumps(
        {"include": [checkout.matrix_entry() for checkout in checkouts]},
        separators=(",", ":"),
        sort_keys=True,
    )


def _run(*arguments: str, cwd: Path | None = None) -> str:
    completed = subprocess.run(
        arguments,
        cwd=cwd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def checkout_all(checkouts: tuple[Checkout, ...], destination_root: Path) -> None:
    destination_root.mkdir(parents=True, exist_ok=True)
    for checkout in checkouts:
        destination = destination_root / checkout.directory
        if destination.exists():
            raise ExactRefCIError(f"Checkout destination already exists: {destination}")
        _run("git", "init", "--quiet", str(destination))
        _run(
            "git",
            "remote",
            "add",
            "origin",
            checkout.pack_ref.repository,
            cwd=destination,
        )
        _run(
            "git",
            "fetch",
            "--quiet",
            "--depth=1",
            "origin",
            checkout.pack_ref.ref,
            cwd=destination,
        )
        _run(
            "git",
            "checkout",
            "--quiet",
            "--detach",
            "FETCH_HEAD",
            cwd=destination,
        )
        verify_checkout(checkout, destination)


def verify_checkout(checkout: Checkout, checkout_root: Path) -> None:
    try:
        actual_ref = _run("git", "rev-parse", "HEAD", cwd=checkout_root)
    except (OSError, subprocess.CalledProcessError) as error:
        raise ExactRefCIError(f"Cannot inspect checkout at {checkout_root}: {error}") from error
    if actual_ref != checkout.pack_ref.ref:
        raise ExactRefCIError(
            f"{checkout.pack_ref.id} checkout is {actual_ref}, "
            f"expected {checkout.pack_ref.ref}."
        )

    manifest_path = checkout_root / "comfycolab-pack.json"
    try:
        manifest = load_pack_manifest(
            manifest_path,
            expected_sha256=checkout.pack_ref.manifest_sha256,
        )
    except (OSError, PackContractError) as error:
        raise ExactRefCIError(
            f"{checkout.pack_ref.id} manifest verification failed: {error}"
        ) from error
    if manifest.id != checkout.pack_ref.id:
        raise ExactRefCIError(
            f"Manifest id {manifest.id!r} does not match registry id "
            f"{checkout.pack_ref.id!r}."
        )


def verify_all(checkouts: tuple[Checkout, ...], destination_root: Path) -> None:
    for checkout in checkouts:
        verify_checkout(checkout, destination_root / checkout.directory)


def _checkout_for_id(checkouts: tuple[Checkout, ...], pack_id: str) -> Checkout:
    for checkout in checkouts:
        if checkout.pack_ref.id == pack_id:
            return checkout
    raise ExactRefCIError(f"Unknown daughter pack id: {pack_id}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    matrix_parser = subparsers.add_parser("matrix")
    matrix_parser.add_argument("registry", type=Path)

    checkout_parser = subparsers.add_parser("checkout")
    checkout_parser.add_argument("registry", type=Path)
    checkout_parser.add_argument("destination", type=Path)

    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("registry", type=Path)
    verify_parser.add_argument("destination", type=Path)

    verify_one_parser = subparsers.add_parser("verify-one")
    verify_one_parser.add_argument("registry", type=Path)
    verify_one_parser.add_argument("pack_id")
    verify_one_parser.add_argument("checkout", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        checkouts = load_checkouts(arguments.registry)
        if arguments.command == "matrix":
            print(matrix(checkouts))
        elif arguments.command == "checkout":
            checkout_all(checkouts, arguments.destination)
        elif arguments.command == "verify":
            verify_all(checkouts, arguments.destination)
        else:
            verify_checkout(
                _checkout_for_id(checkouts, arguments.pack_id),
                arguments.checkout,
            )
    except ExactRefCIError as error:
        print(f"exact-ref CI error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
