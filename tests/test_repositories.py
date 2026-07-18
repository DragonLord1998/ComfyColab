from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from comfycolab.repositories import (
    RepositoryError,
    checkout_repository,
    validate_author_ref,
    validate_repository_url,
)


class RepositoryTests(unittest.TestCase):
    def test_repository_url_rejects_credentials_and_non_https(self) -> None:
        self.assertEqual(
            validate_repository_url("https://github.com/example/repo.git"),
            "https://github.com/example/repo.git",
        )
        for value in (
            "http://github.com/example/repo.git",
            "https://token@github.com/example/repo.git",
            "https://github.com/example/repo",
        ):
            with self.subTest(value=value), self.assertRaises(RepositoryError):
                validate_repository_url(value)

    def test_author_ref_rejects_option_and_revision_injection(self) -> None:
        self.assertEqual(validate_author_ref("release/v1.0.0"), "release/v1.0.0")
        for value in ("--upload-pack=evil", "main..other", "refs//heads/main", "main^{}"):
            with self.subTest(value=value), self.assertRaises(RepositoryError):
                validate_author_ref(value)

    def test_exact_checkout_verifies_resolved_commit(self) -> None:
        calls: list[list[str]] = []
        requested = "a" * 40

        def runner(command, **kwargs):
            calls.append(command)
            stdout = requested + "\n" if command[-2:] == ["rev-parse", "HEAD"] else ""
            return subprocess.CompletedProcess(command, 0, stdout, "")

        with tempfile.TemporaryDirectory() as directory:
            commit = checkout_repository(
                "https://github.com/example/repo.git",
                requested,
                Path(directory) / "checkout",
                runner=runner,
            )
        self.assertEqual(commit, requested)
        self.assertTrue(any(command[:2] == ["git", "fetch"] for command in calls))


if __name__ == "__main__":
    unittest.main()
