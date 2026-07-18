"""Errors raised by the versioned ComfyColab pack contract."""

from __future__ import annotations


class PackContractError(RuntimeError):
    """Base error for invalid or incompatible pack data."""


class PackSchemaError(PackContractError):
    """A pack document does not conform to its declared schema."""


class PackConflictError(PackContractError):
    """Two otherwise valid pack declarations cannot be composed safely."""


class PackIntegrityError(PackContractError):
    """A pack document does not match its expected content digest."""
