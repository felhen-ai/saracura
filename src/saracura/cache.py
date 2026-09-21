"""Fail-closed schema cache contract for future criteria encoders."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from saracura.contracts.errors import ErrorCode, SaracuraError
from saracura.serialization import frame_segments


@dataclass(frozen=True, slots=True)
class SchemaCacheKey:
    serializer_version: str
    model_id: str
    model_revision: str
    checkpoint_sha256: str
    architecture_config_sha256: str
    tokenizer_revision: str
    locale: str
    schema_bytes: bytes

    def digest(self) -> str:
        """Hash every compatibility axis and the complete canonical schema bytes."""

        payload = frame_segments(
            self.serializer_version,
            self.model_id,
            self.model_revision,
            self.checkpoint_sha256,
            self.architecture_config_sha256,
            self.tokenizer_revision,
            self.locale,
            self.schema_bytes,
        )
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class SchemaCacheEntry:
    key: SchemaCacheKey
    payload: bytes


class SchemaCache:
    """Small in-memory reference cache with a persistence-safe restore boundary."""

    def __init__(self) -> None:
        self._entries: dict[str, SchemaCacheEntry] = {}

    def put(self, key: SchemaCacheKey, payload: bytes) -> str:
        digest = key.digest()
        existing = self._entries.get(digest)
        if existing is not None and existing.key != key:
            self._raise_incompatible()
        self._entries[digest] = SchemaCacheEntry(key=key, payload=payload)
        return digest

    def get(self, key: SchemaCacheKey) -> bytes | None:
        entry = self._entries.get(key.digest())
        if entry is None:
            return None
        if entry.key != key:
            self._raise_incompatible()
        return entry.payload

    def restore(self, declared_digest: str, entry: SchemaCacheEntry) -> None:
        """Restore a persisted entry only when its metadata reproduces its key."""

        if declared_digest != entry.key.digest():
            self._raise_incompatible()
        existing = self._entries.get(declared_digest)
        if existing is not None and existing.key != entry.key:
            self._raise_incompatible()
        self._entries[declared_digest] = entry

    @staticmethod
    def _raise_incompatible() -> None:
        raise SaracuraError(
            ErrorCode.CACHE_INCOMPATIBLE,
            "Schema cache entry is incompatible with its declared key.",
            "/cache",
        )


__all__ = ["SchemaCache", "SchemaCacheEntry", "SchemaCacheKey"]
