from __future__ import annotations

import pytest

from saracura.cache import SchemaCache, SchemaCacheEntry, SchemaCacheKey
from saracura.contracts import ErrorCode, SaracuraError
from saracura.runtime import known_scaling_questions
from saracura.runtime.engine import (
    FIXTURE_ARCHITECTURE_SHA256,
    FIXTURE_TOKENIZER_REVISION,
)
from saracura.serialization import SERIALIZER_VERSION, serialize_question


def _key(*, model_revision: str = "fixture-choice-v1") -> SchemaCacheKey:
    return SchemaCacheKey(
        serializer_version=SERIALIZER_VERSION,
        model_id="fixture-choice",
        model_revision=model_revision,
        checkpoint_sha256="0" * 64,
        architecture_config_sha256=FIXTURE_ARCHITECTURE_SHA256,
        tokenizer_revision=FIXTURE_TOKENIZER_REVISION,
        locale="pt-BR",
        schema_bytes=serialize_question(known_scaling_questions(1)[0]),
    )


def test_cache_key_contains_full_canonical_schema_and_all_revision_axes() -> None:
    key = _key()
    cache = SchemaCache()
    digest = cache.put(key, b"encoded-schema")

    assert key.schema_bytes
    assert cache.get(key) == b"encoded-schema"
    assert digest != _key(model_revision="fixture-choice-v2").digest()


def test_cache_restore_rejects_poisoned_metadata() -> None:
    expected = _key()
    poisoned = _key(model_revision="fixture-choice-v2")
    cache = SchemaCache()

    with pytest.raises(SaracuraError) as captured:
        cache.restore(
            expected.digest(),
            SchemaCacheEntry(key=poisoned, payload=b"poison"),
        )

    assert captured.value.payload.code == ErrorCode.CACHE_INCOMPATIBLE
    assert cache.get(expected) is None
