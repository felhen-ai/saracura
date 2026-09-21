"""Versioned Unicode normalization, canonical JSON, and unambiguous framing."""

from __future__ import annotations

import unicodedata
from collections.abc import Mapping, Sequence
from typing import TypeAlias, cast

import rfc8785
from pydantic import JsonValue

from saracura.contracts.errors import ErrorCode, SaracuraError
from saracura.contracts.models import ChoiceQuestion, DecisionRequest

SERIALIZER_VERSION = "nfc-jcs-length-prefixed-v1"
JsonObject: TypeAlias = dict[str, JsonValue]


def _normalize(value: JsonValue, path: str = "$") -> JsonValue:
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, Mapping):
        normalized: JsonObject = {}
        for key, child in value.items():
            normalized_key = unicodedata.normalize("NFC", key)
            if normalized_key in normalized:
                raise SaracuraError(
                    ErrorCode.REQUEST_INVALID,
                    "Object keys collide after NFC normalization.",
                    path,
                )
            normalized[normalized_key] = _normalize(child, f"{path}/*")
        return normalized
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_normalize(item, f"{path}/{index}") for index, item in enumerate(value)]
    return value


def normalize_nfc(value: JsonValue) -> JsonValue:
    """Apply the serializer's explicit Unicode transform, rejecting key collisions."""

    return _normalize(value)


def canonical_json_bytes(value: JsonValue) -> bytes:
    """Normalize with NFC, then encode with RFC 8785 JSON canonicalization."""

    normalized = normalize_nfc(value)
    try:
        return rfc8785.dumps(normalized)
    except rfc8785.CanonicalizationError as error:
        raise SaracuraError(
            ErrorCode.REQUEST_INVALID,
            "Value cannot be represented by the RFC 8785 numeric domain.",
            "$",
        ) from error


def frame_segments(*segments: bytes | str) -> bytes:
    """Encode segments with an unsigned 64-bit byte length prefix."""

    framed = bytearray()
    for segment in segments:
        payload = (
            unicodedata.normalize("NFC", segment).encode("utf-8")
            if isinstance(segment, str)
            else segment
        )
        framed.extend(len(payload).to_bytes(8, byteorder="big", signed=False))
        framed.extend(payload)
    return bytes(framed)


def serialize_state(request: DecisionRequest) -> bytes:
    """Serialize only state-scoped data; questions cannot affect this representation."""

    state = cast(JsonValue, request.state)
    return frame_segments(
        SERIALIZER_VERSION,
        request.locale,
        request.domain,
        canonical_json_bytes(state),
    )


def serialize_question(question: ChoiceQuestion) -> bytes:
    """Serialize a choice question with criteria ordered by their non-semantic id."""

    segments: list[str] = [SERIALIZER_VERSION, question.id, question.type, question.instruction]
    for criterion in sorted(question.criteria, key=lambda item: item.id):
        segments.extend((criterion.id, criterion.description))
    return frame_segments(*segments)


def canonical_request_bytes(request: DecisionRequest) -> bytes:
    """Return byte-identical canonical request JSON for evidence and fixture tests."""

    payload = request.model_dump(mode="json")
    questions = cast(list[dict[str, JsonValue]], payload["questions"])
    for question in questions:
        criteria = cast(list[dict[str, JsonValue]], question["criteria"])
        question["criteria"] = cast(
            JsonValue,
            sorted(criteria, key=_criterion_id),
        )
    return canonical_json_bytes(cast(JsonValue, payload))


def _criterion_id(item: dict[str, JsonValue]) -> str:
    value = item.get("id")
    if not isinstance(value, str):
        raise TypeError("validated criterion id must be a string")
    return value
