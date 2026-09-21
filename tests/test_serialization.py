from __future__ import annotations

import json
from pathlib import Path

import pytest

from saracura.contracts import ChoiceCriterion
from saracura.contracts.errors import ErrorCode, SaracuraError
from saracura.contracts.models import parse_request_json
from saracura.runtime import known_scaling_questions
from saracura.serialization import (
    canonical_json_bytes,
    canonical_request_bytes,
    frame_segments,
    serialize_question,
    serialize_state,
)


def test_nfc_values_are_byte_identical() -> None:
    assert canonical_json_bytes({"text": "Cafe\u0301"}) == canonical_json_bytes({"text": "Café"})


def test_nfc_key_collision_is_rejected() -> None:
    with pytest.raises(SaracuraError) as captured:
        canonical_json_bytes({"Cafe\u0301": 1, "Café": 2})
    assert captured.value.payload.code == ErrorCode.REQUEST_INVALID
    assert captured.value.payload.details == {}


def test_nfc_collision_error_does_not_echo_sensitive_keys() -> None:
    decomposed = "segredo-Cafe\u0301"
    composed = "segredo-Café"
    with pytest.raises(SaracuraError) as captured:
        canonical_json_bytes({decomposed: 1, composed: 2})

    envelope = json.dumps(captured.value.as_dict(), ensure_ascii=False)
    assert decomposed not in envelope
    assert composed not in envelope


def test_choice_criterion_order_is_not_semantic() -> None:
    question = known_scaling_questions(1)[0]
    reversed_question = question.model_copy(update={"criteria": tuple(reversed(question.criteria))})
    assert serialize_question(question) == serialize_question(reversed_question)


def test_strings_are_not_reinterpreted_as_json() -> None:
    encoded = canonical_json_bytes({"payload": '{"admin":true}'})
    assert json.loads(encoded) == {"payload": '{"admin":true}'}
    assert encoded != canonical_json_bytes({"payload": {"admin": True}})


@pytest.mark.parametrize("number", [float("nan"), float("inf"), 2**60])
def test_values_outside_rfc8785_numeric_domain_fail_as_request_errors(number: float) -> None:
    with pytest.raises(SaracuraError) as captured:
        canonical_json_bytes({"number": number})
    assert captured.value.payload.code == ErrorCode.REQUEST_INVALID


def test_length_prefixes_make_delimiter_like_content_unambiguous() -> None:
    first = frame_segments("a|b", "c")
    second = frame_segments("a", "b|c")
    assert first != second
    assert first[:8] == (3).to_bytes(8, "big")


def test_question_serializer_normalizes_descriptions() -> None:
    question = known_scaling_questions(1)[0]
    decomposed = question.model_copy(
        update={
            "criteria": (
                ChoiceCriterion(id="route-a", description="Rota determinística A"),
                ChoiceCriterion(id="route-b", description="Cafe\u0301"),
            )
        }
    )
    composed = decomposed.model_copy(
        update={
            "criteria": (
                ChoiceCriterion(id="route-a", description="Rota determinística A"),
                ChoiceCriterion(id="route-b", description="Café"),
            )
        }
    )
    assert serialize_question(decomposed) == serialize_question(composed)


def test_serialization_matches_cross_platform_golden_bytes() -> None:
    root = Path(__file__).parents[1]
    golden = json.loads((root / "tests/fixtures/serialization-v1.json").read_text(encoding="utf-8"))
    request = parse_request_json((root / "examples/ptbr-support-request.json").read_bytes())

    assert serialize_state(request).hex() == golden["state_hex"]
    assert serialize_question(request.questions[0]).hex() == golden["question_hex"]
    assert canonical_request_bytes(request).hex() == golden["request_hex"]
