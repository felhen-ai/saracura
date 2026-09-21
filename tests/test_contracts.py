from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from saracura.backends import DeterministicFixtureBackend
from saracura.contracts import DecisionRequest, ErrorCode, SaracuraError, parse_request_json
from saracura.runtime import DecisionEngine, default_workflows, known_scaling_questions
from tests.helpers import calibrations_for, decision_request


def test_contracts_reject_unknown_fields_and_unsupported_mode_or_type() -> None:
    payload = decision_request(known_scaling_questions(1)).model_dump(mode="json")

    with_extra = {**payload, "automation_ready": True}
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        DecisionRequest.model_validate(with_extra)

    with_mode = {**payload, "mode": "automation"}
    with pytest.raises(ValidationError, match="research"):
        DecisionRequest.model_validate(with_mode)

    with_type = json.loads(json.dumps(payload))
    with_type["questions"][0]["type"] = "boolean"
    with pytest.raises(ValidationError, match="choice"):
        DecisionRequest.model_validate(with_type)


@pytest.mark.parametrize(
    ("field", "replacement", "expected"),
    [
        ("api_version", "v2", ErrorCode.API_VERSION_UNSUPPORTED),
        ("mode", "automation", ErrorCode.REQUEST_INVALID),
    ],
)
def test_json_boundary_returns_explicit_regime_errors(
    field: str,
    replacement: str,
    expected: ErrorCode,
) -> None:
    payload = decision_request(known_scaling_questions(1)).model_dump(mode="json")
    payload[field] = replacement
    with pytest.raises(SaracuraError) as captured:
        parse_request_json(json.dumps(payload))
    assert captured.value.payload.code == expected


def test_json_boundary_returns_schema_error_for_unsupported_type() -> None:
    payload = decision_request(known_scaling_questions(1)).model_dump(mode="json")
    payload["questions"][0]["type"] = "ordinal"
    with pytest.raises(SaracuraError) as captured:
        parse_request_json(json.dumps(payload))
    assert captured.value.payload.code == ErrorCode.SCHEMA_UNSUPPORTED


def test_validation_errors_do_not_echo_request_values() -> None:
    payload = decision_request(known_scaling_questions(1)).model_dump(mode="json")
    payload["unexpected"] = "sensitive-input-must-not-appear"

    with pytest.raises(SaracuraError) as captured:
        parse_request_json(json.dumps(payload))

    serialized = json.dumps(captured.value.as_dict())
    assert "sensitive-input-must-not-appear" not in serialized
    violations = captured.value.payload.details["violations"]
    assert isinstance(violations, list)
    assert violations


@pytest.mark.parametrize(
    ("workflow_id", "workflow_revision", "expected"),
    [
        ("unknown", "2026-09-21", ErrorCode.WORKFLOW_UNSUPPORTED),
        ("decision-scaling", "latest", ErrorCode.WORKFLOW_UNSUPPORTED),
    ],
)
def test_unknown_workflow_or_alias_fails_explicitly(
    workflow_id: str,
    workflow_revision: str,
    expected: ErrorCode,
) -> None:
    questions = known_scaling_questions(1)
    request = decision_request(
        questions,
        workflow_id=workflow_id,
        workflow_revision=workflow_revision,
    )
    backend = DeterministicFixtureBackend()
    engine = DecisionEngine(
        backend=backend,
        workflows=default_workflows(),
        calibrations=calibrations_for(request, backend),
    )

    with pytest.raises(SaracuraError) as captured:
        engine.decide(request)
    assert captured.value.payload.code == expected


def test_question_outside_known_schema_fails_explicitly() -> None:
    original = known_scaling_questions(1)[0]
    changed = original.model_copy(update={"instruction": "Uma pergunta nova em runtime"})
    request = decision_request((changed,))
    backend = DeterministicFixtureBackend()
    engine = DecisionEngine(
        backend=backend,
        workflows=default_workflows(),
        calibrations=calibrations_for(request, backend),
    )

    with pytest.raises(SaracuraError) as captured:
        engine.decide(request)
    assert captured.value.payload.code == ErrorCode.SCHEMA_UNSUPPORTED


def test_model_revision_alias_is_not_resolved() -> None:
    request = decision_request(known_scaling_questions(1)).model_copy(update={"model": "latest"})
    backend = DeterministicFixtureBackend()
    engine = DecisionEngine(
        backend=backend,
        workflows=default_workflows(),
        calibrations=calibrations_for(request, backend),
    )

    with pytest.raises(SaracuraError) as captured:
        engine.decide(request)
    assert captured.value.payload.code == ErrorCode.MODEL_ALIAS_FORBIDDEN
