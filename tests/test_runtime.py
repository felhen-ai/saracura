from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import pytest

from saracura.backends import DeterministicFixtureBackend
from saracura.backends.base import BackendCapabilities, EncodedState, ScoredChoice
from saracura.calibration import load_calibration
from saracura.contracts import ChoiceQuestion, DecisionRequest, ErrorCode, SaracuraError
from saracura.runtime import DecisionEngine, default_workflows, known_scaling_questions
from saracura.runtime.engine import NUMERIC_ISOLATION_ABSOLUTE_TOLERANCE, calibration_context
from tests.helpers import calibrations_for, decision_request


class InvalidScoreBackend(DeterministicFixtureBackend):
    def __init__(self, mode: str) -> None:
        super().__init__()
        self.mode = mode

    def score_choice(
        self,
        encoded_state: EncodedState,
        question: ChoiceQuestion,
        question_payload: bytes,
    ) -> ScoredChoice:
        scored = super().score_choice(encoded_state, question, question_payload)
        if self.mode == "missing":
            return ScoredChoice(question_id=scored.question_id, raw_scores={})
        if self.mode == "question":
            return ScoredChoice(question_id="another-question", raw_scores=scored.raw_scores)
        return ScoredChoice(
            question_id=scored.question_id,
            raw_scores={**scored.raw_scores, next(iter(scored.raw_scores)): math.inf},
        )


class NarrowCriteriaBackend(DeterministicFixtureBackend):
    @property
    def capabilities(self) -> BackendCapabilities:
        return replace(super().capabilities, max_criteria=1)


@pytest.mark.parametrize("question_count", [1, 10, 50])
def test_state_is_encoded_exactly_once_for_scaling_matrix(question_count: int) -> None:
    request = decision_request(known_scaling_questions(question_count))
    backend = DeterministicFixtureBackend()
    engine = DecisionEngine(
        backend=backend,
        workflows=default_workflows(),
        calibrations=calibrations_for(request, backend),
    )

    response = engine.decide(request)

    assert backend.state_encode_calls == 1
    assert len(response.answers) == question_count
    assert response.usage.questions == question_count
    assert response.timing is None


def test_adding_and_reordering_questions_preserves_existing_answers() -> None:
    questions = known_scaling_questions(10)
    request_one = decision_request(questions[:1])
    request_ten = decision_request(questions)
    request_reordered = decision_request(tuple(reversed(questions)))

    def run(request: DecisionRequest) -> dict[str, tuple[dict[str, float], dict[str, float]]]:
        backend = DeterministicFixtureBackend()
        engine = DecisionEngine(
            backend=backend,
            workflows=default_workflows(),
            calibrations=calibrations_for(request, backend),
        )
        response = engine.decide(request)
        assert backend.state_encode_calls == 1
        return {
            answer.question_id: (answer.raw_scores, answer.probabilities)
            for answer in response.answers
        }

    one = run(request_one)
    ten = run(request_ten)
    reordered = run(request_reordered)
    for question_id, (raw_scores, probabilities) in one.items():
        assert ten[question_id][0] == pytest.approx(
            raw_scores, abs=NUMERIC_ISOLATION_ABSOLUTE_TOLERANCE
        )
        assert ten[question_id][1] == pytest.approx(
            probabilities, abs=NUMERIC_ISOLATION_ABSOLUTE_TOLERANCE
        )
    for question_id, (raw_scores, probabilities) in ten.items():
        assert reordered[question_id][0] == pytest.approx(
            raw_scores, abs=NUMERIC_ISOLATION_ABSOLUTE_TOLERANCE
        )
        assert reordered[question_id][1] == pytest.approx(
            probabilities, abs=NUMERIC_ISOLATION_ABSOLUTE_TOLERANCE
        )


def test_ptbr_fixture_runs_end_to_end() -> None:
    root = Path(__file__).parents[1]
    request = DecisionRequest.model_validate_json(
        (root / "examples/ptbr-support-request.json").read_bytes()
    )
    backend = DeterministicFixtureBackend()
    question = request.questions[0]
    expected = calibration_context(request, question.id, len(question.criteria), backend)
    artifact = load_calibration(root / "examples/ptbr-support-calibration.json", expected)
    engine = DecisionEngine(
        backend=backend,
        workflows=default_workflows(),
        calibrations={question.id: artifact},
    )

    response = engine.decide(request)

    assert response.api_version == "v1alpha1"
    assert response.answers[0].status == "fixture_only"
    assert response.answers[0].calibration.status == "fixture_only"
    assert response.answers[0].calibration.risk_policy_id is None
    assert sum(response.answers[0].probabilities.values()) == pytest.approx(1.0)
    assert backend.capabilities.quality_claims is False
    json.loads(response.model_dump_json())


@pytest.mark.parametrize("mode", ["missing", "question", "nonfinite"])
def test_invalid_backend_scores_fail_closed(mode: str) -> None:
    request = decision_request(known_scaling_questions(1))
    backend = InvalidScoreBackend(mode)
    engine = DecisionEngine(
        backend=backend,
        workflows=default_workflows(),
        calibrations=calibrations_for(request, backend),
    )

    with pytest.raises(SaracuraError) as captured:
        engine.decide(request)
    assert captured.value.payload.code == ErrorCode.BACKEND_UNAVAILABLE


def test_backend_specific_cardinality_limit_fails_closed() -> None:
    request = decision_request(known_scaling_questions(1))
    backend = NarrowCriteriaBackend()
    engine = DecisionEngine(
        backend=backend,
        workflows=default_workflows(),
        calibrations=calibrations_for(request, backend),
    )

    with pytest.raises(SaracuraError) as captured:
        engine.decide(request)
    assert captured.value.payload.code == ErrorCode.CARDINALITY_EXCEEDED
