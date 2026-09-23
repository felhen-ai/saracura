"""Focused contracts for the Phase 4D universal Choice tier."""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from saracura.backends import DeterministicFixtureBackend
from saracura.backends.base import BackendCapabilities, ScoredChoice
from saracura.contracts import (
    Answer,
    CalibrationReference,
    ChoiceCriterion,
    ChoiceQuestion,
    DecisionRequest,
    ErrorCode,
    ModelReference,
    SaracuraError,
    WorkflowReference,
)
from saracura.runtime import DecisionEngine, default_workflows
from saracura.runtime.workflows import (
    UNIVERSAL_CHOICE_WORKFLOW_ID,
    UNIVERSAL_CHOICE_WORKFLOW_REVISION,
)
from saracura.serialization import serialize_state


class FakeUniversalBackend:
    """Deterministic question-conditioned backend with no optional imports."""

    def __init__(self, mode: str = "valid") -> None:
        self.mode = mode
        self.calls: list[tuple[DecisionRequest, ChoiceQuestion, bytes]] = []
        self._model = ModelReference(
            id="fake-universal",
            revision="fake-universal-v1",
            checkpoint_sha256="a" * 64,
        )
        self._capabilities = BackendCapabilities(
            execution_tier="universal",
            decision_types=frozenset({"choice"}),
            max_questions=50,
            max_criteria=20,
            execution_boundary="deterministic-phase4d-test",
            cold_warm_semantics="no-load",
            quality_claims=False,
            dynamic_workflows=frozenset(
                {(UNIVERSAL_CHOICE_WORKFLOW_ID, UNIVERSAL_CHOICE_WORKFLOW_REVISION)}
            ),
        )

    @property
    def capabilities(self) -> BackendCapabilities:
        return self._capabilities

    @property
    def model(self) -> ModelReference:
        return self._model

    def validate_request(self, request: DecisionRequest) -> None:
        del request

    def score_universal_choice(
        self,
        request: DecisionRequest,
        question: ChoiceQuestion,
        state_payload: bytes,
    ) -> ScoredChoice:
        self.calls.append((request, question, state_payload))
        scores = {
            criterion.id: float(index)
            for index, criterion in reversed(tuple(enumerate(question.criteria)))
        }
        if self.mode == "missing":
            scores.pop(question.criteria[0].id)
        if self.mode == "unexpected":
            scores["unexpected"] = 0.0
        if self.mode == "nonfinite":
            scores[question.criteria[0].id] = math.inf
        return ScoredChoice(question_id=question.id, raw_scores=scores, input_tokens=11)


def _dynamic_request(locale: str = "pt-BR") -> DecisionRequest:
    return DecisionRequest(
        api_version="v1alpha1",
        model="fake-universal-v1",
        mode="research",
        locale=locale,
        domain="email-triage",
        workflow=WorkflowReference(
            id=UNIVERSAL_CHOICE_WORKFLOW_ID,
            revision=UNIVERSAL_CHOICE_WORKFLOW_REVISION,
        ),
        state={"subject": "Mensagem sintética", "body": "Conteúdo exclusivamente de teste."},
        questions=(
            ChoiceQuestion(
                id="triage",
                type="choice",
                instruction="Escolha a categoria mais adequada.",
                criteria=(
                    ChoiceCriterion(id="action_required", description="Exige acompanhamento."),
                    ChoiceCriterion(
                        id="billing_or_accounting", description="Cobrança ou contabilidade."
                    ),
                    ChoiceCriterion(
                        id="marketing_or_newsletter", description="Comunicação comercial."
                    ),
                ),
            ),
        ),
    )


@pytest.mark.parametrize("locale", ["pt-BR", "en"])
def test_universal_dynamic_workflow_returns_abstained_ranking_only_response(locale: str) -> None:
    request = _dynamic_request(locale)
    backend = FakeUniversalBackend()
    response = DecisionEngine(
        backend=backend,
        workflows=default_workflows(),
        calibrations={},
    ).decide(request)

    answer = response.answers[0]
    assert backend.calls == [(request, request.questions[0], serialize_state(request))]
    assert tuple(answer.raw_scores) == tuple(
        criterion.id for criterion in request.questions[0].criteria
    )
    assert tuple(answer.probabilities) == tuple(
        criterion.id for criterion in request.questions[0].criteria
    )
    assert answer.value == "marketing_or_newsletter"
    assert answer.status == "uncalibrated"
    assert answer.score_semantics == "ranking_weights"
    assert answer.abstained is True
    assert answer.reason == "uncalibrated_research"
    assert answer.calibration is None
    assert response.automation_allowed is False
    assert response.usage.input_tokens == 11


@pytest.mark.parametrize("mode", ["missing", "unexpected", "nonfinite"])
def test_universal_score_keys_and_values_fail_closed(mode: str) -> None:
    request = _dynamic_request()
    backend = FakeUniversalBackend(mode)
    engine = DecisionEngine(backend=backend, workflows=default_workflows(), calibrations={})

    with pytest.raises(SaracuraError) as captured:
        engine.decide(request)

    assert captured.value.payload.code == ErrorCode.BACKEND_UNAVAILABLE


def test_dynamic_workflow_rejects_compiled_and_unregistered_references() -> None:
    dynamic_request = _dynamic_request()
    compiled = DeterministicFixtureBackend()
    compiled_engine = DecisionEngine(
        backend=compiled,
        workflows=default_workflows(),
        calibrations={},
    )
    with pytest.raises(SaracuraError) as compiled_error:
        compiled_engine.decide(
            dynamic_request.model_copy(update={"model": compiled.model.revision})
        )
    assert compiled_error.value.payload.code == ErrorCode.WORKFLOW_UNSUPPORTED
    assert compiled.state_encode_calls == 0

    backend = FakeUniversalBackend()
    unregistered = dynamic_request.model_copy(
        update={"workflow": WorkflowReference(id=UNIVERSAL_CHOICE_WORKFLOW_ID, revision="unknown")}
    )
    with pytest.raises(SaracuraError) as universal_error:
        DecisionEngine(backend=backend, workflows=default_workflows(), calibrations={}).decide(
            unregistered
        )
    assert universal_error.value.payload.code == ErrorCode.WORKFLOW_UNSUPPORTED
    assert backend.calls == []


def test_dynamic_workflow_enforces_its_lower_question_limit_before_scoring() -> None:
    request = _dynamic_request()
    template = request.questions[0]
    too_many_questions = tuple(
        template.model_copy(update={"id": f"triage-{index}"}) for index in range(11)
    )
    backend = FakeUniversalBackend()

    with pytest.raises(SaracuraError) as captured:
        DecisionEngine(backend=backend, workflows=default_workflows(), calibrations={}).decide(
            request.model_copy(update={"questions": too_many_questions})
        )

    assert captured.value.payload.code == ErrorCode.CARDINALITY_EXCEEDED
    assert backend.calls == []


def test_universal_backend_rejects_injected_calibration_before_scoring() -> None:
    backend = FakeUniversalBackend()
    engine = DecisionEngine(
        backend=backend,
        workflows=default_workflows(),
        calibrations={"triage": object()},  # type: ignore[dict-item]
    )

    with pytest.raises(SaracuraError) as captured:
        engine.decide(_dynamic_request())

    assert captured.value.payload.code == ErrorCode.CALIBRATION_INCOMPATIBLE
    assert backend.calls == []


def test_answer_and_response_models_reject_inconsistent_research_semantics() -> None:
    response = DecisionEngine(
        backend=FakeUniversalBackend(),
        workflows=default_workflows(),
        calibrations={},
    ).decide(_dynamic_request())
    answer_payload = response.answers[0].model_dump(mode="json")
    answer_payload["calibration"] = CalibrationReference(
        id="fixture", status="fixture_only"
    ).model_dump(mode="json")
    with pytest.raises(ValidationError):
        Answer.model_validate(answer_payload)

    response_payload = response.model_dump(mode="json")
    response_payload["automation_allowed"] = True
    with pytest.raises(ValidationError):
        type(response).model_validate(response_payload)
