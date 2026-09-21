"""Self-authored factories used only by the local correctness suite."""

from __future__ import annotations

from collections.abc import Iterable

from saracura.backends import DeterministicFixtureBackend
from saracura.calibration import CalibrationArtifact, TemperatureParameters
from saracura.contracts import ChoiceQuestion, DecisionRequest, WorkflowReference
from saracura.runtime.engine import calibration_context


def decision_request(
    questions: Iterable[ChoiceQuestion],
    *,
    workflow_id: str = "decision-scaling",
    workflow_revision: str = "2026-09-21",
    domain: str = "benchmark",
) -> DecisionRequest:
    return DecisionRequest(
        api_version="v1alpha1",
        model="fixture-choice-v1",
        mode="research",
        locale="pt-BR",
        domain=domain,
        workflow=WorkflowReference(id=workflow_id, revision=workflow_revision),
        state={
            "subject": "Estado autoescrito para o teste",
            "body": "O mesmo estado deve servir a todas as decisões conhecidas.",
        },
        questions=tuple(questions),
    )


def calibration_for(
    request: DecisionRequest,
    question: ChoiceQuestion,
    backend: DeterministicFixtureBackend,
    *,
    calibration_id: str | None = None,
) -> CalibrationArtifact:
    context = calibration_context(request, question.id, len(question.criteria), backend)
    return CalibrationArtifact(
        **context.model_dump(),
        schema_version=1,
        calibration_id=calibration_id or f"fixture-{question.id}-v1",
        status="fixture_only",
        method="temperature-scaling",
        parameters=TemperatureParameters(temperature=1.0),
        fit_sample_count=1,
        evaluation_sample_count=1,
        fit_independent_state_count=1,
        evaluation_independent_state_count=1,
        minimum_independent_state_count=1,
        metrics_before={},
        metrics_after={},
        created_at="2026-09-21T00:00:00Z",
    )


def calibrations_for(
    request: DecisionRequest,
    backend: DeterministicFixtureBackend,
) -> dict[str, CalibrationArtifact]:
    return {
        question.id: calibration_for(request, question, backend) for question in request.questions
    }
