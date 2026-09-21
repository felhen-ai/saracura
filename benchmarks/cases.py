"""Versioned first-party cases for the decision-scaling benchmark."""

from __future__ import annotations

from saracura.backends.base import Backend
from saracura.calibration import CalibrationArtifact, TemperatureParameters
from saracura.contracts import ChoiceQuestion, DecisionRequest, WorkflowReference
from saracura.runtime import known_scaling_questions
from saracura.runtime.engine import calibration_context

WORKFLOW_ID = "decision-scaling"
WORKFLOW_REVISION = "2026-09-21"


def request_for(question_count: int) -> DecisionRequest:
    if question_count not in (1, 10, 50):
        raise ValueError("question_count must be 1, 10, or 50")
    return DecisionRequest(
        api_version="v1alpha1",
        model="fixture-choice-v1",
        mode="research",
        locale="pt-BR",
        domain="benchmark",
        workflow=WorkflowReference(id=WORKFLOW_ID, revision=WORKFLOW_REVISION),
        state={
            "subject": "Estado autoescrito para o benchmark de escala.",
            "body": "O mesmo estado deve servir a todas as decisões conhecidas.",
        },
        questions=known_scaling_questions(question_count),
    )


def calibrations_for(request: DecisionRequest, backend: Backend) -> dict[str, CalibrationArtifact]:
    return {
        question.id: _calibration_for(request, question, backend) for question in request.questions
    }


def _calibration_for(
    request: DecisionRequest,
    question: ChoiceQuestion,
    backend: Backend,
) -> CalibrationArtifact:
    context = calibration_context(request, question.id, len(question.criteria), backend)
    return CalibrationArtifact(
        **context.model_dump(),
        schema_version=1,
        calibration_id=f"fixture-{question.id}-v1",
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
