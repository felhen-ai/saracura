"""Research-only in-process execution with one state encoding per request."""

from __future__ import annotations

import math
from collections.abc import Mapping
from time import perf_counter
from typing import cast

from pydantic import JsonValue

from saracura.backends.base import Backend
from saracura.calibration.models import CalibrationArtifact, CalibrationContext
from saracura.contracts.errors import ErrorCode, SaracuraError
from saracura.contracts.models import (
    Answer,
    CalibrationReference,
    ChoiceQuestion,
    DecisionRequest,
    DecisionResponse,
    Timing,
    Usage,
)
from saracura.runtime.workflows import WorkflowRegistry
from saracura.serialization import SERIALIZER_VERSION, serialize_question, serialize_state

FIXTURE_ARCHITECTURE_SHA256 = "d6b67f81f2469584707a7aafc7a945b98c19decc3b932254034b7f10bbaad18a"
FIXTURE_TOKENIZER_REVISION = "fixture-bytes-v1"
FIXTURE_TRUNCATION_POLICY = "no-truncation-v1"
FIXTURE_OUTPUT_TRANSFORM = "choice-softmax-v1"
FIXTURE_DATASET_ID = "self-authored-ptbr-fixture"
FIXTURE_DATASET_REVISION = "v1"
FIXTURE_SPLIT_MANIFEST_SHA256 = "d80de4e329009e0528febd5a6dc0323af14930dea68dbba0f07854e06d44cd71"
MAX_STATE_PAYLOAD_BYTES = 1_000_000
NUMERIC_ISOLATION_ABSOLUTE_TOLERANCE = 1e-12


def cardinality_bucket(cardinality: int) -> str:
    if cardinality < 2:
        raise ValueError("choice requires at least two criteria")
    if cardinality <= 5:
        return "2-5"
    if cardinality <= 20:
        return "6-20"
    raise SaracuraError(
        ErrorCode.CARDINALITY_EXCEEDED,
        "Choice cardinality exceeds the verified maximum.",
        "/questions/*/criteria",
    )


def calibration_context(
    request: DecisionRequest,
    question_id: str,
    criteria_count: int,
    backend: Backend,
) -> CalibrationContext:
    return CalibrationContext(
        model_id=backend.model.id,
        model_revision=backend.model.revision,
        checkpoint_sha256=backend.model.checkpoint_sha256,
        architecture_config_sha256=FIXTURE_ARCHITECTURE_SHA256,
        serializer_version=SERIALIZER_VERSION,
        tokenizer_revision=FIXTURE_TOKENIZER_REVISION,
        truncation_policy_id=FIXTURE_TRUNCATION_POLICY,
        precision="fp64",
        quantization="none",
        output_transform=FIXTURE_OUTPUT_TRANSFORM,
        workflow_id=request.workflow.id,
        workflow_revision=request.workflow.revision,
        question_id=question_id,
        dataset_id=FIXTURE_DATASET_ID,
        dataset_revision=FIXTURE_DATASET_REVISION,
        split_manifest_sha256=FIXTURE_SPLIT_MANIFEST_SHA256,
        locale=request.locale,
        domain=request.domain,
        head="choice",
        cardinality_bucket=cardinality_bucket(criteria_count),
        risk_policy=None,
    )


def _probabilities(scores: dict[str, float], temperature: float) -> dict[str, float]:
    scaled = {key: value / temperature for key, value in scores.items()}
    peak = max(scaled.values())
    exponentials = {key: math.exp(value - peak) for key, value in scaled.items()}
    denominator = sum(exponentials.values())
    return {key: value / denominator for key, value in exponentials.items()}


class DecisionEngine:
    def __init__(
        self,
        *,
        backend: Backend,
        workflows: WorkflowRegistry,
        calibrations: Mapping[str, CalibrationArtifact],
    ) -> None:
        self._backend = backend
        self._workflows = workflows
        self._calibrations = calibrations

    def decide(self, request: DecisionRequest, *, include_timing: bool = False) -> DecisionResponse:
        started = perf_counter()
        self._validate_backend_and_request(request)
        self._workflows.validate(request)

        encoding_started = perf_counter()
        state_payload = serialize_state(request)
        if len(state_payload) > MAX_STATE_PAYLOAD_BYTES:
            raise SaracuraError(
                ErrorCode.REQUEST_INVALID,
                "Serialized state exceeds the research runtime byte limit.",
                "/state",
                details={"max_bytes": MAX_STATE_PAYLOAD_BYTES},
            )
        encoded_state = self._backend.encode_state(state_payload)
        encoding_finished = perf_counter()

        answers: list[Answer] = []
        for question in request.questions:
            artifact = self._calibrations.get(question.id)
            if artifact is None:
                raise SaracuraError(
                    ErrorCode.CALIBRATION_MISSING,
                    "No calibration artifact is registered for this question.",
                    f"/questions/{question.id}",
                )
            expected = calibration_context(
                request, question.id, len(question.criteria), self._backend
            )
            actual = artifact.context().model_dump(mode="json")
            wanted = expected.model_dump(mode="json")
            mismatches = sorted(key for key in wanted if actual[key] != wanted[key])
            if mismatches:
                raise SaracuraError(
                    ErrorCode.CALIBRATION_INCOMPATIBLE,
                    "Calibration artifact is incompatible with the request and runtime.",
                    f"/questions/{question.id}/calibration",
                    details={"mismatched_axes": cast(JsonValue, mismatches)},
                )
            scored = self._backend.score_choice(
                encoded_state,
                question,
                serialize_question(question),
            )
            self._validate_scored_choice(question, scored.question_id, scored.raw_scores)
            probabilities = _probabilities(
                scored.raw_scores,
                artifact.parameters.temperature,
            )
            value = max(probabilities, key=probabilities.__getitem__)
            answers.append(
                Answer(
                    question_id=question.id,
                    type="choice",
                    value=value,
                    raw_scores=scored.raw_scores,
                    probabilities=probabilities,
                    status=(
                        "calibrated"
                        if artifact.status == "verified_for_research"
                        else "fixture_only"
                    ),
                    abstained=False,
                    reason=None,
                    calibration=CalibrationReference(
                        id=artifact.calibration_id,
                        status=artifact.status,
                        risk_policy_id=None,
                    ),
                )
            )
        finished = perf_counter()

        timing = None
        if include_timing:
            timing = Timing(
                tokenization_ms=0.0,
                state_encoding_ms=(encoding_finished - encoding_started) * 1000,
                decision_ms=(finished - encoding_finished) * 1000,
                total_ms=(finished - started) * 1000,
            )
        return DecisionResponse(
            api_version="v1alpha1",
            answers=tuple(answers),
            model=self._backend.model,
            timing=timing,
            usage=Usage(
                input_tokens=0,
                questions=len(request.questions),
                criteria=sum(len(question.criteria) for question in request.questions),
            ),
        )

    def _validate_backend_and_request(self, request: DecisionRequest) -> None:
        if request.model != self._backend.model.revision:
            code = (
                ErrorCode.MODEL_ALIAS_FORBIDDEN
                if request.model in {"latest", "main", "master"}
                else ErrorCode.MODEL_NOT_FOUND
            )
            raise SaracuraError(
                code,
                "The immutable model revision is not available.",
                "/model",
            )
        if len(request.questions) > self._backend.capabilities.max_questions:
            raise SaracuraError(
                ErrorCode.CARDINALITY_EXCEEDED,
                "Question count exceeds backend capabilities.",
                "/questions",
            )
        if "choice" not in self._backend.capabilities.decision_types:
            raise SaracuraError(
                ErrorCode.BACKEND_UNAVAILABLE,
                "Backend does not support choice decisions.",
                "/model",
            )
        if any(
            len(question.criteria) > self._backend.capabilities.max_criteria
            for question in request.questions
        ):
            raise SaracuraError(
                ErrorCode.CARDINALITY_EXCEEDED,
                "Choice cardinality exceeds backend capabilities.",
                "/questions/*/criteria",
            )

    @staticmethod
    def _validate_scored_choice(
        question: ChoiceQuestion,
        scored_question_id: str,
        scores: Mapping[str, float],
    ) -> None:
        expected_ids = {criterion.id for criterion in question.criteria}
        if not all(isinstance(identifier, str) for identifier in scores):
            raise SaracuraError(
                ErrorCode.BACKEND_UNAVAILABLE,
                "Backend returned a non-string criterion identifier.",
                f"/questions/{question.id}",
            )
        actual_ids = set(scores)
        if scored_question_id != question.id or actual_ids != expected_ids:
            raise SaracuraError(
                ErrorCode.BACKEND_UNAVAILABLE,
                "Backend returned scores for an incompatible question schema.",
                f"/questions/{question.id}",
                details={
                    "missing_criteria": cast(JsonValue, sorted(expected_ids - actual_ids)),
                    "unexpected_criteria": cast(JsonValue, sorted(actual_ids - expected_ids)),
                },
            )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in scores.values()
        ):
            raise SaracuraError(
                ErrorCode.BACKEND_UNAVAILABLE,
                "Backend returned a non-finite score.",
                f"/questions/{question.id}",
            )
