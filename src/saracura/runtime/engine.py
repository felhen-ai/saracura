"""Research-only in-process execution with one state encoding per request."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from time import perf_counter
from typing import cast

from pydantic import JsonValue

from saracura.backends.base import Backend, BackendCalibrationMetadata
from saracura.backends.fixture import (
    FIXTURE_ARCHITECTURE_SHA256,
    FIXTURE_OUTPUT_TRANSFORM,
    FIXTURE_TOKENIZER_REVISION,
    FIXTURE_TRUNCATION_POLICY,
    DeterministicFixtureBackend,
)
from saracura.calibration.models import (
    AnyCalibrationArtifact,
    CalibrationContext,
    CalibrationDatasetProfile,
)
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

FIXTURE_DATASET_ID = "self-authored-ptbr-fixture"
FIXTURE_DATASET_REVISION = "v1"
FIXTURE_SPLIT_MANIFEST_SHA256 = "d80de4e329009e0528febd5a6dc0323af14930dea68dbba0f07854e06d44cd71"
MAX_STATE_PAYLOAD_BYTES = 1_000_000
NUMERIC_ISOLATION_ABSOLUTE_TOLERANCE = 1e-12
_IDENTIFIER_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
FIXTURE_CALIBRATION_PROFILE = CalibrationDatasetProfile(
    dataset_id=FIXTURE_DATASET_ID,
    dataset_revision=FIXTURE_DATASET_REVISION,
    split_manifest_sha256=FIXTURE_SPLIT_MANIFEST_SHA256,
)
PHASE4A_IDENTITY_PROFILE = CalibrationDatasetProfile(
    dataset_id="no-fit-synthetic-identity",
    dataset_revision="phase4a.v1",
    split_manifest_sha256="fc73d3e22b04898301563915e5f43051ec627414b83c2c74371f5a2c2b75a0c7",
)

__all__ = [
    "FIXTURE_ARCHITECTURE_SHA256",
    "FIXTURE_CALIBRATION_PROFILE",
    "FIXTURE_DATASET_ID",
    "FIXTURE_DATASET_REVISION",
    "FIXTURE_OUTPUT_TRANSFORM",
    "FIXTURE_SPLIT_MANIFEST_SHA256",
    "FIXTURE_TOKENIZER_REVISION",
    "FIXTURE_TRUNCATION_POLICY",
    "MAX_STATE_PAYLOAD_BYTES",
    "NUMERIC_ISOLATION_ABSOLUTE_TOLERANCE",
    "PHASE4A_IDENTITY_PROFILE",
    "DecisionEngine",
    "calibration_context",
    "cardinality_bucket",
]


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
    profile: CalibrationDatasetProfile | None = None,
) -> CalibrationContext:
    metadata = _calibration_metadata(backend)
    resolved_profile = profile or _implicit_fixture_profile(backend)
    if resolved_profile is None:
        raise SaracuraError(
            ErrorCode.CALIBRATION_MISSING,
            "A non-fixture backend requires an explicit calibration dataset profile.",
            f"/questions/{question_id}/calibration",
        )
    return CalibrationContext(
        model_id=backend.model.id,
        model_revision=backend.model.revision,
        checkpoint_sha256=backend.model.checkpoint_sha256,
        architecture_config_sha256=metadata.architecture_config_sha256,
        serializer_version=SERIALIZER_VERSION,
        tokenizer_revision=metadata.tokenizer_revision,
        truncation_policy_id=metadata.truncation_policy_id,
        precision=metadata.precision,
        quantization=metadata.quantization,
        output_transform=metadata.output_transform,
        workflow_id=request.workflow.id,
        workflow_revision=request.workflow.revision,
        question_id=question_id,
        dataset_id=resolved_profile.dataset_id,
        dataset_revision=resolved_profile.dataset_revision,
        split_manifest_sha256=resolved_profile.split_manifest_sha256,
        locale=request.locale,
        domain=request.domain,
        head="choice",
        cardinality_bucket=cardinality_bucket(criteria_count),
        risk_policy=None,
    )


def _calibration_metadata(backend: Backend) -> BackendCalibrationMetadata:
    """Reject legacy structural backends instead of borrowing fixture metadata."""

    try:
        metadata = backend.calibration_metadata
        if not isinstance(metadata, BackendCalibrationMetadata) or not _metadata_is_valid(metadata):
            raise ValueError("backend calibration metadata is invalid")
    except Exception as error:
        raise SaracuraError(
            ErrorCode.BACKEND_UNAVAILABLE,
            "Backend calibration compatibility metadata is unavailable.",
            "/model",
        ) from error
    return metadata


def _metadata_is_valid(metadata: BackendCalibrationMetadata) -> bool:
    """Match the closed calibration-context constraints before any encoding work."""

    return (
        _is_sha256(metadata.architecture_config_sha256)
        and _is_revision(metadata.tokenizer_revision)
        and all(
            _is_identifier(value)
            for value in (
                metadata.truncation_policy_id,
                metadata.precision,
                metadata.quantization,
                metadata.output_transform,
            )
        )
    )


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256_PATTERN.fullmatch(value) is not None


def _is_revision(value: object) -> bool:
    return isinstance(value, str) and 1 <= len(value) <= 256


def _is_identifier(value: object) -> bool:
    return (
        isinstance(value, str)
        and 1 <= len(value) <= 128
        and _IDENTIFIER_PATTERN.fullmatch(value) is not None
    )


def _implicit_fixture_profile(backend: Backend) -> CalibrationDatasetProfile | None:
    if isinstance(backend, DeterministicFixtureBackend):
        return FIXTURE_CALIBRATION_PROFILE
    # The Phase 2A harness decorates the concrete fixture only to count calls.
    # It cannot turn any other backend into a fixture: the wrapped object must
    # still be the exact deterministic fixture implementation.
    wrapped = getattr(backend, "_backend", None)
    if isinstance(wrapped, DeterministicFixtureBackend):
        return FIXTURE_CALIBRATION_PROFILE
    return None


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
        calibrations: Mapping[str, AnyCalibrationArtifact],
        calibration_profiles: Mapping[str, CalibrationDatasetProfile] | None = None,
    ) -> None:
        self._backend = backend
        self._workflows = workflows
        self._calibrations = calibrations
        self._calibration_profiles = calibration_profiles or {}

    def decide(self, request: DecisionRequest, *, include_timing: bool = False) -> DecisionResponse:
        started = perf_counter()
        self._validate_backend_and_request(request)
        self._workflows.validate(request)
        resolved_calibrations = self._resolve_calibrations(request)

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
        for question, artifact in resolved_calibrations:
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
                input_tokens=self._input_tokens(encoded_state.input_tokens),
                questions=len(request.questions),
                criteria=sum(len(question.criteria) for question in request.questions),
            ),
        )

    def _resolve_calibrations(
        self, request: DecisionRequest
    ) -> tuple[tuple[ChoiceQuestion, AnyCalibrationArtifact], ...]:
        """Resolve and compare every axis before state serialization or device work."""

        resolved: list[tuple[ChoiceQuestion, AnyCalibrationArtifact]] = []
        for question in request.questions:
            artifact = self._calibrations.get(question.id)
            if artifact is None:
                raise SaracuraError(
                    ErrorCode.CALIBRATION_MISSING,
                    "No calibration artifact is registered for this question.",
                    f"/questions/{question.id}",
                )
            profile = self._calibration_profiles.get(question.id)
            expected = calibration_context(
                request,
                question.id,
                len(question.criteria),
                self._backend,
                profile,
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
            resolved.append((question, artifact))
        return tuple(resolved)

    @staticmethod
    def _input_tokens(value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise SaracuraError(
                ErrorCode.BACKEND_UNAVAILABLE,
                "Backend returned invalid token usage.",
                "/model",
            )
        return value

    def _validate_backend_and_request(self, request: DecisionRequest) -> None:
        _calibration_metadata(self._backend)
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
