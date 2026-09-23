"""Closed request and response models for the research-only v1alpha1 API."""

from __future__ import annotations

import json
from typing import Annotated, Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

Identifier = Annotated[str, Field(min_length=1, max_length=128, pattern=r"^[a-z0-9][a-z0-9._-]*$")]
ModelIdentifier = Annotated[
    str,
    Field(
        min_length=1,
        max_length=256,
        pattern=r"^[a-z0-9][a-z0-9._-]*(?:/[a-z0-9][a-z0-9._-]*)?$",
    ),
]
Revision = Annotated[str, Field(min_length=1, max_length=256)]
Locale = Annotated[
    str, Field(min_length=2, max_length=35, pattern=r"^[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*$")
]


class ClosedModel(BaseModel):
    """Base model that rejects undeclared input and cannot be mutated."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class WorkflowReference(ClosedModel):
    id: Identifier
    revision: Revision


class ChoiceCriterion(ClosedModel):
    id: Identifier
    description: Annotated[str, Field(min_length=1, max_length=4096)]


class ChoiceQuestion(ClosedModel):
    id: Identifier
    type: Literal["choice"]
    instruction: Annotated[str, Field(min_length=1, max_length=4096)]
    criteria: Annotated[tuple[ChoiceCriterion, ...], Field(min_length=2, max_length=20)]

    @model_validator(mode="after")
    def unique_criterion_ids(self) -> ChoiceQuestion:
        ids = [criterion.id for criterion in self.criteria]
        if len(ids) != len(set(ids)):
            raise ValueError("criterion ids must be unique within a question")
        return self


class DecisionRequest(ClosedModel):
    api_version: Literal["v1alpha1"]
    model: Revision
    mode: Literal["research"]
    locale: Locale
    domain: Identifier
    workflow: WorkflowReference
    state: Annotated[dict[str, JsonValue], Field(min_length=1, max_length=256)]
    questions: Annotated[tuple[ChoiceQuestion, ...], Field(min_length=1, max_length=50)]

    @model_validator(mode="after")
    def unique_question_ids(self) -> DecisionRequest:
        ids = [question.id for question in self.questions]
        if len(ids) != len(set(ids)):
            raise ValueError("question ids must be unique within a request")
        return self


class CalibrationReference(ClosedModel):
    id: Identifier
    status: Literal["verified_for_research", "fixture_only"]
    risk_policy_id: None = None


class Answer(ClosedModel):
    question_id: Identifier
    type: Literal["choice"]
    value: Identifier
    raw_scores: dict[str, float]
    probabilities: dict[str, float]
    status: Literal["calibrated", "fixture_only", "uncalibrated"]
    score_semantics: Literal["calibrated_confidence", "fixture_distribution", "ranking_weights"]
    abstained: bool
    reason: str | None
    calibration: CalibrationReference | None

    @model_validator(mode="after")
    def consistent_research_semantics(self) -> Answer:
        if self.status == "uncalibrated":
            if (
                self.score_semantics != "ranking_weights"
                or not self.abstained
                or self.reason != "uncalibrated_research"
                or self.calibration is not None
            ):
                raise ValueError("uncalibrated answers must remain abstained ranking-only results")
            return self

        expected_semantics = (
            "calibrated_confidence" if self.status == "calibrated" else "fixture_distribution"
        )
        expected_calibration_status = (
            "verified_for_research" if self.status == "calibrated" else "fixture_only"
        )
        if (
            self.score_semantics != expected_semantics
            or self.calibration is None
            or self.calibration.status != expected_calibration_status
        ):
            raise ValueError("answer status, semantics, and calibration must agree")
        return self


class ModelReference(ClosedModel):
    id: ModelIdentifier
    revision: Revision
    checkpoint_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class Timing(ClosedModel):
    tokenization_ms: Annotated[float, Field(ge=0)] = 0.0
    state_encoding_ms: Annotated[float, Field(ge=0)] = 0.0
    decision_ms: Annotated[float, Field(ge=0)] = 0.0
    total_ms: Annotated[float, Field(ge=0)] = 0.0


class Usage(ClosedModel):
    input_tokens: Annotated[int, Field(ge=0)]
    questions: Annotated[int, Field(ge=1)]
    criteria: Annotated[int, Field(ge=2)]


class DecisionResponse(ClosedModel):
    api_version: Literal["v1alpha1"]
    answers: tuple[Answer, ...]
    model: ModelReference
    timing: Timing | None = None
    usage: Usage
    automation_allowed: Literal[False] = False


def closed_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Return a JSON schema while preserving Pydantic's closed-object declarations."""

    return model.model_json_schema()


def parse_request_json(raw: bytes | str) -> DecisionRequest:
    """Parse a request and map unsupported top-level regimes to stable error codes."""

    from pydantic import ValidationError

    from saracura.contracts.errors import ErrorCode, SaracuraError

    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, child in pairs:
            if key in value:
                raise ValueError("duplicate JSON object key")
            value[key] = child
        return value

    try:
        payload = json.loads(raw, object_pairs_hook=reject_duplicate_keys)
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
        raise SaracuraError(
            ErrorCode.REQUEST_INVALID,
            "Request is not valid UTF-8 JSON.",
            "/",
        ) from error
    if not isinstance(payload, dict):
        raise SaracuraError(
            ErrorCode.REQUEST_INVALID,
            "Request root must be an object.",
            "/",
        )
    if payload.get("api_version") != "v1alpha1":
        raise SaracuraError(
            ErrorCode.API_VERSION_UNSUPPORTED,
            "Only API version v1alpha1 is supported.",
            "/api_version",
        )
    if payload.get("mode") != "research":
        raise SaracuraError(
            ErrorCode.REQUEST_INVALID,
            "Only research mode is supported in v1alpha1.",
            "/mode",
        )
    questions = payload.get("questions")
    if isinstance(questions, list):
        for index, question in enumerate(questions):
            if isinstance(question, dict) and question.get("type") != "choice":
                raise SaracuraError(
                    ErrorCode.SCHEMA_UNSUPPORTED,
                    "Only choice questions are supported in v1alpha1.",
                    f"/questions/{index}/type",
                )
    try:
        return DecisionRequest.model_validate(payload)
    except ValidationError as error:
        violations = [
            {
                "type": item["type"],
                "path": "/" + "/".join(str(part) for part in item["loc"]),
            }
            for item in error.errors(
                include_url=False,
                include_context=False,
                include_input=False,
            )
        ]
        raise SaracuraError(
            ErrorCode.REQUEST_INVALID,
            "Request failed closed-schema validation.",
            "/",
            details={"violations": cast(JsonValue, violations)},
        ) from error
