"""Closed, content-minimizing contracts for the generic shadow evaluator."""

from __future__ import annotations

import json
import math
import re
import unicodedata
from hashlib import sha256
from typing import Annotated, Any, Final, Literal, TypeVar, cast

from pydantic import Field, StrictStr, ValidationError, field_validator, model_validator

from saracura.contracts.errors import ErrorCode, SaracuraError
from saracura.contracts.models import (
    ChoiceCriterion,
    ClosedModel,
    DecisionResponse,
    Identifier,
    ModelReference,
    Revision,
    WorkflowReference,
)
from saracura.serialization import canonical_json_bytes
from saracura.universal.tasks import UNIVERSAL_DOMAINS

SHADOW_POLICY_SCHEMA: Final = "saracura-shadow-policy.v1"
SHADOW_ITEM_SCHEMA: Final = "saracura-shadow-item.v1"
SHADOW_DECISION_SCHEMA: Final = "saracura-shadow-decision.v1"
SHADOW_FEEDBACK_SCHEMA: Final = "saracura-shadow-feedback.v1"
SHADOW_SUMMARY_SCHEMA: Final = "saracura-shadow-evaluation.v1"
SHADOW_STATE_MAX_FIELDS: Final = 16
SHADOW_STATE_MAX_CODEPOINTS: Final = 200
SHADOW_STATE_MAX_BYTES: Final = 800
_UNIVERSAL_WORKFLOW_ID = "universal-choice"
_WORKFLOW_REVISIONS = frozenset({"phase4e-saracura-ranker.v1"})
_ALIASES = frozenset({"latest", "main", "master"})
_ITEM_REFERENCE_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _invalid(message: str, path: str = "/") -> SaracuraError:
    return SaracuraError(ErrorCode.REQUEST_INVALID, message, path)


def _is_nfc(value: str) -> bool:
    return unicodedata.normalize("NFC", value) == value


def _reject_non_nfc(value: str) -> None:
    if not _is_nfc(value):
        raise ValueError("text must already be NFC")


def _reject_alias(value: str) -> None:
    if value.casefold() in _ALIASES:
        raise ValueError("aliases are not immutable revisions")


def _shadow_state_capacity_issue(state: object) -> Literal["invalid", "capacity", "ok"]:
    if type(state) is not dict or not state:
        return "invalid"
    if len(state) > SHADOW_STATE_MAX_FIELDS:
        return "capacity"
    for key, value in state.items():
        if type(key) is not str or type(value) is not str:
            return "invalid"
        if not _is_nfc(key) or not _is_nfc(value):
            return "invalid"
    try:
        canonical = canonical_json_bytes(cast(Any, state))
        codepoints = len(canonical.decode("utf-8"))
    except (SaracuraError, UnicodeError, ValueError, TypeError):
        return "invalid"
    if codepoints > SHADOW_STATE_MAX_CODEPOINTS or len(canonical) > SHADOW_STATE_MAX_BYTES:
        return "capacity"
    return "ok"


def validate_shadow_state_capacity(state: object) -> None:
    """Validate canonical public state capacity before constructing a ShadowItem.

    This provider-neutral boundary owns JSON framing, UTF-8 byte count, and
    codepoint limits. Policy-key equality remains the responsibility of
    ``validate_shadow_item_state``.
    """

    issue = _shadow_state_capacity_issue(state)
    if issue == "capacity":
        raise SaracuraError(
            ErrorCode.CAPACITY_EXCEEDED,
            "Shadow state exceeds the installed capacity.",
            "/state",
        )
    if issue == "invalid":
        raise _invalid("Shadow state is invalid.", "/state")


class ShadowPolicy(ClosedModel):
    """Immutable data binding one universal choice policy and its accepted state keys."""

    schema_version: Literal["saracura-shadow-policy.v1"]
    id: Identifier
    revision: Revision
    model: Revision
    locale: Literal["pt-BR", "en"]
    domain: Identifier
    workflow: WorkflowReference
    state_keys: Annotated[tuple[Identifier, ...], Field(min_length=1, max_length=16)]
    question_id: Identifier
    instruction: Annotated[StrictStr, Field(min_length=1, max_length=120)]
    criteria: Annotated[tuple[ChoiceCriterion, ...], Field(min_length=2, max_length=8)]

    @model_validator(mode="after")
    def policy_is_supported_and_bounded(self) -> ShadowPolicy:
        if self.domain not in UNIVERSAL_DOMAINS:
            raise ValueError("domain is not in the closed universal taxonomy")
        if (
            self.workflow.id != _UNIVERSAL_WORKFLOW_ID
            or self.workflow.revision not in _WORKFLOW_REVISIONS
        ):
            raise ValueError("workflow revision is unsupported")
        _reject_alias(self.revision)
        _reject_alias(self.model)
        _reject_alias(self.workflow.revision)
        for value in (self.revision, self.model, self.workflow.id, self.workflow.revision):
            _reject_non_nfc(value)
        if len(self.state_keys) != len(set(self.state_keys)):
            raise ValueError("state keys must be unique")
        if len(self.criteria) != len({criterion.id for criterion in self.criteria}):
            raise ValueError("criterion ids must be unique")
        _reject_non_nfc(self.instruction)
        if len(self.instruction.encode("utf-8")) > 480:
            raise ValueError("instruction exceeds the installed byte limit")
        for criterion in self.criteria:
            _reject_non_nfc(criterion.description)
            if len(criterion.description) > 120 or len(criterion.description.encode("utf-8")) > 480:
                raise ValueError("criterion description exceeds the installed limits")
        return self


class ShadowItem(ClosedModel):
    """One in-memory-only item using caller-owned opaque reference and string state."""

    schema_version: Literal["saracura-shadow-item.v1"]
    item_ref: Annotated[StrictStr, Field(min_length=1, max_length=128)]
    state: Annotated[
        dict[Identifier, StrictStr],
        Field(min_length=1, max_length=SHADOW_STATE_MAX_FIELDS),
    ]

    @field_validator("state", mode="before")
    @classmethod
    def require_string_state_keys(cls, value: object) -> object:
        if not isinstance(value, dict) or any(type(key) is not str for key in value):
            raise ValueError("state keys must be strings")
        return value

    @field_validator("item_ref")
    @classmethod
    def opaque_reference_shape(cls, value: str) -> str:
        if _ITEM_REFERENCE_PATTERN.fullmatch(value) is None:
            raise ValueError("item reference is not an opaque lowercase identifier")
        return value

    @model_validator(mode="after")
    def state_is_nfc_and_bounded(self) -> ShadowItem:
        issue = _shadow_state_capacity_issue(self.state)
        if issue == "invalid":
            raise ValueError("state is invalid")
        if issue == "capacity":
            raise ValueError("state exceeds the installed capacity")
        return self


class ShadowRankingEntry(ClosedModel):
    """One content-free policy label and its uncalibrated ranking weight."""

    label: Identifier
    ranking_weight: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]

    @field_validator("ranking_weight", mode="before")
    @classmethod
    def reject_boolean_weight(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("ranking weight must be numeric")
        return value


class ShadowDecisionRecord(ClosedModel):
    """Content-free projection of a single uncalibrated universal decision."""

    schema_version: Literal["saracura-shadow-decision.v1"]
    item_ref: Annotated[StrictStr, Field(min_length=1, max_length=128)]
    policy_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    model: ModelReference
    suggested_label: Identifier
    ranking: Annotated[tuple[ShadowRankingEntry, ...], Field(min_length=2, max_length=8)]
    status: Literal["uncalibrated"]
    score_semantics: Literal["ranking_weights"]
    abstained: Literal[True]
    reason: Literal["uncalibrated_research"]
    automation_allowed: Literal[False]

    @field_validator("item_ref")
    @classmethod
    def opaque_reference_shape(cls, value: str) -> str:
        if _ITEM_REFERENCE_PATTERN.fullmatch(value) is None:
            raise ValueError("item reference is not an opaque lowercase identifier")
        return value

    @model_validator(mode="after")
    def ranking_is_closed_and_ordered(self) -> ShadowDecisionRecord:
        _reject_non_nfc(self.model.id)
        _reject_non_nfc(self.model.revision)
        labels = [entry.label for entry in self.ranking]
        if len(labels) != len(set(labels)):
            raise ValueError("ranking labels must be unique")
        if self.suggested_label != self.ranking[0].label:
            raise ValueError("suggested label must be first in ranking order")
        if any(
            left.ranking_weight < right.ranking_weight
            for left, right in zip(self.ranking, self.ranking[1:], strict=False)
        ):
            raise ValueError("ranking weights must be descending")
        if not math.isclose(sum(item.ranking_weight for item in self.ranking), 1.0, abs_tol=1e-6):
            raise ValueError("ranking weights must sum to one")
        return self


class ShadowFeedbackRecord(ClosedModel):
    """Content-free operator label, or an explicit skipped disposition."""

    schema_version: Literal["saracura-shadow-feedback.v1"]
    item_ref: Annotated[StrictStr, Field(min_length=1, max_length=128)]
    policy_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    disposition: Literal["labeled", "skipped"]
    operator_label: Identifier | None

    @field_validator("item_ref")
    @classmethod
    def opaque_reference_shape(cls, value: str) -> str:
        if _ITEM_REFERENCE_PATTERN.fullmatch(value) is None:
            raise ValueError("item reference is not an opaque lowercase identifier")
        return value

    @model_validator(mode="after")
    def disposition_matches_label(self) -> ShadowFeedbackRecord:
        if self.disposition == "labeled" and self.operator_label is None:
            raise ValueError("labeled feedback requires an operator label")
        if self.disposition == "skipped" and self.operator_label is not None:
            raise ValueError("skipped feedback cannot contain an operator label")
        return self


class ShadowEvaluationSummary(ClosedModel):
    """Descriptive aggregate only; it is not a fit or calibration artifact."""

    schema_version: Literal["saracura-shadow-evaluation.v1"]
    policy_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    prediction_count: Annotated[int, Field(ge=0, strict=True)]
    feedback_count: Annotated[int, Field(ge=0, strict=True)]
    labeled_count: Annotated[int, Field(ge=0, strict=True)]
    skipped_count: Annotated[int, Field(ge=0, strict=True)]
    unreviewed_prediction_count: Annotated[int, Field(ge=0, strict=True)]
    feedback_coverage: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    agreement_count: Annotated[int, Field(ge=0, strict=True)]
    agreement_rate: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)] | None
    confusion_matrix: dict[Identifier, dict[Identifier, Annotated[int, Field(ge=0, strict=True)]]]

    @model_validator(mode="after")
    def counts_and_matrix_are_consistent(self) -> ShadowEvaluationSummary:
        if (
            self.feedback_count != self.labeled_count + self.skipped_count
            or self.unreviewed_prediction_count != self.prediction_count - self.feedback_count
            or self.feedback_count > self.prediction_count
            or self.agreement_count > self.labeled_count
        ):
            raise ValueError("summary counts are inconsistent")
        expected_coverage = (
            self.feedback_count / self.prediction_count if self.prediction_count else 0.0
        )
        expected_agreement = (
            self.agreement_count / self.labeled_count if self.labeled_count else None
        )
        if not math.isclose(self.feedback_coverage, expected_coverage, abs_tol=1e-12):
            raise ValueError("feedback coverage is inconsistent")
        if expected_agreement is None:
            if self.agreement_rate is not None:
                raise ValueError("zero-labeled agreement rate must be null")
        elif self.agreement_rate is None or not math.isclose(
            self.agreement_rate, expected_agreement, abs_tol=1e-12
        ):
            raise ValueError("agreement rate is inconsistent")
        labels = set(self.confusion_matrix)
        if not labels or any(set(row) != labels for row in self.confusion_matrix.values()):
            raise ValueError("confusion matrix axes must be a closed square")
        if sum(sum(row.values()) for row in self.confusion_matrix.values()) != self.labeled_count:
            raise ValueError("confusion matrix total is inconsistent")
        diagonal_total = sum(self.confusion_matrix[label][label] for label in labels)
        if diagonal_total != self.agreement_count:
            raise ValueError("confusion matrix agreement total is inconsistent")
        return self


ModelT = TypeVar("ModelT", bound=ClosedModel)


def _revalidate(model: ModelT, model_type: type[ModelT], message: str) -> ModelT:
    if not isinstance(model, model_type):
        raise _invalid(message)
    try:
        validated = model_type.model_validate(model.model_dump(mode="python"))
    except (ValidationError, ValueError, TypeError):
        invalid = True
    else:
        invalid = False
    if invalid:
        raise _invalid(message)
    return validated


def _parse_shadow_json(
    raw: bytes | str,
    model_type: type[ModelT],
    *,
    message: str,
) -> ModelT:
    """Parse duplicate-free JSON and return a stable error without input context."""

    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        duplicate = False
        for key, child in pairs:
            if key in result:
                duplicate = True
            result[key] = child
        if duplicate:
            raise ValueError("duplicate object key")
        return result

    try:
        payload = json.loads(raw, object_pairs_hook=reject_duplicate_keys)
        result = model_type.model_validate(payload)
    except (UnicodeError, ValueError, TypeError, ValidationError):
        invalid = True
    else:
        invalid = False
    if invalid:
        raise _invalid(message)
    return result


def parse_shadow_policy_json(raw: bytes | str) -> ShadowPolicy:
    """Parse a policy with duplicate-key and closed-schema rejection."""

    return _parse_shadow_json(raw, ShadowPolicy, message="Shadow policy is invalid.")


def parse_shadow_item_json(raw: bytes | str) -> ShadowItem:
    """Parse one item without reflecting any state key or value in errors."""

    return _parse_shadow_json(raw, ShadowItem, message="Shadow item is invalid.")


def parse_shadow_decision_json(raw: bytes | str) -> ShadowDecisionRecord:
    """Parse one closed content-free decision record."""

    return _parse_shadow_json(raw, ShadowDecisionRecord, message="Shadow decision is invalid.")


def parse_shadow_feedback_json(raw: bytes | str) -> ShadowFeedbackRecord:
    """Parse one closed content-free feedback record."""

    return _parse_shadow_json(raw, ShadowFeedbackRecord, message="Shadow feedback is invalid.")


def shadow_policy_digest(policy: ShadowPolicy) -> str:
    """Return the SHA-256 binding of canonical, NFC-normalized policy JSON."""

    policy = _revalidate(policy, ShadowPolicy, "Shadow policy is invalid.")
    payload = cast(Any, policy.model_dump(mode="json"))
    return sha256(canonical_json_bytes(payload)).hexdigest()


def validate_shadow_item_state(policy: ShadowPolicy, item: ShadowItem) -> None:
    """Require the item state to match policy keys exactly, without echoing names."""

    policy = _revalidate(policy, ShadowPolicy, "Shadow policy is invalid.")
    item = _revalidate(item, ShadowItem, "Shadow item is invalid.")
    validate_shadow_state_capacity(item.state)
    if set(item.state) != set(policy.state_keys):
        raise _invalid("Shadow item state does not match policy state keys.", "/state")


def project_shadow_decision(
    policy: ShadowPolicy,
    item_ref: str,
    response: DecisionResponse,
) -> ShadowDecisionRecord:
    """Project one existing response to an ordered, content-free shadow record."""

    policy = _revalidate(policy, ShadowPolicy, "Shadow policy is invalid.")
    response = _revalidate(response, DecisionResponse, "Decision response is invalid.")
    labels = tuple(criterion.id for criterion in policy.criteria)
    label_set = set(labels)
    if (
        response.automation_allowed is not False
        or response.model.revision != policy.model
        or len(response.answers) != 1
    ):
        raise _invalid("Decision response does not match the shadow policy.")
    answer = response.answers[0]
    if (
        answer.question_id != policy.question_id
        or answer.type != "choice"
        or answer.status != "uncalibrated"
        or answer.score_semantics != "ranking_weights"
        or answer.abstained is not True
        or answer.reason != "uncalibrated_research"
        or answer.calibration is not None
        or set(answer.raw_scores) != label_set
        or set(answer.probabilities) != label_set
        or answer.value not in label_set
        or any(
            isinstance(score, bool) or not math.isfinite(score)
            for score in answer.raw_scores.values()
        )
        or any(
            isinstance(weight, bool) or not math.isfinite(weight) or not 0 <= weight <= 1
            for weight in answer.probabilities.values()
        )
    ):
        raise _invalid("Decision response does not match the shadow policy.")
    order = {label: index for index, label in enumerate(labels)}
    ranking_values = sorted(
        answer.probabilities.items(), key=lambda item: (-item[1], order[item[0]])
    )
    ranking = tuple(
        ShadowRankingEntry(label=label, ranking_weight=weight) for label, weight in ranking_values
    )
    if answer.value != ranking_values[0][0]:
        raise _invalid("Decision response ranking is inconsistent.")
    try:
        record = ShadowDecisionRecord(
            schema_version=SHADOW_DECISION_SCHEMA,
            item_ref=item_ref,
            policy_sha256=shadow_policy_digest(policy),
            model=response.model,
            suggested_label=ranking_values[0][0],
            ranking=ranking,
            status="uncalibrated",
            score_semantics="ranking_weights",
            abstained=True,
            reason="uncalibrated_research",
            automation_allowed=False,
        )
    except (ValidationError, ValueError, TypeError):
        invalid = True
    else:
        invalid = False
    if invalid:
        raise _invalid("Decision response cannot be projected to a shadow record.")
    return record
