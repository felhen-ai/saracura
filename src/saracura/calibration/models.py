"""Immutable research-only calibration artifact and closed compatibility axes."""

from __future__ import annotations

import hashlib
import math
from datetime import datetime
from typing import Annotated, Literal, TypeAlias

from pydantic import Field, JsonValue, TypeAdapter, field_validator, model_validator

from saracura.contracts.models import ClosedModel, Identifier, Locale, Revision
from saracura.serialization import canonical_json_bytes

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class TemperatureParameters(ClosedModel):
    temperature: Annotated[float, Field(ge=0.5, le=5.0)]
    bounds: tuple[float, float] = (0.5, 5.0)

    @field_validator("bounds")
    @classmethod
    def fixed_bounds(cls, value: tuple[float, float]) -> tuple[float, float]:
        if value != (0.5, 5.0):
            raise ValueError("temperature bounds must remain [0.5, 5.0]")
        return value


class FitMetrics(ClosedModel):
    """The three aggregate quantities emitted by the calibration-only fit."""

    nll: Annotated[float, Field(ge=0)]
    brier: Annotated[float, Field(ge=0)]
    ece_15: Annotated[float, Field(ge=0, le=1)]


class PerLabelMetrics(ClosedModel):
    support: Annotated[int, Field(ge=0)]
    precision: Annotated[float, Field(ge=0, le=1)]
    recall: Annotated[float, Field(ge=0, le=1)]
    f1: Annotated[float, Field(ge=0, le=1)]


class BlindMetrics(ClosedModel):
    """Closed, aggregate-only blind metrics.  No row-level evidence is kept."""

    count: Annotated[int, Field(gt=0)]
    per_label: dict[str, PerLabelMetrics]
    accuracy: Annotated[float, Field(ge=0, le=1)]
    macro_f1: Annotated[float, Field(ge=0, le=1)]
    confusion_matrix: list[list[Annotated[int, Field(ge=0)]]]
    nll: Annotated[float, Field(ge=0)]
    brier: Annotated[float, Field(ge=0)]
    ece_15: Annotated[float, Field(ge=0, le=1)]
    mean_confidence: Annotated[float, Field(ge=0, le=1)]
    max_confidence: Annotated[float, Field(ge=0, le=1)]

    @model_validator(mode="after")
    def closed_shape(self) -> BlindMetrics:
        labels = (
            "billing",
            "technical_support",
            "account_access",
            "subscription_cancellation",
            "order_delivery",
        )
        # RFC 8785 sorts object members lexically on disk; membership is
        # closed here while the fixed taxonomy defines metric interpretation.
        if set(self.per_label) != set(labels):
            raise ValueError("blind per-label metrics must use the frozen label set")
        if len(self.confusion_matrix) != 5 or any(len(row) != 5 for row in self.confusion_matrix):
            raise ValueError("blind confusion matrix must be 5 by 5")
        if sum(sum(row) for row in self.confusion_matrix) != self.count:
            raise ValueError("blind confusion matrix count does not reconcile")
        if sum(item.support for item in self.per_label.values()) != self.count:
            raise ValueError("blind per-label support does not reconcile")
        trace = 0
        f1s: list[float] = []
        for index, name in enumerate(labels):
            support = sum(self.confusion_matrix[index])
            predicted = sum(row[index] for row in self.confusion_matrix)
            true_positive = self.confusion_matrix[index][index]
            precision = true_positive / predicted if predicted else 0.0
            recall = true_positive / support if support else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            declared = self.per_label[name]
            if declared.support != support or any(
                not math.isclose(getattr(declared, field), expected, rel_tol=1e-12, abs_tol=1e-12)
                for field, expected in (("precision", precision), ("recall", recall), ("f1", f1))
            ):
                raise ValueError("blind per-label metrics do not reconcile")
            trace += true_positive
            f1s.append(f1)
        if not math.isclose(self.accuracy, trace / self.count, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("blind accuracy does not reconcile")
        if not math.isclose(self.macro_f1, sum(f1s) / 5, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("blind macro F1 does not reconcile")
        return self


class CalibrationContext(ClosedModel):
    model_id: Identifier
    model_revision: Revision
    checkpoint_sha256: Sha256
    architecture_config_sha256: Sha256
    serializer_version: Revision
    tokenizer_revision: Revision
    truncation_policy_id: Identifier
    precision: Identifier
    quantization: Identifier
    output_transform: Identifier
    workflow_id: Identifier
    workflow_revision: Revision
    question_id: Identifier
    dataset_id: Identifier
    dataset_revision: Revision
    split_manifest_sha256: Sha256
    locale: Locale
    domain: Identifier
    head: Literal["choice"]
    cardinality_bucket: Identifier
    risk_policy: None = None


class CalibrationDatasetProfile(ClosedModel):
    """Immutable evidence identity kept separate from model runtime metadata."""

    dataset_id: Identifier
    dataset_revision: Revision
    split_manifest_sha256: Sha256


class CalibrationArtifact(CalibrationContext):
    schema_version: Literal[1]
    calibration_id: Identifier
    # Version 1 has no signed release boundary and is therefore fixture-only.
    status: Literal["fixture_only"]
    method: Literal["temperature-scaling"]
    parameters: TemperatureParameters
    fit_sample_count: Annotated[int, Field(gt=0)]
    evaluation_sample_count: Annotated[int, Field(gt=0)]
    fit_independent_state_count: Annotated[int, Field(gt=0)]
    evaluation_independent_state_count: Annotated[int, Field(gt=0)]
    minimum_independent_state_count: Annotated[int, Field(gt=0)]
    metrics_before: dict[str, JsonValue]
    metrics_after: dict[str, JsonValue]
    created_at: Annotated[str, Field(pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")]

    @model_validator(mode="after")
    def enough_independent_states(self) -> CalibrationArtifact:
        if self.fit_independent_state_count > self.fit_sample_count:
            raise ValueError("fit independent states cannot exceed fit samples")
        if self.evaluation_independent_state_count > self.evaluation_sample_count:
            raise ValueError("evaluation independent states cannot exceed evaluation samples")
        if (
            min(
                self.fit_independent_state_count,
                self.evaluation_independent_state_count,
            )
            < self.minimum_independent_state_count
        ):
            raise ValueError("artifact does not meet its minimum independent state count")
        return self

    def context(self) -> CalibrationContext:
        names = set(CalibrationContext.model_fields)
        values = {name: getattr(self, name) for name in names}
        return CalibrationContext.model_validate(values)


class IdentityParameters(ClosedModel):
    temperature: float = 1.0

    @field_validator("temperature")
    @classmethod
    def fixed_temperature(cls, value: float) -> float:
        if value != 1.0:
            raise ValueError("identity calibration temperature must be exactly 1.0")
        return value


class IdentityCalibrationArtifact(CalibrationContext):
    """No-fit Phase 4A identity transform, never a fitted calibration."""

    schema_version: Literal[1]
    calibration_id: Identifier
    status: Literal["fixture_only"]
    method: Literal["identity"]
    parameters: IdentityParameters
    fit_sample_count: Literal[0]
    evaluation_sample_count: Literal[0]
    fit_independent_state_count: Literal[0]
    evaluation_independent_state_count: Literal[0]
    minimum_independent_state_count: Literal[0]
    metrics_before: dict[str, JsonValue]
    metrics_after: dict[str, JsonValue]
    created_at: Annotated[str, Field(pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")]

    @field_validator("created_at")
    @classmethod
    def canonical_utc_timestamp(cls, value: str) -> str:
        try:
            parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
        except ValueError as error:
            raise ValueError("created_at must be a real UTC calendar timestamp") from error
        if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != value:
            raise ValueError("created_at must be canonical UTC")
        return value

    @model_validator(mode="after")
    def derived_identity_id(self) -> IdentityCalibrationArtifact:
        if self.metrics_before or self.metrics_after:
            raise ValueError("identity calibration metrics must be empty")
        if self.calibration_id != identity_calibration_id(self.context()):
            raise ValueError("identity calibration id does not match its immutable context")
        return self

    def context(self) -> CalibrationContext:
        names = set(CalibrationContext.model_fields)
        values = {name: getattr(self, name) for name in names}
        return CalibrationContext.model_validate(values)


class ResearchCalibrationCandidate(CalibrationContext):
    """Unsigned, non-runtime result of the single blind exposure."""

    schema_version: Literal["research-calibration-candidate.v2"]
    status: Literal["pending_maintainer_release"]
    method: Literal["temperature-scaling"]
    parameters: TemperatureParameters
    fit_sample_count: Annotated[int, Field(gt=0)]
    evaluation_sample_count: Annotated[int, Field(gt=0)]
    fit_independent_state_count: Annotated[int, Field(ge=100)]
    evaluation_independent_state_count: Annotated[int, Field(ge=100)]
    minimum_independent_state_count: Literal[100]
    metrics_before: BlindMetrics
    metrics_after: BlindMetrics
    fit_sha256: Sha256
    blind_view_sha256: Sha256
    exposure_run_sha256: Sha256
    blind_report_sha256: Sha256
    created_at: Annotated[str, Field(pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")]
    candidate_sha256: Sha256

    @model_validator(mode="after")
    def candidate_digest_and_counts(self) -> ResearchCalibrationCandidate:
        _canonical_timestamp(self.created_at)
        if self.fit_independent_state_count > self.fit_sample_count or (
            self.evaluation_independent_state_count > self.evaluation_sample_count
        ):
            raise ValueError("candidate independent states cannot exceed samples")
        payload = self.model_dump(mode="json")
        recorded = payload.pop("candidate_sha256")
        if recorded != hashlib.sha256(canonical_json_bytes(payload)).hexdigest():
            raise ValueError("candidate digest does not match immutable fields")
        return self


class ResearchCalibrationArtifact(CalibrationContext):
    """Signed schema-v2 research artifact; it cannot authorize automation."""

    schema_version: Literal[2]
    status: Literal["verified_for_research"]
    calibration_id: Identifier
    method: Literal["temperature-scaling"]
    parameters: TemperatureParameters
    fit_sample_count: Annotated[int, Field(gt=0)]
    evaluation_sample_count: Annotated[int, Field(gt=0)]
    fit_independent_state_count: Annotated[int, Field(ge=100)]
    evaluation_independent_state_count: Annotated[int, Field(ge=100)]
    minimum_independent_state_count: Literal[100]
    metrics_before: BlindMetrics
    metrics_after: BlindMetrics
    fit_sha256: Sha256
    blind_view_sha256: Sha256
    exposure_run_sha256: Sha256
    blind_report_sha256: Sha256
    candidate_sha256: Sha256
    release_receipt: dict[str, JsonValue]
    release_receipt_sha256: Sha256
    created_at: Annotated[str, Field(pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")]

    @model_validator(mode="after")
    def runtime_artifact_invariants(self) -> ResearchCalibrationArtifact:
        _canonical_timestamp(self.created_at)
        if self.fit_independent_state_count > self.fit_sample_count or (
            self.evaluation_independent_state_count > self.evaluation_sample_count
        ):
            raise ValueError("artifact independent states cannot exceed samples")
        expected = research_calibration_id(
            self.context(),
            self.method,
            self.parameters,
            self.fit_sha256,
            self.blind_view_sha256,
            self.exposure_run_sha256,
            self.blind_report_sha256,
            self.release_receipt_sha256,
        )
        if self.calibration_id != expected:
            raise ValueError("research calibration id does not match immutable evidence")
        return self

    def context(self) -> CalibrationContext:
        names = set(CalibrationContext.model_fields)
        return CalibrationContext.model_validate({name: getattr(self, name) for name in names})

    def dataset_profile(self) -> CalibrationDatasetProfile:
        """Return the structural profile after runtime verification at use."""
        return CalibrationDatasetProfile(
            dataset_id=self.dataset_id,
            dataset_revision=self.dataset_revision,
            split_manifest_sha256=self.split_manifest_sha256,
        )


AnyCalibrationArtifact: TypeAlias = (
    CalibrationArtifact | IdentityCalibrationArtifact | ResearchCalibrationArtifact
)

_ARTIFACT_ADAPTER: TypeAdapter[AnyCalibrationArtifact] = TypeAdapter(
    CalibrationArtifact | IdentityCalibrationArtifact | ResearchCalibrationArtifact
)


def revalidate_calibration_artifact(artifact: AnyCalibrationArtifact) -> AnyCalibrationArtifact:
    """Re-enter the closed artifact union after an in-memory boundary.

    ``model_copy(update=...)`` intentionally skips Pydantic validation.  Runtime
    callers must therefore never infer the schema lane from an object's Python
    class; serialize its declared fields and validate the closed union again.
    """

    return _ARTIFACT_ADAPTER.validate_python(artifact.model_dump(mode="json"))


def candidate_from_research_artifact(
    artifact: ResearchCalibrationArtifact,
) -> ResearchCalibrationCandidate:
    """Reconstruct the exact unsigned candidate frozen into a final artifact.

    This deliberately uses the candidate model rather than comparing a selected
    subset of fields: its digest covers temperature, every context axis, both
    counts, both aggregate metric objects, and the timestamp.
    """

    payload = {
        name: getattr(artifact, name)
        for name in ResearchCalibrationCandidate.model_fields
        if name not in {"schema_version", "status"}
    }
    payload["schema_version"] = "research-calibration-candidate.v2"
    payload["status"] = "pending_maintainer_release"
    return ResearchCalibrationCandidate.model_validate(payload)


def _canonical_timestamp(value: str) -> None:
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as error:
        raise ValueError("created_at must be a real UTC calendar timestamp") from error
    if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != value:
        raise ValueError("created_at must be canonical UTC")


def identity_calibration_id(context: CalibrationContext) -> str:
    """Return the specified deterministic identity-artifact identifier."""

    payload: JsonValue = {
        "context": context.model_dump(mode="json"),
        "method": "identity",
        "parameters": {"temperature": 1.0},
    }
    digest = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    return f"identity-v1-{digest[:32]}"


def research_calibration_id(
    context: CalibrationContext,
    method: str,
    parameters: TemperatureParameters,
    fit_sha256: str,
    blind_view_sha256: str,
    exposure_run_sha256: str,
    blind_report_sha256: str,
    release_receipt_sha256: str,
) -> str:
    """Derive the schema-v2 ID from exactly the frozen evidence tuple."""

    payload: JsonValue = {
        "context": context.model_dump(mode="json"),
        "method": method,
        "parameters": parameters.model_dump(mode="json"),
        "fit_sha256": fit_sha256,
        "blind_view_sha256": blind_view_sha256,
        "exposure_run_sha256": exposure_run_sha256,
        "blind_report_sha256": blind_report_sha256,
        "release_receipt_sha256": release_receipt_sha256,
    }
    digest = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    return f"research-calibration-v2-{digest[:32]}"
