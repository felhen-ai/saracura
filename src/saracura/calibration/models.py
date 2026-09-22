"""Immutable research-only calibration artifact and closed compatibility axes."""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Annotated, Literal, TypeAlias

from pydantic import Field, JsonValue, field_validator, model_validator

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
            raise ValueError("temperature bounds must remain [0.5, 5.0] in schema version 1")
        return value


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
    status: Literal["verified_for_research", "fixture_only"]
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


AnyCalibrationArtifact: TypeAlias = CalibrationArtifact | IdentityCalibrationArtifact


def identity_calibration_id(context: CalibrationContext) -> str:
    """Return the specified deterministic identity-artifact identifier."""

    payload: JsonValue = {
        "context": context.model_dump(mode="json"),
        "method": "identity",
        "parameters": {"temperature": 1.0},
    }
    digest = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    return f"identity-v1-{digest[:32]}"
