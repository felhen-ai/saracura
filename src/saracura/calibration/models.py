"""Immutable research-only calibration artifact and closed compatibility axes."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, JsonValue, field_validator, model_validator

from saracura.contracts.models import ClosedModel, Identifier, Locale, Revision

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
