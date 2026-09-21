"""Closed, versioned contracts for Phase 2A benchmark evidence."""

from __future__ import annotations

import math
import uuid
from datetime import UTC, datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from saracura.backends.fixture import FIXTURE_CHECKPOINT_SHA256
from saracura.runtime.engine import FIXTURE_SPLIT_MANIFEST_SHA256

PublicIdentifier = Annotated[
    str, Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
]


class BenchmarkModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False, strict=True)


class ManifestIdentity(BenchmarkModel):
    id: Literal["self-authored-ptbr-fixture"]
    revision: Literal["v1"]
    sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]

    @model_validator(mode="after")
    def matches_frozen_manifest(self) -> ManifestIdentity:
        if self.sha256 != FIXTURE_SPLIT_MANIFEST_SHA256:
            raise ValueError("fixture manifest hash does not match phase2a.v1")
        return self


class EnvironmentCapture(BenchmarkModel):
    os_family: PublicIdentifier
    os_release: PublicIdentifier
    architecture: PublicIdentifier
    python_implementation: PublicIdentifier
    python_version: PublicIdentifier
    saracura_version: PublicIdentifier
    backend_revision: Literal["fixture-backend-v1"]
    hardware_label: PublicIdentifier | None = None


class ModelIdentity(BenchmarkModel):
    id: Literal["fixture-choice"]
    revision: Literal["fixture-choice-v1"]
    checkpoint_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]

    @model_validator(mode="after")
    def matches_frozen_checkpoint(self) -> ModelIdentity:
        if self.checkpoint_sha256 != FIXTURE_CHECKPOINT_SHA256:
            raise ValueError("fixture checkpoint does not match phase2a.v1")
        return self


class RunIdentity(BenchmarkModel):
    suite: Literal["decision-scaling"]
    schema_version: Literal["phase2a.v1"]
    run_id: str
    created_at: str
    code_revision: PublicIdentifier
    backend_id: Literal["fixture"]
    backend_revision: Literal["fixture-backend-v1"]
    model: ModelIdentity
    environment: EnvironmentCapture

    @model_validator(mode="after")
    def validate_identity(self) -> RunIdentity:
        try:
            parsed_uuid = uuid.UUID(self.run_id)
            parsed_time = datetime.fromisoformat(self.created_at.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError("run identity is not canonical") from error
        if str(parsed_uuid) != self.run_id or parsed_uuid.version != 4:
            raise ValueError("run_id must be a canonical UUID v4")
        if parsed_time.tzinfo != UTC or not self.created_at.endswith("Z"):
            raise ValueError("created_at must be a UTC timestamp")
        if self.environment.backend_revision != self.backend_revision:
            raise ValueError("environment backend revision must agree")
        return self


class WorkloadDefinition(BenchmarkModel):
    question_count: Literal[1, 10, 50]
    warmup_iterations: Annotated[int, Field(ge=0, le=100)]
    measured_iterations: Annotated[int, Field(ge=1, le=1000)]
    state_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    fixture_manifest: ManifestIdentity


class TimingBoundaries(BenchmarkModel):
    total_ms: float = Field(ge=0)
    state_encoding_ms: float = Field(ge=0)
    decision_ms: float = Field(ge=0)

    @model_validator(mode="after")
    def timing_is_consistent(self) -> TimingBoundaries:
        if not all(math.isfinite(value) for value in self.model_dump().values()):
            raise ValueError("timings must be finite")
        if self.total_ms + 1e-9 < self.state_encoding_ms + self.decision_ms:
            raise ValueError("total_ms must include encoding and decision time")
        return self


class Sample(BenchmarkModel):
    phase: Literal["measured"]
    sample_index: Annotated[int, Field(ge=0)]
    state_encoding_count: Annotated[int, Field(ge=1, le=1)]
    timing: TimingBoundaries
    answer_count: Annotated[int, Field(ge=1, le=50)]
    answer_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class SummaryStatistic(BenchmarkModel):
    count: Annotated[int, Field(ge=1)]
    minimum_ms: float = Field(ge=0)
    p50_ms: float = Field(ge=0)
    p95_ms: float = Field(ge=0)
    maximum_ms: float = Field(ge=0)

    @model_validator(mode="after")
    def values_are_finite(self) -> SummaryStatistic:
        values = self.model_dump().values()
        if not all(math.isfinite(value) for value in values if isinstance(value, float)):
            raise ValueError("summary values must be finite")
        return self


class WorkloadResult(BenchmarkModel):
    workload: WorkloadDefinition
    samples: tuple[Sample, ...]
    total: SummaryStatistic
    state_encoding: SummaryStatistic
    decision: SummaryStatistic

    @model_validator(mode="after")
    def validate_samples_and_summaries(self) -> WorkloadResult:
        if len(self.samples) != self.workload.measured_iterations:
            raise ValueError("sample count does not match workload")
        if [sample.sample_index for sample in self.samples] != list(range(len(self.samples))):
            raise ValueError("sample indexes must be contiguous")
        if any(sample.answer_count != self.workload.question_count for sample in self.samples):
            raise ValueError("answer count does not match question count")
        if len({sample.answer_sha256 for sample in self.samples}) != 1:
            raise ValueError("deterministic fixture samples must share one answer digest")
        for field, summary in (
            ("total_ms", self.total),
            ("state_encoding_ms", self.state_encoding),
            ("decision_ms", self.decision),
        ):
            values = [getattr(sample.timing, field) for sample in self.samples]
            expected = summary_from_values(values)
            if summary != expected:
                raise ValueError(f"summary for {field} does not match raw samples")
        return self


class Limitations(BenchmarkModel):
    research_only: Literal[True] = True
    backend_kind: Literal["fixture"] = "fixture"
    quality_claims: Literal[False] = False
    automation_authorized: Literal[False] = False
    interpretation: Literal[
        "Fixture timing validates the harness and runtime invariants only; "
        "it is not learned-model latency or quality evidence. No cold-start or "
        "warm-path claim is made."
    ] = (
        "Fixture timing validates the harness and runtime invariants only; "
        "it is not learned-model latency or quality evidence. No cold-start or "
        "warm-path claim is made."
    )


class BenchmarkResult(BenchmarkModel):
    schema_version: Literal["phase2a.v1"]
    run: RunIdentity
    workloads: tuple[WorkloadResult, ...]
    limitations: Limitations

    @model_validator(mode="after")
    def fixed_workload_order(self) -> BenchmarkResult:
        if [item.workload.question_count for item in self.workloads] != [1, 10, 50]:
            raise ValueError("workloads must be ordered Q=1, Q=10, Q=50")
        first = self.workloads[0].workload
        for item in self.workloads[1:]:
            workload = item.workload
            if workload.state_sha256 != first.state_sha256:
                raise ValueError("all workloads must share the same state hash")
            if workload.fixture_manifest != first.fixture_manifest:
                raise ValueError("all workloads must share the same fixture manifest")
            if workload.warmup_iterations != first.warmup_iterations:
                raise ValueError("all workloads must share warmup configuration")
            if workload.measured_iterations != first.measured_iterations:
                raise ValueError("all workloads must share measured configuration")
        if self.run.environment.backend_revision != self.run.backend_revision:
            raise ValueError("run environment and backend revisions must agree")
        return self


def nearest_rank(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def summary_from_values(values: list[float]) -> SummaryStatistic:
    if not values or any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("summary values must be finite and non-negative")
    return SummaryStatistic(
        count=len(values),
        minimum_ms=min(values),
        p50_ms=nearest_rank(values, 0.50),
        p95_ms=nearest_rank(values, 0.95),
        maximum_ms=max(values),
    )
