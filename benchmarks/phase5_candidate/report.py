"""Phase 5B loopback wire contract and sanitized systems report.

The Kev service exposes the public TypeSafe-style ``POST /v1/systemone``
contract plus the pinned ``/v1/models``, ``/v1/systemone/separate`` and
``/v1/systemone/permute`` endpoints.  This module only models the closed
response shapes Saracura validates; it never imports Kev or the optional
orchestration modules.
"""

from __future__ import annotations

import math
import re
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

_EVIDENCE_DIGEST = r"^[0-9a-f]{64}$"
_REVISION = r"^[0-9a-f]{40}$"


class ReportModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelsResponse(ReportModel):
    backend: Literal["mlx"]
    dtype: Literal["bfloat16"]
    base_model_id: str
    run: str

    @model_validator(mode="after")
    def exact_local_run(self) -> ModelsResponse:
        # ``run`` is the absolute local checkpoint path on the wire; the
        # sanitized report replaces it with the public candidate id.
        if not self.run.startswith("/"):
            raise ValueError("models.run must be an absolute local checkpoint")
        return self


class ChoiceAnswer(ReportModel):
    choice: str
    probabilities: dict[str, float]
    confidence: float

    @model_validator(mode="after")
    def probabilities_are_valid(self) -> ChoiceAnswer:
        values = tuple(self.probabilities.values())
        if not values or any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values
        ):
            raise ValueError("probabilities must be finite values between zero and one")
        if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-5):
            raise ValueError("probabilities must sum to one")
        if self.choice not in self.probabilities:
            raise ValueError("selected choice must exist in probabilities")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between zero and one")
        return self


class SystemOneResponse(ReportModel):
    answers: dict[str, ChoiceAnswer]
    usage: dict[str, int]
    # run identity is compared internally and sanitized away in public output.
    run: str

    @model_validator(mode="after")
    def response_is_valid(self) -> SystemOneResponse:
        if not self.run.startswith("/"):
            raise ValueError("systemone run must be an absolute local checkpoint")
        if any(value < 0 for value in self.usage.values()):
            raise ValueError("usage counters cannot be negative")
        return self


class EvaluationResult(ReportModel):
    """One measured sample for a workload."""

    latency_ms: float
    answers: tuple[tuple[str, str, float], ...]
    probabilities: dict[str, dict[str, float]]


class CapacityFailure(ReportModel):
    category: str
    detail: str


class SanitizedReport(ReportModel):
    schema_version: Literal["phase5b-report.v1"]
    saracura_commit: Annotated[str, Field(min_length=1)]
    acquisition_descriptor_sha256: Annotated[str, Field(pattern=_EVIDENCE_DIGEST)]
    threshold_digest: Annotated[str, Field(pattern=_EVIDENCE_DIGEST)]
    source_revision: Annotated[str, Field(pattern=_REVISION)]
    model_revision: Annotated[str, Field(pattern=_REVISION)]
    base_model_revision: Annotated[str, Field(pattern=_REVISION)]
    fixture_digest: Annotated[str, Field(pattern=_EVIDENCE_DIGEST)]
    os: str
    architecture: str
    chip_family: str
    physical_memory_bucket: str
    backend: Literal["mlx"]
    dtype: Literal["bfloat16"]
    model_bytes: int
    cold_load_seconds: float | None
    request_p50_ms: float
    request_p95_ms: float
    per_decision_p50_ms: float
    per_decision_p95_ms: float
    decisions_per_second: float
    peak_service_rss_gib: float | None
    peak_physical_footprint_gib: float | None
    swap_used_delta_gib: float | None
    pageout_delta: int | None
    memory_pressure_state: str
    invalid_output_count: int
    repeat_stability_max_delta: float
    isolation_delta_max: float
    option_order_behavior: str
    capacity_failures: tuple[CapacityFailure, ...]
    disposition: Literal["continue", "conditional", "reject_local", "blocked_upstream"]
    raw_confidence_concentration: float | None = None
    state_truncation_contract_failure: bool = False
    per_q_p95_ms: dict[str, float] = Field(default_factory=dict)
    max_request_ms: float = 0.0
    selected_choice_stability: bool = False
    repeat_probability_delta: float = 0.0
    together_separate_probability_delta: float = 0.0
    unauthorized_accepted_count: int = 0

    @model_validator(mode="after")
    def no_disposition_collision(self) -> SanitizedReport:
        if self.state_truncation_contract_failure and self.disposition != "reject_local":
            raise ValueError("state truncation contract failure must produce reject_local")
        sensitive = re.compile(
            r"(?:/(?:Users|private|Volumes|tmp)/|://|\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\b|"
            r"\bBearer\s+\S+|\b[A-Za-z0-9-]+\.(?:com|net|org|local|internal|home|io)\b)",
            re.I,
        )

        def inspect(value: object) -> bool:
            if isinstance(value, str):
                return sensitive.search(value) is not None
            if isinstance(value, dict):
                return any(inspect(item) for item in value.values())
            if isinstance(value, (tuple, list)):
                return any(inspect(item) for item in value)
            return False

        if inspect(self.model_dump()):
            raise ValueError("sanitized report contains sensitive path, host, user, or bearer data")
        return self


_PUBLIC_BUCKETS = ("8", "16", "32", "64", "128", "256")


def memory_bucket(total_bytes: int) -> str:
    gib = total_bytes / (1024**3)
    for upper in (8, 16, 32, 64, 128, 256):
        if gib <= upper:
            return str(upper)
    return "512"
