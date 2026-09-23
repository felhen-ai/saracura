"""Minimal backend protocol independent of any model vendor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from saracura.contracts.models import ChoiceQuestion, DecisionRequest, ModelReference

ExecutionTier = Literal["compiled", "universal"]


@dataclass(frozen=True, slots=True)
class BackendCalibrationMetadata:
    """The model/runtime axes that make a calibration reusable."""

    architecture_config_sha256: str
    tokenizer_revision: str
    truncation_policy_id: str
    precision: str
    quantization: str
    output_transform: str


@dataclass(frozen=True, slots=True)
class BackendCapabilities:
    execution_tier: ExecutionTier
    decision_types: frozenset[str]
    max_questions: int
    max_criteria: int
    execution_boundary: str
    cold_warm_semantics: str
    quality_claims: bool


@dataclass(frozen=True, slots=True)
class EncodedState:
    payload: bytes
    input_tokens: int = 0
    opaque: object | None = None


@dataclass(frozen=True, slots=True)
class ScoredChoice:
    question_id: str
    raw_scores: dict[str, float]
    input_tokens: int = 0


class Backend(Protocol):
    @property
    def capabilities(self) -> BackendCapabilities: ...

    @property
    def model(self) -> ModelReference: ...

    @property
    def calibration_metadata(self) -> BackendCalibrationMetadata: ...

    def encode_state(self, state_payload: bytes) -> EncodedState: ...

    def score_choice(
        self,
        encoded_state: EncodedState,
        question: ChoiceQuestion,
        question_payload: bytes,
    ) -> ScoredChoice: ...


class UniversalBackend(Protocol):
    """Question-conditioned backend for reviewed dynamic Choice workflows."""

    @property
    def capabilities(self) -> BackendCapabilities: ...

    @property
    def model(self) -> ModelReference: ...

    def score_universal_choice(
        self,
        request: DecisionRequest,
        question: ChoiceQuestion,
        state_payload: bytes,
    ) -> ScoredChoice: ...
