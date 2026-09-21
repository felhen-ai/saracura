"""Minimal backend protocol independent of any model vendor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from saracura.contracts.models import ChoiceQuestion, ModelReference


@dataclass(frozen=True, slots=True)
class BackendCapabilities:
    decision_types: frozenset[str]
    max_questions: int
    max_criteria: int
    execution_boundary: str
    cold_warm_semantics: str
    quality_claims: bool


@dataclass(frozen=True, slots=True)
class EncodedState:
    payload: bytes


@dataclass(frozen=True, slots=True)
class ScoredChoice:
    question_id: str
    raw_scores: dict[str, float]


class Backend(Protocol):
    @property
    def capabilities(self) -> BackendCapabilities: ...

    @property
    def model(self) -> ModelReference: ...

    def encode_state(self, state_payload: bytes) -> EncodedState: ...

    def score_choice(
        self,
        encoded_state: EncodedState,
        question: ChoiceQuestion,
        question_payload: bytes,
    ) -> ScoredChoice: ...
