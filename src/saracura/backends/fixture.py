"""Deterministic non-model backend for contract and runtime tests only."""

from __future__ import annotations

import hashlib
from typing import Final, Literal

from saracura.backends.base import (
    BackendCalibrationMetadata,
    BackendCapabilities,
    EncodedState,
    ScoredChoice,
)
from saracura.contracts.models import ChoiceQuestion, ModelReference
from saracura.serialization import frame_segments

FIXTURE_BACKEND_ID: Final[Literal["fixture"]] = "fixture"
FIXTURE_BACKEND_REVISION: Final[Literal["fixture-backend-v1"]] = "fixture-backend-v1"
FIXTURE_MODEL_ID: Final[Literal["fixture-choice"]] = "fixture-choice"
FIXTURE_MODEL_REVISION: Final[Literal["fixture-choice-v1"]] = "fixture-choice-v1"
FIXTURE_CHECKPOINT_SHA256 = hashlib.sha256(b"saracura-fixture-backend-v1").hexdigest()
FIXTURE_ARCHITECTURE_SHA256 = "d6b67f81f2469584707a7aafc7a945b98c19decc3b932254034b7f10bbaad18a"
FIXTURE_TOKENIZER_REVISION = "fixture-bytes-v1"
FIXTURE_TRUNCATION_POLICY = "no-truncation-v1"
FIXTURE_OUTPUT_TRANSFORM = "choice-softmax-v1"


class DeterministicFixtureBackend:
    """Hash-based fixture with no learned parameters and no quality claim."""

    def __init__(self) -> None:
        self.state_encode_calls = 0
        self._model = ModelReference(
            id=FIXTURE_MODEL_ID,
            revision=FIXTURE_MODEL_REVISION,
            checkpoint_sha256=FIXTURE_CHECKPOINT_SHA256,
        )
        self._capabilities = BackendCapabilities(
            execution_tier="compiled",
            decision_types=frozenset({"choice"}),
            max_questions=50,
            max_criteria=20,
            execution_boundary="in-process-test-fixture",
            cold_warm_semantics="deterministic-no-load",
            quality_claims=False,
        )
        self._calibration_metadata = BackendCalibrationMetadata(
            architecture_config_sha256=FIXTURE_ARCHITECTURE_SHA256,
            tokenizer_revision=FIXTURE_TOKENIZER_REVISION,
            truncation_policy_id=FIXTURE_TRUNCATION_POLICY,
            precision="fp64",
            quantization="none",
            output_transform=FIXTURE_OUTPUT_TRANSFORM,
        )

    @property
    def capabilities(self) -> BackendCapabilities:
        return self._capabilities

    @property
    def model(self) -> ModelReference:
        return self._model

    @property
    def calibration_metadata(self) -> BackendCalibrationMetadata:
        return self._calibration_metadata

    def encode_state(self, state_payload: bytes) -> EncodedState:
        self.state_encode_calls += 1
        return EncodedState(payload=hashlib.sha256(state_payload).digest())

    def score_choice(
        self,
        encoded_state: EncodedState,
        question: ChoiceQuestion,
        question_payload: bytes,
    ) -> ScoredChoice:
        scores: dict[str, float] = {}
        for criterion in sorted(question.criteria, key=lambda item: item.id):
            digest = hashlib.sha256(
                frame_segments(encoded_state.payload, question_payload, criterion.id)
            ).digest()
            integer = int.from_bytes(digest[:8], byteorder="big", signed=False)
            scores[criterion.id] = integer / float(2**64)
        return ScoredChoice(question_id=question.id, raw_scores=scores)
