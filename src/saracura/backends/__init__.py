"""Backend protocols, fixture, and the opt-in local MiniLM backend."""

from saracura.backends.base import (
    Backend,
    BackendCalibrationMetadata,
    BackendCapabilities,
    EncodedState,
    ExecutionTier,
    ScoredChoice,
    UniversalBackend,
)
from saracura.backends.fixture import DeterministicFixtureBackend
from saracura.backends.laya import LayaUniversalBackend
from saracura.backends.minilm import MiniLMRoutingBackend

__all__ = [
    "Backend",
    "BackendCalibrationMetadata",
    "BackendCapabilities",
    "DeterministicFixtureBackend",
    "EncodedState",
    "ExecutionTier",
    "LayaUniversalBackend",
    "MiniLMRoutingBackend",
    "ScoredChoice",
    "UniversalBackend",
]
