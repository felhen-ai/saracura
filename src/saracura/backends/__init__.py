"""Backend protocols, fixture, and the opt-in local MiniLM backend."""

from saracura.backends.base import (
    Backend,
    BackendCalibrationMetadata,
    BackendCapabilities,
    EncodedState,
    ScoredChoice,
)
from saracura.backends.fixture import DeterministicFixtureBackend
from saracura.backends.minilm import MiniLMRoutingBackend

__all__ = [
    "Backend",
    "BackendCalibrationMetadata",
    "BackendCapabilities",
    "DeterministicFixtureBackend",
    "EncodedState",
    "MiniLMRoutingBackend",
    "ScoredChoice",
]
