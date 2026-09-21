"""Backend protocols and first-party test fixture."""

from saracura.backends.base import Backend, BackendCapabilities, EncodedState, ScoredChoice
from saracura.backends.fixture import DeterministicFixtureBackend

__all__ = [
    "Backend",
    "BackendCapabilities",
    "DeterministicFixtureBackend",
    "EncodedState",
    "ScoredChoice",
]
