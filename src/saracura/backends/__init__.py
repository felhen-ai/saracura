"""Backend protocols, fixture, and the opt-in local MiniLM backend."""

from typing import TYPE_CHECKING, Any

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
from saracura.backends.julia import JuliaBackend
from saracura.backends.minilm import MiniLMRoutingBackend
from saracura.backends.saracura_universal import SaracuraUniversalBackend
from saracura.backends.systemone import SystemOneBackend

if TYPE_CHECKING:
    from saracura.backends.laya import LayaUniversalBackend


def __getattr__(name: str) -> Any:
    """Keep Laya's public export without loading it for unrelated backends."""

    if name == "LayaUniversalBackend":
        from saracura.backends.laya import LayaUniversalBackend

        return LayaUniversalBackend
    if name == "SystemOneBackend":
        from saracura.backends.systemone import SystemOneBackend

        return SystemOneBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Backend",
    "BackendCalibrationMetadata",
    "BackendCapabilities",
    "DeterministicFixtureBackend",
    "EncodedState",
    "ExecutionTier",
    "JuliaBackend",
    "LayaUniversalBackend",
    "MiniLMRoutingBackend",
    "SaracuraUniversalBackend",
    "ScoredChoice",
    "SystemOneBackend",
    "UniversalBackend",
]
