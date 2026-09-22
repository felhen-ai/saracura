"""Checkout-only registry and acquisition tools for Phase 4C.2a.

This package deliberately has no optional ML imports.  Model loaders are a
later phase and must not become part of the installed runtime.
"""

from benchmarks.universal_local.registry import (
    Candidate,
    get_candidate,
    load_registry,
    registry_digest,
    validate_registry,
)

__all__ = ["Candidate", "get_candidate", "load_registry", "registry_digest", "validate_registry"]
