"""Checkout-only Phase 4C.2 local-candidate tools.

Importing this package remains lightweight. Optional ML libraries are imported
only by explicit loader, freeze, or run commands and never by the installed
Saracura runtime.
"""

from benchmarks.universal_local.registry import (
    Candidate,
    get_candidate,
    load_registry,
    registry_digest,
    validate_registry,
)

__all__ = ["Candidate", "get_candidate", "load_registry", "registry_digest", "validate_registry"]
