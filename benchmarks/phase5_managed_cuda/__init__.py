"""Phase 5B managed CUDA systems evidence package.

Checkout-only benchmark client and sampler for the immutable Kev-4B candidate
on Linux/CUDA.  The runner never opens, renews or closes a GPU lease and never
acts as an HTTP server, runtime backend, scheduler or GPU arbiter.
"""

from __future__ import annotations

__all__ = [
    "OperatorReceipt",
    "OversizeOutcome",
    "PrivateProvisional",
    "PublicSystemsReport",
    "convert_fixture_to_wire",
    "derive_provisional_disposition",
    "execute",
    "finalize",
    "sample_resources",
    "validate_runtime_identity",
]

from benchmarks.phase5_managed_cuda.cli import execute, finalize
from benchmarks.phase5_managed_cuda.disposition import derive_provisional_disposition
from benchmarks.phase5_managed_cuda.models import (
    OperatorReceipt,
    OversizeOutcome,
    PrivateProvisional,
    PublicSystemsReport,
)
from benchmarks.phase5_managed_cuda.runner import sample_resources, validate_runtime_identity
from benchmarks.phase5_managed_cuda.wire import convert_fixture_to_wire
