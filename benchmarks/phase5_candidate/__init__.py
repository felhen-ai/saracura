"""Phase 5B Kev-4B acquisition and Mac evaluation orchestration.

The four explicit operator steps (``describe``, ``prepare``, ``serve``,
``evaluate``) are the only live acceptance commands.  They never run from
pytest and are not part of ordinary CI.  Optional orchestration dependencies
(``platformdirs``, ``psutil``, ``huggingface_hub``) are imported lazily inside
each command body so the default environment stays weight-free.
"""

from benchmarks.phase5_candidate.descriptor import (
    CANDIDATE_ID,
    SCHEMA_VERSION,
    AcquisitionDescriptor,
)
from benchmarks.phase5_candidate.report import SanitizedReport

__all__ = [
    "CANDIDATE_ID",
    "SCHEMA_VERSION",
    "AcquisitionDescriptor",
    "SanitizedReport",
]
