"""Deterministic disposition derivation for Phase 5B managed CUDA evidence.

v1 protocol cannot emit continue: only conditional, reject_local and blocked_evidence
are valid dispositions.
"""

from __future__ import annotations

from typing import Literal

from benchmarks.phase5_managed_cuda.models import OperatorReceipt, PrivateProvisional

Disposition = Literal["conditional", "reject_local", "blocked_evidence"]


def derive_provisional_disposition(
    evidence: PrivateProvisional, operator: OperatorReceipt
) -> Disposition:
    """Derive disposition only from validated provisional and operator evidence."""
    if operator.preempted or not operator.protected_workload_restored:
        return "blocked_evidence"
    if not evidence.runtime_identity.model_dump() or not evidence.direct_upstream_401:
        return "blocked_evidence"
    if (
        not evidence.primary_resources.identity_survived
        or not evidence.diagnostic_resources.identity_survived
    ):
        return "blocked_evidence"
    if (
        evidence.primary_resources.host_swap_delta_bytes
        and not evidence.primary_resources.candidate_swap_delta_bytes
    ):
        return "blocked_evidence"
    if (
        evidence.diagnostic_resources.host_swap_delta_bytes
        and not evidence.diagnostic_resources.candidate_swap_delta_bytes
    ):
        return "blocked_evidence"

    thresholds = {
        "q1": 5000.0,
        "q10": 15000.0,
        "q50": 45000.0,
    }
    matrix_pass = all(
        cell.p95_ms <= thresholds[key.split("/")[1]] and cell.max_ms <= 60000.0
        for key, cell in evidence.matrix.cells.items()
    )
    supplemental_pass = (
        evidence.supplemental.repeat_probability_delta <= 0.000001
        and evidence.supplemental.together_separate_probability_delta <= 0.04
    )
    primary = evidence.primary_resources
    positive_failure = (
        not matrix_pass
        or not supplemental_pass
        or not evidence.supplemental.together_separate_choice_stable
        or primary.peak_rss_bytes > 22 * 1024**3
        or primary.gpu_peak_mib > 16 * 1024
        or primary.candidate_swap_delta_bytes > 1024**3
        or evidence.cold_load_seconds > 600
        or evidence.invalid_outputs != 0
        or evidence.unauthorized_accepted != 0
        or evidence.oversize.branch_status != 422
        or evidence.oversize.invalid_status != 422
    )
    if positive_failure:
        return "reject_local"
    # This v1 protocol binds the exact pinned source and expects its observed
    # state truncation behavior. It can never emit continue.
    if evidence.oversize.state_outcome == "accepted_with_pinned_source_truncation":
        return "conditional"
    return "blocked_evidence"
