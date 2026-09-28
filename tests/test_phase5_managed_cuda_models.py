from __future__ import annotations

from typing import Any

import pytest

from benchmarks.phase5_managed_cuda.models import (
    MatrixEvidence,
    OperatorReceipt,
    OversizeBranchOutcome,
    OversizeOutcome,
    OversizeStateOutcome,
    PrivateProvisional,
    PublicSystemsReport,
    ResourceEvidence,
)


def test_models_reject_unknown_fields() -> None:
    with pytest.raises(ValueError):
        OperatorReceipt.model_validate(
            {
                "schema_version": "phase5b-managed-cuda-operator-receipt.v1",
                "preempted": False,
                "protected_workload_restored": True,
                "disposition": "conditional",
            }
        )


def test_matrix_requires_all_twelve_cells_and_twenty_latencies() -> None:
    keys = {
        f"{locale}/{workload}/{kind}"
        for locale in ("pt-BR", "en")
        for workload in ("q1", "q10", "q50")
        for kind in ("new_state", "cached_state")
    }
    cell = {
        "n": 20,
        "p95_ms": 1.0,
        "max_ms": 1.0,
        "latencies_ms": [1.0] * 20,
        "last_response_valid": True,
    }
    evidence = MatrixEvidence.model_validate(
        {"measured_requests": 240, "warmups": 36, "cells": {key: cell for key in keys}}
    )
    assert len(evidence.cells) == 12
    with pytest.raises(ValueError):
        MatrixEvidence.model_validate(
            {
                "measured_requests": 240,
                "warmups": 36,
                "cells": {key: cell for key in list(keys)[:-1]},
            }
        )
    with pytest.raises(ValueError):
        MatrixEvidence.model_validate(
            {
                "measured_requests": 240,
                "warmups": 36,
                "cells": {key: {**cell, "latencies_ms": [1.0]} for key in keys},
            }
        )


def test_sampling_cadence_and_device_capacity_are_closed() -> None:
    base: dict[str, Any] = {
        "peak_rss_bytes": 1,
        "peak_hwm_bytes": 1,
        "peak_vmswap_bytes": 0,
        "candidate_swap_delta_bytes": 0,
        "host_swap_delta_bytes": 0,
        "gpu_peak_mib": 1024.0,
        "device_total_mib": 24576.0,
        "proc_timestamps": [0.0, 0.5, 1.0],
        "nvidia_samples": [
            {
                "started": 0.0,
                "ended": 0.1,
                "duration_ms": 100.0,
                "compute_pids": [5],
                "candidate_gpu_mib": 1024.0,
                "device_total_mib": 24576.0,
            },
            {
                "started": 2.0,
                "ended": 2.1,
                "duration_ms": 100.0,
                "compute_pids": [5],
                "candidate_gpu_mib": 1024.0,
                "device_total_mib": 24576.0,
            },
        ],
        "closing_proc_timestamp": 1.25,
        "closing_nvidia_sample": {
            "started": 2.5,
            "ended": 2.6,
            "duration_ms": 100.0,
            "compute_pids": [5],
            "candidate_gpu_mib": 1024.0,
            "device_total_mib": 24576.0,
        },
        "identity_survived": True,
    }
    assert ResourceEvidence.model_validate(base).device_total_mib == 24576
    with pytest.raises(ValueError):
        ResourceEvidence.model_validate({**base, "proc_timestamps": [0.0, 1.01]})
    with pytest.raises(ValueError):
        ResourceEvidence.model_validate({**base, "device_total_mib": 24575})
    with pytest.raises(ValueError):
        ResourceEvidence.model_validate(
            {
                **base,
                "nvidia_samples": [
                    base["nvidia_samples"][0],
                    {**base["nvidia_samples"][1], "started": 3.01},
                ],
            }
        )
    with pytest.raises(ValueError):
        ResourceEvidence.model_validate(
            {key: value for key, value in base.items() if key != "closing_nvidia_sample"}
        )
    with pytest.raises(ValueError):
        ResourceEvidence.model_validate({**base, "closing_proc_timestamp": 2.01})


def test_oversize_models_require_exact_branch_and_bounded_tokens() -> None:
    assert (
        OversizeOutcome(
            branch=OversizeBranchOutcome(kind="branch", control_status=200, probe_status=422),
            state=OversizeStateOutcome(
                kind="state",
                control_status=200,
                probe_status=200,
                outcome="accepted_with_pinned_source_truncation",
                reported_input_tokens=65_536,
            ),
        ).state.reported_input_tokens
        == 65_536
    )
    with pytest.raises(ValueError):
        OversizeBranchOutcome.model_validate(
            {"kind": "branch", "control_status": 200, "probe_status": 200}
        )
    with pytest.raises(ValueError):
        OversizeStateOutcome(
            kind="state",
            control_status=200,
            probe_status=200,
            outcome="accepted_with_pinned_source_truncation",
            reported_input_tokens=73_728,
        )


def test_public_report_rejects_ip_and_unreviewed_fields() -> None:
    with pytest.raises(ValueError):
        PublicSystemsReport.model_validate({"candidate_host": "127.0.0.1"})
    with pytest.raises(ValueError):
        PublicSystemsReport.model_validate({"pid": 1234})


def test_private_provisional_schema_is_closed() -> None:
    with pytest.raises(ValueError):
        PrivateProvisional.model_validate(
            {
                "schema_version": "phase5b-managed-cuda-provisional.v1",
                "private_url": "http://127.0.0.1",
            }
        )
