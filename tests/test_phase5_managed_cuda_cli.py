from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from benchmarks.phase5_managed_cuda.cli import execute, finalize
from benchmarks.phase5_managed_cuda.models import OperatorReceipt, PrivateProvisional
from benchmarks.phase5_managed_cuda.runner import CandidateRejected, EvidenceBlocked

ROOT = Path(__file__).parents[1]


def _resource(gpu_mib: float = 1000.0) -> dict[str, Any]:
    return {
        "peak_rss_bytes": 1024**3,
        "peak_hwm_bytes": 1024**3,
        "peak_vmswap_bytes": 0,
        "candidate_swap_delta_bytes": 0,
        "host_swap_delta_bytes": 0,
        "gpu_peak_mib": gpu_mib,
        "device_total_mib": 24576.0,
        "proc_timestamps": [0.0, 0.5],
        "nvidia_samples": [
            {
                "started": 0.0,
                "ended": 0.1,
                "duration_ms": 100.0,
                "compute_pids": [123],
                "candidate_gpu_mib": gpu_mib,
                "device_total_mib": 24576.0,
            }
        ],
        "closing_proc_timestamp": 0.75,
        "closing_nvidia_sample": {
            "started": 0.2,
            "ended": 0.3,
            "duration_ms": 100.0,
            "compute_pids": [123],
            "candidate_gpu_mib": gpu_mib,
            "device_total_mib": 24576.0,
        },
        "identity_survived": True,
    }


def _matrix() -> dict[str, Any]:
    cells = {}
    for locale in ("pt-BR", "en"):
        for workload in ("q1", "q10", "q50"):
            for kind in ("new_state", "cached_state"):
                cells[f"{locale}/{workload}/{kind}"] = {
                    "n": 20,
                    "p95_ms": 1000.0,
                    "max_ms": 1100.0,
                    "latencies_ms": [1000.0] * 20,
                    "last_response_valid": True,
                }
    return {"measured_requests": 240, "warmups": 36, "cells": cells}


def _provisional() -> PrivateProvisional:
    return PrivateProvisional.model_validate(
        {
            "schema_version": "phase5b-managed-cuda-provisional.v1",
            "saracura_commit": "a" * 40,
            "archive_sha256": "b" * 64,
            "acquisition_descriptor_sha256": "c" * 64,
            "managed_runtime_sha256": "d" * 64,
            "source_revision": "1" * 40,
            "checkpoint_revision": "2" * 40,
            "base_revision": "3" * 40,
            "base_model_id": "Qwen/Qwen3.5-4B-Base",
            "protocol_sha256": "e" * 64,
            "fixture_sha256": "f" * 64,
            "uv_lock_sha256": "4" * 64,
            "benchmark_package_sha256": "5" * 64,
            "cold_load_seconds": 1.0,
            "direct_upstream_401": True,
            "runtime_identity": {
                "cmdline_valid": True,
                "environment_valid": True,
                "socket_valid": True,
                "cpu_attribution_valid": True,
                "pid": 123,
                "uid": 1000,
                "starttime": 99,
                "socket_inode": 17,
            },
            "matrix": _matrix(),
            "supplemental": {
                "repeat_probability_delta": 0.0,
                "together_separate_probability_delta": 0.0,
                "option_permutation_probability_delta": 0.0,
                "together_separate_choice_stable": True,
            },
            "oversize": {
                "control_status": 200,
                "invalid_status": 422,
                "branch_status": 422,
                "state_status": 200,
                "state_outcome": "accepted_with_pinned_source_truncation",
                "reported_input_tokens": 65607,
            },
            "primary_resources": _resource(),
            "diagnostic_resources": _resource(2000.0),
            "invalid_outputs": 0,
            "unauthorized_accepted": 0,
            "sensitive_values": (
                "http://127.0.0.1:8181",
                "127.0.0.1",
                "/private/checkpoint",
                "123",
            ),
        }
    )


class _Monitor:
    def __init__(self, evidence: dict[str, Any]) -> None:
        self.evidence = evidence
        self.stopped = False
        self.ready = False

    def start(self) -> _Monitor:
        self.ready = True
        return self

    def stop(self) -> dict[str, Any]:
        self.stopped = True
        return self.evidence


def test_execute_orchestrates_and_writes_atomic_private_evidence(tmp_path: Path) -> None:
    output = tmp_path / "private.json"
    monitors: list[_Monitor] = []

    def monitor_factory(_pid: int, *, expected_identity: dict[str, int]) -> _Monitor:
        assert expected_identity["pid"] == 123
        monitor = _Monitor(_resource())
        monitors.append(monitor)
        return monitor

    def oversize_runner(
        _url: str, *, timeout: int, on_primary_complete: Any, diagnostic_request: Any
    ) -> dict[str, Any]:
        on_primary_complete()
        result = diagnostic_request(lambda: object())
        assert result is not None
        return {
            "control_status": 200,
            "invalid_status": 422,
            "branch_status": 422,
            "state_status": 200,
            "state_outcome": "accepted_with_pinned_source_truncation",
            "reported_input_tokens": 65607,
        }

    acquisition = ROOT / "benchmarks/manifests/phase5-kev4b-acquisition.v1.json"
    runtime = ROOT / "benchmarks/manifests/phase5-kev4b-managed-cuda.v1.json"

    def matrix_runner(_url: str) -> dict[str, Any]:
        assert monitors and monitors[0].ready
        return _matrix()

    with patch("benchmarks.phase5_managed_cuda.cli._validate_pid"):
        result = execute(
            gateway_url="http://127.0.0.1:8181",
            direct_upstream_url="http://127.0.0.1:8182",
            expected_checkpoint="/private/checkpoint",
            pid=123,
            cold_load_evidence="1.0",
            acquisition_descriptor_path=acquisition,
            managed_runtime_path=runtime,
            commit="a" * 40,
            archive_digest="b" * 64,
            private_output=output,
            sampler_factory=monitor_factory,
            request=lambda *_args, **_kwargs: type("Response", (), {"status": 401})(),
            matrix_runner=matrix_runner,
            supplemental_runner=lambda _url: {
                "repeat_probability_delta": 0.0,
                "together_separate_probability_delta": 0.0,
                "option_permutation_probability_delta": 0.0,
                "together_separate_choice_stable": True,
            },
            oversize_runner=oversize_runner,
            identity_validator=lambda **_kwargs: {
                "cmdline_valid": True,
                "environment_valid": True,
                "socket_valid": True,
                "cpu_attribution_valid": True,
                "pid": 123,
                "uid": 1000,
                "starttime": 99,
                "socket_inode": 17,
            },
        )
    assert result == output
    assert len(monitors) == 2 and all(monitor.stopped for monitor in monitors)
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    provisional = PrivateProvisional.model_validate_json(output.read_text())
    assert provisional.matrix.measured_requests == 240
    assert provisional.uv_lock_sha256 == hashlib.sha256((ROOT / "uv.lock").read_bytes()).hexdigest()
    assert provisional.diagnostic_resources.gpu_peak_mib == 1000


def test_execute_rejects_non_loopback_and_wrong_direct_port(tmp_path: Path) -> None:
    args: dict[str, Any] = {
        "gateway_url": "http://192.168.1.1:8181",
        "direct_upstream_url": "http://127.0.0.1:8182",
        "expected_checkpoint": "/fake",
        "pid": 123,
        "cold_load_evidence": "1",
        "acquisition_descriptor_path": ROOT / "missing.json",
        "managed_runtime_path": ROOT / "missing.json",
        "commit": "a" * 40,
        "archive_digest": "b" * 64,
        "private_output": tmp_path / "private.json",
    }
    with pytest.raises(ValueError, match=r"127\.0\.0\.1"):
        execute(**args)
    args["gateway_url"] = "http://127.0.0.1:8181"
    args["direct_upstream_url"] = "http://127.0.0.1:8183"
    with pytest.raises(ValueError, match="8182"):
        execute(**args)


@pytest.mark.parametrize("which", ["acquisition", "runtime"])
def test_execute_binds_exact_descriptor_bytes_before_runtime_actions(
    tmp_path: Path, which: str
) -> None:
    acquisition = ROOT / "benchmarks/manifests/phase5-kev4b-acquisition.v1.json"
    runtime = ROOT / "benchmarks/manifests/phase5-kev4b-managed-cuda.v1.json"
    changed = tmp_path / f"{which}.json"
    source = acquisition if which == "acquisition" else runtime
    changed.write_bytes(source.read_bytes() + b" ")
    identity_calls: list[bool] = []
    request_calls: list[bool] = []
    sampler_calls: list[bool] = []
    args: dict[str, Any] = {
        "gateway_url": "http://127.0.0.1:8181",
        "direct_upstream_url": "http://127.0.0.1:8182",
        "expected_checkpoint": "/private/checkpoint",
        "pid": 123,
        "cold_load_evidence": "1",
        "acquisition_descriptor_path": changed if which == "acquisition" else acquisition,
        "managed_runtime_path": changed if which == "runtime" else runtime,
        "commit": "a" * 40,
        "archive_digest": "b" * 64,
        "private_output": tmp_path / "private.json",
        "identity_validator": lambda **_kwargs: identity_calls.append(True),
        "request": lambda *_args, **_kwargs: request_calls.append(True),
        "sampler_factory": lambda *_args, **_kwargs: sampler_calls.append(True),
    }
    with (
        patch("benchmarks.phase5_managed_cuda.cli._validate_pid") as pid_validator,
        pytest.raises(ValueError, match="bytes do not match"),
    ):
        execute(**args)
    pid_validator.assert_not_called()
    assert identity_calls == request_calls == sampler_calls == []
    assert not (tmp_path / "private.json").exists()


@pytest.mark.parametrize(
    ("failure", "expected_stage", "expected_classification"),
    [
        ("identity", "runtime_identity", "blocked_evidence"),
        ("direct_503", "direct_upstream_auth", "blocked_evidence"),
        ("direct_transport", "direct_upstream_auth", "blocked_evidence"),
        ("primary_ready", "primary_monitor_readiness", "blocked_evidence"),
        ("matrix", "matrix", "blocked_evidence"),
        ("supplemental", "supplemental", "blocked_evidence"),
        ("oversize", "oversize", "blocked_evidence"),
        ("diagnostic_ready", "diagnostic_monitor_readiness", "blocked_evidence"),
        ("resource_stop", "resource_finalization", "blocked_evidence"),
        ("candidate_invalid", "matrix", "reject_local"),
    ],
)
def test_execute_persists_finalizable_blocked_failure_for_each_mandatory_stage(
    tmp_path: Path, failure: str, expected_stage: str, expected_classification: str
) -> None:
    from benchmarks.phase5_managed_cuda.models import PrivateFailureProvisional, PublicFailureReport

    private = tmp_path / "private.json"
    acquisition = ROOT / "benchmarks/manifests/phase5-kev4b-acquisition.v1.json"
    runtime = ROOT / "benchmarks/manifests/phase5-kev4b-managed-cuda.v1.json"
    identity = {
        "cmdline_valid": True,
        "environment_valid": True,
        "socket_valid": True,
        "cpu_attribution_valid": True,
        "pid": 123,
        "uid": 1000,
        "starttime": 99,
        "socket_inode": 17,
    }
    monitor_count = 0

    class Monitor(_Monitor):
        def start(self) -> Monitor:
            if (failure == "primary_ready" and monitor_count == 1) or (
                failure == "diagnostic_ready" and monitor_count == 2
            ):
                raise EvidenceBlocked("injected sampler readiness failure")
            super().start()
            return self

        def stop(self) -> dict[str, Any]:
            if failure == "resource_stop" and monitor_count == 1:
                raise EvidenceBlocked("injected final sampler failure")
            return super().stop()

    def monitor_factory(_pid: int, *, expected_identity: dict[str, int]) -> Monitor:
        nonlocal monitor_count
        assert expected_identity == identity
        monitor_count += 1
        return Monitor(_resource())

    def matrix_runner(_url: str) -> dict[str, Any]:
        if failure == "matrix":
            raise EvidenceBlocked("injected matrix transport failure")
        if failure == "candidate_invalid":
            from benchmarks.phase5_managed_cuda.runner import CandidateRejected

            raise CandidateRejected("injected invalid response")
        return _matrix()

    def supplemental_runner(_url: str) -> dict[str, Any]:
        if failure == "supplemental":
            raise EvidenceBlocked("injected supplemental transport failure")
        return {
            "repeat_probability_delta": 0.0,
            "together_separate_probability_delta": 0.0,
            "option_permutation_probability_delta": 0.0,
            "together_separate_choice_stable": True,
        }

    def oversize_runner(
        _url: str, *, timeout: int, on_primary_complete: Any, diagnostic_request: Any
    ) -> dict[str, Any]:
        on_primary_complete()
        diagnostic_request(lambda: object())
        if failure == "oversize":
            raise EvidenceBlocked("injected oversize state drift")
        return {
            "control_status": 200,
            "invalid_status": 422,
            "branch_status": 422,
            "state_status": 200,
            "state_outcome": "accepted_with_pinned_source_truncation",
            "reported_input_tokens": 65607,
        }

    def direct_request(*_args: Any, **_kwargs: Any) -> Any:
        if failure == "direct_transport":
            raise EvidenceBlocked("injected transport failure")
        status = 503 if failure == "direct_503" else 401
        return type("Response", (), {"status": status})()

    args: dict[str, Any] = {
        "gateway_url": "http://127.0.0.1:8181",
        "direct_upstream_url": "http://127.0.0.1:8182",
        "expected_checkpoint": "/private/checkpoint",
        "pid": 123,
        "cold_load_evidence": "1",
        "acquisition_descriptor_path": acquisition,
        "managed_runtime_path": runtime,
        "commit": "a" * 40,
        "archive_digest": "b" * 64,
        "private_output": private,
        "sampler_factory": monitor_factory,
        "request": direct_request,
        "matrix_runner": matrix_runner,
        "supplemental_runner": supplemental_runner,
        "oversize_runner": oversize_runner,
        "identity_validator": lambda **_kwargs: (
            (_ for _ in ()).throw(EvidenceBlocked("injected identity drift"))
            if failure == "identity"
            else identity
        ),
    }
    with (
        patch("benchmarks.phase5_managed_cuda.cli._validate_pid"),
        pytest.raises((EvidenceBlocked, CandidateRejected)),
    ):
        execute(**args)
    assert private.exists() and stat.S_IMODE(private.stat().st_mode) == 0o600
    assert list(tmp_path.iterdir()) == [private]
    blocked = PrivateFailureProvisional.model_validate_json(private.read_text())
    assert blocked.stage == expected_stage
    assert blocked.classification == expected_classification
    assert blocked.evidence_code

    operator = tmp_path / "operator.json"
    operator.write_text(
        OperatorReceipt(
            schema_version="phase5b-managed-cuda-operator-receipt.v1",
            preempted=failure == "candidate_invalid",
            protected_workload_restored=True,
        ).model_dump_json(),
        encoding="utf-8",
    )
    public = tmp_path / "public.json"
    finalize(
        private_input=private,
        operator_receipt=operator,
        public_output=public,
        host_facts=lambda: ("linux", "x86_64", "32-63"),
    )
    report = PublicFailureReport.model_validate_json(public.read_text())
    expected_disposition = (
        "blocked_evidence" if failure == "candidate_invalid" else expected_classification
    )
    assert report.disposition == expected_disposition
    serialized = report.model_dump_json()
    assert all(
        value not in serialized for value in ("127.0.0.1", "8181", "8182", "123", "/private")
    )
    assert "resources" not in report.model_dump()
    assert stat.S_IMODE(public.stat().st_mode) == 0o644


def test_blocked_failure_operator_restoration_loss_overrides_reject_local(
    tmp_path: Path,
) -> None:
    from benchmarks.phase5_managed_cuda.models import PrivateFailureProvisional, PublicFailureReport

    private = tmp_path / "private.json"
    failure = PrivateFailureProvisional(
        schema_version="phase5b-managed-cuda-failure-provisional.v1",
        saracura_commit="a" * 40,
        archive_sha256="b" * 64,
        stage="matrix",
        classification="reject_local",
        evidence_code="candidate_invalid_response",
        sensitive_values=("private-input",),
    )
    private.write_text(failure.model_dump_json(), encoding="utf-8")
    operator = tmp_path / "operator.json"
    operator.write_text(
        OperatorReceipt(
            schema_version="phase5b-managed-cuda-operator-receipt.v1",
            preempted=False,
            protected_workload_restored=False,
        ).model_dump_json(),
        encoding="utf-8",
    )
    public = tmp_path / "public.json"
    finalize(
        private_input=private,
        operator_receipt=operator,
        public_output=public,
        host_facts=lambda: ("linux", "x86_64", "32-63"),
    )
    report = PublicFailureReport.model_validate_json(public.read_text())
    assert report.disposition == "blocked_evidence"
    assert "private-input" not in public.read_text()


def test_atomic_create_preserves_existing_bytes(tmp_path: Path) -> None:
    target = tmp_path / "result.json"
    target.write_bytes(b"original")
    from benchmarks.phase5_managed_cuda.cli import _atomic_create

    with pytest.raises(ValueError, match="refusing to clobber"):
        _atomic_create(target, b"changed", 0o644)
    assert target.read_bytes() == b"original"
    assert list(tmp_path.iterdir()) == [target]


def test_atomic_create_cleans_temporary_file_after_link_failure(tmp_path: Path) -> None:
    from benchmarks.phase5_managed_cuda.cli import _atomic_create

    target = tmp_path / "result.json"
    with (
        patch("benchmarks.phase5_managed_cuda.cli.os.link", side_effect=OSError("injected")),
        pytest.raises(OSError, match="injected"),
    ):
        _atomic_create(target, b"content", 0o600)
    assert not target.exists()
    assert list(tmp_path.iterdir()) == []


def test_finalize_derives_disposition_and_sanitizes_exact_private_values(tmp_path: Path) -> None:
    private = tmp_path / "private.json"
    private.write_text(_provisional().model_dump_json())
    operator = tmp_path / "operator.json"
    operator.write_text(
        OperatorReceipt(
            schema_version="phase5b-managed-cuda-operator-receipt.v1",
            preempted=False,
            protected_workload_restored=True,
        ).model_dump_json()
    )
    output = tmp_path / "public.json"
    finalize(
        private_input=private,
        operator_receipt=operator,
        public_output=output,
        host_facts=lambda: ("linux", "x86_64", "32-63"),
    )
    report = json.loads(output.read_text())
    assert report["disposition"] == "conditional"
    assert report["state_size_guard_required"] is True
    serialized = json.dumps(report)
    for forbidden in ("127.0.0.1", "/private/checkpoint", "123", "candidate_host"):
        assert forbidden not in serialized
    assert stat.S_IMODE(output.stat().st_mode) == 0o644


@pytest.mark.parametrize(
    "sensitive",
    [
        "http://127.0.0.1:8181",
        "127.0.0.1",
        "/private/checkpoint",
        "pid=987654321",
        "host.internal.example",
        "operator@example.test",
        "Bearer abc.def.ghi",
    ],
)
def test_public_sanitizer_rejects_exact_private_and_pattern_values(
    tmp_path: Path, sensitive: str
) -> None:
    from benchmarks.phase5_managed_cuda.cli import _assert_public_sanitized
    from benchmarks.phase5_managed_cuda.models import PublicSystemsReport

    private = _provisional().model_dump()
    private["sensitive_values"] = (sensitive,)
    source = tmp_path / "private.json"
    source.write_text(json.dumps(private))
    operator = tmp_path / "operator.json"
    operator.write_text(
        OperatorReceipt(
            schema_version="phase5b-managed-cuda-operator-receipt.v1",
            preempted=False,
            protected_workload_restored=True,
        ).model_dump_json()
    )
    output = tmp_path / "report.json"
    finalize(
        private_input=source,
        operator_receipt=operator,
        public_output=output,
        host_facts=lambda: ("linux", "x86_64", "32-63"),
    )
    report = PublicSystemsReport.model_validate_json(output.read_text())
    dirty = report.model_copy(update={"architecture": sensitive})
    with pytest.raises(ValueError):
        _assert_public_sanitized(dirty, (sensitive,))


def test_inherited_proxy_and_provider_values_do_not_enter_transport_or_evidence() -> None:
    from benchmarks.phase5_managed_cuda import runner

    captured: dict[str, Any] = {}

    class Response:
        status = 200

        def __init__(self) -> None:
            self.headers: dict[str, str] = {}

        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return b"{}"

    class Opener:
        def open(self, request: Any, *, timeout: float) -> Response:
            captured["headers"] = dict(request.header_items())
            captured["data"] = request.data
            captured["timeout"] = timeout
            return Response()

    with (
        patch.dict(
            os.environ,
            {
                "HTTPS_PROXY": "http://private-proxy.invalid:9911",
                "OPENAI_API_KEY": "provider-secret-value",
                "HF_TOKEN": "hf_private_value",
            },
            clear=False,
        ),
        patch.object(urllib.request, "build_opener") as build,
    ):
        build.return_value = Opener()
        result = runner.request_json(
            "http://127.0.0.1:8181/v1/systemone", {"model": "kev-latest"}, timeout=2
        )
        handlers = build.call_args.args
        assert any(
            isinstance(item, urllib.request.ProxyHandler) and getattr(item, "proxies", None) == {}
            for item in handlers
        )
    assert result.status == 200
    serialized = json.dumps({**captured, "data": captured["data"].decode()})
    assert "Authorization" not in captured["headers"]
    assert all(
        value not in serialized for value in ("private-proxy", "provider-secret", "hf_private")
    )


def test_finalize_preemption_overrides_measured_thresholds(tmp_path: Path) -> None:
    private = tmp_path / "private.json"
    evidence = _provisional().model_dump()
    evidence["primary_resources"]["peak_rss_bytes"] = 30 * 1024**3
    private.write_text(json.dumps(evidence))
    operator = tmp_path / "operator.json"
    operator.write_text(
        json.dumps(
            {
                "schema_version": "phase5b-managed-cuda-operator-receipt.v1",
                "preempted": True,
                "protected_workload_restored": True,
            }
        )
    )
    output = tmp_path / "public.json"
    finalize(
        private_input=private,
        operator_receipt=operator,
        public_output=output,
        host_facts=lambda: ("linux", "x86_64", "32-63"),
    )
    assert json.loads(output.read_text())["disposition"] == "blocked_evidence"


def test_candidate_swap_over_one_gib_is_positive_reject_evidence(tmp_path: Path) -> None:
    private = tmp_path / "private.json"
    evidence = _provisional().model_dump()
    evidence["primary_resources"]["candidate_swap_delta_bytes"] = 1024**3 + 1
    private.write_text(json.dumps(evidence), encoding="utf-8")
    operator = tmp_path / "operator.json"
    operator.write_text(
        OperatorReceipt(
            schema_version="phase5b-managed-cuda-operator-receipt.v1",
            preempted=False,
            protected_workload_restored=True,
        ).model_dump_json(),
        encoding="utf-8",
    )
    output = tmp_path / "report.json"
    finalize(
        private_input=private,
        operator_receipt=operator,
        public_output=output,
        host_facts=lambda: ("linux", "x86_64", "32-63"),
    )
    assert json.loads(output.read_text())["disposition"] == "reject_local"


def test_finalize_requires_restoration_and_rejects_receipt_disposition(tmp_path: Path) -> None:
    private = tmp_path / "private.json"
    private.write_text(_provisional().model_dump_json())
    operator = tmp_path / "operator.json"
    operator.write_text(
        json.dumps(
            {
                "schema_version": "phase5b-managed-cuda-operator-receipt.v1",
                "preempted": False,
                "protected_workload_restored": False,
                "disposition": "conditional",
            }
        )
    )
    with pytest.raises(ValueError):
        finalize(
            private_input=private,
            operator_receipt=operator,
            public_output=tmp_path / "public.json",
            host_facts=lambda: ("linux", "x86_64", "32-63"),
        )


def test_clean_archive_sync_is_frozen_offline_and_digest_bound() -> None:
    changed = subprocess.run(
        ["git", "ls-files", "-m", "-o", "--exclude-standard"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.splitlines()
    allowed = (
        "benchmarks/phase5_managed_cuda/",
        "benchmarks/manifests/phase5-kev4b-managed-cuda.v1.json",
        "benchmarks/manifests/phase5-managed-cuda-systems.v1.json",
        "benchmarks/validate_manifests.py",
        "benchmarks/phase5_candidate/client.py",
        "tests/test_phase5_managed_cuda",
        "tests/test_manifests.py",
        "docs/action/specs/phase5b-managed-cuda-systems-closure.md",
    )
    scoped = [path for path in changed if path.startswith(allowed)]
    assert scoped
    with tempfile.TemporaryDirectory(prefix="phase5b-archive-", dir=ROOT) as temporary:
        temp = Path(temporary)
        index = temp / "index"
        env = dict(os.environ, GIT_INDEX_FILE=str(index))
        subprocess.run(["git", "read-tree", "HEAD"], cwd=ROOT, env=env, check=True)
        subprocess.run(["git", "add", "-A", "--", *scoped], cwd=ROOT, env=env, check=True)
        tree = subprocess.run(
            ["git", "write-tree"], cwd=ROOT, env=env, text=True, capture_output=True, check=True
        ).stdout.strip()
        archive = temp / "phase5b.tar"
        with archive.open("wb") as stream:
            subprocess.run(
                ["git", "archive", "--format=tar", tree], cwd=ROOT, stdout=stream, check=True
            )
        checkout = temp / "checkout"
        checkout.mkdir()
        with tarfile.open(archive) as bundle:
            bundle.extractall(checkout, filter="data")
        lock_digest = hashlib.sha256((checkout / "uv.lock").read_bytes()).hexdigest()
        package_digest = hashlib.sha256()
        package = checkout / "benchmarks/phase5_managed_cuda"
        for path in sorted(package.glob("*.py")):
            package_digest.update(path.name.encode())
            package_digest.update(path.read_bytes())
        assert lock_digest == hashlib.sha256((ROOT / "uv.lock").read_bytes()).hexdigest()
        from benchmarks.phase5_managed_cuda.cli import _benchmark_package_digest

        assert package_digest.hexdigest() == _benchmark_package_digest(
            ROOT / "benchmarks/phase5_managed_cuda"
        )
        subprocess.run(
            ["uv", "sync", "--frozen", "--offline", "--directory", str(checkout)],
            cwd=checkout,
            env=dict(os.environ, UV_OFFLINE="1", UV_NO_PROGRESS="1"),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        subprocess.run(
            ["uv", "sync", "--no-dev", "--frozen", "--offline", "--directory", str(checkout)],
            cwd=checkout,
            env=dict(os.environ, UV_OFFLINE="1", UV_NO_PROGRESS="1"),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        subprocess.run(
            [str(checkout / ".venv/bin/python"), "benchmarks/validate_default_environment.py"],
            cwd=checkout,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
