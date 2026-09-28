"""Offline boundary tests for the Phase 5B acquisition operations.

These tests never touch the network, never import the optional orchestration
modules, and never read weights.  They exercise the fail-closed boundaries that
govern describe/prepare/serve/evaluate before any live run.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from collections import namedtuple
from pathlib import Path

import pytest

from benchmarks.phase5_candidate import operations
from benchmarks.phase5_candidate.descriptor import AcquisitionDescriptor


def _install_fake_psutil(monkeypatch: pytest.MonkeyPatch, connections: list[str]) -> None:
    import sys
    import types
    from typing import Any

    psutil = types.ModuleType("psutil")

    class _NoSuchProcess(Exception):
        pass

    _Laddr = namedtuple("_Laddr", ["ip", "port"])
    _Conn = namedtuple("_Conn", ["laddr", "raddr", "status"])

    class _Process:
        def __init__(self, pid: int) -> None:
            if pid != 1:
                raise _NoSuchProcess()

        def net_connections(self, kind: str) -> list[Any]:
            return [
                _Conn(laddr=_Laddr(ip, 8000), raddr=None, status="LISTEN") for ip in connections
            ]

    psutil.NoSuchProcess = _NoSuchProcess  # type: ignore[attr-defined]
    psutil.Process = _Process  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "psutil", psutil)


def test_require_optional_modules_fails_closed_when_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    # Simulate the default environment where ``platformdirs`` is not importable.
    import builtins
    from typing import Any

    real_import = builtins.__import__

    def _fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "platformdirs":
            raise ImportError("no module named platformdirs")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    with pytest.raises(ImportError):
        operations._require_optional_modules()


def test_loops_only_fails_closed_when_psutil_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins
    from typing import Any

    real_import = builtins.__import__

    def _fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "psutil":
            raise ImportError("no module named psutil")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    # Missing psutil must fail closed (raise), never return True.
    with pytest.raises(ImportError):
        operations._loops_only(1)


def test_loops_only_fails_closed_when_process_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_psutil(monkeypatch, [])
    assert operations._loops_only(2**22) is False


def test_loops_only_rejects_non_loopback(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_psutil(monkeypatch, ["127.0.0.1", "192.168.1.5"])
    assert operations._loops_only(1) is False


def test_loops_only_accepts_pure_loopback(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_psutil(monkeypatch, ["127.0.0.1"])
    assert operations._loops_only(1) is True


def test_pid_port_ownership_requires_exact_loopback_listener(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_psutil(monkeypatch, ["127.0.0.1"])
    assert operations._pid_owns_port(1, 8000) is True
    assert operations._pid_owns_port(1, 8001) is False


def test_allowlisted_url_rejects_unlisted_host() -> None:
    with pytest.raises(ValueError, match="allowlist"):
        operations._allowlisted_url("https://example.evil.com/model")


def test_allowlisted_url_accepts_allowlist_subdomain() -> None:
    operations._allowlisted_url("https://cdn-lfs.huggingface.co/x/y")
    operations._allowlisted_url("https://github.com/jaredpalmer/kev")


def test_verify_ledger_rejects_extra_file(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "b.txt").write_text("b")
    entries = [
        {"path": "a.txt", "bytes": 1, "sha256": hashlib.sha256(b"a").hexdigest(), "role": "source"}
    ]
    with pytest.raises(RuntimeError, match="ledger"):
        operations._verify_ledger(tmp_path, entries, "source")


def test_verify_ledger_rejects_digest_mismatch(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("a")
    entries = [{"path": "a.txt", "bytes": 1, "sha256": "0" * 64, "role": "source"}]
    with pytest.raises(RuntimeError, match="ledger mismatch"):
        operations._verify_ledger(tmp_path, entries, "source")


def test_verify_ledger_accepts_exact_tree(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("a")
    data = b"a"
    entries = [
        {"path": "a.txt", "bytes": 1, "sha256": hashlib.sha256(data).hexdigest(), "role": "source"}
    ]
    operations._verify_ledger(tmp_path, entries, "source")


def test_source_checkout_removes_git_metadata_before_ledgering(tmp_path: Path) -> None:
    checkout = tmp_path / "source" / "kev"
    (checkout / ".git" / "objects").mkdir(parents=True)
    (checkout / ".git" / "objects" / "metadata").write_text("private")
    (checkout / "tracked.py").write_text("print('fixture')\n")

    operations._remove_git_metadata(checkout)

    assert (checkout / "tracked.py").is_file()
    assert not (checkout / ".git").exists()
    entries = operations._ledger_for(checkout.parent, "source")
    operations._verify_ledger(checkout.parent, entries, "source")


def test_hf_safety_flags_override_inherited_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_HUB_DISABLE_IMPLICIT_TOKEN", "0")
    monkeypatch.setenv("HF_HUB_DISABLE_XET", "0")

    operations._force_hf_safety_flags()

    assert __import__("os").environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] == "1"
    assert __import__("os").environ["HF_HUB_DISABLE_XET"] == "1"


def test_prepare_rejects_descriptor_capacity_before_acquisition() -> None:
    descriptor = _minimal_descriptor().model_copy(
        update={"checkpoint": _minimal_descriptor().checkpoint.model_copy(update={"files": ()})}
    )
    with pytest.raises(ValueError, match="capacity"):
        operations._validate_acquisition_capacity(
            descriptor,
            free_disk_bytes=10,
            physical_memory_bytes=10,
        )


def test_models_run_must_match_exact_checkpoint_path(tmp_path: Path) -> None:
    from benchmarks.phase5_candidate.report import ModelsResponse

    response = ModelsResponse(
        backend="mlx",
        dtype="bfloat16",
        base_model_id="Qwen/Qwen3.5-4B-Base",
        run=str(tmp_path / "wrong-checkpoint"),
    )
    with pytest.raises(ValueError, match="exact checkpoint"):
        operations._validate_models_identity(
            response,
            expected_checkpoint=tmp_path / "checkpoint",
            expected_base_id="Qwen/Qwen3.5-4B-Base",
        )


def test_child_environment_creates_private_runtime_directories(tmp_path: Path) -> None:
    env = operations._child_environment(_minimal_descriptor(), tmp_path, "run-private", "secret")
    for key in ("HOME", "TMPDIR", "XDG_CACHE_HOME", "UV_CACHE_DIR", "PYTHONPYCACHEPREFIX"):
        directory = Path(env[key])
        assert directory.is_dir()
        assert directory.stat().st_mode & 0o077 == 0


def test_systemone_response_rejects_invalid_probability_shape() -> None:
    from benchmarks.phase5_candidate.report import SystemOneResponse

    with pytest.raises(ValueError):
        SystemOneResponse.model_validate(
            {
                "answers": {
                    "q0": {
                        "choice": "a",
                        "probabilities": {"a": 0.8, "b": 0.8},
                        "confidence": 0.8,
                    }
                },
                "usage": {"input_tokens": 1},
                "run": "/checkpoint",
            }
        )


def test_systemone_result_rejects_choices_outside_request(tmp_path: Path) -> None:
    payload = {
        "answers": {
            "q0": {
                "choice": "unexpected",
                "probabilities": {"a": 0.5, "b": 0.5},
                "confidence": 0.5,
            }
        },
        "usage": {"input_tokens": 1},
        "run": str(tmp_path / "checkpoint"),
    }
    with pytest.raises(ValueError, match="request criteria"):
        operations._validate_systemone_result(
            payload,
            question_options={"q0": {"a", "b"}},
            expected_run=tmp_path / "checkpoint",
        )


def test_evaluation_plan_covers_locales_workloads_and_repeat_lanes() -> None:
    plan = operations._evaluation_matrix_plan()
    assert [(locale, count) for locale, count, *_ in plan] == [
        ("pt-BR", 1),
        ("pt-BR", 10),
        ("pt-BR", 50),
        ("en", 1),
        ("en", 10),
        ("en", 50),
    ]
    assert all((new, cached, exact) == (20, 20, 3) for _, _, new, cached, exact in plan)


def test_probability_comparison_accepts_only_near_tie_argmax_flip() -> None:
    reference: dict[str, object] = {
        "q0": {"choice": "a", "probabilities": {"a": 0.51, "b": 0.49}, "confidence": 0.51}
    }
    near_tie: dict[str, object] = {
        "q0": {"choice": "b", "probabilities": {"a": 0.49, "b": 0.51}, "confidence": 0.51}
    }
    delta, stable, flips = operations._compare_choice_answers(
        reference, near_tie, near_tie_margin=0.04
    )
    assert delta == pytest.approx(0.02)
    assert stable is True
    assert len(flips) == 1
    assert flips[0][0] == "q0"
    assert flips[0][1] == pytest.approx(0.02)

    far_tie: dict[str, object] = {
        "q0": {"choice": "b", "probabilities": {"a": 0.8, "b": 0.2}, "confidence": 0.8}
    }
    _, stable, flips = operations._compare_choice_answers(reference, far_tie, near_tie_margin=0.04)
    assert stable is False
    assert flips == ()


def test_macos_measurement_parsers_return_observed_values() -> None:
    assert operations._parse_footprint_bytes("Physical footprint: 1.5G") == int(1.5 * 1024**3)
    assert operations._parse_pageouts("Pages out: 123.") == 123
    assert operations._parse_swap_used_bytes(
        "vm.swapusage: total = 2G used = 1.25M free = 1G"
    ) == int(1.25 * 1024**2)
    assert operations._parse_footprint_bytes("no measurement") is None


def test_sanitized_report_rejects_paths_hostnames_and_bearer() -> None:
    from benchmarks.phase5_candidate.report import SanitizedReport

    safe_report = {
        "schema_version": "phase5b-report.v1",
        "saracura_commit": "a" * 40,
        "acquisition_descriptor_sha256": "a" * 64,
        "threshold_digest": "b" * 64,
        "source_revision": "c" * 40,
        "model_revision": "d" * 40,
        "base_model_revision": "e" * 40,
        "fixture_digest": "f" * 64,
        "os": "Darwin",
        "architecture": "arm64",
        "chip_family": "apple_silicon",
        "physical_memory_bucket": "32",
        "backend": "mlx",
        "dtype": "bfloat16",
        "model_bytes": 1,
        "cold_load_seconds": 1.0,
        "request_p50_ms": 1.0,
        "request_p95_ms": 1.0,
        "per_decision_p50_ms": 1.0,
        "per_decision_p95_ms": 1.0,
        "decisions_per_second": 1.0,
        "peak_service_rss_gib": 1.0,
        "peak_physical_footprint_gib": 1.0,
        "swap_used_delta_gib": 0.0,
        "pageout_delta": 0,
        "memory_pressure_state": "normal",
        "invalid_output_count": 0,
        "repeat_stability_max_delta": 0.0,
        "isolation_delta_max": 0.0,
        "option_order_behavior": "recorded_descriptive_only",
        "capacity_failures": [],
        "disposition": "continue",
    }
    with pytest.raises(ValueError, match="sensitive"):
        SanitizedReport.model_validate({**safe_report, "os": "host.example.com /Users/alice"})


def test_verify_ledger_rejects_symlink(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "link").symlink_to("a.txt")
    entries = [
        {"path": "a.txt", "bytes": 1, "sha256": hashlib.sha256(b"a").hexdigest(), "role": "source"}
    ]
    with pytest.raises(RuntimeError, match="symlink"):
        operations._verify_ledger(tmp_path, entries, "source")


def test_child_environment_is_allowlisted_and_isolated(tmp_path: Path) -> None:

    import importlib

    assert importlib  # silence unused
    # Build a minimal descriptor without any optional import.
    descriptor = _minimal_descriptor()
    env = operations._child_environment(descriptor, tmp_path, "run-1", "secret-bearer")
    assert env["KEV_BACKEND"] == "mlx"
    assert env["KEV_DTYPE"] == "bf16"
    assert env["KEV_API_KEY"] == "secret-bearer"
    assert env["HF_HUB_OFFLINE"] == "1"
    assert env["TRANSFORMERS_OFFLINE"] == "1"
    assert env["HF_HUB_DISABLE_IMPLICIT_TOKEN"] == "1"
    assert env["HF_HUB_DISABLE_XET"] == "1"
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"
    assert env["HTTP_PROXY"].startswith("http://127.0.0.1:")
    assert env["HTTPS_PROXY"].startswith("http://127.0.0.1:")
    assert env["NO_PROXY"] == "127.0.0.1,localhost"
    assert env["UV_CACHE_DIR"].startswith(str(tmp_path))
    assert "PYTHONPYCACHEPREFIX" in env
    assert "HOME" in env and env["HOME"].startswith(str(tmp_path))
    assert "TMPDIR" in env and env["TMPDIR"].startswith(str(tmp_path))
    assert "XDG_CACHE_HOME" in env and env["XDG_CACHE_HOME"].startswith(str(tmp_path))


def test_child_environment_strips_inherited_kev_vars(tmp_path: Path) -> None:
    descriptor = _minimal_descriptor()
    env = operations._child_environment(descriptor, tmp_path, "run-1", "b")
    # Only the allowlisted KEV_* keys may survive.
    for key in env:
        if key.startswith("KEV_"):
            assert key in {
                "KEV_BACKEND",
                "KEV_DTYPE",
                "KEV_LORA_SCALE",
                "KEV_MERGE",
                "KEV_DATE_FACTS",
                "KEV_CUDA_GRAPHS",
                "KEV_FUSED",
                "KEV_PREFIX_CACHE",
                "KEV_PREFIX_MIN_TOKENS",
                "KEV_PREFIX_MAX_TOKENS",
                "KEV_API_KEY",
            }


def test_reject_unrecognized_kev_env() -> None:
    with pytest.raises(RuntimeError, match="unrecognized"):
        operations._reject_unrecognized_kev_env({"KEV_EVIL": "1"})


def test_reject_unrecognized_kev_env_accepts_known() -> None:
    operations._reject_unrecognized_kev_env({"KEV_BACKEND": "mlx", "HOME": "/x"})


def test_select_port_respects_descriptor_range() -> None:
    from types import SimpleNamespace

    descriptor = SimpleNamespace(runtime=SimpleNamespace(port_range=(49152, 49152)))
    port = operations._select_port(descriptor)  # type: ignore[arg-type]
    assert port == 49152


def test_descriptor_root_is_digest_scoped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    descriptor = tmp_path / "descriptor.json"
    descriptor.write_text('{"a":1}', encoding="utf-8")
    expected_digest = hashlib.sha256(descriptor.read_bytes()).hexdigest()
    # Patch the private root to a temp location to avoid real platformdirs.
    monkeypatch.setattr(operations, "_private_root", lambda: tmp_path / "root")
    root = operations._descriptor_root(descriptor)
    assert root == tmp_path / "root" / expected_digest


def test_default_environment_does_not_import_optional_modules() -> None:
    code = (
        "import sys; "
        "import benchmarks.phase5_candidate.operations; "
        "print('platformdirs' in sys.modules, 'psutil' in sys.modules, "
        "'huggingface_hub' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False False False"


def _minimal_descriptor() -> AcquisitionDescriptor:
    from benchmarks.phase5_candidate.descriptor import (
        AcquisitionDescriptor,
        LicenseBlock,
        LimitsBlock,
        RuntimeBlock,
        SnapshotBlock,
        SourceBlock,
        ThresholdsBlock,
    )

    return AcquisitionDescriptor(
        schema_version="phase5-kev4b-acquisition.v1",
        reviewed_at="2026-09-27",
        candidate_id="kev-4b",
        licenses=LicenseBlock(
            source="Apache-2.0",
            adapter="Apache-2.0",
            base="Apache-2.0",
            source_url="https://github.com/jaredpalmer/kev/blob/main/LICENSE",
            adapter_url="https://huggingface.co/jaredpalmer/kev-4b/blob/1/LICENSE",
            base_url="https://huggingface.co/Qwen/Qwen3.5-4B-Base/blob/1/LICENSE",
        ),
        source=SourceBlock(
            repository="https://github.com/jaredpalmer/kev",
            revision="9c41005b2180347c3c646dfc9e50c4428483ec6b",
            python_constraint=">=3.12,<3.14",
            lock_sha256="a9922dbb89acdef78299fd2b4a8c3f7f0fa1b2bc08b55595b6926fa785a9c466",
            files=(),
        ),
        checkpoint=SnapshotBlock(
            model_id="jaredpalmer/kev-4b",
            revision="139fdd94f1b6a6ad80cc15e08fcb99cac885a101",
            files=(),
        ),
        base_model=SnapshotBlock(
            model_id="Qwen/Qwen3.5-4B-Base",
            revision="1001bb4d826a52d1f399e183466143f4da7b741b",
            files=(),
        ),
        runtime=RuntimeBlock(
            python="3.12",
            loopback_host="127.0.0.1",
            port_range=(49152, 65535),
            backend="mlx",
            dtype="bf16",
            offline_environment_flags=("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"),
            lock_digest="a9922dbb89acdef78299fd2b4a8c3f7f0fa1b2bc08b55595b6926fa785a9c466",
        ),
        limits=LimitsBlock(
            max_total_bytes=13_000_000_000,
            max_source_file_count=4096,
            max_source_bytes=500_000_000,
            min_free_disk_bytes=20_000_000_000,
            min_physical_memory_bytes=16 * 1024**3,
            startup_timeout_seconds=600,
            request_timeout_seconds=60,
        ),
        thresholds=ThresholdsBlock(
            max_latency_seconds=60.0,
            p95_latency_seconds_q1=5.0,
            p95_latency_seconds_q10=15.0,
            p95_latency_seconds_q50=45.0,
            max_repeat_probability_delta=1e-6,
            max_together_separate_probability_delta=0.04,
            near_tie_margin=0.04,
            max_peak_rss_gib=22.0,
            max_peak_physical_footprint_gib=22.0,
            preferred_swap_delta_gib=2.0,
            degraded_swap_delta_gib=8.0,
            max_swap_delta_gib=8.0,
            cold_readiness_timeout_seconds=600,
        ),
        allowed_readiness_evidence=(
            "pinned_license_reviewed_acquisition",
            "local_systems_report",
        ),
    )
