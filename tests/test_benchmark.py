from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

import benchmarks.io as benchmark_io
import benchmarks.run as benchmark_run
from benchmarks.cases import calibrations_for, request_for
from benchmarks.contracts import BenchmarkResult, summary_from_values
from benchmarks.io import atomic_create, render_report, write_result
from benchmarks.run import run_benchmark
from saracura.backends import DeterministicFixtureBackend
from saracura.runtime import DecisionEngine, default_workflows
from saracura.serialization import canonical_json_bytes


def test_nearest_rank_summary_is_frozen() -> None:
    summary = summary_from_values([1.0, 2.0, 3.0, 4.0, 5.0])
    assert summary.count == 5
    assert summary.p50_ms == 3.0
    assert summary.p95_ms == 5.0


def test_runner_has_fixed_workloads_and_encode_once() -> None:
    result = run_benchmark(
        warmups=0,
        iterations=2,
        code_revision="test-revision",
        hardware_label="macbook-m4-pro-24gb",
    )
    assert [item.workload.question_count for item in result.workloads] == [1, 10, 50]
    assert len({item.workload.state_sha256 for item in result.workloads}) == 1
    assert all(
        sample.state_encoding_count == 1 for item in result.workloads for sample in item.samples
    )
    assert all(
        sample.answer_count == item.workload.question_count
        for item in result.workloads
        for sample in item.samples
    )


def test_same_semantic_run_keeps_answer_digests() -> None:
    first = run_benchmark(warmups=1, iterations=2, code_revision="same-config", hardware_label=None)
    second = run_benchmark(
        warmups=1, iterations=2, code_revision="same-config", hardware_label=None
    )
    first_digests = [[sample.answer_sha256 for sample in item.samples] for item in first.workloads]
    second_digests = [
        [sample.answer_sha256 for sample in item.samples] for item in second.workloads
    ]
    assert first_digests == second_digests
    assert first.run.run_id != second.run.run_id


def test_result_and_report_are_privacy_allowlisted(tmp_path: Path) -> None:
    result = run_benchmark(
        warmups=0,
        iterations=1,
        code_revision="privacy-test",
        hardware_label="macbook",
    )
    raw = json.dumps(result.model_dump(mode="json"))
    report = render_report(result)
    for text in (raw, report):
        assert "/Users/" not in text
        assert "/Volumes/" not in text
        assert "hostname" not in text.lower()
        assert "username" not in text.lower()
        assert "serial" not in text.lower()
        assert "127.0.0.1" not in text
        assert "OP_SERVICE" not in text


def test_atomic_artifact_never_overwrites(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"
    atomic_create(path, b"first")
    with pytest.raises(FileExistsError):
        atomic_create(path, b"second")
    assert path.read_bytes() == b"first"
    assert not list(tmp_path.glob(".*artifact.json.*"))


def test_raw_result_round_trips_closed_schema() -> None:
    result = run_benchmark(
        warmups=0,
        iterations=1,
        code_revision="round-trip",
        hardware_label=None,
    )
    parsed = BenchmarkResult.model_validate_json(result.model_dump_json())
    assert parsed == result


def test_contract_rejects_unknown_nan_and_private_values() -> None:
    result = run_benchmark(
        warmups=0,
        iterations=1,
        code_revision="contract-test",
        hardware_label=None,
    )
    payload = result.model_dump(mode="json")
    payload["run"]["environment"]["os_release"] = "/Users/private"
    with pytest.raises(ValidationError):
        BenchmarkResult.model_validate(payload)

    payload = result.model_dump(mode="json")
    payload["run"]["environment"]["os_release"] = "release with spaces"
    with pytest.raises(ValidationError):
        BenchmarkResult.model_validate(payload)

    payload = result.model_dump(mode="json")
    payload["limitations"]["interpretation"] = "token=private"
    with pytest.raises(ValidationError):
        BenchmarkResult.model_validate(payload)

    payload = result.model_dump(mode="json")
    payload["workloads"][0]["samples"][0]["timing"]["total_ms"] = float("nan")
    with pytest.raises(ValidationError):
        BenchmarkResult.model_validate(payload)

    payload = result.model_dump(mode="json")
    payload["run"]["unknown"] = True
    with pytest.raises(ValidationError):
        BenchmarkResult.model_validate(payload)

    payload = result.model_dump(mode="json")
    payload["workloads"][0]["samples"][0]["timing"]["decision_ms"] = -1
    with pytest.raises(ValidationError):
        BenchmarkResult.model_validate(payload)

    payload = result.model_dump(mode="json")
    payload["workloads"][0]["workload"]["warmup_iterations"] = "0"
    with pytest.raises(ValidationError):
        BenchmarkResult.model_validate(payload)

    payload = result.model_dump(mode="json")
    payload["workloads"][0]["workload"]["question_count"] = True
    with pytest.raises(ValidationError):
        BenchmarkResult.model_validate(payload)

    payload = result.model_dump(mode="json")
    payload["workloads"][0]["samples"][0]["state_encoding_count"] = "1"
    with pytest.raises(ValidationError):
        BenchmarkResult.model_validate(payload)

    payload = result.model_dump(mode="json")
    payload["workloads"][0]["samples"][0]["state_encoding_count"] = True
    with pytest.raises(ValidationError):
        BenchmarkResult.model_validate(payload)

    payload = result.model_dump(mode="json")
    payload["workloads"][0]["samples"][0]["timing"]["total_ms"] = "1.25"
    with pytest.raises(ValidationError):
        BenchmarkResult.model_validate(payload)


def test_contract_rejects_cross_workload_contradictions() -> None:
    result = run_benchmark(
        warmups=0,
        iterations=1,
        code_revision="cross-workload",
        hardware_label=None,
    )
    payload = result.model_dump(mode="json")
    payload["workloads"][1]["workload"]["state_sha256"] = "0" * 64
    with pytest.raises(ValidationError):
        BenchmarkResult.model_validate(payload)

    payload = result.model_dump(mode="json")
    payload["workloads"][2]["workload"]["measured_iterations"] = 2
    with pytest.raises(ValidationError):
        BenchmarkResult.model_validate(payload)

    payload = result.model_dump(mode="json")
    payload["workloads"][0]["workload"]["fixture_manifest"]["sha256"] = "0" * 64
    with pytest.raises(ValidationError):
        BenchmarkResult.model_validate(payload)

    payload = result.model_dump(mode="json")
    payload["run"]["model"]["checkpoint_sha256"] = "0" * 64
    with pytest.raises(ValidationError):
        BenchmarkResult.model_validate(payload)

    result = run_benchmark(
        warmups=0,
        iterations=2,
        code_revision="digest-contradiction",
        hardware_label=None,
    )
    payload = result.model_dump(mode="json")
    payload["workloads"][0]["samples"][1]["answer_sha256"] = "0" * 64
    with pytest.raises(ValidationError):
        BenchmarkResult.model_validate(payload)


def test_invalid_bounds_fail_before_runtime_work(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_manifest_read() -> None:
        raise AssertionError("manifest/runtime work must not execute for invalid bounds")

    monkeypatch.setattr(benchmark_run, "_manifest_identity", unexpected_manifest_read)
    with pytest.raises(ValueError, match="benchmark configuration is invalid"):
        run_benchmark(
            warmups=101,
            iterations=1,
            code_revision="invalid-bounds",
            hardware_label=None,
        )


def test_common_question_answers_are_byte_identical() -> None:
    answers: dict[str, bytes] = {}
    for count in (1, 10, 50):
        request = request_for(count)
        backend = DeterministicFixtureBackend()
        response = DecisionEngine(
            backend=backend,
            workflows=default_workflows(),
            calibrations=calibrations_for(request, backend),
        ).decide(request)
        current = {
            answer.question_id: canonical_json_bytes(answer.model_dump(mode="json"))
            for answer in response.answers
        }
        for question_id, previous in answers.items():
            assert current[question_id] == previous
        answers.update(current)


def test_manifest_drift_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(benchmark_run, "FIXTURE_SPLIT_MANIFEST_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="manifest validation failed"):
        benchmark_run._manifest_identity()


def test_cli_maps_failures_without_echoing_private_input(
    capsys: pytest.CaptureFixture[str],
) -> None:
    sentinel = "/Users/private/sentinel-token"
    assert (
        benchmark_run.main(
            [
                "--suite",
                "decision-scaling",
                "--backend",
                "fixture",
                "--warmups",
                "-1",
                "--iterations",
                "1",
                "--code-revision",
                sentinel,
                "--output-dir",
                sentinel,
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert captured.err == "benchmark failed: invalid configuration or manifest\n"
    assert sentinel not in captured.out + captured.err

    assert (
        benchmark_run.main(
            [
                "--suite",
                "decision-scaling",
                "--backend",
                "fixture",
                "--warmups",
                sentinel,
                "--iterations",
                "1",
                "--code-revision",
                "safe",
                "--output-dir",
                ".artifacts",
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert captured.err == "benchmark failed: invalid configuration or manifest\n"
    assert sentinel not in captured.out + captured.err


def test_report_contains_required_provenance_and_p95_fields() -> None:
    result = run_benchmark(
        warmups=0,
        iterations=1,
        code_revision="report-test",
        hardware_label="macbook",
    )
    report = render_report(result)
    for marker in (
        "Saracura version",
        "Model:",
        "checkpoint",
        "Hardware label",
        "Manifest:",
        "State sha256",
        "Encoding p95",
        "Decision p95",
        result.limitations.interpretation,
    ):
        assert marker in report


def test_write_result_removes_raw_after_report_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = run_benchmark(
        warmups=0,
        iterations=1,
        code_revision="io-failure",
        hardware_label=None,
    )
    original = benchmark_io.atomic_create
    calls = 0

    def fail_on_report(path: Path, payload: bytes) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated report failure")
        original(path, payload)

    monkeypatch.setattr(benchmark_io, "atomic_create", fail_on_report)
    with pytest.raises(OSError):
        write_result(result, tmp_path)
    assert not list(tmp_path.glob("raw-*.json"))
    assert not list(tmp_path.glob(".*"))
