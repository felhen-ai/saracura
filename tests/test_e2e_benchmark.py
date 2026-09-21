from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from email.message import Message
from io import BytesIO
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import JsonValue

import benchmarks.e2e_benchmark as bench
from saracura.serialization import canonical_json_bytes


def test_protocol_sizes_orders_and_nearest_rank() -> None:
    assert bench.LOCAL_BATCH_SIZES == (1, 8, 20, 32, 64, 128)
    assert bench.LOCAL_ORDER == (1, 128, 8, 64, 20, 32)
    assert bench.JEV_BATCH_SIZES == (1, 8, 20)
    assert bench.JEV_ORDER == (1, 20, 8)
    assert bench.nearest_rank([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.95) == 10


def test_throughput_is_aggregate_not_reciprocal() -> None:
    assert bench.throughput(8, [100.0, 200.0]) == pytest.approx(53.3333333333)


def test_mps_event_order_and_timing_arithmetic() -> None:
    events: list[str] = []

    class Clock:
        value = 0

        def perf_counter_ns(self) -> int:
            self.value += 10
            events.append("clock")
            return self.value

    class Components:
        def synchronize(self) -> None:
            events.append("sync")

        def tokenize(self, texts: Sequence[str]) -> object:
            events.append("tokenize")
            return texts

        def infer_and_pool(self, encoded: object) -> object:
            events.append("infer")
            return encoded

        def to_cpu(self, pooled: object) -> object:
            events.append("cpu")
            return pooled

        def predict(self, pooled: object) -> list[int]:
            events.append("head")
            return [0]

        def token_lengths(self, texts: Sequence[str]) -> list[int]:
            return [1 for _ in texts]

    sample = bench.measure_local_sample(Components(), ["texto"], Clock())
    assert events == [
        "sync",
        "clock",
        "tokenize",
        "clock",
        "infer",
        "sync",
        "clock",
        "cpu",
        "sync",
        "clock",
        "head",
        "clock",
    ]
    assert sample["components_ns"] == {
        "tokenization_ns": 10,
        "cpu_to_mps_encoder_pooling_ns": 10,
        "mps_to_cpu_ns": 10,
        "cpu_head_ns": 10,
        "total_ns": 40,
    }


def _rows(count: int) -> list[dict[str, str]]:
    return [
        {"family_id": f"family-{i:03d}", "text": f"mensagem {i}", "candidate_label": "billing"}
        for i in range(count)
    ]


def test_jev_flattening_and_provider_controls_are_exact() -> None:
    payload = bench.build_jev_request(_rows(8))
    assert payload["state"] == {
        "records": [
            {"id": "r000", "record": {"text": "mensagem 0"}},
            {"id": "r001", "record": {"text": "mensagem 1"}},
            *[{"id": f"r{i:03d}", "record": {"text": f"mensagem {i}"}} for i in range(2, 8)],
        ]
    }
    assert set(payload["questions"]) == {f"r{i:03d}__routing" for i in range(8)}
    assert payload["provider"] == bench.PROVIDER_CONTROLS
    with pytest.raises(ValueError):
        bench.build_jev_request(_rows(3))


def test_jev_response_validation_rejects_missing_extra_and_bad_usage() -> None:
    response = {
        "id": "response",
        "model": "typesafe/jev-1.13-20260921",
        "provider": "TypeSafe",
        "answers": {"r000__routing": {"type": "choice", "choice": "billing"}},
        "usage": {"cost": 0.001, "input_tokens": 2, "output_tokens": 1},
    }
    reduced = bench.validate_jev_response(response, ["r000__routing"])
    assert reduced["usage"]["cost"] == "0.001"
    for bad in (
        {**response, "answers": {}},
        {**response, "usage": {"cost": None, "input_tokens": 2, "output_tokens": 1}},
    ):
        with pytest.raises(ValueError):
            bench.validate_jev_response(bad, ["r000__routing"])


def test_prediction_digest_and_privacy_free_inputs() -> None:
    assert bench.prediction_digest([0, 4]) == bench.prediction_digest([0, 4])
    assert bench.opaque_input_digest(_rows(2)) != bench.opaque_input_digest(_rows(1))
    with pytest.raises(ValueError):
        bench.prediction_digest([5])


def test_decimal_journal_reservation_settlement_and_restart_uncertainty(tmp_path: Path) -> None:
    path = tmp_path / "cost-journal.jsonl"
    journal = bench.CostJournal(path, predecessor_sha256="a" * 64)
    journal.reserve("r1", Decimal("0.002"))
    journal.settle("r1", Decimal("0.001"))
    assert journal.settled == Decimal("0.001")
    journal.reserve("r2", Decimal("0.002"))
    restarted = bench.CostJournal(path, predecessor_sha256="a" * 64)
    assert restarted.uncertain is True
    with pytest.raises(ValueError):
        restarted.reserve("r3", Decimal("0.001"))


def test_decimal_journal_carries_predecessor_stage_and_total_budget(tmp_path: Path) -> None:
    journal = bench.CostJournal(
        tmp_path / "journal",
        predecessor_sha256="a" * 64,
        initial_stage_spent=Decimal("0.0399"),
        initial_total_spent=Decimal("0.2499"),
    )
    with pytest.raises(ValueError):
        journal.reserve("r1", Decimal("0.0002"))


def test_journal_billed_failure_and_cost_above_reservation_are_fail_closed(tmp_path: Path) -> None:
    journal = bench.CostJournal(tmp_path / "journal", predecessor_sha256="b" * 64)
    journal.reserve("r1", Decimal("0.002"))
    with pytest.raises(ValueError):
        journal.settle("r1", Decimal("0.003"))
    journal = bench.CostJournal(tmp_path / "journal-2", predecessor_sha256="b" * 64)
    journal.reserve("r1", Decimal("0.002"))
    journal.settle("r1", Decimal("0.002"), status="billed_failure")
    assert (
        json.loads((tmp_path / "journal-2").read_text().splitlines()[1])["status"]
        == "billed_failure"
    )


def test_claim_is_immutable_and_reused_predecessor_fails(tmp_path: Path) -> None:
    path = tmp_path / "claims" / ("a" * 64 + ".json")
    assert len(bench.create_claim(path, "a" * 64)) == 64
    with pytest.raises(ValueError):
        bench.create_claim(path, "a" * 64)
    assert path.stat().st_mode & 0o777 == 0o600


def test_claim_creation_is_concurrency_safe(tmp_path: Path) -> None:
    path = tmp_path / "claims" / ("f" * 64 + ".json")

    def attempt() -> bool:
        try:
            bench.create_claim(path, "f" * 64)
            return True
        except ValueError:
            return False

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(lambda _: attempt(), range(2))) == [False, True]


def test_dispatch_is_single_attempt_and_uses_sixty_second_timeout(tmp_path: Path) -> None:
    calls: list[tuple[str, float]] = []

    class Clock:
        value = 0

        def perf_counter_ns(self) -> int:
            self.value += 1_000_000
            return self.value

    def transport(
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes,
        *,
        timeout: float,
    ) -> tuple[int, Mapping[str, str], bytes]:
        calls.append((method, timeout))
        return (
            200,
            {},
            json.dumps(
                {
                    "id": "response",
                    "model": "typesafe/jev-1.13-20260921",
                    "provider": "TypeSafe",
                    "answers": {},
                    "usage": {"cost": 0, "input_tokens": 1, "output_tokens": 0},
                }
            ).encode(),
        )

    journal = bench.CostJournal(tmp_path / "journal", predecessor_sha256="d" * 64)
    response, elapsed = bench.dispatch_jev(
        {
            "model": bench.JEV_MODEL,
            "state": {},
            "questions": {},
            "provider": bench.PROVIDER_CONTROLS,
        },
        api_key="secret",
        journal=journal,
        transport=transport,
        clock=Clock(),
    )
    assert response["provider"] == "TypeSafe" and elapsed == 1.0
    assert calls == [("POST", 60.0)]
    assert journal.settled == Decimal("0")


def test_dispatch_malformed_response_marks_journal_uncertain(tmp_path: Path) -> None:
    journal = bench.CostJournal(tmp_path / "journal", predecessor_sha256="e" * 64)

    def transport(
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes,
        *,
        timeout: float,
    ) -> tuple[int, Mapping[str, str], bytes]:
        return 200, {}, b"not-json"

    with pytest.raises(json.JSONDecodeError):
        bench.dispatch_jev(
            {
                "model": bench.JEV_MODEL,
                "state": {},
                "questions": {},
                "provider": bench.PROVIDER_CONTROLS,
            },
            api_key="secret",
            journal=journal,
            transport=transport,
        )
    assert journal.uncertain is True


def test_dispatch_records_billed_http_failure_and_stops(tmp_path: Path) -> None:
    journal = bench.CostJournal(tmp_path / "journal", predecessor_sha256="e" * 64)
    calls = 0

    def transport(
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes,
        *,
        timeout: float,
    ) -> tuple[int, Mapping[str, str], bytes]:
        nonlocal calls
        calls += 1
        return 500, {}, json.dumps({"usage": {"cost": 0.000001}}).encode()

    with pytest.raises(bench.RemoteRunError):
        bench.run_jev_workloads(
            _rows(20),
            api_key="secret",
            journal=journal,
            transport=transport,
            iterations=1,
        )
    rows = [json.loads(line) for line in (tmp_path / "journal").read_text().splitlines()]
    assert calls == 1
    assert rows[-1]["status"] == "billed_failure"
    assert journal.uncertain is False


def test_urllib_transport_returns_http_error_body_and_disables_proxy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handlers: tuple[object, ...] = ()

    class Opener:
        def open(self, request: object, timeout: float) -> object:
            raise urllib.error.HTTPError(
                "https://openrouter.ai/api/alpha/decisions",
                429,
                "limited",
                Message(),
                BytesIO(b'{"error":{"code":429}}'),
            )

    def build_opener(*values: object) -> Opener:
        nonlocal handlers
        handlers = values
        return Opener()

    monkeypatch.setattr(urllib.request, "build_opener", build_opener)
    status, _, body = bench._urllib_transport(
        "POST",
        "https://openrouter.ai/api/alpha/decisions",
        {},
        b"{}",
        timeout=60,
    )
    assert status == 429 and b"429" in body
    proxy = next(value for value in handlers if isinstance(value, urllib.request.ProxyHandler))
    assert cast(Any, proxy).proxies == {}
    assert any(type(value).__name__ == "NoRedirect" for value in handlers)


def test_local_precheck_runs_before_any_timed_work() -> None:
    class NeverClock:
        calls = 0

        def perf_counter_ns(self) -> int:
            self.calls += 1
            return self.calls

    class Components:
        def synchronize(self) -> None:  # pragma: no cover - must not run
            raise AssertionError

        def tokenize(self, texts: Sequence[str]) -> object:
            raise AssertionError

        def infer_and_pool(self, encoded: object) -> object:
            raise AssertionError

        def to_cpu(self, pooled: object) -> object:
            raise AssertionError

        def predict(self, pooled: object) -> list[int]:
            raise AssertionError

        def token_lengths(self, texts: Sequence[str]) -> list[int]:
            return [129, *([1] * (len(texts) - 1))]

    clock = NeverClock()
    with pytest.raises(ValueError, match="untruncated"):
        bench.run_local_workloads(_rows(128), Components(), clock=clock, warmups=0, iterations=1)
    assert clock.calls == 0


def test_jev_workload_order_is_fixed(tmp_path: Path) -> None:
    calls: list[int] = []

    class Clock:
        value = 0

        def perf_counter_ns(self) -> int:
            self.value += 1_000_000
            return self.value

    def transport(
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes,
        *,
        timeout: float,
    ) -> tuple[int, Mapping[str, str], bytes]:
        request = json.loads(body)
        calls.append(len(request["state"]["records"]))
        answers = {key: {"type": "choice", "choice": "billing"} for key in request["questions"]}
        return (
            200,
            {},
            json.dumps(
                {
                    "id": f"r{len(calls)}",
                    "model": "typesafe/jev-1.13-20260921",
                    "provider": "TypeSafe",
                    "answers": answers,
                    "usage": {"cost": 0, "input_tokens": 1, "output_tokens": 0},
                }
            ).encode(),
        )

    result = bench.run_jev_workloads(
        _rows(20),
        api_key="secret",
        journal=bench.CostJournal(tmp_path / "journal", predecessor_sha256="a" * 64),
        transport=transport,
        clock=Clock(),
        iterations=1,
    )
    assert calls == [1, 20, 8]
    assert list(result["workloads"]) == ["1", "20", "8"]
    assert result["completed_call_count"] == 3


def test_artifact_manifest_allowlist_and_report_sanitization(tmp_path: Path) -> None:
    result = b'{"schema_version":"phase3c.v1"}\n'
    report = bench.sanitized_report(
        {"local": {"samples": [1.0]}, "limitations": "synthetic only"}
    ).encode()
    journal = b""
    output = tmp_path / "out"
    output.mkdir(mode=0o700)
    manifest = bench.write_artifact_manifest(
        output,
        result,
        report,
        journal,
        packet_sha256="a" * 64,
        training_sha256="b" * 64,
        predecessor_sha256="c" * 64,
        claim_sha256=None,
    )
    value = json.loads(manifest)
    assert set(value["files"]) == {"result.json", "report.md", "cost-journal.jsonl"}
    assert value["sources"]["benchmarks/e2e_benchmark.py"] == bench.digest_bytes(
        Path("benchmarks/e2e_benchmark.py").read_bytes()
    )
    with pytest.raises(ValueError):
        bench.sanitized_report({"local": "Bearer secret"})


def _closed_local_result() -> dict[str, object]:
    workloads: dict[str, object] = {}
    for batch_size in bench.LOCAL_BATCH_SIZES:
        samples = []
        for index in range(bench.LOCAL_ITERATIONS):
            start = index * 10
            boundaries = {
                "start": start,
                "tokenized": start + 1,
                "device": start + 2,
                "transferred": start + 3,
                "stop": start + 4,
            }
            samples.append(
                {
                    "boundaries_ns": boundaries,
                    "components_ns": bench.component_timings_ns(*boundaries.values()),
                    "prediction_digest": "d" * 64,
                    "accuracy": 1.0,
                    "total_ms": 0.000004,
                }
            )
        totals = [0.000004] * bench.LOCAL_ITERATIONS
        workloads[str(batch_size)] = {
            "batch_size": batch_size,
            "warmups": bench.LOCAL_WARMUPS,
            "iterations": bench.LOCAL_ITERATIONS,
            "input_id_digest": "a" * 64,
            "prediction_digest": "d" * 64,
            "token_lengths": [1] * batch_size,
            "token_lengths_digest": bench.digest_json([1] * batch_size),
            "truncated": False,
            "accuracy": 1.0,
            "samples": samples,
            "samples_sha256": bench.digest_json(samples),
            "summary": bench.summarize(totals),
            "per_item_summary": bench.summarize([value / batch_size for value in totals]),
            "throughput_items_per_second": bench.throughput(batch_size, totals),
        }
    return {
        "schema_version": "phase3c.v1",
        "sealed": True,
        "protocol": {
            "local_batch_sizes": list(bench.LOCAL_BATCH_SIZES),
            "local_order": list(bench.LOCAL_ORDER),
            "local_warmups": bench.LOCAL_WARMUPS,
            "local_iterations": bench.LOCAL_ITERATIONS,
            "max_tokens": bench.MAX_TOKENS,
            "device": "mps",
            "jev_batch_sizes": list(bench.JEV_BATCH_SIZES),
            "jev_order": list(bench.JEV_ORDER),
            "jev_iterations": bench.JEV_ITERATIONS,
        },
        "provenance": {
            "packet_manifest_sha256": "a" * 64,
            "training_manifest_sha256": "b" * 64,
            "checkpoint_sha256": "c" * 64,
            "prior_cost_ledger_sha256": "d" * 64,
            "claim_sha256": None,
            "benchmark_source_sha256": bench.digest_bytes(Path(bench.__file__).read_bytes()),
            "encoder_revision": "e" * 40,
        },
        "environment": {
            "python": "3.14",
            "platform": "test",
            "processor": "arm64",
            "torch": "test",
            "transformers": "test",
            "safetensors": "test",
        },
        "local": {
            "fresh_process_load_ms": 1.0,
            "page_cache_sensitive": True,
            "load_includes_import_and_snapshot_verification": True,
            "rss_metric": "ru_maxrss_bytes",
            "rss_before_load_bytes": 1,
            "rss_after_load_bytes": 2,
            "rss_after_workloads_bytes": 3,
            "workloads": workloads,
        },
        "jev": {"status": "not_requested", "completed_call_count": 0},
        "comparison": None,
        "budget": {
            "approved_total_usd": "0.25",
            "jev_stage_limit_usd": "0.04",
            "prior_total_usd": "0.01",
            "phase3c_jev_usd": "0",
        },
        "limitations": list(bench.MANDATORY_LIMITATIONS),
    }


def test_closed_result_and_artifact_validator_reject_extras(tmp_path: Path) -> None:
    result = _closed_local_result()
    with pytest.raises(ValueError):
        bench.Phase3CResultModel.model_validate({**result, "extra": True})
    output = tmp_path / "out"
    bench.write_phase3c_artifacts(
        output,
        result,
        b"",
        packet_sha256="a" * 64,
        training_sha256="b" * 64,
        predecessor_sha256="d" * 64,
        claim_sha256=None,
    )
    assert bench.validate_phase3c_artifacts(output).schema_version == "phase3c.v1"
    (output / "extra.txt").touch(mode=0o600)
    with pytest.raises(ValueError, match="allowlist"):
        bench.validate_phase3c_artifacts(output)


def _reseal_result(output: Path, result: dict[str, object]) -> None:
    result_bytes = canonical_json_bytes(cast(JsonValue, result)) + b"\n"
    result_path = output / "result.json"
    result_path.write_bytes(result_bytes)
    result_path.chmod(0o600)
    manifest_path = output / "artifact-manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["files"]["result.json"] = {
        "bytes": len(result_bytes),
        "sha256": bench.digest_bytes(result_bytes),
    }
    unsigned = dict(manifest)
    unsigned.pop("self_sha256")
    manifest["self_sha256"] = bench.digest_json(unsigned)
    manifest_path.write_bytes(canonical_json_bytes(manifest) + b"\n")
    manifest_path.chmod(0o600)


def test_artifact_validator_rejects_resealed_timing_contradiction(tmp_path: Path) -> None:
    output = tmp_path / "out"
    bench.write_phase3c_artifacts(
        output,
        _closed_local_result(),
        b"",
        packet_sha256="a" * 64,
        training_sha256="b" * 64,
        predecessor_sha256="d" * 64,
        claim_sha256=None,
    )
    result_path = output / "result.json"
    result = json.loads(result_path.read_bytes())
    workload = result["local"]["workloads"]["1"]
    workload["samples"][0]["components_ns"]["cpu_head_ns"] = 2
    workload["samples_sha256"] = bench.digest_json(workload["samples"])
    _reseal_result(output, result)
    with pytest.raises(ValueError, match="timing"):
        bench.validate_phase3c_artifacts(output)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("source", "provenance"),
        ("limitations", "limitations"),
        ("protocol", "protocol"),
    ],
)
def test_artifact_validator_rejects_resealed_contract_drift(
    tmp_path: Path, case: str, message: str
) -> None:
    output = tmp_path / case
    bench.write_phase3c_artifacts(
        output,
        _closed_local_result(),
        b"",
        packet_sha256="a" * 64,
        training_sha256="b" * 64,
        predecessor_sha256="d" * 64,
        claim_sha256=None,
    )
    result = json.loads((output / "result.json").read_bytes())
    if case == "source":
        result["provenance"]["benchmark_source_sha256"] = "f" * 64
    elif case == "limitations":
        result["limitations"].pop()
    else:
        result["protocol"]["local_order"] = list(bench.LOCAL_BATCH_SIZES)
    _reseal_result(output, result)
    with pytest.raises(ValueError, match=message):
        bench.validate_phase3c_artifacts(output)


def test_predecessor_binding_checks_evaluation_ledger_tail(tmp_path: Path) -> None:
    ledger_path = tmp_path / "jev-cost-ledger.json"
    ledger_path.write_text(
        json.dumps(
            {
                "schema_version": "cost-ledger.v1",
                "final": True,
                "entries": [
                    {
                        "stage": "jev",
                        "cost": 0.001,
                        "request_id": "r1",
                        "status": "complete",
                    }
                ],
                "total": 0.001,
            }
        )
    )
    evaluation = {
        "schema_version": "synthetic-evaluation.v1",
        "sealed": True,
        "packet_manifest_sha256": "a" * 64,
        "training_manifest_sha256": "b" * 64,
        "local": {},
        "jev": {"status": "complete", "digests": ["c" * 64], "usage": {"cost": 0.001}},
        "jev_usage_snapshots": [],
    }
    (tmp_path / "evaluation-manifest.json").write_text(json.dumps(evaluation))
    bench.validate_predecessor_binding(
        ledger_path, packet_sha256="a" * 64, training_sha256="b" * 64
    )
    evaluation["jev"]["usage"]["cost"] = 0.002  # type: ignore[index]
    (tmp_path / "evaluation-manifest.json").write_text(json.dumps(evaluation))
    with pytest.raises(ValueError, match="ledger tail"):
        bench.validate_predecessor_binding(
            ledger_path, packet_sha256="a" * 64, training_sha256="b" * 64
        )


def test_import_does_not_load_optional_ml_modules() -> None:
    import sys

    assert "torch" not in sys.modules or sys.modules["torch"].__name__ != "torch"
    assert "transformers" not in sys.modules
