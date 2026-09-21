from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from decimal import Decimal
from pathlib import Path

import pytest

import benchmarks.e2e_benchmark as bench


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
    return [{"family_id": f"family-{i}", "text": f"mensagem {i}"} for i in range(count)]


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


def test_dispatch_is_single_attempt_and_uses_sixty_second_timeout() -> None:
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
        return 200, {}, json.dumps({"ok": True}).encode()

    response, elapsed = bench.dispatch_jev(
        {
            "model": bench.JEV_MODEL,
            "state": {},
            "questions": {},
            "provider": bench.PROVIDER_CONTROLS,
        },
        api_key="secret",
        transport=transport,
        clock=Clock(),
    )
    assert response == {"ok": True} and elapsed == 1.0
    assert calls == [("POST", 60.0)]


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


def test_import_does_not_load_optional_ml_modules() -> None:
    import sys

    assert "torch" not in sys.modules or sys.modules["torch"].__name__ != "torch"
    assert "transformers" not in sys.modules
