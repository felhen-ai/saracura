from __future__ import annotations

import json
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from benchmarks import typesafe_native as native
from benchmarks.e2e_benchmark import JEV_BATCH_SIZES, digest_bytes
from benchmarks.synthetic_research import LABELS


def rows(count: int = 20) -> list[dict[str, Any]]:
    return [
        {"family_id": f"f{i:03d}", "text": f"texto {i}", "candidate_label": LABELS[i % 5]}
        for i in range(count)
    ]


def response(request: dict[str, Any]) -> dict[str, Any]:
    answers = {}
    for key in request["questions"]:
        answers[key] = {
            "type": "choice",
            "choice": "billing",
            "probabilities": {label: (1.0 if label == "billing" else 0.0) for label in LABELS},
            "confidence": 1.0,
        }
    return {
        "model": native.MODEL_ID,
        "answers": answers,
        "usage": {"input_tokens": 10, "output_tokens": 2},
    }


def test_request_is_native_and_pinned() -> None:
    request = native.native_request(rows(1))
    assert set(request) == {"model", "state", "questions"}
    assert request["model"] == native.MODEL_ID


def test_response_is_typed_and_reduced() -> None:
    request = native.native_request(rows(1))
    value = native.validate_native_response(response(request), list(request["questions"]))
    sample = native.reduce_sample(value, 12.5, ["billing"])
    assert sample["computed_cost_usd"] == "0.00000042"


@pytest.mark.parametrize(
    "change", [lambda x: {**x, "model": "jev-latest"}, lambda x: {**x, "answers": {"extra": {}}}]
)
def test_response_rejects_drift(change: Any) -> None:
    request = native.native_request(rows(1))
    with pytest.raises(ValueError):
        native.validate_native_response(change(response(request)), list(request["questions"]))


def test_journal_open_reservation_fails_closed(tmp_path: Path) -> None:
    journal = native.Journal(tmp_path / "journal", "a" * 64)
    journal.reserve("native-b001-i00", 1, 0)
    assert native.Journal(tmp_path / "journal", "a" * 64).uncertain
    with pytest.raises(ValueError):
        native.Journal(tmp_path / "journal", "a" * 64).reserve("native-b001-i01", 1, 1)


def test_dispatch_is_single_attempt_and_literal_endpoint(tmp_path: Path) -> None:
    calls = []
    request = native.native_request(rows(1))

    def transport(
        method: str, url: str, headers: Mapping[str, str], body: bytes, *, timeout: float
    ) -> tuple[int, Mapping[str, str], bytes]:
        calls.append((method, url, timeout))
        return 200, {}, json.dumps(response(request)).encode()

    native.dispatch_native(
        request,
        api_key="secret",
        journal=native.Journal(tmp_path / "journal", "b" * 64),
        batch_size=1,
        iteration=0,
        transport=transport,
    )
    assert calls == [("POST", native.ENDPOINT, 60.0)]


def test_default_http_transport_disables_proxies_redirects_and_reuses_no_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builders: list[tuple[object, ...]] = []
    opened: list[tuple[str, str, float]] = []

    class Response:
        status = 200
        headers: Mapping[str, str] = {}

        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def read(self) -> bytes:
            return b"{}"

    class Opener:
        def open(self, request: urllib.request.Request, *, timeout: float) -> Response:
            opened.append((request.full_url, request.get_method(), timeout))
            return Response()

    def build_opener(*handlers: object) -> Opener:
        builders.append(handlers)
        return Opener()

    monkeypatch.setattr(urllib.request, "build_opener", build_opener)
    for _ in range(2):
        assert (
            native._urllib_transport(
                "POST",
                native.ENDPOINT,
                {"Authorization": "Bearer test-only", "Content-Type": "application/json"},
                b"{}",
                timeout=60.0,
            )[0]
            == 200
        )
    assert len(builders) == 2
    for handlers in builders:
        proxy = next(item for item in handlers if isinstance(item, urllib.request.ProxyHandler))
        redirect = next(
            item for item in handlers if isinstance(item, urllib.request.HTTPRedirectHandler)
        )
        proxy_any: Any = proxy
        redirect_any: Any = redirect
        assert proxy_any.proxies == {}
        assert (
            redirect_any.redirect_request(None, None, 302, "redirect", {}, "https://evil.test")
            is None
        )
    assert opened == [(native.ENDPOINT, "POST", 60.0)] * 2
    with pytest.raises(ValueError, match="literal"):
        native._urllib_transport("POST", "https://evil.test", {}, b"{}", timeout=60.0)


def test_timeout_is_not_retried(tmp_path: Path) -> None:
    calls = 0
    request = native.native_request(rows(1))

    def transport(*args: Any, **kwargs: Any) -> tuple[int, dict[str, str], bytes]:
        nonlocal calls
        calls += 1
        raise TimeoutError

    with pytest.raises(TimeoutError):
        native.dispatch_native(
            request,
            api_key="secret",
            journal=native.Journal(tmp_path / "journal", "c" * 64),
            batch_size=1,
            iteration=0,
            transport=transport,
        )
    assert calls == 1


def test_journal_rejects_noncanonical_and_out_of_order_transitions(tmp_path: Path) -> None:
    path = tmp_path / "cost-journal.jsonl"
    path.write_text(
        '{"sequence":1,"kind":"reserve","request_id":"native-b020-i00",'
        '"batch_size":20,"iteration":0,"reservation_cost_usd":"0.00320",'
        f'"predecessor_sha256":"{"a" * 64}"}}\n'
    )
    with pytest.raises(ValueError):
        native.Journal(path, "a" * 64)


def test_duplicate_response_keys_and_probability_drift_fail_closed() -> None:
    request = native.native_request(rows(1))
    duplicate = (
        b'{"model":"jev-1.13.0","model":"jev-1.13.0",'
        b'"answers":{},"usage":{"input_tokens":0,"output_tokens":0}}'
    )
    with pytest.raises(ValueError):
        native._load(duplicate)
    value = response(request)
    answer = next(iter(value["answers"].values()))
    answer["probabilities"]["billing"] = 0.5
    with pytest.raises(ValueError):
        native.validate_native_response(value, list(request["questions"]))


def test_report_is_human_text_not_embedded_result_json() -> None:
    partial = {
        "native": {
            "status": "partial",
            "computed_cost_usd": "0",
            "completed_call_count": 0,
            "journal_state": "open_reservation",
            "failure_category": "process_interrupted",
        },
        "limitations": list(native.MANDATORY_LIMITATIONS),
    }
    report = native.render_report(partial)
    assert "Partial result" in report
    assert '"native"' not in report


def test_http_failure_is_redacted_and_permanently_stops_dispatch(tmp_path: Path) -> None:
    request = native.native_request(rows(1))

    def transport(
        method: str, url: str, headers: Mapping[str, str], body: bytes, *, timeout: float
    ) -> tuple[int, Mapping[str, str], bytes]:
        return 429, {}, b'{"detail":"Bearer secret must not escape"}'

    journal = native.Journal(tmp_path / "journal", "d" * 64)
    with pytest.raises(RuntimeError) as failure:
        native.dispatch_native(
            request,
            api_key="secret",
            journal=journal,
            batch_size=1,
            iteration=0,
            transport=transport,
        )
    assert "secret" not in str(failure.value)
    assert journal.uncertain
    assert (
        native._load((tmp_path / "journal").read_bytes().splitlines()[-1])["category"]
        == "http_error"
    )


def test_artifact_validator_rejects_extra_file_and_wrong_permissions(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir(mode=0o700)
    (output / "extra.txt").write_text("unexpected")
    with pytest.raises(ValueError, match="allowlist"):
        native.validate_phase3d_artifacts(output, tmp_path / "predecessor", tmp_path / "packet")
    (output / "extra.txt").unlink()
    output.chmod(0o755)
    with pytest.raises(ValueError, match="directory"):
        native.validate_phase3d_artifacts(output, tmp_path / "predecessor", tmp_path / "packet")


def test_closed_models_reject_unknown_fields_and_nonfinite_values() -> None:
    with pytest.raises(ValueError):
        native.SummaryModel.model_validate(
            {"count": 1, "min_ms": 1.0, "p50_ms": 1.0, "p95_ms": 1.0, "max_ms": 1.0, "x": 1}
        )
    with pytest.raises(ValueError):
        native.SummaryModel.model_validate(
            {"count": 1, "min_ms": float("nan"), "p50_ms": 1.0, "p95_ms": 1.0, "max_ms": 1.0}
        )


def test_default_import_does_not_require_a_typesafe_sdk() -> None:
    import sys

    assert "typesafe" not in sys.modules


def _phase3c_control() -> Any:
    def workload(batch_size: int, *, remote: bool) -> Any:
        request_p50 = float(batch_size * (20 if remote else 2))
        return SimpleNamespace(
            summary=SimpleNamespace(p50_ms=request_p50, p95_ms=request_p50 + 1),
            per_item_summary=SimpleNamespace(
                p50_ms=request_p50 / batch_size,
                p95_ms=(request_p50 + 1) / batch_size,
            ),
            throughput_items_per_second=1000 * batch_size / request_p50,
            accuracy=0.8 if remote else 0.7,
            cost_usd="0.001" if remote else None,
        )

    return SimpleNamespace(
        local=SimpleNamespace(
            workloads={str(size): workload(size, remote=False) for size in JEV_BATCH_SIZES}
        ),
        jev=SimpleNamespace(
            status="complete",
            completed_call_count=30,
            workloads={str(size): workload(size, remote=True) for size in JEV_BATCH_SIZES},
        ),
    )


def _controlled_inputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[Path, Path, list[dict[str, Any]]]:
    predecessor = tmp_path / "phase3c"
    packet = tmp_path / "packet"
    predecessor.mkdir(mode=0o700)
    packet.mkdir(mode=0o700)
    (predecessor / "artifact-manifest.json").write_bytes(b'{"sealed":true}\n')
    packet_manifest = packet / "packet-manifest.json"
    packet_manifest.write_bytes(b'{"sealed":true}\n')
    source_rows = rows(20)
    phase3c = _phase3c_control()

    def preflight(_: Path, __: Path) -> tuple[Any, Path, list[dict[str, Any]]]:
        return phase3c, packet_manifest, source_rows

    monkeypatch.setattr(native, "_preflight_inputs", preflight)
    return predecessor, packet, source_rows


class _Clock:
    value = 0

    def perf_counter_ns(self) -> int:
        self.value += 5_000_000
        return self.value


def _native_transport(
    method: str, url: str, headers: Mapping[str, str], body: bytes, *, timeout: float
) -> tuple[int, Mapping[str, str], bytes]:
    request = json.loads(body)
    answers = {
        key: {
            "type": "choice",
            "choice": "billing",
            "probabilities": {
                label: (1.0 if label == "billing" else 0.0) for label in reversed(LABELS)
            },
            "confidence": 1.0,
        }
        for key in request["questions"]
    }
    return (
        200,
        {},
        json.dumps(
            {
                "model": native.MODEL_ID,
                "answers": answers,
                "usage": {"input_tokens": 100, "output_tokens": 20},
            }
        ).encode(),
    )


def _complete_artifact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[Path, Path, Path, dict[str, Any]]:
    predecessor, packet, source_rows = _controlled_inputs(monkeypatch, tmp_path)
    output = tmp_path / "output"
    predecessor_sha256 = digest_bytes((predecessor / "artifact-manifest.json").read_bytes())
    journal = native._create_run_output(output, predecessor, predecessor_sha256)
    run = native.run_native_workloads(
        source_rows,
        api_key="test-only",
        journal=journal,
        transport=_native_transport,
        clock=_Clock(),
    )
    assert run["completed_call_count"] == 30
    result = native.finalize_phase3d(predecessor, packet, output)
    return predecessor, packet, output, result


def test_full_run_finalization_comparison_and_single_use_claim(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    predecessor, packet, output, result = _complete_artifact(monkeypatch, tmp_path)
    validated = native.validate_phase3d_artifacts(output, predecessor, packet)
    assert validated.native.completed_call_count == 30
    assert result["comparison"]["1"]["local_to_native_per_item_p50_ratio"] == 2.5
    assert "per-item p50/p95" in (output / "report.md").read_text()
    assert native.finalize_phase3d(predecessor, packet, output) == result
    second_output = tmp_path / "second-output"
    predecessor_sha256 = digest_bytes((predecessor / "artifact-manifest.json").read_bytes())
    with pytest.raises(ValueError, match="claim"):
        native._create_run_output(second_output, predecessor, predecessor_sha256)
    assert not second_output.exists()


def test_resealed_comparison_tamper_is_rejected_against_predecessor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    predecessor, packet, output, result = _complete_artifact(monkeypatch, tmp_path)
    result["comparison"]["1"]["native_request_p50_ms"] = 999.0
    result_bytes = native._canon(result) + b"\n"
    (output / "result.json").write_bytes(result_bytes)
    (output / "result.json").chmod(0o600)
    (output / "report.md").write_bytes(native.render_report(result).encode())
    (output / "report.md").chmod(0o600)
    _, packet_manifest, source_rows = native._preflight_inputs(predecessor, packet)
    provenance = native._provenance(predecessor, packet_manifest, source_rows)
    claim = native._validate_claim(predecessor, provenance["predecessor_artifact_manifest_sha256"])
    (output / "artifact-manifest.json").write_bytes(
        native._manifest_bytes(output, provenance, claim)
    )
    (output / "artifact-manifest.json").chmod(0o600)
    with pytest.raises(ValueError, match="reconstruction"):
        native.validate_phase3d_artifacts(output, predecessor, packet)


@pytest.mark.parametrize(
    ("terminal", "category", "state"),
    [
        ("failure", "http_error", "failed"),
        ("open", "process_interrupted", "open_reservation"),
    ],
)
def test_no_network_partial_finalization_is_retry_safe(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    terminal: str,
    category: str,
    state: str,
) -> None:
    predecessor, packet, _ = _controlled_inputs(monkeypatch, tmp_path)
    output = tmp_path / "output"
    predecessor_sha256 = digest_bytes((predecessor / "artifact-manifest.json").read_bytes())
    journal = native._create_run_output(output, predecessor, predecessor_sha256)
    journal.reserve("native-b001-i00", 1, 0)
    if terminal == "failure":
        journal.terminal("native-b001-i00", 1, 0, "failure", category="http_error", status=429)
    result = native.finalize_partial(predecessor, packet, output)
    assert result["native"]["failure_category"] == category
    assert result["native"]["journal_state"] == state
    assert native.finalize_partial(predecessor, packet, output) == result


def test_settled_over_limit_stops_and_seals_partial(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    predecessor, packet, source_rows = _controlled_inputs(monkeypatch, tmp_path)
    output = tmp_path / "output"
    predecessor_sha256 = digest_bytes((predecessor / "artifact-manifest.json").read_bytes())
    journal = native._create_run_output(output, predecessor, predecessor_sha256)

    def over_limit(
        method: str, url: str, headers: Mapping[str, str], body: bytes, *, timeout: float
    ) -> tuple[int, Mapping[str, str], bytes]:
        _, response_headers, payload = _native_transport(
            method, url, headers, body, timeout=timeout
        )
        value = json.loads(payload)
        value["usage"]["input_tokens"] = 64001
        return 200, response_headers, json.dumps(value).encode()

    with pytest.raises(native.PartialRunError):
        native.run_native_workloads(
            source_rows,
            api_key="test-only",
            journal=journal,
            transport=over_limit,
            clock=_Clock(),
        )
    result = native.finalize_phase3d(predecessor, packet, output)
    assert result["native"]["failure_category"] == "reported_usage_over_limit"
    assert result["native"]["completed_call_count"] == 1
    assert len(journal.rows) == 2
