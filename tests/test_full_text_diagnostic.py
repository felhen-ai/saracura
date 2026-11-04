import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Literal

import pytest


def _load_research_module(name: str) -> Any:
    path = Path(__file__).parents[1] / "research" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_load_research_module("ptbr_bench")
d = _load_research_module("full_text_diagnostic")


def _legacy_rows() -> dict[str, list[dict[str, Any]]]:
    return {
        task: [
            {
                "type": "choice",
                "text": f"{task} text {index}",
                "instructions": "choose",
                "criteria": {"a": "A", "b": "B"},
                "gold": "a",
            }
            for index in range(count)
        ]
        for task, count in d.LEGACY_COUNTS.items()
    }


def test_v3_full_preparation_freezes_all_893_legal_rows_and_legacy_rng(tmp_path: Path) -> None:
    rows, source, output = _legacy_rows(), tmp_path / "rows.json", tmp_path / "prepared.json"
    source.write_text(json.dumps(rows))
    prepared = d.prepare(source, output, mode="v3legalfull")

    assert prepared["legacy_counts"] == d.LEGACY_COUNTS
    assert prepared["legacy_order_stream_sha256"] == d.sha(d.legacy_orders(rows))
    assert len(prepared["tasks"]["juristcu_area"]) == 893
    assert (
        prepared["tasks"]["juristcu_area"][0]["question_order"]
        == d.legacy_orders(rows)["juristcu_area"][0]
    )
    assert d.load_prepared(output)["prepared_sha256"] == prepared["prepared_sha256"]


def test_full_prepared_mode_is_rejected_by_diagnostic_execution_helpers(tmp_path: Path) -> None:
    rows, source, prepared_file = _legacy_rows(), tmp_path / "rows.json", tmp_path / "prepared.json"
    source.write_text(json.dumps(rows))
    prepared = d.prepare(source, prepared_file, mode="v3legalfull")
    checkpoint = {"id": "model", "revision": "revision", "base": "base"}
    with pytest.raises(ValueError, match="not supported"):
        d._expected(prepared, rows, "model", checkpoint, {}, "evidence")
    with pytest.raises(ValueError, match="not supported"):
        d.validate_journal(
            tmp_path / "journal.jsonl",
            prepared,
            rows,
            "model",
            checkpoint,
            {},
            "evidence",
            complete=False,
        )
    with pytest.raises(ValueError, match="not supported"):
        d.summarize(
            prepared,
            rows,
            tmp_path / "journal.jsonl",
            model="model",
            checkpoint=checkpoint,
            server_evidence_file="evidence",
        )
    with pytest.raises(ValueError, match="not supported"):
        d.run_pairs(
            prepared_file,
            source,
            tmp_path / "journal.jsonl",
            "http://127.0.0.1:18032",
            "model",
            checkpoint=checkpoint,
            server_evidence_file="evidence",
        )


def test_v3_preparation_freezes_32_case_sensitive_orders(tmp_path: Path) -> None:
    names = [f"n{i}" for i in range(32)]
    rows = [
        {
            "type": "choice",
            "text": str(i),
            "criteria": {name: name for name in names},
            "instructions": "x",
            "accepted_labels": [names[i % 32]],
        }
        for i in range(599)
    ]
    bp, out = tmp_path / "b.jsonl", tmp_path / "p.json"
    bp.write_text("".join(json.dumps(row) + "\n" for row in rows))
    p = d.prepare(bp, out, mode="v3camara", builder_file=bp)
    assert p["tasks"][0:] if False else p["tasks"]["camara_tema"][0]["question_order"] != names
    assert len(p["tasks"]["camara_tema"][0]["question_order"]) == 32
    assert d.load_prepared(out)["prepared_sha256"] == p["prepared_sha256"]


def test_loopback_only_and_resume_identity_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        d.loopback_url("https://example.com")
    journal = tmp_path / "attempts.jsonl"
    d.append_jsonl(journal, {"identity": {"x": 1}, "original_index": 0, "arm": "prefix"})
    assert json.loads(journal.read_text())["identity"] == {"x": 1}


def _llm_health_response(value: Any) -> Any:
    class Response:
        def __init__(self) -> None:
            self.headers: dict[str, str] = {}

        def __enter__(self) -> Any:
            return self

        def __exit__(self, *_args: object) -> Literal[False]:
            return False

        def read(self, _limit: int) -> bytes:
            return json.dumps(value).encode("utf-8")

    return Response()


def _bound_llm_evidence() -> tuple[dict[str, Any], dict[str, Any]]:
    bridge = _load_research_module("llm_bridge")

    config = bridge.Config(
        "fixture/model",
        "fixture-revision",
        "cpu",
        16384,
        2,
        {},
        dtype="bfloat16",
        readout="constrained-letter",
    )
    backend = {"runtime": "transformers", "dtype": "bfloat16", "python": "fixture"}
    live = bridge.build_evidence(config, backend, "a" * 64)
    expected = {
        "checkpoint": {
            "id": "fixture/model",
            "revision": "fixture-revision",
            "base": "fixture/base",
        },
        "backend": live["backend"],
        "runtime": live["backend"],
        "dtype": live["backend"]["dtype"],
        "context": live["limits"]["context"],
        "prompt": live["prompt"],
        "server": {
            "health_evidence": live,
            "health_evidence_sha256": d.sha(d.canonical(live) + b"\n"),
        },
        "preflight": {
            "no_truncation": True,
            "context_check": {"ok": True},
            "token_check": {"ok": True},
        },
    }
    return expected, live


def test_wire_body_preserves_frozen_criteria_insertion_order() -> None:
    question = {"criteria": {"third": "3", "first": "1", "second": "2"}}
    assert d.wire_body("model", "state", question) == (
        b'{"model":"model","state":"state","questions":{"q":'
        b'{"criteria":{"third":"3","first":"1","second":"2"}}}}'
    )


def test_llm_live_evidence_binding_accepts_exact_pinned_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected, live = _bound_llm_evidence()
    health = {"evidence": live, "evidence_sha256": d.sha(d.canonical(live) + b"\n")}
    monkeypatch.setattr(
        d.urllib.request, "urlopen", lambda *_args, **_kwargs: _llm_health_response(health)
    )
    assert d.preflight("http://127.0.0.1:18032", expected) == health


@pytest.mark.parametrize("tamper", ["checkpoint", "backend", "context"])
def test_llm_live_evidence_binding_rejects_metadata_tampering(
    monkeypatch: pytest.MonkeyPatch, tamper: str
) -> None:
    expected, live = _bound_llm_evidence()
    live = json.loads(json.dumps(live))
    if tamper == "checkpoint":
        live["model"]["id"] = "other/model"
    elif tamper == "backend":
        live["backend"]["runtime"] = "other-backend"
    else:
        live["limits"]["context"] = 1
    expected["server"] = {
        "health_evidence": live,
        "health_evidence_sha256": d.sha(d.canonical(live) + b"\n"),
    }
    health = {"evidence": live, "evidence_sha256": d.sha(d.canonical(live) + b"\n")}
    monkeypatch.setattr(
        d.urllib.request, "urlopen", lambda *_args, **_kwargs: _llm_health_response(health)
    )
    with pytest.raises(ValueError):
        d.preflight("http://127.0.0.1:18032", expected)


def test_actual_v2_pair_shape_requires_all_893_rows() -> None:
    rows = [{"gold": "a", "criteria": {"a": "A", "b": "B"}} for _ in range(893)]
    report = {"items": {"juristcu_area": [["a", "b"] for _ in rows]}}
    assert d._report_predictions(report, rows) == ["b"] * 893
    report["items"]["juristcu_area"].pop()
    with pytest.raises(ValueError, match="893"):
        d._report_predictions(report, rows)


def test_native_probability_rejects_bool_and_missing_keys() -> None:
    question = {"criteria": {"a": "A", "b": "B"}}
    assert d._prediction({"probabilities": {"a": True, "b": 0.0}}, question) is None
    assert d._prediction({"probabilities": {"a": 1.0}}, question) is None


def test_nested_native_probabilities_bind_the_recorded_prediction() -> None:
    question = {"criteria": {"a": "A", "b": "B"}}
    assert (
        d._prediction({"answers": {"q": {"probabilities": {"a": 1.0, "b": 0.0}}}}, question) == "a"
    )
    assert (
        d._prediction(
            {"answers": {"q": {"choice": "a", "probabilities": {"a": 0.0, "b": 1.0}}}},
            question,
        )
        is None
    )
    assert (
        d._prediction({"answers": {"q": {"choice": "a", "error": "bad payload"}}}, question) is None
    )


def test_native_choice_accepts_rounded_probability_tie() -> None:
    q = {"criteria": {"first": "First", "second": "Second"}}
    response: dict[str, Any] = {
        "answers": {"q": {"choice": "second", "probabilities": {"first": 0.5, "second": 0.5}}}
    }
    assert d._prediction(response, q) == "second"
    response["answers"]["q"]["probabilities"]["second"] = 0.4
    assert d._prediction(response, q) is None


def test_run_pairs_stops_after_preserving_first_operational_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    journal = tmp_path / "attempts.jsonl"
    expected = {
        (0, "prefix"): {
            "original_index": 0,
            "arm": "prefix",
            "question": {"criteria": {"a": "A"}},
            "state": "first",
        },
        (0, "full"): {
            "original_index": 0,
            "arm": "full",
            "question": {"criteria": {"a": "A"}},
            "state": "second",
        },
    }
    calls: list[str] = []
    monkeypatch.setattr(d, "load_prepared", lambda _path: {"rows_source_sha256": "source"})
    monkeypatch.setattr(d, "_read_rows", lambda _path: {})
    monkeypatch.setattr(d, "source_hash", lambda _path: "source")
    monkeypatch.setattr(d, "_checkpoint", lambda _checkpoint: None)
    monkeypatch.setattr(d, "load_server_evidence", lambda *_args: {})
    monkeypatch.setattr(d, "preflight", lambda *_args: None)
    monkeypatch.setattr(d, "validate_journal", lambda *_args, **_kwargs: (expected, {}))

    def failing_request(*_args: object, **_kwargs: object) -> tuple[None, None, float, str, str]:
        calls.append("request")
        return None, None, 0.0, "invalid_json", "request-sha"

    monkeypatch.setattr(d, "request", failing_request)
    monkeypatch.setattr(
        d, "summarize", lambda *_args, **_kwargs: pytest.fail("summary must not run")
    )
    with pytest.raises(ValueError, match="operational failure preserved"):
        d.run_pairs(
            "prepared",
            "rows",
            journal,
            "http://127.0.0.1:18032",
            "fixture/model",
            checkpoint={"id": "fixture/model"},
            server_evidence_file="evidence",
        )
    assert calls == ["request"]
    record = json.loads(journal.read_text())
    assert record["error"] == "invalid_json"
    assert record["arm"] == "prefix"
