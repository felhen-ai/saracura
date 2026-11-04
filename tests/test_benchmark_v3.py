"""Self-authored protocol tests for the V3 local reporting CLI."""

from __future__ import annotations

import importlib.util
import json
import sys
import threading
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast

import pytest


def _load_research_module(name: str) -> Any:
    path = Path(__file__).resolve().parents[1] / "research" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


camara_v3 = _load_research_module("camara_v3")
ptbr_bench = _load_research_module("ptbr_bench")
diagnostic = _load_research_module("full_text_diagnostic")
benchmark_v3 = _load_research_module("benchmark_v3")


def _criteria() -> dict[str, str]:
    return {f"topic_{index:02d}": f"Tema {index}" for index in range(32)}


def _camara_rows() -> list[dict[str, Any]]:
    criteria = _criteria()
    return [
        {
            "id": f"camara-v3-{index}",
            "text": f"Texto {index}",
            "instructions": "Escolha um tema.",
            "type": "choice",
            "criteria": criteria,
            "accepted_labels": [f"topic_{index % 32:02d}"],
        }
        for index in range(599)
    ]


def _checkpoint() -> dict[str, str]:
    return {"id": "org/model", "revision": "r1", "base": "base/model"}


def _response_digest() -> str:
    return cast(str, diagnostic.raw_response_sha('{"answers":{"q":{}}}'))


def _decisions(
    task: str,
    rows: list[dict[str, Any]],
    questions: list[tuple[dict[str, Any], int]],
    *,
    api_alias: str = "alias",
) -> list[dict[str, Any]]:
    return [
        benchmark_v3._decision(
            task,
            index,
            row,
            question,
            chars,
            row["accepted_labels"][0] if task == "camara_tema" else row["gold"],
            None,
            diagnostic.sha(diagnostic.wire_body(api_alias, row["text"], question)),
            _response_digest(),
        )
        for index, (row, (question, chars)) in enumerate(zip(rows, questions, strict=True))
    ]


def test_camara_scorer_accepts_a_nonfirst_official_alternative() -> None:
    criteria = _criteria()
    rows = [
        {
            "id": "one",
            "criteria": criteria,
            "accepted_labels": ["topic_00", "topic_17"],
        },
        {"id": "two", "criteria": criteria, "accepted_labels": ["topic_01"]},
    ]
    score = camara_v3.score_predictions(rows, {"one": "topic_17", "two": "topic_01"})
    assert score["raw_any_topic_hit"] == 100.0
    assert score["macro_valid_topic"] == 100.0


def test_camara_wire_order_keeps_all_32_choices() -> None:
    rows = _camara_rows()
    questions = benchmark_v3._camara_questions(rows)
    first_question, _ = questions[0]
    body = diagnostic.wire_body("fake", "state", first_question)
    decoded = json.loads(body)
    assert len(decoded["questions"]["q"]["criteria"]) == 32
    assert list(decoded["questions"]["q"]["criteria"]) == list(first_question["criteria"])


def test_fake_loopback_http_uses_shared_wire_and_strict_prediction() -> None:
    criteria = _criteria()
    observed: list[dict[str, Any]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers["Content-Length"])
            observed.append(json.loads(self.rfile.read(length)))
            payload = json.dumps({"answers": {"q": {"choice": "topic_17"}}}).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, _format: str, *_args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    binding = benchmark_v3._server_binding(f"http://127.0.0.1:{server.server_port}", None, {})
    assert "endpoint" not in binding
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        question = {"type": "choice", "instructions": "x", "criteria": criteria}
        result, raw, _duration, error, digest = diagnostic.request(
            f"http://127.0.0.1:{server.server_port}", "fake", "state", question, 5
        )
    finally:
        server.shutdown()
        thread.join()
    assert error is None
    assert benchmark_v3._typed_prediction(result, question) == "topic_17"
    assert raw is not None and len(digest) == 64
    assert list(observed[0]["questions"]["q"]["criteria"]) == list(criteria)


def test_task_report_rejects_extra_ids_errors_and_wrong_seed_order() -> None:
    rows = _camara_rows()
    questions = benchmark_v3._camara_questions(rows)
    decisions = _decisions("camara_tema", rows, questions)
    report = benchmark_v3._task_report(
        task="camara_tema",
        rows=rows,
        decisions=decisions,
        checkpoint=_checkpoint(),
        api_alias="alias",
        source_digest="c" * 64,
        source_manifest_identity="d" * 64,
        server={"binding": "self-reported"},
    )
    benchmark_v3._validate_report(
        report,
        task="camara_tema",
        rows=rows,
        source_digest="c" * 64,
        source_identity="d" * 64,
    )
    with pytest.raises(benchmark_v3.ReportError, match="source identity"):
        benchmark_v3._validate_report(
            report,
            task="camara_tema",
            rows=rows,
            source_digest="e" * 64,
            source_identity="d" * 64,
        )
    report["rows"].append(dict(report["rows"][0]))
    report["attempted_rows"] += 1
    with pytest.raises(benchmark_v3.ReportError, match="extra IDs"):
        benchmark_v3._validate_report(
            report,
            task="camara_tema",
            rows=rows,
            source_digest="c" * 64,
            source_identity="d" * 64,
        )
    report["rows"].pop()
    report["attempted_rows"] -= 1
    report["rows"][0]["error"] = "transport"
    report["failures"] = 1
    report["complete"] = False
    with pytest.raises(benchmark_v3.ReportError, match="complete zero-failure"):
        benchmark_v3._validate_report(
            report,
            task="camara_tema",
            rows=rows,
            source_digest="c" * 64,
            source_identity="d" * 64,
        )


def test_task_report_rejects_missing_tampered_or_wrong_api_digests() -> None:
    rows = _camara_rows()
    report = benchmark_v3._task_report(
        task="camara_tema",
        rows=rows,
        decisions=_decisions("camara_tema", rows, benchmark_v3._camara_questions(rows)),
        checkpoint=_checkpoint(),
        api_alias="alias",
        source_digest="c" * 64,
        source_manifest_identity="d" * 64,
        server={"binding": "self-reported"},
    )
    for field, value, message in (
        ("request_sha256", None, "request digest"),
        ("response_sha256", None, "response digest"),
        ("response_sha256", "not-a-sha", "response digest"),
        ("request_sha256", "0" * 64, "request digest"),
    ):
        mutated = deepcopy(report)
        if value is None:
            del mutated["rows"][0][field]
        else:
            mutated["rows"][0][field] = value
        with pytest.raises(benchmark_v3.ReportError, match=message):
            benchmark_v3._validate_report(
                mutated,
                task="camara_tema",
                rows=rows,
                source_digest="c" * 64,
                source_identity="d" * 64,
            )
    mutated = deepcopy(report)
    mutated["api_alias"] = "different-served-model"
    with pytest.raises(benchmark_v3.ReportError, match="request digest"):
        benchmark_v3._validate_report(
            mutated,
            task="camara_tema",
            rows=rows,
            source_digest="c" * 64,
            source_identity="d" * 64,
        )
    altered_rows = deepcopy(rows)
    altered_rows[0]["text"] = "Changed"
    with pytest.raises(benchmark_v3.ReportError, match="request digest"):
        benchmark_v3._validate_report(
            report,
            task="camara_tema",
            rows=altered_rows,
            source_digest="c" * 64,
            source_identity="d" * 64,
        )


def test_legacy_boolean_predictions_are_checked_as_booleans() -> None:
    legacy: dict[str, list[dict[str, object]]] = {}
    hashes = {}
    for task, count in benchmark_v3.LEGACY_COUNTS.items():
        legacy[task] = [
            {"type": "noul", "text": "x", "instructions": "x", "gold": bool(index % 2)}
            for index in range(count)
        ]
        hashes[task] = f"{len(task):064x}"
    items = {task: [[row["gold"], row["gold"]] for row in rows] for task, rows in legacy.items()}
    report: dict[str, Any] = {
        "revision": "a" * 40,
        "checkpoint": {"id": "org/model", "revision": "r1"},
        "items": items,
        "dataset_files": {task: {"sha256": hashes[task]} for task in legacy},
        "tasks": {
            task: {"n": len(rows), "source_n": len(rows), "failed": 0, "balanced_accuracy": 1.0}
            for task, rows in legacy.items()
        },
    }
    benchmark_v3._validate_legacy_report(report, legacy, hashes, "a" * 40)
    report["items"]["olid_alvo"][0][1] = "True"
    with pytest.raises(benchmark_v3.ReportError, match="typed pair"):
        benchmark_v3._validate_legacy_report(report, legacy, hashes, "a" * 40)


def test_full_legal_report_never_accepts_429_as_complete() -> None:
    rows = [
        {
            "id": f"juristcu_area-{index}",
            "text": "x",
            "gold": "a",
            "type": "choice",
            "criteria": {"a": "A", "b": "B"},
        }
        for index in range(429)
    ]
    decisions = [
        {
            "index": index,
            "id": row["id"],
            "gold": "a",
            "prediction": "a",
            "error": None,
            "exact_chars_sent": 1,
            "question_order": ["a", "b"],
            "request_sha256": "a" * 64,
            "response_sha256": "b" * 64,
        }
        for index, row in enumerate(rows)
    ]
    report = {
        "schema_version": 1,
        "kind": "task-report-v3",
        "task": "juristcu_area",
        "checkpoint": _checkpoint(),
        "api_alias": "alias",
        "source_sha256": "c" * 64,
        "source_manifest_identity": None,
        "full_input": True,
        "option_order_seed": 20261009,
        "expected_rows": 429,
        "attempted_rows": 429,
        "failures": 0,
        "complete": True,
        "rows": decisions,
        "computed_score": {"balanced_accuracy": 1.0},
    }
    with pytest.raises(benchmark_v3.ReportError, match="row count metadata"):
        benchmark_v3._validate_report(
            report,
            task="juristcu_area",
            rows=rows,
            source_digest="c" * 64,
            source_identity=None,
        )


def _legacy_rows() -> tuple[dict[str, list[dict[str, Any]]], dict[str, str]]:
    rows = {
        task: [
            {
                "id": f"{task}-{index}",
                "type": "choice",
                "text": f"{task} text {index}",
                "instructions": "Escolha.",
                "gold": "a" if index % 2 == 0 else "b",
                "criteria": {"a": "A", "b": "B"},
            }
            for index in range(count)
        ]
        for task, count in benchmark_v3.LEGACY_COUNTS.items()
    }
    return rows, {task: f"{len(task):064x}" for task in rows}


def _legal_report(
    legacy: dict[str, list[dict[str, Any]]], hashes: dict[str, str]
) -> dict[str, Any]:
    rows = legacy["juristcu_area"]
    return cast(
        dict[str, Any],
        benchmark_v3._task_report(
            task="juristcu_area",
            rows=rows,
            decisions=_decisions(
                "juristcu_area",
                rows,
                benchmark_v3._legal_questions(legacy),
                api_alias="legal-alias",
            ),
            checkpoint=_checkpoint(),
            api_alias="legal-alias",
            source_digest=hashes["juristcu_area"],
            source_manifest_identity=None,
            server={"binding": "self-reported"},
        ),
    )


def _legacy_v2_report(
    legacy: dict[str, list[dict[str, Any]]], hashes: dict[str, str]
) -> dict[str, Any]:
    items: dict[str, list[list[str]]] = {}
    tasks: dict[str, dict[str, Any]] = {}
    for task, rows in legacy.items():
        pairs = [
            [
                row["gold"],
                row["gold"] if index % (len(task) % 4 + 2) else "b" if row["gold"] == "a" else "a",
            ]
            for index, row in enumerate(rows)
        ]
        items[task] = pairs
        tasks[task] = {
            "n": len(rows),
            "source_n": len(rows),
            "failed": 0,
            "balanced_accuracy": ptbr_bench.balanced(pairs),
        }
    return {
        "revision": "a" * 40,
        "checkpoint": {"id": "org/model", "revision": "r1"},
        "items": items,
        "dataset_files": {task: {"sha256": hashes[task]} for task in legacy},
        "tasks": tasks,
    }


def test_validate_rejects_reversed_isolated_legal_replay_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    legacy, hashes = _legacy_rows()
    report = _legal_report(legacy, hashes)
    report_path = tmp_path / "legal.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.setattr(benchmark_v3, "_load_legacy", lambda _cache, _revision: (legacy, hashes))
    args = __import__("argparse").Namespace(
        report=report_path,
        camara_dir=None,
        legacy_cache=tmp_path,
        legacy_revision="a" * 40,
        legacy_report=None,
        camara_report=None,
        protocol=None,
        legal_report=None,
    )
    assert benchmark_v3.validate(args) == 0
    report["rows"][0]["question_order"].reverse()
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(benchmark_v3.ReportError, match="question order"):
        benchmark_v3.validate(args)


def test_combine_uses_actual_equal_seven_task_mean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    legacy, hashes = _legacy_rows()
    camara_rows = _camara_rows()
    camara_decisions = _decisions(
        "camara_tema", camara_rows, benchmark_v3._camara_questions(camara_rows)
    )
    for index, decision in enumerate(camara_decisions):
        if index % 3 == 0:
            decision["prediction"] = f"topic_{(index + 1) % 32:02d}"
    camara = benchmark_v3._task_report(
        task="camara_tema",
        rows=camara_rows,
        decisions=camara_decisions,
        checkpoint=_checkpoint(),
        api_alias="alias",
        source_digest="c" * 64,
        source_manifest_identity="d" * 64,
        server={"binding": "self-reported"},
    )
    legacy_path, camara_path, protocol_path, output = (
        tmp_path / "legacy.json",
        tmp_path / "camara.json",
        tmp_path / "protocol.json",
        tmp_path / "combined.json",
    )
    legacy_path.write_text(json.dumps(_legacy_v2_report(legacy, hashes)), encoding="utf-8")
    camara_path.write_text(json.dumps(camara), encoding="utf-8")
    protocol_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "scoring_version": "ptbr-official-topics-v3",
                "legal_input": "prefix3500",
                "camara_data_sha256": "c" * 64,
                "legacy_dataset_revision": "a" * 40,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(benchmark_v3, "_load_legacy", lambda _cache, _revision: (legacy, hashes))
    monkeypatch.setattr(
        benchmark_v3,
        "_load_camara",
        lambda _directory: (camara_rows, {}, "c" * 64, "d" * 64),
    )
    assert (
        benchmark_v3.combine(
            __import__("argparse").Namespace(
                protocol=protocol_path,
                legacy_cache=tmp_path,
                legacy_report=legacy_path,
                camara_dir=tmp_path,
                camara_report=camara_path,
                legal_report=None,
                output=output,
            )
        )
        == 0
    )
    combined = json.loads(output.read_text(encoding="utf-8"))
    assert combined["source_digests"]["camara_tema"] == "c" * 64
    assert combined["legacy_source_digests"]["camara_tema"] == hashes["camara_tema"]
    points = [value["normalized_0_100_points"] for value in combined["tasks"].values()]
    assert combined["normalized_0_100_points"] == sum(points) / 7
    weighted_by_rows = sum(
        value["normalized_0_100_points"] * len(value["decisions"])
        for value in combined["tasks"].values()
    ) / sum(len(value["decisions"]) for value in combined["tasks"].values())
    assert combined["normalized_0_100_points"] != weighted_by_rows


def test_combine_rejects_checkpoint_mismatch_and_wrong_protocol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    camara_rows = _camara_rows()
    questions = benchmark_v3._camara_questions(camara_rows)
    decisions = _decisions(
        "camara_tema", camara_rows, questions, api_alias="different-alias-is-allowed"
    )
    camara = benchmark_v3._task_report(
        task="camara_tema",
        rows=camara_rows,
        decisions=decisions,
        checkpoint=_checkpoint(),
        api_alias="different-alias-is-allowed",
        source_digest="c" * 64,
        source_manifest_identity="d" * 64,
        server={"binding": "self-reported"},
    )
    legacy: dict[str, list[dict[str, object]]] = {}
    hashes = {}
    for task, count in benchmark_v3.LEGACY_COUNTS.items():
        legacy[task] = [
            {"type": "noul", "text": "x", "instructions": "x", "gold": False} for _ in range(count)
        ]
        hashes[task] = f"{len(task):064x}"
    legacy_report = {
        "revision": "a" * 40,
        "checkpoint": {"id": "other/model", "revision": "r1"},
        "items": {task: [[False, False] for _ in rows] for task, rows in legacy.items()},
        "dataset_files": {task: {"sha256": hashes[task]} for task in legacy},
        "tasks": {
            task: {"n": len(rows), "source_n": len(rows), "failed": 0, "balanced_accuracy": 1.0}
            for task, rows in legacy.items()
        },
    }
    monkeypatch.setattr(benchmark_v3, "_load_legacy", lambda _cache, _revision: (legacy, hashes))
    monkeypatch.setattr(
        benchmark_v3,
        "_load_camara",
        lambda _directory: (camara_rows, {}, "c" * 64, "d" * 64),
    )
    legacy_path, camara_path, protocol_path = (
        tmp_path / "legacy.json",
        tmp_path / "camara.json",
        tmp_path / "protocol.json",
    )
    legacy_path.write_text(json.dumps(legacy_report), encoding="utf-8")
    camara_path.write_text(json.dumps(camara), encoding="utf-8")
    protocol = {
        "schema_version": 1,
        "scoring_version": "ptbr-official-topics-v3",
        "legal_input": "prefix3500",
        "camara_data_sha256": "c" * 64,
        "legacy_dataset_revision": "a" * 40,
    }
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    args = __import__("argparse").Namespace(
        protocol=protocol_path,
        legacy_cache=tmp_path,
        legacy_report=legacy_path,
        camara_dir=tmp_path,
        camara_report=camara_path,
        legal_report=None,
        output=tmp_path / "output.json",
    )
    with pytest.raises(benchmark_v3.ReportError, match="checkpoint id/revision differ"):
        benchmark_v3.combine(args)
    protocol["scoring_version"] = "wrong"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    with pytest.raises(benchmark_v3.ReportError, match="protocol schema/version"):
        benchmark_v3.combine(args)
