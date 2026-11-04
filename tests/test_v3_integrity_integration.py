"""Decision-gate QA using self-authored corpora and real loopback HTTP only."""

import importlib.util
import json
import random
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest


def _load_research_module(name: str) -> Any:
    path = Path(__file__).parents[1] / "research" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


legacy = _load_research_module("ptbr_bench")
d = _load_research_module("full_text_diagnostic")

CHECKPOINT = {"id": "fixture/model", "revision": "fixture-revision", "base": "fixture/base"}
PairedFixture = tuple[Path, dict[str, Any], dict[str, Any], Path, Path, dict[str, Any]]


def evidence(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "checkpoint": CHECKPOINT,
                "runtime": "self-authored-test-runtime",
                "backend": "fake",
                "dtype": "test",
                "context": 16384,
                "prompt": {"fixture": True},
                "server": {"fixture": True},
                "preflight": {
                    "no_truncation": True,
                    "context_check": {"fixture": True},
                    "token_check": {"fixture": True},
                },
            }
        )
    )


@contextmanager
def endpoint(
    mode: str, metadata: Path | None = None, wrong_binding: bool = False
) -> Iterator[tuple[str, list[dict[str, Any]]]]:
    calls: list[dict[str, Any]] = []
    digest = (
        "0" * 64 if wrong_binding else d.sha(json.loads(metadata.read_text())) if metadata else None
    )

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: object) -> None:
            pass

        def respond(self, value: Any) -> None:
            body = json.dumps(value).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            if self.command == "GET" and digest is not None:
                self.send_header("X-Saracura-Evidence-SHA256", digest)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            self.respond({"fixture": True})

        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append(body)
            index = int(body["state"].split("|", 1)[0])
            criteria = body["questions"]["q"]["criteria"]
            if mode == "camara":
                prediction = f"c{index % 31}"
                self.respond(
                    {
                        "answers": {
                            "q": {
                                "probabilities": {
                                    key: 1.0 if key == prediction else 0.0 for key in criteria
                                }
                            }
                        }
                    }
                )
                return
            prediction = f"c{index % 10}"
            full = len(body["state"]) > 3500
            if index < 400 and index % 10 == 0 and not full:
                prediction = "c1"
            if index == 1:
                prediction = "c3" if full else "c2"
            if index == 2 and full:
                prediction = "c3"
            self.respond({"answers": {"q": {"choice": prediction}}})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", calls
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.fixture(scope="module")
def paired(tmp_path_factory: pytest.TempPathFactory) -> PairedFixture:
    root = tmp_path_factory.mktemp("paired-http")
    rows = {}
    for task, count in d.LEGACY_COUNTS.items():
        boolean = task in {"faquad_resposta", "olid_ofensivo"}
        criteria = {f"c{i}": f"Label {i}" for i in range(10)}
        rows[task] = [
            dict(
                type="noul" if boolean else "choice",
                instructions="Choose a label",
                text=f"{i}|" + ("x" * 3501 if task == "juristcu_area" and i < 429 else "short"),
                gold=bool(i % 2) if boolean else f"c{i % 10}",
                **({} if boolean else {"criteria": criteria}),
            )
            for i in range(count)
        ]
    source, prepared, journal, meta = [
        root / x for x in ("rows.json", "prepared.json", "journal.jsonl", "evidence.json")
    ]
    source.write_text(json.dumps(rows))
    p = d.prepare(source, prepared)
    rng = random.Random(legacy.SEED)
    expected = {}
    for task in legacy.TASKS:
        expected[task] = [list(legacy.question(row, rng).get("criteria", {})) for row in rows[task]]
    assert [r["question_order"] for r in p["tasks"]["juristcu_area"]] == expected["juristcu_area"]
    evidence(meta)
    report = {"items": {"juristcu_area": [[r["gold"], r["gold"]] for r in rows["juristcu_area"]]}}
    with endpoint("legal", meta) as (base, calls):
        result = d.run_pairs(
            prepared,
            source,
            journal,
            base,
            CHECKPOINT["id"],
            checkpoint=CHECKPOINT,
            server_evidence_file=meta,
            original_report=report,
        )
        assert len(calls) == 858
        for i, (before, after) in enumerate(zip(calls[::2], calls[1::2], strict=True)):
            assert (
                list(before["questions"]["q"]["criteria"])
                == p["tasks"]["juristcu_area"][i]["question_order"]
            )
            assert before["questions"] == after["questions"]
            assert before["state"] == after["state"][:3500]
        d.run_pairs(
            prepared,
            source,
            journal,
            base,
            CHECKPOINT["id"],
            checkpoint=CHECKPOINT,
            server_evidence_file=meta,
            original_report=report,
        )
        assert len(calls) == 858
    return root, rows, p, journal, meta, result


def test_complete_paired_gate_and_real_v2_shape(paired: PairedFixture) -> None:
    _, _, _, _, _, result = paired
    assert result["complete"] and result["material_gate"]
    assert result["completed_attempts"] == 858
    assert result["delta_balanced_accuracy_points"] == pytest.approx(100 / 10 * (40 / 90 - 1 / 90))
    assert result["raw_flip_counts"] == {
        "wrong_to_right": 40,
        "right_to_wrong": 1,
        "wrong_to_other_wrong": 1,
        "unchanged_prediction": 387,
    }
    assert result["prefix_vs_original_v2_concordance"] == pytest.approx(388 / 429)
    assert result["prefix_agreement_denominator"] == 429


@pytest.mark.parametrize(
    "field",
    [
        "identity",
        "request_sha256",
        "source_file_hashes",
        "gold",
        "prediction",
        "legal_prediction",
        "truncated",
        "original_index",
        "question_order",
    ],
)
def test_changed_evidence_cannot_open_gate(
    paired: PairedFixture,
    tmp_path: Path,
    field: str,
) -> None:
    _, rows, prepared, journal, meta, _ = paired
    records = [json.loads(line) for line in journal.read_text().splitlines()]
    r = records[0]
    replacements = {
        "identity": {},
        "request_sha256": "wrong",
        "source_file_hashes": {},
        "gold": "c9",
        "prediction": "illegal",
        "truncated": True,
        "original_index": False,
        "question_order": list(reversed(r["question_order"])),
    }
    if field == "legal_prediction":
        r["prediction"] = "c9"
    else:
        r[field] = replacements[field]
    corrupted = tmp_path / "corrupted.jsonl"
    corrupted.write_text("".join(json.dumps(row) + "\n" for row in records))
    with pytest.raises(ValueError):
        d.summarize(
            prepared,
            rows,
            corrupted,
            model=CHECKPOINT["id"],
            checkpoint=CHECKPOINT,
            server_evidence_file=meta,
        )


@pytest.mark.parametrize("kind", ["missing", "duplicate", "extra", "partial-line"])
def test_incomplete_or_duplicate_journal_is_rejected(
    paired: PairedFixture,
    tmp_path: Path,
    kind: str,
) -> None:
    _, rows, prepared, journal, meta, _ = paired
    lines = journal.read_text().splitlines()
    if kind == "missing":
        lines.pop()
    elif kind == "duplicate":
        lines.append(lines[0])
    elif kind == "extra":
        rec = json.loads(lines[0])
        rec["original_index"] = 900
        lines.append(json.dumps(rec))
    value = "\n".join(lines) + ("" if kind == "partial-line" else "\n")
    target = tmp_path / "incomplete.jsonl"
    target.write_text(value)
    with pytest.raises(ValueError):
        d.summarize(
            prepared,
            rows,
            target,
            model=CHECKPOINT["id"],
            checkpoint=CHECKPOINT,
            server_evidence_file=meta,
        )


def test_camara_599_native_probabilities_and_full_state(tmp_path: Path) -> None:
    criteria = {f"c{i}": f"Official fixture {i}" for i in range(32)}
    rows = [
        dict(
            id=str(i),
            type="choice",
            text=f"{i}|full text",
            instructions="One official theme",
            criteria=criteria,
            accepted_labels=[f"c{i % 31}"],
        )
        for i in range(599)
    ]
    source, prepared, journal, meta = [
        tmp_path / x for x in ("rows.jsonl", "prepared.json", "journal.jsonl", "meta.json")
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    d.prepare(source, prepared, mode="v3camara")
    evidence(meta)
    with endpoint("camara", meta) as (base, calls):
        result = d.run_pairs(
            prepared,
            source,
            journal,
            base,
            CHECKPOINT["id"],
            checkpoint=CHECKPOINT,
            server_evidence_file=meta,
        )
    assert result["complete"] and result["completed_attempts"] == 599
    assert [r["state"] for r in calls] == [r["text"] for r in rows]
    frozen = d.load_prepared(prepared)["tasks"]["camara_tema"]
    assert all(len(r["questions"]["q"]["criteria"]) == 32 for r in calls)
    assert [list(r["questions"]["q"]["criteria"]) for r in calls] == [
        r["question_order"] for r in frozen
    ]


def test_payload_error_or_truncation_never_becomes_valid_choice() -> None:
    q = {"criteria": {"a": "A", "b": "B"}}
    assert d._prediction({"answers": {"q": {"probabilities": {"a": True, "b": 0.0}}}}, q) is None
    assert (
        d._prediction({"answers": {"q": {"probabilities": {"a": float("nan"), "b": 0.0}}}}, q)
        is None
    )


def test_original_report_must_be_complete(paired: PairedFixture) -> None:
    _, rows, _, _, _, _ = paired
    with pytest.raises(ValueError):
        d._report_predictions({"items": {"juristcu_area": [["c0", "c0"]]}}, rows["juristcu_area"])


def test_wrong_live_server_binding_is_rejected_before_evaluation(tmp_path: Path) -> None:
    meta = tmp_path / "evidence.json"
    evidence(meta)
    expected = json.loads(meta.read_text())
    with endpoint("camara", meta, wrong_binding=True) as (base, calls):
        with pytest.raises(ValueError):
            d.preflight(base, expected)
        assert not calls


def test_binding_accepts_actual_legacy_bridge_health_schema() -> None:
    bridge = _load_research_module("llm_bridge")

    config = bridge.Config(
        CHECKPOINT["id"],
        CHECKPOINT["revision"],
        "cpu",
        16384,
        2,
        {},
        dtype="bfloat16",
        readout="constrained-letter",
    )
    backend = {"runtime": "transformers", "dtype": "bfloat16", "python": "fixture"}
    live = bridge.build_evidence(config, backend, "a" * 64)
    assert "runtime" not in live and "dtype" not in live
    expected = {
        "checkpoint": CHECKPOINT,
        "backend": live["backend"],
        "runtime": live["backend"],
        "dtype": live["backend"]["dtype"],
        "context": live["limits"]["context"],
        "prompt": live["prompt"],
        "server": {
            "health_evidence": live,
            "health_evidence_sha256": d.sha(d.canonical(live) + b"\n"),
        },
    }
    d._live_llm_evidence(
        {"evidence": live, "evidence_sha256": d.sha(d.canonical(live) + b"\n")}, expected
    )
