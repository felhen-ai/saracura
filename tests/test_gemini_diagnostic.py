"""Synthetic, offline tests for the refusal-aware Gemini diagnostic."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def mod() -> Any:
    root = Path(__file__).parents[1] / "research"
    sys.path.insert(0, str(root))
    path = root / "gemini_diagnostic.py"
    spec = importlib.util.spec_from_file_location("gemini_diagnostic", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["gemini_diagnostic"] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop("gemini_diagnostic", None)
    sys.path.remove(str(root))


def item(mod: Any, number: int = 0, *, task: str = "factck_veracidade", gold: str = "a") -> Any:
    question = {
        "type": "choice",
        "instructions": "instrução",
        "criteria": {"a": "A", "b": "B"},
    }
    payload, _parsed = mod._provider_payload("texto", question)
    return mod.PlanItem(
        f"{task}:{number}",
        task,
        number,
        "texto",
        question,
        gold,
        "a" * 64,
        payload,
        mod._sha(payload),
        "b" * 64,
    )


def success(mod: Any, winner: int = 0, identity: str | None = "gemini-decision-1@v1") -> Any:
    body: dict[str, Any] = {
        "answers": {
            "q": {
                "type": "choice",
                "choice": f"{winner:02d}",
                "probabilities": {
                    "00": 1.0 if winner == 0 else 0.0,
                    "01": 1.0 if winner == 1 else 0.0,
                },
                "confidence": 1.0,
            }
        },
        "usage": {"total_input_tokens": 8, "total_output_tokens": 1, "total_tokens": 9},
    }
    if identity is not None:
        body["model"] = identity
    return mod.gemini_bridge.TransportResponse(
        200,
        json.dumps(body).encode(),
    )


def test_refusal_is_closed_set_only(mod: Any) -> None:
    assert mod.refusal_code(400, b'{"error":{"code":"safety","message":"private"}}') == "safety"
    for body in (
        b'{"error":{"code":"SAFETY"}}',
        b'{"error":{"code":400}}',
        b'{"error":{}}',
        b"not-json",
        b"x" * (mod.MAX_ERROR_BODY + 1),
    ):
        assert mod.refusal_code(400, body) is None
    assert mod.refusal_code(401, b'{"error":{"code":"safety"}}') is None


def test_refusal_continues_and_is_wrong_in_denominator(mod: Any, tmp_path: Path) -> None:
    plan = [item(mod, 0), item(mod, 1, gold="b")]
    replies = iter(
        [
            mod.gemini_bridge.TransportResponse(
                400, b'{"error":{"code":"content_blocked","message":"never store"}}'
            ),
            success(mod, 1),
        ]
    )
    journal = mod.AppendJournal(tmp_path / "journal.jsonl")
    try:
        rows = mod.run_plan(plan, journal, lambda _payload, _timeout: next(replies))
    finally:
        journal.close()
    assert [row["outcome"] for row in rows] == ["refusal", "answered"]
    result = mod.report(plan, rows)["tasks"]["factck_veracidade"]
    assert result["refused"] == 1
    assert result["correct"] == 1
    assert result["balanced_accuracy"] == 0.5
    assert "message" not in (tmp_path / "journal.jsonl").read_text()


def test_run_identity_binds_fresh_rows_refusals_and_operational_evidence(
    mod: Any, tmp_path: Path
) -> None:
    run_id = "9b04ec5f-ddcd-43d1-ab7e-237d7ef2b6e5"
    plan = [item(mod), item(mod, 1, gold="b")]
    journal = mod.AppendJournal(tmp_path / "refusal.jsonl")
    try:
        rows = mod.run_plan(
            plan,
            journal,
            lambda _payload, _timeout: mod.gemini_bridge.TransportResponse(
                400, b'{"error":{"code":"safety"}}'
            ),
            run_id=run_id,
        )
    finally:
        journal.close()
    assert rows[0]["run_id"] == run_id
    assert mod.report(plan, rows, run_id=run_id)["run_id"] == run_id

    journal = mod.AppendJournal(tmp_path / "operational.jsonl")
    try:
        with pytest.raises(mod.DiagnosticError, match="operational_http"):
            mod.run_plan(
                plan,
                journal,
                lambda _payload, _timeout: mod.gemini_bridge.TransportResponse(401, b"{}"),
                run_id=run_id,
            )
    finally:
        journal.close()
    assert json.loads((tmp_path / "operational.jsonl").read_text())["run_id"] == run_id


@pytest.mark.parametrize(
    "reply",
    [
        lambda mod: mod.gemini_bridge.TransportResponse(401, b"{}"),
        lambda mod: mod.gemini_bridge.TransportResponse(429, b"{}"),
        lambda mod: mod.gemini_bridge.TransportResponse(200, b"{}"),
    ],
)
def test_non_refusal_failures_stop(mod: Any, tmp_path: Path, reply: Any) -> None:
    journal = mod.AppendJournal(tmp_path / "journal.jsonl")
    with pytest.raises(mod.DiagnosticError):
        mod.run_plan([item(mod)], journal, lambda _payload, _timeout: reply(mod))
    journal.close()
    row = json.loads((tmp_path / "journal.jsonl").read_text())
    assert row["outcome"] == "operational"
    assert "message" not in row


def test_exact_provider_bytes_and_legacy_wire_default_spacing(mod: Any) -> None:
    question = {"type": "choice", "instructions": "ç", "criteria": {"b": "β", "a": "á"}}
    payload, parsed = mod._provider_payload("texto", question)
    translated, _reverse = mod.gemini_bridge.translate(parsed)
    expected = json.dumps(
        translated, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode()
    assert payload == expected
    wire = mod._legacy_wire("texto", question)
    assert (
        wire
        == json.dumps({"model": mod.MODEL, "state": "texto", "questions": {"q": question}}).encode()
    )
    assert b", " in wire and b"\\u00e7" in wire


def test_resume_rejects_duplicate_order_hash_and_refusal_binding(mod: Any) -> None:
    plan = [item(mod, 0), item(mod, 1)]
    manifest = {"plan_sha256": mod._sha([mod._plan_row(value) for value in plan])}
    row = mod._safe_record(
        plan[0], outcome="answered", prediction="a", usage="unknown", latency_ms=0, identity=None
    )
    assert mod.validate_resume(plan, manifest, [row]) == {plan[0].item_id}
    with pytest.raises(mod.DiagnosticError):
        mod.validate_resume(plan, manifest, [row, row])
    refusal = mod._safe_record(
        plan[0],
        outcome="refusal",
        prediction="a",
        usage="unknown",
        latency_ms=0,
        identity=None,
        reason="safety",
    )
    with pytest.raises(mod.DiagnosticError):
        mod.validate_resume(plan, manifest, [refusal])
    mutated = dict(row, provider_payload_sha256="0" * 64)
    with pytest.raises(mod.DiagnosticError):
        mod.validate_resume(plan, manifest, [mutated])


def test_execution_identity_resume_rejects_missing_and_mixed_rows(mod: Any) -> None:
    plan = [item(mod, 0), item(mod, 1)]
    run_id = "1fcdb369-6e9c-4165-bbb3-4c93f45983ba"
    manifest = mod._execution_manifest(
        {"plan_sha256": mod._sha([mod._plan_row(value) for value in plan])}, run_id
    )
    row = mod._safe_record(
        plan[0],
        outcome="answered",
        prediction="a",
        usage="unknown",
        latency_ms=0,
        identity=None,
        run_id=run_id,
    )
    assert mod.validate_resume(plan, manifest, [row], require_run_id=True) == {plan[0].item_id}
    with pytest.raises(mod.DiagnosticError, match="resume_run_id_mismatch"):
        mod.validate_resume(plan, manifest, [dict(row, run_id="unknown")], require_run_id=True)
    with pytest.raises(mod.DiagnosticError, match="manifest_run_id_missing"):
        mod.validate_resume(
            plan, {"plan_sha256": manifest["plan_sha256"]}, [row], require_run_id=True
        )


def test_run_plan_retains_prior_identity_and_rebinds_imported_rows(
    mod: Any, tmp_path: Path
) -> None:
    run_id = "9d0d00e6-ee3b-4f99-9953-a7b9f6756166"
    plan = [item(mod, 0), item(mod, 1)]
    prior = [
        mod._safe_record(
            plan[0],
            outcome="answered",
            prediction="a",
            usage="unknown",
            latency_ms=0,
            identity=None,
            run_id=run_id,
        )
    ]
    reused = [
        mod._safe_record(
            plan[1],
            outcome="answered",
            prediction="a",
            usage="unknown",
            latency_ms=0,
            identity=None,
            provenance="legacy-import",
        )
        | {"import_provenance": {"timestamp_utc": "2026-01-01T00:00:00+00:00"}}
    ]
    journal = mod.AppendJournal(tmp_path / "journal.jsonl")
    try:
        rows = mod.run_plan(
            plan, journal, lambda *_args: pytest.fail("called"), prior=prior, reused=reused
        )
    finally:
        journal.close()
    assert {row["run_id"] for row in rows} == {run_id}
    mismatch_journal = mod.AppendJournal(tmp_path / "mismatch.jsonl")
    with pytest.raises(mod.DiagnosticError, match="prior_run_id_mismatch"):
        mod.run_plan(
            plan,
            mismatch_journal,
            lambda *_args: success(mod),
            prior=prior,
            run_id="53b1a34d-dda9-421d-8d5a-3f61e8724560",
        )
    mismatch_journal.close()


def test_cli_run_binds_manifest_journal_and_report_identity(
    mod: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = [item(mod), item(mod, 1, gold="b")]
    manifest = {"plan_sha256": mod._sha([mod._plan_row(value) for value in plan])}
    run_id = "5ce6ed34-52f2-4bd1-b5a6-ae4b3bdc7268"
    monkeypatch.setattr(mod, "build_plan", lambda *_args: (plan, manifest))
    monkeypatch.setattr(mod.uuid, "uuid4", lambda: mod.uuid.UUID(run_id))
    monkeypatch.setattr(
        mod.gemini_bridge, "https_transport", lambda _key: lambda *_args: success(mod)
    )
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    arguments = [
        "run",
        "--legacy-cache",
        "cache",
        "--legacy-revision",
        "revision",
        "--camara-dir",
        "camara",
        "--output",
        str(tmp_path),
    ]
    assert mod.main(arguments) == 0
    assert json.loads((tmp_path / "manifest.json").read_text())["run_id"] == run_id
    journal_rows = [
        json.loads(line) for line in (tmp_path / "journal.jsonl").read_text().splitlines()
    ]
    assert {row["run_id"] for row in journal_rows} == {run_id}
    assert json.loads((tmp_path / "report.json").read_text())["run_id"] == run_id


def test_cli_resume_preserves_identity_without_repeating_completed_call(
    mod: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = [item(mod), item(mod, 1, gold="b")]
    plan_manifest = {"plan_sha256": mod._sha([mod._plan_row(value) for value in plan])}
    run_id = "78e3806a-8a3f-4f22-983b-35b5bf9cfe34"
    (tmp_path / "manifest.json").write_text(
        json.dumps(mod._execution_manifest(plan_manifest, run_id)), encoding="utf-8"
    )
    completed = mod._safe_record(
        plan[0],
        outcome="answered",
        prediction="a",
        usage="unknown",
        latency_ms=0,
        identity=None,
        run_id=run_id,
    )
    (tmp_path / "journal.jsonl").write_text(json.dumps(completed) + "\n", encoding="utf-8")
    monkeypatch.setattr(mod, "build_plan", lambda *_args: (plan, plan_manifest))
    called: list[bytes] = []

    def transport(payload: bytes, _timeout: float) -> Any:
        called.append(payload)
        return success(mod, winner=1)

    monkeypatch.setattr(mod.gemini_bridge, "https_transport", lambda _key: transport)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    assert (
        mod.main(
            [
                "resume",
                "--legacy-cache",
                "cache",
                "--legacy-revision",
                "revision",
                "--camara-dir",
                "camara",
                "--output",
                str(tmp_path),
            ]
        )
        == 0
    )
    rows = [json.loads(line) for line in (tmp_path / "journal.jsonl").read_text().splitlines()]
    assert len(called) == 1
    assert {row["run_id"] for row in rows} == {run_id}


def test_cli_report_requires_execution_identity_and_rejects_tampering(
    mod: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = [item(mod)]
    manifest = {"plan_sha256": mod._sha([mod._plan_row(value) for value in plan])}
    monkeypatch.setattr(mod, "build_plan", lambda *_args: (plan, manifest))
    arguments = [
        "report",
        "--legacy-cache",
        "cache",
        "--legacy-revision",
        "revision",
        "--camara-dir",
        "camara",
        "--output",
        str(tmp_path),
    ]
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "journal.jsonl").write_text("", encoding="utf-8")
    assert mod.main(arguments) == 2

    run_id = "2e8f0956-929b-4ed9-8fd7-9ebc0e45a450"
    (tmp_path / "manifest.json").write_text(
        json.dumps(mod._execution_manifest(manifest, run_id)), encoding="utf-8"
    )
    tampered = mod._safe_record(
        plan[0], outcome="answered", prediction="a", usage="unknown", latency_ms=0, identity=None
    )
    (tmp_path / "journal.jsonl").write_text(json.dumps(tampered) + "\n", encoding="utf-8")
    assert mod.main(arguments) == 2


def test_pacer_preflight_and_no_repeated_completed_calls(mod: Any, tmp_path: Path) -> None:
    oversized = item(mod)
    oversized = mod.PlanItem(
        oversized.item_id,
        oversized.task,
        oversized.index,
        oversized.text,
        oversized.question,
        oversized.gold,
        oversized.source_wire_sha256,
        b"x" * mod.TOKEN_LIMIT,
        "c" * 64,
        oversized.source_sha256,
    )
    with pytest.raises(mod.DiagnosticError, match="unrunnable_quota"):
        mod.Pacer().admit(oversized.reservation)
    plan = [item(mod, 0), item(mod, 1)]
    first = mod._safe_record(
        plan[0], outcome="answered", prediction="a", usage="unknown", latency_ms=0, identity=None
    )
    called: list[bytes] = []

    def transport(payload: bytes, _timeout: float) -> Any:
        called.append(payload)
        return success(mod)

    journal = mod.AppendJournal(tmp_path / "journal.jsonl")
    try:
        records = mod.run_plan(
            plan,
            journal,
            transport,
            prior=[first],
        )
    finally:
        journal.close()
    assert len(called) == 1 and len(records) == 2


def test_resume_retains_prior_model_identity_pin(mod: Any, tmp_path: Path) -> None:
    plan = [item(mod)]
    prior = [
        mod._safe_record(
            plan[0],
            outcome="answered",
            prediction="a",
            usage="unknown",
            latency_ms=0,
            identity="gemini-decision-1@old",
        )
    ]
    journal = mod.AppendJournal(tmp_path / "journal.jsonl")
    try:
        with pytest.raises(mod.DiagnosticError, match="model_drift"):
            mod.run_plan(
                [item(mod, 1)],
                journal,
                lambda _payload, _timeout: success(mod, identity="gemini-decision-1@new"),
                prior=prior,
            )
    finally:
        journal.close()
    record = json.loads((tmp_path / "journal.jsonl").read_text())
    assert record["outcome"] == "operational"
    assert record["refusal_code"] == "model_drift"


def test_resume_model_identity_accepts_same_and_unknown(mod: Any, tmp_path: Path) -> None:
    plan = [item(mod)]
    same = mod._safe_record(
        item(mod, 99),
        outcome="answered",
        prediction="a",
        usage="unknown",
        latency_ms=0,
        identity="gemini-decision-1@v1",
    )
    journal = mod.AppendJournal(tmp_path / "same.jsonl")
    try:
        records = mod.run_plan(plan, journal, lambda _payload, _timeout: success(mod), prior=[same])
        assert records[-1]["model_identity"] == "gemini-decision-1@v1"
    finally:
        journal.close()
    unknown = mod._safe_record(
        item(mod, 98),
        outcome="answered",
        prediction="a",
        usage="unknown",
        latency_ms=0,
        identity=None,
    )
    journal = mod.AppendJournal(tmp_path / "unknown.jsonl")
    try:
        assert (
            mod.run_plan(
                plan,
                journal,
                lambda _payload, _timeout: success(mod, identity=None),
                prior=[unknown],
            )[-1]["model_identity"]
            == "unknown"
        )
    finally:
        journal.close()


def test_resume_rejects_multiple_known_model_identities_before_transport(
    mod: Any, tmp_path: Path
) -> None:
    plan = [item(mod)]
    prior = [
        mod._safe_record(
            plan[0],
            outcome="answered",
            prediction="a",
            usage="unknown",
            latency_ms=0,
            identity="gemini-decision-1@one",
        ),
        mod._safe_record(
            item(mod, 1),
            outcome="answered",
            prediction="a",
            usage="unknown",
            latency_ms=0,
            identity="gemini-decision-1@two",
        ),
    ]
    journal = mod.AppendJournal(tmp_path / "journal.jsonl")
    try:
        with pytest.raises(mod.DiagnosticError, match="prior_model_drift"):
            mod.run_plan(
                plan, journal, lambda _payload, _timeout: pytest.fail("called"), prior=prior
            )
    finally:
        journal.close()


def test_pacer_waits_through_multiple_expirations(mod: Any) -> None:
    now = [0.0]
    waits: list[float] = []

    def sleep(delay: float) -> None:
        waits.append(delay)
        now[0] += delay

    pacer = mod.Pacer(lambda: now[0], sleep)
    pacer.admit(16_000)
    now[0] = 1
    pacer.admit(16_000)
    now[0] = 2
    pacer.admit(16_000)
    now[0] = 3
    pacer.admit(30_000)
    assert len(waits) >= 2
    assert sum(tokens for _at, tokens in pacer.window) <= mod.TOKEN_LIMIT


def test_import_seed_is_fifo_compatible_only_and_reusable(mod: Any) -> None:
    plan = [item(mod, 0), item(mod, 1)]
    report = {
        "revision": "a" * 40,
        "run": {"mode": "full", "partial": False},
        "timing": {
            "started_at": "2026-01-01T00:00:00+00:00",
            "finished_at": "2026-01-01T01:00:00+00:00",
        },
        "dataset_files": {"factck_veracidade": {"sha256": plan[0].source_sha256}},
        "items": {"factck_veracidade": [["a", "a"], ["a", "a"]]},
    }
    rows = [
        {
            "timestamp_utc": "2025-12-31T23:00:00+00:00",  # smoke/outside range
            "status": "ok",
            "original_request_sha256": plan[0].source_wire_sha256,
            "translated_request_sha256": plan[0].provider_payload_sha256,
            "answer": [{"winner_position": 0}],
        },
        *[
            {
                "timestamp_utc": "2026-01-01T00:30:00+00:00",
                "status": "ok",
                "original_request_sha256": entry.source_wire_sha256,
                "translated_request_sha256": entry.provider_payload_sha256,
                "upstream_response_sha256": "d" * 64,
                "answer": [{"winner_position": 0}],
            }
            for entry in plan
        ],
    ]
    seed = mod.import_legacy_seed(plan, report, rows, "a" * 40)
    assert [record["id"] for record in seed["records"]] == [entry.item_id for entry in plan]
    imported = mod._seed_records(plan, seed)
    assert len(imported) == 2 and all(row["provenance"] == "legacy-import" for row in imported)
    seed["records"][0]["prediction"] = "bad"
    with pytest.raises(mod.DiagnosticError):
        mod._seed_records(plan, seed)


def test_incomplete_report_does_not_score_unattempted_as_wrong(mod: Any) -> None:
    plan = [item(mod, 0), item(mod, 1)]
    one = mod._safe_record(
        plan[0], outcome="answered", prediction="a", usage="unknown", latency_ms=0, identity=None
    )
    result = mod.report(plan, [one])
    task = result["tasks"]["factck_veracidade"]
    assert task["correct"] == 1
    assert task["balanced_accuracy"] is None
    assert result["headline_refusal_inclusive_normalized_mean"] is None


def test_report_counts_terminal_operational_as_new_attempt(mod: Any) -> None:
    plan = [item(mod)]
    operational = mod._safe_record(
        plan[0],
        outcome="operational",
        prediction=None,
        usage="unknown",
        latency_ms=0,
        identity=None,
        reason="operational_http",
    )
    result = mod.report(plan, [operational])
    assert {
        key: result[key]
        for key in ("processed", "attempted", "new", "reused", "resolved", "operational")
    } == {
        "processed": 1,
        "attempted": 1,
        "new": 1,
        "reused": 0,
        "resolved": 0,
        "operational": 1,
    }
    assert result["headline_refusal_inclusive_normalized_mean"] is None


def test_report_partitions_imported_and_operational_records(mod: Any) -> None:
    plan = [item(mod, 0), item(mod, 1)]
    imported = mod._safe_record(
        plan[0],
        outcome="answered",
        prediction="a",
        usage="unknown",
        latency_ms=0,
        identity=None,
        provenance="legacy-import",
    )
    operational = mod._safe_record(
        plan[1],
        outcome="operational",
        prediction=None,
        usage="unknown",
        latency_ms=0,
        identity=None,
        reason="operational_http",
    )
    result = mod.report(plan, [imported, operational])
    assert {
        key: result[key]
        for key in ("processed", "attempted", "new", "reused", "resolved", "operational")
    } == {
        "processed": 2,
        "attempted": 1,
        "new": 1,
        "reused": 1,
        "resolved": 1,
        "operational": 1,
    }
    assert result["headline_refusal_inclusive_normalized_mean"] is None


def test_complete_seven_task_report_uses_normalized_points(mod: Any) -> None:
    plan = []
    for index, task in enumerate(mod.ptbr_bench.TASKS):
        if task == "camara_tema":
            criteria = {f"label-{number}": str(number) for number in range(32)}
            question = {"type": "choice", "instructions": "i", "criteria": criteria}
            payload, _parsed = mod._provider_payload("t", question)
            for position, label in enumerate(("label-0", "label-1")):
                plan.append(
                    mod.PlanItem(
                        f"{task}:{position}",
                        task,
                        position,
                        "t",
                        question,
                        [label],
                        "x" * 64,
                        payload,
                        mod._sha(payload),
                        "y" * 64,
                    )
                )
        else:
            plan.extend(
                (item(mod, index * 2, task=task), item(mod, index * 2 + 1, task=task, gold="b"))
            )
    rows = [
        mod._safe_record(
            entry,
            outcome="answered",
            prediction=(entry.gold[0] if entry.task == "camara_tema" else entry.gold),
            usage="unknown",
            latency_ms=0,
            identity=None,
        )
        for entry in plan
    ]
    result = mod.report(plan, rows)
    assert result["complete"] is True
    assert result["headline_refusal_inclusive_normalized_mean"] == pytest.approx(100)
    assert result["attempted"] + result["reused"] == result["planned"]
    assert result["answered"] + result["refused"] == result["resolved"] == result["planned"]


def test_camara_score_matches_official_without_refusals_and_penalizes_null(mod: Any) -> None:
    criteria = {f"label-{index}": f"Label {index}" for index in range(32)}
    rows = []
    official_rows = []
    for index, (_label, accepted) in enumerate(
        (("label-0", ["label-0"]), ("label-1", ["label-1"]))
    ):
        question = {"type": "choice", "instructions": "i", "criteria": criteria}
        payload, _parsed = mod._provider_payload("t", question)
        rows.append(
            mod.PlanItem(
                f"camara_tema:{index}",
                "camara_tema",
                index,
                "t",
                question,
                accepted,
                "x" * 64,
                payload,
                mod._sha(payload),
                "y" * 64,
            )
        )
        official_rows.append(
            {"id": f"camara_tema:{index}", "criteria": criteria, "accepted_labels": accepted}
        )
    predictions = {row.item_id: row.gold[0] for row in rows}
    score = mod._camara_score(rows, predictions)
    expected = sys.modules["camara_v3"].score_predictions(official_rows, predictions)
    assert score["macro_valid_topic"] == expected["macro_valid_topic"]
    assert score["uniform_random_chance"] == expected["uniform_random_chance"]
    assert score["normalized"] == expected["normalized"]
    assert (
        mod._camara_score(rows, {rows[0].item_id: "label-0", rows[1].item_id: None})[
            "balanced_accuracy"
        ]
        < score["balanced_accuracy"]
    )
