"""Offline operational checks for the Phase 4E command boundary."""

from __future__ import annotations

import json
import socket
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from benchmarks import phase4e_pipeline as pipeline
from benchmarks import saracura_universal_corpus as corpus
from benchmarks import saracura_universal_training as training


@pytest.fixture
def bypass_aggregate_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep single-request transport tests focused below the plan-wide gate."""
    monkeypatch.setattr(pipeline, "_aggregate_preflight", lambda *args: {})


def test_plan_is_no_clobber_and_constructs_no_socket(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        socket,
        "socket",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("socket attempted")),
    )
    output = tmp_path / "plan.json"
    assert pipeline.main(["plan", "--output", str(output)]) == 0
    corpus.validate_plan(json.loads(output.read_bytes()))
    assert pipeline.main(["plan", "--output", str(output)]) == 2


def test_corpus_requires_literal_network_opt_in_before_environment_or_io(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    secret = "phase4e-secret-must-not-appear"
    monkeypatch.setenv("OPENROUTER_API_KEY", secret)
    assert (
        pipeline.main(
            [
                "corpus",
                "--plan",
                str(tmp_path / "missing-plan.json"),
                "--snapshot",
                str(tmp_path / "snapshot"),
                "--work-dir",
                str(tmp_path / "work"),
                "--packet",
                str(tmp_path / "packet"),
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert "literal --allow-network" in captured.err
    assert secret not in captured.err + captured.out
    assert not (tmp_path / "work").exists()


def test_fake_transport_has_pinned_shape_and_does_not_store_secret(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bypass_aggregate_preflight: None
) -> None:
    plan = tmp_path / "plan.json"
    pipeline.write_plan(plan)
    monkeypatch.setenv("OPENROUTER_API_KEY", "phase4e-test-secret")
    monkeypatch.setattr(
        corpus.VerifiedMiniLMTokenizerReceipt,
        "create",
        classmethod(lambda cls, snapshot: _Counter()),
    )
    captured: dict[str, Any] = {}

    def fake(
        method: str, url: str, headers: object, body: bytes
    ) -> tuple[int, dict[str, str], bytes]:
        captured.update(method=method, url=url, headers=headers, body=json.loads(body))
        return 200, {}, b'{"id":"fake","usage":{"cost":0},"choices":[{"message":{"content":"{}"}}]}'

    with pytest.raises(corpus.CorpusError, match="author record count"):
        pipeline.run_corpus(
            plan,
            tmp_path / "unused-snapshot",
            tmp_path / "work",
            tmp_path / "packet",
            allow_network=True,
            transport=fake,
        )
    assert captured["method"] == "POST"
    assert captured["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert captured["body"]["model"] == corpus.AUTHOR_MODEL
    assert captured["body"]["provider"] == corpus.provider_preferences("corpus_author")
    assert captured["body"]["reasoning_effort"] == "none"
    assert captured["body"]["response_format"]["json_schema"]["name"] == corpus.RESPONSE_SCHEMA_NAME
    assert "phase4e-test-secret" not in json.dumps(captured["body"])
    assert (
        "phase4e-test-secret" not in (tmp_path / "work" / "ledger" / "ledger-0000.json").read_text()
    )


def test_resume_refuses_unresolved_pre_send_reservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = tmp_path / "plan.json"
    pipeline.write_plan(plan)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    ledger = corpus.BudgetLedger()
    ledger.reserve_request("corpus_author", "reservation-" + "a" * 64, Decimal("0.01"))
    corpus.write_ledger_snapshot(tmp_path / "work" / "ledger", ledger)
    called = False

    def fake(*args: object, **kwargs: object) -> tuple[int, dict[str, str], bytes]:
        nonlocal called
        del args, kwargs
        called = True
        return 500, {}, b"{}"

    with pytest.raises(corpus.CorpusError, match="unresolved pre-send reservation"):
        pipeline.run_corpus(
            plan,
            tmp_path / "unused-snapshot",
            tmp_path / "work",
            tmp_path / "packet",
            allow_network=True,
            transport=fake,
        )
    assert not called


def test_settled_invalid_author_response_is_atomically_resolved_without_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bypass_aggregate_preflight: None
) -> None:
    plan = tmp_path / "plan.json"
    pipeline.write_plan(plan)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    slot = next(slot for slot in corpus.build_plan()["slots"] if slot["pair_id"] is None)
    monkeypatch.setattr(pipeline, "_author_batches", lambda _plan: [[slot]])
    monkeypatch.setattr(
        corpus.VerifiedMiniLMTokenizerReceipt,
        "create",
        classmethod(lambda cls, snapshot: _Counter()),
    )

    def invalid_author(
        method: str, url: str, headers: object, body: bytes
    ) -> tuple[int, dict[str, str], bytes]:
        del method, url, headers, body
        return (
            200,
            {},
            b'{"id":"author-invalid","usage":{"cost":0},"choices":[{"message":{"content":"{}"}}]}',
        )

    with pytest.raises(corpus.CorpusError, match="author record count"):
        pipeline.run_corpus(
            plan,
            tmp_path / "snapshot",
            tmp_path / "work",
            tmp_path / "packet",
            allow_network=True,
            transport=invalid_author,
        )
    ledger = corpus.resume_ledger(tmp_path / "work" / "ledger")
    assert ledger.provider_journal == [
        {
            "stage": "corpus_author",
            "reservation_id": ledger.entries[0]["reservation_id"],
            "request_id": "author-invalid",
            "task_ids": [slot["task_id"]],
            "response_sha256": ledger.entries[0]["response_sha256"],
        }
    ]

    called = False

    def must_not_retry(*args: object, **kwargs: object) -> tuple[int, dict[str, str], bytes]:
        nonlocal called
        del args, kwargs
        called = True
        return 500, {}, b"{}"

    with pytest.raises(corpus.CorpusError, match="incomplete corpus resolution"):
        pipeline.run_corpus(
            plan,
            tmp_path / "snapshot",
            tmp_path / "work",
            tmp_path / "packet",
            allow_network=True,
            transport=must_not_retry,
        )
    assert not called
    records = list((tmp_path / "work" / "resolved").glob("call-*.json"))
    assert len(records) == 1
    fallback = json.loads(records[0].read_bytes())
    assert fallback["kind"] == "settled_fallback"
    assert fallback["resolutions"] == [
        {
            "task_id": slot["task_id"],
            "split": slot["split"],
            "reason": "author_validation_failure",
            **corpus.lineage_from_journal(ledger.provider_journal[0], "author"),
        }
    ]


def test_settled_author_overspend_is_atomically_resolved_without_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bypass_aggregate_preflight: None
) -> None:
    plan = tmp_path / "plan.json"
    pipeline.write_plan(plan)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    slot = next(slot for slot in corpus.build_plan()["slots"] if slot["pair_id"] is None)
    monkeypatch.setattr(pipeline, "_author_batches", lambda _plan: [[slot]])

    def overspent_author(
        method: str, url: str, headers: object, body: bytes
    ) -> tuple[int, dict[str, str], bytes]:
        del method, url, headers, body
        charged = corpus.STAGE_LIMITS["corpus_author"] + Decimal("0.01")
        return (
            200,
            {},
            json.dumps(
                {"id": "author-overspent", "usage": {"cost": str(charged)}, "choices": []}
            ).encode(),
        )

    with pytest.raises(corpus.CorpusError, match="reported stage budget exhausted"):
        pipeline.run_corpus(
            plan,
            tmp_path / "snapshot",
            tmp_path / "work",
            tmp_path / "packet",
            allow_network=True,
            transport=overspent_author,
        )

    ledger = corpus.resume_ledger(tmp_path / "work" / "ledger")
    assert ledger.entries[0]["status"] == "overspent"
    fallback = json.loads(next((tmp_path / "work" / "resolved").glob("call-*.json")).read_bytes())
    assert fallback["kind"] == "settled_fallback"
    assert fallback["resolutions"][0] == {
        "task_id": slot["task_id"],
        "split": slot["split"],
        "reason": "author_response_failure",
        **corpus.lineage_from_journal(ledger.provider_journal[0], "author"),
    }

    with pytest.raises(corpus.CorpusError, match="incomplete corpus resolution"):
        pipeline.run_corpus(
            plan,
            tmp_path / "snapshot",
            tmp_path / "work",
            tmp_path / "packet",
            allow_network=True,
            transport=lambda *args: (_ for _ in ()).throw(AssertionError("provider retry")),
        )


@pytest.mark.parametrize(
    ("failure", "reason"),
    (
        ("review_resolution", "review_resolution_failure"),
        ("normal_persistence", "result_persistence_failure"),
    ),
)
def test_post_reviewer_failure_atomically_resolves_all_settled_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bypass_aggregate_preflight: None,
    failure: str,
    reason: str,
) -> None:
    plan = tmp_path / "plan.json"
    pipeline.write_plan(plan)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    slot = next(slot for slot in corpus.build_plan()["slots"] if slot["pair_id"] is None)
    monkeypatch.setattr(pipeline, "_author_batches", lambda _plan: [[slot]])
    monkeypatch.setattr(
        corpus.VerifiedMiniLMTokenizerReceipt,
        "create",
        classmethod(lambda cls, snapshot: _Counter()),
    )
    calls = 0
    authored: dict[str, dict[str, Any]] = {}

    def fake(
        method: str, url: str, headers: object, body: bytes
    ) -> tuple[int, dict[str, str], bytes]:
        nonlocal calls
        del method, url, headers
        calls += 1
        request = json.loads(body)
        if calls == 1:
            records = [
                _record(item) for item in json.loads(request["messages"][1]["content"])["slots"]
            ]
            authored.update({record["task_id"]: record for record in records})
            content = json.dumps({"records": records})
            request_id = "author-valid"
        else:
            task = json.loads(request["messages"][1]["content"])["tasks"][0]
            source = authored[task["task_id"]]
            selected = source["selected_criterion_id"]
            content = json.dumps(
                {
                    "reviews": [
                        {
                            "task_id": task["task_id"],
                            "status": "accepted",
                            "selected_criterion_id": selected,
                            "reason_codes": [],
                            "natural_language": True,
                            "fictional": True,
                            "exclusive_options": True,
                            "private_or_sensitive": False,
                            "semantic_equivalence_attestation": source[
                                "semantic_equivalence_attestation"
                            ],
                        }
                    ]
                }
            )
            request_id = "reviewer-valid"
        return (
            200,
            {},
            json.dumps(
                {
                    "id": request_id,
                    "usage": {"cost": 0},
                    "choices": [{"message": {"content": content}}],
                }
            ).encode(),
        )

    if failure == "review_resolution":
        monkeypatch.setattr(
            corpus,
            "resolve_reviews",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                corpus.CorpusError("injected resolution failure")
            ),
        )
    else:
        monkeypatch.setattr(
            pipeline,
            "_store_call_resolution",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                corpus.CorpusError("injected persistence failure")
            ),
        )
    with pytest.raises(corpus.CorpusError, match=r"injected .* failure"):
        pipeline.run_corpus(
            plan,
            tmp_path / "snapshot",
            tmp_path / "work",
            tmp_path / "packet",
            allow_network=True,
            transport=fake,
        )
    ledger = corpus.resume_ledger(tmp_path / "work" / "ledger")
    assert [entry["stage"] for entry in ledger.provider_journal] == [
        "corpus_author",
        "corpus_reviewer",
    ]

    with pytest.raises(corpus.CorpusError, match="incomplete corpus resolution"):
        pipeline.run_corpus(
            plan,
            tmp_path / "snapshot",
            tmp_path / "work",
            tmp_path / "packet",
            allow_network=True,
            transport=lambda *args: (_ for _ in ()).throw(AssertionError("provider retry")),
        )
    fallback = json.loads(next((tmp_path / "work" / "resolved").glob("call-*.json")).read_bytes())
    assert fallback["kind"] == "settled_fallback"
    assert fallback["resolutions"][0]["reason"] == reason
    assert set(fallback["resolutions"][0]) == {
        "task_id",
        "split",
        "reason",
        "author_response_sha256",
        "author_reservation_id",
        "author_request_id",
        "reviewer_response_sha256",
        "reviewer_reservation_id",
        "reviewer_request_id",
    }


def test_settled_invalid_reviewer_response_is_atomically_resolved_without_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bypass_aggregate_preflight: None
) -> None:
    plan = tmp_path / "plan.json"
    pipeline.write_plan(plan)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    slot = next(slot for slot in corpus.build_plan()["slots"] if slot["pair_id"] is None)
    monkeypatch.setattr(pipeline, "_author_batches", lambda _plan: [[slot]])
    monkeypatch.setattr(
        corpus.VerifiedMiniLMTokenizerReceipt,
        "create",
        classmethod(lambda cls, snapshot: _Counter()),
    )
    calls = 0

    def invalid_reviewer(
        method: str, url: str, headers: object, body: bytes
    ) -> tuple[int, dict[str, str], bytes]:
        nonlocal calls
        del method, url, headers
        calls += 1
        content = (
            json.dumps(
                {
                    "records": [
                        _record(item)
                        for item in json.loads(json.loads(body)["messages"][1]["content"])["slots"]
                    ]
                }
            )
            if calls == 1
            else "{}"
        )
        return (
            200,
            {},
            json.dumps(
                {
                    "id": f"request-{calls}",
                    "usage": {"cost": 0},
                    "choices": [{"message": {"content": content}}],
                }
            ).encode(),
        )

    with pytest.raises(corpus.CorpusError, match="review response"):
        pipeline.run_corpus(
            plan,
            tmp_path / "snapshot",
            tmp_path / "work",
            tmp_path / "packet",
            allow_network=True,
            transport=invalid_reviewer,
        )
    assert [
        entry["stage"]
        for entry in corpus.resume_ledger(tmp_path / "work" / "ledger").provider_journal
    ] == ["corpus_author", "corpus_reviewer"]
    with pytest.raises(corpus.CorpusError, match="incomplete corpus resolution"):
        pipeline.run_corpus(
            plan,
            tmp_path / "snapshot",
            tmp_path / "work",
            tmp_path / "packet",
            allow_network=True,
            transport=lambda *args: (_ for _ in ()).throw(AssertionError("provider retry")),
        )
    fallback = json.loads(next((tmp_path / "work" / "resolved").glob("call-*.json")).read_bytes())
    assert fallback["kind"] == "settled_fallback"
    assert fallback["resolutions"][0]["reason"] == "reviewer_response_failure"
    assert {key for key in fallback["resolutions"][0] if key.startswith("reviewer_")} == {
        "reviewer_response_sha256",
        "reviewer_reservation_id",
        "reviewer_request_id",
    }


def test_capacity_resolution_uses_atomic_call_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bypass_aggregate_preflight: None
) -> None:
    plan = tmp_path / "plan.json"
    pipeline.write_plan(plan)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(
        corpus.VerifiedMiniLMTokenizerReceipt,
        "create",
        classmethod(lambda cls, snapshot: _Counter(129)),
    )
    calls = 0

    def fake(
        method: str, url: str, headers: object, body: bytes
    ) -> tuple[int, dict[str, str], bytes]:
        nonlocal calls
        del method, url, headers
        calls += 1
        slots = json.loads(json.loads(body)["messages"][1]["content"])["slots"]
        records = [_record(slot) for slot in slots]
        content = json.dumps({"records": records}) if calls == 1 else "{}"
        return (
            200,
            {},
            json.dumps(
                {
                    "id": f"fake-{calls}",
                    "usage": {"cost": 0},
                    "choices": [{"message": {"content": content}}],
                }
            ).encode(),
        )

    with pytest.raises(corpus.CorpusError, match="author record count"):
        pipeline.run_corpus(
            plan,
            tmp_path / "unused-snapshot",
            tmp_path / "work",
            tmp_path / "packet",
            allow_network=True,
            transport=fake,
        )
    assert len(list((tmp_path / "work" / "resolved").glob("call-*.json"))) == 2


def test_source_contract_resolution_is_durable_and_has_settled_author_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bypass_aggregate_preflight: None
) -> None:
    plan = tmp_path / "plan.json"
    pipeline.write_plan(plan)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(
        corpus.VerifiedMiniLMTokenizerReceipt,
        "create",
        classmethod(lambda cls, snapshot: _Counter()),
    )
    calls = 0

    def fake(
        method: str, url: str, headers: object, body: bytes
    ) -> tuple[int, dict[str, str], bytes]:
        nonlocal calls
        del method, url, headers
        calls += 1
        slots = json.loads(json.loads(body)["messages"][1]["content"])["slots"]
        records = []
        for slot in slots:
            authored = _record(slot)
            records.append(
                {
                    "instruction": authored["instruction"],
                    "state": {"summary": "x" * 181},
                    "criteria": [
                        {"description": criterion["description"]}
                        for criterion in authored["criteria"]
                    ],
                    "selected_index": slot["gold_position"],
                    "semantic_equivalence_attestation": authored[
                        "semantic_equivalence_attestation"
                    ],
                }
            )
        content = json.dumps({"records": records}) if calls == 1 else "{}"
        return (
            200,
            {},
            json.dumps(
                {
                    "id": f"source-contract-{calls}",
                    "usage": {"cost": 0},
                    "choices": [{"message": {"content": content}}],
                }
            ).encode(),
        )

    with pytest.raises(corpus.CorpusError, match="author record count"):
        pipeline.run_corpus(
            plan,
            tmp_path / "unused-snapshot",
            tmp_path / "work",
            tmp_path / "packet",
            allow_network=True,
            transport=fake,
        )
    resolved = [
        entry["row"]
        for path in (tmp_path / "work" / "resolved").glob("call-*.json")
        for entry in json.loads(path.read_bytes())["resolutions"]
        if "row" in entry
    ]
    assert len(resolved) == 2
    assert all(
        row["reason"] == "author_record_schema__state.summary:string_too_long" for row in resolved
    )
    assert all(row["author_response_sha256"] for row in resolved)
    assert all(row["author_reservation_id"] for row in resolved)
    assert all(row["author_request_id"] == "source-contract-1" for row in resolved)


def test_loader_expands_legacy_task_and_strict_call_records(tmp_path: Path) -> None:
    plan = corpus.build_plan()
    pair = next(batch for batch in pipeline._author_batches(plan) if len(batch) == 2)
    single = next(batch[0] for batch in pipeline._author_batches(plan) if len(batch) == 1)
    legacy_row = {
        "task_id": single["task_id"],
        "split": single["split"],
        "reason": "source_contract",
    }
    pipeline._store_resolution(tmp_path, legacy_row, "rejected")
    lineage = {
        "author_response_sha256": "a" * 64,
        "author_reservation_id": "reservation-" + "a" * 64,
        "author_request_id": "author-request",
    }
    fallback = {
        "schema_version": "phase4e-corpus-call-resolution.v1",
        "kind": "settled_fallback",
        "resolutions": [
            {
                "task_id": slot["task_id"],
                "split": slot["split"],
                "reason": "author_validation_failure",
                **lineage,
            }
            for slot in pair
        ],
    }
    call_path = pipeline._call_work_path(tmp_path, [slot["task_id"] for slot in pair])
    call_path.parent.mkdir(parents=True, exist_ok=True)
    call_path.write_bytes(pipeline._canonical(fallback) + b"\n")

    loaded = pipeline._load_resolved(tmp_path, plan)
    assert set(loaded) == {single["task_id"], *(slot["task_id"] for slot in pair)}
    assert all(loaded[slot["task_id"]]["status"] == "rejected" for slot in pair)


@pytest.mark.parametrize(
    "failure", ("duplicate", "unknown", "partial", "malformed", "content", "filename")
)
def test_loader_rejects_invalid_or_partial_call_resolution(tmp_path: Path, failure: str) -> None:
    plan = corpus.build_plan()
    pair = next(batch for batch in pipeline._author_batches(plan) if len(batch) == 2)
    lineage = {
        "author_response_sha256": "a" * 64,
        "author_reservation_id": "reservation-" + "a" * 64,
        "author_request_id": "author-request",
    }
    entries = [
        {
            "task_id": slot["task_id"],
            "split": slot["split"],
            "reason": "author_validation_failure",
            **lineage,
        }
        for slot in pair
    ]
    if failure == "unknown":
        entries[1]["task_id"] = "task-" + "0" * 64
    elif failure == "partial":
        entries.pop()
    elif failure == "content":
        entries[0]["raw_content"] = "must never persist"
    payload: dict[str, Any] = {
        "schema_version": "phase4e-corpus-call-resolution.v1",
        "kind": "settled_fallback",
        "resolutions": entries,
    }
    if failure == "malformed":
        payload["unexpected"] = True
    call_path = pipeline._call_work_path(tmp_path, [slot["task_id"] for slot in pair])
    if failure == "filename":
        call_path = call_path.with_name("call-" + "f" * 64 + ".json")
    call_path.parent.mkdir(parents=True, exist_ok=True)
    call_path.write_bytes(pipeline._canonical(payload) + b"\n")
    if failure == "duplicate":
        pipeline._store_resolution(
            tmp_path,
            {"task_id": pair[0]["task_id"], "split": pair[0]["split"], "reason": "legacy"},
            "rejected",
        )

    with pytest.raises(corpus.CorpusError, match="work record"):
        pipeline._load_resolved(tmp_path, plan)


def test_lineage_lookup_never_reuses_a_previous_settled_call() -> None:
    previous = {
        "stage": "corpus_author",
        "reservation_id": "reservation-" + "a" * 64,
        "request_id": "previous-request",
        "task_ids": ["task-" + "a" * 64],
        "response_sha256": "b" * 64,
    }

    class _Client:
        def last_journal(self, stage: str) -> dict[str, Any]:
            assert stage == "corpus_author"
            return previous

    assert (
        pipeline._lineage_or_none(
            _Client(),  # type: ignore[arg-type]
            "author",
            ["task-" + "c" * 64],
        )
        is None
    )


def test_conservative_preflight_refuses_stage_overage() -> None:
    with pytest.raises(corpus.CorpusError, match="stage budget exhausted"):
        pipeline._preflight_request("corpus_author", b"x", 10_000_000)


def test_aggregate_preflight_calculates_full_plan_and_stops_before_transport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = corpus.build_plan()
    totals = pipeline._aggregate_preflight(plan, {}, corpus.BudgetLedger())
    assert totals == {
        "corpus_author": Decimal("4.4026231"),
        "corpus_reviewer": Decimal("9.99871629"),
    }
    assert totals["corpus_author"] < Decimal("5")
    assert totals["corpus_reviewer"] < Decimal("10")

    plan_path = tmp_path / "plan.json"
    pipeline.write_plan(plan_path)
    monkeypatch.setattr(
        corpus,
        "STAGE_LIMITS",
        {"corpus_author": Decimal("4.00"), "corpus_reviewer": Decimal("10.00")},
    )
    monkeypatch.setattr(corpus, "TOTAL_BUDGET", Decimal("17.00"))
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    called = False

    def transport(*args: object, **kwargs: object) -> tuple[int, dict[str, str], bytes]:
        nonlocal called
        del args, kwargs
        called = True
        return 500, {}, b"{}"

    with pytest.raises(corpus.CorpusError, match="stage budget exhausted"):
        pipeline.run_corpus(
            plan_path,
            tmp_path / "snapshot",
            tmp_path / "work",
            tmp_path / "packet",
            allow_network=True,
            transport=transport,
        )
    assert not called


def test_aggregate_preflight_adds_ledger_spend_for_only_unresolved_plan() -> None:
    plan = corpus.build_plan()
    pending = next(slot for slot in plan["slots"] if slot["pair_id"] is None)
    resolved = {
        slot["task_id"]: {"status": "accepted"}
        for slot in plan["slots"]
        if slot["task_id"] != pending["task_id"]
    }
    ledger = corpus.BudgetLedger()
    ledger.record("corpus_author", "previous", Decimal("4.9999"), Decimal(), "a" * 64)
    with pytest.raises(corpus.CorpusError, match="stage budget exhausted"):
        pipeline._aggregate_preflight(plan, resolved, ledger)


def test_wilson_boundary_persists_full_projection_in_ledger_directory(tmp_path: Path) -> None:
    plan = corpus.build_plan()
    resolved = [{"task_id": slot["task_id"], "status": "accepted"} for slot in plan["slots"][:200]]
    path = pipeline._write_wilson_projection(tmp_path / "ledger", plan, resolved, None)
    payload = json.loads(path.read_bytes())
    assert payload["schema_version"] == "phase4e-wilson-ledger.v1"
    assert payload["resolved_tasks"] == 200
    assert {"global", "synthetic_train", "synthetic_dev", "synthetic_holdout"} <= set(
        payload["projections"]
    )
    assert len(payload["projections"]) == 46
    assert payload["projections"]["global"]["observations"] == 200
    assert payload["projections"]["global"]["wilson_eligible"] is True
    assert payload["projections"]["global"]["wilson_projected_accepted"] is not None


def test_wilson_ledger_marks_eight_observations_descriptive_only(tmp_path: Path) -> None:
    plan = corpus.build_plan()
    resolved = [{"task_id": slot["task_id"], "status": "accepted"} for slot in plan["slots"][:8]]
    payload = json.loads(
        pipeline._write_wilson_projection(tmp_path / "ledger", plan, resolved, None).read_bytes()
    )
    global_projection = payload["projections"]["global"]
    assert global_projection["observations"] == 8
    assert global_projection["wilson_eligible"] is False
    assert global_projection["wilson_projected_accepted"] is None


@pytest.mark.parametrize("failure", ("pre_holdout_binding", "training_verification"))
def test_verify_never_opens_holdout_before_a_claim_or_after_training_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    calls: list[str] = []

    def full_validator(_packet: Path) -> None:
        raise AssertionError("run_verify must not open the accepted holdout")

    monkeypatch.setattr(corpus, "validate_accepted_packet", full_validator)
    if failure == "pre_holdout_binding":
        monkeypatch.setattr(
            training,
            "validate_accepted_packet_binding",
            lambda _packet: (_ for _ in ()).throw(RuntimeError("no training claim")),
        )
        with pytest.raises(RuntimeError, match="no training claim"):
            pipeline.run_verify(tmp_path / "packet", tmp_path / "embeddings", tmp_path / "training")
        return

    binding = object()

    def bind_packet(_packet: Path) -> object:
        calls.append("binding")
        return binding

    def validate_embeddings(_embeddings: Path, received: object) -> None:
        if received is not binding:
            raise AssertionError("binding changed")
        calls.append("embeddings")

    monkeypatch.setattr(
        training,
        "validate_accepted_packet_binding",
        bind_packet,
    )
    monkeypatch.setattr(
        training,
        "validate_embedding_capsule",
        validate_embeddings,
    )
    monkeypatch.setattr(
        training,
        "verify_training_capsule",
        lambda _training: (_ for _ in ()).throw(RuntimeError("training verification failed")),
    )

    with pytest.raises(RuntimeError, match="training verification failed"):
        pipeline.run_verify(tmp_path / "packet", tmp_path / "embeddings", tmp_path / "training")
    assert calls == ["binding", "embeddings"]


@pytest.mark.parametrize("field", ("packet_receipt_sha256", "embedding_descriptor_sha256"))
def test_verify_rejects_cross_mixed_embedding_and_training_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    binding = object()
    capsule = SimpleNamespace(
        manifest={"packet_receipt_sha256": "a" * 64}, descriptor_sha256="b" * 64
    )
    manifest = {
        "packet_receipt_sha256": "a" * 64,
        "embedding_descriptor_sha256": "b" * 64,
    }
    manifest[field] = "c" * 64
    monkeypatch.setattr(training, "validate_accepted_packet_binding", lambda _: binding)
    monkeypatch.setattr(
        training,
        "validate_embedding_capsule",
        lambda _, received: (
            capsule if received is binding else pytest.fail("packet binding changed")
        ),
    )
    monkeypatch.setattr(training, "verify_training_capsule", lambda _: manifest)

    with pytest.raises(training.TrainingError, match="lineage binding"):
        pipeline.run_verify(tmp_path / "packet", tmp_path / "embeddings", tmp_path / "training")


def test_verify_accepts_matching_embedding_and_training_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binding = object()
    capsule = SimpleNamespace(
        manifest={"packet_receipt_sha256": "a" * 64}, descriptor_sha256="b" * 64
    )
    manifest = {
        "packet_receipt_sha256": "a" * 64,
        "embedding_descriptor_sha256": "b" * 64,
    }
    monkeypatch.setattr(training, "validate_accepted_packet_binding", lambda _: binding)
    monkeypatch.setattr(
        training,
        "validate_embedding_capsule",
        lambda _, received: (
            capsule if received is binding else pytest.fail("packet binding changed")
        ),
    )
    monkeypatch.setattr(training, "verify_training_capsule", lambda _: manifest)

    pipeline.run_verify(tmp_path / "packet", tmp_path / "embeddings", tmp_path / "training")


class _Counter:
    def __init__(self, count: int = 1) -> None:
        self._count = count

    def count(self, text: str) -> int:
        del text
        return self._count


def _record(slot: dict[str, Any]) -> dict[str, Any]:
    criteria = [
        {"id": f"route-{index}", "description": f"Fictional route {index}."}
        for index in range(slot["option_count"])
    ]
    target = slot["semantic_target"]
    roles = list(target["criterion_roles"])
    roles.insert(slot["gold_position"], roles.pop(0))
    semantic_attestation = {
        "scenario": target["scenario"],
        "criterion_roles": roles,
        "selected_role": "matches_rule",
    }
    record = {
        **{
            key: slot[key]
            for key in (
                "task_id",
                "family_id",
                "locale",
                "domain",
                "axes",
                "option_count",
                "gold_position",
            )
        },
        "instruction": "Choose the safe fictional route.",
        "state": {"summary": "Fictional only."},
        "criteria": criteria,
        "selected_criterion_id": criteria[slot["gold_position"]]["id"],
        "semantic_equivalence_attestation": semantic_attestation,
    }
    if slot["pair_id"] is not None:
        record["cross_locale_attestation"] = {
            "pair_id": slot["pair_id"],
            **semantic_attestation,
        }
    return record
