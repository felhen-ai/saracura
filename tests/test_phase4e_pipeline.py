"""Offline operational checks for the Phase 4E command boundary."""

from __future__ import annotations

import json
import socket
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from benchmarks import phase4e_pipeline as pipeline
from benchmarks import saracura_universal_corpus as corpus


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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
    assert captured["body"]["provider"] == {"data_collection": "deny", "zdr": True}
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


def test_resume_refuses_settled_author_response_after_author_validation_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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

    with pytest.raises(
        corpus.CorpusError, match="settled provider call lacks complete task resolution"
    ):
        pipeline.run_corpus(
            plan,
            tmp_path / "snapshot",
            tmp_path / "work",
            tmp_path / "packet",
            allow_network=True,
            transport=must_not_retry,
        )
    assert not called


def test_resume_refuses_settled_reviewer_response_after_validation_or_persistence_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
            position = next(
                index
                for index, criterion in enumerate(task["criteria"])
                if criterion["id"] == selected
            )
            roles = [f"role-{index}" for index in range(len(task["criteria"]))]
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
                            "semantic_equivalence_attestation": {
                                "scenario": "fictional scenario",
                                "criterion_roles": roles,
                                "selected_role": roles[position],
                            },
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

    monkeypatch.setattr(
        corpus,
        "resolve_reviews",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            corpus.CorpusError("injected persistence failure")
        ),
    )
    with pytest.raises(corpus.CorpusError, match="injected persistence failure"):
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

    with pytest.raises(
        corpus.CorpusError, match="settled provider call lacks complete task resolution"
    ):
        pipeline.run_corpus(
            plan,
            tmp_path / "snapshot",
            tmp_path / "work",
            tmp_path / "packet",
            allow_network=True,
            transport=lambda *args: (_ for _ in ()).throw(AssertionError("provider retry")),
        )


def test_resume_refuses_settled_reviewer_response_after_reviewer_validation_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
    with pytest.raises(
        corpus.CorpusError, match="settled provider call lacks complete task resolution"
    ):
        pipeline.run_corpus(
            plan,
            tmp_path / "snapshot",
            tmp_path / "work",
            tmp_path / "packet",
            allow_network=True,
            transport=lambda *args: (_ for _ in ()).throw(AssertionError("provider retry")),
        )


def test_capacity_resolution_uses_task_identity_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
    assert len(list((tmp_path / "work" / "resolved").glob("task-*.json"))) == 2


def test_conservative_preflight_refuses_stage_overage() -> None:
    with pytest.raises(corpus.CorpusError, match="stage budget exhausted"):
        pipeline._preflight_request("corpus_author", b"x", 10_000_000)


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
    }
    if slot["pair_id"] is not None:
        roles = [f"role-{index}" for index in range(slot["option_count"])]
        record["cross_locale_attestation"] = {
            "pair_id": slot["pair_id"],
            "scenario": "fictional shared scenario",
            "criterion_roles": roles,
            "selected_role": roles[slot["gold_position"]],
        }
    return record
