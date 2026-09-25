"""Offline completion checks for the Phase 4E post-pilot corpus lane."""

from __future__ import annotations

import json
from collections.abc import Mapping
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import pytest

from benchmarks import phase4e_pipeline as pipeline
from benchmarks import saracura_universal_corpus as corpus
from benchmarks import saracura_universal_pilot as pilot
from benchmarks import saracura_universal_training as training


class _Counter:
    def count(self, text: str) -> int:
        del text
        return 1


def _record(slot: Mapping[str, Any]) -> dict[str, Any]:
    criteria = [
        {"id": f"route-{index}", "description": f"Fictional route {index}."}
        for index in range(slot["option_count"])
    ]
    attestation = {
        "scenario": slot["semantic_target"]["scenario"],
        "criterion_roles": slot["semantic_target"]["criterion_roles"],
        "selected_role": "matches_rule",
    }
    return {
        "instruction": "Choose the safe fictional route.",
        "state": {"summary": "Fictional only."},
        "criteria": [{"description": item["description"]} for item in criteria],
        "selected_index": 0,
        "semantic_equivalence_attestation": attestation,
    }


def _completion(content: dict[str, Any], request_id: str) -> tuple[int, dict[str, str], bytes]:
    return (
        200,
        {},
        json.dumps(
            {
                "id": request_id,
                "usage": {"cost": "0"},
                "choices": [{"message": {"content": json.dumps(content)}}],
            }
        ).encode(),
    )


def _client(tmp_path: Path, transport: corpus.Transport) -> corpus.OpenRouterCorpusClient:
    policy = corpus.POST_PILOT_CORPUS_LEDGER_POLICY
    return corpus.OpenRouterCorpusClient(
        transport=transport,
        allow_network=True,
        ledger_directory=tmp_path / "ledger",
        policy=policy,
        author_model="openai/gpt-4.1",
        reviewer_model="openai/gpt-4.1-mini",
        provider_preferences_by_stage={
            "corpus_author": {
                "order": ["Azure"],
                "allow_fallbacks": False,
                "require_parameters": True,
                "data_collection": "deny",
                "zdr": True,
            },
            "corpus_reviewer": {
                "order": ["Azure"],
                "allow_fallbacks": False,
                "require_parameters": True,
                "data_collection": "deny",
                "zdr": True,
            },
        },
        author_reasoning_effort=None,
        reviewer_temperature=0,
    )


def test_post_pilot_settled_validation_retries_are_bounded_and_diagnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        corpus.VerifiedMiniLMTokenizerReceipt,
        "create",
        classmethod(lambda cls, snapshot: _Counter()),
    )
    complete_plan = corpus.build_post_pilot_plan()
    slot = next(
        item
        for item in complete_plan["slots"]
        if item["pair_id"] is None and item["split"] == "synthetic_train"
    )
    calls = 0

    def transport(
        method: str, url: str, headers: object, body: bytes
    ) -> tuple[int, dict[str, str], bytes]:
        nonlocal calls
        del method, url, headers
        calls += 1
        if calls == 1:
            return _completion({}, "invalid-author")
        if calls == 2:
            return _completion({"records": [_record(slot)]}, "valid-author")
        if calls == 3:
            return _completion({}, "invalid-reviewer")
        selected = f"criterion-{slot['gold_position']}"
        return _completion(
            {
                "reviews": [
                    {
                        "task_id": slot["task_id"],
                        "status": "accepted",
                        "selected_criterion_id": selected,
                        "reason_codes": [],
                        "natural_language": True,
                        "generic_or_invented": True,
                        "exclusive_options": True,
                        "private_or_sensitive": False,
                        "semantic_equivalence_attestation": _record(slot)[
                            "semantic_equivalence_attestation"
                        ],
                    }
                ]
            },
            "valid-reviewer",
        )

    work = tmp_path / "work"
    client = _client(work, transport)
    diagnostics = pipeline._post_pilot_blank_diagnostics()
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    pipeline._run_post_pilot_batch(
        client,
        [slot],
        tmp_path / "snapshot",
        "test-key",
        accepted,
        rejected,
        diagnostics,
        work,
    )
    assert calls == 4
    assert diagnostics["author_response_failures"] == 1
    assert diagnostics["reviewer_response_failures"] == 1
    assert diagnostics["retry_recoveries"] == 2
    assert len(client.ledger.entries) == 4
    assert len({entry["reservation_id"] for entry in client.ledger.entries}) == 4
    payload = json.loads(pipeline._post_pilot_diagnostics_path(work, [slot]).read_bytes())
    assert payload["task_ids"] == [slot["task_id"]]
    assert "content" not in json.dumps(payload)
    corpus._validate_provider_lineage(client.ledger, accepted, rejected, {slot["task_id"]})

    plan = {
        **complete_plan,
        "slots": [slot],
        "author_batch_order": [[slot["task_id"]]],
    }
    monkeypatch.setattr(corpus, "validate_plan", lambda _plan: None)
    monkeypatch.setattr(corpus, "_minimums", lambda _rows: [])
    packet = tmp_path / "retry-packet"
    corpus.seal_packet(packet, plan, accepted, rejected, client.ledger.as_json(final=True))
    assert (packet / "packet.json").is_file()


def test_transport_uncertainty_closes_exact_lineage_and_later_batch_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        corpus.VerifiedMiniLMTokenizerReceipt,
        "create",
        classmethod(lambda cls, snapshot: _Counter()),
    )
    singles = [item for item in corpus.build_post_pilot_plan()["slots"] if item["pair_id"] is None]
    first, later = singles[:2]
    calls = 0

    def transport(
        method: str, url: str, headers: object, body: bytes
    ) -> tuple[int, dict[str, str], bytes]:
        nonlocal calls
        del method, url, headers, body
        calls += 1
        if calls == 1:
            raise OSError("outcome unknown")
        if calls == 2:
            return _completion({"records": [_record(later)]}, "later-author")
        return _completion(
            {
                "reviews": [
                    {
                        "task_id": later["task_id"],
                        "status": "accepted",
                        "selected_criterion_id": f"criterion-{later['gold_position']}",
                        "reason_codes": [],
                        "natural_language": True,
                        "generic_or_invented": True,
                        "exclusive_options": True,
                        "private_or_sensitive": False,
                        "semantic_equivalence_attestation": _record(later)[
                            "semantic_equivalence_attestation"
                        ],
                    }
                ]
            },
            "later-reviewer",
        )

    work = tmp_path / "work"
    client = _client(work, transport)
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    diagnostics = pipeline._post_pilot_blank_diagnostics()
    pipeline._run_post_pilot_batch(
        client, [first], tmp_path / "snapshot", "test-key", accepted, rejected, diagnostics, work
    )
    pipeline._run_post_pilot_batch(
        client, [later], tmp_path / "snapshot", "test-key", accepted, rejected, diagnostics, work
    )
    assert calls == 3
    assert rejected[0]["reason"] == "author_transport_uncertain"
    assert rejected[0]["author_response_sha256"] == "0" * 64
    assert rejected[0]["author_request_id"] == rejected[0]["author_reservation_id"]
    assert accepted[0]["task_id"] == later["task_id"]
    plan = corpus.build_post_pilot_plan()
    resolved = pipeline._load_resolved(work, plan)
    pipeline._require_post_pilot_uncertain_resolutions(
        client.ledger, resolved, diagnostics["transport_uncertain_calls"]
    )
    resolved[first["task_id"]]["row"]["author_request_id"] = "tampered"
    with pytest.raises(corpus.CorpusError, match="unbound uncertain resolution"):
        pipeline._require_post_pilot_uncertain_resolutions(
            client.ledger, resolved, diagnostics["transport_uncertain_calls"]
        )


def test_v4_transport_order_is_supplemental_only_and_keeps_post_pilot_contract() -> None:
    plan = corpus.build_post_pilot_recovery_plan()
    base_ids = {slot["task_id"] for slot in plan["slots"][:1600]}
    supplement_ids = set(plan["supplemental_task_ids"])
    batches = pipeline._author_batches(plan)

    assert len(batches) == 100
    assert all(len(batch) == 1 for batch in batches)
    assert {slot["task_id"] for batch in batches for slot in batch} == supplement_ids
    assert not {slot["task_id"] for batch in batches for slot in batch} & base_ids
    assert pipeline._post_pilot_config(plan) is not None
    assert pipeline._call_task_sets(plan) == {frozenset([task_id]) for task_id in supplement_ids}
    totals = pipeline._aggregate_preflight(
        plan,
        {},
        corpus.BudgetLedger(policy=corpus.POST_PILOT_CORPUS_LEDGER_POLICY),
    )
    assert totals["corpus_author"] > 0
    assert totals["corpus_reviewer"] > 0


def test_v4_rejects_missing_base_capsule_and_requires_private_recovery_paths(
    tmp_path: Path,
) -> None:
    plan = corpus.build_post_pilot_recovery_plan()
    with pytest.raises(corpus.CorpusError):
        pipeline._load_v4_base_capsule(tmp_path / "missing-capsule", plan)

    plan_path = tmp_path / "plan.json"
    plan_path.write_bytes(corpus._canonical(plan) + b"\n")
    with pytest.raises(corpus.CorpusError, match="requires report and durable research root"):
        pipeline.run_corpus(
            plan_path,
            tmp_path / "snapshot",
            tmp_path / "work",
            tmp_path / "packet",
            allow_network=True,
            transport=lambda *_args: (_ for _ in ()).throw(AssertionError("transport attempted")),
        )


def test_v4_circuit_stops_after_three_uncertain_calls_and_resume_skips_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The resumable v4 lane never repeats an uncertain singleton after a circuit stop."""

    assert (
        pipeline._v4_streak(
            [
                {"status": "uncertain"},
                {"status": "uncertain"},
                {"status": "settled"},
                {"status": "uncertain"},
            ],
            0,
            0,
        )
        == 1
    )
    assert (
        pipeline._v4_streak(
            [{"status": "uncertain"}, {"status": "overspent"}, {"status": "uncertain"}],
            0,
            0,
        )
        == 1
    )

    monkeypatch.setattr(
        corpus.VerifiedMiniLMTokenizerReceipt,
        "create",
        classmethod(lambda cls, snapshot: _Counter()),
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    plan = corpus.build_post_pilot_recovery_plan()
    suffix = plan["slots"][1600:1604]
    plan = {
        **plan,
        "slots": [*plan["base_slots"], *suffix],
        "supplemental_task_ids": [slot["task_id"] for slot in suffix],
        "supplemental_author_batch_order": [[slot["task_id"]] for slot in suffix],
    }
    base_ledger = corpus.BudgetLedger(policy=corpus.POST_PILOT_CORPUS_LEDGER_POLICY)
    monkeypatch.setattr(pipeline, "_v4_base_capsule", lambda *_args: tmp_path / "base")
    monkeypatch.setattr(
        pilot, "require_pilot_research_history", lambda *_args: (_Counter(), "a" * 64)
    )
    monkeypatch.setattr(pilot, "validate_research_capsule", lambda *_args: None)
    monkeypatch.setattr(
        pipeline,
        "_load_v4_base_capsule",
        lambda *_args: ([], [], base_ledger),
    )
    monkeypatch.setattr(
        pipeline,
        "_resolution_sha256",
        lambda path: (
            "a" * 64
            if path.name == "base"
            else pipeline.hashlib.sha256(pipeline._canonical([])).hexdigest()
        ),
    )
    monkeypatch.setattr(corpus, "_minimums", lambda _rows: [])

    work = tmp_path / "work"
    report = tmp_path / "report"
    root = tmp_path / "research"
    packet = tmp_path / "packet"
    calls: list[str] = []

    def failing_transport(
        method: str, url: str, headers: object, body: bytes
    ) -> tuple[int, dict[str, str], bytes]:
        del method, url, headers
        calls.append(body.decode())
        raise OSError("unknown")

    with pytest.raises(corpus.CorpusError, match="transport circuit open"):
        pipeline._run_post_pilot_recovery(
            plan, tmp_path / "snapshot", work, packet, report, root, transport=failing_transport
        )
    assert len(calls) == 3
    assert (
        json.loads(sorted((work / "ledger").glob("ledger-*.json"))[-1].read_bytes())["stop_reason"]
        == "transport_circuit_open"
    )
    assert (
        json.loads((report / "report-0000.json").read_bytes())["outcome"]
        == "incomplete_transport_circuit"
    )

    completed: list[str] = []

    def succeeding_transport(
        method: str, url: str, headers: object, body: bytes
    ) -> tuple[int, dict[str, str], bytes]:
        del method, url, headers
        stage = "reviews" if '"reviews"' in body.decode() else "records"
        task = suffix[3]
        completed.append(stage)
        if stage == "records":
            return _completion({"records": [_record(task)]}, "author")
        return _completion(
            {
                "reviews": [
                    {
                        "task_id": task["task_id"],
                        "status": "accepted",
                        "selected_criterion_id": f"criterion-{task['gold_position']}",
                        "reason_codes": [],
                        "natural_language": True,
                        "generic_or_invented": True,
                        "exclusive_options": True,
                        "private_or_sensitive": False,
                        "semantic_equivalence_attestation": _record(task)[
                            "semantic_equivalence_attestation"
                        ],
                    }
                ]
            },
            "reviewer",
        )

    def conflicting_seal(*_args: object, **_kwargs: object) -> Path:
        raise FileExistsError("simulated create-only publication race")

    original_diagnostics = pipeline._load_post_pilot_diagnostics

    def incomplete_after_network(*args: object, **kwargs: object) -> tuple[dict[str, int], bool]:
        diagnostics, complete = original_diagnostics(*args, **kwargs)
        resolved_ids = args[2]
        return diagnostics, complete and len(resolved_ids) < 4  # type: ignore[arg-type]

    monkeypatch.setattr(pipeline, "_load_post_pilot_diagnostics", incomplete_after_network)
    with pytest.raises(corpus.CorpusError, match="diagnostics incomplete"):
        pipeline._run_post_pilot_recovery(
            plan, tmp_path / "snapshot", work, packet, report, root, transport=succeeding_transport
        )
    assert completed == ["records", "reviews"]
    assert json.loads((report / "report-0001.json").read_bytes())["outcome"] == (
        "incomplete_operational"
    )

    original_publish = pipeline._publish_v4_revision
    original_seal = corpus.seal_post_pilot_recovery_packet

    def failing_publish(*_args: object, **_kwargs: object) -> tuple[Path, str, pilot.SpendScan]:
        raise OSError("simulated catalog failure")

    monkeypatch.setattr(pipeline, "_load_post_pilot_diagnostics", original_diagnostics)
    monkeypatch.setattr(
        corpus,
        "seal_post_pilot_recovery_packet",
        lambda *_args, **_kwargs: packet / "packet.json",
    )
    monkeypatch.setattr(pipeline, "_publish_v4_revision", failing_publish)
    with pytest.raises(OSError, match="catalog failure"):
        pipeline._run_post_pilot_recovery(
            plan, tmp_path / "snapshot", work, packet, report, root, transport=succeeding_transport
        )
    publication_report = json.loads((report / "report-0002.json").read_bytes())
    assert publication_report["outcome"] == "incomplete_operational"
    assert publication_report["errors"] == ["research_publication_failed"]
    assert publication_report["research_inventory_sha256"] == "0" * 64

    monkeypatch.setattr(pipeline, "_publish_v4_revision", original_publish)
    monkeypatch.setattr(corpus, "seal_post_pilot_recovery_packet", original_seal)
    monkeypatch.setattr(corpus, "seal_post_pilot_recovery_packet", conflicting_seal)
    with pytest.raises(FileExistsError, match="publication race"):
        pipeline._run_post_pilot_recovery(
            plan, tmp_path / "snapshot", work, packet, report, root, transport=succeeding_transport
        )
    assert completed == ["records", "reviews"]
    assert json.loads((report / "report-0003.json").read_bytes())["outcome"] == (
        "seal_validation_failed"
    )

    def fake_seal(target: Path, *_args: object, **_kwargs: object) -> Path:
        target.mkdir()
        (target / "packet.json").write_bytes(b'{"sealed":true}\n')
        return target / "packet.json"

    monkeypatch.setattr(corpus, "seal_post_pilot_recovery_packet", fake_seal)
    monkeypatch.setattr(
        corpus, "validate_post_pilot_recovery_packet_binding", lambda *_args, **_kwargs: None
    )
    pipeline._run_post_pilot_recovery(
        plan, tmp_path / "snapshot", work, packet, report, root, transport=succeeding_transport
    )
    assert completed == ["records", "reviews"]
    assert [path.name for path in sorted(report.glob("report-*.json"))] == [
        "report-0000.json",
        "report-0001.json",
        "report-0002.json",
        "report-0003.json",
        "report-0004.json",
    ]


@pytest.mark.parametrize("failure", ["history", "raw"])
def test_v4_recovery_preflight_rejects_incomplete_base_migration_without_transport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    plan = corpus.build_post_pilot_recovery_plan()
    root = tmp_path / "research-ledgers"
    base = root / "base-capsule"
    calls = 0

    def transport(*_args: object) -> tuple[int, dict[str, str], bytes]:
        nonlocal calls
        calls += 1
        raise AssertionError("transport attempted")

    monkeypatch.setattr(pipeline, "_v4_base_capsule", lambda *_args: base)
    if failure == "history":
        monkeypatch.setattr(
            pilot,
            "require_pilot_research_history",
            lambda *_args: (_ for _ in ()).throw(corpus.CorpusError("cumulative spend baseline")),
        )
        monkeypatch.setattr(
            pilot,
            "validate_research_capsule",
            lambda *_args: (_ for _ in ()).throw(AssertionError("raw validation attempted")),
        )
        expected = "cumulative spend baseline"
    else:
        monkeypatch.setattr(
            pilot, "require_pilot_research_history", lambda *_args: (_Counter(), "a" * 64)
        )
        monkeypatch.setattr(
            pilot,
            "validate_research_capsule",
            lambda *_args: (_ for _ in ()).throw(
                corpus.CorpusError("research capsule raw inventory")
            ),
        )
        expected = "research capsule raw inventory"

    with pytest.raises(corpus.CorpusError, match=expected):
        pipeline._run_post_pilot_recovery(
            plan,
            tmp_path / "snapshot",
            tmp_path / "supplement-work",
            tmp_path / "packet",
            tmp_path / "report",
            root,
            transport=transport,
        )
    assert calls == 0


def test_recovery_prepare_migrates_validates_and_writes_v4_plan_create_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    phase = tmp_path / "external" / "phase4e"
    paths = {
        "phase_root": phase,
        "research_ledgers": phase / "research-ledgers",
        "raw_evidence": phase / "raw-evidence",
        "model_artifacts": phase / "model-artifacts",
        "recovery_plan": phase / "recovery-plan-v4" / "plan.json",
    }
    internal = tmp_path / "internal" / "phase4e"
    calls: list[str] = []
    monkeypatch.setattr(pilot, "require_live_private_phase4e_root", lambda: phase)
    monkeypatch.setattr(pilot, "private_phase4e_paths", lambda: paths)
    monkeypatch.setattr(pilot, "_platform_phase4e_root", lambda: internal)
    monkeypatch.setattr(pipeline, "_POST_PILOT_WORK_DIR", tmp_path / "corpus-work-v47")

    def require_history(root: Path) -> tuple[_Counter, str]:
        calls.append(f"history:{root.name}")
        if root == paths["research_ledgers"] and "copy-catalog" not in calls:
            raise corpus.CorpusError("research inventory missing")
        return _Counter(), "a" * 64

    monkeypatch.setattr(pilot, "require_pilot_research_history", require_history)
    monkeypatch.setattr(
        pilot,
        "copy_research_catalog",
        lambda *_args: (calls.append("copy-catalog"), (_Counter(), "a" * 64))[1],
    )
    monkeypatch.setattr(
        pilot,
        "copy_raw_research_evidence",
        lambda *_args: calls.append("copy-raw") or "b" * 64,
    )
    monkeypatch.setattr(
        pilot,
        "create_research_capsule",
        lambda *_args: calls.append("create-capsule") or tmp_path / "capsule.json",
    )
    monkeypatch.setattr(
        pilot, "validate_research_capsule", lambda *_args: calls.append("validate-capsule")
    )
    monkeypatch.setattr(
        pilot,
        "record_research_catalog",
        lambda *_args: (calls.append("record-catalog") or "c" * 64, _Counter()),
    )
    monkeypatch.setattr(
        pilot,
        "write_research_catalog_migration_marker",
        lambda *_args: calls.append("write-marker") or tmp_path / "marker.json",
    )

    output = pipeline.run_prepare_recovery()
    assert output == paths["recovery_plan"]
    assert json.loads(output.read_bytes())["schema_version"] == "phase4e-universal-plan.v4"
    assert calls == [
        "history:research-ledgers",
        "history:research-ledgers",
        "copy-catalog",
        "copy-raw",
        "create-capsule",
        "validate-capsule",
        "record-catalog",
        "history:research-ledgers",
        "write-marker",
    ]
    assert pipeline.run_prepare_recovery() == output


def test_resolution_hash_uses_global_posix_path_order(tmp_path: Path) -> None:
    diagnostics = tmp_path / "diagnostics" / "call-z.json"
    resolved = tmp_path / "resolved" / "call-a.json"
    diagnostics.parent.mkdir()
    resolved.parent.mkdir()
    diagnostics.write_bytes(b'{"kind":"diagnostic"}\n')
    resolved.write_bytes(b'{"kind":"resolution"}\n')
    entries = [
        {
            "path": path.relative_to(tmp_path).as_posix(),
            "sha256": pipeline.hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in (diagnostics, resolved)
    ]
    expected = pipeline.hashlib.sha256(pipeline._canonical(entries)).hexdigest()
    assert pipeline._resolution_sha256(tmp_path) == expected


def test_ptbr_recovery_boundary_needs_sixty_six_accepted_rows() -> None:
    """The unchanged locale gate reaches 60% only at the reviewed 66-row boundary."""

    def rows(locale: str, amount: int) -> list[dict[str, Any]]:
        return [
            {
                "locale": locale,
                "split": "synthetic_train",
                "domain": "email_triage",
                "axes": {"explicitness": "explicit", "negation": "absent"},
                "option_count": 2,
                "family_id": f"{locale}-{index}",
                "pair_id": None,
            }
            for index in range(amount)
        ]

    base = [*rows("pt-BR", 848), *rows("en", 609)]
    assert "locale_minimum" in corpus._minimums([*base, *rows("pt-BR", 65)])
    assert "locale_minimum" not in corpus._minimums([*base, *rows("pt-BR", 66)])


def test_catalog_scans_historical_and_new_ledger_schemas_with_open_reservation(
    tmp_path: Path,
) -> None:
    root = tmp_path / "research"
    policies = (
        corpus.PILOT_LEDGER_POLICY_V1,
        corpus.PILOT_LEDGER_POLICY_V2,
        corpus.PILOT_LEDGER_POLICY_V3,
        corpus.PILOT_LEDGER_POLICY_V4,
        corpus.PILOT_LEDGER_POLICY,
    )
    for index, policy in enumerate(policies):
        ledger = corpus.BudgetLedger(policy=policy)
        corpus.write_ledger_snapshot(root / f"run-{index}" / "ledger", ledger)

    historical = corpus.BudgetLedger(policy=corpus.CORPUS_LEDGER_POLICY)
    for item in range(21):
        historical.reserve_request(
            "corpus_author", f"reservation-{5:02x}{item:062x}", Decimal("0.01")
        )
    corpus.write_ledger_snapshot(root / "run-5" / "ledger", historical)

    new = corpus.BudgetLedger(policy=corpus.POST_PILOT_CORPUS_LEDGER_POLICY)
    for item in range(6):
        reservation = f"reservation-{6:02x}{item:062x}"
        new.reserve_request("corpus_author", reservation, Decimal("0.01"))
        new.mark_uncertain(reservation)
    corpus.write_ledger_snapshot(root / "run-6" / "ledger", new)

    # This matches the v12 durable-index cardinality before the v3 work tree
    # is copied: 58 directories and 3,340 entries with historical open debt.
    for directory in range(7, 58):
        ledger = corpus.BudgetLedger(policy=corpus.PILOT_LEDGER_POLICY)
        total = 63 if directory == 57 else 65
        for item in range(total):
            reservation = f"reservation-{directory:02x}{item:062x}"
            ledger.reserve_request("pilot_author", reservation, Decimal("0.01"))
            ledger.settle_request(reservation, f"request-{directory}-{item}", Decimal(), "b" * 64)
        corpus.write_ledger_snapshot(root / f"run-{directory}" / "ledger", ledger)
    inventory, scan = pilot.record_research_catalog(root)
    assert len(inventory) == 64
    assert scan.directory_count == 58
    assert scan.entry_count == 3340
    assert scan.settled_count == 3313
    assert scan.uncertain_count == 6
    assert scan.open_reservation_count == 21
    assert pilot.scan_research_ledgers(root) == scan


def test_completed_post_pilot_work_recovers_seal_and_report_without_transport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A post-call sealing failure can be retried from durable local evidence only."""

    complete_plan = corpus.build_post_pilot_plan()
    slot = next(item for item in complete_plan["slots"] if item["pair_id"] is None)
    plan = {**complete_plan, "slots": [slot], "author_batch_order": [[slot["task_id"]]]}
    work = tmp_path / "work"
    rejected = {"task_id": slot["task_id"], "split": slot["split"], "reason": "capacity"}
    pipeline._store_call_resolution(work, [slot], [(rejected, "rejected")])
    corpus.write_ledger_snapshot(
        work / "ledger", corpus.BudgetLedger(policy=corpus.POST_PILOT_CORPUS_LEDGER_POLICY)
    )
    pipeline._store_post_pilot_diagnostics(work, [slot], pipeline._post_pilot_blank_diagnostics())
    # Fixture-sized stand-in for the many historical snapshots omitted by the
    # compact capsule, so the real 10x compactness guard remains exercised.
    (work / "raw-evidence.bin").write_bytes(b"x" * 100_000)
    packet = tmp_path / "packet"
    report = tmp_path / "report"
    root = tmp_path / "research"
    calls = 0

    def fake_seal(*args: object, **kwargs: object) -> Path:
        nonlocal calls
        del args, kwargs
        calls += 1
        packet.mkdir()
        (packet / "packet.json").write_bytes(b'{"sealed":true}\n')
        return packet / "packet.json"

    monkeypatch.setattr(corpus, "seal_packet", fake_seal)
    monkeypatch.setattr(corpus, "validate_accepted_packet_pre_holdout", lambda path: None)
    monkeypatch.setattr(corpus, "_minimums", lambda _rows: [])
    monkeypatch.setattr(
        pilot,
        "_copy_work_tree",
        lambda *_args: (_ for _ in ()).throw(AssertionError("full work tree must not publish")),
    )

    def transport(
        method: str, url: str, headers: Mapping[str, str], body: bytes
    ) -> tuple[int, Mapping[str, str], bytes]:
        del method, url, headers, body
        raise AssertionError("transport must not run")

    pipeline._run_post_pilot_corpus(
        plan,
        tmp_path / "snapshot",
        work,
        packet,
        report,
        root,
        transport=transport,
    )
    first_report = (report / "report.json").read_bytes()
    pipeline._run_post_pilot_corpus(
        plan,
        tmp_path / "snapshot",
        work,
        packet,
        report,
        root,
        transport=transport,
    )
    assert calls == 1
    assert (report / "report.json").read_bytes() == first_report


def test_missing_credential_preflight_creates_no_work_or_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    plan = corpus.build_post_pilot_plan()
    work = tmp_path / "work"
    packet = tmp_path / "packet"
    report = tmp_path / "report"
    root = tmp_path / "research"

    with pytest.raises(corpus.CorpusError, match="missing provider credential"):
        pipeline._run_post_pilot_corpus(
            plan,
            tmp_path / "snapshot",
            work,
            packet,
            report,
            root,
            transport=lambda *_args: (_ for _ in ()).throw(AssertionError("transport attempted")),
        )
    assert not work.exists()
    assert not packet.exists()
    assert not report.exists()
    assert not root.exists()


def test_public_v3_lane_rejects_private_artifacts_inside_checkout(tmp_path: Path) -> None:
    plan = corpus.build_post_pilot_plan()
    plan_dir = tmp_path / "corpus-plan-v3"
    plan_dir.mkdir()
    plan_path = plan_dir / "plan.json"
    plan_path.write_bytes(corpus._canonical(plan) + b"\n")

    with pytest.raises(corpus.CorpusError, match="post-pilot artifact path"):
        pipeline.run_corpus(
            plan_path,
            tmp_path / "snapshot",
            tmp_path / "corpus-work-v47",
            tmp_path / "packet-in-checkout",
            allow_network=True,
            report_dir=tmp_path / "corpus-report-v3",
            artifact_root=tmp_path / "research-ledgers",
        )


def test_reviewed_v3_paths_require_packet_below_private_model_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan_path = tmp_path / "corpus-plan-v3" / "plan.json"
    work = tmp_path / "corpus-work-v47"
    report = tmp_path / "corpus-report-v3"
    research = tmp_path / "research-ledgers"
    model_root = tmp_path / "model-artifacts"
    monkeypatch.setattr(pipeline, "_POST_PILOT_PLAN_PATH", plan_path.resolve())
    monkeypatch.setattr(pipeline, "_POST_PILOT_WORK_DIR", work.resolve())
    monkeypatch.setattr(pipeline, "_POST_PILOT_REPORT_DIR", report.resolve())
    monkeypatch.setattr(pipeline, "_POST_PILOT_MODEL_ARTIFACT_ROOT", model_root.resolve())
    monkeypatch.setattr(pilot, "canonical_research_ledger_root", lambda: research)

    pipeline._require_post_pilot_artifact_paths(
        plan_path,
        work,
        model_root / "accepted-packet-v3",
        report,
        research,
    )
    with pytest.raises(corpus.CorpusError, match="post-pilot packet path"):
        pipeline._require_post_pilot_artifact_paths(
            plan_path,
            work,
            tmp_path / "packet-outside-model-root",
            report,
            research,
        )


def test_private_state_root_moves_research_not_the_holdout_claim_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys
    import types

    module = types.ModuleType("platformdirs")
    module.user_state_path = lambda *_args, **_kwargs: "/tmp/saracura-platform-state"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "platformdirs", module)
    baseline_claim_root = training._holdout_release_registry_directory()
    configured = tmp_path / "private-state"
    configured.mkdir(mode=0o700)
    monkeypatch.setenv("SARACURA_PRIVATE_STATE_ROOT", str(configured))

    assert pilot.canonical_research_ledger_root() == configured / "phase4e/research-ledgers"
    assert pilot.private_phase4e_paths()["raw_evidence"] == configured / "phase4e/raw-evidence"
    assert training._holdout_release_registry_directory() == baseline_claim_root


def test_live_private_state_root_requires_external_unsynchronized_volume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    volume = tmp_path / "external-volume"
    volume.mkdir(mode=0o700)
    private = volume / "private-state"
    private.mkdir(mode=0o700)
    monkeypatch.setattr(pilot, "_LIVE_PRIVATE_VOLUME_ROOT", volume)
    monkeypatch.setattr(pilot, "_SYNCED_FELHEN_ROOT", tmp_path / "felhencloud")
    monkeypatch.setenv("SARACURA_PRIVATE_STATE_ROOT", str(private))
    assert pilot.require_live_private_phase4e_root() == private / "phase4e"

    synced = tmp_path / "felhencloud" / "private-state"
    synced.mkdir(mode=0o700, parents=True)
    monkeypatch.setattr(pilot, "_LIVE_PRIVATE_VOLUME_ROOT", tmp_path)
    monkeypatch.setenv("SARACURA_PRIVATE_STATE_ROOT", str(synced))
    with pytest.raises(corpus.CorpusError, match="private state volume"):
        pilot.require_live_private_phase4e_root()


@pytest.mark.parametrize("value", ["relative-state", "missing-state"])
def test_private_state_root_rejects_nonprivate_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    configured = tmp_path / value
    root_value = str(configured) if value != "relative-state" else value
    monkeypatch.setenv("SARACURA_PRIVATE_STATE_ROOT", root_value)
    with pytest.raises(corpus.CorpusError, match="private state root"):
        pilot.configured_private_phase4e_root()


def test_private_state_root_rejects_permissive_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configured = tmp_path / "private-state"
    configured.mkdir(mode=0o700)
    configured.chmod(0o755)
    monkeypatch.setenv("SARACURA_PRIVATE_STATE_ROOT", str(configured))
    with pytest.raises(corpus.CorpusError, match="private state root"):
        pilot.configured_private_phase4e_root()


def test_private_state_root_rejects_symlink_and_checkout_descendant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "target"
    target.mkdir(mode=0o700)
    link = tmp_path / "private-link"
    link.symlink_to(target, target_is_directory=True)
    monkeypatch.setenv("SARACURA_PRIVATE_STATE_ROOT", str(link))
    with pytest.raises(corpus.CorpusError, match="private state root"):
        pilot.configured_private_phase4e_root()

    checkout = tmp_path / "checkout"
    checkout.mkdir(mode=0o700)
    nested = checkout / "private-state"
    nested.mkdir(mode=0o700)
    monkeypatch.setattr(pilot, "_git_protected_paths", lambda: {checkout})
    monkeypatch.setenv("SARACURA_PRIVATE_STATE_ROOT", str(nested))
    with pytest.raises(corpus.CorpusError, match="private state root"):
        pilot.configured_private_phase4e_root()


def test_catalog_migration_marker_blocks_internal_research_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys
    import types

    module = types.ModuleType("platformdirs")
    module.user_state_path = lambda *_args, **_kwargs: tmp_path / "platform-state"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "platformdirs", module)
    configured = tmp_path / "private-state"
    configured.mkdir(mode=0o700)
    external = configured / "phase4e" / "research-ledgers"
    corpus.write_ledger_snapshot(
        external / "history" / "ledger",
        corpus.BudgetLedger(policy=corpus.POST_PILOT_CORPUS_LEDGER_POLICY),
    )
    inventory, _scan = pilot.record_research_catalog(external)
    marker = pilot.write_research_catalog_migration_marker(external, inventory)
    payload = json.loads(marker.read_bytes())
    assert str(external) not in marker.read_text()
    assert payload["inventory_sha256"] == inventory

    monkeypatch.delenv("SARACURA_PRIVATE_STATE_ROOT", raising=False)
    with pytest.raises(corpus.CorpusError, match="private state root required"):
        pilot.canonical_research_ledger_root()
    internal = tmp_path / "platform-state" / "phase4e" / "research-ledgers"
    with pytest.raises(corpus.CorpusError, match="private state root required"):
        pilot.require_research_catalog_write_root(internal)
    with pytest.raises(corpus.CorpusError, match="private state root required"):
        pilot.record_research_catalog(internal)

    monkeypatch.setenv("SARACURA_PRIVATE_STATE_ROOT", str(configured))
    assert pilot.canonical_research_ledger_root() == external


def test_v4_packet_cannot_enter_legacy_training_before_phase_4e3b(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = corpus.build_post_pilot_recovery_plan()
    packet = tmp_path / "packet"
    rejected = [
        {"task_id": slot["task_id"], "split": slot["split"], "reason": "capacity"}
        for slot in plan["slots"]
    ]
    monkeypatch.setattr(corpus, "_minimums", lambda _rows: [])
    corpus.seal_packet(
        packet,
        plan,
        [],
        rejected,
        corpus.BudgetLedger(policy=corpus.POST_PILOT_CORPUS_LEDGER_POLICY).as_json(final=True),
    )
    monkeypatch.setattr(training, "require_phase4e_authorization", lambda _action: {})
    registry = tmp_path / "holdout-releases"
    monkeypatch.setattr(training, "_holdout_release_registry_directory", lambda: registry)
    snapshot = tmp_path / "missing-snapshot"
    embedding_output = tmp_path / "embeddings"
    training_output = tmp_path / "models"

    with pytest.raises(training.TrainingError, match=r"requires Phase 4E\.3B"):
        training.extract_and_seal_embeddings(packet, snapshot, "cpu", embedding_output)
    with pytest.raises(training.TrainingError, match=r"requires Phase 4E\.3B"):
        training.train_and_seal(
            tmp_path / "missing-capsule",
            packet,
            snapshot,
            "cpu",
            training_output,
            "run",
        )
    binding = training.AcceptedPacketBinding(
        *("0" * 64 for _ in range(5)),
        train_dev_identities=(),
        holdout_identities=(),
        identities=(),
    )
    with pytest.raises(training.TrainingError, match=r"requires Phase 4E\.3B"):
        training.verify_holdout_descriptor_bound_embeddings(
            cast(training.EmbeddingCapsule, object()),
            packet,
            binding,
            snapshot,
            "cpu",
            "0" * 64,
            "0" * 64,
        )
    assert not snapshot.exists()
    assert not embedding_output.exists()
    assert not training_output.exists()
    assert not registry.exists()


def test_minimum_failure_writes_a_numbered_report_before_sealing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    complete_plan = corpus.build_post_pilot_plan()
    slot = next(item for item in complete_plan["slots"] if item["pair_id"] is None)
    plan = {**complete_plan, "slots": [slot], "author_batch_order": [[slot["task_id"]]]}
    work = tmp_path / "work"
    rejected = {"task_id": slot["task_id"], "split": slot["split"], "reason": "capacity"}
    pipeline._store_call_resolution(work, [slot], [(rejected, "rejected")])
    corpus.write_ledger_snapshot(
        work / "ledger", corpus.BudgetLedger(policy=corpus.POST_PILOT_CORPUS_LEDGER_POLICY)
    )
    pipeline._store_post_pilot_diagnostics(work, [slot], pipeline._post_pilot_blank_diagnostics())
    (work / "raw-evidence.bin").write_bytes(b"x" * 100_000)
    monkeypatch.setattr(corpus, "_minimums", lambda _rows: ["locale_minimum"])
    monkeypatch.setattr(
        corpus,
        "seal_packet",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("seal must not run")),
    )
    monkeypatch.setattr(
        corpus,
        "validate_accepted_packet_pre_holdout",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("holdout must not open")),
    )
    monkeypatch.setattr(
        pilot,
        "_copy_work_tree",
        lambda *_args: (_ for _ in ()).throw(AssertionError("full work tree must not publish")),
    )
    packet = tmp_path / "packet"
    report = tmp_path / "report"
    root = tmp_path / "research"

    with pytest.raises(corpus.CorpusError, match="minimum_failed"):
        pipeline._run_post_pilot_corpus(
            plan, tmp_path / "snapshot", work, packet, report, root, transport=None
        )
    payload = json.loads((report / "report-0000.json").read_bytes())
    assert payload["outcome"] == "minimum_failed"
    assert payload["errors"] == ["locale_minimum"]
    assert payload["unresolved_count"] == 0
    assert not packet.exists()

    with pytest.raises(corpus.CorpusError, match="minimum_failed"):
        pipeline._run_post_pilot_corpus(
            plan, tmp_path / "snapshot", work, packet, report, root, transport=None
        )
    assert [path.name for path in report.glob("report-*.json")] == ["report-0000.json"]
