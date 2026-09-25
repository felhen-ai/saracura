"""Offline completion checks for the Phase 4E post-pilot corpus lane."""

from __future__ import annotations

import json
from collections.abc import Mapping
from decimal import Decimal
from pathlib import Path
from typing import Any

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

    monkeypatch.setenv("SARACURA_PRIVATE_STATE_ROOT", str(configured))
    assert pilot.canonical_research_ledger_root() == external


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
