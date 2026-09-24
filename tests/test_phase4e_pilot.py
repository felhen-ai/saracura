"""Adversarial checks for the isolated Phase 4E.2b protocol pilot."""

from __future__ import annotations

import hashlib
import json
import socket
from collections import Counter
from decimal import Decimal
from pathlib import Path
from typing import Any, cast, get_args

import pytest
from pydantic import ValidationError

from benchmarks import saracura_universal_corpus as corpus
from benchmarks import saracura_universal_pilot as pilot
from benchmarks.data_policy_registry import (
    bundled_registry_path,
    bundled_registry_v2_path,
    load_registry,
    load_registry_v2,
)
from benchmarks.first_party_gate import validate_protocol_bytes
from benchmarks.saracura_universal_corpus import (
    AcceptedPacketRow,
    ValidatedAuthorRow,
    resolve_reviews,
    validate_author_rows,
)

V1_REGISTRY_SHA256 = "633245cde07b9f944647e615e17855d392048074b1dbb4a021f5f29718a5e43c"
ROOT = Path(__file__).parents[1]


class _Counter:
    def count(self, text: str) -> int:
        del text
        return 1


def _record(slot: dict[str, Any]) -> dict[str, Any]:
    criteria = [
        {"id": f"route-{index}", "description": f"Route criterion {index}."}
        for index in range(int(slot["option_count"]))
    ]
    target = cast(dict[str, Any], slot["semantic_target"])
    roles = list(cast(list[str], target["criterion_roles"]))
    gold = int(slot["gold_position"])
    roles.insert(gold, roles.pop(0))
    record = {
        "task_id": slot["task_id"],
        "family_id": slot["family_id"],
        "locale": slot["locale"],
        "domain": slot["domain"],
        "axes": slot["axes"],
        "option_count": slot["option_count"],
        "gold_position": gold,
        "instruction": "Choose the safest fictional route.",
        "state": {"summary": "A fictional request."},
        "criteria": criteria,
        "selected_criterion_id": criteria[gold]["id"],
        "semantic_equivalence_attestation": {
            "scenario": target["scenario"],
            "criterion_roles": roles,
            "selected_role": "matches_rule",
        },
    }
    if isinstance(slot["pair_id"], str):
        record["cross_locale_attestation"] = {
            "pair_id": slot["pair_id"],
            **record["semantic_equivalence_attestation"],
        }
    return record


def _review(row: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    review = {
        "task_id": row["task_id"],
        "status": "accepted",
        "selected_criterion_id": row["selected_criterion_id"],
        "reason_codes": [],
        "natural_language": True,
        "fictional": True,
        "exclusive_options": True,
        "private_or_sensitive": False,
        "semantic_equivalence_attestation": row["semantic_equivalence_attestation"],
    }
    review.update(overrides)
    return review


def _write_ledger(work: Path, ledger: corpus.BudgetLedger) -> None:
    directory = work / "ledger"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "ledger-0000.json").write_bytes(corpus._canonical(ledger.as_json()) + b"\n")


def _baseline_root(root: Path, *, debit_delta: Decimal = Decimal()) -> Path:
    """Synthesize the committed entry counts. Open reservations keep their full debit."""

    open_each = Decimal("0.01971612") / Decimal(18)
    assert open_each * 18 == Decimal("0.01971612")
    settled_debit = Decimal("5.12153043") - Decimal("0.01971612") + debit_delta
    ledger = corpus.BudgetLedger()
    ledger.entries.append(
        {
            "stage": "corpus_reviewer",
            "reservation_id": "reservation-" + f"{1:064x}",
            "request_id": "historical-settled",
            "local_worst_case_usd": str(settled_debit),
            "provider_cost_usd": "0.81263107",
            "debit_usd": str(max(settled_debit, Decimal("0.81263107"))),
            "response_sha256": f"{2:064x}",
            "status": "settled",
        }
    )
    for index in range(2098):
        ledger.entries.append(
            {
                "stage": "corpus_reviewer",
                "reservation_id": "reservation-" + f"{index + 10:064x}",
                "request_id": f"historical-{index}",
                "local_worst_case_usd": "0",
                "provider_cost_usd": "0",
                "debit_usd": "0",
                "response_sha256": f"{index + 10:064x}",
                "status": "settled",
            }
        )
    for index in range(18):
        reservation = "reservation-" + f"{index + 3000:064x}"
        ledger.entries.append(
            {
                "stage": "corpus_author",
                "reservation_id": reservation,
                "request_id": reservation,
                "local_worst_case_usd": str(open_each),
                "provider_cost_usd": "0",
                "debit_usd": str(open_each),
                "response_sha256": "0" * 64,
                "status": "reserved",
            }
        )
    _write_ledger(root / "corpus-work-00", ledger)
    (root / "corpus-work-00" / "ledger" / "wilson-0000.json").write_bytes(
        b'{"schema_version":"phase4e-wilson-ledger.v1"}\n'
    )
    resolved = root / "corpus-work-00" / "resolved"
    resolved.mkdir(parents=True)
    (resolved / "task-note.json").write_bytes(b'{"kind":"baseline"}\n')
    for index in range(1, 46):
        _write_ledger(root / f"corpus-work-{index:02d}", corpus.BudgetLedger())
    pilot.record_research_catalog(root)
    return root


def _completion(payload: dict[str, Any], *, cost: str = "0") -> tuple[int, dict[str, str], bytes]:
    return (
        200,
        {},
        json.dumps(
            {
                "usage": {"cost": cost},
                "choices": [{"message": {"content": json.dumps(payload)}}],
            }
        ).encode(),
    )


def _raw_completion(content: str, *, cost: str = "0") -> tuple[int, dict[str, str], bytes]:
    return (
        200,
        {},
        json.dumps(
            {
                "usage": {"cost": cost},
                "choices": [{"message": {"content": content}}],
            }
        ).encode(),
    )


def _author_wire(slots: list[dict[str, Any]]) -> dict[str, Any]:
    wire: dict[str, Any] = {}
    for index, slot in enumerate(slots):
        target = cast(dict[str, Any], slot["semantic_target"])
        prefix = f"record_{index}_"
        token = cast(str, slot["task_id"])
        wire[f"{prefix}instruction"] = f"Pick fictional route {token}."
        wire[f"{prefix}state_summary"] = f"Rule {token} admits one fictional match."
        wire[f"{prefix}selected_index"] = 0
        wire[f"{prefix}scenario"] = target["scenario"]
        wire[f"{prefix}selected_role"] = "matches_rule"
        for criterion_index, role in enumerate(cast(list[str], target["criterion_roles"])):
            wire[f"{prefix}criterion_{criterion_index}_description"] = (
                f"{token} option {criterion_index} stands alone."
            )
            wire[f"{prefix}criterion_{criterion_index}_role"] = role
    return wire


def _reviewer_payload(
    slot: dict[str, Any], *, disagree_answer: bool, disagree_scenario: bool
) -> dict[str, Any]:
    roles = list(cast(list[str], slot["semantic_target"]["criterion_roles"]))
    selected_role = roles.pop(0)
    roles.insert(int(slot["gold_position"]), selected_role)
    scenario = cast(str, slot["semantic_target"]["scenario"])
    if disagree_scenario:
        scenario = next(code for code in corpus.SCENARIO_CODES if code != scenario)
    selected = int(slot["gold_position"])
    if disagree_answer:
        selected = (selected + 1) % int(slot["option_count"])
    return {
        "reviews": [
            {
                "status": "accepted",
                "selected_criterion_id": f"criterion-{selected}",
                "reason_codes": [],
                "natural_language": True,
                "generic_or_invented": True,
                "exclusive_options": True,
                "private_or_sensitive": False,
                "semantic_equivalence_attestation": {
                    "scenario": scenario,
                    "criterion_roles": roles,
                    "selected_role": "matches_rule",
                },
            }
        ]
    }


def test_registry_v1_bytes_stay_bound_and_v2_is_pilot_only() -> None:
    raw_v1 = bundled_registry_path().read_bytes()
    raw_v2 = bundled_registry_v2_path().read_bytes()
    assert hashlib.sha256(raw_v1).hexdigest() == V1_REGISTRY_SHA256
    assert b"protocol_pilot" not in raw_v1
    assert load_registry(raw_v1).schema_version == "training-data-source-policies.v1"
    with pytest.raises(ValueError, match="invalid data policy registry"):
        load_registry(raw_v2)
    loaded = load_registry_v2(raw_v2)
    protocol_pilot = loaded.exceptions[1].model_dump(mode="json")["protocol_pilot"]
    assert loaded.schema_version == "training-data-source-policies.v2"
    assert protocol_pilot["training_authorized"] is False
    assert protocol_pilot["evaluation_only"] is True
    protocol = (ROOT / "benchmarks/manifests/support-routing-protocol.v1.json").read_bytes()
    validate_protocol_bytes(protocol)
    assert b"training-data-source-policies.v1.json" in protocol
    assert V1_REGISTRY_SHA256.encode() in protocol
    assert "protocol_pilot" not in get_args(ValidatedAuthorRow.model_fields["split"].annotation)
    assert "protocol_pilot" not in get_args(AcceptedPacketRow.model_fields["split"].annotation)


def test_pilot_plan_is_disjoint_balanced_and_cost_is_report_only(tmp_path: Path) -> None:
    output = tmp_path / "pilot-plan-v6" / "plan.json"
    pilot.write_pilot_plan(output)
    plan = json.loads(output.read_bytes())
    pilot.validate_pilot_plan(plan)
    slots = plan["slots"]
    assert plan["schema_version"] == pilot.PILOT_PLAN_SCHEMA
    assert (
        plan["reviewer_system_sha256"]
        == hashlib.sha256(pilot._pilot_reviewer_system().encode("utf-8")).hexdigest()
    )
    assert plan["seed"] == pilot.PILOT_SEED
    assert len(slots) == 140
    corpus_ids = {slot["task_id"] for slot in corpus.build_plan()["slots"]}
    pilot_ids = {slot["task_id"] for slot in slots}
    assert corpus_ids.isdisjoint(pilot_ids)
    pairs: dict[str, list[dict[str, Any]]] = {}
    for slot in slots:
        pairs.setdefault(slot["pair_id"], []).append(slot)
    assert len(pairs) == 70
    domain_counts: Counter[str] = Counter()
    option_counts: Counter[int] = Counter()
    policy = pilot.validate_pilot_policy()
    for members in pairs.values():
        assert {row["locale"] for row in members} == {"pt-BR", "en"}
        assert len({row["domain"] for row in members}) == 1
        assert len({json.dumps(row["semantic_target"]) for row in members}) == 1
        domain = members[0]["domain"]
        domain_counts[domain] += 1
        option_counts[members[0]["option_count"]] += 1
        scenario = members[0]["semantic_target"]["scenario"]
        assert scenario in policy["domain_scenario_map"][domain]
        assert scenario == pilot._scenario_for(
            pilot.PILOT_SEED, members[0]["pair_id"], domain, policy["domain_scenario_map"]
        )
    assert option_counts == {count: 10 for count in range(2, 9)}
    assert sum(count == 6 for count in domain_counts.values()) == 10
    assert sum(count == 5 for count in domain_counts.values()) == 2
    for locale in ("pt-BR", "en"):
        for option_count in range(2, 9):
            positions = [
                slot["gold_position"]
                for slot in slots
                if slot["locale"] == locale and slot["option_count"] == option_count
            ]
            assert len(positions) == 10
            counts = Counter(positions)
            assert max(counts.values()) - min(counts.values()) <= 1
    estimate = plan["cost_estimate"]
    assert estimate["mode"] == "report_only"
    assert estimate["report_interval_usd"] == "10.00"
    assert Decimal(estimate["pilot_author_usd"]) + Decimal(
        estimate["pilot_reviewer_usd"]
    ) == Decimal(estimate["total_usd"])
    with pytest.raises(corpus.CorpusError, match="pilot plan cannot enter corpus"):
        corpus.validate_plan(plan)
    with pytest.raises(corpus.CorpusError, match="pilot task cannot enter packet"):
        corpus._validate_packet_resolution([], [{"task_id": next(iter(pilot_ids))}], {"slots": []})
    with pytest.raises(ValidationError):
        AcceptedPacketRow.model_validate({"split": "protocol_pilot"})


def test_pilot_reviewer_uses_genericity_wire_contract() -> None:
    system = pilot._pilot_reviewer_system()
    assert "observable content-safety flag" in system
    assert "generic_or_invented and private_or_sensitive only from instruction" in system
    assert "Ignore task_id, criterion IDs, locale, domain" in system
    assert "trusted routing metadata" in system
    assert "name without an observable real-world anchor" in system
    assert "do not search" in system
    assert "require external provenance" in system
    assert "generic_or_invented=false" in system
    assert "reason code real_world_anchor" in system
    assert "fictional=" not in system
    assert "do not treat provenance or stated synthetic intent as proof" not in system

    schema = pilot.pilot_reviewer_schema(1)
    item = schema["properties"]["reviews"]["items"]
    assert "generic_or_invented" in item["properties"]
    assert "generic_or_invented" in item["required"]
    assert "fictional" not in item["properties"]
    assert "fictional" not in item["required"]
    genericity = item["properties"]["generic_or_invented"]
    assert genericity["title"] == "Generic Or Invented"
    assert "observable real-world anchor" in genericity["description"]
    encoded_schema = json.dumps(schema)
    assert "Fictional" not in encoded_schema
    assert '"fictional"' not in encoded_schema
    assert "real_world_anchor" in encoded_schema

    task_id = pilot.build_plan()["slots"][0]["task_id"]
    payload = _reviewer_payload(
        pilot.build_plan()["slots"][0], disagree_answer=False, disagree_scenario=False
    )["reviews"][0]
    bound = pilot.bind_pilot_reviewer_record(payload, task_id)
    assert bound["fictional"] is True
    assert "generic_or_invented" not in bound
    for invalid in (
        {key: value for key, value in payload.items() if key != "generic_or_invented"},
        {**payload, "generic_or_invented": "yes"},
        {**payload, "fictional": True},
        {**payload, "reason_codes": ["fictional"]},
    ):
        with pytest.raises(corpus.CorpusError, match="genericity schema"):
            pilot.bind_pilot_reviewer_record(invalid, task_id)


def test_v2_ledger_reports_cost_without_enforcing_a_ceiling(tmp_path: Path) -> None:
    task_id = pilot.build_plan()["slots"][0]["task_id"]

    def transport(
        method: str, url: str, headers: object, body: bytes
    ) -> tuple[int, dict[str, str], bytes]:
        del method, url, headers, body
        return _completion({}, cost="25.00")

    client = corpus.OpenRouterCorpusClient(
        transport=transport,
        allow_network=True,
        ledger_directory=tmp_path / "ledger",
        policy=corpus.PILOT_LEDGER_POLICY,
        author_stage=pilot.AUTHOR_STAGE,
        reviewer_stage=pilot.REVIEWER_STAGE,
    )
    request = {
        "stage": pilot.AUTHOR_STAGE,
        "model": corpus.AUTHOR_MODEL,
        "messages": [{"role": "user", "content": "test"}],
        "response_schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {},
        },
        "max_output_tokens": 1,
        "api_key": "test-key",
        "task_ids": [task_id],
    }
    client._request(**request)
    client._request(**request)
    assert [entry["status"] for entry in client.ledger.entries] == ["settled", "settled"]
    assert sum(Decimal(entry["provider_cost_usd"]) for entry in client.ledger.entries) == Decimal(
        "50.00"
    )
    assert len({entry["reservation_id"] for entry in client.ledger.entries}) == 2
    client.ledger.require_complete_provider_journal()


def test_domain_map_must_cover_every_domain(tmp_path: Path) -> None:
    policy = json.loads(pilot.POLICY_PATH.read_bytes())
    policy["domain_scenario_map"].pop("finance")
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(policy), encoding="utf-8")
    with pytest.raises(corpus.CorpusError, match="pilot domain map"):
        pilot.validate_pilot_policy(path)
    policy = json.loads(pilot.POLICY_PATH.read_bytes())
    policy["domain_scenario_map"]["finance"] = []
    path.write_text(json.dumps(policy), encoding="utf-8")
    with pytest.raises(corpus.CorpusError, match="pilot domain map"):
        pilot.validate_pilot_policy(path)


def test_semantic_disagreement_is_diagnostic_and_corpus_default_still_rejects() -> None:
    slot = next(
        row
        for row in corpus.build_plan()["slots"]
        if row["pair_id"] is None and row["option_count"] == 2
    )
    row = validate_author_rows([_record(slot)], [slot], _Counter())[0]
    other = next(
        code
        for code in corpus.SCENARIO_CODES
        if code != row["semantic_equivalence_attestation"]["scenario"]
    )
    review = _review(
        row,
        semantic_equivalence_attestation={
            **row["semantic_equivalence_attestation"],
            "scenario": other,
        },
    )
    accepted, rejected = resolve_reviews([row], [review])
    assert accepted == []
    assert rejected[0]["reason"] == "scenario_disagreement"
    accepted, rejected = resolve_reviews(
        [row],
        [review],
        acceptance=pilot.pilot_acceptance,
        pair_resolution=pilot.pilot_pair_resolution,
    )
    assert rejected == []
    assert accepted[0]["task_id"] == row["task_id"]
    quality = _review(row, fictional=False)
    accepted, rejected = resolve_reviews(
        [row],
        [quality],
        acceptance=pilot.pilot_acceptance,
        pair_resolution=pilot.pilot_pair_resolution,
    )
    assert accepted == []
    assert rejected[0]["reason"] == "review_quality_fictional"


def test_pilot_pair_keeps_a_clean_sibling_when_corpus_would_drop_it() -> None:
    plan = pilot.build_plan()
    pair = [slot for slot in plan["slots"] if slot["pair_id"] == plan["slots"][0]["pair_id"]]
    rows = []
    for index, slot in enumerate(pair):
        generated = {
            "instruction": f"Choose fictional route {slot['task_id']}.",
            "state": {"summary": f"Rule {slot['task_id']} admits one match."},
            "criteria": [
                {"description": f"{slot['task_id']} option {criterion}."}
                for criterion in range(slot["option_count"])
            ],
            "selected_index": 0,
        }
        target = cast(dict[str, Any], slot["semantic_target"])
        attestation: dict[str, Any] = {
            "scenario": target["scenario"],
            "criterion_roles": target["criterion_roles"],
            "selected_role": "matches_rule",
        }
        if index == 1:
            attestation = {
                **attestation,
                "scenario": "content_safety"
                if target["scenario"] != "content_safety"
                else "action_required",
            }
        generated["semantic_equivalence_attestation"] = attestation
        rows.append(generated)
    usable, rejected = corpus.classify_author_rows(rows, pair, _Counter())
    usable, rejected = pilot.apply_pair_author_target_rule(pair, usable, rejected)
    assert usable == []
    assert {row["reason"] for row in rejected} == {
        "semantic_target_mismatch",
        "paired_author_target_mismatch",
    }
    clean = []
    for slot in pair:
        generated = {
            "instruction": f"Choose fictional route {slot['task_id']}.",
            "state": {"summary": f"Rule {slot['task_id']} admits one match."},
            "criteria": [
                {"description": f"{slot['task_id']} option {criterion}."}
                for criterion in range(slot["option_count"])
            ],
            "selected_index": 0,
            "semantic_equivalence_attestation": {
                "scenario": slot["semantic_target"]["scenario"],
                "criterion_roles": slot["semantic_target"]["criterion_roles"],
                "selected_role": "matches_rule",
            },
        }
        clean.append(corpus.classify_author_rows([generated], [slot], _Counter())[0][0])
    reviews = [_review(clean[0]), _review(clean[1], fictional=False)]
    corpus_accepted, corpus_rejected = resolve_reviews(clean, reviews)
    assert corpus_accepted == []
    assert {row["reason"] for row in corpus_rejected} == {
        "review_quality_fictional",
        "paired_member_rejected",
    }
    pilot_accepted, pilot_rejected = resolve_reviews(
        clean,
        reviews,
        acceptance=pilot.pilot_acceptance,
        pair_resolution=pilot.pilot_pair_resolution,
    )
    assert len(pilot_accepted) == 1
    assert pilot_rejected[0]["reason"] == "review_quality_fictional"


def test_reviewer_view_omits_family_pair_and_answer_fields() -> None:
    plan = pilot.build_plan()
    slot = plan["slots"][0]
    row = {
        "task_id": slot["task_id"],
        "family_id": slot["family_id"],
        "pair_id": slot["pair_id"],
        "split": slot["split"],
        "locale": slot["locale"],
        "domain": slot["domain"],
        "instruction": "Choose a fictional route.",
        "state": {"summary": "One fictional fact."},
        "criteria": [
            {"id": "criterion-0", "description": "Matching fictional route."},
            {"id": "criterion-1", "description": "Other fictional route."},
        ],
        "selected_criterion_id": "criterion-0",
        "gold_position": slot["gold_position"],
        "semantic_target": slot["semantic_target"],
    }
    view = pilot.pilot_reviewer_view(row)
    assert set(view) == {"task_id", "locale", "domain", "instruction", "state", "criteria"}
    encoded = json.dumps(pilot.pilot_reviewer_messages(row))
    assert slot["family_id"] not in encoded
    assert slot["pair_id"] not in encoded
    assert "selected_criterion_id" not in encoded
    assert "gold_position" not in encoded
    assert "semantic_target" not in encoded


def test_open_reservations_count_toward_the_baseline(tmp_path: Path) -> None:
    root = _baseline_root(tmp_path / "root")
    scan = pilot.scan_research_ledgers(root)
    assert scan.directory_count == 46
    assert scan.entry_count == 2117
    assert scan.settled_count == 2099
    assert scan.open_reservation_count == 18
    assert scan.open_reservation_debit_usd == Decimal("0.01971612")
    assert scan.conservative_debit_usd == Decimal("5.12153043")
    assert scan.provider_reported_cost_usd == Decimal("0.81263107")
    pilot.require_pilot_research_history(root)


def test_inventory_rejects_an_unlisted_ledger_snapshot(tmp_path: Path) -> None:
    root = _baseline_root(tmp_path / "root")
    ledger_dir = root / "corpus-work-01" / "ledger"
    (ledger_dir / "ledger-0001.json").write_bytes(
        corpus._canonical(corpus.BudgetLedger().as_json()) + b"\n"
    )
    with pytest.raises(corpus.CorpusError, match="inventory does not match artifact root"):
        pilot.require_pilot_research_history(root)


def test_ledger_chain_rejects_a_newer_snapshot_that_hides_spend(tmp_path: Path) -> None:
    root = _baseline_root(tmp_path / "root")
    ledger_dir = root / "corpus-work-00" / "ledger"
    (ledger_dir / "ledger-0001.json").write_bytes(
        corpus._canonical(corpus.BudgetLedger().as_json()) + b"\n"
    )
    with pytest.raises(corpus.CorpusError, match="research ledger chain"):
        pilot.scan_research_ledgers(root)


def test_ledger_chain_rejects_a_missing_numbered_snapshot(tmp_path: Path) -> None:
    root = _baseline_root(tmp_path / "root")
    ledger_dir = root / "corpus-work-01" / "ledger"
    first = ledger_dir / "ledger-0000.json"
    (ledger_dir / "ledger-0001.json").write_bytes(first.read_bytes())
    first.unlink()
    with pytest.raises(corpus.CorpusError, match="ledger chain"):
        pilot.scan_research_ledgers(root)


def test_spend_below_baseline_refuses_before_transport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _baseline_root(tmp_path / "low", debit_delta=Decimal("-0.00000001"))
    plan = tmp_path / "pilot-plan-v6" / "plan.json"
    pilot.write_pilot_plan(plan)
    monkeypatch.setenv("OPENROUTER_API_KEY", "pilot-secret")

    def transport(*args: object, **kwargs: object) -> tuple[int, dict[str, str], bytes]:
        del args, kwargs
        raise AssertionError("transport")

    with pytest.raises(corpus.CorpusError, match="cumulative spend baseline"):
        pilot.run_pilot(
            plan,
            None,
            tmp_path / "pilot-work-v6",
            tmp_path / "pilot-report-v6",
            root,
            allow_network=True,
            transport=transport,
            counter=_Counter(),
        )


def test_pilot_without_opt_in_constructs_no_socket(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        socket,
        "socket",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("socket attempted")),
    )
    with pytest.raises(corpus.CorpusError, match="literal --allow-network"):
        pilot.run_pilot(
            tmp_path / "pilot-plan-v6" / "plan.json",
            None,
            tmp_path / "pilot-work-v6",
            tmp_path / "pilot-report-v6",
            tmp_path / "root",
            allow_network=False,
            counter=_Counter(),
        )


def test_pre_v6_paths_are_rejected_before_ledger_or_transport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = tmp_path / "pilot-plan-v6" / "plan.json"
    pilot.write_pilot_plan(plan)

    def forbidden(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("ledger or transport reached")

    monkeypatch.setattr(corpus, "resume_ledger", forbidden)
    cases = (
        (
            tmp_path / "pilot-plan-v3" / "plan.json",
            tmp_path / "pilot-work-v6",
            tmp_path / "pilot-report-v6",
        ),
        (
            tmp_path / "pilot-plan-v4" / "plan.json",
            tmp_path / "pilot-work-v6",
            tmp_path / "pilot-report-v6",
        ),
        (
            tmp_path / "pilot-plan-v5" / "plan.json",
            tmp_path / "pilot-work-v6",
            tmp_path / "pilot-report-v6",
        ),
        (plan, tmp_path / "pilot-work-v3", tmp_path / "pilot-report-v6"),
        (plan, tmp_path / "pilot-work-v4", tmp_path / "pilot-report-v6"),
        (plan, tmp_path / "pilot-work-v5", tmp_path / "pilot-report-v6"),
        (plan, tmp_path / "pilot-work-v6", tmp_path / "pilot-report-v3"),
        (plan, tmp_path / "pilot-work-v6", tmp_path / "pilot-report-v4"),
        (plan, tmp_path / "pilot-work-v6", tmp_path / "pilot-report-v5"),
        (
            tmp_path / "pilot-plan-v6" / ".." / "pilot-plan-v5" / "plan.json",
            tmp_path / "pilot-work-v6",
            tmp_path / "pilot-report-v6",
        ),
        (
            plan,
            tmp_path / "pilot-work-v6" / ".." / "pilot-work-v5",
            tmp_path / "pilot-report-v6",
        ),
        (
            plan,
            tmp_path / "pilot-work-v6",
            tmp_path / "pilot-report-v6" / ".." / "pilot-report-v5",
        ),
    )
    for plan_path, work_dir, report_dir in cases:
        with pytest.raises(corpus.CorpusError, match="pilot artifact path"):
            pilot.run_pilot(
                plan_path,
                None,
                work_dir,
                report_dir,
                tmp_path / "root",
                allow_network=True,
                transport=forbidden,
                counter=_Counter(),
            )


def test_import_is_create_only_and_digest_checked(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_ledger(source / "corpus-work-a", corpus.BudgetLedger())
    wilson = source / "corpus-work-a" / "ledger" / "wilson-0000.json"
    wilson.write_bytes(b'{"schema_version":"phase4e-wilson-ledger.v1"}\n')
    resolved = source / "corpus-work-a" / "resolved"
    resolved.mkdir()
    (resolved / "task-note.json").write_bytes(b'{"kind":"import"}\n')
    destination = tmp_path / "dest"
    first = pilot.import_research_artifacts(source, destination)
    assert (
        destination / "corpus-work-a" / "ledger" / "wilson-0000.json"
    ).read_bytes() == wilson.read_bytes()
    assert pilot.verify_inventory(destination) == first
    second = pilot.import_research_artifacts(source, destination)
    assert second == first
    wilson.write_bytes(b'{"schema_version":"phase4e-wilson-ledger.v1","changed":true}\n')
    with pytest.raises(FileExistsError):
        pilot.import_research_artifacts(source, destination)


def test_fake_transport_passes_with_diagnostic_semantic_disagreement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _baseline_root(tmp_path / "root")
    plan_path = tmp_path / "pilot-plan-v6" / "plan.json"
    pilot.write_pilot_plan(plan_path)
    plan = json.loads(plan_path.read_bytes())
    by_id = {slot["task_id"]: slot for slot in plan["slots"]}
    monkeypatch.setenv("OPENROUTER_API_KEY", "pilot-secret")
    monkeypatch.setattr(
        corpus,
        "seal_packet",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("packet")),
    )
    monkeypatch.setattr(
        corpus,
        "stop_projection",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("wilson")),
    )
    reviewer_calls = 0
    blinded = {"ok": False}

    def transport(
        method: str, url: str, headers: object, body: bytes
    ) -> tuple[int, dict[str, str], bytes]:
        nonlocal reviewer_calls
        del method, url, headers
        message = json.loads(json.loads(body)["messages"][1]["content"])
        if "slots" in message:
            return _completion(_author_wire(message["slots"]))
        reviewer_calls += 1
        task = message["tasks"][0]
        slot = by_id[task["task_id"]]
        assert set(task) == {"task_id", "locale", "domain", "instruction", "state", "criteria"}
        assert slot["family_id"] not in body.decode()
        assert slot["pair_id"] not in body.decode()
        sibling = next(
            other["task_id"]
            for other in plan["slots"]
            if other["pair_id"] == slot["pair_id"] and other["task_id"] != slot["task_id"]
        )
        assert sibling not in body.decode()
        blinded["ok"] = True
        return _completion(
            _reviewer_payload(slot, disagree_answer=False, disagree_scenario=reviewer_calls == 1)
        )

    report_path = pilot.run_pilot(
        plan_path,
        None,
        tmp_path / "pilot-work-v6",
        tmp_path / "pilot-report-v6",
        root,
        allow_network=True,
        transport=transport,
        counter=_Counter(),
    )
    report = json.loads(report_path.read_bytes())
    assert blinded["ok"] is True
    assert reviewer_calls == 140
    assert report["decision"] == "PASS"
    assert report["accepted_count"] == 140
    assert report["unresolved_count"] == 0
    assert report["semantic_diagnostics"]["scenario_disagreement"] == 1
    assert report["complete_pairs"] == 70
    assert report["local_privacy_violations"] == 0
    assert report["reviewer_privacy_flags"] == 0
    assert Decimal(report["cumulative_conservative_debit_usd"]) > Decimal("5.12153043")
    assert report["artifact_root"] == str(root.resolve())
    assert report["inventory_sha256"] == pilot.verify_inventory(root)
    assert "packet" not in pilot.run_pilot.__code__.co_varnames


def test_settled_malformed_review_retries_and_cost_only_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _baseline_root(tmp_path / "root")
    plan_path = tmp_path / "pilot-plan-v6" / "plan.json"
    pilot.write_pilot_plan(plan_path)
    plan = json.loads(plan_path.read_bytes())
    by_id = {slot["task_id"]: slot for slot in plan["slots"]}
    monkeypatch.setenv("OPENROUTER_API_KEY", "pilot-secret")
    author_calls = 0
    reviewer_calls = 0

    def transport(
        method: str, url: str, headers: object, body: bytes
    ) -> tuple[int, dict[str, str], bytes]:
        nonlocal author_calls, reviewer_calls
        del method, url, headers
        message = json.loads(json.loads(body)["messages"][1]["content"])
        if "slots" in message:
            author_calls += 1
            wire = _author_wire(message["slots"])
            if author_calls == 1:
                wire["record_0_instruction"] = "x" * 121
            return _completion(wire, cost="0.10")
        reviewer_calls += 1
        if reviewer_calls == 1:
            return _raw_completion(
                '{"reviews":[{"generic_or_invented":true,"generic_or_invented":false}]}',
                cost="0.10",
            )
        slot = by_id[message["tasks"][0]["task_id"]]
        return _completion(
            _reviewer_payload(slot, disagree_answer=False, disagree_scenario=False),
            cost="0.10",
        )

    report = json.loads(
        pilot.run_pilot(
            plan_path,
            None,
            tmp_path / "pilot-work-v6",
            tmp_path / "pilot-report-v6",
            root,
            allow_network=True,
            transport=transport,
            counter=_Counter(),
        ).read_bytes()
    )
    assert author_calls == 71
    assert reviewer_calls == 141
    assert report["decision"] == "PASS"
    assert report["accepted_count"] == 140
    assert report["response_diagnostics"] == {
        "author_failures": 1,
        "reviewer_failures": 1,
        "retry_recoveries": 2,
        "orphaned_settled_calls": 0,
    }
    assert Decimal(report["provider_reported_cost_usd"]) == Decimal("21.20")
    output = capsys.readouterr().out
    assert "crossed USD 10.00" in output
    assert "crossed USD 20.00" in output
    assert "final provider spend USD 21.20" in output


def test_settled_http_error_is_not_retried_and_other_pairs_continue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _baseline_root(tmp_path / "root")
    plan_path = tmp_path / "pilot-plan-v6" / "plan.json"
    pilot.write_pilot_plan(plan_path)
    plan = json.loads(plan_path.read_bytes())
    by_id = {slot["task_id"]: slot for slot in plan["slots"]}
    monkeypatch.setenv("OPENROUTER_API_KEY", "pilot-secret")
    author_calls = 0
    reviewer_calls = 0

    def transport(
        method: str, url: str, headers: object, body: bytes
    ) -> tuple[int, dict[str, str], bytes]:
        nonlocal author_calls, reviewer_calls
        del method, url, headers
        message = json.loads(json.loads(body)["messages"][1]["content"])
        if "slots" in message:
            author_calls += 1
            completion = _completion(_author_wire(message["slots"]))
            if author_calls == 1:
                return 500, completion[1], completion[2]
            return completion
        reviewer_calls += 1
        slot = by_id[message["tasks"][0]["task_id"]]
        return _completion(_reviewer_payload(slot, disagree_answer=False, disagree_scenario=False))

    report = json.loads(
        pilot.run_pilot(
            plan_path,
            None,
            tmp_path / "pilot-work-v6",
            tmp_path / "pilot-report-v6",
            root,
            allow_network=True,
            transport=transport,
            counter=_Counter(),
        ).read_bytes()
    )
    assert author_calls == 70
    assert reviewer_calls == 138
    assert report["decision"] == "PASS"
    assert report["accepted_count"] == 138
    assert report["rejected_count"] == 2
    assert report["unresolved_count"] == 0
    assert report["response_diagnostics"]["author_failures"] == 1
    assert report["response_diagnostics"]["retry_recoveries"] == 0


def test_fake_transport_resolves_all_140_without_wilson_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _baseline_root(tmp_path / "root")
    plan_path = tmp_path / "pilot-plan-v6" / "plan.json"
    pilot.write_pilot_plan(plan_path)
    plan = json.loads(plan_path.read_bytes())
    by_id = {slot["task_id"]: slot for slot in plan["slots"]}
    monkeypatch.setenv("OPENROUTER_API_KEY", "pilot-secret")
    monkeypatch.setattr(
        corpus,
        "stop_projection",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("wilson")),
    )
    calls = 0

    def transport(
        method: str, url: str, headers: object, body: bytes
    ) -> tuple[int, dict[str, str], bytes]:
        nonlocal calls
        del method, url, headers
        calls += 1
        message = json.loads(json.loads(body)["messages"][1]["content"])
        if "slots" in message:
            return _completion(_author_wire(message["slots"]))
        slot = by_id[message["tasks"][0]["task_id"]]
        return _completion(_reviewer_payload(slot, disagree_answer=True, disagree_scenario=False))

    report = json.loads(
        pilot.run_pilot(
            plan_path,
            None,
            tmp_path / "pilot-work-v6",
            tmp_path / "pilot-report-v6",
            root,
            allow_network=True,
            transport=transport,
            counter=_Counter(),
        ).read_bytes()
    )
    assert calls == 210
    assert report["decision"] == "FAIL"
    assert report["accepted_count"] == 0
    assert report["rejected_count"] == 140
    assert report["unresolved_count"] == 0
    corpus_plan = corpus.build_plan()
    statuses = [
        {"task_id": slot["task_id"], "status": "rejected"} for slot in corpus_plan["slots"][:20]
    ]
    monkeypatch.undo()
    assert corpus.stop_projection(statuses, corpus_plan) == "wilson_projected_minimum"


def test_uncertain_transport_is_terminal_and_inconclusive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _baseline_root(tmp_path / "root")
    plan_path = tmp_path / "pilot-plan-v6" / "plan.json"
    pilot.write_pilot_plan(plan_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "pilot-secret")
    calls = 0

    def transport(*args: object, **kwargs: object) -> tuple[int, dict[str, str], bytes]:
        nonlocal calls
        del args, kwargs
        calls += 1
        raise RuntimeError("rate limited")

    work = tmp_path / "pilot-work-v6"
    first = json.loads(
        pilot.run_pilot(
            plan_path,
            None,
            work,
            tmp_path / "first" / "pilot-report-v6",
            root,
            allow_network=True,
            transport=transport,
            counter=_Counter(),
        ).read_bytes()
    )
    assert calls == 1
    assert first["decision"] == "INCONCLUSIVE"
    calls = 0
    second = json.loads(
        pilot.run_pilot(
            plan_path,
            None,
            work,
            tmp_path / "second" / "pilot-report-v6",
            root,
            allow_network=True,
            transport=transport,
            counter=_Counter(),
        ).read_bytes()
    )
    assert calls == 0
    assert second["decision"] == "INCONCLUSIVE"


def test_resume_never_replays_a_settled_call_without_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _baseline_root(tmp_path / "root")
    plan_path = tmp_path / "pilot-plan-v6" / "plan.json"
    pilot.write_pilot_plan(plan_path)
    plan = json.loads(plan_path.read_bytes())
    pair = plan["slots"][:2]
    work = tmp_path / "pilot-work-v6"
    ledger = corpus.BudgetLedger(policy=corpus.PILOT_LEDGER_POLICY)
    reservation_id = "reservation-" + "a" * 64
    ledger.reserve_request(pilot.AUTHOR_STAGE, reservation_id, Decimal("0.01"))
    ledger.settle_request(reservation_id, "settled-request", Decimal("0.001"), "b" * 64)
    ledger.record_provider_journal(
        stage=pilot.AUTHOR_STAGE,
        reservation_id=reservation_id,
        task_ids=[slot["task_id"] for slot in pair],
    )
    corpus.write_ledger_snapshot(work / "ledger", ledger)
    monkeypatch.setenv("OPENROUTER_API_KEY", "pilot-secret")
    calls = 0

    def forbidden_transport(*args: object, **kwargs: object) -> tuple[int, dict[str, str], bytes]:
        nonlocal calls
        del args, kwargs
        calls += 1
        raise AssertionError("orphaned settlement must not be replayed")

    report = json.loads(
        pilot.run_pilot(
            plan_path,
            None,
            work,
            tmp_path / "pilot-report-v6",
            root,
            allow_network=True,
            transport=forbidden_transport,
            counter=_Counter(),
        ).read_bytes()
    )
    assert calls == 0
    assert report["decision"] == "INCONCLUSIVE"
    assert report["unresolved_count"] == 140
    assert report["response_diagnostics"]["orphaned_settled_calls"] == 1


def test_resume_restores_persisted_diagnostics_without_another_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _baseline_root(tmp_path / "root")
    plan_path = tmp_path / "pilot-plan-v6" / "plan.json"
    pilot.write_pilot_plan(plan_path)
    plan = json.loads(plan_path.read_bytes())
    by_id = {slot["task_id"]: slot for slot in plan["slots"]}
    monkeypatch.setenv("OPENROUTER_API_KEY", "pilot-secret")
    author_calls = 0
    reviewer_calls = 0

    def first_transport(
        method: str, url: str, headers: object, body: bytes
    ) -> tuple[int, dict[str, str], bytes]:
        nonlocal author_calls, reviewer_calls
        del method, url, headers
        message = json.loads(json.loads(body)["messages"][1]["content"])
        if "slots" in message:
            author_calls += 1
            if author_calls == 2:
                raise RuntimeError("stop after one completed pair")
            return _completion(_author_wire(message["slots"]))
        reviewer_calls += 1
        slot = by_id[message["tasks"][0]["task_id"]]
        return _completion(
            _reviewer_payload(
                slot,
                disagree_answer=False,
                disagree_scenario=reviewer_calls == 1,
            )
        )

    work = tmp_path / "pilot-work-v6"
    first = json.loads(
        pilot.run_pilot(
            plan_path,
            None,
            work,
            tmp_path / "first" / "pilot-report-v6",
            root,
            allow_network=True,
            transport=first_transport,
            counter=_Counter(),
        ).read_bytes()
    )
    assert author_calls == 2
    assert reviewer_calls == 2
    assert first["decision"] == "INCONCLUSIVE"
    assert first["semantic_diagnostics"] == {
        "reviewed": 2,
        "scenario_disagreement": 1,
        "criterion_role_disagreement": 0,
    }

    def forbidden_transport(*args: object, **kwargs: object) -> tuple[int, dict[str, str], bytes]:
        del args, kwargs
        raise AssertionError("terminal resume must not call the provider")

    second = json.loads(
        pilot.run_pilot(
            plan_path,
            None,
            work,
            tmp_path / "second" / "pilot-report-v6",
            root,
            allow_network=True,
            transport=forbidden_transport,
            counter=_Counter(),
        ).read_bytes()
    )
    assert second["decision"] == "INCONCLUSIVE"
    assert second["semantic_diagnostics"] == first["semantic_diagnostics"]


def test_orphan_diagnostics_fail_closed_without_transport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _baseline_root(tmp_path / "root")
    plan_path = tmp_path / "pilot-plan-v6" / "plan.json"
    pilot.write_pilot_plan(plan_path)
    work = tmp_path / "pilot-work-v6"
    diagnostics = work / "diagnostics"
    diagnostics.mkdir(parents=True)
    (diagnostics / ("call-" + "0" * 64 + ".json")).write_text("{}\n")
    monkeypatch.setenv("OPENROUTER_API_KEY", "pilot-secret")

    def forbidden_transport(*args: object, **kwargs: object) -> tuple[int, dict[str, str], bytes]:
        del args, kwargs
        raise AssertionError("orphan diagnostics must stop before transport")

    report = json.loads(
        pilot.run_pilot(
            plan_path,
            None,
            work,
            tmp_path / "pilot-report-v6",
            root,
            allow_network=True,
            transport=forbidden_transport,
            counter=_Counter(),
        ).read_bytes()
    )
    assert report["decision"] == "INCONCLUSIVE"
    assert report["unresolved_count"] == 140


def test_canonical_research_root_uses_platformdirs(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    import types

    module = types.ModuleType("platformdirs")

    def user_state_path(name: str) -> str:
        assert name == "saracura"
        return "/tmp/saracura-state"

    module.user_state_path = user_state_path  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "platformdirs", module)
    assert pilot.canonical_research_ledger_root() == Path(
        "/tmp/saracura-state/phase4e/research-ledgers"
    )
