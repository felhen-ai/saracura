"""Adversarial, network-free checks for the Phase 4E.2a corpus lane."""

from __future__ import annotations

import json
import socket
from collections.abc import Mapping
from decimal import Decimal
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import ValidationError

from benchmarks import saracura_universal_corpus as corpus
from benchmarks.saracura_universal_corpus import (
    LOCALES,
    OPTION_COUNTS,
    SPLITS,
    AcceptedPacketRow,
    BudgetLedger,
    CorpusError,
    OpenRouterCorpusClient,
    ValidatedAuthorRow,
    author_messages,
    author_schema,
    build_plan,
    classify_author_rows,
    resolve_reviews,
    resume_ledger,
    reviewer_messages,
    semantic_fingerprint,
    stop_projection,
    validate_author_rows,
    validate_author_rows_with_verified_minilm,
    validate_plan,
    write_ledger_snapshot,
)


class _Counter:
    def __init__(self, count: int = 1) -> None:
        self.count_value = count

    def count(self, text: str) -> int:
        del text
        return self.count_value


def _record(slot: Mapping[str, Any]) -> dict[str, Any]:
    criteria = [
        {"id": f"route-{index}", "description": f"Route criterion {index}."}
        for index in range(int(slot["option_count"]))
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
        "instruction": "Choose the safest fictional route.",
        "state": {"summary": "A fictional request."},
        "criteria": criteria,
        "selected_criterion_id": criteria[int(slot["gold_position"])]["id"],
    }
    if isinstance(slot["pair_id"], str):
        roles = [f"role-{index}" for index in range(int(slot["option_count"]))]
        record["cross_locale_attestation"] = {
            "pair_id": slot["pair_id"],
            "scenario": "shared fictional routing scenario",
            "criterion_roles": roles,
            "selected_role": roles[int(slot["gold_position"])],
        }
    return record


def _review(row: Mapping[str, Any], **overrides: Any) -> dict[str, Any]:
    criteria = row["criteria"]
    selected_id = row["selected_criterion_id"]
    selected_position = next(
        index for index, item in enumerate(criteria) if item["id"] == selected_id
    )
    author_attestation = row.get("cross_locale_attestation")
    if isinstance(author_attestation, Mapping):
        semantic_attestation = {
            key: author_attestation[key] for key in ("scenario", "criterion_roles", "selected_role")
        }
    else:
        roles = [f"role-{index}" for index in range(len(criteria))]
        semantic_attestation = {
            "scenario": "fictional routing scenario",
            "criterion_roles": roles,
            "selected_role": roles[selected_position],
        }
    return {
        "task_id": row["task_id"],
        "status": "accepted",
        "selected_criterion_id": selected_id,
        "reason_codes": [],
        "natural_language": True,
        "fictional": True,
        "exclusive_options": True,
        "private_or_sensitive": False,
        "semantic_equivalence_attestation": semantic_attestation,
        **overrides,
    }


def _paired_records(slots: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    records = [_record(slot) for slot in slots]
    english = next(record for record in records if record["locale"] == "en")
    english["instruction"] = "Choose the safest fictional path in this situation."
    english["state"] = {"summary": "An invented scenario needing a route."}
    english["criteria"] = [
        {**criterion, "description": f"English route option {index}."}
        for index, criterion in enumerate(english["criteria"])
    ]
    return records


def _accepted_row(slot: Mapping[str, Any], **overrides: Any) -> dict[str, Any]:
    record = {**_record(slot), **overrides}
    row = validate_author_rows([record], [slot], _Counter())[0]
    accepted, rejected = resolve_reviews([row], [_review(row)])
    assert rejected == []
    return accepted[0]


def _complete_packet_resolution(
    plan: Mapping[str, Any], accepted: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accepted_ids = {row["task_id"] for row in accepted}
    rejected = [
        {"task_id": slot["task_id"], "split": slot["split"], "reason": "review_disagreement"}
        for slot in plan["slots"]
        if slot["task_id"] not in accepted_ids
    ]
    return accepted, rejected


def test_plan_is_preprovider_text_free_balanced_and_cross_locale_isolated() -> None:
    plan = build_plan()
    validate_plan(plan)
    assert len(plan["slots"]) == 1600
    assert not {"instruction", "state", "criteria", "selected_criterion_id"}.intersection(
        plan["slots"][0]
    )
    for split in SPLITS:
        for locale in LOCALES:
            for count in OPTION_COUNTS:
                assert any(
                    row["split"] == split
                    and row["locale"] == locale
                    and row["option_count"] == count
                    for row in plan["slots"]
                )
    pairs: dict[str, list[dict[str, Any]]] = {}
    for row in plan["slots"]:
        if row["pair_id"]:
            pairs.setdefault(row["pair_id"], []).append(row)
    assert len(pairs) == 120
    assert all({row["locale"] for row in pair} == {"pt-BR", "en"} for pair in pairs.values())


def test_author_capacity_gate_and_reviewer_blindness_are_fail_closed() -> None:
    slots = build_plan()["slots"][:2]
    rows = validate_author_rows([_record(slot) for slot in slots], slots, _Counter())
    reviewer_payload = json.dumps(
        reviewer_messages([ValidatedAuthorRow.model_validate(rows[0])]), ensure_ascii=False
    )
    assert "gold_position" not in reviewer_payload
    assert "selected_criterion_id" not in reviewer_payload
    assert "synthetic_train" not in reviewer_payload
    with pytest.raises(CorpusError, match="tokenizer capacity"):
        validate_author_rows([_record(slot) for slot in slots], slots, _Counter(129))
    usable, rejected = classify_author_rows([_record(slot) for slot in slots], slots, _Counter(129))
    assert usable == []
    assert rejected == [
        {"task_id": slot["task_id"], "split": slot["split"], "reason": "capacity"} for slot in slots
    ]

    changed = _record(slots[0])
    changed["split"] = "synthetic_dev"
    with pytest.raises(CorpusError, match="author record schema"):
        validate_author_rows([changed, _record(slots[1])], slots, _Counter())
    assert "gold_position" in json.dumps(author_messages(slots))


def test_reviewer_blindness_is_structural_and_preserves_legitimate_text() -> None:
    slot = next(slot for slot in build_plan()["slots"] if slot["pair_id"] is None)
    author_row = _record(slot)
    forbidden_words = "split axes pair_id gold_position selected_criterion_id"
    author_row["instruction"] = f"Instruction mentions {forbidden_words}."
    author_row["state"] = {"note": f"State mentions {forbidden_words}."}
    author_row["criteria"][0]["description"] = f"Criterion mentions {forbidden_words}."
    row = ValidatedAuthorRow.model_validate(
        validate_author_rows([author_row], [slot], _Counter())[0]
    )

    payload = json.loads(reviewer_messages([row])[1]["content"])
    task = payload["tasks"][0]
    assert set(task) == {
        "task_id",
        "family_id",
        "locale",
        "domain",
        "instruction",
        "state",
        "criteria",
    }
    assert forbidden_words in task["instruction"]
    assert forbidden_words in task["state"]["note"]
    assert forbidden_words in task["criteria"][0]["description"]
    assert "selected_criterion_id" not in task
    assert "split" not in task
    assert "pair_id" not in task
    assert "axes" not in task
    assert "gold_position" not in task

    cast(dict[str, Any], row.state)["invalid"] = object()
    with pytest.raises(CorpusError, match="reviewer transport value"):
        reviewer_messages([row])


def test_verified_minilm_capacity_classification_resolves_every_preassigned_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    slots = build_plan()["slots"][:3]
    monkeypatch.setattr(
        corpus.VerifiedMiniLMTokenizerReceipt,
        "create",
        classmethod(lambda cls, snapshot: _Counter(129)),
    )
    usable, rejected = validate_author_rows_with_verified_minilm(
        [_record(slot) for slot in slots], slots, Path("unused-verified-snapshot")
    )
    assert usable == []
    assert rejected == [
        {"task_id": slot["task_id"], "split": slot["split"], "reason": "capacity"} for slot in slots
    ]


def test_author_schema_requires_exact_planned_cardinality_and_full_cross_locale_pair() -> None:
    plan = build_plan()
    pair_id = next(slot["pair_id"] for slot in plan["slots"] if slot["pair_id"])
    pair = [slot for slot in plan["slots"] if slot["pair_id"] == pair_id]
    schema = author_schema(pair)
    records = schema["properties"]["records"]
    assert records["minItems"] == records["maxItems"] == 2
    assert records["items"] is False
    assert [item["properties"]["locale"]["const"] for item in records["prefixItems"]] == [
        "pt-BR",
        "en",
    ]
    with pytest.raises(CorpusError, match="cross-locale author batch"):
        author_schema(pair[:1])


def test_review_rejects_disagreement_privacy_and_normalized_duplicates() -> None:
    slots = build_plan()["slots"][:2]
    rows = validate_author_rows([_record(slot) for slot in slots], slots, _Counter())
    reviews = [_review(rows[0]), _review(rows[1], selected_criterion_id="route-0")]
    accepted, rejected = resolve_reviews(rows, reviews)
    assert len(accepted) == 1
    assert rejected[0]["task_id"] == rows[1]["task_id"]
    assert rejected[0]["split"] == rows[1]["split"]
    assert rejected[0]["reason"] == "review_disagreement"
    assert set(rejected[0]) == {
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


def test_semantic_fingerprint_rejects_criterion_permutation_and_renaming_globally() -> None:
    slots = build_plan()["slots"][:2]
    original = validate_author_rows([_record(slots[0])], [slots[0]], _Counter())[0]
    candidate = {
        **validate_author_rows([_record(slots[1])], [slots[1]], _Counter())[0],
        "family_id": "family-" + "f" * 64,
        "split": "synthetic_holdout",
        "instruction": original["instruction"],
        "state": original["state"],
        "criteria": [
            {"id": f"renamed-{index}", "description": criterion["description"]}
            for index, criterion in enumerate(reversed(original["criteria"]))
        ],
    }
    selected_description = next(
        criterion["description"]
        for criterion in original["criteria"]
        if criterion["id"] == original["selected_criterion_id"]
    )
    candidate["selected_criterion_id"] = next(
        criterion["id"]
        for criterion in candidate["criteria"]
        if criterion["description"] == selected_description
    )
    review = _review(candidate)
    assert semantic_fingerprint(candidate) == semantic_fingerprint(original)
    accepted, rejected = resolve_reviews([candidate], [review], prior_rows=[original])
    assert accepted == []
    assert rejected[0]["task_id"] == candidate["task_id"]
    assert rejected[0]["split"] == "synthetic_holdout"
    assert rejected[0]["reason"] == "semantic_duplicate"
    assert rejected[0]["author_response_sha256"]
    assert rejected[0]["reviewer_response_sha256"]


def test_accepted_rows_are_closed_synthetic_and_bound_to_a_matching_review() -> None:
    slots = build_plan()["slots"][:1]
    rows = validate_author_rows([_record(slot) for slot in slots], slots, _Counter())
    reviews = [_review(rows[0])]
    accepted, rejected = resolve_reviews(rows, reviews)
    assert rejected == []
    AcceptedPacketRow.model_validate(accepted[0])
    for field, value in (
        ("synthetic_only", False),
        ("review_sha256", "0" * 64),
        ("unexpected", True),
    ):
        invalid = {**accepted[0], field: value}
        with pytest.raises(ValidationError):
            AcceptedPacketRow.model_validate(invalid)
    different_criterion = next(
        criterion["id"]
        for criterion in accepted[0]["criteria"]
        if criterion["id"] != accepted[0]["selected_criterion_id"]
    )
    mismatched = {
        **accepted[0],
        "review": {**accepted[0]["review"], "selected_criterion_id": different_criterion},
    }
    with pytest.raises(ValidationError):
        AcceptedPacketRow.model_validate(mismatched)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("natural_language", False),
        ("fictional", False),
        ("exclusive_options", False),
        ("private_or_sensitive", True),
    ],
)
def test_final_packet_validation_requires_every_positive_review_certification(
    field: str, value: bool
) -> None:
    plan = build_plan()
    slot = next(slot for slot in plan["slots"] if slot["pair_id"] is None)
    accepted = _accepted_row(slot)
    review = {**accepted["review"], field: value}
    invalid = {
        **accepted,
        "review": review,
        "review_sha256": corpus._sha(corpus._canonical(review)),
    }
    rows, rejected = _complete_packet_resolution(plan, [invalid])
    with pytest.raises(CorpusError, match="accepted packet row schema"):
        corpus._validate_packet_resolution(rows, rejected, plan)


def test_final_packet_validation_binds_pair_and_attestation_to_planned_slot() -> None:
    plan = build_plan()
    pair_id = next(slot["pair_id"] for slot in plan["slots"] if slot["pair_id"])
    slots = [slot for slot in plan["slots"] if slot["pair_id"] == pair_id]
    rows = validate_author_rows(_paired_records(slots), slots, _Counter())
    accepted, rejected = resolve_reviews(rows, [_review(row) for row in rows])
    assert rejected == []
    assert len(accepted) == 2

    unattested = [{**accepted[0], "cross_locale_attestation": None}, accepted[1]]
    packet_rows, packet_rejected = _complete_packet_resolution(plan, unattested)
    with pytest.raises(CorpusError, match="accepted packet row schema"):
        corpus._validate_packet_resolution(packet_rows, packet_rejected, plan)

    tampered = [
        {
            **accepted[0],
            "pair_id": None,
            "cross_locale_attestation": None,
        },
        accepted[1],
    ]
    packet_rows, packet_rejected = _complete_packet_resolution(plan, tampered)
    with pytest.raises(CorpusError, match="packet split or family mutation"):
        corpus._validate_packet_resolution(packet_rows, packet_rejected, plan)


def test_final_packet_validation_rejects_normalized_and_near_duplicates_globally() -> None:
    plan = build_plan()
    source_slot = next(slot for slot in plan["slots"] if slot["pair_id"] is None)
    comparable_slot = next(
        slot
        for slot in plan["slots"]
        if slot["pair_id"] is None
        and slot["split"] == "synthetic_dev"
        and slot["option_count"] == source_slot["option_count"]
        and slot["gold_position"] == source_slot["gold_position"]
    )
    source = _accepted_row(source_slot, state={"summary": "A fictional Ü request."})
    normalized_duplicate = _accepted_row(
        comparable_slot,
        state={"summary": "A fictional ü request."},
    )
    assert semantic_fingerprint(source) != semantic_fingerprint(normalized_duplicate)
    assert corpus._normalized(source) == corpus._normalized(normalized_duplicate)
    rows, rejected = _complete_packet_resolution(plan, [source, normalized_duplicate])
    with pytest.raises(CorpusError, match="packet normalized duplicate"):
        corpus._validate_packet_resolution(rows, rejected, plan)

    near_duplicate = _accepted_row(
        comparable_slot,
        state={"summary": "A fictional Ü request."},
        instruction="Choose the safest fictional route right now.",
    )
    assert semantic_fingerprint(source) != semantic_fingerprint(near_duplicate)
    assert (
        SequenceMatcher(
            None, corpus._normalized(source), corpus._normalized(near_duplicate)
        ).ratio()
        >= 0.92
    )
    rows, rejected = _complete_packet_resolution(plan, [source, near_duplicate])
    with pytest.raises(CorpusError, match="packet near duplicate"):
        corpus._validate_packet_resolution(rows, rejected, plan)


def test_cross_locale_acceptance_requires_matching_author_and_reviewer_attestations() -> None:
    plan = build_plan()
    pair_id = next(slot["pair_id"] for slot in plan["slots"] if slot["pair_id"])
    slots = [slot for slot in plan["slots"] if slot["pair_id"] == pair_id]
    rows = validate_author_rows(_paired_records(slots), slots, _Counter())

    accepted, rejected = resolve_reviews(rows, [_review(row) for row in rows])
    assert len(accepted) == 2
    assert rejected == []
    for row in accepted:
        AcceptedPacketRow.model_validate(row)

    tampered = [_review(row) for row in rows]
    tampered[1]["semantic_equivalence_attestation"] = {
        **tampered[1]["semantic_equivalence_attestation"],
        "scenario": "a different fictional scenario",
    }
    accepted, rejected = resolve_reviews(rows, tampered)
    assert accepted == []
    assert {row["reason"] for row in rejected} == {"cross_locale_semantic_attestation"}
    assert {row["task_id"] for row in rejected} == {row["task_id"] for row in rows}


def test_budget_uses_larger_debit_and_never_borrows_stage_budget() -> None:
    ledger = BudgetLedger()
    ledger.record("corpus_author", "request-a", Decimal("0.01"), Decimal("0.02"), "a" * 64)
    assert ledger.spent("corpus_author") == Decimal("0.02")
    with pytest.raises(CorpusError, match="stage budget"):
        ledger.reserve("corpus_author", Decimal("1.99"))
    with pytest.raises(CorpusError, match="network"):
        OpenRouterCorpusClient()


def test_packet_lineage_binds_accepted_and_rejected_rows_to_settled_journal() -> None:
    slots = [slot for slot in build_plan()["slots"] if slot["pair_id"] is None][:2]
    rows = validate_author_rows([_record(slot) for slot in slots], slots, _Counter())
    ledger = BudgetLedger()
    author_reservation = "reservation-" + "a" * 64
    reviewer_reservations = {
        row["task_id"]: "reservation-" + format(index + 1, "x") * 64
        for index, row in enumerate(rows)
    }
    ledger.reserve_request("corpus_author", author_reservation, Decimal("0.01"))
    ledger.settle_request(author_reservation, "author-request", Decimal(), "b" * 64)
    ledger.record_provider_journal(
        stage="corpus_author",
        reservation_id=author_reservation,
        task_ids=[row["task_id"] for row in rows],
    )
    author_lineage = corpus.lineage_from_journal(ledger.provider_journal[0], "author")
    journalled_rows = [{**row, **author_lineage} for row in rows]
    reviewer_lineages: dict[str, dict[str, str]] = {}
    for index, row in enumerate(rows):
        reservation_id = reviewer_reservations[row["task_id"]]
        ledger.reserve_request("corpus_reviewer", reservation_id, Decimal("0.01"))
        ledger.settle_request(
            reservation_id,
            f"reviewer-request-{index}",
            Decimal(),
            format(index + 2, "x") * 64,
        )
        ledger.record_provider_journal(
            stage="corpus_reviewer", reservation_id=reservation_id, task_ids=[row["task_id"]]
        )
        reviewer_lineages[row["task_id"]] = corpus.lineage_from_journal(
            ledger.provider_journal[-1], "reviewer"
        )
    disagreement = next(
        criterion["id"]
        for criterion in journalled_rows[1]["criteria"]
        if criterion["id"] != journalled_rows[1]["selected_criterion_id"]
    )
    reviews = [
        _review(journalled_rows[0]),
        _review(journalled_rows[1], selected_criterion_id=disagreement),
    ]
    accepted, rejected = resolve_reviews(
        journalled_rows, reviews, reviewer_lineages=reviewer_lineages
    )
    corpus._validate_provider_lineage(
        ledger, accepted, rejected, {row["task_id"] for row in journalled_rows}
    )
    tampered = [{**accepted[0], "author_response_sha256": "0" * 64}]
    with pytest.raises(CorpusError, match="accepted provider lineage"):
        corpus._validate_provider_lineage(
            ledger, tampered, rejected, {row["task_id"] for row in journalled_rows}
        )


def test_ledger_resume_snapshots_are_append_only(tmp_path: Path) -> None:
    ledger = BudgetLedger()
    ledger.record("corpus_author", "request-a", Decimal("0.01"), Decimal("0.01"), "a" * 64)
    first = write_ledger_snapshot(tmp_path, ledger)
    ledger.record("corpus_reviewer", "request-b", Decimal("0.01"), Decimal("0.01"), "b" * 64)
    second = write_ledger_snapshot(tmp_path, ledger)
    assert first.read_bytes() != second.read_bytes()
    assert first.name == "ledger-0000.json" and second.name == "ledger-0001.json"


def test_reservation_uses_complete_body_including_response_schema(tmp_path: Path) -> None:
    ledger_directory = tmp_path / "ledger"
    captured: list[tuple[dict[str, Any], dict[str, str]]] = []

    def transport(
        method: str, url: str, headers: Mapping[str, str], body: bytes
    ) -> tuple[int, dict[str, str], bytes]:
        del method, url, headers
        snapshot = resume_ledger(ledger_directory)
        captured.append((json.loads(body), snapshot.entries[-1]))
        request_id = f"request-{len(captured)}"
        return 200, {}, json.dumps({"id": request_id, "usage": {"cost": 0}}).encode()

    client = OpenRouterCorpusClient(
        transport=transport, allow_network=True, ledger_directory=ledger_directory
    )
    slots = build_plan()["slots"]
    small = next(slot for slot in slots if slot["pair_id"] is None and slot["option_count"] == 2)
    large = next(slot for slot in slots if slot["pair_id"] is None and slot["option_count"] == 8)
    for slot in (small, large):
        client.author(
            slots=[slot],
            max_output_tokens=1,
            api_key="test-only",
        )

    assert captured[0][0]["response_format"] != captured[1][0]["response_format"]
    assert captured[0][1]["reservation_id"] != captured[1][1]["reservation_id"]
    assert Decimal(captured[0][1]["local_worst_case_usd"]) != Decimal(
        captured[1][1]["local_worst_case_usd"]
    )
    for body, reservation in captured:
        expected = corpus.request_worst_case(
            "corpus_author", corpus._canonical(body), max_output_tokens=1
        )
        assert Decimal(reservation["local_worst_case_usd"]) == expected


def test_reported_cost_and_terminal_overspend_are_persisted_before_raising(tmp_path: Path) -> None:
    slot = next(slot for slot in build_plan()["slots"] if slot["pair_id"] is None)

    def transport(
        method: str, url: str, headers: Mapping[str, str], body: bytes
    ) -> tuple[int, dict[str, str], bytes]:
        del method, url, headers, body
        return 429, {}, b'{"id":"charged-request","usage":{"cost":"2.01"}}'

    ledger_directory = tmp_path / "ledger"
    client = OpenRouterCorpusClient(
        transport=transport, allow_network=True, ledger_directory=ledger_directory
    )
    with pytest.raises(CorpusError, match="reported stage budget exhausted"):
        client.author(
            slots=[slot],
            max_output_tokens=1,
            api_key="test-only",
        )
    snapshot = json.loads(sorted(ledger_directory.glob("ledger-*.json"))[-1].read_bytes())
    entry = snapshot["entries"][0]
    assert entry["provider_cost_usd"] == "2.01"
    assert entry["debit_usd"] == "2.01"
    assert entry["status"] == "overspent"
    assert snapshot["stop_reason"] == "reported_provider_overspend"
    with pytest.raises(CorpusError, match="terminal overspend"):
        resume_ledger(ledger_directory).reserve("corpus_author", Decimal("0"))


def test_packet_is_validated_in_a_sibling_staging_directory_before_atomic_publish(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    packet = tmp_path / "accepted-packet"
    validated: list[Path] = []

    def validate(staged: Path) -> None:
        assert staged.parent == packet.parent
        assert staged != packet
        if not validated:
            assert not packet.exists()
        assert {path.name for path in staged.iterdir()} == {
            "plan.json",
            "accepted-train-dev.jsonl",
            "accepted-holdout.jsonl",
            "holdout-identities.json",
            "rejected.jsonl",
            "ledger.json",
            "packet.json",
        }
        validated.append(staged)

    monkeypatch.setattr(corpus, "validate_accepted_packet", validate)
    receipt = corpus.seal_packet(packet, {}, [], [], {})
    assert receipt == packet / "packet.json"
    assert validated and validated[0] != packet
    original = receipt.read_bytes()
    with pytest.raises(FileExistsError):
        corpus.seal_packet(packet, {}, [], [], {})
    assert receipt.read_bytes() == original
    assert not list(packet.parent.glob(f".{packet.name}.*"))


def test_holdout_identity_projection_is_canonical_and_content_free() -> None:
    slot = next(slot for slot in build_plan()["slots"] if slot["split"] == "synthetic_holdout")
    identity = corpus._identity_projection(_accepted_row(slot))
    assert set(identity) == {
        "task_id",
        "family_id",
        "pair_id",
        "split",
        "locale",
        "domain",
        "axes",
        "option_count",
        "gold_position",
        "criterion_ids",
    }
    rendered = corpus._canonical(identity).decode("utf-8")
    for forbidden in ("instruction", "state", "description", "review", "token", "embedding"):
        assert f'"{forbidden}"' not in rendered


def test_preholdout_packet_validator_never_opens_or_parses_holdout_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = build_plan()
    train_slot = next(
        slot
        for slot in plan["slots"]
        if slot["split"] == "synthetic_train" and slot["pair_id"] is None
    )
    holdout_slot = next(
        slot
        for slot in plan["slots"]
        if slot["split"] == "synthetic_holdout" and slot["pair_id"] is None
    )
    train = _accepted_row(train_slot)
    holdout = _accepted_row(holdout_slot)
    packet = tmp_path / "packet"
    packet.mkdir()
    files = {
        "plan.json": corpus._canonical({"slots": [train_slot, holdout_slot]}) + b"\n",
        "accepted-train-dev.jsonl": corpus._jsonl([train]),
        "accepted-holdout.jsonl": corpus._jsonl([holdout]),
        "holdout-identities.json": corpus._canonical(
            {
                "schema_version": "phase4e-holdout-identities.v1",
                "rows": [corpus._identity_projection(holdout)],
            }
        )
        + b"\n",
        "rejected.jsonl": b"",
        "ledger.json": corpus._canonical(corpus.BudgetLedger().as_json()) + b"\n",
    }
    for name, raw in files.items():
        (packet / name).write_bytes(raw)
    (packet / "packet.json").write_bytes(
        corpus._canonical(
            {
                "schema_version": "phase4e-accepted-packet.v2",
                "sealed": True,
                "files": {name: corpus._sha(raw) for name, raw in files.items()},
            }
        )
        + b"\n"
    )
    holdout_path = (packet / "accepted-holdout.jsonl").resolve()
    holdout_raw = files["accepted-holdout.jsonl"]
    original_read_bytes = Path.read_bytes
    original_json_loads = json.loads
    opened: list[Path] = []

    def tracked_read_bytes(path: Path) -> bytes:
        if path.resolve() == holdout_path:
            opened.append(path)
        return original_read_bytes(path)

    def guarded_json_loads(value: object, *args: object, **kwargs: object) -> object:
        if value == holdout_raw:
            raise AssertionError("preclaim parsed accepted holdout")
        return original_json_loads(cast(str | bytes | bytearray, value), *args, **cast(Any, kwargs))

    monkeypatch.setattr(corpus, "validate_plan", lambda _plan: None)
    monkeypatch.setattr(corpus, "_minimums", lambda _rows: [])
    monkeypatch.setattr(corpus, "_validate_provider_lineage", lambda *args: None)
    monkeypatch.setattr(Path, "read_bytes", tracked_read_bytes)
    monkeypatch.setattr(json, "loads", guarded_json_loads)
    corpus.validate_accepted_packet_pre_holdout(packet)
    assert not opened


@pytest.mark.parametrize("kind", ["failure", "timeout", "invalid"])
def test_presend_reservation_is_atomic_and_charged_after_transport_errors(
    tmp_path: Path, kind: str
) -> None:
    slot = next(slot for slot in build_plan()["slots"] if slot["pair_id"] is None)
    ledger_directory = tmp_path / "ledger"

    def transport(
        method: str, url: str, headers: Mapping[str, str], body: bytes
    ) -> tuple[int, dict[str, str], bytes]:
        del method, url, headers, body
        snapshot = resume_ledger(ledger_directory)
        assert len(snapshot.entries) == 1
        assert snapshot.entries[0]["status"] == "reserved"
        if kind == "timeout":
            raise TimeoutError("injected timeout")
        if kind == "invalid":
            return 200, {}, b"not-json"
        return 503, {}, b"{}"

    client = OpenRouterCorpusClient(
        transport=transport, allow_network=True, ledger_directory=ledger_directory
    )
    with pytest.raises((CorpusError, TimeoutError)):
        client.author(
            slots=[slot],
            max_output_tokens=1,
            api_key="test-only",
        )
    resumed = resume_ledger(ledger_directory)
    assert resumed.entries == client.ledger.entries
    assert resumed.entries[0]["status"] == "reserved"
    assert resumed.spent("corpus_author") > 0


def test_fake_transport_is_pinned_resumable_and_never_needs_socket(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def blocked(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("socket access")

    monkeypatch.setattr(socket, "create_connection", blocked)
    captured: dict[str, Any] = {}

    def fake(
        method: str, url: str, headers: Mapping[str, str], body: bytes
    ) -> tuple[int, dict[str, str], bytes]:
        captured.update(method=method, url=url, headers=dict(headers), body=json.loads(body))
        return 200, {}, b'{"id":"fake-request","usage":{"cost":0.000001}}'

    client = OpenRouterCorpusClient(
        transport=fake, allow_network=True, ledger_directory=tmp_path / "ledger"
    )
    unpaired_slot = next(slot for slot in build_plan()["slots"] if slot["pair_id"] is None)
    client.author(
        slots=[unpaired_slot],
        max_output_tokens=1,
        api_key="test-only",
    )
    assert captured["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert captured["body"]["provider"] == {"data_collection": "deny", "zdr": True}
    assert "test-only" not in json.dumps(captured["body"])
    assert client.ledger.entries[0]["request_id"] == "fake-request"
    assert client.ledger.entries[0]["status"] == "settled"


def test_reviewer_transport_accepts_only_one_validated_blind_author_row(
    tmp_path: Path,
) -> None:
    slot = next(slot for slot in build_plan()["slots"] if slot["pair_id"] is None)
    author_row = validate_author_rows([_record(slot)], [slot], _Counter())[0]
    row = ValidatedAuthorRow.model_validate(author_row)
    captured: dict[str, Any] = {}

    def transport(
        method: str, url: str, headers: Mapping[str, str], body: bytes
    ) -> tuple[int, dict[str, str], bytes]:
        del method, url, headers
        captured["body"] = json.loads(body)
        return 200, {}, b'{"id":"blind-review","usage":{"cost":0}}'

    client = OpenRouterCorpusClient(
        transport=transport, allow_network=True, ledger_directory=tmp_path / "ledger"
    )
    client.reviewer(row=row, max_output_tokens=1, api_key="test-only")

    messages = captured["body"]["messages"]
    assert messages == reviewer_messages([row])
    reviewer_prompt = messages[1]["content"]
    for field in (
        "split",
        "gold_position",
        "selected_criterion_id",
        "axes",
        "pair_id",
        "cross_locale_attestation",
        "author_reasoning",
    ):
        assert field not in reviewer_prompt

    with pytest.raises(CorpusError, match="validated author row"):
        reviewer_messages([author_row])  # type: ignore[list-item]
    with pytest.raises(ValidationError, match="author_reasoning"):
        ValidatedAuthorRow.model_validate({**author_row, "author_reasoning": "INJECT_GOLD"})
    reviewer_call: Any = client.reviewer
    with pytest.raises(TypeError, match="messages"):
        reviewer_call(
            row=row,
            max_output_tokens=1,
            api_key="test-only",
            messages=[{"role": "user", "content": "INJECT_GOLD"}],
        )
    assert "INJECT_GOLD" not in reviewer_prompt


def test_wilson_stop_projection_fails_before_a_required_cell_can_be_minted() -> None:
    plan = build_plan()
    resolved = [{"task_id": slot["task_id"], "status": "rejected"} for slot in plan["slots"][:200]]
    assert stop_projection(resolved, plan) in {
        "impossible_required_minimum",
        "wilson_projected_minimum",
    }


def test_wilson_skips_unobserved_cells_but_enforces_global_and_split_minima() -> None:
    plan = build_plan()
    train = [slot for slot in plan["slots"] if slot["split"] == "synthetic_train"]
    assert (
        stop_projection(
            [{"task_id": slot["task_id"], "status": "accepted"} for slot in train[:200]], plan
        )
        is None
    )

    dev = [slot for slot in plan["slots"] if slot["split"] == "synthetic_dev"]
    resolved = [
        *[{"task_id": slot["task_id"], "status": "accepted"} for slot in train[:800]],
        *[{"task_id": slot["task_id"], "status": "rejected"} for slot in dev[:61]],
    ]
    assert stop_projection(resolved, plan) == "impossible_required_minimum"
