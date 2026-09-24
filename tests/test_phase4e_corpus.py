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
    decode_author_response,
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
    target = cast(dict[str, Any], slot["semantic_target"])
    roles = list(cast(list[str], target["criterion_roles"]))
    gold = int(slot["gold_position"])
    roles.insert(gold, roles.pop(0))
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
        "instruction": "Choose the safest fictional route.",
        "state": {"summary": "A fictional request."},
        "criteria": criteria,
        "selected_criterion_id": criteria[int(slot["gold_position"])]["id"],
        "semantic_equivalence_attestation": semantic_attestation,
    }
    if isinstance(slot["pair_id"], str):
        record["cross_locale_attestation"] = {
            "pair_id": slot["pair_id"],
            **semantic_attestation,
        }
    return record


def _review(row: Mapping[str, Any], **overrides: Any) -> dict[str, Any]:
    selected_id = row["selected_criterion_id"]
    semantic_attestation = row["semantic_equivalence_attestation"]
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
    assert plan["schema_version"] == "phase4e-universal-plan.v2"
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
    assert len(pairs) == 300
    assert all({row["locale"] for row in pair} == {"pt-BR", "en"} for pair in pairs.values())
    assert all(pair[0]["semantic_target"] == pair[1]["semantic_target"] for pair in pairs.values())
    assert {
        split: sum(1 for pair in pairs.values() if pair[0]["split"] == split) for split in SPLITS
    } == {"synthetic_train": 210, "synthetic_dev": 45, "synthetic_holdout": 45}

    mutated = json.loads(json.dumps(plan))
    mutated["slots"][0]["semantic_target"]["scenario"] = "content_safety"
    with pytest.raises(CorpusError, match="immutable plan mismatch"):
        validate_plan(mutated)


def test_author_must_restate_the_selected_first_target_before_local_reordering() -> None:
    slot = next(
        slot
        for slot in build_plan()["slots"]
        if slot["pair_id"] is None and slot["option_count"] == 2 and slot["gold_position"] == 1
    )
    target = cast(dict[str, Any], slot["semantic_target"])
    generated: dict[str, Any] = {
        "instruction": "Choose the fictional route supported by the stated rule.",
        "state": {"summary": "The rule requires the sole route supported by the fictional facts."},
        "criteria": [
            {"description": "Take the route directly supported by the stated rule."},
            {"description": "Take a route that conflicts with the stated rule."},
        ],
        "selected_index": 0,
        "semantic_equivalence_attestation": {**target, "selected_role": "matches_rule"},
    }
    row = validate_author_rows([generated], [slot], _Counter())[0]
    assert row["selected_criterion_id"] == "criterion-1"
    assert row["semantic_equivalence_attestation"]["criterion_roles"] == [
        target["criterion_roles"][1],
        "matches_rule",
    ]

    mismatched = json.loads(json.dumps(generated))
    mismatched["semantic_equivalence_attestation"]["scenario"] = "content_safety"
    if target["scenario"] == "content_safety":
        mismatched["semantic_equivalence_attestation"]["scenario"] = "action_required"
    with pytest.raises(CorpusError, match="author semantic target mismatch"):
        validate_author_rows([mismatched], [slot], _Counter())

    labelled = json.loads(json.dumps(generated))
    labelled["instruction"] = str(target["scenario"])
    with pytest.raises(CorpusError, match="author wrote semantic label"):
        validate_author_rows([labelled], [slot], _Counter())


def test_author_capacity_gate_and_reviewer_blindness_are_fail_closed() -> None:
    slots = build_plan()["slots"][:2]
    rows = validate_author_rows([_record(slot) for slot in slots], slots, _Counter())
    reviewer_payload = json.dumps(
        reviewer_messages([ValidatedAuthorRow.model_validate(rows[0])]), ensure_ascii=False
    )
    assert "gold_position" not in reviewer_payload
    assert "selected_criterion_id" not in reviewer_payload
    assert "semantic_target" not in reviewer_payload
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
    unpaired = next(slot for slot in build_plan()["slots"] if slot["pair_id"] is None)
    assert "gold_position" in json.dumps(author_messages([unpaired]))
    assert "semantic_target" in json.dumps(author_messages([unpaired]))
    assert "flat JSON object" in author_messages([unpaired])[0]["content"]


def test_author_prompts_share_closed_role_definitions_and_forbid_role_labels() -> None:
    slot = next(slot for slot in build_plan()["slots"] if slot["pair_id"] is None)
    generated_row = validate_author_rows([_record(slot)], [slot], _Counter())[0]
    row = ValidatedAuthorRow.model_validate(generated_row)
    author_prompt = author_messages([slot])[0]["content"]
    reviewer_prompt = reviewer_messages([row])[0]["content"]
    assert corpus.SCENARIO_CODEBOOK in author_prompt
    assert corpus.SCENARIO_CODEBOOK in reviewer_prompt
    assert author_prompt.count(corpus.SCENARIO_CODEBOOK) == 1
    assert reviewer_prompt.count(corpus.SCENARIO_CODEBOOK) == 1
    assert "Never copy a scenario code or criterion-role token literally" in author_prompt
    assert "do not write role labels" in reviewer_prompt
    assert "state.summary at most 180 characters" in author_prompt


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
    assert "sole top-level key is reviews" in reviewer_messages([row])[0]["content"]
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
    assert "semantic_equivalence_attestation" not in task

    corpus._validate_reviewer_task_view(task)
    for field, value in (
        ("synthetic_provenance", True),
        ("selected_criterion_id", "criterion-0"),
        ("semantic_equivalence_attestation", {"scenario": "topic_routing"}),
    ):
        with pytest.raises(CorpusError, match="reviewer transport fields"):
            corpus._validate_reviewer_task_view({**task, field: value})

    prompt = reviewer_messages([row])[0]["content"]
    assert "Independently judge fictionality from supplied content" in prompt
    assert "do not treat provenance or stated synthetic intent as proof" in prompt

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


def test_source_contract_failure_resolves_planned_slot_but_tampering_is_fatal() -> None:
    slot = next(
        slot
        for slot in build_plan()["slots"]
        if slot["pair_id"] is None and slot["option_count"] == 2
    )
    generated: dict[str, Any] = {
        "instruction": "Choose the route supported by the fictional state.",
        "state": {"summary": "x" * 181},
        "criteria": [
            {"description": "Route A conflicts with the rule."},
            {"description": "Route B satisfies the rule."},
        ],
        "selected_index": 1,
        "semantic_equivalence_attestation": {
            "scenario": "topic_routing",
            "criterion_roles": ["contradicts_rule", "matches_rule"],
            "selected_role": "matches_rule",
        },
    }
    usable, rejected = classify_author_rows([generated], [slot], _Counter())
    assert usable == []
    assert rejected[0]["task_id"] == slot["task_id"]
    assert rejected[0]["split"] == slot["split"]
    assert rejected[0]["reason"].startswith("author_record_schema__state.summary:")

    tampered = {**_record(slot), "domain": "tampered_domain"}
    with pytest.raises(CorpusError, match="author changed planner-owned field"):
        classify_author_rows([tampered], [slot], _Counter())


def test_author_rejection_reasons_are_durable_and_content_free() -> None:
    slot = next(
        slot
        for slot in build_plan()["slots"]
        if slot["pair_id"] is None and slot["option_count"] == 2
    )
    generated = {
        "instruction": "Choose the fictional route supported by the stated rule.",
        "state": {"summary": "The fictional rule supports one route."},
        "criteria": [
            {"description": "Take the route supported by the rule."},
            {"description": "Take the route that conflicts with the rule."},
        ],
        "selected_index": 0,
        "semantic_equivalence_attestation": {
            **slot["semantic_target"],
            "selected_role": "matches_rule",
        },
    }
    target_mismatch = json.loads(json.dumps(generated))
    target_mismatch["semantic_equivalence_attestation"]["scenario"] = "content_safety"
    if slot["semantic_target"]["scenario"] == "content_safety":
        target_mismatch["semantic_equivalence_attestation"]["scenario"] = "action_required"
    label_leakage = json.loads(json.dumps(generated))
    label_leakage["instruction"] = slot["semantic_target"]["scenario"]
    source_failure = json.loads(json.dumps(generated))
    source_failure["state"] = {"summary": "x" * 181}

    for record, reason in (
        (target_mismatch, "semantic_target_mismatch"),
        (label_leakage, "semantic_label_leakage"),
        (source_failure, "author_record_schema__state.summary:string_too_long"),
    ):
        usable, rejected = classify_author_rows([record], [slot], _Counter())
        assert usable == []
        assert rejected == [{"task_id": slot["task_id"], "split": slot["split"], "reason": reason}]

    usable, rejected = classify_author_rows([{}], [slot], _Counter())
    assert usable == []
    assert rejected[0]["reason"].startswith(
        "author_record_schema__instruction:missing_state:missing_criteria:missing_"
    )


def test_author_schema_reason_never_persists_provider_controlled_extra_key() -> None:
    slot = next(
        slot
        for slot in build_plan()["slots"]
        if slot["pair_id"] is None and slot["option_count"] == 2
    )
    record = {**_record(slot), "raw_marker_leak_123": "private provider value"}
    usable, rejected = classify_author_rows([record], [slot], _Counter())
    assert usable == []
    assert rejected[0]["reason"] == "author_record_schema__unknown:extra_forbidden"
    assert "raw_marker" not in rejected[0]["reason"]
    assert "private" not in rejected[0]["reason"]


def test_author_schema_requires_exact_planned_cardinality_and_full_cross_locale_pair() -> None:
    plan = build_plan()
    pair_id = next(slot["pair_id"] for slot in plan["slots"] if slot["pair_id"])
    pair = [slot for slot in plan["slots"] if slot["pair_id"] == pair_id]
    schema = author_schema(pair)
    assert "$defs" not in schema
    assert "$ref" not in json.dumps(schema)
    properties = schema["properties"]
    assert set(schema["required"]) == set(properties)
    assert properties["record_0_state_summary"] == {
        "type": "string",
        "minLength": 1,
        "maxLength": 180,
    }
    assert properties["record_0_criterion_0_description"]["maxLength"] == 120
    assert properties["record_0_selected_index"] == {"type": "integer", "enum": [0]}
    target = pair[0]["semantic_target"]
    assert properties["record_0_scenario"] == {
        "type": "string",
        "enum": [target["scenario"]],
    }
    assert properties["record_0_criterion_0_role"] == {
        "type": "string",
        "enum": [target["criterion_roles"][0]],
    }
    assert properties["record_1_selected_role"] == {
        "type": "string",
        "enum": ["matches_rule"],
    }
    assert not any(value.get("type") in {"object", "array"} for value in properties.values())

    wire: dict[str, Any] = {}
    for index, slot in enumerate(pair):
        target = slot["semantic_target"]
        prefix = f"record_{index}_"
        wire[f"{prefix}instruction"] = f"Instruction {index}"
        wire[f"{prefix}state_summary"] = f"State {index}"
        wire[f"{prefix}selected_index"] = 0
        wire[f"{prefix}scenario"] = target["scenario"]
        wire[f"{prefix}selected_role"] = "matches_rule"
        for criterion_index, role in enumerate(target["criterion_roles"]):
            wire[f"{prefix}criterion_{criterion_index}_description"] = (
                f"Criterion {index}-{criterion_index}"
            )
            wire[f"{prefix}criterion_{criterion_index}_role"] = role
    decoded = decode_author_response(wire, pair)
    assert len(decoded) == 2
    assert decoded[0]["instruction"] == "Instruction 0"
    assert decoded[1]["semantic_equivalence_attestation"] == {
        **pair[1]["semantic_target"],
        "selected_role": "matches_rule",
    }
    tampered = decode_author_response({**wire, "provider_private_marker": "never persist"}, pair)
    usable, rejected = classify_author_rows(tampered, pair, _Counter())
    assert usable == []
    assert {row["reason"] for row in rejected} == {"author_record_schema__unknown:extra_forbidden"}
    assert "provider_private_marker" not in json.dumps(rejected)
    assert "never persist" not in json.dumps(rejected)
    with pytest.raises(CorpusError, match="cross-locale author batch"):
        author_schema(pair[:1])
    mismatched_pair = json.loads(json.dumps(pair))
    mismatched_pair[1]["semantic_target"]["scenario"] = "content_safety"
    if mismatched_pair[0]["semantic_target"]["scenario"] == "content_safety":
        mismatched_pair[1]["semantic_target"]["scenario"] = "action_required"
    with pytest.raises(CorpusError, match="cross-locale author semantic target"):
        author_messages(mismatched_pair)
    with pytest.raises(CorpusError, match="cross-locale author semantic target"):
        author_schema(mismatched_pair)

    review = corpus.reviewer_schema(1)
    assert "$defs" not in review
    assert "$ref" not in json.dumps(review)
    assert set(review["properties"]["reviews"]["items"]["properties"]) == {
        "status",
        "selected_criterion_id",
        "reason_codes",
        "natural_language",
        "fictional",
        "exclusive_options",
        "private_or_sensitive",
        "semantic_equivalence_attestation",
    }
    review_attestation = review["properties"]["reviews"]["items"]["properties"][
        "semantic_equivalence_attestation"
    ]
    assert review_attestation["properties"]["selected_role"] == {
        "type": "string",
        "enum": ["matches_rule"],
    }
    assert review["properties"]["reviews"]["items"]["properties"]["reason_codes"]["items"][
        "enum"
    ] == list(corpus.REVIEWER_REASON_CODES)


def test_generated_author_choice_is_reordered_to_planned_gold_position() -> None:
    slot = next(
        slot
        for slot in build_plan()["slots"]
        if slot["pair_id"] is None and slot["option_count"] == 2
    )
    target = cast(dict[str, Any], slot["semantic_target"])
    generated = {
        "instruction": "Choose the route supported by the fictional state.",
        "state": {"summary": "Only route B satisfies the stated rule."},
        "criteria": [
            {"description": "Route B satisfies the rule."},
            {"description": "Route A does not satisfy the rule."},
        ],
        "selected_index": 0,
        "semantic_equivalence_attestation": {
            "scenario": target["scenario"],
            "criterion_roles": target["criterion_roles"],
            "selected_role": "matches_rule",
        },
    }
    row = validate_author_rows([generated], [slot], _Counter())[0]
    gold = int(slot["gold_position"])
    assert row["selected_criterion_id"] == f"criterion-{gold}"
    assert "satisfies the rule" in row["criteria"][gold]["description"]
    assert row["semantic_equivalence_attestation"]["criterion_roles"][gold] == "matches_rule"


def test_semantic_attestations_are_provider_emitted_not_position_derived() -> None:
    slot = next(slot for slot in build_plan()["slots"] if slot["pair_id"] is None)
    row = validate_author_rows([_record(slot)], [slot], _Counter())[0]
    typed = ValidatedAuthorRow.model_validate(row)
    generated_review = {
        "status": "accepted",
        "selected_criterion_id": row["selected_criterion_id"],
        "reason_codes": [],
        "natural_language": True,
        "fictional": True,
        "exclusive_options": True,
        "private_or_sensitive": False,
    }
    with pytest.raises(CorpusError, match="review record schema"):
        corpus.materialize_reviewer_record(generated_review, typed)

    attestation = {
        "scenario": "topic_routing",
        "criterion_roles": row["semantic_equivalence_attestation"]["criterion_roles"],
        "selected_role": "matches_rule",
    }
    materialized = corpus.materialize_reviewer_record(
        {**generated_review, "semantic_equivalence_attestation": attestation}, typed
    )
    assert materialized["semantic_equivalence_attestation"] == attestation
    marker = "provider_private_review_key"
    with pytest.raises(
        CorpusError, match=r"review record schema \(unknown:extra_forbidden\)"
    ) as caught:
        corpus.materialize_reviewer_record(
            {
                **generated_review,
                "semantic_equivalence_attestation": attestation,
                marker: "never persist",
            },
            typed,
        )
    assert marker not in str(caught.value)
    assert "never persist" not in str(caught.value)

    private_reason_marker = "private reviewer explanation must not persist"
    with pytest.raises(
        CorpusError, match=r"review record schema \(reason_codes:literal_error\)"
    ) as caught:
        corpus.materialize_reviewer_record(
            {
                **generated_review,
                "reason_codes": [private_reason_marker],
                "semantic_equivalence_attestation": attestation,
            },
            typed,
        )
    assert private_reason_marker not in str(caught.value)


def test_same_gold_position_with_different_closed_author_semantics_rejects_pair() -> None:
    plan = build_plan()
    pair_id = next(slot["pair_id"] for slot in plan["slots"] if slot["pair_id"])
    slots = [slot for slot in plan["slots"] if slot["pair_id"] == pair_id]
    rows = validate_author_rows(_paired_records(slots), slots, _Counter())
    assert rows[0]["selected_criterion_id"] == f"route-{slots[0]['gold_position']}"
    assert rows[1]["selected_criterion_id"] == f"route-{slots[1]['gold_position']}"
    rows[1]["semantic_equivalence_attestation"] = {
        **rows[1]["semantic_equivalence_attestation"],
        "scenario": "deadline_risk",
    }
    rows[1]["cross_locale_attestation"] = {
        **rows[1]["cross_locale_attestation"],
        "scenario": "deadline_risk",
    }
    accepted, rejected = resolve_reviews(rows, [_review(row) for row in rows])
    assert accepted == []
    assert {row["reason"] for row in rejected} == {"scenario_disagreement"}

    role_rows = validate_author_rows(_paired_records(slots), slots, _Counter())
    roles = list(role_rows[1]["semantic_equivalence_attestation"]["criterion_roles"])
    non_selected = next(index for index, role in enumerate(roles) if role != "matches_rule")
    roles[non_selected] = next(
        role
        for role in (
            "contradicts_rule",
            "irrelevant_to_rule",
            "insufficient_evidence",
            "unsafe_action",
            "premature_action",
            "overbroad_action",
            "duplicate_action",
        )
        if role not in roles
    )
    role_rows[1]["semantic_equivalence_attestation"] = {
        **role_rows[1]["semantic_equivalence_attestation"],
        "criterion_roles": roles,
    }
    role_rows[1]["cross_locale_attestation"] = {
        **role_rows[1]["cross_locale_attestation"],
        "criterion_roles": roles,
    }
    accepted, rejected = resolve_reviews(role_rows, [_review(row) for row in role_rows])
    assert accepted == []
    assert {row["reason"] for row in rejected} == {"criterion_role_disagreement"}


def test_generated_author_text_is_preserved_and_invalid_content_rejects() -> None:
    slot = next(
        slot
        for slot in build_plan()["slots"]
        if slot["pair_id"] is None and slot["option_count"] == 2
    )
    generated: dict[str, Any] = {
        "instruction": "Choose  the supported fictional route. ",
        "state": {"summary": " A fictional request with  two spaces. "},
        "criteria": [
            {"description": " Route  A conflicts with the rule. "},
            {"description": " Route  B satisfies the rule. "},
        ],
        "selected_index": 0,
        "semantic_equivalence_attestation": {
            "scenario": slot["semantic_target"]["scenario"],
            "criterion_roles": slot["semantic_target"]["criterion_roles"],
            "selected_role": "matches_rule",
        },
    }
    original = json.dumps(generated, ensure_ascii=False, sort_keys=True)
    row = validate_author_rows([generated], [slot], _Counter())[0]
    assert json.dumps(generated, ensure_ascii=False, sort_keys=True) == original
    assert row["instruction"] == generated["instruction"]
    assert row["state"] == generated["state"]
    assert {item["description"] for item in row["criteria"]} == {
        item["description"] for item in generated["criteria"]
    }

    over_limit = {**generated, "instruction": "x" * 121}
    over_limit_original = json.dumps(over_limit, ensure_ascii=False, sort_keys=True)
    with pytest.raises(CorpusError, match="author record schema"):
        validate_author_rows([over_limit], [slot], _Counter())
    assert json.dumps(over_limit, ensure_ascii=False, sort_keys=True) == over_limit_original

    non_nfc = {**generated, "instruction": "Cafe\u0301 fictional route."}
    non_nfc_original = json.dumps(non_nfc, ensure_ascii=False, sort_keys=True)
    with pytest.raises(CorpusError, match="author task contract"):
        validate_author_rows([non_nfc], [slot], _Counter())
    assert json.dumps(non_nfc, ensure_ascii=False, sort_keys=True) == non_nfc_original

    state_over_final_limit = {**generated, "state": {"summary": "x" * 181}}
    with pytest.raises(CorpusError, match="author record schema"):
        validate_author_rows([state_over_final_limit], [slot], _Counter())


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


@pytest.mark.parametrize(
    ("overrides", "reason"),
    (
        ({"status": "rejected", "private_or_sensitive": True}, "privacy"),
        ({"status": "rejected", "fictional": False}, "review_quality_fictional"),
        (
            {"status": "rejected", "natural_language": False},
            "review_quality_natural_language",
        ),
        (
            {"status": "rejected", "exclusive_options": False},
            "review_quality_exclusive_options",
        ),
        ({"status": "rejected"}, "review_rejected"),
    ),
)
def test_reviewer_rejection_keeps_only_closed_quality_diagnostics(
    overrides: dict[str, Any], reason: str
) -> None:
    slot = next(slot for slot in build_plan()["slots"] if slot["pair_id"] is None)
    row = validate_author_rows([_record(slot)], [slot], _Counter())[0]
    accepted, rejected = resolve_reviews([row], [_review(row, **overrides)])
    assert accepted == []
    assert rejected[0]["reason"] == reason
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


def test_accepted_review_requires_empty_closed_reason_codes() -> None:
    slot = next(slot for slot in build_plan()["slots"] if slot["pair_id"] is None)
    row = validate_author_rows([_record(slot)], [slot], _Counter())[0]
    accepted, rejected = resolve_reviews([row], [_review(row, reason_codes=["semantics"])])
    assert accepted == []
    assert rejected[0]["reason"] == "review_rejected"

    accepted, rejected = resolve_reviews([row], [_review(row)])
    assert rejected == []
    assert accepted[0]["review"]["reason_codes"] == []


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
        "scenario": "deadline_risk",
    }
    accepted, rejected = resolve_reviews(rows, tampered)
    assert accepted == []
    assert {row["reason"] for row in rejected} == {
        "scenario_disagreement",
        "paired_member_rejected",
    }
    assert {row["task_id"] for row in rejected} == {row["task_id"] for row in rows}


def test_budget_uses_larger_debit_and_never_borrows_stage_budget() -> None:
    ledger = BudgetLedger()
    ledger.record("corpus_author", "request-a", Decimal("0.01"), Decimal("0.02"), "a" * 64)
    assert ledger.spent("corpus_author") == Decimal("0.02")
    with pytest.raises(CorpusError, match="stage budget"):
        ledger.reserve("corpus_author", corpus.STAGE_LIMITS["corpus_author"] - Decimal("0.01"))
    with pytest.raises(CorpusError, match="network"):
        OpenRouterCorpusClient()


def test_provider_error_payload_is_never_interpolated_or_persisted(tmp_path: Path) -> None:
    marker = "private provider trace marker"
    slot = next(slot for slot in build_plan()["slots"] if slot["pair_id"] is None)

    def transport(
        method: str, url: str, headers: Mapping[str, str], body: bytes
    ) -> tuple[int, dict[str, str], bytes]:
        del method, url, headers, body
        return (
            400,
            {},
            json.dumps(
                {
                    "error": {
                        "code": marker,
                        "message": "Provider returned error",
                        "metadata": {"raw": marker},
                    }
                }
            ).encode(),
        )

    client = OpenRouterCorpusClient(
        transport=transport,
        allow_network=True,
        ledger_directory=tmp_path / "ledger",
    )
    with pytest.raises(CorpusError, match=r"^provider response error$") as caught:
        client.author(slots=[slot], max_output_tokens=640, api_key="test-key")
    assert marker not in str(caught.value)
    assert marker not in "".join(path.read_text() for path in (tmp_path / "ledger").glob("*.json"))


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
    assert all(body["reasoning_effort"] == "none" for body, _ in captured)
    assert all(
        body["response_format"]["json_schema"]["name"] == corpus.RESPONSE_SCHEMA_NAME
        for body, _ in captured
    )
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
        charged = corpus.STAGE_LIMITS["corpus_author"] + Decimal("0.01")
        return (
            429,
            {},
            json.dumps({"id": "charged-request", "usage": {"cost": str(charged)}}).encode(),
        )

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
    expected_charge = str(corpus.STAGE_LIMITS["corpus_author"] + Decimal("0.01"))
    assert entry["provider_cost_usd"] == expected_charge
    assert entry["debit_usd"] == expected_charge
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

    monkeypatch.setattr(corpus, "validate_accepted_packet_pre_holdout", validate)
    monkeypatch.setattr(
        corpus,
        "validate_accepted_packet",
        lambda _staged: (_ for _ in ()).throw(AssertionError("seal opened holdout")),
    )
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
    rejected_slot = next(
        slot
        for slot in plan["slots"]
        if slot["split"] == "synthetic_dev" and slot["pair_id"] is None
    )
    train = _accepted_row(train_slot)
    holdout = _accepted_row(holdout_slot)
    packet = tmp_path / "packet"
    packet.mkdir()
    files = {
        "plan.json": corpus._canonical({"slots": [train_slot, rejected_slot, holdout_slot]})
        + b"\n",
        "accepted-train-dev.jsonl": corpus._jsonl([train]),
        "accepted-holdout.jsonl": corpus._jsonl([holdout]),
        "holdout-identities.json": corpus._canonical(
            {
                "schema_version": "phase4e-holdout-identities.v1",
                "rows": [corpus._identity_projection(holdout)],
            }
        )
        + b"\n",
        "rejected.jsonl": corpus._jsonl(
            [
                {
                    "task_id": rejected_slot["task_id"],
                    "split": rejected_slot["split"],
                    "reason": "capacity",
                    "author_response_sha256": "a" * 64,
                    "author_reservation_id": "reservation-" + "a" * 64,
                    "author_request_id": "author-rejected",
                }
            ]
        ),
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
    captured_rejections: list[list[dict[str, Any]]] = []
    monkeypatch.setattr(
        corpus,
        "_validate_provider_lineage",
        lambda _ledger, _accepted, rejected, _resolved: captured_rejections.append(list(rejected)),
    )
    monkeypatch.setattr(Path, "read_bytes", tracked_read_bytes)
    monkeypatch.setattr(json, "loads", guarded_json_loads)
    corpus.validate_accepted_packet_pre_holdout(packet)
    assert not opened
    assert captured_rejections[0][0]["author_request_id"] == "author-rejected"


def test_rejected_lineage_shape_rejects_partial_or_unbound_fields() -> None:
    slot = next(slot for slot in build_plan()["slots"] if slot["pair_id"] is None)
    slots = {slot["task_id"]: slot}
    valid = {
        "task_id": slot["task_id"],
        "split": slot["split"],
        "reason": "capacity",
        "author_response_sha256": "a" * 64,
        "author_reservation_id": "reservation-" + "a" * 64,
        "author_request_id": "author-rejected",
    }
    corpus._validate_rejected_row(valid, slots)
    with pytest.raises(CorpusError, match="packet rejection"):
        corpus._validate_rejected_row({**valid, "unexpected": True}, slots)

    ledger = BudgetLedger()
    with pytest.raises(CorpusError, match="rejected provider lineage"):
        corpus._validate_provider_lineage(ledger, [], [valid], {slot["task_id"]})


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
        return (
            200,
            {},
            b'{"id":"private provider id must not persist","usage":{"cost":0.000001}}',
        )

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
    assert captured["body"]["provider"] == corpus.provider_preferences("corpus_author")
    assert "test-only" not in json.dumps(captured["body"])
    entry = client.ledger.entries[0]
    assert entry["request_id"] == corpus._local_response_request_id(
        entry["reservation_id"], entry["response_sha256"]
    )
    assert "private provider id must not persist" not in json.dumps(client.ledger.as_json())
    assert client.ledger.entries[0]["status"] == "settled"


def test_reviewer_transport_omits_unsupported_reasoning_effort_and_is_answer_blind(
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

    assert captured["body"]["provider"] == corpus.provider_preferences("corpus_reviewer")
    # CoreWeave's pinned Llama endpoint does not advertise this parameter.
    assert "reasoning_effort" not in captured["body"]
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
        "semantic_equivalence_attestation",
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


def _resolved_with_cohort_outcomes(
    plan: Mapping[str, Any], cohort: list[Mapping[str, Any]], observed: int, accepted: int
) -> list[dict[str, Any]]:
    cohort_ids = {slot["task_id"] for slot in cohort}
    observed_ids = {slot["task_id"] for slot in cohort[:observed]}
    accepted_ids = {slot["task_id"] for slot in cohort[:accepted]}
    return [
        {
            "task_id": slot["task_id"],
            "status": (
                "accepted"
                if slot["task_id"] not in cohort_ids or slot["task_id"] in accepted_ids
                else "rejected"
            ),
        }
        for slot in plan["slots"]
        if slot["task_id"] not in cohort_ids or slot["task_id"] in observed_ids
    ]


def test_wilson_projection_is_not_authoritative_for_four_accepted_of_eight() -> None:
    plan = build_plan()
    cohort = [
        slot
        for slot in plan["slots"]
        if slot["split"] == "synthetic_train"
        and slot["locale"] == "pt-BR"
        and slot["option_count"] == 2
    ]
    resolved = _resolved_with_cohort_outcomes(plan, cohort, observed=8, accepted=4)
    assert stop_projection(resolved, plan) is None


def test_wilson_projection_becomes_authoritative_at_twenty_outcomes() -> None:
    plan = build_plan()
    cohort = [
        slot
        for slot in plan["slots"]
        if slot["split"] == "synthetic_train"
        and slot["locale"] == "pt-BR"
        and slot["option_count"] == 2
    ]
    resolved = _resolved_with_cohort_outcomes(plan, cohort, observed=20, accepted=4)
    assert stop_projection(resolved, plan) == "wilson_projected_minimum"


def test_impossible_capacity_remains_authoritative_below_wilson_eligibility() -> None:
    plan = build_plan()
    cohort = [
        slot
        for slot in plan["slots"]
        if slot["split"] == "synthetic_dev"
        and slot["locale"] == "pt-BR"
        and slot["option_count"] == 2
    ]
    resolved = _resolved_with_cohort_outcomes(plan, cohort, observed=12, accepted=0)
    assert stop_projection(resolved, plan) == "impossible_required_minimum"


def test_impossible_capacity_is_active_even_before_twenty_total_resolutions() -> None:
    plan = build_plan()
    cohort = [
        slot
        for slot in plan["slots"]
        if slot["split"] == "synthetic_dev"
        and slot["locale"] == "pt-BR"
        and slot["option_count"] == 2
    ]
    resolved = [{"task_id": slot["task_id"], "status": "rejected"} for slot in cohort[:12]]
    assert len(resolved) == 12
    assert stop_projection(resolved, plan) == "impossible_required_minimum"


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
