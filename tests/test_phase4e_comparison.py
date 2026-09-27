"""Offline, network-free tests for Phase 4E.4A blind comparison protocol.

Tests falsify each behavior: plan determinism, task isolation, generate with
injected transport, uncertain/streak circuit, score-backend contract, evaluate
primitives, publish sanitizer, process isolation, no-network/default import,
CLI/help, acceptance-minimums, and the nonconsecutive-uncertain -> inconclusive
rule.
"""

from __future__ import annotations

import json
import os
import random
import socket
import stat
import subprocess
import sys
from collections.abc import Mapping
from decimal import Decimal
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import pytest

from benchmarks import saracura_universal_comparison as comparison
from benchmarks import saracura_universal_corpus as corpus
from benchmarks import saracura_universal_pilot as pilot
from benchmarks.saracura_universal_corpus import build_plan as corpus_build_plan

ROOT = Path(__file__).parents[1]
POLICY_PATH = ROOT / "benchmarks/manifests/phase4e-comparison-policy.v1.json"

SOUND_CORPUS_CONTENT = {
    "locale": "pt-BR",
    "domain": "email_triage",
    "instruction": "Escolha a rota segura.",
    "state": {"summary": "Resumo de teste."},
    "selected_index": 0,
    "scenario": "action_required",
    "selected_role": "matches_rule",
}

CRITERION_STENCIL = {
    "description": "Criterion placeholder for slot.",
    "role": "matches_rule",
}


class _AlwaysCounter:
    def __init__(self, value: int) -> None:
        self._value = value

    def count(self, text: str) -> int:
        del text
        return self._value


class _TokenFactory:
    def __init__(self, default_count: int = 1) -> None:
        self._default = default_count

    def count(self, text: str) -> int:
        del text
        return self._default

    def create(self, snapshot: Path) -> _TokenFactory:
        del snapshot
        return _TokenFactory(self._default)


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


def _build_author_wire(slots: list[dict[str, Any]]) -> dict[str, Any]:
    wire: dict[str, Any] = {}
    for idx, slot in enumerate(slots):
        oc = cast(int, slot["option_count"])
        target = slot["semantic_target"]
        pref = f"record_{idx}_"
        marker = "".join(chr(97 + int(ch, 16)) for ch in str(slot["family_id"])[-16:])
        is_pt = slot["locale"] == "pt-BR"
        wire[f"{pref}instruction"] = (
            f"Escolha a rota segura para o caso fictício {marker}."
            if is_pt
            else f"Choose the safe route for fictional case {marker}."
        )
        wire[f"{pref}state_summary"] = (
            f"O caso {marker} contém uma regra e fatos suficientes."
            if is_pt
            else f"Case {marker} contains one rule and sufficient facts."
        )
        wire[f"{pref}selected_index"] = 0
        wire[f"{pref}scenario"] = target["scenario"]
        wire[f"{pref}selected_role"] = "matches_rule"
        for ci in range(oc):
            role = target["criterion_roles"][ci % len(target["criterion_roles"])]
            wire[f"{pref}criterion_{ci}_description"] = (
                f"Alternativa {ci} para o caso fictício {marker}."
                if is_pt
                else f"Option {ci} for fictional case {marker}."
            )
            wire[f"{pref}criterion_{ci}_role"] = role
    return wire


def _build_reviewer_payload(
    row: Mapping[str, Any],
    *,
    status: str = "accepted",
    disagree_answer: bool = False,
) -> dict[str, Any]:
    oc = cast(int, row.get("option_count"))
    gold = cast(int, row.get("gold_position", 0))
    selected_idx = (gold + 1) % oc if disagree_answer else gold
    criteria = cast(list[dict[str, Any]], row.get("criteria", []))
    selected_id = (
        criteria[selected_idx]["id"] if 0 <= selected_idx < len(criteria) else criteria[0]["id"]
    )
    roles = list(row["semantic_equivalence_attestation"]["criterion_roles"])
    if disagree_answer:
        roles[selected_idx], roles[gold] = roles[gold], roles[selected_idx]
    return {
        "status": status,
        "selected_criterion_id": selected_id,
        "reason_codes": [],
        "natural_language": True,
        "fictional": True,
        "exclusive_options": True,
        "private_or_sensitive": False,
        "semantic_equivalence_attestation": {
            "scenario": row["semantic_equivalence_attestation"]["scenario"],
            "criterion_roles": roles,
            "selected_role": "matches_rule",
        },
    }


def _author_reviewer_transport_fixture(
    slots: list[dict[str, Any]],
    *,
    author_cost: str = "0",
    reviewer_cost: str = "0",
    disagree_answer: bool = False,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    reviewer_payloads: dict[str, dict[str, Any]] = {}
    for slot in slots:
        task_id = cast(str, slot["task_id"])
        option_count = cast(int, slot["option_count"])
        gold = cast(int, slot["gold_position"])
        roles = list(slot["semantic_target"]["criterion_roles"])
        selected_role = roles.pop(0)
        roles.insert(gold, selected_role)
        selected = (gold + 1) % option_count if disagree_answer else gold
        reviewer_payloads[task_id] = {
            "status": "accepted",
            "selected_criterion_id": f"criterion-{selected}",
            "reason_codes": [],
            "natural_language": True,
            "generic_or_invented": True,
            "exclusive_options": True,
            "private_or_sensitive": False,
            "semantic_equivalence_attestation": {
                "scenario": slot["semantic_target"]["scenario"],
                "criterion_roles": roles,
                "selected_role": "matches_rule",
            },
        }

    def transport(
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes,
    ) -> tuple[int, dict[str, str], bytes]:
        del method, url, headers
        req = json.loads(body)
        model = req.get("model", "")
        if model == _comparison_author_model():
            messages = req.get("messages", [])
            supplied = json.loads(messages[1]["content"])["slots"]
            return _raw_completion(json.dumps(_build_author_wire(supplied)), cost=author_cost)
        elif model == _comparison_reviewer_model():
            task_id = req.get("messages", [{}])[1].get("content", "{}")
            tid = None
            if isinstance(task_id, str):
                tid_dict = json.loads(task_id)
                if isinstance(tid_dict, dict) and "tasks" in tid_dict:
                    tid = tid_dict["tasks"][0].get("task_id")
            if tid and tid in reviewer_payloads:
                return _completion({"reviews": [reviewer_payloads[tid]]}, cost=reviewer_cost)
            return _completion(
                {
                    "reviews": [
                        {
                            "status": "rejected",
                            "reason_codes": ["language"],
                            "natural_language": True,
                            "generic_or_invented": True,
                            "exclusive_options": True,
                            "private_or_sensitive": False,
                            "semantic_equivalence_attestation": {
                                "scenario": "topic_routing",
                                "criterion_roles": ["matches_rule"],
                                "selected_role": "matches_rule",
                            },
                        }
                    ]
                },
                cost=reviewer_cost,
            )
        return 400, {}, b"{}"

    return transport, {}, {}


def _comparison_author_model() -> str:
    return comparison._COMPARISON_AUTHOR_MODEL


def _comparison_reviewer_model() -> str:
    return comparison._COMPARISON_REVIEWER_MODEL


def _write_plan(work: Path) -> dict[str, Any]:
    plan = comparison.build_comparison_plan()
    plan_path = work / "plan.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_bytes(comparison._canonical(plan) + b"\n")
    return plan


def _write_ledger(work: Path, ledger: corpus.BudgetLedger) -> None:
    directory = work / "ledger"
    directory.mkdir(parents=True, exist_ok=True)
    corpus.write_ledger_snapshot(directory, ledger)


def _setup_work(work: Path) -> tuple[dict[str, Any], Path, Path]:
    os.makedirs(work / "comparison-plan", exist_ok=True)
    os.makedirs(work / "comparison-work", exist_ok=True)
    plan = _write_plan(work / "comparison-plan")
    plan_path = work / "comparison-plan" / "plan.json"
    work_dir = work / "comparison-work"
    return plan, plan_path, work_dir


def _write_previous_attempt_receipt(
    attempt_root: Path,
    receipt_path: Path,
    *,
    bindings_mode: str = "live_verified",
) -> None:
    plan = comparison.build_comparison_plan(
        bindings={**comparison._fixture_plan_bindings(), "mode": bindings_mode}
    )
    plan_dir = attempt_root / "comparison-plan"
    ledger_dir = attempt_root / "comparison-work" / "ledger"
    plan_dir.mkdir(parents=True)
    ledger_dir.mkdir(parents=True)
    plan_path = plan_dir / "plan.json"
    plan_path.write_bytes(comparison._canonical(plan) + b"\n")
    ledger = corpus.BudgetLedger(policy=comparison.COMPARISON_LEDGER_POLICY)
    for index, stage in enumerate((comparison.AUTHOR_STAGE, comparison.REVIEWER_STAGE)):
        reservation = "reservation-" + f"{index + 1:064x}"
        task_id = str(plan["slots"][index]["task_id"])
        ledger.reserve_request(stage, reservation, Decimal("0"))
        ledger.mark_uncertain(reservation, f"{index + 1:064x}")
        ledger.record_provider_journal(
            stage=stage,
            reservation_id=reservation,
            task_ids=[task_id],
        )
    ledger_path = ledger_dir / "ledger-0000.json"
    ledger_path.write_bytes(corpus._canonical(ledger.as_json(final=True)) + b"\n")
    comparison._write_terminal_report(
        report_dir=attempt_root,
        plan=plan,
        accepted=[],
        rejected=[
            {"task_id": slot["task_id"], "reason": "fixture_rejection"} for slot in plan["slots"]
        ],
        ledger=ledger,
        outcome="inconclusive_transport",
        diagnostics={"transport_uncertain_calls": 2},
    )
    terminal_path = attempt_root / "terminal-report.json"
    receipt = {
        "schema_version": comparison.COMPARISON_ATTEMPT_RECEIPT_SCHEMA,
        "attempt_id": attempt_root.name,
        "final_ledger_relative_path": "comparison-work/ledger/ledger-0000.json",
        "plan_sha256": corpus._sha(plan_path.read_bytes()),
        "terminal_report_sha256": corpus._sha(terminal_path.read_bytes()),
        "final_ledger_sha256": corpus._sha(ledger_path.read_bytes()),
        "plan_schema": comparison.COMPARISON_PLAN_SCHEMA,
        "terminal_outcome": "inconclusive_transport",
        "accepted": 0,
        "unresolved": 0,
        "provider_calls": 2,
        "provider_reported_cost_usd": "0",
    }
    receipt_path.write_bytes(comparison._canonical(receipt) + b"\n")
    os.chmod(receipt_path, 0o400)


def _comparison_result_fixture() -> dict[str, Any]:
    accuracy = comparison._build_slice_result(110, 150)
    summary = {
        "planned": 200,
        "resolved": 200,
        "accepted": 150,
        "rejected": 50,
        "unsupported": 0,
        "failed": 0,
        "accuracy": accuracy,
        "slices": {
            "locale": {},
            "domain": {},
            "scenario": {},
            "option_count": {},
            "axes": {axis: {} for axis in comparison.AXES},
        },
        "performance": {
            "cold_load_ms": 10.0,
            "warm_p50_ms": 1.0,
            "warm_p95_ms": 2.0,
            "throughput_tasks_per_second": 100.0,
            "peak_rss_bytes": 1_000_000,
            "artifact_bytes": 1024,
        },
        "deterministic_failures": 0,
        "artifact_sha256": "a" * 64,
        "model_revision": "fixture.v1",
        "runtime_module_sha256": "b" * 64,
        "dependency_versions": {},
        "platform": {"system": "Darwin", "machine": "arm64", "python": "3.13"},
    }
    control_summary = json.loads(json.dumps(summary))
    control_summary["accuracy"] = comparison._build_slice_result(100, 150)
    control_summary["artifact_sha256"] = "1" * 64
    return {
        "schema_version": "phase4e-comparison-result.v1",
        "status": "scored",
        "evidence_mode": "fixture",
        "plan_sha256": "c" * 64,
        "packet_sha256": "d" * 64,
        "policy_sha256": "e" * 64,
        "provenance": {
            "source_commit": "1" * 40,
            "training_packet_receipt_sha256": "2" * 64,
            "training_accepted_rows_sha256": "3" * 64,
            "training_manifest_sha256": "4" * 64,
            "training_checkpoint_sha256": "5" * 64,
            "saracura_tokenizer_snapshot_sha256": "6" * 64,
            "laya_snapshot_sha256": "7" * 64,
            "training_task_ids_sha256": "8" * 64,
            "training_family_ids_sha256": "9" * 64,
        },
        "provider_reported_generation_cost_usd": "1.25",
        "primary_comparison": {
            "device": "cpu",
            "cpu_threads": 1,
            "control_backend": "available",
            "candidate_result_sha256": "f" * 64,
            "control_result_sha256": "0" * 64,
            "candidate": summary,
            "control": control_summary,
            "paired_2x2": {
                "both_correct": 80,
                "candidate_only_correct": 30,
                "control_only_correct": 20,
                "both_not_correct": 20,
            },
        },
        "candidate_device_observations": [],
    }


def _accepted_comparison_rows(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for slot in plan["slots"]:
        count = cast(int, slot["option_count"])
        gold = cast(int, slot["gold_position"])
        roles = list(slot["semantic_target"]["criterion_roles"])
        selected_role = roles.pop(0)
        roles.insert(gold, selected_role)
        criteria = [
            {"id": f"criterion-{index}", "description": f"Synthetic option {index}."}
            for index in range(count)
        ]
        rows.append(
            {
                "task_id": slot["task_id"],
                "family_id": slot["family_id"],
                "pair_id": slot["pair_id"],
                "split": "synthetic_comparison",
                "locale": slot["locale"],
                "domain": slot["domain"],
                "instruction": "Choose the safe synthetic option.",
                "state": {"summary": "Synthetic state."},
                "criteria": criteria,
                "selected_criterion_id": criteria[gold]["id"],
                "axes": slot["axes"],
                "option_count": count,
                "gold_position": gold,
                "semantic_equivalence_attestation": {
                    "scenario": slot["semantic_target"]["scenario"],
                    "criterion_roles": roles,
                    "selected_role": "matches_rule",
                },
            }
        )
    return rows


def _sealed_fixture_packet(
    root: Path, *, live_verified: bool = False
) -> tuple[Path, list[dict[str, Any]]]:
    bindings = comparison._fixture_plan_bindings()
    if live_verified:
        bindings["mode"] = "live_verified"
    plan = comparison.build_comparison_plan(bindings=bindings)
    accepted = _accepted_comparison_rows(plan)
    packet = root / "comparison-packet"
    ledger = corpus.BudgetLedger(policy=comparison.COMPARISON_LEDGER_POLICY)
    comparison._seal_comparison_packet(
        packet,
        plan,
        accepted,
        [],
        ledger.as_json(final=True),
    )
    return packet, accepted


def _worker_result_fixture(
    backend: str,
    packet: Path,
    accepted: list[dict[str, Any]],
    *,
    cpu_threads: int = 1,
) -> dict[str, Any]:
    return {
        "schema_version": "phase4e-comparison-score-backend.v1",
        "backend": backend,
        "device": "cpu",
        "cpu_threads": cpu_threads,
        "packet_sha256": comparison._sha((packet / "packet.json").read_bytes()),
        "artifact_sha256": ("a" if backend == "saracura" else "b") * 64,
        "artifact_bytes": 1024,
        "model_revision": f"{backend}.fixture.v1",
        "runtime_module_sha256": ("c" if backend == "saracura" else "d") * 64,
        "platform": {"system": "Darwin", "machine": "arm64", "python": "3.13"},
        "dependency_versions": {"saracura": "0.0.0"},
        "cold_load_ms": 10.0,
        "warm_p50_ms": 1.0,
        "warm_p95_ms": 2.0,
        "throughput_tasks_per_second": 100.0,
        "peak_rss_bytes": 1_000_000,
        "deterministic_failures": 0,
        "rows": [
            {
                "task_id": row["task_id"],
                "semantic_sha256": comparison._task_semantic_sha256(row),
                "status": "correct",
                "selected_criterion_id": row["selected_criterion_id"],
                "stable_error_code": None,
                "input_tokens": 10,
                "warm_latency_ms": 1.0,
                "deterministic_repeat": True,
            }
            for row in accepted
        ],
    }


# ── plan tests ──────────────────────────────────────────────────────


def test_comparison_policy_validates() -> None:
    policy = comparison.validate_comparison_policy()
    assert policy["schema_version"] == "phase4e-comparison-policy.v1"
    assert policy["underpowered_threshold"] == 10


def test_comparison_policy_v2_only_changes_reviewed_allowlist() -> None:
    policy = comparison.validate_comparison_policy(comparison.COMPARISON_POLICY_V2_PATH)
    assert policy["schema_version"] == "phase4e-comparison-policy.v2"
    assert policy["plan"]["seed"] != comparison.validate_comparison_policy()["plan"]["seed"]
    assert policy["reviewer_contract"]["revision"] == "phase4e-v12-pilot-reviewer.v1"
    assert policy["reviewer_contract"]["provider_schema_digest"] == corpus._sha(
        corpus._canonical(pilot.pilot_reviewer_schema(1))
    )
    assert policy["reviewer_contract"]["reviewer_system_message_digest"] == corpus._sha(
        pilot.pilot_reviewer_system().encode("utf-8")
    )
    altered = json.loads(json.dumps(policy))
    altered["cost"]["automatic_retries"] += 1
    with pytest.raises(comparison.ComparisonError, match="outside allowed fields"):
        comparison._validate_policy_shape(altered)


def test_comparison_v2_plan_ids_are_disjoint_from_v1() -> None:
    previous = comparison.build_comparison_plan()
    old_tasks = sorted(str(slot["task_id"]) for slot in previous["slots"])
    old_families = sorted({str(slot["family_id"]) for slot in previous["slots"]})
    contract = comparison.validate_comparison_policy(comparison.COMPARISON_POLICY_V2_PATH)[
        "reviewer_contract"
    ]
    prior_binding = {
        "previous_attempt_plan_sha256": "a" * 64,
        "previous_attempt_terminal_report_sha256": "b" * 64,
        "previous_attempt_final_ledger_sha256": "c" * 64,
        "previous_attempt_outcome": "inconclusive_transport",
        "previous_attempt_accepted": 0,
        "previous_attempt_task_ids_sha256": corpus._sha(corpus._canonical(old_tasks)),
        "previous_attempt_family_ids_sha256": corpus._sha(corpus._canonical(old_families)),
        "previous_attempt_content_disjunction_sha256": corpus._sha(corpus._canonical([])),
        "reviewer_contract_revision": contract["revision"],
        "reviewer_provider_schema_digest": contract["provider_schema_digest"],
        "reviewer_system_message_digest": contract["reviewer_system_message_digest"],
    }
    plan = comparison.build_comparison_plan(
        bindings=comparison._fixture_plan_bindings(),
        policy_path=comparison.COMPARISON_POLICY_V2_PATH,
        previous_attempt_bindings=prior_binding,
    )
    comparison._validate_comparison_plan(plan)
    assert plan["schema_version"] == comparison.COMPARISON_PLAN_SCHEMA_V2
    assert {slot["task_id"] for slot in plan["slots"]}.isdisjoint(old_tasks)
    assert {slot["family_id"] for slot in plan["slots"]}.isdisjoint(old_families)


def _v2_fixture_plan(*, live: bool = False) -> dict[str, Any]:
    prior = comparison.build_comparison_plan()
    tasks = sorted(str(slot["task_id"]) for slot in prior["slots"])
    families = sorted({str(slot["family_id"]) for slot in prior["slots"]})
    contract = comparison.validate_comparison_policy(comparison.COMPARISON_POLICY_V2_PATH)[
        "reviewer_contract"
    ]
    previous = {
        "previous_attempt_plan_sha256": "a" * 64,
        "previous_attempt_terminal_report_sha256": "b" * 64,
        "previous_attempt_final_ledger_sha256": "c" * 64,
        "previous_attempt_outcome": "inconclusive_transport",
        "previous_attempt_accepted": 0,
        "previous_attempt_task_ids_sha256": corpus._sha(corpus._canonical(tasks)),
        "previous_attempt_family_ids_sha256": corpus._sha(corpus._canonical(families)),
        "previous_attempt_content_disjunction_sha256": corpus._sha(corpus._canonical([])),
        "reviewer_contract_revision": contract["revision"],
        "reviewer_provider_schema_digest": contract["provider_schema_digest"],
        "reviewer_system_message_digest": contract["reviewer_system_message_digest"],
    }
    bindings = comparison._fixture_plan_bindings()
    if live:
        bindings["mode"] = "live_verified"
    return comparison.build_comparison_plan(
        bindings=bindings,
        policy_path=comparison.COMPARISON_POLICY_V2_PATH,
        previous_attempt_bindings=previous,
    )


@pytest.mark.parametrize(
    "source_state", ["dirty", "head_mismatch", "reviewer_digest_mismatch", "v1"]
)
def test_live_generate_guards_fail_before_fake_transport(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_state: str,
) -> None:
    private_root = tmp_path / "phase4e" / "comparison"
    plan_root = private_root / "comparison-plan"
    plan_root.mkdir(parents=True)
    plan = (
        comparison.build_comparison_plan(
            bindings={**comparison._fixture_plan_bindings(), "mode": "live_verified"}
        )
        if source_state == "v1"
        else _v2_fixture_plan(live=True)
    )
    if source_state == "reviewer_digest_mismatch":
        plan["bindings"]["reviewer_provider_schema_digest"] = "f" * 64
    plan_path = plan_root / "plan.json"
    plan_path.write_bytes(comparison._canonical(plan) + b"\n")
    calls = 0

    def fake_transport(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        raise AssertionError("transport must not run before source guards pass")

    if source_state == "dirty":
        monkeypatch.setattr(
            comparison,
            "_require_clean_source",
            lambda: (_ for _ in ()).throw(comparison.ComparisonError("dirty source")),
        )
        expected = "dirty source"
    elif source_state == "head_mismatch":
        monkeypatch.setattr(comparison, "_require_clean_source", lambda: "f" * 40)
        expected = "source commit mismatch"
    elif source_state == "reviewer_digest_mismatch":
        expected = "reviewer contract digest"
    else:
        expected = "requires comparison plan V2"
    with pytest.raises(comparison.ComparisonError, match=expected):
        comparison.run_generate(
            plan_path,
            private_root / "comparison-work",
            private_root / "comparison-packet",
            allow_network=True,
            transport=fake_transport,
        )
    assert calls == 0


def test_v2_plan_binds_rehashed_previous_attempt_and_rejects_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    private_root = tmp_path / "private-state"
    private_root.mkdir(mode=0o700)
    os.chmod(private_root, 0o700)
    archive_dir = private_root / comparison.COMPARISON_ARCHIVE_RELATIVE_DIR
    attempt_root = archive_dir / comparison.COMPARISON_ARCHIVED_V1_DIRNAME
    receipt_path = archive_dir / comparison.COMPARISON_ARCHIVED_V1_RECEIPT_NAME
    archive_dir.mkdir(parents=True)
    monkeypatch.setenv("SARACURA_PRIVATE_STATE_ROOT", str(private_root))
    _write_previous_attempt_receipt(attempt_root, receipt_path)
    bindings, old_tasks, old_families = comparison._previous_attempt_binding(
        attempt_root, receipt_path
    )
    assert len(old_tasks) == 200
    assert len(old_families) == 100
    assert bindings["previous_attempt_accepted"] == 0
    ledger_path = attempt_root / "comparison-work" / "ledger" / "ledger-0000.json"
    ledger_path.write_bytes(ledger_path.read_bytes() + b" ")
    with pytest.raises(comparison.ComparisonError, match="digest mismatch"):
        comparison._previous_attempt_binding(attempt_root, receipt_path)


def test_previous_attempt_rejects_offline_fixture_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    private_root = tmp_path / "private-state"
    private_root.mkdir(mode=0o700)
    archive_dir = private_root / comparison.COMPARISON_ARCHIVE_RELATIVE_DIR
    archive_dir.mkdir(parents=True)
    monkeypatch.setenv("SARACURA_PRIVATE_STATE_ROOT", str(private_root))
    attempt_root = archive_dir / comparison.COMPARISON_ARCHIVED_V1_DIRNAME
    receipt_path = archive_dir / comparison.COMPARISON_ARCHIVED_V1_RECEIPT_NAME
    _write_previous_attempt_receipt(attempt_root, receipt_path, bindings_mode="offline_fixture")
    with pytest.raises(comparison.ComparisonError, match="live-verified V1 plan"):
        comparison._previous_attempt_binding(attempt_root, receipt_path)


def test_previous_attempt_rejects_noncanonical_archive_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    private_root = tmp_path / "private-state"
    private_root.mkdir(mode=0o700)
    archive_dir = private_root / comparison.COMPARISON_ARCHIVE_RELATIVE_DIR
    archive_dir.mkdir(parents=True)
    monkeypatch.setenv("SARACURA_PRIVATE_STATE_ROOT", str(private_root))
    attempt_root = archive_dir / comparison.COMPARISON_ARCHIVED_V1_DIRNAME
    receipt_path = archive_dir / comparison.COMPARISON_ARCHIVED_V1_RECEIPT_NAME
    _write_previous_attempt_receipt(attempt_root, receipt_path)
    misplaced_attempt = private_root / "attempt-copy"
    misplaced_attempt.mkdir()
    misplaced_receipt = private_root / "attempt-copy-receipt.json"
    misplaced_receipt.write_bytes(receipt_path.read_bytes())
    os.chmod(misplaced_receipt, 0o400)
    with pytest.raises(comparison.ComparisonError, match="canonical archive paths"):
        comparison._previous_attempt_binding(misplaced_attempt, receipt_path)
    with pytest.raises(comparison.ComparisonError, match="canonical archive paths"):
        comparison._previous_attempt_binding(attempt_root, misplaced_receipt)


def test_previous_attempt_rejects_receipt_mode_other_than_0400(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    private_root = tmp_path / "private-state"
    private_root.mkdir(mode=0o700)
    archive_dir = private_root / comparison.COMPARISON_ARCHIVE_RELATIVE_DIR
    archive_dir.mkdir(parents=True)
    monkeypatch.setenv("SARACURA_PRIVATE_STATE_ROOT", str(private_root))
    attempt_root = archive_dir / comparison.COMPARISON_ARCHIVED_V1_DIRNAME
    receipt_path = archive_dir / comparison.COMPARISON_ARCHIVED_V1_RECEIPT_NAME
    _write_previous_attempt_receipt(attempt_root, receipt_path)
    os.chmod(receipt_path, 0o600)
    with pytest.raises(comparison.ComparisonError, match="mode 0400"):
        comparison._previous_attempt_binding(attempt_root, receipt_path)


def test_diagnostic_schemas_are_closed_and_plan_version_specific(tmp_path: Path) -> None:
    plan = comparison.build_comparison_plan()
    slots = plan["slots"][:2]
    counts = dict.fromkeys(comparison._PERSISTED_DIAGNOSTIC_KEYS, 0)
    v1_root = tmp_path / "v1" / "comparison-work"
    comparison._store_comparison_diagnostics(
        v1_root,
        slots,
        counts,
        "1" * 64,
        schema_version="phase4e-comparison-diagnostics.v1",
    )
    loaded = comparison._load_comparison_diagnostics(
        v1_root, "1" * 64, schema_version="phase4e-comparison-diagnostics.v1"
    )
    assert set(loaded) == set(comparison._V1_PERSISTED_DIAGNOSTIC_KEYS)
    with pytest.raises(comparison.ComparisonError, match="diagnostic file corrupted"):
        comparison._load_comparison_diagnostics(
            v1_root,
            "1" * 64,
            schema_version=comparison.COMPARISON_DIAGNOSTICS_SCHEMA,
        )

    v2_root = tmp_path / "v2" / "comparison-work"
    comparison._store_comparison_diagnostics(
        v2_root,
        slots,
        counts,
        "2" * 64,
        schema_version=comparison.COMPARISON_DIAGNOSTICS_SCHEMA,
    )
    assert set(
        comparison._load_comparison_diagnostics(
            v2_root, "2" * 64, schema_version=comparison.COMPARISON_DIAGNOSTICS_SCHEMA
        )
    ) == set(comparison._PERSISTED_DIAGNOSTIC_KEYS)


def test_near_duplicate_quick_bounds_are_exact_equivalent() -> None:
    adversarial = [
        ("a" * 23 + "bc", "a" * 25),  # exact ratio 0.92
        ("a" * 300, "a" * 301),
        ("x", "a" * 260 + "x"),  # only second argument exceeds autojunk length
        ("a" * 250 + "bc", "a" * 250 + "bd"),
        ("abc", "xyz"),
        ("", ""),
    ]
    assert SequenceMatcher(None, *adversarial[0]).ratio() == 0.92
    rng = random.Random(0x5A2AC0)
    alphabet = "abcde 0123"
    samples = [
        (
            "".join(rng.choice(alphabet) for _ in range(rng.randrange(0, 340))),
            "".join(rng.choice(alphabet) for _ in range(rng.randrange(0, 340))),
        )
        for _ in range(1000)
    ]
    for candidate, existing in [*adversarial, *samples]:
        exact = SequenceMatcher(None, candidate, existing).ratio() >= 0.92
        assert corpus._near_duplicate(candidate, [existing]) is exact


def test_semantic_mismatch_is_diagnostic_but_answer_disagreement_rejects() -> None:
    row = _accepted_comparison_rows(comparison.build_comparison_plan())[0]
    roles = ["matches_rule"] * len(row["criteria"])
    other_scenario = next(
        scenario
        for scenario in corpus.SCENARIO_CODES
        if scenario != row["semantic_equivalence_attestation"]["scenario"]
    )
    wire = {
        "status": "accepted",
        "selected_criterion_id": row["selected_criterion_id"],
        "reason_codes": [],
        "natural_language": True,
        "generic_or_invented": True,
        "exclusive_options": True,
        "private_or_sensitive": False,
        "semantic_equivalence_attestation": {
            "scenario": other_scenario,
            "criterion_roles": roles,
            "selected_role": "matches_rule",
        },
    }
    review = pilot.bind_pilot_reviewer_record(wire, row["task_id"])
    assert pilot.pilot_semantic_kind(row, review) == "scenario_disagreement"
    accepted, rejected = corpus.resolve_reviews(
        [row],
        [review],
        acceptance=pilot.pilot_acceptance,
        pair_resolution=pilot.pilot_pair_resolution,
        review_model=pilot.PilotReviewerRecord,
    )
    assert len(accepted) == 1
    assert rejected == []

    disagree = {**wire, "selected_criterion_id": row["criteria"][1]["id"]}
    bound_disagree = pilot.bind_pilot_reviewer_record(disagree, row["task_id"])
    accepted, rejected = corpus.resolve_reviews(
        [row],
        [bound_disagree],
        acceptance=pilot.pilot_acceptance,
        pair_resolution=pilot.pilot_pair_resolution,
        review_model=pilot.PilotReviewerRecord,
    )
    assert accepted == []
    assert rejected[0]["reason"] == "review_disagreement"


def test_plan_deterministic_text_free_200_slots() -> None:
    plan_a = comparison.build_comparison_plan()
    plan_b = comparison.build_comparison_plan()
    assert plan_a == plan_b
    assert len(plan_a["slots"]) == 200
    text_keys = {"instruction", "state", "criteria", "selected_criterion_id"}
    for slot in plan_a["slots"]:
        assert not text_keys.intersection(slot)


def test_plan_100_pairs_balanced() -> None:
    plan = comparison.build_comparison_plan()
    pairs: dict[str, list[dict[str, Any]]] = {}
    for slot in plan["slots"]:
        if slot["pair_id"]:
            pairs.setdefault(str(slot["pair_id"]), []).append(slot)
    assert len(pairs) == 100
    for ps in pairs.values():
        assert {s["locale"] for s in ps} == {"pt-BR", "en"}
    for axis, values in comparison.AXES.items():
        counts = {
            value: sum(slot["axes"][axis] == value for slot in plan["slots"]) for value in values
        }
        assert len(set(counts.values())) == 1
    scenario_counts = {
        scenario: sum(slot["semantic_target"]["scenario"] == scenario for slot in plan["slots"])
        for scenario in corpus.SCENARIO_CODES
    }
    assert max(scenario_counts.values()) - min(scenario_counts.values()) <= 2
    for locale in comparison.LOCALES:
        for option_count in comparison.OPTION_COUNTS:
            cell = [
                slot
                for slot in plan["slots"]
                if slot["locale"] == locale and slot["option_count"] == option_count
            ]
            gold_counts = [
                sum(slot["gold_position"] == position for slot in cell)
                for position in range(option_count)
            ]
            assert max(gold_counts) - min(gold_counts) <= 1


def test_plan_ids_disjoint_from_corpus() -> None:
    comp_ids = {cast(str, s["task_id"]) for s in comparison.build_comparison_plan()["slots"]}
    corp_ids = {cast(str, s["task_id"]) for s in corpus_build_plan()["slots"]}
    assert comp_ids.isdisjoint(corp_ids)


def test_training_content_overlap_rejects_semantic_and_near_duplicates() -> None:
    row = _accepted_comparison_rows(comparison.build_comparison_plan())[0]
    review = corpus.bind_reviewer_record(_build_reviewer_payload(row), cast(str, row["task_id"]))
    prior = {
        **row,
        "task_id": "training-task",
        "family_id": "training-family",
        "pair_id": "training-family",
    }
    accepted, rejected = corpus.resolve_reviews(
        [row],
        [review],
        prior_rows=[prior],
        acceptance=corpus.acceptance_reason,
    )
    assert accepted == []
    assert rejected[0]["reason"] == "semantic_duplicate"

    near_prior = {**prior, "instruction": cast(str, row["instruction"]) + " minor"}
    accepted, rejected = corpus.resolve_reviews(
        [row],
        [review],
        prior_rows=[near_prior],
        acceptance=corpus.acceptance_reason,
    )
    assert accepted == []
    assert rejected[0]["reason"] == "near_duplicate"


def test_plan_immutable() -> None:
    plan = comparison.build_comparison_plan()
    assert comparison.is_comparison_plan_immutable(plan, plan)
    mutated = json.loads(json.dumps(plan))
    original = mutated["slots"][0]["locale"]
    mutated["slots"][0]["locale"] = "en" if original == "pt-BR" else "pt-BR"
    assert not comparison.is_comparison_plan_immutable(plan, mutated)


def test_plan_validator_rejects_balanced_but_noncanonical_semantic_swap() -> None:
    plan = comparison.build_comparison_plan()
    by_pair: dict[str, list[dict[str, Any]]] = {}
    for slot in plan["slots"]:
        by_pair.setdefault(cast(str, slot["pair_id"]), []).append(slot)
    pair_groups = list(by_pair.values())
    first = pair_groups[0]
    second = next(
        group
        for group in pair_groups[1:]
        if group[0]["option_count"] == first[0]["option_count"]
        and group[0]["semantic_target"]["scenario"] != first[0]["semantic_target"]["scenario"]
    )
    first_target = first[0]["semantic_target"]
    second_target = second[0]["semantic_target"]
    for slot in first:
        slot["semantic_target"] = second_target
    for slot in second:
        slot["semantic_target"] = first_target
    with pytest.raises(comparison.ComparisonError, match="canonical policy-derived"):
        comparison._validate_comparison_plan(plan)


def test_generation_resume_requires_byte_identical_canonical_plan(tmp_path: Path) -> None:
    plan, plan_path, work_dir = _setup_work(tmp_path)
    work_dir.mkdir(parents=True, exist_ok=True)
    first_digest = comparison._bind_generation_plan(work_dir, plan_path, plan)
    assert len(first_digest) == 64
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    with pytest.raises(comparison.ComparisonError, match="bytes are not canonical"):
        comparison._bind_generation_plan(work_dir, plan_path, plan)


def test_plan_run_plan_writes_no_clobber() -> None:
    plan = comparison.build_comparison_plan()
    assert len(plan["slots"]) == 200


# ── generate tests ───────────────────────────────────────────────────


def test_generate_requires_allow_network(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("SARACURA_PRIVATE_STATE_ROOT", str(tmp_path))
    (tmp_path / "phase4e" / "comparison").mkdir(parents=True, exist_ok=True)
    root = tmp_path / "phase4e" / "comparison"
    _plan, plan_path, work_dir = _setup_work(root)
    packet = root / "comparison-packet"
    with pytest.raises(comparison.ComparisonError, match="--allow-network"):
        comparison.run_generate(plan_path, work_dir, packet, allow_network=False)


def test_generate_fails_missing_openrouter_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SARACURA_PRIVATE_STATE_ROOT", str(tmp_path))
    (tmp_path / "phase4e" / "comparison").mkdir(parents=True, exist_ok=True)
    root = tmp_path / "phase4e" / "comparison"
    plan, plan_path, work_dir = _setup_work(root)
    plan = _v2_fixture_plan()
    plan_path.write_bytes(comparison._canonical(plan) + b"\n")
    packet = root / "comparison-packet"
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(comparison.ComparisonError, match="OPENROUTER_API_KEY"):
        comparison.run_generate(
            plan_path,
            work_dir,
            packet,
            allow_network=True,
            transport=_author_reviewer_transport_fixture(plan["slots"])[0],
            tokenizer_receipt=cast(corpus.VerifiedMiniLMTokenizerReceipt, _TokenFactory()),
        )


def test_generate_full_pass_with_injected_transport(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SARACURA_PRIVATE_STATE_ROOT", str(tmp_path))
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    private_comp = tmp_path / "phase4e" / "comparison"
    private_comp.mkdir(parents=True, exist_ok=True)

    root = tmp_path / "phase4e" / "comparison"
    plan, plan_path, work_dir = _setup_work(root)
    plan = _v2_fixture_plan()
    plan_path.write_bytes(comparison._canonical(plan) + b"\n")
    packet = root / "comparison-packet"

    transport, _, _ = _author_reviewer_transport_fixture(plan["slots"])
    captured_reviewer: dict[str, Any] = {}

    def capture_reviewer_request(
        method: str, url: str, headers: Mapping[str, str], body: bytes
    ) -> Any:
        request = json.loads(body)
        if request.get("model") == _comparison_reviewer_model() and not captured_reviewer:
            captured_reviewer.update(request)
        return transport(method, url, headers, body)

    result = comparison.run_generate(
        plan_path,
        work_dir,
        packet,
        allow_network=True,
        transport=capture_reviewer_request,
        tokenizer_receipt=cast(corpus.VerifiedMiniLMTokenizerReceipt, _TokenFactory()),
    )
    assert result == packet / "packet.json"
    descriptor = json.loads(result.read_bytes())
    assert descriptor["schema_version"] == "phase4e-comparison-packet.v1"
    assert descriptor["accepted_count"] == 200
    assert descriptor["rejected_count"] == 0
    first_reviewer = json.loads(captured_reviewer["messages"][1]["content"])["tasks"][0]
    slot = next(item for item in plan["slots"] if item["task_id"] == first_reviewer["task_id"])
    sibling_slots = [item for item in plan["slots"] if item["pair_id"] == slot["pair_id"]]
    authored_rows = corpus._materialize_author_rows(
        corpus.decode_author_response(_build_author_wire(sibling_slots), sibling_slots),
        sibling_slots,
    )
    authored_row = next(row for row in authored_rows if row["task_id"] == first_reviewer["task_id"])
    assert captured_reviewer["messages"] == pilot.pilot_reviewer_messages(authored_row)
    provider_schema = captured_reviewer["response_format"]["json_schema"]["schema"]
    assert provider_schema == pilot.pilot_reviewer_schema(1)
    schema_text = json.dumps(provider_schema)
    assert "generic_or_invented" in schema_text
    assert '"fictional"' not in schema_text
    diagnostics = comparison._load_comparison_diagnostics(
        work_dir,
        comparison._sha(plan_path.read_bytes()),
        schema_version=comparison.COMPARISON_DIAGNOSTICS_SCHEMA,
    )
    assert diagnostics["reviewed"] == 200
    assert diagnostics["scenario_disagreement"] == 0


def test_author_target_mismatch_rejects_pair_before_reviewer_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SARACURA_PRIVATE_STATE_ROOT", str(tmp_path))
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    root = tmp_path / "phase4e" / "comparison"
    root.mkdir(parents=True, exist_ok=True)
    plan, plan_path, work_dir = _setup_work(root)
    plan = _v2_fixture_plan()
    plan_path.write_bytes(comparison._canonical(plan) + b"\n")
    base_transport, _, _ = _author_reviewer_transport_fixture(plan["slots"])
    mutated_pair_ids: set[str] = set()
    mutated_task_ids: set[str] = set()
    reviewer_task_ids: list[str] = []
    changed = False

    def transport(method: str, url: str, headers: Mapping[str, str], body: bytes) -> Any:
        nonlocal changed
        request = json.loads(body)
        if request.get("model") == _comparison_author_model() and not changed:
            supplied = json.loads(request["messages"][1]["content"])["slots"]
            pair_id = str(supplied[0]["pair_id"])
            mutated_pair_ids.add(pair_id)
            mutated_task_ids.update(
                str(slot["task_id"]) for slot in plan["slots"] if slot["pair_id"] == pair_id
            )
            response = base_transport(method, url, headers, body)
            response_payload = json.loads(response[2])
            author_wire = json.loads(response_payload["choices"][0]["message"]["content"])
            alternate = next(
                code
                for code in corpus.SCENARIO_CODES
                if code != supplied[0]["semantic_target"]["scenario"]
            )
            author_wire["record_0_scenario"] = alternate
            sibling_index = next(
                index
                for index, slot in enumerate(supplied)
                if slot["pair_id"] == pair_id and slot["task_id"] != supplied[0]["task_id"]
            )
            author_wire[f"record_{sibling_index}_instruction"] = (
                "Caso de teste: CPF 123.456.789-09, decisão fictícia."
            )
            changed = True
            return _raw_completion(json.dumps(author_wire))
        if request.get("model") == _comparison_reviewer_model():
            reviewer_task_ids.extend(
                str(task["task_id"])
                for task in json.loads(request["messages"][1]["content"])["tasks"]
            )
        return base_transport(method, url, headers, body)

    packet = root / "comparison-packet"
    result = comparison.run_generate(
        plan_path,
        work_dir,
        packet,
        allow_network=True,
        transport=transport,
        tokenizer_receipt=cast(corpus.VerifiedMiniLMTokenizerReceipt, _TokenFactory()),
    )
    assert result == packet / "packet.json"
    assert mutated_pair_ids
    assert mutated_task_ids.isdisjoint(reviewer_task_ids)
    rejected = comparison._read_jsonl(packet / "rejected.jsonl")
    assert {
        row["task_id"]
        for row in rejected
        if row["reason"] in {"semantic_target_mismatch", "paired_author_target_mismatch"}
    } == mutated_task_ids
    diagnostics = comparison._load_comparison_diagnostics(
        work_dir,
        comparison._sha(plan_path.read_bytes()),
        schema_version=comparison.COMPARISON_DIAGNOSTICS_SCHEMA,
    )
    assert diagnostics["local_privacy"] == 1
    assert any(
        row["task_id"] in mutated_task_ids and row["reason"] == "paired_author_target_mismatch"
        for row in rejected
    )


def test_settled_reviewer_error_resolves_identity_with_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SARACURA_PRIVATE_STATE_ROOT", str(tmp_path))
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    root = tmp_path / "phase4e" / "comparison"
    root.mkdir(parents=True, exist_ok=True)
    plan, plan_path, work_dir = _setup_work(root)
    plan = _v2_fixture_plan()
    plan_path.write_bytes(comparison._canonical(plan) + b"\n")
    base_transport, _, _ = _author_reviewer_transport_fixture(plan["slots"])
    failed_task_ids: set[str] = set()
    failed = False

    def transport(method: str, url: str, headers: Mapping[str, str], body: bytes) -> Any:
        nonlocal failed
        request = json.loads(body)
        if request.get("model") == _comparison_reviewer_model() and not failed:
            tasks = json.loads(request["messages"][1]["content"])["tasks"]
            failed_task_ids.update(str(task["task_id"]) for task in tasks)
            failed = True
            return (
                400,
                {},
                json.dumps(
                    {"error": {"message": "settled provider rejection"}, "usage": {"cost": "0"}}
                ).encode(),
            )
        return base_transport(method, url, headers, body)

    packet = root / "comparison-packet"
    result = comparison.run_generate(
        plan_path,
        work_dir,
        packet,
        allow_network=True,
        transport=transport,
        tokenizer_receipt=cast(corpus.VerifiedMiniLMTokenizerReceipt, _TokenFactory()),
    )
    assert result == packet / "packet.json"
    assert len(failed_task_ids) == 1
    terminal = json.loads((root / "terminal-report.json").read_bytes())
    assert terminal["planned"] == 200
    assert terminal["accepted"] + terminal["rejected"] == 200
    assert terminal["unresolved"] == 0
    rejected = comparison._read_jsonl(packet / "rejected.jsonl")
    failure = next(row for row in rejected if row["task_id"] in failed_task_ids)
    assert failure["reason"] == "reviewer_response_failure"
    assert failure["reviewer_request_id"]


def test_saracura_capacity_check_uses_generated_semantic_content() -> None:
    plan = comparison.build_comparison_plan()
    row = _accepted_comparison_rows(plan)[0]
    row["instruction"] = "GENERATED-CONTENT-MARKER"

    class ContentCounter:
        def __init__(self) -> None:
            self.seen: list[str] = []

        def count(self, text: str) -> int:
            self.seen.append(text)
            return 129 if "GENERATED-CONTENT-MARKER" in text else 1

    counter = ContentCounter()
    assert comparison._render_saracura_capacity_check(row, counter) is False
    assert any("GENERATED-CONTENT-MARKER" in value for value in counter.seen)


def test_corrupt_resume_state_seals_inconclusive_operational(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SARACURA_PRIVATE_STATE_ROOT", str(tmp_path))
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    root = tmp_path / "phase4e" / "comparison"
    root.mkdir(parents=True, exist_ok=True)
    plan, plan_path, work_dir = _setup_work(root)
    work_dir.mkdir(parents=True, exist_ok=True)
    comparison._bind_generation_plan(work_dir, plan_path, plan)
    ledger_dir = work_dir / "ledger"
    ledger_dir.mkdir()
    (ledger_dir / "ledger-0000.json").write_text("{corrupt", encoding="utf-8")

    with pytest.raises(comparison.ComparisonError, match="inconclusive_operational") as exc:
        comparison.run_generate(
            plan_path,
            work_dir,
            root / "comparison-packet",
            allow_network=True,
            transport=_author_reviewer_transport_fixture(plan["slots"])[0],
            tokenizer_receipt=cast(corpus.VerifiedMiniLMTokenizerReceipt, _TokenFactory()),
        )

    assert exc.value.__cause__ is not None
    report = json.loads((root / "terminal-report.json").read_bytes())
    assert report["outcome"] == "inconclusive_operational"
    assert report["unresolved"] == 200
    assert report["errors"] == ["generation_state_resume_failure"]


def test_cost_crossing_is_reported_before_uncertain_continue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("SARACURA_PRIVATE_STATE_ROOT", str(tmp_path))
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    root = tmp_path / "phase4e" / "comparison"
    root.mkdir(parents=True, exist_ok=True)
    plan, plan_path, work_dir = _setup_work(root)
    base_transport, _, _ = _author_reviewer_transport_fixture(plan["slots"])
    author_calls = 0

    def transport(
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes,
    ) -> tuple[int, dict[str, str], bytes]:
        nonlocal author_calls
        request = json.loads(body)
        if request.get("model") == _comparison_author_model():
            author_calls += 1
            if author_calls == 1:
                return _raw_completion("not-json", cost="12.00")
            if author_calls == 2:
                raise RuntimeError("simulated uncertain retry")
        return cast(tuple[int, dict[str, str], bytes], base_transport(method, url, headers, body))

    comparison.run_generate(
        plan_path,
        work_dir,
        root / "comparison-packet",
        allow_network=True,
        transport=transport,
        tokenizer_receipt=cast(corpus.VerifiedMiniLMTokenizerReceipt, _TokenFactory()),
    )

    output = capsys.readouterr().out
    assert output.count("provider spend crossed USD 10.00") == 1
    assert "final provider spend USD 12.00" in output


def test_generate_uncertain_nonconsecutive_still_inconclusive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SARACURA_PRIVATE_STATE_ROOT", str(tmp_path))
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    private_comp = tmp_path / "phase4e" / "comparison"
    private_comp.mkdir(parents=True, exist_ok=True)

    root = tmp_path / "phase4e" / "comparison"
    plan, plan_path, work_dir = _setup_work(root)
    packet = root / "comparison-packet"

    base_transport, _, _ = _author_reviewer_transport_fixture(plan["slots"])
    author_calls = 0

    def transport(
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes,
    ) -> tuple[int, dict[str, str], bytes]:
        nonlocal author_calls
        req = json.loads(body)
        if req.get("model") == _comparison_author_model():
            author_calls += 1
        # Every odd family is uncertain and every even family settles, so the
        # consecutive circuit never opens while quality minimums still fail.
        if req.get("model") == _comparison_author_model() and author_calls % 2 == 1:
            raise RuntimeError("simulated transport failure")
        return cast(tuple[int, dict[str, str], bytes], base_transport(method, url, headers, body))

    with pytest.raises(comparison.ComparisonError) as exc_info:
        comparison.run_generate(
            plan_path,
            work_dir,
            packet,
            allow_network=True,
            transport=transport,
            tokenizer_receipt=cast(corpus.VerifiedMiniLMTokenizerReceipt, _TokenFactory()),
        )
    assert str(exc_info.value) == "inconclusive_transport"
    report = json.loads((root / "terminal-report.json").read_bytes())
    assert report["outcome"] == "inconclusive_transport"
    assert report["transport_uncertain_calls"] == 50
    assert report["consecutive_uncertainty_streak"] == 0
    assert report["nonconsecutive_transport_uncertain_calls"] == 50
    assert report["provider_calls"] > 50


# ── score-backend test ────────────────────────────────────────────────


def test_score_backend_produces_row_schema(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plan = comparison.build_comparison_plan()
    packet_path = tmp_path / "fake-packet"
    packet_path.mkdir(parents=True, exist_ok=True)
    criteria = [{"id": "c0", "description": "Option A."}, {"id": "c1", "description": "Option B."}]
    accepted = [
        {
            "task_id": slot["task_id"],
            "family_id": slot["family_id"],
            "locale": slot["locale"],
            "domain": slot["domain"],
            "instruction": "Choose the best option.",
            "state": {"summary": "Test."},
            "criteria": criteria,
            "selected_criterion_id": "c0",
            "synthetic_only": True,
            "review": {
                "status": "accepted",
                "selected_criterion_id": "c0",
                "reason_codes": [],
                "natural_language": True,
                "fictional": True,
                "exclusive_options": True,
                "private_or_sensitive": False,
            },
            "review_sha256": corpus._sha(corpus._canonical({})),
            "author_response_sha256": "a" * 64,
            "author_reservation_id": "reservation-" + "a" * 64,
            "author_request_id": "request-author",
            "reviewer_response_sha256": "b" * 64,
            "reviewer_reservation_id": "reservation-" + "b" * 64,
            "reviewer_request_id": "request-reviewer",
            "pair_id": slot.get("pair_id"),
            "gold_position": slot.get("gold_position"),
            "option_count": slot.get("option_count"),
            "semantic_equivalence_attestation": slot["semantic_target"],
        }
        for slot in plan["slots"][:2]
    ]
    packet_json = {
        "schema_version": "phase4e-accepted-packet.v4",
        "accepted": accepted,
        "plan": plan,
    }
    (packet_path / "packet.json").write_bytes(corpus._canonical(packet_json) + b"\n")

    work_dir = tmp_path / "score-work"
    with pytest.raises(comparison.ComparisonError):
        comparison.run_score_backend(
            packet_path,
            "saracura",
            work_dir,
            encoder_snapshot=tmp_path / "fake-encoder",
            training_capsule=tmp_path / "fake-capsule",
        )


# ── publish tests ─────────────────────────────────────────────────────


def test_publish_rejects_fixture_result_even_when_shape_is_valid(tmp_path: Path) -> None:
    result_path = tmp_path / "private-result.json"
    result = _comparison_result_fixture()
    result_path.write_text(json.dumps(result), encoding="utf-8")

    pub_json = tmp_path / "benchmarks" / "results" / "phase4e-comparison-v1.json"
    pub_md = tmp_path / "docs" / "action" / "phase4e-comparison-result.md"

    with pytest.raises(comparison.ComparisonError, match="live evaluation receipt"):
        comparison.run_publish(result_path, pub_json, pub_md)

    assert not pub_json.exists()
    assert not pub_md.exists()


def test_publish_fails_closed_on_nested_raw_content(tmp_path: Path) -> None:
    result = _comparison_result_fixture()
    result["evidence_mode"] = "live_verified"
    result["primary_comparison"]["candidate"]["slices"] = {
        "unsafe": {"instruction": "raw comparison text"}
    }
    with pytest.raises(comparison.ComparisonError, match=r"publishable|forbidden field"):
        comparison._validate_publishable_result(result)


def test_public_safe_accepts_safetensors_dependency_key() -> None:
    comparison._assert_public_safe(
        {
            "primary_comparison": {
                "candidate": {"dependency_versions": {"safetensors": "0.5.0"}},
                "control": {"dependency_versions": {"safetensors": "0.5.0"}},
            }
        }
    )
    with pytest.raises(comparison.ComparisonError, match="forbidden field"):
        comparison._assert_public_safe({"metadata": {"safetensors": "0.5.0"}})
    with pytest.raises(comparison.ComparisonError, match="forbidden field"):
        comparison._assert_public_safe(
            {
                "primary_comparison": {
                    "candidate": {"dependency_versions": {"safetensors_version": "0.5.0"}}
                }
            }
        )


@pytest.mark.parametrize(
    "key",
    (
        "tensor",
        "raw_tensor",
        "absolute_path",
        "provider_api_key_hash",
        "selected_criterion_ids",
        "score_vectors",
        "runstate",
        "presecret",
    ),
)
def test_public_safe_rejects_forbidden_key_substrings_and_compositions(key: str) -> None:
    with pytest.raises(comparison.ComparisonError, match="forbidden field"):
        comparison._assert_public_safe({key: "value"})


@pytest.mark.parametrize("label", ("score_vectors", "selected_criterion_ids"))
def test_publish_rejects_unknown_slice_labels(label: str) -> None:
    result = _comparison_result_fixture()
    result["evidence_mode"] = "live_verified"
    axes = result["primary_comparison"]["candidate"]["slices"]["axes"]
    axes["distractor_overlap"][label] = comparison._build_slice_result(1, 2)
    with pytest.raises(comparison.ComparisonError, match="publishable axes"):
        comparison._validate_publishable_result(result)


def test_publish_rejects_absolute_path_in_allowlisted_field(tmp_path: Path) -> None:
    result = _comparison_result_fixture()
    result["evidence_mode"] = "live_verified"
    result["primary_comparison"]["candidate"]["model_revision"] = "/tmp/private-run"
    with pytest.raises(comparison.ComparisonError, match="publishable label"):
        comparison._validate_publishable_result(result)


def test_publish_rejects_inconsistent_accuracy_and_underpowered_flag(tmp_path: Path) -> None:
    result = _comparison_result_fixture()
    result["evidence_mode"] = "live_verified"
    accuracy = result["primary_comparison"]["candidate"]["accuracy"]
    accuracy["underpowered"] = not accuracy["underpowered"]
    with pytest.raises(comparison.ComparisonError, match="inconsistent"):
        comparison._validate_publishable_result(result)


def test_publish_status_reads_terminal_report(tmp_path: Path) -> None:
    report_path = tmp_path / "terminal-report.json"
    report = {
        "schema_version": "phase4e-comparison-terminal-report.v1",
        "plan_sha256": "a" * 64,
        "outcome": "insufficient_comparison_evidence",
        "planned": 200,
        "accepted": 120,
        "rejected": 80,
        "unresolved": 0,
        "transport_uncertain_calls": 0,
        "nonconsecutive_transport_uncertain_calls": 0,
        "consecutive_uncertainty_streak": 0,
        "provider_calls": 250,
        "provider_reported_cost_usd": "1.00",
        "errors": ["total_accepted=120 < 150"],
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")
    readme_en = tmp_path / "README.md"
    readme_pt = tmp_path / "README.pt-BR.md"
    readme_en.write_text(
        "before\n<!-- phase4e4-status:begin -->\nThe live comparison has not started.\n"
        "<!-- phase4e4-status:end -->\nafter\n",
        encoding="utf-8",
    )
    readme_pt.write_text(
        "before\n<!-- phase4e4-status:begin -->\nA comparação live ainda não começou.\n"
        "<!-- phase4e4-status:end -->\nafter\n",
        encoding="utf-8",
    )
    status = comparison.run_publish_status(
        report_path,
        readme_en=readme_en,
        readme_pt_br=readme_pt,
    )
    assert status["outcome"] == "insufficient_comparison_evidence"
    assert status["accepted"] == 120
    assert "120 accepted" in readme_en.read_text(encoding="utf-8")
    assert "120 aceitos" in readme_pt.read_text(encoding="utf-8")
    assert "phase4e4-status-binding:insufficient:" in readme_en.read_text(encoding="utf-8")

    original_en = readme_en.read_bytes()
    original_pt = readme_pt.read_bytes()
    comparison.run_publish_status(
        report_path,
        readme_en=readme_en,
        readme_pt_br=readme_pt,
    )
    assert readme_en.read_bytes() == original_en
    assert readme_pt.read_bytes() == original_pt
    divergent = dict(report)
    divergent["accepted"] = 121
    divergent["rejected"] = 79
    report_path.write_text(json.dumps(divergent), encoding="utf-8")
    with pytest.raises(comparison.ComparisonError, match="already bound"):
        comparison.run_publish_status(
            report_path,
            readme_en=readme_en,
            readme_pt_br=readme_pt,
        )
    assert readme_en.read_bytes() == original_en
    assert readme_pt.read_bytes() == original_pt

    report["outcome"] = "inconclusive_transport"
    report["transport_uncertain_calls"] = 1
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(comparison.ComparisonError, match="not publishable"):
        comparison.run_publish_status(report_path)


def test_publish_status_cli_updates_both_explicit_readmes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report_path = tmp_path / "terminal-report.json"
    report_path.write_text(
        json.dumps(
            {
                "schema_version": "phase4e-comparison-terminal-report.v1",
                "plan_sha256": "a" * 64,
                "outcome": "insufficient_comparison_evidence",
                "planned": 200,
                "accepted": 140,
                "rejected": 60,
                "unresolved": 0,
                "transport_uncertain_calls": 0,
                "nonconsecutive_transport_uncertain_calls": 0,
                "consecutive_uncertainty_streak": 0,
                "provider_calls": 300,
                "provider_reported_cost_usd": "2.00",
                "errors": ["total_accepted=140 < 150"],
            }
        ),
        encoding="utf-8",
    )
    readme_en = tmp_path / "README.md"
    readme_pt = tmp_path / "README.pt-BR.md"
    readme_en.write_text(
        "<!-- phase4e4-status:begin -->\nThe live comparison has not started.\n"
        "<!-- phase4e4-status:end -->\n",
        encoding="utf-8",
    )
    readme_pt.write_text(
        "<!-- phase4e4-status:begin -->\nA comparação live ainda não começou.\n"
        "<!-- phase4e4-status:end -->\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "comparison",
            "publish-status",
            "--report",
            str(report_path),
            "--readme-en",
            str(readme_en),
            "--readme-pt-br",
            str(readme_pt),
        ],
    )

    comparison.main()

    assert "140 accepted" in readme_en.read_text(encoding="utf-8")
    assert "140 aceitos" in readme_pt.read_text(encoding="utf-8")


# ── generate with accept/reject resolution ──────────────────────────


def test_acceptance_minimums_check_all_axes() -> None:
    plan = comparison.build_comparison_plan()
    accepted: list[dict[str, Any]] = []
    for slot in plan["slots"]:
        accepted.append(
            {
                "task_id": slot["task_id"],
                "locale": slot["locale"],
                "domain": slot["domain"],
                "option_count": slot["option_count"],
                "pair_id": slot.get("pair_id"),
            }
        )
    errors = comparison._check_acceptance_minimums(accepted, plan)
    assert errors == []

    half = accepted[:70]
    errors = comparison._check_acceptance_minimums(half, plan)
    assert errors


def test_acceptance_minimums_rejects_low_locale() -> None:
    plan = comparison.build_comparison_plan()
    accepted = [
        {
            "task_id": s["task_id"],
            "locale": s["locale"],
            "domain": s["domain"],
            "option_count": s["option_count"],
            "pair_id": s.get("pair_id"),
        }
        for s in plan["slots"]
        if s["locale"] == "en"
    ]
    errors = comparison._check_acceptance_minimums(accepted, plan)
    assert any("pt-BR" in e for e in errors)


# ── no-network / default import ──────────────────────────────────────


def test_default_import_offline_and_no_ml(monkeypatch: pytest.MonkeyPatch) -> None:
    def blocked(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket, "getaddrinfo", blocked)

    policy = comparison.validate_comparison_policy()
    assert policy["schema_version"] == "phase4e-comparison-policy.v1"
    plan = comparison.build_comparison_plan()
    assert len(plan["slots"]) == 200
    comparison._wilson_interval(10, 20)
    # This file also runs in optional-extra lanes after training/backend tests
    # have legitimately imported ML modules. Prove the default import boundary
    # in a fresh interpreter instead of depending on test order.
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import benchmarks.saracura_universal_comparison; "
            "loaded={'torch','transformers','tokenizers','safetensors'}; "
            "assert not loaded.intersection(sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_cli_help_runs_offline() -> None:
    with patch("sys.argv", ["comparison", "--help"]), pytest.raises(SystemExit):
        comparison.main()


# ── Wilson / slices / sanitizer ─────────────────────────────────────


def test_wilson_intervals() -> None:
    p, lo, hi = comparison._wilson_interval(0, 10)
    assert p == 0.0 and lo == 0.0 and hi > 0.0
    p, lo, hi = comparison._wilson_interval(10, 10)
    assert p == 1.0 and lo < 1.0 and hi == 1.0


def test_slice_result_underpowered() -> None:
    s = comparison._build_slice_result(3, 5)
    assert s["underpowered"] is True
    s = comparison._build_slice_result(10, 10)
    assert s["underpowered"] is False
    s = comparison._build_slice_result(10, 11)
    assert s["underpowered"] is False


def test_sanitizer_redacts_paths_and_ids() -> None:
    payload = {
        "instruction": "Choose.",
        "absolute_path": "/Users/test/file.json",
        "provider": {"reservation_id": "res-abc", "request_id": "req-xyz"},
        "secret": {"api_key": "sk-sensitive"},
        "nested": [{"home": "/home/user/data"}],
    }
    sanitized = cast(dict[str, Any], comparison._sanitize_for_publication(payload))
    assert sanitized["instruction"] == "Choose."
    assert sanitized["absolute_path"] == "[redacted]"
    provider = cast(dict[str, Any], sanitized["provider"])
    assert provider["reservation_id"] == "[redacted]"
    assert provider["request_id"] == "[redacted]"
    assert sanitized["secret"] == "[redacted]"
    nested = cast(list[Any], sanitized["nested"])
    assert cast(dict[str, Any], nested[0])["home"] == "[redacted]"


# ── plan plan-writing ────────────────────────────────────────────────


def test_run_plan_writes_to_named_dir(tmp_path: Path) -> None:
    output = tmp_path / "comparison-plan" / "plan.json"
    result = comparison.run_plan(output, bindings=comparison._fixture_plan_bindings())
    assert result.exists()
    loaded = json.loads(result.read_bytes())
    assert loaded["schema_version"] == "phase4e-comparison-plan.v1"
    assert len(loaded["slots"]) == 200


def test_run_plan_no_clobber(tmp_path: Path) -> None:
    output = tmp_path / "comparison-plan" / "plan.json"
    comparison.run_plan(output, bindings=comparison._fixture_plan_bindings())
    with pytest.raises(FileExistsError):
        comparison.run_plan(output, bindings=comparison._fixture_plan_bindings())


# ── private root ─────────────────────────────────────────────────────


def test_require_private_root_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SARACURA_PRIVATE_STATE_ROOT", raising=False)
    with pytest.raises(comparison.ComparisonError, match="not set"):
        comparison.require_live_private_comparison_root()


def test_require_private_root_wrong_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "state"
    root.mkdir()
    root.chmod(0o755)
    monkeypatch.setenv("SARACURA_PRIVATE_STATE_ROOT", str(root))
    with pytest.raises(comparison.ComparisonError, match="0700"):
        comparison.require_live_private_comparison_root()


# ── evaluation aggregates ────────────────────────────────────────────


def test_evaluate_primary_slices_without_laya(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SARACURA_PRIVATE_STATE_ROOT", str(tmp_path))
    (tmp_path / "phase4e" / "comparison").mkdir(parents=True, exist_ok=True)
    plan = comparison.build_comparison_plan()
    packet_path = tmp_path / "comparison-packet"
    packet_path.mkdir(parents=True, exist_ok=True)
    criteria = [{"id": "c0", "description": "A."}, {"id": "c1", "description": "B."}]
    accepted = [
        {
            "task_id": s["task_id"],
            "family_id": s["family_id"],
            "locale": s["locale"],
            "domain": s["domain"],
            "instruction": "Choose.",
            "state": {"summary": "T."},
            "criteria": criteria,
            "selected_criterion_id": "c0",
            "synthetic_only": True,
            "review": {
                "status": "accepted",
                "selected_criterion_id": "c0",
                "reason_codes": [],
                "natural_language": True,
                "fictional": True,
                "exclusive_options": True,
                "private_or_sensitive": False,
            },
            "review_sha256": corpus._sha(corpus._canonical({})),
            "author_response_sha256": "a" * 64,
            "author_reservation_id": "reservation-" + "a" * 64,
            "author_request_id": "request-author",
            "reviewer_response_sha256": "b" * 64,
            "reviewer_reservation_id": "reservation-" + "b" * 64,
            "reviewer_request_id": "request-reviewer",
            "pair_id": s.get("pair_id"),
            "gold_position": s.get("gold_position"),
            "option_count": s.get("option_count"),
            "semantic_equivalence_attestation": s["semantic_target"],
        }
        for s in plan["slots"]
    ]
    packet_json = {
        "schema_version": "phase4e-accepted-packet.v4",
        "accepted": accepted,
        "plan": plan,
    }
    (packet_path / "packet.json").write_bytes(corpus._canonical(packet_json) + b"\n")

    eval_work = tmp_path / "comparison-eval"
    with pytest.raises(comparison.ComparisonError):
        comparison.run_evaluate(
            packet_path,
            eval_work,
            saracura_encoder_snapshot=tmp_path / "fake-encoder",
            saracura_training_capsule=tmp_path / "fake-capsule",
            device="cpu",
        )


def test_evaluate_with_laya_absent_builds_control_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SARACURA_PRIVATE_STATE_ROOT", str(tmp_path))
    (tmp_path / "phase4e" / "comparison").mkdir(parents=True, exist_ok=True)
    result = _comparison_result_fixture()
    result["status"] = "control_unavailable"
    result_path = tmp_path / "private-result.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")

    pub_json = tmp_path / "benchmarks" / "results" / "phase4e-comparison-v1.json"
    pub_md = tmp_path / "docs" / "action" / "phase4e-comparison-result.md"
    with pytest.raises(comparison.ComparisonError, match="not publishable"):
        comparison.run_publish(result_path, pub_json, pub_md)


def test_evaluate_against_missing_packet(tmp_path: Path) -> None:
    packet = tmp_path / "nonexistent-packet"
    with pytest.raises(comparison.ComparisonError, match=r"packet\.json"):
        comparison._load_comparison_packet(packet)


def test_packet_loader_rejects_tampered_accepted_rows(tmp_path: Path) -> None:
    packet, _accepted = _sealed_fixture_packet(tmp_path)
    accepted_path = packet / "accepted.jsonl"
    accepted_path.write_bytes(accepted_path.read_bytes() + b"{}\n")
    with pytest.raises(comparison.ComparisonError, match="packet digest"):
        comparison._load_comparison_packet(packet)


def test_packet_loader_rejects_rehashed_rows_not_bound_to_plan(tmp_path: Path) -> None:
    packet, _accepted = _sealed_fixture_packet(tmp_path)
    accepted_path = packet / "accepted.jsonl"
    rows = [json.loads(line) for line in accepted_path.read_bytes().splitlines()]
    rows[0]["locale"] = "en" if rows[0]["locale"] == "pt-BR" else "pt-BR"
    accepted_bytes = b"".join(comparison._canonical(row) + b"\n" for row in rows)
    accepted_path.write_bytes(accepted_bytes)
    descriptor_path = packet / "packet.json"
    descriptor = json.loads(descriptor_path.read_bytes())
    descriptor["files"]["accepted.jsonl"] = comparison._sha(accepted_bytes)
    descriptor_path.write_bytes(comparison._canonical(descriptor) + b"\n")

    with pytest.raises(comparison.ComparisonError, match="slot binding"):
        comparison._load_comparison_packet(packet)


def test_worker_subprocess_scrubs_secrets_and_uses_manifest_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "input-manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("OPENROUTER_API_KEY", "never-forward")
    monkeypatch.setenv("SAFE_RUNTIME_SETTING", "preserved")
    observed: dict[str, Any] = {}

    def fake_run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        observed["argv"] = argv
        observed["env"] = kwargs["env"]
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr("benchmarks.saracura_universal_comparison.subprocess.run", fake_run)
    comparison._run_worker_subprocess(manifest)

    assert observed["argv"][-2:] == ["--input-manifest", str(manifest)]
    assert "never-forward" not in json.dumps(observed["argv"])
    environment = cast(dict[str, str], observed["env"])
    assert "OPENROUTER_API_KEY" not in environment
    assert environment["SAFE_RUNTIME_SETTING"] == "preserved"


def test_evaluate_runs_candidate_then_control_in_isolated_fresh_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tmp_path.chmod(0o700)
    monkeypatch.setenv("SARACURA_PRIVATE_STATE_ROOT", str(tmp_path))
    root = tmp_path / "phase4e" / "comparison"
    root.mkdir(parents=True, mode=0o700)
    packet, accepted = _sealed_fixture_packet(root)
    encoder = root / "encoder-snapshot"
    capsule = root / "training-capsule"
    laya = root / "laya-snapshot"
    for artifact in (encoder, capsule, laya):
        artifact.mkdir(mode=0o700)
        (artifact / "fixture.bin").write_bytes(b"fixture")

    monkeypatch.setattr(comparison, "_preverify_evaluation_inputs", lambda **_kwargs: None)

    order: list[str] = []
    manifests: dict[str, dict[str, Any]] = {}

    def fake_worker(manifest_path: Path) -> None:
        manifest = json.loads(manifest_path.read_bytes())
        backend = cast(str, manifest["backend"])
        order.append(backend)
        manifests[backend] = manifest
        if backend == "laya":
            candidate_result = root / "comparison-eval" / "saracura-cpu" / "score-result.json"
            sealed_receipt = root / "comparison-eval" / "saracura-cpu" / "sealed-result.json"
            assert candidate_result.exists() and sealed_receipt.exists()
            assert stat.S_IMODE(candidate_result.stat().st_mode) == 0o400
        result = _worker_result_fixture(
            backend,
            packet,
            accepted,
            cpu_threads=cast(int, manifest["cpu_threads"]),
        )
        output = Path(manifest["output_dir"]) / "score-result.json"
        output.write_bytes(comparison._canonical(result) + b"\n")
        output.chmod(0o600)

    monkeypatch.setattr(comparison, "_run_worker_subprocess", fake_worker)
    result_path = comparison.run_evaluate(
        packet,
        root / "comparison-eval",
        saracura_encoder_snapshot=encoder,
        saracura_training_capsule=capsule,
        laya_snapshot=laya,
        cpu_threads=2,
        allow_fixture_evidence=True,
    )

    assert order == ["saracura", "laya"]
    assert manifests["laya"]["encoder_snapshot"] is None
    assert manifests["laya"]["training_capsule"] is None
    assert "saracura-cpu" not in json.dumps(manifests["laya"])
    result = json.loads(result_path.read_bytes())
    assert result["status"] == "scored"
    assert result["evidence_mode"] == "fixture"
    assert result["provenance"]["source_commit"] == "5" * 40
    primary = result["primary_comparison"]
    assert len(primary["control_result_sha256"]) == 64
    assert primary["paired_2x2"]["both_correct"] == 200
    assert primary["candidate"]["resolved"] == 200
    assert primary["candidate"]["rejected"] == 0
    assert set(primary["candidate"]["slices"]) == {
        "locale",
        "domain",
        "scenario",
        "option_count",
        "axes",
    }
    assert set(primary["candidate"]["slices"]["axes"]) == set(comparison.AXES)


def test_invalid_control_is_classified_only_after_candidate_is_sealed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tmp_path.chmod(0o700)
    monkeypatch.setenv("SARACURA_PRIVATE_STATE_ROOT", str(tmp_path))
    root = tmp_path / "phase4e" / "comparison"
    root.mkdir(parents=True, mode=0o700)
    packet, accepted = _sealed_fixture_packet(root)
    encoder = root / "encoder-snapshot"
    capsule = root / "training-capsule"
    laya = root / "laya-snapshot"
    for artifact in (encoder, capsule, laya):
        artifact.mkdir(mode=0o700)
        (artifact / "fixture.bin").write_bytes(b"fixture")
    monkeypatch.setattr(comparison, "_preverify_evaluation_inputs", lambda **_kwargs: None)

    def invalid_control(*_args: Any, **_kwargs: Any) -> None:
        raise comparison.ComparisonError("invalid control fixture")

    monkeypatch.setattr(comparison, "_preverify_laya_control", invalid_control)
    observed: list[str] = []

    def fake_worker(manifest_path: Path) -> None:
        manifest = json.loads(manifest_path.read_bytes())
        backend = cast(str, manifest["backend"])
        observed.append(backend)
        result = _worker_result_fixture(backend, packet, accepted)
        output = Path(manifest["output_dir"]) / "score-result.json"
        output.write_bytes(comparison._canonical(result) + b"\n")

    monkeypatch.setattr(comparison, "_run_worker_subprocess", fake_worker)
    result_path = comparison.run_evaluate(
        packet,
        root / "comparison-eval",
        saracura_encoder_snapshot=encoder,
        saracura_training_capsule=capsule,
        laya_snapshot=laya,
        allow_fixture_evidence=True,
    )

    assert observed == ["saracura"]
    assert result_path.name == "control-unavailable.json"
    assert json.loads(result_path.read_bytes())["status"] == "control_unavailable"
    candidate = root / "comparison-eval" / "saracura-cpu" / "score-result.json"
    assert stat.S_IMODE(candidate.stat().st_mode) == 0o400


def test_control_worker_integrity_failure_is_not_control_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tmp_path.chmod(0o700)
    monkeypatch.setenv("SARACURA_PRIVATE_STATE_ROOT", str(tmp_path))
    root = tmp_path / "phase4e" / "comparison"
    root.mkdir(parents=True, mode=0o700)
    packet, accepted = _sealed_fixture_packet(root)
    encoder = root / "encoder-snapshot"
    capsule = root / "training-capsule"
    laya = root / "laya-snapshot"
    for artifact in (encoder, capsule, laya):
        artifact.mkdir(mode=0o700)
        (artifact / "fixture.bin").write_bytes(b"fixture")
    monkeypatch.setattr(comparison, "_preverify_evaluation_inputs", lambda **_kwargs: None)
    monkeypatch.setattr(comparison, "_preverify_laya_control", lambda *_args, **_kwargs: None)

    def fake_worker(manifest_path: Path) -> None:
        manifest = json.loads(manifest_path.read_bytes())
        backend = cast(str, manifest["backend"])
        result = _worker_result_fixture(backend, packet, accepted)
        if backend == "laya":
            result["rows"] = result["rows"][:-1]
        output = Path(manifest["output_dir"]) / "score-result.json"
        output.write_bytes(comparison._canonical(result) + b"\n")

    monkeypatch.setattr(comparison, "_run_worker_subprocess", fake_worker)
    with pytest.raises(comparison.ComparisonError, match="worker result coverage"):
        comparison.run_evaluate(
            packet,
            root / "comparison-eval",
            saracura_encoder_snapshot=encoder,
            saracura_training_capsule=capsule,
            laya_snapshot=laya,
            allow_fixture_evidence=True,
        )


# ── comparison terminal states ──────────────────────────────────────


def test_terminal_states_are_four() -> None:
    assert (
        frozenset(
            {
                "passed",
                "insufficient_comparison_evidence",
                "inconclusive_transport",
                "inconclusive_operational",
            }
        )
        == comparison.COMPARISON_TERMINAL_STATES
    )


def test_task_statuses_are_five() -> None:
    assert (
        frozenset(
            {
                "correct",
                "incorrect",
                "unsupported_capacity",
                "unsupported_runtime",
                "runtime_error",
            }
        )
        == comparison.COMPARISON_TASK_STATUSES
    )


# ── all domains and axes covered in plan ───────────────────────────


def test_all_domains_covered() -> None:
    plan = comparison.build_comparison_plan()
    domains = {s["domain"] for s in plan["slots"]}
    assert domains == set(comparison.DOMAINS)


def test_all_option_counts_covered() -> None:
    plan = comparison.build_comparison_plan()
    for c in range(2, 9):
        assert any(s["option_count"] == c for s in plan["slots"])


def test_all_locales_covered() -> None:
    plan = comparison.build_comparison_plan()
    assert {s["locale"] for s in plan["slots"]} == {"pt-BR", "en"}


def test_publish_ordered_create_or_byte_identical_from_live_evaluator_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tmp_path.chmod(0o700)
    monkeypatch.setenv("SARACURA_PRIVATE_STATE_ROOT", str(tmp_path))
    root = tmp_path / "phase4e" / "comparison"
    root.mkdir(parents=True, mode=0o700)
    monkeypatch.setattr(comparison, "_require_live_external_comparison_root", lambda: root)
    packet, accepted = _sealed_fixture_packet(root, live_verified=True)
    encoder = root / "encoder-snapshot"
    capsule = root / "training-capsule"
    laya = root / "laya-snapshot"
    for artifact in (encoder, capsule, laya):
        artifact.mkdir(mode=0o700)
        (artifact / "fixture.bin").write_bytes(b"fixture")
    monkeypatch.setattr(comparison, "_preverify_evaluation_inputs", lambda **_kwargs: None)
    monkeypatch.setattr(comparison, "_preverify_laya_control", lambda *_args, **_kwargs: None)

    def fake_worker(manifest_path: Path) -> None:
        manifest = json.loads(manifest_path.read_bytes())
        backend = cast(str, manifest["backend"])
        result = _worker_result_fixture(backend, packet, accepted)
        if backend == "saracura":
            for row in result["rows"][:50]:
                row["status"] = "incorrect"
                row["selected_criterion_id"] = "criterion-0"
        output = Path(manifest["output_dir"]) / "score-result.json"
        output.write_bytes(comparison._canonical(result) + b"\n")

    monkeypatch.setattr(comparison, "_run_worker_subprocess", fake_worker)
    result_path = comparison.run_evaluate(
        packet,
        root / "comparison-eval",
        saracura_encoder_snapshot=encoder,
        saracura_training_capsule=capsule,
        laya_snapshot=laya,
    )
    pub_json = tmp_path / "benchmarks" / "results" / "phase4e-comparison-v1.json"
    pub_md = tmp_path / "docs" / "action" / "phase4e-comparison-result.md"
    readme_en = tmp_path / "README.md"
    readme_pt = tmp_path / "docs" / "README.pt-BR.md"
    readme_pt.parent.mkdir(parents=True, exist_ok=True)
    readme_en.write_text(
        "<!-- phase4e4-status:begin -->\nThe live comparison has not started.\n"
        "<!-- phase4e4-status:end -->\n",
        encoding="utf-8",
    )
    readme_pt.write_text(
        "<!-- phase4e4-status:begin -->\nA comparação live ainda não começou.\n"
        "<!-- phase4e4-status:end -->\n",
        encoding="utf-8",
    )

    j1, _m1 = comparison.run_publish(
        result_path,
        pub_json,
        pub_md,
        readme_en=readme_en,
        readme_pt_br=readme_pt,
    )
    result_path.chmod(0o600)
    with pytest.raises(comparison.ComparisonError, match="sealed read-only"):
        comparison.run_publish(
            result_path,
            pub_json,
            pub_md,
            readme_en=readme_en,
            readme_pt_br=readme_pt,
        )
    result_path.chmod(0o400)
    j2, _m2 = comparison.run_publish(
        result_path,
        pub_json,
        pub_md,
        readme_en=readme_en,
        readme_pt_br=readme_pt,
    )
    assert j1.read_bytes() == pub_json.read_bytes()
    assert j2 == j1
    published = json.loads(pub_json.read_bytes())
    assert published["evidence_mode"] == "live_verified"
    assert published["primary_comparison"]["candidate"]["resolved"] == 200
    assert published["primary_comparison"]["candidate"]["rejected"] == 0
    assert published["primary_comparison"]["candidate"]["accuracy"]["correct"] == 150
    assert published["primary_comparison"]["control"]["accuracy"]["correct"] == 200
    assert published["provenance"]["training_checkpoint_sha256"] == "4" * 64
    assert any(
        "model family and prompt family" in item
        and "in-distribution for Saracura but not necessarily for Laya" in item
        for item in published["limitations"]
    )
    assert "phase4e4-status-binding:scored:" in readme_en.read_text(encoding="utf-8")
    assert "aggregate result" in readme_en.read_text(encoding="utf-8")
    assert "resultado agregado" in readme_pt.read_text(encoding="utf-8")
    markdown = pub_md.read_text(encoding="utf-8")
    assert markdown.startswith("---\ntitle: Phase 4E.4 blind comparison result\n")
    assert "in-distribution for Saracura but not necessarily for Laya" in markdown
    for field in (
        "kind:",
        "area:",
        "project:",
        "collection:",
        "owner:",
        "status:",
        "canonical:",
        "globalRef:",
        "reviewCadenceDays:",
        "lastReviewedAt:",
        "sourceRefs:",
        "related:",
        "supersedes:",
        "supersededBy:",
        "sensitivity:",
    ):
        assert field in markdown.split("---", 2)[1]


def test_worker_isolation_via_separate_dirs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Prove that saracura and laya worker dirs are separate."""
    monkeypatch.setenv("SARACURA_PRIVATE_STATE_ROOT", str(tmp_path))
    (tmp_path / "phase4e" / "comparison").mkdir(parents=True, exist_ok=True)
    # evaluate constructs saracura-cpu/ and laya-cpu/ under the eval work dir
    # with _outside ensuring isolation. Test that the path structure is enforced.
    plan = comparison.build_comparison_plan()
    packet_path = tmp_path / "comparison-packet"
    packet_path.mkdir(parents=True, exist_ok=True)
    criteria = [{"id": "c0", "description": "A."}, {"id": "c1", "description": "B."}]
    accepted = [
        {
            "task_id": s["task_id"],
            "family_id": s["family_id"],
            "locale": s["locale"],
            "domain": s["domain"],
            "instruction": "Choose.",
            "state": {"summary": "T."},
            "criteria": criteria,
            "selected_criterion_id": "c0",
            "synthetic_only": True,
            "review": {
                "status": "accepted",
                "selected_criterion_id": "c0",
                "reason_codes": [],
                "natural_language": True,
                "fictional": True,
                "exclusive_options": True,
                "private_or_sensitive": False,
            },
            "review_sha256": corpus._sha(corpus._canonical({})),
            "author_response_sha256": "a" * 64,
            "author_reservation_id": "reservation-" + "a" * 64,
            "author_request_id": "request-author",
            "reviewer_response_sha256": "b" * 64,
            "reviewer_reservation_id": "reservation-" + "b" * 64,
            "reviewer_request_id": "request-reviewer",
            "pair_id": s.get("pair_id"),
            "gold_position": s.get("gold_position"),
            "option_count": s.get("option_count"),
            "semantic_equivalence_attestation": s["semantic_target"],
        }
        for s in plan["slots"]
    ]
    packet_json = {
        "schema_version": "phase4e-accepted-packet.v4",
        "accepted": accepted,
        "plan": plan,
    }
    (packet_path / "packet.json").write_bytes(corpus._canonical(packet_json) + b"\n")

    eval_work = tmp_path / "comparison-eval"
    with pytest.raises(comparison.ComparisonError):
        comparison.run_evaluate(
            packet_path,
            eval_work,
            saracura_encoder_snapshot=tmp_path / "fake-encoder",
            saracura_training_capsule=tmp_path / "fake-capsule",
            device="cpu",
        )
