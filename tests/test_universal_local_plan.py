from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from benchmarks.universal_local import plan


def test_plan_v2_preserves_exact_matrix_and_materializes_scenario_four() -> None:
    value = plan.validate_plan()
    scenario = value["scenarios"][3]
    assert (scenario["id"], scenario["N"], scenario["Q"], scenario["K"]) == (
        "scenario-04",
        8,
        10,
        20,
    )
    states, questions, choices = plan.materialize(value, scenario)
    assert len(states) == 8
    assert all(len(row) == 10 for row in questions)
    assert all(len(row) == 10 and all(len(options) == 20 for options in row) for row in choices)


def test_input_seed_changes_rule_generated_materialization() -> None:
    value = plan.validate_plan()
    scenario = value["scenarios"][1]
    altered = cast(plan.Scenario, dict(scenario))
    altered["input_seed"] = "phase4c2-support-ptbr-other"
    first = plan.materialize(value, scenario)
    second = plan.materialize(value, altered)
    assert first != second


def test_plan_rejects_matrix_and_report_policy_mutations() -> None:
    payload = json.loads(plan.PLAN_PATH.read_text(encoding="utf-8"))
    payload["scenarios"][3]["K"] = 19
    with pytest.raises(ValueError, match="scenario matrix"):
        plan.load_plan(json.dumps(payload).encode())
    payload = json.loads(plan.PLAN_PATH.read_text(encoding="utf-8"))
    payload["report_policy"]["allowed_metrics"].append("raw_scores")
    with pytest.raises(ValueError, match="report policy"):
        plan.load_plan(json.dumps(payload).encode())


def test_plan_rejects_mutated_predecessor_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mutated = tmp_path / "v1.json"
    mutated.write_bytes(plan.V1_PLAN_PATH.read_bytes() + b"\n")
    monkeypatch.setattr(plan, "V1_PLAN_PATH", mutated)
    with pytest.raises(ValueError, match="predecessor plan bytes"):
        plan.load_plan()


def test_materializer_rejects_reserved_frame_in_source_bank() -> None:
    value = plan.validate_plan()
    mutated = dict(value)
    contract = dict(value["generator_contract"])
    banks = json.loads(json.dumps(contract["sentence_banks"]))
    banks["support"]["pt-BR"]["instruction"] = ["Texto com S12: reservado"]
    contract["sentence_banks"] = banks
    mutated["generator_contract"] = contract
    with pytest.raises(ValueError, match="reserved framing"):
        plan.materialize(cast(plan.Plan, mutated), value["scenarios"][0])
