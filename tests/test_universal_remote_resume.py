"""Offline regression coverage for the bounded recovery response contract."""

from __future__ import annotations

import json
from collections.abc import Mapping
from decimal import Decimal
from pathlib import Path

import pytest

from benchmarks.universal_remote import runner as predecessor_runner
from benchmarks.universal_remote.plan import MODEL, materialize_requests
from benchmarks.universal_remote_resume import plan, runner


def _response(*, total: Decimal = Decimal("1")) -> dict[str, object]:
    request = materialize_requests()[10]["request"]
    answers: dict[str, object] = {}
    for key, question in request["questions"].items():
        criteria = list(question["criteria"])
        probability = total / len(criteria)
        answers[key] = {
            "type": "choice",
            "choice": criteria[0],
            "probabilities": {criterion: probability for criterion in criteria},
            "confidence": Decimal("0.5"),
        }
    return {
        "model": MODEL,
        "answers": answers,
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }


def _binding() -> dict[str, str]:
    return {
        key: "a" * 64
        for key in (
            "recovery_plan",
            "phase4c3_partial",
            "plan_v3",
            "prior_ledger",
            "training",
            "packet",
            "phase3c",
            "phase3d",
            "laya",
            "mdeberta",
        )
    }


def _reduced(ordinal: int, *, input_tokens: int = 1) -> dict[str, object]:
    return runner.sample(
        materialize_requests()[ordinal],
        {
            "input_tokens": input_tokens,
            "output_tokens": 1,
            "probability_sum_tolerance_used": False,
        },
        1.0,
    )


def _local_comparison() -> list[dict[str, object]]:
    return [
        {
            "candidate_id": "laya-multilingual",
            "completed_decision_count": 182,
            "latency_p50_ms": 1.0,
            "decisions_per_second": 1.0,
        },
        {
            "candidate_id": "mdeberta-nli",
            "completed_decision_count": 183,
            "latency_p50_ms": 1.0,
            "decisions_per_second": 1.0,
        },
    ]


def _wire_response(request: dict[str, object], *, valid_sum: bool = True) -> bytes:
    questions = request["questions"]
    assert isinstance(questions, dict)
    answers: dict[str, object] = {}
    for key, question in questions.items():
        assert isinstance(key, str) and isinstance(question, dict)
        criteria = question["criteria"]
        assert isinstance(criteria, dict)
        choices = list(criteria)
        answers[key] = {
            "type": "choice",
            "choice": choices[0],
            "probabilities": {
                choice: (1.0 if valid_sum and index == 0 else 0.0)
                for index, choice in enumerate(choices)
            },
            "confidence": 1.0,
        }
    return json.dumps(
        {
            "model": MODEL,
            "answers": answers,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
        separators=(",", ":"),
    ).encode()


def test_recovery_plan_binds_every_materialized_wire() -> None:
    value = plan.validate_plan()
    assert len(value["wire_sha256_by_ordinal"]) == 57


def test_recovery_plan_rejects_source_or_commit_substitution() -> None:
    value = json.loads(plan.PLAN_PATH.read_bytes())
    value["sources"].pop("benchmarks/universal_local/inputs.py")
    with pytest.raises(ValueError, match="recovery plan binding"):
        plan.validate_plan(json.dumps(value).encode())
    value = json.loads(plan.PLAN_PATH.read_bytes())
    value["source_commit"] = "0" * 40
    with pytest.raises(ValueError, match="recovery plan binding"):
        plan.validate_plan(json.dumps(value).encode())


def test_exact_decimal_tolerance_boundaries() -> None:
    request = materialize_requests()[10]["request"]
    accepted = _response(total=Decimal("0.99"))
    assert plan.validate_response(accepted, request)["probability_sum_tolerance_used"] is True
    rejected = _response(total=Decimal("0.9899"))
    with pytest.raises(plan.ResponseError, match="probability_sum_out_of_tolerance"):
        plan.validate_response(rejected, request)


def test_json_strictly_rejects_duplicate_and_nonfinite_values() -> None:
    with pytest.raises(plan.ResponseError, match="json_invalid"):
        plan.load_json(b'{"model":"x","model":"y"}')
    with pytest.raises(plan.ResponseError, match="json_invalid"):
        plan.load_json(json.dumps({"n": float("nan")}).encode())


def test_decimal_bombs_and_selected_nonmaximal_are_bounded() -> None:
    request = materialize_requests()[10]["request"]
    response = _response()
    answers = response["answers"]
    assert isinstance(answers, dict)
    answer = next(iter(answers.values()))
    assert isinstance(answer, dict)
    answer["confidence"] = Decimal("1e-1001")
    with pytest.raises(plan.ResponseError, match="confidence_invalid"):
        plan.validate_response(response, request)
    response = _response()
    answers = response["answers"]
    assert isinstance(answers, dict)
    answer = next(iter(answers.values()))
    assert isinstance(answer, dict)
    probabilities = answer["probabilities"]
    assert isinstance(probabilities, dict)
    first, second = list(probabilities)[:2]
    probabilities[first], probabilities[second] = Decimal("0.04"), Decimal("0.06")
    with pytest.raises(plan.ResponseError, match="choice_not_maximal"):
        plan.validate_response(response, request)


def test_non_200_large_body_preserves_http_classification() -> None:
    meta = materialize_requests()[10]

    def transport(
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes,
        timeout: float,
    ) -> tuple[int, Mapping[str, str], bytes]:
        assert method == "POST" and headers and body and timeout == 60.0 and url
        return 500, {}, b"x" * (plan.BODY_CEILING + 1)

    with pytest.raises(RuntimeError, match="http_error"):
        runner.dispatch(meta, "test-only", transport)


def test_settlement_sample_rejects_provider_text_in_latency() -> None:
    meta = materialize_requests()[10]
    reduced = runner.sample(
        meta,
        {
            "input_tokens": 1,
            "output_tokens": 1,
            "probability_sum_tolerance_used": False,
        },
        1.0,
    )
    reduced["latency_ms"] = "provider-sentinel-that-must-not-be-published"
    with pytest.raises(ValueError, match="sample latency"):
        runner._validate_sample(reduced, 10)


def test_partial_report_states_provider_receipt_uncertainty() -> None:
    report = runner._report(
        {
            "status": "partial",
            "failure_category": "timeout",
            "completed_request_count": 10,
        }
    ).decode()
    assert "whether the provider received or executed it is unknown" in report
    confirmed = runner._report(
        {
            "status": "partial",
            "failure_category": "response_invalid",
            "completed_request_count": 10,
        }
    ).decode()
    assert "second provider execution" in confirmed


def test_settled_over_limit_counts_as_completed_continuation(tmp_path: Path) -> None:
    journal_path = tmp_path / "cost-journal.jsonl"
    journal_path.touch(mode=0o600)
    journal = runner.Journal(journal_path, "a" * 64)
    reduced = _reduced(10, input_tokens=64_001)
    journal.append(
        "reserve",
        10,
        reservation_cost_usd="0.0032",
        wire_sha256=plan.validate_plan()["wire_sha256_by_ordinal"]["10"],
    )
    journal.append(
        "settled_over_limit",
        10,
        sample=reduced,
        computed_cost_usd=reduced["computed_cost_usd"],
    )
    prior = [
        _reduced(ordinal)
        | {
            "measurement_source": "phase4c3_predecessor",
            "response_validation_contract": "strict_sum_1e-6",
        }
        for ordinal in range(10)
    ]
    result = runner.reconstruct(
        journal,
        {},
        Decimal("0.0032"),
        Decimal("0.01"),
        prior,
        _binding(),
        [],
    )
    assert result["failure_category"] == "usage_overflow"
    assert result["completed_request_count"] == 11
    assert result["completed_decision_count"] == 29


def test_claim_store_rejects_intermediate_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    common = tmp_path / "common"
    outside = tmp_path / "outside"
    common.mkdir()
    outside.mkdir()
    (common / "saracura-local-claims").symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(predecessor_runner, "git_common_dir", lambda _: common)
    with pytest.raises(ValueError, match="claim directory"):
        runner._claim_path(tmp_path, "a" * 64, create=True)


def test_output_directory_is_fsynced_before_journal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[Path] = []
    monkeypatch.setattr(runner, "_fsync", calls.append)
    output = tmp_path / "output"
    runner.create_output(output)
    assert calls == [tmp_path, output]
    assert (output / "cost-journal.jsonl").is_file()


def test_finalization_cleanup_removes_only_recognized_temporaries(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir(mode=0o700)
    recognized = output / ".tmp-result.json-crash"
    recognized.touch(mode=0o600)
    runner._remove_writer_temporaries(output)
    assert not recognized.exists()
    unrecognized = output / ".tmp-provider-body-crash"
    unrecognized.touch(mode=0o600)
    with pytest.raises(ValueError, match="unrecognized atomic temporary"):
        runner._remove_writer_temporaries(output)


def test_finalization_cleanup_rejects_symlink_before_removing_temporary(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir(mode=0o700)
    temporary = target / ".tmp-result.json-crash"
    temporary.touch(mode=0o600)
    output = tmp_path / "output"
    output.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="output path"):
        runner._remove_writer_temporaries(output)
    assert temporary.is_file()


def test_response_invalid_failure_requires_bounded_detail(tmp_path: Path) -> None:
    journal_path = tmp_path / "cost-journal.jsonl"
    journal_path.touch(mode=0o600)
    journal = runner.Journal(journal_path, "a" * 64)
    journal.append(
        "reserve",
        10,
        reservation_cost_usd="0.0032",
        wire_sha256=plan.validate_plan()["wire_sha256_by_ordinal"]["10"],
    )
    with pytest.raises(ValueError, match="failure transition"):
        journal.append(
            "failure",
            10,
            category="response_invalid",
            response_invalid_detail=None,
        )


@pytest.mark.parametrize("next_kind", ("reserve", "halt"))
def test_open_reservation_rejects_next_ordinal_transition(tmp_path: Path, next_kind: str) -> None:
    journal_path = tmp_path / "cost-journal.jsonl"
    journal_path.touch(mode=0o600)
    journal = runner.Journal(journal_path, "a" * 64)
    wires = plan.validate_plan()["wire_sha256_by_ordinal"]
    journal.append("reserve", 10, reservation_cost_usd="0.0032", wire_sha256=wires["10"])
    with pytest.raises(ValueError):
        if next_kind == "reserve":
            journal.append("reserve", 11, reservation_cost_usd="0.0032", wire_sha256=wires["11"])
        else:
            journal.append(
                "halt",
                11,
                category="budget_uncertain",
                response_invalid_detail=None,
            )


def test_halt_without_reservation_has_zero_continuation_charge(tmp_path: Path) -> None:
    journal_path = tmp_path / "cost-journal.jsonl"
    journal_path.touch(mode=0o600)
    journal = runner.Journal(journal_path, "a" * 64)
    journal.append(
        "halt",
        10,
        category="budget_uncertain",
        response_invalid_detail=None,
    )
    assert journal.charged() == Decimal(0)
    prior = [
        _reduced(ordinal)
        | {
            "measurement_source": "phase4c3_predecessor",
            "response_validation_contract": "strict_sum_1e-6",
        }
        for ordinal in range(10)
    ]
    result = runner.reconstruct(
        journal,
        {},
        Decimal("0.003"),
        Decimal("0.246"),
        prior,
        _binding(),
        [],
    )
    assert result["failure_category"] == "budget_uncertain"
    assert result["remote_charge"]["continuation_journal_usd"] == "0"
    assert result["remote_charge"]["cumulative_authorized_use_usd"] == "0.249"


def test_artifact_manifest_binds_every_resume_source() -> None:
    assert set(runner._source_digests()) == {
        "benchmarks/universal_remote_resume/__init__.py",
        "benchmarks/universal_remote_resume/__main__.py",
        "benchmarks/universal_remote_resume/plan.py",
        "benchmarks/universal_remote_resume/runner.py",
    }


def test_partial_declared_charge_must_equal_recomputed_journal_charge() -> None:
    assert runner._validate_partial_charge(
        {"charged_cost_usd": "0.00350702"}, Decimal("0.00350702")
    ) == Decimal("0.00350702")
    with pytest.raises(ValueError, match="partial charge"):
        runner._validate_partial_charge({"charged_cost_usd": "0.00350703"}, Decimal("0.00350702"))
    with pytest.raises(ValueError, match="partial charge"):
        runner._validate_partial_charge({"charged_cost_usd": "0.003507020"}, Decimal("0.00350702"))


def test_complete_fake_continuation_is_sealed_and_validated_read_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    common = tmp_path / "common"
    common.mkdir()
    monkeypatch.setattr(predecessor_runner, "git_common_dir", lambda _: common)
    predecessor_binding = {
        key: "a" * 64
        for key in (
            "plan_v3",
            "prior_ledger",
            "training",
            "packet",
            "phase3c",
            "phase3d",
            "laya",
            "mdeberta",
        )
    }
    predecessor_calls = 0

    def predecessor_transport(
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes,
        timeout: float,
    ) -> tuple[int, Mapping[str, str], bytes]:
        nonlocal predecessor_calls
        assert method == "POST" and url and headers and timeout == 60.0
        request = json.loads(body)
        response = _wire_response(request, valid_sum=predecessor_calls != 10)
        predecessor_calls += 1
        return 200, {}, response

    partial_dir = tmp_path / "partial"
    partial = predecessor_runner._run_prevalidated(
        tmp_path,
        partial_dir,
        predecessor_binding,
        _local_comparison(),
        api_key="test-only",
        transport=predecessor_transport,
    )
    assert predecessor_calls == 11
    assert partial["status"] == "partial"
    binding = {
        "recovery_plan": plan.digest(plan.PLAN_PATH),
        "phase4c3_partial": plan.digest(partial_dir / "artifact-manifest.json"),
        **predecessor_binding,
    }
    continuation_calls = 0

    def continuation_transport(
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes,
        timeout: float,
    ) -> tuple[int, Mapping[str, str], bytes]:
        nonlocal continuation_calls
        assert method == "POST" and url and headers and timeout == 60.0
        continuation_calls += 1
        return 200, {}, _wire_response(json.loads(body))

    output = tmp_path / "continuation"
    result = runner.run_prevalidated(
        tmp_path,
        output,
        partial_dir,
        binding,
        _local_comparison(),
        Decimal("0.051104882"),
        api_key="test-only",
        transport=continuation_transport,
    )
    assert continuation_calls == 47
    assert result["status"] == "complete"
    assert result["summary"]["completed_request_count"] == 57
    assert result["summary"]["completed_decision_count"] == 183
    report = (output / "report.md").read_text()
    assert "Composite summary (ordinals 0-56): 57 requests, 183 decisions" in report
    assert "Continuation-only summary (ordinals 10-56): 47 requests, 164 decisions" in report
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    validated = runner.validate_artifact(
        tmp_path,
        output,
        partial_dir,
        binding,
        _local_comparison(),
        Decimal("0.051104882"),
    )
    after = {path.name: path.read_bytes() for path in output.iterdir()}
    assert validated == result and after == before
    assert "probabilities" not in (output / "result.json").read_text()
