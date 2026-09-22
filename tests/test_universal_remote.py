from __future__ import annotations

import json
import math
import urllib.error
import urllib.request
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from benchmarks import inspect_wheel
from benchmarks.universal_remote import plan, runner


def _response(request: dict[str, Any], *, tokens: int = 10) -> bytes:
    answers = {}
    for key, question in request["questions"].items():
        labels = list(question["criteria"])
        answers[key] = {
            "type": "choice",
            "choice": labels[0],
            "probabilities": {label: float(label == labels[0]) for label in labels},
            "confidence": 1.0,
        }
    return json.dumps(
        {
            "model": plan.MODEL,
            "answers": answers,
            "usage": {"input_tokens": tokens, "output_tokens": 0},
        }
    ).encode()


def test_plan_and_partition_are_closed_and_include_seventy_seven_choices() -> None:
    assert plan.validate_plan()["remote_budget"]["requests"] == 57
    requests = plan.materialize_requests()
    assert len(requests) == 57
    last = requests[-1]
    assert last["scenario_id"] == "scenario-08"
    assert list(last["request"]["questions"]["q0"]["criteria"])[-1] == "choice-76"


def test_wire_order_preserves_numeric_choices() -> None:
    item = next(
        entry for entry in plan.materialize_requests() if entry["scenario_id"] == "scenario-04"
    )
    raw = item["wire"]
    assert raw.index(b'"choice-2"') < raw.index(b'"choice-10"')
    assert raw.startswith(b'{"state":')


def test_response_validation_rejects_unknown_probability_and_model_drift() -> None:
    request = plan.materialize_requests()[0]["request"]
    payload = json.loads(_response(request))
    assert plan.validate_response(payload, request)["input_tokens"] == 10
    payload["model"] = "jev-latest"
    with pytest.raises(ValueError, match="model drift"):
        plan.validate_response(payload, request)
    payload = json.loads(_response(request))
    payload["answers"]["q0"]["probabilities"]["extra"] = 0
    with pytest.raises(ValueError, match="probability keys"):
        plan.validate_response(payload, request)
    payload = json.loads(_response(request))
    payload["answers"]["q0"]["probabilities"] = {
        key: 0.0 for key in payload["answers"]["q0"]["probabilities"]
    }
    with pytest.raises(ValueError, match="sum to one"):
        plan.validate_response(payload, request)
    payload = json.loads(_response(request))
    payload["answers"]["q0"]["confidence"] = math.nan
    with pytest.raises(ValueError, match="non-finite"):
        plan.validate_response(payload, request)


def test_journal_charges_open_reservation_and_rejects_out_of_order(tmp_path: Path) -> None:
    path = tmp_path / "cost-journal.jsonl"
    path.touch()
    path.chmod(0o600)
    journal = runner.Journal(path, "a" * 64)
    journal.append("reserve", 0, reservation_cost_usd="0.0032")
    assert journal.charged() == runner.RESERVATION
    with pytest.raises(ValueError, match="settlement transition"):
        journal.append("settle", 1, sample={}, computed_cost_usd="0")


def test_budget_constants_match_the_closed_reservation_envelope() -> None:
    assert Decimal("64000") * Decimal("0.05") / 1_000_000 == runner.RESERVATION
    assert Decimal("0.18240") == runner.RESERVATION * plan.REQUEST_COUNT
    assert Decimal("0.06760") == runner.BUDGET - runner.RESERVATION * plan.REQUEST_COUNT


def test_fake_transport_is_single_attempt_and_never_persists_answers(tmp_path: Path) -> None:
    request = plan.materialize_requests()[0]["request"]
    calls: list[bytes] = []

    def transport(
        method: str, url: str, headers: Mapping[str, str], body: bytes, timeout: float
    ) -> tuple[int, Mapping[str, str], bytes]:
        assert method == "POST" and url == plan.ENDPOINT and timeout == 60.0
        calls.append(body)
        return 200, {}, _response(request)

    value, latency = runner.dispatch(request, "test-only", transport)
    reduced = runner.sample(
        {
            "ordinal": 0,
            "scenario_id": "scenario-01",
            "state_ordinal": 0,
            "question_offset": 0,
            "question_count": 1,
            "request": request,
        },
        value,
        latency,
    )
    assert len(calls) == 1 and "answers" not in reduced and "probabilities" not in reduced


def test_urllib_timeout_is_classified_as_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    class Opener:
        def open(self, request: object, timeout: float) -> object:
            raise urllib.error.URLError(TimeoutError())

    monkeypatch.setattr(urllib.request, "build_opener", lambda *args: Opener())
    with pytest.raises(TimeoutError):
        runner._urllib_transport("POST", plan.ENDPOINT, {}, b"{}", 60.0)


def test_claim_is_single_use(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    common = tmp_path / "common"
    common.mkdir()
    monkeypatch.setattr(runner, "git_common_dir", lambda _: common)
    binding = {
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
    runner.create_claim(tmp_path, "b" * 64, binding)
    with pytest.raises(ValueError, match="claim already exists"):
        runner.create_claim(tmp_path, "b" * 64, binding)


def test_missing_credential_cannot_create_output_or_consume_claim(tmp_path: Path) -> None:
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="TYPESAFE_API_KEY required"):
        runner._run_prevalidated(tmp_path, output, _binding(), _local_comparison(), api_key="")
    assert not output.exists()
    assert not (tmp_path / "saracura-local-claims").exists()


def test_claim_validation_rejects_a_changed_comparison_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    common = tmp_path / "common"
    common.mkdir()
    monkeypatch.setattr(runner, "git_common_dir", lambda _: common)
    binding = _binding()
    runner.create_claim(tmp_path, "b" * 64, binding)
    changed = {**binding, "laya": "b" * 64}
    with pytest.raises(ValueError, match="claim binding"):
        runner.validate_claim(tmp_path, "b" * 64, changed)


def test_claim_is_shared_across_relocated_inputs_and_concurrent_worktrees(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    common = tmp_path / "shared-git-common"
    common.mkdir()
    first_repo = tmp_path / "first-worktree"
    second_repo = tmp_path / "second-worktree"
    first_repo.mkdir()
    second_repo.mkdir()
    monkeypatch.setattr(runner, "git_common_dir", lambda _: common)
    binding = _binding()
    lineage = "b" * 64

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(lambda repo: _claim_result(repo, lineage, binding), (first_repo, second_repo))
        )
    assert results.count("created") == 1
    assert results.count("claim already exists") == 1


def _claim_result(repo: Path, lineage: str, binding: dict[str, str]) -> str:
    try:
        runner.create_claim(repo, lineage, binding)
    except ValueError as error:
        return str(error)
    return "created"


def test_partial_finalization_is_deterministic_and_no_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "output"
    output.mkdir(mode=0o700)
    journal_path = output / "cost-journal.jsonl"
    journal_path.touch()
    journal_path.chmod(0o600)
    lineage = "a" * 64
    journal = runner.Journal(journal_path, lineage)
    journal.append("reserve", 0, reservation_cost_usd="0.0032")
    provenance = {
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
    common = tmp_path / "common"
    common.mkdir()
    monkeypatch.setattr(runner, "git_common_dir", lambda _: common)
    runner.create_claim(tmp_path, lineage, provenance)
    first = runner.finalize(tmp_path, output, lineage, provenance, _local_comparison())
    second = runner.finalize(tmp_path, output, lineage, provenance, _local_comparison())
    assert first == second and first["failure_category"] == "process_interrupted"
    assert (
        runner.validate_artifact(tmp_path, output, lineage, provenance, _local_comparison())
        == first
    )


def test_complete_fake_run_seals_reduced_systems_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    common = tmp_path / "common"
    common.mkdir()
    monkeypatch.setattr(runner, "git_common_dir", lambda _: common)
    provenance = _binding()
    calls = 0

    def transport(
        method: str, url: str, headers: Mapping[str, str], body: bytes, timeout: float
    ) -> tuple[int, Mapping[str, str], bytes]:
        nonlocal calls
        calls += 1
        request = json.loads(body)
        return 200, {}, _response(request)

    result = runner._run_prevalidated(
        tmp_path,
        tmp_path / "output",
        provenance,
        _local_comparison(),
        api_key="test",
        transport=transport,
    )
    assert calls == 57
    assert result["status"] == "complete"
    assert result["summary"]["completed_decision_count"] == 183
    assert "answers" not in (tmp_path / "output" / "result.json").read_text()


def test_over_limit_response_seals_once_without_a_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    common = tmp_path / "common"
    common.mkdir()
    monkeypatch.setattr(runner, "git_common_dir", lambda _: common)
    calls = 0

    def transport(
        method: str, url: str, headers: Mapping[str, str], body: bytes, timeout: float
    ) -> tuple[int, Mapping[str, str], bytes]:
        nonlocal calls
        calls += 1
        return 200, {}, _response(json.loads(body), tokens=64001)

    result = runner._run_prevalidated(
        tmp_path,
        tmp_path / "output",
        _binding(),
        _local_comparison(),
        api_key="test",
        transport=transport,
    )
    assert calls == 1
    assert result["status"] == "partial"
    assert result["failure_category"] == "usage_overflow"
    assert result["completed_request_count"] == 1
    assert result["completed_decision_count"] == 1


def test_http_body_is_redacted_from_partial_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    common = tmp_path / "common"
    common.mkdir()
    monkeypatch.setattr(runner, "git_common_dir", lambda _: common)

    def transport(
        method: str, url: str, headers: Mapping[str, str], body: bytes, timeout: float
    ) -> tuple[int, Mapping[str, str], bytes]:
        return 500, {}, b"provider-private-response"

    result = runner._run_prevalidated(
        tmp_path,
        tmp_path / "output",
        _binding(),
        _local_comparison(),
        api_key="test",
        transport=transport,
    )
    assert result["failure_category"] == "http_error"
    for name in ("result.json", "cost-journal.jsonl", "report.md"):
        assert "provider-private-response" not in (tmp_path / "output" / name).read_text()


def test_artifact_tampering_is_rejected_after_a_complete_fake_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    common = tmp_path / "common"
    common.mkdir()
    monkeypatch.setattr(runner, "git_common_dir", lambda _: common)

    def transport(
        method: str, url: str, headers: Mapping[str, str], body: bytes, timeout: float
    ) -> tuple[int, Mapping[str, str], bytes]:
        return 200, {}, _response(json.loads(body))

    output = tmp_path / "output"
    provenance = _binding()
    runner._run_prevalidated(
        tmp_path,
        output,
        provenance,
        _local_comparison(),
        api_key="test",
        transport=transport,
    )
    result = json.loads((output / "result.json").read_bytes())
    result["summary"]["input_tokens"] += 1
    (output / "result.json").write_bytes(plan.canonical(result) + b"\n")
    with pytest.raises(ValueError, match="result reconstruction"):
        runner.validate_artifact(
            tmp_path,
            output,
            provenance["prior_ledger"],
            provenance,
            _local_comparison(),
        )


@pytest.mark.parametrize("name", ("result.json", "report.md", "artifact-manifest.json"))
def test_finalization_recovers_after_each_atomic_publication_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    common = tmp_path / "common"
    common.mkdir()
    monkeypatch.setattr(runner, "git_common_dir", lambda _: common)
    output = tmp_path / "output"
    output.mkdir(mode=0o700)
    journal_path = output / "cost-journal.jsonl"
    journal_path.touch(mode=0o600)
    provenance = _binding()
    runner.create_claim(tmp_path, provenance["prior_ledger"], provenance)
    original = runner._atomic_match

    def interrupt(path: Path, payload: bytes) -> None:
        if path.name == name:
            raise RuntimeError("simulated interruption")
        original(path, payload)

    monkeypatch.setattr(runner, "_atomic_match", interrupt)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        runner.finalize(
            tmp_path,
            output,
            provenance["prior_ledger"],
            provenance,
            _local_comparison(),
        )
    monkeypatch.setattr(runner, "_atomic_match", original)
    recovered = runner.finalize(
        tmp_path,
        output,
        provenance["prior_ledger"],
        provenance,
        _local_comparison(),
    )
    assert recovered["status"] == "partial"


def test_remote_tooling_is_checkout_only_and_excluded_from_wheel() -> None:
    assert not inspect_wheel._allowed_entry("benchmarks/universal_remote/runner.py")


def test_default_environment_does_not_load_provider_sdk() -> None:
    assert "typesafe" not in __import__("sys").modules


def _binding() -> dict[str, str]:
    return {
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


def _local_comparison() -> list[dict[str, Any]]:
    return [
        {
            "candidate_id": "laya-multilingual",
            "completed_decision_count": 182,
            "latency_p50_ms": 10.0,
            "decisions_per_second": 18_200.0,
        },
        {
            "candidate_id": "mdeberta-nli",
            "completed_decision_count": 183,
            "latency_p50_ms": 20.0,
            "decisions_per_second": 9_150.0,
        },
    ]
