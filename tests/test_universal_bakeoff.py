import hashlib
import json
import socket
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from benchmarks import universal_bakeoff as bakeoff


def test_phase4c_manifests_validate_and_bind_exact_registry_bytes() -> None:
    registry = bakeoff.load_candidate_registry()
    plan = bakeoff.load_plan()
    assert [candidate.id for candidate in registry.candidates] == list(bakeoff.EXPECTED_IDS)
    assert (
        plan.candidate_registry_sha256
        == hashlib.sha256(bakeoff.REGISTRY_PATH.read_bytes()).hexdigest()
    )
    assert len(plan.scenarios) == 8


def test_duplicate_keys_and_unknown_fields_fail_closed() -> None:
    raw = (
        b'{"schema_version":"decision-backend-candidates.v1",'
        b'"schema_version":"x","reviewed_at":"2026-09-22","candidates":[]}'
    )
    with pytest.raises(ValueError, match="invalid closed benchmark contract"):
        bakeoff.load_candidate_registry(raw)
    payload = json.loads(bakeoff.REGISTRY_PATH.read_text(encoding="utf-8"))
    payload["unexpected"] = True
    with pytest.raises(ValueError, match="invalid closed benchmark contract"):
        bakeoff.load_candidate_registry(json.dumps(payload).encode())


@pytest.mark.parametrize(
    ("candidate_index", "field", "value"),
    (
        (0, "source_url", "https://github.com/felhen-ai/saracura/tree/main"),
        (1, "source_revision", "main"),
        (2, "license_evidence_url", "https://user@example.com/license"),
        (3, "id", "laya-multilingual"),
        (4, "claimed_local_mac", True),
        (5, "model_id", "mutable-service-alias"),
        (6, "exclusion_reason", None),
    ),
)
def test_candidate_registry_rejects_identity_and_boundary_mutations(
    candidate_index: int, field: str, value: Any
) -> None:
    payload = json.loads(bakeoff.REGISTRY_PATH.read_text(encoding="utf-8"))
    payload["candidates"][candidate_index][field] = value
    with pytest.raises(ValueError):
        bakeoff.load_candidate_registry(json.dumps(payload).encode())


def test_candidate_registry_rejects_control_characters() -> None:
    payload = json.loads(bakeoff.REGISTRY_PATH.read_text(encoding="utf-8"))
    payload["candidates"][0]["notes"] = "A sufficiently long note with a hidden\ncontrol character."
    with pytest.raises(ValueError, match="printable"):
        bakeoff.load_candidate_registry(json.dumps(payload).encode())


def test_plan_rejects_digest_budget_dimensions_metrics_and_authority() -> None:
    payload = json.loads(bakeoff.PLAN_PATH.read_text(encoding="utf-8"))
    mutations: tuple[Callable[[dict[str, Any]], None], ...] = (
        lambda value: value.update(candidate_registry_sha256="0" * 64),
        lambda value: value["remote_budget"].update(requests=1),
        lambda value: value["scenarios"][0].update(K=77),
        lambda value: value["scenarios"][0]["authorized_metrics"].append("accuracy"),
        lambda value: value["report_policy"].update(conclusion_authority="recommend"),
        lambda value: value.update(generator_revision="mutable-generator"),
        lambda value: value.update(plan_id="x" * 10_000),
    )
    for mutate in mutations:
        candidate = json.loads(json.dumps(payload))
        mutate(candidate)
        with pytest.raises(ValueError):
            bakeoff.load_plan(json.dumps(candidate).encode())


def test_plan_rejects_non_finite_numbers() -> None:
    raw = bakeoff.PLAN_PATH.read_bytes().replace(b'"usd": 0', b'"usd": NaN')
    with pytest.raises(ValueError, match="invalid closed benchmark contract"):
        bakeoff.load_plan(raw)


def test_reference_adapters_are_fixture_only_and_shape_distinct() -> None:
    plan = bakeoff.load_plan()
    rows = bakeoff._reference_results(plan)
    assert len(rows) == 32
    assert {row["adapter"] for row in rows} == {
        "compiled_reference_adapter",
        "joint_encoder_reference_adapter",
        "pairwise_nli_reference_adapter",
        "constrained_lm_reference_adapter",
    }
    assert all(row["status"] == "fixture_only" for row in rows)
    assert all(row["metric_authority"] == "declared_analytic_model" for row in rows)
    assert all(set(row["metrics"]) == set(bakeoff.METRICS) for row in rows)
    assert all(row["placeholder_ranking"] == "deterministic_placeholder" for row in rows)
    first = {row["adapter"]: row for row in rows if row["scenario_id"] == "scenario-01"}
    assert first["compiled_reference_adapter"]["metrics"]["declared_sequence_count"] == 1
    assert first["joint_encoder_reference_adapter"]["metrics"]["declared_sequence_count"] == 1
    assert first["pairwise_nli_reference_adapter"]["metrics"]["declared_sequence_count"] == 5
    assert first["constrained_lm_reference_adapter"]["metrics"]["declared_sequence_count"] == 1
    assert len({row["metrics"]["declared_attention_work"] for row in first.values()}) == 4


def test_fixture_generator_materializes_declared_dimensions() -> None:
    for scenario in bakeoff.load_plan().scenarios:
        states, questions, choices = bakeoff._materialize_scenario(scenario)
        assert len(states) == scenario.N
        assert all(len(state_questions) == scenario.Q for state_questions in questions)
        assert all(
            len(choice_set) == scenario.K
            for state_choices in choices
            for choice_set in state_choices
        )


def test_semantic_frames_are_length_prefixed_and_unambiguous() -> None:
    assert bakeoff._frame(b"a", b"bc") != bakeoff._frame(b"ab", b"c")
    assert bakeoff._frame(b"a", b"bc") == b"\x00\x00\x00\x00\x00\x00\x00\x01a" + (
        b"\x00\x00\x00\x00\x00\x00\x00\x02bc"
    )


def test_reference_run_writes_immutable_three_file_artifact_without_network_or_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(bakeoff, "ARTIFACT_ROOT", tmp_path / "universal-bakeoff")
    monkeypatch.setattr(
        socket, "socket", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network"))
    )
    monkeypatch.setattr(
        "os.getenv", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("credentials"))
    )
    output = bakeoff.run_reference("offline-run")
    assert {path.name for path in output.iterdir()} == {
        "result.json",
        "report.md",
        "artifact-manifest.json",
    }
    assert output.stat().st_mode & 0o777 == 0o700
    assert all(path.stat().st_mode & 0o777 == 0o600 for path in output.iterdir())
    manifest = json.loads((output / "artifact-manifest.json").read_text(encoding="utf-8"))
    for name in ("result.json", "report.md"):
        assert manifest["files"][name]["bytes"] == (output / name).stat().st_size
        assert (
            manifest["files"][name]["sha256"]
            == hashlib.sha256((output / name).read_bytes()).hexdigest()
        )
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    assert manifest["manifest_sha256"] == hashlib.sha256(bakeoff._canonical(unsigned)).hexdigest()
    with pytest.raises(ValueError, match="output already exists"):
        bakeoff.run_reference("offline-run")
    second = bakeoff.run_reference("offline-run-2")
    for name in ("result.json", "report.md", "artifact-manifest.json"):
        assert (output / name).read_bytes() == (second / name).read_bytes()
    result = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert result["generator_revision"] == bakeoff.FIXTURE_GENERATOR_REVISION
    serialized = json.dumps(result)
    for forbidden in ("latency", "throughput", "accuracy", "winner", "recommendation", "tokens"):
        assert forbidden not in serialized


def test_reference_run_reads_and_hashes_one_plan_snapshot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    original = bakeoff.PLAN_PATH.read_bytes()

    class FlippingPlan:
        calls = 0

        def read_bytes(self) -> bytes:
            self.calls += 1
            return original if self.calls == 1 else b"{}"

    plan = FlippingPlan()
    monkeypatch.setattr(bakeoff, "PLAN_PATH", plan)
    monkeypatch.setattr(bakeoff, "ARTIFACT_ROOT", tmp_path / "universal-bakeoff")
    output = bakeoff.run_reference("single-snapshot")
    result = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert plan.calls == 1
    assert result["plan_sha256"] == hashlib.sha256(original).hexdigest()


def test_public_cli_error_does_not_echo_input(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert bakeoff.main(["run-reference", "--output", str(tmp_path / "secret-scenario")]) == 2
    assert "secret-scenario" not in capsys.readouterr().err


def test_public_cli_parse_error_does_not_echo_input(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert bakeoff.main(["validate", "--secret-path=/private/redaction-sentinel"]) == 2
    captured = capsys.readouterr()
    assert "redaction-sentinel" not in captured.out
    assert "redaction-sentinel" not in captured.err
    assert captured.err.strip() == "error:contract_invalid"


def test_public_cli_success_does_not_echo_local_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    artifact_root = tmp_path / "private-user-path" / "universal-bakeoff"
    monkeypatch.setattr(bakeoff, "ARTIFACT_ROOT", artifact_root)
    monkeypatch.setattr(
        socket, "socket", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network"))
    )
    monkeypatch.setattr(
        "os.getenv", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("credentials"))
    )
    assert bakeoff.main(["validate"]) == 0
    assert bakeoff.main(["run-reference", "--output", "bounded-run"]) == 0
    captured = capsys.readouterr()
    assert "private-user-path" not in captured.out
    assert "private-user-path" not in captured.err
