import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from benchmarks.v02_evaluation import (
    CONTROL_REVISIONS,
    PROTOCOL_SPEC_SHA256,
    combined_content_canonical_bytes,
    combined_content_fingerprint,
    validate_protocol,
    validate_report,
)
from benchmarks.validate_manifests import validate_routed_manifest


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _run(root: Path, *args: str) -> str:
    return subprocess.run(args, cwd=root, check=True, text=True, capture_output=True).stdout.strip()


def _descriptor() -> dict[str, Any]:
    distribution = {str(n): 14 for n in range(2, 9)}
    distribution["4"] = 16
    gold = {str(n): {str(pos): 14 // n + (pos < 14 % n) for pos in range(n)} for n in range(2, 9)}
    gold["4"] = {str(pos): 4 for pos in range(4)}
    return {
        "schema_version": "v02-sealed-descriptor.v1",
        "record_count": 100,
        "payload_digest": "a" * 64,
        "identity_set_digest": "b" * 64,
        "state_question_fingerprint_set_digest": "c" * 64,
        "combined_content_fingerprint_set_digest": "d" * 64,
        "descriptive_option_multiset_fingerprint_set_digest": "e" * 64,
        "slice_counts": {"domain": {"general": 100}},
        "option_count_distribution": distribution,
        "gold_position_distribution": gold,
        "source_class": "original_contributed",
        "contamination_risk": "unknown",
        "max_rendered_input_tokens_observed": 512,
        "kev_base_model_revision": "1001bb4d826a52d1f399e183466143f4da7b741b",
        "kev_adapter_revision": CONTROL_REVISIONS["kev"],
        "kev_rendering_function_digest": (
            "eca2a60af37c539c984e89cf920c53e8d1c93cff6e980dea1a24dd520f86e169"
        ),
        "truncation_disabled": True,
        "all_records_kev_preflight_passed": True,
        "permutation_subset_size": 50,
        "permutation_selection_seed": "saracura-v02-permutation-v1",
        "permutation_selection_digest": "f" * 64,
        "permutation_strategy": "cyclic_left_rotation_one",
    }


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    repo = tmp_path / "repo"
    repo.mkdir(parents=True)
    _run(repo, "git", "init")
    _run(repo, "git", "config", "user.email", "test@example.com")
    _run(repo, "git", "config", "user.name", "Test")
    descriptor = repo / "receipts/sealed.json"
    _write(descriptor, _descriptor())
    _run(repo, "git", "add", ".")
    _run(repo, "git", "commit", "-m", "descriptor")
    descriptor_commit = _run(repo, "git", "rev-parse", "HEAD")
    _write(repo / "marker.txt", {"version": 1})
    _run(repo, "git", "add", ".")
    _run(repo, "git", "commit", "-m", "selection")
    selection_commit = _run(repo, "git", "rev-parse", "HEAD")
    digest = hashlib.sha256(descriptor.read_bytes()).hexdigest()
    receipt = tmp_path / "receipt.json"
    receipt_data: dict[str, Any] = {
        "schema_version": "v02-disjointness-receipt.v1",
        "training_descriptor_digest": "1" * 64,
        "sealed_descriptor_digest": digest,
        "training_identity_digest": "2" * 64,
        "public_dev_identity_digest": "3" * 64,
        "sealed_test_identity_digest": "b" * 64,
        "training_state_question_digest": "4" * 64,
        "public_dev_state_question_digest": "5" * 64,
        "sealed_test_state_question_digest": "c" * 64,
        "training_combined_content_digest": "6" * 64,
        "public_dev_combined_content_digest": "7" * 64,
        "sealed_test_combined_content_digest": "d" * 64,
        "training_option_multiset_digest": "8" * 64,
        "public_dev_option_multiset_digest": "9" * 64,
        "sealed_test_option_multiset_digest": "e" * 64,
        "pairwise_intersection_counts": {
            kind: {
                pair: 0
                for pair in (
                    "training_public_dev",
                    "training_sealed_test",
                    "public_dev_sealed_test",
                )
            }
            for kind in ("identity", "state_question", "combined_content", "option_multiset")
        },
    }
    receipt_data["pairwise_intersection_counts"]["option_multiset"]["training_public_dev"] = 3
    _write(receipt, receipt_data)
    selection = tmp_path / "selection.json"
    selection_data = {
        "schema_version": "v02-selection-receipt.v1",
        "candidate": {
            "id": "saracura-v02",
            "checkpoint_revision": "f" * 40,
            "tokenizer_revision": "a" * 40,
        },
        "training_capsule": "capsule-v1",
        "code_revision": selection_commit,
        "hyperparameters": {"seed": 1},
        "seed": "selection-v1",
        "development_report_digest": "0" * 64,
        "sealed_descriptor_digest": digest,
        "descriptor_commit": descriptor_commit,
        "descriptor_path": "receipts/sealed.json",
        "selection_commit": selection_commit,
        "candidate_deployment_class": "qwen35_4b_pointer_head",
    }
    _write(selection, selection_data)
    report = tmp_path / "report.json"
    settings = {
        "task": "choice",
        "locale": "pt-BR",
        "preprocessing": "frozen-v1",
        "scoring": "planned-top1-v1",
        "planned_denominator": 100,
        "host_class": "test",
    }

    def row(name: str, correct: int, revision: str) -> dict[str, Any]:
        model_ids = {
            "saracura": "saracura-v02",
            "kev": "kev-4b",
            "laya": "laya-multilingual",
            "julia": "julia-1-cyclic-mean",
        }
        return {
            "status": "available",
            "model_id": model_ids[name],
            "model_revision": revision,
            "tokenizer_revision": "a" * 40,
            "code_revision": selection_commit,
            "quality_settings": settings,
            "counts": {"correct": correct, "valid": 100, "invalid": 0, "errors": 0},
            "metrics": {
                "planned_top1_accuracy": correct / 100,
                "coverage": 1.0,
                "valid_only_top1_accuracy": correct / 100,
                "macro_f1_by_option_position": correct / 100,
                "invalid_output_rate": 0.0,
                "error_rate": 0.0,
            },
            "input_truncation_detected": 0,
            "training_overlap_disclosure": "unknown",
            "systems_comparability_group": "qwen-cuda",
        }

    report_data = {
        "schema_version": "saracura-v02-evaluation-report.v1",
        "protocol_digest": PROTOCOL_SPEC_SHA256,
        "sealed_descriptor_digest": digest,
        "disjointness_receipt_digest": hashlib.sha256(receipt.read_bytes()).hexdigest(),
        "selection_receipt_digest": hashlib.sha256(selection.read_bytes()).hexdigest(),
        "planned_denominator": 100,
        "quality_settings": settings,
        "model_rows": {
            "saracura": row("saracura", 82, "f" * 40),
            "kev": row("kev", 75, CONTROL_REVISIONS["kev"]),
            "laya": row("laya", 70, CONTROL_REVISIONS["laya"]),
            "julia": row("julia", 60, CONTROL_REVISIONS["julia"]),
        },
        "paired_comparison_counters": {
            "candidate_correct_control_wrong": 9,
            "candidate_wrong_control_correct": 2,
        },
        "slice_aggregates": {
            "domain": {"general": {"planned": 100, "correct": 82, "accuracy": 0.82}},
            "option_count": {"4": {"planned": 100, "correct": 82, "accuracy": 0.82}},
            "state_length_bucket": {"short": {"planned": 100, "correct": 82, "accuracy": 0.82}},
            "gold_position": {"0": {"planned": 100, "correct": 82, "accuracy": 0.82}},
        },
        "systems_measurements": None,
        "limitations": ["no_calibration_or_automation"],
        "option_order_stability": {
            "subset_size": 50,
            "selection_seed": "saracura-v02-permutation-v1",
            "selection_digest": "f" * 64,
            "strategy": "cyclic_left_rotation_one",
            "stable": 50,
            "evaluated": 50,
            "stability_rate": 1.0,
        },
        "deterministic_repeat_stability": {
            "subset_size": 50,
            "selection_seed": "saracura-v02-permutation-v1",
            "selection_digest": "f" * 64,
            "repetitions": 2,
            "stable": 50,
            "evaluated": 50,
            "stability_rate": 1.0,
        },
        "claim": None,
    }
    _write(report, report_data)
    return repo, descriptor, receipt, selection, report


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _save(path: Path, data: dict[str, Any]) -> None:
    _write(path, data)


def _refresh_report_digests(report: Path, descriptor: Path, receipt: Path, selection: Path) -> None:
    data = _load(report)
    data["sealed_descriptor_digest"] = hashlib.sha256(descriptor.read_bytes()).hexdigest()
    data["disjointness_receipt_digest"] = hashlib.sha256(receipt.read_bytes()).hexdigest()
    data["selection_receipt_digest"] = hashlib.sha256(selection.read_bytes()).hexdigest()
    _save(report, data)


def test_protocol_and_routed_manifest_pass() -> None:
    validate_protocol()
    validate_routed_manifest(
        Path(__file__).parents[1] / "benchmarks/manifests/v02-model-evaluation-protocol.v1.json"
    )


def test_combined_content_is_order_and_id_invariant_with_duplicates() -> None:
    first = combined_content_canonical_bytes(
        {"Café": "  SIM "}, " Q ", [{"id": "x", "text": "Não"}, {"id": "y", "text": "Não"}]
    )
    second = combined_content_canonical_bytes(
        {"café": "sim"}, "q", [{"id": "different", "text": "não"}, {"id": "other", "text": "NÃO"}]
    )
    assert first == second
    assert combined_content_fingerprint(
        {"Café": " SIM "}, "Q", [{"text": "Não"}, {"text": "Não"}]
    ) == combined_content_fingerprint({"café": "sim"}, "q", [{"text": "NÃO"}, {"text": "não"}])


def test_combined_content_rejects_normalized_key_collision() -> None:
    with pytest.raises(ValueError, match="key collision"):
        combined_content_canonical_bytes({"Café": 1, "café": 2}, "q", [])


def test_combined_content_supports_finite_rfc8785_numbers() -> None:
    first = combined_content_canonical_bytes({"amount": 1.5}, "q", [{"text": "yes"}])
    second = combined_content_canonical_bytes({"amount": 1.5}, "q", [{"text": "yes"}])
    assert first == second
    assert b"1.5" in first


def test_valid_fixture_and_option_only_overlap_pass(tmp_path: Path) -> None:
    repo, descriptor, receipt, selection, report = _fixture(tmp_path)
    validate_report(report, descriptor, receipt, selection, repo)


@pytest.mark.parametrize("kind", ["identity", "state_question", "combined_content"])
def test_nonzero_required_pairwise_intersection_fails(tmp_path: Path, kind: str) -> None:
    repo, descriptor, receipt, selection, report = _fixture(tmp_path)
    data = _load(receipt)
    data["pairwise_intersection_counts"][kind]["training_sealed_test"] = 1
    _save(receipt, data)
    _refresh_report_digests(report, descriptor, receipt, selection)
    with pytest.raises(ValueError, match="pairwise intersections"):
        validate_report(report, descriptor, receipt, selection, repo)


def test_descriptor_gates_and_receipt_binding_fail(tmp_path: Path) -> None:
    repo, descriptor, receipt, selection, report = _fixture(tmp_path)
    data = _load(descriptor)
    data["record_count"] = 99
    _save(descriptor, data)
    with pytest.raises(ValueError, match="record_count"):
        validate_report(report, descriptor, receipt, selection, repo)
    data = _descriptor()
    data["option_count_distribution"]["2"] = 13
    _save(descriptor, data)
    with pytest.raises(ValueError, match="at least 14"):
        validate_report(report, descriptor, receipt, selection, repo)


def test_receipt_digest_binding_and_committed_blob_gate_fail(tmp_path: Path) -> None:
    repo, descriptor, receipt, selection, report = _fixture(tmp_path)
    receipt_data = _load(receipt)
    receipt_data["sealed_test_identity_digest"] = "0" * 64
    _save(receipt, receipt_data)
    _refresh_report_digests(report, descriptor, receipt, selection)
    with pytest.raises(ValueError, match="does not bind descriptor"):
        validate_report(report, descriptor, receipt, selection, repo)

    repo, descriptor, receipt, selection, report = _fixture(tmp_path / "blob")
    descriptor_data = _load(descriptor)
    descriptor_data["payload_digest"] = "0" * 64
    _save(descriptor, descriptor_data)
    digest = hashlib.sha256(descriptor.read_bytes()).hexdigest()
    receipt_data = _load(receipt)
    receipt_data["sealed_descriptor_digest"] = digest
    _save(receipt, receipt_data)
    selection_data = _load(selection)
    selection_data["sealed_descriptor_digest"] = digest
    _save(selection, selection_data)
    _refresh_report_digests(report, descriptor, receipt, selection)
    with pytest.raises(ValueError, match="blob digest"):
        validate_report(report, descriptor, receipt, selection, repo)


def test_descriptor_requires_source_and_contamination_disclosure(tmp_path: Path) -> None:
    repo, descriptor, receipt, selection, report = _fixture(tmp_path)
    data = _load(descriptor)
    data["contamination_risk"] = "none"
    _save(descriptor, data)
    with pytest.raises(ValueError, match="source class or contamination"):
        validate_report(report, descriptor, receipt, selection, repo)


def test_descriptor_proves_frozen_kev_token_envelope(tmp_path: Path) -> None:
    repo, descriptor, receipt, selection, report = _fixture(tmp_path)
    data = _load(descriptor)
    data["max_rendered_input_tokens_observed"] = 513
    _save(descriptor, data)
    with pytest.raises(ValueError, match="token envelope"):
        validate_report(report, descriptor, receipt, selection, repo)


def test_stability_results_bind_sealed_permutation_subset(tmp_path: Path) -> None:
    repo, descriptor, receipt, selection, report = _fixture(tmp_path)
    data = _load(report)
    data["option_order_stability"]["selection_digest"] = "0" * 64
    _save(report, data)
    with pytest.raises(ValueError, match="deterministic subset"):
        validate_report(report, descriptor, receipt, selection, repo)


def test_selection_path_blob_and_class_gates_fail(tmp_path: Path) -> None:
    repo, descriptor, receipt, selection, report = _fixture(tmp_path)
    data = _load(selection)
    data["descriptor_path"] = "../sealed.json"
    _save(selection, data)
    _refresh_report_digests(report, descriptor, receipt, selection)
    with pytest.raises(ValueError, match="safe repository-relative"):
        validate_report(report, descriptor, receipt, selection, repo)
    data["descriptor_path"] = "receipts/sealed.json"
    data["candidate_deployment_class"] = "other"
    _save(selection, data)
    _refresh_report_digests(report, descriptor, receipt, selection)
    validate_report(report, descriptor, receipt, selection, repo)
    report_data = _load(report)
    report_data["claim"] = {
        "type": "quality",
        "reference": "kev-4b",
        "task": "choice",
        "locale": "pt-BR",
        "target_model_revision": "f" * 40,
        "reference_model_revision": CONTROL_REVISIONS["kev"],
        "hardware": "test",
        "metric": "planned_top1_accuracy",
        "denominator": 100,
        "winning_margin": 0.07,
    }
    _save(report, report_data)
    with pytest.raises(ValueError, match="scope"):
        validate_report(report, descriptor, receipt, selection, repo)


def test_closed_unavailable_row_and_common_quality_gate(tmp_path: Path) -> None:
    repo, descriptor, receipt, selection, report = _fixture(tmp_path)
    data = _load(report)
    data["model_rows"]["laya"] = {
        "status": "unavailable",
        "reason": "not_installed",
        "training_overlap_disclosure": "unknown",
    }
    data["model_rows"]["julia"]["quality_settings"]["scoring"] = "other"
    _save(report, data)
    with pytest.raises(ValueError, match="quality settings"):
        validate_report(report, descriptor, receipt, selection, repo)
    data["model_rows"]["julia"]["quality_settings"] = data["quality_settings"]
    _save(report, data)
    validate_report(report, descriptor, receipt, selection, repo)


def test_correct_count_cannot_exceed_valid_outputs(tmp_path: Path) -> None:
    repo, descriptor, receipt, selection, report = _fixture(tmp_path)
    data = _load(report)
    row = data["model_rows"]["saracura"]
    row["counts"] = {"correct": 100, "valid": 0, "invalid": 100, "errors": 0}
    row["metrics"] = {
        "planned_top1_accuracy": 1.0,
        "coverage": 0.0,
        "valid_only_top1_accuracy": 0.0,
        "macro_f1_by_option_position": 0.0,
        "invalid_output_rate": 1.0,
        "error_rate": 0.0,
    }
    _save(report, data)
    with pytest.raises(ValueError, match="cannot exceed valid"):
        validate_report(report, descriptor, receipt, selection, repo)


def test_macro_f1_must_be_finite_and_bounded(tmp_path: Path) -> None:
    repo, descriptor, receipt, selection, report = _fixture(tmp_path)
    data = _load(report)
    data["model_rows"]["saracura"]["metrics"]["macro_f1_by_option_position"] = float("nan")
    _save(report, data)
    with pytest.raises(ValueError, match="macro F1"):
        validate_report(report, descriptor, receipt, selection, repo)


def test_control_model_identity_cannot_carry_free_text(tmp_path: Path) -> None:
    repo, descriptor, receipt, selection, report = _fixture(tmp_path)
    data = _load(report)
    data["model_rows"]["kev"]["model_id"] = "RAW HELD-OUT STATE AND QUESTION"
    _save(report, data)
    with pytest.raises(ValueError, match="identity or revision"):
        validate_report(report, descriptor, receipt, selection, repo)


@pytest.mark.parametrize("field", ["raw_content", "calibration", "automation_authorization"])
def test_closed_report_rejects_forbidden_or_unknown_content(tmp_path: Path, field: str) -> None:
    repo, descriptor, receipt, selection, report = _fixture(tmp_path)
    data = _load(report)
    data[field] = True
    _save(report, data)
    with pytest.raises(ValueError, match="aggregate report"):
        validate_report(report, descriptor, receipt, selection, repo)


def test_nested_prohibited_report_content_is_rejected(tmp_path: Path) -> None:
    repo, descriptor, receipt, selection, report = _fixture(tmp_path)
    data = _load(report)
    data["slice_aggregates"]["domain"]["general"]["raw_content"] = "private"
    _save(report, data)
    with pytest.raises(ValueError, match="prohibited field"):
        validate_report(report, descriptor, receipt, selection, repo)


def test_truncation_and_selection_ancestry_gates_fail(tmp_path: Path) -> None:
    repo, descriptor, receipt, selection, report = _fixture(tmp_path)
    data = _load(report)
    data["model_rows"]["kev"]["input_truncation_detected"] = 1
    _save(report, data)
    with pytest.raises(ValueError, match="truncation"):
        validate_report(report, descriptor, receipt, selection, repo)

    repo, descriptor, receipt, selection, report = _fixture(tmp_path / "ancestry")
    selection_data = _load(selection)
    selection_data["descriptor_commit"] = _run(repo, "git", "rev-parse", "HEAD")
    selection_data["selection_commit"] = _run(repo, "git", "rev-list", "--max-parents=0", "HEAD")
    _save(selection, selection_data)
    _refresh_report_digests(report, descriptor, receipt, selection)
    with pytest.raises(ValueError, match="not an ancestor"):
        validate_report(report, descriptor, receipt, selection, repo)


def test_quality_claim_recomputes_exact_mcnemar_not_supplied_value(tmp_path: Path) -> None:
    repo, descriptor, receipt, selection, report = _fixture(tmp_path)
    data = _load(report)
    data["claim"] = {
        "type": "quality",
        "reference": "kev-4b",
        "task": "choice",
        "locale": "pt-BR",
        "target_model_revision": "f" * 40,
        "reference_model_revision": CONTROL_REVISIONS["kev"],
        "hardware": "test",
        "metric": "planned_top1_accuracy",
        "denominator": 100,
        "winning_margin": 0.07,
    }
    _save(report, data)
    validate_report(report, descriptor, receipt, selection, repo)
    data["paired_comparison_counters"] = {
        "candidate_correct_control_wrong": 15,
        "candidate_wrong_control_correct": 8,
    }
    _save(report, data)
    with pytest.raises(ValueError, match="McNemar"):
        validate_report(report, descriptor, receipt, selection, repo)


def test_quality_claim_uses_named_discordant_counters(tmp_path: Path) -> None:
    repo, descriptor, receipt, selection, report = _fixture(tmp_path)
    data = _load(report)
    data["claim"] = {
        "type": "quality",
        "reference": "kev-4b",
        "task": "choice",
        "locale": "pt-BR",
        "target_model_revision": "f" * 40,
        "reference_model_revision": CONTROL_REVISIONS["kev"],
        "hardware": "test",
        "metric": "planned_top1_accuracy",
        "denominator": 100,
        "winning_margin": 0.07,
    }
    data["paired_comparison_counters"] = {
        "candidate_wrong_control_correct": 9,
        "candidate_correct_control_wrong": 2,
    }
    report.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="McNemar"):
        validate_report(report, descriptor, receipt, selection, repo)


def test_systems_claim_requires_named_identical_group(tmp_path: Path) -> None:
    repo, descriptor, receipt, selection, report = _fixture(tmp_path)
    data = _load(report)
    measure = {
        "group": "qwen-cuda",
        "host_class": "h",
        "runtime_boundary": "r",
        "device_class": "gpu",
        "precision": "bf16",
        "batch": 1,
        "concurrency": 1,
        "timing_procedure": "t",
        "cold_load_time_ms": 100.0,
        "warm_request_p50_ms": 10.0,
        "warm_request_p95_ms": 15.0,
        "decisions_per_second": 100.0,
        "peak_host_rss_bytes": 1024,
        "peak_device_memory_bytes": 2048,
        "artifact_bytes": 4096,
    }
    data["systems_measurements"] = {"saracura": measure, "kev": dict(measure)}
    data["claim"] = {"type": "systems", "reference": "kev-4b", "target": "saracura"}
    _save(report, data)
    validate_report(report, descriptor, receipt, selection, repo)
    data["systems_measurements"]["kev"]["group"] = "other"
    _save(report, data)
    with pytest.raises(ValueError, match="comparability"):
        validate_report(report, descriptor, receipt, selection, repo)
    data["systems_measurements"]["kev"]["group"] = "qwen-cuda"
    data["systems_measurements"]["kev"]["host_class"] = "different"
    _save(report, data)
    with pytest.raises(ValueError, match="comparability"):
        validate_report(report, descriptor, receipt, selection, repo)
