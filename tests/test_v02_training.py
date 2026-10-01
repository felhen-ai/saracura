import hashlib
import json
import os
import stat
import subprocess
from pathlib import Path

import pytest

from benchmarks import v02_training
from benchmarks.v02_evaluation import validate_selection
from benchmarks.validate_manifests import validate_routed_manifest


def _record(identifier: str, split: str) -> dict[str, object]:
    return {
        "opaque_record_id": identifier,
        "split": split,
        "state": {"policy": " Café "},
        "question": " Qual ação? ",
        "options": [{"id": "a", "description": "Agir"}, {"id": "b", "description": "Parar"}],
    }


def _candidate(identifier: str, score: float = 0.5) -> dict[str, object]:
    return {
        "id": identifier,
        "coverage": 1.0,
        "invalid_output_rate": 0.0,
        "repeat_stability": 1.0,
        "option_order_stability": 1.0,
        "planned_top1_accuracy": score,
        "warm_p95_ms": 10.0,
        "valid": True,
        "truncation_detected": False,
        "identity_match": True,
    }


def _run(root: Path, *args: str) -> str:
    return subprocess.run(args, cwd=root, check=True, text=True, capture_output=True).stdout.strip()


def test_manifest_is_closed_and_routed() -> None:
    manifest = v02_training.load_manifest()
    assert manifest["candidate_grid"] == list(v02_training.CANDIDATE_GRID)
    validate_routed_manifest(v02_training.MANIFEST_PATH)


def test_sealer_includes_train_and_internal_dev_with_exact_normalized_fingerprints() -> None:
    value = v02_training.seal_training(
        [_record("one", "train"), _record("two", "internal_dev")],
        accepted_packet_digest="a" * 64,
        successor_grant_digest="b" * 64,
        source_policy_identifiers=["phase4e-train", "phase4e-internal-dev"],
    )
    v02_training.validate_training_descriptor(value)
    assert value["split_counts"] == {"train": 1, "internal_dev": 1}
    assert (
        value["candidate_rendering_digest"]
        == v02_training.RENDERER_CONTRACT["candidate_rendering_digest"]
    )
    with pytest.raises(ValueError, match="both"):
        v02_training.seal_training(
            [_record("one", "train")],
            accepted_packet_digest="a" * 64,
            successor_grant_digest="b" * 64,
            source_policy_identifiers=["phase4e-train"],
        )


def test_create_only_rejects_overwrite_symlink_and_unsafe_parent(tmp_path: Path) -> None:
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    output = private / "receipt.json"
    v02_training.create_json(output, {"schema_version": "fixture"})
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    with pytest.raises(FileExistsError):
        v02_training.create_json(output, {"schema_version": "fixture"})
    target = private / "target.json"
    target.write_text("{}", encoding="utf-8")
    link = private / "link.json"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="non-symlink"):
        v02_training.create_json(link, {"schema_version": "fixture"})
    public = tmp_path / "public"
    public.mkdir(mode=0o755)
    os.chmod(public, 0o755)
    with pytest.raises(ValueError, match="private"):
        v02_training.create_json(public / "x.json", {"schema_version": "fixture"})


def test_selection_is_exact_deterministic_and_terminal_without_candidate() -> None:
    rows = [_candidate("c1-r8", 0.6), _candidate("c2-r16", 0.6), _candidate("c3-r32", 0.5)]
    rows[0]["warm_p95_ms"] = 12.0
    rows[1]["warm_p95_ms"] = 8.0
    assert v02_training.select_candidate(rows) == {"status": "SELECTED", "candidate_id": "c2-r16"}
    rows[1]["option_order_stability"] = 0.94
    assert v02_training.select_candidate(rows) == {"status": "SELECTED", "candidate_id": "c1-r8"}
    for row in rows:
        row["coverage"] = 0.97
    assert v02_training.select_candidate(rows) == {"status": "NO_RELEASE_CANDIDATE"}


def test_public_dev_subset_is_deterministic_and_exactly_fifty() -> None:
    rows = [_record(f"id-{index:03d}", "train") for index in range(60)]
    first = v02_training.diagnostic_subset(rows)
    assert len(first) == 50
    assert first == v02_training.diagnostic_subset(list(reversed(rows)))
    with pytest.raises(ValueError, match="at least 50"):
        v02_training.diagnostic_subset(rows[:49])


def test_loopback_plan_and_global_code_revision_replacement_fail_closed() -> None:
    plan = v02_training.loopback_worker_plan()
    assert plan["runtime_boundary"] == "loopback_http_python_worker_v1"
    assert plan["timing_procedure"] == "saracura-v02-loopback-single-request-v1"
    assert plan["loopback_host"] == "127.0.0.1"
    revision = "a" * 40
    v02_training.validate_code_revisions({"descriptor": revision, "smoke": revision}, revision)
    with pytest.raises(ValueError, match="globally"):
        v02_training.validate_code_revisions({"descriptor": revision, "smoke": "b" * 40}, revision)
    with pytest.raises(ValueError, match="successor"):
        v02_training.validate_code_revisions({"selection_commit": revision}, revision)


def _sealed_descriptor() -> dict[str, object]:
    distribution = {str(value): 14 for value in range(2, 9)}
    distribution["4"] = 16
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
        "gold_position_distribution": {
            str(n): {
                str(position): distribution[str(n)] // n + (position < distribution[str(n)] % n)
                for position in range(n)
            }
            for n in range(2, 9)
        },
        "source_class": "original_contributed",
        "contamination_risk": "unknown",
        "max_rendered_input_tokens_observed": 512,
        "kev_base_model_revision": "1001bb4d826a52d1f399e183466143f4da7b741b",
        "kev_adapter_revision": "139fdd94f1b6a6ad80cc15e08fcb99cac885a101",
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


def test_validate_selection_requires_main_reachability_without_report(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _run(repo, "git", "init")
    _run(repo, "git", "config", "user.email", "test@example.com")
    _run(repo, "git", "config", "user.name", "Test")
    descriptor = repo / "receipts" / "sealed.json"
    descriptor.parent.mkdir()
    descriptor.write_text(json.dumps(_sealed_descriptor()), encoding="utf-8")
    _run(repo, "git", "add", ".")
    _run(repo, "git", "commit", "-m", "descriptor")
    descriptor_commit = _run(repo, "git", "rev-parse", "HEAD")
    (repo / "marker").write_text("selection", encoding="utf-8")
    _run(repo, "git", "add", ".")
    _run(repo, "git", "commit", "-m", "selection")
    selection_commit = _run(repo, "git", "rev-parse", "HEAD")
    descriptor_digest = hashlib.sha256(descriptor.read_bytes()).hexdigest()
    receipt = tmp_path / "disjointness.json"
    receipt.write_text(
        json.dumps(
            {
                "schema_version": "v02-disjointness-receipt.v1",
                "training_descriptor_digest": "1" * 64,
                "sealed_descriptor_digest": descriptor_digest,
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
                    for kind in (
                        "identity",
                        "state_question",
                        "combined_content",
                        "option_multiset",
                    )
                },
            }
        ),
        encoding="utf-8",
    )
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "schema_version": "v02-selection-receipt.v1",
                "candidate": {
                    "id": "c1-r8",
                    "checkpoint_revision": "a" * 64,
                    "tokenizer_revision": "1001bb4d826a52d1f399e183466143f4da7b741b",
                },
                "training_capsule": "1" * 64,
                "code_revision": selection_commit,
                "hyperparameters": "2" * 64,
                "seed": "20260929",
                "development_report_digest": "3" * 64,
                "sealed_descriptor_digest": descriptor_digest,
                "descriptor_commit": descriptor_commit,
                "descriptor_path": "receipts/sealed.json",
                "selection_commit": selection_commit,
                "candidate_deployment_class": "qwen35_4b_pointer_head",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="origin/main"):
        validate_selection(descriptor, receipt, selection, repo)
    _run(repo, "git", "update-ref", "refs/remotes/origin/main", selection_commit)
    assert validate_selection(descriptor, receipt, selection, repo) == 0
