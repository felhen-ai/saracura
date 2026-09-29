"""Tests for the v0.2 distillation-safe corpus offline planner."""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest
from pydantic import JsonValue

import benchmarks.v02_corpus as v02_corpus
from benchmarks.validate_manifests import (
    validate_routed_manifest,
    validate_v02_distillation_safe_corpus,
)
from saracura.serialization import canonical_json_bytes

REPO_ROOT = Path(__file__).parents[1]
MANIFEST_PATH = REPO_ROOT / "benchmarks/manifests/v02-distillation-safe-corpus.v1.json"


def _mkdir_0700(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    os.chmod(path, 0o700)


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key: {key}")
        result[key] = value
    return result


def _canonical_manifest_bytes(payload: dict[str, object]) -> bytes:
    return canonical_json_bytes(cast(JsonValue, payload)) + b"\n"


def test_manifest_is_closed_and_routed() -> None:
    validate_v02_distillation_safe_corpus(MANIFEST_PATH)
    validate_routed_manifest(MANIFEST_PATH)


def test_manifest_rejects_duplicate_keys(tmp_path: Path) -> None:
    raw = MANIFEST_PATH.read_bytes()
    payload = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    payload["extra_field"] = "tamper"
    path = tmp_path / "v02-distillation-safe-corpus.v1.json"
    path.write_bytes(_canonical_manifest_bytes(payload))
    with pytest.raises(ValueError, match="schema is not closed"):
        validate_v02_distillation_safe_corpus(path)


def test_manifest_rejects_missing_field(tmp_path: Path) -> None:
    raw = MANIFEST_PATH.read_bytes()
    payload = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    del payload["terminal_values"]
    path = tmp_path / "v02-distillation-safe-corpus.v1.json"
    path.write_bytes(_canonical_manifest_bytes(payload))
    with pytest.raises(ValueError, match="schema mismatch"):
        validate_v02_distillation_safe_corpus(path)


def test_manifest_rejects_noncanonical_bytes(tmp_path: Path) -> None:
    tampered = b"\n" + MANIFEST_PATH.read_bytes()
    path = tmp_path / "v02-distillation-safe-corpus.v1.json"
    path.write_bytes(tampered)
    with pytest.raises(ValueError, match="not canonical"):
        validate_v02_distillation_safe_corpus(path)


def test_manifest_rejects_model_revision_change(tmp_path: Path) -> None:
    raw = MANIFEST_PATH.read_bytes()
    payload = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    payload["model_roles"]["training_author"]["revision"] = "0" * 40
    path = tmp_path / "v02-distillation-safe-corpus.v1.json"
    path.write_bytes(_canonical_manifest_bytes(payload))
    with pytest.raises(ValueError, match="revision mismatch"):
        validate_v02_distillation_safe_corpus(path)


def test_validate_protocol_rejects_unknown_nested_model_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    payload["model_roles"]["training_author"]["unexpected"] = True
    path = tmp_path / "manifest.json"
    path.write_bytes(_canonical_manifest_bytes(payload))
    monkeypatch.setattr(v02_corpus, "MANIFEST_PATH", path)
    with pytest.raises(ValueError, match=r"model role|schema"):
        v02_corpus.validate_protocol()


def test_manifest_rejects_candidate_renderer_digest_drift(tmp_path: Path) -> None:
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    payload["renderer_contracts"]["candidate_rendering_digest"] = "0" * 64
    path = tmp_path / "manifest.json"
    path.write_bytes(_canonical_manifest_bytes(payload))
    with pytest.raises(ValueError, match="candidate_rendering_digest"):
        validate_v02_distillation_safe_corpus(path)


def test_manifest_rejects_license_change(tmp_path: Path) -> None:
    raw = MANIFEST_PATH.read_bytes()
    payload = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    payload["model_roles"]["sealed_author"]["license"] = "BSD-3"
    path = tmp_path / "v02-distillation-safe-corpus.v1.json"
    path.write_bytes(_canonical_manifest_bytes(payload))
    with pytest.raises(ValueError, match="license mismatch"):
        validate_v02_distillation_safe_corpus(path)


def test_manifest_rejects_count_change(tmp_path: Path) -> None:
    raw = MANIFEST_PATH.read_bytes()
    payload = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    payload["corpus_plan"]["total_slots"] = 1599
    path = tmp_path / "v02-distillation-safe-corpus.v1.json"
    path.write_bytes(_canonical_manifest_bytes(payload))
    with pytest.raises(ValueError, match="total_slots mismatch"):
        validate_v02_distillation_safe_corpus(path)


def test_manifest_rejects_threshold_change(tmp_path: Path) -> None:
    raw = MANIFEST_PATH.read_bytes()
    payload = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    payload["corpus_plan"]["acceptance_floor"]["min_accepted_rows"] = 1199
    path = tmp_path / "v02-distillation-safe-corpus.v1.json"
    path.write_bytes(_canonical_manifest_bytes(payload))
    with pytest.raises(ValueError, match="acceptance_floor"):
        validate_v02_distillation_safe_corpus(path)


def test_manifest_rejects_domain_order(tmp_path: Path) -> None:
    raw = MANIFEST_PATH.read_bytes()
    payload = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    payload["corpus_plan"]["domains"] = list(reversed(payload["corpus_plan"]["domains"]))
    path = tmp_path / "v02-distillation-safe-corpus.v1.json"
    path.write_bytes(_canonical_manifest_bytes(payload))
    with pytest.raises(ValueError, match="domain order"):
        validate_v02_distillation_safe_corpus(path)


def test_manifest_rejects_scenario_codes_order(tmp_path: Path) -> None:
    raw = MANIFEST_PATH.read_bytes()
    payload = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    payload["corpus_plan"]["scenario_codes"] = list(
        reversed(payload["corpus_plan"]["scenario_codes"])
    )
    path = tmp_path / "v02-distillation-safe-corpus.v1.json"
    path.write_bytes(_canonical_manifest_bytes(payload))
    with pytest.raises(ValueError, match="scenario_codes"):
        validate_v02_distillation_safe_corpus(path)


def test_manifest_rejects_criterion_roles_order(tmp_path: Path) -> None:
    raw = MANIFEST_PATH.read_bytes()
    payload = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    payload["corpus_plan"]["criterion_roles"] = list(
        reversed(payload["corpus_plan"]["criterion_roles"])
    )
    path = tmp_path / "v02-distillation-safe-corpus.v1.json"
    path.write_bytes(_canonical_manifest_bytes(payload))
    with pytest.raises(ValueError, match="criterion_roles"):
        validate_v02_distillation_safe_corpus(path)


def test_manifest_rejects_domain_scenario_map_drift(tmp_path: Path) -> None:
    raw = MANIFEST_PATH.read_bytes()
    payload = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    payload["corpus_plan"]["domain_scenario_map"]["email_triage"] = ["injected"]
    path = tmp_path / "v02-distillation-safe-corpus.v1.json"
    path.write_bytes(_canonical_manifest_bytes(payload))
    with pytest.raises(ValueError, match="domain_scenario_map"):
        validate_v02_distillation_safe_corpus(path)


def test_manifest_rejects_privacy_boundary_drift(tmp_path: Path) -> None:
    raw = MANIFEST_PATH.read_bytes()
    payload = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    payload["privacy_boundary"]["no_raw_rows_in_git"] = False
    path = tmp_path / "v02-distillation-safe-corpus.v1.json"
    path.write_bytes(_canonical_manifest_bytes(payload))
    with pytest.raises(ValueError, match="privacy_boundary"):
        validate_v02_distillation_safe_corpus(path)


def test_manifest_rejects_file_schema_drift(tmp_path: Path) -> None:
    raw = MANIFEST_PATH.read_bytes()
    payload = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    payload["file_schemas"]["plan"]["required_fields"].append("extra_field")
    path = tmp_path / "v02-distillation-safe-corpus.v1.json"
    path.write_bytes(_canonical_manifest_bytes(payload))
    with pytest.raises(ValueError, match="file_schemas"):
        validate_v02_distillation_safe_corpus(path)


def test_manifest_rejects_privacy_boundary_extra_field(tmp_path: Path) -> None:
    raw = MANIFEST_PATH.read_bytes()
    payload = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    payload["privacy_boundary"]["injected"] = True
    path = tmp_path / "v02-distillation-safe-corpus.v1.json"
    path.write_bytes(_canonical_manifest_bytes(payload))
    with pytest.raises(ValueError, match="privacy_boundary extra"):
        validate_v02_distillation_safe_corpus(path)


def test_manifest_rejects_file_schema_extra_field(tmp_path: Path) -> None:
    raw = MANIFEST_PATH.read_bytes()
    payload = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    payload["file_schemas"]["injected"] = {"schema_version": "x", "required_fields": []}
    path = tmp_path / "v02-distillation-safe-corpus.v1.json"
    path.write_bytes(_canonical_manifest_bytes(payload))
    with pytest.raises(ValueError, match="file_schemas extra"):
        validate_v02_distillation_safe_corpus(path)


def test_validate_protocol_passes() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "benchmarks.v02_corpus", "validate-protocol"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "protocol valid" in result.stdout


def test_validate_protocol_rejects_corrupted_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corrupted = tmp_path / "v02-distillation-safe-corpus.v1.tampered.json"
    corrupted.write_bytes(MANIFEST_PATH.read_bytes().replace(b"Apache-2.0", b"BSD-3"))
    monkeypatch.setattr(v02_corpus, "MANIFEST_PATH", corrupted)
    with pytest.raises(ValueError, match="license mismatch"):
        v02_corpus.validate_protocol()


def test_plan_training_creates_exact_counts(tmp_path: Path) -> None:
    output_parent = tmp_path / "output"
    _mkdir_0700(output_parent)
    path = v02_corpus.plan("training", output_parent)
    assert path.exists()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["schema_version"] == "v02-plan.v1"
    assert data["lane"] == "training"
    assert data["namespace"] == "saracura-v02-cleanroom-v1"
    assert data["seed"] == 20260929
    slot_ids = data["slot_ids"]
    assert len(slot_ids) == 1600
    assert len(set(slot_ids)) == 1600
    assert data["split_totals"]["train"] == 1360
    assert data["split_totals"]["internal_dev"] == 240
    assert data["locale_totals"]["pt_br"] == 960
    assert data["locale_totals"]["english"] == 640
    assert data["pair_totals"]["train"] == 255
    assert data["pair_totals"]["internal_dev"] == 45
    pilot = data["pilot_prefix"]
    assert pilot["count"] == 140
    assert pilot["train_count"] == 119
    assert pilot["internal_dev_count"] == 21
    assert pilot["complete_bilingual_pairs"] == 21


def test_plan_sealed_creates_exact_counts(tmp_path: Path) -> None:
    output_parent = tmp_path / "output"
    _mkdir_0700(output_parent)
    path = v02_corpus.plan("sealed", output_parent)
    assert path.exists()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["schema_version"] == "v02-plan.v1"
    assert data["lane"] == "sealed"
    assert data["seed"] == 20260930
    identities = data["identities"]
    assert len(identities) == 140
    assert data["identity_count"] == 140
    assert data["identities_per_option_count"] == 20
    for oc in [2, 3, 4, 5, 6, 7, 8]:
        count = sum(1 for i in identities if i["option_count"] == oc)
        assert count == 20, f"option_count {oc} has {count} identities"
    pilot = data["pilot_prefix"]
    assert pilot["count"] == 21
    assert pilot["per_option_count"] == 3
    assert data["permutation_selection_seed"] == "saracura-v02-sealed-permutation-v1"
    assert data["permutation_subset_size"] == 50
    assert data["author_target_hidden"] is True
    for ident in identities:
        assert "author_target" not in ident
        assert "gold_position" not in ident


def test_plan_training_option_count_balance() -> None:
    plan = v02_corpus._generate_training_plan()
    oc_alloc = plan["option_count_allocation"]
    counts = list(oc_alloc.values())
    assert sum(counts) == 1600
    sl_alloc = plan["split_locale_option_count_allocation"]
    for key, counts_dict in sl_alloc.items():
        counts = list(counts_dict.values())
        assert max(counts) - min(counts) <= 1, f"OC not balanced in {key}: {counts}"


def test_plan_training_gold_position_balance() -> None:
    plan = v02_corpus._generate_training_plan()
    gold_alloc = plan["gold_position_allocation"]
    for cell_key, golds in gold_alloc.items():
        counts = list(golds.values())
        assert max(counts) - min(counts) <= 1, (
            f"gold positions not balanced in {cell_key}: {counts}"
        )


def test_plan_training_split_locale_totals() -> None:
    plan = v02_corpus._generate_training_plan()
    assert len(plan["slot_ids"]) == 1600
    assert plan["split_totals"]["train"] == 1360
    assert plan["split_totals"]["internal_dev"] == 240
    assert plan["locale_totals"]["pt_br"] == 960
    assert plan["locale_totals"]["english"] == 640


def test_plan_no_absolute_paths_in_bytes(tmp_path: Path) -> None:
    output_parent = tmp_path / "output"
    _mkdir_0700(output_parent)
    path = v02_corpus.plan("training", output_parent)
    data = path.read_text(encoding="utf-8")
    repo_root = str(REPO_ROOT.resolve())
    assert repo_root not in data


def test_plan_rejects_relative_output_parent(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="absolute path"):
        v02_corpus.plan("training", Path("relative-output"))


def test_plan_rejects_nonexistent_parent(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="existing directory"):
        v02_corpus.plan("training", tmp_path / "nonexistent")


def test_plan_rejects_existing_destination(tmp_path: Path) -> None:
    output_parent = tmp_path / "output"
    _mkdir_0700(output_parent)
    v02_corpus.plan("training", output_parent)
    with pytest.raises(ValueError, match="already exists"):
        v02_corpus.plan("training", output_parent)


def test_plan_rejects_symlink_destination(tmp_path: Path) -> None:
    output_parent = tmp_path / "output"
    _mkdir_0700(output_parent)
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    symlink = output_parent / "symlink-output"
    symlink.symlink_to(real_dir)
    with pytest.raises(ValueError, match="symlink"):
        v02_corpus._safe_output_parent(symlink)


def test_plan_rejects_repo_output_parent(tmp_path: Path) -> None:
    repo_sub = REPO_ROOT / ".b1-test-output"
    repo_sub.mkdir(mode=0o700, exist_ok=True)
    try:
        with pytest.raises(ValueError, match="outside the repository"):
            v02_corpus._safe_output_parent(repo_sub)
    finally:
        repo_sub.rmdir()


def test_plan_rejects_0755_output_parent(tmp_path: Path) -> None:
    output_parent = tmp_path / "output-0755"
    output_parent.mkdir(parents=True)
    os.chmod(output_parent, 0o755)
    with pytest.raises(ValueError, match=r"0700|symlink"):
        v02_corpus._safe_output_parent(output_parent)


def test_plan_rejects_1777_output_parent(tmp_path: Path) -> None:
    output_parent = tmp_path / "output-1777"
    output_parent.mkdir(parents=True)
    os.chmod(output_parent, 0o1777)
    with pytest.raises(ValueError, match=r"0700|symlink"):
        v02_corpus._safe_output_parent(output_parent)


def test_plan_rejects_git_common_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    git_common = tmp_path / "git-common"
    output_parent = git_common / "private-output"
    _mkdir_0700(output_parent)
    monkeypatch.setattr(v02_corpus, "_git_common_dir", lambda _repo: git_common.resolve())
    with pytest.raises(ValueError, match="outside Git"):
        v02_corpus._safe_output_parent(output_parent)


def test_deterministic_training_plan() -> None:
    plan1 = v02_corpus._generate_training_plan()
    plan2 = v02_corpus._generate_training_plan()
    assert json.dumps(plan1, sort_keys=True) == json.dumps(plan2, sort_keys=True)


def test_deterministic_sealed_plan() -> None:
    plan1 = v02_corpus._generate_sealed_plan()
    plan2 = v02_corpus._generate_sealed_plan()
    assert json.dumps(plan1, sort_keys=True) == json.dumps(plan2, sort_keys=True)


def test_validate_v02_distillation_safe_corpus_passes() -> None:
    validate_v02_distillation_safe_corpus(MANIFEST_PATH)


def test_validate_v02_distillation_safe_corpus_rejects_wrong_schema(tmp_path: Path) -> None:
    raw = MANIFEST_PATH.read_bytes()
    payload = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    payload["schema_version"] = "unknown-schema.v1"
    path = tmp_path / "manifest.json"
    path.write_bytes(_canonical_manifest_bytes(payload))
    with pytest.raises(ValueError, match="schema version"):
        validate_v02_distillation_safe_corpus(path)


def test_validate_v02_distillation_safe_corpus_rejects_runtime_drift(tmp_path: Path) -> None:
    raw = MANIFEST_PATH.read_bytes()
    payload = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    payload["runtime_policy"]["quantization"] = "GPTQ"
    path = tmp_path / "manifest.json"
    path.write_bytes(_canonical_manifest_bytes(payload))
    with pytest.raises(ValueError, match="quantization mismatch"):
        validate_v02_distillation_safe_corpus(path)


def test_validate_v02_distillation_safe_corpus_rejects_renderer_drift(tmp_path: Path) -> None:
    raw = MANIFEST_PATH.read_bytes()
    payload = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    payload["renderer_contracts"]["max_rendered_input_tokens"] = 256
    path = tmp_path / "manifest.json"
    path.write_bytes(_canonical_manifest_bytes(payload))
    with pytest.raises(ValueError, match="max_rendered_input_tokens mismatch"):
        validate_v02_distillation_safe_corpus(path)


def test_main_validation_passes() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "benchmarks.validate_manifests"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_plan_training_pilot_prefix_details() -> None:
    plan = v02_corpus._generate_training_plan()
    pilot = plan["pilot_prefix"]
    assert pilot["count"] == 140
    assert pilot["train_count"] == 119
    assert pilot["internal_dev_count"] == 21
    assert pilot["complete_bilingual_pairs"] == 21
    assert pilot["slots_per_locale_cardinality_cell"] == 10


def test_training_pairs_and_pilot_are_derived_from_slot_assignments() -> None:
    plan = v02_corpus._generate_training_plan()
    slots = plan["slots"]
    pairs: dict[str, list[dict[str, object]]] = {}
    for slot in slots:
        pair_id = slot["bilingual_pair_id"]
        if pair_id is not None:
            pairs.setdefault(pair_id, []).append(slot)
    assert len(pairs) == 300
    assert sum(pair_id.startswith("bp-train-") for pair_id in pairs) == 255
    assert sum(pair_id.startswith("bp-internal_dev-") for pair_id in pairs) == 45
    for members in pairs.values():
        assert len(members) == 2
        assert {member["locale"] for member in members} == {"pt_br", "english"}
        assert len({member["split"] for member in members}) == 1
        assert len({member["domain"] for member in members}) == 1
        assert len({member["option_count"] for member in members}) == 1
        assert len({member["family_id"] for member in members}) == 1
        assert len({member["scenario_code"] for member in members}) == 1
        for member in members:
            roles = cast(list[str], member["criterion_roles"])
            gold_position = cast(int, member["gold_position"])
            assert roles[gold_position] == "matches_rule"
            assert len(roles) == member["option_count"]
    pilot = [slot for slot in slots if slot["in_pilot"]]
    assert len(pilot) == 140
    assert sum(slot["split"] == "train" for slot in pilot) == 119
    assert sum(slot["split"] == "internal_dev" for slot in pilot) == 21
    for locale in ("pt_br", "english"):
        for option_count in range(2, 9):
            assert (
                sum(
                    slot["locale"] == locale and slot["option_count"] == option_count
                    for slot in pilot
                )
                == 10
            )
    pilot_pair_ids = {slot["bilingual_pair_id"] for slot in pilot if slot["bilingual_pair_id"]}
    assert len(pilot_pair_ids) == 21
    assert all(all(member["in_pilot"] for member in pairs[pair_id]) for pair_id in pilot_pair_ids)
    for split in ("train", "internal_dev"):
        for locale in ("pt_br", "english"):
            cell_slots = [
                slot for slot in slots if slot["split"] == split and slot["locale"] == locale
            ]
            domain_counts = [
                sum(slot["domain"] == domain for slot in cell_slots)
                for domain in v02_corpus.DOMAINS
            ]
            assert max(domain_counts) - min(domain_counts) <= 1
            for option_count in range(2, 9):
                gold_counts = [
                    sum(
                        slot["option_count"] == option_count and slot["gold_position"] == position
                        for slot in cell_slots
                    )
                    for position in range(option_count)
                ]
                assert max(gold_counts) - min(gold_counts) <= 1


def test_training_taxonomy_is_derived_from_frozen_manifest() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    corpus = manifest["corpus_plan"]
    plan = v02_corpus._generate_training_plan()
    for slot in plan["slots"]:
        assert slot["domain"] in corpus["domains"]
        assert slot["scenario_code"] in corpus["domain_scenario_map"][slot["domain"]]
        assert set(slot["criterion_roles"]).issubset(set(corpus["criterion_roles"]))
        assert len(set(slot["criterion_roles"])) == slot["option_count"]


def test_sealed_identity_hides_semantic_target_fields() -> None:
    plan = v02_corpus._generate_sealed_plan()
    for identity in [*plan["identities"], *plan["pilot_prefix"]["identities"]]:
        assert set(identity) == {"identity_id", "locale", "option_count", "permutation_rank"}


def test_atomic_write_does_not_overwrite_racing_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "plan.json"
    original_link = os.link
    raced = False

    def race_link(source: str | Path, target: str | Path) -> None:
        nonlocal raced
        if Path(target) == destination and not raced:
            raced = True
            destination.write_bytes(b"racer")
        original_link(source, target)

    monkeypatch.setattr(os, "link", race_link)
    with pytest.raises(ValueError, match="already exists"):
        v02_corpus._atomic_write(destination, b"planned")
    assert destination.read_bytes() == b"racer"


def test_sealed_plan_no_content_generation() -> None:
    plan = v02_corpus._generate_sealed_plan()
    for ident in plan["identities"]:
        assert "content" not in ident
        assert "prompt" not in ident
        assert "text" not in ident


def test_sealed_pilot_is_stratified_three_per_count() -> None:
    plan = v02_corpus._generate_sealed_plan()
    pilot = plan["pilot_prefix"]
    for oc in [2, 3, 4, 5, 6, 7, 8]:
        count = sum(1 for i in pilot["identities"] if i["option_count"] == oc)
        assert count == 3, f"pilot option_count {oc} has {count}, expected 3"
    assert pilot["identities"] == plan["identities"][:21]


def test_training_slots_are_opaque(tmp_path: Path) -> None:
    output_parent = tmp_path / "output"
    _mkdir_0700(output_parent)
    path = v02_corpus.plan("training", output_parent)
    data = json.loads(path.read_text(encoding="utf-8"))
    for slot_id in data["slot_ids"]:
        assert len(slot_id) == 32
        assert all(c in "0123456789abcdef" for c in slot_id)


def test_sealed_identities_are_opaque(tmp_path: Path) -> None:
    output_parent = tmp_path / "output"
    _mkdir_0700(output_parent)
    path = v02_corpus.plan("sealed", output_parent)
    data = json.loads(path.read_text(encoding="utf-8"))
    for ident in data["identities"]:
        assert len(ident["identity_id"]) == 32
        assert all(c in "0123456789abcdef" for c in ident["identity_id"])


def test_manifest_writes_mode_0600(tmp_path: Path) -> None:
    output_parent = tmp_path / "output"
    _mkdir_0700(output_parent)
    path = v02_corpus.plan("training", output_parent)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_manifest_validate_protocol_cli() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "benchmarks.v02_corpus", "validate-protocol"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "protocol valid" in result.stdout


def test_plan_via_cli_training(tmp_path: Path) -> None:
    output_parent = tmp_path / "output"
    _mkdir_0700(output_parent)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.v02_corpus",
            "plan",
            "--lane",
            "training",
            "--output-parent",
            str(output_parent),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "plan written" in result.stdout


def test_plan_via_cli_sealed(tmp_path: Path) -> None:
    output_parent = tmp_path / "output"
    _mkdir_0700(output_parent)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.v02_corpus",
            "plan",
            "--lane",
            "sealed",
            "--output-parent",
            str(output_parent),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "plan written" in result.stdout


def test_sealed_pilot_contains_21_complete_identities(tmp_path: Path) -> None:
    output_parent = tmp_path / "output"
    _mkdir_0700(output_parent)
    path = v02_corpus.plan("sealed", output_parent)
    data = json.loads(path.read_text(encoding="utf-8"))
    pilot = data["pilot_prefix"]
    assert len(pilot["identities"]) == 21
    for ident in pilot["identities"]:
        assert ident["locale"] == "pt_br"


def test_plan_lane_validation(tmp_path: Path) -> None:
    output_parent = tmp_path / "output"
    _mkdir_0700(output_parent)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.v02_corpus",
            "plan",
            "--lane",
            "invalid",
            "--output-parent",
            str(output_parent),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0


def _all_gates() -> dict[str, bool]:
    return {gate: True for gate in v02_corpus.LOCAL_GATES}


def _training_envelope(
    plan: dict[str, object], role: str, answer: str | None = None, *, index: int = 0
) -> dict[str, object]:
    slot = cast(list[dict[str, object]], plan["slots"])[index]
    return {
        "schema_version": "v02-envelope.v1",
        "role": role,
        "model": v02_corpus._model_identity(role),
        "lane": "training",
        "identity_id": slot["slot_id"],
        "content_digest": "c" * 64,
        "answer": answer or f"option_{slot['gold_position']}",
        "semantic": {
            "scenario_code": slot["scenario_code"],
            "criterion_roles": slot["criterion_roles"],
        },
        "gates": _all_gates(),
    }


def _sealed_envelope(
    plan: dict[str, object], role: str, label: str = "option_0", *, index: int = 0
) -> dict[str, object]:
    identity = cast(list[dict[str, object]], plan["identities"])[index]
    payload: dict[str, object] = {
        "schema_version": "v02-envelope.v1",
        "role": role,
        "model": v02_corpus._model_identity(role),
        "lane": "sealed",
        "identity_id": identity["identity_id"],
        "content_digest": "c" * 64,
        "gates": _all_gates(),
    }
    payload[{"sealed_author": "target", "sealed_adjudicator": "choice"}.get(role, "label")] = label
    return payload


def test_training_ledger_is_closed_create_only_and_resumable(tmp_path: Path) -> None:
    root = tmp_path / "ledger"
    _mkdir_0700(root)
    plan = v02_corpus._generate_training_plan()
    ledger = v02_corpus.OfflineLedger(plan, root)
    author = _training_envelope(plan, "training_author")
    reviewer = _training_envelope(plan, "independent_reviewer")
    with pytest.raises(ValueError, match="out of order"):
        ledger.commit("training_reviewer", reviewer)
    assert ledger.commit("training_author", author) is True
    assert ledger.commit("training_author", author) is False
    changed = dict(author)
    changed["content_digest"] = "d" * 64
    with pytest.raises(ValueError, match="duplicate"):
        ledger.commit("training_author", changed)
    assert ledger.commit("training_reviewer", reviewer) is True
    metrics = v02_corpus.reduce_training(plan, ledger.events())
    assert metrics["accepted"] == 1
    assert metrics["denominator"] == 1600


def test_validators_reject_rewrites_unknowns_and_target_leakage() -> None:
    training = v02_corpus._generate_training_plan()
    reviewer = _training_envelope(training, "independent_reviewer")
    reviewer["unexpected"] = True
    with pytest.raises(ValueError, match="closed"):
        v02_corpus.validate_training_reviewer_decision(reviewer, training)
    sealed = v02_corpus._generate_sealed_plan()
    annotator = _sealed_envelope(sealed, "sealed_annotator_a")
    annotator["target"] = "A"
    with pytest.raises(ValueError, match="closed"):
        v02_corpus.validate_blind_annotator_decision(annotator, sealed)


def test_sealed_ledger_requires_two_blind_labels_before_adjudication(tmp_path: Path) -> None:
    root = tmp_path / "sealed"
    _mkdir_0700(root)
    plan = v02_corpus._generate_sealed_plan()
    ledger = v02_corpus.OfflineLedger(plan, root)
    author = _sealed_envelope(plan, "sealed_author")
    label_a = _sealed_envelope(plan, "sealed_annotator_a", "option_0")
    label_b = _sealed_envelope(plan, "sealed_annotator_b", "option_1")
    adjudicator = _sealed_envelope(plan, "sealed_adjudicator", "option_0")
    ledger.commit("sealed_author", author)
    ledger.commit("sealed_annotator_a", label_a)
    with pytest.raises(ValueError, match="out of order"):
        ledger.commit("sealed_adjudicator", adjudicator)
    ledger.commit("sealed_annotator_b", label_b)
    ledger.commit("sealed_adjudicator", adjudicator)
    metrics = v02_corpus.reduce_sealed(plan, ledger.events())
    assert metrics["accepted"] == 1
    assert metrics["adjudications"] == 1


def test_receipt_is_aggregate_only_and_verifiable(tmp_path: Path) -> None:
    root = tmp_path / "receipt"
    _mkdir_0700(root)
    plan = v02_corpus._generate_training_plan()
    metrics = v02_corpus.reduce_training(plan, [])
    receipt = v02_corpus.create_aggregate_receipt(
        root,
        artifact_id="training-capsule",
        plan_data=plan,
        events=[],
        metrics=metrics,
        status="PENDING",
        ancestry=v02_corpus._root_ancestry(root, "training"),
    )
    data = receipt.read_text(encoding="utf-8")
    assert "answer" not in data and str(root) not in data
    assert stat.S_IMODE(receipt.stat().st_mode) == 0o600
    verified = v02_corpus.verify_receipt(
        receipt,
        plan_data=plan,
        events=[],
        expected_lane="training",
        other_root_binding="b" * 64,
    )
    assert verified["status"] == "PENDING"


def test_training_pilot_has_exact_pass_and_impossibility_math() -> None:
    plan = v02_corpus._generate_training_plan()
    slots = [
        slot for slot in plan["slots"] if slot["slot_id"] in set(plan["pilot_prefix"]["slot_ids"])
    ]
    identifiers = {cast(str, slot["slot_id"]) for slot in slots}
    passing = v02_corpus._training_pilot_metrics(slots, identifiers, identifiers)
    assert passing["denominator"] == 140
    assert passing["status"] == "PASS"
    failed = v02_corpus._training_pilot_metrics(slots, set(), identifiers)
    assert failed["status"] == "NO_GO"


def test_sealed_pilot_has_exact_pass_and_impossibility_math() -> None:
    plan = v02_corpus._generate_sealed_plan()
    identities = cast(list[dict[str, object]], plan["identities"])
    pilot_ids = {
        cast(str, identity["identity_id"])
        for identity in cast(list[dict[str, object]], plan["pilot_prefix"]["identities"])
    }
    passed = v02_corpus._sealed_metrics(
        identities, pilot_ids, pilot_ids, pilot_ids, pilot_ids, set(), 15, 2, 14, 7
    )
    assert passed["denominator"] == 21
    assert passed["status"] == "PASS"
    failed = v02_corpus._sealed_metrics(
        identities, pilot_ids, set(), pilot_ids, set(), set(), 15, 2, 14, 7
    )
    assert failed["status"] == "NO_GO"


def _event(transition: str, envelope: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": v02_corpus.LEDGER_SCHEMA,
        "lane": envelope["lane"],
        "identity_id": envelope["identity_id"],
        "transition": transition,
        "envelope_digest": v02_corpus._sha256(canonical_json_bytes(cast(JsonValue, envelope))),
        "envelope": envelope,
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("answer", "unplanned"),
        ("semantic", "attested"),
        ("content_digest", "invalid"),
        ("model", "other@revision"),
    ],
)
def test_training_envelope_rejects_invalid_values(field: str, value: str) -> None:
    plan = v02_corpus._generate_training_plan()
    envelope = _training_envelope(plan, "training_author")
    envelope[field] = value
    with pytest.raises(ValueError):
        v02_corpus.validate_training_author_output(envelope, plan)


def test_blind_binding_excludes_author_targets_and_semantic_attestation() -> None:
    for plan, role in (
        (v02_corpus._generate_training_plan(), "training_author"),
        (v02_corpus._generate_sealed_plan(), "sealed_author"),
    ):
        envelope = (
            _training_envelope(plan, role)
            if role == "training_author"
            else _sealed_envelope(plan, role)
        )
        assert set(v02_corpus.blind_review_binding(envelope, plan)) == {
            "lane",
            "identity_id",
            "content_digest",
        }


@pytest.mark.parametrize(
    "mutation", ["identity", "digest", "missing_author", "duplicate", "unplanned"]
)
def test_reducer_revalidates_the_full_history(mutation: str) -> None:
    plan = v02_corpus._generate_training_plan()
    author = _event("training_author", _training_envelope(plan, "training_author"))
    reviewer = _event("training_reviewer", _training_envelope(plan, "independent_reviewer"))
    events = [author, reviewer]
    if mutation == "identity":
        author["identity_id"] = plan["slots"][1]["slot_id"]
    elif mutation == "digest":
        author["envelope_digest"] = "0" * 64
    elif mutation == "missing_author":
        events = [reviewer]
    elif mutation == "duplicate":
        events.append(author)
    else:
        cast(dict[str, object], author["envelope"])["identity_id"] = "unknown"
    with pytest.raises(ValueError):
        v02_corpus.reduce_training(plan, events)


def test_reviewer_must_bind_the_exact_authored_content(tmp_path: Path) -> None:
    _mkdir_0700(tmp_path)
    plan = v02_corpus._generate_training_plan()
    ledger = v02_corpus.OfflineLedger(plan, tmp_path)
    ledger.commit("training_author", _training_envelope(plan, "training_author"))
    reviewer = _training_envelope(plan, "independent_reviewer")
    reviewer["content_digest"] = "e" * 64
    with pytest.raises(ValueError, match="content digest"):
        ledger.commit("training_reviewer", reviewer)


def test_model_failure_settles_identity_and_preserves_denominator(tmp_path: Path) -> None:
    _mkdir_0700(tmp_path)
    plan = v02_corpus._generate_training_plan()
    ledger = v02_corpus.OfflineLedger(plan, tmp_path)
    failure = _training_envelope(plan, "training_author")
    for key in ("answer", "semantic", "gates", "content_digest"):
        failure.pop(key)
    failure["error_code"] = "invalid_output"
    ledger.commit("training_author_failure", failure)
    assert ledger.commit("training_author_failure", failure) is False
    metrics = v02_corpus.reduce_training(plan, ledger.events())
    assert (metrics["accepted"], metrics["resolved"], metrics["denominator"]) == (0, 1, 1600)
    with pytest.raises(ValueError, match="settled"):
        ledger.commit("training_author", _training_envelope(plan, "training_author"))


def test_resume_rejects_renamed_or_symlinked_events(tmp_path: Path) -> None:
    _mkdir_0700(tmp_path)
    plan = v02_corpus._generate_training_plan()
    ledger = v02_corpus.OfflineLedger(plan, tmp_path)
    ledger.commit("training_author", _training_envelope(plan, "training_author"))
    original = next(ledger.events_root.glob("*.json"))
    renamed = original.with_name("renamed.json")
    original.rename(renamed)
    with pytest.raises(ValueError, match="filename"):
        ledger.events()
    renamed.rename(original)
    original.with_name("link.json").symlink_to(original)
    with pytest.raises(ValueError, match="regular"):
        ledger.events()


def test_lanes_cannot_share_a_root_or_nested_ancestry(tmp_path: Path) -> None:
    _mkdir_0700(tmp_path)
    training = v02_corpus.OfflineLedger(v02_corpus._generate_training_plan(), tmp_path)
    with pytest.raises(ValueError, match="different plan or lane"):
        v02_corpus.OfflineLedger(v02_corpus._generate_sealed_plan(), tmp_path)
    child = tmp_path / "sealed-child"
    _mkdir_0700(child)
    with pytest.raises(ValueError, match="separation"):
        v02_corpus.OfflineLedger(
            v02_corpus._generate_sealed_plan(), child, other_ancestry=training.ancestry
        )


def test_pilot_must_finish_before_full_lane_commits(tmp_path: Path) -> None:
    _mkdir_0700(tmp_path)
    plan = v02_corpus._generate_training_plan()
    index = next(index for index, slot in enumerate(plan["slots"]) if not slot["in_pilot"])
    ledger = v02_corpus.OfflineLedger(plan, tmp_path)
    with pytest.raises(ValueError, match="pilot"):
        ledger.commit("training_author", _training_envelope(plan, "training_author", index=index))


def test_terminal_no_go_rejects_further_commits(tmp_path: Path) -> None:
    _mkdir_0700(tmp_path)
    plan = v02_corpus._generate_training_plan()
    ledger = v02_corpus.OfflineLedger(plan, tmp_path)
    indices = [
        index
        for index, slot in enumerate(plan["slots"])
        if slot["in_pilot"] and slot["option_count"] == 2
    ]
    for index in indices[:6]:
        envelope = _training_envelope(plan, "training_author", index=index)
        for key in ("answer", "semantic", "gates", "content_digest"):
            envelope.pop(key)
        envelope["error_code"] = "model_error"
        ledger.commit("training_author_failure", envelope)
    assert v02_corpus.reduce_training(plan, ledger.events())["status"] == "NO_GO"
    with pytest.raises(ValueError, match="terminal"):
        ledger.commit(
            "training_author", _training_envelope(plan, "training_author", index=indices[6])
        )


def test_training_language_floors_use_final_accepted_denominator() -> None:
    plan = v02_corpus._generate_training_plan()
    slots = plan["slots"]
    all_ids = {slot["slot_id"] for slot in slots}
    accepted = {
        slot["slot_id"]
        for slot in slots
        if slot["locale"] == "english" or slot["locale"] == "pt_br"
    }
    assert v02_corpus._training_full_status(slots, accepted, all_ids) == "READY"
    # All English rows plus 900 Portuguese rows: 900/1540 < 60%, despite exceeding 720.
    pt_ids = [slot["slot_id"] for slot in slots if slot["locale"] == "pt_br"]
    accepted.difference_update(pt_ids[900:])
    assert v02_corpus._training_full_status(slots, accepted, all_ids) == "NO_GO"


def test_sealed_agreement_on_wrong_target_and_outside_adjudication_reject() -> None:
    plan = v02_corpus._generate_sealed_plan()
    events = [
        _event("sealed_author", _sealed_envelope(plan, "sealed_author")),
        _event("sealed_annotator_a", _sealed_envelope(plan, "sealed_annotator_a", "option_1")),
        _event("sealed_annotator_b", _sealed_envelope(plan, "sealed_annotator_b", "option_1")),
    ]
    assert v02_corpus.reduce_sealed(plan, events)["accepted"] == 0
    events.append(_event("sealed_adjudicator", _sealed_envelope(plan, "sealed_adjudicator")))
    with pytest.raises(ValueError, match="agreement"):
        v02_corpus.reduce_sealed(plan, events)


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "READY"),
        ("counts", {"accepted": 1, "resolved": 1, "unresolved": 1599, "denominator": 1600}),
        ("cohort_metrics", {"invented": 1}),
        ("artifact_id", "../escape"),
    ],
)
def test_receipt_rejects_forged_valid_looking_metrics(
    tmp_path: Path, field: str, value: object
) -> None:
    _mkdir_0700(tmp_path)
    plan = v02_corpus._generate_training_plan()
    metrics = v02_corpus.reduce_training(plan, [])
    path = v02_corpus.create_aggregate_receipt(
        tmp_path,
        artifact_id="training",
        plan_data=plan,
        events=[],
        metrics=metrics,
        status="PENDING",
        ancestry=v02_corpus._root_ancestry(tmp_path, "training"),
    )
    receipt = json.loads(path.read_bytes())
    receipt[field] = value
    path.write_bytes(canonical_json_bytes(receipt) + b"\n")
    with pytest.raises(ValueError):
        v02_corpus.verify_receipt(path, plan_data=plan, events=[], expected_lane="training")


def test_verify_receipt_cli_is_content_free(tmp_path: Path) -> None:
    _mkdir_0700(tmp_path)
    v02_corpus.plan("training", tmp_path)
    plan = v02_corpus._generate_training_plan()
    ledger = v02_corpus.OfflineLedger(plan, tmp_path)
    v02_corpus.create_aggregate_receipt(
        tmp_path,
        artifact_id="training",
        plan_data=plan,
        events=[],
        metrics=v02_corpus.reduce_training(plan, []),
        status="PENDING",
        ancestry=ledger.ancestry,
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.v02_corpus",
            "verify-receipt",
            "--artifact-root",
            str(tmp_path),
            "--lane",
            "training",
            "--receipt-id",
            "training",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "receipt verified\n"
    assert str(tmp_path) not in result.stdout + result.stderr


def test_training_pilot_exact_threshold_and_one_more_failure() -> None:
    plan = v02_corpus._generate_training_plan()
    slots = [slot for slot in plan["slots"] if slot["in_pilot"]]
    resolved = {slot["slot_id"] for slot in slots}
    rejected: set[str] = set()
    for locale in ("pt_br", "english"):
        extras = {2, 3, 4} if locale == "pt_br" else {5, 6, 7}
        for count in range(2, 9):
            bucket = [
                slot for slot in slots if slot["locale"] == locale and slot["option_count"] == count
            ]
            rejected.update(slot["slot_id"] for slot in bucket[: 2 + (count in extras)])
    accepted = resolved - rejected
    metric = v02_corpus._training_pilot_metrics(slots, accepted, resolved)
    assert metric["accepted"] == 106
    assert metric["locale_accepted"] == {"pt_br": 53, "english": 53}
    assert metric["status"] == "PASS"
    accepted.remove(
        next(
            slot["slot_id"]
            for slot in slots
            if slot["locale"] == "pt_br" and slot["slot_id"] in accepted
        )
    )
    assert v02_corpus._training_pilot_metrics(slots, accepted, resolved)["status"] == "NO_GO"


def test_sealed_pilot_exact_threshold_and_direct_agreement_floor() -> None:
    plan = v02_corpus._generate_sealed_plan()
    identities = plan["identities"]
    selected = {item["identity_id"] for item in plan["pilot_prefix"]["identities"]}
    accepted: set[str] = set()
    for count in range(2, 9):
        bucket = [
            item
            for item in identities
            if item["identity_id"] in selected and item["option_count"] == count
        ]
        accepted.update(item["identity_id"] for item in bucket[: 3 if count == 2 else 2])
    adjudicated = {next(iter(accepted))}
    direct = accepted - adjudicated
    assert (
        v02_corpus._sealed_metrics(
            identities, selected, accepted, selected, direct, adjudicated, 15, 2, 14, 7
        )["status"]
        == "PASS"
    )
    direct.pop()
    assert (
        v02_corpus._sealed_metrics(
            identities, selected, accepted, selected, direct, adjudicated, 15, 2, 14, 7
        )["status"]
        == "NO_GO"
    )


def test_same_event_race_resumes_without_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mkdir_0700(tmp_path)
    plan = v02_corpus._generate_training_plan()
    ledger = v02_corpus.OfflineLedger(plan, tmp_path)
    original = v02_corpus._atomic_write

    def competing_write(path: Path, data: bytes) -> None:
        original(path, data)
        raise ValueError("another writer published identical bytes")

    monkeypatch.setattr(v02_corpus, "_atomic_write", competing_write)
    assert ledger.commit("training_author", _training_envelope(plan, "training_author")) is False
    assert len(ledger.events()) == 1


def test_frozen_plan_mutation_is_rejected(tmp_path: Path) -> None:
    _mkdir_0700(tmp_path)
    plan = v02_corpus._generate_training_plan()
    plan["slots"][0]["locale"] = "changed"
    with pytest.raises(ValueError, match="frozen plan"):
        v02_corpus.OfflineLedger(plan, tmp_path)
