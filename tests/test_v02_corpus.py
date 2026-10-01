"""Tests for the v0.2 distillation-safe corpus offline planner."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import urllib.request
from collections.abc import Mapping
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import pytest
from pydantic import JsonValue

import benchmarks.v02_corpus as v02_corpus
from benchmarks.v02_evaluation import combined_content_fingerprint, state_question_fingerprint
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


@pytest.mark.parametrize("timed_out", [False, True])
def test_default_transport_uses_model_timeout_without_retry(
    monkeypatch: pytest.MonkeyPatch, timed_out: bool
) -> None:
    calls: list[tuple[Any, int]] = []

    class Response:
        status = 200

        def __enter__(self) -> Response:
            return self

        def __exit__(self, *args: Any) -> None:
            pass

        def read(self, size: int) -> bytes:
            assert size == v02_corpus._RESPONSE_LIMIT + 1
            return b"{}"

    def open_request(request: Any, *, timeout: int) -> Response:
        calls.append((request, timeout))
        if timed_out:
            raise TimeoutError
        return Response()

    monkeypatch.setattr(
        urllib.request,
        "build_opener",
        lambda *args: SimpleNamespace(open=open_request),
    )
    request = {
        "url": "http://127.0.0.1:8000/v1/chat/completions",
        "headers": {},
        "method": "POST",
        "body": b"{}",
    }
    if timed_out:
        with pytest.raises(v02_corpus._ModelTransportError, match="timeout"):
            v02_corpus._default_transport(request)
    else:
        assert v02_corpus._default_transport(request) == {"status": 200, "body": b"{}"}
    assert len(calls) == 1
    assert calls[0][1] == 180
    assert calls[0][0].get_method() == "POST"
    assert calls[0][0].data == b"{}"


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


def test_c7a_manifest_requires_the_closed_grounding_micro_pilot(tmp_path: Path) -> None:
    payload = json.loads(MANIFEST_PATH.read_bytes(), object_pairs_hook=_reject_duplicate_keys)
    assert payload["grounding_micro_pilot"] == v02_corpus.GROUNDING_MICRO_PILOT
    for field, value in (("training_slots", 27), ("unexpected", True)):
        changed = json.loads(json.dumps(payload))
        changed["grounding_micro_pilot"][field] = value
        path = tmp_path / f"micro-{field}.json"
        path.write_bytes(_canonical_manifest_bytes(changed))
        with pytest.raises(ValueError, match="grounding_micro_pilot"):
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
    assert data["namespace"] == "saracura-v02-text-contract-author-v1"
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


@pytest.mark.parametrize(
    "synchronized_ancestor",
    [
        "CloudStorage",
        "Mobile Documents",
        "CloudDocs",
        "com~apple~CloudDocs",
        "Dropbox",
        "OneDrive",
        "Google Drive",
        "Nextcloud",
        "felhencloud",
    ],
)
def test_private_storage_rejects_known_synchronized_ancestry(
    tmp_path: Path, synchronized_ancestor: str
) -> None:
    output_parent = tmp_path / synchronized_ancestor / "private-output"
    _mkdir_0700(output_parent)
    with pytest.raises(ValueError, match="synchronized ancestry"):
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
    selected = answer or f"option_{slot['gold_position']}"
    payload: dict[str, object] = {
        "schema_version": "v02-envelope.v2" if role == "training_author" else "v02-envelope.v1",
        "role": role,
        "model": v02_corpus._model_identity(role),
        "lane": "training",
        "identity_id": slot["slot_id"],
        "content_digest": "c" * 64,
        "answer": selected,
        "semantic": {
            "scenario_code": slot["scenario_code"],
            "criterion_roles": slot["criterion_roles"],
        },
        "gates": _all_gates(),
    }
    if role == "training_author":
        payload["construction"] = {
            "rule_quote": "Fictional policy",
            "option_checks": [
                {
                    "option_id": f"option_{index}",
                    "supported": f"option_{index}" == selected,
                    "reason": "Fictional policy check",
                }
                for index in range(cast(int, slot["option_count"]))
            ],
        }
    return payload


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


def test_c6a_reviewer_repeated_roles_are_valid_but_closed_shape_and_vocab_remain_required() -> None:
    training = v02_corpus._generate_training_plan()
    reviewer = _training_envelope(training, "independent_reviewer")
    semantic = cast(dict[str, object], reviewer["semantic"])
    roles = cast(list[str], semantic["criterion_roles"])
    semantic["criterion_roles"] = [roles[0]] * len(roles)
    v02_corpus.validate_training_reviewer_decision(reviewer, training)
    semantic["criterion_roles"] = ["unknown"] * len(roles)
    with pytest.raises(ValueError, match="closed vocabulary"):
        v02_corpus.validate_training_reviewer_decision(reviewer, training)
    semantic["criterion_roles"] = [roles[0]] * (len(roles) - 1)
    with pytest.raises(ValueError, match="closed vocabulary"):
        v02_corpus.validate_training_reviewer_decision(reviewer, training)


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
    assert verified["schema_version"] == v02_corpus.TRAINING_RECEIPT_SCHEMA
    assert verified["training_acceptance_policy"] == v02_corpus.TRAINING_ACCEPTANCE_POLICY
    assert metrics["training_acceptance_policy"] == v02_corpus.TRAINING_ACCEPTANCE_POLICY


def test_c6a_training_receipt_requires_current_policy_and_closed_diagnostics(
    tmp_path: Path,
) -> None:
    root = tmp_path / "training"
    _mkdir_0700(root)
    plan = v02_corpus._generate_training_plan()
    metrics = v02_corpus.reduce_training(plan, [])
    receipt_path = v02_corpus.create_aggregate_receipt(
        root,
        artifact_id="training",
        plan_data=plan,
        events=[],
        metrics=metrics,
        status="PENDING",
        ancestry=v02_corpus._root_ancestry(root, "training"),
    )
    receipt = json.loads(receipt_path.read_bytes())
    for field, value in (
        ("schema_version", v02_corpus.RECEIPT_SCHEMA),
        ("training_acceptance_policy", "wrong-policy"),
    ):
        changed = {**receipt, field: value}
        receipt_path.write_bytes(canonical_json_bytes(changed) + b"\n")
        with pytest.raises(ValueError, match="historical policy/source required"):
            v02_corpus.verify_receipt(
                receipt_path, plan_data=plan, events=[], expected_lane="training"
            )
    historical_v1 = {**receipt, "schema_version": v02_corpus.RECEIPT_SCHEMA}
    historical_v1.pop("training_acceptance_policy")
    historical_v1.pop("auxiliary_semantic_diagnostics")
    historical_v1.pop("grounding_micro_pilot_metrics")
    receipt_path.write_bytes(canonical_json_bytes(historical_v1) + b"\n")
    with pytest.raises(ValueError, match="historical policy/source required"):
        v02_corpus.verify_receipt(receipt_path, plan_data=plan, events=[], expected_lane="training")
    changed = {**receipt, "auxiliary_semantic_diagnostics": {"reviewed_rows": True}}
    receipt_path.write_bytes(canonical_json_bytes(changed) + b"\n")
    with pytest.raises(ValueError, match="auxiliary semantic diagnostics"):
        v02_corpus.verify_receipt(receipt_path, plan_data=plan, events=[], expected_lane="training")


def _commit_micro_pass(ledger: v02_corpus.OfflineLedger) -> list[int]:
    """Commit the self-authored valid micro cohort without bypassing admission."""
    plan = ledger.plan
    slots = cast(list[dict[str, Any]], plan["slots"])
    selected = set(v02_corpus.grounding_micro_pilot(plan))
    indices = [index for index, slot in enumerate(slots) if slot["slot_id"] in selected]
    for index in indices:
        slot = slots[index]
        case = _case(cast(int, slot["option_count"]))
        digest = v02_corpus.case_digest(case)
        author = _training_envelope(plan, "training_author", index=index)
        reviewer = _training_envelope(plan, "independent_reviewer", index=index)
        author["content_digest"] = digest
        cast(dict[str, Any], author["construction"])["rule_quote"] = "Fictional"
        reviewer["content_digest"] = digest
        v02_corpus._store_private_case(ledger.root, "training", cast(str, slot["slot_id"]), case)
        assert ledger.commit("training_author", author)
    for index in indices:
        slot = slots[index]
        case = _case(cast(int, slot["option_count"]))
        reviewer = _training_envelope(plan, "independent_reviewer", index=index)
        reviewer["content_digest"] = v02_corpus.case_digest(case)
        assert ledger.commit("training_reviewer", reviewer)
    assert v02_corpus.reduce_grounding_micro_pilot(plan, ledger.events())["status"] == "PASS"
    return indices


def test_c7a_ledger_rebinds_author_construction_when_private_case_is_available(
    tmp_path: Path,
) -> None:
    _mkdir_0700(tmp_path)
    plan = v02_corpus._generate_training_plan()
    slot = cast(list[dict[str, Any]], plan["slots"])[0]
    identity_id = cast(str, slot["slot_id"])
    ledger = v02_corpus.OfflineLedger(plan, tmp_path)
    case = _case(cast(int, slot["option_count"]), state="Stored policy permits fictional action")
    author = _training_envelope(plan, "training_author")
    author["content_digest"] = v02_corpus.case_digest(case)
    construction = cast(dict[str, Any], author["construction"])
    construction["rule_quote"] = "Stored policy"
    v02_corpus._store_private_case(ledger.root, "training", identity_id, case)
    assert ledger.commit("training_author", author)
    assert ledger.events()

    event_path = next(ledger.events_root.glob("*.json"))
    event = json.loads(event_path.read_bytes())
    envelope = cast(dict[str, Any], event["envelope"])
    envelope["construction"]["rule_quote"] = "Absent policy"
    event["envelope_digest"] = v02_corpus._sha256(canonical_json_bytes(event["envelope"]))
    event_path.write_bytes(canonical_json_bytes(event) + b"\n")
    os.chmod(event_path, 0o600)
    assert v02_corpus.reduce_training(plan, [event])["resolved"] == 0
    with pytest.raises(ValueError, match="rule_quote must occur in case state"):
        ledger.events()

    envelope["construction"]["rule_quote"] = "Stored policy"
    envelope["content_digest"] = "d" * 64
    event["envelope_digest"] = v02_corpus._sha256(canonical_json_bytes(event["envelope"]))
    event_path.write_bytes(canonical_json_bytes(event) + b"\n")
    with pytest.raises(ValueError, match="private case digest mismatch"):
        ledger.events()

    envelope["content_digest"] = v02_corpus.case_digest(case)
    event["envelope_digest"] = v02_corpus._sha256(canonical_json_bytes(event["envelope"]))
    event_path.write_bytes(canonical_json_bytes(event) + b"\n")
    case_path = v02_corpus._case_path(ledger.root, "training", identity_id)
    case_path.unlink()
    case_path.symlink_to(tmp_path / "missing-private-case.json")
    with pytest.raises(ValueError, match="private case is closed"):
        ledger.events()


def test_c7a_author_construction_is_closed_and_uses_unicode_code_points() -> None:
    plan = v02_corpus._generate_training_plan()
    slot = cast(list[dict[str, Any]], plan["slots"])[0]
    state = "é" * 240
    response: dict[str, Any] = {
        **_case(cast(int, slot["option_count"]), state=state),
        "answer": f"option_{slot['gold_position']}",
        "construction": _construction(slot, state, f"option_{slot['gold_position']}"),
        "semantic": {
            "scenario_code": slot["scenario_code"],
            "criterion_roles": slot["criterion_roles"],
        },
        **_judgments(),
    }
    response["construction"]["rule_quote"] = state
    response["construction"]["option_checks"][0]["reason"] = "é" * 160
    escaped = json.dumps(response, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    assert v02_corpus.parse_role_response(escaped, plan, slot["slot_id"], "training_author")
    mutations: list[dict[str, Any]] = []
    missing = json.loads(json.dumps(response))
    missing.pop("construction")
    mutations.append(missing)
    missing_quote = json.loads(json.dumps(response))
    missing_quote["construction"].pop("rule_quote")
    mutations.append(missing_quote)
    extra = json.loads(json.dumps(response))
    extra["construction"]["extra"] = True
    mutations.append(extra)
    missing_check = json.loads(json.dumps(response))
    missing_check["construction"]["option_checks"].pop()
    mutations.append(missing_check)
    duplicate_check = json.loads(json.dumps(response))
    duplicate_check["construction"]["option_checks"].append(
        duplicate_check["construction"]["option_checks"][0]
    )
    mutations.append(duplicate_check)
    wrong_boolean = json.loads(json.dumps(response))
    wrong_boolean["construction"]["option_checks"][0]["supported"] = 1
    mutations.append(wrong_boolean)
    reordered = json.loads(json.dumps(response))
    reordered["construction"]["option_checks"].reverse()
    mutations.append(reordered)
    too_long = json.loads(json.dumps(response))
    too_long["construction"]["rule_quote"] = "é" * 241
    mutations.append(too_long)
    reason_too_long = json.loads(json.dumps(response))
    reason_too_long["construction"]["option_checks"][0]["reason"] = "é" * 161
    mutations.append(reason_too_long)
    out_of_state = json.loads(json.dumps(response))
    out_of_state["construction"]["rule_quote"] = "different policy"
    mutations.append(out_of_state)
    blank_quote = json.loads(json.dumps(response))
    blank_quote["construction"]["rule_quote"] = " "
    mutations.append(blank_quote)
    control_quote = json.loads(json.dumps(response))
    control_quote["construction"]["rule_quote"] = "\u0001"
    mutations.append(control_quote)
    non_nfc_quote = json.loads(json.dumps(response))
    non_nfc_quote["construction"]["rule_quote"] = "e\u0301"
    blank_reason = json.loads(json.dumps(response))
    blank_reason["construction"]["option_checks"][0]["reason"] = " "
    mutations.append(blank_reason)
    control_reason = json.loads(json.dumps(response))
    control_reason["construction"]["option_checks"][0]["reason"] = "\u0001"
    mutations.append(control_reason)
    non_nfc_reason = json.loads(json.dumps(response))
    non_nfc_reason["construction"]["option_checks"][0]["reason"] = "e\u0301"
    zero_supports = json.loads(json.dumps(response))
    for check in zero_supports["construction"]["option_checks"]:
        check["supported"] = False
    mutations.append(zero_supports)
    multiple_supports = json.loads(json.dumps(response))
    multiple_supports["construction"]["option_checks"][0]["supported"] = True
    multiple_supports["construction"]["option_checks"][1]["supported"] = True
    mutations.append(multiple_supports)
    support_answer_mismatch = json.loads(json.dumps(response))
    support_answer_mismatch["answer"] = (
        f"option_{(int(slot['gold_position']) + 1) % int(slot['option_count'])}"
    )
    mutations.append(support_answer_mismatch)
    for invalid in mutations:
        with pytest.raises(ValueError):
            v02_corpus.parse_role_response(_raw(invalid), plan, slot["slot_id"], "training_author")
    for invalid in (non_nfc_quote, non_nfc_reason):
        with pytest.raises(ValueError):
            v02_corpus.parse_role_response(
                json.dumps(invalid, ensure_ascii=True, separators=(",", ":")).encode("utf-8"),
                plan,
                slot["slot_id"],
                "training_author",
            )


def _training_author_text_schemas(schema: dict[str, Any]) -> dict[str, dict[str, Any]]:
    properties = cast(dict[str, Any], schema["properties"])
    options = cast(dict[str, Any], properties["options"])
    option_item = cast(
        dict[str, Any],
        options["items"] if options.get("items") is not False else options["prefixItems"][0],
    )
    construction = cast(dict[str, Any], properties["construction"]["properties"])
    checks = cast(dict[str, Any], construction["option_checks"])
    check_item = cast(
        dict[str, Any],
        checks["items"] if checks.get("items") is not False else checks["prefixItems"][0],
    )
    return {
        "state": cast(dict[str, Any], properties["state"]),
        "question": cast(dict[str, Any], properties["question"]),
        "description": cast(dict[str, Any], option_item["properties"]["description"]),
        "rule_quote": cast(dict[str, Any], construction["rule_quote"]),
        "reason": cast(dict[str, Any], check_item["properties"]["reason"]),
    }


def test_c8_training_author_patterns_restrict_only_the_generation_subset() -> None:
    plan = v02_corpus._generate_training_plan()
    slot = cast(list[dict[str, Any]], plan["slots"])[0]
    full = _training_author_text_schemas(v02_corpus._role_response_schema("training_author", slot))
    native = _training_author_text_schemas(
        v02_corpus.native_decoder_schema("training_author", slot)
    )
    unbounded_pattern = r'^[^\u0000-\u001F\u007F-\u009F"\\]+$'
    quote_pattern = r'^[^\u0000-\u001F\u007F-\u009F"\\]{1,240}$'
    reason_pattern = r'^[^\u0000-\u001F\u007F-\u009F"\\]{1,160}$'
    assert {name: schema["pattern"] for name, schema in full.items()} == {
        "state": unbounded_pattern,
        "question": unbounded_pattern,
        "description": unbounded_pattern,
        "rule_quote": quote_pattern,
        "reason": reason_pattern,
    }
    assert {name: schema["pattern"] for name, schema in native.items()} == {
        "state": unbounded_pattern,
        "question": unbounded_pattern,
        "description": unbounded_pattern,
        "rule_quote": quote_pattern,
        "reason": reason_pattern,
    }
    for schema in full.values():
        assert schema["minLength"] == 1
    assert full["rule_quote"]["maxLength"] == 240
    assert full["reason"]["maxLength"] == 160
    for schema in native.values():
        assert "minLength" not in schema and "maxLength" not in schema

    valid = json.loads(json.dumps("Decisão PT-BR and English l'option café 🧭"))
    forbidden = [chr(codepoint) for codepoint in (*range(0x00, 0x20), *range(0x7F, 0xA0))]
    for schema in (*full.values(), *native.values()):
        pattern = cast(str, schema["pattern"])
        assert re.fullmatch(pattern, valid)
        for character in [*forbidden, '"', "\\"]:
            assert re.fullmatch(pattern, json.loads(json.dumps(character))) is None
    assert re.fullmatch(quote_pattern, "é" * 240)
    assert re.fullmatch(quote_pattern, "é" * 241) is None
    assert re.fullmatch(reason_pattern, "é" * 160)
    assert re.fullmatch(reason_pattern, "é" * 161) is None


def test_c8_final_training_validator_still_accepts_quotes_and_backslashes() -> None:
    plan = v02_corpus._generate_training_plan()
    slot = cast(list[dict[str, Any]], plan["slots"])[0]
    answer = f"option_{slot['gold_position']}"
    state = 'Policy says "approve" only through \\review.'
    response: dict[str, Any] = {
        **_case(cast(int, slot["option_count"]), state=state),
        "answer": answer,
        "construction": _construction(slot, state, answer),
        "semantic": {
            "scenario_code": slot["scenario_code"],
            "criterion_roles": slot["criterion_roles"],
        },
        **_judgments(),
    }
    response["question"] = 'Which "action" follows \\review?'
    for index, option in enumerate(cast(list[dict[str, str]], response["options"])):
        option["description"] = f'Use "approved" \\review path {index}.'
    response["construction"]["rule_quote"] = state
    for check in cast(list[dict[str, Any]], response["construction"]["option_checks"]):
        check["reason"] = 'Checked "policy" at \\review.'
    assert v02_corpus.parse_role_response(
        _raw(cast(dict[str, object], response)),
        plan,
        slot["slot_id"],
        "training_author",
    )


def test_c7a_author_envelopes_require_v2_but_failure_v1_stays_valid(tmp_path: Path) -> None:
    _mkdir_0700(tmp_path)
    plan = v02_corpus._generate_training_plan()
    author = _training_envelope(plan, "training_author")
    author["schema_version"] = "v02-envelope.v1"
    author.pop("construction")
    with pytest.raises(ValueError, match="historical protocol/source required"):
        v02_corpus.validate_training_author_output(author, plan)
    failure = v02_corpus._failure_envelope(
        "training_author",
        cast(str, cast(list[dict[str, Any]], plan["slots"])[0]["slot_id"]),
        "invalid_output",
    )
    ledger = v02_corpus.OfflineLedger(plan, tmp_path)
    assert ledger.commit("training_author_failure", failure)


def test_c7a_proof_stays_outside_case_and_blind_role_contracts() -> None:
    plan = v02_corpus._generate_training_plan()
    slot = cast(list[dict[str, Any]], plan["slots"])[0]
    case = _case(cast(int, slot["option_count"]), state="Policy allows only fictional action")
    parsed: dict[str, Any] = {
        **case,
        "answer": f"option_{slot['gold_position']}",
        "construction": _construction(
            slot, cast(str, case["state"]), f"option_{slot['gold_position']}"
        ),
        "semantic": {
            "scenario_code": slot["scenario_code"],
            "criterion_roles": slot["criterion_roles"],
        },
        **_judgments(),
    }
    parsed["construction"]["rule_quote"] = "Policy allows only fictional action"
    assert v02_corpus.case_from_author(parsed, plan, cast(str, slot["slot_id"])) == case
    author = v02_corpus.role_envelope(
        parsed, plan, cast(str, slot["slot_id"]), "training_author", case, _all_gates()
    )
    assert "construction" not in v02_corpus.blind_review_binding(author, plan)
    reviewer_request = v02_corpus.role_request(
        plan, cast(str, slot["slot_id"]), "independent_reviewer", case
    )
    assert "construction" not in reviewer_request[1]["content"]
    sealed = v02_corpus._generate_sealed_plan()
    identity = cast(list[dict[str, Any]], sealed["identities"])[0]
    for role in ("sealed_author", "sealed_annotator_a", "sealed_annotator_b", "sealed_adjudicator"):
        schema = v02_corpus._role_response_schema(role, identity, ["option_0", "option_1"])
        assert "construction" not in schema["properties"]


def test_c7a_micro_selection_reducer_and_admission_are_terminal(tmp_path: Path) -> None:
    _mkdir_0700(tmp_path)
    plan = v02_corpus._generate_training_plan()
    selected = v02_corpus.grounding_micro_pilot(plan)
    selected_ids = set(selected)
    slots = cast(list[dict[str, Any]], plan["slots"])
    selected_rows = [slot for slot in slots if slot["slot_id"] in selected_ids]
    assert selected == [slot["slot_id"] for slot in slots if slot["slot_id"] in selected_ids]
    assert len(selected) == 28 and len(set(selected)) == 28
    assert all(
        slot["split"] == "train" and slot["bilingual_pair_id"] is None for slot in selected_rows
    )
    cell_counts = {
        (slot["locale"], slot["option_count"]): sum(
            other["locale"] == slot["locale"] and other["option_count"] == slot["option_count"]
            for other in selected_rows
        )
        for slot in selected_rows
    }
    expected_cell_counts = {
        (locale, count): 2 for locale in ("pt_br", "english") for count in v02_corpus.OPTION_COUNTS
    }
    assert cell_counts == expected_cell_counts
    assert v02_corpus.reduce_grounding_micro_pilot(plan, [])["status"] == "PENDING"
    selected_indices = [
        index for index, slot in enumerate(slots) if slot["slot_id"] in selected_ids
    ]
    failed_events = [
        _event(
            "training_author_failure",
            v02_corpus._failure_envelope(
                "training_author", cast(str, slots[index]["slot_id"]), "model_error"
            ),
        )
        for index in selected_indices[:2]
    ]
    assert v02_corpus.reduce_grounding_micro_pilot(plan, failed_events)["status"] == "NO_GO"
    rejected_cells = {("pt_br", 2), ("english", 3), ("pt_br", 4), ("english", 5)}
    rejected_ids = {
        next(
            cast(str, slot["slot_id"])
            for slot in selected_rows
            if (slot["locale"], slot["option_count"]) == cell
        )
        for cell in rejected_cells
    }
    passing_events: list[dict[str, Any]] = []
    for index in selected_indices:
        slot = slots[index]
        if slot["slot_id"] in rejected_ids:
            passing_events.append(
                _event(
                    "training_author_failure",
                    v02_corpus._failure_envelope(
                        "training_author", cast(str, slot["slot_id"]), "invalid_output"
                    ),
                )
            )
            continue
        passing_events.extend(
            [
                _event("training_author", _training_envelope(plan, "training_author", index=index)),
                _event(
                    "training_reviewer",
                    _training_envelope(plan, "independent_reviewer", index=index),
                ),
            ]
        )
    passing = v02_corpus.reduce_grounding_micro_pilot(plan, passing_events)
    assert passing["status"] == "PASS"
    assert passing["accepted"] == 24 and passing["resolved"] == 28
    assert passing["locale_accepted"] == {"pt_br": 12, "english": 12}
    assert all(value >= 3 for value in passing["option_count_accepted"].values())
    nonselected_index = next(
        index for index, slot in enumerate(slots) if slot["slot_id"] not in selected_ids
    )
    passing_events.extend(
        [
            _event(
                "training_author",
                _training_envelope(plan, "training_author", index=nonselected_index),
            ),
            _event(
                "training_reviewer",
                _training_envelope(plan, "independent_reviewer", index=nonselected_index),
            ),
        ]
    )
    assert v02_corpus.reduce_grounding_micro_pilot(plan, passing_events) == passing
    ledger = v02_corpus.OfflineLedger(plan, tmp_path)
    non_micro_candidates = (
        index for index, slot in enumerate(slots) if slot["slot_id"] not in selected_ids
    )
    non_micro = next(non_micro_candidates)
    with pytest.raises(ValueError, match="grounding micro"):
        ledger.commit(
            "training_author", _training_envelope(plan, "training_author", index=non_micro)
        )
    assert ledger.events() == []
    _commit_micro_pass(ledger)
    assert ledger.commit(
        "training_author", _training_envelope(plan, "training_author", index=non_micro)
    )
    reduced = v02_corpus.reduce_training(plan, ledger.events())
    assert reduced["grounding_micro_pilot"]["accepted"] == 28


def test_c7a_micro_receipt_v3_roundtrip_and_tampering(tmp_path: Path) -> None:
    _mkdir_0700(tmp_path)
    plan = v02_corpus._generate_training_plan()
    metrics = v02_corpus.reduce_training(plan, [])
    receipt_path = v02_corpus.create_aggregate_receipt(
        tmp_path,
        artifact_id="c7a-training",
        plan_data=plan,
        events=[],
        metrics=metrics,
        status="PENDING",
        ancestry=v02_corpus._root_ancestry(tmp_path, "training"),
    )
    receipt = json.loads(receipt_path.read_bytes())
    assert receipt["schema_version"] == "v02-aggregate-receipt.v3"
    assert receipt["grounding_micro_pilot_metrics"] == metrics["grounding_micro_pilot"]
    v02_corpus.verify_receipt(receipt_path, plan_data=plan, events=[], expected_lane="training")
    for field, value in (
        ("accepted", 1),
        ("resolved", 1),
        ("unresolved", 27),
        ("denominator", 27),
        ("locale_accepted", {"pt_br": 0, "english": 1}),
        (
            "option_count_accepted",
            {str(count): 1 if count == 2 else 0 for count in v02_corpus.OPTION_COUNTS},
        ),
        ("status", "PASS"),
    ):
        changed = json.loads(json.dumps(receipt))
        changed["grounding_micro_pilot_metrics"][field] = value
        receipt_path.write_bytes(canonical_json_bytes(changed) + b"\n")
        with pytest.raises(ValueError):
            v02_corpus.verify_receipt(
                receipt_path, plan_data=plan, events=[], expected_lane="training"
            )
    historical = json.loads(json.dumps(receipt))
    historical["schema_version"] = "v02-aggregate-receipt.v2"
    receipt_path.write_bytes(canonical_json_bytes(historical) + b"\n")
    with pytest.raises(ValueError, match="historical policy/source required"):
        v02_corpus.verify_receipt(receipt_path, plan_data=plan, events=[], expected_lane="training")


def test_c7a_micro_no_go_blocks_commit_dispatch_and_receipt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _mkdir_0700(tmp_path)
    plan = v02_corpus._generate_training_plan()
    slots = cast(list[dict[str, Any]], plan["slots"])
    micro_ids = set(v02_corpus.grounding_micro_pilot(plan))
    same_n_indices = [
        index
        for index, slot in enumerate(slots)
        if slot["slot_id"] in micro_ids and slot["option_count"] == 2
    ]
    assert len(same_n_indices) == 4
    ledger = v02_corpus.OfflineLedger(plan, tmp_path)
    for index in same_n_indices[:2]:
        assert ledger.commit(
            "training_author_failure",
            v02_corpus._failure_envelope(
                "training_author", cast(str, slots[index]["slot_id"]), "model_error"
            ),
        )
    metrics = v02_corpus.reduce_training(plan, ledger.events())
    assert metrics["grounding_micro_pilot"]["status"] == "NO_GO"
    assert metrics["pilot"]["status"] == "PENDING"
    assert metrics["status"] == "NO_GO"
    receipt_path = v02_corpus.create_aggregate_receipt(
        tmp_path,
        artifact_id="c7a-micro-no-go",
        plan_data=plan,
        events=ledger.events(),
        metrics=metrics,
        status="NO_GO",
        ancestry=v02_corpus._root_ancestry(tmp_path, "training"),
    )
    verified = v02_corpus.verify_receipt(
        receipt_path, plan_data=plan, events=ledger.events(), expected_lane="training"
    )
    assert verified["schema_version"] == v02_corpus.TRAINING_RECEIPT_SCHEMA
    assert verified["status"] == "NO_GO"
    later_index = same_n_indices[2]
    later_id = cast(str, slots[later_index]["slot_id"])
    with pytest.raises(ValueError, match="terminal"):
        ledger.commit(
            "training_author_failure",
            v02_corpus._failure_envelope("training_author", later_id, "model_error"),
        )
    renderer, tokenizer, inventory = _load_runtime(monkeypatch, tmp_path / "runtime")
    grant = _grant(inventory)
    transport = _Loopback(v02_corpus.MODEL_ROLES["training_author"]["model"], "{}")
    with pytest.raises(ValueError, match="terminal"):
        _run_role(
            ledger,
            later_id,
            "training_author",
            grant,
            _license_bytes(),
            renderer,
            tokenizer,
            transport,
        )
    assert transport.calls == []
    assert not (tmp_path / "role-reservations").exists()


def test_c8_frozen_non_author_contracts_and_allocations_match_source643(tmp_path: Path) -> None:
    baseline_root = tmp_path / "baseline" / "benchmarks"
    (baseline_root / "manifests").mkdir(parents=True)
    for relative in (
        "benchmarks/v02_corpus.py",
        "benchmarks/manifests/v02-distillation-safe-corpus.v1.json",
    ):
        result = subprocess.run(
            ["git", "show", f"643e52e:{relative}"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
        )
        target = tmp_path / "baseline" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(result.stdout)
    spec = spec_from_file_location(
        "v02_corpus_source643", tmp_path / "baseline/benchmarks/v02_corpus.py"
    )
    assert spec is not None and spec.loader is not None
    baseline = module_from_spec(spec)
    spec.loader.exec_module(baseline)
    current_plan = v02_corpus._generate_sealed_plan()
    current_training = v02_corpus._generate_training_plan()
    baseline_plan = baseline._generate_sealed_plan()
    baseline_training = baseline._generate_training_plan()
    assert current_plan == baseline_plan
    assert v02_corpus._ROLE_SYSTEM_TEMPLATE == baseline._ROLE_SYSTEM_TEMPLATE
    assert v02_corpus.ROLE_TEMPLATES["training_author"] == (
        baseline.ROLE_TEMPLATES["training_author"]
        + " Authoring fields must use nonempty single-paragraph NFC Unicode text without "
        "category Cc characters, double quotes, or backslashes."
    )
    frozen_training_fields = (
        "schema_version",
        "lane",
        "seed",
        "split_totals",
        "locale_totals",
        "pair_totals",
        "domain_allocation",
        "option_count_allocation",
        "split_locale_option_count_allocation",
        "gold_position_allocation",
    )
    assert {field: current_training[field] for field in frozen_training_fields} == {
        field: baseline_training[field] for field in frozen_training_fields
    }
    assert {
        field: current_training["pilot_prefix"][field]
        for field in ("count", "train_count", "internal_dev_count", "complete_bilingual_pairs")
    } == {
        field: baseline_training["pilot_prefix"][field]
        for field in ("count", "train_count", "internal_dev_count", "complete_bilingual_pairs")
    }
    for role in sorted(set(v02_corpus.MODEL_ROLES) - {"training_author"}):
        assert v02_corpus.ROLE_TEMPLATES[role] == baseline.ROLE_TEMPLATES[role]
        lane = v02_corpus._ROLE_LANES[role]
        identity = (
            current_plan["identities"][0] if lane == "sealed" else current_training["slots"][0]
        )
        labels = ["option_0", "option_1"] if role == "sealed_adjudicator" else None
        assert v02_corpus._role_response_schema(
            role, identity, labels
        ) == baseline._role_response_schema(role, identity, labels)
        assert v02_corpus.native_decoder_schema(
            role, identity, labels
        ) == baseline.native_decoder_schema(role, identity, labels)


def test_c6a_sealed_receipt_stays_v1(tmp_path: Path) -> None:
    root = tmp_path / "sealed"
    _mkdir_0700(root)
    plan = v02_corpus._generate_sealed_plan()
    metrics = v02_corpus.reduce_sealed(plan, [])
    receipt_path = v02_corpus.create_aggregate_receipt(
        root,
        artifact_id="sealed",
        plan_data=plan,
        events=[],
        metrics=metrics,
        status="PENDING",
        ancestry=v02_corpus._root_ancestry(root, "sealed"),
    )
    receipt = json.loads(receipt_path.read_bytes())
    assert receipt["schema_version"] == v02_corpus.RECEIPT_SCHEMA
    assert "training_acceptance_policy" not in receipt
    v02_corpus.verify_receipt(receipt_path, plan_data=plan, events=[], expected_lane="sealed")


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
    for key in ("answer", "semantic", "gates", "content_digest", "construction"):
        failure.pop(key)
    failure["schema_version"] = "v02-envelope.v1"
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
    selected = set(v02_corpus.grounding_micro_pilot(plan))
    indices = [index for index, slot in enumerate(plan["slots"]) if slot["slot_id"] in selected]
    for index in indices[:2]:
        envelope = _training_envelope(plan, "training_author", index=index)
        for key in ("answer", "semantic", "gates", "content_digest", "construction"):
            envelope.pop(key)
        envelope["schema_version"] = "v02-envelope.v1"
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


def _case(
    option_count: int, *, state: str = "Fictional target label semantic state"
) -> dict[str, object]:
    return {
        "state": state,
        "question": "Which fictional option matches the rule?",
        "options": [
            {"id": f"option_{index}", "description": f"Distinct fictional choice {index}"}
            for index in range(option_count)
        ],
    }


def _judgments() -> dict[str, bool]:
    return {
        "fictionality_valid": True,
        "exclusive_options_valid": True,
        "ambiguity_free": True,
    }


def _construction(slot: Mapping[str, Any], state: str, answer: str) -> dict[str, Any]:
    return {
        "rule_quote": state.split()[0],
        "option_checks": [
            {
                "option_id": f"option_{index}",
                "supported": f"option_{index}" == answer,
                "reason": "Fictional policy check",
            }
            for index in range(cast(int, slot["option_count"]))
        ],
    }


def _raw(value: dict[str, object]) -> bytes:
    return canonical_json_bytes(cast(JsonValue, value))


def test_c1_prompt_contract_is_deterministic_and_six_roles_are_bound() -> None:
    first = v02_corpus.prompt_contract()
    assert first == v02_corpus.prompt_contract()
    assert set(first["template_digests"]) == set(v02_corpus.MODEL_ROLES)
    assert first["model_bindings"]["training_author"] == v02_corpus._model_identity(
        "training_author"
    )
    assert first["one_invocation_per_role_identity"] is True
    assert first["retry_policy"] == "forbidden"
    assert (
        first["author_metadata_const_policy"] == "planned-scenario-and-ordered-criterion-roles.v1"
    )
    training = v02_corpus._generate_training_plan()
    slot = training["slots"][0]
    request = v02_corpus.role_request(training, slot["slot_id"], "training_author")
    assert slot["scenario_code"] in request[1]["content"]
    assert f"option_{slot['gold_position']}" in request[1]["content"]
    sealed = v02_corpus._generate_sealed_plan()
    identity = sealed["identities"][0]
    sealed_request = v02_corpus.role_request(sealed, identity["identity_id"], "sealed_author")
    assert "fictional_diversity" in sealed_request[1]["content"]
    assert "target" not in json.loads(sealed_request[1]["content"].split("\n", 1)[1])


def test_c1_author_reviewer_annotator_and_adjudicator_round_trip() -> None:
    training = v02_corpus._generate_training_plan()
    slot = training["slots"][0]
    training_case = _case(slot["option_count"])
    author_response = {
        **training_case,
        "answer": f"option_{slot['gold_position']}",
        "construction": _construction(
            slot, cast(str, training_case["state"]), f"option_{slot['gold_position']}"
        ),
        "semantic": {
            "scenario_code": slot["scenario_code"],
            "criterion_roles": slot["criterion_roles"],
        },
        **_judgments(),
    }
    parsed_author = v02_corpus.parse_role_response(
        _raw(author_response), training, slot["slot_id"], "training_author"
    )
    author = v02_corpus.role_envelope(
        parsed_author, training, slot["slot_id"], "training_author", training_case, _all_gates()
    )
    assert author["content_digest"] == v02_corpus.case_digest(training_case)
    reviewer_response = {
        "answer": author_response["answer"],
        "semantic": author_response["semantic"],
        **_judgments(),
    }
    parsed_reviewer = v02_corpus.parse_role_response(
        _raw(reviewer_response), training, slot["slot_id"], "independent_reviewer"
    )
    reviewer = v02_corpus.role_envelope(
        parsed_reviewer,
        training,
        slot["slot_id"],
        "independent_reviewer",
        training_case,
        _all_gates(),
    )
    assert reviewer["content_digest"] == author["content_digest"]
    sealed = v02_corpus._generate_sealed_plan()
    identity = sealed["identities"][0]
    sealed_case = _case(identity["option_count"])
    sealed_response = {**sealed_case, "target": "option_0", **_judgments()}
    parsed_sealed = v02_corpus.parse_role_response(
        _raw(sealed_response), sealed, identity["identity_id"], "sealed_author"
    )
    v02_corpus.role_envelope(
        parsed_sealed, sealed, identity["identity_id"], "sealed_author", sealed_case, _all_gates()
    )
    for role, label in (("sealed_annotator_a", "option_0"), ("sealed_annotator_b", "option_1")):
        parsed = v02_corpus.parse_role_response(
            _raw({"label": label, **_judgments()}), sealed, identity["identity_id"], role
        )
        envelope = v02_corpus.role_envelope(
            parsed, sealed, identity["identity_id"], role, sealed_case, _all_gates()
        )
        assert envelope["label"] == label
    parsed_adjudicator = v02_corpus.parse_role_response(
        _raw({"choice": "option_0", **_judgments()}),
        sealed,
        identity["identity_id"],
        "sealed_adjudicator",
        ["option_0", "option_1"],
    )
    assert (
        v02_corpus.role_envelope(
            parsed_adjudicator,
            sealed,
            identity["identity_id"],
            "sealed_adjudicator",
            sealed_case,
            _all_gates(),
            ["option_0", "option_1"],
        )["choice"]
        == "option_0"
    )


def test_c1_blind_requests_and_closed_json_fail_closed() -> None:
    training = v02_corpus._generate_training_plan()
    slot = training["slots"][0]
    case = _case(slot["option_count"])
    request = v02_corpus.role_request(training, slot["slot_id"], "independent_reviewer", case)
    content = request[1]["content"]
    assert "gold_position" not in content and "family_id" not in content and "split" not in content
    with pytest.raises(ValueError, match="exactly a case"):
        v02_corpus.role_request(
            training, slot["slot_id"], "independent_reviewer", case, ["option_0", "option_1"]
        )
    malformed = (
        b'{"answer":"option_0","answer":"option_1","semantic":{},'
        b'"fictionality_valid":true,"exclusive_options_valid":true,"ambiguity_free":true}'
    )
    with pytest.raises(ValueError, match="pure JSON"):
        v02_corpus.parse_role_response(malformed, training, slot["slot_id"], "independent_reviewer")
    nonfinite = (
        b'{"answer":"option_0","semantic":{"scenario_code":"x","criterion_roles":[]},'
        b'"fictionality_valid":NaN,"exclusive_options_valid":true,"ambiguity_free":true}'
    )
    with pytest.raises(ValueError, match="pure JSON"):
        v02_corpus.parse_role_response(nonfinite, training, slot["slot_id"], "independent_reviewer")
    bad_case = _case(slot["option_count"])
    cast(list[dict[str, object]], bad_case["options"])[0]["extra"] = "metadata"
    with pytest.raises(ValueError, match="closed"):
        v02_corpus.role_request(training, slot["slot_id"], "independent_reviewer", bad_case)
    bad_unicode = _case(slot["option_count"], state="e\u0301")
    with pytest.raises(ValueError, match="NFC"):
        v02_corpus.validate_case(bad_unicode, slot)


def test_c1_bilingual_source_and_gate_binding_are_constrained() -> None:
    training = v02_corpus._generate_training_plan()
    slots = training["slots"]
    destination = next(slot for slot in slots if slot["bilingual_pair_id"] is not None)
    source = next(
        slot
        for slot in slots
        if slot["family_id"] == destination["family_id"]
        and slot["slot_id"] != destination["slot_id"]
    )
    source_case = _case(source["option_count"])
    request = v02_corpus.role_request(
        training,
        destination["slot_id"],
        "training_author",
        bilingual_source={"source_identity_id": source["slot_id"], "case": source_case},
    )
    assert source["slot_id"] in request[1]["content"]
    with pytest.raises(ValueError, match="distinct identity"):
        v02_corpus.role_request(
            training,
            destination["slot_id"],
            "training_author",
            bilingual_source={"source_identity_id": destination["slot_id"], "case": source_case},
        )
    case = _case(destination["option_count"])
    response = {
        **case,
        "answer": f"option_{destination['gold_position']}",
        "construction": _construction(
            destination, cast(str, case["state"]), f"option_{destination['gold_position']}"
        ),
        "semantic": {
            "scenario_code": destination["scenario_code"],
            "criterion_roles": destination["criterion_roles"],
        },
        **_judgments(),
    }
    parsed = v02_corpus.parse_role_response(
        _raw(response), training, destination["slot_id"], "training_author"
    )
    gates = _all_gates()
    gates["privacy_valid"] = False
    gates["fictionality_valid"] = True
    envelope = v02_corpus.role_envelope(
        parsed, training, destination["slot_id"], "training_author", case, gates
    )
    assert envelope["gates"]["privacy_valid"] is False
    with pytest.raises(ValueError, match="local gates are closed"):
        v02_corpus.role_envelope(
            parsed,
            training,
            destination["slot_id"],
            "training_author",
            case,
            {"schema_valid": True},
        )


@pytest.mark.parametrize("role", sorted(v02_corpus.MODEL_ROLES))
def test_c1_all_requests_provide_explicit_closed_response_schema(role: str) -> None:
    lane = v02_corpus._ROLE_LANES[role]
    plan = (
        v02_corpus._generate_training_plan()
        if lane == "training"
        else v02_corpus._generate_sealed_plan()
    )
    identity = (plan["slots"] if lane == "training" else plan["identities"])[0]
    identity_id = identity.get("slot_id", identity.get("identity_id"))
    case = None if role in {"training_author", "sealed_author"} else _case(identity["option_count"])
    labels = ["option_0", "option_1"] if role == "sealed_adjudicator" else None
    first = v02_corpus.role_request(plan, identity_id, role, case, labels)
    assert first == v02_corpus.role_request(plan, identity_id, role, case, labels)
    context = json.loads(first[1]["content"].split("\n", 1)[1])
    schema = context["response_schema"]
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == v02_corpus._response_fields(role)
    assert set(schema["properties"]) == set(schema["required"])
    if role == "sealed_adjudicator":
        assert context["case"] == case
        assert set(context["labels"]) == set(labels or [])
        assert set(schema["properties"]["choice"]["enum"]) == set(labels or [])
    if role in {
        "independent_reviewer",
        "sealed_annotator_a",
        "sealed_annotator_b",
        "sealed_adjudicator",
    }:
        assert "target" not in context and "semantic" not in context and "answer" not in context
        assert "gold_position" not in context and "family_id" not in context


@pytest.mark.parametrize("role", sorted(v02_corpus.MODEL_ROLES))
@pytest.mark.parametrize("option_count", range(2, 9))
def test_c5a_native_response_formats_are_closed_and_do_not_force_blind_judgments(
    role: str, option_count: int
) -> None:
    lane = v02_corpus._ROLE_LANES[role]
    plan = (
        v02_corpus._generate_training_plan()
        if lane == "training"
        else v02_corpus._generate_sealed_plan()
    )
    identity = next(
        item
        for item in (plan["slots"] if lane == "training" else plan["identities"])
        if item["option_count"] == option_count
    )
    identity_id = cast(str, identity.get("slot_id", identity.get("identity_id")))
    labels = ["option_0", "option_1"] if role == "sealed_adjudicator" else None
    prompt_schema = v02_corpus._role_response_schema(role, identity, labels)
    response_format = v02_corpus.native_response_format(role, identity, labels)
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["name"] == f"saracura_v02_{role}"
    schema = response_format["json_schema"]["schema"]
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == v02_corpus._response_fields(role)
    assert prompt_schema == v02_corpus._role_response_schema(role, identity, labels)
    for judgment in v02_corpus._JUDGMENT_FIELDS:
        assert schema["properties"][judgment] == {"type": "boolean"}
    label_field = {
        "training_author": "answer",
        "independent_reviewer": "answer",
        "sealed_author": "target",
        "sealed_annotator_a": "label",
        "sealed_annotator_b": "label",
        "sealed_adjudicator": "choice",
    }[role]
    assert schema["properties"][label_field] == {
        "type": "string",
        "enum": labels or [f"option_{index}" for index in range(option_count)],
    }
    if role in {"training_author", "sealed_author"}:
        prompt_options = prompt_schema["properties"]["options"]
        assert "prefixItems" not in prompt_options
        assert prompt_options["items"]["properties"]["id"] == {
            "type": "string",
            "enum": [f"option_{index}" for index in range(option_count)],
        }
        options = schema["properties"]["options"]
        assert options["items"] is False
        assert [item["properties"]["id"] for item in options["prefixItems"]] == [
            {"const": f"option_{index}"} for index in range(identity["option_count"])
        ]
    if role == "training_author":
        assert (
            prompt_schema["properties"]["semantic"]["properties"]["criterion_roles"]["uniqueItems"]
            is True
        )
        assert schema["properties"]["semantic"]["const"] == {
            "scenario_code": identity["scenario_code"],
            "criterion_roles": identity["criterion_roles"],
        }
        prompt_construction = prompt_schema["properties"]["construction"]
        native_construction = schema["properties"]["construction"]
        assert prompt_construction["properties"]["rule_quote"] == {
            "type": "string",
            "minLength": 1,
            "maxLength": 240,
            "pattern": r'^[^\u0000-\u001F\u007F-\u009F"\\]{1,240}$',
        }
        assert native_construction["properties"]["rule_quote"] == {
            "type": "string",
            "pattern": r'^[^\u0000-\u001F\u007F-\u009F"\\]{1,240}$',
        }
        for construction_schema in (prompt_construction, native_construction):
            checks = construction_schema["properties"]["option_checks"]
            assert checks["minItems"] == checks["maxItems"] == option_count
            if "items" in checks and checks["items"] is not False:
                check_items = [checks["items"]]
            else:
                check_items = checks["prefixItems"]
                assert len(check_items) == option_count
            for check in check_items:
                assert check["properties"]["supported"] == {"type": "boolean"}
                expected_reason: dict[str, object] = {
                    "type": "string",
                    "pattern": r'^[^\u0000-\u001F\u007F-\u009F"\\]{1,160}$',
                }
                if construction_schema is prompt_construction:
                    expected_reason.update({"minLength": 1, "maxLength": 160})
                assert check["properties"]["reason"] == expected_reason
    else:
        for judgment in v02_corpus._JUDGMENT_FIELDS:
            assert schema["properties"][judgment] == {"type": "boolean"}
    if role == "independent_reviewer":
        prompt_schema = v02_corpus._role_response_schema(role, identity)
        prompt_roles = prompt_schema["properties"]["semantic"]["properties"]["criterion_roles"]
        native_roles = schema["properties"]["semantic"]["properties"]["criterion_roles"]
        assert "uniqueItems" not in prompt_roles
        assert "uniqueItems" not in native_roles
    if role == "sealed_adjudicator":
        assert schema["properties"]["choice"]["enum"] == labels
    assert identity_id


def test_c8_fresh_training_ids_reject_source643_plans_and_grants(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    current_training = v02_corpus._generate_training_plan()
    current_sealed = v02_corpus._generate_sealed_plan()
    baseline_root = tmp_path / "baseline" / "benchmarks"
    (baseline_root / "manifests").mkdir(parents=True)
    for relative in (
        "benchmarks/v02_corpus.py",
        "benchmarks/manifests/v02-distillation-safe-corpus.v1.json",
    ):
        result = subprocess.run(
            ["git", "show", f"643e52e:{relative}"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
        )
        target = tmp_path / "baseline" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(result.stdout)
    spec = spec_from_file_location(
        "v02_corpus_source643_ids", tmp_path / "baseline/benchmarks/v02_corpus.py"
    )
    assert spec is not None and spec.loader is not None
    baseline = module_from_spec(spec)
    spec.loader.exec_module(baseline)
    old_training = baseline._generate_training_plan()
    assert set(current_training["slot_ids"]).isdisjoint(old_training["slot_ids"])
    assert current_sealed["namespace"] == v02_corpus.SEALED_NAMESPACE
    legacy_root = tmp_path / "legacy"
    _mkdir_0700(legacy_root)
    with pytest.raises(ValueError, match="frozen plan"):
        v02_corpus.OfflineLedger(old_training, legacy_root)
    assert not (legacy_root / "role-reservations").exists()
    root = tmp_path / "current"
    _mkdir_0700(root)
    renderer, tokenizer, inventory = _load_runtime(monkeypatch, tmp_path / "runtime")
    grant = _grant(inventory)
    grant["training_plan_digest"] = v02_corpus._sha256(
        canonical_json_bytes(cast(JsonValue, old_training))
    )
    ledger = v02_corpus.OfflineLedger(current_training, root)
    slot = _pilot_training_slot(current_training, pilot=True)
    transport = _Loopback(v02_corpus.MODEL_ROLES["training_author"]["model"], "{}")
    with pytest.raises(ValueError, match="environment grant value mismatch"):
        _run_role(
            ledger,
            cast(str, slot["slot_id"]),
            "training_author",
            grant,
            _license_bytes(),
            renderer,
            tokenizer,
            transport,
        )
    assert transport.calls == []
    assert not (root / "role-reservations").exists()


def test_c1_adjudicator_needs_case_and_cannot_choose_a_third_label() -> None:
    plan = v02_corpus._generate_sealed_plan()
    identity = next(row for row in plan["identities"] if row["option_count"] == 3)
    labels = ["option_0", "option_1"]
    case = _case(3)
    with pytest.raises(ValueError, match="needs a case"):
        v02_corpus.role_request(
            plan, identity["identity_id"], "sealed_adjudicator", committed_labels=labels
        )
    raw = _raw({"choice": "option_2", **_judgments()})
    with pytest.raises(ValueError, match="committed label"):
        v02_corpus.parse_role_response(
            raw, plan, identity["identity_id"], "sealed_adjudicator", labels
        )
    with pytest.raises(ValueError, match="committed label"):
        v02_corpus.role_envelope(
            json.loads(raw),
            plan,
            identity["identity_id"],
            "sealed_adjudicator",
            case,
            _all_gates(),
            labels,
        )
    first = v02_corpus.role_request(
        plan, identity["identity_id"], "sealed_adjudicator", case, labels
    )
    assert first == v02_corpus.role_request(
        plan, identity["identity_id"], "sealed_adjudicator", case, list(reversed(labels))
    )
    with pytest.raises(ValueError, match="closed"):
        v02_corpus.role_request(
            plan,
            identity["identity_id"],
            "sealed_adjudicator",
            {**case, "target": "option_0"},
            labels,
        )


@pytest.mark.parametrize("gate", sorted(v02_corpus.LOCAL_GATES))
def test_c1_every_gate_is_explicit_and_false_is_preserved(gate: str) -> None:
    plan = v02_corpus._generate_sealed_plan()
    identity = plan["identities"][0]
    case = _case(identity["option_count"])
    response = {**case, "target": "option_0", **_judgments()}
    gates = _all_gates()
    gates[gate] = False
    result = v02_corpus.role_envelope(
        response, plan, identity["identity_id"], "sealed_author", case, gates
    )
    assert result["gates"][gate] is False
    gates.pop(gate)
    with pytest.raises(ValueError):
        v02_corpus.role_envelope(
            response, plan, identity["identity_id"], "sealed_author", case, gates
        )
    gates[gate] = cast(bool, 1)
    with pytest.raises(ValueError):
        v02_corpus.role_envelope(
            response, plan, identity["identity_id"], "sealed_author", case, gates
        )
    with pytest.raises(ValueError):
        v02_corpus.role_envelope(
            response,
            plan,
            identity["identity_id"],
            "sealed_author",
            case,
            {**_all_gates(), "unknown": True},
        )


@pytest.mark.parametrize("judgment", sorted(v02_corpus._JUDGMENT_FIELDS))
def test_c1_model_judgment_false_cannot_be_promoted(judgment: str) -> None:
    plan = v02_corpus._generate_sealed_plan()
    identity = plan["identities"][0]
    case = _case(identity["option_count"])
    response = {**case, "target": "option_0", **_judgments(), judgment: False}
    result = v02_corpus.role_envelope(
        response, plan, identity["identity_id"], "sealed_author", case, _all_gates()
    )
    assert result["gates"][judgment] is False


def test_c1_author_binding_never_normalizes_or_rewrites_content() -> None:
    plan = v02_corpus._generate_sealed_plan()
    identity = plan["identities"][0]
    case = _case(identity["option_count"], state="é")
    response = {**case, "target": "option_0", **_judgments()}
    with pytest.raises(ValueError, match="NFC"):
        v02_corpus.role_envelope(
            {**response, "state": "e\u0301"},
            plan,
            identity["identity_id"],
            "sealed_author",
            case,
            _all_gates(),
        )
    with pytest.raises(ValueError, match="differs"):
        v02_corpus.role_envelope(
            {**response, "state": "Different facts"},
            plan,
            identity["identity_id"],
            "sealed_author",
            case,
            _all_gates(),
        )


def test_c1_schema_and_system_changes_change_contract_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = v02_corpus.prompt_contract()
    assert set(original["response_schema_digests"]) == set(v02_corpus.MODEL_ROLES)
    monkeypatch.setattr(v02_corpus, "_ROLE_SYSTEM_TEMPLATE", "changed")
    assert v02_corpus.prompt_contract()["contract_digest"] != original["contract_digest"]
    monkeypatch.undo()
    monkeypatch.setattr(v02_corpus, "NATIVE_DECODER_POLICY", "changed")
    assert v02_corpus.prompt_contract()["contract_digest"] != original["contract_digest"]
    monkeypatch.undo()
    native_decoder_schema = v02_corpus.native_decoder_schema

    def changed_native_decoder_schema(
        role: str, identity: dict[str, Any], labels: list[str] | None = None
    ) -> dict[str, Any]:
        schema = native_decoder_schema(role, identity, labels)
        schema["native_schema_mutation"] = True
        return schema

    monkeypatch.setattr(v02_corpus, "native_decoder_schema", changed_native_decoder_schema)
    assert v02_corpus.prompt_contract()["contract_digest"] != original["contract_digest"]


@pytest.mark.parametrize(
    "raw",
    [
        b"{} trailing",
        b"```json\n{}\n```",
        b"\xff",
        b'{"semantic":{"x":1,"x":2}}',
        b'{"x":Infinity}',
    ],
)
def test_c1_malformed_model_bytes_are_rejected(raw: bytes) -> None:
    plan = v02_corpus._generate_sealed_plan()
    with pytest.raises(ValueError):
        v02_corpus.parse_role_response(
            raw, plan, plan["identities"][0]["identity_id"], "sealed_author"
        )


def test_c1_case_limits_option_ids_and_descriptions() -> None:
    identity = {"option_count": 2}
    for invalid in (
        _case(1),
        _case(2, state=" "),
        _case(2, state="line\nnext"),
        _case(2, state="x" * 16384),
    ):
        with pytest.raises(ValueError):
            v02_corpus.validate_case(invalid, identity)
    case = _case(2)
    options = cast(list[dict[str, object]], case["options"])
    options[1]["description"] = " DISTINCT fictional choice 0 "
    with pytest.raises(ValueError, match="distinct"):
        v02_corpus.validate_case(case, identity)
    options[1]["description"] = "Other"
    options[1]["id"] = "option_0"
    with pytest.raises(ValueError, match="id/order"):
        v02_corpus.validate_case(case, identity)


def test_c1_bilingual_context_rejects_other_families_and_blind_roles() -> None:
    plan = v02_corpus._generate_training_plan()
    destination = next(row for row in plan["slots"] if row["bilingual_pair_id"] is not None)
    source = next(row for row in plan["slots"] if row["family_id"] != destination["family_id"])
    context = {"source_identity_id": source["slot_id"], "case": _case(source["option_count"])}
    with pytest.raises(ValueError, match="family"):
        v02_corpus.role_request(
            plan, destination["slot_id"], "training_author", bilingual_source=context
        )
    with pytest.raises(ValueError):
        v02_corpus.role_request(
            plan,
            destination["slot_id"],
            "independent_reviewer",
            _case(destination["option_count"]),
            bilingual_source=context,
        )
    with pytest.raises(ValueError):
        v02_corpus.role_request(plan, destination["slot_id"], "sealed_author")
    mutated = json.loads(json.dumps(plan))
    mutated["slots"][0]["gold_position"] = 100
    with pytest.raises(ValueError, match="frozen plan"):
        v02_corpus.role_request(mutated, destination["slot_id"], "training_author")


def test_c6a_semantic_disagreement_is_diagnostic_and_three_way_choice_is_accepted(
    tmp_path: Path,
) -> None:
    _mkdir_0700(tmp_path)
    plan = v02_corpus._generate_training_plan()
    author = _training_envelope(plan, "training_author")
    reviewer = _training_envelope(plan, "independent_reviewer")
    semantic = cast(dict[str, object], reviewer["semantic"])
    roles = cast(list[str], semantic["criterion_roles"])
    semantic["criterion_roles"] = [roles[0]] * len(roles)
    ledger = v02_corpus.OfflineLedger(plan, tmp_path)
    ledger.commit("training_author", author)
    ledger.commit("training_reviewer", reviewer)
    metrics = v02_corpus.reduce_training(plan, ledger.events())
    assert metrics["accepted"] == 1 and metrics["resolved"] == 1
    assert metrics["denominator"] == 1600
    assert metrics["auxiliary_semantic_diagnostics"] == {
        "reviewed_rows": 1,
        "full_agreement": 0,
        "full_disagreement": 1,
        "scenario_disagreement": 0,
        "role_vector_disagreement": 1,
    }


def test_c6a_auxiliary_diagnostics_count_successful_reviewer_envelopes() -> None:
    plan = v02_corpus._generate_training_plan()
    events: list[dict[str, object]] = []
    for index in range(3):
        author = _training_envelope(plan, "training_author", index=index)
        reviewer = _training_envelope(plan, "independent_reviewer", index=index)
        semantic = cast(dict[str, object], reviewer["semantic"])
        if index == 1:
            slot = cast(list[dict[str, object]], plan["slots"])[index]
            scenarios = v02_corpus._validated_corpus_taxonomy()[3][cast(str, slot["domain"])]
            semantic["scenario_code"] = next(
                code for code in scenarios if code != semantic["scenario_code"]
            )
        if index == 2:
            semantic["criterion_roles"] = list(
                reversed(cast(list[str], semantic["criterion_roles"]))
            )
        events.extend([_event("training_author", author), _event("training_reviewer", reviewer)])
    metrics = v02_corpus.reduce_training(plan, events)
    assert metrics["accepted"] == 3
    assert metrics["auxiliary_semantic_diagnostics"] == {
        "reviewed_rows": 3,
        "full_agreement": 1,
        "full_disagreement": 2,
        "scenario_disagreement": 1,
        "role_vector_disagreement": 1,
    }


def test_c6a_rejected_primary_results_still_count_successful_reviewer_diagnostics() -> None:
    plan = v02_corpus._generate_training_plan()
    events: list[dict[str, object]] = []
    for index, rejected_by in enumerate(("wrong_choice", "author_gate", "reviewer_gate")):
        slot = cast(list[dict[str, object]], plan["slots"])[index]
        wrong_choice = (
            f"option_{(cast(int, slot['gold_position']) + 1) % cast(int, slot['option_count'])}"
        )
        author = _training_envelope(
            plan,
            "training_author",
            answer=wrong_choice if rejected_by == "wrong_choice" else None,
            index=index,
        )
        reviewer = _training_envelope(
            plan,
            "independent_reviewer",
            answer=wrong_choice if rejected_by == "wrong_choice" else None,
            index=index,
        )
        if rejected_by == "author_gate":
            cast(dict[str, bool], author["gates"])["privacy_valid"] = False
        if rejected_by == "reviewer_gate":
            cast(dict[str, bool], reviewer["gates"])["privacy_valid"] = False
        events.extend([_event("training_author", author), _event("training_reviewer", reviewer)])
    failure = _training_envelope(plan, "training_author", index=3)
    for key in ("answer", "semantic", "gates", "content_digest", "construction"):
        failure.pop(key)
    failure["schema_version"] = "v02-envelope.v1"
    failure["error_code"] = "invalid_output"
    events.append(_event("training_author_failure", failure))
    metrics = v02_corpus.reduce_training(plan, events)
    assert metrics["accepted"] == 0
    assert metrics["auxiliary_semantic_diagnostics"] == {
        "reviewed_rows": 3,
        "full_agreement": 3,
        "full_disagreement": 0,
        "scenario_disagreement": 0,
        "role_vector_disagreement": 0,
    }


_RENDERER_SOURCE = """\
SPECIAL = [
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|box_start|>",
    "<|box_end|>",
    "<|fim_suffix|>",
]
MAX_STATE = 384
MAX_BRANCH = 1024
MAX_PACKED = 2048
SERVE_MAX_STATE = 384
SERVE_MAX_BRANCH = 1024
SERVE_MAX_PACKED = 2048
MAX_TRAIN_STATE = 384
OPT_NONE = 0
OPT_DECIDE = 1
_SPECIAL_RE = re.compile("fim")


class ContextOverflow(ValueError):
    pass


def training_context():
    return {"max_branch": MAX_BRANCH, "max_packed": MAX_PACKED, "max_state": MAX_STATE}


def user_tokens(tokenizer, text):
    rewritten = text
    for marker in SPECIAL:
        name = marker[2:-2]
        rewritten = rewritten.replace(marker, "<\u00a6" + name + "\u00a6>")
    return tokenizer(rewritten, add_special_tokens=False).input_ids


def encode(tokenizer, record, max_state=384, max_branch=1024, strict=True, option_isolation=False):
    if (
        max_state != 384
        or max_branch != 1024
        or strict is not True
        or option_isolation is not False
    ):
        raise ValueError("closed")
    state = record["state"]
    if len(state) > max_state:
        raise ContextOverflow("state")
    ids = user_tokens(tokenizer, state)
    return {"ids": ids, "labels": [OPT_NONE], "state_truncated": "TRUNCATE" in state}
"""

_LOCK_HOLDER = """\
import fcntl
import os
import sys

fd = os.open(sys.argv[1] + "/execution.lock", os.O_RDWR | os.O_CREAT, 0o600)
fcntl.flock(fd, fcntl.LOCK_EX)
sys.stdout.write("locked\\n")
sys.stdout.flush()
sys.stdin.readline()
"""


class _FixedTokenizer:
    def __init__(self, count: int = 4) -> None:
        self.count = count

    def __call__(self, text: str, add_special_tokens: bool = False) -> SimpleNamespace:
        del text
        if add_special_tokens:
            raise ValueError("tokenizer special tokens are disabled")
        return SimpleNamespace(input_ids=[1] * self.count)

    def convert_tokens_to_ids(self, token: str) -> int:
        del token
        return 3


class _Loopback:
    def __init__(self, served: str, content: str, *, fail: str | None = None) -> None:
        self.served = served
        self.content = content
        self.fail = fail
        self.calls: list[Mapping[str, Any]] = []

    def __call__(self, request: Mapping[str, Any]) -> dict[str, Any]:
        self.calls.append(request)
        if self.fail == "timeout":
            raise TimeoutError("timed out")
        if self.fail == "redirect":
            return {"status": 302, "body": b"redirect"}
        url = request["url"]
        if not isinstance(url, str):
            raise AssertionError("url")
        if url.endswith("/v1/models"):
            return {"status": 200, "body": json.dumps({"data": [{"id": self.served}]}).encode()}
        payload = {"choices": [{"message": {"content": self.content}}]}
        return {"status": 200, "body": json.dumps(payload).encode()}

    def posts(self) -> list[Mapping[str, Any]]:
        return [call for call in self.calls if str(call["url"]).endswith("/chat/completions")]


def _reject_network(request: Mapping[str, Any]) -> dict[str, Any]:
    del request
    raise AssertionError("network")


def _case_any(
    option_count: int, *, state: str = "Fictional target label semantic state"
) -> dict[str, Any]:
    return cast(dict[str, Any], _case(option_count, state=state))


def _pin_renderer(monkeypatch: pytest.MonkeyPatch, source: str) -> bytes:
    raw = source.encode()
    monkeypatch.setattr(
        v02_corpus, "PINNED_RENDERER_SOURCE_SHA256", hashlib.sha256(raw).hexdigest()
    )
    return raw


def _write_snapshot(
    root: Path, payload: bytes = b"reviewed-tokenizer"
) -> tuple[Path, list[dict[str, str]]]:
    path = root / "tokenizer"
    path.mkdir(parents=True)
    os.chmod(path, 0o700)
    target = path / "tokenizer.json"
    target.write_bytes(payload)
    os.chmod(target, 0o600)
    digest = hashlib.sha256(payload).hexdigest()
    return path, [{"filename": "tokenizer.json", "sha256": digest}]


def _license_bytes() -> dict[str, bytes]:
    return {role: f"license:{role}".encode() for role in sorted(v02_corpus._MODEL_ROLE_NAMES)}


def _grant(inventory: list[dict[str, str]]) -> dict[str, Any]:
    licenses = _license_bytes()
    grant: dict[str, Any] = {
        "account_boundary_digest": "33" * 32,
        "candidate_rendering_digest": v02_corpus.candidate_rendering_digest(),
        "cloud_legal_service_name": "reviewed-operator",
        "deletion_mechanism": "operator-volume-delete",
        "instance_class": "gpu-40gib",
        "kev_rendering_function_digest": v02_corpus.kev_rendering_function_digest(),
        "license_sha256": {role: v02_corpus._sha256(blob) for role, blob in licenses.items()},
        "log_retention": "24h",
        "model_identities": {
            role: v02_corpus._model_identity(role) for role in sorted(v02_corpus._MODEL_ROLE_NAMES)
        },
        "post_run_deletion_obligation": "delete-volumes-after-run",
        "prompt_contract_digest": v02_corpus.prompt_contract()["contract_digest"],
        "region": "reviewed-region",
        "renderer_source_sha256": v02_corpus.PINNED_RENDERER_SOURCE_SHA256,
        "runtime_image_digest": "11" * 32,
        "runtime_lock_digest": "22" * 32,
        "schema_version": "v02-environment-grant.v1",
        "sealed_plan_digest": v02_corpus._sha256(v02_corpus._frozen_plan_bytes("sealed")),
        "storage_retention": "24h",
        "tokenizer_inventory": inventory,
        "training_plan_digest": v02_corpus._sha256(v02_corpus._frozen_plan_bytes("training")),
        "transport": "tls-ssh",
    }
    for field in v02_corpus._GRANT_TRUE_FIELDS:
        grant[field] = True
    assert set(grant) == v02_corpus._GRANT_FIELDS
    return grant


def _runtime(
    role: str, grant: Mapping[str, Any], grant_digest: str, inventory_digest: str
) -> dict[str, Any]:
    evidence: dict[str, Any] = {
        "compute": "bf16",
        "cpu_offload": False,
        "dependency_lock_digest": "44" * 32,
        "full_load_passed": True,
        "gpu_class": "NVIDIA-L40S-CUDA",
        "gpu_count": 1,
        "gpu_vram_gib": 48,
        "grant_digest": grant_digest,
        "model_identity": v02_corpus._model_identity(role),
        "quantization": "bitsandbytes-nf4",
        "renderer_verified": True,
        "role": role,
        "runtime_image_digest": grant["runtime_image_digest"],
        "runtime_lock_digest": grant["runtime_lock_digest"],
        "schema_version": "v02-runtime-evidence.v1",
        "served_model_name": v02_corpus.MODEL_ROLES[role]["model"],
        "source_snapshot_verified": True,
        "tokenizer_inventory_digest": inventory_digest,
        "tokenizer_revision": v02_corpus.PINNED_TOKENIZER_REVISION,
        "tokenizer_verified": True,
    }
    assert set(evidence) == v02_corpus._RUNTIME_FIELDS
    return evidence


def _load_runtime(
    monkeypatch: pytest.MonkeyPatch, root: Path, *, count: int = 4, source: str = _RENDERER_SOURCE
) -> tuple[v02_corpus.VerifiedRenderer, v02_corpus.VerifiedTokenizer, list[dict[str, str]]]:
    raw = _pin_renderer(monkeypatch, source)

    def load_tokenizer(snapshot: Path) -> _FixedTokenizer:
        del snapshot
        return _FixedTokenizer(count)

    monkeypatch.setattr(v02_corpus, "_load_transformers_tokenizer", load_tokenizer)
    renderer = v02_corpus.load_pinned_renderer(raw)
    snapshot, inventory = _write_snapshot(root)
    tokenizer = v02_corpus.load_verified_tokenizer(snapshot, inventory)
    return renderer, tokenizer, inventory


def _run_role(
    ledger: v02_corpus.OfflineLedger,
    identity_id: str,
    role: str,
    grant: dict[str, Any],
    licenses: dict[str, bytes],
    renderer: v02_corpus.VerifiedRenderer,
    tokenizer: v02_corpus.VerifiedTokenizer,
    transport: _Loopback | Any,
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if evidence is None:
        grant_digest = v02_corpus.validate_environment_grant(grant, licenses)
        evidence = _runtime(role, grant, grant_digest, tokenizer.inventory_digest)
    return v02_corpus.execute_role(
        ledger,
        identity_id,
        role,
        evidence,
        grant,
        licenses,
        "http://127.0.0.1:9",
        "test-bearer",
        renderer,
        tokenizer,
        transport=transport,
    )


def _write_private_canonical(path: Path, value: Any) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    v02_corpus._atomic_write(path, canonical_json_bytes(cast(JsonValue, value)) + b"\n")


def _write_private_pretty_json(path: Path, value: Any) -> None:
    """Create self-authored noncanonical metadata for legacy-byte regression tests."""
    v02_corpus._atomic_write(
        path, json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    )


def _sealer_lock(inventory: list[dict[str, str]], source_revision: str) -> dict[str, Any]:
    role_map = {role: 1 for role in v02_corpus._MODEL_ROLE_NAMES}
    cohort_bindings = {
        "training": {
            cohort: {
                "phase_id": f"training-{cohort}",
                "selection_digest": f"{index + 10:064x}",
            }
            for index, cohort in enumerate(("micro", "pilot", "full"))
        },
        "sealed": {
            cohort: {
                "phase_id": f"sealed-{cohort}",
                "selection_digest": f"{index + 20:064x}",
            }
            for index, cohort in enumerate(("pilot", "full"))
        },
    }
    return {
        "schema_version": "private-runtime-lock.v1",
        "source_commit": source_revision,
        "public_source_integrity": {
            "source_commit": source_revision,
            "archive_sha256": "a" * 64,
            "source_inventory_digest": "b" * 64,
            "files_verified": 1,
        },
        "code_sha256": {"v02_corpus.py": "c" * 64},
        "cohort_bindings": cohort_bindings,
        "cpu_offload_gb": 0,
        "dependency_lock_sha256": "d" * 64,
        "driver": "test",
        "dtype": "bf16",
        "gpu": "test",
        "gpu_memory_mib": 1,
        "historical_measurements": {},
        "historical_model_metadata": {},
        "historical_model_metadata_sha256": "e" * 64,
        "host": "test",
        "image_digest": "f" * 64,
        "installed_versions": {"python": "3.11"},
        "max_model_len_by_role": role_map,
        "max_num_seqs": 1,
        "max_output_tokens": 1,
        "native_json_grammar": True,
        "port": 1,
        "public_runner_sha256": "1" * 64,
        "python": "3.11",
        "quantization": "nf4",
        "renderer_source_sha256": v02_corpus.PINNED_RENDERER_SOURCE_SHA256,
        "request_body_logging": False,
        "teacher_input_ceiling_by_role": role_map,
        "timeout_seconds": 1,
        "tokenizer_inventory": inventory,
    }


def _sealer_fixture(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Path]:
    root = tmp_path / "training"
    _mkdir_0700(root)
    source_raw = _pin_renderer(monkeypatch, _RENDERER_SOURCE)
    source_revision = "9" * 40
    plan = v02_corpus._generate_training_plan()
    plan_path = root / "training-plan.json"
    _write_private_canonical(plan_path, plan)
    ledger = v02_corpus.OfflineLedger(plan, root)
    snapshot, inventory = _write_snapshot(root)
    monkeypatch.setattr(
        v02_corpus, "_load_transformers_tokenizer", lambda _snapshot: _FixedTokenizer(4)
    )
    inventory_path = root / "tokenizer-inventory.json"
    _write_private_canonical(inventory_path, inventory)
    renderer_path = root / "renderer.py"
    v02_corpus._atomic_write(renderer_path, source_raw)
    lock = _sealer_lock(inventory, source_revision)
    lock_path = root / "runtime-lock.json"
    _write_private_canonical(lock_path, lock)
    grant = _grant(inventory)
    grant["runtime_lock_digest"] = v02_corpus._sha256(canonical_json_bytes(lock))
    licenses = _license_bytes()
    license_files: dict[str, str] = {}
    for role, content in licenses.items():
        relative = f"licenses/{role}.txt"
        license_files[role] = relative
        target = root / relative
        target.parent.mkdir(mode=0o700, exist_ok=True)
        os.chmod(target.parent, 0o700)
        v02_corpus._atomic_write(target, content)
    grant_path = root / "environment-grant.json"
    _write_private_canonical(grant_path, grant)
    grant_digest = v02_corpus.validate_environment_grant(grant, licenses)
    prompt_digest = v02_corpus.prompt_contract()["contract_digest"]
    renderer_lock_digest = v02_corpus._renderer_lock_digest()
    control = root / "private-control"
    control.mkdir(mode=0o700)
    os.chmod(control, 0o700)
    _write_private_canonical(
        control / "environment-binding.json",
        {
            "schema_version": "v02-environment-binding.v1",
            "grant_digest": grant_digest,
            "prompt_contract_digest": prompt_digest,
            "renderer_lock_digest": renderer_lock_digest,
        },
    )
    for role in ("training_author", "independent_reviewer"):
        _write_private_canonical(
            control / f"{role}-binding.json",
            {
                "schema_version": "v02-role-binding.v1",
                "role": role,
                "grant_digest": grant_digest,
                "runtime_evidence_digest": "2" * 64,
                "prompt_contract_digest": prompt_digest,
                "renderer_lock_digest": renderer_lock_digest,
            },
        )
    reservations = root / "role-reservations"
    reservations.mkdir(mode=0o700)
    os.chmod(reservations, 0o700)
    events: list[dict[str, Any]] = []
    for index, slot in enumerate(cast(list[dict[str, Any]], plan["slots"])):
        identity_id = cast(str, slot["slot_id"])
        case = {
            "state": f"Fictional policy scenario {index}",
            "question": f"Which fictional action applies to scenario {index}?",
            "options": [
                {
                    "id": f"option_{option_index}",
                    "description": f"Fictional scenario {index} action {option_index}",
                }
                for option_index in range(cast(int, slot["option_count"]))
            ],
        }
        digest = v02_corpus.case_digest(case)
        v02_corpus._store_private_case(root, "training", identity_id, case)
        author = _training_envelope(plan, "training_author", index=index)
        reviewer = _training_envelope(plan, "independent_reviewer", index=index)
        author["content_digest"] = digest
        reviewer["content_digest"] = digest
        cast(dict[str, Any], author["construction"])["rule_quote"] = "Fictional"
        for transition, envelope in (
            ("training_author", author),
            ("training_reviewer", reviewer),
        ):
            event = _event(transition, envelope)
            events.append(cast(dict[str, Any], event))
            v02_corpus._atomic_write(
                ledger.events_root / v02_corpus._event_filename(identity_id, transition),
                canonical_json_bytes(cast(JsonValue, event)) + b"\n",
            )
        for role in ("training_author", "independent_reviewer"):
            _write_private_canonical(
                v02_corpus._reservation_path(root, identity_id, role),
                {
                    "schema_version": "v02-role-reservation.v1",
                    "identity_id": identity_id,
                    "role": role,
                    "grant_digest": grant_digest,
                    "runtime_evidence_digest": "2" * 64,
                    "prompt_contract_digest": prompt_digest,
                    "renderer_lock_digest": renderer_lock_digest,
                    "request_digest": "3" * 64,
                },
            )
    assert len(ledger.events()) == 3200
    metrics = v02_corpus.reduce_training(plan, events)
    assert metrics["status"] == "READY"
    receipt_path = v02_corpus.create_aggregate_receipt(
        root,
        artifact_id="training-ready",
        plan_data=plan,
        events=events,
        metrics=metrics,
        status="READY",
        ancestry=v02_corpus._root_ancestry(root, "training"),
    )
    historical_manifest = "4" * 64
    historical_source = "5" * 64
    phase_a = {
        "packet_manifest_sha256": historical_manifest,
        "status": "BLOCKED_DATA_RIGHTS",
    }
    phase_a_path = root / "phase-a-receipt.json"
    _write_private_canonical(phase_a_path, phase_a)
    phase_a_hash = v02_corpus._sha256(phase_a_path.read_bytes())
    exclusions = {
        "schema_version": v02_corpus.HISTORICAL_EXCLUSION_SCHEMA,
        "status": "DERIVED_NOT_SEALING_READY",
        "historical_manifest_sha256": historical_manifest,
        "historical_source_sha256": historical_source,
        "phase_a_receipt_sha256": phase_a_hash,
        "source_records": 1304,
        "normalization": v02_corpus.HISTORICAL_EXCLUSION_NORMALIZATION,
        "identity_source_field": "task_id",
        "question_source_field": "instruction",
        "option_source_field": "criteria; IDs excluded by canonical combined fingerprint",
        "sets": {
            name: [f"{number:064x}" for number in range(1, 1305)]
            for name in ("identity", "state_question", "combined_content")
        },
        "literal_serialized_state_alias_sets": {
            name: [f"{number:064x}" for number in range(1, 1305)]
            for name in ("state_question", "combined_content")
        },
        "historical_holdout_opened": False,
        "raw_rows_uploaded": False,
        "limitation": "train-dev fingerprints only",
    }
    exclusions_path = root / "historical-exclusions.json"
    _write_private_canonical(exclusions_path, exclusions)
    exclusions_hash = v02_corpus._sha256(exclusions_path.read_bytes())
    monkeypatch.setattr(v02_corpus, "HISTORICAL_PACKET_MANIFEST_SHA256", historical_manifest)
    monkeypatch.setattr(v02_corpus, "HISTORICAL_TRAIN_DEV_SOURCE_SHA256", historical_source)
    monkeypatch.setattr(v02_corpus, "HISTORICAL_PHASE_A_RECEIPT_SHA256", phase_a_hash)
    monkeypatch.setattr(v02_corpus, "HISTORICAL_EXCLUSIONS_SHA256", exclusions_hash)
    inputs = {
        "schema_version": v02_corpus.TRAINING_SEALER_INPUTS_SCHEMA,
        "artifact_id": "training-capsule",
        "corpus_source_revision": source_revision,
        "policy_sha256": v02_corpus._sha256(
            canonical_json_bytes(v02_corpus.TRAINING_ACCEPTANCE_POLICY)
        ),
        "environment_grant_file": grant_path.name,
        "environment_grant_sha256": v02_corpus._sha256(grant_path.read_bytes()),
        "license_files": license_files,
        "runtime_lock_file": lock_path.name,
        "runtime_lock_sha256": v02_corpus._sha256(lock_path.read_bytes()),
        "renderer_source_file": renderer_path.name,
        "renderer_source_sha256": v02_corpus._sha256(renderer_path.read_bytes()),
        "tokenizer_directory": snapshot.name,
        "tokenizer_inventory_file": inventory_path.name,
        "tokenizer_inventory_sha256": v02_corpus._sha256(inventory_path.read_bytes()),
        "exclusions_sha256": exclusions_hash,
        "historical_phase_a_receipt_file": phase_a_path.name,
        "historical_phase_a_receipt_sha256": phase_a_hash,
        "historical_source_sha256": historical_source,
        "historical_manifest_sha256": historical_manifest,
    }
    inputs_path = root / "sealer-inputs.json"
    _write_private_canonical(inputs_path, inputs)
    output_parent = tmp_path / "capsules"
    _mkdir_0700(output_parent)
    return {
        "root": root,
        "plan": plan_path,
        "receipt": receipt_path,
        "exclusions": exclusions_path,
        "inputs": inputs_path,
        "output_parent": output_parent,
    }


def test_training_capsule_sealer_and_verifier_recompute_real_training_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fixture = _sealer_fixture(monkeypatch, tmp_path)
    measured_calls = 0
    original_measure = v02_corpus.VerifiedRenderer.measure

    def count_measure(
        renderer: v02_corpus.VerifiedRenderer,
        case: Mapping[str, Any],
        tokenizer: v02_corpus.VerifiedTokenizer,
    ) -> dict[str, int]:
        nonlocal measured_calls
        measured_calls += 1
        return original_measure(renderer, case, tokenizer)

    monkeypatch.setattr(v02_corpus.VerifiedRenderer, "measure", count_measure)
    capsule = v02_corpus.seal_training_capsule(
        root=fixture["root"],
        plan_path=fixture["plan"],
        receipt_path=fixture["receipt"],
        exclusions_path=fixture["exclusions"],
        output_parent=fixture["output_parent"],
        artifact_id="training-capsule",
        sealer_inputs_path=fixture["inputs"],
    )
    assert measured_calls == 1600
    descriptor, rows = v02_corpus.verify_training_capsule(
        capsule,
        root=fixture["root"],
        plan_path=fixture["plan"],
        receipt_path=fixture["receipt"],
        exclusions_path=fixture["exclusions"],
        sealer_inputs_path=fixture["inputs"],
    )
    assert descriptor["counts"]["accepted"] == 1600
    assert descriptor["counts"]["complete_bilingual_pairs"] == 300
    assert len(rows) == 1600
    assert stat.S_IMODE(capsule.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(capsule.stat().st_mode) == 0o600
    assert stat.S_IMODE((capsule.parent / "rows.jsonl").stat().st_mode) == 0o600
    under_floor = dict(descriptor["counts"])
    under_floor["accepted"] = 1199
    with pytest.raises(ValueError, match=r"floor|inconsistent"):
        v02_corpus._validate_capsule_counts(under_floor)
    leaked = [rows[0], rows[0] | {"split": "internal_dev", "identity_id": "leaked-id"}]
    with pytest.raises(ValueError, match=r"duplicate|leaked|leakage"):
        v02_corpus._validate_row_disjointness(
            leaked,
            {"identity": set(), "state_question": set(), "combined_content": set()},
        )
    tampered = rows[0] | {"gold_index": 1}
    row_path = capsule.parent / "rows.jsonl"
    row_path.write_bytes(canonical_json_bytes(tampered) + b"\n")
    os.chmod(row_path, 0o600)
    with pytest.raises(ValueError, match=r"digest|evidence|rows"):
        v02_corpus.verify_training_capsule(
            capsule,
            root=fixture["root"],
            plan_path=fixture["plan"],
            receipt_path=fixture["receipt"],
            exclusions_path=fixture["exclusions"],
            sealer_inputs_path=fixture["inputs"],
        )


def test_training_capsule_verifier_rejects_root_mode_and_reservation_tampering(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fixture = _sealer_fixture(monkeypatch, tmp_path)
    capsule = v02_corpus.seal_training_capsule(
        root=fixture["root"],
        plan_path=fixture["plan"],
        receipt_path=fixture["receipt"],
        exclusions_path=fixture["exclusions"],
        output_parent=fixture["output_parent"],
        artifact_id="training-capsule",
        sealer_inputs_path=fixture["inputs"],
    )
    copied_receipt = fixture["root"] / "copied-receipt.json"
    v02_corpus._atomic_write(copied_receipt, fixture["receipt"].read_bytes())
    with pytest.raises(ValueError, match="logical artifact ID"):
        v02_corpus.verify_training_capsule(
            capsule,
            root=fixture["root"],
            plan_path=fixture["plan"],
            receipt_path=copied_receipt,
            exclusions_path=fixture["exclusions"],
            sealer_inputs_path=fixture["inputs"],
        )
    os.chmod(fixture["root"], 0o755)
    with pytest.raises(ValueError, match="0700"):
        v02_corpus.verify_training_capsule(
            capsule,
            root=fixture["root"],
            plan_path=fixture["plan"],
            receipt_path=fixture["receipt"],
            exclusions_path=fixture["exclusions"],
            sealer_inputs_path=fixture["inputs"],
        )
    os.chmod(fixture["root"], 0o700)
    current_uid = os.geteuid()
    monkeypatch.setattr(os, "geteuid", lambda: current_uid + 1)
    with pytest.raises(ValueError, match="owned"):
        v02_corpus.verify_training_capsule(
            capsule,
            root=fixture["root"],
            plan_path=fixture["plan"],
            receipt_path=fixture["receipt"],
            exclusions_path=fixture["exclusions"],
            sealer_inputs_path=fixture["inputs"],
        )
    monkeypatch.setattr(os, "geteuid", lambda: current_uid)
    reservation = next((fixture["root"] / "role-reservations").glob("*.json"))
    data = json.loads(reservation.read_bytes())
    data["grant_digest"] = "0" * 64
    reservation.write_bytes(canonical_json_bytes(data) + b"\n")
    os.chmod(reservation, 0o600)
    with pytest.raises(ValueError, match="reservation"):
        v02_corpus.verify_training_capsule(
            capsule,
            root=fixture["root"],
            plan_path=fixture["plan"],
            receipt_path=fixture["receipt"],
            exclusions_path=fixture["exclusions"],
            sealer_inputs_path=fixture["inputs"],
        )


def test_training_capsule_sealer_rejects_synchronized_source_and_output_storage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fixture = _sealer_fixture(monkeypatch, tmp_path)
    source_root = fixture["root"]
    synchronized_root = tmp_path / "CloudStorage" / source_root.name
    _mkdir_0700(synchronized_root.parent)
    source_root.rename(synchronized_root)
    relocated = {
        key: synchronized_root / value.relative_to(source_root)
        for key, value in fixture.items()
        if key != "output_parent"
    }
    with pytest.raises(ValueError, match="synchronized ancestry"):
        v02_corpus.seal_training_capsule(
            root=relocated["root"],
            plan_path=relocated["plan"],
            receipt_path=relocated["receipt"],
            exclusions_path=relocated["exclusions"],
            output_parent=fixture["output_parent"],
            artifact_id="training-capsule",
            sealer_inputs_path=relocated["inputs"],
        )

    fixture = _sealer_fixture(monkeypatch, tmp_path / "independent-source")
    synchronized_output = tmp_path / "Dropbox" / "capsules"
    _mkdir_0700(synchronized_output)
    with pytest.raises(ValueError, match="synchronized ancestry"):
        v02_corpus.seal_training_capsule(
            root=fixture["root"],
            plan_path=fixture["plan"],
            receipt_path=fixture["receipt"],
            exclusions_path=fixture["exclusions"],
            output_parent=synchronized_output,
            artifact_id="training-capsule",
            sealer_inputs_path=fixture["inputs"],
        )


def test_training_capsule_verifier_rejects_synchronized_capsule_storage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fixture = _sealer_fixture(monkeypatch, tmp_path)
    capsule = v02_corpus.seal_training_capsule(
        root=fixture["root"],
        plan_path=fixture["plan"],
        receipt_path=fixture["receipt"],
        exclusions_path=fixture["exclusions"],
        output_parent=fixture["output_parent"],
        artifact_id="training-capsule",
        sealer_inputs_path=fixture["inputs"],
    )
    synchronized_output = tmp_path / "Nextcloud" / "capsules"
    _mkdir_0700(synchronized_output)
    relocated_root = synchronized_output / capsule.parent.name
    capsule.parent.rename(relocated_root)
    with pytest.raises(ValueError, match="synchronized ancestry"):
        v02_corpus.verify_training_capsule(
            relocated_root / capsule.name,
            root=fixture["root"],
            plan_path=fixture["plan"],
            receipt_path=fixture["receipt"],
            exclusions_path=fixture["exclusions"],
            sealer_inputs_path=fixture["inputs"],
        )


def test_training_capsule_exclusion_aliases_are_independent_and_enforced(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fixture = _sealer_fixture(monkeypatch, tmp_path)
    exclusions_path = fixture["exclusions"]
    exclusions = json.loads(exclusions_path.read_bytes())
    row = {
        "identity_id": "alias-only",
        "state": "Fictional alias overlap state",
        "instruction": "Which fictional action applies?",
        "options": [{"id": "option_0", "description": "Fictional action"}],
        "split": "train",
    }
    fingerprint = state_question_fingerprint(row["state"], row["instruction"])
    alias_values = [f"{number:064x}" for number in range(1, 1304)] + [fingerprint]
    assert len(set(alias_values)) == 1304
    exclusions["literal_serialized_state_alias_sets"]["state_question"] = sorted(alias_values)

    def refresh_exclusion_digest() -> dict[str, Any]:
        exclusions_path.unlink()
        _write_private_canonical(exclusions_path, exclusions)
        digest = v02_corpus._sha256(exclusions_path.read_bytes())
        monkeypatch.setattr(v02_corpus, "HISTORICAL_EXCLUSIONS_SHA256", digest)
        inputs = json.loads(fixture["inputs"].read_bytes())
        inputs["exclusions_sha256"] = digest
        return cast(dict[str, Any], inputs)

    inputs = refresh_exclusion_digest()
    validated = v02_corpus._validate_historical_exclusions(exclusions_path, inputs)
    assert fingerprint in validated["state_question"]
    with pytest.raises(ValueError, match="historical exclusion overlap on state_question"):
        v02_corpus._validate_row_disjointness([row], validated)

    exclusions["sets"]["state_question"] = exclusions["sets"]["state_question"][:-1]
    inputs = refresh_exclusion_digest()
    with pytest.raises(ValueError, match="historical exclusion hashes are closed"):
        v02_corpus._validate_historical_exclusions(exclusions_path, inputs)


def test_historical_raw_pinned_metadata_allows_pretty_json_only_after_hash_check(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fixture = _sealer_fixture(monkeypatch, tmp_path)
    phase_a_path = fixture["root"] / "phase-a-receipt.json"
    exclusions_path = fixture["exclusions"]
    phase_a = json.loads(phase_a_path.read_bytes())
    phase_a_path.unlink()
    _write_private_pretty_json(phase_a_path, phase_a)
    phase_a_hash = v02_corpus._sha256(phase_a_path.read_bytes())
    exclusions = json.loads(exclusions_path.read_bytes())
    exclusions["phase_a_receipt_sha256"] = phase_a_hash
    exclusions_path.unlink()
    _write_private_pretty_json(exclusions_path, exclusions)
    exclusions_hash = v02_corpus._sha256(exclusions_path.read_bytes())
    monkeypatch.setattr(v02_corpus, "HISTORICAL_PHASE_A_RECEIPT_SHA256", phase_a_hash)
    monkeypatch.setattr(v02_corpus, "HISTORICAL_EXCLUSIONS_SHA256", exclusions_hash)
    inputs = json.loads(fixture["inputs"].read_bytes())
    inputs["historical_phase_a_receipt_sha256"] = phase_a_hash
    inputs["exclusions_sha256"] = exclusions_hash

    v02_corpus._verify_phase_a_receipt(fixture["root"], inputs)
    validated = v02_corpus._validate_historical_exclusions(exclusions_path, inputs)
    assert {name: len(values) for name, values in validated.items()} == {
        "identity": 1304,
        "state_question": 1304,
        "combined_content": 1304,
    }

    changed_inputs = dict(inputs)
    changed_inputs["exclusions_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="raw digest mismatch"):
        v02_corpus._validate_historical_exclusions(exclusions_path, changed_inputs)

    duplicate = b'{"status":"BLOCKED_DATA_RIGHTS","status":"BLOCKED_DATA_RIGHTS"}\n'
    phase_a_path.unlink()
    v02_corpus._atomic_write(phase_a_path, duplicate)
    duplicate_hash = v02_corpus._sha256(duplicate)
    monkeypatch.setattr(v02_corpus, "HISTORICAL_PHASE_A_RECEIPT_SHA256", duplicate_hash)
    duplicate_inputs = dict(inputs)
    duplicate_inputs["historical_phase_a_receipt_sha256"] = duplicate_hash
    with pytest.raises(ValueError, match="not valid closed JSON"):
        v02_corpus._verify_phase_a_receipt(fixture["root"], duplicate_inputs)


def test_training_capsule_runtime_lock_requires_actual_cohort_shape() -> None:
    lock = _sealer_lock([], "9" * 40)
    v02_corpus._validate_runtime_lock(lock, "9" * 40)
    malformed = json.loads(json.dumps(lock))
    malformed["cohort_bindings"]["training"].pop("micro")
    with pytest.raises(ValueError, match="cohort bindings"):
        v02_corpus._validate_runtime_lock(malformed, "9" * 40)


def _pilot_training_slot(plan: Mapping[str, Any], *, pilot: bool) -> dict[str, Any]:
    slots = cast(list[dict[str, Any]], plan["slots"])
    return next(slot for slot in slots if bool(slot["in_pilot"]) is pilot)


def _pilot_pair(plan: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for slot in cast(list[dict[str, Any]], plan["slots"]):
        pair_id = slot.get("bilingual_pair_id")
        if slot.get("in_pilot") is True and isinstance(pair_id, str):
            groups.setdefault(pair_id, []).append(slot)
    pair = next(group for group in groups.values() if len(group) == 2)
    return pair[0], pair[1]


def _training_content(
    slot: Mapping[str, Any],
    *,
    state: str = "Fictional target label semantic state",
    fictionality: bool = True,
) -> str:
    case = _case(cast(int, slot["option_count"]), state=state)
    payload: dict[str, Any] = {
        **case,
        "answer": f"option_{slot['gold_position']}",
        "construction": _construction(slot, state, f"option_{slot['gold_position']}"),
        "semantic": {
            "scenario_code": slot["scenario_code"],
            "criterion_roles": slot["criterion_roles"],
        },
        **_judgments(),
    }
    payload["fictionality_valid"] = fictionality
    return json.dumps(payload)


def _reviewer_content(slot: Mapping[str, Any]) -> str:
    return json.dumps(
        {
            "answer": f"option_{slot['gold_position']}",
            "semantic": {
                "scenario_code": slot["scenario_code"],
                "criterion_roles": slot["criterion_roles"],
            },
            **_judgments(),
        }
    )


def _sealed_author_content(identity: Mapping[str, Any]) -> str:
    case = _case(
        cast(int, identity["option_count"]),
        state=f"Fictional sealed state {identity['identity_id']}",
    )
    return json.dumps({**case, "target": "option_0", **_judgments()})


def _message_context(call: Mapping[str, Any]) -> dict[str, Any]:
    body = json.loads(cast(bytes, call["body"]))
    content = cast(str, body["messages"][1]["content"])
    context = json.loads(content.split("\n", 1)[1])
    assert isinstance(context, dict)
    return cast(dict[str, Any], context)


def _chat_body(call: Mapping[str, Any]) -> dict[str, Any]:
    body = json.loads(cast(bytes, call["body"]))
    assert isinstance(body, dict)
    return cast(dict[str, Any], body)


def test_c2_canonical_preimages_reproduce_authorized_digests() -> None:
    assert v02_corpus.candidate_rendering_digest() == (
        "0f5592b54f096ac0328b45d69e5a74b9cb579a804f2349fc4ceea9b1f8a40e40"
    )
    assert v02_corpus.kev_rendering_function_digest() == (
        "eca2a60af37c539c984e89cf920c53e8d1c93cff6e980dea1a24dd520f86e169"
    )
    preimage = v02_corpus.candidate_renderer_preimage()
    contract = v02_corpus.kev_renderer_contract()
    assert preimage["source_repository"] == "https://github.com/jaredpalmer/kev"
    assert preimage["source_revision"] == "9c41005b2180347c3c646dfc9e50c4428483ec6b"
    assert preimage["source_sha256"] == v02_corpus.PINNED_RENDERER_SOURCE_SHA256
    assert preimage["parameters"] == {
        "head_dim": 256,
        "lora_targets": "all",
        "max_branch": 1024,
        "max_packed": 2048,
        "max_state": 384,
        "option_isolation": False,
        "special_embeddings": False,
        "strict": True,
    }
    assert contract["schema_version"] == "v02-kev-renderer.v2"
    assert contract["model"] == "jaredpalmer/kev-4b"
    assert contract["model_revision"] == v02_corpus.PINNED_ADAPTER_REVISION
    assert contract["base_model"] == "Qwen/Qwen3.5-4B-Base"
    assert contract["base_model_revision"] == v02_corpus.PINNED_TOKENIZER_REVISION
    assert contract["max_rendered_input_tokens"] == 512
    assert contract["truncation_disabled"] is True
    assert contract["renderer"] == preimage


def test_c2_direct_wrapper_construction_and_tamper_are_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="verified runtime wrapper is required"):
        v02_corpus.VerifiedRenderer(
            lambda *_args, **_kwargs: None,
            lambda *_args, **_kwargs: None,
            lambda: {},
            "ab" * 32,
        )
    snapshot, inventory = _write_snapshot(tmp_path)
    with pytest.raises(ValueError, match="verified runtime wrapper is required"):
        v02_corpus.VerifiedTokenizer(
            _FixedTokenizer(), snapshot, inventory, {"tokenizer.json": "ab" * 32}
        )
    renderer, tokenizer, _inventory = _load_runtime(monkeypatch, tmp_path / "loaded")
    renderer._encode = lambda *_args, **_kwargs: None
    with pytest.raises(ValueError, match="drifted"):
        renderer.revalidate()
    tokenizer._inner = _FixedTokenizer()
    with pytest.raises(ValueError, match="drift"):
        tokenizer.revalidate()


def test_c2_renderer_source_identity_and_extraction_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _RENDERER_SOURCE.encode()
    with pytest.raises(ValueError, match="renderer source identity mismatch"):
        v02_corpus.load_pinned_renderer(raw)
    ignored_import = "import not_a_real_torch_module\n" + _RENDERER_SOURCE
    loaded = v02_corpus.load_pinned_renderer(_pin_renderer(monkeypatch, ignored_import))
    encoded = loaded._encode(
        _FixedTokenizer(),
        {"state": "ab", "questions": [{"instr": "q", "options": ["o"], "label": 4}]},
        max_state=384,
        max_branch=1024,
        strict=True,
        option_isolation=False,
    )
    assert encoded["labels"] == [0]
    variants = {
        _RENDERER_SOURCE.replace("OPT_DECIDE = 1\n", ""): "binding",
        _RENDERER_SOURCE.replace(
            '    state = record["state"]\n', '    import torch\n    state = record["state"]\n'
        ): "binding",
        _RENDERER_SOURCE.replace("<|fim_suffix|>", "<|fim_pad|>"): "markers",
        _RENDERER_SOURCE.replace(
            """def user_tokens(tokenizer, text):
    rewritten = text
    for marker in SPECIAL:
        name = marker[2:-2]
        rewritten = rewritten.replace(marker, "<\u00a6" + name + "\u00a6>")
    return tokenizer(rewritten, add_special_tokens=False).input_ids
""",
            """def user_tokens(tokenizer, text):
    return tokenizer(text, add_special_tokens=False).input_ids
""",
        ): "markers",
        _RENDERER_SOURCE.replace("[OPT_NONE]", "[OPT_DECIDE]"): "neutral",
    }
    for source, match in variants.items():
        with pytest.raises(ValueError, match=match):
            v02_corpus.load_pinned_renderer(_pin_renderer(monkeypatch, source))


def test_c2_renderer_bounds_truncation_and_local_gates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    renderer, tokenizer, _inventory = _load_runtime(monkeypatch, tmp_path)
    identity = {"option_count": 2}

    def gates(
        count: int, *, state: str = "Fictional target label semantic state"
    ) -> dict[str, bool]:
        def load_tokenizer(snapshot: Path) -> _FixedTokenizer:
            del snapshot
            return _FixedTokenizer(count)

        monkeypatch.setattr(v02_corpus, "_load_transformers_tokenizer", load_tokenizer)
        directory = tmp_path / f"count-{count}-{len(state)}"
        directory.mkdir()
        snapshot, inventory = _write_snapshot(directory)
        measured = v02_corpus.load_verified_tokenizer(snapshot, inventory)
        return v02_corpus.local_case_gates(_case_any(2, state=state), identity, renderer, measured)

    accepted = gates(512)
    assert accepted["renderer_valid"] is True and accepted["length_valid"] is True
    boundary = gates(513)
    assert boundary["renderer_valid"] is True and boundary["length_valid"] is False
    packed = gates(2048)
    assert packed["renderer_valid"] is True and packed["length_valid"] is False
    overflow = gates(2049)
    assert overflow["renderer_valid"] is False and overflow["length_valid"] is False
    truncated = gates(4, state="TRUNCATE fictional state")
    assert truncated["renderer_valid"] is False and truncated["length_valid"] is False
    schema_only = v02_corpus.local_case_gates(_case_any(1), identity, renderer, tokenizer)
    assert schema_only["schema_valid"] is False
    assert schema_only["fictionality_valid"] is True
    assert schema_only["exclusive_options_valid"] is True
    assert schema_only["ambiguity_free"] is True
    leaked_state = _case_any(2, state="Reach person@example.com today")
    assert (
        v02_corpus.local_case_gates(leaked_state, identity, renderer, tokenizer)["privacy_valid"]
        is False
    )
    leaked_question = _case_any(2)
    leaked_question["question"] = "See https://example.test/path"
    assert (
        v02_corpus.local_case_gates(leaked_question, identity, renderer, tokenizer)["privacy_valid"]
        is False
    )
    for index in (0, 1):
        leaked_option = _case_any(2)
        cast(list[dict[str, Any]], leaked_option["options"])[index]["description"] = (
            "token secret=value"
        )
        result = v02_corpus.local_case_gates(leaked_option, identity, renderer, tokenizer)
        assert result["schema_valid"] is True and result["privacy_valid"] is False
    leaked_id = _case_any(2)
    cast(list[dict[str, Any]], leaked_id["options"])[0]["id"] = "@someone"
    assert v02_corpus._privacy_clear(leaked_id) is False
    left = _case_any(2, state="Same fictional facts")
    right = _case_any(2, state="same   fictional   facts")
    assert state_question_fingerprint(
        left["state"], left["question"]
    ) == state_question_fingerprint(right["state"], right["question"])
    assert (
        v02_corpus.local_case_gates(right, identity, renderer, tokenizer, duplicate_history=[left])[
            "duplicate_valid"
        ]
        is False
    )
    distinct = _case_any(2, state="Entirely different fictional facts")
    assert (
        v02_corpus.local_case_gates(
            distinct, identity, renderer, tokenizer, duplicate_history=[left]
        )["duplicate_valid"]
        is True
    )
    options = [{"id": "b", "description": "beta"}, {"id": "a", "description": "alpha"}]
    flipped = [{"description": "alpha"}, {"id": "zzz", "description": "beta"}]
    assert combined_content_fingerprint("S", "Q", options) == combined_content_fingerprint(
        "S", "Q", flipped
    )
    again = v02_corpus.local_case_gates(
        right, identity, renderer, tokenizer, duplicate_history=[left]
    )
    assert again["duplicate_valid"] is False


def test_c2_tokenizer_loader_is_local_only_and_inventory_is_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: dict[str, object] = {}

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(path: str, **kwargs: object) -> _FixedTokenizer:
            seen["path"] = path
            seen["kwargs"] = dict(kwargs)
            return _FixedTokenizer()

    module = ModuleType("transformers")
    vars(module)["AutoTokenizer"] = AutoTokenizer
    monkeypatch.setitem(sys.modules, "transformers", module)
    snapshot, inventory = _write_snapshot(tmp_path)
    loaded = v02_corpus.load_verified_tokenizer(snapshot, inventory)
    assert seen["kwargs"] == {"local_files_only": True, "trust_remote_code": False}
    assert seen["path"] == os.fspath(snapshot)
    (snapshot / "tokenizer.json").write_bytes(b"changed-bytes")
    os.chmod(snapshot / "tokenizer.json", 0o600)
    with pytest.raises(ValueError, match="drift"):
        loaded.revalidate()
    fresh, fresh_inventory = _write_snapshot(tmp_path / "fresh", b"other-reviewed-bytes")
    extra = fresh / "extra.json"
    extra.write_bytes(b"x")
    os.chmod(extra, 0o600)
    with pytest.raises(ValueError, match="closed"):
        v02_corpus.load_verified_tokenizer(fresh, fresh_inventory)
    extra.unlink()
    wrong = [{"filename": "tokenizer.json", "sha256": "ab" * 32}]
    with pytest.raises(ValueError, match="drift"):
        v02_corpus.load_verified_tokenizer(fresh, wrong)


def test_c2_grant_and_runtime_evidence_fail_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _mkdir_0700(tmp_path)
    renderer, tokenizer, inventory = _load_runtime(monkeypatch, tmp_path / "runtime")
    licenses = _license_bytes()
    grant = _grant(inventory)
    grant_digest = v02_corpus.validate_environment_grant(grant, licenses)
    evidence = _runtime("training_author", grant, grant_digest, tokenizer.inventory_digest)
    assert v02_corpus.validate_runtime_evidence(evidence, "training_author", grant_digest) == (
        v02_corpus._sha256(canonical_json_bytes(cast(JsonValue, evidence)))
    )
    forty = dict(evidence)
    forty["gpu_vram_gib"] = 40
    v02_corpus.validate_runtime_evidence(forty, "training_author", grant_digest)
    mutations: list[tuple[dict[str, Any], dict[str, bytes], str]] = []
    missing = dict(grant)
    missing.pop("region")
    mutations.append((missing, licenses, "closed"))
    extra = dict(grant)
    extra["unexpected"] = "x"
    mutations.append((extra, licenses, "closed"))
    nonbool = dict(grant)
    nonbool["authorize_training"] = "yes"
    mutations.append((nonbool, licenses, "not boolean"))
    denied = dict(grant)
    denied["prohibit_raw_publication"] = False
    mutations.append((denied, licenses, "value mismatch"))
    unknown = dict(grant)
    unknown["region"] = "unknown"
    mutations.append((unknown, licenses, "value mismatch"))
    absent = dict(licenses)
    absent["training_author"] = b""
    mutations.append((dict(grant), absent, "absent"))
    partial = dict(licenses)
    partial.pop("sealed_adjudicator")
    mutations.append((dict(grant), partial, "closed"))
    for mutated_grant, mutated_licenses, match in mutations:
        with pytest.raises(ValueError, match=match):
            v02_corpus.validate_environment_grant(mutated_grant, mutated_licenses)
    runtime_mutations: list[tuple[dict[str, Any], str]] = []
    low = dict(evidence)
    low["gpu_vram_gib"] = 39
    runtime_mutations.append((low, "value mismatch"))
    fallback = dict(evidence)
    fallback["gpu_class"] = "CUDA-24GiB"
    runtime_mutations.append((fallback, "value mismatch"))
    offload = dict(evidence)
    offload["cpu_offload"] = True
    runtime_mutations.append((offload, "value mismatch"))
    offload_text = dict(evidence)
    offload_text["cpu_offload"] = "false"
    runtime_mutations.append((offload_text, "not boolean"))
    missing_runtime = dict(evidence)
    missing_runtime.pop("quantization")
    runtime_mutations.append((missing_runtime, "closed"))
    extra_runtime = dict(evidence)
    extra_runtime["driver"] = "moving"
    runtime_mutations.append((extra_runtime, "closed"))
    renamed = dict(evidence)
    renamed["model_identity"] = "other-model@revision"
    runtime_mutations.append((renamed, "value mismatch"))
    for mutated, match in runtime_mutations:
        with pytest.raises(ValueError, match=match):
            v02_corpus.validate_runtime_evidence(mutated, "training_author", grant_digest)
    plan = v02_corpus._generate_training_plan()
    ledger = v02_corpus.OfflineLedger(plan, tmp_path)
    slot = _pilot_training_slot(plan, pilot=True)
    other_inventory = [{"filename": "tokenizer.json", "sha256": "ab" * 32}]
    drifted_grant = _grant(other_inventory)
    with pytest.raises(ValueError, match="runtime evidence value mismatch"):
        _run_role(
            ledger,
            cast(str, slot["slot_id"]),
            "training_author",
            drifted_grant,
            licenses,
            renderer,
            tokenizer,
            _reject_network,
        )
    assert ledger.events() == []
    del renderer


def test_c2_six_roles_dispatch_once_and_resume_without_network(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    training_root = tmp_path / "training"
    sealed_root = tmp_path / "sealed"
    _mkdir_0700(training_root)
    _mkdir_0700(sealed_root)
    renderer, tokenizer, inventory = _load_runtime(monkeypatch, tmp_path / "runtime")
    grant = _grant(inventory)
    licenses = _license_bytes()
    training = v02_corpus._generate_training_plan()
    slot = _pilot_training_slot(training, pilot=True)
    identity_id = cast(str, slot["slot_id"])
    ledger = v02_corpus.OfflineLedger(training, training_root)
    author = _Loopback(v02_corpus.MODEL_ROLES["training_author"]["model"], _training_content(slot))
    first = _run_role(
        ledger, identity_id, "training_author", grant, licenses, renderer, tokenizer, author
    )
    assert first == {"status": "committed", "dispatch": True, "transition": "training_author"}
    assert len(author.calls) == 2 and len(author.posts()) == 1
    chat = _chat_body(author.posts()[0])
    assert chat["temperature"] == 0
    assert chat["max_tokens"] == 2048
    assert chat["seed"] == v02_corpus.SEED_TRAINING
    assert chat["response_format"] == v02_corpus.native_response_format("training_author", slot)
    assert chat["chat_template_kwargs"] == {"enable_thinking": False}
    assert chat["model"] == v02_corpus.MODEL_ROLES["training_author"]["model"]
    assert "no retry" in cast(str, chat["messages"][0]["content"])
    resumed = _run_role(
        ledger,
        identity_id,
        "training_author",
        grant,
        licenses,
        renderer,
        tokenizer,
        _reject_network,
    )
    assert resumed["dispatch"] is False and resumed["status"] == "resumed"
    reservation = training_root / "role-reservations"
    reservation_name = v02_corpus._sha256(f"{identity_id}:training_author".encode()) + ".json"
    reservation_path = reservation / reservation_name
    assert reservation_path.is_file()
    assert stat.S_IMODE(reservation.stat().st_mode) == 0o700
    assert stat.S_IMODE(reservation_path.stat().st_mode) == 0o600
    assert reservation_path.parent == training_root / "role-reservations"
    assert reservation_path.resolve().parent != (training_root / "ledger-events").resolve()
    reviewer = _Loopback(
        v02_corpus.MODEL_ROLES["independent_reviewer"]["model"], _reviewer_content(slot)
    )
    reviewed = _run_role(
        ledger,
        identity_id,
        "independent_reviewer",
        grant,
        licenses,
        renderer,
        tokenizer,
        reviewer,
    )
    assert reviewed["dispatch"] is True and len(reviewer.posts()) == 1
    context = _message_context(reviewer.posts()[0])
    assert "gold_position" not in context and "target" not in context and "answer" not in context
    assert context["case"]["state"] == "Fictional target label semantic state"
    author_envelope = cast(
        dict[str, Any], ledger.by_identity(identity_id)["training_author"]["envelope"]
    )
    reviewer_envelope = cast(
        dict[str, Any], ledger.by_identity(identity_id)["training_reviewer"]["envelope"]
    )
    assert author_envelope["content_digest"] == reviewer_envelope["content_digest"]
    assert author_envelope["gates"]["schema_valid"] is True
    sealed = v02_corpus._generate_sealed_plan()
    identity = cast(list[dict[str, Any]], sealed["pilot_prefix"]["identities"])[0]
    sealed_id = cast(str, identity["identity_id"])
    sealed_ledger = v02_corpus.OfflineLedger(sealed, sealed_root)
    roles = (
        ("sealed_author", _sealed_author_content(identity)),
        ("sealed_annotator_a", json.dumps({"label": "option_0", **_judgments()})),
        ("sealed_annotator_b", json.dumps({"label": "option_1", **_judgments()})),
        ("sealed_adjudicator", json.dumps({"choice": "option_0", **_judgments()})),
    )
    for role, content in roles:
        transport = _Loopback(v02_corpus.MODEL_ROLES[role]["model"], content)
        result = _run_role(
            sealed_ledger, sealed_id, role, grant, licenses, renderer, tokenizer, transport
        )
        assert result["dispatch"] is True and len(transport.posts()) == 1
        payload = _chat_body(transport.posts()[0])
        assert payload["seed"] == v02_corpus.SEED_SEALED
        labels = ["option_0", "option_1"] if role == "sealed_adjudicator" else None
        assert payload["response_format"] == v02_corpus.native_response_format(
            role, identity, labels
        )
        if role == "sealed_adjudicator":
            adjudication = _message_context(transport.posts()[0])
            assert set(cast(list[str], adjudication["labels"])) == {"option_0", "option_1"}
            assert "target" not in adjudication
    choice = cast(
        dict[str, Any], sealed_ledger.by_identity(sealed_id)["sealed_adjudicator"]["envelope"]
    )
    assert choice["choice"] == "option_0"
    assert v02_corpus.reduce_training(training, ledger.events())["denominator"] == 1600
    assert v02_corpus.reduce_sealed(sealed, sealed_ledger.events())["denominator"] == 140


def test_c2_transport_errors_are_single_attempt_failures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _mkdir_0700(tmp_path)
    renderer, tokenizer, inventory = _load_runtime(monkeypatch, tmp_path / "runtime")
    grant = _grant(inventory)
    licenses = _license_bytes()
    plan = v02_corpus._generate_training_plan()
    micro_ids = set(v02_corpus.grounding_micro_pilot(plan))
    slots = [
        next(
            slot
            for slot in cast(list[dict[str, Any]], plan["slots"])
            if slot["slot_id"] in micro_ids and slot["option_count"] == option_count
        )
        for option_count in v02_corpus.OPTION_COUNTS
    ]
    ledger = v02_corpus.OfflineLedger(plan, tmp_path)
    timeout = _Loopback(v02_corpus.MODEL_ROLES["training_author"]["model"], "{}", fail="timeout")
    identity_id = cast(str, slots[0]["slot_id"])
    failed = _run_role(
        ledger, identity_id, "training_author", grant, licenses, renderer, tokenizer, timeout
    )
    assert failed["dispatch"] is False
    assert failed["transition"] == "training_author_failure"
    envelope = cast(
        dict[str, Any], ledger.by_identity(identity_id)["training_author_failure"]["envelope"]
    )
    assert envelope["error_code"] == "model_error"
    assert set(envelope) == {
        "schema_version",
        "role",
        "model",
        "lane",
        "identity_id",
        "error_code",
    }
    assert len(timeout.posts()) == 0
    again = _run_role(
        ledger,
        identity_id,
        "training_author",
        grant,
        licenses,
        renderer,
        tokenizer,
        _reject_network,
    )
    assert again["dispatch"] is False and again["status"] == "resumed"
    redirect = _Loopback(v02_corpus.MODEL_ROLES["training_author"]["model"], "{}", fail="redirect")
    redirect_id = cast(str, slots[1]["slot_id"])
    redirected = _run_role(
        ledger, redirect_id, "training_author", grant, licenses, renderer, tokenizer, redirect
    )
    assert redirected["dispatch"] is False
    assert (
        cast(
            dict[str, Any], ledger.by_identity(redirect_id)["training_author_failure"]["envelope"]
        )["error_code"]
        == "model_error"
    )
    invalid = _Loopback(v02_corpus.MODEL_ROLES["training_author"]["model"], "not-json")
    invalid_id = cast(str, slots[2]["slot_id"])
    parsed = _run_role(
        ledger, invalid_id, "training_author", grant, licenses, renderer, tokenizer, invalid
    )
    assert parsed["dispatch"] is True and len(invalid.posts()) == 1
    assert (
        cast(dict[str, Any], ledger.by_identity(invalid_id)["training_author_failure"]["envelope"])[
            "error_code"
        ]
        == "invalid_output"
    )
    retry = _Loopback(
        v02_corpus.MODEL_ROLES["training_author"]["model"], _training_content(slots[2])
    )
    resumed = _run_role(
        ledger, invalid_id, "training_author", grant, licenses, renderer, tokenizer, retry
    )
    assert resumed["dispatch"] is False and retry.calls == []
    for base in ("http://example.test", "https://127.0.0.1:9", "http://127.0.0.1:9/v1"):
        with pytest.raises(ValueError, match="loopback"):
            v02_corpus.execute_role(
                ledger,
                cast(str, slots[3]["slot_id"]),
                "training_author",
                _runtime(
                    "training_author",
                    grant,
                    v02_corpus.validate_environment_grant(grant, licenses),
                    tokenizer.inventory_digest,
                ),
                grant,
                licenses,
                base,
                "test-bearer",
                renderer,
                tokenizer,
                transport=_reject_network,
            )
    assert "training_author" not in ledger.by_identity(cast(str, slots[3]["slot_id"]))
    assert v02_corpus.reduce_training(plan, ledger.events())["denominator"] == 1600


def test_c2_role_errors_cover_every_model_and_skip_downstream_gpu(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    training_root = tmp_path / "training"
    sealed_root = tmp_path / "sealed"
    _mkdir_0700(training_root)
    _mkdir_0700(sealed_root)
    renderer, tokenizer, inventory = _load_runtime(monkeypatch, tmp_path / "runtime")
    grant = _grant(inventory)
    licenses = _license_bytes()
    training = v02_corpus._generate_training_plan()
    slot = _pilot_training_slot(training, pilot=True)
    identity_id = cast(str, slot["slot_id"])
    ledger = v02_corpus.OfflineLedger(training, training_root)
    leaked = _Loopback(
        v02_corpus.MODEL_ROLES["training_author"]["model"],
        _training_content(slot, state="Reach person@example.com today"),
    )
    _run_role(ledger, identity_id, "training_author", grant, licenses, renderer, tokenizer, leaked)
    author_gates = cast(
        dict[str, Any], ledger.by_identity(identity_id)["training_author"]["envelope"]
    )["gates"]
    assert author_gates["privacy_valid"] is False and author_gates["schema_valid"] is True
    blocked = _run_role(
        ledger,
        identity_id,
        "independent_reviewer",
        grant,
        licenses,
        renderer,
        tokenizer,
        _reject_network,
    )
    assert blocked == {
        "status": "committed",
        "dispatch": False,
        "transition": "training_reviewer_failure",
    }
    failure = cast(
        dict[str, Any], ledger.by_identity(identity_id)["training_reviewer_failure"]["envelope"]
    )
    assert failure["error_code"] == "local_gate_failure"
    sealed = v02_corpus._generate_sealed_plan()
    identities = cast(list[dict[str, Any]], sealed["pilot_prefix"]["identities"])
    sealed_ledger = v02_corpus.OfflineLedger(sealed, sealed_root)
    specs = (
        (identities[0], "sealed_author", None),
        (identities[3], "sealed_annotator_a", "sealed_author"),
        (identities[6], "sealed_annotator_b", "sealed_author"),
        (identities[9], "sealed_adjudicator", "both"),
    )
    for identity, role, prior in specs:
        sealed_id = cast(str, identity["identity_id"])
        if prior is not None:
            prepared = _Loopback(
                v02_corpus.MODEL_ROLES["sealed_author"]["model"], _sealed_author_content(identity)
            )
            _run_role(
                sealed_ledger,
                sealed_id,
                "sealed_author",
                grant,
                licenses,
                renderer,
                tokenizer,
                prepared,
            )
        if prior == "both":
            for annotator, label in (
                ("sealed_annotator_a", "option_0"),
                ("sealed_annotator_b", "option_1"),
            ):
                prepared = _Loopback(
                    v02_corpus.MODEL_ROLES[annotator]["model"],
                    json.dumps({"label": label, **_judgments()}),
                )
                _run_role(
                    sealed_ledger,
                    sealed_id,
                    annotator,
                    grant,
                    licenses,
                    renderer,
                    tokenizer,
                    prepared,
                )
        transport = _Loopback(v02_corpus.MODEL_ROLES[role]["model"], "{}", fail="timeout")
        result = _run_role(
            sealed_ledger, sealed_id, role, grant, licenses, renderer, tokenizer, transport
        )
        assert result["dispatch"] is False and transport.posts() == []
        transition = f"{v02_corpus._ROLE_TRANSITIONS[role]}_failure"
        assert result["transition"] == transition
        assert (
            cast(dict[str, Any], sealed_ledger.by_identity(sealed_id)[transition]["envelope"])[
                "error_code"
            ]
            == "model_error"
        )


def test_c2_admission_gates_do_not_dispatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runtime_root = tmp_path / "runtime"
    renderer, tokenizer, inventory = _load_runtime(monkeypatch, runtime_root)
    grant = _grant(inventory)
    licenses = _license_bytes()
    training = v02_corpus._generate_training_plan()
    order_root = tmp_path / "order"
    _mkdir_0700(order_root)
    order = v02_corpus.OfflineLedger(training, order_root)
    slot = _pilot_training_slot(training, pilot=True)
    with pytest.raises(ValueError, match="out of order"):
        _run_role(
            order,
            cast(str, slot["slot_id"]),
            "independent_reviewer",
            grant,
            licenses,
            renderer,
            tokenizer,
            _reject_network,
        )
    assert order.events() == []
    sealed = v02_corpus._generate_sealed_plan()
    sealed_root = tmp_path / "sealed-order"
    _mkdir_0700(sealed_root)
    sealed_ledger = v02_corpus.OfflineLedger(sealed, sealed_root)
    identity = cast(list[dict[str, Any]], sealed["pilot_prefix"]["identities"])[0]
    sealed_id = cast(str, identity["identity_id"])
    with pytest.raises(ValueError, match="out of order"):
        _run_role(
            sealed_ledger,
            sealed_id,
            "sealed_annotator_a",
            grant,
            licenses,
            renderer,
            tokenizer,
            _reject_network,
        )
    sealed_ledger.commit("sealed_author", _sealed_envelope(sealed, "sealed_author"))
    with pytest.raises(ValueError, match="out of order"):
        _run_role(
            sealed_ledger,
            sealed_id,
            "sealed_adjudicator",
            grant,
            licenses,
            renderer,
            tokenizer,
            _reject_network,
        )
    concordant_root = tmp_path / "concordant"
    _mkdir_0700(concordant_root)
    concordant = v02_corpus.OfflineLedger(sealed, concordant_root)
    concordant.commit("sealed_author", _sealed_envelope(sealed, "sealed_author"))
    concordant.commit(
        "sealed_annotator_a", _sealed_envelope(sealed, "sealed_annotator_a", "option_0")
    )
    concordant.commit(
        "sealed_annotator_b", _sealed_envelope(sealed, "sealed_annotator_b", "option_0")
    )
    with pytest.raises(ValueError, match="agreement"):
        _run_role(
            concordant,
            sealed_id,
            "sealed_adjudicator",
            grant,
            licenses,
            renderer,
            tokenizer,
            _reject_network,
        )
    assert "sealed_adjudicator" not in concordant.by_identity(sealed_id)
    assert "sealed_adjudicator_failure" not in concordant.by_identity(sealed_id)
    full_root = tmp_path / "full"
    _mkdir_0700(full_root)
    full = v02_corpus.OfflineLedger(training, full_root)
    full_slot = _pilot_training_slot(training, pilot=False)
    with pytest.raises(ValueError, match="pilot"):
        _run_role(
            full,
            cast(str, full_slot["slot_id"]),
            "training_author",
            grant,
            licenses,
            renderer,
            tokenizer,
            _reject_network,
        )
    assert full.events() == []
    no_go_root = tmp_path / "nogo"
    _mkdir_0700(no_go_root)
    no_go = v02_corpus.OfflineLedger(training, no_go_root)
    micro_ids = set(v02_corpus.grounding_micro_pilot(training))
    indices = [
        index
        for index, row in enumerate(cast(list[dict[str, Any]], training["slots"]))
        if row["slot_id"] in micro_ids
    ]
    for index in indices[:2]:
        envelope = _training_envelope(training, "training_author", index=index)
        for key in ("answer", "semantic", "gates", "content_digest", "construction"):
            envelope.pop(key)
        envelope["schema_version"] = "v02-envelope.v1"
        envelope["error_code"] = "model_error"
        no_go.commit("training_author_failure", envelope)
    assert v02_corpus.reduce_training(training, no_go.events())["status"] == "NO_GO"
    with pytest.raises(ValueError, match="terminal"):
        _run_role(
            no_go,
            cast(str, training["slots"][indices[2]]["slot_id"]),
            "training_author",
            grant,
            licenses,
            renderer,
            tokenizer,
            _reject_network,
        )
    assert len(no_go.events()) == 2
    parameters = set(inspect_signature())
    assert parameters.isdisjoint(
        {"case", "bilingual_source", "committed_labels", "duplicate_history"}
    )


def inspect_signature() -> set[str]:
    import inspect

    return set(inspect.signature(v02_corpus.execute_role).parameters)


def test_c2_store_history_case_drift_and_binding_drift_reject_replay(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _mkdir_0700(tmp_path)
    renderer, tokenizer, inventory = _load_runtime(monkeypatch, tmp_path / "runtime")
    grant = _grant(inventory)
    licenses = _license_bytes()
    plan = v02_corpus._generate_training_plan()
    left, right = _pilot_pair(plan)
    ledger = v02_corpus.OfflineLedger(plan, tmp_path)
    _commit_micro_pass(ledger)
    shared_state = "Shared fictional facts for both locales"
    first = _Loopback(
        v02_corpus.MODEL_ROLES["training_author"]["model"],
        _training_content(left, state=shared_state),
    )
    _run_role(
        ledger,
        cast(str, left["slot_id"]),
        "training_author",
        grant,
        licenses,
        renderer,
        tokenizer,
        first,
    )
    second = _Loopback(
        v02_corpus.MODEL_ROLES["training_author"]["model"],
        _training_content(right, state=shared_state),
    )
    _run_role(
        ledger,
        cast(str, right["slot_id"]),
        "training_author",
        grant,
        licenses,
        renderer,
        tokenizer,
        second,
    )
    context = _message_context(second.posts()[0])
    source = cast(dict[str, Any], context["bilingual_source"])
    assert source["source_identity_id"] == left["slot_id"]
    assert cast(dict[str, Any], source["case"])["state"] == shared_state
    right_envelope = cast(
        dict[str, Any],
        ledger.by_identity(cast(str, right["slot_id"]))["training_author"]["envelope"],
    )
    assert right_envelope["gates"]["duplicate_valid"] is False
    history = v02_corpus._same_lane_history(ledger, cast(str, right["slot_id"]))
    repeated = v02_corpus.local_case_gates(
        _case_any(cast(int, right["option_count"]), state=shared_state),
        right,
        renderer,
        tokenizer,
        duplicate_history=history,
    )
    assert repeated["duplicate_valid"] is False
    resumed = _run_role(
        ledger,
        cast(str, right["slot_id"]),
        "training_author",
        grant,
        licenses,
        renderer,
        tokenizer,
        _reject_network,
    )
    assert resumed["dispatch"] is False
    case_path = v02_corpus._case_path(ledger.root, "training", cast(str, left["slot_id"]))
    original_case_bytes = case_path.read_bytes()
    stored = json.loads(original_case_bytes)
    cast(dict[str, Any], stored["case"])["state"] = "Changed fictional state"
    case_path.write_bytes(canonical_json_bytes(cast(JsonValue, stored)) + b"\n")
    os.chmod(case_path, 0o600)
    with pytest.raises(ValueError, match="digest"):
        _run_role(
            ledger,
            cast(str, left["slot_id"]),
            "independent_reviewer",
            grant,
            licenses,
            renderer,
            tokenizer,
            _reject_network,
        )
    case_path.write_bytes(original_case_bytes)
    os.chmod(case_path, 0o600)
    assert "training_reviewer" not in ledger.by_identity(cast(str, left["slot_id"]))
    drifted = dict(grant)
    drifted["account_boundary_digest"] = "55" * 32
    with pytest.raises(ValueError, match="runtime binding drift"):
        _run_role(
            ledger,
            cast(str, right["slot_id"]),
            "training_author",
            drifted,
            licenses,
            renderer,
            tokenizer,
            _reject_network,
        )
    moved_runtime = _runtime(
        "training_author",
        grant,
        v02_corpus.validate_environment_grant(grant, licenses),
        tokenizer.inventory_digest,
    )
    moved_runtime["dependency_lock_digest"] = "66" * 32
    with pytest.raises(ValueError, match="runtime binding drift"):
        _run_role(
            ledger,
            cast(str, right["slot_id"]),
            "training_author",
            grant,
            licenses,
            renderer,
            tokenizer,
            _reject_network,
            moved_runtime,
        )
    monkeypatch.setitem(v02_corpus.ROLE_TEMPLATES, "training_author", "changed {locale} {domain}")
    rebound = dict(grant)
    rebound["prompt_contract_digest"] = v02_corpus.prompt_contract()["contract_digest"]
    with pytest.raises(ValueError, match="runtime binding drift"):
        _run_role(
            ledger,
            cast(str, right["slot_id"]),
            "training_author",
            rebound,
            licenses,
            renderer,
            tokenizer,
            _reject_network,
        )


def test_c2_unresolved_reservation_and_foreign_lock_do_not_retry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _mkdir_0700(tmp_path)
    renderer, tokenizer, inventory = _load_runtime(monkeypatch, tmp_path / "runtime")
    grant = _grant(inventory)
    licenses = _license_bytes()
    plan = v02_corpus._generate_training_plan()
    slot = _pilot_training_slot(plan, pilot=True)
    identity_id = cast(str, slot["slot_id"])
    ledger = v02_corpus.OfflineLedger(plan, tmp_path)
    grant_digest = v02_corpus.validate_environment_grant(grant, licenses)
    evidence = _runtime("training_author", grant, grant_digest, tokenizer.inventory_digest)
    evidence_digest = v02_corpus.validate_runtime_evidence(
        evidence, "training_author", grant_digest
    )
    payload = {
        "schema_version": "v02-role-reservation.v1",
        "identity_id": identity_id,
        "role": "training_author",
        "grant_digest": "ab" * 32,
        "runtime_evidence_digest": evidence_digest,
        "prompt_contract_digest": v02_corpus.prompt_contract()["contract_digest"],
        "renderer_lock_digest": v02_corpus._renderer_lock_digest(),
        "request_digest": "cd" * 32,
    }
    v02_corpus._private_dir(ledger.root, "role-reservations")
    path = v02_corpus._reservation_path(ledger.root, identity_id, "training_author")
    v02_corpus._write_match_or_create(path, payload, "runtime binding drift")
    with pytest.raises(ValueError, match="runtime binding drift"):
        _run_role(
            ledger,
            identity_id,
            "training_author",
            grant,
            licenses,
            renderer,
            tokenizer,
            _reject_network,
            evidence,
        )
    assert ledger.events() == []
    path.unlink()
    payload["grant_digest"] = grant_digest
    v02_corpus._write_match_or_create(path, payload, "runtime binding drift")
    holder = subprocess.Popen(
        [sys.executable, "-c", _LOCK_HOLDER, os.fspath(ledger.root)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    assert holder.stdout is not None and holder.stdin is not None
    try:
        assert holder.stdout.readline().strip() == "locked"
        busy = _run_role(
            ledger,
            identity_id,
            "training_author",
            grant,
            licenses,
            renderer,
            tokenizer,
            _reject_network,
            evidence,
        )
        assert busy == {"status": "busy", "dispatch": False}
        assert ledger.events() == []
        assert json.loads(path.read_text(encoding="utf-8"))["request_digest"] == "cd" * 32
    finally:
        holder.stdin.write("\n")
        holder.stdin.flush()
        holder.wait(timeout=10)
    failed = _run_role(
        ledger,
        identity_id,
        "training_author",
        grant,
        licenses,
        renderer,
        tokenizer,
        _reject_network,
        evidence,
    )
    assert failed["dispatch"] is False
    assert failed["transition"] == "training_author_failure"
    assert (
        cast(
            dict[str, Any], ledger.by_identity(identity_id)["training_author_failure"]["envelope"]
        )["error_code"]
        == "model_error"
    )
    resumed = _run_role(
        ledger,
        identity_id,
        "training_author",
        grant,
        licenses,
        renderer,
        tokenizer,
        _reject_network,
        evidence,
    )
    assert resumed == {
        "status": "resumed",
        "dispatch": False,
        "transition": "training_author_failure",
    }
