import json
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import pytest

import benchmarks.data_policy_gate as data_policy_gate
from benchmarks.data_policy_gate import main as data_policy_main
from benchmarks.data_policy_registry import (
    FIXED_METADATA,
    IDENTITY,
    MINIMUMS,
    bundled_registry_path,
    load_registry,
)

ROOT = Path(__file__).parents[1]
MINIMUM_CONTROL_CASES = [
    (source_id, control)
    for source_id, controls in sorted(MINIMUMS.items())
    for control in sorted(controls)
]
FIXED_IDENTITY_FIELDS = ("source_type", "origin", "authoring_mode", "license", "policy_state")
FIXED_METADATA_FIELDS = (
    "locales",
    "source_url",
    "license_url",
    "upstream_revision",
    "record_identity_state",
    "rights_review_state",
    "teacher_lineage_state",
    "redistribution_state",
)


def payload() -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(bundled_registry_path().read_text(encoding="utf-8")))


def assert_rejected(candidate: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="invalid data policy registry"):
        load_registry(json.dumps(candidate).encode())


def test_bundled_registry_has_exact_eight_ids_and_no_approvals() -> None:
    registry = load_registry(bundled_registry_path().read_bytes())
    assert len(registry.sources) == 8
    assert all(source.artifact_approved_for_use is False for source in registry.sources)
    assert {source.id for source in registry.sources} == set(MINIMUMS)


def test_duplicate_json_keys_are_rejected() -> None:
    raw = b'{"schema_version":"training-data-source-policies.v1","schema_version":"x","sources":[]}'
    with pytest.raises(ValueError, match="invalid data policy registry"):
        load_registry(raw)


def test_duplicate_ids_are_rejected() -> None:
    candidate = payload()
    candidate["sources"][1]["id"] = candidate["sources"][0]["id"]
    assert_rejected(candidate)


@pytest.mark.parametrize("source_id", sorted(IDENTITY))
@pytest.mark.parametrize("field", FIXED_IDENTITY_FIELDS)
def test_every_fixed_identity_field_is_enforced(source_id: str, field: str) -> None:
    alternatives = {
        "source_type": ("unvetted_public", "human_original"),
        "origin": ("unknown", "first_party"),
        "authoring_mode": ("not_applicable", "human_original"),
        "license": ("unknown", "Apache-2.0"),
        "policy_state": ("blocked", "quarantined"),
    }
    candidate = payload()
    source = next(item for item in candidate["sources"] if item["id"] == source_id)
    first, second = alternatives[field]
    source[field] = second if source[field] == first else first
    assert_rejected(candidate)


@pytest.mark.parametrize(("source_id", "control"), MINIMUM_CONTROL_CASES)
def test_removing_any_minimum_control_is_rejected(source_id: str, control: str) -> None:
    candidate = payload()
    source = next(item for item in candidate["sources"] if item["id"] == source_id)
    source["required_controls"].remove(control)
    assert_rejected(candidate)


def test_deterministic_derivatives_cannot_enter_holdout_or_calibration() -> None:
    candidate = payload()
    source = candidate["sources"][1]
    source["planned_uses"] = ["train", "dev", "calibration", "blind_test"]
    assert_rejected(candidate)


def test_model_assisted_source_cannot_be_marked_human_original() -> None:
    candidate = payload()
    source = candidate["sources"][2]
    source["authoring_mode"] = "human_original"
    assert_rejected(candidate)


def test_model_assisted_source_cannot_be_promoted_to_training() -> None:
    candidate = payload()
    source = candidate["sources"][2]
    source["planned_uses"] = ["train"]
    assert_rejected(candidate)


@pytest.mark.parametrize(
    "source_id", ["felhen-aios-operational", "customer-support-exports", "unvetted-public-text"]
)
def test_private_and_unvetted_sources_cannot_gain_uses(source_id: str) -> None:
    candidate = payload()
    source = next(item for item in candidate["sources"] if item["id"] == source_id)
    source["planned_uses"] = ["train"]
    assert_rejected(candidate)


def test_massive_ptpt_cannot_be_retyped_or_used_as_ptbr() -> None:
    candidate = payload()
    source = candidate["sources"][3]
    source["locales"] = ["pt-BR"]
    assert_rejected(candidate)


def test_locales_cannot_be_empty_or_changed_for_fixed_policy() -> None:
    candidate = payload()
    candidate["sources"][0]["locales"] = []
    assert_rejected(candidate)

    candidate = payload()
    candidate["sources"][0]["locales"] = ["any"]
    assert_rejected(candidate)


@pytest.mark.parametrize("source_id", ["amazon-massive-ptpt", "community-massive-ptbr"])
def test_third_party_identity_metadata_is_exact(source_id: str) -> None:
    candidate = payload()
    source = next(item for item in candidate["sources"] if item["id"] == source_id)
    source["source_url"] = "https://huggingface.co/datasets/example/other/tree/dev"
    assert_rejected(candidate)

    candidate = payload()
    source = next(item for item in candidate["sources"] if item["id"] == source_id)
    source["upstream_revision"] = "a" * 40
    assert_rejected(candidate)


def test_every_policy_has_fixed_metadata_keys() -> None:
    assert set(FIXED_METADATA) == set(IDENTITY)


@pytest.mark.parametrize("source_id", sorted(FIXED_METADATA))
@pytest.mark.parametrize("field", FIXED_METADATA_FIELDS)
def test_every_fixed_metadata_field_is_enforced(source_id: str, field: str) -> None:
    alternatives: dict[str, tuple[Any, ...]] = {
        "locales": (["any"], ["pt-BR"]),
        "source_url": (None, "https://example.test/other/tree/dev"),
        "license_url": (None, "https://example.test/other-license"),
        "upstream_revision": (None, "a" * 40),
        "record_identity_state": ("not_created", "not_acquired", "unavailable"),
        "rights_review_state": ("policy_only", "unknown"),
        "teacher_lineage_state": ("not_used", "required_missing", "not_applicable"),
        "redistribution_state": (
            "policy_allowed",
            "requires_artifact_review",
            "prohibited",
        ),
    }
    candidate = payload()
    source = next(item for item in candidate["sources"] if item["id"] == source_id)
    source[field] = next(value for value in alternatives[field] if value != source[field])
    assert_rejected(candidate)


def test_artifact_and_privacy_state_fail_closed() -> None:
    candidate = payload()
    candidate["sources"][0]["artifact_approved_for_use"] = True
    assert_rejected(candidate)

    candidate = payload()
    candidate["sources"][0]["privacy_review_state"] = "reviewed"
    assert_rejected(candidate)

    candidate = payload()
    source = candidate["sources"][3]
    source["planned_uses"] = ["train"]
    assert_rejected(candidate)


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_url", "https://user:pass@example.test/source"),
        ("source_url", "http://example.test/source"),
        ("source_url", "https://example.test/source?mutable=1"),
        ("source_url", "https://example.test/tree/main"),
        ("upstream_revision", "main"),
        ("upstream_revision", "A" * 40),
        ("license_url", "https://user:pass@example.test/license"),
    ],
)
def test_urls_and_revisions_are_immutable_and_credential_free(field: str, value: str) -> None:
    candidate = payload()
    source = candidate["sources"][3]
    source[field] = value
    assert_rejected(candidate)


def test_known_license_requires_attribution_and_license_url() -> None:
    candidate = payload()
    source = candidate["sources"][3]
    source["attribution"] = ""
    assert_rejected(candidate)


def test_first_party_source_cannot_claim_upstream_metadata() -> None:
    candidate = payload()
    source = candidate["sources"][0]
    source["source_url"] = "https://example.test/source"
    source["upstream_revision"] = "a" * 40
    assert_rejected(candidate)

    candidate = payload()
    source = candidate["sources"][3]
    source["license_url"] = None
    assert_rejected(candidate)


def test_unknown_schema_and_unknown_fields_are_rejected() -> None:
    candidate = payload()
    candidate["schema_version"] = "training-data-source-policies.v2"
    assert_rejected(candidate)
    candidate = payload()
    candidate["sources"][0]["unexpected"] = True
    assert_rejected(candidate)


def test_validator_is_network_free(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_network(*args: object, **kwargs: object) -> object:
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket, "create_connection", fail_network)
    monkeypatch.setattr(socket, "getaddrinfo", fail_network)
    assert data_policy_main(["validate-registry"]) == 0


def test_cli_surface_is_closed() -> None:
    assert data_policy_main([]) == 2
    assert data_policy_main(["validate-registry", "--source", "anything"]) == 2


def test_cli_bounds_registry_read_errors(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail_read() -> object:
        raise OSError("/private/local/path")

    monkeypatch.setattr(data_policy_gate, "load_bundled_registry", fail_read)
    assert data_policy_main(["validate-registry"]) == 1
    captured = capsys.readouterr()
    assert captured.err == "data-policy registry: invalid\n"
    assert "/private" not in captured.err


def test_default_environment_does_not_import_dataset_loader() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import benchmarks.data_policy_gate; print('datasets' in sys.modules)",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "False"
