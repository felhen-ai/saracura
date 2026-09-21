from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from saracura.backends import DeterministicFixtureBackend
from saracura.calibration import CalibrationArtifact, load_calibration, write_calibration_atomic
from saracura.calibration.models import CalibrationContext
from saracura.contracts import ErrorCode, SaracuraError
from saracura.runtime import known_scaling_questions
from tests.helpers import calibration_for, decision_request


def _fixture() -> tuple[CalibrationContext, CalibrationArtifact]:
    request = decision_request(known_scaling_questions(1))
    backend = DeterministicFixtureBackend()
    artifact = calibration_for(request, request.questions[0], backend)
    return artifact.context(), artifact


@pytest.mark.parametrize(
    ("axis", "replacement"),
    [
        ("model_id", "another-model"),
        ("model_revision", "another-revision"),
        ("checkpoint_sha256", "0" * 64),
        ("architecture_config_sha256", "1" * 64),
        ("serializer_version", "another-serializer"),
        ("tokenizer_revision", "another-tokenizer"),
        ("truncation_policy_id", "truncate-v2"),
        ("precision", "fp32"),
        ("quantization", "int8"),
        ("output_transform", "another-transform"),
        ("workflow_id", "another-workflow"),
        ("workflow_revision", "another-revision"),
        ("question_id", "another-question"),
        ("dataset_id", "another-dataset"),
        ("dataset_revision", "another-revision"),
        ("split_manifest_sha256", "2" * 64),
        ("locale", "en-US"),
        ("domain", "another-domain"),
        ("cardinality_bucket", "6-20"),
    ],
)
def test_calibration_mismatch_fails_closed(
    tmp_path: Path,
    axis: str,
    replacement: str,
) -> None:
    context, artifact = _fixture()
    path = tmp_path / "calibration.json"
    write_calibration_atomic(path, artifact)
    expected_payload = context.model_dump(mode="json")
    expected_payload[axis] = replacement
    expected = CalibrationContext.model_validate(expected_payload)

    with pytest.raises(SaracuraError) as captured:
        load_calibration(path, expected)
    assert captured.value.payload.code == ErrorCode.CALIBRATION_INCOMPATIBLE
    mismatched_axes = captured.value.payload.details["mismatched_axes"]
    assert isinstance(mismatched_axes, list)
    assert axis in mismatched_axes


def test_atomic_write_is_canonical_and_immutable_by_default(tmp_path: Path) -> None:
    context, artifact = _fixture()
    path = tmp_path / "calibration.json"

    write_calibration_atomic(path, artifact)

    assert load_calibration(path, context) == artifact
    assert path.read_bytes().endswith(b"\n")
    assert list(tmp_path.glob(".*.tmp")) == []
    with pytest.raises(FileExistsError):
        write_calibration_atomic(path, artifact)


def test_atomic_create_does_not_overwrite_a_concurrent_winner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, artifact = _fixture()
    path = tmp_path / "calibration.json"
    original_link = os.link

    def collide(source: Path, destination: Path) -> None:
        path.write_bytes(b"concurrent-winner")
        original_link(source, destination)

    monkeypatch.setattr(os, "link", collide)
    with pytest.raises(FileExistsError):
        write_calibration_atomic(path, artifact)

    assert path.read_bytes() == b"concurrent-winner"
    assert list(tmp_path.glob(".*.tmp")) == []


def test_invalid_calibration_does_not_echo_artifact_values(tmp_path: Path) -> None:
    context, artifact = _fixture()
    payload = artifact.model_dump(mode="json")
    payload["unexpected"] = "sensitive-artifact-value"
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SaracuraError) as captured:
        load_calibration(path, context)

    assert "sensitive-artifact-value" not in str(captured.value.as_dict())
    violations = captured.value.payload.details["violations"]
    assert isinstance(violations, list)
    assert violations
