from __future__ import annotations

import json
import os
import unicodedata
from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest

from benchmarks.human_calibration import calibration_candidate, metrics
from saracura.backends import DeterministicFixtureBackend
from saracura.calibration import (
    CalibrationArtifact,
    ResearchCalibrationArtifact,
    load_calibration,
    research_calibration_id,
    write_calibration_atomic,
)
from saracura.calibration.io import write_calibration_atomic_at
from saracura.calibration.models import BlindMetrics, CalibrationContext, TemperatureParameters
from saracura.contracts import ErrorCode, SaracuraError
from saracura.research_trust import canonical
from saracura.runtime import DecisionEngine, default_workflows, known_scaling_questions
from saracura.serialization import canonical_json_bytes
from tests.helpers import calibration_for, decision_request
from tests.support_human_research import inject_test_trust, signed_receipt


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


def test_descriptor_atomic_write_retries_temporary_cleanup_after_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, artifact = _fixture()
    parent_fd = os.open(tmp_path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    original_unlink = os.unlink
    original_fsync = os.fsync
    failed_once = False
    events: list[str] = []

    def fail_first_temporary_unlink(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        *,
        dir_fd: int | None = None,
    ) -> None:
        nonlocal failed_once
        if not failed_once and str(path).endswith(".tmp"):
            failed_once = True
            events.append("unlink-fail")
            raise OSError("injected temporary unlink failure")
        events.append("unlink-retry")
        original_unlink(path, dir_fd=dir_fd)

    def record_fsync(fd: int) -> None:
        events.append("fsync-parent" if fd == parent_fd else "fsync-temp")
        original_fsync(fd)

    monkeypatch.setattr(os, "unlink", fail_first_temporary_unlink)
    monkeypatch.setattr(os, "fsync", record_fsync)
    try:
        with pytest.raises(OSError, match="injected temporary unlink failure"):
            write_calibration_atomic_at(parent_fd, "calibration.json", artifact)
    finally:
        os.close(parent_fd)

    assert (tmp_path / "calibration.json").exists()
    assert list(tmp_path.glob(".*.tmp")) == []
    assert events.index("unlink-fail") < events.index("unlink-retry") < events.index("fsync-parent")


def test_descriptor_atomic_write_surfaces_fsync_after_recovered_unlink_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, artifact = _fixture()
    parent_fd = os.open(tmp_path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    original_unlink = os.unlink
    original_fsync = os.fsync
    failed_once = False

    def fail_first_temporary_unlink(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        *,
        dir_fd: int | None = None,
    ) -> None:
        nonlocal failed_once
        if not failed_once and str(path).endswith(".tmp"):
            failed_once = True
            raise OSError("injected temporary unlink failure")
        original_unlink(path, dir_fd=dir_fd)

    def fail_parent_fsync(fd: int) -> None:
        if fd == parent_fd:
            raise OSError("injected parent fsync failure")
        original_fsync(fd)

    monkeypatch.setattr(os, "unlink", fail_first_temporary_unlink)
    monkeypatch.setattr(os, "fsync", fail_parent_fsync)
    try:
        with pytest.raises(OSError, match="injected parent fsync failure") as error:
            write_calibration_atomic_at(parent_fd, "calibration.json", artifact)
    finally:
        os.close(parent_fd)

    assert error.value.__cause__ is not None
    assert "injected temporary unlink failure" in str(error.value.__cause__)
    assert (tmp_path / "calibration.json").exists()
    assert list(tmp_path.glob(".*.tmp")) == []


def test_signed_v2_rejects_equivalent_decomposed_unicode_before_evidence_reconstruction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    crypto = pytest.importorskip("cryptography.hazmat.primitives.asymmetric.ed25519")
    private_key = crypto.Ed25519PrivateKey.generate()
    inject_test_trust(monkeypatch, private_key, scopes=["research_calibration"])
    context = _human_context().model_copy(
        update={"tokenizer_revision": "minilm-tokenizer-révision"}
    )
    monkeypatch.setattr(
        "tests.test_calibration._human_context",
        lambda: context,
    )
    artifact = _v2_artifact(private_key)
    payload = artifact.model_dump(mode="json")
    original_revision = payload["tokenizer_revision"]
    assert isinstance(original_revision, str)
    payload["tokenizer_revision"] = unicodedata.normalize("NFD", original_revision)
    assert payload["tokenizer_revision"] != original_revision
    assert unicodedata.normalize("NFC", payload["tokenizer_revision"]) == original_revision
    # Receipt, candidate digest, and calibration ID remain byte-for-byte from
    # the valid NFC artifact.  This reproduces the exact signed-equivalence
    # attack: only the raw context string is decomposed.
    assert payload["release_receipt"] == artifact.release_receipt
    assert payload["candidate_sha256"] == artifact.candidate_sha256
    assert payload["calibration_id"] == artifact.calibration_id
    path = tmp_path / "decomposed-equivalent.json"
    path.write_bytes(canonical(payload) + b"\n")
    path.chmod(0o600)
    with pytest.raises(SaracuraError):
        load_calibration(path, artifact.context())


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


def test_schema_v1_never_claims_verified_for_research(tmp_path: Path) -> None:
    context, artifact = _fixture()
    payload = artifact.model_dump(mode="json")
    payload["status"] = "verified_for_research"
    path = tmp_path / "v1.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SaracuraError) as captured:
        load_calibration(path, context)

    assert captured.value.payload.code == ErrorCode.CALIBRATION_INCOMPATIBLE


def _human_context() -> CalibrationContext:
    return CalibrationContext(
        model_id="saracura-minilm-routing",
        model_revision="minilm-routing-v1.test",
        checkpoint_sha256="a" * 64,
        architecture_config_sha256="b" * 64,
        serializer_version="nfc-jcs-length-prefixed-v1",
        tokenizer_revision="minilm-tokenizer-test",
        truncation_policy_id="padding-truncation-max128-v1",
        precision="encoder-fp32-mps-pool-fp32-mps-transfer-head-fp32-cpu-v1",
        quantization="none",
        output_transform="linear-logits-softmax-v1",
        workflow_id="support-routing",
        workflow_revision="phase4a-local-minilm-routing.v1",
        question_id="department",
        dataset_id="support-routing-human-ptbr",
        dataset_revision="human-ptbr-v1." + "c" * 32,
        split_manifest_sha256="c" * 64,
        locale="pt-BR",
        domain="support",
        head="choice",
        cardinality_bucket="2-5",
        risk_policy=None,
    )


def _aggregate_metrics() -> dict[str, Any]:
    logits = [
        [3.0 if index == label else 0.0 for index in range(5)]
        for label in range(5)
        for _ in range(20)
    ]
    labels = [label for label in range(5) for _ in range(20)]
    return metrics(logits, labels, 1.0)


def _v2_artifact(private_key: Any) -> ResearchCalibrationArtifact:
    context = _human_context()
    aggregate = _aggregate_metrics()
    candidate = calibration_candidate(
        context,
        temperature=1.0,
        fit_sample_count=100,
        evaluation_sample_count=100,
        fit_independent_state_count=100,
        evaluation_independent_state_count=100,
        fit_sha256="d" * 64,
        blind_view_sha256="e" * 64,
        exposure_run_sha256="f" * 64,
        blind_report_sha256="1" * 64,
        metrics_before=aggregate,
        metrics_after=aggregate,
        created_at="2026-09-22T00:00:00Z",
    )
    evidence = {
        **{
            name: "2" * 64
            for name in (
                "protocol",
                "policy_registry",
                "states",
                "split_plan",
                "contributors",
                "annotations",
                "adjudications",
                "controls",
                "takedown_ledger",
            )
        },
        "packet": context.split_manifest_sha256,
        "training_manifest": "3" * 64,
        "checkpoint": context.checkpoint_sha256,
        "fit": candidate["fit_sha256"],
        "blind_view": candidate["blind_view_sha256"],
        "exposure_run": candidate["exposure_run_sha256"],
        "blind_report": candidate["blind_report_sha256"],
        "calibration_candidate": candidate["candidate_sha256"],
    }
    receipt_raw = signed_receipt(private_key, evidence, scope="research_calibration")
    receipt = json.loads(receipt_raw)
    receipt_digest = sha256(receipt_raw).hexdigest()
    return ResearchCalibrationArtifact(
        **context.model_dump(mode="python"),
        schema_version=2,
        status="verified_for_research",
        calibration_id=research_calibration_id(
            context,
            "temperature-scaling",
            TemperatureParameters(temperature=1.0),
            candidate["fit_sha256"],
            candidate["blind_view_sha256"],
            candidate["exposure_run_sha256"],
            candidate["blind_report_sha256"],
            receipt_digest,
        ),
        method="temperature-scaling",
        parameters=TemperatureParameters(temperature=1.0),
        fit_sample_count=100,
        evaluation_sample_count=100,
        fit_independent_state_count=100,
        evaluation_independent_state_count=100,
        minimum_independent_state_count=100,
        metrics_before=BlindMetrics.model_validate(aggregate),
        metrics_after=BlindMetrics.model_validate(aggregate),
        fit_sha256=candidate["fit_sha256"],
        blind_view_sha256=candidate["blind_view_sha256"],
        exposure_run_sha256=candidate["exposure_run_sha256"],
        blind_report_sha256=candidate["blind_report_sha256"],
        candidate_sha256=candidate["candidate_sha256"],
        release_receipt=receipt,
        release_receipt_sha256=receipt_digest,
        created_at=candidate["created_at"],
    )


def test_v2_requires_verified_loader_even_when_constructed_in_memory() -> None:
    """An unsigned model_validate result cannot cross the DecisionEngine boundary."""

    crypto = pytest.importorskip("cryptography.hazmat.primitives.asymmetric.ed25519")
    artifact = _v2_artifact(crypto.Ed25519PrivateKey.generate())
    request = decision_request(known_scaling_questions(1))
    backend = DeterministicFixtureBackend()
    engine = DecisionEngine(
        backend=backend,
        workflows=default_workflows(),
        calibrations={request.questions[0].id: artifact},
    )

    with pytest.raises(SaracuraError) as captured:
        engine.decide(request)

    assert captured.value.payload.code == ErrorCode.CALIBRATION_INCOMPATIBLE


def test_engine_revalidates_model_copy_before_selecting_schema_lane() -> None:
    """A v1 instance cannot become v2 merely by bypassing Pydantic validators."""

    _context_value, artifact = _fixture()
    forged = artifact.model_copy(update={"schema_version": 2, "status": "verified_for_research"})
    request = decision_request(known_scaling_questions(1))
    engine = DecisionEngine(
        backend=DeterministicFixtureBackend(),
        workflows=default_workflows(),
        calibrations={request.questions[0].id: forged},
    )

    with pytest.raises(SaracuraError) as captured:
        engine.decide(request)

    assert captured.value.payload.code == ErrorCode.CALIBRATION_INCOMPATIBLE


def test_v2_has_no_public_runtime_verification_marker() -> None:
    """Importing the model module cannot bless an arbitrary v2 object."""

    import saracura.calibration.models as calibration_models

    assert not hasattr(calibration_models, "mark_research_artifact_runtime_verified")
    assert not hasattr(calibration_models, "require_runtime_verified")


@pytest.mark.parametrize("mode", (0o750, 0o711))
def test_legacy_calibration_accepts_safe_read_execute_ancestors(tmp_path: Path, mode: int) -> None:
    context, artifact = _fixture()
    parent = tmp_path / f"safe-{mode:o}"
    parent.mkdir(mode=0o700)
    path = parent / "legacy.json"
    write_calibration_atomic(path, artifact)
    path.chmod(0o644)
    parent.chmod(mode)
    try:
        assert load_calibration(path, context) == artifact
    finally:
        parent.chmod(0o700)


@pytest.mark.parametrize("mode", (0o770, 0o707))
def test_calibration_rejects_group_or_world_writable_ancestor(tmp_path: Path, mode: int) -> None:
    context, artifact = _fixture()
    parent = tmp_path / f"unsafe-{mode:o}"
    parent.mkdir(mode=0o700)
    path = parent / "legacy.json"
    write_calibration_atomic(path, artifact)
    parent.chmod(mode)
    try:
        with pytest.raises(SaracuraError):
            load_calibration(path, context)
    finally:
        parent.chmod(0o700)


def test_v2_ephemeral_ed25519_receipt_binds_every_candidate_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    crypto = pytest.importorskip("cryptography.hazmat.primitives.asymmetric.ed25519")
    private_key = crypto.Ed25519PrivateKey.generate()
    inject_test_trust(monkeypatch, private_key, scopes=["research_calibration"])
    artifact = _v2_artifact(private_key)
    parent = tmp_path / "calibrations"
    parent.mkdir(mode=0o700)
    parent.chmod(0o700)
    path = parent / "research-calibration.json"
    write_calibration_atomic(path, artifact)

    loaded = load_calibration(path, artifact.context())
    assert isinstance(loaded, ResearchCalibrationArtifact)
    from saracura.calibration.io import _verify_research_artifact

    with pytest.raises(ValueError, match="research calibration id"):
        _verify_research_artifact(artifact.model_copy(update={"calibration_id": "forged-id"}))
    fixture_context, _ = _fixture()
    with pytest.raises(SaracuraError, match="incompatible"):
        load_calibration(path, fixture_context)

    payload = artifact.model_dump(mode="json")
    mutations: dict[str, Any] = {
        "parameters": {"temperature": 1.1, "bounds": [0.5, 5.0]},
        "tokenizer_revision": "minilm-tokenizer-tampered",
        "dataset_revision": "human-ptbr-v1.tampered",
        "fit_sample_count": 101,
        "metrics_after": {**payload["metrics_after"], "mean_confidence": 0.99},
        "created_at": "2026-09-22T00:00:01Z",
    }
    for index, (name, replacement) in enumerate(mutations.items()):
        tampered = dict(payload)
        tampered[name] = replacement
        target = parent / f"tampered-{index}.json"
        target.write_bytes(canonical_json_bytes(tampered) + b"\n")
        target.chmod(0o600)
        with pytest.raises(SaracuraError, match="failed schema validation"):
            load_calibration(target, artifact.context())


@pytest.mark.parametrize(
    ("name", "mutate", "noncanonical_schema_number"),
    [
        (
            "string-integer",
            lambda value: value.__setitem__("fit_sample_count", "100"),
            False,
        ),
        (
            "boolean-temperature",
            lambda value: value["parameters"].__setitem__("temperature", True),
            False,
        ),
        (
            "boolean-confusion-entry",
            lambda value: value["metrics_before"]["confusion_matrix"][0].__setitem__(0, True),
            False,
        ),
        (
            "float-schema-version",
            lambda value: value.__setitem__("schema_version", 2.0),
            True,
        ),
        (
            "float-integer-count",
            lambda value: value["metrics_after"].__setitem__("count", 100.0),
            False,
        ),
        (
            "wrong-array-shape-type",
            lambda value: value["parameters"].__setitem__("bounds", {"lower": 0.5, "upper": 5.0}),
            False,
        ),
        (
            "wrong-null-type",
            lambda value: value.__setitem__("risk_policy", False),
            False,
        ),
        (
            "wrong-receipt-container-type",
            lambda value: value.__setitem__("release_receipt", []),
            False,
        ),
    ],
)
def test_schema_v2_loader_rejects_all_wire_type_coercions(
    tmp_path: Path,
    name: str,
    mutate: Any,
    noncanonical_schema_number: bool,
) -> None:
    """Schema-derived wire validation precedes model conversion and receipts."""

    crypto = pytest.importorskip("cryptography.hazmat.primitives.asymmetric.ed25519")
    artifact = _v2_artifact(crypto.Ed25519PrivateKey.generate())
    payload = artifact.model_dump(mode="json")
    mutate(payload)
    raw = canonical_json_bytes(payload) + b"\n"
    if noncanonical_schema_number:
        # RFC 8785 serializes 2.0 as 2.  Keep the intentionally wrong wire
        # lexeme to prove the type gate runs before the canonicality gate.
        raw = raw.replace(b'"schema_version":2', b'"schema_version":2.0')
        assert b'"schema_version":2.0' in raw
    parent = tmp_path / "calibrations"
    parent.mkdir(mode=0o700)
    parent.chmod(0o700)
    path = parent / f"{name}.json"
    path.write_bytes(raw)
    path.chmod(0o600)

    with pytest.raises(SaracuraError) as captured:
        load_calibration(path, artifact.context())

    assert captured.value.payload.code == ErrorCode.CALIBRATION_INCOMPATIBLE


def test_calibration_reader_rejects_link_fifo_and_oversize_without_path_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context, artifact = _fixture()
    parent = tmp_path / "calibrations"
    parent.mkdir(mode=0o700)
    parent.chmod(0o700)
    ordinary = parent / "fixture.json"
    write_calibration_atomic(ordinary, artifact)
    monkeypatch.setattr(Path, "read_bytes", lambda _path: (_ for _ in ()).throw(AssertionError))
    assert load_calibration(ordinary, context) == artifact

    link = parent / "link.json"
    link.symlink_to(ordinary)
    with pytest.raises(SaracuraError):
        load_calibration(link, context)
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(parent, target_is_directory=True)
    with pytest.raises(SaracuraError):
        load_calibration(linked_parent / "fixture.json", context)
    fifo = parent / "fifo.json"
    os.mkfifo(fifo, 0o600)
    with pytest.raises(SaracuraError):
        load_calibration(fifo, context)
    oversized = parent / "oversized.json"
    oversized.write_bytes(b"x" * (1_024 * 1_024 + 1))
    oversized.chmod(0o600)
    with pytest.raises(SaracuraError):
        load_calibration(oversized, context)
