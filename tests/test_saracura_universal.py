"""Public, artifact-free contracts for the installed Saracura universal lane."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import socket
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import cast

import pytest
from pydantic import JsonValue

import saracura.backends.saracura_universal as saracura_universal
import saracura.cli as cli
from saracura.backends.base import BackendCapabilities, ScoredChoice
from saracura.backends.minilm import MiniLMRoutingBackend
from saracura.backends.saracura_universal import (
    CAPSULE_FILE_SET,
    SaracuraBackendError,
    SaracuraCandidate,
    SaracuraUniversalBackend,
    VerifiedSaracuraCapsule,
    load_saracura_candidate,
    validate_saracura_model_identity,
    validate_saracura_request_structure,
    verify_training_capsule,
)
from saracura.contracts import (
    ChoiceCriterion,
    ChoiceQuestion,
    DecisionRequest,
    ErrorCode,
    SaracuraError,
    WorkflowReference,
)
from saracura.runtime import DecisionEngine, default_workflows
from saracura.serialization import canonical_json_bytes
from saracura.universal.checkpoint import (
    BASE_ENCODER_ID,
    BASE_ENCODER_REVISION,
    CHECKPOINT_ARCHITECTURE_REVISION,
)
from saracura.universal.rendering import RenderedTask, render_choice, render_task
from saracura.universal.tasks import (
    SARACURA_UNIVERSAL_WORKFLOW_ID,
    SARACURA_UNIVERSAL_WORKFLOW_REVISION,
    UNIVERSAL_DOMAINS,
    UniversalTask,
)
from saracura.verified_bytes import HUMAN_TRAINING_CAPSULE_FILES


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json(value: object) -> bytes:
    return canonical_json_bytes(cast(JsonValue, value)) + b"\n"


def _candidate_for(values: dict[str, bytes], manifest: dict[str, object]) -> SaracuraCandidate:
    manifest_digest = str(manifest["manifest_sha256"])
    return SaracuraCandidate(
        id="saracura/universal-ranker",
        model_revision=f"{SARACURA_UNIVERSAL_WORKFLOW_REVISION}.{manifest_digest}",
        checkpoint_sha256=_sha(values["checkpoint.safetensors"]),
        training_manifest_internal_sha256=manifest_digest,
        training_manifest_file_sha256=_sha(values["training-manifest.json"]),
        capsule_descriptor_file_sha256=_sha(values["capsule-descriptor.json"]),
        capsule_descriptor_internal_sha256=json.loads(values["capsule-descriptor.json"])[
            "descriptor_sha256"
        ],
        conformance_manifest_file_sha256=_sha(values["conformance-manifest.json"]),
        conformance_vectors_sha256=_sha(values["conformance-vectors.safetensors"]),
        architecture=CHECKPOINT_ARCHITECTURE_REVISION,
        foundation_encoder=BASE_ENCODER_ID,
        encoder_revision=BASE_ENCODER_REVISION,
        encoder_snapshot_complete_sha256="a" * 64,
        training_source_commit="b" * 40,
        disposition="synthetic_only_research",
        capsule_files=tuple(sorted(CAPSULE_FILE_SET)),
    )


def _synthetic_capsule(root: Path) -> SaracuraCandidate:
    """Create synthetic bytes for installed-verifier negative tests only."""

    root.mkdir(mode=0o700)
    values = {name: _json({}) for name in CAPSULE_FILE_SET}
    values["checkpoint.safetensors"] = b"synthetic-checkpoint"
    values["conformance-vectors.safetensors"] = b"synthetic-conformance"
    conformance = {
        "schema_version": "phase4e-conformance-manifest.v1",
        "cells": [f"{locale}:{count}" for locale in ("pt-BR", "en") for count in range(2, 9)],
        "tensor_sha256": _sha(values["conformance-vectors.safetensors"]),
        "tensor_shapes": {
            "context_embeddings": [14, 384],
            "criterion_embeddings": [14, 8, 384],
            "option_masks": [14, 8],
            "option_counts": [14],
            "expected_logits": [14, 8],
        },
    }
    conformance["manifest_sha256"] = _sha(_json(conformance)[:-1])
    values["conformance-manifest.json"] = _json(conformance)
    manifest: dict[str, object] = {
        "architecture_revision": CHECKPOINT_ARCHITECTURE_REVISION,
        "base_encoder": {
            "id": BASE_ENCODER_ID,
            "revision": BASE_ENCODER_REVISION,
            "snapshot_complete_sha256": "a" * 64,
            "frozen": True,
        },
        "baselines": {},
        "checkpoint_sha256": _sha(values["checkpoint.safetensors"]),
        "code_sha256": "c" * 64,
        "code_sources": {"synthetic": "d" * 64},
        "dev_gates": {"synthetic": True},
        "embedding_descriptor_sha256": "e" * 64,
        "grant_digest": "f" * 64,
        "outcome": "passed",
        "packet_receipt_sha256": "1" * 64,
        "pre_holdout_gate_descriptor_sha256": "2" * 64,
        "rendering_revision": "phase4e-universal-renderer.v1",
        "schema_version": "phase4e-training-manifest.v2",
        "sealed_report_sha256": "3" * 64,
        "selected": {},
        "source_commit": "b" * 40,
        "synthetic_only": True,
        "training": {},
    }
    manifest["manifest_sha256"] = _sha(_json(manifest)[:-1])
    values["training-manifest.json"] = _json(manifest)
    ledger = [
        {"path": name, "bytes": len(values[name]), "sha256": _sha(values[name])}
        for name in sorted(CAPSULE_FILE_SET - {"capsule-descriptor.json"})
    ]
    descriptor = {
        "schema_version": "phase4e-training-capsule.v1",
        "capsule_kind": "training",
        "files": ledger,
    }
    descriptor["descriptor_sha256"] = _sha(_json(descriptor)[:-1])
    values["capsule-descriptor.json"] = _json(descriptor)
    for name, value in values.items():
        target = root / name
        target.write_bytes(value)
        target.chmod(0o600)
    return _candidate_for(values, manifest)


def _request(
    *,
    criteria_count: int = 2,
    domain: str = "email_triage",
    locale: str = "en",
    question_count: int = 1,
    model: str | None = None,
    state: dict[str, JsonValue] | None = None,
    instruction: str = "Choose the most appropriate route.",
    description: str = "Synthetic route.",
) -> DecisionRequest:
    criteria = tuple(
        ChoiceCriterion(id=f"route-{index}", description=f"{description} {index}")
        for index in range(criteria_count)
    )
    return DecisionRequest(
        api_version="v1alpha1",
        model=model or load_saracura_candidate().model_revision,
        mode="research",
        locale=locale,
        domain=domain,
        workflow=WorkflowReference(
            id=SARACURA_UNIVERSAL_WORKFLOW_ID,
            revision=SARACURA_UNIVERSAL_WORKFLOW_REVISION,
        ),
        state=state or {"summary": "synthetic public request"},
        questions=tuple(
            ChoiceQuestion(
                id=f"route-{question_index}",
                type="choice",
                instruction=instruction,
                criteria=criteria,
            )
            for question_index in range(question_count)
        ),
    )


def test_candidate_registry_pins_every_approved_identity_and_closed_ledger() -> None:
    candidate = load_saracura_candidate()
    assert candidate.id == "saracura/universal-ranker"
    assert candidate.capsule_files == tuple(sorted(CAPSULE_FILE_SET))

    package = (
        Path(__file__).parents[1] / "src" / "saracura" / "saracura-candidates.v1.json"
    ).read_bytes()
    for replacement in (b"0", b"A"):
        altered = package.replace(candidate.checkpoint_sha256.encode(), replacement * 64, 1)
        with pytest.raises(SaracuraBackendError):
            load_saracura_candidate(altered)
    with pytest.raises(SaracuraBackendError):
        load_saracura_candidate(package.replace(b'"disposition"', b'"unexpected"', 1))
    with pytest.raises(SaracuraBackendError):
        load_saracura_candidate(package.replace(b'"id":', b'"id":"x","id":', 1))


@pytest.mark.parametrize(
    "field,replacement",
    (
        ("id", "different/model"),
        ("model_revision", "phase4e-saracura-ranker.v1." + "0" * 64),
        ("checkpoint_sha256", "0" * 64),
        ("training_manifest_internal_sha256", "0" * 64),
        ("training_manifest_file_sha256", "0" * 64),
        ("capsule_descriptor_file_sha256", "0" * 64),
        ("capsule_descriptor_internal_sha256", "0" * 64),
        ("conformance_manifest_file_sha256", "0" * 64),
        ("conformance_vectors_sha256", "0" * 64),
        ("architecture", "different-architecture"),
        ("foundation_encoder", "different/encoder"),
        ("encoder_revision", "0" * 40),
        ("encoder_snapshot_complete_sha256", "0" * 64),
        ("training_source_commit", "0" * 40),
        ("disposition", "different-disposition"),
        ("capsule_files", []),
    ),
)
def test_candidate_registry_rejects_every_changed_approved_field(
    field: str, replacement: object
) -> None:
    payload = json.loads(
        (Path(__file__).parents[1] / "src" / "saracura" / "saracura-candidates.v1.json").read_text(
            encoding="utf-8"
        )
    )
    payload["candidates"][0][field] = replacement
    with pytest.raises(SaracuraBackendError):
        load_saracura_candidate(json.dumps(payload).encode("utf-8"))


def test_synthetic_capsule_requires_exact_files_descriptor_and_canonical_bindings(
    tmp_path: Path,
) -> None:
    root = tmp_path / "capsule"
    candidate = _synthetic_capsule(root)
    verified = verify_training_capsule(root, candidate)
    assert set(verified.values) == CAPSULE_FILE_SET
    assert verified.manifest["outcome"] == "passed"
    with pytest.raises(TypeError):
        cast(dict[str, object], verified.manifest)["outcome"] = "altered"
    with pytest.raises(TypeError):
        cast(dict[str, object], verified.manifest["base_encoder"])["id"] = "altered"

    extra = root / "extra.json"
    extra.write_bytes(b"{}\n")
    extra.chmod(0o600)
    with pytest.raises(SaracuraBackendError):
        verify_training_capsule(root, candidate)


@pytest.mark.parametrize(
    "filename",
    (
        "capsule-descriptor.json",
        "training-manifest.json",
        "checkpoint.safetensors",
        "conformance-manifest.json",
        "conformance-vectors.safetensors",
    ),
)
def test_synthetic_capsule_rejects_each_descriptor_manifest_checkpoint_and_conformance_change(
    tmp_path: Path, filename: str
) -> None:
    root = tmp_path / filename
    candidate = _synthetic_capsule(root)
    target = root / filename
    target.write_bytes(target.read_bytes() + b"x")
    target.chmod(0o600)
    with pytest.raises(SaracuraBackendError):
        verify_training_capsule(root, candidate)


def test_render_choice_is_byte_identical_without_training_row_identifiers() -> None:
    task = UniversalTask(
        task_id="training-id",
        family_id="family-id",
        locale="en",
        domain="email_triage",
        instruction="Choose a route.",
        state={"summary": "synthetic"},
        criteria=(
            ChoiceCriterion(id="one", description="First route."),
            ChoiceCriterion(id="two", description="Second route."),
        ),
        selected_criterion_id="one",
    )
    assert render_choice(
        locale=task.locale,
        domain=task.domain,
        instruction=task.instruction,
        state=task.state,
        criteria=task.criteria,
    ) == render_task(task)


def test_request_limits_fail_before_backend_preparation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = SaracuraUniversalBackend(
        encoder_snapshot=tmp_path / "snapshot",
        training_capsule=tmp_path / "capsule",
        device="cpu",
    )
    monkeypatch.setattr(
        backend, "prepare", lambda: (_ for _ in ()).throw(AssertionError("prepared"))
    )
    request = _request(criteria_count=9)
    with pytest.raises(SaracuraError) as captured:
        backend.score_universal_choice(request, request.questions[0], b"")
    assert captured.value.payload.code == ErrorCode.CARDINALITY_EXCEEDED

    invalid_domain = _request(domain="unapproved")
    with pytest.raises(SaracuraError) as captured:
        backend.score_universal_choice(invalid_domain, invalid_domain.questions[0], b"")
    assert captured.value.payload.code == ErrorCode.DOMAIN_UNVERIFIED


def test_direct_backend_checks_exact_model_identity_before_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = SaracuraUniversalBackend(
        encoder_snapshot=tmp_path / "snapshot",
        training_capsule=tmp_path / "capsule",
        device="cpu",
    )
    monkeypatch.setattr(
        backend, "prepare", lambda: (_ for _ in ()).throw(AssertionError("prepared"))
    )
    request = _request(model="different-revision")
    with pytest.raises(SaracuraError) as captured:
        backend.score_universal_choice(request, request.questions[0], b"")
    assert captured.value.payload.code == ErrorCode.MODEL_NOT_FOUND


def test_cli_rejects_saracura_capacity_and_exclusive_arguments_before_capsule_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    capsule_reads: list[Path] = []

    def unexpected_capsule_read(path: Path) -> None:
        capsule_reads.append(path)
        raise AssertionError("capsule verifier must not run during CLI preflight")

    monkeypatch.setattr(cli, "verify_training_capsule", unexpected_capsule_read)
    common = [
        "decide",
        "--backend",
        "saracura-universal",
        "--encoder-snapshot",
        str(tmp_path / "snapshot"),
        "--training-capsule",
        str(tmp_path / "capsule"),
        "--device",
        "cpu",
    ]
    over_capacity = tmp_path / "over-capacity.json"
    over_capacity.write_text(_request(criteria_count=9).model_dump_json(), encoding="utf-8")

    assert cli.main([*common, "--request", str(over_capacity)]) == 2
    assert capsule_reads == []
    assert json.loads(capsys.readouterr().err)["error"]["code"] == ErrorCode.CARDINALITY_EXCEEDED

    valid = tmp_path / "valid.json"
    valid.write_text(_request().model_dump_json(), encoding="utf-8")
    assert cli.main([*common, "--request", str(valid), "--calibration", "forbidden.json"]) == 2
    assert capsule_reads == []
    assert json.loads(capsys.readouterr().err)["error"]["code"] == ErrorCode.REQUEST_INVALID


@pytest.mark.parametrize(
    ("decision_request", "expected"),
    (
        (_request(locale="fr"), ErrorCode.LOCALE_UNVERIFIED),
        (_request(domain="unapproved"), ErrorCode.DOMAIN_UNVERIFIED),
        (_request(question_count=11), ErrorCode.CARDINALITY_EXCEEDED),
        (_request(criteria_count=9), ErrorCode.CARDINALITY_EXCEEDED),
        (_request(instruction="x" * 121), ErrorCode.CAPACITY_EXCEEDED),
        (_request(description="x" * 121), ErrorCode.CAPACITY_EXCEEDED),
        (_request(state={str(index): index for index in range(65)}), ErrorCode.CAPACITY_EXCEEDED),
        (_request(state={"summary": "x" * 201}), ErrorCode.CAPACITY_EXCEEDED),
        (_request(state={"summary": "Cafe\u0301"}), ErrorCode.REQUEST_INVALID),
    ),
)
def test_cli_rejects_every_saracura_request_only_limit_before_capsule_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    decision_request: DecisionRequest,
    expected: ErrorCode,
) -> None:
    request_path = tmp_path / "request.json"
    request_path.write_text(decision_request.model_dump_json(), encoding="utf-8")
    monkeypatch.setattr(
        cli,
        "verify_training_capsule",
        lambda _path: (_ for _ in ()).throw(AssertionError("capsule read")),
    )
    assert cli.main([*_cli_common(tmp_path), "--request", str(request_path)]) == 2
    assert json.loads(capsys.readouterr().err)["error"]["code"] == expected


@pytest.mark.parametrize("alias", ("latest", "main", "master"))
def test_candidate_registry_rejects_aliases_only_after_a_closed_registry_parse(alias: str) -> None:
    payload = json.loads(
        (Path(__file__).parents[1] / "src" / "saracura" / "saracura-candidates.v1.json").read_text(
            encoding="utf-8"
        )
    )
    payload["candidates"][0]["model_revision"] = alias
    with pytest.raises(SaracuraBackendError):
        load_saracura_candidate(json.dumps(payload).encode("utf-8"))


def test_candidate_registry_rejects_unknown_and_ambiguous_candidates() -> None:
    source = Path(__file__).parents[1] / "src" / "saracura" / "saracura-candidates.v1.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["candidates"][0]["unexpected"] = True
    with pytest.raises(SaracuraBackendError):
        load_saracura_candidate(json.dumps(payload).encode("utf-8"))

    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["candidates"].append(dict(payload["candidates"][0]))
    with pytest.raises(SaracuraBackendError):
        load_saracura_candidate(json.dumps(payload).encode("utf-8"))


@pytest.mark.parametrize("locale", ("pt-BR", "en"))
def test_saracura_structural_request_accepts_both_locales(locale: str) -> None:
    validate_saracura_request_structure(_request(locale=locale))


@pytest.mark.parametrize("domain", sorted(UNIVERSAL_DOMAINS))
def test_saracura_structural_request_accepts_every_closed_domain(domain: str) -> None:
    validate_saracura_request_structure(_request(domain=domain))


@pytest.mark.parametrize("criteria_count", (2, 8))
def test_saracura_structural_request_accepts_trained_option_counts(criteria_count: int) -> None:
    validate_saracura_request_structure(_request(criteria_count=criteria_count))


def test_saracura_structural_request_accepts_ten_questions() -> None:
    validate_saracura_request_structure(_request(question_count=10))


def test_saracura_structural_limits_map_cardinality_and_capacity_to_stable_codes() -> None:
    # The shared workflow/engine contract maps declared collection sizes to
    # CARDINALITY_EXCEEDED.  Phase 4E.3c calls the text/state/token envelope a
    # capacity, so every such overflow maps to CAPACITY_EXCEEDED instead.
    with pytest.raises(SaracuraError) as captured:
        validate_saracura_request_structure(_request(criteria_count=9))
    assert captured.value.payload.code == ErrorCode.CARDINALITY_EXCEEDED

    with pytest.raises(SaracuraError) as captured:
        validate_saracura_request_structure(_request(question_count=11))
    assert captured.value.payload.code == ErrorCode.CARDINALITY_EXCEEDED

    with pytest.raises(SaracuraError) as captured:
        validate_saracura_request_structure(_request(instruction="x" * 121))
    assert captured.value.payload.code == ErrorCode.CAPACITY_EXCEEDED

    with pytest.raises(SaracuraError) as captured:
        validate_saracura_request_structure(_request(description="x" * 121))
    assert captured.value.payload.code == ErrorCode.CAPACITY_EXCEEDED

    with pytest.raises(SaracuraError) as captured:
        validate_saracura_request_structure(
            _request(state={str(index): index for index in range(65)})
        )
    assert captured.value.payload.code == ErrorCode.CAPACITY_EXCEEDED

    with pytest.raises(SaracuraError) as captured:
        validate_saracura_request_structure(_request(state={"summary": "x" * 201}))
    assert captured.value.payload.code == ErrorCode.CAPACITY_EXCEEDED


def test_saracura_structural_limits_accept_exact_codepoint_and_utf8_boundaries() -> None:
    question = (
        _request()
        .questions[0]
        .model_copy(
            update={
                "instruction": "😀" * 120,
                "criteria": (
                    ChoiceCriterion(id="first", description="😀" * 120),
                    ChoiceCriterion(id="second", description="😀" * 120),
                ),
            }
        )
    )
    request = _request().model_copy(update={"questions": (question,)})
    assert len(question.instruction) == 120
    assert len(question.instruction.encode("utf-8")) == 480
    assert len(question.criteria[0].description) == 120
    assert len(question.criteria[0].description.encode("utf-8")) == 480
    validate_saracura_request_structure(request)


def test_saracura_structural_request_rejects_non_nfc_before_artifacts() -> None:
    with pytest.raises(SaracuraError) as captured:
        validate_saracura_request_structure(_request(state={"summary": "Cafe\u0301"}))
    assert captured.value.payload.code == ErrorCode.REQUEST_INVALID


def test_saracura_model_identity_is_separate_from_structural_validation() -> None:
    request = _request(model="different-revision")
    validate_saracura_request_structure(request)
    with pytest.raises(SaracuraError) as captured:
        validate_saracura_model_identity(request, load_saracura_candidate())
    assert captured.value.payload.code == ErrorCode.MODEL_NOT_FOUND


@pytest.mark.parametrize("unsafe", ("mode", "symlink", "hardlink", "fifo", "size"))
def test_capsule_reader_rejects_custody_and_byte_failures(tmp_path: Path, unsafe: str) -> None:
    root = tmp_path / unsafe
    candidate = _synthetic_capsule(root)
    target = root / "architecture.json"
    if unsafe == "mode":
        target.chmod(0o644)
    elif unsafe == "symlink":
        target.unlink()
        os.symlink(tmp_path / "missing", target)
    elif unsafe == "hardlink":
        linked = tmp_path / "linked-architecture.json"
        os.link(target, linked)
    elif unsafe == "fifo":
        target.unlink()
        os.mkfifo(target, 0o600)
    else:
        target.write_bytes(target.read_bytes() + b"x")
        target.chmod(0o600)
    with pytest.raises(SaracuraBackendError):
        verify_training_capsule(root, candidate)


def test_capsule_verifier_never_constructs_a_socket(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "capsule"
    candidate = _synthetic_capsule(root)

    def blocked(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket, "socket", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    assert verify_training_capsule(root, candidate).candidate == candidate


@pytest.mark.parametrize("changed", ("marker", "repository", "revision"))
def test_snapshot_receipt_or_encoder_registry_change_cannot_match_fixed_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changed: str
) -> None:
    backend = SaracuraUniversalBackend(
        encoder_snapshot=tmp_path / "snapshot",
        training_capsule=tmp_path / "capsule",
        device="cpu",
    )
    marker = (
        b"different-reviewed-snapshot-receipt"
        if changed == "marker"
        else bytes.fromhex(backend._candidate.encoder_snapshot_complete_sha256)
    )
    snapshot_candidate = SimpleNamespace(
        repository=(
            "different/encoder"
            if changed == "repository"
            else backend._candidate.foundation_encoder
        ),
        revision=("0" * 40 if changed == "revision" else backend._candidate.encoder_revision),
    )
    monkeypatch.setattr(
        saracura_universal,
        "read_verified_minilm_snapshot_with_marker",
        lambda _path: (snapshot_candidate, MappingProxyType({}), marker),
    )
    with pytest.raises(SaracuraBackendError):
        backend._verify_snapshot()


def test_direct_construction_verifies_capsule_once_then_clears_paths_without_reopen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot_path = tmp_path / "snapshot"
    capsule_path = tmp_path / "capsule"
    backend = SaracuraUniversalBackend(
        encoder_snapshot=snapshot_path,
        training_capsule=capsule_path,
        device="cpu",
    )
    marker = b"bound-snapshot-marker"
    candidate = replace(
        backend._candidate,
        encoder_snapshot_complete_sha256=_sha(marker),
    )
    backend._candidate = candidate
    verified = VerifiedSaracuraCapsule(
        candidate=candidate,
        values=MappingProxyType({}),
        manifest=MappingProxyType({}),
        conformance=MappingProxyType({}),
    )
    calls: list[str] = []

    def verify(path: Path, bound: SaracuraCandidate) -> VerifiedSaracuraCapsule:
        calls.append(f"capsule:{path.name}:{bound.id}")
        return verified

    monkeypatch.setattr(
        saracura_universal,
        "verify_training_capsule",
        verify,
    )
    snapshot_candidate = SimpleNamespace(
        repository=candidate.foundation_encoder,
        revision=candidate.encoder_revision,
    )

    def read_snapshot(path: Path) -> tuple[object, Mapping[str, bytes], bytes]:
        calls.append(f"snapshot:{path.name}")
        return snapshot_candidate, MappingProxyType({}), marker

    monkeypatch.setattr(
        saracura_universal, "read_verified_minilm_snapshot_with_marker", read_snapshot
    )

    def assert_memory_only() -> None:
        assert backend._training_capsule is None
        assert backend._encoder_snapshot is None
        calls.append("memory-only")

    def blocked_network(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket, "socket", blocked_network)
    monkeypatch.setattr(socket, "create_connection", blocked_network)
    monkeypatch.setattr(backend, "_load_ml_components", assert_memory_only)
    monkeypatch.setattr(backend, "_load_ranker_and_conformance", assert_memory_only)
    backend.prepare()
    backend.prepare()
    assert calls == [
        "capsule:capsule:saracura/universal-ranker",
        "snapshot:snapshot",
        "memory-only",
        "memory-only",
    ]
    assert backend._training_capsule is None
    assert backend._encoder_snapshot is None
    backend.close()


def test_forged_verified_capsule_cannot_bypass_handoff_verification(tmp_path: Path) -> None:
    candidate = load_saracura_candidate()
    forged = VerifiedSaracuraCapsule(
        candidate=candidate,
        values=MappingProxyType({name: b"forged" for name in CAPSULE_FILE_SET}),
        manifest=MappingProxyType({}),
        conformance=MappingProxyType({}),
    )
    assert "verified_capsule" not in inspect.signature(SaracuraUniversalBackend).parameters
    with pytest.raises(SaracuraError) as captured:
        SaracuraUniversalBackend._from_verified_capsule(
            encoder_snapshot=tmp_path / "must-not-read",
            verified_capsule=forged,
            device="cpu",
        )
    assert captured.value.payload.code == ErrorCode.BACKEND_UNAVAILABLE


def test_cli_handoff_revalidates_bytes_and_retains_no_capsule_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capsule_path = tmp_path / "capsule"
    candidate = _synthetic_capsule(capsule_path)
    verified = verify_training_capsule(capsule_path, candidate)
    monkeypatch.setattr(saracura_universal, "load_saracura_candidate", lambda: candidate)
    backend = SaracuraUniversalBackend._from_verified_capsule(
        encoder_snapshot=tmp_path / "snapshot",
        verified_capsule=verified,
        device="cpu",
    )
    assert backend._training_capsule is None
    assert backend._verified_capsule is not verified
    assert backend._verified_capsule is not None
    assert backend._verified_capsule.values is not verified.values
    backend.close()


class _CaptureTokenizer:
    def __init__(self, rows: list[object]) -> None:
        self.rows = rows
        self.calls: list[tuple[list[str], dict[str, object]]] = []

    def __call__(self, texts: list[str], **kwargs: object) -> dict[str, list[object]]:
        self.calls.append((texts, kwargs))
        return {"input_ids": self.rows}


@pytest.mark.parametrize(
    ("rows", "code"),
    (
        ([[0] * 129, [1]], ErrorCode.CAPACITY_EXCEEDED),
        ([[0], [1] * 97], ErrorCode.CAPACITY_EXCEEDED),
        ([["not-an-id"], [1]], ErrorCode.BACKEND_UNAVAILABLE),
        ([[0], [1], [2]], ErrorCode.BACKEND_UNAVAILABLE),
        ([[], [1]], ErrorCode.BACKEND_UNAVAILABLE),
        ([None, [1]], ErrorCode.BACKEND_UNAVAILABLE),
    ),
)
def test_tokenizer_enforces_complete_untruncated_capacity_before_encoder(
    tmp_path: Path, rows: list[object], code: ErrorCode
) -> None:
    backend = SaracuraUniversalBackend(
        encoder_snapshot=tmp_path / "snapshot",
        training_capsule=tmp_path / "capsule",
        device="cpu",
    )
    tokenizer = _CaptureTokenizer(rows)
    backend._tokenizer = tokenizer
    backend._torch = object()
    rendered = RenderedTask(context="context", criteria=("first",))
    with pytest.raises(SaracuraError) as captured:
        backend._tokenize_joint(rendered)
    assert captured.value.payload.code == code
    assert tokenizer.calls[0][1]["truncation"] is False
    assert tokenizer.calls[0][1]["padding"] is False


@pytest.mark.parametrize(
    "failure",
    [
        TypeError("tokenizer-canary"),
        ValueError("tokenizer-canary"),
        RuntimeError("tokenizer-canary"),
    ],
)
def test_tokenizer_exceptions_are_sanitized_backend_unavailable(
    tmp_path: Path, failure: BaseException
) -> None:
    backend = SaracuraUniversalBackend(
        encoder_snapshot=tmp_path / "snapshot",
        training_capsule=tmp_path / "capsule",
        device="cpu",
    )

    class BrokenTokenizer:
        def __call__(self, *_args: object, **_kwargs: object) -> object:
            raise failure

    backend._tokenizer = BrokenTokenizer()
    backend._torch = object()
    with pytest.raises(SaracuraError) as captured:
        backend._tokenize_joint(RenderedTask(context="context", criteria=("criterion",)))
    error = captured.value
    assert error.payload.code == ErrorCode.BACKEND_UNAVAILABLE
    assert error.__cause__ is None
    assert error.__context__ is None
    assert "tokenizer-canary" not in str(error)


def test_renderer_and_engine_preserve_order_for_repeated_descriptions_and_ties() -> None:
    request = _request()
    question = request.questions[0].model_copy(
        update={
            "criteria": (
                ChoiceCriterion(id="first", description="same description"),
                ChoiceCriterion(id="second", description="same description"),
            )
        }
    )
    request = request.model_copy(update={"questions": (question,)})
    rendered = render_choice(
        locale=request.locale,
        domain=request.domain,
        instruction=question.instruction,
        state=request.state,
        criteria=question.criteria,
    )
    assert rendered.criteria[0] != rendered.criteria[1]

    class TieBackend:
        capabilities = BackendCapabilities(
            execution_tier="universal",
            decision_types=frozenset({"choice"}),
            max_questions=10,
            max_criteria=8,
            execution_boundary="test",
            cold_warm_semantics="test",
            quality_claims=False,
            dynamic_workflows=frozenset(
                {(SARACURA_UNIVERSAL_WORKFLOW_ID, SARACURA_UNIVERSAL_WORKFLOW_REVISION)}
            ),
        )
        model = SaracuraUniversalBackend(
            encoder_snapshot=Path("/not-read"), training_capsule=Path("/not-read"), device="cpu"
        ).model

        def validate_request(self, _request: DecisionRequest) -> None:
            pass

        def score_universal_choice(
            self, _request: DecisionRequest, scored: ChoiceQuestion, _state: bytes
        ) -> ScoredChoice:
            return ScoredChoice(
                question_id=scored.id,
                raw_scores={criterion.id: 0.0 for criterion in scored.criteria},
            )

    response = DecisionEngine(
        backend=TieBackend(), workflows=default_workflows(), calibrations={}
    ).decide(request)
    assert response.answers[0].value == "first"


def test_release_restores_threads_and_releases_the_resident_slot(tmp_path: Path) -> None:
    backend = SaracuraUniversalBackend(
        encoder_snapshot=tmp_path / "snapshot",
        training_capsule=tmp_path / "capsule",
        device="cpu",
    )

    class ThreadTorch:
        def __init__(self) -> None:
            self.restored: list[int] = []

        def set_num_threads(self, value: int) -> None:
            self.restored.append(value)

    previous_resident = saracura_universal._MODEL_RESIDENT
    try:
        saracura_universal._MODEL_RESIDENT = True
        backend._owns_resident_slot = True
        backend._prior_threads = 3
        torch = ThreadTorch()
        backend._torch = torch
        backend._release_loaded_state()
        assert saracura_universal._MODEL_RESIDENT is False
        assert torch.restored == [3]
        assert backend._torch is None
        assert backend._prior_threads is None
    finally:
        saracura_universal._MODEL_RESIDENT = previous_resident


def _cli_common(tmp_path: Path) -> list[str]:
    return [
        "decide",
        "--backend",
        "saracura-universal",
        "--encoder-snapshot",
        str(tmp_path / "snapshot"),
        "--training-capsule",
        str(tmp_path / "capsule"),
        "--device",
        "cpu",
    ]


@pytest.mark.parametrize(
    "forbidden",
    (
        ("--calibration", "calibration.json"),
        ("--model-snapshot", "laya-snapshot"),
        ("--training-manifest", "manifest.json"),
        ("--checkpoint", "checkpoint.safetensors"),
    ),
)
def test_cli_rejects_every_exclusive_backend_argument_before_capsule_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    forbidden: tuple[str, str],
) -> None:
    request_path = tmp_path / "request.json"
    request_path.write_text(_request().model_dump_json(), encoding="utf-8")
    monkeypatch.setattr(
        cli,
        "verify_training_capsule",
        lambda _path: (_ for _ in ()).throw(AssertionError("capsule read")),
    )
    assert cli.main([*_cli_common(tmp_path), "--request", str(request_path), *forbidden]) == 2
    assert json.loads(capsys.readouterr().err)["error"]["code"] == ErrorCode.REQUEST_INVALID


def test_cli_orders_structural_validation_then_capsule_then_model_before_snapshot_or_ml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    request_path = tmp_path / "request.json"
    request_path.write_text(
        _request(model="different-revision").model_dump_json(), encoding="utf-8"
    )
    candidate = load_saracura_candidate()
    verified = VerifiedSaracuraCapsule(
        candidate=candidate,
        values=MappingProxyType({}),
        manifest=MappingProxyType({}),
        conformance=MappingProxyType({}),
    )
    events: list[str] = []

    def verify(_path: Path) -> VerifiedSaracuraCapsule:
        events.append("capsule")
        return verified

    def backend_construction_must_not_run(*_args: object, **_kwargs: object) -> object:
        events.append("backend")
        raise AssertionError("snapshot or ML construction")

    monkeypatch.setattr(
        cli,
        "verify_training_capsule",
        verify,
    )
    monkeypatch.setattr(cli, "_require_saracura_arguments", backend_construction_must_not_run)
    assert cli.main([*_cli_common(tmp_path), "--request", str(request_path)]) == 2
    assert events == ["capsule"]
    assert json.loads(capsys.readouterr().err)["error"]["code"] == ErrorCode.MODEL_NOT_FOUND


def test_cli_reads_one_verified_capsule_before_snapshot_or_ml_without_retaining_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capsule_path = tmp_path / "valid-capsule"
    candidate = _synthetic_capsule(capsule_path)
    verified = verify_training_capsule(capsule_path, candidate)
    request_path = tmp_path / "request.json"
    request_path.write_text(
        _request(model=candidate.model_revision).model_dump_json(), encoding="utf-8"
    )
    capsule_reads: list[Path] = []
    received: list[SaracuraUniversalBackend] = []

    def verify(path: Path) -> VerifiedSaracuraCapsule:
        capsule_reads.append(path)
        return verified

    monkeypatch.setattr(cli, "verify_training_capsule", verify)
    monkeypatch.setattr(saracura_universal, "load_saracura_candidate", lambda: candidate)

    class FakeEngine:
        def __init__(self, *, backend: SaracuraUniversalBackend, **_kwargs: object) -> None:
            received.append(backend)

        def decide(self, *_args: object, **_kwargs: object) -> object:
            assert received[0]._training_capsule is None
            assert received[0]._encoder_snapshot == tmp_path / "snapshot"
            assert received[0]._torch is None
            return SimpleNamespace(model_dump_json=lambda **_json_kwargs: "{}")

    monkeypatch.setattr(cli, "DecisionEngine", FakeEngine)
    assert cli.main([*_cli_common(tmp_path), "--request", str(request_path)]) == 0
    assert len(capsule_reads) == 1
    assert received[0]._training_capsule is None
    assert received[0]._encoder_snapshot is None
    assert received[0]._torch is None


def test_cli_redacts_capsule_failure_and_closes_describe_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    private_marker = str(tmp_path / "private-capsule-detail")
    request_path = tmp_path / "request.json"
    request_path.write_text(_request().model_dump_json(), encoding="utf-8")
    monkeypatch.setattr(
        cli,
        "verify_training_capsule",
        lambda _path: (_ for _ in ()).throw(SaracuraBackendError(private_marker)),
    )
    assert cli.main([*_cli_common(tmp_path), "--request", str(request_path)]) == 2
    assert private_marker not in capsys.readouterr().err

    closed: list[bool] = []

    class DescribeBackend:
        model = SaracuraUniversalBackend(
            encoder_snapshot=Path("/not-read"), training_capsule=Path("/not-read"), device="cpu"
        ).model

        def prepare(self) -> None:
            pass

        def close(self) -> None:
            closed.append(True)

    monkeypatch.setattr(cli, "_require_saracura_arguments", lambda _values: DescribeBackend())
    assert (
        cli.main(
            [
                "describe-backend",
                "--backend",
                "saracura-universal",
                "--encoder-snapshot",
                str(tmp_path / "snapshot"),
                "--training-capsule",
                str(tmp_path / "capsule"),
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["workflow"]["criteria_per_question"]["maximum"] == 8
    assert payload["response"]["automation_allowed"] is False
    assert closed == [True]


def test_cli_closes_saracura_describe_backend_after_prepare_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    closed: list[bool] = []

    class FailingDescribeBackend:
        def prepare(self) -> None:
            raise SaracuraError(ErrorCode.BACKEND_UNAVAILABLE, "safe failure", "/model")

        def close(self) -> None:
            closed.append(True)

    monkeypatch.setattr(
        cli,
        "_require_saracura_arguments",
        lambda _values: FailingDescribeBackend(),
    )
    assert (
        cli.main(
            [
                "describe-backend",
                "--backend",
                "saracura-universal",
                "--encoder-snapshot",
                str(tmp_path / "snapshot"),
                "--training-capsule",
                str(tmp_path / "capsule"),
                "--device",
                "cpu",
            ]
        )
        == 2
    )
    assert closed == [True]


def test_crossed_phase4e_and_minilm_capsules_fail_before_snapshot_loading(tmp_path: Path) -> None:
    root = tmp_path / "phase4e-capsule"
    _synthetic_capsule(root)
    with pytest.raises(SaracuraError) as captured:
        MiniLMRoutingBackend(
            encoder_snapshot=tmp_path / "must-not-read",
            training_capsule=root,
            device="cpu",
        )
    assert captured.value.payload.code == ErrorCode.BACKEND_UNAVAILABLE


def test_cli_rejects_human_minilm_capsule_before_saracura_snapshot_or_ml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capsule = tmp_path / "human-minilm-capsule"
    capsule.mkdir(mode=0o700)
    for name in HUMAN_TRAINING_CAPSULE_FILES:
        artifact = capsule / name
        artifact.write_bytes(b"synthetic")
        artifact.chmod(0o600)
    request_path = tmp_path / "request.json"
    request_path.write_text(_request().model_dump_json(), encoding="utf-8")
    monkeypatch.setattr(
        cli,
        "_require_saracura_arguments",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("snapshot or ML load")),
    )
    assert (
        cli.main(
            [
                "decide",
                "--backend",
                "saracura-universal",
                "--request",
                str(request_path),
                "--encoder-snapshot",
                str(tmp_path / "must-not-read"),
                "--training-capsule",
                str(capsule),
                "--device",
                "cpu",
            ]
        )
        == 2
    )


def test_saracura_direct_import_help_and_invalid_request_keep_laya_and_optional_ml_unloaded() -> (
    None
):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import saracura.backends.saracura_universal; "
            "from saracura.cli import _parser,main; _parser().format_help(); "
            "assert main(['decide','--backend','saracura-universal','--request','missing.json',"
            "'--encoder-snapshot','snapshot','--training-capsule','capsule',"
            "'--device','cpu']) == 2; "
            "assert not {'saracura.backends.laya','torch','transformers','tokenizers',"
            "'safetensors'} & set(sys.modules)",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_saracura_valid_cli_pre_ml_ordering_does_not_import_laya() -> None:
    code = """
import sys
from types import MappingProxyType

import saracura.cli as cli
from saracura.backends.saracura_universal import VerifiedSaracuraCapsule
from saracura.backends.saracura_universal import load_saracura_candidate
from saracura.contracts import ChoiceCriterion, ChoiceQuestion, DecisionRequest
from saracura.contracts import ErrorCode, SaracuraError, WorkflowReference

candidate = load_saracura_candidate()
request = DecisionRequest(
    api_version="v1alpha1",
    model=candidate.model_revision,
    mode="research",
    locale="en",
    domain="email_triage",
    workflow=WorkflowReference(id="universal-choice", revision="phase4e-saracura-ranker.v1"),
    state={"summary": "public synthetic"},
    questions=(
        ChoiceQuestion(
            id="route",
            type="choice",
            instruction="Choose a route.",
            criteria=(
                ChoiceCriterion(id="first", description="First route."),
                ChoiceCriterion(id="second", description="Second route."),
            ),
        ),
    ),
)
verified = VerifiedSaracuraCapsule(
    candidate=candidate,
    values=MappingProxyType({}),
    manifest=MappingProxyType({}),
    conformance=MappingProxyType({}),
)
events = []

def verify(_path):
    events.append("capsule")
    return verified

def backend_factory(*_args, **_kwargs):
    events.append("backend")
    raise SaracuraError(ErrorCode.BACKEND_UNAVAILABLE, "pre-ML sentinel", "/model")

def read_request(*_args, **_kwargs):
    return request.model_dump_json().encode("utf-8")

cli.verify_training_capsule = verify
cli._require_saracura_arguments = backend_factory
cli.read_public_external_file = read_request
assert cli.main([
    "decide", "--backend", "saracura-universal", "--request", "request.json",
    "--encoder-snapshot", "snapshot", "--training-capsule", "capsule", "--device", "cpu",
]) == 2
assert events == ["capsule", "backend"], events
assert not {
    "saracura.backends.laya", "torch", "transformers", "tokenizers", "safetensors"
} & set(sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_backends_preserves_lazy_laya_public_export() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import saracura.backends; "
            "assert 'saracura.backends.laya' not in sys.modules; "
            "from saracura.backends import LayaUniversalBackend; "
            "assert LayaUniversalBackend.__name__ == 'LayaUniversalBackend'; "
            "assert 'saracura.backends.laya' in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
