"""Focused offline tests for the Phase 4A opt-in MiniLM boundary."""

from __future__ import annotations

import hashlib
import json
import os
import socket
from pathlib import Path
from types import MappingProxyType
from typing import ClassVar, cast

import pytest
from pydantic import JsonValue, ValidationError

import saracura.cli as cli
import saracura.verified_bytes as verified_bytes
from saracura.backends.base import (
    BackendCalibrationMetadata,
    BackendCapabilities,
    EncodedState,
    ScoredChoice,
)
from saracura.backends.minilm import MAX_STATE_FRAME_BYTES, MiniLMRoutingBackend
from saracura.calibration import (
    CalibrationArtifact,
    IdentityCalibrationArtifact,
    TemperatureParameters,
    create_identity_calibration,
    load_calibration,
)
from saracura.contracts import (
    ChoiceQuestion,
    DecisionRequest,
    ErrorCode,
    ModelReference,
    SaracuraError,
    parse_request_json,
)
from saracura.runtime import (
    MINILM_ROUTING_LABELS,
    MINILM_ROUTING_QUESTION,
    MINILM_ROUTING_WORKFLOW_ID,
    MINILM_ROUTING_WORKFLOW_REVISION,
    PHASE4A_IDENTITY_PROFILE,
    DecisionEngine,
    default_workflows,
    known_scaling_questions,
)
from saracura.runtime.engine import calibration_context
from saracura.serialization import canonical_json_bytes, frame_segments
from saracura.verified_bytes import (
    MiniLMCandidate,
    RegistryFile,
    VerifiedBytesError,
    load_verified_minilm_bytes,
    package_registry_bytes,
    validate_training_manifest,
)
from tests.helpers import decision_request


class _NonFixtureBackend:
    def __init__(self) -> None:
        self.encode_calls = 0
        self.score_calls = 0
        self._model = ModelReference(
            id="test-local-backend",
            revision="test-local-backend-v1",
            checkpoint_sha256="a" * 64,
        )
        self._metadata = BackendCalibrationMetadata(
            architecture_config_sha256="b" * 64,
            tokenizer_revision="test-tokenizer-v1",
            truncation_policy_id="test-truncation-v1",
            precision="test-cpu-path-v1",
            quantization="none",
            output_transform="test-softmax-v1",
        )
        self._capabilities = BackendCapabilities(
            decision_types=frozenset({"choice"}),
            max_questions=1,
            max_criteria=20,
            execution_boundary="test",
            cold_warm_semantics="test",
            quality_claims=False,
        )

    @property
    def capabilities(self) -> BackendCapabilities:
        return self._capabilities

    @property
    def model(self) -> ModelReference:
        return self._model

    @property
    def calibration_metadata(self) -> BackendCalibrationMetadata:
        return self._metadata

    def encode_state(self, state_payload: bytes) -> EncodedState:
        self.encode_calls += 1
        return EncodedState(payload=state_payload, input_tokens=7)

    def score_choice(
        self,
        encoded_state: EncodedState,
        question: ChoiceQuestion,
        question_payload: bytes,
    ) -> ScoredChoice:
        self.score_calls += 1
        return ScoredChoice(
            question_id=question.id,
            raw_scores={
                criterion.id: float(index) for index, criterion in enumerate(question.criteria)
            },
        )


class _LegacyBackend(_NonFixtureBackend):
    @property
    def calibration_metadata(self) -> BackendCalibrationMetadata:
        raise AttributeError("legacy backend")


class _ExplodingMetadataBackend(_NonFixtureBackend):
    @property
    def calibration_metadata(self) -> BackendCalibrationMetadata:
        raise ValueError("untrusted backend metadata failure")


class _CliMiniLMBackend:
    instances: ClassVar[list[_CliMiniLMBackend]] = []

    def __init__(self, **_kwargs: object) -> None:
        self.instances.append(self)
        self.encode_calls = 0
        self.score_calls = 0
        self._model = ModelReference(
            id="saracura-minilm-routing",
            revision="minilm-routing-v1." + "a" * 64,
            checkpoint_sha256="b" * 64,
        )
        self._metadata = BackendCalibrationMetadata(
            architecture_config_sha256="c" * 64,
            tokenizer_revision="minilm-tokenizer-test-v1",
            truncation_policy_id="padding-truncation-max128-v1",
            precision="encoder-fp32-cpu-pool-fp32-cpu-head-fp32-cpu-v1",
            quantization="none",
            output_transform="linear-logits-softmax-v1",
        )
        self._capabilities = BackendCapabilities(
            decision_types=frozenset({"choice"}),
            max_questions=1,
            max_criteria=5,
            execution_boundary="test-local-minilm",
            cold_warm_semantics="test-no-cache",
            quality_claims=False,
        )

    @property
    def capabilities(self) -> BackendCapabilities:
        return self._capabilities

    @property
    def model(self) -> ModelReference:
        return self._model

    @property
    def calibration_metadata(self) -> BackendCalibrationMetadata:
        return self._metadata

    @property
    def execution_path(self) -> str:
        return self._metadata.precision

    def encode_state(self, state_payload: bytes) -> EncodedState:
        self.encode_calls += 1
        return EncodedState(payload=state_payload, input_tokens=3)

    def score_choice(
        self,
        encoded_state: EncodedState,
        question: ChoiceQuestion,
        question_payload: bytes,
    ) -> ScoredChoice:
        self.score_calls += 1
        return ScoredChoice(
            question_id=question.id,
            raw_scores={label: float(index) for index, label in enumerate(MINILM_ROUTING_LABELS)},
        )


def _artifact_for_nonfixture(
    backend: _NonFixtureBackend, *, changed_precision: bool = False
) -> tuple[DecisionRequest, CalibrationArtifact]:
    request = decision_request(known_scaling_questions(1)).model_copy(
        update={"model": backend.model.revision}
    )
    question = request.questions[0]
    context = calibration_context(
        request,
        question.id,
        len(question.criteria),
        backend,
        PHASE4A_IDENTITY_PROFILE,
    )
    artifact = CalibrationArtifact(
        **context.model_dump(),
        schema_version=1,
        calibration_id="test-temperature-v1",
        status="fixture_only",
        method="temperature-scaling",
        parameters=TemperatureParameters(temperature=1.0),
        fit_sample_count=1,
        evaluation_sample_count=1,
        fit_independent_state_count=1,
        evaluation_independent_state_count=1,
        minimum_independent_state_count=1,
        metrics_before={},
        metrics_after={},
        created_at="2026-09-21T00:00:00Z",
    )
    if changed_precision:
        artifact = artifact.model_copy(update={"precision": "wrong-path-v1"})
    return request, artifact


def test_phase4a_profile_and_package_registry_are_exact() -> None:
    expected = {
        "dataset_id": "no-fit-synthetic-identity",
        "dataset_revision": "phase4a.v1",
        "fit": [],
        "evaluation": [],
        "purpose": "identity-transform-only",
    }
    assert (
        PHASE4A_IDENTITY_PROFILE.split_manifest_sha256
        == hashlib.sha256(canonical_json_bytes(cast(JsonValue, expected))).hexdigest()
    )
    assert (
        package_registry_bytes()
        == Path("benchmarks/manifests/encoder-candidates.v1.json").read_bytes()
    )


def test_phase4a_workflow_order_is_semantic_at_the_registry_boundary() -> None:
    request = decision_request(
        (MINILM_ROUTING_QUESTION,),
        workflow_id=MINILM_ROUTING_WORKFLOW_ID,
        workflow_revision=MINILM_ROUTING_WORKFLOW_REVISION,
        domain="support",
    )
    default_workflows().validate(request)
    reordered = MINILM_ROUTING_QUESTION.model_copy(
        update={"criteria": tuple(reversed(MINILM_ROUTING_QUESTION.criteria))}
    )
    changed = request.model_copy(update={"questions": (reordered,)})
    with pytest.raises(SaracuraError) as captured:
        default_workflows().validate(changed)
    assert captured.value.payload.code == ErrorCode.SCHEMA_UNSUPPORTED


@pytest.mark.parametrize(
    "raw",
    [
        b'{"api_version":"v1alpha1","api_version":"v1alpha1"}',
        b'{"api_version":"v1alpha1","mode":"research","state":{"a":1,"a":2}}',
    ],
)
def test_request_json_rejects_recursive_duplicate_keys(raw: bytes) -> None:
    with pytest.raises(SaracuraError) as captured:
        parse_request_json(raw)
    assert captured.value.payload.code == ErrorCode.REQUEST_INVALID


def test_identity_calibration_is_derived_canonical_and_immutable(tmp_path: Path) -> None:
    backend = _NonFixtureBackend()
    request, _ = _artifact_for_nonfixture(backend)
    question = request.questions[0]
    context = calibration_context(
        request,
        question.id,
        len(question.criteria),
        backend,
        PHASE4A_IDENTITY_PROFILE,
    )
    path = tmp_path / "identity.json"
    artifact = create_identity_calibration(path, context, "2026-09-22T00:00:00Z")

    assert isinstance(artifact, IdentityCalibrationArtifact)
    assert artifact.status == "fixture_only"
    assert artifact.parameters.temperature == 1.0
    assert path.read_bytes() == canonical_json_bytes(artifact.model_dump(mode="json")) + b"\n"
    assert load_calibration(path, context) == artifact
    with pytest.raises(FileExistsError):
        create_identity_calibration(path, context, "2026-09-22T00:00:00Z")
    with pytest.raises(ValidationError):
        IdentityCalibrationArtifact.model_validate({**artifact.model_dump(), "fit_sample_count": 1})
    with pytest.raises(ValidationError):
        IdentityCalibrationArtifact.model_validate(
            {**artifact.model_dump(), "created_at": "2026-02-30T00:00:00Z"}
        )
    with pytest.raises(ValidationError):
        IdentityCalibrationArtifact.model_validate(
            {**artifact.model_dump(), "parameters": {"temperature": 0.9}}
        )


def test_nonfixture_profile_and_calibration_are_resolved_before_encoding() -> None:
    backend = _NonFixtureBackend()
    request, artifact = _artifact_for_nonfixture(backend)

    missing_profile = DecisionEngine(
        backend=backend,
        workflows=default_workflows(),
        calibrations={request.questions[0].id: artifact},
    )
    with pytest.raises(SaracuraError) as captured:
        missing_profile.decide(request)
    assert captured.value.payload.code == ErrorCode.CALIBRATION_MISSING
    assert backend.encode_calls == backend.score_calls == 0

    incompatible = _NonFixtureBackend()
    changed_request, changed_artifact = _artifact_for_nonfixture(
        incompatible,
        changed_precision=True,
    )
    engine = DecisionEngine(
        backend=incompatible,
        workflows=default_workflows(),
        calibrations={changed_request.questions[0].id: changed_artifact},
        calibration_profiles={changed_request.questions[0].id: PHASE4A_IDENTITY_PROFILE},
    )
    with pytest.raises(SaracuraError) as captured:
        engine.decide(changed_request)
    assert captured.value.payload.code == ErrorCode.CALIBRATION_INCOMPATIBLE
    assert incompatible.encode_calls == incompatible.score_calls == 0

    runnable = _NonFixtureBackend()
    runnable_request, runnable_artifact = _artifact_for_nonfixture(runnable)
    successful = DecisionEngine(
        backend=runnable,
        workflows=default_workflows(),
        calibrations={runnable_request.questions[0].id: runnable_artifact},
        calibration_profiles={runnable_request.questions[0].id: PHASE4A_IDENTITY_PROFILE},
    )
    response = successful.decide(runnable_request)
    assert response.usage.input_tokens == 7
    assert runnable.encode_calls == runnable.score_calls == 1


def test_legacy_backend_metadata_fails_before_encoding() -> None:
    seed = _NonFixtureBackend()
    request, artifact = _artifact_for_nonfixture(seed)
    backend = _LegacyBackend()
    engine = DecisionEngine(
        backend=backend,
        workflows=default_workflows(),
        calibrations={request.questions[0].id: artifact},
        calibration_profiles={request.questions[0].id: PHASE4A_IDENTITY_PROFILE},
    )
    with pytest.raises(SaracuraError) as captured:
        engine.decide(request)
    assert captured.value.payload.code == ErrorCode.BACKEND_UNAVAILABLE
    assert backend.encode_calls == backend.score_calls == 0


@pytest.mark.parametrize(
    "metadata",
    (
        BackendCalibrationMetadata(
            architecture_config_sha256="uppercase-sha256-is-invalid",
            tokenizer_revision="test-tokenizer-v1",
            truncation_policy_id="test-truncation-v1",
            precision="test-cpu-path-v1",
            quantization="none",
            output_transform="test-softmax-v1",
        ),
        BackendCalibrationMetadata(
            architecture_config_sha256="b" * 64,
            tokenizer_revision="test-tokenizer-v1",
            truncation_policy_id="invalid identifier",
            precision="test-cpu-path-v1",
            quantization="none",
            output_transform="test-softmax-v1",
        ),
    ),
)
def test_malformed_backend_metadata_is_sanitized_before_encoding(
    metadata: BackendCalibrationMetadata,
) -> None:
    seed = _NonFixtureBackend()
    request, artifact = _artifact_for_nonfixture(seed)
    backend = _NonFixtureBackend()
    backend._metadata = metadata
    engine = DecisionEngine(
        backend=backend,
        workflows=default_workflows(),
        calibrations={request.questions[0].id: artifact},
        calibration_profiles={request.questions[0].id: PHASE4A_IDENTITY_PROFILE},
    )
    with pytest.raises(SaracuraError) as captured:
        engine.decide(request)
    assert captured.value.payload.code == ErrorCode.BACKEND_UNAVAILABLE
    assert backend.encode_calls == backend.score_calls == 0


def test_metadata_property_failure_is_sanitized_before_encoding() -> None:
    seed = _NonFixtureBackend()
    request, artifact = _artifact_for_nonfixture(seed)
    backend = _ExplodingMetadataBackend()
    engine = DecisionEngine(
        backend=backend,
        workflows=default_workflows(),
        calibrations={request.questions[0].id: artifact},
        calibration_profiles={request.questions[0].id: PHASE4A_IDENTITY_PROFILE},
    )
    with pytest.raises(SaracuraError) as captured:
        engine.decide(request)
    assert captured.value.payload.code == ErrorCode.BACKEND_UNAVAILABLE
    assert backend.encode_calls == backend.score_calls == 0


def test_minilm_state_decoder_rejects_noncanonical_and_framing_attacks() -> None:
    backend = object.__new__(MiniLMRoutingBackend)
    valid = frame_segments(
        "nfc-jcs-length-prefixed-v1",
        "pt-BR",
        "support",
        canonical_json_bytes({"text": "mensagem NFC"}),
    )
    assert backend._decode_state_payload(valid) == "mensagem NFC"

    malformed = (
        valid[:-1],
        valid + frame_segments("unexpected"),
        (MAX_STATE_FRAME_BYTES + 1).to_bytes(8, "big"),
        frame_segments(
            "nfc-jcs-length-prefixed-v1",
            "pt-BR",
            "support",
            b'{"text": "not canonical"}',
        ),
        frame_segments(
            "nfc-jcs-length-prefixed-v1",
            "pt-BR",
            "support",
            b'{"text":"a","text":"b"}',
        ),
        frame_segments(
            "nfc-jcs-length-prefixed-v1",
            "pt-BR",
            "support",
            b'{"text":"na\xcc\x83o"}',
        ),
    )
    for payload in malformed:
        with pytest.raises(SaracuraError) as captured:
            backend._decode_state_payload(payload)
        assert captured.value.payload.code == ErrorCode.REQUEST_INVALID


def test_minilm_conformance_resource_is_closed_without_local_weights() -> None:
    backend = object.__new__(MiniLMRoutingBackend)
    backend._device = "cpu"
    vector = backend._conformance_vector()
    assert all(
        len(tokens[0]) == 14
        for tokens in (vector.input_ids, vector.attention_mask, vector.token_type_ids)
    )
    assert len(vector.logits) == len(MINILM_ROUTING_LABELS)


def test_minilm_special_token_map_matches_the_reviewed_snapshot_contract() -> None:
    backend = object.__new__(MiniLMRoutingBackend)
    valid = {
        "unk_token": "<unk>",
        "sep_token": "</s>",
        "pad_token": "<pad>",
        "cls_token": "<s>",
        "bos_token": "<s>",
        "eos_token": "</s>",
        "mask_token": {
            "content": "<mask>",
            "single_word": False,
            "lstrip": True,
            "rstrip": False,
            "normalized": False,
        },
    }
    assert (
        backend._parse_special_tokens(canonical_json_bytes(cast(JsonValue, valid)))["mask_token"]
        == "<mask>"
    )
    cast(dict[str, JsonValue], valid["mask_token"])["normalized"] = True
    with pytest.raises(ValueError, match="mask token contract"):
        backend._parse_special_tokens(canonical_json_bytes(cast(JsonValue, valid)))


def test_opt_in_local_mps_conformance_uses_only_operator_paths() -> None:
    if os.environ.get("SARACURA_PHASE4A_LIVE_MPS") != "1":
        pytest.skip("set SARACURA_PHASE4A_LIVE_MPS=1 with local operator artifact paths")
    required = (
        "SARACURA_PHASE4A_SNAPSHOT",
        "SARACURA_PHASE4A_TRAINING_MANIFEST",
        "SARACURA_PHASE4A_CHECKPOINT",
    )
    if any(not os.environ.get(name) for name in required):
        pytest.fail("local MPS conformance inputs were not provided")
    backend = MiniLMRoutingBackend(
        encoder_snapshot=Path(os.environ["SARACURA_PHASE4A_SNAPSHOT"]),
        training_manifest=Path(os.environ["SARACURA_PHASE4A_TRAINING_MANIFEST"]),
        checkpoint=Path(os.environ["SARACURA_PHASE4A_CHECKPOINT"]),
        device="mps",
    )
    assert backend.model.id == "saracura-minilm-routing"
    assert backend.capabilities.quality_claims is False


def test_new_minilm_cli_paths_are_socket_free_with_controlled_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def blocked_socket(*_args: object, **_kwargs: object) -> socket.socket:
        raise AssertionError("network call attempted")

    _CliMiniLMBackend.instances.clear()
    monkeypatch.setattr(socket, "socket", blocked_socket)
    monkeypatch.setattr(cli, "MiniLMRoutingBackend", _CliMiniLMBackend)
    request = decision_request(
        (MINILM_ROUTING_QUESTION,),
        workflow_id=MINILM_ROUTING_WORKFLOW_ID,
        workflow_revision=MINILM_ROUTING_WORKFLOW_REVISION,
        domain="support",
    ).model_copy(update={"model": "minilm-routing-v1." + "a" * 64})
    request_path = tmp_path / "request.json"
    request_path.write_bytes(canonical_json_bytes(request.model_dump(mode="json")))
    calibration_path = tmp_path / "identity.json"
    backend_arguments = [
        "--backend",
        "minilm-routing",
        "--encoder-snapshot",
        str(tmp_path / "snapshot"),
        "--training-manifest",
        str(tmp_path / "training-manifest.json"),
        "--checkpoint",
        str(tmp_path / "checkpoint.safetensors"),
        "--device",
        "cpu",
    ]

    assert cli.main(["describe-backend", *backend_arguments]) == 0
    assert (
        cli.main(
            [
                "create-identity-calibration",
                *backend_arguments,
                "--request",
                str(request_path),
                "--output",
                str(calibration_path),
                "--created-at",
                "2026-09-22T00:00:00Z",
            ]
        )
        == 0
    )
    assert (
        cli.main(
            [
                "decide",
                *backend_arguments,
                "--request",
                str(request_path),
                "--calibration",
                str(calibration_path),
                "--timing",
            ]
        )
        == 0
    )
    assert len(_CliMiniLMBackend.instances) == 3
    assert _CliMiniLMBackend.instances[-1].encode_calls == 1
    assert _CliMiniLMBackend.instances[-1].score_calls == 1


def _valid_training_manifest(checkpoint: bytes) -> bytes:
    metrics = {
        "synthetic_only": True,
        "epochs": 1,
        "selected_epoch": 1,
        "epoch_ledger": [
            {
                "epoch": 1,
                "train_loss": 0.5,
                "dev_macro_f1": 1.0,
            }
        ],
        "train": {
            "count": 1,
            "accuracy": 1.0,
            "macro_f1": 1.0,
            "confusion_matrix": [[0, 0, 0, 0, 0] for _ in range(5)],
            "per_label": {
                label: {"support": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
                for label in (
                    "billing",
                    "technical_support",
                    "account_access",
                    "subscription_cancellation",
                    "order_delivery",
                )
            },
        },
        "dev": {
            "count": 1,
            "accuracy": 1.0,
            "macro_f1": 1.0,
            "confusion_matrix": [[0, 0, 0, 0, 0] for _ in range(5)],
            "per_label": {
                label: {"support": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
                for label in (
                    "billing",
                    "technical_support",
                    "account_access",
                    "subscription_cancellation",
                    "order_delivery",
                )
            },
        },
    }
    manifest: dict[str, object] = {
        "schema_version": "synthetic-training-manifest.v1",
        "workflow_revision": "phase3b-synthetic-research-training.v2",
        "synthetic_only": True,
        "packet_manifest_sha256": "1" * 64,
        "training_code_sha256": "2" * 64,
        "protocol_sha256": "3" * 64,
        "encoder": {
            "candidate": "multilingual-minilm-l12",
            "revision": "e8f8c211226b894fcb81acc59f3b34ba3efd5f42",
            "registry_sha256": verified_bytes.package_registry_sha256(),
            "device": "mps",
            "frozen": True,
        },
        "labels": [
            "billing",
            "technical_support",
            "account_access",
            "subscription_cancellation",
            "order_delivery",
        ],
        "seed": 20260921,
        "head": {
            "width": 384,
            "classes": 5,
            "optimizer": "AdamW",
            "lr": 0.01,
            "weight_decay": 0.01,
            "batch_size": 32,
            "maximum_epochs": 80,
            "early_stopping_patience": 10,
            "early_stopping_min_delta": 0.001,
        },
        "files": {
            "embeddings.safetensors": "4" * 64,
            "checkpoint.safetensors": hashlib.sha256(checkpoint).hexdigest(),
        },
        "metrics": metrics,
        "environment": {
            "python": "3.11.0",
            "torch": "2.5.0",
            "safetensors": "0.5.0",
            "platform": "test",
        },
        "authorizations": {"calibration": False, "automation": False, "quality_claims": False},
    }
    manifest["manifest_sha256"] = hashlib.sha256(
        canonical_json_bytes(cast(JsonValue, manifest))
    ).hexdigest()
    return canonical_json_bytes(cast(JsonValue, manifest)) + b"\n"


def _rehashed_manifest_bytes(manifest: dict[str, object]) -> bytes:
    unhashed = dict(manifest)
    unhashed.pop("manifest_sha256", None)
    manifest["manifest_sha256"] = hashlib.sha256(
        canonical_json_bytes(cast(JsonValue, unhashed))
    ).hexdigest()
    return canonical_json_bytes(cast(JsonValue, manifest)) + b"\n"


def _safe_write(path: Path, content: bytes) -> None:
    path.write_bytes(content)
    os.chmod(path, 0o600)


def test_phase3b_v3_manifest_metrics_regression_without_tracking_artifacts() -> None:
    raw_path = os.environ.get("SARACURA_PHASE3B_V3_MANIFEST")
    if raw_path is None:
        pytest.skip("set SARACURA_PHASE3B_V3_MANIFEST to an ignored Phase 3B v3 manifest")
    path = Path(raw_path)
    if not path.is_file():
        pytest.fail("the configured Phase 3B v3 manifest is unavailable")
    raw = path.read_bytes()
    manifest = json.loads(raw)
    assert isinstance(manifest, dict)
    files = manifest["files"]
    assert isinstance(files, dict)
    checkpoint_sha256 = files["checkpoint.safetensors"]
    assert isinstance(checkpoint_sha256, str)
    checked = validate_training_manifest(raw, checkpoint_sha256=checkpoint_sha256)
    metrics = checked["metrics"]
    assert isinstance(metrics, dict)
    assert metrics["epochs"] == len(metrics["epoch_ledger"])
    assert 1 <= metrics["selected_epoch"] <= metrics["epochs"]


def test_training_epoch_metrics_contract_is_closed() -> None:
    checkpoint = b"head"
    base = json.loads(_valid_training_manifest(checkpoint))
    assert isinstance(base, dict)
    metrics = base["metrics"]
    assert isinstance(metrics, dict)

    metrics["epochs"] = 2
    with pytest.raises(VerifiedBytesError):
        validate_training_manifest(
            _rehashed_manifest_bytes(base),
            checkpoint_sha256=hashlib.sha256(checkpoint).hexdigest(),
        )

    base = json.loads(_valid_training_manifest(checkpoint))
    assert isinstance(base, dict)
    metrics = base["metrics"]
    assert isinstance(metrics, dict)
    metrics["selected_epoch"] = 0
    with pytest.raises(VerifiedBytesError):
        validate_training_manifest(
            _rehashed_manifest_bytes(base),
            checkpoint_sha256=hashlib.sha256(checkpoint).hexdigest(),
        )

    base = json.loads(_valid_training_manifest(checkpoint))
    assert isinstance(base, dict)
    metrics = base["metrics"]
    assert isinstance(metrics, dict)
    ledger = metrics["epoch_ledger"]
    assert isinstance(ledger, list)
    assert isinstance(ledger[0], dict)
    ledger[0]["epoch"] = 2
    with pytest.raises(VerifiedBytesError):
        validate_training_manifest(
            _rehashed_manifest_bytes(base),
            checkpoint_sha256=hashlib.sha256(checkpoint).hexdigest(),
        )


def _small_verified_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, Path]:
    names = (
        "config.json",
        "model.safetensors",
        "sentencepiece.bpe.model",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    )
    files = tuple(
        RegistryFile(name=name, size=1, sha256=hashlib.sha256(name[:1].encode()).hexdigest())
        for name in names
    )
    candidate = MiniLMCandidate(
        id="multilingual-minilm-l12",
        repository="test/repository",
        revision="e8f8c211226b894fcb81acc59f3b34ba3efd5f42",
        architecture="BertModel",
        hidden_width=384,
        tokenizer="PreTrainedTokenizerFast",
        files=files,
    )
    monkeypatch.setattr(verified_bytes, "load_minilm_candidate", lambda: candidate)
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir(mode=0o700)
    os.chmod(snapshot, 0o700)
    for item in files:
        _safe_write(snapshot / item.name, item.name[:1].encode())
    marker = {
        "candidate_id": candidate.id,
        "revision": candidate.revision,
        "registry_sha256": verified_bytes.package_registry_sha256(),
        "files": [
            {"path": item.name, "bytes": item.size, "sha256": item.sha256}
            for item in candidate.files
        ],
    }
    _safe_write(
        snapshot / "snapshot.complete.json",
        canonical_json_bytes(cast(JsonValue, marker)),
    )

    artifacts = tmp_path / "artifacts"
    artifacts.mkdir(mode=0o700)
    os.chmod(artifacts, 0o700)
    checkpoint = artifacts / "checkpoint.safetensors"
    _safe_write(checkpoint, b"head")
    manifest = artifacts / "training-manifest.json"
    _safe_write(manifest, _valid_training_manifest(checkpoint.read_bytes()))
    return snapshot, manifest, checkpoint


def test_verified_byte_loader_rejects_snapshot_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot, manifest, checkpoint = _small_verified_inputs(tmp_path, monkeypatch)
    loaded = load_verified_minilm_bytes(
        encoder_snapshot=snapshot,
        training_manifest=manifest,
        checkpoint=checkpoint,
    )
    assert loaded.snapshot["config.json"] == b"c"
    assert isinstance(loaded.snapshot, MappingProxyType)

    _safe_write(snapshot / "extra.json", b"x")
    with pytest.raises(VerifiedBytesError):
        load_verified_minilm_bytes(
            encoder_snapshot=snapshot,
            training_manifest=manifest,
            checkpoint=checkpoint,
        )


@pytest.mark.parametrize("artifact", ("snapshot", "training-manifest", "checkpoint"))
def test_verified_byte_loader_rejects_symlinked_ancestor_for_every_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact: str,
) -> None:
    snapshot, manifest, checkpoint = _small_verified_inputs(tmp_path, monkeypatch)
    alias = tmp_path / "symlinked-ancestor"
    alias.symlink_to(tmp_path, target_is_directory=True)
    unsafe_snapshot = alias / snapshot.name if artifact == "snapshot" else snapshot
    unsafe_manifest = manifest
    unsafe_checkpoint = checkpoint
    if artifact == "training-manifest":
        unsafe_manifest = alias / manifest.parent.name / manifest.name
    elif artifact == "checkpoint":
        unsafe_checkpoint = alias / checkpoint.parent.name / checkpoint.name

    with pytest.raises(VerifiedBytesError):
        load_verified_minilm_bytes(
            encoder_snapshot=unsafe_snapshot,
            training_manifest=unsafe_manifest,
            checkpoint=unsafe_checkpoint,
        )


def test_verified_byte_loader_detects_same_process_mutation_during_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot, manifest, checkpoint = _small_verified_inputs(tmp_path, monkeypatch)
    target_inode = os.stat(manifest, follow_symlinks=False).st_ino
    original_read = os.read
    mutated = False

    def mutate_after_read(descriptor: int, count: int) -> bytes:
        nonlocal mutated
        data = original_read(descriptor, count)
        if not mutated and os.fstat(descriptor).st_ino == target_inode:
            mutated = True
            _safe_write(manifest, b"{}")
        return data

    monkeypatch.setattr(os, "read", mutate_after_read)
    with pytest.raises(VerifiedBytesError, match="changed"):
        load_verified_minilm_bytes(
            encoder_snapshot=snapshot,
            training_manifest=manifest,
            checkpoint=checkpoint,
        )
    assert mutated


@pytest.mark.parametrize("attack", ("symlink", "mode", "checkpoint-ledger"))
def test_verified_byte_loader_rejects_unsafe_external_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    attack: str,
) -> None:
    snapshot, manifest, checkpoint = _small_verified_inputs(tmp_path, monkeypatch)
    if attack == "symlink":
        target = snapshot / "config.json"
        target.unlink()
        target.symlink_to(snapshot / "tokenizer.json")
    elif attack == "mode":
        os.chmod(snapshot / "config.json", 0o644)
    else:
        _safe_write(checkpoint, b"changed")

    with pytest.raises(VerifiedBytesError):
        load_verified_minilm_bytes(
            encoder_snapshot=snapshot,
            training_manifest=manifest,
            checkpoint=checkpoint,
        )
