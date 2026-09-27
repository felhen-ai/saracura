"""Explicit, verified local runtime for the sealed Saracura universal ranker.

No optional ML package is imported at module import time. The descriptor
boundary snapshots the exact operator artifact before the runtime can load ML.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import threading
import unicodedata
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, cast

from saracura.backends.base import BackendCapabilities, ScoredChoice
from saracura.contracts.errors import ErrorCode, SaracuraError
from saracura.contracts.models import ChoiceQuestion, DecisionRequest, ModelReference
from saracura.serialization import canonical_json_bytes
from saracura.universal.checkpoint import (
    BASE_ENCODER_ID,
    BASE_ENCODER_REVISION,
    CHECKPOINT_ARCHITECTURE_REVISION,
    verify_checkpoint_tensors,
)
from saracura.universal.rendering import (
    MAX_CONTEXT_TOKENS,
    MAX_CRITERION_TOKENS,
    RENDERER_REVISION,
    RenderedTask,
    render_choice,
)
from saracura.universal.runtime_ranker import TorchSaracuraRanker
from saracura.universal.tasks import (
    SARACURA_UNIVERSAL_WORKFLOW_ID,
    SARACURA_UNIVERSAL_WORKFLOW_REVISION,
    UNIVERSAL_DOMAINS,
)
from saracura.verified_bytes import (
    MAX_MANIFEST_BYTES,
    VerifiedBytesError,
    read_verified_directory,
    read_verified_minilm_snapshot_with_marker,
)

_SARACURA_MODEL_ID = "saracura/universal-ranker"
_CAPSULE_FILENAMES = (
    "accepted-packet-receipt.json",
    "architecture.json",
    "capsule-descriptor.json",
    "checkpoint.safetensors",
    "conformance-manifest.json",
    "conformance-vectors.safetensors",
    "dependency-versions.json",
    "dev-selection.json",
    "embedding-descriptor.json",
    "epoch-ledger.json",
    "holdout-release.json",
    "holdout-report.json",
    "pre-holdout-gate.json",
    "training-manifest.json",
)
CAPSULE_FILE_SET = frozenset(_CAPSULE_FILENAMES)
_JSON_CAPSULE_FILES = CAPSULE_FILE_SET - {
    "checkpoint.safetensors",
    "conformance-vectors.safetensors",
}
_CAPSULE_MAXIMUMS = {
    name: MAX_MANIFEST_BYTES if name in _JSON_CAPSULE_FILES else 16 * 1024 * 1024
    for name in _CAPSULE_FILENAMES
}
_MAX_QUESTIONS = 10
_MAX_CRITERIA = 8
_MAX_INSTRUCTION_CODEPOINTS = 120
_MAX_INSTRUCTION_BYTES = 480
_MAX_CRITERION_CODEPOINTS = 120
_MAX_CRITERION_BYTES = 480
_MAX_STATE_FIELDS = 64
_MAX_STATE_CODEPOINTS = 200
_MAX_STATE_BYTES = 800
_CPU_THREAD_MIN = 1
_CPU_THREAD_MAX = 8
_VALID_LOCALES = frozenset({"pt-BR", "en"})
_SPECIAL_TOKENS = {
    "unk_token": "<unk>",
    "sep_token": "</s>",
    "pad_token": "<pad>",
    "cls_token": "<s>",
    "bos_token": "<s>",
    "eos_token": "</s>",
    "mask_token": "<mask>",
}
_CONFORMANCE_CELLS = tuple(
    f"{locale}:{count}" for locale in ("pt-BR", "en") for count in range(2, 9)
)
_CONFORMANCE_SHAPES = {
    "context_embeddings": [14, 384],
    "criterion_embeddings": [14, 8, 384],
    "option_masks": [14, 8],
    "option_counts": [14],
    "expected_logits": [14, 8],
}
_APPROVED_CANDIDATE = {
    "id": "saracura/universal-ranker",
    "model_revision": (
        "phase4e-saracura-ranker.v1."
        "f1d72c34cc535ddefbf7e24cf45d1be6e0aee40ba881228b4edd092d78640d5e"
    ),
    "checkpoint_sha256": ("6914195d5526fb7b629eedaac7f33ae04f77eccd8448c75fed9104ccc57b728c"),
    "training_manifest_internal_sha256": (
        "f1d72c34cc535ddefbf7e24cf45d1be6e0aee40ba881228b4edd092d78640d5e"
    ),
    "training_manifest_file_sha256": (
        "1daee9317425007a408646fdf07747fe5b5b59e6ef20104788bc577c659e1afb"
    ),
    "capsule_descriptor_file_sha256": (
        "44693fc40beb64c3c556b8f411c8f4f5e04088388ab24660b47340a09af03801"
    ),
    "capsule_descriptor_internal_sha256": (
        "036e6c6dc85d8b7d619de6e7f7fc943975a088ae5be1c27e5c786b57be9b7a64"
    ),
    "conformance_manifest_file_sha256": (
        "e2406ce5119f8261fd15224ceaf831c35e34004ffd0920b6d4fd0aca7196640a"
    ),
    "conformance_vectors_sha256": (
        "7682148b18edc3058ce2038300234c243c973dc06bf2f2ba0b7c7913fa7a0bde"
    ),
    "architecture": "saracura-universal-ranker.v0",
    "foundation_encoder": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "encoder_revision": "e8f8c211226b894fcb81acc59f3b34ba3efd5f42",
    "encoder_snapshot_complete_sha256": (
        "62827e84e66f4546f996b7ac92f01c0bf96e322e309618d69101f1925fe5b823"
    ),
    "training_source_commit": "cbff1989c2c3f802f4b4186c65f339466310296c",
    "disposition": "synthetic_only_research",
}
_MODEL_RESIDENT = False
_MODEL_STATE_LOCK = threading.Lock()


class SaracuraBackendError(ValueError):
    """Private verifier detail redacted by the public runtime boundary."""


@dataclass(frozen=True, slots=True)
class SaracuraCandidate:
    id: str
    model_revision: str
    checkpoint_sha256: str
    training_manifest_internal_sha256: str
    training_manifest_file_sha256: str
    capsule_descriptor_file_sha256: str
    capsule_descriptor_internal_sha256: str
    conformance_manifest_file_sha256: str
    conformance_vectors_sha256: str
    architecture: str
    foundation_encoder: str
    encoder_revision: str
    encoder_snapshot_complete_sha256: str
    training_source_commit: str
    disposition: str
    capsule_files: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class VerifiedSaracuraCapsule:
    """Descriptor-bound in-memory bytes with no retained caller path."""

    candidate: SaracuraCandidate
    values: Mapping[str, bytes]
    manifest: Mapping[str, Any]
    conformance: Mapping[str, Any]


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, child in pairs:
        if key in value:
            raise SaracuraBackendError("duplicate JSON object key")
        value[key] = child
    return value


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


def _is_commit(value: object) -> bool:
    return (
        isinstance(value, str) and len(value) == 40 and all(c in "0123456789abcdef" for c in value)
    )


def _package_saracura_registry_bytes() -> bytes:
    return resources.files("saracura").joinpath("saracura-candidates.v1.json").read_bytes()


def load_saracura_candidate(raw: bytes | None = None) -> SaracuraCandidate:
    """Load the one approved candidate from a closed metadata-only registry."""

    try:
        registry = json.loads(
            _package_saracura_registry_bytes() if raw is None else raw,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (UnicodeDecodeError, ValueError) as error:
        raise SaracuraBackendError("candidate registry JSON is invalid") from error
    if (
        not isinstance(registry, dict)
        or set(registry) != {"schema_version", "candidates"}
        or registry.get("schema_version") != "saracura-candidates.v1"
        or not isinstance(registry.get("candidates"), list)
        or len(registry["candidates"]) != 1
        or not isinstance(registry["candidates"][0], dict)
    ):
        raise SaracuraBackendError("candidate registry shape is invalid")
    item = registry["candidates"][0]
    assert isinstance(item, dict)
    fields = {
        "id",
        "model_revision",
        "checkpoint_sha256",
        "training_manifest_internal_sha256",
        "training_manifest_file_sha256",
        "capsule_descriptor_file_sha256",
        "capsule_descriptor_internal_sha256",
        "conformance_manifest_file_sha256",
        "conformance_vectors_sha256",
        "architecture",
        "foundation_encoder",
        "encoder_revision",
        "encoder_snapshot_complete_sha256",
        "training_source_commit",
        "disposition",
        "capsule_files",
    }
    hash_fields = {
        "checkpoint_sha256",
        "training_manifest_internal_sha256",
        "training_manifest_file_sha256",
        "capsule_descriptor_file_sha256",
        "capsule_descriptor_internal_sha256",
        "conformance_manifest_file_sha256",
        "conformance_vectors_sha256",
        "encoder_snapshot_complete_sha256",
    }
    if set(item) != fields or any(not _is_sha256(item.get(name)) for name in hash_fields):
        raise SaracuraBackendError("candidate registry fields are invalid")
    if (
        not _is_commit(item.get("training_source_commit"))
        or item.get("id") != _SARACURA_MODEL_ID
        or item.get("model_revision")
        != f"{SARACURA_UNIVERSAL_WORKFLOW_REVISION}.{item['training_manifest_internal_sha256']}"
        or item.get("architecture") != CHECKPOINT_ARCHITECTURE_REVISION
        or item.get("foundation_encoder") != BASE_ENCODER_ID
        or item.get("encoder_revision") != BASE_ENCODER_REVISION
        or item.get("disposition") != "synthetic_only_research"
        or not isinstance(item.get("capsule_files"), list)
        or tuple(item["capsule_files"]) != _CAPSULE_FILENAMES
    ):
        raise SaracuraBackendError("candidate registry identity is invalid")
    if any(item[name] != value for name, value in _APPROVED_CANDIDATE.items()):
        raise SaracuraBackendError("candidate registry is not the approved candidate")
    return SaracuraCandidate(
        id=cast(str, item["id"]),
        model_revision=cast(str, item["model_revision"]),
        checkpoint_sha256=cast(str, item["checkpoint_sha256"]),
        training_manifest_internal_sha256=cast(str, item["training_manifest_internal_sha256"]),
        training_manifest_file_sha256=cast(str, item["training_manifest_file_sha256"]),
        capsule_descriptor_file_sha256=cast(str, item["capsule_descriptor_file_sha256"]),
        capsule_descriptor_internal_sha256=cast(str, item["capsule_descriptor_internal_sha256"]),
        conformance_manifest_file_sha256=cast(str, item["conformance_manifest_file_sha256"]),
        conformance_vectors_sha256=cast(str, item["conformance_vectors_sha256"]),
        architecture=cast(str, item["architecture"]),
        foundation_encoder=cast(str, item["foundation_encoder"]),
        encoder_revision=cast(str, item["encoder_revision"]),
        encoder_snapshot_complete_sha256=cast(str, item["encoder_snapshot_complete_sha256"]),
        training_source_commit=cast(str, item["training_source_commit"]),
        disposition=cast(str, item["disposition"]),
        capsule_files=tuple(cast(list[str], item["capsule_files"])),
    )


def _finite_json(value: object) -> bool:
    if value is None or isinstance(value, (str, bool, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_finite_json(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _finite_json(item) for key, item in value.items())
    return False


def _nfc_json(value: object) -> bool:
    if isinstance(value, str):
        return value == unicodedata.normalize("NFC", value)
    if isinstance(value, list):
        return all(_nfc_json(item) for item in value)
    if isinstance(value, dict):
        return all(
            isinstance(key, str) and key == unicodedata.normalize("NFC", key) and _nfc_json(item)
            for key, item in value.items()
        )
    return True


def _freeze_public_json(value: Any) -> Any:
    """Make parsed public metadata transitively immutable after verification."""

    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_public_json(child) for key, child in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_public_json(child) for child in value)
    return value


def _canonical_json_object(raw: bytes) -> dict[str, Any]:
    if not raw or len(raw) > MAX_MANIFEST_BYTES or not raw.endswith(b"\n") or b"\n" in raw[:-1]:
        raise SaracuraBackendError("capsule JSON framing is invalid")
    try:
        value = json.loads(raw[:-1], object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeDecodeError, ValueError) as error:
        raise SaracuraBackendError("capsule JSON is invalid") from error
    if (
        not isinstance(value, dict)
        or not _finite_json(value)
        or not _nfc_json(value)
        or canonical_json_bytes(cast(Any, value)) != raw[:-1]
    ):
        raise SaracuraBackendError("capsule JSON is not canonical")
    return cast(dict[str, Any], value)


def _verify_descriptor(values: Mapping[str, bytes], candidate: SaracuraCandidate) -> None:
    raw = values["capsule-descriptor.json"]
    descriptor = _canonical_json_object(raw)
    expected = {"schema_version", "capsule_kind", "files", "descriptor_sha256"}
    if (
        set(descriptor) != expected
        or descriptor.get("schema_version") != "phase4e-training-capsule.v1"
        or descriptor.get("capsule_kind") != "training"
        or not _is_sha256(descriptor.get("descriptor_sha256"))
        or hashlib.sha256(raw).hexdigest() != candidate.capsule_descriptor_file_sha256
        or descriptor["descriptor_sha256"] != candidate.capsule_descriptor_internal_sha256
    ):
        raise SaracuraBackendError("capsule descriptor identity is invalid")
    unsigned = {key: value for key, value in descriptor.items() if key != "descriptor_sha256"}
    if (
        hashlib.sha256(canonical_json_bytes(cast(Any, unsigned))).hexdigest()
        != descriptor["descriptor_sha256"]
    ):
        raise SaracuraBackendError("capsule descriptor digest is invalid")
    ledger = descriptor.get("files")
    expected_names = [name for name in _CAPSULE_FILENAMES if name != "capsule-descriptor.json"]
    if (
        not isinstance(ledger, list)
        or len(ledger) != len(expected_names)
        or [entry.get("path") for entry in ledger if isinstance(entry, dict)] != expected_names
    ):
        raise SaracuraBackendError("capsule descriptor ledger is invalid")
    for entry in ledger:
        if (
            not isinstance(entry, dict)
            or set(entry) != {"path", "bytes", "sha256"}
            or entry.get("path") not in CAPSULE_FILE_SET
            or type(entry.get("bytes")) is not int
            or entry["bytes"] <= 0
            or not _is_sha256(entry.get("sha256"))
        ):
            raise SaracuraBackendError("capsule descriptor ledger is invalid")
        stored = values[cast(str, entry["path"])]
        if entry["bytes"] != len(stored) or entry["sha256"] != hashlib.sha256(stored).hexdigest():
            raise SaracuraBackendError("capsule descriptor ledger does not match bytes")


def _verify_manifest(
    values: Mapping[str, bytes], candidate: SaracuraCandidate
) -> Mapping[str, Any]:
    raw = values["training-manifest.json"]
    manifest = _canonical_json_object(raw)
    required = {
        "architecture_revision",
        "base_encoder",
        "baselines",
        "checkpoint_sha256",
        "code_sha256",
        "code_sources",
        "dev_gates",
        "embedding_descriptor_sha256",
        "grant_digest",
        "manifest_sha256",
        "outcome",
        "packet_receipt_sha256",
        "pre_holdout_gate_descriptor_sha256",
        "rendering_revision",
        "schema_version",
        "sealed_report_sha256",
        "selected",
        "source_commit",
        "synthetic_only",
        "training",
    }
    hash_names = (
        "checkpoint_sha256",
        "code_sha256",
        "embedding_descriptor_sha256",
        "grant_digest",
        "manifest_sha256",
        "packet_receipt_sha256",
        "pre_holdout_gate_descriptor_sha256",
        "sealed_report_sha256",
    )
    base_encoder = manifest.get("base_encoder")
    if (
        set(manifest) != required
        or any(not _is_sha256(manifest.get(name)) for name in hash_names)
        or manifest.get("schema_version") != "phase4e-training-manifest.v2"
        or manifest.get("synthetic_only") is not True
        or manifest.get("outcome") != "passed"
        or manifest.get("checkpoint_sha256") != candidate.checkpoint_sha256
        or manifest.get("architecture_revision") != candidate.architecture
        or manifest.get("rendering_revision") != RENDERER_REVISION
        or manifest.get("source_commit") != candidate.training_source_commit
        or base_encoder
        != {
            "id": candidate.foundation_encoder,
            "revision": candidate.encoder_revision,
            "snapshot_complete_sha256": candidate.encoder_snapshot_complete_sha256,
            "frozen": True,
        }
        or hashlib.sha256(raw).hexdigest() != candidate.training_manifest_file_sha256
    ):
        raise SaracuraBackendError("training manifest identity is invalid")
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if (
        manifest["manifest_sha256"] != candidate.training_manifest_internal_sha256
        or hashlib.sha256(canonical_json_bytes(cast(Any, unsigned))).hexdigest()
        != manifest["manifest_sha256"]
        or candidate.model_revision
        != f"{SARACURA_UNIVERSAL_WORKFLOW_REVISION}.{manifest['manifest_sha256']}"
    ):
        raise SaracuraBackendError("training manifest digest is invalid")
    if (
        not isinstance(manifest["code_sources"], dict)
        or not manifest["code_sources"]
        or not isinstance(manifest["baselines"], dict)
        or not isinstance(manifest["dev_gates"], dict)
        or not all(value is True for value in manifest["dev_gates"].values())
        or not isinstance(manifest["selected"], dict)
        or not isinstance(manifest["training"], dict)
    ):
        raise SaracuraBackendError("training manifest public bindings are invalid")
    return cast(Mapping[str, Any], _freeze_public_json(manifest))


def _verify_conformance_manifest(
    values: Mapping[str, bytes], candidate: SaracuraCandidate
) -> Mapping[str, Any]:
    raw = values["conformance-manifest.json"]
    manifest = _canonical_json_object(raw)
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if (
        set(manifest)
        != {"schema_version", "cells", "tensor_sha256", "tensor_shapes", "manifest_sha256"}
        or manifest.get("schema_version") != "phase4e-conformance-manifest.v1"
        or manifest.get("cells") != list(_CONFORMANCE_CELLS)
        or manifest.get("tensor_shapes") != _CONFORMANCE_SHAPES
        or not _is_sha256(manifest.get("tensor_sha256"))
        or not _is_sha256(manifest.get("manifest_sha256"))
        or hashlib.sha256(raw).hexdigest() != candidate.conformance_manifest_file_sha256
        or hashlib.sha256(values["conformance-vectors.safetensors"]).hexdigest()
        != candidate.conformance_vectors_sha256
        or manifest.get("tensor_sha256") != candidate.conformance_vectors_sha256
        or hashlib.sha256(canonical_json_bytes(cast(Any, unsigned))).hexdigest()
        != manifest["manifest_sha256"]
    ):
        raise SaracuraBackendError("conformance manifest identity is invalid")
    return cast(Mapping[str, Any], _freeze_public_json(manifest))


def verify_training_capsule(
    training_capsule: Path, candidate: SaracuraCandidate | None = None
) -> VerifiedSaracuraCapsule:
    """Verify exact private capsule bytes without ML imports or path retention."""

    resolved = candidate or load_saracura_candidate()
    try:
        values = read_verified_directory(training_capsule, expected_files=_CAPSULE_MAXIMUMS)
    except (OSError, VerifiedBytesError) as error:
        raise SaracuraBackendError("training capsule cannot be read") from error
    return _verify_training_capsule_bytes(values, resolved)


def _verify_training_capsule_bytes(
    values: Mapping[str, bytes], candidate: SaracuraCandidate
) -> VerifiedSaracuraCapsule:
    """Revalidate and freeze one already-read capsule snapshot.

    This is also the trust boundary for the CLI handoff.  The caller-provided
    wrapper and parsed metadata are deliberately ignored: only a fresh copy of
    the raw bytes is checked against the package-owned candidate.
    """

    try:
        snapshot = dict(values)
    except Exception as error:
        raise SaracuraBackendError("capsule byte snapshot is invalid") from error
    if set(snapshot) != CAPSULE_FILE_SET or any(
        type(value) is not bytes or not value or len(value) > _CAPSULE_MAXIMUMS[name]
        for name, value in snapshot.items()
    ):
        raise SaracuraBackendError("capsule byte snapshot is invalid")
    frozen = MappingProxyType(snapshot)
    _verify_descriptor(frozen, candidate)
    manifest = _verify_manifest(frozen, candidate)
    conformance = _verify_conformance_manifest(frozen, candidate)
    return VerifiedSaracuraCapsule(
        candidate=candidate,
        values=frozen,
        manifest=manifest,
        conformance=conformance,
    )


def validate_saracura_request_structure(request: DecisionRequest) -> None:
    """Apply artifact-free Saracura request gates before model identity is known."""

    if (
        request.workflow.id != SARACURA_UNIVERSAL_WORKFLOW_ID
        or request.workflow.revision != SARACURA_UNIVERSAL_WORKFLOW_REVISION
    ):
        raise SaracuraError(
            ErrorCode.WORKFLOW_UNSUPPORTED, "Saracura workflow is not supported.", "/workflow"
        )
    if request.locale not in _VALID_LOCALES:
        raise SaracuraError(
            ErrorCode.LOCALE_UNVERIFIED, "Saracura backend accepts only pt-BR or en.", "/locale"
        )
    if request.domain not in UNIVERSAL_DOMAINS:
        raise SaracuraError(
            ErrorCode.DOMAIN_UNVERIFIED, "Saracura backend does not support this domain.", "/domain"
        )
    if len(request.questions) > _MAX_QUESTIONS:
        raise SaracuraError(
            ErrorCode.CARDINALITY_EXCEEDED, "Saracura supports at most ten questions.", "/questions"
        )
    state = cast(Any, request.state)
    state_bytes = canonical_json_bytes(state)
    if (
        len(state) > _MAX_STATE_FIELDS
        or len(state_bytes.decode("utf-8")) > _MAX_STATE_CODEPOINTS
        or len(state_bytes) > _MAX_STATE_BYTES
    ):
        raise SaracuraError(
            ErrorCode.CAPACITY_EXCEEDED, "State exceeds Saracura capacity.", "/state"
        )
    for index, question in enumerate(request.questions):
        if len(question.criteria) > _MAX_CRITERIA:
            raise SaracuraError(
                ErrorCode.CARDINALITY_EXCEEDED,
                "Saracura supports 2 through 8 criteria.",
                f"/questions/{index}/criteria",
            )
        if (
            len(question.instruction) > _MAX_INSTRUCTION_CODEPOINTS
            or len(question.instruction.encode("utf-8")) > _MAX_INSTRUCTION_BYTES
        ):
            raise SaracuraError(
                ErrorCode.CAPACITY_EXCEEDED,
                "Instruction exceeds Saracura capacity.",
                f"/questions/{index}/instruction",
            )
        for criterion in question.criteria:
            if (
                len(criterion.description) > _MAX_CRITERION_CODEPOINTS
                or len(criterion.description.encode("utf-8")) > _MAX_CRITERION_BYTES
            ):
                raise SaracuraError(
                    ErrorCode.CAPACITY_EXCEEDED,
                    "Criterion exceeds Saracura capacity.",
                    f"/questions/{index}/criteria/{criterion.id}",
                )
    if not _nfc_json(request.model_dump(mode="json")):
        raise SaracuraError(ErrorCode.REQUEST_INVALID, "Universal input must already be NFC.", "/")


def validate_saracura_model_identity(
    request: DecisionRequest, candidate: SaracuraCandidate
) -> None:
    """Bind one structurally valid request to the verified capsule candidate."""

    if request.model != candidate.model_revision:
        raise SaracuraError(
            ErrorCode.MODEL_NOT_FOUND, "The immutable model revision is not available.", "/model"
        )


def validate_saracura_request(
    request: DecisionRequest, candidate: SaracuraCandidate | None = None
) -> None:
    """Apply structural and immutable-model gates for direct backend callers."""

    validate_saracura_request_structure(request)
    validate_saracura_model_identity(request, candidate or load_saracura_candidate())


class SaracuraUniversalBackend:
    """Question-conditioned, research-only backend bound to one sealed capsule.

    ``cpu_threads`` changes torch's process-global CPU setting while this
    backend is prepared and restores the prior setting on close when possible.
    """

    def __init__(
        self,
        *,
        encoder_snapshot: Path,
        training_capsule: Path,
        device: Literal["cpu", "mps"],
        cpu_threads: int = 1,
    ) -> None:
        candidate = self._load_approved_candidate(device, cpu_threads)
        self._initialize(
            candidate=candidate,
            encoder_snapshot=encoder_snapshot,
            training_capsule=training_capsule,
            device=device,
            cpu_threads=cpu_threads,
            verified_capsule=None,
        )

    @classmethod
    def _from_verified_capsule(
        cls,
        *,
        encoder_snapshot: Path,
        verified_capsule: VerifiedSaracuraCapsule,
        device: Literal["cpu", "mps"],
        cpu_threads: int = 1,
    ) -> SaracuraUniversalBackend:
        """Construct from a CLI snapshot only after package-bound revalidation."""

        candidate = cls._load_approved_candidate(device, cpu_threads)
        try:
            capsule = _verify_training_capsule_bytes(verified_capsule.values, candidate)
        except Exception as error:
            raise SaracuraError(
                ErrorCode.BACKEND_UNAVAILABLE, "Saracura backend is unavailable.", "/model"
            ) from error
        backend = cls.__new__(cls)
        backend._initialize(
            candidate=candidate,
            encoder_snapshot=encoder_snapshot,
            training_capsule=None,
            device=device,
            cpu_threads=cpu_threads,
            verified_capsule=capsule,
        )
        return backend

    @staticmethod
    def _load_approved_candidate(
        device: Literal["cpu", "mps"], cpu_threads: int
    ) -> SaracuraCandidate:
        if (
            device not in {"cpu", "mps"}
            or type(cpu_threads) is not int
            or not 1 <= cpu_threads <= 8
        ):
            raise SaracuraError(
                ErrorCode.BACKEND_UNAVAILABLE, "Saracura backend is unavailable.", "/model"
            )
        try:
            return load_saracura_candidate()
        except SaracuraBackendError as error:
            raise SaracuraError(
                ErrorCode.BACKEND_UNAVAILABLE, "Saracura backend is unavailable.", "/model"
            ) from error

    def _initialize(
        self,
        *,
        candidate: SaracuraCandidate,
        encoder_snapshot: Path,
        training_capsule: Path | None,
        device: Literal["cpu", "mps"],
        cpu_threads: int,
        verified_capsule: VerifiedSaracuraCapsule | None,
    ) -> None:
        self._candidate = candidate
        self._encoder_snapshot: Path | None = encoder_snapshot
        self._training_capsule: Path | None = training_capsule
        self._verified_capsule = verified_capsule
        self._device = device
        self._cpu_threads = cpu_threads
        self._load_lock = threading.Lock()
        self._prepared = False
        self._closed = False
        self._owns_resident_slot = False
        self._prior_threads: int | None = None
        self._torch: Any = None
        self._encoder: Any = None
        self._tokenizer: Any = None
        self._ranker: TorchSaracuraRanker | None = None
        self._snapshot: Mapping[str, bytes] | None = None
        self._capabilities = BackendCapabilities(
            execution_tier="universal",
            decision_types=frozenset({"choice"}),
            max_questions=_MAX_QUESTIONS,
            max_criteria=_MAX_CRITERIA,
            execution_boundary="explicit-verified-local-saracura-universal",
            cold_warm_semantics="one-process-resident-model",
            quality_claims=False,
            dynamic_workflows=frozenset(
                {(SARACURA_UNIVERSAL_WORKFLOW_ID, SARACURA_UNIVERSAL_WORKFLOW_REVISION)}
            ),
        )
        self._model = ModelReference(
            id=self._candidate.id,
            revision=self._candidate.model_revision,
            checkpoint_sha256=self._candidate.checkpoint_sha256,
        )

    @property
    def capabilities(self) -> BackendCapabilities:
        return self._capabilities

    @property
    def model(self) -> ModelReference:
        return self._model

    def prepare(self) -> None:
        with self._load_lock:
            if self._closed:
                raise SaracuraError(
                    ErrorCode.BACKEND_UNAVAILABLE, "Saracura backend is unavailable.", "/model"
                )
            if self._prepared:
                return
            try:
                if self._verified_capsule is None:
                    training_capsule = self._training_capsule
                    if training_capsule is None:
                        raise SaracuraBackendError("training capsule is unavailable")
                    self._verified_capsule = verify_training_capsule(
                        training_capsule, self._candidate
                    )
                # Later decoder construction receives only the descriptor-bound
                # in-memory bytes in ``_verified_capsule``.
                self._training_capsule = None
                self._verify_snapshot()
                self._load_ml_components()
                self._load_ranker_and_conformance()
                self._prepared = True
            except SaracuraError:
                self._release_loaded_state()
                raise
            except Exception as error:
                self._release_loaded_state()
                raise SaracuraError(
                    ErrorCode.BACKEND_UNAVAILABLE, "Saracura backend is unavailable.", "/model"
                ) from error

    def close(self) -> None:
        with self._load_lock:
            if self._closed:
                return
            self._closed = True
            self._release_loaded_state()
            self._verified_capsule = None
            self._snapshot = None
            self._training_capsule = None
            self._encoder_snapshot = None

    def _release_loaded_state(self) -> None:
        global _MODEL_RESIDENT
        torch, prior = self._torch, self._prior_threads
        self._encoder = self._tokenizer = self._ranker = self._torch = None
        self._prepared = False
        self._prior_threads = None
        if self._owns_resident_slot:
            self._owns_resident_slot = False
            with _MODEL_STATE_LOCK:
                _MODEL_RESIDENT = False
        if torch is not None and prior is not None:
            with suppress(Exception):
                torch.set_num_threads(prior)

    def validate_request(self, request: DecisionRequest) -> None:
        validate_saracura_request(request, self._candidate)

    def score_universal_choice(
        self, request: DecisionRequest, question: ChoiceQuestion, state_payload: bytes
    ) -> ScoredChoice:
        del state_payload
        self.validate_request(request)
        self.prepare()
        with self._load_lock:
            if (
                self._ranker is None
                or self._torch is None
                or self._encoder is None
                or self._tokenizer is None
            ):
                raise SaracuraError(
                    ErrorCode.BACKEND_UNAVAILABLE, "Saracura backend is unavailable.", "/model"
                )
            rendered = render_choice(
                locale=request.locale,
                domain=request.domain,
                instruction=question.instruction,
                state=request.state,
                criteria=question.criteria,
            )
            tokens = self._tokenize_joint(rendered)
            attention = tokens["attention_mask"]
            try:
                with self._torch.inference_mode():
                    output = self._encoder(
                        input_ids=tokens["input_ids"], attention_mask=attention, return_dict=True
                    )
                    embeddings = self._mean_pool(output.last_hidden_state, attention)
                    logits = self._ranker.logits(embeddings[:1], embeddings[1:])
                values = logits.detach().to(device="cpu", dtype=self._torch.float32).tolist()
                if tuple(embeddings.shape) != (len(question.criteria) + 1, 384) or not all(
                    math.isfinite(value) for value in values
                ):
                    raise ValueError("invalid encoder or ranker output")
            except SaracuraError:
                raise
            except Exception as error:
                raise SaracuraError(
                    ErrorCode.BACKEND_UNAVAILABLE, "Saracura backend is unavailable.", "/model"
                ) from error
            return ScoredChoice(
                question_id=question.id,
                raw_scores={
                    criterion.id: float(value)
                    for criterion, value in zip(question.criteria, values, strict=True)
                },
                input_tokens=int(attention.sum().item()),
            )

    def _verify_snapshot(self) -> None:
        encoder_snapshot = self._encoder_snapshot
        if encoder_snapshot is None:
            raise SaracuraBackendError("encoder snapshot is unavailable")
        try:
            candidate, snapshot, marker = read_verified_minilm_snapshot_with_marker(
                encoder_snapshot
            )
        except (OSError, VerifiedBytesError) as error:
            raise SaracuraBackendError("encoder snapshot cannot be read") from error
        if (
            candidate.repository != self._candidate.foundation_encoder
            or candidate.revision != self._candidate.encoder_revision
            or hashlib.sha256(marker).hexdigest()
            != self._candidate.encoder_snapshot_complete_sha256
        ):
            raise SaracuraBackendError("encoder snapshot identity is invalid")
        self._snapshot = snapshot
        # The model loader below gets this immutable mapping and cannot reopen
        # an operator-selected snapshot directory.
        self._encoder_snapshot = None

    @staticmethod
    def _json_bytes(raw: bytes) -> dict[str, Any]:
        try:
            value = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
        except (UnicodeDecodeError, ValueError) as error:
            raise SaracuraBackendError("verified MiniLM JSON is invalid") from error
        if not isinstance(value, dict):
            raise SaracuraBackendError("verified MiniLM JSON is invalid")
        return cast(dict[str, Any], value)

    @classmethod
    def _special_tokens(cls, raw: bytes) -> dict[str, str]:
        value = cls._json_bytes(raw)
        if set(value) != set(_SPECIAL_TOKENS):
            raise SaracuraBackendError("special token map is invalid")
        normalized: dict[str, str] = {}
        for name, expected in _SPECIAL_TOKENS.items():
            actual = value[name]
            if name == "mask_token":
                if actual != {
                    "content": "<mask>",
                    "single_word": False,
                    "lstrip": True,
                    "rstrip": False,
                    "normalized": False,
                }:
                    raise SaracuraBackendError("mask token is invalid")
                normalized[name] = "<mask>"
            elif isinstance(actual, str):
                normalized[name] = actual
            else:
                raise SaracuraBackendError("special token map is invalid")
            if normalized[name] != expected:
                raise SaracuraBackendError("special token map is incompatible")
        return normalized

    def _load_ml_components(self) -> None:
        global _MODEL_RESIDENT
        with _MODEL_STATE_LOCK:
            if _MODEL_RESIDENT:
                raise SaracuraBackendError("another Saracura model is resident")
            _MODEL_RESIDENT = True
            self._owns_resident_slot = True
        try:
            import importlib

            torch = importlib.import_module("torch")
            transformers = importlib.import_module("transformers")
            tokenizers = importlib.import_module("tokenizers")
            safetensors = importlib.import_module("safetensors.torch")
            fallback = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")
            if self._device == "mps" and (
                not torch.backends.mps.is_available()
                or (fallback is not None and fallback.lower() not in {"", "0", "false"})
            ):
                raise SaracuraBackendError("MPS is unavailable or fallback-enabled")
            self._torch = torch
            self._prior_threads = torch.get_num_threads()
            torch.set_num_threads(self._cpu_threads)
            if self._snapshot is None:
                raise SaracuraBackendError("verified snapshot is unavailable")
            snapshot = self._snapshot
            config = self._json_bytes(snapshot["config.json"])
            tokenizer_config = self._json_bytes(snapshot["tokenizer_config.json"])
            special_tokens = self._special_tokens(snapshot["special_tokens_map.json"])
            if (
                config.get("model_type") != "bert"
                or config.get("hidden_size") != 384
                or tokenizer_config.get("tokenizer_class") != "PreTrainedTokenizerFast"
                or special_tokens != _SPECIAL_TOKENS
            ):
                raise SaracuraBackendError("verified MiniLM configuration is incompatible")
            model_config = transformers.BertConfig(**config)
            encoder = transformers.BertModel(model_config)
            state = safetensors.load(snapshot["model.safetensors"])
            expected = encoder.state_dict()
            position_ids = state.pop("embeddings.position_ids", None)
            expected_position_ids = torch.arange(
                model_config.max_position_embeddings, dtype=torch.long
            ).unsqueeze(0)
            if (
                not isinstance(position_ids, torch.Tensor)
                or position_ids.dtype != torch.long
                or not torch.equal(position_ids, expected_position_ids)
                or set(state) != set(expected)
            ):
                raise SaracuraBackendError("verified MiniLM state is incompatible")
            for name, expected_value in expected.items():
                actual = state[name]
                if (
                    not isinstance(actual, torch.Tensor)
                    or actual.shape != expected_value.shape
                    or actual.dtype != expected_value.dtype
                    or not actual.is_floating_point()
                    or not bool(torch.isfinite(actual).all().item())
                ):
                    raise SaracuraBackendError("verified MiniLM tensor is incompatible")
            encoder.load_state_dict(state, strict=True)
            token_object = tokenizers.Tokenizer.from_str(
                snapshot["tokenizer.json"].decode("utf-8", "strict")
            )
            mask = transformers.AddedToken(
                "<mask>", single_word=False, lstrip=True, rstrip=False, normalized=False
            )
            tokenizer = transformers.PreTrainedTokenizerFast(
                tokenizer_object=token_object,
                unk_token="<unk>",
                sep_token="</s>",
                pad_token="<pad>",
                cls_token="<s>",
                bos_token="<s>",
                eos_token="</s>",
                mask_token=mask,
                model_max_length=512,
                padding_side="right",
                truncation_side="right",
            )
            if (
                tokenizer.__class__.__name__ != "PreTrainedTokenizerFast"
                or tokenizer.vocab_size != 250002
                or len(tokenizer) != 250002
                or tokenizer.model_max_length != 512
                or {key: getattr(tokenizer, key) for key in _SPECIAL_TOKENS} != _SPECIAL_TOKENS
            ):
                raise SaracuraBackendError("verified MiniLM tokenizer is incompatible")
            self._encoder = encoder.to(self._device).eval()
            self._tokenizer = tokenizer
        except SaracuraBackendError:
            raise
        except Exception as error:
            raise SaracuraBackendError("optional ML runtime cannot be loaded") from error

    def _load_ranker_and_conformance(self) -> None:
        capsule, torch = self._verified_capsule, self._torch
        if capsule is None or torch is None:
            raise SaracuraBackendError("verified capsule is unavailable")
        try:
            import importlib

            load = importlib.import_module("safetensors.torch").load
            checkpoint = load(capsule.values["checkpoint.safetensors"])
            verify_checkpoint_tensors(checkpoint)
            vectors = load(capsule.values["conformance-vectors.safetensors"])
            self._verify_conformance_tensors(vectors)
            self._ranker = TorchSaracuraRanker(torch, checkpoint, device=self._device)
            self._run_conformance(vectors)
        except SaracuraBackendError:
            raise
        except Exception as error:
            raise SaracuraBackendError("Saracura model conformance failed") from error

    def _verify_conformance_tensors(self, vectors: Mapping[str, Any]) -> None:
        torch = self._torch
        if torch is None or set(vectors) != set(_CONFORMANCE_SHAPES):
            raise SaracuraBackendError("conformance tensor schema is invalid")
        if any(list(vectors[name].shape) != shape for name, shape in _CONFORMANCE_SHAPES.items()):
            raise SaracuraBackendError("conformance tensor shapes are invalid")
        if (
            vectors["context_embeddings"].dtype != torch.float32
            or vectors["criterion_embeddings"].dtype != torch.float32
            or vectors["expected_logits"].dtype != torch.float32
            or vectors["option_masks"].dtype != torch.uint8
            or vectors["option_counts"].dtype != torch.int64
            or not all(
                bool(torch.isfinite(vectors[name]).all().item())
                for name in ("context_embeddings", "criterion_embeddings", "expected_logits")
            )
        ):
            raise SaracuraBackendError("conformance tensor values are invalid")
        counts, masks = vectors["option_counts"], vectors["option_masks"]
        if not bool(((counts >= 2) & (counts <= 8)).all().item()) or not bool(
            ((masks == 0) | (masks == 1)).all().item()
        ):
            raise SaracuraBackendError("conformance masks are invalid")
        for index, cell in enumerate(_CONFORMANCE_CELLS):
            count = int(counts[index].item())
            expected_mask = torch.cat(
                (torch.ones(count, dtype=torch.uint8), torch.zeros(8 - count, dtype=torch.uint8))
            )
            if count != int(cell.rsplit(":", 1)[1]) or not torch.equal(
                masks[index].cpu(), expected_mask
            ):
                raise SaracuraBackendError("conformance cells are invalid")

    def _run_conformance(self, vectors: Mapping[str, Any]) -> None:
        if self._ranker is None or self._torch is None:
            raise SaracuraBackendError("ranker is unavailable")
        torch = self._torch
        with torch.inference_mode():
            for index in range(14):
                count = int(vectors["option_counts"][index].item())
                actual = (
                    self._ranker.logits(
                        vectors["context_embeddings"][index : index + 1].to(self._device),
                        vectors["criterion_embeddings"][index, :count].to(self._device),
                    )
                    .detach()
                    .to(device="cpu", dtype=torch.float32)
                )
                expected = vectors["expected_logits"][index, :count]
                if self._device == "cpu":
                    if not torch.equal(actual, expected):
                        raise SaracuraBackendError("CPU conformance failed")
                else:
                    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
                    if int(torch.argmax(actual).item()) != int(torch.argmax(expected).item()):
                        raise SaracuraBackendError("MPS conformance ranking failed")

    def _tokenize_joint(self, rendered: RenderedTask) -> dict[str, Any]:
        if self._tokenizer is None or self._torch is None:
            raise SaracuraError(
                ErrorCode.BACKEND_UNAVAILABLE, "Saracura backend is unavailable.", "/model"
            )
        texts = (rendered.context, *rendered.criteria)
        try:
            encoded = self._tokenizer(
                list(texts), truncation=False, padding=False, add_special_tokens=True
            )
            rows: object = encoded.get("input_ids") if isinstance(encoded, Mapping) else None
        except Exception:
            malformed = True
        else:
            malformed = False
        if malformed:
            raise SaracuraError(
                ErrorCode.BACKEND_UNAVAILABLE, "Saracura backend is unavailable.", "/model"
            )
        limits = (MAX_CONTEXT_TOKENS, *(MAX_CRITERION_TOKENS for _ in rendered.criteria))
        if (
            not isinstance(rows, list)
            or len(rows) != len(texts)
            or any(
                not isinstance(row, list) or not row or not all(type(token) is int for token in row)
                for row in rows
            )
        ):
            raise SaracuraError(
                ErrorCode.BACKEND_UNAVAILABLE, "Saracura backend is unavailable.", "/model"
            )
        if any(len(row) > limit for row, limit in zip(rows, limits, strict=True)):
            raise SaracuraError(
                ErrorCode.CAPACITY_EXCEEDED,
                "Rendered input exceeds tokenizer capacity.",
                "/questions",
            )
        torch = self._torch
        width = max(len(row) for row in rows)
        input_ids = torch.zeros((len(rows), width), dtype=torch.int64)
        attention = torch.zeros((len(rows), width), dtype=torch.int64)
        for index, row in enumerate(rows):
            input_ids[index, : len(row)] = torch.tensor(row, dtype=torch.int64)
            attention[index, : len(row)] = 1
        return {
            "input_ids": input_ids.to(self._device),
            "attention_mask": attention.to(self._device),
        }

    def _mean_pool(self, hidden: Any, attention_mask: Any) -> Any:
        if self._torch is None:
            raise ValueError("torch is unavailable")
        torch = self._torch
        if (
            hidden.ndim != 3
            or tuple(hidden.shape[2:]) != (384,)
            or attention_mask.ndim != 2
            or tuple(hidden.shape[:2]) != tuple(attention_mask.shape)
            or bool((attention_mask.sum(dim=1) <= 0).any().item())
        ):
            raise ValueError("encoder attention mask is invalid")
        mask = attention_mask.to(dtype=torch.float32).unsqueeze(-1)
        pooled = (hidden.to(dtype=torch.float32) * mask).sum(dim=1) / mask.sum(dim=1)
        if pooled.dtype != torch.float32 or not bool(torch.isfinite(pooled).all().item()):
            raise ValueError("encoder pooling is invalid")
        return pooled
