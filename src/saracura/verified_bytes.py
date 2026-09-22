"""Package-owned, descriptor-verified bytes for the optional MiniLM backend."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat as statmod
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from saracura.serialization import canonical_json_bytes

MAX_MANIFEST_BYTES: Final = 1_024 * 1_024
MAX_CHECKPOINT_BYTES: Final = 16 * 1_024 * 1_024
MAX_SNAPSHOT_OVERHEAD_BYTES: Final = 1_024 * 1_024
MINILM_CANDIDATE_ID: Final = "multilingual-minilm-l12"
MINILM_REVISION: Final = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"


class VerifiedBytesError(ValueError):
    """An operator artifact failed the local verified-bytes boundary."""


@dataclass(frozen=True, slots=True)
class RegistryFile:
    name: str
    size: int
    sha256: str


@dataclass(frozen=True, slots=True)
class MiniLMCandidate:
    id: str
    repository: str
    revision: str
    architecture: str
    hidden_width: int
    tokenizer: str
    files: tuple[RegistryFile, ...]

    @property
    def total_bytes(self) -> int:
        return sum(item.size for item in self.files)


@dataclass(frozen=True, slots=True)
class VerifiedMiniLMBytes:
    snapshot: Mapping[str, bytes]
    training_manifest: Mapping[str, Any]
    training_manifest_bytes: bytes
    checkpoint_bytes: bytes
    candidate: MiniLMCandidate


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, child in pairs:
        if key in value:
            raise VerifiedBytesError("duplicate JSON object key")
        value[key] = child
    return value


def _json_object(raw: bytes, *, maximum: int) -> dict[str, Any]:
    if len(raw) > maximum:
        raise VerifiedBytesError("JSON input exceeds its byte limit")
    try:
        value = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise VerifiedBytesError("JSON input is invalid") from error
    if not isinstance(value, dict):
        raise VerifiedBytesError("JSON root must be an object")
    return value


def package_registry_bytes() -> bytes:
    """Read the only runtime encoder registry from package resources."""

    return resources.files("saracura").joinpath("encoder-candidates.v1.json").read_bytes()


def package_registry_sha256() -> str:
    return hashlib.sha256(package_registry_bytes()).hexdigest()


def load_minilm_candidate(raw: bytes | None = None) -> MiniLMCandidate:
    """Parse only the reviewed MiniLM descriptor, never a benchmark registry."""

    registry_bytes = raw if raw is not None else package_registry_bytes()
    registry = _json_object(registry_bytes, maximum=1_024 * 1_024)
    if (
        set(registry) != {"schema_version", "candidates"}
        or registry.get("schema_version") != "encoder-candidates.v1"
    ):
        raise VerifiedBytesError("runtime registry shape is invalid")
    candidates = registry.get("candidates")
    if not isinstance(candidates, list):
        raise VerifiedBytesError("runtime registry candidates are invalid")
    selected = [
        item
        for item in candidates
        if isinstance(item, dict) and item.get("id") == MINILM_CANDIDATE_ID
    ]
    if len(selected) != 1:
        raise VerifiedBytesError("reviewed MiniLM candidate is unavailable")
    candidate = selected[0]
    required = {
        "id",
        "repository",
        "revision",
        "license",
        "role",
        "disposition",
        "reason",
        "reviewed_at",
        "model_type",
        "architecture",
        "hidden_width",
        "loader",
        "tokenizer",
        "files",
    }
    if set(candidate) != required or (
        candidate.get("id"),
        candidate.get("repository"),
        candidate.get("revision"),
        candidate.get("license"),
        candidate.get("role"),
        candidate.get("disposition"),
        candidate.get("model_type"),
        candidate.get("architecture"),
        candidate.get("hidden_width"),
        candidate.get("loader"),
        candidate.get("tokenizer"),
    ) != (
        MINILM_CANDIDATE_ID,
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        MINILM_REVISION,
        "Apache-2.0",
        "speed-control",
        "eligible",
        "bert",
        "BertModel",
        384,
        "bert",
        "PreTrainedTokenizerFast",
    ):
        raise VerifiedBytesError("runtime registry MiniLM descriptor is invalid")
    files_value = candidate.get("files")
    if not isinstance(files_value, list):
        raise VerifiedBytesError("runtime registry file ledger is invalid")
    files: list[RegistryFile] = []
    for entry in files_value:
        if not isinstance(entry, dict) or set(entry) != {"path", "bytes", "sha256"}:
            raise VerifiedBytesError("runtime registry file entry is invalid")
        name, size, digest = entry.get("path"), entry.get("bytes"), entry.get("sha256")
        if (
            not isinstance(name, str)
            or "/" in name
            or name in {"", ".", ".."}
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size <= 0
            or not _sha256(digest)
        ):
            raise VerifiedBytesError("runtime registry file entry is invalid")
        assert isinstance(digest, str)
        files.append(RegistryFile(name=name, size=size, sha256=digest))
    names = tuple(item.name for item in files)
    expected_names = (
        "config.json",
        "model.safetensors",
        "sentencepiece.bpe.model",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    )
    if names != expected_names:
        raise VerifiedBytesError("runtime registry file ledger is not the reviewed MiniLM ledger")
    return MiniLMCandidate(
        id=MINILM_CANDIDATE_ID,
        repository="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        revision=MINILM_REVISION,
        architecture="BertModel",
        hidden_width=384,
        tokenizer="PreTrainedTokenizerFast",
        files=tuple(files),
    )


def _sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mode,
        value.st_uid,
        value.st_nlink,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _directory_components(path: Path) -> tuple[str, ...]:
    """Return lexical path components without resolving any filesystem links."""

    absolute = path if path.is_absolute() else Path.cwd() / path
    if not absolute.is_absolute() or any(component == ".." for component in absolute.parts):
        raise VerifiedBytesError("operator directory path is invalid")
    anchor = absolute.anchor
    if not anchor or absolute.parts[0] != anchor:
        raise VerifiedBytesError("operator directory path is invalid")
    return tuple(component for component in absolute.parts[1:] if component not in {"", "."})


def _open_directory(path: Path) -> tuple[int, os.stat_result]:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    components = _directory_components(path)
    descriptor: int | None = None
    try:
        descriptor = os.open(Path("/").anchor, flags)
        for component in components:
            next_descriptor = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
    except OSError as error:
        if descriptor is not None:
            os.close(descriptor)
        raise VerifiedBytesError("operator directory cannot be opened safely") from error
    assert descriptor is not None
    result = os.fstat(descriptor)
    if (
        not statmod.S_ISDIR(result.st_mode)
        or result.st_uid != os.getuid()
        or result.st_mode & 0o777 != 0o700
    ):
        os.close(descriptor)
        raise VerifiedBytesError("operator directory ownership or mode is unsafe")
    return descriptor, result


def _open_parent(path: Path) -> tuple[int, str]:
    if path.name in {"", ".", ".."}:
        raise VerifiedBytesError("operator file path is invalid")
    descriptor, _ = _open_directory(path.parent)
    return descriptor, path.name


def _read_regular_at(
    directory_descriptor: int,
    name: str,
    *,
    maximum: int,
    exact_size: int | None = None,
    exact_sha256: str | None = None,
) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=directory_descriptor)
    except OSError as error:
        raise VerifiedBytesError("operator file cannot be opened safely") from error
    try:
        before = os.fstat(descriptor)
        if (
            not statmod.S_ISREG(before.st_mode)
            or before.st_uid != os.getuid()
            or before.st_mode & 0o777 != 0o600
            or before.st_nlink != 1
            or before.st_size > maximum
            or (exact_size is not None and before.st_size != exact_size)
        ):
            raise VerifiedBytesError("operator file ownership, mode, or size is unsafe")
        chunks: list[bytes] = []
        digest = hashlib.sha256()
        total = 0
        while True:
            chunk = os.read(descriptor, min(1_024 * 1_024, maximum + 1 - total))
            if not chunk:
                break
            total += len(chunk)
            if total > maximum:
                raise VerifiedBytesError("operator file exceeds its byte limit")
            chunks.append(chunk)
            digest.update(chunk)
        after = os.fstat(descriptor)
        try:
            named = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
        except OSError as error:
            raise VerifiedBytesError("operator file changed while being read") from error
        if (
            _stat_identity(before) != _stat_identity(after)
            or _stat_identity(before) != _stat_identity(named)
            or total != before.st_size
            or (exact_size is not None and total != exact_size)
            or (exact_sha256 is not None and digest.hexdigest() != exact_sha256)
        ):
            raise VerifiedBytesError("operator file changed or failed its descriptor ledger")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _entry_names(directory_descriptor: int) -> set[str]:
    try:
        with os.scandir(directory_descriptor) as entries:
            return {entry.name for entry in entries}
    except OSError as error:
        raise VerifiedBytesError("operator directory cannot be enumerated safely") from error


def _read_snapshot(snapshot: Path, candidate: MiniLMCandidate) -> Mapping[str, bytes]:
    directory, _ = _open_directory(snapshot)
    try:
        expected_names = {item.name for item in candidate.files} | {"snapshot.complete.json"}
        if _entry_names(directory) != expected_names:
            raise VerifiedBytesError("snapshot file set does not match the reviewed ledger")
        marker_bytes = _read_regular_at(
            directory,
            "snapshot.complete.json",
            maximum=MAX_MANIFEST_BYTES,
        )
        marker = _json_object(marker_bytes, maximum=MAX_MANIFEST_BYTES)
        expected_ledger = [
            {"path": item.name, "bytes": item.size, "sha256": item.sha256}
            for item in candidate.files
        ]
        if set(marker) != {"candidate_id", "revision", "registry_sha256", "files"} or (
            marker.get("candidate_id"),
            marker.get("revision"),
            marker.get("registry_sha256"),
            marker.get("files"),
        ) != (candidate.id, candidate.revision, package_registry_sha256(), expected_ledger):
            raise VerifiedBytesError("snapshot completion marker is invalid")
        values: dict[str, bytes] = {}
        total = len(marker_bytes)
        for item in candidate.files:
            data = _read_regular_at(
                directory,
                item.name,
                maximum=item.size,
                exact_size=item.size,
                exact_sha256=item.sha256,
            )
            values[item.name] = data
            total += len(data)
        if total > candidate.total_bytes + MAX_SNAPSHOT_OVERHEAD_BYTES:
            raise VerifiedBytesError("snapshot exceeds the descriptor total-byte limit")
        if _entry_names(directory) != expected_names:
            raise VerifiedBytesError("snapshot changed while being read")
        return MappingProxyType(values)
    finally:
        os.close(directory)


def _read_external_file(path: Path, *, maximum: int) -> bytes:
    directory, name = _open_parent(path)
    try:
        return _read_regular_at(directory, name, maximum=maximum)
    finally:
        os.close(directory)


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


def _require_hashes(value: Mapping[str, Any], keys: set[str]) -> None:
    if set(value) != keys or any(not _sha256(value[key]) for key in keys):
        raise VerifiedBytesError("training manifest digest fields are invalid")


def _validate_metrics(value: object) -> None:
    labels = (
        "billing",
        "technical_support",
        "account_access",
        "subscription_cancellation",
        "order_delivery",
    )
    if not isinstance(value, dict) or set(value) != {
        "synthetic_only",
        "epochs",
        "selected_epoch",
        "epoch_ledger",
        "train",
        "dev",
    }:
        raise VerifiedBytesError("training metrics shape is invalid")
    if value.get("synthetic_only") is not True:
        raise VerifiedBytesError("training metrics must remain synthetic-only")
    epochs = value.get("epochs")
    selected_epoch = value.get("selected_epoch")
    epoch_ledger = value.get("epoch_ledger")
    if (
        not isinstance(epochs, int)
        or isinstance(epochs, bool)
        or not isinstance(selected_epoch, int)
        or isinstance(selected_epoch, bool)
        or not isinstance(epoch_ledger, list)
        or not 1 <= epochs <= 80
        or not 1 <= selected_epoch <= epochs
        or len(epoch_ledger) != epochs
    ):
        raise VerifiedBytesError("training epoch ledger is invalid")
    best = float("-inf")
    expected_selected_epoch = 0
    stale = 0
    for expected_epoch, entry in enumerate(epoch_ledger, start=1):
        if (
            not isinstance(entry, dict)
            or set(entry) != {"epoch", "train_loss", "dev_macro_f1"}
            or entry.get("epoch") != expected_epoch
            or not isinstance(entry.get("train_loss"), (int, float))
            or isinstance(entry.get("train_loss"), bool)
            or not isinstance(entry.get("dev_macro_f1"), (int, float))
            or isinstance(entry.get("dev_macro_f1"), bool)
        ):
            raise VerifiedBytesError("training epoch ledger is invalid")
        train_loss = float(entry["train_loss"])
        dev_macro_f1 = float(entry["dev_macro_f1"])
        if (
            not math.isfinite(train_loss)
            or train_loss < 0.0
            or not math.isfinite(dev_macro_f1)
            or not 0.0 <= dev_macro_f1 <= 1.0
        ):
            raise VerifiedBytesError("training epoch ledger is invalid")
        if dev_macro_f1 > best + 0.001:
            best = dev_macro_f1
            expected_selected_epoch = expected_epoch
            stale = 0
        else:
            stale += 1
        if stale >= 10 and expected_epoch != epochs:
            raise VerifiedBytesError("training epoch ledger violates early stopping")
    if selected_epoch != expected_selected_epoch:
        raise VerifiedBytesError("training selected epoch is invalid")
    for split in ("train", "dev"):
        metric = value.get(split)
        if not isinstance(metric, dict) or set(metric) != {
            "count",
            "accuracy",
            "macro_f1",
            "confusion_matrix",
            "per_label",
        }:
            raise VerifiedBytesError("training metrics shape is invalid")
        if (
            not isinstance(metric["count"], int)
            or isinstance(metric["count"], bool)
            or metric["count"] <= 0
            or not isinstance(metric["confusion_matrix"], list)
            or len(metric["confusion_matrix"]) != 5
            or not isinstance(metric["per_label"], dict)
            or set(metric["per_label"]) != set(labels)
        ):
            raise VerifiedBytesError("training metrics content is invalid")
        for row in metric["confusion_matrix"]:
            if (
                not isinstance(row, list)
                or len(row) != 5
                or any(
                    not isinstance(item, int) or isinstance(item, bool) or item < 0 for item in row
                )
            ):
                raise VerifiedBytesError("training confusion matrix is invalid")
        for label in labels:
            per_label = metric["per_label"][label]
            if not isinstance(per_label, dict) or set(per_label) != {
                "support",
                "precision",
                "recall",
                "f1",
            }:
                raise VerifiedBytesError("training per-label metrics are invalid")
        if not _finite_json(metric):
            raise VerifiedBytesError("training metrics contain non-finite values")


def validate_training_manifest(raw: bytes, *, checkpoint_sha256: str) -> Mapping[str, Any]:
    """Validate the closed Phase 3B-S provenance shape needed by Phase 4A."""

    manifest = _json_object(raw, maximum=MAX_MANIFEST_BYTES)
    expected_keys = {
        "schema_version",
        "workflow_revision",
        "synthetic_only",
        "packet_manifest_sha256",
        "training_code_sha256",
        "protocol_sha256",
        "encoder",
        "labels",
        "seed",
        "head",
        "files",
        "metrics",
        "environment",
        "authorizations",
        "manifest_sha256",
    }
    if set(manifest) != expected_keys:
        raise VerifiedBytesError("training manifest shape is invalid")
    if (
        manifest.get("schema_version") != "synthetic-training-manifest.v1"
        or manifest.get("workflow_revision") != "phase3b-synthetic-research-training.v2"
        or manifest.get("synthetic_only") is not True
        or manifest.get("labels")
        != [
            "billing",
            "technical_support",
            "account_access",
            "subscription_cancellation",
            "order_delivery",
        ]
        or manifest.get("seed") != 20260921
        or manifest.get("authorizations")
        != {"calibration": False, "automation": False, "quality_claims": False}
    ):
        raise VerifiedBytesError("training manifest boundary is invalid")
    digest_fields = {
        "packet_manifest_sha256": manifest.get("packet_manifest_sha256"),
        "training_code_sha256": manifest.get("training_code_sha256"),
        "protocol_sha256": manifest.get("protocol_sha256"),
        "manifest_sha256": manifest.get("manifest_sha256"),
    }
    _require_hashes(digest_fields, set(digest_fields))
    unhashed = dict(manifest)
    recorded = unhashed.pop("manifest_sha256")
    if recorded != hashlib.sha256(canonical_json_bytes(unhashed)).hexdigest():
        raise VerifiedBytesError("training manifest self digest is invalid")
    encoder = manifest.get("encoder")
    if (
        not isinstance(encoder, dict)
        or set(encoder)
        != {
            "candidate",
            "revision",
            "registry_sha256",
            "device",
            "frozen",
        }
        or (
            encoder.get("candidate"),
            encoder.get("revision"),
            encoder.get("registry_sha256"),
            encoder.get("device"),
            encoder.get("frozen"),
        )
        != (
            MINILM_CANDIDATE_ID,
            MINILM_REVISION,
            package_registry_sha256(),
            "mps",
            True,
        )
    ):
        raise VerifiedBytesError("training encoder provenance is invalid")
    head = manifest.get("head")
    if not isinstance(head, dict) or head != {
        "width": 384,
        "classes": 5,
        "optimizer": "AdamW",
        "lr": 0.01,
        "weight_decay": 0.01,
        "batch_size": 32,
        "maximum_epochs": 80,
        "early_stopping_patience": 10,
        "early_stopping_min_delta": 0.001,
    }:
        raise VerifiedBytesError("training head contract is invalid")
    files = manifest.get("files")
    if (
        not isinstance(files, dict)
        or set(files) != {"embeddings.safetensors", "checkpoint.safetensors"}
        or not all(_sha256(value) for value in files.values())
        or files["checkpoint.safetensors"] != checkpoint_sha256
    ):
        raise VerifiedBytesError("training checkpoint ledger is invalid")
    environment = manifest.get("environment")
    if (
        not isinstance(environment, dict)
        or set(environment) != {"python", "torch", "safetensors", "platform"}
        or any(not isinstance(value, str) or not value for value in environment.values())
    ):
        raise VerifiedBytesError("training environment provenance is invalid")
    _validate_metrics(manifest.get("metrics"))
    return MappingProxyType(manifest)


def load_verified_minilm_bytes(
    *,
    encoder_snapshot: Path,
    training_manifest: Path,
    checkpoint: Path,
) -> VerifiedMiniLMBytes:
    """Read every operator artifact once through descriptor-checked file descriptors."""

    try:
        candidate = load_minilm_candidate()
        snapshot = _read_snapshot(encoder_snapshot, candidate)
        manifest_bytes = _read_external_file(training_manifest, maximum=MAX_MANIFEST_BYTES)
        checkpoint_bytes = _read_external_file(checkpoint, maximum=MAX_CHECKPOINT_BYTES)
        manifest = validate_training_manifest(
            manifest_bytes,
            checkpoint_sha256=hashlib.sha256(checkpoint_bytes).hexdigest(),
        )
        return VerifiedMiniLMBytes(
            snapshot=snapshot,
            training_manifest=manifest,
            training_manifest_bytes=manifest_bytes,
            checkpoint_bytes=checkpoint_bytes,
            candidate=candidate,
        )
    except VerifiedBytesError:
        raise
    except (OSError, TypeError, KeyError, UnicodeError) as error:
        raise VerifiedBytesError("operator artifacts could not be verified") from error
