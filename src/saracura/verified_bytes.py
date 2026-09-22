"""Package-owned, descriptor-verified bytes for the optional MiniLM backend."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat as statmod
import unicodedata
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
MAX_RECEIPT_BYTES: Final = 64 * 1_024
MINILM_CANDIDATE_ID: Final = "multilingual-minilm-l12"
MINILM_REVISION: Final = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
HUMAN_TRAINING_CAPSULE_FILES: Final = {
    "checkpoint.safetensors": MAX_CHECKPOINT_BYTES,
    "training-manifest.json": MAX_MANIFEST_BYTES,
    "training-report.json": MAX_MANIFEST_BYTES,
    "packet-manifest.json": MAX_MANIFEST_BYTES,
    "packet-release-receipt.json": MAX_RECEIPT_BYTES,
    "capsule-descriptor.json": MAX_MANIFEST_BYTES,
}


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


def _open_calibration_directory(path: Path) -> tuple[int, os.stat_result]:
    """Open a no-follow directory walk for calibration envelopes.

    Historic schema-v1 fixtures are intentionally public (0644) and commonly
    live below root-owned or shared read/execute-only directories.  The walk
    accepts owner/current-or-root directories with any non-writable mode
    (including 0750 and 0711), while rejecting every link and every
    group/world-writable ancestor.  Schema-v2 additionally requires an owned
    0600 leaf after this single read.
    """

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    components = _directory_components(path)
    descriptor: int | None = None
    try:
        descriptor = os.open(Path("/").anchor, flags)
        for component in components:
            current = os.fstat(descriptor)
            if (
                not statmod.S_ISDIR(current.st_mode)
                or current.st_uid not in {os.getuid(), 0}
                # The root-owned sticky /tmp ancestor and a root-owned mount
                # root are system-provided traversal boundaries.  They cannot
                # be substituted as an ordinary writable project ancestor;
                # every descendant and leaf is still no-follow and identity-
                # checked. Other group/world-writable ancestors remain unsafe.
                or (
                    current.st_mode & 0o022
                    and not (
                        current.st_uid == 0
                        and (
                            (current.st_mode & statmod.S_ISVTX and current.st_mode & 0o777 == 0o777)
                            or _is_root_mount(descriptor, current)
                        )
                    )
                )
            ):
                raise VerifiedBytesError("calibration directory ownership or mode is unsafe")
            next_descriptor = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
    except (OSError, VerifiedBytesError) as error:
        if descriptor is not None:
            os.close(descriptor)
        raise VerifiedBytesError("calibration directory cannot be opened safely") from error
    assert descriptor is not None
    result = os.fstat(descriptor)
    if (
        not statmod.S_ISDIR(result.st_mode)
        or result.st_uid not in {os.getuid(), 0}
        or result.st_mode & 0o022
    ):
        os.close(descriptor)
        raise VerifiedBytesError("calibration directory ownership or mode is unsafe")
    return descriptor, result


def _is_root_mount(directory_descriptor: int, current: os.stat_result) -> bool:
    """Recognize a root-owned mounted volume without trusting a pathname."""

    if current.st_uid != 0:
        return False
    try:
        parent = os.stat("..", dir_fd=directory_descriptor, follow_symlinks=False)
    except OSError:
        return False
    return current.st_dev != parent.st_dev


def _read_calibration_regular_at(
    directory_descriptor: int, name: str, *, maximum: int
) -> tuple[bytes, bool]:
    """Read a single 0600/0644 regular calibration envelope without a reread."""

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=directory_descriptor)
    except OSError as error:
        raise VerifiedBytesError("calibration file cannot be opened safely") from error
    try:
        before = os.fstat(descriptor)
        mode = before.st_mode & 0o777
        if (
            not statmod.S_ISREG(before.st_mode)
            or before.st_uid not in {os.getuid(), 0}
            or mode not in {0o600, 0o644}
            or before.st_nlink != 1
            or before.st_size > maximum
        ):
            raise VerifiedBytesError("calibration file ownership, mode, or size is unsafe")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1_024 * 1_024, maximum + 1 - total))
            if not chunk:
                break
            total += len(chunk)
            if total > maximum:
                raise VerifiedBytesError("calibration file exceeds its byte limit")
            chunks.append(chunk)
        after = os.fstat(descriptor)
        try:
            named = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
        except OSError as error:
            raise VerifiedBytesError("calibration file changed while being read") from error
        if (
            _stat_identity(before) != _stat_identity(after)
            or _stat_identity(before) != _stat_identity(named)
            or total != before.st_size
        ):
            raise VerifiedBytesError("calibration file changed while being read")
        return b"".join(chunks), before.st_uid == os.getuid() and mode == 0o600
    finally:
        os.close(descriptor)


def _read_regular_at(
    directory_descriptor: int,
    name: str,
    *,
    maximum: int,
    exact_size: int | None = None,
    exact_sha256: str | None = None,
) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
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


def read_verified_external_file(path: Path, *, maximum: int) -> bytes:
    """Read one explicit 0600 external file through the descriptor boundary."""

    return _read_external_file(path, maximum=maximum)


def read_calibration_external_file(path: Path, *, maximum: int) -> tuple[bytes, bool, bool]:
    """Read one calibration envelope once, bounded and no-follow.

    The booleans report whether the exact opened leaf and its final parent are
    private 0600/0700.  Signed schema-v2 requires the owned 0600 leaf; every
    ancestor is independently no-follow and non-group/world-writable, while
    legacy schema-v1 may retain a root-owned/current-user 0644 leaf.  No caller
    gets an unverified probe followed by a second parse read.
    """

    if path.name in {"", ".", ".."}:
        raise VerifiedBytesError("calibration file path is invalid")
    directory, parent = _open_calibration_directory(path.parent)
    try:
        raw, private_leaf = _read_calibration_regular_at(directory, path.name, maximum=maximum)
        return raw, private_leaf, parent.st_mode & 0o777 == 0o700
    finally:
        os.close(directory)


def read_public_external_file(path: Path, *, maximum: int) -> bytes:
    """Read one explicit public input without following links or blocking.

    Requests are allowed to use the historic 0644 checked-in fixture mode, but
    they still receive the calibration walk's bounded, O_NONBLOCK regular-file
    and identity checks.  This is intentionally separate from the private
    capsule reader so it does not relax capsule custody rules.
    """

    if path.name in {"", ".", ".."}:
        raise VerifiedBytesError("public file path is invalid")
    directory, _parent = _open_calibration_directory(path.parent)
    try:
        raw, _private_leaf = _read_calibration_regular_at(directory, path.name, maximum=maximum)
        return raw
    finally:
        os.close(directory)


def read_verified_minilm_snapshot(
    encoder_snapshot: Path,
) -> tuple[MiniLMCandidate, Mapping[str, bytes]]:
    """Snapshot the explicit reviewed MiniLM directory once through package I/O.

    The returned byte mapping is the only encoder input accepted by the human
    research extractor.  It deliberately exposes no path to transformers, so
    callers cannot re-open a cache or a snapshot sibling after verification.
    """

    candidate = load_minilm_candidate()
    return candidate, _read_snapshot(encoder_snapshot, candidate)


def open_verified_directory(path: Path) -> int:
    """Open one explicit owned 0700 directory through the no-follow walker.

    The caller owns the returned descriptor and must close it.
    """

    descriptor, _ = _open_directory(path)
    return descriptor


def read_verified_directory(
    path: Path, *, expected_files: Mapping[str, int]
) -> Mapping[str, bytes]:
    """Snapshot an exact 0700 capsule directory through one verified directory FD.

    ``expected_files`` is a closed basename-to-maximum-byte ledger. Every
    sibling is enumerated before and after reads, and each file is snapshotted
    once by :func:`_read_regular_at` while proving descriptor identity.
    """

    if not expected_files or any(
        not isinstance(name, str)
        or not name
        or "/" in name
        or name in {".", ".."}
        or not isinstance(maximum, int)
        or isinstance(maximum, bool)
        or maximum <= 0
        for name, maximum in expected_files.items()
    ):
        raise VerifiedBytesError("verified directory ledger is invalid")
    directory, _ = _open_directory(path)
    expected_names = set(expected_files)
    try:
        if _entry_names(directory) != expected_names:
            raise VerifiedBytesError("operator directory file set is invalid")
        values = {
            name: _read_regular_at(directory, name, maximum=maximum)
            for name, maximum in expected_files.items()
        }
        if _entry_names(directory) != expected_names:
            raise VerifiedBytesError("operator directory changed while being read")
        return MappingProxyType(values)
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


def _validate_metrics(value: object, *, synthetic_only: bool, strict_reconciliation: bool) -> None:
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
    if value.get("synthetic_only") is not synthetic_only:
        raise VerifiedBytesError("training metrics synthetic boundary is invalid")
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
        if (
            not _finite_json(metric)
            or not isinstance(metric["accuracy"], (int, float))
            or isinstance(metric["accuracy"], bool)
            or not isinstance(metric["macro_f1"], (int, float))
            or isinstance(metric["macro_f1"], bool)
            or not 0.0 <= float(metric["accuracy"]) <= 1.0
            or not 0.0 <= float(metric["macro_f1"]) <= 1.0
            or (
                strict_reconciliation
                and sum(sum(row) for row in metric["confusion_matrix"]) != metric["count"]
            )
        ):
            raise VerifiedBytesError("training metrics contain non-finite values")
        if not strict_reconciliation:
            continue
        supports = 0
        f1s: list[float] = []
        correct = 0
        for index, label in enumerate(labels):
            per_label = metric["per_label"][label]
            if (
                not isinstance(per_label["support"], int)
                or isinstance(per_label["support"], bool)
                or per_label["support"] != sum(metric["confusion_matrix"][index])
                or any(
                    not isinstance(per_label[name], (int, float))
                    or isinstance(per_label[name], bool)
                    or not 0.0 <= float(per_label[name]) <= 1.0
                    for name in ("precision", "recall", "f1")
                )
            ):
                raise VerifiedBytesError("training metric reconciliation is invalid")
            matrix = metric["confusion_matrix"]
            true_positive = matrix[index][index]
            false_positive = sum(matrix[row][index] for row in range(5)) - true_positive
            false_negative = sum(matrix[index]) - true_positive
            expected_precision = (
                true_positive / (true_positive + false_positive)
                if true_positive + false_positive
                else 0.0
            )
            expected_recall = (
                true_positive / (true_positive + false_negative)
                if true_positive + false_negative
                else 0.0
            )
            expected_f1 = (
                2 * expected_precision * expected_recall / (expected_precision + expected_recall)
                if expected_precision + expected_recall
                else 0.0
            )
            if (
                not math.isclose(
                    float(per_label["precision"]), expected_precision, rel_tol=0.0, abs_tol=1e-12
                )
                or not math.isclose(
                    float(per_label["recall"]), expected_recall, rel_tol=0.0, abs_tol=1e-12
                )
                or not math.isclose(float(per_label["f1"]), expected_f1, rel_tol=0.0, abs_tol=1e-12)
            ):
                raise VerifiedBytesError("training metric reconciliation is invalid")
            supports += per_label["support"]
            correct += metric["confusion_matrix"][index][index]
            f1s.append(expected_f1)
        if (
            supports != metric["count"]
            or not math.isclose(
                float(metric["accuracy"]), correct / metric["count"], rel_tol=0.0, abs_tol=1e-12
            )
            or not math.isclose(float(metric["macro_f1"]), sum(f1s) / 5, rel_tol=0.0, abs_tol=1e-12)
        ):
            raise VerifiedBytesError("training metric reconciliation is invalid")
    selected_dev_macro_f1 = float(epoch_ledger[selected_epoch - 1]["dev_macro_f1"])
    if not math.isclose(
        float(value["dev"]["macro_f1"]), selected_dev_macro_f1, rel_tol=0.0, abs_tol=1e-12
    ):
        raise VerifiedBytesError("training selected metric is invalid")


def validate_training_manifest(raw: bytes, *, checkpoint_sha256: str) -> Mapping[str, Any]:
    """Validate the closed Phase 3B-S provenance shape needed by Phase 4A."""

    manifest = _json_object(raw, maximum=MAX_MANIFEST_BYTES)
    if manifest.get("schema_version") == "human-training-manifest.v1":
        return _validate_human_training_manifest(manifest, checkpoint_sha256=checkpoint_sha256)
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
    _validate_metrics(manifest.get("metrics"), synthetic_only=True, strict_reconciliation=False)
    return MappingProxyType(manifest)


def _validate_human_training_manifest(
    manifest: dict[str, Any], *, checkpoint_sha256: str
) -> Mapping[str, Any]:
    """Validate the discriminated, checkpoint-specific human training contract.

    This deliberately shares no permissive fields with the synthetic branch.  A
    human manifest is useful to the runtime only after its own serving vector
    and immutable artifact digests have been checked.
    """
    expected = {
        "schema_version",
        "workflow_revision",
        "synthetic_only",
        "packet_manifest_sha256",
        "packet_release_receipt_sha256",
        "embedding_manifest_sha256",
        "training_code_sha256",
        "protocol_sha256",
        "encoder",
        "labels",
        "seed",
        "head",
        "files",
        "metrics",
        "environment",
        "serving_conformance",
        "authorizations",
        "manifest_sha256",
    }
    if set(manifest) != expected or manifest["schema_version"] != "human-training-manifest.v1":
        raise VerifiedBytesError("human training manifest shape is invalid")
    if (
        manifest["workflow_revision"] != "phase4b-human-research-training.v1"
        or manifest["synthetic_only"] is not False
        or manifest["labels"]
        != [
            "billing",
            "technical_support",
            "account_access",
            "subscription_cancellation",
            "order_delivery",
        ]
        or manifest["seed"] != 20260921
        or manifest["authorizations"]
        != {"calibration": True, "automation": False, "quality_claims": False}
    ):
        raise VerifiedBytesError("human training manifest boundary is invalid")
    _require_hashes(
        {
            name: manifest.get(name)
            for name in (
                "packet_manifest_sha256",
                "packet_release_receipt_sha256",
                "embedding_manifest_sha256",
                "training_code_sha256",
                "protocol_sha256",
                "manifest_sha256",
            )
        },
        {
            "packet_manifest_sha256",
            "packet_release_receipt_sha256",
            "embedding_manifest_sha256",
            "training_code_sha256",
            "protocol_sha256",
            "manifest_sha256",
        },
    )
    unsigned = dict(manifest)
    recorded = unsigned.pop("manifest_sha256")
    if recorded != hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest():
        raise VerifiedBytesError("human training manifest self digest is invalid")
    encoder = manifest["encoder"]
    if (
        not isinstance(encoder, dict)
        or set(encoder) != {"candidate", "revision", "registry_sha256", "device", "frozen"}
        or (
            encoder.get("candidate"),
            encoder.get("revision"),
            encoder.get("registry_sha256"),
            encoder.get("device"),
            encoder.get("frozen"),
        )
        != (MINILM_CANDIDATE_ID, MINILM_REVISION, package_registry_sha256(), "mps", True)
    ):
        raise VerifiedBytesError("human training encoder provenance is invalid")
    if manifest["head"] != {
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
        raise VerifiedBytesError("human training head contract is invalid")
    files = manifest["files"]
    if (
        not isinstance(files, dict)
        or set(files) != {"embeddings.safetensors", "checkpoint.safetensors"}
        or any(not _sha256(value) for value in files.values())
        or files["checkpoint.safetensors"] != checkpoint_sha256
    ):
        raise VerifiedBytesError("human training checkpoint ledger is invalid")
    environment = manifest["environment"]
    if (
        not isinstance(environment, dict)
        or set(environment)
        != {"python", "torch", "transformers", "tokenizers", "safetensors", "platform"}
        or any(not isinstance(value, str) or not value for value in environment.values())
    ):
        raise VerifiedBytesError("human training environment provenance is invalid")
    _validate_metrics(manifest["metrics"], synthetic_only=False, strict_reconciliation=True)
    conformance = manifest["serving_conformance"]
    if (
        not isinstance(conformance, dict)
        or set(conformance)
        != {
            "schema",
            "token_payload_sha256",
            "labels",
            "cpu_logits",
            "mps_logits",
            "cpu_logits_sha256",
            "mps_logits_sha256",
            "comparison",
        }
        or conformance["schema"] != "human-minilm-conformance.v1"
        or conformance["labels"] != manifest["labels"]
        or not _sha256(conformance["token_payload_sha256"])
        or not _sha256(conformance["cpu_logits_sha256"])
        or not _sha256(conformance["mps_logits_sha256"])
        or conformance["comparison"] != {"rtol": 0.00001, "atol": 0.000001}
        or not isinstance(conformance["cpu_logits"], list)
        or not isinstance(conformance["mps_logits"], list)
        or len(conformance["cpu_logits"]) != 5
        or len(conformance["mps_logits"]) != 5
        or any(
            not isinstance(item, (int, float)) or isinstance(item, bool) or not math.isfinite(item)
            for item in [*conformance["cpu_logits"], *conformance["mps_logits"]]
        )
        or hashlib.sha256(canonical_json_bytes(conformance["cpu_logits"])).hexdigest()
        != conformance["cpu_logits_sha256"]
        or hashlib.sha256(canonical_json_bytes(conformance["mps_logits"])).hexdigest()
        != conformance["mps_logits_sha256"]
        or all(float(item) == 0.0 for item in conformance["cpu_logits"])
        or all(float(item) == 0.0 for item in conformance["mps_logits"])
        or any(
            not math.isclose(float(cpu), float(mps), rel_tol=1e-5, abs_tol=1e-6)
            for cpu, mps in zip(conformance["cpu_logits"], conformance["mps_logits"], strict=True)
        )
    ):
        raise VerifiedBytesError("human serving conformance is invalid")
    return MappingProxyType(manifest)


def _canonical_external_object(raw: bytes, *, maximum: int) -> dict[str, Any]:
    if (
        not raw
        or len(raw) > maximum
        or raw.startswith(b"\xef\xbb\xbf")
        or b"\r" in raw
        or not raw.endswith(b"\n")
        or b"\n" in raw[:-1]
    ):
        raise VerifiedBytesError("human capsule JSON framing is invalid")
    value = _json_object(raw[:-1], maximum=maximum)

    def nfc(item: object) -> bool:
        if isinstance(item, str):
            return item == unicodedata.normalize("NFC", item)
        if isinstance(item, list):
            return all(nfc(child) for child in item)
        if isinstance(item, dict):
            return all(
                isinstance(key, str) and key == unicodedata.normalize("NFC", key) and nfc(child)
                for key, child in item.items()
            )
        return item is None or type(item) in {bool, int, float}

    if not nfc(value) or not _finite_json(value):
        raise VerifiedBytesError("human capsule JSON values are invalid")
    if canonical_json_bytes(value) != raw[:-1]:
        raise VerifiedBytesError("human capsule JSON is not canonical")
    return value


def _human_descriptor(values: Mapping[str, bytes]) -> Mapping[str, Any]:
    descriptor = _canonical_external_object(
        values["capsule-descriptor.json"], maximum=MAX_MANIFEST_BYTES
    )
    expected = {
        "schema_version",
        "capsule_kind",
        "packet_sha256",
        "packet_release_receipt_sha256",
        "files",
        "descriptor_sha256",
    }
    if (
        set(descriptor) != expected
        or descriptor.get("schema_version") != "human-capsule.v1"
        or descriptor.get("capsule_kind") != "training"
        or not _sha256(descriptor.get("packet_sha256"))
        or not _sha256(descriptor.get("packet_release_receipt_sha256"))
        or not _sha256(descriptor.get("descriptor_sha256"))
    ):
        raise VerifiedBytesError("human training capsule descriptor is invalid")
    unsigned = dict(descriptor)
    recorded = unsigned.pop("descriptor_sha256")
    if hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest() != recorded:
        raise VerifiedBytesError("human training capsule descriptor digest is invalid")
    ledger = descriptor.get("files")
    names = set(values) - {"capsule-descriptor.json"}
    if (
        not isinstance(ledger, list)
        or len(ledger) != len(names)
        or [item.get("path") for item in ledger if isinstance(item, dict)] != sorted(names)
    ):
        raise VerifiedBytesError("human training capsule ledger is invalid")
    for item in ledger:
        if (
            not isinstance(item, dict)
            or set(item) != {"path", "bytes", "sha256"}
            or item.get("path") not in names
            or type(item.get("bytes")) is not int
            or item["bytes"] <= 0
            or not _sha256(item.get("sha256"))
            or item["bytes"] != len(values[item["path"]])
            or item["sha256"] != hashlib.sha256(values[item["path"]]).hexdigest()
        ):
            raise VerifiedBytesError("human training capsule ledger is invalid")
    return MappingProxyType(descriptor)


def _human_packet(packet_raw: bytes) -> Mapping[str, Any]:
    packet = _canonical_external_object(packet_raw, maximum=MAX_MANIFEST_BYTES)
    required = {
        "schema_version",
        "protocol_sha256",
        "policy_registry_sha256",
        "states_sha256",
        "split_plan_sha256",
        "contributors_sha256",
        "annotations_sha256",
        "adjudications_sha256",
        "controls_sha256",
        "takedown_ledger_sha256",
        "files",
        "accepted_counts",
        "excluded_counts",
        "labels",
        "split_algorithm",
        "source_type",
        "locale",
        "synthetic_only",
        "authorizations",
        "packet_sha256",
    }
    hash_fields = {name for name in required if name.endswith("_sha256")}
    if (
        set(packet) != required
        or packet.get("schema_version") != "support-routing-human-packet.v1"
        or any(not _sha256(packet.get(name)) for name in hash_fields)
        or packet.get("labels")
        != [
            "billing",
            "technical_support",
            "account_access",
            "subscription_cancellation",
            "order_delivery",
        ]
        or packet.get("split_algorithm") != "support-routing-stratified-sha256.v1"
        or packet.get("source_type") != "human_original"
        or packet.get("locale") != "pt-BR"
        or packet.get("synthetic_only") is not False
        or packet.get("authorizations")
        != {
            "training": False,
            "calibration": False,
            "blind_test": False,
            "automation": False,
            "broad_quality_claims": False,
        }
    ):
        raise VerifiedBytesError("human packet manifest is invalid")
    unsigned = dict(packet)
    digest = unsigned.pop("packet_sha256")
    if hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest() != digest:
        raise VerifiedBytesError("human packet manifest digest is invalid")
    files = packet["files"]
    counts = packet["accepted_counts"]
    excluded = packet["excluded_counts"]
    splits = ("train", "dev", "calibration", "blind_test")
    labels = (
        "billing",
        "technical_support",
        "account_access",
        "subscription_cancellation",
        "order_delivery",
    )
    if (
        not isinstance(files, dict)
        or set(files) != set(splits)
        or not isinstance(counts, dict)
        or set(counts) != {"total", "by_split", "by_split_label"}
        or type(counts["total"]) is not int
        or counts["total"] < 500
        or not isinstance(counts["by_split"], dict)
        or set(counts["by_split"]) != set(splits)
        or not isinstance(counts["by_split_label"], dict)
        or set(counts["by_split_label"]) != set(splits)
        or not isinstance(excluded, dict)
        or set(excluded) != {"total", "by_reason"}
        or type(excluded["total"]) is not int
        or excluded["total"] < 0
        or not isinstance(excluded["by_reason"], dict)
        or set(excluded["by_reason"])
        != {
            "backend_input_limit",
            "review_rejected",
            "review_disagreement_excluded",
            "adjudication_excluded",
        }
        or any(type(value) is not int or value < 0 for value in excluded["by_reason"].values())
        or excluded["total"] != sum(excluded["by_reason"].values())
    ):
        raise VerifiedBytesError("human packet manifest ledger is invalid")
    total = 0
    for split in splits:
        ledger = files[split]
        split_labels = counts["by_split_label"][split]
        split_count = counts["by_split"][split]
        if (
            not isinstance(ledger, dict)
            or set(ledger) != {"bytes", "sha256"}
            or type(ledger["bytes"]) is not int
            or ledger["bytes"] <= 0
            or not _sha256(ledger["sha256"])
            or type(split_count) is not int
            or split_count < (200 if split == "train" else 100)
            or not isinstance(split_labels, dict)
            or set(split_labels) != set(labels)
            or any(
                type(split_labels[label]) is not int or split_labels[label] < 10 for label in labels
            )
            or sum(split_labels.values()) != split_count
        ):
            raise VerifiedBytesError("human packet manifest ledger is invalid")
        total += split_count
    if total != counts["total"]:
        raise VerifiedBytesError("human packet manifest count is invalid")
    return MappingProxyType(packet)


def _verified_human_training_capsule(
    training_capsule: Path,
) -> tuple[Mapping[str, bytes], Mapping[str, Any], bytes, bytes]:
    """Verify the complete closed human runtime input before model construction."""

    values = read_verified_directory(training_capsule, expected_files=HUMAN_TRAINING_CAPSULE_FILES)
    descriptor = _human_descriptor(values)
    packet = _human_packet(values["packet-manifest.json"])
    try:
        from saracura.research_trust import verify_release_receipt

        receipt = verify_release_receipt(values["packet-release-receipt.json"])
    except Exception as exc:
        raise VerifiedBytesError("human packet receipt is invalid") from exc
    evidence = receipt.get("evidence")
    expected_evidence = {
        "protocol": packet["protocol_sha256"],
        "policy_registry": packet["policy_registry_sha256"],
        "states": packet["states_sha256"],
        "split_plan": packet["split_plan_sha256"],
        "contributors": packet["contributors_sha256"],
        "annotations": packet["annotations_sha256"],
        "adjudications": packet["adjudications_sha256"],
        "controls": packet["controls_sha256"],
        "takedown_ledger": packet["takedown_ledger_sha256"],
        "packet": packet["packet_sha256"],
    }
    if receipt.get("scope") != "packet_training" or evidence != expected_evidence:
        raise VerifiedBytesError("human packet receipt binding is invalid")
    if (
        descriptor["packet_sha256"] != packet["packet_sha256"]
        or descriptor["packet_release_receipt_sha256"]
        != hashlib.sha256(values["packet-release-receipt.json"]).hexdigest()
    ):
        raise VerifiedBytesError("human training capsule governance is invalid")
    manifest_raw = values["training-manifest.json"]
    checkpoint = values["checkpoint.safetensors"]
    manifest = validate_training_manifest(
        manifest_raw,
        checkpoint_sha256=hashlib.sha256(checkpoint).hexdigest(),
    )
    report = _canonical_external_object(values["training-report.json"], maximum=MAX_MANIFEST_BYTES)
    if (
        manifest.get("schema_version") != "human-training-manifest.v1"
        or manifest["packet_manifest_sha256"]
        != hashlib.sha256(values["packet-manifest.json"]).hexdigest()
        or manifest["packet_release_receipt_sha256"]
        != hashlib.sha256(values["packet-release-receipt.json"]).hexdigest()
        or manifest["protocol_sha256"] != packet["protocol_sha256"]
        or report != {"schema_version": "human-training-report.v1", "metrics": manifest["metrics"]}
    ):
        raise VerifiedBytesError("human training manifest capsule binding is invalid")
    return values, manifest, manifest_raw, checkpoint


def load_verified_minilm_bytes(
    *,
    encoder_snapshot: Path,
    training_manifest: Path | None = None,
    checkpoint: Path | None = None,
    training_capsule: Path | None = None,
) -> VerifiedMiniLMBytes:
    """Read a synthetic pair or a fully sealed human capsule before loading ML.

    The historical Phase 4A synthetic lane remains its explicit manifest plus
    checkpoint pair.  A human manifest is never accepted through that pair:
    it must arrive inside the descriptor-backed training capsule with its
    signed packet-training receipt.
    """

    try:
        candidate = load_minilm_candidate()
        if training_capsule is not None:
            if training_manifest is not None or checkpoint is not None:
                raise VerifiedBytesError("human training capsule input is ambiguous")
            _values, manifest, manifest_bytes, checkpoint_bytes = _verified_human_training_capsule(
                training_capsule
            )
        else:
            if training_manifest is None or checkpoint is None:
                raise VerifiedBytesError("synthetic training inputs are incomplete")
            manifest_bytes = _read_external_file(training_manifest, maximum=MAX_MANIFEST_BYTES)
            checkpoint_bytes = _read_external_file(checkpoint, maximum=MAX_CHECKPOINT_BYTES)
            manifest = validate_training_manifest(
                manifest_bytes,
                checkpoint_sha256=hashlib.sha256(checkpoint_bytes).hexdigest(),
            )
            if manifest.get("schema_version") == "human-training-manifest.v1":
                raise VerifiedBytesError("human runtime requires a sealed training capsule")
        snapshot = _read_snapshot(encoder_snapshot, candidate)
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
