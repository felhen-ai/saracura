"""Offline deterministic training for the Phase 4E Saracura ranker.

This checkout-only lane consumes explicit, sealed embedding capsules.  Before
training, it re-derives only train/dev tensors from an explicit
descriptor-verified local MiniLM snapshot.  Holdout content remains sealed
until the checkpoint digest, frozen pre-holdout gate, and irreversible claim
exist; it is then fully re-verified before scoring.  This module has no
provider client, cache lookup, or socket path.  Optional ML imports and encoder
loading remain delayed until extraction or training.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from benchmarks.encoder_loader import LoadedEncoder, VerifiedSnapshot, load_encoder
from benchmarks.encoder_registry import get_candidate
from benchmarks.io import atomic_create
from benchmarks.saracura_universal_corpus import (
    CorpusError,
    _publish_packet_create_if_absent,
    validate_accepted_packet,
    validate_accepted_packet_pre_holdout,
)
from benchmarks.saracura_universal_policy import validate_phase4e_policy
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
    render_task,
)
from saracura.universal.tasks import UniversalTask

_HEX = set("0123456789abcdef")
_SPLITS = ("synthetic_train", "synthetic_dev", "synthetic_holdout")
_CELLS = tuple((locale, count) for locale in ("pt-BR", "en") for count in range(2, 9))
_CAPSULE_FILES = {
    "train-dev.safetensors",
    "embedding-manifest.json",
    "accepted-packet-receipt.json",
    "capsule-descriptor.json",
}
_PRE_HOLDOUT_FAILED_OUTPUT_FILES = {
    "checkpoint.safetensors",
    "training-manifest.json",
    "epoch-ledger.json",
    "dev-selection.json",
    "pre-holdout-gate.json",
    "accepted-packet-receipt.json",
    "embedding-descriptor.json",
    "architecture.json",
    "capsule-descriptor.json",
}
_FINAL_OUTPUT_FILES = _PRE_HOLDOUT_FAILED_OUTPUT_FILES | {
    "holdout-release.json",
    "holdout-report.json",
    "conformance-vectors.safetensors",
    "conformance-manifest.json",
    "dependency-versions.json",
}
_EMBEDDING_TENSORS = {
    "context_embeddings",
    "criterion_embeddings",
    "context_token_ids",
    "context_attention_masks",
    "criterion_token_ids",
    "criterion_attention_masks",
}
_PRE_HOLDOUT_GATE_SCHEMA = "phase4e-pre-holdout-gate.v1"
_HOLDOUT_RELEASE_SCHEMA = "phase4e-holdout-release.v4"
_REQUIRED_LOCALES = ("pt-BR", "en")
_REQUIRED_OPTION_COUNTS = tuple(range(2, 9))
_MAX_ENCODER_MICROBATCH = 20
_ZERO_INVALID_INVARIANTS = {
    "non_finite_logits": 0,
    "missing_choices": 0,
    "extra_choices": 0,
    "silent_truncations": 0,
    "family_overlaps": 0,
    "deterministic_repeat_mismatches": 0,
}


class TrainingError(ValueError):
    """A closed Phase 4E offline training invariant was violated."""


def _canonical(value: object) -> bytes:
    return canonical_json_bytes(cast(Any, value))


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _is_sha(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, ValueError) as error:
        raise TrainingError("training capsule JSON") from error
    if not isinstance(value, dict):
        raise TrainingError("training capsule JSON")
    return cast(dict[str, Any], value)


def _descriptor(files: Mapping[str, bytes], *, kind: str) -> bytes:
    ledger = [
        {"path": name, "bytes": len(files[name]), "sha256": _sha(files[name])}
        for name in sorted(files)
    ]
    value: dict[str, object] = {
        "schema_version": "phase4e-training-capsule.v1",
        "capsule_kind": kind,
        "files": ledger,
        "descriptor_sha256": "",
    }
    value["descriptor_sha256"] = _sha(
        _canonical({k: v for k, v in value.items() if k != "descriptor_sha256"})
    )
    return _canonical(value) + b"\n"


def _validate_descriptor(
    path: Path,
    names: set[str],
    *,
    kind: str,
    content_names: set[str] | None = None,
) -> None:
    """Validate a closed descriptor and its listed file bytes."""
    value = _json(path)
    expected = {"schema_version", "capsule_kind", "files", "descriptor_sha256"}
    if (
        set(value) != expected
        or value["schema_version"] != "phase4e-training-capsule.v1"
        or value["capsule_kind"] != kind
    ):
        raise TrainingError("training capsule descriptor")
    recorded = value["descriptor_sha256"]
    if not _is_sha(recorded) or recorded != _sha(
        _canonical({k: v for k, v in value.items() if k != "descriptor_sha256"})
    ):
        raise TrainingError("training capsule descriptor digest")
    ledger = value["files"]
    if not isinstance(ledger, list) or [
        item.get("path") for item in ledger if isinstance(item, dict)
    ] != sorted(names):
        raise TrainingError("training capsule ledger")
    checked_names = names if content_names is None else content_names
    if not checked_names <= names:
        raise TrainingError("training capsule ledger")
    for item in ledger:
        if not isinstance(item, dict) or set(item) != {"path", "bytes", "sha256"}:
            raise TrainingError("training capsule ledger")
        name = item["path"]
        if (
            not isinstance(name, str)
            or name not in names
            or not isinstance(item["bytes"], int)
            or not _is_sha(item["sha256"])
        ):
            raise TrainingError("training capsule ledger")
        if name in checked_names:
            raw = (path.parent / name).read_bytes()
            if item["bytes"] != len(raw) or item["sha256"] != _sha(raw):
                raise TrainingError("training capsule ledger")


@dataclass(frozen=True)
class EmbeddingCapsule:
    path: Path
    manifest: dict[str, Any]
    receipt: dict[str, Any]
    descriptor_sha256: str


@dataclass(frozen=True)
class AcceptedPacketBinding:
    """Lineage derived from a sealed packet, never from an embedding receipt."""

    packet_json_sha256: str
    accepted_train_dev_jsonl_sha256: str
    accepted_holdout_jsonl_sha256: str
    holdout_identities_json_sha256: str
    accepted_rows_sha256: str
    train_dev_identities: tuple[dict[str, Any], ...]
    holdout_identities: tuple[dict[str, Any], ...]
    identities: tuple[dict[str, Any], ...]


def _identity_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    """Return the complete embedding identity for accepted or embedded rows."""

    identities: list[dict[str, Any]] = []
    for row in rows:
        try:
            criteria = row["criteria"]
            selected = row["selected_criterion_id"]
            option_count = len(criteria)
            gold_position = next(
                index
                for index, criterion in enumerate(criteria)
                if isinstance(criterion, Mapping) and criterion.get("id") == selected
            )
        except (KeyError, StopIteration, TypeError) as error:
            # Embedding rows carry their derived values, while packet rows carry
            # the source criteria.  Keep both representations closed.
            try:
                option_count = row["option_count"]
                gold_position = row["gold_position"]
            except KeyError:
                raise TrainingError("accepted packet row identity") from error
        identity = {
            "task_id": row.get("task_id"),
            "family_id": row.get("family_id"),
            "pair_id": row.get("pair_id"),
            "split": row.get("split"),
            "locale": row.get("locale"),
            "domain": row.get("domain"),
            "axes": row.get("axes"),
            "option_count": option_count,
            "gold_position": gold_position,
            "criterion_ids": [
                criterion.get("id") if isinstance(criterion, Mapping) else None
                for criterion in criteria
            ]
            if "criteria" in row
            else row.get("criterion_ids"),
        }
        if (
            not isinstance(identity["task_id"], str)
            or not isinstance(identity["family_id"], str)
            or (identity["pair_id"] is not None and not isinstance(identity["pair_id"], str))
            or identity["split"] not in _SPLITS
            or identity["locale"] not in {"pt-BR", "en"}
            or not isinstance(identity["domain"], str)
            or not isinstance(identity["axes"], dict)
            or type(identity["option_count"]) is not int
            or not 2 <= identity["option_count"] <= 8
            or type(identity["gold_position"]) is not int
            or not 0 <= identity["gold_position"] < identity["option_count"]
            or not isinstance(identity["criterion_ids"], list)
            or len(identity["criterion_ids"]) != identity["option_count"]
            or not all(isinstance(item, str) for item in identity["criterion_ids"])
            or len(set(cast(list[str], identity["criterion_ids"]))) != identity["option_count"]
        ):
            raise TrainingError("accepted packet row identity")
        identities.append(identity)
    if len({cast(str, row["task_id"]) for row in identities}) != len(identities):
        raise TrainingError("accepted packet row identity")
    return tuple(sorted(identities, key=lambda row: cast(str, row["task_id"])))


def _accepted_identity_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    """Bind every identity axis, including valid repeated cross-locale families."""

    return _sha(_canonical(_identity_rows(rows)))


def validate_accepted_packet_binding(packet: Path) -> AcceptedPacketBinding:
    """Bind the pre-claim train/dev payload and holdout identity projection."""

    try:
        validate_accepted_packet_pre_holdout(packet)
    except (CorpusError, OSError, TypeError, ValueError) as error:
        raise TrainingError("accepted packet validation") from error
    try:
        packet_raw = (packet / "packet.json").read_bytes()
        manifest = json.loads(packet_raw)
        train_dev_raw = (packet / "accepted-train-dev.jsonl").read_bytes()
        train_dev = [
            cast(dict[str, Any], json.loads(line)) for line in train_dev_raw.splitlines() if line
        ]
        identities_raw = (packet / "holdout-identities.json").read_bytes()
        identities_payload = json.loads(identities_raw)
    except (OSError, TypeError, ValueError) as error:
        raise TrainingError("accepted packet binding") from error
    if (
        not isinstance(manifest, dict)
        or not isinstance(manifest.get("files"), dict)
        or not isinstance(identities_payload, dict)
        or not isinstance(identities_payload.get("rows"), list)
    ):
        raise TrainingError("accepted packet binding")
    train_dev_identities = _identity_rows(train_dev)
    holdout_identities = _identity_rows(cast(list[Mapping[str, Any]], identities_payload["rows"]))
    identities = tuple(
        sorted((*train_dev_identities, *holdout_identities), key=lambda row: row["task_id"])
    )
    files = cast(dict[str, Any], manifest["files"])
    holdout_digest = files.get("accepted-holdout.jsonl")
    if not _is_sha(holdout_digest):
        raise TrainingError("accepted packet binding")
    return AcceptedPacketBinding(
        packet_json_sha256=_sha(packet_raw),
        accepted_train_dev_jsonl_sha256=_sha(train_dev_raw),
        accepted_holdout_jsonl_sha256=cast(str, holdout_digest),
        holdout_identities_json_sha256=_sha(identities_raw),
        accepted_rows_sha256=_sha(_canonical(identities)),
        train_dev_identities=train_dev_identities,
        holdout_identities=holdout_identities,
        identities=identities,
    )


def validate_full_accepted_packet_binding(
    packet: Path,
    expected: AcceptedPacketBinding,
    capsule: EmbeddingCapsule,
    checkpoint_sha256: str,
    pre_holdout_gate_descriptor_sha256: str,
) -> AcceptedPacketBinding:
    """Open and bind accepted holdout content only after its release claim."""

    _validate_holdout_release_claim(
        expected,
        capsule,
        checkpoint_sha256,
        pre_holdout_gate_descriptor_sha256,
    )
    try:
        validate_accepted_packet(packet)
        holdout_raw = (packet / "accepted-holdout.jsonl").read_bytes()
        holdout = [
            cast(dict[str, Any], json.loads(line)) for line in holdout_raw.splitlines() if line
        ]
    except (CorpusError, OSError, TypeError, ValueError) as error:
        raise TrainingError("full accepted packet validation") from error
    actual = validate_accepted_packet_binding(packet)
    holdout_identities = _identity_rows(holdout)
    if (
        _sha(holdout_raw) != expected.accepted_holdout_jsonl_sha256
        or holdout_identities != expected.holdout_identities
        or actual != expected
    ):
        raise TrainingError("full accepted packet binding")
    return actual


def validate_embedding_capsule(
    path: Path, accepted_packet: AcceptedPacketBinding | None = None
) -> EmbeddingCapsule:
    """Validate a v3 pre-holdout capsule without opening holdout content."""

    if not path.is_dir() or {child.name for child in path.iterdir()} != _CAPSULE_FILES:
        raise TrainingError("embedding capsule file set")
    _validate_descriptor(
        path / "capsule-descriptor.json",
        _CAPSULE_FILES - {"capsule-descriptor.json"},
        kind="embedding",
    )
    receipt = _json(path / "accepted-packet-receipt.json")
    if (
        set(receipt)
        != {
            "schema_version",
            "packet_json_sha256",
            "accepted_train_dev_jsonl_sha256",
            "accepted_holdout_jsonl_sha256",
            "holdout_identities_json_sha256",
            "accepted_rows_sha256",
            "synthetic_only",
        }
        or receipt["schema_version"] != "phase4e-accepted-packet-receipt.v4"
        or receipt["synthetic_only"] is not True
        or not _is_sha(receipt["packet_json_sha256"])
        or not _is_sha(receipt["accepted_train_dev_jsonl_sha256"])
        or not _is_sha(receipt["accepted_holdout_jsonl_sha256"])
        or not _is_sha(receipt["holdout_identities_json_sha256"])
        or not _is_sha(receipt["accepted_rows_sha256"])
    ):
        raise TrainingError("accepted packet receipt")
    manifest = _json(path / "embedding-manifest.json")
    expected = {
        "schema_version",
        "packet_receipt_sha256",
        "base_encoder",
        "rendering_revision",
        "extraction",
        "train_dev",
        "holdout_identities",
        "tensor_digests",
        "manifest_sha256",
    }
    if set(manifest) != expected or manifest["schema_version"] != "phase4e-embedding-capsule.v3":
        raise TrainingError("embedding manifest schema")
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if not _is_sha(manifest["manifest_sha256"]) or manifest["manifest_sha256"] != _sha(
        _canonical(unsigned)
    ):
        raise TrainingError("embedding manifest digest")
    if manifest["packet_receipt_sha256"] != _sha(
        (path / "accepted-packet-receipt.json").read_bytes()
    ):
        raise TrainingError("embedding packet receipt binding")
    if (
        manifest["base_encoder"]
        != {
            "id": BASE_ENCODER_ID,
            "revision": BASE_ENCODER_REVISION,
            "snapshot_complete_sha256": manifest["base_encoder"].get("snapshot_complete_sha256")
            if isinstance(manifest["base_encoder"], dict)
            else None,
            "frozen": True,
        }
        or not isinstance(manifest["base_encoder"], dict)
        or not _is_sha(manifest["base_encoder"].get("snapshot_complete_sha256"))
    ):
        raise TrainingError("embedding base encoder binding")
    if manifest["rendering_revision"] != RENDERER_REVISION:
        raise TrainingError("embedding rendering revision")
    extraction = manifest["extraction"]
    if (
        not isinstance(extraction, dict)
        or set(extraction) != {"device", "runtime_abi"}
        or extraction["device"] != "cpu"
        or not _is_sha(extraction["runtime_abi"])
    ):
        raise TrainingError("embedding extraction ABI")
    train_dev = manifest["train_dev"]
    if (
        not isinstance(train_dev, dict)
        or set(train_dev) != {"file", "sha256", "rows"}
        or train_dev["file"] != "train-dev.safetensors"
        or not _is_sha(train_dev["sha256"])
        or train_dev["sha256"] != _sha((path / "train-dev.safetensors").read_bytes())
        or not isinstance(train_dev["rows"], list)
    ):
        raise TrainingError("embedding split descriptor")
    _validate_rows(cast(list[object], train_dev["rows"]), {"synthetic_train", "synthetic_dev"})
    holdout = manifest["holdout_identities"]
    if (
        not isinstance(holdout, dict)
        or set(holdout) != {"sha256", "rows"}
        or not _is_sha(holdout["sha256"])
        or not isinstance(holdout["rows"], list)
    ):
        raise TrainingError("embedding holdout identity descriptor")
    try:
        holdout_identities = _identity_rows(cast(list[Mapping[str, Any]], holdout["rows"]))
    except (TypeError, ValueError) as error:
        raise TrainingError("embedding holdout identity descriptor") from error
    all_rows = [*cast(list[dict[str, Any]], train_dev["rows"]), *holdout_identities]
    identities = _identity_rows(all_rows)
    if _sha(_canonical(identities)) != receipt["accepted_rows_sha256"]:
        raise TrainingError("accepted packet row identity binding")
    _validate_lineage(all_rows)
    if accepted_packet is not None and (
        receipt["packet_json_sha256"] != accepted_packet.packet_json_sha256
        or receipt["accepted_train_dev_jsonl_sha256"]
        != accepted_packet.accepted_train_dev_jsonl_sha256
        or receipt["accepted_holdout_jsonl_sha256"] != accepted_packet.accepted_holdout_jsonl_sha256
        or receipt["holdout_identities_json_sha256"]
        != accepted_packet.holdout_identities_json_sha256
        or receipt["accepted_rows_sha256"] != accepted_packet.accepted_rows_sha256
        or identities != accepted_packet.identities
        or holdout_identities != accepted_packet.holdout_identities
        or holdout["sha256"] != accepted_packet.holdout_identities_json_sha256
    ):
        raise TrainingError("validated accepted packet binding")
    digests = manifest["tensor_digests"]
    if not isinstance(digests, dict) or set(digests) != {"train_dev"}:
        raise TrainingError("embedding tensor digests")
    for split in digests.values():
        if (
            not isinstance(split, dict)
            or set(split) != _EMBEDDING_TENSORS
            or any(not _is_sha(value) for value in split.values())
        ):
            raise TrainingError("embedding tensor digests")
    capsule = EmbeddingCapsule(
        path, manifest, receipt, _sha((path / "capsule-descriptor.json").read_bytes())
    )
    _validate_embedding_tensors(capsule, "train_dev")
    return capsule


def _validate_rows(rows: list[object], allowed_splits: set[str]) -> None:
    if not rows:
        raise TrainingError("embedding rows")
    seen_tasks: set[str] = set()
    for index, raw in enumerate(rows):
        if not isinstance(raw, dict) or set(raw) != {
            "task_id",
            "family_id",
            "pair_id",
            "split",
            "locale",
            "domain",
            "axes",
            "option_count",
            "gold_position",
            "row_index",
            "criterion_offset",
            "criterion_ids",
        }:
            raise TrainingError("embedding row identity")
        if (
            raw["split"] not in allowed_splits
            or raw["locale"] not in {"pt-BR", "en"}
            or not isinstance(raw["domain"], str)
            or not isinstance(raw["axes"], dict)
            or type(raw["option_count"]) is not int
            or not 2 <= raw["option_count"] <= 8
            or type(raw["gold_position"]) is not int
            or not 0 <= raw["gold_position"] < raw["option_count"]
            or raw["row_index"] != index
            or type(raw["criterion_offset"]) is not int
            or raw["criterion_offset"] < 0
            or not isinstance(raw["criterion_ids"], list)
            or len(raw["criterion_ids"]) != raw["option_count"]
            or not all(isinstance(item, str) for item in raw["criterion_ids"])
            or len(set(raw["criterion_ids"])) != raw["option_count"]
            or not isinstance(raw["task_id"], str)
            or not isinstance(raw["family_id"], str)
            or (raw["pair_id"] is not None and not isinstance(raw["pair_id"], str))
        ):
            raise TrainingError("embedding row identity")
        if raw["task_id"] in seen_tasks:
            raise TrainingError("embedding task identity")
        seen_tasks.add(raw["task_id"])


def _validate_lineage(rows: Sequence[Mapping[str, Any]]) -> None:
    """Permit same-split family members, but reject every split/pair leak."""

    task_ids: set[str] = set()
    family_splits: dict[str, str] = {}
    pairs: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        task_id = cast(str, row["task_id"])
        family_id = cast(str, row["family_id"])
        split = cast(str, row["split"])
        if task_id in task_ids:
            raise TrainingError("embedding task identity")
        task_ids.add(task_id)
        prior_split = family_splits.setdefault(family_id, split)
        if prior_split != split:
            raise TrainingError("cross-split family leakage")
        pair_id = row["pair_id"]
        if pair_id is not None:
            pairs.setdefault(cast(str, pair_id), []).append(row)
    for pair_id, members in pairs.items():
        if (
            len(members) > 2
            or {cast(str, row["family_id"]) for row in members} != {pair_id}
            or len({cast(str, row["split"]) for row in members}) != 1
            or len({cast(str, row["locale"]) for row in members}) != len(members)
        ):
            raise TrainingError("embedding pair lineage")
        if len(members) == 2 and {cast(str, row["locale"]) for row in members} != {"pt-BR", "en"}:
            raise TrainingError("embedding pair lineage")


def _ml() -> tuple[Any, Any, Any]:
    try:
        from importlib import import_module

        safetensors_torch = import_module("safetensors.torch")
        torch = import_module("torch")
    except ImportError as error:
        raise TrainingError("local-minilm extra is required") from error
    return torch, safetensors_torch.load, safetensors_torch.save


def _tensor_digest(value: Any) -> str:
    value = value.detach().to(device="cpu").contiguous()
    return _sha(
        str(value.dtype).encode("ascii") + _canonical(list(value.shape)) + value.numpy().tobytes()
    )


def _load_verified_minilm(snapshot: Path) -> LoadedEncoder:
    """Open only the reviewed, descriptor-verified local MiniLM snapshot."""

    candidate = get_candidate("multilingual-minilm-l12")
    if candidate.id != "multilingual-minilm-l12":
        raise TrainingError("reviewed MiniLM candidate")
    try:
        verified = VerifiedSnapshot.create(candidate, snapshot)
        loaded = load_encoder(candidate, snapshot, verified=verified)
    except (OSError, RuntimeError, ValueError) as error:
        raise TrainingError("verified MiniLM snapshot") from error
    if (
        loaded.candidate.id != candidate.id
        or loaded.candidate.revision != candidate.revision
        or loaded.snapshot.resolve() != snapshot.resolve()
    ):
        raise TrainingError("verified MiniLM snapshot")
    return loaded


def _runtime_abi(torch: Any, loaded: LoadedEncoder) -> str:
    """Record the exact local extraction ABI without a network/package lookup."""

    return _sha(
        _canonical(
            {
                "torch": str(torch.__version__),
                "model": type(loaded.model).__module__ + "." + type(loaded.model).__qualname__,
                "tokenizer": (
                    type(loaded.tokenizer).__module__ + "." + type(loaded.tokenizer).__qualname__
                ),
            }
        )
    )


def _select_device(torch: Any, device: str) -> Any:
    if device == "cpu":
        return torch.device("cpu")
    raise TrainingError("Phase 4E.2c extraction is CPU-only")


def _tokenize_exact(
    tokenizer: Any, texts: Sequence[str], limit: int, torch: Any
) -> tuple[Any, Any]:
    """Tokenize ordered texts in bounded batches and globally pad without truncation.

    Each tokenizer invocation stays within the workflow's criterion-batch
    limit.  Token rows are collected before tensor construction so the final
    right-padding width is the same global maximum that an unbatched
    derivation would produce; chunk-local widths cannot affect the sealed
    tensor shape or digest.
    """

    ids_rows: list[list[int]] = []
    for start in range(0, len(texts), _MAX_ENCODER_MICROBATCH):
        chunk = list(texts[start : start + _MAX_ENCODER_MICROBATCH])
        try:
            encoded = tokenizer(chunk, truncation=False, padding=False, add_special_tokens=True)
            ids = encoded.get("input_ids") if isinstance(encoded, Mapping) else None
        except (TypeError, ValueError) as error:
            raise TrainingError("verified tokenizer output") from error
        if (
            not isinstance(ids, list)
            or len(ids) != len(chunk)
            or not all(
                isinstance(row, list) and all(type(value) is int for value in row) for row in ids
            )
        ):
            raise TrainingError("verified tokenizer output")
        if any(not row or len(row) > limit for row in ids):
            raise TrainingError("embedding extraction would truncate")
        ids_rows.extend(ids)
    if not ids_rows:
        raise TrainingError("verified tokenizer output")
    width = max(len(row) for row in ids_rows)
    values = torch.zeros((len(ids_rows), width), dtype=torch.int64)
    masks = torch.zeros((len(ids_rows), width), dtype=torch.uint8)
    for index, row in enumerate(ids_rows):
        values[index, : len(row)] = torch.tensor(row, dtype=torch.int64)
        masks[index, : len(row)] = 1
    return values, masks


def _extract_bounded_embeddings(
    loaded: LoadedEncoder,
    texts: Sequence[str],
    limit: int,
    device: Any,
    torch: Any,
) -> tuple[Any, Any, Any]:
    """Derive canonical token tensors and embeddings in batches of at most 20.

    Tokenization performs a complete, globally padded derivation first.  Each
    model call then receives an ordered slice of those canonical tensors, and
    the float32 outputs are concatenated in the original text order.
    """

    token_ids, masks = _tokenize_exact(loaded.tokenizer, texts, limit, torch)
    embeddings = [
        _mean_pool(
            loaded,
            token_ids[start : start + _MAX_ENCODER_MICROBATCH],
            masks[start : start + _MAX_ENCODER_MICROBATCH],
            device,
            torch,
        )
        for start in range(0, token_ids.shape[0], _MAX_ENCODER_MICROBATCH)
    ]
    return token_ids, masks, torch.cat(embeddings, dim=0).to(dtype=torch.float32).contiguous()


def _mean_pool(loaded: LoadedEncoder, token_ids: Any, masks: Any, device: Any, torch: Any) -> Any:
    try:
        model = loaded.model.to(device)
        model.eval()
        with torch.inference_mode():
            output = model(
                input_ids=token_ids.to(device), attention_mask=masks.to(device), return_dict=True
            )
            hidden = getattr(output, "last_hidden_state", None)
            if hidden is None and isinstance(output, Mapping):
                hidden = output.get("last_hidden_state")
            pooled = (hidden * masks.to(device).unsqueeze(-1)).sum(dim=1) / masks.to(device).sum(
                dim=1, keepdim=True
            )
            result = pooled.to(device="cpu", dtype=torch.float32).contiguous()
    except (AttributeError, RuntimeError, TypeError, ValueError) as error:
        raise TrainingError("verified MiniLM embedding extraction") from error
    if (
        result.ndim != 2
        or tuple(result.shape[1:]) != (384,)
        or result.dtype != torch.float32
        or not bool(torch.isfinite(result).all().item())
    ):
        raise TrainingError("verified MiniLM embedding extraction")
    return result


def _packet_tasks(
    packet: Path,
    binding: AcceptedPacketBinding,
    *,
    parts: Sequence[Literal["train_dev", "holdout"]],
) -> tuple[dict[str, Any], ...]:
    """Read only the explicitly released accepted payload split(s)."""

    typed: list[dict[str, Any]] = []
    if not parts or len(set(parts)) != len(parts) or set(parts) - {"train_dev", "holdout"}:
        raise TrainingError("accepted packet source split")
    sources = {
        "train_dev": (
            "accepted-train-dev.jsonl",
            binding.accepted_train_dev_jsonl_sha256,
            binding.train_dev_identities,
        ),
        "holdout": (
            "accepted-holdout.jsonl",
            binding.accepted_holdout_jsonl_sha256,
            binding.holdout_identities,
        ),
    }
    for part in parts:
        name, expected_digest, expected_identities = sources[part]
        try:
            accepted_raw = (packet / name).read_bytes()
            records = [json.loads(line) for line in accepted_raw.splitlines() if line]
        except (OSError, ValueError) as error:
            raise TrainingError("accepted packet source text") from error
        if _sha(accepted_raw) != expected_digest or not all(
            isinstance(record, dict) for record in records
        ):
            raise TrainingError("accepted packet source digest binding")
        for record in cast(list[dict[str, Any]], records):
            try:
                task = UniversalTask.model_validate(
                    {key: record[key] for key in UniversalTask.model_fields}
                )
                split, pair_id, axes = record["split"], record["pair_id"], record["axes"]
            except (KeyError, ValueError) as error:
                raise TrainingError("accepted packet source text") from error
            if split not in _SPLITS or (pair_id is not None and not isinstance(pair_id, str)):
                raise TrainingError("accepted packet source text")
            value = task.model_dump(mode="json")
            value["split"] = split
            value["pair_id"] = pair_id
            value["axes"] = axes
            typed.append(value)
        if _identity_rows(typed[-len(records) :]) != expected_identities:
            raise TrainingError("accepted packet source identity binding")
    if parts == ("train_dev",) and any(row["split"] == "synthetic_holdout" for row in typed):
        raise TrainingError("accepted packet source identity binding")
    return tuple(typed)


def _derive_descriptor_bound_embeddings(
    packet: Path,
    binding: AcceptedPacketBinding,
    snapshot: Path,
    device_name: str,
    *,
    parts: Sequence[Literal["train_dev", "holdout"]] = ("train_dev",),
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Derive the requested capsule tensors from packet text and verified MiniLM."""

    if not parts or len(set(parts)) != len(parts) or set(parts) - {"train_dev", "holdout"}:
        raise TrainingError("descriptor-bound embedding parts")

    torch, _load, _save = _ml()
    loaded = _load_verified_minilm(snapshot)
    device = _select_device(torch, device_name)
    try:
        complete_sha256 = _sha((snapshot / "snapshot.complete.json").read_bytes())
    except OSError as error:
        raise TrainingError("verified MiniLM snapshot marker") from error
    tasks = _packet_tasks(packet, binding, parts=parts)
    sections: dict[str, dict[str, Any]] = {}
    for part, splits in (
        ("train_dev", {"synthetic_train", "synthetic_dev"}),
        ("holdout", {"synthetic_holdout"}),
    ):
        if part not in parts:
            continue
        selected = [task for task in tasks if task["split"] in splits]
        rendered = [
            render_task(
                UniversalTask.model_validate({key: task[key] for key in UniversalTask.model_fields})
            )
            for task in selected
        ]
        context_ids, context_masks, context_embeddings = _extract_bounded_embeddings(
            loaded,
            [item.context for item in rendered],
            MAX_CONTEXT_TOKENS,
            device,
            torch,
        )
        criterion_texts = [text for item in rendered for text in item.criteria]
        criterion_ids, criterion_masks, criterion_embeddings = _extract_bounded_embeddings(
            loaded, criterion_texts, MAX_CRITERION_TOKENS, device, torch
        )
        rows: list[dict[str, Any]] = []
        offset = 0
        for index, task in enumerate(selected):
            criteria = cast(list[dict[str, Any]], task["criteria"])
            gold = next(
                position
                for position, criterion in enumerate(criteria)
                if criterion["id"] == task["selected_criterion_id"]
            )
            rows.append(
                {
                    "task_id": task["task_id"],
                    "family_id": task["family_id"],
                    "pair_id": task["pair_id"],
                    "split": task["split"],
                    "locale": task["locale"],
                    "domain": task["domain"],
                    "axes": task.get("axes", {}),
                    "option_count": len(criteria),
                    "gold_position": gold,
                    "row_index": index,
                    "criterion_offset": offset,
                    "criterion_ids": [criterion["id"] for criterion in criteria],
                }
            )
            offset += len(criteria)
        tensors = {
            "context_embeddings": context_embeddings,
            "criterion_embeddings": criterion_embeddings,
            "context_token_ids": context_ids,
            "context_attention_masks": context_masks,
            "criterion_token_ids": criterion_ids,
            "criterion_attention_masks": criterion_masks,
        }
        sections[part] = {"rows": rows, "tensors": tensors}
    metadata = {
        "base_encoder": {
            "id": BASE_ENCODER_ID,
            "revision": BASE_ENCODER_REVISION,
            "snapshot_complete_sha256": complete_sha256,
            "frozen": True,
        },
        "extraction": {"device": device_name, "runtime_abi": _runtime_abi(torch, loaded)},
    }
    return sections, metadata


def extract_and_seal_embeddings(packet: Path, snapshot: Path, device: str, output: Path) -> Path:
    """Create the exact no-clobber, holdout-blind v3 embedding capsule.

    This is the only public extraction producer.  It validates the packet's
    train/dev and identity projection, derives train/dev tensors once, and
    never opens the holdout payload or creates a holdout tensor artifact.
    """

    if device != "cpu":
        raise TrainingError("Phase 4E.2c extraction is CPU-only")
    binding = validate_accepted_packet_binding(packet)
    sections, metadata = _derive_descriptor_bound_embeddings(
        packet, binding, snapshot, device, parts=("train_dev",)
    )
    train_dev = sections["train_dev"]
    torch, _load, save = _ml()
    del torch
    tensor_bytes = save(train_dev["tensors"])
    receipt = {
        "schema_version": "phase4e-accepted-packet-receipt.v4",
        "packet_json_sha256": binding.packet_json_sha256,
        "accepted_train_dev_jsonl_sha256": binding.accepted_train_dev_jsonl_sha256,
        "accepted_holdout_jsonl_sha256": binding.accepted_holdout_jsonl_sha256,
        "holdout_identities_json_sha256": binding.holdout_identities_json_sha256,
        "accepted_rows_sha256": binding.accepted_rows_sha256,
        "synthetic_only": True,
    }
    receipt_raw = _canonical(receipt) + b"\n"
    manifest: dict[str, Any] = {
        "schema_version": "phase4e-embedding-capsule.v3",
        "packet_receipt_sha256": _sha(receipt_raw),
        "base_encoder": metadata["base_encoder"],
        "rendering_revision": RENDERER_REVISION,
        "extraction": metadata["extraction"],
        "train_dev": {
            "file": "train-dev.safetensors",
            "sha256": _sha(tensor_bytes),
            "rows": train_dev["rows"],
        },
        "holdout_identities": {
            "sha256": binding.holdout_identities_json_sha256,
            "rows": list(binding.holdout_identities),
        },
        "tensor_digests": {
            "train_dev": {
                name: _tensor_digest(value) for name, value in train_dev["tensors"].items()
            }
        },
        "manifest_sha256": "",
    }
    manifest["manifest_sha256"] = _sha(
        _canonical({key: value for key, value in manifest.items() if key != "manifest_sha256"})
    )
    files = {
        "train-dev.safetensors": tensor_bytes,
        "embedding-manifest.json": _canonical(manifest) + b"\n",
        "accepted-packet-receipt.json": receipt_raw,
    }
    files["capsule-descriptor.json"] = _descriptor(files, kind="embedding")
    if set(files) != _CAPSULE_FILES:
        raise AssertionError("embedding capsule file set")
    return _write_capsule(output.parent, output.name, files)


def verify_descriptor_bound_embeddings(
    capsule: EmbeddingCapsule,
    packet: Path,
    binding: AcceptedPacketBinding,
    snapshot: Path,
    device: str,
) -> None:
    """Re-derive train/dev tensors before deterministic training may begin."""

    sections, metadata = _derive_descriptor_bound_embeddings(
        packet, binding, snapshot, device, parts=("train_dev",)
    )
    if (
        capsule.manifest["base_encoder"] != metadata["base_encoder"]
        or capsule.manifest["extraction"] != metadata["extraction"]
    ):
        raise TrainingError("descriptor-bound extraction binding")
    for part in ("train_dev",):
        section = cast(dict[str, Any], capsule.manifest[part])
        derived = sections[part]
        if section["rows"] != derived["rows"]:
            raise TrainingError("descriptor-bound task or criterion order")
        stored, _rows = _decode_embedding_tensors(capsule, part)
        for name in _EMBEDDING_TENSORS:
            if not bool((stored[name] == derived["tensors"][name]).all().item()):
                raise TrainingError("descriptor-bound embedding mismatch")


def derive_holdout_embeddings_in_memory(
    capsule: EmbeddingCapsule,
    packet: Path,
    binding: AcceptedPacketBinding,
    snapshot: Path,
    device: str,
    checkpoint_sha256: str,
    pre_holdout_gate_descriptor_sha256: str,
) -> _Rows:
    """Open holdout exactly after claim and return its transient tensors only."""

    validate_full_accepted_packet_binding(
        packet,
        binding,
        capsule,
        checkpoint_sha256,
        pre_holdout_gate_descriptor_sha256,
    )

    sections, metadata = _derive_descriptor_bound_embeddings(
        packet, binding, snapshot, device, parts=("holdout",)
    )
    if (
        capsule.manifest["base_encoder"] != metadata["base_encoder"]
        or capsule.manifest["extraction"] != metadata["extraction"]
    ):
        raise TrainingError("descriptor-bound extraction binding")
    derived = sections["holdout"]
    sealed = cast(dict[str, Any], capsule.manifest["holdout_identities"])
    if _identity_rows(cast(list[Mapping[str, Any]], sealed["rows"])) != _identity_rows(
        cast(list[Mapping[str, Any]], derived["rows"])
    ):
        raise TrainingError("descriptor-bound task or criterion order")
    return _rows_from_tensors(derived["rows"], derived["tensors"])


def verify_holdout_descriptor_bound_embeddings(
    capsule: EmbeddingCapsule,
    packet: Path,
    binding: AcceptedPacketBinding,
    snapshot: Path,
    device: str,
    checkpoint_sha256: str,
    pre_holdout_gate_descriptor_sha256: str,
) -> None:
    """Compatibility verifier; never persists or compares a holdout capsule."""

    derive_holdout_embeddings_in_memory(
        capsule,
        packet,
        binding,
        snapshot,
        device,
        checkpoint_sha256,
        pre_holdout_gate_descriptor_sha256,
    )


def _decode_embedding_tensors(
    capsule: EmbeddingCapsule, part: Literal["train_dev"]
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    """Decode one sealed split without scoring or constructing a ranker."""

    _torch, load, _save = _ml()
    section = cast(dict[str, Any], capsule.manifest[part])
    try:
        tensors = load((capsule.path / cast(str, section["file"])).read_bytes())
    except Exception as error:
        raise TrainingError("embedding safetensors") from error
    if not isinstance(tensors, dict):
        raise TrainingError("embedding safetensors")
    return cast(dict[str, Any], tensors), tuple(cast(list[dict[str, Any]], section["rows"]))


def _validate_embedding_tensor_data(
    capsule: EmbeddingCapsule,
    part: Literal["train_dev"],
    tensors: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    """Fail closed on tensor structure, alignment, values, and exact digests."""

    torch, _load, _save = _ml()
    expected = _EMBEDDING_TENSORS
    if set(tensors) != expected or not all(torch.is_tensor(value) for value in tensors.values()):
        raise TrainingError("embedding tensor set")
    context, criteria = tensors["context_embeddings"], tensors["criterion_embeddings"]
    context_ids, context_attention = (
        tensors["context_token_ids"],
        tensors["context_attention_masks"],
    )
    criterion_ids, criterion_attention = (
        tensors["criterion_token_ids"],
        tensors["criterion_attention_masks"],
    )
    if (
        context.dtype != torch.float32
        or criteria.dtype != torch.float32
        or context.ndim != 2
        or criteria.ndim != 2
        or tuple(context.shape[1:]) != (384,)
        or tuple(criteria.shape[1:]) != (384,)
        or not bool(torch.isfinite(context).all().item())
        or not bool(torch.isfinite(criteria).all().item())
        or context_ids.dtype != torch.int64
        or criterion_ids.dtype != torch.int64
        or context_attention.dtype not in {torch.uint8, torch.int64}
        or criterion_attention.dtype not in {torch.uint8, torch.int64}
        or context_ids.ndim != 2
        or criterion_ids.ndim != 2
        or context_attention.ndim != 2
        or criterion_attention.ndim != 2
        or context_ids.shape != context_attention.shape
        or criterion_ids.shape != criterion_attention.shape
        or context_ids.shape[0] != len(rows)
        or context_ids.shape[1] < 1
        or context_ids.shape[1] > MAX_CONTEXT_TOKENS
        or criterion_ids.shape[1] < 1
        or criterion_ids.shape[1] > MAX_CRITERION_TOKENS
    ):
        raise TrainingError("embedding tensor contract")
    expected_digests = cast(
        dict[str, str], cast(dict[str, Any], capsule.manifest["tensor_digests"])[part]
    )
    for name, value in tensors.items():
        if _tensor_digest(value) != expected_digests[name]:
            raise TrainingError("embedding tensor digest")
    if context.shape[0] != len(rows):
        raise TrainingError("embedding row/tensor alignment")
    offset = 0
    for row in rows:
        if row["criterion_offset"] != offset:
            raise TrainingError("embedding row/tensor alignment")
        offset += cast(int, row["option_count"])
    if offset != criteria.shape[0]:
        raise TrainingError("embedding row/tensor alignment")
    if criterion_ids.shape[0] != offset:
        raise TrainingError("embedding row/tensor alignment")
    for _ids, masks, limit in (
        (context_ids, context_attention, MAX_CONTEXT_TOKENS),
        (criterion_ids, criterion_attention, MAX_CRITERION_TOKENS),
    ):
        sums = masks.to(dtype=torch.int64).sum(dim=1)
        if not bool(((sums >= 1) & (sums <= limit)).all().item()) or not bool(
            ((masks == 0) | (masks == 1)).all().item()
        ):
            raise TrainingError("embedding token mask contract")


def _validate_embedding_tensors(capsule: EmbeddingCapsule, part: Literal["train_dev"]) -> None:
    """Preflight one split without returning rows to training or selection."""

    tensors, rows = _decode_embedding_tensors(capsule, part)
    _validate_embedding_tensor_data(capsule, part, tensors, rows)


@dataclass(frozen=True)
class _Rows:
    rows: tuple[dict[str, Any], ...]
    context: Any
    criteria: Any


def _load_rows(capsule: EmbeddingCapsule, part: Literal["train_dev"]) -> _Rows:
    tensors, rows = _decode_embedding_tensors(capsule, part)
    _validate_embedding_tensor_data(capsule, part, tensors, rows)
    context, criteria = tensors["context_embeddings"], tensors["criterion_embeddings"]
    return _Rows(rows, context, criteria)


def _rows_from_tensors(rows: Sequence[dict[str, Any]], tensors: Mapping[str, Any]) -> _Rows:
    """Validate an unpersisted extraction section before it reaches scoring."""

    torch, _load, _save = _ml()
    expected = _EMBEDDING_TENSORS
    if set(tensors) != expected or not all(torch.is_tensor(value) for value in tensors.values()):
        raise TrainingError("in-memory holdout tensor set")
    context = tensors["context_embeddings"]
    criteria = tensors["criterion_embeddings"]
    if (
        context.dtype != torch.float32
        or criteria.dtype != torch.float32
        or tuple(context.shape) != (len(rows), 384)
        or criteria.ndim != 2
        or tuple(criteria.shape[1:]) != (384,)
        or not bool(torch.isfinite(context).all().item())
        or not bool(torch.isfinite(criteria).all().item())
    ):
        raise TrainingError("in-memory holdout tensor contract")
    offset = 0
    for index, row in enumerate(rows):
        if row["row_index"] != index or row["criterion_offset"] != offset:
            raise TrainingError("in-memory holdout row alignment")
        offset += cast(int, row["option_count"])
    if criteria.shape[0] != offset:
        raise TrainingError("in-memory holdout row alignment")
    return _Rows(tuple(rows), context, criteria)


def _metric(rows: _Rows, logits: Any) -> dict[str, Any]:
    predictions = [int(item.argmax().item()) for item in logits]
    targets = [int(row["gold_position"]) for row in rows.rows]
    correct = sum(left == right for left, right in zip(predictions, targets, strict=True))
    cells: dict[str, float] = {}
    for locale, count in _CELLS:
        matched = [
            index
            for index, row in enumerate(rows.rows)
            if row["locale"] == locale and row["option_count"] == count
        ]
        if not matched:
            raise TrainingError("stratified metric missing cell")
        cells[f"{locale}:{count}"] = sum(
            predictions[index] == targets[index] for index in matched
        ) / len(matched)
    return {
        "accuracy": correct / len(rows.rows),
        "stratified_macro_accuracy": sum(cells.values()) / len(cells),
        "cells": cells,
        "predictions": predictions,
    }


def _logits(model: Any, rows: _Rows, indices: Sequence[int] | None = None) -> list[Any]:
    torch, _load, _save = _ml()
    selected = list(range(len(rows.rows))) if indices is None else list(indices)
    values: list[Any] = []
    for index in selected:
        row = rows.rows[index]
        offset, count = row["criterion_offset"], row["option_count"]
        context_index = row["row_index"]
        context = model.project_context(rows.context[context_index : context_index + 1])
        criteria = model.project_criterion(rows.criteria[offset : offset + count])
        context = torch.nn.functional.normalize(context, dim=-1)
        criteria = torch.nn.functional.normalize(criteria, dim=-1)
        logits = (context @ criteria.T).reshape(-1) * model.log_scale.exp()
        if not bool(torch.isfinite(logits).all().item()):
            raise TrainingError("non-finite ranker logits")
        values.append(logits)
    return values


def _new_model(torch: Any) -> Any:
    class Ranker(torch.nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.context_weight = torch.nn.Parameter(torch.empty(192, 384))
            self.context_bias = torch.nn.Parameter(torch.zeros(192))
            self.context_norm_weight = torch.nn.Parameter(torch.ones(192))
            self.context_norm_bias = torch.nn.Parameter(torch.zeros(192))
            self.criterion_weight = torch.nn.Parameter(torch.empty(192, 384))
            self.criterion_bias = torch.nn.Parameter(torch.zeros(192))
            self.criterion_norm_weight = torch.nn.Parameter(torch.ones(192))
            self.criterion_norm_bias = torch.nn.Parameter(torch.zeros(192))
            self.log_scale = torch.nn.Parameter(torch.zeros(1))
            torch.nn.init.xavier_uniform_(self.context_weight)
            torch.nn.init.xavier_uniform_(self.criterion_weight)

        def _project(
            self, value: Any, weight: Any, bias: Any, norm_weight: Any, norm_bias: Any
        ) -> Any:
            return torch.nn.functional.layer_norm(
                torch.nn.functional.gelu(value @ weight.T + bias), (192,), norm_weight, norm_bias
            )

        def project_context(self, value: Any) -> Any:
            return self._project(
                value,
                self.context_weight,
                self.context_bias,
                self.context_norm_weight,
                self.context_norm_bias,
            )

        def project_criterion(self, value: Any) -> Any:
            return self._project(
                value,
                self.criterion_weight,
                self.criterion_bias,
                self.criterion_norm_weight,
                self.criterion_norm_bias,
            )

        def checkpoint(self) -> dict[str, Any]:
            return {
                "context_projection.weight": self.context_weight.detach().cpu().contiguous(),
                "context_projection.bias": self.context_bias.detach().cpu().contiguous(),
                "context_projection.layernorm.weight": self.context_norm_weight.detach()
                .cpu()
                .contiguous(),
                "context_projection.layernorm.bias": self.context_norm_bias.detach()
                .cpu()
                .contiguous(),
                "criterion_projection.weight": self.criterion_weight.detach().cpu().contiguous(),
                "criterion_projection.bias": self.criterion_bias.detach().cpu().contiguous(),
                "criterion_projection.layernorm.weight": self.criterion_norm_weight.detach()
                .cpu()
                .contiguous(),
                "criterion_projection.layernorm.bias": self.criterion_norm_bias.detach()
                .cpu()
                .contiguous(),
                "log_scale": self.log_scale.detach().cpu().clamp(-4.0, 4.0).contiguous(),
            }

    return Ranker()


def _run_once(
    train: _Rows, dev: _Rows, policy: Mapping[str, Any]
) -> tuple[bytes, list[dict[str, Any]], dict[str, Any]]:
    torch, _load, save = _ml()
    seed, threads = int(policy["seed"]), int(policy["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    model = _new_model(torch)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(policy["learning_rate"]),
        weight_decay=float(policy["weight_decay"]),
    )
    best_metric, best_epoch, stale, best_state = -1.0, 0, 0, None
    ledger: list[dict[str, Any]] = []
    train_indices = list(range(len(train.rows)))
    for epoch in range(1, int(policy["maximum_epochs"]) + 1):
        generator = torch.Generator(device="cpu").manual_seed(seed + epoch)
        ordered = torch.randperm(len(train_indices), generator=generator).tolist()
        model.train()
        losses: list[float] = []
        for start in range(0, len(ordered), int(policy["batch_size"])):
            batch = ordered[start : start + int(policy["batch_size"])]
            optimizer.zero_grad(set_to_none=True)
            scored = _logits(model, train, batch)
            maximum_options = max(value.shape[0] for value in scored)
            # Variable-option batches are padded only with -inf, so padding
            # cannot become either a scored choice or a loss target.
            padded = torch.full((len(batch), maximum_options), float("-inf"))
            for offset, value in enumerate(scored):
                padded[offset, : value.shape[0]] = value
            targets = torch.tensor(
                [train.rows[index]["gold_position"] for index in batch], dtype=torch.long
            )
            loss = torch.nn.functional.cross_entropy(padded, targets)
            if not bool(torch.isfinite(loss).item()):
                raise TrainingError("non-finite training loss")
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                model.log_scale.clamp_(-4.0, 4.0)
            losses.append(float(loss.detach().item()))
        model.eval()
        with torch.inference_mode():
            dev_metric = _metric(dev, _logits(model, dev))
        score = float(dev_metric["stratified_macro_accuracy"])
        ledger.append({"epoch": epoch, "loss": sum(losses) / len(losses), "dev": dev_metric})
        if score > best_metric:
            best_metric, best_epoch, stale = score, epoch, 0
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.checkpoint().items()
            }
        else:
            stale += 1
            if stale >= int(policy["early_stopping_patience"]):
                break
    if best_state is None:
        raise TrainingError("training selection")
    verify_checkpoint_tensors(best_state)
    return save(best_state), ledger, {"epoch": best_epoch, "dev": ledger[best_epoch - 1]["dev"]}


def _baseline(rows: _Rows, *, random_projection: bool, seed: int) -> dict[str, Any]:
    torch, _load, _save = _ml()
    context, criteria = rows.context, rows.criteria
    if random_projection:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        left, right = (
            torch.randn((192, 384), generator=generator),
            torch.randn((192, 384), generator=generator),
        )
        context, criteria = context @ left.T, criteria @ right.T
    values: list[Any] = []
    for row in rows.rows:
        candidate = criteria[
            row["criterion_offset"] : row["criterion_offset"] + row["option_count"]
        ]
        values.append(
            (
                torch.nn.functional.normalize(
                    context[row["row_index"] : row["row_index"] + 1], dim=-1
                )
                @ torch.nn.functional.normalize(candidate, dim=-1).T
            ).reshape(-1)
        )
    return _metric(rows, values)


def _validate_stratified_metric(value: object, *, label: str) -> dict[str, Any]:
    """Accept only the fixed 14-cell macro metric used for selection and baselines."""

    if not isinstance(value, dict) or set(value) != {
        "accuracy",
        "stratified_macro_accuracy",
        "cells",
        "predictions",
    }:
        raise TrainingError(f"{label} metric")
    cells = value["cells"]
    if (
        not isinstance(cells, dict)
        or set(cells) != {f"{locale}:{count}" for locale, count in _CELLS}
        or not all(isinstance(score, (int, float)) for score in cells.values())
        or not isinstance(value["accuracy"], (int, float))
        or not isinstance(value["stratified_macro_accuracy"], (int, float))
        or not isinstance(value["predictions"], list)
        or not all(type(prediction) is int for prediction in value["predictions"])
    ):
        raise TrainingError(f"{label} metric")
    macro = sum(float(cells[f"{locale}:{count}"]) for locale, count in _CELLS) / len(_CELLS)
    if float(value["stratified_macro_accuracy"]) != macro:
        raise TrainingError(f"{label} macro metric")
    return cast(dict[str, Any], value)


def _holdout_identity_baselines(capsule: EmbeddingCapsule) -> dict[str, Any]:
    """Freeze holdout-only baselines from sealed identity rows, never tensors."""

    rows = cast(
        list[dict[str, Any]], cast(dict[str, Any], capsule.manifest["holdout_identities"])["rows"]
    )
    task_positions = [
        {
            "task_id": row["task_id"],
            "gold_position": row["gold_position"],
            "option_count": row["option_count"],
        }
        for row in sorted(rows, key=lambda row: cast(str, row["task_id"]))
    ]
    if not task_positions or len({row["task_id"] for row in task_positions}) != len(task_positions):
        raise TrainingError("holdout sealed identities")
    positions = [cast(int, row["gold_position"]) for row in task_positions]
    return {
        "task_ids": [cast(str, row["task_id"]) for row in task_positions],
        "task_positions": task_positions,
        "task_positions_sha256": _sha(_canonical(task_positions)),
        "denominator": len(task_positions),
        "expected_random_accuracy": sum(
            1 / cast(int, row["option_count"]) for row in task_positions
        )
        / len(task_positions),
        "most_frequent_gold_position_accuracy": max(
            positions.count(position) for position in set(positions)
        )
        / len(positions),
    }


def _success_thresholds(policy: Mapping[str, Any]) -> dict[str, Any]:
    improvement = policy.get("dev_improvement_over_baseline")
    if not isinstance(improvement, (int, float)):
        raise TrainingError("training policy threshold")
    return {
        "dev_absolute_improvement": float(improvement),
        "dev_formula": (
            "selected_dev_stratified_macro_accuracy >= "
            "baseline_stratified_macro_accuracy + dev_absolute_improvement"
        ),
        "holdout_overall_minimum": 0.45,
        "holdout_expected_random_addend": 0.20,
        "holdout_most_frequent_position_addend": 0.15,
        "holdout_overall_formula": (
            "holdout_accuracy >= max(holdout_overall_minimum, "
            "expected_random_accuracy + holdout_expected_random_addend, "
            "most_frequent_gold_position_accuracy + holdout_most_frequent_position_addend)"
        ),
        "locale_minimum_accuracy": {"pt-BR": 0.40, "en": 0.35},
        "option_count_expected_random_accuracy": {
            str(count): 1 / count for count in _REQUIRED_OPTION_COUNTS
        },
        "option_count_formula": "option_count_accuracy > 1 / option_count",
    }


def _training_code_sources() -> dict[str, str]:
    """Return the closed, deterministic dependency ledger for a training run."""

    root = Path(__file__).parents[1]
    names = (
        "benchmarks/saracura_universal_training.py",
        "benchmarks/saracura_universal_corpus.py",
        "benchmarks/saracura_universal_policy.py",
        "benchmarks/encoder_loader.py",
        "benchmarks/encoder_registry.py",
        "benchmarks/io.py",
        "benchmarks/manifests/phase4e-saracura-universal-policy.v1.json",
        "src/saracura/serialization.py",
        "src/saracura/universal/checkpoint.py",
        "src/saracura/universal/rendering.py",
        "src/saracura/universal/tasks.py",
    )
    try:
        return {name: _sha((root / name).read_bytes()) for name in names}
    except OSError as error:
        raise TrainingError("training code source") from error


def _training_code_digest() -> str:
    return _sha(_canonical(_training_code_sources()))


def _pre_holdout_gate_descriptor(
    capsule: EmbeddingCapsule,
    accepted_packet: AcceptedPacketBinding,
    policy_manifest: Mapping[str, Any],
    selected: Mapping[str, Any],
    cosine: Mapping[str, Any],
    random_projection: Mapping[str, Any],
    checkpoint_sha256: str,
    code_sha256: str,
) -> bytes:
    """Build every threshold and baseline before any evaluative holdout read."""

    if not _is_sha(checkpoint_sha256) or not _is_sha(code_sha256):
        raise TrainingError("pre-holdout checkpoint or code digest")
    policy = policy_manifest.get("training")
    if not isinstance(policy, dict) or policy != validate_phase4e_policy()["training"]:
        raise TrainingError("pre-holdout training policy")
    selected_value = dict(selected)
    if set(selected_value) != {"epoch", "dev"} or type(selected_value["epoch"]) is not int:
        raise TrainingError("pre-holdout selection")
    _validate_stratified_metric(selected_value["dev"], label="selected dev")
    _validate_stratified_metric(cosine, label="untrained cosine")
    _validate_stratified_metric(random_projection, label="random projection")
    value: dict[str, Any] = {
        "schema_version": _PRE_HOLDOUT_GATE_SCHEMA,
        "checkpoint_sha256": checkpoint_sha256,
        "embedding_descriptor_sha256": capsule.descriptor_sha256,
        "accepted_packet": {
            "packet_json_sha256": accepted_packet.packet_json_sha256,
            "accepted_train_dev_jsonl_sha256": accepted_packet.accepted_train_dev_jsonl_sha256,
            "accepted_holdout_jsonl_sha256": accepted_packet.accepted_holdout_jsonl_sha256,
            "holdout_identities_json_sha256": accepted_packet.holdout_identities_json_sha256,
            "accepted_rows_sha256": accepted_packet.accepted_rows_sha256,
            "packet_receipt_sha256": capsule.manifest["packet_receipt_sha256"],
        },
        "policy_sha256": _sha(_canonical(policy_manifest)),
        "training": policy,
        "selected": selected_value,
        "dev_baselines": {
            "untrained_cosine": dict(cosine),
            "random_projection": dict(random_projection),
        },
        "holdout_identity_baselines": _holdout_identity_baselines(capsule),
        "success_thresholds": _success_thresholds(policy),
        "required_buckets": {
            "locales": list(_REQUIRED_LOCALES),
            "option_counts": list(_REQUIRED_OPTION_COUNTS),
            "stratified_cells": [f"{locale}:{count}" for locale, count in _CELLS],
        },
        "zero_invalid_invariants": _ZERO_INVALID_INVARIANTS,
        "code_sha256": code_sha256,
        "descriptor_sha256": "",
    }
    value["descriptor_sha256"] = _sha(
        _canonical({key: item for key, item in value.items() if key != "descriptor_sha256"})
    )
    return _canonical(value) + b"\n"


def _validate_pre_holdout_gate_descriptor(raw: bytes) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except ValueError as error:
        raise TrainingError("pre-holdout gate descriptor") from error
    expected = {
        "schema_version",
        "checkpoint_sha256",
        "embedding_descriptor_sha256",
        "accepted_packet",
        "policy_sha256",
        "training",
        "selected",
        "dev_baselines",
        "holdout_identity_baselines",
        "success_thresholds",
        "required_buckets",
        "zero_invalid_invariants",
        "code_sha256",
        "descriptor_sha256",
    }
    if (
        not isinstance(value, dict)
        or set(value) != expected
        or value["schema_version"] != _PRE_HOLDOUT_GATE_SCHEMA
    ):
        raise TrainingError("pre-holdout gate descriptor")
    if (
        not _is_sha(value["checkpoint_sha256"])
        or not _is_sha(value["embedding_descriptor_sha256"])
        or not _is_sha(value["policy_sha256"])
        or not _is_sha(value["code_sha256"])
        or not _is_sha(value["descriptor_sha256"])
        or value["descriptor_sha256"]
        != _sha(
            _canonical({key: item for key, item in value.items() if key != "descriptor_sha256"})
        )
    ):
        raise TrainingError("pre-holdout gate digest")
    accepted = value["accepted_packet"]
    if (
        not isinstance(accepted, dict)
        or set(accepted)
        != {
            "packet_json_sha256",
            "accepted_train_dev_jsonl_sha256",
            "accepted_holdout_jsonl_sha256",
            "holdout_identities_json_sha256",
            "accepted_rows_sha256",
            "packet_receipt_sha256",
        }
        or not all(_is_sha(item) for item in accepted.values())
    ):
        raise TrainingError("pre-holdout accepted packet binding")
    if value["training"] != validate_phase4e_policy()["training"] or value["policy_sha256"] != _sha(
        _canonical(validate_phase4e_policy())
    ):
        raise TrainingError("pre-holdout training policy")
    selected = value["selected"]
    if (
        not isinstance(selected, dict)
        or set(selected) != {"epoch", "dev"}
        or type(selected["epoch"]) is not int
    ):
        raise TrainingError("pre-holdout selection")
    _validate_stratified_metric(selected["dev"], label="selected dev")
    baselines = value["dev_baselines"]
    if not isinstance(baselines, dict) or set(baselines) != {
        "untrained_cosine",
        "random_projection",
    }:
        raise TrainingError("pre-holdout dev baselines")
    for label, metric in baselines.items():
        _validate_stratified_metric(metric, label=label)
    identity = value["holdout_identity_baselines"]
    if not isinstance(identity, dict) or set(identity) != {
        "task_ids",
        "task_positions",
        "task_positions_sha256",
        "denominator",
        "expected_random_accuracy",
        "most_frequent_gold_position_accuracy",
    }:
        raise TrainingError("pre-holdout identity baselines")
    task_positions = identity["task_positions"]
    if (
        not isinstance(task_positions, list)
        or not task_positions
        or identity["task_ids"]
        != [item.get("task_id") for item in task_positions if isinstance(item, dict)]
        or identity["denominator"] != len(task_positions)
        or identity["task_positions_sha256"] != _sha(_canonical(task_positions))
        or not _is_sha(identity["task_positions_sha256"])
    ):
        raise TrainingError("pre-holdout identity baselines")
    if task_positions != sorted(
        task_positions, key=lambda item: item["task_id"] if isinstance(item, dict) else ""
    ):
        raise TrainingError("pre-holdout identity ordering")
    try:
        expected_random = sum(1 / item["option_count"] for item in task_positions) / len(
            task_positions
        )
        positions = [item["gold_position"] for item in task_positions]
        position_baseline = max(positions.count(position) for position in set(positions)) / len(
            positions
        )
    except (KeyError, TypeError, ZeroDivisionError) as error:
        raise TrainingError("pre-holdout identity baselines") from error
    if (
        any(
            not isinstance(item, dict)
            or set(item) != {"task_id", "gold_position", "option_count"}
            or not isinstance(item["task_id"], str)
            or type(item["option_count"]) is not int
            or not 2 <= item["option_count"] <= 8
            or type(item["gold_position"]) is not int
            or not 0 <= item["gold_position"] < item["option_count"]
            for item in task_positions
        )
        or len(set(identity["task_ids"])) != len(task_positions)
        or identity["expected_random_accuracy"] != expected_random
        or identity["most_frequent_gold_position_accuracy"] != position_baseline
    ):
        raise TrainingError("pre-holdout identity baselines")
    if value["success_thresholds"] != _success_thresholds(
        cast(Mapping[str, Any], value["training"])
    ):
        raise TrainingError("pre-holdout success thresholds")
    if (
        value["required_buckets"]
        != {
            "locales": list(_REQUIRED_LOCALES),
            "option_counts": list(_REQUIRED_OPTION_COUNTS),
            "stratified_cells": [f"{locale}:{count}" for locale, count in _CELLS],
        }
        or value["zero_invalid_invariants"] != _ZERO_INVALID_INVARIANTS
    ):
        raise TrainingError("pre-holdout gate invariants")
    return cast(dict[str, Any], value)


def _freeze_pre_holdout_gate(parent: Path, raw: bytes) -> tuple[dict[str, Any], str]:
    """Atomically persist the exact gate before the irreversible holdout claim."""

    descriptor = _validate_pre_holdout_gate_descriptor(raw)
    digest = _sha(raw)
    parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    target = parent / f".phase4e-pre-holdout-{digest}.json"
    try:
        atomic_create(target, raw)
        os.chmod(target, 0o600)
    except FileExistsError as error:
        raise TrainingError("pre-holdout gate was already frozen") from error
    return descriptor, digest


def _dev_improvement_gates(gate: Mapping[str, Any]) -> dict[str, bool]:
    """Evaluate both frozen dev gates before any claim or holdout access."""

    thresholds = cast(Mapping[str, Any], gate["success_thresholds"])
    selected = cast(Mapping[str, Any], gate["selected"])
    baselines = cast(Mapping[str, Any], gate["dev_baselines"])
    value = float(cast(Mapping[str, Any], selected["dev"])["stratified_macro_accuracy"])
    margin = float(thresholds["dev_absolute_improvement"])
    return {
        "dev_over_cosine": value
        >= float(
            cast(Mapping[str, Any], baselines["untrained_cosine"])["stratified_macro_accuracy"]
        )
        + margin,
        "dev_over_random_projection": value
        >= float(
            cast(Mapping[str, Any], baselines["random_projection"])["stratified_macro_accuracy"]
        )
        + margin,
    }


def _holdout_gates(
    rows: _Rows,
    metric: Mapping[str, Any],
    gate: Mapping[str, Any],
    gate_descriptor_sha256: str,
) -> dict[str, Any]:
    if not _is_sha(gate_descriptor_sha256):
        raise TrainingError("pre-holdout gate descriptor digest")
    thresholds = cast(dict[str, Any], gate["success_thresholds"])
    identity_baselines = cast(dict[str, Any], gate["holdout_identity_baselines"])
    by_locale = {
        locale: [index for index, row in enumerate(rows.rows) if row["locale"] == locale]
        for locale in _REQUIRED_LOCALES
    }
    buckets = {
        str(count): [index for index, row in enumerate(rows.rows) if row["option_count"] == count]
        for count in _REQUIRED_OPTION_COUNTS
    }
    if any(not indexes for indexes in by_locale.values()) or any(
        not indexes for indexes in buckets.values()
    ):
        raise TrainingError("holdout required buckets")
    predictions = cast(list[int], metric["predictions"])
    locale_scores = {
        locale: sum(predictions[index] == rows.rows[index]["gold_position"] for index in indexes)
        / len(indexes)
        for locale, indexes in by_locale.items()
    }
    bucket_scores = {
        count: sum(predictions[index] == rows.rows[index]["gold_position"] for index in indexes)
        / len(indexes)
        for count, indexes in buckets.items()
    }
    dev = float(cast(Mapping[str, Any], gate["selected"])["dev"]["stratified_macro_accuracy"])
    cosine = cast(
        Mapping[str, Any], cast(Mapping[str, Any], gate["dev_baselines"])["untrained_cosine"]
    )
    random_projection = cast(
        Mapping[str, Any], cast(Mapping[str, Any], gate["dev_baselines"])["random_projection"]
    )
    expected_random = float(identity_baselines["expected_random_accuracy"])
    position_baseline = float(identity_baselines["most_frequent_gold_position_accuracy"])
    locale_minima = cast(Mapping[str, Any], thresholds["locale_minimum_accuracy"])
    gates = {
        "dev_over_cosine": dev
        >= float(cosine["stratified_macro_accuracy"])
        + float(thresholds["dev_absolute_improvement"]),
        "dev_over_random_projection": dev
        >= float(random_projection["stratified_macro_accuracy"])
        + float(thresholds["dev_absolute_improvement"]),
        "holdout_overall": float(metric["accuracy"])
        >= max(
            float(thresholds["holdout_overall_minimum"]),
            expected_random + float(thresholds["holdout_expected_random_addend"]),
            position_baseline + float(thresholds["holdout_most_frequent_position_addend"]),
        ),
        "holdout_ptbr": locale_scores["pt-BR"] >= float(locale_minima["pt-BR"]),
        "holdout_en": locale_scores["en"] >= float(locale_minima["en"]),
        "option_counts": all(
            bucket_scores[count]
            > float(
                cast(Mapping[str, Any], thresholds["option_count_expected_random_accuracy"])[count]
            )
            for count in bucket_scores
        ),
    }
    return {
        "pre_holdout_gate_descriptor_sha256": gate_descriptor_sha256,
        "gates": gates,
        "passed": all(gates.values()),
        "precomputed_holdout_identity_baselines": identity_baselines,
        "success_thresholds": thresholds,
        "zero_invalid_invariants": gate["zero_invalid_invariants"],
        "locale_accuracy": locale_scores,
        "option_count_accuracy": bucket_scores,
        "metric": metric,
    }


def _write_capsule(parent: Path, name: str, files: dict[str, bytes]) -> Path:
    target = parent / name
    parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError("training capsule already exists")
    staged = Path(tempfile.mkdtemp(prefix=f".{name}.", dir=parent))
    os.chmod(staged, 0o700)
    try:
        for filename, raw in files.items():
            atomic_create(staged / filename, raw)
            os.chmod(staged / filename, 0o600)
        _publish_packet_create_if_absent(staged, target)
    except BaseException:
        for child in staged.glob("*"):
            child.unlink()
        staged.rmdir()
        raise
    return target


def _holdout_release_registry_directory() -> Path:
    """Return Saracura's canonical per-user holdout-release registry.

    This has no production configuration surface: it is derived only through
    ``platformdirs``.  Tests may monkeypatch this internal function to isolate
    their per-user state without making the registry caller controlled.
    """

    try:
        from importlib import import_module

        user_state_path = import_module("platformdirs").user_state_path
    except ImportError as error:
        raise TrainingError("local-minilm extra is required") from error
    return Path(user_state_path("saracura", appauthor=False)) / "phase4e" / "holdout-releases"


def _holdout_release_claim_key(
    packet_json_sha256: str, embedding_descriptor_sha256: str = ""
) -> str:
    """Use the packet manifest digest alone as the irreversible release key.

    ``embedding_descriptor_sha256`` remains an ignored compatibility argument
    for checkout-only tests; it must never alter the release identity.
    """

    del embedding_descriptor_sha256
    if not _is_sha(packet_json_sha256):
        raise TrainingError("holdout release identity")
    return packet_json_sha256


def _holdout_release_payload(
    accepted_packet: AcceptedPacketBinding,
    capsule: EmbeddingCapsule,
    checkpoint_sha256: str,
    pre_holdout_gate_descriptor_sha256: str,
) -> bytes:
    """Create the immutable record that authorizes one holdout observation."""

    if (
        not _is_sha(accepted_packet.packet_json_sha256)
        or not _is_sha(capsule.descriptor_sha256)
        or not _is_sha(checkpoint_sha256)
        or not _is_sha(pre_holdout_gate_descriptor_sha256)
    ):
        raise TrainingError("holdout release binding")
    return (
        _canonical(
            {
                "schema_version": _HOLDOUT_RELEASE_SCHEMA,
                "packet_json_sha256": accepted_packet.packet_json_sha256,
                "embedding_descriptor_sha256": capsule.descriptor_sha256,
                "checkpoint_sha256": checkpoint_sha256,
                "pre_holdout_gate_descriptor_sha256": pre_holdout_gate_descriptor_sha256,
            }
        )
        + b"\n"
    )


def _claim_holdout_once(
    accepted_packet: AcceptedPacketBinding,
    capsule: EmbeddingCapsule,
    checkpoint_sha256: str,
    pre_holdout_gate_descriptor_sha256: str,
) -> None:
    """Irreversibly bind one holdout release to all frozen pre-holdout bytes."""

    registry = _holdout_release_registry_directory()
    registry.mkdir(mode=0o700, parents=True, exist_ok=True)
    key = _holdout_release_claim_key(accepted_packet.packet_json_sha256)
    claim = registry / f"{key}.json"
    payload = _holdout_release_payload(
        accepted_packet, capsule, checkpoint_sha256, pre_holdout_gate_descriptor_sha256
    )
    try:
        atomic_create(claim, payload)
        os.chmod(claim, 0o600)
    except FileExistsError as error:
        # A crash after the release claim must remain conservative: another
        # command cannot gain an unrecorded second holdout observation.
        raise TrainingError("holdout was already released") from error


def _validate_holdout_release_claim(
    accepted_packet: AcceptedPacketBinding,
    capsule: EmbeddingCapsule,
    checkpoint_sha256: str,
    pre_holdout_gate_descriptor_sha256: str,
) -> None:
    """Require the canonical immutable claim before any holdout payload access."""

    registry = _holdout_release_registry_directory()
    claim = registry / f"{_holdout_release_claim_key(accepted_packet.packet_json_sha256)}.json"
    expected = _holdout_release_payload(
        accepted_packet,
        capsule,
        checkpoint_sha256,
        pre_holdout_gate_descriptor_sha256,
    )
    try:
        actual = claim.read_bytes()
    except OSError as error:
        raise TrainingError("holdout release claim") from error
    if actual != expected:
        raise TrainingError("holdout release claim binding")


def _holdout_release(
    accepted_packet: AcceptedPacketBinding,
    capsule: EmbeddingCapsule,
    checkpoint_sha256: str,
    pre_holdout_gate_descriptor_sha256: str,
) -> bytes:
    return _holdout_release_payload(
        accepted_packet, capsule, checkpoint_sha256, pre_holdout_gate_descriptor_sha256
    )


def _model_from_checkpoint(checkpoint: bytes) -> Any:
    torch, load, _save = _ml()
    try:
        state = load(checkpoint)
    except Exception as error:
        raise TrainingError("checkpoint safetensors") from error
    verify_checkpoint_tensors(state)
    model = _new_model(torch)
    with torch.no_grad():
        model.context_weight.copy_(state["context_projection.weight"])
        model.context_bias.copy_(state["context_projection.bias"])
        model.context_norm_weight.copy_(state["context_projection.layernorm.weight"])
        model.context_norm_bias.copy_(state["context_projection.layernorm.bias"])
        model.criterion_weight.copy_(state["criterion_projection.weight"])
        model.criterion_bias.copy_(state["criterion_projection.bias"])
        model.criterion_norm_weight.copy_(state["criterion_projection.layernorm.weight"])
        model.criterion_norm_bias.copy_(state["criterion_projection.layernorm.bias"])
        model.log_scale.copy_(state["log_scale"])
    return model


def _conformance_vectors(holdout: _Rows, model: Any) -> tuple[bytes, bytes]:
    """Seal one deterministic, raw-text-free vector for each fixed cell."""

    torch, _load, save = _ml()
    selected: list[tuple[str, int]] = []
    for locale, count in _CELLS:
        matches = [
            index
            for index, row in enumerate(holdout.rows)
            if row["locale"] == locale and row["option_count"] == count
        ]
        if not matches:
            raise TrainingError("conformance missing cell")
        selected.append(
            (f"{locale}:{count}", min(matches, key=lambda index: holdout.rows[index]["task_id"]))
        )
    contexts = torch.stack([holdout.context[index] for _cell, index in selected]).contiguous()
    criteria = torch.zeros((len(selected), 8, 384), dtype=torch.float32)
    masks = torch.zeros((len(selected), 8), dtype=torch.uint8)
    counts = torch.zeros((len(selected),), dtype=torch.int64)
    logits = torch.zeros((len(selected), 8), dtype=torch.float32)
    with torch.inference_mode():
        values = _logits(model, holdout, [index for _cell, index in selected])
    for row_index, ((_cell, source_index), value) in enumerate(zip(selected, values, strict=True)):
        row = holdout.rows[source_index]
        count = cast(int, row["option_count"])
        offset = cast(int, row["criterion_offset"])
        criteria[row_index, :count] = holdout.criteria[offset : offset + count]
        masks[row_index, :count] = 1
        counts[row_index] = count
        logits[row_index, :count] = value.to(dtype=torch.float32)
    tensors = {
        "context_embeddings": contexts,
        "criterion_embeddings": criteria,
        "option_masks": masks,
        "option_counts": counts,
        "expected_logits": logits,
    }
    if not bool(torch.isfinite(contexts).all().item()) or not bool(
        torch.isfinite(logits).all().item()
    ):
        raise TrainingError("conformance non-finite tensors")
    raw = save(tensors)
    manifest: dict[str, Any] = {
        "schema_version": "phase4e-conformance-manifest.v1",
        "cells": [cell for cell, _index in selected],
        "tensor_sha256": _sha(raw),
        "tensor_shapes": {name: list(value.shape) for name, value in tensors.items()},
        "manifest_sha256": "",
    }
    manifest["manifest_sha256"] = _sha(
        _canonical({key: value for key, value in manifest.items() if key != "manifest_sha256"})
    )
    return raw, _canonical(manifest) + b"\n"


def _dependency_versions() -> bytes:
    """Record only readable local package versions; no paths or environment data."""

    import platform
    from importlib import metadata

    names = ("torch", "safetensors", "numpy", "transformers", "tokenizers")
    try:
        versions = {name: metadata.version(name) for name in names}
    except metadata.PackageNotFoundError as error:
        raise TrainingError("local-minilm dependency version") from error
    return (
        _canonical(
            {
                "schema_version": "phase4e-dependency-versions.v1",
                "python": platform.python_version(),
                "packages": versions,
            }
        )
        + b"\n"
    )


def train_and_seal(
    capsule_path: Path,
    accepted_packet_path: Path,
    snapshot_path: Path,
    device: str,
    output_parent: Path,
    output_name: str,
) -> Path:
    """Train only from packet text re-derived on an explicit verified device."""

    # This must precede every embedding read.  A receipt inside the capsule is
    # only a copied binding; it is never an authority to train.
    accepted_packet = validate_accepted_packet_binding(accepted_packet_path)
    capsule = validate_embedding_capsule(capsule_path, accepted_packet)
    # Self-consistent capsule hashes are not an authority.  This production
    # path has no caller-supplied encoder or derivation callback: it must
    # reproduce every token, mask, and embedding from the packet and snapshot.
    verify_descriptor_bound_embeddings(
        capsule, accepted_packet_path, accepted_packet, snapshot_path, device
    )
    policy_manifest = validate_phase4e_policy()
    policy = cast(dict[str, Any], policy_manifest["training"])
    train_dev = _load_rows(capsule, "train_dev")
    families = {row["family_id"] for row in train_dev.rows}
    train = _Rows(
        tuple(row for row in train_dev.rows if row["split"] == "synthetic_train"), None, None
    )
    dev = _Rows(tuple(row for row in train_dev.rows if row["split"] == "synthetic_dev"), None, None)
    if (
        not train.rows
        or not dev.rows
        or any((row["locale"], row["option_count"]) not in _CELLS for row in dev.rows)
    ):
        raise TrainingError("train/dev split")
    # Preserve tensor storage while selecting view rows; all fixture rows are ordered.
    train = _Rows(train.rows, train_dev.context, train_dev.criteria)
    dev = _Rows(dev.rows, train_dev.context, train_dev.criteria)
    if {row["family_id"] for row in train.rows} & {row["family_id"] for row in dev.rows}:
        raise TrainingError("train/dev family overlap")
    if {(row["locale"], row["option_count"]) for row in dev.rows} != set(_CELLS):
        raise TrainingError("dev stratified cells")
    first, ledger, selected = _run_once(train, dev, policy)
    second, second_ledger, second_selected = _run_once(train, dev, policy)
    if (
        first != second
        or _canonical(ledger) != _canonical(second_ledger)
        or _canonical(selected) != _canonical(second_selected)
    ):
        raise TrainingError("CPU deterministic repeat mismatch")
    checkpoint_sha256 = _sha(first)
    # These values use only train/dev tensors and sealed holdout identity rows.
    # No holdout tensor is loaded or scored after this point until the immutable
    # descriptor and its linked irreversible claim both exist.
    cosine, random_projection = (
        _baseline(dev, random_projection=False, seed=int(policy["seed"])),
        _baseline(dev, random_projection=True, seed=int(policy["seed"])),
    )
    code_sha256 = _training_code_digest()
    gate, gate_descriptor_sha256 = _freeze_pre_holdout_gate(
        output_parent,
        _pre_holdout_gate_descriptor(
            capsule,
            accepted_packet,
            policy_manifest,
            selected,
            cosine,
            random_projection,
            checkpoint_sha256,
            code_sha256,
        ),
    )
    dev_gates = _dev_improvement_gates(gate)
    common_manifest: dict[str, Any] = {
        "schema_version": "phase4e-training-manifest.v2",
        "synthetic_only": True,
        "checkpoint_sha256": checkpoint_sha256,
        "pre_holdout_gate_descriptor_sha256": gate_descriptor_sha256,
        "packet_receipt_sha256": _sha((capsule.path / "accepted-packet-receipt.json").read_bytes()),
        "embedding_descriptor_sha256": capsule.descriptor_sha256,
        "base_encoder": capsule.manifest["base_encoder"],
        "rendering_revision": capsule.manifest["rendering_revision"],
        "architecture_revision": CHECKPOINT_ARCHITECTURE_REVISION,
        "training": policy,
        "code_sha256": code_sha256,
        "code_sources": _training_code_sources(),
        "selected": selected,
        "baselines": {"untrained_cosine": cosine, "random_projection": random_projection},
        "dev_gates": dev_gates,
        "outcome": "",
        "manifest_sha256": "",
    }
    base_files = {
        "checkpoint.safetensors": first,
        "epoch-ledger.json": _canonical(ledger) + b"\n",
        "dev-selection.json": _canonical(
            {
                "selected": selected,
                "checkpoint_sha256": checkpoint_sha256,
                "pre_holdout_gate_descriptor_sha256": gate_descriptor_sha256,
                "dev_gates": dev_gates,
                "holdout_opened_after_pre_holdout_gate_and_claim": False,
            }
        )
        + b"\n",
        "pre-holdout-gate.json": _canonical(gate) + b"\n",
        "accepted-packet-receipt.json": (
            capsule.path / "accepted-packet-receipt.json"
        ).read_bytes(),
        "embedding-descriptor.json": (capsule.path / "capsule-descriptor.json").read_bytes(),
        "architecture.json": _canonical(
            {
                "architecture_revision": CHECKPOINT_ARCHITECTURE_REVISION,
                "tensor_set": sorted(_model_from_checkpoint(first).checkpoint()),
                "base_encoder": capsule.manifest["base_encoder"],
            }
        )
        + b"\n",
    }
    if not all(dev_gates.values()):
        # This return is deliberately before the claim, full validator, or any
        # holdout payload/tensor derivation.  The sealed result is evidence of a
        # rejected experiment, never a release candidate.
        common_manifest["outcome"] = "pre_holdout_failed"
        common_manifest["manifest_sha256"] = _sha(
            _canonical(
                {key: value for key, value in common_manifest.items() if key != "manifest_sha256"}
            )
        )
        files = {
            **base_files,
            "training-manifest.json": _canonical(common_manifest) + b"\n",
        }
        files["capsule-descriptor.json"] = _descriptor(files, kind="training")
        if set(files) != _PRE_HOLDOUT_FAILED_OUTPUT_FILES:
            raise AssertionError("pre-holdout output file set")
        return _write_capsule(output_parent, output_name, files)
    _claim_holdout_once(accepted_packet, capsule, checkpoint_sha256, gate_descriptor_sha256)
    # The claim is intentionally adjacent to the first holdout-content read.
    # Now that retry or reselection is impossible, verify every descriptor,
    # digest, token, mask, and embedding before constructing any score.
    holdout = derive_holdout_embeddings_in_memory(
        capsule,
        accepted_packet_path,
        accepted_packet,
        snapshot_path,
        device,
        checkpoint_sha256,
        gate_descriptor_sha256,
    )
    if families & {row["family_id"] for row in holdout.rows}:
        raise TrainingError("holdout family overlap")
    if {(row["locale"], row["option_count"]) for row in holdout.rows} != set(_CELLS):
        raise TrainingError("holdout stratified cells")
    torch, _load, _save = _ml()
    model = _model_from_checkpoint(first)
    with torch.inference_mode():
        holdout_metric = _metric(holdout, _logits(model, holdout))
    report = _holdout_gates(holdout, holdout_metric, gate, gate_descriptor_sha256)
    conformance_vectors, conformance_manifest = _conformance_vectors(holdout, model)
    common_manifest["outcome"] = "passed" if report["passed"] else "holdout_failed"
    common_manifest["manifest_sha256"] = _sha(
        _canonical(
            {key: value for key, value in common_manifest.items() if key != "manifest_sha256"}
        )
    )
    files = {
        **base_files,
        "training-manifest.json": _canonical(common_manifest) + b"\n",
        "dev-selection.json": _canonical(
            {
                "selected": selected,
                "checkpoint_sha256": checkpoint_sha256,
                "pre_holdout_gate_descriptor_sha256": gate_descriptor_sha256,
                "dev_gates": dev_gates,
                "holdout_opened_after_pre_holdout_gate_and_claim": True,
            }
        )
        + b"\n",
        "holdout-release.json": _holdout_release(
            accepted_packet, capsule, checkpoint_sha256, gate_descriptor_sha256
        ),
        "holdout-report.json": _canonical(report) + b"\n",
        "conformance-vectors.safetensors": conformance_vectors,
        "conformance-manifest.json": conformance_manifest,
        "dependency-versions.json": _dependency_versions(),
    }
    files["capsule-descriptor.json"] = _descriptor(files, kind="training")
    if set(files) != _FINAL_OUTPUT_FILES:
        raise AssertionError("training output file set")
    return _write_capsule(output_parent, output_name, files)


def verify_training_capsule(path: Path) -> dict[str, Any]:
    """Validate the sealed, synthetic-only result and its closed ranker state."""

    if not path.is_dir():
        raise TrainingError("training output file set")
    manifest = _json(path / "training-manifest.json")
    required = {
        "schema_version",
        "synthetic_only",
        "outcome",
        "checkpoint_sha256",
        "pre_holdout_gate_descriptor_sha256",
        "packet_receipt_sha256",
        "embedding_descriptor_sha256",
        "base_encoder",
        "rendering_revision",
        "architecture_revision",
        "training",
        "code_sha256",
        "code_sources",
        "selected",
        "baselines",
        "dev_gates",
        "manifest_sha256",
    }
    if (
        set(manifest) != required
        or manifest["schema_version"] != "phase4e-training-manifest.v2"
        or manifest["synthetic_only"] is not True
        or manifest["outcome"] not in {"pre_holdout_failed", "holdout_failed", "passed"}
        or manifest["checkpoint_sha256"] != _sha((path / "checkpoint.safetensors").read_bytes())
        or not _is_sha(manifest["pre_holdout_gate_descriptor_sha256"])
        or manifest["architecture_revision"] != CHECKPOINT_ARCHITECTURE_REVISION
    ):
        raise TrainingError("training manifest")
    expected_files = (
        _PRE_HOLDOUT_FAILED_OUTPUT_FILES
        if manifest["outcome"] == "pre_holdout_failed"
        else _FINAL_OUTPUT_FILES
    )
    if {child.name for child in path.iterdir()} != expected_files:
        raise TrainingError("training output file set")
    _validate_descriptor(
        path / "capsule-descriptor.json",
        expected_files - {"capsule-descriptor.json"},
        kind="training",
    )
    if manifest["manifest_sha256"] != _sha(
        _canonical({key: value for key, value in manifest.items() if key != "manifest_sha256"})
    ):
        raise TrainingError("training manifest digest")
    if (
        manifest["training"] != validate_phase4e_policy()["training"]
        or manifest["packet_receipt_sha256"]
        != _sha((path / "accepted-packet-receipt.json").read_bytes())
        or manifest["embedding_descriptor_sha256"]
        != _sha((path / "embedding-descriptor.json").read_bytes())
        or not isinstance(manifest["base_encoder"], dict)
        or manifest["base_encoder"].get("id") != BASE_ENCODER_ID
        or manifest["base_encoder"].get("revision") != BASE_ENCODER_REVISION
        or manifest["base_encoder"].get("frozen") is not True
        or not _is_sha(manifest["base_encoder"].get("snapshot_complete_sha256"))
        or not _is_sha(manifest["code_sha256"])
        or manifest["code_sources"] != _training_code_sources()
        or manifest["code_sha256"] != _training_code_digest()
    ):
        raise TrainingError("training manifest binding")
    gate_raw = (path / "pre-holdout-gate.json").read_bytes()
    gate = _validate_pre_holdout_gate_descriptor(gate_raw)
    if (
        manifest["pre_holdout_gate_descriptor_sha256"] != _sha(gate_raw)
        or gate["checkpoint_sha256"] != manifest["checkpoint_sha256"]
        or gate["embedding_descriptor_sha256"] != manifest["embedding_descriptor_sha256"]
        or gate["accepted_packet"]["packet_receipt_sha256"] != manifest["packet_receipt_sha256"]
        or gate["selected"] != manifest["selected"]
        or gate["dev_baselines"] != manifest["baselines"]
        or gate["training"] != manifest["training"]
        or gate["code_sha256"] != manifest["code_sha256"]
    ):
        raise TrainingError("pre-holdout gate descriptor binding")
    dev_gates = _dev_improvement_gates(gate)
    if manifest["dev_gates"] != dev_gates:
        raise TrainingError("pre-holdout dev gate binding")
    selection = _json(path / "dev-selection.json")
    opened = manifest["outcome"] != "pre_holdout_failed"
    if selection != {
        "selected": manifest["selected"],
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "pre_holdout_gate_descriptor_sha256": manifest["pre_holdout_gate_descriptor_sha256"],
        "dev_gates": dev_gates,
        "holdout_opened_after_pre_holdout_gate_and_claim": opened,
    }:
        raise TrainingError("dev selection pre-holdout gate binding")
    if manifest["outcome"] == "pre_holdout_failed":
        if all(dev_gates.values()):
            raise TrainingError("pre-holdout failed outcome")
        _torch, load, _save = _ml()
        try:
            state = load((path / "checkpoint.safetensors").read_bytes())
        except Exception as error:
            raise TrainingError("checkpoint safetensors") from error
        verify_checkpoint_tensors(state)
        return manifest
    if not all(dev_gates.values()):
        raise TrainingError("postclaim outcome without dev gate")
    release = _json(path / "holdout-release.json")
    if release != {
        "schema_version": _HOLDOUT_RELEASE_SCHEMA,
        "packet_json_sha256": gate["accepted_packet"]["packet_json_sha256"],
        "embedding_descriptor_sha256": manifest["embedding_descriptor_sha256"],
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "pre_holdout_gate_descriptor_sha256": manifest["pre_holdout_gate_descriptor_sha256"],
    }:
        raise TrainingError("holdout release descriptor")
    report = _json(path / "holdout-report.json")
    if (
        report.get("pre_holdout_gate_descriptor_sha256")
        != manifest["pre_holdout_gate_descriptor_sha256"]
        or report.get("precomputed_holdout_identity_baselines")
        != gate["holdout_identity_baselines"]
        or report.get("success_thresholds") != gate["success_thresholds"]
        or report.get("zero_invalid_invariants") != gate["zero_invalid_invariants"]
    ):
        raise TrainingError("holdout report pre-holdout gate binding")
    conformance = _json(path / "conformance-manifest.json")
    conformance_required = {
        "schema_version",
        "cells",
        "tensor_sha256",
        "tensor_shapes",
        "manifest_sha256",
    }
    if (
        set(conformance) != conformance_required
        or conformance["schema_version"] != "phase4e-conformance-manifest.v1"
        or conformance["cells"] != [f"{locale}:{count}" for locale, count in _CELLS]
        or conformance["tensor_sha256"]
        != _sha((path / "conformance-vectors.safetensors").read_bytes())
        or conformance["manifest_sha256"]
        != _sha(
            _canonical(
                {key: value for key, value in conformance.items() if key != "manifest_sha256"}
            )
        )
    ):
        raise TrainingError("conformance manifest")
    _torch, load, _save = _ml()
    try:
        state = load((path / "checkpoint.safetensors").read_bytes())
    except Exception as error:
        raise TrainingError("checkpoint safetensors") from error
    verify_checkpoint_tensors(state)
    try:
        vectors = load((path / "conformance-vectors.safetensors").read_bytes())
    except Exception as error:
        raise TrainingError("conformance safetensors") from error
    expected_shapes = {
        "context_embeddings": [14, 384],
        "criterion_embeddings": [14, 8, 384],
        "option_masks": [14, 8],
        "option_counts": [14],
        "expected_logits": [14, 8],
    }
    if (
        set(vectors) != set(expected_shapes)
        or conformance["tensor_shapes"] != expected_shapes
        or any(list(vectors[name].shape) != shape for name, shape in expected_shapes.items())
        or vectors["context_embeddings"].dtype != _torch.float32
        or vectors["criterion_embeddings"].dtype != _torch.float32
        or vectors["option_masks"].dtype != _torch.uint8
        or vectors["option_counts"].dtype != _torch.int64
        or vectors["expected_logits"].dtype != _torch.float32
        or not all(
            bool(_torch.isfinite(vectors[name]).all().item())
            for name in ("context_embeddings", "criterion_embeddings", "expected_logits")
        )
    ):
        raise TrainingError("conformance tensor contract")
    counts = vectors["option_counts"]
    masks = vectors["option_masks"]
    if not bool(((counts >= 2) & (counts <= 8)).all().item()) or not bool(
        ((masks == 0) | (masks == 1)).all().item()
    ):
        raise TrainingError("conformance masks")
    model = _model_from_checkpoint((path / "checkpoint.safetensors").read_bytes())
    for index in range(14):
        count = int(counts[index].item())
        if int(masks[index].sum().item()) != count:
            raise TrainingError("conformance masks")
        row = _Rows(
            (
                {
                    "row_index": 0,
                    "criterion_offset": 0,
                    "option_count": count,
                    "gold_position": 0,
                    "locale": "pt-BR",
                },
            ),
            vectors["context_embeddings"][index : index + 1],
            vectors["criterion_embeddings"][index, :count],
        )
        actual = _logits(model, row)[0]
        if not bool(_torch.equal(actual, vectors["expected_logits"][index, :count])):
            raise TrainingError("conformance logits")
    dependencies = _json(path / "dependency-versions.json")
    if (
        set(dependencies) != {"schema_version", "python", "packages"}
        or dependencies["schema_version"] != "phase4e-dependency-versions.v1"
        or not isinstance(dependencies["python"], str)
        or not isinstance(dependencies["packages"], dict)
        or set(dependencies["packages"])
        != {"torch", "safetensors", "numpy", "transformers", "tokenizers"}
        or not all(isinstance(value, str) and value for value in dependencies["packages"].values())
    ):
        raise TrainingError("dependency versions")
    return manifest
