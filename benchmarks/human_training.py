"""Phase 4B.2 sealed train/dev embedding and CPU-head stages.

The module intentionally has no packet discovery, cache lookup, network client,
calibration, or blind-set entry point.  Every stage accepts one explicit
descriptor-backed capsule and creates a new immutable output child.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import platform
from pathlib import Path
from typing import Any, Never, cast

from benchmarks.first_party_packet import LABELS
from benchmarks.human_research import (
    HumanResearchError,
    _bind_receipt,
    _finish_stage,
    _json,
    _jsonl,
    _ledger,
    _read_capsule,
    _safe_name,
    _stage,
    _write_at,
    digest,
    reconstruct_train_dev_descriptor,
    validate_packet_view_subset,
)
from benchmarks.synthetic_research import SyntheticError, train_head_from_embeddings
from saracura.backends.minilm import (
    extract_verified_minilm_embeddings,
    human_checkpoint_conformance,
)
from saracura.research_trust import canonical, verify_release_receipt
from saracura.serialization import frame_segments
from saracura.verified_bytes import MINILM_REVISION, package_registry_sha256

EMBEDDING_LIMITS = {
    "embeddings.safetensors": 64 * 1024 * 1024,
    "embedding-manifest.json": 1 * 1024 * 1024,
    "packet-manifest.json": 1 * 1024 * 1024,
    "packet-release-receipt.json": 64 * 1024,
    "capsule-descriptor.json": 1 * 1024 * 1024,
}
TRAINING_LIMITS = {
    "checkpoint.safetensors": 16 * 1024 * 1024,
    "training-manifest.json": 1 * 1024 * 1024,
    "training-report.json": 1 * 1024 * 1024,
    "packet-manifest.json": 1 * 1024 * 1024,
    "packet-release-receipt.json": 64 * 1024,
    "capsule-descriptor.json": 1 * 1024 * 1024,
}
TRAIN_DEV_LIMITS = {
    "train.jsonl": 16 * 1024 * 1024,
    "dev.jsonl": 16 * 1024 * 1024,
    "packet-manifest.json": 1 * 1024 * 1024,
    "packet-release-receipt.json": 64 * 1024,
    "capsule-descriptor.json": 1 * 1024 * 1024,
}
LABEL_TO_INDEX = {label: index for index, label in enumerate(LABELS)}


class _SanitizedArgumentParser(argparse.ArgumentParser):
    """Render no path, filename, or malformed operator value on CLI failure."""

    def error(self, _message: str) -> Never:
        raise HumanResearchError("command arguments")


def _descriptor(values: dict[str, bytes], *, kind: str) -> dict[str, Any]:
    raw = values["capsule-descriptor.json"]
    descriptor = _json(raw, maximum=1 * 1024 * 1024)
    if (
        set(descriptor)
        != {
            "schema_version",
            "capsule_kind",
            "packet_sha256",
            "packet_release_receipt_sha256",
            "files",
            "descriptor_sha256",
        }
        or descriptor["schema_version"] != "human-capsule.v1"
        or descriptor["capsule_kind"] != kind
        or not isinstance(descriptor["packet_sha256"], str)
        or len(descriptor["packet_sha256"]) != 64
        or not isinstance(descriptor["packet_release_receipt_sha256"], str)
        or len(descriptor["packet_release_receipt_sha256"]) != 64
    ):
        raise HumanResearchError("capsule descriptor boundary")
    unsigned = dict(descriptor)
    recorded = unsigned.pop("descriptor_sha256")
    if not isinstance(recorded, str) or digest(canonical(unsigned)) != recorded:
        raise HumanResearchError("capsule descriptor digest")
    expected = set(values) - {"capsule-descriptor.json"}
    ledger = descriptor["files"]
    if (
        not isinstance(ledger, list)
        or len(ledger) != len(expected)
        or [item.get("path") for item in ledger if isinstance(item, dict)] != sorted(expected)
    ):
        raise HumanResearchError("capsule descriptor ledger")
    for item in ledger:
        if (
            not isinstance(item, dict)
            or set(item) != {"path", "bytes", "sha256"}
            or item["path"] not in expected
            or item["bytes"] != len(values[item["path"]])
            or item["sha256"] != digest(values[item["path"]])
        ):
            raise HumanResearchError("capsule descriptor ledger")
    return descriptor


def _trusted_capsule(
    path: Path, limits: dict[str, int], *, kind: str
) -> tuple[dict[str, bytes], dict[str, Any]]:
    values = dict(_read_capsule(path, limits))
    descriptor = _descriptor(values, kind=kind)
    if kind == "train-dev":
        packet = validate_packet_view_subset(
            values["packet-manifest.json"],
            {"train": values["train.jsonl"], "dev": values["dev.jsonl"]},
        )
    else:
        # Downstream capsules do not carry packet views.  Their source stages
        # have reconciled those views and bind them through this signed packet.
        packet = _packet_only(values["packet-manifest.json"])
    try:
        receipt = verify_release_receipt(values["packet-release-receipt.json"])
    except Exception as exc:
        raise HumanResearchError("packet release receipt") from exc
    _bind_receipt(receipt, packet)
    if descriptor["packet_sha256"] != packet["packet_sha256"] or descriptor[
        "packet_release_receipt_sha256"
    ] != digest(values["packet-release-receipt.json"]):
        raise HumanResearchError("capsule governance binding")
    return values, descriptor


def _packet_only(raw: bytes) -> dict[str, Any]:
    """Parse every packet field without inventing absent split files.

    ``validate_packet_view_subset`` requires at least one supplied view.  The
    embedding/training capsules deliberately carry none, so a tiny dummy view
    is not acceptable; their packet has already been verified at the upstream
    stage.  This function still makes the manifest's self digest and all static
    schema claims go through the same closed validator by extracting no data.
    """

    packet = _json(raw, maximum=1 * 1024 * 1024)
    # A synthetic zero-byte view must never be accepted.  Instead validate the
    # full closed shape and self-digest here, while signature verification below
    # cryptographically binds every unavailable view digest.
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
    digest_fields = {
        "protocol_sha256",
        "policy_registry_sha256",
        "states_sha256",
        "split_plan_sha256",
        "contributors_sha256",
        "annotations_sha256",
        "adjudications_sha256",
        "controls_sha256",
        "takedown_ledger_sha256",
    }
    if (
        set(packet) != required
        or packet.get("schema_version") != "support-routing-human-packet.v1"
        or any(
            not isinstance(packet[name], str) or len(packet[name]) != 64 for name in digest_fields
        )
        or packet.get("labels") != list(LABELS)
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
        raise HumanResearchError("packet manifest schema")
    unsigned = dict(packet)
    recorded = unsigned.pop("packet_sha256", None)
    if not isinstance(recorded, str) or digest(canonical(unsigned)) != recorded:
        raise HumanResearchError("packet manifest digest")
    # Calling the subset validator occurs at both packet-consuming stages that
    # have records.  Here we still reject malformed static counts/file ledgers.
    files, counts = packet.get("files"), packet.get("accepted_counts")
    excluded = packet.get("excluded_counts")
    if (
        not isinstance(files, dict)
        or set(files) != {"train", "dev", "calibration", "blind_test"}
        or not isinstance(counts, dict)
        or set(counts) != {"total", "by_split", "by_split_label"}
        or type(counts["total"]) is not int
        or counts["total"] < 500
        or not isinstance(counts["by_split"], dict)
        or not isinstance(counts["by_split_label"], dict)
        or set(counts["by_split"]) != set(files)
        or set(counts["by_split_label"]) != set(files)
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
        raise HumanResearchError("packet manifest ledger")
    claimed_total = 0
    for split, minimum in (
        ("train", 200),
        ("dev", 100),
        ("calibration", 100),
        ("blind_test", 100),
    ):
        ledger = files[split]
        label_counts = counts["by_split_label"][split]
        split_count = counts["by_split"][split]
        if (
            not isinstance(ledger, dict)
            or set(ledger) != {"bytes", "sha256"}
            or type(ledger["bytes"]) is not int
            or ledger["bytes"] <= 0
            or not isinstance(ledger["sha256"], str)
            or len(ledger["sha256"]) != 64
            or type(split_count) is not int
            or split_count < minimum
            or not isinstance(label_counts, dict)
            or set(label_counts) != set(LABELS)
            or any(
                type(label_counts[label]) is not int or label_counts[label] < 10 for label in LABELS
            )
            or sum(label_counts.values()) != split_count
        ):
            raise HumanResearchError("packet manifest ledger")
        claimed_total += split_count
    if claimed_total != counts["total"]:
        raise HumanResearchError("packet manifest ledger")
    return packet


def _write_capsule(parent: Path, name: str, values: dict[str, bytes]) -> Path:
    final = _safe_name(name)
    parent_fd, stage_name, stage_fd = _stage(parent, final)
    try:
        for filename in sorted(values):
            _write_at(stage_fd, filename, values[filename])
        _finish_stage(parent_fd, stage_name, stage_fd, final)
    finally:
        import os

        os.close(stage_fd)
        os.close(parent_fd)
    return parent / final


def _make_descriptor(
    kind: str, values: dict[str, bytes], packet: dict[str, Any], receipt: bytes
) -> bytes:
    value: dict[str, Any] = {
        "schema_version": "human-capsule.v1",
        "capsule_kind": kind,
        "packet_sha256": packet["packet_sha256"],
        "packet_release_receipt_sha256": digest(receipt),
        "files": _ledger(values, set(values)),
        "descriptor_sha256": "",
    }
    value["descriptor_sha256"] = digest(
        canonical({k: v for k, v in value.items() if k != "descriptor_sha256"})
    )
    return canonical(value) + b"\n"


def _load_rows(raw: bytes, split: str) -> list[dict[str, Any]]:
    rows = _jsonl(raw, maximum=16 * 1024 * 1024)
    prior = ""
    for row in rows:
        if set(row) != {
            "state_id",
            "family_id",
            "locale",
            "candidate_label",
            "text",
            "content_digest",
        }:
            raise HumanResearchError("training view schema")
        if (
            row["state_id"] <= prior
            or row["locale"] != "pt-BR"
            or row["candidate_label"] not in LABEL_TO_INDEX
        ):
            raise HumanResearchError("training view order")
        if (
            not isinstance(row["text"], str)
            or len(row["text"]) > 320
            or len(row["text"].encode()) > 1280
        ):
            raise HumanResearchError("training input limit")
        prior = row["state_id"]
    return rows


def _environment(torch: Any, safetensors: Any) -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "transformers": importlib.metadata.version("transformers"),
        "tokenizers": importlib.metadata.version("tokenizers"),
        "safetensors": str(safetensors.__version__),
        "platform": platform.platform(),
    }


def _training_code_sha256() -> str:
    """Bind the 4B.2 wrapper and the exact reused Phase 3B core."""

    return digest(
        frame_segments(
            "benchmarks/human_training.py",
            Path(__file__).read_bytes(),
            "benchmarks/synthetic_research.py",
            Path(__file__).with_name("synthetic_research.py").read_bytes(),
        )
    )


def _extract_with_encoder(rows: list[dict[str, Any]], snapshot: Path) -> Any:
    """Extract only from one package-owned snapshot of the explicit directory.

    The Phase 4A benchmark loader is intentionally not involved: it resolves a
    cache path and re-opens files.  ``extract_verified_minilm_embeddings``
    reads the descriptor-owned bytes once, then performs the exact Phase 4A
    attention-mask float32 pooling in deterministic batches of at most 32.
    """

    try:
        return extract_verified_minilm_embeddings(
            encoder_snapshot=snapshot,
            texts=[cast(str, row["text"]) for row in rows],
            device="mps",
            batch_size=32,
        )
    except Exception as exc:
        raise HumanResearchError("verified encoder snapshot required") from exc


def _match_received_train_dev_descriptor(values: dict[str, bytes]) -> tuple[dict[str, Any], bytes]:
    """Prove that the received source descriptor is the signed packet's one."""

    packet = _packet_only(values["packet-manifest.json"])
    receipt = values["packet-release-receipt.json"]
    rebuilt_descriptor = reconstruct_train_dev_descriptor(
        packet, values["packet-manifest.json"], receipt
    )
    # Extraction has the actual source descriptor.  It must be byte-identical
    # to the reconstruction, rather than only sharing a claimed manifest hash.
    if rebuilt_descriptor != values["capsule-descriptor.json"]:
        raise HumanResearchError("train/dev descriptor binding")
    return packet, receipt


def extract_train_dev_embeddings(
    train_dev_capsule: Path, encoder_snapshot: Path, output_parent: Path, output_name: str
) -> Path:
    values, _descriptor_value = _trusted_capsule(
        train_dev_capsule, TRAIN_DEV_LIMITS, kind="train-dev"
    )
    packet, receipt = _match_received_train_dev_descriptor(values)
    train = _load_rows(values["train.jsonl"], "train")
    dev = _load_rows(values["dev.jsonl"], "dev")
    rows = train + dev
    embeddings = _extract_with_encoder(rows, encoder_snapshot)
    try:
        safetensors = importlib.import_module("safetensors")
        torch = importlib.import_module("torch")
        save = importlib.import_module("safetensors.torch").save
    except ImportError as exc:
        raise HumanResearchError("local-minilm extra is required") from exc
    labels = torch.tensor(
        [LABEL_TO_INDEX[row["candidate_label"]] for row in rows], dtype=torch.int64
    )
    splits = torch.tensor([0] * len(train) + [1] * len(dev), dtype=torch.uint8)
    embedding_raw = save({"embeddings": embeddings, "labels": labels, "splits": splits})
    try:
        round_trip = importlib.import_module("safetensors.torch").load(embedding_raw)
    except Exception as exc:
        raise HumanResearchError("embedding safetensors round-trip") from exc
    if set(round_trip) != {"embeddings", "labels", "splits"} or any(
        not torch.equal(round_trip[name], expected)
        for name, expected in {
            "embeddings": embeddings,
            "labels": labels,
            "splits": splits,
        }.items()
    ):
        raise HumanResearchError("embedding safetensors round-trip")
    manifest: dict[str, Any] = {
        "schema_version": "human-embedding-manifest.v1",
        "packet_manifest_sha256": digest(values["packet-manifest.json"]),
        "packet_release_receipt_sha256": digest(receipt),
        "train_dev_capsule_sha256": digest(values["capsule-descriptor.json"]),
        "protocol_sha256": packet["protocol_sha256"],
        "encoder": {
            "candidate": "multilingual-minilm-l12",
            "revision": MINILM_REVISION,
            "registry_sha256": package_registry_sha256(),
            "device": "mps",
            "frozen": True,
        },
        "labels": list(LABELS),
        "device": "mps",
        "batch_size": 32,
        "max_length": 128,
        "records": [
            {
                "split": "train" if i < len(train) else "dev",
                "state_id": row["state_id"],
                "family_id": row["family_id"],
                "label_index": LABEL_TO_INDEX[row["candidate_label"]],
                "row_index": i,
            }
            for i, row in enumerate(rows)
        ],
        "files": {
            "embeddings.safetensors": {"bytes": len(embedding_raw), "sha256": digest(embedding_raw)}
        },
        "environment": _environment(torch, safetensors),
        "manifest_sha256": "",
    }
    manifest["manifest_sha256"] = digest(
        canonical({k: v for k, v in manifest.items() if k != "manifest_sha256"})
    )
    output = {
        "embeddings.safetensors": embedding_raw,
        "embedding-manifest.json": canonical(manifest) + b"\n",
        "packet-manifest.json": values["packet-manifest.json"],
        "packet-release-receipt.json": receipt,
    }
    output["capsule-descriptor.json"] = _make_descriptor("embedding", output, packet, receipt)
    return _write_capsule(output_parent, output_name, output)


def _validate_embedding_manifest(
    raw: bytes,
    *,
    tensor_raw: bytes,
    packet_raw: bytes,
    receipt_raw: bytes,
    descriptor_raw: bytes,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reconcile every sealed embedding field before a CPU head is constructed."""

    manifest = _json(raw, maximum=EMBEDDING_LIMITS["embedding-manifest.json"])
    required = {
        "schema_version",
        "packet_manifest_sha256",
        "packet_release_receipt_sha256",
        "train_dev_capsule_sha256",
        "protocol_sha256",
        "encoder",
        "labels",
        "device",
        "batch_size",
        "max_length",
        "records",
        "files",
        "environment",
        "manifest_sha256",
    }
    if (
        set(manifest) != required
        or manifest["schema_version"] != "human-embedding-manifest.v1"
        or manifest["labels"] != list(LABELS)
        or manifest["device"] != "mps"
        or manifest["batch_size"] != 32
        or manifest["max_length"] != 128
        or not isinstance(manifest["records"], list)
        or not manifest["records"]
        or not isinstance(manifest["files"], dict)
        or set(manifest["files"]) != {"embeddings.safetensors"}
        or not isinstance(manifest["environment"], dict)
        or set(manifest["environment"])
        != {"python", "torch", "transformers", "tokenizers", "safetensors", "platform"}
        or any(
            not isinstance(value, str) or not value for value in manifest["environment"].values()
        )
    ):
        raise HumanResearchError("embedding manifest schema")
    unsigned = dict(manifest)
    self_digest = unsigned.pop("manifest_sha256")
    if not isinstance(self_digest, str) or digest(canonical(unsigned)) != self_digest:
        raise HumanResearchError("embedding manifest digest")
    hashes = (
        "packet_manifest_sha256",
        "packet_release_receipt_sha256",
        "train_dev_capsule_sha256",
        "protocol_sha256",
    )
    if any(not isinstance(manifest[name], str) or len(manifest[name]) != 64 for name in hashes):
        raise HumanResearchError("embedding manifest binding")
    if manifest["packet_manifest_sha256"] != digest(packet_raw) or manifest[
        "packet_release_receipt_sha256"
    ] != digest(receipt_raw):
        raise HumanResearchError("embedding manifest binding")
    encoder = manifest["encoder"]
    if not isinstance(encoder, dict) or encoder != {
        "candidate": "multilingual-minilm-l12",
        "revision": MINILM_REVISION,
        "registry_sha256": package_registry_sha256(),
        "device": "mps",
        "frozen": True,
    }:
        raise HumanResearchError("embedding encoder binding")
    file_ledger = manifest["files"]["embeddings.safetensors"]
    if (
        not isinstance(file_ledger, dict)
        or set(file_ledger) != {"bytes", "sha256"}
        or file_ledger["bytes"] != len(tensor_raw)
        or file_ledger["sha256"] != digest(tensor_raw)
    ):
        raise HumanResearchError("embedding tensor ledger")
    packet = _packet_only(packet_raw)
    rebuilt_train_dev_descriptor = reconstruct_train_dev_descriptor(packet, packet_raw, receipt_raw)
    if manifest["train_dev_capsule_sha256"] != digest(rebuilt_train_dev_descriptor):
        raise HumanResearchError("embedding train/dev descriptor binding")
    if manifest["protocol_sha256"] != packet["protocol_sha256"]:
        raise HumanResearchError("embedding protocol binding")
    try:
        torch = importlib.import_module("torch")
        load = importlib.import_module("safetensors.torch").load
    except ImportError as exc:
        raise HumanResearchError("local-minilm extra is required") from exc
    try:
        tensors = load(tensor_raw)
    except Exception as exc:
        raise HumanResearchError("embedding safetensors") from exc
    if (
        set(tensors) != {"embeddings", "labels", "splits"}
        or tensors["embeddings"].dtype != torch.float32
        or tensors["labels"].dtype != torch.int64
        or tensors["splits"].dtype != torch.uint8
        or tensors["embeddings"].ndim != 2
        or tuple(tensors["embeddings"].shape[1:]) != (384,)
        or tensors["labels"].ndim != 1
        or tensors["splits"].ndim != 1
        or tensors["labels"].shape[0] != tensors["embeddings"].shape[0]
        or tensors["splits"].shape[0] != tensors["embeddings"].shape[0]
        or not bool(torch.isfinite(tensors["embeddings"]).all().item())
        or not bool(((tensors["labels"] >= 0) & (tensors["labels"] < 5)).all().item())
        or not bool(((tensors["splits"] == 0) | (tensors["splits"] == 1)).all().item())
    ):
        raise HumanResearchError("embedding tensor contract")
    records = manifest["records"]
    if len(records) != tensors["embeddings"].shape[0]:
        raise HumanResearchError("embedding record count")
    prior: tuple[int, str] | None = None
    state_ids: set[str] = set()
    family_ids: set[str] = set()
    split_counts = {"train": 0, "dev": 0}
    label_counts = {"train": {label: 0 for label in LABELS}, "dev": {label: 0 for label in LABELS}}
    for index, record in enumerate(records):
        if (
            not isinstance(record, dict)
            or set(record) != {"split", "state_id", "family_id", "label_index", "row_index"}
            or record["split"] not in {"train", "dev"}
            or not isinstance(record["state_id"], str)
            or not isinstance(record["family_id"], str)
            or type(record["label_index"]) is not int
            or type(record["row_index"]) is not int
            or record["row_index"] != index
            or not 0 <= record["label_index"] < 5
            or int(tensors["labels"][index].item()) != record["label_index"]
            or int(tensors["splits"][index].item()) != (0 if record["split"] == "train" else 1)
            or record["state_id"] in state_ids
            or record["family_id"] in family_ids
        ):
            raise HumanResearchError("embedding records")
        key = (
            0 if record["split"] == "train" else 1,
            record["state_id"],
        )
        if prior is not None and key <= prior:
            raise HumanResearchError("embedding record order")
        prior = key
        state_ids.add(record["state_id"])
        family_ids.add(record["family_id"])
        split = cast(str, record["split"])
        split_counts[split] += 1
        label_counts[split][LABELS[record["label_index"]]] += 1
    counts = packet["accepted_counts"]
    if split_counts != {split: counts["by_split"][split] for split in ("train", "dev")} or any(
        label_counts[split] != counts["by_split_label"][split] for split in ("train", "dev")
    ):
        raise HumanResearchError("embedding packet reconciliation")
    return manifest, tensors


def _classification(predicted: list[int], actual: list[int]) -> dict[str, Any]:
    matrix = [[0] * 5 for _ in range(5)]
    for prediction, target in zip(predicted, actual, strict=True):
        matrix[target][prediction] += 1
    per_label: dict[str, Any] = {}
    f1s: list[float] = []
    for index, label in enumerate(LABELS):
        tp = matrix[index][index]
        fp = sum(matrix[row][index] for row in range(5)) - tp
        fn = sum(matrix[index]) - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1s.append(f1)
        per_label[label] = {
            "support": sum(matrix[index]),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return {
        "count": len(actual),
        "accuracy": sum(p == a for p, a in zip(predicted, actual, strict=True)) / len(actual),
        "macro_f1": sum(f1s) / 5,
        "confusion_matrix": matrix,
        "per_label": per_label,
    }


def train_head(
    embedding_capsule: Path,
    encoder_snapshot: Path,
    output_parent: Path,
    output_name: str,
) -> Path:
    values, _ = _trusted_capsule(embedding_capsule, EMBEDDING_LIMITS, kind="embedding")
    manifest, sealed = _validate_embedding_manifest(
        values["embedding-manifest.json"],
        tensor_raw=values["embeddings.safetensors"],
        packet_raw=values["packet-manifest.json"],
        receipt_raw=values["packet-release-receipt.json"],
        descriptor_raw=values["capsule-descriptor.json"],
    )
    try:
        safetensors = importlib.import_module("safetensors")
        torch = importlib.import_module("torch")
    except ImportError as exc:
        raise HumanResearchError("local-minilm extra is required") from exc
    train_mask = sealed["splits"] == 0
    dev_mask = sealed["splits"] == 1
    vectors = sealed["embeddings"]
    targets = sealed["labels"]
    if not bool(train_mask.any().item()) or not bool(dev_mask.any().item()):
        raise HumanResearchError("embedding split contract")
    train_vectors, train_targets = vectors[train_mask], targets[train_mask]
    dev_vectors, dev_targets = vectors[dev_mask], targets[dev_mask]
    try:
        # Phase 3B's core executes the two deterministic CPU runs, finite loss
        # checks, early stopping, safetensors round-trip, and byte/ledger
        # equality. ``synthetic_holdout`` is an internal reload assertion only;
        # no blind or calibration view is supplied to this stage.
        core = train_head_from_embeddings(
            {
                "synthetic_train": (train_vectors, train_targets),
                "synthetic_dev": (dev_vectors, dev_targets),
                "synthetic_holdout": (dev_vectors, dev_targets),
            },
            seed=20260921,
            max_epochs=80,
            patience=10,
        )
    except SyntheticError as exc:
        raise HumanResearchError("CPU head determinism failure") from exc
    checkpoint = core["checkpoint"]
    result_metrics = dict(core["metrics"])
    result_metrics["synthetic_only"] = False
    packet = _json(values["packet-manifest.json"], maximum=1 * 1024 * 1024)
    receipt = values["packet-release-receipt.json"]
    try:
        conformance = human_checkpoint_conformance(
            encoder_snapshot=encoder_snapshot,
            checkpoint=checkpoint,
            expected_encoder=manifest["encoder"],
        )
    except Exception as exc:
        raise HumanResearchError("human serving conformance") from exc
    human_manifest: dict[str, Any] = {
        "schema_version": "human-training-manifest.v1",
        "workflow_revision": "phase4b-human-research-training.v1",
        "synthetic_only": False,
        "packet_manifest_sha256": digest(values["packet-manifest.json"]),
        "packet_release_receipt_sha256": digest(receipt),
        "embedding_manifest_sha256": digest(values["embedding-manifest.json"]),
        "training_code_sha256": _training_code_sha256(),
        "protocol_sha256": packet["protocol_sha256"],
        "encoder": manifest["encoder"],
        "labels": list(LABELS),
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
            "embeddings.safetensors": digest(values["embeddings.safetensors"]),
            "checkpoint.safetensors": digest(checkpoint),
        },
        "metrics": result_metrics,
        "environment": _environment(torch, safetensors),
        "serving_conformance": conformance,
        "authorizations": {"calibration": True, "automation": False, "quality_claims": False},
        "manifest_sha256": "",
    }
    human_manifest["manifest_sha256"] = digest(
        canonical({k: v for k, v in human_manifest.items() if k != "manifest_sha256"})
    )
    report = (
        canonical({"schema_version": "human-training-report.v1", "metrics": result_metrics}) + b"\n"
    )
    output = {
        "checkpoint.safetensors": checkpoint,
        "training-manifest.json": canonical(human_manifest) + b"\n",
        "training-report.json": report,
        "packet-manifest.json": values["packet-manifest.json"],
        "packet-release-receipt.json": receipt,
    }
    output["capsule-descriptor.json"] = _make_descriptor("training", output, packet, receipt)
    return _write_capsule(output_parent, output_name, output)


def main(argv: list[str] | None = None) -> int:
    parser = _SanitizedArgumentParser(prog="python -m benchmarks.human_training")
    commands = parser.add_subparsers(
        dest="command", required=True, parser_class=_SanitizedArgumentParser
    )
    extract = commands.add_parser("extract-train-dev-embeddings")
    extract.add_argument("--train-dev-capsule", type=Path, required=True)
    extract.add_argument("--encoder-snapshot", type=Path, required=True)
    extract.add_argument("--output-parent", type=Path, required=True)
    extract.add_argument("--output-name", required=True)
    train = commands.add_parser("train-head")
    train.add_argument("--embedding-capsule", type=Path, required=True)
    train.add_argument("--encoder-snapshot", type=Path, required=True)
    train.add_argument("--output-parent", type=Path, required=True)
    train.add_argument("--output-name", required=True)
    try:
        args = parser.parse_args(argv)
        if args.command == "extract-train-dev-embeddings":
            extract_train_dev_embeddings(
                args.train_dev_capsule,
                args.encoder_snapshot,
                args.output_parent,
                args.output_name,
            )
        else:
            train_head(
                args.embedding_capsule,
                args.encoder_snapshot,
                args.output_parent,
                args.output_name,
            )
    except Exception:
        import sys

        print("GOVERNANCE_INVALID", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
