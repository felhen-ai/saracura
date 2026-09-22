"""Adversarial tests for the closed Phase 4B.2 training boundary.

All packet values are generated in process and are intentionally simulated
fixtures.  Nothing in this file is an author, reviewer, grant, human record,
encoder snapshot, or model weight.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest

import benchmarks.human_training as human_training
import saracura.backends.minilm as minilm
from benchmarks.first_party_packet import LABELS
from benchmarks.human_research import (
    HumanResearchError,
    digest,
    reconstruct_train_dev_descriptor,
    validate_packet_view_subset,
)
from benchmarks.human_research import main as research_main
from benchmarks.human_training import (
    _match_received_train_dev_descriptor,
    _validate_embedding_manifest,
)
from benchmarks.human_training import main as training_main
from saracura.research_trust import canonical
from saracura.verified_bytes import (
    MINILM_REVISION,
    MiniLMCandidate,
    VerifiedBytesError,
    load_verified_minilm_bytes,
    package_registry_sha256,
    validate_training_manifest,
)


def _json(value: dict[str, Any]) -> bytes:
    return canonical(value) + b"\n"


def _view(split: str, start: int, count_per_label: int) -> bytes:
    rows: list[bytes] = []
    number = start
    for label in LABELS:
        for _ in range(count_per_label):
            row = {
                "state_id": f"state_{number:032x}",
                "family_id": f"fam_{number:032x}",
                "locale": "pt-BR",
                "candidate_label": label,
                "text": f"fixture-only sealed record {number}",
                "content_digest": "a" * 64,
            }
            rows.append(_json(row))
            number += 1
    assert split in {"train", "dev", "calibration", "blind_test"}
    return b"".join(rows)


def _packet() -> tuple[bytes, dict[str, bytes]]:
    views = {
        "train": _view("train", 1, 40),
        "dev": _view("dev", 1_000, 20),
        "calibration": _view("calibration", 2_000, 20),
        "blind_test": _view("blind_test", 3_000, 20),
    }
    counts = {"train": 200, "dev": 100, "calibration": 100, "blind_test": 100}
    label_counts = {
        split: {label: count // len(LABELS) for label in LABELS} for split, count in counts.items()
    }
    packet: dict[str, Any] = {
        "schema_version": "support-routing-human-packet.v1",
        "protocol_sha256": "1" * 64,
        "policy_registry_sha256": "2" * 64,
        "states_sha256": "3" * 64,
        "split_plan_sha256": "4" * 64,
        "contributors_sha256": "5" * 64,
        "annotations_sha256": "6" * 64,
        "adjudications_sha256": "7" * 64,
        "controls_sha256": "8" * 64,
        "takedown_ledger_sha256": "9" * 64,
        "files": {
            split: {"bytes": len(raw), "sha256": digest(raw)} for split, raw in views.items()
        },
        "accepted_counts": {"total": 500, "by_split": counts, "by_split_label": label_counts},
        "excluded_counts": {
            "total": 0,
            "by_reason": {
                "backend_input_limit": 0,
                "review_rejected": 0,
                "review_disagreement_excluded": 0,
                "adjudication_excluded": 0,
            },
        },
        "labels": list(LABELS),
        "split_algorithm": "support-routing-stratified-sha256.v1",
        "source_type": "human_original",
        "locale": "pt-BR",
        "synthetic_only": False,
        "authorizations": {
            "training": False,
            "calibration": False,
            "blind_test": False,
            "automation": False,
            "broad_quality_claims": False,
        },
    }
    packet["packet_sha256"] = digest(canonical(packet))
    return _json(packet), views


def test_packet_train_dev_views_are_reconciled_against_the_closed_manifest() -> None:
    packet, views = _packet()
    checked = validate_packet_view_subset(packet, {"train": views["train"], "dev": views["dev"]})
    assert checked["accepted_counts"]["by_split"]["train"] == 200
    with pytest.raises(HumanResearchError, match="packet view ledger"):
        validate_packet_view_subset(packet, {"train": views["train"] + b" "})


def test_extraction_rejects_a_swapped_received_train_dev_descriptor() -> None:
    packet, _views = _packet()
    receipt = b"fixture receipt"
    source_descriptor = reconstruct_train_dev_descriptor(json.loads(packet), packet, receipt)
    values = {
        "packet-manifest.json": packet,
        "packet-release-receipt.json": receipt,
        "capsule-descriptor.json": source_descriptor,
    }
    matched_packet, matched_receipt = _match_received_train_dev_descriptor(values)
    assert matched_packet["packet_sha256"] == json.loads(packet)["packet_sha256"]
    assert matched_receipt == receipt
    values["capsule-descriptor.json"] = b"swapped descriptor"
    with pytest.raises(HumanResearchError, match="train/dev descriptor binding"):
        _match_received_train_dev_descriptor(values)


def _metrics() -> dict[str, Any]:
    matrix = [[1, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    per_label = {
        label: {
            "support": 1 if index == 0 else 0,
            "precision": 1.0 if index == 0 else 0.0,
            "recall": 1.0 if index == 0 else 0.0,
            "f1": 1.0 if index == 0 else 0.0,
        }
        for index, label in enumerate(LABELS)
    }
    metric = {
        "count": 1,
        "accuracy": 1.0,
        "macro_f1": 0.2,
        "confusion_matrix": matrix,
        "per_label": per_label,
    }
    return {
        "synthetic_only": False,
        "epochs": 1,
        "selected_epoch": 1,
        "epoch_ledger": [{"epoch": 1, "train_loss": 0.5, "dev_macro_f1": 0.2}],
        "train": metric,
        "dev": metric,
    }


def _human_manifest(
    checkpoint: bytes, logits: list[float], *, mps_logits: list[float] | None = None
) -> bytes:
    mps = logits if mps_logits is None else mps_logits
    conformance: dict[str, Any] = {
        "schema": "human-minilm-conformance.v1",
        "token_payload_sha256": "c" * 64,
        "labels": list(LABELS),
        "cpu_logits": logits,
        "mps_logits": mps,
        "cpu_logits_sha256": hashlib.sha256(canonical(logits)).hexdigest(),
        "mps_logits_sha256": hashlib.sha256(canonical(mps)).hexdigest(),
        "comparison": {"rtol": 0.00001, "atol": 0.000001},
    }
    manifest: dict[str, Any] = {
        "schema_version": "human-training-manifest.v1",
        "workflow_revision": "phase4b-human-research-training.v1",
        "synthetic_only": False,
        "packet_manifest_sha256": "1" * 64,
        "packet_release_receipt_sha256": "2" * 64,
        "embedding_manifest_sha256": "3" * 64,
        "training_code_sha256": "4" * 64,
        "protocol_sha256": "5" * 64,
        "encoder": {
            "candidate": "multilingual-minilm-l12",
            "revision": MINILM_REVISION,
            "registry_sha256": package_registry_sha256(),
            "device": "mps",
            "frozen": True,
        },
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
            "embeddings.safetensors": "6" * 64,
            "checkpoint.safetensors": hashlib.sha256(checkpoint).hexdigest(),
        },
        "metrics": _metrics(),
        "environment": {
            "python": "3.11.0",
            "torch": "2.5.0",
            "transformers": "4.47.0",
            "tokenizers": "0.21.0",
            "safetensors": "0.5.0",
            "platform": "fixture",
        },
        "serving_conformance": conformance,
        "authorizations": {"calibration": True, "automation": False, "quality_claims": False},
    }
    manifest["manifest_sha256"] = hashlib.sha256(canonical(manifest)).hexdigest()
    return _json(manifest)


def _rehash_manifest(value: dict[str, Any]) -> bytes:
    conformance = value["serving_conformance"]
    conformance["cpu_logits_sha256"] = hashlib.sha256(
        canonical(conformance["cpu_logits"])
    ).hexdigest()
    conformance["mps_logits_sha256"] = hashlib.sha256(
        canonical(conformance["mps_logits"])
    ).hexdigest()
    value["manifest_sha256"] = hashlib.sha256(
        canonical({key: child for key, child in value.items() if key != "manifest_sha256"})
    ).hexdigest()
    return _json(value)


def test_human_manifest_rejects_the_zero_logit_sentinel() -> None:
    checkpoint = b"fixture checkpoint"
    with pytest.raises(ValueError, match="human serving conformance"):
        validate_training_manifest(
            _human_manifest(checkpoint, [0.0] * 5),
            checkpoint_sha256=hashlib.sha256(checkpoint).hexdigest(),
        )
    assert (
        validate_training_manifest(
            _human_manifest(checkpoint, [0.125, -0.25, 0.5, 0.75, 1.0]),
            checkpoint_sha256=hashlib.sha256(checkpoint).hexdigest(),
        )["schema_version"]
        == "human-training-manifest.v1"
    )


@pytest.mark.parametrize(
    ("cpu_logits", "mps_logits"),
    [
        ([0.0] * 5, [0.125, -0.25, 0.5, 0.75, 1.0]),
        ([0.125, -0.25, 0.5, 0.75, 1.0], [0.0] * 5),
        ([0.125, -0.25, 0.5, 0.75, 1.0], [0.125, -0.25, 0.5, 0.75, 1.1]),
    ],
)
def test_human_manifest_rejects_each_logit_sentinel_and_mps_mismatch(
    cpu_logits: list[float], mps_logits: list[float]
) -> None:
    checkpoint = b"fixture checkpoint"
    with pytest.raises(ValueError, match="human serving conformance"):
        validate_training_manifest(
            _human_manifest(checkpoint, cpu_logits, mps_logits=mps_logits),
            checkpoint_sha256=hashlib.sha256(checkpoint).hexdigest(),
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda manifest: manifest["metrics"]["train"]["per_label"]["billing"].update(
            {"precision": 0.5}
        ),
        lambda manifest: manifest["metrics"]["dev"].update({"accuracy": 0.5}),
        lambda manifest: manifest["metrics"]["dev"].update({"macro_f1": 0.5}),
    ],
)
def test_human_manifest_recomputes_all_metric_claims_from_matrix(mutation: Any) -> None:
    checkpoint = b"fixture checkpoint"
    manifest = json.loads(_human_manifest(checkpoint, [0.125, -0.25, 0.5, 0.75, 1.0]))
    mutation(manifest)
    with pytest.raises(ValueError, match="metric reconciliation"):
        validate_training_manifest(
            _rehash_manifest(manifest), checkpoint_sha256=hashlib.sha256(checkpoint).hexdigest()
        )


def test_human_manifest_rejects_selected_epoch_metric_mismatch() -> None:
    checkpoint = b"fixture checkpoint"
    manifest = json.loads(_human_manifest(checkpoint, [0.125, -0.25, 0.5, 0.75, 1.0]))
    manifest["metrics"]["epoch_ledger"][0]["dev_macro_f1"] = 0.1
    with pytest.raises(ValueError, match="selected metric"):
        validate_training_manifest(
            _rehash_manifest(manifest), checkpoint_sha256=hashlib.sha256(checkpoint).hexdigest()
        )


def test_human_runtime_rejects_loose_manifest_checkpoint_before_snapshot(tmp_path: Path) -> None:
    """A human manifest can never use the legacy Phase 4A pair entry point."""

    checkpoint = b"fixture checkpoint"
    manifest = tmp_path / "training-manifest.json"
    checkpoint_path = tmp_path / "checkpoint.safetensors"
    manifest.write_bytes(_human_manifest(checkpoint, [0.125, -0.25, 0.5, 0.75, 1.0]))
    checkpoint_path.write_bytes(checkpoint)
    os.chmod(manifest, 0o600)
    os.chmod(checkpoint_path, 0o600)
    os.chmod(tmp_path, 0o700)
    with pytest.raises(VerifiedBytesError, match="sealed training capsule"):
        load_verified_minilm_bytes(
            encoder_snapshot=tmp_path / "unread-snapshot",
            training_manifest=manifest,
            checkpoint=checkpoint_path,
        )


def _invalid_receipt_capsule(tmp_path: Path) -> Path:
    """Create an ephemeral closed shape with a deliberately unsigned receipt."""

    capsule = tmp_path / "training"
    capsule.mkdir(mode=0o700)
    checkpoint = b"fixture checkpoint"
    packet, _views = _packet()
    packet_value = json.loads(packet)
    report = _json({"schema_version": "human-training-report.v1", "metrics": _metrics()})
    payload = {
        "checkpoint.safetensors": checkpoint,
        "training-manifest.json": _human_manifest(checkpoint, [0.125, -0.25, 0.5, 0.75, 1.0]),
        "training-report.json": report,
        "packet-manifest.json": packet,
        "packet-release-receipt.json": b"unsigned fixture receipt\n",
    }
    descriptor: dict[str, Any] = {
        "schema_version": "human-capsule.v1",
        "capsule_kind": "training",
        "packet_sha256": packet_value["packet_sha256"],
        "packet_release_receipt_sha256": digest(payload["packet-release-receipt.json"]),
        "files": [
            {"path": name, "bytes": len(raw), "sha256": digest(raw)}
            for name, raw in sorted(payload.items())
        ],
    }
    descriptor["descriptor_sha256"] = digest(canonical(descriptor))
    payload["capsule-descriptor.json"] = _json(descriptor)
    for name, raw in payload.items():
        target = capsule / name
        target.write_bytes(raw)
        os.chmod(target, 0o600)
    return capsule


def test_human_runtime_requires_a_valid_packet_training_receipt(tmp_path: Path) -> None:
    with pytest.raises(VerifiedBytesError, match="packet receipt"):
        load_verified_minilm_bytes(
            encoder_snapshot=tmp_path / "unread-snapshot",
            training_capsule=_invalid_receipt_capsule(tmp_path),
        )


def test_training_cli_never_echoes_an_operator_path(capsys: pytest.CaptureFixture[str]) -> None:
    assert training_main(["train-head", "--embedding-capsule", "/private/operator/path"]) == 2
    captured = capsys.readouterr()
    assert captured.err == "GOVERNANCE_INVALID\n"
    assert "/private/operator/path" not in captured.err


@pytest.mark.parametrize("exception", [RuntimeError("/private/torch/failure"), RuntimeError("MPS")])
def test_training_cli_sanitizes_optional_ml_failures(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], exception: RuntimeError
) -> None:
    def fail(*_args: object, **_kwargs: object) -> Path:
        raise exception

    monkeypatch.setattr(human_training, "train_head", fail)
    assert (
        training_main(
            [
                "train-head",
                "--embedding-capsule",
                "/private/input",
                "--encoder-snapshot",
                "/private/snapshot",
                "--output-parent",
                "/private/output",
                "--output-name",
                "sealed",
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert captured.err == "GOVERNANCE_INVALID\n"
    assert "/private/torch/failure" not in captured.err


def test_human_research_dispatch_sanitizes_transformers_failures(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail(*_args: object, **_kwargs: object) -> Path:
        raise RuntimeError("/private/transformers/failure")

    monkeypatch.setattr(human_training, "train_head", fail)
    assert (
        research_main(
            [
                "train-head",
                "--embedding-capsule",
                "/private/input",
                "--encoder-snapshot",
                "/private/snapshot",
                "--output-parent",
                "/private/output",
                "--output-name",
                "sealed",
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert captured.err == "GOVERNANCE_INVALID\n"
    assert "/private/transformers/failure" not in captured.err


def test_training_cli_preserves_keyboard_interrupt(monkeypatch: pytest.MonkeyPatch) -> None:
    def interrupt(*_args: object, **_kwargs: object) -> Path:
        raise KeyboardInterrupt

    monkeypatch.setattr(human_training, "train_head", interrupt)
    with pytest.raises(KeyboardInterrupt):
        training_main(
            [
                "train-head",
                "--embedding-capsule",
                "input",
                "--encoder-snapshot",
                "snapshot",
                "--output-parent",
                "output",
                "--output-name",
                "sealed",
            ]
        )


def test_training_cli_preserves_system_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    def exit_now(*_args: object, **_kwargs: object) -> Path:
        raise SystemExit(17)

    monkeypatch.setattr(human_training, "train_head", exit_now)
    with pytest.raises(SystemExit, match="17"):
        training_main(
            [
                "train-head",
                "--embedding-capsule",
                "input",
                "--encoder-snapshot",
                "snapshot",
                "--output-parent",
                "output",
                "--output-name",
                "sealed",
            ]
        )


def _candidate_for_conformance() -> MiniLMCandidate:
    return MiniLMCandidate(
        id="multilingual-minilm-l12",
        repository="fixture-only",
        revision=MINILM_REVISION,
        architecture="fixture-only",
        hidden_width=384,
        tokenizer="fixture-only",
        files=(),
    )


def _expected_encoder() -> dict[str, Any]:
    return {
        "candidate": "multilingual-minilm-l12",
        "revision": MINILM_REVISION,
        "registry_sha256": package_registry_sha256(),
        "device": "mps",
        "frozen": True,
    }


def test_human_conformance_rejects_snapshot_manifest_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        minilm, "read_verified_minilm_snapshot", lambda _path: (_candidate_for_conformance(), {})
    )
    expected = _expected_encoder()
    expected["revision"] = "wrong"
    with pytest.raises(ValueError, match="snapshot binding"):
        minilm.human_checkpoint_conformance(
            encoder_snapshot=Path("explicit-snapshot"),
            checkpoint=b"fixture",
            expected_encoder=expected,
        )


@pytest.mark.parametrize(
    ("cpu_logits", "mps_logits", "message"),
    [
        ([0.0] * 5, [0.1] * 5, "sentinel"),
        ([0.1] * 5, [0.0] * 5, "sentinel"),
        ([0.1] * 5, [0.1, 0.1, 0.1, 0.1, 0.2], "equivalence"),
    ],
)
def test_human_conformance_rejects_sentinel_and_cross_device_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    cpu_logits: list[float],
    mps_logits: list[float],
    message: str,
) -> None:
    pytest.importorskip("torch")
    monkeypatch.setattr(
        minilm, "read_verified_minilm_snapshot", lambda _path: (_candidate_for_conformance(), {})
    )

    def fake_conformance(
        _snapshot: dict[str, bytes], _checkpoint: bytes, device: str
    ) -> tuple[dict[str, Any], list[float]]:
        return {"input_ids": [[1]], "attention_mask": [[1]], "token_type_ids": [[0]]}, (
            cpu_logits if device == "cpu" else mps_logits
        )

    monkeypatch.setattr(minilm, "_human_conformance_on", fake_conformance)
    with pytest.raises(ValueError, match=message):
        minilm.human_checkpoint_conformance(
            encoder_snapshot=Path("explicit-snapshot"),
            checkpoint=b"fixture",
            expected_encoder=_expected_encoder(),
        )


def test_opt_in_human_checkpoint_cpu_mps_conformance_uses_only_explicit_paths() -> None:
    """Run only when an operator deliberately supplies a sealed local capsule."""

    if os.environ.get("SARACURA_PHASE4B_LIVE_MPS") != "1":
        pytest.skip("set SARACURA_PHASE4B_LIVE_MPS=1 with local operator artifact paths")
    required = ("SARACURA_PHASE4B_SNAPSHOT", "SARACURA_PHASE4B_TRAINING_CAPSULE")
    if any(not os.environ.get(name) for name in required):
        pytest.fail("local human CPU/MPS conformance inputs were not provided")
    verified = load_verified_minilm_bytes(
        encoder_snapshot=Path(os.environ["SARACURA_PHASE4B_SNAPSHOT"]),
        training_capsule=Path(os.environ["SARACURA_PHASE4B_TRAINING_CAPSULE"]),
    )
    conformance = minilm.human_checkpoint_conformance(
        encoder_snapshot=Path(os.environ["SARACURA_PHASE4B_SNAPSHOT"]),
        checkpoint=verified.checkpoint_bytes,
        expected_encoder=verified.training_manifest["encoder"],
    )
    assert any(conformance["cpu_logits"])
    assert any(conformance["mps_logits"])


def test_embedding_manifest_reconciles_tensor_rows_and_labels() -> None:
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")
    packet, views = _packet()
    records: list[dict[str, Any]] = []
    labels: list[int] = []
    splits: list[int] = []
    for split, raw, split_number in (("train", views["train"], 0), ("dev", views["dev"], 1)):
        for row in raw.splitlines():
            value = json.loads(row)
            label = LABELS.index(value["candidate_label"])
            records.append(
                {
                    "split": split,
                    "state_id": value["state_id"],
                    "family_id": value["family_id"],
                    "label_index": label,
                    "row_index": len(records),
                }
            )
            labels.append(label)
            splits.append(split_number)
    tensor_raw = safetensors.save(
        {
            "embeddings": torch.ones((len(records), 384), dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "splits": torch.tensor(splits, dtype=torch.uint8),
        }
    )
    receipt = b"fixture receipt"
    descriptor = b"different source descriptor"
    train_dev_descriptor = reconstruct_train_dev_descriptor(json.loads(packet), packet, receipt)
    manifest: dict[str, Any] = {
        "schema_version": "human-embedding-manifest.v1",
        "packet_manifest_sha256": digest(packet),
        "packet_release_receipt_sha256": digest(receipt),
        "train_dev_capsule_sha256": digest(train_dev_descriptor),
        "protocol_sha256": "1" * 64,
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
        "records": records,
        "files": {
            "embeddings.safetensors": {"bytes": len(tensor_raw), "sha256": digest(tensor_raw)}
        },
        "environment": {
            "python": "3.11.0",
            "torch": "2.5.0",
            "transformers": "4.47.0",
            "tokenizers": "0.21.0",
            "safetensors": "0.5.0",
            "platform": "fixture",
        },
    }
    manifest["manifest_sha256"] = digest(canonical(manifest))
    raw = _json(manifest)
    _validate_embedding_manifest(
        raw,
        tensor_raw=tensor_raw,
        packet_raw=packet,
        receipt_raw=receipt,
        descriptor_raw=descriptor,
    )
    forged_packet = json.loads(packet)
    forged_packet["files"]["train"] = {
        "bytes": forged_packet["files"]["train"]["bytes"] + 1,
        "sha256": "f" * 64,
    }
    forged_packet["packet_sha256"] = digest(
        canonical({key: value for key, value in forged_packet.items() if key != "packet_sha256"})
    )
    forged_raw = _json(forged_packet)
    manifest["packet_manifest_sha256"] = digest(forged_raw)
    manifest["manifest_sha256"] = digest(
        canonical({key: value for key, value in manifest.items() if key != "manifest_sha256"})
    )
    with pytest.raises(HumanResearchError, match="train/dev descriptor binding"):
        _validate_embedding_manifest(
            _json(manifest),
            tensor_raw=tensor_raw,
            packet_raw=forged_raw,
            receipt_raw=receipt,
            descriptor_raw=descriptor,
        )
    manifest["packet_manifest_sha256"] = digest(packet)
    swapped_receipt = b"different fixture receipt"
    manifest["packet_release_receipt_sha256"] = digest(swapped_receipt)
    manifest["manifest_sha256"] = digest(
        canonical({key: value for key, value in manifest.items() if key != "manifest_sha256"})
    )
    with pytest.raises(HumanResearchError, match="train/dev descriptor binding"):
        _validate_embedding_manifest(
            _json(manifest),
            tensor_raw=tensor_raw,
            packet_raw=packet,
            receipt_raw=swapped_receipt,
            descriptor_raw=descriptor,
        )
    manifest["packet_release_receipt_sha256"] = digest(receipt)
    records[0]["label_index"] = 4
    manifest["manifest_sha256"] = digest(
        canonical({key: value for key, value in manifest.items() if key != "manifest_sha256"})
    )
    with pytest.raises(HumanResearchError, match="embedding records"):
        _validate_embedding_manifest(
            _json(manifest),
            tensor_raw=tensor_raw,
            packet_raw=packet,
            receipt_raw=receipt,
            descriptor_raw=descriptor,
        )
