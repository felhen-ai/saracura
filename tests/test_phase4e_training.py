"""Adversarial offline checks for the Phase 4E.2c training capsule."""

from __future__ import annotations

import importlib.util
import json
import os
import socket
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from benchmarks import saracura_universal_training as training
from benchmarks.encoder_loader import LoadedEncoder
from benchmarks.encoder_registry import get_candidate
from benchmarks.saracura_universal_policy import Phase4EPolicyError
from benchmarks.saracura_universal_policy import (
    require_phase4e_authorization as real_require_phase4e_authorization,
)

pytestmark = pytest.mark.skipif(
    not all(importlib.util.find_spec(name) is not None for name in ("torch", "safetensors")),
    reason="requires local-minilm",
)


@pytest.fixture(autouse=True)
def historical_training_lane(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise the frozen trainer below the current post-pilot policy gate."""

    monkeypatch.setattr(training, "require_phase4e_authorization", lambda _action: {})


def _torch() -> Any:
    return pytest.importorskip("torch")


def _safe() -> Any:
    return pytest.importorskip("safetensors.torch")


class _Tokenizer:
    def __call__(self, texts: list[str], **kwargs: object) -> dict[str, list[list[int]]]:
        assert kwargs == {"truncation": False, "padding": False, "add_special_tokens": True}
        return {
            "input_ids": [
                [101, *[3 + byte % 91 for byte in text.encode()[:8]], 102] for text in texts
            ]
        }


class _Model:
    def to(self, device: object) -> _Model:
        del device
        return self

    def eval(self) -> None:
        return None

    def __call__(self, *, input_ids: Any, attention_mask: Any, return_dict: bool) -> object:
        assert return_dict is True
        torch = _torch()
        hidden = torch.zeros((*input_ids.shape, 384), dtype=torch.float32)
        hidden[..., 0] = input_ids
        hidden[..., 1] = attention_mask
        return type("Output", (), {"last_hidden_state": hidden})()


def _task(split: str, serial: int, locale: str, count: int) -> dict[str, Any]:
    criteria = [
        {"id": f"criterion-{serial}-{pos}", "description": f"Option {serial} {pos}"}
        for pos in range(count)
    ]
    return {
        "task_id": f"task-{split}-{serial}",
        "family_id": f"family-{split}-{serial}",
        "pair_id": None,
        "split": split,
        "locale": locale,
        "domain": "email_triage",
        "axes": {
            "explicitness": "explicit",
            "negation": "absent",
            "distractor_overlap": "low",
            "urgency": "normal",
        },
        "instruction": f"Choose {serial}",
        "state": {"item": str(serial)},
        "criteria": criteria,
        "selected_criterion_id": criteria[0]["id"],
    }


def _metric(rows: training._Rows, score: float) -> dict[str, Any]:
    cells = {f"{locale}:{count}": score for locale, count in training._CELLS}
    return {
        "accuracy": score,
        "stratified_macro_accuracy": score,
        "cells": cells,
        "predictions": [0] * len(rows.rows),
    }


def _fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, Path, training.AcceptedPacketBinding]:
    packet = tmp_path / "packet"
    packet.mkdir()
    tasks = [
        _task(split, offset + index, locale, count)
        for split, offset in (
            ("synthetic_train", 0),
            ("synthetic_dev", 100),
            ("synthetic_holdout", 200),
        )
        for index, (locale, count) in enumerate(training._CELLS)
    ]
    train_dev = [row for row in tasks if row["split"] != "synthetic_holdout"]
    holdout = [row for row in tasks if row["split"] == "synthetic_holdout"]
    train_raw = b"".join(training._canonical(row) + b"\n" for row in train_dev)
    holdout_raw = b"".join(training._canonical(row) + b"\n" for row in holdout)
    identities_raw = (
        training._canonical(
            {
                "schema_version": "phase4e-holdout-identities.v1",
                "rows": list(training._identity_rows(holdout)),
            }
        )
        + b"\n"
    )
    (packet / "accepted-train-dev.jsonl").write_bytes(train_raw)
    (packet / "accepted-holdout.jsonl").write_bytes(holdout_raw)
    (packet / "holdout-identities.json").write_bytes(identities_raw)
    (packet / "packet.json").write_bytes(
        training._canonical(
            {
                "schema_version": "fixture",
                "files": {"accepted-holdout.jsonl": training._sha(holdout_raw)},
            }
        )
    )
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "snapshot.complete.json").write_bytes(b'{"fixture":"v1"}\n')
    candidate = get_candidate("multilingual-minilm-l12")
    monkeypatch.setattr(
        training,
        "_load_verified_minilm",
        lambda _: LoadedEncoder(_Model(), _Tokenizer(), candidate, snapshot),
    )
    binding = training.AcceptedPacketBinding(
        "a" * 64,
        training._sha(train_raw),
        training._sha(holdout_raw),
        training._sha(identities_raw),
        training._accepted_identity_digest(tasks),
        training._identity_rows(train_dev),
        training._identity_rows(holdout),
        training._identity_rows(tasks),
    )
    monkeypatch.setattr(
        training, "validate_accepted_packet_binding", lambda _packet, **_kwargs: binding
    )
    monkeypatch.setattr(training, "validate_accepted_packet_holdout_bytes", lambda *_args: None)
    return tmp_path / "capsule", packet, snapshot, binding


def _extract(capsule: Path, packet: Path, snapshot: Path) -> Path:
    return training.extract_and_seal_embeddings(packet, snapshot, "cpu", capsule)


def _passing_run(monkeypatch: pytest.MonkeyPatch) -> None:
    checkpoint = _safe().save(training._new_model(_torch()).checkpoint())

    def run(
        _train: training._Rows, dev: training._Rows, _policy: dict[str, Any]
    ) -> tuple[bytes, list[dict[str, Any]], dict[str, Any]]:
        metric = _metric(dev, 1.0)
        return checkpoint, [{"epoch": 1, "loss": 0.0, "dev": metric}], {"epoch": 1, "dev": metric}

    monkeypatch.setattr(training, "_run_once", run)
    monkeypatch.setattr(training, "_baseline", lambda rows, **kwargs: _metric(rows, 0.0))


def test_extract_is_holdout_blind_and_has_exact_v3_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capsule, packet, snapshot, binding = _fixture(tmp_path, monkeypatch)
    holdout = (packet / "accepted-holdout.jsonl").resolve()
    original = Path.read_bytes

    def guard(path: Path) -> bytes:
        if path.resolve() == holdout:
            raise AssertionError("extract opened holdout")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", guard)
    _extract(capsule, packet, snapshot)
    assert {item.name for item in capsule.iterdir()} == training._CAPSULE_FILES
    assert not list(capsule.glob("*holdout*"))
    assert (
        training.validate_embedding_capsule(capsule, binding).manifest["schema_version"]
        == "phase4e-embedding-capsule.v3"
    )


def test_extract_no_clobber_and_cpu_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    capsule, packet, snapshot, _ = _fixture(tmp_path, monkeypatch)
    _extract(capsule, packet, snapshot)
    with pytest.raises(FileExistsError):
        _extract(capsule, packet, snapshot)
    with pytest.raises(training.TrainingError, match="CPU-only"):
        training.extract_and_seal_embeddings(packet, snapshot, "mps", tmp_path / "mps")


def test_dev_failure_seals_without_claim_or_holdout_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capsule, packet, snapshot, _ = _fixture(tmp_path, monkeypatch)
    _extract(capsule, packet, snapshot)
    checkpoint = _safe().save(training._new_model(_torch()).checkpoint())
    monkeypatch.setattr(
        training,
        "_run_once",
        lambda _train, dev, _policy: (
            checkpoint,
            [{"epoch": 1, "loss": 1.0, "dev": _metric(dev, 0.0)}],
            {"epoch": 1, "dev": _metric(dev, 0.0)},
        ),
    )
    registry = tmp_path / "registry"
    monkeypatch.setattr(training, "_holdout_release_registry_directory", lambda: registry)
    holdout = (packet / "accepted-holdout.jsonl").resolve()
    original = Path.read_bytes
    monkeypatch.setattr(
        Path,
        "read_bytes",
        lambda path: (
            (_ for _ in ()).throw(AssertionError("opened holdout"))
            if path.resolve() == holdout
            else original(path)
        ),
    )
    output = training.train_and_seal(capsule, packet, snapshot, "cpu", tmp_path / "out", "failed")
    assert training.verify_training_capsule(output)["outcome"] == "pre_holdout_failed"
    assert not registry.exists() and "holdout-report.json" not in {
        item.name for item in output.iterdir()
    }


def test_postclaim_derives_holdout_in_memory_and_seals_conformance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capsule, packet, snapshot, _ = _fixture(tmp_path, monkeypatch)
    _extract(capsule, packet, snapshot)
    _passing_run(monkeypatch)
    registry = tmp_path / "registry"
    monkeypatch.setattr(training, "_holdout_release_registry_directory", lambda: registry)
    original = training.derive_holdout_embeddings_in_memory

    def after_claim(*args: object, **kwargs: object) -> training._Rows:
        claims = list(registry.glob("*.json"))
        assert len(claims) == 1
        claim = json.loads(claims[0].read_bytes())
        assert claim["checkpoint_sha256"] == args[5]
        assert claim["pre_holdout_gate_descriptor_sha256"] == args[6]
        return original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(training, "derive_holdout_embeddings_in_memory", after_claim)
    output = training.train_and_seal(capsule, packet, snapshot, "cpu", tmp_path / "out", "passed")
    assert training.verify_training_capsule(output)["outcome"] in {"passed", "holdout_failed"}
    assert {
        "conformance-vectors.safetensors",
        "conformance-manifest.json",
        "dependency-versions.json",
    } <= {item.name for item in output.iterdir()}


def test_direct_holdout_derivation_requires_exact_immutable_claim_before_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capsule_path, packet, snapshot, binding = _fixture(tmp_path, monkeypatch)
    _extract(capsule_path, packet, snapshot)
    capsule = training.validate_embedding_capsule(capsule_path, binding)
    registry = tmp_path / "registry"
    monkeypatch.setattr(training, "_holdout_release_registry_directory", lambda: registry)
    checkpoint_sha256 = "b" * 64
    gate_descriptor_sha256 = "c" * 64
    holdout = (packet / "accepted-holdout.jsonl").resolve()
    original = Path.read_bytes
    reads: list[Path] = []

    def guard(path: Path) -> bytes:
        if path.resolve() == holdout:
            reads.append(path)
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", guard)
    with pytest.raises(training.TrainingError, match="holdout release claim"):
        training.derive_holdout_embeddings_in_memory(
            capsule,
            packet,
            binding,
            snapshot,
            "cpu",
            checkpoint_sha256,
            gate_descriptor_sha256,
        )
    assert not reads

    training._claim_holdout_once(binding, capsule, checkpoint_sha256, gate_descriptor_sha256)
    with pytest.raises(training.TrainingError, match="holdout release claim binding"):
        training.derive_holdout_embeddings_in_memory(
            capsule,
            packet,
            binding,
            snapshot,
            "cpu",
            "d" * 64,
            gate_descriptor_sha256,
        )
    assert not reads

    claim = registry / f"{training._holdout_release_claim_key(binding.packet_json_sha256)}.json"
    claim.write_bytes(
        training._holdout_release_payload(binding, capsule, checkpoint_sha256, "e" * 64)
    )
    with pytest.raises(training.TrainingError, match="holdout release claim binding"):
        training.derive_holdout_embeddings_in_memory(
            capsule,
            packet,
            binding,
            snapshot,
            "cpu",
            checkpoint_sha256,
            gate_descriptor_sha256,
        )
    assert not reads

    claim.write_bytes(
        training._holdout_release_payload(
            binding, capsule, checkpoint_sha256, gate_descriptor_sha256
        )
    )
    holdout_rows = training.derive_holdout_embeddings_in_memory(
        capsule,
        packet,
        binding,
        snapshot,
        "cpu",
        checkpoint_sha256,
        gate_descriptor_sha256,
    )
    assert holdout_rows.rows and reads


def test_packet_only_claim_blocks_different_capsule_before_holdout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first, packet, snapshot, _ = _fixture(tmp_path, monkeypatch)
    _extract(first, packet, snapshot)
    second = tmp_path / "second-capsule"
    _extract(second, packet, snapshot)
    _passing_run(monkeypatch)
    registry = tmp_path / "registry"
    monkeypatch.setattr(training, "_holdout_release_registry_directory", lambda: registry)
    training.train_and_seal(first, packet, snapshot, "cpu", tmp_path / "one", "sealed")
    holdout = (packet / "accepted-holdout.jsonl").resolve()
    original = Path.read_bytes
    monkeypatch.setattr(
        Path,
        "read_bytes",
        lambda path: (
            (_ for _ in ()).throw(AssertionError("second opened holdout"))
            if path.resolve() == holdout
            else original(path)
        ),
    )
    with pytest.raises(training.TrainingError, match="already released"):
        training.train_and_seal(second, packet, snapshot, "cpu", tmp_path / "two", "sealed")
    assert len(list(registry.glob("*.json"))) == 1


def test_conformance_mutation_dependency_schema_and_socket_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capsule, packet, snapshot, _ = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        socket,
        "socket",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network attempted")),
    )
    _extract(capsule, packet, snapshot)
    _passing_run(monkeypatch)
    monkeypatch.setattr(
        training, "_holdout_release_registry_directory", lambda: tmp_path / "registry"
    )
    output = training.train_and_seal(capsule, packet, snapshot, "cpu", tmp_path / "out", "sealed")
    vectors = _safe().load((output / "conformance-vectors.safetensors").read_bytes())
    vectors["expected_logits"][0, 0] += 1
    vector_raw = _safe().save(vectors)
    (output / "conformance-vectors.safetensors").write_bytes(vector_raw)
    manifest = json.loads((output / "conformance-manifest.json").read_bytes())
    manifest["tensor_sha256"] = training._sha(vector_raw)
    manifest["manifest_sha256"] = training._sha(
        training._canonical(
            {key: value for key, value in manifest.items() if key != "manifest_sha256"}
        )
    )
    (output / "conformance-manifest.json").write_bytes(training._canonical(manifest) + b"\n")
    files = {
        name: (output / name).read_bytes()
        for name in training._FINAL_OUTPUT_FILES
        if name != "capsule-descriptor.json"
    }
    (output / "capsule-descriptor.json").write_bytes(training._descriptor(files, kind="training"))
    with pytest.raises(training.TrainingError, match="conformance logits"):
        training.verify_training_capsule(output)
    dependencies = json.loads((output / "dependency-versions.json").read_bytes())
    assert set(dependencies["packages"]) == {
        "torch",
        "safetensors",
        "numpy",
        "transformers",
        "tokenizers",
    }


def test_v4_report_binding_requires_latest_sealed_matching_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packet = b'{"schema_version":"phase4e-accepted-packet.v4"}\n'
    reports = tmp_path / "reports"
    reports.mkdir()
    payload = {
        "schema_version": "phase4e-corpus-terminal-report.v2",
        "outcome": "sealed",
        "packet_manifest_sha256": training._sha(packet),
        "recovery_plan_sha256": "1" * 64,
        "supplemental_resolution_sha256": "2" * 64,
        "supplemental_ledger_sha256": "3" * 64,
        "research_inventory_sha256": "4" * 64,
        "accepted_count": 1552,
        "rejected_count": 148,
        "unresolved_count": 0,
        "counts": {"locale": {"pt-BR": 1060, "en": 640}},
    }
    selected = reports / "report-0000.json"
    selected.write_bytes(training._canonical(payload) + b"\n")
    monkeypatch.setattr(training, "_V4_PACKET_MANIFEST_SHA256", training._sha(packet))
    monkeypatch.setattr(training, "_V4_RECOVERY_PLAN_SHA256", "1" * 64)
    monkeypatch.setattr(training, "_V4_SUPPLEMENTAL_RESOLUTION_SHA256", "2" * 64)
    monkeypatch.setattr(training, "_V4_SUPPLEMENTAL_LEDGER_SHA256", "3" * 64)
    monkeypatch.setattr(training, "_V4_RESEARCH_INVENTORY_SHA256", "4" * 64)
    monkeypatch.setattr(training, "_V4_SEALED_REPORT_SHA256", training._sha(selected.read_bytes()))
    binding = training._validate_v4_report_binding(packet, reports)
    assert binding.path == selected
    (reports / "report-0001.json").write_bytes(
        training._canonical({**payload, "outcome": "minimum_failed"}) + b"\n"
    )
    with pytest.raises(training.TrainingError, match="packet-v4 sealed report"):
        training._validate_v4_report_binding(packet, reports)


def test_v4_accepted_locale_counts_are_independent_from_report_totals() -> None:
    accepted = [
        *({"locale": "pt-BR"} for _ in range(943)),
        *({"locale": "en"} for _ in range(609)),
    ]
    training._require_v4_accepted_locale_counts(accepted)
    accepted[-1] = {"locale": "pt-BR"}
    with pytest.raises(training.TrainingError, match="packet-v4 accepted locale counts"):
        training._require_v4_accepted_locale_counts(accepted)


def test_v4_manifest_layout_precedes_every_nonmanifest_packet_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packet = tmp_path / "arbitrary-packet"
    packet.mkdir()
    manifest = packet / "packet.json"
    manifest.write_bytes(training._canonical({"schema_version": "phase4e-accepted-packet.v4"}))
    for name in ("plan.json", "accepted-train-dev.jsonl", "ledger.json", "holdout-identities.json"):
        (packet / name).write_bytes(b"must not be read\n")
    monkeypatch.setattr(
        training,
        "_v4_artifact_paths",
        lambda: training.V4ArtifactPaths(
            tmp_path / "canonical" / "accepted-packet-v4",
            tmp_path / "canonical" / "embedding-capsule-v1",
            tmp_path / "canonical" / "training-capsule-v1",
            tmp_path / "canonical" / "reports" / "ptbr-recovery-v1",
        ),
    )
    original = Path.read_bytes
    reads: list[Path] = []

    def record(path: Path) -> bytes:
        if path.parent == packet:
            reads.append(path)
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", record)
    with pytest.raises(training.TrainingError, match="packet-v4 packet path"):
        training.validate_accepted_packet_binding(packet)
    assert reads == [manifest]


def test_rehashed_v4_encoder_receipt_tampering_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capsule, packet, snapshot, binding = _fixture(tmp_path, monkeypatch)
    _extract(capsule, packet, snapshot)
    expected = {
        "registry_sha256": "a" * 64,
        "candidate": get_candidate("multilingual-minilm-l12").model_dump(mode="json"),
        "uv_lock_sha256": "b" * 64,
    }
    v4_binding = training.AcceptedPacketBinding(
        binding.packet_json_sha256,
        binding.accepted_train_dev_jsonl_sha256,
        binding.accepted_holdout_jsonl_sha256,
        binding.holdout_identities_json_sha256,
        binding.accepted_rows_sha256,
        binding.train_dev_identities,
        binding.holdout_identities,
        binding.identities,
        sealed_report_sha256="c" * 64,
    )

    def reseal(receipt: dict[str, object]) -> None:
        receipt_raw = training._canonical(receipt) + b"\n"
        (capsule / "accepted-packet-receipt.json").write_bytes(receipt_raw)
        manifest = json.loads((capsule / "embedding-manifest.json").read_bytes())
        manifest["packet_receipt_sha256"] = training._sha(receipt_raw)
        manifest["manifest_sha256"] = training._sha(
            training._canonical(
                {key: value for key, value in manifest.items() if key != "manifest_sha256"}
            )
        )
        (capsule / "embedding-manifest.json").write_bytes(training._canonical(manifest) + b"\n")
        files = {
            name: (capsule / name).read_bytes()
            for name in training._CAPSULE_FILES
            if name != "capsule-descriptor.json"
        }
        (capsule / "capsule-descriptor.json").write_bytes(
            training._descriptor(files, kind="embedding")
        )

    receipt = json.loads((capsule / "accepted-packet-receipt.json").read_bytes())
    receipt["sealed_report_sha256"] = v4_binding.sealed_report_sha256
    receipt["encoder_receipt"] = json.loads(json.dumps(expected))
    reseal(receipt)
    training.validate_embedding_capsule(capsule, v4_binding, expected_encoder_receipt=expected)
    receipt["encoder_receipt"]["candidate"]["tokenizer"] = "ForeignTokenizer"
    reseal(receipt)
    with pytest.raises(training.TrainingError, match="encoder receipt binding"):
        training.validate_embedding_capsule(capsule, v4_binding, expected_encoder_receipt=expected)


def _reseal_training_capsule(path: Path) -> None:
    """Rehash every mutable capsule file to model an internal tamper attempt."""

    receipt_raw = (path / "accepted-packet-receipt.json").read_bytes()
    gate = json.loads((path / "pre-holdout-gate.json").read_bytes())
    gate["accepted_packet"]["packet_receipt_sha256"] = training._sha(receipt_raw)
    gate["descriptor_sha256"] = training._sha(
        training._canonical(
            {key: value for key, value in gate.items() if key != "descriptor_sha256"}
        )
    )
    gate_raw = training._canonical(gate) + b"\n"
    (path / "pre-holdout-gate.json").write_bytes(gate_raw)
    gate_sha256 = training._sha(gate_raw)
    manifest = json.loads((path / "training-manifest.json").read_bytes())
    manifest["packet_receipt_sha256"] = training._sha(receipt_raw)
    manifest["pre_holdout_gate_descriptor_sha256"] = gate_sha256
    manifest["manifest_sha256"] = training._sha(
        training._canonical(
            {key: value for key, value in manifest.items() if key != "manifest_sha256"}
        )
    )
    (path / "training-manifest.json").write_bytes(training._canonical(manifest) + b"\n")
    selection = json.loads((path / "dev-selection.json").read_bytes())
    selection["pre_holdout_gate_descriptor_sha256"] = gate_sha256
    (path / "dev-selection.json").write_bytes(training._canonical(selection) + b"\n")
    release = json.loads((path / "holdout-release.json").read_bytes())
    release["pre_holdout_gate_descriptor_sha256"] = gate_sha256
    (path / "holdout-release.json").write_bytes(training._canonical(release) + b"\n")
    report = json.loads((path / "holdout-report.json").read_bytes())
    report["pre_holdout_gate_descriptor_sha256"] = gate_sha256
    (path / "holdout-report.json").write_bytes(training._canonical(report) + b"\n")
    files = {
        name: (path / name).read_bytes()
        for name in training._FINAL_OUTPUT_FILES
        if name != "capsule-descriptor.json"
    }
    (path / "capsule-descriptor.json").write_bytes(training._descriptor(files, kind="training"))


def test_verify_v4_training_capsule_rejects_rehashed_receipt_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capsule, packet, snapshot, _binding = _fixture(tmp_path, monkeypatch)
    _extract(capsule, packet, snapshot)
    _passing_run(monkeypatch)
    monkeypatch.setattr(
        training, "_holdout_release_registry_directory", lambda: tmp_path / "releases"
    )
    output = training.train_and_seal(capsule, packet, snapshot, "cpu", tmp_path / "out", "run")
    sealed_report = "c" * 64
    historical_receipt = {
        "registry_sha256": "a" * 64,
        "candidate": get_candidate("multilingual-minilm-l12").model_dump(mode="json"),
        "uv_lock_sha256": "b" * 64,
    }
    source_commit = "d" * 40
    grant_raw = training._GRANT_MANIFEST_PATH.read_bytes()
    monkeypatch.setattr(training, "_V4_SEALED_REPORT_SHA256", sealed_report)
    monkeypatch.setattr(training, "verify_historical_code_sources", lambda *_args: None)
    monkeypatch.setattr(
        training, "_historical_v4_encoder_receipt", lambda commit: historical_receipt
    )
    monkeypatch.setattr(
        training,
        "_historical_v4_training_grant",
        lambda _commit: (json.loads(grant_raw), grant_raw),
    )
    receipt = json.loads((output / "accepted-packet-receipt.json").read_bytes())
    receipt["sealed_report_sha256"] = sealed_report
    receipt["encoder_receipt"] = json.loads(json.dumps(historical_receipt))
    (output / "accepted-packet-receipt.json").write_bytes(training._canonical(receipt) + b"\n")
    manifest = json.loads((output / "training-manifest.json").read_bytes())
    manifest["source_commit"] = source_commit
    manifest["sealed_report_sha256"] = sealed_report
    manifest["grant_digest"] = training._sha(grant_raw)
    (output / "training-manifest.json").write_bytes(training._canonical(manifest) + b"\n")
    _reseal_training_capsule(output)
    assert training.verify_training_capsule(output)["source_commit"] == source_commit

    # An attacker can rehash the receipt, manifest, and descriptor together,
    # but cannot make the altered tokenizer agree with the recorded commit.
    receipt["encoder_receipt"]["candidate"]["tokenizer"] = "ForeignTokenizer"
    (output / "accepted-packet-receipt.json").write_bytes(training._canonical(receipt) + b"\n")
    _reseal_training_capsule(output)
    with pytest.raises(training.TrainingError, match="encoder receipt binding"):
        training.verify_training_capsule(output)


def test_verify_v4_training_capsule_requires_pinned_report_in_receipt_and_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capsule, packet, snapshot, _binding = _fixture(tmp_path, monkeypatch)
    _extract(capsule, packet, snapshot)
    _passing_run(monkeypatch)
    monkeypatch.setattr(
        training, "_holdout_release_registry_directory", lambda: tmp_path / "releases"
    )
    output = training.train_and_seal(capsule, packet, snapshot, "cpu", tmp_path / "out", "run")
    sealed_report = "c" * 64
    grant_raw = training._GRANT_MANIFEST_PATH.read_bytes()
    historical_receipt = {
        "registry_sha256": "a" * 64,
        "candidate": get_candidate("multilingual-minilm-l12").model_dump(mode="json"),
        "uv_lock_sha256": "b" * 64,
    }
    monkeypatch.setattr(training, "_V4_SEALED_REPORT_SHA256", sealed_report)
    monkeypatch.setattr(training, "verify_historical_code_sources", lambda *_args: None)
    monkeypatch.setattr(
        training, "_historical_v4_encoder_receipt", lambda _commit: historical_receipt
    )
    monkeypatch.setattr(
        training,
        "_historical_v4_training_grant",
        lambda _commit: (json.loads(grant_raw), grant_raw),
    )
    receipt = json.loads((output / "accepted-packet-receipt.json").read_bytes())
    receipt.update(sealed_report_sha256="e" * 64, encoder_receipt=historical_receipt)
    (output / "accepted-packet-receipt.json").write_bytes(training._canonical(receipt) + b"\n")
    manifest = json.loads((output / "training-manifest.json").read_bytes())
    manifest.update(
        source_commit="d" * 40,
        sealed_report_sha256=sealed_report,
        grant_digest=training._sha(grant_raw),
    )
    (output / "training-manifest.json").write_bytes(training._canonical(manifest) + b"\n")
    _reseal_training_capsule(output)
    with pytest.raises(training.TrainingError, match="accepted packet receipt"):
        training.verify_training_capsule(output)


def test_terminal_record_is_create_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    releases = tmp_path / "releases"
    monkeypatch.setattr(training, "_holdout_release_registry_directory", lambda: releases)
    binding = training.AcceptedPacketBinding(
        packet_json_sha256="a" * 64,
        accepted_train_dev_jsonl_sha256="b" * 64,
        accepted_holdout_jsonl_sha256="c" * 64,
        holdout_identities_json_sha256="d" * 64,
        accepted_rows_sha256="e" * 64,
        train_dev_identities=(),
        holdout_identities=(),
        identities=(),
    )
    capsule = cast(training.EmbeddingCapsule, SimpleNamespace(descriptor_sha256="f" * 64))
    training._seal_holdout_terminal(binding, capsule, "1" * 64, "2" * 64, "holdout_invalid")
    terminal = releases.parent / "holdout-terminals" / ("a" * 64 + ".json")
    assert set(json.loads(terminal.read_bytes())) == {
        "schema_version",
        "packet_json_sha256",
        "claim_sha256",
        "embedding_descriptor_sha256",
        "checkpoint_sha256",
        "pre_holdout_gate_descriptor_sha256",
        "sealed_report_sha256",
        "outcome",
        "terminal_sha256",
    }
    with pytest.raises(training.TrainingError, match="already sealed"):
        training._seal_holdout_terminal(binding, capsule, "1" * 64, "2" * 64, "passed")


@pytest.mark.parametrize("kind", ["symlink", "file", "mode", "owner"])
def test_canonical_registry_directories_fail_closed(
    kind: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = tmp_path / "registry"
    if kind == "symlink":
        target = tmp_path / "target"
        target.mkdir(mode=0o700)
        registry.symlink_to(target, target_is_directory=True)
    elif kind == "file":
        registry.write_bytes(b"not a directory")
    else:
        registry.mkdir(mode=0o700)
        if kind == "mode":
            os.chmod(registry, 0o755)
        else:
            uid = os.getuid()
            monkeypatch.setattr(os, "getuid", lambda: uid + 1)
    with pytest.raises(training.TrainingError, match="holdout release claim"):
        training._secure_registry_directory(registry, "holdout release claim", create=True)


def test_registry_files_fail_closed_on_symlink_type_or_mode(tmp_path: Path) -> None:
    registry = tmp_path / "registry"
    registry.mkdir(mode=0o700)
    target = registry / "target"
    target.write_bytes(b"x")
    os.chmod(target, 0o600)
    link = registry / "claim.json"
    link.symlink_to(target)
    with pytest.raises(training.TrainingError, match="holdout release claim"):
        training._read_secure_registry_file(link, "holdout release claim")
    wrong_mode = registry / "wrong-mode.json"
    wrong_mode.write_bytes(b"x")
    os.chmod(wrong_mode, 0o644)
    with pytest.raises(training.TrainingError, match="holdout release claim"):
        training._read_secure_registry_file(wrong_mode, "holdout release claim")


@pytest.mark.parametrize("lane", ["release", "terminal", "lock"])
@pytest.mark.parametrize("ancestor", ["parent", "grandparent"])
def test_registry_lanes_reject_symlinked_existing_ancestors(
    lane: str, ancestor: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "platform-state"
    phase = root / "phase4e"
    target = tmp_path / "safe-target"
    target.mkdir()
    if ancestor == "parent":
        root.mkdir()
        phase.symlink_to(target, target_is_directory=True)
    else:
        root.symlink_to(target, target_is_directory=True)
    releases = phase / "holdout-releases"
    monkeypatch.setattr(training, "_holdout_release_registry_directory", lambda: releases)
    with pytest.raises(training.TrainingError):
        if lane == "release":
            training._secure_registry_directory(releases, "holdout release claim", create=True)
        elif lane == "terminal":
            training._secure_registry_directory(
                training._holdout_terminal_registry_directory(), "holdout terminal", create=True
            )
        else:
            with training._holdout_registry_lock("a" * 64):
                pass


def _record_validate_training_grant(calls: list[str]) -> Any:
    def _validate(source_commit: str) -> tuple[dict[str, object], str]:
        calls.append(source_commit)
        return {"id": "grant"}, training._sha(b"grant-bytes")

    return _validate


def test_v4_source_ledger_is_bound_to_reviewed_commit_not_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = "a" * 40
    ledger = {"benchmarks/saracura_universal_training.py": "b" * 64}
    grant_digest = training._sha(b"grant-bytes")
    calls: list[object] = []
    source_calls: list[str] = []
    monkeypatch.setattr(training, "_clean_reviewed_source_commit", lambda: source)
    monkeypatch.setattr(training, "_v4_training_code_sources", lambda: ledger)
    monkeypatch.setattr(
        training,
        "verify_historical_code_sources",
        lambda commit, sources: calls.append((commit, sources)),
    )
    monkeypatch.setattr(training, "_validate_v4_source_receipts", lambda commit: {"commit": commit})
    monkeypatch.setattr(
        training,
        "_validate_v4_training_grant",
        _record_validate_training_grant(source_calls),
    )
    result = training._prepare_v4_source_ledger()
    assert result == (source, ledger, {"commit": source}, {"id": "grant"}, grant_digest)
    assert calls == [(source, ledger)]
    assert source_calls == [source]


def test_differing_gate_for_same_v4_revision_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    descriptor = {"code_sha256": "a" * 64}
    monkeypatch.setattr(training, "_validate_pre_holdout_gate_descriptor", lambda _raw: descriptor)
    source = "b" * 40
    training._freeze_pre_holdout_gate(tmp_path, b"one\n", source_commit=source)
    with pytest.raises(training.TrainingError, match="different frozen pre-holdout gate"):
        training._freeze_pre_holdout_gate(tmp_path, b"two\n", source_commit=source)


def _v4_binding() -> training.AcceptedPacketBinding:
    return training.AcceptedPacketBinding(
        packet_json_sha256="a" * 64,
        accepted_train_dev_jsonl_sha256="b" * 64,
        accepted_holdout_jsonl_sha256="c" * 64,
        holdout_identities_json_sha256="d" * 64,
        accepted_rows_sha256="e" * 64,
        train_dev_identities=(),
        holdout_identities=(),
        identities=(),
        sealed_report_sha256="f" * 64,
    )


def test_orphan_interrupted_terminal_preserves_validated_v4_report_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = tmp_path / "releases"
    monkeypatch.setattr(training, "_holdout_release_registry_directory", lambda: registry)
    monkeypatch.setattr(training, "_V4_SEALED_REPORT_SHA256", "f" * 64)
    binding = _v4_binding()
    capsule = cast(training.EmbeddingCapsule, SimpleNamespace(descriptor_sha256="1" * 64))
    training._claim_holdout_once(binding, capsule, "2" * 64, "3" * 64)
    claim = json.loads((registry / (binding.packet_json_sha256 + ".json")).read_bytes())
    invalid = training.AcceptedPacketBinding(
        binding.packet_json_sha256,
        binding.accepted_train_dev_jsonl_sha256,
        binding.accepted_holdout_jsonl_sha256,
        binding.holdout_identities_json_sha256,
        binding.accepted_rows_sha256,
        binding.train_dev_identities,
        binding.holdout_identities,
        binding.identities,
        sealed_report_sha256=None,
    )
    with pytest.raises(training.TrainingError, match="packet-v4 sealed report"):
        training._seal_terminal_from_claim(invalid, claim, "holdout_interrupted")
    with pytest.raises(training.TrainingError, match="terminal interrupted"):
        training._reconcile_or_refuse_prior_claim(binding, tmp_path / "out", "run")
    terminal = training._read_holdout_terminal(binding.packet_json_sha256)
    assert terminal is not None
    assert terminal["outcome"] == "holdout_interrupted"
    assert terminal["sealed_report_sha256"] == binding.sealed_report_sha256


def test_orphan_recovery_prefers_complete_capsule_and_preserves_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = tmp_path / "releases"
    monkeypatch.setattr(training, "_holdout_release_registry_directory", lambda: registry)
    monkeypatch.setattr(training, "_V4_SEALED_REPORT_SHA256", "f" * 64)
    binding = _v4_binding()
    descriptor = "1" * 64
    checkpoint = "2" * 64
    gate = "3" * 64
    capsule = cast(training.EmbeddingCapsule, SimpleNamespace(descriptor_sha256=descriptor))
    training._claim_holdout_once(binding, capsule, checkpoint, gate)
    target = tmp_path / "out" / "run"
    target.mkdir(parents=True)
    (target / "embedding-descriptor.json").write_bytes(b"descriptor")
    claim = json.loads((registry / (binding.packet_json_sha256 + ".json")).read_bytes())
    monkeypatch.setattr(
        training,
        "verify_training_capsule",
        lambda _path: {"outcome": "passed", "sealed_report_sha256": binding.sealed_report_sha256},
    )
    monkeypatch.setattr(
        training, "_json", lambda path: claim if path.name == "holdout-release.json" else {}
    )
    original_sha = training._sha
    monkeypatch.setattr(
        training, "_sha", lambda raw: descriptor if raw == b"descriptor" else original_sha(raw)
    )
    with pytest.raises(training.TrainingError, match="terminal recovered"):
        training._reconcile_or_refuse_prior_claim(binding, tmp_path / "out", "run")
    assert training._read_holdout_terminal(binding.packet_json_sha256)["outcome"] == "passed"  # type: ignore[index]


@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
def test_postpublication_failure_seals_capsule_outcome_not_invalid(
    failure: type[BaseException], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capsule, packet, snapshot, binding = _fixture(tmp_path, monkeypatch)
    _extract(capsule, packet, snapshot)
    _passing_run(monkeypatch)
    monkeypatch.setattr(
        training, "_holdout_release_registry_directory", lambda: tmp_path / "releases"
    )
    original_write = training._write_capsule

    def publish_then_fail(*args: object, **kwargs: object) -> Path:
        original_write(*args, **kwargs)  # type: ignore[arg-type]
        raise failure("after publication")

    monkeypatch.setattr(training, "_write_capsule", publish_then_fail)
    with pytest.raises(failure, match="after publication"):
        training.train_and_seal(capsule, packet, snapshot, "cpu", tmp_path / "out", "run")
    terminal = training._read_holdout_terminal(binding.packet_json_sha256)
    assert terminal is not None
    assert terminal["outcome"] in {"passed", "holdout_failed"}
    assert (tmp_path / "out" / "run").is_dir()


def test_reentry_recovers_complete_capsule_before_interrupted_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capsule, packet, snapshot, binding = _fixture(tmp_path, monkeypatch)
    _extract(capsule, packet, snapshot)
    _passing_run(monkeypatch)
    monkeypatch.setattr(
        training, "_holdout_release_registry_directory", lambda: tmp_path / "releases"
    )
    original_seal = training._seal_holdout_terminal
    monkeypatch.setattr(
        training,
        "_seal_holdout_terminal",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(KeyboardInterrupt("before terminal")),
    )
    with pytest.raises(KeyboardInterrupt, match="before terminal"):
        training.train_and_seal(capsule, packet, snapshot, "cpu", tmp_path / "out", "run")
    assert (tmp_path / "out" / "run").is_dir()
    assert training._read_holdout_terminal(binding.packet_json_sha256) is None

    monkeypatch.setattr(training, "_seal_holdout_terminal", original_seal)
    with (
        training._holdout_registry_lock(binding.packet_json_sha256),
        pytest.raises(training.TrainingError, match="terminal recovered"),
    ):
        training._reconcile_or_refuse_prior_claim(binding, tmp_path / "out", "run")
    terminal = training._read_holdout_terminal(binding.packet_json_sha256)
    assert terminal is not None
    assert terminal["outcome"] in {"passed", "holdout_failed"}


def test_lock_rejects_concurrent_owner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        training, "_holdout_release_registry_directory", lambda: tmp_path / "releases"
    )
    digest = "a" * 64
    with (
        training._holdout_registry_lock(digest),
        pytest.raises(training.TrainingError, match="already active"),
        training._holdout_registry_lock(digest),
    ):
        pass


def test_complete_training_opens_holdout_payload_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capsule, packet, snapshot, _ = _fixture(tmp_path, monkeypatch)
    _extract(capsule, packet, snapshot)
    _passing_run(monkeypatch)
    monkeypatch.setattr(
        training, "_holdout_release_registry_directory", lambda: tmp_path / "releases"
    )
    holdout = (packet / "accepted-holdout.jsonl").resolve()
    original = Path.read_bytes
    opens = 0

    def count(path: Path) -> bytes:
        nonlocal opens
        if path.resolve() == holdout:
            opens += 1
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", count)
    training.train_and_seal(capsule, packet, snapshot, "cpu", tmp_path / "out", "run")
    assert opens == 1


def test_postclaim_failure_seals_invalid_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capsule, packet, snapshot, binding = _fixture(tmp_path, monkeypatch)
    _extract(capsule, packet, snapshot)
    _passing_run(monkeypatch)
    monkeypatch.setattr(
        training, "_holdout_release_registry_directory", lambda: tmp_path / "releases"
    )
    monkeypatch.setattr(
        training,
        "derive_holdout_embeddings_in_memory",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("injected")),
    )
    with pytest.raises(RuntimeError, match="injected"):
        training.train_and_seal(capsule, packet, snapshot, "cpu", tmp_path / "out", "run")
    terminal = training._read_holdout_terminal(binding.packet_json_sha256)
    assert terminal is not None and terminal["outcome"] == "holdout_invalid"


def test_grant_manifest_rejects_missing_or_tampered_auth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grant = json.loads(training._GRANT_MANIFEST_PATH.read_bytes())
    training.validate_v4_training_grant_bytes(training._GRANT_MANIFEST_PATH.read_bytes())
    tampered = json.loads(json.dumps(grant))
    tampered["authorizations"]["synthetic_research_training_authorized"] = False
    with pytest.raises(training.TrainingError, match="v4 training grant authorizations"):
        training.validate_v4_training_grant_bytes(json.dumps(tampered).encode())
    tampered = json.loads(json.dumps(grant))
    tampered["authorizations"]["runtime_registration_authorized"] = True
    with pytest.raises(training.TrainingError, match="v4 training grant authorizations"):
        training.validate_v4_training_grant_bytes(json.dumps(tampered).encode())
    tampered = json.loads(json.dumps(grant))
    del tampered["authorizations"]
    with pytest.raises(training.TrainingError, match="v4 training grant shape"):
        training.validate_v4_training_grant_bytes(json.dumps(tampered).encode())
    tampered = json.loads(json.dumps(grant))
    tampered["packet_manifest_sha256"] = None
    with pytest.raises(training.TrainingError, match="v4 training grant digest"):
        training.validate_v4_training_grant_bytes(json.dumps(tampered).encode())


def test_legacy_training_dispatch_still_reaches_real_historical_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packet = tmp_path / "packet"
    packet.mkdir()
    (packet / "packet.json").write_bytes(
        training._canonical({"schema_version": "phase4e-accepted-packet.v3"})
    )
    monkeypatch.setattr(
        training, "require_phase4e_authorization", real_require_phase4e_authorization
    )
    with pytest.raises(Phase4EPolicyError, match="not authorized before a reviewed pilot PASS"):
        training.train_and_seal(
            tmp_path / "capsule",
            packet,
            tmp_path / "snapshot",
            "cpu",
            tmp_path / "out",
            "run",
        )


def test_v4_consumes_grant_under_one_lock_before_training_impl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packet = tmp_path / "packet"
    packet.mkdir()
    (packet / "packet.json").write_bytes(
        training._canonical({"schema_version": "phase4e-accepted-packet.v4"})
    )
    paths = training.V4ArtifactPaths(
        packet,
        tmp_path / "embedding-capsule-v1",
        tmp_path / "training-capsule-v1",
        tmp_path / "reports" / "ptbr-recovery-v1",
    )
    binding = training.AcceptedPacketBinding(
        packet_json_sha256="a" * 64,
        accepted_train_dev_jsonl_sha256="b" * 64,
        accepted_holdout_jsonl_sha256="c" * 64,
        holdout_identities_json_sha256="d" * 64,
        accepted_rows_sha256="e" * 64,
        train_dev_identities=(),
        holdout_identities=(),
        identities=(),
        sealed_report_sha256="f" * 64,
    )
    source_commit = "1" * 40
    source_ledger = {"source": "2" * 64}
    grant = {
        "packet_manifest_sha256": binding.packet_json_sha256,
        "sealed_report_sha256": binding.sealed_report_sha256,
    }
    grant_digest = "3" * 64
    calls: list[str] = []

    monkeypatch.setattr(training, "_require_v4_layout", lambda *_args, **_kwargs: paths)
    monkeypatch.setattr(
        training,
        "_prepare_v4_source_ledger",
        lambda: (source_commit, source_ledger, {"encoder": "receipt"}, grant, grant_digest),
    )
    monkeypatch.setattr(
        training, "validate_accepted_packet_binding", lambda *_args, **_kwargs: binding
    )
    monkeypatch.setattr(
        training,
        "_validate_v4_grant_packet_binding",
        lambda *_args: calls.append("grant-binding"),
    )

    @contextmanager
    def one_lock(_digest: str) -> Any:
        calls.append("lock-enter")
        try:
            yield
        finally:
            calls.append("lock-exit")

    monkeypatch.setattr(training, "_holdout_registry_lock", one_lock)
    monkeypatch.setattr(
        training,
        "_require_grant_consumption",
        lambda *_args: calls.append("grant-consumption"),
    )

    def run_impl(*_args: object, **kwargs: object) -> Path:
        assert kwargs["registry_lock_held"] is True
        calls.append("training-impl")
        return paths.training

    monkeypatch.setattr(training, "_train_and_seal_impl", run_impl)
    result = training.train_and_seal(
        paths.embeddings,
        packet,
        tmp_path / "snapshot",
        "cpu",
        paths.training.parent,
        paths.training.name,
    )
    assert result == paths.training
    assert calls == [
        "grant-binding",
        "lock-enter",
        "grant-consumption",
        "training-impl",
        "lock-exit",
    ]


def test_grant_consumption_record_is_create_only_and_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    releases = tmp_path / "releases"
    monkeypatch.setattr(training, "_holdout_release_registry_directory", lambda: releases)
    grant_dir = releases.parent / "training-grant-consumptions"
    packet_digest = "a" * 64
    grant_digest = "b" * 64
    source_commit = "c" * 40
    code_ledger = "d" * 64
    training._require_grant_consumption(
        packet_digest, grant_digest, source_commit, code_ledger, "saracura-universal-ranker.v0"
    )
    target = grant_dir / (packet_digest + ".json")
    assert training._read_secure_registry_file(target, "training grant consumption")
    payload = json.loads(target.read_bytes())
    assert payload["grant_digest"] == grant_digest
    # Byte-identical re-entry is allowed
    training._require_grant_consumption(
        packet_digest, grant_digest, source_commit, code_ledger, "saracura-universal-ranker.v0"
    )
    # Different grant digest fails
    with pytest.raises(training.TrainingError, match="different grant consumption record exists"):
        training._require_grant_consumption(
            packet_digest, "e" * 64, source_commit, code_ledger, "saracura-universal-ranker.v0"
        )
    # Different source commit fails
    with pytest.raises(training.TrainingError, match="different grant consumption record exists"):
        training._require_grant_consumption(
            packet_digest, grant_digest, "f" * 40, code_ledger, "saracura-universal-ranker.v0"
        )
    # Different code ledger fails
    with pytest.raises(training.TrainingError, match="different grant consumption record exists"):
        training._require_grant_consumption(
            packet_digest, grant_digest, source_commit, "1" * 64, "saracura-universal-ranker.v0"
        )
    # Different architecture revision fails
    with pytest.raises(training.TrainingError, match="grant consumption record binding"):
        training._require_grant_consumption(
            packet_digest, grant_digest, source_commit, code_ledger, "different-revision.v1"
        )
    # Symlinked directory fails
    other_releases = tmp_path / "other" / "releases"
    monkeypatch.setattr(training, "_holdout_release_registry_directory", lambda: other_releases)
    registry_dir = other_releases.parent
    registry_dir.mkdir(parents=True)
    symlink_target = tmp_path / "safe-target"
    symlink_target.mkdir()
    target_symlink = registry_dir / "training-grant-consumptions"
    target_symlink.symlink_to(symlink_target, target_is_directory=True)
    with pytest.raises(training.TrainingError, match="training grant consumption"):
        training._require_grant_consumption(
            "2" * 64, grant_digest, source_commit, code_ledger, "saracura-universal-ranker.v0"
        )


def test_v4_training_manifest_rejects_rehashed_grant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capsule, packet, snapshot, _binding = _fixture(tmp_path, monkeypatch)
    _extract(capsule, packet, snapshot)
    _passing_run(monkeypatch)
    monkeypatch.setattr(
        training, "_holdout_release_registry_directory", lambda: tmp_path / "releases"
    )
    output = training.train_and_seal(capsule, packet, snapshot, "cpu", tmp_path / "out", "run")
    sealed_report = "c" * 64
    monkeypatch.setattr(training, "_V4_SEALED_REPORT_SHA256", sealed_report)
    monkeypatch.setattr(training, "verify_historical_code_sources", lambda *_args: None)
    historical_receipt = {
        "registry_sha256": "a" * 64,
        "candidate": get_candidate("multilingual-minilm-l12").model_dump(mode="json"),
        "uv_lock_sha256": "b" * 64,
    }
    monkeypatch.setattr(
        training, "_historical_v4_encoder_receipt", lambda _commit: historical_receipt
    )
    grant_path = (
        training._GRANT_MANIFEST_PATH.parents[0]
        / "phase4e-saracura-universal-training-authorization.v1.json"
    )
    grant_digest = training._sha(grant_path.read_bytes())

    receipt = json.loads((output / "accepted-packet-receipt.json").read_bytes())
    receipt["sealed_report_sha256"] = sealed_report
    receipt["encoder_receipt"] = json.loads(json.dumps(historical_receipt))
    (output / "accepted-packet-receipt.json").write_bytes(training._canonical(receipt) + b"\n")
    manifest = json.loads((output / "training-manifest.json").read_bytes())
    manifest["source_commit"] = "d" * 40
    manifest["sealed_report_sha256"] = sealed_report
    manifest["grant_digest"] = grant_digest
    (output / "training-manifest.json").write_bytes(training._canonical(manifest) + b"\n")
    _reseal_training_capsule(output)
    # Correct grant digest passes
    monkeypatch.setattr(
        training,
        "_historical_v4_training_grant",
        lambda _commit: (json.loads(grant_path.read_bytes()), grant_path.read_bytes()),
    )
    assert training.verify_training_capsule(output)["source_commit"] == "d" * 40

    # A wrong grant digest fails historical verification
    manifest["grant_digest"] = "e" * 64
    (output / "training-manifest.json").write_bytes(training._canonical(manifest) + b"\n")
    _reseal_training_capsule(output)
    with pytest.raises(training.TrainingError, match="v4 training grant digest binding"):
        training.verify_training_capsule(output)
