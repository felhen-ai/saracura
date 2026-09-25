"""Adversarial offline checks for the Phase 4E.2c training capsule."""

from __future__ import annotations

import importlib.util
import json
import socket
from pathlib import Path
from typing import Any

import pytest

from benchmarks import saracura_universal_training as training
from benchmarks.encoder_loader import LoadedEncoder
from benchmarks.encoder_registry import get_candidate

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
    monkeypatch.setattr(training, "validate_accepted_packet_binding", lambda _: binding)
    monkeypatch.setattr(training, "validate_accepted_packet", lambda _: None)
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
