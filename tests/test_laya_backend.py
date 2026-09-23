"""Focused no-download contracts for the installed Laya backend boundary."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from pydantic import JsonValue

from benchmarks import inspect_wheel
from saracura.backends.laya import render_laya_choice
from saracura.contracts import (
    ChoiceCriterion,
    ChoiceQuestion,
    DecisionRequest,
    ErrorCode,
    SaracuraError,
    WorkflowReference,
)
from saracura.laya_snapshot import (
    LayaCandidate,
    LayaLedgerFile,
    LayaSnapshotError,
    LayaSnapshotReceipt,
    load_laya_candidate,
    verify_laya_snapshot,
)
from saracura.runtime.workflows import (
    UNIVERSAL_CHOICE_WORKFLOW_ID,
    UNIVERSAL_CHOICE_WORKFLOW_REVISION,
)


class ByteTokenizer:
    cls_token_id = 101
    sep_token_id = 102
    mask_token_id = 103
    all_special_tokens = ("<bos>", "<eos>", "<mask>", "<pad>", "<unk>")

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return list(range(10, 10 + len(text.encode("utf-8"))))


def _request(
    *, state: dict[str, JsonValue] | None = None, instruction: str = "Escolha uma opção."
) -> DecisionRequest:
    return DecisionRequest(
        api_version="v1alpha1",
        model="052592a15d198d9ad47da779604259b10b47b7aa",
        mode="research",
        locale="pt-BR",
        domain="email-triage",
        workflow=WorkflowReference(
            id=UNIVERSAL_CHOICE_WORKFLOW_ID, revision=UNIVERSAL_CHOICE_WORKFLOW_REVISION
        ),
        state=state or {"subject": "Aviso sintético", "body": "Texto apenas de teste."},
        questions=(
            ChoiceQuestion(
                id="triage",
                type="choice",
                instruction=instruction,
                criteria=(
                    ChoiceCriterion(id="first", description="Primeira categoria."),
                    ChoiceCriterion(id="second", description="Segunda categoria."),
                ),
            ),
        ),
    )


def test_package_ledger_is_exact_and_independent_of_benchmarks() -> None:
    candidate = load_laya_candidate()
    assert candidate.model_id == "convaiinnovations/laya-multilingual"
    assert candidate.model_revision == "052592a15d198d9ad47da779604259b10b47b7aa"
    assert (
        candidate.checkpoint_sha256
        == "9d628fd971b700382ac6f65920a86f149777b2e748e0c955fb3b19695aa8f204"
    )
    assert tuple(item.path for item in candidate.files) == (
        "encoder/config.json",
        "rl_agent_config.json",
        "tokenizer/tokenizer_config.json",
        "tokenizer/tokenizer.json",
        "model.safetensors",
    )
    with pytest.raises(LayaSnapshotError):
        load_laya_candidate(b'{"schema_version":"laya-candidate.v1"}')
    assert inspect_wheel._allowed_entry("saracura/laya-candidate.v1.json")


def test_renderer_preserves_order_and_never_truncates() -> None:
    request = _request()
    rendered = render_laya_choice(ByteTokenizer(), request, request.questions[0])

    assert rendered.input_ids[rendered.marker_positions[0]] == 103
    assert rendered.input_ids[rendered.marker_positions[1]] == 103
    assert rendered.marker_positions == tuple(sorted(rendered.marker_positions))
    assert rendered.input_tokens <= 1024


@pytest.mark.parametrize(
    ("decision_request", "code"),
    [
        (_request(state={"na\u0303o": "valor"}), ErrorCode.REQUEST_INVALID),
        (_request(instruction="Use <mask> agora."), ErrorCode.REQUEST_INVALID),
        (_request(state={"body": "x" * 1200}), ErrorCode.CAPACITY_EXCEEDED),
        (_request(instruction="x" * 300), ErrorCode.CAPACITY_EXCEEDED),
    ],
)
def test_renderer_rejects_malicious_or_over_capacity_complete_inputs(
    decision_request: DecisionRequest, code: ErrorCode
) -> None:
    with pytest.raises(SaracuraError) as captured:
        render_laya_choice(ByteTokenizer(), decision_request, decision_request.questions[0])
    assert captured.value.payload.code == code


def _tiny_candidate() -> LayaCandidate:
    values = {
        "encoder/config.json": b"config",
        "rl_agent_config.json": b"head",
        "tokenizer/tokenizer_config.json": b"tokenizer-config",
        "tokenizer/tokenizer.json": b"tokenizer-graph",
        "model.safetensors": b"safe-weights",
    }
    return LayaCandidate(
        candidate="laya-multilingual",
        model_id="convaiinnovations/laya-multilingual",
        model_revision="r" * 40,
        upstream_source_revision="s" * 40,
        acquisition_contract_digest="a" * 64,
        conformance_vector_digest="b" * 64,
        runtime_disposition="research_only_unresolved_provenance",
        files=tuple(
            LayaLedgerFile(path, len(value), hashlib.sha256(value).hexdigest())
            for path, value in values.items()
        ),
    )


def _write_tiny_snapshot(root: Path, candidate: LayaCandidate) -> None:
    root.mkdir(mode=0o700)
    (root / "encoder").mkdir(mode=0o700)
    (root / "tokenizer").mkdir(mode=0o700)
    for item in candidate.files:
        target = root / item.path
        target.write_bytes(
            {
                "encoder/config.json": b"config",
                "rl_agent_config.json": b"head",
                "tokenizer/tokenizer_config.json": b"tokenizer-config",
                "tokenizer/tokenizer.json": b"tokenizer-graph",
                "model.safetensors": b"safe-weights",
            }[item.path]
        )
        target.chmod(0o600)


def test_snapshot_receipt_rejects_extra_path_and_live_name_replacement(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    candidate = _tiny_candidate()
    monkeypatch.setattr("saracura.laya_snapshot.load_laya_candidate", lambda: candidate)
    root = tmp_path / "snapshot"
    _write_tiny_snapshot(root, candidate)
    (root / "unexpected").write_text("x")
    (root / "unexpected").chmod(0o600)
    with pytest.raises(LayaSnapshotError):
        verify_laya_snapshot(root)
    (root / "unexpected").unlink()

    receipt = verify_laya_snapshot(root)
    try:
        replacement = root / "replacement"
        replacement.write_bytes(b"config")
        replacement.chmod(0o600)
        os.replace(replacement, root / "encoder" / "config.json")
        with pytest.raises(LayaSnapshotError):
            receipt.read_verified_bytes("encoder/config.json")
    finally:
        receipt.close()


def test_snapshot_rejects_symlink_hardlink_and_mode_drift(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    candidate = _tiny_candidate()
    monkeypatch.setattr("saracura.laya_snapshot.load_laya_candidate", lambda: candidate)
    root = tmp_path / "snapshot"
    _write_tiny_snapshot(root, candidate)
    model = root / "model.safetensors"

    model.chmod(0o644)
    with pytest.raises(LayaSnapshotError):
        verify_laya_snapshot(root)
    model.chmod(0o600)

    model.chmod(0o4600)
    with pytest.raises(LayaSnapshotError):
        verify_laya_snapshot(root)
    model.chmod(0o600)

    external_link = tmp_path / "same-inode"
    os.link(model, external_link)
    with pytest.raises(LayaSnapshotError):
        verify_laya_snapshot(root)
    external_link.unlink()

    config = root / "encoder" / "config.json"
    config.unlink()
    os.symlink(tmp_path / "missing", config)
    with pytest.raises(LayaSnapshotError):
        verify_laya_snapshot(root)


def test_snapshot_validation_failure_closes_every_opened_descriptor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    candidate = _tiny_candidate()
    monkeypatch.setattr("saracura.laya_snapshot.load_laya_candidate", lambda: candidate)
    root = tmp_path / "snapshot"
    _write_tiny_snapshot(root, candidate)
    (root / "model.safetensors").chmod(0o644)
    opened: set[int] = set()
    closed: set[int] = set()
    original_open = os.open
    original_close = os.close

    def observe_open(
        path: Any,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        fd = original_open(path, flags, mode, dir_fd=dir_fd)
        opened.add(fd)
        return fd

    def observe_close(fd: int) -> None:
        if fd in opened:
            closed.add(fd)
        original_close(fd)

    monkeypatch.setattr(os, "open", observe_open)
    monkeypatch.setattr(os, "close", observe_close)

    with pytest.raises(LayaSnapshotError):
        verify_laya_snapshot(root)

    assert opened == closed


def test_snapshot_hash_failure_closes_transferred_receipt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    candidate = _tiny_candidate()
    first = candidate.files[0]
    corrupted = replace(
        candidate,
        files=(replace(first, sha256="0" * 64), *candidate.files[1:]),
    )
    monkeypatch.setattr("saracura.laya_snapshot.load_laya_candidate", lambda: corrupted)
    closed = False
    original_close = LayaSnapshotReceipt.close

    def observe_close(receipt: LayaSnapshotReceipt) -> None:
        nonlocal closed
        closed = True
        original_close(receipt)

    monkeypatch.setattr("saracura.laya_snapshot.LayaSnapshotReceipt.close", observe_close)
    root = tmp_path / "snapshot"
    _write_tiny_snapshot(root, candidate)

    with pytest.raises(LayaSnapshotError, match="ledger mismatch"):
        verify_laya_snapshot(root)

    assert closed


def test_default_import_does_not_load_optional_ml_modules() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import saracura.backends,sys; "
            "forbidden={'torch','transformers','tokenizers',"
            "'safetensors','psutil','platformdirs'}; "
            "assert not forbidden & set(sys.modules)",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
