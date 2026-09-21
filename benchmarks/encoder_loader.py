"""Safetensors-only, local-only loader for reviewed encoder snapshots."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from benchmarks.encoder_acquisition import verify_snapshot
from benchmarks.encoder_registry import Candidate


@dataclass(frozen=True)
class LoadedEncoder:
    model: Any
    tokenizer: Any
    candidate: Candidate
    snapshot: Path


_VERIFICATION_SEAL = object()


@dataclass(frozen=True, init=False)
class VerifiedSnapshot:
    """Capability proving that this exact candidate snapshot was verified."""

    candidate_id: str
    revision: str
    path: Path
    _seal: object = field(repr=False, compare=False)

    @classmethod
    def create(cls, candidate: Candidate, snapshot: Path) -> VerifiedSnapshot:
        if not verify_snapshot(candidate, snapshot):
            raise ValueError("encoder snapshot failed verification")
        verified = object.__new__(cls)
        object.__setattr__(verified, "candidate_id", candidate.id)
        object.__setattr__(verified, "revision", candidate.revision)
        object.__setattr__(verified, "path", snapshot.resolve())
        object.__setattr__(verified, "_seal", _VERIFICATION_SEAL)
        return verified


def load_encoder(
    candidate: Candidate,
    snapshot: Path,
    *,
    verified: VerifiedSnapshot | None = None,
) -> LoadedEncoder:
    if verified is None:
        verified = VerifiedSnapshot.create(candidate, snapshot)
    if (
        getattr(verified, "_seal", None) is not _VERIFICATION_SEAL
        or verified.candidate_id != candidate.id
        or verified.revision != candidate.revision
        or verified.path != snapshot.resolve()
    ):
        raise ValueError("encoder snapshot verification token does not match load")
    try:
        import torch  # type: ignore[import-not-found]
        from transformers import (  # type: ignore[import-not-found]
            BertConfig,
            BertModel,
            PreTrainedTokenizerFast,
            XLMRobertaConfig,
            XLMRobertaModel,
            XLMRobertaTokenizerFast,
        )
    except ImportError as error:
        raise RuntimeError("encoder-eval extra is required for local probing") from error

    config_data = json.loads((snapshot / "config.json").read_bytes())
    tokenizer_data = json.loads((snapshot / "tokenizer_config.json").read_bytes())
    if config_data.get("auto_map") is not None or tokenizer_data.get("auto_map") is not None:
        raise ValueError("dynamic remote code is not allowed")
    expected = {
        "xlm-roberta": (XLMRobertaConfig, XLMRobertaModel, XLMRobertaTokenizerFast),
        "bert": (BertConfig, BertModel, PreTrainedTokenizerFast),
    }
    selected = expected.get(candidate.model_type or "")
    if selected is None:
        raise ValueError("candidate loader is not allowlisted")
    config_class, model_class, tokenizer_class = selected
    config = config_class.from_pretrained(snapshot, local_files_only=True)
    if config.model_type != candidate.model_type:
        raise ValueError("runtime model type does not match registry")
    if model_class.__name__ != candidate.architecture:
        raise ValueError("runtime architecture does not match registry")
    if int(getattr(config, "hidden_size", -1)) != candidate.hidden_width:
        raise ValueError("runtime hidden width does not match registry")
    tokenizer = tokenizer_class.from_pretrained(
        snapshot,
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("fast tokenizer is required")
    if tokenizer.__class__.__name__ != candidate.tokenizer:
        raise ValueError("runtime tokenizer class does not match registry")
    model = model_class.from_pretrained(
        snapshot,
        config=config,
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
    )
    model.eval()
    if not isinstance(model, torch.nn.Module):
        raise ValueError("unexpected model class")
    return LoadedEncoder(model=model, tokenizer=tokenizer, candidate=candidate, snapshot=snapshot)
