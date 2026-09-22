"""Descriptor-bound local model construction without hub or cache discovery."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from benchmarks.universal_local.acquisition import VerificationReceipt
from benchmarks.universal_local.registry import Candidate

_MODEL_RESIDENT = False
_MIN_FREE_DISK = 128 * 1024 * 1024
_MIN_FREE_MEMORY = 512 * 1024 * 1024


class LoadError(ValueError):
    """A bounded loader failure. Its cause is never a public CLI error."""


@dataclass(frozen=True)
class DirectTokenizer:
    """Minimal direct fast-tokenizer surface used by the closed renderers."""

    backend: Any
    cls_token_id: int
    sep_token_id: int
    mask_token_id: int
    pad_token_id: int
    all_special_tokens: tuple[str, ...]
    nli_pair: bool

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        return list(self.backend.encode(text, add_special_tokens=add_special_tokens).ids)

    def num_special_tokens_to_add(self, *, pair: bool = False) -> int:
        if pair and not self.nli_pair:
            raise ValueError("pair encoding is unavailable for this tokenizer")
        return 3 if pair else 2

    def build_inputs_with_special_tokens(
        self, token_ids_0: list[int], token_ids_1: list[int] | None = None
    ) -> list[int]:
        if token_ids_1 is None:
            return [self.cls_token_id, *token_ids_0, self.sep_token_id]
        if not self.nli_pair:
            raise ValueError("pair encoding is unavailable for this tokenizer")
        return [self.cls_token_id, *token_ids_0, self.sep_token_id, *token_ids_1, self.sep_token_id]


@dataclass
class LoadedCandidate:
    model: Any
    tokenizer: DirectTokenizer
    candidate: Candidate
    process_threads: int

    def close(self) -> None:
        global _MODEL_RESIDENT
        self.model = None
        _MODEL_RESIDENT = False


def _closed_json(receipt: VerificationReceipt, name: str) -> dict[str, Any]:
    try:
        value = json.loads(receipt.read_verified_bytes(name).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise LoadError("invalid verified metadata") from error
    if not isinstance(value, dict) or "auto_map" in value or value.get("trust_remote_code"):
        raise LoadError("unapproved executable metadata")
    return value


def _require(value: dict[str, Any], expected: dict[str, Any], message: str) -> None:
    if any(value.get(key) != wanted for key, wanted in expected.items()):
        raise LoadError(message)


def _specials(config: dict[str, Any], required: dict[str, str]) -> tuple[str, ...]:
    _require(config, required, "tokenizer special-token contract mismatch")
    values = tuple(required.values())
    if len(set(values)) < 4 or any(not value for value in values):
        raise LoadError("tokenizer special-token contract mismatch")
    return values


def _tokenizer(receipt: VerificationReceipt, candidate: Candidate) -> DirectTokenizer:
    try:
        from tokenizers import Tokenizer  # type: ignore[import-not-found]
    except ImportError as error:
        raise LoadError("universal-local extra is required") from error
    if candidate.id == "laya-multilingual":
        config = _closed_json(receipt, "tokenizer/tokenizer_config.json")
        specials = _specials(
            config,
            {
                "cls_token": "<bos>",
                "sep_token": "<eos>",
                "mask_token": "<mask>",
                "pad_token": "<pad>",
                "unk_token": "<unk>",
            },
        )
        raw_name = "tokenizer/tokenizer.json"
        pair = False
    else:
        config = _closed_json(receipt, "tokenizer_config.json")
        special_map = _closed_json(receipt, "special_tokens_map.json")
        required = {
            "cls_token": "[CLS]",
            "sep_token": "[SEP]",
            "mask_token": "[MASK]",
            "pad_token": "[PAD]",
            "unk_token": "[UNK]",
        }
        specials = _specials(config, required)
        for key, expected in required.items():
            actual = special_map.get(key)
            if isinstance(actual, dict):
                actual = actual.get("content")
            if actual != expected:
                raise LoadError("tokenizer special-token map mismatch")
        raw_name = "tokenizer.json"
        pair = True
    try:
        backend = Tokenizer.from_str(receipt.read_verified_bytes(raw_name).decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as error:
        raise LoadError("invalid verified tokenizer graph") from error
    ids = {value: backend.token_to_id(value) for value in specials}
    if any(value is None or not isinstance(value, int) for value in ids.values()):
        raise LoadError("tokenizer structural IDs missing")
    cls, sep, mask, pad, _unk = specials
    return DirectTokenizer(
        backend=backend,
        cls_token_id=ids[cls],
        sep_token_id=ids[sep],
        mask_token_id=ids[mask],
        pad_token_id=ids[pad],
        all_special_tokens=specials,
        nli_pair=pair,
    )


def _preflight(receipt: VerificationReceipt, candidate: Candidate) -> None:
    """Refuse allocation when the already-verified local lane lacks headroom."""
    try:
        import psutil  # type: ignore[import-untyped]
    except ImportError as error:
        raise LoadError("universal-local extra is required") from error
    try:
        available_memory = int(psutil.virtual_memory().available)
        free_disk = receipt.descriptor_free_bytes("model.safetensors")
    except OSError as error:
        raise LoadError("local preflight unavailable") from error
    # F16 source bytes become FP32 parameters. The reader maps each tensor, so
    # this deliberately does not reserve a second full source-state copy.
    required_memory = candidate.total_bytes * 2 + _MIN_FREE_MEMORY
    if available_memory < required_memory or free_disk < _MIN_FREE_DISK:
        raise LoadError("insufficient local resources for reviewed FP32 construction")


def _laya_model(torch: Any, config_data: dict[str, Any], head: dict[str, Any]) -> Any:
    from transformers import (  # type: ignore[import-not-found]
        ModernBertConfig,
        ModernBertModel,
    )

    _require(
        config_data,
        {
            "model_type": "modernbert",
            "hidden_size": 768,
            "num_hidden_layers": 22,
            "num_attention_heads": 12,
            "vocab_size": 256000,
        },
        "ModernBERT config does not match reviewed contract",
    )
    _require(
        head,
        {"head_layers": 2, "max_len": 1024, "head_max_len": 256, "temperature": [1.0, 1.0, 1.0]},
        "Laya decision-head config does not match reviewed contract",
    )
    if head.get("temperature_by_options") != {} or "auto_map" in config_data:
        raise LoadError("Laya metadata is not closed")
    config = ModernBertConfig(**config_data)
    config._attn_implementation = "sdpa"

    class LayaDecisionModel(torch.nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.encoder = ModernBertModel(config)
            layer = torch.nn.TransformerEncoderLayer(
                d_model=768,
                nhead=12,
                dim_feedforward=3072,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.head = torch.nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
            self.type_emb = torch.nn.Embedding(3, 768)
            self.scorer = torch.nn.Sequential(
                torch.nn.LayerNorm(768),
                torch.nn.Linear(768, 768),
                torch.nn.GELU(),
                torch.nn.Linear(768, 1),
            )
            self.act_head = torch.nn.Sequential(
                torch.nn.Linear(772, 256), torch.nn.GELU(), torch.nn.Linear(256, 2)
            )
            self.register_buffer("temperature", torch.ones(3, dtype=torch.float32))

        def forward(
            self,
            input_ids: Any,
            attention_mask: Any,
            marker_pos: Any,
            marker_mask: Any,
            qtype: Any,
        ) -> Any:
            encoded = self.encoder(
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state
            refined = encoded + self.type_emb(qtype)[:, None, :]
            padding = ~attention_mask.bool()
            for head_layer in self.head.layers:
                refined = head_layer(refined, src_key_padding_mask=padding)
            gather_index = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, refined.size(-1))
            marker_states = torch.gather(refined, 1, gather_index)
            logits = self.scorer(marker_states).squeeze(-1).float()
            return logits.masked_fill(~marker_mask, -1e4)

    return LayaDecisionModel().float()


def _mdeberta_model(receipt: VerificationReceipt) -> Any:
    from transformers import (
        DebertaV2Config,
        DebertaV2ForSequenceClassification,
    )

    config_data = _closed_json(receipt, "config.json")
    _require(
        config_data,
        {
            "model_type": "deberta-v2",
            "architectures": ["DebertaV2ForSequenceClassification"],
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "vocab_size": 251000,
            "label2id": {"entailment": 0, "neutral": 1, "contradiction": 2},
        },
        "mDeBERTa config does not match reviewed contract",
    )
    if "auto_map" in config_data:
        raise LoadError("mDeBERTa metadata is not closed")
    config = DebertaV2Config(**config_data)
    config._attn_implementation = "eager"
    return DebertaV2ForSequenceClassification(config).float()


def _load_strict(
    receipt: VerificationReceipt, candidate: Candidate, model: Any, torch: Any
) -> None:
    """Map the safetensors file by retained descriptor and copy one tensor at a time."""
    try:
        from safetensors import safe_open  # type: ignore[import-not-found]
    except ImportError as error:
        raise LoadError("universal-local extra is required") from error
    receipt.assert_live()
    expected = model.state_dict()
    try:
        with safe_open(
            receipt.descriptor_path("model.safetensors"), framework="pt", device="cpu"
        ) as source:
            keys = set(source.keys())
            dtypes: dict[str, int] = {}
            for key in keys:
                dtype = str(source.get_slice(key).get_dtype())
                dtypes[dtype] = dtypes.get(dtype, 0) + 1
            if dtypes != candidate.source_weight_dtypes:
                raise LoadError("source Safetensors dtype contract mismatch")
            if candidate.id == "mdeberta-nli":
                auxiliary = "deberta.embeddings.position_ids"
                if auxiliary not in keys:
                    raise LoadError("mDeBERTa position ID tensor missing")
                position = source.get_tensor(auxiliary)
                if (
                    list(position.shape) != [1, 512]
                    or position.dtype != torch.int64
                    or position.tolist() != [list(range(512))]
                ):
                    raise LoadError("mDeBERTa position ID tensor mismatch")
                keys.remove(auxiliary)
                if auxiliary in expected:
                    raise LoadError("Transformers position ID ownership changed")
            elif candidate.id == "laya-multilingual":
                if (
                    "temperature" not in keys
                    or list(source.get_slice("temperature").get_shape()) != [3]
                    or str(source.get_slice("temperature").get_dtype()) != "F32"
                ):
                    raise LoadError("Laya temperature tensor mismatch")
            if keys != set(expected):
                raise LoadError("strict state dictionary key mismatch")
            with torch.no_grad():
                for key in sorted(keys):
                    descriptor = source.get_slice(key)
                    if list(descriptor.get_shape()) != list(expected[key].shape):
                        raise LoadError("strict state dictionary shape mismatch")
                    dtype = str(descriptor.get_dtype())
                    if key == "temperature":
                        if dtype != "F32":
                            raise LoadError("Laya temperature dtype mismatch")
                    elif dtype != "F16":
                        raise LoadError("unexpected non-F16 source tensor")
                    tensor = source.get_tensor(key)
                    expected[key].copy_(tensor if key == "temperature" else tensor.float())
                    del tensor
    except (OSError, RuntimeError, ValueError) as error:
        if isinstance(error, LoadError):
            raise
        raise LoadError("verified Safetensors load failed") from error
    receipt.assert_live()


def load_candidate(
    receipt: VerificationReceipt, candidate: Candidate, device: str
) -> LoadedCandidate:
    """Construct exactly one reviewed candidate as FP32 on explicit CPU or MPS."""
    global _MODEL_RESIDENT
    if _MODEL_RESIDENT:
        raise LoadError("only one local candidate may be resident")
    if device not in {"cpu", "mps"}:
        raise LoadError("unsupported device")
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError as error:
        raise LoadError("universal-local extra is required") from error
    if device == "mps" and (
        not torch.backends.mps.is_available()
        or os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "0"
    ):
        raise LoadError("MPS unavailable or fallback enabled")
    _preflight(receipt, candidate)
    threads = max(1, min(8, os.cpu_count() or 1))
    torch.set_num_threads(threads)
    tokenizer = _tokenizer(receipt, candidate)
    model = (
        _laya_model(
            torch,
            _closed_json(receipt, "encoder/config.json"),
            _closed_json(receipt, "rl_agent_config.json"),
        )
        if candidate.loader_family == "laya_option_marker"
        else _mdeberta_model(receipt)
        if candidate.loader_family == "mdeberta_nli"
        else None
    )
    if model is None:
        raise LoadError("unapproved loader family")
    try:
        _load_strict(receipt, candidate, model, torch)
        model.to(device).eval()
    except BaseException:
        del model
        raise
    _MODEL_RESIDENT = True
    return LoadedCandidate(
        model=model, tokenizer=tokenizer, candidate=candidate, process_threads=threads
    )
