"""Opt-in, local-only Laya universal Choice backend.

All optional ML imports are delayed until backend construction.  The backend
does not acquire models, discover caches, or permit input truncation.
"""

from __future__ import annotations

import json
import math
import os
import threading
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, cast

from pydantic import JsonValue

from saracura.backends.base import BackendCapabilities, ScoredChoice
from saracura.contracts.errors import ErrorCode, SaracuraError
from saracura.contracts.models import ChoiceQuestion, DecisionRequest, ModelReference
from saracura.laya_snapshot import (
    LayaSnapshotError,
    LayaSnapshotReceipt,
    load_laya_candidate,
    verify_laya_snapshot,
)
from saracura.serialization import canonical_json_bytes

MAX_INPUT_TOKENS = 1024
MAX_PREFIX_TOKENS = 256
_MIN_FREE_DISK = 128 * 1024 * 1024
_MIN_FREE_MEMORY = 512 * 1024 * 1024
_MODEL_RESIDENT = False
_MODEL_STATE_LOCK = threading.Lock()


class LayaTokenizer(Protocol):
    @property
    def cls_token_id(self) -> int: ...

    @property
    def sep_token_id(self) -> int: ...

    @property
    def mask_token_id(self) -> int: ...

    @property
    def all_special_tokens(self) -> Sequence[str]: ...

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]: ...


@dataclass(frozen=True, slots=True)
class RenderedLayaChoice:
    input_ids: tuple[int, ...]
    marker_positions: tuple[int, ...]

    @property
    def input_tokens(self) -> int:
        return len(self.input_ids)


class LayaBackendError(ValueError):
    """Internal closed-loader error; its detail is never exposed publicly."""


@dataclass(frozen=True, slots=True)
class _DirectTokenizer:
    backend: Any
    cls_token_id: int
    sep_token_id: int
    mask_token_id: int
    all_special_tokens: tuple[str, ...]

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        return list(self.backend.encode(text, add_special_tokens=add_special_tokens).ids)


def _nfc(value: object) -> bool:
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value) == value
    if isinstance(value, dict):
        return all(
            isinstance(key, str) and _nfc(key) and _nfc(child) for key, child in value.items()
        )
    if isinstance(value, list):
        return all(_nfc(child) for child in value)
    return True


def _frame(subject: str, body: str) -> str:
    return f"S{len(subject.encode('utf-8'))}:{subject}B{len(body.encode('utf-8'))}:{body}"


def render_laya_choice(
    tokenizer: LayaTokenizer, request: DecisionRequest, question: ChoiceQuestion
) -> RenderedLayaChoice:
    """Render complete reviewed marker input or fail before model execution."""

    state = cast(JsonValue, request.state)
    if not _nfc(state) or not all(
        _nfc(value)
        for value in (
            request.locale,
            request.domain,
            question.instruction,
            *(criterion.id for criterion in question.criteria),
            *(criterion.description for criterion in question.criteria),
        )
    ):
        raise SaracuraError(ErrorCode.REQUEST_INVALID, "Universal input must already be NFC.", "/")
    state_json = canonical_json_bytes(state).decode("utf-8", "strict")
    subject = f"locale={request.locale};domain={request.domain}"
    strings = (
        subject,
        state_json,
        question.instruction,
        *(criterion.id for criterion in question.criteria),
        *(criterion.description for criterion in question.criteria),
    )
    special_tokens = tuple(token for token in tokenizer.all_special_tokens if token)
    if any(token in value for token in special_tokens for value in strings):
        raise SaracuraError(
            ErrorCode.REQUEST_INVALID,
            "Universal input contains a tokenizer special token.",
            "/",
        )
    try:
        head = tokenizer.encode(
            "choice question: " + question.instruction, add_special_tokens=False
        )
        descriptions = [
            tokenizer.encode(" " + criterion.description, add_special_tokens=False)
            for criterion in question.criteria
        ]
        state_ids = tokenizer.encode(_frame(subject, state_json), add_special_tokens=False)
    except (TypeError, ValueError) as error:
        raise SaracuraError(
            ErrorCode.BACKEND_UNAVAILABLE,
            "Laya tokenizer output is incompatible.",
            "/model",
        ) from error
    if not head or not state_ids or any(not item for item in descriptions):
        raise SaracuraError(
            ErrorCode.CAPACITY_EXCEEDED,
            "Complete universal input does not fit the Laya envelope.",
            "/questions",
        )
    fixed = 1 + len(head) + 1 + len(descriptions) + 1
    prefix = fixed + sum(len(item) for item in descriptions)
    if prefix > MAX_PREFIX_TOKENS or prefix + len(state_ids) + 1 > MAX_INPUT_TOKENS:
        raise SaracuraError(
            ErrorCode.CAPACITY_EXCEEDED,
            "Complete universal input does not fit the Laya envelope.",
            "/questions",
        )
    identifiers = (tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.mask_token_id)
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in identifiers
    ):
        raise SaracuraError(
            ErrorCode.BACKEND_UNAVAILABLE,
            "Laya tokenizer output is incompatible.",
            "/model",
        )
    ids = [tokenizer.cls_token_id, *head, tokenizer.sep_token_id]
    positions: list[int] = []
    for description in descriptions:
        positions.append(len(ids))
        ids.extend((tokenizer.mask_token_id, *description))
    ids.extend((tokenizer.sep_token_id, *state_ids, tokenizer.sep_token_id))
    if len(ids) > MAX_INPUT_TOKENS or len(ids) != prefix + len(state_ids) + 1:
        raise SaracuraError(
            ErrorCode.CAPACITY_EXCEEDED,
            "Complete universal input does not fit the Laya envelope.",
            "/questions",
        )
    return RenderedLayaChoice(tuple(ids), tuple(positions))


def _closed_json(receipt: LayaSnapshotReceipt, path: str) -> dict[str, Any]:
    try:
        value = json.loads(receipt.read_verified_bytes(path).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, LayaSnapshotError) as error:
        raise LayaBackendError("verified metadata is invalid") from error
    if not isinstance(value, dict) or "auto_map" in value or value.get("trust_remote_code"):
        raise LayaBackendError("metadata is not closed")
    return value


def _require(data: dict[str, Any], required: dict[str, Any], error: str) -> None:
    if any(data.get(key) != value for key, value in required.items()):
        raise LayaBackendError(error)


def _direct_tokenizer(receipt: LayaSnapshotReceipt) -> _DirectTokenizer:
    try:
        from tokenizers import Tokenizer  # type: ignore[import-not-found]
    except ImportError as error:
        raise LayaBackendError("universal-local extra is required") from error
    config = _closed_json(receipt, "tokenizer/tokenizer_config.json")
    required = {
        "cls_token": "<bos>",
        "sep_token": "<eos>",
        "mask_token": "<mask>",
        "pad_token": "<pad>",
        "unk_token": "<unk>",
    }
    _require(config, required, "tokenizer special-token contract mismatch")
    try:
        tokenizer_json = receipt.read_verified_bytes("tokenizer/tokenizer.json").decode("utf-8")
        backend = Tokenizer.from_str(tokenizer_json)
    except (UnicodeDecodeError, ValueError, LayaSnapshotError) as error:
        raise LayaBackendError("verified tokenizer graph is invalid") from error
    ids = {name: backend.token_to_id(value) for name, value in required.items()}
    if any(not isinstance(value, int) for value in ids.values()):
        raise LayaBackendError("tokenizer structural IDs missing")
    return _DirectTokenizer(
        backend=backend,
        cls_token_id=ids["cls_token"],
        sep_token_id=ids["sep_token"],
        mask_token_id=ids["mask_token"],
        all_special_tokens=tuple(required.values()),
    )


def _preflight(receipt: LayaSnapshotReceipt) -> None:
    try:
        import psutil  # type: ignore[import-untyped]

        available = int(psutil.virtual_memory().available)
        free_disk = receipt.descriptor_free_bytes("model.safetensors")
    except (ImportError, OSError, LayaSnapshotError) as error:
        raise LayaBackendError("local resource preflight unavailable") from error
    if (
        available < receipt.candidate.total_bytes * 2 + _MIN_FREE_MEMORY
        or free_disk < _MIN_FREE_DISK
    ):
        raise LayaBackendError("insufficient local resources")


def _laya_model(torch: Any, config_data: dict[str, Any], head_data: dict[str, Any]) -> Any:
    try:
        from transformers import ModernBertConfig, ModernBertModel  # type: ignore[import-not-found]
    except ImportError as error:
        raise LayaBackendError("universal-local extra is required") from error
    _require(
        config_data,
        {
            "model_type": "modernbert",
            "hidden_size": 768,
            "num_hidden_layers": 22,
            "num_attention_heads": 12,
            "vocab_size": 256000,
        },
        "ModernBERT configuration mismatch",
    )
    _require(
        head_data,
        {"head_layers": 2, "max_len": 1024, "head_max_len": 256, "temperature": [1.0, 1.0, 1.0]},
        "Laya head configuration mismatch",
    )
    if head_data.get("temperature_by_options") != {}:
        raise LayaBackendError("Laya head metadata is not closed")
    config = ModernBertConfig(**config_data)
    config._attn_implementation = "sdpa"

    class _LayaDecisionModel(torch.nn.Module):  # type: ignore[misc]
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
            self, input_ids: Any, attention_mask: Any, marker_pos: Any, marker_mask: Any, qtype: Any
        ) -> Any:
            encoded = self.encoder(
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state
            refined = encoded + self.type_emb(qtype)[:, None, :]
            padding = ~attention_mask.bool()
            for head_layer in self.head.layers:
                refined = head_layer(refined, src_key_padding_mask=padding)
            gather_index = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, refined.size(-1))
            markers = torch.gather(refined, 1, gather_index)
            return self.scorer(markers).squeeze(-1).float().masked_fill(~marker_mask, -1e4)

    return _LayaDecisionModel().float()


def _load_weights(receipt: LayaSnapshotReceipt, model: Any, torch: Any) -> None:
    try:
        from safetensors import safe_open  # type: ignore[import-not-found]
    except ImportError as error:
        raise LayaBackendError("universal-local extra is required") from error
    receipt.assert_live()
    expected = model.state_dict()
    try:
        with safe_open(
            receipt.descriptor_path("model.safetensors"), framework="pt", device="cpu"
        ) as source:
            keys = set(source.keys())
            dtypes: dict[str, int] = {}
            for key in keys:
                source_dtype = str(source.get_slice(key).get_dtype())
                dtypes[source_dtype] = dtypes.get(source_dtype, 0) + 1
            if dtypes != {"F16": 169, "F32": 1} or keys != set(expected):
                raise LayaBackendError("source tensor contract mismatch")
            temperature = source.get_slice("temperature")
            if list(temperature.get_shape()) != [3] or str(temperature.get_dtype()) != "F32":
                raise LayaBackendError("temperature tensor contract mismatch")
            with torch.no_grad():
                for key in sorted(keys):
                    descriptor = source.get_slice(key)
                    if list(descriptor.get_shape()) != list(expected[key].shape):
                        raise LayaBackendError("source tensor shape mismatch")
                    dtype = str(descriptor.get_dtype())
                    if (key == "temperature" and dtype != "F32") or (
                        key != "temperature" and dtype != "F16"
                    ):
                        raise LayaBackendError("source tensor dtype mismatch")
                    tensor = source.get_tensor(key)
                    if not bool(torch.isfinite(tensor).all().item()):
                        raise LayaBackendError("source tensor is non-finite")
                    expected[key].copy_(tensor if key == "temperature" else tensor.float())
    except (OSError, RuntimeError, ValueError, LayaSnapshotError) as error:
        if isinstance(error, LayaBackendError):
            raise
        raise LayaBackendError("verified Safetensors load failed") from error
    for value in model.state_dict().values():
        if value.is_floating_point() and (
            value.dtype != torch.float32 or not bool(torch.isfinite(value).all().item())
        ):
            raise LayaBackendError("constructed model is incompatible")
    receipt.assert_live()


class LayaUniversalBackend:
    """One resident FP32 Laya model backed only by an explicit snapshot path."""

    def __init__(
        self,
        *,
        model_snapshot: Path,
        device: Literal["cpu", "mps"],
        cpu_threads: int | None = None,
    ) -> None:
        try:
            if device not in {"cpu", "mps"}:
                raise LayaBackendError("device must be explicit")
            threads = cpu_threads if cpu_threads is not None else min(8, os.cpu_count() or 1)
            if not isinstance(threads, int) or isinstance(threads, bool) or not 1 <= threads <= 8:
                raise LayaBackendError("CPU thread count is outside the reviewed range")
            candidate = load_laya_candidate()
            self._model_snapshot = Path(model_snapshot)
            self._device = device
            self._threads = threads
            self._load_lock = threading.Lock()
            self._model_loaded = False
            self._closed = False
            self._receipt: LayaSnapshotReceipt | None = None
            self._model_object: Any = None
            self._torch: Any = None
            self._tokenizer: LayaTokenizer | None = None
            self._capabilities = BackendCapabilities(
                execution_tier="universal",
                decision_types=frozenset({"choice"}),
                max_questions=10,
                max_criteria=20,
                execution_boundary="explicit-verified-local-laya",
                cold_warm_semantics="one-process-resident-model",
                quality_claims=False,
            )
            self._reference = ModelReference(
                id=candidate.model_id,
                revision=candidate.model_revision,
                checkpoint_sha256=candidate.checkpoint_sha256,
            )
        except (LayaBackendError, LayaSnapshotError, OSError) as error:
            raise SaracuraError(
                ErrorCode.BACKEND_UNAVAILABLE, "Laya backend is unavailable.", "/model"
            ) from error

    @property
    def capabilities(self) -> BackendCapabilities:
        return self._capabilities

    @property
    def model(self) -> ModelReference:
        return self._reference

    def prepare(self) -> None:
        """Verify and load the explicit snapshot without accepting a decision request."""

        with self._load_lock:
            self._ensure_loaded_locked()

    def validate_request(self, request: DecisionRequest) -> None:
        """Validate tokenizer-independent universal input without touching the snapshot."""

        state = cast(JsonValue, request.state)
        if not _nfc(state) or not all(
            _nfc(value)
            for value in (
                request.locale,
                request.domain,
                *(question.instruction for question in request.questions),
                *(
                    criterion.id
                    for question in request.questions
                    for criterion in question.criteria
                ),
                *(
                    criterion.description
                    for question in request.questions
                    for criterion in question.criteria
                ),
            )
        ):
            raise SaracuraError(
                ErrorCode.REQUEST_INVALID,
                "Universal input must already be NFC.",
                "/",
            )

    def _ensure_loaded_locked(self) -> None:
        self._ensure_tokenizer_loaded_locked()
        self._ensure_model_loaded_locked()

    def _ensure_tokenizer_loaded_locked(self) -> None:
        if self._closed:
            raise SaracuraError(
                ErrorCode.BACKEND_UNAVAILABLE, "Laya backend is unavailable.", "/model"
            )
        if self._tokenizer is not None:
            return
        receipt: LayaSnapshotReceipt | None = None
        try:
            receipt = verify_laya_snapshot(self._model_snapshot)
            tokenizer = _direct_tokenizer(receipt)
            receipt.assert_live()
            self._receipt = receipt
            self._tokenizer = tokenizer
        except (LayaBackendError, LayaSnapshotError, OSError, RuntimeError) as error:
            if receipt is not None:
                receipt.close()
            raise SaracuraError(
                ErrorCode.BACKEND_UNAVAILABLE, "Laya backend is unavailable.", "/model"
            ) from error

    def _ensure_model_loaded_locked(self) -> None:
        global _MODEL_RESIDENT
        if self._closed:
            raise SaracuraError(
                ErrorCode.BACKEND_UNAVAILABLE, "Laya backend is unavailable.", "/model"
            )
        if self._model_loaded:
            return
        self._ensure_tokenizer_loaded_locked()
        receipt = self._receipt
        if receipt is None:
            raise SaracuraError(
                ErrorCode.BACKEND_UNAVAILABLE, "Laya backend is unavailable.", "/model"
            )
        with _MODEL_STATE_LOCK:
            if _MODEL_RESIDENT:
                raise SaracuraError(
                    ErrorCode.BACKEND_UNAVAILABLE, "Laya backend is unavailable.", "/model"
                )
            _MODEL_RESIDENT = True
        try:
            try:
                import torch  # type: ignore[import-not-found]
            except ImportError as error:
                raise LayaBackendError("universal-local extra is required") from error
            if self._device == "mps" and (
                not torch.backends.mps.is_available()
                or os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") not in {None, "0"}
            ):
                raise LayaBackendError("MPS is unavailable or fallback is enabled")
            _preflight(receipt)
            torch.set_num_threads(self._threads)
            model = _laya_model(
                torch,
                _closed_json(receipt, "encoder/config.json"),
                _closed_json(receipt, "rl_agent_config.json"),
            )
            _load_weights(receipt, model, torch)
            receipt.assert_live()
            self._model_object = model.to(self._device).eval()
            self._torch = torch
            self._model_loaded = True
        except (LayaBackendError, LayaSnapshotError, OSError, RuntimeError) as error:
            with _MODEL_STATE_LOCK:
                _MODEL_RESIDENT = False
            raise SaracuraError(
                ErrorCode.BACKEND_UNAVAILABLE, "Laya backend is unavailable.", "/model"
            ) from error

    def close(self) -> None:
        global _MODEL_RESIDENT
        with self._load_lock:
            if self._closed:
                return
            self._closed = True
            if self._receipt is not None:
                try:
                    self._receipt.close()
                finally:
                    self._receipt = None
                    self._model_object = None
                    self._torch = None
                    self._tokenizer = None
                    if self._model_loaded:
                        self._model_loaded = False
                        with _MODEL_STATE_LOCK:
                            _MODEL_RESIDENT = False

    def score_universal_choice(
        self, request: DecisionRequest, question: ChoiceQuestion, state_payload: bytes
    ) -> ScoredChoice:
        del state_payload  # The universal renderer intentionally uses canonical state JSON jointly.
        self.validate_request(request)
        with self._load_lock:
            try:
                self._ensure_tokenizer_loaded_locked()
                receipt = self._receipt
                tokenizer = self._tokenizer
                if receipt is None or tokenizer is None:
                    raise LayaBackendError("Laya runtime state is unavailable")
                receipt.assert_live()
                rendered = render_laya_choice(tokenizer, request, question)
                self._ensure_model_loaded_locked()
                torch = self._torch
                input_ids = torch.tensor(
                    [rendered.input_ids], dtype=torch.long, device=self._device
                )
                attention = torch.ones_like(input_ids, dtype=torch.long)
                positions = torch.tensor(
                    [rendered.marker_positions], dtype=torch.long, device=self._device
                )
                marker_mask = torch.ones_like(positions, dtype=torch.bool)
                qtype = torch.zeros(1, dtype=torch.long, device=self._device)
                with torch.inference_mode():
                    values = self._model_object(input_ids, attention, positions, marker_mask, qtype)
                values = values.detach().to(device="cpu", dtype=torch.float32)
                if tuple(values.shape) != (1, len(question.criteria)) or not bool(
                    torch.isfinite(values).all().item()
                ):
                    raise LayaBackendError("Laya logits are incompatible")
                scores = {
                    criterion.id: float(value)
                    for criterion, value in zip(question.criteria, values[0].tolist(), strict=True)
                }
                if any(not math.isfinite(value) for value in scores.values()):
                    raise LayaBackendError("Laya logits are non-finite")
                receipt.assert_live()
                return ScoredChoice(
                    question_id=question.id,
                    raw_scores=scores,
                    input_tokens=rendered.input_tokens,
                )
            except SaracuraError:
                raise
            except (LayaBackendError, LayaSnapshotError, RuntimeError, ValueError) as error:
                raise SaracuraError(
                    ErrorCode.BACKEND_UNAVAILABLE, "Laya backend is unavailable.", "/model"
                ) from error
