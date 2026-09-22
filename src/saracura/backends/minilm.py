"""Explicit local-only MiniLM routing backend with a verified-bytes boundary."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import sys
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Final, Literal, cast

from pydantic import JsonValue

from saracura.backends.base import (
    BackendCalibrationMetadata,
    BackendCapabilities,
    EncodedState,
    ScoredChoice,
)
from saracura.contracts.errors import ErrorCode, SaracuraError
from saracura.contracts.models import ChoiceQuestion, ModelReference
from saracura.runtime.workflows import (
    MINILM_ROUTING_LABELS,
    MINILM_ROUTING_QUESTION,
    MINILM_ROUTING_WORKFLOW_ID,
    MINILM_ROUTING_WORKFLOW_REVISION,
)
from saracura.serialization import (
    SERIALIZER_VERSION,
    canonical_json_bytes,
    frame_segments,
    ordered_question_bytes,
    serialize_question,
)
from saracura.verified_bytes import (
    VerifiedMiniLMBytes,
    load_verified_minilm_bytes,
    package_registry_sha256,
    read_verified_minilm_snapshot,
)

MINILM_MODEL_ID: Final = "saracura-minilm-routing"
MINILM_TOKENIZER_REVISION: Final = "minilm-tokenizer-e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
MINILM_TRUNCATION_POLICY: Final = "padding-truncation-max128-v1"
MINILM_OUTPUT_TRANSFORM: Final = "linear-logits-softmax-v1"
MINILM_STATE_EXTRACTION: Final = "single-text-field-v1"
MINILM_MEAN_POOLING: Final = "attention-mask-fp32-v1"
MINILM_TOKEN_COUNT: Final = "post-truncation-attention-mask-sum-v1"
MAX_STATE_FRAME_BYTES: Final = 1_000_000
MAX_TEXT_CODEPOINTS: Final = 320
MAX_TEXT_BYTES: Final = 1_280

_CONFORMANCE_FILE: Final = "minilm-conformance.v1.json"
_SPECIAL_TOKENS: Final = {
    "unk_token": "<unk>",
    "sep_token": "</s>",
    "pad_token": "<pad>",
    "cls_token": "<s>",
    "bos_token": "<s>",
    "eos_token": "</s>",
    "mask_token": "<mask>",
}


@dataclass(frozen=True, slots=True)
class _ConformanceVector:
    text: str
    input_ids: list[list[int]]
    attention_mask: list[list[int]]
    token_type_ids: list[list[int]]
    token_payload_sha256: str
    logits: list[float]
    logits_sha256: str
    rtol: float
    atol: float


class MiniLMRoutingBackend:
    """The narrow Phase 4A execution path; importing it does not import ML code."""

    def __init__(
        self,
        *,
        encoder_snapshot: Path,
        training_manifest: Path | None = None,
        checkpoint: Path | None = None,
        training_capsule: Path | None = None,
        device: Literal["mps", "cpu"],
    ) -> None:
        self._device = device
        try:
            verified = load_verified_minilm_bytes(
                encoder_snapshot=encoder_snapshot,
                training_manifest=training_manifest,
                checkpoint=checkpoint,
                training_capsule=training_capsule,
            )
            self._verified_training_manifest = verified.training_manifest
            self._load_components(verified)
            self._architecture_descriptor = self._architecture_descriptor_for(verified)
            architecture_sha256 = hashlib.sha256(
                canonical_json_bytes(cast(JsonValue, self._architecture_descriptor))
            ).hexdigest()
            self._calibration_metadata = BackendCalibrationMetadata(
                architecture_config_sha256=architecture_sha256,
                tokenizer_revision=MINILM_TOKENIZER_REVISION,
                truncation_policy_id=MINILM_TRUNCATION_POLICY,
                precision=self.execution_path,
                quantization="none",
                output_transform=MINILM_OUTPUT_TRANSFORM,
            )
            self._model = ModelReference(
                id=MINILM_MODEL_ID,
                revision=f"minilm-routing-v1.{architecture_sha256}",
                checkpoint_sha256=hashlib.sha256(verified.checkpoint_bytes).hexdigest(),
            )
            self._capabilities = BackendCapabilities(
                decision_types=frozenset({"choice"}),
                max_questions=1,
                max_criteria=5,
                execution_boundary="local-verified-bytes-minilm",
                cold_warm_semantics="explicit-local-load-no-cache",
                quality_claims=False,
            )
            self._run_conformance_vector()
        except SaracuraError:
            raise
        except Exception as error:
            raise SaracuraError(
                ErrorCode.BACKEND_UNAVAILABLE,
                "The local MiniLM backend could not be loaded.",
                "/model",
            ) from error

    @property
    def capabilities(self) -> BackendCapabilities:
        return self._capabilities

    @property
    def model(self) -> ModelReference:
        return self._model

    @property
    def calibration_metadata(self) -> BackendCalibrationMetadata:
        return self._calibration_metadata

    @property
    def architecture_descriptor(self) -> dict[str, JsonValue]:
        """A copy prevents callers from changing the digest-bearing descriptor."""

        return cast(dict[str, JsonValue], json.loads(json.dumps(self._architecture_descriptor)))

    @property
    def execution_path(self) -> str:
        if self._device == "mps":
            return "encoder-fp32-mps-pool-fp32-mps-transfer-head-fp32-cpu-v1"
        return "encoder-fp32-cpu-pool-fp32-cpu-head-fp32-cpu-v1"

    def encode_state(self, state_payload: bytes) -> EncodedState:
        text = self._decode_state_payload(state_payload)
        tokens = self._tokenize(text)
        embedding = self._embed(tokens)
        attention = tokens["attention_mask"]
        input_tokens = int(attention.sum().item())
        return EncodedState(
            payload=state_payload,
            input_tokens=input_tokens,
            opaque=embedding,
        )

    def score_choice(
        self,
        encoded_state: EncodedState,
        question: ChoiceQuestion,
        question_payload: bytes,
    ) -> ScoredChoice:
        if ordered_question_bytes(question) != ordered_question_bytes(
            MINILM_ROUTING_QUESTION
        ) or question_payload != serialize_question(MINILM_ROUTING_QUESTION):
            raise SaracuraError(
                ErrorCode.SCHEMA_UNSUPPORTED,
                "MiniLM routing accepts only its immutable ordered question.",
                "/questions/0",
            )
        embedding = encoded_state.opaque
        if (
            not isinstance(embedding, self._torch.Tensor)
            or embedding.device.type != "cpu"
            or embedding.dtype != self._torch.float32
            or tuple(embedding.shape) != (1, 384)
            or not bool(self._torch.isfinite(embedding).all().item())
        ):
            raise SaracuraError(
                ErrorCode.BACKEND_UNAVAILABLE,
                "MiniLM received an invalid encoded state.",
                "/state",
            )
        with self._torch.inference_mode():
            values = self._head(embedding)
        if (
            tuple(values.shape) != (1, 5)
            or values.dtype != self._torch.float32
            or not bool(self._torch.isfinite(values).all().item())
        ):
            raise SaracuraError(
                ErrorCode.BACKEND_UNAVAILABLE,
                "MiniLM produced invalid routing logits.",
                "/model",
            )
        logits = [float(value) for value in values[0].tolist()]
        return ScoredChoice(
            question_id=MINILM_ROUTING_QUESTION.id,
            raw_scores=dict(zip(MINILM_ROUTING_LABELS, logits, strict=True)),
        )

    def _load_encoder_components(self, snapshot: Mapping[str, bytes]) -> None:
        try:
            safetensors = importlib.import_module("safetensors")
            safetensors_torch = importlib.import_module("safetensors.torch")
            torch = importlib.import_module("torch")
            transformers = importlib.import_module("transformers")
        except ImportError as error:
            raise ValueError("local-minilm extra is required") from error
        self._torch: Any = torch
        self._safetensors: Any = safetensors
        self._tokenizers: Any = importlib.import_module("tokenizers")
        self._BertConfig: Any = transformers.BertConfig
        self._BertModel: Any = transformers.BertModel
        self._PreTrainedTokenizerFast: Any = transformers.PreTrainedTokenizerFast
        self._safetensors_load: Any = safetensors_torch.load
        self._AddedToken: Any = transformers.AddedToken
        self._require_device()

        config = self._parse_config(snapshot["config.json"])
        tokenizer_config = self._parse_tokenizer_config(snapshot["tokenizer_config.json"])
        special_tokens = self._parse_special_tokens(snapshot["special_tokens_map.json"])
        self._parse_tokenizer_json(snapshot["tokenizer.json"])
        if config["model_type"] != "bert" or config["hidden_size"] != 384:
            raise ValueError("encoder configuration does not match the reviewed MiniLM")
        if tokenizer_config.get("tokenizer_class") != "PreTrainedTokenizerFast":
            raise ValueError("tokenizer configuration does not match the reviewed MiniLM")
        if special_tokens != _SPECIAL_TOKENS:
            raise ValueError("special token map does not match the reviewed MiniLM")

        model_config = self._BertConfig(**config)
        if model_config.model_type != "bert" or model_config.hidden_size != 384:
            raise ValueError("constructed encoder configuration is incompatible")
        encoder = self._BertModel(model_config)
        state = self._safetensors_load(snapshot["model.safetensors"])
        expected_state = encoder.state_dict()
        position_ids = state.pop("embeddings.position_ids", None)
        expected_position_ids = torch.arange(
            model_config.max_position_embeddings,
            dtype=torch.long,
        ).unsqueeze(0)
        if (
            not isinstance(position_ids, torch.Tensor)
            or position_ids.dtype != torch.long
            or not torch.equal(position_ids, expected_position_ids)
            or set(state) != set(expected_state)
        ):
            raise ValueError("encoder safetensors key set is incompatible")
        for name, expected in expected_state.items():
            value = state[name]
            if (
                not isinstance(value, torch.Tensor)
                or value.shape != expected.shape
                or value.dtype != expected.dtype
                or not value.is_floating_point()
                or not bool(torch.isfinite(value).all().item())
            ):
                raise ValueError("encoder safetensors tensor is incompatible")
        encoder.load_state_dict(state, strict=True)

        tokenizer_object = self._tokenizers.Tokenizer.from_str(
            snapshot["tokenizer.json"].decode("utf-8", "strict")
        )
        mask = self._AddedToken(
            "<mask>",
            single_word=False,
            lstrip=True,
            rstrip=False,
            normalized=True,
        )
        tokenizer = self._PreTrainedTokenizerFast(
            tokenizer_object=tokenizer_object,
            unk_token="<unk>",
            sep_token="</s>",
            pad_token="<pad>",
            cls_token="<s>",
            bos_token="<s>",
            eos_token="</s>",
            mask_token=mask,
            model_max_length=512,
            padding_side="right",
            truncation_side="right",
        )
        if (
            tokenizer.__class__.__name__ != "PreTrainedTokenizerFast"
            or tokenizer.vocab_size != 250002
            or len(tokenizer) != 250002
            or tokenizer.model_max_length != 512
            or tokenizer.padding_side != "right"
            or tokenizer.truncation_side != "right"
            or {
                "unk_token": tokenizer.unk_token,
                "sep_token": tokenizer.sep_token,
                "pad_token": tokenizer.pad_token,
                "cls_token": tokenizer.cls_token,
                "bos_token": tokenizer.bos_token,
                "eos_token": tokenizer.eos_token,
                "mask_token": tokenizer.mask_token,
            }
            != _SPECIAL_TOKENS
        ):
            raise ValueError("constructed tokenizer is incompatible")

        self._encoder: Any = encoder.to(self._device).eval()
        self._tokenizer: Any = tokenizer

    def _load_components(self, verified: VerifiedMiniLMBytes) -> None:
        self._load_encoder_components(verified.snapshot)
        torch = self._torch
        head_state = self._safetensors_load(verified.checkpoint_bytes)
        if set(head_state) != {"weight", "bias"}:
            raise ValueError("routing head safetensors key set is incompatible")
        for name, shape in (("weight", (5, 384)), ("bias", (5,))):
            value = head_state[name]
            if (
                not isinstance(value, torch.Tensor)
                or tuple(value.shape) != shape
                or value.dtype != torch.float32
                or not bool(torch.isfinite(value).all().item())
            ):
                raise ValueError("routing head tensor is incompatible")
        head = torch.nn.Linear(384, 5, device="cpu")
        head.load_state_dict(head_state, strict=True)
        self._head: Any = head.eval()

    def _require_device(self) -> None:
        if self._device not in {"mps", "cpu"}:
            raise ValueError("device must be explicit mps or cpu")
        fallback = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")
        if fallback is not None and fallback.lower() not in {"", "0", "false"}:
            raise ValueError("MPS fallback must remain disabled")
        if self._device == "mps" and not self._torch.backends.mps.is_available():
            raise ValueError("MPS is unavailable")

    @staticmethod
    def _parse_json(raw: bytes) -> dict[str, Any]:
        def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
            value: dict[str, object] = {}
            for key, child in pairs:
                if key in value:
                    raise ValueError("duplicate JSON object key")
                value[key] = child
            return value

        value = json.loads(raw, object_pairs_hook=reject_duplicates)
        if not isinstance(value, dict):
            raise ValueError("JSON root must be an object")
        return value

    def _parse_config(self, raw: bytes) -> dict[str, Any]:
        value = self._parse_json(raw)
        allowed = {
            "_name_or_path",
            "architectures",
            "attention_probs_dropout_prob",
            "classifier_dropout",
            "gradient_checkpointing",
            "hidden_act",
            "hidden_dropout_prob",
            "hidden_size",
            "initializer_range",
            "intermediate_size",
            "layer_norm_eps",
            "max_position_embeddings",
            "model_type",
            "num_attention_heads",
            "num_hidden_layers",
            "pad_token_id",
            "position_embedding_type",
            "transformers_version",
            "type_vocab_size",
            "use_cache",
            "vocab_size",
        }
        required = {
            "architectures",
            "hidden_size",
            "intermediate_size",
            "max_position_embeddings",
            "model_type",
            "num_attention_heads",
            "num_hidden_layers",
            "pad_token_id",
            "type_vocab_size",
            "vocab_size",
        }
        if set(value) - allowed or not required <= set(value) or "auto_map" in value:
            raise ValueError("encoder config shape is invalid")
        return value

    def _parse_tokenizer_config(self, raw: bytes) -> dict[str, Any]:
        value = self._parse_json(raw)
        if "auto_map" in value or value.get("tokenizer_class") != "PreTrainedTokenizerFast":
            raise ValueError("tokenizer config shape is invalid")
        return value

    def _parse_special_tokens(self, raw: bytes) -> dict[str, str]:
        value = self._parse_json(raw)
        if set(value) != set(_SPECIAL_TOKENS):
            raise ValueError("special token map shape is invalid")
        normalized: dict[str, str] = {}
        for name, expected in _SPECIAL_TOKENS.items():
            actual = value[name]
            if name == "mask_token":
                if not isinstance(actual, dict) or actual != {
                    "content": "<mask>",
                    "single_word": False,
                    "lstrip": True,
                    "rstrip": False,
                    "normalized": False,
                }:
                    raise ValueError("mask token contract is invalid")
                normalized[name] = "<mask>"
            elif not isinstance(actual, str):
                raise ValueError("special token map is invalid")
            else:
                normalized[name] = actual
            if normalized[name] != expected:
                raise ValueError("special token map is incompatible")
        return normalized

    def _parse_tokenizer_json(self, raw: bytes) -> None:
        value = self._parse_json(raw)
        if set(value) != {
            "version",
            "truncation",
            "padding",
            "added_tokens",
            "normalizer",
            "pre_tokenizer",
            "post_processor",
            "decoder",
            "model",
        }:
            raise ValueError("tokenizer JSON shape is invalid")

    def _decode_state_payload(self, state_payload: bytes) -> str:
        if len(state_payload) > MAX_STATE_FRAME_BYTES:
            raise SaracuraError(
                ErrorCode.REQUEST_INVALID,
                "Serialized state exceeds the MiniLM frame byte limit.",
                "/state",
            )
        cursor = 0
        segments: list[bytes] = []
        for _ in range(4):
            if cursor + 8 > len(state_payload):
                raise self._state_error()
            length = int.from_bytes(state_payload[cursor : cursor + 8], "big", signed=False)
            cursor += 8
            if length > MAX_STATE_FRAME_BYTES or length > len(state_payload) - cursor:
                raise self._state_error()
            segments.append(state_payload[cursor : cursor + length])
            cursor += length
        if cursor != len(state_payload):
            raise self._state_error()
        try:
            serializer, locale, domain = (item.decode("utf-8", "strict") for item in segments[:3])
        except UnicodeDecodeError as error:
            raise self._state_error() from error
        normalized = (serializer, locale, domain)
        strings_are_nfc = all(unicodedata.normalize("NFC", value) == value for value in normalized)
        expected_header = (
            SERIALIZER_VERSION,
            "pt-BR",
            "support",
        )
        if not strings_are_nfc or normalized != expected_header:
            raise self._state_error()
        try:
            state = self._parse_json(segments[3])
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            raise self._state_error() from error
        if set(state) != {"text"} or not isinstance(state["text"], str):
            raise self._state_error()
        text = state["text"]
        if (
            unicodedata.normalize("NFC", text) != text
            or not text
            or len(text) > MAX_TEXT_CODEPOINTS
            or len(text.encode("utf-8")) > MAX_TEXT_BYTES
        ):
            raise self._state_error()
        try:
            if canonical_json_bytes(cast(JsonValue, state)) != segments[3]:
                raise self._state_error()
        except SaracuraError as error:
            raise self._state_error() from error
        return text

    @staticmethod
    def _state_error() -> SaracuraError:
        return SaracuraError(
            ErrorCode.REQUEST_INVALID,
            "State is not the canonical MiniLM single-text serialization.",
            "/state",
        )

    def _tokenize(self, text: str) -> dict[str, Any]:
        tokens = self._tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        if set(tokens) != {"input_ids", "token_type_ids", "attention_mask"}:
            raise SaracuraError(
                ErrorCode.BACKEND_UNAVAILABLE,
                "MiniLM tokenizer output is incompatible.",
                "/model",
            )
        values = {name: tokens[name] for name in ("input_ids", "token_type_ids", "attention_mask")}
        if any(
            value.device.type != "cpu" or value.ndim != 2 or value.dtype != self._torch.long
            for value in values.values()
        ):
            raise SaracuraError(
                ErrorCode.BACKEND_UNAVAILABLE,
                "MiniLM tokenizer output is incompatible.",
                "/model",
            )
        return values

    def _embed(self, tokens: dict[str, Any]) -> Any:
        embedding = self._embed_batch(tokens)
        if tuple(embedding.shape) != (1, 384):
            raise SaracuraError(
                ErrorCode.BACKEND_UNAVAILABLE,
                "MiniLM embedding is incompatible.",
                "/model",
            )
        return embedding

    def _embed_batch(self, tokens: dict[str, Any]) -> Any:
        attention = tokens["attention_mask"]
        if (
            attention.ndim != 2
            or attention.shape[0] <= 0
            or bool((attention.sum(dim=1) <= 0).any().item())
        ):
            raise SaracuraError(
                ErrorCode.BACKEND_UNAVAILABLE,
                "MiniLM tokenizer produced an empty attention mask.",
                "/model",
            )
        with self._torch.inference_mode():
            if self._device == "mps":
                self._torch.mps.synchronize()
                model_tokens = {name: value.to("mps") for name, value in tokens.items()}
                hidden = self._encoder(**model_tokens).last_hidden_state.float()
                mask = model_tokens["attention_mask"].to(dtype=self._torch.float32).unsqueeze(-1)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
                self._torch.mps.synchronize()
                embedding = pooled.to("cpu")
            else:
                hidden = self._encoder(**tokens).last_hidden_state.float()
                mask = tokens["attention_mask"].to(dtype=self._torch.float32).unsqueeze(-1)
                embedding = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
        embedding = embedding.detach().to(device="cpu", dtype=self._torch.float32).contiguous()
        if (
            embedding.ndim != 2
            or embedding.shape[0] <= 0
            or tuple(embedding.shape[1:]) != (384,)
            or not bool(self._torch.isfinite(embedding).all().item())
        ):
            raise SaracuraError(
                ErrorCode.BACKEND_UNAVAILABLE,
                "MiniLM embedding is incompatible.",
                "/model",
            )
        return embedding

    def _run_conformance_vector(self) -> None:
        vector = self._conformance_vector()
        tokens = self._tokenize(vector.text)
        payload = {
            "input_ids": tokens["input_ids"].tolist(),
            "attention_mask": tokens["attention_mask"].tolist(),
            "token_type_ids": tokens["token_type_ids"].tolist(),
        }
        if (
            payload
            != {
                "input_ids": vector.input_ids,
                "attention_mask": vector.attention_mask,
                "token_type_ids": vector.token_type_ids,
            }
            or hashlib.sha256(canonical_json_bytes(cast(JsonValue, payload))).hexdigest()
            != vector.token_payload_sha256
        ):
            raise ValueError("MiniLM tokenizer conformance failed")
        embedding = self._embed(tokens)
        with self._torch.inference_mode():
            logits = self._head(embedding)[0].detach().cpu()
        expected = self._torch.tensor(vector.logits, dtype=self._torch.float32)
        self._torch.testing.assert_close(logits, expected, rtol=vector.rtol, atol=vector.atol)
        actual = [float(value) for value in logits.tolist()]
        logits_sha256 = hashlib.sha256(canonical_json_bytes(cast(JsonValue, actual))).hexdigest()
        if logits_sha256 != vector.logits_sha256:
            raise ValueError("MiniLM logits conformance digest failed")

    def _conformance_vector(self) -> _ConformanceVector:
        raw = resources.files("saracura").joinpath(_CONFORMANCE_FILE).read_bytes()
        try:
            value = MiniLMRoutingBackend._parse_json(raw)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            raise ValueError("MiniLM conformance resource is invalid") from error
        expected_keys = {
            "schema",
            "text",
            "input_ids",
            "attention_mask",
            "token_type_ids",
            "token_payload_sha256",
            "cpu_reference_logits",
            "cpu_logits_sha256",
            "mps_reference_logits",
            "mps_logits_sha256",
            "comparison",
            "labels",
        }
        if set(value) != expected_keys or value.get("schema") != "minilm-conformance.v1":
            raise ValueError("MiniLM conformance resource shape is invalid")
        vector = _conformance_from_resource(value, device=self._device)
        manifest = getattr(self, "_verified_training_manifest", None)
        if (
            isinstance(manifest, Mapping)
            and manifest.get("schema_version") == "human-training-manifest.v1"
        ):
            serving = manifest["serving_conformance"]
            logits = serving["mps_logits"] if self._device == "mps" else serving["cpu_logits"]
            logits_digest = (
                serving["mps_logits_sha256"]
                if self._device == "mps"
                else serving["cpu_logits_sha256"]
            )
            vector = _ConformanceVector(
                vector.text,
                vector.input_ids,
                vector.attention_mask,
                vector.token_type_ids,
                serving["token_payload_sha256"],
                logits,
                logits_digest,
                serving["comparison"]["rtol"],
                serving["comparison"]["atol"],
            )
        return vector

    def _architecture_descriptor_for(self, verified: VerifiedMiniLMBytes) -> dict[str, JsonValue]:
        conformance_bytes = resources.files("saracura").joinpath(_CONFORMANCE_FILE).read_bytes()
        manifest = verified.training_manifest
        runtime_abi = self._runtime_abi()
        return {
            "schema": "minilm-routing-backend.v1",
            "encoder": {
                "candidate": verified.candidate.id,
                "revision": verified.candidate.revision,
                "reviewed_registry_sha256": hashlib.sha256(
                    resources.files("saracura").joinpath("encoder-candidates.v1.json").read_bytes()
                ).hexdigest(),
                "architecture": verified.candidate.architecture,
                "hidden_width": verified.candidate.hidden_width,
            },
            "training": {
                "manifest_sha256": hashlib.sha256(verified.training_manifest_bytes).hexdigest(),
                "manifest_self_sha256": manifest["manifest_sha256"],
                "packet_manifest_sha256": manifest["packet_manifest_sha256"],
                "protocol_sha256": manifest["protocol_sha256"],
                "workflow_revision": manifest["workflow_revision"],
            },
            "checkpoint": {
                "sha256": hashlib.sha256(verified.checkpoint_bytes).hexdigest(),
                "ordered_labels": list(MINILM_ROUTING_LABELS),
                "width": 384,
                "classes": 5,
                "head_tensor_schema": {
                    "weight": {"dtype": "float32", "shape": [5, 384]},
                    "bias": {"dtype": "float32", "shape": [5]},
                },
                "checkpoint_to_index": {
                    label: index for index, label in enumerate(MINILM_ROUTING_LABELS)
                },
            },
            "workflow": {
                "id": MINILM_ROUTING_WORKFLOW_ID,
                "revision": MINILM_ROUTING_WORKFLOW_REVISION,
                "ordered_question_sha256": hashlib.sha256(
                    ordered_question_bytes(MINILM_ROUTING_QUESTION)
                ).hexdigest(),
                "locale": "pt-BR",
                "domain": "support",
            },
            "state": {
                "extraction": MINILM_STATE_EXTRACTION,
                "mean_pooling": MINILM_MEAN_POOLING,
                "maximum_length": 128,
                "truncation": True,
                "padding": True,
                "token_count": MINILM_TOKEN_COUNT,
            },
            "execution_path": self.execution_path,
            "runtime_abi": runtime_abi,
            "runtime_implementation_sha256": _runtime_implementation_sha256(),
            "platform_backend_abi_sha256": self._platform_backend_abi_sha256(),
            "conformance_sha256": hashlib.sha256(conformance_bytes).hexdigest(),
            "output_transform": MINILM_OUTPUT_TRANSFORM,
        }

    def _runtime_abi(self) -> dict[str, JsonValue]:
        return {
            "python_version": sys.version,
            "versions": {
                name: importlib.metadata.version(name)
                for name in ("torch", "transformers", "tokenizers", "safetensors")
            },
            "classes": {
                "config": f"{self._BertConfig.__module__}.{self._BertConfig.__name__}",
                "encoder": f"{self._BertModel.__module__}.{self._BertModel.__name__}",
                "tokenizer": (
                    f"{self._PreTrainedTokenizerFast.__module__}."
                    f"{self._PreTrainedTokenizerFast.__name__}"
                ),
                "tokenizers_tokenizer": (
                    f"{self._tokenizers.Tokenizer.__module__}.{self._tokenizers.Tokenizer.__name__}"
                ),
                "safetensors_loader": (
                    f"{self._safetensors_load.__module__}.{self._safetensors_load.__name__}"
                ),
            },
        }

    def _platform_backend_abi_sha256(self) -> str:
        payload: JsonValue = {
            "python_implementation": sys.implementation.name,
            "os_name": os.name,
            "os_release": platform.release(),
            "machine": platform.machine(),
            "torch_version": self._torch.__version__,
            "torch_git_version": self._torch.version.git_version,
            "torch_config_sha256": hashlib.sha256(
                self._torch.__config__.show().encode("utf-8")
            ).hexdigest(),
            "mps_built": bool(self._torch.backends.mps.is_built()),
            "mps_available": bool(self._torch.backends.mps.is_available()),
            "execution_path": self.execution_path,
        }
        return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _verified_encoder_runtime(
    encoder_snapshot: Path, device: Literal["cpu", "mps"]
) -> MiniLMRoutingBackend:
    """Construct only the reviewed encoder from a single snapshot of explicit bytes."""

    _candidate, snapshot = read_verified_minilm_snapshot(encoder_snapshot)
    return _encoder_runtime_from_snapshot(snapshot, device)


def _encoder_runtime_from_snapshot(
    snapshot: Mapping[str, bytes], device: Literal["cpu", "mps"]
) -> MiniLMRoutingBackend:
    """Build components only from already-snapshotted reviewed bytes."""

    runtime = object.__new__(MiniLMRoutingBackend)
    runtime._device = device
    runtime._load_encoder_components(snapshot)
    return runtime


def _tokens_for_texts(runtime: MiniLMRoutingBackend, texts: list[str]) -> dict[str, Any]:
    tokens = runtime._tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )
    expected = {"input_ids", "token_type_ids", "attention_mask"}
    if set(tokens) != expected:
        raise ValueError("MiniLM tokenizer output is incompatible")
    values = {name: tokens[name] for name in expected}
    if any(
        value.device.type != "cpu"
        or value.ndim != 2
        or value.shape[0] != len(texts)
        or value.dtype != runtime._torch.long
        for value in values.values()
    ):
        raise ValueError("MiniLM tokenizer output is incompatible")
    return values


def extract_verified_minilm_embeddings(
    *,
    encoder_snapshot: Path,
    texts: list[str],
    device: Literal["cpu", "mps"],
    batch_size: int,
) -> Any:
    """Run exact Phase 4A pooling from package-snapshotted bytes in batches of 32."""

    if (
        device != "mps"
        or batch_size != 32
        or not texts
        or any(not isinstance(text, str) for text in texts)
    ):
        raise ValueError("human extraction contract")
    runtime = _verified_encoder_runtime(encoder_snapshot, device)
    chunks: list[Any] = []
    for offset in range(0, len(texts), batch_size):
        batch = texts[offset : offset + batch_size]
        if not batch or len(batch) > batch_size:
            raise ValueError("human extraction batch contract")
        chunks.append(runtime._embed_batch(_tokens_for_texts(runtime, batch)))
    result = runtime._torch.cat(chunks, dim=0).to(device="cpu", dtype=runtime._torch.float32)
    if tuple(result.shape) != (len(texts), 384) or not bool(
        runtime._torch.isfinite(result).all().item()
    ):
        raise ValueError("human extraction output")
    return result.contiguous()


def _human_head(checkpoint: bytes, runtime: MiniLMRoutingBackend) -> Any:
    state = runtime._safetensors_load(checkpoint)
    if set(state) != {"weight", "bias"}:
        raise ValueError("human checkpoint tensor set")
    expected = (("weight", (5, 384)), ("bias", (5,)))
    for name, shape in expected:
        value = state[name]
        if (
            not isinstance(value, runtime._torch.Tensor)
            or tuple(value.shape) != shape
            or value.dtype != runtime._torch.float32
            or not bool(runtime._torch.isfinite(value).all().item())
        ):
            raise ValueError("human checkpoint tensor contract")
    head = runtime._torch.nn.Linear(384, 5, device="cpu")
    head.load_state_dict(state, strict=True)
    return head.eval()


def _human_conformance_on(
    snapshot: Mapping[str, bytes], checkpoint: bytes, device: Literal["cpu", "mps"]
) -> tuple[dict[str, Any], list[float]]:
    runtime = _encoder_runtime_from_snapshot(snapshot, device)
    raw = resources.files("saracura").joinpath(_CONFORMANCE_FILE).read_bytes()
    value = MiniLMRoutingBackend._parse_json(raw)
    vector = _conformance_from_resource(value, device=device)
    tokens = _tokens_for_texts(runtime, [vector.text])
    payload = {
        "input_ids": tokens["input_ids"].tolist(),
        "attention_mask": tokens["attention_mask"].tolist(),
        "token_type_ids": tokens["token_type_ids"].tolist(),
    }
    if payload != {
        "input_ids": vector.input_ids,
        "attention_mask": vector.attention_mask,
        "token_type_ids": vector.token_type_ids,
    }:
        raise ValueError("human tokenizer conformance")
    head = _human_head(checkpoint, runtime)
    with runtime._torch.inference_mode():
        logits = head(runtime._embed_batch(tokens))[0].detach().cpu()
    if tuple(logits.shape) != (5,) or not bool(runtime._torch.isfinite(logits).all().item()):
        raise ValueError("human conformance logits")
    return payload, [float(item) for item in logits.tolist()]


def _assert_human_conformance_snapshot_binding(
    candidate: Any, expected_encoder: Mapping[str, Any]
) -> None:
    """Bind the explicit Phase 4A snapshot to the sealed embedding manifest."""

    if dict(expected_encoder) != {
        "candidate": candidate.id,
        "revision": candidate.revision,
        "registry_sha256": package_registry_sha256(),
        "device": "mps",
        "frozen": True,
    }:
        raise ValueError("human conformance snapshot binding")


def human_checkpoint_conformance(
    *, encoder_snapshot: Path, checkpoint: bytes, expected_encoder: Mapping[str, Any]
) -> dict[str, Any]:
    """Produce checkpoint-specific CPU/MPS conformance evidence from real logits."""

    candidate, snapshot = read_verified_minilm_snapshot(encoder_snapshot)
    _assert_human_conformance_snapshot_binding(candidate, expected_encoder)
    cpu_payload, cpu_logits = _human_conformance_on(snapshot, checkpoint, "cpu")
    mps_payload, mps_logits = _human_conformance_on(snapshot, checkpoint, "mps")
    if cpu_payload != mps_payload or not any(cpu_logits) or not any(mps_logits):
        raise ValueError("human conformance sentinel")
    try:
        torch = importlib.import_module("torch")
        torch.testing.assert_close(
            torch.tensor(cpu_logits, dtype=torch.float32),
            torch.tensor(mps_logits, dtype=torch.float32),
            rtol=1e-5,
            atol=1e-6,
        )
    except AssertionError as exc:
        raise ValueError("human conformance equivalence") from exc
    payload_raw = canonical_json_bytes(cast(JsonValue, cpu_payload))
    return {
        "schema": "human-minilm-conformance.v1",
        "token_payload_sha256": hashlib.sha256(payload_raw).hexdigest(),
        "labels": list(MINILM_ROUTING_LABELS),
        "cpu_logits": cpu_logits,
        "mps_logits": mps_logits,
        "cpu_logits_sha256": hashlib.sha256(
            canonical_json_bytes(cast(JsonValue, cpu_logits))
        ).hexdigest(),
        "mps_logits_sha256": hashlib.sha256(
            canonical_json_bytes(cast(JsonValue, mps_logits))
        ).hexdigest(),
        "comparison": {"rtol": 0.00001, "atol": 0.000001},
    }


def _conformance_from_resource(
    value: dict[str, Any], *, device: Literal["cpu", "mps"]
) -> _ConformanceVector:
    logits_key = "cpu_reference_logits" if device == "cpu" else "mps_reference_logits"
    digest_key = "cpu_logits_sha256" if device == "cpu" else "mps_logits_sha256"
    comparison = value.get("comparison")
    if (
        value.get("labels") != list(MINILM_ROUTING_LABELS)
        or not isinstance(value.get("text"), str)
        or not isinstance(comparison, dict)
        or set(comparison) != {"rtol", "atol"}
        or not all(isinstance(comparison[name], float) for name in ("rtol", "atol"))
        or not isinstance(value.get(logits_key), list)
        or len(value[logits_key]) != 5
        or not all(isinstance(item, float) for item in value[logits_key])
        or not isinstance(value.get(digest_key), str)
        or not isinstance(value.get("token_payload_sha256"), str)
    ):
        raise ValueError("MiniLM conformance content is invalid")
    tensor_lists: list[list[list[int]]] = []
    for key in ("input_ids", "attention_mask", "token_type_ids"):
        item = value.get(key)
        if (
            not isinstance(item, list)
            or len(item) != 1
            or not isinstance(item[0], list)
            or len(item[0]) != 14
            or any(not isinstance(token, int) or isinstance(token, bool) for token in item[0])
        ):
            raise ValueError("MiniLM conformance token vector is invalid")
        tensor_lists.append(item)
    return _ConformanceVector(
        text=value["text"],
        input_ids=tensor_lists[0],
        attention_mask=tensor_lists[1],
        token_type_ids=tensor_lists[2],
        token_payload_sha256=value["token_payload_sha256"],
        logits=value[logits_key],
        logits_sha256=value[digest_key],
        rtol=comparison["rtol"],
        atol=comparison["atol"],
    )


def _runtime_implementation_sha256() -> str:
    package = Path(__file__).parents[1]
    resources_to_hash = (
        ("serialization.py", package / "serialization.py"),
        ("runtime/engine.py", package / "runtime" / "engine.py"),
        ("runtime/workflows.py", package / "runtime" / "workflows.py"),
        ("backends/minilm.py", package / "backends" / "minilm.py"),
        ("verified_bytes.py", package / "verified_bytes.py"),
        ("calibration/models.py", package / "calibration" / "models.py"),
    )
    framed: list[bytes | str] = []
    for name, path in resources_to_hash:
        framed.extend((name, path.read_bytes()))
    return hashlib.sha256(frame_segments(*framed)).hexdigest()
