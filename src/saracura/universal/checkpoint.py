"""Closed descriptor validator for a future Saracura-only checkpoint.

The validator accepts metadata only in Phase 4E.1.  It deliberately does not
open safetensors or permit a fixture/random tensor payload to look trained.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Annotated, Any, Literal

from pydantic import Field, model_validator

from saracura.contracts.models import ClosedModel

CHECKPOINT_ARCHITECTURE_REVISION = "saracura-universal-ranker.v0"
BASE_ENCODER_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
BASE_ENCODER_REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
_SHA256 = r"^[0-9a-f]{64}$"
_TENSOR_SHAPES = {
    "context_projection.weight": (192, 384),
    "context_projection.bias": (192,),
    "context_projection.layernorm.weight": (192,),
    "context_projection.layernorm.bias": (192,),
    "criterion_projection.weight": (192, 384),
    "criterion_projection.bias": (192,),
    "criterion_projection.layernorm.weight": (192,),
    "criterion_projection.layernorm.bias": (192,),
    "log_scale": (1,),
}


class CheckpointTensor(ClosedModel):
    name: str = Field(min_length=1, max_length=96)
    shape: tuple[Annotated[int, Field(gt=0)], ...] = Field(min_length=1, max_length=2)
    dtype: Literal["F32"]


class CheckpointDescriptor(ClosedModel):
    architecture_revision: str
    base_encoder_id: str
    base_encoder_revision: str
    base_encoder_snapshot_sha256: Annotated[str, Field(pattern=_SHA256)]
    training_manifest_sha256: Annotated[str, Field(pattern=_SHA256)]
    packet_sha256: Annotated[str, Field(pattern=_SHA256)]
    code_sha256: Annotated[str, Field(pattern=_SHA256)]
    tensors: tuple[CheckpointTensor, ...]

    @model_validator(mode="after")
    def exact_saracura_tensor_set(self) -> CheckpointDescriptor:
        if (
            self.architecture_revision != CHECKPOINT_ARCHITECTURE_REVISION
            or self.base_encoder_id != BASE_ENCODER_ID
            or self.base_encoder_revision != BASE_ENCODER_REVISION
        ):
            raise ValueError("checkpoint architecture or base encoder identity is invalid")
        names = [tensor.name for tensor in self.tensors]
        if len(names) != len(set(names)) or set(names) != set(_TENSOR_SHAPES):
            raise ValueError("checkpoint tensor set is not closed")
        for tensor in self.tensors:
            if tensor.shape != _TENSOR_SHAPES[tensor.name]:
                raise ValueError("checkpoint tensor shape is invalid")
            if "encoder" in tensor.name.lower() or "laya" in tensor.name.lower():
                raise ValueError("checkpoint cannot contain foundation or Laya tensors")
        return self


def validate_checkpoint_descriptor(value: object) -> CheckpointDescriptor:
    """Validate metadata; actual weights stay unavailable until Phase 4E.3."""

    return CheckpointDescriptor.model_validate(value)


def verify_checkpoint_tensors(tensors: Mapping[str, Any]) -> None:
    """Check the closed Phase 4E decision-layer state without importing ML packages.

    Callers own safetensors decoding.  Keeping this verifier structural makes
    default imports lightweight and prevents a checkpoint from smuggling base
    encoder, optimizer, or Laya state through a future loader.
    """

    if set(tensors) != set(_TENSOR_SHAPES):
        raise ValueError("checkpoint tensor set is not closed")
    for name, expected_shape in _TENSOR_SHAPES.items():
        value = tensors[name]
        shape = tuple(getattr(value, "shape", ()))
        dtype = str(getattr(value, "dtype", "")).replace("torch.", "")
        if shape != expected_shape or dtype != "float32":
            raise ValueError("checkpoint tensor contract is invalid")
        try:
            finite = bool(value.isfinite().all().item())
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError("checkpoint tensor contract is invalid") from error
        if not finite:
            raise ValueError("checkpoint tensor is non-finite")
    scale = tensors["log_scale"]
    try:
        log_scale = float(scale.reshape(-1)[0].item())
    except (AttributeError, IndexError, TypeError, ValueError) as error:
        raise ValueError("checkpoint log scale is invalid") from error
    if not math.isfinite(log_scale) or not -4.0 <= log_scale <= 4.0:
        raise ValueError("checkpoint log scale is invalid")
