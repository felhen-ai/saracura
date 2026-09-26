"""Installed torch implementation of the sealed Phase 4E projection ranker.

The module deliberately accepts a torch module object rather than importing
torch itself, preserving the default package's optional-dependency boundary.
"""

from __future__ import annotations

from typing import Any


class TorchSaracuraRanker:
    """FP32 projection ranker with the exact sealed operation sequence."""

    def __init__(self, torch: Any, tensors: dict[str, Any], *, device: str) -> None:
        self._torch = torch
        self._functional = torch.nn.functional
        self._device = device
        self._context_weight = tensors["context_projection.weight"].to(
            device=device, dtype=torch.float32
        )
        self._context_bias = tensors["context_projection.bias"].to(
            device=device, dtype=torch.float32
        )
        self._context_norm_weight = tensors["context_projection.layernorm.weight"].to(
            device=device, dtype=torch.float32
        )
        self._context_norm_bias = tensors["context_projection.layernorm.bias"].to(
            device=device, dtype=torch.float32
        )
        self._criterion_weight = tensors["criterion_projection.weight"].to(
            device=device, dtype=torch.float32
        )
        self._criterion_bias = tensors["criterion_projection.bias"].to(
            device=device, dtype=torch.float32
        )
        self._criterion_norm_weight = tensors["criterion_projection.layernorm.weight"].to(
            device=device, dtype=torch.float32
        )
        self._criterion_norm_bias = tensors["criterion_projection.layernorm.bias"].to(
            device=device, dtype=torch.float32
        )
        self._log_scale = tensors["log_scale"].to(device=device, dtype=torch.float32)

    def _project(self, value: Any, *, kind: str) -> Any:
        if kind == "context":
            weight, bias = self._context_weight, self._context_bias
            norm_weight, norm_bias = self._context_norm_weight, self._context_norm_bias
        else:
            weight, bias = self._criterion_weight, self._criterion_bias
            norm_weight, norm_bias = self._criterion_norm_weight, self._criterion_norm_bias
        # This mirrors the sealed training implementation: linear -> GELU ->
        # LayerNorm. L2 normalization and cosine ranking happen in logits().
        return self._functional.layer_norm(
            self._functional.gelu(value @ weight.T + bias),
            (192,),
            norm_weight,
            norm_bias,
        )

    def logits(self, context: Any, criteria: Any) -> Any:
        """Return one finite unmasked logit per supplied criterion in order."""

        torch = self._torch
        if (
            context.dtype != torch.float32
            or criteria.dtype != torch.float32
            or tuple(context.shape) != (1, 384)
            or criteria.ndim != 2
            or tuple(criteria.shape[1:]) != (384,)
            or criteria.shape[0] < 1
        ):
            raise ValueError("ranker input contract is invalid")
        projected_context = self._functional.normalize(
            self._project(context, kind="context"), dim=-1
        )
        projected_criteria = self._functional.normalize(
            self._project(criteria, kind="criterion"), dim=-1
        )
        logits = (projected_context @ projected_criteria.T).reshape(-1) * self._log_scale.exp()
        if not bool(torch.isfinite(logits).all().item()):
            raise ValueError("ranker logits are non-finite")
        return logits
