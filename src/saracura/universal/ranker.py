"""Pure-Python variable-option ranker math used by Phase 4E fake-encoder tests.

This is deliberately not a backend.  It loads no model and registers no CLI;
Phase 4E.3 will bind verified MiniLM embeddings and checkpoint tensors.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from saracura.universal.rendering import RenderedTask, render_task
from saracura.universal.tasks import UniversalTask

Vector = tuple[float, ...]


class FakeEncoder(Protocol):
    def encode(self, inputs: Sequence[str]) -> tuple[Vector, ...]: ...


@dataclass(frozen=True, slots=True)
class VectorProjection:
    """Linear -> GELU -> LayerNorm projection with explicitly supplied tensors."""

    weight: tuple[Vector, ...]
    bias: Vector
    layernorm_weight: Vector
    layernorm_bias: Vector

    def __post_init__(self) -> None:
        output_size = len(self.weight)
        input_size = len(self.weight[0]) if self.weight else 0
        if not output_size or not input_size or any(len(row) != input_size for row in self.weight):
            raise ValueError("projection weight shape is invalid")
        if any(
            len(value) != output_size
            for value in (self.bias, self.layernorm_weight, self.layernorm_bias)
        ):
            raise ValueError("projection parameter shape is invalid")
        if not all(math.isfinite(value) for row in self.weight for value in row) or not all(
            math.isfinite(value)
            for values in (self.bias, self.layernorm_weight, self.layernorm_bias)
            for value in values
        ):
            raise ValueError("projection parameters must be finite")

    @property
    def input_size(self) -> int:
        return len(self.weight[0])

    @property
    def output_size(self) -> int:
        return len(self.weight)

    def project(self, value: Vector) -> Vector:
        if len(value) != self.input_size or not all(math.isfinite(item) for item in value):
            raise ValueError("encoder embedding is invalid")
        linear = tuple(
            sum(weight * item for weight, item in zip(row, value, strict=True)) + bias
            for row, bias in zip(self.weight, self.bias, strict=True)
        )
        activated = tuple(0.5 * item * (1.0 + math.erf(item / math.sqrt(2.0))) for item in linear)
        mean = sum(activated) / len(activated)
        variance = sum((item - mean) ** 2 for item in activated) / len(activated)
        normalized = tuple((item - mean) / math.sqrt(variance + 1e-5) for item in activated)
        return tuple(
            weight * item + bias
            for weight, item, bias in zip(
                self.layernorm_weight, normalized, self.layernorm_bias, strict=True
            )
        )


@dataclass(frozen=True, slots=True)
class RankerParameters:
    context_projection: VectorProjection
    criterion_projection: VectorProjection
    log_scale: float

    def __post_init__(self) -> None:
        if self.context_projection.input_size != self.criterion_projection.input_size:
            raise ValueError("context and criterion input dimensions must agree")
        if self.context_projection.output_size != self.criterion_projection.output_size:
            raise ValueError("context and criterion output dimensions must agree")
        if not math.isfinite(self.log_scale) or not -4.0 <= self.log_scale <= 4.0:
            raise ValueError("log scale is outside the bounded Phase 4E range")


def _normalized(value: Vector) -> Vector:
    length = math.sqrt(sum(item * item for item in value))
    if not math.isfinite(length) or length == 0.0:
        raise ValueError("projection output cannot be normalized")
    return tuple(item / length for item in value)


class SaracuraUniversalRanker:
    """Score exactly the declared variable options with shared bi-encoder projections."""

    def __init__(self, encoder: FakeEncoder, parameters: RankerParameters) -> None:
        self._encoder = encoder
        self._parameters = parameters

    def score(self, task: UniversalTask) -> dict[str, float]:
        rendered = render_task(task)
        vectors = self._encoder.encode((rendered.context, *rendered.criteria))
        if len(vectors) != len(task.criteria) + 1:
            raise ValueError("encoder returned a mismatched embedding batch")
        context = _normalized(self._parameters.context_projection.project(vectors[0]))
        scale = math.exp(self._parameters.log_scale)
        scores: dict[str, float] = {}
        for criterion, vector in zip(task.criteria, vectors[1:], strict=True):
            candidate = _normalized(self._parameters.criterion_projection.project(vector))
            score = scale * sum(
                left * right for left, right in zip(context, candidate, strict=True)
            )
            if not math.isfinite(score):
                raise ValueError("ranker score is non-finite")
            scores[criterion.id] = score
        return scores

    def rendered(self, task: UniversalTask) -> RenderedTask:
        return render_task(task)
