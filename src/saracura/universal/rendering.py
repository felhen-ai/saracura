"""Canonical byte-length-framed renderer for planned Phase 4E tasks."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from saracura.serialization import canonical_json_bytes
from saracura.universal.tasks import UniversalTask

RENDERER_REVISION = "phase4e-universal-renderer.v1"
MAX_CONTEXT_TOKENS = 128
MAX_CRITERION_TOKENS = 96
MAX_CRITERION_BATCH = 20


class CapacityError(ValueError):
    """A complete rendered semantic segment cannot fit; it must not be shortened."""


class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...


@dataclass(frozen=True, slots=True)
class RenderedTask:
    context: str
    criteria: tuple[str, ...]


def _render(*segments: bytes | str) -> str:
    """Return valid UTF-8 text whose semantic fields remain byte-length framed.

    The MiniLM tokenizer consumes Unicode text, so binary length prefixes (or a
    Latin-1 surrogate transport) would alter PT-BR and English input before the
    model sees it.  A decimal ASCII byte count followed by the untouched UTF-8
    field is still self-delimiting: a parser consumes exactly the declared bytes
    and never searches for a separator inside user data.
    """

    framed: list[str] = []
    for segment in segments:
        payload = segment.encode("utf-8") if isinstance(segment, str) else segment
        framed.append(f"{len(payload)}:{payload.decode('utf-8')}")
    return "".join(framed)


def render_task(task: UniversalTask) -> RenderedTask:
    """Render context once and criteria in declared order, with no implicit sorting."""

    context = _render(
        RENDERER_REVISION,
        "context",
        task.locale,
        task.domain,
        task.instruction,
        canonical_json_bytes(task.state),
    )
    criteria = tuple(
        _render(RENDERER_REVISION, "criterion", criterion.id, criterion.description)
        for criterion in task.criteria
    )
    return RenderedTask(context=context, criteria=criteria)


def validate_rendered_capacity(rendered: RenderedTask, counter: TokenCounter) -> None:
    """Check whole inputs independently before an encoder may see any of them."""

    if len(rendered.criteria) > MAX_CRITERION_BATCH:
        raise CapacityError("criterion batch exceeds the closed maximum")
    if counter.count(rendered.context) > MAX_CONTEXT_TOKENS:
        raise CapacityError("context exceeds capacity")
    if any(counter.count(criterion) > MAX_CRITERION_TOKENS for criterion in rendered.criteria):
        raise CapacityError("criterion exceeds capacity")


def rendered_bytes(rendered: RenderedTask) -> Sequence[bytes]:
    """Expose exact UTF-8 framed bytes for a future tokenizer adapter."""

    return (rendered.context.encode("utf-8"), *(item.encode("utf-8") for item in rendered.criteria))
