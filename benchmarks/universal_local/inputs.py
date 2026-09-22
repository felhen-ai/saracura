"""Closed textual inputs and delimiter-safe renderers for Phase 4C.2b.

This module intentionally has no optional ML imports.  Tokenizers are passed as
small protocols so the complete capacity and framing contract is unit-testable
with fakes.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


class Tokenizer(Protocol):
    @property
    def cls_token_id(self) -> int: ...

    @property
    def sep_token_id(self) -> int: ...

    @property
    def mask_token_id(self) -> int: ...

    @property
    def all_special_tokens(self) -> Sequence[str]: ...

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]: ...

    def num_special_tokens_to_add(self, *, pair: bool = False) -> int: ...

    def build_inputs_with_special_tokens(
        self, token_ids_0: list[int], token_ids_1: list[int] | None = None
    ) -> list[int]: ...


@dataclass(frozen=True)
class TextState:
    subject: str
    body: str


@dataclass(frozen=True)
class TextChoice:
    choice_id: str
    description: str


@dataclass(frozen=True)
class Rendered:
    status: str
    input_ids: tuple[int, ...] = ()
    marker_positions: tuple[int, ...] = ()
    pair_positions: tuple[int, ...] = ()
    premise: str = ""
    hypothesis: str = ""
    state_tokens_original: int = 0
    state_tokens_retained: int = 0
    choice_tokens_original: int = 0
    choice_tokens_retained: int = 0

    @property
    def truncated_state_tokens(self) -> int:
        return self.state_tokens_original - self.state_tokens_retained

    @property
    def truncated_choice_tokens(self) -> int:
        return self.choice_tokens_original - self.choice_tokens_retained


def frame_state(state: TextState) -> str:
    return f"S{len(state.subject.encode())}:{state.subject}B{len(state.body.encode())}:{state.body}"


def frame_hypothesis(instruction: str, description: str, locale: str) -> str:
    assertion = {
        "pt-BR": "A descrição acima é a opção correta.",
        "en": "The description above is the correct option.",
    }.get(locale)
    if assertion is None:
        raise ValueError("unsupported locale")
    return (
        f"I{len(instruction.encode())}:{instruction}D{len(description.encode())}:"
        f"{description}{assertion}"
    )


def _safe(values: Sequence[str], tokenizer: Tokenizer) -> bool:
    specials = tuple(token for token in tokenizer.all_special_tokens if token)
    return all(unicodedata.normalize("NFC", value) == value for value in values) and not any(
        token in value for value in values for token in specials
    )


def _nli_pair(tokenizer: Tokenizer, premise: list[int], hypothesis: list[int]) -> list[int]:
    pair = tokenizer.build_inputs_with_special_tokens(premise, hypothesis)
    if len(pair) != len(premise) + len(hypothesis) + tokenizer.num_special_tokens_to_add(pair=True):
        raise ValueError("tokenizer pair-special-token contract is invalid")
    return pair


def render_laya(
    tokenizer: Tokenizer, state: TextState, instruction: str, choices: Sequence[TextChoice]
) -> Rendered:
    if not choices or not _safe(
        [state.subject, state.body, instruction, *(choice.description for choice in choices)],
        tokenizer,
    ):
        return Rendered("unsupported_input")
    head = tokenizer.encode("choice question: " + instruction, add_special_tokens=False)
    descriptions = [
        tokenizer.encode(" " + choice.description, add_special_tokens=False) for choice in choices
    ]
    original_choices = sum(map(len, descriptions))
    retained = [item[:48] for item in descriptions]
    # Equal share applies to the 48-token-capped descriptions, not to source
    # values and not to their structural mask IDs.
    fixed = 1 + len(head) + 1 + len(choices) + 1
    available = 256 - fixed
    if available < 4 * len(choices):
        return Rendered("unsupported_capacity")
    if sum(map(len, retained)) > available:
        share, remainder = divmod(available, len(choices))
        retained = [
            item[: share + (1 if index < remainder else 0)] for index, item in enumerate(retained)
        ]
        if any(len(item) < 4 for item in retained):
            return Rendered("unsupported_capacity")
    state_ids = tokenizer.encode(frame_state(state), add_special_tokens=False)
    room = 1024 - (fixed + sum(map(len, retained))) - 1
    state_kept = state_ids[:room]
    if not state_kept:
        return Rendered("unsupported_capacity")
    ids = [tokenizer.cls_token_id, *head, tokenizer.sep_token_id]
    positions: list[int] = []
    for item in retained:
        positions.append(len(ids))
        ids.extend((tokenizer.mask_token_id, *item))
    ids.extend((tokenizer.sep_token_id, *state_kept, tokenizer.sep_token_id))
    choice_truncated = sum(map(len, retained)) < original_choices
    state_truncated = len(state_kept) < len(state_ids)
    status = (
        "supported_with_state_and_choice_truncation"
        if state_truncated and choice_truncated
        else "supported_with_state_truncation"
        if state_truncated
        else "supported_with_choice_truncation"
        if choice_truncated
        else "supported"
    )
    return Rendered(
        status,
        tuple(ids),
        tuple(positions),
        (),
        state_tokens_original=len(state_ids),
        state_tokens_retained=len(state_kept),
        choice_tokens_original=original_choices,
        choice_tokens_retained=sum(map(len, retained)),
    )


def render_nli(
    tokenizer: Tokenizer, state: TextState, instruction: str, choice: TextChoice, locale: str
) -> Rendered:
    if not _safe([state.subject, state.body, instruction, choice.description], tokenizer):
        return Rendered("unsupported_input")
    try:
        hypothesis = frame_hypothesis(instruction, choice.description, locale)
    except ValueError:
        return Rendered("unsupported_input")
    hypothesis_ids = tokenizer.encode(hypothesis, add_special_tokens=False)
    original_premise = frame_state(state)
    original_premise_ids = tokenizer.encode(original_premise, add_special_tokens=False)
    if not state.body:
        return Rendered("unsupported_capacity")

    def render_prefix(length: int) -> tuple[str, list[int], list[int]]:
        retained = TextState(state.subject, state.body[:length])
        premise = frame_state(retained)
        premise_ids = tokenizer.encode(premise, add_special_tokens=False)
        return premise, premise_ids, _nli_pair(tokenizer, premise_ids, hypothesis_ids)

    try:
        premise, premise_ids, pair_ids = render_prefix(len(state.body))
        retained_body_length = len(state.body)
        if len(pair_ids) > 512:
            low, high = 1, len(state.body) - 1
            best: tuple[int, str, list[int], list[int]] | None = None
            while low <= high:
                middle = (low + high) // 2
                candidate_premise, candidate_ids, candidate_pair = render_prefix(middle)
                if len(candidate_pair) <= 512:
                    best = (middle, candidate_premise, candidate_ids, candidate_pair)
                    low = middle + 1
                else:
                    high = middle - 1
            if best is None:
                return Rendered("unsupported_capacity")
            retained_body_length, premise, premise_ids, pair_ids = best
    except ValueError:
        return Rendered("unsupported_input")
    status = (
        "supported_with_state_truncation" if retained_body_length < len(state.body) else "supported"
    )
    return Rendered(
        status,
        input_ids=tuple(pair_ids),
        pair_positions=(
            1,
            1 + len(premise_ids),
            2 + len(premise_ids),
            2 + len(premise_ids) + len(hypothesis_ids),
        ),
        premise=premise,
        hypothesis=hypothesis,
        state_tokens_original=len(original_premise_ids),
        state_tokens_retained=len(premise_ids),
        choice_tokens_original=len(hypothesis_ids),
        choice_tokens_retained=len(hypothesis_ids),
    )
