"""Pure deterministic protocol logic for the native PT-BR benchmark."""

from __future__ import annotations

import hashlib
import math
import re
import struct
import unicodedata
from dataclasses import dataclass
from typing import Any, cast

from pydantic import JsonValue

from saracura.contracts.models import (
    ChoiceCriterion,
    ChoiceQuestion,
    DecisionRequest,
    WorkflowReference,
)
from saracura.serialization import canonical_json_bytes

SEED = "saracura-phase5d-faq-bacen-v1"
PLAN_SHA256 = "089a83874ed0cc57da45eb143f4b6f6494125dbfabddb660f8e7ca5105175f6f"
OPTION_IDS = ("a", "b", "c", "d")
INSTRUCTION = "Escolha a alternativa que começa a responder corretamente à pergunta."
_ID = re.compile(r"^[\x21-\x7e]+$")
_TOKEN = re.compile(r"[^\W_]+", flags=re.UNICODE)


@dataclass(frozen=True, slots=True)
class PlannedRow:
    query_id: str
    query_text: str
    ordered_choice_ids: tuple[str, str, str, str]
    ordered_choice_snippets: tuple[str, str, str, str]
    gold_position: int

    def digest_record(self) -> dict[str, JsonValue]:
        return {
            "query_id": self.query_id,
            "query_sha256": hashlib.sha256(self.query_text.encode("utf-8")).hexdigest(),
            "ordered_choice_ids": list(self.ordered_choice_ids),
            "ordered_choice_snippet_sha256": [
                hashlib.sha256(value.encode("utf-8")).hexdigest()
                for value in self.ordered_choice_snippets
            ],
            "gold_position": self.gold_position,
        }


def normalize_text(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("dataset text must be a string")
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", value)).strip()


def answer_snippet(value: str) -> str:
    return normalize_text(value)[:120]


def framed_hash(*segments: str) -> bytes:
    digest = hashlib.sha256()
    for segment in segments:
        encoded = segment.encode("utf-8")
        digest.update(struct.pack(">Q", len(encoded)))
        digest.update(encoded)
    return digest.digest()


def lexical_tokens(value: str) -> set[str]:
    normalized = unicodedata.normalize("NFC", value).casefold()
    return {token for token in _TOKEN.findall(normalized) if len(token) >= 4}


def earliest_argmax(values: list[float]) -> int:
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("argmax requires finite values")
    return max(range(len(values)), key=values.__getitem__)


def _valid_id(value: object) -> str:
    if not isinstance(value, str) or not value or "\x00" in value or _ID.fullmatch(value) is None:
        raise ValueError("dataset ids must be non-empty printable ASCII")
    return value


def build_plan(
    corpus_rows: list[dict[str, Any]],
    query_rows: list[dict[str, Any]],
    qrel_rows: list[dict[str, Any]],
    *,
    enforce_frozen_counts: bool = True,
) -> tuple[PlannedRow, ...]:
    expected = (
        (len(corpus_rows), len(query_rows), len(qrel_rows)) == (1673, 373, 373)
        if enforce_frozen_counts
        else bool(corpus_rows and query_rows and qrel_rows)
    )
    if not expected:
        raise ValueError("dataset row cardinalities drifted")
    corpus: dict[str, str] = {}
    for row in corpus_rows:
        if set(row) != {"_id", "title", "text"}:
            raise ValueError("corpus columns drifted")
        corpus_id = _valid_id(row["_id"])
        if corpus_id in corpus:
            raise ValueError("duplicate corpus id")
        corpus[corpus_id] = answer_snippet(row["text"])
    queries: dict[str, str] = {}
    for row in query_rows:
        if set(row) != {"_id", "text"}:
            raise ValueError("query columns drifted")
        query_id = _valid_id(row["_id"])
        if query_id in queries:
            raise ValueError("duplicate query id")
        queries[query_id] = normalize_text(row["text"])
    gold: dict[str, str] = {}
    for row in qrel_rows:
        if set(row) != {"query-id", "corpus-id", "score"}:
            raise ValueError("qrel columns drifted")
        query_id = _valid_id(row["query-id"])
        corpus_id = _valid_id(row["corpus-id"])
        if row["score"] != 1 or query_id in gold:
            raise ValueError("qrels must contain one score-1 relation per query")
        if query_id not in queries or corpus_id not in corpus:
            raise ValueError("qrel references an unknown id")
        gold[query_id] = corpus_id
    if set(gold) != set(queries):
        raise ValueError("qrels do not cover every query exactly once")
    if enforce_frozen_counts and len(set(gold.values())) != 372:
        raise ValueError("unique gold answer cardinality drifted")

    corpus_ids = sorted(corpus, key=lambda value: value.encode("utf-8"))
    plan: list[PlannedRow] = []
    for index, query_id in enumerate(sorted(queries, key=lambda value: value.encode("utf-8"))):
        gold_id = gold[query_id]
        gold_snippet = corpus[gold_id]
        if not gold_snippet:
            raise ValueError("gold snippet is empty")
        used = {gold_snippet}
        candidates = sorted(
            (candidate for candidate in corpus_ids if candidate != gold_id),
            key=lambda candidate: framed_hash(SEED, query_id, candidate),
        )
        distractors: list[str] = []
        for candidate in candidates:
            snippet = corpus[candidate]
            if snippet and snippet not in used:
                distractors.append(candidate)
                used.add(snippet)
            if len(distractors) == 3:
                break
        if len(distractors) != 3:
            raise ValueError("not enough distinct distractor snippets")
        position = index % 4
        ordered = distractors.copy()
        ordered.insert(position, gold_id)
        plan.append(
            PlannedRow(
                query_id=query_id,
                query_text=queries[query_id],
                ordered_choice_ids=cast(tuple[str, str, str, str], tuple(ordered)),
                ordered_choice_snippets=cast(
                    tuple[str, str, str, str], tuple(corpus[item] for item in ordered)
                ),
                gold_position=position,
            )
        )
    if enforce_frozen_counts:
        plan_json = [row.digest_record() for row in plan]
        digest = hashlib.sha256(canonical_json_bytes(cast(JsonValue, plan_json))).hexdigest()
        if digest != PLAN_SHA256:
            raise ValueError("benchmark plan digest drifted")
        counts = tuple(sum(row.gold_position == position for row in plan) for position in range(4))
        if counts != (94, 93, 93, 93):
            raise ValueError("gold position balance drifted")
        if lexical_baseline_correct(tuple(plan)) != 265:
            raise ValueError("lexical baseline drifted")
    return tuple(plan)


def lexical_baseline_correct(plan: tuple[PlannedRow, ...]) -> int:
    correct = 0
    for row in plan:
        query_tokens = lexical_tokens(row.query_text)
        scores = [
            len(query_tokens & lexical_tokens(choice)) for choice in row.ordered_choice_snippets
        ]
        correct += earliest_argmax([float(score) for score in scores]) == row.gold_position
    return correct


def build_request(row: PlannedRow, *, model: str, backend: str) -> DecisionRequest:
    revisions = {
        "julia": "phase5c-julia.v1",
        "saracura-universal": "phase4e-saracura-ranker.v1",
    }
    try:
        revision = revisions[backend]
    except KeyError:
        raise ValueError("unsupported benchmark backend") from None
    return DecisionRequest(
        api_version="v1alpha1",
        model=model,
        mode="research",
        locale="pt-BR",
        domain="finance",
        workflow=WorkflowReference(id="universal-choice", revision=revision),
        state={"pergunta": row.query_text},
        questions=(
            ChoiceQuestion(
                id="resposta",
                type="choice",
                instruction=INSTRUCTION,
                criteria=tuple(
                    ChoiceCriterion(id=option_id, description=description)
                    for option_id, description in zip(
                        OPTION_IDS, row.ordered_choice_snippets, strict=True
                    )
                ),
            ),
        ),
    )


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]
