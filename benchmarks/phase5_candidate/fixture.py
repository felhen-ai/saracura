"""Phase 5B public bilingual fixture (self-authored, non-sensitive).

The fixture is integration and systems evidence only.  Phase 5B never computes
language-quality accuracy from these examples.  Every request uses the same
typed Saracura Choice schema and yields Q=1, Q=10 and Q=50 workloads in both
PT-BR and English.
"""

from __future__ import annotations

from typing import Literal

from saracura.contracts.models import (
    ChoiceCriterion,
    ChoiceQuestion,
    DecisionRequest,
    WorkflowReference,
)

FIXTURE_WORKFLOW = WorkflowReference(id="universal-choice", revision="phase4e-saracura-ranker.v1")

_CRITERIA = (
    ChoiceCriterion(id="latency", description="Responsiveness of the candidate under load"),
    ChoiceCriterion(id="clarity", description="How unambiguous the candidate decision reads"),
    ChoiceCriterion(id="coverage", description="Whether the candidate covers the requested scope"),
    ChoiceCriterion(id="cost", description="Relative cost to evaluate this candidate"),
    ChoiceCriterion(id="safety", description="Stays conservative near thresholds"),
)


def _questions(n: Literal[1, 10, 50], locale: Literal["pt-BR", "en"]) -> tuple[ChoiceQuestion, ...]:
    if locale == "pt-BR":
        instruction = "Escolha a opção mais adequada entre os critérios fornecidos."
    else:
        instruction = "Choose the most suitable option among the provided criteria."
    questions: list[ChoiceQuestion] = []
    for index in range(n):
        question_id = f"q{index:02d}-{locale.lower()}"
        criteria = _CRITERIA if n == 1 or index % 5 == 0 else _CRITERIA[:3]
        questions.append(
            ChoiceQuestion(
                id=question_id,
                type="choice",
                instruction=instruction,
                criteria=tuple(criteria),
            )
        )
    return tuple(questions)


def request_for(
    locale: Literal["pt-BR", "en"], question_count: Literal[1, 10, 50]
) -> DecisionRequest:
    return DecisionRequest(
        api_version="v1alpha1",
        model="phase4e-saracura-ranker.v1",
        mode="research",
        locale=locale,
        domain="phase5b-integration",
        workflow=FIXTURE_WORKFLOW,
        state={"fixture": f"{locale}-q{question_count}"},
        questions=_questions(question_count, locale),
    )


def matrix() -> dict[str, dict[str, DecisionRequest]]:
    """Return the full public fixture matrix keyed by locale then workload."""
    return {
        "pt-BR": {
            "q1": request_for("pt-BR", 1),
            "q10": request_for("pt-BR", 10),
            "q50": request_for("pt-BR", 50),
        },
        "en": {
            "q1": request_for("en", 1),
            "q10": request_for("en", 10),
            "q50": request_for("en", 50),
        },
    }
