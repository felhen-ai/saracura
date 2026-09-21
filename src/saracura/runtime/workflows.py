"""Known, immutable workflow schemas for the first research cycle."""

from __future__ import annotations

from dataclasses import dataclass

from saracura.contracts.errors import ErrorCode, SaracuraError
from saracura.contracts.models import ChoiceCriterion, ChoiceQuestion, DecisionRequest
from saracura.serialization import serialize_question


@dataclass(frozen=True, slots=True)
class WorkflowSchema:
    id: str
    revision: str
    locale: str
    domain: str
    questions: dict[str, ChoiceQuestion]


class WorkflowRegistry:
    def __init__(self, schemas: tuple[WorkflowSchema, ...]) -> None:
        self._schemas = {(schema.id, schema.revision): schema for schema in schemas}

    def validate(self, request: DecisionRequest) -> WorkflowSchema:
        key = (request.workflow.id, request.workflow.revision)
        schema = self._schemas.get(key)
        if schema is None:
            raise SaracuraError(
                ErrorCode.WORKFLOW_UNSUPPORTED,
                "Workflow id or immutable revision is not supported.",
                "/workflow",
            )
        if request.locale != schema.locale:
            raise SaracuraError(
                ErrorCode.LOCALE_UNVERIFIED,
                "Workflow is not verified for the requested locale.",
                "/locale",
            )
        if request.domain != schema.domain:
            raise SaracuraError(
                ErrorCode.DOMAIN_UNVERIFIED,
                "Workflow is not verified for the requested domain.",
                "/domain",
            )

        for index, question in enumerate(request.questions):
            expected = schema.questions.get(question.id)
            if expected is None or serialize_question(question) != serialize_question(expected):
                raise SaracuraError(
                    ErrorCode.SCHEMA_UNSUPPORTED,
                    "Question does not match the known workflow schema.",
                    f"/questions/{index}",
                    details={"question_id": question.id},
                )
        return schema


def _scaling_question(index: int) -> ChoiceQuestion:
    return ChoiceQuestion(
        id=f"decision-{index:02d}",
        type="choice",
        instruction=f"Qual rota conhecida deve tratar a decisão {index:02d}?",
        criteria=(
            ChoiceCriterion(id="route-a", description="Rota determinística A"),
            ChoiceCriterion(id="route-b", description="Rota determinística B"),
        ),
    )


def known_scaling_questions(count: int = 50) -> tuple[ChoiceQuestion, ...]:
    """Return an immutable prefix of the first-party decision-scaling workflow."""

    if not 1 <= count <= 50:
        raise ValueError("count must be between 1 and 50")
    return tuple(_scaling_question(index) for index in range(1, count + 1))


def default_workflows() -> WorkflowRegistry:
    support = ChoiceQuestion(
        id="department",
        type="choice",
        instruction="Qual área deve tratar o caso?",
        criteria=(
            ChoiceCriterion(id="billing", description="Cobranças, pagamentos e estornos"),
            ChoiceCriterion(id="support", description="Falhas técnicas e uso do produto"),
        ),
    )
    scaling = known_scaling_questions()
    return WorkflowRegistry(
        (
            WorkflowSchema(
                id="support-routing",
                revision="2026-09-21",
                locale="pt-BR",
                domain="support",
                questions={support.id: support},
            ),
            WorkflowSchema(
                id="decision-scaling",
                revision="2026-09-21",
                locale="pt-BR",
                domain="benchmark",
                questions={question.id: question for question in scaling},
            ),
        )
    )
