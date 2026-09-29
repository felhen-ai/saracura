"""Known, immutable workflow schemas for the first research cycle."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from saracura.backends.base import BackendCapabilities
from saracura.contracts.errors import ErrorCode, SaracuraError
from saracura.contracts.models import ChoiceCriterion, ChoiceQuestion, DecisionRequest
from saracura.serialization import ordered_question_bytes, serialize_question

MINILM_ROUTING_WORKFLOW_ID = "support-routing"
MINILM_ROUTING_WORKFLOW_REVISION = "phase4a-local-minilm-routing.v1"
MINILM_ROUTING_LABELS = (
    "billing",
    "technical_support",
    "account_access",
    "subscription_cancellation",
    "order_delivery",
)
UNIVERSAL_CHOICE_WORKFLOW_ID = "universal-choice"
UNIVERSAL_CHOICE_WORKFLOW_REVISION = "phase4d-laya.v1"
SARACURA_UNIVERSAL_CHOICE_WORKFLOW_REVISION = "phase4e-saracura-ranker.v1"
SYSTEMONE_CHOICE_WORKFLOW_REVISION = "phase5c-systemone.v1"
JULIA_CHOICE_WORKFLOW_REVISION = "phase5c-julia.v1"
JULIA_POSITION_ENSEMBLE_WORKFLOW_REVISION = "phase5d1-julia-cyclic-mean.v1"
_UNIVERSAL_CHOICE_LOCALES = frozenset({"pt-BR", "en"})
_UNIVERSAL_CHOICE_MAX_QUESTIONS = 10
_UNIVERSAL_CHOICE_MAX_CRITERIA = 20


@dataclass(frozen=True, slots=True)
class WorkflowSchema:
    id: str
    revision: str
    locale: str
    domain: str
    questions: dict[str, ChoiceQuestion]
    criteria_order_semantic: bool = False


class WorkflowRegistry:
    def __init__(self, schemas: tuple[WorkflowSchema, ...]) -> None:
        self._schemas = {(schema.id, schema.revision): schema for schema in schemas}

    def validate(
        self,
        request: DecisionRequest,
        execution_tier: Literal["compiled", "universal"] = "compiled",
        capabilities: BackendCapabilities | None = None,
    ) -> WorkflowSchema:
        key = (request.workflow.id, request.workflow.revision)
        schema = self._schemas.get(key)
        if schema is None:
            if key in (
                (UNIVERSAL_CHOICE_WORKFLOW_ID, UNIVERSAL_CHOICE_WORKFLOW_REVISION),
                (UNIVERSAL_CHOICE_WORKFLOW_ID, SARACURA_UNIVERSAL_CHOICE_WORKFLOW_REVISION),
                (UNIVERSAL_CHOICE_WORKFLOW_ID, SYSTEMONE_CHOICE_WORKFLOW_REVISION),
                (UNIVERSAL_CHOICE_WORKFLOW_ID, JULIA_CHOICE_WORKFLOW_REVISION),
                (UNIVERSAL_CHOICE_WORKFLOW_ID, JULIA_POSITION_ENSEMBLE_WORKFLOW_REVISION),
            ):
                return self._validate_universal_choice(request, execution_tier, capabilities)
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
            same_question = expected is not None and (
                ordered_question_bytes(question) == ordered_question_bytes(expected)
                if schema.criteria_order_semantic
                else serialize_question(question) == serialize_question(expected)
            )
            if not same_question:
                raise SaracuraError(
                    ErrorCode.SCHEMA_UNSUPPORTED,
                    "Question does not match the known workflow schema.",
                    f"/questions/{index}",
                    details={"question_id": question.id},
                )
        return schema

    @staticmethod
    def _validate_universal_choice(
        request: DecisionRequest,
        execution_tier: Literal["compiled", "universal"],
        capabilities: BackendCapabilities | None,
    ) -> WorkflowSchema:
        if execution_tier != "universal":
            raise SaracuraError(
                ErrorCode.WORKFLOW_UNSUPPORTED,
                "The dynamic workflow requires a universal backend.",
                "/workflow",
            )
        if (
            capabilities is None
            or (
                request.workflow.id,
                request.workflow.revision,
            )
            not in capabilities.dynamic_workflows
        ):
            raise SaracuraError(
                ErrorCode.WORKFLOW_UNSUPPORTED,
                "The selected backend does not support this dynamic workflow revision.",
                "/workflow",
            )
        if request.locale not in _UNIVERSAL_CHOICE_LOCALES:
            raise SaracuraError(
                ErrorCode.LOCALE_UNVERIFIED,
                "The dynamic workflow is not verified for the requested locale.",
                "/locale",
            )
        # The workflow contract remains capped at ten questions even when a
        # generic universal test backend advertises a higher internal limit.
        # A registered backend may impose a lower criterion capacity (the
        # Phase 4E Saracura checkpoint permits eight rather than Laya's 20).
        maximum_questions = 10 if capabilities is None else min(10, capabilities.max_questions)
        maximum_criteria = 20 if capabilities is None else min(20, capabilities.max_criteria)
        if len(request.questions) > maximum_questions:
            raise SaracuraError(
                ErrorCode.CARDINALITY_EXCEEDED,
                "Dynamic universal workflows support at most ten questions.",
                "/questions",
            )
        if (
            request.workflow.revision == JULIA_POSITION_ENSEMBLE_WORKFLOW_REVISION
            and sum(len(question.criteria) for question in request.questions) > 20
        ):
            raise SaracuraError(
                ErrorCode.CARDINALITY_EXCEEDED,
                "Julia position ensemble supports at most 20 criteria per request.",
                "/questions",
            )
        if any(len(question.criteria) > maximum_criteria for question in request.questions):
            raise SaracuraError(
                ErrorCode.CARDINALITY_EXCEEDED,
                "Dynamic universal workflow criteria exceed the backend limit.",
                "/questions/*/criteria",
            )
        return WorkflowSchema(
            id=UNIVERSAL_CHOICE_WORKFLOW_ID,
            revision=request.workflow.revision,
            locale=request.locale,
            domain=request.domain,
            questions={},
            criteria_order_semantic=True,
        )


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


MINILM_ROUTING_QUESTION = ChoiceQuestion(
    id="department",
    type="choice",
    instruction=(
        "Classifique a mensagem de suporte em exatamente uma categoria. "
        "Aplique as definições e a prioridade descritas nos critérios."
    ),
    criteria=(
        ChoiceCriterion(
            id="billing",
            description=(
                "cobrança, pagamento, estorno, fatura ou método de pagamento, "
                "salvo cancelamento puro"
            ),
        ),
        ChoiceCriterion(
            id="technical_support",
            description=(
                "falha, configuração, compatibilidade ou ajuda de uso não bloqueada por acesso"
            ),
        ),
        ChoiceCriterion(
            id="account_access",
            description=(
                "autenticação, identidade, credencial ou acesso à conta; "
                "tem prioridade quando bloqueia outra ação"
            ),
        ),
        ChoiceCriterion(
            id="subscription_cancellation",
            description=(
                "parar assinatura, renovação ou plano recorrente sem cobrança ou estorno separado"
            ),
        ),
        ChoiceCriterion(
            id="order_delivery",
            description=("envio, rastreio, entrega, pacote ausente/danificado ou pedido físico"),
        ),
    ),
)


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
                id=MINILM_ROUTING_WORKFLOW_ID,
                revision=MINILM_ROUTING_WORKFLOW_REVISION,
                locale="pt-BR",
                domain="support",
                questions={MINILM_ROUTING_QUESTION.id: MINILM_ROUTING_QUESTION},
                criteria_order_semantic=True,
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
