"""Closed synthetic-task contract for the planned Phase 4E ranker."""

from __future__ import annotations

import unicodedata
from typing import Annotated, Literal

from pydantic import Field, JsonValue, model_validator

from saracura.contracts.models import ChoiceCriterion, ClosedModel, Identifier
from saracura.serialization import canonical_json_bytes

SARACURA_UNIVERSAL_WORKFLOW_ID = "universal-choice"
SARACURA_UNIVERSAL_WORKFLOW_REVISION = "phase4e-saracura-ranker.v1"
UNIVERSAL_DOMAINS = frozenset(
    {
        "email_triage",
        "customer_support",
        "finance",
        "accounting",
        "commerce",
        "operations",
        "scheduling",
        "document_routing",
        "browser_action",
        "security_triage",
        "content_moderation",
        "personal_productivity",
    }
)


def _require_nfc(value: JsonValue) -> None:
    if isinstance(value, str):
        if unicodedata.normalize("NFC", value) != value:
            raise ValueError("strings and keys must already be NFC")
        return
    if isinstance(value, dict):
        for key, child in value.items():
            if unicodedata.normalize("NFC", key) != key:
                raise ValueError("strings and keys must already be NFC")
            _require_nfc(child)
        return
    if isinstance(value, list):
        for child in value:
            _require_nfc(child)


class UniversalTask(ClosedModel):
    """One ordered Choice ranking task; training rows add provenance in Phase 4E.2."""

    task_id: Identifier
    family_id: Identifier
    locale: Literal["pt-BR", "en"]
    domain: Identifier
    instruction: Annotated[str, Field(min_length=1, max_length=120)]
    state: Annotated[dict[str, JsonValue], Field(min_length=1, max_length=64)]
    criteria: Annotated[tuple[ChoiceCriterion, ...], Field(min_length=2, max_length=8)]
    selected_criterion_id: Identifier

    @model_validator(mode="after")
    def phase4e_capacity_and_semantics(self) -> UniversalTask:
        if self.domain not in UNIVERSAL_DOMAINS:
            raise ValueError("domain is outside the closed Phase 4E taxonomy")
        _require_nfc(self.instruction)
        _require_nfc(self.state)
        for criterion in self.criteria:
            _require_nfc(criterion.id)
            _require_nfc(criterion.description)
            if len(criterion.description) > 120 or len(criterion.description.encode("utf-8")) > 480:
                raise ValueError("criterion description exceeds Phase 4E capacity")
        if len(self.instruction.encode("utf-8")) > 480:
            raise ValueError("instruction exceeds Phase 4E capacity")
        state_json = canonical_json_bytes(self.state)
        if len(state_json.decode("utf-8")) > 200 or len(state_json) > 800:
            raise ValueError("state exceeds Phase 4E capacity")
        if self.selected_criterion_id not in {criterion.id for criterion in self.criteria}:
            raise ValueError("selected criterion must be declared")
        return self
