# ruff: noqa: E501
"""Offline-first, fail-closed corpus contracts for Phase 4E.2a.

This is intentionally a checkout-only research tool.  It never obtains a
credential and the injected transport is unreachable unless the caller has
already made an explicit, higher-phase network decision.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from ctypes import CDLL, c_char_p, c_int, get_errno, set_errno
from dataclasses import dataclass, field
from decimal import Decimal
from difflib import SequenceMatcher
from errno import EEXIST
from pathlib import Path
from typing import Any, Literal, Protocol, cast

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError, model_validator

from benchmarks.encoder_loader import VerifiedSnapshot, load_encoder
from benchmarks.encoder_registry import get_candidate
from benchmarks.io import atomic_create
from benchmarks.saracura_universal_policy import AUTHOR_MODEL as AUTHOR_MODEL
from benchmarks.saracura_universal_policy import (
    AUTHOR_PROVIDER,
    REVIEWER_PROVIDER,
    validate_phase4e_policy,
)
from benchmarks.saracura_universal_policy import REVIEWER_MODEL as REVIEWER_MODEL
from benchmarks.saracura_universal_policy import STAGE_LIMITS as STAGE_LIMITS
from benchmarks.saracura_universal_policy import TOTAL_BUDGET as TOTAL_BUDGET
from saracura.contracts.models import ChoiceCriterion
from saracura.serialization import canonical_json_bytes
from saracura.universal.rendering import CapacityError, render_task, validate_rendered_capacity
from saracura.universal.tasks import UniversalTask

SPLITS = ("synthetic_train", "synthetic_dev", "synthetic_holdout")
LOCALES = ("pt-BR", "en")
OPTION_COUNTS = tuple(range(2, 9))
DOMAINS = (
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
)
AXES: dict[str, tuple[str, str]] = {
    "explicitness": ("explicit", "implicit"),
    "negation": ("absent", "present"),
    "distractor_overlap": ("low", "high"),
    "urgency": ("normal", "urgent"),
}
SCENARIO_CODES = (
    "action_required",
    "informational_only",
    "suspected_abuse",
    "missing_information",
    "deadline_risk",
    "policy_violation",
    "duplicate_record",
    "topic_routing",
    "rule_eligibility",
    "urgency_priority",
    "threshold_approval",
    "reconciliation_mismatch",
    "fulfillment_exception",
    "access_risk",
    "schedule_conflict",
    "content_safety",
)
CRITERION_ROLES = (
    "matches_rule",
    "contradicts_rule",
    "irrelevant_to_rule",
    "insufficient_evidence",
    "unsafe_action",
    "premature_action",
    "overbroad_action",
    "duplicate_action",
)
SCENARIO_CODEBOOK = (
    "Scenarios: action_required=required next action; "
    "informational_only=information only; suspected_abuse=suspected abuse; "
    "missing_information=missing facts; deadline_risk=deadline risk; "
    "policy_violation=policy breach; duplicate_record=duplicate record; "
    "topic_routing=topic routing; rule_eligibility=rule eligibility; "
    "urgency_priority=urgency priority; threshold_approval=approval threshold; "
    "reconciliation_mismatch=reconciliation mismatch; "
    "fulfillment_exception=fulfillment exception; access_risk=access risk; "
    "schedule_conflict=schedule conflict; content_safety=content safety."
)
CRITERION_ROLE_CODEBOOK = (
    "Roles: matches_rule=match; contradicts_rule=conflict; "
    "irrelevant_to_rule=irrelevant; insufficient_evidence=missing facts; "
    "unsafe_action=unsafe; premature_action=early; overbroad_action=too broad; "
    "duplicate_action=duplicate."
)
SPLIT_SIZES = {"synthetic_train": 1120, "synthetic_dev": 240, "synthetic_holdout": 240}
PAIR_COUNTS = {"synthetic_train": 210, "synthetic_dev": 45, "synthetic_holdout": 45}
PRICES = {
    "corpus_author": (Decimal("0.30"), Decimal("2.50")),
    "corpus_reviewer": (Decimal("0.71"), Decimal("0.71")),
}
RESPONSE_SCHEMA_NAME = "saracura_phase4e_response"
_SENSITIVE = re.compile(r"(?:\b\d{3}[.]?\d{3}[.]?\d{3}-?\d{2}\b|\b\d{13,16}\b|@|https?://)", re.I)


def provider_preferences(
    stage: Literal["corpus_author", "corpus_reviewer"],
) -> dict[str, Any]:
    """Pin a provider proven to honor the stage's exact request contract."""

    provider = AUTHOR_PROVIDER if stage == "corpus_author" else REVIEWER_PROVIDER
    return {
        "order": [provider],
        "allow_fallbacks": False,
        "require_parameters": True,
        "data_collection": "deny",
        "zdr": True,
    }


class CorpusError(ValueError):
    """A closed corpus invariant was violated."""


class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...


class Transport(Protocol):
    def __call__(self, method: str, url: str, headers: Mapping[str, str], body: bytes) -> Any: ...


@dataclass(frozen=True)
class VerifiedMiniLMTokenizerReceipt:
    """Exact tokenizer capability bound to the reviewed MiniLM snapshot.

    Construction verifies descriptor bytes before loading with ``local_files_only``;
    it is therefore the only production capacity path and cannot acquire a model.
    """

    _tokenizer: Any = field(repr=False, compare=False)
    candidate_id: str
    revision: str

    @classmethod
    def create(cls, snapshot: Path) -> VerifiedMiniLMTokenizerReceipt:
        candidate = get_candidate("multilingual-minilm-l12")
        verified = VerifiedSnapshot.create(candidate, snapshot)
        loaded = load_encoder(candidate, snapshot, verified=verified)
        if (
            loaded.candidate.id != "multilingual-minilm-l12"
            or loaded.candidate.revision != "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
        ):
            raise CorpusError("verified tokenizer is not reviewed MiniLM")
        return cls(loaded.tokenizer, loaded.candidate.id, loaded.candidate.revision)

    def count(self, text: str) -> int:
        encoded = self._tokenizer(text, truncation=False, add_special_tokens=True)
        ids = encoded.get("input_ids") if isinstance(encoded, Mapping) else None
        if not isinstance(ids, list) or not all(isinstance(item, int) for item in ids):
            raise CorpusError("verified tokenizer output")
        return len(ids)


def _canonical(value: Any) -> bytes:
    return canonical_json_bytes(value)


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _seeded(seed: str, *parts: object) -> int:
    return int(_sha("\0".join((seed, *(str(part) for part in parts))).encode())[:16], 16)


def _cell_counts(split: str) -> dict[tuple[str, int], int]:
    """Allocate each fixed split before any provider result is visible."""
    total = SPLIT_SIZES[split]
    if split == "synthetic_train":
        pt = [96] * 7
        en = [64] * 7
    elif split == "synthetic_dev":
        pt = [21, 21, 21, 21, 20, 20, 20]
        en = [14, 14, 14, 14, 14, 13, 13]
    else:
        pt = [21, 21, 21, 21, 20, 20, 20]
        en = [14, 14, 14, 14, 14, 13, 13]
    result = {("pt-BR", count): pt[index] for index, count in enumerate(OPTION_COUNTS)}
    result.update({("en", count): en[index] for index, count in enumerate(OPTION_COUNTS)})
    if sum(result.values()) != total:
        raise AssertionError("fixed split allocation")
    return result


def _semantic_target(seed: str, task_id: str, option_count: int) -> dict[str, Any]:
    """Derive the closed selected-first target before any provider call."""

    if not 2 <= option_count <= len(CRITERION_ROLES):
        raise CorpusError("semantic target option count")
    distractors = sorted(
        CRITERION_ROLES[1:], key=lambda role: _seeded(seed, "role", task_id, role)
    )[: option_count - 1]
    return {
        "scenario": SCENARIO_CODES[_seeded(seed, "scenario", task_id) % len(SCENARIO_CODES)],
        "criterion_roles": ["matches_rule", *distractors],
    }


def build_plan() -> dict[str, Any]:
    """Return the immutable, text-free 1,600-slot public-seed plan."""
    policy = validate_phase4e_policy()
    seed = cast(str, policy["planning"]["seed"])
    slots: list[dict[str, Any]] = []
    serial = 0
    for split in SPLITS:
        for locale, option_count in (
            (locale_value, count) for locale_value in LOCALES for count in OPTION_COUNTS
        ):
            for within_cell in range(_cell_counts(split)[(locale, option_count)]):
                axes = {
                    key: values[_seeded(seed, split, locale, option_count, within_cell, key) % 2]
                    for key, values in AXES.items()
                }
                task_id = "task-" + _sha(f"{seed}\0task\0{serial}".encode())
                slots.append(
                    {
                        "slot": serial,
                        "task_id": task_id,
                        "family_id": "family-" + _sha(f"{seed}\0family\0{serial}".encode()),
                        "pair_id": None,
                        "split": split,
                        "locale": locale,
                        "domain": DOMAINS[
                            _seeded(seed, split, locale, option_count, within_cell) % len(DOMAINS)
                        ],
                        "axes": axes,
                        "option_count": option_count,
                        "gold_position": within_cell % option_count,
                        "semantic_target": _semantic_target(seed, task_id, option_count),
                    }
                )
                serial += 1
    # Pair slots only after cell/split allocation.  Thus a partial pair cannot leak across a split.
    for split in SPLITS:
        needed = PAIR_COUNTS[split]
        by_cell: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
        for slot in slots:
            if slot["split"] == split:
                by_cell[(cast(str, slot["locale"]), cast(int, slot["option_count"]))].append(slot)
        pair_index = 0
        while pair_index < needed:
            option_count = OPTION_COUNTS[pair_index % len(OPTION_COUNTS)]
            pt = by_cell[("pt-BR", option_count)].pop(0)
            en = by_cell[("en", option_count)].pop(0)
            pair_id = "family-" + _sha(f"{seed}\0pair\0{split}\0{pair_index}".encode())
            # Cross-locale members share every semantic axis; locale is deliberately separate.
            en["family_id"] = pt["family_id"] = pair_id
            en["pair_id"] = pt["pair_id"] = pair_id
            en["domain"] = pt["domain"]
            en["axes"] = pt["axes"]
            en["gold_position"] = pt["gold_position"]
            en["semantic_target"] = {
                "scenario": pt["semantic_target"]["scenario"],
                "criterion_roles": list(pt["semantic_target"]["criterion_roles"]),
            }
            pair_index += 1
    return {
        "schema_version": "phase4e-universal-plan.v2",
        "workflow_revision": "phase4e-saracura-universal-synthetic.v1",
        "seed": seed,
        "slots": slots,
    }


def validate_plan(value: Mapping[str, Any]) -> None:
    """Fail closed on any mutation; no provider result can alter this plan."""
    if _canonical(value) != _canonical(build_plan()):
        raise CorpusError("immutable plan mismatch")
    slots = value.get("slots")
    if not isinstance(slots, list) or len(slots) != 1600:
        raise CorpusError("plan slot count")
    counts = Counter(cast(str, row["split"]) for row in slots)
    if counts != SPLIT_SIZES:
        raise CorpusError("plan split allocation")
    pairs: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    families: dict[str, set[str]] = defaultdict(set)
    for row in slots:
        family = row.get("family_id")
        split = row.get("split")
        if not isinstance(family, str) or not isinstance(split, str):
            raise CorpusError("plan family identity")
        families[family].add(split)
        pair = row.get("pair_id")
        if pair is not None:
            pairs[cast(str, pair)].append(row)
        _planned_semantic_target(row)
    if any(len(splits) != 1 for splits in families.values()):
        raise CorpusError("family split isolation")
    if len(pairs) != 300 or any(len(rows) != 2 for rows in pairs.values()):
        raise CorpusError("cross-locale plan")
    for pair_id, rows in pairs.items():
        if (
            {row["locale"] for row in rows} != set(LOCALES)
            or len({row["split"] for row in rows}) != 1
            or {row["family_id"] for row in rows} != {pair_id}
            or len({_canonical(row["semantic_target"]) for row in rows}) != 1
        ):
            raise CorpusError("cross-locale split or semantic isolation")
    for split in SPLITS:
        for locale in LOCALES:
            for count in OPTION_COUNTS:
                positions = [
                    cast(int, row["gold_position"])
                    for row in slots
                    if row["split"] == split
                    and row["locale"] == locale
                    and row["option_count"] == count
                ]
                if (
                    not positions
                    or max(Counter(positions).values()) - min(Counter(positions).values()) > 1
                ):
                    raise CorpusError("gold positions are not balanced")


class _Closed(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


ScenarioCode = Literal[
    "action_required",
    "informational_only",
    "suspected_abuse",
    "missing_information",
    "deadline_risk",
    "policy_violation",
    "duplicate_record",
    "topic_routing",
    "rule_eligibility",
    "urgency_priority",
    "threshold_approval",
    "reconciliation_mismatch",
    "fulfillment_exception",
    "access_risk",
    "schedule_conflict",
    "content_safety",
]
CriterionRole = Literal[
    "matches_rule",
    "contradicts_rule",
    "irrelevant_to_rule",
    "insufficient_evidence",
    "unsafe_action",
    "premature_action",
    "overbroad_action",
    "duplicate_action",
]


def _planned_semantic_target(planned: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact closed, selected-first semantic target for one slot."""

    target = planned.get("semantic_target")
    option_count = planned.get("option_count")
    if (
        not isinstance(target, Mapping)
        or set(target) != {"scenario", "criterion_roles"}
        or not isinstance(option_count, int)
    ):
        raise CorpusError("planned semantic target")
    try:
        parsed = SemanticEquivalenceAttestation.model_validate(
            {**target, "selected_role": "matches_rule"}
        )
    except ValidationError as error:
        raise CorpusError("planned semantic target") from error
    if len(parsed.criterion_roles) != option_count or parsed.criterion_roles[0] != "matches_rule":
        raise CorpusError("planned semantic target")
    return {
        "scenario": parsed.scenario,
        "criterion_roles": list(parsed.criterion_roles),
        "selected_role": parsed.selected_role,
    }


def _semantic_labels_absent(record: Mapping[str, Any], target: Mapping[str, Any]) -> bool:
    """Keep closed labels out of task text, where they would leak the answer."""

    labels = [cast(str, target["scenario"]), *cast(list[str], target["criterion_roles"])]

    def text_values(value: object) -> Iterable[str]:
        if isinstance(value, str):
            yield value
        elif isinstance(value, Mapping):
            for child in value.values():
                yield from text_values(child)
        elif isinstance(value, list):
            for child in value:
                yield from text_values(child)

    texts = [record.get("instruction"), *text_values(record.get("state"))]
    criteria = record.get("criteria")
    if isinstance(criteria, list):
        texts.extend(
            criterion.get("description") for criterion in criteria if isinstance(criterion, Mapping)
        )
    return all(
        isinstance(text, str) and all(label.casefold() not in text.casefold() for label in labels)
        for text in texts
    )


def _reordered_target(planned: Mapping[str, Any]) -> dict[str, Any]:
    """Return the target after the local, planner-owned gold-position move."""

    target = _planned_semantic_target(planned)
    gold_position = planned.get("gold_position")
    roles = list(cast(list[str], target["criterion_roles"]))
    if not isinstance(gold_position, int) or not 0 <= gold_position < len(roles):
        raise CorpusError("planned gold position")
    selected_role = roles.pop(0)
    roles.insert(gold_position, selected_role)
    return {
        "scenario": target["scenario"],
        "criterion_roles": roles,
        "selected_role": target["selected_role"],
    }


class AuthorRecord(_Closed):
    task_id: str = Field(pattern=r"^task-[0-9a-f]{64}$")
    family_id: str = Field(pattern=r"^family-[0-9a-f]{64}$")
    locale: Literal["pt-BR", "en"]
    domain: str
    axes: dict[str, str]
    option_count: int = Field(ge=2, le=8)
    gold_position: int = Field(ge=0, le=7)
    instruction: str = Field(min_length=1, max_length=120)
    state: dict[str, Any] = Field(min_length=1, max_length=64)
    criteria: list[ChoiceCriterion] = Field(min_length=2, max_length=8)
    selected_criterion_id: str
    semantic_equivalence_attestation: SemanticEquivalenceAttestation
    cross_locale_attestation: CrossLocaleAttestation | None = None


class SemanticEquivalenceAttestation(_Closed):
    """Language-neutral semantics independently stated by an actor."""

    scenario: ScenarioCode
    criterion_roles: list[CriterionRole] = Field(min_length=2, max_length=8)
    selected_role: CriterionRole

    @model_validator(mode="after")
    def distinct_roles_and_selected_role(self) -> SemanticEquivalenceAttestation:
        if len(set(self.criterion_roles)) != len(self.criterion_roles):
            raise ValueError("cross-locale criterion roles must be distinct")
        if self.selected_role != "matches_rule" or self.selected_role not in self.criterion_roles:
            raise ValueError("cross-locale selected role must match the rule")
        return self


class CrossLocaleAttestation(SemanticEquivalenceAttestation):
    """Author's semantic contract bound to one planned locale pair."""

    pair_id: str = Field(pattern=r"^family-[0-9a-f]{64}$")


class AuthorGeneratedState(_Closed):
    summary: str = Field(min_length=1, max_length=180)


class AuthorGeneratedCriterion(_Closed):
    description: str = Field(min_length=1, max_length=120)


class AuthorGeneratedRecord(_Closed):
    """Only fields the remote author is allowed to generate."""

    instruction: str = Field(min_length=1, max_length=120)
    state: AuthorGeneratedState
    criteria: list[AuthorGeneratedCriterion] = Field(min_length=2, max_length=8)
    selected_index: int = Field(ge=0, le=7)
    semantic_equivalence_attestation: SemanticEquivalenceAttestation


class ReviewerGeneratedRecord(_Closed):
    status: Literal["accepted", "rejected"]
    selected_criterion_id: str | None
    reason_codes: list[str] = Field(max_length=8)
    natural_language: bool
    fictional: bool
    exclusive_options: bool
    private_or_sensitive: bool
    semantic_equivalence_attestation: SemanticEquivalenceAttestation


class ReviewerRecord(_Closed):
    task_id: str = Field(pattern=r"^task-[0-9a-f]{64}$")
    status: Literal["accepted", "rejected"]
    selected_criterion_id: str | None
    reason_codes: list[str] = Field(max_length=8)
    natural_language: bool
    fictional: bool
    exclusive_options: bool
    private_or_sensitive: bool
    semantic_equivalence_attestation: SemanticEquivalenceAttestation


class ValidatedAuthorRow(_Closed):
    """One locally validated author result eligible for a blind review request.

    This keeps planner and answer fields available to the local acceptance
    pipeline while making the reviewer transport derive its public view from a
    closed, typed row rather than from caller-provided chat messages.
    """

    task_id: str = Field(pattern=r"^task-[0-9a-f]{64}$")
    family_id: str = Field(pattern=r"^family-[0-9a-f]{64}$")
    locale: Literal["pt-BR", "en"]
    domain: str
    instruction: str = Field(min_length=1, max_length=120)
    state: dict[str, JsonValue] = Field(min_length=1, max_length=64)
    criteria: list[ChoiceCriterion] = Field(min_length=2, max_length=8)
    selected_criterion_id: str
    split: Literal["synthetic_train", "synthetic_dev", "synthetic_holdout"]
    pair_id: str | None
    axes: dict[str, str]
    semantic_equivalence_attestation: SemanticEquivalenceAttestation
    cross_locale_attestation: CrossLocaleAttestation | None
    author_response_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    author_reservation_id: str | None = Field(default=None, pattern=r"^reservation-[0-9a-f]{64}$")
    author_request_id: str | None = Field(default=None, min_length=1, max_length=256)

    @model_validator(mode="after")
    def valid_author_task(self) -> ValidatedAuthorRow:
        try:
            UniversalTask.model_validate(
                {
                    "task_id": self.task_id,
                    "family_id": self.family_id,
                    "locale": self.locale,
                    "domain": self.domain,
                    "instruction": self.instruction,
                    "state": self.state,
                    "criteria": self.criteria,
                    "selected_criterion_id": self.selected_criterion_id,
                }
            )
        except ValidationError as error:
            raise ValueError("author task contract") from error
        return self


def _lineage_placeholder(actor: Literal["author", "reviewer"], task_id: str) -> dict[str, str]:
    """Supply syntactic lineage only for isolated contract tests.

    Production rows overwrite this with a settled provider-journal entry before
    review or packet persistence. The packet validator never treats this value
    as evidence because it cannot bind it to the durable journal/ledger.
    """

    digest = _sha(_canonical({"actor": actor, "task_id": task_id}))
    return {
        f"{actor}_response_sha256": digest,
        f"{actor}_reservation_id": f"reservation-{digest}",
        f"{actor}_request_id": f"fixture-{actor}-{digest}",
    }


def lineage_from_journal(
    journal: Mapping[str, Any], actor: Literal["author", "reviewer"]
) -> dict[str, str]:
    """Return the non-secret per-record lineage bound to one settled request."""

    required = {"stage", "reservation_id", "request_id", "task_ids", "response_sha256"}
    if set(journal) != required or not all(
        isinstance(journal[key], str) for key in required - {"task_ids"}
    ):
        raise CorpusError("provider journal lineage")
    stage = f"corpus_{actor}"
    if journal["stage"] != stage:
        raise CorpusError("provider journal stage")
    reservation_id = cast(str, journal["reservation_id"])
    response_sha256 = cast(str, journal["response_sha256"])
    if not re.fullmatch(r"reservation-[0-9a-f]{64}", reservation_id) or not re.fullmatch(
        r"[0-9a-f]{64}", response_sha256
    ):
        raise CorpusError("provider journal lineage")
    return {
        f"{actor}_response_sha256": response_sha256,
        f"{actor}_reservation_id": reservation_id,
        f"{actor}_request_id": cast(str, journal["request_id"]),
    }


class AcceptedPacketRow(_Closed):
    """One sealed synthetic row with its independently matching review receipt."""

    task_id: str = Field(pattern=r"^task-[0-9a-f]{64}$")
    family_id: str = Field(pattern=r"^family-[0-9a-f]{64}$")
    locale: Literal["pt-BR", "en"]
    domain: str
    instruction: str = Field(min_length=1, max_length=120)
    state: dict[str, JsonValue] = Field(min_length=1, max_length=64)
    criteria: list[ChoiceCriterion] = Field(min_length=2, max_length=8)
    selected_criterion_id: str
    split: Literal["synthetic_train", "synthetic_dev", "synthetic_holdout"]
    pair_id: str | None
    axes: dict[str, str]
    semantic_equivalence_attestation: SemanticEquivalenceAttestation
    cross_locale_attestation: CrossLocaleAttestation | None
    synthetic_only: Literal[True]
    review: ReviewerRecord
    review_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    author_response_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    author_reservation_id: str = Field(pattern=r"^reservation-[0-9a-f]{64}$")
    author_request_id: str = Field(min_length=1, max_length=256)
    reviewer_response_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reviewer_reservation_id: str = Field(pattern=r"^reservation-[0-9a-f]{64}$")
    reviewer_request_id: str = Field(min_length=1, max_length=256)

    @model_validator(mode="after")
    def matching_review_receipt(self) -> AcceptedPacketRow:
        try:
            UniversalTask.model_validate(
                {
                    "task_id": self.task_id,
                    "family_id": self.family_id,
                    "locale": self.locale,
                    "domain": self.domain,
                    "instruction": self.instruction,
                    "state": self.state,
                    "criteria": self.criteria,
                    "selected_criterion_id": self.selected_criterion_id,
                }
            )
        except ValidationError as error:
            raise ValueError("accepted task contract") from error
        review = self.review.model_dump(mode="json")
        if (
            self.review.status != "accepted"
            or self.review.task_id != self.task_id
            or self.review.selected_criterion_id != self.selected_criterion_id
            or not self.review.natural_language
            or not self.review.fictional
            or not self.review.exclusive_options
            or self.review.private_or_sensitive
        ):
            raise ValueError("accepted review does not match task")
        if self.review_sha256 != _sha(_canonical(review)):
            raise ValueError("accepted review integrity")
        selected_position = next(
            index
            for index, criterion in enumerate(self.criteria)
            if criterion.id == self.selected_criterion_id
        )
        author_attestation = self.semantic_equivalence_attestation
        if (
            len(author_attestation.criterion_roles) != len(self.criteria)
            or author_attestation.selected_role
            != author_attestation.criterion_roles[selected_position]
            or _canonical(author_attestation.model_dump(mode="json"))
            != _canonical(self.review.semantic_equivalence_attestation.model_dump(mode="json"))
        ):
            raise ValueError("author and reviewer semantic attestations do not match")
        if self.pair_id is None:
            if self.cross_locale_attestation is not None:
                raise ValueError("unpaired task cannot have a cross-locale attestation")
        else:
            attestation = self.cross_locale_attestation
            if (
                attestation is None
                or attestation.pair_id != self.pair_id
                or _canonical(_author_semantic_attestation(attestation))
                != _canonical(author_attestation.model_dump(mode="json"))
            ):
                raise ValueError("cross-locale attestation does not bind task semantics")
        return self


def _author_wire_properties(slot: Mapping[str, Any], index: int) -> dict[str, dict[str, Any]]:
    """Return a flat Gemini-compatible schema for one logical author record.

    The pinned route has demonstrated that object schemas nested inside array
    items can be reduced to empty objects in transit. Flat scalar properties
    keep every semantic constraint provider-visible; local code rehydrates the
    logical record before the authoritative Pydantic validation.
    """

    required = {
        "task_id",
        "family_id",
        "locale",
        "domain",
        "axes",
        "option_count",
        "gold_position",
        "pair_id",
        "semantic_target",
    }
    if not required <= set(slot):
        raise CorpusError("author batch slot")
    target = _planned_semantic_target(slot)
    option_count = slot["option_count"]
    if not isinstance(option_count, int) or not 2 <= option_count <= 8:
        raise CorpusError("author batch option count")
    prefix = f"record_{index}_"
    properties: dict[str, dict[str, Any]] = {
        f"{prefix}instruction": {
            "type": "string",
            "minLength": 1,
            "maxLength": 120,
        },
        f"{prefix}state_summary": {
            "type": "string",
            "minLength": 1,
            "maxLength": 180,
        },
        f"{prefix}selected_index": {"type": "integer", "enum": [0]},
        f"{prefix}scenario": {"type": "string", "enum": [target["scenario"]]},
        f"{prefix}selected_role": {
            "type": "string",
            "enum": [target["selected_role"]],
        },
    }
    for criterion_index, role in enumerate(target["criterion_roles"]):
        properties[f"{prefix}criterion_{criterion_index}_description"] = {
            "type": "string",
            "minLength": 1,
            "maxLength": 120,
        }
        properties[f"{prefix}criterion_{criterion_index}_role"] = {
            "type": "string",
            "enum": [role],
        }
    return properties


def _validate_author_batch(slots: Sequence[Mapping[str, Any]]) -> None:
    if not slots:
        raise CorpusError("invalid author batch")
    if len(slots) > 2:
        raise CorpusError("invalid author batch")
    targets = [_planned_semantic_target(slot) for slot in slots]
    if len(slots) == 1:
        if slots[0].get("pair_id") is not None:
            raise CorpusError("cross-locale author batch")
        return
    pair_ids = {slot.get("pair_id") for slot in slots}
    if (
        len(pair_ids) != 1
        or not isinstance(next(iter(pair_ids)), str)
        or {slot.get("locale") for slot in slots} != set(LOCALES)
    ):
        raise CorpusError("cross-locale author batch")
    if _canonical(targets[0]) != _canonical(targets[1]):
        raise CorpusError("cross-locale author semantic target")


def author_schema(slots: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return an exact response schema for this planned author request.

    The transport schema is deliberately flat because the pinned Gemini route
    has returned empty objects for schemas nested inside array items. The
    logical response remains one or two records and is reconstructed locally.
    """

    _validate_author_batch(slots)
    properties: dict[str, dict[str, Any]] = {}
    for index, slot in enumerate(slots):
        properties.update(_author_wire_properties(slot, index))
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(properties),
        "properties": properties,
    }


def decode_author_response(
    payload: Mapping[str, Any], slots: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Rehydrate the flat provider wire format without retaining unknown keys."""

    _validate_author_batch(slots)
    if set(payload) == {"records"}:
        legacy_records = payload["records"]
        if not isinstance(legacy_records, list):
            raise CorpusError("author record count")
        return [dict(record) if isinstance(record, Mapping) else {} for record in legacy_records]
    if not payload:
        raise CorpusError("author record count")
    expected = {
        key for index, slot in enumerate(slots) for key in _author_wire_properties(slot, index)
    }
    unexpected = bool(set(payload) - expected)
    output_records: list[dict[str, Any]] = []
    for index, slot in enumerate(slots):
        prefix = f"record_{index}_"
        option_count = cast(int, slot["option_count"])
        record: dict[str, Any] = {}
        instruction_key = f"{prefix}instruction"
        if instruction_key in payload:
            record["instruction"] = payload[instruction_key]
        state_key = f"{prefix}state_summary"
        record["state"] = {"summary": payload[state_key]} if state_key in payload else {}
        criteria: list[dict[str, Any]] = []
        roles: list[Any] = []
        for criterion_index in range(option_count):
            description_key = f"{prefix}criterion_{criterion_index}_description"
            role_key = f"{prefix}criterion_{criterion_index}_role"
            criteria.append(
                {"description": payload[description_key]} if description_key in payload else {}
            )
            if role_key in payload:
                roles.append(payload[role_key])
        record["criteria"] = criteria
        selected_index_key = f"{prefix}selected_index"
        if selected_index_key in payload:
            record["selected_index"] = payload[selected_index_key]
        attestation: dict[str, Any] = {"criterion_roles": roles}
        scenario_key = f"{prefix}scenario"
        selected_role_key = f"{prefix}selected_role"
        if scenario_key in payload:
            attestation["scenario"] = payload[scenario_key]
        if selected_role_key in payload:
            attestation["selected_role"] = payload[selected_role_key]
        record["semantic_equivalence_attestation"] = attestation
        if unexpected:
            record["_wire_schema_error"] = True
        output_records.append(record)
    return output_records


def reviewer_schema(batch_size: int) -> dict[str, Any]:
    if batch_size < 1:
        raise CorpusError("invalid reviewer batch")
    schema = ReviewerGeneratedRecord.model_json_schema()
    definitions = schema.pop("$defs", {})
    result: dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "required": ["reviews"],
        "properties": {
            "reviews": {
                "type": "array",
                "minItems": batch_size,
                "maxItems": batch_size,
                "items": schema,
            }
        },
    }
    if definitions:
        result["$defs"] = definitions
    return result


def author_messages(slots: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    _validate_author_batch(slots)
    return [
        {
            "role": "system",
            "content": "Return the flat JSON object defined by the response schema, one numbered field group per supplied slot in order. Each group contains instruction, state summary, criterion descriptions, selected index, scenario, criterion roles, and selected role. Restate each closed semantic_target exactly, materialize it in rule, facts, and options, and set every selected_index=0. "
            + SCENARIO_CODEBOOK
            + " "
            + CRITERION_ROLE_CODEBOOK
            + " Infer roles from rule, facts, and options; do not write scenario or role labels in task text. State has the rule and needed facts; options are distinct, mutually exclusive, and have exactly one match. No ties, conditions, compounds, or catch-alls. Use generic fictional entities only: no personal data, credentials, real organizations, URLs, identifiers, @, or digit sequences over four. Instruction and criteria are at most 120 characters; state.summary at most 180 characters. Paired locales have equivalent meaning and identical attestations. Local code binds metadata, IDs, and gold position.",
        },
        {"role": "user", "content": _canonical({"slots": list(slots)}).decode("utf-8")},
    ]


def reviewer_messages(rows: Sequence[ValidatedAuthorRow]) -> list[dict[str, str]]:
    """Build the only reviewer prompt from one validated author row.

    Review happens one task at a time so a reviewer cannot infer pair metadata
    from siblings.  The transport view is an allowlist, not a redaction of
    caller-provided messages.
    """
    if len(rows) != 1 or not isinstance(rows[0], ValidatedAuthorRow):
        raise CorpusError("reviewer requires one validated author row")
    safe = [_reviewer_task_view(rows[0])]
    _validate_reviewer_task_view(safe[0])
    payload = _canonical({"tasks": safe}).decode("utf-8")
    return [
        {
            "role": "system",
            "content": "Return a JSON object whose sole top-level key is reviews, one review with status, chosen criterion ID, reason codes, four quality flags, and independent semantic_equivalence_attestation. "
            + SCENARIO_CODEBOOK
            + " "
            + CRITERION_ROLE_CODEBOOK
            + " Infer roles from rule, facts, and options; do not write role labels into criteria. Attest one scenario, distinct ordered roles, and selected_role=matches_rule at the chosen position. Select one criterion or reject. Independently judge fictionality from supplied content; do not treat provenance or stated synthetic intent as proof. fictional=true only for fictional content without an identifiable real person, organization, account, URL, credential, or private record. exclusive_options=true only if exactly one criterion is best. Check language, sufficiency, privacy, and sensitive patterns. You do not receive answer, author attestation, pair, gold position, sibling, or split metadata.",
        },
        {"role": "user", "content": payload},
    ]


def _reviewer_task_view(row: ValidatedAuthorRow) -> dict[str, Any]:
    """Construct the exact answer-blind transport object from a typed row."""

    return {
        "task_id": row.task_id,
        "family_id": row.family_id,
        "locale": row.locale,
        "domain": row.domain,
        "instruction": row.instruction,
        "state": row.state,
        "criteria": [
            {"id": criterion.id, "description": criterion.description} for criterion in row.criteria
        ],
    }


def _validate_reviewer_json(value: object) -> None:
    """Validate JSON structure without treating any string value as metadata."""

    if value is None or isinstance(value, (bool, str, int, float)):
        return
    if isinstance(value, list):
        for item in value:
            _validate_reviewer_json(item)
        return
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        for item in value.values():
            _validate_reviewer_json(item)
        return
    raise CorpusError("reviewer transport value")


def _validate_reviewer_task_view(value: object) -> None:
    """Prove the review transport view has only its explicit public fields."""

    if not isinstance(value, dict) or set(value) != {
        "task_id",
        "family_id",
        "locale",
        "domain",
        "instruction",
        "state",
        "criteria",
    }:
        raise CorpusError("reviewer transport fields")
    if any(
        not isinstance(value[field], str)
        for field in ("task_id", "family_id", "locale", "domain", "instruction")
    ) or not isinstance(value["state"], dict):
        raise CorpusError("reviewer transport types")
    _validate_reviewer_json(value["state"])
    criteria = value["criteria"]
    if not isinstance(criteria, list) or not 2 <= len(criteria) <= 8:
        raise CorpusError("reviewer transport criteria")
    for criterion in criteria:
        if (
            not isinstance(criterion, dict)
            or set(criterion) != {"id", "description"}
            or not isinstance(criterion["id"], str)
            or not isinstance(criterion["description"], str)
        ):
            raise CorpusError("reviewer transport criterion")


def _materialize_author_record(
    record: Mapping[str, Any], planned: Mapping[str, Any]
) -> dict[str, Any]:
    """Bind model-generated content to immutable local planner metadata."""

    planner_fields = (
        "task_id",
        "family_id",
        "locale",
        "domain",
        "axes",
        "option_count",
        "gold_position",
    )
    if any(field in record for field in planner_fields):
        try:
            parsed_full = AuthorRecord.model_validate(record)
        except ValidationError as error:
            details = ",".join(
                f"{'.'.join(str(part) for part in item['loc'])}:{item['type']}"
                for item in error.errors(include_input=False, include_url=False)[:8]
            )
            raise CorpusError(f"author record schema ({details})") from error
        if any(getattr(parsed_full, field) != planned[field] for field in planner_fields):
            raise CorpusError("author changed planner-owned field")
        if _canonical(
            parsed_full.semantic_equivalence_attestation.model_dump(mode="json")
        ) != _canonical(_reordered_target(planned)):
            raise CorpusError("author semantic target mismatch")
        if not _semantic_labels_absent(
            parsed_full.model_dump(mode="json"), _planned_semantic_target(planned)
        ):
            raise CorpusError("author wrote semantic label into task text")
        return parsed_full.model_dump(mode="json")
    try:
        generated = AuthorGeneratedRecord.model_validate(record).model_dump(mode="json")
    except ValidationError as error:
        details = ",".join(
            f"{'.'.join(str(part) for part in item['loc'])}:{item['type']}"
            for item in error.errors(include_input=False, include_url=False)[:8]
        )
        raise CorpusError(f"author record schema ({details})") from error
    state = generated["state"]
    if not isinstance(state, dict) or not isinstance(state.get("summary"), str):
        raise CorpusError("author state schema")
    criteria = generated["criteria"]
    selected_index = generated.pop("selected_index")
    semantic_attestation = generated["semantic_equivalence_attestation"]
    option_count = planned["option_count"]
    gold_position = planned["gold_position"]
    target = _planned_semantic_target(planned)
    if (
        not isinstance(criteria, list)
        or not isinstance(option_count, int)
        or len(criteria) != option_count
        or not isinstance(selected_index, int)
        or not 0 <= selected_index < option_count
        or not isinstance(semantic_attestation, dict)
        or not isinstance(semantic_attestation.get("criterion_roles"), list)
        or len(semantic_attestation["criterion_roles"]) != option_count
        or semantic_attestation.get("selected_role") != "matches_rule"
        or semantic_attestation["criterion_roles"][selected_index]
        != semantic_attestation["selected_role"]
        or not isinstance(gold_position, int)
        or not 0 <= gold_position < option_count
        or selected_index != 0
        or _canonical(semantic_attestation) != _canonical(target)
    ):
        raise CorpusError("author semantic target mismatch")
    if not _semantic_labels_absent(generated, target):
        raise CorpusError("author wrote semantic label into task text")
    selected_criterion = criteria.pop(selected_index)
    criteria.insert(gold_position, selected_criterion)
    selected_role = semantic_attestation["criterion_roles"].pop(selected_index)
    semantic_attestation["criterion_roles"].insert(gold_position, selected_role)
    criteria = [
        {
            "id": f"criterion-{index}",
            "description": criterion["description"],
        }
        for index, criterion in enumerate(criteria)
    ]
    generated["criteria"] = criteria
    selected = criteria[gold_position]
    if not isinstance(selected, dict) or not isinstance(selected.get("id"), str):
        raise CorpusError("author criterion identity")
    generated["selected_criterion_id"] = selected["id"]
    generated["cross_locale_attestation"] = (
        None
        if planned["pair_id"] is None
        else {
            "pair_id": planned["pair_id"],
            **semantic_attestation,
        }
    )
    return {**{field: planned[field] for field in planner_fields}, **generated}


def _author_records_in_plan_order(
    records: Any, slots: Sequence[Mapping[str, Any]]
) -> list[Mapping[str, Any]]:
    """Bind a well-formed response batch to its immutable planned order.

    Count and identity failures mean the provider response cannot be assigned
    safely to planned slots, so they are intentionally fatal rather than slot
    rejections.
    """
    if not isinstance(records, list) or len(records) != len(slots):
        raise CorpusError("author record count")
    if not all(isinstance(record, Mapping) for record in records):
        raise CorpusError("author record schema")
    typed = cast(list[Mapping[str, Any]], records)
    has_identity = ["task_id" in record for record in typed]
    if any(has_identity) and not all(has_identity):
        raise CorpusError("author record schema")
    if all(has_identity):
        by_id: dict[str, Mapping[str, Any]] = {}
        for record in typed:
            task_id = record.get("task_id")
            if not isinstance(task_id, str):
                raise CorpusError("author record schema")
            if task_id in by_id:
                raise CorpusError("duplicate author task")
            by_id[task_id] = record
        if set(by_id) != {cast(str, slot["task_id"]) for slot in slots}:
            raise CorpusError("author task identity")
        return [by_id[cast(str, slot["task_id"])] for slot in slots]
    return typed


def _materialize_author_rows(
    records: Any, slots: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    return [
        _materialize_author_record(record, slot)
        for record, slot in zip(_author_records_in_plan_order(records, slots), slots, strict=True)
    ]


def materialize_reviewer_record(
    record: Mapping[str, Any], row: ValidatedAuthorRow
) -> dict[str, Any]:
    """Bind a blind provider review to local identity without deriving semantics."""

    if "task_id" in record:
        try:
            full = ReviewerRecord.model_validate(record)
        except ValidationError as error:
            raise CorpusError("review record schema") from error
        if full.task_id != row.task_id:
            raise CorpusError("review task identity")
        return full.model_dump(mode="json")
    try:
        generated = ReviewerGeneratedRecord.model_validate(record).model_dump(mode="json")
    except ValidationError as error:
        raise CorpusError("review record schema") from error
    generated["task_id"] = row.task_id
    return generated


def _author_task(record: Mapping[str, Any], planned: Mapping[str, Any]) -> UniversalTask:
    try:
        parsed = AuthorRecord.model_validate(record)
    except ValidationError as error:
        raise CorpusError("author record schema") from error
    planned_fields = (
        "task_id",
        "family_id",
        "locale",
        "domain",
        "axes",
        "option_count",
        "gold_position",
    )
    if any(getattr(parsed, field) != planned[field] for field in planned_fields):
        raise CorpusError("author changed planner-owned field")
    if parsed.gold_position >= parsed.option_count or len(parsed.criteria) != parsed.option_count:
        raise CorpusError("author option cardinality")
    if parsed.criteria[parsed.gold_position].id != parsed.selected_criterion_id:
        raise CorpusError("author gold position mismatch")
    semantic_attestation = parsed.semantic_equivalence_attestation
    selected_position = next(
        index
        for index, criterion in enumerate(parsed.criteria)
        if criterion.id == parsed.selected_criterion_id
    )
    if (
        len(semantic_attestation.criterion_roles) != parsed.option_count
        or semantic_attestation.selected_role
        != semantic_attestation.criterion_roles[selected_position]
    ):
        raise CorpusError("author semantic attestation")
    if _canonical(semantic_attestation.model_dump(mode="json")) != _canonical(
        _reordered_target(planned)
    ):
        raise CorpusError("author semantic target mismatch")
    if not _semantic_labels_absent(
        parsed.model_dump(mode="json"), _planned_semantic_target(planned)
    ):
        raise CorpusError("author wrote semantic label into task text")
    attestation = parsed.cross_locale_attestation
    if planned["pair_id"] is None:
        if attestation is not None:
            raise CorpusError("unpaired cross-locale attestation")
    elif (
        attestation is None
        or attestation.pair_id != planned["pair_id"]
        or _canonical(_author_semantic_attestation(attestation))
        != _canonical(semantic_attestation.model_dump(mode="json"))
    ):
        raise CorpusError("cross-locale semantic attestation")
    try:
        task = UniversalTask(
            task_id=parsed.task_id,
            family_id=parsed.family_id,
            locale=parsed.locale,
            domain=parsed.domain,
            instruction=parsed.instruction,
            state=parsed.state,
            criteria=tuple(parsed.criteria),
            selected_criterion_id=parsed.selected_criterion_id,
        )
    except ValidationError as error:
        details = ",".join(
            f"{'.'.join(str(part) for part in item['loc'])}:{item['type']}"
            for item in error.errors(include_input=False, include_url=False)[:8]
        )
        raise CorpusError(f"author task contract ({details})") from error
    return task


def validate_author_rows(
    records: Any, slots: Sequence[Mapping[str, Any]], counter: TokenCounter
) -> list[dict[str, Any]]:
    materialized = _materialize_author_rows(records, slots)
    result: list[dict[str, Any]] = []
    for record, slot in zip(materialized, slots, strict=True):
        task = _author_task(record, slot)
        try:
            validate_rendered_capacity(render_task(task), counter)
        except CapacityError as error:
            raise CorpusError("verified tokenizer capacity rejection") from error
        rendered = task.model_dump(mode="json")
        rendered["split"] = slot["split"]
        rendered["pair_id"] = slot["pair_id"]
        rendered["axes"] = slot["axes"]
        rendered["semantic_equivalence_attestation"] = AuthorRecord.model_validate(
            record
        ).semantic_equivalence_attestation.model_dump(mode="json")
        rendered["cross_locale_attestation"] = (
            task_attestation.model_dump(mode="json")
            if (task_attestation := AuthorRecord.model_validate(record).cross_locale_attestation)
            is not None
            else None
        )
        result.append(rendered)
    return result


def _author_rejection_reason(
    error: CorpusError,
) -> str:
    """Classify a local author failure without retaining rejected content."""

    if str(error) == "author semantic target mismatch":
        return "semantic_target_mismatch"
    if str(error) == "author wrote semantic label into task text":
        return "semantic_label_leakage"
    message = str(error)
    if message.startswith("author record schema"):
        return _bounded_contract_reason("author_record_schema", message)
    if message == "author state schema":
        return "author_state_schema"
    if message == "author option cardinality":
        return "author_option_cardinality"
    if message == "author gold position mismatch":
        return "author_gold_position_mismatch"
    if message == "author semantic attestation":
        return "author_semantic_attestation"
    if message == "author criterion identity":
        return "author_criterion_identity"
    if message in {"unpaired cross-locale attestation", "cross-locale semantic attestation"}:
        return "author_cross_locale_attestation"
    if message.startswith("author task contract"):
        return _bounded_contract_reason("author_task_contract", message)
    return "source_contract"


def _bounded_contract_reason(prefix: str, message: str) -> str:
    """Retain only Pydantic field paths/error types, never rejected values."""

    if "(" not in message or not message.endswith(")"):
        return prefix
    details = message.split("(", 1)[1][:-1].casefold()
    signatures: list[str] = []
    for detail in details.split(",")[:8]:
        path, separator, error_type = detail.rpartition(":")
        if not separator or not re.fullmatch(r"[a-z0-9_]+", error_type):
            signatures.append("unknown:unknown")
            continue
        signatures.append(f"{_known_contract_path(path)}:{error_type}")
    bounded = "_".join(signatures)[:160]
    return f"{prefix}__{bounded}" if bounded else prefix


def _known_contract_path(path: str) -> str:
    """Map provider-influenced validation locations to a static taxonomy."""

    if path in {"instruction", "state", "state.summary", "criteria", "selected_index"}:
        return path
    if re.fullmatch(r"criteria\.\d+\.description", path):
        return "criteria.description"
    if path in {
        "semantic_equivalence_attestation",
        "semantic_equivalence_attestation.scenario",
        "semantic_equivalence_attestation.criterion_roles",
        "semantic_equivalence_attestation.selected_role",
    }:
        return path
    return "unknown"


def classify_author_rows(
    records: Any, slots: Sequence[Mapping[str, Any]], counter: TokenCounter
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Resolve each planned identity once without preserving rejected content.

    Source and tokenizer failures belong to the already-settled planned slot.
    In contrast, batch count/identity corruption and planner-field tampering
    are fatal because accepting a partial or altered plan would break the
    precommitted identity boundary.
    """
    raw_records = _author_records_in_plan_order(records, slots)
    usable: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for raw_record, slot in zip(raw_records, slots, strict=True):
        try:
            record = _materialize_author_record(raw_record, slot)
            task = _author_task(record, slot)
        except CorpusError as error:
            if str(error) == "author changed planner-owned field":
                raise
            rejected.append(
                {
                    "task_id": slot["task_id"],
                    "split": slot["split"],
                    "reason": _author_rejection_reason(error),
                }
            )
            continue
        try:
            validate_rendered_capacity(render_task(task), counter)
        except CapacityError:
            rejected.append(
                {"task_id": slot["task_id"], "split": slot["split"], "reason": "capacity"}
            )
            continue
        row = task.model_dump(mode="json")
        row["split"] = slot["split"]
        row["pair_id"] = slot["pair_id"]
        row["axes"] = slot["axes"]
        row["semantic_equivalence_attestation"] = AuthorRecord.model_validate(
            record
        ).semantic_equivalence_attestation.model_dump(mode="json")
        row["cross_locale_attestation"] = (
            task_attestation.model_dump(mode="json")
            if (task_attestation := AuthorRecord.model_validate(record).cross_locale_attestation)
            is not None
            else None
        )
        usable.append(row)
    return usable, rejected


def validate_author_rows_with_verified_minilm(
    records: Any, slots: Sequence[Mapping[str, Any]], snapshot: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Resolve every planned slot with the verified MiniLM tokenizer.

    Capacity is a planned-slot outcome, not an exceptional path in production:
    an overflow is retained as an immutable rejection in its assigned split.
    """
    return classify_author_rows(records, slots, VerifiedMiniLMTokenizerReceipt.create(snapshot))


def _normalized(row: Mapping[str, Any]) -> str:
    value = _canonical({key: row[key] for key in ("instruction", "state", "criteria")}).decode(
        "utf-8"
    )
    return " ".join(value.casefold().split())


def _author_semantic_attestation(
    attestation: CrossLocaleAttestation,
) -> dict[str, Any]:
    """Remove planner-only pair identity before comparing independent statements."""
    return {
        "scenario": attestation.scenario,
        "criterion_roles": attestation.criterion_roles,
        "selected_role": attestation.selected_role,
    }


def _cross_locale_attestation(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return the author's language-neutral attestation for a paired row."""
    try:
        attestation = CrossLocaleAttestation.model_validate(row["cross_locale_attestation"])
    except (KeyError, ValidationError) as error:
        raise CorpusError("cross-locale semantic attestation") from error
    if attestation.pair_id != row.get("pair_id"):
        raise CorpusError("cross-locale semantic attestation")
    return _author_semantic_attestation(attestation)


def _reviewer_semantic_attestation(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return the reviewer's independently emitted semantic statement."""
    try:
        review = ReviewerRecord.model_validate(row["review"])
    except (KeyError, ValidationError) as error:
        raise CorpusError("cross-locale reviewer attestation") from error
    return review.semantic_equivalence_attestation.model_dump(mode="json")


def _attestation_disagreement_reason(
    author: SemanticEquivalenceAttestation,
    reviewer: SemanticEquivalenceAttestation,
    *,
    option_count: int,
    selected_position: int,
) -> Literal["scenario_disagreement", "criterion_role_disagreement"] | None:
    """Classify only a closed attestation mismatch; never retain reasoning."""
    if author.scenario != reviewer.scenario:
        return "scenario_disagreement"
    if (
        len(reviewer.criterion_roles) != option_count
        or reviewer.selected_role != reviewer.criterion_roles[selected_position]
        or author.criterion_roles != reviewer.criterion_roles
        or author.selected_role != reviewer.selected_role
    ):
        return "criterion_role_disagreement"
    return None


def _cross_locale_pair_disagreement_reason(
    rows: Sequence[Mapping[str, Any]],
) -> Literal["scenario_disagreement", "criterion_role_disagreement"]:
    """Classify a pair-only mismatch without retaining row content."""
    try:
        author_attestations = [_cross_locale_attestation(row) for row in rows]
        reviewer_attestations = [_reviewer_semantic_attestation(row) for row in rows]
    except CorpusError:
        return "criterion_role_disagreement"
    if (
        len({attestation["scenario"] for attestation in author_attestations}) != 1
        or len({attestation["scenario"] for attestation in reviewer_attestations}) != 1
    ):
        return "scenario_disagreement"
    return "criterion_role_disagreement"


def _cross_locale_pair_is_attested(rows: Sequence[Mapping[str, Any]]) -> bool:
    """Require matching author and reviewer semantics before pair acceptance."""
    if len(rows) != 2 or {row.get("locale") for row in rows} != set(LOCALES):
        return False
    try:
        author_attestations = [_canonical(_cross_locale_attestation(row)) for row in rows]
        reviewer_attestations = [_canonical(_reviewer_semantic_attestation(row)) for row in rows]
    except CorpusError:
        return False
    return (
        author_attestations[0] == author_attestations[1]
        and reviewer_attestations[0] == reviewer_attestations[1]
        and author_attestations == reviewer_attestations
    )


def semantic_fingerprint(row: Mapping[str, Any]) -> str:
    """Return the global content identity, excluding criterion IDs and order.

    Family and split are deliberately absent: a semantic duplicate must be
    rejected rather than moved into another family or partition.
    """
    criteria = row.get("criteria")
    selected_id = row.get("selected_criterion_id")
    if not isinstance(criteria, Sequence) or isinstance(criteria, (str, bytes)):
        raise CorpusError("semantic fingerprint criteria")
    descriptions: list[str] = []
    selected_description: str | None = None
    for criterion in criteria:
        if not isinstance(criterion, Mapping):
            raise CorpusError("semantic fingerprint criterion")
        criterion_id = criterion.get("id")
        description = criterion.get("description")
        if not isinstance(criterion_id, str) or not isinstance(description, str):
            raise CorpusError("semantic fingerprint criterion")
        normalized_description = " ".join(description.casefold().split())
        descriptions.append(normalized_description)
        if criterion_id == selected_id:
            selected_description = normalized_description
    if not isinstance(row.get("instruction"), str) or selected_description is None:
        raise CorpusError("semantic fingerprint task")
    value = {
        "instruction": " ".join(cast(str, row["instruction"]).casefold().split()),
        "state": row.get("state"),
        "criteria": sorted(descriptions),
        "selected_criterion_description": selected_description,
    }
    return _sha(_canonical(value))


def _privacy(row: Mapping[str, Any]) -> bool:
    return bool(_SENSITIVE.search(_normalized(row)))


def resolve_reviews(
    rows: Sequence[Mapping[str, Any]],
    reviews: Any,
    prior_rows: Iterable[Mapping[str, Any]] = (),
    *,
    reviewer_lineages: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not isinstance(reviews, list) or len(reviews) != len(rows):
        raise CorpusError("review record count")
    by_id: dict[str, ReviewerRecord] = {}
    for raw in reviews:
        try:
            review = ReviewerRecord.model_validate(raw)
        except ValidationError as error:
            raise CorpusError("review schema") from error
        if review.task_id in by_id:
            raise CorpusError("duplicate review task")
        by_id[review.task_id] = review
    if set(by_id) != {cast(str, row["task_id"]) for row in rows}:
        raise CorpusError("review identity")
    seen = {semantic_fingerprint(row) for row in prior_rows}
    normalized_seen = [_normalized(row) for row in prior_rows]
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for row in rows:
        review = by_id[cast(str, row["task_id"])]
        task_id = cast(str, row["task_id"])
        author_lineage = {
            key: row[key]
            for key in (
                "author_response_sha256",
                "author_reservation_id",
                "author_request_id",
            )
            if key in row
        }
        if not author_lineage:
            author_lineage = _lineage_placeholder("author", task_id)
        if set(author_lineage) != {
            "author_response_sha256",
            "author_reservation_id",
            "author_request_id",
        }:
            raise CorpusError("author provider lineage")
        reviewer_lineage = (
            dict(reviewer_lineages[task_id])
            if reviewer_lineages is not None and task_id in reviewer_lineages
            else _lineage_placeholder("reviewer", task_id)
        )
        if set(reviewer_lineage) != {
            "reviewer_response_sha256",
            "reviewer_reservation_id",
            "reviewer_request_id",
        }:
            raise CorpusError("reviewer provider lineage")
        reason: str | None = None
        if review.status != "accepted":
            reason = "review_rejected"
        elif review.selected_criterion_id != row["selected_criterion_id"]:
            reason = "review_disagreement"
        else:
            selected_position = next(
                index
                for index, criterion in enumerate(row["criteria"])
                if criterion["id"] == review.selected_criterion_id
            )
            reason = _attestation_disagreement_reason(
                SemanticEquivalenceAttestation.model_validate(
                    row["semantic_equivalence_attestation"]
                ),
                review.semantic_equivalence_attestation,
                option_count=len(row["criteria"]),
                selected_position=selected_position,
            )
        if reason is None and not review.natural_language:
            reason = "review_quality_natural_language"
        elif reason is None and not review.fictional:
            reason = "review_quality_fictional"
        elif reason is None and not review.exclusive_options:
            reason = "review_quality_exclusive_options"
        elif reason is None and (review.private_or_sensitive or _privacy(row)):
            reason = "privacy"
        elif reason is None:
            fingerprint = semantic_fingerprint(row)
            normalized = _normalized(row)
            if fingerprint in seen:
                reason = "semantic_duplicate"
            elif any(
                SequenceMatcher(None, normalized, item).ratio() >= 0.92 for item in normalized_seen
            ):
                reason = "near_duplicate"
        if reason:
            rejected.append(
                {
                    "task_id": task_id,
                    "split": row["split"],
                    "reason": reason,
                    **author_lineage,
                    **reviewer_lineage,
                }
            )
        else:
            review_payload = review.model_dump(mode="json")
            accepted.append(
                {
                    **row,
                    "synthetic_only": True,
                    "review": review_payload,
                    "review_sha256": _sha(_canonical(review_payload)),
                    **author_lineage,
                    **reviewer_lineage,
                }
            )
            seen.add(fingerprint)
            normalized_seen.append(normalized)
    paired: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if isinstance(row.get("pair_id"), str):
            paired[cast(str, row["pair_id"])].append(row)
    accepted_by_task = {cast(str, row["task_id"]): row for row in accepted}
    invalid_pair_reasons: dict[str, str] = {}
    for members in paired.values():
        # Unit-level validators may inspect one row in isolation. Runtime
        # author batches always contain both preplanned members.
        if len(members) != 2:
            continue
        accepted_members = [
            accepted_by_task[cast(str, row["task_id"])]
            for row in members
            if cast(str, row["task_id"]) in accepted_by_task
        ]
        if len(accepted_members) != len(members):
            for row in accepted_members:
                invalid_pair_reasons[cast(str, row["task_id"])] = "paired_member_rejected"
        elif not _cross_locale_pair_is_attested(accepted_members):
            reason = _cross_locale_pair_disagreement_reason(accepted_members)
            for row in accepted_members:
                invalid_pair_reasons[cast(str, row["task_id"])] = reason
    if invalid_pair_reasons:
        paired_lineages = {
            cast(str, row["task_id"]): {
                key: row[key]
                for key in (
                    "author_response_sha256",
                    "author_reservation_id",
                    "author_request_id",
                    "reviewer_response_sha256",
                    "reviewer_reservation_id",
                    "reviewer_request_id",
                )
            }
            for row in accepted
            if row["task_id"] in invalid_pair_reasons
        }
        accepted = [row for row in accepted if row["task_id"] not in invalid_pair_reasons]
        rejected.extend(
            {
                "task_id": task_id,
                "split": next(row["split"] for row in rows if row["task_id"] == task_id),
                "reason": invalid_pair_reasons[task_id],
                **paired_lineages[task_id],
            }
            for task_id in sorted(invalid_pair_reasons)
        )
    return accepted, rejected


def _identity_projection(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return the canonical, non-content identity for one accepted row.

    This is deliberately the only representation of accepted holdout rows that
    may be released before the one-time holdout claim.  In particular, it
    never carries task text, state, criterion descriptions, review receipts,
    token data, or embeddings.
    """
    criterion_ids: object
    gold_position: object
    option_count: object
    criteria = row.get("criteria")
    selected = row.get("selected_criterion_id")
    if isinstance(criteria, Sequence) and not isinstance(criteria, (str, bytes)):
        criterion_ids = [
            criterion.get("id") if isinstance(criterion, Mapping) else None
            for criterion in criteria
        ]
        try:
            gold_position = next(
                index
                for index, criterion_id in enumerate(criterion_ids)
                if criterion_id == selected
            )
        except StopIteration as error:
            raise CorpusError("holdout identity criterion") from error
        option_count = len(criterion_ids)
    else:
        criterion_ids = row.get("criterion_ids")
        gold_position = row.get("gold_position")
        option_count = row.get("option_count")
    identity = {
        "task_id": row.get("task_id"),
        "family_id": row.get("family_id"),
        "pair_id": row.get("pair_id"),
        "split": row.get("split"),
        "locale": row.get("locale"),
        "domain": row.get("domain"),
        "axes": row.get("axes"),
        "option_count": option_count,
        "gold_position": gold_position,
        "criterion_ids": criterion_ids,
    }
    if (
        not isinstance(identity["task_id"], str)
        or not isinstance(identity["family_id"], str)
        or (identity["pair_id"] is not None and not isinstance(identity["pair_id"], str))
        or identity["split"] not in SPLITS
        or identity["locale"] not in LOCALES
        or identity["domain"] not in DOMAINS
        or not isinstance(identity["axes"], dict)
        or identity["axes"] != {key: identity["axes"].get(key) for key in AXES}
        or any(identity["axes"].get(key) not in values for key, values in AXES.items())
        or type(identity["option_count"]) is not int
        or not 2 <= identity["option_count"] <= 8
        or type(identity["gold_position"]) is not int
        or not 0 <= identity["gold_position"] < identity["option_count"]
        or not isinstance(identity["criterion_ids"], list)
        or len(identity["criterion_ids"]) != identity["option_count"]
        or not all(isinstance(item, str) for item in identity["criterion_ids"])
        or len(set(cast(list[str], identity["criterion_ids"]))) != identity["option_count"]
    ):
        raise CorpusError("holdout identity schema")
    return cast(dict[str, Any], identity)


def _identity_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    identities = [_identity_projection(row) for row in rows]
    if len({row["task_id"] for row in identities}) != len(identities):
        raise CorpusError("holdout identity coverage")
    return sorted(identities, key=lambda row: cast(str, row["task_id"]))


def _read_holdout_identities(path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_bytes())
    except (OSError, ValueError) as error:
        raise CorpusError("holdout identities") from error
    if not isinstance(payload, dict) or set(payload) != {"schema_version", "rows"}:
        raise CorpusError("holdout identities")
    if payload["schema_version"] != "phase4e-holdout-identities.v1" or not isinstance(
        payload["rows"], list
    ):
        raise CorpusError("holdout identities")
    identities = _identity_rows(cast(list[Mapping[str, Any]], payload["rows"]))
    if identities != payload["rows"]:
        raise CorpusError("holdout identities canonical order")
    if any(row["split"] != "synthetic_holdout" for row in identities):
        raise CorpusError("holdout identities split")
    return identities


def _row_option_count(row: Mapping[str, Any]) -> int | None:
    """Read the closed option count without treating absent criteria as truthy."""

    declared = row.get("option_count")
    if type(declared) is int:
        return declared
    criteria = row.get("criteria")
    if isinstance(criteria, Sequence) and not isinstance(criteria, (str, bytes)):
        return len(criteria)
    return None


def _minimums(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    errors: list[str] = []
    split = Counter(cast(str, row["split"]) for row in rows)
    if (
        len(rows) < 1200
        or split["synthetic_train"] < 840
        or split["synthetic_dev"] < 180
        or split["synthetic_holdout"] < 180
    ):
        errors.append("global_or_split_minimum")
    if len({row["domain"] for row in rows}) < 12:
        errors.append("domain_minimum")
    for part, floor in (("synthetic_train", 20), ("synthetic_dev", 10), ("synthetic_holdout", 10)):
        if {row["domain"] for row in rows if row["split"] == part} != set(DOMAINS):
            errors.append("split_domain_minimum")
        if not any(
            row["split"] == part
            and row["axes"]["explicitness"] == "implicit"
            and row["axes"]["negation"] == "present"
            for row in rows
        ):
            errors.append("split_difficulty_minimum")
        for locale in LOCALES:
            for count in OPTION_COUNTS:
                if (
                    sum(
                        row["split"] == part
                        and row["locale"] == locale
                        and _row_option_count(row) == count
                        for row in rows
                    )
                    < floor
                ):
                    errors.append("cell_minimum")
                    break
    for part in SPLITS:
        families = {row["family_id"] for row in rows if row["split"] == part}
        if len(families) < 100:
            errors.append("family_minimum")
    locale_count = Counter(cast(str, row["locale"]) for row in rows)
    if locale_count["pt-BR"] * 100 < len(rows) * 60 or locale_count["en"] * 100 < len(rows) * 20:
        errors.append("locale_minimum")
    pair_families: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if row.get("pair_id"):
            pair_families[cast(str, row["pair_id"])].add(cast(str, row["locale"]))
    if sum(locales == set(LOCALES) for locales in pair_families.values()) * 2 < 240:
        errors.append("cross_locale_minimum")
    return sorted(set(errors))


_REJECTED_LINEAGE_FIELDS = frozenset(
    {
        "author_response_sha256",
        "author_reservation_id",
        "author_request_id",
        "reviewer_response_sha256",
        "reviewer_reservation_id",
        "reviewer_request_id",
    }
)


def _validate_rejected_row(
    rejected_row: Mapping[str, Any], slots: Mapping[str, Mapping[str, Any]]
) -> None:
    if (
        not {"task_id", "split", "reason"} <= set(rejected_row)
        or set(rejected_row) - ({"task_id", "split", "reason"} | _REJECTED_LINEAGE_FIELDS)
        or rejected_row.get("task_id") not in slots
    ):
        raise CorpusError("packet rejection")
    task_id = cast(str, rejected_row["task_id"])
    if rejected_row["split"] != slots[task_id]["split"]:
        raise CorpusError("packet rejected split mutation")


def _validate_packet_resolution(
    accepted: Sequence[Mapping[str, Any]],
    rejected: Sequence[Mapping[str, Any]],
    plan: Mapping[str, Any],
) -> None:
    slots = {
        cast(str, slot["task_id"]): slot for slot in cast(list[Mapping[str, Any]], plan["slots"])
    }
    resolved = [*accepted, *rejected]
    ids = [row.get("task_id") for row in resolved]
    if len(ids) != len(set(ids)) or set(ids) != set(slots):
        raise CorpusError("packet resolution coverage")
    fingerprints: set[str] = set()
    normalized_rows: list[str] = []
    paired_accepted: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw_row in accepted:
        try:
            row = AcceptedPacketRow.model_validate(raw_row).model_dump(mode="json")
        except ValidationError as error:
            raise CorpusError("accepted packet row schema") from error
        task_id = row["task_id"]
        if not isinstance(task_id, str) or task_id not in slots:
            raise CorpusError("packet task identity")
        planned = slots[task_id]
        if any(
            row.get(key) != planned[key]
            for key in ("family_id", "split", "locale", "domain", "axes", "pair_id")
        ):
            raise CorpusError("packet split or family mutation")
        criteria = row.get("criteria")
        if not isinstance(criteria, list) or len(criteria) != planned["option_count"]:
            raise CorpusError("packet criterion cardinality")
        position = planned["gold_position"]
        if (
            not isinstance(position, int)
            or not isinstance(row.get("selected_criterion_id"), str)
            or criteria[position].get("id") != row["selected_criterion_id"]
        ):
            raise CorpusError("packet gold position")
        if _privacy(row):
            raise CorpusError("packet privacy")
        fingerprint = semantic_fingerprint(row)
        if fingerprint in fingerprints:
            raise CorpusError("packet semantic duplicate")
        normalized = _normalized(row)
        if normalized in normalized_rows:
            raise CorpusError("packet normalized duplicate")
        if any(SequenceMatcher(None, normalized, item).ratio() >= 0.92 for item in normalized_rows):
            raise CorpusError("packet near duplicate")
        fingerprints.add(fingerprint)
        normalized_rows.append(normalized)
        if planned["pair_id"] is not None:
            paired_accepted[cast(str, planned["pair_id"])].append(row)
    for members in paired_accepted.values():
        if len(members) == 2 and not _cross_locale_pair_is_attested(members):
            raise CorpusError("packet cross-locale semantic attestation")
    for rejected_row in rejected:
        _validate_rejected_row(rejected_row, slots)


def _validate_provider_lineage(
    ledger: BudgetLedger,
    accepted: Sequence[Mapping[str, Any]],
    rejected: Sequence[Mapping[str, Any]],
    resolved_task_ids: set[str],
) -> None:
    """Bind every persisted provider-derived row to the sealed journal and ledger."""

    ledger.require_complete_provider_journal()
    evidence: dict[tuple[str, str], Mapping[str, Any]] = {}
    for journal in ledger.provider_journal:
        for task_id in cast(list[str], journal["task_ids"]):
            key = (cast(str, journal["stage"]), task_id)
            if key in evidence or task_id not in resolved_task_ids:
                raise CorpusError("provider journal task resolution")
            evidence[key] = journal

    def fields(journal: Mapping[str, Any], actor: Literal["author", "reviewer"]) -> dict[str, Any]:
        return {
            f"{actor}_response_sha256": journal["response_sha256"],
            f"{actor}_reservation_id": journal["reservation_id"],
            f"{actor}_request_id": journal["request_id"],
        }

    for row in accepted:
        accepted_task_id = row.get("task_id")
        if not isinstance(accepted_task_id, str):
            raise CorpusError("provider lineage task")
        author = evidence.get(("corpus_author", accepted_task_id))
        reviewer = evidence.get(("corpus_reviewer", accepted_task_id))
        if (
            author is None
            or reviewer is None
            or any(
                row.get(key) != value
                for source, actor in cast(
                    tuple[tuple[Mapping[str, Any], Literal["author", "reviewer"]], ...],
                    ((author, "author"), (reviewer, "reviewer")),
                )
                for key, value in fields(source, actor).items()
            )
        ):
            raise CorpusError("accepted provider lineage")
    for row in rejected:
        rejected_task_id = row.get("task_id")
        if not isinstance(rejected_task_id, str):
            raise CorpusError("provider lineage task")
        for stage, actor in cast(
            tuple[
                tuple[Literal["corpus_author", "corpus_reviewer"], Literal["author", "reviewer"]],
                ...,
            ],
            (("corpus_author", "author"), ("corpus_reviewer", "reviewer")),
        ):
            rejected_journal = evidence.get((stage, rejected_task_id))
            expected = fields(rejected_journal, actor) if rejected_journal is not None else {}
            present = {key: row[key] for key in fields_placeholder(actor) if key in row}
            if present != expected:
                raise CorpusError("rejected provider lineage")


def fields_placeholder(actor: Literal["author", "reviewer"]) -> dict[str, None]:
    return {
        f"{actor}_response_sha256": None,
        f"{actor}_reservation_id": None,
        f"{actor}_request_id": None,
    }


_PACKET_FILES = {
    "plan.json",
    "accepted-train-dev.jsonl",
    "accepted-holdout.jsonl",
    "holdout-identities.json",
    "rejected.jsonl",
    "ledger.json",
    "packet.json",
}
_PACKET_NON_HOLDOUT_FILES = _PACKET_FILES - {"accepted-holdout.jsonl", "packet.json"}


def _packet_manifest(packet: Path) -> dict[str, Any]:
    expected = _PACKET_FILES
    if not packet.is_dir() or {path.name for path in packet.iterdir()} != expected:
        raise CorpusError("packet file set")
    try:
        manifest = json.loads((packet / "packet.json").read_bytes())
    except (OSError, ValueError) as error:
        raise CorpusError("packet manifest") from error
    if (
        not isinstance(manifest, dict)
        or set(manifest) != {"schema_version", "files", "sealed"}
        or manifest["schema_version"] != "phase4e-accepted-packet.v2"
        or manifest["sealed"] is not True
        or not isinstance(manifest["files"], dict)
        or set(manifest["files"]) != expected - {"packet.json"}
        or any(
            not isinstance(value, str) or len(value) != 64 for value in manifest["files"].values()
        )
    ):
        raise CorpusError("packet manifest shape")
    return cast(dict[str, Any], manifest)


def _validate_packet_pre_holdout(
    packet: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate every non-holdout byte and the sealed identity projection.

    This deliberately never opens, parses, or hashes the holdout payload.  Its
    digest in ``packet.json`` is only precommitted metadata at this stage.
    """
    manifest = _packet_manifest(packet)
    files = cast(dict[str, str], manifest["files"])
    for name in _PACKET_NON_HOLDOUT_FILES:
        if _sha((packet / name).read_bytes()) != files[name]:
            raise CorpusError("packet digest")
    try:
        plan = json.loads((packet / "plan.json").read_bytes())
    except (OSError, ValueError) as error:
        raise CorpusError("packet plan") from error
    validate_plan(plan)
    train_dev = _read_jsonl(packet / "accepted-train-dev.jsonl")
    rejected = _read_jsonl(packet / "rejected.jsonl")
    try:
        ledger = BudgetLedger.from_json(json.loads((packet / "ledger.json").read_bytes()))
    except (OSError, ValueError) as error:
        raise CorpusError("packet ledger") from error
    holdout_identities = _read_holdout_identities(packet / "holdout-identities.json")
    slots = {
        cast(str, slot["task_id"]): slot for slot in cast(list[Mapping[str, Any]], plan["slots"])
    }
    accepted_identities: list[dict[str, Any]] = []
    for raw_row in train_dev:
        try:
            row = AcceptedPacketRow.model_validate(raw_row).model_dump(mode="json")
        except ValidationError as error:
            raise CorpusError("accepted packet row schema") from error
        if row["split"] not in {"synthetic_train", "synthetic_dev"}:
            raise CorpusError("packet train/dev split")
        accepted_identities.append(_identity_projection(row))
    accepted_identities.extend(holdout_identities)
    if len({row["task_id"] for row in accepted_identities}) != len(accepted_identities):
        raise CorpusError("packet resolution coverage")
    for identity in accepted_identities:
        task_id = identity["task_id"]
        planned = slots.get(task_id)
        if planned is None or any(
            identity[key] != planned[key]
            for key in (
                "family_id",
                "pair_id",
                "split",
                "locale",
                "domain",
                "axes",
                "option_count",
                "gold_position",
            )
        ):
            raise CorpusError("packet split or family mutation")
    resolved = [
        *(row["task_id"] for row in accepted_identities),
        *(row.get("task_id") for row in rejected),
    ]
    if len(resolved) != len(set(resolved)) or set(resolved) != set(slots):
        raise CorpusError("packet resolution coverage")
    for rejected_row in rejected:
        _validate_rejected_row(rejected_row, slots)
    _validate_provider_lineage(ledger, train_dev, rejected, set(cast(list[str], resolved)))
    if _minimums(accepted_identities):
        raise CorpusError("accepted corpus minimum")
    families: dict[str, set[str]] = defaultdict(set)
    for identity in accepted_identities:
        families[cast(str, identity["family_id"])].add(cast(str, identity["split"]))
    if any(len(splits) != 1 for splits in families.values()):
        raise CorpusError("packet family split isolation")
    return manifest, train_dev, holdout_identities


def validate_accepted_packet_pre_holdout(packet: Path) -> None:
    """Run the fail-closed pre-claim packet validator without holdout access."""
    _validate_packet_pre_holdout(packet)


def validate_accepted_packet(packet: Path) -> None:
    """Fully validate the immutable v2 packet after the holdout claim."""
    manifest, train_dev, holdout_identities = _validate_packet_pre_holdout(packet)
    files = cast(dict[str, str], manifest["files"])
    holdout_path = packet / "accepted-holdout.jsonl"
    if _sha(holdout_path.read_bytes()) != files["accepted-holdout.jsonl"]:
        raise CorpusError("packet digest")
    holdout = _read_jsonl(holdout_path)
    if any(row.get("split") != "synthetic_holdout" for row in holdout):
        raise CorpusError("packet holdout split")
    if _identity_rows(holdout) != holdout_identities:
        raise CorpusError("holdout identity projection binding")
    try:
        plan = json.loads((packet / "plan.json").read_bytes())
    except (OSError, ValueError) as error:
        raise CorpusError("packet plan") from error
    rejected = _read_jsonl(packet / "rejected.jsonl")
    _validate_packet_resolution([*train_dev, *holdout], rejected, plan)
    try:
        ledger = BudgetLedger.from_json(json.loads((packet / "ledger.json").read_bytes()))
    except (OSError, ValueError) as error:
        raise CorpusError("packet ledger") from error
    _validate_provider_lineage(
        ledger,
        [*train_dev, *holdout],
        rejected,
        {cast(str, slot["task_id"]) for slot in cast(list[Mapping[str, Any]], plan["slots"])},
    )
    if _minimums([*train_dev, *holdout]):
        raise CorpusError("accepted corpus minimum")


def _publish_packet_create_if_absent(staged: Path, packet: Path) -> None:
    """Atomically publish a fully validated sibling directory without replacement."""
    if packet.exists():
        raise FileExistsError(f"benchmark artifact already exists: {packet.name}")
    libc = CDLL(None, use_errno=True)
    rename_exclusive = getattr(libc, "renameatx_np", None)
    flag = 0x00000004  # Darwin RENAME_EXCL
    if rename_exclusive is None:
        rename_exclusive = getattr(libc, "renameat2", None)
        flag = 1  # Linux RENAME_NOREPLACE
    if rename_exclusive is None:
        raise CorpusError("atomic directory publication is unavailable")
    rename_exclusive.argtypes = (c_int, c_char_p, c_int, c_char_p, c_int)
    rename_exclusive.restype = c_int
    # Both APIs perform one same-volume exclusive rename. The initial exists
    # check is only diagnostic; this final operation is the no-clobber guard.
    set_errno(0)
    if rename_exclusive(-2, os.fsencode(staged), -2, os.fsencode(packet), flag) != 0:
        error_number = get_errno()
        if error_number == EEXIST:
            raise FileExistsError(f"benchmark artifact already exists: {packet.name}")
        raise OSError(error_number, os.strerror(error_number), packet)


def seal_packet(
    packet: Path,
    plan: Mapping[str, Any],
    accepted: Sequence[Mapping[str, Any]],
    rejected: Sequence[Mapping[str, Any]],
    ledger: Mapping[str, Any],
) -> Path:
    """Atomically create the immutable corpus packet. Existing packets are never replaced."""
    packet.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix=f".{packet.name}.", dir=packet.parent))
    os.chmod(staged, 0o700)
    accepted_train_dev = sorted(
        (row for row in accepted if row.get("split") in {"synthetic_train", "synthetic_dev"}),
        key=lambda row: cast(str, row.get("task_id")),
    )
    accepted_holdout = sorted(
        (row for row in accepted if row.get("split") == "synthetic_holdout"),
        key=lambda row: cast(str, row.get("task_id")),
    )
    identities = _identity_rows(accepted_holdout)
    files = {
        "plan.json": _canonical(plan) + b"\n",
        "accepted-train-dev.jsonl": _jsonl(accepted_train_dev),
        "accepted-holdout.jsonl": _jsonl(accepted_holdout),
        "holdout-identities.json": _canonical(
            {"schema_version": "phase4e-holdout-identities.v1", "rows": identities}
        )
        + b"\n",
        "rejected.jsonl": _jsonl(rejected),
        "ledger.json": _canonical(ledger) + b"\n",
    }
    try:
        for name, body in files.items():
            atomic_create(staged / name, body)
            os.chmod(staged / name, 0o600)
        manifest = {
            "schema_version": "phase4e-accepted-packet.v2",
            "sealed": True,
            "files": {name: _sha(body) for name, body in files.items()},
        }
        atomic_create(staged / "packet.json", _canonical(manifest) + b"\n")
        os.chmod(staged / "packet.json", 0o600)
        validate_accepted_packet_pre_holdout(staged)
        _publish_packet_create_if_absent(staged, packet)
    except BaseException:
        if staged.exists():
            for child in staged.iterdir():
                child.unlink()
            staged.rmdir()
        raise
    return packet / "packet.json"


def _jsonl(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(_canonical(row) + b"\n" for row in rows)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        return [
            cast(dict[str, Any], json.loads(line))
            for line in path.read_bytes().splitlines()
            if line
        ]
    except (OSError, ValueError) as error:
        raise CorpusError("invalid jsonl") from error


@dataclass
class BudgetLedger:
    """Parameterized four-stage ledger; the larger local/server debit wins."""

    entries: list[dict[str, str]] = field(default_factory=list)
    provider_journal: list[dict[str, Any]] = field(default_factory=list)

    def spent(self, stage: str) -> Decimal:
        return sum(
            (Decimal(entry["debit_usd"]) for entry in self.entries if entry["stage"] == stage),
            Decimal(),
        )

    def reserve(self, stage: str, local_worst_case: Decimal) -> None:
        if stage not in STAGE_LIMITS or local_worst_case < 0:
            raise CorpusError("budget stage")
        if any(entry["status"] == "overspent" for entry in self.entries):
            raise CorpusError("budget terminal overspend")
        if self.spent(stage) + local_worst_case > STAGE_LIMITS[stage]:
            raise CorpusError("stage budget exhausted")
        if (
            sum((Decimal(item["debit_usd"]) for item in self.entries), Decimal()) + local_worst_case
            > TOTAL_BUDGET
        ):
            raise CorpusError("total budget exhausted")

    def reserve_request(self, stage: str, reservation_id: str, local_worst_case: Decimal) -> None:
        """Charge a deterministic request reservation before transport starts."""
        self.reserve(stage, local_worst_case)
        if not re.fullmatch(r"reservation-[0-9a-f]{64}", reservation_id) or any(
            entry["reservation_id"] == reservation_id for entry in self.entries
        ):
            raise CorpusError("duplicate budget reservation")
        self.entries.append(
            {
                "stage": stage,
                "reservation_id": reservation_id,
                "request_id": reservation_id,
                "local_worst_case_usd": str(local_worst_case),
                "provider_cost_usd": "0",
                "debit_usd": str(local_worst_case),
                "response_sha256": "0" * 64,
                "status": "reserved",
            }
        )

    def settle_request(
        self,
        reservation_id: str,
        request_id: str,
        provider_cost: Decimal,
        response_sha256: str,
    ) -> None:
        """Settle an already charged reservation without ever releasing it."""
        entry = next(
            (item for item in self.entries if item["reservation_id"] == reservation_id), None
        )
        if entry is None or entry["status"] != "reserved":
            raise CorpusError("budget reservation")
        if (
            not request_id
            or provider_cost < 0
            or not re.fullmatch(r"[0-9a-f]{64}", response_sha256)
            or any(
                item["status"] == "settled" and item["request_id"] == request_id
                for item in self.entries
            )
        ):
            raise CorpusError("invalid provider cost")
        local_worst_case = Decimal(entry["local_worst_case_usd"])
        debit = max(local_worst_case, provider_cost)
        entry.update(
            request_id=request_id,
            provider_cost_usd=str(provider_cost),
            debit_usd=str(debit),
            response_sha256=response_sha256,
            status="settled",
        )
        if self.spent(entry["stage"]) > STAGE_LIMITS[entry["stage"]]:
            entry["status"] = "overspent"
            raise CorpusError("reported stage budget exhausted")
        if sum((Decimal(item["debit_usd"]) for item in self.entries), Decimal()) > TOTAL_BUDGET:
            entry["status"] = "overspent"
            raise CorpusError("reported total budget exhausted")

    def record(
        self,
        stage: str,
        request_id: str,
        local_worst_case: Decimal,
        provider_cost: Decimal,
        response_sha256: str,
    ) -> None:
        self.reserve(stage, local_worst_case)
        if (
            provider_cost < 0
            or not re.fullmatch(r"[0-9a-f]{64}", response_sha256)
            or not request_id
            or any(entry["request_id"] == request_id for entry in self.entries)
        ):
            raise CorpusError("invalid provider cost")
        debit = max(local_worst_case, provider_cost)
        entry = {
            "stage": stage,
            "reservation_id": "record-" + request_id,
            "request_id": request_id,
            "local_worst_case_usd": str(local_worst_case),
            "provider_cost_usd": str(provider_cost),
            "debit_usd": str(debit),
            "response_sha256": response_sha256,
            "status": "settled",
        }
        self.entries.append(entry)
        if (
            self.spent(stage) > STAGE_LIMITS[stage]
            or sum((Decimal(item["debit_usd"]) for item in self.entries), Decimal()) > TOTAL_BUDGET
        ):
            entry["status"] = "overspent"
            raise CorpusError("reported budget exhausted")

    def record_provider_journal(
        self,
        *,
        stage: Literal["corpus_author", "corpus_reviewer"],
        reservation_id: str,
        task_ids: Sequence[str],
    ) -> None:
        """Atomically bind a settled response to its preplanned task identities.

        This record deliberately contains no prompt, response payload, headers,
        or credential. Its response digest is the one already settled in the
        cost ledger, so the two durable receipts cannot be mixed later.
        """

        entry = next(
            (item for item in self.entries if item["reservation_id"] == reservation_id), None
        )
        if (
            entry is None
            or entry["stage"] != stage
            or entry["status"] not in {"settled", "overspent"}
            or not task_ids
            or len(task_ids) != len(set(task_ids))
            or not all(re.fullmatch(r"task-[0-9a-f]{64}", task_id) for task_id in task_ids)
            or any(item["reservation_id"] == reservation_id for item in self.provider_journal)
        ):
            raise CorpusError("provider journal")
        self.provider_journal.append(
            {
                "stage": stage,
                "reservation_id": reservation_id,
                "request_id": entry["request_id"],
                "task_ids": sorted(task_ids),
                "response_sha256": entry["response_sha256"],
            }
        )

    def require_complete_provider_journal(self) -> None:
        """Fail closed if a settled paid call has no immutable task correlation."""

        journal_ids = {entry["reservation_id"] for entry in self.provider_journal}
        settled_ids = {
            entry["reservation_id"]
            for entry in self.entries
            if entry["status"] in {"settled", "overspent"}
        }
        if journal_ids != settled_ids:
            raise CorpusError("settled provider call lacks durable journal")

    def as_json(self, *, final: bool = False, stop_reason: str | None = None) -> dict[str, Any]:
        return {
            "schema_version": "phase4e-cost-ledger.v2",
            "final": final,
            "entries": self.entries,
            "provider_journal": self.provider_journal,
            "stop_reason": stop_reason,
        }

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> BudgetLedger:
        if set(value) != {"schema_version", "final", "entries", "provider_journal", "stop_reason"}:
            raise CorpusError("ledger shape")
        if (
            value["schema_version"] != "phase4e-cost-ledger.v2"
            or not isinstance(value["entries"], list)
            or not isinstance(value["provider_journal"], list)
        ):
            raise CorpusError("ledger identity")
        ledger = cls()
        required_entry = {
            "stage",
            "reservation_id",
            "request_id",
            "local_worst_case_usd",
            "provider_cost_usd",
            "debit_usd",
            "response_sha256",
            "status",
        }
        for raw_entry in value["entries"]:
            if not isinstance(raw_entry, Mapping) or set(raw_entry) != required_entry:
                raise CorpusError("ledger entry")
            entry = {key: raw_entry[key] for key in required_entry}
            if not all(isinstance(item, str) for item in entry.values()):
                raise CorpusError("ledger entry")
            try:
                local_worst_case = Decimal(cast(str, entry["local_worst_case_usd"]))
                provider_cost = Decimal(cast(str, entry["provider_cost_usd"]))
                debit = Decimal(cast(str, entry["debit_usd"]))
            except ArithmeticError as error:
                raise CorpusError("ledger entry") from error
            if (
                entry["stage"] not in STAGE_LIMITS
                or local_worst_case < 0
                or provider_cost < 0
                or debit != max(local_worst_case, provider_cost)
                or entry["status"] not in {"reserved", "settled", "overspent"}
                or not re.fullmatch(r"[0-9a-f]{64}", entry["response_sha256"])
            ):
                raise CorpusError("ledger entry")
            if entry["status"] == "reserved":
                if (
                    entry["request_id"] != entry["reservation_id"]
                    or provider_cost != 0
                    or entry["response_sha256"] != "0" * 64
                    or not re.fullmatch(r"reservation-[0-9a-f]{64}", entry["reservation_id"])
                ):
                    raise CorpusError("ledger reservation")
            elif not entry["request_id"]:
                raise CorpusError("ledger entry")
            if any(item["reservation_id"] == entry["reservation_id"] for item in ledger.entries):
                raise CorpusError("ledger entry")
            ledger.entries.append(cast(dict[str, str], entry))
            over_budget = (
                ledger.spent(entry["stage"]) > STAGE_LIMITS[entry["stage"]]
                or sum((Decimal(item["debit_usd"]) for item in ledger.entries), Decimal())
                > TOTAL_BUDGET
            )
            if over_budget != (entry["status"] == "overspent"):
                raise CorpusError("ledger budget")
        journal_entry_keys = {
            "stage",
            "reservation_id",
            "request_id",
            "task_ids",
            "response_sha256",
        }
        for raw_journal in value["provider_journal"]:
            if not isinstance(raw_journal, Mapping) or set(raw_journal) != journal_entry_keys:
                raise CorpusError("provider journal")
            stage = raw_journal["stage"]
            reservation_id = raw_journal["reservation_id"]
            request_id = raw_journal["request_id"]
            task_ids = raw_journal["task_ids"]
            response_sha256 = raw_journal["response_sha256"]
            matching = next(
                (entry for entry in ledger.entries if entry["reservation_id"] == reservation_id),
                None,
            )
            if (
                stage not in {"corpus_author", "corpus_reviewer"}
                or not isinstance(reservation_id, str)
                or not isinstance(request_id, str)
                or not isinstance(response_sha256, str)
                or not isinstance(task_ids, list)
                or task_ids != sorted(task_ids)
                or len(task_ids) != len(set(task_ids))
                or not task_ids
                or not all(
                    isinstance(task_id, str) and re.fullmatch(r"task-[0-9a-f]{64}", task_id)
                    for task_id in task_ids
                )
                or matching is None
                or matching["stage"] != stage
                or matching["status"] not in {"settled", "overspent"}
                or matching["request_id"] != request_id
                or matching["response_sha256"] != response_sha256
                or any(item["reservation_id"] == reservation_id for item in ledger.provider_journal)
            ):
                raise CorpusError("provider journal")
            ledger.provider_journal.append(cast(dict[str, Any], raw_journal))
        return ledger


def write_ledger_snapshot(
    directory: Path, ledger: BudgetLedger, *, stop_reason: str | None = None
) -> Path:
    """Append one immutable resume point; an existing sequence number is never overwritten."""
    if directory.exists() and not directory.is_dir():
        raise CorpusError("ledger directory")
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    snapshots = sorted(directory.glob("ledger-*.json"))
    if snapshots:
        previous = json.loads(snapshots[-1].read_bytes())
        previous_ledger = BudgetLedger.from_json(previous)
        if len(previous_ledger.entries) > len(ledger.entries):
            raise CorpusError("ledger resume mismatch")
        for earlier, current in zip(
            previous_ledger.entries, ledger.entries[: len(previous_ledger.entries)], strict=True
        ):
            if earlier == current:
                continue
            mutable = {
                "request_id",
                "provider_cost_usd",
                "debit_usd",
                "response_sha256",
                "status",
            }
            if (
                earlier["status"] != "reserved"
                or current["status"] not in {"settled", "overspent"}
                or any(earlier[key] != current[key] for key in set(earlier) - mutable)
                or Decimal(current["debit_usd"]) < Decimal(earlier["debit_usd"])
            ):
                raise CorpusError("ledger resume mismatch")
        if (
            previous_ledger.provider_journal
            != ledger.provider_journal[: len(previous_ledger.provider_journal)]
        ):
            raise CorpusError("provider journal resume mismatch")
    path = directory / f"ledger-{len(snapshots):04d}.json"
    atomic_create(path, _canonical(ledger.as_json(stop_reason=stop_reason)) + b"\n")
    os.chmod(path, 0o600)
    return path


def resume_ledger(directory: Path) -> BudgetLedger:
    """Load the newest immutable ledger state; reservations remain charged."""
    if not directory.exists():
        return BudgetLedger()
    if not directory.is_dir():
        raise CorpusError("ledger directory")
    snapshots = sorted(directory.glob("ledger-*.json"))
    if not snapshots:
        return BudgetLedger()
    try:
        return BudgetLedger.from_json(json.loads(snapshots[-1].read_bytes()))
    except (OSError, ValueError) as error:
        raise CorpusError("ledger resume") from error


def request_worst_case(
    stage: Literal["corpus_author", "corpus_reviewer"],
    request_payload: bytes,
    max_output_tokens: int,
) -> Decimal:
    if stage not in PRICES or not request_payload or max_output_tokens < 0:
        raise CorpusError("budget request")
    input_price, output_price = PRICES[stage]
    # Reserve against the complete, canonical final provider request: messages,
    # strict schema, provider controls, and every other input field are included.
    return (
        Decimal(len(request_payload) + 256) * input_price
        + Decimal(max_output_tokens) * output_price
    ) / Decimal(1_000_000)


def wilson_lower_bound(successes: int, total: int) -> float:
    if total <= 0 or successes < 0 or successes > total:
        raise CorpusError("Wilson inputs")
    z = 1.6448536269514722  # one-sided 95%
    p = successes / total
    denominator = 1 + z * z / total
    return (
        p + z * z / (2 * total) - z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total)
    ) / denominator


def stop_projection(resolved: Sequence[Mapping[str, Any]], plan: Mapping[str, Any]) -> str | None:
    """Project every required cell at ten-batch boundaries, without relaxing a floor."""
    by_task = {cast(str, row["task_id"]): row for row in resolved}
    slots = cast(list[Mapping[str, Any]], plan["slots"])

    def project(planned: Sequence[Mapping[str, Any]], minimum: int) -> str | None:
        done = [
            by_task[cast(str, slot["task_id"])] for slot in planned if slot["task_id"] in by_task
        ]
        accepted = sum(row.get("status") == "accepted" for row in done)
        unresolved = len(planned) - len(done)
        if accepted + unresolved < minimum:
            return "impossible_required_minimum"
        # The deterministic capacity check above is always authoritative. A
        # Wilson projection needs a minimally informative cohort, however:
        # smaller cohorts record their descriptive rate but cannot stop work.
        if (
            len(done) >= 20
            and accepted + math.floor(unresolved * wilson_lower_bound(accepted, len(done)))
            < minimum
        ):
            return "wilson_projected_minimum"
        return None

    if stop := project(slots, 1200):
        return stop
    for split, minimum in (
        ("synthetic_train", 840),
        ("synthetic_dev", 180),
        ("synthetic_holdout", 180),
    ):
        if stop := project([slot for slot in slots if slot["split"] == split], minimum):
            return stop
    for split in SPLITS:
        for locale in LOCALES:
            for option_count in OPTION_COUNTS:
                planned = [
                    slot
                    for slot in slots
                    if slot["split"] == split
                    and slot["locale"] == locale
                    and slot["option_count"] == option_count
                ]
                minimum = 20 if split == "synthetic_train" else 10
                if stop := project(planned, minimum):
                    return stop
    return None


class OpenRouterCorpusClient:
    """Injected transport boundary. It has no environment or credential lookup."""

    def __init__(
        self,
        *,
        transport: Transport | None = None,
        allow_network: bool = False,
        ledger: BudgetLedger | None = None,
        ledger_directory: Path | None = None,
    ) -> None:
        if not allow_network or transport is None:
            raise CorpusError("network requires explicit injected transport")
        if ledger_directory is None:
            raise CorpusError("network requires durable ledger")
        self._transport = transport
        self._ledger_directory = ledger_directory
        self._last_journal: dict[str, Any] | None = None
        resumed = resume_ledger(ledger_directory)
        if (
            ledger is not None
            and ledger.entries
            and resumed.entries
            and ledger.entries != resumed.entries
        ):
            raise CorpusError("ledger resume mismatch")
        self.ledger = resumed if resumed.entries else ledger or resumed

    def author(
        self,
        *,
        slots: Sequence[Mapping[str, Any]],
        max_output_tokens: int,
        api_key: str,
    ) -> dict[str, Any]:
        """Submit planned author slots using the fixed author protocol."""
        _validate_author_batch(slots)
        return self._request(
            stage="corpus_author",
            model=AUTHOR_MODEL,
            messages=author_messages(slots),
            response_schema=author_schema(slots),
            max_output_tokens=max_output_tokens,
            api_key=api_key,
            task_ids=[cast(str, slot["task_id"]) for slot in slots],
        )

    def reviewer(
        self,
        *,
        row: ValidatedAuthorRow,
        max_output_tokens: int,
        api_key: str,
    ) -> dict[str, Any]:
        """Submit exactly one locally validated row through the blind protocol."""
        return self._request(
            stage="corpus_reviewer",
            model=REVIEWER_MODEL,
            messages=reviewer_messages([row]),
            response_schema=reviewer_schema(1),
            max_output_tokens=max_output_tokens,
            api_key=api_key,
            task_ids=[row.task_id],
        )

    def last_journal(self, stage: Literal["corpus_author", "corpus_reviewer"]) -> dict[str, Any]:
        """Return the durable journal record produced by the immediately prior request."""

        if self._last_journal is None or self._last_journal["stage"] != stage:
            raise CorpusError("provider journal unavailable")
        return dict(self._last_journal)

    def _request(
        self,
        *,
        stage: Literal["corpus_author", "corpus_reviewer"],
        model: str,
        messages: Sequence[Mapping[str, str]],
        response_schema: Mapping[str, Any],
        max_output_tokens: int,
        api_key: str,
        task_ids: Sequence[str],
    ) -> dict[str, Any]:
        if not api_key:
            raise CorpusError("pinned provider request")
        request: dict[str, Any] = {
            "model": model,
            "messages": list(messages),
            "max_tokens": max_output_tokens,
            "temperature": 0,
            "provider": provider_preferences(stage),
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": RESPONSE_SCHEMA_NAME,
                    "strict": True,
                    "schema": response_schema,
                },
            },
        }
        if stage == "corpus_author" and model == AUTHOR_MODEL:
            request["reasoning_effort"] = "none"
        body = _canonical(request)
        worst = request_worst_case(stage, body, max_output_tokens)
        reservation_id = "reservation-" + _sha(_canonical({"stage": stage, "body": body.hex()}))
        self.ledger.reserve_request(stage, reservation_id, worst)
        write_ledger_snapshot(self._ledger_directory, self.ledger)
        result = self._transport(
            "POST",
            "https://openrouter.ai/api/v1/chat/completions",
            {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            body,
        )
        try:
            status, _, raw = cast(tuple[int, Mapping[str, str], bytes], result)
            response = cast(dict[str, Any], json.loads(raw))
            response_error = response.get("error")
            if isinstance(response_error, dict):
                error_message = " ".join(str(response_error.get("message", "")).split())[:240]
                metadata = response_error.get("metadata")
                if error_message == "Provider returned error" and isinstance(metadata, dict):
                    error_message = " ".join(str(metadata.get("raw", "")).split())[:240]
                raise CorpusError(
                    f"provider response error ({response_error.get('code', 'unknown')}): "
                    f"{error_message or 'no message'}"
                )
            cost = Decimal(str(response["usage"]["cost"]))
            request_id = cast(str, response["id"])
        except CorpusError:
            raise
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            if "status" in locals() and status != 200:
                try:
                    error_payload = json.loads(raw)
                    error_code = error_payload.get("error", {}).get("code", "unknown")
                except (AttributeError, TypeError, ValueError, json.JSONDecodeError):
                    error_code = "invalid"
                raise CorpusError(f"provider HTTP {status} ({error_code})") from error
            raise CorpusError("provider response cost") from error
        try:
            self.ledger.settle_request(reservation_id, request_id, cost, _sha(raw))
        except CorpusError:
            write_ledger_snapshot(
                self._ledger_directory, self.ledger, stop_reason="reported_provider_overspend"
            )
            self.ledger.record_provider_journal(
                stage=stage, reservation_id=reservation_id, task_ids=task_ids
            )
            self._last_journal = dict(self.ledger.provider_journal[-1])
            write_ledger_snapshot(
                self._ledger_directory, self.ledger, stop_reason="reported_provider_overspend"
            )
            raise
        write_ledger_snapshot(self._ledger_directory, self.ledger)
        self.ledger.record_provider_journal(
            stage=stage, reservation_id=reservation_id, task_ids=task_ids
        )
        write_ledger_snapshot(self._ledger_directory, self.ledger)
        self._last_journal = dict(self.ledger.provider_journal[-1])
        if status != 200:
            response_error = response.get("error")
            error_code = (
                response_error.get("code", "unknown")
                if isinstance(response_error, dict)
                else "unknown"
            )
            raise CorpusError(f"provider HTTP {status} ({error_code})")
        return response
