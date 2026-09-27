"""Phase 4E.4 blind comparison — checkout-only CLI, offline-first.

Subcommands: plan, generate, score-backend, evaluate, publish, publish-status.
All live provider calls require literal ``--allow-network`` and an injected transport.
Default import, plan, and publish sanitizer never touch the network or ML.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import math
import os
import platform
import re
import resource
import stat
import subprocess
import sys
import tempfile
import time
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal, Protocol, cast

from benchmarks import saracura_universal_corpus as corpus
from benchmarks.io import atomic_create
from benchmarks.saracura_universal_corpus import (
    AXES as AXES,
)
from benchmarks.saracura_universal_corpus import (
    DOMAINS as DOMAINS,
)
from benchmarks.saracura_universal_corpus import (
    LOCALES as LOCALES,
)
from benchmarks.saracura_universal_corpus import (
    OPTION_COUNTS as OPTION_COUNTS,
)
from benchmarks.saracura_universal_corpus import (
    SCENARIO_CODES,
)
from benchmarks.saracura_universal_policy import (
    POST_PILOT_AUTHOR_MODEL,
    POST_PILOT_REVIEWER_MODEL,
)
from saracura.contracts.models import ChoiceCriterion
from saracura.universal.rendering import (
    render_choice,
    validate_rendered_capacity,
)
from saracura.universal.tasks import SARACURA_UNIVERSAL_WORKFLOW_REVISION

COMPARISON_WORKFLOW_ID = "universal-choice"
COMPARISON_SARACURA_REVISION = SARACURA_UNIVERSAL_WORKFLOW_REVISION
COMPARISON_LAYA_REVISION = "phase4d-laya.v1"
COMPARISON_POLICY_PATH = Path(__file__).parent / "manifests" / "phase4e-comparison-policy.v1.json"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
README_EN_PATH = REPOSITORY_ROOT / "README.md"
README_PT_BR_PATH = REPOSITORY_ROOT / "docs" / "README.pt-BR.md"
PUBLIC_RESULT_JSON_PATH = REPOSITORY_ROOT / "benchmarks" / "results" / "phase4e-comparison-v1.json"
PUBLIC_RESULT_MD_PATH = REPOSITORY_ROOT / "docs" / "action" / "phase4e-comparison-result.md"
COMPARISON_PLAN_SCHEMA = "phase4e-comparison-plan.v1"
COMPARISON_PLANNER_REVISION = "phase4e-comparison-planner.v1"
COMPARISON_NAMESPACE_PREFIX = "task-"
COMPARISON_FAMILY_PREFIX = "family-"
COMPARISON_UNDERPOWERED_THRESHOLD = 10

COMPARISON_TASK_STATUSES = frozenset(
    {
        "correct",
        "incorrect",
        "unsupported_capacity",
        "unsupported_runtime",
        "runtime_error",
    }
)

COMPARISON_TERMINAL_STATES = frozenset(
    {
        "passed",
        "insufficient_comparison_evidence",
        "inconclusive_transport",
        "inconclusive_operational",
    }
)

__all__ = [
    "AXES",
    "COMPARISON_TASK_STATUSES",
    "COMPARISON_TERMINAL_STATES",
    "DOMAINS",
    "ComparisonError",
    "build_comparison_plan",
    "validate_comparison_policy",
]

AUTHOR_STAGE = "comparison_author"
REVIEWER_STAGE = "comparison_reviewer"
COMPARISON_ACCEPTED_SCHEMA = "phase4e-comparison-accepted.v1"
COMPARISON_RESOLUTION_SCHEMA = "phase4e-comparison-resolution.v1"
COMPARISON_TERMINAL_REPORT_SCHEMA = "phase4e-comparison-terminal-report.v1"
COMPARISON_EVALUATION_RECEIPT_SCHEMA = "phase4e-comparison-evaluation-receipt.v1"
COMPARISON_WORKER_RECEIPT_SCHEMA = "phase4e-comparison-worker-receipt.v1"
COMPARISON_GENERATION_BINDING_SCHEMA = "phase4e-comparison-generation-binding.v1"
_PERSISTED_DIAGNOSTIC_KEYS = (
    "author_response_failures",
    "reviewer_response_failures",
    "retry_recoveries",
    "transport_uncertain_calls",
)
_AUTHOR_LINEAGE_KEYS = ("author_response_sha256", "author_reservation_id", "author_request_id")
_REVIEWER_LINEAGE_KEYS = (
    "reviewer_response_sha256",
    "reviewer_reservation_id",
    "reviewer_request_id",
)
_COMPARISON_AUTHOR_MODEL = POST_PILOT_AUTHOR_MODEL
_COMPARISON_REVIEWER_MODEL = POST_PILOT_REVIEWER_MODEL
_COMPARISON_PROVIDER_POLICY = {
    "order": ["Azure"],
    "allow_fallbacks": False,
    "require_parameters": True,
    "data_collection": "deny",
    "zdr": True,
}
_AUTHOR_MAX_TOKENS = 1024
_REVIEWER_MAX_TOKENS = 512
_TRANSPORT_TIMEOUT = 120

_PRIVATE_STATE_ROOT_ENV = "SARACURA_PRIVATE_STATE_ROOT"

COMPARISON_LEDGER_POLICY = corpus.COMPARISON_LEDGER_POLICY


class ComparisonError(ValueError):
    """A closed comparison invariant was violated."""


class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...


class Transport(Protocol):
    def __call__(
        self, method: str, url: str, headers: Mapping[str, str], body: bytes
    ) -> tuple[int, Mapping[str, str], bytes]: ...


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in pairs:
        if key in value:
            raise ValueError("duplicate key")
        value[key] = child
    return value


def _canonical(value: object) -> bytes:
    return corpus._canonical(value)


def _sha(value: bytes) -> str:
    return corpus._sha(value)


def _seeded(seed: str, *parts: object) -> int:
    return int(_sha("\0".join((seed, *(str(part) for part in parts))).encode())[:16], 16)


# ── policy ───────────────────────────────────────────────────────────────


def _load_policy(path: Path = COMPARISON_POLICY_PATH) -> dict[str, Any]:
    try:
        policy: dict[str, Any] = json.loads(path.read_bytes(), object_pairs_hook=_no_duplicate_keys)
    except (OSError, ValueError) as error:
        raise ComparisonError("comparison policy is not valid JSON") from error
    _validate_policy_shape(policy)
    return policy


def validate_comparison_policy(path: Path = COMPARISON_POLICY_PATH) -> dict[str, Any]:
    return _load_policy(path)


def _validate_policy_shape(policy: dict[str, Any]) -> None:
    expected = {
        "schema_version",
        "id",
        "plan",
        "accepted_task_id_namespace",
        "family_id_namespace",
        "acceptance_minimums",
        "provider",
        "cost",
        "execution",
        "underpowered_threshold",
    }
    if set(policy) != expected:
        raise ComparisonError("comparison policy shape is closed")
    if (
        policy["schema_version"] != "phase4e-comparison-policy.v1"
        or policy["id"] != "phase4e-comparison"
    ):
        raise ComparisonError("comparison policy identity")
    if (
        policy["accepted_task_id_namespace"] != "task:sha256(saracura-phase4e-comparison-v1)"
        or policy["family_id_namespace"] != "family:sha256(saracura-phase4e-comparison-v1)"
    ):
        raise ComparisonError("comparison identity namespace")
    plan = policy["plan"]
    if not isinstance(plan, dict):
        raise ComparisonError("comparison plan key missing")
    if (
        plan.get("seed") != "saracura-phase4e-comparison-v1"
        or plan.get("task_slots") != 200
        or plan.get("locales") != list(LOCALES)
        or plan.get("domains") != list(DOMAINS)
        or plan.get("cross_locale_pairs") != 100
        or plan.get("option_counts") != list(OPTION_COUNTS)
    ):
        raise ComparisonError("comparison plan metadata invalid")
    if plan.get("axes") != {
        "explicitness": ["explicit", "implicit"],
        "negation": ["absent", "present"],
        "distractor_overlap": ["low", "high"],
        "urgency": ["normal", "urgent"],
    }:
        raise ComparisonError("comparison axes invalid")
    mins = policy["acceptance_minimums"]
    if mins != {
        "total_accepted": 150,
        "minimum_per_locale": 60,
        "minimum_per_domain": 5,
        "minimum_per_option_count": 10,
        "minimum_complete_pairs": 60,
    }:
        raise ComparisonError("comparison acceptance minimums invalid")
    provider = policy["provider"]
    if (
        provider.get("host") != "openrouter.ai"
        or provider.get("author_model") != POST_PILOT_AUTHOR_MODEL
        or provider.get("reviewer_model") != POST_PILOT_REVIEWER_MODEL
    ):
        raise ComparisonError("comparison provider policy invalid")
    pp = provider.get("provider_policy")
    if not isinstance(pp, dict) or pp != _COMPARISON_PROVIDER_POLICY:
        raise ComparisonError("comparison provider policy invalid")
    if provider.get("transport_timeout_seconds") != 120:
        raise ComparisonError("comparison transport timeout invalid")
    if provider.get("author_request") != {
        "max_output_tokens": 1024,
        "temperature": 0,
        "reasoning": None,
    }:
        raise ComparisonError("comparison author request policy invalid")
    if provider.get("reviewer_request") != {
        "max_output_tokens": 512,
        "temperature": 0,
        "reasoning": None,
    }:
        raise ComparisonError("comparison reviewer request policy invalid")
    if (
        policy.get("cost")
        != {"mode": "report_only", "report_interval_usd": "10.00", "automatic_retries": 4}
        or policy.get("underpowered_threshold") != 10
    ):
        raise ComparisonError("comparison cost or underpowered threshold invalid")
    if policy.get("execution") != {
        "maximum_provider_attempts": 5,
        "transport_uncertainty_circuit": 3,
        "primary_device": "cpu",
        "cpu_threads": 1,
        "capacity_intersection": {"context_tokens": 128, "criterion_tokens": 96},
        "schemas": {
            "plan": COMPARISON_PLAN_SCHEMA,
            "packet": "phase4e-comparison-packet.v1",
            "worker_input": "phase4e-comparison-worker-input.v1",
            "worker_result": "phase4e-comparison-score-backend.v1",
            "result": "phase4e-comparison-result.v1",
        },
    }:
        raise ComparisonError("comparison execution policy invalid")


# ── plan ──────────────────────────────────────────────────────────────────


def _semantic_target(
    seed: str,
    pair_index: int,
    option_count: int,
    scenario: str,
) -> dict[str, Any]:
    distractors = sorted(
        corpus.CRITERION_ROLES[1:],
        key=lambda role: _seeded(seed, "role", str(pair_index), role),
    )[: option_count - 1]
    return {
        "scenario": scenario,
        "criterion_roles": ["matches_rule", *distractors],
    }


def _fixture_plan_bindings() -> dict[str, Any]:
    """Deterministic offline-only lineage used by tests with injected transports."""

    return {
        "mode": "offline_fixture",
        "training_packet_receipt_sha256": "1" * 64,
        "training_accepted_rows_sha256": "2" * 64,
        "training_manifest_sha256": "3" * 64,
        "training_checkpoint_sha256": "4" * 64,
        "source_commit": "5" * 40,
        "saracura_tokenizer_snapshot_sha256": "6" * 64,
        "laya_snapshot_sha256": "7" * 64,
        "training_task_ids_sha256": "8" * 64,
        "training_family_ids_sha256": "9" * 64,
        "identities_disjoint": True,
    }


def build_comparison_plan(*, bindings: Mapping[str, Any] | None = None) -> dict[str, Any]:
    policy = _load_policy()
    seed = cast(str, policy["plan"]["seed"])
    slots: list[dict[str, Any]] = []
    serial = 0
    option_seen: Counter[int] = Counter()
    for pair_index in range(100):
        option_count = OPTION_COUNTS[pair_index % len(OPTION_COUNTS)]
        domain = DOMAINS[pair_index % len(DOMAINS)]
        within_option_count = option_seen[option_count]
        option_seen[option_count] += 1
        axis_permutations = ((1, 0), (3, 17), (7, 31), (9, 47))
        axes = {
            key: values[
                0
                if (
                    pair_index * axis_permutations[axis_index][0] + axis_permutations[axis_index][1]
                )
                % 100
                < 50
                else 1
            ]
            for axis_index, (key, values) in enumerate(AXES.items())
        }
        gold_position = within_option_count % option_count
        pt_task_id = COMPARISON_NAMESPACE_PREFIX + _sha(f"{seed}\0task\0pt\0{serial}".encode())
        en_task_id = COMPARISON_NAMESPACE_PREFIX + _sha(f"{seed}\0task\0en\0{serial + 1}".encode())
        family_id = COMPARISON_FAMILY_PREFIX + _sha(f"{seed}\0family\0{pair_index}".encode())
        target = _semantic_target(
            seed,
            pair_index,
            option_count,
            SCENARIO_CODES[pair_index % len(SCENARIO_CODES)],
        )
        for locale in ("pt-BR", "en"):
            task_id = pt_task_id if locale == "pt-BR" else en_task_id
            slots.append(
                {
                    "slot": serial,
                    "task_id": task_id,
                    "family_id": family_id,
                    "pair_id": family_id,
                    "split": "synthetic_comparison",
                    "locale": locale,
                    "domain": domain,
                    "axes": dict(axes),
                    "option_count": option_count,
                    "gold_position": gold_position,
                    "semantic_target": dict(target),
                }
            )
            serial += 1
    sorted_slots = sorted(slots, key=lambda s: (s["locale"], s["option_count"]))
    result: list[dict[str, Any]] = []
    i = 0
    while i < len(sorted_slots):
        j = i
        while (
            j < len(sorted_slots)
            and sorted_slots[j]["locale"] == sorted_slots[i]["locale"]
            and sorted_slots[j]["option_count"] == sorted_slots[i]["option_count"]
        ):
            j += 1
        cell = sorted_slots[i:j]
        cell.sort(key=lambda s: _seeded(seed, "gold-shuffle", cast(str, s["task_id"])))
        result.extend(cell)
        i = j
    return {
        "schema_version": COMPARISON_PLAN_SCHEMA,
        "workflow_revision": SARACURA_UNIVERSAL_WORKFLOW_REVISION,
        "planner_revision": COMPARISON_PLANNER_REVISION,
        "seed": seed,
        "policy_sha256": _sha(COMPARISON_POLICY_PATH.read_bytes()),
        "execution": policy["execution"],
        "bindings": dict(bindings) if bindings is not None else _fixture_plan_bindings(),
        "slots": result,
    }


def is_comparison_plan_immutable(original: object, candidate: object) -> bool:
    return _canonical(original) == _canonical(candidate)


# ── plan validation ─────────────────────────────────────────────────────


def _validate_comparison_slot(slot: dict[str, Any]) -> None:
    required = {
        "slot",
        "task_id",
        "family_id",
        "split",
        "locale",
        "domain",
        "axes",
        "option_count",
        "gold_position",
        "semantic_target",
    }
    allowed_extra = {"pair_id"}
    if set(slot) - required - allowed_extra:
        raise ComparisonError("slot shape is not closed")
    if required - set(slot):
        raise ComparisonError("slot shape missing required fields")
    if not isinstance(slot["task_id"], str) or not slot["task_id"].startswith(
        COMPARISON_NAMESPACE_PREFIX
    ):
        raise ComparisonError("task ID namespace")
    if not isinstance(slot["family_id"], str) or not slot["family_id"].startswith(
        COMPARISON_FAMILY_PREFIX
    ):
        raise ComparisonError("family ID namespace")
    if slot["split"] != "synthetic_comparison":
        raise ComparisonError("comparison split")
    if slot["pair_id"] is not None and not isinstance(slot["pair_id"], str):
        raise ComparisonError("pair ID type")
    if slot["locale"] not in LOCALES:
        raise ComparisonError("locale")
    if slot["domain"] not in DOMAINS:
        raise ComparisonError("domain")
    if not isinstance(slot["axes"], dict) or set(slot["axes"]) != set(AXES):
        raise ComparisonError("axes shape")
    for key, values in AXES.items():
        if slot["axes"].get(key) not in values:
            raise ComparisonError(f"axis value: {key}")
    option_count = int(slot["option_count"])
    gold = int(slot["gold_position"])
    if not 2 <= option_count <= 8:
        raise ComparisonError("option count")
    if not 0 <= gold < option_count:
        raise ComparisonError("gold position")
    target = slot.get("semantic_target")
    if not isinstance(target, dict) or set(target) != {"scenario", "criterion_roles"}:
        raise ComparisonError("semantic target shape")
    if target["scenario"] not in SCENARIO_CODES:
        raise ComparisonError("scenario")
    roles = target["criterion_roles"]
    if not isinstance(roles, list) or len(roles) != option_count:
        raise ComparisonError("criterion roles count")
    if roles[0] != "matches_rule":
        raise ComparisonError("first role must be matches_rule")
    for role in roles:
        if role not in corpus.CRITERION_ROLES:
            raise ComparisonError("unknown criterion role")


def _validate_comparison_balance(slots: list[dict[str, Any]]) -> None:
    for locale in LOCALES:
        locale_slots = [s for s in slots if s["locale"] == locale]
        if len(locale_slots) != 100:
            raise ComparisonError(f"locale count: {locale}")
    domain_counts = Counter(cast(str, s["domain"]) for s in slots)
    if max(domain_counts.values()) - min(domain_counts.values()) > 2:
        raise ComparisonError("domain allocation is not integer-balanced")
    option_counts = Counter(cast(int, s["option_count"]) for s in slots)
    if (
        set(option_counts) != set(OPTION_COUNTS)
        or max(option_counts.values()) - min(option_counts.values()) > 2
    ):
        raise ComparisonError("option-count allocation is not integer-balanced")
    scenario_counts = Counter(cast(str, s["semantic_target"]["scenario"]) for s in slots)
    if (
        set(scenario_counts) != set(SCENARIO_CODES)
        or max(scenario_counts.values()) - min(scenario_counts.values()) > 2
    ):
        raise ComparisonError("scenario allocation is not integer-balanced")
    for axis, values in AXES.items():
        axis_counts = Counter(cast(str, s["axes"][axis]) for s in slots)
        if (
            set(axis_counts) != set(values)
            or max(axis_counts.values()) - min(axis_counts.values()) > 0
        ):
            raise ComparisonError(f"axis allocation is not integer-balanced: {axis}")
    for locale in LOCALES:
        for count in OPTION_COUNTS:
            cell_slots = [s for s in slots if s["locale"] == locale and s["option_count"] == count]
            if not cell_slots:
                raise ComparisonError(f"missing cell: {locale}/{count}")
            gold_counts = Counter(cast(int, s["gold_position"]) for s in cell_slots)
            if (
                set(gold_counts) != set(range(count))
                or max(gold_counts.values()) - min(gold_counts.values()) > 1
            ):
                raise ComparisonError(f"gold allocation is not integer-balanced: {locale}/{count}")


def _validate_comparison_plan(plan: dict[str, Any]) -> None:
    if not isinstance(plan, dict):
        raise ComparisonError("plan must be a JSON object")
    if set(plan) != {
        "schema_version",
        "workflow_revision",
        "planner_revision",
        "seed",
        "policy_sha256",
        "execution",
        "bindings",
        "slots",
    }:
        raise ComparisonError("plan shape is closed")
    if plan.get("schema_version") != COMPARISON_PLAN_SCHEMA:
        raise ComparisonError("plan schema mismatch")
    if plan.get("seed") != "saracura-phase4e-comparison-v1":
        raise ComparisonError("plan seed mismatch")
    if plan.get("planner_revision") != COMPARISON_PLANNER_REVISION:
        raise ComparisonError("plan planner revision")
    policy = _load_policy()
    if (
        plan.get("workflow_revision") != SARACURA_UNIVERSAL_WORKFLOW_REVISION
        or plan.get("policy_sha256") != _sha(COMPARISON_POLICY_PATH.read_bytes())
        or plan.get("execution") != policy["execution"]
    ):
        raise ComparisonError("plan protocol binding")
    bindings = plan.get("bindings")
    binding_fields = {
        "mode",
        "training_packet_receipt_sha256",
        "training_accepted_rows_sha256",
        "training_manifest_sha256",
        "training_checkpoint_sha256",
        "source_commit",
        "saracura_tokenizer_snapshot_sha256",
        "laya_snapshot_sha256",
        "training_task_ids_sha256",
        "training_family_ids_sha256",
        "identities_disjoint",
    }
    if (
        not isinstance(bindings, dict)
        or set(bindings) != binding_fields
        or bindings.get("mode") not in {"live_verified", "offline_fixture"}
        or bindings.get("identities_disjoint") is not True
        or not isinstance(bindings.get("source_commit"), str)
        or re.fullmatch(r"[0-9a-f]{40}", bindings["source_commit"]) is None
    ):
        raise ComparisonError("plan evidence bindings")
    digest_fields = binding_fields - {"mode", "source_commit", "identities_disjoint"}
    if any(
        not isinstance(bindings.get(field), str)
        or len(bindings[field]) != 64
        or any(character not in "0123456789abcdef" for character in bindings[field])
        for field in digest_fields
    ):
        raise ComparisonError("plan evidence digest")
    slots = plan.get("slots")
    if not isinstance(slots, list) or len(slots) != 200:
        raise ComparisonError("plan must have exactly 200 slots")
    for slot in slots:
        _validate_comparison_slot(slot)
    _validate_comparison_balance(slots)
    task_ids = {cast(str, slot["task_id"]) for slot in slots}
    if len(task_ids) != 200:
        raise ComparisonError("duplicate task IDs")
    pairs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for slot in slots:
        pid = slot["pair_id"]
        if isinstance(pid, str):
            pairs[pid].append(slot)
    if len(pairs) != 100:
        raise ComparisonError("expected 100 pairs")
    for pair_slots in pairs.values():
        if len(pair_slots) != 2:
            raise ComparisonError("each pair needs exactly 2 slots")
        if {s["locale"] for s in pair_slots} != {"pt-BR", "en"}:
            raise ComparisonError("pair must have pt-BR and en")
        if pair_slots[0]["option_count"] != pair_slots[1]["option_count"]:
            raise ComparisonError("pair option count mismatch")
        if pair_slots[0]["gold_position"] != pair_slots[1]["gold_position"]:
            raise ComparisonError("pair gold position mismatch")
        if pair_slots[0]["semantic_target"] != pair_slots[1]["semantic_target"]:
            raise ComparisonError("pair semantic target mismatch")
        if pair_slots[0]["axes"] != pair_slots[1]["axes"]:
            raise ComparisonError("pair axes mismatch")
    text_keys = {"instruction", "state", "criteria", "selected_criterion_id"}
    for slot in slots:
        if text_keys.intersection(slot):
            raise ComparisonError("plan contains natural-language content")
    expected = build_comparison_plan(bindings=cast(Mapping[str, Any], bindings))
    if _canonical(plan) != _canonical(expected):
        raise ComparisonError("plan is not the canonical policy-derived plan")


# ── private root / outside ───────────────────────────────────────────────


def _outside(child: Path, parent: Path) -> None:
    try:
        child.relative_to(parent)
        raise ComparisonError("private path is inside the artifact root")
    except ValueError:
        pass


def _require_within(child: Path, parent: Path) -> Path:
    resolved_child = child.resolve(strict=False)
    resolved_parent = parent.resolve()
    try:
        resolved_child.relative_to(resolved_parent)
    except ValueError as error:
        raise ComparisonError("comparison artifact must stay inside the private root") from error
    return resolved_child


def require_live_private_comparison_root() -> Path:
    candidate = os.environ.get(_PRIVATE_STATE_ROOT_ENV)
    if not candidate:
        raise ComparisonError("SARACURA_PRIVATE_STATE_ROOT is not set")
    root = Path(candidate).resolve()
    if not root.exists() or not root.is_dir():
        raise ComparisonError("private state root does not exist")
    if (root.stat().st_mode & 0o777) != 0o700:
        raise ComparisonError("private state root must be private (0700)")
    checkout = Path(__file__).resolve().parents[1]
    try:
        root.relative_to(checkout)
        raise ComparisonError("private state root is inside the checkout")
    except ValueError:
        pass
    return root


def _require_component(path: Path, component: str, *, file_path: bool = False) -> None:
    resolved = path.resolve()
    directory = resolved.parent if file_path else resolved
    if directory.name != component:
        raise ComparisonError("comparison artifact path")


def _require_public_destination(path: Path, parent_parts: tuple[str, ...], filename: str) -> None:
    resolved = path.resolve(strict=False)
    if (
        resolved.name != filename
        or tuple(resolved.parent.parts[-len(parent_parts) :]) != parent_parts
    ):
        raise ComparisonError("comparison public destination")


# ── semantics ────────────────────────────────────────────────────────────


def _semantic_fingerprint(row: Mapping[str, Any]) -> str:
    instr = str(row.get("instruction", ""))
    st = str(row.get("state", ""))
    descs = "|".join(
        c["description"] if isinstance(c, Mapping) else str(c)
        for c in cast(Sequence[Any], row.get("criteria", []))
    )
    return _sha(
        _canonical(
            {
                "locale": row.get("locale"),
                "instruction": instr,
                "state": st,
                "descriptions": descs.strip(),
            }
        )
    )


def _normalized(row: Mapping[str, Any]) -> str:
    instr = unicodedata.normalize("NFKD", str(row.get("instruction", ""))).casefold()
    sv = row.get("state")
    st = _canonical(sv).decode("utf-8", "replace") if isinstance(sv, dict) else str(sv)
    st = unicodedata.normalize("NFKD", st).casefold()
    descs = "|".join(
        unicodedata.normalize(
            "NFKD", c["description"] if isinstance(c, Mapping) else str(c)
        ).casefold()
        for c in cast(Sequence[Any], row.get("criteria", []))
    )
    return f"{instr}\0{st}\0{descs}"


# ── capacity check ──────────────────────────────────────────────────────


def _render_saracura_capacity_check(row: Mapping[str, Any], token_counter: TokenCounter) -> bool:
    try:
        criteria = tuple(
            ChoiceCriterion(
                id=cast(str, criterion["id"]),
                description=cast(str, criterion["description"]),
            )
            for criterion in cast(Sequence[Mapping[str, Any]], row["criteria"])
        )
        rendered = render_choice(
            locale=cast(str, row["locale"]),
            domain=cast(str, row["domain"]),
            instruction=cast(str, row["instruction"]),
            state=row["state"],
            criteria=criteria,
        )
        validate_rendered_capacity(rendered, token_counter)
        return True
    except Exception:
        return False


# ── cost report & streak ────────────────────────────────────────────────


def _v4_streak(entries: Sequence[Mapping[str, str]], start: int, streak: int) -> int:
    for entry in entries[start:]:
        status = entry.get("status")
        if status == "uncertain":
            streak += 1
        elif status in {"settled", "overspent"}:
            streak = 0
    return streak


def _provider_spend(ledger: corpus.BudgetLedger) -> Decimal:
    return sum((Decimal(entry["provider_cost_usd"]) for entry in ledger.entries), Decimal())


def _emit_provider_cost_progress(
    ledger: corpus.BudgetLedger, next_cost: Decimal, interval: Decimal
) -> Decimal:
    spend = _provider_spend(ledger)
    while spend >= next_cost:
        print(f"phase4e comparison provider spend crossed USD {next_cost}", flush=True)
        next_cost += interval
    return next_cost


# ── durable resolution state ────────────────────────────────────────────


def _call_work_path(work_dir: Path, task_ids: list[str]) -> Path:
    return work_dir / "resolved" / f"call-{_sha(_canonical(sorted(task_ids)))}.json"


def _store_comparison_resolution(
    work_dir: Path,
    slots: Sequence[Mapping[str, Any]],
    accepted: Sequence[Mapping[str, Any]],
    rejected: Sequence[Mapping[str, Any]],
    stage: str,
    plan_sha256: str,
) -> None:
    task_ids = sorted(cast(str, s["task_id"]) for s in slots)
    if set(task_ids) != {cast(str, r["task_id"]) for r in (*accepted, *rejected)}:
        raise ComparisonError("resolution coverage")
    resolutions = [
        {"task_id": cast(str, r["task_id"]), "status": "accepted", "row": dict(r)} for r in accepted
    ] + [
        {"task_id": cast(str, r["task_id"]), "status": "rejected", "row": dict(r)} for r in rejected
    ]
    payload = {
        "schema_version": COMPARISON_RESOLUTION_SCHEMA,
        "plan_sha256": plan_sha256,
        "stage": stage,
        "task_ids": task_ids,
        "resolutions": resolutions,
    }
    atomic_create(_call_work_path(work_dir, task_ids), _canonical(payload) + b"\n")


def _store_comparison_diagnostics(
    work_dir: Path,
    slots: Sequence[Mapping[str, Any]],
    diagnostics: Mapping[str, int],
    plan_sha256: str,
) -> None:
    task_ids = sorted(cast(str, s["task_id"]) for s in slots)
    atomic_create(
        work_dir / "diagnostics" / f"call-{_sha(_canonical(task_ids))}.json",
        _canonical(
            {
                "schema_version": "phase4e-comparison-diagnostics.v1",
                "plan_sha256": plan_sha256,
                "task_ids": task_ids,
                "counts": {key: diagnostics[key] for key in _PERSISTED_DIAGNOSTIC_KEYS},
            }
        )
        + b"\n",
    )


def _load_resolved(
    work_dir: Path, plan: Mapping[str, Any], plan_sha256: str
) -> dict[str, dict[str, Any]]:
    resolved_dir = work_dir / "resolved"
    if not resolved_dir.exists():
        return {}
    all_task_ids = {cast(str, s["task_id"]) for s in cast(list[Mapping[str, Any]], plan["slots"])}
    result: dict[str, dict[str, Any]] = {}
    for path in sorted(resolved_dir.glob("call-*.json")):
        try:
            payload = json.loads(path.read_bytes())
        except (OSError, ValueError) as error:
            raise ComparisonError("resolution file corrupted") from error
        if (
            not isinstance(payload, dict)
            or set(payload) != {"schema_version", "plan_sha256", "stage", "task_ids", "resolutions"}
            or payload.get("schema_version") != COMPARISON_RESOLUTION_SCHEMA
            or payload.get("plan_sha256") != plan_sha256
        ):
            raise ComparisonError("resolution plan binding")
        for resolution in cast(list[dict[str, Any]], payload.get("resolutions", [])):
            tid = resolution.get("task_id")
            if isinstance(tid, str) and tid in all_task_ids:
                result[tid] = resolution
    return result


def _load_comparison_diagnostics(work_dir: Path, plan_sha256: str) -> dict[str, int]:
    totals: dict[str, int] = dict.fromkeys(_PERSISTED_DIAGNOSTIC_KEYS, 0)
    directory = work_dir / "diagnostics"
    if not directory.exists():
        return totals
    for path in sorted(directory.glob("call-*.json")):
        try:
            payload = json.loads(path.read_bytes(), object_pairs_hook=_no_duplicate_keys)
        except (OSError, ValueError) as error:
            raise ComparisonError("diagnostic file corrupted") from error
        if (
            not isinstance(payload, dict)
            or set(payload) != {"schema_version", "plan_sha256", "task_ids", "counts"}
            or payload.get("schema_version") != "phase4e-comparison-diagnostics.v1"
            or payload.get("plan_sha256") != plan_sha256
            or not isinstance(payload.get("task_ids"), list)
            or not isinstance(payload.get("counts"), dict)
            or set(payload["counts"]) != set(_PERSISTED_DIAGNOSTIC_KEYS)
        ):
            raise ComparisonError("diagnostic file corrupted")
        for key in _PERSISTED_DIAGNOSTIC_KEYS:
            value = payload["counts"][key]
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ComparisonError("diagnostic file corrupted")
            totals[key] += value
    return totals


def _bind_generation_plan(work_dir: Path, plan_path: Path, plan: Mapping[str, Any]) -> str:
    """Bind every resumable generation artifact to the exact canonical plan bytes."""

    raw_plan = plan_path.read_bytes()
    if raw_plan != _canonical(dict(plan)) + b"\n":
        raise ComparisonError("comparison plan bytes are not canonical")
    plan_sha256 = _sha(raw_plan)
    binding_path = work_dir / "generation-binding.json"
    if not binding_path.exists():
        resumable_roots = (work_dir / "ledger", work_dir / "resolved", work_dir / "diagnostics")
        if any(root.exists() and any(root.iterdir()) for root in resumable_roots):
            raise ComparisonError("existing generation state has no plan binding")
    payload = {
        "schema_version": COMPARISON_GENERATION_BINDING_SCHEMA,
        "plan_sha256": plan_sha256,
    }
    _create_or_identical(binding_path, _canonical(payload) + b"\n", mode=0o400)
    return plan_sha256


# ── author/reviewer transport ───────────────────────────────────────────


FROM_CONTENT = cast(dict[str, Any], None)


def _comparison_author_messages(slots: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    return corpus.author_messages(slots)


def _comparison_author_schema(slots: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return corpus.author_schema(slots)


def _comparison_reviewer_view(row: Mapping[str, Any]) -> dict[str, Any]:
    criteria = row.get("criteria")
    if not isinstance(criteria, list):
        raise ComparisonError("reviewer transport criteria")
    view: dict[str, Any] = {
        "task_id": row["task_id"],
        "locale": row["locale"],
        "domain": row["domain"],
        "instruction": row["instruction"],
        "state": row["state"],
        "criteria": [
            {"id": c["id"], "description": c["description"]} if isinstance(c, Mapping) else {}
            for c in criteria
        ],
    }
    if set(view) != {"task_id", "locale", "domain", "instruction", "state", "criteria"}:
        raise ComparisonError("reviewer transport fields")
    forbidden = {
        "family_id",
        "pair_id",
        "split",
        "gold_position",
        "selected_criterion_id",
        "axes",
        "review",
    }
    if forbidden & set(view):
        raise ComparisonError("reviewer transport fields")
    return view


def _comparison_reviewer_system() -> str:
    return (
        "Return a JSON object whose sole top-level key is reviews, one review with status, "
        "chosen criterion ID, reason codes, quality flags natural_language, fictional, "
        "exclusive_options, and private_or_sensitive, plus independent "
        "semantic_equivalence_attestation. Accepted reviews require reason_codes=[]. "
        + corpus.SCENARIO_CODEBOOK
        + " "
        + corpus.CRITERION_ROLE_CODEBOOK
        + " Infer roles from rule, facts, and options; do not write role labels into criteria. "
        "Choose the single scenario that directly explains the decision. "
        "Attest one scenario, distinct ordered roles, and selected_role=matches_rule at the "
        "chosen position. Select one criterion or reject. "
        "Evaluate fictional and private_or_sensitive only from instruction, state.summary, and "
        "criteria[].description. You do not receive answer, author attestation, family, pair, "
        "gold position, sibling, or split metadata."
    )


def _comparison_reviewer_messages(row: Mapping[str, Any]) -> list[dict[str, str]]:
    safe = _comparison_reviewer_view(row)
    return [
        {"role": "system", "content": _comparison_reviewer_system()},
        {"role": "user", "content": _canonical({"tasks": [safe]}).decode("utf-8")},
    ]


# ── publish and sanitiser ───────────────────────────────────────────────


PUBLISH_FORBIDDEN_PATTERNS = {
    "absolute_path": __import__("re").compile(r"(?:/Users|/home|/Volumes|/root)/\S+"),
    "provider_id": __import__("re").compile(
        r"(?:reservation[_-]?id|request[_-]?id|response[_-]?id)"
    ),
    "credential": __import__("re").compile(
        r"(?:api[_-]?key|secret|token|password)", __import__("re").IGNORECASE
    ),
}


def _sanitize_for_publication(value: object) -> object:
    if isinstance(value, str):
        for pattern_name in ("absolute_path", "provider_id", "credential"):
            if PUBLISH_FORBIDDEN_PATTERNS[pattern_name].search(value):
                return "[redacted]"
        return value
    if isinstance(value, dict):
        sanitized: dict[str, object] = {}
        for key, child in value.items():
            folded = key.casefold()
            if any(
                marker in folded
                for marker in (
                    "absolute_path",
                    "reservation_id",
                    "request_id",
                    "response_id",
                    "api_key",
                    "secret",
                    "password",
                    "credential",
                    "home",
                )
            ):
                sanitized[key] = "[redacted]"
            else:
                sanitized[key] = _sanitize_for_publication(child)
        return sanitized
    if isinstance(value, list):
        return [_sanitize_for_publication(item) for item in value]
    return value


def _wilson_interval(correct: int, total: int, z: float = 1.96) -> tuple[float, float, float]:
    if total <= 0:
        return (0.0, 0.0, 0.0)
    n = float(total)
    p = correct / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1.0 - p) / n + z * z / (4 * n * n)) / denom
    return (p, max(0.0, center - margin), min(1.0, center + margin))


def _build_slice_result(correct: int, total: int) -> dict[str, Any]:
    accuracy, lower, upper = _wilson_interval(correct, total)
    return {
        "denominator": total,
        "correct": correct,
        "accuracy": round(accuracy, 6),
        "wilson_95_lower": round(lower, 6),
        "wilson_95_upper": round(upper, 6),
        "underpowered": total < COMPARISON_UNDERPOWERED_THRESHOLD,
    }


# ── CLI ──────────────────────────────────────────────────────────────────


def _openrouter_transport(timeout_seconds: int = _TRANSPORT_TIMEOUT) -> Transport:
    import urllib.request as urllib_request

    class _NoRedirect(urllib_request.HTTPRedirectHandler):
        def redirect_request(
            self,
            req: object,
            fp: object,
            code: object,
            msg: object,
            headers: object,
            newurl: object,
        ) -> None:
            raise OSError("redirects are disabled")

    def transport(
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes,
    ) -> tuple[int, Mapping[str, str], bytes]:
        if not isinstance(body, bytes):
            raise ComparisonError("transport body must be bytes")
        request = urllib_request.Request(url, data=body, method=method)
        for key, value in headers.items():
            request.add_header(key, value)
        opener = urllib_request.build_opener(urllib_request.ProxyHandler({}), _NoRedirect)
        try:
            response = opener.open(request, timeout=timeout_seconds)
        except urllib_request.HTTPError as error:
            return error.code, dict(error.headers), error.read()
        return response.getcode(), dict(response.headers), response.read()

    return transport


def _configured_private_root() -> Path | None:
    raw = os.environ.get(_PRIVATE_STATE_ROOT_ENV)
    if raw is None:
        return None
    candidate = Path(raw)
    if not candidate.is_absolute() or candidate.is_symlink():
        raise ComparisonError("private state root must be an absolute path")
    cursor = Path(candidate.anchor)
    for part in candidate.parts[1:]:
        cursor /= part
        if cursor.is_symlink():
            raise ComparisonError("private state root cannot contain symlinks")
    try:
        metadata = candidate.stat()
    except OSError as exc:
        raise ComparisonError("private state root cannot be stat'd") from exc
    if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.getuid():
        raise ComparisonError("private state root must be owned by the user")
    if stat.S_IMODE(metadata.st_mode) != 0o700:
        raise ComparisonError("private state root must be chmod 0700")
    resolved = candidate.resolve()
    checkout = Path(__file__).resolve().parents[1]
    try:
        resolved.relative_to(checkout)
    except ValueError:
        return resolved
    raise ComparisonError("private state root is inside the checkout")


def _require_private_root() -> Path:
    candidate = _configured_private_root()
    if candidate is None:
        raise ComparisonError("SARACURA_PRIVATE_STATE_ROOT is not set")
    return candidate / "phase4e" / "comparison"


def _require_live_external_comparison_root() -> Path:
    try:
        from benchmarks.saracura_universal_pilot import require_live_private_phase4e_root

        return require_live_private_phase4e_root() / "comparison"
    except Exception as error:
        raise ComparisonError(
            "live comparison requires the reviewed external private root"
        ) from error


def _live_work_root() -> Path:
    return _require_private_root()


def _canonical_task_projection(slot: Mapping[str, Any], workflow_revision: str) -> dict[str, Any]:
    """Project one slot into canonical semantic fields; only workflow/model identity differ."""
    oc = cast(int, slot["option_count"])
    criteria = tuple(
        {"id": f"criterion-{index}", "description": f"Criterion option {index}."}
        for index in range(oc)
    )
    return {
        "task_id": slot["task_id"],
        "family_id": slot["family_id"],
        "locale": slot["locale"],
        "domain": slot["domain"],
        "instruction": "Choose the best option given the state and rule.",
        "state": {"summary": "Synthetic comparison state anchored to the planned slot."},
        "criteria": criteria,
        "selected_criterion_id": criteria[cast(int, slot["gold_position"])]["id"],
        "semantic_equivalence_attestation": {
            "scenario": slot["semantic_target"]["scenario"],
            "criterion_roles": list(slot["semantic_target"]["criterion_roles"]),
            "selected_role": "matches_rule",
        },
        "option_count": oc,
        "gold_position": slot["gold_position"],
        "split": "synthetic_comparison",
        "synthetic_only": True,
    }


def _check_acceptance_minimums(
    accepted: Sequence[Mapping[str, Any]],
    plan: Mapping[str, Any],
) -> list[str]:
    errors: list[str] = []
    total = len(accepted)
    if total < 150:
        errors.append(f"total_accepted={total} < 150")
    for locale in LOCALES:
        lc = sum(1 for r in accepted if r.get("locale") == locale)
        if lc < 60:
            errors.append(f"locale {locale} accepted={lc} < 60")
    for domain in DOMAINS:
        dc = sum(1 for r in accepted if r.get("domain") == domain)
        if dc < 5:
            errors.append(f"domain {domain} accepted={dc} < 5")
    for count in OPTION_COUNTS:
        cc = sum(1 for r in accepted if r.get("option_count") == count)
        if cc < 10:
            errors.append(f"option_count {count} accepted={cc} < 10")
    pairs: dict[str, set[str]] = defaultdict(set)
    for r in accepted:
        pid = r.get("pair_id")
        loc = r.get("locale")
        if isinstance(pid, str) and isinstance(loc, str):
            pairs[pid].add(loc)
    complete = sum(s == {"pt-BR", "en"} for s in pairs.values())
    if complete < 60:
        errors.append(f"complete_pairs={complete} < 60")
    return errors


def _comparison_pair_batches(slots: Sequence[Mapping[str, Any]]) -> list[list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    order: list[str] = []
    for slot in slots:
        pid = slot.get("pair_id")
        if not isinstance(pid, str):
            pid = cast(str, slot["task_id"])
        if pid not in grouped:
            order.append(pid)
        grouped[pid].append(slot)
    return [grouped[pid] for pid in order]


# ── subcommand: plan ────────────────────────────────────────────────────


def _live_plan_bindings(
    *,
    training_packet: Path,
    training_report_dir: Path | None,
    training_capsule: Path,
    saracura_encoder_snapshot: Path,
    laya_snapshot: Path,
) -> dict[str, Any]:
    try:
        from benchmarks import saracura_universal_training as training
        from saracura.laya_snapshot import verify_laya_snapshot

        packet_binding = training.validate_accepted_packet_binding(
            training_packet,
            report_dir=training_report_dir,
        )
        training_manifest = training.verify_training_capsule(training_capsule)
        if training_manifest.get("outcome") != "passed":
            raise ComparisonError("comparison requires a passed training capsule")
        corpus.VerifiedMiniLMTokenizerReceipt.create(saracura_encoder_snapshot)
        laya_receipt = verify_laya_snapshot(laya_snapshot)
        laya_receipt.close()
    except ComparisonError:
        raise
    except Exception as error:
        raise ComparisonError("live plan evidence verification failed") from error

    task_ids = sorted(cast(str, identity["task_id"]) for identity in packet_binding.identities)
    family_ids = sorted(
        {cast(str, identity["family_id"]) for identity in packet_binding.identities}
    )
    seed = cast(str, _load_policy()["plan"]["seed"])
    comparison_task_ids = {
        COMPARISON_NAMESPACE_PREFIX + _sha(f"{seed}\0task\0{locale}\0{serial}".encode())
        for serial, locale in enumerate(("pt", "en") * 100)
    }
    comparison_family_ids = {
        COMPARISON_FAMILY_PREFIX + _sha(f"{seed}\0family\0{index}".encode()) for index in range(100)
    }
    if comparison_task_ids.intersection(task_ids) or comparison_family_ids.intersection(family_ids):
        raise ComparisonError("comparison identities overlap training identities")
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    source_commit = completed.stdout.strip()
    if completed.returncode != 0 or len(source_commit) != 40:
        raise ComparisonError("comparison source commit unavailable")
    encoder_digest, _encoder_bytes = _artifact_digest_and_size(saracura_encoder_snapshot)
    laya_digest, _laya_bytes = _artifact_digest_and_size(laya_snapshot)
    return {
        "mode": "live_verified",
        "training_packet_receipt_sha256": packet_binding.packet_json_sha256,
        "training_accepted_rows_sha256": packet_binding.accepted_rows_sha256,
        "training_manifest_sha256": cast(str, training_manifest["manifest_sha256"]),
        "training_checkpoint_sha256": cast(str, training_manifest["checkpoint_sha256"]),
        "source_commit": source_commit,
        "saracura_tokenizer_snapshot_sha256": encoder_digest,
        "laya_snapshot_sha256": laya_digest,
        "training_task_ids_sha256": _sha(_canonical(task_ids)),
        "training_family_ids_sha256": _sha(_canonical(family_ids)),
        "identities_disjoint": True,
    }


def run_plan(
    output: Path,
    *,
    bindings: Mapping[str, Any] | None = None,
    training_packet: Path | None = None,
    training_report_dir: Path | None = None,
    training_capsule: Path | None = None,
    saracura_encoder_snapshot: Path | None = None,
    laya_snapshot: Path | None = None,
) -> Path:
    _require_component(output, "comparison-plan", file_path=True)
    if bindings is None:
        if any(
            value is None
            for value in (
                training_packet,
                training_capsule,
                saracura_encoder_snapshot,
                laya_snapshot,
            )
        ):
            raise ComparisonError("live plan requires explicit sealed evidence")
        bindings = _live_plan_bindings(
            training_packet=cast(Path, training_packet),
            training_report_dir=training_report_dir,
            training_capsule=cast(Path, training_capsule),
            saracura_encoder_snapshot=cast(Path, saracura_encoder_snapshot),
            laya_snapshot=cast(Path, laya_snapshot),
        )
        _require_within(output, _require_live_external_comparison_root())
    plan = build_comparison_plan(bindings=bindings)
    _validate_comparison_plan(plan)
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_create(output, _canonical(plan) + b"\n")
    os.chmod(output, 0o600)
    return output


def _load_full_training_rows(
    packet: Path,
    report_dir: Path | None,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    try:
        from benchmarks import saracura_universal_training as training

        binding = training.validate_accepted_packet_binding(packet, report_dir=report_dir)
        train_dev_raw = (packet / "accepted-train-dev.jsonl").read_bytes()
        holdout_raw = (packet / "accepted-holdout.jsonl").read_bytes()
        corpus.validate_accepted_packet_holdout_bytes(packet, holdout_raw)
        rows = [
            cast(dict[str, Any], json.loads(line))
            for raw in (train_dev_raw, holdout_raw)
            for line in raw.splitlines()
            if line
        ]
    except Exception as error:
        raise ComparisonError("accepted training packet verification failed") from error
    identity_ids = {cast(str, identity["task_id"]) for identity in binding.identities}
    if (
        len(rows) != len(identity_ids)
        or {cast(str, row["task_id"]) for row in rows} != identity_ids
    ):
        raise ComparisonError("accepted training packet content coverage")
    task_ids = sorted(identity_ids)
    family_ids = sorted({cast(str, row["family_id"]) for row in rows})
    return rows, {
        "training_packet_receipt_sha256": binding.packet_json_sha256,
        "training_accepted_rows_sha256": binding.accepted_rows_sha256,
        "training_task_ids_sha256": _sha(_canonical(task_ids)),
        "training_family_ids_sha256": _sha(_canonical(family_ids)),
    }


def _verify_generation_plan_bindings(
    bindings: Mapping[str, Any],
    *,
    training_metadata: Mapping[str, str],
    saracura_encoder_snapshot: Path,
    laya_snapshot: Path,
) -> None:
    encoder_digest, _encoder_bytes = _artifact_digest_and_size(saracura_encoder_snapshot)
    laya_digest, _laya_bytes = _artifact_digest_and_size(laya_snapshot)
    expected = {
        **training_metadata,
        "saracura_tokenizer_snapshot_sha256": encoder_digest,
        "laya_snapshot_sha256": laya_digest,
    }
    if any(bindings.get(key) != value for key, value in expected.items()):
        raise ComparisonError("generation evidence does not match the sealed plan")


# ── subcommand: generate ────────────────────────────────────────────────


def run_generate(
    plan_path: Path,
    work_dir: Path,
    packet: Path,
    *,
    allow_network: bool,
    transport: Transport | None = None,
    tokenizer_receipt: corpus.VerifiedMiniLMTokenizerReceipt | None = None,
    saracura_encoder_snapshot: Path | None = None,
    laya_snapshot: Path | None = None,
    training_packet: Path | None = None,
    training_report_dir: Path | None = None,
    training_rows: Sequence[Mapping[str, Any]] | None = None,
    laya_capacity_check: Callable[[Mapping[str, Any]], bool] | None = None,
) -> Path:
    _require_component(plan_path, "comparison-plan", file_path=True)
    _require_component(work_dir, "comparison-work")
    _require_component(packet, "comparison-packet", file_path=False)
    if not allow_network:
        raise ComparisonError("generate requires literal --allow-network")
    plan = json.loads(plan_path.read_bytes(), object_pairs_hook=_no_duplicate_keys)
    _validate_comparison_plan(plan)
    private = (
        _require_private_root()
        if transport is not None
        else _require_live_external_comparison_root()
    )
    _require_within(plan_path, private)
    _require_within(work_dir, private)
    _require_within(packet, private)
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ComparisonError("OPENROUTER_API_KEY is not set")
    live_plan = plan["bindings"]["mode"] == "live_verified"
    if tokenizer_receipt is None:
        if saracura_encoder_snapshot is None:
            raise ComparisonError("generate requires verified tokenizer receipt")
        try:
            tokenizer_receipt = corpus.VerifiedMiniLMTokenizerReceipt.create(
                saracura_encoder_snapshot
            )
        except Exception as error:
            raise ComparisonError("Saracura tokenizer verification failed") from error
    laya_receipt: Any | None = None
    if laya_capacity_check is None:
        if laya_snapshot is None:
            if transport is None:
                raise ComparisonError("generate requires the verified Laya snapshot")

            def accept_fixture_capacity(_row: Mapping[str, Any]) -> bool:
                return True

            laya_capacity_check = accept_fixture_capacity
        else:
            try:
                from saracura.backends.laya import _direct_tokenizer, render_laya_choice
                from saracura.laya_snapshot import verify_laya_snapshot

                laya_receipt = verify_laya_snapshot(laya_snapshot)
                laya_tokenizer = _direct_tokenizer(laya_receipt)

                def check_laya(row: Mapping[str, Any]) -> bool:
                    try:
                        request = _accepted_row_to_request(
                            row,
                            model_revision=laya_receipt.candidate.model_revision,
                            workflow_revision=COMPARISON_LAYA_REVISION,
                        )
                        render_laya_choice(laya_tokenizer, request, request.questions[0])
                        return True
                    except Exception:
                        return False

                laya_capacity_check = check_laya
            except Exception as error:
                if laya_receipt is not None:
                    laya_receipt.close()
                raise ComparisonError("Laya tokenizer verification failed") from error
    training_metadata: dict[str, str] | None = None
    if training_rows is None:
        if training_packet is None:
            if transport is None:
                if laya_receipt is not None:
                    laya_receipt.close()
                raise ComparisonError("generate requires the accepted training packet")
            training_rows = ()
        else:
            training_rows, training_metadata = _load_full_training_rows(
                training_packet,
                training_report_dir,
            )
    if live_plan:
        if saracura_encoder_snapshot is None or laya_snapshot is None or training_packet is None:
            if laya_receipt is not None:
                laya_receipt.close()
            raise ComparisonError("live generation requires all plan-bound evidence")
        if training_metadata is None:
            if laya_receipt is not None:
                laya_receipt.close()
            raise ComparisonError("live generation cannot use injected training rows")
        _verify_generation_plan_bindings(
            cast(Mapping[str, Any], plan["bindings"]),
            training_metadata=cast(Mapping[str, str], training_metadata),
            saracura_encoder_snapshot=saracura_encoder_snapshot,
            laya_snapshot=laya_snapshot,
        )
    if plan["bindings"]["mode"] != "live_verified" and transport is None:
        if laya_receipt is not None:
            laya_receipt.close()
        raise ComparisonError("live generation requires a live-verified plan")
    counter: TokenCounter = tokenizer_receipt
    chosen_transport = transport if transport is not None else _openrouter_transport()
    work_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    plan_sha256 = _bind_generation_plan(work_dir, plan_path, plan)
    slots = cast(list[Mapping[str, Any]], plan["slots"])
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    diagnostics: dict[str, int] = dict.fromkeys(_PERSISTED_DIAGNOSTIC_KEYS, 0)
    ledger = corpus.BudgetLedger(policy=COMPARISON_LEDGER_POLICY)
    try:
        resolved = _load_resolved(work_dir, plan, plan_sha256)
        for value in resolved.values():
            row = value.get("row")
            if isinstance(row, dict):
                if value.get("status") == "accepted":
                    accepted.append(row)
                elif value.get("status") == "rejected":
                    rejected.append(row)
        resumed = corpus.resume_ledger(work_dir / "ledger")
        ledger = resumed if resumed.entries else ledger
        if ledger.policy.schema_version != COMPARISON_LEDGER_POLICY.schema_version:
            raise ComparisonError("ledger resume mismatch")
        diagnostics = _load_comparison_diagnostics(work_dir, plan_sha256)
    except Exception as error:
        _write_terminal_report(
            report_dir=packet.parent,
            plan=plan,
            accepted=accepted,
            rejected=rejected,
            ledger=ledger,
            outcome="inconclusive_operational",
            diagnostics=diagnostics,
            errors=("generation_state_resume_failure",),
        )
        if laya_receipt is not None:
            laya_receipt.close()
        raise ComparisonError("inconclusive_operational") from error
    client = corpus.OpenRouterCorpusClient(
        transport=cast(corpus.Transport, chosen_transport),
        allow_network=True,
        ledger=ledger,
        ledger_directory=work_dir / "ledger",
        policy=COMPARISON_LEDGER_POLICY,
        author_stage=AUTHOR_STAGE,
        reviewer_stage=REVIEWER_STAGE,
        author_model=_COMPARISON_AUTHOR_MODEL,
        reviewer_model=_COMPARISON_REVIEWER_MODEL,
        provider_preferences_by_stage={
            AUTHOR_STAGE: _COMPARISON_PROVIDER_POLICY,
            REVIEWER_STAGE: _COMPARISON_PROVIDER_POLICY,
        },
        author_reasoning_effort=None,
        reviewer_temperature=0,
        reviewer_reasoning=None,
    )
    execution_policy = cast(Mapping[str, Any], _load_policy()["execution"])
    maximum_attempts = cast(int, execution_policy["maximum_provider_attempts"])
    uncertainty_circuit = cast(int, execution_policy["transport_uncertainty_circuit"])
    report_interval = Decimal("10.00")
    existing_spend = _provider_spend(client.ledger)
    next_cost = (existing_spend // report_interval + 1) * report_interval
    completed: set[str] = set(resolved)
    all_task_ids = {cast(str, s["task_id"]) for s in slots}
    streak = _v4_streak(client.ledger.entries, 0, 0)
    try:
        if completed != all_task_ids:
            batches = _comparison_pair_batches(slots)
            for batch_slots in batches:
                batch_ids = [cast(str, s["task_id"]) for s in batch_slots]
                if set(batch_ids) <= completed:
                    continue
                if any(tid in completed for tid in batch_ids):
                    raise ComparisonError("partial pair resume")
                before_entries = len(client.ledger.entries)

                try:
                    _run_comparison_batch(
                        client,
                        batch_slots,
                        counter,
                        api_key,
                        work_dir,
                        accepted,
                        rejected,
                        diagnostics,
                        maximum_attempts,
                        training_rows,
                        laya_capacity_check,
                        plan_sha256,
                    )
                except ComparisonError:
                    if not client.ledger.entries:
                        raise
                    last = client.ledger.entries[-1]
                    if last["status"] != "uncertain":
                        raise
                    diagnostics["transport_uncertain_calls"] = max(
                        diagnostics["transport_uncertain_calls"], 1
                    )
                    streak = _v4_streak(client.ledger.entries, before_entries, streak)
                    next_cost = _emit_provider_cost_progress(
                        client.ledger, next_cost, report_interval
                    )
                    if streak >= uncertainty_circuit:
                        corpus.write_ledger_snapshot(
                            work_dir / "ledger",
                            client.ledger,
                            stop_reason="transport_circuit_open",
                        )
                        _write_terminal_report(
                            report_dir=packet.parent,
                            plan=plan,
                            accepted=accepted,
                            rejected=rejected,
                            ledger=client.ledger,
                            outcome="inconclusive_transport",
                            diagnostics=diagnostics,
                            streak=streak,
                        )
                        spend = _provider_spend(client.ledger)
                        print(
                            f"phase4e comparison final provider spend USD {spend}",
                            flush=True,
                        )
                        raise ComparisonError("transport circuit open") from None
                    completed.update(batch_ids)
                    continue

                streak = _v4_streak(client.ledger.entries, before_entries, streak)
                completed.update(batch_ids)
                next_cost = _emit_provider_cost_progress(client.ledger, next_cost, report_interval)

            if streak >= uncertainty_circuit:
                corpus.write_ledger_snapshot(
                    work_dir / "ledger",
                    client.ledger,
                    stop_reason="transport_circuit_open",
                )
                _write_terminal_report(
                    report_dir=packet.parent,
                    plan=plan,
                    accepted=accepted,
                    rejected=rejected,
                    ledger=client.ledger,
                    outcome="inconclusive_transport",
                    diagnostics=diagnostics,
                    streak=streak,
                )
                raise ComparisonError("transport circuit open")
    except Exception as error:
        if isinstance(error, ComparisonError) and str(error) == "transport circuit open":
            raise
        next_cost = _emit_provider_cost_progress(client.ledger, next_cost, report_interval)
        _write_terminal_report(
            report_dir=packet.parent,
            plan=plan,
            accepted=accepted,
            rejected=rejected,
            ledger=client.ledger,
            outcome="inconclusive_operational",
            diagnostics=diagnostics,
            streak=streak,
            errors=("generation_operational_failure",),
        )
        raise ComparisonError("inconclusive_operational") from error
    finally:
        if laya_receipt is not None:
            laya_receipt.close()

    accept_errors = _check_acceptance_minimums(accepted, plan)
    has_transport_uncertain = any(
        r.get("reason") in {"author_transport_uncertain", "reviewer_transport_uncertain"}
        for r in rejected
    )
    if accept_errors:
        if has_transport_uncertain:
            outcome: Literal[
                "insufficient_comparison_evidence",
                "inconclusive_transport",
                "inconclusive_operational",
                "passed",
            ] = "inconclusive_transport"
        else:
            outcome = "insufficient_comparison_evidence"
        _write_terminal_report(
            report_dir=packet.parent,
            plan=plan,
            accepted=accepted,
            rejected=rejected,
            ledger=client.ledger,
            outcome=outcome,
            diagnostics=diagnostics,
            errors=accept_errors,
        )
        next_cost = _emit_provider_cost_progress(client.ledger, next_cost, report_interval)
        spend = _provider_spend(client.ledger)
        print(f"phase4e comparison final provider spend USD {spend}", flush=True)
        raise ComparisonError(outcome)

    try:
        _seal_comparison_packet(packet, plan, accepted, rejected, client.ledger.as_json(final=True))
    except Exception as error:
        next_cost = _emit_provider_cost_progress(client.ledger, next_cost, report_interval)
        _write_terminal_report(
            report_dir=packet.parent,
            plan=plan,
            accepted=accepted,
            rejected=rejected,
            ledger=client.ledger,
            outcome="inconclusive_operational",
            diagnostics=diagnostics,
            errors=("comparison_packet_seal_failure",),
        )
        raise ComparisonError("inconclusive_operational") from error
    next_cost = _emit_provider_cost_progress(client.ledger, next_cost, report_interval)
    spend = _provider_spend(client.ledger)
    print(f"phase4e comparison final provider spend USD {spend}", flush=True)
    _write_terminal_report(
        report_dir=packet.parent,
        plan=plan,
        accepted=accepted,
        rejected=rejected,
        ledger=client.ledger,
        outcome="passed",
        diagnostics=diagnostics,
    )
    return packet / "packet.json"


def _run_comparison_batch(
    client: corpus.OpenRouterCorpusClient,
    slots: Sequence[Mapping[str, Any]],
    counter: TokenCounter,
    api_key: str,
    work_dir: Path,
    accepted: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    diagnostics: dict[str, int],
    maximum_attempts: int,
    training_rows: Sequence[Mapping[str, Any]],
    laya_capacity_check: Callable[[Mapping[str, Any]], bool],
    plan_sha256: str,
) -> None:
    diag_before = dict(diagnostics)
    messages = _comparison_author_messages(slots)
    auth_schema = _comparison_author_schema(slots)
    task_ids = [cast(str, s["task_id"]) for s in slots]
    usable: list[dict[str, Any]] | None = None
    rejected_author: list[dict[str, Any]] | None = None
    author_lineage: dict[str, str] | None = None
    for attempt in range(maximum_attempts):
        try:
            provider_response = client._request(
                stage=AUTHOR_STAGE,
                model=_COMPARISON_AUTHOR_MODEL,
                messages=messages,
                response_schema=auth_schema,
                max_output_tokens=_AUTHOR_MAX_TOKENS,
                api_key=api_key,
                task_ids=task_ids,
            )
        except corpus.CorpusError:
            diagnostics["author_response_failures"] += 1
            if client.ledger.entries and client.ledger.entries[-1]["status"] == "uncertain":
                diagnostics["transport_uncertain_calls"] += 1
                uncertain_lineage = _uncertain_lineage(client, "author")
                rows = [
                    {
                        "task_id": cast(str, s["task_id"]),
                        "split": s["split"],
                        "reason": "author_transport_uncertain",
                        **uncertain_lineage,
                    }
                    for s in slots
                ]
                _store_comparison_resolution(
                    work_dir, slots, [], rows, "author_uncertain", plan_sha256
                )
                _store_comparison_diagnostics(
                    work_dir,
                    slots,
                    {k: diagnostics[k] - diag_before[k] for k in _PERSISTED_DIAGNOSTIC_KEYS},
                    plan_sha256,
                )
                rejected.extend(rows)
                raise ComparisonError("author transport uncertain") from None
            author_lineage = corpus.lineage_from_journal(
                client.last_journal(AUTHOR_STAGE),
                "author",
                expected_stage=AUTHOR_STAGE,
            )
            break

        author_lineage = corpus.lineage_from_journal(
            client.last_journal(AUTHOR_STAGE),
            "author",
            expected_stage=AUTHOR_STAGE,
        )
        try:
            raw_content = _response_content(provider_response)
            decoded = corpus.decode_author_response(raw_content, slots)
            c_usable, c_rejected = corpus.classify_author_rows(decoded, slots, counter)
            if c_rejected:
                diagnostics["author_response_failures"] += 1
                if attempt + 1 < maximum_attempts:
                    continue
                rejected_author = c_rejected
                break
            usable, rejected_author = c_usable, c_rejected
            if attempt:
                diagnostics["retry_recoveries"] += 1
            break
        except corpus.CorpusError:
            diagnostics["author_response_failures"] += 1
            if attempt + 1 < maximum_attempts:
                continue
            break
    if usable is None or rejected_author is None:
        if author_lineage is None:
            raise ComparisonError("author recovery invariant")
        rows = [
            {
                "task_id": cast(str, s["task_id"]),
                "split": s["split"],
                "reason": "author_response_failure",
                **author_lineage,
            }
            for s in slots
        ]
        _store_comparison_resolution(work_dir, slots, [], rows, "author_fallback", plan_sha256)
        _store_comparison_diagnostics(
            work_dir,
            slots,
            {k: diagnostics[k] - diag_before[k] for k in _PERSISTED_DIAGNOSTIC_KEYS},
            plan_sha256,
        )
        rejected.extend(rows)
        return
    if author_lineage is None:
        raise ComparisonError("author lineage missing")
    slots_by_id = {cast(str, slot["task_id"]): slot for slot in slots}
    usable = [
        {
            **row,
            "option_count": slots_by_id[cast(str, row["task_id"])]["option_count"],
            "gold_position": slots_by_id[cast(str, row["task_id"])]["gold_position"],
            **author_lineage,
        }
        for row in usable
    ]
    saracura_rejected = [
        {
            "task_id": row["task_id"],
            "split": row["split"],
            "reason": "saracura_capacity_rejection",
            **author_lineage,
        }
        for row in usable
        if not _render_saracura_capacity_check(row, counter)
    ]
    rejected_saracura_ids = {cast(str, row["task_id"]) for row in saracura_rejected}
    usable = [row for row in usable if row["task_id"] not in rejected_saracura_ids]
    laya_rejected = [
        {
            "task_id": row["task_id"],
            "split": row["split"],
            "reason": "laya_capacity_or_tokenizer_rejection",
            **author_lineage,
        }
        for row in usable
        if not laya_capacity_check(row)
    ]
    rejected_laya_ids = {cast(str, row["task_id"]) for row in laya_rejected}
    usable = [row for row in usable if row["task_id"] not in rejected_laya_ids]
    rejected_author = [*rejected_author, *saracura_rejected, *laya_rejected]
    rejected_author = [{**row, **author_lineage} for row in rejected_author]

    # Review the full pair before resolving it so pair gates and duplicate checks
    # operate on the same precommitted family.
    reviews: list[dict[str, Any]] = []
    reviewed_rows: list[dict[str, Any]] = []
    reviewer_failures: list[dict[str, Any]] = []
    reviewer_lineages: dict[str, dict[str, str]] = {}
    for row in usable:
        task_id = cast(str, row["task_id"])
        for rattempt in range(maximum_attempts):
            try:
                rev_response = client._request(
                    stage=REVIEWER_STAGE,
                    model=_COMPARISON_REVIEWER_MODEL,
                    messages=_comparison_reviewer_messages(row),
                    response_schema=corpus.reviewer_schema(1),
                    max_output_tokens=_REVIEWER_MAX_TOKENS,
                    api_key=api_key,
                    task_ids=[task_id],
                )
            except corpus.CorpusError:
                diagnostics["reviewer_response_failures"] += 1
                if client.ledger.entries and client.ledger.entries[-1]["status"] == "uncertain":
                    diagnostics["transport_uncertain_calls"] += 1
                    ulin = _uncertain_lineage(client, "reviewer")
                    reviewer_lineages[task_id] = ulin
                    reviewer_failures.append(
                        {
                            "task_id": task_id,
                            "split": row["split"],
                            "reason": "reviewer_transport_uncertain",
                            **author_lineage,
                            **ulin,
                        }
                    )
                    break
                reviewer_lineages[task_id] = corpus.lineage_from_journal(
                    client.last_journal(REVIEWER_STAGE),
                    "reviewer",
                    expected_stage=REVIEWER_STAGE,
                )
                break
            reviewer_lineages[task_id] = corpus.lineage_from_journal(
                client.last_journal(REVIEWER_STAGE),
                "reviewer",
                expected_stage=REVIEWER_STAGE,
            )
            try:
                response = _response_content(rev_response)
                raw_reviews = response.get("reviews")
                if (
                    not isinstance(raw_reviews, list)
                    or len(raw_reviews) != 1
                    or not isinstance(raw_reviews[0], Mapping)
                ):
                    raise corpus.CorpusError("review response")
                review = corpus.bind_reviewer_record(raw_reviews[0], task_id)
                reviews.append(review)
                reviewed_rows.append(row)
                break
            except corpus.CorpusError:
                diagnostics["reviewer_response_failures"] += 1
                if rattempt + 1 < maximum_attempts:
                    continue
                reviewer_failures.append(
                    {
                        "task_id": task_id,
                        "split": row["split"],
                        "reason": "reviewer_response_failure",
                        **author_lineage,
                        **reviewer_lineages[task_id],
                    }
                )
                break
    batch_accepted, batch_rejected = corpus.resolve_reviews(
        reviewed_rows,
        reviews,
        prior_rows=(*training_rows, *accepted),
        reviewer_lineages=reviewer_lineages,
        acceptance=corpus.acceptance_reason,
    )
    batch_rejected = [*rejected_author, *reviewer_failures, *batch_rejected]
    _store_comparison_resolution(
        work_dir, slots, batch_accepted, batch_rejected, "complete", plan_sha256
    )
    _store_comparison_diagnostics(
        work_dir,
        slots,
        {k: diagnostics[k] - diag_before[k] for k in _PERSISTED_DIAGNOSTIC_KEYS},
        plan_sha256,
    )
    accepted.extend(batch_accepted)
    rejected.extend(batch_rejected)


def _uncertain_lineage(
    client: corpus.OpenRouterCorpusClient,
    actor: Literal["author", "reviewer"],
) -> dict[str, str]:
    if not client.ledger.entries:
        raise ComparisonError("uncertain lineage")
    entry = client.ledger.entries[-1]
    if entry["status"] != "uncertain":
        raise ComparisonError("uncertain lineage")
    return {
        f"{actor}_response_sha256": entry["response_sha256"],
        f"{actor}_reservation_id": entry["reservation_id"],
        f"{actor}_request_id": entry["request_id"],
    }


def _response_content(provider_response: dict[str, object]) -> dict[str, Any]:
    choices = cast(list[dict[str, Any]], provider_response.get("choices", []))
    if not choices:
        raise corpus.CorpusError("provider response missing choices")
    content = choices[0].get("message", {}).get("content", "")
    if not isinstance(content, str):
        raise corpus.CorpusError("provider response missing content")
    try:
        return cast(dict[str, Any], json.loads(content))
    except (json.JSONDecodeError, TypeError) as exc:
        raise corpus.CorpusError("provider response not valid JSON") from exc


def _jsonl(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(_canonical(dict(row)) + b"\n" for row in rows)


def _seal_comparison_packet(
    packet: Path,
    plan: Mapping[str, Any],
    accepted: Sequence[Mapping[str, Any]],
    rejected: Sequence[Mapping[str, Any]],
    ledger: Mapping[str, Any],
) -> Path:
    """Create a descriptor-bound private packet without using corpus packet schemas."""

    _validate_comparison_plan(cast(dict[str, Any], dict(plan)))
    planned_ids = {
        cast(str, slot["task_id"]) for slot in cast(Sequence[Mapping[str, Any]], plan["slots"])
    }
    accepted_ids = [cast(str, row.get("task_id")) for row in accepted]
    rejected_ids = [cast(str, row.get("task_id")) for row in rejected]
    if (
        len(set(accepted_ids)) != len(accepted_ids)
        or len(set(rejected_ids)) != len(rejected_ids)
        or set(accepted_ids) & set(rejected_ids)
        or set(accepted_ids) | set(rejected_ids) != planned_ids
    ):
        raise ComparisonError("comparison packet task coverage")
    _validate_packet_rows(accepted, rejected, plan)
    payloads = {
        "plan.json": _canonical(dict(plan)) + b"\n",
        "accepted.jsonl": _jsonl(accepted),
        "rejected.jsonl": _jsonl(rejected),
        "ledger.json": _canonical(dict(ledger)) + b"\n",
    }
    descriptor = {
        "schema_version": "phase4e-comparison-packet.v1",
        "plan_sha256": _sha(payloads["plan.json"]),
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "files": {name: _sha(payload) for name, payload in payloads.items()},
    }
    packet.mkdir(mode=0o700, parents=True, exist_ok=False)
    for name, payload in payloads.items():
        atomic_create(packet / name, payload)
        os.chmod(packet / name, 0o600)
    atomic_create(packet / "packet.json", _canonical(descriptor) + b"\n")
    os.chmod(packet / "packet.json", 0o600)
    return packet / "packet.json"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        return [cast(dict[str, Any], json.loads(line)) for line in path.read_bytes().splitlines()]
    except (OSError, ValueError) as error:
        raise ComparisonError("comparison packet JSONL") from error


# ── terminal report ─────────────────────────────────────────────────────


def _write_terminal_report(
    *,
    report_dir: Path,
    plan: Mapping[str, Any],
    accepted: Sequence[Mapping[str, Any]],
    rejected: Sequence[Mapping[str, Any]],
    ledger: corpus.BudgetLedger,
    outcome: str,
    diagnostics: Mapping[str, int] | None = None,
    streak: int = 0,
    errors: Sequence[str] = (),
) -> Path:
    slots = cast(list[Mapping[str, Any]], plan["slots"])
    accepted_ids = {cast(str, r["task_id"]) for r in accepted}
    rejected_ids = {cast(str, r["task_id"]) for r in rejected}
    unresolved = [s for s in slots if s["task_id"] not in accepted_ids | rejected_ids]
    total_uncertain = diagnostics.get("transport_uncertain_calls", 0) if diagnostics else 0
    spend = sum((Decimal(e["provider_cost_usd"]) for e in ledger.entries), Decimal())
    report = {
        "schema_version": COMPARISON_TERMINAL_REPORT_SCHEMA,
        "plan_sha256": _sha(_canonical(dict(plan)) + b"\n"),
        "outcome": outcome,
        "planned": 200,
        "accepted": len(accepted),
        "rejected": len(rejected),
        "unresolved": len(unresolved),
        "transport_uncertain_calls": total_uncertain,
        "nonconsecutive_transport_uncertain_calls": max(0, total_uncertain - streak),
        "consecutive_uncertainty_streak": streak,
        "provider_calls": len(ledger.entries),
        "provider_reported_cost_usd": str(spend),
        "errors": list(errors),
    }
    report_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    report_path = report_dir / "terminal-report.json"
    atomic_create(report_path, _canonical(report) + b"\n")
    return report_path


# ── subcommand: score-backend ───────────────────────────────────────────


def run_score_backend(
    packet: Path,
    backend: Literal["saracura", "laya"],
    work_dir: Path,
    *,
    device: Literal["cpu", "mps"] = "cpu",
    cpu_threads: int = 1,
    encoder_snapshot: Path | None = None,
    training_capsule: Path | None = None,
    laya_snapshot: Path | None = None,
) -> Path:
    from saracura.contracts.errors import SaracuraError

    if backend == "saracura" and (encoder_snapshot is None or training_capsule is None):
        raise ComparisonError("saracura backend requires encoder snapshot and training capsule")
    if backend == "laya" and laya_snapshot is None:
        raise ComparisonError("laya backend requires laya snapshot")
    if backend not in {"saracura", "laya"}:
        raise ComparisonError("unknown backend")
    if device not in {"cpu", "mps"} or not 1 <= cpu_threads <= 8:
        raise ComparisonError("worker execution settings")

    accepted, _plan = _load_comparison_packet(packet)
    work_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    packet_digest = _sha((packet / "packet.json").read_bytes())
    artifact = training_capsule if backend == "saracura" else laya_snapshot
    if artifact is None:
        raise ComparisonError("backend artifact missing")
    artifact_digest, artifact_bytes = _artifact_digest_and_size(artifact)
    module_name = (
        "saracura.backends.saracura_universal"
        if backend == "saracura"
        else "saracura.backends.laya"
    )
    module = importlib.import_module(module_name)
    backend_object: Any | None = None
    cold_start = time.perf_counter()
    prepare_error: SaracuraError | None = None
    try:
        if backend == "saracura":
            backend_object = module.SaracuraUniversalBackend(
                encoder_snapshot=encoder_snapshot,
                training_capsule=training_capsule,
                device=device,
                cpu_threads=cpu_threads,
            )
        else:
            backend_object = module.LayaUniversalBackend(
                model_snapshot=laya_snapshot,
                device=device,
                cpu_threads=cpu_threads,
            )
        backend_object.prepare()
    except SaracuraError as error:
        prepare_error = error
    cold_load_ms = (time.perf_counter() - cold_start) * 1000.0

    rows: list[dict[str, Any]] = []
    try:
        for accepted_row in accepted:
            semantic_sha256 = _task_semantic_sha256(accepted_row)
            if prepare_error is not None or backend_object is None:
                rows.append(
                    {
                        "task_id": accepted_row["task_id"],
                        "semantic_sha256": semantic_sha256,
                        "status": "unsupported_runtime",
                        "selected_criterion_id": None,
                        "stable_error_code": (
                            str(prepare_error.payload.code)
                            if prepare_error is not None
                            else "BACKEND_UNAVAILABLE"
                        ),
                        "input_tokens": None,
                        "warm_latency_ms": None,
                        "deterministic_repeat": True,
                    }
                )
                continue
            request = _accepted_row_to_request(
                accepted_row,
                model_revision=backend_object.model.revision,
                workflow_revision=(
                    COMPARISON_SARACURA_REVISION
                    if backend == "saracura"
                    else COMPARISON_LAYA_REVISION
                ),
            )
            first_start = time.perf_counter()
            expected = cast(str, accepted_row["selected_criterion_id"])
            first = _score_once(module, backend_object, backend, request, expected)
            first_latency_ms = (time.perf_counter() - first_start) * 1000.0
            second = _score_once(module, backend_object, backend, request, expected)
            deterministic = (
                first["status"] == second["status"]
                and first["selected_criterion_id"] == second["selected_criterion_id"]
                and first["stable_error_code"] == second["stable_error_code"]
            )
            rows.append(
                {
                    "task_id": accepted_row["task_id"],
                    "semantic_sha256": semantic_sha256,
                    **first,
                    "warm_latency_ms": first_latency_ms,
                    "deterministic_repeat": deterministic,
                }
            )
    finally:
        if backend_object is not None:
            backend_object.close()

    latencies = [
        cast(float, row["warm_latency_ms"])
        for row in rows
        if isinstance(row.get("warm_latency_ms"), (int, float))
    ]
    warm_seconds = sum(latencies) / 1000.0
    dependency_versions: dict[str, str] = {}
    for distribution in ("saracura", "torch", "transformers", "tokenizers", "safetensors"):
        try:
            dependency_versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            dependency_versions[distribution] = "unavailable"
    module_path = module.__file__
    if not isinstance(module_path, str):
        raise ComparisonError("backend runtime module path unavailable")
    result: dict[str, Any] = {
        "schema_version": "phase4e-comparison-score-backend.v1",
        "backend": backend,
        "device": device,
        "cpu_threads": cpu_threads,
        "packet_sha256": packet_digest,
        "artifact_sha256": artifact_digest,
        "artifact_bytes": artifact_bytes,
        "model_revision": (
            backend_object.model.revision if backend_object is not None else "unavailable"
        ),
        "runtime_module_sha256": _sha(Path(module_path).read_bytes()),
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "dependency_versions": dependency_versions,
        "cold_load_ms": cold_load_ms,
        "warm_p50_ms": _p50(latencies) if latencies else None,
        "warm_p95_ms": _p95(latencies) if latencies else None,
        "throughput_tasks_per_second": len(latencies) / warm_seconds if warm_seconds else None,
        "peak_rss_bytes": _peak_rss_bytes(),
        "deterministic_failures": sum(1 for row in rows if row["deterministic_repeat"] is not True),
        "rows": rows,
    }
    result_path = work_dir / "score-result.json"
    atomic_create(result_path, _canonical(result) + b"\n")
    os.chmod(result_path, 0o600)
    return result_path


def _accepted_row_to_request(
    row: Mapping[str, Any], *, model_revision: str, workflow_revision: str
) -> Any:
    from saracura.contracts.models import ChoiceCriterion as CC
    from saracura.contracts.models import ChoiceQuestion as CQ
    from saracura.contracts.models import DecisionRequest as DR
    from saracura.contracts.models import WorkflowReference as WR

    return DR(
        api_version="v1alpha1",
        model=model_revision,
        mode="research",
        locale=cast(str, row["locale"]),
        domain=cast(str, row["domain"]),
        workflow=WR(id=COMPARISON_WORKFLOW_ID, revision=workflow_revision),
        state=cast(Any, row.get("state", {})),
        questions=(
            CQ(
                id="q1",
                type="choice",
                instruction=cast(str, row["instruction"]),
                criteria=tuple(
                    CC(id=c["id"], description=c["description"])
                    for c in cast(list[dict[str, Any]], row.get("criteria", []))
                ),
            ),
        ),
    )


def _score_once(
    module: Any,
    backend_object: Any,
    backend: str,
    request: Any,
    expected_selected_id: str,
) -> dict[str, Any]:
    from saracura.contracts.errors import ErrorCode, SaracuraError

    question = request.questions[0]
    try:
        if backend == "saracura":
            rendered = render_choice(
                locale=request.locale,
                domain=request.domain,
                instruction=question.instruction,
                state=request.state,
                criteria=question.criteria,
            )
            tokenized = backend_object._tokenize_joint(rendered)
            input_tokens = int(tokenized["attention_mask"].sum().item())
        else:
            rendered_laya = module.render_laya_choice(backend_object._tokenizer, request, question)
            input_tokens = rendered_laya.input_tokens
        scored = backend_object.score_universal_choice(request, question, _canonical(request.state))
        selected = _best_criterion(scored.raw_scores)
        return {
            "status": "correct" if selected == expected_selected_id else "incorrect",
            "selected_criterion_id": selected,
            "stable_error_code": None,
            "input_tokens": input_tokens,
        }
    except SaracuraError as error:
        status = (
            "unsupported_capacity"
            if error.payload.code == ErrorCode.CAPACITY_EXCEEDED
            else "unsupported_runtime"
            if error.payload.code == ErrorCode.BACKEND_UNAVAILABLE
            else "runtime_error"
        )
        return {
            "status": status,
            "selected_criterion_id": None,
            "stable_error_code": str(error.payload.code),
            "input_tokens": None,
        }
    except Exception:
        return {
            "status": "runtime_error",
            "selected_criterion_id": None,
            "stable_error_code": "UNEXPECTED_RUNTIME_ERROR",
            "input_tokens": None,
        }


def _task_semantic_sha256(row: Mapping[str, Any]) -> str:
    return _sha(
        _canonical(
            {
                key: row[key]
                for key in (
                    "locale",
                    "domain",
                    "instruction",
                    "state",
                    "criteria",
                    "selected_criterion_id",
                    "axes",
                    "option_count",
                    "gold_position",
                )
            }
        )
    )


def _artifact_digest_and_size(path: Path) -> tuple[str, int]:
    if path.is_file():
        payload = path.read_bytes()
        return _sha(payload), len(payload)
    if not path.is_dir():
        raise ComparisonError("backend artifact does not exist")
    entries: list[dict[str, Any]] = []
    total = 0
    for child in sorted(path.rglob("*")):
        if child.is_symlink():
            raise ComparisonError("backend artifact contains a symlink")
        if child.is_file():
            payload = child.read_bytes()
            total += len(payload)
            entries.append(
                {
                    "path": child.relative_to(path).as_posix(),
                    "bytes": len(payload),
                    "sha256": _sha(payload),
                }
            )
    return _sha(_canonical(entries)), total


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _best_criterion(scores: Mapping[str, float]) -> str:
    return max(scores, key=lambda k: scores[k])


def _p50(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[len(s) // 2]


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = max(0, int(len(s) * 0.95) - 1)
    return s[idx]


# ── subcommand: evaluate ────────────────────────────────────────────────


def run_evaluate(
    packet: Path,
    work_dir: Path,
    *,
    saracura_encoder_snapshot: Path,
    saracura_training_capsule: Path,
    device: Literal["cpu", "mps"] = "cpu",
    cpu_threads: int = 1,
    laya_snapshot: Path | None = None,
    allow_fixture_evidence: bool = False,
) -> Path:
    _require_component(packet, "comparison-packet", file_path=False)
    _require_component(work_dir, "comparison-eval")
    private = (
        _require_private_root()
        if allow_fixture_evidence
        else _require_live_external_comparison_root()
    )
    _require_within(packet, private)
    _require_within(work_dir, private)
    if device != "cpu":
        raise ComparisonError("the primary comparison device is CPU")
    accepted_rows, plan = _load_comparison_packet(packet)
    minimum_errors = _check_acceptance_minimums(accepted_rows, plan)
    if minimum_errors:
        raise ComparisonError("comparison packet does not meet scoring minimums")
    bindings = cast(Mapping[str, Any], plan["bindings"])
    if bindings["mode"] != "live_verified" and not allow_fixture_evidence:
        raise ComparisonError("evaluation requires a live-verified plan")
    _preverify_evaluation_inputs(
        saracura_encoder_snapshot=saracura_encoder_snapshot,
        saracura_training_capsule=saracura_training_capsule,
        bindings=bindings,
        allow_fixture_evidence=allow_fixture_evidence,
    )
    work_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    saracura_work = work_dir / "saracura-cpu"
    saracura_work.mkdir(mode=0o700, exist_ok=True)
    saracura_manifest = _write_worker_manifest(
        saracura_work / "input-manifest.json",
        backend="saracura",
        packet=packet,
        output_dir=saracura_work,
        device="cpu",
        cpu_threads=cpu_threads,
        encoder_snapshot=saracura_encoder_snapshot,
        training_capsule=saracura_training_capsule,
        laya_snapshot=None,
    )
    sr_path = saracura_work / "score-result.json"
    if not sr_path.exists():
        _run_worker_subprocess(saracura_manifest)
    saracura_data = _load_worker_result(
        sr_path,
        "saracura",
        "cpu",
        cpu_threads,
        accepted_rows,
        packet,
    )
    os.chmod(saracura_manifest, 0o400)
    os.chmod(sr_path, 0o400)
    sr_digest = _sha(sr_path.read_bytes())
    receipt_path = saracura_work / "sealed-result.json"
    receipt = {
        "schema_version": COMPARISON_WORKER_RECEIPT_SCHEMA,
        "backend": "saracura",
        "packet_sha256": _sha((packet / "packet.json").read_bytes()),
        "score_result_sha256": sr_digest,
    }
    _create_or_identical(receipt_path, _canonical(receipt) + b"\n", mode=0o400)
    candidate_receipt_sha256 = _sha(receipt_path.read_bytes())

    control_status = "control_unavailable"
    laya_result: dict[str, Any] | None = None
    control_result_sha256: str | None = None
    control_receipt_sha256: str | None = None
    if laya_snapshot is not None:
        try:
            _preverify_laya_control(
                laya_snapshot,
                bindings=bindings,
                allow_fixture_evidence=allow_fixture_evidence,
            )
        except ComparisonError:
            pass
        else:
            laya_work = work_dir / "laya-cpu"
            laya_work.mkdir(mode=0o700, exist_ok=True)
            laya_manifest = _write_worker_manifest(
                laya_work / "input-manifest.json",
                backend="laya",
                packet=packet,
                output_dir=laya_work,
                device="cpu",
                cpu_threads=cpu_threads,
                encoder_snapshot=None,
                training_capsule=None,
                laya_snapshot=laya_snapshot,
            )
            if str(saracura_work) in _canonical(json.loads(laya_manifest.read_bytes())).decode():
                raise ComparisonError("control manifest references candidate output")
            lr_path = laya_work / "score-result.json"
            if not lr_path.exists():
                _run_worker_subprocess(laya_manifest)
            laya_result = _load_worker_result(
                lr_path,
                "laya",
                "cpu",
                cpu_threads,
                accepted_rows,
                packet,
            )
            os.chmod(laya_manifest, 0o400)
            os.chmod(lr_path, 0o400)
            control_result_sha256 = _sha(lr_path.read_bytes())
            laya_receipt_path = laya_work / "sealed-result.json"
            laya_receipt = {
                "schema_version": COMPARISON_WORKER_RECEIPT_SCHEMA,
                "backend": "laya",
                "packet_sha256": _sha((packet / "packet.json").read_bytes()),
                "score_result_sha256": control_result_sha256,
            }
            _create_or_identical(laya_receipt_path, _canonical(laya_receipt) + b"\n", mode=0o400)
            control_receipt_sha256 = _sha(laya_receipt_path.read_bytes())
            control_status = "available"

    candidate_summary = _aggregate_worker_result(saracura_data, accepted_rows)
    primary: dict[str, Any] = {
        "device": "cpu",
        "cpu_threads": cpu_threads,
        "control_backend": control_status,
        "candidate_result_sha256": sr_digest,
        "control_result_sha256": control_result_sha256,
        "candidate": candidate_summary,
        "control": None,
        "paired_2x2": None,
    }
    if laya_result is not None:
        primary["control"] = _aggregate_worker_result(laya_result, accepted_rows)
        candidate_rows = {row["task_id"]: row for row in saracura_data["rows"]}
        control_rows = {row["task_id"]: row for row in laya_result["rows"]}
        paired = {
            "both_correct": 0,
            "candidate_only_correct": 0,
            "control_only_correct": 0,
            "both_not_correct": 0,
        }
        for task_id in candidate_rows:
            candidate_correct = candidate_rows[task_id]["status"] == "correct"
            control_correct = control_rows[task_id]["status"] == "correct"
            if candidate_correct and control_correct:
                paired["both_correct"] += 1
            elif candidate_correct:
                paired["candidate_only_correct"] += 1
            elif control_correct:
                paired["control_only_correct"] += 1
            else:
                paired["both_not_correct"] += 1
        primary["paired_2x2"] = paired

    provenance = {
        key: bindings[key]
        for key in (
            "source_commit",
            "training_packet_receipt_sha256",
            "training_accepted_rows_sha256",
            "training_manifest_sha256",
            "training_checkpoint_sha256",
            "saracura_tokenizer_snapshot_sha256",
            "laya_snapshot_sha256",
            "training_task_ids_sha256",
            "training_family_ids_sha256",
        )
    }
    evidence_mode = "fixture" if allow_fixture_evidence else "live_verified"
    result: dict[str, Any] = {
        "schema_version": "phase4e-comparison-result.v1",
        "status": "scored" if laya_result is not None else "control_unavailable",
        "evidence_mode": evidence_mode,
        "plan_sha256": _sha((packet / "plan.json").read_bytes()),
        "packet_sha256": _sha((packet / "packet.json").read_bytes()),
        "policy_sha256": _sha(COMPARISON_POLICY_PATH.read_bytes()),
        "provenance": provenance,
        "provider_reported_generation_cost_usd": _packet_generation_cost(packet),
        "primary_comparison": primary,
        "candidate_device_observations": [],
    }
    result_path = work_dir / (
        "eval-result.json" if laya_result is not None else "control-unavailable.json"
    )
    _create_or_identical(result_path, _canonical(result) + b"\n", mode=0o400)
    evaluation_receipt = {
        "schema_version": COMPARISON_EVALUATION_RECEIPT_SCHEMA,
        "evidence_mode": evidence_mode,
        "result_sha256": _sha(result_path.read_bytes()),
        "plan_sha256": result["plan_sha256"],
        "packet_sha256": result["packet_sha256"],
        "candidate_result_sha256": sr_digest,
        "candidate_receipt_sha256": candidate_receipt_sha256,
        "control_result_sha256": control_result_sha256,
        "control_receipt_sha256": control_receipt_sha256,
    }
    _create_or_identical(
        work_dir
        / (
            "evaluation-receipt.json"
            if laya_result is not None
            else "control-unavailable-receipt.json"
        ),
        _canonical(evaluation_receipt) + b"\n",
        mode=0o400,
    )
    return result_path


def _preverify_evaluation_inputs(
    *,
    saracura_encoder_snapshot: Path,
    saracura_training_capsule: Path,
    bindings: Mapping[str, Any],
    allow_fixture_evidence: bool,
) -> None:
    """Verify every explicit artifact in the parent before any worker starts."""

    try:
        from benchmarks.saracura_universal_training import verify_training_capsule

        if bindings.get("mode") != "live_verified":
            if allow_fixture_evidence:
                return
            raise ComparisonError("evaluation requires a live-verified plan")
        manifest = verify_training_capsule(saracura_training_capsule)
        if manifest.get("outcome") != "passed":
            raise ComparisonError("Saracura training capsule is not a passed sealed result")
        corpus.VerifiedMiniLMTokenizerReceipt.create(saracura_encoder_snapshot)
        encoder_digest, _encoder_bytes = _artifact_digest_and_size(saracura_encoder_snapshot)
        if (
            manifest.get("manifest_sha256") != bindings.get("training_manifest_sha256")
            or manifest.get("checkpoint_sha256") != bindings.get("training_checkpoint_sha256")
            or encoder_digest != bindings.get("saracura_tokenizer_snapshot_sha256")
        ):
            raise ComparisonError("evaluation artifacts do not match the sealed plan")
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0 or completed.stdout.strip() != bindings.get("source_commit"):
            raise ComparisonError("evaluation source commit does not match the sealed plan")
    except ComparisonError:
        raise
    except Exception as error:
        raise ComparisonError("evaluation artifact verification failed") from error


def _preverify_laya_control(
    laya_snapshot: Path,
    *,
    bindings: Mapping[str, Any],
    allow_fixture_evidence: bool,
) -> None:
    if bindings.get("mode") != "live_verified":
        if allow_fixture_evidence:
            return
        raise ComparisonError("evaluation requires a live-verified plan")
    try:
        from saracura.laya_snapshot import verify_laya_snapshot

        receipt = verify_laya_snapshot(laya_snapshot)
        receipt.close()
        laya_digest, _laya_bytes = _artifact_digest_and_size(laya_snapshot)
    except Exception as error:
        raise ComparisonError("Laya control verification failed") from error
    if laya_digest != bindings.get("laya_snapshot_sha256"):
        raise ComparisonError("evaluation control does not match the sealed plan")


def _write_worker_manifest(
    path: Path,
    *,
    backend: Literal["saracura", "laya"],
    packet: Path,
    output_dir: Path,
    device: Literal["cpu", "mps"],
    cpu_threads: int,
    encoder_snapshot: Path | None,
    training_capsule: Path | None,
    laya_snapshot: Path | None,
) -> Path:
    payload = {
        "schema_version": "phase4e-comparison-worker-input.v1",
        "backend": backend,
        "packet": str(packet.resolve()),
        "output_dir": str(output_dir.resolve()),
        "device": device,
        "cpu_threads": cpu_threads,
        "encoder_snapshot": str(encoder_snapshot.resolve()) if encoder_snapshot else None,
        "training_capsule": str(training_capsule.resolve()) if training_capsule else None,
        "laya_snapshot": str(laya_snapshot.resolve()) if laya_snapshot else None,
    }
    _create_or_identical(path, _canonical(payload) + b"\n", mode=0o600)
    return path


def run_score_backend_manifest(manifest_path: Path) -> Path:
    try:
        manifest = json.loads(manifest_path.read_bytes(), object_pairs_hook=_no_duplicate_keys)
    except (OSError, ValueError) as error:
        raise ComparisonError("worker input manifest") from error
    expected = {
        "schema_version",
        "backend",
        "packet",
        "output_dir",
        "device",
        "cpu_threads",
        "encoder_snapshot",
        "training_capsule",
        "laya_snapshot",
    }
    if not isinstance(manifest, dict) or set(manifest) != expected:
        raise ComparisonError("worker input manifest")
    if manifest["schema_version"] != "phase4e-comparison-worker-input.v1":
        raise ComparisonError("worker input manifest")
    backend = manifest["backend"]
    if backend not in {"saracura", "laya"}:
        raise ComparisonError("worker input backend")
    return run_score_backend(
        Path(manifest["packet"]),
        backend,
        Path(manifest["output_dir"]),
        device=manifest["device"],
        cpu_threads=manifest["cpu_threads"],
        encoder_snapshot=(
            Path(manifest["encoder_snapshot"]) if manifest["encoder_snapshot"] else None
        ),
        training_capsule=(
            Path(manifest["training_capsule"]) if manifest["training_capsule"] else None
        ),
        laya_snapshot=(Path(manifest["laya_snapshot"]) if manifest["laya_snapshot"] else None),
    )


def _run_worker_subprocess(manifest_path: Path) -> None:
    environment = {
        key: value
        for key, value in os.environ.items()
        if not any(marker in key.upper() for marker in ("KEY", "TOKEN", "SECRET", "PASSWORD"))
    }
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.saracura_universal_comparison",
            "score-backend",
            "--input-manifest",
            str(manifest_path),
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
    )
    if completed.returncode != 0:
        raise ComparisonError("backend worker failed")


def _load_worker_result(
    path: Path,
    backend: str,
    device: str,
    cpu_threads: int,
    accepted_rows: Sequence[Mapping[str, Any]],
    packet: Path,
) -> dict[str, Any]:
    try:
        result = json.loads(path.read_bytes(), object_pairs_hook=_no_duplicate_keys)
    except (OSError, ValueError) as error:
        raise ComparisonError("worker result") from error
    required = {
        "schema_version",
        "backend",
        "device",
        "cpu_threads",
        "packet_sha256",
        "artifact_sha256",
        "artifact_bytes",
        "model_revision",
        "runtime_module_sha256",
        "platform",
        "dependency_versions",
        "cold_load_ms",
        "warm_p50_ms",
        "warm_p95_ms",
        "throughput_tasks_per_second",
        "peak_rss_bytes",
        "deterministic_failures",
        "rows",
    }
    if (
        not isinstance(result, dict)
        or set(result) != required
        or result.get("schema_version") != "phase4e-comparison-score-backend.v1"
        or result.get("backend") != backend
        or result.get("device") != device
        or result.get("cpu_threads") != cpu_threads
        or result.get("packet_sha256") != _sha((packet / "packet.json").read_bytes())
        or not isinstance(result.get("rows"), list)
    ):
        raise ComparisonError("worker result binding")
    expected = {cast(str, row["task_id"]): row for row in accepted_rows}
    actual_rows = cast(list[dict[str, Any]], result["rows"])
    if len(actual_rows) != len(expected) or len({row.get("task_id") for row in actual_rows}) != len(
        expected
    ):
        raise ComparisonError("worker result coverage")
    for row in actual_rows:
        task_id = row.get("task_id")
        status_value = row.get("status")
        selected = row.get("selected_criterion_id")
        stable_error = row.get("stable_error_code")
        expected_row_keys = {
            "task_id",
            "semantic_sha256",
            "status",
            "selected_criterion_id",
            "stable_error_code",
            "input_tokens",
            "warm_latency_ms",
            "deterministic_repeat",
        }
        if (
            not isinstance(row, dict)
            or set(row) != expected_row_keys
            or task_id not in expected
            or row.get("semantic_sha256") != _task_semantic_sha256(expected[cast(str, task_id)])
            or status_value not in COMPARISON_TASK_STATUSES
            or not isinstance(row.get("deterministic_repeat"), bool)
            or (
                status_value in {"correct", "incorrect"}
                and (not isinstance(selected, str) or stable_error is not None)
            )
            or (
                status_value not in {"correct", "incorrect"}
                and (selected is not None or not isinstance(stable_error, str))
            )
        ):
            raise ComparisonError("worker result task binding")
    return result


def _aggregate_worker_result(
    result: Mapping[str, Any], accepted_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    results_by_id = {
        cast(str, row["task_id"]): row for row in cast(Sequence[Mapping[str, Any]], result["rows"])
    }

    def cell(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        correct = sum(
            results_by_id[cast(str, row["task_id"])]["status"] == "correct" for row in rows
        )
        return _build_slice_result(correct, len(rows))

    slices: dict[str, Any] = {
        "locale": {},
        "domain": {},
        "scenario": {},
        "option_count": {},
        "axes": {axis: {} for axis in AXES},
    }
    dimensions: dict[str, Callable[[Mapping[str, Any]], object]] = {
        "locale": lambda row: row["locale"],
        "domain": lambda row: row["domain"],
        "scenario": lambda row: row["semantic_equivalence_attestation"]["scenario"],
        "option_count": lambda row: str(row["option_count"]),
    }
    for name, selector in dimensions.items():
        values = sorted({cast(str, selector(row)) for row in accepted_rows})
        slices[name] = {
            value: cell([row for row in accepted_rows if str(selector(row)) == value])
            for value in values
        }
    for axis in AXES:
        values = sorted({cast(str, row["axes"][axis]) for row in accepted_rows})
        slices["axes"][axis] = {
            value: cell([row for row in accepted_rows if row["axes"][axis] == value])
            for value in values
        }
    rows = cast(Sequence[Mapping[str, Any]], result["rows"])
    return {
        "planned": 200,
        "resolved": 200,
        "accepted": len(accepted_rows),
        "rejected": 200 - len(accepted_rows),
        "unsupported": sum(str(row["status"]).startswith("unsupported_") for row in rows),
        "failed": sum(row["status"] == "runtime_error" for row in rows),
        "accuracy": cell(accepted_rows),
        "slices": slices,
        "performance": {
            key: result.get(key)
            for key in (
                "cold_load_ms",
                "warm_p50_ms",
                "warm_p95_ms",
                "throughput_tasks_per_second",
                "peak_rss_bytes",
                "artifact_bytes",
            )
        },
        "deterministic_failures": result.get("deterministic_failures"),
        "artifact_sha256": result.get("artifact_sha256"),
        "model_revision": result.get("model_revision"),
        "runtime_module_sha256": result.get("runtime_module_sha256"),
        "dependency_versions": result.get("dependency_versions"),
        "platform": result.get("platform"),
    }


def _packet_generation_cost(packet: Path) -> str:
    try:
        ledger = json.loads((packet / "ledger.json").read_bytes())
        return str(
            sum(
                (Decimal(entry["provider_cost_usd"]) for entry in ledger["entries"]),
                Decimal(),
            )
        )
    except (OSError, ValueError, KeyError, ArithmeticError) as error:
        raise ComparisonError("comparison packet ledger") from error


def _create_or_identical(path: Path, payload: bytes, *, mode: int) -> Path:
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"comparison artifact already exists: {path.name}")
        return path
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    atomic_create(path, payload)
    os.chmod(path, mode)
    return path


# ── subcommand: publish ─────────────────────────────────────────────────


def _load_comparison_packet(packet: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    packet_json = packet / "packet.json"
    if not packet_json.exists():
        raise ComparisonError("packet.json not found")
    try:
        descriptor = json.loads(packet_json.read_bytes(), object_pairs_hook=_no_duplicate_keys)
    except (OSError, ValueError) as error:
        raise ComparisonError("comparison packet descriptor") from error
    expected_keys = {"schema_version", "plan_sha256", "accepted_count", "rejected_count", "files"}
    if (
        not isinstance(descriptor, dict)
        or set(descriptor) != expected_keys
        or descriptor["schema_version"] != "phase4e-comparison-packet.v1"
        or not isinstance(descriptor["files"], dict)
    ):
        raise ComparisonError("comparison packet descriptor")
    expected_files = {"plan.json", "accepted.jsonl", "rejected.jsonl", "ledger.json"}
    if set(descriptor["files"]) != expected_files:
        raise ComparisonError("comparison packet files")
    payloads: dict[str, bytes] = {}
    for name in expected_files:
        path = packet / name
        try:
            payload = path.read_bytes()
        except OSError as error:
            raise ComparisonError("comparison packet file missing") from error
        if _sha(payload) != descriptor["files"][name]:
            raise ComparisonError("comparison packet digest")
        payloads[name] = payload
    if _sha(payloads["plan.json"]) != descriptor["plan_sha256"]:
        raise ComparisonError("comparison plan binding")
    try:
        plan = cast(
            dict[str, Any], json.loads(payloads["plan.json"], object_pairs_hook=_no_duplicate_keys)
        )
    except ValueError as error:
        raise ComparisonError("comparison packet plan") from error
    _validate_comparison_plan(plan)
    accepted = _read_jsonl(packet / "accepted.jsonl")
    rejected = _read_jsonl(packet / "rejected.jsonl")
    if (
        len(accepted) != descriptor["accepted_count"]
        or len(rejected) != descriptor["rejected_count"]
    ):
        raise ComparisonError("comparison packet counts")
    _validate_packet_rows(accepted, rejected, plan)
    return accepted, plan


def _validate_packet_rows(
    accepted: Sequence[Mapping[str, Any]],
    rejected: Sequence[Mapping[str, Any]],
    plan: Mapping[str, Any],
) -> None:
    slots = {
        cast(str, slot["task_id"]): slot
        for slot in cast(Sequence[Mapping[str, Any]], plan["slots"])
    }
    accepted_ids = [row.get("task_id") for row in accepted]
    rejected_ids = [row.get("task_id") for row in rejected]
    all_ids = [*accepted_ids, *rejected_ids]
    if (
        not all(isinstance(task_id, str) for task_id in all_ids)
        or len(set(all_ids)) != len(all_ids)
        or set(all_ids) != set(slots)
    ):
        raise ComparisonError("comparison packet task coverage")
    for row in rejected:
        task_id = cast(str, row["task_id"])
        if row.get("split") != slots[task_id]["split"] or not isinstance(row.get("reason"), str):
            raise ComparisonError("comparison rejected-row binding")
    for row in accepted:
        task_id = cast(str, row["task_id"])
        slot = slots[task_id]
        expected_fields = {
            "family_id": slot["family_id"],
            "pair_id": slot["pair_id"],
            "split": slot["split"],
            "locale": slot["locale"],
            "domain": slot["domain"],
            "axes": slot["axes"],
            "option_count": slot["option_count"],
            "gold_position": slot["gold_position"],
        }
        if any(row.get(key) != value for key, value in expected_fields.items()):
            raise ComparisonError("comparison accepted-row slot binding")
        criteria = row.get("criteria")
        option_count = cast(int, slot["option_count"])
        expected_ids = [f"criterion-{index}" for index in range(option_count)]
        if (
            not isinstance(criteria, list)
            or len(criteria) != option_count
            or [criterion.get("id") for criterion in criteria if isinstance(criterion, Mapping)]
            != expected_ids
            or any(
                not isinstance(criterion, Mapping)
                or not isinstance(criterion.get("description"), str)
                for criterion in criteria
            )
            or row.get("selected_criterion_id") != expected_ids[cast(int, slot["gold_position"])]
        ):
            raise ComparisonError("comparison accepted-row option binding")
        target = cast(Mapping[str, Any], slot["semantic_target"])
        expected_roles = list(cast(Sequence[str], target["criterion_roles"]))
        selected_role = expected_roles.pop(0)
        expected_roles.insert(cast(int, slot["gold_position"]), selected_role)
        attestation = row.get("semantic_equivalence_attestation")
        if attestation != {
            "scenario": target["scenario"],
            "criterion_roles": expected_roles,
            "selected_role": "matches_rule",
        }:
            raise ComparisonError("comparison accepted-row semantic binding")


def _verify_evaluation_receipt(
    receipt_path: Path,
    result_path: Path,
    result_bytes: bytes,
    result: Mapping[str, Any],
) -> str:
    try:
        receipt = json.loads(receipt_path.read_bytes(), object_pairs_hook=_no_duplicate_keys)
    except (OSError, ValueError) as error:
        raise ComparisonError("live evaluation receipt is required") from error
    required = {
        "schema_version",
        "evidence_mode",
        "result_sha256",
        "plan_sha256",
        "packet_sha256",
        "candidate_result_sha256",
        "candidate_receipt_sha256",
        "control_result_sha256",
        "control_receipt_sha256",
    }
    primary = result.get("primary_comparison")
    if (
        not isinstance(receipt, dict)
        or set(receipt) != required
        or receipt.get("schema_version") != COMPARISON_EVALUATION_RECEIPT_SCHEMA
        or receipt.get("evidence_mode") != "live_verified"
        or result.get("evidence_mode") != "live_verified"
        or receipt.get("result_sha256") != _sha(result_bytes)
        or receipt.get("plan_sha256") != result.get("plan_sha256")
        or receipt.get("packet_sha256") != result.get("packet_sha256")
        or not isinstance(primary, Mapping)
        or receipt.get("candidate_result_sha256") != primary.get("candidate_result_sha256")
        or receipt.get("control_result_sha256") != primary.get("control_result_sha256")
        or not _is_digest(receipt.get("candidate_receipt_sha256"))
        or not _is_digest(receipt.get("control_receipt_sha256"))
    ):
        raise ComparisonError("live evaluation receipt binding")

    if receipt_path.resolve().parent != result_path.resolve().parent:
        raise ComparisonError("live evaluation receipt must be colocated with its result")
    if result_bytes != _canonical(dict(result)) + b"\n":
        raise ComparisonError("live evaluation result bytes are not canonical")
    sealed_paths = (receipt_path, result_path)
    if any(stat.S_IMODE(path.stat().st_mode) != 0o400 for path in sealed_paths):
        raise ComparisonError("live evaluation evidence is not sealed read-only")

    for backend, result_digest_key, receipt_digest_key in (
        ("saracura", "candidate_result_sha256", "candidate_receipt_sha256"),
        ("laya", "control_result_sha256", "control_receipt_sha256"),
    ):
        worker_root = result_path.parent / f"{backend}-cpu"
        worker_result_path = worker_root / "score-result.json"
        worker_receipt_path = worker_root / "sealed-result.json"
        try:
            worker_result_bytes = worker_result_path.read_bytes()
            worker_receipt_bytes = worker_receipt_path.read_bytes()
            worker_receipt = json.loads(worker_receipt_bytes, object_pairs_hook=_no_duplicate_keys)
        except (OSError, ValueError) as error:
            raise ComparisonError("sealed worker evidence is required") from error
        if (
            stat.S_IMODE(worker_result_path.stat().st_mode) != 0o400
            or stat.S_IMODE(worker_receipt_path.stat().st_mode) != 0o400
            or not isinstance(worker_receipt, dict)
            or set(worker_receipt)
            != {"schema_version", "backend", "packet_sha256", "score_result_sha256"}
            or worker_receipt.get("schema_version") != COMPARISON_WORKER_RECEIPT_SCHEMA
            or worker_receipt.get("backend") != backend
            or worker_receipt.get("packet_sha256") != result.get("packet_sha256")
            or worker_receipt.get("score_result_sha256") != _sha(worker_result_bytes)
            or _sha(worker_result_bytes) != receipt[result_digest_key]
            or _sha(worker_receipt_bytes) != receipt[receipt_digest_key]
        ):
            raise ComparisonError("sealed worker evidence binding")
    return _sha(receipt_path.read_bytes())


def run_publish(
    result_path: Path,
    publish_json: Path,
    publish_md: Path,
    *,
    evaluation_receipt: Path | None = None,
    readme_en: Path = README_EN_PATH,
    readme_pt_br: Path = README_PT_BR_PATH,
) -> tuple[Path, Path]:
    _require_public_destination(
        publish_json, ("benchmarks", "results"), "phase4e-comparison-v1.json"
    )
    _require_public_destination(publish_md, ("docs", "action"), "phase4e-comparison-result.md")
    try:
        result_bytes = result_path.read_bytes()
        private_raw = json.loads(result_bytes, object_pairs_hook=_no_duplicate_keys)
    except (OSError, ValueError) as error:
        raise ComparisonError("private comparison result") from error
    required = {
        "schema_version",
        "status",
        "evidence_mode",
        "plan_sha256",
        "packet_sha256",
        "policy_sha256",
        "provenance",
        "provider_reported_generation_cost_usd",
        "primary_comparison",
        "candidate_device_observations",
    }
    if (
        not isinstance(private_raw, dict)
        or set(private_raw) != required
        or private_raw["schema_version"] != "phase4e-comparison-result.v1"
        or private_raw["status"] != "scored"
    ):
        raise ComparisonError("private comparison result is not publishable")
    receipt_sha256 = _verify_evaluation_receipt(
        evaluation_receipt or result_path.with_name("evaluation-receipt.json"),
        result_path,
        result_bytes,
        private_raw,
    )
    _validate_publishable_result(private_raw)
    projected: dict[str, Any] = {
        "schema_version": "phase4e-comparison-v1",
        "status": private_raw["status"],
        "evidence_mode": private_raw["evidence_mode"],
        "evaluation_receipt_sha256": receipt_sha256,
        "plan_sha256": private_raw["plan_sha256"],
        "packet_sha256": private_raw["packet_sha256"],
        "policy_sha256": private_raw["policy_sha256"],
        "provenance": private_raw["provenance"],
        "provider_reported_generation_cost_usd": private_raw[
            "provider_reported_generation_cost_usd"
        ],
        "primary_comparison": private_raw["primary_comparison"],
        "candidate_device_observations": private_raw["candidate_device_observations"],
        "limitations": [
            "Synthetic comparison records share the training author/reviewer model family.",
            "The common workload is restricted to Saracura's 128-context and "
            "96-criterion-token envelope, not Laya's larger envelope.",
            "This is descriptive research evidence, not a superiority, calibration, "
            "production, or automation claim.",
        ],
    }
    _assert_public_safe(projected)
    projected_bytes = _canonical(projected) + b"\n"
    _create_or_identical(publish_json, projected_bytes, mode=0o400)
    markdown = _comparison_markdown(projected, _sha(projected_bytes)).encode("utf-8")
    _create_or_identical(publish_md, markdown, mode=0o400)
    primary = cast(Mapping[str, Any], projected["primary_comparison"])
    candidate = cast(Mapping[str, Any], primary["candidate"])
    control = cast(Mapping[str, Any], primary["control"])
    candidate_accuracy = cast(Mapping[str, Any], candidate["accuracy"])
    control_accuracy = cast(Mapping[str, Any], control["accuracy"])
    _publish_bilingual_status(
        readme_en=readme_en,
        readme_pt_br=readme_pt_br,
        binding_kind="scored",
        binding_sha256=_sha(projected_bytes),
        body_en=(
            "The Phase 4E.4 live blind comparison is complete. "
            f"Saracura scored {candidate_accuracy['correct']}/"
            f"{candidate_accuracy['denominator']} and Laya scored "
            f"{control_accuracy['correct']}/{control_accuracy['denominator']} on the "
            "same accepted synthetic records. See the "
            "[aggregate result](docs/action/phase4e-comparison-result.md). This remains "
            "synthetic research evidence and does not authorize automation."
        ),
        body_pt_br=(
            "A comparação cega live da Fase 4E.4 foi concluída. "
            f"O Saracura acertou {candidate_accuracy['correct']}/"
            f"{candidate_accuracy['denominator']} e o Laya acertou "
            f"{control_accuracy['correct']}/{control_accuracy['denominator']} nos mesmos "
            "registros sintéticos aceitos. Veja o "
            "[resultado agregado](action/phase4e-comparison-result.md). Isso permanece "
            "evidência de pesquisa sintética e não autoriza automação."
        ),
    )
    return publish_json, publish_md


def _is_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_accuracy_cell(value: object) -> None:
    required = {
        "denominator",
        "correct",
        "accuracy",
        "wilson_95_lower",
        "wilson_95_upper",
        "underpowered",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ComparisonError("publishable accuracy cell")
    denominator = value["denominator"]
    correct = value["correct"]
    numbers = (value["accuracy"], value["wilson_95_lower"], value["wilson_95_upper"])
    if (
        not isinstance(denominator, int)
        or isinstance(denominator, bool)
        or denominator < 0
        or not isinstance(correct, int)
        or isinstance(correct, bool)
        or not 0 <= correct <= denominator
        or not all(isinstance(item, (int, float)) and math.isfinite(item) for item in numbers)
        or not all(0.0 <= float(item) <= 1.0 for item in numbers)
        or not isinstance(value["underpowered"], bool)
    ):
        raise ComparisonError("publishable accuracy cell")
    expected = _build_slice_result(correct, denominator)
    if value != expected:
        raise ComparisonError("publishable accuracy cell is inconsistent")


def _validate_public_label(value: object, *, allow_empty: bool = False) -> None:
    if (
        not isinstance(value, str)
        or (not allow_empty and not value)
        or len(value) > 160
        or any(character in value for character in ("/", "\\", "\0", "\n", "\r"))
        or not re.fullmatch(r"[A-Za-z0-9 ._:+()=-]*", value)
    ):
        raise ComparisonError("publishable label")


def _validate_slice_tree(summary: Mapping[str, Any]) -> None:
    slices = summary.get("slices")
    if not isinstance(slices, dict) or set(slices) != {
        "locale",
        "domain",
        "scenario",
        "option_count",
        "axes",
    }:
        raise ComparisonError("publishable slices")
    for dimension in ("locale", "domain", "scenario", "option_count"):
        cells = slices[dimension]
        if not isinstance(cells, dict):
            raise ComparisonError("publishable slices")
        for cell in cells.values():
            _validate_accuracy_cell(cell)
    axes = slices["axes"]
    if not isinstance(axes, dict) or set(axes) != set(AXES):
        raise ComparisonError("publishable axes")
    for cells in axes.values():
        if not isinstance(cells, dict):
            raise ComparisonError("publishable axes")
        for cell in cells.values():
            _validate_accuracy_cell(cell)


def _validate_backend_summary(summary: object) -> None:
    required = {
        "planned",
        "resolved",
        "accepted",
        "rejected",
        "unsupported",
        "failed",
        "accuracy",
        "slices",
        "performance",
        "deterministic_failures",
        "artifact_sha256",
        "model_revision",
        "runtime_module_sha256",
        "dependency_versions",
        "platform",
    }
    if not isinstance(summary, dict) or set(summary) != required:
        raise ComparisonError("publishable backend summary")
    if any(
        not isinstance(summary[key], int) or isinstance(summary[key], bool) or summary[key] < 0
        for key in (
            "planned",
            "resolved",
            "accepted",
            "rejected",
            "unsupported",
            "failed",
            "deterministic_failures",
        )
    ):
        raise ComparisonError("publishable backend counts")
    if not _is_digest(summary["artifact_sha256"]) or not _is_digest(
        summary["runtime_module_sha256"]
    ):
        raise ComparisonError("publishable backend digest")
    _validate_public_label(summary["model_revision"])
    if not isinstance(summary["dependency_versions"], dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in summary["dependency_versions"].items()
    ):
        raise ComparisonError("publishable dependencies")
    for key, dependency_version in summary["dependency_versions"].items():
        _validate_public_label(key)
        _validate_public_label(dependency_version)
    platform_value = summary["platform"]
    if (
        not isinstance(platform_value, dict)
        or set(platform_value) != {"system", "machine", "python"}
        or not all(isinstance(item, str) for item in platform_value.values())
    ):
        raise ComparisonError("publishable platform")
    for platform_label in platform_value.values():
        _validate_public_label(platform_label)
    performance = summary["performance"]
    if not isinstance(performance, dict) or set(performance) != {
        "cold_load_ms",
        "warm_p50_ms",
        "warm_p95_ms",
        "throughput_tasks_per_second",
        "peak_rss_bytes",
        "artifact_bytes",
    }:
        raise ComparisonError("publishable performance")
    if not all(
        value is None
        or (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and value >= 0
        )
        for value in performance.values()
    ):
        raise ComparisonError("publishable performance")
    _validate_accuracy_cell(summary["accuracy"])
    if (
        summary["planned"] != 200
        or summary["resolved"] != summary["planned"]
        or summary["accepted"] + summary["rejected"] != summary["resolved"]
        or summary["accuracy"]["denominator"] != summary["accepted"]
        or summary["unsupported"] + summary["failed"] > summary["accepted"]
        or summary["deterministic_failures"] > summary["accepted"]
    ):
        raise ComparisonError("publishable backend counts are inconsistent")
    _validate_slice_tree(summary)


def _validate_publishable_result(value: Mapping[str, Any]) -> None:
    if value.get("evidence_mode") != "live_verified":
        raise ComparisonError("publishable result requires live evidence")
    if not all(_is_digest(value[key]) for key in ("plan_sha256", "packet_sha256", "policy_sha256")):
        raise ComparisonError("publishable result digests")
    provenance = value.get("provenance")
    provenance_fields = {
        "source_commit",
        "training_packet_receipt_sha256",
        "training_accepted_rows_sha256",
        "training_manifest_sha256",
        "training_checkpoint_sha256",
        "saracura_tokenizer_snapshot_sha256",
        "laya_snapshot_sha256",
        "training_task_ids_sha256",
        "training_family_ids_sha256",
    }
    if (
        not isinstance(provenance, dict)
        or set(provenance) != provenance_fields
        or not isinstance(provenance.get("source_commit"), str)
        or re.fullmatch(r"[0-9a-f]{40}", cast(str, provenance.get("source_commit"))) is None
        or not all(_is_digest(provenance[field]) for field in provenance_fields - {"source_commit"})
    ):
        raise ComparisonError("publishable provenance")
    try:
        provider_cost = Decimal(str(value["provider_reported_generation_cost_usd"]))
        if not provider_cost.is_finite() or provider_cost < 0:
            raise ArithmeticError
    except (ArithmeticError, ValueError):
        raise ComparisonError("publishable provider cost") from None
    primary = value["primary_comparison"]
    if (
        not isinstance(primary, dict)
        or set(primary)
        != {
            "device",
            "cpu_threads",
            "control_backend",
            "candidate_result_sha256",
            "control_result_sha256",
            "candidate",
            "control",
            "paired_2x2",
        }
        or primary["device"] != "cpu"
        or primary["control_backend"] != "available"
        or not isinstance(primary["cpu_threads"], int)
        or isinstance(primary["cpu_threads"], bool)
        or not 1 <= primary["cpu_threads"] <= 8
        or not _is_digest(primary["candidate_result_sha256"])
        or not _is_digest(primary["control_result_sha256"])
    ):
        raise ComparisonError("publishable primary comparison")
    _validate_backend_summary(primary["candidate"])
    _validate_backend_summary(primary["control"])
    paired = primary["paired_2x2"]
    paired_fields = {
        "both_correct",
        "candidate_only_correct",
        "control_only_correct",
        "both_not_correct",
    }
    if (
        not isinstance(paired, dict)
        or set(paired) != paired_fields
        or any(
            not isinstance(item, int) or isinstance(item, bool) or item < 0
            for item in paired.values()
        )
    ):
        raise ComparisonError("publishable paired outcomes")
    candidate = cast(Mapping[str, Any], primary["candidate"])
    control = cast(Mapping[str, Any], primary["control"])
    paired_total = sum(cast(int, item) for item in paired.values())
    if (
        paired_total != candidate["accepted"]
        or paired_total != control["accepted"]
        or candidate["accuracy"]["correct"]
        != paired["both_correct"] + paired["candidate_only_correct"]
        or control["accuracy"]["correct"] != paired["both_correct"] + paired["control_only_correct"]
    ):
        raise ComparisonError("publishable paired outcomes are inconsistent")
    observations = value["candidate_device_observations"]
    if observations != []:
        raise ComparisonError("candidate device observations are not implemented")


def _assert_public_safe(value: object) -> None:
    forbidden_keys = {
        "instruction",
        "state",
        "criteria",
        "selected_criterion_id",
        "raw_scores",
        "logits",
        "tensor",
        "reservation_id",
        "request_id",
        "response_id",
        "path",
        "api_key",
        "secret",
        "password",
        "credential",
        "username",
        "home",
        "volume",
    }
    if isinstance(value, dict):
        for key, child in value.items():
            folded = key.casefold()
            if folded in forbidden_keys or any(marker in folded for marker in forbidden_keys):
                raise ComparisonError("public projection contains a forbidden field")
            _assert_public_safe(child)
        return
    if isinstance(value, list):
        for child in value:
            _assert_public_safe(child)
        return
    if isinstance(value, str):
        if (
            PUBLISH_FORBIDDEN_PATTERNS["absolute_path"].search(value)
            or value.startswith(("/", "\\\\"))
            or re.match(r"^[A-Za-z]:[\\/]", value) is not None
        ):
            raise ComparisonError("public projection contains forbidden content")
        if re.search(r"\b(?:reservation|request|response)-[A-Za-z0-9._-]+", value):
            raise ComparisonError("public projection contains forbidden content")
        if value.startswith(("sk-", "op://")):
            raise ComparisonError("public projection contains credential-like content")


def _comparison_markdown(projected: Mapping[str, Any], json_sha256: str) -> str:
    primary = cast(Mapping[str, Any], projected["primary_comparison"])
    candidate = cast(Mapping[str, Any], primary["candidate"])
    control = cast(Mapping[str, Any], primary["control"])
    candidate_accuracy = cast(Mapping[str, Any], candidate["accuracy"])
    control_accuracy = cast(Mapping[str, Any], control["accuracy"])
    paired = cast(Mapping[str, Any], primary["paired_2x2"])
    return (
        "---\n"
        "title: Phase 4E.4 blind comparison result\n"
        "kind: report\n"
        "area: development\n"
        "project: saracura\n"
        "collection: saracura\n"
        "owner: saracura-maintainers\n"
        "status: current\n"
        "canonical: docs/action/phase4e-comparison-result.md\n"
        "globalRef: qmd://saracura/docs/action/phase4e-comparison-result.md\n"
        "reviewCadenceDays: 30\n"
        "lastReviewedAt: 2026-09-26\n"
        "sourceRefs:\n"
        "  - benchmarks/results/phase4e-comparison-v1.json\n"
        "related:\n"
        "  - docs/action/specs/phase4e4-blind-comparison-and-public-evidence.md\n"
        "  - README.md\n"
        "  - docs/README.pt-BR.md\n"
        "supersedes: []\n"
        "supersededBy: []\n"
        "sensitivity: public\n"
        "---\n"
        "# Phase 4E.4 blind comparison\n\n"
        f"Status: `{projected['status']}`. Primary cell: CPU, "
        f"{primary['cpu_threads']} thread(s).\n\n"
        f"Saracura: {candidate_accuracy['correct']}/{candidate_accuracy['denominator']} "
        f"correct (accuracy {candidate_accuracy['accuracy']}; 95% Wilson "
        f"[{candidate_accuracy['wilson_95_lower']}, "
        f"{candidate_accuracy['wilson_95_upper']}]).\n\n"
        f"Laya: {control_accuracy['correct']}/{control_accuracy['denominator']} correct "
        f"(accuracy {control_accuracy['accuracy']}; 95% Wilson "
        f"[{control_accuracy['wilson_95_lower']}, "
        f"{control_accuracy['wilson_95_upper']}]).\n\n"
        "Paired outcomes: "
        f"both correct {paired['both_correct']}; candidate only "
        f"{paired['candidate_only_correct']}; control only "
        f"{paired['control_only_correct']}; both not correct "
        f"{paired['both_not_correct']}.\n\n"
        + "\n".join(f"- {item}" for item in projected["limitations"])
        + f"\n\nCanonical JSON SHA-256: `{json_sha256}`.\n"
    )


def run_publish_status(
    report_path: Path,
    *,
    readme_en: Path = README_EN_PATH,
    readme_pt_br: Path = README_PT_BR_PATH,
) -> dict[str, Any]:
    try:
        raw = json.loads(report_path.read_bytes(), object_pairs_hook=_no_duplicate_keys)
    except (OSError, ValueError) as error:
        raise ComparisonError("terminal comparison report") from error
    required = {
        "schema_version",
        "plan_sha256",
        "outcome",
        "planned",
        "accepted",
        "rejected",
        "unresolved",
        "transport_uncertain_calls",
        "nonconsecutive_transport_uncertain_calls",
        "consecutive_uncertainty_streak",
        "provider_calls",
        "provider_reported_cost_usd",
        "errors",
    }
    if (
        not isinstance(raw, dict)
        or set(raw) != required
        or raw["schema_version"] != COMPARISON_TERMINAL_REPORT_SCHEMA
        or raw["outcome"] != "insufficient_comparison_evidence"
        or raw["planned"] != 200
        or raw["unresolved"] != 0
        or raw["accepted"] + raw["rejected"] != 200
        or raw["transport_uncertain_calls"] != 0
        or not _is_digest(raw["plan_sha256"])
        or any(
            not isinstance(raw[field], int) or isinstance(raw[field], bool) or raw[field] < 0
            for field in (
                "planned",
                "accepted",
                "rejected",
                "unresolved",
                "transport_uncertain_calls",
                "nonconsecutive_transport_uncertain_calls",
                "consecutive_uncertainty_streak",
                "provider_calls",
            )
        )
        or not isinstance(raw["errors"], list)
        or not all(isinstance(error, str) for error in raw["errors"])
    ):
        raise ComparisonError("terminal report is not publishable")
    try:
        cost = Decimal(str(raw["provider_reported_cost_usd"]))
        if not cost.is_finite() or cost < 0:
            raise ArithmeticError
    except (ArithmeticError, ValueError):
        raise ComparisonError("terminal report is not publishable") from None
    status = {
        "outcome": raw.get("outcome"),
        "accepted": raw.get("accepted"),
        "rejected": raw.get("rejected"),
        "unresolved": raw.get("unresolved"),
    }
    _assert_public_safe(status)
    _publish_bilingual_status(
        readme_en=readme_en,
        readme_pt_br=readme_pt_br,
        binding_kind="insufficient",
        binding_sha256=_sha(report_path.read_bytes()),
        body_en=(
            "Phase 4E.4 ended without enough accepted synthetic comparison evidence: "
            f"{status['accepted']} accepted and {status['rejected']} rejected. No scored "
            "comparison was published, and Phase 4F may proceed only with this limitation."
        ),
        body_pt_br=(
            "A Fase 4E.4 terminou sem evidência sintética aceita suficiente para a "
            f"comparação: {status['accepted']} aceitos e {status['rejected']} rejeitados. "
            "Nenhuma comparação pontuada foi publicada; a Fase 4F pode avançar somente "
            "com essa limitação explícita."
        ),
    )
    return status


def _prepare_status_section(
    path: Path,
    body: str,
    *,
    binding_kind: str,
    binding_sha256: str,
    initial_phrase: str,
) -> bytes | None:
    begin = "<!-- phase4e4-status:begin -->"
    end = "<!-- phase4e4-status:end -->"
    original = path.read_text(encoding="utf-8")
    if original.count(begin) != 1 or original.count(end) != 1:
        raise ComparisonError("README status markers")
    prefix, remainder = original.split(begin, 1)
    current, suffix = remainder.split(end, 1)
    binding = f"<!-- phase4e4-status-binding:{binding_kind}:{binding_sha256} -->"
    replacement = "\n" + binding + "\n" + body + "\n"
    if current == replacement:
        return None
    if "<!-- phase4e4-status-binding:" in current:
        raise ComparisonError("README status is already bound to different evidence")
    if initial_phrase not in current:
        raise ComparisonError("README initial status is not recognized")
    updated = prefix + begin + replacement + end + suffix
    return updated.encode("utf-8")


def _commit_status_section(path: Path, payload: bytes | None) -> None:
    if payload is None:
        return
    mode = stat.S_IMODE(path.stat().st_mode)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _publish_bilingual_status(
    *,
    readme_en: Path,
    readme_pt_br: Path,
    binding_kind: str,
    binding_sha256: str,
    body_en: str,
    body_pt_br: str,
) -> None:
    if binding_kind not in {"scored", "insufficient"} or not _is_digest(binding_sha256):
        raise ComparisonError("README status binding")
    _assert_public_safe(
        {
            "binding_kind": binding_kind,
            "binding_sha256": binding_sha256,
            "body_en": body_en,
            "body_pt_br": body_pt_br,
        }
    )
    en_payload = _prepare_status_section(
        readme_en,
        body_en,
        binding_kind=binding_kind,
        binding_sha256=binding_sha256,
        initial_phrase="live comparison has not started",
    )
    pt_payload = _prepare_status_section(
        readme_pt_br,
        body_pt_br,
        binding_kind=binding_kind,
        binding_sha256=binding_sha256,
        initial_phrase="comparação live ainda não começou",
    )
    _commit_status_section(readme_en, en_payload)
    _commit_status_section(readme_pt_br, pt_payload)


# ── CLI entry point ──────────────────────────────────────────────────────


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Phase 4E.4 blind comparison CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    plan_p = sub.add_parser("plan")
    plan_p.add_argument("--output", type=Path, required=True)
    plan_p.add_argument("--training-packet", type=Path, required=True)
    plan_p.add_argument("--training-report-dir", type=Path)
    plan_p.add_argument("--training-capsule", type=Path, required=True)
    plan_p.add_argument("--saracura-encoder-snapshot", type=Path, required=True)
    plan_p.add_argument("--laya-snapshot", type=Path, required=True)

    gen_p = sub.add_parser("generate")
    gen_p.add_argument("--plan", type=Path, required=True)
    gen_p.add_argument("--work-dir", type=Path, required=True)
    gen_p.add_argument("--packet", type=Path, required=True)
    gen_p.add_argument("--allow-network", action="store_true", default=False)
    gen_p.add_argument("--training-packet", type=Path, required=True)
    gen_p.add_argument("--training-report-dir", type=Path)
    gen_p.add_argument("--saracura-encoder-snapshot", type=Path, required=True)
    gen_p.add_argument("--laya-snapshot", type=Path, required=True)

    sb_p = sub.add_parser("score-backend")
    sb_p.add_argument("--input-manifest", type=Path)
    sb_p.add_argument("--packet", type=Path)
    sb_p.add_argument("--backend", choices=["saracura", "laya"])
    sb_p.add_argument("--work-dir", type=Path)
    sb_p.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    sb_p.add_argument("--cpu-threads", type=int, default=1)
    sb_p.add_argument("--encoder-snapshot", type=Path)
    sb_p.add_argument("--training-capsule", type=Path)
    sb_p.add_argument("--laya-snapshot", type=Path)

    eval_p = sub.add_parser("evaluate")
    eval_p.add_argument("--packet", type=Path, required=True)
    eval_p.add_argument("--work-dir", type=Path, required=True)
    eval_p.add_argument("--saracura-encoder-snapshot", type=Path, required=True)
    eval_p.add_argument("--saracura-training-capsule", type=Path, required=True)
    eval_p.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    eval_p.add_argument("--cpu-threads", type=int, default=1)
    eval_p.add_argument("--laya-snapshot", type=Path)

    pub_p = sub.add_parser("publish")
    pub_p.add_argument("--result", type=Path, required=True)
    pub_p.add_argument("--publish-json", type=Path, default=PUBLIC_RESULT_JSON_PATH)
    pub_p.add_argument("--publish-md", type=Path, default=PUBLIC_RESULT_MD_PATH)
    pub_p.add_argument("--evaluation-receipt", type=Path)
    pub_p.add_argument("--readme-en", type=Path, default=README_EN_PATH)
    pub_p.add_argument("--readme-pt-br", type=Path, default=README_PT_BR_PATH)

    ps_p = sub.add_parser("publish-status")
    ps_p.add_argument("--report", type=Path, required=True)
    ps_p.add_argument("--readme-en", type=Path, default=README_EN_PATH)
    ps_p.add_argument("--readme-pt-br", type=Path, default=README_PT_BR_PATH)

    args = parser.parse_args()
    command = cast(str, args.command)

    if command == "plan":
        run_plan(
            args.output,
            training_packet=args.training_packet,
            training_report_dir=args.training_report_dir,
            training_capsule=args.training_capsule,
            saracura_encoder_snapshot=args.saracura_encoder_snapshot,
            laya_snapshot=args.laya_snapshot,
        )
        print(f"comparison plan written to {args.output}")
    elif command == "generate":
        run_generate(
            args.plan,
            args.__dict__.get("work_dir", Path()),
            args.packet,
            allow_network=args.allow_network,
            training_packet=args.training_packet,
            training_report_dir=args.training_report_dir,
            saracura_encoder_snapshot=args.saracura_encoder_snapshot,
            laya_snapshot=args.laya_snapshot,
        )
    elif command == "score-backend":
        if args.input_manifest is not None:
            run_score_backend_manifest(args.input_manifest)
        elif args.packet is not None and args.backend is not None and args.work_dir is not None:
            run_score_backend(
                args.packet,
                args.backend,
                args.work_dir,
                device=args.device,
                cpu_threads=args.cpu_threads,
                encoder_snapshot=args.__dict__.get("encoder_snapshot"),
                training_capsule=args.__dict__.get("training_capsule"),
                laya_snapshot=args.__dict__.get("laya_snapshot"),
            )
        else:
            parser.error("score-backend requires --input-manifest or explicit backend inputs")
    elif command == "evaluate":
        run_evaluate(
            args.packet,
            args.__dict__.get("work_dir", Path()),
            saracura_encoder_snapshot=args.saracura_encoder_snapshot,
            saracura_training_capsule=args.saracura_training_capsule,
            device=args.device,
            cpu_threads=args.cpu_threads,
            laya_snapshot=args.__dict__.get("laya_snapshot"),
        )
    elif command == "publish":
        run_publish(
            args.result,
            args.publish_json,
            args.publish_md,
            evaluation_receipt=args.evaluation_receipt,
            readme_en=args.readme_en,
            readme_pt_br=args.readme_pt_br,
        )
    elif command == "publish-status":
        status = run_publish_status(
            args.report,
            readme_en=args.readme_en,
            readme_pt_br=args.readme_pt_br,
        )
        print(json.dumps(status))


if __name__ == "__main__":
    main()
