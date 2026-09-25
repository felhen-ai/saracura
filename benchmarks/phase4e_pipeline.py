"""Operational, offline-first entry point for the Phase 4E research lane.

The module is deliberately outside the installed package.  Only ``corpus`` has
an HTTP implementation; the other commands neither import nor construct a
socket.  Provider replies are reduced to typed rows and digests in the corpus
ledger -- raw replies and credentials are never written to disk.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import math
import os
import ssl
import sys
from collections.abc import Callable, Mapping, Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal, cast

from benchmarks import saracura_universal_corpus as corpus
from benchmarks import saracura_universal_pilot as pilot
from benchmarks import saracura_universal_training as training
from benchmarks.io import atomic_create
from benchmarks.saracura_universal_policy import (
    POST_PILOT_AUTHOR_MODEL,
    POST_PILOT_REVIEWER_MODEL,
    require_phase4e_authorization,
    require_post_pilot_phase4e_authorization,
    validate_post_pilot_phase4e_policy,
)
from saracura.contracts.models import ChoiceCriterion

_AUTHOR_SINGLE_MAX_OUTPUT_TOKENS = 640
_AUTHOR_PAIR_MAX_OUTPUT_TOKENS = 1024
_REVIEWER_MAX_OUTPUT_TOKENS = 192
_OPENROUTER_HOST = "openrouter.ai"
_OPENROUTER_PATH = "/api/v1/chat/completions"
_WORK_SCHEMA = "phase4e-corpus-work.v1"
_CALL_WORK_SCHEMA = "phase4e-corpus-call-resolution.v1"
_FALLBACK_REASONS = frozenset(
    {
        "author_response_failure",
        "author_validation_failure",
        "reviewer_response_failure",
        "review_resolution_failure",
        "result_persistence_failure",
    }
)
_AUTHOR_LINEAGE_FIELDS = frozenset(
    {
        "author_response_sha256",
        "author_reservation_id",
        "author_request_id",
    }
)
_REVIEWER_LINEAGE_FIELDS = frozenset(
    {
        "reviewer_response_sha256",
        "reviewer_reservation_id",
        "reviewer_request_id",
    }
)
_BOUNDARY_TEXT = "\U0001f9ea"
_BOUNDARY_STATE_SUMMARY_CODEPOINTS = 180
_POST_PILOT_DIAGNOSTIC_KEYS = (
    "reviewed",
    "scenario_disagreement",
    "criterion_role_disagreement",
    "local_privacy",
    "reviewer_privacy_flags",
    "author_response_failures",
    "reviewer_response_failures",
    "retry_recoveries",
    "transport_uncertain_calls",
)
_POST_PILOT_MAXIMUM_ATTEMPTS = 5
_REPO_ROOT = Path(__file__).parents[1].resolve()
_POST_PILOT_PLAN_PATH = _REPO_ROOT / ".artifacts/phase4e/corpus-plan-v3/plan.json"
_POST_PILOT_WORK_DIR = _REPO_ROOT / ".artifacts/phase4e/corpus-work-v47"
_POST_PILOT_REPORT_DIR = _REPO_ROOT / ".artifacts/phase4e/corpus-report-v3"
_POST_PILOT_MODEL_ARTIFACT_ROOT = (
    Path.home() / "Library/Application Support/saracura/phase4e/model-artifacts"
).resolve()

Transport = Callable[[str, str, Mapping[str, str], bytes], tuple[int, Mapping[str, str], bytes]]


def _canonical(value: Any) -> bytes:
    return corpus._canonical(value)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, ValueError) as error:
        raise corpus.CorpusError("invalid pipeline JSON") from error
    if not isinstance(value, dict):
        raise corpus.CorpusError("invalid pipeline JSON")
    return cast(dict[str, Any], value)


def _read_plan(path: Path) -> dict[str, Any]:
    plan = _read_json(path)
    corpus.validate_plan(plan)
    return plan


def write_plan(output: Path) -> Path:
    """Create the historical v2 plan for legacy fixture callers."""
    atomic_create(output, _canonical(corpus.build_plan()) + b"\n")
    return output


def write_post_pilot_plan(output: Path) -> Path:
    """Create the only new corpus plan exposed by the phase command."""
    if output.resolve() != _POST_PILOT_PLAN_PATH:
        raise corpus.CorpusError("post-pilot plan path")
    atomic_create(output, _canonical(corpus.build_post_pilot_plan()) + b"\n")
    return output


def run_prepare_recovery() -> Path:
    """Prepare the reviewed external v4 state create-only before any provider call."""

    pilot.require_live_private_phase4e_root()
    paths = pilot.private_phase4e_paths()
    internal_catalog = pilot._platform_phase4e_root() / "research-ledgers"
    external_catalog = paths["research_ledgers"]
    raw_destination = paths["raw_evidence"] / _POST_PILOT_WORK_DIR.name
    capsule = external_catalog / f"{_POST_PILOT_WORK_DIR.name}-capsule"

    pilot.require_pilot_research_history(internal_catalog)
    try:
        pilot.require_pilot_research_history(external_catalog)
    except corpus.CorpusError:
        pilot.copy_research_catalog(internal_catalog, external_catalog)
    pilot.copy_raw_research_evidence(_POST_PILOT_WORK_DIR, raw_destination)
    pilot.create_research_capsule(raw_destination, capsule)
    pilot.validate_research_capsule(capsule, raw_destination)
    inventory_sha256, _scan = pilot.record_research_catalog(external_catalog)
    pilot.require_pilot_research_history(external_catalog)
    pilot.write_research_catalog_migration_marker(external_catalog, inventory_sha256)

    output = paths["recovery_plan"]
    encoded = _canonical(corpus.build_post_pilot_recovery_plan()) + b"\n"
    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if output.exists():
        if output.is_symlink() or output.read_bytes() != encoded:
            raise FileExistsError(f"benchmark artifact already exists: {output.name}")
    else:
        atomic_create(output, encoded)
        os.chmod(output, 0o600)
    return output


def _require_post_pilot_artifact_paths(
    plan_path: Path,
    work_dir: Path,
    packet: Path,
    report_dir: Path,
    artifact_root: Path,
) -> None:
    """Keep private v3 evidence in its reviewed checkout/state boundaries."""

    if (
        plan_path.resolve() != _POST_PILOT_PLAN_PATH
        or work_dir.resolve() != _POST_PILOT_WORK_DIR
        or report_dir.resolve() != _POST_PILOT_REPORT_DIR
        or artifact_root.resolve() != pilot.canonical_research_ledger_root().resolve()
    ):
        raise corpus.CorpusError("post-pilot artifact path")
    packet_path = packet.resolve()
    if (
        packet_path in (_POST_PILOT_MODEL_ARTIFACT_ROOT, _REPO_ROOT)
        or _POST_PILOT_MODEL_ARTIFACT_ROOT not in packet_path.parents
        or _REPO_ROOT in packet_path.parents
    ):
        raise corpus.CorpusError("post-pilot packet path")
    pilot._outside(work_dir, artifact_root)
    pilot._outside(report_dir, artifact_root)
    pilot._outside(packet, artifact_root)


def _provider_body(
    *,
    stage: str,
    model: str,
    messages: Sequence[Mapping[str, str]],
    response_schema: Mapping[str, Any],
    max_output_tokens: int,
    author_stage: str = "corpus_author",
    author_model: str = corpus.AUTHOR_MODEL,
    provider_override: Mapping[str, Any] | None = None,
    author_reasoning_effort: str | None = "none",
    reviewer_temperature: int | float | None = 0,
) -> bytes:
    return corpus.provider_request_bytes(
        stage=stage,
        model=model,
        messages=messages,
        response_schema=response_schema,
        max_output_tokens=max_output_tokens,
        author_stage=author_stage,
        author_model=author_model,
        provider_override=provider_override,
        author_reasoning_effort=author_reasoning_effort,
        reviewer_temperature=reviewer_temperature,
    )


def _preflight_request(
    stage: Literal["corpus_author", "corpus_reviewer"],
    body: bytes,
    max_output_tokens: int,
) -> None:
    """Calculate the full conservative request envelope before transport."""
    worst = corpus.request_worst_case(stage, body, max_output_tokens)
    # ``reserve`` is intentionally non-mutating: OpenRouterCorpusClient makes
    # the durable pre-send reservation immediately before the actual request.
    corpus.BudgetLedger().reserve(stage, worst)


def _author_batches(plan: Mapping[str, Any]) -> list[list[Mapping[str, Any]]]:
    slots = cast(list[Mapping[str, Any]], plan["slots"])
    if plan.get("schema_version") == "phase4e-universal-plan.v4":
        order = plan.get("supplemental_author_batch_order")
        supplemental_ids = plan.get("supplemental_task_ids")
        base_slots = plan.get("base_slots")
        if (
            not isinstance(order, list)
            or not isinstance(supplemental_ids, list)
            or not isinstance(base_slots, list)
        ):
            raise corpus.CorpusError("post-pilot recovery author batch order")
        by_id = {cast(str, slot["task_id"]): slot for slot in slots}
        base_ids = {cast(str, slot["task_id"]) for slot in base_slots}
        supplemental_batches: list[list[Mapping[str, Any]]] = []
        for ids in order:
            if not isinstance(ids, list) or len(ids) != 1:
                raise corpus.CorpusError("post-pilot recovery author batch order")
            batch = [
                by_id[task_id] for task_id in ids if isinstance(task_id, str) and task_id in by_id
            ]
            if len(batch) != 1 or cast(str, batch[0]["task_id"]) in base_ids:
                raise corpus.CorpusError("post-pilot recovery author batch order")
            supplemental_batches.append(batch)
        ordered_ids = [
            cast(str, slot["task_id"]) for batch in supplemental_batches for slot in batch
        ]
        if (
            len(ordered_ids) != len(supplemental_ids)
            or set(ordered_ids) != set(supplemental_ids)
            or set(ordered_ids) & base_ids
        ):
            raise corpus.CorpusError("post-pilot recovery author batch coverage")
        return supplemental_batches
    if plan.get("schema_version") == "phase4e-universal-plan.v3":
        order = plan.get("author_batch_order")
        by_id = {cast(str, slot["task_id"]): slot for slot in slots}
        if not isinstance(order, list):
            raise corpus.CorpusError("post-pilot author batch order")
        ordered_batches: list[list[Mapping[str, Any]]] = []
        for ids in order:
            if not isinstance(ids, list) or not ids or len(ids) > 2:
                raise corpus.CorpusError("post-pilot author batch order")
            batch = [
                by_id[task_id] for task_id in ids if isinstance(task_id, str) and task_id in by_id
            ]
            if len(batch) != len(ids):
                raise corpus.CorpusError("post-pilot author batch order")
            ordered_batches.append(batch)
        if {cast(str, slot["task_id"]) for batch in ordered_batches for slot in batch} != set(
            by_id
        ):
            raise corpus.CorpusError("post-pilot author batch coverage")
        return ordered_batches
    pairs: dict[str, list[Mapping[str, Any]]] = {}
    for slot in slots:
        pair_id = slot["pair_id"]
        if pair_id is not None:
            pairs.setdefault(cast(str, pair_id), []).append(slot)
    batches: list[list[Mapping[str, Any]]] = []
    emitted: set[str] = set()
    for slot in slots:
        pair_id = slot["pair_id"]
        if pair_id is None:
            batches.append([slot])
        elif cast(str, pair_id) not in emitted:
            batches.append(pairs[cast(str, pair_id)])
            emitted.add(cast(str, pair_id))
    return batches


def _author_max_output_tokens(slots: Sequence[Mapping[str, Any]]) -> int:
    return _AUTHOR_PAIR_MAX_OUTPUT_TOKENS if len(slots) == 2 else _AUTHOR_SINGLE_MAX_OUTPUT_TOKENS


def _post_pilot_config(plan: Mapping[str, Any]) -> dict[str, Any] | None:
    if plan.get("schema_version") not in {
        "phase4e-universal-plan.v3",
        "phase4e-universal-plan.v4",
    }:
        return None
    policy = validate_post_pilot_phase4e_policy()
    provider = cast(dict[str, Any], policy["provider"])
    expected = {
        "provider_policy_sha256": corpus._sha(corpus._canonical(provider["provider_policy"])),
        "author_request": provider["author_request"],
        "reviewer_request": provider["reviewer_request"],
        "models": {
            "corpus_author": POST_PILOT_AUTHOR_MODEL,
            "corpus_reviewer": POST_PILOT_REVIEWER_MODEL,
        },
    }
    if any(plan.get(key) != value for key, value in expected.items()):
        raise corpus.CorpusError("post-pilot plan request binding")
    return provider


def _load_v4_base_capsule(
    capsule: Path, plan: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], corpus.BudgetLedger]:
    """Reconstruct immutable v3 rows from the compact capsule, never raw work evidence."""

    if plan.get("schema_version") != "phase4e-universal-plan.v4":
        raise corpus.CorpusError("post-pilot recovery plan")
    corpus.validate_plan(plan)
    base = corpus.build_post_pilot_plan()
    if (
        plan.get("base_plan_sha256") != corpus._sha(corpus._canonical(base))
        or plan.get("base_final_ledger_sha256") is None
    ):
        raise corpus.CorpusError("post-pilot recovery base binding")
    pilot.validate_research_capsule(capsule)
    manifest = _read_json(capsule / "research-capsule.json")
    expected_ledger_sha = plan["base_final_ledger_sha256"]
    if (
        manifest.get("final_ledger_sha256") != expected_ledger_sha
        or hashlib.sha256((capsule / "ledger" / "ledger-0000.json").read_bytes()).hexdigest()
        != expected_ledger_sha
    ):
        raise corpus.CorpusError("post-pilot recovery base ledger")
    resolved = _load_resolved(capsule, base)
    all_base_ids = {
        cast(str, slot["task_id"]) for slot in cast(list[Mapping[str, Any]], base["slots"])
    }
    if set(resolved) != all_base_ids:
        raise corpus.CorpusError("post-pilot recovery base resolution")
    ledger = _load_ledger_or_fail_closed(capsule)
    if ledger.policy.schema_version != corpus.POST_PILOT_CORPUS_LEDGER_POLICY.schema_version:
        raise corpus.CorpusError("post-pilot recovery base ledger")
    _require_settled_task_resolution(ledger, resolved)
    diagnostics, complete = _load_post_pilot_diagnostics(capsule, base, set(resolved))
    if not complete:
        raise corpus.CorpusError("post-pilot recovery base diagnostics")
    _require_post_pilot_uncertain_resolutions(
        ledger, resolved, diagnostics["transport_uncertain_calls"]
    )
    accepted = [
        cast(dict[str, Any], item["row"])
        for item in resolved.values()
        if item["status"] == "accepted"
    ]
    rejected = [
        cast(dict[str, Any], item["row"])
        for item in resolved.values()
        if item["status"] == "rejected"
    ]
    return accepted, rejected, ledger


def _boundary_reviewer_row(slot: Mapping[str, Any]) -> corpus.ValidatedAuthorRow:
    """Create a text-free, capacity-saturating reviewer transport fixture.

    It contains only public reviewer fields and planner-owned identities.  The
    state value reserves the canonical state envelope, while every textual
    field uses a four-byte NFC scalar so the byte ceiling, not an ASCII-only
    approximation, drives the conservative debit.
    """
    option_count = cast(int, slot["option_count"])
    return corpus.ValidatedAuthorRow.model_construct(
        task_id=slot["task_id"],
        family_id=slot["family_id"],
        locale=slot["locale"],
        domain=slot["domain"],
        instruction=_BOUNDARY_TEXT * 120,
        state={"summary": _BOUNDARY_TEXT * _BOUNDARY_STATE_SUMMARY_CODEPOINTS},
        criteria=[
            ChoiceCriterion.model_construct(
                id=f"criterion-{index}", description=_BOUNDARY_TEXT * 120
            )
            for index in range(option_count)
        ],
        selected_criterion_id="criterion-0",
        split=slot["split"],
        pair_id=slot["pair_id"],
        axes=slot["axes"],
        semantic_equivalence_attestation={},
        cross_locale_attestation=None,
    )


def _aggregate_preflight(
    plan: Mapping[str, Any],
    resolved: Mapping[str, Mapping[str, Any]],
    ledger: corpus.BudgetLedger,
) -> dict[str, Decimal]:
    """Fail before transport if every unresolved planned call cannot fit.

    Individual reservations still protect each send.  This separate,
    non-mutating calculation prevents starting a plan that cannot finish under
    its nonfungible stage limits, including durable spend from a resumed run.
    """
    completed = set(resolved)
    config = _post_pilot_config(plan)
    policy = corpus.POST_PILOT_CORPUS_LEDGER_POLICY if config is not None else None
    author_total = Decimal()
    reviewer_total = Decimal()
    for slots in _author_batches(plan):
        task_ids = {cast(str, slot["task_id"]) for slot in slots}
        if task_ids <= completed:
            continue
        if task_ids & completed:
            raise corpus.CorpusError("partial author family resume")
        max_output_tokens = 1024 if config is not None else _author_max_output_tokens(slots)
        author_total += corpus.request_worst_case(
            "corpus_author",
            _provider_body(
                stage="corpus_author",
                model=POST_PILOT_AUTHOR_MODEL if config is not None else corpus.AUTHOR_MODEL,
                messages=corpus.author_messages(slots),
                response_schema=corpus.author_schema(slots),
                max_output_tokens=max_output_tokens,
                author_model=POST_PILOT_AUTHOR_MODEL if config is not None else corpus.AUTHOR_MODEL,
                provider_override=config["provider_policy"] if config is not None else None,
                author_reasoning_effort=None if config is not None else "none",
            ),
            max_output_tokens,
            prices=policy.prices if policy is not None else None,
        )
        for slot in slots:
            row = _boundary_reviewer_row(slot)
            reviewer_total += corpus.request_worst_case(
                "corpus_reviewer",
                _provider_body(
                    stage="corpus_reviewer",
                    model=POST_PILOT_REVIEWER_MODEL
                    if config is not None
                    else corpus.REVIEWER_MODEL,
                    messages=(
                        pilot.pilot_reviewer_messages(row.model_dump(mode="json"))
                        if config is not None
                        else corpus.reviewer_messages([row])
                    ),
                    response_schema=(
                        pilot.pilot_reviewer_schema(1)
                        if config is not None
                        else corpus.reviewer_schema(1)
                    ),
                    max_output_tokens=512 if config is not None else _REVIEWER_MAX_OUTPUT_TOKENS,
                    author_model=POST_PILOT_AUTHOR_MODEL
                    if config is not None
                    else corpus.AUTHOR_MODEL,
                    provider_override=config["provider_policy"] if config is not None else None,
                    author_reasoning_effort=None if config is not None else "none",
                ),
                512 if config is not None else _REVIEWER_MAX_OUTPUT_TOKENS,
                prices=policy.prices if policy is not None else None,
            )
    totals = {"corpus_author": author_total, "corpus_reviewer": reviewer_total}
    for stage, total in totals.items():
        ledger.reserve(stage, total)
    if policy is None and (
        sum((Decimal(entry["debit_usd"]) for entry in ledger.entries), Decimal())
        + sum(totals.values(), Decimal())
        > corpus.TOTAL_BUDGET
    ):
        raise corpus.CorpusError("total budget exhausted")
    return totals


def _response_content(response: Mapping[str, Any]) -> dict[str, Any]:
    try:
        choice = response["choices"][0]
        message = choice["message"]
        content = message["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise corpus.CorpusError("provider response content (missing)") from error
    if not isinstance(message, Mapping):
        raise corpus.CorpusError("provider response content (message)")
    # Some OpenRouter providers attach reasoning metadata even when strict
    # structured output is honored. It is untrusted transport metadata: ignore
    # it and return only the decoded content below. Nothing from those fields is
    # retained in rows, ledgers, diagnostics, or reports.
    if not isinstance(content, str):
        raise corpus.CorpusError("provider response content type")
    try:
        value = json.loads(content, object_pairs_hook=_unique_json_object)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise corpus.CorpusError("provider response JSON") from error
    if not isinstance(value, dict):
        raise corpus.CorpusError("provider response content")
    return cast(dict[str, Any], value)


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate provider keys instead of silently accepting the last value."""

    value: dict[str, Any] = {}
    for key, child in pairs:
        if key in value:
            raise ValueError("duplicate provider JSON key")
        value[key] = child
    return value


def _openrouter_transport(timeout_seconds: int = 30) -> Transport:
    """Create the only live transport: fixed HTTPS, no proxy, no redirects."""

    if type(timeout_seconds) is not int or timeout_seconds < 1:
        raise corpus.CorpusError("provider transport timeout")

    context = ssl.create_default_context()

    def transport(
        method: str, url: str, headers: Mapping[str, str], body: bytes
    ) -> tuple[int, Mapping[str, str], bytes]:
        if method != "POST" or url != f"https://{_OPENROUTER_HOST}{_OPENROUTER_PATH}":
            raise corpus.CorpusError("pinned provider request")
        # http.client has neither proxy discovery nor redirect-following.
        connection = http.client.HTTPSConnection(
            _OPENROUTER_HOST, 443, context=context, timeout=timeout_seconds
        )
        try:
            connection.request(method, _OPENROUTER_PATH, body=body, headers=dict(headers))
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            connection.close()

    return transport


def _work_path(work_dir: Path, task_id: str) -> Path:
    if not task_id.startswith("task-"):
        raise corpus.CorpusError("work task identity")
    return work_dir / "resolved" / f"{task_id}.json"


def _call_work_path(work_dir: Path, task_ids: Sequence[str]) -> Path:
    if not task_ids or any(not task_id.startswith("task-") for task_id in task_ids):
        raise corpus.CorpusError("work record")
    digest = corpus._sha(_canonical(sorted(task_ids)))
    return work_dir / "resolved" / f"call-{digest}.json"


def _plan_slots_by_id(plan: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    slots = cast(list[Mapping[str, Any]], plan["slots"])
    by_id = {cast(str, slot["task_id"]): slot for slot in slots}
    if len(by_id) != len(slots):
        raise corpus.CorpusError("work record")
    return by_id


def _call_task_sets(plan: Mapping[str, Any]) -> set[frozenset[str]]:
    return {
        frozenset(cast(str, slot["task_id"]) for slot in batch) for batch in _author_batches(plan)
    }


def _validate_legacy_resolution(
    value: Mapping[str, Any], slots: Mapping[str, Mapping[str, Any]]
) -> tuple[str, dict[str, Any]]:
    if (
        set(value) != {"schema_version", "task_id", "status", "row"}
        or value["schema_version"] != _WORK_SCHEMA
    ):
        raise corpus.CorpusError("work record")
    task_id = value["task_id"]
    row = value["row"]
    if (
        not isinstance(task_id, str)
        or task_id not in slots
        or value["status"] not in {"accepted", "rejected"}
        or not isinstance(row, dict)
        or row.get("task_id") != task_id
        or (value["status"] == "rejected" and row.get("split") != slots[task_id]["split"])
    ):
        raise corpus.CorpusError("work record")
    return task_id, cast(dict[str, Any], value)


def _validate_fallback_resolution(
    value: Mapping[str, Any], slots: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    task_id = value.get("task_id")
    allowed = {"task_id", "split", "reason"} | _AUTHOR_LINEAGE_FIELDS | _REVIEWER_LINEAGE_FIELDS
    author_fields = set(value) & _AUTHOR_LINEAGE_FIELDS
    reviewer_fields = set(value) & _REVIEWER_LINEAGE_FIELDS
    if (
        set(value) - allowed
        or not isinstance(task_id, str)
        or task_id not in slots
        or value.get("split") != slots[task_id]["split"]
        or value.get("reason") not in _FALLBACK_REASONS
        or author_fields not in (set(), set(_AUTHOR_LINEAGE_FIELDS))
        or reviewer_fields not in (set(), set(_REVIEWER_LINEAGE_FIELDS))
        or any(not isinstance(value[key], str) for key in author_fields | reviewer_fields)
    ):
        raise corpus.CorpusError("work record")
    return {
        "schema_version": _WORK_SCHEMA,
        "task_id": task_id,
        "status": "rejected",
        "row": dict(value),
    }


def _validate_call_resolution(
    value: Mapping[str, Any],
    slots: Mapping[str, Mapping[str, Any]],
    call_task_sets: set[frozenset[str]],
) -> list[tuple[str, dict[str, Any]]]:
    if (
        set(value) != {"schema_version", "kind", "resolutions"}
        or value["schema_version"] != _CALL_WORK_SCHEMA
    ):
        raise corpus.CorpusError("work record")
    kind = value["kind"]
    resolutions = value["resolutions"]
    if (
        kind not in {"normal", "settled_fallback"}
        or not isinstance(resolutions, list)
        or not resolutions
    ):
        raise corpus.CorpusError("work record")
    task_ids = [entry.get("task_id") for entry in resolutions if isinstance(entry, dict)]
    if (
        len(task_ids) != len(resolutions)
        or any(not isinstance(task_id, str) for task_id in task_ids)
        or len(set(task_ids)) != len(task_ids)
        or frozenset(cast(str, task_id) for task_id in task_ids) not in call_task_sets
    ):
        raise corpus.CorpusError("work record")
    expanded: list[tuple[str, dict[str, Any]]] = []
    for entry in cast(list[dict[str, Any]], resolutions):
        task_id = cast(str, entry["task_id"])
        if kind == "normal":
            if set(entry) != {"task_id", "status", "row"}:
                raise corpus.CorpusError("work record")
            expanded.append(
                _validate_legacy_resolution({"schema_version": _WORK_SCHEMA, **entry}, slots)
            )
        else:
            expanded.append((task_id, _validate_fallback_resolution(entry, slots)))
    return expanded


def _load_resolved(work_dir: Path, plan: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    if not work_dir.exists():
        return {}
    if not work_dir.is_dir():
        raise corpus.CorpusError("work directory")
    slots = _plan_slots_by_id(plan)
    call_task_sets = _call_task_sets(plan)
    resolved: dict[str, dict[str, Any]] = {}
    for path in (
        sorted((work_dir / "resolved").glob("*.json")) if (work_dir / "resolved").exists() else []
    ):
        value = _read_json(path)
        if value.get("schema_version") == _WORK_SCHEMA:
            entries = [_validate_legacy_resolution(value, slots)]
        elif value.get("schema_version") == _CALL_WORK_SCHEMA:
            entries = _validate_call_resolution(value, slots, call_task_sets)
            if path != _call_work_path(work_dir, [task_id for task_id, _ in entries]):
                raise corpus.CorpusError("work record")
        else:
            raise corpus.CorpusError("work record")
        for task_id, entry in entries:
            if task_id in resolved:
                raise corpus.CorpusError("work record")
            resolved[task_id] = entry
    return resolved


def _store_resolution(work_dir: Path, row: Mapping[str, Any], status: str) -> None:
    task_id = row.get("task_id")
    if not isinstance(task_id, str) or status not in {"accepted", "rejected"}:
        raise corpus.CorpusError("work record")
    atomic_create(
        _work_path(work_dir, task_id),
        _canonical(
            {"schema_version": _WORK_SCHEMA, "task_id": task_id, "status": status, "row": row}
        )
        + b"\n",
    )


def _store_call_resolution(
    work_dir: Path,
    slots: Sequence[Mapping[str, Any]],
    resolutions: Sequence[tuple[Mapping[str, Any], str]],
) -> None:
    expected_ids = {cast(str, slot["task_id"]) for slot in slots}
    actual_ids = [row.get("task_id") for row, _ in resolutions]
    if (
        not expected_ids
        or len(actual_ids) != len(expected_ids)
        or set(actual_ids) != expected_ids
        or any(status not in {"accepted", "rejected"} for _, status in resolutions)
    ):
        raise corpus.CorpusError("work record")
    atomic_create(
        _call_work_path(work_dir, sorted(expected_ids)),
        _canonical(
            {
                "schema_version": _CALL_WORK_SCHEMA,
                "kind": "normal",
                "resolutions": [
                    {"task_id": row["task_id"], "status": status, "row": row}
                    for row, status in resolutions
                ],
            }
        )
        + b"\n",
    )


def _lineage_or_none(
    client: corpus.OpenRouterCorpusClient,
    actor: Literal["author", "reviewer"],
    expected_task_ids: Sequence[str],
) -> dict[str, str] | None:
    try:
        stage: Literal["corpus_author", "corpus_reviewer"] = (
            "corpus_author" if actor == "author" else "corpus_reviewer"
        )
        journal = client.last_journal(stage)
        journal_task_ids = journal.get("task_ids")
        if (
            not isinstance(journal_task_ids, list)
            or len(journal_task_ids) != len(expected_task_ids)
            or set(journal_task_ids) != set(expected_task_ids)
        ):
            return None
        return corpus.lineage_from_journal(journal, actor)
    except corpus.CorpusError:
        return None


def _store_settled_fallback(
    work_dir: Path,
    slots: Sequence[Mapping[str, Any]],
    *,
    reason: str,
    author_lineage: Mapping[str, str] | None,
    reviewer_lineages: Mapping[str, Mapping[str, str]],
) -> list[dict[str, Any]]:
    if reason not in _FALLBACK_REASONS or not slots:
        raise corpus.CorpusError("work record")
    if author_lineage is not None and set(author_lineage) != _AUTHOR_LINEAGE_FIELDS:
        raise corpus.CorpusError("work record")
    rows: list[dict[str, Any]] = []
    for slot in slots:
        task_id = cast(str, slot["task_id"])
        reviewer_lineage = reviewer_lineages.get(task_id)
        if reviewer_lineage is not None and set(reviewer_lineage) != _REVIEWER_LINEAGE_FIELDS:
            raise corpus.CorpusError("work record")
        rows.append(
            {
                "task_id": task_id,
                "split": slot["split"],
                "reason": reason,
                **(dict(author_lineage) if author_lineage is not None else {}),
                **(dict(reviewer_lineage) if reviewer_lineage is not None else {}),
            }
        )
    atomic_create(
        _call_work_path(work_dir, sorted(cast(str, slot["task_id"]) for slot in slots)),
        _canonical(
            {
                "schema_version": _CALL_WORK_SCHEMA,
                "kind": "settled_fallback",
                "resolutions": rows,
            }
        )
        + b"\n",
    )
    return rows


def _store_settled_fallback_required(
    work_dir: Path,
    slots: Sequence[Mapping[str, Any]],
    *,
    reason: str,
    author_lineage: Mapping[str, str] | None,
    reviewer_lineages: Mapping[str, Mapping[str, str]],
) -> list[dict[str, Any]]:
    """Persist terminal call resolution or surface the persistence failure."""

    try:
        return _store_settled_fallback(
            work_dir,
            slots,
            reason=reason,
            author_lineage=author_lineage,
            reviewer_lineages=reviewer_lineages,
        )
    except Exception as error:
        raise corpus.CorpusError("settled fallback persistence failure") from error


def _load_ledger_or_fail_closed(work_dir: Path) -> corpus.BudgetLedger:
    ledger = corpus.resume_ledger(work_dir / "ledger")
    if any(entry["status"] == "reserved" for entry in ledger.entries):
        raise corpus.CorpusError("unresolved pre-send reservation")
    ledger.require_complete_provider_journal()
    return ledger


def _require_settled_task_resolution(
    ledger: corpus.BudgetLedger, resolved: Mapping[str, Mapping[str, Any]]
) -> None:
    """Never resend a paid response whose planned task IDs were not persisted."""

    completed = set(resolved)
    for journal in ledger.provider_journal:
        task_ids = journal["task_ids"]
        if not set(cast(list[str], task_ids)) <= completed:
            raise corpus.CorpusError("settled provider call lacks complete task resolution")


def _write_wilson_projection(
    ledger_directory: Path,
    plan: Mapping[str, Any],
    resolved: Sequence[Mapping[str, str]],
    stop_reason: str | None,
) -> Path:
    """Persist the required ten-batch projection beside immutable ledger states."""
    slots = cast(list[Mapping[str, Any]], plan["slots"])
    statuses = {row["task_id"]: row["status"] for row in resolved}

    def observation(candidates: Sequence[Mapping[str, Any]], minimum: int) -> dict[str, Any]:
        done = [slot for slot in candidates if cast(str, slot["task_id"]) in statuses]
        accepted = sum(statuses[cast(str, slot["task_id"])] == "accepted" for slot in done)
        total = len(done)
        eligible = total >= 20
        lower_bound = corpus.wilson_lower_bound(accepted, total) if total else None
        return {
            "accepted": accepted,
            "observations": total,
            "unresolved": len(candidates) - total,
            "observed_acceptance": accepted / total if total else None,
            "wilson_lower_bound": lower_bound,
            "wilson_eligible": eligible,
            "wilson_projected_accepted": (
                accepted + math.floor((len(candidates) - total) * lower_bound)
                if eligible and lower_bound is not None
                else None
            ),
            "minimum": minimum,
        }

    projections: dict[str, dict[str, Any]] = {"global": observation(slots, 1200)}
    for split, minimum in (
        ("synthetic_train", 840),
        ("synthetic_dev", 180),
        ("synthetic_holdout", 180),
    ):
        projections[split] = observation(
            [slot for slot in slots if slot["split"] == split], minimum
        )
    for split in corpus.SPLITS:
        for locale in corpus.LOCALES:
            for option_count in corpus.OPTION_COUNTS:
                key = f"{split}:{locale}:{option_count}"
                projections[key] = observation(
                    [
                        slot
                        for slot in slots
                        if slot["split"] == split
                        and slot["locale"] == locale
                        and slot["option_count"] == option_count
                    ],
                    20 if split == "synthetic_train" else 10,
                )
    ledger_directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    existing = sorted(ledger_directory.glob("wilson-*.json"))
    path = ledger_directory / f"wilson-{len(existing):04d}.json"
    atomic_create(
        path,
        _canonical(
            {
                "schema_version": "phase4e-wilson-ledger.v1",
                "resolved_tasks": len(resolved),
                "stop_reason": stop_reason,
                "projections": projections,
            }
        )
        + b"\n",
    )
    return path


def _review_one(
    client: corpus.OpenRouterCorpusClient,
    row: Mapping[str, Any],
    api_key: str,
    *,
    post_pilot: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    typed = corpus.ValidatedAuthorRow.model_validate(row)
    messages = (
        pilot.pilot_reviewer_messages(typed.model_dump(mode="json"))
        if post_pilot
        else corpus.reviewer_messages([typed])
    )
    _preflight_request(
        "corpus_reviewer",
        _provider_body(
            stage="corpus_reviewer",
            model=POST_PILOT_REVIEWER_MODEL if post_pilot else corpus.REVIEWER_MODEL,
            messages=messages,
            response_schema=pilot.pilot_reviewer_schema(1)
            if post_pilot
            else corpus.reviewer_schema(1),
            max_output_tokens=512 if post_pilot else _REVIEWER_MAX_OUTPUT_TOKENS,
        ),
        _REVIEWER_MAX_OUTPUT_TOKENS,
    )
    payload = _response_content(
        client.reviewer(
            row=typed,
            max_output_tokens=512 if post_pilot else _REVIEWER_MAX_OUTPUT_TOKENS,
            api_key=api_key,
        )
    )
    reviews = payload.get("reviews")
    if not isinstance(reviews, list) or len(reviews) != 1 or not isinstance(reviews[0], dict):
        raise corpus.CorpusError("review response")
    return (
        (
            pilot.bind_pilot_reviewer_record(cast(dict[str, Any], reviews[0]), typed.task_id)
            if post_pilot
            else corpus.materialize_reviewer_record(cast(dict[str, Any], reviews[0]), typed)
        ),
        client.last_journal("corpus_reviewer"),
    )


def _post_pilot_blank_diagnostics() -> dict[str, int]:
    return {key: 0 for key in _POST_PILOT_DIAGNOSTIC_KEYS}


def _post_pilot_diagnostics_path(work_dir: Path, slots: Sequence[Mapping[str, Any]]) -> Path:
    task_ids = sorted(cast(str, slot["task_id"]) for slot in slots)
    return work_dir / "diagnostics" / f"call-{corpus._sha(_canonical(task_ids))}.json"


def _store_post_pilot_diagnostics(
    work_dir: Path, slots: Sequence[Mapping[str, Any]], counts: Mapping[str, int]
) -> None:
    if set(counts) != set(_POST_PILOT_DIAGNOSTIC_KEYS) or any(
        type(counts[key]) is not int or counts[key] < 0 for key in _POST_PILOT_DIAGNOSTIC_KEYS
    ):
        raise corpus.CorpusError("post-pilot diagnostics")
    task_ids = sorted(cast(str, slot["task_id"]) for slot in slots)
    atomic_create(
        _post_pilot_diagnostics_path(work_dir, slots),
        _canonical(
            {
                "schema_version": "phase4e-corpus-diagnostics.v3",
                "task_ids": task_ids,
                "counts": dict(counts),
            }
        )
        + b"\n",
    )


def _load_post_pilot_diagnostics(
    work_dir: Path, plan: Mapping[str, Any], completed: set[str]
) -> tuple[dict[str, int], bool]:
    totals = _post_pilot_blank_diagnostics()
    expected: set[Path] = set()
    complete = True
    for slots in _author_batches(plan):
        task_ids = {cast(str, slot["task_id"]) for slot in slots}
        if not task_ids <= completed:
            continue
        path = _post_pilot_diagnostics_path(work_dir, slots)
        expected.add(path)
        try:
            value = _read_json(path)
        except corpus.CorpusError:
            complete = False
            continue
        counts = value.get("counts")
        if (
            set(value) != {"schema_version", "task_ids", "counts"}
            or value.get("schema_version") != "phase4e-corpus-diagnostics.v3"
            or value.get("task_ids") != sorted(task_ids)
            or not isinstance(counts, dict)
            or set(counts) != set(_POST_PILOT_DIAGNOSTIC_KEYS)
            or any(
                type(counts[key]) is not int or counts[key] < 0
                for key in _POST_PILOT_DIAGNOSTIC_KEYS
            )
        ):
            complete = False
            continue
        for key in _POST_PILOT_DIAGNOSTIC_KEYS:
            totals[key] += counts[key]
    actual = (
        set((work_dir / "diagnostics").glob("call-*.json"))
        if (work_dir / "diagnostics").exists()
        else set()
    )
    return totals, complete and actual == expected


def _uncertain_lineage(
    ledger: corpus.BudgetLedger, actor: Literal["author", "reviewer"], stage: str
) -> dict[str, str]:
    if not ledger.entries:
        raise corpus.CorpusError("uncertain lineage")
    entry = ledger.entries[-1]
    if (
        entry["status"] != "uncertain"
        or entry["stage"] != stage
        or entry["response_sha256"] != "0" * 64
        or entry["request_id"] != entry["reservation_id"]
    ):
        raise corpus.CorpusError("uncertain lineage")
    return {
        f"{actor}_response_sha256": entry["response_sha256"],
        f"{actor}_reservation_id": entry["reservation_id"],
        f"{actor}_request_id": entry["request_id"],
    }


def _require_post_pilot_uncertain_resolutions(
    ledger: corpus.BudgetLedger, resolved: Mapping[str, Mapping[str, Any]], diagnostics: int
) -> None:
    uncertain = [entry for entry in ledger.entries if entry["status"] == "uncertain"]
    if diagnostics != len(uncertain):
        raise corpus.CorpusError("unbound uncertain resolution")
    journals = {
        (entry["stage"], entry["reservation_id"]): entry for entry in ledger.provider_journal
    }
    if len(journals) != len(ledger.provider_journal):
        raise corpus.CorpusError("unbound uncertain resolution")
    for entry in uncertain:
        stage = entry["stage"]
        actor = (
            "author"
            if stage == "corpus_author"
            else "reviewer"
            if stage == "corpus_reviewer"
            else None
        )
        journal = journals.get((stage, entry["reservation_id"]))
        if actor is None or journal is None or journal["task_ids"] != sorted(journal["task_ids"]):
            raise corpus.CorpusError("unbound uncertain resolution")
        expected = _uncertain_lineage_for_entry(entry, cast(Literal["author", "reviewer"], actor))
        expected_reason = f"{actor}_transport_uncertain"
        for task_id in journal["task_ids"]:
            item = resolved.get(task_id)
            row = item.get("row") if item is not None else None
            if (
                item is None
                or item.get("status") != "rejected"
                or not isinstance(row, dict)
                or row.get("reason") != expected_reason
                or any(row.get(key) != value for key, value in expected.items())
            ):
                raise corpus.CorpusError("unbound uncertain resolution")


def _uncertain_lineage_for_entry(
    entry: Mapping[str, str], actor: Literal["author", "reviewer"]
) -> dict[str, str]:
    if entry.get("response_sha256") != "0" * 64 or entry.get("request_id") != entry.get(
        "reservation_id"
    ):
        raise corpus.CorpusError("uncertain lineage")
    return {
        f"{actor}_response_sha256": entry["response_sha256"],
        f"{actor}_reservation_id": entry["reservation_id"],
        f"{actor}_request_id": entry["request_id"],
    }


def _post_pilot_semantic_diagnostics(
    row: Mapping[str, Any], review: Mapping[str, Any]
) -> tuple[int, int]:
    author = row.get("semantic_equivalence_attestation")
    observed = review.get("semantic_equivalence_attestation")
    if not isinstance(author, dict) or not isinstance(observed, dict):
        raise corpus.CorpusError("review diagnostic")
    scenario = int(author.get("scenario") != observed.get("scenario"))
    roles = int(author.get("criterion_roles") != observed.get("criterion_roles"))
    return scenario, roles


def _post_pilot_report(
    *,
    plan: Mapping[str, Any],
    accepted: Sequence[Mapping[str, Any]],
    rejected: Sequence[Mapping[str, Any]],
    ledger: corpus.BudgetLedger,
    diagnostics: Mapping[str, int],
    inventory_sha256: str,
    packet: Path,
    work_dir: Path,
    historical: pilot.SpendScan,
) -> dict[str, Any]:
    slots = cast(list[Mapping[str, Any]], plan["slots"])
    counts: dict[str, dict[str, dict[str, int]]] = {
        key: {"accepted": {}, "rejected": {}}
        for key in ("split", "locale", "domain", "option_count")
    }
    for status, rows in (("accepted", accepted), ("rejected", rejected)):
        for row in rows:
            task_id = row.get("task_id")
            slot = next((item for item in slots if item["task_id"] == task_id), None)
            if slot is None:
                raise corpus.CorpusError("report task identity")
            for key, value in (
                ("split", slot["split"]),
                ("locale", slot["locale"]),
                ("domain", slot["domain"]),
                ("option_count", str(slot["option_count"])),
            ):
                bucket = counts[key][status]
                bucket[str(value)] = bucket.get(str(value), 0) + 1
    rejected_counts: dict[str, int] = {}
    for row in rejected:
        reason = row.get("reason")
        if isinstance(reason, str):
            rejected_counts[reason] = rejected_counts.get(reason, 0) + 1
    provider_cost = sum(
        (Decimal(entry["provider_cost_usd"]) for entry in ledger.entries), Decimal()
    )
    debit = sum((Decimal(entry["debit_usd"]) for entry in ledger.entries), Decimal())
    projection = corpus.stop_projection(
        [{"task_id": row["task_id"], "status": "accepted"} for row in accepted]
        + [{"task_id": row["task_id"], "status": "rejected"} for row in rejected],
        plan,
    )
    snapshots = sorted((work_dir / "ledger").glob("ledger-*.json"))
    if not snapshots:
        raise corpus.CorpusError("post-pilot ledger missing")
    return {
        "schema_version": "phase4e-corpus-report.v3",
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "counts": {**counts, "reason": dict(sorted(rejected_counts.items()))},
        "semantic_diagnostics": {
            key: diagnostics[key]
            for key in ("reviewed", "scenario_disagreement", "criterion_role_disagreement")
        },
        "transport_validation_diagnostics": {
            key: diagnostics[key]
            for key in (
                "local_privacy",
                "reviewer_privacy_flags",
                "author_response_failures",
                "reviewer_response_failures",
                "retry_recoveries",
                "transport_uncertain_calls",
            )
        },
        "wilson_stop_projection": projection,
        "provider_reported_cost_usd": str(provider_cost),
        "conservative_debit_usd": str(debit),
        "cumulative_provider_reported_cost_usd": str(historical.provider_reported_cost_usd),
        "cumulative_conservative_debit_usd": str(historical.conservative_debit_usd),
        "plan_sha256": corpus._sha(_canonical(plan)),
        "ledger_sha256": hashlib.sha256(snapshots[-1].read_bytes()).hexdigest(),
        "packet_manifest_sha256": hashlib.sha256((packet / "packet.json").read_bytes()).hexdigest(),
        "research_inventory_sha256": inventory_sha256,
    }


def _resolution_sha256(work_dir: Path) -> str:
    files: list[dict[str, str]] = []
    for directory in ("resolved", "diagnostics"):
        for path in sorted((work_dir / directory).glob("call-*.json")):
            if not path.is_file() or path.is_symlink():
                raise corpus.CorpusError("post-pilot resolution evidence")
            files.append(
                {
                    "path": path.relative_to(work_dir).as_posix(),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    return hashlib.sha256(_canonical(sorted(files, key=lambda item: item["path"]))).hexdigest()


def _terminal_outcome_report(
    *,
    plan: Mapping[str, Any],
    accepted: Sequence[Mapping[str, Any]],
    rejected: Sequence[Mapping[str, Any]],
    ledger: corpus.BudgetLedger,
    work_dir: Path,
    outcome: Literal["minimum_failed", "seal_validation_failed"],
    errors: Sequence[str],
    inventory_sha256: str,
) -> dict[str, Any]:
    """Return closed, text-free evidence for a post-network terminal outcome."""

    slots = {
        cast(str, slot["task_id"]): slot for slot in cast(list[Mapping[str, Any]], plan["slots"])
    }
    counts: dict[str, dict[str, int]] = {
        "split": {},
        "locale": {},
        "domain": {},
        "option_count": {},
        "reason": {},
    }
    for row in [*accepted, *rejected]:
        task_id = row.get("task_id")
        if not isinstance(task_id, str) or task_id not in slots:
            raise corpus.CorpusError("report task identity")
        slot = slots[task_id]
        for name, value in (
            ("split", slot["split"]),
            ("locale", slot["locale"]),
            ("domain", slot["domain"]),
            ("option_count", str(slot["option_count"])),
        ):
            bucket = counts[name]
            string_value = str(value)
            bucket[string_value] = bucket.get(string_value, 0) + 1
    for row in rejected:
        reason = row.get("reason")
        if isinstance(reason, str):
            counts["reason"][reason] = counts["reason"].get(reason, 0) + 1
    snapshots = sorted((work_dir / "ledger").glob("ledger-*.json"))
    if not snapshots:
        raise corpus.CorpusError("post-pilot ledger missing")
    provider_cost = sum(
        (Decimal(entry["provider_cost_usd"]) for entry in ledger.entries), Decimal()
    )
    debit = sum((Decimal(entry["debit_usd"]) for entry in ledger.entries), Decimal())
    all_ids = set(slots)
    settled_ids = {
        cast(str, row["task_id"])
        for row in [*accepted, *rejected]
        if isinstance(row.get("task_id"), str)
    }
    return {
        "schema_version": "phase4e-corpus-terminal-report.v1",
        "outcome": outcome,
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "unresolved_count": len(all_ids - settled_ids),
        "counts": {key: dict(sorted(value.items())) for key, value in counts.items()},
        "errors": sorted(set(errors)),
        "provider_reported_cost_usd": str(provider_cost),
        "conservative_debit_usd": str(debit),
        "plan_sha256": corpus._sha(_canonical(plan)),
        "ledger_sha256": hashlib.sha256(snapshots[-1].read_bytes()).hexdigest(),
        "resolution_sha256": _resolution_sha256(work_dir),
        "research_inventory_sha256": inventory_sha256,
    }


def _write_terminal_outcome_report(report_dir: Path, payload: Mapping[str, Any]) -> Path:
    """Append a numbered report, or reuse an identical terminal outcome."""

    report_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    encoded = _canonical(payload) + b"\n"
    reports = pilot._numbered_files(report_dir, "report")
    for path in reports:
        if path.read_bytes() == encoded:
            return path
    # A sealed v4 packet is its report's terminal authority.  In particular a
    # recovery/publication fallback must never append report-(n+1) after the
    # exact sealed report has been observed.
    for path in reports:
        try:
            existing = json.loads(path.read_bytes())
        except (OSError, ValueError) as error:
            raise corpus.CorpusError("terminal report") from error
        if (
            isinstance(existing, dict)
            and existing.get("schema_version") == "phase4e-corpus-terminal-report.v2"
            and existing.get("outcome") == "sealed"
        ):
            raise corpus.CorpusError("sealed terminal report")
    return pilot._append_numbered(report_dir, "report", payload)


def _persist_terminal_outcome(
    *,
    plan: Mapping[str, Any],
    accepted: Sequence[Mapping[str, Any]],
    rejected: Sequence[Mapping[str, Any]],
    ledger: corpus.BudgetLedger,
    work_dir: Path,
    report_dir: Path,
    artifact_root: Path,
    outcome: Literal["minimum_failed", "seal_validation_failed"],
    errors: Sequence[str],
) -> Path:
    _publish_research_capsule(work_dir, artifact_root)
    inventory_sha, _historical = pilot.record_research_catalog(artifact_root)
    return _write_terminal_outcome_report(
        report_dir,
        _terminal_outcome_report(
            plan=plan,
            accepted=accepted,
            rejected=rejected,
            ledger=ledger,
            work_dir=work_dir,
            outcome=outcome,
            errors=errors,
            inventory_sha256=inventory_sha,
        ),
    )


def _publish_research_capsule(work_dir: Path, artifact_root: Path) -> Path:
    """Publish the compact work evidence without importing its full ledger chain."""

    return pilot.create_research_capsule(work_dir, artifact_root / f"{work_dir.name}-capsule")


def _run_post_pilot_batch(
    client: corpus.OpenRouterCorpusClient,
    slots: Sequence[Mapping[str, Any]],
    snapshot: Path,
    api_key: str,
    accepted: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    diagnostics: dict[str, int],
    work_dir: Path,
    *,
    token_counter: corpus.TokenCounter | None = None,
) -> None:
    """Resolve one precommitted author call without replaying uncertain transport."""

    before = {key: diagnostics[key] for key in _POST_PILOT_DIAGNOSTIC_KEYS}
    task_ids = [cast(str, slot["task_id"]) for slot in slots]
    author_lineage: dict[str, str] | None = None
    usable: list[dict[str, Any]] | None = None
    author_rejected: list[dict[str, Any]] | None = None
    for attempt in range(_POST_PILOT_MAXIMUM_ATTEMPTS):
        try:
            response = client._request(
                stage="corpus_author",
                model=POST_PILOT_AUTHOR_MODEL,
                messages=corpus.author_messages(slots),
                response_schema=corpus.author_schema(slots),
                max_output_tokens=1024,
                api_key=api_key,
                task_ids=task_ids,
            )
        except corpus.CorpusError:
            diagnostics["author_response_failures"] += 1
            if client.ledger.entries and client.ledger.entries[-1]["status"] == "uncertain":
                diagnostics["transport_uncertain_calls"] += 1
                lineage = _uncertain_lineage(client.ledger, "author", "corpus_author")
                rows = [
                    {
                        "task_id": slot["task_id"],
                        "split": slot["split"],
                        "reason": "author_transport_uncertain",
                        **lineage,
                    }
                    for slot in slots
                ]
                _store_call_resolution(work_dir, slots, [(row, "rejected") for row in rows])
                _store_post_pilot_diagnostics(
                    work_dir,
                    slots,
                    {key: diagnostics[key] - before[key] for key in _POST_PILOT_DIAGNOSTIC_KEYS},
                )
                rejected.extend(rows)
                return
            author_lineage = corpus.lineage_from_journal(
                client.last_journal("corpus_author"), "author"
            )
            if attempt + 1 < _POST_PILOT_MAXIMUM_ATTEMPTS:
                continue
            break
        author_lineage = corpus.lineage_from_journal(client.last_journal("corpus_author"), "author")
        try:
            decoded = corpus.decode_author_response(_response_content(response), slots)
            counter = token_counter or corpus.VerifiedMiniLMTokenizerReceipt.create(snapshot)
            candidate_usable, candidate_rejected = corpus.classify_author_rows(
                decoded, slots, counter
            )
        except RuntimeError:
            diagnostics["author_response_failures"] += 1
            rows = _store_settled_fallback_required(
                work_dir,
                slots,
                reason="author_validation_failure",
                author_lineage=author_lineage,
                reviewer_lineages={},
            )
            _store_post_pilot_diagnostics(
                work_dir,
                slots,
                {key: diagnostics[key] - before[key] for key in _POST_PILOT_DIAGNOSTIC_KEYS},
            )
            rejected.extend(rows)
            raise
        except corpus.CorpusError:
            diagnostics["author_response_failures"] += 1
            if attempt + 1 < _POST_PILOT_MAXIMUM_ATTEMPTS:
                continue
            break
        usable = candidate_usable
        author_rejected = candidate_rejected
        if attempt:
            diagnostics["retry_recoveries"] += 1
        break
    if usable is None or author_rejected is None or author_lineage is None:
        if author_lineage is None:
            raise corpus.CorpusError("author recovery invariant")
        rows = _store_settled_fallback_required(
            work_dir,
            slots,
            reason="author_response_failure",
            author_lineage=author_lineage,
            reviewer_lineages={},
        )
        _store_post_pilot_diagnostics(
            work_dir,
            slots,
            {key: diagnostics[key] - before[key] for key in _POST_PILOT_DIAGNOSTIC_KEYS},
        )
        rejected.extend(rows)
        return
    usable = [{**row, **author_lineage} for row in usable]
    author_rejected = [{**row, **author_lineage} for row in author_rejected]
    diagnostics["local_privacy"] += sum(
        corpus._privacy(row) for row in [*usable, *author_rejected] if "instruction" in row
    )
    usable, author_rejected = pilot.apply_pair_author_target_rule(slots, usable, author_rejected)
    reviewed_rows: list[dict[str, Any]] = []
    reviews: list[dict[str, Any]] = []
    reviewer_failures: list[dict[str, Any]] = []
    reviewer_lineages: dict[str, dict[str, str]] = {}
    for row in usable:
        task_id = cast(str, row["task_id"])
        review: dict[str, Any] | None = None
        for attempt in range(_POST_PILOT_MAXIMUM_ATTEMPTS):
            try:
                response = client._request(
                    stage="corpus_reviewer",
                    model=POST_PILOT_REVIEWER_MODEL,
                    messages=pilot.pilot_reviewer_messages(row),
                    response_schema=pilot.pilot_reviewer_schema(1),
                    max_output_tokens=512,
                    api_key=api_key,
                    task_ids=[task_id],
                )
            except corpus.CorpusError:
                diagnostics["reviewer_response_failures"] += 1
                if client.ledger.entries and client.ledger.entries[-1]["status"] == "uncertain":
                    diagnostics["transport_uncertain_calls"] += 1
                    reviewer_lineages[task_id] = _uncertain_lineage(
                        client.ledger, "reviewer", "corpus_reviewer"
                    )
                    reviewer_failures.append(
                        {
                            "task_id": task_id,
                            "split": row["split"],
                            "reason": "reviewer_transport_uncertain",
                            **author_lineage,
                            **reviewer_lineages[task_id],
                        }
                    )
                    break
                reviewer_lineages[task_id] = corpus.lineage_from_journal(
                    client.last_journal("corpus_reviewer"), "reviewer"
                )
                if attempt + 1 < _POST_PILOT_MAXIMUM_ATTEMPTS:
                    continue
                break
            reviewer_lineages[task_id] = corpus.lineage_from_journal(
                client.last_journal("corpus_reviewer"), "reviewer"
            )
            try:
                payload = _response_content(response)
                raw = payload.get("reviews")
                if not isinstance(raw, list) or len(raw) != 1 or not isinstance(raw[0], dict):
                    raise corpus.CorpusError("review response")
                review = pilot.bind_pilot_reviewer_record(cast(dict[str, Any], raw[0]), task_id)
            except corpus.CorpusError:
                diagnostics["reviewer_response_failures"] += 1
                if attempt + 1 < _POST_PILOT_MAXIMUM_ATTEMPTS:
                    continue
                break
            if attempt:
                diagnostics["retry_recoveries"] += 1
            break
        if review is None:
            if not any(item["task_id"] == task_id for item in reviewer_failures):
                reviewer_failures.append(
                    {
                        "task_id": task_id,
                        "split": row["split"],
                        "reason": "reviewer_response_failure",
                        **author_lineage,
                        **reviewer_lineages[task_id],
                    }
                )
            continue
        diagnostics["reviewed"] += 1
        diagnostics["reviewer_privacy_flags"] += int(review["private_or_sensitive"])
        scenario, roles = _post_pilot_semantic_diagnostics(row, review)
        diagnostics["scenario_disagreement"] += scenario
        diagnostics["criterion_role_disagreement"] += roles
        reviewed_rows.append(row)
        reviews.append(review)
    newly_accepted, newly_rejected = corpus.resolve_reviews(
        reviewed_rows,
        reviews,
        prior_rows=accepted,
        reviewer_lineages=reviewer_lineages,
        acceptance=corpus.post_pilot_acceptance_reason,
        pair_resolution=corpus.post_pilot_pair_resolution,
        review_model=corpus.PostPilotReviewerRecord,
    )
    outcomes = [
        *[(row, "rejected") for row in author_rejected],
        *[(row, "rejected") for row in reviewer_failures],
        *[(row, "accepted") for row in newly_accepted],
        *[(row, "rejected") for row in newly_rejected],
    ]
    _store_call_resolution(work_dir, slots, outcomes)
    _store_post_pilot_diagnostics(
        work_dir,
        slots,
        {key: diagnostics[key] - before[key] for key in _POST_PILOT_DIAGNOSTIC_KEYS},
    )
    accepted.extend(newly_accepted)
    rejected.extend([*author_rejected, *reviewer_failures, *newly_rejected])


def _v4_base_capsule(artifact_root: Path, plan: Mapping[str, Any]) -> Path:
    candidates: list[Path] = []
    for manifest in sorted(artifact_root.rglob("research-capsule.json")):
        try:
            payload = _read_json(manifest)
        except corpus.CorpusError:
            continue
        if payload.get("schema_version") == "research-capsule.v1" and payload.get(
            "final_ledger_sha256"
        ) == plan.get("base_final_ledger_sha256"):
            candidates.append(manifest.parent)
    if len(candidates) != 1:
        raise corpus.CorpusError("post-pilot recovery base capsule")
    return candidates[0]


def _v4_supplement_resolved(work_dir: Path, plan: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    resolved = _load_resolved(work_dir, plan)
    supplemental = set(cast(list[str], plan["supplemental_task_ids"]))
    if not set(resolved) <= supplemental:
        raise corpus.CorpusError("post-pilot recovery supplemental resolution")
    return resolved


def _reconcile_v4_operational_gap(
    work_dir: Path,
    report_dir: Path,
    plan: Mapping[str, Any],
    ledger: corpus.BudgetLedger,
    resolved: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Close one report-bound trailing provider task set without replaying it."""

    completed = set(resolved)
    gaps: list[tuple[int, Mapping[str, Any], tuple[str, ...]]] = []
    for index, journal in enumerate(ledger.provider_journal):
        task_ids = journal.get("task_ids")
        if not isinstance(task_ids, list) or not all(isinstance(item, str) for item in task_ids):
            raise corpus.CorpusError("settled operational reconciliation")
        ids = tuple(cast(list[str], task_ids))
        missing = set(ids) - completed
        if missing:
            if missing != set(ids):
                raise corpus.CorpusError("settled operational reconciliation")
            gaps.append((index, journal, ids))
    if not gaps:
        return resolved
    gap_ids = {frozenset(item[2]) for item in gaps}
    first_index = gaps[0][0]
    if (
        len(gap_ids) != 1
        or len(next(iter(gap_ids))) != 1
        or [item[0] for item in gaps] != list(range(first_index, len(ledger.provider_journal)))
    ):
        raise corpus.CorpusError("settled operational reconciliation")
    reports = pilot._numbered_files(report_dir, "report")
    snapshots = sorted((work_dir / "ledger").glob("ledger-*.json"))
    if not reports or not snapshots:
        raise corpus.CorpusError("settled operational reconciliation")
    report = _read_json(reports[-1])
    execution_id = cast(str, pilot._execution_marker_payload(work_dir)["execution_id"])
    if (
        report.get("outcome") != "incomplete_operational"
        or report.get("execution_id") != execution_id
        or report.get("supplemental_ledger_sha256")
        != hashlib.sha256(snapshots[-1].read_bytes()).hexdigest()
        or report.get("supplemental_resolution_sha256") != _resolution_sha256(work_dir)
    ):
        raise corpus.CorpusError("settled operational reconciliation")
    reservations = {entry["reservation_id"]: entry for entry in ledger.entries}
    if any(
        journal.get("reservation_id") not in reservations
        or reservations[cast(str, journal["reservation_id"])]["status"]
        not in {"settled", "overspent"}
        for _index, journal, _ids in gaps
    ):
        raise corpus.CorpusError("settled operational reconciliation")
    ids = gaps[-1][2]
    slots_by_id = {
        cast(str, slot["task_id"]): slot for slot in cast(list[Mapping[str, Any]], plan["slots"])
    }
    supplemental = set(cast(list[str], plan["supplemental_task_ids"]))
    if set(ids) - supplemental or any(task_id not in slots_by_id for task_id in ids):
        raise corpus.CorpusError("settled operational reconciliation")
    slots = [slots_by_id[task_id] for task_id in ids]
    final_journal = gaps[-1][1]
    stage = final_journal.get("stage")
    diagnostics = _post_pilot_blank_diagnostics()
    if stage == "corpus_author" and len(gaps) == 1:
        rows = _store_settled_fallback_required(
            work_dir,
            slots,
            reason="author_validation_failure",
            author_lineage=corpus.lineage_from_journal(final_journal, "author"),
            reviewer_lineages={},
        )
        diagnostics["author_response_failures"] = 1
    elif stage == "corpus_reviewer" and len(gaps) == 2:
        author_journal = gaps[0][1]
        if author_journal.get("stage") != "corpus_author" or gaps[0][2] != ids:
            raise corpus.CorpusError("settled operational reconciliation")
        rows = _store_settled_fallback_required(
            work_dir,
            slots,
            reason="review_resolution_failure",
            author_lineage=corpus.lineage_from_journal(author_journal, "author"),
            reviewer_lineages={ids[0]: corpus.lineage_from_journal(final_journal, "reviewer")},
        )
        diagnostics["reviewer_response_failures"] = 1
    else:
        raise corpus.CorpusError("settled operational reconciliation")
    _store_post_pilot_diagnostics(work_dir, slots, diagnostics)
    del rows
    return _v4_supplement_resolved(work_dir, plan)


def _v4_streak(entries: Sequence[Mapping[str, str]], start: int, streak: int) -> int:
    """Advance only from durably closed provider outcomes in call order."""

    for entry in entries[start:]:
        status = entry.get("status")
        if status == "uncertain":
            streak += 1
        elif status in {"settled", "overspent"}:
            streak = 0
        elif status != "reserved":
            raise corpus.CorpusError("transport circuit ledger status")
    return streak


def _v4_capsule_path(work_dir: Path, artifact_root: Path) -> Path:
    ledger_paths = sorted((work_dir / "ledger").glob("ledger-*.json"))
    if not ledger_paths:
        raise corpus.CorpusError("post-pilot ledger missing")
    execution_id = pilot.ensure_research_execution_marker(work_dir)
    return artifact_root / (
        f"supplement-{execution_id}-s{len(ledger_paths):04d}-"
        f"{hashlib.sha256(ledger_paths[-1].read_bytes()).hexdigest()}-"
        f"{_resolution_sha256(work_dir)}"
    )


def _publish_v4_revision(work_dir: Path, artifact_root: Path) -> tuple[Path, str, pilot.SpendScan]:
    capsule = pilot.create_resumable_research_capsule(
        work_dir, _v4_capsule_path(work_dir, artifact_root)
    )
    inventory_sha, scan = pilot.record_research_catalog(artifact_root)
    return capsule, inventory_sha, scan


def _v4_terminal_report(
    *,
    plan: Mapping[str, Any],
    base_accepted: Sequence[Mapping[str, Any]],
    base_rejected: Sequence[Mapping[str, Any]],
    supplemental_accepted: Sequence[Mapping[str, Any]],
    supplemental_rejected: Sequence[Mapping[str, Any]],
    base_ledger: corpus.BudgetLedger,
    supplemental_ledger: corpus.BudgetLedger,
    work_dir: Path,
    outcome: Literal[
        "incomplete_transport_circuit",
        "incomplete_operational",
        "minimum_failed",
        "seal_validation_failed",
        "sealed",
    ],
    errors: Sequence[str],
    inventory_sha256: str,
    base_resolution_sha256: str,
    packet: Path | None = None,
    streak: int = 0,
) -> dict[str, Any]:
    slots = {
        cast(str, slot["task_id"]): slot for slot in cast(list[Mapping[str, Any]], plan["slots"])
    }
    counts: dict[str, dict[str, int]] = {
        key: {} for key in ("split", "locale", "domain", "option_count", "reason")
    }
    for row in [
        *base_accepted,
        *base_rejected,
        *supplemental_accepted,
        *supplemental_rejected,
    ]:
        task_id = row.get("task_id")
        if not isinstance(task_id, str) or task_id not in slots:
            raise corpus.CorpusError("report task identity")
        slot = slots[task_id]
        for key, value in (
            ("split", slot["split"]),
            ("locale", slot["locale"]),
            ("domain", slot["domain"]),
            ("option_count", str(slot["option_count"])),
        ):
            bucket = counts[key]
            text = str(value)
            bucket[text] = bucket.get(text, 0) + 1
    for row in [*base_rejected, *supplemental_rejected]:
        if isinstance(row.get("reason"), str):
            reason = cast(str, row["reason"])
            counts["reason"][reason] = counts["reason"].get(reason, 0) + 1
    supplemental_ids = set(cast(list[str], plan["supplemental_task_ids"]))
    settled = {
        cast(str, row["task_id"]) for row in [*supplemental_accepted, *supplemental_rejected]
    }
    snapshots = sorted((work_dir / "ledger").glob("ledger-*.json"))
    if not snapshots:
        raise corpus.CorpusError("post-pilot ledger missing")
    base_cost = sum(
        (Decimal(entry["provider_cost_usd"]) for entry in base_ledger.entries), Decimal()
    )
    supplemental_cost = sum(
        (Decimal(entry["provider_cost_usd"]) for entry in supplemental_ledger.entries), Decimal()
    )
    supplemental_debit = sum(
        (Decimal(entry["debit_usd"]) for entry in supplemental_ledger.entries), Decimal()
    )
    result: dict[str, Any] = {
        "schema_version": "phase4e-corpus-terminal-report.v2",
        "outcome": outcome,
        "accepted_count": len(base_accepted) + len(supplemental_accepted),
        "rejected_count": len(base_rejected) + len(supplemental_rejected),
        "unresolved_count": len(supplemental_ids - settled),
        "counts": {key: dict(sorted(value.items())) for key, value in counts.items()},
        "errors": sorted(set(errors)),
        "transport_uncertain_calls": sum(
            entry["status"] == "uncertain" for entry in supplemental_ledger.entries
        ),
        "consecutive_uncertainty_streak": streak,
        "base_plan_sha256": plan["base_plan_sha256"],
        "recovery_plan_sha256": corpus._sha(_canonical(plan)),
        "base_ledger_sha256": plan["base_final_ledger_sha256"],
        "base_resolution_sha256": base_resolution_sha256,
        "supplemental_ledger_sha256": hashlib.sha256(snapshots[-1].read_bytes()).hexdigest(),
        "supplemental_resolution_sha256": _resolution_sha256(work_dir),
        "base_provider_reported_cost_usd": str(base_cost),
        "supplemental_provider_reported_cost_usd": str(supplemental_cost),
        "supplemental_conservative_debit_usd": str(supplemental_debit),
        "research_inventory_sha256": inventory_sha256,
        "execution_id": pilot.ensure_research_execution_marker(work_dir),
    }
    if packet is not None:
        result["packet_manifest_sha256"] = hashlib.sha256(
            (packet / "packet.json").read_bytes()
        ).hexdigest()
    return result


def _persist_v4_terminal_report(
    *,
    plan: Mapping[str, Any],
    base_accepted: Sequence[Mapping[str, Any]],
    base_rejected: Sequence[Mapping[str, Any]],
    supplemental_accepted: Sequence[Mapping[str, Any]],
    supplemental_rejected: Sequence[Mapping[str, Any]],
    base_ledger: corpus.BudgetLedger,
    supplemental_ledger: corpus.BudgetLedger,
    work_dir: Path,
    report_dir: Path,
    artifact_root: Path,
    outcome: Literal[
        "incomplete_transport_circuit",
        "incomplete_operational",
        "minimum_failed",
        "seal_validation_failed",
        "sealed",
    ],
    errors: Sequence[str],
    base_resolution_sha256: str,
    packet: Path | None = None,
    streak: int = 0,
) -> Path:
    """Publish a revision before its report, with a fail-closed report on publication failure."""

    try:
        _capsule, inventory, _scan = _publish_v4_revision(work_dir, artifact_root)
    except Exception as publication_error:
        try:
            _write_terminal_outcome_report(
                report_dir,
                _v4_terminal_report(
                    plan=plan,
                    base_accepted=base_accepted,
                    base_rejected=base_rejected,
                    supplemental_accepted=supplemental_accepted,
                    supplemental_rejected=supplemental_rejected,
                    base_ledger=base_ledger,
                    supplemental_ledger=supplemental_ledger,
                    work_dir=work_dir,
                    outcome="incomplete_operational",
                    errors=["research_publication_failed"],
                    inventory_sha256="0" * 64,
                    base_resolution_sha256=base_resolution_sha256,
                    streak=streak,
                ),
            )
        except Exception as report_error:
            raise publication_error from report_error
        raise
    return _write_terminal_outcome_report(
        report_dir,
        _v4_terminal_report(
            plan=plan,
            base_accepted=base_accepted,
            base_rejected=base_rejected,
            supplemental_accepted=supplemental_accepted,
            supplemental_rejected=supplemental_rejected,
            base_ledger=base_ledger,
            supplemental_ledger=supplemental_ledger,
            work_dir=work_dir,
            outcome=outcome,
            errors=errors,
            inventory_sha256=inventory,
            base_resolution_sha256=base_resolution_sha256,
            packet=packet,
            streak=streak,
        ),
    )


def _require_v4_artifact_paths(
    work_dir: Path, packet: Path, report_dir: Path, artifact_root: Path
) -> None:
    pilot.require_live_private_phase4e_root()
    paths = pilot.private_phase4e_paths()
    if (
        artifact_root.resolve() != paths["research_ledgers"].resolve()
        or paths["raw_evidence"].resolve() not in work_dir.resolve().parents
        or paths["model_artifacts"].resolve() not in packet.resolve().parents
        or paths["phase_root"].resolve() not in report_dir.resolve().parents
    ):
        raise corpus.CorpusError("post-pilot recovery artifact path")
    pilot._outside(work_dir, artifact_root)
    pilot._outside(packet, artifact_root)
    pilot._outside(report_dir, artifact_root)


def _run_post_pilot_recovery(
    plan: Mapping[str, Any],
    snapshot: Path,
    work_dir: Path,
    packet: Path,
    report_dir: Path,
    artifact_root: Path,
    *,
    transport: Transport | None,
) -> Path:
    """Run only the v4 supplemental order; base evidence is capsule-only and read-only."""

    require_post_pilot_phase4e_authorization("synthetic_generation")
    _post_pilot_config(plan)
    pilot.require_pilot_research_history(artifact_root)
    base_capsule = _v4_base_capsule(artifact_root, plan)
    base_raw = artifact_root.parent / "raw-evidence" / _POST_PILOT_WORK_DIR.name
    pilot.validate_research_capsule(base_capsule, base_raw)
    base_accepted, _base_rejected, base_ledger = _load_v4_base_capsule(base_capsule, plan)
    base_resolution = _resolution_sha256(base_capsule)
    resolved = _v4_supplement_resolved(work_dir, plan)
    ledger = _load_ledger_or_fail_closed(work_dir)
    if (work_dir / "ledger").exists():
        # Resume only from the complete append-only chain, including the
        # circuit-open snapshot that deliberately leaves later tasks unresolved.
        ledger = pilot._validate_ledger_chain(work_dir / "ledger")
    if not ledger.entries:
        ledger = corpus.BudgetLedger(policy=corpus.POST_PILOT_CORPUS_LEDGER_POLICY)
    elif ledger.policy.schema_version != corpus.POST_PILOT_CORPUS_LEDGER_POLICY.schema_version:
        raise corpus.CorpusError("ledger resume mismatch")
    token_counter: corpus.TokenCounter | None = None
    try:
        _require_settled_task_resolution(ledger, resolved)
    except corpus.CorpusError as error:
        if str(error) != "settled provider call lacks complete task resolution":
            raise
        token_counter = corpus.VerifiedMiniLMTokenizerReceipt.create(snapshot)
        resolved = _reconcile_v4_operational_gap(work_dir, report_dir, plan, ledger, resolved)
        _require_settled_task_resolution(ledger, resolved)
    diagnostics, complete_diagnostics = _load_post_pilot_diagnostics(work_dir, plan, set(resolved))
    if not complete_diagnostics:
        raise corpus.CorpusError("post-pilot diagnostics incomplete")
    _require_post_pilot_uncertain_resolutions(
        ledger, resolved, diagnostics["transport_uncertain_calls"]
    )
    supplemental_ids = set(cast(list[str], plan["supplemental_task_ids"]))
    accepted = [
        cast(dict[str, Any], item["row"])
        for item in resolved.values()
        if item["status"] == "accepted"
    ]
    rejected = [
        cast(dict[str, Any], item["row"])
        for item in resolved.values()
        if item["status"] == "rejected"
    ]
    if set(resolved) != supplemental_ids:
        if token_counter is None:
            token_counter = corpus.VerifiedMiniLMTokenizerReceipt.create(snapshot)
        _aggregate_preflight(plan, resolved, ledger)
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise corpus.CorpusError("missing provider credential")
        execution_id = pilot.ensure_research_execution_marker(work_dir)
        del execution_id
        config = cast(dict[str, Any], _post_pilot_config(plan))
        client = corpus.OpenRouterCorpusClient(
            transport=cast(corpus.Transport, transport or _openrouter_transport(120)),
            allow_network=True,
            ledger=ledger,
            ledger_directory=work_dir / "ledger",
            policy=corpus.POST_PILOT_CORPUS_LEDGER_POLICY,
            author_model=POST_PILOT_AUTHOR_MODEL,
            reviewer_model=POST_PILOT_REVIEWER_MODEL,
            provider_preferences_by_stage={
                "corpus_author": config["provider_policy"],
                "corpus_reviewer": config["provider_policy"],
            },
            author_reasoning_effort=None,
            reviewer_temperature=0,
        )
        completed = set(resolved)
        streak = 0
        next_cost = Decimal("10.00")
        try:
            for slots in _author_batches(plan):
                ids = {cast(str, slot["task_id"]) for slot in slots}
                if ids <= completed:
                    continue
                if ids & completed:
                    raise corpus.CorpusError("partial author family resume")
                before = len(client.ledger.entries)
                _run_post_pilot_batch(
                    client,
                    slots,
                    snapshot,
                    api_key,
                    [*base_accepted, *accepted],
                    rejected,
                    diagnostics,
                    work_dir,
                    token_counter=token_counter,
                )
                # The author/reviewer resolver receives base rows for dedup, but only
                # supplemental additions belong in this resume work directory.
                current = _v4_supplement_resolved(work_dir, plan)
                accepted = [
                    cast(dict[str, Any], item["row"])
                    for item in current.values()
                    if item["status"] == "accepted"
                ]
                rejected = [
                    cast(dict[str, Any], item["row"])
                    for item in current.values()
                    if item["status"] == "rejected"
                ]
                completed = set(current)
                streak = _v4_streak(client.ledger.entries, before, streak)
                spend = sum(
                    (Decimal(entry["provider_cost_usd"]) for entry in client.ledger.entries),
                    Decimal(),
                )
                while spend >= next_cost:
                    print(f"phase4e recovery provider spend crossed USD {next_cost}", flush=True)
                    next_cost += Decimal("10.00")
                if streak >= 3:
                    corpus.write_ledger_snapshot(
                        work_dir / "ledger", client.ledger, stop_reason="transport_circuit_open"
                    )
                    _persist_v4_terminal_report(
                        plan=plan,
                        base_accepted=base_accepted,
                        base_rejected=_base_rejected,
                        supplemental_accepted=accepted,
                        supplemental_rejected=rejected,
                        base_ledger=base_ledger,
                        supplemental_ledger=client.ledger,
                        work_dir=work_dir,
                        report_dir=report_dir,
                        artifact_root=artifact_root,
                        outcome="incomplete_transport_circuit",
                        errors=["transport_circuit_open"],
                        base_resolution_sha256=base_resolution,
                        streak=streak,
                    )
                    print(f"phase4e recovery final provider spend USD {spend}", flush=True)
                    raise corpus.CorpusError("transport circuit open")
        except Exception as error:
            if isinstance(error, corpus.CorpusError) and str(error) == "transport circuit open":
                raise
            if client.ledger.entries:
                try:
                    durable = _v4_supplement_resolved(work_dir, plan)
                    accepted = [
                        cast(dict[str, Any], item["row"])
                        for item in durable.values()
                        if item["status"] == "accepted"
                    ]
                    rejected = [
                        cast(dict[str, Any], item["row"])
                        for item in durable.values()
                        if item["status"] == "rejected"
                    ]
                    _persist_v4_terminal_report(
                        plan=plan,
                        base_accepted=base_accepted,
                        base_rejected=_base_rejected,
                        supplemental_accepted=accepted,
                        supplemental_rejected=rejected,
                        base_ledger=base_ledger,
                        supplemental_ledger=client.ledger,
                        work_dir=work_dir,
                        report_dir=report_dir,
                        artifact_root=artifact_root,
                        outcome="incomplete_operational",
                        errors=["operational_failure"],
                        base_resolution_sha256=base_resolution,
                        streak=streak,
                    )
                    spend = sum(
                        (Decimal(entry["provider_cost_usd"]) for entry in client.ledger.entries),
                        Decimal(),
                    )
                    print(f"phase4e recovery final provider spend USD {spend}", flush=True)
                except Exception as report_error:
                    raise error from report_error
            raise
        ledger = client.ledger
    final_spend = sum((Decimal(entry["provider_cost_usd"]) for entry in ledger.entries), Decimal())
    try:
        resolved = _v4_supplement_resolved(work_dir, plan)
        if set(resolved) != supplemental_ids:
            raise corpus.CorpusError("incomplete corpus resolution")
        diagnostics, complete_diagnostics = _load_post_pilot_diagnostics(
            work_dir, plan, set(resolved)
        )
        if not complete_diagnostics:
            raise corpus.CorpusError("post-pilot diagnostics incomplete")
        _require_post_pilot_uncertain_resolutions(
            ledger, resolved, diagnostics["transport_uncertain_calls"]
        )
        _require_settled_task_resolution(ledger, resolved)
        accepted = [
            cast(dict[str, Any], item["row"])
            for item in resolved.values()
            if item["status"] == "accepted"
        ]
        rejected = [
            cast(dict[str, Any], item["row"])
            for item in resolved.values()
            if item["status"] == "rejected"
        ]
    except Exception as error:
        if ledger.entries:
            try:
                durable = _v4_supplement_resolved(work_dir, plan)
                accepted = [
                    cast(dict[str, Any], item["row"])
                    for item in durable.values()
                    if item["status"] == "accepted"
                ]
                rejected = [
                    cast(dict[str, Any], item["row"])
                    for item in durable.values()
                    if item["status"] == "rejected"
                ]
                _persist_v4_terminal_report(
                    plan=plan,
                    base_accepted=base_accepted,
                    base_rejected=_base_rejected,
                    supplemental_accepted=accepted,
                    supplemental_rejected=rejected,
                    base_ledger=base_ledger,
                    supplemental_ledger=ledger,
                    work_dir=work_dir,
                    report_dir=report_dir,
                    artifact_root=artifact_root,
                    outcome="incomplete_operational",
                    errors=["operational_failure"],
                    base_resolution_sha256=base_resolution,
                )
                print(f"phase4e recovery final provider spend USD {final_spend}", flush=True)
            except Exception as report_error:
                raise error from report_error
        raise
    snapshots = sorted((work_dir / "ledger").glob("ledger-*.json"))
    if not snapshots or _read_json(snapshots[-1]).get("stop_reason") != "complete":
        corpus.write_ledger_snapshot(work_dir / "ledger", ledger, stop_reason="complete")
    minimum_errors = corpus._minimums([*base_accepted, *accepted])
    if minimum_errors:
        _persist_v4_terminal_report(
            plan=plan,
            base_accepted=base_accepted,
            base_rejected=_base_rejected,
            supplemental_accepted=accepted,
            supplemental_rejected=rejected,
            base_ledger=base_ledger,
            supplemental_ledger=ledger,
            work_dir=work_dir,
            report_dir=report_dir,
            artifact_root=artifact_root,
            outcome="minimum_failed",
            errors=minimum_errors,
            base_resolution_sha256=base_resolution,
        )
        print(f"phase4e recovery final provider spend USD {final_spend}", flush=True)
        raise corpus.CorpusError("minimum_failed")
    try:
        if not packet.exists():
            corpus.seal_post_pilot_recovery_packet(
                packet,
                plan,
                base_accepted,
                _base_rejected,
                base_ledger.as_json(final=True),
                accepted,
                rejected,
                ledger.as_json(final=True),
            )
        else:
            corpus.validate_post_pilot_recovery_packet_binding(
                packet,
                plan,
                base_accepted,
                _base_rejected,
                base_ledger.as_json(final=True),
                accepted,
                rejected,
                ledger.as_json(final=True),
            )
    except (corpus.CorpusError, FileExistsError) as error:
        try:
            _persist_v4_terminal_report(
                plan=plan,
                base_accepted=base_accepted,
                base_rejected=_base_rejected,
                supplemental_accepted=accepted,
                supplemental_rejected=rejected,
                base_ledger=base_ledger,
                supplemental_ledger=ledger,
                work_dir=work_dir,
                report_dir=report_dir,
                artifact_root=artifact_root,
                outcome="seal_validation_failed",
                errors=["seal_validation_failed"],
                base_resolution_sha256=base_resolution,
            )
        except Exception as report_error:
            raise error from report_error
        print(f"phase4e recovery final provider spend USD {final_spend}", flush=True)
        raise
    _persist_v4_terminal_report(
        plan=plan,
        base_accepted=base_accepted,
        base_rejected=_base_rejected,
        supplemental_accepted=accepted,
        supplemental_rejected=rejected,
        base_ledger=base_ledger,
        supplemental_ledger=ledger,
        work_dir=work_dir,
        report_dir=report_dir,
        artifact_root=artifact_root,
        outcome="sealed",
        errors=[],
        base_resolution_sha256=base_resolution,
        packet=packet,
    )
    print(f"phase4e recovery final provider spend USD {final_spend}", flush=True)
    return packet / "packet.json"


def _run_post_pilot_corpus(
    plan: Mapping[str, Any],
    snapshot: Path,
    work_dir: Path,
    packet: Path,
    report_dir: Path,
    artifact_root: Path,
    *,
    transport: Transport | None,
) -> Path:
    """Continue or seal the v3 corpus; a completed work tree never opens transport."""

    require_post_pilot_phase4e_authorization("synthetic_generation")
    _post_pilot_config(plan)
    resolved = _load_resolved(work_dir, plan)
    ledger = _load_ledger_or_fail_closed(work_dir)
    if not ledger.entries:
        ledger = corpus.BudgetLedger(policy=corpus.POST_PILOT_CORPUS_LEDGER_POLICY)
    elif ledger.policy.schema_version != corpus.POST_PILOT_CORPUS_LEDGER_POLICY.schema_version:
        raise corpus.CorpusError("ledger resume mismatch")
    _require_settled_task_resolution(ledger, resolved)
    diagnostics, complete_diagnostics = _load_post_pilot_diagnostics(work_dir, plan, set(resolved))
    if not complete_diagnostics:
        raise corpus.CorpusError("post-pilot diagnostics incomplete")
    _require_post_pilot_uncertain_resolutions(
        ledger, resolved, diagnostics["transport_uncertain_calls"]
    )
    all_task_ids = {
        cast(str, slot["task_id"]) for slot in cast(list[Mapping[str, Any]], plan["slots"])
    }
    if set(resolved) != all_task_ids:
        _aggregate_preflight(plan, resolved, ledger)
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise corpus.CorpusError("missing provider credential")
        config = cast(dict[str, Any], _post_pilot_config(plan))
        work_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        client = corpus.OpenRouterCorpusClient(
            transport=cast(corpus.Transport, transport or _openrouter_transport(120)),
            allow_network=True,
            ledger=ledger,
            ledger_directory=work_dir / "ledger",
            policy=corpus.POST_PILOT_CORPUS_LEDGER_POLICY,
            author_model=POST_PILOT_AUTHOR_MODEL,
            reviewer_model=POST_PILOT_REVIEWER_MODEL,
            provider_preferences_by_stage={
                "corpus_author": config["provider_policy"],
                "corpus_reviewer": config["provider_policy"],
            },
            author_reasoning_effort=None,
            reviewer_temperature=0,
        )
        accepted = [
            cast(dict[str, Any], item["row"])
            for item in resolved.values()
            if item["status"] == "accepted"
        ]
        rejected = [
            cast(dict[str, Any], item["row"])
            for item in resolved.values()
            if item["status"] == "rejected"
        ]
        completed = set(resolved)
        next_cost = Decimal("10.00")
        for slots in _author_batches(plan):
            ids = {cast(str, slot["task_id"]) for slot in slots}
            if ids <= completed:
                continue
            if ids & completed:
                raise corpus.CorpusError("partial author family resume")
            _run_post_pilot_batch(
                client, slots, snapshot, api_key, accepted, rejected, diagnostics, work_dir
            )
            completed.update(ids)
            spend = sum(
                (Decimal(entry["provider_cost_usd"]) for entry in client.ledger.entries),
                Decimal(),
            )
            while spend >= next_cost:
                print(f"phase4e corpus provider spend crossed USD {next_cost}", flush=True)
                next_cost += Decimal("10.00")
        ledger = client.ledger
        resolved = _load_resolved(work_dir, plan)
    if set(resolved) != all_task_ids:
        raise corpus.CorpusError("incomplete corpus resolution")
    diagnostics, complete_diagnostics = _load_post_pilot_diagnostics(work_dir, plan, set(resolved))
    if not complete_diagnostics:
        raise corpus.CorpusError("post-pilot diagnostics incomplete")
    _require_post_pilot_uncertain_resolutions(
        ledger, resolved, diagnostics["transport_uncertain_calls"]
    )
    _require_settled_task_resolution(ledger, resolved)
    accepted = [
        cast(dict[str, Any], item["row"])
        for item in resolved.values()
        if item["status"] == "accepted"
    ]
    rejected = [
        cast(dict[str, Any], item["row"])
        for item in resolved.values()
        if item["status"] == "rejected"
    ]
    snapshots = sorted((work_dir / "ledger").glob("ledger-*.json"))
    if not snapshots or _read_json(snapshots[-1]).get("stop_reason") != "complete":
        corpus.write_ledger_snapshot(work_dir / "ledger", ledger, stop_reason="complete")
    minimum_errors = corpus._minimums(accepted)
    if minimum_errors:
        if packet.exists():
            raise corpus.CorpusError("minimum failed packet present")
        _persist_terminal_outcome(
            plan=plan,
            accepted=accepted,
            rejected=rejected,
            ledger=ledger,
            work_dir=work_dir,
            report_dir=report_dir,
            artifact_root=artifact_root,
            outcome="minimum_failed",
            errors=minimum_errors,
        )
        provider_cost = sum(
            (Decimal(entry["provider_cost_usd"]) for entry in ledger.entries), Decimal()
        )
        print(f"phase4e corpus final provider spend USD {provider_cost}", flush=True)
        raise corpus.CorpusError("minimum_failed")
    if not packet.exists():
        try:
            corpus.seal_packet(packet, plan, accepted, rejected, ledger.as_json(final=True))
        except corpus.CorpusError as error:
            try:
                _persist_terminal_outcome(
                    plan=plan,
                    accepted=accepted,
                    rejected=rejected,
                    ledger=ledger,
                    work_dir=work_dir,
                    report_dir=report_dir,
                    artifact_root=artifact_root,
                    outcome="seal_validation_failed",
                    errors=["seal_validation_failed"],
                )
            except Exception as report_error:
                raise error from report_error
            raise
    else:
        corpus.validate_accepted_packet_pre_holdout(packet)
    _publish_research_capsule(work_dir, artifact_root)
    inventory_sha, historical = pilot.record_research_catalog(artifact_root)
    report_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    report_path = report_dir / "report.json"
    report = _post_pilot_report(
        plan=plan,
        accepted=accepted,
        rejected=rejected,
        ledger=ledger,
        diagnostics=diagnostics,
        inventory_sha256=inventory_sha,
        packet=packet,
        work_dir=work_dir,
        historical=historical,
    )
    if not report_path.exists():
        atomic_create(report_path, _canonical(report) + b"\n")
    elif report_path.read_bytes() != _canonical(report) + b"\n":
        raise FileExistsError(f"benchmark artifact already exists: {report_path.name}")
    print(
        f"phase4e corpus final provider spend USD {report['provider_reported_cost_usd']}",
        flush=True,
    )
    return packet / "packet.json"


def run_corpus(
    plan_path: Path,
    snapshot: Path,
    work_dir: Path,
    packet: Path,
    *,
    allow_network: bool,
    transport: Transport | None = None,
    report_dir: Path | None = None,
    artifact_root: Path | None = None,
) -> Path:
    """Execute the deliberate provider lane with no-clobber task resolution."""
    if not allow_network:
        raise corpus.CorpusError("corpus requires literal --allow-network")
    # v2 is frozen behind its historical gate; v3 carries the reviewed
    # synthetic-generation exception but not training/runtime authorization.
    if not plan_path.exists():
        # Preserve the historical no-IO authorization boundary for a missing
        # legacy plan; a v3 plan is selected only after its bytes are read.
        require_phase4e_authorization("synthetic_generation")
    plan = _read_plan(plan_path)
    if plan.get("schema_version") == "phase4e-universal-plan.v4":
        if report_dir is None or artifact_root is None:
            raise corpus.CorpusError(
                "post-pilot recovery requires report and durable research root"
            )
        _require_v4_artifact_paths(work_dir, packet, report_dir, artifact_root)
        return _run_post_pilot_recovery(
            plan,
            snapshot,
            work_dir,
            packet,
            report_dir,
            artifact_root,
            transport=transport,
        )
    post_pilot = plan.get("schema_version") == "phase4e-universal-plan.v3"
    if post_pilot:
        if report_dir is None or artifact_root is None:
            raise corpus.CorpusError("post-pilot corpus requires report and durable research root")
        _require_post_pilot_artifact_paths(
            plan_path,
            work_dir,
            packet,
            report_dir,
            artifact_root,
        )
        return _run_post_pilot_corpus(
            plan,
            snapshot,
            work_dir,
            packet,
            report_dir,
            artifact_root,
            transport=transport,
        )
    else:
        require_phase4e_authorization("synthetic_generation")
    # This is the only environment read in this module.  It is never accepted
    # by argparse, persisted, included in an exception, or printed.
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise corpus.CorpusError("missing provider credential")
    config = _post_pilot_config(plan)
    work_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    if packet.exists():
        raise FileExistsError(f"benchmark artifact already exists: {packet.name}")
    resolved = _load_resolved(work_dir, plan)
    ledger = _load_ledger_or_fail_closed(work_dir)
    _require_settled_task_resolution(ledger, resolved)
    _aggregate_preflight(plan, resolved, ledger)
    client = corpus.OpenRouterCorpusClient(
        transport=cast(corpus.Transport, transport or _openrouter_transport()),
        allow_network=True,
        ledger=ledger,
        ledger_directory=work_dir / "ledger",
        policy=corpus.POST_PILOT_CORPUS_LEDGER_POLICY if post_pilot else None,
        author_model=POST_PILOT_AUTHOR_MODEL if post_pilot else corpus.AUTHOR_MODEL,
        reviewer_model=POST_PILOT_REVIEWER_MODEL if post_pilot else corpus.REVIEWER_MODEL,
        provider_preferences_by_stage=(
            {
                "corpus_author": config["provider_policy"],
                "corpus_reviewer": config["provider_policy"],
            }
            if config is not None
            else None
        ),
        author_reasoning_effort=None if post_pilot else "none",
        reviewer_temperature=0,
    )
    accepted: list[dict[str, Any]] = [
        cast(dict[str, Any], value["row"])
        for value in resolved.values()
        if value["status"] == "accepted"
    ]
    rejected: list[dict[str, Any]] = [
        cast(dict[str, Any], value["row"])
        for value in resolved.values()
        if value["status"] == "rejected"
    ]
    completed = set(resolved)
    batches = _author_batches(plan)
    for index, slots in enumerate(batches, start=1):
        if all(cast(str, slot["task_id"]) in completed for slot in slots):
            continue
        if any(cast(str, slot["task_id"]) in completed for slot in slots):
            raise corpus.CorpusError("partial author family resume")
        messages = corpus.author_messages(slots)
        author_max_output_tokens = 1024 if post_pilot else _author_max_output_tokens(slots)
        _preflight_request(
            "corpus_author",
            _provider_body(
                stage="corpus_author",
                model=POST_PILOT_AUTHOR_MODEL if post_pilot else corpus.AUTHOR_MODEL,
                messages=messages,
                response_schema=corpus.author_schema(slots),
                max_output_tokens=author_max_output_tokens,
                author_model=POST_PILOT_AUTHOR_MODEL if post_pilot else corpus.AUTHOR_MODEL,
                provider_override=config["provider_policy"] if config is not None else None,
                author_reasoning_effort=None if post_pilot else "none",
            ),
            author_max_output_tokens,
        )
        try:
            author_response = _response_content(
                client.author(
                    slots=slots, max_output_tokens=author_max_output_tokens, api_key=api_key
                )
            )
        except Exception:
            settled_author_lineage = _lineage_or_none(
                client, "author", [cast(str, slot["task_id"]) for slot in slots]
            )
            if settled_author_lineage is not None:
                _store_settled_fallback_required(
                    work_dir,
                    slots,
                    reason="author_response_failure",
                    author_lineage=settled_author_lineage,
                    reviewer_lineages={},
                )
            raise
        try:
            author_lineage = corpus.lineage_from_journal(
                client.last_journal("corpus_author"), "author"
            )
            authored = corpus.decode_author_response(author_response, slots)
            usable, capacity_rejected = corpus.validate_author_rows_with_verified_minilm(
                authored, slots, snapshot
            )
        except Exception:
            settled_author_lineage = _lineage_or_none(
                client, "author", [cast(str, slot["task_id"]) for slot in slots]
            )
            if settled_author_lineage is not None:
                _store_settled_fallback_required(
                    work_dir,
                    slots,
                    reason="author_validation_failure",
                    author_lineage=settled_author_lineage,
                    reviewer_lineages={},
                )
            raise
        usable = [{**row, **author_lineage} for row in usable]
        capacity_rejected = [{**row, **author_lineage} for row in capacity_rejected]
        review_results: list[tuple[dict[str, Any], dict[str, Any]]] = []
        reviewer_lineages: dict[str, dict[str, str]] = {}
        for row in usable:
            try:
                review, journal = _review_one(client, row, api_key, post_pilot=post_pilot)
            except Exception:
                settled_reviewer_lineage = _lineage_or_none(
                    client, "reviewer", [cast(str, row["task_id"])]
                )
                if settled_reviewer_lineage is not None:
                    reviewer_lineages[cast(str, row["task_id"])] = settled_reviewer_lineage
                _store_settled_fallback_required(
                    work_dir,
                    slots,
                    reason="reviewer_response_failure",
                    author_lineage=author_lineage,
                    reviewer_lineages=reviewer_lineages,
                )
                raise
            review_results.append((review, journal))
            reviewer_lineages[cast(str, row["task_id"])] = corpus.lineage_from_journal(
                journal, "reviewer"
            )
        reviews = [review for review, _ in review_results]
        try:
            newly_accepted, newly_rejected = corpus.resolve_reviews(
                usable,
                reviews,
                prior_rows=accepted,
                reviewer_lineages=reviewer_lineages,
                acceptance=corpus.post_pilot_acceptance_reason if post_pilot else None,
                pair_resolution=corpus.post_pilot_pair_resolution if post_pilot else None,
                review_model=corpus.PostPilotReviewerRecord
                if post_pilot
                else corpus.ReviewerRecord,
            )
        except Exception:
            _store_settled_fallback_required(
                work_dir,
                slots,
                reason="review_resolution_failure",
                author_lineage=author_lineage,
                reviewer_lineages=reviewer_lineages,
            )
            raise
        normal_resolutions = [
            *[(item, "rejected") for item in capacity_rejected],
            *[(item, "accepted") for item in newly_accepted],
            *[(item, "rejected") for item in newly_rejected],
        ]
        try:
            _store_call_resolution(work_dir, slots, normal_resolutions)
        except Exception:
            _store_settled_fallback_required(
                work_dir,
                slots,
                reason="result_persistence_failure",
                author_lineage=author_lineage,
                reviewer_lineages=reviewer_lineages,
            )
            raise
        accepted.extend(newly_accepted)
        rejected.extend([*capacity_rejected, *newly_rejected])
        completed.update(cast(str, item["task_id"]) for item, _ in normal_resolutions)
        if index % 10 == 0 and len(completed) >= 20:
            accepted_ids = {cast(str, row["task_id"]) for row in accepted}
            resolution_statuses = [
                {
                    "task_id": task_id,
                    "status": "accepted" if task_id in accepted_ids else "rejected",
                }
                for task_id in sorted(completed)
            ]
            projection = corpus.stop_projection(
                resolution_statuses,
                plan,
            )
            _write_wilson_projection(work_dir / "ledger", plan, resolution_statuses, projection)
            if projection is not None:
                corpus.write_ledger_snapshot(
                    work_dir / "ledger", client.ledger, stop_reason=projection
                )
                raise corpus.CorpusError(f"Wilson stop: {projection}")
    if completed != {
        cast(str, slot["task_id"]) for slot in cast(list[Mapping[str, Any]], plan["slots"])
    }:
        raise corpus.CorpusError("incomplete corpus resolution")
    corpus.write_ledger_snapshot(work_dir / "ledger", client.ledger, stop_reason="complete")
    return corpus.seal_packet(packet, plan, accepted, rejected, client.ledger.as_json(final=True))


def run_extract(packet: Path, snapshot: Path, device: str, output: Path) -> Path:
    return training.extract_and_seal_embeddings(packet, snapshot, device, output)


def run_train(
    packet: Path,
    snapshot: Path,
    embeddings: Path,
    device: str,
    output_parent: Path,
    output_name: str,
) -> Path:
    return training.train_and_seal(embeddings, packet, snapshot, device, output_parent, output_name)


def run_verify(packet: Path, embeddings: Path, training_path: Path) -> None:
    binding = training.validate_accepted_packet_binding(packet)
    capsule = training.validate_embedding_capsule(embeddings, binding)
    manifest = training.verify_training_capsule(training_path)
    if (
        manifest.get("packet_receipt_sha256") != capsule.manifest["packet_receipt_sha256"]
        or manifest.get("embedding_descriptor_sha256") != capsule.descriptor_sha256
    ):
        raise training.TrainingError("training and embedding lineage binding")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m benchmarks.phase4e_pipeline")
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--output", type=Path, required=True)
    corpus_parser = commands.add_parser("corpus")
    corpus_parser.add_argument("--plan", type=Path, required=True)
    corpus_parser.add_argument("--snapshot", type=Path, required=True)
    corpus_parser.add_argument("--work-dir", type=Path, required=True)
    corpus_parser.add_argument("--packet", type=Path, required=True)
    # They are mandatory for the v3 plan at runtime; keeping parsing permissive
    # preserves the historical missing-plan/no-network failure order.
    corpus_parser.add_argument("--report", type=Path)
    corpus_parser.add_argument("--artifact-root", type=Path)
    corpus_parser.add_argument("--allow-network", action="store_true")
    extract = commands.add_parser("extract")
    extract.add_argument("--packet", type=Path, required=True)
    extract.add_argument("--snapshot", type=Path, required=True)
    extract.add_argument("--device", choices=("cpu",), required=True)
    extract.add_argument("--output", type=Path, required=True)
    train = commands.add_parser("train")
    train.add_argument("--packet", type=Path, required=True)
    train.add_argument("--snapshot", type=Path, required=True)
    train.add_argument("--embeddings", type=Path, required=True)
    train.add_argument("--device", choices=("cpu",), required=True)
    train.add_argument("--output-parent", type=Path, required=True)
    train.add_argument("--output-name", required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--packet", type=Path, required=True)
    verify.add_argument("--embeddings", type=Path, required=True)
    verify.add_argument("--training", type=Path, required=True)
    pilot_plan = commands.add_parser("pilot-plan")
    pilot_plan.add_argument("--output", type=Path, required=True)
    pilot_run = commands.add_parser("pilot")
    pilot_run.add_argument("--plan", type=Path, required=True)
    pilot_run.add_argument("--snapshot", type=Path, required=True)
    pilot_run.add_argument("--work-dir", type=Path, required=True)
    pilot_run.add_argument("--report", type=Path, required=True)
    pilot_run.add_argument("--artifact-root", type=Path, required=True)
    pilot_run.add_argument("--allow-network", action="store_true")
    import_ledgers = commands.add_parser("import-ledgers")
    import_ledgers.add_argument("--source", type=Path, required=True)
    import_ledgers.add_argument("--artifact-root", type=Path, required=True)
    commands.add_parser("recovery-prepare")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "plan":
            write_post_pilot_plan(args.output)
        elif args.command == "corpus":
            run_corpus(
                args.plan,
                args.snapshot,
                args.work_dir,
                args.packet,
                allow_network=args.allow_network,
                report_dir=args.report,
                artifact_root=args.artifact_root,
            )
        elif args.command == "extract":
            run_extract(args.packet, args.snapshot, args.device, args.output)
        elif args.command == "train":
            run_train(
                args.packet,
                args.snapshot,
                args.embeddings,
                args.device,
                args.output_parent,
                args.output_name,
            )
        elif args.command == "verify":
            run_verify(args.packet, args.embeddings, args.training)
        elif args.command == "pilot-plan":
            pilot.write_pilot_plan(args.output)
        elif args.command == "pilot":
            pilot.run_pilot(
                args.plan,
                args.snapshot,
                args.work_dir,
                args.report,
                args.artifact_root,
                allow_network=args.allow_network,
            )
        elif args.command == "import-ledgers":
            pilot.import_research_artifacts(args.source, args.artifact_root)
        elif args.command == "recovery-prepare":
            run_prepare_recovery()
        else:  # argparse makes this unreachable; keep it fail-closed.
            raise ValueError("unknown pipeline command")
    except (
        corpus.CorpusError,
        training.TrainingError,
        FileExistsError,
        OSError,
        ValueError,
    ) as error:
        print(f"phase4e pipeline failed: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
