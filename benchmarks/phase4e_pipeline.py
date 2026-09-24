"""Operational, offline-first entry point for the Phase 4E research lane.

The module is deliberately outside the installed package.  Only ``corpus`` has
an HTTP implementation; the other commands neither import nor construct a
socket.  Provider replies are reduced to typed rows and digests in the corpus
ledger -- raw replies and credentials are never written to disk.
"""

from __future__ import annotations

import argparse
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
from benchmarks import saracura_universal_training as training
from benchmarks.io import atomic_create
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
    """Create one immutable public-seed plan without a network path."""
    atomic_create(output, _canonical(corpus.build_plan()) + b"\n")
    return output


def _provider_body(
    *,
    stage: Literal["corpus_author", "corpus_reviewer"],
    model: str,
    messages: Sequence[Mapping[str, str]],
    response_schema: Mapping[str, Any],
    max_output_tokens: int,
) -> bytes:
    request: dict[str, Any] = {
        "model": model,
        "messages": list(messages),
        "max_tokens": max_output_tokens,
        "temperature": 0,
        "provider": corpus.provider_preferences(stage),
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": corpus.RESPONSE_SCHEMA_NAME,
                "strict": True,
                "schema": response_schema,
            },
        },
    }
    if stage == "corpus_author" and model == corpus.AUTHOR_MODEL:
        request["reasoning_effort"] = "none"
    return _canonical(request)


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
    author_total = Decimal()
    reviewer_total = Decimal()
    for slots in _author_batches(plan):
        task_ids = {cast(str, slot["task_id"]) for slot in slots}
        if task_ids <= completed:
            continue
        if task_ids & completed:
            raise corpus.CorpusError("partial author family resume")
        max_output_tokens = _author_max_output_tokens(slots)
        author_total += corpus.request_worst_case(
            "corpus_author",
            _provider_body(
                stage="corpus_author",
                model=corpus.AUTHOR_MODEL,
                messages=corpus.author_messages(slots),
                response_schema=corpus.author_schema(slots),
                max_output_tokens=max_output_tokens,
            ),
            max_output_tokens,
        )
        for slot in slots:
            row = _boundary_reviewer_row(slot)
            reviewer_total += corpus.request_worst_case(
                "corpus_reviewer",
                _provider_body(
                    stage="corpus_reviewer",
                    model=corpus.REVIEWER_MODEL,
                    messages=corpus.reviewer_messages([row]),
                    response_schema=corpus.reviewer_schema(1),
                    max_output_tokens=_REVIEWER_MAX_OUTPUT_TOKENS,
                ),
                _REVIEWER_MAX_OUTPUT_TOKENS,
            )
    totals = {"corpus_author": author_total, "corpus_reviewer": reviewer_total}
    for stage, total in totals.items():
        ledger.reserve(stage, total)
    if (
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
        finish_reason = choice.get("finish_reason", "unknown")
    except (KeyError, IndexError, TypeError) as error:
        raise corpus.CorpusError("provider response content (missing)") from error
    if not isinstance(message, Mapping):
        raise corpus.CorpusError("provider response content (message)")
    if message.get("reasoning") not in (None, "", []) or message.get("reasoning_details") not in (
        None,
        [],
    ):
        raise corpus.CorpusError("provider response reasoning")
    if not isinstance(content, str):
        raise corpus.CorpusError(
            f"provider response content ({type(content).__name__}, {finish_reason})"
        )
    try:
        value = json.loads(content)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise corpus.CorpusError(f"provider response JSON ({finish_reason})") from error
    if not isinstance(value, dict):
        raise corpus.CorpusError("provider response content")
    return cast(dict[str, Any], value)


def _openrouter_transport() -> Transport:
    """Create the only live transport: fixed HTTPS, no proxy, no redirects."""

    context = ssl.create_default_context()

    def transport(
        method: str, url: str, headers: Mapping[str, str], body: bytes
    ) -> tuple[int, Mapping[str, str], bytes]:
        if method != "POST" or url != f"https://{_OPENROUTER_HOST}{_OPENROUTER_PATH}":
            raise corpus.CorpusError("pinned provider request")
        # http.client has neither proxy discovery nor redirect-following.
        connection = http.client.HTTPSConnection(_OPENROUTER_HOST, 443, context=context, timeout=30)
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
) -> tuple[dict[str, Any], dict[str, Any]]:
    typed = corpus.ValidatedAuthorRow.model_validate(row)
    messages = corpus.reviewer_messages([typed])
    _preflight_request(
        "corpus_reviewer",
        _provider_body(
            stage="corpus_reviewer",
            model=corpus.REVIEWER_MODEL,
            messages=messages,
            response_schema=corpus.reviewer_schema(1),
            max_output_tokens=_REVIEWER_MAX_OUTPUT_TOKENS,
        ),
        _REVIEWER_MAX_OUTPUT_TOKENS,
    )
    payload = _response_content(
        client.reviewer(row=typed, max_output_tokens=_REVIEWER_MAX_OUTPUT_TOKENS, api_key=api_key)
    )
    reviews = payload.get("reviews")
    if not isinstance(reviews, list) or len(reviews) != 1 or not isinstance(reviews[0], dict):
        raise corpus.CorpusError("review response")
    return (
        corpus.materialize_reviewer_record(cast(dict[str, Any], reviews[0]), typed),
        client.last_journal("corpus_reviewer"),
    )


def run_corpus(
    plan_path: Path,
    snapshot: Path,
    work_dir: Path,
    packet: Path,
    *,
    allow_network: bool,
    transport: Transport | None = None,
) -> Path:
    """Execute the deliberate provider lane with no-clobber task resolution."""
    if not allow_network:
        raise corpus.CorpusError("corpus requires literal --allow-network")
    # This is the only environment read in this module.  It is never accepted
    # by argparse, persisted, included in an exception, or printed.
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise corpus.CorpusError("missing provider credential")
    plan = _read_plan(plan_path)
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
        author_max_output_tokens = _author_max_output_tokens(slots)
        _preflight_request(
            "corpus_author",
            _provider_body(
                stage="corpus_author",
                model=corpus.AUTHOR_MODEL,
                messages=messages,
                response_schema=corpus.author_schema(slots),
                max_output_tokens=author_max_output_tokens,
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
                review, journal = _review_one(client, row, api_key)
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
                usable, reviews, prior_rows=accepted, reviewer_lineages=reviewer_lineages
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "plan":
            write_plan(args.output)
        elif args.command == "corpus":
            run_corpus(
                args.plan,
                args.snapshot,
                args.work_dir,
                args.packet,
                allow_network=args.allow_network,
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
