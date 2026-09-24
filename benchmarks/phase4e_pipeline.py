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
        content = choice["message"]["content"]
        finish_reason = choice.get("finish_reason", "unknown")
    except (KeyError, IndexError, TypeError) as error:
        raise corpus.CorpusError("provider response content (missing)") from error
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


def _load_resolved(work_dir: Path, plan: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    if not work_dir.exists():
        return {}
    if not work_dir.is_dir():
        raise corpus.CorpusError("work directory")
    known = {cast(str, slot["task_id"]) for slot in cast(list[Mapping[str, Any]], plan["slots"])}
    resolved: dict[str, dict[str, Any]] = {}
    for path in (
        sorted((work_dir / "resolved").glob("*.json")) if (work_dir / "resolved").exists() else []
    ):
        value = _read_json(path)
        if set(value) != {"schema_version", "task_id", "status", "row"}:
            raise corpus.CorpusError("work record")
        task_id = value["task_id"]
        if not isinstance(task_id, str) or task_id not in known or task_id in resolved:
            raise corpus.CorpusError("work record")
        if value["status"] not in {"accepted", "rejected"} or not isinstance(value["row"], dict):
            raise corpus.CorpusError("work record")
        resolved[task_id] = value
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
        return {
            "accepted": accepted,
            "unresolved": len(candidates) - total,
            "observed_acceptance": accepted / total if total else None,
            "wilson_lower_bound": corpus.wilson_lower_bound(accepted, total) if total else None,
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
        author_response = _response_content(
            client.author(slots=slots, max_output_tokens=author_max_output_tokens, api_key=api_key)
        )
        author_lineage = corpus.lineage_from_journal(client.last_journal("corpus_author"), "author")
        authored = author_response.get("records")
        usable, capacity_rejected = corpus.validate_author_rows_with_verified_minilm(
            authored, slots, snapshot
        )
        usable = [{**row, **author_lineage} for row in usable]
        capacity_rejected = [{**row, **author_lineage} for row in capacity_rejected]
        for item in capacity_rejected:
            _store_resolution(work_dir, item, "rejected")
            rejected.append(item)
            completed.add(cast(str, item["task_id"]))
        review_results = [_review_one(client, row, api_key) for row in usable]
        reviews = [review for review, _ in review_results]
        reviewer_lineages = {
            cast(str, row["task_id"]): corpus.lineage_from_journal(journal, "reviewer")
            for row, (_, journal) in zip(usable, review_results, strict=True)
        }
        newly_accepted, newly_rejected = corpus.resolve_reviews(
            usable, reviews, prior_rows=accepted, reviewer_lineages=reviewer_lineages
        )
        for item in newly_accepted:
            _store_resolution(work_dir, item, "accepted")
            accepted.append(item)
            completed.add(cast(str, item["task_id"]))
        for item in newly_rejected:
            _store_resolution(work_dir, item, "rejected")
            rejected.append(item)
            completed.add(cast(str, item["task_id"]))
        if index % 10 == 0 and len(completed) >= 200:
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
