"""Deterministic, local-only V3 task reports and seven-task combination.

This module deliberately does not start a server, fetch data, download models,
or contact a hosted API.  ``run-task`` talks only to an explicitly supplied
loopback System One endpoint; all source inputs must already be frozen locally.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import camara_v3
import full_text_diagnostic as diagnostic
import ptbr_bench

TASK_COUNTS = {"camara_tema": 599, "juristcu_area": 893}
LEGACY_COUNTS = dict(diagnostic.LEGACY_COUNTS)
REPORT_SCHEMA = 1
COMBINED_SCHEMA = 3
CAMARA_SEED = 42
LEGAL_SEED = 20261009
SHA256 = "0123456789abcdef"


class ReportError(ValueError):
    """A local artifact does not meet the frozen reporting protocol."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _sha(value: bytes | Any) -> str:
    return hashlib.sha256(value if isinstance(value, bytes) else _canonical(value)).hexdigest()


def _is_sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in SHA256 for char in value)


def _read(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReportError(f"cannot read JSON {path}: {error}") from error


def _read_jsonl(path: Path, task: str) -> tuple[bytes, list[dict[str, Any]]]:
    try:
        raw = path.read_bytes()
        rows = [json.loads(line) for line in raw.splitlines()]
    except (OSError, json.JSONDecodeError) as error:
        raise ReportError(f"cannot read {task} rows: {error}") from error
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ReportError(f"{task} rows are not JSON objects")
    return raw, rows


def _exclusive(path: Path, value: Any) -> None:
    """Atomically create a result and never replace a previous artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _canonical(value) + b"\n"
    temporary: str | None = None
    try:
        fd, temporary = __import__("tempfile").mkstemp(dir=path.parent, prefix=f".{path.name}.")
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise ReportError(f"refusing to overwrite output {path}") from error
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def _checkpoint(args: argparse.Namespace) -> dict[str, str]:
    checkpoint = {
        "id": args.checkpoint_id,
        "revision": args.checkpoint_revision,
        "base": args.checkpoint_base,
    }
    if not all(isinstance(value, str) and value for value in checkpoint.values()):
        raise ReportError("checkpoint id, revision, and base are required")
    return checkpoint


def _load_camara(directory: Path) -> tuple[list[dict[str, Any]], dict[str, Any], str, str]:
    try:
        manifest = camara_v3.validate_derived_artifacts(directory)
    except (OSError, ValueError, camara_v3.SourceError) as error:
        raise ReportError(f"invalid Camara derived data: {error}") from error
    raw, rows = _read_jsonl(directory / "camara_tema_v3.test.jsonl", "camara_tema")
    if len(rows) != TASK_COUNTS["camara_tema"]:
        raise ReportError("Camara report requires exactly 599 rows")
    source_digest = _sha(raw)
    source_identity = manifest.get("source_manifest_sha256")
    if not _is_sha(source_identity):
        raise ReportError("Camara manifest has no source identity")
    return rows, manifest, source_digest, source_identity


def _cache_file(cache: Path, revision: str, task: str) -> Path:
    candidates = (cache / f"{task}.test.jsonl", cache / revision / f"{task}.test.jsonl")
    found = [candidate for candidate in candidates if candidate.is_file()]
    if len(found) != 1:
        raise ReportError(f"frozen legacy cache must contain one {task}.test.jsonl")
    return found[0]


def _load_legacy(
    cache: Path, revision: str
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str]]:
    if not ptbr_bench.is_immutable_revision(revision):
        raise ReportError("legacy dataset revision is not immutable")
    rows: dict[str, list[dict[str, Any]]] = {}
    hashes: dict[str, str] = {}
    for task, count in LEGACY_COUNTS.items():
        raw, _task_rows = _read_jsonl(_cache_file(cache, revision, task), task)
        try:
            parsed = ptbr_bench.parse_rows(raw, task)
        except ValueError as error:
            raise ReportError(str(error)) from error
        if len(parsed) != count:
            raise ReportError(f"{task} frozen cache count is not {count}")
        rows[task], hashes[task] = parsed, _sha(raw)
    return rows, hashes


def _camara_questions(rows: Sequence[Mapping[str, Any]]) -> list[tuple[dict[str, Any], int]]:
    if len(rows) != TASK_COUNTS["camara_tema"]:
        raise ReportError("Camara execution requires exactly 599 rows")
    criteria = rows[0].get("criteria") if rows else None
    if not isinstance(criteria, dict) or len(criteria) != 32:
        raise ReportError("Camara requires exactly 32 official choices")
    rng = random.Random(CAMARA_SEED)
    result = []
    for index, row in enumerate(rows):
        if row.get("criteria") != criteria or not isinstance(row.get("text"), str):
            raise ReportError(f"invalid Camara row {index}")
        keys = list(criteria)
        rng.shuffle(keys)
        result.append(
            (
                {
                    "type": "choice",
                    "instructions": row.get("instructions"),
                    "criteria": {key: criteria[key] for key in keys},
                },
                len(row["text"]),
            )
        )
    return result


def _legal_questions(
    rows: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[tuple[dict[str, Any], int]]:
    if set(rows) != set(LEGACY_COUNTS):
        raise ReportError("legal replay needs all seven original frozen tasks")
    rng = random.Random(LEGAL_SEED)
    legal: list[tuple[dict[str, Any], int]] = []
    for task in sorted(rows):
        if len(rows[task]) != LEGACY_COUNTS[task]:
            raise ReportError("legal replay has an incomplete original task")
        for row in rows[task]:
            question = ptbr_bench.question(dict(row), rng)
            if task == "juristcu_area":
                legal.append((question, len(str(row["text"]))))
    if len(legal) != TASK_COUNTS["juristcu_area"]:
        raise ReportError("legal replay did not produce 893 rows")
    return legal


def _typed_prediction(answer: Any, question: Mapping[str, Any]) -> str | bool | None:
    if question["type"] == "choice":
        return diagnostic._prediction(answer, question)  # shared strict native decoder
    if not isinstance(answer, dict):
        return None
    candidate = (
        answer.get("answers", {}).get("q") if isinstance(answer.get("answers"), dict) else None
    )
    return ptbr_bench.pick(candidate, question)[0] if isinstance(candidate, dict) else None


def _server_binding(
    base_url: str, evidence_path: Path | None, checkpoint: Mapping[str, str]
) -> dict[str, Any]:
    base = diagnostic.loopback_url(base_url)
    if evidence_path is None:
        return {"binding": "self-reported", "endpoint": base}
    try:
        evidence = diagnostic.load_server_evidence(evidence_path, dict(checkpoint))
        diagnostic.preflight(base, evidence)
    except (OSError, ValueError) as error:
        raise ReportError(f"strict server evidence/live binding failed: {error}") from error
    return {"binding": "strict-live", "endpoint": base, "evidence_sha256": _sha(evidence)}


def _score_task(
    task: str, rows: Sequence[Mapping[str, Any]], decisions: Sequence[Mapping[str, Any]]
) -> dict[str, Any] | None:
    if any(
        decision.get("error") is not None or decision.get("prediction") is None
        for decision in decisions
    ):
        return None
    if task == "camara_tema":
        predictions = {
            str(row["id"]): decision["prediction"]
            for row, decision in zip(rows, decisions, strict=True)
        }
        return camara_v3.score_predictions(rows, predictions)
    pairs = [
        (row["gold"], decision["prediction"]) for row, decision in zip(rows, decisions, strict=True)
    ]
    return {
        "balanced_accuracy": ptbr_bench.balanced(pairs),
        "chance": 1 / len({str(row["gold"]) for row in rows}),
    }


def _task_report(
    *,
    task: str,
    rows: Sequence[Mapping[str, Any]],
    decisions: Sequence[Mapping[str, Any]],
    checkpoint: Mapping[str, str],
    api_alias: str,
    source_digest: str,
    source_manifest_identity: str | None,
    server: Mapping[str, Any],
    imported: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    expected = TASK_COUNTS[task]
    failures = sum(
        decision.get("error") is not None or decision.get("prediction") is None
        for decision in decisions
    )
    complete = len(decisions) == expected and failures == 0
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA,
        "kind": "task-report-v3",
        "task": task,
        "checkpoint": dict(checkpoint),
        "api_alias": api_alias,
        "source_sha256": source_digest,
        "source_manifest_identity": source_manifest_identity,
        "full_input": True,
        "option_order_seed": CAMARA_SEED if task == "camara_tema" else LEGAL_SEED,
        "option_order_protocol": "camara-own-seed"
        if task == "camara_tema"
        else "original-seven-task-replay",
        "expected_rows": expected,
        "attempted_rows": len(decisions),
        "failures": failures,
        "complete": complete,
        "server": dict(server),
        "rows": list(decisions),
    }
    if imported is not None:
        report["strict_journal_import"] = dict(imported)
    score = _score_task(task, rows, decisions) if complete else None
    if score is not None:
        report["computed_score"] = score
    return report


def _decision(
    task: str,
    index: int,
    row: Mapping[str, Any],
    question: Mapping[str, Any],
    chars: int,
    prediction: str | bool | None,
    error: str | None,
    request_digest: str,
    response_digest: str | None,
) -> dict[str, Any]:
    gold_key = "accepted_set" if task == "camara_tema" else "gold"
    gold = row.get("accepted_labels") if task == "camara_tema" else row.get("gold")
    return {
        "index": index,
        "id": row.get("id", f"{task}-{index}"),
        gold_key: gold,
        "prediction": prediction,
        "error": error,
        "exact_chars_sent": chars,
        "question_order": list(question.get("criteria", {})),
        "request_sha256": request_digest,
        "response_sha256": response_digest,
    }


def run_task(args: argparse.Namespace) -> int:
    checkpoint = _checkpoint(args)
    server = _server_binding(args.base_url, args.server_evidence, checkpoint)
    if args.task == "camara_tema":
        rows, _manifest, digest, identity = _load_camara(args.data)
        questions = _camara_questions(rows)
    else:
        legacy, hashes = _load_legacy(args.data, args.legacy_revision)
        rows, digest, identity = legacy[args.task], hashes[args.task], None
        questions = _legal_questions(legacy)
    limit = args.limit if args.limit is not None else len(rows)
    if limit < 1 or limit > len(rows):
        raise ReportError("limit must be between one and the complete frozen task count")
    if args.journal.exists() and args.journal.stat().st_size:
        raise ReportError("attempt journal already exists; it is append-only evidence")
    decisions = []
    for index, (row, (question, chars)) in enumerate(
        zip(rows[:limit], questions[:limit], strict=True)
    ):
        answer, raw, _duration, error, request_digest = diagnostic.request(
            args.base_url, args.model, row["text"], question, args.timeout
        )
        prediction = _typed_prediction(answer, question)
        if error is None and prediction is None:
            error = "server_error_or_invalid_prediction"
        response_digest = diagnostic.raw_response_sha(raw) if isinstance(raw, str) else None
        record = _decision(
            args.task,
            index,
            row,
            question,
            chars,
            prediction,
            error,
            request_digest,
            response_digest,
        )
        diagnostic.append_jsonl(args.journal, record)
        decisions.append(record)
        if error is not None:
            raise ReportError(
                f"operational failure at {args.task}[{index}]: {error}; partial journal retained"
            )
    report = _task_report(
        task=args.task,
        rows=rows[:limit],
        decisions=decisions,
        checkpoint=checkpoint,
        api_alias=args.model,
        source_digest=digest,
        source_manifest_identity=identity,
        server=server,
    )
    _exclusive(args.output, report)
    return 0


def _validate_report(
    report: Any,
    *,
    task: str,
    rows: Sequence[Mapping[str, Any]],
    source_digest: str,
    source_identity: str | None,
    expected_questions: Sequence[tuple[dict[str, Any], int]] | None = None,
    complete: bool = True,
) -> dict[str, Any]:
    if (
        not isinstance(report, dict)
        or report.get("schema_version") != REPORT_SCHEMA
        or report.get("kind") != "task-report-v3"
    ):
        raise ReportError("unsupported task report schema")
    if (
        report.get("task") != task
        or report.get("source_sha256") != source_digest
        or report.get("source_manifest_identity") != source_identity
    ):
        raise ReportError("task report task/source identity mismatch")
    checkpoint = report.get("checkpoint")
    if (
        not isinstance(checkpoint, dict)
        or set(checkpoint) != {"id", "revision", "base"}
        or not all(isinstance(value, str) and value for value in checkpoint.values())
    ):
        raise ReportError("task report checkpoint is invalid")
    api_alias = report.get("api_alias")
    if not isinstance(api_alias, str) or not api_alias:
        raise ReportError("task report API model alias is invalid")
    expected = TASK_COUNTS[task]
    decisions = report.get("rows")
    if (
        not isinstance(decisions, list)
        or report.get("expected_rows") != expected
        or report.get("attempted_rows") != len(decisions)
    ):
        raise ReportError("task report row count metadata is invalid")
    if len(decisions) > expected:
        raise ReportError("task report has extra IDs")
    if complete and (
        report.get("complete") is not True
        or len(decisions) != expected
        or report.get("failures") != 0
    ):
        raise ReportError("combine accepts only complete zero-failure reports")
    expected_ids = [row.get("id", f"{task}-{i}") for i, row in enumerate(rows)]
    if expected_questions is None and task == "camara_tema":
        expected_questions = _camara_questions(rows)
    if expected_questions is None and task == "juristcu_area":
        raise ReportError("legal report requires the original seven-task replay order")
    if expected_questions is not None and len(expected_questions) != len(rows):
        raise ReportError("task report expected question replay is incomplete")
    predictions: dict[str, str] = {}
    failures = 0
    for index, (decision, row) in enumerate(zip(decisions, rows, strict=False)):
        if (
            not isinstance(decision, dict)
            or decision.get("index") != index
            or decision.get("id") != expected_ids[index]
        ):
            raise ReportError("task report IDs are missing, reordered, or extra")
        if decision.get("question_order") is None or not isinstance(
            decision["question_order"], list
        ):
            raise ReportError("task report lacks question order")
        if decision.get("exact_chars_sent") != len(row.get("text", "")):
            raise ReportError("task report exact character count differs from frozen text")
        if expected_questions is not None:
            expected_question, _chars = expected_questions[index]
            if decision["question_order"] != list(expected_question["criteria"]):
                raise ReportError("task report question order does not match frozen replay")
            expected_request = diagnostic.sha(
                diagnostic.wire_body(api_alias, str(row["text"]), expected_question)
            )
            if decision.get("request_sha256") != expected_request:
                raise ReportError("task report request digest does not match frozen API request")
        if not _is_sha(decision.get("response_sha256")):
            raise ReportError("task report response digest is missing or invalid")
        gold_key = "accepted_set" if task == "camara_tema" else "gold"
        expected_gold = row.get("accepted_labels") if task == "camara_tema" else row.get("gold")
        if decision.get(gold_key) != expected_gold:
            raise ReportError("task report gold does not match frozen data")
        prediction = decision.get("prediction")
        error = decision.get("error")
        if error is not None or prediction is None:
            failures += 1
        elif task == "camara_tema":
            if not isinstance(prediction, str) or prediction not in row["criteria"]:
                raise ReportError("Camara prediction type/key invalid")
            predictions[str(row["id"])] = prediction
        elif type(prediction) is not type(row["gold"]) or (
            isinstance(prediction, str) and prediction not in row.get("criteria", {})
        ):
            raise ReportError("legacy prediction type/key invalid")
    if failures != report.get("failures"):
        raise ReportError("task report failure count mismatch")
    if report.get("full_input") is not True:
        raise ReportError("task report is not full-input")
    if report.get("option_order_seed") != (CAMARA_SEED if task == "camara_tema" else LEGAL_SEED):
        raise ReportError("task report option-order seed differs from protocol")
    if task == "camara_tema" and not failures and len(predictions) == len(rows):
        recomputed = camara_v3.score_predictions(rows, predictions)
        claimed = report.get("computed_score")
        if claimed != recomputed:
            raise ReportError("Camara score is not format/score consistent")
    elif task == "juristcu_area" and not failures:
        pairs = [
            (row["gold"], decision["prediction"])
            for row, decision in zip(rows, decisions, strict=True)
        ]
        claimed = report.get("computed_score", {})
        if claimed.get("balanced_accuracy") != ptbr_bench.balanced(pairs):
            raise ReportError("legal score is not format/score consistent")
    return report


def from_strict_journal(args: argparse.Namespace) -> int:
    checkpoint = _checkpoint(args)
    rows, _manifest, digest, identity = _load_camara(args.camara_dir)
    prepared = diagnostic.load_prepared(args.prepared)
    if prepared.get("mode") != "v3camara":
        raise ReportError("strict-journal import accepts only a Camara V3 prepared journal")
    if prepared.get("rows_source_sha256") != digest:
        raise ReportError("prepared Camara journal does not bind the frozen data")
    try:
        evidence = diagnostic.load_server_evidence(args.server_evidence, checkpoint)
        summary = diagnostic.summarize(
            prepared,
            {"camara_tema": rows},
            args.journal,
            model=args.model,
            checkpoint=checkpoint,
            server_evidence_file=args.server_evidence,
        )
        _expected, records = diagnostic.validate_journal(
            args.journal,
            prepared,
            {"camara_tema": rows},
            args.model,
            checkpoint,
            evidence,
            args.server_evidence,
            complete=True,
        )
    except (OSError, ValueError) as error:
        raise ReportError(f"strict journal is invalid: {error}") from error
    if summary.get("expected_attempts") != 599 or len(records) != 599:
        raise ReportError("strict journal does not contain complete 599-row Camara evidence")
    questions = _camara_questions(rows)
    decisions = []
    for index, row in enumerate(rows):
        record = records.get((index, "full"))
        if record is None:
            raise ReportError("strict journal is incomplete")
        question, chars = questions[index]
        decisions.append(
            _decision(
                "camara_tema",
                index,
                row,
                question,
                chars,
                record["prediction"],
                None,
                record["request_sha256"],
                record["response_sha256"],
            )
        )
    report = _task_report(
        task="camara_tema",
        rows=rows,
        decisions=decisions,
        checkpoint=checkpoint,
        api_alias=args.model,
        source_digest=digest,
        source_manifest_identity=identity,
        server={"binding": "strict-journal-import", "evidence_sha256": _sha(evidence)},
        imported={
            "journal_sha256": _sha(args.journal.read_bytes()),
            "evidence_sha256": _sha(evidence),
            "summary": summary,
        },
    )
    _exclusive(args.output, report)
    return 0


def _validate_legacy_report(
    report: Any,
    legacy: Mapping[str, Sequence[Mapping[str, Any]]],
    hashes: Mapping[str, str],
    revision: str,
) -> dict[str, Any]:
    if not isinstance(report, dict) or report.get("revision") != revision:
        raise ReportError("legacy report revision does not match frozen protocol")
    checkpoint = report.get("checkpoint")
    if (
        not isinstance(checkpoint, dict)
        or not isinstance(checkpoint.get("id"), str)
        or not isinstance(checkpoint.get("revision"), str)
    ):
        raise ReportError("legacy report checkpoint identity/revision missing")
    items, tasks, files = report.get("items"), report.get("tasks"), report.get("dataset_files")
    if (
        not isinstance(items, dict)
        or not isinstance(tasks, dict)
        or not isinstance(files, dict)
        or set(items) != set(LEGACY_COUNTS)
        or set(tasks) != set(LEGACY_COUNTS)
    ):
        raise ReportError("legacy report must contain exactly seven tasks")
    total = failures = 0
    for task, frozen in legacy.items():
        pairs = items[task]
        if (
            not isinstance(pairs, list)
            or len(pairs) != len(frozen)
            or files.get(task, {}).get("sha256") != hashes[task]
        ):
            raise ReportError("legacy report data digest/count mismatch")
        normalized: list[tuple[Any, Any]] = []
        for index, (pair, row) in enumerate(zip(pairs, frozen, strict=True)):
            if (
                not isinstance(pair, list)
                or len(pair) != 2
                or type(pair[0]) is not type(row["gold"])
                or pair[0] != row["gold"]
                or type(pair[1]) is not type(row["gold"])
            ):
                raise ReportError(f"legacy typed pair invalid: {task}[{index}]")
            if row["type"] == "choice" and pair[1] not in row["criteria"]:
                raise ReportError(f"legacy choice prediction invalid: {task}[{index}]")
            normalized.append((pair[0], pair[1]))
        meta = tasks[task]
        if (
            not isinstance(meta, dict)
            or meta.get("n") != len(frozen)
            or meta.get("source_n") != len(frozen)
            or meta.get("failed") != 0
        ):
            raise ReportError("legacy report is incomplete or has failures")
        if (
            abs(ptbr_bench.balanced(normalized) - meta.get("balanced_accuracy", math.nan))
            > 0.000051
        ):
            raise ReportError("legacy metric does not match frozen typed predictions")
        total += len(normalized)
        failures += meta["failed"]
    if total != 5196 or failures:
        raise ReportError("legacy report must have 5196 decisions and zero failures")
    return report


def _protocol(path: Path) -> dict[str, Any]:
    protocol = _read(path)
    if (
        not isinstance(protocol, dict)
        or protocol.get("schema_version") != 1
        or protocol.get("scoring_version") != "ptbr-official-topics-v3"
        or protocol.get("legal_input") not in {"prefix3500", "full"}
        or not _is_sha(protocol.get("camara_data_sha256"))
        or not isinstance(protocol.get("legacy_dataset_revision"), str)
    ):
        raise ReportError("protocol schema/version is invalid")
    return protocol


def combine(args: argparse.Namespace) -> int:
    protocol = _protocol(args.protocol)
    legacy, hashes = _load_legacy(args.legacy_cache, protocol["legacy_dataset_revision"])
    original = _validate_legacy_report(
        _read(args.legacy_report), legacy, hashes, protocol["legacy_dataset_revision"]
    )
    camara_rows, _manifest, camara_digest, identity = _load_camara(args.camara_dir)
    if camara_digest != protocol["camara_data_sha256"]:
        raise ReportError("protocol Camara digest does not match frozen Camara data")
    camara = _validate_report(
        _read(args.camara_report),
        task="camara_tema",
        rows=camara_rows,
        source_digest=camara_digest,
        source_identity=identity,
    )
    old_checkpoint, new_checkpoint = original["checkpoint"], camara["checkpoint"]
    if (
        old_checkpoint["id"] != new_checkpoint["id"]
        or old_checkpoint["revision"] != new_checkpoint["revision"]
    ):
        raise ReportError("legacy and Camara checkpoint id/revision differ")
    selected = {
        task: {
            "rows": legacy[task],
            "pairs": original["items"][task],
            "provenance": "reused-original-v2",
        }
        for task in LEGACY_COUNTS
        if task != "camara_tema"
    }
    if protocol["legal_input"] == "full":
        if args.legal_report is None:
            raise ReportError("full protocol requires complete 893-row legal report")
        legal_digest = hashes["juristcu_area"]
        legal = _validate_report(
            _read(args.legal_report),
            task="juristcu_area",
            rows=legacy["juristcu_area"],
            source_digest=legal_digest,
            source_identity=None,
            expected_questions=_legal_questions(legacy),
        )
        if (
            legal["checkpoint"]["id"] != new_checkpoint["id"]
            or legal["checkpoint"]["revision"] != new_checkpoint["revision"]
        ):
            raise ReportError("full legal checkpoint id/revision differs")
        selected["juristcu_area"] = {
            "rows": legacy["juristcu_area"],
            "pairs": [
                [row["gold"], decision["prediction"]]
                for row, decision in zip(legacy["juristcu_area"], legal["rows"], strict=True)
            ],
            "provenance": "new-full-input-v3",
        }
    camara_predictions = {
        row["id"]: decision["prediction"]
        for row, decision in zip(camara_rows, camara["rows"], strict=True)
    }
    camara_score = camara_v3.score_predictions(camara_rows, camara_predictions)
    tasks: dict[str, Any] = {
        "camara_tema": {
            "decisions": camara["rows"],
            "provenance": "new-official-topics-v3",
            "source_sha256": camara_digest,
            "balanced_valid_topic": camara_score["macro_valid_topic"],
            "chance": camara_score["uniform_random_chance"],
            "adjusted_signed_points": camara_score["normalized_signed_unclipped"],
            "normalized_0_100_points": camara_score["normalized"],
        }
    }
    for task, value in selected.items():
        pairs = [(pair[0], pair[1]) for pair in value["pairs"]]
        classes = len({str(row["gold"]) for row in value["rows"]})
        ba, chance = ptbr_bench.balanced(pairs), 1 / classes
        signed = (ba - chance) / (1 - chance)
        tasks[task] = {
            "decisions": [[gold, prediction] for gold, prediction in pairs],
            "provenance": value["provenance"],
            "source_sha256": hashes[task],
            "balanced_accuracy": ba,
            "chance": chance,
            "adjusted_signed_points": 100 * signed,
            "normalized_0_100_points": 100 * max(0, signed),
        }
    if set(tasks) != set(LEGACY_COUNTS):
        raise ReportError("combined report lacks exactly seven tasks")
    output = {
        "schema_version": COMBINED_SCHEMA,
        "kind": "combined-report-v3",
        "scoring_version": protocol["scoring_version"],
        "checkpoint": new_checkpoint,
        "protocol": protocol,
        "source_digests": {"camara_tema": camara_digest, **hashes},
        "tasks": tasks,
        "normalized_0_100_points": sum(value["normalized_0_100_points"] for value in tasks.values())
        / 7,
        "adjusted_signed_points": sum(value["adjusted_signed_points"] for value in tasks.values())
        / 7,
        "inference_claim": "none; reused task evidence is explicitly provenance-labeled",
    }
    _exclusive(args.output, output)
    return 0


def validate(args: argparse.Namespace) -> int:
    """Validate format/score consistency by recomputing from frozen source inputs."""
    report = _read(args.report)
    if report.get("kind") == "task-report-v3":
        task = report.get("task")
        if task == "camara_tema":
            rows, _manifest, digest, identity = _load_camara(args.camara_dir)
        elif task == "juristcu_area":
            if args.legacy_cache is None or args.legacy_revision is None:
                raise ReportError("legal validation requires frozen legacy cache and revision")
            legacy, hashes = _load_legacy(args.legacy_cache, args.legacy_revision)
            rows, digest, identity = legacy[task], hashes[task], None
        else:
            raise ReportError("unknown task report")
        _validate_report(
            report,
            task=task,
            rows=rows,
            source_digest=digest,
            source_identity=identity,
            expected_questions=_legal_questions(legacy) if task == "juristcu_area" else None,
            complete=report.get("complete") is True,
        )
    elif report.get("kind") == "combined-report-v3":
        required = (
            args.legacy_report,
            args.legacy_cache,
            args.camara_dir,
            args.camara_report,
            args.protocol,
        )
        if any(value is None for value in required):
            raise ReportError("combined validation requires all source inputs")
        # ``combine`` is intentionally not used: validation must not create an output.
        protocol = _protocol(args.protocol)
        legacy, hashes = _load_legacy(args.legacy_cache, protocol["legacy_dataset_revision"])
        original = _validate_legacy_report(
            _read(args.legacy_report), legacy, hashes, protocol["legacy_dataset_revision"]
        )
        camara_rows, _m, digest, identity = _load_camara(args.camara_dir)
        camara = _validate_report(
            _read(args.camara_report),
            task="camara_tema",
            rows=camara_rows,
            source_digest=digest,
            source_identity=identity,
        )
        if (
            original["checkpoint"]["id"] != camara["checkpoint"]["id"]
            or original["checkpoint"]["revision"] != camara["checkpoint"]["revision"]
        ):
            raise ReportError("source checkpoint mismatch")
        # Recombine into a temporary exclusive path and compare semantic result, never report fields alone.
        temporary = Path(args.report).with_name(f".{Path(args.report).name}.validate.tmp")
        if temporary.exists():
            raise ReportError("validation temporary path already exists")
        try:
            temporary_args = argparse.Namespace(**vars(args), output=temporary)
            combine(temporary_args)
            rebuilt = _read(temporary)
        finally:
            temporary.unlink(missing_ok=True)
        if rebuilt != report:
            raise ReportError("combined report is not format/score consistent with source inputs")
    else:
        raise ReportError("unknown report kind")
    print("format/score consistency: PASS")
    return 0


def _path(value: str) -> Path:
    return Path(value).expanduser()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run-task")
    run.add_argument("--task", choices=sorted(TASK_COUNTS), required=True)
    run.add_argument(
        "--data",
        type=_path,
        required=True,
        help="Camara derived directory or seven-task legacy cache",
    )
    run.add_argument("--legacy-revision")
    run.add_argument("--model", required=True)
    run.add_argument("--checkpoint-id", required=True)
    run.add_argument("--checkpoint-revision", required=True)
    run.add_argument("--checkpoint-base", required=True)
    run.add_argument("--base-url", required=True)
    run.add_argument("--output", type=_path, required=True)
    run.add_argument("--journal", type=_path, required=True)
    run.add_argument("--server-evidence", type=_path)
    run.add_argument("--limit", type=int)
    run.add_argument("--timeout", type=float, default=120)
    run.set_defaults(func=run_task)
    strict = commands.add_parser("from-strict-journal")
    strict.add_argument("--camara-dir", type=_path, required=True)
    strict.add_argument("--prepared", type=_path, required=True)
    strict.add_argument("--journal", type=_path, required=True)
    strict.add_argument("--server-evidence", type=_path, required=True)
    strict.add_argument("--model", required=True)
    strict.add_argument("--checkpoint-id", required=True)
    strict.add_argument("--checkpoint-revision", required=True)
    strict.add_argument("--checkpoint-base", required=True)
    strict.add_argument("--output", type=_path, required=True)
    strict.set_defaults(func=from_strict_journal)
    combine_parser = commands.add_parser("combine")
    combine_parser.add_argument("--legacy-report", type=_path, required=True)
    combine_parser.add_argument("--legacy-cache", type=_path, required=True)
    combine_parser.add_argument("--camara-dir", type=_path, required=True)
    combine_parser.add_argument("--camara-report", type=_path, required=True)
    combine_parser.add_argument("--protocol", type=_path, required=True)
    combine_parser.add_argument("--legal-report", type=_path)
    combine_parser.add_argument("--output", type=_path, required=True)
    combine_parser.set_defaults(func=combine)
    check = commands.add_parser("validate")
    check.add_argument("--report", type=_path, required=True)
    check.add_argument("--camara-dir", type=_path)
    check.add_argument("--legacy-cache", type=_path)
    check.add_argument("--legacy-revision")
    check.add_argument("--legacy-report", type=_path)
    check.add_argument("--camara-report", type=_path)
    check.add_argument("--protocol", type=_path)
    check.add_argument("--legal-report", type=_path)
    check.set_defaults(func=validate)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "run-task" and args.task == "juristcu_area" and not args.legacy_revision:
            raise ReportError("juristcu_area requires --legacy-revision for seven-task replay")
        return args.func(args)
    except ReportError as error:
        print(f"benchmark_v3: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
