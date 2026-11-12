"""Private, opt-in refusal-aware diagnostic for Gemini Decision 1.

This is deliberately not an official evaluator or submission producer.  It
replays the frozen V3 inputs locally, stores only hashes and positional answer
evidence, and calls the existing direct Gemini transport only when ``run`` is
explicitly requested and ``GEMINI_API_KEY`` is present.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import tempfile
import time
from collections import Counter, defaultdict, deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import benchmark_v3
import gemini_bridge
import ptbr_bench

MODEL = gemini_bridge.MODEL
REFUSAL_CODES = frozenset(
    {"safety", "prohibited_content", "blocklist", "image_safety", "content_blocked"}
)
MAX_ERROR_BODY = 64 * 1024
RESERVE_OVERHEAD = 2048
TOKEN_LIMIT = 49_000
WINDOW_SECONDS = 61.0
MIN_INTERVAL = 0.4


class DiagnosticError(ValueError):
    """A deterministic plan, artifact, or operational diagnostic error."""


@dataclass(frozen=True)
class PlanItem:
    item_id: str
    task: str
    index: int
    text: str
    question: dict[str, Any]
    gold: str | bool | list[str]
    source_wire_sha256: str
    provider_payload: bytes
    provider_payload_sha256: str
    source_sha256: str

    @property
    def reservation(self) -> int:
        return len(self.provider_payload) + RESERVE_OVERHEAD


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _sha(value: bytes | Any) -> str:
    return hashlib.sha256(value if isinstance(value, bytes) else _canonical(value)).hexdigest()


def _utc() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _exclusive(path: Path, value: Any) -> None:
    """Atomically create a 0600 JSON artifact, never replacing an attempt."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _canonical(value) + b"\n"
    temporary: str | None = None
    try:
        fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise DiagnosticError("refusing_to_overwrite") from error
    finally:
        if temporary:
            Path(temporary).unlink(missing_ok=True)


class AppendJournal:
    def __init__(self, path: Path, *, resume: bool = False) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        flags = (
            os.O_WRONLY | os.O_APPEND | os.O_CREAT
            if resume
            else os.O_WRONLY | os.O_CREAT | os.O_EXCL
        )
        try:
            self.fd = os.open(path, flags, 0o600)
            os.chmod(path, 0o600)
        except OSError as error:
            raise DiagnosticError("journal_unavailable") from error

    def append(self, value: Mapping[str, Any]) -> None:
        line = _canonical(dict(value)) + b"\n"
        try:
            view = memoryview(line)
            while view:
                written = os.write(self.fd, view)
                if not isinstance(written, int) or written <= 0:
                    raise OSError("short_write")
                view = view[written:]
            os.fsync(self.fd)
        except OSError as error:
            raise DiagnosticError("journal_failure") from error

    def close(self) -> None:
        os.close(self.fd)


def _provider_payload(
    text: str, question: Mapping[str, Any]
) -> tuple[bytes, gemini_bridge.ParsedRequest]:
    raw = json.dumps({"model": MODEL, "state": text, "questions": {"q": question}}).encode("utf-8")
    parsed = gemini_bridge.parse_request(raw)
    translated, _reverse = gemini_bridge.translate(parsed)
    return (
        json.dumps(translated, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        ),
        parsed,
    )


def _legacy_wire(text: str, question: Mapping[str, Any]) -> bytes:
    # This intentionally retains json.dumps defaults, exactly like ptbr_bench.call.
    return json.dumps(
        {"model": MODEL, "state": text[: ptbr_bench.MAX_CHARS], "questions": {"q": question}}
    ).encode()


def _load_inputs(
    legacy_cache: Path, legacy_revision: str, camara_dir: Path
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str], list[dict[str, Any]], str, str]:
    legacy, legacy_hashes = benchmark_v3._load_legacy(legacy_cache, legacy_revision)
    camara_rows, _manifest, camara_digest, source_identity = benchmark_v3._load_camara(camara_dir)
    return legacy, legacy_hashes, camara_rows, camara_digest, source_identity


def build_plan(
    legacy_cache: Path, legacy_revision: str, camara_dir: Path
) -> tuple[list[PlanItem], dict[str, Any]]:
    """Build the exact all-seven-task V3 plan without writing data or calling a network."""
    legacy, legacy_hashes, camara_rows, camara_digest, camara_identity = _load_inputs(
        legacy_cache, legacy_revision, camara_dir
    )
    if set(legacy) != set(ptbr_bench.TASKS):
        raise DiagnosticError("legacy_tasks_mismatch")
    items: list[PlanItem] = []
    rng = random.Random(ptbr_bench.SEED)
    for task in ptbr_bench.TASKS:
        rows = legacy[task]
        for index, row in enumerate(rows):
            question = ptbr_bench.question(dict(row), rng)
            text = row["text"] if task == "juristcu_area" else row["text"][: ptbr_bench.MAX_CHARS]
            payload, _parsed = _provider_payload(text, question)
            items.append(
                PlanItem(
                    f"{task}:{index}",
                    task,
                    index,
                    text,
                    question,
                    row["gold"],
                    _sha(_legacy_wire(row["text"], question)),
                    payload,
                    _sha(payload),
                    legacy_hashes[task],
                )
            )
    rng = random.Random(benchmark_v3.CAMARA_SEED)
    criteria = camara_rows[0].get("criteria") if camara_rows else None
    if not isinstance(criteria, dict) or len(criteria) != 32:
        raise DiagnosticError("camara_criteria_invalid")
    # Replace the historical Camara plan with V3 full-text / 32-choice inputs.
    items = [item for item in items if item.task != "camara_tema"]
    camara_items: list[PlanItem] = []
    for index, row in enumerate(camara_rows):
        keys = list(criteria)
        rng.shuffle(keys)
        question = {
            "type": "choice",
            "instructions": row["instructions"],
            "criteria": {key: criteria[key] for key in keys},
        }
        payload, _parsed = _provider_payload(row["text"], question)
        camara_items.append(
            PlanItem(
                f"camara_tema:{index}",
                "camara_tema",
                index,
                row["text"],
                question,
                list(row["accepted_labels"]),
                _sha(_legacy_wire(row["text"], question)),
                payload,
                _sha(payload),
                camara_digest,
            )
        )
    # The original task order is preserved; Camara is its first task.
    by_task: dict[str, list[PlanItem]] = defaultdict(list)
    for item in items + camara_items:
        by_task[item.task].append(item)
    ordered = [
        item for task in ptbr_bench.TASKS for item in sorted(by_task[task], key=lambda x: x.index)
    ]
    expected = sum(benchmark_v3.LEGACY_COUNTS.values())
    if len(ordered) != expected or len(camara_items) != benchmark_v3.TASK_COUNTS["camara_tema"]:
        raise DiagnosticError("plan_count_mismatch")
    oversized = [item.item_id for item in ordered if item.reservation > TOKEN_LIMIT]
    if oversized:
        raise DiagnosticError("unrunnable_quota")
    metadata = {
        "schema": "gemini-refusal-diagnostic-plan-v1",
        "model": MODEL,
        "legacy_revision": legacy_revision,
        "legacy_source_sha256": legacy_hashes,
        "camara_source_sha256": camara_digest,
        "camara_source_manifest_identity": camara_identity,
        "count": len(ordered),
        "plan_sha256": _sha([_plan_row(item) for item in ordered]),
        "max_reservation": max(item.reservation for item in ordered),
    }
    return ordered, metadata


def _plan_row(item: PlanItem) -> dict[str, Any]:
    return {
        "id": item.item_id,
        "task": item.task,
        "index": item.index,
        "source_wire_sha256": item.source_wire_sha256,
        "provider_payload_sha256": item.provider_payload_sha256,
        "source_sha256": item.source_sha256,
        "input_chars": len(item.text),
        "question_order": list(item.question.get("criteria", {})),
        "reservation": item.reservation,
    }


def refusal_code(status: int, body: bytes) -> str | None:
    """Return a closed content-refusal code; every other response is operational."""
    if status != 400 or len(body) > MAX_ERROR_BODY:
        return None
    try:
        decoded = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    error = decoded.get("error") if isinstance(decoded, dict) else None
    code = error.get("code") if isinstance(error, dict) else None
    return code if isinstance(code, str) and code in REFUSAL_CODES else None


class Pacer:
    def __init__(
        self,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.clock, self.sleep = clock, sleep
        self.last: float | None = None
        self.window: deque[tuple[float, int]] = deque()

    def admit(self, reserve: int) -> None:
        if reserve > TOKEN_LIMIT:
            raise DiagnosticError("unrunnable_quota")
        now = self.clock()
        if self.last is not None and now - self.last < MIN_INTERVAL:
            self.sleep(MIN_INTERVAL - (now - self.last))
            now = self.clock()
        while True:
            while self.window and now - self.window[0][0] >= WINDOW_SECONDS:
                self.window.popleft()
            if sum(tokens for _at, tokens in self.window) + reserve <= TOKEN_LIMIT:
                break
            if not self.window:
                raise DiagnosticError("quota_unavailable")
            self.sleep(max(0.0, WINDOW_SECONDS - (now - self.window[0][0])))
            now = self.clock()
        self.window.append((now, reserve))
        self.last = now

    def replace_last(self, usage: dict[str, int] | str) -> None:
        if (
            isinstance(usage, dict)
            and isinstance(usage.get("total_input_tokens"), int)
            and self.window
        ):
            at, _reserved = self.window.pop()
            self.window.append((at, usage["total_input_tokens"]))


def _safe_record(
    item: PlanItem,
    *,
    outcome: str,
    prediction: str | bool | None,
    usage: dict[str, int] | str,
    latency_ms: int,
    identity: str | None,
    reason: str | None = None,
    answers: list[dict[str, Any]] | None = None,
    provenance: str = "new",
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "timestamp_utc": _utc(),
        "id": item.item_id,
        "task": item.task,
        "index": item.index,
        "outcome": outcome,
        "prediction": prediction,
        "refusal_code": reason,
        "source_wire_sha256": item.source_wire_sha256,
        "provider_payload_sha256": item.provider_payload_sha256,
        "source_sha256": item.source_sha256,
        "input_chars": len(item.text),
        "question_order": list(item.question.get("criteria", {})),
        "usage": usage,
        "upstream_latency_ms": latency_ms,
        "model_identity": identity or "unknown",
        "provenance": provenance,
    }
    if answers is not None:
        record["answers"] = answers
    return record


def _prior_model_identity(prior: Sequence[Mapping[str, Any]]) -> str | None:
    """Return the sole known prior identity, failing closed on resume drift."""
    identities = {
        identity
        for row in prior
        if row.get("outcome") in {"answered", "refusal"}
        and isinstance((identity := row.get("model_identity")), str)
        and identity not in {"", "unknown"}
    }
    if len(identities) > 1:
        raise DiagnosticError("prior_model_drift")
    return next(iter(identities), None)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        raw = path.read_bytes()
        if raw and not raw.endswith(b"\n"):
            raise DiagnosticError("journal_unterminated")
        result = [json.loads(line) for line in raw.splitlines()]
    except (OSError, json.JSONDecodeError) as error:
        raise DiagnosticError("journal_invalid") from error
    if not all(isinstance(row, dict) for row in result):
        raise DiagnosticError("journal_invalid")
    return result


def validate_resume(
    plan: Sequence[PlanItem], manifest: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> set[str]:
    if manifest.get("plan_sha256") != _sha([_plan_row(item) for item in plan]):
        raise DiagnosticError("manifest_plan_mismatch")
    completed: set[str] = set()
    for expected, row in zip(plan, rows, strict=False):
        if row.get("id") != expected.item_id or row.get("outcome") not in {"answered", "refusal"}:
            raise DiagnosticError("resume_not_ordered_prefix")
        if (
            row.get("id") in completed
            or row.get("provider_payload_sha256") != expected.provider_payload_sha256
            or row.get("source_wire_sha256") != expected.source_wire_sha256
        ):
            raise DiagnosticError("resume_hash_mismatch")
        prediction = row.get("prediction")
        if (
            row.get("outcome") == "answered"
            and prediction not in expected.question.get("criteria", {})
            and not (expected.question.get("type") == "noul" and isinstance(prediction, bool))
        ):
            raise DiagnosticError("resume_prediction_invalid")
        if row.get("outcome") == "refusal" and (
            prediction is not None or row.get("refusal_code") not in REFUSAL_CODES
        ):
            raise DiagnosticError("resume_refusal_invalid")
        completed.add(expected.item_id)
    if len(rows) > len(plan):
        raise DiagnosticError("resume_extra_rows")
    return completed


def _seed_records(plan: Sequence[PlanItem], seed: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Validate immutable import evidence and return only compatible answered rows."""
    if seed.get("schema") != "gemini-refusal-diagnostic-seed-v1":
        raise DiagnosticError("seed_schema_invalid")
    if seed.get("plan_sha256") != _sha([_plan_row(item) for item in plan]):
        raise DiagnosticError("seed_plan_mismatch")
    records = seed.get("records")
    if not isinstance(records, list) or seed.get("records_sha256") != _sha(records):
        raise DiagnosticError("seed_records_mismatch")
    by_id = {item.item_id: item for item in plan}
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source in records:
        if not isinstance(source, dict):
            raise DiagnosticError("seed_record_invalid")
        item = by_id.get(source.get("id"))
        prediction = source.get("prediction")
        if (
            item is None
            or item.item_id in seen
            or source.get("provider_payload_sha256") != item.provider_payload_sha256
            or source.get("source_wire_sha256") != item.source_wire_sha256
            or (
                prediction not in item.question.get("criteria", {})
                and not (item.question.get("type") == "noul" and isinstance(prediction, bool))
            )
        ):
            raise DiagnosticError("seed_record_invalid")
        seen.add(item.item_id)
        result.append(
            _safe_record(
                item,
                outcome="answered",
                prediction=prediction,
                usage="unknown",
                latency_ms=0,
                identity=None,
                provenance="legacy-import",
            )
            | {"import_provenance": source["provenance"]}
        )
    return result


def import_legacy_seed(
    plan: Sequence[PlanItem],
    report: Mapping[str, Any],
    bridge_rows: Sequence[Mapping[str, Any]],
    legacy_revision: str,
) -> dict[str, Any]:
    """Create a compatible-only seed from a frozen legacy full attempt.

    This deliberately accepts failed legacy pairs, but never imports them, a
    refusal, or historical Camara24 evidence.
    """
    timing = report.get("timing")
    items = report.get("items")
    files = report.get("dataset_files")
    run = report.get("run")
    if (
        report.get("revision") != legacy_revision
        or not ptbr_bench.is_immutable_revision(legacy_revision)
        or not isinstance(timing, dict)
        or not isinstance(items, dict)
        or not isinstance(files, dict)
    ):
        raise DiagnosticError("legacy_report_invalid")
    if not isinstance(run, dict) or run.get("mode") != "full" or run.get("partial") is not False:
        raise DiagnosticError("legacy_report_not_full")
    start, finish = timing.get("started_at"), timing.get("finished_at")
    if not isinstance(start, str) or not isinstance(finish, str) or start > finish:
        raise DiagnosticError("legacy_report_timing_invalid")
    eligible = [item for item in plan if item.task != "camara_tema"]
    queues: dict[str, deque[Mapping[str, Any]]] = defaultdict(deque)
    for row in bridge_rows:
        timestamp = row.get("timestamp_utc")
        if (
            isinstance(timestamp, str)
            and start <= timestamp <= finish
            and row.get("status") == "ok"
            and isinstance(row.get("original_request_sha256"), str)
            and isinstance(row.get("translated_request_sha256"), str)
            and isinstance(row.get("answer"), list)
        ):
            queues[row["original_request_sha256"]].append(row)
    records: list[dict[str, Any]] = []
    for item in eligible:
        pair_rows = items.get(item.task)
        digest = (
            files.get(item.task, {}).get("sha256")
            if isinstance(files.get(item.task), dict)
            else None
        )
        if (
            not isinstance(pair_rows, list)
            or item.index >= len(pair_rows)
            or digest != item.source_sha256
        ):
            raise DiagnosticError("legacy_report_source_mismatch")
        pair = pair_rows[item.index]
        if not isinstance(pair, list) or len(pair) != 2 or pair[0] != item.gold:
            raise DiagnosticError("legacy_report_gold_mismatch")
        prediction = pair[1]
        if prediction is None:
            continue
        candidates = queues[item.source_wire_sha256]
        while (
            candidates
            and candidates[0].get("translated_request_sha256") != item.provider_payload_sha256
        ):
            candidates.popleft()
        if not candidates:
            continue
        evidence = candidates.popleft()
        answer = evidence["answer"][0]
        if not isinstance(answer, dict):
            raise DiagnosticError("legacy_journal_answer_invalid")
        if item.question["type"] == "choice":
            position = answer.get("winner_position")
            keys = list(item.question["criteria"])
            decoded: str | bool | None = (
                keys[position] if isinstance(position, int) and 0 <= position < len(keys) else None
            )
        else:
            value = answer.get("noul")
            decoded = (
                value >= 0.5
                if isinstance(value, (int, float)) and not isinstance(value, bool)
                else None
            )
        if decoded != prediction:
            raise DiagnosticError("legacy_journal_prediction_mismatch")
        records.append(
            {
                "id": item.item_id,
                "prediction": prediction,
                "source_wire_sha256": item.source_wire_sha256,
                "provider_payload_sha256": item.provider_payload_sha256,
                "provenance": {
                    "source_task": item.task,
                    "source_index": item.index,
                    "timestamp_utc": evidence["timestamp_utc"],
                    "upstream_response_sha256": evidence.get("upstream_response_sha256", "unknown"),
                },
            }
        )
    return {
        "schema": "gemini-refusal-diagnostic-seed-v1",
        "plan_sha256": _sha([_plan_row(item) for item in plan]),
        "source_report_sha256": _sha(report),
        "source_journal_sha256": _sha(list(bridge_rows)),
        "records": records,
        "records_sha256": _sha(records),
        "camara24": "incompatible_excluded",
    }


def run_plan(
    plan: Sequence[PlanItem],
    journal: AppendJournal,
    transport: Callable[[bytes, float], gemini_bridge.TransportResponse],
    *,
    prior: Sequence[Mapping[str, Any]] = (),
    reused: Sequence[Mapping[str, Any]] = (),
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    timeout: float = gemini_bridge.UPSTREAM_TIMEOUT,
) -> list[dict[str, Any]]:
    """Sequentially run unresolved items. Operational failures are journaled then raised."""
    completed = {str(row["id"]) for row in prior if row.get("outcome") in {"answered", "refusal"}}
    records = [dict(row) for row in prior]
    reuse_by_id = {str(row["id"]): dict(row) for row in reused}
    # A resumed hosted-alias attempt must retain its previously declared
    # identity pin.  Unknown remains unknown; it never becomes a revision.
    pacer, pinned = Pacer(clock, sleep), _prior_model_identity(prior)
    for item in plan:
        if item.item_id in completed:
            continue
        if item.item_id in reuse_by_id:
            record = reuse_by_id[item.item_id]
            journal.append(record)
            records.append(record)
            completed.add(item.item_id)
            continue
        pacer.admit(item.reservation)
        started = clock()
        try:
            response = transport(item.provider_payload, timeout)
        except (OSError, TimeoutError, Exception) as error:
            # No text from a provider or exception is persisted.
            journal.append(
                _safe_record(
                    item,
                    outcome="operational",
                    prediction=None,
                    usage="unknown",
                    latency_ms=int((clock() - started) * 1000),
                    identity=None,
                    reason="transport",
                )
            )
            raise DiagnosticError("transport") from error
        latency = int((clock() - started) * 1000)
        refused = refusal_code(response.status, response.body)
        if refused:
            record = _safe_record(
                item,
                outcome="refusal",
                prediction=None,
                usage="unknown",
                latency_ms=latency,
                identity=None,
                reason=refused,
            )
            journal.append(record)
            records.append(record)
            continue
        if response.status != 200:
            journal.append(
                _safe_record(
                    item,
                    outcome="operational",
                    prediction=None,
                    usage="unknown",
                    latency_ms=latency,
                    identity=None,
                    reason="http_status",
                )
            )
            raise DiagnosticError("operational_http")
        try:
            parsed = gemini_bridge.parse_request(
                json.dumps(
                    {"model": MODEL, "state": item.text, "questions": {"q": item.question}}
                ).encode()
            )
            _translated, reverse = gemini_bridge.translate(parsed)
            result, identity, usage = gemini_bridge.remap_response(response.body, parsed, reverse)
            answer = result["answers"]["q"]
            prediction = (
                answer.get("choice") if item.question["type"] == "choice" else answer["noul"] >= 0.5
            )
            evidence = gemini_bridge.sanitized_answers(parsed, result["answers"])
        except (
            gemini_bridge.RequestError,
            gemini_bridge.UpstreamError,
            KeyError,
            TypeError,
            ValueError,
        ) as error:
            journal.append(
                _safe_record(
                    item,
                    outcome="operational",
                    prediction=None,
                    usage="unknown",
                    latency_ms=latency,
                    identity=None,
                    reason="invalid_response",
                )
            )
            raise DiagnosticError("invalid_response") from error
        if pinned is not None and identity is not None and identity != pinned:
            journal.append(
                _safe_record(
                    item,
                    outcome="operational",
                    prediction=None,
                    usage=usage,
                    latency_ms=latency,
                    identity=identity,
                    reason="model_drift",
                )
            )
            raise DiagnosticError("model_drift")
        if identity is not None:
            pinned = identity
        pacer.replace_last(usage)
        record = _safe_record(
            item,
            outcome="answered",
            prediction=prediction,
            usage=usage,
            latency_ms=latency,
            identity=identity,
            answers=evidence,
        )
        journal.append(record)
        records.append(record)
    return records


def _camara_score(
    rows: Sequence[PlanItem], predictions: Mapping[str, str | bool | None]
) -> dict[str, float]:
    """Official D2 weighting with null predictions treated as no hit."""
    criteria = rows[0].question["criteria"]
    official = set(criteria)
    supports = Counter(label for row in rows for label in row.gold)  # type: ignore[arg-type]
    positive = sorted(label for label in official if supports[label])
    hits = {label: 0.0 for label in official}
    chance = 0.0
    for row in rows:
        accepted = set(row.gold)  # type: ignore[arg-type]
        hit = predictions.get(row.item_id) in accepted
        weight = sum(1 / supports[label] for label in accepted) / len(positive)
        chance += weight * len(accepted) / len(official)
        for label in accepted:
            hits[label] += float(hit) / supports[label]
    macro = sum(hits[label] for label in positive) / len(positive)
    return {
        "balanced_accuracy": macro,
        "chance": chance,
        # These preserve camara_v3's public point scales.  The generic fields
        # above deliberately remain fractions for all seven tasks.
        "macro_valid_topic": 100 * macro,
        "uniform_random_chance": 100 * chance,
        "normalized": 100 * min(1, max(0, (macro - chance) / (1 - chance))),
    }


def report(plan: Sequence[PlanItem], records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_id = {
        str(row.get("id")): row for row in records if row.get("outcome") in {"answered", "refusal"}
    }
    tasks: dict[str, dict[str, Any]] = {}
    normalized: list[float] = []
    complete = len(by_id) == len(plan)
    for task in (name for name in ptbr_bench.TASKS if any(item.task == name for item in plan)):
        items = [item for item in plan if item.task == task]
        rows = [by_id.get(item.item_id) for item in items]
        answered = sum(row is not None and row.get("outcome") == "answered" for row in rows)
        refusals = Counter(
            str(row.get("refusal_code")) for row in rows if row and row.get("outcome") == "refusal"
        )
        predictions = {
            item.item_id: row.get("prediction") if row else None
            for item, row in zip(items, rows, strict=True)
        }
        resolved = all(row is not None for row in rows)
        correct = sum(
            predictions[item.item_id] in item.gold
            if task == "camara_tema"
            else item.gold == predictions[item.item_id]
            for item in items
        )
        score: dict[str, float] | None
        if not resolved:
            score = None
        elif task == "camara_tema":
            score = _camara_score(items, predictions)
        else:
            pairs = [(item.gold, predictions[item.item_id]) for item in items]
            ba = ptbr_bench.balanced(pairs)
            labels = {item.gold for item in items}
            chance = 1 / len(labels)
            score = {
                "balanced_accuracy": ba,
                "chance": chance,
                "normalized": 100 * max(0, (ba - chance) / (1 - chance)),
            }
        if score is not None:
            normalized.append(score["normalized"])
        tasks[task] = {
            "planned": len(items),
            "answered": answered,
            "refused": sum(refusals.values()),
            "refusal_reasons": dict(refusals),
            "correct": correct,
            "balanced_accuracy": score["balanced_accuracy"] if score else None,
            "chance": score["chance"] if score else None,
            "macro_valid_topic": score.get("macro_valid_topic") if score else None,
            "uniform_random_chance": score.get("uniform_random_chance") if score else None,
            "normalized": score["normalized"] if score else None,
            "answered_only_accuracy": correct / answered if answered else None,
        }
    # `records` is the validated journal sequence.  Resolved coverage is a
    # unique answered/refusal prefix; attempted/new deliberately count every
    # non-imported outbound event, including the terminal operational record.
    # Imported rows are reusable evidence and never represent a new API call.
    operational = sum(row.get("outcome") == "operational" for row in records)
    reused = sum(row.get("provenance") == "legacy-import" for row in records)
    attempted = sum(row.get("provenance") != "legacy-import" for row in records)
    return {
        "schema": "gemini-refusal-diagnostic-report-v1",
        "official_submission_eligible": False,
        "model_revision": "unknown",
        "complete": complete and not operational,
        "tasks": tasks,
        "planned": len(plan),
        "processed": len(records),
        "resolved": len(by_id),
        "attempted": attempted,
        "reused": reused,
        "new": attempted,
        "answered": sum(row.get("outcome") == "answered" for row in records),
        "refused": sum(row.get("outcome") == "refusal" for row in records),
        "operational": operational,
        "headline_refusal_inclusive_normalized_mean": sum(normalized) / 7
        if complete and not operational and len(normalized) == 7
        else None,
        "note": "Internal unranked diagnostic; answered-only accuracy is conditional and not a benchmark score.",
    }


def _paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    root = args.output
    return root / "manifest.json", root / "journal.jsonl", root / "report.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "run", "resume", "report", "import-legacy"):
        command = sub.add_parser(name)
        command.add_argument("--legacy-cache", type=Path, required=True)
        command.add_argument("--legacy-revision", required=True)
        command.add_argument("--camara-dir", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        if name == "import-legacy":
            command.add_argument("--legacy-report", type=Path, required=True)
            command.add_argument("--bridge-journal", type=Path, required=True)
        if name in {"run", "resume"}:
            command.add_argument("--seed", type=Path)
            command.add_argument("--seed-sha256")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        plan, manifest = build_plan(args.legacy_cache, args.legacy_revision, args.camara_dir)
        manifest_path, journal_path, report_path = _paths(args)
        if args.command == "plan":
            _exclusive(manifest_path, manifest)
            print(
                json.dumps({"planned": len(plan), "max_reservation": manifest["max_reservation"]})
            )
            return 0
        if args.command == "report":
            _exclusive(report_path, report(plan, _read_jsonl(journal_path)))
            return 0
        if args.command == "import-legacy":
            source_report = json.loads(args.legacy_report.read_text(encoding="utf-8"))
            source_journal = _read_jsonl(args.bridge_journal)
            if not isinstance(source_report, dict):
                raise DiagnosticError("legacy_report_invalid")
            seed = import_legacy_seed(plan, source_report, source_journal, args.legacy_revision)
            _exclusive(args.output / "seed.json", seed)
            return 0
        seed_rows: list[dict[str, Any]] = []
        if args.seed is not None:
            raw_seed = args.seed.read_bytes()
            if args.seed_sha256 != _sha(raw_seed):
                raise DiagnosticError("seed_digest_mismatch")
            loaded_seed = json.loads(raw_seed)
            if not isinstance(loaded_seed, dict):
                raise DiagnosticError("seed_schema_invalid")
            seed_rows = _seed_records(plan, loaded_seed)
        if args.command == "run":
            _exclusive(manifest_path, manifest)
            prior = []
            journal = AppendJournal(journal_path)
        else:
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
            prior = _read_jsonl(journal_path)
            validate_resume(plan, loaded, prior)
            journal = AppendJournal(journal_path, resume=True)
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise DiagnosticError("GEMINI_API_KEY_required")
        try:
            records = run_plan(
                plan, journal, gemini_bridge.https_transport(key), prior=prior, reused=seed_rows
            )
        finally:
            journal.close()
        _exclusive(report_path, report(plan, records))
        return 0
    except (DiagnosticError, OSError, json.JSONDecodeError) as error:
        print(f"gemini_diagnostic: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
