"""Fail-closed local HTTP diagnostic for frozen V2 legal and V3 Câmara tasks."""

import argparse
import hashlib
import json
import math
import os
import random
import tempfile
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

import ptbr_bench

SEED, V3_CAMARA_SEED, PREFIX_CHARS, AFFECTED_ROWS = 20261009, 42, 3500, 429
LEGACY_COUNTS = {
    "camara_tema": 599,
    "factck_veracidade": 259,
    "faquad_resposta": 650,
    "juristcu_area": 893,
    "olid_alvo": 995,
    "olid_ofensivo": 1000,
    "scielo_area": 800,
}
PREPARED_SCHEMA = 3


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def wire_body(model, state, q):
    """Encode the inference request without reordering frozen choice criteria."""
    return json.dumps(
        {"model": model, "state": state, "questions": {"q": q}},
        sort_keys=False,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def sha(value):
    return hashlib.sha256(value if isinstance(value, bytes) else canonical(value)).hexdigest()


def raw_response_sha(value):
    if not isinstance(value, str):
        raise ValueError("raw response must be UTF-8 text")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def source_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def publish_exclusive(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix="." + path.name)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(canonical(value))
            f.flush()
            os.fsync(f.fileno())
        try:
            os.link(tmp, path)
        except FileExistsError:
            raise ValueError("refusing to overwrite immutable output") from None
    finally:
        Path(tmp).unlink(missing_ok=True)


def append_jsonl(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as f:
        f.write(canonical(value) + b"\n")
        f.flush()
        os.fsync(f.fileno())


def loopback_url(value):
    part = urllib.parse.urlsplit(value)
    if (
        part.scheme != "http"
        or part.hostname not in ("127.0.0.1", "::1", "localhost")
        or part.username
        or part.password
    ):
        raise ValueError("endpoint must be loopback http URL without credentials")
    return value.rstrip("/")


def _read_rows(path):
    raw = Path(path).read_bytes()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        try:
            rows = [json.loads(line) for line in raw.splitlines() if line]
        except json.JSONDecodeError as e:
            raise ValueError("invalid JSONL rows") from e
        if not rows or not all(isinstance(x, dict) for x in rows):
            raise ValueError("invalid builder JSONL") from None
        return {"camara_tema": rows}
    if not isinstance(value, dict) or not all(isinstance(x, list) for x in value.values()):
        raise ValueError("rows must be task->list or builder JSONL")
    return value


def legacy_orders(rows):
    rng, result = random.Random(SEED), {}
    for task in sorted(rows):
        result[task] = [
            list(ptbr_bench.question(row, rng).get("criteria", {})) for row in rows[task]
        ]
    return result


def _legacy(rows):
    if set(rows) != set(LEGACY_COUNTS) or {k: len(v) for k, v in rows.items()} != LEGACY_COUNTS:
        raise ValueError("legacy diagnostic requires exact V2 seven-task counts")
    for task, values in rows.items():
        for i, row in enumerate(values):
            if (
                not isinstance(row, dict)
                or not isinstance(row.get("text"), str)
                or not isinstance(row.get("instructions"), str)
                or "\0" in row["text"]
                or "\0" in row["instructions"]
            ):
                raise ValueError(f"invalid legacy row {task}[{i}]")
            if row.get("type") == "noul":
                if not isinstance(row.get("gold"), bool):
                    raise ValueError(f"invalid legacy row {task}[{i}]")
                continue
            criteria = row.get("criteria")
            if (
                row.get("type") != "choice"
                or not isinstance(criteria, dict)
                or not criteria
                or not all(
                    isinstance(key, str)
                    and isinstance(value, str)
                    and "\0" not in key
                    and "\0" not in value
                    for key, value in criteria.items()
                )
                or not isinstance(row.get("gold"), str)
                or row["gold"] not in criteria
            ):
                raise ValueError(f"invalid legacy row {task}[{i}]")


def _v3(rows):
    if len(rows) != 599:
        raise ValueError("V3 Câmara requires exactly 599 builder rows")
    criteria = rows[0].get("criteria") if rows else None
    if (
        not isinstance(criteria, dict)
        or len(criteria) != 32
        or not all(
            isinstance(key, str)
            and isinstance(value, str)
            and "\0" not in key
            and "\0" not in value
            for key, value in criteria.items()
        )
    ):
        raise ValueError("V3 builder must carry 32 official criteria")
    for i, row in enumerate(rows):
        labels = row.get("accepted_labels")
        if (
            row.get("type") != "choice"
            or not isinstance(row.get("text"), str)
            or "\0" in row["text"]
            or not isinstance(row.get("instructions"), str)
            or "\0" in row["instructions"]
            or row.get("criteria") != criteria
            or not isinstance(labels, list)
            or not labels
            or len(labels) != len(set(labels))
            or not all(isinstance(x, str) and x in criteria for x in labels)
        ):
            raise ValueError(f"invalid V3 builder row {i}")
    return list(criteria)


def prepare(rows_file, output, *, mode="v1legacydiag", builder_file=None):
    source = builder_file if mode == "v3camara" and builder_file else rows_file
    rows = _read_rows(source)
    value = {
        "schema_version": PREPARED_SCHEMA,
        "mode": mode,
        "seed": SEED,
        "rows_source_sha256": source_hash(source),
        "tasks": {},
    }
    if mode == "v1legacydiag":
        _legacy(rows)
        orders = legacy_orders(rows)
        legal = rows["juristcu_area"]
        if sum(len(x["text"]) > PREFIX_CHARS for x in legal) != AFFECTED_ROWS:
            raise ValueError("affected legal rows must be exactly 429")
        value.update(
            {
                "legacy_counts": LEGACY_COUNTS,
                "legacy_order_stream_sha256": sha(orders),
                "tasks": {
                    "juristcu_area": [
                        {
                            "original_index": i,
                            "gold": r["gold"],
                            "text_sha256": sha(r["text"]),
                            "row_sha256": sha(r),
                            "question_order": orders["juristcu_area"][i],
                        }
                        for i, r in enumerate(legal)
                    ]
                },
            }
        )
    elif mode == "v3legalfull":
        _legacy(rows)
        orders = legacy_orders(rows)
        legal = rows["juristcu_area"]
        value.update(
            {
                "legacy_counts": LEGACY_COUNTS,
                "legacy_order_stream_sha256": sha(orders),
                "tasks": {
                    "juristcu_area": [
                        {
                            "original_index": i,
                            "gold": r["gold"],
                            "text_sha256": sha(r["text"]),
                            "row_sha256": sha(r),
                            "question_order": orders["juristcu_area"][i],
                        }
                        for i, r in enumerate(legal)
                    ]
                },
            }
        )
    elif mode == "v3camara":
        builder = rows.get("camara_tema")
        names = _v3(builder) if isinstance(builder, list) else None
        if names is None:
            raise ValueError("V3 builder has no Câmara rows")
        rng = random.Random(V3_CAMARA_SEED)
        frozen = []
        for i, r in enumerate(builder):
            order = list(names)
            rng.shuffle(order)
            frozen.append(
                {
                    "original_index": i,
                    "accepted_labels": list(r["accepted_labels"]),
                    "text_sha256": sha(r["text"]),
                    "row_sha256": sha(r),
                    "question_order": order,
                }
            )
        value.update(
            {"v3_seed": V3_CAMARA_SEED, "official_names": names, "tasks": {"camara_tema": frozen}}
        )
    else:
        raise ValueError("unknown mode")
    value["prepared_sha256"] = sha(value)
    publish_exclusive(output, value)
    return value


def load_prepared(path):
    return _validate_prepared(_read_json(path))


def _validate_prepared(value):
    value = dict(value) if isinstance(value, dict) else value
    claimed = value.pop("prepared_sha256", None) if isinstance(value, dict) else None
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != PREPARED_SCHEMA
        or not isinstance(claimed, str)
        or sha(value) != claimed
    ):
        raise ValueError("prepared identity/schema mismatch")
    value["prepared_sha256"] = claimed
    if value.get("mode") == "v1legacydiag":
        tasks = value.get("tasks")
        frozen = tasks.get("juristcu_area") if isinstance(tasks, dict) else None
        if (
            value.get("seed") == SEED
            and value.get("legacy_counts") == LEGACY_COUNTS
            and isinstance(tasks, dict)
            and set(tasks) == {"juristcu_area"}
            and _sha256_string(value.get("rows_source_sha256"))
            and _sha256_string(value.get("legacy_order_stream_sha256"))
            and isinstance(frozen, list)
            and len(frozen) == LEGACY_COUNTS["juristcu_area"]
            and _frozen_legal_rows(frozen)
        ):
            return value
    if value.get("mode") == "v3legalfull":
        tasks = value.get("tasks")
        frozen = tasks.get("juristcu_area") if isinstance(tasks, dict) else None
        if (
            value.get("seed") == SEED
            and value.get("legacy_counts") == LEGACY_COUNTS
            and isinstance(tasks, dict)
            and set(tasks) == {"juristcu_area"}
            and _sha256_string(value.get("rows_source_sha256"))
            and _sha256_string(value.get("legacy_order_stream_sha256"))
            and isinstance(frozen, list)
            and len(frozen) == LEGACY_COUNTS["juristcu_area"]
            and _frozen_legal_rows(frozen)
        ):
            return value
    if value.get("mode") == "v3camara":
        tasks = value.get("tasks")
        frozen = tasks.get("camara_tema") if isinstance(tasks, dict) else None
        names = value.get("official_names")
        if (
            value.get("v3_seed") == V3_CAMARA_SEED
            and isinstance(tasks, dict)
            and set(tasks) == {"camara_tema"}
            and _sha256_string(value.get("rows_source_sha256"))
            and isinstance(names, list)
            and len(names) == 32
            and len(set(names)) == 32
            and all(isinstance(name, str) and "\0" not in name for name in names)
            and isinstance(frozen, list)
            and len(frozen) == LEGACY_COUNTS["camara_tema"]
            and _frozen_v3_rows(frozen, names)
        ):
            return value
    raise ValueError("prepared mode contract missing")


def _sha256_string(value):
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


def _frozen_legal_rows(rows):
    expected_keys = {"original_index", "gold", "text_sha256", "row_sha256", "question_order"}
    return all(
        isinstance(row, dict)
        and set(row) == expected_keys
        and isinstance(row["original_index"], int)
        and not isinstance(row["original_index"], bool)
        and row["original_index"] == i
        and isinstance(row["gold"], str)
        and _sha256_string(row["text_sha256"])
        and _sha256_string(row["row_sha256"])
        and isinstance(row["question_order"], list)
        and len(row["question_order"]) == len(set(row["question_order"]))
        and all(isinstance(key, str) for key in row["question_order"])
        for i, row in enumerate(rows)
    )


def _frozen_v3_rows(rows, names):
    expected_keys = {
        "original_index",
        "accepted_labels",
        "text_sha256",
        "row_sha256",
        "question_order",
    }
    names_set = set(names)
    return all(
        isinstance(row, dict)
        and set(row) == expected_keys
        and isinstance(row["original_index"], int)
        and not isinstance(row["original_index"], bool)
        and row["original_index"] == i
        and isinstance(row["accepted_labels"], list)
        and row["accepted_labels"]
        and len(row["accepted_labels"]) == len(set(row["accepted_labels"]))
        and all(isinstance(label, str) and label in names_set for label in row["accepted_labels"])
        and _sha256_string(row["text_sha256"])
        and _sha256_string(row["row_sha256"])
        and isinstance(row["question_order"], list)
        and len(row["question_order"]) == len(names)
        and set(row["question_order"]) == names_set
        for i, row in enumerate(rows)
    )


def _checkpoint(c):
    if (
        not isinstance(c, dict)
        or set(c) != {"id", "revision", "base"}
        or not all(isinstance(x, str) and x for x in c.values())
    ):
        raise ValueError("checkpoint must declare id, revision, and base")


def load_server_evidence(path, checkpoint):
    e = _read_json(path)
    required = (
        "checkpoint",
        "runtime",
        "backend",
        "dtype",
        "context",
        "prompt",
        "server",
        "preflight",
    )
    if (
        not isinstance(e, dict)
        or any(not e.get(k) for k in required)
        or not isinstance(e["checkpoint"], dict)
        or any(e["checkpoint"].get(k) != checkpoint[k] for k in checkpoint)
    ):
        raise ValueError("server evidence lacks mandatory pinned metadata")
    p = e["preflight"]
    if (
        not isinstance(p, dict)
        or p.get("no_truncation") is not True
        or not p.get("context_check")
        or not p.get("token_check")
    ):
        raise ValueError("server preflight lacks no-truncation context/token proof")
    return e


def _live_llm_evidence(health, expected):
    server = expected.get("server") if isinstance(expected, dict) else None
    pinned = server.get("health_evidence") if isinstance(server, dict) else None
    pinned_sha256 = server.get("health_evidence_sha256") if isinstance(server, dict) else None
    actual = health.get("evidence") if isinstance(health, dict) else None
    claimed_sha256 = health.get("evidence_sha256") if isinstance(health, dict) else None
    if (
        not isinstance(pinned, dict)
        or not _sha256_string(pinned_sha256)
        or not isinstance(actual, dict)
        or not _sha256_string(claimed_sha256)
        or sha(canonical(actual) + b"\n") != claimed_sha256
        or claimed_sha256 != pinned_sha256
        or actual != pinned
    ):
        raise ValueError("live LLM health evidence is not bound to independent evidence")
    model = actual.get("model")
    backend = actual.get("backend")
    limits = actual.get("limits")
    if (
        not isinstance(model, dict)
        or model.get("id") != expected["checkpoint"]["id"]
        or model.get("revision") != expected["checkpoint"]["revision"]
        or not isinstance(backend, dict)
        or backend != expected["backend"]
        or backend != expected["runtime"]
        or backend.get("dtype") != expected["dtype"]
        or not isinstance(limits, dict)
        or limits.get("context") != expected["context"]
        or actual.get("prompt") != expected["prompt"]
    ):
        raise ValueError("live LLM health evidence disagrees with pinned runtime metadata")


def preflight(base, expected_independent_evidence):
    with urllib.request.urlopen(loopback_url(base) + "/health", timeout=10) as r:
        raw = r.read(1024 * 1024 + 1)
        header_digest = r.headers.get("X-Saracura-Evidence-SHA256")
    if len(raw) > 1024 * 1024:
        raise ValueError("oversized server health")
    try:
        answer = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError("invalid server health") from e
    if not isinstance(answer, dict):
        raise ValueError("invalid server health")
    if header_digest is not None:
        if not _sha256_string(header_digest) or header_digest != sha(expected_independent_evidence):
            raise ValueError("native health evidence header does not bind independent evidence")
        return answer
    _live_llm_evidence(answer, expected_independent_evidence)
    return answer


def _number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _prediction(result, q):
    if not isinstance(result, dict):
        return None
    answers = result.get("answers")
    answer = answers.get("q") if isinstance(answers, dict) else None
    if not isinstance(answer, dict) or any(
        part.get("truncated", False) is not False or part.get("error") not in (None, False, "")
        for part in (result, answers, answer)
        if isinstance(part, dict)
    ):
        return None
    choice = answer.get("choice")
    if "choice" in answer and (not isinstance(choice, str) or choice not in q["criteria"]):
        return None
    probabilities = answer.get("probabilities")
    native_prediction = None
    if "probabilities" in answer:
        if (
            not isinstance(probabilities, dict)
            or set(probabilities) != set(q["criteria"])
            or not all(_number(v) for v in probabilities.values())
        ):
            return None
        native_prediction = max(q["criteria"], key=lambda key: probabilities[key])
    if (
        choice is not None
        and native_prediction is not None
        and probabilities[choice] != probabilities[native_prediction]
    ):
        return None
    return choice if choice is not None else native_prediction


def request(base, model, state, q, timeout=120):
    body = wire_body(model, state, q)
    started = time.monotonic()
    req = urllib.request.Request(
        loopback_url(base) + "/v1/systemone",
        data=body,
        headers={"content-type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read(1024 * 1024 + 1)
        try:
            raw_text = raw.decode("utf-8")
        except UnicodeDecodeError:
            return None, None, round(time.monotonic() - started, 6), "invalid_utf8", sha(body)
        if len(raw) > 1024 * 1024:
            return (
                None,
                raw_text,
                round(time.monotonic() - started, 6),
                "oversized_response",
                sha(body),
            )
        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            return None, raw_text, round(time.monotonic() - started, 6), "invalid_json", sha(body)
        if _prediction(result, q) is None:
            return (
                result,
                raw_text,
                round(time.monotonic() - started, 6),
                "server_error_or_invalid_prediction",
                sha(body),
            )
        return (
            result,
            raw_text,
            round(time.monotonic() - started, 6),
            None,
            sha(body),
        )
    except (OSError, ValueError) as e:
        return None, None, round(time.monotonic() - started, 6), type(e).__name__, sha(body)


def _expected(prepared, rows, model, checkpoint, evidence, evidence_file):
    if prepared.get("mode") == "v3legalfull":
        raise ValueError("full legal prepared mode is not supported by diagnostic execution")
    task = "juristcu_area" if prepared["mode"] == "v1legacydiag" else "camara_tema"
    original = rows.get(task)
    frozen = prepared["tasks"].get(task)
    if (
        not isinstance(original, list)
        or not isinstance(frozen, list)
        or len(original) != len(frozen)
    ):
        raise ValueError("prepared rows mismatch")
    if task == "juristcu_area":
        _legacy(rows)
        replayed_orders = legacy_orders(rows)[task]
        targets = [i for i, r in enumerate(original) if len(r["text"]) > PREFIX_CHARS]
        if len(targets) != AFFECTED_ROWS:
            raise ValueError("affected scope is not exactly 429")
        arms = ("prefix", "full")
    else:
        names = _v3(original)
        if prepared.get("official_names") != names:
            raise ValueError("prepared official criteria mismatch")
        rng = random.Random(V3_CAMARA_SEED)
        replayed_orders = []
        for _ in original:
            order = list(names)
            rng.shuffle(order)
            replayed_orders.append(order)
        targets = list(range(599))
        arms = ("full",)
    identity = {
        "prepared_sha256": prepared["prepared_sha256"],
        "rows_sha256": prepared["rows_source_sha256"],
        "model": model,
        "checkpoint": checkpoint,
        "server_evidence_sha256": sha(evidence),
        "server_evidence_file_sha256": source_hash(evidence_file),
    }
    output = {}
    for i in targets:
        row, f = original[i], frozen[i]
        if (
            f.get("original_index") != i
            or sha(row) != f.get("row_sha256")
            or sha(row["text"]) != f.get("text_sha256")
            or f.get("question_order") != replayed_orders[i]
        ):
            raise ValueError("source row or frozen order changed since prepare")
        order, criteria = f.get("question_order"), row.get("criteria")
        if (
            not isinstance(order, list)
            or not isinstance(criteria, dict)
            or set(order) != set(criteria)
        ):
            raise ValueError("frozen question order mismatch")
        q = {
            "type": "choice",
            "instructions": row.get("instructions"),
            "criteria": {k: criteria[k] for k in order},
        }
        gold = f.get("gold", f.get("accepted_labels"))
        if gold != (row.get("gold") if task == "juristcu_area" else row.get("accepted_labels")):
            raise ValueError("frozen gold mismatch")
        for arm in arms:
            state = row["text"] if arm == "full" else row["text"][:PREFIX_CHARS]
            output[i, arm] = {
                "identity": identity,
                "original_index": i,
                "task": task,
                "mode": prepared["mode"],
                "arm": arm,
                "exact_chars_sent": len(state),
                "gold": gold,
                "question_order": order,
                "checkpoint": checkpoint,
                "runtime": evidence["runtime"],
                "backend": evidence["backend"],
                "dtype": evidence["dtype"],
                "context": evidence["context"],
                "prompt_sha256": sha(evidence["prompt"]),
                "server_sha256": sha(evidence["server"]),
                "preflight_sha256": sha(evidence["preflight"]),
                "server_evidence_sha256": sha(evidence),
                "source_file_hashes": {
                    "rows": prepared["rows_source_sha256"],
                    "prepared": prepared["prepared_sha256"],
                    "server_evidence": source_hash(evidence_file),
                },
                "question": q,
                "state": state,
            }
    return output


def validate_journal(
    journal, prepared, rows, model, checkpoint, evidence, evidence_file, *, complete
):
    prepared = _validate_prepared(prepared)
    if prepared.get("mode") == "v3legalfull":
        raise ValueError(
            "full legal prepared mode is not supported by diagnostic journal validation"
        )
    expected = _expected(prepared, rows, model, checkpoint, evidence, evidence_file)
    path = Path(journal)
    raw = path.read_bytes() if path.exists() else b""
    if raw and not raw.endswith(b"\n"):
        raise ValueError("journal missing terminal newline")
    records = {}
    expected_keys = set(next(iter(expected.values())).keys()) - {"question", "state"} | {
        "truncated",
        "prediction",
        "raw_response",
        "response_sha256",
        "duration_seconds",
        "error",
        "request_sha256",
    }
    for line in raw.splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError("invalid journal JSON") from e
        if (
            not isinstance(r, dict)
            or not isinstance(r.get("original_index"), int)
            or isinstance(r["original_index"], bool)
            or not isinstance(r.get("arm"), str)
        ):
            raise ValueError("journal identity is invalid")
        key = (r["original_index"], r["arm"])
        if key not in expected or key in records:
            raise ValueError("journal has an extra, duplicate, or out-of-scope record")
        x = expected[key]
        if set(r) != expected_keys or any(
            r[k] != x[k]
            for k in expected_keys
            - {
                "truncated",
                "prediction",
                "raw_response",
                "response_sha256",
                "duration_seconds",
                "error",
                "request_sha256",
            }
        ):
            raise ValueError("journal metadata mismatch")
        if r["truncated"] is not False or r["error"] is not None:
            raise ValueError("journal preserves a failed or truncated attempt and blocks resume")
        if (
            not isinstance(r["prediction"], str)
            or r["prediction"] not in x["question"]["criteria"]
            or not isinstance(r["raw_response"], str)
            or r["response_sha256"] != raw_response_sha(r["raw_response"])
            or not _number(r["duration_seconds"])
            or r["duration_seconds"] < 0
        ):
            raise ValueError("journal prediction/response invalid")
        try:
            raw_response = json.loads(r["raw_response"])
        except json.JSONDecodeError as e:
            raise ValueError("journal response is not JSON") from e
        if _prediction(raw_response, x["question"]) != r["prediction"]:
            raise ValueError("journal prediction does not match response")
        if r["request_sha256"] != sha(wire_body(model, x["state"], x["question"])):
            raise ValueError("journal request hash mismatch")
        records[key] = r
    if complete and set(records) != set(expected):
        raise ValueError("journal is incomplete")
    return expected, records


def run_pairs(
    prepared_file,
    rows_file,
    journal,
    base,
    model,
    *,
    checkpoint,
    server_evidence_file,
    original_report=None,
    timeout=120,
):
    prepared = load_prepared(prepared_file)
    if prepared.get("mode") == "v3legalfull":
        raise ValueError("full legal prepared mode is not supported by diagnostic pairs")
    rows = _read_rows(rows_file)
    if source_hash(rows_file) != prepared["rows_source_sha256"]:
        raise ValueError("source hash mismatch")
    _checkpoint(checkpoint)
    evidence = load_server_evidence(server_evidence_file, checkpoint)
    preflight(base, evidence)
    expected, prior = validate_journal(
        journal, prepared, rows, model, checkpoint, evidence, server_evidence_file, complete=False
    )
    for key, x in expected.items():
        if key in prior:
            continue
        answer, raw, duration, error, request_hash = request(
            base, model, x["state"], x["question"], timeout
        )
        prediction = _prediction(answer, x["question"])
        if error is None and prediction is None:
            error = "invalid_choice"
        record = {k: v for k, v in x.items() if k not in {"question", "state"}} | {
            "truncated": False,
            "prediction": prediction,
            "raw_response": raw,
            "response_sha256": raw_response_sha(raw) if isinstance(raw, str) else None,
            "duration_seconds": duration,
            "error": error,
            "request_sha256": request_hash,
        }
        append_jsonl(journal, record)
        if error is not None:
            raise ValueError("operational failure preserved in attempt journal")
    return summarize(
        prepared,
        rows,
        journal,
        model=model,
        checkpoint=checkpoint,
        server_evidence_file=server_evidence_file,
        original_report=original_report,
    )


def _report_predictions(report, rows):
    if report is None:
        return None
    v = _read_json(report) if isinstance(report, (str, Path)) else report
    pairs = v.get("items", {}).get("juristcu_area") if isinstance(v, dict) else None
    if not isinstance(pairs, list) or len(pairs) != 893 or len(rows) != 893:
        raise ValueError("original V2 report lacks all 893 legal pairs")
    out = []
    for i, (pair, row) in enumerate(zip(pairs, rows, strict=True)):
        if (
            not isinstance(pair, list)
            or len(pair) != 2
            or not all(isinstance(x, str) for x in pair)
            or pair[0] != row.get("gold")
            or pair[1] not in row.get("criteria", {})
        ):
            raise ValueError(f"invalid original V2 legal pair {i}")
        out.append(pair[1])
    return out


def summarize(
    prepared, rows, journal, *, model, checkpoint, server_evidence_file, original_report=None
):
    prepared = (
        _validate_prepared(prepared) if isinstance(prepared, dict) else load_prepared(prepared)
    )
    if prepared.get("mode") == "v3legalfull":
        raise ValueError("full legal prepared mode is not supported by diagnostic summary")
    _checkpoint(checkpoint)
    evidence = load_server_evidence(server_evidence_file, checkpoint)
    expected, records = validate_journal(
        journal, prepared, rows, model, checkpoint, evidence, server_evidence_file, complete=True
    )
    task = "juristcu_area" if prepared["mode"] == "v1legacydiag" else "camara_tema"
    targets = sorted({k[0] for k in expected})
    support = defaultdict(int)
    diff = defaultdict(int)
    flips = {
        "wrong_to_right": 0,
        "right_to_wrong": 0,
        "wrong_to_other_wrong": 0,
        "unchanged_prediction": 0,
    }
    for row in rows[task]:
        if isinstance(row.get("gold"), str):
            support[row["gold"]] += 1
    if task == "juristcu_area":
        for i in targets:
            b, a = records[i, "prefix"], records[i, "full"]
            gold = expected[i, "prefix"]["gold"]
            bo, ao = b["prediction"] == gold, a["prediction"] == gold
            diff[gold] += int(ao) - int(bo)
            if not bo and ao:
                flips["wrong_to_right"] += 1
            elif bo and not ao:
                flips["right_to_wrong"] += 1
            elif not bo and not ao and b["prediction"] != a["prediction"]:
                flips["wrong_to_other_wrong"] += 1
            else:
                flips["unchanged_prediction"] += 1
    original = _report_predictions(original_report, rows[task])
    agreement = (
        None
        if original is None or task != "juristcu_area"
        else sum(records[i, "prefix"]["prediction"] == original[i] for i in targets) / len(targets)
    )
    delta = 100 * sum(diff[k] / v for k, v in support.items()) / len(support) if support else 0
    return {
        "complete": True,
        "expected_pairs": len(targets),
        "completed_pairs": len(targets),
        "expected_attempts": len(expected),
        "completed_attempts": len(records),
        "hybrid_reconstruction": task == "juristcu_area",
        "not_a_new_893_row_full_run": task == "juristcu_area",
        "delta_balanced_accuracy_points": delta,
        "per_class_support": dict(support),
        "raw_flip_counts": flips,
        "prefix_vs_original_v2_concordance": agreement,
        "prefix_agreement_denominator": len(targets) if agreement is not None else None,
        "material_threshold_points": 3.0,
        "material_gate": bool(task == "juristcu_area" and abs(delta) >= 3),
        "no_statistical_no_effect_claim": True,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subs = parser.add_subparsers(dest="command", required=True)
    p = subs.add_parser("prepare")
    p.add_argument("--rows", required=True)
    p.add_argument("--output", required=True)
    p.add_argument(
        "--mode", choices=("v1legacydiag", "v3legalfull", "v3camara"), default="v1legacydiag"
    )
    p.add_argument("--builder")
    run = subs.add_parser("run")
    for n in (
        "prepared",
        "rows",
        "journal",
        "base-url",
        "model",
        "checkpoint-id",
        "checkpoint-revision",
        "checkpoint-base",
        "server-evidence",
    ):
        run.add_argument("--" + n, required=True)
    run.add_argument("--original-report")
    run.add_argument("--timeout", type=float, default=120)
    run.add_argument("--summary-output")
    a = parser.parse_args(argv)
    if a.command == "prepare":
        prepare(a.rows, a.output, mode=a.mode, builder_file=a.builder)
        return
    result = run_pairs(
        a.prepared,
        a.rows,
        a.journal,
        a.base_url,
        a.model,
        checkpoint={
            "id": a.checkpoint_id,
            "revision": a.checkpoint_revision,
            "base": a.checkpoint_base,
        },
        server_evidence_file=a.server_evidence,
        original_report=a.original_report,
        timeout=a.timeout,
    )
    if a.summary_output:
        publish_exclusive(a.summary_output, result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
