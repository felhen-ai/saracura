"""Phase 3D native TypeSafe control benchmark (no SDK, offline by default)."""

from __future__ import annotations

import argparse
import json
import math
import os
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from contextlib import suppress
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Literal, Protocol, cast

from pydantic import BaseModel, ConfigDict, JsonValue

from benchmarks.e2e_benchmark import (
    JEV_BATCH_SIZES as BATCH_SIZES,
)
from benchmarks.e2e_benchmark import (
    JEV_ITERATIONS as ITERATIONS,
)
from benchmarks.e2e_benchmark import (
    JEV_ORDER as BATCH_ORDER,
)
from benchmarks.e2e_benchmark import (
    build_jev_request,
    digest_bytes,
    digest_json,
    opaque_input_digest,
    summarize,
    throughput,
    validate_phase3c_artifacts,
)
from benchmarks.io import atomic_create
from benchmarks.synthetic_research import LABELS, _read_accepted, validate_packet_manifest
from saracura.serialization import canonical_json_bytes

ROOT = Path(__file__).parents[1]
ENDPOINT = "https://api.typesafe.ai/v1/systemone"
MODEL_ID = "jev-1.13.0"
SPEC_PATH = ROOT / "docs/action/specs/phase3d-typesafe-native-control.md"
PHASE3C_SOURCE = ROOT / "benchmarks/e2e_benchmark.py"
RESERVATION_TOKENS = 64000
RESERVATION = Decimal("0.0032")
INPUT_RATE = Decimal("0.042")
BUDGET = Decimal("0.25")
SOURCE_ALLOWLIST = (
    "benchmarks/typesafe_native.py",
    "benchmarks/e2e_benchmark.py",
    "docs/action/specs/phase3d-typesafe-native-control.md",
)
EXPECTED_PROTOCOL = {
    "batch_sizes": list(BATCH_SIZES),
    "batch_order": list(BATCH_ORDER),
    "iterations": ITERATIONS,
    "endpoint": ENDPOINT,
    "model": MODEL_ID,
    "reservation_tokens": RESERVATION_TOKENS,
    "reservation_rate_usd_per_million": "0.05",
    "computed_input_rate_usd_per_million": "0.042",
    "timeout_seconds": 60,
    "attempts": 1,
}

MANDATORY_LIMITATIONS = (
    "the prior local warm path excludes one-time model load",
    "both remote paths include WAN latency and uncontrollable provider state",
    "TypeSafe native and OpenRouter use different public revision identifiers; "
    "equal Jev-family naming does not prove identical weights or serving stacks",
    "native computed cost is not a provider billing receipt",
    "confidence describes distribution concentration, not workflow correctness",
    "accuracy is an in-sample diagnostic on synthetic accepted rows",
    "the synthetic PT-BR workload cannot establish general PT-BR quality",
    "no result may be used to choose an automation threshold",
    "Phase 3C retained no corresponding OpenRouter probability distributions; "
    "cross-gateway probability comparison is unavailable",
    "raw remote p95 remains descriptive over ten samples",
)


class ClosedModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False, frozen=True)


class SummaryModel(ClosedModel):
    count: int
    min_ms: float
    p50_ms: float
    p95_ms: float
    max_ms: float


class AnswerModel(ClosedModel):
    choice: Literal[
        "billing",
        "technical_support",
        "account_access",
        "subscription_cancellation",
        "order_delivery",
    ]
    probabilities: dict[
        Literal[
            "billing",
            "technical_support",
            "account_access",
            "subscription_cancellation",
            "order_delivery",
        ],
        float,
    ]
    confidence: float


class NativeSampleModel(ClosedModel):
    latency_ms: float
    answers: dict[str, AnswerModel]
    selected_choice_digest: str
    probability_distribution_digest: str
    accuracy: float
    confidence_mean: float
    confidence_min: float
    confidence_max: float
    input_tokens: int
    output_tokens: int
    computed_cost_usd: str
    model: Literal["jev-1.13.0"]


class NativeWorkloadModel(ClosedModel):
    batch_size: int
    iterations: Literal[10]
    input_id_digest: str
    request_structure_digest: str
    samples: list[NativeSampleModel]
    samples_sha256: str
    summary: SummaryModel
    per_item_summary: SummaryModel
    throughput_items_per_second: float
    accuracy: float
    input_tokens: int
    output_tokens: int
    computed_cost_usd: str


class ProtocolModel(ClosedModel):
    batch_sizes: list[int]
    batch_order: list[int]
    iterations: Literal[10]
    endpoint: Literal["https://api.typesafe.ai/v1/systemone"]
    model: Literal["jev-1.13.0"]
    reservation_tokens: Literal[64000]
    reservation_rate_usd_per_million: Literal["0.05"]
    computed_input_rate_usd_per_million: Literal["0.042"]
    timeout_seconds: Literal[60]
    attempts: Literal[1]


class ProvenanceModel(ClosedModel):
    predecessor_artifact_manifest_sha256: str
    packet_manifest_sha256: str
    phase3c_request_builder_sha256: str
    phase3d_source_sha256: str
    spec_sha256: str
    request_structure_digests: dict[str, str]


class ComparisonEntryModel(ClosedModel):
    batch_size: int
    local_request_p50_ms: float
    local_request_p95_ms: float
    local_per_item_p50_ms: float
    local_per_item_p95_ms: float
    local_throughput_items_per_second: float
    local_accuracy: float
    openrouter_request_p50_ms: float
    openrouter_request_p95_ms: float
    openrouter_per_item_p50_ms: float
    openrouter_per_item_p95_ms: float
    openrouter_throughput_items_per_second: float
    openrouter_accuracy: float
    openrouter_provider_cost_usd: str
    native_request_p50_ms: float
    native_request_p95_ms: float
    native_per_item_p50_ms: float
    native_per_item_p95_ms: float
    native_throughput_items_per_second: float
    native_accuracy: float
    native_computed_cost_usd: str
    local_to_native_per_item_p50_ratio: float
    local_to_openrouter_per_item_p50_ratio: float
    native_to_openrouter_request_p50_ratio: float


class NativeDiagnosticsModel(ClosedModel):
    input_tokens: int
    output_tokens: int
    confidence_mean: float
    confidence_min: float
    confidence_max: float
    probability_distribution_digests: list[str]


class CompleteNativeModel(ClosedModel):
    status: Literal["complete"]
    completed_call_count: Literal[30]
    resolved_models: list[Literal["jev-1.13.0"]]
    computed_cost_usd: str
    workloads: dict[str, NativeWorkloadModel]
    diagnostics: dict[str, NativeDiagnosticsModel]


class PartialNativeModel(ClosedModel):
    status: Literal["partial"]
    completed_call_count: int
    completed_sample_digests: list[str]
    resolved_models: list[Literal["jev-1.13.0"]]
    computed_cost_usd: str
    failure_category: (
        Literal[
            "http_error",
            "timeout",
            "transport_error",
            "response_invalid",
            "process_interrupted",
            "reported_usage_over_limit",
        ]
        | None
    )
    journal_state: Literal["settled", "failed", "open_reservation", "settled_over_limit"]


class CompleteResultModel(ClosedModel):
    schema_version: Literal["phase3d.v1"]
    sealed: Literal[True]
    protocol: ProtocolModel
    provenance: ProvenanceModel
    native: CompleteNativeModel
    comparison: dict[str, ComparisonEntryModel]
    limitations: list[str]


class PartialResultModel(ClosedModel):
    schema_version: Literal["phase3d.v1"]
    sealed: Literal[True]
    protocol: ProtocolModel
    provenance: ProvenanceModel
    native: PartialNativeModel
    limitations: list[str]


class Transport(Protocol):
    def __call__(
        self, method: str, url: str, headers: Mapping[str, str], body: bytes, *, timeout: float
    ) -> tuple[int, Mapping[str, str], bytes]: ...


class NativeTransportTimeout(TimeoutError):
    """Stable timeout signal that cannot include provider or request content."""


def _no_dupes(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError("duplicate JSON key")
        out[key] = value
    return out


def _load(raw: bytes) -> Any:
    return json.loads(raw, object_pairs_hook=_no_dupes)


def _dec(value: Any) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError("invalid decimal") from exc
    if not result.is_finite() or result < 0:
        raise ValueError("invalid decimal")
    return result


def decimal_string(value: Decimal) -> str:
    value = _dec(value)
    return "0" if value == 0 else format(value, "f").rstrip("0").rstrip(".")


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _canon(value: Any) -> bytes:
    return canonical_json_bytes(cast(Any, value))


def native_request(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    source = build_jev_request(rows)
    if set(source) != {"model", "state", "questions", "provider"}:
        raise ValueError("Phase 3C request shape drift")
    result = dict(source)
    result.pop("provider")
    result["model"] = MODEL_ID
    if set(result) != {"model", "state", "questions"}:
        raise ValueError("native request shape")
    return result


build_typesafe_request = native_request


def payload_structure_digest(payload: Mapping[str, Any]) -> str:
    return digest_json(payload)


def validate_native_response(
    response: Mapping[str, Any], question_keys: Sequence[str]
) -> dict[str, Any]:
    if set(response) != {"model", "answers", "usage"} or response.get("model") != MODEL_ID:
        raise ValueError("native response shape or model")
    answers = response["answers"]
    if not isinstance(answers, Mapping) or set(answers) != set(question_keys):
        raise ValueError("native answer keys")
    reduced: dict[str, Any] = {}
    for key in question_keys:
        answer = answers[key]
        if (
            not isinstance(answer, Mapping)
            or set(answer) != {"type", "choice", "probabilities", "confidence"}
            or answer["type"] != "choice"
            or answer["choice"] not in LABELS
        ):
            raise ValueError("native choice shape")
        probabilities = answer["probabilities"]
        if not isinstance(probabilities, Mapping) or set(probabilities) != set(LABELS):
            raise ValueError("native probability labels")
        values = [probabilities[label] for label in LABELS]
        confidence = answer["confidence"]
        if any(not _finite(v) or not 0 <= float(v) <= 1 for v in values) or not math.isclose(
            sum(map(float, values)), 1, rel_tol=0, abs_tol=1e-6
        ):
            raise ValueError("native probabilities")
        if (
            not _finite(confidence)
            or not 0 <= float(confidence) <= 1
            or float(probabilities[answer["choice"]]) < max(map(float, values))
        ):
            raise ValueError("native confidence or selected choice")
        reduced[key] = {
            "choice": answer["choice"],
            "probabilities": {label: float(probabilities[label]) for label in LABELS},
            "confidence": float(confidence),
        }
    usage = response["usage"]
    if (
        not isinstance(usage, Mapping)
        or set(usage) != {"input_tokens", "output_tokens"}
        or any(
            not isinstance(usage[k], int) or isinstance(usage[k], bool) or usage[k] < 0
            for k in usage
        )
    ):
        raise ValueError("native usage")
    return {"model": MODEL_ID, "answers": reduced, "usage": dict(usage)}


def reduce_sample(
    validated: Mapping[str, Any], latency_ms: float, expected: Sequence[str]
) -> dict[str, Any]:
    answers = validated["answers"]
    choices = [answers[k]["choice"] for k in answers]
    probs = [[answers[k]["probabilities"][label] for label in LABELS] for k in answers]
    conf = [float(answers[k]["confidence"]) for k in answers]
    tokens = validated["usage"]
    if not _finite(latency_ms) or latency_ms < 0:
        raise ValueError("latency")
    return {
        "latency_ms": float(latency_ms),
        "answers": dict(answers),
        "selected_choice_digest": digest_json(choices),
        "probability_distribution_digest": digest_json(probs),
        "accuracy": sum(a == b for a, b in zip(choices, expected, strict=True)) / len(expected),
        "confidence_mean": sum(conf) / len(conf),
        "confidence_min": min(conf),
        "confidence_max": max(conf),
        "input_tokens": tokens["input_tokens"],
        "output_tokens": tokens["output_tokens"],
        "computed_cost_usd": decimal_string(
            Decimal(tokens["input_tokens"]) * INPUT_RATE / Decimal(1000000)
        ),
        "model": MODEL_ID,
    }


def _urllib_transport(
    method: str, url: str, headers: Mapping[str, str], body: bytes, *, timeout: float
) -> tuple[int, Mapping[str, str], bytes]:
    parsed = urllib.parse.urlparse(url)
    if (
        url != ENDPOINT
        or parsed.scheme != "https"
        or parsed.hostname != "api.typesafe.ai"
        or parsed.path != "/v1/systemone"
    ):
        raise ValueError("literal TypeSafe endpoint required")
    request = urllib.request.Request(url, data=body, headers=dict(headers), method=method)

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(
            self, request: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str
        ) -> Any:
            return None

    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        NoRedirect(),
        urllib.request.HTTPSHandler(context=ssl.create_default_context()),
    )
    try:
        with opener.open(request, timeout=timeout) as response:
            return response.status, dict(response.headers), response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers), exc.read()
    except TimeoutError as exc:
        raise NativeTransportTimeout("native transport timed out") from exc
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, TimeoutError):
            raise NativeTransportTimeout("native transport timed out") from exc
        raise RuntimeError("native transport failed") from exc


class Journal:
    def __init__(self, path: Path, predecessor_sha256: str, budget: Decimal = BUDGET) -> None:
        self.path = path
        self.predecessor_sha256 = predecessor_sha256
        self.budget = budget
        self.rows = self._read()
        self.uncertain = any(
            r["kind"] == "reserve" and not self._closed(r["request_id"]) for r in self.rows
        )

    def _read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        raw = self.path.read_bytes()
        if raw and not raw.endswith(b"\n"):
            raise ValueError("truncated journal")
        rows: list[dict[str, Any]] = []
        for index, line in enumerate(raw.splitlines(), 1):
            row = _load(line)
            if (
                not isinstance(row, dict)
                or row.get("sequence") != index
                or row.get("predecessor_sha256") != self.predecessor_sha256
                or _canon(row) != line
            ):
                raise ValueError("journal sequence or provenance")
            rows.append(row)
        validate_journal_rows(rows, self.predecessor_sha256)
        return rows

    def _closed(self, request_id: str) -> bool:
        return any(
            r.get("request_id") == request_id and r["kind"] in {"settle", "settled_over_limit"}
            for r in self.rows
        )

    def _has_open_reservation(self) -> bool:
        return any(
            row["kind"] == "reserve" and not self._closed(row["request_id"]) for row in self.rows
        )

    @property
    def settled(self) -> Decimal:
        return sum(
            (
                _dec(r["computed_cost_usd"])
                for r in self.rows
                if r["kind"] in {"settle", "settled_over_limit"}
            ),
            Decimal(0),
        )

    def _append(self, row: dict[str, Any]) -> None:
        row = {"sequence": len(self.rows) + 1, **row, "predecessor_sha256": self.predecessor_sha256}
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with self.path.open("ab") as handle:
            handle.write(_canon(row) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(self.path, 0o600)
        self.rows.append(row)

    def reserve(self, request_id: str, batch_size: int, iteration: int) -> None:
        if (
            self.uncertain
            or self._has_open_reservation()
            or self._closed(request_id)
            or self.settled + RESERVATION > self.budget
            or any(r.get("request_id") == request_id for r in self.rows)
        ):
            raise ValueError("budget or journal unavailable")
        self._append(
            {
                "kind": "reserve",
                "request_id": request_id,
                "batch_size": batch_size,
                "iteration": iteration,
                "reservation_cost_usd": decimal_string(RESERVATION),
            }
        )

    def terminal(
        self,
        request_id: str,
        batch_size: int,
        iteration: int,
        kind: str,
        *,
        cost: Decimal = Decimal(0),
        sample: Mapping[str, Any] | None = None,
        category: str | None = None,
        status: int | None = None,
    ) -> None:
        if (
            kind not in {"settle", "settled_over_limit", "failure"}
            or not any(
                r.get("request_id") == request_id and r["kind"] == "reserve" for r in self.rows
            )
            or self._closed(request_id)
        ):
            self.uncertain = True
            raise ValueError("invalid journal terminal")
        row = {
            "kind": kind,
            "request_id": request_id,
            "batch_size": batch_size,
            "iteration": iteration,
        }
        if kind == "failure":
            row["category"] = category
            row.update({"http_status": status}) if status is not None else None
        else:
            row.update({"computed_cost_usd": decimal_string(cost), "sample": dict(sample or {})})
            if kind == "settled_over_limit":
                row["category"] = "reported_usage_over_limit"
        self._append(row)
        if kind in {"settled_over_limit", "failure"}:
            self.uncertain = True


CostJournal = Journal


def _expected_requests() -> list[tuple[str, int, int]]:
    return [
        (f"native-b{batch_size:03d}-i{iteration:02d}", batch_size, iteration)
        for batch_size in BATCH_ORDER
        for iteration in range(ITERATIONS)
    ]


def _canonical_decimal(value: Any) -> Decimal:
    parsed = _dec(value)
    if not isinstance(value, str) or value != decimal_string(parsed):
        raise ValueError("noncanonical decimal")
    return parsed


def _validate_sample_shape(sample: Any) -> NativeSampleModel:
    model = NativeSampleModel.model_validate(sample)
    answers = model.answers
    if not answers:
        raise ValueError("empty answers")
    choices = [answer.choice for answer in answers.values()]
    distributions = [
        [cast(dict[str, float], answer.probabilities)[label] for label in LABELS]
        for answer in answers.values()
    ]
    confidences = [answer.confidence for answer in answers.values()]
    for answer in answers.values():
        if set(answer.probabilities) != set(LABELS):
            raise ValueError("sample probability order")
        values = list(answer.probabilities.values())
        if any(not _finite(value) or not 0 <= value <= 1 for value in values):
            raise ValueError("sample probability range")
        if not math.isclose(sum(values), 1.0, rel_tol=0, abs_tol=1e-6):
            raise ValueError("sample probability sum")
        if answer.probabilities[answer.choice] < max(values):
            raise ValueError("sample choice maximum")
    if (
        not _finite(model.latency_ms)
        or model.latency_ms < 0
        or any(not _finite(value) or not 0 <= value <= 1 for value in confidences)
        or not _finite(model.accuracy)
        or not 0 <= model.accuracy <= 1
        or model.confidence_mean != sum(confidences) / len(confidences)
        or model.confidence_min != min(confidences)
        or model.confidence_max != max(confidences)
        or model.selected_choice_digest != digest_json(choices)
        or model.probability_distribution_digest != digest_json(distributions)
        or model.input_tokens < 0
        or model.output_tokens < 0
        or _canonical_decimal(model.computed_cost_usd)
        != Decimal(model.input_tokens) * INPUT_RATE / Decimal(1_000_000)
    ):
        raise ValueError("sample contradiction")
    return model


def validate_journal_rows(rows: Sequence[Mapping[str, Any]], predecessor_sha256: str) -> None:
    """Validate the append-only prefix of the frozen thirty-call protocol."""
    expected = _expected_requests()
    expected_index = 0
    open_request: tuple[str, int, int] | None = None
    settled = Decimal(0)
    for sequence, row in enumerate(rows, 1):
        if row.get("sequence") != sequence or row.get("predecessor_sha256") != predecessor_sha256:
            raise ValueError("journal sequence or predecessor")
        kind = row.get("kind")
        if kind == "reserve":
            if open_request is not None or expected_index >= len(expected):
                raise ValueError("journal reserve order")
            request_id, batch_size, iteration = expected[expected_index]
            if (
                set(row)
                != {
                    "sequence",
                    "kind",
                    "request_id",
                    "batch_size",
                    "iteration",
                    "reservation_cost_usd",
                    "predecessor_sha256",
                }
                or (row["request_id"], row["batch_size"], row["iteration"])
                != (request_id, batch_size, iteration)
                or _canonical_decimal(row["reservation_cost_usd"]) != RESERVATION
            ):
                raise ValueError("journal reserve shape")
            if settled + RESERVATION > BUDGET:
                raise ValueError("journal budget")
            open_request = expected[expected_index]
            continue
        if open_request is None or kind not in {"settle", "settled_over_limit", "failure"}:
            raise ValueError("journal terminal order")
        request_id, batch_size, iteration = open_request
        if (row.get("request_id"), row.get("batch_size"), row.get("iteration")) != open_request:
            raise ValueError("journal terminal identity")
        if kind == "failure":
            allowed = {
                "sequence",
                "kind",
                "request_id",
                "batch_size",
                "iteration",
                "category",
                "predecessor_sha256",
            }
            category = row.get("category")
            if category not in {
                "http_error",
                "timeout",
                "transport_error",
                "response_invalid",
                "process_interrupted",
            }:
                raise ValueError("journal failure category")
            if category == "http_error":
                allowed.add("http_status")
                if not isinstance(row.get("http_status"), int) or isinstance(
                    row["http_status"], bool
                ):
                    raise ValueError("journal HTTP status")
            elif "http_status" in row:
                raise ValueError("unexpected HTTP status")
            if set(row) != allowed:
                raise ValueError("journal failure shape")
        else:
            expected_fields = {
                "sequence",
                "kind",
                "request_id",
                "batch_size",
                "iteration",
                "computed_cost_usd",
                "sample",
                "predecessor_sha256",
            }
            if kind == "settled_over_limit":
                expected_fields.add("category")
                if row.get("category") != "reported_usage_over_limit":
                    raise ValueError("journal over-limit category")
            if set(row) != expected_fields:
                raise ValueError("journal settlement shape")
            cost = _canonical_decimal(row["computed_cost_usd"])
            sample = _validate_sample_shape(row["sample"])
            expected_question_keys = [f"r{index:03d}__routing" for index in range(batch_size)]
            if list(sample.answers) != expected_question_keys:
                raise ValueError("journal sample question keys")
            if cost != _canonical_decimal(sample.computed_cost_usd):
                raise ValueError("journal sample cost")
            if kind == "settle" and (
                sample.input_tokens > RESERVATION_TOKENS or cost > RESERVATION
            ):
                raise ValueError("journal settlement limit")
            if kind == "settled_over_limit" and not (
                sample.input_tokens > RESERVATION_TOKENS or cost > RESERVATION
            ):
                raise ValueError("journal over-limit transition")
            settled += cost
        open_request = None
        expected_index += 1
        if kind in {"failure", "settled_over_limit"} and sequence != len(rows):
            raise ValueError("dispatch continued after terminal uncertainty")


def create_claim(path: Path, predecessor_sha256: str) -> str:
    if path.exists() or path.is_symlink():
        raise ValueError("predecessor claim already exists")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    value = {"schema_version": "phase3d-claim.v1", "predecessor_sha256": predecessor_sha256}
    try:
        atomic_create(path, _canon(value) + b"\n")
    except FileExistsError as exc:
        raise ValueError("predecessor claim already exists") from exc
    os.chmod(path, 0o600)
    return digest_json(value)


def dispatch_native(
    payload: Mapping[str, Any],
    *,
    api_key: str,
    journal: Journal,
    batch_size: int,
    iteration: int,
    transport: Transport = _urllib_transport,
    clock: Any = time,
    expected: Sequence[str] | None = None,
) -> tuple[dict[str, Any], float]:
    if (
        not api_key
        or set(payload) != {"model", "state", "questions"}
        or payload["model"] != MODEL_ID
    ):
        raise ValueError("invalid native dispatch")
    request_id = f"native-b{batch_size:03d}-i{iteration:02d}"
    journal.reserve(request_id, batch_size, iteration)
    started = clock.perf_counter_ns()
    try:
        status, _, raw = transport(
            "POST",
            ENDPOINT,
            {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            _canon(payload),
            timeout=60.0,
        )
    except TimeoutError:
        journal.terminal(request_id, batch_size, iteration, "failure", category="timeout")
        raise
    except Exception as exc:
        journal.terminal(request_id, batch_size, iteration, "failure", category="transport_error")
        raise RuntimeError("native transport failed") from exc
    elapsed = (clock.perf_counter_ns() - started) / 1000000
    if status != 200:
        journal.terminal(
            request_id, batch_size, iteration, "failure", category="http_error", status=status
        )
        raise RuntimeError("native provider error")
    try:
        validated = validate_native_response(
            _load(raw), list(cast(Mapping[str, Any], payload["questions"]))
        )
        cost = Decimal(validated["usage"]["input_tokens"]) * INPUT_RATE / Decimal(1000000)
        sample = reduce_sample(
            validated,
            elapsed,
            list(expected) if expected is not None else [""] * len(validated["answers"]),
        )
        kind = (
            "settled_over_limit"
            if validated["usage"]["input_tokens"] > RESERVATION_TOKENS or cost > RESERVATION
            else "settle"
        )
        journal.terminal(request_id, batch_size, iteration, kind, cost=cost, sample=sample)
        if kind != "settle":
            raise RuntimeError("native usage exceeded reviewed limit")
        return validated, elapsed
    except RuntimeError:
        raise
    except Exception as exc:
        if not journal._closed(request_id):
            journal.terminal(
                request_id, batch_size, iteration, "failure", category="response_invalid"
            )
        raise ValueError("native response rejected") from exc


class PartialRunError(RuntimeError):
    def __init__(self, completed: int, workloads: Mapping[str, Any], cause: BaseException) -> None:
        super().__init__("native control unavailable")
        self.partial = {
            "status": "partial",
            "completed_call_count": completed,
            "workloads": dict(workloads),
            "resolved_model": MODEL_ID,
        }
        self.__cause__ = cause


def run_native_workloads(
    rows: Sequence[Mapping[str, Any]],
    *,
    api_key: str,
    journal: Journal,
    transport: Transport = _urllib_transport,
    clock: Any = time,
    iterations: int = ITERATIONS,
) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda r: str(r.get("family_id", "")))
    if len(ordered) < max(BATCH_SIZES) or iterations != ITERATIONS:
        raise ValueError("insufficient or non-frozen workload")
    workloads = {}
    completed = 0
    try:
        for batch_size in BATCH_ORDER:
            batch = ordered[:batch_size]
            payload = native_request(batch)
            expected = [str(r.get("candidate_label")) for r in batch]
            samples = []
            for iteration in range(iterations):
                validated, elapsed = dispatch_native(
                    payload,
                    api_key=api_key,
                    journal=journal,
                    batch_size=batch_size,
                    iteration=iteration,
                    transport=transport,
                    clock=clock,
                    expected=expected,
                )
                samples.append(reduce_sample(validated, elapsed, expected))
                completed += 1
            latencies = [s["latency_ms"] for s in samples]
            workloads[str(batch_size)] = {
                "batch_size": batch_size,
                "iterations": iterations,
                "input_id_digest": opaque_input_digest(batch),
                "request_structure_digest": payload_structure_digest(payload),
                "samples": samples,
                "samples_sha256": digest_json(samples),
                "summary": summarize(latencies),
                "per_item_summary": summarize([v / batch_size for v in latencies]),
                "throughput_items_per_second": throughput(batch_size, latencies),
                "accuracy": sum(s["accuracy"] for s in samples) / iterations,
                "computed_cost_usd": decimal_string(
                    sum((_dec(s["computed_cost_usd"]) for s in samples), Decimal(0))
                ),
            }
    except Exception as exc:
        raise PartialRunError(completed, workloads, exc) from exc
    return {
        "status": "complete",
        "completed_call_count": completed,
        "workloads": workloads,
        "resolved_model": MODEL_ID,
        "computed_cost_usd": decimal_string(journal.settled),
    }


def render_report(result: Mapping[str, Any]) -> str:
    raw = json.dumps(result, ensure_ascii=False, sort_keys=True)
    if any(x in raw for x in ("Authorization", "Bearer", "TYPESAFE_API_KEY", str(Path.home()))):
        raise ValueError("sensitive value in report")
    lines = [
        "# Saracura Phase 3D native TypeSafe control",
        "",
        "Synthetic control evidence only; no automation or production-readiness claim.",
        "",
        f"- Status: `{result['native']['status']}`",
        f"- Model: `{MODEL_ID}`",
        f"- Native computed cost: `${result['native']['computed_cost_usd']}`",
        "",
    ]
    if result["native"]["status"] == "complete":
        lines += [
            "## Three-way comparison",
            "",
            "| Batch | Local request p50/p95 ms | OpenRouter request p50/p95 ms | "
            "Native request p50/p95 ms |",
            "| ---: | ---: | ---: | ---: |",
        ]
        for key in ("1", "8", "20"):
            entry = result["comparison"][key]
            lines.append(
                "| {batch_size} | {local_request_p50_ms:.3f}/{local_request_p95_ms:.3f} | "
                "{openrouter_request_p50_ms:.3f}/{openrouter_request_p95_ms:.3f} | "
                "{native_request_p50_ms:.3f}/{native_request_p95_ms:.3f} |".format(**entry)
            )
            lines.append(
                "  - per-item p50/p95 local/OpenRouter/native: "
                "{local_per_item_p50_ms:.6f}/{local_per_item_p95_ms:.6f}, "
                "{openrouter_per_item_p50_ms:.6f}/{openrouter_per_item_p95_ms:.6f}, "
                "{native_per_item_p50_ms:.6f}/{native_per_item_p95_ms:.6f} ms; "
                "p50 ratios local-to-native/local-to-OpenRouter: "
                "{local_to_native_per_item_p50_ratio:.3f}/"
                "{local_to_openrouter_per_item_p50_ratio:.3f}; "
                "native-to-OpenRouter request p50 ratio: "
                "{native_to_openrouter_request_p50_ratio:.3f}.".format(**entry)
            )
            lines.append(
                "  - throughput local/OpenRouter/native: "
                "{local_throughput_items_per_second:.6f}/"
                "{openrouter_throughput_items_per_second:.6f}/"
                "{native_throughput_items_per_second:.6f} items/s; "
                "accuracy: {local_accuracy:.6f}/{openrouter_accuracy:.6f}/{native_accuracy:.6f}; "
                "cost OpenRouter/native: ${openrouter_provider_cost_usd}/"
                "${native_computed_cost_usd}.".format(**entry)
            )
        lines += ["", "## Native-only diagnostics", ""]
        for key in ("1", "8", "20"):
            diagnostic = result["native"]["diagnostics"][key]
            lines.append(
                f"- Batch {key}: input/output tokens {diagnostic['input_tokens']}/"
                f"{diagnostic['output_tokens']}; confidence mean/min/max "
                f"{diagnostic['confidence_mean']:.6f}/{diagnostic['confidence_min']:.6f}/"
                f"{diagnostic['confidence_max']:.6f}; probability-distribution diagnostics: "
                f"{len(diagnostic['probability_distribution_digests'])} sealed digests."
            )
        lines += [
            "",
            "Phase 3C retained no OpenRouter probability distributions; "
            "cross-gateway probability comparison is unavailable.",
            "",
        ]
    else:
        lines += [
            "## Partial result",
            "",
            f"- Completed calls: {result['native']['completed_call_count']}",
            f"- Journal state: `{result['native']['journal_state']}`",
            f"- Failure category: `{result['native']['failure_category']}`",
            "",
        ]
    lines += ["## Interpretation limits", ""]
    lines.extend(f"- {limitation}" for limitation in result["limitations"])
    return "\n".join(lines) + "\n"


def _preflight_inputs(predecessor: Path, packet: Path) -> tuple[Any, Path, list[dict[str, Any]]]:
    phase3c = validate_phase3c_artifacts(predecessor)
    if phase3c.jev.status != "complete" or phase3c.jev.workloads is None:
        raise ValueError("Phase 3C Jev predecessor is incomplete")
    packet_manifest = (
        packet if packet.name == "packet-manifest.json" else packet / "packet-manifest.json"
    )
    validate_packet_manifest(packet_manifest)
    if digest_bytes(packet_manifest.read_bytes()) != phase3c.provenance.packet_manifest_sha256:
        raise ValueError("packet digest mismatch")
    rows = sorted(_read_accepted(packet_manifest.parent), key=lambda row: str(row["family_id"]))
    for batch_size in BATCH_SIZES:
        input_digest = opaque_input_digest(rows[:batch_size])
        if (
            input_digest != phase3c.local.workloads[str(batch_size)].input_id_digest
            or input_digest != phase3c.jev.workloads[str(batch_size)].input_id_digest
        ):
            raise ValueError("Phase 3C workload identity mismatch")
    return phase3c, packet_manifest, rows


def _provenance(
    predecessor: Path, packet_manifest: Path, rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    return {
        "predecessor_artifact_manifest_sha256": digest_bytes(
            (predecessor / "artifact-manifest.json").read_bytes()
        ),
        "packet_manifest_sha256": digest_bytes(packet_manifest.read_bytes()),
        "phase3c_request_builder_sha256": digest_bytes(PHASE3C_SOURCE.read_bytes()),
        "phase3d_source_sha256": digest_bytes(Path(__file__).read_bytes()),
        "spec_sha256": digest_bytes(SPEC_PATH.read_bytes()),
        "request_structure_digests": {
            str(batch_size): payload_structure_digest(native_request(rows[:batch_size]))
            for batch_size in BATCH_SIZES
        },
    }


def _workload_from_samples(
    batch_size: int, samples: list[dict[str, Any]], rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    latencies = [float(sample["latency_ms"]) for sample in samples]
    expected = [str(row["candidate_label"]) for row in rows[:batch_size]]
    for sample in samples:
        model = _validate_sample_shape(sample)
        if list(model.answers) != [f"r{index:03d}__routing" for index in range(batch_size)]:
            raise ValueError("sample question keys")
        choices = [answer.choice for answer in model.answers.values()]
        if (
            model.accuracy
            != sum(a == b for a, b in zip(choices, expected, strict=True)) / batch_size
        ):
            raise ValueError("sample accuracy")
    return {
        "batch_size": batch_size,
        "iterations": ITERATIONS,
        "input_id_digest": opaque_input_digest(rows[:batch_size]),
        "request_structure_digest": payload_structure_digest(native_request(rows[:batch_size])),
        "samples": samples,
        "samples_sha256": digest_json(samples),
        "summary": summarize(latencies),
        "per_item_summary": summarize([value / batch_size for value in latencies]),
        "throughput_items_per_second": throughput(batch_size, latencies),
        "accuracy": sum(float(sample["accuracy"]) for sample in samples) / ITERATIONS,
        "input_tokens": sum(int(sample["input_tokens"]) for sample in samples),
        "output_tokens": sum(int(sample["output_tokens"]) for sample in samples),
        "computed_cost_usd": decimal_string(
            sum((_canonical_decimal(sample["computed_cost_usd"]) for sample in samples), Decimal(0))
        ),
    }


def _native_diagnostics(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    confidences = [
        float(answer["confidence"])
        for sample in samples
        for answer in cast(Mapping[str, Any], sample["answers"]).values()
    ]
    return {
        "input_tokens": sum(int(sample["input_tokens"]) for sample in samples),
        "output_tokens": sum(int(sample["output_tokens"]) for sample in samples),
        "confidence_mean": sum(confidences) / len(confidences),
        "confidence_min": min(confidences),
        "confidence_max": max(confidences),
        "probability_distribution_digests": [
            str(sample["probability_distribution_digest"]) for sample in samples
        ],
    }


def build_comparison(phase3c: Any, native_workloads: Mapping[str, Any]) -> dict[str, Any]:
    if phase3c.jev.workloads is None:
        raise ValueError("missing Phase 3C OpenRouter workloads")
    comparison: dict[str, Any] = {}
    for batch_size in BATCH_SIZES:
        key = str(batch_size)
        local = phase3c.local.workloads[key]
        openrouter = phase3c.jev.workloads[key]
        native = NativeWorkloadModel.model_validate(native_workloads[key])
        if native.per_item_summary.p50_ms <= 0 or openrouter.summary.p50_ms <= 0:
            raise ValueError("comparison zero latency")
        comparison[key] = {
            "batch_size": batch_size,
            "local_request_p50_ms": local.summary.p50_ms,
            "local_request_p95_ms": local.summary.p95_ms,
            "local_per_item_p50_ms": local.per_item_summary.p50_ms,
            "local_per_item_p95_ms": local.per_item_summary.p95_ms,
            "local_throughput_items_per_second": local.throughput_items_per_second,
            "local_accuracy": local.accuracy,
            "openrouter_request_p50_ms": openrouter.summary.p50_ms,
            "openrouter_request_p95_ms": openrouter.summary.p95_ms,
            "openrouter_per_item_p50_ms": openrouter.per_item_summary.p50_ms,
            "openrouter_per_item_p95_ms": openrouter.per_item_summary.p95_ms,
            "openrouter_throughput_items_per_second": openrouter.throughput_items_per_second,
            "openrouter_accuracy": openrouter.accuracy,
            "openrouter_provider_cost_usd": openrouter.cost_usd,
            "native_request_p50_ms": native.summary.p50_ms,
            "native_request_p95_ms": native.summary.p95_ms,
            "native_per_item_p50_ms": native.per_item_summary.p50_ms,
            "native_per_item_p95_ms": native.per_item_summary.p95_ms,
            "native_throughput_items_per_second": native.throughput_items_per_second,
            "native_accuracy": native.accuracy,
            "native_computed_cost_usd": native.computed_cost_usd,
            "local_to_native_per_item_p50_ratio": native.per_item_summary.p50_ms
            / local.per_item_summary.p50_ms,
            "local_to_openrouter_per_item_p50_ratio": openrouter.per_item_summary.p50_ms
            / local.per_item_summary.p50_ms,
            "native_to_openrouter_request_p50_ratio": native.summary.p50_ms
            / openrouter.summary.p50_ms,
        }
    return comparison


def _reconstruct_result(
    phase3c: Any, provenance: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], journal: Journal
) -> dict[str, Any]:
    validate_journal_rows(journal.rows, str(provenance["predecessor_artifact_manifest_sha256"]))
    terminals = [row for row in journal.rows if row["kind"] in {"settle", "settled_over_limit"}]
    failures = [row for row in journal.rows if row["kind"] == "failure"]
    open_reservation = any(
        row["kind"] == "reserve" and not journal._closed(row["request_id"]) for row in journal.rows
    )
    complete = len(terminals) == 30 and not failures and not open_reservation
    base = {
        "schema_version": "phase3d.v1",
        "sealed": True,
        "protocol": EXPECTED_PROTOCOL,
        "provenance": dict(provenance),
        "limitations": list(MANDATORY_LIMITATIONS),
    }
    if not complete:
        category: str | None = None
        state = "settled"
        if failures:
            category, state = str(failures[-1]["category"]), "failed"
        elif open_reservation:
            category, state = "process_interrupted", "open_reservation"
        elif any(row["kind"] == "settled_over_limit" for row in terminals):
            category, state = "reported_usage_over_limit", "settled_over_limit"
        partial = {
            "status": "partial",
            "completed_call_count": len(terminals),
            "completed_sample_digests": [digest_json(row["sample"]) for row in terminals],
            "resolved_models": sorted({row["sample"]["model"] for row in terminals}),
            "computed_cost_usd": decimal_string(journal.settled),
            "failure_category": category,
            "journal_state": state,
        }
        return PartialResultModel.model_validate({**base, "native": partial}).model_dump(
            mode="json"
        )
    grouped: dict[str, list[dict[str, Any]]] = {str(batch_size): [] for batch_size in BATCH_SIZES}
    for row in terminals:
        grouped[str(row["batch_size"])].append(dict(row["sample"]))
    workloads = {
        key: _workload_from_samples(int(key), samples, rows) for key, samples in grouped.items()
    }
    diagnostics = {key: _native_diagnostics(samples) for key, samples in grouped.items()}
    native = {
        "status": "complete",
        "completed_call_count": 30,
        "resolved_models": [MODEL_ID],
        "computed_cost_usd": decimal_string(journal.settled),
        "workloads": workloads,
        "diagnostics": diagnostics,
    }
    comparison = build_comparison(phase3c, workloads)
    return CompleteResultModel.model_validate(
        {**base, "native": native, "comparison": comparison}
    ).model_dump(mode="json")


def _mode(path: Path, expected: int) -> None:
    if path.is_symlink() or not path.is_file() or path.stat().st_mode & 0o777 != expected:
        raise ValueError("artifact type or permission")


def _atomic_match(path: Path, payload: bytes) -> None:
    if path.exists():
        _mode(path, 0o600)
        if path.read_bytes() != payload:
            raise ValueError("existing artifact differs")
        return
    atomic_create(path, payload)
    os.chmod(path, 0o600)


def _validate_claim(predecessor: Path, predecessor_sha256: str) -> str:
    claim = predecessor.parent / ".phase3d-claims" / f"{predecessor_sha256}.json"
    _mode(claim, 0o600)
    raw = claim.read_bytes()
    value = _load(raw)
    if (
        not isinstance(value, dict)
        or raw != _canon(value) + b"\n"
        or value != {"schema_version": "phase3d-claim.v1", "predecessor_sha256": predecessor_sha256}
    ):
        raise ValueError("claim binding")
    return digest_json(value)


def _manifest_bytes(output: Path, provenance: Mapping[str, Any], claim_sha256: str) -> bytes:
    sources = {name: digest_bytes((ROOT / name).read_bytes()) for name in SOURCE_ALLOWLIST}
    unsigned: dict[str, JsonValue] = {
        "schema_version": "phase3d-artifact-manifest.v1",
        "files": cast(
            JsonValue,
            {
                name: {
                    "bytes": len((output / name).read_bytes()),
                    "sha256": digest_bytes((output / name).read_bytes()),
                }
                for name in ("result.json", "report.md", "cost-journal.jsonl")
            },
        ),
        "sources": cast(JsonValue, sources),
        "predecessor_artifact_manifest_sha256": provenance["predecessor_artifact_manifest_sha256"],
        "packet_manifest_sha256": provenance["packet_manifest_sha256"],
        "claim_sha256": claim_sha256,
        "protocol": cast(JsonValue, EXPECTED_PROTOCOL),
    }
    manifest = {**unsigned, "self_sha256": digest_json(unsigned)}
    return _canon(manifest) + b"\n"


def finalize_phase3d(predecessor: Path, packet: Path, output: Path) -> dict[str, Any]:
    """Finalize a journal deterministically. This function never dispatches HTTP."""
    phase3c, packet_manifest, rows = _preflight_inputs(predecessor, packet)
    if output.is_symlink() or not output.is_dir() or output.stat().st_mode & 0o777 != 0o700:
        raise ValueError("output directory")
    names = {path.name for path in output.iterdir()}
    if not names <= {"cost-journal.jsonl", "result.json", "report.md", "artifact-manifest.json"}:
        raise ValueError("output allowlist")
    journal_path = output / "cost-journal.jsonl"
    _mode(journal_path, 0o600)
    provenance = _provenance(predecessor, packet_manifest, rows)
    claim_sha256 = _validate_claim(predecessor, provenance["predecessor_artifact_manifest_sha256"])
    journal = Journal(journal_path, provenance["predecessor_artifact_manifest_sha256"])
    result = _reconstruct_result(phase3c, provenance, rows, journal)
    result_bytes = _canon(result) + b"\n"
    report_bytes = render_report(result).encode("utf-8")
    _atomic_match(output / "result.json", result_bytes)
    _atomic_match(output / "report.md", report_bytes)
    _atomic_match(
        output / "artifact-manifest.json", _manifest_bytes(output, provenance, claim_sha256)
    )
    validate_phase3d_artifacts(output, predecessor, packet)
    return result


def validate_phase3d_artifacts(
    output: Path, predecessor: Path, packet: Path
) -> CompleteResultModel | PartialResultModel:
    artifact_names = {"result.json", "report.md", "cost-journal.jsonl", "artifact-manifest.json"}
    if output.is_symlink() or not output.is_dir() or output.stat().st_mode & 0o777 != 0o700:
        raise ValueError("artifact directory")
    paths = {path.name: path for path in output.iterdir()}
    if set(paths) != artifact_names:
        raise ValueError("artifact allowlist")
    for path in paths.values():
        _mode(path, 0o600)
    manifest_raw = paths["artifact-manifest.json"].read_bytes()
    manifest = _load(manifest_raw)
    if not isinstance(manifest, dict) or manifest_raw != _canon(manifest) + b"\n":
        raise ValueError("manifest canonical JSON")
    required = {
        "schema_version",
        "files",
        "sources",
        "predecessor_artifact_manifest_sha256",
        "packet_manifest_sha256",
        "claim_sha256",
        "protocol",
        "self_sha256",
    }
    if set(manifest) != required or manifest["schema_version"] != "phase3d-artifact-manifest.v1":
        raise ValueError("manifest shape")
    if not isinstance(manifest["files"], dict) or set(manifest["files"]) != {
        "result.json",
        "report.md",
        "cost-journal.jsonl",
    }:
        raise ValueError("manifest file allowlist")
    unsigned = dict(manifest)
    if (
        unsigned.pop("self_sha256") != digest_json(unsigned)
        or manifest["protocol"] != EXPECTED_PROTOCOL
    ):
        raise ValueError("manifest self digest or protocol")
    for name in ("result.json", "report.md", "cost-journal.jsonl"):
        raw = paths[name].read_bytes()
        if manifest["files"].get(name) != {"bytes": len(raw), "sha256": digest_bytes(raw)}:
            raise ValueError("manifest file digest")
    if manifest["sources"] != {
        name: digest_bytes((ROOT / name).read_bytes()) for name in SOURCE_ALLOWLIST
    }:
        raise ValueError("manifest source digest")
    result_raw = paths["result.json"].read_bytes()
    result_value = _load(result_raw)
    if not isinstance(result_value, dict) or result_raw != _canon(result_value) + b"\n":
        raise ValueError("result canonical JSON")
    if result_value.get("native", {}).get("status") == "complete":
        result: CompleteResultModel | PartialResultModel = CompleteResultModel.model_validate(
            result_value
        )
    else:
        result = PartialResultModel.model_validate(result_value)
    if (
        result.protocol.model_dump() != EXPECTED_PROTOCOL
        or result.provenance.predecessor_artifact_manifest_sha256
        != manifest["predecessor_artifact_manifest_sha256"]
        or result.provenance.packet_manifest_sha256 != manifest["packet_manifest_sha256"]
        or result.provenance.phase3d_source_sha256
        != manifest["sources"]["benchmarks/typesafe_native.py"]
        or result.provenance.phase3c_request_builder_sha256
        != manifest["sources"]["benchmarks/e2e_benchmark.py"]
        or result.provenance.spec_sha256
        != manifest["sources"]["docs/action/specs/phase3d-typesafe-native-control.md"]
        or tuple(result.limitations) != MANDATORY_LIMITATIONS
        or paths["report.md"].read_bytes()
        != render_report(result.model_dump(mode="json")).encode("utf-8")
    ):
        raise ValueError("result/manifest/report contradiction")
    journal = Journal(
        paths["cost-journal.jsonl"], result.provenance.predecessor_artifact_manifest_sha256
    )
    terminal_samples = [
        row["sample"] for row in journal.rows if row["kind"] in {"settle", "settled_over_limit"}
    ]
    if isinstance(result, CompleteResultModel):
        if set(result.native.workloads) != {"1", "8", "20"} or set(result.native.diagnostics) != {
            "1",
            "8",
            "20",
        }:
            raise ValueError("complete workload set")
        reconstructed_samples: list[dict[str, Any]] = []
        for key, workload in result.native.workloads.items():
            if (
                int(key) != workload.batch_size
                or len(workload.samples) != ITERATIONS
                or workload.samples_sha256
                != digest_json([sample.model_dump(mode="json") for sample in workload.samples])
            ):
                raise ValueError("workload sample binding")
            latencies = [sample.latency_ms for sample in workload.samples]
            samples = [sample.model_dump(mode="json") for sample in workload.samples]
            costs = sum(
                (_canonical_decimal(sample.computed_cost_usd) for sample in workload.samples),
                Decimal(0),
            )
            if (
                workload.summary.model_dump() != summarize(latencies)
                or workload.per_item_summary.model_dump()
                != summarize([value / workload.batch_size for value in latencies])
                or workload.throughput_items_per_second
                != throughput(workload.batch_size, latencies)
                or workload.accuracy
                != sum(sample.accuracy for sample in workload.samples) / ITERATIONS
                or workload.input_tokens != sum(sample.input_tokens for sample in workload.samples)
                or workload.output_tokens
                != sum(sample.output_tokens for sample in workload.samples)
                or _canonical_decimal(workload.computed_cost_usd) != costs
                or result.native.diagnostics[key]
                != NativeDiagnosticsModel.model_validate(_native_diagnostics(samples))
            ):
                raise ValueError("workload aggregate contradiction")
            reconstructed_samples.extend(samples)
        if (
            reconstructed_samples != terminal_samples
            or _canonical_decimal(result.native.computed_cost_usd) != journal.settled
            or result.native.resolved_models != [MODEL_ID]
        ):
            raise ValueError("complete journal contradiction")
    else:
        if (
            result.native.completed_call_count != len(terminal_samples)
            or result.native.completed_sample_digests
            != [digest_json(sample) for sample in terminal_samples]
            or _canonical_decimal(result.native.computed_cost_usd) != journal.settled
        ):
            raise ValueError("partial journal contradiction")
    phase3c, packet_manifest, rows = _preflight_inputs(predecessor, packet)
    expected = _reconstruct_result(
        phase3c, _provenance(predecessor, packet_manifest, rows), rows, journal
    )
    if result.model_dump(mode="json") != expected:
        raise ValueError("predecessor reconstruction")
    if (
        _validate_claim(predecessor, result.provenance.predecessor_artifact_manifest_sha256)
        != manifest["claim_sha256"]
    ):
        raise ValueError("claim digest")
    if isinstance(result, CompleteResultModel):
        if len(journal.rows) != 60:
            raise ValueError("complete journal count")
        return result
    if result.native.completed_call_count != len(
        [r for r in journal.rows if r["kind"] in {"settle", "settled_over_limit"}]
    ):
        raise ValueError("partial count")
    return result


def finalize_partial(predecessor: Path, packet: Path, output: Path) -> dict[str, Any]:
    return finalize_phase3d(predecessor, packet, output)


def _create_run_output(output: Path, predecessor: Path, predecessor_sha256: str) -> Journal:
    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    output.mkdir(mode=0o700, exist_ok=False)
    try:
        journal_path = output / "cost-journal.jsonl"
        atomic_create(journal_path, b"")
        os.chmod(journal_path, 0o600)
        create_claim(
            predecessor.parent / ".phase3d-claims" / f"{predecessor_sha256}.json",
            predecessor_sha256,
        )
    except ValueError:
        journal_path = output / "cost-journal.jsonl"
        if journal_path.exists() and journal_path.read_bytes() == b"":
            journal_path.unlink()
            output.rmdir()
        raise
    return Journal(journal_path, predecessor_sha256)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="phase3d.v1")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    finalize = commands.add_parser("finalize-partial")
    for command in (run, finalize):
        command.add_argument("--phase3c-artifact-dir", type=Path, required=True)
        command.add_argument("--packet", type=Path, required=True)
        command.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--budget-usd", required=True)
    run.add_argument("--allow-network", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "finalize-partial":
        finalize_phase3d(args.phase3c_artifact_dir, args.packet, args.output_dir)
        return 0
    if args.budget_usd != "0.25":
        raise SystemExit("approved budget must be exactly 0.25")
    if not args.allow_network:
        raise SystemExit("--allow-network is required for native dispatch")
    api_key = os.environ.get("TYPESAFE_API_KEY", "")
    if not api_key:
        raise SystemExit("TYPESAFE_API_KEY is required with --allow-network")
    _, packet_manifest, rows = _preflight_inputs(args.phase3c_artifact_dir, args.packet)
    predecessor_sha256 = digest_bytes(
        (args.phase3c_artifact_dir / "artifact-manifest.json").read_bytes()
    )
    journal = _create_run_output(args.output_dir, args.phase3c_artifact_dir, predecessor_sha256)
    with suppress(PartialRunError):
        run_native_workloads(rows, api_key=api_key, journal=journal)
    finalize_phase3d(args.phase3c_artifact_dir, packet_manifest, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
