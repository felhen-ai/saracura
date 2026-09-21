"""Closed, checkout-only Phase 3C end-to-end benchmark.

The module is intentionally importable in the default lightweight environment:
ML packages are imported only by :func:`load_local_components`.  The small
protocol objects below are also the seam used by the offline test suite.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import resource
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Literal, Protocol, Self, cast

from pydantic import BaseModel, ConfigDict, JsonValue, model_validator

from benchmarks.encoder_registry import get_candidate
from benchmarks.io import atomic_create
from benchmarks.synthetic_research import (
    LABELS,
    TAXONOMY,
    BudgetLedger,
    SyntheticError,
    _read_accepted,
    validate_packet_manifest,
    validate_training_manifest,
)
from saracura.serialization import canonical_json_bytes

JEV_MODEL = "typesafe/jev-1.13"
JEV_INPUT_PRICE = Decimal("0.042")

ROOT = Path(__file__).parents[1]
SPEC_PATH = ROOT / "docs/action/specs/phase3c-end-to-end-throughput-benchmark.md"
SOURCE_ALLOWLIST = (
    "benchmarks/e2e_benchmark.py",
    "benchmarks/synthetic_research.py",
    "benchmarks/encoder_registry.py",
    "docs/action/specs/phase3c-end-to-end-throughput-benchmark.md",
)
LOCAL_BATCH_SIZES = (1, 8, 20, 32, 64, 128)
LOCAL_ORDER = (1, 128, 8, 64, 20, 32)
JEV_BATCH_SIZES = (1, 8, 20)
JEV_ORDER = (1, 20, 8)
LOCAL_WARMUPS = 5
LOCAL_ITERATIONS = 30
JEV_ITERATIONS = 10
MAX_TOKENS = 128
TOTAL_BUDGET = Decimal("0.25")
JEV_BUDGET = Decimal("0.04")
PROVIDER_CONTROLS = {
    "zdr": True,
    "data_collection": "deny",
    "enforce_distillable_text": True,
    "allow_fallbacks": False,
    "require_parameters": True,
    "max_price": {"prompt": "0.042", "completion": "0"},
}
MANDATORY_LIMITATIONS = (
    "local warm-path excludes one-time model load",
    "Jev includes WAN and provider routing when enabled",
    "provider cold/warm state is not controllable",
    "local and remote measured-iteration counts differ",
    "synthetic PT-BR routing is narrow and cannot establish general quality",
    "Jev accuracy is contextual diagnostic evidence, not a training or selection signal",
    "remote p95 over ten samples is the maximum nearest-rank value and is descriptive only",
    "throughput on one M4 Pro does not predict every deployment target",
    "accuracy is an in-sample diagnostic on accepted synthetic packet rows",
    "fresh-process load includes lazy imports and snapshot verification",
)


class ClosedModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False, frozen=True)


class SummaryModel(ClosedModel):
    count: int
    min_ms: float
    p50_ms: float
    p95_ms: float
    max_ms: float


class BoundariesModel(ClosedModel):
    start: int
    tokenized: int
    device: int
    transferred: int
    stop: int


class ComponentsModel(ClosedModel):
    tokenization_ns: int
    cpu_to_mps_encoder_pooling_ns: int
    mps_to_cpu_ns: int
    cpu_head_ns: int
    total_ns: int


class LocalSampleModel(ClosedModel):
    boundaries_ns: BoundariesModel
    components_ns: ComponentsModel
    prediction_digest: str
    accuracy: float
    total_ms: float


class LocalWorkloadModel(ClosedModel):
    batch_size: int
    warmups: int
    iterations: int
    input_id_digest: str
    prediction_digest: str
    token_lengths: list[int]
    token_lengths_digest: str
    truncated: Literal[False]
    accuracy: float
    samples: list[LocalSampleModel]
    samples_sha256: str
    summary: SummaryModel
    per_item_summary: SummaryModel
    throughput_items_per_second: float


class LocalResultModel(ClosedModel):
    fresh_process_load_ms: float
    page_cache_sensitive: Literal[True]
    load_includes_import_and_snapshot_verification: Literal[True]
    rss_metric: Literal["ru_maxrss_bytes"]
    rss_before_load_bytes: int
    rss_after_load_bytes: int
    rss_after_workloads_bytes: int
    workloads: dict[str, LocalWorkloadModel]


class JevUsageModel(ClosedModel):
    cost: str
    input_tokens: int
    output_tokens: int


class JevSampleModel(ClosedModel):
    latency_ms: float
    answer_digest: str
    accuracy: float
    model: str
    provider: Literal["TypeSafe"]
    usage: JevUsageModel


class JevWorkloadModel(ClosedModel):
    batch_size: int
    iterations: int
    input_id_digest: str
    samples: list[JevSampleModel]
    samples_sha256: str
    summary: SummaryModel
    per_item_summary: SummaryModel
    throughput_items_per_second: float
    accuracy: float
    cost_usd: str


class JevResultModel(ClosedModel):
    status: Literal["complete", "external_control_unavailable", "not_requested"]
    completed_call_count: int
    completed_answer_digests: list[str] | None = None
    resolved_models: list[str] | None = None
    provider: Literal["TypeSafe"] | None = None
    cost_usd: str | None = None
    workloads: dict[str, JevWorkloadModel] | None = None
    key_usage_pre_usd: str | None = None
    key_usage_post_usd: str | None = None

    @model_validator(mode="after")
    def validate_status_shape(self) -> Self:
        if self.status == "complete" and (
            self.workloads is None
            or self.provider != "TypeSafe"
            or self.completed_call_count != JEV_ITERATIONS * len(JEV_BATCH_SIZES)
        ):
            raise ValueError("complete Jev result shape")
        if self.status != "complete" and self.workloads is not None:
            raise ValueError("partial Jev result cannot contain aggregates")
        return self


class ComparisonEntryModel(ClosedModel):
    batch_size: int
    local_per_item_p50_ms: float
    jev_per_item_p50_ms: float
    local_speedup_over_jev: float


class ProtocolModel(ClosedModel):
    local_batch_sizes: list[int]
    local_order: list[int]
    local_warmups: int
    local_iterations: int
    max_tokens: int
    device: Literal["mps"]
    jev_batch_sizes: list[int]
    jev_order: list[int]
    jev_iterations: int


class ProvenanceModel(ClosedModel):
    packet_manifest_sha256: str
    training_manifest_sha256: str
    checkpoint_sha256: str
    prior_cost_ledger_sha256: str
    claim_sha256: str | None
    benchmark_source_sha256: str
    encoder_revision: str


class EnvironmentModel(ClosedModel):
    python: str
    platform: str
    processor: str
    torch: str
    transformers: str
    safetensors: str


class BudgetModel(ClosedModel):
    approved_total_usd: str
    jev_stage_limit_usd: str
    prior_total_usd: str
    phase3c_jev_usd: str


class Phase3CResultModel(ClosedModel):
    schema_version: Literal["phase3c.v1"]
    sealed: Literal[True]
    protocol: ProtocolModel
    provenance: ProvenanceModel
    environment: EnvironmentModel
    local: LocalResultModel
    jev: JevResultModel
    comparison: dict[str, ComparisonEntryModel] | None
    budget: BudgetModel
    limitations: list[str]


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest_json(value: Any) -> str:
    return digest_bytes(canonical_json_bytes(value))


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float, Decimal))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def nearest_rank(values: Sequence[float], fraction: float) -> float:
    """Return the nearest-rank percentile, including the specified max p95."""
    if not values or fraction <= 0 or fraction > 1 or any(not _finite(v) or v < 0 for v in values):
        raise ValueError("invalid percentile input")
    ordered = sorted(float(v) for v in values)
    rank = max(1, math.ceil(fraction * len(ordered)))
    return ordered[rank - 1]


def summarize(values: Sequence[float]) -> dict[str, float | int]:
    if not values or any(not _finite(v) or v < 0 for v in values):
        raise ValueError("invalid samples")
    return {
        "count": len(values),
        "min_ms": min(values),
        "p50_ms": nearest_rank(values, 0.5),
        "p95_ms": nearest_rank(values, 0.95),
        "max_ms": max(values),
    }


def throughput(items_per_sample: int, samples_ms: Sequence[float]) -> float:
    if items_per_sample <= 0 or not samples_ms or any(v <= 0 or not _finite(v) for v in samples_ms):
        raise ValueError("invalid throughput input")
    return items_per_sample * len(samples_ms) / (sum(samples_ms) / 1000.0)


def component_timings_ns(
    start: int, tokenized: int, device: int, transferred: int, stop: int
) -> dict[str, int]:
    values = {
        "tokenization_ns": tokenized - start,
        "cpu_to_mps_encoder_pooling_ns": device - tokenized,
        "mps_to_cpu_ns": transferred - device,
        "cpu_head_ns": stop - transferred,
        "total_ns": stop - start,
    }
    if any(value < 0 for value in values.values()):
        raise ValueError("negative timing component")
    if sum(values[key] for key in values if key != "total_ns") != values["total_ns"]:
        raise ValueError("timing components do not sum to total")
    return values


class Clock(Protocol):
    def perf_counter_ns(self) -> int: ...


class LocalComponents(Protocol):
    def synchronize(self) -> None: ...
    def tokenize(self, texts: Sequence[str]) -> Any: ...
    def infer_and_pool(self, encoded: Any) -> Any: ...
    def to_cpu(self, pooled: Any) -> Any: ...
    def predict(self, pooled: Any) -> Sequence[int]: ...
    def token_lengths(self, texts: Sequence[str]) -> Sequence[int]: ...


class Transport(Protocol):
    def __call__(
        self, method: str, url: str, headers: Mapping[str, str], body: bytes, *, timeout: float
    ) -> tuple[int, Mapping[str, str], bytes]: ...


def measure_local_sample(
    components: LocalComponents, texts: Sequence[str], clock: Clock
) -> dict[str, Any]:
    """Execute the frozen event sequence and retain recomputable boundaries."""
    components.synchronize()
    start = clock.perf_counter_ns()
    encoded = components.tokenize(texts)
    tokenized = clock.perf_counter_ns()
    pooled = components.infer_and_pool(encoded)
    components.synchronize()
    device = clock.perf_counter_ns()
    cpu = components.to_cpu(pooled)
    components.synchronize()
    transferred = clock.perf_counter_ns()
    predictions = list(components.predict(cpu))
    stop = clock.perf_counter_ns()
    timings = component_timings_ns(start, tokenized, device, transferred, stop)
    if len(predictions) != len(texts) or any(
        not isinstance(v, int) or v not in range(5) for v in predictions
    ):
        raise ValueError("invalid local predictions")
    return {
        "boundaries_ns": {
            "start": start,
            "tokenized": tokenized,
            "device": device,
            "transferred": transferred,
            "stop": stop,
        },
        "components_ns": timings,
        "prediction_digest": prediction_digest(predictions),
        "prediction_labels": predictions,
        "total_ms": timings["total_ns"] / 1_000_000,
    }


def prediction_digest(predictions: Sequence[int]) -> str:
    if any(not isinstance(v, int) or v not in range(5) for v in predictions):
        raise ValueError("prediction labels")
    return digest_json(list(predictions))


def run_local_workloads(
    rows: Sequence[Mapping[str, Any]],
    components: LocalComponents,
    *,
    clock: Clock = time,
    warmups: int = LOCAL_WARMUPS,
    iterations: int = LOCAL_ITERATIONS,
) -> dict[str, Any]:
    """Run the fixed local order and return only opaque, recomputable evidence."""
    ordered = sorted(rows, key=lambda row: str(row.get("family_id", "")))
    if len(ordered) < max(LOCAL_BATCH_SIZES) or warmups < 0 or iterations <= 0:
        raise ValueError("insufficient local workload")
    workloads: dict[str, Any] = {}
    for batch_size in LOCAL_ORDER:
        batch = ordered[:batch_size]
        texts = [str(row["text"]) for row in batch]
        lengths = list(components.token_lengths(texts))
        if len(lengths) != batch_size or any(
            not isinstance(value, int) or value <= 0 or value > MAX_TOKENS for value in lengths
        ):
            raise ValueError("input exceeds the untruncated token limit")
        samples: list[dict[str, Any]] = []
        for _ in range(warmups):
            measure_local_sample(components, texts, clock)
        for _ in range(iterations):
            samples.append(measure_local_sample(components, texts, clock))
        digests = {str(sample["prediction_digest"]) for sample in samples}
        if len(digests) != 1:
            raise ValueError("prediction drift")
        expected = [str(row.get("candidate_label")) for row in batch]
        first_predictions = [int(value) for value in samples[0]["prediction_labels"]]
        accuracy = (
            sum(
                LABELS[index] == label
                for index, label in zip(first_predictions, expected, strict=True)
            )
            / batch_size
        )
        public_samples = []
        for sample in samples:
            public = dict(sample)
            public.pop("prediction_labels")
            public["accuracy"] = accuracy
            public_samples.append(public)
        total_ms = [float(sample["total_ms"]) for sample in samples]
        workloads[str(batch_size)] = {
            "batch_size": batch_size,
            "warmups": warmups,
            "iterations": iterations,
            "input_id_digest": opaque_input_digest(batch),
            "prediction_digest": next(iter(digests)),
            "token_lengths": lengths,
            "token_lengths_digest": digest_json(lengths),
            "truncated": False,
            "accuracy": accuracy,
            "samples": public_samples,
            "samples_sha256": digest_json(public_samples),
            "summary": summarize(total_ms),
            "per_item_summary": summarize([value / batch_size for value in total_ms]),
            "throughput_items_per_second": throughput(batch_size, total_ms),
        }
    return workloads


def opaque_input_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    ids = [row.get("family_id") for row in rows]
    if any(not isinstance(value, str) or not value for value in ids):
        raise ValueError("input identity")
    return digest_json(ids)


def build_jev_request(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(rows) not in JEV_BATCH_SIZES:
        raise ValueError("unsupported Jev batch size")
    records: list[dict[str, Any]] = []
    questions: dict[str, Any] = {}
    criteria = {label: TAXONOMY[label] for label in LABELS}
    for index, row in enumerate(rows):
        text = row.get("text")
        if not isinstance(text, str) or not text:
            raise ValueError("invalid synthetic text")
        record_id = f"r{index:03d}"
        records.append({"id": record_id, "record": {"text": text}})
        questions[f"{record_id}__routing"] = {
            "type": "choice",
            "instructions": f"Classifique exatamente o registro {record_id} em uma categoria.",
            "criteria": criteria,
        }
    return {
        "model": JEV_MODEL,
        "state": {"records": records},
        "questions": questions,
        "provider": PROVIDER_CONTROLS,
    }


def validate_jev_response(
    response: Mapping[str, Any], question_keys: Sequence[str]
) -> dict[str, Any]:
    required = {"id", "model", "provider", "answers", "usage"}
    if set(response) != required:
        raise ValueError("Jev response shape")
    if (
        not isinstance(response["id"], str)
        or not response["id"]
        or not isinstance(response["model"], str)
        or not response["model"].startswith(f"{JEV_MODEL}-")
        or response["provider"] != "TypeSafe"
    ):
        raise ValueError("Jev response identity")
    answers = response["answers"]
    if not isinstance(answers, Mapping) or set(answers) != set(question_keys):
        raise ValueError("Jev answer keys")
    reduced: dict[str, str] = {}
    for key in question_keys:
        answer = answers[key]
        if (
            not isinstance(answer, Mapping)
            or not {"type", "choice"}.issubset(answer)
            or not set(answer).issubset({"type", "choice", "confidence", "probabilities"})
            or answer["type"] != "choice"
            or answer["choice"] not in LABELS
        ):
            raise ValueError("Jev choice answer")
        reduced[key] = str(answer["choice"])
    usage = response["usage"]
    if not isinstance(usage, Mapping) or set(usage) != {"cost", "input_tokens", "output_tokens"}:
        raise ValueError("Jev usage shape")
    cost = usage["cost"]
    if (
        not _finite(cost)
        or float(cost) < 0
        or any(
            not isinstance(usage[key], int) or isinstance(usage[key], bool) or usage[key] < 0
            for key in ("input_tokens", "output_tokens")
        )
    ):
        raise ValueError("Jev usage values")
    return {
        "answer_digest": digest_json(reduced),
        "answers": reduced,
        "model": response["model"],
        "provider": response["provider"],
        "usage": {
            "cost": str(Decimal(str(cost))),
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
        },
    }


def _decimal(value: Any) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValueError("invalid decimal") from exc
    if not result.is_finite() or result < 0:
        raise ValueError("invalid decimal")
    return result


class CostJournal:
    """Durable Decimal journal; unsettled reservations permanently stop dispatch."""

    def __init__(
        self,
        path: Path,
        *,
        predecessor_sha256: str,
        stage_limit: Decimal = JEV_BUDGET,
        total_limit: Decimal = TOTAL_BUDGET,
        initial_stage_spent: Decimal = Decimal(0),
        initial_total_spent: Decimal = Decimal(0),
    ) -> None:
        self.path, self.predecessor_sha256 = path, predecessor_sha256
        self.stage_limit, self.total_limit = stage_limit, total_limit
        self.initial_stage_spent = initial_stage_spent
        self.initial_total_spent = initial_total_spent
        self.uncertain = False
        self._entries = self._read()
        if any(
            entry["kind"] == "reserve"
            for entry in self._entries
            if entry.get("settled") is not True
        ):
            self.uncertain = True

    def _read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows: list[dict[str, Any]] = []
        for line in self.path.read_bytes().splitlines():
            value = json.loads(line)
            if (
                not isinstance(value, dict)
                or value.get("predecessor_sha256") != self.predecessor_sha256
            ):
                raise ValueError("cost journal provenance")
            rows.append(value)
        settled_ids = {row.get("request_id") for row in rows if row.get("kind") == "settle"}
        for row in rows:
            if row.get("kind") == "reserve" and row.get("request_id") in settled_ids:
                row["settled"] = True
        return rows

    @property
    def settled(self) -> Decimal:
        return self.initial_total_spent + sum(
            (_decimal(row["cost"]) for row in self._entries if row["kind"] == "settle"), Decimal(0)
        )

    @property
    def stage_settled(self) -> Decimal:
        return self.initial_stage_spent + sum(
            (_decimal(row["cost"]) for row in self._entries if row["kind"] == "settle"),
            Decimal(0),
        )

    def _append(self, value: Mapping[str, Any]) -> None:
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with self.path.open("ab") as handle:
            handle.write(canonical_json_bytes(cast(JsonValue, dict(value))) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(self.path, 0o600)
        self._entries.append(dict(value))

    def reserve(self, request_id: str, worst_case: Decimal) -> None:
        if any(
            row.get("kind") == "reserve" and row.get("settled") is not True for row in self._entries
        ):
            self.uncertain = True
        if (
            self.uncertain
            or self.stage_settled + worst_case > self.stage_limit
            or self.settled + worst_case > self.total_limit
        ):
            raise ValueError("budget unavailable")
        self._append(
            {
                "kind": "reserve",
                "request_id": request_id,
                "cost": str(worst_case),
                "predecessor_sha256": self.predecessor_sha256,
                "settled": False,
            }
        )

    def settle(self, request_id: str, cost: Decimal, *, status: str = "complete") -> None:
        if self.uncertain or status not in {"complete", "billed_failure"} or cost < 0:
            self.uncertain = True
            raise ValueError("journal is uncertain")
        reservations = [
            row
            for row in self._entries
            if row.get("kind") == "reserve" and row.get("request_id") == request_id
        ]
        if (
            not reservations
            or any(row.get("settled") is True for row in reservations)
            or cost > _decimal(reservations[-1]["cost"])
        ):
            self.uncertain = True
            raise ValueError("invalid settlement")
        self._append(
            {
                "kind": "settle",
                "request_id": request_id,
                "cost": str(cost),
                "status": status,
                "predecessor_sha256": self.predecessor_sha256,
            }
        )
        for row in self._entries:
            if row.get("kind") == "reserve" and row.get("request_id") == request_id:
                row["settled"] = True

    def uncertain_outcome(self, request_id: str) -> None:
        self._append(
            {
                "kind": "uncertain",
                "request_id": request_id,
                "predecessor_sha256": self.predecessor_sha256,
            }
        )
        self.uncertain = True


def create_claim(path: Path, predecessor_sha256: str) -> str:
    if path.exists() or path.is_symlink():
        raise ValueError("predecessor claim already exists")
    value = {"schema_version": "phase3c-claim.v1", "predecessor_sha256": predecessor_sha256}
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    try:
        atomic_create(path, canonical_json_bytes(cast(JsonValue, value)) + b"\n")
    except FileExistsError as exc:
        raise ValueError("predecessor claim already exists") from exc
    os.chmod(path, 0o600)
    return digest_json(value)


def validate_predecessor_binding(
    ledger_path: Path, *, packet_sha256: str, training_sha256: str
) -> None:
    if ledger_path.name != "jev-cost-ledger.json" or ledger_path.is_symlink():
        raise ValueError("Phase 3B predecessor ledger path")
    evaluation_path = ledger_path.with_name("evaluation-manifest.json")
    if evaluation_path.is_symlink() or not evaluation_path.is_file():
        raise ValueError("Phase 3B evaluation manifest missing")
    evaluation = json.loads(evaluation_path.read_bytes())
    if (
        not isinstance(evaluation, dict)
        or set(evaluation)
        != {
            "schema_version",
            "sealed",
            "packet_manifest_sha256",
            "training_manifest_sha256",
            "local",
            "jev",
            "jev_usage_snapshots",
        }
        or evaluation["schema_version"] != "synthetic-evaluation.v1"
        or evaluation["sealed"] is not True
        or evaluation["packet_manifest_sha256"] != packet_sha256
        or evaluation["training_manifest_sha256"] != training_sha256
        or evaluation.get("jev", {}).get("status") != "complete"
    ):
        raise ValueError("Phase 3B predecessor is not bound to these inputs")
    ledger_value = json.loads(ledger_path.read_bytes())
    ledger = BudgetLedger.from_json(ledger_value)
    if ledger_value.get("final") is not True:
        raise ValueError("Phase 3B predecessor ledger is not final")
    jev_digests = evaluation["jev"].get("digests")
    jev_usage = evaluation["jev"].get("usage", {}).get("cost")
    jev_entries = [entry for entry in ledger.entries if entry["stage"] == "jev"]
    if (
        not isinstance(jev_digests, list)
        or not jev_digests
        or len(jev_entries) < len(jev_digests)
        or not _finite(jev_usage)
        or abs(
            sum(
                (Decimal(str(entry["cost"])) for entry in jev_entries[-len(jev_digests) :]),
                Decimal(0),
            )
            - Decimal(str(jev_usage))
        )
        > Decimal("1e-15")
    ):
        raise ValueError("Phase 3B evaluation does not bind the ledger tail")


def sanitized_report(result: Mapping[str, Any]) -> str:
    raw = json.dumps(result, ensure_ascii=False)
    for marker in ("Authorization", "Bearer", "OPENROUTER_API_KEY", str(Path.home())):
        if marker in raw:
            raise ValueError("sensitive value in report input")
    lines = [
        "# Saracura Phase 3C throughput benchmark",
        "",
        "Performance evidence only; no automation or quality claim.",
        "",
    ]
    for section in (
        "protocol",
        "provenance",
        "environment",
        "local",
        "jev",
        "comparison",
        "budget",
        "limitations",
    ):
        if section in result:
            lines.extend(
                [
                    f"## {section}",
                    "",
                    "```json",
                    json.dumps(result[section], ensure_ascii=False, sort_keys=True),
                    "```",
                    "",
                ]
            )
    return "\n".join(lines)


def write_artifact_manifest(
    output_dir: Path,
    result: bytes,
    report: bytes,
    journal: bytes,
    *,
    packet_sha256: str,
    training_sha256: str,
    predecessor_sha256: str,
    claim_sha256: str | None,
) -> bytes:
    files = {
        name: {"bytes": len(data), "sha256": digest_bytes(data)}
        for name, data in (
            ("result.json", result),
            ("report.md", report),
            ("cost-journal.jsonl", journal),
        )
    }
    sources = {name: digest_bytes((ROOT / name).read_bytes()) for name in SOURCE_ALLOWLIST}
    unsigned: dict[str, JsonValue] = {
        "schema_version": "phase3c-artifact-manifest.v1",
        "files": cast(JsonValue, files),
        "sources": cast(JsonValue, sources),
        "packet_manifest_sha256": packet_sha256,
        "training_manifest_sha256": training_sha256,
        "predecessor_ledger_sha256": predecessor_sha256,
        "claim_sha256": claim_sha256,
    }
    manifest = {**unsigned, "self_sha256": digest_json(unsigned)}
    target = output_dir / "artifact-manifest.json"
    atomic_create(target, canonical_json_bytes(cast(JsonValue, manifest)) + b"\n")
    os.chmod(target, 0o600)
    return canonical_json_bytes(manifest) + b"\n"


def write_phase3c_artifacts(
    output_dir: Path,
    result: Mapping[str, Any],
    journal: bytes,
    *,
    packet_sha256: str,
    training_sha256: str,
    predecessor_sha256: str,
    claim_sha256: str | None,
) -> None:
    """Create the exact four-file result directory without replacing anything."""
    validated_result = Phase3CResultModel.model_validate(result).model_dump(mode="json")
    if output_dir.exists():
        if output_dir.is_symlink() or not output_dir.is_dir():
            raise ValueError("invalid output directory")
        if claim_sha256 is None:
            raise ValueError("existing local output directory")
        if {item.name for item in output_dir.iterdir()} - {"cost-journal.jsonl"}:
            raise ValueError("output directory is not an active Phase 3C run")
    else:
        output_dir.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        output_dir.mkdir(mode=0o700, parents=False, exist_ok=False)
    os.chmod(output_dir, 0o700)
    result_bytes = canonical_json_bytes(cast(JsonValue, validated_result)) + b"\n"
    report_bytes = sanitized_report(validated_result).encode("utf-8")
    for name, payload in (("result.json", result_bytes), ("report.md", report_bytes)):
        atomic_create(output_dir / name, payload)
        os.chmod(output_dir / name, 0o600)
    journal_path = output_dir / "cost-journal.jsonl"
    if journal_path.exists():
        if journal_path.read_bytes() != journal:
            raise ValueError("cost journal changed during finalization")
    else:
        atomic_create(journal_path, journal)
        os.chmod(journal_path, 0o600)
    write_artifact_manifest(
        output_dir,
        result_bytes,
        report_bytes,
        journal,
        packet_sha256=packet_sha256,
        training_sha256=training_sha256,
        predecessor_sha256=predecessor_sha256,
        claim_sha256=claim_sha256,
    )
    validate_phase3c_artifacts(output_dir)


def validate_phase3c_artifacts(output_dir: Path) -> Phase3CResultModel:
    expected = {"result.json", "report.md", "cost-journal.jsonl", "artifact-manifest.json"}
    if (
        output_dir.is_symlink()
        or not output_dir.is_dir()
        or output_dir.stat().st_mode & 0o777 != 0o700
    ):
        raise ValueError("artifact directory mode")
    paths = {item.name: item for item in output_dir.iterdir()}
    if set(paths) != expected:
        raise ValueError("artifact allowlist")
    for path in paths.values():
        if path.is_symlink() or not path.is_file() or path.stat().st_mode & 0o777 != 0o600:
            raise ValueError("artifact file type or mode")
    manifest = json.loads(paths["artifact-manifest.json"].read_bytes())
    if not isinstance(manifest, dict) or set(manifest) != {
        "schema_version",
        "files",
        "sources",
        "packet_manifest_sha256",
        "training_manifest_sha256",
        "predecessor_ledger_sha256",
        "claim_sha256",
        "self_sha256",
    }:
        raise ValueError("artifact manifest shape")
    unsigned = dict(manifest)
    self_digest = unsigned.pop("self_sha256")
    if manifest["schema_version"] != "phase3c-artifact-manifest.v1" or self_digest != digest_json(
        unsigned
    ):
        raise ValueError("artifact manifest self digest")
    if set(manifest["files"]) != expected - {"artifact-manifest.json"}:
        raise ValueError("artifact file ledger")
    for name, metadata in manifest["files"].items():
        raw = paths[name].read_bytes()
        if metadata != {"bytes": len(raw), "sha256": digest_bytes(raw)}:
            raise ValueError("artifact digest drift")
    if set(manifest["sources"]) != set(SOURCE_ALLOWLIST):
        raise ValueError("source allowlist")
    for name, expected_digest in manifest["sources"].items():
        if expected_digest != digest_bytes((ROOT / name).read_bytes()):
            raise ValueError("source digest drift")
    result = Phase3CResultModel.model_validate_json(paths["result.json"].read_bytes())
    expected_protocol = {
        "local_batch_sizes": list(LOCAL_BATCH_SIZES),
        "local_order": list(LOCAL_ORDER),
        "local_warmups": LOCAL_WARMUPS,
        "local_iterations": LOCAL_ITERATIONS,
        "max_tokens": MAX_TOKENS,
        "device": "mps",
        "jev_batch_sizes": list(JEV_BATCH_SIZES),
        "jev_order": list(JEV_ORDER),
        "jev_iterations": JEV_ITERATIONS,
    }
    if result.protocol.model_dump() != expected_protocol:
        raise ValueError("protocol drift")
    if (
        result.provenance.packet_manifest_sha256 != manifest["packet_manifest_sha256"]
        or result.provenance.training_manifest_sha256 != manifest["training_manifest_sha256"]
        or result.provenance.prior_cost_ledger_sha256 != manifest["predecessor_ledger_sha256"]
        or result.provenance.claim_sha256 != manifest["claim_sha256"]
        or result.provenance.benchmark_source_sha256
        != manifest["sources"]["benchmarks/e2e_benchmark.py"]
    ):
        raise ValueError("artifact provenance contradiction")
    if set(result.local.workloads) != {str(value) for value in LOCAL_BATCH_SIZES}:
        raise ValueError("local workload set")
    if tuple(result.limitations) != MANDATORY_LIMITATIONS:
        raise ValueError("mandatory limitations drift")
    for key, local_workload in result.local.workloads.items():
        if (
            int(key) != local_workload.batch_size
            or local_workload.warmups != LOCAL_WARMUPS
            or local_workload.iterations != LOCAL_ITERATIONS
            or len(local_workload.samples) != LOCAL_ITERATIONS
            or len(local_workload.token_lengths) != local_workload.batch_size
            or any(value <= 0 or value > MAX_TOKENS for value in local_workload.token_lengths)
            or digest_json(local_workload.token_lengths) != local_workload.token_lengths_digest
            or digest_json([sample.model_dump(mode="json") for sample in local_workload.samples])
            != local_workload.samples_sha256
        ):
            raise ValueError("local sample binding")
        for sample in local_workload.samples:
            boundaries = sample.boundaries_ns
            components = component_timings_ns(
                boundaries.start,
                boundaries.tokenized,
                boundaries.device,
                boundaries.transferred,
                boundaries.stop,
            )
            if (
                components != sample.components_ns.model_dump()
                or not math.isclose(
                    sample.total_ms,
                    components["total_ns"] / 1_000_000,
                    rel_tol=0,
                    abs_tol=1e-12,
                )
                or sample.prediction_digest != local_workload.prediction_digest
                or sample.accuracy != local_workload.accuracy
            ):
                raise ValueError("local timing or accuracy contradiction")
        totals = [sample.total_ms for sample in local_workload.samples]
        if (
            summarize(totals) != local_workload.summary.model_dump()
            or summarize([value / local_workload.batch_size for value in totals])
            != local_workload.per_item_summary.model_dump()
            or throughput(local_workload.batch_size, totals)
            != local_workload.throughput_items_per_second
        ):
            raise ValueError("local summary contradiction")
    if result.jev.workloads is not None:
        if set(result.jev.workloads) != {str(value) for value in JEV_BATCH_SIZES}:
            raise ValueError("Jev workload set")
        all_models: set[str] = set()
        jev_cost = Decimal(0)
        for key, jev_workload in result.jev.workloads.items():
            if (
                int(key) != jev_workload.batch_size
                or jev_workload.iterations != JEV_ITERATIONS
                or len(jev_workload.samples) != JEV_ITERATIONS
                or digest_json([sample.model_dump(mode="json") for sample in jev_workload.samples])
                != jev_workload.samples_sha256
            ):
                raise ValueError("Jev sample binding")
            totals = [sample.latency_ms for sample in jev_workload.samples]
            workload_cost = sum(
                (Decimal(sample.usage.cost) for sample in jev_workload.samples), Decimal(0)
            )
            if (
                summarize(totals) != jev_workload.summary.model_dump()
                or summarize([value / jev_workload.batch_size for value in totals])
                != jev_workload.per_item_summary.model_dump()
                or throughput(jev_workload.batch_size, totals)
                != jev_workload.throughput_items_per_second
                or sum(sample.accuracy for sample in jev_workload.samples) / JEV_ITERATIONS
                != jev_workload.accuracy
                or workload_cost != Decimal(jev_workload.cost_usd)
            ):
                raise ValueError("Jev summary contradiction")
            all_models.update(sample.model for sample in jev_workload.samples)
            jev_cost += workload_cost
        if (
            result.jev.resolved_models != sorted(all_models)
            or result.jev.completed_call_count != JEV_ITERATIONS * len(JEV_BATCH_SIZES)
            or jev_cost != Decimal(result.jev.cost_usd or "-1")
        ):
            raise ValueError("Jev aggregate contradiction")
        expected_comparison = build_comparison(
            {key: value.model_dump() for key, value in result.local.workloads.items()},
            result.jev.model_dump(),
        )
        if (
            result.comparison is None
            or {key: value.model_dump() for key, value in result.comparison.items()}
            != expected_comparison
        ):
            raise ValueError("comparison contradiction")
    elif result.comparison is not None:
        raise ValueError("partial result cannot contain comparison")

    journal_rows = [
        json.loads(line) for line in paths["cost-journal.jsonl"].read_bytes().splitlines()
    ]
    reservations: dict[str, Decimal] = {}
    settled_ids: set[str] = set()
    settled_cost = Decimal(0)
    for row in journal_rows:
        if (
            not isinstance(row, dict)
            or row.get("predecessor_sha256") != result.provenance.prior_cost_ledger_sha256
        ):
            raise ValueError("journal provenance")
        if row.get("kind") == "reserve" and set(row) == {
            "kind",
            "request_id",
            "cost",
            "predecessor_sha256",
            "settled",
        }:
            reservations[str(row["request_id"])] = _decimal(row["cost"])
        elif row.get("kind") == "settle" and set(row) == {
            "kind",
            "request_id",
            "cost",
            "status",
            "predecessor_sha256",
        }:
            request_id = str(row["request_id"])
            cost = _decimal(row["cost"])
            if (
                request_id not in reservations
                or request_id in settled_ids
                or cost > reservations[request_id]
            ):
                raise ValueError("journal settlement")
            settled_ids.add(request_id)
            settled_cost += cost
        elif row.get("kind") == "uncertain" and set(row) == {
            "kind",
            "request_id",
            "predecessor_sha256",
        }:
            pass
        else:
            raise ValueError("journal row shape")
    open_reservations = set(reservations) - settled_ids
    if (
        settled_cost != Decimal(result.budget.phase3c_jev_usd)
        or settled_cost != Decimal(result.jev.cost_usd or "0")
        or (result.jev.status == "complete" and (open_reservations or len(settled_ids) != 30))
        or (result.jev.status == "not_requested" and journal_rows)
    ):
        raise ValueError("journal/result contradiction")
    sanitized_report(result.model_dump(mode="json"))
    return result


def load_local_components(
    snapshot: Path, checkpoint: Path, *, expected_checkpoint_sha256: str
) -> LocalComponents:
    """Load only the reviewed local MPS stack; never silently falls back to CPU."""
    try:
        import torch  # type: ignore[import-not-found]
        from safetensors.torch import load  # type: ignore[import-not-found]
    except ImportError as exc:
        raise SyntheticError("encoder-eval extra is required") from exc
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "").lower() not in {"", "0", "false"}:
        raise SyntheticError("MPS fallback must be disabled")
    if not torch.backends.mps.is_available():
        raise SyntheticError("Apple MPS is required")
    from benchmarks.encoder_loader import VerifiedSnapshot, load_encoder

    candidate = get_candidate("multilingual-minilm-l12")
    loaded = load_encoder(
        candidate, snapshot, verified=VerifiedSnapshot.create(candidate, snapshot)
    )
    checkpoint_bytes = checkpoint.read_bytes()
    if digest_bytes(checkpoint_bytes) != expected_checkpoint_sha256:
        raise SyntheticError("checkpoint changed after manifest validation")
    head = torch.nn.Linear(384, 5)
    head.load_state_dict(load(checkpoint_bytes))
    loaded.model.to("mps").eval()
    head.eval()

    class Components:
        def synchronize(self) -> None:
            torch.mps.synchronize()

        def tokenize(self, texts: Sequence[str]) -> Any:
            return loaded.tokenizer(
                list(texts),
                padding=True,
                truncation=True,
                max_length=MAX_TOKENS,
                return_tensors="pt",
            )

        def infer_and_pool(self, encoded: Any) -> Any:
            inputs = {key: value.to("mps") for key, value in encoded.items()}
            with torch.inference_mode():
                output = loaded.model(**inputs).last_hidden_state.float()
                mask = inputs["attention_mask"].unsqueeze(-1).to(dtype=output.dtype)
                return (output * mask).sum(1) / mask.sum(1)

        def to_cpu(self, pooled: Any) -> Any:
            return pooled.cpu()

        def predict(self, pooled: Any) -> Sequence[int]:
            with torch.inference_mode():
                return [int(value) for value in head(pooled).argmax(dim=1).tolist()]

        def token_lengths(self, texts: Sequence[str]) -> Sequence[int]:
            encoded = loaded.tokenizer(list(texts), padding=False, truncation=False)
            return [len(ids) for ids in encoded["input_ids"]]

    return Components()


def _urllib_transport(
    method: str, url: str, headers: Mapping[str, str], body: bytes, *, timeout: float
) -> tuple[int, Mapping[str, str], bytes]:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != "openrouter.ai":
        raise SyntheticError("literal OpenRouter host required")
    request = urllib.request.Request(
        url, data=None if method == "GET" else body, headers=dict(headers), method=method
    )
    context = ssl.create_default_context()

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(
            self, request: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str
        ) -> Any:
            return None

    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        NoRedirect(),
        urllib.request.HTTPSHandler(context=context),
    )
    try:
        with opener.open(request, timeout=timeout) as response:
            return response.status, dict(response.headers), response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers), exc.read()
    except (urllib.error.URLError, TimeoutError) as exc:
        raise SyntheticError("OpenRouter request failed") from exc


def dispatch_jev(
    payload: Mapping[str, Any],
    *,
    api_key: str,
    journal: CostJournal,
    transport: Transport = _urllib_transport,
    clock: Clock = time,
    request_id: str = "jev-000",
) -> tuple[dict[str, Any], float]:
    if (
        not api_key
        or set(payload) != {"model", "state", "questions", "provider"}
        or payload.get("model") != JEV_MODEL
        or payload.get("provider") != PROVIDER_CONTROLS
    ):
        raise ValueError("invalid Jev dispatch")
    body = canonical_json_bytes(cast(JsonValue, dict(payload)))
    worst_case = Decimal(len(body)) * JEV_INPUT_PRICE / Decimal(1_000_000)
    journal.reserve(request_id, worst_case)
    started = clock.perf_counter_ns()
    try:
        status, _, raw = transport(
            "POST",
            "https://openrouter.ai/api/alpha/decisions",
            {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            body,
            timeout=60.0,
        )
    except Exception:
        journal.uncertain_outcome(request_id)
        raise
    elapsed = (clock.perf_counter_ns() - started) / 1_000_000
    if status != 200:
        try:
            failed = json.loads(raw)
            usage = failed.get("usage") if isinstance(failed, dict) else None
            cost = usage.get("cost") if isinstance(usage, dict) else None
            if _finite(cost):
                journal.settle(request_id, Decimal(str(cost)), status="billed_failure")
            else:
                journal.uncertain_outcome(request_id)
        except Exception:
            if not journal.uncertain:
                journal.uncertain_outcome(request_id)
        raise SyntheticError("provider error")
    try:
        response = json.loads(raw)
        if not isinstance(response, dict):
            raise ValueError("provider response shape")
        keys = payload.get("questions", {}).keys()
        validated = validate_jev_response(response, list(keys))
        journal.settle(request_id, Decimal(validated["usage"]["cost"]))
    except Exception:
        if not journal.uncertain:
            journal.uncertain_outcome(request_id)
        raise
    return response, elapsed


def audit_openrouter_key(api_key: str, *, transport: Transport = _urllib_transport) -> Decimal:
    status, _, raw = transport(
        "GET",
        "https://openrouter.ai/api/v1/key",
        {"Authorization": f"Bearer {api_key}"},
        b"",
        timeout=60.0,
    )
    if status != 200:
        raise SyntheticError("OpenRouter key audit failed")
    value = json.loads(raw)
    usage = value.get("data", {}).get("usage") if isinstance(value, dict) else None
    return _decimal(usage)


def process_max_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if platform.system() == "Darwin" else value * 1024)


class RemoteRunError(RuntimeError):
    def __init__(self, partial: Mapping[str, Any], cause: BaseException) -> None:
        super().__init__("Jev external control became unavailable")
        self.partial = dict(partial)
        self.__cause__ = cause


def run_jev_workloads(
    rows: Sequence[Mapping[str, Any]],
    *,
    api_key: str,
    journal: CostJournal,
    transport: Transport = _urllib_transport,
    clock: Clock = time,
    iterations: int = JEV_ITERATIONS,
) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: str(row.get("family_id", "")))
    if len(ordered) < max(JEV_BATCH_SIZES) or iterations <= 0:
        raise ValueError("insufficient Jev workload")
    workloads: dict[str, Any] = {}
    completed_digests: list[str] = []
    resolved_models: set[str] = set()
    completed = 0
    total_cost = Decimal(0)
    try:
        for batch_size in JEV_ORDER:
            batch = ordered[:batch_size]
            expected = [str(row.get("candidate_label")) for row in batch]
            samples: list[dict[str, Any]] = []
            for attempt in range(iterations):
                payload = build_jev_request(batch)
                question_keys = list(payload["questions"])
                request_id = f"jev-b{batch_size:03d}-i{attempt:02d}"
                response, elapsed_ms = dispatch_jev(
                    payload,
                    api_key=api_key,
                    transport=transport,
                    clock=clock,
                    journal=journal,
                    request_id=request_id,
                )
                reduced = validate_jev_response(response, question_keys)
                predicted = [reduced["answers"][key] for key in question_keys]
                accuracy = (
                    sum(
                        actual == wanted for actual, wanted in zip(predicted, expected, strict=True)
                    )
                    / batch_size
                )
                sample = {
                    "latency_ms": elapsed_ms,
                    "answer_digest": reduced["answer_digest"],
                    "accuracy": accuracy,
                    "model": reduced["model"],
                    "provider": reduced["provider"],
                    "usage": reduced["usage"],
                }
                samples.append(sample)
                completed += 1
                completed_digests.append(str(reduced["answer_digest"]))
                resolved_models.add(str(reduced["model"]))
                total_cost += Decimal(str(reduced["usage"]["cost"]))
            latencies = [float(sample["latency_ms"]) for sample in samples]
            workloads[str(batch_size)] = {
                "batch_size": batch_size,
                "iterations": iterations,
                "input_id_digest": opaque_input_digest(batch),
                "samples": samples,
                "samples_sha256": digest_json(samples),
                "summary": summarize(latencies),
                "per_item_summary": summarize([value / batch_size for value in latencies]),
                "throughput_items_per_second": throughput(batch_size, latencies),
                "accuracy": sum(float(sample["accuracy"]) for sample in samples) / iterations,
                "cost_usd": str(
                    sum((Decimal(str(sample["usage"]["cost"])) for sample in samples), Decimal(0))
                ),
            }
    except Exception as exc:
        raise RemoteRunError(
            {
                "status": "external_control_unavailable",
                "completed_call_count": completed,
                "completed_answer_digests": completed_digests,
                "resolved_models": sorted(resolved_models),
                "cost_usd": str(total_cost),
            },
            exc,
        ) from exc
    return {
        "status": "complete",
        "completed_call_count": completed,
        "completed_answer_digests": completed_digests,
        "resolved_models": sorted(resolved_models),
        "provider": "TypeSafe",
        "cost_usd": str(total_cost),
        "workloads": workloads,
    }


def build_comparison(local: Mapping[str, Any], jev: Mapping[str, Any]) -> dict[str, Any] | None:
    if jev.get("status") != "complete":
        return None
    comparison: dict[str, Any] = {}
    remote = jev["workloads"]
    for batch_size in JEV_BATCH_SIZES:
        key = str(batch_size)
        local_p50 = float(local[key]["per_item_summary"]["p50_ms"])
        jev_p50 = float(remote[key]["per_item_summary"]["p50_ms"])
        comparison[key] = {
            "batch_size": batch_size,
            "local_per_item_p50_ms": local_p50,
            "jev_per_item_p50_ms": jev_p50,
            "local_speedup_over_jev": jev_p50 / local_p50,
        }
    return comparison


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="phase3c.v1")
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--prior-cost-ledger", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--budget-usd", required=True)
    args = parser.parse_args(argv)
    if args.budget_usd != "0.25":
        raise SystemExit("approved budget must be exactly 0.25")
    packet_manifest = (
        args.packet
        if args.packet.name == "packet-manifest.json"
        else args.packet / "packet-manifest.json"
    )
    packet_dir = packet_manifest.parent
    validate_packet_manifest(packet_manifest)
    training = validate_training_manifest(args.training, packet_manifest)
    prior_raw = args.prior_cost_ledger.read_bytes()
    prior = json.loads(prior_raw)
    prior_ledger = BudgetLedger.from_json(prior)
    if prior.get("final") is not True:
        raise SystemExit("prior cost ledger must be final")
    predecessor = digest_bytes(prior_raw)
    packet_sha256 = digest_bytes(packet_manifest.read_bytes())
    training_sha256 = digest_bytes(args.training.read_bytes())
    validate_predecessor_binding(
        args.prior_cost_ledger,
        packet_sha256=packet_sha256,
        training_sha256=training_sha256,
    )
    rows = sorted(_read_accepted(packet_dir), key=lambda row: str(row.get("family_id", "")))
    if len(rows) < max(LOCAL_BATCH_SIZES):
        raise SystemExit("accepted packet is smaller than the fixed local workload")
    try:
        from benchmarks.encoder_acquisition import snapshot_path

        snapshot = snapshot_path(get_candidate("multilingual-minilm-l12"))
        rss_before = process_max_rss_bytes()
        load_started = time.perf_counter_ns()
        components = load_local_components(
            snapshot,
            args.training.parent / "checkpoint.safetensors",
            expected_checkpoint_sha256=training["files"]["checkpoint.safetensors"],
        )
        load_ms = (time.perf_counter_ns() - load_started) / 1_000_000
        rss_after_load = process_max_rss_bytes()
        local = run_local_workloads(rows, components)
        rss_after_workloads = process_max_rss_bytes()
    except Exception:
        raise SystemExit("local Phase 3C measurement was rejected") from None
    local_result = {
        "fresh_process_load_ms": load_ms,
        "page_cache_sensitive": True,
        "load_includes_import_and_snapshot_verification": True,
        "rss_metric": "ru_maxrss_bytes",
        "rss_before_load_bytes": rss_before,
        "rss_after_load_bytes": rss_after_load,
        "rss_after_workloads_bytes": rss_after_workloads,
        "workloads": local,
    }

    journal_path = args.output_dir / "cost-journal.jsonl"
    claim_sha256: str | None = None
    if args.allow_network:
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            raise SystemExit("OPENROUTER_API_KEY is required with --allow-network")
        key_usage_pre = audit_openrouter_key(api_key)
        claim_path = args.prior_cost_ledger.parent / ".phase3c-claims" / f"{predecessor}.json"
        claim_sha256 = create_claim(claim_path, predecessor)
        args.output_dir.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        args.output_dir.mkdir(mode=0o700, parents=False, exist_ok=False)
        os.chmod(args.output_dir, 0o700)
        initial_total = sum(
            (Decimal(str(entry["cost"])) for entry in prior_ledger.entries), Decimal(0)
        )
        initial_jev = sum(
            (
                Decimal(str(entry["cost"]))
                for entry in prior_ledger.entries
                if entry["stage"] == "jev"
            ),
            Decimal(0),
        )
        journal = CostJournal(
            journal_path,
            predecessor_sha256=predecessor,
            initial_stage_spent=initial_jev,
            initial_total_spent=initial_total,
        )
        try:
            jev = run_jev_workloads(rows, api_key=api_key, journal=journal)
        except RemoteRunError as exc:
            jev = exc.partial
        jev["cost_usd"] = str(journal.stage_settled - initial_jev)
        jev["key_usage_pre_usd"] = str(key_usage_pre)
        try:
            jev["key_usage_post_usd"] = str(audit_openrouter_key(api_key))
        except Exception:
            jev["key_usage_post_usd"] = None
    else:
        jev = {"status": "not_requested", "completed_call_count": 0}
    result = {
        "schema_version": "phase3c.v1",
        "sealed": True,
        "protocol": {
            "local_batch_sizes": list(LOCAL_BATCH_SIZES),
            "local_order": list(LOCAL_ORDER),
            "local_warmups": LOCAL_WARMUPS,
            "local_iterations": LOCAL_ITERATIONS,
            "max_tokens": MAX_TOKENS,
            "device": "mps",
            "jev_batch_sizes": list(JEV_BATCH_SIZES),
            "jev_order": list(JEV_ORDER),
            "jev_iterations": JEV_ITERATIONS,
        },
        "provenance": {
            "packet_manifest_sha256": digest_bytes(packet_manifest.read_bytes()),
            "training_manifest_sha256": digest_bytes(args.training.read_bytes()),
            "checkpoint_sha256": training["files"]["checkpoint.safetensors"],
            "prior_cost_ledger_sha256": predecessor,
            "claim_sha256": claim_sha256,
            "benchmark_source_sha256": digest_bytes(Path(__file__).read_bytes()),
            "encoder_revision": training["encoder"]["revision"],
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "torch": importlib.metadata.version("torch"),
            "transformers": importlib.metadata.version("transformers"),
            "safetensors": importlib.metadata.version("safetensors"),
        },
        "local": local_result,
        "jev": jev,
        "comparison": build_comparison(local, jev),
        "budget": {
            "approved_total_usd": str(TOTAL_BUDGET),
            "jev_stage_limit_usd": str(JEV_BUDGET),
            "prior_total_usd": str(
                initial_total if args.allow_network else _decimal(prior_ledger.total)
            ),
            "phase3c_jev_usd": str(jev.get("cost_usd", "0")),
        },
        "limitations": list(MANDATORY_LIMITATIONS),
    }
    write_phase3c_artifacts(
        args.output_dir,
        result,
        journal_path.read_bytes() if journal_path.exists() else b"",
        packet_sha256=packet_sha256,
        training_sha256=training_sha256,
        predecessor_sha256=predecessor,
        claim_sha256=claim_sha256,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
