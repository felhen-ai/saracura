"""Offline-only machinery for the first Saracura-owned checkpoint.

This module intentionally has no model import.  It seals aggregate evidence,
defines the immutable candidate grid, and validates selection evidence.  Model
loading is deferred to the separately authorised live phases.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import random
import re
import stat
import tempfile
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from pydantic import JsonValue

from benchmarks.v02_corpus import RENDERER_CONTRACT
from benchmarks.v02_evaluation import (
    _normalise,
    _option_description,
    _validate_descriptor,
    _validate_receipt,
    combined_content_fingerprint,
    state_question_fingerprint,
)
from benchmarks.v02_training_loop import run_optimization
from saracura.serialization import canonical_json_bytes

MANIFEST_PATH = Path(__file__).parent / "manifests/v02-first-checkpoint.v1.json"
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
SOURCE_POLICY_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9._:-]{0,127}$")
PUBLIC_DEV_QUERY_ID = re.compile(r"^[\x21-\x7e]+$")
CANDIDATE_GRID = (
    {
        "id": "c1-r8",
        "lora_rank": 8,
        "lora_alpha": 16,
        "dropout": 0.05,
        "peak_learning_rate": "2e-4",
    },
    {
        "id": "c2-r16",
        "lora_rank": 16,
        "lora_alpha": 32,
        "dropout": 0.05,
        "peak_learning_rate": "1e-4",
    },
    {
        "id": "c3-r32",
        "lora_rank": 32,
        "lora_alpha": 64,
        "dropout": 0.05,
        "peak_learning_rate": "5e-5",
    },
)

# This is deliberately a semantic inventory, rather than PEFT's broad
# ``all-linear`` convenience alias.  Qwen 3.5 4B alternates 24 linear-attention
# blocks (five linear-attention projections plus MLP) and eight full-attention
# blocks (q/k/v/o plus MLP), for exactly 248 targets.
_LINEAR_ATTENTION_LAYERS = frozenset(index for index in range(32) if (index + 1) % 4)
_LINEAR_ATTENTION_PROJECTIONS = (
    "linear_attn.out_proj",
    "linear_attn.in_proj_qkv",
    "linear_attn.in_proj_z",
    "linear_attn.in_proj_b",
    "linear_attn.in_proj_a",
)
_FULL_ATTENTION_PROJECTIONS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
)
_MLP_PROJECTIONS = ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")


def qwen35_lora_targets() -> tuple[str, ...]:
    """Return the closed, pinned 248-module Qwen text inventory."""
    targets: list[str] = []
    for layer in range(32):
        projections = (
            _LINEAR_ATTENTION_PROJECTIONS
            if layer in _LINEAR_ATTENTION_LAYERS
            else _FULL_ATTENTION_PROJECTIONS
        )
        targets.extend(f"layers.{layer}.{projection}" for projection in projections)
        targets.extend(f"layers.{layer}.{projection}" for projection in _MLP_PROJECTIONS)
    if len(targets) != 248 or len(set(targets)) != len(targets):
        raise RuntimeError("pinned Qwen target inventory drifted")
    return tuple(targets)


TRAINING_DESCRIPTOR_FIELDS = frozenset(
    {
        "schema_version",
        "record_count",
        "split_counts",
        "identity_set_digest",
        "state_question_fingerprint_set_digest",
        "combined_content_fingerprint_set_digest",
        "option_multiset_fingerprint_set_digest",
        "accepted_packet_digest",
        "successor_grant_digest",
        "candidate_rendering_digest",
        "source_policy_identifiers",
    }
)
READINESS_DESCRIPTOR_FIELDS = frozenset(
    {
        "schema_version",
        "training_descriptor_digest",
        "sealed_descriptor_digest",
        "disjointness_receipt_digest",
        "candidate_rendering_digest",
        "kev_rendering_digest",
        "all_records_candidate_preflight_passed",
        "all_records_kev_preflight_passed",
    }
)
PUBLIC_DEV_PLANNED_ROW_FIELDS = frozenset(
    {
        "query_id",
        "query_text",
        "ordered_choice_ids",
        "ordered_choice_snippets",
        "gold_position",
    }
)


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key: {key}")
        result[key] = value
    return result


def _load_closed(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicates)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError("JSON root must be an object")
    return value


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _digest_set(values: Iterable[str]) -> str:
    return _digest(sorted(set(values)))


def _require_digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or not HEX64.fullmatch(value):
        raise ValueError(f"{label} must be sha256")
    return value


def load_manifest() -> dict[str, Any]:
    manifest = _load_closed(MANIFEST_PATH)
    expected = {
        "schema_version",
        "architecture",
        "candidate_grid",
        "early_stopping",
        "evaluation",
        "renderer",
        "selection",
        "training",
    }
    if set(manifest) != expected or manifest["schema_version"] != "v02-first-checkpoint.v1":
        raise ValueError("first checkpoint manifest is not closed")
    if manifest["candidate_grid"] != list(CANDIDATE_GRID):
        raise ValueError("candidate grid drifted")
    renderer = manifest["renderer"]
    if not isinstance(renderer, dict) or renderer != {
        "candidate_rendering_digest": RENDERER_CONTRACT["candidate_rendering_digest"],
        "candidate_renderer_config": {
            "head_dim": 256,
            "lora_targets": "qwen35_4b_text_248",
            "max_branch": 1024,
            "max_packed": 2048,
            "max_state": 384,
            "option_isolation": False,
            "special_embeddings": False,
            "strict": True,
        },
        "source_path": "kev/model.py",
        "source_revision": RENDERER_CONTRACT["candidate_renderer_source_revision"],
        "source_sha256": RENDERER_CONTRACT["candidate_renderer_source_sha256"],
        "token_limit": 512,
        "truncation_disabled": True,
    }:
        raise ValueError("candidate renderer ledger drifted")
    if manifest["evaluation"] != {
        "batch": 1,
        "concurrency": 1,
        "determinism_flags": [
            "CUBLAS_WORKSPACE_CONFIG",
            "torch.use_deterministic_algorithms",
            "cudnn_benchmark_false",
        ],
        "diagnostic_subset_records": 50,
        "loopback_host": "127.0.0.1",
        "runtime_boundary": "loopback_http_python_worker_v1",
        "timing_procedure": "saracura-v02-loopback-single-request-v1",
        "warmups": 10,
        "warmup_measurements": 100,
    }:
        raise ValueError("loopback timing boundary drifted")
    return manifest


def _safe_output(path: Path) -> None:
    if not path.is_absolute() or path.name in {"", ".", ".."} or path.is_symlink():
        raise ValueError("output path must be an absolute non-symlink file")
    parent = path.parent
    if any(ancestor.is_symlink() for ancestor in (parent, *parent.parents)):
        raise ValueError("output path must not traverse a symlink")
    if not parent.is_dir() or stat.S_IMODE(parent.stat().st_mode) != 0o700:
        raise ValueError("output parent must be a private existing directory")
    if path.exists():
        raise FileExistsError("create-only output already exists")


def create_json(path: Path, value: dict[str, Any]) -> None:
    """Atomically create a 0600 canonical JSON artifact, never replacing one."""
    _safe_output(path)
    payload = canonical_json_bytes(value) + b"\n"
    descriptor, temporary = tempfile.mkstemp(prefix=".v02-", dir=path.parent)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError("create-only output already exists") from exc
    finally:
        with suppress(FileNotFoundError):
            os.unlink(temporary)


def _record(record: Any) -> dict[str, Any]:
    if not isinstance(record, dict) or set(record) != {
        "opaque_record_id",
        "split",
        "state",
        "question",
        "options",
    }:
        raise ValueError("training record is not closed")
    if record["split"] not in {"train", "internal_dev"} or not isinstance(
        record["opaque_record_id"], str
    ):
        raise ValueError("training record identity or split invalid")
    if not isinstance(record["options"], list):
        raise ValueError("training options invalid")
    return record


def public_dev_planned_row(record: Any) -> dict[str, Any]:
    """Validate the frozen Phase 5D planned-row shape without treating it as training data."""
    if not isinstance(record, dict) or set(record) != PUBLIC_DEV_PLANNED_ROW_FIELDS:
        raise ValueError("public development planned row is not closed")
    query_id = record["query_id"]
    choice_ids = record["ordered_choice_ids"]
    snippets = record["ordered_choice_snippets"]
    gold_position = record["gold_position"]
    if (
        not isinstance(query_id, str)
        or not PUBLIC_DEV_QUERY_ID.fullmatch(query_id)
        or not isinstance(record["query_text"], str)
        or not record["query_text"]
        or not isinstance(choice_ids, (list, tuple))
        or len(choice_ids) != 4
        or len(set(choice_ids)) != 4
        or not all(
            isinstance(value, str) and PUBLIC_DEV_QUERY_ID.fullmatch(value) for value in choice_ids
        )
        or not isinstance(snippets, (list, tuple))
        or len(snippets) != 4
        or not all(isinstance(value, str) and value for value in snippets)
        or not isinstance(gold_position, int)
        or isinstance(gold_position, bool)
        or not 0 <= gold_position < 4
    ):
        raise ValueError("public development planned row is invalid")
    return record


def public_dev_opaque_record_id(record: Any) -> str:
    """Return the Phase 5D canonical opaque identity used by the frozen subset order."""
    return cast(str, public_dev_planned_row(record)["query_id"])


def _option_multiset_fingerprint(options: list[Any]) -> str:
    """Use the same #44 NFC/casefold/RFC-8785 option-description normalization."""
    normalized = [_normalise(_option_description(option)) for option in options]
    return hashlib.sha256(
        canonical_json_bytes(
            [json.loads(item) for item in sorted(canonical_json_bytes(x) for x in normalized)]
        )
    ).hexdigest()


def seal_training(
    records: list[Any],
    *,
    accepted_packet_digest: str,
    successor_grant_digest: str,
    source_policy_identifiers: list[str],
) -> dict[str, Any]:
    """Produce only aggregate descriptor material; callers retain raw rows privately."""
    load_manifest()
    if (
        not records
        or not source_policy_identifiers
        or len(source_policy_identifiers) != len(set(source_policy_identifiers))
        or not all(
            isinstance(item, str) and SOURCE_POLICY_IDENTIFIER.fullmatch(item)
            for item in source_policy_identifiers
        )
    ):
        raise ValueError("training lanes and source policies are required")
    parsed = [_record(item) for item in records]
    if {item["split"] for item in parsed} != {"train", "internal_dev"}:
        raise ValueError("Phase 4E train and internal dev must both be in training lane")
    identities = [item["opaque_record_id"] for item in parsed]
    if len(identities) != len(set(identities)):
        raise ValueError("training identity collision")
    return {
        "schema_version": "v02-training-descriptor.v1",
        "record_count": len(parsed),
        "split_counts": {
            split: sum(item["split"] == split for item in parsed)
            for split in ("train", "internal_dev")
        },
        "identity_set_digest": _digest_set(identities),
        "state_question_fingerprint_set_digest": _digest_set(
            state_question_fingerprint(item["state"], item["question"]) for item in parsed
        ),
        "combined_content_fingerprint_set_digest": _digest_set(
            combined_content_fingerprint(item["state"], item["question"], item["options"])
            for item in parsed
        ),
        "option_multiset_fingerprint_set_digest": _digest_set(
            _option_multiset_fingerprint(item["options"]) for item in parsed
        ),
        "accepted_packet_digest": _require_digest(accepted_packet_digest, "accepted packet"),
        "successor_grant_digest": _require_digest(successor_grant_digest, "successor grant"),
        "candidate_rendering_digest": RENDERER_CONTRACT["candidate_rendering_digest"],
        "source_policy_identifiers": sorted(source_policy_identifiers),
    }


def validate_training_descriptor(value: dict[str, Any]) -> None:
    if (
        set(value) != TRAINING_DESCRIPTOR_FIELDS
        or value.get("schema_version") != "v02-training-descriptor.v1"
    ):
        raise ValueError("training descriptor is not closed")
    if (
        not isinstance(value["record_count"], int)
        or isinstance(value["record_count"], bool)
        or value["record_count"] <= 0
        or not isinstance(value["split_counts"], dict)
        or set(value["split_counts"]) != {"train", "internal_dev"}
        or not all(
            isinstance(count, int) and not isinstance(count, bool) and count > 0
            for count in value["split_counts"].values()
        )
        or sum(value["split_counts"].values()) != value["record_count"]
    ):
        raise ValueError("training descriptor counts invalid")
    for key in TRAINING_DESCRIPTOR_FIELDS - {
        "schema_version",
        "record_count",
        "split_counts",
        "source_policy_identifiers",
    }:
        _require_digest(value[key], key)
    if value["candidate_rendering_digest"] != RENDERER_CONTRACT["candidate_rendering_digest"]:
        raise ValueError("candidate renderer digest mismatch")
    policies = value["source_policy_identifiers"]
    if (
        not isinstance(policies, list)
        or not policies
        or len(policies) != len(set(policies))
        or policies != sorted(policies)
        or not all(
            isinstance(item, str) and SOURCE_POLICY_IDENTIFIER.fullmatch(item) for item in policies
        )
    ):
        raise ValueError("training source policy identifiers are invalid")


def _length_frame(value: bytes) -> bytes:
    return len(value).to_bytes(8, "big") + value


def _diagnostic_order_key(row: dict[str, Any]) -> bytes:
    identifier = public_dev_opaque_record_id(row)
    segments = (
        b"saracura-v02-public-dev-order-v1",
        identifier.encode("utf-8"),
    )
    return hashlib.sha256(b"".join(_length_frame(segment) for segment in segments)).digest()


def diagnostic_subset(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(records) < 50:
        raise ValueError("public development lane must contain at least 50 records")
    identities = [public_dev_opaque_record_id(record) for record in records]
    if len(identities) != len(set(identities)):
        raise ValueError("public development lane has an opaque identity collision")
    return sorted(
        records,
        key=_diagnostic_order_key,
    )[:50]


def select_candidate(candidates: list[dict[str, Any]]) -> dict[str, str]:
    """Apply the frozen development-only rule without receiving a held-out path."""
    if len(candidates) != 3 or {row.get("id") for row in candidates} != {
        item["id"] for item in CANDIDATE_GRID
    }:
        raise ValueError("selection requires exactly the frozen three candidates")
    eligible: list[dict[str, Any]] = []
    for row in candidates:
        if candidate_is_eligible(row):
            eligible.append(row)
    if not eligible:
        return {"status": "NO_RELEASE_CANDIDATE"}
    winner = sorted(
        eligible,
        key=lambda row: (
            -row["planned_top1_accuracy"],
            -row["option_order_stability"],
            row["warm_p95_ms"],
            row["id"],
        ),
    )[0]
    return {"status": "SELECTED", "candidate_id": str(winner["id"])}


def candidate_is_eligible(candidate: dict[str, Any]) -> bool:
    """Validate and apply the frozen per-candidate development eligibility rule."""
    if set(candidate) != {
        "id",
        "coverage",
        "invalid_output_rate",
        "repeat_stability",
        "option_order_stability",
        "planned_top1_accuracy",
        "warm_p95_ms",
        "valid",
        "truncation_detected",
        "identity_match",
    }:
        raise ValueError("candidate report is not closed")
    metrics = (
        "coverage",
        "invalid_output_rate",
        "repeat_stability",
        "option_order_stability",
        "planned_top1_accuracy",
        "warm_p95_ms",
    )
    if not all(
        isinstance(candidate[key], (int, float))
        and not isinstance(candidate[key], bool)
        and math.isfinite(float(candidate[key]))
        and float(candidate[key]) >= 0
        for key in metrics
    ) or any(float(candidate[key]) > 1 for key in metrics[:-1]):
        raise ValueError("candidate metric invalid")
    if not all(
        isinstance(candidate[key], bool)
        for key in ("valid", "truncation_detected", "identity_match")
    ):
        raise ValueError("candidate status invalid")
    return (
        candidate["valid"] is True
        and candidate["truncation_detected"] is False
        and candidate["identity_match"] is True
        and candidate["coverage"] >= 0.98
        and candidate["invalid_output_rate"] == 0
        and candidate["repeat_stability"] == 1
        and candidate["option_order_stability"] >= 0.95
        and candidate["planned_top1_accuracy"] >= 0.4
    )


def loopback_worker_plan() -> dict[str, Any]:
    """Closed configuration shared by candidate and Kev loopback workers.

    The actual model process is intentionally deferred to the authorised GPU
    phase; this plan is the offline proof that neither side can choose a
    different endpoint, batching, or measurement procedure.
    """
    evaluation = load_manifest()["evaluation"]
    assert isinstance(evaluation, dict)
    return dict(evaluation)


LOOPBACK_REQUEST_SCHEMA = "saracura-v02-loopback-request.v1"
LOOPBACK_RESPONSE_SCHEMA = "saracura-v02-loopback-response.v1"


def _loopback_endpoint(endpoint: str) -> str:
    parsed = urlparse(endpoint)
    if (
        parsed.scheme != "http"
        or parsed.hostname != "127.0.0.1"
        or parsed.port is None
        or parsed.path not in {"", "/"}
        or parsed.params
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("loopback endpoint must be plain http://127.0.0.1:<port>")
    return f"http://127.0.0.1:{parsed.port}/"


def _closed_loopback_response(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "selected_option_id",
        "worker_rss_bytes",
        "device_memory_bytes",
        "worker_started_monotonic_ns",
    }:
        raise ValueError("loopback response is not closed")
    if (
        value["schema_version"] != LOOPBACK_RESPONSE_SCHEMA
        or not isinstance(value["selected_option_id"], str)
        or not value["selected_option_id"]
        or any(
            not isinstance(value[key], int) or isinstance(value[key], bool) or value[key] < 0
            for key in ("worker_rss_bytes", "device_memory_bytes", "worker_started_monotonic_ns")
        )
    ):
        raise ValueError("loopback response fields invalid")
    return cast(dict[str, Any], value)


def _loopback_request(endpoint: str, record: dict[str, Any]) -> tuple[dict[str, Any], float]:
    payload = canonical_json_bytes({"schema_version": LOOPBACK_REQUEST_SCHEMA, "record": record})
    started = time.monotonic()
    request = Request(
        _loopback_endpoint(endpoint),
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=10) as response:
            if response.status != 200:
                raise ValueError("loopback worker returned non-success status")
            value = json.loads(response.read(), object_pairs_hook=_reject_duplicates)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("loopback request failed") from exc
    return _closed_loopback_response(value), (time.monotonic() - started) * 1000


@contextmanager
def loopback_worker(
    prediction_handler: Callable[[dict[str, Any]], dict[str, Any]],
) -> Iterator[str]:
    """Serve the frozen candidate/Kev loopback boundary through an injected local handler.

    Phase B deliberately supplies only an in-process test handler.  Later
    authorised phases may inject a verified local candidate or Kev callable;
    model paths, downloads, remote endpoints and implicit fallbacks are not an
    input to this boundary.
    """
    started_ns = time.monotonic_ns()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            if self.path != "/" or self.headers.get("Content-Type") != "application/json":
                self.send_error(400)
                return
            try:
                length = int(self.headers["Content-Length"])
                request = json.loads(self.rfile.read(length), object_pairs_hook=_reject_duplicates)
                if not isinstance(request, dict) or set(request) != {"schema_version", "record"}:
                    raise ValueError
                if request["schema_version"] != LOOPBACK_REQUEST_SCHEMA or not isinstance(
                    request["record"], dict
                ):
                    raise ValueError
                result = prediction_handler(request["record"])
                if not isinstance(result, dict) or set(result) != {
                    "selected_option_id",
                    "worker_rss_bytes",
                    "device_memory_bytes",
                }:
                    raise ValueError
                response = {
                    "schema_version": LOOPBACK_RESPONSE_SCHEMA,
                    **result,
                    "worker_started_monotonic_ns": started_ns,
                }
                _closed_loopback_response(response)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                self.send_error(400)
                return
            body = canonical_json_bytes(response)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format: str, *_args: object) -> None:
            return

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def measure_loopback_timing(endpoint: str, ordered_subset: list[dict[str, Any]]) -> dict[str, Any]:
    """Execute the frozen cold/warmup/measurement procedure, one request at a time."""
    plan = loopback_worker_plan()
    if len(ordered_subset) != plan["diagnostic_subset_records"]:
        raise ValueError("timing requires the ordered 50-record diagnostic subset")
    endpoint = _loopback_endpoint(endpoint)
    first, _first_request_ms = _loopback_request(endpoint, ordered_subset[0])
    cold_ms = (time.monotonic_ns() - first["worker_started_monotonic_ns"]) / 1_000_000
    for index in range(plan["warmups"]):
        _loopback_request(endpoint, ordered_subset[index % len(ordered_subset)])
    measured: list[float] = []
    rss = first["worker_rss_bytes"]
    device_memory = first["device_memory_bytes"]
    started = time.monotonic()
    for index in range(plan["warmup_measurements"]):
        response, elapsed = _loopback_request(endpoint, ordered_subset[index % len(ordered_subset)])
        measured.append(elapsed)
        rss = max(rss, response["worker_rss_bytes"])
        device_memory = max(device_memory, response["device_memory_bytes"])
    wall_seconds = time.monotonic() - started

    def nearest_rank(percentile: float) -> float:
        return sorted(measured)[math.ceil(len(measured) * percentile) - 1]

    return {
        "schema_version": "saracura-v02-loopback-timing.v1",
        "cold_process_start_to_first_valid_response_ms": cold_ms,
        "warmups": plan["warmups"],
        "measured_requests": plan["warmup_measurements"],
        "warm_p50_ms": nearest_rank(0.50),
        "warm_p95_ms": nearest_rank(0.95),
        "throughput_decisions_per_second": len(measured) / wall_seconds,
        "peak_worker_rss_bytes": rss,
        "peak_device_memory_bytes": device_memory,
    }


def validate_code_revisions(revisions: dict[str, str], replacement_revision: str) -> None:
    """Fail closed unless every pre-held-out artifact uses one replacement revision."""
    if not revisions or not HEX40.fullmatch(replacement_revision):
        raise ValueError("code revision is invalid")
    terminal = {"selection_commit", "NO_RELEASE_CANDIDATE", "held_out_access"}
    if terminal.intersection(revisions):
        raise ValueError("terminal evidence requires a successor protocol")
    if any(not HEX40.fullmatch(value) for value in revisions.values()):
        raise ValueError("code revision is invalid")
    if any(value != replacement_revision for value in revisions.values()):
        raise ValueError("code correction invalidates phases C, D, and E globally")


def validate_readiness(training: Path, sealed: Path, readiness: Path, receipt: Path) -> int:
    training_value = _load_closed(training)
    validate_training_descriptor(training_value)
    training_digest = hashlib.sha256(training.read_bytes()).hexdigest()
    sealed_value = _load_closed(sealed)
    _validate_descriptor(sealed_value)
    receipt_value = _load_closed(receipt)
    _validate_receipt(
        receipt_value,
        sealed_value,
        hashlib.sha256(sealed.read_bytes()).hexdigest(),
    )
    training_receipt_bindings = {
        "training_descriptor_digest": training_digest,
        "training_identity_digest": training_value["identity_set_digest"],
        "training_state_question_digest": training_value["state_question_fingerprint_set_digest"],
        "training_combined_content_digest": training_value[
            "combined_content_fingerprint_set_digest"
        ],
        "training_option_multiset_digest": training_value["option_multiset_fingerprint_set_digest"],
    }
    if any(receipt_value[key] != expected for key, expected in training_receipt_bindings.items()):
        raise ValueError("disjointness receipt training bindings invalid")
    value = _load_closed(readiness)
    if (
        set(value) != READINESS_DESCRIPTOR_FIELDS
        or value.get("schema_version") != "v02-readiness-descriptor.v1"
    ):
        raise ValueError("readiness descriptor is not closed")
    for key in READINESS_DESCRIPTOR_FIELDS - {
        "schema_version",
        "all_records_candidate_preflight_passed",
        "all_records_kev_preflight_passed",
    }:
        _require_digest(value[key], key)
    if (
        value["training_descriptor_digest"] != training_digest
        or value["sealed_descriptor_digest"] != hashlib.sha256(sealed.read_bytes()).hexdigest()
        or value["disjointness_receipt_digest"] != hashlib.sha256(receipt.read_bytes()).hexdigest()
        or value["candidate_rendering_digest"] != RENDERER_CONTRACT["candidate_rendering_digest"]
        or value["all_records_candidate_preflight_passed"] is not True
        or value["all_records_kev_preflight_passed"] is not True
    ):
        raise ValueError("readiness bindings invalid")
    print("readiness valid")
    return 0


_ADMISSION_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_id",
        "private_identity",
        "capsule_sha256",
        "aggregate_receipt_sha256",
        "rights_receipt_sha256",
        "license_inventory_sha256",
        "environment_receipt_sha256",
        "corpus_source_revision",
        "trainer_source_revision",
        "trainer_code_inventory_sha256",
        "base_id",
        "base_revision",
        "base_inventory_sha256",
        "tokenizer_inventory_sha256",
        "renderer_sha256",
        "module_inventory_sha256",
        "candidate_config_sha256",
        "renderer_preflight_sha256",
        "sealed_plan_sha256",
        "sealed_prompt_inventory_sha256",
        "sealed_model_inventory_sha256",
        "seed",
        "training_authorized",
        "private_only",
        "publication_authorized",
    }
)


@dataclass(frozen=True)
class AdmissionProof:
    """An unforgeable-by-accident result produced before any model import/load."""

    admission_sha256: str


def verify_private_admission(
    admission_path: Path,
    receipt_path: Path,
    plan_path: Path,
    events_path: Path,
    *,
    verify_receipt: Callable[..., dict[str, Any]] | None = None,
) -> AdmissionProof:
    """Live-verify original receipt/rights material before model loading is possible."""
    paths = (admission_path, receipt_path, plan_path, events_path)
    if any(not path.is_absolute() or path.is_symlink() or not path.is_file() for path in paths):
        raise ValueError("private admission requires original absolute regular source files")
    admission = _load_closed(admission_path)
    if (
        set(admission) != _ADMISSION_FIELDS
        or admission.get("schema_version") != "v02-private-training-admission.v1"
    ):
        raise ValueError("private admission is not closed")
    if (
        admission["training_authorized"] is not True
        or admission["private_only"] is not True
        or admission["publication_authorized"] is not False
        or admission["seed"] != 20260929
        or admission["base_id"] != "Qwen/Qwen3.5-4B-Base"
        or admission["base_revision"] != "1001bb4d826a52d1f399e183466143f4da7b741b"
        or any(
            not isinstance(admission[key], str) or not HEX64.fullmatch(admission[key])
            for key in _ADMISSION_FIELDS
            - {
                "schema_version",
                "artifact_id",
                "private_identity",
                "corpus_source_revision",
                "trainer_source_revision",
                "base_id",
                "base_revision",
                "seed",
                "training_authorized",
                "private_only",
                "publication_authorized",
            }
        )
    ):
        raise ValueError("private admission flags or digests are invalid")
    if (
        hashlib.sha256(receipt_path.read_bytes()).hexdigest()
        != admission["aggregate_receipt_sha256"]
    ):
        raise ValueError("private admission original receipt digest mismatch")
    plan = _load_closed(plan_path)
    events_value = _load_closed(events_path).get("events")
    if not isinstance(events_value, list):
        raise ValueError("private admission events are not a closed list")
    if verify_receipt is None:
        from benchmarks.v02_corpus import verify_receipt as live_verify_receipt

        verify_receipt = live_verify_receipt
    receipt = verify_receipt(
        receipt_path, plan_data=plan, events=events_value, expected_lane="training"
    )
    if receipt.get("status") != "READY":
        raise ValueError("private admission corpus rights receipt is not ready")
    return AdmissionProof(hashlib.sha256(admission_path.read_bytes()).hexdigest())


def _training_runtime() -> tuple[Any, Any, Any, Any]:
    """Load the opt-in ML stack only for an explicitly invoked local run."""
    try:
        torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")
        peft = importlib.import_module("peft")
        safetensors = importlib.import_module("safetensors.torch")
    except ImportError as exc:
        raise RuntimeError("install the v02-training extra before a local training run") from exc
    return torch, transformers, peft, safetensors


def candidate_config(candidate_id: str) -> dict[str, Any]:
    """Return the immutable QLoRA and pointer-head contract for one candidate."""
    manifest = load_manifest()
    candidate = next((item for item in CANDIDATE_GRID if item["id"] == candidate_id), None)
    if candidate is None:
        raise ValueError("candidate is outside the frozen grid")
    return {
        "base_model": manifest["architecture"]["base_model"],
        "base_revision": manifest["architecture"]["base_revision"],
        "quantization": "nf4",
        "compute_dtype": "bfloat16",
        "lora_targets": qwen35_lora_targets(),
        "head_dim": 256,
        "trainable": "lora_and_pointer_readout_head",
        **candidate,
        **manifest["training"],
        **manifest["early_stopping"],
    }


def configure_determinism(torch: Any, seed: int) -> None:
    """Seed every declared source before constructing any trainable tensor."""
    configured = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if configured not in {None, ":4096:8"}:
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG conflicts with the frozen v0.2 procedure")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    try:
        numpy = importlib.import_module("numpy")
    except ImportError as exc:
        raise RuntimeError("v0.2 training requires NumPy for its declared seed contract") from exc
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if bool(torch.cuda.is_available()):
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def qwen35_text_config(model_config: Any) -> Any:
    """Return only a semantically pinned Qwen 3.5 4B text configuration."""
    text_config = getattr(model_config, "text_config", None)
    expected = {
        "model_type": "qwen3_5_text",
        "hidden_size": 2560,
        "num_hidden_layers": 32,
        "intermediate_size": 9216,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "vocab_size": 248320,
        "max_position_embeddings": 262144,
    }
    layers = list(getattr(text_config, "layer_types", ()))
    if (
        getattr(model_config, "model_type", None) != "qwen3_5"
        or list(getattr(model_config, "architectures", ())) != ["Qwen3_5ForConditionalGeneration"]
        or any(getattr(text_config, key, None) != value for key, value in expected.items())
        or len(layers) != 32
        or [index for index, value in enumerate(layers) if value == "full_attention"]
        != list(range(3, 32, 4))
    ):
        raise ValueError("base config is not the pinned Qwen3.5 text configuration")
    return text_config


def _semantic_text_config_matches(left: Any, right: Any) -> bool:
    return all(
        getattr(left, name, None) == getattr(right, name, None)
        for name in ("model_type", "hidden_size", "num_hidden_layers", "vocab_size")
    )


def _expected_adapter_shapes(rank: int) -> dict[str, tuple[int, int]]:
    dimensions = {
        "linear_attn.out_proj": (4096, 2560),
        "linear_attn.in_proj_qkv": (2560, 8192),
        "linear_attn.in_proj_z": (2560, 4096),
        "linear_attn.in_proj_b": (2560, 32),
        "linear_attn.in_proj_a": (2560, 32),
        "self_attn.q_proj": (2560, 8192),
        "self_attn.k_proj": (2560, 1024),
        "self_attn.v_proj": (2560, 1024),
        "self_attn.o_proj": (4096, 2560),
        "mlp.gate_proj": (2560, 9216),
        "mlp.up_proj": (2560, 9216),
        "mlp.down_proj": (9216, 2560),
    }
    result: dict[str, tuple[int, int]] = {}
    for target in qwen35_lora_targets():
        input_size, output_size = dimensions[target.split(".", 2)[2]]
        base = f"base_model.model.{target}"
        result[f"{base}.lora_A.default.weight"] = (rank, input_size)
        result[f"{base}.lora_B.default.weight"] = (output_size, rank)
    return result


def validate_trainable_inventory(adapter: Any, pointer_head: Any, config: dict[str, Any]) -> None:
    """Close all 496 adapter and both pointer tensor names/shapes."""
    actual = {
        name: tuple(value.shape)
        for name, value in adapter.named_parameters()
        if value.requires_grad and (".lora_A." in name or ".lora_B." in name)
    }
    if actual != _expected_adapter_shapes(int(config["lora_rank"])) or len(actual) != 496:
        raise ValueError("Qwen LoRA tensor inventory is not the pinned 496 tensors")
    head = {
        name: tuple(value.shape)
        for name, value in pointer_head.named_parameters()
        if value.requires_grad
    }
    if head != {"query.weight": (256, 2560), "key.weight": (256, 2560)}:
        raise ValueError("pointer head tensor inventory is not the pinned two tensors")


def build_qlora_components(
    base_directory: Path, candidate_id: str, *, admission: AdmissionProof | None = None
) -> tuple[Any, Any, dict[str, Any]]:
    """Build frozen-base QLoRA and a trainable 256-wide pointer head from local bytes.

    `local_files_only=True` is intentional: this source never downloads a base
    checkpoint.  Callers must supply a pre-acquired immutable local snapshot.
    """
    if not isinstance(admission, AdmissionProof):
        raise ValueError("private admission must be live-verified before model load")
    if (
        not base_directory.is_absolute()
        or not base_directory.is_dir()
        or base_directory.is_symlink()
    ):
        raise ValueError("base directory must be an absolute non-symlink local directory")
    torch, transformers, peft, _safetensors = _training_runtime()
    config = candidate_config(candidate_id)
    configure_determinism(torch, config["seed"])
    quantization = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    config_loader = getattr(transformers, "AutoConfig", None)
    conditional_model = getattr(transformers, "Qwen3_5ForConditionalGeneration", None)
    if config_loader is None or conditional_model is None:
        raise RuntimeError("v0.2 requires Transformers 5.16.0 with Qwen3.5 support")
    model_config = config_loader.from_pretrained(
        str(base_directory), local_files_only=True, revision=config["base_revision"]
    )
    qwen35_text_config(model_config)
    base = conditional_model.from_pretrained(
        str(base_directory),
        local_files_only=True,
        revision=config["base_revision"],
        config=model_config,
        quantization_config=quantization,
        torch_dtype=torch.bfloat16,
    )
    base.requires_grad_(False)
    text_model = getattr(getattr(base, "model", None), "language_model", None)
    if text_model is None or not _semantic_text_config_matches(
        getattr(text_model, "config", None), qwen35_text_config(model_config)
    ):
        raise ValueError("Qwen3.5 conditional model did not expose its pinned text path")
    lora = peft.get_peft_model(
        text_model,
        peft.LoraConfig(
            r=config["lora_rank"],
            lora_alpha=config["lora_alpha"],
            lora_dropout=config["dropout"],
            bias="none",
            target_modules=list(config["lora_targets"]),
            task_type=peft.TaskType.FEATURE_EXTRACTION,
        ),
    )

    class PointerReadoutHead(torch.nn.Module):  # type: ignore[name-defined, misc]
        def __init__(self, hidden_size: int) -> None:
            super().__init__()
            self.query = torch.nn.Linear(
                hidden_size, config["head_dim"], bias=False, dtype=torch.bfloat16
            )
            self.key = torch.nn.Linear(
                hidden_size, config["head_dim"], bias=False, dtype=torch.bfloat16
            )

        def forward(self, state: Any, options: Any) -> Any:
            # The frozen architecture specifies scaled query/key with temperature 1.0.
            scale = config["head_dim"] ** -0.5
            return (self.query(state).unsqueeze(1) * self.key(options)).sum(dim=-1) * scale

    hidden_size = qwen35_text_config(model_config).hidden_size
    head = PointerReadoutHead(hidden_size)
    validate_trainable_inventory(lora, head, config)
    return lora, head, config


def _score_token_batch(adapter: Any, pointer_head: Any, batch: dict[str, Any]) -> Any:
    """Score a row using exactly one causal forward over state and all options."""
    state = adapter(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    head_dtype = next(pointer_head.parameters()).dtype
    hidden = state.last_hidden_state.to(dtype=head_dtype)
    row_index = batch["row_index"]
    state_vector = hidden[row_index, batch["decide_indices"]]
    option_vectors = hidden[row_index.unsqueeze(1), batch["end_option_indices"]]
    scores = pointer_head(state_vector, option_vectors)
    return scores.masked_fill(~batch["option_present"], float("-inf"))


def _prepared_examples(path: Path) -> list[dict[str, Any]]:
    """Read validated, already-rendered one-forward causal rows only."""
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise ValueError("prepared examples must be an absolute non-symlink local file")
    value = _load_closed(path).get("records")
    if not isinstance(value, list) or not value:
        raise ValueError("prepared examples require a non-empty records list")
    expected = {"split", "input_ids", "decide_index", "end_option_indices", "target", "stratum"}
    parsed: list[dict[str, Any]] = []
    for row in value:
        if (
            not isinstance(row, dict)
            or set(row) != expected
            or row["split"] not in {"train", "internal_dev"}
        ):
            raise ValueError("prepared example is not closed")
        if not isinstance(row["stratum"], str) or not row["stratum"]:
            raise ValueError("prepared example stratum invalid")
        ids, ends, decide, target = (
            row["input_ids"],
            row["end_option_indices"],
            row["decide_index"],
            row["target"],
        )
        if (
            not isinstance(ids, list)
            or not 1 <= len(ids) <= 512
            or any(type(token) is not int or token < 0 for token in ids)
            or not isinstance(ends, list)
            or not 2 <= len(ends) <= 8
            or any(type(index) is not int or not 0 <= index < len(ids) for index in ends)
            or ends != sorted(ends)
            or len(set(ends)) != len(ends)
            or type(decide) is not int
            or not 0 <= decide < len(ids)
            or decide <= ends[-1]
            or type(target) is not int
            or not 0 <= target < len(ends)
        ):
            raise ValueError("prepared causal row is invalid")
        parsed.append(row)
    if {row["split"] for row in parsed} != {"train", "internal_dev"}:
        raise ValueError("prepared examples require train and internal_dev lanes")
    return parsed


def _token_batches(
    torch: Any, records: list[dict[str, Any]], batch_size: int
) -> Iterable[dict[str, Any]]:
    for start in range(0, len(records), batch_size):
        rows = records[start : start + batch_size]
        max_tokens = max(len(row["input_ids"]) for row in rows)
        max_options = max(len(row["end_option_indices"]) for row in rows)
        input_ids = torch.zeros((len(rows), max_tokens), dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        ends = torch.zeros((len(rows), max_options), dtype=torch.long)
        option_present = torch.zeros((len(rows), max_options), dtype=torch.bool)
        targets = torch.tensor([row["target"] for row in rows], dtype=torch.long)
        decide = torch.tensor([row["decide_index"] for row in rows], dtype=torch.long)
        for index, row in enumerate(rows):
            tokens = torch.tensor(row["input_ids"], dtype=torch.long)
            input_ids[index, : len(tokens)] = tokens
            attention_mask[index, : len(tokens)] = True
            for option_index, end in enumerate(row["end_option_indices"]):
                ends[index, option_index] = end
                option_present[index, option_index] = True
        yield {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "decide_indices": decide,
            "end_option_indices": ends,
            "option_present": option_present,
            "targets": targets,
            "row_index": torch.arange(len(rows), dtype=torch.long),
            "strata": [str(row["stratum"]) for row in rows],
        }


def _to_device(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    return {
        key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()
    }


def _adapter_state_on_cpu(peft: Any, adapter: Any) -> dict[str, Any]:
    """Keep only trainable LoRA tensors when remembering the best dev epoch."""
    state = peft.get_peft_model_state_dict(adapter)
    if not state:
        raise ValueError("QLoRA adapter state is empty")
    return {name: value.detach().cpu().clone() for name, value in state.items()}


def _module_state_on_cpu(module: Any) -> dict[str, Any]:
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def train_pointer_candidate(
    adapter: Any,
    pointer_head: Any,
    prepared_examples: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Optimize only LoRA and pointer tensors, selecting the earliest best dev epoch.

    Prepared examples contain already-rendered token IDs.  This keeps corpus
    text, tokenizer acquisition, and model download outside this offline source
    module while still providing the real QLoRA/pointer optimization loop.
    """
    torch, _transformers, peft, _safetensors = _training_runtime()
    configure_determinism(torch, config["seed"])
    if not bool(torch.cuda.is_available()):
        raise RuntimeError("candidate training requires an authorised CUDA worker")
    device = torch.device("cuda")
    adapter.to(device)
    pointer_head.to(device)
    result = run_optimization(
        torch=torch,
        adapter=adapter,
        pointer_head=pointer_head,
        train_rows=[row for row in prepared_examples if row["split"] == "train"],
        dev_rows=[row for row in prepared_examples if row["split"] == "internal_dev"],
        config=config,
        device=device,
        token_batches=_token_batches,
        to_device=_to_device,
        score_batch=_score_token_batch,
        adapter_state=lambda model: _adapter_state_on_cpu(peft, model),
        load_adapter_state=peft.set_peft_model_state_dict,
    )
    if result["interrupted"]:
        raise RuntimeError("production candidate loop unexpectedly interrupted")
    return {
        "best_epoch": result["best_epoch"],
        "internal_dev_stratified_macro_accuracy": result["internal_dev_stratified_macro_accuracy"],
    }


def tiny_causal_cpu_proof(torch: Any) -> dict[str, Any]:
    """Run a deterministic, self-authored CPU-only proof; never a model result.

    It exercises the same one-forward index path, finite optimisation, frozen
    base, and independent best-state retention without acquiring a checkpoint.
    """
    configure_determinism(torch, 20260929)

    class Base(torch.nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.embedding = torch.nn.Embedding(32, 4)
            for parameter in self.parameters():
                parameter.requires_grad_(False)

        def forward(self, input_ids: Any, attention_mask: Any) -> Any:
            del attention_mask
            return type("Output", (), {"last_hidden_state": self.embedding(input_ids)})()

    class Head(torch.nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.query = torch.nn.Linear(4, 2, bias=False)
            self.key = torch.nn.Linear(4, 2, bias=False)

        def forward(self, state: Any, options: Any) -> Any:
            return (self.query(state).unsqueeze(1) * self.key(options)).sum(dim=-1) / (2**0.5)

    base, head = Base(), Head()
    base_before = {name: value.detach().clone() for name, value in base.state_dict().items()}
    rows = [
        {
            "split": "train",
            "input_ids": [1, 2, 3, 4, 5],
            "end_option_indices": [2, 3],
            "decide_index": 4,
            "target": 0,
            "stratum": "a",
        },
        {
            "split": "internal_dev",
            "input_ids": [6, 7, 8, 9, 10],
            "end_option_indices": [2, 3],
            "decide_index": 4,
            "target": 1,
            "stratum": "a",
        },
    ]
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.05, weight_decay=0.01)
    initial = {name: value.detach().clone() for name, value in head.state_dict().items()}
    best_state: dict[str, Any] | None = None
    best_loss = float("inf")
    for _ in range(3):
        batch = next(iter(_token_batches(torch, [rows[0]], 1)))
        loss = torch.nn.functional.cross_entropy(
            _score_token_batch(base, head, batch), batch["targets"]
        )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("test-only tiny causal proof produced a non-finite loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach())
        if loss_value < best_loss:
            best_loss = loss_value
            best_state = {
                name: value.detach().cpu().clone() for name, value in head.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError("test-only tiny causal proof retained no best state")
    head.load_state_dict(best_state)
    if any(not torch.equal(value, base_before[name]) for name, value in base.state_dict().items()):
        raise RuntimeError("test-only tiny causal proof changed frozen base")
    if not any(not torch.equal(value, initial[name]) for name, value in head.state_dict().items()):
        raise RuntimeError("test-only tiny causal proof did not update head")
    return {
        "test_only": True,
        "base_unchanged": True,
        "best_state_cpu": True,
        "finite_updates": True,
    }


def _private_output_directory(path: Path) -> None:
    if not path.is_absolute() or path.is_symlink():
        raise ValueError("artifact output must be a new absolute non-symlink directory")
    if path.exists():
        raise FileExistsError("create-only artifact output already exists")
    if any(ancestor.is_symlink() for ancestor in (path.parent, *path.parent.parents)):
        raise ValueError("artifact output path must not traverse a symlink")
    if not path.parent.is_dir() or stat.S_IMODE(path.parent.stat().st_mode) != 0o700:
        raise ValueError("artifact output parent must be a private existing directory")


def _claim_output_directory(path: Path) -> None:
    """Claim the final name atomically; mkdir cannot replace a competing directory."""
    _private_output_directory(path)
    try:
        path.mkdir(mode=0o700)
    except FileExistsError as exc:
        raise FileExistsError("create-only artifact output already exists") from exc


def _file_digests(root: Path) -> dict[str, str]:
    if not root.is_absolute() or any(ancestor.is_symlink() for ancestor in (root, *root.parents)):
        raise ValueError("artifact path must not traverse a symlink")
    entries = list(root.rglob("*"))
    if any(item.is_symlink() for item in entries):
        raise ValueError("artifact contains symlink")
    files = sorted(item for item in entries if item.is_file())
    return {
        str(item.relative_to(root)): hashlib.sha256(item.read_bytes()).hexdigest() for item in files
    }


def _safetensor_keys(path: Path) -> set[str]:
    """Inspect tensor names without loading tensors or importing a model runtime."""
    try:
        safetensors = importlib.import_module("safetensors")
        safe_open = safetensors.safe_open
        with safe_open(str(path), framework="pt", device="cpu") as reader:
            return set(reader.keys())
    except (ImportError, OSError, ValueError) as exc:
        raise ValueError("candidate safetensor inventory is unreadable") from exc


def _validate_exported_tensor_inventory(output: Path, files: dict[str, str]) -> None:
    allowed = {
        "adapter/adapter_model.safetensors",
        "adapter/adapter_config.json",
        "adapter/README.md",
        "pointer_head.safetensors",
    }
    if (
        not {"adapter/adapter_model.safetensors", "pointer_head.safetensors"} <= set(files)
        or not set(files) <= allowed
    ):
        raise ValueError("candidate artifact file allowlist mismatch")
    adapter_keys = _safetensor_keys(output / "adapter/adapter_model.safetensors")
    if not adapter_keys or any(
        "lora_" not in key.lower()
        or "kev" in key.lower()
        or "embed" in key.lower()
        or "pointer" in key.lower()
        for key in adapter_keys
    ):
        raise ValueError("candidate adapter tensor allowlist mismatch")
    pointer_keys = _safetensor_keys(output / "pointer_head.safetensors")
    if pointer_keys != {"pointer_head.query.weight", "pointer_head.key.weight"}:
        raise ValueError("candidate pointer tensor allowlist mismatch")


def export_candidate_artifact(
    adapter: Any, pointer_head: Any, output: Path, config: dict[str, Any]
) -> dict[str, Any]:
    """Atomically create a LoRA-plus-pointer artifact and never export base tensors."""
    _claim_output_directory(output)
    _torch, _transformers, _peft, safetensors = _training_runtime()
    try:
        adapter_path = output / "adapter"
        if not callable(getattr(adapter, "save_pretrained", None)):
            raise ValueError("QLoRA adapter cannot be safely exported")
        adapter.save_pretrained(str(adapter_path), safe_serialization=True)
        tensors = {
            f"pointer_head.{name}": value.detach().cpu().contiguous()
            for name, value in pointer_head.state_dict().items()
        }
        if not tensors or any(not name.startswith("pointer_head.") for name in tensors):
            raise ValueError("pointer head export is invalid")
        safetensors.save_file(tensors, str(output / "pointer_head.safetensors"))
        files = _file_digests(output)
        _validate_exported_tensor_inventory(output, files)
        descriptor = {
            "schema_version": "v02-candidate-artifact.v2",
            "candidate_id": config["id"],
            "base_model": config["base_model"],
            "base_revision": config["base_revision"],
            "adapter_files_sha256": files,
            "base_tensors_exported": False,
        }
        create_json(output / "artifact.json", descriptor)
    except BaseException:
        with suppress(FileNotFoundError):
            for item in sorted(output.rglob("*"), reverse=True):
                if item.is_file() or item.is_symlink():
                    item.unlink()
                elif item.is_dir():
                    item.rmdir()
            output.rmdir()
        raise
    return descriptor


def verify_candidate_artifact(descriptor: dict[str, Any], output: Path) -> None:
    expected = {
        "schema_version",
        "candidate_id",
        "base_model",
        "base_revision",
        "adapter_files_sha256",
        "base_tensors_exported",
    }
    manifest = load_manifest()["architecture"]
    if (
        set(descriptor) != expected
        or descriptor["schema_version"] != "v02-candidate-artifact.v2"
        or descriptor["candidate_id"] not in {item["id"] for item in CANDIDATE_GRID}
        or descriptor["base_model"] != manifest["base_model"]
        or descriptor["base_revision"] != manifest["base_revision"]
        or descriptor["base_tensors_exported"] is not False
        or not isinstance(descriptor["adapter_files_sha256"], dict)
    ):
        raise ValueError("candidate artifact descriptor drifted")
    expected_files = descriptor["adapter_files_sha256"]
    if not expected_files or any(
        not isinstance(name, str) or not HEX64.fullmatch(digest)
        for name, digest in expected_files.items()
    ):
        raise ValueError("candidate artifact file digest invalid")
    if (
        not output.is_absolute()
        or not output.is_dir()
        or any(ancestor.is_symlink() for ancestor in (output, *output.parents))
    ):
        raise ValueError("candidate artifact directory invalid")
    actual = _file_digests(output)
    actual.pop("artifact.json", None)
    if actual != expected_files or "pointer_head.safetensors" not in actual:
        raise ValueError("candidate artifact digest mismatch")
    _validate_exported_tensor_inventory(output, actual)


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight")
    check = sub.add_parser("validate-readiness")
    for name in ("training", "sealed", "readiness", "receipt"):
        check.add_argument(f"--{name}", type=Path, required=True)
    train = sub.add_parser("train-candidate")
    train.add_argument("--base-directory", type=Path, required=True)
    train.add_argument("--prepared-examples", type=Path, required=True)
    train.add_argument(
        "--candidate", choices=[item["id"] for item in CANDIDATE_GRID], required=True
    )
    train.add_argument("--output", type=Path, required=True)
    select = sub.add_parser("select")
    select.add_argument("--reports", type=Path, required=True)
    verify = sub.add_parser("verify-artifact")
    verify.add_argument("--artifact-directory", type=Path, required=True)
    for name in ("seal-readiness", "smoke", "evaluate"):
        sub.add_parser(name)
    args = parser.parse_args()
    if args.command == "preflight":
        load_manifest()
        print("preflight valid")
        return 0
    if args.command == "validate-readiness":
        return validate_readiness(args.training, args.sealed, args.readiness, args.receipt)
    if args.command == "select":
        reports = _load_closed(args.reports).get("candidates")
        if not isinstance(reports, list):
            raise ValueError("candidate reports must be a closed candidates list")
        print(canonical_json_bytes(cast(JsonValue, select_candidate(reports))).decode("utf-8"))
        return 0
    if args.command == "verify-artifact":
        descriptor = _load_closed(args.artifact_directory / "artifact.json")
        verify_candidate_artifact(descriptor, args.artifact_directory)
        print("candidate artifact valid")
        return 0
    if args.command == "train-candidate":
        adapter, pointer_head, config = build_qlora_components(args.base_directory, args.candidate)
        outcome = train_pointer_candidate(
            adapter, pointer_head, _prepared_examples(args.prepared_examples), config
        )
        descriptor = export_candidate_artifact(adapter, pointer_head, args.output, config)
        print(canonical_json_bytes({"artifact": descriptor, "training": outcome}).decode("utf-8"))
        return 0
    raise ValueError(
        f"{args.command} requires the separately authorised live-phase command contract"
    )


if __name__ == "__main__":
    raise SystemExit(main())
