"""Offline-only machinery for the first Saracura-owned checkpoint.

This module intentionally has no model import.  It seals aggregate evidence,
defines the immutable candidate grid, and validates selection evidence.  Model
loading is deferred to the separately authorised live phases.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import tempfile
from collections.abc import Iterable
from contextlib import suppress
from pathlib import Path
from typing import Any

from benchmarks.v02_corpus import RENDERER_CONTRACT
from benchmarks.v02_evaluation import (
    _normalise,
    _option_description,
    combined_content_fingerprint,
    state_question_fingerprint,
)
from saracura.serialization import canonical_json_bytes

MANIFEST_PATH = Path(__file__).parent / "manifests/v02-first-checkpoint.v1.json"
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
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
    if not parent.is_dir() or parent.is_symlink() or stat.S_IMODE(parent.stat().st_mode) != 0o700:
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
        or not all(isinstance(item, str) and item for item in source_policy_identifiers)
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
        "source_policy_identifiers": sorted(set(source_policy_identifiers)),
    }


def validate_training_descriptor(value: dict[str, Any]) -> None:
    if (
        set(value) != TRAINING_DESCRIPTOR_FIELDS
        or value.get("schema_version") != "v02-training-descriptor.v1"
    ):
        raise ValueError("training descriptor is not closed")
    if (
        not isinstance(value["record_count"], int)
        or value["record_count"] <= 0
        or value["split_counts"]
        != {
            "train": value["split_counts"].get("train"),
            "internal_dev": value["split_counts"].get("internal_dev"),
        }
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


def diagnostic_subset(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(records) < 50:
        raise ValueError("public development lane must contain at least 50 records")
    return sorted(
        records,
        key=lambda row: hashlib.sha256(
            b"saracura-v02-public-dev-order-v1"
            + len(row["opaque_record_id"].encode()).to_bytes(8, "big")
            + row["opaque_record_id"].encode()
        ).digest(),
    )[:50]


def select_candidate(candidates: list[dict[str, Any]]) -> dict[str, str]:
    """Apply the frozen development-only rule without receiving a held-out path."""
    if len(candidates) != 3 or {row.get("id") for row in candidates} != {
        item["id"] for item in CANDIDATE_GRID
    }:
        raise ValueError("selection requires exactly the frozen three candidates")
    eligible: list[dict[str, Any]] = []
    for row in candidates:
        if set(row) != {
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
        if not all(
            isinstance(row[key], (int, float)) and not isinstance(row[key], bool)
            for key in (
                "coverage",
                "invalid_output_rate",
                "repeat_stability",
                "option_order_stability",
                "planned_top1_accuracy",
                "warm_p95_ms",
            )
        ):
            raise ValueError("candidate metric invalid")
        if (
            row["valid"] is True
            and row["truncation_detected"] is False
            and row["identity_match"] is True
            and row["coverage"] >= 0.98
            and row["invalid_output_rate"] == 0
            and row["repeat_stability"] == 1
            and row["option_order_stability"] >= 0.95
            and row["planned_top1_accuracy"] >= 0.4
        ):
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


def loopback_worker_plan() -> dict[str, Any]:
    """Closed configuration shared by candidate and Kev loopback workers.

    The actual model process is intentionally deferred to the authorised GPU
    phase; this plan is the offline proof that neither side can choose a
    different endpoint, batching, or measurement procedure.
    """
    evaluation = load_manifest()["evaluation"]
    assert isinstance(evaluation, dict)
    return dict(evaluation)


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
        value["training_descriptor_digest"] != hashlib.sha256(training.read_bytes()).hexdigest()
        or value["sealed_descriptor_digest"] != hashlib.sha256(sealed.read_bytes()).hexdigest()
        or value["disjointness_receipt_digest"] != hashlib.sha256(receipt.read_bytes()).hexdigest()
        or value["candidate_rendering_digest"] != RENDERER_CONTRACT["candidate_rendering_digest"]
        or value["all_records_candidate_preflight_passed"] is not True
        or value["all_records_kev_preflight_passed"] is not True
    ):
        raise ValueError("readiness bindings invalid")
    print("readiness valid")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight")
    check = sub.add_parser("validate-readiness")
    for name in ("training", "sealed", "readiness", "receipt"):
        check.add_argument(f"--{name}", type=Path, required=True)
    for name in (
        "seal-readiness",
        "smoke",
        "train-candidate",
        "select",
        "evaluate",
        "verify-artifact",
    ):
        sub.add_parser(name)
    args = parser.parse_args()
    if args.command == "preflight":
        load_manifest()
        print("preflight valid")
        return 0
    if args.command == "validate-readiness":
        return validate_readiness(args.training, args.sealed, args.readiness, args.receipt)
    raise ValueError(
        f"{args.command} requires the separately authorised live-phase command contract"
    )


if __name__ == "__main__":
    raise SystemExit(main())
