"""Offline, aggregate-only validator for the v0.2 evaluation protocol."""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import sys
import unicodedata
from pathlib import Path, PurePosixPath
from typing import Any

from saracura.serialization import canonical_json_bytes

HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
PHASE5D_MANIFEST_PATH = Path(__file__).parent / "manifests/phase5d-ptbr-native.v1.json"
PHASE5D_MANIFEST_SHA256 = "af983ce181018a2f6c5abec351ce1b78e875764616875aed3f59aa416a499ff6"
PROTOCOL_SPEC_PATH = (
    Path(__file__).parent.parent / "docs/action/specs/v02-model-evaluation-protocol.md"
)
PROTOCOL_SPEC_SHA256 = "035447830c76c302cd5da6ca0560be64ffa98d10bc1649c7f710d08cfd7f2621"
PROTOCOL_MANIFEST_SHA256 = "68cd2aa62165029890c13d521bac52ad8ee426604427d16aa93a8e2f6f5453e8"

DESCRIPTOR_FIELDS = frozenset(
    {
        "schema_version",
        "record_count",
        "payload_digest",
        "identity_set_digest",
        "state_question_fingerprint_set_digest",
        "combined_content_fingerprint_set_digest",
        "descriptive_option_multiset_fingerprint_set_digest",
        "slice_counts",
        "option_count_distribution",
        "gold_position_distribution",
        "source_class",
        "contamination_risk",
        "max_rendered_input_tokens_observed",
        "kev_base_model_revision",
        "kev_adapter_revision",
        "kev_rendering_function_digest",
        "truncation_disabled",
        "all_records_kev_preflight_passed",
        "permutation_subset_size",
        "permutation_selection_seed",
        "permutation_selection_digest",
        "permutation_strategy",
    }
)
RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "training_descriptor_digest",
        "sealed_descriptor_digest",
        "training_identity_digest",
        "public_dev_identity_digest",
        "sealed_test_identity_digest",
        "training_state_question_digest",
        "public_dev_state_question_digest",
        "sealed_test_state_question_digest",
        "training_combined_content_digest",
        "public_dev_combined_content_digest",
        "sealed_test_combined_content_digest",
        "training_option_multiset_digest",
        "public_dev_option_multiset_digest",
        "sealed_test_option_multiset_digest",
        "pairwise_intersection_counts",
    }
)
SELECTION_FIELDS = frozenset(
    {
        "schema_version",
        "candidate",
        "training_capsule",
        "code_revision",
        "hyperparameters",
        "seed",
        "development_report_digest",
        "sealed_descriptor_digest",
        "descriptor_commit",
        "descriptor_path",
        "selection_commit",
        "candidate_deployment_class",
    }
)
REPORT_FIELDS = frozenset(
    {
        "schema_version",
        "protocol_digest",
        "sealed_descriptor_digest",
        "disjointness_receipt_digest",
        "selection_receipt_digest",
        "planned_denominator",
        "quality_settings",
        "model_rows",
        "paired_comparison_counters",
        "slice_aggregates",
        "systems_measurements",
        "limitations",
        "option_order_stability",
        "deterministic_repeat_stability",
        "claim",
    }
)
PAIR_KEYS = frozenset({"training_public_dev", "training_sealed_test", "public_dev_sealed_test"})
REQUIRED_MODELS = ("saracura", "kev", "laya", "julia")
CONTROL_REVISIONS = {
    "kev": "139fdd94f1b6a6ad80cc15e08fcb99cac885a101",
    "laya": "052592a15d198d9ad47da779604259b10b47b7aa",
    "julia": "a85b127321d580d65176c89ced8273f305745d85",
}
QUALITY_SETTINGS_FIELDS = frozenset(
    {"task", "locale", "preprocessing", "scoring", "planned_denominator", "host_class"}
)
AVAILABLE_ROW_FIELDS = frozenset(
    {
        "status",
        "model_id",
        "model_revision",
        "tokenizer_revision",
        "code_revision",
        "quality_settings",
        "counts",
        "metrics",
        "input_truncation_detected",
        "training_overlap_disclosure",
        "systems_comparability_group",
    }
)
UNAVAILABLE_ROW_FIELDS = frozenset({"status", "reason", "training_overlap_disclosure"})
COUNTS_FIELDS = frozenset({"correct", "valid", "invalid", "errors"})
METRICS_FIELDS = frozenset(
    {
        "planned_top1_accuracy",
        "coverage",
        "valid_only_top1_accuracy",
        "macro_f1_by_option_position",
        "invalid_output_rate",
        "error_rate",
    }
)
SYSTEM_GROUP_FIELDS = frozenset(
    {
        "host_class",
        "runtime_boundary",
        "device_class",
        "precision",
        "batch",
        "concurrency",
        "timing_procedure",
    }
)
SYSTEM_METRIC_FIELDS = frozenset(
    {
        "cold_load_time_ms",
        "warm_request_p50_ms",
        "warm_request_p95_ms",
        "decisions_per_second",
        "peak_host_rss_bytes",
        "peak_device_memory_bytes",
        "artifact_bytes",
    }
)
SLICE_DIMENSIONS = frozenset({"domain", "option_count", "state_length_bucket", "gold_position"})
SLICE_RESULT_FIELDS = frozenset({"planned", "correct", "accuracy"})
OPTION_ORDER_STABILITY_FIELDS = frozenset(
    {
        "subset_size",
        "selection_seed",
        "selection_digest",
        "strategy",
        "stable",
        "evaluated",
        "stability_rate",
    }
)
REPEAT_STABILITY_FIELDS = frozenset(
    {
        "subset_size",
        "selection_seed",
        "selection_digest",
        "repetitions",
        "stable",
        "evaluated",
        "stability_rate",
    }
)
OPAQUE_ID = re.compile(r"^[a-z0-9][a-z0-9._:/-]{0,127}$")
ALLOWED_LIMITATIONS = frozenset(
    {
        "no_calibration_or_automation",
        "public_dev_four_options_only",
        "control_unavailable",
        "slice_denominator_below_30_descriptive_only",
        "contamination_unknown",
    }
)
FORBIDDEN_REPORT_KEYS = frozenset(
    {
        "raw_content",
        "record_id",
        "record_ids",
        "record_identifier",
        "record_identifiers",
        "absolute_path",
        "absolute_paths",
        "exception",
        "exception_message",
        "exception_messages",
        "calibration",
        "calibration_metrics",
        "confidence_thresholds",
        "risk_coverage_curves",
        "automation_allowed",
        "automation_authorization",
        "automation_authorized",
        "automation_gates",
    }
)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key: {key}")
        result[key] = value
    return result


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path}: root must be an object")
    return value


def _closed(value: Any, fields: frozenset[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{label} shape is not closed")
    return value


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or not HEX64.fullmatch(value):
        raise ValueError(f"{label} must be a sha256 digest")
    return value


def _normalise(value: Any) -> Any:
    if isinstance(value, str):
        return " ".join(unicodedata.normalize("NFC", value).casefold().strip().split())
    if isinstance(value, list):
        return [_normalise(item) for item in value]
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("combined content keys must be strings")
            normal_key = _normalise(key)
            if normal_key in result:
                raise ValueError("combined content has post-normalization key collision")
            result[normal_key] = _normalise(item)
        return result
    if value is None or isinstance(value, (bool, int, float)):
        return value
    raise ValueError("combined content contains unsupported value")


def _canonical_json(value: Any) -> bytes:
    """Serialize normalized record content with the project's RFC 8785 implementation."""
    return canonical_json_bytes(value)


def _option_description(option: Any) -> Any:
    if not isinstance(option, dict):
        return option
    return {key: value for key, value in option.items() if key not in {"id", "option_id"}}


def combined_content_canonical_bytes(state: Any, question: Any, options: list[Any]) -> bytes:
    """Canonical order/ID-invariant bytes used by the sealed combined fingerprint."""
    if not isinstance(options, list):
        raise ValueError("combined content options must be a list")
    option_bytes = sorted(
        _canonical_json(_normalise(_option_description(option))) for option in options
    )
    return _canonical_json(
        {
            "state": _normalise(state),
            "question": _normalise(question),
            "options": [json.loads(item) for item in option_bytes],
        }
    )


def combined_content_fingerprint(state: Any, question: Any, options: list[Any]) -> str:
    return hashlib.sha256(combined_content_canonical_bytes(state, question, options)).hexdigest()


def _validate_manifest_schema(manifest: dict[str, Any]) -> None:
    fields = frozenset(
        {
            "schema_version",
            "protocol_digest",
            "development_lane",
            "lanes",
            "controls",
            "kev_rendering_contract",
            "metrics",
            "comparability_groups",
            "descriptor_schema",
            "disjointness_receipt_schema",
            "selection_receipt_schema",
            "report_schema",
            "prohibited_claims",
            "task",
            "selection",
            "held_out_payload",
        }
    )
    _closed(manifest, fields, "v02 protocol manifest")
    if (
        manifest["schema_version"] != "v02-model-evaluation-protocol.v1"
        or manifest["protocol_digest"] != PROTOCOL_SPEC_SHA256
    ):
        raise ValueError("protocol manifest version or digest mismatch")
    dev = _closed(
        manifest["development_lane"],
        frozenset(
            {
                "bound_manifest",
                "bound_manifest_sha256",
                "planned_records",
                "planned_choice_count",
                "candidate_capability",
                "presented_as_held_out",
            }
        ),
        "development_lane",
    )
    if (
        dev["bound_manifest"] != "benchmarks/manifests/phase5d-ptbr-native.v1.json"
        or dev["bound_manifest_sha256"] != PHASE5D_MANIFEST_SHA256
        or dev["presented_as_held_out"] is not False
    ):
        raise ValueError("development lane held out binding mismatch")
    schemas = (
        ("descriptor_schema", "v02-sealed-descriptor.v1", DESCRIPTOR_FIELDS),
        ("disjointness_receipt_schema", "v02-disjointness-receipt.v1", RECEIPT_FIELDS),
        ("selection_receipt_schema", "v02-selection-receipt.v1", SELECTION_FIELDS),
        ("report_schema", "saracura-v02-evaluation-report.v1", REPORT_FIELDS),
    )
    for name, version, schema_fields in schemas:
        schema = _closed(
            manifest[name],
            frozenset({"schema_version", "required_fields"})
            if name != "disjointness_receipt_schema"
            else frozenset({"schema_version", "required_fields", "pairwise_intersection_counts"}),
            name,
        )
        if (
            schema["schema_version"] != version
            or not isinstance(schema["required_fields"], list)
            or frozenset(schema["required_fields"]) != schema_fields
            or len(schema["required_fields"]) != len(schema_fields)
        ):
            raise ValueError(f"{name} is not an exact closed schema")
    if manifest["disjointness_receipt_schema"]["pairwise_intersection_counts"] != {
        "zero_required": ["identity", "state_question", "combined_content"],
        "descriptive_only": ["option_multiset"],
    }:
        raise ValueError("disjointness receipt schema pairwise policy mismatch")
    controls = manifest["controls"]
    if not isinstance(controls, list):
        raise ValueError("controls must be a list")
    by_id = {item.get("id"): item for item in controls if isinstance(item, dict)}
    if set(by_id) != {"laya-multilingual", "julia-1-cyclic-mean", "kev-4b", "lexical-baseline"}:
        raise ValueError("unknown control or missing frozen control")
    for row_name, control_id in (
        ("laya", "laya-multilingual"),
        ("julia", "julia-1-cyclic-mean"),
        ("kev", "kev-4b"),
    ):
        if by_id[control_id].get("revision") != CONTROL_REVISIONS[row_name]:
            raise ValueError(f"control {control_id} revision is not pinned")
    kev = manifest["kev_rendering_contract"]
    if not isinstance(kev, dict) or kev.get("max_rendered_input_tokens") != 512:
        raise ValueError("Kev rendering contract must freeze 512 tokens")
    prohibited = manifest["prohibited_claims"]
    if not isinstance(prohibited, list) or "calibration_metrics" not in prohibited:
        raise ValueError("calibration_metrics must be prohibited")


def validate_protocol() -> int:
    path = Path(__file__).parent / "manifests/v02-model-evaluation-protocol.v1.json"
    return validate_protocol_manifest(path)


def validate_protocol_manifest(path: Path) -> int:
    manifest = _load(path)
    _validate_manifest_schema(manifest)
    if _sha(path) != PROTOCOL_MANIFEST_SHA256:
        raise ValueError("v0.2 protocol manifest bytes drifted")
    if (
        _sha(PHASE5D_MANIFEST_PATH) != PHASE5D_MANIFEST_SHA256
        or _sha(PROTOCOL_SPEC_PATH) != PROTOCOL_SPEC_SHA256
    ):
        raise ValueError("bound protocol bytes drifted")
    sealed = manifest["lanes"].get("sealed_ptbr_test")
    if (
        not isinstance(sealed, dict)
        or sealed.get("minimum_records") != 100
        or sealed.get("minimum_records_per_option_count") != 14
        or set(sealed.get("option_counts_covered", [])) != set(range(2, 9))
        or sealed.get("gold_position_balance_per_option_count") is not True
    ):
        raise ValueError("sealed lane requirements mismatch")
    controls = manifest["controls"]
    if not isinstance(controls, list) or {
        item.get("id") for item in controls if isinstance(item, dict)
    } != {"laya-multilingual", "julia-1-cyclic-mean", "kev-4b", "lexical-baseline"}:
        raise ValueError("unknown control or missing frozen control")
    kev = manifest["kev_rendering_contract"]
    if (
        not isinstance(kev, dict)
        or kev.get("max_rendered_input_tokens") != 512
        or kev.get("truncation_disabled") is not True
        or not isinstance(kev.get("rendering_function_digest"), str)
        or not HEX64.fullmatch(kev["rendering_function_digest"])
    ):
        raise ValueError("Kev rendering contract mismatch")
    print("protocol valid")
    return 0


def _validate_descriptor(descriptor: dict[str, Any]) -> None:
    _closed(descriptor, DESCRIPTOR_FIELDS, "sealed descriptor")
    if (
        descriptor["schema_version"] != "v02-sealed-descriptor.v1"
        or not isinstance(descriptor["record_count"], int)
        or isinstance(descriptor["record_count"], bool)
        or descriptor["record_count"] < 100
    ):
        raise ValueError("sealed descriptor record_count must be at least 100")
    if descriptor["source_class"] not in {
        "public_source_derivative",
        "original_contributed",
        "mixed",
    } or descriptor["contamination_risk"] not in {
        "unknown",
        "possible",
        "controlled_no_known_overlap",
    }:
        raise ValueError("sealed descriptor source class or contamination risk is invalid")
    if (
        not isinstance(descriptor["max_rendered_input_tokens_observed"], int)
        or isinstance(descriptor["max_rendered_input_tokens_observed"], bool)
        or descriptor["max_rendered_input_tokens_observed"] < 0
        or descriptor["max_rendered_input_tokens_observed"] > 512
        or descriptor["kev_base_model_revision"] != "1001bb4d826a52d1f399e183466143f4da7b741b"
        or descriptor["kev_adapter_revision"] != CONTROL_REVISIONS["kev"]
        or descriptor["kev_rendering_function_digest"]
        != "9f42035579e68f6c0e535df2e107b189314b9c93f3899b442855a9dd4e6a9c66"
        or descriptor["truncation_disabled"] is not True
        or descriptor["all_records_kev_preflight_passed"] is not True
    ):
        raise ValueError("sealed descriptor does not prove the frozen Kev token envelope")
    if (
        not isinstance(descriptor["permutation_subset_size"], int)
        or isinstance(descriptor["permutation_subset_size"], bool)
        or not 50 <= descriptor["permutation_subset_size"] <= descriptor["record_count"]
        or not isinstance(descriptor["permutation_selection_seed"], str)
        or not OPAQUE_ID.fullmatch(descriptor["permutation_selection_seed"])
        or not isinstance(descriptor["permutation_selection_digest"], str)
        or not HEX64.fullmatch(descriptor["permutation_selection_digest"])
        or descriptor["permutation_strategy"] != "cyclic_left_rotation_one"
    ):
        raise ValueError("sealed descriptor permutation subset is invalid")
    for name in (
        "payload_digest",
        "identity_set_digest",
        "state_question_fingerprint_set_digest",
        "combined_content_fingerprint_set_digest",
        "descriptive_option_multiset_fingerprint_set_digest",
    ):
        _digest(descriptor[name], name)
    distribution = descriptor["option_count_distribution"]
    if not isinstance(distribution, dict) or {str(key) for key in distribution} != {
        str(n) for n in range(2, 9)
    }:
        raise ValueError("descriptor option_count_distribution must cover 2 through 8")
    for count in range(2, 9):
        value = distribution.get(str(count), distribution.get(count))
        if not isinstance(value, int) or isinstance(value, bool) or value < 14:
            raise ValueError("descriptor each option count must have at least 14 records")
    distribution_total = sum(
        distribution[str(n)] if str(n) in distribution else distribution[n] for n in range(2, 9)
    )
    if distribution_total != descriptor["record_count"]:
        raise ValueError("descriptor option counts must equal record_count")
    gold = descriptor["gold_position_distribution"]
    if not isinstance(gold, dict) or {str(key) for key in gold} != {str(n) for n in range(2, 9)}:
        raise ValueError("descriptor gold positions must be keyed by option count")
    for count in range(2, 9):
        positions = gold.get(str(count), gold.get(count))
        if not isinstance(positions, dict) or {str(key) for key in positions} != {
            str(n) for n in range(count)
        }:
            raise ValueError("descriptor gold positions are incomplete")
        values = list(positions.values())
        if (
            not all(
                isinstance(value, int) and not isinstance(value, bool) and value >= 0
                for value in values
            )
            or max(values) - min(values) > 1
            or sum(values) != distribution.get(str(count), distribution.get(count))
        ):
            raise ValueError("descriptor gold positions are not balanced")
    if not isinstance(descriptor["slice_counts"], dict):
        raise ValueError("descriptor slice_counts must be an object")


def _validate_receipt(
    receipt: dict[str, Any], descriptor: dict[str, Any], descriptor_digest: str
) -> None:
    _closed(receipt, RECEIPT_FIELDS, "disjointness receipt")
    if receipt["schema_version"] != "v02-disjointness-receipt.v1":
        raise ValueError("disjointness receipt schema_version mismatch")
    for field in RECEIPT_FIELDS - {"schema_version", "pairwise_intersection_counts"}:
        _digest(receipt[field], field)
    if receipt["sealed_descriptor_digest"] != descriptor_digest:
        raise ValueError("receipt sealed descriptor digest mismatch")
    bindings = {
        "sealed_test_identity_digest": "identity_set_digest",
        "sealed_test_state_question_digest": "state_question_fingerprint_set_digest",
        "sealed_test_combined_content_digest": "combined_content_fingerprint_set_digest",
        "sealed_test_option_multiset_digest": "descriptive_option_multiset_fingerprint_set_digest",
    }
    for receipt_field, descriptor_field in bindings.items():
        if receipt[receipt_field] != descriptor[descriptor_field]:
            raise ValueError(f"receipt {receipt_field} does not bind descriptor")
    counts = _closed(
        receipt["pairwise_intersection_counts"],
        frozenset({"identity", "state_question", "combined_content", "option_multiset"}),
        "pairwise intersection counts",
    )
    for kind, value in counts.items():
        values = _closed(value, PAIR_KEYS, f"{kind} pairwise counts")
        if not all(
            isinstance(number, int) and not isinstance(number, bool) and number >= 0
            for number in values.values()
        ):
            raise ValueError("pairwise intersection counts must be non-negative integers")
        if kind != "option_multiset" and any(values.values()):
            raise ValueError(f"{kind} pairwise intersections must all be zero")


def _git(root: Path, *args: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(root), *args], check=True, text=True, capture_output=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("full repository Git proof unavailable") from exc


def _validate_selection(
    selection: dict[str, Any], descriptor_digest: str, repository: Path
) -> None:
    _closed(selection, SELECTION_FIELDS, "selection receipt")
    if (
        selection["schema_version"] != "v02-selection-receipt.v1"
        or selection["sealed_descriptor_digest"] != descriptor_digest
    ):
        raise ValueError("selection receipt schema or descriptor digest mismatch")
    if (
        not isinstance(selection["candidate_deployment_class"], str)
        or not selection["candidate_deployment_class"]
    ):
        raise ValueError("selection candidate deployment class is required")
    candidate = _closed(
        selection["candidate"],
        frozenset({"id", "checkpoint_revision", "tokenizer_revision"}),
        "selection candidate",
    )
    if not all(isinstance(candidate[key], str) and candidate[key] for key in candidate):
        raise ValueError("selection candidate is incomplete")
    if (
        not OPAQUE_ID.fullmatch(candidate["id"])
        or not (
            HEX40.fullmatch(candidate["checkpoint_revision"])
            or HEX64.fullmatch(candidate["checkpoint_revision"])
        )
        or not (
            HEX40.fullmatch(candidate["tokenizer_revision"])
            or HEX64.fullmatch(candidate["tokenizer_revision"])
        )
    ):
        raise ValueError("selection candidate identifiers must be bounded opaque values")
    if not all(
        isinstance(selection[key], str) and HEX40.fullmatch(selection[key])
        for key in ("code_revision", "descriptor_commit", "selection_commit")
    ):
        raise ValueError("selection Git revisions must be full SHA-1 commits")
    path = selection["descriptor_path"]
    if (
        not isinstance(path, str)
        or not path
        or path.startswith("/")
        or ".." in PurePosixPath(path).parts
        or PurePosixPath(path).as_posix() != path
    ):
        raise ValueError("selection descriptor path must be safe repository-relative")
    if _git(repository, "rev-parse", "--is-shallow-repository") == "true":
        raise ValueError("shallow repository cannot prove selection ancestry")
    _git(repository, "cat-file", "-e", f"{selection['descriptor_commit']}^{{commit}}")
    _git(repository, "cat-file", "-e", f"{selection['selection_commit']}^{{commit}}")
    if (
        subprocess.run(
            [
                "git",
                "-C",
                str(repository),
                "merge-base",
                "--is-ancestor",
                selection["descriptor_commit"],
                selection["selection_commit"],
            ]
        ).returncode
        != 0
    ):
        raise ValueError("descriptor commit is not an ancestor of selection commit")
    blob = _git(repository, "rev-parse", f"{selection['descriptor_commit']}:{path}")
    if not HEX40.fullmatch(blob) or _git(repository, "cat-file", "-t", blob) != "blob":
        raise ValueError("selection descriptor path does not resolve to a blob")
    content = subprocess.run(
        ["git", "-C", str(repository), "show", f"{selection['descriptor_commit']}:{path}"],
        check=True,
        capture_output=True,
    ).stdout
    if hashlib.sha256(content).hexdigest() != descriptor_digest:
        raise ValueError("descriptor blob digest mismatch")


def _ratio(numerator: int, denominator: int, label: str) -> float:
    if (
        not isinstance(numerator, int)
        or isinstance(numerator, bool)
        or numerator < 0
        or numerator > denominator
    ):
        raise ValueError(f"{label} count invalid")
    return numerator / denominator


def _validate_row(
    name: str,
    row: Any,
    denominator: int,
    quality_settings: dict[str, Any],
    selection: dict[str, Any],
) -> bool:
    if not isinstance(row, dict) or row.get("status") not in {"available", "unavailable"}:
        raise ValueError(f"model row {name} must explicitly state availability")
    if row["status"] == "unavailable":
        _closed(row, UNAVAILABLE_ROW_FIELDS, f"unavailable {name} row")
        if row["reason"] not in {
            "not_installed",
            "unsupported_hardware",
            "execution_failure",
            "policy_forbidden",
        } or row["training_overlap_disclosure"] not in {"none", "unknown", "declared_overlap"}:
            raise ValueError(f"unavailable {name} row is invalid")
        return False
    _closed(row, AVAILABLE_ROW_FIELDS, f"available {name} row")
    if (
        row["training_overlap_disclosure"] not in {"none", "unknown", "declared_overlap"}
        or row["input_truncation_detected"] != 0
        or row["quality_settings"] != quality_settings
    ):
        raise ValueError(
            f"available {name} row disclosure, truncation, or quality settings invalid"
        )
    counts = _closed(row["counts"], COUNTS_FIELDS, f"{name} counts")
    correct, valid, invalid, errors = (
        _ratio(counts[key], denominator, f"{name} {key}")
        for key in ("correct", "valid", "invalid", "errors")
    )
    if counts["valid"] + counts["invalid"] + counts["errors"] != denominator:
        raise ValueError(f"{name} counts must reconcile to denominator")
    if counts["correct"] > counts["valid"]:
        raise ValueError(f"{name} correct count cannot exceed valid outputs")
    metrics = _closed(row["metrics"], METRICS_FIELDS, f"{name} metrics")
    expected = {
        "planned_top1_accuracy": correct,
        "coverage": valid,
        "valid_only_top1_accuracy": (
            counts["correct"] / counts["valid"] if counts["valid"] else 0.0
        ),
        "invalid_output_rate": invalid,
        "error_rate": errors,
    }
    for metric, value in expected.items():
        if (
            not isinstance(metrics[metric], (int, float))
            or isinstance(metrics[metric], bool)
            or not math.isclose(metrics[metric], value, rel_tol=0.0, abs_tol=1e-12)
        ):
            raise ValueError(f"{name} {metric} does not match counts")
    macro_f1 = metrics["macro_f1_by_option_position"]
    if (
        not isinstance(macro_f1, (int, float))
        or isinstance(macro_f1, bool)
        or not math.isfinite(macro_f1)
        or not 0.0 <= macro_f1 <= 1.0
    ):
        raise ValueError(f"{name} macro F1 must be numeric")
    if name == "saracura":
        candidate = selection["candidate"]
        if (
            row["model_id"] != candidate["id"]
            or row["model_revision"] != candidate["checkpoint_revision"]
            or row["tokenizer_revision"] != candidate["tokenizer_revision"]
            or row["code_revision"] != selection["code_revision"]
        ):
            raise ValueError("Saracura row does not bind selection candidate")
    else:
        expected_ids = {
            "kev": "kev-4b",
            "laya": "laya-multilingual",
            "julia": "julia-1-cyclic-mean",
        }
        if (
            row["model_id"] != expected_ids[name]
            or row["model_revision"] != CONTROL_REVISIONS[name]
        ):
            raise ValueError(f"{name} row identity or revision is not pinned")
    if (
        not isinstance(row["model_id"], str)
        or not OPAQUE_ID.fullmatch(row["model_id"])
        or not isinstance(row["tokenizer_revision"], str)
        or not (
            HEX40.fullmatch(row["tokenizer_revision"]) or HEX64.fullmatch(row["tokenizer_revision"])
        )
        or not isinstance(row["code_revision"], str)
        or not (HEX40.fullmatch(row["code_revision"]) or HEX64.fullmatch(row["code_revision"]))
        or not isinstance(row["systems_comparability_group"], str)
        or not OPAQUE_ID.fullmatch(row["systems_comparability_group"])
    ):
        raise ValueError(f"{name} row identifiers must be bounded opaque values")
    return True


def _mcnemar_one_sided(ccw: int, cwc: int) -> float:
    total = ccw + cwc
    return float(sum(math.comb(total, k) for k in range(ccw, total + 1)) / 2**total)


def _reject_prohibited_report_content(value: Any, path: str = "report") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key.casefold() in FORBIDDEN_REPORT_KEYS:
                raise ValueError(f"aggregate report contains prohibited field at {path}.{key}")
            _reject_prohibited_report_content(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_prohibited_report_content(item, f"{path}[{index}]")
    elif isinstance(value, str) and value.startswith("/"):
        raise ValueError(f"aggregate report contains an absolute path at {path}")


def _validate_slice_aggregates(value: Any) -> None:
    dimensions = _closed(value, SLICE_DIMENSIONS, "slice aggregates")
    for dimension, buckets in dimensions.items():
        if not isinstance(buckets, dict):
            raise ValueError(f"slice aggregate {dimension} must be an object")
        for bucket, result in buckets.items():
            if not isinstance(bucket, str) or not OPAQUE_ID.fullmatch(bucket):
                raise ValueError(f"slice aggregate {dimension} has an invalid bucket")
            row = _closed(result, SLICE_RESULT_FIELDS, f"slice {dimension}.{bucket}")
            planned = row["planned"]
            correct = row["correct"]
            accuracy = row["accuracy"]
            if (
                not isinstance(planned, int)
                or isinstance(planned, bool)
                or planned < 0
                or not isinstance(correct, int)
                or isinstance(correct, bool)
                or correct < 0
                or correct > planned
                or not isinstance(accuracy, (int, float))
                or isinstance(accuracy, bool)
                or not math.isfinite(accuracy)
                or not math.isclose(accuracy, correct / planned if planned else 0.0, abs_tol=1e-12)
            ):
                raise ValueError(f"slice aggregate {dimension}.{bucket} is inconsistent")


def _validate_limitations(value: Any) -> None:
    if (
        not isinstance(value, list)
        or not all(isinstance(item, str) and item in ALLOWED_LIMITATIONS for item in value)
        or len(set(value)) != len(value)
    ):
        raise ValueError("limitations must use the closed aggregate-only vocabulary")


def _validate_stability_results(report: dict[str, Any], descriptor: dict[str, Any]) -> None:
    for field, schema in (
        ("option_order_stability", OPTION_ORDER_STABILITY_FIELDS),
        ("deterministic_repeat_stability", REPEAT_STABILITY_FIELDS),
    ):
        result = _closed(report[field], schema, field)
        subset_size = descriptor["permutation_subset_size"]
        if (
            result["subset_size"] != subset_size
            or result["selection_seed"] != descriptor["permutation_selection_seed"]
            or result["selection_digest"] != descriptor["permutation_selection_digest"]
            or result["evaluated"] != subset_size
            or not isinstance(result["stable"], int)
            or isinstance(result["stable"], bool)
            or not 0 <= result["stable"] <= result["evaluated"]
            or not isinstance(result["stability_rate"], (int, float))
            or isinstance(result["stability_rate"], bool)
            or not math.isfinite(result["stability_rate"])
            or not math.isclose(
                result["stability_rate"], result["stable"] / result["evaluated"], abs_tol=1e-12
            )
        ):
            raise ValueError(f"{field} does not bind the sealed deterministic subset")
    if report["option_order_stability"]["strategy"] != "cyclic_left_rotation_one":
        raise ValueError("option order stability strategy is not frozen")
    if report["deterministic_repeat_stability"]["repetitions"] != 2:
        raise ValueError("deterministic repeat stability requires exactly two runs")


def _validate_systems_measurements(value: Any) -> None:
    if value is None:
        return
    if (
        not isinstance(value, dict)
        or not value
        or not set(value).issubset(frozenset(REQUIRED_MODELS))
    ):
        raise ValueError("systems measurements must name only frozen model rows")
    for name, raw_measurement in value.items():
        measurement = _closed(
            raw_measurement,
            SYSTEM_GROUP_FIELDS | SYSTEM_METRIC_FIELDS | frozenset({"group"}),
            f"{name} systems measurement",
        )
        if not isinstance(measurement["group"], str) or not OPAQUE_ID.fullmatch(
            measurement["group"]
        ):
            raise ValueError(f"{name} systems measurement group is invalid")
        for field in SYSTEM_GROUP_FIELDS:
            field_value = measurement[field]
            if isinstance(field_value, str):
                valid = bool(OPAQUE_ID.fullmatch(field_value))
            else:
                valid = (
                    isinstance(field_value, int)
                    and not isinstance(field_value, bool)
                    and field_value > 0
                )
            if not valid:
                raise ValueError(f"{name} systems comparability field {field} is invalid")
        for metric in SYSTEM_METRIC_FIELDS:
            metric_value = measurement[metric]
            if metric == "peak_device_memory_bytes" and metric_value is None:
                continue
            if (
                not isinstance(metric_value, (int, float))
                or isinstance(metric_value, bool)
                or not math.isfinite(metric_value)
                or metric_value < 0
            ):
                raise ValueError(f"{name} systems metric {metric} is invalid")


def _validate_claim(
    report: dict[str, Any], available: dict[str, bool], candidate_deployment_class: str
) -> None:
    claim = report["claim"]
    if claim is None:
        return
    if not isinstance(claim, dict):
        raise ValueError("claim must be null or object")
    kind = claim.get("type")
    if kind == "quality":
        required = frozenset(
            {
                "type",
                "reference",
                "task",
                "locale",
                "target_model_revision",
                "reference_model_revision",
                "hardware",
                "metric",
                "denominator",
                "winning_margin",
            }
        )
        _closed(claim, required, "quality claim")
        if (
            not available["saracura"]
            or not available["kev"]
            or claim["reference"] != "kev-4b"
            or candidate_deployment_class != "qwen35_4b_pointer_head"
            or claim["task"] != "choice"
            or claim["locale"] != "pt-BR"
            or claim["metric"] != "planned_top1_accuracy"
            or claim["denominator"] != report["planned_denominator"]
            or claim["hardware"] != report["quality_settings"]["host_class"]
        ):
            raise ValueError("quality claim scope is invalid")
        saracura, kev = report["model_rows"]["saracura"], report["model_rows"]["kev"]
        if (
            claim["target_model_revision"] != saracura["model_revision"]
            or claim["reference_model_revision"] != kev["model_revision"]
        ):
            raise ValueError("quality claim revisions mismatch")
        counters = _closed(
            report["paired_comparison_counters"],
            frozenset({"candidate_correct_control_wrong", "candidate_wrong_control_correct"}),
            "paired counters",
        )
        ccw = counters["candidate_correct_control_wrong"]
        cwc = counters["candidate_wrong_control_correct"]
        candidate_correct, kev_correct = saracura["counts"]["correct"], kev["counts"]["correct"]
        margin = (candidate_correct - kev_correct) / report["planned_denominator"]
        if (
            not all(
                isinstance(value, int) and not isinstance(value, bool) and value >= 0
                for value in (ccw, cwc)
            )
            or ccw + cwc > report["planned_denominator"]
            or ccw - cwc != candidate_correct - kev_correct
            or candidate_correct <= kev_correct
            or not math.isclose(claim["winning_margin"], margin, abs_tol=1e-12)
            or _mcnemar_one_sided(ccw, cwc) >= 0.05
        ):
            raise ValueError("quality claim fails exact one-sided McNemar gate")
    elif kind == "systems":
        _closed(claim, frozenset({"type", "reference", "target"}), "systems claim")
        if (
            claim["reference"] != "kev-4b"
            or claim["target"] != "saracura"
            or not available["saracura"]
            or not available["kev"]
            or candidate_deployment_class != "qwen35_4b_pointer_head"
        ):
            raise ValueError("systems claim requires available named Saracura and Kev rows")
        saracura, kev = report["model_rows"]["saracura"], report["model_rows"]["kev"]
        if saracura["systems_comparability_group"] != kev[
            "systems_comparability_group"
        ] or not isinstance(saracura["systems_comparability_group"], str):
            raise ValueError("systems claim requires one identical comparability group")
        measures = report["systems_measurements"]
        if not isinstance(measures, dict) or set(measures) != {"saracura", "kev"}:
            raise ValueError("systems claim requires only named Saracura and Kev measurements")
        measurements: list[dict[str, Any]] = []
        for name in ("saracura", "kev"):
            measurement = _closed(
                measures[name],
                SYSTEM_GROUP_FIELDS | SYSTEM_METRIC_FIELDS | frozenset({"group"}),
                f"{name} systems measurement",
            )
            for metric in SYSTEM_METRIC_FIELDS:
                value = measurement[metric]
                if metric == "peak_device_memory_bytes" and value is None:
                    continue
                if (
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or not math.isfinite(value)
                    or value < 0
                ):
                    raise ValueError(f"{name} systems metric {metric} is invalid")
            measurements.append(measurement)
        if (
            measurements[0]["group"] != measurements[1]["group"]
            or measurements[0]["group"] != saracura["systems_comparability_group"]
            or any(
                measurements[0][field] != measurements[1][field] for field in SYSTEM_GROUP_FIELDS
            )
        ):
            raise ValueError("systems measurement comparability group mismatch")
    else:
        raise ValueError("unsupported claim type")


def validate_report(
    report_path: Path,
    sealed_descriptor_path: Path,
    disjointness_receipt_path: Path,
    selection_receipt_path: Path,
    repository: Path | None = None,
) -> int:
    report, descriptor, receipt, selection = (
        _load(path)
        for path in (
            report_path,
            sealed_descriptor_path,
            disjointness_receipt_path,
            selection_receipt_path,
        )
    )
    _closed(report, REPORT_FIELDS, "aggregate report")
    _reject_prohibited_report_content(report)
    _validate_slice_aggregates(report["slice_aggregates"])
    _validate_limitations(report["limitations"])
    _validate_descriptor(descriptor)
    _validate_stability_results(report, descriptor)
    descriptor_digest = _sha(sealed_descriptor_path)
    _validate_receipt(receipt, descriptor, descriptor_digest)
    _validate_selection(selection, descriptor_digest, repository or Path(__file__).parents[1])
    if (
        report["schema_version"] != "saracura-v02-evaluation-report.v1"
        or report["protocol_digest"] != PROTOCOL_SPEC_SHA256
        or report["sealed_descriptor_digest"] != descriptor_digest
        or report["disjointness_receipt_digest"] != _sha(disjointness_receipt_path)
        or report["selection_receipt_digest"] != _sha(selection_receipt_path)
    ):
        raise ValueError("report digest or schema binding mismatch")
    denominator = report["planned_denominator"]
    settings = _closed(report["quality_settings"], QUALITY_SETTINGS_FIELDS, "quality settings")
    if (
        not isinstance(denominator, int)
        or isinstance(denominator, bool)
        or denominator <= 0
        or settings["planned_denominator"] != denominator
        or settings["task"] != "choice"
        or settings["locale"] != "pt-BR"
        or not isinstance(settings["preprocessing"], str)
        or not OPAQUE_ID.fullmatch(settings["preprocessing"])
        or not isinstance(settings["scoring"], str)
        or not OPAQUE_ID.fullmatch(settings["scoring"])
        or not isinstance(settings["host_class"], str)
        or not OPAQUE_ID.fullmatch(settings["host_class"])
    ):
        raise ValueError("planned denominator mismatch")
    rows = _closed(report["model_rows"], frozenset(REQUIRED_MODELS), "model rows")
    available = {
        name: _validate_row(name, rows[name], denominator, settings, selection)
        for name in REQUIRED_MODELS
    }
    _validate_systems_measurements(report["systems_measurements"])
    _validate_claim(report, available, selection["candidate_deployment_class"])
    print("report valid")
    return 0


def main() -> int:
    args = sys.argv[1:]
    if args == ["validate-protocol"]:
        return validate_protocol()
    if (
        len(args) == 9
        and args[0] == "validate-report"
        and args[1::2]
        == ["--report", "--sealed-descriptor", "--disjointness-receipt", "--selection-receipt"]
    ):
        return validate_report(Path(args[2]), Path(args[4]), Path(args[6]), Path(args[8]))
    print("ARGUMENTS_INVALID", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
