"""Frozen Phase 4B.3 aggregate-only calibration mathematics.

This module intentionally has no filesystem or model loading.  Keeping the
numerics separate makes it possible to test the exact golden algorithm without
opening a capsule or retaining text/logits in an output artifact.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from benchmarks.first_party_packet import LABELS
from saracura.calibration.models import CalibrationContext
from saracura.research_trust import canonical

ALGORITHM_REVISION = "bounded-log-temperature-golden-v1"
BOUNDS = (0.5, 5.0)
GOLDEN_RESOURCE = "human-calibration-golden.v1.json"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


class HumanCalibrationError(ValueError):
    """A non-secret numerical or closed-artifact contract failure."""


def _sequential_sum(values: Sequence[float]) -> float:
    """Use one specified IEEE-754 addition order on every supported Python."""

    total = 0.0
    for value in values:
        total = total + value
    return total


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _finite(value: object) -> float:
    if type(value) is float:
        number = value
    elif type(value) is int:
        number = float(value)
    else:
        raise HumanCalibrationError("non-finite calibration number")
    if not math.isfinite(number):
        raise HumanCalibrationError("non-finite calibration number")
    return number


def _rows(
    logits: Sequence[Sequence[float]], labels: Sequence[int]
) -> list[tuple[tuple[float, ...], int]]:
    if not logits or len(logits) != len(labels):
        raise HumanCalibrationError("calibration record count")
    result: list[tuple[tuple[float, ...], int]] = []
    for row, label in zip(logits, labels, strict=True):
        if len(row) != 5 or type(label) is not int or not 0 <= label < 5:
            raise HumanCalibrationError("calibration label or logits")
        result.append((tuple(_finite(value) for value in row), label))
    return result


def probabilities(logits: Sequence[float], temperature: float) -> tuple[float, ...]:
    """Stable float64 softmax using the specified scaled maximum."""

    if len(logits) != 5 or not math.isfinite(temperature) or not 0.5 <= temperature <= 5.0:
        raise HumanCalibrationError("temperature bounds")
    scaled = tuple(_finite(value) / temperature for value in logits)
    maximum = max(scaled)
    exponentials = tuple(math.exp(value - maximum) for value in scaled)
    total = _sequential_sum(exponentials)
    if not math.isfinite(total) or total <= 0:
        raise HumanCalibrationError("softmax intermediate")
    result = tuple(value / total for value in exponentials)
    if not all(math.isfinite(value) and 0 <= value <= 1 for value in result):
        raise HumanCalibrationError("softmax result")
    return result


def _nll(rows: Sequence[tuple[tuple[float, ...], int]], temperature: float) -> float:
    values: list[float] = []
    for logits, label in rows:
        scaled = tuple(value / temperature for value in logits)
        maximum = max(scaled)
        log_sum = maximum + math.log(
            _sequential_sum(tuple(math.exp(value - maximum) for value in scaled))
        )
        item = log_sum - scaled[label]
        if not math.isfinite(item):
            raise HumanCalibrationError("NLL intermediate")
        values.append(item)
    result = _sequential_sum(values) / len(values)
    if not math.isfinite(result) or result < 0:
        raise HumanCalibrationError("NLL result")
    return result


def fit_temperature(logits: Sequence[Sequence[float]], labels: Sequence[int]) -> float:
    """Run the frozen 96-update golden-section search in log-temperature."""

    rows = _rows(logits, labels)
    a, b = math.log(BOUNDS[0]), math.log(BOUNDS[1])
    r = (math.sqrt(5) - 1) / 2
    c, d = b - r * (b - a), a + r * (b - a)
    fc, fd = _nll(rows, math.exp(c)), _nll(rows, math.exp(d))
    for _ in range(96):
        old_a, old_b, old_c, old_d, old_fc, old_fd = a, b, c, d, fc, fd
        if (old_fc, math.exp(old_c)) <= (old_fd, math.exp(old_d)):
            a, b, d, fd = old_a, old_d, old_c, old_fc
            c = old_d - r * (old_d - old_a)
            fc = _nll(rows, math.exp(c))
        else:
            a, b, c, fc = old_c, old_b, old_d, old_fd
            d = old_c + r * (old_b - old_c)
            fd = _nll(rows, math.exp(d))
    candidates = (0.5, 1.0, 5.0, math.exp(a), math.exp(b), math.exp(c), math.exp(d))
    best = min((_nll(rows, value), value) for value in candidates)
    if not math.isfinite(best[1]):
        raise HumanCalibrationError("temperature result")
    return best[1]


def metrics(
    logits: Sequence[Sequence[float]], labels: Sequence[int], temperature: float
) -> dict[str, Any]:
    """Compute the exact aggregate classification and calibration metrics."""

    rows = _rows(logits, labels)
    matrix = [[0 for _ in range(5)] for _ in range(5)]
    briers: list[float] = []
    confidences: list[float] = []
    correct: list[float] = []
    for row, label in rows:
        probs = probabilities(row, temperature)
        prediction = max(range(5), key=lambda index: (probs[index], -index))
        confidence = probs[prediction]
        matrix[label][prediction] += 1
        briers.append(
            _sequential_sum(
                tuple(
                    (item - (1.0 if index == label else 0.0)) ** 2
                    for index, item in enumerate(probs)
                )
            )
        )
        confidences.append(confidence)
        correct.append(1.0 if prediction == label else 0.0)
    per_label: dict[str, dict[str, float | int]] = {}
    f1s: list[float] = []
    for index, name in enumerate(LABELS):
        support = sum(matrix[index])
        predicted = sum(matrix[row][index] for row in range(5))
        tp = matrix[index][index]
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[name] = {
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        f1s.append(f1)
    ece = 0.0
    for bucket in range(15):
        members = [
            index
            for index, item in enumerate(confidences)
            if min(14, math.floor(item * 15)) == bucket
        ]
        if members:
            ece += (
                len(members)
                / len(rows)
                * abs(
                    _sequential_sum(tuple(correct[index] for index in members)) / len(members)
                    - _sequential_sum(tuple(confidences[index] for index in members)) / len(members)
                )
            )
    result: dict[str, Any] = {
        "count": len(rows),
        "per_label": per_label,
        "accuracy": _sequential_sum(correct) / len(rows),
        "macro_f1": _sequential_sum(f1s) / 5,
        "confusion_matrix": matrix,
        "nll": _nll(rows, temperature),
        "brier": _sequential_sum(briers) / len(rows),
        "ece_15": ece,
        "mean_confidence": _sequential_sum(confidences) / len(rows),
        "max_confidence": max(confidences),
    }
    if not all(
        math.isfinite(float(result[name]))
        for name in (
            "accuracy",
            "macro_f1",
            "nll",
            "brier",
            "ece_15",
            "mean_confidence",
            "max_confidence",
        )
    ):
        raise HumanCalibrationError("metric result")
    return result


def fit_metrics(
    logits: Sequence[Sequence[float]], labels: Sequence[int], temperature: float
) -> dict[str, float]:
    value = metrics(logits, labels, temperature)
    return {name: float(value[name]) for name in ("nll", "brier", "ece_15")}


def canonical_timestamp(now: datetime | None = None) -> str:
    instant = now or datetime.now(UTC)
    return instant.astimezone(UTC).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


def _timestamp(value: str) -> None:
    if TIMESTAMP.fullmatch(value) is None:
        raise HumanCalibrationError("calibration timestamp")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise HumanCalibrationError("calibration timestamp") from exc
    if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != value:
        raise HumanCalibrationError("calibration timestamp")


def _digests(*values: str) -> None:
    if any(HEX64.fullmatch(value) is None for value in values):
        raise HumanCalibrationError("calibration digest")


def fit_artifact(
    context: CalibrationContext,
    *,
    packet_manifest_sha256: str,
    packet_release_receipt_sha256: str,
    calibration_view_sha256: str,
    model_manifest_sha256: str,
    checkpoint_sha256: str,
    golden_resource_sha256: str,
    logits: Sequence[Sequence[float]],
    labels: Sequence[int],
    independent_state_count: int,
    created_at: str,
) -> dict[str, Any]:
    rows = _rows(logits, labels)
    _digests(
        packet_manifest_sha256,
        packet_release_receipt_sha256,
        calibration_view_sha256,
        model_manifest_sha256,
        checkpoint_sha256,
        golden_resource_sha256,
    )
    _timestamp(created_at)
    counts = {name: 0 for name in LABELS}
    for _row, label in rows:
        counts[LABELS[label]] += 1
    if (
        independent_state_count < 100
        or len(rows) < 100
        or independent_state_count > len(rows)
        or any(value < 10 for value in counts.values())
    ):
        raise HumanCalibrationError("calibration minimum")
    temperature = fit_temperature(logits, labels)
    value: dict[str, Any] = {
        "schema_version": "temperature-fit.v1",
        **context.model_dump(mode="json"),
        "packet_manifest_sha256": packet_manifest_sha256,
        "packet_release_receipt_sha256": packet_release_receipt_sha256,
        "calibration_view_sha256": calibration_view_sha256,
        "training_manifest_sha256": model_manifest_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "golden_resource_sha256": golden_resource_sha256,
        "algorithm_revision": ALGORITHM_REVISION,
        "temperature": temperature,
        "fit_sample_count": len(rows),
        "fit_independent_state_count": independent_state_count,
        "per_label_counts": counts,
        "metrics_before": fit_metrics(logits, labels, 1.0),
        "metrics_after": fit_metrics(logits, labels, temperature),
        "created_at": created_at,
        "fit_sha256": "",
    }
    value["fit_sha256"] = digest(
        canonical({key: item for key, item in value.items() if key != "fit_sha256"})
    )
    return value


def blind_report(
    *,
    model_manifest_sha256: str,
    packet_manifest_sha256: str,
    fit_sha256: str,
    blind_view_sha256: str,
    exposure_run_sha256: str,
    golden_resource_sha256: str,
    logits: Sequence[Sequence[float]],
    labels: Sequence[int],
    temperature: float,
    created_at: str,
) -> dict[str, Any]:
    """Build the immutable, aggregate-only report for one sealed exposure."""

    _rows(logits, labels)
    _digests(
        model_manifest_sha256,
        packet_manifest_sha256,
        fit_sha256,
        blind_view_sha256,
        exposure_run_sha256,
        golden_resource_sha256,
    )
    _timestamp(created_at)
    value: dict[str, Any] = {
        "schema_version": "blind-report.v1",
        "model_manifest_sha256": model_manifest_sha256,
        "packet_manifest_sha256": packet_manifest_sha256,
        "fit_sha256": fit_sha256,
        "blind_view_sha256": blind_view_sha256,
        "exposure_run_sha256": exposure_run_sha256,
        "golden_resource_sha256": golden_resource_sha256,
        "metrics_before": metrics(logits, labels, 1.0),
        "metrics_after": metrics(logits, labels, temperature),
        "created_at": created_at,
        "report_sha256": "",
    }
    value["report_sha256"] = digest(
        canonical({key: item for key, item in value.items() if key != "report_sha256"})
    )
    return value


def calibration_candidate(
    context: CalibrationContext,
    *,
    temperature: float,
    fit_sample_count: int,
    evaluation_sample_count: int,
    fit_independent_state_count: int,
    evaluation_independent_state_count: int,
    fit_sha256: str,
    blind_view_sha256: str,
    exposure_run_sha256: str,
    blind_report_sha256: str,
    metrics_before: dict[str, Any],
    metrics_after: dict[str, Any],
    created_at: str,
) -> dict[str, Any]:
    if (
        fit_sample_count < 100
        or evaluation_sample_count < 100
        or fit_independent_state_count < 100
        or evaluation_independent_state_count < 100
        or fit_independent_state_count > fit_sample_count
        or evaluation_independent_state_count > evaluation_sample_count
    ):
        raise HumanCalibrationError("calibration candidate minimum")
    _digests(fit_sha256, blind_view_sha256, exposure_run_sha256, blind_report_sha256)
    _timestamp(created_at)
    if not math.isfinite(temperature) or not 0.5 <= temperature <= 5.0:
        raise HumanCalibrationError("temperature bounds")
    value: dict[str, Any] = {
        "schema_version": "research-calibration-candidate.v2",
        **context.model_dump(mode="json"),
        "status": "pending_maintainer_release",
        "method": "temperature-scaling",
        "parameters": {"temperature": temperature, "bounds": [0.5, 5.0]},
        "fit_sample_count": fit_sample_count,
        "evaluation_sample_count": evaluation_sample_count,
        "fit_independent_state_count": fit_independent_state_count,
        "evaluation_independent_state_count": evaluation_independent_state_count,
        "minimum_independent_state_count": 100,
        "metrics_before": metrics_before,
        "metrics_after": metrics_after,
        "fit_sha256": fit_sha256,
        "blind_view_sha256": blind_view_sha256,
        "exposure_run_sha256": exposure_run_sha256,
        "blind_report_sha256": blind_report_sha256,
        "created_at": created_at,
        "candidate_sha256": "",
    }
    value["candidate_sha256"] = digest(
        canonical({key: item for key, item in value.items() if key != "candidate_sha256"})
    )
    return value
