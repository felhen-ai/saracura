"""Real, local-only execution and sealed systems-evidence artifacts."""

from __future__ import annotations

import hashlib
import os
import platform
import resource
import statistics
import sys
import tempfile
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

import rfc8785

from benchmarks.universal_local.acquisition import verification_receipt
from benchmarks.universal_local.inputs import Rendered, render_laya, render_nli
from benchmarks.universal_local.loaders import LoadedCandidate, LoadError, load_candidate
from benchmarks.universal_local.plan import Plan, Scenario, materialize, plan_digest, validate_plan
from benchmarks.universal_local.registry import Candidate

ROOT = Path(__file__).parents[2]
ARTIFACT_ROOT = ROOT / ".artifacts" / "universal-bakeoff"
_NAME = __import__("re").compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_STATUSES = {
    "supported",
    "supported_with_state_truncation",
    "supported_with_choice_truncation",
    "supported_with_state_and_choice_truncation",
    "unsupported_capacity",
    "unsupported_input",
    "numerical_error",
}


def _canonical(value: Any) -> bytes:
    return rfc8785.dumps(value)


def _sync(torch: Any, device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()


def _write_exclusive(path: Path, raw: bytes) -> None:
    fd, temporary = tempfile.mkstemp(prefix=".tmp-", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        _fsync(path.parent)
    finally:
        with suppress(FileNotFoundError):
            os.unlink(temporary)


def _fsync(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _artifact_dir(name: str) -> Path:
    if not _NAME.fullmatch(name):
        raise ValueError("invalid output name")
    if ARTIFACT_ROOT.is_symlink() or ARTIFACT_ROOT.parent.is_symlink():
        raise ValueError("invalid artifact root")
    ARTIFACT_ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(ARTIFACT_ROOT, 0o700)
    target = ARTIFACT_ROOT / name
    try:
        target.mkdir(mode=0o700)
    except FileExistsError as error:
        raise ValueError("output already exists") from error
    _fsync(ARTIFACT_ROOT)
    return target


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def _empty_counters() -> dict[str, int]:
    return {
        "supported_count": 0,
        "supported_with_state_truncation_count": 0,
        "supported_with_choice_truncation_count": 0,
        "supported_with_state_and_choice_truncation_count": 0,
        "unsupported_count": 0,
        "failed_count": 0,
        "state_tokens_original": 0,
        "state_tokens_retained": 0,
        "truncated_state_tokens": 0,
        "choice_tokens_original": 0,
        "choice_tokens_retained": 0,
        "truncated_choice_tokens": 0,
        "input_token_count_min": 0,
        "input_token_count_max": 0,
    }


def _record(rendered: Rendered, counters: dict[str, int]) -> None:
    if rendered.status not in _STATUSES:
        raise ValueError("unrecognized adapter status")
    if rendered.status == "supported":
        counters["supported_count"] += 1
    elif rendered.status == "supported_with_state_truncation":
        counters["supported_with_state_truncation_count"] += 1
    elif rendered.status == "supported_with_choice_truncation":
        counters["supported_with_choice_truncation_count"] += 1
    elif rendered.status == "supported_with_state_and_choice_truncation":
        counters["supported_with_state_and_choice_truncation_count"] += 1
    elif rendered.status in {"unsupported_capacity", "unsupported_input"}:
        counters["unsupported_count"] += 1
    else:
        counters["failed_count"] += 1
    counters["state_tokens_original"] += rendered.state_tokens_original
    counters["state_tokens_retained"] += rendered.state_tokens_retained
    counters["truncated_state_tokens"] += rendered.truncated_state_tokens
    counters["choice_tokens_original"] += rendered.choice_tokens_original
    counters["choice_tokens_retained"] += rendered.choice_tokens_retained
    counters["truncated_choice_tokens"] += rendered.truncated_choice_tokens
    if rendered.input_ids:
        count = len(rendered.input_ids)
        counters["input_token_count_min"] = (
            count
            if counters["input_token_count_min"] == 0
            else min(counters["input_token_count_min"], count)
        )
        counters["input_token_count_max"] = max(counters["input_token_count_max"], count)


def _select_laya_choice_logits(logits: Any, index: int, choice_count: int) -> Any:
    """Select the already-gathered choice logits returned by Laya."""
    return logits[index, :choice_count]


def _decisions_per_second(decisions_per_iteration: int, timings_ms: list[float]) -> float:
    total_ms = sum(timings_ms)
    return 1000.0 * decisions_per_iteration * len(timings_ms) / total_ms if total_ms > 0.0 else 0.0


def _run_laya(
    loaded: LoadedCandidate,
    scenario: Scenario,
    plan: Plan,
    torch: Any,
    device: str,
    counters: dict[str, int],
) -> tuple[tuple[float, ...], ...]:
    states, questions, choices = materialize(plan, scenario)
    outputs: list[tuple[float, ...]] = []
    compatible: dict[int, list[Rendered]] = {}
    for state, state_questions, state_choices in zip(states, questions, choices, strict=True):
        for instruction, option_set in zip(state_questions, state_choices, strict=True):
            rendered = render_laya(loaded.tokenizer, state, instruction, option_set)
            _record(rendered, counters)
            if not rendered.status.startswith("supported"):
                continue
            compatible.setdefault(len(rendered.input_ids), []).append(rendered)
    for length in sorted(compatible):
        rows = compatible[length]
        for offset in range(0, len(rows), 16):
            batch = rows[offset : offset + 16]
            ids = torch.tensor([row.input_ids for row in batch], dtype=torch.long, device=device)
            mask = torch.ones_like(ids)
            marker_pos = torch.tensor(
                [row.marker_positions for row in batch], dtype=torch.long, device=device
            )
            marker_mask = torch.ones_like(marker_pos, dtype=torch.bool)
            qtype = torch.zeros(len(batch), dtype=torch.long, device=device)
            logits = loaded.model(ids, mask, marker_pos, marker_mask, qtype)
            for index, row in enumerate(batch):
                selected = _select_laya_choice_logits(
                    logits, index, len(row.marker_positions)
                ).float()
                weights = torch.softmax(selected, dim=0)
                if not torch.isfinite(weights).all():
                    raise ValueError("numerical_error")
                outputs.append(tuple(float(item) for item in weights.cpu().tolist()))
    return tuple(outputs)


def _batch_nli(
    loaded: LoadedCandidate, rows: list[Rendered], torch: Any, device: str
) -> list[float]:
    values: list[float] = []
    for offset in range(0, len(rows), 16):
        batch = rows[offset : offset + 16]
        width = max(len(row.input_ids) for row in batch)
        ids = [
            list(row.input_ids) + [loaded.tokenizer.pad_token_id] * (width - len(row.input_ids))
            for row in batch
        ]
        attention = [[1] * len(row.input_ids) + [0] * (width - len(row.input_ids)) for row in batch]
        logits = loaded.model(
            input_ids=torch.tensor(ids, dtype=torch.long, device=device),
            attention_mask=torch.tensor(attention, dtype=torch.long, device=device),
        ).logits.float()
        probabilities = torch.softmax(logits, dim=-1)[:, 0]
        if not torch.isfinite(probabilities).all():
            raise ValueError("numerical_error")
        values.extend(float(item) for item in probabilities.cpu().tolist())
    return values


def _run_nli(
    loaded: LoadedCandidate,
    scenario: Scenario,
    plan: Plan,
    torch: Any,
    device: str,
    counters: dict[str, int],
) -> tuple[tuple[float, ...], ...]:
    states, questions, choices = materialize(plan, scenario)
    outputs: list[tuple[float, ...]] = []
    for state, state_questions, state_choices in zip(states, questions, choices, strict=True):
        for instruction, option_set in zip(state_questions, state_choices, strict=True):
            rendered = [
                render_nli(loaded.tokenizer, state, instruction, choice, scenario["locale"])
                for choice in option_set
            ]
            for item in rendered:
                _record(item, counters)
            if not all(item.status.startswith("supported") for item in rendered):
                continue
            entailment = _batch_nli(loaded, rendered, torch, device)
            denominator = sum(entailment)
            if not denominator > 0.0 or not __import__("math").isfinite(denominator):
                raise ValueError("numerical_error")
            outputs.append(tuple(item / denominator for item in entailment))
    return tuple(outputs)


def _execute(
    loaded: LoadedCandidate, plan: Plan, device: str
) -> tuple[tuple[tuple[float, ...], ...], dict[str, int]]:
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError as error:
        raise LoadError("universal-local extra is required") from error
    scenarios = [
        scenario
        for scenario in plan["scenarios"]
        if device == "mps" or scenario["id"] in {"scenario-01", "scenario-05"}
    ]
    counters = _empty_counters()
    outputs: list[tuple[float, ...]] = []
    with torch.inference_mode():
        for scenario in scenarios:
            _sync(torch, device)
            if loaded.candidate.loader_family == "laya_option_marker":
                rows = _run_laya(loaded, scenario, plan, torch, device, counters)
            else:
                rows = _run_nli(loaded, scenario, plan, torch, device, counters)
            outputs.extend(rows)
            _sync(torch, device)
    return tuple(outputs), counters


def _runtime_fields(loaded: LoadedCandidate, device: str) -> dict[str, Any]:
    import safetensors  # type: ignore[import-not-found]
    import tokenizers  # type: ignore[import-not-found]
    import torch
    import transformers  # type: ignore[import-not-found]

    return {
        "python_version": (
            f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        ),
        "torch_version": str(torch.__version__),
        "transformers_version": str(transformers.__version__),
        "tokenizers_version": str(tokenizers.__version__),
        "safetensors_version": str(safetensors.__version__),
        "os_version": f"{platform.system()}-{platform.release()}",
        "architecture": platform.machine(),
        "device_type": device,
        "execution_dtype": "fp32",
        "source_weight_dtypes": loaded.candidate.source_weight_dtypes,
        "process_threads": loaded.process_threads,
        "attention_implementation": (
            "sdpa" if loaded.candidate.id == "laya-multilingual" else "eager"
        ),
        "mps_fallback_enabled": False,
        "microbatch_size": 16,
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        * (1 if platform.system() == "Darwin" else 1024),
        "mps_current_allocated_bytes": int(torch.mps.current_allocated_memory())
        if device == "mps"
        else None,
        "mps_driver_allocated_bytes": int(torch.mps.driver_allocated_memory())
        if device == "mps"
        else None,
    }


def _result(candidate: Candidate, error_code: str | None) -> dict[str, Any]:
    return {
        "schema_version": "universal-bakeoff-result.v2",
        "status": "blocked" if error_code else "real_local_candidate",
        "metric_authority": "measured_systems_only",
        "conclusion_authority": "systems_compatibility_only",
        "candidate_id": candidate.id,
        "model_id": candidate.model_id,
        "model_revision": candidate.model_revision,
        "acquisition_contract_digest": candidate.acquisition_contract_digest,
        "conformance_vector_sha256": candidate.conformance_vector_sha256,
        "conformance_state": candidate.conformance_state,
        "architecture_attribution": candidate.architecture_attribution,
        "plan_sha256": plan_digest(),
        "error_code": error_code,
    }


def _seal(output: Path, result: dict[str, Any]) -> None:
    result_raw = _canonical(result) + b"\n"
    report_raw = (
        "# Universal local candidate run\n\n"
        + f"status: {result['status']}\n"
        + f"candidate_id: {result['candidate_id']}\n"
        + f"error_code: {result['error_code'] or 'none'}\n"
        + "conclusion_authority: systems_compatibility_only\n"
    ).encode("utf-8")
    manifest = {
        "schema_version": "universal-bakeoff-artifact-manifest.v1",
        "files": {
            name: {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
            for name, raw in (("result.json", result_raw), ("report.md", report_raw))
        },
        "manifest_sha256": "",
    }
    manifest["manifest_sha256"] = hashlib.sha256(
        _canonical({key: value for key, value in manifest.items() if key != "manifest_sha256"})
    ).hexdigest()
    paths = (
        (output / "result.json", result_raw),
        (output / "report.md", report_raw),
        (output / "artifact-manifest.json", _canonical(manifest) + b"\n"),
    )
    try:
        for path, raw in paths:
            _write_exclusive(path, raw)
        _fsync(output)
    except Exception:
        for path, _ in paths:
            with suppress(FileNotFoundError):
                path.unlink()
        with suppress(OSError):
            output.rmdir()
        raise


def run(
    candidate: Candidate, device: str, warmups: int, iterations: int, output_name: str
) -> dict[str, Any]:
    """Execute a reviewed candidate or seal its bounded blocked outcome."""
    if device not in {"cpu", "mps"} or not 1 <= warmups <= 3 or not 1 <= iterations <= 10:
        raise ValueError("invalid run parameters")
    if device == "cpu" and (warmups, iterations) != (1, 1):
        raise ValueError("CPU run shape is not approved")
    output = _artifact_dir(output_name)
    if (
        candidate.conformance_state != "source_contract_reviewed"
        or not candidate.conformance_vector_sha256
    ):
        result = _result(candidate, "conformance_pending")
        _seal(output, result)
        return {"candidate_id": candidate.id, "status": "blocked"}
    try:
        plan = validate_plan()
        with verification_receipt(candidate) as receipt:
            from benchmarks.universal_local.conformance import validate_conformance_receipt

            validate_conformance_receipt(receipt, candidate)
            started = time.perf_counter_ns()
            loaded = load_candidate(receipt, candidate, device)
            load_ms = (time.perf_counter_ns() - started) / 1_000_000
            try:
                import torch

                warmup_started = time.perf_counter_ns()
                for _ in range(warmups):
                    receipt.assert_live()
                    _execute(loaded, plan, device)
                warmup_ms = (time.perf_counter_ns() - warmup_started) / 1_000_000
                timings: list[float] = []
                first_counters: dict[str, int] | None = None
                first_outputs: tuple[tuple[float, ...], ...] | None = None
                stable = True
                for _ in range(iterations):
                    receipt.assert_live()
                    _sync(torch, device)
                    started = time.perf_counter_ns()
                    outputs, counters = _execute(loaded, plan, device)
                    _sync(torch, device)
                    timings.append((time.perf_counter_ns() - started) / 1_000_000)
                    stable = stable and (first_counters is None or counters == first_counters)
                    stable = stable and (first_outputs is None or outputs == first_outputs)
                    first_counters = counters
                    first_outputs = outputs
                receipt.assert_live()
                result = _result(candidate, None)
                result.update(_runtime_fields(loaded, device))
                result.update(first_counters or _empty_counters())
                result.update(
                    {
                        "load_duration_ms": load_ms,
                        "warmup_duration_ms": warmup_ms,
                        "latency_samples_ms": timings,
                        "latency_p50_ms": statistics.median(timings),
                        "latency_p95_ms": _percentile(timings, 0.95),
                        "decisions_per_second": _decisions_per_second(
                            len(first_outputs or ()), timings
                        ),
                        "deterministic_repeat_match": stable,
                        "comparison_eligible": device == "mps" and warmups == 1 and iterations == 3,
                    }
                )
            finally:
                loaded.close()
    except Exception:
        result = _result(candidate, "candidate_execution_blocked")
    _seal(output, result)
    return {"candidate_id": candidate.id, "status": result["status"]}
