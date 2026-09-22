"""Fail-closed, checkout-only remote comparison runner.

The implementation is intentionally boring: it uses no provider SDK, has no
retry path, and stores only reduced systems measurements in a chained journal.
"""

from __future__ import annotations

import hashlib
import math
import os
import stat
import statistics
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from contextlib import suppress
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from benchmarks.universal_local import plan as local_plan
from benchmarks.universal_local.registry import get_candidate
from benchmarks.universal_remote.plan import (
    ENDPOINT,
    MODEL,
    REQUEST_COUNT,
    canonical,
    digest,
    load_json,
    materialize_requests,
    validate_plan,
    validate_response,
    wire_bytes,
)

RESERVATION = Decimal("0.00320")
RATE = Decimal("0.042")
BUDGET = Decimal("0.25")
ALLOWED = {"result.json", "report.md", "cost-journal.jsonl", "artifact-manifest.json"}
FAILURES = {
    "http_error",
    "rate_limited",
    "provider_unavailable",
    "timeout",
    "transport_error",
    "response_invalid",
    "model_drift",
    "usage_overflow",
    "budget_uncertain",
    "process_interrupted",
}

Transport = Callable[
    [str, str, Mapping[str, str], bytes, float], tuple[int, Mapping[str, str], bytes]
]


def _decimal(value: Any) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as error:
        raise ValueError("invalid decimal") from error
    if not result.is_finite() or result < 0:
        raise ValueError("invalid decimal")
    return result


def decimal_string(value: Decimal) -> str:
    value = _decimal(value)
    return "0" if not value else format(value, "f").rstrip("0").rstrip(".")


def _fsync(directory: Path) -> None:
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_match(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.is_symlink() or path.read_bytes() != payload:
            raise ValueError("immutable artifact mismatch")
        return
    fd, temporary_name = tempfile.mkstemp(prefix=f".tmp-{path.name}-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        _fsync(path.parent)
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()


def _mode(path: Path, mode: int) -> None:
    try:
        status = path.lstat()
    except FileNotFoundError as error:
        raise ValueError("missing artifact path") from error
    expected_type = stat.S_IFDIR if path.is_dir() else stat.S_IFREG
    if (
        stat.S_ISLNK(status.st_mode)
        or stat.S_IFMT(status.st_mode) != expected_type
        or status.st_mode & 0o777 != mode
    ):
        raise ValueError("permission or link drift")


def create_output(path: Path) -> None:
    if path.is_absolute() is False or path.is_symlink():
        raise ValueError("output path must be absolute and non-symlink")
    try:
        path.mkdir(mode=0o700)
    except FileExistsError as error:
        raise ValueError("output already exists") from error
    _mode(path, 0o700)
    _fsync(path.parent)
    _atomic_match(path / "cost-journal.jsonl", b"")


def git_common_dir(repo: Path) -> Path:
    output = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--git-common-dir"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    common = Path(output)
    if not common.is_absolute():
        common = (repo / common).resolve()
    if not common.is_absolute() or common.is_symlink() or not common.is_dir():
        raise ValueError("cannot prove git common directory")
    return common


def _claim_path(repo: Path, lineage: str, *, create: bool) -> Path:
    if len(lineage) != 64 or any(item not in "0123456789abcdef" for item in lineage):
        raise ValueError("invalid lineage digest")
    common = git_common_dir(repo)
    claims = common / "saracura-local-claims"
    root = claims / "phase4c3"
    if create:
        for parent, directory in ((common, claims), (claims, root)):
            try:
                directory.mkdir(mode=0o700)
            except FileExistsError:
                pass
            else:
                _fsync(parent)
    for directory in (common, claims, root):
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError("cannot prove claim directory")
    _mode(claims, 0o700)
    _mode(root, 0o700)
    return root / f"{lineage}.json"


def create_claim(repo: Path, lineage: str, binding: dict[str, str]) -> str:
    path = _claim_path(repo, lineage, create=True)
    required = {
        "plan_v3",
        "prior_ledger",
        "training",
        "packet",
        "phase3c",
        "phase3d",
        "laya",
        "mdeberta",
    }
    if set(binding) != required or any(
        len(value) != 64 or any(char not in "0123456789abcdef" for char in value)
        for value in binding.values()
    ):
        raise ValueError("claim binding")
    raw = (
        canonical({"schema_version": "phase4c3-claim.v1", "lineage": lineage, "binding": binding})
        + b"\n"
    )
    # A byte-identical existing claim is still consumed entitlement, not a
    # resumable run.  Recovery uses `finalize`, never this creator.
    if path.exists():
        raise ValueError("claim already exists")
    try:
        _atomic_match(path, raw)
    except FileExistsError as error:
        # Another worktree may have won between the existence check and link.
        raise ValueError("claim already exists") from error
    _mode(path, 0o600)
    return hashlib.sha256(raw).hexdigest()


def validate_claim(repo: Path, lineage: str, binding: dict[str, str]) -> str:
    """Require the exact consumed entitlement; validation never creates one."""
    path = _claim_path(repo, lineage, create=False)
    _mode(path, 0o600)
    raw = path.read_bytes()
    value = load_json(raw)
    expected = {"schema_version", "lineage", "binding"}
    if (
        not isinstance(value, dict)
        or set(value) != expected
        or value.get("schema_version") != "phase4c3-claim.v1"
        or value.get("lineage") != lineage
        or value.get("binding") != binding
        or canonical(value) + b"\n" != raw
    ):
        raise ValueError("claim binding")
    return hashlib.sha256(raw).hexdigest()


class Journal:
    """Append-only canonical hash chain. Open reservations are permanently charged."""

    def __init__(self, path: Path, lineage: str) -> None:
        self.path, self.lineage = path, lineage
        self.rows: list[dict[str, Any]] = []
        self._previous = "0" * 64
        if path.exists():
            _mode(path, 0o600)
            for line in path.read_bytes().splitlines():
                row = load_json(line)
                if not isinstance(row, dict) or canonical(row) != line:
                    raise ValueError("journal canonicality")
                self._check(row)
                self.rows.append(row)
                self._previous = row["row_sha256"]

    def _check(self, row: dict[str, Any]) -> None:
        expected = {"sequence", "kind", "ordinal", "lineage", "previous_sha256", "row_sha256"}
        if (
            not expected <= set(row)
            or row["sequence"] != len(self.rows) + 1
            or row["lineage"] != self.lineage
        ):
            raise ValueError("journal sequence")
        if row["previous_sha256"] != self._previous or row["kind"] not in {
            "reserve",
            "settle",
            "settled_over_limit",
            "failure",
        }:
            raise ValueError("journal chain")
        unsigned = dict(row)
        actual = unsigned.pop("row_sha256")
        if actual != hashlib.sha256(canonical(unsigned)).hexdigest():
            raise ValueError("journal digest")
        ordinal = row["ordinal"]
        if not isinstance(ordinal, int) or ordinal < 0 or ordinal >= REQUEST_COUNT:
            raise ValueError("journal ordinal")
        if self.terminal_failure is not None:
            raise ValueError("journal already terminal")
        prior = [item for item in self.rows if item["ordinal"] == ordinal]
        if row["kind"] == "reserve":
            if ordinal != len({item["ordinal"] for item in self.rows}):
                raise ValueError("reservation ordinal")
            if (
                prior
                or set(row) != expected | {"reservation_cost_usd"}
                or row["reservation_cost_usd"] != "0.0032"
            ):
                raise ValueError("reservation transition")
        elif row["kind"] in {"settle", "settled_over_limit"}:
            if (
                len(prior) != 1
                or prior[0]["kind"] != "reserve"
                or set(row) != expected | {"sample", "computed_cost_usd"}
            ):
                raise ValueError("settlement transition")
            _validate_sample(row["sample"], ordinal)
            if row["computed_cost_usd"] != row["sample"]["computed_cost_usd"]:
                raise ValueError("settlement cost")
            is_over_limit = row["sample"]["input_tokens"] > 64000
            if (row["kind"] == "settled_over_limit") != is_over_limit:
                raise ValueError("settlement usage limit")
        else:
            if (
                len(prior) != 1
                or prior[0]["kind"] != "reserve"
                or set(row) != expected | {"category"}
                or row["category"] not in FAILURES - {"process_interrupted"}
            ):
                raise ValueError("failure transition")

    def append(self, kind: str, ordinal: int, **values: Any) -> dict[str, Any]:
        row = {
            "sequence": len(self.rows) + 1,
            "kind": kind,
            "ordinal": ordinal,
            "lineage": self.lineage,
            "previous_sha256": self._previous,
            **values,
        }
        row["row_sha256"] = hashlib.sha256(canonical(row)).hexdigest()
        self._check(row)
        raw = canonical(row) + b"\n"
        with self.path.open("ab") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        self.rows.append(row)
        self._previous = row["row_sha256"]
        return row

    def charged(self) -> Decimal:
        total = Decimal(0)
        by_ordinal: dict[int, list[dict[str, Any]]] = {}
        for row in self.rows:
            by_ordinal.setdefault(row["ordinal"], []).append(row)
        for rows in by_ordinal.values():
            terminal = rows[-1]
            total += _decimal(terminal.get("computed_cost_usd", RESERVATION))
        return total

    @property
    def terminal_failure(self) -> str | None:
        for row in self.rows:
            if row["kind"] == "failure":
                return str(row["category"])
            if row["kind"] == "settled_over_limit":
                return "usage_overflow"
        return None


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_sample(value: Any, ordinal: int) -> None:
    """Bind each reduced sample to its reviewed request without retaining text."""
    fields = {
        "ordinal",
        "scenario_id",
        "state_ordinal",
        "question_offset",
        "question_count",
        "decision_count",
        "option_count_min",
        "option_count_max",
        "latency_ms",
        "model",
        "input_tokens",
        "output_tokens",
        "computed_cost_usd",
        "response_shape_valid",
    }
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("settlement sample")
    meta = materialize_requests()[ordinal]
    request = meta["request"]
    option_counts = [len(question["criteria"]) for question in request["questions"].values()]
    expected = {
        "ordinal": ordinal,
        "scenario_id": meta["scenario_id"],
        "state_ordinal": meta["state_ordinal"],
        "question_offset": meta["question_offset"],
        "question_count": meta["question_count"],
        "decision_count": meta["question_count"],
        "option_count_min": min(option_counts),
        "option_count_max": max(option_counts),
        "model": MODEL,
        "response_shape_valid": True,
    }
    if any(value[key] != expected_value for key, expected_value in expected.items()):
        raise ValueError("settlement sample binding")
    if not isinstance(value["latency_ms"], (int, float)) or isinstance(value["latency_ms"], bool):
        raise ValueError("settlement latency")
    if not math.isfinite(float(value["latency_ms"])) or float(value["latency_ms"]) <= 0:
        raise ValueError("settlement latency")
    if not all(
        _is_int(value[key]) and value[key] >= 0 for key in ("input_tokens", "output_tokens")
    ):
        raise ValueError("settlement usage")
    computed = decimal_string(Decimal(value["input_tokens"]) * RATE / Decimal(1_000_000))
    if value["computed_cost_usd"] != computed:
        raise ValueError("settlement computed cost")


def _urllib_transport(
    method: str, url: str, headers: Mapping[str, str], body: bytes, timeout: float
) -> tuple[int, Mapping[str, str], bytes]:
    if method != "POST" or url != ENDPOINT or timeout != 60.0:
        raise ValueError("literal transport contract")

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args: Any, **kwargs: Any) -> None:
            return None

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    request = urllib.request.Request(url, data=body, headers=dict(headers), method="POST")
    try:
        with opener.open(request, timeout=timeout) as response:
            return int(response.status), dict(response.headers), response.read()
    except urllib.error.HTTPError as error:
        return int(error.code), dict(error.headers or {}), error.read()
    except urllib.error.URLError as error:
        if isinstance(error.reason, TimeoutError):
            raise TimeoutError from None
        raise


def dispatch(
    request: dict[str, Any], api_key: str, transport: Transport = _urllib_transport
) -> tuple[dict[str, Any], float]:
    if not api_key:
        raise ValueError("TYPESAFE_API_KEY required")
    started = time.perf_counter_ns()
    status, _, body = transport(
        "POST",
        ENDPOINT,
        {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        wire_bytes(request),
        60.0,
    )
    latency = (time.perf_counter_ns() - started) / 1_000_000
    if status != 200:
        if status == 429:
            raise RuntimeError("rate_limited")
        if status == 529:
            raise RuntimeError("provider_unavailable")
        raise RuntimeError("http_error")
    try:
        return validate_response(load_json(body), request), latency
    except ValueError as error:
        if "model drift" in str(error):
            raise RuntimeError("model_drift") from None
        raise RuntimeError("response_invalid") from None


def sample(meta: dict[str, Any], validated: dict[str, Any], latency_ms: float) -> dict[str, Any]:
    request = meta["request"]
    sizes = [len(question["criteria"]) for question in request["questions"].values()]
    return {
        "ordinal": meta["ordinal"],
        "scenario_id": meta["scenario_id"],
        "state_ordinal": meta["state_ordinal"],
        "question_offset": meta["question_offset"],
        "question_count": meta["question_count"],
        "decision_count": meta["question_count"],
        "option_count_min": min(sizes),
        "option_count_max": max(sizes),
        "latency_ms": latency_ms,
        "model": MODEL,
        "input_tokens": validated["input_tokens"],
        "output_tokens": validated["output_tokens"],
        "computed_cost_usd": decimal_string(
            Decimal(validated["input_tokens"]) * RATE / Decimal(1_000_000)
        ),
        "response_shape_valid": True,
    }


def _summarize(samples: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [float(item["latency_ms"]) for item in samples]
    total_ms = sum(latencies)
    decisions = sum(int(item["decision_count"]) for item in samples)
    return {
        "completed_request_count": len(samples),
        "completed_decision_count": decisions,
        "total_sequential_measured_duration_ms": total_ms,
        "request_latency_p50_ms": statistics.median(latencies) if latencies else None,
        "request_latency_p95_ms": sorted(latencies)[round((len(latencies) - 1) * 0.95)]
        if latencies
        else None,
        "decisions_per_second": (1000 * decisions / total_ms) if total_ms else None,
        "input_tokens": sum(int(item["input_tokens"]) for item in samples),
        "output_tokens": sum(int(item["output_tokens"]) for item in samples),
        "computed_cost_usd": decimal_string(
            sum((_decimal(item["computed_cost_usd"]) for item in samples), Decimal(0))
        ),
    }


def _scenario_summaries(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for scenario_id in validate_plan()["scenarios"]:
        subset = [item for item in samples if item["scenario_id"] == scenario_id]
        values.append({"scenario_id": scenario_id, **_summarize(subset)})
    return values


def validate_local_comparison(value: list[dict[str, Any]]) -> None:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("local comparison shape")
    expected = (("laya-multilingual", 182), ("mdeberta-nli", 183))
    fields = {
        "candidate_id",
        "completed_decision_count",
        "latency_p50_ms",
        "decisions_per_second",
    }
    for row, (candidate_id, decisions) in zip(value, expected, strict=True):
        if (
            not isinstance(row, dict)
            or set(row) != fields
            or row["candidate_id"] != candidate_id
            or row["completed_decision_count"] != decisions
        ):
            raise ValueError("local comparison shape")
        for key in ("latency_p50_ms", "decisions_per_second"):
            metric = row[key]
            if isinstance(metric, bool) or not isinstance(metric, (int, float)):
                raise ValueError("local comparison metric")
            if not math.isfinite(float(metric)) or float(metric) <= 0:
                raise ValueError("local comparison metric")


def _validate_result_shape(
    result: dict[str, Any],
    journal: Journal,
    provenance: dict[str, str],
    local_comparison: list[dict[str, Any]],
) -> None:
    """Only the two explicitly specified result forms may be sealed."""
    expected = reconstruct(journal, provenance, local_comparison)
    if result != expected:
        raise ValueError("result reconstruction")
    if result["status"] == "complete":
        if set(result) != {
            "schema_version",
            "status",
            "systems_samples",
            "summary",
            "per_scenario",
            "local_comparison",
            "predecessors",
            "failure_category",
        }:
            raise ValueError("complete result shape")
        if (
            len(result["systems_samples"]) != REQUEST_COUNT
            or result["failure_category"] is not None
        ):
            raise ValueError("complete result state")
        if result["summary"] != _summarize(result["systems_samples"]):
            raise ValueError("complete summary")
        if result["per_scenario"] != _scenario_summaries(result["systems_samples"]):
            raise ValueError("scenario summaries")
        validate_local_comparison(result["local_comparison"])
    elif result["status"] == "partial":
        if set(result) != {
            "schema_version",
            "status",
            "completed_request_count",
            "completed_decision_count",
            "failure_category",
            "charged_cost_usd",
            "predecessors",
        }:
            raise ValueError("partial result shape")
        if result["failure_category"] not in FAILURES:
            raise ValueError("partial result failure")
    else:
        raise ValueError("result status")


def reconstruct(
    journal: Journal,
    provenance: dict[str, str],
    local_comparison: list[dict[str, Any]],
) -> dict[str, Any]:
    validate_local_comparison(local_comparison)
    normalized_local = load_json(canonical(local_comparison))
    if not isinstance(normalized_local, list):
        raise ValueError("local comparison shape")
    settlements = [row["sample"] for row in journal.rows if row["kind"] == "settle"]
    completed = [
        row["sample"] for row in journal.rows if row["kind"] in {"settle", "settled_over_limit"}
    ]
    failure = journal.terminal_failure
    open_row = next(
        (
            row
            for row in journal.rows
            if row["kind"] == "reserve"
            and not any(
                later["ordinal"] == row["ordinal"] and later["kind"] != "reserve"
                for later in journal.rows
            )
        ),
        None,
    )
    if open_row is not None and failure is None:
        failure = "process_interrupted"
    if len(settlements) == REQUEST_COUNT and failure is None:
        return {
            "schema_version": "phase4c3-result.v1",
            "status": "complete",
            "systems_samples": settlements,
            "summary": _summarize(settlements),
            "per_scenario": _scenario_summaries(settlements),
            "local_comparison": normalized_local,
            "predecessors": provenance,
            "failure_category": None,
        }
    if failure is None:
        failure = "process_interrupted"
    return {
        "schema_version": "phase4c3-result.v1",
        "status": "partial",
        "completed_request_count": len(completed),
        "completed_decision_count": sum(int(item["decision_count"]) for item in completed),
        "failure_category": failure,
        "charged_cost_usd": decimal_string(journal.charged()),
        "predecessors": provenance,
    }


def render_report(result: dict[str, Any]) -> bytes:
    lines = ["# Phase 4C.3 controlled TypeSafe comparison", "", f"- Status: `{result['status']}`"]
    if result["status"] == "complete":
        summary = result["summary"]
        duration = summary["total_sequential_measured_duration_ms"]
        remote_p50 = summary["request_latency_p50_ms"]
        remote_p95 = summary["request_latency_p95_ms"]
        lines.extend(
            [
                f"- Completed decisions: `{summary['completed_decision_count']}`",
                f"- Sequential remote duration: `{duration}` ms",
                f"- Remote request p50/p95: `{remote_p50}` / `{remote_p95}` ms",
                f"- Remote decisions/second: `{summary['decisions_per_second']}`",
                f"- Computed (not billed) cost: `${summary['computed_cost_usd']}`",
                "",
                "## Whole-matrix local evidence",
                "",
            ]
        )
        for local in result["local_comparison"]:
            lines.append(
                f"- `{local['candidate_id']}`: {local['completed_decision_count']} decisions, "
                f"matrix p50 `{local['latency_p50_ms']}` ms, "
                f"`{local['decisions_per_second']}` decisions/second"
            )
        lines.extend(["", "## Remote per-scenario evidence", ""])
        for scenario in result["per_scenario"]:
            scenario_p50 = scenario["request_latency_p50_ms"]
            scenario_p95 = scenario["request_latency_p95_ms"]
            lines.append(
                f"- `{scenario['scenario_id']}`: {scenario['completed_request_count']} requests, "
                f"{scenario['completed_decision_count']} decisions, p50/p95 "
                f"`{scenario_p50}` / `{scenario_p95}` ms, "
                f"`{scenario['decisions_per_second']}` decisions/second, "
                f"tokens `{scenario['input_tokens']}`/`{scenario['output_tokens']}`, "
                f"cost `${scenario['computed_cost_usd']}`"
            )
    else:
        lines.extend(
            [
                f"- Completed decisions: `{result['completed_decision_count']}`",
                f"- Failure category: `{result['failure_category']}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "This is local-versus-WAN systems evidence, not quality, calibration, PT-BR parity, "
            "or a product-winner result.",
            "Local timings exclude model load and network. Remote timing includes WAN and provider "
            "service time; remote cold/warm state is uncontrolled.",
            "The local three-iteration evidence attests one warm-up but cannot independently "
            "recover it. Laya covers 182 decisions; mDeBERTa and the remote matrix cover 183.",
            "TypeSafe groups up to ten questions per state; Laya scores choices jointly; "
            "mDeBERTa NLI evaluates one state-choice pair at a time.",
            "",
        ]
    )
    return "\n".join(lines).encode("utf-8")


def _manifest(
    result_raw: bytes,
    report_raw: bytes,
    journal_raw: bytes,
    provenance: dict[str, str],
    claim_sha256: str,
) -> bytes:
    payloads = {
        "result.json": result_raw,
        "report.md": report_raw,
        "cost-journal.jsonl": journal_raw,
    }
    files = {
        name: {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        for name, raw in payloads.items()
    }
    unsigned = {
        "schema_version": "phase4c3-artifact-manifest.v1",
        "claim_sha256": claim_sha256,
        "files": files,
        "predecessors": provenance,
        "sources": {
            "benchmarks/universal_remote/plan.py": digest(Path(__file__).with_name("plan.py")),
            "benchmarks/universal_remote/runner.py": digest(Path(__file__)),
        },
    }
    return (
        canonical({**unsigned, "self_sha256": hashlib.sha256(canonical(unsigned)).hexdigest()})
        + b"\n"
    )


def _remove_writer_temporaries(output: Path) -> None:
    removed = False
    for path in output.iterdir():
        if not path.name.startswith(".tmp-"):
            continue
        _mode(path, 0o600)
        if not any(
            path.name.startswith(f".tmp-{target}-")
            for target in ("result.json", "report.md", "artifact-manifest.json")
        ):
            raise ValueError("unrecognized atomic temporary")
        path.unlink()
        removed = True
    if removed:
        _fsync(output)


def finalize(
    repo: Path,
    output: Path,
    lineage: str,
    provenance: dict[str, str],
    local_comparison: list[dict[str, Any]],
) -> dict[str, Any]:
    """Credential-free sealing.  It cannot dispatch or release an entitlement."""
    if not output.is_absolute() or output.is_symlink():
        raise ValueError("output path must be absolute and non-symlink")
    _mode(output, 0o700)
    if not (output / "cost-journal.jsonl").is_file():
        raise ValueError("artifact allowlist")
    claim_sha256 = validate_claim(repo, lineage, provenance)
    journal = Journal(output / "cost-journal.jsonl", lineage)
    result = reconstruct(journal, provenance, local_comparison)
    result_raw = canonical(result) + b"\n"
    report_raw = render_report(result)
    journal_raw = (output / "cost-journal.jsonl").read_bytes()
    manifest_raw = _manifest(result_raw, report_raw, journal_raw, provenance, claim_sha256)
    _remove_writer_temporaries(output)
    names = {path.name for path in output.iterdir()}
    if not names <= ALLOWED or "cost-journal.jsonl" not in names:
        raise ValueError("artifact allowlist")
    _atomic_match(output / "result.json", result_raw)
    _atomic_match(output / "report.md", report_raw)
    _atomic_match(output / "artifact-manifest.json", manifest_raw)
    validate_artifact(repo, output, lineage, provenance, local_comparison)
    return result


def validate_artifact(
    repo: Path,
    output: Path,
    lineage: str,
    provenance: dict[str, str],
    local_comparison: list[dict[str, Any]],
) -> dict[str, Any]:
    if not output.is_absolute() or output.is_symlink():
        raise ValueError("output path must be absolute and non-symlink")
    _mode(output, 0o700)
    entries = {path.name: path for path in output.iterdir()}
    if set(entries) != ALLOWED:
        raise ValueError("artifact allowlist")
    for path in entries.values():
        _mode(path, 0o600)
    claim_sha256 = validate_claim(repo, lineage, provenance)
    journal = Journal(entries["cost-journal.jsonl"], lineage)
    raw_result = entries["result.json"].read_bytes()
    result = load_json(raw_result)
    if not isinstance(result, dict):
        raise ValueError("result shape")
    if canonical(result) + b"\n" != raw_result:
        raise ValueError("result canonical")
    _validate_result_shape(result, journal, provenance, local_comparison)
    if entries["report.md"].read_bytes() != render_report(result):
        raise ValueError("report contradiction")
    raw_manifest = entries["artifact-manifest.json"].read_bytes()
    manifest = load_json(raw_manifest)
    if canonical(manifest) + b"\n" != raw_manifest or not isinstance(manifest, dict):
        raise ValueError("manifest canonical")
    if (
        set(manifest)
        != {
            "schema_version",
            "claim_sha256",
            "files",
            "predecessors",
            "sources",
            "self_sha256",
        }
        or not isinstance(manifest["files"], dict)
        or not isinstance(manifest["sources"], dict)
    ):
        raise ValueError("manifest shape")
    unsigned = dict(manifest)
    self_digest = unsigned.pop("self_sha256", None)
    if (
        self_digest != hashlib.sha256(canonical(unsigned)).hexdigest()
        or manifest.get("schema_version") != "phase4c3-artifact-manifest.v1"
        or manifest.get("claim_sha256") != claim_sha256
        or manifest.get("predecessors") != provenance
        or manifest.get("files", {}).keys() != {"result.json", "report.md", "cost-journal.jsonl"}
        or manifest.get("sources")
        != {
            "benchmarks/universal_remote/plan.py": digest(Path(__file__).with_name("plan.py")),
            "benchmarks/universal_remote/runner.py": digest(Path(__file__)),
        }
    ):
        raise ValueError("manifest provenance")
    expected_manifest = _manifest(
        entries["result.json"].read_bytes(),
        entries["report.md"].read_bytes(),
        entries["cost-journal.jsonl"].read_bytes(),
        provenance,
        claim_sha256,
    )
    if raw_manifest != expected_manifest:
        raise ValueError("manifest provenance")
    for name, metadata in manifest["files"].items():
        if metadata != {
            "bytes": len(entries[name].read_bytes()),
            "sha256": digest(entries[name]),
        }:
            raise ValueError("manifest file digest")
    forbidden = (
        "subject",
        "body",
        "instructions",
        "criteria",
        "answers",
        "probabilities",
        "confidence",
        "api_key",
        "secret",
        "path",
    )
    if any(term in raw_result.decode("utf-8").lower() for term in forbidden):
        raise ValueError("forbidden result field")
    return result


def validate_phase4c2_artifact(output: Path, candidate_id: str) -> dict[str, Any]:
    """Additive validator for existing v2 local evidence; it never rewrites it."""
    _mode(output, 0o700)
    files = {path.name: path for path in output.iterdir()}
    if set(files) != {"result.json", "report.md", "artifact-manifest.json"}:
        raise ValueError("local artifact allowlist")
    for path in files.values():
        _mode(path, 0o600)
    raw_manifest = files["artifact-manifest.json"].read_bytes()
    manifest = load_json(raw_manifest)
    if (
        not isinstance(manifest, dict)
        or set(manifest) != {"schema_version", "files", "manifest_sha256"}
        or manifest.get("schema_version") != "universal-bakeoff-artifact-manifest.v1"
        or not isinstance(manifest.get("files"), dict)
        or set(manifest["files"]) != {"result.json", "report.md"}
        or canonical(manifest) + b"\n" != raw_manifest
    ):
        raise ValueError("local manifest")
    unsigned = dict(manifest)
    claimed = unsigned.pop("manifest_sha256", None)
    if claimed != hashlib.sha256(canonical(unsigned)).hexdigest():
        raise ValueError("local manifest digest")
    for name in ("result.json", "report.md"):
        if manifest.get("files", {}).get(name) != {
            "bytes": len(files[name].read_bytes()),
            "sha256": digest(files[name]),
        }:
            raise ValueError("local file digest")
    raw_result = files["result.json"].read_bytes()
    result = load_json(raw_result)
    base_fields = {
        "schema_version",
        "status",
        "metric_authority",
        "conclusion_authority",
        "candidate_id",
        "model_id",
        "model_revision",
        "acquisition_contract_digest",
        "conformance_vector_sha256",
        "conformance_state",
        "architecture_attribution",
        "source_weight_dtypes",
        "plan_sha256",
        "error_code",
    }
    expected_fields = base_fields | local_plan.REPORT_METRICS
    if (
        not isinstance(result, dict)
        or set(result) != expected_fields
        or canonical(result) + b"\n" != raw_result
        or result.get("schema_version") != "universal-bakeoff-result.v2"
        or result.get("candidate_id") != candidate_id
    ):
        raise ValueError("local result identity")
    if any(
        result.get(key) != value
        for key, value in {
            "status": "real_local_candidate",
            "metric_authority": "measured_systems_only",
            "conclusion_authority": "systems_compatibility_only",
            "error_code": None,
            "device_type": "mps",
        }.items()
    ):
        raise ValueError("local result gates")
    if result.get("comparison_eligible") is not True:
        raise ValueError("local result gates")
    candidate = get_candidate(candidate_id)
    for key in (
        "model_id",
        "model_revision",
        "acquisition_contract_digest",
        "conformance_vector_sha256",
        "conformance_state",
        "architecture_attribution",
        "source_weight_dtypes",
    ):
        if result.get(key) != getattr(candidate, key):
            raise ValueError("local registry identity")
    if result.get("attention_implementation") != (
        "sdpa" if candidate_id == "laya-multilingual" else "eager"
    ):
        raise ValueError("local attention implementation")
    if result.get("plan_sha256") != local_plan.plan_digest():
        raise ValueError("local plan digest")
    report = (
        "# Universal local candidate run\n\n"
        + f"status: {result['status']}\n"
        + f"candidate_id: {result['candidate_id']}\n"
        + "error_code: none\n"
        + "conclusion_authority: systems_compatibility_only\n"
    ).encode("utf-8")
    if files["report.md"].read_bytes() != report:
        raise ValueError("local report")
    timings = result.get("latency_samples_ms")
    if (
        not isinstance(timings, list)
        or len(timings) != 3
        or any(isinstance(x, bool) or not isinstance(x, (int, float)) or x <= 0 for x in timings)
    ):
        raise ValueError("local latency samples")
    if any(not math.isfinite(float(x)) for x in timings):
        raise ValueError("local latency samples")
    if (
        result.get("latency_p50_ms") != statistics.median(timings)
        or result.get("latency_p95_ms") != sorted(timings)[2]
    ):
        raise ValueError("local latency summaries")
    expected_decisions = 182 if candidate_id == "laya-multilingual" else 183
    counter_keys = (
        "supported_count",
        "supported_with_state_truncation_count",
        "supported_with_choice_truncation_count",
        "supported_with_state_and_choice_truncation_count",
    )
    if any(
        isinstance(result.get(key), bool) or not _is_int(result.get(key)) for key in counter_keys
    ):
        raise ValueError("local coverage")
    counters = tuple(result[key] for key in counter_keys)
    expected_counters = (
        (52, 0, 130, 0) if candidate_id == "laya-multilingual" else (1813, 1028, 0, 0)
    )
    if (
        counters != expected_counters
        or result.get("deterministic_repeat_match") is not True
        or result.get("unsupported_count") != (1 if candidate_id == "laya-multilingual" else 0)
        or result.get("failed_count") != 0
        or result.get("decisions_per_second")
        != 1000.0 * expected_decisions * len(timings) / sum(float(item) for item in timings)
    ):
        raise ValueError("local coverage")
    return result


def _run_prevalidated(
    repo: Path,
    output: Path,
    provenance: dict[str, str],
    local_comparison: list[dict[str, Any]],
    *,
    api_key: str,
    transport: Transport = _urllib_transport,
) -> dict[str, Any]:
    """Private fake-transport seam used only after the CLI has completed preflight."""
    if not api_key:
        raise ValueError("TYPESAFE_API_KEY required")
    plan = validate_plan()
    requests = materialize_requests()  # all bodies are checked before claim/output/spend
    if plan["remote_budget"]["requests"] != len(requests):
        raise ValueError("request budget drift")
    lineage = provenance["prior_ledger"]
    create_output(output)
    try:
        create_claim(repo, lineage, provenance)
    except Exception:
        # The just-created empty evidence holder is safe to remove; no claim
        # succeeded in this invocation.
        with suppress(FileNotFoundError):
            (output / "cost-journal.jsonl").unlink()
        with suppress(OSError):
            output.rmdir()
        raise
    journal = Journal(output / "cost-journal.jsonl", lineage)
    for meta in requests:
        if journal.charged() + RESERVATION > BUDGET:
            journal.append("failure", meta["ordinal"], category="budget_uncertain")
            break
        journal.append("reserve", meta["ordinal"], reservation_cost_usd="0.0032")
        try:
            validated, latency = dispatch(meta["request"], api_key, transport)
            reduced = sample(meta, validated, latency)
            kind = "settled_over_limit" if validated["input_tokens"] > 64000 else "settle"
            journal.append(
                kind,
                meta["ordinal"],
                sample=reduced,
                computed_cost_usd=reduced["computed_cost_usd"],
            )
            if kind != "settle":
                break
        except TimeoutError:
            journal.append("failure", meta["ordinal"], category="timeout")
            break
        except RuntimeError as error:
            category = str(error)
            journal.append(
                "failure",
                meta["ordinal"],
                category=category if category in FAILURES else "transport_error",
            )
            break
        except Exception:
            journal.append("failure", meta["ordinal"], category="transport_error")
            break
    return finalize(repo, output, lineage, provenance, local_comparison)
