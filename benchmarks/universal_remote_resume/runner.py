"""Immutable, bounded Phase 4C.3c continuation evidence writer.

This module intentionally has an injectable transport.  The public CLI is the
only live entrypoint and is not used by this offline implementation phase.
"""

from __future__ import annotations

import hashlib
import math
import os
import stat
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from contextlib import suppress
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from benchmarks.universal_remote import runner as predecessor
from benchmarks.universal_remote.plan import (
    MODEL,
    canonical,
    materialize_requests,
)
from benchmarks.universal_remote.plan import (
    load_json as predecessor_load_json,
)
from benchmarks.universal_remote_resume.plan import (
    BODY_CEILING,
    DIAGNOSTICS,
    FIRST_ORDINAL,
    LAST_ORDINAL,
    ResponseError,
    digest,
    load_json,
    validate_plan,
    validate_response,
)

RESERVATION = Decimal("0.0032")
RATE = Decimal("0.042")
BUDGET = Decimal("0.25")
ARTIFACT_NAMES = {
    "result.json",
    "report.md",
    "cost-journal.jsonl",
    "artifact-manifest.json",
}
SOURCE_NAMES = (
    "__init__.py",
    "__main__.py",
    "plan.py",
    "runner.py",
)
FAILURES = frozenset(
    {
        "http_error",
        "rate_limited",
        "provider_unavailable",
        "timeout",
        "transport_error",
        "response_invalid",
        "model_drift",
    }
)
Transport = Callable[
    [str, str, Mapping[str, str], bytes, float], tuple[int, Mapping[str, str], bytes]
]


def _read_limited(stream: Any) -> bytes:
    """Read at most one extra byte so a provider body is never retained unbounded."""
    body = stream.read(BODY_CEILING + 1)
    if not isinstance(body, bytes):
        raise ValueError("transport body")
    return body


def _urllib_transport(
    method: str, url: str, headers: Mapping[str, str], body: bytes, timeout: float
) -> tuple[int, Mapping[str, str], bytes]:
    if method != "POST" or url != validate_plan()["endpoint"] or timeout != 60.0:
        raise ValueError("literal transport contract")

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args: Any, **kwargs: Any) -> None:
            return None

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    request = urllib.request.Request(url, data=body, headers=dict(headers), method="POST")
    try:
        with opener.open(request, timeout=timeout) as response:
            return int(response.status), dict(response.headers), _read_limited(response)
    except urllib.error.HTTPError as error:
        return int(error.code), dict(error.headers or {}), _read_limited(error)
    except urllib.error.URLError as error:
        if isinstance(error.reason, TimeoutError):
            raise TimeoutError from None
        raise


def decimal_string(value: Decimal) -> str:
    if not value.is_finite() or value < 0:
        raise ValueError("invalid decimal")
    return "0" if not value else format(value, "f").rstrip("0").rstrip(".")


def _hash(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _atomic(path: Path, raw: bytes) -> None:
    if path.exists():
        if path.is_symlink() or path.read_bytes() != raw:
            raise ValueError("immutable artifact mismatch")
        return
    fd, name = tempfile.mkstemp(prefix=f".tmp-{path.name}-", dir=path.parent)
    temporary = Path(name)
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
            temporary.unlink()


def _fsync(directory: Path) -> None:
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _mode(path: Path, expected: int) -> None:
    status = path.lstat()
    expected_type = stat.S_IFDIR if expected == 0o700 else stat.S_IFREG
    if (
        stat.S_ISLNK(status.st_mode)
        or stat.S_IFMT(status.st_mode) != expected_type
        or status.st_mode & 0o777 != expected
    ):
        raise ValueError("permission or link drift")


def create_output(output: Path) -> None:
    if not output.is_absolute() or output.is_symlink():
        raise ValueError("output path must be absolute and non-symlink")
    try:
        output.mkdir(mode=0o700)
    except FileExistsError as error:
        raise ValueError("output already exists") from error
    _mode(output, 0o700)
    _fsync(output.parent)
    _atomic(output / "cost-journal.jsonl", b"")


def _claim_path(repo: Path, lineage: str, *, create: bool) -> Path:
    if len(lineage) != 64 or not all(x in "0123456789abcdef" for x in lineage):
        raise ValueError("invalid lineage")
    common = predecessor.git_common_dir(repo)
    claims = common / "saracura-local-claims"
    root = claims / "phase4c3c"
    for parent, directory in ((common, claims), (claims, root)):
        if create:
            try:
                directory.mkdir(mode=0o700)
            except FileExistsError:
                pass
            else:
                _fsync(parent)
        try:
            _mode(directory, 0o700)
        except (FileNotFoundError, NotADirectoryError, ValueError) as error:
            raise ValueError("claim directory") from error
    return root / f"{lineage}.json"


def create_claim(repo: Path, lineage: str, binding: dict[str, str]) -> str:
    expected = {
        "recovery_plan",
        "phase4c3_partial",
        "plan_v3",
        "prior_ledger",
        "training",
        "packet",
        "phase3c",
        "phase3d",
        "laya",
        "mdeberta",
    }
    if set(binding) != expected or not all(
        len(value) == 64 and all(char in "0123456789abcdef" for char in value)
        for value in binding.values()
    ):
        raise ValueError("claim binding")
    raw = (
        canonical({"schema_version": "phase4c3c-claim.v1", "lineage": lineage, "binding": binding})
        + b"\n"
    )
    path = _claim_path(repo, lineage, create=True)
    if path.exists():
        raise ValueError("claim already exists")
    try:
        _atomic(path, raw)
    except FileExistsError as error:
        raise ValueError("claim already exists") from error
    os.chmod(path, 0o600)
    return _hash(raw)


def validate_claim(repo: Path, lineage: str, binding: dict[str, str]) -> str:
    path = _claim_path(repo, lineage, create=False)
    _mode(path, 0o600)
    raw = path.read_bytes()
    value = predecessor_load_json(raw)
    if (
        not isinstance(value, dict)
        or value != {"schema_version": "phase4c3c-claim.v1", "lineage": lineage, "binding": binding}
        or canonical(value) + b"\n" != raw
    ):
        raise ValueError("claim binding")
    return _hash(raw)


class Journal:
    """Closed continuation journal: ordinals only move forward from 10."""

    def __init__(self, path: Path, lineage: str, plan: dict[str, Any] | None = None) -> None:
        self.path, self.lineage, self.plan = (
            path,
            lineage,
            validate_plan() if plan is None else plan,
        )
        self.rows: list[dict[str, Any]] = []
        self.previous = "0" * 64
        if path.exists():
            _mode(path, 0o600)
            for raw in path.read_bytes().splitlines():
                value = predecessor_load_json(raw)
                if not isinstance(value, dict) or canonical(value) != raw:
                    raise ValueError("journal canonicality")
                self._check(value)
                self.rows.append(value)
                self.previous = value["row_sha256"]

    def _check(self, row: dict[str, Any]) -> None:
        base = {"sequence", "kind", "ordinal", "lineage", "previous_sha256", "row_sha256"}
        if (
            not base <= set(row)
            or row["sequence"] != len(self.rows) + 1
            or row["lineage"] != self.lineage
            or row["previous_sha256"] != self.previous
        ):
            raise ValueError("journal chain")
        if self.terminal is not None:
            raise ValueError("journal already terminal")
        unsigned = dict(row)
        if row["row_sha256"] != _hash(
            canonical(unsigned | {"row_sha256": ""})[:-0]
            if False
            else canonical({key: value for key, value in unsigned.items() if key != "row_sha256"})
        ):
            raise ValueError("journal digest")
        ordinal = row["ordinal"]
        if not isinstance(ordinal, int) or not FIRST_ORDINAL <= ordinal <= LAST_ORDINAL:
            raise ValueError("journal ordinal")
        prior = [item for item in self.rows if item["ordinal"] == ordinal]
        expected_ordinal = FIRST_ORDINAL + len({item["ordinal"] for item in self.rows})
        kind = row["kind"]
        if kind == "reserve":
            if (
                set(row) != base | {"reservation_cost_usd", "wire_sha256"}
                or prior
                or ordinal != expected_ordinal
                or (self.rows and self.rows[-1]["kind"] != "settle")
                or row["reservation_cost_usd"] != "0.0032"
                or row["wire_sha256"] != self.plan["wire_sha256_by_ordinal"][str(ordinal)]
            ):
                raise ValueError("reserve transition")
        elif kind in {"settle", "settled_over_limit"}:
            if (
                len(prior) != 1
                or prior[0]["kind"] != "reserve"
                or not self.rows
                or self.rows[-1] is not prior[0]
                or set(row) != base | {"sample", "computed_cost_usd"}
            ):
                raise ValueError("settlement transition")
            _validate_sample(row["sample"], ordinal)
            if row["computed_cost_usd"] != row["sample"]["computed_cost_usd"] or (
                kind == "settled_over_limit"
            ) != (row["sample"]["input_tokens"] > 64000):
                raise ValueError("settlement binding")
        elif kind == "failure":
            if (
                len(prior) != 1
                or prior[0]["kind"] != "reserve"
                or not self.rows
                or self.rows[-1] is not prior[0]
                or set(row) != base | {"category", "response_invalid_detail"}
                or row["category"] not in FAILURES
                or (
                    row["category"] == "response_invalid"
                    and row["response_invalid_detail"] not in DIAGNOSTICS
                )
                or (
                    row["category"] != "response_invalid"
                    and row["response_invalid_detail"] is not None
                )
            ):
                raise ValueError("failure transition")
        elif kind == "halt":
            if (
                prior
                or ordinal != expected_ordinal
                or (self.rows and self.rows[-1]["kind"] != "settle")
                or set(row) != base | {"category", "response_invalid_detail"}
                or row["category"] != "budget_uncertain"
                or row["response_invalid_detail"] is not None
            ):
                raise ValueError("halt transition")
        else:
            raise ValueError("journal kind")

    @property
    def terminal(self) -> str | None:
        for row in self.rows:
            if row["kind"] == "failure":
                return str(row["category"])
            if row["kind"] == "halt":
                return "budget_uncertain"
            if row["kind"] == "settled_over_limit":
                return "usage_overflow"
        return None

    def append(self, kind: str, ordinal: int, **fields: Any) -> dict[str, Any]:
        row = {
            "sequence": len(self.rows) + 1,
            "kind": kind,
            "ordinal": ordinal,
            "lineage": self.lineage,
            "previous_sha256": self.previous,
            **fields,
        }
        row["row_sha256"] = _hash(canonical(row))
        self._check(row)
        with self.path.open("ab") as stream:
            stream.write(canonical(row) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.rows.append(row)
        self.previous = row["row_sha256"]
        return row

    def charged(self) -> Decimal:
        total = Decimal(0)
        for ordinal in {row["ordinal"] for row in self.rows}:
            rows = [row for row in self.rows if row["ordinal"] == ordinal]
            if not any(row["kind"] == "reserve" for row in rows):
                continue
            total += Decimal(rows[-1].get("computed_cost_usd", "0.0032"))
        return total


def _validate_sample(sample: Any, ordinal: int) -> None:
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
        "probability_sum_tolerance_used",
    }
    if not isinstance(sample, dict) or set(sample) != fields:
        raise ValueError("sample shape")
    meta = materialize_requests()[ordinal]
    sizes = [len(question["criteria"]) for question in meta["request"]["questions"].values()]
    expected = {
        "ordinal": ordinal,
        "scenario_id": meta["scenario_id"],
        "state_ordinal": meta["state_ordinal"],
        "question_offset": meta["question_offset"],
        "question_count": meta["question_count"],
        "decision_count": meta["question_count"],
        "option_count_min": min(sizes),
        "option_count_max": max(sizes),
        "model": MODEL,
        "response_shape_valid": True,
    }
    if any(sample[key] != value for key, value in expected.items()) or not isinstance(
        sample["probability_sum_tolerance_used"], bool
    ):
        raise ValueError("sample binding")
    if any(
        isinstance(sample[key], bool)
        or not isinstance(sample[key], int)
        or sample[key] < 0
        or sample[key] > 2**53 - 1
        for key in ("input_tokens", "output_tokens")
    ):
        raise ValueError("sample usage")
    latency = sample["latency_ms"]
    if (
        isinstance(latency, bool)
        or not isinstance(latency, (int, float))
        or not math.isfinite(float(latency))
        or float(latency) <= 0
    ):
        raise ValueError("sample latency")
    if sample["computed_cost_usd"] != decimal_string(
        Decimal(sample["input_tokens"]) * RATE / Decimal(1_000_000)
    ):
        raise ValueError("sample cost")


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
        "probability_sum_tolerance_used": validated["probability_sum_tolerance_used"],
    }


def dispatch(
    meta: dict[str, Any], api_key: str, transport: Transport
) -> tuple[dict[str, Any], float]:
    if not api_key:
        raise ValueError("TYPESAFE_API_KEY required")
    plan = validate_plan()
    wire = meta["wire"]
    if _hash(wire) != plan["wire_sha256_by_ordinal"][str(meta["ordinal"])]:
        raise ValueError("wire digest drift")
    started = time.perf_counter_ns()
    status, _, body = transport(
        "POST",
        plan["endpoint"],
        {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        wire,
        60.0,
    )
    elapsed = (time.perf_counter_ns() - started) / 1_000_000
    if status != 200:
        raise RuntimeError(
            {429: "rate_limited", 529: "provider_unavailable"}.get(status, "http_error")
        )
    if len(body) > BODY_CEILING:
        raise ResponseError("response_too_large")
    try:
        return validate_response(load_json(body), meta["request"]), elapsed
    except ResponseError:
        raise
    except Exception:
        # The response body and exception text remain process-local.
        raise ResponseError("validator_internal_error") from None


def _predecessor_samples(
    partial: dict[str, Any], journal: predecessor.Journal
) -> list[dict[str, Any]]:
    samples = [row["sample"] for row in journal.rows if row["kind"] == "settle"]
    if len(samples) != 10 or [sample["ordinal"] for sample in samples] != list(range(10)):
        raise ValueError("partial settlement lineage")
    if (
        partial.get("completed_request_count") != 10
        or partial.get("completed_decision_count") != 19
    ):
        raise ValueError("partial count")
    return [
        sample
        | {
            "measurement_source": "phase4c3_predecessor",
            "response_validation_contract": "strict_sum_1e-6",
            "probability_sum_tolerance_used": False,
        }
        for sample in samples
    ]


def _validate_partial_charge(result: dict[str, Any], computed: Decimal) -> Decimal:
    raw = result.get("charged_cost_usd")
    if not isinstance(raw, str):
        raise ValueError("partial charge")
    try:
        declared = Decimal(raw)
    except InvalidOperation as error:
        raise ValueError("partial charge") from error
    if (
        not declared.is_finite()
        or declared < 0
        or decimal_string(declared) != raw
        or declared != computed
    ):
        raise ValueError("partial charge")
    return computed


def validate_partial_artifact(
    repo: Path, artifact: Path, provenance: dict[str, str], local_comparison: list[dict[str, Any]]
) -> tuple[dict[str, Any], str, Decimal, list[dict[str, Any]]]:
    """Validate the sealed predecessor before output, claim, or transport use."""
    predecessor.validate_artifact(
        repo,
        artifact,
        provenance["prior_ledger"],
        {
            key: value
            for key, value in provenance.items()
            if key not in {"recovery_plan", "phase4c3_partial"}
        },
        local_comparison,
    )
    raw = (artifact / "result.json").read_bytes()
    result = predecessor_load_json(raw)
    if (
        not isinstance(result, dict)
        or result.get("schema_version") != "phase4c3-result.v1"
        or result.get("status") != "partial"
        or result.get("failure_category") != "response_invalid"
    ):
        raise ValueError("partial contract")
    journal = predecessor.Journal(artifact / "cost-journal.jsonl", provenance["prior_ledger"])
    rows = journal.rows
    if (
        len(rows) != 22
        or [row["kind"] for row in rows[:-2]]
        != [item for _ in range(10) for item in ("reserve", "settle")]
        or rows[-2]["kind"] != "reserve"
        or rows[-1].get("category") != "response_invalid"
    ):
        raise ValueError("partial journal")
    manifest_digest = digest(artifact / "artifact-manifest.json")
    if provenance["phase4c3_partial"] != manifest_digest:
        raise ValueError("partial manifest binding")
    charge = _validate_partial_charge(result, journal.charged())
    return result, manifest_digest, charge, _predecessor_samples(result, journal)


def _continuation_samples(journal: Journal) -> list[dict[str, Any]]:
    return [
        row["sample"]
        | {
            "measurement_source": "phase4c3c_continuation",
            "response_validation_contract": "approx_sum_0.01",
        }
        for row in journal.rows
        if row["kind"] in {"settle", "settled_over_limit"}
    ]


def reconstruct(
    journal: Journal,
    predecessor_partial: dict[str, Any],
    predecessor_charge: Decimal,
    prior_use: Decimal,
    prior_samples: list[dict[str, Any]],
    binding: dict[str, str],
    local_comparison: list[dict[str, Any]],
) -> dict[str, Any]:
    continuation = _continuation_samples(journal)
    completed = prior_samples + continuation
    terminal = journal.terminal
    open_reservation = bool(journal.rows and journal.rows[-1]["kind"] == "reserve")
    failure = terminal or (
        "process_interrupted" if open_reservation or len(continuation) != 47 else None
    )
    charge = {
        "earlier_governed_use_usd": decimal_string(prior_use),
        "predecessor_partial_usd": decimal_string(predecessor_charge),
        "continuation_journal_usd": decimal_string(journal.charged()),
        "cumulative_authorized_use_usd": decimal_string(
            prior_use + predecessor_charge + journal.charged()
        ),
    }
    predecessor_info = {
        "artifact_manifest_sha256": binding["phase4c3_partial"],
        "result_sha256": digest(Path("/dev/null")) if False else "0" * 64,
        "journal_sha256": "0" * 64,
        "claim_sha256": "0" * 64,
        "completed_request_count": 10,
        "completed_decision_count": 19,
        "charged_cost_usd": decimal_string(predecessor_charge),
    }
    # The caller replaces immutable-file hashes; placeholders cannot be published.
    if len(continuation) == 47 and failure is None:
        return {
            "schema_version": "phase4c3c-result.v1",
            "status": "complete",
            "systems_samples": completed,
            "summary": _summary(completed),
            "continuation_summary": _summary(continuation),
            "per_scenario": _per_scenario(completed),
            "local_comparison": local_comparison,
            "remote_charge": charge,
            "predecessor_partial": predecessor_info,
            "predecessors": binding,
            "failure_category": None,
            "response_invalid_detail": None,
        }
    detail = next(
        (row.get("response_invalid_detail") for row in journal.rows if row["kind"] == "failure"),
        None,
    )
    return {
        "schema_version": "phase4c3c-result.v1",
        "status": "partial",
        "completed_request_count": len(completed),
        "completed_decision_count": sum(int(sample["decision_count"]) for sample in completed),
        "failure_category": failure,
        "response_invalid_detail": detail if failure == "response_invalid" else None,
        "remote_charge": charge,
        "predecessor_partial": predecessor_info,
        "predecessors": binding,
    }


def _summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "completed_request_count": len(samples),
        "completed_decision_count": sum(int(item["decision_count"]) for item in samples),
        "input_tokens": sum(int(item["input_tokens"]) for item in samples),
        "output_tokens": sum(int(item["output_tokens"]) for item in samples),
        "computed_cost_usd": decimal_string(
            sum((Decimal(item["computed_cost_usd"]) for item in samples), Decimal(0))
        ),
    }


def _per_scenario(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scenarios = sorted({str(sample["scenario_id"]) for sample in samples})
    return [
        {
            "scenario_id": scenario,
            **_summary([item for item in samples if item["scenario_id"] == scenario]),
        }
        for scenario in scenarios
    ]


def _with_partial_hashes(result: dict[str, Any], partial_dir: Path) -> dict[str, Any]:
    partial = dict(result["predecessor_partial"])
    partial.update(
        {
            "result_sha256": digest(partial_dir / "result.json"),
            "journal_sha256": digest(partial_dir / "cost-journal.jsonl"),
            "claim_sha256": predecessor_load_json(
                (partial_dir / "artifact-manifest.json").read_bytes()
            )["claim_sha256"],
        }
    )
    return result | {"predecessor_partial": partial}


def _report(result: dict[str, Any]) -> bytes:
    lines = [
        "# Phase 4C.3c TypeSafe response-contract recovery",
        "",
        f"- Status: `{result['status']}`",
        "- Ordinals 0 through 9 are reused immutable predecessor measurements; "
        "ordinals 10 through 56 are new continuation measurements.",
        "- Ordinal 10 returned HTTP 200 in the prior failed session and is the first "
        "continuation request.",
        "- This is systems-compatibility evidence only; it is not quality, calibration, "
        "or billing evidence.",
        "- Predecessor samples use strict sum 1e-6; continuation samples use "
        "approximate sum 0.01 and discard probability values.",
    ]
    if result["status"] == "complete":
        lines.append(
            "- This complete result represents 57 samples from 58 provider executions; "
            "the sessions are not uninterrupted and ordinal-10 timing may be cache affected."
        )
        for label, key in (
            ("Composite summary (ordinals 0-56)", "summary"),
            ("Continuation-only summary (ordinals 10-56)", "continuation_summary"),
        ):
            summary = result[key]
            lines.append(
                f"- {label}: {summary['completed_request_count']} requests, "
                f"{summary['completed_decision_count']} decisions, "
                f"{summary['input_tokens']} input tokens, "
                f"{summary['output_tokens']} output tokens, computed cost USD "
                f"{summary['computed_cost_usd']}."
            )
    else:
        lines.append(f"- Failure category: `{result['failure_category']}`")
        if result["completed_request_count"] > 10 or result["failure_category"] in {
            "response_invalid",
            "model_drift",
            "usage_overflow",
        }:
            lines.append(
                "- The continuation received provider evidence for ordinal 10; it is the "
                "second provider execution of that request."
            )
        else:
            lines.append(
                "- The continuation did not produce validated provider evidence for ordinal "
                "10; whether the provider received or executed it is unknown."
            )
    return ("\n".join(lines) + "\n").encode()


def _manifest(
    result: bytes, report: bytes, journal: bytes, binding: dict[str, str], claim: str
) -> bytes:
    files = {
        name: {"bytes": len(raw), "sha256": _hash(raw)}
        for name, raw in {
            "result.json": result,
            "report.md": report,
            "cost-journal.jsonl": journal,
        }.items()
    }
    unsigned = {
        "schema_version": "phase4c3c-artifact-manifest.v1",
        "claim_sha256": claim,
        "files": files,
        "predecessors": binding,
        "sources": _source_digests(),
    }
    return canonical(unsigned | {"self_sha256": _hash(canonical(unsigned))}) + b"\n"


def _source_digests() -> dict[str, str]:
    package = Path(__file__).parent
    return {
        f"benchmarks/universal_remote_resume/{name}": digest(package / name)
        for name in SOURCE_NAMES
    }


def _artifact_entries(output: Path, *, complete: bool) -> dict[str, Path]:
    _validate_output_directory(output)
    entries = {path.name: path for path in output.iterdir()}
    if complete and set(entries) != ARTIFACT_NAMES:
        raise ValueError("artifact allowlist")
    if not complete and not set(entries) <= ARTIFACT_NAMES:
        raise ValueError("artifact allowlist")
    if "cost-journal.jsonl" not in entries:
        raise ValueError("artifact allowlist")
    for path in entries.values():
        _mode(path, 0o600)
    return entries


def _validate_output_directory(output: Path) -> None:
    if not output.is_absolute():
        raise ValueError("output path must be absolute and non-symlink")
    try:
        _mode(output, 0o700)
    except (FileNotFoundError, ValueError) as error:
        raise ValueError("output path must be absolute and non-symlink") from error


def _remove_writer_temporaries(output: Path) -> None:
    _validate_output_directory(output)
    removed = False
    allowed = ("result.json", "report.md", "artifact-manifest.json")
    for path in output.iterdir():
        if not path.name.startswith(".tmp-"):
            continue
        if not any(path.name.startswith(f".tmp-{name}-") for name in allowed):
            raise ValueError("unrecognized atomic temporary")
        _mode(path, 0o600)
        path.unlink()
        removed = True
    if removed:
        _fsync(output)


def validate_artifact(
    repo: Path,
    output: Path,
    partial_dir: Path,
    binding: dict[str, str],
    local_comparison: list[dict[str, Any]],
    prior_use: Decimal,
) -> dict[str, Any]:
    """Read-only reconstruction and validation of a sealed continuation artifact."""
    entries = _artifact_entries(output, complete=True)
    claim = validate_claim(repo, binding["prior_ledger"], binding)
    partial, _, charge, samples = validate_partial_artifact(
        repo, partial_dir, binding, local_comparison
    )
    journal = Journal(entries["cost-journal.jsonl"], binding["phase4c3_partial"])
    result = _with_partial_hashes(
        reconstruct(journal, partial, charge, prior_use, samples, binding, local_comparison),
        partial_dir,
    )
    result_raw = canonical(result) + b"\n"
    report_raw = _report(result)
    journal_raw = entries["cost-journal.jsonl"].read_bytes()
    manifest_raw = _manifest(result_raw, report_raw, journal_raw, binding, claim)
    expected = {
        "result.json": result_raw,
        "report.md": report_raw,
        "cost-journal.jsonl": journal_raw,
        "artifact-manifest.json": manifest_raw,
    }
    if any(entries[name].read_bytes() != raw for name, raw in expected.items()):
        raise ValueError("artifact reconstruction")
    manifest = predecessor_load_json(manifest_raw)
    if (
        not isinstance(manifest, dict)
        or canonical(manifest) + b"\n" != manifest_raw
        or manifest.get("sources") != _source_digests()
    ):
        raise ValueError("artifact manifest")
    return result


def finalize(
    repo: Path,
    output: Path,
    partial_dir: Path,
    binding: dict[str, str],
    local_comparison: list[dict[str, Any]],
    prior_use: Decimal,
) -> dict[str, Any]:
    """Credential-free publication. It cannot dispatch or consume a new claim."""
    _remove_writer_temporaries(output)
    _artifact_entries(output, complete=False)
    claim = validate_claim(repo, binding["prior_ledger"], binding)
    partial, _, charge, samples = validate_partial_artifact(
        repo, partial_dir, binding, local_comparison
    )
    journal = Journal(output / "cost-journal.jsonl", binding["phase4c3_partial"])
    result = _with_partial_hashes(
        reconstruct(journal, partial, charge, prior_use, samples, binding, local_comparison),
        partial_dir,
    )
    result_raw, report_raw, journal_raw = (
        canonical(result) + b"\n",
        _report(result),
        (output / "cost-journal.jsonl").read_bytes(),
    )
    _atomic(output / "result.json", result_raw)
    _atomic(output / "report.md", report_raw)
    _atomic(
        output / "artifact-manifest.json",
        _manifest(result_raw, report_raw, journal_raw, binding, claim),
    )
    return validate_artifact(repo, output, partial_dir, binding, local_comparison, prior_use)


def run_prevalidated(
    repo: Path,
    output: Path,
    partial_dir: Path,
    binding: dict[str, str],
    local_comparison: list[dict[str, Any]],
    prior_use: Decimal,
    *,
    api_key: str,
    transport: Transport,
) -> dict[str, Any]:
    """Private test seam; production calls it only after complete offline preflight."""
    if not api_key:
        raise ValueError("TYPESAFE_API_KEY required")
    _, _, predecessor_charge, _ = validate_partial_artifact(
        repo, partial_dir, binding, local_comparison
    )
    if prior_use + predecessor_charge + RESERVATION * 47 > BUDGET:
        raise ValueError("cumulative budget envelope")
    requests = materialize_requests()
    plan = validate_plan()
    create_output(output)
    try:
        create_claim(repo, binding["prior_ledger"], binding)
    except Exception:
        with suppress(FileNotFoundError):
            (output / "cost-journal.jsonl").unlink()
        with suppress(OSError):
            output.rmdir()
        raise
    journal = Journal(output / "cost-journal.jsonl", binding["phase4c3_partial"], plan)
    for ordinal in range(FIRST_ORDINAL, LAST_ORDINAL + 1):
        if prior_use + predecessor_charge + journal.charged() + RESERVATION > BUDGET:
            journal.append(
                "halt", ordinal, category="budget_uncertain", response_invalid_detail=None
            )
            break
        meta = requests[ordinal]
        if _hash(meta["wire"]) != plan["wire_sha256_by_ordinal"][str(ordinal)]:
            raise ValueError("wire digest drift")
        journal.append(
            "reserve", ordinal, reservation_cost_usd="0.0032", wire_sha256=_hash(meta["wire"])
        )
        try:
            validated, latency = dispatch(meta, api_key, transport)
            reduced = sample(meta, validated, latency)
            kind = "settled_over_limit" if validated["input_tokens"] > 64000 else "settle"
            journal.append(
                kind, ordinal, sample=reduced, computed_cost_usd=reduced["computed_cost_usd"]
            )
            if kind == "settled_over_limit":
                break
        except ResponseError as error:
            if error.category == "model_drift":
                journal.append(
                    "failure", ordinal, category="model_drift", response_invalid_detail=None
                )
            else:
                journal.append(
                    "failure",
                    ordinal,
                    category="response_invalid",
                    response_invalid_detail=error.category,
                )
            break
        except TimeoutError:
            journal.append("failure", ordinal, category="timeout", response_invalid_detail=None)
            break
        except RuntimeError as error:
            category = str(error)
            journal.append(
                "failure",
                ordinal,
                category=category if category in FAILURES else "transport_error",
                response_invalid_detail=None,
            )
            break
        except Exception:
            journal.append(
                "failure", ordinal, category="transport_error", response_invalid_detail=None
            )
            break
    return finalize(repo, output, partial_dir, binding, local_comparison, prior_use)


_run_prevalidated = run_prevalidated
