"""Dataset I/O, backend execution, and aggregate-only report generation."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import os
import platform
import secrets
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from pydantic import JsonValue, ValidationError

from benchmarks.ptbr_native.models import (
    BenchmarkReportAny,
    BenchmarkReportV2,
    load_position_ensemble_manifest,
    load_protocol_manifest,
    load_report,
)
from benchmarks.ptbr_native.protocol import (
    OPTION_IDS,
    build_plan,
    build_request,
    earliest_argmax,
    percentile,
)
from saracura.backends.julia import (
    JULIA_CHECKPOINT_SHA256,
    JULIA_MODEL_ID,
    JULIA_MODEL_REVISION,
    JuliaBackend,
)
from saracura.backends.saracura_universal import (
    SaracuraUniversalBackend,
    load_saracura_candidate,
)
from saracura.contracts.errors import ErrorCode, SaracuraError
from saracura.runtime.engine import DecisionEngine
from saracura.runtime.workflows import WorkflowRegistry
from saracura.serialization import canonical_json_bytes

MANIFEST_PATH = Path(__file__).parents[1] / "manifests" / "phase5d-ptbr-native.v1.json"
DATA_FILES = {
    "corpus/test-00000-of-00001.parquet": (
        "378c43e9126a31419680d66c4e43a178a66c8da45a98f64b89af94afb6ecd3e0"
    ),
    "queries/test-00000-of-00001.parquet": (
        "5467fc92f2387b436526b609463cdb7f2251fb684667ac9fb4f589aba2e528c0"
    ),
    "qrels/test-00000-of-00001.parquet": (
        "17c173c566e5e021a4ead335717e97927f925df4f82994ccf4a987aeb8a6affc"
    ),
}
_RELEASE_PATHS = (
    "benchmarks/ptbr_native",
    "benchmarks/manifests/phase5d-ptbr-native.v1.json",
    "benchmarks/manifests/phase5d1-position-ensemble.v1.json",
    "benchmarks/manifests/training-data-source-policies.v4.json",
    "benchmarks/data_policy_registry.py",
    "benchmarks/validate_manifests.py",
    "src/saracura",
    "pyproject.toml",
    "uv.lock",
)


def _sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_dataset(data_root: Path) -> tuple[list[dict[str, Any]], ...]:
    if not data_root.is_absolute() or not data_root.is_dir():
        raise ValueError("data root must be an existing absolute directory")
    resolved = data_root.resolve(strict=True)
    paths: list[Path] = []
    for relative, expected in DATA_FILES.items():
        path = resolved / relative
        if not path.is_file():
            raise ValueError("dataset file is missing")
        if _sha256_file(path) != expected:
            raise ValueError("dataset file digest drifted")
        paths.append(path)
    try:
        parquet = importlib.import_module("pyarrow.parquet")
    except ImportError as error:
        raise ValueError("install the ptbr-benchmark extra to read Parquet") from error
    tables = [parquet.read_table(path) for path in paths]
    expected_columns = (
        ["_id", "title", "text"],
        ["_id", "text"],
        ["query-id", "corpus-id", "score"],
    )
    for table, columns in zip(tables, expected_columns, strict=True):
        if table.column_names != columns:
            raise ValueError("dataset columns drifted")
    return tuple(table.to_pylist() for table in tables)


def _git_output(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def release_code_sha256(root: Path, commit: str = "HEAD") -> str:
    listing = subprocess.run(
        ["git", "ls-tree", "-r", "-z", commit, "--", *_RELEASE_PATHS],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    files: list[tuple[str, str]] = []
    for entry in listing.split(b"\0"):
        if not entry:
            continue
        metadata, raw_path = entry.split(b"\t", 1)
        mode, kind, object_id = metadata.decode("ascii").split(" ")
        relative = raw_path.decode("utf-8")
        if kind == "blob" and mode in {"100644", "100755", "120000"}:
            files.append((relative, object_id))
    digest = hashlib.sha256()
    for relative, object_id in sorted(files):
        encoded = relative.encode("utf-8")
        content = subprocess.run(
            ["git", "cat-file", "blob", object_id],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _assert_release_paths_clean(root: Path) -> None:
    status = _git_output(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--ignored=matching",
        "--",
        *_RELEASE_PATHS,
    )
    changed = []
    for line in status.splitlines():
        relative = line[3:].strip().strip('"')
        if "__pycache__" in Path(relative).parts or relative.endswith(".pyc"):
            continue
        changed.append(relative)
    if changed:
        raise ValueError("benchmark release paths have tracked, untracked, or ignored changes")


def _git_blob_sha256(root: Path, commit: str, path: str) -> str:
    content = subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    return hashlib.sha256(content).hexdigest()


def _dependency_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in ("saracura", "pydantic", "rfc8785", "pyarrow", "torch", "transformers"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def _backend(
    name: str,
    *,
    model_snapshot: Path | None,
    encoder_snapshot: Path | None,
    training_capsule: Path | None,
    inference_strategy: str = "single_pass",
) -> JuliaBackend | SaracuraUniversalBackend:
    if name == "julia":
        if model_snapshot is None or not model_snapshot.is_absolute():
            raise ValueError("Julia requires an absolute model snapshot")
        return JuliaBackend(
            model_snapshot=model_snapshot,
            device="cpu",
            position_ensemble=inference_strategy == "cyclic_mean",
        )
    if name == "saracura-universal":
        if (
            encoder_snapshot is None
            or training_capsule is None
            or not encoder_snapshot.is_absolute()
            or not training_capsule.is_absolute()
        ):
            raise ValueError("Saracura requires absolute encoder and capsule paths")
        return SaracuraUniversalBackend(
            encoder_snapshot=encoder_snapshot,
            training_capsule=training_capsule,
            device="cpu",
        )
    raise ValueError("unsupported benchmark backend")


def _category(error: BaseException) -> tuple[str, bool]:
    if isinstance(error, SaracuraError):
        code = error.payload.code
        if code is ErrorCode.CAPACITY_EXCEEDED and error.payload.path == "/state":
            return "request_capacity", True
        if code in {ErrorCode.CAPACITY_EXCEEDED, ErrorCode.CARDINALITY_EXCEEDED}:
            return "backend_capacity", True
        if code is ErrorCode.BACKEND_UNAVAILABLE:
            if isinstance(error.__cause__, ValueError) and str(error.__cause__) == (
                "Option exceeds 48-token model contract"
            ):
                return "backend_capacity", True
            return "backend_unavailable", False
        return "other_error", False
    if isinstance(error, (ValidationError, ValueError, TypeError)):
        return "invalid_response", False
    return "other_error", False


def run_benchmark(
    *,
    backend_name: str,
    data_root: Path,
    model_snapshot: Path | None = None,
    encoder_snapshot: Path | None = None,
    training_capsule: Path | None = None,
    inference_strategy: str = "single_pass",
) -> BenchmarkReportAny:
    root = Path(__file__).resolve().parents[2]
    _assert_release_paths_clean(root)
    if inference_strategy not in {"single_pass", "cyclic_mean"}:
        raise ValueError("unsupported inference strategy")
    if inference_strategy == "cyclic_mean" and backend_name != "julia":
        raise ValueError("cyclic_mean strategy is supported only by Julia")
    manifest_raw = MANIFEST_PATH.read_bytes()
    load_protocol_manifest(manifest_raw)
    ensemble_manifest_raw = None
    if inference_strategy == "cyclic_mean":
        ensemble_manifest_raw = (
            MANIFEST_PATH.parent / "phase5d1-position-ensemble.v1.json"
        ).read_bytes()
        load_position_ensemble_manifest(ensemble_manifest_raw)
    plan = build_plan(*load_dataset(data_root))
    backend = _backend(
        backend_name,
        model_snapshot=model_snapshot,
        encoder_snapshot=encoder_snapshot,
        training_capsule=training_capsule,
        inference_strategy=inference_strategy,
    )
    engine = DecisionEngine(backend=backend, workflows=WorkflowRegistry(()), calibrations={})
    valid = correct = rejected = errors = divergences = 0
    positions = [dict(planned=0, valid=0, correct=0) for _ in range(4)]
    categories = {
        "request_capacity": 0,
        "backend_capacity": 0,
        "backend_unavailable": 0,
        "invalid_response": 0,
        "other_error": 0,
    }
    latencies: list[float] = []
    started = time.perf_counter()
    try:
        for row in plan:
            positions[row.gold_position]["planned"] += 1
            request = build_request(
                row,
                model=backend.model.revision,
                backend=backend_name,
                strategy=inference_strategy,
            )
            row_started = time.perf_counter()
            try:
                response = engine.decide(request)
                answer = response.answers[0]
                values = [answer.raw_scores[option] for option in OPTION_IDS]
                predicted = earliest_argmax(values)
                if answer.value != OPTION_IDS[predicted]:
                    divergences += 1
                valid += 1
                positions[row.gold_position]["valid"] += 1
                if predicted == row.gold_position:
                    correct += 1
                    positions[row.gold_position]["correct"] += 1
            except Exception as error:
                category, is_reject = _category(error)
                categories[category] += 1
                if is_reject:
                    rejected += 1
                else:
                    errors += 1
            finally:
                latencies.append((time.perf_counter() - row_started) * 1000)
    finally:
        backend.close()
    elapsed = time.perf_counter() - started
    code_commit = _git_output(root, "rev-parse", "HEAD")
    counts = {
        "planned": 373,
        "valid": valid,
        "correct": correct,
        "incorrect": valid - correct,
        "rejected": rejected,
        "error": errors,
    }
    report = {
        "schema_version": (
            "phase5d-ptbr-faq-bacen-report.v2"
            if inference_strategy == "cyclic_mean"
            else "phase5d-ptbr-faq-bacen-report.v1"
        ),
        "benchmark_id": "phase5d-ptbr-faq-bacen",
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "code_commit": code_commit,
        "release_code_sha256": release_code_sha256(root, code_commit),
        "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "protocol_plan_sha256": "089a83874ed0cc57da45eb143f4b6f6494125dbfabddb660f8e7ca5105175f6f",
        "backend": backend_name,
        "model_id": backend.model.id,
        "model_revision": backend.model.revision,
        "checkpoint_sha256": backend.model.checkpoint_sha256,
        "device": "cpu",
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "dependencies": _dependency_versions(),
        "counts": counts,
        "planned_top1_accuracy": correct / 373,
        "coverage": valid / 373,
        "valid_top1_accuracy": None if valid == 0 else correct / valid,
        "by_gold_position": [
            {
                "position": index,
                **item,
                "planned_top1_accuracy": item["correct"] / item["planned"],
            }
            for index, item in enumerate(positions)
        ],
        "latency": {
            "cold_request_ms": latencies[0] if latencies else None,
            "warm_p50_ms": percentile(latencies[1:], 0.50),
            "warm_p95_ms": percentile(latencies[1:], 0.95),
            "throughput_rows_per_second": 373 / elapsed if elapsed else 0.0,
        },
        "chance_accuracy": 0.25,
        "lexical_baseline_correct": 265,
        "lexical_baseline_accuracy": 265 / 373,
        "rejection_categories": categories,
        "backend_choice_argmax_divergences": divergences,
        "calibration_status": "uncalibrated",
        "automation_allowed": False,
        "limitations": [
            "Single four-way Banco Central FAQ answer-selection task; not a general quality claim.",
            "Julia-1 training overlap with this public dataset is unknown.",
            "The Saracura-owned ranker remains synthetic_only_research.",
        ],
    }
    if inference_strategy == "cyclic_mean":
        assert ensemble_manifest_raw is not None
        report.update(
            {
                "inference_strategy": "cyclic_mean",
                "inferences_per_valid_decision": 4,
                "position_ensemble_manifest_sha256": hashlib.sha256(
                    ensemble_manifest_raw
                ).hexdigest(),
            }
        )
    return load_report(canonical_json_bytes(cast(JsonValue, report)))


def atomic_write_report(path: Path, report: BenchmarkReportAny) -> None:
    if not path.is_absolute() or not path.parent.is_dir():
        raise ValueError("output must be an absolute path in an existing directory")
    raw = canonical_json_bytes(cast(JsonValue, report.model_dump(mode="json"))) + b"\n"
    temporary = path.with_name(f".{path.name}.tmp-{secrets.token_hex(8)}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        try:
            view = memoryview(raw)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("short report write")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def verify_report(path: Path) -> BenchmarkReportAny:
    if not path.is_file():
        raise ValueError("report must be an existing file")
    report = load_report(path.read_bytes())
    root = Path(__file__).resolve().parents[2]
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", report.code_commit, "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if ancestor.returncode != 0:
        raise ValueError("report code commit is not an ancestor of the checkout")
    strategy = getattr(report, "inference_strategy", None)
    expected_names = {
        ("phase5d-ptbr-faq-bacen-report.v1", "julia", None): (
            "phase5d-ptbr-faq-bacen-julia-cpu.json"
        ),
        ("phase5d-ptbr-faq-bacen-report.v1", "saracura-universal", None): (
            "phase5d-ptbr-faq-bacen-saracura-universal-cpu.json"
        ),
        ("phase5d-ptbr-faq-bacen-report.v2", "julia", "cyclic_mean"): (
            "phase5d1-ptbr-faq-bacen-julia-cyclic-mean-cpu.json"
        ),
    }
    if path.name != expected_names.get((report.schema_version, report.backend, strategy)):
        raise ValueError("report filename does not match its schema, backend, and strategy")
    expected_manifest = _git_blob_sha256(
        root, report.code_commit, "benchmarks/manifests/phase5d-ptbr-native.v1.json"
    )
    if report.manifest_sha256 != expected_manifest:
        raise ValueError("report manifest digest does not match its historical Git tree")
    if report.release_code_sha256 != release_code_sha256(root, report.code_commit):
        raise ValueError("report code digest does not match its historical Git tree")
    if isinstance(report, BenchmarkReportV2):
        ensemble_manifest = _git_blob_sha256(
            root,
            report.code_commit,
            "benchmarks/manifests/phase5d1-position-ensemble.v1.json",
        )
        if report.position_ensemble_manifest_sha256 != ensemble_manifest:
            raise ValueError("report ensemble manifest digest does not match its Git tree")
    model_identity = (
        (JULIA_MODEL_ID, JULIA_MODEL_REVISION, JULIA_CHECKPOINT_SHA256)
        if report.backend == "julia"
        else (
            (candidate := load_saracura_candidate()).id,
            candidate.model_revision,
            candidate.checkpoint_sha256,
        )
    )
    if (
        report.model_id,
        report.model_revision,
        report.checkpoint_sha256,
    ) != model_identity:
        raise ValueError("report model identity does not match the checkout")
    return report
