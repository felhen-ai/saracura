"""Immutable benchmark artifact writes and deterministic report rendering."""

from __future__ import annotations

import os
import tempfile
from contextlib import suppress
from pathlib import Path

from benchmarks.contracts import BenchmarkResult
from saracura.serialization import canonical_json_bytes


def atomic_create(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            with suppress(OSError):
                os.fsync(handle.fileno())
        os.link(temporary, path)
        _fsync_directory(path.parent)
    except FileExistsError as error:
        raise FileExistsError(f"benchmark artifact already exists: {path.name}") from error
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()


def _fsync_directory(directory: Path) -> None:
    try:
        descriptor = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        with suppress(OSError):
            os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_result(result: BenchmarkResult, output_dir: Path) -> tuple[Path, Path]:
    raw_path = output_dir / f"raw-{result.run.run_id}.json"
    report_path = output_dir / f"report-{result.run.run_id}.md"
    raw = canonical_json_bytes(result.model_dump(mode="json")) + b"\n"
    atomic_create(raw_path, raw)
    try:
        atomic_create(report_path, render_report(result).encode("utf-8"))
    except BaseException:
        with suppress(FileNotFoundError):
            raw_path.unlink()
            _fsync_directory(output_dir)
        raise
    return raw_path, report_path


def _md(value: str) -> str:
    return value.replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def render_report(result: BenchmarkResult) -> str:
    env = result.run.environment
    lines = [
        "# Saracura benchmark — decision scaling",
        "",
        f"- Schema: `{_md(result.schema_version)}`",
        f"- Run: `{_md(result.run.run_id)}`",
        f"- Code revision: `{_md(result.run.code_revision)}`",
        f"- Backend: `{_md(result.run.backend_id)}@{_md(result.run.backend_revision)}`",
        f"- Saracura version: `{_md(env.saracura_version)}`",
        (
            f"- Model: `{_md(result.run.model.id)}@{_md(result.run.model.revision)}` "
            f"(checkpoint `{_md(result.run.model.checkpoint_sha256)}`)"
        ),
        f"- Hardware label: `{_md(env.hardware_label or 'not provided')}`",
        (
            f"- Manifest: `{_md(result.workloads[0].workload.fixture_manifest.id)}@"
            f"{_md(result.workloads[0].workload.fixture_manifest.revision)}` "
            f"(sha256 `{_md(result.workloads[0].workload.fixture_manifest.sha256)}`)"
        ),
        f"- State sha256: `{_md(result.workloads[0].workload.state_sha256)}`",
        (
            f"- Environment: `{_md(env.os_family)} {_md(env.os_release)}`, "
            f"`{_md(env.architecture)}`, "
            f"`{_md(env.python_implementation)} {_md(env.python_version)}`"
        ),
        "",
        (
            "| Workload | Samples | Total p50 (ms) | Total p95 (ms) | "
            "Encoding p50 (ms) | Encoding p95 (ms) | Decision p50 (ms) | "
            "Decision p95 (ms) | Encode calls/sample |"
        ),
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for workload in result.workloads:
        lines.append(
            "| Q={} | {} | {:.6f} | {:.6f} | {:.6f} | {:.6f} | {:.6f} | {:.6f} | {} |".format(
                workload.workload.question_count,
                workload.total.count,
                workload.total.p50_ms,
                workload.total.p95_ms,
                workload.state_encoding.p50_ms,
                workload.state_encoding.p95_ms,
                workload.decision.p50_ms,
                workload.decision.p95_ms,
                ", ".join(str(sample.state_encoding_count) for sample in workload.samples),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            f"{_md(result.limitations.interpretation)}",
            "",
        ]
    )
    return "\n".join(lines)
