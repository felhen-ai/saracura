"""Run the local, CPU-only Phase 2A decision-scaling benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import NoReturn, cast

from pydantic import JsonValue

from benchmarks.cases import calibrations_for, request_for
from benchmarks.contracts import (
    BenchmarkResult,
    EnvironmentCapture,
    Limitations,
    ManifestIdentity,
    ModelIdentity,
    RunIdentity,
    Sample,
    TimingBoundaries,
    WorkloadDefinition,
    WorkloadResult,
    summary_from_values,
)
from benchmarks.io import write_result
from benchmarks.validate_manifests import validate_manifest
from saracura import __version__
from saracura.backends import DeterministicFixtureBackend
from saracura.backends.base import (
    Backend,
    BackendCalibrationMetadata,
    BackendCapabilities,
    EncodedState,
    ScoredChoice,
)
from saracura.backends.fixture import (
    FIXTURE_BACKEND_ID,
    FIXTURE_BACKEND_REVISION,
    FIXTURE_MODEL_ID,
    FIXTURE_MODEL_REVISION,
)
from saracura.contracts.models import ChoiceQuestion, DecisionResponse, ModelReference
from saracura.runtime import DecisionEngine, default_workflows
from saracura.runtime.engine import FIXTURE_SPLIT_MANIFEST_SHA256
from saracura.serialization import canonical_json_bytes, serialize_state

ROOT = Path(__file__).parents[1]
MANIFEST_PATH = ROOT / "benchmarks/manifests/ptbr-fixture-v1.json"
PUBLIC_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise ValueError("invalid command arguments")


class CountingBackend:
    """Public Backend decorator that counts only harness-observed encodings."""

    def __init__(self, backend: Backend) -> None:
        self._backend = backend
        self.encode_calls = 0

    @property
    def capabilities(self) -> BackendCapabilities:
        return self._backend.capabilities

    @property
    def model(self) -> ModelReference:
        return self._backend.model

    @property
    def calibration_metadata(self) -> BackendCalibrationMetadata:
        return self._backend.calibration_metadata

    def encode_state(self, state_payload: bytes) -> EncodedState:
        self.encode_calls += 1
        return self._backend.encode_state(state_payload)

    def score_choice(
        self,
        encoded_state: EncodedState,
        question: ChoiceQuestion,
        question_payload: bytes,
    ) -> ScoredChoice:
        return self._backend.score_choice(encoded_state, question, question_payload)


def _manifest_identity() -> ManifestIdentity:
    payload = MANIFEST_PATH.read_bytes()
    validate_manifest(MANIFEST_PATH)
    digest = hashlib.sha256(payload).hexdigest()
    if digest != FIXTURE_SPLIT_MANIFEST_SHA256:
        raise ValueError("manifest validation failed")
    metadata = json.loads(payload)
    return ManifestIdentity(id=metadata["id"], revision=metadata["revision"], sha256=digest)


def _environment(hardware_label: str | None) -> EnvironmentCapture:
    return EnvironmentCapture(
        os_family=_public_platform_value(platform.system()),
        os_release=_public_platform_value(platform.release()),
        architecture=_public_platform_value(platform.machine()),
        python_implementation=_public_platform_value(platform.python_implementation()),
        python_version=_public_platform_value(platform.python_version()),
        saracura_version=_public_platform_value(__version__),
        backend_revision=FIXTURE_BACKEND_REVISION,
        hardware_label=hardware_label,
    )


def _public_platform_value(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("._-")
    return (normalized or "unknown")[:128]


def _answer_digest(response: DecisionResponse) -> str:
    answers = [answer.model_dump(mode="json") for answer in response.answers]
    return hashlib.sha256(canonical_json_bytes(cast(JsonValue, answers))).hexdigest()


def _answer_payloads(response: DecisionResponse) -> dict[str, bytes]:
    return {
        answer.question_id: canonical_json_bytes(cast(JsonValue, answer.model_dump(mode="json")))
        for answer in response.answers
    }


def _validate_arguments(
    *, warmups: int, iterations: int, code_revision: str, hardware_label: str | None
) -> None:
    if not 0 <= warmups <= 100 or not 1 <= iterations <= 1000:
        raise ValueError("benchmark configuration is invalid")
    for value in (code_revision, hardware_label):
        if value is not None and not PUBLIC_IDENTIFIER.fullmatch(value):
            raise ValueError("public identifier is invalid")


def run_benchmark(
    *, warmups: int, iterations: int, code_revision: str, hardware_label: str | None
) -> BenchmarkResult:
    _validate_arguments(
        warmups=warmups,
        iterations=iterations,
        code_revision=code_revision,
        hardware_label=hardware_label,
    )
    manifest = _manifest_identity()
    workloads: list[WorkloadResult] = []
    state_hashes: set[str] = set()
    common_answers: dict[str, bytes] | None = None
    for question_count in (1, 10, 50):
        request = request_for(question_count)
        state_hash = hashlib.sha256(serialize_state(request)).hexdigest()
        state_hashes.add(state_hash)
        backend = CountingBackend(DeterministicFixtureBackend())
        engine = DecisionEngine(
            backend=backend,
            workflows=default_workflows(),
            calibrations=calibrations_for(request, backend),
        )
        for _ in range(warmups):
            warmup_before = backend.encode_calls
            engine.decide(request, include_timing=True)
            if backend.encode_calls - warmup_before != 1:
                raise RuntimeError("harness invariant failed")
        samples: list[Sample] = []
        for sample_index in range(iterations):
            before = backend.encode_calls
            response = engine.decide(request, include_timing=True)
            delta = backend.encode_calls - before
            if delta != 1:
                raise RuntimeError("harness invariant failed")
            if response.timing is None:
                raise RuntimeError("harness invariant failed")
            payloads = _answer_payloads(response)
            if common_answers is None:
                common_answers = payloads
            else:
                for question_id, previous in common_answers.items():
                    current = payloads.get(question_id)
                    if current is not None and current != previous:
                        raise RuntimeError("harness invariant failed")
                common_answers.update(payloads)
            samples.append(
                Sample(
                    phase="measured",
                    sample_index=sample_index,
                    state_encoding_count=delta,
                    timing=TimingBoundaries(
                        total_ms=response.timing.total_ms,
                        state_encoding_ms=response.timing.state_encoding_ms,
                        decision_ms=response.timing.decision_ms,
                    ),
                    answer_count=len(response.answers),
                    answer_sha256=_answer_digest(response),
                )
            )
        total = [sample.timing.total_ms for sample in samples]
        encoding = [sample.timing.state_encoding_ms for sample in samples]
        decision = [sample.timing.decision_ms for sample in samples]
        workloads.append(
            WorkloadResult(
                workload=WorkloadDefinition(
                    question_count=question_count,
                    warmup_iterations=warmups,
                    measured_iterations=iterations,
                    state_sha256=state_hash,
                    fixture_manifest=manifest,
                ),
                samples=tuple(samples),
                total=summary_from_values(total),
                state_encoding=summary_from_values(encoding),
                decision=summary_from_values(decision),
            )
        )
    if len(state_hashes) != 1 or common_answers is None:
        raise RuntimeError("harness invariant failed")
    backend_ref = DeterministicFixtureBackend().model
    return BenchmarkResult(
        schema_version="phase2a.v1",
        run=RunIdentity(
            suite="decision-scaling",
            schema_version="phase2a.v1",
            run_id=str(uuid.uuid4()),
            created_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            code_revision=code_revision,
            backend_id=FIXTURE_BACKEND_ID,
            backend_revision=FIXTURE_BACKEND_REVISION,
            model=ModelIdentity(
                id=FIXTURE_MODEL_ID,
                revision=FIXTURE_MODEL_REVISION,
                checkpoint_sha256=backend_ref.checkpoint_sha256,
            ),
            environment=_environment(hardware_label),
        ),
        workloads=tuple(workloads),
        limitations=Limitations(),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = SafeArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=["decision-scaling"], required=True)
    parser.add_argument("--backend", choices=["fixture"], required=True)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--hardware-label")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        result = run_benchmark(
            warmups=args.warmups,
            iterations=args.iterations,
            code_revision=args.code_revision,
            hardware_label=args.hardware_label,
        )
        write_result(result, args.output_dir)
    except ValueError:
        print("benchmark failed: invalid configuration or manifest", file=sys.stderr)
        return 2
    except OSError:
        print("benchmark failed: filesystem operation", file=sys.stderr)
        return 2
    except RuntimeError:
        print("benchmark failed: runtime invariant", file=sys.stderr)
        return 2
    except Exception:
        print("benchmark failed: unexpected internal error", file=sys.stderr)
        return 2
    print("benchmark completed: artifacts written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
