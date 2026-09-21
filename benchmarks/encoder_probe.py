"""Fixed, encoder-only probe and closed phase2b.v1 evidence contract."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal, TypeVar, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from benchmarks.encoder_acquisition import snapshot_path
from benchmarks.encoder_loader import VerifiedSnapshot, load_encoder
from benchmarks.encoder_registry import get_candidate, registry_digest

CORPUS_PATH = Path(__file__).parent / "fixtures" / "encoder-probe-corpus.v1.json"
CORPUS_DIGEST = "f5d09a8b1d968579dfd31c656c855b83530c8dac1ffde866cdc380792e7f6b64"
SAMPLES = 20
WARMUPS = 3
MAX_LENGTH = 128
Safe = Annotated[str, Field(min_length=1, max_length=120, pattern=r"^[A-Za-z0-9._+ -]+$")]
Repository = Annotated[str, Field(pattern=r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")]
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
T = TypeVar("T")


class Closed(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)


class FileEvidence(Closed):
    path: Safe
    bytes: Annotated[int, Field(gt=0)]
    sha256: Digest


class CandidateEvidence(Closed):
    id: Safe
    repository: Repository
    revision: Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
    declared_license: Literal["MIT", "Apache-2.0"]
    registry_sha256: Digest
    verified_files: tuple[FileEvidence, ...]
    model_type: Literal["xlm-roberta", "bert"]
    architecture: Literal["XLMRobertaModel", "BertModel"]
    hidden_width: Literal[384, 768]
    loader: Literal["xlm-roberta", "bert"]
    tokenizer: Literal["XLMRobertaTokenizerFast", "PreTrainedTokenizerFast"]


class CorpusEvidence(Closed):
    id: Literal["encoder-probe-corpus.v1"]
    sha256: Digest
    case_ids: tuple[Literal["ptbr-support-refund", "en-support-refund"], ...]
    locales: tuple[Literal["pt-BR", "en"], ...]

    @model_validator(mode="after")
    def fixed_digest(self) -> CorpusEvidence:
        if self.sha256 != CORPUS_DIGEST:
            raise ValueError("probe corpus digest mismatch")
        return self

    @model_validator(mode="after")
    def fixed_order(self) -> CorpusEvidence:
        if self.case_ids != ("ptbr-support-refund", "en-support-refund"):
            raise ValueError("probe corpus case order is not canonical")
        if self.locales != ("pt-BR", "en"):
            raise ValueError("probe corpus locale order is not canonical")
        return self


class Protocol(Closed):
    warmup_iterations: Literal[3]
    measured_iterations: Literal[20]
    max_length: Literal[128]
    padding: Literal["True"]
    truncation: Literal["True"]
    pooling: Literal["attention-mask-weighted-mean-float32-no-normalization"]
    measured_boundary: Literal["tokenization-device-transfer-forward-pooling"]


class Environment(Closed):
    os_family: Safe
    os_release: Safe
    architecture: Safe
    machine_class: Safe
    python_version: Safe
    saracura_version: Safe
    torch_version: Safe
    transformers_version: Safe
    tokenizers_version: Safe


class Sample(Closed):
    latency_ms: float = Field(ge=0)
    embedding_sha256: Digest


class ProbeResult(Closed):
    schema_version: Literal["phase2b.v1"]
    run_id: str
    created_at: str
    candidate: CandidateEvidence
    corpus: CorpusEvidence
    protocol: Protocol
    environment: Environment
    device: Literal["cpu", "mps"]
    pooling_dtype: Literal["float32"]
    fresh_process_load_ms: float = Field(ge=0)
    parameter_count: Annotated[int, Field(gt=0)]
    snapshot_bytes: Annotated[int, Field(gt=0)]
    rss_before_load: Annotated[int, Field(gt=0)]
    rss_after_load: Annotated[int, Field(gt=0)]
    rss_after_probe: Annotated[int, Field(gt=0)]
    token_lengths: tuple[Annotated[int, Field(gt=0)], Annotated[int, Field(gt=0)]]
    hidden_width: Annotated[int, Field(gt=0)]
    truncated: Literal[False]
    embedding_sha256: Digest
    samples: tuple[Sample, ...]
    p50_ms: float = Field(ge=0)
    p95_ms: float = Field(ge=0)

    @model_validator(mode="after")
    def consistent(self) -> ProbeResult:
        try:
            parsed = uuid.UUID(self.run_id)
            created = datetime.fromisoformat(self.created_at.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError("run identity is not canonical") from error
        if (
            str(parsed) != self.run_id
            or parsed.version != 4
            or created.tzinfo != UTC
            or not self.created_at.endswith("Z")
        ):
            raise ValueError("run identity is not canonical")
        if len(self.samples) != SAMPLES or any(sample.latency_ms < 0 for sample in self.samples):
            raise ValueError("sample count is not frozen")
        digests = {sample.embedding_sha256 for sample in self.samples}
        if digests != {self.embedding_sha256}:
            raise ValueError("sample digests do not match top-level digest")
        values = sorted(sample.latency_ms for sample in self.samples)
        if (
            self.p50_ms != values[math.ceil(0.50 * SAMPLES) - 1]
            or self.p95_ms != values[math.ceil(0.95 * SAMPLES) - 1]
        ):
            raise ValueError("percentiles do not match raw samples")
        candidate = get_candidate(self.candidate.id)
        if (
            self.candidate.repository != candidate.repository
            or self.candidate.revision != candidate.revision
            or self.candidate.declared_license != candidate.license
            or self.candidate.registry_sha256 != registry_digest()
            or self.candidate.model_type != candidate.model_type
            or self.candidate.architecture != candidate.architecture
            or self.candidate.hidden_width != candidate.hidden_width
            or self.candidate.loader != candidate.loader
            or self.candidate.tokenizer != candidate.tokenizer
            or [item.model_dump() for item in self.candidate.verified_files]
            != [item.model_dump() for item in candidate.files]
        ):
            raise ValueError("candidate provenance does not match registry")
        if self.snapshot_bytes != candidate.total_bytes:
            raise ValueError("snapshot bytes do not match registry")
        if self.hidden_width != candidate.hidden_width:
            raise ValueError("embedding width does not match registry")
        if self.protocol != Protocol(
            warmup_iterations=3,
            measured_iterations=20,
            max_length=128,
            padding="True",
            truncation="True",
            pooling="attention-mask-weighted-mean-float32-no-normalization",
            measured_boundary="tokenization-device-transfer-forward-pooling",
        ):
            raise ValueError("probe protocol is not canonical")
        return self


def _nearest(values: list[float], percentile: float) -> float:
    return sorted(values)[max(1, math.ceil(percentile * len(values))) - 1]


def _rss() -> int:
    import psutil  # type: ignore[import-untyped]

    return int(psutil.Process().memory_info().rss)


def _digest_tensor(tensor: Any) -> str:
    import numpy as np  # type: ignore[import-not-found]

    data = tensor.detach().float().cpu().contiguous().numpy().astype("<f4", copy=False)
    return hashlib.sha256(np.ascontiguousarray(data).tobytes(order="C")).hexdigest()


def _timed_region(
    operation: Callable[[], T],
    synchronize: Callable[[], None],
    clock: Callable[[], float] = time.perf_counter,
) -> tuple[T, float]:
    synchronize()
    started = clock()
    value = operation()
    synchronize()
    return value, (clock() - started) * 1000


def _environment() -> Environment:
    import tokenizers  # type: ignore[import-not-found]
    import torch  # type: ignore[import-not-found]
    import transformers  # type: ignore[import-not-found]

    import saracura

    def clean(value: str) -> str:
        value = re.sub(r"[^A-Za-z0-9._+ -]", "_", value)
        return value[:120] or "unknown"

    return Environment(
        os_family=clean(platform.system()),
        os_release=clean(platform.release()),
        architecture=clean(platform.machine()),
        machine_class=clean(platform.platform().split("-")[0]),
        python_version=clean(platform.python_version()),
        saracura_version=clean(saracura.__version__),
        torch_version=clean(torch.__version__),
        transformers_version=clean(transformers.__version__),
        tokenizers_version=clean(tokenizers.__version__),
    )


def _load_corpus() -> tuple[list[str], CorpusEvidence]:
    raw = CORPUS_PATH.read_bytes()
    if hashlib.sha256(raw).hexdigest() != CORPUS_DIGEST:
        raise ValueError("probe corpus digest mismatch")
    payload = json.loads(raw)
    cases = payload["cases"]
    if payload["schema_version"] != "encoder-probe-corpus.v1" or len(cases) != 2:
        raise ValueError("probe corpus is invalid")
    return [str(case["text"]) for case in cases], CorpusEvidence(
        id="encoder-probe-corpus.v1",
        sha256=CORPUS_DIGEST,
        case_ids=("ptbr-support-refund", "en-support-refund"),
        locales=("pt-BR", "en"),
    )


def probe(
    candidate_id: str, device: Literal["cpu", "mps"], output_dir: Path, *, root: Path | None = None
) -> tuple[Path, Path]:
    candidate = get_candidate(candidate_id)
    if candidate.disposition != "eligible":
        raise ValueError("candidate is blocked by the reviewed registry")
    assert (
        candidate.model_type is not None
        and candidate.architecture is not None
        and candidate.hidden_width is not None
        and candidate.loader is not None
        and candidate.tokenizer is not None
    )
    if device == "mps":
        import torch

        if not torch.backends.mps.is_available():
            raise ValueError("requested device is unavailable")
    snapshot = snapshot_path(candidate, root)
    verified_snapshot = VerifiedSnapshot.create(candidate, snapshot)
    texts, corpus = _load_corpus()
    import tokenizers
    import torch
    import transformers

    _ = (tokenizers, transformers)

    before = _rss()
    started = time.perf_counter()
    loaded = load_encoder(candidate, snapshot, verified=verified_snapshot)

    model = loaded.model.to(device)
    if device == "mps":
        torch.mps.synchronize()
    load_ms = (time.perf_counter() - started) * 1000
    after_load = _rss()
    with torch.inference_mode():
        untruncated_lengths = [
            len(loaded.tokenizer(text, padding=False, truncation=False)["input_ids"])
            for text in texts
        ]
        if any(length > MAX_LENGTH for length in untruncated_lengths):
            raise ValueError("probe input exceeds fixed maximum length")

        def encode_region() -> Any:
            check = loaded.tokenizer(
                texts, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt"
            )
            inputs = {key: value.to(device) for key, value in check.items()}
            output = model(**inputs).last_hidden_state
            pooled = (output.float() * inputs["attention_mask"].unsqueeze(-1)).sum(1) / inputs[
                "attention_mask"
            ].sum(1, keepdim=True)
            return pooled

        synchronize = torch.mps.synchronize if device == "mps" else lambda: None

        def run_once() -> tuple[Any, float]:
            return _timed_region(encode_region, synchronize)

        lengths_check = loaded.tokenizer(
            texts, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt"
        )
        lengths = tuple(int(row.sum()) for row in lengths_check["attention_mask"])
        for _ in range(WARMUPS):
            run_once()
        samples: list[Sample] = []
        digest = ""
        rss_after_probe = 0
        for sample_index in range(SAMPLES):
            pooled, latency_ms = run_once()
            if sample_index == SAMPLES - 1:
                rss_after_probe = _rss()
            digest = _digest_tensor(pooled)
            samples.append(
                Sample(
                    latency_ms=latency_ms,
                    embedding_sha256=digest,
                )
            )
    result = ProbeResult(
        schema_version="phase2b.v1",
        run_id=str(uuid.uuid4()),
        created_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        candidate=CandidateEvidence(
            id=candidate.id,
            repository=candidate.repository,
            revision=candidate.revision,
            declared_license=candidate.license,
            registry_sha256=registry_digest(),
            verified_files=tuple(
                FileEvidence(path=item.path, bytes=item.bytes, sha256=item.sha256)
                for item in candidate.files
            ),
            model_type=candidate.model_type,
            architecture=candidate.architecture,
            hidden_width=candidate.hidden_width,
            loader=candidate.loader,
            tokenizer=candidate.tokenizer,
        ),
        corpus=corpus,
        protocol=Protocol(
            warmup_iterations=3,
            measured_iterations=20,
            max_length=128,
            padding="True",
            truncation="True",
            pooling="attention-mask-weighted-mean-float32-no-normalization",
            measured_boundary="tokenization-device-transfer-forward-pooling",
        ),
        environment=_environment(),
        device=device,
        pooling_dtype="float32",
        fresh_process_load_ms=load_ms,
        parameter_count=sum(int(parameter.numel()) for parameter in model.parameters()),
        snapshot_bytes=sum(item.bytes for item in candidate.files),
        rss_before_load=before,
        rss_after_load=after_load,
        rss_after_probe=rss_after_probe,
        token_lengths=cast(tuple[int, int], lengths),
        hidden_width=int(pooled.shape[-1]),
        truncated=False,
        embedding_sha256=digest,
        samples=tuple(samples),
        p50_ms=_nearest([item.latency_ms for item in samples], 0.5),
        p95_ms=_nearest([item.latency_ms for item in samples], 0.95),
    )
    return write_probe_result(result, output_dir)


def write_probe_result(result: ProbeResult, output_dir: Path) -> tuple[Path, Path]:
    run_id = result.run_id
    candidate_id = result.candidate.id
    device = result.device
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / f"{candidate_id}-{device}-{run_id}.json"
    md_path = output_dir / f"{candidate_id}-{device}-{run_id}.md"
    _atomic_create(
        raw_path,
        json.dumps(result.model_dump(mode="json"), sort_keys=True, separators=(",", ":")).encode()
        + b"\n",
    )
    try:
        _atomic_create(md_path, render_report(result).encode())
    except BaseException:
        raw_path.unlink(missing_ok=True)
        directory_fd = os.open(output_dir, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        raise
    return raw_path, md_path


def _atomic_create(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def render_report(result: ProbeResult) -> str:
    return "\n".join(
        [
            "# Saracura encoder probe",
            "",
            f"- Candidate: `{result.candidate.id}@{result.candidate.revision}`",
            f"- Device: `{result.device}`",
            f"- Run: `{result.run_id}`",
            f"- Snapshot bytes: `{result.snapshot_bytes}`",
            f"- Fresh-process load (page-cache-sensitive): `{result.fresh_process_load_ms:.3f} ms`",
            f"- Warm encode p50/p95: `{result.p50_ms:.3f}/{result.p95_ms:.3f} ms`",
            "",
            "This is encoder-only, research-only evidence. It does not measure decision quality, "
            "trained-head performance, calibration, end-to-end Saracura latency, or readiness "
            "for automation.",
            "",
        ]
    )
