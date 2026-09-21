from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import sys
import urllib.request
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Literal, cast

import pytest

import benchmarks.encoder_acquisition as acquisition
import benchmarks.encoder_loader as encoder_loader
import benchmarks.encoder_probe as encoder_probe
from benchmarks.encoder_acquisition import (
    _ClosedRedirectHandler,
    _copy_bounded,
    _stream,
    acquire,
    verify_snapshot,
)
from benchmarks.encoder_gate import main
from benchmarks.encoder_probe import (
    CORPUS_DIGEST,
    ProbeResult,
    _timed_region,
    write_probe_result,
)
from benchmarks.encoder_registry import (
    Candidate,
    RegistryFile,
    get_candidate,
    load_registry,
    registry_digest,
)
from benchmarks.validate_manifests import validate_routed_manifest


def _small_candidate() -> tuple[Candidate, dict[str, bytes]]:
    payloads = {
        "config.json": b"{}",
        "model.safetensors": b"safe-weights",
        "tokenizer_config.json": b"{}",
    }
    files = tuple(
        RegistryFile(path=path, bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
        for path, payload in payloads.items()
    )
    return Candidate(
        id="multilingual-minilm-l12",
        repository="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        revision="e8f8c211226b894fcb81acc59f3b34ba3efd5f42",
        license="Apache-2.0",
        role="speed-control",
        disposition="eligible",
        reason="test candidate",
        reviewed_at="2026-09-21",
        files=files,
        model_type="bert",
        architecture="BertModel",
        hidden_width=384,
        loader="bert",
        tokenizer="PreTrainedTokenizerFast",
    ), payloads


def _write_snapshot(candidate: Candidate, payloads: dict[str, bytes], path: Path) -> None:
    path.mkdir(parents=True, mode=0o700)
    os.chmod(path, 0o700)
    for name, payload in payloads.items():
        target = path / name
        target.write_bytes(payload)
        os.chmod(target, 0o600)
    marker = {
        "candidate_id": candidate.id,
        "revision": candidate.revision,
        "registry_sha256": registry_digest(),
        "files": [item.model_dump() for item in candidate.files],
    }
    marker_path = path / "snapshot.complete.json"
    marker_path.write_text(
        json.dumps(marker, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    os.chmod(marker_path, 0o600)


def _valid_result_payload() -> dict[str, Any]:
    candidate = get_candidate("multilingual-minilm-l12")
    digest = "a" * 64
    return {
        "schema_version": "phase2b.v1",
        "run_id": "123e4567-e89b-42d3-a456-426614174000",
        "created_at": "2026-09-21T12:00:00Z",
        "candidate": {
            "id": candidate.id,
            "repository": candidate.repository,
            "revision": candidate.revision,
            "declared_license": candidate.license,
            "registry_sha256": registry_digest(),
            "verified_files": tuple(item.model_dump() for item in candidate.files),
            "model_type": candidate.model_type,
            "architecture": candidate.architecture,
            "hidden_width": candidate.hidden_width,
            "loader": candidate.loader,
            "tokenizer": candidate.tokenizer,
        },
        "corpus": {
            "id": "encoder-probe-corpus.v1",
            "sha256": CORPUS_DIGEST,
            "case_ids": ("ptbr-support-refund", "en-support-refund"),
            "locales": ("pt-BR", "en"),
        },
        "protocol": {
            "warmup_iterations": 3,
            "measured_iterations": 20,
            "max_length": 128,
            "padding": "True",
            "truncation": "True",
            "pooling": "attention-mask-weighted-mean-float32-no-normalization",
            "measured_boundary": "tokenization-device-transfer-forward-pooling",
        },
        "environment": {
            "os_family": "Darwin",
            "os_release": "test",
            "architecture": "arm64",
            "machine_class": "MacBookPro",
            "python_version": "3.11.0",
            "saracura_version": "0.1.0a1",
            "torch_version": "2.5.0",
            "transformers_version": "4.47.0",
            "tokenizers_version": "0.21.0",
        },
        "device": "cpu",
        "pooling_dtype": "float32",
        "fresh_process_load_ms": 1.0,
        "parameter_count": 10,
        "snapshot_bytes": candidate.total_bytes,
        "rss_before_load": 1,
        "rss_after_load": 2,
        "rss_after_probe": 3,
        "token_lengths": (10, 11),
        "hidden_width": candidate.hidden_width,
        "truncated": False,
        "embedding_sha256": digest,
        "samples": tuple({"latency_ms": 1.0, "embedding_sha256": digest} for _ in range(20)),
        "p50_ms": 1.0,
        "p95_ms": 1.0,
    }


def test_registry_has_reviewed_candidates_and_blocks_mmbert() -> None:
    registry = load_registry()
    assert registry.schema_version == "encoder-candidates.v1"
    assert [candidate.id for candidate in registry.candidates] == [
        "mmbert-base",
        "xlm-roberta-base",
        "multilingual-minilm-l12",
    ]
    assert get_candidate("mmbert-base").disposition == "blocked"
    assert get_candidate("mmbert-base").files == ()
    assert all(candidate.total_bytes > 0 for candidate in registry.candidates[1:])


def test_registry_rejects_duplicate_keys() -> None:
    raw = (
        b'{"schema_version":"encoder-candidates.v1",'
        b'"schema_version":"encoder-candidates.v1","candidates":[]}'
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_registry(raw)


def test_corpus_is_immutable_and_has_no_runtime_inputs_in_contract(tmp_path: Path) -> None:
    corpus = Path("benchmarks/fixtures/encoder-probe-corpus.v1.json").read_bytes()
    assert hashlib.sha256(corpus).hexdigest() == CORPUS_DIGEST
    assert "text" in json.loads(corpus)["cases"][0]
    assert "text" not in ProbeResult.model_fields
    assert "token_ids" not in ProbeResult.model_fields


def test_blocked_candidate_fails_before_snapshot_verification(tmp_path: Path) -> None:
    candidate = get_candidate("mmbert-base")
    with pytest.raises(ValueError, match="blocked"):
        verify_snapshot(candidate, tmp_path)


def test_registry_digest_is_exact_bundled_bytes() -> None:
    raw = Path("benchmarks/manifests/encoder-candidates.v1.json").read_bytes()
    assert registry_digest() == hashlib.sha256(raw).hexdigest()


def test_bounded_stream_rejects_overflow() -> None:
    with pytest.raises(ValueError, match="exceeds"):
        _copy_bounded(io.BytesIO(b"12345"), io.BytesIO(), 4)


def test_bounded_stream_rejects_short_read() -> None:
    with pytest.raises(ValueError, match="does not match"):
        _copy_bounded(io.BytesIO(b"123"), io.BytesIO(), 4)


def test_redirect_handler_rejects_http_and_untrusted_hosts() -> None:
    handler = _ClosedRedirectHandler()
    with pytest.raises(ValueError, match="allowlist"):
        handler.redirect_request(  # type: ignore[no-untyped-call]
            None, None, 302, "", {}, "http://huggingface.co/x"
        )
    with pytest.raises(ValueError, match="allowlist"):
        handler.redirect_request(  # type: ignore[no-untyped-call]
            None, None, 302, "", {}, "https://evil.example/x"
        )


def test_stream_is_anonymous_proxy_free_and_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"safe"
    captured: dict[str, Any] = {}

    class Response(io.BytesIO):
        def __enter__(self) -> Response:
            return self

        def __exit__(self, *args: object) -> None:
            self.close()

        def geturl(self) -> str:
            return "https://cdn.hf.co/model"

    class Opener:
        def open(self, request: urllib.request.Request, timeout: int) -> Response:
            captured["headers"] = dict(request.header_items())
            captured["timeout"] = timeout
            return Response(payload)

    def build_opener(*handlers: object) -> Opener:
        captured["handlers"] = handlers
        return Opener()

    monkeypatch.setattr(urllib.request, "build_opener", build_opener)
    destination = tmp_path / "config.json"
    _stream("https://huggingface.co/owner/repo/file", destination, len(payload))
    handlers = captured["handlers"]
    assert isinstance(handlers[0], urllib.request.ProxyHandler)
    proxy_handler: Any = handlers[0]
    assert proxy_handler.proxies == {}
    assert isinstance(handlers[1], _ClosedRedirectHandler)
    assert "Authorization" not in captured["headers"]
    assert destination.read_bytes() == payload


def test_acquire_requires_explicit_network_before_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate, _ = _small_candidate()
    monkeypatch.setattr(acquisition, "get_candidate", lambda _: candidate)
    called = False

    def stream(*args: object) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(acquisition, "_stream", stream)
    with pytest.raises(ValueError, match="allow-network"):
        acquire(candidate.id, allow_network=False, root=tmp_path)
    assert called is False


def test_acquire_is_hash_verified_atomic_and_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate, payloads = _small_candidate()
    monkeypatch.setattr(acquisition, "get_candidate", lambda _: candidate)
    monkeypatch.setattr(shutil, "disk_usage", lambda _: SimpleNamespace(free=10_000_000_000))
    calls: list[str] = []

    def stream(url: str, destination: Path, expected_size: int) -> None:
        payload = payloads[destination.name]
        assert len(payload) == expected_size
        descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
        calls.append(url)

    monkeypatch.setattr(acquisition, "_stream", stream)
    installed = acquire(candidate.id, allow_network=True, root=tmp_path)
    assert verify_snapshot(candidate, installed)
    assert len(calls) == len(candidate.files)
    acquire(candidate.id, allow_network=True, root=tmp_path)
    assert len(calls) == len(candidate.files)


def test_acquire_free_space_fails_before_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate, _ = _small_candidate()
    monkeypatch.setattr(acquisition, "get_candidate", lambda _: candidate)
    monkeypatch.setattr(shutil, "disk_usage", lambda _: SimpleNamespace(free=0))
    monkeypatch.setattr(
        acquisition,
        "_stream",
        lambda *args: pytest.fail("network must not run after free-space rejection"),
    )
    with pytest.raises(ValueError, match="free space"):
        acquire(candidate.id, allow_network=True, root=tmp_path)


def test_acquire_does_not_clobber_destination_created_at_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate, payloads = _small_candidate()
    monkeypatch.setattr(acquisition, "get_candidate", lambda _: candidate)
    monkeypatch.setattr(shutil, "disk_usage", lambda _: SimpleNamespace(free=10_000_000_000))

    def stream(url: str, destination: Path, expected_size: int) -> None:
        payload = payloads[destination.name]
        destination.write_bytes(payload)
        os.chmod(destination, 0o600)

    monkeypatch.setattr(acquisition, "_stream", stream)

    def inject_destination(path: Path) -> None:
        path.mkdir(mode=0o700)
        (path / "foreign").write_text("keep", encoding="utf-8")
        raise FileExistsError("injected race")

    monkeypatch.setattr(acquisition, "_create_final", inject_destination)
    with pytest.raises(FileExistsError, match="injected race"):
        acquire(candidate.id, allow_network=True, root=tmp_path)
    final = acquisition.snapshot_path(candidate, tmp_path)
    assert (final / "foreign").read_text(encoding="utf-8") == "keep"


def test_acquire_rejects_concurrent_claim_before_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate, _ = _small_candidate()
    monkeypatch.setattr(acquisition, "get_candidate", lambda _: candidate)
    parent = acquisition.snapshot_path(candidate, tmp_path).parent
    parent.mkdir(parents=True, mode=0o700)
    (parent / f".{candidate.revision}.claim").mkdir(mode=0o700)
    monkeypatch.setattr(
        acquisition,
        "_stream",
        lambda *args: pytest.fail("network must not run with a concurrent claim"),
    )
    with pytest.raises(ValueError, match="already in progress"):
        acquire(candidate.id, allow_network=True, root=tmp_path)


def test_valid_snapshot_wins_over_stale_claim_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate, payloads = _small_candidate()
    monkeypatch.setattr(acquisition, "get_candidate", lambda _: candidate)
    final = acquisition.snapshot_path(candidate, tmp_path)
    _write_snapshot(candidate, payloads, final)
    (final.parent / f".{candidate.revision}.claim").mkdir(mode=0o700)
    monkeypatch.setattr(
        acquisition,
        "_stream",
        lambda *args: pytest.fail("valid snapshot must not make a network request"),
    )
    assert acquire(candidate.id, allow_network=True, root=tmp_path) == final


def test_snapshot_rejects_final_symlink_and_weight_hardlink(tmp_path: Path) -> None:
    candidate, payloads = _small_candidate()
    real = tmp_path / "real"
    _write_snapshot(candidate, payloads, real)
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)
    assert verify_snapshot(candidate, linked) is False
    os.link(real / "model.safetensors", tmp_path / "weight-copy")
    assert verify_snapshot(candidate, real) is False


def test_cli_errors_and_success_do_not_echo_private_values(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["acquire", "--candidate", "mmbert-base", "--allow-network"]) == 2
    captured = capsys.readouterr()
    assert "mmbert-base" not in captured.err

    private = "private-input-value"
    assert main(["probe", "--unknown", private]) == 2
    captured = capsys.readouterr()
    assert private not in captured.err
    assert "allow-network" not in captured.err
    assert main(["acquire", "--candidate", "mmbert-base"]) == 2
    captured = capsys.readouterr()
    assert "mmbert-base" not in captured.err


def test_snapshot_rejects_marker_hardlink_and_unsafe_mode(tmp_path: Path) -> None:
    candidate = get_candidate("xlm-roberta-base")
    snapshot = tmp_path / candidate.id / candidate.revision
    snapshot.mkdir(parents=True, mode=0o700)
    marker = snapshot / "snapshot.complete.json"
    marker.write_text("{}", encoding="utf-8")
    os.chmod(marker, 0o600)
    os.link(marker, snapshot / "marker-copy")
    assert verify_snapshot(candidate, snapshot) is False


def test_probe_result_reconciles_hidden_width_and_percentiles() -> None:
    payload = _valid_result_payload()
    assert ProbeResult.model_validate(payload).hidden_width == 384
    payload["hidden_width"] = 999
    with pytest.raises(ValueError, match="embedding width"):
        ProbeResult.model_validate(payload)

    payload = _valid_result_payload()
    payload["p95_ms"] = 2.0
    with pytest.raises(ValueError, match="percentiles"):
        ProbeResult.model_validate(payload)


def test_timed_region_synchronizes_around_operation_only() -> None:
    events: list[str] = []
    times = iter((10.0, 10.25))

    def operation() -> str:
        events.append("operation")
        return "embedding"

    def synchronize() -> None:
        events.append("sync")

    value, elapsed_ms = _timed_region(operation, synchronize, lambda: next(times))
    assert value == "embedding"
    assert elapsed_ms == 250.0
    assert events == ["sync", "operation", "sync"]


def test_atomic_pair_removes_raw_if_report_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = ProbeResult.model_validate(_valid_result_payload())
    original = encoder_probe._atomic_create
    calls = 0

    def fail_second(path: Path, payload: bytes) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected report failure")
        original(path, payload)

    monkeypatch.setattr(encoder_probe, "_atomic_create", fail_second)
    with pytest.raises(OSError, match="injected"):
        write_probe_result(result, tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_manifest_router_rejects_unknown_schema(tmp_path: Path) -> None:
    path = tmp_path / "future.json"
    path.write_text('{"schema_version":"unknown"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported manifest schema"):
        validate_routed_manifest(path)


def test_loader_uses_closed_local_safetensors_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = get_candidate("multilingual-minilm-l12")
    calls: dict[str, Any] = {}

    class FakeModule:
        def eval(self) -> FakeModule:
            return self

    class BertModel(FakeModule):
        @classmethod
        def from_pretrained(cls, path: Path, **kwargs: Any) -> BertModel:
            calls["model"] = (path, kwargs)
            return cls()

    class BertConfig:
        model_type = "bert"
        hidden_size = 384

        @classmethod
        def from_pretrained(cls, path: Path, **kwargs: Any) -> BertConfig:
            calls["config"] = (path, kwargs)
            return cls()

    class PreTrainedTokenizerFast:
        is_fast = True

        @classmethod
        def from_pretrained(cls, path: Path, **kwargs: Any) -> PreTrainedTokenizerFast:
            calls["tokenizer"] = (path, kwargs)
            return cls()

    fake_torch: Any = ModuleType("torch")
    fake_torch.nn = SimpleNamespace(Module=FakeModule)
    fake_transformers: Any = ModuleType("transformers")
    fake_transformers.BertConfig = BertConfig
    fake_transformers.BertModel = BertModel
    fake_transformers.PreTrainedTokenizerFast = PreTrainedTokenizerFast
    fake_transformers.XLMRobertaConfig = object
    fake_transformers.XLMRobertaModel = object
    fake_transformers.XLMRobertaTokenizerFast = object
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr(encoder_loader, "verify_snapshot", lambda *_: True)
    verified = encoder_loader.VerifiedSnapshot.create(candidate, tmp_path)
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "tokenizer_config.json").write_text("{}")

    loaded = encoder_loader.load_encoder(candidate, tmp_path, verified=verified)
    assert isinstance(loaded.model, BertModel)
    assert calls["config"][1] == {"local_files_only": True}
    assert calls["tokenizer"][1] == {
        "local_files_only": True,
        "trust_remote_code": False,
        "use_fast": True,
    }
    assert calls["model"][1]["local_files_only"] is True
    assert calls["model"][1]["trust_remote_code"] is False
    assert calls["model"][1]["use_safetensors"] is True


def test_loader_rejects_metadata_auto_map_and_token_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = get_candidate("multilingual-minilm-l12")
    monkeypatch.setattr(encoder_loader, "verify_snapshot", lambda *_: True)
    verified = encoder_loader.VerifiedSnapshot.create(candidate, tmp_path)
    monkeypatch.setitem(sys.modules, "torch", ModuleType("torch"))
    fake_transformers: Any = ModuleType("transformers")
    fake_transformers.BertConfig = object
    fake_transformers.BertModel = object
    fake_transformers.PreTrainedTokenizerFast = object
    fake_transformers.XLMRobertaConfig = object
    fake_transformers.XLMRobertaModel = object
    fake_transformers.XLMRobertaTokenizerFast = object
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    (tmp_path / "config.json").write_text('{"auto_map": {"Model": "remote"}}')
    (tmp_path / "tokenizer_config.json").write_text("{}")
    with pytest.raises(ValueError, match="remote code"):
        encoder_loader.load_encoder(candidate, tmp_path, verified=verified)


def test_loader_requires_verification_token_to_match_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = get_candidate("multilingual-minilm-l12")
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.setattr(encoder_loader, "verify_snapshot", lambda *_: True)
    token = encoder_loader.VerifiedSnapshot.create(candidate, other)
    with pytest.raises(ValueError, match="verification token"):
        encoder_loader.load_encoder(candidate, tmp_path, verified=token)


def test_loader_reports_missing_optional_dependencies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = get_candidate("multilingual-minilm-l12")
    monkeypatch.setattr(encoder_loader, "verify_snapshot", lambda *_: True)
    token = encoder_loader.VerifiedSnapshot.create(candidate, tmp_path)
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "transformers", None)
    with pytest.raises(RuntimeError, match="encoder-eval extra"):
        encoder_loader.load_encoder(candidate, tmp_path, verified=token)


def test_loader_rejects_direct_or_artisanal_verification_tokens(tmp_path: Path) -> None:
    candidate = get_candidate("multilingual-minilm-l12")
    with pytest.raises(TypeError):
        encoder_loader.VerifiedSnapshot(candidate.id, candidate.revision, tmp_path)  # type: ignore[call-arg]

    token = object.__new__(encoder_loader.VerifiedSnapshot)
    object.__setattr__(token, "candidate_id", candidate.id)
    object.__setattr__(token, "revision", candidate.revision)
    object.__setattr__(token, "path", tmp_path.resolve())
    with pytest.raises(ValueError, match="verification token"):
        encoder_loader.load_encoder(candidate, tmp_path, verified=token)


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_probe_fake_end_to_end_covers_fixed_protocol_and_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, device: str
) -> None:
    events: list[str] = []

    class Row:
        def __init__(self, length: int) -> None:
            self.length = length

        def sum(self) -> int:
            return self.length

    class Tensor:
        shape = (2, 384)

        def __iter__(self) -> Any:
            return iter((Row(5), Row(6)))

        def to(self, _device: str) -> Tensor:
            return self

        def float(self) -> Tensor:
            return self

        def unsqueeze(self, _axis: int) -> Tensor:
            return self

        def sum(self, *_args: object, **_kwargs: object) -> Tensor:
            return self

        def __mul__(self, _other: object) -> Tensor:
            return self

        def __truediv__(self, _other: object) -> Tensor:
            return self

        def detach(self) -> Tensor:
            return self

        def cpu(self) -> Tensor:
            return self

        def contiguous(self) -> Tensor:
            return self

    class FakeTokenizer:
        is_fast = True

        def __call__(self, payload: Any, **_kwargs: Any) -> dict[str, Any]:
            events.append("tokenize")
            if isinstance(payload, str):
                return {"input_ids": [1, 2, 3, 4, 5]}
            return {"input_ids": Tensor(), "attention_mask": Tensor()}

    class FakeModel:
        def to(self, _device: str) -> FakeModel:
            events.append("transfer")
            return self

        def __call__(self, **_kwargs: Any) -> Any:
            events.append("forward")
            return SimpleNamespace(last_hidden_state=Tensor())

        def parameters(self) -> list[Any]:
            return [SimpleNamespace(numel=lambda: 123)]

    class Context:
        def __enter__(self) -> Context:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    class FakeMPS:
        def is_available(self) -> bool:
            return True

        def synchronize(self) -> None:
            events.append("sync")

    fake_torch: Any = ModuleType("torch")
    fake_torch.backends = SimpleNamespace(mps=FakeMPS())
    fake_torch.mps = fake_torch.backends.mps
    fake_torch.inference_mode = lambda: Context()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "tokenizers", ModuleType("tokenizers"))
    monkeypatch.setitem(sys.modules, "transformers", ModuleType("transformers"))
    candidate = get_candidate("multilingual-minilm-l12")
    monkeypatch.setattr(encoder_probe, "snapshot_path", lambda *_args, **_kwargs: tmp_path)
    monkeypatch.setattr(
        encoder_probe,
        "VerifiedSnapshot",
        SimpleNamespace(create=lambda *_args, **_kwargs: object()),
    )
    monkeypatch.setattr(
        encoder_probe,
        "load_encoder",
        lambda *_args, **_kwargs: SimpleNamespace(model=FakeModel(), tokenizer=FakeTokenizer()),
    )
    rss = iter(range(100, 123))
    monkeypatch.setattr(encoder_probe, "_rss", lambda: next(rss))
    monkeypatch.setattr(encoder_probe, "_digest_tensor", lambda _tensor: "b" * 64)
    monkeypatch.setattr(
        encoder_probe,
        "_environment",
        lambda: encoder_probe.Environment(
            os_family="Darwin",
            os_release="test",
            architecture="arm64",
            machine_class="MacBookPro",
            python_version="3.11",
            saracura_version="0.1.0a1",
            torch_version="fake",
            transformers_version="fake",
            tokenizers_version="fake",
        ),
    )

    raw, report = encoder_probe.probe(
        candidate.id, cast(Literal["cpu", "mps"], device), tmp_path / "out"
    )
    payload = json.loads(raw.read_bytes())
    assert report.exists()
    assert payload["device"] == device
    assert payload["protocol"]["warmup_iterations"] == 3
    assert payload["protocol"]["measured_iterations"] == 20
    assert len(payload["samples"]) == 20
    assert payload["truncated"] is False
    assert payload["token_lengths"] == [5, 6]
    assert payload["rss_after_probe"] > payload["rss_after_load"]
    assert events.count("forward") == 23
    if device == "mps":
        assert events.count("sync") >= 46
