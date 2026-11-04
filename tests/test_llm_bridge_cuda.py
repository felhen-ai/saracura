"""Network-free tests for the CUDA SDPA launcher."""

from __future__ import annotations

import functools
import hashlib
import importlib.util
import sys
import types
from pathlib import Path
from typing import Any, ClassVar

import pytest


def _load_research_module(name: str) -> Any:
    path = Path(__file__).parents[1] / "research" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


llm_bridge = _load_research_module("llm_bridge")
cuda = _load_research_module("llm_bridge_cuda")
REVISION = "c" * 40


class _Tokenizer:
    chat_template = "{{ messages }}"

    @staticmethod
    def apply_chat_template(*_: Any, **__: Any) -> str:
        return "prompt"


class _Model:
    def __init__(self, attention: str) -> None:
        self.config = types.SimpleNamespace(_attn_implementation=attention)
        self.devices: list[str] = []
        self.evaluated = False

    def to(self, device: str) -> None:
        self.devices.append(device)

    def eval(self) -> None:
        self.evaluated = True


class _AutoTokenizer:
    calls: ClassVar[list[tuple[str, dict[str, Any]]]] = []

    @classmethod
    def from_pretrained(cls, model: str, **kwargs: Any) -> _Tokenizer:
        cls.calls.append((model, kwargs))
        return _Tokenizer()


class _AutoModel:
    calls: ClassVar[list[tuple[str, dict[str, Any]]]] = []
    attention: ClassVar[str] = "sdpa"

    @classmethod
    def from_pretrained(cls, model: str, **kwargs: Any) -> _Model:
        cls.calls.append((model, kwargs))
        return _Model(cls.attention)


def _fake_modules() -> dict[str, Any]:
    _AutoTokenizer.calls = []
    _AutoModel.calls = []
    torch = types.SimpleNamespace(
        __version__="fake-torch",
        cuda=types.SimpleNamespace(is_available=lambda: True),
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False)),
        bfloat16=object(),
        float16=object(),
        float32=object(),
        set_num_threads=lambda _: None,
    )
    transformers = types.SimpleNamespace(
        __version__="fake-transformers",
        AutoTokenizer=_AutoTokenizer,
        AutoModelForCausalLM=_AutoModel,
        AutoModelForImageTextToText=_AutoModel,
        StoppingCriteria=object,
        StoppingCriteriaList=list,
    )
    return {"torch": torch, "transformers": transformers}


def _config(device: str = "cuda") -> Any:
    return llm_bridge.Config(
        model="org/model",
        revision=REVISION,
        device=device,
        context=8192,
        threads=2,
        template_kwargs={},
    )


def test_proxy_forces_sdpa_with_original_pinned_arguments_and_no_global_mutation() -> None:
    modules = _fake_modules()
    original_descriptor = _AutoModel.__dict__["from_pretrained"]

    backend, info = cuda.load_transformers_backend(_config(), import_module=modules.__getitem__)

    assert backend.model.config._attn_implementation == "sdpa"
    assert info["attention"] == "sdpa"
    assert _AutoModel.__dict__["from_pretrained"] is original_descriptor
    model, kwargs = _AutoModel.calls[0]
    assert model == "org/model"
    assert kwargs["attn_implementation"] == "sdpa"
    assert kwargs["revision"] == REVISION
    assert kwargs["local_files_only"] is True
    assert kwargs["trust_remote_code"] is False
    assert kwargs["dtype"] is modules["torch"].bfloat16


def test_non_cuda_is_rejected_before_any_import() -> None:
    with pytest.raises(llm_bridge.BridgeError, match="requires --device cuda"):
        cuda.load_transformers_backend(
            _config("cpu"), import_module=lambda _: pytest.fail("imported")
        )


def test_actual_attention_mismatch_fails_closed() -> None:
    modules = _fake_modules()
    _AutoModel.attention = "eager"
    try:
        with pytest.raises(llm_bridge.BridgeError, match="not sdpa"):
            cuda.load_transformers_backend(_config(), import_module=modules.__getitem__)
    finally:
        _AutoModel.attention = "sdpa"


def test_evidence_records_actual_attention_and_both_source_hashes() -> None:
    modules = _fake_modules()
    _, info = cuda.load_transformers_backend(_config(), import_module=modules.__getitem__)

    evidence = cuda.build_evidence(_config(), info, cuda.LAUNCHER_SHA256)

    assert evidence["backend"]["attention"] == "sdpa"
    assert evidence["bridge_sha256"] == cuda.LAUNCHER_SHA256
    assert evidence["source_hashes"] == {
        "launcher": cuda.LAUNCHER_SHA256,
        "legacy_bridge": hashlib.sha256(Path(llm_bridge.__file__).read_bytes()).hexdigest(),
    }


def test_main_uses_captured_loader_without_recursion_and_restores_hooks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    modules = _fake_modules()
    config = _config()
    original_loader = llm_bridge.load_transformers_backend
    original_evidence = llm_bridge.build_evidence
    real_wrapper = cuda.load_transformers_backend
    monkeypatch.setattr(
        cuda,
        "load_transformers_backend",
        functools.partial(real_wrapper, import_module=modules.__getitem__),
    )
    observed = []

    def fake_main(argv: list[str]) -> None:
        backend, info = llm_bridge.load_transformers_backend(config)
        evidence = llm_bridge.build_evidence(config, info, "legacy-call-site-hash")
        observed.append((argv, backend.model.config._attn_implementation, evidence))

    monkeypatch.setattr(llm_bridge, "main", fake_main)
    cuda.main(["--device", "cuda"])
    assert observed[0][1] == "sdpa"
    assert observed[0][2]["bridge_sha256"] == cuda.LAUNCHER_SHA256
    assert llm_bridge.load_transformers_backend is original_loader
    assert llm_bridge.build_evidence is original_evidence
