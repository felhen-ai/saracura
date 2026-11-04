"""CUDA-only SDPA launcher for the immutable V1 Transformers bridge.

This launcher deliberately delegates parsing, loading, evidence serialization,
and serving to :mod:`llm_bridge`.  Its only runtime difference is a local
``transformers`` proxy that replaces the legacy eager attention request with
SDPA for the selected AutoModel class.
"""

from __future__ import annotations

import hashlib
import importlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import llm_bridge as legacy

LAUNCHER_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
LEGACY_BRIDGE_SHA256 = hashlib.sha256(Path(legacy.__file__).read_bytes()).hexdigest()
_ORIGINAL_BUILD_EVIDENCE = legacy.build_evidence
_ORIGINAL_LOAD_TRANSFORMERS_BACKEND = legacy.load_transformers_backend


class _AutoClassSDPAProxy:
    """Delegate an AutoClass without modifying its process-global definition."""

    def __init__(self, original: Any) -> None:
        self._original = original

    def from_pretrained(self, *args: Any, **kwargs: Any) -> Any:
        requested = dict(kwargs)
        requested["attn_implementation"] = "sdpa"
        return self._original.from_pretrained(*args, **requested)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)


class _TransformersSDPAProxy:
    """Delegate every Transformers attribute except the requested AutoClass."""

    def __init__(self, original: Any, auto_class: str) -> None:
        self._original = original
        self._auto_class = auto_class
        self._selected = _AutoClassSDPAProxy(getattr(original, auto_class))

    def __getattr__(self, name: str) -> Any:
        if name == self._auto_class:
            return self._selected
        return getattr(self._original, name)


def _actual_attention(model: Any) -> str | None:
    """Read the actual loaded attention implementation, including text configs."""
    config = getattr(model, "config", None)
    for candidate in (config, getattr(config, "text_config", None)):
        attention = getattr(candidate, "_attn_implementation", None)
        if isinstance(attention, str):
            return attention
    return None


def load_transformers_backend(
    config: legacy.Config,
    import_module: Callable[[str], Any] = importlib.import_module,
) -> tuple[legacy.TransformersBackend, dict[str, Any]]:
    """Run the legacy loader through a CUDA SDPA-only Transformers proxy."""
    if config.device != "cuda":
        raise legacy.BridgeError("llm_bridge_cuda requires --device cuda")

    def import_sdpa(name: str) -> Any:
        module = import_module(name)
        if name == "transformers":
            return _TransformersSDPAProxy(module, config.auto_class)
        return module

    backend, info = _ORIGINAL_LOAD_TRANSFORMERS_BACKEND(config, import_module=import_sdpa)
    actual_attention = _actual_attention(backend.model)
    if actual_attention != "sdpa":
        raise legacy.BridgeError("loaded model attention implementation is not sdpa")
    info["attention"] = actual_attention
    return backend, info


def build_evidence(config: legacy.Config, info: dict[str, Any], code_sha256: str) -> dict[str, Any]:
    """Add launcher and original-bridge provenance to the legacy evidence shape."""
    evidence = _ORIGINAL_BUILD_EVIDENCE(config, info, code_sha256)
    evidence["source_hashes"] = {
        "launcher": code_sha256,
        "legacy_bridge": LEGACY_BRIDGE_SHA256,
    }
    return evidence


def main(argv: list[str] | None = None) -> None:
    """Invoke the original CLI while substituting only its injectable loader hooks."""
    original_loader = legacy.load_transformers_backend
    original_evidence = legacy.build_evidence

    def load_cuda(config: legacy.Config) -> tuple[legacy.TransformersBackend, dict[str, Any]]:
        return load_transformers_backend(config)

    def build_cuda_evidence(config: legacy.Config, info: dict[str, Any], _: str) -> dict[str, Any]:
        return build_evidence(config, info, LAUNCHER_SHA256)

    legacy.load_transformers_backend = load_cuda
    legacy.build_evidence = build_cuda_evidence
    try:
        legacy.main(argv)
    finally:
        legacy.load_transformers_backend = original_loader
        legacy.build_evidence = original_evidence


if __name__ == "__main__":
    main()
