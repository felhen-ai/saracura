"""Prove the default environment is free of optional ML runtimes."""

from __future__ import annotations

import importlib.util

FORBIDDEN = (
    "torch",
    "transformers",
    "tokenizers",
    "safetensors",
    "platformdirs",
    "psutil",
    "huggingface_hub",
    "sentence_transformers",
    "datasets",
)


def main() -> int:
    present = [name for name in FORBIDDEN if importlib.util.find_spec(name) is not None]
    if present:
        raise SystemExit("optional modules present in default environment: " + ",".join(present))
    print("default environment is offline and lightweight")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
