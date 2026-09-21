"""Offline CLI for the Phase 2C source-policy gate."""

from __future__ import annotations

import sys

from benchmarks.data_policy_registry import load_bundled_registry


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args != ["validate-registry"]:
        print("usage: python -m benchmarks.data_policy_gate validate-registry", file=sys.stderr)
        return 2
    try:
        load_bundled_registry()
    except (OSError, ValueError):
        print("data-policy registry: invalid", file=sys.stderr)
        return 1
    print("data-policy registry: valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
