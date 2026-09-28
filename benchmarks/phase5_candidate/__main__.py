"""Phase 5B candidate orchestration CLI.

Usage::

    python -m benchmarks.phase5_candidate describe --output TMP.json
    python -m benchmarks.phase5_candidate prepare --descriptor MANIFEST.json
    python -m benchmarks.phase5_candidate serve --descriptor MANIFEST.json
    python -m benchmarks.phase5_candidate evaluate --descriptor MANIFEST.json

Each command lazily imports the optional orchestration dependencies only when
invoked.  None of these commands are available inside ``pytest``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _fail(message: str) -> int:
    print(f"phase5-candidate: {message}", file=sys.stderr)
    return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="phase5_candidate", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    describe_parser = subparsers.add_parser("describe")
    describe_parser.add_argument("--output", type=Path, required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--descriptor", type=Path, required=True)

    serve_parser = subparsers.add_parser("serve")
    serve_parser.add_argument("--descriptor", type=Path, required=True)

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--descriptor", type=Path, required=True)

    args = parser.parse_args(argv)
    try:
        if args.command == "describe":
            from benchmarks.phase5_candidate.operations import describe

            return describe(args.output)
        if args.command == "prepare":
            from benchmarks.phase5_candidate.operations import prepare

            return prepare(args.descriptor)
        if args.command == "serve":
            from benchmarks.phase5_candidate.operations import serve

            return serve(args.descriptor)
        if args.command == "evaluate":
            from benchmarks.phase5_candidate.operations import evaluate

            return evaluate(args.descriptor)
    except ValueError as error:
        return _fail(f"invalid argument or contract: {error}")
    except OSError as error:
        return _fail(f"filesystem operation: {error}")
    except RuntimeError as error:
        return _fail(f"blocked or failed closed: {error}")
    except ModuleNotFoundError as error:
        return _fail(f"optional orchestration dependency missing: {error.name}")
    return _fail("unknown command")


if __name__ == "__main__":
    raise SystemExit(main())
