"""Closed CLI for the opt-in encoder candidate research lane."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import NoReturn

from benchmarks.encoder_acquisition import acquire, verify_snapshot
from benchmarks.encoder_probe import probe
from benchmarks.encoder_registry import get_candidate, validate_registry


class _SanitizedParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise ValueError("invalid encoder gate command")


def _parser() -> argparse.ArgumentParser:
    parser = _SanitizedParser(prog="python -m benchmarks.encoder_gate")
    commands = parser.add_subparsers(dest="command", required=True, parser_class=_SanitizedParser)
    acquire_parser = commands.add_parser("acquire")
    acquire_parser.add_argument("--candidate", required=True)
    acquire_parser.add_argument("--allow-network", action="store_true")
    verify = commands.add_parser("verify")
    verify.add_argument("--candidate", required=True)
    probe_parser = commands.add_parser("probe")
    probe_parser.add_argument("--candidate", required=True)
    probe_parser.add_argument("--device", choices=("cpu", "mps"), required=True)
    probe_parser.add_argument("--output-dir", type=Path, required=True)
    commands.add_parser("validate-registry")
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        if args.command == "validate-registry":
            registry = validate_registry()
            print(
                json.dumps(
                    {
                        "schema_version": registry.schema_version,
                        "candidates": len(registry.candidates),
                    }
                )
            )
        elif args.command == "acquire":
            acquire(args.candidate, allow_network=args.allow_network)
            print(json.dumps({"candidate": args.candidate, "status": "installed"}))
        elif args.command == "verify":
            candidate = get_candidate(args.candidate)
            print(json.dumps({"candidate": args.candidate, "verified": verify_snapshot(candidate)}))
        else:
            raw, report = probe(args.candidate, args.device, args.output_dir)
            print(json.dumps({"raw": raw.name, "report": report.name}))
        return 0
    except Exception as error:
        # Public output is intentionally bounded: no paths, URLs, environment, or input text.
        code = (
            "OPTIONAL_DEPENDENCY_MISSING"
            if isinstance(error, (ImportError, RuntimeError))
            else "ENCODER_GATE_REJECTED"
        )
        print(
            json.dumps({"error": {"code": code, "message": "encoder gate request was rejected"}}),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
