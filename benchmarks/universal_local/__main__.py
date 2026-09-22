"""CLI for the checkout-only Phase 4C.2 local acquisition and execution lane."""

from __future__ import annotations

import argparse
import json
import sys

from benchmarks.universal_local.acquisition import (
    SnapshotConflictError,
    acquire,
    verification_receipt,
)
from benchmarks.universal_local.plan import validate_plan
from benchmarks.universal_local.registry import get_candidate, validate_registry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m benchmarks.universal_local")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("validate-registry")
    acquire_parser = commands.add_parser("acquire")
    acquire_parser.add_argument(
        "--candidate", required=True, choices=["laya-multilingual", "mdeberta-nli"]
    )
    acquire_parser.add_argument("--allow-network", action="store_true")
    verify_parser = commands.add_parser("verify")
    verify_parser.add_argument(
        "--candidate", required=True, choices=["laya-multilingual", "mdeberta-nli"]
    )
    freeze_parser = commands.add_parser("freeze-conformance")
    freeze_parser.add_argument(
        "--candidate", required=True, choices=["laya-multilingual", "mdeberta-nli"]
    )
    freeze_parser.add_argument("--output", required=True)
    run_parser = commands.add_parser("run")
    run_parser.add_argument(
        "--candidate", required=True, choices=["laya-multilingual", "mdeberta-nli"]
    )
    run_parser.add_argument("--device", required=True, choices=["cpu", "mps"])
    run_parser.add_argument("--warmups", required=True, type=int, choices=range(1, 4))
    run_parser.add_argument("--iterations", required=True, type=int, choices=range(1, 11))
    run_parser.add_argument("--output", required=True)
    commands.add_parser("validate-plan")
    args = parser.parse_args(argv)
    try:
        if args.command == "validate-registry":
            candidates = validate_registry()
            print(
                json.dumps(
                    {"schema": "decision-backend-candidates.v2", "candidates": len(candidates)}
                )
            )
            return 0
        if args.command == "validate-plan":
            plan = validate_plan()
            print(
                json.dumps({"schema": plan["schema_version"], "scenarios": len(plan["scenarios"])})
            )
            return 0
        candidate = get_candidate(args.candidate)
        if args.command == "acquire":
            path = acquire(args.candidate, allow_network=args.allow_network)
            print(json.dumps({"candidate": candidate.id, "status": "acquired", "path": str(path)}))
            return 0
        if args.command == "verify":
            with verification_receipt(candidate) as receipt:
                print(json.dumps(receipt.summary(), sort_keys=True))
            return 0
        if args.command == "freeze-conformance":
            from benchmarks.universal_local.conformance import freeze_fragment

            with verification_receipt(candidate) as receipt:
                path = freeze_fragment(receipt, candidate, args.output)
            print(
                json.dumps(
                    {"candidate": candidate.id, "status": "fragment_frozen", "name": path.name}
                )
            )
            return 0
        if args.command == "run":
            from benchmarks.universal_local.runner import run

            print(
                json.dumps(run(candidate, args.device, args.warmups, args.iterations, args.output))
            )
            return 0
    except SnapshotConflictError:
        print(
            "universal-local command failed: an invalid immutable snapshot exists; "
            "inspect and move it aside before retrying",
            file=sys.stderr,
        )
        return 2
    except (KeyError, ValueError, OSError):
        print(
            "universal-local command failed: invalid registry, acquisition state, or snapshot",
            file=sys.stderr,
        )
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
