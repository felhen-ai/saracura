"""Phase 5B managed CUDA CLI entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from benchmarks.phase5_managed_cuda.cli import execute, finalize
from benchmarks.phase5_managed_cuda.runner import CandidateRejected, EvidenceBlocked


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="phase5-managed-cuda", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    exec_parser = subparsers.add_parser("execute")
    exec_parser.add_argument("--gateway-url", required=True)
    exec_parser.add_argument("--direct-upstream-url", required=True)
    exec_parser.add_argument("--expected-checkpoint", required=True)
    exec_parser.add_argument("--pid", type=int, required=True)
    exec_parser.add_argument("--cold-load-evidence", required=True)
    exec_parser.add_argument("--acquisition-descriptor", type=Path, required=True)
    exec_parser.add_argument("--managed-runtime", type=Path, required=True)
    exec_parser.add_argument("--commit", required=True)
    exec_parser.add_argument("--archive-digest", required=True)
    exec_parser.add_argument("--private-output", type=Path, required=True)

    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("--private-provisional", type=Path, required=True)
    finalize_parser.add_argument("--operator-receipt", type=Path, required=True)
    finalize_parser.add_argument("--public-output", type=Path, required=True)

    args = parser.parse_args(argv)
    try:
        if args.command == "execute":
            execute(
                gateway_url=args.gateway_url,
                direct_upstream_url=args.direct_upstream_url,
                expected_checkpoint=args.expected_checkpoint,
                pid=args.pid,
                cold_load_evidence=args.cold_load_evidence,
                acquisition_descriptor_path=args.acquisition_descriptor,
                managed_runtime_path=args.managed_runtime,
                commit=args.commit,
                archive_digest=args.archive_digest,
                private_output=args.private_output,
            )
            return 0
        if args.command == "finalize":
            return (
                0
                if finalize(
                    private_input=args.private_provisional,
                    operator_receipt=args.operator_receipt,
                    public_output=args.public_output,
                )
                else 1
            )
    except (ValueError, CandidateRejected, EvidenceBlocked) as error:
        print(f"phase5-managed-cuda: {error}", file=sys.stderr)
        return 2
    return _fail("unknown command")


def _fail(message: str) -> int:
    print(f"phase5-managed-cuda: {message}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
