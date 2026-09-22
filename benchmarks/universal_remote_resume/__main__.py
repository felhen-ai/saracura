"""Command-line entrypoint for the additive, checkout-only recovery runner."""

from __future__ import annotations

import argparse
import os
from decimal import Decimal
from pathlib import Path

from benchmarks.e2e_benchmark import validate_phase3c_artifacts
from benchmarks.typesafe_native import validate_phase3d_artifacts
from benchmarks.universal_remote import __main__ as predecessor_cli
from benchmarks.universal_remote_resume.plan import digest, validate_plan
from benchmarks.universal_remote_resume.runner import (
    _urllib_transport,
    finalize,
    run_prevalidated,
    validate_artifact,
)


def _inputs(parser: argparse.ArgumentParser) -> None:
    predecessor_cli._inputs(parser)
    parser.add_argument("--phase4c3-partial-artifact-dir", required=True, type=Path)


def _prior_use(args: argparse.Namespace) -> Decimal:
    """Repeat the predecessor calculation; never rely on an unreturned local."""
    packet = (
        args.packet
        if args.packet.name == "packet-manifest.json"
        else args.packet / "packet-manifest.json"
    )
    phase3c = validate_phase3c_artifacts(args.phase3c_artifact_dir)
    phase3d = validate_phase3d_artifacts(
        args.phase3d_artifact_dir, args.phase3c_artifact_dir, packet
    )
    return (
        Decimal(phase3c.budget.prior_total_usd)
        + Decimal(phase3c.budget.phase3c_jev_usd)
        + Decimal(phase3d.native.computed_cost_usd)
    )


def _preflight(args: argparse.Namespace) -> tuple[dict[str, str], list[dict[str, object]], Decimal]:
    base, comparison = predecessor_cli._preflight(args)
    binding = {
        "recovery_plan": digest(
            Path(__file__).parents[2] / "benchmarks/manifests/phase4c3c-resume-plan.v1.json"
        ),
        "phase4c3_partial": digest(args.phase4c3_partial_artifact_dir / "artifact-manifest.json"),
        **base,
    }
    return binding, comparison, _prior_use(args)


def main() -> int:
    parser = argparse.ArgumentParser(prog="universal_remote_resume")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("validate-plan")
    run = commands.add_parser("run")
    finalize_partial = commands.add_parser("finalize-partial")
    validate = commands.add_parser("validate-artifact")
    for command in (run, finalize_partial, validate):
        _inputs(command)
    run.add_argument("--allow-network", action="store_true")
    run.add_argument("--budget-usd", required=True)
    args = parser.parse_args()
    if args.command == "validate-plan":
        validate_plan()
        print("validated phase4c3c recovery plan")
        return 0
    binding, comparison, prior_use = _preflight(args)
    if args.command == "run":
        if not args.allow_network or args.budget_usd != "0.25":
            raise ValueError("run requires --allow-network --budget-usd 0.25")
        run_prevalidated(
            Path.cwd(),
            args.output_dir,
            args.phase4c3_partial_artifact_dir,
            binding,
            comparison,
            prior_use,
            api_key=os.environ.get("TYPESAFE_API_KEY", ""),
            transport=_urllib_transport,
        )
        return 0
    if args.command == "finalize-partial":
        finalize(
            Path.cwd(),
            args.output_dir,
            args.phase4c3_partial_artifact_dir,
            binding,
            comparison,
            prior_use,
        )
        return 0
    validate_artifact(
        Path.cwd(),
        args.output_dir,
        args.phase4c3_partial_artifact_dir,
        binding,
        comparison,
        prior_use,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
