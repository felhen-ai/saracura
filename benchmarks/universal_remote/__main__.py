"""CLI entrypoint. Validation and recovery are deliberately credential-free."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from benchmarks.e2e_benchmark import validate_phase3c_artifacts, validate_predecessor_binding
from benchmarks.synthetic_research import (
    BudgetLedger,
    validate_packet_manifest,
    validate_training_manifest,
)
from benchmarks.typesafe_native import validate_phase3d_artifacts
from benchmarks.universal_remote.plan import digest, validate_plan
from benchmarks.universal_remote.runner import (
    BUDGET,
    RESERVATION,
    _decimal,
    _run_prevalidated,
    finalize,
    validate_artifact,
    validate_phase4c2_artifact,
)


def _inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--prior-cost-ledger", required=True, type=Path)
    parser.add_argument("--training-manifest", required=True, type=Path)
    parser.add_argument("--phase3c-artifact-dir", required=True, type=Path)
    parser.add_argument("--packet", required=True, type=Path)
    parser.add_argument("--phase3d-artifact-dir", required=True, type=Path)
    parser.add_argument("--laya-mps-artifact-dir", required=True, type=Path)
    parser.add_argument("--mdeberta-mps-artifact-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)


def _provenance(args: argparse.Namespace) -> dict[str, str]:
    paths = {
        "prior_ledger": args.prior_cost_ledger,
        "training": args.training_manifest,
        "phase3c": args.phase3c_artifact_dir / "artifact-manifest.json",
        "packet": args.packet,
        "phase3d": args.phase3d_artifact_dir / "artifact-manifest.json",
        "laya": args.laya_mps_artifact_dir / "artifact-manifest.json",
        "mdeberta": args.mdeberta_mps_artifact_dir / "artifact-manifest.json",
    }
    values = {key: digest(path) for key, path in paths.items()}
    values["plan_v3"] = digest(
        Path(__file__).parents[2] / "benchmarks/manifests/universal-bakeoff-plan.v3.json"
    )
    return values


def _preflight(args: argparse.Namespace) -> tuple[dict[str, str], list[dict[str, object]]]:
    """Re-run every predecessor validator before a claim can be created."""
    repo = Path.cwd().resolve()
    output = args.output_dir.resolve(strict=False)
    try:
        output.relative_to(repo)
    except ValueError:
        pass
    else:
        ignored = subprocess.run(
            ["git", "-C", str(repo), "check-ignore", "-q", "--", str(output)],
            check=False,
        )
        if ignored.returncode != 0:
            raise ValueError("output directory must be outside the checkout or gitignored")
    packet = (
        args.packet
        if args.packet.name == "packet-manifest.json"
        else args.packet / "packet-manifest.json"
    )
    validate_packet_manifest(packet)
    validate_training_manifest(args.training_manifest, packet)
    validate_predecessor_binding(
        args.prior_cost_ledger,
        packet_sha256=digest(packet),
        training_sha256=digest(args.training_manifest),
    )
    phase3c = validate_phase3c_artifacts(args.phase3c_artifact_dir)
    phase3d = validate_phase3d_artifacts(
        args.phase3d_artifact_dir, args.phase3c_artifact_dir, packet
    )
    if (
        phase3d.native.status != "complete"
        or phase3d.native.resolved_models != ["jev-1.13.0"]
        or digest(args.prior_cost_ledger) != phase3c.provenance.prior_cost_ledger_sha256
    ):
        raise ValueError("Phase 3C/3D binding")
    ledger = BudgetLedger.from_json(json.loads(args.prior_cost_ledger.read_bytes()))
    if _decimal(ledger.total) != _decimal(phase3c.budget.prior_total_usd):
        raise ValueError("prior ledger total")
    prior = (
        _decimal(phase3c.budget.prior_total_usd)
        + _decimal(phase3c.budget.phase3c_jev_usd)
        + _decimal(phase3d.native.computed_cost_usd)
    )
    if prior > BUDGET - RESERVATION * 57:
        raise ValueError("prior cumulative budget")
    laya = validate_phase4c2_artifact(args.laya_mps_artifact_dir, "laya-multilingual")
    mdeberta = validate_phase4c2_artifact(args.mdeberta_mps_artifact_dir, "mdeberta-nli")
    shared = (
        "python_version",
        "torch_version",
        "transformers_version",
        "tokenizers_version",
        "safetensors_version",
        "os_version",
        "architecture",
        "device_type",
        "execution_dtype",
        "process_threads",
        "mps_fallback_enabled",
        "microbatch_size",
    )
    if any(laya.get(key) != mdeberta.get(key) for key in shared):
        raise ValueError("local comparison axes")
    local_comparison: list[dict[str, object]] = [
        {
            "candidate_id": "laya-multilingual",
            "completed_decision_count": 182,
            "latency_p50_ms": laya["latency_p50_ms"],
            "decisions_per_second": laya["decisions_per_second"],
        },
        {
            "candidate_id": "mdeberta-nli",
            "completed_decision_count": 183,
            "latency_p50_ms": mdeberta["latency_p50_ms"],
            "decisions_per_second": mdeberta["decisions_per_second"],
        },
    ]
    return _provenance(args), local_comparison


def main() -> int:
    parser = argparse.ArgumentParser(prog="universal_remote")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("validate-plan")
    run_parser = commands.add_parser("run")
    _inputs(run_parser)
    run_parser.add_argument("--allow-network", action="store_true")
    run_parser.add_argument("--budget-usd", required=True)
    final_parser = commands.add_parser("finalize-partial")
    _inputs(final_parser)
    validate_parser = commands.add_parser("validate-artifact")
    _inputs(validate_parser)
    args = parser.parse_args()
    if args.command == "validate-plan":
        validate_plan()
        print("validated phase4c3 plan")
        return 0
    provenance, local_comparison = _preflight(args)
    if args.command == "run":
        if not args.allow_network or args.budget_usd != "0.25":
            raise ValueError("run requires --allow-network --budget-usd 0.25")
        _run_prevalidated(
            Path.cwd(),
            args.output_dir,
            provenance,
            local_comparison,
            api_key=os.environ.get("TYPESAFE_API_KEY", ""),
        )
        return 0
    if args.command == "finalize-partial":
        finalize(
            Path.cwd(),
            args.output_dir,
            provenance["prior_ledger"],
            provenance,
            local_comparison,
        )
        return 0
    validate_artifact(
        Path.cwd(),
        args.output_dir,
        provenance["prior_ledger"],
        provenance,
        local_comparison,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
