"""Command-line entry point for the checkout-only Phase 5D benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmarks.ptbr_native.runner import atomic_write_report, run_benchmark, verify_report


def _absolute(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("path must be absolute")
    return path


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="python -m benchmarks.ptbr_native")
    commands = result.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--backend", choices=("julia", "saracura-universal"), required=True)
    run.add_argument(
        "--strategy",
        choices=("single_pass", "cyclic_mean"),
        default="single_pass",
    )
    run.add_argument("--device", choices=("cpu",), required=True)
    run.add_argument("--data-root", type=_absolute, required=True)
    run.add_argument("--output", type=_absolute, required=True)
    run.add_argument("--model-snapshot", type=_absolute)
    run.add_argument("--encoder-snapshot", type=_absolute)
    run.add_argument("--training-capsule", type=_absolute)
    verify = commands.add_parser("verify")
    verify.add_argument("report", type=_absolute)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "verify":
        verify_report(args.report)
        print("Phase 5D aggregate report is valid")
        return 0
    report = run_benchmark(
        backend_name=args.backend,
        data_root=args.data_root,
        model_snapshot=args.model_snapshot,
        encoder_snapshot=args.encoder_snapshot,
        training_capsule=args.training_capsule,
        inference_strategy=args.strategy,
    )
    atomic_write_report(args.output, report)
    print(
        f"wrote {args.output.name}: accuracy={report.planned_top1_accuracy:.4f} "
        f"coverage={report.coverage:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
