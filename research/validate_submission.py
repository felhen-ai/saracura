"""Check a ptbr_bench.py report before submitting it.

    python validate_submission.py my-model.json

Standard library only; read-only (nothing is written, nothing from the report is executed or imported). Downloads the
seven test splits fresh at the commit the report names, then checks gold labels, exact item counts and the legality of
every prediction against those bytes, recomputes every score and the confidence interval instead of trusting the
summary, and rejects smoke, partial, failed or malformed runs. Unknown metadata is listed as a warning.

A passing report is internally consistent and matches the public data. It is a declaration, not proof that the
declared checkpoint produced the predictions: maintainers still reproduce a result before listing it.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ptbr_bench as bench

MAX_REPORT_BYTES = 100 * 1024 * 1024
MAX_MESSAGES = 20


def _reject_constant(name):
    raise ValueError(f"non-finite number {name}")


def load_report(path):
    path = Path(path)
    if not path.is_file():
        raise SystemExit(f"{path} is not a file")
    if path.stat().st_size > MAX_REPORT_BYTES:
        raise SystemExit(f"{path} is larger than {MAX_REPORT_BYTES} bytes")
    try:
        return json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    except (ValueError, UnicodeDecodeError) as e:
        raise SystemExit(f"{path} is not valid JSON: {e}") from None


def _same(a, b):
    return type(a) is type(b) and a == b


def _legal(pred, row):
    if pred is None:
        return True
    if row["type"] == "noul":
        return isinstance(pred, bool)
    return isinstance(pred, str) and pred in row["criteria"]


def _check_items(task, entries, codes, rows, errors):
    """Per-item checks against the source rows. Returns the verified [gold, pred] pairs."""
    if not isinstance(entries, list) or not isinstance(codes, list):
        errors.append(f"{task}: items or errors is not a list")
        return None
    if len(entries) != len(rows) or len(codes) != len(rows):
        errors.append(f"{task}: {len(entries)} items for {len(rows)} source rows")
        return None
    bad = 0
    pairs = []
    for i, (entry, code, row) in enumerate(zip(entries, codes, rows, strict=True)):
        ok = isinstance(entry, list) and len(entry) == 2 and _same(entry[0], row["gold"])
        if ok:
            pred = entry[1]
            ok = _legal(pred, row) and (code is None or code in bench.ERROR_CODES)
            ok = ok and ((pred is None) == (code is not None))
        if not ok:
            bad += 1
            if bad <= 3:
                errors.append(
                    f"{task}: item {i} does not match the source (gold, prediction or error code)"
                )
            continue
        pairs.append([entry[0], entry[1]])
    if bad > 3:
        errors.append(f"{task}: {bad} items do not match the source in total")
    return None if bad else pairs


def validate(report, source=None):
    """(errors, warnings). `source(revision, task)` returns the split bytes; defaults to a fresh Hub download."""
    source = source or bench.download
    errors, warnings = [], []
    if not isinstance(report, dict):
        return ["the report is not a JSON object"], warnings
    if report.get("schema_version") != bench.SCHEMA_VERSION:
        return [
            "unsupported or legacy report: this validator checks schema_version "
            f"{bench.SCHEMA_VERSION}; rerun with the current ptbr_bench.py"
        ], warnings
    if report.get("dataset") != bench.REPO:
        errors.append("dataset is not " + bench.REPO)
    if report.get("protocol") != bench.protocol():
        errors.append("protocol differs from the published one (seed, max chars, tasks or scoring)")
    revision = report.get("revision")
    if not isinstance(revision, str) or not bench.SHA1_RE.fullmatch(revision):
        return [*errors, "dataset revision is not an immutable 40-hex commit"], warnings

    tasks, items, codes = report.get("tasks"), report.get("items"), report.get("errors")
    files = report.get("dataset_files")
    for label, value in (
        ("tasks", tasks),
        ("items", items),
        ("errors", codes),
        ("dataset_files", files),
    ):
        if not isinstance(value, dict) or sorted(value) != bench.TASKS:
            errors.append(f"{label} must hold exactly the seven benchmark tasks")
    if errors:
        return errors, warnings

    values, verified = [], {}
    for task in bench.TASKS:
        data = source(revision, task)
        try:
            rows = bench.parse_rows(data, task)
        except ValueError as e:
            errors.append(f"source data: {e}")
            continue
        meta, file = tasks[task], files[task]
        digest = bench.hashlib.sha256(data).hexdigest()
        if (
            not isinstance(file, dict)
            or file.get("sha256") != digest
            or file.get("bytes") != len(data)
        ):
            errors.append(f"{task}: dataset_files does not match the source at {revision}")
        pairs = _check_items(task, items[task], codes[task], rows, errors)
        if pairs is None:
            continue
        verified[task] = pairs
        score = round(bench.balanced(pairs), 4)
        failed = sum(p is None for _, p in pairs)
        expected = {
            "balanced_accuracy": score,
            "n": len(pairs),
            "failed": failed,
            "source_n": len(rows),
        }
        if not isinstance(meta, dict) or any(
            not _same(meta.get(k), v) for k, v in expected.items()
        ):
            errors.append(f"{task}: summary does not match the recomputed {expected}")
        values.append(score)

    if len(verified) == len(bench.TASKS):
        mean = round(sum(values) / len(values), 4)
        if not _same(report.get("mean_balanced_accuracy"), mean):
            errors.append(f"mean_balanced_accuracy does not match the recomputed {mean}")
        if report.get("mean_ci95") != list(bench.interval(verified)):
            errors.append("mean_ci95 does not match the recomputed interval")

    for issue in bench.readiness(report):
        errors.append(f"not submission-ready: {issue}")
    run = report.get("run")
    if not isinstance(run, dict) or run.get("submission_ready") is not True:
        errors.append("the report does not declare itself submission-ready")
    evaluator = report.get("evaluator")
    sha = evaluator.get("sha256") if isinstance(evaluator, dict) else None
    if not isinstance(sha, str) or not bench.SHA256_RE.fullmatch(sha):
        errors.append("evaluator sha256 is missing or malformed")
    elif sha != bench.self_sha256():
        warnings.append(
            "evaluator script differs from this checkout: confirm it is a published revision"
        )
    timing = report.get("timing")
    if not isinstance(timing, dict) or not isinstance(timing.get("seconds"), int | float):
        errors.append("timing is missing")
    ckpt = report.get("checkpoint")
    if not isinstance(ckpt, dict) or not all(
        isinstance(ckpt.get(f), str) for f in bench.CHECKPOINT_FIELDS
    ):
        errors.append("checkpoint metadata is missing or malformed")
    else:
        for field in bench.CHECKPOINT_FIELDS:
            if ckpt[field] == bench.UNKNOWN:
                warnings.append(f"checkpoint.{field} is unknown")
    for label in ("name", "model"):
        value = report.get(label)
        if not isinstance(value, str):
            errors.append(f"{label} is missing")
    declared = [
        report.get("name"),
        report.get("model"),
        *(ckpt.values() if isinstance(ckpt, dict) else []),
    ]
    if any(isinstance(v, str) and "://" in v for v in declared):
        errors.append("a declared label contains a URL; reports must not carry endpoints")
    return errors, warnings


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("report", type=Path)
    args = ap.parse_args(argv)
    report = load_report(args.report)
    errors, warnings = validate(report)
    for w in warnings:
        print(f"warning: {w}")
    if errors:
        for e in errors[:MAX_MESSAGES]:
            print(f"error: {e}")
        print("INVALID: do not submit this report")
        return 1
    print(
        f"VALID: mean balanced accuracy {report['mean_balanced_accuracy']:.1%} over {len(bench.TASKS)} tasks "
        "(declared metadata, recomputed scores)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
