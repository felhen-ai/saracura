"""One-command evaluation of a typed-decision model on the PT-BR benchmark.

    python ptbr_bench.py --base-url http://127.0.0.1:8000 --model my-model --name my-model \
        --checkpoint-id org/model --checkpoint-revision <40-hex commit>

Standard library only. Resolves the dataset revision (felhen-ai/ptbr-typed-decisions-bench) to an immutable commit,
downloads the test splits at that commit, sends one request per item to a System One-compatible server
(POST /v1/systemone), and reports balanced accuracy per task and the mean, with a 95% bootstrap interval.
The protocol matches eval_http.py: tasks in alphabetical order, option order shuffled per item with seed 20261009,
text cut at 3,500 characters. Writes <name>.json (or --output PATH) with per-item answers, never overwriting a file.
Model metadata is declared by the caller and recorded as given; whatever is not declared is recorded as "unknown".
Check a finished report with validate_submission.py before submitting it.
"""

import argparse
import hashlib
import http.client
import json
import math
import os
import random
import re
import ssl
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

REPO = "felhen-ai/ptbr-typed-decisions-bench"
TASKS = sorted(
    [
        "camara_tema",
        "factck_veracidade",
        "faquad_resposta",
        "juristcu_area",
        "olid_alvo",
        "olid_ofensivo",
        "scielo_area",
    ]
)
SEED = 20261002 + 7
MAX_CHARS = 3500
SCHEMA_VERSION = 2
PROTOCOL_ID = "ptbr-typed-decisions-v1"
UNKNOWN = "unknown"
RETRY_DELAY = 2.0
MAX_DATA_BYTES = 64 * 1024 * 1024
MAX_RESPONSE_BYTES = 1024 * 1024
# Closed set of per-item failure codes: reports carry these, never raw server text.
ERROR_CODES = (
    "transport",
    "http_client_error",
    "http_server_error",
    "bad_response",
    "bad_answer",
    "no_answer",
    "invalid_noul",
    "invalid_choice",
    "no_prediction",
)
SHA1_RE = re.compile(r"[0-9a-f]{40}")
DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}")
SHA256_RE = re.compile(r"[0-9a-f]{64}")
NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}")
REVISION_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,99}")
CHECKPOINT_FIELDS = (
    "id",
    "revision",
    "subfolder",
    "context",
    "quantization",
    "hardware",
    "server",
    "server_version",
)
ARG_FOR_FIELD = {"id": "checkpoint_id", "revision": "checkpoint_revision"}
CERT_HELP = (
    "TLS certificate verification failed. Do not disable verification. Use a Python that ships its own CA bundle, "
    "for example `uv run --python 3.12 research/ptbr_bench.py ...`, or point SSL_CERT_FILE at your CA bundle."
)


def protocol():
    return {
        "id": PROTOCOL_ID,
        "seed": SEED,
        "max_chars": MAX_CHARS,
        "tasks": list(TASKS),
        "option_order": "shuffled per item, one random.Random(seed) stream across tasks in alphabetical order",
        "scoring": "balanced accuracy per task, unweighted mean over tasks, 95% bootstrap CI (1000 resamples, seed 7)",
    }


def is_immutable_revision(value):
    return isinstance(value, str) and bool(SHA1_RE.fullmatch(value) or DIGEST_RE.fullmatch(value))


def _get(url, limit, what, timeout=120):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            data = r.read(limit + 1)
    except urllib.error.URLError as e:
        if isinstance(e.reason, ssl.SSLCertVerificationError):
            raise SystemExit(CERT_HELP) from None
        raise SystemExit(f"could not download {what}: {e.reason}") from None
    except (OSError, http.client.HTTPException) as e:
        raise SystemExit(f"could not download {what}: {type(e).__name__}") from None
    if len(data) > limit:
        raise SystemExit(f"{what} is larger than {limit} bytes")
    return data


def resolve_revision(revision):
    """Turn a branch, tag or commit into the immutable 40-hex commit all downloads are pinned to."""
    if SHA1_RE.fullmatch(revision):
        return revision
    url = f"https://huggingface.co/api/datasets/{REPO}/revision/{urllib.parse.quote(revision, safe='')}"
    data = _get(url, MAX_RESPONSE_BYTES, "dataset revision")
    try:
        sha = json.loads(data)["sha"]
    except (ValueError, KeyError, TypeError):
        sha = None
    if not isinstance(sha, str) or not SHA1_RE.fullmatch(sha):
        raise SystemExit("the Hub did not return a commit for the requested dataset revision")
    return sha


def download(revision, task):
    """Source bytes of one test split at an immutable commit."""
    url = f"https://huggingface.co/datasets/{REPO}/resolve/{revision}/data/{task}/test.jsonl"
    return _get(url, MAX_DATA_BYTES, f"{task} test split")


def parse_rows(data, task):
    """Rows of one split; ValueError on anything that is not the published row format."""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        raise ValueError(f"{task}: not valid UTF-8") from None
    rows = []
    for number, line in enumerate(text.split("\n"), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            raise ValueError(f"{task}: line {number} is not JSON") from None
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("text"), str)
            or not isinstance(row.get("instructions"), str)
        ):
            raise ValueError(f"{task}: line {number} lacks text or instructions")
        if row.get("type") == "noul":
            if not isinstance(row.get("gold"), bool):
                raise ValueError(f"{task}: line {number} has a non-boolean gold")
        elif row.get("type") == "choice":
            criteria = row.get("criteria")
            if (
                not isinstance(criteria, dict)
                or not criteria
                or not all(isinstance(k, str) for k in criteria)
                or not isinstance(row.get("gold"), str)
                or row["gold"] not in criteria
            ):
                raise ValueError(f"{task}: line {number} has invalid criteria or gold")
        else:
            raise ValueError(f"{task}: line {number} has an unknown type")
        rows.append(row)
    if not rows:
        raise ValueError(f"{task}: no rows")
    return rows


def fetch(revision, task, cache):
    """(bytes, rows) for one split; the cache is keyed by the immutable commit and written atomically."""
    path = Path(cache) / revision / f"{task}.test.jsonl"
    if path.is_file():
        data = path.read_bytes()
    else:
        data = download(revision, task)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{task}.")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
    return data, parse_rows(data, task)


def question(row, rng):
    if row["type"] == "noul":
        return {"type": "noul", "instructions": row["instructions"]}
    keys = list(row["criteria"])
    rng.shuffle(keys)
    return {
        "type": "choice",
        "instructions": row["instructions"],
        "criteria": {k: row["criteria"][k] for k in keys},
    }


def call(base, model, state, q, timeout):
    """(answer object, None) or (None, error code). Never raises on server or network misbehavior."""
    body = json.dumps({"model": model, "state": state, "questions": {"q": q}}).encode()
    req = urllib.request.Request(
        f"{base}/v1/systemone", data=body, headers={"content-type": "application/json"}
    )
    code = "transport"
    for attempt in range(3):
        if attempt:
            time.sleep(RETRY_DELAY)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw = r.read(MAX_RESPONSE_BYTES + 1)
        except urllib.error.HTTPError as e:
            if e.code < 500:
                return None, "http_client_error"
            code = "http_server_error"
            continue
        except (OSError, http.client.HTTPException):
            code = "transport"
            continue
        if len(raw) > MAX_RESPONSE_BYTES:
            return None, "bad_response"
        try:
            payload = json.loads(raw)
        except ValueError:
            return None, "bad_response"
        if not isinstance(payload, dict) or not isinstance(payload.get("answers"), dict):
            return None, "bad_answer"
        ans = payload["answers"].get("q")
        if not isinstance(ans, dict):
            return None, "bad_answer"
        return ans, None
    return None, code


def _number(v):
    return isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v)


def pick(ans, q):
    """(prediction, None) or (None, error code) for a validated answer object and the question it answers."""
    if not ans:
        return None, "no_answer"
    if q["type"] == "noul":
        v = ans.get("noul", ans.get("probability"))
        if isinstance(v, bool):
            return v, None
        if not _number(v):
            return None, "invalid_noul"
        return v >= 0.5, None
    if "choice" in ans:
        choice = ans["choice"]
        if not isinstance(choice, str) or choice not in q["criteria"]:
            return None, "invalid_choice"
        return choice, None
    probs = ans.get("probabilities")
    if not isinstance(probs, dict) or not probs:
        return None, "no_prediction"
    if not all(isinstance(k, str) and _number(v) for k, v in probs.items()):
        return None, "invalid_choice"
    best = max(probs, key=probs.get)
    if best not in q["criteria"]:
        return None, "invalid_choice"
    return best, None


def balanced(pairs):
    by = defaultdict(lambda: [0, 0])
    for g, p in pairs:
        by[g][0] += g == p
        by[g][1] += 1
    return sum(c / n for c, n in by.values()) / len(by) if by else 0.0


def interval(items, b=1000, seed=7):
    rng = random.Random(seed)
    means = []
    for _ in range(b):
        vals = [balanced([v[rng.randrange(len(v))] for _ in v]) for v in items.values()]
        means.append(sum(vals) / len(vals))
    means.sort()
    return round(means[int(0.025 * b)], 4), round(means[int(0.975 * b) - 1], 4)


def self_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def readiness(report):
    """Reasons a report is not submission-ready (empty list when it is). Used by the evaluator and the validator."""
    issues = []
    run = report.get("run") if isinstance(report.get("run"), dict) else {}
    if run.get("mode") != "full" or run.get("limit") != 0 or run.get("partial") is not False:
        issues.append("partial or smoke run (--limit)")
    tasks = report.get("tasks") if isinstance(report.get("tasks"), dict) else {}
    if sorted(tasks) != TASKS:
        issues.append("not all seven tasks present")
    for task, meta in tasks.items():
        if not isinstance(meta, dict) or not meta.get("n"):
            issues.append(f"{task}: no items")
        elif meta.get("n") != meta.get("source_n"):
            issues.append(f"{task}: incomplete task size")
        elif meta.get("failed"):
            issues.append(f"{task}: {meta['failed']} failed items")
    ckpt = report.get("checkpoint") if isinstance(report.get("checkpoint"), dict) else {}
    if ckpt.get("id", UNKNOWN) == UNKNOWN:
        issues.append("checkpoint id not declared")
    if not is_immutable_revision(ckpt.get("revision")):
        issues.append("checkpoint revision is not a 40-hex commit or sha256:<64 hex> digest")
    if not SHA1_RE.fullmatch(str(report.get("revision", ""))):
        issues.append("dataset revision is not an immutable commit")
    return issues


def publish_exclusive(path, payload):
    """Write payload so the final path appears complete or not at all, and never replaces an existing file."""
    path = Path(path)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        try:
            os.link(tmp, path)
        except FileExistsError:
            raise SystemExit(f"refusing to overwrite existing file {path}") from None
        except OSError as e:
            raise SystemExit(f"could not create {path} without overwriting: {e.strerror}") from None
    finally:
        Path(tmp).unlink(missing_ok=True)


def _positive_float(text):
    try:
        v = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a number") from None
    if not math.isfinite(v) or v <= 0:
        raise argparse.ArgumentTypeError("must be a finite number above 0")
    return v


def _non_negative_int(text):
    try:
        v = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError("must be an integer") from None
    if v < 0:
        raise argparse.ArgumentTypeError("must be 0 or more (0 runs every item)")
    return v


def looks_like_uri(text):
    """True for anything that could carry an endpoint: any `://`, http(s) scheme forms (also without slashes,
    any case) and network-path references starting with `//`. Unparseable text counts as a URI."""
    v = text.strip()
    if "://" in v or v.startswith("//"):
        return True
    try:
        return urllib.parse.urlsplit(v).scheme.lower() in ("http", "https")
    except ValueError:
        return True


def _label(text):
    v = text.strip()
    if not v or len(v) > 200 or not v.isprintable() or looks_like_uri(v):
        raise argparse.ArgumentTypeError(
            "must be 1 to 200 printable characters and must not contain a URL"
        )
    return v


def _model_alias(text):
    v = text.strip()
    if not v:
        return ""
    return _label(v)


def _name(text):
    if not NAME_RE.fullmatch(text):
        raise argparse.ArgumentTypeError(
            "must be a plain label (letters, digits, '.', '_', '-'), not a path; use --output PATH to choose where the report goes"
        )
    return text


def _revision(text):
    if not REVISION_RE.fullmatch(text) or ".." in text:
        raise argparse.ArgumentTypeError("must be a branch, tag or commit name")
    return text


def _base_url(text):
    parts = urllib.parse.urlsplit(text)
    if (
        parts.scheme not in ("http", "https")
        or not parts.hostname
        or parts.username
        or parts.password
    ):
        raise argparse.ArgumentTypeError("must be an http(s) URL without credentials")
    return text.rstrip("/")


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--base-url", required=True, type=_base_url, help="server root; never written to the report"
    )
    ap.add_argument(
        "--model",
        default="",
        type=_model_alias,
        help="server model alias or public identifier sent in each request, never an endpoint URL",
    )
    ap.add_argument("--name", required=True, type=_name, help="label for the results file")
    ap.add_argument(
        "--output",
        type=Path,
        help="report path (default ./<name>.json); an existing file is never overwritten",
    )
    ap.add_argument(
        "--revision", default="main", type=_revision, help="dataset revision, resolved to a commit"
    )
    ap.add_argument(
        "--limit",
        type=_non_negative_int,
        default=0,
        help="items per task: smoke run, never submission-ready",
    )
    ap.add_argument("--timeout", type=_positive_float, default=120)
    ap.add_argument("--cache", default=str(Path.home() / ".cache" / "ptbr-typed-decisions-bench"))
    declared = ap.add_argument_group(
        "declared model metadata (recorded as given, otherwise 'unknown')"
    )
    declared.add_argument("--checkpoint-id", type=_label, help="public model id, e.g. org/model")
    declared.add_argument(
        "--checkpoint-revision",
        type=_label,
        help="40-hex Hub commit or sha256:<64 hex> weights digest",
    )
    declared.add_argument(
        "--subfolder", type=_label, help="checkpoint subfolder, e.g. multilingual"
    )
    declared.add_argument(
        "--context", type=_label, help="context settings, e.g. 'max_len 1024, head 256'"
    )
    declared.add_argument("--quantization", type=_label)
    declared.add_argument("--hardware", type=_label)
    declared.add_argument("--server", type=_label, help="serving software")
    declared.add_argument("--server-version", type=_label)
    return ap


def _timing_now():
    return datetime.now(UTC).isoformat(timespec="seconds")


def run(args):
    out = Path(args.output) if args.output else Path(f"{args.name}.json")
    if os.path.lexists(out):
        raise SystemExit(f"refusing to overwrite existing file {out}")
    parent = out.parent if str(out.parent) else Path(".")
    if not parent.is_dir() or not os.access(parent, os.W_OK):
        raise SystemExit(f"output directory {parent} does not exist or is not writable")

    revision = resolve_revision(args.revision)
    cache = Path(args.cache)
    loaded = {}
    for task in TASKS:
        try:
            data, rows = fetch(revision, task, cache)
        except ValueError as e:
            raise SystemExit(f"invalid dataset: {e}") from None
        loaded[task] = (data, rows)

    started, t0 = _timing_now(), time.monotonic()
    rng = random.Random(SEED)
    files = {}
    report = {
        "schema_version": SCHEMA_VERSION,
        "name": args.name,
        "model": args.model,
        "dataset": REPO,
        "revision": revision,
        "requested_revision": args.revision,
        "dataset_files": files,
        "protocol": protocol(),
        "evaluator": {"script": "ptbr_bench.py", "sha256": self_sha256()},
        "checkpoint": {
            field: getattr(args, ARG_FOR_FIELD.get(field, field), None) or UNKNOWN
            for field in CHECKPOINT_FIELDS
        },
        "tasks": {},
        "items": {},
        "errors": {},
    }
    seconds = {}
    for task in TASKS:
        data, rows = loaded[task]
        files[task] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        total = len(rows)
        if args.limit:
            rows = rows[: args.limit]
        t_task = time.monotonic()
        pairs, codes = [], []
        for row in rows:
            q = question(row, rng)
            ans, code = call(args.base_url, args.model, row["text"][:MAX_CHARS], q, args.timeout)
            pred = None
            if code is None:
                pred, code = pick(ans, q)
            pairs.append([row["gold"], pred])
            codes.append(code)
        failed = sum(c is not None for c in codes)
        ba = balanced(pairs)
        report["tasks"][task] = {
            "balanced_accuracy": round(ba, 4),
            "n": len(rows),
            "failed": failed,
            "source_n": total,
        }
        report["items"][task] = pairs
        report["errors"][task] = codes
        seconds[task] = round(time.monotonic() - t_task, 1)
        why = ", ".join(f"{k}={v}" for k, v in sorted(Counter(c for c in codes if c).items()))
        print(
            f"{task:20s} {ba:.1%}  n={len(rows)}  failed={failed}" + (f" ({why})" if why else ""),
            flush=True,
        )
    vals = [m["balanced_accuracy"] for m in report["tasks"].values()]
    report["mean_balanced_accuracy"] = round(sum(vals) / len(vals), 4)
    report["mean_ci95"] = interval(report["items"])
    report["run"] = {
        "mode": "smoke" if args.limit else "full",
        "limit": args.limit,
        "partial": bool(args.limit),
    }
    issues = readiness(report)
    report["run"]["submission_ready"] = not issues
    report["run"]["readiness_issues"] = issues
    report["timing"] = {
        "started_at": started,
        "finished_at": _timing_now(),
        "seconds": round(time.monotonic() - t0, 1),
        "per_task_seconds": seconds,
    }
    lo, hi = report["mean_ci95"]
    print(f"{'MEAN':20s} {report['mean_balanced_accuracy']:.1%}  (95% CI {lo:.1%} to {hi:.1%})")
    publish_exclusive(out, json.dumps(report, ensure_ascii=False).encode("utf-8"))
    print(f"saved {out}")
    if issues:
        print("NOT submission-ready: " + "; ".join(issues))
    else:
        print("Declared submission-ready. Confirm with: python validate_submission.py " + str(out))
    return 0


def main(argv=None):
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
