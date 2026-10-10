"""One-command evaluation of a typed-decision model on the PT-BR benchmark.

    python ptbr_bench.py --base-url http://127.0.0.1:8000 --model my-model --name my-model

Standard library only. Downloads the test splits from the Hugging Face Hub
(felhen-ai/ptbr-typed-decisions-bench, pinned by --revision), sends one request per item to a System One-compatible
server (POST /v1/systemone), and reports balanced accuracy per task and the mean, with a 95% bootstrap interval.
The protocol matches eval_http.py: tasks in alphabetical order, option order shuffled per item with seed 20261009,
text cut at 3,500 characters. Writes <name>.json with per-item answers, ready to submit to the results page.
"""

import argparse
import json
import random
import time
import urllib.error
import urllib.request
from collections import defaultdict
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


def fetch(revision, task, cache):
    path = cache / revision / f"{task}.test.jsonl"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/datasets/{REPO}/resolve/{revision}/data/{task}/test.jsonl"
        with urllib.request.urlopen(url, timeout=120) as r:
            path.write_bytes(r.read())
    return [json.loads(line) for line in path.open(encoding="utf-8")]


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
    body = json.dumps({"model": model, "state": state, "questions": {"q": q}}).encode()
    req = urllib.request.Request(
        f"{base}/v1/systemone", data=body, headers={"content-type": "application/json"}
    )
    for _ in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return (json.loads(r.read()).get("answers") or {}).get("q")
        except urllib.error.HTTPError as e:
            if e.code < 500:
                return None
            time.sleep(2)
        except (urllib.error.URLError, TimeoutError):
            time.sleep(2)
    return None


def pick(ans, qtype):
    if not ans:
        return None
    if qtype == "noul":
        v = ans.get("noul", ans.get("probability"))
        return None if v is None else v >= 0.5
    if "choice" in ans:
        return ans["choice"]
    probs = ans.get("probabilities") or {}
    return max(probs, key=probs.get) if probs else None


def balanced(pairs):
    by = defaultdict(lambda: [0, 0])
    for g, p in pairs:
        by[g][0] += g == p
        by[g][1] += 1
    return sum(c / n for c, n in by.values()) / len(by)


def interval(items, b=1000, seed=7):
    rng = random.Random(seed)
    means = []
    for _ in range(b):
        vals = [balanced([v[rng.randrange(len(v))] for _ in v]) for v in items.values()]
        means.append(sum(vals) / len(vals))
    means.sort()
    return round(means[int(0.025 * b)], 4), round(means[int(0.975 * b) - 1], 4)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", default="")
    ap.add_argument("--name", required=True, help="label for the results file")
    ap.add_argument("--revision", default="main", help="dataset revision (commit sha recommended)")
    ap.add_argument("--limit", type=int, default=0, help="items per task, for a quick smoke run")
    ap.add_argument("--timeout", type=float, default=120)
    ap.add_argument("--cache", default=str(Path.home() / ".cache" / "ptbr-typed-decisions-bench"))
    args = ap.parse_args()

    rng = random.Random(SEED)
    report = {
        "name": args.name,
        "model": args.model,
        "dataset": REPO,
        "revision": args.revision,
        "tasks": {},
        "items": {},
    }
    for task in TASKS:
        rows = fetch(args.revision, task, Path(args.cache))
        if args.limit:
            rows = rows[: args.limit]
        pairs, failed = [], 0
        for row in rows:
            q = question(row, rng)
            p = pick(
                call(args.base_url, args.model, row["text"][:MAX_CHARS], q, args.timeout),
                row["type"],
            )
            failed += p is None
            pairs.append([row["gold"], p])
        ba = balanced(pairs)
        report["tasks"][task] = {
            "balanced_accuracy": round(ba, 4),
            "n": len(rows),
            "failed": failed,
        }
        report["items"][task] = pairs
        print(f"{task:20s} {ba:.1%}  n={len(rows)}  failed={failed}", flush=True)
    vals = [m["balanced_accuracy"] for m in report["tasks"].values()]
    report["mean_balanced_accuracy"] = round(sum(vals) / len(vals), 4)
    report["mean_ci95"] = interval(report["items"])
    lo, hi = report["mean_ci95"]
    print(f"{'MEAN':20s} {report['mean_balanced_accuracy']:.1%}  (95% CI {lo:.1%} to {hi:.1%})")
    Path(f"{args.name}.json").write_text(json.dumps(report, ensure_ascii=False))
    print(f"saved {args.name}.json")


if __name__ == "__main__":
    main()
