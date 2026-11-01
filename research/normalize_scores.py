import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

if sys.flags.optimize:
    raise SystemExit("Run without -O: integrity assertions are required.")

parser = argparse.ArgumentParser(
    description="Derive scoring V2 from frozen reports; never executes models."
)
parser.add_argument("--reports", type=Path, required=True)
parser.add_argument("--dataset-cache", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
ROOT = args.output
ROOT.mkdir(parents=True, exist_ok=True)
REPO = Path(__file__).resolve().parents[1]
ranking = json.loads((REPO / "research/rounds/2026-10-21/final-ranking.json").read_text())
BUNDLE = args.reports
CACHE = args.dataset_cache
public = {}
classes = {}
task_hashes = {}
for p in sorted(CACHE.glob("*.test.jsonl")):
    task = p.name.removesuffix(".test.jsonl")
    rows = [json.loads(line) for line in p.read_text().splitlines()]
    ks = {2 if r["type"] == "noul" else len(r["criteria"]) for r in rows}
    assert len(ks) == 1
    k = ks.pop()
    assert k == len({str(r["gold"]) for r in rows})
    public[task] = rows
    classes[task] = k
    task_hashes[task] = hashlib.sha256(p.read_bytes()).hexdigest()
assert len(classes) == 7
results = []
for model in ranking["models"]:
    path = BUNDLE / model["report"]
    payload = path.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == model["sha256"], model["model"]
    report = json.loads(payload)
    assert report["revision"] == ranking["dataset_revision"]
    assert (
        sum(t["n"] for t in report["tasks"].values()) == 5196
        and sum(t["failed"] for t in report["tasks"].values()) == 0
    )
    assert set(report["items"]) == set(classes) == set(report["tasks"])
    tasks = {}
    for task, pairs in report["items"].items():
        assert [p[0] for p in pairs] == [r["gold"] for r in public[task]]
        assert report["dataset_files"][task]["sha256"] == task_hashes[task]
        assert len(pairs) == len(public[task])
        assert all(pred is not None for _, pred in pairs)
        k = classes[task]
        counts = defaultdict(lambda: [0, 0])
        for gold, pred in pairs:
            counts[str(gold)][1] += 1
            counts[str(gold)][0] += gold == pred
        ba = sum(ok / n for ok, n in counts.values()) / k
        assert abs(ba - report["tasks"][task]["balanced_accuracy"]) <= 0.000051
        signed = (ba - 1 / k) / (1 - 1 / k)
        tasks[task] = {
            "classes": k,
            "chance": 1 / k,
            "balanced_accuracy": ba,
            "adjusted_signed": signed,
            "normalized_clipped": max(0.0, signed),
        }
    raw_exact = sum(t["balanced_accuracy"] for t in tasks.values()) / 7 * 100
    signed = sum(t["adjusted_signed"] for t in tasks.values()) / 7 * 100
    normalized = sum(t["normalized_clipped"] for t in tasks.values()) / 7 * 100
    results.append(
        {
            "model": model["model"],
            "tuned_on_benchmark_tasks": model["tuned_ptbr"],
            "published_original_score": model["score"],
            "exact_recomputed_ba": raw_exact,
            "adjusted_balanced_accuracy_points": signed,
            "normalized_0_100_points": normalized,
            "tasks": tasks,
            "report_sha256": model["sha256"],
        }
    )
out = {
    "scoring_version": "ptbr-chance-corrected-v2",
    "status": "official scoring V2; original reports and evaluation protocol unchanged",
    "method": "For each task t: chance=1/K_t; score_t=max(0,(balanced_accuracy_t-chance)/(1-chance)); normalized=100*mean(score_t) across the original seven tasks. Signed adjusted BA is also retained without clipping.",
    "precision": "Derived from original per-item predictions, not rounded task metrics; published original scores are retained unchanged. Rounding differences below 0.01 points can occur.",
    "uncertainty": "No normalized confidence intervals computed; original raw-score intervals must not be reused.",
    "no_model_execution": True,
    "no_paid_services": True,
    "dataset_revision": ranking["dataset_revision"],
    "task_classes": classes,
    "task_sha256": task_hashes,
    "sources": [
        "https://huggingface.co/docs/leaderboards/open_llm_leaderboard/normalization",
        "https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html",
    ],
    "results": results,
}
(ROOT / "scoring-v2.json").write_text(json.dumps(out, indent=2))
fields = [
    "model",
    "tuned_on_benchmark_tasks",
    "published_original_score",
    "normalized_0_100_points",
    "adjusted_balanced_accuracy_points",
]
with (ROOT / "scoring-v2.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows({k: r[k] for k in fields} for r in results)
print("K:", classes)
for r in sorted(
    results, key=lambda x: (x["tuned_on_benchmark_tasks"], -x["normalized_0_100_points"])
):
    print(
        r["model"],
        f"{r['published_original_score']:.2f} -> {r['normalized_0_100_points']:.2f}",
        "task-tuned" if r["tuned_on_benchmark_tasks"] else "",
    )
print("All 14 report hashes, dataset hashes and gold-label sequences verified. No model execution.")
