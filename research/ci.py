"""95% confidence intervals (item bootstrap, 2,000 resamples) of balanced accuracy per task and of the mean.

Reads results that carry per-item answers ("items": task -> [[gold, pred], ...]), as written by eval_http.py.
Usage: python ci.py results/ptbr_board/adapt_n0_v1.json [...]; for the TF-IDF curve: python ci.py adapt_tfidf_v1.json:50
"""

import json
import random
import sys
from collections import defaultdict


def bal(pairs):
    by = defaultdict(lambda: [0, 0])
    for g, p in pairs:
        by[g][0] += g == p
        by[g][1] += 1
    return sum(c / n for c, n in by.values()) / len(by)


def ci(items, b=2000, seed=7):
    rng = random.Random(seed)
    point = {t: bal(v) for t, v in items.items()}
    boots = defaultdict(list)
    means = []
    for _ in range(b):
        vals = []
        for t, v in items.items():
            s = [v[rng.randrange(len(v))] for _ in range(len(v))]
            x = bal(s)
            boots[t].append(x)
            vals.append(x)
        means.append(sum(vals) / len(vals))

    def q(xs):
        xs = sorted(xs)
        return round(xs[int(0.025 * len(xs))], 4), round(xs[int(0.975 * len(xs)) - 1], 4)

    return {
        "mean": round(sum(point.values()) / len(point), 4),
        "mean_ci95": q(means),
        "tasks": {t: {"balanced_accuracy": round(point[t], 4), "ci95": q(boots[t])} for t in items},
    }


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        path, _, key = arg.partition(":")
        with open(path) as f:
            d = json.load(f)
        d = d[key] if key else d
        r = ci(d["items"])
        print(arg, r["mean"], r["mean_ci95"], {t: v["ci95"] for t, v in r["tasks"].items()})
