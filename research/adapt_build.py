"""Adaptation track: N training examples per benchmark task, in the kev.train request format.

    BENCH_DIR=data/bench_v1 python adapt_build.py --n 50 --repeat 4 --out adapt_n50.jsonl

The draw is fixed per task (seed 20261008), so the N of a larger size contain those of a smaller one.
"""

import argparse
import json
import random

from bench import load, tasks

DRAW_SEED = 20261008
MAX_CHARS = 3500


def request(row):
    q = {"type": row["type"], "instructions": row["instructions"], "label": row["gold"]}
    if row["type"] == "choice":
        q["criteria"] = row["criteria"]
    return {"state": row["text"][:MAX_CHARS], "questions": {"q": q}}


def draw(task, n):
    train = load(task, "train")
    order = list(range(len(train)))
    random.Random(DRAW_SEED).shuffle(order)
    return [train[i] for i in (order[:n] if n else order)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--repeat", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = [request(r) for t in tasks() for r in draw(t, args.n)]
    mix = rows * args.repeat
    random.Random(args.n).shuffle(mix)
    with open(args.out, "w") as f:
        for r in mix:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"n={args.n} examples={len(rows)} x{args.repeat} -> {len(mix)}")


if __name__ == "__main__":
    main()
