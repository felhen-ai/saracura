"""Adaptation curve of the classical baseline (TF-IDF + logistic regression) with the same N examples per task.

    BENCH_DIR=data/bench_v1 BENCH_TAG=_v1 python adapt_tfidf.py

Balanced accuracy on each task's test split; per-item answers saved for confidence intervals (ci.py).
"""

import json

from adapt_build import MAX_CHARS, draw
from bench import ROOT, TAG, load, score, tasks
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

SIZES = (50, 100, 400, 0)  # 0 = full train split


def main():
    out = {}
    for n in SIZES:
        rep = {"tasks": {}, "items": {}}
        for t in tasks():
            sub = draw(t, n)
            test = load(t, "test")
            ys = [str(r["gold"]) for r in sub]
            if len(set(ys)) < 2:
                pred = [ys[0]] * len(test)
            else:
                clf = make_pipeline(
                    TfidfVectorizer(
                        ngram_range=(1, 2),
                        min_df=1 if n else 2,
                        sublinear_tf=True,
                        max_features=200_000,
                    ),
                    LogisticRegression(max_iter=2000, C=10.0, class_weight="balanced"),
                )
                clf.fit([r["text"][:MAX_CHARS] for r in sub], ys)
                pred = list(clf.predict([r["text"][:MAX_CHARS] for r in test]))
            gold = [str(r["gold"]) for r in test]
            rep["tasks"][t] = score(gold, pred)
            rep["items"][t] = [[g, p] for g, p in zip(gold, pred, strict=True)]
        vals = [m["balanced_accuracy"] for m in rep["tasks"].values()]
        rep["mean_balanced_accuracy"] = round(sum(vals) / len(vals), 4)
        out[str(n or "full")] = rep
        print("tfidf n =", n or "full", rep["mean_balanced_accuracy"], flush=True)
    path = ROOT / "results" / "ptbr_board" / f"adapt_tfidf{TAG}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
