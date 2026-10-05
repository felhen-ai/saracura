"""Treino e avaliação multitarefa sobre o benchmark v0 (data/bench).

  python bench.py prep --model-dir models/laya/multilingual --out items/bench.pt [--exclude tarefa,...]
  python bench.py eval --name base --model models/laya/multilingual [--device cuda]
  python bench.py baselines

Métricas por tarefa: acurácia e acurácia balanceada (média do recall por classe), porque várias
tarefas são desbalanceadas. A ordem das opções é embaralhada por item no treino e na avaliação.
"""

import argparse
import json
import os
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BENCH = Path(os.environ.get("BENCH_DIR", ROOT / "data" / "bench"))
TAG = os.environ.get("BENCH_TAG", "")  # sufixo dos arquivos de resultado, ex.: "_v1"
SEED = 20261002
MAX_CHARS = int(os.environ.get("BENCH_MAX_CHARS", 1500))
MAX_LEN = int(os.environ.get("BENCH_MAX_LEN", 1024))
HEAD_MAX_LEN = int(os.environ.get("BENCH_HEAD_MAX_LEN", 256))


def tasks():
    return sorted(json.loads((BENCH / "manifest.json").read_text()))


def load(task, split):
    return [json.loads(line) for line in (BENCH / f"{task}.{split}.jsonl").open()]


def question(row, rng=None):
    if row["type"] == "noul":
        return {"type": "noul", "instructions": row["instructions"]}
    keys = list(row["criteria"])
    if rng:
        rng.shuffle(keys)
    return {
        "type": "choice",
        "instructions": row["instructions"],
        "criteria": {k: row["criteria"][k] for k in keys},
    }


def score(gold, pred):
    acc = sum(g == p for g, p in zip(gold, pred, strict=False)) / len(gold)
    by = defaultdict(lambda: [0, 0])
    for g, p in zip(gold, pred, strict=False):
        by[g][0] += g == p
        by[g][1] += 1
    bal = sum(c / n for c, n in by.values()) / len(by)
    return {"accuracy": round(acc, 4), "balanced_accuracy": round(bal, 4), "n": len(gold)}


def cmd_prep(args):
    import torch
    from laya_finetune_mps import build_training_item
    from transformers import AutoTokenizer

    model_dir = Path(args.model_dir)
    cfg = json.loads((model_dir / "rl_agent_config.json").read_text())
    cfg = {**cfg, "max_len": MAX_LEN, "head_max_len": HEAD_MAX_LEN}
    tokenizer = AutoTokenizer.from_pretrained(model_dir / "tokenizer")
    exclude = set(filter(None, (args.exclude or "").split(",")))
    rng = random.Random(SEED)
    items, counts, skipped = [], Counter(), 0
    for task in tasks():
        if task in exclude:
            continue
        for row in load(task, args.split):
            if row["type"] == "noul":
                gold = {
                    "probabilities": {
                        "true": 1.0 if row["gold"] else 0.0,
                        "false": 0.0 if row["gold"] else 1.0,
                    }
                }
            else:
                gold = {"probabilities": {row["gold"]: 1.0}}
            item = build_training_item(
                tokenizer, cfg, row["text"][:MAX_CHARS], question(row, rng), gold
            )
            if item is None:
                skipped += 1
            else:
                items.append(item)
                counts[task] += 1
    rng.shuffle(items)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(items, out)
    print(f"itens={len(items)} pulados={skipped} por_tarefa={dict(counts)} -> {out}")


def cmd_eval(args):
    import laya

    agent = laya.load(args.model, subfolder=args.subfolder, device=args.device)
    rng = random.Random(SEED + 7)
    report = {"model": args.model, "subfolder": args.subfolder, "tasks": {}}
    for task in tasks():
        rows = load(task, args.split)
        gold, pred, truncated = [], [], 0
        t0 = time.time()
        for row in rows:
            res = agent.predict(
                row["text"][:MAX_CHARS],
                {"q": question(row, rng)},
                max_len=MAX_LEN,
                head_max_len=HEAD_MAX_LEN,
            )
            ans = res["answers"]["q"]
            pred.append(ans["noul"] >= 0.5 if row["type"] == "noul" else ans["choice"])
            gold.append(row["gold"])
            truncated += bool(res["usage"]["truncated"] or res["usage"]["truncated_questions"])
        m = score(gold, pred)
        m.update(
            {"truncated": truncated, "ms_per_item": round(1000 * (time.time() - t0) / len(rows), 1)}
        )
        report["tasks"][task] = m
        print(args.name, task, m, flush=True)
    vals = [m["balanced_accuracy"] for m in report["tasks"].values()]
    report["mean_balanced_accuracy"] = round(sum(vals) / len(vals), 4)
    print(args.name, "MEDIA balanced_accuracy", report["mean_balanced_accuracy"])
    report["split"] = args.split
    (ROOT / "results" / f"bench{TAG}_{args.name}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1)
    )


def cmd_baselines(_args):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    report = {"tasks": {}}
    for task in tasks():
        train, test = load(task, "train"), load(task, "test")
        gold = [r["gold"] for r in test]
        majority = Counter(r["gold"] for r in train).most_common(1)[0][0]
        clf = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200_000),
            LogisticRegression(max_iter=2000, C=10.0, class_weight="balanced"),
        )
        clf.fit([r["text"] for r in train], [str(r["gold"]) for r in train])
        pred = [
            p == "True" if test[0]["type"] == "noul" else p
            for p in clf.predict([r["text"] for r in test])
        ]
        report["tasks"][task] = {
            "majority": score(gold, [majority] * len(gold)),
            "tfidf_logreg": score(gold, pred),
            "train": len(train),
        }
        print(task, report["tasks"][task], flush=True)
    for k in ("majority", "tfidf_logreg"):
        vals = [t[k]["balanced_accuracy"] for t in report["tasks"].values()]
        report[f"mean_balanced_accuracy_{k}"] = round(sum(vals) / len(vals), 4)
    print({k: v for k, v in report.items() if k.startswith("mean")})
    (ROOT / "results" / f"bench{TAG}_baselines.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1)
    )


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("--model-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--exclude")
    p.add_argument("--split", default="train")
    e = sub.add_parser("eval")
    e.add_argument("--name", required=True)
    e.add_argument("--model", required=True)
    e.add_argument("--subfolder")
    e.add_argument("--device", default="cuda")
    e.add_argument("--split", default="test")
    sub.add_parser("baselines")
    args = ap.parse_args()
    {"prep": cmd_prep, "eval": cmd_eval, "baselines": cmd_baselines}[args.cmd](args)


if __name__ == "__main__":
    main()
