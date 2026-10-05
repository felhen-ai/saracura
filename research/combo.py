"""Checkpoint combinado: benchmark aberto + dataset traduzido (telepatia, Apache-2.0) + anúncios com rótulos abertos.

  python combo.py prep --model-dir models/laya/multilingual --out items/combo.pt [--ads 8000] [--no-ads]
  python combo.py prep-ads --model-dir models/laya/multilingual --out items/ads_open.pt
  python combo.py eval-ads --name ads_open --model runs/ads_open

Os rótulos dos anúncios vêm de data/relabel_open.jsonl (Qwen 3.8 27B, Apache-2.0); o split de teste é o mesmo
das medições anteriores. `eval-ads` reporta acerto contra os rótulos abertos e contra os anteriores.
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import bench
from segmento import MAX_CHARS, ROOT, SEED, SEGMENTOS, load_rows, metrics, question, split


def ads_rows():
    rows = load_rows()
    new = [json.loads(line)["segmento"] for line in (ROOT / "data" / "relabel_open.jsonl").open()]
    assert len(new) == len(rows), (len(new), len(rows))
    for r, label in zip(rows, new, strict=False):
        r["segmento_open"] = label
    return split(rows)


def _tools(model_dir):
    from laya_finetune_mps import build_training_item
    from transformers import AutoTokenizer

    model_dir = Path(model_dir)
    cfg = json.loads((model_dir / "rl_agent_config.json").read_text())
    cfg = {**cfg, "max_len": bench.MAX_LEN, "head_max_len": bench.HEAD_MAX_LEN}
    return AutoTokenizer.from_pretrained(model_dir / "tokenizer"), cfg, build_training_item


def ads_items(tokenizer, cfg, build, n, rng):
    train, _ = ads_rows()
    items = []
    for row in train[: n or len(train)]:
        if row["segmento_open"] not in SEGMENTOS:
            continue
        order = list(SEGMENTOS)
        rng.shuffle(order)
        item = build(
            tokenizer,
            cfg,
            row["text"][:MAX_CHARS],
            question(order),
            {"probabilities": {row["segmento_open"]: 1.0}},
        )
        if item:
            items.append(item)
    return items


def save(items, out, counts):
    import torch

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(items, out)
    print(f"itens={len(items)} por_fonte={dict(counts)} -> {out}")


def cmd_prep(args):
    from datasets import load_dataset

    tokenizer, cfg, build = _tools(args.model_dir)
    rng = random.Random(SEED)
    items, counts = [], Counter()

    for task in bench.tasks():
        for row in bench.load(task, "train"):
            if row["type"] == "noul":
                gold = {
                    "probabilities": {"true": float(row["gold"]), "false": float(not row["gold"])}
                }
            else:
                gold = {"probabilities": {row["gold"]: 1.0}}
            item = build(
                tokenizer, cfg, row["text"][: bench.MAX_CHARS], bench.question(row, rng), gold
            )
            if item:
                items.append(item)
                counts["bench"] += 1

    for row in load_dataset("telepatia-ai/typed-decisions-pt-es", "pt", split="train"):
        state, questions, gold = (
            json.loads(v) if isinstance(v, str) else v
            for v in (row["state"], row["questions"], row["gold"])
        )
        for qid, q in questions.items():
            if qid in gold:
                item = build(tokenizer, cfg, state, q, gold[qid])
                if item:
                    items.append(item)
                    counts["traduzido"] += 1

    if not args.no_ads:
        ads = ads_items(tokenizer, cfg, build, args.ads, rng)
        items += ads
        counts["anuncios"] += len(ads)

    rng.shuffle(items)
    save(items, args.out, counts)


def cmd_prep_ads(args):
    tokenizer, cfg, build = _tools(args.model_dir)
    items = ads_items(tokenizer, cfg, build, 0, random.Random(SEED))
    save(items, args.out, {"anuncios": len(items)})


def cmd_eval_ads(args):
    import laya

    _, test = ads_rows()
    agent = laya.load(args.model, device=args.device)
    results = agent.predict_batch(
        [r["text"][:MAX_CHARS] for r in test],
        {"segmento": question()},
        batch_size=8,
        max_len=bench.MAX_LEN,
        head_max_len=bench.HEAD_MAX_LEN,
    )
    pred = [r["answers"]["segmento"]["choice"] for r in results]
    out = {
        "model": args.model,
        "test": len(test),
        "vs_rotulos_abertos": metrics([r["segmento_open"] for r in test], pred),
        "vs_rotulos_anteriores": metrics([r["segmento"] for r in test], pred),
    }
    (ROOT / "results" / f"ads{bench.TAG}_{args.name}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1)
    )
    print(
        args.name,
        "abertos",
        out["vs_rotulos_abertos"]["accuracy"],
        out["vs_rotulos_abertos"]["macro_recall"],
        "| anteriores",
        out["vs_rotulos_anteriores"]["accuracy"],
    )


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("--model-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--ads", type=int, default=8000)
    p.add_argument("--no-ads", action="store_true")
    a = sub.add_parser("prep-ads")
    a.add_argument("--model-dir", required=True)
    a.add_argument("--out", required=True)
    e = sub.add_parser("eval-ads")
    e.add_argument("--name", required=True)
    e.add_argument("--model", required=True)
    e.add_argument("--device", default="cuda")
    args = ap.parse_args()
    {"prep": cmd_prep, "prep-ads": cmd_prep_ads, "eval-ads": cmd_eval_ads}[args.cmd](args)


if __name__ == "__main__":
    main()
