"""Mede um checkpoint Laya (repo do Hub ou diretório local) no teste fixo de `segmento`.

Uso: python eval_laya.py --name ft_n4000 --model runs/ft_n4000 [--device mps] [--shuffle-options]
"""

import argparse
import json
import random
import time

import laya
from segmento import MAX_CHARS, ROOT, SEED, SEGMENTOS, load_rows, metrics, question, split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--subfolder")
    ap.add_argument("--device", default="mps")
    ap.add_argument(
        "--shuffle-options", action="store_true", help="ordem das opções aleatória por item"
    )
    args = ap.parse_args()

    _, test = split(load_rows())
    agent = laya.load(args.model, subfolder=args.subfolder, device=args.device)
    rng = random.Random(SEED + 1)
    pred, truncated = [], 0
    t0 = time.time()
    if args.shuffle_options:
        for row in test:
            order = list(SEGMENTOS)
            rng.shuffle(order)
            res = agent.predict(row["text"][:MAX_CHARS], {"segmento": question(order)})
            pred.append(res["answers"]["segmento"]["choice"])
            truncated += bool(res["usage"]["truncated"] or res["usage"]["truncated_questions"])
    else:
        results = agent.predict_batch(
            [r["text"][:MAX_CHARS] for r in test], {"segmento": question()}, batch_size=8
        )
        for res in results:
            pred.append(res["answers"]["segmento"]["choice"])
            truncated += bool(res["usage"]["truncated"] or res["usage"]["truncated_questions"])
    dt = time.time() - t0

    m = metrics([r["segmento"] for r in test], pred)
    m.update(
        {
            "model": args.model,
            "subfolder": args.subfolder,
            "device": args.device,
            "test": len(test),
            "shuffle_options": args.shuffle_options,
            "truncated_items": truncated,
            "ms_per_item": round(1000 * dt / len(test), 1),
        }
    )
    out = ROOT / "results" / f"{args.name}.json"
    out.write_text(json.dumps(m, ensure_ascii=False, indent=1))
    print(
        args.name, {k: m[k] for k in ("accuracy", "macro_recall", "ms_per_item", "truncated_items")}
    )


if __name__ == "__main__":
    main()
