"""Avalia o Eikos (caiovicentino1/Eikos-4B) no benchmark v1 com o mesmo protocolo do bench.py.

Uso (na VM): BENCH_DIR=data/bench_v1 BENCH_TAG=_v1 .venv/bin/python eval_eikos.py --model models/eikos-4b --name eikos4b
Usa o código de inferência publicado pelo próprio Eikos (decision_core.py, letter_adapter.py no diretório do modelo).
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from bench import MAX_CHARS, SEED, TAG, load, question, score, tasks  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    mdir = Path(args.model).resolve()
    cfg = json.loads((mdir / "decision_config.json").read_text())
    os.environ["PROMPT_STYLE"] = cfg["prompt_version"].rsplit("-", 1)[-1]
    sys.path.insert(0, str(mdir))
    from decision_core import options_of
    from letter_adapter import LetterAdapter

    ad = LetterAdapter(str(mdir), device=args.device, calib=str(mdir / "calib.json"))
    ad.load()
    rng = random.Random(SEED + 7)  # mesma semente do bench.py eval: mesma ordem de opções por item
    report = {"model": str(mdir), "prompt_version": cfg["prompt_version"], "tasks": {}}
    for task in tasks():
        rows = load(task, args.split)
        if args.limit:
            rows = rows[: args.limit]
        gold, pred, maxtok = [], [], 0
        t0 = time.time()
        for row in rows:
            q = question(row, rng)
            opts = options_of(q)
            p, n_tok = ad.dist(row["text"][:MAX_CHARS], q, opts)
            maxtok = max(maxtok, n_tok)
            if row["type"] == "noul":
                pred.append(p.get("yes", 0.0) >= 0.5)
            else:
                pred.append(max(p, key=p.get))
            gold.append(row["gold"])
        m = score(gold, pred)
        m.update(
            {"max_tokens": maxtok, "ms_per_item": round(1000 * (time.time() - t0) / len(rows), 1)}
        )
        report["tasks"][task] = m
        print(args.name, task, m, flush=True)
    vals = [m["balanced_accuracy"] for m in report["tasks"].values()]
    report["mean_balanced_accuracy"] = round(sum(vals) / len(vals), 4)
    report["split"] = args.split
    print(args.name, "MEDIA balanced_accuracy", report["mean_balanced_accuracy"], flush=True)
    (ROOT / "results" / f"bench{TAG}_{args.name}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1)
    )


if __name__ == "__main__":
    main()
