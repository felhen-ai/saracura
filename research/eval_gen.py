"""Mede um checkpoint Laya no conjunto de perguntas inéditas (data/gen/heldout.jsonl): concordância com o professor.

Uso: python eval_gen.py --name v1_e4 --model runs/v1_e4
"""

import argparse
import json
import time
from collections import defaultdict

import bench
import laya

GEN = bench.ROOT / "data" / "gen"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--file", default="heldout.jsonl", help="arquivo em data/gen com as perguntas de teste"
    )
    ap.add_argument("--tag", default="", help="sufixo do arquivo de resultado")
    args = ap.parse_args()

    agent = laya.load(args.model, device=args.device)
    rows = [json.loads(line) for line in (GEN / args.file).open()]
    by_type, by_source = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
    t0 = time.time()
    for row in rows:
        res = agent.predict(
            row["state"], row["questions"], max_len=bench.MAX_LEN, head_max_len=bench.HEAD_MAX_LEN
        )
        for qid, q in row["questions"].items():
            probs = row["gold"][qid]["probabilities"]
            teacher = max(probs, key=probs.get)
            ans = res["answers"][qid]
            if q["type"] == "choice":
                pred = ans["choice"]
            elif q["type"] == "noul":
                pred = "true" if ans["noul"] >= 0.5 else "false"
            else:
                pred = max(ans["probabilities"], key=ans["probabilities"].get)
            hit = pred == teacher
            for bucket in (by_type[q["type"]], by_source[row["source"]]):
                bucket[0] += hit
                bucket[1] += 1
    report = {
        "model": args.model,
        "rows": len(rows),
        "by_type": {k: {"agreement": round(c / n, 4), "n": n} for k, (c, n) in by_type.items()},
        "by_source": {k: {"agreement": round(c / n, 4), "n": n} for k, (c, n) in by_source.items()},
        "seconds": round(time.time() - t0, 1),
    }
    total = sum(v[0] for v in by_type.values()), sum(v[1] for v in by_type.values())
    report["agreement"] = round(total[0] / total[1], 4)
    (bench.ROOT / "results" / f"gen{args.tag}_{args.name}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1)
    )
    print(
        args.name,
        "perguntas inéditas: concordância",
        report["agreement"],
        "| por tipo:",
        report["by_type"],
    )


if __name__ == "__main__":
    main()
