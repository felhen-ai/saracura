"""Acurácia e acurácia balanceada do Kev por tarefa, a partir de predictions.jsonl e do JSONL de teste."""

import json
from collections import defaultdict
from pathlib import Path

root = Path.home() / "saracura-research"
out = {}
for d in sorted((root / "runs").glob("kev4b-eval*/test_*")):
    task = d.name[5:]
    pred_file = d / "predictions.jsonl"
    if not pred_file.exists():
        continue
    rows = [json.loads(line) for line in (root / "data" / "kev" / f"test_{task}.jsonl").open()]
    gold = {}
    for i, r in enumerate(rows):
        for qid, q in r["questions"].items():
            gold[(i, qid)] = q["label"]
    by = defaultdict(lambda: [0, 0])
    n = 0
    for line in pred_file.open():
        p = json.loads(line)
        i = int(p["id"].split("/")[1])
        for qid, probs in p["prediction"]["probabilities"].items():
            g = gold[(i, qid)]
            if isinstance(g, bool):
                pred = (
                    probs.get("true", probs.get("yes", 0)) >= 0.5
                    if isinstance(probs, dict)
                    else probs >= 0.5
                )
            else:
                pred = max(probs, key=probs.get)
            by[str(g)][0] += pred == g
            by[str(g)][1] += 1
            n += 1
    acc = sum(c for c, _ in by.values()) / n
    bal = sum(c / t for c, t in by.values()) / len(by)
    key = f"{d.parent.name}/{task}"
    out[key] = {
        "n": n,
        "de": len(gold),
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal, 4),
    }
    print(key, out[key])
(root / "results" / "kev4b_scores.json").write_text(json.dumps(out, indent=1))
