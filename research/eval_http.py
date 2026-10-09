"""Avalia qualquer servidor compatível com System One (/v1/systemone) no benchmark PT-BR v1 e no FAQ Bacen.

Mesmo protocolo do bench.py: ordem das opções embaralhada por item (semente SEED + 7), texto cortado em
BENCH_MAX_CHARS, acurácia balanceada por tarefa. FAQ Bacen: mesmo plano do faq_bacen.py.

Uso: BENCH_DIR=data/bench_v1 BENCH_TAG=_v1 BENCH_MAX_CHARS=3500 \
     .venv/bin/python eval_http.py --base-url http://127.0.0.1:8020 --model kev-latest --name kev4b
"""

import argparse
import json
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from bench import MAX_CHARS, SEED, TAG, load, question, score, tasks  # noqa: E402
from faq_bacen import INSTRUCTION, build_plan  # noqa: E402


def call(base, model, state, questions, timeout):
    body = json.dumps({"model": model, "state": state, "questions": questions}).encode()
    req = urllib.request.Request(
        f"{base}/v1/systemone", data=body, headers={"content-type": "application/json"}
    )
    for _attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code < 500:
                return {"error": e.code, "detail": e.read()[:300].decode("utf-8", "replace")}
            time.sleep(2)
        except (urllib.error.URLError, TimeoutError):
            time.sleep(2)
    return {"error": "unreachable"}


def pick(ans, qtype):
    if qtype == "noul":
        v = ans.get("noul", ans.get("probability"))
        return None if v is None else v >= 0.5
    if "choice" in ans:
        return ans["choice"]
    probs = ans.get("probabilities") or {}
    return max(probs, key=probs.get) if probs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", default="")
    ap.add_argument("--name", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=120)
    ap.add_argument("--skip-bacen", action="store_true")
    args = ap.parse_args()

    report = {
        "base_url": args.base_url,
        "model": args.model,
        "max_chars": MAX_CHARS,
        "split": args.split,
        "tasks": {},
    }
    rng = random.Random(SEED + 7)
    for task in tasks():
        rows = load(task, args.split)
        if args.limit:
            rows = rows[: args.limit]
        gold, pred, failed, lat = [], [], 0, []
        for row in rows:
            q = question(row, rng)
            t0 = time.time()
            res = call(args.base_url, args.model, row["text"][:MAX_CHARS], {"q": q}, args.timeout)
            lat.append(time.time() - t0)
            ans = (res.get("answers") or {}).get("q")
            p = pick(ans, row["type"]) if ans else None
            if p is None:
                failed += 1
            pred.append(p)
            gold.append(row["gold"])
        m = score(gold, pred)
        lat.sort()
        m.update({"failed": failed, "median_ms": round(1000 * lat[len(lat) // 2], 1)})
        report["tasks"][task] = m
        report.setdefault("items", {})[task] = [[g, p] for g, p in zip(gold, pred, strict=True)]
        print(args.name, task, m, flush=True)
    vals = [m["balanced_accuracy"] for m in report["tasks"].values()]
    report["mean_balanced_accuracy"] = round(sum(vals) / len(vals), 4)
    print(args.name, "MEDIA balanced_accuracy", report["mean_balanced_accuracy"], flush=True)

    if not args.skip_bacen:
        plan = build_plan()
        correct, failed = 0, 0
        for row in plan:
            qs = {
                "resposta": {
                    "type": "choice",
                    "instructions": INSTRUCTION,
                    "criteria": dict(zip("abcd", row["choices"], strict=True)),
                }
            }
            res = call(args.base_url, args.model, row["query"], qs, args.timeout)
            ans = (res.get("answers") or {}).get("resposta")
            c = pick(ans, "choice") if ans else None
            failed += c is None
            correct += c == "abcd"[row["gold"]]
        report["faq_bacen"] = {
            "accuracy": round(correct / len(plan), 4),
            "correct": correct,
            "rows": len(plan),
            "failed": failed,
        }
        print(args.name, "FAQ_BACEN", report["faq_bacen"], flush=True)

    out = ROOT / "results" / "ptbr_board" / f"{args.name}{TAG}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print("SALVO", out)


if __name__ == "__main__":
    main()
