"""Tarefa FAQ Bacen (choice, 4 opções) — teste de transferência para um domínio não visto no treino.

Reconstrói o mesmo plano do harness do repositório (benchmarks/ptbr_native/protocol.py):
pergunta de FAQ -> escolher o início da resposta correta entre 4 trechos. A reconstrução é
conferida pelos invariantes congelados do harness (baseline lexical 265/373, posições 94/93/93/93).

Uso: python faq_bacen.py --name faq_base --model convaiinnovations/laya --subfolder multilingual
"""

import argparse
import hashlib
import json
import os
import re
import struct
import time
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "faq-bacen"
SEED = "saracura-phase5d-faq-bacen-v1"
INSTRUCTION = "Escolha a alternativa que começa a responder corretamente à pergunta."
MAX_LEN = int(os.environ.get("BENCH_MAX_LEN", 1024))
HEAD_MAX_LEN = int(os.environ.get("BENCH_HEAD_MAX_LEN", 256))
_TOKEN = re.compile(r"[^\W_]+", flags=re.UNICODE)


def normalize_text(value):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", value)).strip()


def framed_hash(*segments):
    digest = hashlib.sha256()
    for segment in segments:
        encoded = segment.encode("utf-8")
        digest.update(struct.pack(">Q", len(encoded)))
        digest.update(encoded)
    return digest.digest()


def lexical_tokens(value):
    return {
        t for t in _TOKEN.findall(unicodedata.normalize("NFC", value).casefold()) if len(t) >= 4
    }


def build_plan():
    import pyarrow.parquet as pq

    def rows(name):
        return pq.read_table(DATA / name / "test-00000-of-00001.parquet").to_pylist()

    corpus = {r["_id"]: normalize_text(r["text"])[:120] for r in rows("corpus")}
    queries = {r["_id"]: normalize_text(r["text"]) for r in rows("queries")}
    gold = {r["query-id"]: r["corpus-id"] for r in rows("qrels")}
    corpus_ids = sorted(corpus, key=lambda v: v.encode("utf-8"))
    plan = []
    for index, qid in enumerate(sorted(queries, key=lambda v: v.encode("utf-8"))):
        gold_id = gold[qid]
        used = {corpus[gold_id]}
        distractors = []
        for cand in sorted(
            (c for c in corpus_ids if c != gold_id), key=lambda c: framed_hash(SEED, qid, c)
        ):
            if corpus[cand] and corpus[cand] not in used:
                distractors.append(cand)
                used.add(corpus[cand])
            if len(distractors) == 3:
                break
        position = index % 4
        ordered = distractors.copy()
        ordered.insert(position, gold_id)
        plan.append(
            {"query": queries[qid], "choices": [corpus[c] for c in ordered], "gold": position}
        )

    lexical = 0
    for row in plan:
        q = lexical_tokens(row["query"])
        scores = [len(q & lexical_tokens(c)) for c in row["choices"]]
        lexical += max(range(4), key=scores.__getitem__) == row["gold"]
    counts = tuple(sum(r["gold"] == p for r in plan) for p in range(4))
    assert (len(plan), lexical, counts) == (373, 265, (94, 93, 93, 93)), (
        len(plan),
        lexical,
        counts,
    )
    return plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--subfolder")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import laya

    plan = build_plan()
    agent = laya.load(args.model, subfolder=args.subfolder, device=args.device)
    correct, truncated = 0, 0
    by_pos = [0, 0, 0, 0]
    t0 = time.time()
    for row in plan:
        question = {
            "resposta": {
                "type": "choice",
                "instructions": INSTRUCTION,
                "criteria": dict(zip("abcd", row["choices"], strict=False)),
            }
        }
        res = agent.predict(row["query"], question, max_len=MAX_LEN, head_max_len=HEAD_MAX_LEN)
        choice = res["answers"]["resposta"]["choice"]
        by_pos["abcd".index(choice)] += 1
        correct += choice == "abcd"[row["gold"]]
        truncated += bool(res["usage"]["truncated"] or res["usage"]["truncated_questions"])
    out = {
        "task": "faq-bacen/choice/4",
        "model": args.model,
        "subfolder": args.subfolder,
        "rows": len(plan),
        "accuracy": round(correct / len(plan), 4),
        "correct": correct,
        "chance": 0.25,
        "lexical_baseline": round(265 / 373, 4),
        "pred_by_position": by_pos,
        "truncated_items": truncated,
        "ms_per_item": round(1000 * (time.time() - t0) / len(plan), 1),
    }
    (ROOT / "results" / f"{args.name}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1)
    )
    print(
        args.name,
        {k: out[k] for k in ("accuracy", "correct", "pred_by_position", "truncated_items")},
    )


if __name__ == "__main__":
    main()
