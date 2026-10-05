"""Converte os dados do benchmark v1, do dataset traduzido e dos anúncios para o formato de treino do Kev.

Saída em data/kev/: train.jsonl (uma requisição por linha, com `label` em cada pergunta),
test_<tarefa>.jsonl por tarefa do benchmark e test_faq_bacen.jsonl (transferência).

Uso: BENCH_DIR=... python kev_data.py
"""

import json
import random

import bench
import faq_bacen
from combo import ads_rows
from datasets import load_dataset
from segmento import MAX_CHARS as ADS_MAX_CHARS
from segmento import SEED, SEGMENTOS
from segmento import question as ads_question

OUT = bench.ROOT / "data" / "kev"


def bench_request(row, rng=None):
    q = bench.question(row, rng)
    q["label"] = row["gold"]
    return {"state": row["text"][: bench.MAX_CHARS], "questions": {"q": q}}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    train = []
    for task in bench.tasks():
        train += [bench_request(r, rng) for r in bench.load(task, "train")]
        with (OUT / f"test_{task}.jsonl").open("w") as f:
            for r in bench.load(task, "test"):
                f.write(json.dumps(bench_request(r, rng), ensure_ascii=False) + "\n")

    for row in load_dataset("telepatia-ai/typed-decisions-pt-es", "pt", split="train"):
        state, questions, gold = (
            json.loads(v) if isinstance(v, str) else v
            for v in (row["state"], row["questions"], row["gold"])
        )
        qs = {}
        for qid, q in questions.items():
            g = gold.get(qid)
            if not g or "label" not in g:
                continue
            q = dict(q)
            if q["type"] == "score":
                levels = q.get("criteria") or []
                label = g["label"]
                q["label"] = levels.index(label) if label in levels else int(label)
            elif q["type"] == "noul":
                q["label"] = str(g["label"]).lower() == "true"
            else:
                q["label"] = g["label"]
            qs[qid] = q
        if qs:
            train.append({"state": state, "questions": qs})

    # documentos internos (tipo, área, estado) e perguntas geradas pelos professores (rótulo = opção mais provável)
    docs_dir = bench.ROOT / "data" / "docs"
    for f in sorted(docs_dir.glob("*.train.jsonl")):
        train += [bench_request(json.loads(line), rng) for line in f.open()]
    for name in ("train.jsonl", "train_or.jsonl", "train_uc_train.jsonl"):
        path = bench.ROOT / "data" / "gen" / name
        if not path.exists():
            continue
        for line in path.open():
            row = json.loads(line)
            qs = {}
            for qid, q in row["questions"].items():
                probs = row["gold"][qid]["probabilities"]
                top = max(probs, key=probs.get)
                q = dict(q)
                if q["type"] == "noul":
                    q["label"] = top == "true"
                elif q["type"] == "score":
                    q["label"] = int(top)
                else:
                    q["label"] = top
                qs[qid] = q
            train.append({"state": row["state"], "questions": qs})

    ads_train, ads_test = ads_rows()
    for r in ads_train[:8000]:
        if r["segmento_open"] in SEGMENTOS:
            order = list(SEGMENTOS)
            rng.shuffle(order)
            q = ads_question(order)
            q["label"] = r["segmento_open"]
            train.append({"state": r["text"][:ADS_MAX_CHARS], "questions": {"segmento": q}})
    with (OUT / "test_anuncios.jsonl").open("w") as f:
        for r in ads_test:
            if r["segmento_open"] in SEGMENTOS:
                q = ads_question()
                q["label"] = r["segmento_open"]
                f.write(
                    json.dumps(
                        {"state": r["text"][:ADS_MAX_CHARS], "questions": {"segmento": q}},
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    with (OUT / "test_faq_bacen.jsonl").open("w") as f:
        for row in faq_bacen.build_plan():
            q = {
                "type": "choice",
                "instructions": faq_bacen.INSTRUCTION,
                "criteria": dict(zip("abcd", row["choices"], strict=False)),
                "label": "abcd"[row["gold"]],
            }
            f.write(
                json.dumps(
                    {"state": row["query"], "questions": {"resposta": q}}, ensure_ascii=False
                )
                + "\n"
            )

    rng.shuffle(train)
    with (OUT / "train.jsonl").open("w") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(
        "treino:",
        len(train),
        "requisições;",
        sum(len(r["questions"]) for r in train),
        "decisões ->",
        OUT,
    )


if __name__ == "__main__":
    main()
